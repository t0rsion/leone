use leone::{
    token_stream_sha256, DecodeExecution, GenerateOptions, GenerationSession, KvCacheDtype,
    Runtime, SessionReuseClass,
};
use leone_cuda::CudaBackend;
use serde::Serialize;
use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_CUTS: usize = 32;
const DEFAULT_WARMUPS: usize = 2;
const DEFAULT_REPETITIONS: usize = 5;
const DEFAULT_PREFIX: usize = 3584;
const DEFAULT_CONTINUATION: usize = 32;
const DEFAULT_SPEEDUP_GATE: f64 = 10.0;

#[derive(Debug)]
struct Arguments {
    model: PathBuf,
    cuts: usize,
    warmups: usize,
    repetitions: usize,
    prefix_tokens: usize,
    continuation_tokens: usize,
    speedup_gate: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ForkGate {
    schema_version: u32,
    randomized_comparisons: usize,
    randomized_mismatches: usize,
    randomized_failures: Vec<RandomizedFailure>,
    cancellation_child_empty: bool,
    cancellation_parent_unchanged: bool,
    depths: Vec<DepthResult>,
    exact_gate_passed: bool,
    speed_gate_passed: bool,
    gate_passed: bool,
}

#[derive(Debug, Serialize)]
struct RandomizedFailure {
    index: usize,
    cut: usize,
    branch: &'static str,
    target: &'static str,
    token_match: bool,
    replay_class: &'static str,
    reused_tokens: usize,
    computed_tokens: usize,
}

#[derive(Debug, Serialize)]
struct DepthResult {
    name: &'static str,
    prefix_tokens: usize,
    context_tokens: usize,
    continuation_tokens: usize,
    copied_bytes: u64,
    fork_nanoseconds: Vec<u64>,
    cold_nanoseconds: Vec<u64>,
    median_fork_nanoseconds: u64,
    median_cold_nanoseconds: u64,
    median_speedup: f64,
    child_transcript_sha256: String,
    oracle_transcript_sha256: String,
    exact_child_tokens: bool,
    unchanged_parent_tokens: bool,
    shared_prefix_reused_tokens: usize,
    shared_prefix_computed_tokens: usize,
    device_fork_replay: bool,
    speed_gate: Option<f64>,
    gate_passed: bool,
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, &arguments.model)?;
    let seed_tokens = runtime.model().tokenizer().encode(
        "Exact session forks preserve a shared prefix while independent branches continue.",
    )?;
    if seed_tokens.is_empty() {
        return Err(invalid_data("fork gate seed text produced no tokens").into());
    }
    let maximum_prefix = arguments.prefix_tokens.max(128);
    let corpus = repeated_prefix(&seed_tokens, maximum_prefix);
    let (randomized_comparisons, randomized_failures) =
        randomized_differential(&mut runtime, &corpus, arguments.cuts)?;
    let randomized_mismatches = randomized_failures.len();
    let (cancellation_child_empty, cancellation_parent_unchanged) =
        cancellation_differential(&mut runtime, &corpus[..96])?;

    let shallow = measure_depth(
        &mut runtime,
        "shallow-losing-case",
        &corpus[..32],
        arguments.continuation_tokens,
        arguments.warmups,
        arguments.repetitions,
        None,
    )?;
    let release = measure_depth(
        &mut runtime,
        "release-gate",
        &corpus[..arguments.prefix_tokens],
        arguments.continuation_tokens,
        arguments.warmups,
        arguments.repetitions,
        arguments.speedup_gate,
    )?;
    let exact_gate_passed = randomized_mismatches == 0
        && cancellation_child_empty
        && cancellation_parent_unchanged
        && shallow.exact_child_tokens
        && shallow.unchanged_parent_tokens
        && shallow.device_fork_replay
        && release.exact_child_tokens
        && release.unchanged_parent_tokens
        && release.device_fork_replay;
    let speed_gate_passed = release.gate_passed;
    let gate = ForkGate {
        schema_version: 1,
        randomized_comparisons,
        randomized_mismatches,
        randomized_failures,
        cancellation_child_empty,
        cancellation_parent_unchanged,
        depths: vec![shallow, release],
        exact_gate_passed,
        speed_gate_passed,
        gate_passed: exact_gate_passed && speed_gate_passed,
    };
    serde_json::to_writer_pretty(io::stdout().lock(), &gate)?;
    println!();
    if !gate.gate_passed {
        return Err(invalid_data("exact live session fork gate failed").into());
    }
    Ok(())
}

fn parse(arguments: &[String]) -> Result<Arguments, io::Error> {
    let mut model = None;
    let mut cuts = DEFAULT_CUTS;
    let mut warmups = DEFAULT_WARMUPS;
    let mut repetitions = DEFAULT_REPETITIONS;
    let mut prefix_tokens = DEFAULT_PREFIX;
    let mut continuation_tokens = DEFAULT_CONTINUATION;
    let mut speedup_gate = Some(DEFAULT_SPEEDUP_GATE);
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-m" | "--model" => model = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--cuts" => cuts = positive(value(arguments, &mut index)?, "cuts")?,
            "--warmups" => warmups = positive(value(arguments, &mut index)?, "warmups")?,
            "--repetitions" => {
                repetitions = positive(value(arguments, &mut index)?, "repetitions")?
            }
            "--prefix-tokens" => {
                prefix_tokens = positive(value(arguments, &mut index)?, "prefix tokens")?
            }
            "--continuation-tokens" => {
                continuation_tokens =
                    positive(value(arguments, &mut index)?, "continuation tokens")?
            }
            "--speedup-gate" => {
                let raw = value(arguments, &mut index)?;
                let parsed: f64 = raw
                    .parse()
                    .map_err(|_| invalid_data(format!("speedup gate is invalid: {raw}")))?;
                if !parsed.is_finite() || parsed <= 0.0 {
                    return Err(invalid_data("speedup gate must be finite and positive"));
                }
                speedup_gate = Some(parsed);
            }
            "--exact-only" => speedup_gate = None,
            other => {
                return Err(invalid_data(format!(
                    "verify fork argument is invalid: {other}"
                )))
            }
        }
        index += 1;
    }
    Ok(Arguments {
        model: model.ok_or_else(|| invalid_data("verify fork requires -m <gguf>"))?,
        cuts,
        warmups,
        repetitions,
        prefix_tokens,
        continuation_tokens,
        speedup_gate,
    })
}

fn randomized_differential(
    runtime: &mut Runtime<CudaBackend>,
    corpus: &[u32],
    cuts: usize,
) -> Result<(usize, Vec<RandomizedFailure>), Box<dyn Error>> {
    let vocab = runtime.model().config().vocab_size;
    let mut draw = 0x4c65_6f6e_652d_7635_u64;
    let mut comparisons = 0;
    let mut failures = Vec::new();
    for index in 0..cuts {
        draw = draw
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .rotate_left(17)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let cut = 65 + draw as usize % 32;
        let prefix = &corpus[..cut];
        let mut parent = prepare_session(runtime, prefix)?;
        let mut child = runtime.fork_session(&parent)?;
        let (branch_name, branch) = match index % 3 {
            0 => ("exact-repeat", prefix.to_vec()),
            1 => {
                let mut tokens = prefix.to_vec();
                tokens.push(alternate_token(prefix[cut - 1], vocab));
                ("append-only", tokens)
            }
            _ => {
                let mut tokens = prefix.to_vec();
                tokens[cut - 1] = alternate_token(tokens[cut - 1], vocab);
                ("arbitrary-branch", tokens)
            }
        };
        let child_result = runtime.generate_session_tokens(
            &mut child,
            &branch,
            options(8),
            |_| Ok(()),
            || false,
        )?;
        let mut oracle_session = prepare_session(runtime, prefix)?;
        let oracle = runtime.generate_session_tokens(
            &mut oracle_session,
            &branch,
            options(8),
            |_| Ok(()),
            || false,
        )?;
        comparisons += 1;
        let replay = child.last_replay();
        let token_match = child_result.tokens == oracle.tokens;
        if !token_match || replay.reuse_class != SessionReuseClass::DeviceFork {
            failures.push(RandomizedFailure {
                index,
                cut,
                branch: branch_name,
                target: "child",
                token_match,
                replay_class: reuse_class_name(replay.reuse_class),
                reused_tokens: replay.reused_tokens,
                computed_tokens: replay.computed_tokens,
            });
        }
        let parent_result = runtime.generate_session_tokens(
            &mut parent,
            prefix,
            options(8),
            |_| Ok(()),
            || false,
        )?;
        let mut parent_oracle_session = prepare_session(runtime, prefix)?;
        let parent_oracle = runtime.generate_session_tokens(
            &mut parent_oracle_session,
            prefix,
            options(8),
            |_| Ok(()),
            || false,
        )?;
        comparisons += 1;
        if parent_result.tokens != parent_oracle.tokens {
            let replay = parent.last_replay();
            failures.push(RandomizedFailure {
                index,
                cut,
                branch: branch_name,
                target: "parent",
                token_match: false,
                replay_class: reuse_class_name(replay.reuse_class),
                reused_tokens: replay.reused_tokens,
                computed_tokens: replay.computed_tokens,
            });
        }
    }
    Ok((comparisons, failures))
}

fn cancellation_differential(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
) -> Result<(bool, bool), Box<dyn Error>> {
    let mut parent_oracle_session = prepare_session(runtime, prefix)?;
    let parent_oracle = runtime.generate_session_tokens(
        &mut parent_oracle_session,
        prefix,
        options(8),
        |_| Ok(()),
        || false,
    )?;
    let mut parent = prepare_session(runtime, prefix)?;
    let mut child = runtime.fork_session(&parent)?;
    let cancelled =
        runtime.generate_session_tokens(&mut child, prefix, options(8), |_| Ok(()), || true)?;
    let parent_result =
        runtime.generate_session_tokens(&mut parent, prefix, options(8), |_| Ok(()), || false)?;
    Ok((
        cancelled.stats.cancelled && child.is_empty(),
        parent_result.tokens == parent_oracle.tokens,
    ))
}

fn measure_depth(
    runtime: &mut Runtime<CudaBackend>,
    name: &'static str,
    prefix: &[u32],
    continuation_tokens: usize,
    warmups: usize,
    repetitions: usize,
    speed_gate: Option<f64>,
) -> Result<DepthResult, Box<dyn Error>> {
    for _ in 0..warmups {
        let parent = prepare_session(runtime, prefix)?;
        drop(runtime.fork_session(&parent)?);
        drop(prepare_session(runtime, prefix)?);
    }
    let mut fork_nanoseconds = Vec::with_capacity(repetitions);
    let mut cold_nanoseconds = Vec::with_capacity(repetitions);
    let mut copied_bytes = 0;
    let mut context_tokens = 0;
    for repetition in 0..repetitions {
        let parent = prepare_session(runtime, prefix)?;
        if repetition % 2 == 0 {
            let started = Instant::now();
            let child = runtime.fork_session(&parent)?;
            fork_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
            let record = child
                .last_fork()
                .ok_or_else(|| invalid_data("fork record is missing"))?;
            copied_bytes = record.copied_bytes;
            context_tokens = record.context_tokens;
            drop(child);
            let started = Instant::now();
            drop(prepare_session(runtime, prefix)?);
            cold_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
        } else {
            let started = Instant::now();
            drop(prepare_session(runtime, prefix)?);
            cold_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
            let started = Instant::now();
            let child = runtime.fork_session(&parent)?;
            fork_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
            let record = child
                .last_fork()
                .ok_or_else(|| invalid_data("fork record is missing"))?;
            copied_bytes = record.copied_bytes;
            context_tokens = record.context_tokens;
        }
    }
    let median_fork_nanoseconds = median(&fork_nanoseconds);
    let median_cold_nanoseconds = median(&cold_nanoseconds);
    let median_speedup = median_cold_nanoseconds as f64 / median_fork_nanoseconds as f64;

    let mut parent = prepare_session(runtime, prefix)?;
    let mut child = runtime.fork_session(&parent)?;
    let child_result = runtime.generate_session_tokens(
        &mut child,
        prefix,
        options(continuation_tokens),
        |_| Ok(()),
        || false,
    )?;
    let replay = child.last_replay();
    let mut oracle_session = prepare_session(runtime, prefix)?;
    let oracle = runtime.generate_session_tokens(
        &mut oracle_session,
        prefix,
        options(continuation_tokens),
        |_| Ok(()),
        || false,
    )?;
    let parent_result = runtime.generate_session_tokens(
        &mut parent,
        prefix,
        options(continuation_tokens),
        |_| Ok(()),
        || false,
    )?;
    let exact_child_tokens = child_result.tokens == oracle.tokens;
    let unchanged_parent_tokens = parent_result.tokens == oracle.tokens;
    let device_fork_replay = replay.reuse_class == SessionReuseClass::DeviceFork
        && replay.reused_tokens == prefix.len()
        && replay.replayed_tokens == 0;
    let passed_speed = speed_gate
        .map(|gate| median_speedup >= gate)
        .unwrap_or(true);
    Ok(DepthResult {
        name,
        prefix_tokens: prefix.len(),
        context_tokens,
        continuation_tokens,
        copied_bytes,
        fork_nanoseconds,
        cold_nanoseconds,
        median_fork_nanoseconds,
        median_cold_nanoseconds,
        median_speedup,
        child_transcript_sha256: token_stream_sha256(&child_result.tokens),
        oracle_transcript_sha256: token_stream_sha256(&oracle.tokens),
        exact_child_tokens,
        unchanged_parent_tokens,
        shared_prefix_reused_tokens: replay.reused_tokens,
        shared_prefix_computed_tokens: prefix.len().saturating_sub(replay.reused_tokens),
        device_fork_replay,
        speed_gate,
        gate_passed: exact_child_tokens
            && unchanged_parent_tokens
            && device_fork_replay
            && passed_speed,
    })
}

fn prepare_session(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
) -> Result<GenerationSession<CudaBackend>, Box<dyn Error>> {
    let mut session = GenerationSession::new();
    runtime.generate_session_tokens(&mut session, prefix, options(1), |_| Ok(()), || false)?;
    Ok(session)
}

fn options(tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::greedy(tokens);
    options.decode_execution = DecodeExecution::Eager;
    options.kv_cache_dtype = KvCacheDtype::F16;
    options
}

fn repeated_prefix(seed: &[u32], tokens: usize) -> Vec<u32> {
    seed.iter().copied().cycle().take(tokens).collect()
}

fn alternate_token(token: u32, vocab: usize) -> u32 {
    let token = usize::try_from(token).unwrap_or(0);
    u32::try_from((token + 1) % vocab).unwrap_or(0)
}

fn median(samples: &[u64]) -> u64 {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn reuse_class_name(class: SessionReuseClass) -> &'static str {
    match class {
        SessionReuseClass::Cold => "cold",
        SessionReuseClass::ExactRepeat => "exact-repeat",
        SessionReuseClass::AppendOnly => "append-only",
        SessionReuseClass::ArbitraryBranch => "arbitrary-branch",
        SessionReuseClass::RestoreReplay => "restore-replay",
        SessionReuseClass::DeviceFork => "device-fork",
        SessionReuseClass::HostWake => "host-wake",
    }
}

fn nanoseconds(value: u128) -> Result<u64, io::Error> {
    u64::try_from(value).map_err(|_| invalid_data("duration exceeds u64 nanoseconds"))
}

fn positive(value: &str, name: &str) -> Result<usize, io::Error> {
    let parsed = value
        .parse()
        .map_err(|_| invalid_data(format!("{name} is invalid: {value}")))?;
    if parsed == 0 {
        return Err(invalid_data(format!("{name} must be nonzero")));
    }
    Ok(parsed)
}

fn value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid_data("an option value is missing"))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
