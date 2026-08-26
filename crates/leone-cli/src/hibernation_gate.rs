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
struct Gate {
    schema_version: u32,
    randomized_comparisons: usize,
    randomized_mismatches: usize,
    cancellation_child_empty: bool,
    depths: Vec<DepthResult>,
    exact_gate_passed: bool,
    speed_gate_passed: bool,
    gate_passed: bool,
}

#[derive(Debug, Serialize)]
struct DepthResult {
    name: &'static str,
    prefix_tokens: usize,
    context_tokens: usize,
    continuation_tokens: usize,
    host_bytes: u64,
    hibernate_nanoseconds: Vec<u64>,
    wake_nanoseconds: Vec<u64>,
    cold_nanoseconds: Vec<u64>,
    median_hibernate_nanoseconds: u64,
    median_wake_nanoseconds: u64,
    median_cold_nanoseconds: u64,
    median_wake_speedup: f64,
    wake_transcript_sha256: String,
    oracle_transcript_sha256: String,
    exact_wake_tokens: bool,
    source_session_released: bool,
    shared_prefix_reused_tokens: usize,
    shared_prefix_computed_tokens: usize,
    host_wake_replay: bool,
    speed_gate: Option<f64>,
    gate_passed: bool,
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, &arguments.model)?;
    let seed = runtime.model().tokenizer().encode(
        "Exact host hibernation releases device memory and restores an unchanged continuation.",
    )?;
    if seed.is_empty() {
        return Err(invalid_data("hibernation gate seed text produced no tokens").into());
    }
    let corpus = seed
        .iter()
        .copied()
        .cycle()
        .take(arguments.prefix_tokens.max(128))
        .collect::<Vec<_>>();
    let (randomized_comparisons, randomized_mismatches) =
        randomized_differential(&mut runtime, &corpus, arguments.cuts)?;
    let cancellation_child_empty = cancellation_differential(&mut runtime, &corpus[..96])?;
    let shallow = measure_depth(
        &mut runtime,
        "shallow-characterization",
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
        && shallow.exact_wake_tokens
        && shallow.source_session_released
        && shallow.host_wake_replay
        && release.exact_wake_tokens
        && release.source_session_released
        && release.host_wake_replay;
    let speed_gate_passed = release.gate_passed;
    let gate = Gate {
        schema_version: 1,
        randomized_comparisons,
        randomized_mismatches,
        cancellation_child_empty,
        depths: vec![shallow, release],
        exact_gate_passed,
        speed_gate_passed,
        gate_passed: exact_gate_passed && speed_gate_passed,
    };
    serde_json::to_writer_pretty(io::stdout().lock(), &gate)?;
    println!();
    if !gate.gate_passed {
        return Err(invalid_data("exact session hibernation gate failed").into());
    }
    Ok(())
}

fn randomized_differential(
    runtime: &mut Runtime<CudaBackend>,
    corpus: &[u32],
    cuts: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    let vocab = runtime.model().config().vocab_size;
    let mut draw = 0x4c65_6f6e_652d_7636_u64;
    let mut comparisons = 0;
    let mut mismatches = 0;
    for index in 0..cuts {
        draw = draw
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .rotate_left(17)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let cut = 65 + draw as usize % 32;
        let prefix = &corpus[..cut];
        let mut source = prepare_session(runtime, prefix)?;
        let snapshot = runtime.hibernate_session(&mut source)?;
        let source_released = source.is_empty();
        let mut woken = runtime.wake_session(snapshot)?;
        let branch = match index % 3 {
            0 => prefix.to_vec(),
            1 => {
                let mut tokens = prefix.to_vec();
                tokens.push(alternate_token(prefix[cut - 1], vocab));
                tokens
            }
            _ => {
                let mut tokens = prefix.to_vec();
                tokens[cut - 1] = alternate_token(tokens[cut - 1], vocab);
                tokens
            }
        };
        let result = runtime.generate_session_tokens(
            &mut woken,
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
        if !source_released
            || result.tokens != oracle.tokens
            || woken.last_replay().reuse_class != SessionReuseClass::HostWake
        {
            mismatches += 1;
        }
    }
    Ok((comparisons, mismatches))
}

fn cancellation_differential(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
) -> Result<bool, Box<dyn Error>> {
    let mut source = prepare_session(runtime, prefix)?;
    let snapshot = runtime.hibernate_session(&mut source)?;
    let mut woken = runtime.wake_session(snapshot)?;
    let cancelled =
        runtime.generate_session_tokens(&mut woken, prefix, options(8), |_| Ok(()), || true)?;
    Ok(source.is_empty() && cancelled.stats.cancelled && woken.is_empty())
}

#[allow(clippy::too_many_arguments)]
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
        let mut source = prepare_session(runtime, prefix)?;
        let snapshot = runtime.hibernate_session(&mut source)?;
        drop(runtime.wake_session(snapshot)?);
        drop(prepare_session(runtime, prefix)?);
    }
    let mut hibernate_nanoseconds = Vec::with_capacity(repetitions);
    let mut wake_nanoseconds = Vec::with_capacity(repetitions);
    let mut cold_nanoseconds = Vec::with_capacity(repetitions);
    let mut host_bytes = 0;
    let mut context_tokens = 0;
    let mut source_session_released = true;
    for repetition in 0..repetitions {
        let mut source = prepare_session(runtime, prefix)?;
        if repetition % 2 == 1 {
            let started = Instant::now();
            drop(prepare_session(runtime, prefix)?);
            cold_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
        }
        let started = Instant::now();
        let snapshot = runtime.hibernate_session(&mut source)?;
        hibernate_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
        source_session_released &= source.is_empty();
        host_bytes = snapshot.record().host_bytes;
        context_tokens = snapshot.record().context_tokens;
        let started = Instant::now();
        drop(runtime.wake_session(snapshot)?);
        wake_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
        if repetition % 2 == 0 {
            let started = Instant::now();
            drop(prepare_session(runtime, prefix)?);
            cold_nanoseconds.push(nanoseconds(started.elapsed().as_nanos())?);
        }
    }
    let median_hibernate_nanoseconds = median(&hibernate_nanoseconds);
    let median_wake_nanoseconds = median(&wake_nanoseconds);
    let median_cold_nanoseconds = median(&cold_nanoseconds);
    let median_wake_speedup = median_cold_nanoseconds as f64 / median_wake_nanoseconds as f64;

    let mut source = prepare_session(runtime, prefix)?;
    let snapshot = runtime.hibernate_session(&mut source)?;
    source_session_released &= source.is_empty();
    let mut woken = runtime.wake_session(snapshot)?;
    let result = runtime.generate_session_tokens(
        &mut woken,
        prefix,
        options(continuation_tokens),
        |_| Ok(()),
        || false,
    )?;
    let replay = woken.last_replay();
    let mut oracle_session = prepare_session(runtime, prefix)?;
    let oracle = runtime.generate_session_tokens(
        &mut oracle_session,
        prefix,
        options(continuation_tokens),
        |_| Ok(()),
        || false,
    )?;
    let exact_wake_tokens = result.tokens == oracle.tokens;
    let host_wake_replay = replay.reuse_class == SessionReuseClass::HostWake
        && replay.reused_tokens == prefix.len()
        && replay.replayed_tokens == 0;
    let speed_passed = speed_gate
        .map(|gate| median_wake_speedup >= gate)
        .unwrap_or(true);
    Ok(DepthResult {
        name,
        prefix_tokens: prefix.len(),
        context_tokens,
        continuation_tokens,
        host_bytes,
        hibernate_nanoseconds,
        wake_nanoseconds,
        cold_nanoseconds,
        median_hibernate_nanoseconds,
        median_wake_nanoseconds,
        median_cold_nanoseconds,
        median_wake_speedup,
        wake_transcript_sha256: token_stream_sha256(&result.tokens),
        oracle_transcript_sha256: token_stream_sha256(&oracle.tokens),
        exact_wake_tokens,
        source_session_released,
        shared_prefix_reused_tokens: replay.reused_tokens,
        shared_prefix_computed_tokens: prefix.len().saturating_sub(replay.reused_tokens),
        host_wake_replay,
        speed_gate,
        gate_passed: exact_wake_tokens
            && source_session_released
            && host_wake_replay
            && speed_passed,
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

fn parse(arguments: &[String]) -> Result<Arguments, io::Error> {
    let mut model = None;
    let mut cuts = 32;
    let mut warmups = 2;
    let mut repetitions = 5;
    let mut prefix_tokens = 3584;
    let mut continuation_tokens = 32;
    let mut speedup_gate = Some(5.0_f64);
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
                    "verify hibernate argument is invalid: {other}"
                )))
            }
        }
        index += 1;
    }
    Ok(Arguments {
        model: model.ok_or_else(|| invalid_data("verify hibernate requires -m <gguf>"))?,
        cuts,
        warmups,
        repetitions,
        prefix_tokens,
        continuation_tokens,
        speedup_gate,
    })
}

fn alternate_token(token: u32, vocab: usize) -> u32 {
    let token = usize::try_from(token).unwrap_or(0);
    u32::try_from((token + 1) % vocab).unwrap_or(0)
}

fn median(samples: &[u64]) -> u64 {
    let mut values = samples.to_vec();
    values.sort_unstable();
    values[values.len() / 2]
}

fn nanoseconds(value: u128) -> Result<u64, io::Error> {
    u64::try_from(value).map_err(|_| invalid_data("duration exceeds u64 nanoseconds"))
}

fn positive(value: &str, name: &str) -> Result<usize, io::Error> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data(format!("{name} must be a nonzero integer")))
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
