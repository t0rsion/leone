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

struct DepthMeasurements {
    host_bytes: u64,
    context_tokens: usize,
    hibernate_nanoseconds: Vec<u64>,
    wake_nanoseconds: Vec<u64>,
    cold_nanoseconds: Vec<u64>,
    source_session_released: bool,
}

struct DepthCorrectness {
    result_tokens: Vec<u32>,
    oracle_tokens: Vec<u32>,
    replay_class: SessionReuseClass,
    reused_tokens: usize,
    replayed_tokens: usize,
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, &arguments.model)?;
    let corpus = seed_corpus(&mut runtime, arguments.prefix_tokens)?;
    let (randomized_comparisons, randomized_mismatches) =
        randomized_differential(&mut runtime, &corpus, arguments.cuts)?;
    let cancellation_child_empty = cancellation_differential(&mut runtime, &corpus[..96])?;
    let depths = measure_depths(&mut runtime, &corpus, &arguments)?;
    let gate = build_gate(
        randomized_comparisons,
        randomized_mismatches,
        cancellation_child_empty,
        depths,
    );
    serde_json::to_writer_pretty(io::stdout().lock(), &gate)?;
    println!();
    if !gate.gate_passed {
        return Err(invalid_data("exact session hibernation gate failed").into());
    }
    Ok(())
}

fn seed_corpus(
    runtime: &mut Runtime<CudaBackend>,
    prefix_tokens: usize,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let seed = runtime.model().tokenizer().encode(
        "Exact host hibernation releases device memory and restores an unchanged continuation.",
    )?;
    if seed.is_empty() {
        return Err(invalid_data("hibernation gate seed text produced no tokens").into());
    }
    Ok(seed
        .iter()
        .copied()
        .cycle()
        .take(prefix_tokens.max(128))
        .collect())
}

fn measure_depths(
    runtime: &mut Runtime<CudaBackend>,
    corpus: &[u32],
    arguments: &Arguments,
) -> Result<Vec<DepthResult>, Box<dyn Error>> {
    Ok(vec![
        measure_depth(
            runtime,
            "shallow-characterization",
            &corpus[..32],
            arguments.continuation_tokens,
            arguments.warmups,
            arguments.repetitions,
            None,
        )?,
        measure_depth(
            runtime,
            "release-gate",
            &corpus[..arguments.prefix_tokens],
            arguments.continuation_tokens,
            arguments.warmups,
            arguments.repetitions,
            arguments.speedup_gate,
        )?,
    ])
}

fn build_gate(
    randomized_comparisons: usize,
    randomized_mismatches: usize,
    cancellation_child_empty: bool,
    depths: Vec<DepthResult>,
) -> Gate {
    let exact_gate_passed = randomized_mismatches == 0
        && cancellation_child_empty
        && depths.iter().all(|depth| {
            depth.exact_wake_tokens && depth.source_session_released && depth.host_wake_replay
        });
    let speed_gate_passed = depths.last().is_some_and(|depth| depth.gate_passed);
    Gate {
        schema_version: 1,
        randomized_comparisons,
        randomized_mismatches,
        cancellation_child_empty,
        depths,
        exact_gate_passed,
        speed_gate_passed,
        gate_passed: exact_gate_passed && speed_gate_passed,
    }
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
        let (next_draw, mismatch) = hibernation_case(runtime, corpus, draw, vocab, index)?;
        draw = next_draw;
        comparisons += 1;
        if mismatch {
            mismatches += 1;
        }
    }
    Ok((comparisons, mismatches))
}

fn hibernation_case(
    runtime: &mut Runtime<CudaBackend>,
    corpus: &[u32],
    draw: u64,
    vocab: usize,
    index: usize,
) -> Result<(u64, bool), Box<dyn Error>> {
    let draw = draw
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        .rotate_left(17)
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let cut = 65 + draw as usize % 32;
    let prefix = &corpus[..cut];
    let branch = hibernation_branch(prefix, cut, vocab, index);
    let (source_released, result_tokens, host_wake) =
        hibernated_session_tokens(runtime, prefix, &branch)?;
    let oracle_tokens = reference_session_tokens(runtime, prefix, &branch)?;
    let mismatch = !source_released || result_tokens != oracle_tokens || !host_wake;
    Ok((draw, mismatch))
}

fn hibernated_session_tokens(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    branch: &[u32],
) -> Result<(bool, Vec<u32>, bool), Box<dyn Error>> {
    let mut source = prepare_session(runtime, prefix)?;
    let snapshot = runtime.hibernate_session(&mut source)?;
    let source_released = source.is_empty();
    let mut woken = runtime.wake_session(snapshot)?;
    let result =
        runtime.generate_session_tokens(&mut woken, branch, options(8), |_| Ok(()), || false)?;
    let host_wake = woken.last_replay().reuse_class == SessionReuseClass::HostWake;
    Ok((source_released, result.tokens, host_wake))
}

fn reference_session_tokens(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    branch: &[u32],
) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut oracle_session = prepare_session(runtime, prefix)?;
    let oracle = runtime.generate_session_tokens(
        &mut oracle_session,
        branch,
        options(8),
        |_| Ok(()),
        || false,
    )?;
    Ok(oracle.tokens)
}

fn hibernation_branch(prefix: &[u32], cut: usize, vocab: usize, index: usize) -> Vec<u32> {
    match index % 3 {
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
    }
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
    let measurements = measure_depth_samples(runtime, prefix, warmups, repetitions)?;
    let median_hibernate_nanoseconds = median(&measurements.hibernate_nanoseconds);
    let median_wake_nanoseconds = median(&measurements.wake_nanoseconds);
    let median_cold_nanoseconds = median(&measurements.cold_nanoseconds);
    let median_wake_speedup = median_cold_nanoseconds as f64 / median_wake_nanoseconds as f64;
    let correctness = measure_depth_correctness(runtime, prefix, continuation_tokens)?;
    let exact_wake_tokens = correctness.result_tokens == correctness.oracle_tokens;
    let host_wake_replay = correctness.replay_class == SessionReuseClass::HostWake
        && correctness.reused_tokens == prefix.len()
        && correctness.replayed_tokens == 0;
    let speed_passed = speed_gate
        .map(|gate| median_wake_speedup >= gate)
        .unwrap_or(true);
    let source_session_released = measurements.source_session_released;
    Ok(DepthResult {
        name,
        prefix_tokens: prefix.len(),
        context_tokens: measurements.context_tokens,
        continuation_tokens,
        host_bytes: measurements.host_bytes,
        hibernate_nanoseconds: measurements.hibernate_nanoseconds,
        wake_nanoseconds: measurements.wake_nanoseconds,
        cold_nanoseconds: measurements.cold_nanoseconds,
        median_hibernate_nanoseconds,
        median_wake_nanoseconds,
        median_cold_nanoseconds,
        median_wake_speedup,
        wake_transcript_sha256: token_stream_sha256(&correctness.result_tokens),
        oracle_transcript_sha256: token_stream_sha256(&correctness.oracle_tokens),
        exact_wake_tokens,
        source_session_released,
        shared_prefix_reused_tokens: correctness.reused_tokens,
        shared_prefix_computed_tokens: prefix.len().saturating_sub(correctness.reused_tokens),
        host_wake_replay,
        speed_gate,
        gate_passed: exact_wake_tokens
            && source_session_released
            && host_wake_replay
            && speed_passed,
    })
}

fn measure_depth_samples(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    warmups: usize,
    repetitions: usize,
) -> Result<DepthMeasurements, Box<dyn Error>> {
    for _ in 0..warmups {
        warmup_hibernation(runtime, prefix)?;
    }
    let mut measurements = DepthMeasurements {
        host_bytes: 0,
        context_tokens: 0,
        hibernate_nanoseconds: Vec::with_capacity(repetitions),
        wake_nanoseconds: Vec::with_capacity(repetitions),
        cold_nanoseconds: Vec::with_capacity(repetitions),
        source_session_released: true,
    };
    for repetition in 0..repetitions {
        measure_hibernation_repetition(runtime, prefix, repetition, &mut measurements)?;
    }
    Ok(measurements)
}

fn warmup_hibernation(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
) -> Result<(), Box<dyn Error>> {
    let mut source = prepare_session(runtime, prefix)?;
    let snapshot = runtime.hibernate_session(&mut source)?;
    drop(runtime.wake_session(snapshot)?);
    drop(prepare_session(runtime, prefix)?);
    Ok(())
}

fn measure_hibernation_repetition(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    repetition: usize,
    measurements: &mut DepthMeasurements,
) -> Result<(), Box<dyn Error>> {
    let mut source = prepare_session(runtime, prefix)?;
    if !repetition.is_multiple_of(2) {
        record_cold_prepare(runtime, prefix, &mut measurements.cold_nanoseconds)?;
    }
    let started = Instant::now();
    let snapshot = runtime.hibernate_session(&mut source)?;
    measurements
        .hibernate_nanoseconds
        .push(nanoseconds(started.elapsed().as_nanos())?);
    measurements.source_session_released &= source.is_empty();
    measurements.host_bytes = snapshot.record().host_bytes;
    measurements.context_tokens = snapshot.record().context_tokens;
    let started = Instant::now();
    drop(runtime.wake_session(snapshot)?);
    measurements
        .wake_nanoseconds
        .push(nanoseconds(started.elapsed().as_nanos())?);
    if repetition.is_multiple_of(2) {
        record_cold_prepare(runtime, prefix, &mut measurements.cold_nanoseconds)?;
    }
    Ok(())
}

fn record_cold_prepare(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    samples: &mut Vec<u64>,
) -> Result<(), io::Error> {
    let started = Instant::now();
    drop(prepare_session(runtime, prefix).map_err(|error| invalid_data(error.to_string()))?);
    samples.push(nanoseconds(started.elapsed().as_nanos())?);
    Ok(())
}

fn measure_depth_correctness(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    continuation_tokens: usize,
) -> Result<DepthCorrectness, Box<dyn Error>> {
    let HibernatedTokenEvidence {
        result_tokens,
        replay_class,
        reused_tokens,
        replayed_tokens,
    } = hibernated_tokens(runtime, prefix, continuation_tokens)?;
    let oracle_tokens = session_tokens(runtime, prefix, continuation_tokens)?;
    Ok(DepthCorrectness {
        result_tokens,
        oracle_tokens,
        replay_class,
        reused_tokens,
        replayed_tokens,
    })
}

struct HibernatedTokenEvidence {
    result_tokens: Vec<u32>,
    replay_class: SessionReuseClass,
    reused_tokens: usize,
    replayed_tokens: usize,
}

fn hibernated_tokens(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    continuation_tokens: usize,
) -> Result<HibernatedTokenEvidence, Box<dyn Error>> {
    let mut source = prepare_session(runtime, prefix)?;
    let snapshot = runtime.hibernate_session(&mut source)?;
    let mut woken = runtime.wake_session(snapshot)?;
    let result = runtime.generate_session_tokens(
        &mut woken,
        prefix,
        options(continuation_tokens),
        |_| Ok(()),
        || false,
    )?;
    let replay = woken.last_replay();
    Ok(HibernatedTokenEvidence {
        result_tokens: result.tokens,
        replay_class: replay.reuse_class,
        reused_tokens: replay.reused_tokens,
        replayed_tokens: replay.replayed_tokens,
    })
}

fn session_tokens(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    continuation_tokens: usize,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut oracle_session = prepare_session(runtime, prefix)?;
    Ok(runtime
        .generate_session_tokens(
            &mut oracle_session,
            prefix,
            options(continuation_tokens),
            |_| Ok(()),
            || false,
        )?
        .tokens)
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
    let mut parsed = ArgumentBuilder::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_paths(flag, arguments, &mut index, &mut parsed)?
            || parse_counts(flag, arguments, &mut index, &mut parsed)?
            || parse_gate(flag, arguments, &mut index, &mut parsed)?;
        if !handled {
            return Err(invalid_data(format!(
                "verify hibernate argument is invalid: {flag}"
            )));
        }
        index += 1;
    }
    parsed.finish()
}

#[derive(Default)]
struct ArgumentBuilder {
    model: Option<PathBuf>,
    cuts: Option<usize>,
    warmups: Option<usize>,
    repetitions: Option<usize>,
    prefix_tokens: Option<usize>,
    continuation_tokens: Option<usize>,
    speedup_gate: Option<Option<f64>>,
}

impl ArgumentBuilder {
    fn finish(self) -> Result<Arguments, io::Error> {
        Ok(Arguments {
            model: self
                .model
                .ok_or_else(|| invalid_data("verify hibernate requires -m <gguf>"))?,
            cuts: self.cuts.unwrap_or(32),
            warmups: self.warmups.unwrap_or(2),
            repetitions: self.repetitions.unwrap_or(5),
            prefix_tokens: self.prefix_tokens.unwrap_or(3584),
            continuation_tokens: self.continuation_tokens.unwrap_or(32),
            speedup_gate: self.speedup_gate.unwrap_or(Some(5.0)),
        })
    }
}

fn parse_paths(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut ArgumentBuilder,
) -> Result<bool, io::Error> {
    if flag != "-m" && flag != "--model" {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(value(arguments, index)?));
    Ok(true)
}

fn parse_counts(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut ArgumentBuilder,
) -> Result<bool, io::Error> {
    let target = match flag {
        "--cuts" => (&mut parsed.cuts, "cuts"),
        "--warmups" => (&mut parsed.warmups, "warmups"),
        "--repetitions" => (&mut parsed.repetitions, "repetitions"),
        "--prefix-tokens" => (&mut parsed.prefix_tokens, "prefix tokens"),
        "--continuation-tokens" => (&mut parsed.continuation_tokens, "continuation tokens"),
        _ => return Ok(false),
    };
    *target.0 = Some(positive(value(arguments, index)?, target.1)?);
    Ok(true)
}

fn parse_gate(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut ArgumentBuilder,
) -> Result<bool, io::Error> {
    match flag {
        "--speedup-gate" => {
            let raw = value(arguments, index)?;
            let gate = raw
                .parse::<f64>()
                .map_err(|_| invalid_data(format!("speedup gate is invalid: {raw}")))?;
            if !gate.is_finite() || gate <= 0.0 {
                return Err(invalid_data("speedup gate must be finite and positive"));
            }
            parsed.speedup_gate = Some(Some(gate));
            Ok(true)
        }
        "--exact-only" => {
            parsed.speedup_gate = Some(None);
            Ok(true)
        }
        _ => Ok(false),
    }
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
