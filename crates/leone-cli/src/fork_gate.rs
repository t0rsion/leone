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

struct DepthMeasurements {
    copied_bytes: u64,
    context_tokens: usize,
    fork_nanoseconds: Vec<u64>,
    cold_nanoseconds: Vec<u64>,
}

struct DepthCorrectness {
    child_tokens: Vec<u32>,
    oracle_tokens: Vec<u32>,
    parent_tokens: Vec<u32>,
    replay_class: SessionReuseClass,
    reused_tokens: usize,
    replayed_tokens: usize,
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, &arguments.model)?;
    let corpus = seed_corpus(&mut runtime, arguments.prefix_tokens)?;
    let (randomized_comparisons, randomized_failures) =
        randomized_differential(&mut runtime, &corpus, arguments.cuts)?;
    let (cancellation_child_empty, cancellation_parent_unchanged) =
        cancellation_differential(&mut runtime, &corpus[..96])?;
    let depths = measure_depths(&mut runtime, &corpus, &arguments)?;
    let gate = build_gate(
        randomized_comparisons,
        randomized_failures,
        cancellation_child_empty,
        cancellation_parent_unchanged,
        depths,
    );
    serde_json::to_writer_pretty(io::stdout().lock(), &gate)?;
    println!();
    if !gate.gate_passed {
        return Err(invalid_data("exact live session fork gate failed").into());
    }
    Ok(())
}

fn seed_corpus(
    runtime: &mut Runtime<CudaBackend>,
    prefix_tokens: usize,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let seed_tokens = runtime.model().tokenizer().encode(
        "Exact session forks preserve a shared prefix while independent branches continue.",
    )?;
    if seed_tokens.is_empty() {
        return Err(invalid_data("fork gate seed text produced no tokens").into());
    }
    Ok(repeated_prefix(&seed_tokens, prefix_tokens.max(128)))
}

fn measure_depths(
    runtime: &mut Runtime<CudaBackend>,
    corpus: &[u32],
    arguments: &Arguments,
) -> Result<Vec<DepthResult>, Box<dyn Error>> {
    Ok(vec![
        measure_depth(
            runtime,
            "shallow-losing-case",
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
    randomized_failures: Vec<RandomizedFailure>,
    cancellation_child_empty: bool,
    cancellation_parent_unchanged: bool,
    depths: Vec<DepthResult>,
) -> ForkGate {
    let randomized_mismatches = randomized_failures.len();
    let exact_gate_passed = randomized_mismatches == 0
        && cancellation_child_empty
        && cancellation_parent_unchanged
        && depths.iter().all(|depth| {
            depth.exact_child_tokens && depth.unchanged_parent_tokens && depth.device_fork_replay
        });
    let speed_gate_passed = depths.last().is_some_and(|depth| depth.gate_passed);
    ForkGate {
        schema_version: 1,
        randomized_comparisons,
        randomized_mismatches,
        randomized_failures,
        cancellation_child_empty,
        cancellation_parent_unchanged,
        depths,
        exact_gate_passed,
        speed_gate_passed,
        gate_passed: exact_gate_passed && speed_gate_passed,
    }
}

fn parse(arguments: &[String]) -> Result<Arguments, io::Error> {
    let mut parsed = ForkBuilder::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_fork_paths(&mut parsed, arguments, &mut index)?
            || parse_fork_sizes(&mut parsed, arguments, &mut index)?
            || parse_fork_gate(&mut parsed, arguments, &mut index)?;
        if !handled {
            return Err(invalid_data(format!(
                "verify fork argument is invalid: {flag}"
            )));
        }
        index += 1;
    }
    parsed.finish()
}

#[derive(Debug)]
struct ForkBuilder {
    model: Option<PathBuf>,
    cuts: usize,
    warmups: usize,
    repetitions: usize,
    prefix_tokens: usize,
    continuation_tokens: usize,
    speedup_gate: Option<f64>,
}

impl Default for ForkBuilder {
    fn default() -> Self {
        Self {
            model: None,
            cuts: DEFAULT_CUTS,
            warmups: DEFAULT_WARMUPS,
            repetitions: DEFAULT_REPETITIONS,
            prefix_tokens: DEFAULT_PREFIX,
            continuation_tokens: DEFAULT_CONTINUATION,
            speedup_gate: Some(DEFAULT_SPEEDUP_GATE),
        }
    }
}

impl ForkBuilder {
    fn finish(self) -> Result<Arguments, io::Error> {
        Ok(Arguments {
            model: self
                .model
                .ok_or_else(|| invalid_data("verify fork requires -m <gguf>"))?,
            cuts: self.cuts,
            warmups: self.warmups,
            repetitions: self.repetitions,
            prefix_tokens: self.prefix_tokens,
            continuation_tokens: self.continuation_tokens,
            speedup_gate: self.speedup_gate,
        })
    }
}

fn parse_fork_paths(
    parsed: &mut ForkBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "-m" && arguments[*index] != "--model" {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(value(arguments, index)?));
    Ok(true)
}

fn parse_fork_sizes(
    parsed: &mut ForkBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_fork_counts(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_fork_lengths(parsed, arguments, index)
}

fn parse_fork_counts(
    parsed: &mut ForkBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_fork_cut_warmup(parsed, arguments, index)? {
        return Ok(true);
    }
    if arguments[*index] != "--repetitions" {
        return Ok(false);
    }
    parsed.repetitions = positive(value(arguments, index)?, "repetitions")?;
    Ok(true)
}

fn parse_fork_cut_warmup(
    parsed: &mut ForkBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--cuts" => parsed.cuts = positive(value(arguments, index)?, "cuts")?,
        "--warmups" => parsed.warmups = positive(value(arguments, index)?, "warmups")?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_fork_lengths(
    parsed: &mut ForkBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--prefix-tokens" => {
            parsed.prefix_tokens = positive(value(arguments, index)?, "prefix tokens")?;
        }
        "--continuation-tokens" => {
            parsed.continuation_tokens = positive(value(arguments, index)?, "continuation tokens")?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_fork_gate(
    parsed: &mut ForkBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--speedup-gate" => {
            let raw = value(arguments, index)?;
            let gate: f64 = raw
                .parse()
                .map_err(|_| invalid_data(format!("speedup gate is invalid: {raw}")))?;
            if !gate.is_finite() || gate <= 0.0 {
                return Err(invalid_data("speedup gate must be finite and positive"));
            }
            parsed.speedup_gate = Some(gate);
        }
        "--exact-only" => parsed.speedup_gate = None,
        _ => return Ok(false),
    }
    Ok(true)
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
        let (next_draw, case_failures) = randomized_case(runtime, corpus, index, draw, vocab)?;
        draw = next_draw;
        comparisons += 2;
        failures.extend(case_failures);
    }
    Ok((comparisons, failures))
}

fn randomized_case(
    runtime: &mut Runtime<CudaBackend>,
    corpus: &[u32],
    index: usize,
    draw: u64,
    vocab: usize,
) -> Result<(u64, Vec<RandomizedFailure>), Box<dyn Error>> {
    let draw = draw
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        .rotate_left(17)
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let cut = 65 + draw as usize % 32;
    let prefix = &corpus[..cut];
    let mut parent = prepare_session(runtime, prefix)?;
    let mut child = runtime.fork_session(&parent)?;
    let (branch_name, branch) = fork_branch(prefix, cut, vocab, index);
    let mut failures = Vec::new();
    if let Some(failure) = check_child_branch(
        runtime,
        prefix,
        &mut child,
        &branch,
        index,
        cut,
        branch_name,
    )? {
        failures.push(failure);
    }
    if let Some(failure) =
        check_parent_branch(runtime, prefix, &mut parent, index, cut, branch_name)?
    {
        failures.push(failure);
    }
    Ok((draw, failures))
}

fn check_child_branch(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    child: &mut GenerationSession<CudaBackend>,
    branch: &[u32],
    index: usize,
    cut: usize,
    branch_name: &'static str,
) -> Result<Option<RandomizedFailure>, Box<dyn Error>> {
    let child_result =
        runtime.generate_session_tokens(child, branch, options(8), |_| Ok(()), || false)?;
    let mut oracle_session = prepare_session(runtime, prefix)?;
    let oracle = runtime.generate_session_tokens(
        &mut oracle_session,
        branch,
        options(8),
        |_| Ok(()),
        || false,
    )?;
    let replay = child.last_replay();
    let token_match = child_result.tokens == oracle.tokens;
    if token_match && replay.reuse_class == SessionReuseClass::DeviceFork {
        return Ok(None);
    }
    Ok(Some(RandomizedFailure {
        index,
        cut,
        branch: branch_name,
        target: "child",
        token_match,
        replay_class: reuse_class_name(replay.reuse_class),
        reused_tokens: replay.reused_tokens,
        computed_tokens: replay.computed_tokens,
    }))
}

fn check_parent_branch(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    parent: &mut GenerationSession<CudaBackend>,
    index: usize,
    cut: usize,
    branch_name: &'static str,
) -> Result<Option<RandomizedFailure>, Box<dyn Error>> {
    let parent_result =
        runtime.generate_session_tokens(parent, prefix, options(8), |_| Ok(()), || false)?;
    let mut parent_oracle_session = prepare_session(runtime, prefix)?;
    let parent_oracle = runtime.generate_session_tokens(
        &mut parent_oracle_session,
        prefix,
        options(8),
        |_| Ok(()),
        || false,
    )?;
    if parent_result.tokens == parent_oracle.tokens {
        return Ok(None);
    }
    let replay = parent.last_replay();
    Ok(Some(RandomizedFailure {
        index,
        cut,
        branch: branch_name,
        target: "parent",
        token_match: false,
        replay_class: reuse_class_name(replay.reuse_class),
        reused_tokens: replay.reused_tokens,
        computed_tokens: replay.computed_tokens,
    }))
}

fn fork_branch(prefix: &[u32], cut: usize, vocab: usize, index: usize) -> (&'static str, Vec<u32>) {
    match index % 3 {
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
    }
}

fn cancellation_differential(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
) -> Result<(bool, bool), Box<dyn Error>> {
    let oracle_tokens = cancellation_oracle(runtime, prefix)?;
    let (cancelled, parent_tokens) = cancellation_run(runtime, prefix)?;
    Ok((cancelled, parent_tokens == oracle_tokens))
}

fn cancellation_oracle(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut session = prepare_session(runtime, prefix)?;
    Ok(runtime
        .generate_session_tokens(&mut session, prefix, options(8), |_| Ok(()), || false)?
        .tokens)
}

fn cancellation_run(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
) -> Result<(bool, Vec<u32>), Box<dyn Error>> {
    let mut parent = prepare_session(runtime, prefix)?;
    let mut child = runtime.fork_session(&parent)?;
    let cancelled =
        runtime.generate_session_tokens(&mut child, prefix, options(8), |_| Ok(()), || true)?;
    let parent_result =
        runtime.generate_session_tokens(&mut parent, prefix, options(8), |_| Ok(()), || false)?;
    Ok((
        cancelled.stats.cancelled && child.is_empty(),
        parent_result.tokens,
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
    let measurements = measure_depth_samples(runtime, prefix, warmups, repetitions)?;
    let median_fork_nanoseconds = median(&measurements.fork_nanoseconds);
    let median_cold_nanoseconds = median(&measurements.cold_nanoseconds);
    let median_speedup = median_cold_nanoseconds as f64 / median_fork_nanoseconds as f64;
    let correctness = measure_depth_correctness(runtime, prefix, continuation_tokens)?;
    let exact_child_tokens = correctness.child_tokens == correctness.oracle_tokens;
    let unchanged_parent_tokens = correctness.parent_tokens == correctness.oracle_tokens;
    let device_fork_replay = correctness.replay_class == SessionReuseClass::DeviceFork
        && correctness.reused_tokens == prefix.len()
        && correctness.replayed_tokens == 0;
    let passed_speed = speed_gate
        .map(|gate| median_speedup >= gate)
        .unwrap_or(true);
    Ok(DepthResult {
        name,
        prefix_tokens: prefix.len(),
        context_tokens: measurements.context_tokens,
        continuation_tokens,
        copied_bytes: measurements.copied_bytes,
        fork_nanoseconds: measurements.fork_nanoseconds,
        cold_nanoseconds: measurements.cold_nanoseconds,
        median_fork_nanoseconds,
        median_cold_nanoseconds,
        median_speedup,
        child_transcript_sha256: token_stream_sha256(&correctness.child_tokens),
        oracle_transcript_sha256: token_stream_sha256(&correctness.oracle_tokens),
        exact_child_tokens,
        unchanged_parent_tokens,
        shared_prefix_reused_tokens: correctness.reused_tokens,
        shared_prefix_computed_tokens: prefix.len().saturating_sub(correctness.reused_tokens),
        device_fork_replay,
        speed_gate,
        gate_passed: exact_child_tokens
            && unchanged_parent_tokens
            && device_fork_replay
            && passed_speed,
    })
}

fn measure_depth_samples(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    warmups: usize,
    repetitions: usize,
) -> Result<DepthMeasurements, Box<dyn Error>> {
    for _ in 0..warmups {
        warmup_fork(runtime, prefix)?;
    }
    let mut measurements = DepthMeasurements {
        copied_bytes: 0,
        context_tokens: 0,
        fork_nanoseconds: Vec::with_capacity(repetitions),
        cold_nanoseconds: Vec::with_capacity(repetitions),
    };
    for repetition in 0..repetitions {
        measure_fork_repetition(runtime, prefix, repetition, &mut measurements)?;
    }
    Ok(measurements)
}

fn warmup_fork(runtime: &mut Runtime<CudaBackend>, prefix: &[u32]) -> Result<(), Box<dyn Error>> {
    let parent = prepare_session(runtime, prefix)?;
    drop(runtime.fork_session(&parent)?);
    drop(prepare_session(runtime, prefix)?);
    Ok(())
}

fn measure_fork_repetition(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    repetition: usize,
    measurements: &mut DepthMeasurements,
) -> Result<(), Box<dyn Error>> {
    let parent = prepare_session(runtime, prefix)?;
    if repetition.is_multiple_of(2) {
        record_fork(runtime, prefix, &parent, measurements)?;
        record_cold(runtime, prefix, &mut measurements.cold_nanoseconds)?;
    } else {
        record_cold(runtime, prefix, &mut measurements.cold_nanoseconds)?;
        record_fork(runtime, prefix, &parent, measurements)?;
    }
    Ok(())
}

fn record_fork(
    runtime: &mut Runtime<CudaBackend>,
    _prefix: &[u32],
    parent: &GenerationSession<CudaBackend>,
    measurements: &mut DepthMeasurements,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let child = runtime.fork_session(parent)?;
    measurements
        .fork_nanoseconds
        .push(nanoseconds(started.elapsed().as_nanos())?);
    let record = child
        .last_fork()
        .ok_or_else(|| invalid_data("fork record is missing"))?;
    measurements.copied_bytes = record.copied_bytes;
    measurements.context_tokens = record.context_tokens;
    drop(child);
    Ok(())
}

fn record_cold(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    samples: &mut Vec<u64>,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    drop(prepare_session(runtime, prefix)?);
    samples.push(nanoseconds(started.elapsed().as_nanos())?);
    Ok(())
}

fn measure_depth_correctness(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    continuation_tokens: usize,
) -> Result<DepthCorrectness, Box<dyn Error>> {
    let ForkChildEvidence {
        mut parent,
        child_tokens,
        replay_class,
        reused_tokens,
        replayed_tokens,
    } = fork_child_tokens(runtime, prefix, continuation_tokens)?;
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
    Ok(DepthCorrectness {
        child_tokens,
        oracle_tokens: oracle.tokens,
        parent_tokens: parent_result.tokens,
        replay_class,
        reused_tokens,
        replayed_tokens,
    })
}

struct ForkChildEvidence {
    parent: GenerationSession<CudaBackend>,
    child_tokens: Vec<u32>,
    replay_class: SessionReuseClass,
    reused_tokens: usize,
    replayed_tokens: usize,
}

fn fork_child_tokens(
    runtime: &mut Runtime<CudaBackend>,
    prefix: &[u32],
    continuation_tokens: usize,
) -> Result<ForkChildEvidence, Box<dyn Error>> {
    let parent = prepare_session(runtime, prefix)?;
    let mut child = runtime.fork_session(&parent)?;
    let child_tokens = runtime
        .generate_session_tokens(
            &mut child,
            prefix,
            options(continuation_tokens),
            |_| Ok(()),
            || false,
        )?
        .tokens;
    let replay = child.last_replay();
    Ok(ForkChildEvidence {
        parent,
        child_tokens,
        replay_class: replay.reuse_class,
        reused_tokens: replay.reused_tokens,
        replayed_tokens: replay.replayed_tokens,
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
