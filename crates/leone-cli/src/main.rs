mod build_info;
mod chat;
mod correctable_gate;
mod doctor;
mod eval;
mod execution_plan;
mod fork_gate;
mod hibernation_gate;
mod quality;
mod registry;
mod server;
mod verify;

use chrono::Utc;
use leone::{
    token_stream_sha256, AdaptiveDrafter, AdaptiveDrafterConfig, Backend, CorrectableDrafter,
    CpuBackend, DecodeExecution, DecodeOp, DecodeProfileMode, Determinism, GenerateOptions,
    GenerationResult, KvCacheDtype, LogitCapture, MeasuredRate, Penalties, PrefillWorkspace,
    Runtime, RuntimeError, Sampler, Speculation, SuffixDrafter, Temperature, Truncation,
    DEFAULT_PREFILL_CHUNK_TOKENS,
};
use leone_cuda::CudaBackend;
use leone_gguf::model::ModelConfig;
use leone_gguf::{GgmlType, Gguf};
use leone_receipt::{
    sha256_file, summarize_duration_samples_ms, summarize_samples, write_runtime_receipt,
    DeterminismClaim, DurationSummary, Engine, GpuClocksMhz, Machine, ModelArtifact,
    PrefillMethod as ReceiptPrefillMethod, QualityReceipt, QualitySummary, RateSummary,
    ReductionOrder, Roofline, RuntimeReceipt, RuntimeResults, SamplerRecord, SpeculationRecord,
    TensorClass, UsableBar, Workload, ROOFLINE_DENOMINATOR_DEFINITION, RUNTIME_SCHEMA_VERSION,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SPEC_BANDWIDTH_GBS: f64 = 1008.0;
const MEASURED_ACHIEVABLE_BANDWIDTH_GBS: f64 = 930.0;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if dispatch(&arguments)? {
        return Ok(());
    }
    print_help();
    Err(invalid_data("invalid arguments").into())
}

fn dispatch(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    if dispatch_display(arguments) {
        return Ok(true);
    }
    if dispatch_services(arguments)? {
        return Ok(true);
    }
    if dispatch_generation(arguments)? {
        return Ok(true);
    }
    if dispatch_inspection(arguments)? {
        return Ok(true);
    }
    if dispatch_verification(arguments)? {
        return Ok(true);
    }
    dispatch_benchmarks(arguments)
}

fn dispatch_display(arguments: &[String]) -> bool {
    if arguments.is_empty() {
        print_version();
        return true;
    }
    let [argument] = arguments else {
        return false;
    };
    dispatch_display_flag(argument)
}

fn dispatch_display_flag(argument: &str) -> bool {
    match argument {
        "--version" | "-V" => print_version(),
        "--help" | "-h" => print_help(),
        "--build-info" => build_info::print(),
        _ => return false,
    }
    true
}

fn dispatch_services(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    let [command, rest @ ..] = arguments else {
        return Ok(false);
    };
    if dispatch_chat_doctor(command, rest)? {
        return Ok(true);
    }
    if dispatch_registry_commands(command, rest)? {
        return Ok(true);
    }
    dispatch_runtime_commands(command, rest)
}

fn dispatch_chat_doctor(command: &str, rest: &[String]) -> Result<bool, Box<dyn Error>> {
    match command {
        "chat" => chat::run(rest)?,
        "doctor" => doctor::run(rest)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn dispatch_registry_commands(command: &str, rest: &[String]) -> Result<bool, Box<dyn Error>> {
    match command {
        "pull" => registry::pull(rest)?,
        "run" => registry::run(rest)?,
        "models" => registry::list(rest)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn dispatch_runtime_commands(command: &str, rest: &[String]) -> Result<bool, Box<dyn Error>> {
    match command {
        "serve" => server::run(rest)?,
        "eval" => eval::run(rest)?,
        "quality" => quality::run(rest)?,
        "tune" => execution_plan::tune(rest)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn dispatch_generation(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    let [command, rest @ ..] = arguments else {
        return Ok(false);
    };
    if command != "generate" {
        return Ok(false);
    }
    generate_text(parse_generate(rest)?)?;
    Ok(true)
}

fn dispatch_inspection(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    match arguments {
        [command, plan, rest @ ..] if command == "inspect" && plan == "plan" => {
            execution_plan::inspect(rest)?;
            Ok(true)
        }
        [command, inspect, path] if command == "gguf" && inspect == "inspect" => {
            inspect_gguf(Path::new(path))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn dispatch_verification(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    let [command, rest @ ..] = arguments else {
        return Ok(false);
    };
    if command != "verify" {
        return Ok(false);
    }
    match rest {
        [kind, rest @ ..] => dispatch_verification_kind(kind, rest, &arguments[1..])?,
        [] => verify::run(rest)?,
    }
    Ok(true)
}

fn dispatch_verification_kind(
    kind: &str,
    rest: &[String],
    fallback: &[String],
) -> Result<(), Box<dyn Error>> {
    match kind {
        "session" | "scheduler" => dispatch_session_scheduler(kind, rest)?,
        "fork" | "hibernate" => dispatch_fork_hibernate(kind, rest)?,
        "correctable" => correctable_gate::run(rest)?,
        _ => verify::run(fallback)?,
    }
    Ok(())
}

fn dispatch_session_scheduler(command: &str, rest: &[String]) -> Result<(), Box<dyn Error>> {
    match command {
        "session" => server::run_session_gate(rest)?,
        "scheduler" => server::run_scheduler_gate(rest)?,
        _ => unreachable!(),
    }
    Ok(())
}

fn dispatch_fork_hibernate(command: &str, rest: &[String]) -> Result<(), Box<dyn Error>> {
    match command {
        "fork" => fork_gate::run(rest)?,
        "hibernate" => hibernation_gate::run(rest)?,
        _ => unreachable!(),
    }
    Ok(())
}

fn dispatch_benchmarks(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    if dispatch_bench_command(arguments)? {
        return Ok(true);
    }
    dispatch_receipt_command(arguments)
}

fn dispatch_bench_command(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    let [command, rest @ ..] = arguments else {
        return Ok(false);
    };
    if command != "bench" {
        return Ok(false);
    }
    run_bench(parse_bench(rest)?)?;
    Ok(true)
}

fn dispatch_receipt_command(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    if dispatch_llama_bench_receipt(arguments)? {
        return Ok(true);
    }
    dispatch_response_receipt(arguments)
}

fn dispatch_llama_bench_receipt(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    match arguments {
        [receipt, from_bench, path, rest @ ..]
            if receipt == "receipt" && from_bench == "from-llama-bench" =>
        {
            convert_llama_bench(Path::new(path), parse_quality_ref(rest)?)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn dispatch_response_receipt(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    match arguments {
        [receipt, verify_response, path]
            if receipt == "receipt" && verify_response == "verify-response" =>
        {
            server::verify_response_receipt(Path::new(path))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
fn print_version() {
    println!("leone {VERSION}");
}

fn print_help() {
    println!("leone {VERSION}");
    println!();
    println!("Usage:");
    println!("  leone --build-info    Print source provenance for this executable");
    println!("  leone");
    println!("  leone chat -m <gguf> [options]");
    println!("  leone doctor [-m <gguf>]");
    println!("  leone pull <model> [--registry <toml>]");
    println!("  leone run <model> [--serve] [options]");
    println!("  leone models [--registry <toml>]");
    println!("  leone serve -m <gguf> [options]");
    println!("  leone generate -m <gguf> -p <prompt> -n <tokens> [options]");
    println!("  leone eval -m <gguf> (--corpus <text> --token-limit <n>|--tokens <bin>) [options]");
    println!("  leone quality --oracle <f32> --subject <f32> [options]");
    println!("  leone tune -m <gguf> [--out <json>] [options]");
    println!("  leone inspect plan <json>");
    println!("  leone verify -m <gguf> --tokens <u32le> --commit <sha> [--receipt]");
    println!("  leone verify session -m <gguf> [--cuts <n>] [--tokens <n>]");
    println!("  leone verify scheduler -m <gguf> [--tokens <n>] [--quantum <n>]");
    println!("  leone verify fork -m <gguf> [--cuts <n>] [--repetitions <n>] [--exact-only]");
    println!("  leone verify hibernate -m <gguf> [--cuts <n>] [--repetitions <n>] [--exact-only]");
    println!(
        "  leone verify correctable [-m <gguf>] [--cases <n>] [--receipt] [--release-doc <path>]"
    );
    println!("  leone bench [--receipt] [--prefill-context <tokens>] [options]");
    println!("  leone receipt from-llama-bench <json> [--quality-ref <uuid>]");
    println!("  leone receipt verify-response <json>");
    println!("  leone gguf inspect <file>");
    println!();
    println!("Chat options:");
    println!("  -n, --tokens <n>     Set the maximum response length. Default: 512");
    println!("  --backend cuda|cpu   Select the execution backend. Default: cuda");
    println!("  --eager-decode       Disable decode graph replay");
    println!("  --kv q8|f16|f32      Select KV cache storage. Default: f16");
    println!("  --plan <json>         Load a proof-gated execution plan");
    println!();
    println!("Serve options:");
    println!("  --bind <address>      Listen address. Default: 127.0.0.1:8080");
    println!("  --sessions <n>        Maximum live KV sessions. Default: 2");
    println!("  --context-limit <n>   Per-session context bound. Default: model context.");
    println!("  --prefill-chunk <n>   Prompt tokens per scheduler chunk. Default: plan or 4096.");
    println!("  --batch-size <n>      Maximum requests per decode pass. Default: 8");
    println!("  --hibernated-sessions <n> Maximum host sessions. Default: 8");
    println!("  --kv q8|f16|f32      Select KV cache storage. Default: f16");
    println!("  --plan <json>         Load a proof-gated execution plan");
    println!("  --receipt-dir <path> Write signed response receipts. Default: receipts");
    println!("  --signing-key <path> Set the 32-byte Ed25519 key file");
    println!("  --session-store <path> Persist model-bound session replay state");
    println!("  --allow-remote        Permit a non-loopback listen address");
    println!();
    println!("Generate options:");
    println!("  --backend cuda|cpu    Select the execution backend. Default: cuda");
    println!("  --receipt             Write a runtime receipt");
    println!("  --profile-decode      Profile 64 steady-state decode evaluations");
    println!("  --eager-decode        Disable decode graph replay");
    println!("  --kv q8|f16|f32       Select KV cache storage. Default: f16");
    println!("  --plan <json>          Load a proof-gated execution plan");
    println!("  --debug-tokens <file> Write generated token IDs as JSON");
    println!("  --debug-logits <file> Write the top five logits for each token");
    println!("  --seed <n>            Seed the sampler. Default: 0");
    println!("  --draft <n>           Propose n tokens per step by suffix match");
    println!("  --adaptive-draft      Select a point proposal and width from measured rounds");
    println!(
        "  --correctable         Select a distribution proposal and width from measured rounds"
    );
    println!();
    println!("Sampling options. Truncation stages apply in the order given:");
    println!("  --temp <t>            Scale logits by 1/t. Default: greedy");
    println!("  --top-k <n>           Keep the n highest-probability tokens");
    println!("  --top-p <p>           Keep the smallest set covering mass p");
    println!("  --min-p <m>           Keep tokens above m times the highest");
    println!("  --top-a <a>           Keep tokens above a times the highest squared");
    println!("  --tfs <z>             Tail-free sampling on the second derivative");
    println!("  --typical <p>         Locally typical sampling to mass p");
    println!("  --epsilon <e>         Keep tokens with probability at least e");
    println!("  --eta <e>             Entropy-relaxed probability floor");
    println!("  --min-k <tau>         Keep the top k raw logits at the sharpest drop");
    println!("  --top-n-sigma <n>     Keep raw logits within n deviations of the peak");
    println!();
    println!();
    println!("Penalty options. These rewrite logits from the decode history,");
    println!("before any truncation stage:");
    println!("  --repeat-penalty <r>  Move a seen token's logit toward zero. 1.0 is off");
    println!("  --presence-penalty <p> Subtract once from any seen token");
    println!("  --frequency-penalty <f> Subtract once per earlier occurrence");
    println!("  --penalty-window <n>  Tokens of history considered. Default: 64");
    println!("  --dry-multiplier <m>  Enable the repeated-suffix penalty. 0.0 is off");
    println!("  --dry-base <b>        Growth per token of match. Default: 1.75");
    println!("  --dry-allowed-length <n> Free match length. Default: 2");
    println!("  --dry-window <n>      Tokens of history searched. Default: 64");
    println!();
    println!("Temperature applies before every stage except --min-k and");
    println!("--top-n-sigma, which read the raw logits and are temperature invariant.");
    println!();
    println!("Bench options:");
    println!("  --prefill-context <n> Measure TTFT for an additional prompt length");
    println!(
        "  --prefill-chunk <n>   Set the position block size. Default: {DEFAULT_PREFILL_CHUNK_TOKENS}."
    );
    println!("                        Lower it when memory is tight: the workspace");
    println!("                        grows with the block and with the prompt.");
    println!("  --eager-decode        Disable decode graph replay");
    println!("  --kv q8|f16|f32       Select KV cache storage. Default: f16");
    println!("  --receipt             Write a runtime receipt");
    println!();
    println!("Eval options:");
    println!("  --tokens-out <file>   Write shared little-endian u32 token IDs");
    println!("  --logits <file>       Write row-major little-endian f32 logits");
    println!("  --backend cuda|cpu|cpu-q8_1");
    println!("                         Select the execution backend. Default: cuda");
    println!("  --window <tokens>     Set the overlapping evaluation window. Default: 512");
    println!("  --position-limit <n> Score only the first n positions from --tokens");
}

#[derive(Debug, Clone, Copy)]
struct BenchArgs {
    receipt: bool,
    capacity_only: bool,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    quality_ref: Option<Uuid>,
    prefill_context: Option<usize>,
    prefill_chunk: usize,
}

fn parse_bench(arguments: &[String]) -> Result<BenchArgs, io::Error> {
    let mut parsed = BenchBuilder::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_bench_flag(&mut parsed, flag)?
            || parse_bench_value(&mut parsed, arguments, &mut index)?
            || parse_bench_kv(&mut parsed, arguments, &mut index)?;
        if !handled {
            return Err(invalid_data(format!("bench argument is invalid: {flag}")));
        }
        index += 1;
    }
    Ok(parsed.finish())
}

#[derive(Debug)]
struct BenchBuilder {
    receipt: bool,
    capacity_only: bool,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    quality_ref: Option<Uuid>,
    prefill_context: Option<usize>,
    prefill_chunk: usize,
}

impl Default for BenchBuilder {
    fn default() -> Self {
        Self {
            receipt: false,
            capacity_only: false,
            eager_decode: false,
            kv_cache_dtype: KvCacheDtype::F16,
            quality_ref: None,
            prefill_context: None,
            prefill_chunk: DEFAULT_PREFILL_CHUNK_TOKENS,
        }
    }
}

impl BenchBuilder {
    fn finish(self) -> BenchArgs {
        BenchArgs {
            receipt: self.receipt,
            capacity_only: self.capacity_only,
            eager_decode: self.eager_decode,
            kv_cache_dtype: self.kv_cache_dtype,
            quality_ref: self.quality_ref,
            prefill_context: self.prefill_context,
            prefill_chunk: self.prefill_chunk,
        }
    }
}

fn parse_bench_flag(parsed: &mut BenchBuilder, flag: &str) -> Result<bool, io::Error> {
    match flag {
        "--receipt" => parsed.receipt = true,
        "--capacity-only" => parsed.capacity_only = true,
        "--eager-decode" => parsed.eager_decode = true,
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_bench_value(
    parsed: &mut BenchBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_bench_quality_ref(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_bench_prefill_value(parsed, arguments, index)
}

fn parse_bench_quality_ref(
    parsed: &mut BenchBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--quality-ref" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.quality_ref = Some(
        Uuid::parse_str(value)
            .map_err(|_| invalid_data(format!("quality receipt UUID is invalid: {value}")))?,
    );
    Ok(true)
}

fn parse_bench_prefill_value(
    parsed: &mut BenchBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--prefill-context" => {
            let value = flag_value(arguments, index)?;
            parsed.prefill_context = Some(
                value
                    .parse()
                    .map_err(|_| invalid_data(format!("prefill context is invalid: {value}")))?,
            );
        }
        "--prefill-chunk" => {
            let value = flag_value(arguments, index)?;
            parsed.prefill_chunk = value
                .parse()
                .map_err(|_| invalid_data(format!("prefill chunk size is invalid: {value}")))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_bench_kv(
    parsed: &mut BenchBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--kv" {
        return Ok(false);
    }
    parsed.kv_cache_dtype = match flag_value(arguments, index)? {
        "q8" => KvCacheDtype::Q8,
        "f16" => KvCacheDtype::F16,
        "f32" => KvCacheDtype::F32,
        value => return Err(invalid_data(format!("KV dtype is invalid: {value}"))),
    };
    Ok(true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cuda,
    Cpu,
}

#[derive(Debug)]
struct GenerateArgs {
    model: PathBuf,
    prompt: String,
    tokens: usize,
    backend: BackendChoice,
    receipt: bool,
    profile_decode: bool,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    debug_tokens: Option<PathBuf>,
    debug_logits: Option<PathBuf>,
    sampler: Sampler,
    penalties: Penalties,
    seed: u64,
    speculation: Speculation,
    correctable: bool,
    plan: Option<PathBuf>,
}

/// Parses one finite floating-point flag value.
fn parse_f64(arguments: &[String], index: &mut usize, flag: &str) -> Result<f64, io::Error> {
    let value = flag_value(arguments, index)?;
    let parsed: f64 = value
        .parse()
        .map_err(|_| invalid_data(format!("{flag} is invalid: {value}")))?;
    if !parsed.is_finite() {
        return Err(invalid_data(format!("{flag} must be finite: {value}")));
    }
    Ok(parsed)
}

fn parse_generate(arguments: &[String]) -> Result<GenerateArgs, io::Error> {
    let mut parsed = GenerateBuilder::default();
    parse_generate_arguments(arguments, &mut parsed)?;
    parsed.finish()
}

fn parse_generate_arguments(
    arguments: &[String],
    parsed: &mut GenerateBuilder,
) -> Result<(), io::Error> {
    let mut index = 0;
    while index < arguments.len() {
        if parse_generate_argument(parsed, arguments, &mut index)? {
            index += 1;
            continue;
        }
        return Err(invalid_data(format!(
            "generate argument is invalid: {}",
            arguments[index]
        )));
    }
    Ok(())
}

fn parse_generate_argument(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_paths(parsed, arguments, index)?
        || parse_generate_runtime(parsed, arguments, index)?
    {
        return Ok(true);
    }
    if parse_generate_sampling(parsed, arguments, index)?
        || parse_generate_penalties(parsed, arguments, index)?
    {
        return Ok(true);
    }
    parse_generate_speculation(parsed, arguments, index)
}

#[derive(Debug)]
struct GenerateBuilder {
    model: Option<PathBuf>,
    prompt: Option<String>,
    tokens: Option<usize>,
    backend: BackendChoice,
    receipt: bool,
    profile_decode: bool,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    debug_tokens: Option<PathBuf>,
    debug_logits: Option<PathBuf>,
    temperature: Temperature,
    truncations: Vec<Truncation>,
    seed: u64,
    draft: Option<NonZeroUsize>,
    adaptive_draft: bool,
    correctable: bool,
    plan: Option<PathBuf>,
    penalties: Penalties,
}

impl Default for GenerateBuilder {
    fn default() -> Self {
        Self {
            model: None,
            prompt: None,
            tokens: None,
            backend: BackendChoice::Cuda,
            receipt: false,
            profile_decode: false,
            eager_decode: false,
            kv_cache_dtype: KvCacheDtype::F16,
            debug_tokens: None,
            debug_logits: None,
            temperature: Temperature::Greedy,
            truncations: Vec::new(),
            seed: 0,
            draft: None,
            adaptive_draft: false,
            correctable: false,
            plan: None,
            penalties: Penalties::none(),
        }
    }
}

impl GenerateBuilder {
    fn finish(self) -> Result<GenerateArgs, io::Error> {
        let draft_modes = usize::from(self.draft.is_some())
            + usize::from(self.adaptive_draft)
            + usize::from(self.correctable);
        if draft_modes > 1 {
            return Err(invalid_data(
                "set only one of --draft, --adaptive-draft, or --correctable",
            ));
        }
        Ok(GenerateArgs {
            model: self
                .model
                .ok_or_else(|| invalid_data("generate requires -m <gguf>"))?,
            prompt: self
                .prompt
                .ok_or_else(|| invalid_data("generate requires -p <prompt>"))?,
            tokens: self
                .tokens
                .ok_or_else(|| invalid_data("generate requires -n <tokens>"))?,
            backend: self.backend,
            receipt: self.receipt,
            profile_decode: self.profile_decode,
            eager_decode: self.eager_decode,
            kv_cache_dtype: self.kv_cache_dtype,
            debug_tokens: self.debug_tokens,
            debug_logits: self.debug_logits,
            sampler: Sampler {
                temperature: self.temperature,
                truncations: self.truncations,
            },
            penalties: self.penalties,
            seed: self.seed,
            speculation: build_speculation(self.draft, self.adaptive_draft)?,
            correctable: self.correctable,
            plan: self.plan,
        })
    }
}

fn parse_generate_paths(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_model_prompt(parsed, arguments, index)? {
        return Ok(true);
    }
    if parse_generate_outputs(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_generate_plan(parsed, arguments, index)
}

fn parse_generate_model_prompt(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "-m" | "--model" => parsed.model = Some(PathBuf::from(flag_value(arguments, index)?)),
        "-p" | "--prompt" => parsed.prompt = Some(flag_value(arguments, index)?.to_owned()),
        "-n" | "--tokens" => {
            let value = flag_value(arguments, index)?;
            parsed.tokens = Some(value.parse().map_err(|_| {
                invalid_data(format!("generation token count is invalid: {value}"))
            })?);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_generate_outputs(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--debug-tokens" => {
            parsed.debug_tokens = Some(PathBuf::from(flag_value(arguments, index)?));
        }
        "--debug-logits" => {
            parsed.debug_logits = Some(PathBuf::from(flag_value(arguments, index)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_generate_plan(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--plan" {
        return Ok(false);
    }
    parsed.plan = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_generate_runtime(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_backend(parsed, arguments, index)? {
        return Ok(true);
    }
    if parse_generate_runtime_flags(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_generate_kv(parsed, arguments, index)
}

fn parse_generate_backend(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--backend" {
        return Ok(false);
    }
    parsed.backend = match flag_value(arguments, index)? {
        "cuda" => BackendChoice::Cuda,
        "cpu" => BackendChoice::Cpu,
        value => return Err(invalid_data(format!("backend is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_generate_runtime_flags(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--receipt" => parsed.receipt = true,
        "--profile-decode" => parsed.profile_decode = true,
        "--eager-decode" => parsed.eager_decode = true,
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_generate_kv(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--kv" {
        return Ok(false);
    }
    parsed.kv_cache_dtype = match flag_value(arguments, index)? {
        "q8" => KvCacheDtype::Q8,
        "f16" => KvCacheDtype::F16,
        "f32" => KvCacheDtype::F32,
        value => return Err(invalid_data(format!("KV dtype is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_generate_sampling(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_sampling_control(parsed, arguments, index)?
        || parse_generate_truncation(parsed, arguments, index)?
    {
        return Ok(true);
    }
    Ok(false)
}

fn parse_generate_sampling_control(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--temp" | "--temperature" => {
            parsed.temperature = Temperature::Scaled(parse_f64(arguments, index, "--temp")?);
        }
        "--seed" => {
            let value = flag_value(arguments, index)?;
            parsed.seed = value
                .parse()
                .map_err(|_| invalid_data(format!("seed is invalid: {value}")))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_generate_truncation(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_top_truncation(parsed, arguments, index)? {
        return Ok(true);
    }
    if parse_generate_tail_truncation(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_generate_raw_truncation(parsed, arguments, index)
}

fn parse_generate_top_truncation(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_top_k(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_generate_top_mass(parsed, arguments, index)
}

fn parse_generate_top_k(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--top-k" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    let count: usize = value
        .parse()
        .map_err(|_| invalid_data(format!("top-k is invalid: {value}")))?;
    let count =
        NonZeroUsize::new(count).ok_or_else(|| invalid_data("top-k must be greater than zero"))?;
    parsed.truncations.push(Truncation::TopK(count));
    Ok(true)
}

fn parse_generate_top_mass(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let truncation = match arguments[*index].as_str() {
        "--top-p" => Truncation::TopP(parse_f64(arguments, index, "--top-p")?),
        "--min-p" => Truncation::MinP(parse_f64(arguments, index, "--min-p")?),
        "--top-a" => Truncation::TopA(parse_f64(arguments, index, "--top-a")?),
        _ => return Ok(false),
    };
    parsed.truncations.push(truncation);
    Ok(true)
}

fn parse_generate_tail_truncation(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let truncation = match arguments[*index].as_str() {
        "--tfs" => Truncation::TailFree(parse_f64(arguments, index, "--tfs")?),
        "--typical" => Truncation::Typical(parse_f64(arguments, index, "--typical")?),
        "--epsilon" => Truncation::Epsilon(parse_f64(arguments, index, "--epsilon")?),
        "--eta" => Truncation::Eta(parse_f64(arguments, index, "--eta")?),
        _ => return Ok(false),
    };
    parsed.truncations.push(truncation);
    Ok(true)
}

fn parse_generate_raw_truncation(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let truncation = match arguments[*index].as_str() {
        "--min-k" => Truncation::MinK(parse_f64(arguments, index, "--min-k")?),
        "--top-n-sigma" => Truncation::TopNSigma(parse_f64(arguments, index, "--top-n-sigma")?),
        _ => return Ok(false),
    };
    parsed.truncations.push(truncation);
    Ok(true)
}

fn parse_generate_penalties(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_basic_penalty(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_generate_dry_penalty(parsed, arguments, index)
}

fn parse_generate_basic_penalty(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_penalty_values(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_generate_penalty_window(parsed, arguments, index)
}

fn parse_generate_penalty_values(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--repeat-penalty" => {
            parsed.penalties.repetition = parse_f64(arguments, index, "--repeat-penalty")?;
        }
        "--presence-penalty" => {
            parsed.penalties.presence = parse_f64(arguments, index, "--presence-penalty")?;
        }
        "--frequency-penalty" => {
            parsed.penalties.frequency = parse_f64(arguments, index, "--frequency-penalty")?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_generate_penalty_window(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--penalty-window" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.penalties.window = value
        .parse()
        .map_err(|_| invalid_data(format!("penalty window is invalid: {value}")))?;
    Ok(true)
}

fn parse_generate_dry_penalty(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_generate_dry_values(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_generate_dry_sizes(parsed, arguments, index)
}

fn parse_generate_dry_values(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--dry-multiplier" => {
            parsed.penalties.dry.multiplier = parse_f64(arguments, index, "--dry-multiplier")?;
        }
        "--dry-base" => {
            parsed.penalties.dry.base = parse_f64(arguments, index, "--dry-base")?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_generate_dry_sizes(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--dry-allowed-length" => {
            let value = flag_value(arguments, index)?;
            parsed.penalties.dry.allowed_length = value
                .parse()
                .map_err(|_| invalid_data(format!("DRY allowed length is invalid: {value}")))?;
        }
        "--dry-window" => {
            let value = flag_value(arguments, index)?;
            parsed.penalties.dry.window = value
                .parse()
                .map_err(|_| invalid_data(format!("DRY window is invalid: {value}")))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_generate_speculation(
    parsed: &mut GenerateBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--draft" => {
            let value = flag_value(arguments, index)?;
            let count: usize = value
                .parse()
                .map_err(|_| invalid_data(format!("draft length is invalid: {value}")))?;
            parsed.draft = Some(
                NonZeroUsize::new(count)
                    .ok_or_else(|| invalid_data("draft length must be greater than zero"))?,
            );
        }
        "--adaptive-draft" => parsed.adaptive_draft = true,
        "--correctable" => parsed.correctable = true,
        _ => return Ok(false),
    }
    Ok(true)
}

fn build_speculation(
    draft: Option<NonZeroUsize>,
    adaptive_draft: bool,
) -> Result<Speculation, io::Error> {
    if adaptive_draft {
        return Ok(Speculation::Adaptive(AdaptiveDrafter::new(
            AdaptiveDrafterConfig::default(),
        )));
    }
    let Some(proposal) = draft else {
        return Ok(Speculation::Disabled);
    };
    // Four-token look-back is the shortest match worth trusting.
    // Sixteen tokens is the longest that repeats often enough to be found.
    // Probe 2 measured this range.
    let longest = NonZeroUsize::new(16).expect("sixteen is nonzero");
    let shortest = NonZeroUsize::new(4).expect("four is nonzero");
    SuffixDrafter::new(longest, shortest, proposal)
        .map(Speculation::Suffix)
        .map_err(|error| invalid_data(error.to_string()))
}

fn flag_value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid_data("command flag is missing its value"))
}

fn generate_text(arguments: GenerateArgs) -> Result<(), Box<dyn Error>> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&interrupted);
    ctrlc::set_handler(move || handler_flag.store(true, Ordering::Relaxed))?;
    match arguments.backend {
        BackendChoice::Cuda => run_generation(CudaBackend::new(0)?, arguments, interrupted),
        BackendChoice::Cpu => run_generation(CpuBackend::new(), arguments, interrupted),
    }
}

#[derive(Debug, Deserialize)]
struct BenchManifest {
    model: BenchModel,
    workload: BenchWorkload,
    comparator: BenchComparator,
}

#[derive(Debug, Deserialize)]
struct BenchModel {
    filename: String,
    sha256: String,
    format: String,
}

#[derive(Debug, Deserialize)]
struct BenchWorkload {
    prefill_tokens: Vec<usize>,
    decode_tokens: Vec<usize>,
    context_cap_tokens: usize,
    batch: usize,
    reps: usize,
}

#[derive(Debug, Deserialize)]
struct BenchComparator {
    pin_file: String,
    prefill_raw_json: String,
    prefill_tokens: usize,
}

fn run_bench(arguments: BenchArgs) -> Result<(), Box<dyn Error>> {
    let BenchSetup {
        root,
        manifest,
        prompt_tokens,
        decode_tokens,
        actual_sha256,
        engine_commit,
        linked_quality,
        mut runtime,
        decode_execution,
    } = prepare_bench(&arguments)?;
    if arguments.capacity_only {
        let context = arguments.prefill_context.ok_or_else(|| {
            invalid_data("bench --capacity-only requires --prefill-context <tokens>")
        })?;
        return probe_capacity(&mut runtime, context, arguments.kv_cache_dtype);
    }
    let BenchRunEvidence {
        decode_summary,
        prefill_summary,
        ttft_summary,
        ttft_context,
        method,
        workspace,
        usable_bar,
        determinism,
    } = measure_bench(
        &mut runtime,
        BenchMeasurementInput {
            root: &root,
            prompt_tokens,
            decode_tokens,
            manifest: &manifest,
            arguments: &arguments,
            decode_execution,
            actual_sha256: &actual_sha256,
            linked_quality: linked_quality.as_ref(),
        },
    )?;
    write_bench_receipt_if_requested(
        &root,
        &runtime,
        &manifest,
        arguments,
        decode_summary,
        prefill_summary,
        ttft_summary,
        ttft_context,
        method,
        workspace,
        usable_bar,
        &engine_commit,
        linked_quality.as_ref(),
        determinism,
    )
}

struct BenchRunEvidence {
    decode_summary: RateSummary,
    prefill_summary: RateSummary,
    ttft_summary: DurationSummary,
    ttft_context: usize,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
    usable_bar: Option<UsableBar>,
    determinism: DeterminismClaim,
}

struct BenchMeasurementInput<'a> {
    root: &'a Path,
    prompt_tokens: usize,
    decode_tokens: usize,
    manifest: &'a BenchManifest,
    arguments: &'a BenchArgs,
    decode_execution: DecodeExecution,
    actual_sha256: &'a str,
    linked_quality: Option<&'a QualityReceipt>,
}

fn measure_bench(
    runtime: &mut Runtime<CudaBackend>,
    input: BenchMeasurementInput<'_>,
) -> Result<BenchRunEvidence, Box<dyn Error>> {
    let BenchMeasurementInput {
        root,
        prompt_tokens,
        decode_tokens,
        manifest,
        arguments,
        decode_execution,
        actual_sha256,
        linked_quality,
    } = input;
    let BenchDecodeEvidence {
        prompt,
        decode_samples,
        prefill_samples,
        detokenized_bytes,
        method,
        workspace,
        transcript,
        ttft_samples_ms,
    } = measure_decode(
        runtime,
        prompt_tokens,
        decode_tokens,
        manifest,
        arguments,
        decode_execution,
    )?;
    let determinism = DeterminismClaim::Reproduced {
        order: reduction_order(runtime.backend().determinism()),
        sampler: SamplerRecord::Greedy,
        prompt_sha256: token_stream_sha256(&prompt),
        transcript_sha256: transcript,
        identical_reps: u64::try_from(manifest.workload.reps)?,
    };
    let decode_summary = summarize_samples(&decode_samples)?;
    let prefill_summary = summarize_samples(&prefill_samples)?;
    println!(
        "decode median {:.3} tok/s, p10 {:.3}, p90 {:.3}",
        decode_summary.median, decode_summary.p10, decode_summary.p90
    );
    println!(
        "pp512 median {:.3} tok/s, p10 {:.3}, p90 {:.3}",
        prefill_summary.median, prefill_summary.p10, prefill_summary.p90
    );

    let BenchTtftEvidence {
        context: ttft_context,
        samples: ttft_samples_ms,
        method,
        workspace,
    } = measure_ttft(
        runtime,
        prompt_tokens,
        method,
        workspace,
        ttft_samples_ms,
        manifest,
        arguments,
    )?;
    let ttft_summary = summarize_duration_samples_ms(&ttft_samples_ms)?;
    println!(
        "TTFT at {ttft_context} tokens: median {:.3} ms, p10 {:.3}, p90 {:.3}",
        ttft_summary.median, ttft_summary.p10, ttft_summary.p90
    );
    println!("prefill method: {}", method.name());
    println!("prefill workspace: {} bytes", workspace.total_bytes);

    let usable_bar = bench_usable_bar(
        ttft_context,
        &prefill_summary,
        &ttft_summary,
        root,
        manifest,
        actual_sha256,
    )?;
    println!("detokenized bytes: {detokenized_bytes}");
    print_quality(linked_quality);
    Ok(BenchRunEvidence {
        decode_summary,
        prefill_summary,
        ttft_summary,
        ttft_context,
        method,
        workspace,
        usable_bar,
        determinism,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_bench_receipt_if_requested<B: Backend>(
    root: &Path,
    runtime: &Runtime<B>,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    decode_summary: RateSummary,
    prefill_summary: RateSummary,
    ttft_summary: DurationSummary,
    ttft_context: usize,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
    usable_bar: Option<UsableBar>,
    engine_commit: &str,
    linked_quality: Option<&QualityReceipt>,
    determinism: DeterminismClaim,
) -> Result<(), Box<dyn Error>> {
    if !arguments.receipt {
        return Ok(());
    }
    let prefill_evidence = BenchPrefillEvidence {
        ttft_ms: ttft_summary,
        ttft_context_tokens: u64::try_from(ttft_context)?,
        method,
        workspace,
        usable_bar,
    };
    let path = write_bench_receipt(
        root,
        runtime,
        manifest,
        arguments,
        decode_summary,
        prefill_summary,
        prefill_evidence,
        engine_commit,
        linked_quality,
        determinism,
    )?;
    println!("receipt: {}", path.display());
    Ok(())
}

struct BenchSetup {
    root: PathBuf,
    manifest: BenchManifest,
    prompt_tokens: usize,
    decode_tokens: usize,
    actual_sha256: String,
    engine_commit: String,
    linked_quality: Option<QualityReceipt>,
    runtime: Runtime<CudaBackend>,
    decode_execution: DecodeExecution,
}

fn prepare_bench(arguments: &BenchArgs) -> Result<BenchSetup, Box<dyn Error>> {
    let root = std::env::current_dir()?;
    let (manifest, prompt_tokens, decode_tokens) = load_bench_manifest(&root)?;
    validate_bench_request(arguments, &manifest, prompt_tokens, decode_tokens)?;
    let (actual_sha256, engine_commit, linked_quality) =
        load_bench_identity(&root, &manifest, arguments)?;
    let (runtime, decode_execution) = load_bench_runtime(&root, &manifest, arguments)?;
    Ok(BenchSetup {
        root,
        manifest,
        prompt_tokens,
        decode_tokens,
        actual_sha256,
        engine_commit,
        linked_quality,
        runtime,
        decode_execution,
    })
}

fn load_bench_manifest(root: &Path) -> Result<(BenchManifest, usize, usize), Box<dyn Error>> {
    let manifest: BenchManifest =
        toml::from_str(&fs::read_to_string(root.join("benchmarks/manifest.toml"))?)?;
    let prompt_tokens = one_manifest_value(&manifest.workload.prefill_tokens, "prefill_tokens")?;
    let decode_tokens = one_manifest_value(&manifest.workload.decode_tokens, "decode_tokens")?;
    Ok((manifest, prompt_tokens, decode_tokens))
}

fn load_bench_identity(
    root: &Path,
    manifest: &BenchManifest,
    arguments: &BenchArgs,
) -> Result<(String, String, Option<QualityReceipt>), Box<dyn Error>> {
    let model_path = root.join("models").join(&manifest.model.filename);
    let actual_sha256 = sha256_file(&model_path)?;
    validate_bench_model(&actual_sha256, &manifest.model.sha256)?;
    let engine_commit = command_output("git", &["rev-parse", "HEAD"])?
        .trim()
        .to_owned();
    let linked_quality = load_linked_quality(
        root,
        arguments.quality_ref,
        "leone",
        &engine_commit,
        &actual_sha256,
    )?;
    Ok((actual_sha256, engine_commit, linked_quality))
}

fn load_bench_runtime(
    root: &Path,
    manifest: &BenchManifest,
    arguments: &BenchArgs,
) -> Result<(Runtime<CudaBackend>, DecodeExecution), Box<dyn Error>> {
    let model_path = root.join("models").join(&manifest.model.filename);
    let runtime = Runtime::load(CudaBackend::new(0)?, &model_path)?;
    let decode_execution = bench_decode_execution(&runtime, arguments.eager_decode);
    Ok((runtime, decode_execution))
}

fn validate_bench_request(
    arguments: &BenchArgs,
    manifest: &BenchManifest,
    prompt_tokens: usize,
    decode_tokens: usize,
) -> Result<(), io::Error> {
    if manifest.workload.batch != 1 {
        return Err(invalid_data("bench requires batch = 1"));
    }
    if manifest.workload.reps == 0 {
        return Err(invalid_data("bench requires at least one repetition"));
    }
    if arguments.prefill_chunk == 0 {
        return Err(invalid_data("prefill chunk size must be nonzero"));
    }
    if arguments.prefill_context == Some(0) {
        return Err(invalid_data("prefill context must be nonzero"));
    }
    let requested_context = prompt_tokens
        .checked_add(decode_tokens)
        .ok_or_else(|| invalid_data("bench context length overflowed"))?;
    if requested_context > manifest.workload.context_cap_tokens {
        return Err(invalid_data("bench workload exceeds its context cap"));
    }
    if arguments
        .prefill_context
        .is_some_and(|context| context > manifest.workload.context_cap_tokens)
    {
        return Err(invalid_data(
            "prefill context exceeds the manifest context cap",
        ));
    }
    Ok(())
}

fn validate_bench_model(actual_sha256: &str, expected_sha256: &str) -> Result<(), Box<dyn Error>> {
    if actual_sha256 != expected_sha256 {
        return Err(invalid_data(format!(
            "model SHA-256 is {actual_sha256}, expected {expected_sha256}"
        ))
        .into());
    }
    Ok(())
}

fn bench_decode_execution<B: Backend>(runtime: &Runtime<B>, eager_decode: bool) -> DecodeExecution {
    if eager_decode
        || !runtime
            .model()
            .config()
            .architecture
            .decode_graph_supported()
    {
        DecodeExecution::Eager
    } else {
        DecodeExecution::Graph
    }
}

fn probe_capacity(
    runtime: &mut Runtime<CudaBackend>,
    context: usize,
    dtype: KvCacheDtype,
) -> Result<(), Box<dyn Error>> {
    let probe = runtime.probe_kv_capacity(context, dtype)?;
    println!("capacity_probe=pass");
    println!("kv_dtype={}", kv_dtype_name(probe.dtype));
    println!(
        "requested_context_tokens={}",
        probe.requested_context_tokens
    );
    println!(
        "allocated_context_tokens={}",
        probe.allocated_context_tokens
    );
    println!("kv_cache_bytes={}", probe.cache_bytes);
    Ok(())
}

struct BenchDecodeEvidence {
    prompt: Vec<u32>,
    decode_samples: Vec<f64>,
    prefill_samples: Vec<f64>,
    detokenized_bytes: usize,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
    transcript: String,
    ttft_samples_ms: Vec<f64>,
}

fn measure_decode(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: usize,
    decode_tokens: usize,
    manifest: &BenchManifest,
    arguments: &BenchArgs,
    decode_execution: DecodeExecution,
) -> Result<BenchDecodeEvidence, Box<dyn Error>> {
    let prompt = vec![1_u32; prompt_tokens];
    let mut decode_samples = Vec::with_capacity(manifest.workload.reps);
    let mut prefill_samples = Vec::with_capacity(manifest.workload.reps);
    let mut detokenized_bytes = 0_usize;
    println!("warmup");
    let warmup = runtime.benchmark_decode_with_prefill_chunk(
        &prompt,
        decode_tokens,
        arguments.kv_cache_dtype,
        decode_execution,
        arguments.prefill_chunk,
    )?;
    let method = warmup.prefill_method;
    let workspace = warmup.prefill_workspace;
    let transcript = warmup.transcript_sha256.clone();
    let mut ttft_samples_ms = Vec::with_capacity(manifest.workload.reps);
    println!("rep  prefill tok/s  decode tok/s");
    for repetition in 0..manifest.workload.reps {
        let sample = measure_decode_repetition(
            runtime,
            BenchDecodeRepetitionInput {
                prompt: &prompt,
                decode_tokens,
                arguments,
                decode_execution,
                transcript: &transcript,
                method,
                workspace,
            },
        )?;
        detokenized_bytes = detokenized_bytes
            .checked_add(sample.detokenized_bytes)
            .ok_or_else(|| invalid_data("detokenized byte count overflowed"))?;
        prefill_samples.push(sample.prefill);
        decode_samples.push(sample.decode);
        ttft_samples_ms.push(sample.ttft_ms);
        println!(
            "{:>3}  {:>13.3}  {:>12.3}",
            repetition + 1,
            sample.prefill,
            sample.decode
        );
    }
    Ok(BenchDecodeEvidence {
        prompt,
        decode_samples,
        prefill_samples,
        detokenized_bytes,
        method,
        workspace,
        transcript,
        ttft_samples_ms,
    })
}

struct BenchDecodeSample {
    prefill: f64,
    decode: f64,
    ttft_ms: f64,
    detokenized_bytes: usize,
}

struct BenchDecodeRepetitionInput<'a> {
    prompt: &'a [u32],
    decode_tokens: usize,
    arguments: &'a BenchArgs,
    decode_execution: DecodeExecution,
    transcript: &'a str,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
}

fn measure_decode_repetition(
    runtime: &mut Runtime<CudaBackend>,
    input: BenchDecodeRepetitionInput<'_>,
) -> Result<BenchDecodeSample, Box<dyn Error>> {
    let BenchDecodeRepetitionInput {
        prompt,
        decode_tokens,
        arguments,
        decode_execution,
        transcript,
        method,
        workspace,
    } = input;
    let run = runtime.benchmark_decode_with_prefill_chunk(
        prompt,
        decode_tokens,
        arguments.kv_cache_dtype,
        decode_execution,
        arguments.prefill_chunk,
    )?;
    let prefill = required_rate(run.prefill_rate(), "prefill")?;
    let decode = required_rate(run.decode_rate(), "decode")?;
    if run.transcript_sha256 != transcript {
        return Err(invalid_data(
            "decode transcript changed between benchmark repetitions, so the run is not \
             reproducible",
        )
        .into());
    }
    if run.prefill_method != method || run.prefill_workspace != workspace {
        return Err(invalid_data("prefill plan changed between benchmark repetitions").into());
    }
    Ok(BenchDecodeSample {
        prefill,
        decode,
        ttft_ms: run.ttft_duration.as_secs_f64() * 1_000.0,
        detokenized_bytes: run.detokenized_bytes,
    })
}

struct BenchTtftEvidence {
    context: usize,
    samples: Vec<f64>,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
}

fn measure_ttft(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: usize,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
    samples: Vec<f64>,
    manifest: &BenchManifest,
    arguments: &BenchArgs,
) -> Result<BenchTtftEvidence, Box<dyn Error>> {
    let Some(context) = arguments.prefill_context else {
        return Ok(BenchTtftEvidence {
            context: prompt_tokens,
            samples,
            method,
            workspace,
        });
    };
    measure_prefill_context(runtime, context, manifest.workload.reps, arguments)
}

fn measure_prefill_context(
    runtime: &mut Runtime<CudaBackend>,
    context: usize,
    repetitions: usize,
    arguments: &BenchArgs,
) -> Result<BenchTtftEvidence, Box<dyn Error>> {
    let prompt = vec![1_u32; context];
    println!("prefill warmup at {context} tokens");
    let warmup =
        runtime.benchmark_prefill(&prompt, arguments.kv_cache_dtype, arguments.prefill_chunk)?;
    let method = warmup.prefill_method;
    let workspace = warmup.prefill_workspace;
    let mut samples = Vec::with_capacity(repetitions);
    println!("rep  prompt tok/s       TTFT ms");
    for repetition in 0..repetitions {
        let run = runtime.benchmark_prefill(
            &prompt,
            arguments.kv_cache_dtype,
            arguments.prefill_chunk,
        )?;
        let rate = required_rate(run.prefill_rate(), "prefill")?;
        let ttft_ms = run.ttft_duration.as_secs_f64() * 1_000.0;
        samples.push(ttft_ms);
        if run.prefill_method != method || run.prefill_workspace != workspace {
            return Err(invalid_data("prefill plan changed between benchmark repetitions").into());
        }
        println!("{:>3}  {:>12.3}  {:>12.3}", repetition + 1, rate, ttft_ms);
    }
    Ok(BenchTtftEvidence {
        context,
        samples,
        method,
        workspace,
    })
}

fn bench_usable_bar(
    context: usize,
    prefill: &RateSummary,
    ttft: &DurationSummary,
    root: &Path,
    manifest: &BenchManifest,
    model_sha256: &str,
) -> Result<Option<UsableBar>, Box<dyn Error>> {
    if context != 2_048 {
        return Ok(None);
    }
    let comparator = load_prefill_comparator(root, manifest, model_sha256)?;
    let ratio = prefill.median / comparator.median;
    let ttft_2k_under_1s = ttft.median < 1_000.0;
    let bar_met = ttft_2k_under_1s && ratio >= 1.0 / 3.0;
    println!(
        "pp512 comparator median {:.3} tok/s, ratio {:.6}",
        comparator.median, ratio
    );
    println!(
        "usable prefill bar: {}",
        if bar_met { "met" } else { "missed" }
    );
    Ok(Some(UsableBar {
        ttft_2k_under_1s,
        pp512_ratio_vs_comparator: ratio,
        bar_met,
    }))
}

fn one_manifest_value(values: &[usize], name: &'static str) -> Result<usize, io::Error> {
    match values {
        [value] if *value > 0 => Ok(*value),
        [_] => Err(invalid_data(format!("{name} must be nonzero"))),
        _ => Err(invalid_data(format!("{name} must contain one value"))),
    }
}

fn required_rate(rate: MeasuredRate, name: &'static str) -> Result<f64, io::Error> {
    match rate {
        MeasuredRate::TokensPerSecond(value) => Ok(value),
        MeasuredRate::Unavailable => Err(invalid_data(format!("{name} rate is unavailable"))),
    }
}

#[derive(Debug)]
struct BenchPrefillEvidence {
    ttft_ms: DurationSummary,
    ttft_context_tokens: u64,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
    usable_bar: Option<UsableBar>,
}

fn load_prefill_comparator(
    root: &Path,
    manifest: &BenchManifest,
    model_sha256: &str,
) -> Result<RateSummary, Box<dyn Error>> {
    if manifest.comparator.prefill_tokens != 512 {
        return Err(invalid_data("usable prefill comparator must measure pp512").into());
    }
    let raw_path = root.join(&manifest.comparator.prefill_raw_json);
    let entries: Vec<LlamaBenchEntry> = serde_json::from_slice(&fs::read(&raw_path)?)?;
    let entry = unique_entry(
        &entries,
        |entry| entry.n_prompt == manifest.comparator.prefill_tokens as u64 && entry.n_gen == 0,
        "pp512 prefill",
    )?;
    validate_comparator_identity(root, manifest, model_sha256, entry)?;
    validate_comparator_samples(root, manifest, entry)?;
    summarize_samples(&entry.samples_ts).map_err(Into::into)
}

fn validate_comparator_identity(
    root: &Path,
    manifest: &BenchManifest,
    model_sha256: &str,
    entry: &LlamaBenchEntry,
) -> Result<(), Box<dyn Error>> {
    let pin = fs::read_to_string(root.join(&manifest.comparator.pin_file))?
        .trim()
        .to_owned();
    validate_commit(&pin, &entry.build_commit)?;
    let expected_model = PathBuf::from("models").join(&manifest.model.filename);
    if Path::new(&entry.model_filename) != expected_model {
        return Err(invalid_data(format!(
            "prefill comparator model is {}, expected {}",
            entry.model_filename,
            expected_model.display()
        ))
        .into());
    }
    let raw_model_sha256 = sha256_file(root.join(&entry.model_filename))?;
    if raw_model_sha256 != model_sha256 {
        return Err(invalid_data("prefill comparator model hash does not match").into());
    }
    Ok(())
}

fn validate_comparator_samples(
    root: &Path,
    manifest: &BenchManifest,
    entry: &LlamaBenchEntry,
) -> Result<(), Box<dyn Error>> {
    let gguf = Gguf::open(root.join(&entry.model_filename))?;
    let tensor_bytes = gguf.tensors().iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(tensor.n_bytes)
            .ok_or_else(|| invalid_data("prefill comparator tensor bytes overflowed"))
    })?;
    if entry.model_size != tensor_bytes {
        return Err(invalid_data("prefill comparator tensor bytes do not match").into());
    }
    if entry.samples_ts.len() != manifest.workload.reps {
        return Err(invalid_data("prefill comparator repetition count does not match").into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_bench_receipt<B: Backend>(
    root: &Path,
    runtime: &Runtime<B>,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    decode_tok_s: leone_receipt::RateSummary,
    prefill_tok_s: leone_receipt::RateSummary,
    prefill: BenchPrefillEvidence,
    engine_commit: &str,
    linked_quality: Option<&QualityReceipt>,
    determinism: DeterminismClaim,
) -> Result<PathBuf, Box<dyn Error>> {
    let model = runtime.model();
    let BenchReceiptMetrics {
        context_tokens,
        generated_tokens,
        reps,
        bytes_per_token_by_class,
        bytes_per_token_total,
        weights_resident_bytes_by_class,
        model_bytes_total,
        spec_roofline,
        roofline_measured_achievable,
    } = bench_receipt_metrics(runtime, manifest, arguments, decode_tok_s.median)?;
    let execution = if arguments.eager_decode {
        "eager"
    } else {
        "cuda-graph"
    };
    let kv_dtype = match arguments.kv_cache_dtype {
        KvCacheDtype::Q8 => "q8_0",
        KvCacheDtype::F16 => "f16",
        KvCacheDtype::F32 => "f32",
    };
    let receipt = RuntimeReceipt {
        schema_version: RUNTIME_SCHEMA_VERSION,
        receipt_id: Uuid::new_v4(),
        created_utc: Utc::now(),
        machine: query_machine()?,
        workload: Workload {
            engine: Engine {
                name: "leone".to_owned(),
                git_commit: engine_commit.to_owned(),
                build_flags: vec![
                    "backend=cuda".to_owned(),
                    "batch=1".to_owned(),
                    format!("decode={execution}"),
                    format!("kv={kv_dtype}"),
                    format!("prefill={}", prefill.method.name()),
                    format!("prefill_chunk={}", arguments.prefill_chunk),
                    "sampler=greedy".to_owned(),
                ],
            },
            model_artifact: ModelArtifact {
                path: PathBuf::from("models")
                    .join(&manifest.model.filename)
                    .display()
                    .to_string(),
                sha256: manifest.model.sha256.clone(),
                file_bytes: model.file_bytes(),
                format: manifest.model.format.clone(),
            },
            context_tokens,
            generated_tokens,
            batch: 1,
            reps,
        },
        results: RuntimeResults {
            decode_tok_s,
            prefill_tok_s: Some(prefill_tok_s),
            ttft_ms: Some(prefill.ttft_ms.clone()),
            ttft_context_tokens: Some(prefill.ttft_context_tokens),
            prefill_method: Some(receipt_prefill_method(prefill.method)),
            usable_bar: prefill.usable_bar,
            tokens_emitted: generated_tokens
                .checked_mul(reps)
                .ok_or_else(|| invalid_data("bench emitted token count overflowed"))?,
            bytes_per_token_by_class,
            bytes_per_token_total: Some(bytes_per_token_total),
            weights_resident_bytes_by_class: Some(weights_resident_bytes_by_class),
            model_bytes_total,
            roofline: spec_roofline,
            roofline_measured_achievable: Some(roofline_measured_achievable),
        },
        quality_ref: linked_quality.map(|quality| quality.receipt_id),
        quality_summary: linked_quality.map(quality_summary),
        determinism: Some(determinism),
        speculation: None,
        notes: bench_receipt_notes(runtime, manifest, arguments, &prefill),
    };
    write_runtime_receipt(root.join("receipts"), &receipt).map_err(Into::into)
}

struct BenchReceiptMetrics {
    context_tokens: u64,
    generated_tokens: u64,
    reps: u64,
    bytes_per_token_by_class: BTreeMap<TensorClass, u64>,
    bytes_per_token_total: u64,
    weights_resident_bytes_by_class: BTreeMap<TensorClass, u64>,
    model_bytes_total: u64,
    spec_roofline: Roofline,
    roofline_measured_achievable: Roofline,
}

fn bench_receipt_metrics<B: Backend>(
    runtime: &Runtime<B>,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    decode_median: f64,
) -> Result<BenchReceiptMetrics, Box<dyn Error>> {
    let (context_tokens, generated_tokens, reps) = bench_receipt_dimensions(manifest)?;
    let (bytes_per_token_by_class, bytes_per_token_total) =
        bench_receipt_bytes(runtime, &arguments, context_tokens, generated_tokens)?;
    let model = runtime.model();
    let weights_resident_bytes_by_class = model.weights_resident_bytes_by_class().clone();
    let model_bytes_total = sum_tensor_classes(&weights_resident_bytes_by_class)?;
    Ok(BenchReceiptMetrics {
        context_tokens,
        generated_tokens,
        reps,
        bytes_per_token_by_class,
        bytes_per_token_total,
        weights_resident_bytes_by_class,
        model_bytes_total,
        spec_roofline: roofline(SPEC_BANDWIDTH_GBS, bytes_per_token_total, decode_median),
        roofline_measured_achievable: roofline(
            MEASURED_ACHIEVABLE_BANDWIDTH_GBS,
            bytes_per_token_total,
            decode_median,
        ),
    })
}

fn bench_receipt_dimensions(manifest: &BenchManifest) -> Result<(u64, u64, u64), Box<dyn Error>> {
    let context_tokens = u64::try_from(one_manifest_value(
        &manifest.workload.prefill_tokens,
        "prefill_tokens",
    )?)?;
    let generated_tokens = u64::try_from(one_manifest_value(
        &manifest.workload.decode_tokens,
        "decode_tokens",
    )?)?;
    let reps = u64::try_from(manifest.workload.reps)?;
    Ok((context_tokens, generated_tokens, reps))
}

fn bench_receipt_bytes<B: Backend>(
    runtime: &Runtime<B>,
    arguments: &BenchArgs,
    context_tokens: u64,
    generated_tokens: u64,
) -> Result<(BTreeMap<TensorClass, u64>, u64), Box<dyn Error>> {
    let model = runtime.model();
    let mut bytes_per_token_by_class = model.decode_weight_bytes_by_class().clone();
    add_kv_bytes(
        &mut bytes_per_token_by_class,
        u64::try_from(model.config().n_layer)?,
        u64::try_from(model.config().n_head_kv)?,
        u64::try_from(model.config().head_dim)?,
        arguments.kv_cache_dtype,
        context_tokens,
        generated_tokens,
    )?;
    let bytes_per_token_total = sum_tensor_classes(&bytes_per_token_by_class)?;
    Ok((bytes_per_token_by_class, bytes_per_token_total))
}

fn bench_receipt_notes<B: Backend>(
    runtime: &Runtime<B>,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    prefill: &BenchPrefillEvidence,
) -> Vec<String> {
    let import_metrics = runtime.backend().model_import_metrics();
    with_session_note(vec![
        "The spec roofline uses 1008 GB/s for the RTX 4090.".to_owned(),
        "The measured-achievable roofline uses 930 GB/s. Luo et al., arXiv 2501.12084v2, Table 5, measure 929.8 GB/s on the RTX 4090.".to_owned(),
        "The roofline denominator counts weights read, depth-dependent KV reads, and one embedding row per decode evaluation.".to_owned(),
        "Prompt context repeats token ID 1 for the manifest token count.".to_owned(),
        "One full benchmark repetition runs before the five timed repetitions.".to_owned(),
        "Decode timing includes token D2H, stream synchronization, and allocation-free detokenization.".to_owned(),
        format!(
            "Prefill uses {} with {}-token position blocks.",
            prefill.method.name(),
            arguments.prefill_chunk,
        ),
        "Prefill GEMMs use FP16 inputs and weights, CUBLAS_COMPUTE_32F accumulation, and FP32 outputs.".to_owned(),
        format!(
            "Prefill workspace: {} bytes total, {} dequantized weights, {} converted activations, {} attention, {} cuBLASLt, and {} batch activations.",
            prefill.workspace.total_bytes,
            prefill.workspace.dequantized_weight_bytes,
            prefill.workspace.converted_activation_bytes,
            prefill.workspace.attention_bytes,
            prefill.workspace.cublaslt_bytes,
            prefill.workspace.batch_activation_bytes,
        ),
        "The eval logits path keeps sequential decode prefill and is not the measured prefill subject.".to_owned(),
        "The comparator runs with GGML_CUDA_GRAPH_OPT=1.".to_owned(),
        format!(
            "The pp512 comparator comes from {}.",
            manifest.comparator.prefill_raw_json,
        ),
        "CUDA quantized GEMV uses 32-value q8_1 activation blocks with f16 scales. Logit KLD near 1e-3 against the scalar f32 activation path is expected.".to_owned(),
        format!(
            "Lossless Q4_K repack: {:.3} ms once at load for {} source bytes.",
            import_metrics.lossless_repack_duration.as_secs_f64() * 1_000.0,
            import_metrics.lossless_repack_source_bytes,
        ),
    ])
}

fn run_generation<B: Backend>(
    backend: B,
    arguments: GenerateArgs,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let backend_name = backend.name();
    let mut runtime = Runtime::load(backend, &arguments.model)?;
    let plan =
        execution_plan::load_optional(arguments.plan.as_deref(), &arguments.model, backend_name)?;
    let options = generation_options(&runtime, &arguments, |options| {
        if let Some(plan) = plan.as_ref() {
            plan.apply(options);
        }
    })?;
    let selected_kv_cache_dtype = options.kv_cache_dtype;
    let result = stream_generation(&mut runtime, &arguments.prompt, options, &interrupted)?;
    write_debug_outputs(&arguments, &result)?;
    print_generation_footer(&result);
    print_generation_profile(&result)?;
    write_generation_output_receipt(
        &arguments,
        &runtime,
        &result,
        backend_name,
        selected_kv_cache_dtype,
    )?;
    Ok(())
}

fn generation_options<B: Backend, F: FnOnce(&mut GenerateOptions)>(
    runtime: &Runtime<B>,
    arguments: &GenerateArgs,
    apply_plan: F,
) -> Result<GenerateOptions, Box<dyn Error>> {
    let speculation = generation_speculation(runtime, arguments)?;
    let logit_capture = if arguments.debug_logits.is_some() {
        LogitCapture::Top(NonZeroUsize::new(5).expect("five is nonzero"))
    } else {
        LogitCapture::Disabled
    };
    let mut options = GenerateOptions {
        max_tokens: arguments.tokens,
        prefill_chunk_tokens: DEFAULT_PREFILL_CHUNK_TOKENS,
        logit_capture,
        decode_profile: if arguments.profile_decode {
            DecodeProfileMode::Steps(NonZeroUsize::new(64).expect("64 is nonzero"))
        } else {
            DecodeProfileMode::Disabled
        },
        decode_execution: if arguments.eager_decode
            || !runtime
                .model()
                .config()
                .architecture
                .decode_graph_supported()
        {
            DecodeExecution::Eager
        } else {
            DecodeExecution::Graph
        },
        kv_cache_dtype: arguments.kv_cache_dtype,
        sampler: arguments.sampler.clone(),
        penalties: arguments.penalties.clone(),
        seed: arguments.seed,
        speculation,
        mirostat: None,
        output_constraint: None,
    };
    apply_plan(&mut options);
    Ok(options)
}

fn generation_speculation<B: Backend>(
    runtime: &Runtime<B>,
    arguments: &GenerateArgs,
) -> Result<Speculation, Box<dyn Error>> {
    if !arguments.correctable {
        return Ok(arguments.speculation);
    }
    Ok(Speculation::Correctable(CorrectableDrafter::new(
        runtime.vocab_size(),
        NonZeroUsize::new(7).expect("seven is nonzero"),
    )?))
}

fn stream_generation<B: Backend>(
    runtime: &mut Runtime<B>,
    prompt: &str,
    options: GenerateOptions,
    interrupted: &Arc<AtomicBool>,
) -> Result<GenerationResult, Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    let result = runtime.generate(
        prompt,
        options,
        |token| {
            stdout
                .write_all(&token.bytes)
                .and_then(|()| stdout.flush())
                .map_err(RuntimeError::token_callback)
        },
        || interrupted.load(Ordering::Relaxed),
    )?;
    drop(stdout);
    Ok(result)
}

fn write_debug_outputs(
    arguments: &GenerateArgs,
    result: &GenerationResult,
) -> Result<(), Box<dyn Error>> {
    if let Some(path) = &arguments.debug_tokens {
        let mut bytes = serde_json::to_vec(&result.tokens)?;
        bytes.push(b'\n');
        fs::write(path, bytes)?;
    }
    if let Some(path) = &arguments.debug_logits {
        write_logits(path, result)?;
    }
    Ok(())
}

fn print_generation_profile(result: &GenerationResult) -> Result<(), Box<dyn Error>> {
    if let Some(profile) = &result.decode_profile {
        print_decode_profile(profile)?;
    }
    Ok(())
}

fn write_generation_output_receipt<B: Backend>(
    arguments: &GenerateArgs,
    runtime: &Runtime<B>,
    result: &GenerationResult,
    backend_name: &str,
    kv_cache_dtype: KvCacheDtype,
) -> Result<(), Box<dyn Error>> {
    if arguments.receipt {
        write_generation_receipt(runtime, result, backend_name, kv_cache_dtype)?;
    }
    Ok(())
}

fn print_decode_profile(profile: &leone::DecodeProfile) -> Result<(), Box<dyn Error>> {
    let steps = profile.steps as f64;
    let gpu = profile.gpu_duration();
    eprintln!();
    eprintln!("decode profile: {} evaluations", profile.steps);
    eprintln!(
        "  {:<16} {:>12} {:>14} {:>9}",
        "operation", "total ms", "us/token", "GPU share"
    );
    for operation in DecodeOp::all() {
        let duration = profile
            .gpu_duration_by_op
            .get(&operation)
            .copied()
            .unwrap_or_default();
        let share = if gpu.is_zero() {
            0.0
        } else {
            duration.as_secs_f64() / gpu.as_secs_f64() * 100.0
        };
        eprintln!(
            "  {:<16} {:>12.3} {:>14.3} {:>8.2}%",
            operation.name(),
            duration.as_secs_f64() * 1_000.0,
            duration.as_secs_f64() * 1_000_000.0 / steps,
            share,
        );
    }
    let host_gap = profile.host_gap();
    eprintln!(
        "  {:<16} {:>12.3} {:>14.3} {:>9}",
        "host gap",
        host_gap.as_secs_f64() * 1_000.0,
        host_gap.as_secs_f64() * 1_000_000.0 / steps,
        "",
    );
    eprintln!(
        "  {:<16} {:>12.3} {:>14.3}",
        "wall",
        profile.wall_duration.as_secs_f64() * 1_000.0,
        profile.wall_duration.as_secs_f64() * 1_000_000.0 / steps,
    );
    eprintln!();
    eprintln!("GEMV weight traffic");
    eprintln!(
        "  {:>6} {:>9} {:>9} {:>7} {:>9} {:>12}",
        "format", "rows", "columns", "calls", "total ms", "weight GB/s"
    );
    for (shape, timing) in &profile.gemv_by_shape {
        let bytes = shape
            .layout()?
            .bytes()
            .checked_mul(timing.calls)
            .ok_or_else(|| invalid_data("profile GEMV bytes overflowed"))?;
        let bandwidth = bytes as f64 / timing.gpu_duration.as_secs_f64() / 1e9;
        eprintln!(
            "  {:>6?} {:>9} {:>9} {:>7} {:>9.3} {:>12.1}",
            shape.format(),
            shape.rows(),
            shape.columns(),
            timing.calls,
            timing.gpu_duration.as_secs_f64() * 1_000.0,
            bandwidth,
        );
    }
    eprintln!();
    eprintln!(
        "kernel launches/token: {:.3}",
        profile.kernel_launches as f64 / steps
    );
    eprintln!("H2D copies/token: {:.3}", profile.h2d_copies as f64 / steps);
    eprintln!("D2H copies/token: {:.3}", profile.d2h_copies as f64 / steps);
    eprintln!(
        "stream synchronizations/token: {:.3}",
        profile.stream_synchronizations as f64 / steps
    );
    Ok(())
}

fn write_logits(path: &Path, result: &GenerationResult) -> Result<(), io::Error> {
    let mut output = String::new();
    for snapshot in &result.logits {
        write!(
            output,
            "position={} sampled={} top=",
            snapshot.input_position, snapshot.sampled_token
        )
        .map_err(|_| invalid_data("failed to format logit output"))?;
        for (index, logit) in snapshot.top.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(
                output,
                "{}:{:.9}:0x{:08x}",
                logit.token,
                logit.value,
                logit.value.to_bits()
            )
            .map_err(|_| invalid_data("failed to format logit output"))?;
        }
        output.push('\n');
    }
    fs::write(path, output)
}

fn print_generation_footer(result: &GenerationResult) {
    eprintln!();
    print_rate("decode", result.stats.decode_rate());
    print_rate("prefill", result.stats.prefill_rate());
    eprintln!("tokens emitted: {}", result.stats.emitted_tokens);
    let speculation = result.stats.speculation;
    if speculation.proposed > 0 {
        let accepted = speculation.accepted as f64 / speculation.proposed as f64;
        let decoded = result.stats.emitted_tokens.saturating_sub(1);
        let per_pass = decoded as f64 / result.stats.decode_evaluations as f64;
        eprintln!(
            "speculation: {} of {} proposals accepted over {} rounds ({:.3} accepted, \
             {:.3} tokens per forward pass)",
            speculation.accepted, speculation.proposed, speculation.rounds, accepted, per_pass
        );
        if speculation.verify_passes > 0 {
            eprintln!(
                "verifier: {} positions over {} passes in {:.3} ms",
                speculation.verified_positions,
                speculation.verify_passes,
                result.stats.verify_duration.as_secs_f64() * 1_000.0,
            );
        }
    }
    if speculation.adaptive_plain_rounds != 0 || speculation.adaptive_speculative_rounds != 0 {
        eprintln!(
            "adaptive: {} plain rounds, {} speculative rounds, {} below gate, {} regret-limited, \
             {} suffix proposals, {} recycling proposals, {:.3} ms controller",
            speculation.adaptive_plain_rounds,
            speculation.adaptive_speculative_rounds,
            speculation.adaptive_below_gate_rounds,
            speculation.adaptive_regret_limited_rounds,
            speculation.adaptive_suffix_proposals,
            speculation.adaptive_recycling_proposals,
            speculation.adaptive_controller_duration.as_secs_f64() * 1_000.0,
        );
        eprintln!("adaptive widths: {:?}", speculation.adaptive_width_rounds);
    }
    if speculation.correctable_plain_rounds != 0 || speculation.correctable_speculative_rounds != 0
    {
        let overlap = if speculation.correctable_overlap_proposals == 0 {
            0.0
        } else {
            speculation.correctable_overlap_sum / speculation.correctable_overlap_proposals as f64
        };
        eprintln!(
            "correctable: {} plain rounds, {} speculative rounds, {} below gate, {} regret-limited, \
             {:.6} mean target overlap, {:.3} ms controller",
            speculation.correctable_plain_rounds,
            speculation.correctable_speculative_rounds,
            speculation.correctable_below_gate_rounds,
            speculation.correctable_regret_limited_rounds,
            overlap,
            speculation.correctable_controller_duration.as_secs_f64() * 1_000.0,
        );
        eprintln!(
            "correctable plans [suffix, unigram, bigram, fourgram]: {:?}",
            speculation.correctable_plan_rounds
        );
        eprintln!(
            "correctable widths: {:?}",
            speculation.correctable_width_rounds
        );
    }
    eprintln!("quality: unverified");
}

/// Returns the speculation record, or `None` when drafting was off.
fn speculation_record(result: &GenerationResult) -> Option<SpeculationRecord> {
    let speculation = result.stats.speculation;
    if speculation.rounds == 0
        && speculation.adaptive_plain_rounds == 0
        && speculation.correctable_plain_rounds == 0
    {
        return None;
    }
    let adaptive =
        speculation.adaptive_plain_rounds != 0 || speculation.adaptive_speculative_rounds != 0;
    let correctable = speculation.correctable_plain_rounds != 0
        || speculation.correctable_speculative_rounds != 0;
    Some(SpeculationRecord {
        drafter: if correctable {
            "correctable-history".to_owned()
        } else if adaptive {
            "adaptive-history".to_owned()
        } else {
            "suffix".to_owned()
        },
        rounds: speculation.rounds as u64,
        proposed: speculation.proposed as u64,
        accepted: speculation.accepted as u64,
        evaluations: result.stats.decode_evaluations as u64,
        draft_width: speculation.draft_width as u64,
        verify_passes: speculation.verify_passes as u64,
        verified_positions: speculation.verified_positions as u64,
        verify_duration_ms: result.stats.verify_duration.as_secs_f64() * 1_000.0,
        adaptive_plain_rounds: speculation.adaptive_plain_rounds,
        adaptive_speculative_rounds: speculation.adaptive_speculative_rounds,
        adaptive_below_gate_rounds: speculation.adaptive_below_gate_rounds,
        adaptive_regret_limited_rounds: speculation.adaptive_regret_limited_rounds,
        adaptive_suffix_proposals: speculation.adaptive_suffix_proposals,
        adaptive_recycling_proposals: speculation.adaptive_recycling_proposals,
        adaptive_controller_duration_ms: speculation.adaptive_controller_duration.as_secs_f64()
            * 1_000.0,
        adaptive_width_rounds: speculation.adaptive_width_rounds,
        correctable_plain_rounds: speculation.correctable_plain_rounds,
        correctable_speculative_rounds: speculation.correctable_speculative_rounds,
        correctable_below_gate_rounds: speculation.correctable_below_gate_rounds,
        correctable_regret_limited_rounds: speculation.correctable_regret_limited_rounds,
        correctable_plan_rounds: speculation.correctable_plan_rounds,
        correctable_width_rounds: speculation.correctable_width_rounds,
        correctable_controller_duration_ms: speculation
            .correctable_controller_duration
            .as_secs_f64()
            * 1_000.0,
        correctable_overlap_sum: speculation.correctable_overlap_sum,
        correctable_overlap_proposals: speculation.correctable_overlap_proposals as u64,
    })
}

fn print_rate(label: &str, rate: MeasuredRate) {
    match rate {
        MeasuredRate::TokensPerSecond(value) => eprintln!("{label}: {value:.3} tok/s"),
        MeasuredRate::Unavailable => eprintln!("{label}: unavailable"),
    }
}

fn write_generation_receipt<B: Backend>(
    runtime: &Runtime<B>,
    result: &GenerationResult,
    backend_name: &str,
    kv_cache_dtype: KvCacheDtype,
) -> Result<(), Box<dyn Error>> {
    let rates = generation_receipt_rates(result)?;
    let model = generation_receipt_model(runtime)?;
    let metrics = generation_receipt_metrics(runtime, result, kv_cache_dtype, rates.decode_rate)?;
    let results = generation_receipt_results(result, &rates, &metrics)?;
    let determinism = generation_receipt_determinism(runtime, result);
    let root = std::env::current_dir()?;
    let receipt = RuntimeReceipt {
        schema_version: RUNTIME_SCHEMA_VERSION,
        receipt_id: Uuid::new_v4(),
        created_utc: Utc::now(),
        machine: query_machine()?,
        workload: Workload {
            engine: generation_receipt_engine(result, backend_name)?,
            model_artifact: ModelArtifact {
                path: model.path,
                sha256: model.sha256,
                file_bytes: model.file_bytes,
                format: model.format,
            },
            context_tokens: metrics.context_tokens,
            generated_tokens: metrics.generated_tokens,
            batch: 1,
            reps: 1,
        },
        results,
        quality_ref: None,
        quality_summary: None,
        determinism: Some(determinism),
        speculation: speculation_record(result),
        notes: vec![
            "The spec roofline uses 1008 GB/s for the RTX 4090.".to_owned(),
            "The measured-achievable roofline uses 930 GB/s. Luo et al., arXiv 2501.12084v2, Table 5, measure 929.8 GB/s on the RTX 4090.".to_owned(),
            format!("Prefill uses {}.", result.stats.prefill_method.name()),
            format!(
                "Prefill workspace: {} bytes total, including {} bytes of batch activations.",
                result.stats.prefill_workspace.total_bytes,
                result.stats.prefill_workspace.batch_activation_bytes,
            ),
            "The comparator runs with GGML_CUDA_GRAPH_OPT=1.".to_owned(),
        ],
    };
    let path = write_runtime_receipt(root.join("receipts"), &receipt)?;
    eprintln!("receipt: {}", path.display());
    Ok(())
}

struct GenerationReceiptRates {
    decode_rate: f64,
    prefill_rate: f64,
}

fn generation_receipt_rates(
    result: &GenerationResult,
) -> Result<GenerationReceiptRates, Box<dyn Error>> {
    let decode_rate = match result.stats.decode_rate() {
        MeasuredRate::TokensPerSecond(value) => value,
        MeasuredRate::Unavailable => {
            return Err(invalid_data("a receipt needs at least one decode evaluation").into());
        }
    };
    let prefill_rate = match result.stats.prefill_rate() {
        MeasuredRate::TokensPerSecond(value) => value,
        MeasuredRate::Unavailable => {
            return Err(invalid_data("a receipt needs measured prefill work").into());
        }
    };
    Ok(GenerationReceiptRates {
        decode_rate,
        prefill_rate,
    })
}

struct GenerationReceiptModel {
    path: String,
    sha256: String,
    file_bytes: u64,
    format: String,
}

fn generation_receipt_model<B: Backend>(
    runtime: &Runtime<B>,
) -> Result<GenerationReceiptModel, Box<dyn Error>> {
    let model = runtime.model();
    Ok(GenerationReceiptModel {
        path: model.path().display().to_string(),
        sha256: sha256_file(model.path())?,
        file_bytes: model.file_bytes(),
        format: model_format(&Gguf::open(model.path())?)?,
    })
}

struct GenerationReceiptMetrics {
    context_tokens: u64,
    generated_tokens: u64,
    bytes_per_token_by_class: BTreeMap<TensorClass, u64>,
    bytes_per_token_total: u64,
    weights_resident_bytes_by_class: BTreeMap<TensorClass, u64>,
    model_bytes_total: u64,
    spec_roofline: Roofline,
    achievable_roofline: Roofline,
}

fn generation_receipt_metrics<B: Backend>(
    runtime: &Runtime<B>,
    result: &GenerationResult,
    kv_cache_dtype: KvCacheDtype,
    decode_rate: f64,
) -> Result<GenerationReceiptMetrics, Box<dyn Error>> {
    let model = runtime.model();
    let context_tokens = u64::try_from(result.stats.prompt_tokens)?;
    let generated_tokens = u64::try_from(result.stats.emitted_tokens)?;
    let decode_evaluations = u64::try_from(result.stats.decode_evaluations)?;
    let mut bytes_per_token_by_class = model.decode_weight_bytes_by_class().clone();
    add_kv_bytes(
        &mut bytes_per_token_by_class,
        u64::try_from(model.config().n_layer)?,
        u64::try_from(model.config().n_head_kv)?,
        u64::try_from(model.config().head_dim)?,
        kv_cache_dtype,
        context_tokens,
        decode_evaluations,
    )?;
    let bytes_per_token_total = sum_tensor_classes(&bytes_per_token_by_class)?;
    let weights_resident_bytes_by_class = model.weights_resident_bytes_by_class().clone();
    let model_bytes_total = sum_tensor_classes(&weights_resident_bytes_by_class)?;
    Ok(GenerationReceiptMetrics {
        context_tokens,
        generated_tokens,
        bytes_per_token_by_class,
        bytes_per_token_total,
        weights_resident_bytes_by_class,
        model_bytes_total,
        spec_roofline: roofline(SPEC_BANDWIDTH_GBS, bytes_per_token_total, decode_rate),
        achievable_roofline: roofline(
            MEASURED_ACHIEVABLE_BANDWIDTH_GBS,
            bytes_per_token_total,
            decode_rate,
        ),
    })
}

fn generation_receipt_results(
    result: &GenerationResult,
    rates: &GenerationReceiptRates,
    metrics: &GenerationReceiptMetrics,
) -> Result<RuntimeResults, Box<dyn Error>> {
    Ok(RuntimeResults {
        decode_tok_s: summarize_samples(&[rates.decode_rate])?,
        prefill_tok_s: Some(summarize_samples(&[rates.prefill_rate])?),
        ttft_ms: result
            .stats
            .ttft_duration
            .map(|duration| summarize_duration_samples_ms(&[duration.as_secs_f64() * 1_000.0]))
            .transpose()?,
        ttft_context_tokens: result.stats.ttft_duration.map(|_| metrics.context_tokens),
        prefill_method: Some(receipt_prefill_method(result.stats.prefill_method)),
        usable_bar: None,
        tokens_emitted: metrics.generated_tokens,
        bytes_per_token_by_class: metrics.bytes_per_token_by_class.clone(),
        bytes_per_token_total: Some(metrics.bytes_per_token_total),
        weights_resident_bytes_by_class: Some(metrics.weights_resident_bytes_by_class.clone()),
        model_bytes_total: metrics.model_bytes_total,
        roofline: metrics.spec_roofline.clone(),
        roofline_measured_achievable: Some(metrics.achievable_roofline.clone()),
    })
}

fn generation_receipt_engine(
    result: &GenerationResult,
    backend_name: &str,
) -> Result<Engine, io::Error> {
    Ok(Engine {
        name: "leone".to_owned(),
        git_commit: command_output("git", &["rev-parse", "HEAD"])?
            .trim()
            .to_owned(),
        build_flags: vec![
            format!("backend={backend_name}"),
            "batch=1".to_owned(),
            format!("prefill={}", result.stats.prefill_method.name()),
            "sampler=greedy".to_owned(),
        ],
    })
}

fn generation_receipt_determinism<B: Backend>(
    runtime: &Runtime<B>,
    result: &GenerationResult,
) -> DeterminismClaim {
    DeterminismClaim::Reproduced {
        order: reduction_order(runtime.backend().determinism()),
        sampler: SamplerRecord::Greedy,
        prompt_sha256: token_stream_sha256(&result.prompt_tokens),
        transcript_sha256: token_stream_sha256(&result.tokens),
        identical_reps: 1,
    }
}

fn convert_llama_bench(json_path: &Path, quality_ref: Option<Uuid>) -> Result<(), Box<dyn Error>> {
    let LlamaBenchSource {
        root,
        summary,
        pin,
        file_bytes,
        model_sha256,
        linked_quality,
    } = load_llama_bench_source(json_path, quality_ref)?;
    let LlamaBenchRates {
        decode_tok_s,
        prefill_tok_s,
        ttft_ms,
        ttft_context_tokens,
        prefill_method,
        reps,
    } = llama_bench_rates(&summary)?;
    let LlamaBenchModelData {
        artifact_format,
        weights_resident_bytes_by_class,
        model_bytes_total,
        bytes_per_token_by_class,
        bytes_per_token_total,
        spec_roofline,
        achievable_roofline,
    } = llama_bench_model_data(&root, &summary, &model_sha256, decode_tok_s.median)?;
    let notes = llama_bench_notes(&summary, prefill_tok_s.is_some());

    let receipt = RuntimeReceipt {
        schema_version: RUNTIME_SCHEMA_VERSION,
        receipt_id: Uuid::new_v4(),
        created_utc: Utc::now(),
        machine: query_machine()?,
        workload: Workload {
            engine: Engine {
                name: "llama.cpp".to_owned(),
                git_commit: pin,
                build_flags: vec![
                    "GGML_CUDA=ON".to_owned(),
                    "GGML_CCACHE=OFF".to_owned(),
                    "CMAKE_CUDA_ARCHITECTURES=89".to_owned(),
                    "LLAMA_CURL=OFF".to_owned(),
                    "LLAMA_BUILD_UI=OFF".to_owned(),
                    "LLAMA_USE_PREBUILT_UI=OFF".to_owned(),
                    "GGML_CUDA_GRAPH_OPT=1".to_owned(),
                ],
            },
            model_artifact: ModelArtifact {
                path: summary.model_filename,
                sha256: model_sha256,
                file_bytes,
                format: artifact_format,
            },
            context_tokens: summary.context_tokens,
            generated_tokens: summary.generated_tokens,
            batch: 1,
            reps,
        },
        results: RuntimeResults {
            decode_tok_s,
            prefill_tok_s,
            ttft_ms,
            ttft_context_tokens,
            prefill_method,
            usable_bar: None,
            tokens_emitted: summary.generated_tokens * reps,
            bytes_per_token_by_class,
            bytes_per_token_total: Some(bytes_per_token_total),
            weights_resident_bytes_by_class: Some(weights_resident_bytes_by_class),
            model_bytes_total,
            roofline: spec_roofline,
            roofline_measured_achievable: Some(achievable_roofline),
        },
        quality_ref: linked_quality.as_ref().map(|quality| quality.receipt_id),
        quality_summary: linked_quality.as_ref().map(quality_summary),
        determinism: Some(DeterminismClaim::NotMeasured {
            reason: "llama-bench does not report a token stream".to_owned(),
        }),
        speculation: None,
        notes: with_session_note(notes),
    };
    let path = write_runtime_receipt(root.join("receipts"), &receipt)?;
    println!("receipt: {}", path.display());
    println!("{receipt}");
    Ok(())
}

struct LlamaBenchSource {
    root: PathBuf,
    summary: BenchmarkSummary,
    pin: String,
    file_bytes: u64,
    model_sha256: String,
    linked_quality: Option<QualityReceipt>,
}

fn load_llama_bench_source(
    json_path: &Path,
    quality_ref: Option<Uuid>,
) -> Result<LlamaBenchSource, Box<dyn Error>> {
    let entries: Vec<LlamaBenchEntry> = serde_json::from_slice(&fs::read(json_path)?)?;
    let summary = BenchmarkSummary::from_entries(&entries)?;
    let root = std::env::current_dir()?;
    let pin = fs::read_to_string(root.join("external/PINNED"))?
        .trim()
        .to_owned();
    validate_commit(&pin, &summary.build_commit)?;
    let model_path = PathBuf::from(&summary.model_filename);
    let file_bytes = fs::metadata(&model_path)?.len();
    let model_sha256 = sha256_file(&model_path)?;
    let linked_quality = load_linked_quality(&root, quality_ref, "llama.cpp", &pin, &model_sha256)?;
    Ok(LlamaBenchSource {
        root,
        summary,
        pin,
        file_bytes,
        model_sha256,
        linked_quality,
    })
}

struct LlamaBenchRates {
    decode_tok_s: RateSummary,
    prefill_tok_s: Option<RateSummary>,
    ttft_ms: Option<DurationSummary>,
    ttft_context_tokens: Option<u64>,
    prefill_method: Option<ReceiptPrefillMethod>,
    reps: u64,
}

fn llama_bench_rates(summary: &BenchmarkSummary) -> Result<LlamaBenchRates, Box<dyn Error>> {
    let decode_tok_s = summarize_samples(&summary.decode_samples)?;
    let prefill_tok_s = summary
        .prefill_samples
        .as_deref()
        .map(summarize_samples)
        .transpose()?;
    let ttft_samples_ms = summary.prefill_samples.as_ref().map(|samples| {
        samples
            .iter()
            .map(|rate| summary.context_tokens as f64 / rate * 1_000.0)
            .collect::<Vec<_>>()
    });
    let ttft_ms = ttft_samples_ms
        .as_deref()
        .map(summarize_duration_samples_ms)
        .transpose()?;
    let prefill_method = prefill_tok_s
        .as_ref()
        .map(|_| ReceiptPrefillMethod::SequentialDecode);
    Ok(LlamaBenchRates {
        decode_tok_s,
        prefill_tok_s,
        ttft_context_tokens: ttft_ms.as_ref().map(|_| summary.context_tokens),
        ttft_ms,
        prefill_method,
        reps: u64::try_from(summary.decode_samples.len())?,
    })
}

struct LlamaBenchModelData {
    artifact_format: String,
    weights_resident_bytes_by_class: BTreeMap<TensorClass, u64>,
    model_bytes_total: u64,
    bytes_per_token_by_class: BTreeMap<TensorClass, u64>,
    bytes_per_token_total: u64,
    spec_roofline: Roofline,
    achievable_roofline: Roofline,
}

fn llama_bench_model_data(
    root: &Path,
    summary: &BenchmarkSummary,
    model_sha256: &str,
    decode_median: f64,
) -> Result<LlamaBenchModelData, Box<dyn Error>> {
    let model_path = PathBuf::from(&summary.model_filename);
    let gguf = Gguf::open(&model_path)?;
    let model = ModelConfig::from_metadata(gguf.metadata())?;
    let artifact_format = declared_model_format(root, model_sha256, &gguf)?;
    let weights_resident_bytes_by_class = tensor_class_bytes(&gguf)?;
    let model_bytes_total = sum_tensor_classes(&weights_resident_bytes_by_class)?;
    let mut bytes_per_token_by_class =
        decode_weight_bytes(&gguf, &weights_resident_bytes_by_class, &model)?;
    add_kv_bytes(
        &mut bytes_per_token_by_class,
        model.n_layer,
        model.n_head_kv,
        model.head_dim.unwrap_or(model.n_embd / model.n_head),
        KvCacheDtype::F16,
        summary.context_tokens,
        summary.generated_tokens,
    )?;
    let bytes_per_token_total = sum_tensor_classes(&bytes_per_token_by_class)?;
    Ok(LlamaBenchModelData {
        artifact_format,
        weights_resident_bytes_by_class,
        model_bytes_total,
        bytes_per_token_by_class,
        bytes_per_token_total,
        spec_roofline: roofline(SPEC_BANDWIDTH_GBS, bytes_per_token_total, decode_median),
        achievable_roofline: roofline(
            MEASURED_ACHIEVABLE_BANDWIDTH_GBS,
            bytes_per_token_total,
            decode_median,
        ),
    })
}

fn llama_bench_notes(summary: &BenchmarkSummary, has_prefill: bool) -> Vec<String> {
    let mut notes = vec![
        "The spec roofline uses 1008 GB/s for the RTX 4090.".to_owned(),
        "The measured-achievable roofline uses 930 GB/s. Luo et al., arXiv 2501.12084v2, Table 5, measure 929.8 GB/s on the RTX 4090.".to_owned(),
        "The roofline denominator counts weights read, depth-dependent KV reads, and one embedding row per decode evaluation.".to_owned(),
        "The comparator runs with GGML_CUDA_GRAPH_OPT=1.".to_owned(),
        format!("Decode starts with {} KV cache entries.", summary.context_tokens),
    ];
    if has_prefill {
        notes.push("Prefill: sequential batch-1 prefill, unoptimized.".to_owned());
    }
    notes
}

fn with_session_note(mut notes: Vec<String>) -> Vec<String> {
    if let Ok(note) = std::env::var("LEONE_RECEIPT_SESSION_NOTE") {
        let note = note.trim();
        if !note.is_empty() {
            notes.push(note.to_owned());
        }
    }
    notes
}

fn parse_quality_ref(arguments: &[String]) -> Result<Option<Uuid>, io::Error> {
    match arguments {
        [] => Ok(None),
        [flag, value] if flag == "--quality-ref" => Uuid::parse_str(value)
            .map(Some)
            .map_err(|_| invalid_data(format!("quality receipt UUID is invalid: {value}"))),
        _ => Err(invalid_data(
            "receipt from-llama-bench accepts only --quality-ref <uuid>",
        )),
    }
}

fn load_linked_quality(
    root: &Path,
    quality_ref: Option<Uuid>,
    engine: &str,
    engine_commit: &str,
    model_sha256: &str,
) -> Result<Option<QualityReceipt>, Box<dyn Error>> {
    let Some(receipt_id) = quality_ref else {
        return Ok(None);
    };
    for entry in fs::read_dir(root.join("receipts"))? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !is_quality_receipt_path(&path, name) {
            continue;
        }
        let quality = QualityReceipt::from_json(&fs::read(&path)?)?;
        if quality.receipt_id != receipt_id {
            continue;
        }
        validate_quality_subject(&quality, engine, engine_commit, model_sha256)?;
        return Ok(Some(quality));
    }
    Err(invalid_data(format!("quality receipt {receipt_id} was not found")).into())
}

fn is_quality_receipt_path(path: &Path, name: &str) -> bool {
    name.contains("-quality-") && path.extension().and_then(|value| value.to_str()) == Some("json")
}

fn validate_quality_subject(
    quality: &QualityReceipt,
    engine: &str,
    engine_commit: &str,
    model_sha256: &str,
) -> Result<(), Box<dyn Error>> {
    if quality.subject.engine.name != engine {
        return Err(invalid_data(format!(
            "quality subject engine is {}, expected {engine}",
            quality.subject.engine.name
        ))
        .into());
    }
    if quality.subject.engine.git_commit != engine_commit {
        return Err(invalid_data(format!(
            "quality subject commit is {}, expected {engine_commit}",
            quality.subject.engine.git_commit
        ))
        .into());
    }
    if quality.subject.model_artifact.sha256 != model_sha256 {
        return Err(invalid_data(format!(
            "quality subject model SHA-256 is {}, expected {model_sha256}",
            quality.subject.model_artifact.sha256
        ))
        .into());
    }
    Ok(())
}

fn quality_summary(quality: &QualityReceipt) -> QualitySummary {
    let metrics = quality
        .metrics
        .as_ref()
        .expect("linked runtime quality receipt carries KLD metrics");
    QualitySummary {
        kld_mean: metrics.kld.mean,
        kld_p99: metrics.kld.p99,
        top1_agreement: metrics.top1_agreement,
    }
}

fn print_quality(quality: Option<&QualityReceipt>) {
    match quality {
        Some(quality) => {
            let metrics = quality
                .metrics
                .as_ref()
                .expect("linked runtime quality receipt carries KLD metrics");
            println!(
                "quality: KLD mean {:.6}, p99 {:.6}, top1 {:.6} (receipt {})",
                metrics.kld.mean, metrics.kld.p99, metrics.top1_agreement, quality.receipt_id,
            );
        }
        None => println!("quality: unverified"),
    }
}

fn inspect_gguf(path: &Path) -> Result<(), Box<dyn Error>> {
    let gguf = Gguf::open(path)?;
    let model = ModelConfig::from_metadata(gguf.metadata())?;
    let by_dtype = tensor_dtype_bytes(&gguf)?;
    let by_class = tensor_class_bytes(&gguf)?;
    print_gguf_summary(path, &gguf, &model);
    print_tensor_storage(by_dtype);
    print_tensor_classes(&by_class);
    Ok(())
}

fn tensor_dtype_bytes(gguf: &Gguf) -> Result<BTreeMap<GgmlType, (u64, u64)>, io::Error> {
    let mut by_dtype: BTreeMap<GgmlType, (u64, u64)> = BTreeMap::new();
    for tensor in gguf.tensors() {
        let entry = by_dtype.entry(tensor.dtype).or_default();
        entry.0 = entry
            .0
            .checked_add(1)
            .ok_or_else(|| invalid_data("dtype tensor count overflowed"))?;
        entry.1 = entry
            .1
            .checked_add(tensor.n_bytes)
            .ok_or_else(|| invalid_data("dtype byte total overflowed"))?;
    }
    Ok(by_dtype)
}

fn print_gguf_summary(path: &Path, gguf: &Gguf, model: &ModelConfig) {
    println!("GGUF summary");
    println!("  file                 {}", path.display());
    println!("  version              {}", gguf.version());
    println!("  alignment            {}", gguf.alignment());
    println!("  metadata entries     {}", gguf.metadata().len());
    println!("  tensors              {}", gguf.tensors().len());
    println!("  architecture         {}", model.architecture);
    let architecture = leone_gguf::architecture_support(&model.architecture);
    println!("  architecture class   {}", architecture.class.name());
    println!("  metadata probe       {}", architecture.metadata_probe);
    println!("  text runtime         {}", architecture.text_runtime);
    println!("  vision runtime       {}", architecture.vision_runtime);
    println!("  layers               {}", model.n_layer);
    println!("  embedding length     {}", model.n_embd);
    println!("  feed-forward length  {}", model.n_ff);
    println!("  attention heads      {}", model.n_head);
    println!("  KV heads             {}", model.n_head_kv);
    println!("  context length       {}", model.context_length);
    println!("  vocabulary size      {}", model.vocab_size);
    println!("  tokenizer            {}", model.tokenizer.model);
    println!("  RoPE theta           {}", model.rope_theta);
    print_rope_fields(model);
}

fn print_rope_fields(model: &ModelConfig) {
    match &model.rope_scaling {
        Some(scaling) => println!(
            "  RoPE scaling         {}",
            scaling.kind.as_deref().unwrap_or("fields present")
        ),
        None => println!("  RoPE scaling         not present"),
    }
    println!("  RMS epsilon          {}", model.rms_eps);
    match model.head_dim {
        Some(head_dim) => println!("  head dimension       {head_dim}"),
        None => println!("  head dimension       not present"),
    }
}

fn print_tensor_storage(by_dtype: BTreeMap<GgmlType, (u64, u64)>) {
    println!();
    println!("Tensor storage by dtype");
    println!(
        "  {:<10} {:>8} {:>16} {:>8} {:>8} {:>8}",
        "dtype", "tensors", "bytes", "scalar", "cpu", "cuda"
    );
    for (dtype, (count, bytes)) in by_dtype {
        let support = leone_gguf::quantization_support(dtype);
        println!(
            "  {:<10} {:>8} {:>16} {:>8} {:>8} {:>8}",
            dtype,
            count,
            bytes,
            support.is_some_and(|value| value.scalar_oracle),
            support.is_some_and(|value| value.cpu_runtime),
            support.is_some_and(|value| value.cuda_runtime),
        );
    }
}

fn print_tensor_classes(by_class: &BTreeMap<TensorClass, u64>) {
    println!();
    println!("weights_resident_bytes_by_class");
    println!("  {:<10} {:>16}", "class", "bytes");
    for class in TensorClass::all() {
        println!("  {:<10} {:>16}", class.name(), by_class[&class]);
    }
}

/// Names an artifact the way the benchmark manifest declares it.
///
/// The manifest binds a format string to a pinned SHA-256. Matching
/// artifacts reuse that name so receipts for the same file spell it the
/// same way. Any other artifact falls back to `model_format`.
fn declared_model_format(
    root: &Path,
    model_sha256: &str,
    gguf: &Gguf,
) -> Result<String, Box<dyn Error>> {
    let path = root.join("benchmarks/manifest.toml");
    if let Ok(text) = fs::read_to_string(&path) {
        let manifest: BenchManifest = toml::from_str(&text)?;
        if manifest.model.sha256 == model_sha256 {
            return Ok(manifest.model.format);
        }
    }
    Ok(model_format(gguf)?)
}

/// Maps a backend's declared reduction order onto its receipt spelling.
const fn reduction_order(determinism: Determinism) -> ReductionOrder {
    match determinism {
        Determinism::FixedOrder => ReductionOrder::FixedOrder,
    }
}

/// Names an artifact by the dtype that holds the most tensor bytes.
///
/// A GGUF mixes dtypes, so this reports the dominant one, not the mix a
/// filename claims. `general.file_type` is absent from some published
/// artifacts. Ties resolve to the higher dtype code, so the label is
/// stable for one file.
fn model_format(gguf: &Gguf) -> Result<String, io::Error> {
    let mut by_dtype: BTreeMap<GgmlType, u64> = BTreeMap::new();
    for tensor in gguf.tensors() {
        let entry = by_dtype.entry(tensor.dtype).or_default();
        *entry = entry
            .checked_add(tensor.n_bytes)
            .ok_or_else(|| invalid_data("dtype byte total overflowed"))?;
    }
    let (dtype, _) = by_dtype
        .into_iter()
        .max_by_key(|(dtype, bytes)| (*bytes, *dtype))
        .ok_or_else(|| invalid_data("model has no tensors"))?;
    Ok(format!("GGUF {dtype}"))
}

fn tensor_class_bytes(gguf: &Gguf) -> Result<BTreeMap<TensorClass, u64>, io::Error> {
    let mut totals = TensorClass::zero_map();
    for tensor in gguf.tensors() {
        let class = TensorClass::from_gguf_name(&tensor.name);
        let total = totals[&class]
            .checked_add(tensor.n_bytes)
            .ok_or_else(|| invalid_data(format!("{class:?} tensor byte total overflowed")))?;
        totals.insert(class, total);
    }
    Ok(totals)
}

fn decode_weight_bytes(
    gguf: &Gguf,
    resident: &BTreeMap<TensorClass, u64>,
    model: &ModelConfig,
) -> Result<BTreeMap<TensorClass, u64>, io::Error> {
    let embedding = gguf
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid_data("token_embd.weight is missing"))?;
    let embedding_row_bytes = embedding
        .n_bytes
        .checked_div(model.vocab_size)
        .filter(|_| embedding.n_bytes % model.vocab_size == 0)
        .ok_or_else(|| invalid_data("token embedding row byte count is not integral"))?;
    let mut touched = resident.clone();
    touched.insert(TensorClass::Embed, embedding_row_bytes);
    if gguf.tensor("output.weight").is_none() {
        touched.insert(TensorClass::Head, embedding.n_bytes);
    }
    Ok(touched)
}

#[allow(clippy::too_many_arguments)]
fn add_kv_bytes(
    touched: &mut BTreeMap<TensorClass, u64>,
    layers: u64,
    kv_heads: u64,
    head_dim: u64,
    dtype: KvCacheDtype,
    context_tokens: u64,
    generated_tokens: u64,
) -> Result<(), io::Error> {
    if generated_tokens == 0 {
        return Err(invalid_data(
            "KV byte accounting needs at least one decode evaluation",
        ));
    }
    let twice_average_depth = average_kv_depth(context_tokens, generated_tokens)?;
    let values_per_depth = kv_values_per_depth(layers, kv_heads, head_dim)?;
    let bytes_per_depth = kv_bytes_per_depth(values_per_depth, dtype)?;
    let average_bytes = bytes_per_depth
        .checked_mul(twice_average_depth)
        .ok_or_else(|| invalid_data("average KV bytes overflowed"))?
        / 2;
    touched.insert(TensorClass::Kv, average_bytes);
    Ok(())
}

fn average_kv_depth(context_tokens: u64, generated_tokens: u64) -> Result<u64, io::Error> {
    context_tokens
        .checked_mul(2)
        .and_then(|value| value.checked_add(generated_tokens))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid_data("average KV depth overflowed"))
}

fn kv_values_per_depth(layers: u64, kv_heads: u64, head_dim: u64) -> Result<u64, io::Error> {
    layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(kv_heads))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| invalid_data("KV values per depth overflowed"))
}

fn kv_bytes_per_depth(values_per_depth: u64, dtype: KvCacheDtype) -> Result<u64, io::Error> {
    match dtype {
        KvCacheDtype::Q8 => values_per_depth
            .checked_div(32)
            .filter(|_| values_per_depth.is_multiple_of(32))
            .and_then(|blocks| blocks.checked_mul(34))
            .ok_or_else(|| invalid_data("Q8 KV bytes per depth overflowed")),
        KvCacheDtype::F16 => values_per_depth
            .checked_mul(2)
            .ok_or_else(|| invalid_data("F16 KV bytes per depth overflowed")),
        KvCacheDtype::F32 => values_per_depth
            .checked_mul(4)
            .ok_or_else(|| invalid_data("F32 KV bytes per depth overflowed")),
    }
}

fn sum_tensor_classes(values: &BTreeMap<TensorClass, u64>) -> Result<u64, io::Error> {
    TensorClass::all()
        .into_iter()
        .try_fold(0_u64, |sum, class| {
            sum.checked_add(values[&class])
                .ok_or_else(|| invalid_data("tensor class byte sum overflowed"))
        })
}

const fn receipt_prefill_method(method: leone::PrefillMethod) -> ReceiptPrefillMethod {
    match method {
        leone::PrefillMethod::SequentialDecode => ReceiptPrefillMethod::SequentialDecode,
        leone::PrefillMethod::ChunkedCublasLtFp16 => ReceiptPrefillMethod::ChunkedCublasLtFp16,
        leone::PrefillMethod::TiledCublasLtFp16 => ReceiptPrefillMethod::TiledCublasLtFp16,
    }
}

const fn kv_dtype_name(dtype: KvCacheDtype) -> &'static str {
    match dtype {
        KvCacheDtype::Q8 => "q8_0",
        KvCacheDtype::F16 => "f16",
        KvCacheDtype::F32 => "f32",
    }
}

fn roofline(bandwidth_gbs: f64, bytes_per_token: u64, decode_median: f64) -> Roofline {
    let ceiling_tok_s = bandwidth_gbs * 1e9 / bytes_per_token as f64;
    Roofline {
        bandwidth_gbs_assumed: bandwidth_gbs,
        ceiling_tok_s,
        eta: decode_median / ceiling_tok_s,
        denominator_definition: Some(ROOFLINE_DENOMINATOR_DEFINITION.to_owned()),
    }
}

#[derive(Debug, Deserialize)]
struct LlamaBenchEntry {
    build_commit: String,
    model_filename: String,
    model_size: u64,
    n_prompt: u64,
    n_gen: u64,
    #[serde(default)]
    n_depth: u64,
    samples_ts: Vec<f64>,
}

#[derive(Debug, PartialEq)]
struct BenchmarkSummary {
    build_commit: String,
    model_filename: String,
    model_size: u64,
    context_tokens: u64,
    generated_tokens: u64,
    prefill_samples: Option<Vec<f64>>,
    decode_samples: Vec<f64>,
}

impl BenchmarkSummary {
    fn from_entries(entries: &[LlamaBenchEntry]) -> Result<Self, io::Error> {
        let decode = unique_entry(
            entries,
            |entry| entry.n_prompt == 0 && entry.n_gen > 0,
            "decode",
        )?;
        let prefill = optional_unique_entry(
            entries,
            |entry| entry.n_prompt > 0 && entry.n_gen == 0,
            "prefill",
        )?;
        validate_prefill_entry(decode, prefill)?;
        let context_tokens = entry_context_tokens(decode, prefill)?;

        Ok(Self {
            build_commit: decode.build_commit.clone(),
            model_filename: decode.model_filename.clone(),
            model_size: decode.model_size,
            context_tokens,
            generated_tokens: decode.n_gen,
            prefill_samples: prefill.map(|entry| entry.samples_ts.clone()),
            decode_samples: decode.samples_ts.clone(),
        })
    }
}

fn validate_prefill_entry(
    decode: &LlamaBenchEntry,
    prefill: Option<&LlamaBenchEntry>,
) -> Result<(), io::Error> {
    let Some(prefill) = prefill else {
        return Ok(());
    };
    if prefill.build_commit != decode.build_commit
        || prefill.model_filename != decode.model_filename
        || prefill.model_size != decode.model_size
    {
        return Err(invalid_data(
            "prefill and decode entries do not describe the same build and model",
        ));
    }
    if prefill.samples_ts.len() != decode.samples_ts.len() {
        return Err(invalid_data(
            "prefill and decode entries contain different repetition counts",
        ));
    }
    Ok(())
}

fn entry_context_tokens(
    decode: &LlamaBenchEntry,
    prefill: Option<&LlamaBenchEntry>,
) -> Result<u64, io::Error> {
    if decode.n_depth > 0 {
        return Ok(decode.n_depth);
    }
    prefill
        .map(|entry| entry.n_prompt)
        .ok_or_else(|| invalid_data("depth-0 llama-bench decode has no prefill entry"))
}

fn optional_unique_entry<'a>(
    entries: &'a [LlamaBenchEntry],
    predicate: impl Fn(&LlamaBenchEntry) -> bool,
    kind: &str,
) -> Result<Option<&'a LlamaBenchEntry>, io::Error> {
    let mut matches = entries.iter().filter(|entry| predicate(entry));
    let entry = matches.next();
    if matches.next().is_some() {
        return Err(invalid_data(format!(
            "llama-bench JSON has more than one {kind} entry"
        )));
    }
    Ok(entry)
}

fn unique_entry<'a>(
    entries: &'a [LlamaBenchEntry],
    predicate: impl Fn(&LlamaBenchEntry) -> bool,
    kind: &str,
) -> Result<&'a LlamaBenchEntry, io::Error> {
    let mut matches = entries.iter().filter(|entry| predicate(entry));
    let entry = matches
        .next()
        .ok_or_else(|| invalid_data(format!("llama-bench JSON has no {kind} entry")))?;
    if matches.next().is_some() {
        return Err(invalid_data(format!(
            "llama-bench JSON has more than one {kind} entry"
        )));
    }
    Ok(entry)
}

fn validate_commit(pin: &str, measured: &str) -> Result<(), io::Error> {
    if pin.is_empty() {
        return Err(invalid_data("external/PINNED is empty"));
    }
    if !pin.starts_with(measured) && !measured.starts_with(pin) {
        return Err(invalid_data(format!(
            "llama-bench commit {measured} does not match external/PINNED {pin}"
        )));
    }
    Ok(())
}

fn query_machine() -> Result<Machine, Box<dyn Error>> {
    let fields = query_machine_fields()?;
    Ok(Machine {
        hostname: fs::read_to_string("/proc/sys/kernel/hostname")?
            .trim()
            .to_owned(),
        gpu_name: fields[0].to_owned(),
        gpu_vram_mib: parse_rounded(&fields[1], "GPU memory")?,
        compute_cap: fields[2].to_owned(),
        driver: fields[3].to_owned(),
        cuda: query_cuda_version()?,
        cpu_model: query_cpu_model()?,
        ram_gib: query_ram_gib()?,
        gpu_clocks_mhz: GpuClocksMhz {
            graphics: parse_rounded(&fields[4], "graphics clock")?,
            memory: parse_rounded(&fields[5], "memory clock")?,
        },
        gpu_power_limit_w: fields[6].parse()?,
    })
}

fn query_machine_fields() -> Result<Vec<String>, Box<dyn Error>> {
    let query = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,compute_cap,driver_version,clocks.current.graphics,clocks.current.memory,power.limit",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let fields: Vec<&str> = query.trim().split(',').map(str::trim).collect();
    if fields.len() != 7 {
        return Err(invalid_data(format!(
            "nvidia-smi returned {} fields, expected 7",
            fields.len()
        ))
        .into());
    }
    Ok(fields.into_iter().map(str::to_owned).collect())
}

fn query_cuda_version() -> Result<String, Box<dyn Error>> {
    let output = command_output("nvidia-smi", &[])?;
    for line in output.lines() {
        if let Some(rest) = line.split("CUDA UMD Version:").nth(1) {
            if let Some(version) = rest.split_whitespace().next() {
                return Ok(version.to_owned());
            }
        }
    }
    Err(invalid_data("nvidia-smi did not report a CUDA UMD version").into())
}

fn query_cpu_model() -> Result<String, io::Error> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: "))
        .map(str::to_owned)
        .ok_or_else(|| invalid_data("/proc/cpuinfo did not report a CPU model"))
}

fn query_ram_gib() -> Result<u64, Box<dyn Error>> {
    let meminfo = fs::read_to_string("/proc/meminfo")?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|line| line.split_whitespace().next())
        .ok_or_else(|| invalid_data("/proc/meminfo did not report MemTotal"))?
        .parse::<u64>()?;
    kib.checked_add(524_288)
        .map(|rounded| rounded / 1_048_576)
        .ok_or_else(|| invalid_data("MemTotal is too large").into())
}

fn parse_rounded(value: &str, field: &str) -> Result<u64, io::Error> {
    let number: f64 = value
        .parse()
        .map_err(|_| invalid_data(format!("nvidia-smi returned invalid {field}: {value}")))?;
    let rounded = number.round();
    const U64_UPPER_EXCLUSIVE: f64 = 18_446_744_073_709_551_616.0;
    if !rounded.is_finite() || rounded <= 0.0 || rounded >= U64_UPPER_EXCLUSIVE {
        return Err(invalid_data(format!(
            "nvidia-smi returned out-of-range {field}: {value}"
        )));
    }
    Ok(rounded as u64)
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, io::Error> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|e| io::Error::new(e.kind(), format!("failed to run {program}: {e}")))?;
    if !output.status.success() {
        return Err(invalid_data(format!(
            "{program} exited with status {}",
            output.status
        )));
    }
    String::from_utf8(output.stdout)
        .map_err(|_| invalid_data(format!("{program} returned non-UTF-8 output")))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(n_prompt: u64, n_gen: u64, n_depth: u64, samples_ts: Vec<f64>) -> LlamaBenchEntry {
        LlamaBenchEntry {
            build_commit: "abcdef0".to_owned(),
            model_filename: "models/Qwen3-8B-Q4_K_M.gguf".to_owned(),
            model_size: 4_000_000_000,
            n_prompt,
            n_gen,
            n_depth,
            samples_ts,
        }
    }

    #[test]
    fn parses_one_prefill_and_one_decode_entry() {
        let entries = vec![
            entry(512, 0, 0, vec![8_000.0; 5]),
            entry(0, 128, 0, vec![90.0, 95.0, 100.0, 105.0, 110.0]),
        ];
        let summary = BenchmarkSummary::from_entries(&entries).unwrap();
        assert_eq!(summary.context_tokens, 512);
        assert_eq!(summary.generated_tokens, 128);
        assert_eq!(
            summarize_samples(&summary.decode_samples).unwrap().median,
            100.0
        );
    }

    #[test]
    fn parses_decode_depth_without_a_prefill_entry() {
        let entries = vec![entry(0, 128, 512, vec![160.0; 5])];
        let summary = BenchmarkSummary::from_entries(&entries).unwrap();
        assert_eq!(summary.context_tokens, 512);
        assert_eq!(summary.prefill_samples, None);
    }

    #[test]
    fn rejects_different_repetition_counts() {
        let entries = vec![
            entry(512, 0, 0, vec![8_000.0; 4]),
            entry(0, 128, 0, vec![100.0; 5]),
        ];
        assert!(BenchmarkSummary::from_entries(&entries).is_err());
    }

    #[test]
    fn accepts_full_and_short_forms_of_the_same_commit() {
        assert!(validate_commit("abcdef0123456789", "abcdef0").is_ok());
        assert!(validate_commit("abcdef0", "abcdef0123456789").is_ok());
        assert!(validate_commit("abcdef0", "1234567").is_err());
    }

    #[test]
    fn parses_explicit_prefill_benchmark_shape() {
        let arguments = [
            "--prefill-context".to_owned(),
            "2048".to_owned(),
            "--prefill-chunk".to_owned(),
            "512".to_owned(),
        ];
        let parsed = parse_bench(&arguments).unwrap();
        assert_eq!(parsed.prefill_context, Some(2_048));
        assert_eq!(parsed.prefill_chunk, 512);
    }

    #[test]
    fn kv_bytes_follow_dtype_and_average_decode_depth() {
        let mut touched = TensorClass::zero_map();
        add_kv_bytes(&mut touched, 36, 8, 128, KvCacheDtype::F16, 512, 128).unwrap();
        assert_eq!(touched[&TensorClass::Kv], 85_008_384);

        add_kv_bytes(&mut touched, 36, 8, 128, KvCacheDtype::F32, 512, 128).unwrap();
        assert_eq!(touched[&TensorClass::Kv], 170_016_768);

        add_kv_bytes(&mut touched, 36, 8, 128, KvCacheDtype::Q8, 512, 128).unwrap();
        assert_eq!(touched[&TensorClass::Kv], 45_160_704);
    }
}
