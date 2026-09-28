mod backend_choice;
mod build_info;
mod chat;
mod cli_args;
mod clock;
#[cfg(feature = "cuda")]
mod correctable_gate;
mod doctor;
mod eval;
#[cfg(feature = "cuda")]
mod execution_plan;
#[cfg(feature = "cuda")]
mod fork_gate;
#[cfg(feature = "cuda")]
mod gate_utils;
#[cfg(feature = "cuda")]
mod hibernation_gate;
mod quality;
mod registry;
mod server;
mod server_metrics;
pub mod service_memory;
mod token_io;
mod transport;
#[cfg(feature = "cuda")]
mod verify;

use backend_choice::{default_backend, BackendChoice};
use chrono::Utc;
use cli_args::flag_value;
#[cfg(feature = "metal")]
use leone::MemoryBudget;
#[cfg(feature = "cuda")]
use leone::PrefillWorkspace;
use leone::{
    token_stream_sha256, AdaptiveDrafter, AdaptiveDrafterConfig, Backend, CorrectableDrafter,
    CpuBackend, DecodeExecution, DecodeOp, DecodeProfileMode, Determinism, GenerateOptions,
    GenerationResult, KvCacheDtype, LogitCapture, MeasuredRate, MemoryAccounting, MemoryClass,
    Penalties, PenaltyWindow, Runtime, RuntimeError, Sampler, Speculation, SuffixDrafter,
    Temperature, Truncation, DEFAULT_PREFILL_CHUNK_TOKENS,
};
#[cfg(feature = "cuda")]
use leone_cuda::CudaBackend;
use leone_gguf::model::ModelConfig;
use leone_gguf::{ArchitectureSupport, GgmlType, Gguf};
#[cfg(feature = "metal")]
use leone_metal::MetalBackend;
#[cfg(all(feature = "metal", target_os = "macos"))]
use leone_metal::MetalHostMemorySnapshot;
#[cfg(feature = "cuda")]
use leone_receipt::GpuClocksMhz;
#[cfg(feature = "metal")]
use leone_receipt::MetalMachineMetadata;
#[cfg(feature = "cuda")]
use leone_receipt::UsableBar;
use leone_receipt::{
    sha256_bytes, sha256_file, summarize_duration_samples_ms, summarize_samples,
    write_runtime_receipt, ActiveCompute, BandwidthProvenance, DeterminismClaim, DurationSummary,
    Engine, Machine, MemoryTelemetry, ModelArtifact, OwnedMemoryBudget, OwnedMemoryClass,
    OwnedMemoryTelemetry, PrefillMethod as ReceiptPrefillMethod, ProcessMemorySnapshot,
    QualityReceipt, QualitySummary, RateSummary, ReductionOrder, Roofline, RuntimeReceipt,
    RuntimeResults, SamplerRecord, SpeculationRecord, SystemMemorySnapshot, Telemetry, TensorClass,
    Workload, HOSTNAME_REDACTED, ROOFLINE_DENOMINATOR_DEFINITION, RUNTIME_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SPEC_BANDWIDTH_GBS: f64 = 1008.0;
const LLAMA_BENCH_CAPTURE_SCHEMA_VERSION: u32 = 5;
#[cfg(all(feature = "metal", target_os = "macos"))]
const MAC_MEMORY_OBSERVATION_NOTE: &str =
    "macOS memory telemetry uses one native bridge call for task_info, sysctl hw.memsize, and host_statistics64. Those reads are not atomic. Available bytes estimate free_count plus external_page_count minus speculative_count (file-backed non-swap pages). Virtual bytes are address-space telemetry and do not count as memory savings.";
#[cfg(feature = "cuda")]
const SINGLE_STREAM_BENCH_MANIFEST: &str = "benchmarks/single-stream-prefill-v1.toml";

pub(crate) fn decode_execution_for_runtime<B: Backend>(
    runtime: &Runtime<B>,
    eager_decode: bool,
) -> DecodeExecution {
    select_decode_execution(
        eager_decode,
        runtime
            .model()
            .config()
            .architecture
            .decode_graph_supported(),
        runtime.backend().decode_graph_supported(),
    )
}

fn select_decode_execution(
    eager_decode: bool,
    architecture_supports_graph: bool,
    backend_supports_graph: bool,
) -> DecodeExecution {
    if eager_decode || !architecture_supports_graph || !backend_supports_graph {
        DecodeExecution::Eager
    } else {
        DecodeExecution::Graph
    }
}

fn main() -> ExitCode {
    server::mark_process_start();
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
    if dispatch_display(arguments) || dispatch_subcommand_help(arguments) {
        return Ok(true);
    }
    dispatch_command_handlers(arguments)
}

fn dispatch_command_handlers(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    let handlers = [
        dispatch_services,
        dispatch_generation,
        dispatch_inspection,
        dispatch_verification,
        dispatch_benchmarks,
    ];
    for handler in handlers {
        if handler(arguments)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn dispatch_subcommand_help(arguments: &[String]) -> bool {
    let Some(command) = arguments.first() else {
        return false;
    };
    let Some(help_index) = arguments
        .iter()
        .position(|argument| argument == "--help" || argument == "-h")
    else {
        return false;
    };
    let command_arguments = &arguments[..help_index];
    if dispatch_nested_help(command_arguments) || dispatch_primary_help(command) {
        return true;
    }
    false
}

fn dispatch_nested_help(arguments: &[String]) -> bool {
    let Some(command) = arguments.first().map(String::as_str) else {
        return false;
    };
    let Some(subcommand) = arguments.get(1).map(String::as_str) else {
        return false;
    };
    let entries = [
        (
            "inspect",
            "plan",
            "inspect plan <json>",
            "Validate and show a CUDA execution plan",
        ),
        (
            "verify",
            "session",
            "verify session -m <gguf> [options]",
            "Run the CUDA session replay gate",
        ),
        (
            "verify",
            "scheduler",
            "verify scheduler -m <gguf> [options]",
            "Run the CUDA scheduler gate",
        ),
        (
            "verify",
            "fork",
            "verify fork -m <gguf> [options]",
            "Run the CUDA fork gate",
        ),
        (
            "verify",
            "hibernate",
            "verify hibernate -m <gguf> [options]",
            "Run the CUDA hibernation gate",
        ),
        (
            "verify",
            "correctable",
            "verify correctable [options]",
            "Run the CUDA correctable speculation gate",
        ),
        (
            "receipt",
            "capture-llama-bench",
            "receipt capture-llama-bench <json> [options]",
            "Capture model identity before converting llama-bench JSON",
        ),
        (
            "receipt",
            "from-llama-bench",
            "receipt from-llama-bench <json> [options]",
            "Create a runtime receipt from llama-bench JSON",
        ),
        (
            "receipt",
            "verify-response",
            "receipt verify-response <json>",
            "Verify a signed response receipt",
        ),
        (
            "gguf",
            "inspect",
            "gguf inspect <file>",
            "Inspect GGUF metadata and tensor storage",
        ),
    ];
    for (entry_command, entry_subcommand, usage, description) in entries {
        if command == entry_command && subcommand == entry_subcommand {
            print_usage(usage, description);
            return true;
        }
    }
    false
}

fn dispatch_primary_help(command: &str) -> bool {
    let entries: [(&str, fn()); 14] = [
        ("chat", print_chat_help),
        ("doctor", print_doctor_help),
        ("serve", print_serve_help),
        ("generate", print_generate_help),
        ("eval", print_eval_help),
        ("pull", || {
            print_usage(
                "pull <model> [--registry <toml>]",
                "Download a model from the registry",
            )
        }),
        ("run", print_run_help),
        ("models", || {
            print_usage("models [--registry <toml>]", "List registry models")
        }),
        ("quality", || {
            print_usage(
                "quality --oracle <f32> --subject <f32> [options]",
                "Compare logits against an oracle",
            )
        }),
        ("tune", || {
            print_usage(
                "tune -m <gguf> [--out <json>] [options]",
                "Search CUDA execution plan candidates",
            )
        }),
        ("verify", || {
            print_usage(
                "verify -m <gguf> --tokens <u32le> --commit <sha> [--receipt]",
                "Verify a CUDA token stream",
            )
        }),
        ("inspect", print_inspect_help),
        ("receipt", print_receipt_help),
        ("gguf", print_gguf_help),
    ];
    for (entry_command, print) in entries {
        if command == entry_command {
            print();
            return true;
        }
    }
    if command == "bench" {
        print_usage("bench [options]", "Measure the CUDA benchmark workload");
        return true;
    }
    false
}

fn print_usage(usage: &str, description: &str) {
    println!("Usage: leone {usage}");
    println!("{description}.");
}

fn print_chat_help() {
    print_usage("chat -m <gguf> [options]", "Run an interactive text chat");
    println!("  -m, --model <gguf>    Load the model");
    println!("  -n, --tokens <n>      Set the maximum response length. Default: 512");
    println!(
        "  --backend cuda|cpu|metal  Select the execution backend. Default: {}",
        default_backend_name()
    );
    println!("  --eager-decode       Disable decode graph replay");
    println!("  --kv q8|f16|f32      Select KV cache storage. Default: f16");
    println!("  --plan <json>        Load a CUDA execution plan. CUDA backend only");
    println!("  --temp, --temperature <t> Scale logits by 1/t");
    println!("  --seed <n>           Seed the sampler. Default: 0");
    println!("  --top-k <n>          Keep the n highest-probability tokens");
    println!("  --top-p <p>          Keep the smallest set covering mass p");
    println!("  --min-p <p>          Keep tokens above p times the highest");
    println!("  --top-a <a>          Keep tokens above a times the highest squared");
    println!("  --tfs <z>            Tail-free sampling");
    println!("  --typical <p>        Locally typical sampling");
    println!("  --epsilon <e>        Keep tokens above a probability floor");
    println!("  --eta <e>            Use an entropy-relaxed probability floor");
    println!("  --min-k <tau>        Keep raw logits at the sharpest drop");
    println!("  --top-n-sigma <n>    Keep raw logits within n deviations");
    println!("  --repeat-penalty <r> Apply repetition scaling to seen logits");
    println!("  --presence-penalty <p> Subtract once from any seen token");
    println!("  --frequency-penalty <f> Subtract once per earlier occurrence");
    println!("  --penalty-window <n> Tokens of history considered");
    println!("  --dry-multiplier <m> Enable the repeated-suffix penalty");
    println!("  --dry-base <b>      Set DRY growth per matching token");
    println!("  --dry-allowed-length <n> Set the free match length");
    println!("  --dry-window <n>    Set the history searched by DRY");
    println!("  --draft <n>         Propose n tokens per step by suffix match");
    println!("  --adaptive-draft    Select a proposal from measured rounds");
    println!("  --no-draft          Disable adaptive speculation");
}

fn print_run_help() {
    print_usage("run <model> [--serve] [options]", "Run a registry model");
    println!("  --registry <toml>   Select a model registry");
    println!("  --serve             Start the local OpenAI-compatible server");
    println!("  Without --serve, options match `leone chat --help`.");
    println!("  With --serve, options match `leone serve --help`.");
    println!("  The selected registry entry supplies the verified model path.");
}

fn print_doctor_help() {
    print_usage(
        "doctor [-m <gguf>] [--backend cpu|cuda|metal] [--probe]",
        "Check platform, backend, and model metadata",
    );
    println!("  -m, --model <gguf>    Inspect model metadata");
    println!("  --backend cpu|cuda|metal  Select the probe backend");
    println!("  --probe               Load model weights with the selected backend");
}

fn print_inspect_help() {
    print_usage("inspect plan <json>", "Inspect a CUDA execution plan");
}

fn print_receipt_help() {
    print_usage(
        "receipt <capture-llama-bench|from-llama-bench|verify-response> ...",
        "Work with receipts",
    );
}

fn print_gguf_help() {
    print_usage(
        "gguf inspect <file>",
        "Inspect GGUF metadata and tensor storage",
    );
}

fn print_serve_help() {
    print_usage(
        "serve -m <gguf> [options]",
        "Start the local OpenAI-compatible server",
    );
    println!(
        "  --backend cuda|cpu|metal  Select the execution backend. Default: {}",
        default_backend_name()
    );
    println!("  --bind <address>      Listen address. Default: 127.0.0.1:8080");
    println!(
        "  --sessions <n>        Limit for each: admitted requests and resident idle sessions. Default: 2"
    );
    println!("  --context-limit <n>   Per-session context bound");
    println!("  --prefill-chunk <n>   Prompt tokens per scheduler chunk");
    println!("  --batch-size <n>      Maximum requests per decode pass. Default: min(8, backend maximum)");
    println!("  --hibernated-sessions <n> Maximum host sessions. Default: 8");
    println!("  --kv q8|f16|f32       Select KV cache storage. Default: f16");
    println!("  --plan <json>         Load a CUDA execution plan. CUDA backend only");
    println!("  --receipt-dir <path>  Write signed response receipts. Default: receipts");
    println!("  --signing-key <path> Set the 32-byte Ed25519 key file");
    println!("  --session-store <path> Persist model-bound session replay state");
    println!("  --memory-budget-bytes <auto|bytes>  Backend allocation limit");
    println!("  --host-memory-budget-bytes <auto|bytes>  Host snapshot limit");
    println!("  --kv-reservation-budget-bytes <auto|bytes>  Logical KV limit");
    println!("  --allow-remote        Permit a non-loopback listen address");
    println!("  --max-connections <n> Bound socket workers. Default: 64");
    println!("  --max-connections-per-client <n> Bound one client IP. Default: 4");
    println!("  --max-pending-requests <n> Bound the engine request queue. Default: 64");
    println!("  --max-output-bytes <n> Bound each response queue. Default: 262144");
    println!("  --request-timeout-ms <n> Bound request and service time. Default: 30000");
    println!("  --cors-origin <origin> Allow one explicit browser origin. Repeatable");
    println!("  --proxy-origin <origin> Allow the origin at a trusted proxy boundary");
    println!("  --trusted-proxy <ip> Trust X-Forwarded-For from one proxy IP. Repeatable");
}

fn print_generate_help() {
    print_usage(
        "generate -m <gguf> -p <prompt> -n <tokens> [options]",
        "Generate text",
    );
    println!(
        "  --backend cuda|cpu|metal  Select the execution backend. Default: {}",
        default_backend_name()
    );
    println!("  --receipt             Write a CUDA runtime receipt. CUDA backend only");
    println!("  --profile-decode      Profile CUDA steady-state decode. CUDA backend only");
    println!("  --eager-decode        Disable decode graph replay");
    println!("  --kv q8|f16|f32       Select KV cache storage. Default: f16");
    println!("  --plan <json>         Load a CUDA execution plan. CUDA backend only");
    println!("  --debug-tokens <file> Write generated token IDs as JSON");
    println!("  --debug-logits <file> Write the top five logits for each token");
    println!("  --seed <n>            Seed the sampler. Default: 0");
    println!("  --draft <n>           Propose n tokens per step by suffix match");
    println!("  --adaptive-draft     Select a point proposal and width from measured rounds");
    println!(
        "  --correctable         Select a distribution proposal and width from measured rounds"
    );
    println!("  --temp <t>            Scale logits by 1/t");
    println!("  --top-k <n>           Keep the n highest-probability tokens");
    println!("  --top-p <p>           Keep the smallest set covering mass p");
    println!("  --repeat-penalty <r>  Apply sign-aware scaling to seen logits");
    println!("  --presence-penalty <p> Subtract once from any seen token");
    println!("  --frequency-penalty <f> Subtract once per earlier occurrence");
}

fn print_eval_help() {
    print_usage(
        "eval -m <gguf> (--corpus <text> --token-limit <n>|--tokens <bin>) [options]",
        "Evaluate model logits",
    );
    println!("  --backend cuda|cpu|metal|cpu-q8_1  Select the execution backend");
    println!("  --tokens-out <file>   Write token IDs");
    println!("  --logits <file>       Write logits");
    println!("  --metadata <file>     Write execution and artifact metadata");
    println!("  --window <tokens>     Set the overlapping evaluation window. Default: 512");
    println!("  --position-limit <n>  Score only the first n positions from --tokens");
    println!("  --prefill-chunk <n>   Set the chunk size for corpus evaluation");
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
        "tune" => dispatch_tune(rest)?,
        _ => return Ok(false),
    }
    Ok(true)
}

#[cfg(feature = "cuda")]
fn dispatch_tune(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    execution_plan::tune(arguments)
}

#[cfg(not(feature = "cuda"))]
fn dispatch_tune(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err(invalid_data("tune requires a CUDA-enabled build").into())
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
            dispatch_plan_inspection(rest)?;
            Ok(true)
        }
        [command, inspect, path] if command == "gguf" && inspect == "inspect" => {
            inspect_gguf(Path::new(path))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(feature = "cuda")]
fn dispatch_plan_inspection(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    execution_plan::inspect(arguments)
}

#[cfg(not(feature = "cuda"))]
fn dispatch_plan_inspection(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err(invalid_data("inspect plan requires a CUDA-enabled build").into())
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
        [] => dispatch_verify(rest)?,
    }
    Ok(true)
}

#[cfg(feature = "cuda")]
fn dispatch_verify(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    verify::run(arguments)
}

#[cfg(not(feature = "cuda"))]
fn dispatch_verify(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err(invalid_data("verify requires a CUDA-enabled build").into())
}

fn dispatch_verification_kind(
    kind: &str,
    rest: &[String],
    fallback: &[String],
) -> Result<(), Box<dyn Error>> {
    match kind {
        "session" | "scheduler" => dispatch_session_scheduler(kind, rest)?,
        "fork" | "hibernate" => dispatch_fork_hibernate(kind, rest)?,
        "correctable" => dispatch_correctable(rest)?,
        _ => dispatch_verify(fallback)?,
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn dispatch_correctable(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    correctable_gate::run(arguments)
}

#[cfg(not(feature = "cuda"))]
fn dispatch_correctable(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err(invalid_data("verify correctable requires a CUDA-enabled build").into())
}

#[cfg(feature = "cuda")]
fn dispatch_session_scheduler(command: &str, rest: &[String]) -> Result<(), Box<dyn Error>> {
    match command {
        "session" => server::run_session_gate(rest)?,
        "scheduler" => server::run_scheduler_gate(rest)?,
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn dispatch_session_scheduler(_command: &str, _rest: &[String]) -> Result<(), Box<dyn Error>> {
    Err(invalid_data("verify session and scheduler require a CUDA-enabled build").into())
}

#[cfg(feature = "cuda")]
fn dispatch_fork_hibernate(command: &str, rest: &[String]) -> Result<(), Box<dyn Error>> {
    match command {
        "fork" => fork_gate::run(rest)?,
        "hibernate" => hibernation_gate::run(rest)?,
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn dispatch_fork_hibernate(_command: &str, _rest: &[String]) -> Result<(), Box<dyn Error>> {
    Err(invalid_data("verify fork and hibernate require a CUDA-enabled build").into())
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = rest;
        Err(invalid_data("bench requires a CUDA-enabled build").into())
    }
    #[cfg(feature = "cuda")]
    {
        run_bench(parse_bench(rest)?)?;
        Ok(true)
    }
}

fn dispatch_receipt_command(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    if dispatch_llama_bench_capture(arguments)? {
        return Ok(true);
    }
    if dispatch_llama_bench_receipt(arguments)? {
        return Ok(true);
    }
    dispatch_response_receipt(arguments)
}

fn dispatch_llama_bench_capture(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    match arguments {
        [receipt, capture, path, rest @ ..]
            if receipt == "receipt" && capture == "capture-llama-bench" =>
        {
            let options = parse_llama_bench_capture_options(rest)?;
            write_llama_bench_capture(Path::new(path), &options)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn dispatch_llama_bench_receipt(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    match arguments {
        [receipt, from_bench, path, rest @ ..]
            if receipt == "receipt" && from_bench == "from-llama-bench" =>
        {
            let options = parse_llama_bench_options(rest)?;
            convert_llama_bench(Path::new(path), &options)?;
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

fn default_backend_name() -> &'static str {
    #[cfg(feature = "metal")]
    {
        "metal"
    }
    #[cfg(all(not(feature = "metal"), feature = "cuda"))]
    {
        "cuda"
    }
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    {
        "cpu"
    }
}

fn print_help() {
    println!("leone {VERSION}");
    println!();
    println!("Usage:");
    println!("  leone --build-info    Print source provenance for this executable");
    println!("  leone");
    println!("  leone chat -m <gguf> [options]");
    println!("  leone doctor [-m <gguf>] [--backend cpu|cuda|metal] [--probe]");
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
    println!("  leone receipt capture-llama-bench <json> --output <json> --backend cuda|cpu|metal --model-sha256-before <sha256> --model-file-bytes-before <bytes>");
    println!("  leone receipt from-llama-bench <json> --capture <json> [--quality-ref <uuid>]");
    println!("  leone receipt verify-response <json>");
    println!("  leone gguf inspect <file>");
    println!();
    println!("Chat options:");
    println!("  -n, --tokens <n>     Set the maximum response length. Default: 512");
    println!(
        "  --backend cuda|cpu|metal   Select the execution backend. Default: {}",
        default_backend_name()
    );
    println!("  --eager-decode       Disable decode graph replay");
    println!("  --kv q8|f16|f32      Select KV cache storage. Default: f16");
    println!("  --plan <json>         Load a proof-gated execution plan. CUDA backend only");
    println!();
    println!("Serve options:");
    println!(
        "  --backend cuda|cpu|metal  Select the execution backend. Default: {}",
        default_backend_name()
    );
    println!("  --bind <address>      Listen address. Default: 127.0.0.1:8080");
    println!(
        "  --sessions <n>        Limit for each: admitted requests and resident idle sessions. Default: 2"
    );
    println!("  --context-limit <n>   Per-session context bound. Default: model context.");
    println!("  --prefill-chunk <n>   Prompt tokens per scheduler chunk. Default: plan or 4096.");
    println!("  --batch-size <n>      Maximum requests per decode pass. Default: min(8, backend maximum)");
    println!("  --hibernated-sessions <n> Maximum host sessions. Default: 8");
    println!("  --memory-budget-bytes <auto|bytes>  Backend allocation limit");
    println!("  --host-memory-budget-bytes <auto|bytes>  Host snapshot limit");
    println!("  --kv-reservation-budget-bytes <auto|bytes>  Logical KV limit");
    println!("  --kv q8|f16|f32      Select KV cache storage. Default: f16");
    println!("  --plan <json>         Load a proof-gated execution plan. CUDA backend only");
    println!("  --receipt-dir <path> Write signed response receipts. Default: receipts");
    println!("  --signing-key <path> Set the 32-byte Ed25519 key file");
    println!("  --session-store <path> Persist model-bound session replay state");
    println!("  --allow-remote        Permit a non-loopback listen address");
    println!("  --max-connections <n> Bound socket workers. Default: 64");
    println!("  --max-connections-per-client <n> Bound one client IP. Default: 4");
    println!("  --max-pending-requests <n> Bound engine request queue. Default: 64");
    println!("  --max-output-bytes <n> Bound each response queue. Default: 262144");
    println!(
        "  --request-timeout-ms <n> Bound request read, write, and service time. Default: 30000"
    );
    println!("  --cors-origin <origin> Allow one explicit browser origin. Repeatable");
    println!("  --proxy-origin <origin> Allow the origin at a configured trusted proxy boundary");
    println!("  --trusted-proxy <ip> Trust X-Forwarded-For from one proxy IP. Repeatable");
    println!();
    println!("Generate options:");
    println!(
        "  --backend cuda|cpu|metal    Select the execution backend. Default: {}",
        default_backend_name()
    );
    println!("  --receipt             Write a CUDA runtime receipt. CUDA backend only");
    println!("  --profile-decode      Profile CUDA steady-state decode. CUDA backend only");
    println!("  --eager-decode        Disable decode graph replay");
    println!("  --kv q8|f16|f32       Select KV cache storage. Default: f16");
    println!("  --plan <json>          Load a proof-gated execution plan. CUDA backend only");
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
    println!("  --repeat-penalty <r>  Apply sign-aware scaling to seen logits. 1.0 is off");
    println!("  --presence-penalty <p> Subtract once from any seen token");
    println!("  --frequency-penalty <f> Subtract once per earlier occurrence");
    println!("  --penalty-window <n>  Tokens considered. Zero disables. Default: 64");
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
    println!("  --metadata <file>     Write execution and artifact metadata");
    println!("  --backend cuda|cpu|metal|cpu-q8_1");
    println!(
        "                         Select the execution backend. Default: {}",
        default_backend_name()
    );
    println!("  --window <tokens>     Set the overlapping evaluation window. Default: 512");
    println!("  --position-limit <n> Score only the first n positions from --tokens");
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "cuda")]
struct BenchArgs {
    receipt: bool,
    capacity_only: bool,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    quality_ref: Option<Uuid>,
    prefill_context: Option<usize>,
    prefill_chunk: usize,
}

#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
struct BenchBuilder {
    receipt: bool,
    capacity_only: bool,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    quality_ref: Option<Uuid>,
    prefill_context: Option<usize>,
    prefill_chunk: usize,
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn parse_bench_flag(parsed: &mut BenchBuilder, flag: &str) -> Result<bool, io::Error> {
    match flag {
        "--receipt" => parsed.receipt = true,
        "--capacity-only" => parsed.capacity_only = true,
        "--eager-decode" => parsed.eager_decode = true,
        _ => return Ok(false),
    }
    Ok(true)
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
            backend: default_backend(),
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
        "metal" => BackendChoice::Metal,
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
    let value = value
        .parse()
        .map_err(|_| invalid_data(format!("penalty window is invalid: {value}")))?;
    parsed.penalties.window = PenaltyWindow::from_size(value);
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
    let longest = NonZeroUsize::new(16).expect("sixteen is nonzero");
    let shortest = NonZeroUsize::new(4).expect("four is nonzero");
    SuffixDrafter::new(longest, shortest, proposal)
        .map(Speculation::Suffix)
        .map_err(|error| invalid_data(error.to_string()))
}

fn generate_text(arguments: GenerateArgs) -> Result<(), Box<dyn Error>> {
    validate_generation_options(&arguments)?;
    let interrupted = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&interrupted);
    ctrlc::set_handler(move || handler_flag.store(true, Ordering::Relaxed))?;
    run_generation_backend(arguments, interrupted)
}

fn run_generation_backend(
    arguments: GenerateArgs,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    match arguments.backend {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => run_cuda_generation(arguments, interrupted),
        #[cfg(not(feature = "cuda"))]
        BackendChoice::Cuda => {
            backend_choice::validate_compiled(BackendChoice::Cuda).map_err(Into::into)
        }
        #[cfg(feature = "metal")]
        BackendChoice::Metal => run_metal_generation(arguments, interrupted),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => {
            backend_choice::validate_compiled(BackendChoice::Metal).map_err(Into::into)
        }
        BackendChoice::Cpu => run_cpu_generation(arguments, interrupted),
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_generation(
    arguments: GenerateArgs,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let backend = CudaBackend::new(0)?;
    let receipt_machine = generation_receipt_machine(&arguments, || query_cuda_machine(&backend))?;
    run_generation(backend, arguments, interrupted, receipt_machine)
}

#[cfg(feature = "metal")]
fn run_metal_generation(
    arguments: GenerateArgs,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let backend = MetalBackend::new()?;
    let receipt_machine = generation_receipt_machine(&arguments, || query_metal_machine(&backend))?;
    run_generation(backend, arguments, interrupted, receipt_machine)
}

fn run_cpu_generation(
    arguments: GenerateArgs,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let receipt_machine = generation_receipt_machine(&arguments, query_cpu_machine)?;
    run_generation(CpuBackend::new(), arguments, interrupted, receipt_machine)
}

fn validate_generation_options(arguments: &GenerateArgs) -> Result<(), Box<dyn Error>> {
    if arguments.profile_decode && arguments.backend != BackendChoice::Cuda {
        return Err(invalid_data("generate --profile-decode requires the CUDA backend").into());
    }
    if arguments.plan.is_some() && arguments.backend != BackendChoice::Cuda {
        return Err(invalid_data("--plan requires the CUDA backend").into());
    }
    backend_choice::validate_compiled(arguments.backend).map_err(Into::into)
}

#[derive(Debug, Deserialize)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct BenchManifest {
    model: BenchModel,
    workload: BenchWorkload,
    comparator: BenchComparator,
}

#[derive(Debug, Deserialize)]
struct ArtifactManifest {
    model: BenchModel,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct BenchModel {
    filename: String,
    sha256: String,
    format: String,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct BenchWorkload {
    prefill_tokens: Vec<usize>,
    decode_tokens: Vec<usize>,
    context_cap_tokens: usize,
    batch: usize,
    reps: usize,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct BenchComparator {
    pin_file: String,
    prefill_raw_json: String,
    prefill_tokens: usize,
}

#[cfg(feature = "cuda")]
fn run_bench(arguments: BenchArgs) -> Result<(), Box<dyn Error>> {
    let BenchSetup {
        root,
        manifest,
        prompt_tokens,
        decode_tokens,
        actual_sha256,
        engine_commit,
        linked_quality,
        machine,
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
        machine.as_ref(),
        &engine_commit,
        linked_quality.as_ref(),
        determinism,
    )
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
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
    machine: Option<&Machine>,
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
        machine.ok_or_else(|| invalid_data("bench receipt machine metadata is missing"))?,
        engine_commit,
        linked_quality,
        determinism,
    )?;
    println!("receipt: {}", path.display());
    Ok(())
}

#[cfg(feature = "cuda")]
struct BenchSetup {
    root: PathBuf,
    manifest: BenchManifest,
    prompt_tokens: usize,
    decode_tokens: usize,
    actual_sha256: String,
    engine_commit: String,
    linked_quality: Option<QualityReceipt>,
    machine: Option<Machine>,
    runtime: Runtime<CudaBackend>,
    decode_execution: DecodeExecution,
}

#[cfg(feature = "cuda")]
fn prepare_bench(arguments: &BenchArgs) -> Result<BenchSetup, Box<dyn Error>> {
    let root = std::env::current_dir()?;
    let (manifest, prompt_tokens, decode_tokens) = load_bench_manifest(&root)?;
    validate_bench_request(arguments, &manifest, prompt_tokens, decode_tokens)?;
    let (actual_sha256, engine_commit, linked_quality) =
        load_bench_identity(&root, &manifest, arguments)?;
    let (runtime, decode_execution) = load_bench_runtime(&root, &manifest, arguments)?;
    let machine = if arguments.receipt {
        Some(query_cuda_machine(runtime.backend())?)
    } else {
        None
    };
    Ok(BenchSetup {
        root,
        manifest,
        prompt_tokens,
        decode_tokens,
        actual_sha256,
        engine_commit,
        linked_quality,
        machine,
        runtime,
        decode_execution,
    })
}

#[cfg(feature = "cuda")]
fn load_bench_manifest(root: &Path) -> Result<(BenchManifest, usize, usize), Box<dyn Error>> {
    let manifest: BenchManifest = toml::from_str(&fs::read_to_string(
        root.join(SINGLE_STREAM_BENCH_MANIFEST),
    )?)?;
    let prompt_tokens = one_manifest_value(&manifest.workload.prefill_tokens, "prefill_tokens")?;
    let decode_tokens = one_manifest_value(&manifest.workload.decode_tokens, "decode_tokens")?;
    Ok((manifest, prompt_tokens, decode_tokens))
}

#[cfg(feature = "cuda")]
fn load_bench_identity(
    root: &Path,
    manifest: &BenchManifest,
    arguments: &BenchArgs,
) -> Result<(String, String, Option<QualityReceipt>), Box<dyn Error>> {
    let model_path = resolve_model_path(root, Path::new(&manifest.model.filename))?;
    let actual_sha256 = sha256_file(&model_path)?;
    validate_bench_model(&actual_sha256, &manifest.model.sha256)?;
    let engine_commit = embedded_source_commit()?.to_owned();
    let linked_quality = load_linked_quality(
        root,
        arguments.quality_ref,
        "leone",
        &engine_commit,
        &actual_sha256,
    )?;
    Ok((actual_sha256, engine_commit, linked_quality))
}

#[cfg(feature = "cuda")]
fn load_bench_runtime(
    root: &Path,
    manifest: &BenchManifest,
    arguments: &BenchArgs,
) -> Result<(Runtime<CudaBackend>, DecodeExecution), Box<dyn Error>> {
    let model_path = resolve_model_path(root, Path::new(&manifest.model.filename))?;
    let runtime = Runtime::load(CudaBackend::new(0)?, &model_path)?;
    let decode_execution = bench_decode_execution(&runtime, arguments.eager_decode);
    Ok((runtime, decode_execution))
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn validate_bench_model(actual_sha256: &str, expected_sha256: &str) -> Result<(), Box<dyn Error>> {
    if actual_sha256 != expected_sha256 {
        return Err(invalid_data(format!(
            "model SHA-256 is {actual_sha256}, expected {expected_sha256}"
        ))
        .into());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench_decode_execution<B: Backend>(runtime: &Runtime<B>, eager_decode: bool) -> DecodeExecution {
    decode_execution_for_runtime(runtime, eager_decode)
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
struct BenchDecodeSample {
    prefill: f64,
    decode: f64,
    ttft_ms: f64,
    detokenized_bytes: usize,
}

#[cfg(feature = "cuda")]
struct BenchDecodeRepetitionInput<'a> {
    prompt: &'a [u32],
    decode_tokens: usize,
    arguments: &'a BenchArgs,
    decode_execution: DecodeExecution,
    transcript: &'a str,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
struct BenchTtftEvidence {
    context: usize,
    samples: Vec<f64>,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn one_manifest_value(values: &[usize], name: &'static str) -> Result<usize, io::Error> {
    match values {
        [value] if *value > 0 => Ok(*value),
        [_] => Err(invalid_data(format!("{name} must be nonzero"))),
        _ => Err(invalid_data(format!("{name} must contain one value"))),
    }
}

#[cfg(feature = "cuda")]
fn required_rate(rate: MeasuredRate, name: &'static str) -> Result<f64, io::Error> {
    match rate {
        MeasuredRate::TokensPerSecond(value) => Ok(value),
        MeasuredRate::Unavailable => Err(invalid_data(format!("{name} rate is unavailable"))),
    }
}

#[derive(Debug)]
#[cfg(feature = "cuda")]
struct BenchPrefillEvidence {
    ttft_ms: DurationSummary,
    ttft_context_tokens: u64,
    method: leone::PrefillMethod,
    workspace: PrefillWorkspace,
    usable_bar: Option<UsableBar>,
}

#[cfg(feature = "cuda")]
fn load_prefill_comparator(
    root: &Path,
    manifest: &BenchManifest,
    model_sha256: &str,
) -> Result<RateSummary, Box<dyn Error>> {
    validate_comparator_prefill_tokens(manifest)?;
    let entry = read_prefill_comparator_entry(root, manifest)?;
    validate_prefill_comparator_entry(root, manifest, model_sha256, &entry)?;
    summarize_samples(&entry.samples_ts).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn read_prefill_comparator_entry(
    root: &Path,
    manifest: &BenchManifest,
) -> Result<LlamaBenchEntry, Box<dyn Error>> {
    let raw_path = resolve_repo_relative_path(
        root,
        Path::new(&manifest.comparator.prefill_raw_json),
        "comparator JSON",
    )?;
    let entries: Vec<LlamaBenchEntry> = serde_json::from_slice(&fs::read(&raw_path)?)?;
    Ok(unique_entry(
        &entries,
        |entry| entry.n_prompt == manifest.comparator.prefill_tokens as u64 && entry.n_gen == 0,
        "pp512 prefill",
    )?
    .clone())
}

#[cfg(feature = "cuda")]
fn validate_prefill_comparator_entry(
    root: &Path,
    manifest: &BenchManifest,
    model_sha256: &str,
    entry: &LlamaBenchEntry,
) -> Result<(), Box<dyn Error>> {
    validate_comparator_identity(root, manifest, model_sha256, entry)?;
    validate_llama_bench_prefill_depth(entry)?;
    if llama_bench_backend(entry)? != BackendChoice::Cuda {
        return Err(invalid_data("prefill comparator must record active CUDA execution").into());
    }
    validate_comparator_samples(root, manifest, entry)?;
    Ok(())
}

fn validate_llama_bench_prefill_depth(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.n_depth == 0 {
        Ok(())
    } else {
        Err(invalid_data(
            "llama-bench prefill comparator must start with an empty KV cache",
        ))
    }
}

#[cfg(feature = "cuda")]
fn validate_comparator_prefill_tokens(manifest: &BenchManifest) -> Result<(), io::Error> {
    if manifest.comparator.prefill_tokens == 512 {
        Ok(())
    } else {
        Err(invalid_data("usable prefill comparator must measure pp512"))
    }
}

#[cfg(feature = "cuda")]
fn validate_comparator_identity(
    root: &Path,
    manifest: &BenchManifest,
    model_sha256: &str,
    entry: &LlamaBenchEntry,
) -> Result<(), Box<dyn Error>> {
    let pin_path = resolve_repo_relative_path(
        root,
        Path::new(&manifest.comparator.pin_file),
        "comparator pin",
    )?;
    let pin = fs::read_to_string(pin_path)?.trim().to_owned();
    validate_commit(&pin, &entry.build_commit)?;
    let expected_model = logical_model_path(Path::new(&manifest.model.filename))?;
    if entry.model_filename != expected_model {
        return Err(invalid_data(format!(
            "prefill comparator model is {}, expected {}",
            entry.model_filename, expected_model
        ))
        .into());
    }
    let raw_model_sha256 = sha256_file(root.join(&expected_model))?;
    if raw_model_sha256 != model_sha256 {
        return Err(invalid_data("prefill comparator model hash does not match").into());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_comparator_samples(
    root: &Path,
    manifest: &BenchManifest,
    entry: &LlamaBenchEntry,
) -> Result<(), Box<dyn Error>> {
    let expected_model = logical_model_path(Path::new(&manifest.model.filename))?;
    if entry.model_filename != expected_model {
        return Err(invalid_data("prefill comparator model path is not normalized").into());
    }
    let gguf = Gguf::open(root.join(expected_model))?;
    let model = ModelConfig::from_metadata(gguf.metadata())?;
    validate_llama_bench_gpu_layers(BackendChoice::Cuda, entry.n_gpu_layers, model.n_layer)?;
    let tensor_bytes = gguf_tensor_bytes(&gguf)?;
    if entry.model_size != tensor_bytes {
        return Err(invalid_data("prefill comparator tensor bytes do not match").into());
    }
    if entry.samples_ts.len() != manifest.workload.reps {
        return Err(invalid_data("prefill comparator repetition count does not match").into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn write_bench_receipt<B: Backend>(
    root: &Path,
    runtime: &Runtime<B>,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    decode_tok_s: leone_receipt::RateSummary,
    prefill_tok_s: leone_receipt::RateSummary,
    prefill: BenchPrefillEvidence,
    machine: &Machine,
    engine_commit: &str,
    linked_quality: Option<&QualityReceipt>,
    determinism: DeterminismClaim,
) -> Result<PathBuf, Box<dyn Error>> {
    let model = runtime.model();
    let backend_name = runtime.backend().name();
    let metrics =
        bench_receipt_metrics(runtime, manifest, arguments, decode_tok_s.median, machine)?;
    let notes = bench_receipt_notes(runtime, manifest, arguments, &prefill, machine);
    let receipt = build_bench_receipt(
        model.file_bytes(),
        backend_name,
        manifest,
        arguments,
        machine.clone(),
        metrics,
        decode_tok_s,
        prefill_tok_s,
        prefill,
        notes,
        engine_commit,
        linked_quality,
        determinism,
    )?;
    write_runtime_receipt(root.join("receipts"), &receipt).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn build_bench_receipt(
    model_file_bytes: u64,
    backend_name: &str,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    machine: Machine,
    metrics: BenchReceiptMetrics,
    decode_tok_s: leone_receipt::RateSummary,
    prefill_tok_s: leone_receipt::RateSummary,
    prefill: BenchPrefillEvidence,
    notes: Vec<String>,
    engine_commit: &str,
    linked_quality: Option<&QualityReceipt>,
    determinism: DeterminismClaim,
) -> Result<RuntimeReceipt, Box<dyn Error>> {
    let BenchReceiptMetrics {
        context_tokens,
        generated_tokens,
        reps,
        bytes_per_token_by_class,
        bytes_per_token_total,
        weights_resident_bytes_by_class,
        model_bytes_total,
        spec_roofline,
        measured_roofline,
        bandwidth,
        memory,
    } = metrics;
    let execution = bench_execution_name(arguments.eager_decode);
    let kv_dtype = kv_dtype_name(arguments.kv_cache_dtype);
    let receipt = RuntimeReceipt {
        schema_version: RUNTIME_SCHEMA_VERSION,
        receipt_id: Uuid::new_v4(),
        created_utc: Utc::now(),
        machine,
        workload: Workload {
            engine: Engine {
                name: "leone".to_owned(),
                git_commit: engine_commit.to_owned(),
                build_flags: leone_build_flags(vec![
                    format!("backend={backend_name}"),
                    "batch=1".to_owned(),
                    format!("decode={execution}"),
                    format!("kv={kv_dtype}"),
                    format!("prefill={}", prefill.method.name()),
                    format!("prefill_chunk={}", arguments.prefill_chunk),
                    "sampler=greedy".to_owned(),
                ]),
            },
            model_artifact: ModelArtifact {
                path: public_artifact_path(Path::new(&manifest.model.filename))?,
                sha256: manifest.model.sha256.clone(),
                file_bytes: model_file_bytes,
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
            roofline_measured_achievable: measured_roofline,
            bandwidth: Some(bandwidth),
            memory: Some(memory),
        },
        quality_ref: linked_quality.map(|quality| quality.receipt_id),
        quality_summary: linked_quality.map(quality_summary),
        determinism: Some(determinism),
        speculation: None,
        notes,
    };
    Ok(receipt)
}

#[cfg(feature = "cuda")]
const fn bench_execution_name(eager_decode: bool) -> &'static str {
    if eager_decode {
        "eager"
    } else {
        "cuda-graph"
    }
}

#[cfg(feature = "cuda")]
struct BenchReceiptMetrics {
    context_tokens: u64,
    generated_tokens: u64,
    reps: u64,
    bytes_per_token_by_class: BTreeMap<TensorClass, u64>,
    bytes_per_token_total: u64,
    weights_resident_bytes_by_class: BTreeMap<TensorClass, u64>,
    model_bytes_total: u64,
    spec_roofline: Roofline,
    measured_roofline: Option<Roofline>,
    bandwidth: BandwidthProvenance,
    memory: MemoryTelemetry,
}

#[cfg(feature = "cuda")]
fn bench_receipt_metrics<B: Backend>(
    runtime: &Runtime<B>,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    decode_median: f64,
    machine: &Machine,
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
        spec_roofline: spec_roofline(machine, bytes_per_token_total, decode_median),
        measured_roofline: measured_roofline(machine, bytes_per_token_total, decode_median),
        bandwidth: bandwidth_provenance(machine),
        memory: backend_memory_telemetry(
            runtime.backend().memory_accounting(),
            "bench does not sample process and system memory",
        ),
    })
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn bench_receipt_notes<B: Backend>(
    runtime: &Runtime<B>,
    manifest: &BenchManifest,
    arguments: BenchArgs,
    prefill: &BenchPrefillEvidence,
    machine: &Machine,
) -> Vec<String> {
    let import_metrics = runtime.backend().model_import_metrics();
    let roofline_note = match bandwidth_provenance(machine) {
        BandwidthProvenance::Specification { bandwidth_gbs, .. } => {
            format!("The spec roofline uses {bandwidth_gbs:.0} GB/s for the active CUDA device.")
        }
        BandwidthProvenance::Measured { .. } => {
            "The roofline uses the measured device bandwidth receipt.".to_owned()
        }
        BandwidthProvenance::Unavailable { .. } => {
            "The spec roofline is unavailable for the active device.".to_owned()
        }
    };
    with_session_note(vec![
        roofline_note,
        "Achievable eta is unverified without a matching device bandwidth receipt.".to_owned(),
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
    receipt_machine: Option<Machine>,
) -> Result<(), Box<dyn Error>> {
    let backend_name = backend.name();
    let plan = generation_plan(arguments.plan.as_deref(), &arguments.model, backend_name)?;
    let mut runtime = Runtime::load(backend, &arguments.model)?;
    let options = generation_options(&runtime, &arguments, |options| {
        apply_generation_plan(&plan, options);
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
        receipt_machine.as_ref(),
        selected_kv_cache_dtype,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn generation_plan(
    explicit: Option<&Path>,
    model: &Path,
    backend: &str,
) -> Result<Option<execution_plan::PlanSelection>, Box<dyn Error>> {
    execution_plan::load_optional(explicit, model, backend)
}

#[cfg(not(feature = "cuda"))]
fn generation_plan(
    explicit: Option<&Path>,
    _model: &Path,
    _backend: &str,
) -> Result<Option<()>, Box<dyn Error>> {
    if explicit.is_some() {
        return Err(invalid_data("--plan requires a CUDA-enabled build").into());
    }
    Ok(None)
}

#[cfg(feature = "cuda")]
fn apply_generation_plan(
    plan: &Option<execution_plan::PlanSelection>,
    options: &mut GenerateOptions,
) {
    if let Some(plan) = plan.as_ref() {
        plan.apply(options);
    }
}

#[cfg(not(feature = "cuda"))]
fn apply_generation_plan(_plan: &Option<()>, _options: &mut GenerateOptions) {}

fn generation_receipt_machine<F>(
    arguments: &GenerateArgs,
    build: F,
) -> Result<Option<Machine>, Box<dyn Error>>
where
    F: FnOnce() -> Result<Machine, Box<dyn Error>>,
{
    if !arguments.receipt {
        return Ok(None);
    }
    embedded_source_commit()?;
    Ok(Some(build()?))
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
        decode_execution: decode_execution_for_runtime(runtime, arguments.eager_decode),
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
    machine: Option<&Machine>,
    kv_cache_dtype: KvCacheDtype,
) -> Result<(), Box<dyn Error>> {
    if arguments.receipt {
        let machine = machine.ok_or_else(|| invalid_data("receipt machine metadata is missing"))?;
        write_generation_receipt(runtime, result, backend_name, machine, kv_cache_dtype)?;
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
    machine: &Machine,
    kv_cache_dtype: KvCacheDtype,
) -> Result<(), Box<dyn Error>> {
    let rates = generation_receipt_rates(result)?;
    let model = generation_receipt_model(runtime)?;
    let metrics =
        generation_receipt_metrics(runtime, result, kv_cache_dtype, rates.decode_rate, machine)?;
    let results = generation_receipt_results(result, &rates, &metrics)?;
    let determinism = generation_receipt_determinism(runtime, result);
    let root = std::env::current_dir()?;
    let receipt = RuntimeReceipt {
        schema_version: RUNTIME_SCHEMA_VERSION,
        receipt_id: Uuid::new_v4(),
        created_utc: Utc::now(),
        machine: machine.clone(),
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
        notes: generation_receipt_notes(result, backend_name),
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
        path: public_artifact_path(model.path())?,
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
    measured_roofline: Option<Roofline>,
    bandwidth: BandwidthProvenance,
    memory: MemoryTelemetry,
}

fn generation_receipt_metrics<B: Backend>(
    runtime: &Runtime<B>,
    result: &GenerationResult,
    kv_cache_dtype: KvCacheDtype,
    decode_rate: f64,
    machine: &Machine,
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
        spec_roofline: spec_roofline(machine, bytes_per_token_total, decode_rate),
        measured_roofline: measured_roofline(machine, bytes_per_token_total, decode_rate),
        bandwidth: bandwidth_provenance(machine),
        memory: backend_memory_telemetry(
            runtime.backend().memory_accounting(),
            "generation does not sample process and system memory",
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
        roofline_measured_achievable: metrics.measured_roofline.clone(),
        bandwidth: Some(metrics.bandwidth.clone()),
        memory: Some(metrics.memory.clone()),
    })
}

fn generation_receipt_engine(
    result: &GenerationResult,
    backend_name: &str,
) -> Result<Engine, io::Error> {
    Ok(Engine {
        name: "leone".to_owned(),
        git_commit: embedded_source_commit()?.to_owned(),
        build_flags: leone_build_flags(vec![
            format!("backend={backend_name}"),
            "batch=1".to_owned(),
            format!("prefill={}", result.stats.prefill_method.name()),
            "sampler=greedy".to_owned(),
        ]),
    })
}

fn embedded_source_commit() -> Result<&'static str, io::Error> {
    let commit = env!("LEONE_SOURCE_COMMIT");
    if commit == "unknown" {
        return Err(invalid_data(
            "runtime receipts require an embedded source commit",
        ));
    }
    if env!("LEONE_SOURCE_DIRTY") == "true" {
        return Err(invalid_data("runtime receipts require a clean source tree"));
    }
    Ok(commit)
}

fn leone_build_flags(mut flags: Vec<String>) -> Vec<String> {
    flags.push(format!("target={}", env!("LEONE_BUILD_TARGET")));
    flags.push(format!("profile={}", env!("LEONE_BUILD_PROFILE")));
    flags.push("source_tree_dirty=false".to_owned());
    flags
}

fn generation_receipt_notes(result: &GenerationResult, backend_name: &str) -> Vec<String> {
    let notes = vec![
        format!("Active compute backend: {backend_name}."),
        "Achievable eta is unverified without a matching device bandwidth receipt.".to_owned(),
        format!("Prefill uses {}.", result.stats.prefill_method.name()),
        format!(
            "Prefill workspace: {} bytes total, including {} bytes of batch activations.",
            result.stats.prefill_workspace.total_bytes,
            result.stats.prefill_workspace.batch_activation_bytes,
        ),
    ];
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let notes = {
        let mut notes = notes;
        notes.push(MAC_MEMORY_OBSERVATION_NOTE.to_owned());
        notes
    };
    notes
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

fn convert_llama_bench(
    json_path: &Path,
    options: &LlamaBenchReceiptOptions,
) -> Result<(), Box<dyn Error>> {
    let LlamaBenchSource {
        root,
        summary,
        pin,
        file_bytes,
        model_sha256,
        linked_quality,
        capture,
    } = load_llama_bench_source(json_path, &options.capture_path, options.quality_ref)?;
    let LlamaBenchRates {
        decode_tok_s,
        prefill_tok_s,
        ttft_ms,
        ttft_context_tokens,
        prefill_method,
        reps,
    } = llama_bench_rates(&summary)?;
    let machine = public_capture_machine(&capture.machine);
    let LlamaBenchModelData {
        artifact_format,
        weights_resident_bytes_by_class,
        model_bytes_total,
        bytes_per_token_by_class,
        bytes_per_token_total,
        spec_roofline,
        measured_roofline,
    } = llama_bench_model_data(
        &root,
        &summary,
        &model_sha256,
        decode_tok_s.median,
        &machine,
    )?;
    let notes = llama_bench_notes(
        &summary,
        prefill_tok_s.is_some(),
        &machine,
        &capture.raw_json_sha256,
    );
    let bandwidth = bandwidth_provenance(&machine);

    let receipt = RuntimeReceipt {
        schema_version: RUNTIME_SCHEMA_VERSION,
        receipt_id: Uuid::new_v4(),
        created_utc: capture.captured_utc,
        machine,
        workload: Workload {
            engine: Engine {
                name: capture.engine.name,
                git_commit: pin,
                build_flags: capture.engine.build_flags,
            },
            model_artifact: ModelArtifact {
                path: logical_model_path(Path::new(&summary.model_filename))?,
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
            roofline_measured_achievable: measured_roofline,
            bandwidth: Some(bandwidth),
            memory: Some(unavailable_memory_telemetry(
                "llama-bench conversion has no process and system memory samples",
            )),
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
    capture: LlamaBenchCapture,
}

fn deserialize_boolish_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Bool(value) => Ok(u64::from(value)),
        serde_json::Value::Number(value) => value
            .as_u64()
            .ok_or_else(|| serde::de::Error::custom("expected a nonnegative integer")),
        _ => Err(serde::de::Error::custom("expected a boolean or integer")),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaBenchCapture {
    schema_version: u32,
    captured_utc: chrono::DateTime<Utc>,
    raw_json_sha256: String,
    model_path: String,
    model_sha256: String,
    model_sha256_before: String,
    model_file_bytes: u64,
    model_file_bytes_before: u64,
    backend: String,
    raw_cpu_info: String,
    raw_gpu_info: String,
    raw_backends: String,
    raw_n_gpu_layers: i64,
    raw_n_cpu_moe: i64,
    raw_fit_target: u64,
    raw_fit_min_ctx: u64,
    #[serde(deserialize_with = "deserialize_boolish_u64")]
    raw_embeddings: u64,
    raw_type_k: String,
    raw_type_v: String,
    raw_devices: String,
    raw_n_batch: u64,
    raw_n_ubatch: u64,
    #[serde(deserialize_with = "deserialize_boolish_u64")]
    raw_no_op_offload: u64,
    #[serde(deserialize_with = "deserialize_boolish_u64")]
    raw_no_kv_offload: u64,
    raw_tensor_buft_overrides: String,
    machine: Machine,
    engine: Engine,
}

fn load_llama_bench_source(
    json_path: &Path,
    capture_path: &Path,
    quality_ref: Option<Uuid>,
) -> Result<LlamaBenchSource, Box<dyn Error>> {
    let (summary, raw_json_sha256) = read_llama_bench_summary(json_path)?;
    let root = std::env::current_dir()?;
    let pin = read_llama_bench_pin(&root)?;
    validate_commit(&pin, &summary.build_commit)?;
    let (model_label, file_bytes, model_sha256) = read_llama_bench_model_identity(&root, &summary)?;
    let capture = read_llama_bench_capture(
        capture_path,
        &summary,
        &pin,
        &raw_json_sha256,
        &model_label,
        &model_sha256,
        file_bytes,
    )?;
    let linked_quality = load_linked_quality(&root, quality_ref, "llama.cpp", &pin, &model_sha256)?;
    Ok(LlamaBenchSource {
        root,
        summary,
        pin,
        file_bytes,
        model_sha256,
        linked_quality,
        capture,
    })
}

fn read_llama_bench_summary(
    json_path: &Path,
) -> Result<(BenchmarkSummary, String), Box<dyn Error>> {
    let bytes = fs::read(json_path)?;
    let entries: Vec<LlamaBenchEntry> = serde_json::from_slice(&bytes)?;
    Ok((
        BenchmarkSummary::from_entries(&entries)?,
        sha256_bytes(&bytes),
    ))
}

fn read_llama_bench_pin(root: &Path) -> Result<String, Box<dyn Error>> {
    Ok(fs::read_to_string(root.join("external/PINNED"))?
        .trim()
        .to_owned())
}

fn read_llama_bench_model(path: &Path) -> Result<(u64, String), Box<dyn Error>> {
    Ok((fs::metadata(path)?.len(), sha256_file(path)?))
}

fn read_llama_bench_model_identity(
    root: &Path,
    summary: &BenchmarkSummary,
) -> Result<(String, u64, String), Box<dyn Error>> {
    let model_label = logical_model_path(Path::new(&summary.model_filename))?;
    let model_path = resolve_model_path(root, Path::new(&model_label))?;
    let (file_bytes, model_sha256) = read_llama_bench_model(&model_path)?;
    let gguf = Gguf::open(&model_path)?;
    let model = ModelConfig::from_metadata(gguf.metadata())?;
    validate_llama_bench_gpu_layers(summary.backend, summary.n_gpu_layers, model.n_layer)?;
    let tensor_bytes = gguf_tensor_bytes(&gguf)?;
    validate_llama_bench_model_size(summary.model_size, tensor_bytes)?;
    Ok((model_label, file_bytes, model_sha256))
}

fn gguf_tensor_bytes(gguf: &Gguf) -> Result<u64, io::Error> {
    gguf.tensors().iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(tensor.n_bytes)
            .ok_or_else(|| invalid_data("GGUF tensor bytes overflowed"))
    })
}

fn read_llama_bench_capture(
    path: &Path,
    summary: &BenchmarkSummary,
    pin: &str,
    raw_json_sha256: &str,
    model_path: &str,
    model_sha256: &str,
    model_file_bytes: u64,
) -> Result<LlamaBenchCapture, Box<dyn Error>> {
    let capture: LlamaBenchCapture = serde_json::from_slice(&fs::read(path)?)?;
    validate_llama_bench_capture(
        &capture,
        summary,
        pin,
        raw_json_sha256,
        model_path,
        model_sha256,
        model_file_bytes,
    )?;
    Ok(capture)
}

fn validate_llama_bench_capture(
    capture: &LlamaBenchCapture,
    summary: &BenchmarkSummary,
    pin: &str,
    raw_json_sha256: &str,
    model_path: &str,
    model_sha256: &str,
    model_file_bytes: u64,
) -> Result<(), io::Error> {
    validate_capture_schema(capture)?;
    validate_capture_engine(capture, summary, pin)?;
    validate_capture_raw_hash(capture, raw_json_sha256)?;
    validate_capture_model(capture, model_path, model_sha256, model_file_bytes)?;
    validate_capture_backend(capture, summary)?;
    Ok(())
}

fn validate_capture_schema(capture: &LlamaBenchCapture) -> Result<(), io::Error> {
    if capture.schema_version == LLAMA_BENCH_CAPTURE_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(invalid_data(format!(
            "llama-bench capture schema must be {LLAMA_BENCH_CAPTURE_SCHEMA_VERSION}, got {}",
            capture.schema_version
        )))
    }
}

fn validate_capture_engine(
    capture: &LlamaBenchCapture,
    summary: &BenchmarkSummary,
    pin: &str,
) -> Result<(), io::Error> {
    if capture.engine.name != "llama.cpp" {
        return Err(invalid_data(format!(
            "llama-bench capture engine is {}, expected llama.cpp",
            capture.engine.name
        )));
    }
    validate_commit(pin, &capture.engine.git_commit)?;
    if capture.engine.git_commit == summary.build_commit {
        Ok(())
    } else {
        Err(invalid_data(
            "llama-bench capture commit does not match raw JSON",
        ))
    }
}

fn validate_capture_raw_hash(
    capture: &LlamaBenchCapture,
    raw_json_sha256: &str,
) -> Result<(), io::Error> {
    if capture.raw_json_sha256 == raw_json_sha256 {
        Ok(())
    } else {
        Err(invalid_data(
            "llama-bench capture does not match the raw benchmark JSON",
        ))
    }
}

fn validate_capture_model(
    capture: &LlamaBenchCapture,
    model_path: &str,
    model_sha256: &str,
    model_file_bytes: u64,
) -> Result<(), io::Error> {
    if capture.model_path == model_path
        && capture.model_sha256 == model_sha256
        && capture.model_sha256_before == model_sha256
        && capture.model_file_bytes == model_file_bytes
        && capture.model_file_bytes_before == model_file_bytes
    {
        Ok(())
    } else {
        Err(invalid_data(
            "llama-bench capture does not match the model artifact",
        ))
    }
}

fn validate_capture_backend(
    capture: &LlamaBenchCapture,
    summary: &BenchmarkSummary,
) -> Result<(), io::Error> {
    validate_capture_backend_identity(capture, summary)?;
    validate_capture_raw_execution(capture, summary)?;
    validate_capture_device(&capture.machine, summary, summary.backend)
}

fn validate_capture_backend_identity(
    capture: &LlamaBenchCapture,
    summary: &BenchmarkSummary,
) -> Result<(), io::Error> {
    if capture.machine.active_compute.is_none() {
        return Err(invalid_data(
            "llama-bench capture must include active compute metadata",
        ));
    }
    let active_backend = active_backend_name(&capture.machine)?;
    if capture.backend != backend_name(summary.backend) || capture.backend != active_backend {
        return Err(invalid_data(
            "llama-bench capture backend does not match raw benchmark execution",
        ));
    }
    Ok(())
}

fn validate_capture_raw_execution(
    capture: &LlamaBenchCapture,
    summary: &BenchmarkSummary,
) -> Result<(), io::Error> {
    let identity_matches = (
        capture.raw_cpu_info.as_str(),
        capture.raw_gpu_info.as_str(),
        capture.raw_backends.as_str(),
        capture.raw_n_gpu_layers,
        capture.raw_n_cpu_moe,
        capture.raw_fit_target,
        capture.raw_fit_min_ctx,
        capture.raw_embeddings,
        capture.raw_type_k.as_str(),
        capture.raw_type_v.as_str(),
        capture.raw_devices.as_str(),
    ) == (
        summary.cpu_info.as_str(),
        summary.gpu_info.as_str(),
        summary.backends.as_str(),
        summary.n_gpu_layers,
        summary.n_cpu_moe,
        summary.fit_target,
        summary.fit_min_ctx,
        summary.embeddings,
        summary.type_k.as_str(),
        summary.type_v.as_str(),
        summary.devices.as_str(),
    );
    let shape_matches = (
        capture.raw_n_batch,
        capture.raw_n_ubatch,
        capture.raw_no_op_offload,
        capture.raw_no_kv_offload,
        capture.raw_tensor_buft_overrides.as_str(),
    ) == (
        summary.n_batch,
        summary.n_ubatch,
        summary.no_op_offload,
        summary.no_kv_offload,
        summary.tensor_buft_overrides.as_str(),
    );
    if !(identity_matches && shape_matches) {
        return Err(invalid_data(
            "llama-bench capture execution metadata does not match raw JSON",
        ));
    }
    Ok(())
}

fn validate_capture_device(
    machine: &Machine,
    summary: &BenchmarkSummary,
    backend: BackendChoice,
) -> Result<(), io::Error> {
    let matches = match (backend, machine.active_compute.as_ref()) {
        (BackendChoice::Cpu, Some(ActiveCompute::Cpu { model, .. })) => model == &summary.cpu_info,
        (BackendChoice::Cuda, Some(ActiveCompute::Cuda { device, .. })) => {
            device == &summary.gpu_info
        }
        (BackendChoice::Metal, Some(ActiveCompute::Metal { chip, .. })) => {
            chip == &summary.gpu_info
        }
        _ => false,
    };
    if matches {
        validate_external_comparator_shader(machine, backend)
    } else {
        Err(invalid_data(
            "llama-bench capture device does not match raw benchmark execution",
        ))
    }
}

fn validate_external_comparator_shader(
    machine: &Machine,
    backend: BackendChoice,
) -> Result<(), io::Error> {
    if backend != BackendChoice::Metal {
        return Ok(());
    }
    match machine.active_compute.as_ref() {
        Some(ActiveCompute::Metal {
            shader_hash: Telemetry::Unavailable { reason },
            ..
        }) if !reason.trim().is_empty() => Ok(()),
        Some(ActiveCompute::Metal { .. }) => Err(invalid_data(
            "llama.cpp Metal capture must leave Leone shader metadata unavailable",
        )),
        _ => Err(invalid_data(
            "llama.cpp Metal capture has no Metal shader metadata state",
        )),
    }
}

fn public_capture_machine(machine: &Machine) -> Machine {
    let mut public = machine.clone();
    public.hostname = HOSTNAME_REDACTED.to_owned();
    public
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
            .map(|rate| {
                summary.prefill_tokens.unwrap_or(summary.context_tokens) as f64 / rate * 1_000.0
            })
            .collect::<Vec<_>>()
    });
    let ttft_ms = ttft_samples_ms
        .as_deref()
        .map(summarize_duration_samples_ms)
        .transpose()?;
    let prefill_method = prefill_tok_s
        .as_ref()
        .map(|_| ReceiptPrefillMethod::ExternalBatched);
    Ok(LlamaBenchRates {
        decode_tok_s,
        prefill_tok_s,
        ttft_context_tokens: ttft_ms
            .as_ref()
            .map(|_| summary.prefill_tokens.unwrap_or(summary.context_tokens)),
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
    measured_roofline: Option<Roofline>,
}

fn llama_bench_model_data(
    root: &Path,
    summary: &BenchmarkSummary,
    model_sha256: &str,
    decode_median: f64,
    machine: &Machine,
) -> Result<LlamaBenchModelData, Box<dyn Error>> {
    let model_path = resolve_model_path(root, Path::new(&summary.model_filename))?;
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
        spec_roofline: spec_roofline(machine, bytes_per_token_total, decode_median),
        measured_roofline: measured_roofline(machine, bytes_per_token_total, decode_median),
    })
}

fn llama_bench_notes(
    summary: &BenchmarkSummary,
    has_prefill: bool,
    machine: &Machine,
    raw_json_sha256: &str,
) -> Vec<String> {
    let roofline_note = match bandwidth_provenance(machine) {
        BandwidthProvenance::Specification { bandwidth_gbs, .. } => {
            format!("The spec roofline uses {bandwidth_gbs:.0} GB/s for the active CUDA device.")
        }
        BandwidthProvenance::Measured { .. } => {
            "The roofline uses the measured device bandwidth receipt.".to_owned()
        }
        BandwidthProvenance::Unavailable { .. } => {
            "The spec roofline is unavailable for the active device.".to_owned()
        }
    };
    let mut notes = vec![
        roofline_note,
        "Achievable eta is unverified without a matching device bandwidth receipt.".to_owned(),
        "The roofline denominator counts weights read, depth-dependent KV reads, and one embedding row per decode evaluation.".to_owned(),
        format!("Decode starts with {} KV cache entries.", summary.context_tokens),
        format!(
            "Comparator cache types are {} and {}.",
            summary.type_k, summary.type_v
        ),
        format!(
            "Comparator uses devices={}, n_batch={}, n_ubatch={}, n_gpu_layers={}, n_cpu_moe={}, no_op_offload={}, no_kv_offload={}, tensor_buft_overrides={}, fit_target={}, fit_min_ctx={}, embeddings={}.",
            summary.devices,
            summary.n_batch,
            summary.n_ubatch,
            summary.n_gpu_layers,
            summary.n_cpu_moe,
            summary.no_op_offload,
            summary.no_kv_offload,
            summary.tensor_buft_overrides,
            summary.fit_target,
            summary.fit_min_ctx,
            summary.embeddings
        ),
        format!("Raw llama-bench JSON SHA-256: {raw_json_sha256}."),
    ];
    if has_prefill {
        notes.push(format!(
            "Prefill measures {} tokens with external batched llama.cpp evaluation.",
            summary.prefill_tokens.unwrap_or(summary.context_tokens)
        ));
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

struct LlamaBenchReceiptOptions {
    capture_path: PathBuf,
    quality_ref: Option<Uuid>,
}

struct LlamaBenchCaptureOptions {
    output_path: PathBuf,
    backend: BackendChoice,
    model_sha256_before: String,
    model_file_bytes_before: u64,
}

fn parse_llama_bench_capture_options(
    arguments: &[String],
) -> Result<LlamaBenchCaptureOptions, io::Error> {
    let mut output_path = None;
    let mut backend = None;
    let mut model_sha256_before = None;
    let mut model_file_bytes_before = None;
    let mut index = 0;
    while index < arguments.len() {
        parse_capture_writer_argument(
            arguments,
            &mut index,
            &mut output_path,
            &mut backend,
            &mut model_sha256_before,
            &mut model_file_bytes_before,
        )?;
        index += 1;
    }
    finish_llama_bench_capture_options(
        output_path,
        backend,
        model_sha256_before,
        model_file_bytes_before,
    )
}

fn finish_llama_bench_capture_options(
    output_path: Option<PathBuf>,
    backend: Option<BackendChoice>,
    model_sha256_before: Option<String>,
    model_file_bytes_before: Option<u64>,
) -> Result<LlamaBenchCaptureOptions, io::Error> {
    Ok(LlamaBenchCaptureOptions {
        output_path: output_path.ok_or_else(|| invalid_data("capture output is required"))?,
        backend: backend.ok_or_else(|| invalid_data("capture backend is required"))?,
        model_sha256_before: model_sha256_before
            .ok_or_else(|| invalid_data("pre-benchmark model SHA-256 is required"))?,
        model_file_bytes_before: model_file_bytes_before
            .ok_or_else(|| invalid_data("pre-benchmark model size is required"))?,
    })
}

fn parse_capture_writer_argument(
    arguments: &[String],
    index: &mut usize,
    output_path: &mut Option<PathBuf>,
    backend: &mut Option<BackendChoice>,
    model_sha256_before: &mut Option<String>,
    model_file_bytes_before: &mut Option<u64>,
) -> Result<(), io::Error> {
    match arguments[*index].as_str() {
        "--output" => parse_output_option(arguments, index, output_path),
        "--backend" => parse_backend_option(arguments, index, backend),
        "--model-sha256-before" => {
            parse_model_sha256_before_option(arguments, index, model_sha256_before)
        }
        "--model-file-bytes-before" => {
            parse_model_file_bytes_before_option(arguments, index, model_file_bytes_before)
        }
        _ => Err(invalid_data(
            "receipt capture-llama-bench requires output, backend, and pre-benchmark model identity",
        )),
    }
}

fn parse_output_option(
    arguments: &[String],
    index: &mut usize,
    output_path: &mut Option<PathBuf>,
) -> Result<(), io::Error> {
    if output_path.is_some() {
        return Err(invalid_data("capture output was specified more than once"));
    }
    *index += 1;
    *output_path =
        Some(PathBuf::from(arguments.get(*index).ok_or_else(|| {
            invalid_data("--output is missing its path")
        })?));
    Ok(())
}

fn parse_backend_option(
    arguments: &[String],
    index: &mut usize,
    backend: &mut Option<BackendChoice>,
) -> Result<(), io::Error> {
    if backend.is_some() {
        return Err(invalid_data("capture backend was specified more than once"));
    }
    *index += 1;
    let value = arguments
        .get(*index)
        .ok_or_else(|| invalid_data("--backend is missing its value"))?;
    *backend = Some(match value.as_str() {
        "cuda" => BackendChoice::Cuda,
        "cpu" => BackendChoice::Cpu,
        "metal" => BackendChoice::Metal,
        _ => return Err(invalid_data(format!("capture backend is invalid: {value}"))),
    });
    Ok(())
}

fn parse_model_sha256_before_option(
    arguments: &[String],
    index: &mut usize,
    model_sha256_before: &mut Option<String>,
) -> Result<(), io::Error> {
    if model_sha256_before.is_some() {
        return Err(invalid_data(
            "pre-benchmark model SHA-256 was specified more than once",
        ));
    }
    *index += 1;
    *model_sha256_before = Some(
        arguments
            .get(*index)
            .ok_or_else(|| invalid_data("--model-sha256-before is missing its value"))?
            .to_owned(),
    );
    Ok(())
}

fn parse_model_file_bytes_before_option(
    arguments: &[String],
    index: &mut usize,
    model_file_bytes_before: &mut Option<u64>,
) -> Result<(), io::Error> {
    if model_file_bytes_before.is_some() {
        return Err(invalid_data(
            "pre-benchmark model size was specified more than once",
        ));
    }
    *index += 1;
    let value = arguments
        .get(*index)
        .ok_or_else(|| invalid_data("--model-file-bytes-before is missing its value"))?;
    *model_file_bytes_before = Some(
        value
            .parse()
            .map_err(|_| invalid_data(format!("pre-benchmark model size is invalid: {value}")))?,
    );
    Ok(())
}

fn parse_llama_bench_options(arguments: &[String]) -> Result<LlamaBenchReceiptOptions, io::Error> {
    let mut capture_path = None;
    let mut quality_ref = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--capture" => parse_capture_option(arguments, &mut index, &mut capture_path)?,
            "--quality-ref" => parse_quality_option(arguments, &mut index, &mut quality_ref)?,
            _ => {
                return Err(invalid_data(
                    "receipt from-llama-bench requires --capture <json> and accepts --quality-ref <uuid>",
                ));
            }
        }
        index += 1;
    }
    Ok(LlamaBenchReceiptOptions {
        capture_path: capture_path
            .ok_or_else(|| invalid_data("receipt from-llama-bench requires --capture <json>"))?,
        quality_ref,
    })
}

fn write_llama_bench_capture(
    json_path: &Path,
    options: &LlamaBenchCaptureOptions,
) -> Result<(), Box<dyn Error>> {
    let (summary, raw_json_sha256) = read_llama_bench_summary(json_path)?;
    let root = std::env::current_dir()?;
    let pin = read_llama_bench_pin(&root)?;
    validate_commit(&pin, &summary.build_commit)?;
    let capture = build_llama_bench_capture(&root, &summary, raw_json_sha256, options)?;
    fs::write(&options.output_path, serde_json::to_vec_pretty(&capture)?)?;
    println!("capture: {}", options.output_path.display());
    Ok(())
}

fn build_llama_bench_capture(
    root: &Path,
    summary: &BenchmarkSummary,
    raw_json_sha256: String,
    options: &LlamaBenchCaptureOptions,
) -> Result<LlamaBenchCapture, Box<dyn Error>> {
    let (model_path, model_file_bytes, model_sha256) =
        read_llama_bench_model_identity(root, summary)?;
    validate_capture_options(options, summary, &model_sha256, model_file_bytes)?;
    let mut machine = capture_machine(summary.backend)?;
    clear_external_comparator_shader_hash(&mut machine);
    validate_capture_device(&machine, summary, summary.backend)?;
    let backend = active_backend_name(&machine)?.to_owned();
    Ok(LlamaBenchCapture {
        schema_version: LLAMA_BENCH_CAPTURE_SCHEMA_VERSION,
        captured_utc: Utc::now(),
        raw_json_sha256,
        model_path,
        model_sha256,
        model_sha256_before: options.model_sha256_before.clone(),
        model_file_bytes,
        model_file_bytes_before: options.model_file_bytes_before,
        backend,
        raw_cpu_info: summary.cpu_info.clone(),
        raw_gpu_info: summary.gpu_info.clone(),
        raw_backends: summary.backends.clone(),
        raw_n_gpu_layers: summary.n_gpu_layers,
        raw_n_cpu_moe: summary.n_cpu_moe,
        raw_fit_target: summary.fit_target,
        raw_fit_min_ctx: summary.fit_min_ctx,
        raw_embeddings: summary.embeddings,
        raw_type_k: summary.type_k.clone(),
        raw_type_v: summary.type_v.clone(),
        raw_devices: summary.devices.clone(),
        raw_n_batch: summary.n_batch,
        raw_n_ubatch: summary.n_ubatch,
        raw_no_op_offload: summary.no_op_offload,
        raw_no_kv_offload: summary.no_kv_offload,
        raw_tensor_buft_overrides: summary.tensor_buft_overrides.clone(),
        machine,
        engine: Engine {
            name: "llama.cpp".to_owned(),
            git_commit: summary.build_commit.clone(),
            build_flags: vec!["capture=llama-bench".to_owned()],
        },
    })
}

fn clear_external_comparator_shader_hash(machine: &mut Machine) {
    if let Some(ActiveCompute::Metal { shader_hash, .. }) = machine.active_compute.as_mut() {
        *shader_hash = Telemetry::Unavailable {
            reason: "llama.cpp capture does not include Leone shader metadata".to_owned(),
        };
    }
}

fn validate_capture_options(
    options: &LlamaBenchCaptureOptions,
    summary: &BenchmarkSummary,
    model_sha256: &str,
    model_file_bytes: u64,
) -> Result<(), io::Error> {
    if options.model_sha256_before != model_sha256
        || options.model_file_bytes_before != model_file_bytes
    {
        return Err(invalid_data(
            "model changed between the pre-benchmark and capture checks",
        ));
    }
    if options.backend != summary.backend {
        return Err(invalid_data(
            "capture backend does not match raw benchmark execution",
        ));
    }
    Ok(())
}

fn validate_llama_bench_model_size(
    recorded_bytes: u64,
    actual_bytes: u64,
) -> Result<(), io::Error> {
    if recorded_bytes == actual_bytes {
        Ok(())
    } else {
        Err(invalid_data(format!(
            "llama-bench model_size is {recorded_bytes}, actual GGUF tensor bytes are {actual_bytes}"
        )))
    }
}

fn active_backend_name(machine: &Machine) -> Result<&'static str, io::Error> {
    match machine.active_compute.as_ref() {
        Some(ActiveCompute::Cuda { .. }) => Ok("cuda"),
        Some(ActiveCompute::Cpu { .. }) => Ok("cpu"),
        Some(ActiveCompute::Metal { .. }) => Ok("metal"),
        None => Err(invalid_data("capture machine has no active backend")),
    }
}

const fn backend_name(backend: BackendChoice) -> &'static str {
    match backend {
        BackendChoice::Cuda => "cuda",
        BackendChoice::Cpu => "cpu",
        BackendChoice::Metal => "metal",
    }
}

fn capture_machine(backend: BackendChoice) -> Result<Machine, Box<dyn Error>> {
    match backend {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            let backend = CudaBackend::new(0)?;
            query_cuda_machine(&backend)
        }
        #[cfg(not(feature = "cuda"))]
        BackendChoice::Cuda => {
            Err(invalid_data("capture CUDA requires a CUDA-enabled build").into())
        }
        BackendChoice::Cpu => query_cpu_machine(),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => {
            let backend = MetalBackend::new()?;
            query_metal_machine(&backend)
        }
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => {
            Err(invalid_data("capture Metal requires a Metal-enabled build").into())
        }
    }
}

fn parse_capture_option(
    arguments: &[String],
    index: &mut usize,
    capture_path: &mut Option<PathBuf>,
) -> Result<(), io::Error> {
    if capture_path.is_some() {
        return Err(invalid_data("receipt capture was specified more than once"));
    }
    *index += 1;
    *capture_path =
        Some(PathBuf::from(arguments.get(*index).ok_or_else(|| {
            invalid_data("--capture is missing its path")
        })?));
    Ok(())
}

fn parse_quality_option(
    arguments: &[String],
    index: &mut usize,
    quality_ref: &mut Option<Uuid>,
) -> Result<(), io::Error> {
    if quality_ref.is_some() {
        return Err(invalid_data(
            "quality receipt reference was specified more than once",
        ));
    }
    *index += 1;
    let value = arguments
        .get(*index)
        .ok_or_else(|| invalid_data("--quality-ref is missing its UUID"))?;
    *quality_ref = Some(
        Uuid::parse_str(value)
            .map_err(|_| invalid_data(format!("quality receipt UUID is invalid: {value}")))?,
    );
    Ok(())
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

#[cfg(feature = "cuda")]
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
    let inspection = inspect_gguf_model(&gguf)?;
    let by_dtype = tensor_dtype_bytes(&gguf)?;
    let by_class = tensor_class_bytes(&gguf)?;
    print_gguf_summary(path, &gguf, &inspection);
    print_tensor_storage(by_dtype);
    print_tensor_classes(&by_class);
    Ok(())
}

#[derive(Debug, PartialEq)]
enum GgufInspection {
    Runtime(Box<ModelConfig>),
    MetadataOnly(ArchitectureSupport),
}

fn inspect_gguf_model(gguf: &Gguf) -> Result<GgufInspection, Box<dyn Error>> {
    let architecture = gguf
        .metadata()
        .get("general.architecture")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            invalid_data("GGUF metadata key \"general.architecture\" is missing or not a string")
        })?;
    let support = leone_gguf::architecture_support(architecture);
    if support.metadata_probe && !support.text_runtime {
        return Ok(GgufInspection::MetadataOnly(support));
    }
    Ok(GgufInspection::Runtime(Box::new(
        ModelConfig::from_metadata(gguf.metadata())?,
    )))
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

fn print_gguf_summary(path: &Path, gguf: &Gguf, inspection: &GgufInspection) {
    println!("GGUF summary");
    println!("  file                 {}", path.display());
    println!("  version              {}", gguf.version());
    println!("  alignment            {}", gguf.alignment());
    println!("  metadata entries     {}", gguf.metadata().len());
    println!("  tensors              {}", gguf.tensors().len());
    match inspection {
        GgufInspection::Runtime(model) => print_runtime_summary(model),
        GgufInspection::MetadataOnly(architecture) => {
            println!("  architecture         {}", architecture.architecture);
            print_architecture_support(architecture);
            println!("  runtime config       unavailable (text runtime unsupported)");
        }
    }
}

fn print_runtime_summary(model: &ModelConfig) {
    println!("  architecture         {}", model.architecture);
    let architecture = leone_gguf::architecture_support(&model.architecture);
    print_architecture_support(&architecture);
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

fn print_architecture_support(architecture: &ArchitectureSupport) {
    println!("  architecture class   {}", architecture.class.name());
    println!("  metadata probe       {}", architecture.metadata_probe);
    println!("  text runtime         {}", architecture.text_runtime);
    println!("  vision runtime       {}", architecture.vision_runtime);
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
        let manifest: ArtifactManifest = toml::from_str(&text)?;
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
        leone::PrefillMethod::ChunkedGpu => ReceiptPrefillMethod::ChunkedGpu,
        leone::PrefillMethod::ChunkedCublasLtFp16 => ReceiptPrefillMethod::ChunkedCublasLtFp16,
        leone::PrefillMethod::TiledCublasLtFp16 => ReceiptPrefillMethod::TiledCublasLtFp16,
        leone::PrefillMethod::Reused => ReceiptPrefillMethod::Reused,
    }
}

#[cfg(feature = "cuda")]
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

fn spec_roofline(machine: &Machine, bytes_per_token: u64, decode_median: f64) -> Roofline {
    match bandwidth_provenance(machine) {
        BandwidthProvenance::Specification { bandwidth_gbs, .. } => {
            roofline(bandwidth_gbs, bytes_per_token, decode_median)
        }
        BandwidthProvenance::Measured { receipt } => {
            roofline(receipt.measured_gbs, bytes_per_token, decode_median)
        }
        BandwidthProvenance::Unavailable { .. } => Roofline {
            bandwidth_gbs_assumed: 0.0,
            ceiling_tok_s: 0.0,
            eta: 0.0,
            denominator_definition: None,
        },
    }
}

fn measured_roofline(
    machine: &Machine,
    bytes_per_token: u64,
    decode_median: f64,
) -> Option<Roofline> {
    match bandwidth_provenance(machine) {
        BandwidthProvenance::Measured { receipt } => Some(roofline(
            receipt.measured_gbs,
            bytes_per_token,
            decode_median,
        )),
        BandwidthProvenance::Specification { .. } | BandwidthProvenance::Unavailable { .. } => None,
    }
}

fn bandwidth_provenance(machine: &Machine) -> BandwidthProvenance {
    match machine.active_compute.as_ref() {
        Some(ActiveCompute::Cuda {
            bandwidth_receipt: Some(_),
            ..
        }) => BandwidthProvenance::Unavailable {
            reason: "measured bandwidth evidence writer is unavailable".to_owned(),
        },
        Some(ActiveCompute::Cuda { device, .. }) if device == "NVIDIA GeForce RTX 4090" => {
            BandwidthProvenance::Specification {
                device: device.clone(),
                bandwidth_gbs: SPEC_BANDWIDTH_GBS,
                source: "RTX 4090 specification".to_owned(),
            }
        }
        Some(ActiveCompute::Cuda { device, .. }) => BandwidthProvenance::Unavailable {
            reason: format!("no bandwidth specification is recorded for {device}"),
        },
        Some(ActiveCompute::Cpu { .. }) => BandwidthProvenance::Unavailable {
            reason: "CPU bandwidth telemetry is unavailable".to_owned(),
        },
        Some(ActiveCompute::Metal { chip, .. }) => BandwidthProvenance::Unavailable {
            reason: format!("no measured bandwidth receipt is linked for {chip}"),
        },
        None => BandwidthProvenance::Unavailable {
            reason: "active compute metadata is unavailable".to_owned(),
        },
    }
}

fn unavailable_memory_telemetry(reason: &str) -> MemoryTelemetry {
    MemoryTelemetry {
        process: unavailable_process_memory(reason),
        system: unavailable_system_memory(reason),
        backend_owned: unavailable_owned_memory(reason),
    }
}

fn backend_memory_telemetry(accounting: MemoryAccounting, host_reason: &str) -> MemoryTelemetry {
    let (process, system) = host_memory_telemetry(host_reason);
    MemoryTelemetry {
        process,
        system,
        backend_owned: owned_memory_telemetry(accounting),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn host_memory_telemetry(_host_reason: &str) -> (ProcessMemorySnapshot, SystemMemorySnapshot) {
    let Ok(snapshot) = leone_metal::host_memory_snapshot() else {
        return (
            unavailable_process_memory("macOS Mach host memory snapshot is unavailable"),
            unavailable_system_memory("macOS Mach host memory snapshot is unavailable"),
        );
    };
    (mac_process_memory(snapshot), mac_system_memory(snapshot))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn host_memory_telemetry(host_reason: &str) -> (ProcessMemorySnapshot, SystemMemorySnapshot) {
    (
        unavailable_process_memory(host_reason),
        unavailable_system_memory(host_reason),
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn mac_process_memory(snapshot: MetalHostMemorySnapshot) -> ProcessMemorySnapshot {
    ProcessMemorySnapshot {
        resident_bytes: measured_or_unavailable(
            snapshot.process_resident_bytes,
            "macOS task_info resident bytes are unavailable",
        ),
        virtual_bytes: measured_or_unavailable(
            snapshot.process_virtual_bytes,
            "macOS task_info virtual bytes are unavailable; virtual bytes do not count as memory savings",
        ),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn mac_system_memory(snapshot: MetalHostMemorySnapshot) -> SystemMemorySnapshot {
    SystemMemorySnapshot {
        used_bytes: measured_or_unavailable(
            snapshot.system_used_bytes,
            "macOS used memory estimate is unavailable or inconsistent",
        ),
        available_bytes: measured_or_unavailable(
            snapshot.system_available_bytes,
            "macOS available memory estimate is unavailable or inconsistent",
        ),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn measured_or_unavailable(value: Option<u64>, reason: &str) -> Telemetry<u64> {
    value.map_or_else(
        || Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
        |value| Telemetry::Measured { value },
    )
}

fn unavailable_process_memory(reason: &str) -> ProcessMemorySnapshot {
    ProcessMemorySnapshot {
        resident_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
        virtual_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
    }
}

fn unavailable_system_memory(reason: &str) -> SystemMemorySnapshot {
    SystemMemorySnapshot {
        used_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
        available_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
    }
}

fn unavailable_owned_memory(reason: &str) -> OwnedMemoryTelemetry {
    OwnedMemoryTelemetry {
        live_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
        peak_live_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
        reserved_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
        bytes_by_class: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
        budget: OwnedMemoryBudget::Unavailable {
            reason: reason.to_owned(),
        },
        peak_owned_and_reserved_bytes: Telemetry::Unavailable {
            reason: reason.to_owned(),
        },
    }
}

fn owned_memory_telemetry(accounting: MemoryAccounting) -> OwnedMemoryTelemetry {
    let bytes_by_class = MemoryClass::ALL
        .into_iter()
        .map(|class| {
            let stats = accounting.class(class);
            (
                class.name().to_owned(),
                OwnedMemoryClass {
                    live_bytes: stats.live_bytes,
                    peak_live_bytes: stats.peak_live_bytes,
                },
            )
        })
        .collect();
    OwnedMemoryTelemetry {
        live_bytes: Telemetry::Measured {
            value: accounting.live_bytes,
        },
        peak_live_bytes: Telemetry::Measured {
            value: accounting.peak_live_bytes,
        },
        reserved_bytes: Telemetry::Measured {
            value: accounting.reserved_bytes,
        },
        bytes_by_class: Telemetry::Measured {
            value: bytes_by_class,
        },
        budget: match accounting.budget {
            leone::MemoryBudget::Unlimited => OwnedMemoryBudget::Unlimited,
            leone::MemoryBudget::Bytes(bytes) => OwnedMemoryBudget::Limited { bytes: bytes.get() },
        },
        peak_owned_and_reserved_bytes: Telemetry::Measured {
            value: accounting.peak_owned_and_reserved_bytes,
        },
    }
}

fn public_artifact_path(path: &Path) -> Result<String, io::Error> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != "." && *name != "..")
        .ok_or_else(|| invalid_data("model path has no public filename"))?;
    let mut public = PathBuf::from("models");
    public.push(name);
    Ok(public.display().to_string())
}

fn logical_model_path(path: &Path) -> Result<String, io::Error> {
    let names = logical_model_components(path)?;
    let name = logical_model_filename(&names)?;
    Ok(format!("models/{name}"))
}

fn logical_model_components(path: &Path) -> Result<Vec<&str>, io::Error> {
    path.components()
        .map(|component| match component {
            Component::Normal(name) => name
                .to_str()
                .ok_or_else(|| invalid_data("benchmark model path is not UTF-8")),
            _ => Err(invalid_data(
                "benchmark model path must be relative and contain no parent components",
            )),
        })
        .collect()
}

fn logical_model_filename<'a>(names: &'a [&str]) -> Result<&'a str, io::Error> {
    let name = match names {
        [name] => *name,
        ["models", name] => *name,
        _ => {
            return Err(invalid_data(
                "benchmark model path must be a model filename or models/<filename>",
            ));
        }
    };
    if name.is_empty() || name == "." || name == ".." || name.contains('\\') {
        return Err(invalid_data("benchmark model filename is invalid"));
    }
    Ok(name)
}

fn resolve_model_path(root: &Path, path: &Path) -> Result<PathBuf, io::Error> {
    Ok(root.join(logical_model_path(path)?))
}

#[cfg(feature = "cuda")]
fn resolve_repo_relative_path(root: &Path, path: &Path, field: &str) -> Result<PathBuf, io::Error> {
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::RootDir
                    | Component::Prefix(_)
                    | Component::ParentDir
                    | Component::CurDir
            )
        })
        || path.to_str().is_some_and(|value| value.contains('\\'))
    {
        return Err(invalid_data(format!(
            "{field} path must be relative and contain no parent components"
        )));
    }
    Ok(root.join(path))
}

#[derive(Debug, Clone, Deserialize)]
struct LlamaBenchEntry {
    build_commit: String,
    cpu_info: String,
    gpu_info: String,
    backends: String,
    model_filename: String,
    model_size: u64,
    n_batch: u64,
    n_ubatch: u64,
    type_k: String,
    type_v: String,
    n_gpu_layers: i64,
    n_cpu_moe: i64,
    fit_target: u64,
    fit_min_ctx: u64,
    #[serde(deserialize_with = "deserialize_boolish_u64")]
    embeddings: u64,
    devices: String,
    #[serde(deserialize_with = "deserialize_boolish_u64")]
    no_op_offload: u64,
    #[serde(deserialize_with = "deserialize_boolish_u64")]
    no_kv_offload: u64,
    tensor_buft_overrides: String,
    n_prompt: u64,
    n_gen: u64,
    #[serde(default)]
    n_depth: u64,
    samples_ts: Vec<f64>,
}

#[derive(Debug, PartialEq)]
struct BenchmarkSummary {
    build_commit: String,
    cpu_info: String,
    gpu_info: String,
    backends: String,
    model_filename: String,
    model_size: u64,
    n_batch: u64,
    n_ubatch: u64,
    type_k: String,
    type_v: String,
    n_gpu_layers: i64,
    n_cpu_moe: i64,
    fit_target: u64,
    fit_min_ctx: u64,
    embeddings: u64,
    devices: String,
    no_op_offload: u64,
    no_kv_offload: u64,
    tensor_buft_overrides: String,
    backend: BackendChoice,
    context_tokens: u64,
    prefill_tokens: Option<u64>,
    generated_tokens: u64,
    prefill_samples: Option<Vec<f64>>,
    decode_samples: Vec<f64>,
}

impl BenchmarkSummary {
    fn from_entries(entries: &[LlamaBenchEntry]) -> Result<Self, io::Error> {
        let (decode, prefill) = benchmark_entries(entries)?;
        let context_tokens = entry_context_tokens(decode, prefill)?;
        let backend = llama_bench_backend(decode)?;

        Ok(Self {
            build_commit: decode.build_commit.clone(),
            cpu_info: decode.cpu_info.clone(),
            gpu_info: decode.gpu_info.clone(),
            backends: decode.backends.clone(),
            model_filename: decode.model_filename.clone(),
            model_size: decode.model_size,
            n_batch: decode.n_batch,
            n_ubatch: decode.n_ubatch,
            type_k: decode.type_k.clone(),
            type_v: decode.type_v.clone(),
            n_gpu_layers: decode.n_gpu_layers,
            n_cpu_moe: decode.n_cpu_moe,
            fit_target: decode.fit_target,
            fit_min_ctx: decode.fit_min_ctx,
            embeddings: decode.embeddings,
            devices: decode.devices.clone(),
            no_op_offload: decode.no_op_offload,
            no_kv_offload: decode.no_kv_offload,
            tensor_buft_overrides: decode.tensor_buft_overrides.clone(),
            backend,
            context_tokens,
            prefill_tokens: prefill.map(|entry| entry.n_prompt),
            generated_tokens: decode.n_gen,
            prefill_samples: prefill.map(|entry| entry.samples_ts.clone()),
            decode_samples: decode.samples_ts.clone(),
        })
    }
}

fn benchmark_entries(
    entries: &[LlamaBenchEntry],
) -> Result<(&LlamaBenchEntry, Option<&LlamaBenchEntry>), io::Error> {
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
    Ok((decode, prefill))
}

fn validate_prefill_entry(
    decode: &LlamaBenchEntry,
    prefill: Option<&LlamaBenchEntry>,
) -> Result<(), io::Error> {
    let Some(prefill) = prefill else {
        return Ok(());
    };
    let same_identity = (
        prefill.build_commit.as_str(),
        prefill.model_filename.as_str(),
        prefill.model_size,
        prefill.cpu_info.as_str(),
        prefill.gpu_info.as_str(),
        prefill.backends.as_str(),
    ) == (
        decode.build_commit.as_str(),
        decode.model_filename.as_str(),
        decode.model_size,
        decode.cpu_info.as_str(),
        decode.gpu_info.as_str(),
        decode.backends.as_str(),
    );
    let same_shape = (
        prefill.n_batch,
        prefill.n_ubatch,
        prefill.type_k.as_str(),
        prefill.type_v.as_str(),
        prefill.n_gpu_layers,
        prefill.n_cpu_moe,
    ) == (
        decode.n_batch,
        decode.n_ubatch,
        decode.type_k.as_str(),
        decode.type_v.as_str(),
        decode.n_gpu_layers,
        decode.n_cpu_moe,
    );
    let same_placement = (
        prefill.fit_target,
        prefill.fit_min_ctx,
        prefill.embeddings,
        prefill.devices.as_str(),
        prefill.no_op_offload,
        prefill.no_kv_offload,
        prefill.tensor_buft_overrides.as_str(),
    ) == (
        decode.fit_target,
        decode.fit_min_ctx,
        decode.embeddings,
        decode.devices.as_str(),
        decode.no_op_offload,
        decode.no_kv_offload,
        decode.tensor_buft_overrides.as_str(),
    );
    if !same_identity || !same_shape || !same_placement {
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

fn llama_bench_backend(entry: &LlamaBenchEntry) -> Result<BackendChoice, io::Error> {
    validate_cpu_identity(entry)?;
    validate_llama_bench_shape(entry)?;
    validate_llama_bench_cache_types(entry)?;
    if is_cpu_llama_bench_execution(entry)? {
        return Ok(BackendChoice::Cpu);
    }
    validate_gpu_llama_bench_execution(entry)?;
    active_llama_bench_gpu_backend(&entry.backends)
}

fn is_cpu_llama_bench_execution(entry: &LlamaBenchEntry) -> Result<bool, io::Error> {
    if entry.devices.eq_ignore_ascii_case("none") {
        return Ok(true);
    }
    if entry.n_gpu_layers == 0
        && !entry.backends.trim().is_empty()
        && !entry.backends.eq_ignore_ascii_case("CPU")
    {
        return Err(invalid_data(
            "llama-bench zero GPU layers with an accelerator backend is ambiguous",
        ));
    }
    if entry.backends.trim().is_empty() || entry.backends.eq_ignore_ascii_case("CPU") {
        return if entry.gpu_info.trim().is_empty() {
            Ok(true)
        } else {
            Err(invalid_data(
                "llama-bench reports CPU backends with a GPU identity",
            ))
        };
    }
    Ok(false)
}

fn validate_gpu_llama_bench_execution(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.no_op_offload != 0 {
        return Err(invalid_data(
            "llama-bench no_op_offload prevents a GPU execution claim",
        ));
    }
    if entry.no_kv_offload != 0 {
        return Err(invalid_data(
            "llama-bench no_kv_offload prevents a GPU KV traffic claim",
        ));
    }
    if entry.n_cpu_moe != 0 {
        return Err(invalid_data(
            "llama-bench n_cpu_moe prevents a full GPU placement claim",
        ));
    }
    validate_gpu_identity(entry)?;
    Ok(())
}

fn validate_llama_bench_gpu_layers(
    backend: BackendChoice,
    n_gpu_layers: i64,
    model_layers: u64,
) -> Result<(), io::Error> {
    if backend == BackendChoice::Cpu || n_gpu_layers < 0 {
        return Ok(());
    }
    let required_layers = model_layers
        .checked_add(1)
        .ok_or_else(|| invalid_data("model layer count overflowed"))?;
    let recorded_layers = u64::try_from(n_gpu_layers)
        .map_err(|_| invalid_data("llama-bench GPU layer count is invalid"))?;
    if recorded_layers >= required_layers {
        Ok(())
    } else {
        Err(invalid_data(format!(
            "llama-bench partial GPU offload is not supported for receipts: n_gpu_layers={n_gpu_layers}, model_layers={model_layers}"
        )))
    }
}

fn validate_llama_bench_shape(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    validate_batch_shape(entry)?;
    validate_device_selection(entry)?;
    validate_offload_flags(entry)?;
    validate_unavailable_modes(entry)?;
    validate_tensor_placement(entry)
}

fn validate_batch_shape(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.n_batch == 0 || entry.n_ubatch == 0 || entry.n_ubatch > entry.n_batch {
        return Err(invalid_data(
            "llama-bench batch dimensions must be nonzero with n_ubatch <= n_batch",
        ));
    }
    Ok(())
}

fn validate_device_selection(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.devices.trim().is_empty() {
        return Err(invalid_data("llama-bench entry has no device selection"));
    }
    Ok(())
}

fn validate_offload_flags(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.no_op_offload > 1 {
        return Err(invalid_data("llama-bench no_op_offload must be 0 or 1"));
    }
    if entry.no_kv_offload > 1 {
        return Err(invalid_data("llama-bench no_kv_offload must be 0 or 1"));
    }
    Ok(())
}

fn validate_unavailable_modes(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.fit_target != 0 || entry.fit_min_ctx != 0 {
        return Err(invalid_data(
            "llama-bench fitting parameters prevent a placement claim",
        ));
    }
    if entry.embeddings != 0 {
        return Err(invalid_data(
            "llama-bench embeddings mode is not supported for receipt conversion",
        ));
    }
    Ok(())
}

fn validate_tensor_placement(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if !entry.tensor_buft_overrides.eq_ignore_ascii_case("none") {
        return Err(invalid_data(
            "llama-bench tensor_buft_overrides prevents a placement claim",
        ));
    }
    Ok(())
}

fn validate_llama_bench_cache_types(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if !entry.type_k.eq_ignore_ascii_case("f16") || !entry.type_v.eq_ignore_ascii_case("f16") {
        return Err(invalid_data(
            "llama-bench cache types must be f16 for receipt conversion",
        ));
    }
    Ok(())
}

fn validate_cpu_identity(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.cpu_info.trim().is_empty() {
        Err(invalid_data("llama-bench entry has no CPU identity"))
    } else {
        Ok(())
    }
}

fn validate_gpu_identity(entry: &LlamaBenchEntry) -> Result<(), io::Error> {
    if entry.gpu_info.trim().is_empty() {
        return Err(invalid_data("GPU llama-bench entry has no GPU identity"));
    }
    if entry.gpu_info.split(',').count() != 1 {
        Err(invalid_data(
            "GPU llama-bench entry has more than one possible active device",
        ))
    } else {
        Ok(())
    }
}

fn active_llama_bench_gpu_backend(backends: &str) -> Result<BackendChoice, io::Error> {
    let names = backends
        .split(',')
        .map(str::trim)
        .filter(|backend| !backend.is_empty())
        .collect::<Vec<_>>();
    let cuda = names
        .iter()
        .filter(|backend| backend.eq_ignore_ascii_case("CUDA"))
        .count();
    let metal = names
        .iter()
        .filter(|backend| backend.eq_ignore_ascii_case("METAL"))
        .count();
    let blas = names
        .iter()
        .filter(|backend| backend.eq_ignore_ascii_case("BLAS"))
        .count();
    let supported = cuda + metal + blas;
    if supported != names.len() {
        return Err(invalid_data(format!(
            "llama-bench entry uses unsupported or multiple active backends: {backends}"
        )));
    }
    match (cuda, metal) {
        (1, 0) => Ok(BackendChoice::Cuda),
        (0, 1) => Ok(BackendChoice::Metal),
        _ => Err(invalid_data(format!(
            "llama-bench entry has no single supported active backend: {backends}"
        ))),
    }
}

fn entry_context_tokens(
    decode: &LlamaBenchEntry,
    prefill: Option<&LlamaBenchEntry>,
) -> Result<u64, io::Error> {
    if decode.n_depth == 0 {
        return Err(invalid_data(
            "llama-bench decode entry must report a nonzero n_depth",
        ));
    }
    if let Some(prefill) = prefill {
        validate_llama_bench_prefill_depth(prefill)?;
    }
    if let Some(prefill) = prefill {
        if prefill.n_prompt != decode.n_depth {
            return Err(invalid_data(
                "llama-bench prefill tokens must match decode n_depth",
            ));
        }
    }
    Ok(decode.n_depth)
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
    let valid_pin = pin.len() == 40 && pin.bytes().all(|byte| byte.is_ascii_hexdigit());
    let valid_measured =
        (7..=40).contains(&measured.len()) && measured.bytes().all(|byte| byte.is_ascii_hexdigit());
    if !valid_pin || !valid_measured || !pin.starts_with(measured) {
        return Err(invalid_data(format!(
            "llama-bench commit {measured} does not match external/PINNED {pin}"
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn query_cuda_machine(backend: &CudaBackend) -> Result<Machine, Box<dyn Error>> {
    let info = backend.device_info()?;
    let gpu_vram_mib = bytes_to_mib(info.total_global_mem)?;
    let compute_cap = format!("{}.{}", info.compute_major, info.compute_minor);
    let driver = cuda_version_label(info.driver_version, "driver")?;
    let cuda = cuda_version_label(info.runtime_version, "runtime")?;
    let cpu_model = query_cpu_model()?;
    Ok(Machine {
        hostname: HOSTNAME_REDACTED.to_owned(),
        gpu_name: info.name.clone(),
        gpu_vram_mib,
        compute_cap: compute_cap.clone(),
        driver: driver.clone(),
        cuda: cuda.clone(),
        cpu_model,
        ram_gib: query_ram_gib()?,
        gpu_clocks_mhz: GpuClocksMhz::default(),
        gpu_power_limit_w: 0.0,
        active_compute: Some(ActiveCompute::Cuda {
            device: info.name,
            vram_mib: Telemetry::Measured {
                value: gpu_vram_mib,
            },
            compute_cap,
            driver,
            cuda,
            clocks_mhz: Telemetry::Unavailable {
                reason: "CUDA runtime metadata does not expose current clocks".to_owned(),
            },
            power_limit_w: Telemetry::Unavailable {
                reason: "CUDA runtime metadata does not expose the power limit".to_owned(),
            },
            bandwidth_receipt: None,
        }),
    })
}

#[cfg(feature = "cuda")]
fn query_machine() -> Result<Machine, Box<dyn Error>> {
    let backend = CudaBackend::new(0)?;
    query_cuda_machine(&backend)
}

#[cfg(feature = "metal")]
fn query_metal_machine(backend: &MetalBackend) -> Result<Machine, Box<dyn Error>> {
    let info = backend.device_info();
    let metadata = backend.device_metadata()?;
    let accounting = backend.memory_accounting();
    let guidance = metal_telemetry(info.recommended_working_set, "recommended working set");
    let admitted = metal_admitted_budget(accounting.budget);
    let budget_override = metal_budget_override(accounting.budget, info.recommended_working_set);
    Ok(Machine::from_metal(MetalMachineMetadata {
        chip: metadata.device_name,
        os: metadata.os_version,
        unified_memory: true,
        metal_families: vec![metadata.architecture_name],
        working_set_guidance_bytes: guidance,
        max_buffer_length_bytes: metal_telemetry(info.max_buffer_length, "max buffer length"),
        admitted_budget_bytes: admitted,
        budget_override,
        shader_hash: Telemetry::Measured {
            value: metadata.shader_source_hash,
        },
    }))
}

#[cfg(feature = "metal")]
fn metal_telemetry(value: u64, name: &str) -> Telemetry<u64> {
    if value == 0 {
        Telemetry::Unavailable {
            reason: format!("Metal device metadata did not report {name}"),
        }
    } else {
        Telemetry::Measured { value }
    }
}

#[cfg(feature = "metal")]
fn metal_admitted_budget(budget: MemoryBudget) -> Telemetry<u64> {
    match budget {
        MemoryBudget::Unlimited => Telemetry::Unavailable {
            reason: "Metal backend has no explicit allocation budget".to_owned(),
        },
        MemoryBudget::Bytes(bytes) => Telemetry::Measured { value: bytes.get() },
    }
}

#[cfg(feature = "metal")]
fn metal_budget_override(
    budget: MemoryBudget,
    guidance: u64,
) -> Option<leone_receipt::BudgetOverride> {
    let MemoryBudget::Bytes(bytes) = budget else {
        return None;
    };
    if guidance != 0 && bytes.get() <= guidance {
        return None;
    }
    Some(leone_receipt::BudgetOverride {
        bytes: bytes.get(),
        reason: "configured Metal allocation budget".to_owned(),
    })
}

fn query_cpu_machine() -> Result<Machine, Box<dyn Error>> {
    let cpu_model = query_cpu_model()?;
    let logical_cpus = u64::try_from(std::thread::available_parallelism()?.get())?;
    Ok(Machine::from_cpu(
        cpu_model,
        Telemetry::Measured {
            value: logical_cpus,
        },
        query_ram_gib()?,
    ))
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

#[cfg(feature = "cuda")]
fn bytes_to_mib(bytes: usize) -> Result<u64, io::Error> {
    let bytes = u64::try_from(bytes).map_err(|_| invalid_data("device memory exceeds u64"))?;
    bytes
        .checked_add(524_288)
        .map(|rounded| rounded / 1_048_576)
        .filter(|mib| *mib > 0)
        .ok_or_else(|| invalid_data("device memory is zero or too large"))
}

#[cfg(feature = "cuda")]
fn cuda_version_label(version: i32, field: &str) -> Result<String, io::Error> {
    if version < 1_000 {
        return Err(invalid_data(format!(
            "CUDA returned invalid {field} version: {version}"
        )));
    }
    let major = version / 1_000;
    let minor = (version % 1_000) / 10;
    let patch = version % 10;
    if patch == 0 {
        Ok(format!("{major}.{minor}"))
    } else {
        Ok(format!("{major}.{minor}.{patch}"))
    }
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
            cpu_info: "Example CPU".to_owned(),
            gpu_info: String::new(),
            backends: "CPU".to_owned(),
            model_filename: "models/Qwen3-8B-Q4_K_M.gguf".to_owned(),
            model_size: 4_000_000_000,
            n_batch: 2048,
            n_ubatch: 512,
            type_k: "f16".to_owned(),
            type_v: "f16".to_owned(),
            n_gpu_layers: 0,
            n_cpu_moe: 0,
            fit_target: 0,
            fit_min_ctx: 0,
            embeddings: 0,
            devices: "none".to_owned(),
            no_op_offload: 0,
            no_kv_offload: 0,
            tensor_buft_overrides: "none".to_owned(),
            n_prompt,
            n_gen,
            n_depth,
            samples_ts,
        }
    }

    #[test]
    fn receipt_help_covers_capture_and_parent_commands() {
        assert!(dispatch_nested_help(&[
            "receipt".to_owned(),
            "capture-llama-bench".to_owned(),
        ]));
        assert!(dispatch_primary_help("receipt"));
    }

    #[test]
    fn decode_execution_requires_backend_support() {
        assert_eq!(
            select_decode_execution(false, true, false),
            DecodeExecution::Eager
        );
        assert_eq!(
            select_decode_execution(false, true, true),
            DecodeExecution::Graph
        );
    }

    #[test]
    fn parses_one_prefill_and_one_decode_entry() {
        let entries = vec![
            entry(512, 0, 0, vec![8_000.0; 5]),
            entry(0, 128, 512, vec![90.0, 95.0, 100.0, 105.0, 110.0]),
        ];
        let summary = BenchmarkSummary::from_entries(&entries).unwrap();
        assert_eq!(summary.context_tokens, 512);
        assert_eq!(summary.generated_tokens, 128);
        assert_eq!(
            summarize_samples(&summary.decode_samples).unwrap().median,
            100.0
        );
        assert_eq!(
            llama_bench_rates(&summary).unwrap().prefill_method,
            Some(ReceiptPrefillMethod::ExternalBatched)
        );
    }

    #[test]
    fn rejects_decode_depth_zero_even_with_a_prefill_entry() {
        let entries = vec![
            entry(512, 0, 0, vec![8_000.0; 5]),
            entry(0, 128, 0, vec![100.0; 5]),
        ];
        assert!(BenchmarkSummary::from_entries(&entries).is_err());
    }

    #[test]
    fn rejects_prefill_with_a_nonempty_initial_kv_cache() {
        let entries = vec![
            entry(512, 0, 1, vec![8_000.0; 5]),
            entry(0, 128, 512, vec![100.0; 5]),
        ];
        assert!(BenchmarkSummary::from_entries(&entries).is_err());
    }

    #[test]
    fn rejects_prefill_comparator_with_a_nonempty_initial_kv_cache() {
        let prefill = entry(512, 0, 1, vec![8_000.0; 5]);
        assert!(validate_llama_bench_prefill_depth(&prefill).is_err());
    }

    #[test]
    fn rejects_prefill_with_a_different_depth() {
        let entries = vec![
            entry(1_024, 0, 0, vec![8_000.0; 5]),
            entry(0, 128, 512, vec![100.0; 5]),
        ];
        assert!(BenchmarkSummary::from_entries(&entries).is_err());
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
        let pin = "abcdef012345678901234567890123456789abcd";
        assert!(validate_commit(pin, "abcdef0").is_ok());
        assert!(validate_commit(pin, pin).is_ok());
        assert!(validate_commit(pin, "abcdef").is_err());
        assert!(validate_commit(pin, "1234567").is_err());
    }

    #[test]
    fn llama_bench_receipt_requires_a_capture_sidecar() {
        let arguments = vec!["--quality-ref".to_owned(), Uuid::new_v4().to_string()];
        assert!(parse_llama_bench_options(&arguments).is_err());
        let arguments = vec![
            "--capture".to_owned(),
            "capture.json".to_owned(),
            "--quality-ref".to_owned(),
            Uuid::new_v4().to_string(),
        ];
        let options = parse_llama_bench_options(&arguments).unwrap();
        assert_eq!(options.capture_path, PathBuf::from("capture.json"));
        assert!(options.quality_ref.is_some());
    }

    #[test]
    fn llama_bench_capture_must_identify_the_captured_engine_and_backend() {
        let pin = "abcdef012345678901234567890123456789abcd";
        let measured = "abcdef0";
        let summary = BenchmarkSummary {
            build_commit: measured.to_owned(),
            cpu_info: "Example CPU".to_owned(),
            gpu_info: String::new(),
            backends: "CPU".to_owned(),
            model_filename: "models/model.gguf".to_owned(),
            model_size: 1,
            n_batch: 1,
            n_ubatch: 1,
            type_k: "f16".to_owned(),
            type_v: "f16".to_owned(),
            n_gpu_layers: 0,
            n_cpu_moe: 0,
            fit_target: 0,
            fit_min_ctx: 0,
            embeddings: 0,
            devices: "none".to_owned(),
            no_op_offload: 0,
            no_kv_offload: 0,
            tensor_buft_overrides: "none".to_owned(),
            backend: BackendChoice::Cpu,
            context_tokens: 1,
            prefill_tokens: None,
            generated_tokens: 1,
            prefill_samples: None,
            decode_samples: vec![1.0],
        };
        let capture = LlamaBenchCapture {
            schema_version: LLAMA_BENCH_CAPTURE_SCHEMA_VERSION,
            captured_utc: Utc::now(),
            raw_json_sha256: "a".repeat(64),
            model_path: "models/model.gguf".to_owned(),
            model_sha256: "b".repeat(64),
            model_sha256_before: "b".repeat(64),
            model_file_bytes: 1,
            model_file_bytes_before: 1,
            backend: "cpu".to_owned(),
            raw_cpu_info: "Example CPU".to_owned(),
            raw_gpu_info: String::new(),
            raw_backends: "CPU".to_owned(),
            raw_n_gpu_layers: 0,
            raw_n_cpu_moe: 0,
            raw_fit_target: 0,
            raw_fit_min_ctx: 0,
            raw_embeddings: 0,
            raw_type_k: "f16".to_owned(),
            raw_type_v: "f16".to_owned(),
            raw_devices: "none".to_owned(),
            raw_n_batch: 1,
            raw_n_ubatch: 1,
            raw_no_op_offload: 0,
            raw_no_kv_offload: 0,
            raw_tensor_buft_overrides: "none".to_owned(),
            machine: Machine::from_cpu(
                "Example CPU".to_owned(),
                Telemetry::Measured { value: 4 },
                1,
            ),
            engine: Engine {
                name: "llama.cpp".to_owned(),
                git_commit: measured.to_owned(),
                build_flags: Vec::new(),
            },
        };
        assert!(validate_llama_bench_capture(
            &capture,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_ok());

        let mut invalid = capture.clone();
        invalid.backend = "cuda".to_owned();
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.machine.active_compute = None;
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.model_sha256 = "c".repeat(64);
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.model_sha256_before = "c".repeat(64);
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.raw_no_kv_offload = 1;
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.raw_tensor_buft_overrides = "Q4_K=CPU".to_owned();
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.raw_n_gpu_layers = 1;
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.raw_n_cpu_moe = 1;
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            pin,
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());

        let mut invalid = capture.clone();
        invalid.raw_fit_target = 1;
        assert!(validate_llama_bench_capture(
            &invalid,
            &summary,
            "abcdef0",
            &"a".repeat(64),
            "models/model.gguf",
            &"b".repeat(64),
            1,
        )
        .is_err());
    }

    #[test]
    #[cfg(feature = "metal")]
    fn llama_bench_metal_capture_cannot_claim_leone_shader_metadata() {
        let mut machine = Machine::from_metal(MetalMachineMetadata {
            chip: "Apple M4".to_owned(),
            os: "macOS".to_owned(),
            unified_memory: true,
            metal_families: vec!["apple".to_owned()],
            working_set_guidance_bytes: Telemetry::Unavailable {
                reason: "test".to_owned(),
            },
            max_buffer_length_bytes: Telemetry::Unavailable {
                reason: "test".to_owned(),
            },
            admitted_budget_bytes: Telemetry::Unavailable {
                reason: "test".to_owned(),
            },
            budget_override: None,
            shader_hash: Telemetry::Measured {
                value: "a".repeat(64),
            },
        });
        assert!(validate_external_comparator_shader(&machine, BackendChoice::Metal).is_err());
        if let Some(ActiveCompute::Metal { shader_hash, .. }) = machine.active_compute.as_mut() {
            *shader_hash = Telemetry::Unavailable {
                reason: "llama.cpp capture has no Leone shader metadata".to_owned(),
            };
        }
        assert!(validate_external_comparator_shader(&machine, BackendChoice::Metal).is_ok());
    }

    #[test]
    fn public_artifact_path_keeps_only_a_logical_model_name() {
        assert_eq!(
            public_artifact_path(Path::new("/private/models/model.gguf")).unwrap(),
            "models/model.gguf"
        );
        assert!(public_artifact_path(Path::new("models/..")).is_err());
    }

    #[test]
    fn benchmark_input_paths_must_stay_within_the_models_label() {
        assert_eq!(
            logical_model_path(Path::new("Qwen3.gguf")).unwrap(),
            "models/Qwen3.gguf"
        );
        assert_eq!(
            logical_model_path(Path::new("models/Qwen3.gguf")).unwrap(),
            "models/Qwen3.gguf"
        );
        assert!(logical_model_path(Path::new("/tmp/Qwen3.gguf")).is_err());
        assert!(logical_model_path(Path::new("models/../Qwen3.gguf")).is_err());
    }

    #[test]
    fn llama_bench_model_size_must_match_gguf_tensor_bytes() {
        assert!(validate_llama_bench_model_size(4, 4).is_ok());
        assert!(validate_llama_bench_model_size(999_999, 4).is_err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn comparator_paths_must_be_repository_relative() {
        let root = Path::new("/tmp/repo");
        assert!(resolve_repo_relative_path(root, Path::new("external/PINNED"), "pin").is_ok());
        assert!(resolve_repo_relative_path(root, Path::new("/tmp/PINNED"), "pin").is_err());
        assert!(resolve_repo_relative_path(root, Path::new("../PINNED"), "pin").is_err());
    }

    #[test]
    fn llama_bench_backend_requires_raw_gpu_identity() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();
        assert_eq!(llama_bench_backend(&raw).unwrap(), BackendChoice::Cuda);

        raw.backends = "Metal,BLAS".to_owned();
        raw.gpu_info = "Apple M4".to_owned();
        assert_eq!(llama_bench_backend(&raw).unwrap(), BackendChoice::Metal);

        raw.backends = "CUDA,METAL".to_owned();
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn llama_bench_devices_none_prevents_a_gpu_claim() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "none".to_owned();
        assert_eq!(llama_bench_backend(&raw).unwrap(), BackendChoice::Cpu);
    }

    #[test]
    fn llama_bench_rejects_zero_layers_with_an_accelerator_backend() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.devices = "auto".to_owned();
        raw.n_gpu_layers = 0;
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn llama_bench_cpu_only_build_remains_cpu_with_default_layer_count() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CPU".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();
        assert_eq!(llama_bench_backend(&raw).unwrap(), BackendChoice::Cpu);
    }

    #[test]
    fn llama_bench_rejects_unverified_cache_and_offload_settings() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();
        raw.type_k = "q8_0".to_owned();
        assert!(llama_bench_backend(&raw).is_err());

        raw.type_k = "f16".to_owned();
        raw.no_op_offload = 1;
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn llama_bench_rejects_gpu_kv_offload_disabled() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();
        raw.no_kv_offload = 1;
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn llama_bench_rejects_tensor_placement_overrides() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();
        raw.tensor_buft_overrides = "Q4_K=CPU".to_owned();
        assert!(llama_bench_backend(&raw).is_err());

        raw.backends = "CPU".to_owned();
        raw.gpu_info.clear();
        raw.devices = "none".to_owned();
        raw.n_gpu_layers = 0;
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn llama_bench_rejects_cpu_moe_placement_on_gpu() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();
        raw.n_cpu_moe = 1;
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn llama_bench_rejects_fitting_and_embeddings_modes() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();

        raw.fit_target = 1;
        assert!(llama_bench_backend(&raw).is_err());
        raw.fit_target = 0;
        raw.fit_min_ctx = 1;
        assert!(llama_bench_backend(&raw).is_err());
        raw.fit_min_ctx = 0;
        raw.embeddings = 1;
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn llama_bench_rejects_partial_gpu_offload() {
        assert!(validate_llama_bench_gpu_layers(BackendChoice::Cuda, 1, 36).is_err());
        assert!(validate_llama_bench_gpu_layers(BackendChoice::Cuda, 37, 36).is_ok());
        assert!(validate_llama_bench_gpu_layers(BackendChoice::Cuda, 99, 36).is_ok());
        assert!(validate_llama_bench_gpu_layers(BackendChoice::Cuda, -1, 36).is_ok());
        assert!(validate_llama_bench_gpu_layers(BackendChoice::Cpu, 1, 36).is_ok());
    }

    #[test]
    fn llama_bench_rejects_unsupported_backend_mixes() {
        let mut raw = entry(0, 128, 512, vec![160.0; 5]);
        raw.backends = "CUDA,Vulkan".to_owned();
        raw.gpu_info = "NVIDIA GeForce RTX 4090".to_owned();
        raw.n_gpu_layers = -1;
        raw.devices = "auto".to_owned();
        assert!(llama_bench_backend(&raw).is_err());
    }

    #[test]
    fn prefill_and_decode_must_share_execution_shape() {
        let mut prefill = entry(512, 0, 0, vec![8_000.0; 5]);
        let decode = entry(0, 128, 512, vec![160.0; 5]);
        prefill.n_batch += 1;
        assert!(BenchmarkSummary::from_entries(&[prefill, decode]).is_err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn checked_in_benchmark_manifest_matches_the_cli_schema() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (_, prompt_tokens, decode_tokens) = load_bench_manifest(&root).unwrap();
        assert_eq!(prompt_tokens, 512);
        assert_eq!(decode_tokens, 128);
    }

    #[test]
    #[cfg(feature = "cuda")]
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

    #[test]
    fn gguf_inspect_reads_metadata_only_architectures_from_parsed_fixtures() {
        for architecture in ["qwen3moe", "qwen3vl", "qwen3next"] {
            let fixture = minimal_metadata_only_gguf(architecture);
            inspect_gguf(fixture.path()).expect("metadata-only GGUF inspection");
            let gguf = Gguf::open(fixture.path()).expect("parsed GGUF fixture");
            assert_eq!(gguf.tensors().len(), 1);
            assert!(ModelConfig::from_metadata(gguf.metadata()).is_err());
            let GgufInspection::MetadataOnly(support) =
                inspect_gguf_model(&gguf).expect("metadata-only support report")
            else {
                panic!("metadata-only architecture entered the runtime parser");
            };
            assert_eq!(support.architecture, architecture);
            assert!(support.metadata_probe);
            assert!(!support.text_runtime);
        }
    }

    fn minimal_metadata_only_gguf(architecture: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;

        let mut file = tempfile::NamedTempFile::new().expect("temporary GGUF");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        append_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(leone_gguf::ValueType::String as u32).to_le_bytes());
        append_gguf_string(&mut bytes, architecture);
        append_gguf_string(&mut bytes, "token_embedding");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&leone_gguf::GgmlType::F32.code().to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0; 4]);
        file.write_all(&bytes).expect("write GGUF");
        file.flush().expect("flush GGUF");
        file
    }

    #[test]
    #[cfg(any(not(feature = "cuda"), not(feature = "metal")))]
    fn generation_preflight_rejects_an_uncompiled_backend() {
        let missing_backend = if cfg!(feature = "cuda") {
            "metal"
        } else {
            "cuda"
        };
        let arguments = [
            "-m",
            "model.gguf",
            "-p",
            "prompt",
            "-n",
            "1",
            "--backend",
            missing_backend,
        ]
        .map(str::to_owned);
        let parsed = parse_generate(&arguments).expect("valid generation arguments");
        let error = validate_generation_options(&parsed).expect_err("uncompiled backend");
        assert!(error.to_string().contains(&format!(
            "the {missing_backend} backend is unavailable in this build"
        )));
    }

    fn append_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
