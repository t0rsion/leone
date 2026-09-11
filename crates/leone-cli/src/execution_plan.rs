use chrono::Utc;
use leone::{
    DecodeExecution, GenerateOptions, KvCacheDtype, MeasuredRate, Runtime,
    DEFAULT_PREFILL_CHUNK_TOKENS,
};
use leone_cuda::CudaBackend;
use leone_receipt::{sha256_bytes, sha256_file};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const PLAN_SCHEMA_VERSION: u32 = 1;
const SEARCH_RECEIPT_SCHEMA_VERSION: u32 = 1;
const DEFAULT_REPETITIONS: usize = 9;
const DEFAULT_PROMPT_TOKENS: usize = 4_096;
const DEFAULT_DECODE_TOKENS: usize = 64;
const MINIMUM_RELATIVE_IMPROVEMENT: f64 = 0.01;
const SIGN_TEST_ALPHA: f64 = 0.05;
const MAX_GPU_MEMORY_MIB: u64 = 24_564;
const MAX_GPU_POWER_W: f64 = 500.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlanDecode {
    Eager,
    Graph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanKv {
    Q8,
    F16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanSelection {
    pub decode: PlanDecode,
    pub kv: PlanKv,
    pub prefill_chunk_tokens: usize,
}

impl PlanSelection {
    pub fn apply(self, options: &mut GenerateOptions) {
        options.decode_execution = match self.decode {
            PlanDecode::Eager => DecodeExecution::Eager,
            PlanDecode::Graph => DecodeExecution::Graph,
        };
        options.kv_cache_dtype = self.kv_dtype();
        options.prefill_chunk_tokens = self.prefill_chunk_tokens;
    }

    const fn kv_dtype(self) -> KvCacheDtype {
        match self.kv {
            PlanKv::Q8 => KvCacheDtype::Q8,
            PlanKv::F16 => KvCacheDtype::F16,
            PlanKv::F32 => KvCacheDtype::F32,
        }
    }

    const fn decode_execution(self) -> DecodeExecution {
        match self.decode {
            PlanDecode::Eager => DecodeExecution::Eager,
            PlanDecode::Graph => DecodeExecution::Graph,
        }
    }

    fn id(self) -> String {
        format!(
            "{}-{}-c{}",
            match self.decode {
                PlanDecode::Eager => "eager",
                PlanDecode::Graph => "graph",
            },
            match self.kv {
                PlanKv::Q8 => "q8",
                PlanKv::F16 => "f16",
                PlanKv::F32 => "f32",
            },
            self.prefill_chunk_tokens
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionPlan {
    schema_version: u32,
    model: PlanModel,
    hardware: PlanHardware,
    selection: PlanSelection,
    evidence: PlanEvidence,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanModel {
    architecture: String,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanHardware {
    backend: String,
    compute_cap: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanEvidence {
    receipt_path: String,
    receipt_sha256: String,
    engine_commit: String,
    executable_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchReceipt {
    schema_version: u32,
    created_utc: String,
    engine_commit: String,
    executable_sha256: String,
    model: PlanModel,
    hardware: SearchHardware,
    preregistration: Preregistration,
    baseline: PlanSelection,
    candidates: Vec<CandidateEvidence>,
    selected: PlanSelection,
    promotion_status: PromotionStatus,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchHardware {
    gpu_name: String,
    compute_cap: String,
    memory_limit_mib: u64,
    power_limit_w: f64,
    cpu_mask: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preregistration {
    prompt_tokens: usize,
    decode_tokens: usize,
    repetitions: usize,
    aggregation: String,
    minimum_relative_improvement: f64,
    sign_test_alpha: f64,
    exact_transcript_required: bool,
    alternating_order: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateEvidence {
    selection: PlanSelection,
    paired_baseline_total_ms: Vec<f64>,
    candidate_total_ms: Vec<f64>,
    baseline_median_total_ms: f64,
    candidate_median_total_ms: f64,
    relative_improvement: f64,
    wins: usize,
    non_ties: usize,
    sign_test_p: f64,
    exact_transcript: bool,
    transcript_sha256: String,
    max_gpu_memory_mib: u64,
    max_gpu_power_w: f64,
    gate: CandidateGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum CandidateGate {
    Pass,
    RejectPerformance,
    RejectSignificance,
    RejectTranscript,
    RejectResources,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PromotionStatus {
    Promoted,
    BaselineRetained,
}

struct TuneArgs {
    model: PathBuf,
    output: PathBuf,
    repetitions: usize,
    prompt_tokens: usize,
    decode_tokens: usize,
}

#[derive(Clone, Copy)]
struct HardwareSample {
    memory_mib: u64,
    power_w: f64,
}

struct Trial {
    total_ms: f64,
    transcript_sha256: String,
}

pub fn load_optional(
    explicit: Option<&Path>,
    model: &Path,
    backend: &str,
) -> Result<Option<PlanSelection>, Box<dyn Error>> {
    let path = optional_plan_path(explicit);
    let Some(path) = path else {
        return Ok(None);
    };
    let plan = read_plan(&path)?;
    validate_plan(&plan)?;
    validate_plan_binding(&path, &plan, model, backend)?;
    println!("plan: {} ({})", path.display(), plan.selection.id());
    Ok(Some(plan.selection))
}

fn optional_plan_path(explicit: Option<&Path>) -> Option<PathBuf> {
    explicit
        .map(Path::to_owned)
        .or_else(|| std::env::var_os("LEONE_PLAN").map(PathBuf::from))
}

fn read_plan(path: &Path) -> Result<ExecutionPlan, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)
        .map_err(|error| invalid_data(format!("execution plan is invalid: {error}")))?)
}

fn validate_plan_binding(
    path: &Path,
    plan: &ExecutionPlan,
    model: &Path,
    backend: &str,
) -> Result<(), Box<dyn Error>> {
    validate_backend_binding(plan, backend)?;
    validate_model_binding(plan, model)?;
    validate_hardware_binding(plan)?;
    validate_evidence_binding(path, plan)?;
    Ok(())
}

fn validate_backend_binding(plan: &ExecutionPlan, backend: &str) -> Result<(), Box<dyn Error>> {
    if backend != plan.hardware.backend {
        return Err(invalid_data(format!(
            "execution plan requires backend {}, found {backend}",
            plan.hardware.backend
        ))
        .into());
    }
    Ok(())
}

fn validate_model_binding(plan: &ExecutionPlan, model: &Path) -> Result<(), Box<dyn Error>> {
    let model_sha256 = sha256_file(model)?;
    if model_sha256 != plan.model.sha256 {
        return Err(invalid_data(format!(
            "execution plan model SHA-256 is {}, found {model_sha256}",
            plan.model.sha256
        ))
        .into());
    }
    Ok(())
}

fn validate_hardware_binding(plan: &ExecutionPlan) -> Result<(), Box<dyn Error>> {
    let hardware = hardware_identity()?;
    if hardware.compute_cap != plan.hardware.compute_cap {
        return Err(invalid_data(format!(
            "execution plan requires compute capability {}, found {}",
            plan.hardware.compute_cap, hardware.compute_cap
        ))
        .into());
    }
    Ok(())
}

fn validate_evidence_binding(path: &Path, plan: &ExecutionPlan) -> Result<(), Box<dyn Error>> {
    let receipt_path = resolve_receipt_path(path, &plan.evidence.receipt_path);
    let receipt_bytes = fs::read(&receipt_path).map_err(|error| {
        invalid_data(format!(
            "execution plan evidence {} is unavailable: {error}",
            receipt_path.display()
        ))
    })?;
    let receipt_sha256 = sha256_bytes(&receipt_bytes);
    if receipt_sha256 != plan.evidence.receipt_sha256 {
        return Err(invalid_data(format!(
            "execution plan evidence SHA-256 is {}, found {receipt_sha256}",
            plan.evidence.receipt_sha256
        ))
        .into());
    }
    let receipt: SearchReceipt = serde_json::from_slice(&receipt_bytes)
        .map_err(|error| invalid_data(format!("plan search receipt is invalid: {error}")))?;
    if receipt.selected != plan.selection
        || receipt.model.sha256 != plan.model.sha256
        || receipt.hardware.compute_cap != plan.hardware.compute_cap
    {
        return Err(invalid_data("execution plan and search receipt disagree").into());
    }
    Ok(())
}

pub fn inspect(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let [path] = arguments else {
        return Err(invalid_data("inspect plan requires one JSON path").into());
    };
    let plan: ExecutionPlan = serde_json::from_slice(&fs::read(path)?)?;
    validate_plan(&plan)?;
    println!("schema: {}", plan.schema_version);
    println!("model: {} {}", plan.model.architecture, plan.model.sha256);
    println!(
        "target: {} sm{}",
        plan.hardware.backend,
        plan.hardware.compute_cap.replace('.', "")
    );
    println!("selection: {}", plan.selection.id());
    println!("evidence: {}", plan.evidence.receipt_path);
    println!("evidence sha256: {}", plan.evidence.receipt_sha256);
    Ok(())
}

pub fn tune(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse_tune(arguments)?;
    let setup = tune_setup(&arguments)?;
    let TuneSetup {
        hardware,
        model_sha256,
        engine_commit,
        executable_sha256,
        mut runtime,
        architecture,
    } = setup;
    validate_tune_context(&runtime, &arguments)?;
    let prompt = vec![1_u32; arguments.prompt_tokens];
    let (baseline, evidence) = run_tune_trials(&mut runtime, &prompt, &arguments)?;
    let (selected, promotion_status) = select_candidate(&evidence, baseline);
    let receipt = build_search_receipt(
        SearchReceiptInput {
            arguments: &arguments,
            hardware: &hardware,
            model_sha256: &model_sha256,
            engine_commit: &engine_commit,
            executable_sha256: &executable_sha256,
            architecture: &architecture,
        },
        baseline,
        evidence,
        selected,
        promotion_status,
    );
    write_tune_outputs(&arguments, &receipt, selected, engine_commit)
}

fn validate_tune_context(
    runtime: &Runtime<CudaBackend>,
    arguments: &TuneArgs,
) -> Result<(), Box<dyn Error>> {
    let requested_context = arguments
        .prompt_tokens
        .checked_add(arguments.decode_tokens)
        .ok_or_else(|| invalid_data("tune context length overflowed"))?;
    if requested_context > runtime.model().config().context_length {
        return Err(invalid_data(format!(
            "tune needs {requested_context} positions but the model supports {}",
            runtime.model().config().context_length
        ))
        .into());
    }
    Ok(())
}

fn run_tune_trials(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    arguments: &TuneArgs,
) -> Result<(PlanSelection, Vec<CandidateEvidence>), Box<dyn Error>> {
    let baseline = PlanSelection {
        decode: PlanDecode::Eager,
        kv: PlanKv::F16,
        prefill_chunk_tokens: DEFAULT_PREFILL_CHUNK_TOKENS,
    };
    let candidates = candidate_selections();
    println!("baseline: {}", baseline.id());
    let _ = run_trial(runtime, prompt, arguments.decode_tokens, baseline)?;
    let evidence = measure_candidates(
        runtime,
        prompt,
        arguments.decode_tokens,
        arguments.repetitions,
        baseline,
        &candidates,
    )?;
    Ok((baseline, evidence))
}

fn build_search_receipt(
    input: SearchReceiptInput<'_>,
    baseline: PlanSelection,
    evidence: Vec<CandidateEvidence>,
    selected: PlanSelection,
    promotion_status: PromotionStatus,
) -> SearchReceipt {
    let SearchReceiptInput {
        arguments,
        hardware,
        model_sha256,
        engine_commit,
        executable_sha256,
        architecture,
    } = input;
    SearchReceipt {
        schema_version: SEARCH_RECEIPT_SCHEMA_VERSION,
        created_utc: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        engine_commit: engine_commit.to_owned(),
        executable_sha256: executable_sha256.to_owned(),
        model: PlanModel {
            architecture: architecture.to_owned(),
            sha256: model_sha256.to_owned(),
        },
        hardware: SearchHardware {
            gpu_name: hardware.gpu_name.to_owned(),
            compute_cap: hardware.compute_cap.to_owned(),
            memory_limit_mib: MAX_GPU_MEMORY_MIB,
            power_limit_w: MAX_GPU_POWER_W,
            cpu_mask: "0-3,12-15".to_owned(),
        },
        preregistration: Preregistration {
            prompt_tokens: arguments.prompt_tokens,
            decode_tokens: arguments.decode_tokens,
            repetitions: arguments.repetitions,
            aggregation: "median paired total request time".to_owned(),
            minimum_relative_improvement: MINIMUM_RELATIVE_IMPROVEMENT,
            sign_test_alpha: SIGN_TEST_ALPHA,
            exact_transcript_required: true,
            alternating_order: true,
        },
        baseline,
        candidates: evidence,
        selected,
        promotion_status,
    }
}

struct SearchReceiptInput<'a> {
    arguments: &'a TuneArgs,
    hardware: &'a HardwareIdentity,
    model_sha256: &'a str,
    engine_commit: &'a str,
    executable_sha256: &'a str,
    architecture: &'a str,
}

fn write_tune_outputs(
    arguments: &TuneArgs,
    receipt: &SearchReceipt,
    selected: PlanSelection,
    engine_commit: String,
) -> Result<(), Box<dyn Error>> {
    let receipt_bytes = pretty_json(&receipt)?;
    let receipt_name = format!(
        "{}-plan-search-v1.json",
        Utc::now().format("%Y-%m-%dT%H-%M-%SZ")
    );
    let receipt_path = Path::new("receipts").join(receipt_name);
    atomic_write(&receipt_path, &receipt_bytes)?;
    let plan = ExecutionPlan {
        schema_version: PLAN_SCHEMA_VERSION,
        model: PlanModel {
            architecture: receipt.model.architecture.clone(),
            sha256: receipt.model.sha256.clone(),
        },
        hardware: PlanHardware {
            backend: "cuda".to_owned(),
            compute_cap: receipt.hardware.compute_cap.clone(),
        },
        selection: selected,
        evidence: PlanEvidence {
            receipt_path: receipt_path.display().to_string(),
            receipt_sha256: sha256_bytes(&receipt_bytes),
            engine_commit,
            executable_sha256: receipt.executable_sha256.clone(),
        },
    };
    atomic_write(&arguments.output, &pretty_json(&plan)?)?;
    println!("selected: {}", selected.id());
    println!("receipt: {}", receipt_path.display());
    println!("plan: {}", arguments.output.display());
    Ok(())
}

struct TuneSetup {
    hardware: HardwareIdentity,
    model_sha256: String,
    engine_commit: String,
    executable_sha256: String,
    runtime: Runtime<CudaBackend>,
    architecture: String,
}

fn tune_setup(arguments: &TuneArgs) -> Result<TuneSetup, Box<dyn Error>> {
    let hardware = hardware_identity()?;
    if hardware.compute_cap != "8.9" {
        return Err(invalid_data(format!(
            "tuning requires compute capability 8.9, found {}",
            hardware.compute_cap
        ))
        .into());
    }
    let model_sha256 = sha256_file(&arguments.model)?;
    let engine_commit = command_text("git", &["rev-parse", "HEAD"])?;
    let executable_sha256 = sha256_file(std::env::current_exe()?)?;
    let runtime = Runtime::load(CudaBackend::new(0)?, &arguments.model)?;
    let architecture = runtime.model().config().architecture.name().to_owned();
    Ok(TuneSetup {
        hardware,
        model_sha256,
        engine_commit,
        executable_sha256,
        runtime,
        architecture,
    })
}

fn candidate_selections() -> [PlanSelection; 6] {
    [
        PlanSelection {
            decode: PlanDecode::Graph,
            kv: PlanKv::F16,
            prefill_chunk_tokens: 1_024,
        },
        PlanSelection {
            decode: PlanDecode::Graph,
            kv: PlanKv::F16,
            prefill_chunk_tokens: 2_048,
        },
        PlanSelection {
            decode: PlanDecode::Graph,
            kv: PlanKv::F16,
            prefill_chunk_tokens: 4_096,
        },
        PlanSelection {
            decode: PlanDecode::Graph,
            kv: PlanKv::Q8,
            prefill_chunk_tokens: 2_048,
        },
        PlanSelection {
            decode: PlanDecode::Graph,
            kv: PlanKv::Q8,
            prefill_chunk_tokens: 4_096,
        },
        PlanSelection {
            decode: PlanDecode::Graph,
            kv: PlanKv::F32,
            prefill_chunk_tokens: 4_096,
        },
    ]
}

fn measure_candidates(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    decode_tokens: usize,
    repetitions: usize,
    baseline: PlanSelection,
    candidates: &[PlanSelection],
) -> Result<Vec<CandidateEvidence>, Box<dyn Error>> {
    let mut evidence = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        println!("candidate: {}", candidate.id());
        let _ = run_trial(runtime, prompt, decode_tokens, candidate)?;
        let result = measure_candidate(
            runtime,
            prompt,
            decode_tokens,
            repetitions,
            baseline,
            candidate,
        )?;
        println!(
            "  median {:.3} ms, improvement {:.3}%, wins {}/{}, p {:.6}, {:?}",
            result.candidate_median_total_ms,
            result.relative_improvement * 100.0,
            result.wins,
            result.non_ties,
            result.sign_test_p,
            result.gate,
        );
        evidence.push(result);
    }
    Ok(evidence)
}

fn select_candidate(
    evidence: &[CandidateEvidence],
    baseline: PlanSelection,
) -> (PlanSelection, PromotionStatus) {
    let selected = evidence
        .iter()
        .filter(|candidate| candidate.gate == CandidateGate::Pass)
        .min_by(|left, right| {
            left.candidate_median_total_ms
                .total_cmp(&right.candidate_median_total_ms)
        })
        .map(|candidate| candidate.selection)
        .unwrap_or(baseline);
    let promotion_status = if selected == baseline {
        PromotionStatus::BaselineRetained
    } else {
        PromotionStatus::Promoted
    };
    (selected, promotion_status)
}

fn measure_candidate(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    decode_tokens: usize,
    repetitions: usize,
    baseline: PlanSelection,
    candidate: PlanSelection,
) -> Result<CandidateEvidence, Box<dyn Error>> {
    let samples = collect_candidate_samples(
        runtime,
        prompt,
        decode_tokens,
        repetitions,
        baseline,
        candidate,
    )?;
    let CandidateSamples {
        baseline_samples,
        candidate_samples,
        transcript,
        exact_transcript,
        max_memory,
        max_power,
    } = samples;
    let baseline_median = median(&baseline_samples)?;
    let candidate_median = median(&candidate_samples)?;
    let relative_improvement = (baseline_median - candidate_median) / baseline_median;
    let wins = baseline_samples
        .iter()
        .zip(&candidate_samples)
        .filter(|(baseline, candidate)| candidate < baseline)
        .count();
    let non_ties = baseline_samples
        .iter()
        .zip(&candidate_samples)
        .filter(|(baseline, candidate)| candidate != baseline)
        .count();
    let sign_test_p = sign_test_p(wins, non_ties)?;
    let gate = candidate_gate(
        exact_transcript,
        max_memory,
        max_power,
        relative_improvement,
        sign_test_p,
    );
    Ok(CandidateEvidence {
        selection: candidate,
        paired_baseline_total_ms: baseline_samples,
        candidate_total_ms: candidate_samples,
        baseline_median_total_ms: baseline_median,
        candidate_median_total_ms: candidate_median,
        relative_improvement,
        wins,
        non_ties,
        sign_test_p,
        exact_transcript,
        transcript_sha256: transcript.unwrap_or_default(),
        max_gpu_memory_mib: max_memory,
        max_gpu_power_w: max_power,
        gate,
    })
}

struct CandidateSamples {
    baseline_samples: Vec<f64>,
    candidate_samples: Vec<f64>,
    transcript: Option<String>,
    exact_transcript: bool,
    max_memory: u64,
    max_power: f64,
}

fn collect_candidate_samples(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    decode_tokens: usize,
    repetitions: usize,
    baseline: PlanSelection,
    candidate: PlanSelection,
) -> Result<CandidateSamples, Box<dyn Error>> {
    let mut samples = CandidateSamples {
        baseline_samples: Vec::with_capacity(repetitions),
        candidate_samples: Vec::with_capacity(repetitions),
        transcript: None,
        exact_transcript: true,
        max_memory: 0,
        max_power: 0.0,
    };
    for repetition in 0..repetitions {
        let (baseline_run, candidate_run) = if repetition.is_multiple_of(2) {
            (
                run_trial(runtime, prompt, decode_tokens, baseline)?,
                run_trial(runtime, prompt, decode_tokens, candidate)?,
            )
        } else {
            let candidate_run = run_trial(runtime, prompt, decode_tokens, candidate)?;
            let baseline_run = run_trial(runtime, prompt, decode_tokens, baseline)?;
            (baseline_run, candidate_run)
        };
        let expected = samples
            .transcript
            .get_or_insert_with(|| baseline_run.transcript_sha256.clone());
        samples.exact_transcript &= baseline_run.transcript_sha256 == *expected
            && candidate_run.transcript_sha256 == *expected;
        samples.baseline_samples.push(baseline_run.total_ms);
        samples.candidate_samples.push(candidate_run.total_ms);
        let sample = hardware_sample()?;
        samples.max_memory = samples.max_memory.max(sample.memory_mib);
        samples.max_power = samples.max_power.max(sample.power_w);
    }
    Ok(samples)
}

fn candidate_gate(
    exact_transcript: bool,
    max_memory: u64,
    max_power: f64,
    relative_improvement: f64,
    sign_test_p: f64,
) -> CandidateGate {
    if !exact_transcript {
        CandidateGate::RejectTranscript
    } else if max_memory > MAX_GPU_MEMORY_MIB || max_power > MAX_GPU_POWER_W {
        CandidateGate::RejectResources
    } else if relative_improvement < MINIMUM_RELATIVE_IMPROVEMENT {
        CandidateGate::RejectPerformance
    } else if sign_test_p > SIGN_TEST_ALPHA {
        CandidateGate::RejectSignificance
    } else {
        CandidateGate::Pass
    }
}

fn run_trial(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    decode_tokens: usize,
    selection: PlanSelection,
) -> Result<Trial, Box<dyn Error>> {
    let run = runtime.benchmark_decode_with_prefill_chunk(
        prompt,
        decode_tokens,
        selection.kv_dtype(),
        selection.decode_execution(),
        selection.prefill_chunk_tokens,
    )?;
    let prefill = rate(run.prefill_rate(), "prefill")?;
    let decode = rate(run.decode_rate(), "decode")?;
    let total_seconds = prompt.len() as f64 / prefill + decode_tokens as f64 / decode;
    Ok(Trial {
        total_ms: total_seconds * 1_000.0,
        transcript_sha256: run.transcript_sha256,
    })
}

fn parse_tune(arguments: &[String]) -> Result<TuneArgs, io::Error> {
    let mut parsed = TuneBuilder::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_tune_path_option(&mut parsed, arguments, &mut index)?
            || parse_tune_count_option(&mut parsed, arguments, &mut index)?;
        if !handled {
            return Err(invalid_data(format!("tune argument is invalid: {flag}")));
        }
        index += 1;
    }
    parsed.finish()
}

#[derive(Default)]
struct TuneBuilder {
    model: Option<PathBuf>,
    output: Option<PathBuf>,
    repetitions: Option<usize>,
    prompt_tokens: Option<usize>,
    decode_tokens: Option<usize>,
}

impl TuneBuilder {
    fn finish(self) -> Result<TuneArgs, io::Error> {
        let repetitions = self.repetitions.unwrap_or(DEFAULT_REPETITIONS);
        let prompt_tokens = self.prompt_tokens.unwrap_or(DEFAULT_PROMPT_TOKENS);
        let decode_tokens = self.decode_tokens.unwrap_or(DEFAULT_DECODE_TOKENS);
        if !(5..=31).contains(&repetitions) || repetitions.is_multiple_of(2) {
            return Err(invalid_data(
                "tune repetitions must be an odd value from 5 through 31",
            ));
        }
        if prompt_tokens == 0 || decode_tokens < 2 {
            return Err(invalid_data(
                "tune needs a nonzero prompt and at least two decode tokens",
            ));
        }
        let model = self
            .model
            .ok_or_else(|| invalid_data("tune requires -m <gguf>"))?;
        let output = self.output.unwrap_or_else(|| {
            let stem = model
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("model");
            Path::new("plans").join(format!("{stem}-sm89.json"))
        });
        Ok(TuneArgs {
            model,
            output,
            repetitions,
            prompt_tokens,
            decode_tokens,
        })
    }
}

fn parse_tune_path_option(
    parsed: &mut TuneBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "-m" | "--model" => {
            *index += 1;
            parsed.model = Some(PathBuf::from(required(arguments, *index, "tune model")?));
            Ok(true)
        }
        "--out" => {
            *index += 1;
            parsed.output = Some(PathBuf::from(required(arguments, *index, "plan output")?));
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_tune_count_option(
    parsed: &mut TuneBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--repetitions" => {
            *index += 1;
            parsed.repetitions = Some(parse_usize(arguments, *index, "repetitions")?);
            Ok(true)
        }
        "--prompt-tokens" => {
            *index += 1;
            parsed.prompt_tokens = Some(parse_usize(arguments, *index, "prompt tokens")?);
            Ok(true)
        }
        "--decode-tokens" => {
            *index += 1;
            parsed.decode_tokens = Some(parse_usize(arguments, *index, "decode tokens")?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn validate_plan(plan: &ExecutionPlan) -> Result<(), io::Error> {
    validate_plan_version(plan)?;
    validate_plan_hardware(plan)?;
    validate_plan_model(plan)?;
    validate_plan_selection(plan)?;
    validate_plan_evidence(plan)?;
    Ok(())
}

fn validate_plan_version(plan: &ExecutionPlan) -> Result<(), io::Error> {
    if plan.schema_version != PLAN_SCHEMA_VERSION {
        return Err(invalid_data(format!(
            "execution plan schema {} is unsupported",
            plan.schema_version
        )));
    }
    Ok(())
}

fn validate_plan_hardware(plan: &ExecutionPlan) -> Result<(), io::Error> {
    if plan.hardware.backend != "cuda" || plan.hardware.compute_cap.is_empty() {
        return Err(invalid_data("execution plan hardware target is invalid"));
    }
    Ok(())
}

fn validate_plan_model(plan: &ExecutionPlan) -> Result<(), io::Error> {
    if plan.model.architecture.is_empty()
        || plan.model.sha256.len() != 64
        || !plan
            .model
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(invalid_data("execution plan model identity is invalid"));
    }
    Ok(())
}

fn validate_plan_selection(plan: &ExecutionPlan) -> Result<(), io::Error> {
    if plan.selection.prefill_chunk_tokens == 0 {
        return Err(invalid_data("execution plan prefill chunk must be nonzero"));
    }
    Ok(())
}

fn validate_plan_evidence(plan: &ExecutionPlan) -> Result<(), io::Error> {
    if plan.evidence.receipt_path.is_empty()
        || plan.evidence.receipt_sha256.len() != 64
        || plan.evidence.engine_commit.is_empty()
        || plan.evidence.executable_sha256.len() != 64
    {
        return Err(invalid_data("execution plan evidence binding is invalid"));
    }
    Ok(())
}

fn resolve_receipt_path(plan_path: &Path, receipt: &str) -> PathBuf {
    let receipt = PathBuf::from(receipt);
    if receipt.is_absolute() || receipt.exists() {
        return receipt;
    }
    plan_path
        .parent()
        .and_then(Path::parent)
        .or_else(|| plan_path.parent())
        .unwrap_or_else(|| Path::new("."))
        .join(receipt)
}

struct HardwareIdentity {
    gpu_name: String,
    compute_cap: String,
}

fn hardware_identity() -> Result<HardwareIdentity, io::Error> {
    let output = command_text(
        "nvidia-smi",
        &[
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
    )?;
    let (gpu_name, compute_cap) = output
        .split_once(',')
        .ok_or_else(|| invalid_data("nvidia-smi returned an invalid GPU identity"))?;
    Ok(HardwareIdentity {
        gpu_name: gpu_name.trim().to_owned(),
        compute_cap: compute_cap.trim().to_owned(),
    })
}

fn hardware_sample() -> Result<HardwareSample, io::Error> {
    let output = command_text(
        "nvidia-smi",
        &[
            "--query-gpu=memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
    )?;
    let (memory, power) = output
        .split_once(',')
        .ok_or_else(|| invalid_data("nvidia-smi returned an invalid resource sample"))?;
    Ok(HardwareSample {
        memory_mib: memory
            .trim()
            .parse()
            .map_err(|_| invalid_data("nvidia-smi memory sample is invalid"))?,
        power_w: power
            .trim()
            .parse()
            .map_err(|_| invalid_data("nvidia-smi power sample is invalid"))?,
    })
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String, io::Error> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(invalid_data(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| invalid_data(format!("{program} output is not UTF-8")))
}

fn rate(value: MeasuredRate, name: &str) -> Result<f64, io::Error> {
    match value {
        MeasuredRate::TokensPerSecond(value) if value.is_finite() && value > 0.0 => Ok(value),
        _ => Err(invalid_data(format!("{name} rate is unavailable"))),
    }
}

fn median(values: &[f64]) -> Result<f64, io::Error> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_data("median needs finite samples"));
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    Ok(values[values.len() / 2])
}

fn sign_test_p(wins: usize, trials: usize) -> Result<f64, io::Error> {
    if wins > trials || trials > 63 {
        return Err(invalid_data("sign-test counts are invalid"));
    }
    let numerator = (wins..=trials)
        .map(|successes| binomial(trials, successes))
        .sum::<u128>();
    let denominator = 1_u128 << trials;
    Ok(numerator as f64 / denominator as f64)
}

fn binomial(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);
    (0..k).fold(1_u128, |value, index| {
        value * (n - index) as u128 / (index + 1) as u128
    })
}

fn pretty_json(value: &impl Serialize) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), io::Error> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn required<'a>(arguments: &'a [String], index: usize, name: &str) -> Result<&'a str, io::Error> {
    arguments
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| invalid_data(format!("{name} is missing")))
}

fn parse_usize(arguments: &[String], index: usize, name: &str) -> Result<usize, io::Error> {
    let value = required(arguments, index, name)?;
    value
        .parse()
        .map_err(|_| invalid_data(format!("{name} is invalid: {value}")))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_test_requires_eight_of_nine_wins() {
        assert!(sign_test_p(8, 9).unwrap() <= SIGN_TEST_ALPHA);
        assert!(sign_test_p(7, 9).unwrap() > SIGN_TEST_ALPHA);
    }

    #[test]
    fn selection_identifiers_are_stable() {
        assert_eq!(
            PlanSelection {
                decode: PlanDecode::Graph,
                kv: PlanKv::Q8,
                prefill_chunk_tokens: 2_048,
            }
            .id(),
            "graph-q8-c2048"
        );
    }

    #[test]
    fn installed_plan_resolves_evidence_from_the_install_root() {
        let plan = Path::new("/opt/leone/share/leone/plans/model-sm89.json");
        assert_eq!(
            resolve_receipt_path(plan, "receipts/search.json"),
            Path::new("/opt/leone/share/leone/receipts/search.json")
        );
    }
}
