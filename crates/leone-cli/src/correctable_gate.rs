use chrono::Utc;
use leone::{
    correction_residual, exhaustive_progressive_oracle, select_draft, token_stream_sha256, verify,
    CorrectableDrafter, DecodeExecution, Distribution, GenerateOptions, Runtime, SamplerRng,
    Speculation, Verdict,
};
use leone_cuda::CudaBackend;
use leone_receipt::{sha256_file, Machine};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use uuid::Uuid;

const MANIFEST: &str = include_str!("../assets/correctable-manifest.toml");
const PROMPTS: &str = include_str!("../assets/correctable-prompts.tsv");
const ORACLE_TOLERANCE: f64 = 2e-12;

#[derive(Debug)]
struct Arguments {
    model: Option<PathBuf>,
    cases: u64,
    tokens: usize,
    warmups: usize,
    repetitions: usize,
    seed: u64,
    receipt: bool,
    release_doc: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    release: String,
    target_model: String,
    target_quantization: String,
    hardware: String,
    minimum_repeat_rich_geometric_mean_speedup: f64,
    minimum_all_suite_geometric_mean_speedup: f64,
    minimum_distributional_acceptance_ratio: f64,
    minimum_selected_plan_count: usize,
    worst_suite_speedup_gate: f64,
    maximum_controller_fraction: f64,
    randomized_oracle_cases: u64,
}

#[derive(Debug)]
struct Prompt<'a> {
    name: &'a str,
    class: &'a str,
    text: &'a str,
}

#[derive(Debug, Serialize, Deserialize)]
struct StudyReceipt {
    schema_version: u32,
    release: String,
    created_utc: String,
    git_commit: String,
    manifest_sha256: String,
    oracle: OracleRecord,
    progressive: ProgressiveRecord,
    runtime: Option<RuntimeStudy>,
    quality: Option<QualityStudy>,
    oracle_gate_passed: bool,
    release_gate_passed: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OracleRecord {
    cases: u64,
    seed: u64,
    tolerance: f64,
    reconstruction_mismatches: u64,
    deterministic_verdict_mismatches: u64,
    maximum_absolute_error: f64,
    distributional_acceptance_sum: f64,
    point_acceptance_sum: f64,
    distributional_acceptance_ratio: f64,
    minimum_distributional_acceptance_ratio: f64,
    elapsed_ms: f64,
    gate_passed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProgressiveRecord {
    q4_codes: u64,
    q4_mismatches: u64,
    q6_codes: u64,
    q6_mismatches: u64,
    gate_passed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RuntimeStudy {
    receipt_id: Uuid,
    quality_ref: Uuid,
    model_path: String,
    model_sha256: String,
    target_model: String,
    target_quantization: String,
    hardware: String,
    #[serde(default)]
    gpu_idle_gate_passed: bool,
    #[serde(default)]
    machine_before: Option<Machine>,
    #[serde(default)]
    machine_after: Option<Machine>,
    tokens: usize,
    warmups: usize,
    repetitions: usize,
    #[serde(default)]
    plain_tokens_emitted: u64,
    #[serde(default)]
    correctable_tokens_emitted: u64,
    #[serde(default)]
    plain_decode_evaluations: u64,
    #[serde(default)]
    correctable_decode_evaluations: u64,
    #[serde(default)]
    plain_decode_duration_ms: f64,
    #[serde(default)]
    correctable_decode_duration_ms: f64,
    suites: Vec<SuiteRecord>,
    repeat_rich_geometric_mean_speedup: f64,
    all_suite_geometric_mean_speedup: f64,
    worst_suite_speedup: f64,
    selected_plan_count: usize,
    selected_plan_rounds: [u64; 4],
    controller_fraction: f64,
    gates: RuntimeGates,
    gate_passed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SuiteRecord {
    name: String,
    class: String,
    plain_tok_s: Vec<f64>,
    correctable_tok_s: Vec<f64>,
    median_plain_tok_s: f64,
    median_correctable_tok_s: f64,
    speedup: f64,
    exact_greedy_tokens: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RuntimeGates {
    minimum_repeat_rich_geometric_mean_speedup: f64,
    minimum_all_suite_geometric_mean_speedup: f64,
    worst_suite_speedup: f64,
    minimum_selected_plan_count: usize,
    maximum_controller_fraction: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualityStudy {
    receipt_id: Uuid,
    runtime_ref: Uuid,
    model_sha256: String,
    comparisons: Vec<QualityComparison>,
    exact_greedy_output: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualityComparison {
    suite: String,
    plain_transcript_sha256: String,
    correctable_transcript_sha256: String,
    exact: bool,
}

struct PerformanceEvidence {
    runtime: RuntimeStudy,
    quality: QualityStudy,
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    if handle_release_document_request(arguments)? {
        return Ok(());
    }
    let manifest: Manifest = toml::from_str(MANIFEST)?;
    let arguments = parse(arguments, &manifest)?;
    let receipt = study_receipt(&arguments, manifest)?;
    let receipt_path = emit_receipt(&receipt, arguments.receipt)?;
    write_requested_release_doc(&arguments, &receipt, receipt_path.as_deref())?;
    enforce_gates(&receipt)
}

fn handle_release_document_request(arguments: &[String]) -> Result<bool, Box<dyn Error>> {
    let Some((receipt_path, document_path)) = release_document_request(arguments) else {
        return Ok(false);
    };
    let receipt: StudyReceipt = serde_json::from_slice(&fs::read(receipt_path)?)?;
    write_release_doc(&receipt, Path::new(document_path), Path::new(receipt_path))?;
    Ok(true)
}

fn study_receipt(
    arguments: &Arguments,
    manifest: Manifest,
) -> Result<StudyReceipt, Box<dyn Error>> {
    let (oracle, progressive, evidence) = run_study(arguments, &manifest)?;
    let oracle_gate_passed = oracle.gate_passed && progressive.gate_passed;
    let release_gate_passed = evidence
        .as_ref()
        .map(|evidence| oracle_gate_passed && evidence.runtime.gate_passed);
    build_study_receipt(
        manifest,
        oracle,
        progressive,
        evidence,
        oracle_gate_passed,
        release_gate_passed,
    )
}

fn run_study(
    arguments: &Arguments,
    manifest: &Manifest,
) -> Result<(OracleRecord, ProgressiveRecord, Option<PerformanceEvidence>), Box<dyn Error>> {
    let oracle = run_oracle(arguments.cases, arguments.seed, manifest)?;
    let progressive = progressive_record()?;
    let evidence = arguments
        .model
        .as_deref()
        .map(|model| run_performance(model, arguments, manifest))
        .transpose()?;
    Ok((oracle, progressive, evidence))
}

fn write_requested_release_doc(
    arguments: &Arguments,
    receipt: &StudyReceipt,
    receipt_path: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let Some(path) = arguments.release_doc.as_deref() else {
        return Ok(());
    };
    let receipt_path =
        receipt_path.ok_or_else(|| invalid_data("--release-doc requires --receipt"))?;
    write_release_doc(receipt, path, receipt_path)
}

fn release_document_request(arguments: &[String]) -> Option<(&str, &str)> {
    match arguments {
        [from, receipt_path, release_doc, document_path]
            if from == "--from-receipt" && release_doc == "--release-doc" =>
        {
            Some((receipt_path, document_path))
        }
        _ => None,
    }
}

fn progressive_record() -> Result<ProgressiveRecord, Box<dyn Error>> {
    let stats = exhaustive_progressive_oracle()?;
    Ok(ProgressiveRecord {
        q4_codes: stats.q4_codes,
        q4_mismatches: stats.q4_mismatches,
        q6_codes: stats.q6_codes,
        q6_mismatches: stats.q6_mismatches,
        gate_passed: stats.q4_mismatches == 0 && stats.q6_mismatches == 0,
    })
}

fn build_study_receipt(
    manifest: Manifest,
    oracle: OracleRecord,
    progressive: ProgressiveRecord,
    evidence: Option<PerformanceEvidence>,
    oracle_gate_passed: bool,
    release_gate_passed: Option<bool>,
) -> Result<StudyReceipt, Box<dyn Error>> {
    let (runtime, quality) = evidence
        .map(|evidence| (Some(evidence.runtime), Some(evidence.quality)))
        .unwrap_or((None, None));
    Ok(StudyReceipt {
        schema_version: 1,
        release: manifest.release,
        created_utc: Utc::now().to_rfc3339(),
        git_commit: git_commit()?,
        manifest_sha256: leone_receipt::sha256_bytes(MANIFEST.as_bytes()),
        oracle,
        progressive,
        runtime,
        quality,
        oracle_gate_passed,
        release_gate_passed,
    })
}

fn emit_receipt(
    receipt: &StudyReceipt,
    write_receipt: bool,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(receipt)?;
    bytes.push(b'\n');
    print!("{}", String::from_utf8(bytes.clone())?);
    if !write_receipt {
        return Ok(None);
    }
    fs::create_dir_all("receipts")?;
    let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    let path = PathBuf::from(format!("receipts/correctable-{timestamp}.json"));
    fs::write(&path, bytes)?;
    Ok(Some(path))
}

fn enforce_gates(receipt: &StudyReceipt) -> Result<(), Box<dyn Error>> {
    if !receipt.oracle_gate_passed {
        return Err(invalid_data("correctable oracle gate failed").into());
    }
    if receipt.release_gate_passed == Some(false) {
        return Err(invalid_data("correctable release gate failed").into());
    }
    Ok(())
}

fn parse(arguments: &[String], manifest: &Manifest) -> Result<Arguments, io::Error> {
    let mut parsed = ArgumentBuilder::new(manifest);
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_argument_path(&mut parsed, arguments, &mut index)?
            || parse_argument_count(&mut parsed, arguments, &mut index)?
            || parse_argument_flags(&mut parsed, arguments, &mut index)?;
        if !handled {
            return Err(invalid_data(format!(
                "verify correctable argument is invalid: {flag}"
            )));
        }
        index += 1;
    }
    Ok(parsed.finish())
}

struct ArgumentBuilder {
    parsed: Arguments,
}

impl ArgumentBuilder {
    fn new(manifest: &Manifest) -> Self {
        Self {
            parsed: Arguments {
                model: None,
                cases: manifest.randomized_oracle_cases,
                tokens: 128,
                warmups: 2,
                repetitions: 5,
                seed: 0x6c65_6f6e_655f_7638,
                receipt: false,
                release_doc: None,
            },
        }
    }

    fn finish(self) -> Arguments {
        self.parsed
    }
}

fn parse_argument_path(
    builder: &mut ArgumentBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "-m" | "--model" => {
            builder.parsed.model = Some(PathBuf::from(value(arguments, index)?));
            Ok(true)
        }
        "--release-doc" => {
            builder.parsed.release_doc = Some(PathBuf::from(value(arguments, index)?));
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_argument_count(
    builder: &mut ArgumentBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_argument_sizes(builder, arguments, index)? {
        return Ok(true);
    }
    parse_argument_repetitions(builder, arguments, index)
}

fn parse_argument_sizes(
    builder: &mut ArgumentBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] == "--cases" {
        builder.parsed.cases = positive_u64(value(arguments, index)?, "cases")?;
        return Ok(true);
    }
    if arguments[*index] == "--tokens" {
        builder.parsed.tokens = positive(value(arguments, index)?, "tokens")?;
        return Ok(true);
    }
    if arguments[*index] == "--warmups" {
        builder.parsed.warmups = positive(value(arguments, index)?, "warmups")?;
        return Ok(true);
    }
    Ok(false)
}

fn parse_argument_repetitions(
    builder: &mut ArgumentBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--repetitions" {
        return Ok(false);
    }
    builder.parsed.repetitions = positive(value(arguments, index)?, "repetitions")?;
    Ok(true)
}

fn parse_argument_flags(
    builder: &mut ArgumentBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--seed" => {
            let raw = value(arguments, index)?;
            builder.parsed.seed = raw
                .parse()
                .map_err(|_| invalid_data(format!("seed is invalid: {raw}")))?;
            Ok(true)
        }
        "--receipt" => {
            builder.parsed.receipt = true;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn write_release_doc(
    receipt: &StudyReceipt,
    path: &Path,
    receipt_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let (runtime, quality, receipt_name) = release_document_inputs(receipt, receipt_path)?;
    let output = release_document_text(receipt, runtime, quality, &receipt_name)?;
    fs::write(path, output)?;
    Ok(())
}

fn release_document_inputs<'a>(
    receipt: &'a StudyReceipt,
    receipt_path: &Path,
) -> Result<(&'a RuntimeStudy, &'a QualityStudy, String), Box<dyn Error>> {
    let runtime = receipt
        .runtime
        .as_ref()
        .ok_or_else(|| invalid_data("a release document requires a measured runtime study"))?;
    let quality = receipt
        .quality
        .as_ref()
        .ok_or_else(|| invalid_data("a release document requires linked quality evidence"))?;
    if receipt.release_gate_passed != Some(true) {
        return Err(invalid_data("a failed study cannot generate release evidence").into());
    }
    let receipt_name = receipt_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid_data("receipt filename is not UTF-8"))?
        .to_owned();
    Ok((runtime, quality, receipt_name))
}

fn release_document_text(
    receipt: &StudyReceipt,
    runtime: &RuntimeStudy,
    quality: &QualityStudy,
    receipt_name: &str,
) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    write_release_header(&mut output)?;
    write_generated_result(&mut output, runtime)?;
    write_correctness(&mut output, receipt, quality)?;
    write_evidence(&mut output, runtime, quality, receipt_name)?;
    write_reproduction(&mut output)?;
    write_limits(&mut output)?;
    Ok(output)
}

fn write_release_header(output: &mut String) -> Result<(), Box<dyn Error>> {
    writeln!(output, "# Correctable inference evidence\n")?;
    writeln!(
        output,
        "Closed-loop correctable inference passes the preregistered release gate.\n"
    )?;
    Ok(())
}

fn write_generated_result(
    output: &mut String,
    runtime: &RuntimeStudy,
) -> Result<(), Box<dyn Error>> {
    write_generated_header(output)?;
    write_generated_rows(output, runtime)?;
    write_generated_summary(output, runtime)
}

fn write_generated_header(output: &mut String) -> Result<(), Box<dyn Error>> {
    writeln!(output, "## Generated result\n")?;
    writeln!(
        output,
        "| Suite | Class | Plain tok/s | Correctable tok/s | Speedup | Exact greedy tokens |"
    )?;
    writeln!(output, "|---|---|---:|---:|---:|---|")?;
    Ok(())
}

fn write_generated_rows(output: &mut String, runtime: &RuntimeStudy) -> Result<(), Box<dyn Error>> {
    for suite in &runtime.suites {
        writeln!(
            output,
            "| {} | {} | {:.3} | {:.3} | {:.6}x | {} |",
            suite.name,
            suite.class,
            suite.median_plain_tok_s,
            suite.median_correctable_tok_s,
            suite.speedup,
            if suite.exact_greedy_tokens {
                "match"
            } else {
                "mismatch"
            }
        )?;
    }
    writeln!(output)?;
    Ok(())
}

fn write_generated_summary(
    output: &mut String,
    runtime: &RuntimeStudy,
) -> Result<(), Box<dyn Error>> {
    writeln!(
        output,
        "Repeat-rich geometric-mean speedup: {:.6}x. Gate: {:.2}x.",
        runtime.repeat_rich_geometric_mean_speedup,
        runtime.gates.minimum_repeat_rich_geometric_mean_speedup
    )?;
    writeln!(
        output,
        "All-suite geometric-mean speedup: {:.6}x. Gate: {:.2}x.",
        runtime.all_suite_geometric_mean_speedup,
        runtime.gates.minimum_all_suite_geometric_mean_speedup
    )?;
    writeln!(
        output,
        "Worst-suite speedup: {:.6}x. Gate: {:.2}x.",
        runtime.worst_suite_speedup, runtime.gates.worst_suite_speedup
    )?;
    writeln!(
        output,
        "Controller fraction: {:.6}%. Gate: {:.2}%.",
        runtime.controller_fraction * 100.0,
        runtime.gates.maximum_controller_fraction * 100.0
    )?;
    writeln!(
        output,
        "Selected proposal plans: {}. Gate: {}.\n",
        runtime.selected_plan_count, runtime.gates.minimum_selected_plan_count
    )?;
    Ok(())
}

fn write_correctness(
    output: &mut String,
    receipt: &StudyReceipt,
    quality: &QualityStudy,
) -> Result<(), Box<dyn Error>> {
    writeln!(output, "## Correctness\n")?;
    writeln!(
        output,
        "The FP64 oracle checked {} randomized distribution pairs with {} reconstruction mismatches and {} deterministic verdict mismatches.",
        receipt.oracle.cases,
        receipt.oracle.reconstruction_mismatches,
        receipt.oracle.deterministic_verdict_mismatches
    )?;
    writeln!(
        output,
        "Its maximum absolute reconstruction error was {:.17e}.",
        receipt.oracle.maximum_absolute_error
    )?;
    writeln!(
        output,
        "Distribution-valued acceptance divided by matched point acceptance was {:.6}x. Gate: {:.2}x.",
        receipt.oracle.distributional_acceptance_ratio,
        receipt.oracle.minimum_distributional_acceptance_ratio
    )?;
    writeln!(
        output,
        "The progressive oracle exhausted {} Q4 codes and {} Q6 codes with {} total mismatches.",
        receipt.progressive.q4_codes,
        receipt.progressive.q6_codes,
        receipt.progressive.q4_mismatches + receipt.progressive.q6_mismatches
    )?;
    writeln!(
        output,
        "All {} paired greedy transcripts match. The quality record links to the same model digest as the runtime record.\n",
        quality.comparisons.len()
    )?;
    Ok(())
}

fn write_evidence(
    output: &mut String,
    runtime: &RuntimeStudy,
    quality: &QualityStudy,
    receipt_name: &str,
) -> Result<(), Box<dyn Error>> {
    writeln!(output, "## Evidence\n")?;
    writeln!(
        output,
        "Receipt: [`{receipt_name}`](../receipts/{receipt_name})."
    )?;
    writeln!(output, "Runtime receipt ID: `{}`.", runtime.receipt_id)?;
    writeln!(output, "Quality receipt ID: `{}`.", quality.receipt_id)?;
    writeln!(output, "Model SHA-256: `{}`.", runtime.model_sha256)?;
    writeln!(output)?;
    Ok(())
}

fn write_reproduction(output: &mut String) -> Result<(), Box<dyn Error>> {
    writeln!(output, "## Reproduce\n")?;
    writeln!(output, "```console")?;
    writeln!(
        output,
        "taskset -c 16-31 cargo +1.92 build --release -p leone-cli"
    )?;
    writeln!(
        output,
        "taskset -c 0-3,12-15 target/release/leone verify correctable \\\n  -m models/Qwen3-8B-Q4_K_M.gguf --receipt"
    )?;
    writeln!(output, "```\n")?;
    writeln!(output, "Keep the RTX 4090 idle. The command reads the frozen manifest and prompt corpus compiled into the binary.\n")?;
    Ok(())
}

fn write_limits(output: &mut String) -> Result<(), Box<dyn Error>> {
    writeln!(output, "## Limits\n")?;
    writeln!(
        output,
        "- The performance claim covers the recorded Qwen3-8B Q4_K_M artifact on one RTX 4090."
    )?;
    writeln!(
        output,
        "- Other models and backends have the exact correction contract, but no release speed claim."
    )?;
    writeln!(output, "- Progressive Q4 and Q6 planes are a representation contract. The release makes no progressive-kernel speed claim.")?;
    writeln!(
        output,
        "- The proposal family uses committed token history. It is not a learned drafter."
    )?;
    Ok(())
}

fn run_oracle(cases: u64, seed: u64, manifest: &Manifest) -> Result<OracleRecord, Box<dyn Error>> {
    let started = Instant::now();
    let mut random = OracleRandom::new(seed);
    let accept_target = Distribution::from_probabilities(vec![0.75, 0.25])?;
    let accept_draft = Distribution::from_probabilities(vec![0.5, 0.5])?;
    let reject_target = Distribution::from_probabilities(vec![0.0, 1.0])?;
    let reject_draft = Distribution::from_probabilities(vec![1.0, 0.0])?;
    let totals = collect_oracle_totals(
        &mut random,
        cases,
        seed,
        &accept_target,
        &accept_draft,
        &reject_target,
        &reject_draft,
    )?;
    let OracleTotals {
        reconstruction_mismatches,
        deterministic_verdict_mismatches,
        maximum_absolute_error,
        distributional_acceptance_sum,
        point_acceptance_sum,
    } = totals;
    let distributional_acceptance_ratio = if point_acceptance_sum == 0.0 {
        f64::INFINITY
    } else {
        distributional_acceptance_sum / point_acceptance_sum
    };
    let gate_passed = cases == manifest.randomized_oracle_cases
        && reconstruction_mismatches == 0
        && deterministic_verdict_mismatches == 0
        && distributional_acceptance_ratio >= manifest.minimum_distributional_acceptance_ratio;
    Ok(OracleRecord {
        cases,
        seed,
        tolerance: ORACLE_TOLERANCE,
        reconstruction_mismatches,
        deterministic_verdict_mismatches,
        maximum_absolute_error,
        distributional_acceptance_sum,
        point_acceptance_sum,
        distributional_acceptance_ratio,
        minimum_distributional_acceptance_ratio: manifest.minimum_distributional_acceptance_ratio,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        gate_passed,
    })
}

struct OracleTotals {
    reconstruction_mismatches: u64,
    deterministic_verdict_mismatches: u64,
    maximum_absolute_error: f64,
    distributional_acceptance_sum: f64,
    point_acceptance_sum: f64,
}

fn collect_oracle_totals(
    random: &mut OracleRandom,
    cases: u64,
    seed: u64,
    accept_target: &Distribution,
    accept_draft: &Distribution,
    reject_target: &Distribution,
    reject_draft: &Distribution,
) -> Result<OracleTotals, Box<dyn Error>> {
    let mut totals = OracleTotals {
        reconstruction_mismatches: 0,
        deterministic_verdict_mismatches: 0,
        maximum_absolute_error: 0.0,
        distributional_acceptance_sum: 0.0,
        point_acceptance_sum: 0.0,
    };
    for case in 0..cases {
        let sample = oracle_case(
            random,
            seed,
            case,
            accept_target,
            accept_draft,
            reject_target,
            reject_draft,
        )?;
        totals.maximum_absolute_error = totals
            .maximum_absolute_error
            .max(sample.maximum_absolute_error);
        totals.reconstruction_mismatches += u64::from(sample.reconstruction_mismatch);
        totals.distributional_acceptance_sum += sample.overlap;
        totals.point_acceptance_sum += sample.point_acceptance;
        totals.deterministic_verdict_mismatches += u64::from(sample.verdict_mismatch);
    }
    Ok(totals)
}

struct OracleCase {
    maximum_absolute_error: f64,
    reconstruction_mismatch: bool,
    overlap: f64,
    point_acceptance: f64,
    verdict_mismatch: bool,
}

fn oracle_case(
    random: &mut OracleRandom,
    seed: u64,
    case: u64,
    accept_target: &Distribution,
    accept_draft: &Distribution,
    reject_target: &Distribution,
    reject_draft: &Distribution,
) -> Result<OracleCase, Box<dyn Error>> {
    let vocab = 2 + random.index(15);
    let target = Distribution::from_probabilities(random.weights(vocab))?;
    let draft = Distribution::from_probabilities(random.weights(vocab))?;
    let residual = correction_residual(&target, &draft)?;
    let overlap = target
        .probabilities()
        .iter()
        .zip(draft.probabilities())
        .map(|(target, draft)| target.min(*draft))
        .sum::<f64>();
    let (maximum_absolute_error, reconstruction_mismatch) =
        reconstruction_result(&target, &draft, &residual, 1.0 - overlap);
    let drafted = select_draft(&draft, SamplerRng::new(seed ^ case), case)? as usize;
    let verdict_mismatch = deterministic_verdict_mismatch(
        case,
        accept_target,
        accept_draft,
        reject_target,
        reject_draft,
        seed,
    )?;
    Ok(OracleCase {
        maximum_absolute_error,
        reconstruction_mismatch,
        overlap,
        point_acceptance: target.probabilities()[drafted],
        verdict_mismatch,
    })
}

fn reconstruction_result(
    target: &Distribution,
    draft: &Distribution,
    residual: &Distribution,
    rejection_mass: f64,
) -> (f64, bool) {
    let mut maximum_absolute_error = 0.0_f64;
    let mut mismatch = false;
    for index in 0..target.probabilities().len() {
        let reconstructed = target.probabilities()[index].min(draft.probabilities()[index])
            + rejection_mass * residual.probabilities()[index];
        let error = (reconstructed - target.probabilities()[index]).abs();
        maximum_absolute_error = maximum_absolute_error.max(error);
        mismatch |= error > ORACLE_TOLERANCE;
    }
    (maximum_absolute_error, mismatch)
}

fn deterministic_verdict_mismatch(
    case: u64,
    accept_target: &Distribution,
    accept_draft: &Distribution,
    reject_target: &Distribution,
    reject_draft: &Distribution,
    seed: u64,
) -> Result<bool, Box<dyn Error>> {
    let (target, draft, expected) = if case & 1 == 0 {
        (accept_target, accept_draft, Verdict::Accept)
    } else {
        (reject_target, reject_draft, Verdict::Reject { token: 1 })
    };
    let verdict = verify(target, draft, 0, SamplerRng::new(seed ^ case), case)?;
    Ok(verdict != expected)
}

fn run_performance(
    model: &Path,
    arguments: &Arguments,
    manifest: &Manifest,
) -> Result<PerformanceEvidence, Box<dyn Error>> {
    let PerformanceSetup {
        mut runtime,
        machine_before,
        prompts,
        model_sha256,
        runtime_id,
        quality_id,
    } = prepare_performance(model)?;
    let mut suites = Vec::with_capacity(prompts.len());
    let mut comparisons = Vec::with_capacity(prompts.len());
    let mut totals = PerformanceTotals {
        plan_rounds: [0; 4],
        controller_duration: 0.0,
        correctable_duration: 0.0,
        plain_tokens_emitted: 0,
        correctable_tokens_emitted: 0,
        plain_decode_evaluations: 0,
        correctable_decode_evaluations: 0,
        plain_decode_duration: 0.0,
    };
    for prompt in prompts {
        let evidence = measure_prompt(&mut runtime, prompt, arguments, &mut totals)?;
        suites.push(evidence.suite);
        comparisons.push(evidence.comparison);
    }
    let PerformanceSummary {
        repeat_geomean,
        all_geomean,
        worst,
        selected_plan_count,
        controller_fraction,
        exact_output,
        gate_passed,
    } = summarize_performance(&suites, &comparisons, &totals, manifest);
    ensure_gpu_idle()?;
    let machine_after = super::query_machine()?;
    let runtime_receipt = RuntimeStudy {
        receipt_id: runtime_id,
        quality_ref: quality_id,
        model_path: model.display().to_string(),
        model_sha256: model_sha256.clone(),
        target_model: manifest.target_model.clone(),
        target_quantization: manifest.target_quantization.clone(),
        hardware: manifest.hardware.clone(),
        gpu_idle_gate_passed: true,
        machine_before: Some(machine_before),
        machine_after: Some(machine_after),
        tokens: arguments.tokens,
        warmups: arguments.warmups,
        repetitions: arguments.repetitions,
        plain_tokens_emitted: totals.plain_tokens_emitted,
        correctable_tokens_emitted: totals.correctable_tokens_emitted,
        plain_decode_evaluations: totals.plain_decode_evaluations,
        correctable_decode_evaluations: totals.correctable_decode_evaluations,
        plain_decode_duration_ms: totals.plain_decode_duration * 1_000.0,
        correctable_decode_duration_ms: totals.correctable_duration * 1_000.0,
        suites,
        repeat_rich_geometric_mean_speedup: repeat_geomean,
        all_suite_geometric_mean_speedup: all_geomean,
        worst_suite_speedup: worst,
        selected_plan_count,
        selected_plan_rounds: totals.plan_rounds,
        controller_fraction,
        gates: RuntimeGates {
            minimum_repeat_rich_geometric_mean_speedup: manifest
                .minimum_repeat_rich_geometric_mean_speedup,
            minimum_all_suite_geometric_mean_speedup: manifest
                .minimum_all_suite_geometric_mean_speedup,
            worst_suite_speedup: manifest.worst_suite_speedup_gate,
            minimum_selected_plan_count: manifest.minimum_selected_plan_count,
            maximum_controller_fraction: manifest.maximum_controller_fraction,
        },
        gate_passed,
    };
    let quality = QualityStudy {
        receipt_id: quality_id,
        runtime_ref: runtime_id,
        model_sha256,
        comparisons,
        exact_greedy_output: exact_output,
    };
    Ok(PerformanceEvidence {
        runtime: runtime_receipt,
        quality,
    })
}

struct PerformanceSetup {
    runtime: Runtime<CudaBackend>,
    machine_before: Machine,
    prompts: Vec<Prompt<'static>>,
    model_sha256: String,
    runtime_id: Uuid,
    quality_id: Uuid,
}

fn prepare_performance(model: &Path) -> Result<PerformanceSetup, Box<dyn Error>> {
    ensure_gpu_idle()?;
    Ok(PerformanceSetup {
        runtime: Runtime::load(CudaBackend::new(0)?, model)?,
        machine_before: super::query_machine()?,
        prompts: prompts()?,
        model_sha256: sha256_file(model)?,
        runtime_id: Uuid::new_v4(),
        quality_id: Uuid::new_v4(),
    })
}

struct PerformanceSummary {
    repeat_geomean: f64,
    all_geomean: f64,
    worst: f64,
    selected_plan_count: usize,
    controller_fraction: f64,
    exact_output: bool,
    gate_passed: bool,
}

fn summarize_performance(
    suites: &[SuiteRecord],
    comparisons: &[QualityComparison],
    totals: &PerformanceTotals,
    manifest: &Manifest,
) -> PerformanceSummary {
    let repeat_speedups = suites
        .iter()
        .filter(|suite| suite.class == "repeat-rich")
        .map(|suite| suite.speedup)
        .collect::<Vec<_>>();
    let all_speedups = suites.iter().map(|suite| suite.speedup).collect::<Vec<_>>();
    let repeat_geomean = geometric_mean(&repeat_speedups);
    let all_geomean = geometric_mean(&all_speedups);
    let worst = all_speedups.iter().copied().fold(f64::INFINITY, f64::min);
    let selected_plan_count = totals
        .plan_rounds
        .iter()
        .filter(|rounds| **rounds != 0)
        .count();
    let controller_fraction = totals.controller_duration / totals.correctable_duration;
    let exact_output = comparisons.iter().all(|comparison| comparison.exact);
    let gate_passed = performance_gate(
        exact_output,
        repeat_geomean,
        all_geomean,
        worst,
        selected_plan_count,
        controller_fraction,
        manifest,
    );
    PerformanceSummary {
        repeat_geomean,
        all_geomean,
        worst,
        selected_plan_count,
        controller_fraction,
        exact_output,
        gate_passed,
    }
}

fn performance_gate(
    exact_output: bool,
    repeat_geomean: f64,
    all_geomean: f64,
    worst: f64,
    selected_plan_count: usize,
    controller_fraction: f64,
    manifest: &Manifest,
) -> bool {
    exact_output
        && repeat_geomean >= manifest.minimum_repeat_rich_geometric_mean_speedup
        && all_geomean >= manifest.minimum_all_suite_geometric_mean_speedup
        && worst >= manifest.worst_suite_speedup_gate
        && selected_plan_count >= manifest.minimum_selected_plan_count
        && controller_fraction <= manifest.maximum_controller_fraction
}

struct PerformanceTotals {
    plan_rounds: [u64; 4],
    controller_duration: f64,
    correctable_duration: f64,
    plain_tokens_emitted: u64,
    correctable_tokens_emitted: u64,
    plain_decode_evaluations: u64,
    correctable_decode_evaluations: u64,
    plain_decode_duration: f64,
}

struct PromptEvidence {
    suite: SuiteRecord,
    comparison: QualityComparison,
}

fn measure_prompt(
    runtime: &mut Runtime<CudaBackend>,
    prompt: Prompt<'_>,
    arguments: &Arguments,
    totals: &mut PerformanceTotals,
) -> Result<PromptEvidence, Box<dyn Error>> {
    warmup_prompt(runtime, prompt.text, arguments)?;
    let mut plain_rates = Vec::with_capacity(arguments.repetitions);
    let mut correctable_rates = Vec::with_capacity(arguments.repetitions);
    let mut exact = true;
    let mut last_plain = Vec::new();
    let mut last_correctable = Vec::new();
    for repetition in 0..arguments.repetitions {
        let (plain, correctable) = generate_pair(
            runtime,
            prompt.text,
            arguments.tokens,
            arguments.seed,
            repetition & 1 == 0,
        )?;
        exact &= plain.tokens == correctable.tokens;
        plain_rates.push(decode_rate(&plain)?);
        correctable_rates.push(decode_rate(&correctable)?);
        record_performance(totals, &plain, &correctable)?;
        last_plain = plain.tokens;
        last_correctable = correctable.tokens;
    }
    let median_plain = median(&plain_rates);
    let median_correctable = median(&correctable_rates);
    Ok(PromptEvidence {
        suite: SuiteRecord {
            name: prompt.name.to_owned(),
            class: prompt.class.to_owned(),
            plain_tok_s: plain_rates,
            correctable_tok_s: correctable_rates,
            median_plain_tok_s: median_plain,
            median_correctable_tok_s: median_correctable,
            speedup: median_correctable / median_plain,
            exact_greedy_tokens: exact,
        },
        comparison: QualityComparison {
            suite: prompt.name.to_owned(),
            plain_transcript_sha256: token_stream_sha256(&last_plain),
            correctable_transcript_sha256: token_stream_sha256(&last_correctable),
            exact,
        },
    })
}

fn warmup_prompt(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &str,
    arguments: &Arguments,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..arguments.warmups {
        let _ = generate(runtime, prompt, arguments.tokens, false, arguments.seed)?;
        let _ = generate(runtime, prompt, arguments.tokens, true, arguments.seed)?;
    }
    Ok(())
}

fn generate_pair(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &str,
    tokens: usize,
    seed: u64,
    plain_first: bool,
) -> Result<(leone::GenerationResult, leone::GenerationResult), Box<dyn Error>> {
    if plain_first {
        Ok((
            generate(runtime, prompt, tokens, false, seed)?,
            generate(runtime, prompt, tokens, true, seed)?,
        ))
    } else {
        let correctable = generate(runtime, prompt, tokens, true, seed)?;
        let plain = generate(runtime, prompt, tokens, false, seed)?;
        Ok((plain, correctable))
    }
}

fn record_performance(
    totals: &mut PerformanceTotals,
    plain: &leone::GenerationResult,
    correctable: &leone::GenerationResult,
) -> Result<(), Box<dyn Error>> {
    totals.plain_tokens_emitted = checked_total(
        totals.plain_tokens_emitted,
        plain.stats.emitted_tokens as u64,
        "plain emitted-token accounting overflowed",
    )?;
    totals.correctable_tokens_emitted = checked_total(
        totals.correctable_tokens_emitted,
        correctable.stats.emitted_tokens as u64,
        "correctable emitted-token accounting overflowed",
    )?;
    totals.plain_decode_evaluations = checked_total(
        totals.plain_decode_evaluations,
        plain.stats.decode_evaluations as u64,
        "plain evaluation accounting overflowed",
    )?;
    totals.correctable_decode_evaluations = checked_total(
        totals.correctable_decode_evaluations,
        correctable.stats.decode_evaluations as u64,
        "correctable evaluation accounting overflowed",
    )?;
    totals.plain_decode_duration += plain.stats.decode_duration.as_secs_f64();
    record_plan_rounds(totals, correctable)?;
    totals.controller_duration += correctable
        .stats
        .speculation
        .correctable_controller_duration
        .as_secs_f64();
    totals.correctable_duration += correctable.stats.decode_duration.as_secs_f64();
    Ok(())
}

fn checked_total(current: u64, added: u64, message: &str) -> Result<u64, Box<dyn Error>> {
    current
        .checked_add(added)
        .ok_or_else(|| invalid_data(message).into())
}

fn record_plan_rounds(
    totals: &mut PerformanceTotals,
    correctable: &leone::GenerationResult,
) -> Result<(), Box<dyn Error>> {
    for (total, rounds) in totals
        .plan_rounds
        .iter_mut()
        .zip(correctable.stats.speculation.correctable_plan_rounds)
    {
        *total = checked_total(
            *total,
            rounds,
            "correctable performance plan accounting overflowed",
        )?;
    }
    Ok(())
}

fn ensure_gpu_idle() -> Result<(), Box<dyn Error>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Err(invalid_data("nvidia-smi compute-process query failed").into());
    }
    let own_pid = std::process::id().to_string();
    let competitors = String::from_utf8(output.stdout)?
        .lines()
        .filter(|line| line.split(',').next().map(str::trim) != Some(own_pid.as_str()))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if competitors.is_empty() {
        return Ok(());
    }
    Err(invalid_data(format!(
        "GPU timing requires no competing compute process; found {}",
        competitors.join("; ")
    ))
    .into())
}

fn generate(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &str,
    tokens: usize,
    correctable: bool,
    seed: u64,
) -> Result<leone::GenerationResult, Box<dyn Error>> {
    let mut options = GenerateOptions::greedy(tokens);
    options.decode_execution = DecodeExecution::Eager;
    options.seed = seed;
    if correctable {
        options.speculation = Speculation::Correctable(CorrectableDrafter::new(
            runtime.vocab_size(),
            NonZeroUsize::new(7).expect("seven is nonzero"),
        )?);
    }
    Ok(runtime.generate(prompt, options, |_| Ok(()), || false)?)
}

fn decode_rate(result: &leone::GenerationResult) -> Result<f64, io::Error> {
    let tokens = result.stats.emitted_tokens.saturating_sub(1);
    let seconds = result.stats.decode_duration.as_secs_f64();
    if tokens == 0 || seconds <= 0.0 {
        return Err(invalid_data(
            "performance run produced no timed decode tokens",
        ));
    }
    Ok(tokens as f64 / seconds)
}

fn prompts() -> Result<Vec<Prompt<'static>>, io::Error> {
    PROMPTS
        .lines()
        .map(|line| {
            let mut fields = line.splitn(3, '\t');
            let name = fields.next().unwrap_or_default();
            let class = fields.next().unwrap_or_default();
            let text = fields.next().unwrap_or_default();
            if name.is_empty() || class.is_empty() || text.is_empty() {
                return Err(invalid_data(
                    "a correctable prompt row must have three fields",
                ));
            }
            Ok(Prompt { name, class, text })
        })
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn geometric_mean(values: &[f64]) -> f64 {
    (values.iter().map(|value| value.ln()).sum::<f64>() / values.len() as f64).exp()
}

fn git_commit() -> Result<String, io::Error> {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    if !output.status.success() {
        return Err(invalid_data("git rev-parse HEAD failed"));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| invalid_data("git commit is not UTF-8"))
}

fn value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid_data("command flag is missing its value"))
}

fn positive(raw: &str, name: &str) -> Result<usize, io::Error> {
    raw.parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data(format!("{name} must be a positive integer: {raw}")))
}

fn positive_u64(raw: &str, name: &str) -> Result<u64, io::Error> {
    raw.parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data(format!("{name} must be a positive integer: {raw}")))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

struct OracleRandom(u64);

impl OracleRandom {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn weights(&mut self, values: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(values);
        for _ in 0..values {
            let bits = self.next();
            let value = if bits & 7 == 0 {
                0.0
            } else {
                ((bits >> 11) + 1) as f64
            };
            weights.push(value);
        }
        if weights.iter().all(|value| *value == 0.0) {
            weights[0] = 1.0;
        }
        weights
    }
}
