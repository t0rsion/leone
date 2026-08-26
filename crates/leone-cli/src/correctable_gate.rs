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
    if let [from, receipt_path, release_doc, document_path] = arguments {
        if from == "--from-receipt" && release_doc == "--release-doc" {
            let receipt: StudyReceipt = serde_json::from_slice(&fs::read(receipt_path)?)?;
            write_release_doc(&receipt, Path::new(document_path), Path::new(receipt_path))?;
            return Ok(());
        }
    }
    let manifest: Manifest = toml::from_str(MANIFEST)?;
    let arguments = parse(arguments, &manifest)?;
    let oracle = run_oracle(arguments.cases, arguments.seed, &manifest)?;
    let progressive_stats = exhaustive_progressive_oracle()?;
    let progressive = ProgressiveRecord {
        q4_codes: progressive_stats.q4_codes,
        q4_mismatches: progressive_stats.q4_mismatches,
        q6_codes: progressive_stats.q6_codes,
        q6_mismatches: progressive_stats.q6_mismatches,
        gate_passed: progressive_stats.q4_mismatches == 0 && progressive_stats.q6_mismatches == 0,
    };
    let evidence = arguments
        .model
        .as_deref()
        .map(|model| run_performance(model, &arguments, &manifest))
        .transpose()?;
    let oracle_gate_passed = oracle.gate_passed && progressive.gate_passed;
    let release_gate_passed = evidence
        .as_ref()
        .map(|evidence| oracle_gate_passed && evidence.runtime.gate_passed);
    let (runtime, quality) = match evidence {
        Some(evidence) => (Some(evidence.runtime), Some(evidence.quality)),
        None => (None, None),
    };
    let receipt = StudyReceipt {
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
    };
    let mut bytes = serde_json::to_vec_pretty(&receipt)?;
    bytes.push(b'\n');
    print!("{}", String::from_utf8(bytes.clone())?);
    let receipt_path = if arguments.receipt {
        fs::create_dir_all("receipts")?;
        let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ");
        let path = PathBuf::from(format!("receipts/correctable-{timestamp}.json"));
        fs::write(&path, bytes)?;
        Some(path)
    } else {
        None
    };
    if let Some(path) = arguments.release_doc.as_deref() {
        let receipt_path = receipt_path
            .as_deref()
            .ok_or_else(|| invalid_data("--release-doc requires --receipt"))?;
        write_release_doc(&receipt, path, receipt_path)?;
    }
    if !receipt.oracle_gate_passed {
        return Err(invalid_data("correctable oracle gate failed").into());
    }
    if receipt.release_gate_passed == Some(false) {
        return Err(invalid_data("correctable release gate failed").into());
    }
    Ok(())
}

fn parse(arguments: &[String], manifest: &Manifest) -> Result<Arguments, io::Error> {
    let mut parsed = Arguments {
        model: None,
        cases: manifest.randomized_oracle_cases,
        tokens: 128,
        warmups: 2,
        repetitions: 5,
        seed: 0x6c65_6f6e_655f_7638,
        receipt: false,
        release_doc: None,
    };
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-m" | "--model" => parsed.model = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--cases" => parsed.cases = positive_u64(value(arguments, &mut index)?, "cases")?,
            "--tokens" => parsed.tokens = positive(value(arguments, &mut index)?, "tokens")?,
            "--warmups" => parsed.warmups = positive(value(arguments, &mut index)?, "warmups")?,
            "--repetitions" => {
                parsed.repetitions = positive(value(arguments, &mut index)?, "repetitions")?
            }
            "--seed" => {
                let raw = value(arguments, &mut index)?;
                parsed.seed = raw
                    .parse()
                    .map_err(|_| invalid_data(format!("seed is invalid: {raw}")))?;
            }
            "--receipt" => parsed.receipt = true,
            "--release-doc" => {
                parsed.release_doc = Some(PathBuf::from(value(arguments, &mut index)?))
            }
            other => {
                return Err(invalid_data(format!(
                    "verify correctable argument is invalid: {other}"
                )))
            }
        }
        index += 1;
    }
    Ok(parsed)
}

fn write_release_doc(
    receipt: &StudyReceipt,
    path: &Path,
    receipt_path: &Path,
) -> Result<(), Box<dyn Error>> {
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
        .ok_or_else(|| invalid_data("receipt filename is not UTF-8"))?;
    let mut output = String::new();
    writeln!(output, "# Correctable inference evidence\n")?;
    writeln!(
        output,
        "Closed-loop correctable inference passes the preregistered release gate.\n"
    )?;
    writeln!(output, "## Generated result\n")?;
    writeln!(
        output,
        "| Suite | Class | Plain tok/s | Correctable tok/s | Speedup | Exact greedy tokens |"
    )?;
    writeln!(output, "|---|---|---:|---:|---:|---|")?;
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
    writeln!(output, "## Evidence\n")?;
    writeln!(
        output,
        "Receipt: [`{receipt_name}`](../receipts/{receipt_name})."
    )?;
    writeln!(output, "Runtime receipt ID: `{}`.", runtime.receipt_id)?;
    writeln!(output, "Quality receipt ID: `{}`.", quality.receipt_id)?;
    writeln!(output, "Model SHA-256: `{}`.", runtime.model_sha256)?;
    writeln!(output, "Source commit: `{}`.\n", receipt.git_commit)?;
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
    fs::write(path, output)?;
    Ok(())
}

fn run_oracle(cases: u64, seed: u64, manifest: &Manifest) -> Result<OracleRecord, Box<dyn Error>> {
    let started = Instant::now();
    let mut random = OracleRandom::new(seed);
    let accept_target = Distribution::from_probabilities(vec![0.75, 0.25])?;
    let accept_draft = Distribution::from_probabilities(vec![0.5, 0.5])?;
    let reject_target = Distribution::from_probabilities(vec![0.0, 1.0])?;
    let reject_draft = Distribution::from_probabilities(vec![1.0, 0.0])?;
    let mut reconstruction_mismatches = 0_u64;
    let mut deterministic_verdict_mismatches = 0_u64;
    let mut maximum_absolute_error = 0.0_f64;
    let mut distributional_acceptance_sum = 0.0_f64;
    let mut point_acceptance_sum = 0.0_f64;
    for case in 0..cases {
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
        let rejection_mass = 1.0 - overlap;
        let mut mismatch = false;
        for index in 0..vocab {
            let reconstructed = target.probabilities()[index].min(draft.probabilities()[index])
                + rejection_mass * residual.probabilities()[index];
            let error = (reconstructed - target.probabilities()[index]).abs();
            maximum_absolute_error = maximum_absolute_error.max(error);
            mismatch |= error > ORACLE_TOLERANCE;
        }
        reconstruction_mismatches += u64::from(mismatch);
        distributional_acceptance_sum += overlap;
        let drafted = select_draft(&draft, SamplerRng::new(seed ^ case), case)? as usize;
        point_acceptance_sum += target.probabilities()[drafted];
        let verdict = if case & 1 == 0 {
            verify(
                &accept_target,
                &accept_draft,
                0,
                SamplerRng::new(seed ^ case),
                case,
            )?
        } else {
            verify(
                &reject_target,
                &reject_draft,
                0,
                SamplerRng::new(seed ^ case),
                case,
            )?
        };
        let expected = if case & 1 == 0 {
            Verdict::Accept
        } else {
            Verdict::Reject { token: 1 }
        };
        deterministic_verdict_mismatches += u64::from(verdict != expected);
    }
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

fn run_performance(
    model: &Path,
    arguments: &Arguments,
    manifest: &Manifest,
) -> Result<PerformanceEvidence, Box<dyn Error>> {
    ensure_gpu_idle()?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, model)?;
    let machine_before = super::query_machine()?;
    let prompts = prompts()?;
    let model_sha256 = sha256_file(model)?;
    let runtime_id = Uuid::new_v4();
    let quality_id = Uuid::new_v4();
    let mut suites = Vec::with_capacity(prompts.len());
    let mut comparisons = Vec::with_capacity(prompts.len());
    let mut plan_rounds = [0_u64; 4];
    let mut controller_duration = 0.0_f64;
    let mut correctable_duration = 0.0_f64;
    let mut plain_tokens_emitted = 0_u64;
    let mut correctable_tokens_emitted = 0_u64;
    let mut plain_decode_evaluations = 0_u64;
    let mut correctable_decode_evaluations = 0_u64;
    let mut plain_decode_duration = 0.0_f64;
    for prompt in prompts {
        for _ in 0..arguments.warmups {
            let _ = generate(
                &mut runtime,
                prompt.text,
                arguments.tokens,
                false,
                arguments.seed,
            )?;
            let _ = generate(
                &mut runtime,
                prompt.text,
                arguments.tokens,
                true,
                arguments.seed,
            )?;
        }
        let mut plain_rates = Vec::with_capacity(arguments.repetitions);
        let mut correctable_rates = Vec::with_capacity(arguments.repetitions);
        let mut exact = true;
        let mut last_plain = Vec::new();
        let mut last_correctable = Vec::new();
        for repetition in 0..arguments.repetitions {
            let plain_first = repetition & 1 == 0;
            let (plain, correctable) = if plain_first {
                (
                    generate(
                        &mut runtime,
                        prompt.text,
                        arguments.tokens,
                        false,
                        arguments.seed,
                    )?,
                    generate(
                        &mut runtime,
                        prompt.text,
                        arguments.tokens,
                        true,
                        arguments.seed,
                    )?,
                )
            } else {
                let correctable = generate(
                    &mut runtime,
                    prompt.text,
                    arguments.tokens,
                    true,
                    arguments.seed,
                )?;
                let plain = generate(
                    &mut runtime,
                    prompt.text,
                    arguments.tokens,
                    false,
                    arguments.seed,
                )?;
                (plain, correctable)
            };
            exact &= plain.tokens == correctable.tokens;
            plain_rates.push(decode_rate(&plain)?);
            correctable_rates.push(decode_rate(&correctable)?);
            plain_tokens_emitted = plain_tokens_emitted
                .checked_add(plain.stats.emitted_tokens as u64)
                .ok_or_else(|| invalid_data("plain emitted-token accounting overflowed"))?;
            correctable_tokens_emitted = correctable_tokens_emitted
                .checked_add(correctable.stats.emitted_tokens as u64)
                .ok_or_else(|| invalid_data("correctable emitted-token accounting overflowed"))?;
            plain_decode_evaluations = plain_decode_evaluations
                .checked_add(plain.stats.decode_evaluations as u64)
                .ok_or_else(|| invalid_data("plain evaluation accounting overflowed"))?;
            correctable_decode_evaluations = correctable_decode_evaluations
                .checked_add(correctable.stats.decode_evaluations as u64)
                .ok_or_else(|| invalid_data("correctable evaluation accounting overflowed"))?;
            plain_decode_duration += plain.stats.decode_duration.as_secs_f64();
            for (total, rounds) in plan_rounds
                .iter_mut()
                .zip(correctable.stats.speculation.correctable_plan_rounds)
            {
                *total = total.checked_add(rounds).ok_or_else(|| {
                    invalid_data("correctable performance plan accounting overflowed")
                })?;
            }
            controller_duration += correctable
                .stats
                .speculation
                .correctable_controller_duration
                .as_secs_f64();
            correctable_duration += correctable.stats.decode_duration.as_secs_f64();
            last_plain = plain.tokens;
            last_correctable = correctable.tokens;
        }
        let median_plain = median(&plain_rates);
        let median_correctable = median(&correctable_rates);
        suites.push(SuiteRecord {
            name: prompt.name.to_owned(),
            class: prompt.class.to_owned(),
            plain_tok_s: plain_rates,
            correctable_tok_s: correctable_rates,
            median_plain_tok_s: median_plain,
            median_correctable_tok_s: median_correctable,
            speedup: median_correctable / median_plain,
            exact_greedy_tokens: exact,
        });
        comparisons.push(QualityComparison {
            suite: prompt.name.to_owned(),
            plain_transcript_sha256: token_stream_sha256(&last_plain),
            correctable_transcript_sha256: token_stream_sha256(&last_correctable),
            exact,
        });
    }
    let repeat_speedups = suites
        .iter()
        .filter(|suite| suite.class == "repeat-rich")
        .map(|suite| suite.speedup)
        .collect::<Vec<_>>();
    let all_speedups = suites.iter().map(|suite| suite.speedup).collect::<Vec<_>>();
    let repeat_geomean = geometric_mean(&repeat_speedups);
    let all_geomean = geometric_mean(&all_speedups);
    let worst = all_speedups.iter().copied().fold(f64::INFINITY, f64::min);
    let selected_plan_count = plan_rounds.iter().filter(|rounds| **rounds != 0).count();
    let controller_fraction = controller_duration / correctable_duration;
    ensure_gpu_idle()?;
    let machine_after = super::query_machine()?;
    let exact_output = comparisons.iter().all(|comparison| comparison.exact);
    let gate_passed = exact_output
        && repeat_geomean >= manifest.minimum_repeat_rich_geometric_mean_speedup
        && all_geomean >= manifest.minimum_all_suite_geometric_mean_speedup
        && worst >= manifest.worst_suite_speedup_gate
        && selected_plan_count >= manifest.minimum_selected_plan_count
        && controller_fraction <= manifest.maximum_controller_fraction;
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
        plain_tokens_emitted,
        correctable_tokens_emitted,
        plain_decode_evaluations,
        correctable_decode_evaluations,
        plain_decode_duration_ms: plain_decode_duration * 1_000.0,
        correctable_decode_duration_ms: correctable_duration * 1_000.0,
        suites,
        repeat_rich_geometric_mean_speedup: repeat_geomean,
        all_suite_geometric_mean_speedup: all_geomean,
        worst_suite_speedup: worst,
        selected_plan_count,
        selected_plan_rounds: plan_rounds,
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
