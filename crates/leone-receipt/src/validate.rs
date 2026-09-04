use crate::{
    BatchInvarianceMetric, DeterminismClaim, DurationSummary, Error, Machine, Metrics,
    PrefillMethod, QualityReceipt, RateSummary, Result, RuntimeReceipt, RuntimeResults,
    SpeculationRecord, TensorClass, Workload, KLD_DEFINITION, QUALITY_SCHEMA_VERSION,
    ROOFLINE_DENOMINATOR_DEFINITION, RUNTIME_SCHEMA_VERSION,
};
use uuid::{Uuid, Version};

const RELATIVE_TOLERANCE: f64 = 1e-9;

/// A receipt that can check its schema and field invariants.
pub trait ValidateReceipt {
    /// Checks the receipt and every derived value.
    fn validate_receipt(&self) -> Result<()>;
}

/// Validates a runtime or quality receipt.
pub fn validate(receipt: &(impl ValidateReceipt + ?Sized)) -> Result<()> {
    receipt.validate_receipt()
}

/// Computes median, p10, and p90 from throughput samples.
pub fn summarize_samples(samples: &[f64]) -> Result<RateSummary> {
    if samples.is_empty() {
        return Err(Error::validation("throughput samples must not be empty"));
    }
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err(Error::validation(
            "throughput samples must be finite and greater than zero",
        ));
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(RateSummary {
        median: percentile(&sorted, 0.5),
        p10: percentile(&sorted, 0.1),
        p90: percentile(&sorted, 0.9),
        reps: samples.to_vec(),
    })
}

/// Computes median, p10, and p90 from positive millisecond samples.
pub fn summarize_duration_samples_ms(samples: &[f64]) -> Result<DurationSummary> {
    let summary = summarize_samples(samples)?;
    Ok(DurationSummary {
        median: summary.median,
        p10: summary.p10,
        p90: summary.p90,
        reps: summary.reps,
    })
}

impl ValidateReceipt for RuntimeReceipt {
    fn validate_receipt(&self) -> Result<()> {
        validate_runtime_schema(self)?;
        validate_machine(&self.machine)?;
        validate_workload(&self.workload)?;
        validate_runtime_results(self)?;
        Ok(())
    }
}

fn validate_runtime_schema(receipt: &RuntimeReceipt) -> Result<()> {
    validate_runtime_version(receipt.schema_version)?;
    if let Some(quality_ref) = receipt.quality_ref {
        validate_v4(quality_ref, "quality_ref")?;
    }
    validate_quality_link(receipt)?;
    validate_determinism_link(receipt)?;
    validate_speculation_link(receipt)
}

fn validate_runtime_version(schema_version: u32) -> Result<()> {
    if !(1..=RUNTIME_SCHEMA_VERSION).contains(&schema_version) {
        return Err(Error::validation(format!(
            "runtime schema_version must be between 1 and {RUNTIME_SCHEMA_VERSION}, got {schema_version}"
        )));
    }
    Ok(())
}

fn validate_quality_link(receipt: &RuntimeReceipt) -> Result<()> {
    match (
        receipt.schema_version,
        receipt.quality_ref,
        &receipt.quality_summary,
    ) {
        (1..=3, _, Some(_)) => Err(Error::validation(
            "quality_summary is only valid in runtime schema v4 or later",
        )),
        (4..=RUNTIME_SCHEMA_VERSION, Some(_), Some(summary)) => {
            validate_nonnegative(summary.kld_mean, "quality_summary.kld_mean")?;
            validate_nonnegative(summary.kld_p99, "quality_summary.kld_p99")?;
            validate_fraction(summary.top1_agreement, "quality_summary.top1_agreement")
        }
        (4..=RUNTIME_SCHEMA_VERSION, None, None) | (1..=3, _, None) => Ok(()),
        (4..=RUNTIME_SCHEMA_VERSION, Some(_), None)
        | (4..=RUNTIME_SCHEMA_VERSION, None, Some(_)) => Err(Error::validation(
            "runtime schema v4 or later requires quality_ref and quality_summary together",
        )),
        _ => Err(Error::validation(format!(
            "runtime schema version {} has no quality-link rule in this validator",
            receipt.schema_version
        ))),
    }
}

fn validate_determinism_link(receipt: &RuntimeReceipt) -> Result<()> {
    match (receipt.schema_version, &receipt.determinism) {
        (1..=5, None) => Ok(()),
        (1..=5, Some(_)) => Err(Error::validation(
            "determinism is only valid in runtime schema v6 or later",
        )),
        (6..=RUNTIME_SCHEMA_VERSION, Some(claim)) => validate_determinism(claim),
        (6..=RUNTIME_SCHEMA_VERSION, None) => Err(Error::validation(
            "determinism is required by runtime schema v6 or later",
        )),
        _ => Err(Error::validation(format!(
            "runtime schema version {} has no determinism rule in this validator",
            receipt.schema_version
        ))),
    }
}

fn validate_speculation_link(receipt: &RuntimeReceipt) -> Result<()> {
    match (receipt.schema_version, &receipt.speculation) {
        (1..=5, None) => Ok(()),
        (1..=5, Some(_)) => Err(Error::validation(
            "speculation is only valid in runtime schema v6 or later",
        )),
        (_, Some(record)) => validate_speculation(record, receipt.schema_version),
        (_, None) => Ok(()),
    }
}

fn validate_machine(machine: &Machine) -> Result<()> {
    validate_machine_identity(machine)?;
    validate_machine_capacity(machine)
}

fn validate_machine_identity(machine: &Machine) -> Result<()> {
    validate_nonempty(&machine.hostname, "machine.hostname")?;
    validate_nonempty(&machine.gpu_name, "machine.gpu_name")?;
    validate_nonempty(&machine.compute_cap, "machine.compute_cap")?;
    validate_nonempty(&machine.driver, "machine.driver")?;
    validate_nonempty(&machine.cuda, "machine.cuda")?;
    validate_nonempty(&machine.cpu_model, "machine.cpu_model")?;
    Ok(())
}

fn validate_machine_capacity(machine: &Machine) -> Result<()> {
    validate_positive_u64(machine.gpu_vram_mib, "machine.gpu_vram_mib")?;
    validate_positive_u64(machine.ram_gib, "machine.ram_gib")?;
    validate_positive_u64(
        machine.gpu_clocks_mhz.graphics,
        "machine.gpu_clocks_mhz.graphics",
    )?;
    validate_positive_u64(
        machine.gpu_clocks_mhz.memory,
        "machine.gpu_clocks_mhz.memory",
    )?;
    validate_positive(machine.gpu_power_limit_w, "machine.gpu_power_limit_w")
}

fn validate_workload(workload: &Workload) -> Result<()> {
    validate_nonempty(&workload.engine.name, "workload.engine.name")?;
    validate_nonempty(&workload.engine.git_commit, "workload.engine.git_commit")?;
    validate_nonempty(
        &workload.model_artifact.path,
        "workload.model_artifact.path",
    )?;
    validate_sha256(
        &workload.model_artifact.sha256,
        "workload.model_artifact.sha256",
    )?;
    validate_nonempty(
        &workload.model_artifact.format,
        "workload.model_artifact.format",
    )?;
    validate_positive_u64(
        workload.model_artifact.file_bytes,
        "workload.model_artifact.file_bytes",
    )?;
    validate_positive_u64(workload.context_tokens, "workload.context_tokens")?;
    validate_positive_u64(workload.generated_tokens, "workload.generated_tokens")?;
    validate_positive_u64(workload.batch, "workload.batch")?;
    validate_positive_u64(workload.reps, "workload.reps")
}

fn validate_runtime_results(receipt: &RuntimeReceipt) -> Result<()> {
    validate_runtime_throughput(receipt)?;
    let denominator = validate_runtime_bytes(receipt)?;
    validate_runtime_rooflines(receipt, denominator)
}

fn validate_runtime_throughput(receipt: &RuntimeReceipt) -> Result<()> {
    validate_rate(
        &receipt.results.decode_tok_s,
        receipt.workload.reps,
        "results.decode_tok_s",
    )?;
    if let Some(prefill) = &receipt.results.prefill_tok_s {
        validate_rate(prefill, receipt.workload.reps, "results.prefill_tok_s")?;
    }
    validate_prefill_results(receipt)?;
    let expected_tokens = receipt
        .workload
        .generated_tokens
        .checked_mul(receipt.workload.reps)
        .and_then(|tokens| tokens.checked_mul(receipt.workload.batch))
        .ok_or_else(|| Error::validation("results.tokens_emitted overflowed"))?;
    if receipt.results.tokens_emitted != expected_tokens {
        return Err(Error::validation(format!(
            "results.tokens_emitted must be {expected_tokens}, got {}",
            receipt.results.tokens_emitted
        )));
    }
    Ok(())
}

fn validate_runtime_bytes(receipt: &RuntimeReceipt) -> Result<u64> {
    validate_tensor_class_map(&receipt.results)?;
    validate_positive_u64(
        receipt.results.model_bytes_total,
        "results.model_bytes_total",
    )?;
    if receipt.schema_version == 1 {
        Ok(receipt.results.model_bytes_total)
    } else {
        validate_v2_runtime_results(receipt)
    }
}

fn validate_tensor_class_map(results: &RuntimeResults) -> Result<()> {
    for class in TensorClass::all() {
        if !results.bytes_per_token_by_class.contains_key(&class) {
            return Err(Error::validation(format!(
                "results.bytes_per_token_by_class is missing {class:?}"
            )));
        }
    }
    Ok(())
}

fn validate_runtime_rooflines(receipt: &RuntimeReceipt, denominator: u64) -> Result<()> {
    validate_roofline(
        &receipt.results.roofline,
        denominator,
        receipt.results.decode_tok_s.median,
        "results.roofline",
        receipt.schema_version,
    )?;
    if let Some(roofline) = &receipt.results.roofline_measured_achievable {
        validate_roofline(
            roofline,
            denominator,
            receipt.results.decode_tok_s.median,
            "results.roofline_measured_achievable",
            receipt.schema_version,
        )?;
    }
    Ok(())
}

fn validate_prefill_results(receipt: &RuntimeReceipt) -> Result<()> {
    let results = &receipt.results;
    if !validate_prefill_version(receipt)? {
        return Ok(());
    }
    validate_prefill_pair(results)?;
    validate_ttft_pair(results, receipt.workload.reps)?;
    if let Some(bar) = results.usable_bar {
        validate_usable_bar_inputs(results, receipt.workload.context_tokens)?;
        validate_usable_bar_flags(results, bar)?;
    }
    Ok(())
}

fn validate_prefill_version(receipt: &RuntimeReceipt) -> Result<bool> {
    let results = &receipt.results;
    let has_v5_field = results.ttft_ms.is_some()
        || results.ttft_context_tokens.is_some()
        || results.prefill_method.is_some()
        || results.usable_bar.is_some();
    if receipt.schema_version < 5 && has_v5_field {
        return Err(Error::validation(
            "prefill method, TTFT, and usable_bar are only valid in runtime schema v5",
        ));
    }
    Ok(receipt.schema_version >= 5)
}

fn validate_prefill_pair(results: &RuntimeResults) -> Result<()> {
    match (&results.prefill_tok_s, results.prefill_method) {
        (Some(_), Some(_)) | (None, None) => Ok(()),
        _ => Err(Error::validation(
            "runtime schema v5 requires prefill_tok_s and prefill_method together",
        )),
    }
}

fn validate_ttft_pair(results: &RuntimeResults, reps: u64) -> Result<()> {
    match (&results.ttft_ms, results.ttft_context_tokens) {
        (Some(ttft), Some(context)) => {
            validate_duration(ttft, reps, "results.ttft_ms")?;
            validate_positive_u64(context, "results.ttft_context_tokens")
        }
        (None, None) => Ok(()),
        _ => Err(Error::validation(
            "runtime schema v5 requires ttft_ms and ttft_context_tokens together",
        )),
    }
}

fn validate_usable_bar_inputs(results: &RuntimeResults, context_tokens: u64) -> Result<()> {
    if results.ttft_ms.is_none() {
        return Err(Error::validation(
            "results.usable_bar requires results.ttft_ms",
        ));
    }
    if results.ttft_context_tokens != Some(2_048) {
        return Err(Error::validation(
            "results.usable_bar requires TTFT measured at 2048 tokens",
        ));
    }
    if context_tokens != 512 || results.prefill_tok_s.is_none() {
        return Err(Error::validation(
            "results.usable_bar requires the pp512 prefill measurement",
        ));
    }
    if !matches!(
        results.prefill_method,
        Some(PrefillMethod::ChunkedCublasLtFp16 | PrefillMethod::TiledCublasLtFp16)
    ) {
        return Err(Error::validation(
            "results.usable_bar requires a cuBLASLt FP16 prefill method",
        ));
    }
    Ok(())
}

fn validate_usable_bar_flags(results: &RuntimeResults, bar: crate::UsableBar) -> Result<()> {
    validate_positive(
        bar.pp512_ratio_vs_comparator,
        "results.usable_bar.pp512_ratio_vs_comparator",
    )?;
    let expected_ttft = results
        .ttft_ms
        .as_ref()
        .expect("validated usable_bar TTFT")
        .median
        < 1_000.0;
    if bar.ttft_2k_under_1s != expected_ttft {
        return Err(Error::validation(format!(
            "results.usable_bar.ttft_2k_under_1s must be {expected_ttft}"
        )));
    }
    let expected_met = expected_ttft && bar.pp512_ratio_vs_comparator >= 1.0 / 3.0;
    if bar.bar_met != expected_met {
        return Err(Error::validation(format!(
            "results.usable_bar.bar_met must be {expected_met}"
        )));
    }
    Ok(())
}

fn validate_v2_runtime_results(receipt: &RuntimeReceipt) -> Result<u64> {
    let touched = receipt.results.bytes_per_token_total.ok_or_else(|| {
        Error::validation("results.bytes_per_token_total is required by runtime schema v2")
    })?;
    validate_positive_u64(touched, "results.bytes_per_token_total")?;
    let expected_touched = sum_classes(
        &receipt.results.bytes_per_token_by_class,
        "results.bytes_per_token_by_class",
    )?;
    if touched != expected_touched {
        return Err(Error::validation(format!(
            "results.bytes_per_token_total must be {expected_touched}, got {touched}"
        )));
    }

    validate_resident_bytes(receipt)?;
    validate_measured_roofline(receipt)?;
    Ok(touched)
}

fn validate_resident_bytes(receipt: &RuntimeReceipt) -> Result<()> {
    let resident = receipt
        .results
        .weights_resident_bytes_by_class
        .as_ref()
        .ok_or_else(|| {
            Error::validation(
                "results.weights_resident_bytes_by_class is required by runtime schema v2",
            )
        })?;
    for class in TensorClass::all() {
        if !resident.contains_key(&class) {
            return Err(Error::validation(format!(
                "results.weights_resident_bytes_by_class is missing {class:?}"
            )));
        }
    }
    let expected_resident = sum_classes(resident, "results.weights_resident_bytes_by_class")?;
    if receipt.results.model_bytes_total != expected_resident {
        return Err(Error::validation(format!(
            "results.model_bytes_total must be {expected_resident}, got {}",
            receipt.results.model_bytes_total
        )));
    }
    Ok(())
}

fn validate_measured_roofline(receipt: &RuntimeReceipt) -> Result<()> {
    if receipt.results.roofline_measured_achievable.is_none() {
        return Err(Error::validation(
            "results.roofline_measured_achievable is required by runtime schema v2",
        ));
    }
    Ok(())
}

fn sum_classes(values: &std::collections::BTreeMap<TensorClass, u64>, field: &str) -> Result<u64> {
    TensorClass::all()
        .into_iter()
        .try_fold(0_u64, |sum, class| {
            sum.checked_add(values[&class])
                .ok_or_else(|| Error::validation(format!("{field} byte sum overflowed")))
        })
}

fn validate_roofline(
    roofline: &crate::Roofline,
    denominator: u64,
    decode_median: f64,
    field: &str,
    schema_version: u32,
) -> Result<()> {
    validate_positive(
        roofline.bandwidth_gbs_assumed,
        &format!("{field}.bandwidth_gbs_assumed"),
    )?;
    validate_positive(roofline.ceiling_tok_s, &format!("{field}.ceiling_tok_s"))?;
    validate_nonnegative(roofline.eta, &format!("{field}.eta"))?;
    match (schema_version, roofline.denominator_definition.as_deref()) {
        (3..=RUNTIME_SCHEMA_VERSION, Some(definition))
            if definition == ROOFLINE_DENOMINATOR_DEFINITION => {}
        (3..=RUNTIME_SCHEMA_VERSION, _) => {
            return Err(Error::validation(format!(
                "{field}.denominator_definition must match the runtime schema v3 definition"
            )))
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(Error::validation(format!(
                "{field}.denominator_definition is only valid in runtime schema v3"
            )))
        }
    }

    let expected_ceiling = roofline.bandwidth_gbs_assumed * 1e9 / denominator as f64;
    validate_derived(
        roofline.ceiling_tok_s,
        expected_ceiling,
        &format!("{field}.ceiling_tok_s"),
    )?;
    validate_derived(
        roofline.eta,
        decode_median / expected_ceiling,
        &format!("{field}.eta"),
    )
}

impl ValidateReceipt for QualityReceipt {
    fn validate_receipt(&self) -> Result<()> {
        validate_quality_identity(self)?;
        validate_quality_subject(self)?;
        let Some(metrics) = validate_quality_metric_family(self)? else {
            return Ok(());
        };
        validate_quality_metrics(self, metrics)
    }
}

fn validate_quality_identity(receipt: &QualityReceipt) -> Result<()> {
    validate_quality_schema_and_id(receipt)?;
    validate_quality_corpus(receipt)?;
    validate_quality_oracle(receipt)?;
    validate_quality_model(receipt)
}

fn validate_quality_schema_and_id(receipt: &QualityReceipt) -> Result<()> {
    if !(1..=QUALITY_SCHEMA_VERSION).contains(&receipt.schema_version) {
        return Err(Error::validation(format!(
            "quality schema_version must be between 1 and {QUALITY_SCHEMA_VERSION}, got {}",
            receipt.schema_version
        )));
    }
    validate_v4(receipt.receipt_id, "receipt_id")?;
    Ok(())
}

fn validate_quality_corpus(receipt: &QualityReceipt) -> Result<()> {
    validate_nonempty(&receipt.corpus.name, "corpus.name")?;
    validate_sha256(&receipt.corpus.sha256, "corpus.sha256")?;
    validate_positive_u64(receipt.corpus.n_prompts, "corpus.n_prompts")?;
    validate_positive_u64(receipt.corpus.n_tokens_scored, "corpus.n_tokens_scored")
}

fn validate_quality_oracle(receipt: &QualityReceipt) -> Result<()> {
    validate_nonempty(&receipt.oracle.description, "oracle.description")?;
    validate_sha256(&receipt.oracle.artifact_sha256, "oracle.artifact_sha256")?;
    validate_engine_ref(
        &receipt.oracle.engine.name,
        &receipt.oracle.engine.git_commit,
        "oracle.engine",
    )?;
    validate_nonempty(&receipt.oracle.dtype, "oracle.dtype")
}

fn validate_quality_model(receipt: &QualityReceipt) -> Result<()> {
    validate_sha256(
        &receipt.subject.model_artifact.sha256,
        "subject.model_artifact.sha256",
    )?;
    validate_nonempty(
        &receipt.subject.model_artifact.path,
        "subject.model_artifact.path",
    )
}

fn validate_quality_subject(receipt: &QualityReceipt) -> Result<()> {
    match (receipt.schema_version, &receipt.subject.logits_artifact) {
        (1, None) => {}
        (1, Some(_)) => {
            return Err(Error::validation(
                "subject.logits_artifact is only valid in quality schema v2",
            ));
        }
        (2..=QUALITY_SCHEMA_VERSION, Some(artifact)) => {
            validate_sha256(&artifact.sha256, "subject.logits_artifact.sha256")?;
            validate_nonempty(&artifact.path, "subject.logits_artifact.path")?;
        }
        (2, None) | (3, None) if receipt.batch_invariance.is_none() => {
            return Err(Error::validation(
                "subject.logits_artifact is required by quality schema v2",
            ));
        }
        (3, None) => {}
        _ => {
            return Err(Error::validation(format!(
                "quality schema version {} is not handled by this validator",
                receipt.schema_version
            )));
        }
    }
    validate_engine_ref(
        &receipt.subject.engine.name,
        &receipt.subject.engine.git_commit,
        "subject.engine",
    )
}

fn validate_quality_metric_family(receipt: &QualityReceipt) -> Result<Option<&Metrics>> {
    match (&receipt.metrics, &receipt.batch_invariance) {
        (Some(metrics), None) => Ok(Some(metrics)),
        (None, Some(metric)) if receipt.schema_version >= 3 => {
            validate_batch_invariance(receipt, metric)?;
            Ok(None)
        }
        (None, Some(_)) => Err(Error::validation(
            "batch_invariance is only valid in quality schema v3 or later",
        )),
        _ => Err(Error::validation(
            "quality receipt must carry exactly one metric family",
        )),
    }
}

fn validate_batch_invariance(
    receipt: &QualityReceipt,
    metric: &BatchInvarianceMetric,
) -> Result<()> {
    validate_batch_metadata(metric)?;
    validate_batch_dimensions(metric)?;
    validate_batch_sample_count(receipt, metric)
}

fn validate_batch_metadata(metric: &BatchInvarianceMetric) -> Result<()> {
    validate_nonempty(&metric.definition, "batch_invariance.definition")?;
    validate_nonempty(&metric.kv_cache, "batch_invariance.kv_cache")?;
    validate_positive_u64(metric.compared_floats, "batch_invariance.compared_floats")?;
    if metric.mismatching_floats > metric.compared_floats {
        return Err(Error::validation(
            "batch_invariance.mismatching_floats exceeds compared_floats",
        ));
    }
    Ok(())
}

fn validate_batch_dimensions(metric: &BatchInvarianceMetric) -> Result<()> {
    if metric.widths.is_empty() || metric.context_depths.is_empty() {
        return Err(Error::validation(
            "batch_invariance widths and context_depths must be nonempty",
        ));
    }
    for width in &metric.widths {
        validate_positive_u64(*width, "batch_invariance.widths")?;
    }
    for depth in &metric.context_depths {
        validate_positive_u64(*depth, "batch_invariance.context_depths")?;
    }
    Ok(())
}

fn validate_batch_sample_count(
    receipt: &QualityReceipt,
    metric: &BatchInvarianceMetric,
) -> Result<()> {
    if receipt.sample_count != metric.compared_floats {
        return Err(Error::validation(
            "sample_count must equal batch_invariance.compared_floats",
        ));
    }
    Ok(())
}

fn validate_quality_metrics(receipt: &QualityReceipt, metrics: &Metrics) -> Result<()> {
    validate_nonnegative(metrics.kld.mean, "metrics.kld.mean")?;
    validate_nonnegative(metrics.kld.p99, "metrics.kld.p99")?;
    validate_quality_kld(receipt.schema_version, metrics)?;
    let expected_definition = kld_definition(receipt.schema_version).ok_or_else(|| {
        Error::validation(format!(
            "quality schema version {} has no KLD definition",
            receipt.schema_version
        ))
    })?;
    if metrics.kld.definition != expected_definition {
        return Err(Error::validation(format!(
            "metrics.kld.definition does not match the definition for quality schema v{}",
            receipt.schema_version
        )));
    }
    validate_fraction(metrics.top1_agreement, "metrics.top1_agreement")?;
    if receipt.sample_count != receipt.corpus.n_tokens_scored {
        return Err(Error::validation(format!(
            "sample_count must equal corpus.n_tokens_scored, got {} and {}",
            receipt.sample_count, receipt.corpus.n_tokens_scored
        )));
    }
    validate_positive_u64(receipt.sample_count, "sample_count")
}

fn validate_quality_kld(schema_version: u32, metrics: &Metrics) -> Result<()> {
    match (schema_version, metrics.kld.p50, metrics.kld.max) {
        (1, None, None) => Ok(()),
        (1, _, _) => Err(Error::validation(
            "metrics.kld.p50 and metrics.kld.max are only valid in quality schema v2",
        )),
        (2..=QUALITY_SCHEMA_VERSION, Some(p50), Some(max)) => {
            validate_nonnegative(p50, "metrics.kld.p50")?;
            validate_nonnegative(max, "metrics.kld.max")?;
            if p50 > metrics.kld.p99 || metrics.kld.p99 > max {
                return Err(Error::validation(
                    "metrics KLD percentiles must satisfy p50 <= p99 <= max",
                ));
            }
            Ok(())
        }
        (2..=QUALITY_SCHEMA_VERSION, _, _) => Err(Error::validation(
            "metrics.kld.p50 and metrics.kld.max are required by quality schema v2",
        )),
        _ => Err(Error::validation(format!(
            "quality schema version {} is not handled by this validator",
            schema_version
        ))),
    }
}

fn validate_rate(rate: &RateSummary, reps: u64, field: &str) -> Result<()> {
    let expected_reps = usize::try_from(reps)
        .map_err(|_| Error::validation("workload.reps does not fit in memory"))?;
    if rate.reps.len() != expected_reps {
        return Err(Error::validation(format!(
            "{field}.reps must contain {reps} samples, got {}",
            rate.reps.len()
        )));
    }
    let expected = summarize_samples(&rate.reps)?;
    validate_derived(rate.median, expected.median, &format!("{field}.median"))?;
    validate_derived(rate.p10, expected.p10, &format!("{field}.p10"))?;
    validate_derived(rate.p90, expected.p90, &format!("{field}.p90"))
}

fn validate_duration(summary: &DurationSummary, reps: u64, field: &str) -> Result<()> {
    let rate = RateSummary {
        median: summary.median,
        p10: summary.p10,
        p90: summary.p90,
        reps: summary.reps.clone(),
    };
    validate_rate(&rate, reps, field)
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let rank = percentile * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn validate_v4(value: Uuid, field: &str) -> Result<()> {
    if value.get_version() != Some(Version::Random) {
        return Err(Error::validation(format!("{field} must be a UUID v4")));
    }
    Ok(())
}

fn validate_nonempty(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::validation(format!("{field} must not be empty")));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::validation(format!(
            "{field} must be a lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn validate_positive(value: f64, field: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::validation(format!(
            "{field} must be finite and greater than zero"
        )));
    }
    Ok(())
}

fn validate_nonnegative(value: f64, field: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::validation(format!(
            "{field} must be finite and nonnegative"
        )));
    }
    Ok(())
}

fn validate_fraction(value: f64, field: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::validation(format!(
            "{field} must be between zero and one"
        )));
    }
    Ok(())
}

fn validate_positive_u64(value: u64, field: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::validation(format!(
            "{field} must be greater than zero"
        )));
    }
    Ok(())
}

fn validate_derived(stored: f64, expected: f64, field: &str) -> Result<()> {
    if !stored.is_finite() || relative_error(stored, expected) > RELATIVE_TOLERANCE {
        return Err(Error::validation(format!(
            "{field} differs from its recomputed value: stored {stored}, expected {expected}"
        )));
    }
    Ok(())
}

fn relative_error(stored: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        stored.abs()
    } else {
        (stored - expected).abs() / expected.abs()
    }
}

fn validate_engine_ref(name: &str, git_commit: &str, field: &str) -> Result<()> {
    validate_nonempty(name, &format!("{field}.name"))?;
    validate_nonempty(git_commit, &format!("{field}.git_commit"))
}

/// Returns the KLD definition one quality schema version requires.
///
/// Every version shares one definition today. A version that redefines KLD
/// adds an arm here rather than changing `KLD_DEFINITION`. Receipts written
/// under the old definition keep validating.
const fn kld_definition(schema_version: u32) -> Option<&'static str> {
    match schema_version {
        1..=QUALITY_SCHEMA_VERSION => Some(KLD_DEFINITION),
        _ => None,
    }
}

fn validate_determinism(claim: &DeterminismClaim) -> Result<()> {
    match claim {
        DeterminismClaim::Reproduced {
            prompt_sha256,
            transcript_sha256,
            identical_reps,
            ..
        } => {
            validate_sha256(prompt_sha256, "determinism.prompt_sha256")?;
            validate_sha256(transcript_sha256, "determinism.transcript_sha256")?;
            validate_positive_u64(*identical_reps, "determinism.identical_reps")
        }
        DeterminismClaim::NotMeasured { reason } => validate_nonempty(reason, "determinism.reason"),
    }
}

fn validate_speculation(record: &SpeculationRecord, schema_version: u32) -> Result<()> {
    validate_speculation_counts(record)?;
    validate_speculation_draft(record, schema_version)?;
    validate_speculation_verifier(record)?;
    validate_speculation_durations(record)?;
    validate_adaptive_speculation(record, schema_version)?;
    validate_correctable_speculation(record, schema_version)
}

fn validate_speculation_counts(record: &SpeculationRecord) -> Result<()> {
    validate_nonempty(&record.drafter, "speculation.drafter")?;
    if record.accepted > record.proposed {
        return Err(Error::validation(format!(
            "speculation.accepted must not exceed speculation.proposed, got {} and {}",
            record.accepted, record.proposed
        )));
    }
    if record.proposed > 0 && record.rounds == 0 {
        return Err(Error::validation(
            "speculation.rounds must be positive when tokens were proposed",
        ));
    }
    validate_positive_u64(record.evaluations, "speculation.evaluations")?;
    Ok(())
}

fn validate_speculation_draft(record: &SpeculationRecord, schema_version: u32) -> Result<()> {
    if schema_version >= 7 && record.proposed > 0 && record.draft_width == 0 {
        return Err(Error::validation(
            "speculation.draft_width must be positive",
        ));
    }
    Ok(())
}

fn validate_speculation_verifier(record: &SpeculationRecord) -> Result<()> {
    if record.verify_passes == 0 && record.verified_positions != 0 {
        return Err(Error::validation(
            "speculation.verified_positions requires a verify pass",
        ));
    }
    if record.verify_passes > record.rounds {
        return Err(Error::validation(
            "speculation.verify_passes must not exceed speculation.rounds",
        ));
    }
    Ok(())
}

fn validate_speculation_durations(record: &SpeculationRecord) -> Result<()> {
    if !record.verify_duration_ms.is_finite() || record.verify_duration_ms < 0.0 {
        return Err(Error::validation(
            "speculation.verify_duration_ms must be finite and nonnegative",
        ));
    }
    if !record.adaptive_controller_duration_ms.is_finite()
        || record.adaptive_controller_duration_ms < 0.0
    {
        return Err(Error::validation(
            "speculation.adaptive_controller_duration_ms must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn validate_adaptive_speculation(record: &SpeculationRecord, schema_version: u32) -> Result<()> {
    validate_adaptive_rounds(record)?;
    if schema_version < 8 && has_adaptive_fields(record) {
        return Err(Error::validation(
            "adaptive speculation counters require runtime schema v8 or later",
        ));
    }
    Ok(())
}

fn validate_adaptive_rounds(record: &SpeculationRecord) -> Result<()> {
    let adaptive_width_total = record.adaptive_width_rounds[2..]
        .iter()
        .try_fold(0_u64, |total, rounds| total.checked_add(*rounds))
        .ok_or_else(|| Error::validation("adaptive width accounting overflowed"))?;
    if record.adaptive_width_rounds[0] != 0
        || record.adaptive_width_rounds[1] != record.adaptive_plain_rounds
        || adaptive_width_total != record.adaptive_speculative_rounds
    {
        return Err(Error::validation(
            "adaptive width rounds must match plain and speculative rounds",
        ));
    }
    if record.adaptive_speculative_rounds != 0
        && record.adaptive_speculative_rounds != record.rounds
    {
        return Err(Error::validation(
            "adaptive speculative rounds must match speculation.rounds",
        ));
    }
    Ok(())
}

fn has_adaptive_fields(record: &SpeculationRecord) -> bool {
    [
        record.adaptive_plain_rounds != 0,
        record.adaptive_speculative_rounds != 0,
        record.adaptive_below_gate_rounds != 0,
        record.adaptive_regret_limited_rounds != 0,
        record.adaptive_suffix_proposals != 0,
        record.adaptive_recycling_proposals != 0,
        record.adaptive_controller_duration_ms != 0.0,
        record.adaptive_width_rounds != [0; 9],
    ]
    .into_iter()
    .any(|present| present)
}

fn validate_correctable_speculation(record: &SpeculationRecord, schema_version: u32) -> Result<()> {
    validate_correctable_durations(record)?;
    validate_correctable_overlap(record)?;
    validate_correctable_rounds(record)?;
    if schema_version < 9 && has_correctable_fields(record) {
        return Err(Error::validation(
            "correctable speculation counters require runtime schema v9 or later",
        ));
    }
    Ok(())
}

fn validate_correctable_durations(record: &SpeculationRecord) -> Result<()> {
    if !record.correctable_controller_duration_ms.is_finite()
        || record.correctable_controller_duration_ms < 0.0
    {
        return Err(Error::validation(
            "speculation.correctable_controller_duration_ms must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn validate_correctable_overlap(record: &SpeculationRecord) -> Result<()> {
    if !record.correctable_overlap_sum.is_finite() || record.correctable_overlap_sum < 0.0 {
        return Err(Error::validation(
            "speculation.correctable_overlap_sum must be finite and nonnegative",
        ));
    }
    if record.correctable_overlap_proposals > record.proposed
        || record.correctable_overlap_sum > record.correctable_overlap_proposals as f64 + 1e-9
    {
        return Err(Error::validation(
            "correctable overlap must describe no more than the proposed tokens",
        ));
    }
    Ok(())
}

fn validate_correctable_rounds(record: &SpeculationRecord) -> Result<()> {
    let correctable_plan_total = correctable_plan_total(record)?;
    let correctable_width_total = correctable_width_total(record)?;
    validate_correctable_round_accounting(record, correctable_plan_total, correctable_width_total)?;
    if record.correctable_speculative_rounds != 0
        && record.correctable_speculative_rounds != record.rounds
    {
        return Err(Error::validation(
            "correctable speculative rounds must match speculation.rounds",
        ));
    }
    Ok(())
}

fn correctable_plan_total(record: &SpeculationRecord) -> Result<u64> {
    record
        .correctable_plan_rounds
        .iter()
        .try_fold(0_u64, |total, rounds| total.checked_add(*rounds))
        .ok_or_else(|| Error::validation("correctable plan accounting overflowed"))
}

fn correctable_width_total(record: &SpeculationRecord) -> Result<u64> {
    record.correctable_width_rounds[2..]
        .iter()
        .try_fold(0_u64, |total, rounds| total.checked_add(*rounds))
        .ok_or_else(|| Error::validation("correctable width accounting overflowed"))
}

fn validate_correctable_round_accounting(
    record: &SpeculationRecord,
    correctable_plan_total: u64,
    correctable_width_total: u64,
) -> Result<()> {
    if correctable_plan_total != record.correctable_speculative_rounds
        || record.correctable_width_rounds[0] != 0
        || record.correctable_width_rounds[1] != record.correctable_plain_rounds
        || correctable_width_total != record.correctable_speculative_rounds
    {
        return Err(Error::validation(
            "correctable plan and width rounds must match controller rounds",
        ));
    }
    Ok(())
}

fn has_correctable_fields(record: &SpeculationRecord) -> bool {
    [
        record.correctable_plain_rounds != 0,
        record.correctable_speculative_rounds != 0,
        record.correctable_below_gate_rounds != 0,
        record.correctable_regret_limited_rounds != 0,
        record.correctable_plan_rounds != [0; 4],
        record.correctable_width_rounds != [0; 9],
        record.correctable_controller_duration_ms != 0.0,
        record.correctable_overlap_sum != 0.0,
        record.correctable_overlap_proposals != 0,
    ]
    .into_iter()
    .any(|present| present)
}
