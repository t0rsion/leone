use crate::{
    DeterminismClaim, DurationSummary, Error, PrefillMethod, QualityReceipt, RateSummary, Result,
    RuntimeReceipt, SpeculationRecord, TensorClass, KLD_DEFINITION, QUALITY_SCHEMA_VERSION,
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
        if !(1..=RUNTIME_SCHEMA_VERSION).contains(&self.schema_version) {
            return Err(Error::validation(format!(
                "runtime schema_version must be between 1 and {RUNTIME_SCHEMA_VERSION}, got {}",
                self.schema_version
            )));
        }
        validate_v4(self.receipt_id, "receipt_id")?;
        if let Some(quality_ref) = self.quality_ref {
            validate_v4(quality_ref, "quality_ref")?;
        }
        match (self.schema_version, self.quality_ref, &self.quality_summary) {
            (1..=3, _, Some(_)) => {
                return Err(Error::validation(
                    "quality_summary is only valid in runtime schema v4 or later",
                ));
            }
            (4..=RUNTIME_SCHEMA_VERSION, Some(_), Some(summary)) => {
                validate_nonnegative(summary.kld_mean, "quality_summary.kld_mean")?;
                validate_nonnegative(summary.kld_p99, "quality_summary.kld_p99")?;
                validate_fraction(summary.top1_agreement, "quality_summary.top1_agreement")?;
            }
            (4..=RUNTIME_SCHEMA_VERSION, None, None) | (1..=3, _, None) => {}
            (4..=RUNTIME_SCHEMA_VERSION, Some(_), None)
            | (4..=RUNTIME_SCHEMA_VERSION, None, Some(_)) => {
                return Err(Error::validation(
                    "runtime schema v4 or later requires quality_ref and quality_summary together",
                ));
            }
            _ => {
                return Err(Error::validation(format!(
                    "runtime schema version {} has no quality-link rule in this validator",
                    self.schema_version
                )));
            }
        }
        match (self.schema_version, &self.determinism) {
            (1..=5, None) => {}
            (1..=5, Some(_)) => {
                return Err(Error::validation(
                    "determinism is only valid in runtime schema v6 or later",
                ));
            }
            (6..=RUNTIME_SCHEMA_VERSION, Some(claim)) => validate_determinism(claim)?,
            (6..=RUNTIME_SCHEMA_VERSION, None) => {
                return Err(Error::validation(
                    "determinism is required by runtime schema v6 or later",
                ));
            }
            _ => {
                return Err(Error::validation(format!(
                    "runtime schema version {} has no determinism rule in this validator",
                    self.schema_version
                )));
            }
        }
        match (self.schema_version, &self.speculation) {
            (1..=5, None) => {}
            (1..=5, Some(_)) => {
                return Err(Error::validation(
                    "speculation is only valid in runtime schema v6 or later",
                ));
            }
            (_, Some(record)) => validate_speculation(record, self.schema_version)?,
            (_, None) => {}
        }
        validate_nonempty(&self.machine.hostname, "machine.hostname")?;
        validate_nonempty(&self.machine.gpu_name, "machine.gpu_name")?;
        validate_nonempty(&self.machine.compute_cap, "machine.compute_cap")?;
        validate_nonempty(&self.machine.driver, "machine.driver")?;
        validate_nonempty(&self.machine.cuda, "machine.cuda")?;
        validate_nonempty(&self.machine.cpu_model, "machine.cpu_model")?;
        validate_positive_u64(self.machine.gpu_vram_mib, "machine.gpu_vram_mib")?;
        validate_positive_u64(self.machine.ram_gib, "machine.ram_gib")?;
        validate_positive_u64(
            self.machine.gpu_clocks_mhz.graphics,
            "machine.gpu_clocks_mhz.graphics",
        )?;
        validate_positive_u64(
            self.machine.gpu_clocks_mhz.memory,
            "machine.gpu_clocks_mhz.memory",
        )?;
        validate_positive(self.machine.gpu_power_limit_w, "machine.gpu_power_limit_w")?;

        validate_nonempty(&self.workload.engine.name, "workload.engine.name")?;
        validate_nonempty(
            &self.workload.engine.git_commit,
            "workload.engine.git_commit",
        )?;
        validate_nonempty(
            &self.workload.model_artifact.path,
            "workload.model_artifact.path",
        )?;
        validate_sha256(
            &self.workload.model_artifact.sha256,
            "workload.model_artifact.sha256",
        )?;
        validate_nonempty(
            &self.workload.model_artifact.format,
            "workload.model_artifact.format",
        )?;
        validate_positive_u64(
            self.workload.model_artifact.file_bytes,
            "workload.model_artifact.file_bytes",
        )?;
        validate_positive_u64(self.workload.context_tokens, "workload.context_tokens")?;
        validate_positive_u64(self.workload.generated_tokens, "workload.generated_tokens")?;
        validate_positive_u64(self.workload.batch, "workload.batch")?;
        validate_positive_u64(self.workload.reps, "workload.reps")?;

        validate_rate(
            &self.results.decode_tok_s,
            self.workload.reps,
            "results.decode_tok_s",
        )?;
        if let Some(prefill) = &self.results.prefill_tok_s {
            validate_rate(prefill, self.workload.reps, "results.prefill_tok_s")?;
        }
        validate_prefill_results(self)?;
        let expected_tokens = self
            .workload
            .generated_tokens
            .checked_mul(self.workload.reps)
            .and_then(|tokens| tokens.checked_mul(self.workload.batch))
            .ok_or_else(|| Error::validation("results.tokens_emitted overflowed"))?;
        if self.results.tokens_emitted != expected_tokens {
            return Err(Error::validation(format!(
                "results.tokens_emitted must be {expected_tokens}, got {}",
                self.results.tokens_emitted
            )));
        }
        for class in TensorClass::all() {
            if !self.results.bytes_per_token_by_class.contains_key(&class) {
                return Err(Error::validation(format!(
                    "results.bytes_per_token_by_class is missing {class:?}"
                )));
            }
        }
        validate_positive_u64(self.results.model_bytes_total, "results.model_bytes_total")?;
        let denominator = if self.schema_version == 1 {
            self.results.model_bytes_total
        } else {
            validate_v2_runtime_results(self)?
        };
        validate_roofline(
            &self.results.roofline,
            denominator,
            self.results.decode_tok_s.median,
            "results.roofline",
            self.schema_version,
        )?;
        if let Some(roofline) = &self.results.roofline_measured_achievable {
            validate_roofline(
                roofline,
                denominator,
                self.results.decode_tok_s.median,
                "results.roofline_measured_achievable",
                self.schema_version,
            )?;
        }
        Ok(())
    }
}

fn validate_prefill_results(receipt: &RuntimeReceipt) -> Result<()> {
    let results = &receipt.results;
    let has_v5_field = results.ttft_ms.is_some()
        || results.ttft_context_tokens.is_some()
        || results.prefill_method.is_some()
        || results.usable_bar.is_some();
    if receipt.schema_version < 5 {
        if has_v5_field {
            return Err(Error::validation(
                "prefill method, TTFT, and usable_bar are only valid in runtime schema v5",
            ));
        }
        return Ok(());
    }

    match (&results.prefill_tok_s, results.prefill_method) {
        (Some(_), Some(_)) | (None, None) => {}
        _ => {
            return Err(Error::validation(
                "runtime schema v5 requires prefill_tok_s and prefill_method together",
            ));
        }
    }
    match (&results.ttft_ms, results.ttft_context_tokens) {
        (Some(ttft), Some(context)) => {
            validate_duration(ttft, receipt.workload.reps, "results.ttft_ms")?;
            validate_positive_u64(context, "results.ttft_context_tokens")?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::validation(
                "runtime schema v5 requires ttft_ms and ttft_context_tokens together",
            ));
        }
    }
    if let Some(bar) = results.usable_bar {
        let ttft = results
            .ttft_ms
            .as_ref()
            .ok_or_else(|| Error::validation("results.usable_bar requires results.ttft_ms"))?;
        if results.ttft_context_tokens != Some(2_048) {
            return Err(Error::validation(
                "results.usable_bar requires TTFT measured at 2048 tokens",
            ));
        }
        if receipt.workload.context_tokens != 512 || results.prefill_tok_s.is_none() {
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
        validate_positive(
            bar.pp512_ratio_vs_comparator,
            "results.usable_bar.pp512_ratio_vs_comparator",
        )?;
        let expected_ttft = ttft.median < 1_000.0;
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
    if receipt.results.roofline_measured_achievable.is_none() {
        return Err(Error::validation(
            "results.roofline_measured_achievable is required by runtime schema v2",
        ));
    }
    Ok(touched)
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
        if !(1..=QUALITY_SCHEMA_VERSION).contains(&self.schema_version) {
            return Err(Error::validation(format!(
                "quality schema_version must be between 1 and {QUALITY_SCHEMA_VERSION}, got {}",
                self.schema_version
            )));
        }
        validate_v4(self.receipt_id, "receipt_id")?;
        validate_nonempty(&self.corpus.name, "corpus.name")?;
        validate_sha256(&self.corpus.sha256, "corpus.sha256")?;
        validate_positive_u64(self.corpus.n_prompts, "corpus.n_prompts")?;
        validate_positive_u64(self.corpus.n_tokens_scored, "corpus.n_tokens_scored")?;
        validate_nonempty(&self.oracle.description, "oracle.description")?;
        validate_sha256(&self.oracle.artifact_sha256, "oracle.artifact_sha256")?;
        validate_engine_ref(
            &self.oracle.engine.name,
            &self.oracle.engine.git_commit,
            "oracle.engine",
        )?;
        validate_nonempty(&self.oracle.dtype, "oracle.dtype")?;
        validate_sha256(
            &self.subject.model_artifact.sha256,
            "subject.model_artifact.sha256",
        )?;
        validate_nonempty(
            &self.subject.model_artifact.path,
            "subject.model_artifact.path",
        )?;
        match (self.schema_version, &self.subject.logits_artifact) {
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
            (2, None) | (3, None) if self.batch_invariance.is_none() => {
                return Err(Error::validation(
                    "subject.logits_artifact is required by quality schema v2",
                ));
            }
            (3, None) => {}
            _ => {
                return Err(Error::validation(format!(
                    "quality schema version {} is not handled by this validator",
                    self.schema_version
                )));
            }
        }
        validate_engine_ref(
            &self.subject.engine.name,
            &self.subject.engine.git_commit,
            "subject.engine",
        )?;
        let metrics = match (&self.metrics, &self.batch_invariance) {
            (Some(metrics), None) => metrics,
            (None, Some(metric)) if self.schema_version >= 3 => {
                validate_nonempty(&metric.definition, "batch_invariance.definition")?;
                validate_nonempty(&metric.kv_cache, "batch_invariance.kv_cache")?;
                validate_positive_u64(metric.compared_floats, "batch_invariance.compared_floats")?;
                if metric.mismatching_floats > metric.compared_floats {
                    return Err(Error::validation(
                        "batch_invariance.mismatching_floats exceeds compared_floats",
                    ));
                }
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
                if self.sample_count != metric.compared_floats {
                    return Err(Error::validation(
                        "sample_count must equal batch_invariance.compared_floats",
                    ));
                }
                return Ok(());
            }
            (None, Some(_)) => {
                return Err(Error::validation(
                    "batch_invariance is only valid in quality schema v3 or later",
                ));
            }
            _ => {
                return Err(Error::validation(
                    "quality receipt must carry exactly one metric family",
                ));
            }
        };
        validate_nonnegative(metrics.kld.mean, "metrics.kld.mean")?;
        validate_nonnegative(metrics.kld.p99, "metrics.kld.p99")?;
        match (self.schema_version, metrics.kld.p50, metrics.kld.max) {
            (1, None, None) => {}
            (1, _, _) => {
                return Err(Error::validation(
                    "metrics.kld.p50 and metrics.kld.max are only valid in quality schema v2",
                ));
            }
            (2..=QUALITY_SCHEMA_VERSION, Some(p50), Some(max)) => {
                validate_nonnegative(p50, "metrics.kld.p50")?;
                validate_nonnegative(max, "metrics.kld.max")?;
                if p50 > metrics.kld.p99 || metrics.kld.p99 > max {
                    return Err(Error::validation(
                        "metrics KLD percentiles must satisfy p50 <= p99 <= max",
                    ));
                }
            }
            (2..=QUALITY_SCHEMA_VERSION, _, _) => {
                return Err(Error::validation(
                    "metrics.kld.p50 and metrics.kld.max are required by quality schema v2",
                ));
            }
            _ => {
                return Err(Error::validation(format!(
                    "quality schema version {} is not handled by this validator",
                    self.schema_version
                )));
            }
        }
        let expected_definition = kld_definition(self.schema_version).ok_or_else(|| {
            Error::validation(format!(
                "quality schema version {} has no KLD definition",
                self.schema_version
            ))
        })?;
        if metrics.kld.definition != expected_definition {
            return Err(Error::validation(format!(
                "metrics.kld.definition does not match the definition for quality schema v{}",
                self.schema_version
            )));
        }
        validate_fraction(metrics.top1_agreement, "metrics.top1_agreement")?;
        if self.sample_count != self.corpus.n_tokens_scored {
            return Err(Error::validation(format!(
                "sample_count must equal corpus.n_tokens_scored, got {} and {}",
                self.sample_count, self.corpus.n_tokens_scored
            )));
        }
        validate_positive_u64(self.sample_count, "sample_count")
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
    if schema_version >= 7 && record.proposed > 0 && record.draft_width == 0 {
        return Err(Error::validation(
            "speculation.draft_width must be positive",
        ));
    }
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
    if schema_version < 8
        && (record.adaptive_plain_rounds != 0
            || record.adaptive_speculative_rounds != 0
            || record.adaptive_below_gate_rounds != 0
            || record.adaptive_regret_limited_rounds != 0
            || record.adaptive_suffix_proposals != 0
            || record.adaptive_recycling_proposals != 0
            || record.adaptive_controller_duration_ms != 0.0
            || record.adaptive_width_rounds != [0; 9])
    {
        return Err(Error::validation(
            "adaptive speculation counters require runtime schema v8 or later",
        ));
    }
    if !record.correctable_controller_duration_ms.is_finite()
        || record.correctable_controller_duration_ms < 0.0
    {
        return Err(Error::validation(
            "speculation.correctable_controller_duration_ms must be finite and nonnegative",
        ));
    }
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
    let correctable_plan_total = record
        .correctable_plan_rounds
        .iter()
        .try_fold(0_u64, |total, rounds| total.checked_add(*rounds))
        .ok_or_else(|| Error::validation("correctable plan accounting overflowed"))?;
    let correctable_width_total = record.correctable_width_rounds[2..]
        .iter()
        .try_fold(0_u64, |total, rounds| total.checked_add(*rounds))
        .ok_or_else(|| Error::validation("correctable width accounting overflowed"))?;
    if correctable_plan_total != record.correctable_speculative_rounds
        || record.correctable_width_rounds[0] != 0
        || record.correctable_width_rounds[1] != record.correctable_plain_rounds
        || correctable_width_total != record.correctable_speculative_rounds
    {
        return Err(Error::validation(
            "correctable plan and width rounds must match controller rounds",
        ));
    }
    if record.correctable_speculative_rounds != 0
        && record.correctable_speculative_rounds != record.rounds
    {
        return Err(Error::validation(
            "correctable speculative rounds must match speculation.rounds",
        ));
    }
    if schema_version < 9
        && (record.correctable_plain_rounds != 0
            || record.correctable_speculative_rounds != 0
            || record.correctable_below_gate_rounds != 0
            || record.correctable_regret_limited_rounds != 0
            || record.correctable_plan_rounds != [0; 4]
            || record.correctable_width_rounds != [0; 9]
            || record.correctable_controller_duration_ms != 0.0
            || record.correctable_overlap_sum != 0.0
            || record.correctable_overlap_proposals != 0)
    {
        return Err(Error::validation(
            "correctable speculation counters require runtime schema v9 or later",
        ));
    }
    Ok(())
}
