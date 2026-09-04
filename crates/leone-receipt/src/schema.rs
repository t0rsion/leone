use crate::{validate, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use uuid::Uuid;

/// A versioned runtime measurement and its execution context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeReceipt {
    pub schema_version: u32,
    pub receipt_id: Uuid,
    pub created_utc: DateTime<Utc>,
    pub machine: Machine,
    pub workload: Workload,
    pub results: RuntimeResults,
    pub quality_ref: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_summary: Option<QualitySummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub determinism: Option<DeterminismClaim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculation: Option<SpeculationRecord>,
    pub notes: Vec<String>,
}

/// Drafting counts from a measured run.
///
/// Counts only. The reader computes acceptance rate and tokens per forward
/// pass. Absent when drafting was off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculationRecord {
    /// The drafter that proposed the tokens, for example `suffix`.
    pub drafter: String,
    /// Rounds in which the drafter proposed at least one token.
    pub rounds: u64,
    /// Tokens proposed across those rounds.
    pub proposed: u64,
    /// Proposals the model kept.
    pub accepted: u64,
    /// Forward passes used by the whole decode.
    pub evaluations: u64,
    /// Largest number of proposal tokens requested in one round.
    #[serde(default)]
    pub draft_width: u64,
    /// Forward passes that used the position-major verifier.
    #[serde(default)]
    pub verify_passes: u64,
    /// Decode positions evaluated by the verifier passes.
    #[serde(default)]
    pub verified_positions: u64,
    /// Wall time spent in verifier forward passes and their logit reads.
    #[serde(default)]
    pub verify_duration_ms: f64,
    /// Plain rounds selected by the adaptive controller.
    #[serde(default)]
    pub adaptive_plain_rounds: u64,
    /// Speculative rounds selected by the adaptive controller.
    #[serde(default)]
    pub adaptive_speculative_rounds: u64,
    /// Adaptive decisions rejected by the measured speedup gate.
    #[serde(default)]
    pub adaptive_below_gate_rounds: u64,
    /// Adaptive decisions rejected by the cumulative regret limit.
    #[serde(default)]
    pub adaptive_regret_limited_rounds: u64,
    /// Proposals produced by suffix matching.
    #[serde(default)]
    pub adaptive_suffix_proposals: u64,
    /// Proposals produced by token recycling.
    #[serde(default)]
    pub adaptive_recycling_proposals: u64,
    /// Wall time spent selecting adaptive decode modes.
    #[serde(default)]
    pub adaptive_controller_duration_ms: f64,
    /// Adaptive rounds indexed by verifier position count.
    #[serde(default)]
    pub adaptive_width_rounds: [u64; 9],
    /// Plain rounds selected by the correctable controller.
    #[serde(default)]
    pub correctable_plain_rounds: u64,
    /// Speculative rounds selected by the correctable controller.
    #[serde(default)]
    pub correctable_speculative_rounds: u64,
    /// Correctable decisions rejected by the measured speedup gate.
    #[serde(default)]
    pub correctable_below_gate_rounds: u64,
    /// Correctable decisions rejected by the cumulative regret limit.
    #[serde(default)]
    pub correctable_regret_limited_rounds: u64,
    /// Correctable rounds indexed by suffix, unigram, bigram, and fourgram.
    #[serde(default)]
    pub correctable_plan_rounds: [u64; 4],
    /// Correctable rounds indexed by verifier position count.
    #[serde(default)]
    pub correctable_width_rounds: [u64; 9],
    /// Wall time spent selecting correctable plans.
    #[serde(default)]
    pub correctable_controller_duration_ms: f64,
    /// Sum of `1 - TV(p, q)` over evaluated correctable proposals.
    #[serde(default)]
    pub correctable_overlap_sum: f64,
    /// Evaluated correctable proposals in the overlap sum.
    #[serde(default)]
    pub correctable_overlap_proposals: u64,
}

/// The reproducibility claim a runtime receipt makes about its token stream.
///
/// Required from runtime schema v6. A harness that cannot capture a token
/// stream records `NotMeasured` with a reason rather than omitting the field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum DeterminismClaim {
    /// The engine produced one identical transcript across every repetition.
    ///
    /// The same model artifact, prompt bytes, and sampler reproduce
    /// `transcript_sha256` bit for bit. `identical_reps` is how many
    /// repetitions in this measurement produced that digest.
    Reproduced {
        order: ReductionOrder,
        sampler: SamplerRecord,
        prompt_sha256: String,
        transcript_sha256: String,
        identical_reps: u64,
    },
    /// No token stream was captured. The receipt makes no claim.
    NotMeasured { reason: String },
}

/// The reduction order a backend commits to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionOrder {
    /// Fixed launch and reduction order, no atomics, no opportunistic scheduling.
    #[serde(rename = "fixed-order")]
    FixedOrder,
}

/// The sampler that produced a token stream.
///
/// `Greedy` carries no seed because greedy decoding does not draw from the
/// random stream. A seeded sampler adds a variant rather than reporting a
/// placeholder seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SamplerRecord {
    Greedy,
    Seeded { seed: u64 },
}

/// Quality metrics copied from the linked quality receipt for display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualitySummary {
    pub kld_mean: f64,
    pub kld_p99: f64,
    pub top1_agreement: f64,
}

/// Hardware and software facts for a runtime measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Machine {
    pub hostname: String,
    pub gpu_name: String,
    pub gpu_vram_mib: u64,
    pub compute_cap: String,
    pub driver: String,
    pub cuda: String,
    pub cpu_model: String,
    pub ram_gib: u64,
    pub gpu_clocks_mhz: GpuClocksMhz,
    pub gpu_power_limit_w: f64,
}

/// GPU clock settings recorded for a runtime measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuClocksMhz {
    pub graphics: u64,
    pub memory: u64,
}

/// The engine, model, and token counts used for a runtime measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub engine: Engine,
    pub model_artifact: ModelArtifact,
    pub context_tokens: u64,
    pub generated_tokens: u64,
    pub batch: u64,
    pub reps: u64,
}

/// An engine build used for a runtime measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Engine {
    pub name: String,
    pub git_commit: String,
    pub build_flags: Vec<String>,
}

/// A model file used for a runtime measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifact {
    pub path: String,
    pub sha256: String,
    pub file_bytes: u64,
    pub format: String,
}

/// Measured throughput and its roofline estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeResults {
    pub decode_tok_s: RateSummary,
    pub prefill_tok_s: Option<RateSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<DurationSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_context_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_method: Option<PrefillMethod>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usable_bar: Option<UsableBar>,
    pub tokens_emitted: u64,
    pub bytes_per_token_by_class: BTreeMap<TensorClass, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_per_token_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights_resident_bytes_by_class: Option<BTreeMap<TensorClass, u64>>,
    pub model_bytes_total: u64,
    pub roofline: Roofline,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roofline_measured_achievable: Option<Roofline>,
}

/// A throughput distribution in tokens per second.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateSummary {
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
    pub reps: Vec<f64>,
}

/// A first-token latency distribution in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurationSummary {
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
    pub reps: Vec<f64>,
}

/// The prompt prefill implementation used by a runtime measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefillMethod {
    #[serde(rename = "chunked-cublaslt-fp16")]
    ChunkedCublasLtFp16,
    #[serde(rename = "tiled-cublaslt-fp16")]
    TiledCublasLtFp16,
    #[serde(rename = "sequential-decode")]
    SequentialDecode,
}

/// The v0.1 usable-prefill checks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsableBar {
    pub ttft_2k_under_1s: bool,
    pub pp512_ratio_vs_comparator: f64,
    pub bar_met: bool,
}

/// The estimated memory roofline and measured efficiency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Roofline {
    pub bandwidth_gbs_assumed: f64,
    pub ceiling_tok_s: f64,
    pub eta: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denominator_definition: Option<String>,
}

/// A tensor class used for byte accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TensorClass {
    Attn,
    Ffn,
    Embed,
    Head,
    Kv,
    Other,
}

impl TensorClass {
    /// Returns all tensor classes in serialized order.
    pub const fn all() -> [Self; 6] {
        [
            Self::Attn,
            Self::Ffn,
            Self::Embed,
            Self::Head,
            Self::Kv,
            Self::Other,
        ]
    }

    /// Returns a map containing every tensor class with a zero byte count.
    pub fn zero_map() -> BTreeMap<Self, u64> {
        Self::all().into_iter().map(|class| (class, 0)).collect()
    }

    /// Classifies one canonical GGUF tensor name for byte accounting.
    ///
    /// `token_embd.*` and `position_embd.*` are `Embed`. `output.weight` is
    /// `Head`. Cache tensors are `Kv`. Block names containing `.attn_` or
    /// `.attn.` are `Attn`. Names containing `.ffn_` or `.ffn.` are `Ffn`.
    /// All other names are `Other`. The cache rule does not match attention
    /// weights that contain `kv` in a projection name.
    pub fn from_gguf_name(name: &str) -> Self {
        if name.starts_with("token_embd.") || name.starts_with("position_embd.") {
            Self::Embed
        } else {
            Self::from_non_embedding_gguf_name(name)
        }
    }

    fn from_non_embedding_gguf_name(name: &str) -> Self {
        if name == "output.weight" {
            Self::Head
        } else if is_kv_name(name) {
            Self::Kv
        } else if name.contains(".attn_") || name.contains(".attn.") {
            Self::Attn
        } else if name.contains(".ffn_") || name.contains(".ffn.") {
            Self::Ffn
        } else {
            Self::Other
        }
    }

    /// Returns the stable receipt field name for this class.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Attn => "attn",
            Self::Ffn => "ffn",
            Self::Embed => "embed",
            Self::Head => "head",
            Self::Kv => "kv",
            Self::Other => "other",
        }
    }
}

fn is_kv_name(name: &str) -> bool {
    name.starts_with("kv.")
        || name.starts_with("cache_k.")
        || name.starts_with("cache_v.")
        || name.contains(".kv_cache.")
}

impl RuntimeReceipt {
    /// Parses and validates a runtime receipt from JSON.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let receipt: Self = serde_json::from_slice(bytes)?;
        receipt.validate()?;
        Ok(receipt)
    }

    /// Serializes a validated runtime receipt as stable, pretty JSON.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Validates the runtime receipt and its derived fields.
    pub fn validate(&self) -> Result<()> {
        validate(self)
    }

    /// Returns the quality state printed with the runtime result.
    pub fn quality_status(&self) -> String {
        match (self.quality_ref, &self.quality_summary) {
            (Some(receipt_id), Some(summary)) => format!(
                "quality: KLD mean {:.6}, p99 {:.6}, top1 {:.6} (receipt {receipt_id})",
                summary.kld_mean, summary.kld_p99, summary.top1_agreement
            ),
            (Some(receipt_id), None) => format!("quality: {receipt_id}"),
            (None, _) => "quality: unverified".to_owned(),
        }
    }
}

impl Display for RuntimeReceipt {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "runtime: {}", self.receipt_id)?;
        writeln!(
            formatter,
            "decode: {:.3} tok/s median over {} reps",
            self.results.decode_tok_s.median, self.workload.reps
        )?;
        if let Some(prefill) = &self.results.prefill_tok_s {
            writeln!(formatter, "prefill: {:.3} tok/s median", prefill.median)?;
        }
        if let (Some(ttft), Some(context)) =
            (&self.results.ttft_ms, self.results.ttft_context_tokens)
        {
            writeln!(
                formatter,
                "TTFT at {context} tokens: {:.3} ms median",
                ttft.median
            )?;
        }
        if let Some(bar) = self.results.usable_bar {
            writeln!(
                formatter,
                "usable prefill bar: {} (pp512 ratio {:.4})",
                if bar.bar_met { "met" } else { "missed" },
                bar.pp512_ratio_vs_comparator
            )?;
        }
        write!(formatter, "{}", self.quality_status())
    }
}

/// A versioned quality measurement linked from a runtime receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityReceipt {
    pub schema_version: u32,
    pub receipt_id: Uuid,
    pub created_utc: DateTime<Utc>,
    pub corpus: Corpus,
    pub oracle: Oracle,
    pub subject: Subject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_invariance: Option<BatchInvarianceMetric>,
    pub sample_count: u64,
}

/// Bitwise verifier agreement against consecutive one-position decode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchInvarianceMetric {
    pub definition: String,
    pub compared_floats: u64,
    pub mismatching_floats: u64,
    pub widths: Vec<u64>,
    pub context_depths: Vec<u64>,
    pub kv_cache: String,
}

/// The fixed corpus scored by a quality measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub name: String,
    pub sha256: String,
    pub n_prompts: u64,
    pub n_tokens_scored: u64,
}

/// The reference execution used by a quality measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Oracle {
    pub description: String,
    pub artifact_sha256: String,
    pub engine: EngineRef,
    pub dtype: String,
}

/// The model and engine being evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    pub model_artifact: ArtifactRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits_artifact: Option<ArtifactRef>,
    pub engine: EngineRef,
}

/// A model artifact reference used by a quality receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub sha256: String,
    pub path: String,
}

/// An engine reference used by a quality receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineRef {
    pub name: String,
    pub git_commit: String,
}

/// Quality metrics for a subject execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub kld: KldMetric,
    pub top1_agreement: f64,
}

/// KLD statistics and their fixed definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KldMetric {
    pub mean: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    pub p99: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    pub definition: String,
}

impl QualityReceipt {
    /// Parses and validates a quality receipt from JSON.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let receipt: Self = serde_json::from_slice(bytes)?;
        receipt.validate()?;
        Ok(receipt)
    }

    /// Serializes a validated quality receipt as stable, pretty JSON.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Validates the quality receipt and its metric definition.
    pub fn validate(&self) -> Result<()> {
        validate(self)
    }
}

pub(crate) fn filename_timestamp(created_utc: DateTime<Utc>) -> String {
    created_utc.to_rfc3339_opts(SecondsFormat::Secs, true)
}
