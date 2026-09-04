use crate::model::{DenseLayer, OutputWeight, QkNorm};
use crate::{
    decode_graph_bucket, AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot,
    DecodeOp, DecodeProfile, LoadedModel, ModelLoadError, Position, StateAllocation,
    StateAllocator, StateError, StateKind, StateLifetime, TokenizerError, VectorShape,
};
use crate::{
    distribution, select, verify, Distribution, Draft, MirostatConfig, MirostatState, Penalties,
    Sampler, SamplerRng, SuffixDrafter, Verdict,
};
use crate::{
    CorrectableController, CorrectableControllerConfig, CorrectableDrafter, CorrectableObservation,
};
use crate::{PrefillMethod, PrefillPlan, PrefillWorkspace, RopeShape};
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

/// The default number of prompt positions evaluated in one prefill block.
///
/// A larger block reduces launch and weight-dequantization work. The CUDA
/// backend bounds attention scratch with 1,024-token query tiles. Activation
/// storage grows with this block. A smaller `--prefill-chunk` lowers the
/// block when device memory is tight.
pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 4096;

/// An error returned by model state, forward execution, or token streaming.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Model(#[from] ModelLoadError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    State(#[from] StateError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Sampler(#[from] crate::SamplerError),
    #[error(transparent)]
    Penalty(#[from] crate::PenaltyError),
    #[error(transparent)]
    Correctable(#[from] crate::CorrectableError),
    #[error(transparent)]
    Constraint(#[from] crate::ConstraintError),
    #[error("output constraints cannot be combined with speculative decoding")]
    ConstraintSpeculation,
    #[error("the prompt tokenizes to an empty sequence")]
    EmptyPrompt,
    #[error("the requested generation length must be nonzero")]
    ZeroGeneration,
    #[error("{architecture} models do not support decode graphs; select eager decode")]
    UnsupportedDecodeGraph { architecture: &'static str },
    #[error("the request needs {requested} context positions but the model supports {capacity}")]
    ContextCapacity { requested: usize, capacity: usize },
    #[error("runtime byte accounting overflowed")]
    SizeOverflow,
    #[error("token callback failed: {0}")]
    TokenCallback(String),
    #[error("logit callback failed: {0}")]
    LogitCallback(String),
    #[error("evaluation needs at least two tokens")]
    TooFewEvalTokens,
    #[error("evaluation token {token} at index {index} is outside vocab size {vocab_size}")]
    EvalTokenOutOfRange {
        index: usize,
        token: u32,
        vocab_size: usize,
    },
    #[error("evaluation window must contain at least two tokens")]
    EvalWindowTooSmall,
    #[error("Mirostat cannot be combined with speculative decoding")]
    MirostatSpeculation,
    #[error("a generation session must have complete live device state before it can be forked")]
    SessionForkUnavailable,
    #[error("correctable proposal has {proposed} tokens but {distributions} draft distributions")]
    CorrectableDistributionCount {
        proposed: usize,
        distributions: usize,
    },
}

impl RuntimeError {
    /// Wraps an output error returned by a token callback.
    pub fn token_callback(error: impl std::fmt::Display) -> Self {
        Self::TokenCallback(error.to_string())
    }

    /// Wraps an output error returned by a full-vocabulary logit callback.
    pub fn logit_callback(error: impl std::fmt::Display) -> Self {
        Self::LogitCallback(error.to_string())
    }
}

/// Selects whether generation copies logits back for diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitCapture {
    Disabled,
    Top(NonZeroUsize),
}

/// Selects optional steady-state decode profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeProfileMode {
    Disabled,
    Steps(NonZeroUsize),
}

/// Selects eager launches or backend graph replay for decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeExecution {
    Eager,
    Graph,
}

/// Selects the scalar type stored in the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheDtype {
    Q8,
    F16,
    F32,
}

impl KvCacheDtype {
    fn layout(self, elements: usize) -> Result<BufferLayout, BackendError> {
        match self {
            Self::Q8 => BufferLayout::q8_kv(elements),
            Self::F16 => BufferLayout::f16(elements),
            Self::F32 => BufferLayout::f32(elements),
        }
    }
}

/// How a decode step proposes tokens before the model verifies them.
///
/// `Disabled` is the v0.1 path and stays bit-identical to it. Every proposal
/// is checked against the model's own distribution before it is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speculation {
    Disabled,
    Suffix(SuffixDrafter),
    /// Training-free proposals and verifier widths from measured rounds.
    Adaptive(crate::adaptive_draft::AdaptiveDrafter),
    /// Distribution-valued proposals and verifier widths from measured rounds.
    Correctable(CorrectableDrafter),
}

/// The checked options for one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    /// Maximum prompt positions evaluated in one prefill block.
    pub prefill_chunk_tokens: usize,
    pub logit_capture: LogitCapture,
    pub decode_profile: DecodeProfileMode,
    pub decode_execution: DecodeExecution,
    pub kv_cache_dtype: KvCacheDtype,
    pub sampler: Sampler,
    pub penalties: Penalties,
    pub seed: u64,
    pub speculation: Speculation,
    pub mirostat: Option<MirostatConfig>,
    pub output_constraint: Option<crate::OutputConstraint>,
}

impl GenerateOptions {
    /// Creates greedy generation without copying logits to the host.
    ///
    /// Decode uses graph replay. The KV cache is F16.
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            prefill_chunk_tokens: DEFAULT_PREFILL_CHUNK_TOKENS,
            logit_capture: LogitCapture::Disabled,
            sampler: Sampler::greedy(),
            penalties: Penalties::none(),
            seed: 0,
            speculation: Speculation::Disabled,
            mirostat: None,
            output_constraint: None,
            decode_profile: DecodeProfileMode::Disabled,
            decode_execution: DecodeExecution::Graph,
            kv_cache_dtype: KvCacheDtype::F16,
        }
    }
}

/// What drafting achieved over one generation.
///
/// `accepted / proposed` is the acceptance rate.
/// `(accepted + rounds) / evaluations` is the tokens produced per forward
/// pass. Both are reported rather than summarized. A drafter that wins on
/// one prompt class and loses on another has no single number.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpeculationStats {
    /// Rounds in which the drafter proposed at least one token.
    pub rounds: usize,
    /// Tokens proposed across those rounds.
    pub proposed: usize,
    /// Proposals the model kept.
    pub accepted: usize,
    /// Largest proposal width attempted by the drafter.
    pub draft_width: usize,
    /// Forward passes that used the position-major verifier.
    pub verify_passes: usize,
    /// Plain rounds selected by the adaptive controller.
    pub adaptive_plain_rounds: u64,
    /// Speculative rounds selected by the adaptive controller.
    pub adaptive_speculative_rounds: u64,
    /// Adaptive decisions rejected by the measured speedup gate.
    pub adaptive_below_gate_rounds: u64,
    /// Adaptive decisions rejected by the cumulative regret limit.
    pub adaptive_regret_limited_rounds: u64,
    /// Adaptive proposals produced by suffix matching.
    pub adaptive_suffix_proposals: u64,
    /// Adaptive proposals produced by token recycling.
    pub adaptive_recycling_proposals: u64,
    /// Wall time spent in adaptive mode selection.
    pub adaptive_controller_duration: Duration,
    /// Adaptive rounds indexed by verifier position count.
    pub adaptive_width_rounds: [u64; 9],
    /// Plain rounds selected by the correctable controller.
    pub correctable_plain_rounds: u64,
    /// Speculative rounds selected by the correctable controller.
    pub correctable_speculative_rounds: u64,
    /// Correctable decisions rejected by the measured speedup gate.
    pub correctable_below_gate_rounds: u64,
    /// Correctable decisions rejected by the cumulative regret limit.
    pub correctable_regret_limited_rounds: u64,
    /// Correctable rounds indexed by proposal plan.
    pub correctable_plan_rounds: [u64; 4],
    /// Correctable rounds indexed by verifier position count.
    pub correctable_width_rounds: [u64; 9],
    /// Wall time spent selecting a correctable plan.
    pub correctable_controller_duration: Duration,
    /// Sum of `1 - TV(p, q)` over distribution-valued proposals.
    pub correctable_overlap_sum: f64,
    /// Distribution-valued proposals included in the overlap sum.
    pub correctable_overlap_proposals: usize,
    /// Decode positions evaluated by those verifier passes.
    pub verified_positions: usize,
}

/// What one drafted round produced.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpeculationOutcome {
    /// Tokens the drafter proposed.
    pub proposed: usize,
    /// Proposals the model kept.
    pub accepted: usize,
    /// Forward passes the round cost.
    pub evaluations: usize,
    /// Positions evaluated by a batched verifier pass, or zero on fallback.
    pub verified_positions: usize,
    /// Wall time spent launching and reading the verifier pass.
    pub verify_duration: Duration,
    /// Sum of `1 - TV(p, q)` over distribution-valued proposals.
    pub overlap_sum: f64,
    /// Distribution-valued proposals included in the overlap sum.
    pub overlap_proposals: usize,
}

/// One token emitted during generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedToken {
    pub id: u32,
    pub bytes: Vec<u8>,
}

/// One token and value from a diagnostic logit ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Logit {
    pub token: u32,
    pub value: f32,
}

/// The top logits that produced one sampled token.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitSnapshot {
    pub input_position: usize,
    pub sampled_token: u32,
    pub top: Vec<Logit>,
}

/// A tokens-per-second rate, or `Unavailable` when no evaluation ran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasuredRate {
    TokensPerSecond(f64),
    Unavailable,
}

/// Timing and count data for one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub emitted_tokens: usize,
    pub decode_evaluations: usize,
    pub prefill_duration: Duration,
    pub ttft_duration: Option<Duration>,
    pub decode_duration: Duration,
    pub verify_duration: Duration,
    pub cancelled: bool,
    pub prefill_method: PrefillMethod,
    pub prefill_workspace: PrefillWorkspace,
    pub speculation: SpeculationStats,
}

impl GenerationStats {
    pub fn prefill_rate(&self) -> MeasuredRate {
        measured_rate(self.prompt_tokens, self.prefill_duration)
    }

    pub fn decode_rate(&self) -> MeasuredRate {
        measured_rate(self.emitted_tokens.saturating_sub(1), self.decode_duration)
    }
}

/// Tokens, timings, and optional logit snapshots from one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    /// The tokenized prompt used as decode input.
    pub prompt_tokens: Vec<u32>,
    pub tokens: Vec<u32>,
    pub stats: GenerationStats,
    pub logits: Vec<LogitSnapshot>,
    pub decode_profile: Option<DecodeProfile>,
}

/// How one request relates to the KV state retained by its session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionReuseClass {
    Cold,
    ExactRepeat,
    AppendOnly,
    ArbitraryBranch,
    RestoreReplay,
    DeviceFork,
    HostWake,
}

/// Exact token counts for one session reuse decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionReplay {
    pub reuse_class: SessionReuseClass,
    pub cached_tokens: usize,
    pub reused_tokens: usize,
    pub replayed_tokens: usize,
    pub computed_tokens: usize,
}

/// Exact state and allocation facts for one live session fork.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionFork {
    pub cached_tokens: usize,
    pub context_tokens: usize,
    pub copied_bytes: u64,
    pub kv_cache_dtype: KvCacheDtype,
}

/// Exact byte and context facts for one hibernated generation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionHibernation {
    pub cached_tokens: usize,
    pub context_tokens: usize,
    pub host_bytes: u64,
    pub kv_cache_dtype: KvCacheDtype,
}

/// Host-owned physical state for one suspended generation session.
#[derive(Debug)]
pub struct HibernatedSession {
    state: HibernatedKvState,
    activations: HibernatedActivations,
    evaluated_tokens: Vec<u32>,
    restored_tokens: Vec<u32>,
    prefill_boundary: usize,
    mirostat: Option<MirostatState>,
    adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
    correctable_controller: Option<CorrectableController>,
    record: SessionHibernation,
}

impl HibernatedSession {
    pub fn evaluated_tokens(&self) -> &[u32] {
        &self.evaluated_tokens
    }

    pub const fn record(&self) -> SessionHibernation {
        self.record
    }
}

/// Host state needed to reconstruct a generation session by exact replay.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationCheckpoint {
    evaluated_tokens: Vec<u32>,
    prefill_boundary: usize,
    mirostat: Option<MirostatState>,
}

impl GenerationCheckpoint {
    pub(crate) fn from_parts(
        evaluated_tokens: Vec<u32>,
        prefill_boundary: usize,
        mirostat: Option<MirostatState>,
    ) -> Self {
        Self {
            evaluated_tokens,
            prefill_boundary,
            mirostat,
        }
    }

    pub fn evaluated_tokens(&self) -> &[u32] {
        &self.evaluated_tokens
    }

    pub const fn mirostat(&self) -> Option<MirostatState> {
        self.mirostat
    }

    /// Returns the number of leading tokens evaluated by prompt prefill.
    pub const fn prefill_boundary(&self) -> usize {
        self.prefill_boundary
    }
}

impl Default for SessionReplay {
    fn default() -> Self {
        Self {
            reuse_class: SessionReuseClass::Cold,
            cached_tokens: 0,
            reused_tokens: 0,
            replayed_tokens: 0,
            computed_tokens: 0,
        }
    }
}

/// KV state and sampler feedback retained between generation requests.
///
/// A cancelled or failed request invalidates the session. The next request
/// starts cold. Partial KV state is never exposed.
#[derive(Debug)]
pub struct GenerationSession<B: Backend> {
    state: Option<KvState<B>>,
    activations: Option<Activations<B>>,
    evaluated_tokens: Vec<u32>,
    restored_tokens: Vec<u32>,
    prefill_boundary: usize,
    mirostat: Option<MirostatState>,
    adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
    correctable_controller: Option<CorrectableController>,
    last_replay: SessionReplay,
    pending_fork: Option<SessionFork>,
    last_fork: Option<SessionFork>,
    pending_wake: Option<SessionHibernation>,
    last_hibernation: Option<SessionHibernation>,
}

impl<B: Backend> GenerationSession<B> {
    pub fn new() -> Self {
        Self {
            state: None,
            activations: None,
            evaluated_tokens: Vec::new(),
            restored_tokens: Vec::new(),
            prefill_boundary: 0,
            mirostat: None,
            adaptive_controller: None,
            correctable_controller: None,
            last_replay: SessionReplay::default(),
            pending_fork: None,
            last_fork: None,
            pending_wake: None,
            last_hibernation: None,
        }
    }

    pub fn evaluated_tokens(&self) -> &[u32] {
        &self.evaluated_tokens
    }

    pub const fn last_replay(&self) -> SessionReplay {
        self.last_replay
    }

    /// Returns the most recent device-state fork record.
    pub const fn last_fork(&self) -> Option<SessionFork> {
        self.last_fork
    }

    /// Returns the most recent hibernation record.
    pub const fn last_hibernation(&self) -> Option<SessionHibernation> {
        self.last_hibernation
    }

    /// Returns true when no device or replay state remains.
    pub fn is_empty(&self) -> bool {
        self.state.is_none()
            && self.activations.is_none()
            && self.evaluated_tokens.is_empty()
            && self.restored_tokens.is_empty()
            && self.prefill_boundary == 0
            && self.mirostat.is_none()
            && self.adaptive_controller.is_none()
            && self.correctable_controller.is_none()
    }

    /// Copies the host checkpoint needed to replay this session.
    pub fn checkpoint(&self) -> GenerationCheckpoint {
        GenerationCheckpoint {
            evaluated_tokens: self.evaluated_tokens.clone(),
            prefill_boundary: self.prefill_boundary,
            mirostat: self.mirostat,
        }
    }

    /// Clears device allocations and replay state.
    pub fn invalidate(&mut self) {
        self.state = None;
        self.activations = None;
        self.evaluated_tokens.clear();
        self.restored_tokens.clear();
        self.prefill_boundary = 0;
        self.mirostat = None;
        self.adaptive_controller = None;
        self.correctable_controller = None;
        self.pending_fork = None;
        self.last_fork = None;
        self.pending_wake = None;
        self.last_hibernation = None;
    }

    /// Restores a token checkpoint for exact replay on the next request.
    ///
    /// Checkpoints do not contain backend buffers. The next request rebuilds
    /// KV from these tokens and reports that work as replay, not reuse.
    pub fn restore_tokens(&mut self, tokens: Vec<u32>, mirostat: Option<MirostatState>) {
        self.invalidate();
        self.prefill_boundary = tokens.len();
        self.restored_tokens = tokens;
        self.mirostat = mirostat;
    }

    /// Restores one host checkpoint for exact replay on the next request.
    pub fn restore(&mut self, checkpoint: GenerationCheckpoint) {
        self.invalidate();
        self.restored_tokens = checkpoint.evaluated_tokens;
        self.prefill_boundary = checkpoint.prefill_boundary;
        self.mirostat = checkpoint.mirostat;
    }
}

impl<B: Backend> Default for GenerationSession<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Timings from one fixed-token decode benchmark repetition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBenchmarkRun {
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_duration: Duration,
    pub ttft_duration: Duration,
    pub decode_duration: Duration,
    pub detokenized_bytes: usize,
    pub prefill_method: PrefillMethod,
    pub prefill_workspace: PrefillWorkspace,
    pub transcript_sha256: String,
}

/// Timings from one prompt-only benchmark repetition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillBenchmarkRun {
    pub prompt_tokens: usize,
    pub prefill_duration: Duration,
    pub ttft_duration: Duration,
    pub prefill_method: PrefillMethod,
    pub prefill_workspace: PrefillWorkspace,
}

/// Exact allocation evidence for one KV cache depth and storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCapacityProbe {
    /// Context depth requested by the caller.
    pub requested_context_tokens: usize,
    /// Context bucket allocated by the runtime.
    pub allocated_context_tokens: usize,
    /// Bytes allocated for key and value caches across every layer.
    pub cache_bytes: u64,
    /// KV storage type used by the allocation.
    pub dtype: KvCacheDtype,
}

/// Maximum KV error for one layer against sequential decode prefill.
///
/// Relative error divides by `max(abs(sequential), 1e-6)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefillKvLayerError {
    pub layer: usize,
    pub key_max_abs: f32,
    pub key_max_rel: f32,
    pub value_max_abs: f32,
    pub value_max_rel: f32,
}

/// KV and greedy-token results from the ignored prefill characterization.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillCharacterization {
    pub prompt_tokens: usize,
    pub chunk_tokens: usize,
    pub layers: Vec<PrefillKvLayerError>,
    pub repeated_layers: Vec<PrefillKvLayerError>,
    pub sequential_tokens: Vec<u32>,
    pub chunked_tokens: Vec<u32>,
    pub repeated_chunked_tokens: Vec<u32>,
}

/// Bitwise comparison between sequential decode and one verifier pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyCharacterization {
    pub positions: usize,
    pub compared_floats: usize,
    pub mismatching_floats: usize,
}

impl PrefillBenchmarkRun {
    pub fn prefill_rate(&self) -> MeasuredRate {
        measured_rate(self.prompt_tokens, self.prefill_duration)
    }
}

impl DecodeBenchmarkRun {
    pub fn prefill_rate(&self) -> MeasuredRate {
        measured_rate(self.prompt_tokens, self.prefill_duration)
    }

    pub fn decode_rate(&self) -> MeasuredRate {
        measured_rate(self.decode_tokens, self.decode_duration)
    }
}

/// Owns one backend and one uploaded model.
#[derive(Debug)]
pub struct Runtime<B: Backend> {
    backend: B,
    model: LoadedModel<B>,
}

struct GenerationPreparation<B: Backend> {
    attention_shape: AttentionShape,
    state: KvState<B>,
    activations: Activations<B>,
    reuse_class: SessionReuseClass,
    reused_tokens: usize,
    cached_tokens: usize,
    restored_common: usize,
    common_tokens: usize,
    replay_prefill_boundary: usize,
    verify_activations: Vec<Option<VerifyActivations<B>>>,
}

struct GenerationReuse {
    reuse_class: SessionReuseClass,
    compatible: bool,
    reused_tokens: usize,
    cached_tokens: usize,
    restored_common: usize,
    common_tokens: usize,
    replay_prefill_boundary: usize,
}

struct GenerationDraft {
    drafted: Draft,
    adaptive_round: Option<(std::num::NonZeroUsize, Duration)>,
    correctable_round: Option<(
        Option<crate::CorrectablePlan>,
        std::num::NonZeroUsize,
        Duration,
    )>,
    correctable_distributions: Option<Vec<Distribution>>,
}

struct GenerationStep {
    evaluations: usize,
    outcome: Option<SpeculationOutcome>,
}

struct GenerationInitialization {
    host_sampling: bool,
    mirostat: Option<MirostatState>,
    output_constraint: Option<crate::constraint::JsonObjectConstraint>,
    profile_target: usize,
    profile_start: Option<Instant>,
    use_graph: bool,
    adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
    correctable_controller: Option<CorrectableController>,
}

struct ChunkedEvalWindow<B: Backend> {
    state: KvState<B>,
    full: PrefillEvalActivations<B>,
    tail: Option<PrefillEvalActivations<B>>,
}

struct BenchmarkState<B: Backend> {
    state: KvState<B>,
    activations: Activations<B>,
    prepared: PreparedPrefill<B>,
}

struct AllocatedPromptPrefill<B: Backend> {
    workspace: PrefillWorkspace,
    full: PrefillActivations<B>,
    tail: Option<PrefillActivations<B>>,
    remainder: usize,
}

struct ChunkedEvalRows<'a, B: Backend> {
    window: &'a [u32],
    start: usize,
    offset: usize,
    block: &'a [u32],
    activations: &'a PrefillEvalActivations<B>,
    scored: &'a mut usize,
}

struct CorrectableDraftContext<'a> {
    prompt_tokens: &'a [u32],
    tokens: &'a [u32],
    context: &'a mut Vec<u32>,
    state_position: usize,
    rng: SamplerRng,
    correctable_controller: &'a mut Option<CorrectableController>,
}

struct PrefillBatchContext<'a, B: Backend> {
    prompt_tokens: &'a [u32],
    processed: usize,
    count: usize,
    base_position: usize,
    state: &'a mut KvState<B>,
    batch: &'a mut PrefillActivations<B>,
    attention_shape: AttentionShape,
}

impl<B: Backend> Runtime<B> {
    /// Allocates and releases a complete KV cache without running the model.
    pub fn probe_kv_capacity(
        &mut self,
        context_tokens: usize,
        dtype: KvCacheDtype,
    ) -> Result<KvCapacityProbe, RuntimeError> {
        let allocated_context_tokens = self.validate_probe_context(context_tokens)?;
        let shape = self.probe_attention_shape(allocated_context_tokens)?;
        let cache_bytes = self.probe_cache_bytes(dtype, shape)?;
        let state = KvState::new(&mut self.backend, self.model.config.n_layer, shape, dtype)?;
        self.backend.synchronize()?;
        drop(state);
        Ok(KvCapacityProbe {
            requested_context_tokens: context_tokens,
            allocated_context_tokens,
            cache_bytes,
            dtype,
        })
    }

    fn validate_probe_context(&self, context_tokens: usize) -> Result<usize, RuntimeError> {
        if context_tokens == 0 {
            return Err(RuntimeError::EmptyPrompt);
        }
        if context_tokens > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: context_tokens,
                capacity: self.model.config.context_length,
            });
        }
        decode_graph_bucket(context_tokens).map_err(Into::into)
    }

    fn probe_attention_shape(
        &self,
        allocated_context_tokens: usize,
    ) -> Result<AttentionShape, RuntimeError> {
        AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            allocated_context_tokens,
        )
        .map_err(Into::into)
    }

    fn probe_cache_bytes(
        &self,
        dtype: KvCacheDtype,
        shape: AttentionShape,
    ) -> Result<u64, RuntimeError> {
        let one_cache_bytes = u64::try_from(dtype.layout(shape.cache_elements()?)?.bytes())
            .map_err(|_| RuntimeError::SizeOverflow)?;
        one_cache_bytes
            .checked_mul(2)
            .and_then(|bytes| {
                u64::try_from(self.model.config.n_layer)
                    .ok()
                    .and_then(|layers| bytes.checked_mul(layers))
            })
            .ok_or(RuntimeError::SizeOverflow)
    }

    /// Loads a model into one backend.
    pub fn load(mut backend: B, path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let model = LoadedModel::load(&mut backend, path)?;
        Ok(Self { backend, model })
    }

    /// Creates a runtime from an already uploaded model.
    pub fn from_model(backend: B, model: LoadedModel<B>) -> Self {
        Self { backend, model }
    }

    pub const fn model(&self) -> &LoadedModel<B> {
        &self.model
    }

    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Copies one complete live session into an independent child.
    pub fn fork_session(
        &mut self,
        source: &GenerationSession<B>,
    ) -> Result<GenerationSession<B>, RuntimeError> {
        let source_state = source
            .state
            .as_ref()
            .ok_or(RuntimeError::SessionForkUnavailable)?;
        let source_activations = source
            .activations
            .as_ref()
            .ok_or(RuntimeError::SessionForkUnavailable)?;
        if source.evaluated_tokens.is_empty() {
            return Err(RuntimeError::SessionForkUnavailable);
        }
        let state = KvState::fork(&mut self.backend, source_state)?;
        let activations = Activations::fork(&mut self.backend, source_activations)?;
        self.backend.synchronize()?;
        let copied_bytes = source_state
            ._allocator
            .capacity_bytes()
            .checked_add(4)
            .and_then(|bytes| bytes.checked_add(Activations::<B>::bytes(&self.model.config).ok()?))
            .ok_or(RuntimeError::SizeOverflow)?;
        let fork = SessionFork {
            cached_tokens: source.evaluated_tokens.len(),
            context_tokens: source_state.shape.max_context(),
            copied_bytes,
            kv_cache_dtype: source_state.dtype,
        };
        Ok(GenerationSession {
            state: Some(state),
            activations: Some(activations),
            evaluated_tokens: source.evaluated_tokens.clone(),
            restored_tokens: source.restored_tokens.clone(),
            prefill_boundary: source.prefill_boundary,
            mirostat: source.mirostat,
            adaptive_controller: source.adaptive_controller.clone(),
            correctable_controller: source.correctable_controller.clone(),
            last_replay: SessionReplay {
                reuse_class: SessionReuseClass::DeviceFork,
                cached_tokens: source.evaluated_tokens.len(),
                reused_tokens: source.evaluated_tokens.len(),
                replayed_tokens: 0,
                computed_tokens: 0,
            },
            pending_fork: Some(fork),
            last_fork: Some(fork),
            pending_wake: None,
            last_hibernation: None,
        })
    }

    /// Moves one complete live session into host-owned buffers.
    pub fn hibernate_session(
        &mut self,
        source: &mut GenerationSession<B>,
    ) -> Result<HibernatedSession, RuntimeError> {
        let source_state = source
            .state
            .as_ref()
            .ok_or(RuntimeError::SessionForkUnavailable)?;
        let source_activations = source
            .activations
            .as_ref()
            .ok_or(RuntimeError::SessionForkUnavailable)?;
        if source.evaluated_tokens.is_empty() {
            return Err(RuntimeError::SessionForkUnavailable);
        }
        let state = HibernatedKvState::capture(&mut self.backend, source_state)?;
        let activations = HibernatedActivations::capture(&mut self.backend, source_activations)?;
        let host_bytes = source_state
            ._allocator
            .capacity_bytes()
            .checked_add(4)
            .and_then(|bytes| bytes.checked_add(Activations::<B>::bytes(&self.model.config).ok()?))
            .ok_or(RuntimeError::SizeOverflow)?;
        let record = SessionHibernation {
            cached_tokens: source.evaluated_tokens.len(),
            context_tokens: source_state.shape.max_context(),
            host_bytes,
            kv_cache_dtype: source_state.dtype,
        };
        let hibernated = HibernatedSession {
            state,
            activations,
            evaluated_tokens: source.evaluated_tokens.clone(),
            restored_tokens: source.restored_tokens.clone(),
            prefill_boundary: source.prefill_boundary,
            mirostat: source.mirostat,
            adaptive_controller: source.adaptive_controller.clone(),
            correctable_controller: source.correctable_controller.clone(),
            record,
        };
        source.invalidate();
        Ok(hibernated)
    }

    /// Restores one host-owned session without replaying its token prefix.
    pub fn wake_session(
        &mut self,
        source: HibernatedSession,
    ) -> Result<GenerationSession<B>, RuntimeError> {
        let state = source.state.restore(&mut self.backend)?;
        let activations = source.activations.restore(&mut self.backend)?;
        self.backend.synchronize()?;
        Ok(GenerationSession {
            state: Some(state),
            activations: Some(activations),
            evaluated_tokens: source.evaluated_tokens,
            restored_tokens: source.restored_tokens,
            prefill_boundary: source.prefill_boundary,
            mirostat: source.mirostat,
            adaptive_controller: source.adaptive_controller,
            correctable_controller: source.correctable_controller,
            last_replay: SessionReplay {
                reuse_class: SessionReuseClass::HostWake,
                cached_tokens: source.record.cached_tokens,
                reused_tokens: source.record.cached_tokens,
                replayed_tokens: 0,
                computed_tokens: 0,
            },
            pending_fork: None,
            last_fork: None,
            pending_wake: Some(source.record),
            last_hibernation: Some(source.record),
        })
    }

    /// Returns the prompt prefill method selected by the backend.
    pub fn prefill_method(&self) -> PrefillMethod {
        self.backend.prefill_method()
    }

    /// Returns the token vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }

    /// Evaluates one exact token stream and emits full-vocabulary logits.
    ///
    /// Windows overlap by one token. A window of 512 tokens scores 511
    /// next-token positions.
    pub fn evaluate_logits<F>(
        &mut self,
        tokens: &[u32],
        window_tokens: usize,
        kv_cache_dtype: KvCacheDtype,
        mut on_logits: F,
    ) -> Result<usize, RuntimeError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), RuntimeError>,
    {
        self.validate_eval_tokens(tokens, window_tokens)?;
        let attention_context = decode_graph_bucket(window_tokens)?;
        let attention_shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            attention_context,
        )?;
        let stride = window_tokens - 1;
        let mut logits = vec![0.0_f32; self.model.config.vocab_size];
        let mut scored = 0_usize;
        for start in (0..tokens.len() - 1).step_by(stride) {
            let end = start
                .checked_add(window_tokens)
                .map(|end| end.min(tokens.len()))
                .ok_or(RuntimeError::SizeOverflow)?;
            let window = &tokens[start..end];
            scored = scored
                .checked_add(self.evaluate_logits_window(
                    window,
                    start,
                    attention_shape,
                    kv_cache_dtype,
                    &mut logits,
                    &mut on_logits,
                )?)
                .ok_or(RuntimeError::SizeOverflow)?;
        }
        self.backend.synchronize()?;
        Ok(scored)
    }

    /// Evaluates fixed tokens through the chunked prefill path and emits logits.
    ///
    /// Windows overlap by one token, as in [`Self::evaluate_logits`]. Every
    /// emitted row uses the same prefill path as generation.
    pub fn evaluate_logits_chunked<F>(
        &mut self,
        tokens: &[u32],
        window_tokens: usize,
        chunk_tokens: usize,
        kv_cache_dtype: KvCacheDtype,
        mut on_logits: F,
    ) -> Result<usize, RuntimeError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), RuntimeError>,
    {
        self.validate_chunked_eval_tokens(tokens, window_tokens, chunk_tokens)?;
        let attention_context = decode_graph_bucket(window_tokens)?;
        let attention_shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            attention_context,
        )?;
        let stride = window_tokens - 1;
        let mut scored = 0_usize;
        for start in (0..tokens.len() - 1).step_by(stride) {
            let end = start
                .checked_add(window_tokens)
                .map(|end| end.min(tokens.len()))
                .ok_or(RuntimeError::SizeOverflow)?;
            let window = &tokens[start..end];
            let block_tokens = chunk_tokens.min(window.len());
            let ChunkedEvalWindow {
                mut state,
                mut full,
                mut tail,
            } = self.prepare_chunked_eval_window(
                window.len(),
                block_tokens,
                attention_shape,
                kv_cache_dtype,
            )?;
            self.process_chunked_eval_window(
                window,
                start,
                block_tokens,
                &mut state,
                &mut full,
                &mut tail,
                attention_shape,
                &mut scored,
                &mut on_logits,
            )?;
        }
        self.backend.synchronize()?;
        Ok(scored)
    }

    fn validate_eval_tokens(
        &self,
        tokens: &[u32],
        window_tokens: usize,
    ) -> Result<(), RuntimeError> {
        if tokens.len() < 2 {
            return Err(RuntimeError::TooFewEvalTokens);
        }
        if window_tokens < 2 {
            return Err(RuntimeError::EvalWindowTooSmall);
        }
        if window_tokens > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: window_tokens,
                capacity: self.model.config.context_length,
            });
        }
        self.validate_eval_vocab(tokens)
    }

    fn validate_chunked_eval_tokens(
        &self,
        tokens: &[u32],
        window_tokens: usize,
        chunk_tokens: usize,
    ) -> Result<(), RuntimeError> {
        if tokens.len() < 2 {
            return Err(RuntimeError::TooFewEvalTokens);
        }
        if window_tokens < 2 {
            return Err(RuntimeError::EvalWindowTooSmall);
        }
        if chunk_tokens == 0 {
            return Err(BackendError::Zero {
                field: "evaluation prefill chunk tokens",
            }
            .into());
        }
        if self.backend.prefill_method() == PrefillMethod::SequentialDecode {
            return Err(BackendError::operation(
                "evaluate chunked prefill logits",
                "the backend selects sequential decode prefill",
            )
            .into());
        }
        if window_tokens > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: window_tokens,
                capacity: self.model.config.context_length,
            });
        }
        self.validate_eval_vocab(tokens)
    }

    fn validate_eval_vocab(&self, tokens: &[u32]) -> Result<(), RuntimeError> {
        for (index, token) in tokens.iter().copied().enumerate() {
            if usize::try_from(token)
                .map(|token| token >= self.model.config.vocab_size)
                .unwrap_or(true)
            {
                return Err(RuntimeError::EvalTokenOutOfRange {
                    index,
                    token,
                    vocab_size: self.model.config.vocab_size,
                });
            }
        }
        Ok(())
    }

    fn evaluate_logits_window<F>(
        &mut self,
        window: &[u32],
        start: usize,
        attention_shape: AttentionShape,
        kv_cache_dtype: KvCacheDtype,
        logits: &mut [f32],
        on_logits: &mut F,
    ) -> Result<usize, RuntimeError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), RuntimeError>,
    {
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            kv_cache_dtype,
        )?;
        let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
        let mut scored = 0_usize;
        for (offset, token) in window.iter().copied().enumerate() {
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            self.forward(&mut state, &mut activations, attention_shape)?;
            if offset + 1 < window.len() {
                self.backend.read_f32(&activations.logits, logits)?;
                on_logits(start + offset + 1, logits)?;
                scored = scored.checked_add(1).ok_or(RuntimeError::SizeOverflow)?;
            }
        }
        Ok(scored)
    }

    fn prepare_chunked_eval_window(
        &mut self,
        window_tokens: usize,
        block_tokens: usize,
        attention_shape: AttentionShape,
        kv_cache_dtype: KvCacheDtype,
    ) -> Result<ChunkedEvalWindow<B>, RuntimeError> {
        let plan = PrefillPlan::new(
            block_tokens,
            window_tokens,
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            self.model.config.n_embd,
            self.model.config.n_ff,
            self.model.config.n_ff.max(self.model.config.vocab_size),
        )?;
        self.backend.prepare_prefill(plan)?;
        let state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            kv_cache_dtype,
        )?;
        let full =
            PrefillEvalActivations::new(&mut self.backend, &self.model.config, block_tokens)?;
        let remainder = window_tokens % block_tokens;
        let tail = if remainder == 0 {
            None
        } else {
            Some(PrefillEvalActivations::new(
                &mut self.backend,
                &self.model.config,
                remainder,
            )?)
        };
        Ok(ChunkedEvalWindow { state, full, tail })
    }

    #[allow(clippy::too_many_arguments)]
    fn process_chunked_eval_window<F>(
        &mut self,
        window: &[u32],
        start: usize,
        block_tokens: usize,
        state: &mut KvState<B>,
        full: &mut PrefillEvalActivations<B>,
        tail: &mut Option<PrefillEvalActivations<B>>,
        attention_shape: AttentionShape,
        scored: &mut usize,
        on_logits: &mut F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), RuntimeError>,
    {
        for offset in (0..window.len()).step_by(block_tokens) {
            let block_end = (offset + block_tokens).min(window.len());
            let block = &window[offset..block_end];
            let activations = if block.len() == block_tokens {
                &mut *full
            } else {
                tail.as_mut().ok_or(RuntimeError::SizeOverflow)?
            };
            self.forward_prefill_chunk(
                block,
                offset,
                &mut state.layers,
                &mut activations.forward,
                attention_shape,
            )?;
            self.finish_prefill_logits_batch(activations, block.len())?;
            self.backend
                .read_f32(&activations.logits, &mut activations.host_logits)?;
            self.emit_chunked_eval_rows(
                ChunkedEvalRows {
                    window,
                    start,
                    offset,
                    block,
                    activations,
                    scored,
                },
                on_logits,
            )?;
            state.position = block_end;
        }
        Ok(())
    }

    fn emit_chunked_eval_rows<F>(
        &self,
        rows: ChunkedEvalRows<'_, B>,
        on_logits: &mut F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), RuntimeError>,
    {
        for row_index in 0..rows.block.len() {
            let window_position = rows.offset + row_index;
            if window_position + 1 >= rows.window.len() {
                break;
            }
            let row_start = row_index
                .checked_mul(self.model.config.vocab_size)
                .ok_or(RuntimeError::SizeOverflow)?;
            let row_end = row_start
                .checked_add(self.model.config.vocab_size)
                .ok_or(RuntimeError::SizeOverflow)?;
            on_logits(
                rows.start + window_position + 1,
                rows.activations
                    .host_logits
                    .get(row_start..row_end)
                    .ok_or(RuntimeError::SizeOverflow)?,
            )?;
            let next = (*rows.scored)
                .checked_add(1)
                .ok_or(RuntimeError::SizeOverflow)?;
            *rows.scored = next;
        }
        Ok(())
    }

    /// Measures one fixed-token prefill and contiguous decode repetition.
    ///
    /// The decode timer includes one token D2H copy and allocation-free
    /// detokenization for each evaluation. Graph capture and the initial
    /// post-prefill sample are outside the timer.
    pub fn benchmark_decode(
        &mut self,
        prompt_tokens: &[u32],
        decode_tokens: usize,
        kv_cache_dtype: KvCacheDtype,
        decode_execution: DecodeExecution,
    ) -> Result<DecodeBenchmarkRun, RuntimeError> {
        self.benchmark_decode_with_prefill_chunk(
            prompt_tokens,
            decode_tokens,
            kv_cache_dtype,
            decode_execution,
            DEFAULT_PREFILL_CHUNK_TOKENS,
        )
    }

    /// Measures decode with an explicit prompt position-block size.
    pub fn benchmark_decode_with_prefill_chunk(
        &mut self,
        prompt_tokens: &[u32],
        decode_tokens: usize,
        kv_cache_dtype: KvCacheDtype,
        decode_execution: DecodeExecution,
        prefill_chunk_tokens: usize,
    ) -> Result<DecodeBenchmarkRun, RuntimeError> {
        let attention_shape =
            self.validate_benchmark_decode(prompt_tokens, decode_tokens, decode_execution)?;
        let BenchmarkState {
            mut state,
            mut activations,
            mut prepared,
        } = self.prepare_benchmark_state(
            prompt_tokens.len(),
            attention_shape,
            kv_cache_dtype,
            prefill_chunk_tokens,
        )?;

        let prefill_start = Instant::now();
        let prefill = self.run_prompt_prefill(
            prompt_tokens,
            &mut state,
            &mut activations,
            attention_shape,
            &mut prepared,
            || false,
        )?;
        self.backend.synchronize()?;
        let prefill_duration = prefill_start.elapsed();

        let mut snapshots = Vec::new();
        self.sample(
            &mut activations,
            LogitCapture::Disabled,
            &mut snapshots,
            state.position - 1,
        )?;
        let ttft_duration = prefill_start.elapsed();
        let use_graph = self.prepare_benchmark_decode_graph(
            decode_execution,
            &mut state,
            &mut activations,
            attention_shape,
        )?;

        let decode_start = Instant::now();
        let (detokenized_bytes, transcript) = self.run_benchmark_decode_steps(
            &mut state,
            &mut activations,
            attention_shape,
            decode_tokens,
            use_graph,
            &mut snapshots,
        )?;
        let decode_duration = decode_start.elapsed();
        Ok(DecodeBenchmarkRun {
            prompt_tokens: prompt_tokens.len(),
            decode_tokens,
            prefill_duration,
            ttft_duration,
            decode_duration,
            detokenized_bytes,
            prefill_method: self.backend.prefill_method(),
            prefill_workspace: prefill.workspace,
            transcript_sha256: token_stream_sha256(&transcript),
        })
    }

    fn validate_benchmark_decode(
        &self,
        prompt_tokens: &[u32],
        decode_tokens: usize,
        decode_execution: DecodeExecution,
    ) -> Result<AttentionShape, RuntimeError> {
        if decode_execution != DecodeExecution::Eager
            && !self.model.config.architecture.decode_graph_supported()
        {
            return Err(RuntimeError::UnsupportedDecodeGraph {
                architecture: self.model.config.architecture.name(),
            });
        }
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if decode_tokens == 0 {
            return Err(RuntimeError::ZeroGeneration);
        }
        let requested_context = prompt_tokens
            .len()
            .checked_add(decode_tokens)
            .ok_or(RuntimeError::SizeOverflow)?;
        if requested_context > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: requested_context,
                capacity: self.model.config.context_length,
            });
        }
        let context_bucket = decode_graph_bucket(requested_context)?;
        AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )
        .map_err(Into::into)
    }

    fn prepare_benchmark_decode_graph(
        &mut self,
        decode_execution: DecodeExecution,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
    ) -> Result<bool, RuntimeError> {
        let use_graph =
            decode_execution == DecodeExecution::Graph && self.backend.decode_graph_supported();
        if use_graph {
            self.capture_decode_graph(state, activations, attention_shape)?;
        }
        Ok(use_graph)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_benchmark_decode_steps(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        decode_tokens: usize,
        use_graph: bool,
        snapshots: &mut Vec<LogitSnapshot>,
    ) -> Result<(usize, Vec<u32>), RuntimeError> {
        let mut detokenized = Vec::with_capacity(self.model.tokenizer.max_token_bytes());
        let mut transcript = Vec::with_capacity(decode_tokens);
        let mut detokenized_bytes = 0_usize;
        for _ in 0..decode_tokens {
            let token = self.run_benchmark_decode_step(
                state,
                activations,
                attention_shape,
                use_graph,
                snapshots,
            )?;
            transcript.push(token);
            self.model
                .tokenizer
                .token_bytes_into(token, &mut detokenized)?;
            detokenized_bytes = detokenized_bytes
                .checked_add(detokenized.len())
                .ok_or(RuntimeError::SizeOverflow)?;
        }
        Ok((detokenized_bytes, transcript))
    }

    fn run_benchmark_decode_step(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        use_graph: bool,
        snapshots: &mut Vec<LogitSnapshot>,
    ) -> Result<u32, RuntimeError> {
        if use_graph {
            self.backend.replay_decode_graph()?;
            state.position = state
                .position
                .checked_add(1)
                .ok_or(RuntimeError::SizeOverflow)?;
        } else {
            self.forward(state, activations, attention_shape)?;
            self.enqueue_sample(activations)?;
        }
        self.read_sample(
            activations,
            LogitCapture::Disabled,
            snapshots,
            state.position - 1,
        )
    }

    /// Measures one prompt prefill through the first sampled token.
    pub fn benchmark_prefill(
        &mut self,
        prompt_tokens: &[u32],
        kv_cache_dtype: KvCacheDtype,
        chunk_tokens: usize,
    ) -> Result<PrefillBenchmarkRun, RuntimeError> {
        let context_bucket = self.validate_benchmark_prompt(prompt_tokens)?;
        let attention_shape = self.benchmark_attention_shape(context_bucket)?;
        let BenchmarkState {
            mut state,
            mut activations,
            mut prepared,
        } = self.prepare_benchmark_state(
            prompt_tokens.len(),
            attention_shape,
            kv_cache_dtype,
            chunk_tokens,
        )?;
        let started = Instant::now();
        let prefill = self.run_prompt_prefill(
            prompt_tokens,
            &mut state,
            &mut activations,
            attention_shape,
            &mut prepared,
            || false,
        )?;
        self.backend.synchronize()?;
        let prefill_duration = started.elapsed();
        let mut snapshots = Vec::new();
        self.sample(
            &mut activations,
            LogitCapture::Disabled,
            &mut snapshots,
            state.position - 1,
        )?;
        let ttft_duration = started.elapsed();
        Ok(PrefillBenchmarkRun {
            prompt_tokens: prompt_tokens.len(),
            prefill_duration,
            ttft_duration,
            prefill_method: self.backend.prefill_method(),
            prefill_workspace: prefill.workspace,
        })
    }

    fn validate_benchmark_prompt(&self, prompt_tokens: &[u32]) -> Result<usize, RuntimeError> {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if prompt_tokens.len() > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: prompt_tokens.len(),
                capacity: self.model.config.context_length,
            });
        }
        decode_graph_bucket(prompt_tokens.len()).map_err(Into::into)
    }

    fn benchmark_attention_shape(
        &self,
        context_bucket: usize,
    ) -> Result<AttentionShape, RuntimeError> {
        AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )
        .map_err(Into::into)
    }

    fn prepare_benchmark_state(
        &mut self,
        prompt_tokens: usize,
        attention_shape: AttentionShape,
        kv_cache_dtype: KvCacheDtype,
        chunk_tokens: usize,
    ) -> Result<BenchmarkState<B>, RuntimeError> {
        let state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            kv_cache_dtype,
        )?;
        let activations = Activations::new(&mut self.backend, &self.model.config)?;
        let prepared = if kv_cache_dtype == KvCacheDtype::Q8 && !self.backend.q8_prefill_supported()
        {
            PreparedPrefill::Sequential
        } else {
            self.prepare_prompt_prefill(prompt_tokens, prompt_tokens, chunk_tokens)?
        };
        Ok(BenchmarkState {
            state,
            activations,
            prepared,
        })
    }

    /// Compares chunked prefill with sequential decode prefill.
    ///
    /// This diagnostic uses an FP16 KV cache and eager greedy continuation.
    /// It runs chunked prefill twice to check fixed-order determinism.
    pub fn characterize_prefill(
        &mut self,
        prompt_tokens: &[u32],
        continuation_tokens: usize,
        chunk_tokens: usize,
    ) -> Result<PrefillCharacterization, RuntimeError> {
        let attention_shape =
            self.validate_prefill_characterization(prompt_tokens, continuation_tokens)?;
        let (sequential_kv, sequential_tokens) = self.run_sequential_characterization(
            prompt_tokens,
            continuation_tokens,
            attention_shape,
        )?;
        let mut prepared =
            self.prepare_prompt_prefill(prompt_tokens.len(), prompt_tokens.len(), chunk_tokens)?;
        let (chunked_kv, chunked_tokens) = self.run_chunked_characterization(
            prompt_tokens,
            continuation_tokens,
            attention_shape,
            &mut prepared,
        )?;
        let (repeated_kv, repeated_chunked_tokens) = self.run_chunked_characterization(
            prompt_tokens,
            continuation_tokens,
            attention_shape,
            &mut prepared,
        )?;

        Ok(PrefillCharacterization {
            prompt_tokens: prompt_tokens.len(),
            chunk_tokens: chunk_tokens.min(prompt_tokens.len()),
            layers: compare_prefill_kv(&sequential_kv, &chunked_kv),
            repeated_layers: compare_prefill_kv(&chunked_kv, &repeated_kv),
            sequential_tokens,
            chunked_tokens,
            repeated_chunked_tokens,
        })
    }

    fn validate_prefill_characterization(
        &self,
        prompt_tokens: &[u32],
        continuation_tokens: usize,
    ) -> Result<AttentionShape, RuntimeError> {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if continuation_tokens == 0 {
            return Err(RuntimeError::ZeroGeneration);
        }
        if self.backend.prefill_method() == PrefillMethod::SequentialDecode {
            return Err(BackendError::operation(
                "characterize chunked prefill",
                "the backend selects sequential decode prefill",
            )
            .into());
        }
        let requested_context = prompt_tokens
            .len()
            .checked_add(continuation_tokens.saturating_sub(1))
            .ok_or(RuntimeError::SizeOverflow)?;
        if requested_context > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: requested_context,
                capacity: self.model.config.context_length,
            });
        }
        let context_bucket = decode_graph_bucket(requested_context)?;
        AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )
        .map_err(Into::into)
    }

    fn run_sequential_characterization(
        &mut self,
        prompt_tokens: &[u32],
        continuation_tokens: usize,
        attention_shape: AttentionShape,
    ) -> Result<(PrefillKvSnapshot, Vec<u32>), RuntimeError> {
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            KvCacheDtype::F16,
        )?;
        let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
        for token in prompt_tokens.iter().copied() {
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            self.forward(&mut state, &mut activations, attention_shape)?;
        }
        self.backend.synchronize()?;
        let kv = self.read_prefill_kv(&state, attention_shape, prompt_tokens.len())?;
        let tokens = self.greedy_continuation(
            &mut state,
            &mut activations,
            attention_shape,
            continuation_tokens,
        )?;
        Ok((kv, tokens))
    }

    fn run_chunked_characterization(
        &mut self,
        prompt_tokens: &[u32],
        continuation_tokens: usize,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
    ) -> Result<(PrefillKvSnapshot, Vec<u32>), RuntimeError> {
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            KvCacheDtype::F16,
        )?;
        let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
        self.run_prompt_prefill(
            prompt_tokens,
            &mut state,
            &mut activations,
            attention_shape,
            prepared,
            || false,
        )?;
        self.backend.synchronize()?;
        let kv = self.read_prefill_kv(&state, attention_shape, prompt_tokens.len())?;
        let tokens = self.greedy_continuation(
            &mut state,
            &mut activations,
            attention_shape,
            continuation_tokens,
        )?;
        Ok((kv, tokens))
    }

    /// Compares one verifier pass with consecutive one-position decode passes.
    pub fn characterize_verify(
        &mut self,
        prompt_tokens: &[u32],
        verify_tokens: &[u32],
        kv_cache_dtype: KvCacheDtype,
    ) -> Result<VerifyCharacterization, RuntimeError> {
        let attention_shape =
            self.validate_verify_characterization(prompt_tokens, verify_tokens)?;
        let sequential = self.run_sequential_verify_characterization(
            prompt_tokens,
            verify_tokens,
            attention_shape,
            kv_cache_dtype,
        )?;
        let verifier = self.run_batched_verify_characterization(
            prompt_tokens,
            verify_tokens,
            attention_shape,
            kv_cache_dtype,
        )?;
        let mismatching_floats = sequential
            .iter()
            .zip(&verifier)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        Ok(VerifyCharacterization {
            positions: verify_tokens.len(),
            compared_floats: sequential.len(),
            mismatching_floats,
        })
    }

    fn validate_verify_characterization(
        &self,
        prompt_tokens: &[u32],
        verify_tokens: &[u32],
    ) -> Result<AttentionShape, RuntimeError> {
        if prompt_tokens.is_empty() || verify_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if !self.backend.verify_supported() {
            return Err(BackendError::operation(
                "characterize verifier",
                "the backend does not support batched verification",
            )
            .into());
        }
        if verify_tokens.len() > 8 {
            return Err(BackendError::operation(
                "characterize verifier",
                "verifier width exceeds eight positions",
            )
            .into());
        }
        let context = prompt_tokens
            .len()
            .checked_add(verify_tokens.len())
            .ok_or(RuntimeError::SizeOverflow)?;
        if context > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: context,
                capacity: self.model.config.context_length,
            });
        }
        AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            decode_graph_bucket(context).map_err(RuntimeError::from)?,
        )
        .map_err(Into::into)
    }

    fn run_sequential_verify_characterization(
        &mut self,
        prompt_tokens: &[u32],
        verify_tokens: &[u32],
        attention_shape: AttentionShape,
        kv_cache_dtype: KvCacheDtype,
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut sequential = Vec::with_capacity(
            verify_tokens
                .len()
                .checked_mul(self.model.config.vocab_size)
                .ok_or(RuntimeError::SizeOverflow)?,
        );
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            kv_cache_dtype,
        )?;
        let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
        self.replay_verify_prompt(prompt_tokens, &mut state, &mut activations, attention_shape)?;
        let mut row = vec![0.0; self.model.config.vocab_size];
        self.collect_sequential_verify(
            verify_tokens,
            &mut state,
            &mut activations,
            attention_shape,
            &mut row,
            &mut sequential,
        )?;
        Ok(sequential)
    }

    fn replay_verify_prompt(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        for token in prompt_tokens.iter().copied() {
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            self.forward(state, activations, attention_shape)?;
        }
        Ok(())
    }

    fn collect_sequential_verify(
        &mut self,
        verify_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        row: &mut [f32],
        sequential: &mut Vec<f32>,
    ) -> Result<(), RuntimeError> {
        for token in verify_tokens.iter().copied() {
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            self.forward(state, activations, attention_shape)?;
            self.backend.read_f32(&activations.logits, row)?;
            sequential.extend_from_slice(row);
        }
        Ok(())
    }

    fn run_batched_verify_characterization(
        &mut self,
        prompt_tokens: &[u32],
        verify_tokens: &[u32],
        attention_shape: AttentionShape,
        kv_cache_dtype: KvCacheDtype,
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            kv_cache_dtype,
        )?;
        let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
        for token in prompt_tokens.iter().copied() {
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            self.forward(&mut state, &mut activations, attention_shape)?;
        }
        let mut verify =
            VerifyActivations::new(&mut self.backend, &self.model.config, verify_tokens.len())?;
        self.forward_verify(&mut state, &mut verify, attention_shape, verify_tokens)?;
        self.backend
            .read_f32(&verify.logits, &mut verify.host_logits)?;
        Ok(verify.host_logits)
    }

    /// Generates tokens from a text prompt and streams each token.
    pub fn generate<F, C>(
        &mut self,
        prompt: &str,
        options: GenerateOptions,
        on_token: F,
        cancelled: C,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
    {
        let prompt_tokens = self.model.tokenizer.encode(prompt)?;
        self.generate_tokens(&prompt_tokens, options, on_token, cancelled)
    }

    /// Generates tokens from an exact prompt token stream.
    pub fn generate_tokens<F, C>(
        &mut self,
        prompt_tokens: &[u32],
        options: GenerateOptions,
        on_token: F,
        cancelled: C,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
    {
        let mut session = GenerationSession::new();
        self.generate_session_tokens(&mut session, prompt_tokens, options, on_token, cancelled)
    }

    fn validate_generation_request(
        &self,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
    ) -> Result<AttentionShape, RuntimeError> {
        self.validate_generation_options(options, prompt_tokens)?;
        self.validate_generation_tokens(prompt_tokens)?;
        let context_bucket =
            self.generation_context_bucket(prompt_tokens.len(), options.max_tokens)?;
        AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )
        .map_err(Into::into)
    }

    fn validate_generation_options(
        &self,
        options: &GenerateOptions,
        prompt_tokens: &[u32],
    ) -> Result<(), RuntimeError> {
        if options.decode_execution != DecodeExecution::Eager
            && !self.model.config.architecture.decode_graph_supported()
        {
            return Err(RuntimeError::UnsupportedDecodeGraph {
                architecture: self.model.config.architecture.name(),
            });
        }
        if options.max_tokens == 0 {
            return Err(RuntimeError::ZeroGeneration);
        }
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if options.mirostat.is_some() && options.speculation != Speculation::Disabled {
            return Err(RuntimeError::MirostatSpeculation);
        }
        Ok(())
    }

    fn generation_context_bucket(
        &self,
        prompt_tokens: usize,
        max_tokens: usize,
    ) -> Result<usize, RuntimeError> {
        let decode_context = max_tokens
            .checked_sub(1)
            .ok_or(RuntimeError::SizeOverflow)?;
        let requested_context = prompt_tokens
            .checked_add(decode_context)
            .ok_or(RuntimeError::SizeOverflow)?;
        if requested_context > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: requested_context,
                capacity: self.model.config.context_length,
            });
        }
        decode_graph_bucket(requested_context).map_err(Into::into)
    }

    fn validate_generation_tokens(&self, prompt_tokens: &[u32]) -> Result<(), RuntimeError> {
        for (index, token) in prompt_tokens.iter().copied().enumerate() {
            if usize::try_from(token)
                .map(|token| token >= self.model.config.vocab_size)
                .unwrap_or(true)
            {
                return Err(RuntimeError::EvalTokenOutOfRange {
                    index,
                    token,
                    vocab_size: self.model.config.vocab_size,
                });
            }
        }
        Ok(())
    }

    fn prepare_generation_state(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
        attention_shape: AttentionShape,
    ) -> Result<GenerationPreparation<B>, RuntimeError> {
        let reuse = self.generation_reuse(session, prompt_tokens, options, attention_shape);
        let (state, activations) =
            self.take_generation_state(session, options, attention_shape, &reuse)?;
        let verify_activations = self.configure_verify_activations(options)?;
        Ok(GenerationPreparation {
            attention_shape,
            state,
            activations,
            reuse_class: reuse.reuse_class,
            reused_tokens: reuse.reused_tokens,
            cached_tokens: reuse.cached_tokens,
            restored_common: reuse.restored_common,
            common_tokens: reuse.common_tokens,
            replay_prefill_boundary: reuse.replay_prefill_boundary,
            verify_activations,
        })
    }

    fn generation_reuse(
        &self,
        session: &GenerationSession<B>,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
        attention_shape: AttentionShape,
    ) -> GenerationReuse {
        let cached_tokens = session.evaluated_tokens.len();
        let common_tokens = common_prefix(prompt_tokens, &session.evaluated_tokens);
        let restored_common = common_prefix(prompt_tokens, &session.restored_tokens);
        let replay_prefill_boundary = session.prefill_boundary.min(prompt_tokens.len());
        let compatible = session
            .state
            .as_ref()
            .map(|state| state.shape == attention_shape && state.dtype == options.kv_cache_dtype)
            .unwrap_or(false)
            && session.activations.is_some();
        let reuse_class = Self::base_reuse_class(
            compatible,
            common_tokens,
            cached_tokens,
            restored_common,
            prompt_tokens.len(),
        );
        let reuse_class =
            Self::override_reuse_class(compatible, common_tokens, session, reuse_class);
        let mut reused_tokens = if compatible { common_tokens } else { 0 };
        if compatible && reused_tokens == prompt_tokens.len() && reused_tokens < cached_tokens {
            reused_tokens = reused_tokens.saturating_sub(1);
        }
        GenerationReuse {
            reuse_class,
            compatible,
            reused_tokens,
            cached_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
        }
    }

    fn base_reuse_class(
        compatible: bool,
        common_tokens: usize,
        cached_tokens: usize,
        restored_common: usize,
        prompt_tokens: usize,
    ) -> SessionReuseClass {
        if compatible {
            Self::compatible_reuse_class(common_tokens, cached_tokens, prompt_tokens)
        } else {
            Self::restored_or_cold_reuse_class(common_tokens, cached_tokens, restored_common)
        }
    }

    fn compatible_reuse_class(
        common_tokens: usize,
        cached_tokens: usize,
        prompt_tokens: usize,
    ) -> SessionReuseClass {
        if common_tokens == prompt_tokens && common_tokens == cached_tokens {
            SessionReuseClass::ExactRepeat
        } else if common_tokens == cached_tokens && common_tokens > 0 {
            SessionReuseClass::AppendOnly
        } else {
            SessionReuseClass::ArbitraryBranch
        }
    }

    fn restored_or_cold_reuse_class(
        common_tokens: usize,
        cached_tokens: usize,
        restored_common: usize,
    ) -> SessionReuseClass {
        if (common_tokens == cached_tokens && common_tokens > 0) || restored_common > 0 {
            // Replay host tokens when a changed bucket invalidates device buffers.
            SessionReuseClass::RestoreReplay
        } else {
            SessionReuseClass::Cold
        }
    }

    fn override_reuse_class(
        compatible: bool,
        common_tokens: usize,
        session: &GenerationSession<B>,
        reuse_class: SessionReuseClass,
    ) -> SessionReuseClass {
        if compatible && common_tokens > 0 && session.pending_wake.is_some() {
            SessionReuseClass::HostWake
        } else if compatible && common_tokens > 0 && session.pending_fork.is_some() {
            SessionReuseClass::DeviceFork
        } else {
            reuse_class
        }
    }

    fn take_generation_state(
        &mut self,
        session: &mut GenerationSession<B>,
        options: &GenerateOptions,
        attention_shape: AttentionShape,
        reuse: &GenerationReuse,
    ) -> Result<(KvState<B>, Activations<B>), RuntimeError> {
        if reuse.compatible {
            let mut state = session.state.take().ok_or(RuntimeError::SizeOverflow)?;
            state.position = reuse.reused_tokens;
            let activations = session
                .activations
                .take()
                .ok_or(RuntimeError::SizeOverflow)?;
            Ok((state, activations))
        } else {
            session.state = None;
            session.activations = None;
            Ok((
                KvState::new(
                    &mut self.backend,
                    self.model.config.n_layer,
                    attention_shape,
                    options.kv_cache_dtype,
                )?,
                Activations::new(&mut self.backend, &self.model.config)?,
            ))
        }
    }

    fn configure_verify_activations(
        &mut self,
        options: &GenerateOptions,
    ) -> Result<Vec<Option<VerifyActivations<B>>>, RuntimeError> {
        let mut verify_activations: Vec<Option<VerifyActivations<B>>> =
            (0..9).map(|_| None).collect();
        if let Some(positions) = self.verifier_positions(options)? {
            verify_activations[positions] = Some(VerifyActivations::new(
                &mut self.backend,
                &self.model.config,
                positions,
            )?);
        }
        Ok(verify_activations)
    }

    fn verifier_positions(&self, options: &GenerateOptions) -> Result<Option<usize>, RuntimeError> {
        if !self.backend.verify_supported() || options.kv_cache_dtype == KvCacheDtype::Q8 {
            return Ok(None);
        }
        let positions = match options.speculation {
            Speculation::Suffix(drafter) => drafter
                .proposal()
                .get()
                .checked_add(1)
                .ok_or(RuntimeError::SizeOverflow)?,
            Speculation::Adaptive(drafter) => drafter.maximum_verifier_positions(),
            Speculation::Correctable(drafter) => drafter.maximum_verifier_positions(),
            Speculation::Disabled => return Ok(None),
        };
        Ok((positions <= 8).then_some(positions))
    }

    #[allow(clippy::too_many_arguments)]
    fn choose_generation_draft(
        options: &GenerateOptions,
        prompt_tokens: &[u32],
        tokens: &[u32],
        context: &mut Vec<u32>,
        state_position: usize,
        rng: SamplerRng,
        adaptive_controller: &mut Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: &mut Option<CorrectableController>,
    ) -> Result<GenerationDraft, RuntimeError> {
        match &options.speculation {
            Speculation::Disabled => Ok(Self::plain_generation_draft()),
            Speculation::Suffix(drafter) => {
                context.truncate(prompt_tokens.len());
                context.extend_from_slice(tokens);
                Ok(GenerationDraft {
                    drafted: drafter.draft(context),
                    adaptive_round: None,
                    correctable_round: None,
                    correctable_distributions: None,
                })
            }
            Speculation::Adaptive(drafter) => Self::adaptive_generation_draft(
                options,
                drafter,
                prompt_tokens,
                tokens,
                context,
                adaptive_controller,
            ),
            Speculation::Correctable(drafter) => Self::correctable_generation_draft(
                options,
                drafter,
                CorrectableDraftContext {
                    prompt_tokens,
                    tokens,
                    context,
                    state_position,
                    rng,
                    correctable_controller,
                },
            ),
        }
    }

    fn plain_generation_draft() -> GenerationDraft {
        GenerationDraft {
            drafted: Draft::Nothing,
            adaptive_round: None,
            correctable_round: None,
            correctable_distributions: None,
        }
    }

    fn adaptive_generation_draft(
        options: &GenerateOptions,
        drafter: &crate::adaptive_draft::AdaptiveDrafter,
        prompt_tokens: &[u32],
        tokens: &[u32],
        context: &mut Vec<u32>,
        adaptive_controller: &mut Option<crate::adaptive_draft::AdaptiveController>,
    ) -> Result<GenerationDraft, RuntimeError> {
        let controller_start = Instant::now();
        context.truncate(prompt_tokens.len());
        context.extend_from_slice(tokens);
        let mut proposal = drafter.propose(context);
        let available_proposals = options
            .max_tokens
            .saturating_sub(tokens.len())
            .saturating_sub(1);
        if let Some(candidate) = proposal.as_mut() {
            candidate.tokens.truncate(available_proposals);
            if candidate.tokens.is_empty() {
                proposal = None;
            }
        }
        let proposal_tokens = proposal
            .as_ref()
            .map_or(0, |proposal| proposal.tokens.len());
        let controller = adaptive_controller
            .as_mut()
            .expect("adaptive speculation creates a controller");
        if let Some(proposal) = proposal.as_ref() {
            controller
                .record_source(proposal.source)
                .expect("one generation cannot overflow adaptive counters");
        }
        let decision = controller.decide(proposal_tokens);
        let controller_duration = controller_start.elapsed();
        let (drafted, verifier_positions) = match decision {
            crate::adaptive_draft::AdaptiveDecision::Plain { .. } => (
                Draft::Nothing,
                std::num::NonZeroUsize::new(1).expect("one is nonzero"),
            ),
            crate::adaptive_draft::AdaptiveDecision::Speculate {
                verifier_positions, ..
            } => {
                let mut drafted = proposal
                    .take()
                    .expect("speculation requires a proposal")
                    .tokens;
                drafted.truncate(verifier_positions.get() - 1);
                (Draft::Tokens(drafted), verifier_positions)
            }
        };
        Ok(GenerationDraft {
            drafted,
            adaptive_round: Some((verifier_positions, controller_duration)),
            correctable_round: None,
            correctable_distributions: None,
        })
    }

    fn correctable_generation_draft(
        options: &GenerateOptions,
        drafter: &crate::correctable::CorrectableDrafter,
        draft_context: CorrectableDraftContext<'_>,
    ) -> Result<GenerationDraft, RuntimeError> {
        let CorrectableDraftContext {
            prompt_tokens,
            tokens,
            context,
            state_position,
            rng,
            correctable_controller,
        } = draft_context;
        let controller_start = Instant::now();
        context.truncate(prompt_tokens.len());
        context.extend_from_slice(tokens);
        let remaining = options.max_tokens.saturating_sub(tokens.len());
        let maximum_positions = remaining.min(drafter.maximum_verifier_positions());
        let available = if maximum_positions >= 2 {
            drafter.available_plans(context)
        } else {
            [false; 4]
        };
        let controller = correctable_controller
            .as_mut()
            .expect("correctable speculation creates a controller");
        let decision = controller.decide(available, maximum_positions.max(1))?;
        let controller_duration = controller_start.elapsed();
        match decision {
            crate::CorrectableDecision::Plain { .. } => Ok(GenerationDraft {
                drafted: Draft::Nothing,
                adaptive_round: None,
                correctable_round: Some((
                    None,
                    std::num::NonZeroUsize::new(1).expect("one is nonzero"),
                    controller_duration,
                )),
                correctable_distributions: None,
            }),
            crate::CorrectableDecision::Speculate {
                plan,
                verifier_positions,
                ..
            } => {
                let proposal = drafter.propose(
                    plan,
                    context,
                    verifier_positions.get() - 1,
                    rng,
                    state_position as u64,
                )?;
                Ok(match proposal {
                    Some(proposal) => {
                        let drafted = proposal.tokens();
                        let distributions = Some(proposal.distributions());
                        let correctable_round = Some((
                            Some(proposal.plan),
                            std::num::NonZeroUsize::new(drafted.len() + 1)
                                .expect("a proposal has one or more tokens"),
                            controller_duration,
                        ));
                        GenerationDraft {
                            drafted: Draft::Tokens(drafted),
                            adaptive_round: None,
                            correctable_round,
                            correctable_distributions: distributions,
                        }
                    }
                    None => GenerationDraft {
                        drafted: Draft::Nothing,
                        adaptive_round: None,
                        correctable_round: Some((
                            None,
                            std::num::NonZeroUsize::new(1).expect("one is nonzero"),
                            controller_duration,
                        )),
                        correctable_distributions: None,
                    },
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn initialize_generation_decode<F>(
        &mut self,
        session: &mut GenerationSession<B>,
        options: &GenerateOptions,
        reuse_class: SessionReuseClass,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        logits: &mut Vec<LogitSnapshot>,
        context: &[u32],
        tokens: &mut Vec<u32>,
        on_token: &mut F,
        attention_shape: AttentionShape,
    ) -> Result<GenerationInitialization, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
    {
        if options.output_constraint.is_some()
            && !matches!(options.speculation, Speculation::Disabled)
        {
            return Err(RuntimeError::ConstraintSpeculation);
        }
        let mut output_constraint = options
            .output_constraint
            .map(|_| crate::constraint::JsonObjectConstraint::new());
        let host_sampling = Self::uses_host_sampling(options, output_constraint.is_some());
        let mut mirostat = Self::restore_mirostat(session, options, reuse_class);
        let first = self.sample_first_generation(
            state,
            activations,
            options,
            rng,
            host_sampling,
            row,
            logits,
            context,
            &mut mirostat,
            &mut output_constraint,
        )?;
        emit(&self.model, first, tokens, on_token)?;
        let eos = self.model.tokenizer.eos_token();
        let profile_target = Self::generation_profile_target(options, tokens.len())?;
        let profile_start = self.begin_generation_profile(profile_target, first, eos)?;
        let use_graph = Self::generation_uses_graph(
            options,
            profile_target,
            self.backend.decode_graph_supported(),
            eos,
            first_token(tokens),
        );
        if use_graph {
            self.capture_decode_graph(state, activations, attention_shape)?;
        }
        let (adaptive_controller, correctable_controller) =
            Self::restore_generation_controllers(session, options);
        Ok(GenerationInitialization {
            host_sampling,
            mirostat,
            output_constraint,
            profile_target,
            profile_start,
            use_graph,
            adaptive_controller,
            correctable_controller,
        })
    }

    /// Generates from prompt tokens while retaining verified KV for the next call.
    pub fn generate_session_tokens<F, C>(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: GenerateOptions,
        on_token: F,
        mut cancelled: C,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
    {
        let (_, preparation) = self.prepare_generation_session(session, prompt_tokens, &options)?;
        let GenerationPreparation {
            attention_shape,
            mut state,
            mut activations,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            verify_activations,
        } = preparation;
        let prefill_tokens = &prompt_tokens[reused_tokens..];
        let split_replay = reuse_class == SessionReuseClass::RestoreReplay
            && reused_tokens == 0
            && replay_prefill_boundary > 0
            && replay_prefill_boundary < prompt_tokens.len();
        let mut prepared = self.prepare_generation_prefill(
            prefill_tokens.len(),
            prompt_tokens.len(),
            &options,
            reused_tokens,
            split_replay,
        )?;

        let prefill_start = Instant::now();
        let prefill = self.run_generation_prefill(
            prompt_tokens,
            prefill_tokens,
            replay_prefill_boundary,
            split_replay,
            &mut state,
            &mut activations,
            attention_shape,
            &mut prepared,
            &mut cancelled,
            &options,
        )?;
        self.backend.synchronize()?;
        let prefill_duration = prefill_start.elapsed();
        if prefill.cancelled {
            return self.cancelled_prefill_result(
                session,
                prompt_tokens,
                reused_tokens,
                prefill,
                prefill_duration,
            );
        }
        if cancelled() {
            session.invalidate();
            return Ok(cancelled_result(
                prompt_tokens.to_vec(),
                prefill_duration,
                self.backend.prefill_method(),
                prefill.workspace,
            ));
        }

        self.run_generation_decode(
            session,
            prompt_tokens,
            options,
            prefill_start,
            prefill_duration,
            prefill,
            attention_shape,
            state,
            activations,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            on_token,
            cancelled,
            verify_activations,
        )
    }

    fn prepare_generation_prefill(
        &mut self,
        prefill_tokens: usize,
        context_tokens: usize,
        options: &GenerateOptions,
        reused_tokens: usize,
        split_replay: bool,
    ) -> Result<PreparedPrefill<B>, RuntimeError> {
        if split_replay {
            return Ok(PreparedPrefill::Reused);
        }
        if (options.kv_cache_dtype == KvCacheDtype::Q8 && !self.backend.q8_prefill_supported())
            || reused_tokens > 0
        {
            // Appended session tokens use the same decode kernels as uninterrupted generation.
            return Ok(PreparedPrefill::Sequential);
        }
        self.prepare_prompt_prefill(prefill_tokens, context_tokens, options.prefill_chunk_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation_prefill<C>(
        &mut self,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        replay_prefill_boundary: usize,
        split_replay: bool,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
        cancelled: &mut C,
        options: &GenerateOptions,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        if !split_replay {
            return self.run_prompt_prefill(
                prefill_tokens,
                state,
                activations,
                attention_shape,
                prepared,
                cancelled,
            );
        }
        let mut initial_prepared =
            if options.kv_cache_dtype == KvCacheDtype::Q8 && !self.backend.q8_prefill_supported() {
                PreparedPrefill::Sequential
            } else {
                self.prepare_prompt_prefill(
                    replay_prefill_boundary,
                    replay_prefill_boundary,
                    options.prefill_chunk_tokens,
                )?
            };
        let initial = self.run_prompt_prefill(
            &prompt_tokens[..replay_prefill_boundary],
            state,
            activations,
            attention_shape,
            &mut initial_prepared,
            &mut *cancelled,
        )?;
        if initial.cancelled {
            return Ok(initial);
        }
        let mut continuation_prepared = PreparedPrefill::Sequential;
        let continuation = self.run_prompt_prefill(
            &prompt_tokens[replay_prefill_boundary..],
            state,
            activations,
            attention_shape,
            &mut continuation_prepared,
            &mut *cancelled,
        )?;
        Ok(PrefillExecution {
            processed_tokens: replay_prefill_boundary
                .checked_add(continuation.processed_tokens)
                .ok_or(RuntimeError::SizeOverflow)?,
            cancelled: continuation.cancelled,
            workspace: initial.workspace,
        })
    }

    fn cancelled_prefill_result(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        reused_tokens: usize,
        prefill: PrefillExecution,
        prefill_duration: Duration,
    ) -> Result<GenerationResult, RuntimeError> {
        session.invalidate();
        let processed_count = reused_tokens
            .checked_add(prefill.processed_tokens)
            .ok_or(RuntimeError::SizeOverflow)?;
        let processed = prompt_tokens
            .get(..processed_count)
            .ok_or(RuntimeError::SizeOverflow)?
            .to_vec();
        Ok(cancelled_result(
            processed,
            prefill_duration,
            self.backend.prefill_method(),
            prefill.workspace,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation_step(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        attention_shape: AttentionShape,
        drafted: &Draft,
        correctable_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        rng: SamplerRng,
        use_graph: bool,
        host_sampling: bool,
        prompt_tokens: &[u32],
        tokens: &mut [u32],
        logits: &mut Vec<LogitSnapshot>,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        mirostat: &mut Option<MirostatState>,
        output_constraint: &mut Option<crate::constraint::JsonObjectConstraint>,
    ) -> Result<GenerationStep, RuntimeError> {
        if drafted.is_empty() {
            self.run_plain_generation_step(
                state,
                activations,
                attention_shape,
                options,
                rng,
                use_graph,
                host_sampling,
                prompt_tokens,
                tokens,
                logits,
                row,
                context,
                committed,
                mirostat,
                output_constraint,
            )?;
            return Ok(GenerationStep {
                evaluations: 1,
                outcome: None,
            });
        }
        let verifier_positions = drafted.tokens().len() + 1;
        self.ensure_correctable_verifier_capacity(verify_activations, options, verifier_positions)?;
        let outcome = self.speculative_round(
            state,
            activations,
            verify_activations
                .get_mut(verifier_positions)
                .and_then(Option::as_mut),
            attention_shape,
            drafted.tokens(),
            correctable_distributions,
            options,
            rng,
            row,
            context,
            committed,
        )?;
        Ok(GenerationStep {
            evaluations: outcome.evaluations,
            outcome: Some(outcome),
        })
    }

    fn ensure_correctable_verifier_capacity(
        &mut self,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        options: &GenerateOptions,
        verifier_positions: usize,
    ) -> Result<(), RuntimeError> {
        if matches!(options.speculation, Speculation::Correctable(_))
            && self.backend.verify_supported()
            && options.kv_cache_dtype != KvCacheDtype::Q8
            && verify_activations[verifier_positions].is_none()
        {
            verify_activations[verifier_positions] = Some(VerifyActivations::new(
                &mut self.backend,
                &self.model.config,
                verifier_positions,
            )?);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_plain_generation_step(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        options: &GenerateOptions,
        rng: SamplerRng,
        use_graph: bool,
        host_sampling: bool,
        prompt_tokens: &[u32],
        tokens: &[u32],
        logits: &mut Vec<LogitSnapshot>,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        mirostat: &mut Option<MirostatState>,
        output_constraint: &mut Option<crate::constraint::JsonObjectConstraint>,
    ) -> Result<(), RuntimeError> {
        self.run_plain_forward(
            state,
            activations,
            attention_shape,
            use_graph,
            host_sampling,
        )?;
        if host_sampling {
            let position = (state.position - 1) as u64;
            context.truncate(prompt_tokens.len());
            context.extend_from_slice(tokens);
            let (token, _) = self.sample_on_host(
                activations,
                &options.sampler,
                rng,
                position,
                row,
                &options.penalties,
                context,
                mirostat.as_mut(),
                output_constraint.as_mut(),
            )?;
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            committed.push(token);
        } else {
            committed.push(self.read_sample(
                activations,
                options.logit_capture,
                logits,
                state.position - 1,
            )?);
        }
        Ok(())
    }

    fn run_plain_forward(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        use_graph: bool,
        host_sampling: bool,
    ) -> Result<(), RuntimeError> {
        if use_graph {
            let device_position =
                u32::try_from(state.position).map_err(|_| RuntimeError::ContextCapacity {
                    requested: state.position,
                    capacity: u32::MAX as usize,
                })?;
            self.backend
                .write_u32(&mut state.device_position, &[device_position])?;
            self.backend.replay_decode_graph()?;
            state.position = state
                .position
                .checked_add(1)
                .ok_or(RuntimeError::SizeOverflow)?;
        } else {
            self.forward(state, activations, attention_shape)?;
            if !host_sampling {
                self.enqueue_sample(activations)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn observe_generation_round(
        adaptive_round: Option<(std::num::NonZeroUsize, Duration)>,
        correctable_round: Option<(
            Option<crate::CorrectablePlan>,
            std::num::NonZeroUsize,
            Duration,
        )>,
        committed_len: usize,
        correctable_distributions: &Option<Vec<Distribution>>,
        adaptive_controller: &mut Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: &mut Option<CorrectableController>,
        round_duration: Duration,
        correctable_proposed: usize,
        correctable_accepted: usize,
        correctable_overlap: f64,
    ) -> Result<(), RuntimeError> {
        if let (Some((verifier_positions, controller_duration)), Some(emitted_tokens)) =
            (adaptive_round, std::num::NonZeroUsize::new(committed_len))
        {
            adaptive_controller
                .as_mut()
                .expect("adaptive rounds have a controller")
                .observe(crate::adaptive_draft::AdaptiveObservation {
                    verifier_positions,
                    emitted_tokens,
                    wall_duration: round_duration,
                    controller_duration,
                })
                .expect("runtime verifier widths are in [1, 8]");
        }
        if let (Some((plan, verifier_positions, controller_duration)), Some(emitted_tokens)) = (
            correctable_round,
            std::num::NonZeroUsize::new(committed_len),
        ) {
            correctable_controller
                .as_mut()
                .expect("correctable rounds have a controller")
                .observe(CorrectableObservation {
                    plan,
                    verifier_positions,
                    emitted_tokens,
                    proposed_tokens: correctable_proposed,
                    accepted_tokens: correctable_accepted,
                    overlap_sum: correctable_overlap,
                    overlap_proposals: if plan.is_some() {
                        correctable_distributions
                            .as_ref()
                            .map_or(0, |_| correctable_proposed.min(committed_len))
                    } else {
                        0
                    },
                    wall_duration: round_duration,
                    controller_duration,
                })?;
        }
        Ok(())
    }

    fn emit_generation_tokens<F>(
        &self,
        committed: &mut Vec<u32>,
        tokens: &mut Vec<u32>,
        max_tokens: usize,
        eos: Option<u32>,
        on_token: &mut F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
    {
        for token in committed.drain(..) {
            emit(&self.model, token, tokens, on_token)?;
            if tokens.len() >= max_tokens || Some(token) == eos {
                break;
            }
        }
        Ok(())
    }

    fn record_generation_controller_stats(
        speculation: &mut SpeculationStats,
        adaptive_controller: &Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: &Option<CorrectableController>,
    ) {
        if let Some(controller) = adaptive_controller.as_ref() {
            let adaptive = controller.stats();
            speculation.adaptive_plain_rounds = adaptive.plain_rounds;
            speculation.adaptive_speculative_rounds = adaptive.speculative_rounds;
            speculation.adaptive_below_gate_rounds = adaptive.below_gate_rounds;
            speculation.adaptive_regret_limited_rounds = adaptive.regret_limited_rounds;
            speculation.adaptive_suffix_proposals = adaptive.suffix_proposals;
            speculation.adaptive_recycling_proposals = adaptive.recycling_proposals;
            speculation.adaptive_controller_duration = adaptive.controller_duration;
            speculation.adaptive_width_rounds = adaptive.width_rounds;
        }
        if let Some(controller) = correctable_controller.as_ref() {
            let correctable = controller.stats();
            speculation.correctable_plain_rounds = correctable.plain_rounds;
            speculation.correctable_speculative_rounds = correctable.speculative_rounds;
            speculation.correctable_below_gate_rounds = correctable.below_gate_rounds;
            speculation.correctable_regret_limited_rounds = correctable.regret_limited_rounds;
            speculation.correctable_plan_rounds = correctable.plan_rounds;
            speculation.correctable_width_rounds = correctable.width_rounds;
            speculation.correctable_controller_duration = correctable.controller_duration;
            speculation.correctable_overlap_sum = correctable.overlap_sum;
            speculation.correctable_overlap_proposals =
                usize::try_from(correctable.overlap_proposals).unwrap_or(usize::MAX);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_generation_session(
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        tokens: &[u32],
        state: KvState<B>,
        activations: Activations<B>,
        reuse_class: SessionReuseClass,
        replay_prefill_boundary: usize,
        was_cancelled: bool,
        mirostat: Option<MirostatState>,
        adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: Option<CorrectableController>,
    ) {
        if was_cancelled {
            session.invalidate();
            return;
        }
        let mut evaluated_tokens = prompt_tokens.to_vec();
        evaluated_tokens.extend_from_slice(tokens);
        evaluated_tokens.truncate(state.position.min(evaluated_tokens.len()));
        session.state = Some(state);
        session.activations = Some(activations);
        session.evaluated_tokens = evaluated_tokens;
        session.restored_tokens.clear();
        session.prefill_boundary =
            if reuse_class == SessionReuseClass::Cold || replay_prefill_boundary == 0 {
                prompt_tokens.len()
            } else {
                replay_prefill_boundary
            };
        session.mirostat = mirostat;
        session.adaptive_controller = adaptive_controller;
        session.correctable_controller = correctable_controller;
        session.pending_fork = None;
        session.pending_wake = None;
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation_decode<F, C>(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: GenerateOptions,
        prefill_start: Instant,
        prefill_duration: Duration,
        prefill: PrefillExecution,
        attention_shape: AttentionShape,
        mut state: KvState<B>,
        mut activations: Activations<B>,
        reuse_class: SessionReuseClass,
        reused_tokens: usize,
        cached_tokens: usize,
        restored_common: usize,
        common_tokens: usize,
        replay_prefill_boundary: usize,
        mut on_token: F,
        mut cancelled: C,
        mut verify_activations: Vec<Option<VerifyActivations<B>>>,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
    {
        let mut tokens = Vec::with_capacity(options.max_tokens);
        let mut logits: Vec<LogitSnapshot> = Vec::new();
        let mut row: Vec<f32> = Vec::new();
        let mut committed: Vec<u32> = Vec::new();
        let mut context: Vec<u32> = prompt_tokens.to_vec();
        let mut speculation = SpeculationStats::default();
        let rng = SamplerRng::new(options.seed);
        let initialized = self.initialize_generation_decode(
            session,
            &options,
            reuse_class,
            &mut state,
            &mut activations,
            rng,
            &mut row,
            &mut logits,
            &context,
            &mut tokens,
            &mut on_token,
            attention_shape,
        )?;
        let GenerationInitialization {
            host_sampling,
            mut mirostat,
            mut output_constraint,
            profile_target,
            mut profile_start,
            use_graph,
            mut adaptive_controller,
            mut correctable_controller,
        } = initialized;
        let ttft_duration = prefill_start.elapsed();
        let eos = self.model.tokenizer.eos_token();
        let mut decode_duration = Duration::ZERO;
        let mut verify_duration = Duration::ZERO;
        let mut decode_evaluations = 0;
        let mut profile_steps = 0;
        let mut decode_profile = None;
        let was_cancelled = self.run_generation_iterations(
            prompt_tokens,
            &options,
            attention_shape,
            &mut state,
            &mut activations,
            &mut verify_activations,
            use_graph,
            host_sampling,
            &mut tokens,
            &mut logits,
            &mut row,
            &mut context,
            &mut committed,
            &mut mirostat,
            &mut output_constraint,
            &mut adaptive_controller,
            &mut correctable_controller,
            &mut speculation,
            &mut verify_duration,
            &mut decode_duration,
            &mut decode_evaluations,
            rng,
            &mut on_token,
            &mut cancelled,
            eos,
            profile_target,
            &mut profile_start,
            &mut profile_steps,
            &mut decode_profile,
        )?;
        Self::finish_generation_profile(
            &mut profile_start,
            profile_steps,
            &mut decode_profile,
            |steps, elapsed| {
                self.backend
                    .end_decode_profile(steps, elapsed)
                    .map_err(RuntimeError::from)
            },
        )?;
        Self::record_generation_controller_stats(
            &mut speculation,
            &adaptive_controller,
            &correctable_controller,
        );
        self.finish_generation_decode_state(
            session,
            prompt_tokens,
            &tokens,
            state,
            activations,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            was_cancelled,
            mirostat,
            adaptive_controller,
            correctable_controller,
        )?;
        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.to_vec(),
            stats: GenerationStats {
                prompt_tokens: prompt_tokens.len(),
                emitted_tokens: tokens.len(),
                decode_evaluations,
                prefill_duration,
                ttft_duration: Some(ttft_duration),
                decode_duration,
                verify_duration,
                cancelled: was_cancelled,
                speculation,
                prefill_method: self.backend.prefill_method(),
                prefill_workspace: prefill.workspace,
            },
            tokens,
            logits,
            decode_profile,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation_iterations<F, C>(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
        attention_shape: AttentionShape,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        use_graph: bool,
        host_sampling: bool,
        tokens: &mut Vec<u32>,
        logits: &mut Vec<LogitSnapshot>,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        mirostat: &mut Option<MirostatState>,
        output_constraint: &mut Option<crate::constraint::JsonObjectConstraint>,
        adaptive_controller: &mut Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: &mut Option<CorrectableController>,
        speculation: &mut SpeculationStats,
        verify_duration: &mut Duration,
        decode_duration: &mut Duration,
        decode_evaluations: &mut usize,
        rng: SamplerRng,
        on_token: &mut F,
        cancelled: &mut C,
        eos: Option<u32>,
        profile_target: usize,
        profile_start: &mut Option<Instant>,
        profile_steps: &mut usize,
        decode_profile: &mut Option<DecodeProfile>,
    ) -> Result<bool, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
    {
        let mut was_cancelled = false;
        while tokens.len() < options.max_tokens && Some(first_token(tokens)) != eos {
            if cancelled() {
                was_cancelled = true;
                break;
            }
            self.run_generation_round(
                prompt_tokens,
                options,
                attention_shape,
                state,
                activations,
                verify_activations,
                use_graph,
                host_sampling,
                tokens,
                logits,
                row,
                context,
                committed,
                mirostat,
                output_constraint,
                adaptive_controller,
                correctable_controller,
                speculation,
                verify_duration,
                decode_duration,
                decode_evaluations,
                rng,
                on_token,
            )?;
            Self::update_generation_profile(
                profile_start,
                profile_steps,
                profile_target,
                decode_profile,
                |steps, elapsed| {
                    self.backend
                        .end_decode_profile(steps, elapsed)
                        .map_err(RuntimeError::from)
                },
            )?;
        }
        Ok(was_cancelled)
    }

    fn update_generation_profile<P, F>(
        profile_start: &mut Option<Instant>,
        profile_steps: &mut usize,
        profile_target: usize,
        decode_profile: &mut Option<P>,
        mut end_profile: F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(usize, Duration) -> Result<Option<P>, RuntimeError>,
    {
        if let Some(start) = *profile_start {
            *profile_steps += 1;
            if *profile_steps == profile_target {
                *decode_profile = end_profile(*profile_steps, start.elapsed())?;
                *profile_start = None;
            }
        }
        Ok(())
    }

    fn finish_generation_profile<P, F>(
        profile_start: &mut Option<Instant>,
        profile_steps: usize,
        decode_profile: &mut Option<P>,
        mut end_profile: F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(usize, Duration) -> Result<Option<P>, RuntimeError>,
    {
        if let Some(start) = *profile_start {
            *decode_profile = end_profile(profile_steps, start.elapsed())?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_generation_decode_state(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        tokens: &[u32],
        state: KvState<B>,
        activations: Activations<B>,
        reuse_class: SessionReuseClass,
        reused_tokens: usize,
        cached_tokens: usize,
        restored_common: usize,
        common_tokens: usize,
        replay_prefill_boundary: usize,
        was_cancelled: bool,
        mirostat: Option<MirostatState>,
        adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: Option<CorrectableController>,
    ) -> Result<(), RuntimeError> {
        self.backend.synchronize()?;
        let replayed_tokens = if reuse_class == SessionReuseClass::RestoreReplay {
            restored_common.max(common_tokens)
        } else {
            0
        };
        session.last_replay = SessionReplay {
            reuse_class,
            cached_tokens: cached_tokens.max(session.restored_tokens.len()),
            reused_tokens,
            replayed_tokens,
            computed_tokens: prompt_tokens
                .len()
                .saturating_sub(reused_tokens)
                .saturating_sub(replayed_tokens),
        };
        Self::commit_generation_session(
            session,
            prompt_tokens,
            tokens,
            state,
            activations,
            reuse_class,
            replay_prefill_boundary,
            was_cancelled,
            mirostat,
            adaptive_controller,
            correctable_controller,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation_round<F>(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
        attention_shape: AttentionShape,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        use_graph: bool,
        host_sampling: bool,
        tokens: &mut Vec<u32>,
        logits: &mut Vec<LogitSnapshot>,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        mirostat: &mut Option<MirostatState>,
        output_constraint: &mut Option<crate::constraint::JsonObjectConstraint>,
        adaptive_controller: &mut Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: &mut Option<CorrectableController>,
        speculation: &mut SpeculationStats,
        verify_duration: &mut Duration,
        decode_duration: &mut Duration,
        decode_evaluations: &mut usize,
        rng: SamplerRng,
        on_token: &mut F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
    {
        let decode_start = Instant::now();
        committed.clear();
        let GenerationDraft {
            drafted,
            adaptive_round,
            correctable_round,
            correctable_distributions,
        } = Self::choose_generation_draft(
            options,
            prompt_tokens,
            tokens,
            context,
            state.position,
            rng,
            adaptive_controller,
            correctable_controller,
        )?;
        let step = self.run_generation_step(
            state,
            activations,
            verify_activations,
            attention_shape,
            &drafted,
            correctable_distributions.as_deref(),
            options,
            rng,
            use_graph,
            host_sampling,
            prompt_tokens,
            tokens,
            logits,
            row,
            context,
            committed,
            mirostat,
            output_constraint,
        )?;
        *decode_evaluations = decode_evaluations
            .checked_add(step.evaluations)
            .ok_or(RuntimeError::SizeOverflow)?;
        Self::record_generation_outcome(step.outcome.as_ref(), speculation, verify_duration);
        let round_duration = decode_start.elapsed();
        *decode_duration += round_duration;
        let correctable_proposed = step.outcome.as_ref().map_or(0, |outcome| outcome.proposed);
        let correctable_accepted = step.outcome.as_ref().map_or(0, |outcome| outcome.accepted);
        let correctable_overlap = step
            .outcome
            .as_ref()
            .map_or(0.0, |outcome| outcome.overlap_sum);
        Self::observe_generation_round(
            adaptive_round,
            correctable_round,
            committed.len(),
            &correctable_distributions,
            adaptive_controller,
            correctable_controller,
            round_duration,
            correctable_proposed,
            correctable_accepted,
            correctable_overlap,
        )?;
        self.emit_generation_round_tokens(
            committed,
            tokens,
            options.max_tokens,
            self.model.tokenizer.eos_token(),
            on_token,
        )?;
        Ok(())
    }

    fn prepare_generation_session(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
    ) -> Result<(AttentionShape, GenerationPreparation<B>), RuntimeError> {
        let attention_shape = self.validate_generation_request(prompt_tokens, options)?;
        let preparation =
            self.prepare_generation_state(session, prompt_tokens, options, attention_shape)?;
        Ok((attention_shape, preparation))
    }

    fn record_generation_outcome(
        outcome: Option<&SpeculationOutcome>,
        speculation: &mut SpeculationStats,
        verify_duration: &mut Duration,
    ) {
        if let Some(outcome) = outcome {
            speculation.proposed += outcome.proposed;
            speculation.accepted += outcome.accepted;
            speculation.rounds += 1;
            speculation.draft_width = speculation.draft_width.max(outcome.proposed);
            speculation.correctable_overlap_sum += outcome.overlap_sum;
            speculation.correctable_overlap_proposals += outcome.overlap_proposals;
            if outcome.verified_positions > 0 {
                speculation.verify_passes += 1;
                speculation.verified_positions += outcome.verified_positions;
                *verify_duration += outcome.verify_duration;
            }
        }
    }

    fn emit_generation_round_tokens<F>(
        &mut self,
        committed: &mut Vec<u32>,
        tokens: &mut Vec<u32>,
        max_tokens: usize,
        eos: Option<u32>,
        on_token: &mut F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
    {
        self.emit_generation_tokens(committed, tokens, max_tokens, eos, on_token)
    }

    fn uses_host_sampling(options: &GenerateOptions, has_constraint: bool) -> bool {
        options.sampler != Sampler::greedy()
            || options.penalties.is_active()
            || options.mirostat.is_some()
            || has_constraint
    }

    fn generation_profile_target(
        options: &GenerateOptions,
        emitted_tokens: usize,
    ) -> Result<usize, RuntimeError> {
        match options.decode_profile {
            DecodeProfileMode::Disabled => Ok(0),
            DecodeProfileMode::Steps(steps) => Ok(steps
                .get()
                .min(options.max_tokens.saturating_sub(emitted_tokens))),
        }
    }

    fn begin_generation_profile(
        &mut self,
        target: usize,
        first_token: u32,
        eos: Option<u32>,
    ) -> Result<Option<Instant>, RuntimeError> {
        if target == 0 || Some(first_token) == eos {
            return Ok(None);
        }
        let operations_per_step = self
            .model
            .config
            .n_layer
            .checked_mul(18)
            .and_then(|operations| operations.checked_add(4))
            .ok_or(RuntimeError::SizeOverflow)?;
        let operations = operations_per_step
            .checked_mul(target)
            .ok_or(RuntimeError::SizeOverflow)?;
        self.backend.begin_decode_profile(operations)?;
        Ok(Some(Instant::now()))
    }

    fn generation_uses_graph(
        options: &GenerateOptions,
        profile_target: usize,
        graph_supported: bool,
        eos: Option<u32>,
        first_token: u32,
    ) -> bool {
        options.decode_execution == DecodeExecution::Graph
            && profile_target == 0
            && graph_supported
            && Some(first_token) != eos
    }

    fn restore_mirostat(
        session: &mut GenerationSession<B>,
        options: &GenerateOptions,
        reuse_class: SessionReuseClass,
    ) -> Option<MirostatState> {
        match (options.mirostat, session.mirostat.take()) {
            (Some(config), Some(state))
                if state.config() == config
                    && matches!(
                        reuse_class,
                        SessionReuseClass::ExactRepeat
                            | SessionReuseClass::AppendOnly
                            | SessionReuseClass::RestoreReplay
                            | SessionReuseClass::DeviceFork
                            | SessionReuseClass::HostWake
                    ) =>
            {
                Some(state)
            }
            (Some(config), _) => Some(MirostatState::new(config)),
            (None, _) => None,
        }
    }

    fn restore_generation_controllers(
        session: &mut GenerationSession<B>,
        options: &GenerateOptions,
    ) -> (
        Option<crate::adaptive_draft::AdaptiveController>,
        Option<CorrectableController>,
    ) {
        let adaptive = match options.speculation {
            Speculation::Adaptive(_) => {
                Some(session.adaptive_controller.take().unwrap_or_else(|| {
                    crate::adaptive_draft::AdaptiveController::new(
                        crate::adaptive_draft::AdaptiveControllerConfig::default(),
                    )
                }))
            }
            _ => None,
        };
        let correctable = match options.speculation {
            Speculation::Correctable(_) => {
                Some(session.correctable_controller.take().unwrap_or_else(|| {
                    CorrectableController::new(CorrectableControllerConfig::default())
                }))
            }
            _ => None,
        };
        (adaptive, correctable)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_first_generation(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        options: &GenerateOptions,
        rng: SamplerRng,
        host_sampling: bool,
        row: &mut Vec<f32>,
        logits: &mut Vec<LogitSnapshot>,
        context: &[u32],
        mirostat: &mut Option<MirostatState>,
        output_constraint: &mut Option<crate::constraint::JsonObjectConstraint>,
    ) -> Result<u32, RuntimeError> {
        if host_sampling {
            // Allocate sampling scratch before graph capture begins.
            self.enqueue_sample(activations)?;
            let position = (state.position - 1) as u64;
            let (token, _) = self.sample_on_host(
                activations,
                &options.sampler,
                rng,
                position,
                row,
                &options.penalties,
                context,
                mirostat.as_mut(),
                output_constraint.as_mut(),
            )?;
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            Ok(token)
        } else {
            self.sample(
                activations,
                options.logit_capture,
                logits,
                state.position - 1,
            )
        }
    }

    fn prepare_prompt_prefill(
        &mut self,
        prefill_tokens: usize,
        context_tokens: usize,
        chunk_tokens: usize,
    ) -> Result<PreparedPrefill<B>, RuntimeError> {
        if prefill_tokens == 0 {
            return Ok(PreparedPrefill::Reused);
        }
        if self.backend.prefill_method() == PrefillMethod::SequentialDecode {
            return Ok(PreparedPrefill::Sequential);
        }
        let (chunk_tokens, plan) =
            self.prompt_prefill_plan(prefill_tokens, context_tokens, chunk_tokens)?;
        let AllocatedPromptPrefill {
            mut workspace,
            full,
            tail,
            remainder,
        } = self.allocate_prompt_prefill(plan, chunk_tokens, prefill_tokens)?;
        let activation_bytes = self.prompt_prefill_bytes(chunk_tokens, remainder)?;
        workspace.batch_activation_bytes = activation_bytes;
        workspace.total_bytes = workspace
            .total_bytes
            .checked_add(activation_bytes)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok(PreparedPrefill::Chunked {
            chunk_tokens,
            full,
            tail,
            workspace,
        })
    }

    fn prompt_prefill_plan(
        &self,
        prefill_tokens: usize,
        context_tokens: usize,
        chunk_tokens: usize,
    ) -> Result<(usize, PrefillPlan), RuntimeError> {
        if chunk_tokens == 0 {
            return Err(BackendError::Zero {
                field: "prefill chunk tokens",
            }
            .into());
        }
        let chunk_tokens = chunk_tokens.min(prefill_tokens);
        let plan = PrefillPlan::new(
            chunk_tokens,
            context_tokens,
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            self.model.config.n_embd,
            self.model.config.n_ff,
            self.model.config.n_ff,
        )?;
        Ok((chunk_tokens, plan))
    }

    fn allocate_prompt_prefill(
        &mut self,
        plan: PrefillPlan,
        chunk_tokens: usize,
        prefill_tokens: usize,
    ) -> Result<AllocatedPromptPrefill<B>, RuntimeError> {
        let workspace = self.backend.prepare_prefill(plan)?;
        let full = PrefillActivations::new(&mut self.backend, &self.model.config, chunk_tokens)?;
        let remainder = prefill_tokens % chunk_tokens;
        let tail = if remainder == 0 {
            None
        } else {
            Some(PrefillActivations::new(
                &mut self.backend,
                &self.model.config,
                remainder,
            )?)
        };
        Ok(AllocatedPromptPrefill {
            workspace,
            full,
            tail,
            remainder,
        })
    }

    fn prompt_prefill_bytes(
        &self,
        chunk_tokens: usize,
        remainder: usize,
    ) -> Result<u64, RuntimeError> {
        PrefillActivations::<B>::bytes(&self.model.config, chunk_tokens)?
            .checked_add(if remainder == 0 {
                0
            } else {
                PrefillActivations::<B>::bytes(&self.model.config, remainder)?
            })
            .ok_or(RuntimeError::SizeOverflow)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prompt_prefill<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
        mut cancelled: C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        match prepared {
            PreparedPrefill::Reused => Ok(PrefillExecution {
                processed_tokens: 0,
                cancelled: false,
                workspace: PrefillWorkspace::default(),
            }),
            PreparedPrefill::Sequential => self.run_sequential_prefill(
                prompt_tokens,
                state,
                activations,
                attention_shape,
                &mut cancelled,
            ),
            PreparedPrefill::Chunked {
                chunk_tokens,
                full,
                tail,
                workspace,
            } => self.run_chunked_prefill(
                prompt_tokens,
                state,
                activations,
                attention_shape,
                chunk_tokens,
                full,
                tail,
                workspace,
                &mut cancelled,
            ),
        }
    }

    fn run_sequential_prefill<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        for (processed, token) in prompt_tokens.iter().copied().enumerate() {
            if cancelled() {
                return Ok(PrefillExecution {
                    processed_tokens: processed,
                    cancelled: true,
                    workspace: PrefillWorkspace::default(),
                });
            }
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            self.forward(state, activations, attention_shape)?;
        }
        Ok(PrefillExecution {
            processed_tokens: prompt_tokens.len(),
            cancelled: false,
            workspace: PrefillWorkspace::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chunked_prefill<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        chunk_tokens: &usize,
        full: &mut PrefillActivations<B>,
        tail: &mut Option<PrefillActivations<B>>,
        workspace: &mut PrefillWorkspace,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        let base_position = state.position;
        let mut processed = 0_usize;
        while processed < prompt_tokens.len() {
            if cancelled() {
                return Ok(PrefillExecution {
                    processed_tokens: processed,
                    cancelled: true,
                    workspace: *workspace,
                });
            }
            self.run_chunked_prefill_step(
                prompt_tokens,
                &mut processed,
                base_position,
                state,
                activations,
                attention_shape,
                chunk_tokens,
                full,
                tail,
            )?;
        }
        self.finish_prefill_logits(activations)?;
        Ok(PrefillExecution {
            processed_tokens: processed,
            cancelled: false,
            workspace: *workspace,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chunked_prefill_step(
        &mut self,
        prompt_tokens: &[u32],
        processed: &mut usize,
        base_position: usize,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        chunk_tokens: &usize,
        full: &mut PrefillActivations<B>,
        tail: &mut Option<PrefillActivations<B>>,
    ) -> Result<(), RuntimeError> {
        let count = (*chunk_tokens).min(prompt_tokens.len() - *processed);
        let batch = if count == *chunk_tokens {
            &mut *full
        } else {
            tail.as_mut().ok_or_else(|| {
                BackendError::operation("select prefill tail", "tail activation storage is missing")
            })?
        };
        self.run_prefill_batch(PrefillBatchContext {
            prompt_tokens,
            processed: *processed,
            count,
            base_position,
            state,
            batch,
            attention_shape,
        })?;
        *processed = (*processed)
            .checked_add(count)
            .ok_or(RuntimeError::SizeOverflow)?;
        state.position = base_position
            .checked_add(*processed)
            .ok_or(RuntimeError::SizeOverflow)?;
        if *processed == prompt_tokens.len() {
            self.backend.copy_f32_row(
                &batch.hidden,
                count - 1,
                self.model.config.n_embd,
                &mut activations.hidden,
            )?;
        }
        Ok(())
    }

    fn run_prefill_batch(
        &mut self,
        batch_context: PrefillBatchContext<'_, B>,
    ) -> Result<(), RuntimeError> {
        let PrefillBatchContext {
            prompt_tokens,
            processed,
            count,
            base_position,
            state,
            batch,
            attention_shape,
        } = batch_context;
        self.forward_prefill_chunk(
            &prompt_tokens[processed..processed + count],
            base_position
                .checked_add(processed)
                .ok_or(RuntimeError::SizeOverflow)?,
            &mut state.layers,
            batch,
            attention_shape,
        )
    }

    fn finish_prefill_logits(
        &mut self,
        activations: &mut Activations<B>,
    ) -> Result<(), RuntimeError> {
        let config = &self.model.config;
        self.backend.rms_norm(
            &activations.hidden,
            &self.model.weights.output_norm,
            &mut activations.norm,
            VectorShape::new(1, config.n_embd)?,
            config.rms_epsilon,
        )?;
        let output = match &self.model.weights.output {
            OutputWeight::Separate(weight) => weight,
            OutputWeight::Tied => &self.model.weights.token_embedding,
        };
        self.backend.gemv(
            &output.buffer,
            &activations.norm,
            &mut activations.logits,
            output.shape,
        )?;
        Ok(())
    }

    fn finish_prefill_logits_batch(
        &mut self,
        activations: &mut PrefillEvalActivations<B>,
        tokens: usize,
    ) -> Result<(), RuntimeError> {
        let config = &self.model.config;
        self.backend.prefill_rms_norm(
            &activations.forward.hidden,
            &self.model.weights.output_norm,
            &mut activations.forward.norm,
            VectorShape::new(tokens, config.n_embd)?,
            config.rms_epsilon,
        )?;
        let output = match &self.model.weights.output {
            OutputWeight::Separate(weight) => weight,
            OutputWeight::Tied => &self.model.weights.token_embedding,
        };
        self.backend.prefill_gemm(
            &output.buffer,
            &activations.forward.norm,
            &mut activations.logits,
            output.shape,
            tokens,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_chunk(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        layers: &mut [KvLayer<B>],
        activations: &mut PrefillActivations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        self.prepare_prefill_chunk_inputs(tokens, activations)?;
        let (hidden_shape, query_shape, key_shape, query_rope, key_rope) =
            self.prefill_chunk_shapes(tokens.len())?;
        self.forward_prefill_layers(
            tokens,
            start_position,
            layers,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            query_rope,
            key_rope,
        )?;
        Ok(())
    }

    fn prepare_prefill_chunk_inputs(
        &mut self,
        tokens: &[u32],
        activations: &mut PrefillActivations<B>,
    ) -> Result<(), RuntimeError> {
        self.backend.write_u32(&mut activations.tokens, tokens)?;
        self.backend.embed_gather_batch(
            &self.model.weights.token_embedding.buffer,
            &activations.tokens,
            &mut activations.hidden,
            self.model.weights.token_embedding.shape,
            tokens.len(),
        )?;
        Ok(())
    }

    fn prefill_chunk_shapes(
        &self,
        tokens: usize,
    ) -> Result<(VectorShape, VectorShape, VectorShape, RopeShape, RopeShape), RuntimeError> {
        let config = &self.model.config;
        let query_rows = tokens
            .checked_mul(config.n_head)
            .ok_or(RuntimeError::SizeOverflow)?;
        let key_rows = tokens
            .checked_mul(config.n_head_kv)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok((
            VectorShape::new(tokens, config.n_embd)?,
            VectorShape::new(query_rows, config.head_dim)?,
            VectorShape::new(key_rows, config.head_dim)?,
            RopeShape::new(tokens, config.n_head, config.head_dim)?,
            RopeShape::new(tokens, config.n_head_kv, config.head_dim)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_layers(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        layers: &mut [KvLayer<B>],
        activations: &mut PrefillActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        query_rope: RopeShape,
        key_rope: RopeShape,
    ) -> Result<(), RuntimeError> {
        let epsilon = self.model.config.rms_epsilon;
        let theta = self.model.config.rope_theta;
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            Self::forward_prefill_layer(
                &mut self.backend,
                layer,
                &mut layers[layer_index],
                activations,
                attention_shape,
                hidden_shape,
                query_shape,
                key_shape,
                query_rope,
                key_rope,
                start_position,
                tokens.len(),
                epsilon,
                theta,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_layer(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvLayer<B>,
        activations: &mut PrefillActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        query_rope: RopeShape,
        key_rope: RopeShape,
        start_position: usize,
        tokens: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        Self::prefill_layer_attention(
            backend,
            layer,
            state,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            query_rope,
            key_rope,
            start_position,
            tokens,
            epsilon,
            theta,
        )?;
        Self::prefill_layer_ffn(backend, layer, activations, hidden_shape, tokens, epsilon)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_attention(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvLayer<B>,
        activations: &mut PrefillActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        query_rope: RopeShape,
        key_rope: RopeShape,
        start_position: usize,
        tokens: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        backend.prefill_rms_norm(
            &activations.hidden,
            &layer.attention_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
        )?;
        backend.prefill_gemm(
            &layer.query.buffer,
            &activations.norm,
            &mut activations.query,
            layer.query.shape,
            tokens,
        )?;
        backend.prefill_gemm(
            &layer.key.buffer,
            &activations.norm,
            &mut activations.key,
            layer.key.shape,
            tokens,
        )?;
        backend.prefill_gemm(
            &layer.value.buffer,
            &activations.norm,
            &mut activations.value,
            layer.value.shape,
            tokens,
        )?;
        let normalized = Self::prefill_layer_qk(
            backend,
            &layer.qk_norm,
            activations,
            query_shape,
            key_shape,
            query_rope,
            key_rope,
            start_position,
            epsilon,
            theta,
        )?;
        let (query, key) = Self::prefill_layer_query_key(
            normalized,
            &activations.query,
            &activations.key,
            &activations.query_norm,
            &activations.key_norm,
        );
        backend.kv_append_chunk(
            key,
            &activations.value,
            &mut state.key,
            &mut state.value,
            attention_shape,
            start_position,
            tokens,
        )?;
        backend.attention_prefill(
            query,
            &state.key,
            &state.value,
            &mut activations.attention,
            attention_shape,
            start_position,
            tokens,
        )?;
        backend.prefill_gemm(
            &layer.attention_output.buffer,
            &activations.attention,
            &mut activations.residual,
            layer.attention_output.shape,
            tokens,
        )?;
        backend.residual_add(
            &activations.hidden,
            &activations.residual,
            &mut activations.attention,
        )?;
        Ok(())
    }

    fn prefill_layer_query_key<'a>(
        normalized: bool,
        query: &'a B::Buffer,
        key: &'a B::Buffer,
        query_norm: &'a B::Buffer,
        key_norm: &'a B::Buffer,
    ) -> (&'a B::Buffer, &'a B::Buffer) {
        if normalized {
            (query_norm, key_norm)
        } else {
            (query, key)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_qk(
        backend: &mut B,
        qk_norm: &QkNorm<B>,
        activations: &mut PrefillActivations<B>,
        query_shape: VectorShape,
        key_shape: VectorShape,
        query_rope: RopeShape,
        key_rope: RopeShape,
        start_position: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<bool, RuntimeError> {
        match qk_norm {
            QkNorm::Rms { query, key } => {
                backend.prefill_rms_norm(
                    &activations.query,
                    query,
                    &mut activations.query_norm,
                    query_shape,
                    epsilon,
                )?;
                backend.rope(
                    &mut activations.query_norm,
                    start_position,
                    query_rope,
                    theta,
                )?;
                backend.prefill_rms_norm(
                    &activations.key,
                    key,
                    &mut activations.key_norm,
                    key_shape,
                    epsilon,
                )?;
                backend.rope(&mut activations.key_norm, start_position, key_rope, theta)?;
                Ok(true)
            }
            QkNorm::Identity => {
                backend.rope(&mut activations.query, start_position, query_rope, theta)?;
                backend.rope(&mut activations.key, start_position, key_rope, theta)?;
                Ok(false)
            }
        }
    }

    fn prefill_layer_ffn(
        backend: &mut B,
        layer: &DenseLayer<B>,
        activations: &mut PrefillActivations<B>,
        hidden_shape: VectorShape,
        tokens: usize,
        epsilon: f32,
    ) -> Result<(), RuntimeError> {
        backend.prefill_rms_norm(
            &activations.attention,
            &layer.ffn_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
        )?;
        backend.prefill_gemm(
            &layer.ffn_gate.buffer,
            &activations.norm,
            &mut activations.gate,
            layer.ffn_gate.shape,
            tokens,
        )?;
        backend.prefill_gemm(
            &layer.ffn_up.buffer,
            &activations.norm,
            &mut activations.up,
            layer.ffn_up.shape,
            tokens,
        )?;
        backend.swiglu(&activations.gate, &activations.up, &mut activations.ffn)?;
        backend.prefill_gemm(
            &layer.ffn_down.buffer,
            &activations.ffn,
            &mut activations.residual,
            layer.ffn_down.shape,
            tokens,
        )?;
        backend.residual_add(
            &activations.attention,
            &activations.residual,
            &mut activations.hidden,
        )?;
        Ok(())
    }

    fn capture_decode_graph(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        let device_position =
            u32::try_from(state.position).map_err(|_| RuntimeError::ContextCapacity {
                requested: state.position,
                capacity: u32::MAX as usize,
            })?;
        self.backend
            .write_u32(&mut state.device_position, &[device_position])?;
        self.backend.begin_decode_graph()?;
        self.forward_at(
            &mut state.layers,
            activations,
            attention_shape,
            Position::Device(&state.device_position),
        )?;
        self.enqueue_sample(activations)?;
        self.backend.increment_u32(&mut state.device_position)?;
        self.backend.end_decode_graph()?;
        Ok(())
    }

    fn forward(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        let position = state.position;
        self.forward_at(
            &mut state.layers,
            activations,
            attention_shape,
            Position::Host(position),
        )?;
        state.position = state
            .position
            .checked_add(1)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok(())
    }

    fn forward_at(
        &mut self,
        layers: &mut [KvLayer<B>],
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        position: Position<'_, B::Buffer>,
    ) -> Result<(), RuntimeError> {
        let epsilon = self.model.config.rms_epsilon;
        let theta = self.model.config.rope_theta;
        let (hidden_shape, query_shape, key_shape) =
            self.forward_at_input(activations, position)?;
        self.forward_at_layers(
            layers,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            position,
            epsilon,
            theta,
        )?;
        self.forward_at_output(activations, hidden_shape)
    }

    fn forward_at_input(
        &mut self,
        activations: &mut Activations<B>,
        position: Position<'_, B::Buffer>,
    ) -> Result<(VectorShape, VectorShape, VectorShape), RuntimeError> {
        self.backend.profile_decode_op(DecodeOp::Embed)?;
        self.backend.embed_gather(
            &self.model.weights.token_embedding.buffer,
            &activations.sampled,
            &mut activations.hidden,
            self.model.weights.token_embedding.shape,
        )?;
        let hidden_shape = VectorShape::new(1, self.model.config.n_embd)?;
        let query_shape = VectorShape::new(self.model.config.n_head, self.model.config.head_dim)?;
        let key_shape = VectorShape::new(self.model.config.n_head_kv, self.model.config.head_dim)?;
        self.backend.prepare_rope(position)?;
        Ok((hidden_shape, query_shape, key_shape))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_at_layers(
        &mut self,
        layers: &mut [KvLayer<B>],
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            Self::forward_layer(
                &mut self.backend,
                layer,
                &mut layers[layer_index],
                activations,
                attention_shape,
                hidden_shape,
                query_shape,
                key_shape,
                position,
                epsilon,
                theta,
            )?;
        }
        Ok(())
    }

    fn forward_at_output(
        &mut self,
        activations: &mut Activations<B>,
        hidden_shape: VectorShape,
    ) -> Result<(), RuntimeError> {
        let epsilon = self.model.config.rms_epsilon;
        self.backend.profile_decode_op(DecodeOp::Norm)?;
        self.backend.rms_norm(
            &activations.hidden,
            &self.model.weights.output_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
        )?;
        let output = match &self.model.weights.output {
            OutputWeight::Separate(weight) => weight,
            OutputWeight::Tied => &self.model.weights.token_embedding,
        };
        self.backend.profile_decode_op(DecodeOp::LmHeadGemv)?;
        self.backend.gemv(
            &output.buffer,
            &activations.norm,
            &mut activations.logits,
            output.shape,
        )?;
        Ok(())
    }

    fn forward_verify(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        tokens: &[u32],
    ) -> Result<(), RuntimeError> {
        let positions = self.verify_position_count(activations, tokens)?;
        let start_position = state.position;
        let (hidden_shape, query_shape, key_shape) = self.verify_shapes(positions)?;
        self.prepare_verify_inputs(activations, tokens, positions)?;
        self.forward_verify_layers(
            state,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            start_position,
            positions,
        )?;
        self.finish_forward_verify(activations, hidden_shape, positions)?;
        state.position = state
            .position
            .checked_add(positions)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok(())
    }

    fn verify_position_count(
        &self,
        activations: &VerifyActivations<B>,
        tokens: &[u32],
    ) -> Result<usize, RuntimeError> {
        let positions = tokens.len();
        if positions != activations.positions {
            return Err(BackendError::SizeMismatch {
                name: "verifier positions",
                expected: activations.positions,
                actual: positions,
            }
            .into());
        }
        Ok(positions)
    }

    fn verify_shapes(
        &self,
        positions: usize,
    ) -> Result<(VectorShape, VectorShape, VectorShape), RuntimeError> {
        Ok((
            VectorShape::new(positions, self.model.config.n_embd)?,
            VectorShape::new(self.model.config.n_head, self.model.config.head_dim)?,
            VectorShape::new(self.model.config.n_head_kv, self.model.config.head_dim)?,
        ))
    }

    fn prepare_verify_inputs(
        &mut self,
        activations: &mut VerifyActivations<B>,
        tokens: &[u32],
        positions: usize,
    ) -> Result<(), RuntimeError> {
        self.backend.write_u32(&mut activations.tokens, tokens)?;
        self.backend.embed_gather_batch(
            &self.model.weights.token_embedding.buffer,
            &activations.tokens,
            &mut activations.hidden,
            self.model.weights.token_embedding.shape,
            positions,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_layers(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
    ) -> Result<(), RuntimeError> {
        let epsilon = self.model.config.rms_epsilon;
        let theta = self.model.config.rope_theta;
        let n_ff = self.model.config.n_ff;
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            Self::forward_verify_layer(
                &mut self.backend,
                layer,
                &mut state.layers[layer_index],
                activations,
                attention_shape,
                hidden_shape,
                query_shape,
                key_shape,
                start_position,
                positions,
                n_ff,
                epsilon,
                theta,
            )?;
        }
        Ok(())
    }

    fn finish_forward_verify(
        &mut self,
        activations: &mut VerifyActivations<B>,
        hidden_shape: VectorShape,
        positions: usize,
    ) -> Result<(), RuntimeError> {
        self.backend.prefill_rms_norm(
            &activations.hidden,
            &self.model.weights.output_norm,
            &mut activations.norm,
            hidden_shape,
            self.model.config.rms_epsilon,
        )?;
        let output = match &self.model.weights.output {
            OutputWeight::Separate(weight) => weight,
            OutputWeight::Tied => &self.model.weights.token_embedding,
        };
        self.backend.verify_gemv(
            &output.buffer,
            &activations.norm,
            &mut activations.logits,
            output.shape,
            positions,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_layer(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvLayer<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        n_ff: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        Self::forward_verify_attention(
            backend,
            layer,
            state,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            start_position,
            positions,
            epsilon,
            theta,
        )?;
        Self::forward_verify_ffn(
            backend,
            layer,
            activations,
            hidden_shape,
            positions,
            n_ff,
            epsilon,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_attention(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvLayer<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        backend.prefill_rms_norm(
            &activations.hidden,
            &layer.attention_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
        )?;
        backend.verify_gemv_triple(
            &layer.query.buffer,
            &layer.key.buffer,
            &layer.value.buffer,
            &activations.norm,
            &mut activations.query,
            &mut activations.key,
            &mut activations.value,
            layer.query.shape,
            layer.key.shape,
            layer.value.shape,
            positions,
        )?;
        let normalized = Self::forward_verify_qk(
            backend,
            &layer.qk_norm,
            state,
            activations,
            attention_shape,
            query_shape,
            key_shape,
            start_position,
            positions,
            epsilon,
            theta,
        )?;
        let query = if normalized {
            &activations.query_norm
        } else {
            &activations.query
        };
        let attention_prepares_output = backend.verifier_attention_prepares_output(attention_shape);
        backend.verify_attention(
            query,
            &state.key,
            &state.value,
            &mut activations.attention,
            attention_shape,
            start_position,
            positions,
        )?;
        if attention_prepares_output {
            backend.verify_gemv_residual_prepared(
                &layer.attention_output.buffer,
                &activations.attention,
                &activations.hidden,
                &mut activations.residual,
                layer.attention_output.shape,
                positions,
            )?;
        } else {
            backend.verify_gemv_residual(
                &layer.attention_output.buffer,
                &activations.attention,
                &activations.hidden,
                &mut activations.residual,
                layer.attention_output.shape,
                positions,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_qk(
        backend: &mut B,
        qk_norm: &QkNorm<B>,
        state: &mut KvLayer<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<bool, RuntimeError> {
        match qk_norm {
            QkNorm::Rms { query, key } => {
                backend.verify_qk_norm_rope_kv_append(
                    &activations.query,
                    query,
                    &mut activations.query_norm,
                    query_shape,
                    &activations.key,
                    key,
                    &mut activations.key_norm,
                    key_shape,
                    &activations.value,
                    &mut state.key,
                    &mut state.value,
                    attention_shape,
                    start_position,
                    positions,
                    epsilon,
                    theta,
                )?;
                Ok(true)
            }
            QkNorm::Identity => {
                backend.rope(
                    &mut activations.query,
                    start_position,
                    RopeShape::new(positions, query_shape.rows(), query_shape.columns())?,
                    theta,
                )?;
                backend.rope(
                    &mut activations.key,
                    start_position,
                    RopeShape::new(positions, key_shape.rows(), key_shape.columns())?,
                    theta,
                )?;
                backend.kv_append_chunk(
                    &activations.key,
                    &activations.value,
                    &mut state.key,
                    &mut state.value,
                    attention_shape,
                    start_position,
                    positions,
                )?;
                Ok(false)
            }
        }
    }

    fn forward_verify_ffn(
        backend: &mut B,
        layer: &DenseLayer<B>,
        activations: &mut VerifyActivations<B>,
        hidden_shape: VectorShape,
        positions: usize,
        n_ff: usize,
        epsilon: f32,
    ) -> Result<(), RuntimeError> {
        backend.prefill_rms_norm(
            &activations.residual,
            &layer.ffn_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
        )?;
        backend.verify_gemv_pair(
            &layer.ffn_gate.buffer,
            &layer.ffn_up.buffer,
            &activations.norm,
            &mut activations.gate,
            &mut activations.up,
            layer.ffn_gate.shape,
            layer.ffn_up.shape,
            positions,
        )?;
        backend.verify_swiglu(
            &activations.gate,
            &activations.up,
            &mut activations.ffn,
            n_ff,
            positions,
        )?;
        backend.verify_gemv_residual_prepared(
            &layer.ffn_down.buffer,
            &activations.ffn,
            &activations.residual,
            &mut activations.hidden,
            layer.ffn_down.shape,
            positions,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvLayer<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        Self::forward_layer_attention(
            backend,
            layer,
            state,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            position,
            epsilon,
            theta,
        )?;
        Self::forward_layer_ffn(backend, layer, activations, hidden_shape, epsilon)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_attention(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvLayer<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        Self::forward_layer_inputs(backend, layer, activations, hidden_shape, epsilon)?;
        let normalized = Self::forward_layer_qk(
            backend,
            &layer.qk_norm,
            state,
            activations,
            attention_shape,
            query_shape,
            key_shape,
            position,
            epsilon,
            theta,
        )?;
        Self::forward_layer_attention_output(
            backend,
            layer,
            state,
            activations,
            attention_shape,
            position,
            normalized,
        )
    }

    fn forward_layer_inputs(
        backend: &mut B,
        layer: &DenseLayer<B>,
        activations: &mut Activations<B>,
        hidden_shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), RuntimeError> {
        backend.profile_decode_op(DecodeOp::Norm)?;
        backend.rms_norm(
            &activations.hidden,
            &layer.attention_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
        )?;
        backend.profile_decode_op(DecodeOp::QkvGemv)?;
        backend.qkv_gemv(
            &layer.query.buffer,
            layer.query.shape,
            &layer.key.buffer,
            layer.key.shape,
            &layer.value.buffer,
            layer.value.shape,
            &activations.norm,
            &mut activations.query,
            &mut activations.key,
            &mut activations.value,
        )?;
        Ok(())
    }

    fn forward_layer_attention_output(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvLayer<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        position: Position<'_, B::Buffer>,
        normalized: bool,
    ) -> Result<(), RuntimeError> {
        let query = if normalized {
            &activations.query_norm
        } else {
            &activations.query
        };
        backend.profile_decode_op(DecodeOp::KvAppend)?;
        backend.profile_decode_op(DecodeOp::Attention)?;
        backend.attention_decode(
            query,
            &state.key,
            &state.value,
            &mut activations.attention,
            attention_shape,
            position,
        )?;
        backend.profile_decode_op(DecodeOp::OutputGemv)?;
        backend.gemv_residual(
            &layer.attention_output.buffer,
            &activations.attention,
            &activations.hidden,
            &mut activations.residual,
            layer.attention_output.shape,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_qk(
        backend: &mut B,
        qk_norm: &QkNorm<B>,
        state: &mut KvLayer<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<bool, RuntimeError> {
        match qk_norm {
            QkNorm::Rms { query, key } => Self::forward_layer_rms_qk(
                backend,
                query,
                key,
                state,
                activations,
                attention_shape,
                query_shape,
                key_shape,
                position,
                epsilon,
                theta,
            )
            .map(|()| true),
            QkNorm::Identity => Self::forward_layer_identity_qk(
                backend,
                state,
                activations,
                attention_shape,
                query_shape,
                key_shape,
                position,
                theta,
            )
            .map(|()| false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_rms_qk(
        backend: &mut B,
        query: &B::Buffer,
        key: &B::Buffer,
        state: &mut KvLayer<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        backend.profile_decode_op(DecodeOp::QkNorm)?;
        backend.qk_norm_rope_kv_append(
            &activations.query,
            query,
            &mut activations.query_norm,
            query_shape,
            &activations.key,
            key,
            &mut activations.key_norm,
            key_shape,
            &activations.value,
            &mut state.key,
            &mut state.value,
            attention_shape,
            position,
            epsilon,
            theta,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_identity_qk(
        backend: &mut B,
        state: &mut KvLayer<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        backend.profile_decode_op(DecodeOp::Rope)?;
        backend.rope_position(
            &mut activations.query,
            position,
            RopeShape::new(1, query_shape.rows(), query_shape.columns())?,
            theta,
        )?;
        backend.rope_position(
            &mut activations.key,
            position,
            RopeShape::new(1, key_shape.rows(), key_shape.columns())?,
            theta,
        )?;
        backend.kv_append(
            &activations.key,
            &activations.value,
            &mut state.key,
            &mut state.value,
            attention_shape,
            position,
        )?;
        Ok(())
    }

    fn forward_layer_ffn(
        backend: &mut B,
        layer: &DenseLayer<B>,
        activations: &mut Activations<B>,
        hidden_shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), RuntimeError> {
        backend.profile_decode_op(DecodeOp::Norm)?;
        backend.rms_norm(
            &activations.residual,
            &layer.ffn_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
        )?;
        backend.profile_decode_op(DecodeOp::FfnGemv)?;
        backend.gemv_pair_swiglu(
            &layer.ffn_gate.buffer,
            layer.ffn_gate.shape,
            &layer.ffn_up.buffer,
            layer.ffn_up.shape,
            &activations.norm,
            &mut activations.gate,
            &mut activations.up,
            &mut activations.ffn,
        )?;
        backend.profile_decode_op(DecodeOp::SwiGlu)?;
        backend.profile_decode_op(DecodeOp::FfnGemv)?;
        backend.gemv_residual(
            &layer.ffn_down.buffer,
            &activations.ffn,
            &activations.residual,
            &mut activations.hidden,
            layer.ffn_down.shape,
        )?;
        Ok(())
    }

    fn enqueue_sample(&mut self, activations: &mut Activations<B>) -> Result<(), RuntimeError> {
        self.backend.profile_decode_op(DecodeOp::Argmax)?;
        self.backend
            .argmax(&activations.logits, &mut activations.sampled)?;
        Ok(())
    }

    fn read_sample(
        &mut self,
        activations: &mut Activations<B>,
        capture: LogitCapture,
        snapshots: &mut Vec<LogitSnapshot>,
        input_position: usize,
    ) -> Result<u32, RuntimeError> {
        let mut sampled = [0_u32];
        self.backend.read_u32(&activations.sampled, &mut sampled)?;
        if let LogitCapture::Top(count) = capture {
            let mut values = vec![0.0_f32; self.model.config.vocab_size];
            self.backend.read_f32(&activations.logits, &mut values)?;
            snapshots.push(LogitSnapshot {
                input_position,
                sampled_token: sampled[0],
                top: top_logits(&values, count.get())?,
            });
        }
        Ok(sampled[0])
    }

    /// Reads the full logit row to the host.
    ///
    /// Stochastic sampling and speculative verification both need the whole
    /// distribution. Greedy decode without speculation keeps the device
    /// argmax path, so the v0.1 decode receipt measures the same work.
    fn read_logit_row(
        &mut self,
        activations: &Activations<B>,
        row: &mut Vec<f32>,
        penalties: &Penalties,
        context: &[u32],
    ) -> Result<(), RuntimeError> {
        row.resize(self.model.config.vocab_size, 0.0);
        self.backend.read_f32(&activations.logits, row)?;
        // Penalties rewrite the row from the decode history, before any
        // truncation stage sees it.
        penalties.apply(row, context)?;
        Ok(())
    }

    /// Samples one token on the host from the current logit row.
    #[allow(clippy::too_many_arguments)]
    fn sample_on_host(
        &mut self,
        activations: &mut Activations<B>,
        sampler: &Sampler,
        rng: SamplerRng,
        position: u64,
        row: &mut Vec<f32>,
        penalties: &Penalties,
        context: &[u32],
        mirostat: Option<&mut MirostatState>,
        constraint: Option<&mut crate::constraint::JsonObjectConstraint>,
    ) -> Result<(u32, Distribution), RuntimeError> {
        self.read_logit_row(activations, row, penalties, context)?;
        self.sample_host_distribution(row, sampler, rng, position, mirostat, constraint)
    }

    fn sample_host_distribution(
        &self,
        row: &mut [f32],
        sampler: &Sampler,
        rng: SamplerRng,
        position: u64,
        mirostat: Option<&mut MirostatState>,
        mut constraint: Option<&mut crate::constraint::JsonObjectConstraint>,
    ) -> Result<(u32, Distribution), RuntimeError> {
        if let Some(constraint) = constraint.as_deref_mut() {
            constraint.mask(row, &self.model.tokenizer)?;
        }
        let target = distribution(row, sampler)?;
        let token = match mirostat {
            Some(controller) => controller.select(&target, rng, position)?,
            None => select(&target, rng, position)?,
        };
        if let Some(constraint) = constraint {
            constraint.accept(token, &self.model.tokenizer)?;
        }
        Ok((token, target))
    }

    /// Runs one drafted round and returns the tokens the model committed to.
    ///
    /// Every proposal is checked against the model's own distribution by the
    /// exact rule in [`crate::sampler::verify`]. The emitted distribution
    /// equals the unspeculated one. A rejection ends the round: later
    /// proposals were conditioned on a token that is not there.
    ///
    /// This evaluates one position per proposal. Exactness is not a speedup.
    /// A faster path needs one forward pass over the whole round.
    #[allow(clippy::too_many_arguments)]
    fn speculative_round(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: Option<&mut VerifyActivations<B>>,
        attention_shape: AttentionShape,
        drafted: &[u32],
        draft_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
    ) -> Result<SpeculationOutcome, RuntimeError> {
        Self::validate_draft_distributions(draft_distributions, drafted.len())?;
        if let Some(verify_activations) = verify_activations {
            if verify_activations.positions == drafted.len() + 1 {
                return self.speculative_verify_round(
                    state,
                    activations,
                    verify_activations,
                    attention_shape,
                    drafted,
                    draft_distributions,
                    options,
                    rng,
                    row,
                    context,
                    committed,
                );
            }
        }
        self.speculative_decode_round(
            state,
            activations,
            attention_shape,
            drafted,
            draft_distributions,
            options,
            rng,
            row,
            context,
            committed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn speculative_decode_round(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        drafted: &[u32],
        draft_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
    ) -> Result<SpeculationOutcome, RuntimeError> {
        let vocab = self.model.config.vocab_size;
        let mut accepted = 0;
        let mut overlap_sum = 0.0;
        let mut overlap_proposals = 0;
        for (index, proposal) in drafted.iter().copied().enumerate() {
            if let Some(outcome) = self.speculative_proposal_step(
                state,
                activations,
                attention_shape,
                drafted,
                draft_distributions,
                options,
                rng,
                row,
                context,
                committed,
                index,
                proposal,
                vocab,
                &mut accepted,
                &mut overlap_sum,
                &mut overlap_proposals,
            )? {
                return Ok(outcome);
            }
        }
        self.speculative_bonus_token(
            state,
            activations,
            attention_shape,
            options,
            rng,
            row,
            context,
            committed,
            drafted.len(),
            accepted,
            overlap_sum,
            overlap_proposals,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn speculative_proposal_step(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        drafted: &[u32],
        draft_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        index: usize,
        proposal: u32,
        vocab: usize,
        accepted: &mut usize,
        overlap_sum: &mut f64,
        overlap_proposals: &mut usize,
    ) -> Result<Option<SpeculationOutcome>, RuntimeError> {
        self.forward(state, activations, attention_shape)?;
        let position = (state.position - 1) as u64;
        self.read_logit_row(activations, row, &options.penalties, context)?;
        let (target, draft) = Self::speculative_proposal_distributions(
            row,
            options,
            draft_distributions,
            index,
            proposal,
            vocab,
            overlap_sum,
            overlap_proposals,
        )?;
        let verdict = verify(&target, &draft, proposal, rng, position)?;
        self.finish_speculative_proposal(
            verdict,
            drafted,
            activations,
            proposal,
            context,
            committed,
            accepted,
            index,
            *overlap_sum,
            *overlap_proposals,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn speculative_proposal_distributions(
        row: &[f32],
        options: &GenerateOptions,
        draft_distributions: Option<&[Distribution]>,
        index: usize,
        proposal: u32,
        vocab: usize,
        overlap_sum: &mut f64,
        overlap_proposals: &mut usize,
    ) -> Result<(Distribution, Distribution), RuntimeError> {
        // Every committed token joins the history before the next position is scored.
        let target = distribution(row, &options.sampler)?;
        let draft = match draft_distributions {
            Some(distributions) => distributions[index].clone(),
            None => point_mass(vocab, proposal)?,
        };
        if draft_distributions.is_some() {
            *overlap_sum += 1.0 - crate::total_variation(&target, &draft)?;
            *overlap_proposals += 1;
        }
        Ok((target, draft))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_speculative_proposal(
        &mut self,
        verdict: Verdict,
        drafted: &[u32],
        activations: &mut Activations<B>,
        proposal: u32,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        accepted: &mut usize,
        index: usize,
        overlap_sum: f64,
        overlap_proposals: usize,
    ) -> Result<Option<SpeculationOutcome>, RuntimeError> {
        match verdict {
            Verdict::Accept => {
                *accepted += 1;
                committed.push(proposal);
                context.push(proposal);
                self.backend
                    .write_u32(&mut activations.sampled, &[proposal])?;
                Ok(None)
            }
            Verdict::Reject { token } => {
                committed.push(token);
                context.push(token);
                self.backend.write_u32(&mut activations.sampled, &[token])?;
                Ok(Some(SpeculationOutcome {
                    proposed: drafted.len(),
                    accepted: *accepted,
                    evaluations: index + 1,
                    verified_positions: 0,
                    verify_duration: Duration::ZERO,
                    overlap_sum,
                    overlap_proposals,
                }))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn speculative_bonus_token(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        options: &GenerateOptions,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        proposed: usize,
        accepted: usize,
        overlap_sum: f64,
        overlap_proposals: usize,
    ) -> Result<SpeculationOutcome, RuntimeError> {
        // Every proposal held, so one more evaluation yields a bonus token.
        self.forward(state, activations, attention_shape)?;
        let position = (state.position - 1) as u64;
        let (token, _) = self.sample_on_host(
            activations,
            &options.sampler,
            rng,
            position,
            row,
            &options.penalties,
            context,
            None,
            None,
        )?;
        committed.push(token);
        context.push(token);
        self.backend.write_u32(&mut activations.sampled, &[token])?;
        Ok(SpeculationOutcome {
            proposed,
            accepted,
            evaluations: proposed + 1,
            verified_positions: 0,
            verify_duration: Duration::ZERO,
            overlap_sum,
            overlap_proposals,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn speculative_verify_round(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        drafted: &[u32],
        draft_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
    ) -> Result<SpeculationOutcome, RuntimeError> {
        let (base_position, verify_duration) = self.prepare_verify_round(
            state,
            verify_activations,
            attention_shape,
            drafted,
            context,
        )?;
        let vocab = self.model.config.vocab_size;
        let mut accepted = 0;
        let mut overlap_sum = 0.0;
        let mut overlap_proposals = 0;
        if let Some(outcome) = self.verify_speculative_proposals(
            state,
            activations,
            verify_activations,
            drafted,
            draft_distributions,
            options,
            rng,
            row,
            context,
            committed,
            base_position,
            vocab,
            &mut accepted,
            &mut overlap_sum,
            &mut overlap_proposals,
            verify_duration,
        )? {
            return Ok(outcome);
        }
        self.verify_bonus_token(
            activations,
            verify_activations,
            drafted.len(),
            vocab,
            options,
            rng,
            row,
            context,
            committed,
            base_position,
        )?;
        Ok(SpeculationOutcome {
            proposed: drafted.len(),
            accepted,
            evaluations: 1,
            verified_positions: verify_activations.positions,
            verify_duration,
            overlap_sum,
            overlap_proposals,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_verify_round(
        &mut self,
        state: &mut KvState<B>,
        verify_activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        drafted: &[u32],
        context: &[u32],
    ) -> Result<(usize, Duration), RuntimeError> {
        let base_position = state.position;
        let current = context.last().copied().ok_or(RuntimeError::EmptyPrompt)?;
        verify_activations.input_tokens.clear();
        verify_activations.input_tokens.push(current);
        verify_activations.input_tokens.extend_from_slice(drafted);
        let input_tokens = verify_activations.input_tokens.clone();
        let started = Instant::now();
        self.forward_verify(state, verify_activations, attention_shape, &input_tokens)?;
        self.backend.read_f32(
            &verify_activations.logits,
            &mut verify_activations.host_logits,
        )?;
        Ok((base_position, started.elapsed()))
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_speculative_proposals(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: &VerifyActivations<B>,
        drafted: &[u32],
        draft_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        base_position: usize,
        vocab: usize,
        accepted: &mut usize,
        overlap_sum: &mut f64,
        overlap_proposals: &mut usize,
        verify_duration: Duration,
    ) -> Result<Option<SpeculationOutcome>, RuntimeError> {
        for (index, proposal) in drafted.iter().copied().enumerate() {
            let (target, draft) = Self::verify_proposal_distribution(
                verify_activations,
                index,
                vocab,
                proposal,
                draft_distributions,
                options,
                row,
                context,
            )?;
            if draft_distributions.is_some() {
                *overlap_sum += 1.0 - crate::total_variation(&target, &draft)?;
                *overlap_proposals += 1;
            }
            match verify(
                &target,
                &draft,
                proposal,
                rng,
                (base_position + index) as u64,
            )? {
                Verdict::Accept => {
                    *accepted += 1;
                    committed.push(proposal);
                    context.push(proposal);
                }
                Verdict::Reject { token } => {
                    committed.push(token);
                    context.push(token);
                    state.position = base_position
                        .checked_add(index + 1)
                        .ok_or(RuntimeError::SizeOverflow)?;
                    self.backend.write_u32(&mut activations.sampled, &[token])?;
                    return Ok(Some(SpeculationOutcome {
                        proposed: drafted.len(),
                        accepted: *accepted,
                        evaluations: 1,
                        verified_positions: verify_activations.positions,
                        verify_duration,
                        overlap_sum: *overlap_sum,
                        overlap_proposals: *overlap_proposals,
                    }));
                }
            }
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_proposal_distribution(
        verify_activations: &VerifyActivations<B>,
        index: usize,
        vocab: usize,
        proposal: u32,
        draft_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        row: &mut Vec<f32>,
        context: &[u32],
    ) -> Result<(Distribution, Distribution), RuntimeError> {
        let start = index.checked_mul(vocab).ok_or(RuntimeError::SizeOverflow)?;
        let end = start.checked_add(vocab).ok_or(RuntimeError::SizeOverflow)?;
        row.clear();
        row.extend_from_slice(
            verify_activations
                .host_logits
                .get(start..end)
                .ok_or(RuntimeError::SizeOverflow)?,
        );
        options.penalties.apply(row, context)?;
        let target = distribution(row, &options.sampler)?;
        let draft = match draft_distributions {
            Some(distributions) => distributions[index].clone(),
            None => point_mass(vocab, proposal)?,
        };
        Ok((target, draft))
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_bonus_token(
        &mut self,
        activations: &mut Activations<B>,
        verify_activations: &VerifyActivations<B>,
        drafted: usize,
        vocab: usize,
        options: &GenerateOptions,
        rng: SamplerRng,
        row: &mut Vec<f32>,
        context: &mut Vec<u32>,
        committed: &mut Vec<u32>,
        base_position: usize,
    ) -> Result<u32, RuntimeError> {
        let start = drafted
            .checked_mul(vocab)
            .ok_or(RuntimeError::SizeOverflow)?;
        let end = start.checked_add(vocab).ok_or(RuntimeError::SizeOverflow)?;
        row.clear();
        row.extend_from_slice(
            verify_activations
                .host_logits
                .get(start..end)
                .ok_or(RuntimeError::SizeOverflow)?,
        );
        options.penalties.apply(row, context)?;
        let target = distribution(row, &options.sampler)?;
        let token = select(&target, rng, (base_position + drafted) as u64)?;
        committed.push(token);
        context.push(token);
        self.backend.write_u32(&mut activations.sampled, &[token])?;
        Ok(token)
    }

    fn validate_draft_distributions(
        draft_distributions: Option<&[Distribution]>,
        proposed: usize,
    ) -> Result<(), RuntimeError> {
        if let Some(distributions) = draft_distributions {
            if distributions.len() != proposed {
                return Err(RuntimeError::CorrectableDistributionCount {
                    proposed,
                    distributions: distributions.len(),
                });
            }
        }
        Ok(())
    }

    fn sample(
        &mut self,
        activations: &mut Activations<B>,
        capture: LogitCapture,
        snapshots: &mut Vec<LogitSnapshot>,
        input_position: usize,
    ) -> Result<u32, RuntimeError> {
        self.enqueue_sample(activations)?;
        self.read_sample(activations, capture, snapshots, input_position)
    }

    fn greedy_continuation(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        tokens: usize,
    ) -> Result<Vec<u32>, RuntimeError> {
        let mut output = Vec::with_capacity(tokens);
        let mut snapshots = Vec::new();
        output.push(self.sample(
            activations,
            LogitCapture::Disabled,
            &mut snapshots,
            state.position - 1,
        )?);
        while output.len() < tokens {
            self.forward(state, activations, attention_shape)?;
            output.push(self.sample(
                activations,
                LogitCapture::Disabled,
                &mut snapshots,
                state.position - 1,
            )?);
        }
        self.backend.synchronize()?;
        Ok(output)
    }

    fn read_prefill_kv(
        &mut self,
        state: &KvState<B>,
        shape: AttentionShape,
        positions: usize,
    ) -> Result<PrefillKvSnapshot, RuntimeError> {
        let mut layers = Vec::with_capacity(state.layers.len());
        let valid_elements = self
            .model
            .config
            .n_head_kv
            .checked_mul(positions)
            .and_then(|value| value.checked_mul(self.model.config.head_dim))
            .ok_or(RuntimeError::SizeOverflow)?;
        for layer in &state.layers {
            let mut key_cache = vec![0_u16; shape.cache_elements()?];
            let mut value_cache = vec![0_u16; shape.cache_elements()?];
            self.backend.read_f16(&layer.key, &mut key_cache)?;
            self.backend.read_f16(&layer.value, &mut value_cache)?;
            let mut key = Vec::with_capacity(valid_elements);
            let mut value = Vec::with_capacity(valid_elements);
            for head in 0..self.model.config.n_head_kv {
                let start = head
                    .checked_mul(shape.max_context())
                    .and_then(|offset| offset.checked_mul(self.model.config.head_dim))
                    .ok_or(RuntimeError::SizeOverflow)?;
                let count = positions
                    .checked_mul(self.model.config.head_dim)
                    .ok_or(RuntimeError::SizeOverflow)?;
                key.extend_from_slice(&key_cache[start..start + count]);
                value.extend_from_slice(&value_cache[start..start + count]);
            }
            layers.push(PrefillKvLayerSnapshot { key, value });
        }
        Ok(PrefillKvSnapshot { layers })
    }
}

#[derive(Debug)]
struct PrefillKvSnapshot {
    layers: Vec<PrefillKvLayerSnapshot>,
}

#[derive(Debug)]
struct PrefillKvLayerSnapshot {
    key: Vec<u16>,
    value: Vec<u16>,
}

fn compare_prefill_kv(
    sequential: &PrefillKvSnapshot,
    chunked: &PrefillKvSnapshot,
) -> Vec<PrefillKvLayerError> {
    sequential
        .layers
        .iter()
        .zip(&chunked.layers)
        .enumerate()
        .map(|(layer, (sequential, chunked))| {
            let (key_max_abs, key_max_rel) = half_error(&sequential.key, &chunked.key);
            let (value_max_abs, value_max_rel) = half_error(&sequential.value, &chunked.value);
            PrefillKvLayerError {
                layer,
                key_max_abs,
                key_max_rel,
                value_max_abs,
                value_max_rel,
            }
        })
        .collect()
}

fn half_error(sequential: &[u16], chunked: &[u16]) -> (f32, f32) {
    sequential
        .iter()
        .zip(chunked)
        .fold((0.0_f32, 0.0_f32), |(max_abs, max_rel), (left, right)| {
            let left = half::f16::from_bits(*left).to_f32();
            let right = half::f16::from_bits(*right).to_f32();
            let absolute = (left - right).abs();
            let relative = absolute / left.abs().max(1e-6);
            (max_abs.max(absolute), max_rel.max(relative))
        })
}

#[derive(Debug)]
struct Activations<B: Backend> {
    hidden: B::Buffer,
    norm: B::Buffer,
    query: B::Buffer,
    key: B::Buffer,
    value: B::Buffer,
    query_norm: B::Buffer,
    key_norm: B::Buffer,
    attention: B::Buffer,
    residual: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
    ffn: B::Buffer,
    logits: B::Buffer,
    sampled: B::Buffer,
}

struct DecodePrimaryBuffers<B: Backend> {
    hidden: B::Buffer,
    norm: B::Buffer,
    query: B::Buffer,
    key: B::Buffer,
}

struct DecodeAttentionBuffers<B: Backend> {
    value: B::Buffer,
    query_norm: B::Buffer,
    key_norm: B::Buffer,
    attention: B::Buffer,
}

struct DecodeFfnBuffers<B: Backend> {
    residual: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
    ffn: B::Buffer,
}

#[derive(Debug)]
struct HibernatedActivations {
    buffers: Vec<BufferSnapshot>,
}

impl HibernatedActivations {
    fn capture<B: Backend>(backend: &mut B, source: &Activations<B>) -> Result<Self, BackendError> {
        Ok(Self {
            buffers: vec![
                backend.download_buffer(&source.hidden)?,
                backend.download_buffer(&source.norm)?,
                backend.download_buffer(&source.query)?,
                backend.download_buffer(&source.key)?,
                backend.download_buffer(&source.value)?,
                backend.download_buffer(&source.query_norm)?,
                backend.download_buffer(&source.key_norm)?,
                backend.download_buffer(&source.attention)?,
                backend.download_buffer(&source.residual)?,
                backend.download_buffer(&source.gate)?,
                backend.download_buffer(&source.up)?,
                backend.download_buffer(&source.ffn)?,
                backend.download_buffer(&source.logits)?,
                backend.download_buffer(&source.sampled)?,
            ],
        })
    }

    fn restore<B: Backend>(&self, backend: &mut B) -> Result<Activations<B>, BackendError> {
        let mut buffers = self.buffers.iter();
        let restored = restore_activation_buffers(backend, &mut buffers)?;
        let [hidden, norm, query, key, value, query_norm, key_norm, attention, residual, gate, up, ffn, logits, sampled] =
            restored.try_into().map_err(|_| {
                BackendError::operation("restore activations", "snapshot buffer is missing")
            })?;
        let activations = Activations {
            hidden,
            norm,
            query,
            key,
            value,
            query_norm,
            key_norm,
            attention,
            residual,
            gate,
            up,
            ffn,
            logits,
            sampled,
        };
        if buffers.next().is_some() {
            return Err(BackendError::operation(
                "restore activations",
                "snapshot has extra buffers",
            ));
        }
        Ok(activations)
    }
}

fn restore_activation_buffers<'a, B: Backend, I>(
    backend: &mut B,
    buffers: &mut I,
) -> Result<Vec<B::Buffer>, BackendError>
where
    I: Iterator<Item = &'a BufferSnapshot>,
{
    let mut restored = Vec::with_capacity(14);
    for _ in 0..14 {
        let source = buffers.next().ok_or_else(|| {
            BackendError::operation("restore activations", "snapshot buffer is missing")
        })?;
        restored.push(backend.restore_buffer(source)?);
    }
    Ok(restored)
}

fn clone_activation_buffers<B: Backend>(
    backend: &mut B,
    source: &Activations<B>,
) -> Result<Vec<B::Buffer>, BackendError> {
    let sources = [
        &source.hidden,
        &source.norm,
        &source.query,
        &source.key,
        &source.value,
        &source.query_norm,
        &source.key_norm,
        &source.attention,
        &source.residual,
        &source.gate,
        &source.up,
        &source.ffn,
        &source.logits,
        &source.sampled,
    ];
    sources
        .into_iter()
        .map(|buffer| backend.clone_buffer(buffer))
        .collect()
}

#[derive(Debug)]
struct VerifyActivations<B: Backend> {
    positions: usize,
    tokens: B::Buffer,
    hidden: B::Buffer,
    norm: B::Buffer,
    query: B::Buffer,
    key: B::Buffer,
    value: B::Buffer,
    query_norm: B::Buffer,
    key_norm: B::Buffer,
    attention: B::Buffer,
    residual: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
    ffn: B::Buffer,
    logits: B::Buffer,
    input_tokens: Vec<u32>,
    host_logits: Vec<f32>,
}

struct VerifyPrimaryBuffers<B: Backend> {
    tokens: B::Buffer,
    hidden: B::Buffer,
    norm: B::Buffer,
    query: B::Buffer,
}

struct VerifyProjectedBuffers<B: Backend> {
    key: B::Buffer,
    value: B::Buffer,
    query_norm: B::Buffer,
    key_norm: B::Buffer,
}

struct VerifySecondaryBuffers<B: Backend> {
    attention: B::Buffer,
    residual: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
}

impl<B: Backend> VerifyActivations<B> {
    fn new(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
    ) -> Result<Self, BackendError> {
        let kv_columns =
            config
                .n_head_kv
                .checked_mul(config.head_dim)
                .ok_or(BackendError::SizeOverflow {
                    field: "verifier projected KV elements",
                })?;
        let logit_elements =
            positions
                .checked_mul(config.vocab_size)
                .ok_or(BackendError::SizeOverflow {
                    field: "verifier logit elements",
                })?;
        let VerifyPrimaryBuffers {
            tokens,
            hidden,
            norm,
            query,
        } = Self::allocate_primary(backend, config, positions)?;
        let VerifyProjectedBuffers {
            key,
            value,
            query_norm,
            key_norm,
        } = Self::allocate_projected(backend, config, positions, kv_columns)?;
        let VerifySecondaryBuffers {
            attention,
            residual,
            gate,
            up,
        } = Self::allocate_secondary(backend, config, positions)?;
        let (ffn, logits) = Self::allocate_output(backend, config, positions, logit_elements)?;
        Ok(Self {
            positions,
            tokens,
            hidden,
            norm,
            query,
            key,
            value,
            query_norm,
            key_norm,
            attention,
            residual,
            gate,
            up,
            ffn,
            logits,
            input_tokens: Vec::with_capacity(positions),
            host_logits: vec![0.0; logit_elements],
        })
    }

    fn dense(
        positions: usize,
        columns: usize,
        field: &'static str,
    ) -> Result<BufferLayout, BackendError> {
        positions
            .checked_mul(columns)
            .ok_or(BackendError::SizeOverflow { field })
            .and_then(BufferLayout::f32)
    }

    fn allocate_primary(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
    ) -> Result<VerifyPrimaryBuffers<B>, BackendError> {
        Ok(VerifyPrimaryBuffers {
            tokens: backend.allocate(BufferLayout::u32(positions)?)?,
            hidden: backend.allocate(Self::dense(
                positions,
                config.n_embd,
                "verifier hidden elements",
            )?)?,
            norm: backend.allocate(Self::dense(
                positions,
                config.n_embd,
                "verifier norm elements",
            )?)?,
            query: backend.allocate(Self::dense(
                positions,
                config.n_embd,
                "verifier query elements",
            )?)?,
        })
    }

    fn allocate_projected(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
        kv_columns: usize,
    ) -> Result<VerifyProjectedBuffers<B>, BackendError> {
        Ok(VerifyProjectedBuffers {
            key: backend.allocate(Self::dense(positions, kv_columns, "verifier key elements")?)?,
            value: backend.allocate(Self::dense(
                positions,
                kv_columns,
                "verifier value elements",
            )?)?,
            query_norm: backend.allocate(Self::dense(
                positions,
                config.n_embd,
                "verifier query norm elements",
            )?)?,
            key_norm: backend.allocate(Self::dense(
                positions,
                kv_columns,
                "verifier key norm elements",
            )?)?,
        })
    }

    fn allocate_secondary(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
    ) -> Result<VerifySecondaryBuffers<B>, BackendError> {
        Ok(VerifySecondaryBuffers {
            attention: backend.allocate(Self::dense(
                positions,
                config.n_embd,
                "verifier attention elements",
            )?)?,
            residual: backend.allocate(Self::dense(
                positions,
                config.n_embd,
                "verifier residual elements",
            )?)?,
            gate: backend.allocate(Self::dense(
                positions,
                config.n_ff,
                "verifier gate elements",
            )?)?,
            up: backend.allocate(Self::dense(positions, config.n_ff, "verifier up elements")?)?,
        })
    }

    fn allocate_output(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
        logit_elements: usize,
    ) -> Result<(B::Buffer, B::Buffer), BackendError> {
        Ok((
            backend.allocate(Self::dense(
                positions,
                config.n_ff,
                "verifier FFN elements",
            )?)?,
            backend.allocate(BufferLayout::f32(logit_elements)?)?,
        ))
    }
}

impl<B: Backend> Activations<B> {
    fn new(backend: &mut B, config: &crate::ModelConfig) -> Result<Self, BackendError> {
        let DecodePrimaryBuffers {
            hidden,
            norm,
            query,
            key,
        } = Self::allocate_primary(backend, config)?;
        let DecodeAttentionBuffers {
            value,
            query_norm,
            key_norm,
            attention,
        } = Self::allocate_attention(backend, config)?;
        let DecodeFfnBuffers {
            residual,
            gate,
            up,
            ffn,
        } = Self::allocate_ffn(backend, config)?;
        let (logits, sampled) = Self::allocate_output(backend, config)?;
        Ok(Self {
            hidden,
            norm,
            query,
            key,
            value,
            query_norm,
            key_norm,
            attention,
            residual,
            gate,
            up,
            ffn,
            logits,
            sampled,
        })
    }

    fn allocate_primary(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<DecodePrimaryBuffers<B>, BackendError> {
        Ok(DecodePrimaryBuffers {
            hidden: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            norm: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            query: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            key: backend.allocate(BufferLayout::f32(config.n_head_kv * config.head_dim)?)?,
        })
    }

    fn allocate_attention(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<DecodeAttentionBuffers<B>, BackendError> {
        Ok(DecodeAttentionBuffers {
            value: backend.allocate(BufferLayout::f32(config.n_head_kv * config.head_dim)?)?,
            query_norm: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            key_norm: backend.allocate(BufferLayout::f32(config.n_head_kv * config.head_dim)?)?,
            attention: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
        })
    }

    fn allocate_ffn(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<DecodeFfnBuffers<B>, BackendError> {
        Ok(DecodeFfnBuffers {
            residual: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            gate: backend.allocate(BufferLayout::f32(config.n_ff)?)?,
            up: backend.allocate(BufferLayout::f32(config.n_ff)?)?,
            ffn: backend.allocate(BufferLayout::f32(config.n_ff)?)?,
        })
    }

    fn allocate_output(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<(B::Buffer, B::Buffer), BackendError> {
        Ok((
            backend.allocate(BufferLayout::f32(config.vocab_size)?)?,
            backend.allocate(BufferLayout::u32(1)?)?,
        ))
    }

    fn fork(backend: &mut B, source: &Self) -> Result<Self, BackendError> {
        let cloned = clone_activation_buffers(backend, source)?;
        let [hidden, norm, query, key, value, query_norm, key_norm, attention, residual, gate, up, ffn, logits, sampled] =
            cloned.try_into().map_err(|_| {
                BackendError::operation("fork activations", "activation buffer count differs")
            })?;
        Ok(Self {
            hidden,
            norm,
            query,
            key,
            value,
            query_norm,
            key_norm,
            attention,
            residual,
            gate,
            up,
            ffn,
            logits,
            sampled,
        })
    }

    fn bytes(config: &crate::ModelConfig) -> Result<u64, RuntimeError> {
        let elements = Self::element_count(config)?;
        let bytes = elements
            .checked_mul(4)
            .and_then(|value| value.checked_add(4))
            .ok_or(RuntimeError::SizeOverflow)?;
        u64::try_from(bytes).map_err(|_| RuntimeError::SizeOverflow)
    }

    fn element_count(config: &crate::ModelConfig) -> Result<usize, RuntimeError> {
        let kv_columns = config
            .n_head_kv
            .checked_mul(config.head_dim)
            .ok_or(RuntimeError::SizeOverflow)?;
        config
            .n_embd
            .checked_mul(6)
            .and_then(|value| value.checked_add(kv_columns.checked_mul(3)?))
            .and_then(|value| value.checked_add(config.n_ff.checked_mul(3)?))
            .and_then(|value| value.checked_add(config.vocab_size))
            .ok_or(RuntimeError::SizeOverflow)
    }
}

#[derive(Debug)]
enum PreparedPrefill<B: Backend> {
    Reused,
    Sequential,
    Chunked {
        chunk_tokens: usize,
        full: PrefillActivations<B>,
        tail: Option<PrefillActivations<B>>,
        workspace: PrefillWorkspace,
    },
}

#[derive(Debug, Clone, Copy)]
struct PrefillExecution {
    processed_tokens: usize,
    cancelled: bool,
    workspace: PrefillWorkspace,
}

#[derive(Debug)]
struct PrefillActivations<B: Backend> {
    tokens: B::Buffer,
    hidden: B::Buffer,
    norm: B::Buffer,
    query: B::Buffer,
    key: B::Buffer,
    value: B::Buffer,
    query_norm: B::Buffer,
    key_norm: B::Buffer,
    attention: B::Buffer,
    residual: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
    ffn: B::Buffer,
}

struct PrefillPrimaryBuffers<B: Backend> {
    tokens: B::Buffer,
    hidden: B::Buffer,
    norm: B::Buffer,
    query: B::Buffer,
}

struct PrefillProjectedBuffers<B: Backend> {
    key: B::Buffer,
    value: B::Buffer,
    query_norm: B::Buffer,
    key_norm: B::Buffer,
}

struct PrefillSecondaryBuffers<B: Backend> {
    attention: B::Buffer,
    residual: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
}

#[derive(Debug)]
struct PrefillEvalActivations<B: Backend> {
    forward: PrefillActivations<B>,
    logits: B::Buffer,
    host_logits: Vec<f32>,
}

impl<B: Backend> PrefillEvalActivations<B> {
    fn new(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
    ) -> Result<Self, BackendError> {
        let elements = tokens
            .checked_mul(config.vocab_size)
            .ok_or(BackendError::SizeOverflow {
                field: "prefill evaluation logits",
            })?;
        Ok(Self {
            forward: PrefillActivations::new(backend, config, tokens)?,
            logits: backend.allocate(BufferLayout::f32(elements)?)?,
            host_logits: vec![0.0; elements],
        })
    }
}

impl<B: Backend> PrefillActivations<B> {
    fn new(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
    ) -> Result<Self, BackendError> {
        let kv_columns =
            config
                .n_head_kv
                .checked_mul(config.head_dim)
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill projected KV elements",
                })?;
        let PrefillPrimaryBuffers {
            tokens: tokens_buffer,
            hidden,
            norm,
            query,
        } = Self::allocate_primary(backend, config, tokens)?;
        let PrefillProjectedBuffers {
            key,
            value,
            query_norm,
            key_norm,
        } = Self::allocate_projected(backend, config, tokens, kv_columns)?;
        let PrefillSecondaryBuffers {
            attention,
            residual,
            gate,
            up,
        } = Self::allocate_secondary(backend, config, tokens)?;
        let ffn = backend.allocate(Self::dense(tokens, config.n_ff)?)?;
        Ok(Self {
            tokens: tokens_buffer,
            hidden,
            norm,
            query,
            key,
            value,
            query_norm,
            key_norm,
            attention,
            residual,
            gate,
            up,
            ffn,
        })
    }

    fn dense(tokens: usize, columns: usize) -> Result<BufferLayout, BackendError> {
        tokens
            .checked_mul(columns)
            .ok_or(BackendError::SizeOverflow {
                field: "prefill activation elements",
            })
            .and_then(BufferLayout::f32)
    }

    fn allocate_primary(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
    ) -> Result<PrefillPrimaryBuffers<B>, BackendError> {
        Ok(PrefillPrimaryBuffers {
            tokens: backend.allocate(BufferLayout::u32(tokens)?)?,
            hidden: backend.allocate(Self::dense(tokens, config.n_embd)?)?,
            norm: backend.allocate(Self::dense(tokens, config.n_embd)?)?,
            query: backend.allocate(Self::dense(tokens, config.n_embd)?)?,
        })
    }

    fn allocate_projected(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
        kv_columns: usize,
    ) -> Result<PrefillProjectedBuffers<B>, BackendError> {
        Ok(PrefillProjectedBuffers {
            key: backend.allocate(Self::dense(tokens, kv_columns)?)?,
            value: backend.allocate(Self::dense(tokens, kv_columns)?)?,
            query_norm: backend.allocate(Self::dense(tokens, config.n_embd)?)?,
            key_norm: backend.allocate(Self::dense(tokens, kv_columns)?)?,
        })
    }

    fn allocate_secondary(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
    ) -> Result<PrefillSecondaryBuffers<B>, BackendError> {
        Ok(PrefillSecondaryBuffers {
            attention: backend.allocate(Self::dense(tokens, config.n_embd)?)?,
            residual: backend.allocate(Self::dense(tokens, config.n_embd)?)?,
            gate: backend.allocate(Self::dense(tokens, config.n_ff)?)?,
            up: backend.allocate(Self::dense(tokens, config.n_ff)?)?,
        })
    }

    fn bytes(config: &crate::ModelConfig, tokens: usize) -> Result<u64, RuntimeError> {
        let row_elements = Self::row_elements(config)?;
        let dense_bytes = tokens
            .checked_mul(row_elements)
            .and_then(|value| value.checked_mul(4))
            .ok_or(RuntimeError::SizeOverflow)?;
        let token_bytes = tokens.checked_mul(4).ok_or(RuntimeError::SizeOverflow)?;
        u64::try_from(
            dense_bytes
                .checked_add(token_bytes)
                .ok_or(RuntimeError::SizeOverflow)?,
        )
        .map_err(|_| RuntimeError::SizeOverflow)
    }

    fn row_elements(config: &crate::ModelConfig) -> Result<usize, RuntimeError> {
        let kv_columns = config
            .n_head_kv
            .checked_mul(config.head_dim)
            .ok_or(RuntimeError::SizeOverflow)?;
        config
            .n_embd
            .checked_mul(6)
            .and_then(|value| value.checked_add(kv_columns.checked_mul(3)?))
            .and_then(|value| value.checked_add(config.n_ff.checked_mul(3)?))
            .ok_or(RuntimeError::SizeOverflow)
    }
}

#[derive(Debug)]
struct KvState<B: Backend> {
    _allocator: StateAllocator,
    layers: Vec<KvLayer<B>>,
    position: usize,
    device_position: B::Buffer,
    shape: AttentionShape,
    dtype: KvCacheDtype,
}

#[derive(Debug)]
struct HibernatedKvState {
    layers: Vec<(BufferSnapshot, BufferSnapshot)>,
    position: usize,
    device_position: BufferSnapshot,
    shape: AttentionShape,
    dtype: KvCacheDtype,
}

impl HibernatedKvState {
    fn capture<B: Backend>(backend: &mut B, source: &KvState<B>) -> Result<Self, BackendError> {
        let mut layers = Vec::with_capacity(source.layers.len());
        for layer in &source.layers {
            layers.push((
                backend.download_buffer(&layer.key)?,
                backend.download_buffer(&layer.value)?,
            ));
        }
        Ok(Self {
            layers,
            position: source.position,
            device_position: backend.download_buffer(&source.device_position)?,
            shape: source.shape,
            dtype: source.dtype,
        })
    }

    fn restore<B: Backend>(&self, backend: &mut B) -> Result<KvState<B>, RuntimeError> {
        let (_cache_layout, one_cache_bytes, capacity) =
            kv_state_accounting(self.dtype, self.shape, self.layers.len())?;
        let mut allocator = StateAllocator::new(capacity);
        let layers = restore_kv_layers(backend, &mut allocator, &self.layers, one_cache_bytes)?;
        Ok(KvState {
            _allocator: allocator,
            layers,
            position: self.position,
            device_position: backend.restore_buffer(&self.device_position)?,
            shape: self.shape,
            dtype: self.dtype,
        })
    }
}

impl<B: Backend> KvState<B> {
    fn new(
        backend: &mut B,
        layers: usize,
        shape: AttentionShape,
        dtype: KvCacheDtype,
    ) -> Result<Self, RuntimeError> {
        let (cache_layout, one_cache_bytes, capacity) = kv_state_accounting(dtype, shape, layers)?;
        let mut allocator = StateAllocator::new(capacity);
        let state_layers = allocate_kv_layers(
            backend,
            &mut allocator,
            cache_layout,
            one_cache_bytes,
            layers,
        )?;
        Ok(Self {
            _allocator: allocator,
            layers: state_layers,
            position: 0,
            device_position: backend.allocate(BufferLayout::u32(1)?)?,
            shape,
            dtype,
        })
    }

    fn fork(backend: &mut B, source: &Self) -> Result<Self, RuntimeError> {
        let (_cache_layout, one_cache_bytes, capacity) =
            kv_state_accounting(source.dtype, source.shape, source.layers.len())?;
        let mut allocator = StateAllocator::new(capacity);
        let layers = fork_kv_layers(backend, &mut allocator, &source.layers, one_cache_bytes)?;
        Ok(Self {
            _allocator: allocator,
            layers,
            position: source.position,
            device_position: backend.clone_buffer(&source.device_position)?,
            shape: source.shape,
            dtype: source.dtype,
        })
    }
}

fn kv_state_accounting(
    dtype: KvCacheDtype,
    shape: AttentionShape,
    layers: usize,
) -> Result<(BufferLayout, u64, u64), RuntimeError> {
    let cache_layout = dtype.layout(shape.cache_elements()?)?;
    let one_cache_bytes =
        u64::try_from(cache_layout.bytes()).map_err(|_| RuntimeError::SizeOverflow)?;
    let capacity = one_cache_bytes
        .checked_mul(2)
        .and_then(|bytes| {
            u64::try_from(layers)
                .ok()
                .and_then(|count| bytes.checked_mul(count))
        })
        .ok_or(RuntimeError::SizeOverflow)?;
    Ok((cache_layout, one_cache_bytes, capacity))
}

fn allocate_kv_layers<B: Backend>(
    backend: &mut B,
    allocator: &mut StateAllocator,
    cache_layout: BufferLayout,
    one_cache_bytes: u64,
    layers: usize,
) -> Result<Vec<KvLayer<B>>, RuntimeError> {
    let mut state_layers = Vec::with_capacity(layers);
    for _ in 0..layers {
        let key_allocation =
            allocator.allocate(StateKind::Kv, one_cache_bytes, StateLifetime::Committed)?;
        let value_allocation =
            allocator.allocate(StateKind::Kv, one_cache_bytes, StateLifetime::Committed)?;
        state_layers.push(KvLayer {
            key: backend.allocate(cache_layout)?,
            value: backend.allocate(cache_layout)?,
            _key_allocation: key_allocation,
            _value_allocation: value_allocation,
        });
    }
    Ok(state_layers)
}

fn restore_kv_layers<B: Backend>(
    backend: &mut B,
    allocator: &mut StateAllocator,
    sources: &[(BufferSnapshot, BufferSnapshot)],
    one_cache_bytes: u64,
) -> Result<Vec<KvLayer<B>>, RuntimeError> {
    let mut layers = Vec::with_capacity(sources.len());
    for (key, value) in sources {
        layers.push(KvLayer {
            key: backend.restore_buffer(key)?,
            value: backend.restore_buffer(value)?,
            _key_allocation: allocator.allocate(
                StateKind::Kv,
                one_cache_bytes,
                StateLifetime::Committed,
            )?,
            _value_allocation: allocator.allocate(
                StateKind::Kv,
                one_cache_bytes,
                StateLifetime::Committed,
            )?,
        });
    }
    Ok(layers)
}

fn fork_kv_layers<B: Backend>(
    backend: &mut B,
    allocator: &mut StateAllocator,
    sources: &[KvLayer<B>],
    one_cache_bytes: u64,
) -> Result<Vec<KvLayer<B>>, RuntimeError> {
    let mut layers = Vec::with_capacity(sources.len());
    for source in sources {
        let key_allocation =
            allocator.allocate(StateKind::Kv, one_cache_bytes, StateLifetime::Committed)?;
        let value_allocation =
            allocator.allocate(StateKind::Kv, one_cache_bytes, StateLifetime::Committed)?;
        layers.push(KvLayer {
            key: backend.clone_buffer(&source.key)?,
            value: backend.clone_buffer(&source.value)?,
            _key_allocation: key_allocation,
            _value_allocation: value_allocation,
        });
    }
    Ok(layers)
}

#[derive(Debug)]
struct KvLayer<B: Backend> {
    key: B::Buffer,
    value: B::Buffer,
    _key_allocation: StateAllocation,
    _value_allocation: StateAllocation,
}

fn emit<B: Backend, F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>>(
    model: &LoadedModel<B>,
    token: u32,
    tokens: &mut Vec<u32>,
    callback: &mut F,
) -> Result<(), RuntimeError> {
    let emitted = GeneratedToken {
        id: token,
        bytes: model.tokenizer.token_bytes(token)?,
    };
    callback(&emitted)?;
    tokens.push(token);
    Ok(())
}

fn first_token(tokens: &[u32]) -> u32 {
    tokens[tokens.len() - 1]
}

fn common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn top_logits(values: &[f32], count: usize) -> Result<Vec<Logit>, RuntimeError> {
    let mut logits = Vec::with_capacity(values.len());
    for (token, value) in values.iter().copied().enumerate() {
        if !value.is_nan() {
            logits.push(Logit {
                token: u32::try_from(token).map_err(|_| RuntimeError::SizeOverflow)?,
                value,
            });
        }
    }
    logits.sort_by(|left, right| {
        right
            .value
            .total_cmp(&left.value)
            .then_with(|| left.token.cmp(&right.token))
    });
    logits.truncate(count);
    Ok(logits)
}

/// Returns the SHA-256 digest of a token stream in little-endian u32 order.
///
/// Prompts and transcripts hash the same way. Digests from `generate` and
/// `bench` are comparable.
pub fn token_stream_sha256(tokens: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    leone_receipt::sha256_bytes(&bytes)
}

/// Returns the distribution of a drafter that proposes one token outright.
fn point_mass(vocab: usize, token: u32) -> Result<Distribution, RuntimeError> {
    let mut row = vec![f32::NEG_INFINITY; vocab];
    *row.get_mut(token as usize)
        .ok_or(RuntimeError::SizeOverflow)? = 0.0;
    Ok(distribution(&row, &Sampler::greedy())?)
}

fn measured_rate(tokens: usize, duration: Duration) -> MeasuredRate {
    if tokens == 0 || duration.is_zero() {
        MeasuredRate::Unavailable
    } else {
        MeasuredRate::TokensPerSecond(tokens as f64 / duration.as_secs_f64())
    }
}

fn cancelled_result(
    prompt_tokens: Vec<u32>,
    prefill_duration: Duration,
    prefill_method: PrefillMethod,
    prefill_workspace: PrefillWorkspace,
) -> GenerationResult {
    GenerationResult {
        stats: GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            emitted_tokens: 0,
            decode_evaluations: 0,
            prefill_duration,
            ttft_duration: None,
            decode_duration: Duration::ZERO,
            verify_duration: Duration::ZERO,
            cancelled: true,
            prefill_method,
            prefill_workspace,
            speculation: SpeculationStats::default(),
        },
        prompt_tokens,
        tokens: Vec::new(),
        logits: Vec::new(),
        decode_profile: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_logits_use_lowest_token_on_a_tie() {
        assert_eq!(
            top_logits(&[f32::NAN, 4.0, 3.0, 4.0], 3).unwrap(),
            [
                Logit {
                    token: 1,
                    value: 4.0
                },
                Logit {
                    token: 3,
                    value: 4.0
                },
                Logit {
                    token: 2,
                    value: 3.0
                }
            ]
        );
    }

    #[test]
    fn zero_decode_work_has_an_explicit_unavailable_rate() {
        let stats = GenerationStats {
            prompt_tokens: 1,
            emitted_tokens: 1,
            decode_evaluations: 0,
            prefill_duration: Duration::from_secs(1),
            ttft_duration: Some(Duration::from_secs(1)),
            decode_duration: Duration::ZERO,
            verify_duration: Duration::ZERO,
            cancelled: false,
            speculation: SpeculationStats::default(),
            prefill_method: PrefillMethod::SequentialDecode,
            prefill_workspace: PrefillWorkspace::default(),
        };
        assert_eq!(stats.prefill_rate(), MeasuredRate::TokensPerSecond(1.0));
        assert_eq!(stats.decode_rate(), MeasuredRate::Unavailable);
    }
}
