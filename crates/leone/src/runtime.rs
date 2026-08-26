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

impl<B: Backend> Runtime<B> {
    /// Allocates and releases a complete KV cache without running the model.
    pub fn probe_kv_capacity(
        &mut self,
        context_tokens: usize,
        dtype: KvCacheDtype,
    ) -> Result<KvCapacityProbe, RuntimeError> {
        if context_tokens == 0 {
            return Err(RuntimeError::EmptyPrompt);
        }
        if context_tokens > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: context_tokens,
                capacity: self.model.config.context_length,
            });
        }
        let allocated_context_tokens = decode_graph_bucket(context_tokens)?;
        let shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            allocated_context_tokens,
        )?;
        let one_cache_bytes = u64::try_from(dtype.layout(shape.cache_elements()?)?.bytes())
            .map_err(|_| RuntimeError::SizeOverflow)?;
        let cache_bytes = one_cache_bytes
            .checked_mul(2)
            .and_then(|bytes| {
                u64::try_from(self.model.config.n_layer)
                    .ok()
                    .and_then(|layers| bytes.checked_mul(layers))
            })
            .ok_or(RuntimeError::SizeOverflow)?;
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
            let mut state = KvState::new(
                &mut self.backend,
                self.model.config.n_layer,
                attention_shape,
                kv_cache_dtype,
            )?;
            let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
            for (offset, token) in window.iter().copied().enumerate() {
                self.backend.write_u32(&mut activations.sampled, &[token])?;
                self.forward(&mut state, &mut activations, attention_shape)?;
                if offset + 1 < window.len() {
                    self.backend.read_f32(&activations.logits, &mut logits)?;
                    on_logits(start + offset + 1, &logits)?;
                    scored = scored.checked_add(1).ok_or(RuntimeError::SizeOverflow)?;
                }
            }
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
            let plan = PrefillPlan::new(
                block_tokens,
                window.len(),
                self.model.config.n_head,
                self.model.config.n_head_kv,
                self.model.config.head_dim,
                self.model.config.n_embd,
                self.model.config.n_ff,
                self.model.config.n_ff.max(self.model.config.vocab_size),
            )?;
            self.backend.prepare_prefill(plan)?;
            let mut state = KvState::new(
                &mut self.backend,
                self.model.config.n_layer,
                attention_shape,
                kv_cache_dtype,
            )?;
            let mut full =
                PrefillEvalActivations::new(&mut self.backend, &self.model.config, block_tokens)?;
            let remainder = window.len() % block_tokens;
            let mut tail = if remainder == 0 {
                None
            } else {
                Some(PrefillEvalActivations::new(
                    &mut self.backend,
                    &self.model.config,
                    remainder,
                )?)
            };
            for offset in (0..window.len()).step_by(block_tokens) {
                let block_end = (offset + block_tokens).min(window.len());
                let block = &window[offset..block_end];
                let activations = if block.len() == block_tokens {
                    &mut full
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
                for row_index in 0..block.len() {
                    let window_position = offset + row_index;
                    if window_position + 1 >= window.len() {
                        break;
                    }
                    let row_start = row_index
                        .checked_mul(self.model.config.vocab_size)
                        .ok_or(RuntimeError::SizeOverflow)?;
                    let row_end = row_start
                        .checked_add(self.model.config.vocab_size)
                        .ok_or(RuntimeError::SizeOverflow)?;
                    on_logits(
                        start + window_position + 1,
                        activations
                            .host_logits
                            .get(row_start..row_end)
                            .ok_or(RuntimeError::SizeOverflow)?,
                    )?;
                    scored = scored.checked_add(1).ok_or(RuntimeError::SizeOverflow)?;
                }
                state.position = block_end;
            }
        }
        self.backend.synchronize()?;
        Ok(scored)
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
        let attention_shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )?;
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            kv_cache_dtype,
        )?;
        let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
        let mut prepared = if kv_cache_dtype == KvCacheDtype::Q8 {
            PreparedPrefill::Sequential
        } else {
            self.prepare_prompt_prefill(
                prompt_tokens.len(),
                prompt_tokens.len(),
                prefill_chunk_tokens,
            )?
        };

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
        let use_graph =
            decode_execution == DecodeExecution::Graph && self.backend.decode_graph_supported();
        if use_graph {
            self.capture_decode_graph(&mut state, &mut activations, attention_shape)?;
        }

        let mut detokenized = Vec::with_capacity(self.model.tokenizer.max_token_bytes());
        let mut transcript = Vec::with_capacity(decode_tokens);
        let decode_start = Instant::now();
        let mut detokenized_bytes = 0_usize;
        for _ in 0..decode_tokens {
            if use_graph {
                self.backend.replay_decode_graph()?;
                state.position = state
                    .position
                    .checked_add(1)
                    .ok_or(RuntimeError::SizeOverflow)?;
            } else {
                self.forward(&mut state, &mut activations, attention_shape)?;
                self.enqueue_sample(&mut activations)?;
            }
            let token = self.read_sample(
                &mut activations,
                LogitCapture::Disabled,
                &mut snapshots,
                state.position - 1,
            )?;
            transcript.push(token);
            self.model
                .tokenizer
                .token_bytes_into(token, &mut detokenized)?;
            detokenized_bytes = detokenized_bytes
                .checked_add(detokenized.len())
                .ok_or(RuntimeError::SizeOverflow)?;
        }
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

    /// Measures one prompt prefill through the first sampled token.
    pub fn benchmark_prefill(
        &mut self,
        prompt_tokens: &[u32],
        kv_cache_dtype: KvCacheDtype,
        chunk_tokens: usize,
    ) -> Result<PrefillBenchmarkRun, RuntimeError> {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if prompt_tokens.len() > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: prompt_tokens.len(),
                capacity: self.model.config.context_length,
            });
        }
        let context_bucket = decode_graph_bucket(prompt_tokens.len())?;
        let attention_shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )?;
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            kv_cache_dtype,
        )?;
        let mut activations = Activations::new(&mut self.backend, &self.model.config)?;
        let mut prepared = if kv_cache_dtype == KvCacheDtype::Q8 {
            PreparedPrefill::Sequential
        } else {
            self.prepare_prompt_prefill(prompt_tokens.len(), prompt_tokens.len(), chunk_tokens)?
        };
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
        let attention_shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )?;

        let (sequential_kv, sequential_tokens) = {
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
            (kv, tokens)
        };

        let mut prepared =
            self.prepare_prompt_prefill(prompt_tokens.len(), prompt_tokens.len(), chunk_tokens)?;
        let (chunked_kv, chunked_tokens) = {
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
                &mut prepared,
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
            (kv, tokens)
        };

        let (repeated_kv, repeated_chunked_tokens) = {
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
                &mut prepared,
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
            (kv, tokens)
        };

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

    /// Compares one verifier pass with consecutive one-position decode passes.
    pub fn characterize_verify(
        &mut self,
        prompt_tokens: &[u32],
        verify_tokens: &[u32],
        kv_cache_dtype: KvCacheDtype,
    ) -> Result<VerifyCharacterization, RuntimeError> {
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
        let attention_shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            decode_graph_bucket(context)?,
        )?;
        let mut sequential = Vec::with_capacity(
            verify_tokens
                .len()
                .checked_mul(self.model.config.vocab_size)
                .ok_or(RuntimeError::SizeOverflow)?,
        );
        {
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
            let mut row = vec![0.0; self.model.config.vocab_size];
            for token in verify_tokens.iter().copied() {
                self.backend.write_u32(&mut activations.sampled, &[token])?;
                self.forward(&mut state, &mut activations, attention_shape)?;
                self.backend.read_f32(&activations.logits, &mut row)?;
                sequential.extend_from_slice(&row);
            }
        }
        let verifier = {
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
            verify.host_logits
        };
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

    /// Generates from prompt tokens while retaining verified KV for the next call.
    pub fn generate_session_tokens<F, C>(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: GenerateOptions,
        mut on_token: F,
        mut cancelled: C,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
    {
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
        let decode_context = options
            .max_tokens
            .checked_sub(1)
            .ok_or(RuntimeError::SizeOverflow)?;
        let requested_context = prompt_tokens
            .len()
            .checked_add(decode_context)
            .ok_or(RuntimeError::SizeOverflow)?;
        if requested_context > self.model.config.context_length {
            return Err(RuntimeError::ContextCapacity {
                requested: requested_context,
                capacity: self.model.config.context_length,
            });
        }

        let context_bucket = decode_graph_bucket(requested_context)?;
        let attention_shape = AttentionShape::new(
            self.model.config.n_head,
            self.model.config.n_head_kv,
            self.model.config.head_dim,
            context_bucket,
        )?;
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
        let reuse_class =
            if compatible && common_tokens == prompt_tokens.len() && common_tokens == cached_tokens
            {
                SessionReuseClass::ExactRepeat
            } else if compatible && common_tokens == cached_tokens && common_tokens > 0 {
                SessionReuseClass::AppendOnly
            } else if compatible && common_tokens > 0 {
                SessionReuseClass::ArbitraryBranch
            } else if common_tokens == cached_tokens && common_tokens > 0 {
                // A larger context bucket can invalidate device buffers even
                // when the new prompt only appends to accepted history. Replay
                // those host tokens and keep sampler feedback from that exact
                // prefix.
                SessionReuseClass::RestoreReplay
            } else if restored_common > 0 {
                SessionReuseClass::RestoreReplay
            } else {
                SessionReuseClass::Cold
            };
        let reuse_class = if compatible && common_tokens > 0 && session.pending_wake.is_some() {
            SessionReuseClass::HostWake
        } else if compatible && common_tokens > 0 && session.pending_fork.is_some() {
            SessionReuseClass::DeviceFork
        } else {
            reuse_class
        };
        let mut reused_tokens = if compatible { common_tokens } else { 0 };
        if compatible && reused_tokens == prompt_tokens.len() && reused_tokens < cached_tokens {
            reused_tokens = reused_tokens.saturating_sub(1);
        }
        let (mut state, mut activations) = if compatible {
            let mut state = session.state.take().ok_or(RuntimeError::SizeOverflow)?;
            state.position = reused_tokens;
            let activations = session
                .activations
                .take()
                .ok_or(RuntimeError::SizeOverflow)?;
            (state, activations)
        } else {
            session.state = None;
            session.activations = None;
            (
                KvState::new(
                    &mut self.backend,
                    self.model.config.n_layer,
                    attention_shape,
                    options.kv_cache_dtype,
                )?,
                Activations::new(&mut self.backend, &self.model.config)?,
            )
        };
        let configured_verify_activations = match options.speculation {
            Speculation::Suffix(drafter)
                if self.backend.verify_supported()
                    && options.kv_cache_dtype != KvCacheDtype::Q8 =>
            {
                let positions = drafter
                    .proposal()
                    .get()
                    .checked_add(1)
                    .ok_or(RuntimeError::SizeOverflow)?;
                if positions <= 8 {
                    Some(VerifyActivations::new(
                        &mut self.backend,
                        &self.model.config,
                        positions,
                    )?)
                } else {
                    None
                }
            }
            Speculation::Adaptive(drafter)
                if self.backend.verify_supported()
                    && options.kv_cache_dtype != KvCacheDtype::Q8 =>
            {
                let positions = drafter.maximum_verifier_positions();
                if positions <= 8 {
                    Some(VerifyActivations::new(
                        &mut self.backend,
                        &self.model.config,
                        positions,
                    )?)
                } else {
                    None
                }
            }
            Speculation::Correctable(drafter)
                if self.backend.verify_supported()
                    && options.kv_cache_dtype != KvCacheDtype::Q8 =>
            {
                let positions = drafter.maximum_verifier_positions();
                if positions <= 8 {
                    Some(VerifyActivations::new(
                        &mut self.backend,
                        &self.model.config,
                        positions,
                    )?)
                } else {
                    None
                }
            }
            _ => None,
        };
        let mut verify_activations: Vec<Option<VerifyActivations<B>>> =
            (0..9).map(|_| None).collect();
        if let Some(activations) = configured_verify_activations {
            let positions = activations.positions;
            verify_activations[positions] = Some(activations);
        }
        let prefill_tokens = &prompt_tokens[reused_tokens..];
        let split_replay = reuse_class == SessionReuseClass::RestoreReplay
            && reused_tokens == 0
            && replay_prefill_boundary > 0
            && replay_prefill_boundary < prompt_tokens.len();
        let mut prepared = if split_replay {
            PreparedPrefill::Reused
        } else if options.kv_cache_dtype == KvCacheDtype::Q8 || reused_tokens > 0 {
            // Appended session tokens must use the same decode kernels as an
            // uninterrupted generation. Chunked prefill is quality-equivalent
            // but can change a stochastic token at the continuation boundary.
            PreparedPrefill::Sequential
        } else {
            self.prepare_prompt_prefill(
                prefill_tokens.len(),
                prompt_tokens.len(),
                DEFAULT_PREFILL_CHUNK_TOKENS,
            )?
        };

        let prefill_start = Instant::now();
        let prefill = if split_replay {
            let mut initial_prepared = if options.kv_cache_dtype == KvCacheDtype::Q8 {
                PreparedPrefill::Sequential
            } else {
                self.prepare_prompt_prefill(
                    replay_prefill_boundary,
                    replay_prefill_boundary,
                    DEFAULT_PREFILL_CHUNK_TOKENS,
                )?
            };
            let initial = self.run_prompt_prefill(
                &prompt_tokens[..replay_prefill_boundary],
                &mut state,
                &mut activations,
                attention_shape,
                &mut initial_prepared,
                &mut cancelled,
            )?;
            if initial.cancelled {
                initial
            } else {
                let mut continuation_prepared = PreparedPrefill::Sequential;
                let continuation = self.run_prompt_prefill(
                    &prompt_tokens[replay_prefill_boundary..],
                    &mut state,
                    &mut activations,
                    attention_shape,
                    &mut continuation_prepared,
                    &mut cancelled,
                )?;
                PrefillExecution {
                    processed_tokens: replay_prefill_boundary
                        .checked_add(continuation.processed_tokens)
                        .ok_or(RuntimeError::SizeOverflow)?,
                    cancelled: continuation.cancelled,
                    workspace: initial.workspace,
                }
            }
        } else {
            self.run_prompt_prefill(
                prefill_tokens,
                &mut state,
                &mut activations,
                attention_shape,
                &mut prepared,
                &mut cancelled,
            )?
        };
        self.backend.synchronize()?;
        let prefill_duration = prefill_start.elapsed();
        if prefill.cancelled {
            session.invalidate();
            let processed_count = reused_tokens
                .checked_add(prefill.processed_tokens)
                .ok_or(RuntimeError::SizeOverflow)?;
            let processed = prompt_tokens
                .get(..processed_count)
                .ok_or(RuntimeError::SizeOverflow)?
                .to_vec();
            return Ok(cancelled_result(
                processed,
                prefill_duration,
                self.backend.prefill_method(),
                prefill.workspace,
            ));
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

        let mut tokens = Vec::with_capacity(options.max_tokens);
        let mut logits = Vec::new();
        let mut row: Vec<f32> = Vec::new();
        let mut committed: Vec<u32> = Vec::new();
        let mut context: Vec<u32> = prompt_tokens.to_vec();
        let mut speculation = SpeculationStats::default();
        let rng = SamplerRng::new(options.seed);
        if options.output_constraint.is_some()
            && !matches!(options.speculation, Speculation::Disabled)
        {
            return Err(RuntimeError::ConstraintSpeculation);
        }
        let mut output_constraint = options
            .output_constraint
            .map(|_| crate::constraint::JsonObjectConstraint::new());
        // Greedy decoding without speculation keeps the device argmax, so the
        // v0.1 decode receipt still measures the same work.
        let host_sampling = options.sampler != Sampler::greedy()
            || options.penalties.is_active()
            || options.mirostat.is_some()
            || output_constraint.is_some();
        let mut mirostat = match (options.mirostat, session.mirostat.take()) {
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
        };
        let first = if host_sampling {
            // The captured decode graph contains an argmax, and its scratch is
            // allocated on first use. Allocating inside a capture is illegal,
            // so the scratch has to exist before capture begins.
            self.enqueue_sample(&mut activations)?;
            let position = (state.position - 1) as u64;
            let (token, _) = self.sample_on_host(
                &mut activations,
                &options.sampler,
                rng,
                position,
                &mut row,
                &options.penalties,
                &context,
                mirostat.as_mut(),
                output_constraint.as_mut(),
            )?;
            self.backend.write_u32(&mut activations.sampled, &[token])?;
            token
        } else {
            self.sample(
                &mut activations,
                options.logit_capture,
                &mut logits,
                state.position - 1,
            )?
        };
        let ttft_duration = prefill_start.elapsed();
        emit(&self.model, first, &mut tokens, &mut on_token)?;
        let eos = self.model.tokenizer.eos_token();
        let mut decode_duration = Duration::ZERO;
        let mut verify_duration = Duration::ZERO;
        let mut decode_evaluations = 0;
        let mut was_cancelled = false;
        let profile_target = match options.decode_profile {
            DecodeProfileMode::Disabled => 0,
            DecodeProfileMode::Steps(steps) => steps
                .get()
                .min(options.max_tokens.saturating_sub(tokens.len())),
        };
        let mut profile_start = None;
        let mut profile_steps = 0;
        let mut decode_profile = None;
        if profile_target > 0 && Some(first_token(&tokens)) != eos {
            let operations_per_step = self
                .model
                .config
                .n_layer
                .checked_mul(18)
                .and_then(|operations| operations.checked_add(4))
                .ok_or(RuntimeError::SizeOverflow)?;
            let operations = operations_per_step
                .checked_mul(profile_target)
                .ok_or(RuntimeError::SizeOverflow)?;
            self.backend.begin_decode_profile(operations)?;
            profile_start = Some(Instant::now());
        }

        // A drafted round runs outside the one-position graph. A round with no
        // proposal can still replay it after refreshing the device position.
        let use_graph = options.decode_execution == DecodeExecution::Graph
            && profile_target == 0
            && self.backend.decode_graph_supported()
            && Some(first_token(&tokens)) != eos;
        if use_graph {
            self.capture_decode_graph(&mut state, &mut activations, attention_shape)?;
        }

        let mut adaptive_controller = match options.speculation {
            Speculation::Adaptive(_) => {
                Some(session.adaptive_controller.take().unwrap_or_else(|| {
                    crate::adaptive_draft::AdaptiveController::new(
                        crate::adaptive_draft::AdaptiveControllerConfig::default(),
                    )
                }))
            }
            _ => None,
        };
        let mut correctable_controller = match options.speculation {
            Speculation::Correctable(_) => {
                Some(session.correctable_controller.take().unwrap_or_else(|| {
                    CorrectableController::new(CorrectableControllerConfig::default())
                }))
            }
            _ => None,
        };

        while tokens.len() < options.max_tokens && Some(first_token(&tokens)) != eos {
            if cancelled() {
                was_cancelled = true;
                break;
            }
            let decode_start = Instant::now();
            committed.clear();
            let mut adaptive_round = None;
            let mut correctable_round = None;
            let mut correctable_distributions = None;
            let mut correctable_proposed = 0;
            let mut correctable_accepted = 0;
            let mut correctable_overlap = 0.0;
            let drafted = match &options.speculation {
                Speculation::Disabled => Draft::Nothing,
                Speculation::Suffix(drafter) => {
                    context.truncate(prompt_tokens.len());
                    context.extend_from_slice(&tokens);
                    drafter.draft(&context)
                }
                Speculation::Adaptive(drafter) => {
                    let controller_start = Instant::now();
                    context.truncate(prompt_tokens.len());
                    context.extend_from_slice(&tokens);
                    let mut proposal = drafter.propose(&context);
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
                    match decision {
                        crate::adaptive_draft::AdaptiveDecision::Plain { .. } => {
                            adaptive_round = Some((
                                std::num::NonZeroUsize::new(1).expect("one is nonzero"),
                                controller_duration,
                            ));
                            Draft::Nothing
                        }
                        crate::adaptive_draft::AdaptiveDecision::Speculate {
                            verifier_positions,
                            ..
                        } => {
                            let mut tokens = proposal
                                .take()
                                .expect("speculation requires a proposal")
                                .tokens;
                            tokens.truncate(verifier_positions.get() - 1);
                            adaptive_round = Some((verifier_positions, controller_duration));
                            Draft::Tokens(tokens)
                        }
                    }
                }
                Speculation::Correctable(drafter) => {
                    let controller_start = Instant::now();
                    context.truncate(prompt_tokens.len());
                    context.extend_from_slice(&tokens);
                    let remaining = options.max_tokens.saturating_sub(tokens.len());
                    let maximum_positions = remaining.min(drafter.maximum_verifier_positions());
                    let available = if maximum_positions >= 2 {
                        drafter.available_plans(&context)
                    } else {
                        [false; 4]
                    };
                    let controller = correctable_controller
                        .as_mut()
                        .expect("correctable speculation creates a controller");
                    let decision = controller.decide(available, maximum_positions.max(1))?;
                    let controller_duration = controller_start.elapsed();
                    match decision {
                        crate::CorrectableDecision::Plain { .. } => {
                            correctable_round = Some((
                                None,
                                NonZeroUsize::new(1).expect("one is nonzero"),
                                controller_duration,
                            ));
                            Draft::Nothing
                        }
                        crate::CorrectableDecision::Speculate {
                            plan,
                            verifier_positions,
                            ..
                        } => {
                            let proposal = drafter.propose(
                                plan,
                                &context,
                                verifier_positions.get() - 1,
                                rng,
                                state.position as u64,
                            )?;
                            if let Some(proposal) = proposal {
                                let tokens = proposal.tokens();
                                correctable_distributions = Some(proposal.distributions());
                                correctable_round = Some((
                                    Some(proposal.plan),
                                    NonZeroUsize::new(tokens.len() + 1)
                                        .expect("a proposal has one or more tokens"),
                                    controller_duration,
                                ));
                                Draft::Tokens(tokens)
                            } else {
                                correctable_round = Some((
                                    None,
                                    NonZeroUsize::new(1).expect("one is nonzero"),
                                    controller_duration,
                                ));
                                Draft::Nothing
                            }
                        }
                    }
                }
            };
            if drafted.is_empty() {
                if use_graph {
                    let device_position = u32::try_from(state.position).map_err(|_| {
                        RuntimeError::ContextCapacity {
                            requested: state.position,
                            capacity: u32::MAX as usize,
                        }
                    })?;
                    self.backend
                        .write_u32(&mut state.device_position, &[device_position])?;
                    self.backend.replay_decode_graph()?;
                    state.position = state
                        .position
                        .checked_add(1)
                        .ok_or(RuntimeError::SizeOverflow)?;
                } else {
                    self.forward(&mut state, &mut activations, attention_shape)?;
                    if host_sampling {
                        // Host sampling needs the whole row, so skip the
                        // device argmax that greedy decoding uses.
                    } else {
                        self.enqueue_sample(&mut activations)?;
                    }
                }
                if host_sampling {
                    let position = (state.position - 1) as u64;
                    context.truncate(prompt_tokens.len());
                    context.extend_from_slice(&tokens);
                    let (token, _) = self.sample_on_host(
                        &mut activations,
                        &options.sampler,
                        rng,
                        position,
                        &mut row,
                        &options.penalties,
                        &context,
                        mirostat.as_mut(),
                        output_constraint.as_mut(),
                    )?;
                    self.backend.write_u32(&mut activations.sampled, &[token])?;
                    committed.push(token);
                } else {
                    committed.push(self.read_sample(
                        &mut activations,
                        options.logit_capture,
                        &mut logits,
                        state.position - 1,
                    )?);
                }
                decode_evaluations += 1;
            } else {
                let verifier_positions = drafted.tokens().len() + 1;
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
                let outcome = self.speculative_round(
                    &mut state,
                    &mut activations,
                    verify_activations
                        .get_mut(verifier_positions)
                        .and_then(Option::as_mut),
                    attention_shape,
                    drafted.tokens(),
                    correctable_distributions.as_deref(),
                    &options,
                    rng,
                    &mut row,
                    &mut context,
                    &mut committed,
                )?;
                decode_evaluations += outcome.evaluations;
                speculation.proposed += outcome.proposed;
                speculation.accepted += outcome.accepted;
                speculation.rounds += 1;
                speculation.draft_width = speculation.draft_width.max(outcome.proposed);
                speculation.correctable_overlap_sum += outcome.overlap_sum;
                speculation.correctable_overlap_proposals += outcome.overlap_proposals;
                correctable_proposed = outcome.proposed;
                correctable_accepted = outcome.accepted;
                correctable_overlap = outcome.overlap_sum;
                if outcome.verified_positions > 0 {
                    speculation.verify_passes += 1;
                    speculation.verified_positions += outcome.verified_positions;
                    verify_duration += outcome.verify_duration;
                }
            }
            let round_duration = decode_start.elapsed();
            decode_duration += round_duration;
            if let (Some((verifier_positions, controller_duration)), Some(emitted_tokens)) =
                (adaptive_round, std::num::NonZeroUsize::new(committed.len()))
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
            if let (Some((plan, verifier_positions, controller_duration)), Some(emitted_tokens)) =
                (correctable_round, NonZeroUsize::new(committed.len()))
            {
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
                                .map_or(0, |_| correctable_proposed.min(committed.len()))
                        } else {
                            0
                        },
                        wall_duration: round_duration,
                        controller_duration,
                    })?;
            }
            if let Some(start) = profile_start {
                profile_steps += 1;
                if profile_steps == profile_target {
                    decode_profile = self
                        .backend
                        .end_decode_profile(profile_steps, start.elapsed())?;
                    profile_start = None;
                }
            }
            for token in committed.drain(..) {
                emit(&self.model, token, &mut tokens, &mut on_token)?;
                if tokens.len() >= options.max_tokens || Some(token) == eos {
                    break;
                }
            }
        }
        if let Some(start) = profile_start {
            decode_profile = self
                .backend
                .end_decode_profile(profile_steps, start.elapsed())?;
        }
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
        if was_cancelled {
            session.invalidate();
        } else {
            let mut evaluated_tokens = prompt_tokens.to_vec();
            evaluated_tokens.extend_from_slice(&tokens);
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
        let mut workspace = self.backend.prepare_prefill(plan)?;
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
        let activation_bytes = PrefillActivations::<B>::bytes(&self.model.config, chunk_tokens)?
            .checked_add(if remainder == 0 {
                0
            } else {
                PrefillActivations::<B>::bytes(&self.model.config, remainder)?
            })
            .ok_or(RuntimeError::SizeOverflow)?;
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
            PreparedPrefill::Sequential => {
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
            PreparedPrefill::Chunked {
                chunk_tokens,
                full,
                tail,
                workspace,
            } => {
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
                    let count = (*chunk_tokens).min(prompt_tokens.len() - processed);
                    let batch = if count == *chunk_tokens {
                        &mut *full
                    } else {
                        tail.as_mut().ok_or_else(|| {
                            BackendError::operation(
                                "select prefill tail",
                                "tail activation storage is missing",
                            )
                        })?
                    };
                    self.forward_prefill_chunk(
                        &prompt_tokens[processed..processed + count],
                        base_position
                            .checked_add(processed)
                            .ok_or(RuntimeError::SizeOverflow)?,
                        &mut state.layers,
                        batch,
                        attention_shape,
                    )?;
                    processed = processed
                        .checked_add(count)
                        .ok_or(RuntimeError::SizeOverflow)?;
                    state.position = base_position
                        .checked_add(processed)
                        .ok_or(RuntimeError::SizeOverflow)?;
                    if processed == prompt_tokens.len() {
                        self.backend.copy_f32_row(
                            &batch.hidden,
                            count - 1,
                            self.model.config.n_embd,
                            &mut activations.hidden,
                        )?;
                    }
                }
                self.finish_prefill_logits(activations)?;
                Ok(PrefillExecution {
                    processed_tokens: processed,
                    cancelled: false,
                    workspace: *workspace,
                })
            }
        }
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
        let config = &self.model.config;
        self.backend.write_u32(&mut activations.tokens, tokens)?;
        self.backend.embed_gather_batch(
            &self.model.weights.token_embedding.buffer,
            &activations.tokens,
            &mut activations.hidden,
            self.model.weights.token_embedding.shape,
            tokens.len(),
        )?;
        let hidden_shape = VectorShape::new(tokens.len(), config.n_embd)?;
        let query_rows = tokens
            .len()
            .checked_mul(config.n_head)
            .ok_or(RuntimeError::SizeOverflow)?;
        let key_rows = tokens
            .len()
            .checked_mul(config.n_head_kv)
            .ok_or(RuntimeError::SizeOverflow)?;
        let query_shape = VectorShape::new(query_rows, config.head_dim)?;
        let key_shape = VectorShape::new(key_rows, config.head_dim)?;
        let query_rope = RopeShape::new(tokens.len(), config.n_head, config.head_dim)?;
        let key_rope = RopeShape::new(tokens.len(), config.n_head_kv, config.head_dim)?;

        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            self.backend.prefill_rms_norm(
                &activations.hidden,
                &layer.attention_norm,
                &mut activations.norm,
                hidden_shape,
                config.rms_epsilon,
            )?;
            self.backend.prefill_gemm(
                &layer.query.buffer,
                &activations.norm,
                &mut activations.query,
                layer.query.shape,
                tokens.len(),
            )?;
            self.backend.prefill_gemm(
                &layer.key.buffer,
                &activations.norm,
                &mut activations.key,
                layer.key.shape,
                tokens.len(),
            )?;
            self.backend.prefill_gemm(
                &layer.value.buffer,
                &activations.norm,
                &mut activations.value,
                layer.value.shape,
                tokens.len(),
            )?;
            let (query, key) = match &layer.qk_norm {
                QkNorm::Rms { query, key } => {
                    self.backend.prefill_rms_norm(
                        &activations.query,
                        query,
                        &mut activations.query_norm,
                        query_shape,
                        config.rms_epsilon,
                    )?;
                    self.backend.rope(
                        &mut activations.query_norm,
                        start_position,
                        query_rope,
                        config.rope_theta,
                    )?;
                    self.backend.prefill_rms_norm(
                        &activations.key,
                        key,
                        &mut activations.key_norm,
                        key_shape,
                        config.rms_epsilon,
                    )?;
                    self.backend.rope(
                        &mut activations.key_norm,
                        start_position,
                        key_rope,
                        config.rope_theta,
                    )?;
                    (&activations.query_norm, &activations.key_norm)
                }
                QkNorm::Identity => {
                    self.backend.rope(
                        &mut activations.query,
                        start_position,
                        query_rope,
                        config.rope_theta,
                    )?;
                    self.backend.rope(
                        &mut activations.key,
                        start_position,
                        key_rope,
                        config.rope_theta,
                    )?;
                    (&activations.query, &activations.key)
                }
            };
            self.backend.kv_append_chunk(
                key,
                &activations.value,
                &mut layers[layer_index].key,
                &mut layers[layer_index].value,
                attention_shape,
                start_position,
                tokens.len(),
            )?;
            self.backend.attention_prefill(
                query,
                &layers[layer_index].key,
                &layers[layer_index].value,
                &mut activations.attention,
                attention_shape,
                start_position,
                tokens.len(),
            )?;
            self.backend.prefill_gemm(
                &layer.attention_output.buffer,
                &activations.attention,
                &mut activations.residual,
                layer.attention_output.shape,
                tokens.len(),
            )?;
            self.backend.residual_add(
                &activations.hidden,
                &activations.residual,
                &mut activations.attention,
            )?;
            self.backend.prefill_rms_norm(
                &activations.attention,
                &layer.ffn_norm,
                &mut activations.norm,
                hidden_shape,
                config.rms_epsilon,
            )?;
            self.backend.prefill_gemm(
                &layer.ffn_gate.buffer,
                &activations.norm,
                &mut activations.gate,
                layer.ffn_gate.shape,
                tokens.len(),
            )?;
            self.backend.prefill_gemm(
                &layer.ffn_up.buffer,
                &activations.norm,
                &mut activations.up,
                layer.ffn_up.shape,
                tokens.len(),
            )?;
            self.backend
                .swiglu(&activations.gate, &activations.up, &mut activations.ffn)?;
            self.backend.prefill_gemm(
                &layer.ffn_down.buffer,
                &activations.ffn,
                &mut activations.residual,
                layer.ffn_down.shape,
                tokens.len(),
            )?;
            self.backend.residual_add(
                &activations.attention,
                &activations.residual,
                &mut activations.hidden,
            )?;
        }
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
        let config = &self.model.config;
        self.backend.profile_decode_op(DecodeOp::Embed)?;
        self.backend.embed_gather(
            &self.model.weights.token_embedding.buffer,
            &activations.sampled,
            &mut activations.hidden,
            self.model.weights.token_embedding.shape,
        )?;
        let hidden_shape = VectorShape::new(1, config.n_embd)?;
        let query_shape = VectorShape::new(config.n_head, config.head_dim)?;
        let key_shape = VectorShape::new(config.n_head_kv, config.head_dim)?;
        self.backend.prepare_rope(position)?;

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
                config.rms_epsilon,
                config.rope_theta,
            )?;
        }
        self.backend.profile_decode_op(DecodeOp::Norm)?;
        self.backend.rms_norm(
            &activations.hidden,
            &self.model.weights.output_norm,
            &mut activations.norm,
            hidden_shape,
            config.rms_epsilon,
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
        let positions = tokens.len();
        if positions != activations.positions {
            return Err(BackendError::SizeMismatch {
                name: "verifier positions",
                expected: activations.positions,
                actual: positions,
            }
            .into());
        }
        let config = &self.model.config;
        let start_position = state.position;
        self.backend.write_u32(&mut activations.tokens, tokens)?;
        self.backend.embed_gather_batch(
            &self.model.weights.token_embedding.buffer,
            &activations.tokens,
            &mut activations.hidden,
            self.model.weights.token_embedding.shape,
            positions,
        )?;
        let hidden_shape = VectorShape::new(positions, config.n_embd)?;
        let query_shape = VectorShape::new(config.n_head, config.head_dim)?;
        let key_shape = VectorShape::new(config.n_head_kv, config.head_dim)?;
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            self.backend.prefill_rms_norm(
                &activations.hidden,
                &layer.attention_norm,
                &mut activations.norm,
                hidden_shape,
                config.rms_epsilon,
            )?;
            self.backend.verify_gemv_triple(
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
            let KvLayer {
                key: key_cache,
                value: value_cache,
                ..
            } = &mut state.layers[layer_index];
            let query = match &layer.qk_norm {
                QkNorm::Rms { query, key } => {
                    self.backend.verify_qk_norm_rope_kv_append(
                        &activations.query,
                        query,
                        &mut activations.query_norm,
                        query_shape,
                        &activations.key,
                        key,
                        &mut activations.key_norm,
                        key_shape,
                        &activations.value,
                        key_cache,
                        value_cache,
                        attention_shape,
                        start_position,
                        positions,
                        config.rms_epsilon,
                        config.rope_theta,
                    )?;
                    &activations.query_norm
                }
                QkNorm::Identity => {
                    self.backend.rope(
                        &mut activations.query,
                        start_position,
                        RopeShape::new(positions, config.n_head, config.head_dim)?,
                        config.rope_theta,
                    )?;
                    self.backend.rope(
                        &mut activations.key,
                        start_position,
                        RopeShape::new(positions, config.n_head_kv, config.head_dim)?,
                        config.rope_theta,
                    )?;
                    self.backend.kv_append_chunk(
                        &activations.key,
                        &activations.value,
                        key_cache,
                        value_cache,
                        attention_shape,
                        start_position,
                        positions,
                    )?;
                    &activations.query
                }
            };
            self.backend.verify_attention(
                query,
                &state.layers[layer_index].key,
                &state.layers[layer_index].value,
                &mut activations.attention,
                attention_shape,
                start_position,
                positions,
            )?;
            self.backend.verify_gemv_residual_prepared(
                &layer.attention_output.buffer,
                &activations.attention,
                &activations.hidden,
                &mut activations.residual,
                layer.attention_output.shape,
                positions,
            )?;
            self.backend.prefill_rms_norm(
                &activations.residual,
                &layer.ffn_norm,
                &mut activations.norm,
                hidden_shape,
                config.rms_epsilon,
            )?;
            self.backend.verify_gemv_pair(
                &layer.ffn_gate.buffer,
                &layer.ffn_up.buffer,
                &activations.norm,
                &mut activations.gate,
                &mut activations.up,
                layer.ffn_gate.shape,
                layer.ffn_up.shape,
                positions,
            )?;
            self.backend.verify_swiglu(
                &activations.gate,
                &activations.up,
                &mut activations.ffn,
                config.n_ff,
                positions,
            )?;
            self.backend.verify_gemv_residual_prepared(
                &layer.ffn_down.buffer,
                &activations.ffn,
                &activations.residual,
                &mut activations.hidden,
                layer.ffn_down.shape,
                positions,
            )?;
        }
        self.backend.prefill_rms_norm(
            &activations.hidden,
            &self.model.weights.output_norm,
            &mut activations.norm,
            hidden_shape,
            config.rms_epsilon,
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
        state.position = state
            .position
            .checked_add(positions)
            .ok_or(RuntimeError::SizeOverflow)?;
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
        let query = match &layer.qk_norm {
            QkNorm::Rms { query, key } => {
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
                &activations.query_norm
            }
            QkNorm::Identity => {
                backend.profile_decode_op(DecodeOp::Rope)?;
                let position = match position {
                    Position::Host(position) => position,
                    Position::Device(_) => {
                        return Err(RuntimeError::UnsupportedDecodeGraph {
                            architecture: "llama",
                        });
                    }
                };
                backend.rope(
                    &mut activations.query,
                    position,
                    RopeShape::new(1, query_shape.rows(), query_shape.columns())?,
                    theta,
                )?;
                backend.rope(
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
                    Position::Host(position),
                )?;
                &activations.query
            }
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
        mut constraint: Option<&mut crate::constraint::JsonObjectConstraint>,
    ) -> Result<(u32, Distribution), RuntimeError> {
        self.read_logit_row(activations, row, penalties, context)?;
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
        if let Some(distributions) = draft_distributions {
            if distributions.len() != drafted.len() {
                return Err(RuntimeError::CorrectableDistributionCount {
                    proposed: drafted.len(),
                    distributions: distributions.len(),
                });
            }
        }
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
        let vocab = self.model.config.vocab_size;
        let mut accepted = 0;
        let mut overlap_sum = 0.0;
        let mut overlap_proposals = 0;
        for (index, proposal) in drafted.iter().copied().enumerate() {
            self.forward(state, activations, attention_shape)?;
            let position = (state.position - 1) as u64;
            self.read_logit_row(activations, row, &options.penalties, context)?;
            // Every committed token joins the history before the next
            // position is scored, so penalties see the round as it grows.
            let target = distribution(row, &options.sampler)?;
            let draft = match draft_distributions {
                Some(distributions) => distributions[index].clone(),
                None => point_mass(vocab, proposal)?,
            };
            if draft_distributions.is_some() {
                overlap_sum += 1.0 - crate::total_variation(&target, &draft)?;
                overlap_proposals += 1;
            }
            let token = match verify(&target, &draft, proposal, rng, position)? {
                Verdict::Accept => {
                    accepted += 1;
                    proposal
                }
                Verdict::Reject { token } => {
                    committed.push(token);
                    context.push(token);
                    self.backend.write_u32(&mut activations.sampled, &[token])?;
                    return Ok(SpeculationOutcome {
                        proposed: drafted.len(),
                        accepted,
                        evaluations: index + 1,
                        verified_positions: 0,
                        verify_duration: Duration::ZERO,
                        overlap_sum,
                        overlap_proposals,
                    });
                }
            };
            committed.push(token);
            context.push(token);
            self.backend.write_u32(&mut activations.sampled, &[token])?;
        }
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
            proposed: drafted.len(),
            accepted,
            evaluations: drafted.len() + 1,
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
        let base_position = state.position;
        let current = context.last().copied().ok_or(RuntimeError::EmptyPrompt)?;
        verify_activations.input_tokens.clear();
        verify_activations.input_tokens.push(current);
        verify_activations.input_tokens.extend_from_slice(drafted);
        let started = Instant::now();
        self.forward_verify(
            state,
            verify_activations,
            attention_shape,
            &verify_activations.input_tokens.clone(),
        )?;
        self.backend.read_f32(
            &verify_activations.logits,
            &mut verify_activations.host_logits,
        )?;
        let verify_duration = started.elapsed();
        let vocab = self.model.config.vocab_size;
        let mut accepted = 0;
        let mut overlap_sum = 0.0;
        let mut overlap_proposals = 0;
        for (index, proposal) in drafted.iter().copied().enumerate() {
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
            if draft_distributions.is_some() {
                overlap_sum += 1.0 - crate::total_variation(&target, &draft)?;
                overlap_proposals += 1;
            }
            let position = (base_position + index) as u64;
            match verify(&target, &draft, proposal, rng, position)? {
                Verdict::Accept => {
                    accepted += 1;
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
                    return Ok(SpeculationOutcome {
                        proposed: drafted.len(),
                        accepted,
                        evaluations: 1,
                        verified_positions: verify_activations.positions,
                        verify_duration,
                        overlap_sum,
                        overlap_proposals,
                    });
                }
            }
        }
        let start = drafted
            .len()
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
        let position = (base_position + drafted.len()) as u64;
        let token = select(&target, rng, position)?;
        committed.push(token);
        context.push(token);
        self.backend.write_u32(&mut activations.sampled, &[token])?;
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
        let mut next = || {
            let source = buffers.next().ok_or_else(|| {
                BackendError::operation("restore activations", "snapshot buffer is missing")
            })?;
            backend.restore_buffer(source)
        };
        let activations = Activations {
            hidden: next()?,
            norm: next()?,
            query: next()?,
            key: next()?,
            value: next()?,
            query_norm: next()?,
            key_norm: next()?,
            attention: next()?,
            residual: next()?,
            gate: next()?,
            up: next()?,
            ffn: next()?,
            logits: next()?,
            sampled: next()?,
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

impl<B: Backend> VerifyActivations<B> {
    fn new(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
    ) -> Result<Self, BackendError> {
        let dense = |columns: usize, field: &'static str| {
            positions
                .checked_mul(columns)
                .ok_or(BackendError::SizeOverflow { field })
                .and_then(BufferLayout::f32)
        };
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
        Ok(Self {
            positions,
            tokens: backend.allocate(BufferLayout::u32(positions)?)?,
            hidden: backend.allocate(dense(config.n_embd, "verifier hidden elements")?)?,
            norm: backend.allocate(dense(config.n_embd, "verifier norm elements")?)?,
            query: backend.allocate(dense(config.n_embd, "verifier query elements")?)?,
            key: backend.allocate(dense(kv_columns, "verifier key elements")?)?,
            value: backend.allocate(dense(kv_columns, "verifier value elements")?)?,
            query_norm: backend.allocate(dense(config.n_embd, "verifier query norm elements")?)?,
            key_norm: backend.allocate(dense(kv_columns, "verifier key norm elements")?)?,
            attention: backend.allocate(dense(config.n_embd, "verifier attention elements")?)?,
            residual: backend.allocate(dense(config.n_embd, "verifier residual elements")?)?,
            gate: backend.allocate(dense(config.n_ff, "verifier gate elements")?)?,
            up: backend.allocate(dense(config.n_ff, "verifier up elements")?)?,
            ffn: backend.allocate(dense(config.n_ff, "verifier FFN elements")?)?,
            logits: backend.allocate(BufferLayout::f32(logit_elements)?)?,
            input_tokens: Vec::with_capacity(positions),
            host_logits: vec![0.0; logit_elements],
        })
    }
}

impl<B: Backend> Activations<B> {
    fn new(backend: &mut B, config: &crate::ModelConfig) -> Result<Self, BackendError> {
        Ok(Self {
            hidden: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            norm: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            query: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            key: backend.allocate(BufferLayout::f32(config.n_head_kv * config.head_dim)?)?,
            value: backend.allocate(BufferLayout::f32(config.n_head_kv * config.head_dim)?)?,
            query_norm: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            key_norm: backend.allocate(BufferLayout::f32(config.n_head_kv * config.head_dim)?)?,
            attention: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            residual: backend.allocate(BufferLayout::f32(config.n_embd)?)?,
            gate: backend.allocate(BufferLayout::f32(config.n_ff)?)?,
            up: backend.allocate(BufferLayout::f32(config.n_ff)?)?,
            ffn: backend.allocate(BufferLayout::f32(config.n_ff)?)?,
            logits: backend.allocate(BufferLayout::f32(config.vocab_size)?)?,
            sampled: backend.allocate(BufferLayout::u32(1)?)?,
        })
    }

    fn fork(backend: &mut B, source: &Self) -> Result<Self, BackendError> {
        Ok(Self {
            hidden: backend.clone_buffer(&source.hidden)?,
            norm: backend.clone_buffer(&source.norm)?,
            query: backend.clone_buffer(&source.query)?,
            key: backend.clone_buffer(&source.key)?,
            value: backend.clone_buffer(&source.value)?,
            query_norm: backend.clone_buffer(&source.query_norm)?,
            key_norm: backend.clone_buffer(&source.key_norm)?,
            attention: backend.clone_buffer(&source.attention)?,
            residual: backend.clone_buffer(&source.residual)?,
            gate: backend.clone_buffer(&source.gate)?,
            up: backend.clone_buffer(&source.up)?,
            ffn: backend.clone_buffer(&source.ffn)?,
            logits: backend.clone_buffer(&source.logits)?,
            sampled: backend.clone_buffer(&source.sampled)?,
        })
    }

    fn bytes(config: &crate::ModelConfig) -> Result<u64, RuntimeError> {
        let kv_columns = config
            .n_head_kv
            .checked_mul(config.head_dim)
            .ok_or(RuntimeError::SizeOverflow)?;
        let elements = config
            .n_embd
            .checked_mul(6)
            .and_then(|value| value.checked_add(kv_columns.checked_mul(3)?))
            .and_then(|value| value.checked_add(config.n_ff.checked_mul(3)?))
            .and_then(|value| value.checked_add(config.vocab_size))
            .ok_or(RuntimeError::SizeOverflow)?;
        let bytes = elements
            .checked_mul(4)
            .and_then(|value| value.checked_add(4))
            .ok_or(RuntimeError::SizeOverflow)?;
        u64::try_from(bytes).map_err(|_| RuntimeError::SizeOverflow)
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
        let dense = |rows: usize, columns: usize| {
            rows.checked_mul(columns)
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill activation elements",
                })
                .and_then(BufferLayout::f32)
        };
        let kv_columns =
            config
                .n_head_kv
                .checked_mul(config.head_dim)
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill projected KV elements",
                })?;
        Ok(Self {
            tokens: backend.allocate(BufferLayout::u32(tokens)?)?,
            hidden: backend.allocate(dense(tokens, config.n_embd)?)?,
            norm: backend.allocate(dense(tokens, config.n_embd)?)?,
            query: backend.allocate(dense(tokens, config.n_embd)?)?,
            key: backend.allocate(dense(tokens, kv_columns)?)?,
            value: backend.allocate(dense(tokens, kv_columns)?)?,
            query_norm: backend.allocate(dense(tokens, config.n_embd)?)?,
            key_norm: backend.allocate(dense(tokens, kv_columns)?)?,
            attention: backend.allocate(dense(tokens, config.n_embd)?)?,
            residual: backend.allocate(dense(tokens, config.n_embd)?)?,
            gate: backend.allocate(dense(tokens, config.n_ff)?)?,
            up: backend.allocate(dense(tokens, config.n_ff)?)?,
            ffn: backend.allocate(dense(tokens, config.n_ff)?)?,
        })
    }

    fn bytes(config: &crate::ModelConfig, tokens: usize) -> Result<u64, RuntimeError> {
        let kv_columns = config
            .n_head_kv
            .checked_mul(config.head_dim)
            .ok_or(RuntimeError::SizeOverflow)?;
        let row_elements = config
            .n_embd
            .checked_mul(6)
            .and_then(|value| value.checked_add(kv_columns.checked_mul(3)?))
            .and_then(|value| value.checked_add(config.n_ff.checked_mul(3)?))
            .ok_or(RuntimeError::SizeOverflow)?;
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
        let cache_layout = self.dtype.layout(self.shape.cache_elements()?)?;
        let one_cache_bytes =
            u64::try_from(cache_layout.bytes()).map_err(|_| RuntimeError::SizeOverflow)?;
        let capacity = one_cache_bytes
            .checked_mul(2)
            .and_then(|bytes| {
                u64::try_from(self.layers.len())
                    .ok()
                    .and_then(|layers| bytes.checked_mul(layers))
            })
            .ok_or(RuntimeError::SizeOverflow)?;
        let mut allocator = StateAllocator::new(capacity);
        let mut layers = Vec::with_capacity(self.layers.len());
        for (key, value) in &self.layers {
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
        let mut allocator = StateAllocator::new(capacity);
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
        let cache_layout = source.dtype.layout(source.shape.cache_elements()?)?;
        let one_cache_bytes =
            u64::try_from(cache_layout.bytes()).map_err(|_| RuntimeError::SizeOverflow)?;
        let capacity = one_cache_bytes
            .checked_mul(2)
            .and_then(|bytes| {
                u64::try_from(source.layers.len())
                    .ok()
                    .and_then(|layers| bytes.checked_mul(layers))
            })
            .ok_or(RuntimeError::SizeOverflow)?;
        let mut allocator = StateAllocator::new(capacity);
        let mut layers = Vec::with_capacity(source.layers.len());
        for source_layer in &source.layers {
            let key_allocation =
                allocator.allocate(StateKind::Kv, one_cache_bytes, StateLifetime::Committed)?;
            let value_allocation =
                allocator.allocate(StateKind::Kv, one_cache_bytes, StateLifetime::Committed)?;
            layers.push(KvLayer {
                key: backend.clone_buffer(&source_layer.key)?,
                value: backend.clone_buffer(&source_layer.value)?,
                _key_allocation: key_allocation,
                _value_allocation: value_allocation,
            });
        }
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
