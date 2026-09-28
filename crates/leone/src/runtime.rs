use crate::kv::{KvAppendRange, KvGraphRevision, KvSnapshot, KvState};
use crate::model::{DenseLayer, OutputWeight, QkNorm};
use crate::{
    decode_graph_bucket, AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot,
    DecodeOp, DecodeProfile, LoadedModel, ModelLoadError, Position, StateError, TokenizerError,
    VectorShape,
};
use crate::{
    distribution, select, verify, Distribution, Draft, MirostatConfig, MirostatState, Penalties,
    Sampler, SamplerRng, SuffixDrafter, Verdict,
};
use crate::{AttentionDecodeRow, KvReadSpan, KvReadView};
use crate::{
    CorrectableController, CorrectableControllerConfig, CorrectableDrafter, CorrectableObservation,
};
use crate::{
    HostStaging, MemoryBudget, MemoryClass, PrefillMethod, PrefillNumerics, PrefillPlan,
    PrefillWorkspace, RopeShape,
};
use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(test)]
#[path = "runtime_batch_attention_tests.rs"]
mod batch_attention_tests;

#[cfg(test)]
#[path = "runtime_diagnostics_tests.rs"]
mod diagnostics_tests;

#[cfg(test)]
#[path = "runtime_prefill_routing_tests.rs"]
mod prefill_routing_tests;

/// The default number of prompt positions evaluated in one prefill block.
///
/// A larger block reduces launch and weight-dequantization work. The CUDA
/// backend bounds attention scratch with 1,024-token query tiles. Activation
/// storage grows with this block. A smaller `--prefill-chunk` lowers the
/// block when device memory is tight.
pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 4096;
const MIN_DECODE_KV_GROWTH_TOKENS: usize = 32;

/// Classifies whether a failed session operation preserved its source state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionFailureEffect {
    Unchanged,
    Quarantine,
}

impl fmt::Display for SessionFailureEffect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unchanged => formatter.write_str("unchanged"),
            Self::Quarantine => formatter.write_str("quarantine"),
        }
    }
}

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
    #[error("decode cannot run while prompt prefill is pending")]
    PrefillPending,
    #[error("the requested generation length must be nonzero")]
    ZeroGeneration,
    #[error("request cannot enter batched decode: {0}")]
    BatchUnavailable(&'static str),
    #[error("batched decode has {requested} rows, but the backend accepts at most {maximum}")]
    BatchSizeExceeded { requested: usize, maximum: usize },
    #[error("batched decode session does not match its transcript")]
    BatchSessionMismatch,
    #[error("{architecture} models do not support decode graphs; select eager decode")]
    UnsupportedDecodeGraph { architecture: &'static str },
    #[error("the request needs {requested} context positions but the model supports {capacity}")]
    ContextCapacity { requested: usize, capacity: usize },
    #[error("runtime byte accounting overflowed")]
    SizeOverflow,
    #[error("prefill advance budget {budget} is smaller than the next chunk {chunk}")]
    PrefillBudgetTooSmall { budget: usize, chunk: usize },
    #[error("prefill commit target session is not empty")]
    PrefillSessionBusy,
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
    #[error("the session has no completed raw logit row")]
    SessionLogitsUnavailable,
    #[error("correctable proposal has {proposed} tokens but {distributions} draft distributions")]
    CorrectableDistributionCount {
        proposed: usize,
        distributions: usize,
    },
    #[error("session failure effect is {effect}: {source}")]
    SessionFailure {
        effect: SessionFailureEffect,
        #[source]
        source: Box<RuntimeError>,
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

    /// Returns the session effect attached to this error.
    pub fn session_failure_effect(&self) -> Option<SessionFailureEffect> {
        match self {
            Self::SessionFailure { effect, .. } => Some(*effect),
            _ => None,
        }
    }

    fn with_session_failure_effect(self, effect: SessionFailureEffect) -> Self {
        if self.session_failure_effect().is_some() {
            self
        } else {
            Self::SessionFailure {
                effect,
                source: Box::new(self),
            }
        }
    }

    fn with_forced_session_failure_effect(self, effect: SessionFailureEffect) -> Self {
        Self::SessionFailure {
            effect,
            source: Box::new(self),
        }
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
    pub(crate) fn layout(self, elements: usize) -> Result<BufferLayout, BackendError> {
        match self {
            Self::Q8 => BufferLayout::q8_kv(elements),
            Self::F16 => BufferLayout::f16(elements),
            Self::F32 => BufferLayout::f32(elements),
        }
    }

    /// Returns the exact key and value storage reserved per context token.
    pub fn bytes_per_token(self, config: &crate::ModelConfig) -> Result<u64, RuntimeError> {
        let elements = config
            .n_head_kv
            .checked_mul(config.head_dim)
            .ok_or(RuntimeError::SizeOverflow)?;
        let bytes = u64::try_from(self.layout(elements)?.bytes())
            .map_err(|_| RuntimeError::SizeOverflow)?;
        bytes
            .checked_mul(2)
            .and_then(|value| {
                u64::try_from(config.n_layer)
                    .ok()
                    .and_then(|layers| value.checked_mul(layers))
            })
            .ok_or(RuntimeError::SizeOverflow)
    }
}

/// How a decode step proposes tokens before the model verifies them.
///
/// `Disabled` is plain decode. Every proposal
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

/// One retained session submitted for a single batched decode position.
pub struct BatchSession<'a, B: Backend> {
    pub session: &'a mut GenerationSession<B>,
    pub transcript: &'a [u32],
    pub options: &'a GenerateOptions,
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
    /// Method used by the primary prompt prefill segment.
    /// Restore replay tails remain sequential. `SessionReplay` records the replay and computed counts.
    /// `Reused` means retained state covered the prompt and no segment ran.
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

/// Describes how one generation call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationTermination {
    /// Generation reached its configured limit or an end token.
    Completed,
    /// The caller's stop predicate ended generation and retained the session.
    Stopped,
    /// The caller's cancellation predicate ended generation and invalidated the session.
    Cancelled,
}

/// Tokens, timings, and optional logit snapshots from one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    /// The tokenized prompt used as decode input.
    pub prompt_tokens: Vec<u32>,
    /// Tokens emitted by this call.
    pub tokens: Vec<u32>,
    /// The termination state that determines session retention.
    pub termination: GenerationTermination,
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

impl SessionReplay {
    fn for_prompt(
        reuse_class: SessionReuseClass,
        cached_tokens: usize,
        reused_tokens: usize,
        replay_prefix: usize,
        prompt_tokens: usize,
    ) -> Self {
        let replayed_tokens = if reuse_class == SessionReuseClass::RestoreReplay {
            replay_prefix
        } else {
            0
        };
        Self {
            reuse_class,
            cached_tokens,
            reused_tokens,
            replayed_tokens,
            computed_tokens: prompt_tokens
                .saturating_sub(reused_tokens)
                .saturating_sub(replayed_tokens),
        }
    }
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
    retained_logits_position: Option<usize>,
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

    pub(crate) fn into_parts(self) -> (Vec<u32>, usize, Option<MirostatState>) {
        (self.evaluated_tokens, self.prefill_boundary, self.mirostat)
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
    retained_logits_position: Option<usize>,
    prefill_boundary: usize,
    mirostat: Option<MirostatState>,
    adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
    correctable_controller: Option<CorrectableController>,
    last_replay: SessionReplay,
    pending_fork: Option<SessionFork>,
    last_fork: Option<SessionFork>,
    pending_wake: Option<SessionHibernation>,
    last_hibernation: Option<SessionHibernation>,
    batch_identity: Option<u64>,
}

impl<B: Backend> GenerationSession<B> {
    pub fn new() -> Self {
        Self {
            state: None,
            activations: None,
            evaluated_tokens: Vec::new(),
            restored_tokens: Vec::new(),
            retained_logits_position: None,
            prefill_boundary: 0,
            mirostat: None,
            adaptive_controller: None,
            correctable_controller: None,
            last_replay: SessionReplay::default(),
            pending_fork: None,
            last_fork: None,
            pending_wake: None,
            last_hibernation: None,
            batch_identity: None,
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

    /// Clears session-owned device allocations and replay state.
    ///
    /// A runtime caller uses [`Runtime::discard_session`] so graph-owned
    /// backend resources are retired with the session.
    pub fn invalidate(&mut self) {
        self.state = None;
        self.activations = None;
        self.evaluated_tokens.clear();
        self.restored_tokens.clear();
        self.retained_logits_position = None;
        self.prefill_boundary = 0;
        self.mirostat = None;
        self.adaptive_controller = None;
        self.correctable_controller = None;
        self.pending_fork = None;
        self.last_fork = None;
        self.pending_wake = None;
        self.last_hibernation = None;
        self.batch_identity = None;
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

/// Holds device state while prompt tokens advance in bounded chunks.
///
/// The associated generation session stays empty until [`Runtime::finish_prefill`]
/// commits a ready state. Dropping this value drops partial device state.
#[derive(Debug)]
pub struct PendingPrefill<B: Backend> {
    prompt_tokens: Vec<u32>,
    options: GenerateOptions,
    attention_shape: AttentionShape,
    state: KvState<B>,
    activations: Activations<B>,
    stage: PendingPrefillStage<B>,
    reuse_class: SessionReuseClass,
    reused_tokens: usize,
    cached_tokens: usize,
    restored_tokens: usize,
    restored_common: usize,
    common_tokens: usize,
    replay_prefill_boundary: usize,
    mirostat: Option<MirostatState>,
    adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
    correctable_controller: Option<CorrectableController>,
    workspace: PrefillWorkspace,
    last_fork: Option<SessionFork>,
    last_hibernation: Option<SessionHibernation>,
}

impl<B: Backend> PendingPrefill<B> {
    /// Returns the complete prompt owned by the pending operation.
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    /// Returns the generation options bound to the pending operation.
    pub const fn options(&self) -> &GenerateOptions {
        &self.options
    }

    /// Returns the number of prompt tokens evaluated by this operation.
    pub fn processed_tokens(&self) -> usize {
        self.state.position.saturating_sub(self.reused_tokens)
    }

    /// Returns the number of prompt tokens that remain to be evaluated.
    pub fn remaining_tokens(&self) -> usize {
        self.prompt_tokens.len().saturating_sub(self.state.position)
    }

    /// Returns the reuse class selected before prefill started.
    pub const fn reuse_class(&self) -> SessionReuseClass {
        self.reuse_class
    }

    /// Returns the minimum budget for the next fixed chunk or sequential token.
    pub fn minimum_budget(&self) -> NonZeroUsize {
        let (remaining, prepared) = match &self.stage {
            PendingPrefillStage::Single {
                start,
                end,
                processed,
                prepared,
            }
            | PendingPrefillStage::Continuation {
                start,
                end,
                processed,
                prepared,
            } => (end - start - processed, Some(prepared)),
            PendingPrefillStage::Replay {
                boundary,
                processed,
                prepared,
            } => (boundary - processed, Some(prepared)),
            PendingPrefillStage::Done => (0, None),
        };
        let count = match prepared {
            Some(PreparedPrefill::Chunked { chunk_tokens, .. }) => (*chunk_tokens).min(remaining),
            _ => 1,
        };
        NonZeroUsize::new(count).unwrap_or(NonZeroUsize::MIN)
    }

    /// Returns the planned prefill workspace.
    pub const fn workspace(&self) -> PrefillWorkspace {
        self.workspace
    }
}

/// Owns a complete prompt state that has not yet been committed to a session.
#[derive(Debug)]
pub struct ReadyPrefill<B: Backend> {
    prompt_tokens: Vec<u32>,
    options: GenerateOptions,
    state: KvState<B>,
    activations: Activations<B>,
    reuse_class: SessionReuseClass,
    reused_tokens: usize,
    cached_tokens: usize,
    restored_tokens: usize,
    restored_common: usize,
    common_tokens: usize,
    replay_prefill_boundary: usize,
    mirostat: Option<MirostatState>,
    adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
    correctable_controller: Option<CorrectableController>,
    workspace: PrefillWorkspace,
    last_fork: Option<SessionFork>,
    last_hibernation: Option<SessionHibernation>,
}

impl<B: Backend> ReadyPrefill<B> {
    /// Returns the complete prompt evaluated by this operation.
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    /// Returns the number of prompt tokens evaluated by this operation.
    pub fn processed_tokens(&self) -> usize {
        self.state.position.saturating_sub(self.reused_tokens)
    }

    /// Returns the generation options bound to the ready operation.
    pub const fn options(&self) -> &GenerateOptions {
        &self.options
    }

    /// Returns the reuse class selected before prefill started.
    pub const fn reuse_class(&self) -> SessionReuseClass {
        self.reuse_class
    }

    /// Returns the planned prefill workspace.
    pub const fn workspace(&self) -> PrefillWorkspace {
        self.workspace
    }
}

/// Records a cancelled prefill after partial device state was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelledPrefill {
    processed_tokens: usize,
    workspace: PrefillWorkspace,
}

impl CancelledPrefill {
    /// Returns the number of prompt tokens evaluated before cancellation.
    pub const fn processed_tokens(self) -> usize {
        self.processed_tokens
    }

    /// Returns the planned prefill workspace.
    pub const fn workspace(self) -> PrefillWorkspace {
        self.workspace
    }
}

/// Describes the typed result of one bounded prefill advance.
#[derive(Debug)]
pub enum PrefillProgress<B: Backend> {
    Pending(PendingPrefill<B>),
    Ready(ReadyPrefill<B>),
    Cancelled(CancelledPrefill),
}

#[derive(Debug)]
enum PendingPrefillStage<B: Backend> {
    Single {
        start: usize,
        end: usize,
        processed: usize,
        prepared: PreparedPrefill<B>,
    },
    Replay {
        boundary: usize,
        processed: usize,
        prepared: PreparedPrefill<B>,
    },
    Continuation {
        start: usize,
        end: usize,
        processed: usize,
        prepared: PreparedPrefill<B>,
    },
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillAdvanceStep {
    Continue,
    Pending,
    Ready,
    Cancelled,
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

/// Counts bitwise differences between two diagnostic outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillBitwiseComparison {
    /// Number of scalar values compared.
    pub compared: usize,
    /// Number of compared values with different bit patterns.
    pub mismatching: usize,
}

impl PrefillBitwiseComparison {
    const fn empty() -> Self {
        Self {
            compared: 0,
            mismatching: 0,
        }
    }

    fn add(&mut self, other: Self) -> Result<(), RuntimeError> {
        self.compared = self
            .compared
            .checked_add(other.compared)
            .ok_or(RuntimeError::SizeOverflow)?;
        self.mismatching = self
            .mismatching
            .checked_add(other.mismatching)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok(())
    }
}

/// Bitwise results from one retained-prefix warm prefill comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmPrefillCharacterization {
    /// Number of tokens in the shared cold prefix.
    pub prefix_tokens: usize,
    /// Number of tokens appended by warm prefill.
    pub suffix_tokens: usize,
    /// Requested chunk size for the subject prefill.
    pub chunk_tokens: usize,
    /// Number of greedy decode positions compared after the warm append.
    pub continuation_tokens: usize,
    /// Numerical contract selected for the subject prefill.
    pub numerics: PrefillNumerics,
    /// Method used by the sequential control.
    pub control_method: PrefillMethod,
    /// Prefill method selected for the subject prefill.
    pub subject_method: PrefillMethod,
    /// Number of complete subject chunks.
    pub subject_full_chunks: usize,
    /// Number of tokens in the subject tail chunk.
    pub subject_tail_tokens: usize,
    /// KV bits compared after the shared cold prefix.
    pub prefix_kv: PrefillBitwiseComparison,
    /// Full vocabulary logits compared after the shared cold prefix.
    pub prefix_logits: PrefillBitwiseComparison,
    /// KV bits compared after the warm suffix.
    pub suffix_kv: PrefillBitwiseComparison,
    /// KV bits compared after continued decode.
    pub continued_kv: PrefillBitwiseComparison,
    /// Full vocabulary logits compared for suffix rows.
    pub suffix_logits: PrefillBitwiseComparison,
    /// Full vocabulary logits compared for continued decode decisions.
    pub continuation_logits: PrefillBitwiseComparison,
    /// Greedy tokens selected by the sequential control.
    pub sequential_tokens: Vec<u32>,
    /// Greedy tokens selected by the decode-equivalent subject.
    pub subject_tokens: Vec<u32>,
}

struct WarmPrefillCompletion {
    suffix_kv: PrefillBitwiseComparison,
    continued_kv: PrefillBitwiseComparison,
    sequential_tokens: Vec<u32>,
    subject_tokens: Vec<u32>,
    continuation_logits: PrefillBitwiseComparison,
}

struct WarmPrefillStates<B: Backend> {
    sequential_state: KvState<B>,
    sequential_activations: Activations<B>,
    subject_state: KvState<B>,
    subject_activations: Activations<B>,
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
    speculative_logits: Option<SpeculativeLogitScratch<B>>,
    batch_activations: BTreeMap<usize, VerifyActivations<B>>,
    batch_graph_signature: Option<Vec<u64>>,
    decode_graph_generation: u64,
    next_batch_identity: u64,
}

#[derive(Debug)]
struct SpeculativeLogitScratch<B: Backend> {
    buffer: B::Buffer,
    rows: usize,
}

struct BatchAttentionRow<'a, T> {
    query: &'a T,
    spans: Vec<KvReadSpan<'a, T>>,
    output: &'a mut T,
    shape: AttentionShape,
    position: Position<'a, T>,
}

impl<T> BatchAttentionRow<'_, T> {
    fn descriptor(&mut self) -> Result<AttentionDecodeRow<'_, T>, BackendError> {
        Ok(AttentionDecodeRow {
            query: self.query,
            cache: KvReadView::new(&self.spans)?,
            output: self.output,
            shape: self.shape,
            position: self.position,
        })
    }
}

struct GenerationPreparation<B: Backend> {
    attention_shape: AttentionShape,
    state: KvState<B>,
    activations: Activations<B>,
    prepared: PreparedPrefill<B>,
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

type SpeculativeRound = (SpeculationOutcome, bool);

struct GenerationStep {
    evaluations: usize,
    outcome: Option<SpeculationOutcome>,
    sequential_logits: bool,
}

struct GenerationInitialization {
    termination: Option<GenerationTermination>,
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

type BatchGraphPreparation<B> = (
    Vec<KvAppendRange>,
    (VectorShape, VectorShape, VectorShape),
    Vec<<B as Backend>::Buffer>,
);

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
        let mut state = KvState::new(&mut self.backend, self.model.config.n_layer, shape, dtype)?;
        state.prepare_append(
            &mut self.backend,
            self.model.config.n_layer,
            shape.max_context(),
            shape.max_context(),
        )?;
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
    pub fn load(backend: B, path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        Self::load_with_host_staging(backend, path, &HostStaging::unlimited())
    }

    /// Loads a model while charging host staging through the supplied ledger.
    pub fn load_with_host_staging(
        mut backend: B,
        path: impl AsRef<Path>,
        staging: &HostStaging,
    ) -> Result<Self, RuntimeError> {
        let model = LoadedModel::load_with_host_staging(&mut backend, path, staging)?;
        Ok(Self {
            backend,
            model,
            speculative_logits: None,
            batch_activations: BTreeMap::new(),
            batch_graph_signature: None,
            decode_graph_generation: 0,
            next_batch_identity: 1,
        })
    }

    /// Creates a runtime from an already uploaded model.
    pub fn from_model(backend: B, model: LoadedModel<B>) -> Self {
        Self {
            backend,
            model,
            speculative_logits: None,
            batch_activations: BTreeMap::new(),
            batch_graph_signature: None,
            decode_graph_generation: 0,
            next_batch_identity: 1,
        }
    }

    pub const fn model(&self) -> &LoadedModel<B> {
        &self.model
    }

    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Waits for all previously submitted backend work to complete.
    ///
    /// A backend failure carries a quarantine effect for sessions with pending
    /// work. The caller supplies any required session cleanup.
    pub fn synchronize(&mut self) -> Result<(), RuntimeError> {
        self.backend.synchronize().map_err(|error| {
            RuntimeError::from(error).with_session_failure_effect(SessionFailureEffect::Quarantine)
        })
    }

    /// Reads raw retained logits and returns the predicted next-token position.
    ///
    /// `output` must contain exactly [`Self::vocab_size`] elements. The row
    /// precedes penalties and sampling. The returned position equals the
    /// session's evaluated token count. The method waits for pending backend
    /// work and allocates no output storage. Incomplete or invalidated sessions
    /// return [`RuntimeError::SessionLogitsUnavailable`].
    pub fn read_session_logits(
        &mut self,
        session: &GenerationSession<B>,
        output: &mut [f32],
    ) -> Result<usize, RuntimeError> {
        let (position, logits) = retained_session_logits(session)?;
        let layout = BufferLayout::f32(self.model.config.vocab_size)?;
        if output.len() != layout.elements() {
            return Err(BackendError::SizeMismatch {
                name: "session logit output",
                expected: layout.elements(),
                actual: output.len(),
            }
            .into());
        }
        self.synchronize()?;
        self.backend.read_f32(logits, output).map_err(|error| {
            RuntimeError::from(error).with_session_failure_effect(SessionFailureEffect::Quarantine)
        })?;
        Ok(position)
    }

    /// Drops one session and retires graph resources that pin its buffers.
    pub fn discard_session(
        &mut self,
        session: &mut GenerationSession<B>,
    ) -> Result<(), RuntimeError> {
        self.retire_decode_graph()?;
        session.invalidate();
        Ok(())
    }

    fn retire_decode_graph(&mut self) -> Result<(), RuntimeError> {
        let next_generation = self
            .decode_graph_generation
            .checked_add(1)
            .ok_or(RuntimeError::SizeOverflow)?;
        self.batch_graph_signature = None;
        self.decode_graph_generation = next_generation;
        self.backend.drop_decode_graph().map_err(Into::into)
    }

    fn reserve_decode_graph_generation(&mut self) -> Result<(), RuntimeError> {
        self.decode_graph_generation = self
            .decode_graph_generation
            .checked_add(1)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok(())
    }

    fn graph_revision_token(&self, state: &KvState<B>) -> KvGraphRevision {
        KvGraphRevision {
            layout: state.revision(),
            generation: self.decode_graph_generation,
        }
    }

    /// Sets the checked allocation budget for backend-owned buffers.
    pub fn set_memory_budget(&mut self, budget: MemoryBudget) -> Result<(), RuntimeError> {
        self.backend.set_memory_budget(budget)?;
        Ok(())
    }

    /// Copies one complete live session into an independent child.
    pub fn fork_session(
        &mut self,
        source: &GenerationSession<B>,
    ) -> Result<GenerationSession<B>, RuntimeError> {
        let (source_state, source_activations) = reusable_session_buffers(source)?;
        self.prepare_kv_state_reads(source_state, source_state.shape)?;
        let state = KvState::fork(&mut self.backend, source_state)?;
        let activations = Activations::fork(&mut self.backend, source_activations)?;
        self.backend.synchronize()?;
        let copied_bytes = 4_u64
            .checked_add(Activations::<B>::bytes(&self.model.config)?)
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
            evaluated_tokens: copy_session_tokens(&source.evaluated_tokens)?,
            restored_tokens: copy_session_tokens(&source.restored_tokens)?,
            retained_logits_position: source.retained_logits_position,
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
            batch_identity: None,
        })
    }

    /// Copies a live session and retains only its requested prefix.
    pub fn fork_session_prefix(
        &mut self,
        source: &GenerationSession<B>,
        prefix_tokens: usize,
    ) -> Result<GenerationSession<B>, RuntimeError> {
        if prefix_tokens == 0 {
            return Err(RuntimeError::EmptyPrompt);
        }
        if prefix_tokens > source.evaluated_tokens.len() {
            return Err(RuntimeError::ContextCapacity {
                requested: prefix_tokens,
                capacity: source.evaluated_tokens.len(),
            });
        }
        let position = u32::try_from(prefix_tokens).map_err(|_| RuntimeError::ContextCapacity {
            requested: prefix_tokens,
            capacity: u32::MAX as usize,
        })?;
        let mut child = self.fork_session(source)?;
        let state = child
            .state
            .as_mut()
            .ok_or(RuntimeError::SessionForkUnavailable)?;
        state.truncate(prefix_tokens)?;
        self.backend
            .write_u32(&mut state.device_position, &[position])?;
        child.evaluated_tokens.truncate(prefix_tokens);
        child.restored_tokens.truncate(prefix_tokens);
        child.retained_logits_position = None;
        child.prefill_boundary = child.prefill_boundary.min(prefix_tokens);
        let fork = child
            .last_fork
            .ok_or(RuntimeError::SessionForkUnavailable)?;
        let fork = SessionFork {
            cached_tokens: prefix_tokens,
            ..fork
        };
        child.pending_fork = Some(fork);
        child.last_fork = Some(fork);
        child.last_replay = SessionReplay {
            reuse_class: SessionReuseClass::DeviceFork,
            cached_tokens: prefix_tokens,
            reused_tokens: prefix_tokens,
            replayed_tokens: 0,
            computed_tokens: 0,
        };
        Ok(child)
    }

    /// Forks a resident prefix for a new request with fresh request control state.
    pub fn reuse_prefix_session(
        &mut self,
        source: &GenerationSession<B>,
    ) -> Result<GenerationSession<B>, RuntimeError> {
        let mut session = self.fork_session(source)?;
        session.mirostat = None;
        session.adaptive_controller = None;
        session.correctable_controller = None;
        session.batch_identity = None;
        Ok(session)
    }

    /// Returns the KV and activation payload bytes captured for hibernation.
    ///
    /// Token histories and container metadata require separate host reservations.
    pub fn hibernation_bytes(&self, source: &GenerationSession<B>) -> Result<u64, RuntimeError> {
        let (state, _) = reusable_session_buffers(source)?;
        state
            .snapshot_payload_bytes()?
            .checked_add(Activations::<B>::bytes(&self.model.config)?)
            .ok_or(RuntimeError::SizeOverflow)
    }

    /// Returns host metadata bytes needed while a live session is captured.
    pub fn hibernation_metadata_bytes(
        &self,
        source: &GenerationSession<B>,
    ) -> Result<u64, RuntimeError> {
        let (state, _) = reusable_session_buffers(source)?;
        let resident = state.container_bytes()?;
        let snapshot = state.snapshot_container_bytes()?;
        let activation_bytes =
            checked_capacity_bytes::<BufferSnapshot>(HibernatedActivations::BUFFER_COUNT)?;
        let token_bytes = hibernation_token_metadata(source)?;
        let controller_bytes = controller_metadata_bytes()?;
        checked_metadata_sum([
            resident,
            snapshot,
            activation_bytes,
            token_bytes,
            controller_bytes,
        ])
    }

    /// Returns host metadata bytes needed while a hibernated session wakes.
    pub fn wake_metadata_bytes(&self, source: &HibernatedSession) -> Result<u64, RuntimeError> {
        let kv_bytes = source.state.snapshot.restore_container_bytes::<B>()?;
        let activation_bytes =
            checked_capacity_bytes::<B::Buffer>(HibernatedActivations::BUFFER_COUNT)?;
        let evaluated_bytes = checked_capacity_bytes::<u32>(source.evaluated_tokens.capacity())?;
        let restored_bytes = checked_capacity_bytes::<u32>(source.restored_tokens.capacity())?;
        let controller_bytes = controller_metadata_bytes()?;
        kv_bytes
            .checked_add(activation_bytes)
            .and_then(|bytes| bytes.checked_add(evaluated_bytes))
            .and_then(|bytes| bytes.checked_add(restored_bytes))
            .and_then(|bytes| bytes.checked_add(controller_bytes))
            .ok_or(RuntimeError::SizeOverflow)
    }

    /// Returns a checked host metadata headroom bound for one resident session.
    pub fn resident_metadata_bound(&self, context_tokens: usize) -> Result<u64, RuntimeError> {
        let kv = KvState::<B>::metadata_bound_bytes(context_tokens, self.model.config.n_layer)?;
        let tokens = resident_token_metadata_bound(context_tokens)?;
        let activation = u64::try_from(std::mem::size_of::<Activations<B>>())
            .map_err(|_| RuntimeError::SizeOverflow)?;
        let controllers = resident_controller_metadata_bound()?;
        checked_metadata_sum([kv, tokens, activation, controllers])
    }

    /// Moves one complete live session into host-owned buffers.
    pub fn hibernate_session(
        &mut self,
        source: &mut GenerationSession<B>,
    ) -> Result<HibernatedSession, RuntimeError> {
        let (source_state, source_activations) = reusable_session_buffers(source)?;
        let state = HibernatedKvState::capture(&mut self.backend, source_state)?;
        let activations = HibernatedActivations::capture(&mut self.backend, source_activations)?;
        let host_bytes = KvState::<B>::snapshot_bytes(&state.snapshot)?
            .checked_add(Activations::<B>::bytes(&self.model.config)?)
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
            evaluated_tokens: copy_session_tokens(&source.evaluated_tokens)?,
            restored_tokens: copy_session_tokens(&source.restored_tokens)?,
            retained_logits_position: source.retained_logits_position,
            prefill_boundary: source.prefill_boundary,
            mirostat: source.mirostat,
            adaptive_controller: source.adaptive_controller.clone(),
            correctable_controller: source.correctable_controller.clone(),
            record,
        };
        self.discard_session(source)?;
        Ok(hibernated)
    }

    /// Restores one host-owned session without replaying its token prefix.
    pub fn wake_session(
        &mut self,
        source: &HibernatedSession,
    ) -> Result<GenerationSession<B>, RuntimeError> {
        let state = source.state.restore(&mut self.backend)?;
        let activations = source.activations.restore(&mut self.backend)?;
        self.backend.synchronize()?;
        Ok(GenerationSession {
            state: Some(state),
            activations: Some(activations),
            evaluated_tokens: copy_session_tokens(&source.evaluated_tokens)?,
            restored_tokens: copy_session_tokens(&source.restored_tokens)?,
            retained_logits_position: source.retained_logits_position,
            prefill_boundary: source.prefill_boundary,
            mirostat: source.mirostat,
            adaptive_controller: source.adaptive_controller.clone(),
            correctable_controller: source.correctable_controller.clone(),
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
            batch_identity: None,
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
                state,
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
            debug_assert_eq!(state.position, block_end);
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
            prefill_method: prefill.prefill_method,
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
            let range =
                state.prepare_append(&mut self.backend, self.model.config.n_layer, 1, 32)?;
            if state.graph_revision != Some(self.graph_revision_token(state)) {
                self.capture_decode_graph(state, activations, attention_shape)?;
            } else {
                self.prepare_kv_state_views(state, range, attention_shape)?;
            }
            self.backend.replay_decode_graph()?;
            state.commit_append(range)?;
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
            prefill_method: prefill.prefill_method,
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

    /// Compares decode-equivalent warm prefill with a sequential control.
    ///
    /// The prefix uses backend-preferred cold prefill in both states.
    /// The suffix runs sequentially in the control and in exact chunks in the
    /// subject. The comparison reads every committed F16 KV value and every
    /// F32 vocabulary logit. The continuation uses greedy decode decisions.
    pub fn characterize_warm_prefill(
        &mut self,
        prefix_tokens: &[u32],
        suffix_tokens: &[u32],
        chunk_tokens: NonZeroUsize,
        continuation_tokens: NonZeroUsize,
    ) -> Result<WarmPrefillCharacterization, RuntimeError> {
        let attention_shape = self.validate_warm_prefill_characterization(
            prefix_tokens,
            suffix_tokens,
            continuation_tokens,
        )?;
        let mut states = self.new_warm_prefill_states(attention_shape)?;
        let (prefix_logits, prefix_kv) = self.compare_warm_prefix(
            prefix_tokens,
            chunk_tokens.get(),
            attention_shape,
            &mut states,
        )?;
        let mut prepared = self.prepare_prompt_prefill_with_numerics(
            suffix_tokens.len(),
            attention_shape.max_context(),
            chunk_tokens.get(),
            PrefillNumerics::DecodeEquivalent,
        )?;
        let (subject_method, subject_chunk_tokens, numerics) =
            self.warm_prefill_plan_details(&prepared)?;
        let (suffix_logits, subject_full_chunks, subject_tail_tokens) = self.compare_warm_suffix(
            suffix_tokens,
            &mut states,
            attention_shape,
            &mut prepared,
            subject_chunk_tokens,
        )?;
        let WarmPrefillCompletion {
            suffix_kv,
            continued_kv,
            sequential_tokens,
            subject_tokens,
            continuation_logits,
        } = self.finish_warm_suffix_phase(
            &mut states,
            attention_shape,
            prefix_tokens.len(),
            suffix_tokens.len(),
            continuation_tokens.get(),
        )?;
        Ok(WarmPrefillCharacterization {
            prefix_tokens: prefix_tokens.len(),
            suffix_tokens: suffix_tokens.len(),
            chunk_tokens: subject_chunk_tokens,
            continuation_tokens: continuation_tokens.get(),
            numerics,
            control_method: PrefillMethod::SequentialDecode,
            subject_method,
            subject_full_chunks,
            subject_tail_tokens,
            prefix_kv,
            prefix_logits,
            suffix_kv,
            continued_kv,
            suffix_logits,
            continuation_logits,
            sequential_tokens,
            subject_tokens,
        })
    }

    fn new_warm_prefill_states(
        &mut self,
        attention_shape: AttentionShape,
    ) -> Result<WarmPrefillStates<B>, RuntimeError> {
        let sequential_state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            KvCacheDtype::F16,
        )?;
        let sequential_activations = Activations::new(&mut self.backend, &self.model.config)?;
        let subject_state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            KvCacheDtype::F16,
        )?;
        let subject_activations = Activations::new(&mut self.backend, &self.model.config)?;
        Ok(WarmPrefillStates {
            sequential_state,
            sequential_activations,
            subject_state,
            subject_activations,
        })
    }

    fn compare_warm_prefix(
        &mut self,
        prefix_tokens: &[u32],
        chunk_tokens: usize,
        attention_shape: AttentionShape,
        states: &mut WarmPrefillStates<B>,
    ) -> Result<(PrefillBitwiseComparison, PrefillBitwiseComparison), RuntimeError> {
        let sequential_logits = self.run_cold_prefill_history(
            prefix_tokens,
            &mut states.sequential_state,
            &mut states.sequential_activations,
            attention_shape,
            chunk_tokens,
        )?;
        let subject_logits = self.run_cold_prefill_history(
            prefix_tokens,
            &mut states.subject_state,
            &mut states.subject_activations,
            attention_shape,
            chunk_tokens,
        )?;
        let logits = compare_f32_bits(&sequential_logits, &subject_logits)?;
        let kv = self.compare_warm_kv_states(states, attention_shape, prefix_tokens.len())?;
        Ok((logits, kv))
    }

    fn finish_warm_suffix_phase(
        &mut self,
        states: &mut WarmPrefillStates<B>,
        attention_shape: AttentionShape,
        prefix_tokens: usize,
        suffix_tokens: usize,
        continuation_tokens: usize,
    ) -> Result<WarmPrefillCompletion, RuntimeError> {
        let suffix_positions = prefix_tokens
            .checked_add(suffix_tokens)
            .ok_or(RuntimeError::SizeOverflow)?;
        let suffix_kv = self.compare_warm_kv_states(states, attention_shape, suffix_positions)?;
        self.finish_prefill_logits(&mut states.subject_activations)?;
        let (sequential_tokens, subject_tokens, continuation_logits) =
            self.compare_warm_continuation(states, continuation_tokens)?;
        let continued_positions = suffix_positions
            .checked_add(continuation_tokens.saturating_sub(1))
            .ok_or(RuntimeError::SizeOverflow)?;
        let continued_kv =
            self.compare_warm_kv_states(states, attention_shape, continued_positions)?;
        Ok(WarmPrefillCompletion {
            suffix_kv,
            continued_kv,
            sequential_tokens,
            subject_tokens,
            continuation_logits,
        })
    }

    fn validate_warm_prefill_characterization(
        &self,
        prefix_tokens: &[u32],
        suffix_tokens: &[u32],
        continuation_tokens: NonZeroUsize,
    ) -> Result<AttentionShape, RuntimeError> {
        if prefix_tokens.is_empty() || suffix_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if !self.backend.decode_equivalent_prefill_supported() {
            return Err(BackendError::operation(
                "characterize warm prefill",
                "the backend does not support decode-equivalent prefill",
            )
            .into());
        }
        self.validate_generation_tokens(prefix_tokens)?;
        self.validate_generation_tokens(suffix_tokens)?;
        let requested_context = prefix_tokens
            .len()
            .checked_add(suffix_tokens.len())
            .and_then(|tokens| tokens.checked_add(continuation_tokens.get().saturating_sub(1)))
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

    fn run_cold_prefill_history(
        &mut self,
        tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        chunk_tokens: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut prepared =
            self.prepare_prompt_prefill(tokens.len(), attention_shape.max_context(), chunk_tokens)?;
        if !matches!(prepared, PreparedPrefill::Chunked { .. }) {
            return Err(BackendError::operation(
                "characterize warm prefill",
                "the cold prefix did not select chunked prefill",
            )
            .into());
        }
        self.run_prompt_prefill(
            tokens,
            state,
            activations,
            attention_shape,
            &mut prepared,
            || false,
        )?;
        self.backend.synchronize()?;
        let mut logits = vec![0.0_f32; self.model.config.vocab_size];
        self.backend.read_f32(&activations.logits, &mut logits)?;
        Ok(logits)
    }

    fn warm_prefill_plan_details(
        &self,
        prepared: &PreparedPrefill<B>,
    ) -> Result<(PrefillMethod, usize, PrefillNumerics), RuntimeError> {
        match prepared {
            PreparedPrefill::Chunked {
                plan, chunk_tokens, ..
            } if plan.numerics() == PrefillNumerics::DecodeEquivalent => Ok((
                self.backend.prefill_method(),
                *chunk_tokens,
                plan.numerics(),
            )),
            PreparedPrefill::Chunked { .. } => Err(BackendError::operation(
                "characterize warm prefill",
                "the prepared plan does not require decode-equivalent arithmetic",
            )
            .into()),
            PreparedPrefill::Reused | PreparedPrefill::Sequential => Err(BackendError::operation(
                "characterize warm prefill",
                "the backend did not prepare a chunked decode-equivalent plan",
            )
            .into()),
        }
    }

    fn compare_warm_suffix(
        &mut self,
        suffix_tokens: &[u32],
        states: &mut WarmPrefillStates<B>,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
        chunk_tokens: usize,
    ) -> Result<(PrefillBitwiseComparison, usize, usize), RuntimeError> {
        let plan = self.warm_prefill_plan(prepared)?;
        self.backend.prepare_prefill(plan)?;
        let mut compared = PrefillBitwiseComparison::empty();
        let mut processed = 0_usize;
        let base_position = states.subject_state.position;
        while processed < suffix_tokens.len() {
            let count = chunk_tokens.min(suffix_tokens.len() - processed);
            compared.add(self.compare_warm_suffix_chunk(
                suffix_tokens,
                &mut processed,
                base_position,
                states,
                attention_shape,
                prepared,
                count,
            )?)?;
        }
        let full_chunks = suffix_tokens.len() / chunk_tokens;
        let tail_tokens = suffix_tokens.len() % chunk_tokens;
        Ok((compared, full_chunks, tail_tokens))
    }

    fn warm_prefill_plan(
        &self,
        prepared: &PreparedPrefill<B>,
    ) -> Result<PrefillPlan, RuntimeError> {
        match prepared {
            PreparedPrefill::Chunked { plan, .. }
                if plan.numerics() == PrefillNumerics::DecodeEquivalent =>
            {
                Ok(*plan)
            }
            PreparedPrefill::Chunked { .. } => Err(BackendError::operation(
                "characterize warm prefill",
                "the prepared plan does not require decode-equivalent arithmetic",
            )
            .into()),
            PreparedPrefill::Reused | PreparedPrefill::Sequential => Err(BackendError::operation(
                "characterize warm prefill",
                "the backend did not prepare a chunked decode-equivalent plan",
            )
            .into()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_warm_suffix_chunk(
        &mut self,
        suffix_tokens: &[u32],
        processed: &mut usize,
        base_position: usize,
        states: &mut WarmPrefillStates<B>,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
        count: usize,
    ) -> Result<PrefillBitwiseComparison, RuntimeError> {
        let logits_elements = count
            .checked_mul(self.model.config.vocab_size)
            .ok_or(RuntimeError::SizeOverflow)?;
        let mut batch_logits = self.backend.allocate_classified(
            BufferLayout::f32(logits_elements)?,
            MemoryClass::PrefillScratch,
        )?;
        self.run_warm_subject_chunk(
            suffix_tokens,
            processed,
            base_position,
            &mut states.subject_state,
            &mut states.subject_activations,
            attention_shape,
            prepared,
            count,
            &mut batch_logits,
        )?;
        self.compare_warm_suffix_rows(
            suffix_tokens,
            processed
                .checked_sub(count)
                .ok_or(RuntimeError::SizeOverflow)?,
            count,
            &batch_logits,
            states,
            attention_shape,
        )
    }

    fn compare_warm_suffix_rows(
        &mut self,
        suffix_tokens: &[u32],
        start: usize,
        count: usize,
        batch_logits: &B::Buffer,
        states: &mut WarmPrefillStates<B>,
        attention_shape: AttentionShape,
    ) -> Result<PrefillBitwiseComparison, RuntimeError> {
        let mut compared = PrefillBitwiseComparison::empty();
        let mut subject_row = vec![0.0_f32; self.model.config.vocab_size];
        let mut sequential_row = vec![0.0_f32; self.model.config.vocab_size];
        for row in 0..count {
            compared.add(self.compare_warm_suffix_row(
                suffix_tokens[start + row],
                row,
                batch_logits,
                states,
                attention_shape,
                &mut sequential_row,
                &mut subject_row,
            )?)?;
        }
        Ok(compared)
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_warm_suffix_row(
        &mut self,
        token: u32,
        row: usize,
        batch_logits: &B::Buffer,
        states: &mut WarmPrefillStates<B>,
        attention_shape: AttentionShape,
        sequential_row: &mut [f32],
        subject_row: &mut [f32],
    ) -> Result<PrefillBitwiseComparison, RuntimeError> {
        self.backend.copy_f32_row(
            batch_logits,
            row,
            self.model.config.vocab_size,
            &mut states.subject_activations.logits,
        )?;
        self.backend
            .read_f32(&states.subject_activations.logits, subject_row)?;
        self.backend
            .write_u32(&mut states.sequential_activations.sampled, &[token])?;
        self.forward(
            &mut states.sequential_state,
            &mut states.sequential_activations,
            attention_shape,
        )?;
        self.backend
            .read_f32(&states.sequential_activations.logits, sequential_row)?;
        compare_f32_bits(sequential_row, subject_row)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_warm_subject_chunk(
        &mut self,
        suffix_tokens: &[u32],
        processed: &mut usize,
        base_position: usize,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
        count: usize,
        batch_logits: &mut B::Buffer,
    ) -> Result<(), RuntimeError> {
        let PreparedPrefill::Chunked {
            chunk_tokens,
            full,
            tail,
            ..
        } = prepared
        else {
            return Err(BackendError::operation(
                "characterize warm prefill",
                "the subject prefill is not chunked",
            )
            .into());
        };
        self.run_chunked_prefill_step(
            suffix_tokens,
            processed,
            base_position,
            state,
            activations,
            attention_shape,
            chunk_tokens,
            count,
            full,
            tail,
        )?;
        let batch = if count == *chunk_tokens {
            &mut *full
        } else {
            tail.as_mut().ok_or_else(|| {
                BackendError::operation(
                    "characterize warm prefill",
                    "the tail activation storage is missing",
                )
            })?
        };
        self.finish_warm_prefill_logits(batch, batch_logits, count)
    }

    fn finish_warm_prefill_logits(
        &mut self,
        activations: &mut PrefillActivations<B>,
        logits: &mut B::Buffer,
        tokens: usize,
    ) -> Result<(), RuntimeError> {
        let config = &self.model.config;
        self.backend.prefill_rms_norm(
            &activations.hidden,
            &self.model.weights.output_norm,
            &mut activations.norm,
            VectorShape::new(tokens, config.n_embd)?,
            config.rms_epsilon,
        )?;
        let output = match &self.model.weights.output {
            OutputWeight::Separate(weight) => weight,
            OutputWeight::Tied => &self.model.weights.token_embedding,
        };
        self.backend.prefill_gemm(
            &output.buffer,
            &activations.norm,
            logits,
            output.shape,
            tokens,
        )?;
        Ok(())
    }

    fn compare_warm_continuation(
        &mut self,
        states: &mut WarmPrefillStates<B>,
        tokens: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, PrefillBitwiseComparison), RuntimeError> {
        let mut sequential_row = vec![0.0_f32; self.model.config.vocab_size];
        let mut subject_row = vec![0.0_f32; self.model.config.vocab_size];
        let mut sequential_tokens = Vec::with_capacity(tokens);
        let mut subject_tokens = Vec::with_capacity(tokens);
        let mut compared = PrefillBitwiseComparison::empty();
        for step in 0..tokens {
            let (sequential_token, subject_token, row_comparison) =
                self.compare_warm_continuation_row(states, &mut sequential_row, &mut subject_row)?;
            compared.add(row_comparison)?;
            sequential_tokens.push(sequential_token);
            subject_tokens.push(subject_token);
            if step + 1 < tokens {
                self.advance_warm_continuation(states, sequential_token, subject_token)?;
            }
        }
        self.backend.synchronize()?;
        Ok((sequential_tokens, subject_tokens, compared))
    }

    fn compare_warm_continuation_row(
        &mut self,
        states: &mut WarmPrefillStates<B>,
        sequential_row: &mut [f32],
        subject_row: &mut [f32],
    ) -> Result<(u32, u32, PrefillBitwiseComparison), RuntimeError> {
        self.backend
            .read_f32(&states.sequential_activations.logits, sequential_row)?;
        self.backend
            .read_f32(&states.subject_activations.logits, subject_row)?;
        let comparison = compare_f32_bits(sequential_row, subject_row)?;
        let sequential_token = greedy_token(sequential_row)?;
        let subject_token = greedy_token(subject_row)?;
        Ok((sequential_token, subject_token, comparison))
    }

    fn advance_warm_continuation(
        &mut self,
        states: &mut WarmPrefillStates<B>,
        sequential_token: u32,
        subject_token: u32,
    ) -> Result<(), RuntimeError> {
        self.backend.write_u32(
            &mut states.sequential_activations.sampled,
            &[sequential_token],
        )?;
        let sequential_shape = states.sequential_state.shape;
        self.forward(
            &mut states.sequential_state,
            &mut states.sequential_activations,
            sequential_shape,
        )?;
        self.backend
            .write_u32(&mut states.subject_activations.sampled, &[subject_token])?;
        let subject_shape = states.subject_state.shape;
        self.forward(
            &mut states.subject_state,
            &mut states.subject_activations,
            subject_shape,
        )?;
        Ok(())
    }

    fn compare_warm_kv_states(
        &mut self,
        states: &WarmPrefillStates<B>,
        attention_shape: AttentionShape,
        positions: usize,
    ) -> Result<PrefillBitwiseComparison, RuntimeError> {
        self.backend.synchronize()?;
        let sequential =
            self.read_prefill_kv(&states.sequential_state, attention_shape, positions)?;
        let subject = self.read_prefill_kv(&states.subject_state, attention_shape, positions)?;
        compare_prefill_kv_bits(&sequential, &subject)
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

    /// Returns the physical prefix reusable for this exact request.
    ///
    /// The check includes context capacity, KV type, and branch position.
    pub fn reusable_prefill_tokens(
        &self,
        session: &GenerationSession<B>,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
    ) -> Result<usize, RuntimeError> {
        Ok(self
            .plan_session_replay(session, prompt_tokens, options)?
            .reused_tokens)
    }

    /// Classifies reusable state for a prompt without executing it.
    ///
    /// The result describes the current session. It does not certify that
    /// prefill or decode completes.
    pub fn plan_session_replay(
        &self,
        session: &GenerationSession<B>,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
    ) -> Result<SessionReplay, RuntimeError> {
        let shape = self.validate_generation_request(prompt_tokens, options)?;
        let reuse = self.generation_reuse(session, prompt_tokens, options, shape);
        Ok(SessionReplay::for_prompt(
            reuse.reuse_class,
            reuse.cached_tokens.max(session.restored_tokens.len()),
            reuse.reused_tokens,
            reuse.restored_common.max(reuse.common_tokens),
            prompt_tokens.len(),
        ))
    }

    /// Starts prompt prefill without sampling or exposing partial session state.
    pub fn begin_prefill(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: GenerateOptions,
    ) -> Result<PendingPrefill<B>, RuntimeError> {
        let preparation = self.prepare_resumable_state(session, prompt_tokens, &options)?;
        let restored_tokens = session.restored_tokens.len();
        let GenerationPreparation {
            attention_shape,
            state,
            activations,
            prepared,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            verify_activations: _,
        } = preparation;
        let split_replay = reuse_class == SessionReuseClass::RestoreReplay
            && reused_tokens == 0
            && replay_prefill_boundary > 0
            && replay_prefill_boundary < prompt_tokens.len();
        let workspace = prepared_prefill_workspace(&prepared);
        let stage = if split_replay {
            PendingPrefillStage::Replay {
                boundary: replay_prefill_boundary,
                processed: 0,
                prepared,
            }
        } else {
            PendingPrefillStage::Single {
                start: reused_tokens,
                end: prompt_tokens.len(),
                processed: 0,
                prepared,
            }
        };
        let mirostat = Self::restore_mirostat(session, &options, reuse_class);
        let (adaptive_controller, correctable_controller) =
            Self::restore_generation_controllers(session, &options);
        let pending = PendingPrefill {
            prompt_tokens: prompt_tokens.to_vec(),
            options,
            attention_shape,
            state,
            activations,
            stage,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            mirostat,
            adaptive_controller,
            correctable_controller,
            last_fork: session.last_fork,
            last_hibernation: session.last_hibernation,
            workspace,
        };
        self.discard_session(session)?;
        Ok(pending)
    }

    fn prepare_resumable_state(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
    ) -> Result<GenerationPreparation<B>, RuntimeError> {
        let attention_shape = self
            .validate_generation_request(prompt_tokens, options)
            .map_err(|error| error.with_session_failure_effect(SessionFailureEffect::Unchanged))?;
        let preparation = self
            .prepare_generation_state(session, prompt_tokens, options, attention_shape)
            .map_err(|error| error.with_session_failure_effect(SessionFailureEffect::Unchanged))?;
        self.batch_graph_signature = None;
        session.batch_identity = None;
        Ok(preparation)
    }

    /// Advances prefill by at most the requested number of prompt positions.
    ///
    /// Chunked prefill advances only at fixed chunk boundaries. A budget below
    /// the next chunk returns [`RuntimeError::PrefillBudgetTooSmall`]. Sequential
    /// fallback advances one token at a time and computes output logits for the
    /// last token in each bounded advance.
    /// Errors consume and discard the pending state with a quarantine effect.
    /// Call [`PendingPrefill::minimum_budget`] before selecting a budget.
    pub fn advance_prefill<C>(
        &mut self,
        mut pending: PendingPrefill<B>,
        budget: NonZeroUsize,
        mut cancelled: C,
    ) -> Result<PrefillProgress<B>, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        self.batch_graph_signature = None;
        let mut budget = budget.get();
        loop {
            let step = self
                .advance_prefill_step(&mut pending, &mut budget, &mut cancelled)
                .map_err(|error| {
                    error.with_session_failure_effect(SessionFailureEffect::Quarantine)
                })?;
            match step {
                PrefillAdvanceStep::Continue => {}
                PrefillAdvanceStep::Pending => return Ok(PrefillProgress::Pending(pending)),
                PrefillAdvanceStep::Ready => {
                    return Ok(PrefillProgress::Ready(self.pending_into_ready(pending)))
                }
                PrefillAdvanceStep::Cancelled => {
                    return Ok(PrefillProgress::Cancelled(CancelledPrefill {
                        processed_tokens: pending.processed_tokens(),
                        workspace: pending.workspace,
                    }))
                }
            }
        }
    }

    fn advance_prefill_step<C>(
        &mut self,
        pending: &mut PendingPrefill<B>,
        budget: &mut usize,
        cancelled: &mut C,
    ) -> Result<PrefillAdvanceStep, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        if let Some(step) = self.prefill_early_step(pending, cancelled)? {
            return Ok(step);
        }
        let execution = self.advance_prefill_execution(pending, *budget, cancelled)?;
        self.finish_prefill_step(pending, budget, execution, cancelled)
    }

    fn prefill_early_step<C>(
        &mut self,
        pending: &PendingPrefill<B>,
        cancelled: &mut C,
    ) -> Result<Option<PrefillAdvanceStep>, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        if matches!(pending.stage, PendingPrefillStage::Done) {
            if cancelled() {
                self.backend.synchronize()?;
                return Ok(Some(PrefillAdvanceStep::Cancelled));
            }
            self.backend.synchronize()?;
            return Ok(Some(PrefillAdvanceStep::Ready));
        }
        Ok(None)
    }

    fn advance_prefill_execution<C>(
        &mut self,
        pending: &mut PendingPrefill<B>,
        budget: usize,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        match self.advance_pending_prefill_stage(
            &pending.prompt_tokens,
            &mut pending.state,
            &mut pending.activations,
            &mut pending.stage,
            pending.attention_shape,
            budget,
            cancelled,
        ) {
            Ok(execution) => Ok(execution),
            Err(error) => {
                let _ = self.backend.synchronize();
                Err(error)
            }
        }
    }

    fn finish_prefill_step<C>(
        &mut self,
        pending: &mut PendingPrefill<B>,
        budget: &mut usize,
        execution: PrefillExecution,
        cancelled: &mut C,
    ) -> Result<PrefillAdvanceStep, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        *budget = (*budget).saturating_sub(execution.processed_tokens);
        if execution.cancelled {
            self.backend.synchronize()?;
            return Ok(PrefillAdvanceStep::Cancelled);
        }
        if self.prefill_cancelled_after_execution(&execution, cancelled)? {
            return Ok(PrefillAdvanceStep::Cancelled);
        }
        if execution.complete {
            self.transition_pending_prefill_stage(pending);
            return self.complete_prefill_step(pending, *budget);
        }
        self.incomplete_prefill_step(&execution, *budget)
    }

    fn incomplete_prefill_step(
        &mut self,
        execution: &PrefillExecution,
        budget: usize,
    ) -> Result<PrefillAdvanceStep, RuntimeError> {
        if execution.processed_tokens > 0 && budget > 0 {
            // A chunk boundary can leave a budget remainder below the next chunk.
            self.backend.synchronize()?;
            return Ok(PrefillAdvanceStep::Pending);
        }
        if budget == 0 {
            self.backend.synchronize()?;
            return Ok(PrefillAdvanceStep::Pending);
        }
        Ok(PrefillAdvanceStep::Continue)
    }

    fn prefill_cancelled_after_execution<C>(
        &mut self,
        execution: &PrefillExecution,
        cancelled: &mut C,
    ) -> Result<bool, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        if (execution.processed_tokens > 0 || execution.complete) && cancelled() {
            self.backend.synchronize()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn complete_prefill_step(
        &mut self,
        pending: &PendingPrefill<B>,
        budget: usize,
    ) -> Result<PrefillAdvanceStep, RuntimeError> {
        if matches!(pending.stage, PendingPrefillStage::Done) {
            self.backend.synchronize()?;
            return Ok(PrefillAdvanceStep::Ready);
        }
        if budget == 0 {
            self.backend.synchronize()?;
            return Ok(PrefillAdvanceStep::Pending);
        }
        Ok(PrefillAdvanceStep::Continue)
    }

    /// Commits ready prompt state to a generation session.
    pub fn finish_prefill(
        &mut self,
        ready: ReadyPrefill<B>,
        session: &mut GenerationSession<B>,
    ) -> Result<(), RuntimeError> {
        if !session.is_empty() {
            return Err(RuntimeError::PrefillSessionBusy);
        }
        let ReadyPrefill {
            prompt_tokens,
            options: _,
            state,
            activations,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            mirostat,
            adaptive_controller,
            correctable_controller,
            workspace: _,
            last_fork,
            last_hibernation,
        } = ready;
        session.last_replay = SessionReplay::for_prompt(
            reuse_class,
            cached_tokens.max(restored_tokens),
            reused_tokens,
            restored_common.max(common_tokens),
            prompt_tokens.len(),
        );
        session.last_fork = last_fork;
        session.last_hibernation = last_hibernation;
        Self::commit_generation_session(
            session,
            &prompt_tokens,
            &[],
            state,
            activations,
            reuse_class,
            replay_prefill_boundary,
            GenerationTermination::Completed,
            mirostat,
            adaptive_controller,
            correctable_controller,
        );
        Ok(())
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
        self.retire_graph_for_generation(&reuse)?;
        self.ensure_speculative_logits(options)?;
        let prepared = self.prepare_prefill_for_generation(prompt_tokens, options, &reuse)?;
        let verify_activations = self.configure_verify_activations(options)?;
        let first_append = Self::first_generation_append(
            &prepared,
            prompt_tokens.len().saturating_sub(reuse.reused_tokens),
        );
        let (state, activations) =
            self.take_generation_state(session, options, attention_shape, &reuse, first_append)?;
        Ok(GenerationPreparation {
            attention_shape: state.shape,
            state,
            activations,
            prepared,
            reuse_class: reuse.reuse_class,
            reused_tokens: reuse.reused_tokens,
            cached_tokens: reuse.cached_tokens,
            restored_common: reuse.restored_common,
            common_tokens: reuse.common_tokens,
            replay_prefill_boundary: reuse.replay_prefill_boundary,
            verify_activations,
        })
    }

    fn prepare_prefill_for_generation(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
        reuse: &GenerationReuse,
    ) -> Result<PreparedPrefill<B>, RuntimeError> {
        let split_replay = reuse.reuse_class == SessionReuseClass::RestoreReplay
            && reuse.reused_tokens == 0
            && reuse.replay_prefill_boundary > 0
            && reuse.replay_prefill_boundary < prompt_tokens.len();
        if split_replay {
            self.prepare_replay_prefill(reuse.replay_prefill_boundary, options)
        } else {
            self.prepare_generation_prefill(
                prompt_tokens.len().saturating_sub(reuse.reused_tokens),
                prompt_tokens.len(),
                options,
                reuse.reused_tokens,
                false,
            )
        }
    }

    fn retire_graph_for_generation(&mut self, reuse: &GenerationReuse) -> Result<(), RuntimeError> {
        if reuse.compatible {
            Ok(())
        } else {
            self.retire_decode_graph()
        }
    }

    fn first_generation_append(
        prepared: &PreparedPrefill<B>,
        remaining_tokens: usize,
    ) -> Option<(usize, usize)> {
        if remaining_tokens == 0 {
            return None;
        }
        match prepared {
            PreparedPrefill::Reused => None,
            PreparedPrefill::Sequential => Some((1, 32)),
            PreparedPrefill::Chunked { chunk_tokens, .. } => {
                let tokens = (*chunk_tokens).min(remaining_tokens);
                NonZeroUsize::new(tokens).map(|tokens| (tokens.get(), tokens.get()))
            }
        }
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
            .map(|state| {
                Self::retained_state_compatible(
                    state,
                    attention_shape,
                    options.kv_cache_dtype,
                    cached_tokens,
                    common_tokens,
                )
            })
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

    fn retained_state_compatible(
        state: &KvState<B>,
        attention_shape: AttentionShape,
        dtype: KvCacheDtype,
        cached_tokens: usize,
        common_tokens: usize,
    ) -> bool {
        if state.dtype != dtype {
            return false;
        }
        state.shape == attention_shape
            || (cached_tokens > 0
                && common_tokens == cached_tokens
                && state.can_retain_at(attention_shape))
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
        first_append: Option<(usize, usize)>,
    ) -> Result<(KvState<B>, Activations<B>), RuntimeError> {
        if reuse.compatible {
            self.take_compatible_generation_state(session, attention_shape, reuse, first_append)
        } else {
            self.take_cold_generation_state(session, options, attention_shape, first_append)
        }
    }

    fn take_compatible_generation_state(
        &mut self,
        session: &mut GenerationSession<B>,
        attention_shape: AttentionShape,
        reuse: &GenerationReuse,
        first_append: Option<(usize, usize)>,
    ) -> Result<(KvState<B>, Activations<B>), RuntimeError> {
        if session.activations.is_none() {
            return Err(RuntimeError::SizeOverflow);
        }
        let shape_growth = session
            .state
            .as_ref()
            .is_some_and(|state| Self::needs_shape_growth(state, attention_shape));
        if shape_growth {
            self.retire_decode_graph()?;
        }
        let state = session.state.as_mut().ok_or(RuntimeError::SizeOverflow)?;
        self.prepare_compatible_state(
            state,
            attention_shape,
            reuse.reused_tokens,
            first_append,
            shape_growth,
        )?;
        let state = session.state.take().ok_or(RuntimeError::SizeOverflow)?;
        let activations = session
            .activations
            .take()
            .ok_or(RuntimeError::SizeOverflow)?;
        session.retained_logits_position = None;
        Ok((state, activations))
    }

    fn needs_shape_growth(state: &KvState<B>, attention_shape: AttentionShape) -> bool {
        state.shape.max_context() < attention_shape.max_context()
            && state.can_grow_to(attention_shape)
    }

    fn prepare_compatible_state(
        &mut self,
        state: &mut KvState<B>,
        attention_shape: AttentionShape,
        reused_tokens: usize,
        first_append: Option<(usize, usize)>,
        shape_growth: bool,
    ) -> Result<(), RuntimeError> {
        if shape_growth {
            return state
                .prepare_shape_transition(
                    &mut self.backend,
                    self.model.config.n_layer,
                    attention_shape,
                    reused_tokens,
                    first_append,
                )
                .map_err(Into::into);
        }
        self.prepare_compatible_append(state, reused_tokens, first_append)
    }

    fn prepare_compatible_append(
        &mut self,
        state: &mut KvState<B>,
        reused_tokens: usize,
        first_append: Option<(usize, usize)>,
    ) -> Result<(), RuntimeError> {
        match first_append {
            Some((tokens, growth_tokens)) => state
                .prepare_append_at(
                    &mut self.backend,
                    self.model.config.n_layer,
                    reused_tokens,
                    tokens,
                    growth_tokens,
                )
                .map(|_| ())
                .map_err(Into::into),
            None => state.truncate(reused_tokens).map_err(Into::into),
        }
    }

    fn take_cold_generation_state(
        &mut self,
        session: &mut GenerationSession<B>,
        options: &GenerateOptions,
        attention_shape: AttentionShape,
        first_append: Option<(usize, usize)>,
    ) -> Result<(KvState<B>, Activations<B>), RuntimeError> {
        let mut state = KvState::new(
            &mut self.backend,
            self.model.config.n_layer,
            attention_shape,
            options.kv_cache_dtype,
        )?;
        let activations = Activations::new(&mut self.backend, &self.model.config)?;
        if let Some((tokens, growth_tokens)) = first_append {
            state.prepare_append(
                &mut self.backend,
                self.model.config.n_layer,
                tokens,
                growth_tokens,
            )?;
        }
        session.state = None;
        session.activations = None;
        session.retained_logits_position = None;
        Ok((state, activations))
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
                let available_proposals = options
                    .max_tokens
                    .saturating_sub(tokens.len())
                    .saturating_sub(1);
                let mut drafted = drafter.draft(context);
                if let Draft::Tokens(proposals) = &mut drafted {
                    proposals.truncate(available_proposals);
                    if proposals.is_empty() {
                        drafted = Draft::Nothing;
                    }
                }
                Ok(GenerationDraft {
                    drafted,
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
    fn initialize_generation_decode<F, C, S>(
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
        cancelled: &mut C,
        stopped: &mut S,
        attention_shape: AttentionShape,
    ) -> Result<GenerationInitialization, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
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
        let termination = Self::poll_generation_termination(cancelled, stopped);
        let eos = self.model.tokenizer.eos_token();
        let profile_target = Self::generation_profile_target(options, tokens.len())?;
        let profile_start =
            self.begin_generation_profile_if_active(profile_target, first, eos, termination)?;
        let use_graph = Self::generation_graph_after_first(
            options,
            profile_target,
            self.backend.decode_graph_supported(),
            eos,
            first_token(tokens),
            termination.is_some(),
        );
        if use_graph {
            self.capture_decode_graph(state, activations, attention_shape)?;
        }
        let (adaptive_controller, correctable_controller) =
            Self::restore_generation_controllers(session, options);
        Ok(GenerationInitialization {
            termination,
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
        cancelled: C,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
    {
        self.generate_session_tokens_with_stop(
            session,
            prompt_tokens,
            options,
            on_token,
            cancelled,
            || false,
        )
    }

    /// Generates tokens with a stop predicate that preserves the session.
    pub fn generate_session_tokens_with_stop<F, C, S>(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: GenerateOptions,
        on_token: F,
        cancelled: C,
        stopped: S,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        let (_, preparation) = self.prepare_generation_session(session, prompt_tokens, &options)?;
        self.batch_graph_signature = None;
        session.batch_identity = None;
        let result = self.generate_session_tokens_inner(
            session,
            prompt_tokens,
            options,
            preparation,
            on_token,
            cancelled,
            stopped,
        );
        result.map_err(|error| self.quarantine_generation_failure(session, error))
    }

    fn quarantine_generation_failure(
        &mut self,
        session: &mut GenerationSession<B>,
        error: RuntimeError,
    ) -> RuntimeError {
        match self.retire_decode_graph() {
            Ok(()) => {
                session.invalidate();
                error.with_forced_session_failure_effect(SessionFailureEffect::Quarantine)
            }
            Err(retirement) => {
                retirement.with_forced_session_failure_effect(SessionFailureEffect::Quarantine)
            }
        }
    }

    fn generation_graph_after_first(
        options: &GenerateOptions,
        profile_target: usize,
        graph_supported: bool,
        eos: Option<u32>,
        first_token: u32,
        stopped_after_first: bool,
    ) -> bool {
        !stopped_after_first
            && options.max_tokens > 1
            && Self::generation_uses_graph(
                options,
                profile_target,
                graph_supported,
                eos,
                first_token,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_session_tokens_inner<F, C, S>(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: GenerateOptions,
        preparation: GenerationPreparation<B>,
        on_token: F,
        mut cancelled: C,
        stopped: S,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        let GenerationPreparation {
            attention_shape,
            state,
            activations,
            mut prepared,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            verify_activations,
        } = preparation;
        let mut state = Some(state);
        let mut activations = Some(activations);
        let prefill_tokens = &prompt_tokens[reused_tokens..];
        let split_replay = reuse_class == SessionReuseClass::RestoreReplay
            && reused_tokens == 0
            && replay_prefill_boundary > 0
            && replay_prefill_boundary < prompt_tokens.len();
        let prefill_start = Instant::now();
        let prefill = self.execute_generation_prefill(
            session,
            prompt_tokens,
            prefill_tokens,
            replay_prefill_boundary,
            split_replay,
            &mut state,
            &mut activations,
            attention_shape,
            &mut prepared,
            &mut cancelled,
        )?;
        let prefill_duration = prefill_start.elapsed();
        if prefill.cancelled {
            let result = self.cancelled_prefill_result(
                session,
                prompt_tokens,
                reused_tokens,
                prefill,
                prefill_duration,
            );
            if result.is_err() {
                Self::restore_generation_resources(session, &mut state, &mut activations);
            }
            return result;
        }
        if cancelled() {
            if let Err(error) = self.discard_session(session) {
                Self::restore_generation_resources(session, &mut state, &mut activations);
                return Err(error);
            }
            return Ok(cancelled_result(
                prompt_tokens.to_vec(),
                prefill_duration,
                prefill.prefill_method,
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
            &mut state,
            &mut activations,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            on_token,
            cancelled,
            stopped,
            verify_activations,
        )
        .inspect_err(|_| {
            Self::restore_generation_resources(session, &mut state, &mut activations);
        })
    }

    fn restore_generation_resources(
        session: &mut GenerationSession<B>,
        state: &mut Option<KvState<B>>,
        activations: &mut Option<Activations<B>>,
    ) {
        if session.state.is_none() {
            session.state = state.take();
        }
        if session.activations.is_none() {
            session.activations = activations.take();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_generation_prefill<C>(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        replay_prefill_boundary: usize,
        split_replay: bool,
        state: &mut Option<KvState<B>>,
        activations: &mut Option<Activations<B>>,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        let prefill = match self.run_generation_prefill(
            prompt_tokens,
            prefill_tokens,
            replay_prefill_boundary,
            split_replay,
            state.as_mut().ok_or(RuntimeError::SizeOverflow)?,
            activations.as_mut().ok_or(RuntimeError::SizeOverflow)?,
            attention_shape,
            prepared,
            cancelled,
        ) {
            Ok(prefill) => prefill,
            Err(error) => {
                Self::restore_generation_resources(session, state, activations);
                return Err(error);
            }
        };
        if let Err(error) = self.backend.synchronize() {
            Self::restore_generation_resources(session, state, activations);
            return Err(error.into());
        }
        Ok(prefill)
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
        let Some(numerics) = self.generation_prefill_numerics(options, reused_tokens) else {
            return Ok(PreparedPrefill::Sequential);
        };
        self.prepare_prompt_prefill_with_numerics(
            prefill_tokens,
            context_tokens,
            options.prefill_chunk_tokens,
            numerics,
        )
    }

    fn generation_prefill_numerics(
        &self,
        options: &GenerateOptions,
        reused_tokens: usize,
    ) -> Option<PrefillNumerics> {
        if self.q8_prefill_unavailable(options) {
            return None;
        }
        if reused_tokens == 0 {
            return Some(PrefillNumerics::BackendPreferred);
        }
        if !self.decode_equivalent_warm_prefill_supported(options) {
            return None;
        }
        Some(PrefillNumerics::DecodeEquivalent)
    }

    fn q8_prefill_unavailable(&self, options: &GenerateOptions) -> bool {
        options.kv_cache_dtype == KvCacheDtype::Q8 && !self.backend.q8_prefill_supported()
    }

    fn decode_equivalent_warm_prefill_supported(&self, options: &GenerateOptions) -> bool {
        options.kv_cache_dtype == KvCacheDtype::F16
            && self.backend.decode_equivalent_prefill_supported()
    }

    fn prepare_replay_prefill(
        &mut self,
        replay_tokens: usize,
        options: &GenerateOptions,
    ) -> Result<PreparedPrefill<B>, RuntimeError> {
        if self.q8_prefill_unavailable(options) {
            return Ok(PreparedPrefill::Sequential);
        }
        self.prepare_prompt_prefill(replay_tokens, replay_tokens, options.prefill_chunk_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_prefill_range<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        start: usize,
        end: usize,
        processed: &mut usize,
        prepared: &mut PreparedPrefill<B>,
        attention_shape: AttentionShape,
        budget: usize,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        let offset = start
            .checked_add(*processed)
            .ok_or(RuntimeError::SizeOverflow)?;
        let tokens = prompt_tokens
            .get(offset..end)
            .ok_or(RuntimeError::SizeOverflow)?;
        let execution = self.run_prompt_prefill_bounded(
            tokens,
            state,
            activations,
            attention_shape,
            prepared,
            budget,
            cancelled,
        )?;
        *processed = (*processed)
            .checked_add(execution.processed_tokens)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok(execution)
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_pending_prefill_stage<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        stage: &mut PendingPrefillStage<B>,
        attention_shape: AttentionShape,
        budget: usize,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        match stage {
            PendingPrefillStage::Single {
                start,
                end,
                processed,
                prepared,
            }
            | PendingPrefillStage::Continuation {
                start,
                end,
                processed,
                prepared,
            } => self.advance_prefill_range(
                prompt_tokens,
                state,
                activations,
                *start,
                *end,
                processed,
                prepared,
                attention_shape,
                budget,
                cancelled,
            ),
            PendingPrefillStage::Replay {
                boundary,
                processed,
                prepared,
            } => self.advance_prefill_range(
                prompt_tokens,
                state,
                activations,
                0,
                *boundary,
                processed,
                prepared,
                attention_shape,
                budget,
                cancelled,
            ),
            PendingPrefillStage::Done => Ok(PrefillExecution {
                processed_tokens: 0,
                cancelled: false,
                complete: true,
                workspace: PrefillWorkspace::default(),
                prefill_method: PrefillMethod::Reused,
            }),
        }
    }

    fn transition_pending_prefill_stage(&mut self, pending: &mut PendingPrefill<B>) {
        let stage = std::mem::replace(&mut pending.stage, PendingPrefillStage::Done);
        pending.stage = match stage {
            PendingPrefillStage::Replay { boundary, .. }
                if boundary < pending.prompt_tokens.len() =>
            {
                PendingPrefillStage::Continuation {
                    start: boundary,
                    end: pending.prompt_tokens.len(),
                    processed: 0,
                    prepared: PreparedPrefill::Sequential,
                }
            }
            PendingPrefillStage::Single { .. }
            | PendingPrefillStage::Continuation { .. }
            | PendingPrefillStage::Replay { .. }
            | PendingPrefillStage::Done => PendingPrefillStage::Done,
        };
    }

    fn pending_into_ready(&mut self, pending: PendingPrefill<B>) -> ReadyPrefill<B> {
        let PendingPrefill {
            prompt_tokens,
            options,
            attention_shape: _,
            state,
            activations,
            stage: _,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            mirostat,
            adaptive_controller,
            correctable_controller,
            workspace,
            last_fork,
            last_hibernation,
        } = pending;
        ReadyPrefill {
            prompt_tokens,
            options,
            state,
            activations,
            reuse_class,
            reused_tokens,
            cached_tokens,
            restored_tokens,
            restored_common,
            common_tokens,
            replay_prefill_boundary,
            mirostat,
            adaptive_controller,
            correctable_controller,
            workspace,
            last_fork,
            last_hibernation,
        }
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
        let initial = self.run_prompt_prefill(
            &prompt_tokens[..replay_prefill_boundary],
            state,
            activations,
            attention_shape,
            prepared,
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
            complete: continuation.complete,
            workspace: initial.workspace,
            // Report the primary replay segment. SessionReplay records the sequential tail.
            prefill_method: initial.prefill_method,
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
        self.discard_session(session)?;
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
            prefill.prefill_method,
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
                sequential_logits: false,
            });
        }
        let verifier_positions = drafted.tokens().len() + 1;
        self.ensure_correctable_verifier_capacity(verify_activations, options, verifier_positions)?;
        let (outcome, sequential_logits) = self.speculative_round(
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
            sequential_logits,
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
            self.run_graph_forward(state, activations, attention_shape)?;
        } else {
            self.forward(state, activations, attention_shape)?;
            if !host_sampling {
                self.enqueue_sample(activations)?;
            }
        }
        Ok(())
    }

    fn run_graph_forward(
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
        let range = state.prepare_append(&mut self.backend, self.model.config.n_layer, 1, 32)?;
        if state.graph_revision != Some(self.graph_revision_token(state)) {
            self.capture_decode_graph(state, activations, attention_shape)?;
        } else {
            self.prepare_kv_state_views(state, range, attention_shape)?;
        }
        self.backend.replay_decode_graph()?;
        state.commit_append(range)?;
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
        produced_len: usize,
        adaptive_controller: &mut Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: &mut Option<CorrectableController>,
        round_duration: Duration,
        correctable_proposed: usize,
        correctable_accepted: usize,
        correctable_overlap: f64,
        correctable_overlap_proposals: usize,
    ) -> Result<(), RuntimeError> {
        if let (Some((verifier_positions, controller_duration)), Some(_)) =
            (adaptive_round, std::num::NonZeroUsize::new(committed_len))
        {
            let produced_tokens =
                std::num::NonZeroUsize::new(produced_len).ok_or(RuntimeError::SizeOverflow)?;
            adaptive_controller
                .as_mut()
                .expect("adaptive rounds have a controller")
                .observe(crate::adaptive_draft::AdaptiveObservation {
                    verifier_positions,
                    produced_tokens,
                    wall_duration: round_duration,
                    controller_duration,
                })
                .expect("runtime verifier widths are in [1, 8]");
        }
        if let (Some((plan, verifier_positions, controller_duration)), Some(_)) = (
            correctable_round,
            std::num::NonZeroUsize::new(committed_len),
        ) {
            let produced_tokens =
                std::num::NonZeroUsize::new(produced_len).ok_or(RuntimeError::SizeOverflow)?;
            correctable_controller
                .as_mut()
                .expect("correctable rounds have a controller")
                .observe(CorrectableObservation {
                    plan,
                    verifier_positions,
                    produced_tokens,
                    proposed_tokens: correctable_proposed,
                    accepted_tokens: correctable_accepted,
                    overlap_sum: correctable_overlap,
                    overlap_proposals: correctable_overlap_proposals,
                    wall_duration: round_duration,
                    controller_duration,
                })?;
        }
        Ok(())
    }

    fn generation_produced_tokens(
        outcome: Option<&SpeculationOutcome>,
        emitted: usize,
    ) -> Result<usize, RuntimeError> {
        outcome.map_or(Ok(emitted), |outcome| {
            outcome
                .accepted
                .checked_add(1)
                .ok_or(RuntimeError::SizeOverflow)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_generation_tokens<F, C, S>(
        &self,
        committed: &mut Vec<u32>,
        tokens: &mut Vec<u32>,
        max_tokens: usize,
        eos: Option<u32>,
        on_token: &mut F,
        cancelled: &mut C,
        stopped: &mut S,
    ) -> Result<Option<GenerationTermination>, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        for token in committed.drain(..) {
            emit(&self.model, token, tokens, on_token)?;
            if let Some(termination) = Self::poll_generation_termination(cancelled, stopped) {
                return Ok(Some(termination));
            }
            if tokens.len() >= max_tokens || Some(token) == eos {
                break;
            }
        }
        Ok(None)
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
        termination: GenerationTermination,
        mirostat: Option<MirostatState>,
        adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: Option<CorrectableController>,
    ) {
        if termination == GenerationTermination::Cancelled {
            session.invalidate();
            return;
        }
        let mut evaluated_tokens = prompt_tokens.to_vec();
        evaluated_tokens.extend_from_slice(tokens);
        evaluated_tokens.truncate(state.position.min(evaluated_tokens.len()));
        session.retained_logits_position = Some(state.position);
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
    fn run_generation_decode<F, C, S>(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: GenerateOptions,
        prefill_start: Instant,
        prefill_duration: Duration,
        prefill: PrefillExecution,
        attention_shape: AttentionShape,
        state: &mut Option<KvState<B>>,
        activations: &mut Option<Activations<B>>,
        reuse_class: SessionReuseClass,
        reused_tokens: usize,
        cached_tokens: usize,
        restored_common: usize,
        common_tokens: usize,
        replay_prefill_boundary: usize,
        mut on_token: F,
        mut cancelled: C,
        mut stopped: S,
        mut verify_activations: Vec<Option<VerifyActivations<B>>>,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        let state_ref = state.as_mut().ok_or(RuntimeError::SizeOverflow)?;
        let activations_ref = activations.as_mut().ok_or(RuntimeError::SizeOverflow)?;
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
            state_ref,
            activations_ref,
            rng,
            &mut row,
            &mut logits,
            &context,
            &mut tokens,
            &mut on_token,
            &mut cancelled,
            &mut stopped,
            attention_shape,
        )?;
        let GenerationInitialization {
            termination: initial_termination,
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
        let termination = match initial_termination {
            Some(termination) => termination,
            None => self.run_generation_iterations(
                prompt_tokens,
                &options,
                attention_shape,
                state_ref,
                activations_ref,
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
                &mut stopped,
                eos,
                profile_target,
                &mut profile_start,
                &mut profile_steps,
                &mut decode_profile,
            )?,
        };
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
        let emitted_tokens = tokens.len();
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
            termination,
            mirostat,
            adaptive_controller,
            correctable_controller,
        )?;
        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.to_vec(),
            tokens,
            termination,
            stats: GenerationStats {
                prompt_tokens: prompt_tokens.len(),
                emitted_tokens,
                decode_evaluations,
                prefill_duration,
                ttft_duration: Some(ttft_duration),
                decode_duration,
                verify_duration,
                cancelled: termination == GenerationTermination::Cancelled,
                speculation,
                prefill_method: prefill.prefill_method,
                prefill_workspace: prefill.workspace,
            },
            logits,
            decode_profile,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation_iterations<F, C, S>(
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
        stopped: &mut S,
        eos: Option<u32>,
        profile_target: usize,
        profile_start: &mut Option<Instant>,
        profile_steps: &mut usize,
        decode_profile: &mut Option<DecodeProfile>,
    ) -> Result<GenerationTermination, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        while tokens.len() < options.max_tokens && Some(first_token(tokens)) != eos {
            if let Some(termination) = Self::poll_generation_termination(cancelled, stopped) {
                return Ok(termination);
            }
            if let Some(termination) = self.run_generation_round(
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
                cancelled,
                stopped,
            )? {
                return Ok(termination);
            }
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
            if let Some(termination) = Self::poll_generation_termination(cancelled, stopped) {
                return Ok(termination);
            }
        }
        Ok(GenerationTermination::Completed)
    }

    fn poll_generation_termination<C, S>(
        cancelled: &mut C,
        stopped: &mut S,
    ) -> Option<GenerationTermination>
    where
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        if stopped() {
            return Some(GenerationTermination::Stopped);
        }
        cancelled().then_some(GenerationTermination::Cancelled)
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
        state: &mut Option<KvState<B>>,
        activations: &mut Option<Activations<B>>,
        reuse_class: SessionReuseClass,
        reused_tokens: usize,
        cached_tokens: usize,
        restored_common: usize,
        common_tokens: usize,
        replay_prefill_boundary: usize,
        termination: GenerationTermination,
        mirostat: Option<MirostatState>,
        adaptive_controller: Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: Option<CorrectableController>,
    ) -> Result<(), RuntimeError> {
        self.backend.synchronize()?;
        session.last_replay = SessionReplay::for_prompt(
            reuse_class,
            cached_tokens.max(session.restored_tokens.len()),
            reused_tokens,
            restored_common.max(common_tokens),
            prompt_tokens.len(),
        );
        if termination == GenerationTermination::Cancelled {
            self.retire_decode_graph()?;
        }
        let state = state.take().ok_or(RuntimeError::SizeOverflow)?;
        let activations = activations.take().ok_or(RuntimeError::SizeOverflow)?;
        Self::commit_generation_session(
            session,
            prompt_tokens,
            tokens,
            state,
            activations,
            reuse_class,
            replay_prefill_boundary,
            termination,
            mirostat,
            adaptive_controller,
            correctable_controller,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation_round<F, C, S>(
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
        stopped: &mut S,
    ) -> Result<Option<GenerationTermination>, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        let decode_start = Instant::now();
        let start_position = state.position;
        let emitted_before = tokens.len();
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
        let termination = self.emit_generation_round_tokens(
            committed,
            tokens,
            options.max_tokens,
            self.model.tokenizer.eos_token(),
            on_token,
            cancelled,
            stopped,
        )?;
        let measured_duration = if termination == Some(GenerationTermination::Cancelled) {
            round_duration
        } else {
            self.finish_generation_round(
                state,
                activations,
                verify_activations,
                drafted.tokens().len() + 1,
                &step,
                start_position,
                emitted_before,
                tokens,
                tokens.len(),
                adaptive_round,
                correctable_round,
                adaptive_controller,
                correctable_controller,
                round_duration,
                Instant::now(),
            )?
        };
        *decode_duration += measured_duration;
        Ok(termination)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_generation_round(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        verifier_positions: usize,
        step: &GenerationStep,
        start_position: usize,
        emitted_before: usize,
        tokens: &[u32],
        token_count: usize,
        adaptive_round: Option<(std::num::NonZeroUsize, Duration)>,
        correctable_round: Option<(
            Option<crate::CorrectablePlan>,
            std::num::NonZeroUsize,
            Duration,
        )>,
        adaptive_controller: &mut Option<crate::adaptive_draft::AdaptiveController>,
        correctable_controller: &mut Option<CorrectableController>,
        round_duration: Duration,
        finalization_start: Instant,
    ) -> Result<Duration, RuntimeError> {
        let emitted = token_count.saturating_sub(emitted_before);
        let produced = Self::generation_produced_tokens(step.outcome.as_ref(), emitted)?;
        let (correctable_proposed, correctable_accepted, correctable_overlap, overlap_proposals) =
            Self::generation_outcome_counts(step.outcome.as_ref());
        self.finalize_generation_boundary(
            state,
            activations,
            verify_activations,
            verifier_positions,
            step,
            start_position,
            emitted_before,
            tokens,
            token_count,
            emitted,
        )?;
        let measured_duration = round_duration + finalization_start.elapsed();
        Self::observe_generation_round(
            adaptive_round,
            correctable_round,
            emitted,
            produced,
            adaptive_controller,
            correctable_controller,
            measured_duration,
            correctable_proposed,
            correctable_accepted,
            correctable_overlap,
            overlap_proposals,
        )?;
        Ok(measured_duration)
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_generation_boundary(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        verifier_positions: usize,
        step: &GenerationStep,
        start_position: usize,
        emitted_before: usize,
        tokens: &[u32],
        token_count: usize,
        emitted: usize,
    ) -> Result<(), RuntimeError> {
        let discarded =
            Self::truncate_generation_round(state, start_position, emitted_before, token_count)?;
        if discarded && emitted > 0 {
            let token = *tokens.last().ok_or(RuntimeError::SizeOverflow)?;
            self.backend.write_u32(&mut activations.sampled, &[token])?;
        }
        let restored_logits = self.restore_finished_logits_if_needed(
            activations,
            verify_activations,
            verifier_positions,
            step,
            discarded,
            emitted,
        )?;
        if discarded || restored_logits {
            self.backend.synchronize()?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_finished_logits_if_needed(
        &mut self,
        activations: &mut Activations<B>,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        verifier_positions: usize,
        step: &GenerationStep,
        discarded: bool,
        emitted: usize,
    ) -> Result<bool, RuntimeError> {
        match emitted.checked_sub(1) {
            Some(retained_row) => self.restore_finished_logits(
                activations,
                verify_activations,
                verifier_positions,
                step,
                discarded,
                retained_row,
            ),
            None => Ok(false),
        }
    }

    fn generation_outcome_counts(
        outcome: Option<&SpeculationOutcome>,
    ) -> (usize, usize, f64, usize) {
        outcome.map_or((0, 0, 0.0, 0), |outcome| {
            (
                outcome.proposed,
                outcome.accepted,
                outcome.overlap_sum,
                outcome.overlap_proposals,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_finished_logits(
        &mut self,
        activations: &mut Activations<B>,
        verify_activations: &mut [Option<VerifyActivations<B>>],
        verifier_positions: usize,
        step: &GenerationStep,
        discarded: bool,
        retained_row: usize,
    ) -> Result<bool, RuntimeError> {
        let verify_source = step
            .outcome
            .as_ref()
            .filter(|outcome| outcome.verified_positions > 0)
            .and_then(|_| verify_activations.get(verifier_positions))
            .and_then(Option::as_ref);
        let needs_restore = if step.sequential_logits {
            discarded
        } else {
            verify_source.is_some()
        };
        if needs_restore {
            self.restore_generation_logits(
                activations,
                step.sequential_logits,
                verify_source,
                retained_row,
            )?;
        }
        Ok(needs_restore)
    }

    fn restore_generation_logits(
        &mut self,
        activations: &mut Activations<B>,
        sequential_logits: bool,
        verify_activations: Option<&VerifyActivations<B>>,
        row: usize,
    ) -> Result<(), RuntimeError> {
        if sequential_logits {
            let scratch = self
                .speculative_logits
                .as_ref()
                .ok_or(RuntimeError::SizeOverflow)?;
            if row >= scratch.rows {
                return Err(RuntimeError::SizeOverflow);
            }
            self.backend.copy_f32_row(
                &scratch.buffer,
                row,
                self.model.config.vocab_size,
                &mut activations.logits,
            )?;
        } else if let Some(verify_activations) = verify_activations {
            self.backend.copy_f32_row(
                &verify_activations.logits,
                row,
                self.model.config.vocab_size,
                &mut activations.logits,
            )?;
        }
        Ok(())
    }

    fn truncate_generation_round(
        state: &mut KvState<B>,
        start_position: usize,
        emitted_before: usize,
        token_count: usize,
    ) -> Result<bool, RuntimeError> {
        let emitted = token_count.saturating_sub(emitted_before);
        let target = start_position
            .checked_add(emitted)
            .ok_or(RuntimeError::SizeOverflow)?;
        if state.position > target {
            state.truncate(target)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn speculative_logit_rows(options: &GenerateOptions) -> Result<usize, RuntimeError> {
        let max_rows = options.max_tokens.saturating_sub(1);
        if max_rows == 0 || options.speculation == Speculation::Disabled {
            return Ok(0);
        }
        let maximum_positions = match options.speculation {
            Speculation::Suffix(drafter) => drafter
                .proposal()
                .get()
                .checked_add(1)
                .ok_or(RuntimeError::SizeOverflow)?,
            Speculation::Adaptive(drafter) => drafter.maximum_verifier_positions(),
            Speculation::Correctable(drafter) => drafter.maximum_verifier_positions(),
            Speculation::Disabled => return Ok(0),
        };
        Ok(maximum_positions.min(max_rows))
    }

    fn ensure_speculative_logits(&mut self, options: &GenerateOptions) -> Result<(), RuntimeError> {
        let rows = Self::speculative_logit_rows(options)?;
        let current_rows = self
            .speculative_logits
            .as_ref()
            .map_or(0, |scratch| scratch.rows);
        if rows <= current_rows {
            return Ok(());
        }
        let elements = rows
            .checked_mul(self.model.config.vocab_size)
            .ok_or(RuntimeError::SizeOverflow)?;
        let buffer = self
            .backend
            .allocate_classified(BufferLayout::f32(elements)?, MemoryClass::BackendScratch)?;
        self.speculative_logits = Some(SpeculativeLogitScratch { buffer, rows });
        Ok(())
    }

    fn prepare_generation_session(
        &mut self,
        session: &mut GenerationSession<B>,
        prompt_tokens: &[u32],
        options: &GenerateOptions,
    ) -> Result<(AttentionShape, GenerationPreparation<B>), RuntimeError> {
        let attention_shape = self
            .validate_generation_request(prompt_tokens, options)
            .map_err(|error| error.with_session_failure_effect(SessionFailureEffect::Unchanged))?;
        let preparation = self
            .prepare_generation_state(session, prompt_tokens, options, attention_shape)
            .map_err(|error| error.with_session_failure_effect(SessionFailureEffect::Unchanged))?;
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

    #[allow(clippy::too_many_arguments)]
    fn emit_generation_round_tokens<F, C, S>(
        &mut self,
        committed: &mut Vec<u32>,
        tokens: &mut Vec<u32>,
        max_tokens: usize,
        eos: Option<u32>,
        on_token: &mut F,
        cancelled: &mut C,
        stopped: &mut S,
    ) -> Result<Option<GenerationTermination>, RuntimeError>
    where
        F: FnMut(&GeneratedToken) -> Result<(), RuntimeError>,
        C: FnMut() -> bool,
        S: FnMut() -> bool,
    {
        self.emit_generation_tokens(
            committed, tokens, max_tokens, eos, on_token, cancelled, stopped,
        )
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

    fn begin_generation_profile_if_active(
        &mut self,
        target: usize,
        first_token: u32,
        eos: Option<u32>,
        termination: Option<GenerationTermination>,
    ) -> Result<Option<Instant>, RuntimeError> {
        match termination {
            Some(_) => Ok(None),
            None => self.begin_generation_profile(target, first_token, eos),
        }
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
        self.prepare_prompt_prefill_with_numerics(
            prefill_tokens,
            context_tokens,
            chunk_tokens,
            PrefillNumerics::BackendPreferred,
        )
    }

    fn prepare_prompt_prefill_with_numerics(
        &mut self,
        prefill_tokens: usize,
        context_tokens: usize,
        chunk_tokens: usize,
        numerics: PrefillNumerics,
    ) -> Result<PreparedPrefill<B>, RuntimeError> {
        if prefill_tokens == 0 {
            return Ok(PreparedPrefill::Reused);
        }
        if self.backend.prefill_method() == PrefillMethod::SequentialDecode {
            return Ok(PreparedPrefill::Sequential);
        }
        let (chunk_tokens, plan, allocated) = self.allocate_prompt_prefill_with_numerics(
            prefill_tokens,
            context_tokens,
            chunk_tokens,
            numerics,
        )?;
        let AllocatedPromptPrefill {
            mut workspace,
            full,
            tail,
            remainder,
        } = allocated;
        self.add_prompt_prefill_activation_bytes(&mut workspace, chunk_tokens, remainder)?;
        Ok(PreparedPrefill::Chunked {
            plan,
            chunk_tokens,
            full,
            tail,
            workspace,
        })
    }

    fn allocate_prompt_prefill_with_numerics(
        &mut self,
        prefill_tokens: usize,
        context_tokens: usize,
        chunk_tokens: usize,
        numerics: PrefillNumerics,
    ) -> Result<(usize, PrefillPlan, AllocatedPromptPrefill<B>), RuntimeError> {
        let (chunk_tokens, plan) =
            self.prompt_prefill_plan(prefill_tokens, context_tokens, chunk_tokens)?;
        let plan = plan.with_numerics(numerics);
        let allocation = self.allocate_prompt_prefill(plan, chunk_tokens, prefill_tokens)?;
        Ok((chunk_tokens, plan, allocation))
    }

    fn add_prompt_prefill_activation_bytes(
        &self,
        workspace: &mut PrefillWorkspace,
        chunk_tokens: usize,
        remainder: usize,
    ) -> Result<(), RuntimeError> {
        let activation_bytes = self.prompt_prefill_bytes(chunk_tokens, remainder)?;
        workspace.batch_activation_bytes = activation_bytes;
        workspace.total_bytes = workspace
            .total_bytes
            .checked_add(activation_bytes)
            .ok_or(RuntimeError::SizeOverflow)?;
        Ok(())
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
        self.run_prompt_prefill_bounded(
            prompt_tokens,
            state,
            activations,
            attention_shape,
            prepared,
            usize::MAX,
            &mut cancelled,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prompt_prefill_bounded<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        prepared: &mut PreparedPrefill<B>,
        budget: usize,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        let prefill_method = self.selected_prefill_method(prepared);
        let execution = match prepared {
            PreparedPrefill::Reused => Ok(PrefillExecution {
                processed_tokens: 0,
                cancelled: false,
                complete: true,
                workspace: PrefillWorkspace::default(),
                prefill_method: PrefillMethod::Reused,
            }),
            PreparedPrefill::Sequential => self.run_sequential_prefill_bounded(
                prompt_tokens,
                state,
                activations,
                attention_shape,
                budget,
                cancelled,
            ),
            PreparedPrefill::Chunked {
                plan,
                chunk_tokens,
                full,
                tail,
                workspace,
            } => self.run_chunked_prefill_bounded(
                prompt_tokens,
                state,
                activations,
                attention_shape,
                plan,
                chunk_tokens,
                full,
                tail,
                workspace,
                budget,
                cancelled,
            ),
        }?;
        Ok(PrefillExecution {
            prefill_method,
            ..execution
        })
    }

    fn selected_prefill_method(&self, prepared: &PreparedPrefill<B>) -> PrefillMethod {
        match prepared {
            PreparedPrefill::Chunked { .. } => self.backend.prefill_method(),
            PreparedPrefill::Reused => PrefillMethod::Reused,
            PreparedPrefill::Sequential => PrefillMethod::SequentialDecode,
        }
    }

    fn run_sequential_prefill_bounded<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        budget: usize,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        let last_quantum_position = budget.min(prompt_tokens.len()).saturating_sub(1);
        let mut processed = 0_usize;
        while processed < prompt_tokens.len() && processed < budget {
            if cancelled() {
                return Ok(PrefillExecution {
                    processed_tokens: processed,
                    cancelled: true,
                    complete: false,
                    workspace: PrefillWorkspace::default(),
                    prefill_method: PrefillMethod::SequentialDecode,
                });
            }
            let token = prompt_tokens[processed];
            let output = SequentialPrefillOutput::for_position(processed, last_quantum_position);
            self.run_sequential_prefill_token(token, state, activations, attention_shape, output)?;
            processed = processed.checked_add(1).ok_or(RuntimeError::SizeOverflow)?;
        }
        Ok(PrefillExecution {
            processed_tokens: processed,
            cancelled: false,
            complete: processed == prompt_tokens.len(),
            workspace: PrefillWorkspace::default(),
            prefill_method: PrefillMethod::SequentialDecode,
        })
    }

    fn run_sequential_prefill_token(
        &mut self,
        token: u32,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        output: SequentialPrefillOutput,
    ) -> Result<(), RuntimeError> {
        self.backend.write_u32(&mut activations.sampled, &[token])?;
        match output {
            SequentialPrefillOutput::Hidden => {
                self.forward_hidden(state, activations, attention_shape)
            }
            SequentialPrefillOutput::Logits => self.forward(state, activations, attention_shape),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chunked_prefill_bounded<C>(
        &mut self,
        prompt_tokens: &[u32],
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        plan: &PrefillPlan,
        chunk_tokens: &usize,
        full: &mut PrefillActivations<B>,
        tail: &mut Option<PrefillActivations<B>>,
        workspace: &mut PrefillWorkspace,
        budget: usize,
        cancelled: &mut C,
    ) -> Result<PrefillExecution, RuntimeError>
    where
        C: FnMut() -> bool,
    {
        self.backend.prepare_prefill(*plan)?;
        let base_position = state.position;
        let mut processed = 0_usize;
        while processed < prompt_tokens.len() {
            let remaining = prompt_tokens.len() - processed;
            let Some(count) =
                Self::bounded_prefill_chunk(*chunk_tokens, remaining, budget, processed)?
            else {
                break;
            };
            if cancelled() {
                return Ok(PrefillExecution {
                    processed_tokens: processed,
                    cancelled: true,
                    complete: false,
                    workspace: *workspace,
                    prefill_method: self.backend.prefill_method(),
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
                count,
                full,
                tail,
            )?;
        }
        let complete = self.finish_chunked_prefill(activations, processed, prompt_tokens.len())?;
        Ok(PrefillExecution {
            processed_tokens: processed,
            cancelled: false,
            complete,
            workspace: *workspace,
            prefill_method: self.backend.prefill_method(),
        })
    }

    fn finish_chunked_prefill(
        &mut self,
        activations: &mut Activations<B>,
        processed: usize,
        prompt_tokens: usize,
    ) -> Result<bool, RuntimeError> {
        let complete = processed == prompt_tokens;
        if complete {
            self.finish_prefill_logits(activations)?;
        }
        Ok(complete)
    }

    fn bounded_prefill_chunk(
        chunk_tokens: usize,
        remaining: usize,
        budget: usize,
        processed: usize,
    ) -> Result<Option<usize>, RuntimeError> {
        let count = chunk_tokens.min(remaining);
        if count <= budget.saturating_sub(processed) {
            return Ok(Some(count));
        }
        if processed == 0 {
            return Err(RuntimeError::PrefillBudgetTooSmall {
                budget,
                chunk: count,
            });
        }
        Ok(None)
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
        count: usize,
        full: &mut PrefillActivations<B>,
        tail: &mut Option<PrefillActivations<B>>,
    ) -> Result<(), RuntimeError> {
        let batch = Self::select_prefill_batch(count, *chunk_tokens, full, tail)?;
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
        debug_assert_eq!(
            state.position,
            base_position
                .checked_add(*processed)
                .ok_or(RuntimeError::SizeOverflow)?
        );
        self.copy_prefill_hidden_if_complete(
            batch,
            count,
            *processed,
            prompt_tokens.len(),
            activations,
        )?;
        Ok(())
    }

    fn select_prefill_batch<'a>(
        count: usize,
        chunk_tokens: usize,
        full: &'a mut PrefillActivations<B>,
        tail: &'a mut Option<PrefillActivations<B>>,
    ) -> Result<&'a mut PrefillActivations<B>, RuntimeError> {
        if count == chunk_tokens {
            return Ok(full);
        }
        tail.as_mut().ok_or_else(|| {
            BackendError::operation("select prefill tail", "tail activation storage is missing")
                .into()
        })
    }

    fn copy_prefill_hidden_if_complete(
        &mut self,
        batch: &PrefillActivations<B>,
        count: usize,
        processed: usize,
        prompt_tokens: usize,
        activations: &mut Activations<B>,
    ) -> Result<(), RuntimeError> {
        if processed == prompt_tokens {
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
            state,
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
        state: &mut KvState<B>,
        activations: &mut PrefillActivations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        if start_position != state.position {
            return Err(BackendError::operation(
                "prepare prefill KV append",
                "the chunk start does not match the committed position",
            )
            .into());
        }
        let range = state.prepare_append(
            &mut self.backend,
            self.model.config.n_layer,
            tokens.len(),
            tokens.len(),
        )?;
        self.prepare_kv_state_views(state, range, attention_shape)?;
        self.prepare_prefill_chunk_inputs(tokens, activations)?;
        let (hidden_shape, query_shape, key_shape, query_rope, key_rope) =
            self.prefill_chunk_shapes(tokens.len())?;
        self.forward_prefill_layers(
            tokens,
            start_position,
            state,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            query_rope,
            key_rope,
            range,
        )?;
        state.commit_append(range)?;
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
        state: &mut KvState<B>,
        activations: &mut PrefillActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        query_rope: RopeShape,
        key_rope: RopeShape,
        range: KvAppendRange,
    ) -> Result<(), RuntimeError> {
        let epsilon = self.model.config.rms_epsilon;
        let theta = self.model.config.rope_theta;
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            Self::forward_prefill_layer(
                &mut self.backend,
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
                tokens.len(),
                epsilon,
                theta,
                range,
                layer_index,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_layer(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvState<B>,
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
        range: KvAppendRange,
        layer_index: usize,
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
            range,
            layer_index,
        )?;
        Self::prefill_layer_ffn(backend, layer, activations, hidden_shape, tokens, epsilon)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_attention(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvState<B>,
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
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<(), RuntimeError> {
        let normalized = Self::prefill_layer_qkv(
            backend,
            layer,
            activations,
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
        Self::prefill_layer_attention_apply(
            backend,
            layer,
            state,
            activations,
            attention_shape,
            start_position,
            tokens,
            normalized,
            range,
            layer_index,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_qkv(
        backend: &mut B,
        layer: &DenseLayer<B>,
        activations: &mut PrefillActivations<B>,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        query_rope: RopeShape,
        key_rope: RopeShape,
        start_position: usize,
        tokens: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<bool, RuntimeError> {
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
        Self::prefill_layer_qk(
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
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_attention_apply(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvState<B>,
        activations: &mut PrefillActivations<B>,
        attention_shape: AttentionShape,
        start_position: usize,
        tokens: usize,
        normalized: bool,
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<(), RuntimeError> {
        let (query, key) = Self::prefill_layer_query_key(
            normalized,
            &activations.query,
            &activations.key,
            &activations.query_norm,
            &activations.key_norm,
        );
        let target = state.write_span(layer_index, range)?;
        backend.kv_append_chunk_span(
            key,
            &activations.value,
            target,
            attention_shape,
            start_position,
            tokens,
        )?;
        let spans = state.read_spans(layer_index)?;
        let cache = KvReadView::new(&spans)?;
        backend.attention_prefill_spans(
            query,
            cache,
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
                Self::prefill_layer_qk_rms(
                    backend,
                    activations,
                    query,
                    key,
                    query_shape,
                    key_shape,
                    query_rope,
                    key_rope,
                    start_position,
                    epsilon,
                    theta,
                )?;
                Ok(true)
            }
            QkNorm::Identity => {
                Self::prefill_layer_qk_rope(
                    backend,
                    activations,
                    query_rope,
                    key_rope,
                    start_position,
                    theta,
                )?;
                Ok(false)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_qk_rms(
        backend: &mut B,
        activations: &mut PrefillActivations<B>,
        query_weight: &B::Buffer,
        key_weight: &B::Buffer,
        query_shape: VectorShape,
        key_shape: VectorShape,
        query_rope: RopeShape,
        key_rope: RopeShape,
        start_position: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        backend.prefill_rms_norm_rope(
            &activations.query,
            query_weight,
            &mut activations.query_norm,
            query_shape,
            query_rope,
            start_position,
            epsilon,
            theta,
        )?;
        backend.prefill_rms_norm_rope(
            &activations.key,
            key_weight,
            &mut activations.key_norm,
            key_shape,
            key_rope,
            start_position,
            epsilon,
            theta,
        )?;
        Ok(())
    }

    fn prefill_layer_qk_rope(
        backend: &mut B,
        activations: &mut PrefillActivations<B>,
        query_rope: RopeShape,
        key_rope: RopeShape,
        start_position: usize,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        backend.rope(&mut activations.query, start_position, query_rope, theta)?;
        backend.rope(&mut activations.key, start_position, key_rope, theta)?;
        Ok(())
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
        self.batch_graph_signature = None;
        if state.position >= state.shape.max_context() {
            return Ok(());
        }
        let replacement = self.backend.clone_buffer(&state.device_position)?;
        let mut capture_position = std::mem::replace(&mut state.device_position, replacement);
        let result = self.capture_decode_graph_body(
            state,
            activations,
            attention_shape,
            &mut capture_position,
        );
        state.device_position = capture_position;
        if result.is_err() {
            self.retire_decode_graph()?;
        }
        result?;
        state.graph_revision = Some(self.graph_revision_token(state));
        Ok(())
    }

    fn capture_decode_graph_body(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        capture_position: &mut B::Buffer,
    ) -> Result<(), RuntimeError> {
        let range = self.with_kv_graph_preflight(|runtime| {
            runtime.prepare_decode_capture(state, activations, capture_position, attention_shape)
        })?;
        activations.retain_decode_graph_buffers(&mut self.backend)?;
        let capture_buffers = [&*capture_position];
        self.backend.retain_decode_graph_buffers(&capture_buffers)?;
        self.reserve_decode_graph_generation()?;
        self.backend.begin_decode_graph()?;
        let body = self.capture_decode_graph_work(
            state,
            activations,
            attention_shape,
            capture_position,
            range,
        );
        let end = self.backend.end_decode_graph();
        body?;
        end?;
        Ok(())
    }

    fn prepare_decode_capture(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        capture_position: &mut B::Buffer,
        attention_shape: AttentionShape,
    ) -> Result<KvAppendRange, RuntimeError> {
        let device_position =
            u32::try_from(state.position).map_err(|_| RuntimeError::ContextCapacity {
                requested: state.position,
                capacity: u32::MAX as usize,
            })?;
        self.backend
            .write_u32(capture_position, &[device_position])?;
        let range = state.prepare_append(&mut self.backend, self.model.config.n_layer, 1, 32)?;
        self.prepare_kv_state_views(state, range, attention_shape)?;
        self.prepare_decode_attention_views(state, activations, attention_shape, capture_position)?;
        Ok(range)
    }

    fn prepare_kv_state_reads(
        &mut self,
        state: &KvState<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        let layers = self.model.weights.layers.len();
        for layer_index in 0..layers {
            let spans = state.read_spans(layer_index)?;
            let cache = KvReadView::new(&spans)?;
            self.backend.prepare_kv_read_view(cache, attention_shape)?;
        }
        Ok(())
    }

    fn prepare_kv_state_views(
        &mut self,
        state: &mut KvState<B>,
        range: KvAppendRange,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        let layers = self.model.weights.layers.len();
        for layer_index in 0..layers {
            let target = state.write_span(layer_index, range)?;
            self.backend
                .prepare_kv_write_span(target, attention_shape)?;
            let spans = state.read_spans(layer_index)?;
            let cache = KvReadView::new(&spans)?;
            self.backend.prepare_kv_read_view(cache, attention_shape)?;
        }
        Ok(())
    }

    fn prepare_decode_attention_views(
        &mut self,
        state: &KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        capture_position: &B::Buffer,
    ) -> Result<(), RuntimeError> {
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            let spans = state.read_spans(layer_index)?;
            let cache = KvReadView::new(&spans)?;
            let query = if matches!(layer.qk_norm, QkNorm::Rms { .. }) {
                &activations.query_norm
            } else {
                &activations.query
            };
            let row = AttentionDecodeRow {
                query,
                cache,
                output: &mut activations.attention,
                shape: attention_shape,
                position: Position::Device(capture_position),
            };
            self.backend
                .prepare_attention_decode_batch_spans(std::slice::from_ref(&row))?;
        }
        Ok(())
    }

    fn prepare_batch_kv_views(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        ranges: &[KvAppendRange],
    ) -> Result<(), RuntimeError> {
        for (input, range) in inputs.iter_mut().zip(ranges) {
            let state = input
                .session
                .state
                .as_mut()
                .ok_or(RuntimeError::BatchSessionMismatch)?;
            let shape = state.shape;
            self.prepare_kv_state_views(state, *range, shape)?;
        }
        Ok(())
    }

    fn with_kv_graph_preflight<T, F>(&mut self, work: F) -> Result<T, RuntimeError>
    where
        F: FnOnce(&mut Self) -> Result<T, RuntimeError>,
    {
        self.backend.begin_kv_graph_preflight()?;
        let result = work(self);
        let finish = self.backend.end_kv_graph_preflight(result.is_ok());
        match (result, finish) {
            (Err(error), _) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(error)) => Err(error.into()),
        }
    }

    fn capture_decode_graph_work(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        capture_position: &mut B::Buffer,
        range: KvAppendRange,
    ) -> Result<(), RuntimeError> {
        self.forward_at(
            state,
            activations,
            attention_shape,
            Position::Device(capture_position),
            range,
        )?;
        self.enqueue_sample(activations)?;
        self.backend.increment_u32(capture_position)?;
        Ok(())
    }

    fn forward(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        let (range, hidden_shape) = self.prepare_forward(state, activations, attention_shape)?;
        self.forward_at_output(activations, hidden_shape)?;
        state.commit_append(range)?;
        Ok(())
    }

    fn forward_hidden(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(), RuntimeError> {
        let (range, _) = self.prepare_forward(state, activations, attention_shape)?;
        state.commit_append(range)?;
        Ok(())
    }

    fn prepare_forward(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
    ) -> Result<(KvAppendRange, VectorShape), RuntimeError> {
        let position = state.position;
        let range = state.prepare_append(&mut self.backend, self.model.config.n_layer, 1, 32)?;
        self.prepare_kv_state_views(state, range, attention_shape)?;
        let hidden_shape = self.forward_at_hidden(
            state,
            activations,
            attention_shape,
            Position::Host(position),
            range,
        )?;
        Ok((range, hidden_shape))
    }

    fn forward_at(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        position: Position<'_, B::Buffer>,
        range: KvAppendRange,
    ) -> Result<(), RuntimeError> {
        let hidden_shape =
            self.forward_at_hidden(state, activations, attention_shape, position, range)?;
        self.forward_at_output(activations, hidden_shape)
    }

    fn forward_at_hidden(
        &mut self,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        position: Position<'_, B::Buffer>,
        range: KvAppendRange,
    ) -> Result<VectorShape, RuntimeError> {
        let epsilon = self.model.config.rms_epsilon;
        let theta = self.model.config.rope_theta;
        let (hidden_shape, query_shape, key_shape) =
            self.forward_at_input(activations, position)?;
        self.forward_at_layers(
            state,
            activations,
            attention_shape,
            hidden_shape,
            query_shape,
            key_shape,
            position,
            epsilon,
            theta,
            range,
        )?;
        Ok(hidden_shape)
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
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
    ) -> Result<(), RuntimeError> {
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            Self::forward_layer(
                &mut self.backend,
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
                range,
                layer_index,
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

    /// Advances each retained session by one shared decode pass.
    ///
    /// Weight matrices execute over position-major request rows. Attention and
    /// sampling remain isolated per request. The method rejects features whose
    /// state is not represented by [`GenerationSession`].
    pub fn generate_session_batch_token(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
    ) -> Result<Vec<GeneratedToken>, RuntimeError> {
        if let Err(error) = self.validate_batch_sessions(inputs) {
            return Err(error.with_session_failure_effect(SessionFailureEffect::Unchanged));
        }
        for input in inputs.iter_mut() {
            input.session.retained_logits_position = None;
        }
        let tokens = match self.generate_session_batch_token_inner(inputs) {
            Ok(tokens) => tokens,
            Err(error) => return Err(self.quarantine_batch_failure(inputs, error)),
        };
        for input in inputs {
            input.session.retained_logits_position = Some(input.session.evaluated_tokens.len());
        }
        Ok(tokens)
    }

    fn quarantine_batch_failure(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        error: RuntimeError,
    ) -> RuntimeError {
        self.batch_graph_signature = None;
        let retirement = self.retire_decode_graph();
        if let Err(retirement) = retirement {
            return retirement.with_forced_session_failure_effect(SessionFailureEffect::Quarantine);
        }
        let cleanup = Self::discard_batch_pending(inputs);
        let error = match cleanup {
            Ok(()) => error,
            Err(cleanup) => cleanup.into(),
        };
        error.with_forced_session_failure_effect(SessionFailureEffect::Quarantine)
    }

    fn generate_session_batch_token_inner(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
    ) -> Result<Vec<GeneratedToken>, RuntimeError> {
        self.prepare_batch_appends(inputs, 1)?;
        let signature = self.assign_batch_signature(inputs)?;
        let width = inputs.len();
        let mut activations = match self.batch_activations.remove(&width) {
            Some(activations) => activations,
            None => VerifyActivations::new(&mut self.backend, &self.model.config, width)?,
        };
        let result = if self.batch_graph_signature.as_ref() == Some(&signature) {
            self.replay_batch_token(inputs, &mut activations)
        } else {
            self.run_and_capture_batch_token(inputs, &mut activations)
        };
        self.batch_activations.insert(width, activations);
        result
    }

    fn run_and_capture_batch_token(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
    ) -> Result<Vec<GeneratedToken>, RuntimeError> {
        let result = self.run_batch_token(inputs, activations);
        if !self.backend.decode_graph_supported()
            || inputs
                .iter()
                .any(|input| input.options.decode_execution == DecodeExecution::Eager)
        {
            return result;
        }
        let Ok(tokens) = result.as_ref() else {
            return result;
        };
        let can_capture = inputs.iter().all(|input| {
            input
                .session
                .state
                .as_ref()
                .is_some_and(|state| state.position < state.shape.max_context())
        });
        if !can_capture {
            self.batch_graph_signature = None;
            return result;
        }
        self.capture_batch_graph(inputs, activations, tokens)?;
        self.batch_graph_signature = Some(self.assign_batch_signature(inputs)?);
        result
    }

    fn assign_batch_signature(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
    ) -> Result<Vec<u64>, RuntimeError> {
        let mut signature = Vec::with_capacity(inputs.len());
        for input in inputs {
            let identity = match input.session.batch_identity {
                Some(identity) => identity,
                None => {
                    let identity = self.next_batch_identity;
                    self.next_batch_identity = self
                        .next_batch_identity
                        .checked_add(1)
                        .ok_or(RuntimeError::SizeOverflow)?;
                    input.session.batch_identity = Some(identity);
                    identity
                }
            };
            signature.push(identity);
            let state = input
                .session
                .state
                .as_ref()
                .ok_or(RuntimeError::BatchSessionMismatch)?;
            signature.push(state.revision());
            signature.push(
                u64::try_from(state.shape.max_context()).map_err(|_| RuntimeError::SizeOverflow)?,
            );
        }
        Ok(signature)
    }

    /// Returns true when a session can enter the shared decode path.
    pub fn batch_session_ready(
        &self,
        session: &GenerationSession<B>,
        transcript: &[u32],
        options: &GenerateOptions,
    ) -> bool {
        self.batch_options_supported(options)
            && Self::batch_state_matches(session, transcript)
            && self.backend.verify_supported()
    }

    fn batch_options_supported(&self, options: &GenerateOptions) -> bool {
        matches!(options.speculation, Speculation::Disabled)
            && options.output_constraint.is_none()
            && options.logit_capture == LogitCapture::Disabled
            && options.decode_profile == DecodeProfileMode::Disabled
    }

    fn batch_state_matches(session: &GenerationSession<B>, transcript: &[u32]) -> bool {
        let Some(state) = session.state.as_ref() else {
            return false;
        };
        session.activations.is_some()
            && state.position.checked_add(1) == Some(transcript.len())
            && session.evaluated_tokens == transcript[..state.position]
            && state.position < state.shape.max_context()
    }

    fn validate_batch_sessions(&self, inputs: &[BatchSession<'_, B>]) -> Result<(), RuntimeError> {
        if inputs.is_empty() {
            return Err(RuntimeError::BatchUnavailable("the batch is empty"));
        }
        Self::validate_batch_width(inputs.len(), self.backend.max_batch_size().get())?;
        if !self.backend.verify_supported() {
            return Err(RuntimeError::BatchUnavailable(
                "the backend lacks position-major matrix operations",
            ));
        }
        for input in inputs {
            if !self.batch_options_supported(input.options) {
                return Err(RuntimeError::BatchUnavailable(
                    "the request uses speculation, constrained output, or diagnostics",
                ));
            }
            if !Self::batch_state_matches(input.session, input.transcript) {
                return Err(RuntimeError::BatchSessionMismatch);
            }
        }
        Ok(())
    }

    fn validate_batch_width(requested: usize, maximum: usize) -> Result<(), RuntimeError> {
        if requested > maximum {
            return Err(RuntimeError::BatchSizeExceeded { requested, maximum });
        }
        Ok(())
    }

    fn run_batch_token(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
    ) -> Result<Vec<GeneratedToken>, RuntimeError> {
        let ranges = self.prepare_batch_appends(inputs, 1)?;
        self.prepare_batch_attention_views(inputs, None)?;
        self.forward_batch_token(inputs, activations, &ranges)?;
        self.copy_batch_logits_to_sessions(inputs, activations)?;
        self.read_batch_logits(activations)?;
        Self::commit_batch_appends(inputs, &ranges)?;
        self.sample_batch_rows(inputs, activations)
    }

    fn forward_batch_token(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        ranges: &[KvAppendRange],
    ) -> Result<(), RuntimeError> {
        let width = inputs.len();
        let (hidden_shape, query_shape, key_shape) = self.batch_vector_shapes(width)?;
        self.write_batch_inputs(inputs, activations)?;
        self.embed_batch_inputs(activations, width)?;
        self.run_batch_layers(
            inputs,
            activations,
            hidden_shape,
            query_shape,
            key_shape,
            ranges,
        )?;
        self.finish_batch_forward(activations, hidden_shape, width)
    }

    fn run_batch_layers(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        ranges: &[KvAppendRange],
    ) -> Result<(), RuntimeError> {
        let width = inputs.len();
        for layer_index in 0..self.model.weights.layers.len() {
            self.forward_batch_layer(
                inputs,
                activations,
                layer_index,
                hidden_shape,
                query_shape,
                key_shape,
                width,
                None,
                ranges,
            )?;
        }
        Ok(())
    }

    fn prepare_batch_appends(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        tokens: usize,
    ) -> Result<Vec<KvAppendRange>, RuntimeError> {
        let mut ranges = Vec::with_capacity(inputs.len());
        for input in inputs.iter_mut() {
            let state = input
                .session
                .state
                .as_mut()
                .ok_or(RuntimeError::BatchSessionMismatch)?;
            match state.prepare_append(&mut self.backend, self.model.config.n_layer, tokens, 32) {
                Ok(range) => ranges.push(range),
                Err(error) => return Err(error.into()),
            }
        }
        self.prepare_batch_kv_views(inputs, &ranges)?;
        Ok(ranges)
    }

    fn commit_batch_appends(
        inputs: &mut [BatchSession<'_, B>],
        ranges: &[KvAppendRange],
    ) -> Result<(), RuntimeError> {
        for (input, range) in inputs.iter_mut().zip(ranges) {
            let state = input
                .session
                .state
                .as_mut()
                .ok_or(RuntimeError::BatchSessionMismatch)?;
            state.commit_append(*range)?;
        }
        Ok(())
    }

    fn prepare_batch_attention_views(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        capture_positions: Option<&[B::Buffer]>,
    ) -> Result<(), RuntimeError> {
        self.prepare_batch_attention_layers(inputs, capture_positions)
    }

    fn prepare_batch_attention_layers(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        capture_positions: Option<&[B::Buffer]>,
    ) -> Result<(), RuntimeError> {
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            let normalized = matches!(layer.qk_norm, QkNorm::Rms { .. });
            let mut borrowed = Self::borrow_batch_attention_rows(
                inputs,
                layer_index,
                normalized,
                capture_positions,
            )?;
            let rows = Self::batch_attention_descriptors(&mut borrowed)?;
            self.backend.prepare_attention_decode_batch_spans(&rows)?;
        }
        Ok(())
    }

    fn borrow_batch_attention_rows<'a>(
        inputs: &'a mut [BatchSession<'_, B>],
        layer_index: usize,
        normalized: bool,
        capture_positions: Option<&'a [B::Buffer]>,
    ) -> Result<Vec<BatchAttentionRow<'a, B::Buffer>>, RuntimeError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(inputs.len())
            .map_err(|error| BackendError::operation("allocate batch attention rows", error))?;
        for (row, input) in inputs.iter_mut().enumerate() {
            let (state, single) = Self::batch_session_buffers(input)?;
            let position = Self::batch_attention_position(state.position, row, capture_positions)?;
            let query = if normalized {
                &single.query_norm
            } else {
                &single.query
            };
            rows.push(BatchAttentionRow {
                query,
                spans: state.read_spans(layer_index)?,
                output: &mut single.attention,
                shape: state.shape,
                position,
            });
        }
        Ok(rows)
    }

    fn batch_attention_position(
        host_position: usize,
        row: usize,
        capture_positions: Option<&[B::Buffer]>,
    ) -> Result<Position<'_, B::Buffer>, RuntimeError> {
        match capture_positions {
            Some(positions) => positions
                .get(row)
                .map(Position::Device)
                .ok_or(RuntimeError::BatchSessionMismatch),
            None => Ok(Position::Host(host_position)),
        }
    }

    fn batch_attention_descriptors<'a>(
        borrowed: &'a mut [BatchAttentionRow<'_, B::Buffer>],
    ) -> Result<Vec<AttentionDecodeRow<'a, B::Buffer>>, RuntimeError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(borrowed.len()).map_err(|error| {
            BackendError::operation("allocate batch attention descriptors", error)
        })?;
        for row in borrowed {
            rows.push(row.descriptor()?);
        }
        Ok(rows)
    }

    fn write_batch_inputs(
        &mut self,
        inputs: &[BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
    ) -> Result<(), RuntimeError> {
        activations.input_tokens.clear();
        activations
            .input_tokens
            .extend(inputs.iter().map(|input| first_token(input.transcript)));
        self.backend
            .write_u32(&mut activations.tokens, &activations.input_tokens)?;
        Ok(())
    }

    fn embed_batch_inputs(
        &mut self,
        activations: &mut VerifyActivations<B>,
        width: usize,
    ) -> Result<(), RuntimeError> {
        self.backend.embed_gather_batch(
            &self.model.weights.token_embedding.buffer,
            &activations.tokens,
            &mut activations.hidden,
            self.model.weights.token_embedding.shape,
            width,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_layer(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        layer_index: usize,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        width: usize,
        capture_positions: Option<&[B::Buffer]>,
        ranges: &[KvAppendRange],
    ) -> Result<(), RuntimeError> {
        let layer = &self.model.weights.layers[layer_index];
        let epsilon = self.model.config.rms_epsilon;
        self.backend.prefill_rms_norm(
            &activations.hidden,
            &layer.attention_norm,
            &mut activations.norm,
            hidden_shape,
            epsilon,
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
            width,
        )?;
        Self::forward_batch_attention_rows(
            &mut self.backend,
            layer,
            inputs,
            activations,
            layer_index,
            query_shape,
            key_shape,
            epsilon,
            self.model.config.rope_theta,
            self.model.config.n_embd,
            capture_positions,
            ranges,
        )?;
        self.backend.verify_gemv_residual(
            &layer.attention_output.buffer,
            &activations.attention,
            &activations.hidden,
            &mut activations.residual,
            layer.attention_output.shape,
            width,
        )?;
        Self::forward_batch_ffn(
            &mut self.backend,
            layer,
            activations,
            hidden_shape,
            width,
            self.model.config.n_ff,
            epsilon,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_attention_rows(
        backend: &mut B,
        layer: &DenseLayer<B>,
        inputs: &mut [BatchSession<'_, B>],
        batch: &mut VerifyActivations<B>,
        layer_index: usize,
        query_shape: VectorShape,
        key_shape: VectorShape,
        epsilon: f32,
        theta: f32,
        hidden_columns: usize,
        capture_positions: Option<&[B::Buffer]>,
        ranges: &[KvAppendRange],
    ) -> Result<(), RuntimeError> {
        let key_columns = key_shape.elements()?;
        for (row, input) in inputs.iter_mut().enumerate() {
            Self::append_batch_attention_row(
                backend,
                layer,
                input,
                batch,
                row,
                layer_index,
                query_shape,
                key_shape,
                key_columns,
                epsilon,
                theta,
                hidden_columns,
                capture_positions,
                ranges[row],
            )?;
        }
        let normalized = matches!(layer.qk_norm, QkNorm::Rms { .. });
        let mut borrowed =
            Self::borrow_batch_attention_rows(inputs, layer_index, normalized, capture_positions)?;
        let mut rows = Self::batch_attention_descriptors(&mut borrowed)?;
        backend.attention_decode_batch_spans(&mut rows)?;
        for (row, descriptor) in rows.iter().enumerate() {
            backend.write_f32_row(descriptor.output, &mut batch.attention, row, hidden_columns)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_batch_attention_row(
        backend: &mut B,
        layer: &DenseLayer<B>,
        input: &mut BatchSession<'_, B>,
        batch: &mut VerifyActivations<B>,
        row: usize,
        layer_index: usize,
        query_shape: VectorShape,
        key_shape: VectorShape,
        key_columns: usize,
        epsilon: f32,
        theta: f32,
        hidden_columns: usize,
        capture_positions: Option<&[B::Buffer]>,
        range: KvAppendRange,
    ) -> Result<(), RuntimeError> {
        let (state, single) = Self::batch_session_buffers(input)?;
        let position = Self::batch_attention_position(state.position, row, capture_positions)?;
        Self::copy_batch_qkv(backend, batch, single, row, hidden_columns, key_columns)?;
        backend.prepare_rope(position)?;
        Self::forward_layer_qk(
            backend,
            &layer.qk_norm,
            state,
            single,
            state.shape,
            query_shape,
            key_shape,
            position,
            epsilon,
            theta,
            range,
            layer_index,
        )?;
        Ok(())
    }

    fn batch_session_buffers<'a>(
        input: &'a mut BatchSession<'_, B>,
    ) -> Result<(&'a mut KvState<B>, &'a mut Activations<B>), RuntimeError> {
        let state = input
            .session
            .state
            .as_mut()
            .ok_or(RuntimeError::BatchSessionMismatch)?;
        let activations = input
            .session
            .activations
            .as_mut()
            .ok_or(RuntimeError::BatchSessionMismatch)?;
        Ok((state, activations))
    }

    fn copy_batch_qkv(
        backend: &mut B,
        batch: &VerifyActivations<B>,
        single: &mut Activations<B>,
        row: usize,
        hidden_columns: usize,
        key_columns: usize,
    ) -> Result<(), RuntimeError> {
        backend.copy_f32_row(&batch.query, row, hidden_columns, &mut single.query)?;
        backend.copy_f32_row(&batch.key, row, key_columns, &mut single.key)?;
        backend.copy_f32_row(&batch.value, row, key_columns, &mut single.value)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_ffn(
        backend: &mut B,
        layer: &DenseLayer<B>,
        activations: &mut VerifyActivations<B>,
        hidden_shape: VectorShape,
        width: usize,
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
            width,
        )?;
        backend.verify_swiglu(
            &activations.gate,
            &activations.up,
            &mut activations.ffn,
            n_ff,
            width,
        )?;
        backend.verify_gemv_residual_prepared(
            &layer.ffn_down.buffer,
            &activations.ffn,
            &activations.residual,
            &mut activations.hidden,
            layer.ffn_down.shape,
            width,
        )?;
        Ok(())
    }

    fn finish_batch_forward(
        &mut self,
        activations: &mut VerifyActivations<B>,
        hidden_shape: VectorShape,
        width: usize,
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
            width,
        )?;
        Ok(())
    }

    fn read_batch_logits(
        &mut self,
        activations: &mut VerifyActivations<B>,
    ) -> Result<(), RuntimeError> {
        self.backend
            .read_f32(&activations.logits, &mut activations.host_logits)?;
        Ok(())
    }

    fn replay_batch_token(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
    ) -> Result<Vec<GeneratedToken>, RuntimeError> {
        let ranges = self.prepare_batch_appends(inputs, 1)?;
        self.write_batch_inputs(inputs, activations)?;
        self.write_batch_device_positions(inputs)?;
        self.backend.replay_decode_graph()?;
        self.copy_batch_logits_to_sessions(inputs, activations)?;
        self.read_batch_logits(activations)?;
        Self::commit_batch_appends(inputs, &ranges)?;
        self.sample_batch_rows(inputs, activations)
    }

    fn copy_batch_logits_to_sessions(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
    ) -> Result<(), RuntimeError> {
        let vocab = self.model.config.vocab_size;
        for (row_index, input) in inputs.iter_mut().enumerate() {
            let session_activations = input
                .session
                .activations
                .as_mut()
                .ok_or(RuntimeError::BatchSessionMismatch)?;
            self.backend.copy_f32_row(
                &activations.logits,
                row_index,
                vocab,
                &mut session_activations.logits,
            )?;
        }
        Ok(())
    }

    fn capture_batch_graph(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        next_tokens: &[GeneratedToken],
    ) -> Result<(), RuntimeError> {
        self.batch_graph_signature = None;
        if inputs.iter().any(|input| {
            input
                .session
                .state
                .as_ref()
                .is_none_or(|state| state.position >= state.shape.max_context())
        }) {
            return Ok(());
        }
        self.stage_batch_graph_inputs(inputs, activations, next_tokens)?;
        let (ranges, shapes, mut capture_positions) = self.prepare_batch_graph_capture(inputs)?;
        let result = self.capture_batch_forward_prepared(
            inputs,
            activations,
            shapes,
            &ranges,
            &mut capture_positions,
        );
        self.restore_batch_capture_positions(inputs, capture_positions);
        result
    }

    fn prepare_batch_graph_capture(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
    ) -> Result<BatchGraphPreparation<B>, RuntimeError> {
        self.backend.begin_kv_graph_preflight()?;
        let mut capture_positions = Vec::new();
        let result = (|| {
            let ranges = self.prepare_batch_appends(inputs, 1)?;
            let shapes = self.batch_vector_shapes(inputs.len())?;
            capture_positions = self.take_batch_capture_positions(inputs)?;
            self.prepare_batch_attention_views(inputs, Some(&capture_positions))?;
            Ok((ranges, shapes))
        })();
        let finish = self.backend.end_kv_graph_preflight(result.is_ok());
        match (result, finish) {
            (Ok((ranges, shapes)), Ok(())) => Ok((ranges, shapes, capture_positions)),
            (Err(error), _) => {
                self.restore_batch_capture_positions(inputs, capture_positions);
                Err(error)
            }
            (Ok(_), Err(error)) => {
                self.restore_batch_capture_positions(inputs, capture_positions);
                Err(error.into())
            }
        }
    }

    fn stage_batch_graph_inputs(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        next_tokens: &[GeneratedToken],
    ) -> Result<(), RuntimeError> {
        activations.input_tokens.clear();
        activations
            .input_tokens
            .extend(next_tokens.iter().map(|token| token.id));
        self.backend
            .write_u32(&mut activations.tokens, &activations.input_tokens)?;
        self.write_batch_device_positions(inputs)?;
        Ok(())
    }

    fn batch_vector_shapes(
        &self,
        width: usize,
    ) -> Result<(VectorShape, VectorShape, VectorShape), RuntimeError> {
        Ok((
            VectorShape::new(width, self.model.config.n_embd)?,
            VectorShape::new(self.model.config.n_head, self.model.config.head_dim)?,
            VectorShape::new(self.model.config.n_head_kv, self.model.config.head_dim)?,
        ))
    }

    #[cfg(test)]
    fn capture_batch_forward(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        shapes: (VectorShape, VectorShape, VectorShape),
        ranges: &[KvAppendRange],
        capture_positions: &mut [B::Buffer],
    ) -> Result<(), RuntimeError> {
        self.prepare_batch_attention_views(inputs, Some(capture_positions))?;
        self.capture_batch_forward_prepared(inputs, activations, shapes, ranges, capture_positions)
    }

    fn capture_batch_forward_prepared(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        shapes: (VectorShape, VectorShape, VectorShape),
        ranges: &[KvAppendRange],
        capture_positions: &mut [B::Buffer],
    ) -> Result<(), RuntimeError> {
        self.retain_batch_capture_buffers(inputs, activations, capture_positions)?;
        self.reserve_decode_graph_generation()?;
        self.backend.begin_decode_graph()?;
        let body =
            self.capture_batch_forward_body(inputs, activations, shapes, ranges, capture_positions);
        self.finish_batch_graph_capture(body, capture_positions)
    }

    fn retain_batch_capture_buffers(
        &mut self,
        inputs: &[BatchSession<'_, B>],
        activations: &VerifyActivations<B>,
        capture_positions: &[B::Buffer],
    ) -> Result<(), RuntimeError> {
        activations.retain_decode_graph_buffers(&mut self.backend)?;
        for input in inputs.iter() {
            self.retain_batch_session_buffers(input)?;
        }
        for position in capture_positions.iter() {
            self.backend
                .retain_decode_graph_buffers(std::slice::from_ref(&position))?;
        }
        Ok(())
    }

    fn retain_batch_session_buffers(
        &mut self,
        input: &BatchSession<'_, B>,
    ) -> Result<(), RuntimeError> {
        let session_activations = input
            .session
            .activations
            .as_ref()
            .ok_or(RuntimeError::BatchSessionMismatch)?;
        session_activations.retain_decode_graph_buffers(&mut self.backend)?;
        Ok(())
    }

    fn finish_batch_graph_capture(
        &mut self,
        body: Result<(), RuntimeError>,
        capture_positions: &mut [B::Buffer],
    ) -> Result<(), RuntimeError> {
        let increment = body
            .as_ref()
            .map(|_| self.increment_capture_positions(capture_positions))
            .unwrap_or_else(|_| Ok(()));
        let end = self.backend.end_decode_graph();
        body?;
        increment?;
        end?;
        Ok(())
    }

    fn capture_batch_forward_body(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
        shapes: (VectorShape, VectorShape, VectorShape),
        ranges: &[KvAppendRange],
        capture_positions: &[B::Buffer],
    ) -> Result<(), RuntimeError> {
        let (hidden_shape, query_shape, key_shape) = shapes;
        let width = inputs.len();
        self.embed_batch_inputs(activations, width)?;
        for layer_index in 0..self.model.weights.layers.len() {
            self.forward_batch_layer(
                inputs,
                activations,
                layer_index,
                hidden_shape,
                query_shape,
                key_shape,
                width,
                Some(capture_positions),
                ranges,
            )?;
        }
        self.finish_batch_forward(activations, hidden_shape, width)
    }

    fn take_batch_capture_positions(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
    ) -> Result<Vec<B::Buffer>, RuntimeError> {
        let mut positions = Vec::with_capacity(inputs.len());
        for input in inputs.iter_mut() {
            let state = input
                .session
                .state
                .as_mut()
                .ok_or(RuntimeError::BatchSessionMismatch)?;
            let replacement = match self.backend.clone_buffer(&state.device_position) {
                Ok(replacement) => replacement,
                Err(error) => {
                    self.restore_batch_capture_positions(inputs, positions);
                    return Err(error.into());
                }
            };
            positions.push(std::mem::replace(&mut state.device_position, replacement));
        }
        Ok(positions)
    }

    fn restore_batch_capture_positions(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        positions: Vec<B::Buffer>,
    ) {
        for (input, position) in inputs.iter_mut().zip(positions) {
            if let Some(state) = input.session.state.as_mut() {
                state.device_position = position;
            }
        }
    }

    fn discard_batch_pending(inputs: &mut [BatchSession<'_, B>]) -> Result<(), BackendError> {
        let mut cleanup_error = None;
        for input in inputs {
            if let Some(state) = input.session.state.as_mut() {
                state.graph_revision = None;
                if let Err(error) = state.discard_pending() {
                    if cleanup_error.is_none() {
                        cleanup_error = Some(error);
                    }
                }
            }
        }
        cleanup_error.map_or(Ok(()), Err)
    }

    fn increment_capture_positions(
        &mut self,
        positions: &mut [B::Buffer],
    ) -> Result<(), RuntimeError> {
        for position in positions {
            self.backend.increment_u32(position)?;
        }
        Ok(())
    }

    fn write_batch_device_positions(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
    ) -> Result<(), RuntimeError> {
        for input in inputs {
            let state = input
                .session
                .state
                .as_mut()
                .ok_or(RuntimeError::BatchSessionMismatch)?;
            let position =
                u32::try_from(state.position).map_err(|_| RuntimeError::ContextCapacity {
                    requested: state.position,
                    capacity: u32::MAX as usize,
                })?;
            self.backend
                .write_u32(&mut state.device_position, &[position])?;
        }
        Ok(())
    }

    fn sample_batch_rows(
        &mut self,
        inputs: &mut [BatchSession<'_, B>],
        activations: &mut VerifyActivations<B>,
    ) -> Result<Vec<GeneratedToken>, RuntimeError> {
        let vocab = self.model.config.vocab_size;
        let mut output = Vec::with_capacity(inputs.len());
        for (row_index, input) in inputs.iter_mut().enumerate() {
            output.push(self.sample_batch_row(input, activations, row_index, vocab)?);
        }
        Ok(output)
    }

    fn sample_batch_row(
        &mut self,
        input: &mut BatchSession<'_, B>,
        activations: &mut VerifyActivations<B>,
        row_index: usize,
        vocab: usize,
    ) -> Result<GeneratedToken, RuntimeError> {
        let position = Self::advance_batch_position(input)?;
        let logits = Self::batch_row_logits_mut(&mut activations.host_logits, row_index, vocab)?;
        input.options.penalties.apply(logits, input.transcript)?;
        let token = self.sample_batch_distribution(
            logits,
            input.options,
            input.session.mirostat.as_mut(),
            position,
        )?;
        self.commit_batch_sample(input, token)?;
        Ok(GeneratedToken {
            id: token,
            bytes: self.model.tokenizer.token_bytes(token)?,
        })
    }

    fn advance_batch_position(input: &mut BatchSession<'_, B>) -> Result<usize, RuntimeError> {
        let state = input
            .session
            .state
            .as_mut()
            .ok_or(RuntimeError::BatchSessionMismatch)?;
        state
            .position
            .checked_sub(1)
            .ok_or(RuntimeError::SizeOverflow)
    }

    fn batch_row_logits_mut(
        host_logits: &mut [f32],
        row_index: usize,
        vocab: usize,
    ) -> Result<&mut [f32], RuntimeError> {
        let start = row_index
            .checked_mul(vocab)
            .ok_or(RuntimeError::SizeOverflow)?;
        let end = start.checked_add(vocab).ok_or(RuntimeError::SizeOverflow)?;
        host_logits
            .get_mut(start..end)
            .ok_or(RuntimeError::SizeOverflow)
    }

    fn commit_batch_sample(
        &mut self,
        input: &mut BatchSession<'_, B>,
        token: u32,
    ) -> Result<(), RuntimeError> {
        let activations = input
            .session
            .activations
            .as_mut()
            .ok_or(RuntimeError::BatchSessionMismatch)?;
        self.backend.write_u32(&mut activations.sampled, &[token])?;
        input
            .session
            .evaluated_tokens
            .push(first_token(input.transcript));
        Ok(())
    }

    fn sample_batch_distribution(
        &self,
        logits: &mut [f32],
        options: &GenerateOptions,
        mirostat: Option<&mut MirostatState>,
        position: usize,
    ) -> Result<u32, RuntimeError> {
        let distribution = distribution(logits, &options.sampler)?;
        let rng = SamplerRng::new(options.seed);
        match mirostat {
            Some(controller) => Ok(controller.select(&distribution, rng, position as u64)?),
            None => Ok(select(&distribution, rng, position as u64)?),
        }
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
        let range = state.prepare_append(
            &mut self.backend,
            self.model.config.n_layer,
            positions,
            positions.max(MIN_DECODE_KV_GROWTH_TOKENS),
        )?;
        self.prepare_kv_state_views(state, range, attention_shape)?;
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
            range,
        )?;
        self.finish_forward_verify(activations, hidden_shape, positions)?;
        state.commit_append(range)?;
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
        range: KvAppendRange,
    ) -> Result<(), RuntimeError> {
        let epsilon = self.model.config.rms_epsilon;
        let theta = self.model.config.rope_theta;
        let n_ff = self.model.config.n_ff;
        for (layer_index, layer) in self.model.weights.layers.iter().enumerate() {
            Self::forward_verify_layer(
                &mut self.backend,
                layer,
                state,
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
                range,
                layer_index,
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
        state: &mut KvState<B>,
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
        range: KvAppendRange,
        layer_index: usize,
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
            range,
            layer_index,
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
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<(), RuntimeError> {
        let normalized = Self::forward_verify_inputs(
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
            range,
            layer_index,
        )?;
        Self::forward_verify_attention_apply(
            backend,
            layer,
            state,
            activations,
            attention_shape,
            start_position,
            positions,
            normalized,
            layer_index,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_inputs(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<bool, RuntimeError> {
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
        Self::forward_verify_qk(
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
            range,
            layer_index,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_attention_apply(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        start_position: usize,
        positions: usize,
        normalized: bool,
        layer_index: usize,
    ) -> Result<(), RuntimeError> {
        let query = if normalized {
            &activations.query_norm
        } else {
            &activations.query
        };
        let attention_prepares_output = backend.verifier_attention_prepares_output(attention_shape);
        let spans = state.read_spans(layer_index)?;
        let cache = KvReadView::new(&spans)?;
        backend.verify_attention_spans(
            query,
            cache,
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
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<bool, RuntimeError> {
        match qk_norm {
            QkNorm::Rms { query, key } => Self::forward_verify_rms_qk(
                backend,
                query,
                key,
                state,
                activations,
                attention_shape,
                query_shape,
                key_shape,
                start_position,
                positions,
                epsilon,
                theta,
                range,
                layer_index,
            ),
            QkNorm::Identity => Self::forward_verify_identity_qk(
                backend,
                state,
                activations,
                attention_shape,
                query_shape,
                key_shape,
                start_position,
                positions,
                theta,
                range,
                layer_index,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_rms_qk(
        backend: &mut B,
        query: &B::Buffer,
        key: &B::Buffer,
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<bool, RuntimeError> {
        let target = state.write_span(layer_index, range)?;
        backend.verify_qk_norm_rope_kv_append_span(
            &activations.query,
            query,
            &mut activations.query_norm,
            query_shape,
            &activations.key,
            key,
            &mut activations.key_norm,
            key_shape,
            &activations.value,
            target,
            attention_shape,
            start_position,
            positions,
            epsilon,
            theta,
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_verify_identity_qk(
        backend: &mut B,
        state: &mut KvState<B>,
        activations: &mut VerifyActivations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        start_position: usize,
        positions: usize,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<bool, RuntimeError> {
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
        let target = state.write_span(layer_index, range)?;
        backend.kv_append_chunk_span(
            &activations.key,
            &activations.value,
            target,
            attention_shape,
            start_position,
            positions,
        )?;
        Ok(false)
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
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
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
            range,
            layer_index,
        )?;
        Self::forward_layer_ffn(backend, layer, activations, hidden_shape, epsilon)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_attention(
        backend: &mut B,
        layer: &DenseLayer<B>,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        hidden_shape: VectorShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
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
            range,
            layer_index,
        )?;
        let spans = state.read_spans(layer_index)?;
        let cache = KvReadView::new(&spans)?;
        Self::forward_layer_attention_output(
            backend,
            layer,
            activations,
            attention_shape,
            position,
            normalized,
            cache,
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
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        position: Position<'_, B::Buffer>,
        normalized: bool,
        cache: KvReadView<'_, B::Buffer>,
    ) -> Result<(), RuntimeError> {
        let query = if normalized {
            &activations.query_norm
        } else {
            &activations.query
        };
        backend.profile_decode_op(DecodeOp::KvAppend)?;
        backend.profile_decode_op(DecodeOp::Attention)?;
        backend.attention_decode_spans(
            query,
            cache,
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
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
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
                range,
                layer_index,
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
                range,
                layer_index,
            )
            .map(|()| false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_rms_qk(
        backend: &mut B,
        query: &B::Buffer,
        key: &B::Buffer,
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        epsilon: f32,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
    ) -> Result<(), RuntimeError> {
        backend.profile_decode_op(DecodeOp::QkNorm)?;
        let target = state.write_span(layer_index, range)?;
        backend.qk_norm_rope_kv_append_span(
            &activations.query,
            query,
            &mut activations.query_norm,
            query_shape,
            &activations.key,
            key,
            &mut activations.key_norm,
            key_shape,
            &activations.value,
            target,
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
        state: &mut KvState<B>,
        activations: &mut Activations<B>,
        attention_shape: AttentionShape,
        query_shape: VectorShape,
        key_shape: VectorShape,
        position: Position<'_, B::Buffer>,
        theta: f32,
        range: KvAppendRange,
        layer_index: usize,
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
        let target = state.write_span(layer_index, range)?;
        backend.kv_append_span(
            &activations.key,
            &activations.value,
            target,
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
    /// argmax path, so a decode benchmark measures the same work.
    fn read_logit_row(
        &mut self,
        activations: &Activations<B>,
        row: &mut Vec<f32>,
        penalties: &Penalties,
        context: &[u32],
    ) -> Result<(), RuntimeError> {
        self.read_raw_logit_row(activations, row)?;
        // Penalties rewrite the row from the decode history, before any
        // truncation stage sees it.
        penalties.apply(row, context)?;
        Ok(())
    }

    fn read_raw_logit_row(
        &mut self,
        activations: &Activations<B>,
        row: &mut Vec<f32>,
    ) -> Result<(), RuntimeError> {
        row.resize(self.model.config.vocab_size, 0.0);
        self.backend.read_f32(&activations.logits, row)?;
        Ok(())
    }

    fn capture_speculative_logit(
        &mut self,
        activations: &Activations<B>,
        row: usize,
    ) -> Result<(), RuntimeError> {
        let scratch = self
            .speculative_logits
            .as_mut()
            .ok_or(RuntimeError::SizeOverflow)?;
        if row >= scratch.rows {
            return Err(RuntimeError::SizeOverflow);
        }
        self.backend.write_f32_row(
            &activations.logits,
            &mut scratch.buffer,
            row,
            self.model.config.vocab_size,
        )?;
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
    ) -> Result<SpeculativeRound, RuntimeError> {
        Self::validate_draft_distributions(draft_distributions, drafted.len())?;
        if let Some(verify_activations) = verify_activations {
            if verify_activations.positions == drafted.len() + 1 {
                return self
                    .speculative_verify_round(
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
                    )
                    .map(|outcome| (outcome, false));
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
        .map(|outcome| (outcome, true))
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
        self.capture_speculative_logit(activations, index)?;
        self.read_raw_logit_row(activations, row)?;
        options.penalties.apply(row, context)?;
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
        self.capture_speculative_logit(activations, proposed)?;
        self.read_raw_logit_row(activations, row)?;
        options.penalties.apply(row, context)?;
        let (token, _) =
            self.sample_host_distribution(row, &options.sampler, rng, position, None, None)?;
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
            let (verdict, overlap) = Self::verify_one_proposal(
                verify_activations,
                index,
                vocab,
                proposal,
                draft_distributions,
                options,
                row,
                context,
                rng,
                base_position,
            )?;
            if let Some(overlap) = overlap {
                *overlap_sum += overlap;
                *overlap_proposals += 1;
            }
            match verdict {
                Verdict::Accept => {
                    *accepted += 1;
                    committed.push(proposal);
                    context.push(proposal);
                }
                Verdict::Reject { token } => {
                    committed.push(token);
                    context.push(token);
                    state.truncate(
                        base_position
                            .checked_add(index + 1)
                            .ok_or(RuntimeError::SizeOverflow)?,
                    )?;
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
    fn verify_one_proposal(
        verify_activations: &VerifyActivations<B>,
        index: usize,
        vocab: usize,
        proposal: u32,
        draft_distributions: Option<&[Distribution]>,
        options: &GenerateOptions,
        row: &mut Vec<f32>,
        context: &[u32],
        rng: SamplerRng,
        base_position: usize,
    ) -> Result<(Verdict, Option<f64>), RuntimeError> {
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
        let overlap = draft_distributions
            .map(|_| crate::total_variation(&target, &draft).map(|value| 1.0 - value))
            .transpose()?;
        let verdict = verify(
            &target,
            &draft,
            proposal,
            rng,
            (base_position + index) as u64,
        )?;
        Ok((verdict, overlap))
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
        _shape: AttentionShape,
        positions: usize,
    ) -> Result<PrefillKvSnapshot, RuntimeError> {
        let snapshot = state.snapshot(&mut self.backend)?;
        let layer_count = self.model.config.n_layer;
        let row_elements = self
            .model
            .config
            .n_head_kv
            .checked_mul(self.model.config.head_dim)
            .ok_or(RuntimeError::SizeOverflow)?;
        let valid_elements = row_elements
            .checked_mul(positions)
            .ok_or(RuntimeError::SizeOverflow)?;
        let mut layers = Vec::with_capacity(layer_count);
        for layer_index in 0..layer_count {
            layers.push(self.read_prefill_kv_layer(
                &snapshot,
                layer_index,
                positions,
                valid_elements,
            )?);
        }
        Ok(PrefillKvSnapshot { layers })
    }

    fn read_prefill_kv_layer(
        &self,
        snapshot: &KvSnapshot,
        layer_index: usize,
        positions: usize,
        valid_elements: usize,
    ) -> Result<PrefillKvLayerSnapshot, RuntimeError> {
        let mut key = vec![0_u16; valid_elements];
        let mut value = vec![0_u16; valid_elements];
        let mut covered = vec![false; positions];
        for segment in &snapshot.segments {
            self.read_prefill_kv_segment(
                segment,
                layer_index,
                positions,
                &mut key,
                &mut value,
                &mut covered,
            )?;
        }
        if covered.iter().any(|covered| !covered) {
            return Err(BackendError::operation(
                "read prefill KV snapshot",
                "the snapshot does not cover the requested positions",
            )
            .into());
        }
        Ok(PrefillKvLayerSnapshot { key, value })
    }

    fn read_prefill_kv_segment(
        &self,
        segment: &crate::kv::KvSnapshotSegment,
        layer_index: usize,
        positions: usize,
        key: &mut [u16],
        value: &mut [u16],
        covered: &mut [bool],
    ) -> Result<(), RuntimeError> {
        let key_cache = f16_snapshot_values(&segment.layers[layer_index].0)?;
        let value_cache = f16_snapshot_values(&segment.layers[layer_index].1)?;
        let local_tokens = segment.committed_tokens.min(
            positions
                .checked_sub(segment.logical_start)
                .unwrap_or_default(),
        );
        if local_tokens == 0 {
            return Ok(());
        }
        let head_dim = self.model.config.head_dim;
        for head in 0..self.model.config.n_head_kv {
            copy_prefill_kv_values(
                &key_cache,
                PrefillKvCopyLayout {
                    head,
                    capacity_tokens: segment.capacity_tokens.get(),
                    logical_start: segment.logical_start,
                    tokens: local_tokens,
                    positions,
                    head_dim,
                },
                key,
                covered,
                head == 0,
            )?;
            copy_prefill_kv_values(
                &value_cache,
                PrefillKvCopyLayout {
                    head,
                    capacity_tokens: segment.capacity_tokens.get(),
                    logical_start: segment.logical_start,
                    tokens: local_tokens,
                    positions,
                    head_dim,
                },
                value,
                covered,
                false,
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct PrefillKvCopyLayout {
    head: usize,
    capacity_tokens: usize,
    logical_start: usize,
    tokens: usize,
    positions: usize,
    head_dim: usize,
}

fn copy_prefill_kv_values(
    source: &[u16],
    layout: PrefillKvCopyLayout,
    destination: &mut [u16],
    covered: &mut [bool],
    mark_coverage: bool,
) -> Result<(), RuntimeError> {
    for local in 0..layout.tokens {
        copy_prefill_kv_row(source, layout, local, destination)?;
        if mark_coverage {
            let covered_position = covered
                .get_mut(
                    layout
                        .logical_start
                        .checked_add(local)
                        .ok_or(RuntimeError::SizeOverflow)?,
                )
                .ok_or(RuntimeError::SizeOverflow)?;
            if *covered_position {
                return Err(BackendError::operation(
                    "read prefill KV snapshot",
                    "the snapshot contains overlapping positions",
                )
                .into());
            }
            *covered_position = true;
        }
    }
    Ok(())
}

fn copy_prefill_kv_row(
    source: &[u16],
    layout: PrefillKvCopyLayout,
    local: usize,
    destination: &mut [u16],
) -> Result<(), RuntimeError> {
    let (source_start, source_end, destination_start, destination_end) =
        prefill_kv_row_offsets(layout, local)?;
    let source_row = source.get(source_start..source_end).ok_or_else(|| {
        BackendError::operation("read prefill KV snapshot", "source row is missing")
    })?;
    let destination_row = destination
        .get_mut(destination_start..destination_end)
        .ok_or_else(|| {
            BackendError::operation("read prefill KV snapshot", "destination row is missing")
        })?;
    destination_row.copy_from_slice(source_row);
    Ok(())
}

fn prefill_kv_row_offsets(
    layout: PrefillKvCopyLayout,
    local: usize,
) -> Result<(usize, usize, usize, usize), RuntimeError> {
    let (source_start, source_end) = prefill_kv_source_offsets(layout, local)?;
    let (destination_start, destination_end) = prefill_kv_destination_offsets(layout, local)?;
    Ok((source_start, source_end, destination_start, destination_end))
}

fn prefill_kv_source_offsets(
    layout: PrefillKvCopyLayout,
    local: usize,
) -> Result<(usize, usize), RuntimeError> {
    let source_head = checked_size_mul(
        checked_size_mul(layout.head, layout.capacity_tokens)?,
        layout.head_dim,
    )?;
    let source_start = checked_size_add(source_head, checked_size_mul(local, layout.head_dim)?)?;
    let source_end = checked_size_add(source_start, layout.head_dim)?;
    Ok((source_start, source_end))
}

fn prefill_kv_destination_offsets(
    layout: PrefillKvCopyLayout,
    local: usize,
) -> Result<(usize, usize), RuntimeError> {
    let destination_head = checked_size_mul(
        checked_size_mul(layout.head, layout.positions)?,
        layout.head_dim,
    )?;
    let position = checked_size_add(layout.logical_start, local)?;
    let destination_start = checked_size_add(
        destination_head,
        checked_size_mul(position, layout.head_dim)?,
    )?;
    let destination_end = checked_size_add(destination_start, layout.head_dim)?;
    Ok((destination_start, destination_end))
}

fn checked_size_add(left: usize, right: usize) -> Result<usize, RuntimeError> {
    left.checked_add(right).ok_or(RuntimeError::SizeOverflow)
}

fn copy_session_tokens(tokens: &[u32]) -> Result<Vec<u32>, BackendError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(tokens.len())
        .map_err(|error| BackendError::operation("copy session tokens", error))?;
    copy.extend_from_slice(tokens);
    Ok(copy)
}

fn checked_capacity_bytes<T>(capacity: usize) -> Result<u64, RuntimeError> {
    let capacity = u64::try_from(capacity).map_err(|_| RuntimeError::SizeOverflow)?;
    let element_bytes =
        u64::try_from(std::mem::size_of::<T>()).map_err(|_| RuntimeError::SizeOverflow)?;
    capacity
        .checked_mul(element_bytes)
        .ok_or(RuntimeError::SizeOverflow)
}

fn resident_token_metadata_bound(context_tokens: usize) -> Result<u64, RuntimeError> {
    let capacity = context_tokens
        .checked_add(1)
        .ok_or(RuntimeError::SizeOverflow)?;
    let capacity = metadata_vec_capacity(capacity)?;
    checked_capacity_bytes::<u32>(capacity)?
        .checked_mul(2)
        .ok_or(RuntimeError::SizeOverflow)
}

fn resident_controller_metadata_bound() -> Result<u64, RuntimeError> {
    controller_metadata_bytes()?
        .checked_mul(2)
        .ok_or(RuntimeError::SizeOverflow)
}

fn controller_metadata_bytes() -> Result<u64, RuntimeError> {
    checked_capacity_bytes::<MirostatState>(1)?
        .checked_add(checked_capacity_bytes::<
            crate::adaptive_draft::AdaptiveController,
        >(1)?)
        .ok_or(RuntimeError::SizeOverflow)?
        .checked_add(checked_capacity_bytes::<CorrectableController>(1)?)
        .ok_or(RuntimeError::SizeOverflow)
}

fn metadata_vec_capacity(len: usize) -> Result<usize, RuntimeError> {
    len.checked_mul(2)
        .and_then(|capacity| capacity.checked_add(4))
        .ok_or(RuntimeError::SizeOverflow)
}

fn hibernation_token_metadata<B: Backend>(
    source: &GenerationSession<B>,
) -> Result<u64, RuntimeError> {
    let evaluated = checked_capacity_bytes::<u32>(source.evaluated_tokens.capacity())?;
    let restored = checked_capacity_bytes::<u32>(source.restored_tokens.capacity())?;
    evaluated
        .checked_add(restored)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or(RuntimeError::SizeOverflow)
}

fn checked_metadata_sum<const N: usize>(values: [u64; N]) -> Result<u64, RuntimeError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or(RuntimeError::SizeOverflow)
    })
}

fn reusable_session_buffers<B: Backend>(
    source: &GenerationSession<B>,
) -> Result<(&KvState<B>, &Activations<B>), RuntimeError> {
    if source.evaluated_tokens.is_empty() {
        return Err(RuntimeError::SessionForkUnavailable);
    }
    let state = source
        .state
        .as_ref()
        .ok_or(RuntimeError::SessionForkUnavailable)?;
    let activations = source
        .activations
        .as_ref()
        .ok_or(RuntimeError::SessionForkUnavailable)?;
    Ok((state, activations))
}

fn retained_session_logits<B: Backend>(
    session: &GenerationSession<B>,
) -> Result<(usize, &B::Buffer), RuntimeError> {
    let (state, activations) =
        reusable_session_buffers(session).map_err(|_| RuntimeError::SessionLogitsUnavailable)?;
    let position = session
        .retained_logits_position
        .ok_or(RuntimeError::SessionLogitsUnavailable)?;
    if position != state.position || position != session.evaluated_tokens.len() {
        return Err(RuntimeError::SessionLogitsUnavailable);
    }
    Ok((position, &activations.logits))
}

fn checked_size_mul(left: usize, right: usize) -> Result<usize, RuntimeError> {
    left.checked_mul(right).ok_or(RuntimeError::SizeOverflow)
}

fn f16_snapshot_values(snapshot: &BufferSnapshot) -> Result<Vec<u16>, RuntimeError> {
    if snapshot.layout().storage() != crate::BufferStorage::F16 {
        return Err(BackendError::operation(
            "read prefill KV snapshot",
            "the KV snapshot is not f16",
        )
        .into());
    }
    snapshot
        .bytes()
        .chunks_exact(2)
        .map(|bytes| {
            let pair = [bytes[0], bytes[1]];
            Ok(u16::from_le_bytes(pair))
        })
        .collect()
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

fn compare_prefill_kv_bits(
    sequential: &PrefillKvSnapshot,
    chunked: &PrefillKvSnapshot,
) -> Result<PrefillBitwiseComparison, RuntimeError> {
    if sequential.layers.len() != chunked.layers.len() {
        return Err(BackendError::operation(
            "compare warm prefill KV",
            "the two snapshots have different layer counts",
        )
        .into());
    }
    let mut compared = PrefillBitwiseComparison::empty();
    for (sequential, chunked) in sequential.layers.iter().zip(&chunked.layers) {
        compared.add(compare_u16_bits(&sequential.key, &chunked.key)?)?;
        compared.add(compare_u16_bits(&sequential.value, &chunked.value)?)?;
    }
    Ok(compared)
}

fn compare_u16_bits(left: &[u16], right: &[u16]) -> Result<PrefillBitwiseComparison, RuntimeError> {
    if left.len() != right.len() {
        return Err(BackendError::SizeMismatch {
            name: "warm prefill KV values",
            expected: left.len(),
            actual: right.len(),
        }
        .into());
    }
    Ok(PrefillBitwiseComparison {
        compared: left.len(),
        mismatching: left
            .iter()
            .zip(right)
            .filter(|(left, right)| left != right)
            .count(),
    })
}

fn compare_f32_bits(left: &[f32], right: &[f32]) -> Result<PrefillBitwiseComparison, RuntimeError> {
    if left.len() != right.len() {
        return Err(BackendError::SizeMismatch {
            name: "warm prefill logits",
            expected: left.len(),
            actual: right.len(),
        }
        .into());
    }
    ensure_finite_warm_logits(left)?;
    ensure_finite_warm_logits(right)?;
    Ok(PrefillBitwiseComparison {
        compared: left.len(),
        mismatching: left
            .iter()
            .zip(right)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count(),
    })
}

fn ensure_finite_warm_logits(values: &[f32]) -> Result<(), RuntimeError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BackendError::operation(
            "compare warm prefill logits",
            "the logits contain a non-finite value",
        )
        .into());
    }
    Ok(())
}

fn greedy_token(logits: &[f32]) -> Result<u32, RuntimeError> {
    let token = crate::argmax(logits).ok_or_else(|| {
        BackendError::operation(
            "characterize warm prefill",
            "continued logits contain no finite value",
        )
    })?;
    u32::try_from(token).map_err(|_| RuntimeError::SizeOverflow)
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
    const BUFFER_COUNT: usize = 14;

    fn capture<B: Backend>(backend: &mut B, source: &Activations<B>) -> Result<Self, BackendError> {
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
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(sources.len())
            .map_err(|error| BackendError::operation("allocate activation snapshot", error))?;
        for buffer in sources {
            buffers.push(backend.download_buffer(buffer)?);
        }
        Ok(Self { buffers })
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
    let mut restored = Vec::new();
    restored
        .try_reserve_exact(14)
        .map_err(|error| BackendError::operation("allocate restored activations", error))?;
    for _ in 0..14 {
        let source = buffers.next().ok_or_else(|| {
            BackendError::operation("restore activations", "snapshot buffer is missing")
        })?;
        restored.push(backend.restore_buffer_classified(source, MemoryClass::Activation)?);
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

struct ActivationAllocator<'a, B: Backend> {
    backend: &'a mut B,
    rows: usize,
    class: MemoryClass,
}

impl<'a, B: Backend> ActivationAllocator<'a, B> {
    fn new(backend: &'a mut B, rows: usize, class: MemoryClass) -> Self {
        Self {
            backend,
            rows,
            class,
        }
    }

    fn u32(&mut self) -> Result<B::Buffer, BackendError> {
        self.backend
            .allocate_classified(BufferLayout::u32(self.rows)?, self.class)
    }

    fn f32_rows(&mut self, columns: usize, field: &'static str) -> Result<B::Buffer, BackendError> {
        let elements = self
            .rows
            .checked_mul(columns)
            .ok_or(BackendError::SizeOverflow { field })?;
        self.backend
            .allocate_classified(BufferLayout::f32(elements)?, self.class)
    }

    fn f32_elements(&mut self, elements: usize) -> Result<B::Buffer, BackendError> {
        self.backend
            .allocate_classified(BufferLayout::f32(elements)?, self.class)
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

    fn allocate_primary(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
    ) -> Result<VerifyPrimaryBuffers<B>, BackendError> {
        let mut allocator =
            ActivationAllocator::new(backend, positions, MemoryClass::BackendScratch);
        Ok(VerifyPrimaryBuffers {
            tokens: allocator.u32()?,
            hidden: allocator.f32_rows(config.n_embd, "verifier hidden elements")?,
            norm: allocator.f32_rows(config.n_embd, "verifier norm elements")?,
            query: allocator.f32_rows(config.n_embd, "verifier query elements")?,
        })
    }

    fn allocate_projected(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
        kv_columns: usize,
    ) -> Result<VerifyProjectedBuffers<B>, BackendError> {
        let mut allocator =
            ActivationAllocator::new(backend, positions, MemoryClass::BackendScratch);
        Ok(VerifyProjectedBuffers {
            key: allocator.f32_rows(kv_columns, "verifier key elements")?,
            value: allocator.f32_rows(kv_columns, "verifier value elements")?,
            query_norm: allocator.f32_rows(config.n_embd, "verifier query norm elements")?,
            key_norm: allocator.f32_rows(kv_columns, "verifier key norm elements")?,
        })
    }

    fn allocate_secondary(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
    ) -> Result<VerifySecondaryBuffers<B>, BackendError> {
        let mut allocator =
            ActivationAllocator::new(backend, positions, MemoryClass::BackendScratch);
        Ok(VerifySecondaryBuffers {
            attention: allocator.f32_rows(config.n_embd, "verifier attention elements")?,
            residual: allocator.f32_rows(config.n_embd, "verifier residual elements")?,
            gate: allocator.f32_rows(config.n_ff, "verifier gate elements")?,
            up: allocator.f32_rows(config.n_ff, "verifier up elements")?,
        })
    }

    fn allocate_output(
        backend: &mut B,
        config: &crate::ModelConfig,
        positions: usize,
        logit_elements: usize,
    ) -> Result<(B::Buffer, B::Buffer), BackendError> {
        let mut allocator =
            ActivationAllocator::new(backend, positions, MemoryClass::BackendScratch);
        Ok((
            allocator.f32_rows(config.n_ff, "verifier FFN elements")?,
            allocator.f32_elements(logit_elements)?,
        ))
    }

    fn retain_decode_graph_buffers(&self, backend: &mut B) -> Result<(), BackendError> {
        let buffers = [
            &self.tokens,
            &self.hidden,
            &self.norm,
            &self.query,
            &self.key,
            &self.value,
            &self.query_norm,
            &self.key_norm,
            &self.attention,
            &self.residual,
            &self.gate,
            &self.up,
            &self.ffn,
            &self.logits,
        ];
        backend.retain_decode_graph_buffers(&buffers)
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

    fn retain_decode_graph_buffers(&self, backend: &mut B) -> Result<(), BackendError> {
        let buffers = [
            &self.hidden,
            &self.norm,
            &self.query,
            &self.key,
            &self.value,
            &self.query_norm,
            &self.key_norm,
            &self.attention,
            &self.residual,
            &self.gate,
            &self.up,
            &self.ffn,
            &self.logits,
            &self.sampled,
        ];
        backend.retain_decode_graph_buffers(&buffers)
    }

    fn allocate_primary(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<DecodePrimaryBuffers<B>, BackendError> {
        Ok(DecodePrimaryBuffers {
            hidden: backend
                .allocate_classified(BufferLayout::f32(config.n_embd)?, MemoryClass::Activation)?,
            norm: backend
                .allocate_classified(BufferLayout::f32(config.n_embd)?, MemoryClass::Activation)?,
            query: backend
                .allocate_classified(BufferLayout::f32(config.n_embd)?, MemoryClass::Activation)?,
            key: backend.allocate_classified(
                BufferLayout::f32(config.n_head_kv * config.head_dim)?,
                MemoryClass::Activation,
            )?,
        })
    }

    fn allocate_attention(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<DecodeAttentionBuffers<B>, BackendError> {
        Ok(DecodeAttentionBuffers {
            value: backend.allocate_classified(
                BufferLayout::f32(config.n_head_kv * config.head_dim)?,
                MemoryClass::Activation,
            )?,
            query_norm: backend
                .allocate_classified(BufferLayout::f32(config.n_embd)?, MemoryClass::Activation)?,
            key_norm: backend.allocate_classified(
                BufferLayout::f32(config.n_head_kv * config.head_dim)?,
                MemoryClass::Activation,
            )?,
            attention: backend
                .allocate_classified(BufferLayout::f32(config.n_embd)?, MemoryClass::Activation)?,
        })
    }

    fn allocate_ffn(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<DecodeFfnBuffers<B>, BackendError> {
        Ok(DecodeFfnBuffers {
            residual: backend
                .allocate_classified(BufferLayout::f32(config.n_embd)?, MemoryClass::Activation)?,
            gate: backend
                .allocate_classified(BufferLayout::f32(config.n_ff)?, MemoryClass::Activation)?,
            up: backend
                .allocate_classified(BufferLayout::f32(config.n_ff)?, MemoryClass::Activation)?,
            ffn: backend
                .allocate_classified(BufferLayout::f32(config.n_ff)?, MemoryClass::Activation)?,
        })
    }

    fn allocate_output(
        backend: &mut B,
        config: &crate::ModelConfig,
    ) -> Result<(B::Buffer, B::Buffer), BackendError> {
        Ok((
            backend.allocate_classified(
                BufferLayout::f32(config.vocab_size)?,
                MemoryClass::Activation,
            )?,
            backend.allocate_classified(BufferLayout::u32(1)?, MemoryClass::Activation)?,
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
        plan: PrefillPlan,
        chunk_tokens: usize,
        full: PrefillActivations<B>,
        tail: Option<PrefillActivations<B>>,
        workspace: PrefillWorkspace,
    },
}

#[derive(Debug, Clone, Copy)]
enum SequentialPrefillOutput {
    Hidden,
    Logits,
}

impl SequentialPrefillOutput {
    fn for_position(position: usize, last_quantum_position: usize) -> Self {
        if position == last_quantum_position {
            Self::Logits
        } else {
            Self::Hidden
        }
    }
}

fn prepared_prefill_workspace<B: Backend>(prepared: &PreparedPrefill<B>) -> PrefillWorkspace {
    match prepared {
        PreparedPrefill::Chunked { workspace, .. } => *workspace,
        PreparedPrefill::Reused | PreparedPrefill::Sequential => PrefillWorkspace::default(),
    }
}

#[derive(Debug, Clone, Copy)]
struct PrefillExecution {
    processed_tokens: usize,
    cancelled: bool,
    complete: bool,
    workspace: PrefillWorkspace,
    prefill_method: PrefillMethod,
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
            logits: backend
                .allocate_classified(BufferLayout::f32(elements)?, MemoryClass::PrefillScratch)?,
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
        let mut allocator = ActivationAllocator::new(backend, tokens, MemoryClass::PrefillScratch);
        let ffn = allocator.f32_rows(config.n_ff, "prefill activation elements")?;
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

    fn allocate_primary(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
    ) -> Result<PrefillPrimaryBuffers<B>, BackendError> {
        let mut allocator = ActivationAllocator::new(backend, tokens, MemoryClass::PrefillScratch);
        Ok(PrefillPrimaryBuffers {
            tokens: allocator.u32()?,
            hidden: allocator.f32_rows(config.n_embd, "prefill activation elements")?,
            norm: allocator.f32_rows(config.n_embd, "prefill activation elements")?,
            query: allocator.f32_rows(config.n_embd, "prefill activation elements")?,
        })
    }

    fn allocate_projected(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
        kv_columns: usize,
    ) -> Result<PrefillProjectedBuffers<B>, BackendError> {
        let mut allocator = ActivationAllocator::new(backend, tokens, MemoryClass::PrefillScratch);
        Ok(PrefillProjectedBuffers {
            key: allocator.f32_rows(kv_columns, "prefill activation elements")?,
            value: allocator.f32_rows(kv_columns, "prefill activation elements")?,
            query_norm: allocator.f32_rows(config.n_embd, "prefill activation elements")?,
            key_norm: allocator.f32_rows(kv_columns, "prefill activation elements")?,
        })
    }

    fn allocate_secondary(
        backend: &mut B,
        config: &crate::ModelConfig,
        tokens: usize,
    ) -> Result<PrefillSecondaryBuffers<B>, BackendError> {
        let mut allocator = ActivationAllocator::new(backend, tokens, MemoryClass::PrefillScratch);
        Ok(PrefillSecondaryBuffers {
            attention: allocator.f32_rows(config.n_embd, "prefill activation elements")?,
            residual: allocator.f32_rows(config.n_embd, "prefill activation elements")?,
            gate: allocator.f32_rows(config.n_ff, "prefill activation elements")?,
            up: allocator.f32_rows(config.n_ff, "prefill activation elements")?,
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
struct HibernatedKvState {
    snapshot: KvSnapshot,
}

impl HibernatedKvState {
    fn capture<B: Backend>(backend: &mut B, source: &KvState<B>) -> Result<Self, BackendError> {
        Ok(Self {
            snapshot: source.snapshot(backend)?,
        })
    }

    fn restore<B: Backend>(&self, backend: &mut B) -> Result<KvState<B>, RuntimeError> {
        KvState::restore(backend, &self.snapshot).map_err(Into::into)
    }
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
        termination: GenerationTermination::Cancelled,
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
    use crate::runtime_service::{
        LeoneRuntimeDriver, RuntimeQuantumExecutor, ScheduledGenerationRequest,
    };
    use crate::scheduler::{DispatchKind, RequestId, RequestSpec, RequestStatus, SchedulerPolicy};
    use crate::service::ScheduledService;

    fn kv_test_config() -> crate::ModelConfig {
        crate::ModelConfig {
            architecture: crate::ModelArchitecture::Qwen3,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            n_embd: 128,
            n_ff: 256,
            head_dim: 32,
            vocab_size: 256,
            context_length: 128,
            rope_theta: 10_000.0,
            rope_frequency_factors: None,
            rms_epsilon: 1e-6,
        }
    }

    #[test]
    fn prefill_plan_defaults_to_backend_preferred_numerics() {
        let plan = PrefillPlan::new(4, 8, 4, 2, 32, 128, 256, 256).expect("prefill plan is valid");
        assert_eq!(plan.numerics(), PrefillNumerics::BackendPreferred);
        assert_eq!(
            plan.with_numerics(PrefillNumerics::DecodeEquivalent)
                .numerics(),
            PrefillNumerics::DecodeEquivalent
        );
    }

    #[test]
    fn unsupported_decode_equivalent_plan_is_rejected_by_cpu_backend() {
        let plan = PrefillPlan::new(4, 8, 4, 2, 32, 128, 256, 256)
            .expect("prefill plan is valid")
            .with_numerics(PrefillNumerics::DecodeEquivalent);
        let mut backend = crate::CpuBackend::new();
        let error = backend
            .prepare_prefill(plan)
            .expect_err("CPU has no decode-equivalent prefill capability");
        assert!(error.to_string().contains("decode-equivalent prefill"));
    }

    #[test]
    fn selected_prefill_method_reports_prepared_modes() {
        let runtime = cpu_toy_runtime();
        assert_eq!(
            runtime.selected_prefill_method(&PreparedPrefill::<crate::CpuBackend>::Reused),
            PrefillMethod::Reused
        );
        assert_eq!(
            runtime.selected_prefill_method(&PreparedPrefill::<crate::CpuBackend>::Sequential),
            PrefillMethod::SequentialDecode
        );
    }

    #[test]
    fn reused_prefill_execution_reports_zero_work() {
        let mut runtime = cpu_toy_runtime();
        let config = runtime.model.config.clone();
        let shape = AttentionShape::new(
            config.n_head,
            config.n_head_kv,
            config.head_dim,
            config.context_length,
        )
        .expect("test attention shape is valid");
        let (mut state, mut activations) = cpu_generation_parts(&mut runtime.backend, &config, 1);
        let mut prepared = PreparedPrefill::Reused;
        let mut cancelled = || false;
        let execution = runtime
            .run_prompt_prefill_bounded(
                &[],
                &mut state,
                &mut activations,
                shape,
                &mut prepared,
                1,
                &mut cancelled,
            )
            .expect("reused prefill completes");
        assert_eq!(execution.processed_tokens, 0);
        assert_eq!(execution.prefill_method, PrefillMethod::Reused);
        assert!(execution.complete);
    }

    #[test]
    fn reused_prefill_method_has_a_stable_name() {
        assert_eq!(PrefillMethod::Reused.name(), "reused");
    }

    fn cpu_generation_parts(
        backend: &mut crate::CpuBackend,
        config: &crate::ModelConfig,
        tokens: usize,
    ) -> (KvState<crate::CpuBackend>, Activations<crate::CpuBackend>) {
        let shape = AttentionShape::new(
            config.n_head,
            config.n_head_kv,
            config.head_dim,
            config.context_length,
        )
        .expect("test attention shape is valid");
        let mut state = KvState::new(backend, config.n_layer, shape, KvCacheDtype::F16)
            .expect("test KV state allocates");
        let range = state
            .prepare_append(backend, config.n_layer, tokens, tokens)
            .expect("test KV append allocates");
        state.commit_append(range).expect("test KV append commits");
        let activations = Activations::new(backend, config).expect("test activations allocate");
        (state, activations)
    }

    pub(super) fn cpu_toy_runtime() -> Runtime<crate::CpuBackend> {
        let config = crate::ModelConfig {
            architecture: crate::ModelArchitecture::Qwen3,
            n_layer: 0,
            n_head: 8,
            n_head_kv: 8,
            n_embd: 256,
            n_ff: 256,
            head_dim: 32,
            vocab_size: 256,
            context_length: 128,
            rope_theta: 10_000.0,
            rope_frequency_factors: None,
            rms_epsilon: 1e-6,
        };
        let metadata = BTreeMap::from([
            (
                "tokenizer.ggml.model".to_owned(),
                leone_gguf::MetadataValue::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.pre".to_owned(),
                leone_gguf::MetadataValue::String("qwen2".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                leone_gguf::MetadataValue::Array(leone_gguf::MetadataArray::String(
                    vec!["a".to_owned(); config.vocab_size],
                )),
            ),
            (
                "tokenizer.ggml.token_type".to_owned(),
                leone_gguf::MetadataValue::Array(leone_gguf::MetadataArray::Int32(
                    vec![1; config.vocab_size],
                )),
            ),
            (
                "tokenizer.ggml.merges".to_owned(),
                leone_gguf::MetadataValue::Array(leone_gguf::MetadataArray::String(Vec::new())),
            ),
        ]);
        let tokenizer =
            crate::Tokenizer::from_metadata(&metadata).expect("toy tokenizer metadata is valid");
        let mut backend = crate::CpuBackend::new();
        let embedding_shape =
            crate::QuantMatrix::new(config.vocab_size, config.n_embd, crate::QuantFormat::Q4K)
                .expect("toy embedding shape is valid");
        let embedding = backend
            .allocate(
                embedding_shape
                    .layout()
                    .expect("toy embedding layout is valid"),
            )
            .expect("toy embedding allocates");
        let output_norm = backend
            .allocate(crate::BufferLayout::f32(config.n_embd).expect("toy norm layout is valid"))
            .expect("toy output norm allocates");
        let weights = crate::model::DenseWeights {
            token_embedding: crate::model::QuantWeight {
                buffer: embedding,
                shape: embedding_shape,
            },
            output_norm,
            output: crate::model::OutputWeight::Tied,
            layers: crate::model::StagedVec::empty(),
        };
        let model = crate::LoadedModel::from_test_parts(config, tokenizer, weights);
        Runtime::from_model(backend, model)
    }

    fn cycle_weights(output: bool) -> Vec<u8> {
        let mut bytes = vec![0; 256 * 144];
        for row in 0..4 {
            let block = &mut bytes[row * 144..(row + 1) * 144];
            block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            block[4] = 1;
            let column = if output { (row + 3) % 4 } else { row };
            block[16 + column] = 1;
        }
        bytes
    }

    pub(super) fn cycle_runtime() -> Runtime<crate::CpuBackend> {
        let mut runtime = cpu_toy_runtime();
        runtime.backend.set_test_batch_size(2);
        let shape = runtime.model.weights.token_embedding.shape;
        let layout = shape.layout().expect("cycle weight layout is valid");
        runtime.model.weights.token_embedding.buffer = runtime
            .backend
            .upload(layout, &cycle_weights(false))
            .expect("cycle embedding uploads");
        let output = runtime
            .backend
            .upload(layout, &cycle_weights(true))
            .expect("cycle output uploads");
        runtime.model.weights.output = OutputWeight::Separate(crate::model::QuantWeight {
            buffer: output,
            shape,
        });
        let norm = [1.0_f32; 256]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        runtime.model.weights.output_norm = runtime
            .backend
            .upload(
                BufferLayout::f32(256).expect("cycle norm layout is valid"),
                &norm,
            )
            .expect("cycle norm uploads");
        runtime
    }

    fn graph_runtime_with_session() -> (
        Runtime<crate::CpuBackend>,
        GenerationSession<crate::CpuBackend>,
    ) {
        let mut runtime = cycle_runtime();
        runtime.backend.graph_capture = true;
        runtime.backend.preflight_probe = Some(crate::cpu::PreflightProbe::default());
        let mut session = GenerationSession::new();
        runtime
            .generate_session_tokens(
                &mut session,
                &[0, 1, 2, 3],
                GenerateOptions::greedy(2),
                |_| Ok(()),
                || false,
            )
            .expect("graph-backed generation succeeds");
        (runtime, session)
    }

    fn install_charged_graph_buffer(runtime: &mut Runtime<crate::CpuBackend>) {
        let graph_buffer = runtime
            .backend
            .allocate_classified(
                BufferLayout::u32(256 * 1024).expect("graph buffer layout is valid"),
                MemoryClass::GraphBuffer,
            )
            .expect("test graph buffer allocates");
        runtime
            .backend
            .preflight_probe
            .as_mut()
            .expect("graph probe")
            .graph_buffer = Some(graph_buffer);
    }

    pub(super) fn cycle_options(max_tokens: usize, speculative: bool) -> GenerateOptions {
        let speculation = if speculative {
            Speculation::Suffix(
                SuffixDrafter::new(
                    NonZeroUsize::new(1).unwrap(),
                    NonZeroUsize::new(1).unwrap(),
                    NonZeroUsize::new(3).unwrap(),
                )
                .unwrap(),
            )
        } else {
            Speculation::Disabled
        };
        GenerateOptions {
            decode_execution: DecodeExecution::Eager,
            speculation,
            ..GenerateOptions::greedy(max_tokens)
        }
    }

    fn finish_cpu_prefill(
        runtime: &mut Runtime<crate::CpuBackend>,
        session: &mut GenerationSession<crate::CpuBackend>,
        prompt: &[u32],
        options: GenerateOptions,
    ) {
        let mut pending = runtime
            .begin_prefill(session, prompt, options)
            .expect("CPU prefill starts");
        loop {
            let budget = pending.minimum_budget();
            pending = match runtime
                .advance_prefill(pending, budget, || false)
                .expect("CPU prefill advances")
            {
                PrefillProgress::Pending(next) => next,
                PrefillProgress::Ready(ready) => {
                    runtime
                        .finish_prefill(ready, session)
                        .expect("CPU prefill commits");
                    return;
                }
                PrefillProgress::Cancelled(_) => panic!("CPU prefill was cancelled"),
            };
        }
    }

    fn cpu_service_policy() -> SchedulerPolicy {
        SchedulerPolicy {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_batch_requests: 2,
            max_reserved_kv_bytes: 1 << 20,
            kv_bytes_per_token: 8,
            kv_page_tokens: 16,
            max_prompt_tokens: 64,
            max_output_tokens: 64,
            service_quantum_tokens: 2,
            prefill_chunk_tokens: 4,
            urgent_window_ns: 0,
            max_prefix_credit_tokens: 64,
        }
    }

    fn set_cycle_eos(runtime: &mut Runtime<crate::CpuBackend>, eos: u32) {
        let metadata = BTreeMap::from([
            (
                "tokenizer.ggml.model".to_owned(),
                leone_gguf::MetadataValue::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.pre".to_owned(),
                leone_gguf::MetadataValue::String("qwen2".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                leone_gguf::MetadataValue::Array(leone_gguf::MetadataArray::String(
                    vec!["a".to_owned(); 256],
                )),
            ),
            (
                "tokenizer.ggml.token_type".to_owned(),
                leone_gguf::MetadataValue::Array(leone_gguf::MetadataArray::Int32(vec![1; 256])),
            ),
            (
                "tokenizer.ggml.merges".to_owned(),
                leone_gguf::MetadataValue::Array(leone_gguf::MetadataArray::String(Vec::new())),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_owned(),
                leone_gguf::MetadataValue::Uint32(eos),
            ),
        ]);
        runtime.model.tokenizer =
            crate::Tokenizer::from_metadata(&metadata).expect("cycle tokenizer metadata is valid");
    }

    fn assert_cycle_continuation(
        runtime: &mut Runtime<crate::CpuBackend>,
        session: &mut GenerationSession<crate::CpuBackend>,
    ) {
        super::diagnostics_tests::assert_cycle_row(runtime, session);
        let prefix = session.evaluated_tokens.clone();
        assert_eq!(
            session.state.as_ref().expect("state is retained").position,
            prefix.len()
        );
        let expected = (prefix.last().expect("prefix is nonempty") + 1) % 4;
        let result = runtime
            .generate_session_tokens(
                session,
                &prefix,
                cycle_options(1, false),
                |_| Ok(()),
                || false,
            )
            .expect("exact continuation succeeds");
        assert_eq!(result.tokens, [expected]);
    }

    fn correctable_options(max_tokens: usize) -> GenerateOptions {
        GenerateOptions {
            decode_execution: DecodeExecution::Eager,
            speculation: Speculation::Correctable(
                CorrectableDrafter::new(256, NonZeroUsize::new(3).unwrap()).unwrap(),
            ),
            ..GenerateOptions::greedy(max_tokens)
        }
    }

    #[test]
    fn kv_bytes_per_token_matches_each_storage_layout() {
        let config = kv_test_config();
        assert_eq!(KvCacheDtype::Q8.bytes_per_token(&config).unwrap(), 272);
        assert_eq!(KvCacheDtype::F16.bytes_per_token(&config).unwrap(), 512);
        assert_eq!(KvCacheDtype::F32.bytes_per_token(&config).unwrap(), 1_024);
    }

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

    #[test]
    fn partial_adaptive_round_preserves_measured_speedup() {
        use crate::adaptive_draft::{AdaptiveDecision, AdaptiveObservation, AdaptiveReason};

        let config = crate::AdaptiveControllerConfig::new(
            std::num::NonZeroU64::new(1).unwrap(),
            1.1,
            0.03,
            0.25,
        )
        .unwrap();
        let mut adaptive = crate::AdaptiveController::new(config);
        adaptive
            .observe(AdaptiveObservation {
                verifier_positions: NonZeroUsize::new(1).unwrap(),
                produced_tokens: NonZeroUsize::new(1).unwrap(),
                wall_duration: Duration::from_millis(10),
                controller_duration: Duration::ZERO,
            })
            .unwrap();
        let mut controller = Some(adaptive);
        Runtime::<crate::CpuBackend>::observe_generation_round(
            Some((NonZeroUsize::new(2).unwrap(), Duration::ZERO)),
            None,
            1,
            2,
            &mut controller,
            &mut None,
            Duration::from_millis(12),
            0,
            0,
            0.0,
            0,
        )
        .expect("partial adaptive observation is valid");
        assert_eq!(
            controller.unwrap().decide(1),
            AdaptiveDecision::Speculate {
                verifier_positions: NonZeroUsize::new(2).unwrap(),
                reason: AdaptiveReason::BestMeasuredWidth,
            }
        );
    }

    #[test]
    fn correctable_observation_keeps_executed_work_after_partial_emit() {
        let mut controller = Some(CorrectableController::new(
            CorrectableControllerConfig::default(),
        ));
        Runtime::<crate::CpuBackend>::observe_generation_round(
            None,
            Some((
                Some(crate::CorrectablePlan::Bigram),
                NonZeroUsize::new(3).unwrap(),
                Duration::ZERO,
            )),
            1,
            4,
            &mut None,
            &mut controller,
            Duration::ZERO,
            3,
            2,
            2.0,
            3,
        )
        .expect("partial correctable observation is valid");
        let stats = controller.expect("controller remains present").stats();
        assert_eq!(stats.proposed_tokens, 3);
        assert_eq!(stats.accepted_tokens, 2);
        assert_eq!(stats.overlap_sum, 2.0);
        assert_eq!(stats.overlap_proposals, 3);
        assert!(stats.overlap_sum <= stats.overlap_proposals as f64);
    }

    #[test]
    fn speculative_stop_restores_logits_at_retained_boundary() {
        let mut runtime = cycle_runtime();
        let mut session = GenerationSession::new();
        let emitted = std::rc::Rc::new(std::cell::Cell::new(0_usize));
        let callback_emitted = std::rc::Rc::clone(&emitted);
        let result = runtime
            .generate_session_tokens_with_stop(
                &mut session,
                &[0, 1, 2, 3, 0, 1, 2, 3],
                cycle_options(8, true),
                move |_| {
                    callback_emitted.set(callback_emitted.get() + 1);
                    Ok(())
                },
                || false,
                move || emitted.get() == 2,
            )
            .expect("speculative stop succeeds");
        assert_eq!(result.tokens, [0, 1]);
        assert_eq!(result.termination, GenerationTermination::Stopped);
        let mut sampled = [u32::MAX];
        runtime
            .backend
            .read_u32(
                &session
                    .activations
                    .as_ref()
                    .expect("activations are retained")
                    .sampled,
                &mut sampled,
            )
            .expect("retained sampled token is readable");
        assert_eq!(sampled, [1]);
        let prefix = session.evaluated_tokens.clone();
        let mut fresh_runtime = cycle_runtime();
        let mut fresh_session = GenerationSession::new();
        let expected = fresh_runtime
            .generate_session_tokens(
                &mut fresh_session,
                &prefix,
                cycle_options(1, false),
                |_| Ok(()),
                || false,
            )
            .expect("fresh continuation succeeds");
        let actual = runtime
            .generate_session_tokens(
                &mut session,
                &prefix,
                cycle_options(1, false),
                |_| Ok(()),
                || false,
            )
            .expect("retained continuation succeeds");
        assert_eq!(actual.tokens, expected.tokens);
    }

    #[test]
    fn eos_inside_speculative_round_restores_exact_repeat() {
        for eos in 1..4 {
            let mut runtime = cycle_runtime();
            set_cycle_eos(&mut runtime, eos);
            let mut session = GenerationSession::new();
            let result = runtime
                .generate_session_tokens(
                    &mut session,
                    &[0, 1, 2, 3, 0, 1, 2, 3],
                    cycle_options(8, true),
                    |_| Ok(()),
                    || false,
                )
                .expect("EOS boundary succeeds");
            assert_eq!(result.tokens, (0..=eos).collect::<Vec<_>>());
            assert_eq!(result.termination, GenerationTermination::Completed);
            assert_cycle_continuation(&mut runtime, &mut session);
        }
    }

    #[test]
    fn rejected_speculation_stop_restores_exact_repeat() {
        let mut runtime = cycle_runtime();
        let mut session = GenerationSession::new();
        let emitted = std::rc::Rc::new(std::cell::Cell::new(0_usize));
        let callback_emitted = std::rc::Rc::clone(&emitted);
        let result = runtime
            .generate_session_tokens_with_stop(
                &mut session,
                &[0, 1, 3, 2, 0, 3],
                cycle_options(8, true),
                move |_| {
                    callback_emitted.set(callback_emitted.get() + 1);
                    Ok(())
                },
                || false,
                move || emitted.get() == 2,
            )
            .expect("rejected speculative stop succeeds");
        assert_eq!(result.tokens, [0, 1]);
        assert_eq!(result.stats.speculation.proposed, 3);
        assert_eq!(result.stats.speculation.accepted, 1);
        assert_cycle_continuation(&mut runtime, &mut session);
    }

    #[test]
    fn verifier_restores_exact_repeat_at_each_emission_cut() {
        for emitted in 2..6 {
            let mut runtime = cycle_runtime();
            runtime.backend.enable_reference_verify();
            let mut session = GenerationSession::new();
            let count = std::rc::Rc::new(std::cell::Cell::new(0_usize));
            let callback_count = std::rc::Rc::clone(&count);
            let result = runtime
                .generate_session_tokens_with_stop(
                    &mut session,
                    &[0, 1, 2, 3, 0, 1, 2, 3],
                    cycle_options(8, true),
                    move |_| {
                        callback_count.set(callback_count.get() + 1);
                        Ok(())
                    },
                    || false,
                    move || count.get() == emitted,
                )
                .expect("verifier stop succeeds");
            assert_eq!(
                result.tokens,
                (0..emitted)
                    .map(|index| index as u32 % 4)
                    .collect::<Vec<_>>()
            );
            assert!(result.stats.speculation.verify_passes > 0);
            assert_cycle_continuation(&mut runtime, &mut session);
        }
    }

    #[test]
    fn speculative_limit_trims_kv_to_evaluated_prefix() {
        let mut runtime = cycle_runtime();
        let mut session = GenerationSession::new();
        let result = runtime
            .generate_session_tokens(
                &mut session,
                &[0, 1, 2, 3, 0, 1, 2, 3],
                cycle_options(2, true),
                |_| Ok(()),
                || false,
            )
            .expect("limited speculation succeeds");
        assert_eq!(result.tokens, [0, 1]);
        assert_eq!(
            session.state.as_ref().expect("state is retained").position,
            session.evaluated_tokens.len()
        );
        super::diagnostics_tests::assert_cycle_row(&mut runtime, &session);
    }

    #[test]
    fn suffix_speculation_respects_remaining_context() {
        let mut runtime = cycle_runtime();
        runtime.model.config.context_length = 512;
        let prompt = (0..510).map(|index| index % 4).collect::<Vec<_>>();
        let mut session = GenerationSession::new();
        let result = runtime
            .generate_session_tokens(
                &mut session,
                &prompt,
                cycle_options(2, true),
                |_| Ok(()),
                || false,
            )
            .expect("suffix speculation fits the requested context");
        assert_eq!(result.tokens.len(), 2);
        assert_eq!(
            session.state.as_ref().expect("state is retained").position,
            511
        );
    }

    #[test]
    fn batch_raw_logits_survive_exact_repeat() {
        let mut runtime = cycle_runtime();
        runtime.backend.enable_reference_verify();
        let mut session = GenerationSession::new();
        let options = GenerateOptions {
            penalties: Penalties {
                repetition: 2.0,
                ..Penalties::none()
            },
            ..GenerateOptions::greedy(1)
        };
        runtime
            .generate_session_tokens(
                &mut session,
                &[0, 1, 2, 3],
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("cycle prefill succeeds");
        let transcript = [0, 1, 2, 3, 0];
        let batch_tokens = {
            let mut batch = [BatchSession {
                session: &mut session,
                transcript: &transcript,
                options: &options,
            }];
            let tokens = runtime
                .generate_session_batch_token(&mut batch)
                .expect("CPU reference batch succeeds");
            tokens.iter().map(|token| token.id).collect::<Vec<_>>()
        };
        assert_eq!(batch_tokens, [1]);
        let transcript = [0, 1, 2, 3, 0, 1];
        let second_batch_tokens = {
            let mut batch = [BatchSession {
                session: &mut session,
                transcript: &transcript,
                options: &options,
            }];
            let tokens = runtime
                .generate_session_batch_token(&mut batch)
                .expect("repeated CPU reference batch succeeds");
            tokens.iter().map(|token| token.id).collect::<Vec<_>>()
        };
        assert_eq!(second_batch_tokens, [2]);
        let prefix = session.evaluated_tokens.clone();
        let repeated = runtime
            .generate_session_tokens(&mut session, &prefix, options, |_| Ok(()), || false)
            .expect("exact repeat succeeds");
        assert_eq!(repeated.tokens, second_batch_tokens);
    }

    #[test]
    fn two_row_nonuniform_penalties_preserve_raw_logits() {
        let mut runtime = cycle_runtime();
        runtime.backend.enable_reference_verify();
        let mut first = GenerationSession::new();
        let mut second = GenerationSession::new();
        let seed_options = cycle_options(1, false);
        runtime
            .generate_session_tokens(
                &mut first,
                &[0, 1, 2, 3],
                seed_options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("first seed succeeds");
        runtime
            .generate_session_tokens(
                &mut second,
                &[0, 1, 3, 2],
                seed_options,
                |_| Ok(()),
                || false,
            )
            .expect("second seed succeeds");
        let first_transcript = [0, 1, 2, 3, 0];
        let second_transcript = [0, 1, 3, 2, 3];
        let first_options = GenerateOptions {
            penalties: Penalties {
                repetition: 2.0,
                presence: 10.0,
                frequency: 0.0,
                ..Penalties::none()
            },
            ..cycle_options(1, false)
        };
        let second_options = GenerateOptions {
            penalties: Penalties {
                repetition: 1.5,
                presence: 5.0,
                frequency: 1.0,
                ..Penalties::none()
            },
            ..cycle_options(1, false)
        };
        let batch_tokens = {
            let mut inputs = [
                BatchSession {
                    session: &mut first,
                    transcript: &first_transcript,
                    options: &first_options,
                },
                BatchSession {
                    session: &mut second,
                    transcript: &second_transcript,
                    options: &second_options,
                },
            ];
            runtime
                .generate_session_batch_token(&mut inputs)
                .expect("two-row batch succeeds")
                .into_iter()
                .map(|token| token.id)
                .collect::<Vec<_>>()
        };
        let mut first_raw = vec![0.0; 256];
        runtime
            .backend
            .read_f32(
                &first
                    .activations
                    .as_ref()
                    .expect("first activations")
                    .logits,
                &mut first_raw,
            )
            .expect("first raw logits are readable");
        let mut second_raw = vec![0.0; 256];
        runtime
            .backend
            .read_f32(
                &second
                    .activations
                    .as_ref()
                    .expect("second activations")
                    .logits,
                &mut second_raw,
            )
            .expect("second raw logits are readable");
        let mut first_reference = cycle_runtime();
        let mut first_reference_session = GenerationSession::new();
        let first_expected = first_reference
            .generate_session_tokens(
                &mut first_reference_session,
                &first_transcript,
                first_options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("first scalar reference succeeds");
        let mut first_expected_raw = vec![0.0; 256];
        first_reference
            .backend
            .read_f32(
                &first_reference_session
                    .activations
                    .as_ref()
                    .expect("first reference activations")
                    .logits,
                &mut first_expected_raw,
            )
            .expect("first reference logits are readable");
        let mut second_reference = cycle_runtime();
        let mut second_reference_session = GenerationSession::new();
        let second_expected = second_reference
            .generate_session_tokens(
                &mut second_reference_session,
                &second_transcript,
                second_options,
                |_| Ok(()),
                || false,
            )
            .expect("second scalar reference succeeds");
        let mut second_expected_raw = vec![0.0; 256];
        second_reference
            .backend
            .read_f32(
                &second_reference_session
                    .activations
                    .as_ref()
                    .expect("second reference activations")
                    .logits,
                &mut second_expected_raw,
            )
            .expect("second reference logits are readable");
        assert_eq!(
            batch_tokens,
            [first_expected.tokens[0], second_expected.tokens[0]]
        );
        assert_eq!(first_raw, first_expected_raw);
        assert_eq!(second_raw, second_expected_raw);
        assert_ne!(first_raw, second_raw);
    }

    #[test]
    fn correctable_stop_reports_full_round_work() {
        let mut runtime = cycle_runtime();
        let mut session = GenerationSession::new();
        let emitted = std::rc::Rc::new(std::cell::Cell::new(0_usize));
        let callback_emitted = std::rc::Rc::clone(&emitted);
        let result = runtime
            .generate_session_tokens_with_stop(
                &mut session,
                &[0, 1, 2, 3, 0, 1, 2, 3],
                correctable_options(14),
                move |_| {
                    callback_emitted.set(callback_emitted.get() + 1);
                    Ok(())
                },
                || false,
                move || emitted.get() == 10,
            )
            .expect("correctable stop succeeds");
        assert_eq!(result.termination, GenerationTermination::Stopped);
        assert!(result.stats.speculation.correctable_speculative_rounds > 0);
        assert!(result.stats.speculation.correctable_overlap_proposals > 0);
        assert!(
            result.stats.speculation.correctable_overlap_sum
                <= result.stats.speculation.correctable_overlap_proposals as f64
        );
        super::diagnostics_tests::assert_cycle_row(&mut runtime, &session);
    }

    #[test]
    fn one_token_generation_skips_unused_graph_capture() {
        let options = GenerateOptions {
            decode_execution: DecodeExecution::Graph,
            ..GenerateOptions::greedy(1)
        };
        assert!(!Runtime::<crate::CpuBackend>::generation_graph_after_first(
            &options, 0, true, None, 0, false,
        ));
    }

    #[test]
    fn cold_preparation_retires_a_graph_owned_by_another_session_first() {
        let mut runtime = cycle_runtime();
        runtime.backend.graph_capture = true;
        runtime.backend.preflight_probe = Some(crate::cpu::PreflightProbe::default());
        let mut session = GenerationSession::new();
        runtime
            .generate_session_tokens(
                &mut session,
                &[0, 1, 2, 3],
                GenerateOptions::greedy(2),
                |_| Ok(()),
                || false,
            )
            .expect("graph-backed generation succeeds");
        assert!(session
            .state
            .as_ref()
            .expect("generation retains state")
            .graph_revision
            .is_some());
        let drop_calls = runtime
            .backend
            .preflight_probe
            .as_ref()
            .expect("graph probe")
            .drop_calls;
        runtime.backend.fail_allocations_after(0);
        let mut cold_session = GenerationSession::new();
        let error = match runtime.prepare_generation_session(
            &mut cold_session,
            &[0, 1, 2, 3],
            &cycle_options(2, true),
        ) {
            Ok(_) => panic!("cold speculative preparation allocation was expected to fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("allocate CPU buffer"));
        assert!(
            runtime
                .backend
                .preflight_probe
                .as_ref()
                .expect("graph probe")
                .drop_calls
                > drop_calls
        );
        assert!(!session.is_empty());
    }

    #[test]
    fn cold_preparation_releases_a_charged_graph_before_scratch_allocation() {
        let mut runtime = cycle_runtime();
        runtime.backend.graph_capture = true;
        runtime.backend.preflight_probe = Some(crate::cpu::PreflightProbe::default());
        let mut session = GenerationSession::new();
        runtime
            .generate_session_tokens(
                &mut session,
                &[0, 1, 2, 3],
                GenerateOptions::greedy(2),
                |_| Ok(()),
                || false,
            )
            .expect("graph-backed generation succeeds");
        let graph_buffer = runtime
            .backend
            .allocate_classified(
                BufferLayout::u32(256 * 1024).expect("graph buffer layout is valid"),
                MemoryClass::GraphBuffer,
            )
            .expect("test graph buffer allocates");
        runtime
            .backend
            .preflight_probe
            .as_mut()
            .expect("graph probe")
            .graph_buffer = Some(graph_buffer);
        let budget = runtime.backend.memory_accounting().live_bytes;
        runtime
            .set_memory_budget(MemoryBudget::limited(budget).expect("budget is positive"))
            .expect("budget matches current ownership");
        let drop_calls = runtime
            .backend
            .preflight_probe
            .as_ref()
            .expect("graph probe")
            .drop_calls;
        let mut cold_session = GenerationSession::new();
        runtime
            .prepare_generation_session(&mut cold_session, &[0, 1, 2, 3], &cycle_options(2, true))
            .expect("retiring the graph makes scratch allocation fit");
        assert!(
            runtime
                .backend
                .preflight_probe
                .as_ref()
                .expect("graph probe")
                .drop_calls
                > drop_calls
        );
        assert!(runtime
            .backend
            .preflight_probe
            .as_ref()
            .expect("graph probe")
            .graph_buffer
            .is_none());
    }

    #[test]
    fn graph_retirement_stales_revisions_before_backend_drop_errors() {
        let (mut runtime, mut session) = graph_runtime_with_session();
        install_charged_graph_buffer(&mut runtime);
        runtime.batch_graph_signature = Some(vec![1]);
        let revision = session
            .state
            .as_ref()
            .expect("generation retains state")
            .graph_revision
            .expect("generation captures a graph");
        let generation = runtime.decode_graph_generation;
        runtime
            .backend
            .preflight_probe
            .as_mut()
            .expect("graph probe")
            .fail_drop_before_release = true;
        let error = runtime
            .discard_session(&mut session)
            .expect_err("pre-release graph drop failure is injected");
        assert!(error.to_string().contains("pre-release failure"));
        assert_eq!(runtime.decode_graph_generation, generation + 1);
        assert!(runtime.batch_graph_signature.is_none());
        assert_eq!(
            session
                .state
                .as_ref()
                .expect("failed discard preserves state")
                .graph_revision,
            Some(revision)
        );
        assert!(runtime
            .backend
            .preflight_probe
            .as_ref()
            .expect("graph probe")
            .graph_buffer
            .is_some());
        runtime
            .backend
            .preflight_probe
            .as_mut()
            .expect("graph probe")
            .fail_drop_before_release = false;
        runtime
            .discard_session(&mut session)
            .expect("retry releases the retained graph");

        let (mut runtime, mut session) = graph_runtime_with_session();
        install_charged_graph_buffer(&mut runtime);
        runtime.batch_graph_signature = Some(vec![1]);
        let revision = session
            .state
            .as_ref()
            .expect("generation retains state")
            .graph_revision
            .expect("generation captures a graph");
        let generation = runtime.decode_graph_generation;
        runtime
            .backend
            .preflight_probe
            .as_mut()
            .expect("graph probe")
            .fail_drop_after_release = true;
        let error = runtime
            .discard_session(&mut session)
            .expect_err("post-release graph drop failure is injected");
        assert!(error.to_string().contains("post-release failure"));
        assert_eq!(runtime.decode_graph_generation, generation + 1);
        assert!(runtime.batch_graph_signature.is_none());
        assert_eq!(
            session
                .state
                .as_ref()
                .expect("failed discard preserves state")
                .graph_revision,
            Some(revision)
        );
        assert!(runtime
            .backend
            .preflight_probe
            .as_ref()
            .expect("graph probe")
            .graph_buffer
            .is_none());
        runtime
            .backend
            .preflight_probe
            .as_mut()
            .expect("graph probe")
            .fail_drop_after_release = false;
        runtime
            .discard_session(&mut session)
            .expect("retry clears the stale session");
        assert!(session.is_empty());
    }

    #[test]
    fn prefill_kv_oracle_uses_head_major_absolute_positions() {
        let positions = 5;
        let head_dim = 1;
        let mut output = vec![0_u16; 2 * positions * head_dim];
        let mut covered = vec![false; positions];
        let segments = [(0, 2), (2, 3)];
        for (logical_start, capacity_tokens) in segments {
            let source = (0..2)
                .flat_map(|head| {
                    (0..capacity_tokens)
                        .map(move |local| (head * 100 + logical_start + local) as u16)
                })
                .collect::<Vec<_>>();
            for head in 0..2 {
                copy_prefill_kv_values(
                    &source,
                    PrefillKvCopyLayout {
                        head,
                        capacity_tokens,
                        logical_start,
                        tokens: capacity_tokens,
                        positions,
                        head_dim,
                    },
                    &mut output,
                    &mut covered,
                    head == 0,
                )
                .expect("canonical copy succeeds");
            }
        }
        assert_eq!(output, [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]);
        assert!(covered.into_iter().all(|position| position));
    }

    #[test]
    fn failed_verifier_setup_releases_partial_buffers() {
        let config = kv_test_config();
        let mut backend = crate::CpuBackend::new();
        let before = backend.memory_accounting();
        backend.fail_allocations_after(2);
        let result = VerifyActivations::new(&mut backend, &config, 2);
        assert!(result.is_err());
        let after = backend.memory_accounting();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }

    #[test]
    fn failed_prefill_activation_setup_releases_partial_buffers() {
        let config = kv_test_config();
        let mut backend = crate::CpuBackend::new();
        let before = backend.memory_accounting();
        backend.fail_allocations_after(2);
        let result = PrefillActivations::new(&mut backend, &config, 2);
        assert!(result.is_err());
        let after = backend.memory_accounting();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }

    #[test]
    fn cancelled_child_preserves_reusable_prefix_source() {
        let mut runtime = cycle_runtime();
        let mut source = GenerationSession::new();
        let options = cycle_options(2, false);
        runtime
            .generate_session_tokens(
                &mut source,
                &[0, 1, 2, 3],
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("source generation succeeds");
        let source_tokens = source.evaluated_tokens.clone();
        let mut child = runtime
            .reuse_prefix_session(&source)
            .expect("child fork succeeds");

        let result = runtime
            .generate_session_tokens(&mut child, &source_tokens, options, |_| Ok(()), || true)
            .expect("child cancellation returns a result");

        assert_eq!(result.termination, GenerationTermination::Cancelled);
        assert!(child.is_empty());
        assert_eq!(source.evaluated_tokens, source_tokens);
        assert_eq!(
            runtime
                .reusable_prefill_tokens(&source, &source_tokens, &cycle_options(2, false))
                .expect("source reuse check succeeds"),
            source_tokens.len()
        );
    }

    #[test]
    fn sequential_forked_children_reuse_one_prefix_without_mutating_source() {
        let mut runtime = cycle_runtime();
        let mut source = GenerationSession::new();
        let options = cycle_options(1, false);
        runtime
            .generate_session_tokens(
                &mut source,
                &[0, 1, 2, 3],
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("source generation succeeds");
        let source_tokens = source.evaluated_tokens.clone();
        let mut first = runtime
            .reuse_prefix_session(&source)
            .expect("first child fork succeeds");
        let mut second = runtime
            .reuse_prefix_session(&source)
            .expect("second child fork succeeds");

        let first_result = runtime
            .generate_session_tokens(
                &mut first,
                &source_tokens,
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("first child generation succeeds");
        let second_result = runtime
            .generate_session_tokens(&mut second, &source_tokens, options, |_| Ok(()), || false)
            .expect("second child generation succeeds");

        assert_eq!(first_result.tokens, second_result.tokens);
        assert_eq!(first.evaluated_tokens, second.evaluated_tokens);
        assert_eq!(source.evaluated_tokens, source_tokens);
        assert_eq!(
            runtime
                .reusable_prefill_tokens(&source, &source_tokens, &cycle_options(1, false))
                .expect("source reuse check succeeds"),
            source_tokens.len()
        );
    }

    #[test]
    fn cpu_service_mixed_prefill_decode_keeps_cancellation_local() {
        let decode_request =
            ScheduledGenerationRequest::new(vec![0, 1, 2, 3], cycle_options(2, false));
        let prefill_request =
            ScheduledGenerationRequest::new(vec![0, 1, 2, 3], cycle_options(2, false));
        let driver = LeoneRuntimeDriver::new(cycle_runtime());
        let mut service =
            ScheduledService::new(cpu_service_policy(), RuntimeQuantumExecutor::new(driver))
                .expect("CPU service policy is valid");
        service
            .admit(
                RequestSpec {
                    id: RequestId(1),
                    arrival_ns: 0,
                    prompt_tokens: 4,
                    prefix_reused_tokens: 4,
                    max_output_tokens: 2,
                    priority: 1,
                    deadline_ns: None,
                },
                &decode_request,
                0,
            )
            .expect("decode request admits");
        service
            .admit(
                RequestSpec {
                    id: RequestId(2),
                    arrival_ns: 0,
                    prompt_tokens: 4,
                    prefix_reused_tokens: 0,
                    max_output_tokens: 2,
                    priority: 1,
                    deadline_ns: None,
                },
                &prefill_request,
                0,
            )
            .expect("prefill request admits");
        prefill_request.cancel();

        let quantums = service.tick_batch(0).expect("mixed CPU batch succeeds");
        assert_eq!(quantums.len(), 2);
        assert!(quantums.iter().any(|quantum| {
            quantum.completion.request_id == RequestId(1)
                && quantum.kind == DispatchKind::Decode
                && quantum.completion.status == RequestStatus::Finished
                && quantum.tokens.len() == 2
        }));
        assert!(quantums.iter().any(|quantum| {
            quantum.completion.request_id == RequestId(2)
                && quantum.kind == DispatchKind::Prefill
                && quantum.completion.status == RequestStatus::Cancelled
                && quantum.tokens.is_empty()
        }));
        assert_eq!(service.executor().active_len(), 0);
    }

    #[test]
    fn warm_append_allocation_failure_preserves_session() {
        let mut runtime = cpu_toy_runtime();
        runtime.model.config.n_layer = 1;
        let mut source = GenerationSession::new();
        runtime
            .generate_session_tokens(
                &mut source,
                &[0],
                GenerateOptions::greedy(1),
                |_| Ok(()),
                || false,
            )
            .expect("source generation succeeds");
        let mut child = runtime.fork_session(&source).expect("source fork succeeds");
        let evaluated = child.evaluated_tokens.clone();
        let position = child.state.as_ref().expect("child state exists").position;
        runtime.backend.fail_allocations_after(0);
        let error = runtime
            .generate_session_tokens(
                &mut child,
                &[0, 1],
                GenerateOptions::greedy(1),
                |_| Ok(()),
                || false,
            )
            .expect_err("warm append allocation is denied");
        assert_eq!(
            error.session_failure_effect(),
            Some(SessionFailureEffect::Unchanged)
        );
        assert_eq!(child.evaluated_tokens, evaluated);
        assert_eq!(
            child.state.as_ref().expect("child state remains").position,
            position
        );
        runtime.backend.fail_allocations_after(usize::MAX);
        runtime
            .generate_session_tokens(
                &mut child,
                &[0, 1],
                GenerateOptions::greedy(1),
                |_| Ok(()),
                || false,
            )
            .expect("preserved session accepts the retry");
    }

    #[test]
    fn context_growth_allocation_failure_restores_shape_and_revision() {
        let mut runtime = super::batch_attention_tests::attention_runtime(1, false);
        runtime.model.config.context_length = 2_048;
        let options = cycle_options(1, false);
        let initial_prompt = (0..480).map(|index| index as u32 % 4).collect::<Vec<_>>();
        let mut session = GenerationSession::new();
        runtime
            .generate_session_tokens(
                &mut session,
                &initial_prompt,
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("initial generation succeeds");
        let state = session.state.as_ref().expect("state is retained");
        assert_eq!(state.shape.max_context(), 512);
        assert_eq!(state.position, 480);
        let original_shape = state.shape;
        let original_revision = state.revision();
        let original_position = state.position;
        let original_tokens = session.evaluated_tokens.clone();
        let original_memory = runtime.backend.memory_accounting();
        let mut original_logits = vec![0.0; runtime.vocab_size()];
        runtime
            .read_session_logits(&session, &mut original_logits)
            .expect("initial logits are readable");
        let mut grown_prompt = original_tokens.clone();
        grown_prompt.extend((0..500).map(|index| (index as u32 + 1) % 4));
        runtime.backend.fail_allocations_after(0);
        let error = match runtime.prepare_generation_session(&mut session, &grown_prompt, &options)
        {
            Ok(_) => panic!("growth tail allocation is denied"),
            Err(error) => error,
        };

        assert_eq!(
            error.session_failure_effect(),
            Some(SessionFailureEffect::Unchanged)
        );
        {
            let state = session.state.as_ref().expect("state remains retained");
            assert_eq!(state.shape, original_shape);
            assert_eq!(state.revision(), original_revision);
            assert_eq!(state.position, original_position);
        }
        assert_eq!(session.evaluated_tokens, original_tokens);
        assert_eq!(runtime.backend.memory_accounting(), original_memory);
        let mut failed_logits = vec![0.0; runtime.vocab_size()];
        runtime
            .read_session_logits(&session, &mut failed_logits)
            .expect("failed growth keeps raw logits");
        assert_eq!(failed_logits, original_logits);

        runtime.backend.clear_allocation_failure();
        let state = session.state.as_mut().expect("state remains retained");
        let pending = state
            .prepare_append(&mut runtime.backend, runtime.model.config.n_layer, 1, 1)
            .expect("capture tail stages");
        let pending_revision = state.revision();
        let pending_memory = runtime.backend.memory_accounting();
        let grown_shape = AttentionShape::new(
            state.shape.n_head(),
            state.shape.n_head_kv(),
            state.shape.head_dim(),
            1_024,
        )
        .expect("grown shape is valid");
        runtime.backend.fail_allocations_after(1);
        let error = state
            .prepare_shape_transition(
                &mut runtime.backend,
                runtime.model.config.n_layer,
                grown_shape,
                original_position,
                Some((5, 8)),
            )
            .expect_err("partial growth allocation is denied");
        assert!(error.to_string().contains("allocate CPU buffer"));
        assert_eq!(state.pending_range(), Some(pending));
        assert_eq!(state.shape, original_shape);
        assert_eq!(state.revision(), pending_revision);
        assert_eq!(state.position, original_position);
        let restored_memory = runtime.backend.memory_accounting();
        assert_eq!(restored_memory.live_bytes, pending_memory.live_bytes);
        assert_eq!(
            restored_memory.live_allocations,
            pending_memory.live_allocations
        );
        runtime.backend.clear_allocation_failure();
        let mut restored_logits = vec![0.0; runtime.vocab_size()];
        runtime
            .read_session_logits(&session, &mut restored_logits)
            .expect("restored logits are readable");
        assert_eq!(restored_logits, original_logits);
        runtime
            .generate_session_tokens(
                &mut session,
                &original_tokens,
                options,
                |_| Ok(()),
                || false,
            )
            .expect("the original request retries after growth failure");
    }

    #[test]
    fn prefix_fork_truncates_state_and_reuses_the_retained_prefix() {
        let mut runtime = cpu_toy_runtime();
        let mut source = GenerationSession::new();
        runtime
            .generate_session_tokens(
                &mut source,
                &[0, 1, 2, 3],
                GenerateOptions::greedy(1),
                |_| Ok(()),
                || false,
            )
            .expect("source generation succeeds");

        assert!(matches!(
            runtime.fork_session_prefix(&source, 0),
            Err(RuntimeError::EmptyPrompt)
        ));
        assert!(matches!(
            runtime.fork_session_prefix(&source, source.evaluated_tokens().len() + 1),
            Err(RuntimeError::ContextCapacity { .. })
        ));

        let mut child = runtime
            .fork_session_prefix(&source, 2)
            .expect("prefix fork succeeds");
        assert_eq!(child.evaluated_tokens(), &[0, 1]);
        assert_eq!(child.state.as_ref().expect("child state").position, 2);
        assert_eq!(child.last_fork().expect("fork record").cached_tokens, 2);
        assert_eq!(child.last_replay().reused_tokens, 2);
        assert!(child.retained_logits_position.is_none());

        runtime
            .generate_session_tokens(
                &mut child,
                &[0, 1, 9],
                GenerateOptions::greedy(1),
                |_| Ok(()),
                || false,
            )
            .expect("prefix continuation succeeds");
        assert_eq!(
            child.last_replay().reuse_class,
            SessionReuseClass::DeviceFork
        );
        assert_eq!(child.last_replay().reused_tokens, 2);
    }

    #[test]
    fn cpu_runtime_probe_preserves_nested_prefix_lineage() {
        let mut runtime = cycle_runtime();
        let options = cycle_options(1, false);
        let parent_prompt = [0, 1, 2, 3];
        let child_prompt = [0, 1, 8, 9];
        let mut parent = GenerationSession::new();
        finish_cpu_prefill(&mut runtime, &mut parent, &parent_prompt, options.clone());
        let before_fork = runtime.backend.memory_accounting();
        let mut child = runtime
            .fork_session_prefix(&parent, 2)
            .expect("CPU prefix fork succeeds");
        let after_fork = runtime.backend.memory_accounting();
        assert_eq!(child.evaluated_tokens(), &[0, 1]);
        assert_eq!(child.last_fork().expect("fork record").cached_tokens, 2);
        assert_eq!(child.last_replay().reused_tokens, 2);
        assert!(after_fork.live_allocations > before_fork.live_allocations);
        finish_cpu_prefill(&mut runtime, &mut child, &child_prompt, options);
        assert_eq!(child.evaluated_tokens(), child_prompt);
        assert_eq!(
            child.last_replay().reuse_class,
            SessionReuseClass::DeviceFork
        );
        assert_eq!(child.last_replay().reused_tokens, 2);
        assert_eq!(parent.evaluated_tokens(), parent_prompt);
        assert_eq!(
            runtime
                .reusable_prefill_tokens(&child, &child_prompt, &cycle_options(1, false))
                .expect("child reuse check succeeds"),
            child_prompt.len()
        );
        runtime
            .discard_session(&mut child)
            .expect("child discard succeeds");
        let after_discard = runtime.backend.memory_accounting();
        assert!(after_discard.live_allocations < after_fork.live_allocations);
    }

    #[test]
    fn cpu_runtime_probe_keeps_owned_prefixes_private() {
        let mut runtime = cycle_runtime();
        let options = cycle_options(1, false);
        let prefix = [0, 1];
        let first_prompt = [0, 1, 6];
        let second_prompt = [0, 1, 7];
        let mut first = GenerationSession::new();
        let mut second = GenerationSession::new();
        finish_cpu_prefill(&mut runtime, &mut first, &prefix, options.clone());
        let after_first = runtime.backend.memory_accounting();
        finish_cpu_prefill(&mut runtime, &mut second, &prefix, options.clone());
        let after_second = runtime.backend.memory_accounting();
        assert!(after_second.live_allocations > after_first.live_allocations);
        assert!(first.last_fork().is_none());
        assert!(second.last_fork().is_none());
        runtime
            .generate_session_tokens(
                &mut first,
                &first_prompt,
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("CPU first owned tail succeeds");
        runtime
            .generate_session_tokens(&mut second, &second_prompt, options, |_| Ok(()), || false)
            .expect("CPU second owned tail succeeds");
        assert_eq!(first.evaluated_tokens(), first_prompt);
        assert_eq!(second.evaluated_tokens(), second_prompt);
        assert_eq!(first.state.as_ref().expect("first state").position, 3);
        assert_eq!(second.state.as_ref().expect("second state").position, 3);
        assert!(first.last_fork().is_none());
        assert!(second.last_fork().is_none());
    }

    #[test]
    fn speculative_logit_scratch_is_reused_after_preflight() {
        let mut runtime = cycle_runtime();
        let options = cycle_options(4, true);
        runtime
            .ensure_speculative_logits(&options)
            .expect("speculative scratch preflight succeeds");
        let first = runtime.backend.memory_accounting();
        runtime
            .ensure_speculative_logits(&options)
            .expect("reusing speculative scratch succeeds");
        let second = runtime.backend.memory_accounting();
        let first_class = first.class(MemoryClass::BackendScratch);
        let second_class = second.class(MemoryClass::BackendScratch);
        assert!(first_class.live_bytes > 0);
        assert_eq!(second_class.live_bytes, first_class.live_bytes);
        assert_eq!(second_class.live_allocations, first_class.live_allocations);
    }

    #[test]
    fn speculative_scratch_denial_preserves_warm_session() {
        let mut runtime = cycle_runtime();
        let mut source = GenerationSession::new();
        runtime
            .generate_session_tokens(
                &mut source,
                &[0],
                GenerateOptions::greedy(1),
                |_| Ok(()),
                || false,
            )
            .expect("source generation succeeds");
        let mut child = runtime.fork_session(&source).expect("source fork succeeds");
        let evaluated = child.evaluated_tokens.clone();
        let position = child.state.as_ref().expect("child state exists").position;
        runtime.backend.fail_allocations_after(0);
        let error = runtime
            .generate_session_tokens(
                &mut child,
                &[0, 1],
                cycle_options(2, true),
                |_| Ok(()),
                || false,
            )
            .expect_err("speculative scratch allocation is denied");
        assert_eq!(
            error.session_failure_effect(),
            Some(SessionFailureEffect::Unchanged)
        );
        assert_eq!(child.evaluated_tokens, evaluated);
        assert_eq!(
            child.state.as_ref().expect("state remains").position,
            position
        );
    }

    #[test]
    fn session_failure_effect_is_preserved_by_the_runtime_wrapper() {
        let error = RuntimeError::Backend(BackendError::operation(
            "test operation",
            "injected failure",
        ));
        let unchanged = error.with_session_failure_effect(SessionFailureEffect::Unchanged);
        assert_eq!(
            unchanged.session_failure_effect(),
            Some(SessionFailureEffect::Unchanged)
        );
        let quarantine = RuntimeError::Backend(BackendError::operation(
            "test operation",
            "post-execution failure",
        ))
        .with_session_failure_effect(SessionFailureEffect::Quarantine);
        assert_eq!(
            quarantine.session_failure_effect(),
            Some(SessionFailureEffect::Quarantine)
        );
    }

    #[test]
    fn post_execution_effect_overrides_nested_callback_effect() {
        let callback_error =
            RuntimeError::Backend(BackendError::operation("callback", "injected failure"))
                .with_session_failure_effect(SessionFailureEffect::Unchanged);
        let error =
            callback_error.with_forced_session_failure_effect(SessionFailureEffect::Quarantine);
        assert_eq!(
            error.session_failure_effect(),
            Some(SessionFailureEffect::Quarantine)
        );
    }

    #[test]
    fn termination_poll_latches_stop_before_cancellation() {
        let mut cancelled_calls = 0;
        let mut stopped_calls = 0;
        let termination = Runtime::<crate::CpuBackend>::poll_generation_termination(
            &mut || {
                cancelled_calls += 1;
                true
            },
            &mut || {
                stopped_calls += 1;
                true
            },
        );

        assert_eq!(termination, Some(GenerationTermination::Stopped));
        assert_eq!(cancelled_calls, 0);
        assert_eq!(stopped_calls, 1);
    }

    #[test]
    fn matched_stop_commits_cpu_runtime_and_continues() {
        let mut runtime = cpu_toy_runtime();
        let mut session = GenerationSession::new();
        let stop = std::rc::Rc::new(std::cell::Cell::new(false));
        let callback_stop = std::rc::Rc::clone(&stop);
        let result = runtime
            .generate_session_tokens_with_stop(
                &mut session,
                &[0],
                GenerateOptions {
                    decode_execution: DecodeExecution::Eager,
                    ..GenerateOptions::greedy(1)
                },
                move |_| {
                    callback_stop.set(true);
                    Ok(())
                },
                || false,
                move || stop.get(),
            )
            .expect("matched stop succeeds");

        assert_eq!(result.termination, GenerationTermination::Stopped);
        assert!(!session.is_empty());
        let continuation = runtime
            .generate_session_tokens_with_stop(
                &mut session,
                &[0],
                GenerateOptions {
                    decode_execution: DecodeExecution::Eager,
                    ..GenerateOptions::greedy(1)
                },
                |_| Ok(()),
                || false,
                || false,
            )
            .expect("retained session continues");
        assert_eq!(continuation.termination, GenerationTermination::Completed);
        assert!(!session.is_empty());
    }

    #[test]
    fn callback_cancellation_invalidates_cpu_runtime() {
        let mut runtime = cpu_toy_runtime();
        let mut session = GenerationSession::new();
        let cancelled = std::rc::Rc::new(std::cell::Cell::new(false));
        let callback_cancelled = std::rc::Rc::clone(&cancelled);
        let result = runtime
            .generate_session_tokens_with_stop(
                &mut session,
                &[0],
                GenerateOptions {
                    decode_execution: DecodeExecution::Eager,
                    ..GenerateOptions::greedy(1)
                },
                move |_| {
                    callback_cancelled.set(true);
                    Ok(())
                },
                move || cancelled.get(),
                || false,
            )
            .expect("callback cancellation returns a result");

        assert_eq!(result.termination, GenerationTermination::Cancelled);
        assert!(session.is_empty());
    }

    #[test]
    fn callback_error_quarantines_cpu_runtime() {
        let mut runtime = cpu_toy_runtime();
        let mut session = GenerationSession::new();
        let error = runtime
            .generate_session_tokens_with_stop(
                &mut session,
                &[0],
                GenerateOptions {
                    decode_execution: DecodeExecution::Eager,
                    ..GenerateOptions::greedy(1)
                },
                |_| Err(RuntimeError::token_callback("callback failed")),
                || false,
                || false,
            )
            .expect_err("callback failure is returned");

        assert_eq!(
            error.session_failure_effect(),
            Some(SessionFailureEffect::Quarantine)
        );
        assert!(session.is_empty());
    }

    #[test]
    fn rejected_verifier_width_reuses_decode_growth_tail() {
        let mut backend = crate::CpuBackend::new();
        let shape = AttentionShape::new(4, 2, 32, 64).expect("test shape is valid");
        let mut state = KvState::new(&mut backend, 1, shape, KvCacheDtype::F16)
            .expect("state allocation succeeds");
        for committed in 1..=4 {
            let range = state
                .prepare_append(&mut backend, 1, 8, MIN_DECODE_KV_GROWTH_TOKENS)
                .expect("verifier append allocates or reuses");
            state.commit_append(range).expect("verifier append commits");
            state
                .truncate(committed)
                .expect("rejected suffix truncates");
        }

        let spans = state.read_spans(0).expect("retained KV span is readable");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].capacity_token_count(), MIN_DECODE_KV_GROWTH_TOKENS);
    }

    #[test]
    fn stopped_generation_commits_cpu_state_for_continuation() {
        let config = kv_test_config();
        let mut backend = crate::CpuBackend::new();
        let (state, activations) = cpu_generation_parts(&mut backend, &config, 2);
        let mut session = GenerationSession::new();

        Runtime::<crate::CpuBackend>::commit_generation_session(
            &mut session,
            &[11],
            &[12],
            state,
            activations,
            SessionReuseClass::Cold,
            0,
            GenerationTermination::Stopped,
            None,
            None,
            None,
        );

        assert_eq!(session.evaluated_tokens(), &[11, 12]);
        assert!(!session.is_empty());
        let state = session.state.as_mut().expect("stopped state is retained");
        let range = state
            .prepare_append(&mut backend, config.n_layer, 1, 1)
            .expect("retained state accepts continuation");
        state.commit_append(range).expect("continuation commits");
        assert_eq!(state.position, 3);
    }

    #[test]
    fn cancelled_generation_invalidates_cpu_state() {
        let config = kv_test_config();
        let mut backend = crate::CpuBackend::new();
        let (state, activations) = cpu_generation_parts(&mut backend, &config, 2);
        let mut session = GenerationSession::new();

        Runtime::<crate::CpuBackend>::commit_generation_session(
            &mut session,
            &[11],
            &[12],
            state,
            activations,
            SessionReuseClass::Cold,
            0,
            GenerationTermination::Cancelled,
            None,
            None,
            None,
        );

        assert!(session.is_empty());
    }

    #[test]
    fn callback_failure_quarantines_and_clears_cpu_session() {
        let config = kv_test_config();
        let mut runtime = cpu_toy_runtime();
        let (state, activations) = cpu_generation_parts(&mut runtime.backend, &config, 2);
        let mut session = GenerationSession::new();
        Runtime::<crate::CpuBackend>::commit_generation_session(
            &mut session,
            &[11],
            &[12],
            state,
            activations,
            SessionReuseClass::Cold,
            0,
            GenerationTermination::Completed,
            None,
            None,
            None,
        );
        let callback_error = RuntimeError::token_callback("stop stream failed")
            .with_session_failure_effect(SessionFailureEffect::Unchanged);

        let error = runtime.quarantine_generation_failure(&mut session, callback_error);

        assert!(session.is_empty());
        assert_eq!(
            error.session_failure_effect(),
            Some(SessionFailureEffect::Quarantine)
        );
    }

    #[test]
    fn batch_width_rejects_rows_above_backend_limit() {
        assert!(Runtime::<crate::CpuBackend>::validate_batch_width(1, 1).is_ok());
        match Runtime::<crate::CpuBackend>::validate_batch_width(2, 1) {
            Err(RuntimeError::BatchSizeExceeded {
                requested: 2,
                maximum: 1,
            }) => {}
            other => panic!("unexpected batch validation result: {other:?}"),
        }
    }
}
