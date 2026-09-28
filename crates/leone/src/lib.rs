#![deny(unsafe_code)]

//! Runs batch-1 text inference without exposing backend-specific types.
//!
//! The backend contract owns buffers and kernel launches. The runtime owns
//! model order, state lifetime, tokenization, sampling, and receipts.

pub mod adaptive_draft;
pub mod backend;
mod constraint;
pub mod correctable;
pub mod cpu;
pub mod drafter;
mod kv;
pub mod model;
pub mod multimodal;
mod penalty;
pub mod progressive;
mod runtime;
pub mod runtime_service;
mod sampler;
pub mod scheduler;
pub mod service;
pub mod service_metrics;
pub mod session_archive;
pub mod state;
pub mod tokenizer;

pub use adaptive_draft::{
    AdaptiveController, AdaptiveControllerConfig, AdaptiveControllerStats, AdaptiveDrafter,
    AdaptiveDrafterConfig, AdaptiveError, AdaptiveProposal, ProposalSource,
};
pub use backend::{
    decode_graph_bucket, AttentionDecodeRow, AttentionShape, Backend, BackendError, BufferLayout,
    BufferSnapshot, BufferStorage, DecodeOp, DecodeProfile, Determinism, GemvProfile, HostStaging,
    KvReadSpan, KvReadView, KvWriteSpan, MemoryAccounting, MemoryAllocation, MemoryBudget,
    MemoryCapacity, MemoryClass, MemoryClassStats, MemoryError, MemoryReservation, MemoryTracker,
    MemoryTrackerRoot, ModelImportMetrics, Position, PrefillMethod, PrefillNumerics, PrefillPlan,
    PrefillWorkspace, QuantFormat, QuantMatrix, RopePairing, RopeShape, UntrackedMemory,
    VectorShape,
};
pub use constraint::{ConstraintError, OutputConstraint};
pub use correctable::{
    CorrectableController, CorrectableControllerConfig, CorrectableControllerStats,
    CorrectableDecision, CorrectableDrafter, CorrectableError, CorrectableObservation,
    CorrectablePlan, CorrectableProposal, CorrectableReason, CorrectableStep,
    CORRECTABLE_PLAN_COUNT,
};
pub use cpu::{CpuBackend, CpuBuffer};
pub use drafter::{Draft, DrafterError, SuffixDrafter};
pub use model::{LoadedModel, ModelArchitecture, ModelConfig, ModelLoadError};
pub use multimodal::{ImageDetail, ImageInput, ImageInputError, ImageSource};
pub use penalty::{Dry, Penalties, PenaltyError, PenaltyWindow};
pub use progressive::{
    exhaustive_progressive_oracle, ProgressiveCodes, ProgressiveError, ProgressiveOracleStats,
    ProgressiveWidth,
};
pub use runtime::{
    token_stream_sha256, BatchSession, CancelledPrefill, DecodeBenchmarkRun, DecodeExecution,
    DecodeProfileMode, GenerateOptions, GeneratedToken, GenerationCheckpoint, GenerationResult,
    GenerationSession, GenerationStats, GenerationTermination, HibernatedSession, KvCacheDtype,
    KvCapacityProbe, Logit, LogitCapture, LogitSnapshot, MeasuredRate, PendingPrefill,
    PrefillBenchmarkRun, PrefillBitwiseComparison, PrefillCharacterization, PrefillKvLayerError,
    PrefillProgress, ReadyPrefill, Runtime, RuntimeError, SessionFailureEffect, SessionHibernation,
    SessionReplay, SessionReuseClass, Speculation, SpeculationOutcome, SpeculationStats,
    VerifyCharacterization, WarmPrefillCharacterization, DEFAULT_PREFILL_CHUNK_TOKENS,
};
pub use sampler::{
    argmax, correction_residual, distribution, select, select_draft, total_variation, verify,
    Distribution, Draw, MirostatConfig, MirostatState, Sampler, SamplerError, SamplerRng,
    Temperature, Truncation, Verdict,
};
pub use service_metrics::{
    LatencySummary, Measurement, MeasurementStatus, MetricsConfig, MetricsError,
    PhysicalBytesSample, ProcessMemorySample, RequestControlSample, RequestMetric,
    SchedulerReservationSample, ServiceMetrics, ServiceMetricsSnapshot, SystemMemorySample,
    SERVICE_METRICS_SCHEMA_VERSION,
};
pub use session_archive::{SessionArchive, SessionArchiveError, SESSION_ARCHIVE_SCHEMA_VERSION};
pub use state::{
    AllocationId, StateAllocation, StateAllocator, StateError, StateKind, StateLifetime,
    TransactionId,
};
pub use tokenizer::{TokenType, Tokenizer, TokenizerError};
