use crate::backend_choice::{default_backend, validate_compiled, BackendChoice};
use crate::cli_args::flag_value;
use crate::server_metrics::{
    collector_metadata_bytes, identity_response, metrics_response_bytes, retained_metadata_bytes,
    DeliveryStatus, ForkSource, MemoryTopology as MetricsMemoryTopology, ParentMemoryObservation,
    PrefixSource, ServerMetrics, ServiceIdentity, SlowClientCause, TerminalCause,
    TerminalObservation, PRESSURE_CLOCK_DOMAIN,
};
use crate::service_memory::{
    observe_host_memory, resolve_kv_reservation, resolve_policy, BackendCapacityObservation,
    BudgetRequest, HostMemoryLedger, MemoryTopology as PolicyMemoryTopology,
    ObservationUnavailable, ServiceBudgetArgs, ServiceMemoryInputs, ServiceMemoryPolicy,
    ServiceMemoryPools,
};
use crate::transport::{
    canonical_ip, Cancellation, DeliveryFailurePhase, Incoming, OutputSink, Transport,
    TransportTelemetryHandle,
};
use chrono::Utc;
use ed25519_dalek::SigningKey;
#[cfg(feature = "cuda")]
use leone::runtime_service::{
    LeoneRuntimeDriver, RuntimeQuantumExecutor, ScheduledGenerationRequest,
};
use leone::scheduler::{
    AdmissionOutcome, AdmissionReject, Dispatch, DispatchKind, RequestId, RequestSpec,
    RequestStatus, SchedulerPolicy,
};
use leone::service::{QuantumExecutor, QuantumOutput, ScheduledService};
#[cfg(feature = "cuda")]
use leone::DecodeExecution;
use leone::{
    token_stream_sha256, AdaptiveDrafter, AdaptiveDrafterConfig, Backend, BackendError,
    BatchSession, CpuBackend, GenerateOptions, GeneratedToken, GenerationSession,
    GenerationTermination, HibernatedSession, KvCacheDtype, MemoryAccounting, MemoryAllocation,
    MemoryBudget, MemoryCapacity, MemoryClass, MemoryError, MemoryReservation, MemoryTracker,
    MemoryTrackerRoot, MirostatConfig, OutputConstraint, Penalties, PenaltyWindow, Runtime,
    RuntimeError, Sampler, SessionFailureEffect, SessionReuseClass, Speculation, SuffixDrafter,
    Temperature, Truncation, DEFAULT_PREFILL_CHUNK_TOKENS,
};
#[cfg(feature = "cuda")]
use leone_cuda::CudaBackend;
#[cfg(feature = "metal")]
use leone_metal::MetalBackend;
use leone_receipt::{
    sha256_bytes, sha256_file, write_response_receipt, ResponseClaim, ResponseReceipt,
    SessionReplayRecord, RESPONSE_SCHEMA_VERSION,
};
use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::mem::size_of;
use std::net::{IpAddr, SocketAddr, TcpStream};
#[cfg(test)]
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const DEFAULT_BIND: &str = "127.0.0.1:8080";
const DEFAULT_MAX_TOKENS: usize = 512;
const DEFAULT_SERVE_BATCH_SIZE: usize = 8;
const MAX_SESSION_ID_BYTES: usize = 128;
const MAX_QUEUED_REQUESTS: usize = 64;
const MAX_QUEUED_REQUESTS_U32: u32 = MAX_QUEUED_REQUESTS as u32;
const _: () = assert!(MAX_QUEUED_REQUESTS <= u32::MAX as usize);
static PROCESS_START_NS: OnceLock<u64> = OnceLock::new();
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_JSON_NODES: usize = 32 * 1024;
const MAX_JSON_DEPTH: usize = 64;
const MAX_SESSION_REFERENCE_BYTES: u64 = MAX_BODY_BYTES as u64;
// Serde can retain unescape scratch beside both owned reference strings.
const SESSION_REFERENCE_PARSE_BYTES: u64 = MAX_SESSION_REFERENCE_BYTES * 4 + 512;
#[cfg(test)]
const DEFAULT_SESSION_ARCHIVE_BYTES: u64 = MAX_BODY_BYTES as u64;
const SESSION_ARCHIVE_FIXED_BYTES: u64 = 512;
const STORED_ARCHIVE_METADATA_FIXED_BYTES: u64 = 64;
const PERSISTED_ARCHIVE_TABLE_ENTRY_BYTES: u64 =
    std::mem::size_of::<(String, PersistedArchiveEntry)>() as u64;
const RESPONSE_FINALIZATION_ERROR: &str = "the server could not finalize the response";
const SESSION_PREPARATION_ERROR: &str = "the server could not prepare the session";
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const LLAMA_BEGIN: &str = "<|begin_of_text|>";
const LLAMA_HEADER_START: &str = "<|start_header_id|>";
const LLAMA_HEADER_END: &str = "<|end_header_id|>";
const LLAMA_EOT: &str = "<|eot_id|>";
const QWEN_TEMPLATE_MARKERS: [&str; 8] = [
    IM_START,
    IM_END,
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
];
const LLAMA_TEMPLATE_MARKERS: [&str; 4] =
    [LLAMA_BEGIN, LLAMA_HEADER_START, LLAMA_HEADER_END, LLAMA_EOT];

#[derive(Debug)]
struct ServeArgs {
    model: PathBuf,
    bind: SocketAddr,
    sessions: usize,
    batch_size: Option<usize>,
    hibernated_sessions: usize,
    backend: BackendChoice,
    kv_cache_dtype: KvCacheDtype,
    kv_cache_dtype_explicit: bool,
    receipts: PathBuf,
    signing_key: PathBuf,
    session_store: Option<PathBuf>,
    allow_remote: bool,
    plan: Option<PathBuf>,
    prefill_chunk_tokens: Option<usize>,
    context_limit: Option<usize>,
    memory_budget: BudgetRequest,
    host_memory_budget: BudgetRequest,
    kv_reservation_budget: BudgetRequest,
    max_connections: usize,
    max_connections_per_client: usize,
    max_pending_requests: usize,
    max_output_bytes: usize,
    request_timeout_ms: u64,
    cors_origins: Vec<String>,
    proxy_origin: Option<String>,
    trusted_proxy_ips: Vec<IpAddr>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    leone_template: ChatTemplateMode,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_a: Option<f64>,
    #[serde(default)]
    tfs_z: Option<f64>,
    #[serde(default)]
    typical_p: Option<f64>,
    #[serde(default)]
    repetition_penalty: Option<f64>,
    #[serde(default)]
    repetition_window: Option<usize>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    min_p: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    #[serde(default, rename = "user")]
    _user: Option<String>,
    #[serde(default)]
    leone_session: Option<String>,
    #[serde(default)]
    leone_fork_session: Option<String>,
    #[serde(default)]
    mirostat_tau: Option<f64>,
    #[serde(default)]
    mirostat_eta: Option<f64>,
    #[serde(default)]
    draft_tokens: Option<usize>,
    #[serde(default)]
    adaptive_speculation: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ChatTemplateMode {
    #[default]
    Legacy,
    Official,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<RequestToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    refusal: Option<Value>,
    #[serde(default)]
    annotations: Option<Value>,
    #[serde(default)]
    audio: Option<Value>,
    #[serde(default)]
    function_call: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolDefinition {
    #[serde(rename = "type")]
    kind: String,
    function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolFunction {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: RequestToolCallFunction,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlPart },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageUrlPart {
    url: String,
    #[serde(default)]
    detail: Option<String>,
}

#[derive(Debug, Serialize)]
struct GeneratedToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: GeneratedToolCallFunction,
}

#[derive(Debug, Serialize)]
struct GeneratedToolCallFunction {
    name: String,
    arguments: String,
}

struct StoredSession<B: Backend> {
    generation: GenerationSession<B>,
    metadata_allocation: Option<ResidentMetadataLease>,
    reference_lease: Option<SessionReferenceLease>,
    last_used: u64,
}

struct ResidentMetadataLease {
    allocations: Vec<MemoryAllocation>,
}

impl ResidentMetadataLease {
    fn single(allocation: MemoryAllocation) -> Self {
        Self {
            allocations: vec![allocation],
        }
    }

    fn bytes(&self) -> u64 {
        self.allocations
            .iter()
            .map(MemoryAllocation::bytes)
            .fold(0, u64::saturating_add)
    }

    fn append(&mut self, allocation: MemoryAllocation) -> Result<(), RuntimeError> {
        self.allocations.try_reserve(1).map_err(|error| {
            RuntimeError::Backend(BackendError::operation(
                "grow resident metadata lease",
                error,
            ))
        })?;
        self.allocations.push(allocation);
        Ok(())
    }
}

struct StoredHibernation {
    generation: HibernatedSession,
    host_allocation: MemoryAllocation,
    metadata_bytes: u64,
    reference_lease: Option<SessionReferenceLease>,
    last_used: u64,
}

#[derive(Debug, Clone)]
struct StoredArchive {
    blob_sha256: String,
    last_used: u64,
}

#[derive(Debug)]
struct PersistedArchiveEntry {
    archive: StoredArchive,
    _allocation: MemoryAllocation,
}

impl Deref for PersistedArchiveEntry {
    type Target = StoredArchive;

    fn deref(&self) -> &Self::Target {
        &self.archive
    }
}

#[derive(Debug)]
struct ChargedBytes {
    bytes: Vec<u8>,
    _allocation: MemoryAllocation,
}

#[derive(Debug)]
struct LoadedSessionReference {
    reference: SessionReference,
    _allocation: MemoryAllocation,
}

#[derive(Debug)]
struct LoadedSessionArchive {
    session_id: String,
    archive: StoredArchive,
    allocation: MemoryAllocation,
}

#[derive(Debug)]
struct PersistedArchiveReservation {
    _allocation: MemoryAllocation,
    serialized_bytes: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionReference {
    schema_version: u32,
    session_id: String,
    blob_sha256: String,
    last_used: u64,
}

#[derive(Debug)]
struct SessionReferenceLease {
    reference: PathBuf,
    quarantine: SessionReferenceQuarantinePaths,
    blobs_dir: PathBuf,
    refs_dir: PathBuf,
    quarantine_dir: PathBuf,
    archive_limit: u64,
    state: SessionReferenceLeaseState,
    discard_requested: bool,
    namespace_sync_pending: bool,
    recovery_slot: Option<RecoverySlot>,
    _metadata_allocation: Option<MemoryAllocation>,
    host: HostMemoryLedger,
}

#[derive(Debug)]
struct SessionReferenceQuarantinePaths {
    restore: PathBuf,
    superseded: PathBuf,
    discard: PathBuf,
}

fn session_reference_quarantine_paths(
    quarantine_dir: &Path,
    key: &str,
) -> SessionReferenceQuarantinePaths {
    let quarantine_id = format!("{key}-{}", Uuid::new_v4().simple());
    SessionReferenceQuarantinePaths {
        restore: quarantine_dir.join(format!(
            "{quarantine_id}.{}.json",
            SessionReferenceIntent::Restore.suffix()
        )),
        superseded: quarantine_dir.join(format!(
            "{quarantine_id}.{}.json",
            SessionReferenceIntent::Superseded.suffix()
        )),
        discard: quarantine_dir.join(format!(
            "{quarantine_id}.{}.json",
            SessionReferenceIntent::Discard.suffix()
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionReferenceLeaseState {
    Restore,
    Active,
    Superseded,
    Discard,
    Removed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionReferenceIntent {
    Restore,
    Superseded,
    Discard,
}

impl SessionReferenceIntent {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Restore => "restore",
            Self::Superseded => "superseded",
            Self::Discard => "discard",
        }
    }
}

#[derive(Debug)]
struct RecoverySlot {
    count: Arc<Mutex<usize>>,
    _allocation: Option<MemoryAllocation>,
}

#[derive(Debug, thiserror::Error)]
#[error("session reference recovery capacity is full")]
struct SessionRecoveryCapacity;

#[derive(Debug, thiserror::Error)]
#[error("session memory capacity is full")]
struct SessionMemoryCapacity;

impl Drop for RecoverySlot {
    fn drop(&mut self) {
        let mut count = self.count.lock().unwrap_or_else(|error| error.into_inner());
        *count = count.checked_sub(1).expect("recovery slot count underflow");
    }
}

impl SessionReferenceLease {
    fn discard_pending(&self) -> bool {
        self.discard_requested || self.state == SessionReferenceLeaseState::Discard
    }

    fn has_recovery_slot(&self) -> bool {
        self.recovery_slot.is_some()
    }

    fn active_path(&self) -> &Path {
        match self.state {
            SessionReferenceLeaseState::Restore => &self.quarantine.restore,
            SessionReferenceLeaseState::Active
            | SessionReferenceLeaseState::Superseded
            | SessionReferenceLeaseState::Removed => &self.reference,
            SessionReferenceLeaseState::Discard => &self.quarantine.discard,
        }
    }

    fn restore(&mut self) -> io::Result<()> {
        self.restore_with_sync(&sync_directory)
    }

    fn restore_with_sync<F>(&mut self, sync: &F) -> io::Result<()>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        if self.discard_requested {
            return Err(io::Error::other("session reference discard is pending"));
        }
        self.transition_restore_state(sync)?;
        if self.state == SessionReferenceLeaseState::Removed {
            return Ok(());
        }
        self.sync_pending_namespaces(sync)
    }

    fn transition_restore_state<F>(&mut self, sync: &F) -> io::Result<()>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        match self.state {
            SessionReferenceLeaseState::Restore => self.activate_or_retire_restore(sync),
            SessionReferenceLeaseState::Active => Ok(()),
            SessionReferenceLeaseState::Superseded => {
                self.retire_with_sync_and_remove(sync, &|path| fs::remove_file(path))
            }
            SessionReferenceLeaseState::Discard | SessionReferenceLeaseState::Removed => Err(
                io::Error::other("session reference lease cannot be restored"),
            ),
        }
    }

    fn activate_or_retire_restore<F>(&mut self, sync: &F) -> io::Result<()>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        if self.active_supersedes_restore()? {
            return self.retire_with_sync_and_remove(sync, &|path| fs::remove_file(path));
        }
        fs::rename(&self.quarantine.restore, &self.reference)?;
        self.state = SessionReferenceLeaseState::Active;
        self.namespace_sync_pending = true;
        Ok(())
    }

    fn active_supersedes_restore(&self) -> io::Result<bool> {
        if self.state != SessionReferenceLeaseState::Restore
            || !regular_reference_exists(&self.reference)?
        {
            return Ok(false);
        }
        let active = read_stored_session_reference_charged(&self.reference, &self.host)?;
        let quarantined =
            read_stored_session_reference_charged(&self.quarantine.restore, &self.host)?;
        if active.reference.session_id != quarantined.reference.session_id
            || active.reference.last_used < quarantined.reference.last_used
        {
            return Err(invalid_data(
                "session quarantine conflicts with its reference",
            ));
        }
        let _ = read_validated_session_blob(
            &self.blobs_dir,
            &active.reference,
            self.archive_limit,
            &self.host,
        )?;
        Ok(true)
    }

    fn retire(&mut self) -> io::Result<()> {
        self.retire_with_remove(&|path| fs::remove_file(path))
    }

    fn retire_with_remove<F>(&mut self, remove: &F) -> io::Result<()>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        self.retire_with_sync_and_remove(&sync_directory, remove)
    }

    fn retire_with_sync_and_remove<S, F>(&mut self, sync: &S, remove: &F) -> io::Result<()>
    where
        S: Fn(&Path) -> io::Result<()>,
        F: Fn(&Path) -> io::Result<()>,
    {
        if !self.prepare_retire(sync)? {
            return Ok(());
        }
        self.sync_pending_namespaces(sync)?;
        Self::remove_path(&self.quarantine.superseded, &self.quarantine_dir, remove)?;
        self.state = SessionReferenceLeaseState::Removed;
        self.recovery_slot.take();
        Ok(())
    }

    fn prepare_retire<S>(&mut self, sync: &S) -> io::Result<bool>
    where
        S: Fn(&Path) -> io::Result<()>,
    {
        if self.discard_requested {
            return Err(io::Error::other("session reference discard is pending"));
        }
        match self.state {
            SessionReferenceLeaseState::Restore => {
                fs::rename(&self.quarantine.restore, &self.quarantine.superseded)?;
                self.state = SessionReferenceLeaseState::Superseded;
                self.namespace_sync_pending = true;
                Ok(true)
            }
            SessionReferenceLeaseState::Active => self.finish_active_retire(sync),
            SessionReferenceLeaseState::Superseded => Ok(true),
            SessionReferenceLeaseState::Removed => Ok(false),
            SessionReferenceLeaseState::Discard => Err(io::Error::other(
                "discarded session reference cannot be superseded",
            )),
        }
    }

    fn finish_active_retire<S>(&mut self, sync: &S) -> io::Result<bool>
    where
        S: Fn(&Path) -> io::Result<()>,
    {
        self.sync_pending_namespaces(sync)?;
        self.state = SessionReferenceLeaseState::Removed;
        self.recovery_slot.take();
        Ok(false)
    }

    fn requarantine_with_sync<S>(&mut self, sync: &S) -> io::Result<()>
    where
        S: Fn(&Path) -> io::Result<()>,
    {
        if self.state != SessionReferenceLeaseState::Active {
            return Err(io::Error::other("session reference is not active"));
        }
        self.sync_pending_namespaces(sync)?;
        if !move_reference_to_quarantine(&self.reference, &self.quarantine.restore)? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "active session reference disappeared",
            ));
        }
        self.state = SessionReferenceLeaseState::Restore;
        self.namespace_sync_pending = true;
        self.sync_pending_namespaces(sync)
    }

    fn discard(&mut self) -> io::Result<()> {
        self.discard_with_remove(&|path| fs::remove_file(path))
    }

    fn discard_with_remove<F>(&mut self, remove: &F) -> io::Result<()>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        if self.state == SessionReferenceLeaseState::Removed {
            return Ok(());
        }
        self.discard_requested = true;
        self.mark_discard_intent()?;
        self.sync_pending_namespaces(&sync_directory)?;
        Self::remove_path(&self.reference, &self.refs_dir, remove)?;
        Self::remove_path(&self.quarantine.discard, &self.quarantine_dir, remove)?;
        self.state = SessionReferenceLeaseState::Removed;
        self.discard_requested = false;
        self.recovery_slot.take();
        Ok(())
    }

    fn mark_discard_intent(&mut self) -> io::Result<()> {
        let source = match self.state {
            SessionReferenceLeaseState::Restore => Some(&self.quarantine.restore),
            SessionReferenceLeaseState::Active => Some(&self.reference),
            SessionReferenceLeaseState::Superseded => Some(&self.quarantine.superseded),
            SessionReferenceLeaseState::Discard | SessionReferenceLeaseState::Removed => None,
        };
        if let Some(source) = source {
            fs::rename(source, &self.quarantine.discard)?;
            self.state = SessionReferenceLeaseState::Discard;
            self.namespace_sync_pending = true;
        }
        Ok(())
    }

    fn sync_pending_namespaces<F>(&mut self, sync: &F) -> io::Result<()>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        if !self.namespace_sync_pending {
            return Ok(());
        }
        sync(&self.refs_dir)?;
        sync(&self.quarantine_dir)?;
        self.namespace_sync_pending = false;
        Ok(())
    }

    fn remove_path<F>(path: &Path, directory: &Path, remove: &F) -> io::Result<()>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        match remove(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        sync_directory(directory)
    }
}

fn recovery_lease_action(lease: &SessionReferenceLease) -> io::Result<RecoveryLeaseAction> {
    if lease.discard_requested {
        return Ok(RecoveryLeaseAction::Discard);
    }
    match lease.state {
        SessionReferenceLeaseState::Restore if lease.active_supersedes_restore()? => {
            Ok(RecoveryLeaseAction::Supersede)
        }
        SessionReferenceLeaseState::Restore => Ok(RecoveryLeaseAction::Restore),
        SessionReferenceLeaseState::Active => Ok(RecoveryLeaseAction::Requarantine),
        SessionReferenceLeaseState::Superseded => Ok(RecoveryLeaseAction::Supersede),
        SessionReferenceLeaseState::Discard => Ok(RecoveryLeaseAction::Discard),
        SessionReferenceLeaseState::Removed => Ok(RecoveryLeaseAction::CheckActive),
    }
}

/// Stores references with an atomic namespace move for quarantine.
///
/// Unix moves sync the `refs` and `quarantine` directories before reporting
/// success. A failed cleanup leaves the reference outside the autoload
/// namespace. Other platforms retain atomic namespace moves without a crash
/// durability guarantee from directory syncing.
#[derive(Debug)]
struct SessionStore {
    root: PathBuf,
    host: HostMemoryLedger,
    recovery: Arc<Mutex<RecoveryTable>>,
    recovery_slots: Arc<Mutex<usize>>,
    _recovery_table_allocation: Option<MemoryAllocation>,
    recovery_limit: usize,
    archive_limit: u64,
    startup_archive_transition: Option<MemoryAllocation>,
}

type SessionStoreOpen = (SessionStore, Vec<LoadedSessionArchive>, u64);
type SessionMetadataSource = (PathBuf, Option<SessionReferenceLease>);

enum RecoveryLeaseResolution {
    Restorable(Box<SessionReferenceLease>),
    CheckActive,
    Missing,
}

enum RecoveryLeaseAction {
    Restore,
    Requarantine,
    Supersede,
    Discard,
    CheckActive,
}

enum RecoveryLeaseOutcome {
    Restorable,
    CheckActive,
    Missing,
}

struct RecoveryContext<'a> {
    blobs_dir: &'a Path,
    refs_dir: &'a Path,
    quarantine_dir: &'a Path,
    archive_limit: u64,
    host: &'a HostMemoryLedger,
}

fn resolve_recovery_action<F>(
    lease: &mut SessionReferenceLease,
    sync: &F,
) -> io::Result<RecoveryLeaseOutcome>
where
    F: Fn(&Path) -> io::Result<()>,
{
    match recovery_lease_action(lease) {
        Ok(RecoveryLeaseAction::Restore) => finish_restorable_recovery(lease, sync),
        Ok(RecoveryLeaseAction::Requarantine) => finish_active_recovery(lease, sync),
        Ok(RecoveryLeaseAction::Supersede) => finish_superseded_recovery(lease, sync),
        Ok(RecoveryLeaseAction::Discard) => finish_discard_recovery(lease),
        Ok(RecoveryLeaseAction::CheckActive) => Ok(RecoveryLeaseOutcome::CheckActive),
        Err(error) => Err(error),
    }
}

fn finish_restorable_recovery<F>(
    lease: &mut SessionReferenceLease,
    sync: &F,
) -> io::Result<RecoveryLeaseOutcome>
where
    F: Fn(&Path) -> io::Result<()>,
{
    lease.sync_pending_namespaces(sync)?;
    Ok(RecoveryLeaseOutcome::Restorable)
}

fn finish_active_recovery<F>(
    lease: &mut SessionReferenceLease,
    sync: &F,
) -> io::Result<RecoveryLeaseOutcome>
where
    F: Fn(&Path) -> io::Result<()>,
{
    lease.requarantine_with_sync(sync)?;
    Ok(RecoveryLeaseOutcome::Restorable)
}

fn finish_superseded_recovery<F>(
    lease: &mut SessionReferenceLease,
    sync: &F,
) -> io::Result<RecoveryLeaseOutcome>
where
    F: Fn(&Path) -> io::Result<()>,
{
    lease.retire_with_sync_and_remove(sync, &|path| fs::remove_file(path))?;
    Ok(RecoveryLeaseOutcome::CheckActive)
}

fn finish_discard_recovery(lease: &mut SessionReferenceLease) -> io::Result<RecoveryLeaseOutcome> {
    lease.discard()?;
    Ok(RecoveryLeaseOutcome::Missing)
}

fn recovery_lease_bound_for_root(root: &Path) -> io::Result<u64> {
    let root_bytes = u64::try_from(root.as_os_str().len())
        .map_err(|_| invalid_data("session store path is too large"))?;
    let fixed = u64::try_from(
        std::mem::size_of::<SessionReferenceLease>()
            .saturating_add(std::mem::size_of::<RecoverySlot>())
            .saturating_add(std::mem::size_of::<(String, SessionReferenceLease)>()),
    )
    .map_err(|_| invalid_data("session reference lease size does not fit a byte bound"))?;
    let path_bytes = checked_bound_add(root_bytes, 128, "session reference path bound overflowed")?;
    let path_bound = checked_bound_mul(path_bytes, 2, "session reference path bound overflowed")?;
    let all_paths = checked_bound_mul(path_bound, 4, "session reference lease bound overflowed")?;
    let fixed = checked_bound_add(fixed, 64, "session reference lease bound overflowed")?;
    checked_bound_add(fixed, all_paths, "session reference lease bound overflowed")
}

fn checked_bound_add(left: u64, right: u64, message: &'static str) -> io::Result<u64> {
    left.checked_add(right).ok_or_else(|| invalid_data(message))
}

fn checked_bound_mul(left: u64, right: u64, message: &'static str) -> io::Result<u64> {
    left.checked_mul(right).ok_or_else(|| invalid_data(message))
}

fn recovery_table_bound(limit: usize) -> Result<u64, Box<dyn Error>> {
    let count = u64::try_from(limit)
        .map_err(|_| invalid_data("session recovery limit does not fit a byte bound"))?;
    if count == 0 {
        return Ok(0);
    }
    let entry = u64::try_from(std::mem::size_of::<(String, SessionReferenceLease)>())
        .map_err(|_| invalid_data("session recovery entry does not fit a byte bound"))?;
    let bytes = checked_bound_mul(count, entry, "session recovery table size overflowed")?;
    Ok(checked_bound_add(
        bytes,
        64,
        "session recovery table size overflowed",
    )?)
}

fn persisted_cache_table_bound(limit: usize) -> Result<u64, Box<dyn Error>> {
    let count = u64::try_from(limit)
        .map_err(|_| invalid_data("persisted cache limit does not fit a byte bound"))?;
    if count == 0 {
        return Ok(0);
    }
    let bytes = checked_bound_mul(
        count,
        PERSISTED_ARCHIVE_TABLE_ENTRY_BYTES,
        "persisted cache table size overflowed",
    )?;
    Ok(checked_bound_add(
        bytes,
        128,
        "persisted cache table size overflowed",
    )?)
}

#[derive(Debug)]
struct BoundedTable<K, V> {
    entries: Vec<(K, V)>,
    limit: usize,
}

impl<K, V> BoundedTable<K, V> {
    fn with_capacity(limit: usize) -> Result<Self, Box<dyn Error>> {
        let mut entries = Vec::new();
        entries.try_reserve_exact(limit)?;
        if entries.capacity() != limit {
            return Err(invalid_data("bounded table allocation exceeded its limit").into());
        }
        Ok(Self { entries, limit })
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.entries
            .iter()
            .find(|(entry, _)| entry.borrow() == key)
            .map(|(_, value)| value)
    }

    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    fn can_insert<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.contains_key(key) || self.entries.len() < self.limit
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.remove_entry(key).map(|(_, value)| value)
    }

    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let index = self
            .entries
            .iter()
            .position(|(entry, _)| entry.borrow() == key)?;
        Some(self.entries.swap_remove(index))
    }

    fn insert(&mut self, key: K, value: V) -> Result<Option<V>, (K, V)>
    where
        K: Eq,
    {
        if let Some((_, entry)) = self.entries.iter_mut().find(|(entry, _)| *entry == key) {
            return Ok(Some(std::mem::replace(entry, value)));
        }
        if self.entries.len() >= self.limit {
            return Err((key, value));
        }
        self.entries.push((key, value));
        Ok(None)
    }
}

type RecoveryTable = BoundedTable<String, SessionReferenceLease>;

fn allocate_recovery_table(
    host: &HostMemoryLedger,
    cache_limit: usize,
) -> Result<(RecoveryTable, Option<MemoryAllocation>), Box<dyn Error>> {
    let allocation = if cache_limit == 0 {
        None
    } else {
        Some(
            host.allocate(recovery_table_bound(cache_limit)?)
                .map_err(|error| io::Error::other(format!("session recovery table: {error}")))?,
        )
    };
    let table = RecoveryTable::with_capacity(cache_limit).map_err(|error| {
        invalid_data(format!("session recovery table allocation failed: {error}"))
    })?;
    Ok((table, allocation))
}

fn reserve_startup_archive_transition(
    store: &mut SessionStore,
    capacity: usize,
) -> Result<(), Box<dyn Error>> {
    if capacity != 0 {
        let bytes = startup_archive_transition_bound(capacity)?;
        store.startup_archive_transition = Some(store.host.allocate(bytes)?);
    }
    Ok(())
}

impl SessionStore {
    const REFERENCE_SCHEMA_VERSION: u32 = 1;

    #[cfg(test)]
    fn open(root: PathBuf, cache_limit: usize) -> Result<SessionStoreOpen, Box<dyn Error>> {
        let host = HostMemoryLedger::new(NonZeroU64::new(u64::MAX).expect("nonzero host limit"));
        Self::open_with_archive_limit_and_host(
            root,
            cache_limit,
            DEFAULT_SESSION_ARCHIVE_BYTES,
            host,
        )
    }

    #[cfg(test)]
    fn open_with_archive_limit(
        root: PathBuf,
        cache_limit: usize,
        archive_limit: u64,
    ) -> Result<SessionStoreOpen, Box<dyn Error>> {
        let host = HostMemoryLedger::new(NonZeroU64::new(u64::MAX).expect("nonzero host limit"));
        Self::open_with_archive_limit_and_host(root, cache_limit, archive_limit, host)
    }

    fn open_with_archive_limit_and_host(
        root: PathBuf,
        cache_limit: usize,
        archive_limit: u64,
        host: HostMemoryLedger,
    ) -> Result<SessionStoreOpen, Box<dyn Error>> {
        fs::create_dir_all(root.join("blobs"))?;
        fs::create_dir_all(root.join("refs"))?;
        fs::create_dir_all(root.join("quarantine"))?;
        recover_quarantined_references(&root, archive_limit, &host)?;
        let (recovery, recovery_table_allocation) = allocate_recovery_table(&host, cache_limit)?;
        let mut store = Self {
            root,
            host,
            recovery: Arc::new(Mutex::new(recovery)),
            recovery_slots: Arc::new(Mutex::new(0)),
            _recovery_table_allocation: recovery_table_allocation,
            recovery_limit: cache_limit,
            archive_limit,
            startup_archive_transition: None,
        };
        let capacity = count_session_archive_entries(&store.root, cache_limit)?;
        reserve_startup_archive_transition(&mut store, capacity)?;
        let (archives, clock) = load_session_archives(&store, cache_limit, capacity)?;
        Ok((store, archives, clock))
    }

    #[cfg(test)]
    fn acquire_recovery_slot(&self) -> io::Result<RecoverySlot> {
        let bytes = recovery_lease_bound_for_root(&self.root)?;
        let allocation = self.host.allocate(bytes).map_err(recovery_memory_error)?;
        self.acquire_recovery_slot_with_allocation(allocation)
            .map_err(|(error, _allocation)| error)
    }

    fn acquire_recovery_slot_with_allocation(
        &self,
        allocation: MemoryAllocation,
    ) -> Result<RecoverySlot, (io::Error, MemoryAllocation)> {
        let mut count = self
            .recovery_slots
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if *count >= self.recovery_limit {
            return Err((io::Error::other(SessionRecoveryCapacity), allocation));
        }
        *count += 1;
        Ok(RecoverySlot {
            count: Arc::clone(&self.recovery_slots),
            _allocation: Some(allocation),
        })
    }

    fn persist(
        &self,
        session_id: &str,
        model_sha256: &str,
        checkpoint: leone::GenerationCheckpoint,
        last_used: u64,
    ) -> Result<StoredArchive, Box<dyn Error>> {
        self.persist_with_reference_sync(
            session_id,
            model_sha256,
            checkpoint,
            last_used,
            &sync_directory,
        )
    }

    fn persist_with_reference_sync<F>(
        &self,
        session_id: &str,
        model_sha256: &str,
        checkpoint: leone::GenerationCheckpoint,
        last_used: u64,
        sync: &F,
    ) -> Result<StoredArchive, Box<dyn Error>>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        let archive = leone::SessionArchive::from_checkpoint(model_sha256, checkpoint)?;
        let blob = archive.to_json()?;
        let blob_sha256 = sha256_bytes(&blob);
        let blob_path = self.root.join("blobs").join(format!("{blob_sha256}.json"));
        write_session_blob(&blob_path, &blob, &self.host)?;
        write_session_reference_with_sync(self, session_id, &blob_sha256, last_used, sync)?;
        Ok(StoredArchive {
            blob_sha256,
            last_used,
        })
    }

    fn lease_reference(&self, session_id: &str) -> io::Result<Option<SessionReferenceLease>> {
        self.lease_reference_with_sync(session_id, &sync_directory)
    }

    fn lease_reference_for_discard(
        &self,
        session_id: &str,
        allow_owner_held_without_slot: bool,
    ) -> io::Result<Option<SessionReferenceLease>> {
        if let Some(mut lease) = self.take_recovery_lease(session_id) {
            lease.discard_requested = true;
            return Ok(Some(lease));
        }
        let metadata_allocation = self.allocate_recovery_metadata()?;
        let key = sha256_bytes(session_id.as_bytes());
        let refs_dir = self.root.join("refs");
        let quarantine_dir = self.root.join("quarantine");
        let reference = refs_dir.join(format!("{key}.json"));
        if !discard_reference_may_exist(&reference, allow_owner_held_without_slot)? {
            return Ok(None);
        }
        let quarantine = session_reference_quarantine_paths(&quarantine_dir, &key);
        let (recovery_slot, metadata_allocation) = match self
            .acquire_recovery_slot_with_allocation(metadata_allocation)
        {
            Ok(slot) => (Some(slot), None),
            Err((_error, allocation)) if allow_owner_held_without_slot => (None, Some(allocation)),
            Err((error, _allocation)) => return Err(error),
        };
        Ok(Some(SessionReferenceLease {
            reference,
            quarantine,
            blobs_dir: self.root.join("blobs"),
            refs_dir,
            quarantine_dir,
            archive_limit: self.archive_limit,
            state: SessionReferenceLeaseState::Active,
            discard_requested: true,
            namespace_sync_pending: false,
            recovery_slot,
            _metadata_allocation: metadata_allocation,
            host: self.host.clone(),
        }))
    }

    fn allocate_recovery_metadata(&self) -> io::Result<MemoryAllocation> {
        self.host
            .allocate(recovery_lease_bound_for_root(&self.root)?)
            .map_err(recovery_memory_error)
    }

    fn lease_reference_with_sync<F>(
        &self,
        session_id: &str,
        sync: &F,
    ) -> io::Result<Option<SessionReferenceLease>>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        match self.resolve_recovery_lease_with_sync(session_id, sync)? {
            RecoveryLeaseResolution::Restorable(lease) => return Ok(Some(*lease)),
            RecoveryLeaseResolution::Missing => return Ok(None),
            RecoveryLeaseResolution::CheckActive => {}
        }
        self.lease_active_reference_with_sync(session_id, sync)
    }

    fn lease_active_reference_with_sync<F>(
        &self,
        session_id: &str,
        sync: &F,
    ) -> io::Result<Option<SessionReferenceLease>>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        let Some(mut lease) = self.prepare_active_reference_lease(session_id)? else {
            return Ok(None);
        };
        if let Err(error) = lease.sync_pending_namespaces(sync) {
            let _ = lease.restore_with_sync(sync);
            let mut lease = Some(lease);
            self.retain_recovery_lease(session_id, &mut lease)?;
            return Err(error);
        }
        Ok(Some(lease))
    }

    fn prepare_active_reference_lease(
        &self,
        session_id: &str,
    ) -> io::Result<Option<SessionReferenceLease>> {
        let recovery_allocation = self.allocate_recovery_metadata()?;
        let key = sha256_bytes(session_id.as_bytes());
        let refs_dir = self.root.join("refs");
        let quarantine_dir = self.root.join("quarantine");
        let reference = refs_dir.join(format!("{key}.json"));
        if !regular_reference_exists(&reference)? {
            return Ok(None);
        }
        let recovery_slot = match self.acquire_recovery_slot_with_allocation(recovery_allocation) {
            Ok(slot) => slot,
            Err((error, _allocation)) => return Err(error),
        };
        require_directory(&quarantine_dir, "session quarantine path")?;
        let quarantine = session_reference_quarantine_paths(&quarantine_dir, &key);
        if !move_reference_to_quarantine(&reference, &quarantine.restore)? {
            return Ok(None);
        }
        Ok(Some(SessionReferenceLease {
            reference,
            quarantine,
            blobs_dir: self.root.join("blobs"),
            refs_dir,
            quarantine_dir,
            archive_limit: self.archive_limit,
            state: SessionReferenceLeaseState::Restore,
            discard_requested: false,
            namespace_sync_pending: true,
            recovery_slot: Some(recovery_slot),
            _metadata_allocation: None,
            host: self.host.clone(),
        }))
    }

    fn retain_recovery_lease(
        &self,
        session_id: &str,
        lease: &mut Option<SessionReferenceLease>,
    ) -> io::Result<()> {
        // Taking a retained lease keeps its recovery slot, so its table slot remains available.
        let Some(value) = lease.take() else {
            return Ok(());
        };
        let mut recovery = self
            .recovery
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let key = sha256_bytes(session_id.as_bytes());
        if recovery.contains_key(&key) || recovery.len() >= self.recovery_limit {
            *lease = Some(value);
            return Err(io::Error::other("session recovery table is full"));
        }
        if let Err((_key, value)) = recovery.insert(key, value) {
            *lease = Some(value);
            return Err(io::Error::other("session recovery table is full"));
        }
        Ok(())
    }

    fn take_recovery_lease(&self, session_id: &str) -> Option<SessionReferenceLease> {
        let mut recovery = self
            .recovery
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        recovery.remove(&sha256_bytes(session_id.as_bytes()))
    }

    fn resolve_recovery_lease(&self, session_id: &str) -> io::Result<RecoveryLeaseResolution> {
        self.resolve_recovery_lease_with_sync(session_id, &sync_directory)
    }

    fn resolve_recovery_lease_with_sync<F>(
        &self,
        session_id: &str,
        sync: &F,
    ) -> io::Result<RecoveryLeaseResolution>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        let Some(mut lease) = self.take_recovery_lease(session_id) else {
            return Ok(RecoveryLeaseResolution::CheckActive);
        };
        let outcome = match resolve_recovery_action(&mut lease, sync) {
            Ok(outcome) => outcome,
            Err(error) => {
                let mut lease = Some(lease);
                self.retain_recovery_lease(session_id, &mut lease)?;
                return Err(error);
            }
        };
        match outcome {
            RecoveryLeaseOutcome::Restorable => {
                Ok(RecoveryLeaseResolution::Restorable(Box::new(lease)))
            }
            RecoveryLeaseOutcome::CheckActive => Ok(RecoveryLeaseResolution::CheckActive),
            RecoveryLeaseOutcome::Missing => Ok(RecoveryLeaseResolution::Missing),
        }
    }

    #[cfg(test)]
    fn load_metadata(&self, session_id: &str) -> Result<Option<StoredArchive>, Box<dyn Error>> {
        Ok(self
            .load_metadata_with_allocation(session_id)?
            .map(|loaded| loaded.archive))
    }

    fn load_metadata_with_allocation(
        &self,
        session_id: &str,
    ) -> Result<Option<LoadedSessionArchive>, Box<dyn Error>> {
        let Some((path, lease)) = self.metadata_source(session_id)? else {
            return Ok(None);
        };
        let result = self.load_metadata_at_path(session_id, &path);
        if let Some(lease) = lease {
            let mut lease = Some(lease);
            self.retain_recovery_lease(session_id, &mut lease)?;
        }
        result
    }

    fn metadata_source(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionMetadataSource>, Box<dyn Error>> {
        Ok(match self.resolve_recovery_lease(session_id)? {
            RecoveryLeaseResolution::Restorable(lease) => {
                Some((lease.active_path().to_owned(), Some(*lease)))
            }
            RecoveryLeaseResolution::CheckActive => Some((self.reference_path(session_id), None)),
            RecoveryLeaseResolution::Missing => None,
        })
    }

    fn load_metadata_at_path(
        &self,
        session_id: &str,
        path: &Path,
    ) -> Result<Option<LoadedSessionArchive>, Box<dyn Error>> {
        if !regular_reference_exists(path)? {
            return Ok(None);
        }
        let LoadedSessionReference {
            reference,
            _allocation: reference_allocation,
        } = read_session_reference_charged(path, session_id, &self.host)?;
        let archive = StoredArchive {
            blob_sha256: reference.blob_sha256,
            last_used: reference.last_used,
        };
        let allocation = self.allocate_archive_metadata(session_id, &archive)?;
        drop(reference_allocation);
        Ok(Some(LoadedSessionArchive {
            session_id: session_id.to_owned(),
            allocation,
            archive,
        }))
    }

    fn allocate_archive_metadata(
        &self,
        session_id: &str,
        archive: &StoredArchive,
    ) -> Result<MemoryAllocation, Box<dyn Error>> {
        let bytes = stored_archive_metadata_bound(session_id, archive)?;
        self.host
            .allocate(bytes)
            .map_err(|error| session_store_memory_error("session archive metadata", error))
            .map_err(Into::into)
    }

    fn has_reference(&self, session_id: &str) -> Result<bool, Box<dyn Error>> {
        Ok(self.load_metadata_with_allocation(session_id)?.is_some())
    }

    #[cfg(test)]
    fn load_archive(
        &self,
        session_id: &str,
        stored: &StoredArchive,
        lease: Option<&SessionReferenceLease>,
        model_sha256: &str,
    ) -> Result<leone::SessionArchive, Box<dyn Error>> {
        let blob_limit = self.archive_size(stored)?;
        self.load_archive_with_limit(session_id, stored, lease, model_sha256, blob_limit)
    }

    fn load_archive_with_limit(
        &self,
        session_id: &str,
        stored: &StoredArchive,
        lease: Option<&SessionReferenceLease>,
        model_sha256: &str,
        blob_limit: u64,
    ) -> Result<leone::SessionArchive, Box<dyn Error>> {
        let path = lease
            .map(|lease| lease.active_path().to_owned())
            .unwrap_or_else(|| self.reference_path(session_id));
        let loaded = read_session_reference_charged(&path, session_id, &self.host)?;
        if loaded.reference.blob_sha256 != stored.blob_sha256 {
            return Err(invalid_data("the persisted session reference changed").into());
        }
        let blob = load_session_blob(self, &loaded.reference, blob_limit)?;
        Ok(leone::SessionArchive::from_json(&blob.bytes, model_sha256)?)
    }

    fn archive_size(&self, stored: &StoredArchive) -> io::Result<u64> {
        let path = self
            .root
            .join("blobs")
            .join(format!("{}.json", stored.blob_sha256));
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.file_type().is_file() {
            return Err(invalid_data("session blob is not a regular file"));
        }
        if metadata.len() > self.archive_limit {
            return Err(invalid_data("session archive exceeds the configured size"));
        }
        Ok(metadata.len())
    }

    fn reference_path(&self, session_id: &str) -> PathBuf {
        let key = sha256_bytes(session_id.as_bytes());
        self.root.join("refs").join(format!("{key}.json"))
    }
}

fn recovery_memory_error(error: MemoryError) -> io::Error {
    session_store_memory_error("session reference recovery memory", error)
}

fn session_store_memory_error(context: &str, error: MemoryError) -> io::Error {
    if matches!(&error, MemoryError::BudgetExceeded { .. }) {
        io::Error::other(SessionMemoryCapacity)
    } else {
        io::Error::other(format!("{context}: {error}"))
    }
}

fn discard_reference_may_exist(
    reference: &Path,
    allow_owner_held_without_slot: bool,
) -> io::Result<bool> {
    match regular_reference_exists(reference) {
        Ok(exists) => Ok(exists),
        Err(_) if allow_owner_held_without_slot => Ok(true),
        Err(error) => Err(error),
    }
}

fn regular_reference_exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "session reference is not a regular file",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn require_directory(path: &Path, label: &str) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => Ok(()),
        Ok(_) => Err(io::Error::new(io::ErrorKind::InvalidData, label)),
        Err(error) => Err(error),
    }
}

fn move_reference_to_quarantine(reference: &Path, quarantine: &Path) -> io::Result<bool> {
    match fs::rename(reference, quarantine) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if regular_reference_exists(reference)? {
                Err(error)
            } else {
                Ok(false)
            }
        }
        Err(error) => Err(error),
    }
}

fn recover_quarantined_references(
    root: &Path,
    archive_limit: u64,
    host: &HostMemoryLedger,
) -> io::Result<()> {
    let refs_dir = root.join("refs");
    let quarantine_dir = root.join("quarantine");
    let blobs_dir = root.join("blobs");
    recover_quarantine_intent(
        SessionReferenceIntent::Discard,
        &blobs_dir,
        &refs_dir,
        &quarantine_dir,
        archive_limit,
        host,
    )?;
    recover_quarantine_intent(
        SessionReferenceIntent::Superseded,
        &blobs_dir,
        &refs_dir,
        &quarantine_dir,
        archive_limit,
        host,
    )?;
    recover_quarantine_intent(
        SessionReferenceIntent::Restore,
        &blobs_dir,
        &refs_dir,
        &quarantine_dir,
        archive_limit,
        host,
    )?;
    sync_directory(&refs_dir)?;
    sync_directory(&quarantine_dir)
}

fn recover_quarantine_intent(
    intent: SessionReferenceIntent,
    blobs_dir: &Path,
    refs_dir: &Path,
    quarantine_dir: &Path,
    archive_limit: u64,
    host: &HostMemoryLedger,
) -> io::Result<()> {
    let context = RecoveryContext {
        blobs_dir,
        refs_dir,
        quarantine_dir,
        archive_limit,
        host,
    };
    while let Some((quarantine, key)) = find_quarantine_intent(quarantine_dir, intent)? {
        let loaded = read_stored_session_reference_charged(&quarantine, host)?;
        let quarantined = &loaded.reference;
        match intent {
            SessionReferenceIntent::Discard => {
                recover_discarded_reference(
                    &key,
                    &quarantine,
                    context.refs_dir,
                    context.quarantine_dir,
                )?;
            }
            SessionReferenceIntent::Superseded => {
                recover_superseded_reference(quarantined, &quarantine, &key, &context)?
            }
            SessionReferenceIntent::Restore => {
                recover_restorable_reference(quarantined, &quarantine, &key, &context)?
            }
        }
    }
    Ok(())
}

fn find_quarantine_intent(
    quarantine_dir: &Path,
    intent: SessionReferenceIntent,
) -> io::Result<Option<(PathBuf, String)>> {
    for entry in fs::read_dir(quarantine_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Err(invalid_data(
                "session quarantine contains a non-regular entry",
            ));
        }
        let name = quarantine_reference_name(entry.file_name().as_ref())?;
        if name.intent == intent {
            return Ok(Some((entry.path(), name.key)));
        }
    }
    Ok(None)
}

fn recover_restorable_reference(
    quarantined: &SessionReference,
    quarantine: &Path,
    key: &str,
    context: &RecoveryContext<'_>,
) -> io::Result<()> {
    let reference = context.refs_dir.join(format!("{key}.json"));
    if regular_reference_exists(&reference)? {
        return reconcile_existing_reference(
            &reference,
            quarantined,
            quarantine,
            context.blobs_dir,
            context.quarantine_dir,
            context.archive_limit,
            context.host,
        );
    }
    restore_missing_reference(
        quarantine,
        &reference,
        context.refs_dir,
        context.quarantine_dir,
    )
}

fn recover_superseded_reference(
    quarantined: &SessionReference,
    quarantine: &Path,
    key: &str,
    context: &RecoveryContext<'_>,
) -> io::Result<()> {
    let reference = context.refs_dir.join(format!("{key}.json"));
    if !regular_reference_exists(&reference)? {
        return Err(invalid_data(
            "superseded session reference is missing its replacement",
        ));
    }
    reconcile_existing_reference(
        &reference,
        quarantined,
        quarantine,
        context.blobs_dir,
        context.quarantine_dir,
        context.archive_limit,
        context.host,
    )
}

fn reconcile_existing_reference(
    reference: &Path,
    quarantined: &SessionReference,
    quarantine: &Path,
    blobs_dir: &Path,
    quarantine_dir: &Path,
    archive_limit: u64,
    host: &HostMemoryLedger,
) -> io::Result<()> {
    let active = read_stored_session_reference_charged(reference, host)?;
    if active.reference.session_id != quarantined.session_id
        || active.reference.last_used < quarantined.last_used
    {
        return Err(invalid_data(
            "session quarantine conflicts with its reference",
        ));
    }
    let _ = read_validated_session_blob(blobs_dir, &active.reference, archive_limit, host)?;
    fs::remove_file(quarantine)?;
    sync_directory(quarantine_dir)
}

fn recover_discarded_reference(
    key: &str,
    discard: &Path,
    refs_dir: &Path,
    quarantine_dir: &Path,
) -> io::Result<()> {
    let reference = refs_dir.join(format!("{key}.json"));
    remove_regular_reference(&reference, refs_dir)?;
    remove_other_quarantined_references(key, discard, quarantine_dir)?;
    remove_regular_reference(discard, quarantine_dir)
}

fn remove_other_quarantined_references(
    key: &str,
    discard: &Path,
    quarantine_dir: &Path,
) -> io::Result<()> {
    for entry in fs::read_dir(quarantine_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Err(invalid_data(
                "session quarantine contains a non-regular entry",
            ));
        }
        let name = quarantine_reference_name(entry.file_name().as_ref())?;
        let path = entry.path();
        if name.key == key && path != discard {
            fs::remove_file(path)?;
        }
    }
    sync_directory(quarantine_dir)
}

fn remove_regular_reference(path: &Path, directory: &Path) -> io::Result<()> {
    if regular_reference_exists(path)? {
        fs::remove_file(path)?;
    }
    sync_directory(directory)
}

fn restore_missing_reference(
    quarantine: &Path,
    reference: &Path,
    refs_dir: &Path,
    quarantine_dir: &Path,
) -> io::Result<()> {
    fs::rename(quarantine, reference)?;
    sync_directory(refs_dir)?;
    sync_directory(quarantine_dir)
}

struct QuarantineReferenceName {
    key: String,
    intent: SessionReferenceIntent,
}

fn quarantine_reference_name(name: &std::ffi::OsStr) -> io::Result<QuarantineReferenceName> {
    let Some(name) = name.to_str() else {
        return Err(invalid_data("session quarantine name is not UTF-8"));
    };
    let Some(base) = name.strip_suffix(".json") else {
        return Err(invalid_data("session quarantine name is invalid"));
    };
    let (base, intent) = if let Some(base) = base.strip_suffix(".restore") {
        (base, SessionReferenceIntent::Restore)
    } else if let Some(base) = base.strip_suffix(".superseded") {
        (base, SessionReferenceIntent::Superseded)
    } else if let Some(base) = base.strip_suffix(".discard") {
        (base, SessionReferenceIntent::Discard)
    } else {
        (base, SessionReferenceIntent::Restore)
    };
    let Some((key, nonce)) = base.split_once('-') else {
        return Err(invalid_data("session quarantine name is invalid"));
    };
    if key.len() != 64 || !key.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid_data("session quarantine identity is invalid"));
    }
    if nonce.len() != 32 || !nonce.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid_data("session quarantine nonce is invalid"));
    }
    Ok(QuarantineReferenceName {
        key: key.to_owned(),
        intent,
    })
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn count_session_archive_entries(root: &Path, limit: usize) -> Result<usize, Box<dyn Error>> {
    if limit == 0 {
        return Ok(0);
    }
    let scan_limit = limit
        .checked_add(1)
        .ok_or_else(|| invalid_data("session archive count overflowed"))?;
    count_session_archive_entries_to(root, scan_limit)
}

fn count_session_archive_entries_to(
    root: &Path,
    scan_limit: usize,
) -> Result<usize, Box<dyn Error>> {
    let mut count = 0_usize;
    for entry in fs::read_dir(root.join("refs"))? {
        let entry = entry?;
        if !is_session_archive_entry(&entry)? {
            continue;
        }
        count = count
            .checked_add(1)
            .ok_or_else(|| invalid_data("session archive count overflowed"))?;
        if count == scan_limit {
            break;
        }
    }
    Ok(count)
}

fn is_session_archive_entry(entry: &std::fs::DirEntry) -> io::Result<bool> {
    Ok(entry.file_type()?.is_file() && !is_temporary_reference(&entry.file_name()))
}

fn load_session_archives(
    store: &SessionStore,
    cache_limit: usize,
    capacity: usize,
) -> Result<(Vec<LoadedSessionArchive>, u64), Box<dyn Error>> {
    if cache_limit == 0 {
        return Ok((Vec::new(), 0));
    }
    let mut cached = Vec::new();
    cached.try_reserve_exact(capacity).map_err(|error| {
        invalid_data(format!("session archive table allocation failed: {error}"))
    })?;
    let mut clock = 0;
    for entry in fs::read_dir(store.root.join("refs"))? {
        let entry = entry?;
        let Some(loaded) = load_session_archive_entry(&entry, &store.host)? else {
            continue;
        };
        clock = clock.max(loaded.archive.last_used);
        cache_session_archive(&mut cached, cache_limit, loaded);
    }
    Ok((cached, clock))
}

fn load_session_archive_entry(
    entry: &std::fs::DirEntry,
    host: &HostMemoryLedger,
) -> Result<Option<LoadedSessionArchive>, Box<dyn Error>> {
    if !entry.file_type()?.is_file() || is_temporary_reference(&entry.file_name()) {
        return Ok(None);
    }
    Ok(Some(load_session_metadata(entry, host)?))
}

fn cache_session_archive(
    cached: &mut Vec<LoadedSessionArchive>,
    cache_limit: usize,
    loaded: LoadedSessionArchive,
) {
    if cache_limit == 0 {
        return;
    }
    if cached.len() < cache_limit {
        cached.push(loaded);
        return;
    }
    let Some(oldest) = cached
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| archive_order(left).cmp(&archive_order(right)))
        .map(|(index, _)| index)
    else {
        return;
    };
    if archive_order(&loaded) > archive_order(&cached[oldest]) {
        cached[oldest] = loaded;
    }
}

fn archive_order(archive: &LoadedSessionArchive) -> (u64, &str) {
    (archive.archive.last_used, &archive.session_id)
}

fn is_temporary_reference(name: &std::ffi::OsStr) -> bool {
    name.to_str()
        .is_some_and(|name| name.starts_with('.') && name.ends_with(".tmp"))
}

#[cfg(test)]
fn read_session_reference(
    path: &Path,
    session_id: &str,
) -> Result<SessionReference, Box<dyn Error>> {
    let bytes = read_bounded_regular_file(path, MAX_SESSION_REFERENCE_BYTES)?;
    let reference: SessionReference = serde_json::from_slice(&bytes).map_err(invalid_json)?;
    validate_session_reference(path, &reference)?;
    if reference.session_id != session_id {
        return Err(invalid_data("session reference ID does not match its request").into());
    }
    Ok(reference)
}

fn read_session_reference_charged(
    path: &Path,
    session_id: &str,
    host: &HostMemoryLedger,
) -> Result<LoadedSessionReference, Box<dyn Error>> {
    let loaded = read_stored_session_reference_charged(path, host)?;
    let reference = &loaded.reference;
    if reference.session_id != session_id {
        return Err(invalid_data("session reference ID does not match its request").into());
    }
    Ok(loaded)
}

fn read_stored_session_reference_charged(
    path: &Path,
    host: &HostMemoryLedger,
) -> io::Result<LoadedSessionReference> {
    let bytes = read_bounded_regular_file_charged(path, MAX_SESSION_REFERENCE_BYTES, host)?;
    parse_session_reference_charged(path, &bytes.bytes, host)
}

fn parse_session_reference_charged(
    path: &Path,
    bytes: &[u8],
    host: &HostMemoryLedger,
) -> io::Result<LoadedSessionReference> {
    let allocation = host
        .allocate(SESSION_REFERENCE_PARSE_BYTES)
        .map_err(|error| session_store_memory_error("session reference parse memory", error))?;
    let reference: SessionReference = serde_json::from_slice(bytes).map_err(invalid_json)?;
    validate_session_reference(path, &reference)?;
    Ok(LoadedSessionReference {
        reference,
        _allocation: allocation,
    })
}

fn load_session_metadata(
    entry: &std::fs::DirEntry,
    host: &HostMemoryLedger,
) -> Result<LoadedSessionArchive, Box<dyn Error>> {
    let path = entry.path();
    let bytes = read_bounded_regular_file_charged(&path, MAX_SESSION_REFERENCE_BYTES, host)?;
    let loaded = parse_session_reference_charged(&path, &bytes.bytes, host)?;
    drop(bytes);
    let LoadedSessionReference {
        reference,
        _allocation: reference_allocation,
    } = loaded;
    let session_id = reference.session_id;
    let archive = StoredArchive {
        blob_sha256: reference.blob_sha256,
        last_used: reference.last_used,
    };
    let allocation = host
        .allocate(stored_archive_metadata_bound(&session_id, &archive)?)
        .map_err(|error| session_store_memory_error("session archive metadata", error))?;
    drop(reference_allocation);
    Ok(LoadedSessionArchive {
        session_id,
        archive,
        allocation,
    })
}

fn validate_session_reference(path: &Path, reference: &SessionReference) -> io::Result<()> {
    if reference.schema_version != SessionStore::REFERENCE_SCHEMA_VERSION {
        return Err(invalid_data(format!(
            "unsupported session reference schema {} in {}",
            reference.schema_version,
            path.display()
        )));
    }
    if reference.blob_sha256.len() != 64
        || !reference
            .blob_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(invalid_data("session reference blob digest is invalid"));
    }
    validate_session_identifier("stored session_id", &reference.session_id)?;
    let key = sha256_bytes(reference.session_id.as_bytes());
    let expected_name = format!("{key}.json");
    let valid_name = path
        .file_name()
        .map(|name| {
            name == std::ffi::OsStr::new(&expected_name)
                || quarantine_reference_name(name).is_ok_and(|name| name.key == key)
        })
        .unwrap_or(false);
    if !valid_name {
        return Err(invalid_data(format!(
            "session reference name does not match its session ID: {}",
            path.display()
        )));
    }
    Ok(())
}

fn load_session_blob(
    store: &SessionStore,
    reference: &SessionReference,
    limit: u64,
) -> Result<ChargedBytes, Box<dyn Error>> {
    Ok(read_validated_session_blob(
        &store.root.join("blobs"),
        reference,
        limit,
        &store.host,
    )?)
}

fn read_validated_session_blob(
    blobs_dir: &Path,
    reference: &SessionReference,
    archive_limit: u64,
    host: &HostMemoryLedger,
) -> io::Result<ChargedBytes> {
    let blob_path = blobs_dir.join(format!("{}.json", reference.blob_sha256));
    let blob = read_bounded_regular_file_charged(&blob_path, archive_limit, host)?;
    if sha256_bytes(&blob.bytes) != reference.blob_sha256 {
        return Err(invalid_data(format!(
            "session blob digest does not match its name: {}",
            blob_path.display()
        )));
    }
    Ok(blob)
}

fn regular_file_length(path: &Path, limit: u64) -> io::Result<u64> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(invalid_data("session store entry is not a regular file"));
    }
    if metadata.len() > limit {
        return Err(invalid_data(
            "session store entry exceeds its configured size",
        ));
    }
    Ok(metadata.len())
}

fn read_bounded_regular_file(path: &Path, limit: u64) -> io::Result<Vec<u8>> {
    regular_file_length(path, limit)?;
    let mut file = File::open(path)?;
    read_bounded_file(&mut file, limit)
}

fn read_bounded_regular_file_charged(
    path: &Path,
    limit: u64,
    host: &HostMemoryLedger,
) -> io::Result<ChargedBytes> {
    let allocation = host
        .allocate(limit)
        .map_err(|error| session_store_memory_error("session store read memory", error))?;
    let bytes = read_bounded_regular_file(path, limit)?;
    Ok(ChargedBytes {
        bytes,
        _allocation: allocation,
    })
}

fn read_bounded_file(file: &mut File, limit: u64) -> io::Result<Vec<u8>> {
    let capacity = bounded_file_capacity(file, limit)?;
    let mut bytes = bounded_read_buffer(capacity)?;
    Read::by_ref(file).take(limit).read_to_end(&mut bytes)?;
    reject_bounded_read_overflow(file, bytes.len(), limit)?;
    Ok(bytes)
}

fn bounded_file_capacity(file: &File, limit: u64) -> io::Result<usize> {
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err(invalid_data("session store entry is not a regular file"));
    }
    if metadata.len() > limit {
        return Err(invalid_data(
            "session store entry exceeds its configured size",
        ));
    }
    usize::try_from(limit)
        .map_err(|_| invalid_data("session store size limit does not fit memory capacity"))
}

fn bounded_read_buffer(capacity: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| io::Error::other(format!("session store read capacity: {error}")))?;
    Ok(bytes)
}

fn reject_bounded_read_overflow(file: &mut File, bytes: usize, limit: u64) -> io::Result<()> {
    let bytes =
        u64::try_from(bytes).map_err(|_| invalid_data("session store entry is too large"))?;
    if bytes != limit {
        return Ok(());
    }
    let mut extra = [0_u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(invalid_data(
            "session store entry exceeds its configured size",
        ));
    }
    Ok(())
}

fn write_session_blob(
    path: &Path,
    blob: &[u8],
    host: &HostMemoryLedger,
) -> Result<(), Box<dyn Error>> {
    write_session_blob_with_host(
        path,
        blob,
        &write_new_session_blob,
        &sync_directory,
        Some(host),
    )
}

#[cfg(test)]
fn write_session_blob_with<W, S>(
    path: &Path,
    blob: &[u8],
    write: &W,
    sync: &S,
) -> Result<(), Box<dyn Error>>
where
    W: Fn(File, &[u8]) -> io::Result<()>,
    S: Fn(&Path) -> io::Result<()>,
{
    write_session_blob_with_host(path, blob, write, sync, None)
}

fn write_session_blob_with_host<W, S>(
    path: &Path,
    blob: &[u8],
    write: &W,
    sync: &S,
    host: Option<&HostMemoryLedger>,
) -> Result<(), Box<dyn Error>>
where
    W: Fn(File, &[u8]) -> io::Result<()>,
    S: Fn(&Path) -> io::Result<()>,
{
    let directory = path
        .parent()
        .ok_or_else(|| invalid_data("session blob path has no parent directory"))?;
    if sync_existing_session_blob(path, blob, directory, sync, host)? {
        return Ok(());
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid_data("session blob name is not UTF-8"))?;
    let temporary = directory.join(format!(".{name}.{}.tmp", Uuid::new_v4().simple()));
    let result = publish_session_blob(path, &temporary, blob, write, sync, host);
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = sync(directory);
    }
    result
}

fn sync_existing_session_blob<S>(
    path: &Path,
    blob: &[u8],
    directory: &Path,
    sync: &S,
    host: Option<&HostMemoryLedger>,
) -> Result<bool, Box<dyn Error>>
where
    S: Fn(&Path) -> io::Result<()>,
{
    match fs::symlink_metadata(path) {
        Ok(_) => {
            verify_existing_session_blob_with_host(path, blob, host)?;
            sync(directory)?;
            Ok(true)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn publish_session_blob<W, S>(
    path: &Path,
    temporary: &Path,
    blob: &[u8],
    write: &W,
    sync: &S,
    host: Option<&HostMemoryLedger>,
) -> Result<(), Box<dyn Error>>
where
    W: Fn(File, &[u8]) -> io::Result<()>,
    S: Fn(&Path) -> io::Result<()>,
{
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary)?;
    write(file, blob)?;
    match fs::hard_link(temporary, path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            verify_existing_session_blob_with_host(path, blob, host)?;
        }
        Err(error) => return Err(error.into()),
    }
    fs::remove_file(temporary)?;
    sync(
        path.parent()
            .expect("temporary and canonical blob share a parent"),
    )?;
    Ok(())
}

fn write_new_session_blob(mut file: File, blob: &[u8]) -> io::Result<()> {
    file.write_all(blob)?;
    file.sync_all()
}

fn verify_existing_session_blob_with_host(
    path: &Path,
    blob: &[u8],
    host: Option<&HostMemoryLedger>,
) -> Result<(), Box<dyn Error>> {
    let limit = u64::try_from(blob.len())
        .map_err(|_| invalid_data("session blob length does not fit a byte bound"))?;
    let existing = match host {
        Some(host) => read_bounded_regular_file_charged(path, limit, host)?.bytes,
        None => read_bounded_regular_file(path, limit)?,
    };
    if existing != blob {
        return Err(invalid_data(format!("session blob collision at {}", path.display())).into());
    }
    Ok(())
}

#[cfg(test)]
fn write_session_reference(
    store: &SessionStore,
    session_id: &str,
    blob_sha256: &str,
    last_used: u64,
) -> Result<(), Box<dyn Error>> {
    write_session_reference_with_sync(store, session_id, blob_sha256, last_used, &sync_directory)
}

fn write_session_reference_with_sync<F>(
    store: &SessionStore,
    session_id: &str,
    blob_sha256: &str,
    last_used: u64,
    sync: &F,
) -> Result<(), Box<dyn Error>>
where
    F: Fn(&Path) -> io::Result<()>,
{
    let reference = SessionReference {
        schema_version: SessionStore::REFERENCE_SCHEMA_VERSION,
        session_id: session_id.to_owned(),
        blob_sha256: blob_sha256.to_owned(),
        last_used,
    };
    let key = sha256_bytes(session_id.as_bytes());
    let path = store.root.join("refs").join(format!("{key}.json"));
    let temporary = store
        .root
        .join("refs")
        .join(format!(".{key}.{}.tmp", Uuid::new_v4().simple()));
    let bytes = serde_json::to_vec(&reference)?;
    let result = (|| -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        sync(&store.root.join("refs"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(feature = "cuda")]
type ExecutionPlanSelection = crate::execution_plan::PlanSelection;

#[cfg(not(feature = "cuda"))]
type ExecutionPlanSelection = ();

struct Server<B: Backend> {
    runtime: Runtime<B>,
    model_id: String,
    model_sha256: String,
    sessions: HashMap<String, StoredSession<B>>,
    hibernated: HashMap<String, StoredHibernation>,
    persisted: BoundedTable<String, PersistedArchiveEntry>,
    _persisted_table_allocation: Option<MemoryAllocation>,
    persisted_cache_limit: usize,
    session_store: Option<SessionStore>,
    max_sessions: usize,
    max_hibernated_sessions: usize,
    clock: u64,
    kv_cache_dtype: KvCacheDtype,
    #[cfg(feature = "cuda")]
    execution_plan: Option<ExecutionPlanSelection>,
    prefill_chunk_tokens: usize,
    context_limit: usize,
    receipts: PathBuf,
    signing_key: SigningKey,
    memory: ServerMemory,
    cors_origins: Vec<String>,
    proxy_origin: Option<String>,
    metrics: ServerMetrics,
    identity: ServiceIdentity,
    #[cfg(test)]
    test_finalize_delay: Option<Duration>,
}

#[derive(Debug)]
struct ServerMemory {
    policy: ServiceMemoryPolicy,
    topology: PolicyMemoryTopology,
    host: HostMemoryLedger,
    logical_kv_reservation: u64,
    metrics_metadata: Option<MemoryAllocation>,
}

impl ServerMemory {
    fn with_topology(
        policy: ServiceMemoryPolicy,
        topology: PolicyMemoryTopology,
        host: HostMemoryLedger,
    ) -> Self {
        Self {
            policy,
            topology,
            host,
            logical_kv_reservation: 0,
            metrics_metadata: None,
        }
    }

    fn reserve_metrics_metadata(
        &mut self,
        max_active_requests: usize,
        max_queued_requests: usize,
        max_output_tokens: usize,
        max_output_bytes: usize,
    ) -> Result<(), Box<dyn Error>> {
        let bytes = metrics_metadata_bytes(
            max_active_requests,
            max_queued_requests,
            max_output_tokens,
            max_output_bytes,
        )?;
        self.metrics_metadata = Some(self.host.allocate(bytes).map_err(memory_runtime_error)?);
        Ok(())
    }

    fn reserve_host(&self, bytes: u64) -> Result<MemoryReservation, RuntimeError> {
        self.host
            .reserve(bytes)
            .map_err(|error| RuntimeError::Backend(BackendError::Memory(error)))
    }

    fn resolve_logical_kv<B: Backend>(
        &self,
        runtime: &Runtime<B>,
        dtype: KvCacheDtype,
        context_limit: usize,
        max_active_requests: usize,
    ) -> Result<u64, Box<dyn Error>> {
        let bytes_per_token = dtype.bytes_per_token(runtime.model().config())?;
        let active = host_u64(max_active_requests)?;
        let context = host_u64(context_limit)?;
        let backend = runtime.backend().memory_accounting();
        let (owned_non_kv, transient) = match self.policy.pools() {
            ServiceMemoryPools::Discrete { .. } => (backend.live_bytes, backend.reserved_bytes),
            ServiceMemoryPools::Shared { .. } => {
                let shared = self.host.root().snapshot();
                (shared.live_bytes, shared.reserved_bytes)
            }
        };
        let limit = self.policy.pools().combined_bytes().get();
        Ok(resolve_kv_reservation(
            self.policy.kv_reservation(),
            bytes_per_token,
            context,
            active,
            limit,
            owned_non_kv,
            transient,
        )?
        .get())
    }
    fn parent_observation<B: Backend>(&self, runtime: &Runtime<B>) -> ParentMemoryObservation {
        let accounting = runtime.backend().memory_tracker_root().snapshot();
        ParentMemoryObservation::new(
            metrics_memory_topology(self.topology),
            Some(accounting.peak_live_bytes),
            Some(accounting.peak_owned_and_reserved_bytes),
        )
    }
}

struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

struct ChatPlan {
    request: ChatRequest,
    stop_sequences: Vec<Vec<u8>>,
    prompt_tokens: Vec<u32>,
    options: GenerateOptions,
    session_id: String,
    source: ChatPlanSource,
    prefix_reused_tokens: usize,
    request_sha256: String,
    created: u64,
    completion_id: String,
    cors_origin: Option<String>,
}

fn request_metadata_context(plan: &ChatPlan) -> Result<usize, RuntimeError> {
    plan.prompt_tokens
        .len()
        .checked_add(plan.options.max_tokens)
        .ok_or(RuntimeError::SizeOverflow)
}

enum ChatPlanSource {
    Continue,
    ExplicitFork { parent_id: String },
    PrefixReuse { source_id: String },
}

type ChatPlanInput = (ChatRequest, Vec<Vec<u8>>, Vec<u32>);

struct PendingChat {
    plan: ChatPlan,
    stream: OutputSink,
    cancellation: Cancellation,
    deadline: std::time::Instant,
}

struct ChatTask<B: Backend> {
    request: ChatRequest,
    prompt_tokens: Vec<u32>,
    transcript: Vec<u32>,
    options: GenerateOptions,
    pending_prefill: Option<leone::PendingPrefill<B>>,
    request_replay: Option<leone::SessionReplay>,
    planned_prefix_tokens: usize,
    remaining_tokens: usize,
    session_id: String,
    stored: StoredSession<B>,
    stream: OutputSink,
    cancellation: Cancellation,
    deadline: std::time::Instant,
    deadline_expired: bool,
    deadline_expired_at_ns: Option<u64>,
    request_sha256: String,
    created: u64,
    completion_id: String,
    cors_origin: Option<String>,
    streamed: Utf8Stream,
    stop: StopMatcher,
    stop_hit: bool,
    tool_header_stop: bool,
    tokens: Vec<u32>,
    token_boundaries_ns: Vec<u64>,
    eos: bool,
    disconnected: bool,
    disconnected_at_ns: Option<u64>,
    failure: Option<String>,
    failure_effect: Option<SessionFailureEffect>,
    client_cancelled: bool,
    client_cancelled_at_ns: Option<u64>,
    headers_written: bool,
    prefill_chunks: u64,
    prefill_tokens: u64,
    prefill_processed: usize,
    decode_quanta: u64,
    phase_trace: Vec<DispatchKind>,
}

struct ChatAdmission(RefCell<Option<PendingChat>>);

struct ServerExecutor<B: Backend> {
    server: Server<B>,
    tasks: BTreeMap<RequestId, ChatTask<B>>,
    wire_request_ids: BTreeMap<String, RequestId>,
    queued_outputs: BTreeMap<RequestId, QueuedOutput>,
    terminal_outputs: VecDeque<TerminalOutput>,
    trace: ServiceTraceRecorder,
}

struct QueuedOutput {
    wire_request_id: String,
    stream: OutputSink,
    cancel_requested_at_ns: Option<u64>,
    deadline_expired_at_ns: Option<u64>,
}

struct TerminalOutput {
    request_id: RequestId,
    wire_request_id: String,
    telemetry: TransportTelemetryHandle,
    observed_queue_high_water_bytes: u64,
    observed_socket_blocked_ns: u64,
    observed_socket_blocked_events: u64,
    observed_interval_count: usize,
    observed_intervals_dropped: u64,
    observed_delivery_failed: bool,
}

const MAX_TERMINAL_OUTPUTS: usize = 256;

type ChatService<B> = ScheduledService<ServerExecutor<B>>;

const SERVICE_TRACE_SCHEMA: &str = "leone.service-trace.v1";
const MAX_SERVICE_TRACE_EVENTS: usize = 512;
const MAX_MEMORY_SAMPLES: usize = 128;
const TRACE_TEXT_BYTES: usize = 256;
const TRACE_MAP_ENTRY_HEADROOM: usize = std::mem::size_of::<usize>() * 8;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ServiceTraceEvent {
    PrefillChunk {
        request_id: u64,
        at_ns: u64,
        token_budget: u32,
        processed_tokens: u32,
        ready: bool,
        resident_request_ids: Vec<u64>,
    },
    ResidentDecodeProgress {
        request_id: u64,
        at_ns: u64,
        token_budget: u32,
        emitted_tokens: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        during_prefill_request_id: Option<u64>,
        during_prefill_request_ids: Vec<u64>,
        resident_request_ids: Vec<u64>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct TraceMemoryClass {
    live_bytes: u64,
    peak_live_bytes: u64,
    live_allocations: u64,
    peak_live_allocations: u64,
    allocations: u64,
    frees: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceUntrackedMemory {
    graph_objects: u64,
    library_handles: u64,
    execution_streams: u64,
    execution_events: u64,
    object_count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceParentMemory {
    topology: MetricsMemoryTopology,
    live_bytes: u64,
    reserved_bytes: u64,
    peak_live_bytes: u64,
    peak_live_and_reserved_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceHostMemory {
    live_bytes: u64,
    reserved_bytes: u64,
    peak_live_bytes: u64,
    peak_live_and_reserved_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceMemorySnapshot {
    live_bytes: u64,
    reserved_bytes: u64,
    peak_live_bytes: u64,
    peak_live_and_reserved_bytes: u64,
    live_allocations: u64,
    peak_live_allocations: u64,
    allocations: u64,
    frees: u64,
    classes: BTreeMap<String, TraceMemoryClass>,
    untracked: TraceUntrackedMemory,
    parent: TraceParentMemory,
    host: Option<TraceHostMemory>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceMemorySample {
    phase: String,
    request_id: Option<u64>,
    logical_reserved_kv_bytes: u64,
    memory: TraceMemorySnapshot,
}

#[derive(Debug, Serialize)]
struct ServiceTraceResponse {
    schema_version: &'static str,
    clock_domain: &'static str,
    source_id: String,
    process_instance_id: String,
    workload_epoch: String,
    events: Vec<ServiceTraceEvent>,
    memory_samples: Vec<TraceMemorySample>,
    logical_reserved_kv_bytes: u64,
    dropped_events: u64,
    dropped_memory_samples: u64,
}

#[derive(Debug)]
struct ServiceTraceRecorder {
    source_id: String,
    process_instance_id: String,
    workload_epoch: String,
    events: Vec<ServiceTraceEvent>,
    memory_samples: Vec<TraceMemorySample>,
    active_prefill: BTreeMap<RequestId, ()>,
    logical_reserved_kv_bytes: u64,
    dropped_events: u64,
    dropped_memory_samples: u64,
    memory_topology: MetricsMemoryTopology,
    host_tracker: Option<MemoryTracker>,
}

impl ServiceTraceRecorder {
    #[cfg(test)]
    fn new<B: Backend>(
        runtime: &Runtime<B>,
        memory_topology: MetricsMemoryTopology,
        host_tracker: Option<MemoryTracker>,
    ) -> Self {
        Self::with_identity(runtime, memory_topology, host_tracker, "", "", "")
    }

    fn with_identity<B: Backend>(
        runtime: &Runtime<B>,
        memory_topology: MetricsMemoryTopology,
        host_tracker: Option<MemoryTracker>,
        source_id: &str,
        process_instance_id: &str,
        workload_epoch: &str,
    ) -> Self {
        let mut recorder = Self {
            source_id: source_id.to_owned(),
            process_instance_id: process_instance_id.to_owned(),
            workload_epoch: workload_epoch.to_owned(),
            events: Vec::new(),
            memory_samples: Vec::new(),
            active_prefill: BTreeMap::new(),
            logical_reserved_kv_bytes: 0,
            dropped_events: 0,
            dropped_memory_samples: 0,
            memory_topology,
            host_tracker,
        };
        recorder.sample(runtime, "start", None, 0);
        recorder
    }

    fn begin_prefill(&mut self, request_id: RequestId) {
        self.active_prefill.insert(request_id, ());
    }

    fn end_prefill(&mut self, request_id: RequestId) {
        self.active_prefill.remove(&request_id);
    }

    fn prefill_event(
        &mut self,
        dispatch: Dispatch,
        progress: leone::service::PrefillProgress,
        resident_request_ids: &[RequestId],
    ) {
        self.push_event(ServiceTraceEvent::PrefillChunk {
            request_id: dispatch.request_id.0,
            at_ns: scheduler_now_ns(),
            token_budget: dispatch.token_budget,
            processed_tokens: progress.processed_tokens,
            ready: progress.ready,
            resident_request_ids: resident_request_ids
                .iter()
                .map(|request_id| request_id.0)
                .collect(),
        });
    }

    #[cfg(test)]
    fn decode_event(&mut self, dispatch: Dispatch, emitted_tokens: usize) {
        self.decode_event_with_residents(dispatch, emitted_tokens, &[]);
    }

    fn decode_event_with_residents(
        &mut self,
        dispatch: Dispatch,
        emitted_tokens: usize,
        resident_request_ids: &[RequestId],
    ) {
        if emitted_tokens == 0 {
            return;
        }
        let during_prefill_request_ids: Vec<u64> = self
            .active_prefill
            .keys()
            .map(|request_id| request_id.0)
            .collect();
        if during_prefill_request_ids.is_empty() && resident_request_ids.is_empty() {
            return;
        }
        let during_prefill_request_id =
            (during_prefill_request_ids.len() == 1).then(|| during_prefill_request_ids[0]);
        self.push_event(ServiceTraceEvent::ResidentDecodeProgress {
            request_id: dispatch.request_id.0,
            at_ns: scheduler_now_ns(),
            token_budget: dispatch.token_budget,
            emitted_tokens: u32::try_from(emitted_tokens).unwrap_or(u32::MAX),
            during_prefill_request_id,
            during_prefill_request_ids,
            resident_request_ids: resident_request_ids
                .iter()
                .map(|request_id| request_id.0)
                .collect(),
        });
    }

    fn push_event(&mut self, event: ServiceTraceEvent) {
        if self.events.len() < MAX_SERVICE_TRACE_EVENTS {
            self.events.push(event);
        } else {
            self.dropped_events = self.dropped_events.saturating_add(1);
        }
    }

    fn sample<B: Backend>(
        &mut self,
        runtime: &Runtime<B>,
        phase: &str,
        request_id: Option<RequestId>,
        logical_reserved_kv_bytes: u64,
    ) {
        self.logical_reserved_kv_bytes = logical_reserved_kv_bytes;
        if self.memory_samples.len() >= MAX_MEMORY_SAMPLES {
            self.dropped_memory_samples = self.dropped_memory_samples.saturating_add(1);
            return;
        }
        self.memory_samples.push(TraceMemorySample {
            phase: phase.to_owned(),
            request_id: request_id.map(|id| id.0),
            logical_reserved_kv_bytes,
            memory: trace_memory_snapshot(
                runtime.backend().memory_accounting(),
                runtime.backend().memory_tracker_root().snapshot(),
                self.host_tracker.as_ref().map(MemoryTracker::snapshot),
                self.memory_topology,
            ),
        });
    }

    fn response<B: Backend>(
        &mut self,
        runtime: &Runtime<B>,
        logical_reserved_kv_bytes: u64,
    ) -> ServiceTraceResponse {
        self.sample(runtime, "read", None, logical_reserved_kv_bytes);
        ServiceTraceResponse {
            schema_version: SERVICE_TRACE_SCHEMA,
            clock_domain: PRESSURE_CLOCK_DOMAIN,
            source_id: self.source_id.clone(),
            process_instance_id: self.process_instance_id.clone(),
            workload_epoch: self.workload_epoch.clone(),
            events: self.events.clone(),
            memory_samples: self.memory_samples.clone(),
            logical_reserved_kv_bytes: self.logical_reserved_kv_bytes,
            dropped_events: self.dropped_events,
            dropped_memory_samples: self.dropped_memory_samples,
        }
    }
}

fn trace_memory_snapshot(
    accounting: MemoryAccounting,
    parent: MemoryAccounting,
    host: Option<MemoryAccounting>,
    topology: MetricsMemoryTopology,
) -> TraceMemorySnapshot {
    let classes = MemoryClass::ALL
        .into_iter()
        .map(|class| {
            let stats = accounting.class(class);
            (
                class.name().to_owned(),
                TraceMemoryClass {
                    live_bytes: stats.live_bytes,
                    peak_live_bytes: stats.peak_live_bytes,
                    live_allocations: stats.live_allocations,
                    peak_live_allocations: stats.peak_live_allocations,
                    allocations: stats.allocations,
                    frees: stats.frees,
                },
            )
        })
        .collect();
    let untracked = accounting.untracked;
    TraceMemorySnapshot {
        live_bytes: accounting.live_bytes,
        reserved_bytes: accounting.reserved_bytes,
        peak_live_bytes: accounting.peak_live_bytes,
        peak_live_and_reserved_bytes: accounting.peak_owned_and_reserved_bytes,
        live_allocations: accounting.live_allocations,
        peak_live_allocations: accounting.peak_live_allocations,
        allocations: accounting.allocations,
        frees: accounting.frees,
        classes,
        untracked: TraceUntrackedMemory {
            graph_objects: untracked.graph_objects,
            library_handles: untracked.library_handles,
            execution_streams: untracked.execution_streams,
            execution_events: untracked.execution_events,
            object_count: untracked.object_count(),
        },
        parent: TraceParentMemory {
            topology,
            live_bytes: parent.live_bytes,
            reserved_bytes: parent.reserved_bytes,
            peak_live_bytes: parent.peak_live_bytes,
            peak_live_and_reserved_bytes: parent.peak_owned_and_reserved_bytes,
        },
        host: host.map(|snapshot| TraceHostMemory {
            live_bytes: snapshot.live_bytes,
            reserved_bytes: snapshot.reserved_bytes,
            peak_live_bytes: snapshot.peak_live_bytes,
            peak_live_and_reserved_bytes: snapshot.peak_owned_and_reserved_bytes,
        }),
    }
}

impl<B: Backend> ServerExecutor<B> {
    fn request_id_by_wire(&self, wire_request_id: &str) -> Option<RequestId> {
        self.wire_request_ids.get(wire_request_id).copied()
    }

    fn cancel_requested(&mut self, request_id: RequestId) -> bool {
        if let Some(task) = self.tasks.get_mut(&request_id) {
            task.client_cancelled = true;
            task.client_cancelled_at_ns = Some(scheduler_now_ns());
            return true;
        }
        if let Some(queued) = self.queued_outputs.get_mut(&request_id) {
            queued.cancel_requested_at_ns = Some(scheduler_now_ns());
            queued.stream.cancel();
            return true;
        }
        false
    }

    fn cancel(&mut self, request_id: RequestId) {
        if let Some(task) = self.tasks.get_mut(&request_id) {
            task.cancellation.cancel();
            task.stream.cancel();
            mark_transport_disconnect(task);
        }
    }

    fn expire(&mut self, request_id: RequestId) {
        if let Some(task) = self.tasks.get_mut(&request_id) {
            task.cancellation.cancel();
            task.deadline_expired = true;
            task.deadline_expired_at_ns = Some(scheduler_now_ns());
        } else if let Some(queued) = self.queued_outputs.get_mut(&request_id) {
            queued.deadline_expired_at_ns = Some(scheduler_now_ns());
            queued.stream.cancel();
        }
    }

    fn sample_trace(&mut self, phase: &str, reserved_kv_bytes: u64) {
        self.trace
            .sample(&self.server.runtime, phase, None, reserved_kv_bytes);
    }

    fn drain_transport_observations(&mut self) {
        for terminal in &mut self.terminal_outputs {
            let observation = terminal.telemetry.snapshot();
            let blocked_events = observation
                .socket_blocked_events
                .saturating_sub(terminal.observed_socket_blocked_events);
            if blocked_events != 0 {
                let blocked_ns = observation
                    .socket_blocked_ns
                    .saturating_sub(terminal.observed_socket_blocked_ns);
                self.server.metrics.slow_client_observed(
                    terminal.request_id,
                    blocked_ns,
                    SlowClientCause::WriteBlocked,
                );
                terminal.observed_socket_blocked_events = observation.socket_blocked_events;
                terminal.observed_socket_blocked_ns = observation.socket_blocked_ns;
            }
            if observation.queue_high_water_bytes > terminal.observed_queue_high_water_bytes
                && observation.delivery_failure_phase == DeliveryFailurePhase::EnqueueRejected
            {
                self.server.metrics.slow_client_observed(
                    terminal.request_id,
                    0,
                    SlowClientCause::OutputQueueHighWater,
                );
                terminal.observed_queue_high_water_bytes = observation.queue_high_water_bytes;
            }
            let interval_count = observation
                .socket_blocked_interval_count
                .min(observation.socket_blocked_intervals.len());
            for interval in &observation.socket_blocked_intervals
                [terminal.observed_interval_count..interval_count]
            {
                self.server.metrics.slow_client_interval_observed(
                    terminal.request_id,
                    &terminal.wire_request_id,
                    interval.start_ns,
                    interval.end_ns,
                    SlowClientCause::WriteBlocked,
                );
            }
            terminal.observed_interval_count = interval_count;
            let dropped_intervals = observation
                .socket_blocked_intervals_dropped
                .saturating_sub(terminal.observed_intervals_dropped);
            if dropped_intervals != 0 {
                self.server
                    .metrics
                    .transport_observation_dropped_count(dropped_intervals);
                terminal.observed_intervals_dropped = observation.socket_blocked_intervals_dropped;
            }
            if observation.delivery_failed && !terminal.observed_delivery_failed {
                let _ = self.server.metrics.mark_delivery_failed(
                    &terminal.wire_request_id,
                    delivery_status(observation.delivery_failure_phase),
                );
                terminal.observed_delivery_failed = true;
            }
        }
    }

    fn retain_terminal_output(
        &mut self,
        request_id: RequestId,
        wire_request_id: String,
        stream: OutputSink,
    ) {
        if self.terminal_outputs.len() == MAX_TERMINAL_OUTPUTS {
            self.terminal_outputs.pop_front();
            self.server.metrics.terminal_output_dropped();
        }
        let telemetry = stream.telemetry_handle();
        drop(stream);
        self.terminal_outputs.push_back(TerminalOutput {
            request_id,
            wire_request_id,
            telemetry,
            observed_queue_high_water_bytes: 0,
            observed_socket_blocked_ns: 0,
            observed_socket_blocked_events: 0,
            observed_interval_count: 0,
            observed_intervals_dropped: 0,
            observed_delivery_failed: false,
        });
    }

    fn register_queued_output(
        &mut self,
        request_id: RequestId,
        wire_request_id: String,
        stream: OutputSink,
    ) {
        self.queued_outputs.insert(
            request_id,
            QueuedOutput {
                wire_request_id,
                stream,
                cancel_requested_at_ns: None,
                deadline_expired_at_ns: None,
            },
        );
    }

    fn take_queued_output(&mut self, request_id: RequestId) -> Option<QueuedOutput> {
        self.queued_outputs.remove(&request_id)
    }

    fn execute_prefill(&mut self, dispatch: &Dispatch) -> Result<QuantumOutput, io::Error> {
        let budget = usize::try_from(dispatch.token_budget)
            .map_err(|_| invalid_data("prefill budget does not fit usize"))?;
        let output = {
            let task = self.tasks.get_mut(&dispatch.request_id).ok_or_else(|| {
                invalid_data(format!("unknown request {}", dispatch.request_id.0))
            })?;
            execute_task_prefill(&mut self.server, task, budget)?
        };
        if let Some(progress) = output.prefill {
            self.trace.begin_prefill(dispatch.request_id);
            self.trace.prefill_event(*dispatch, progress, &[]);
            if progress.ready || output.cancelled {
                self.trace.end_prefill(dispatch.request_id);
            }
        }
        Ok(output)
    }
}

impl<B: Backend> QuantumExecutor for ServerExecutor<B> {
    type Request = ChatAdmission;
    type Error = io::Error;

    fn begin(&mut self, request_id: RequestId, request: &Self::Request) -> Result<(), Self::Error> {
        if self.tasks.contains_key(&request_id) {
            return Err(invalid_data(format!(
                "executor state already exists for request {}",
                request_id.0
            )));
        }
        let mut pending = request
            .0
            .borrow_mut()
            .take()
            .ok_or_else(|| invalid_data("chat admission was already consumed"))?;
        let fork_source = match &pending.plan.source {
            ChatPlanSource::ExplicitFork { parent_id } => {
                Some(fork_source_for_session(&self.server, parent_id))
            }
            ChatPlanSource::Continue | ChatPlanSource::PrefixReuse { .. } => None,
        };
        let prefix_source = prefix_source_for_plan(&self.server, &pending.plan);
        let fork_started_at_ns = fork_source.map(|_| scheduler_now_ns());
        let stored = lease_pending_chat(&mut self.server, &mut pending)?;
        self.queued_outputs.remove(&request_id);
        record_chat_plan_metrics(&mut self.server, &pending.plan, prefix_source, fork_source);
        if let (Some(source), Some(started_at_ns)) = (fork_source, fork_started_at_ns) {
            self.server
                .metrics
                .fork_observed(source, scheduler_now_ns().saturating_sub(started_at_ns));
        }
        let ChatPlan {
            request,
            stop_sequences,
            prompt_tokens,
            options,
            session_id,
            source: _,
            prefix_reused_tokens,
            request_sha256,
            created,
            completion_id,
            cors_origin,
        } = pending.plan;
        let wire_request_id = completion_id.clone();
        let remaining_tokens = options.max_tokens;
        self.tasks.insert(
            request_id,
            ChatTask {
                request,
                transcript: prompt_tokens.clone(),
                prompt_tokens,
                options,
                pending_prefill: None,
                request_replay: None,
                planned_prefix_tokens: prefix_reused_tokens,
                remaining_tokens,
                session_id,
                stored,
                stream: pending.stream,
                cancellation: pending.cancellation,
                deadline: pending.deadline,
                deadline_expired: false,
                deadline_expired_at_ns: None,
                request_sha256,
                created,
                completion_id,
                cors_origin,
                streamed: Utf8Stream::default(),
                stop: StopMatcher::new(stop_sequences),
                stop_hit: false,
                tool_header_stop: false,
                tokens: Vec::with_capacity(remaining_tokens),
                token_boundaries_ns: Vec::with_capacity(remaining_tokens),
                eos: false,
                disconnected: false,
                disconnected_at_ns: None,
                failure: None,
                failure_effect: None,
                client_cancelled: false,
                client_cancelled_at_ns: None,
                headers_written: false,
                prefill_chunks: 0,
                prefill_tokens: 0,
                prefill_processed: 0,
                decode_quanta: 0,
                phase_trace: Vec::new(),
            },
        );
        self.wire_request_ids.insert(wire_request_id, request_id);
        Ok(())
    }

    fn execute(
        &mut self,
        request_id: RequestId,
        token_budget: u32,
    ) -> Result<QuantumOutput, Self::Error> {
        let task = self
            .tasks
            .get_mut(&request_id)
            .ok_or_else(|| invalid_data(format!("unknown request {}", request_id.0)))?;
        let model_id = self.server.model_id.clone();
        if write_executor_headers(task, &model_id).is_err() {
            record_decode_telemetry(task);
            return Ok(QuantumOutput::decode(Vec::new(), false, true));
        }
        let budget = usize::try_from(token_budget)
            .expect("u32 fits usize")
            .min(task.remaining_tokens);
        let (tokens, cancelled) =
            match execute_task_quantum(&mut self.server.runtime, task, budget, &model_id) {
                Ok(result) => result,
                Err(error) => {
                    record_decode_telemetry(task);
                    return Ok(quantum_error(task, error));
                }
            };
        if tokens.len() > budget {
            return Err(invalid_data(format!(
                "runtime emitted {} tokens for a {budget}-token quantum",
                tokens.len()
            )));
        }
        task.remaining_tokens -= tokens.len();
        task.transcript.extend_from_slice(&tokens);
        task.tokens.extend_from_slice(&tokens);
        task.eos = quantum_reached_eos(
            task.stop_hit,
            &tokens,
            budget,
            self.server.runtime.model().tokenizer().eos_token(),
            cancelled,
        );
        record_decode_telemetry(task);
        Ok(QuantumOutput {
            tokens,
            eos: task.eos || task.remaining_tokens == 0,
            cancelled,
            prefill: None,
        })
    }

    fn execute_dispatch(&mut self, dispatch: &Dispatch) -> Result<QuantumOutput, Self::Error> {
        match dispatch.kind {
            DispatchKind::Prefill => self.execute_prefill(dispatch),
            DispatchKind::Decode => {
                let output = self.execute(dispatch.request_id, dispatch.token_budget)?;
                let resident_request_ids = self.tasks.keys().copied().collect::<Vec<_>>();
                self.trace.decode_event_with_residents(
                    *dispatch,
                    output.tokens.len(),
                    &resident_request_ids,
                );
                Ok(output)
            }
        }
    }

    fn execute_batch(
        &mut self,
        dispatches: &[leone::scheduler::Dispatch],
    ) -> Result<Vec<QuantumOutput>, Self::Error> {
        let mut states = Vec::with_capacity(dispatches.len());
        for dispatch in dispatches {
            let Some(task) = self.tasks.remove(&dispatch.request_id) else {
                restore_batch_tasks(&mut self.tasks, &mut states)?;
                return Err(invalid_data(format!(
                    "unknown request {}",
                    dispatch.request_id.0
                )));
            };
            states.push(ServerBatchState::new(*dispatch, task));
        }
        let result = execute_server_batch(&mut self.server, &mut self.trace, &mut states);
        restore_batch_tasks(&mut self.tasks, &mut states)?;
        result
    }

    fn finish(&mut self, request_id: RequestId, status: RequestStatus) -> Result<(), Self::Error> {
        self.trace.end_prefill(request_id);
        let Some(task) = self.tasks.remove(&request_id) else {
            return finish_queued(self, request_id, status);
        };
        finish_active(self, request_id, status, task)
    }

    fn finish_at(
        &mut self,
        request_id: RequestId,
        status: RequestStatus,
        at_ns: u64,
    ) -> Result<(), Self::Error> {
        if status == RequestStatus::DeadlineExpired {
            if let Some(task) = self.tasks.get_mut(&request_id) {
                task.deadline_expired = true;
                task.deadline_expired_at_ns = Some(at_ns);
            }
            if let Some(output) = self.queued_outputs.get_mut(&request_id) {
                output.deadline_expired_at_ns = Some(at_ns);
            }
        }
        self.finish(request_id, status)
    }
}

fn finish_queued<B: Backend>(
    executor: &mut ServerExecutor<B>,
    request_id: RequestId,
    status: RequestStatus,
) -> Result<(), io::Error> {
    let Some(output) = executor.queued_outputs.remove(&request_id) else {
        return Err(invalid_data(format!("unknown request {}", request_id.0)));
    };
    let cancel_requested_at_ns = output.cancel_requested_at_ns;
    let deadline_expired_at_ns = output.deadline_expired_at_ns;
    let disconnected_at_ns = output
        .stream
        .transport_telemetry()
        .client_disconnected_at_ns;
    let disconnected = disconnected_at_ns.is_some();
    output.stream.cancel();
    executor.server.metrics.request_finished_with_wire_id(
        request_id,
        &output.wire_request_id,
        TerminalObservation {
            completed_at_ns: scheduler_now_ns(),
            status,
            cause: queued_terminal_cause_with_timestamps(
                status,
                cancel_requested_at_ns,
                deadline_expired_at_ns,
                disconnected_at_ns,
            ),
            disconnected,
            failed: false,
            reclaimed: true,
            delivery_status: DeliveryStatus::NotAttempted,
        },
    );
    executor.wire_request_ids.remove(&output.wire_request_id);
    executor.retain_terminal_output(request_id, output.wire_request_id, output.stream);
    Ok(())
}

fn finish_active<B: Backend>(
    executor: &mut ServerExecutor<B>,
    request_id: RequestId,
    status: RequestStatus,
    task: ChatTask<B>,
) -> Result<(), io::Error> {
    let wire_request_id = task.completion_id.clone();
    let output = task.stream.clone();
    executor.server.metrics.token_boundaries_partial(
        request_id,
        &task.token_boundaries_ns,
        task.tokens.len(),
        scheduler_now_ns(),
    );
    let terminal_context = FinishContext {
        client_cancelled: task.client_cancelled,
        client_cancelled_at_ns: task.client_cancelled_at_ns,
        deadline_expired: task.deadline_expired,
        deadline_expired_at_ns: task.deadline_expired_at_ns,
        disconnected: task.disconnected,
        disconnected_at_ns: task.disconnected_at_ns,
        failed: task.failure.is_some(),
    };
    if let Some(replay) = task.request_replay {
        executor.server.metrics.prefill_realized(
            task.prompt_tokens.len(),
            task.planned_prefix_tokens,
            replay,
        );
    }
    let result = executor.server.finish_scheduled_chat(task, status);
    let transport = output.transport_telemetry();
    let delivery_failure_phase = transport.delivery_failure_phase;
    let terminal = finish_observation(
        status,
        &terminal_context,
        &result,
        output.terminal_claimed(),
        delivery_failure_phase,
        transport.client_disconnected_at_ns,
    );
    executor
        .server
        .metrics
        .request_finished_with_wire_id(request_id, &wire_request_id, terminal);
    if result.is_err() {
        let _ = executor
            .server
            .metrics
            .mark_delivery_failed(&wire_request_id, delivery_status(delivery_failure_phase));
    }
    executor.wire_request_ids.remove(&wire_request_id);
    executor.retain_terminal_output(request_id, wire_request_id, output);
    if let Err(error) = result {
        if !is_disconnected(error.as_ref()) {
            return Err(invalid_data(error.to_string()));
        }
    }
    Ok(())
}

struct FinishContext {
    client_cancelled: bool,
    client_cancelled_at_ns: Option<u64>,
    deadline_expired: bool,
    deadline_expired_at_ns: Option<u64>,
    disconnected: bool,
    disconnected_at_ns: Option<u64>,
    failed: bool,
}

fn queued_terminal_cause(status: RequestStatus) -> TerminalCause {
    queued_terminal_cause_with_timestamps(status, None, None, None)
}

fn queued_terminal_cause_with_timestamps(
    status: RequestStatus,
    cancel_requested_at_ns: Option<u64>,
    deadline_expired_at_ns: Option<u64>,
    disconnected_at_ns: Option<u64>,
) -> TerminalCause {
    match status {
        RequestStatus::DeadlineExpired => TerminalCause::DeadlineExpired {
            requested_at_ns: deadline_expired_at_ns,
        },
        RequestStatus::Cancelled => cancelled_terminal_cause(
            cancel_requested_at_ns,
            deadline_expired_at_ns,
            disconnected_at_ns,
        ),
        RequestStatus::Rejected => TerminalCause::Rejected,
        RequestStatus::Finished => TerminalCause::Completed,
        RequestStatus::Queued | RequestStatus::Running => TerminalCause::Unclassified,
    }
}

fn cancelled_terminal_cause(
    cancel_requested_at_ns: Option<u64>,
    deadline_expired_at_ns: Option<u64>,
    disconnected_at_ns: Option<u64>,
) -> TerminalCause {
    if let Some(requested_at_ns) = deadline_expired_at_ns {
        return TerminalCause::DeadlineExpired {
            requested_at_ns: Some(requested_at_ns),
        };
    }
    if let Some(requested_at_ns) = cancel_requested_at_ns {
        return TerminalCause::ClientCancellation {
            requested_at_ns: Some(requested_at_ns),
        };
    }
    disconnected_at_ns
        .map(|requested_at_ns| TerminalCause::Disconnected {
            requested_at_ns: Some(requested_at_ns),
        })
        .unwrap_or(TerminalCause::ClientCancellation {
            requested_at_ns: None,
        })
}

fn finish_observation(
    status: RequestStatus,
    context: &FinishContext,
    result: &Result<(), Box<dyn Error>>,
    terminal_claimed: bool,
    failure_phase: DeliveryFailurePhase,
    client_disconnected_at_ns: Option<u64>,
) -> TerminalObservation {
    let transport_failed = failure_phase != DeliveryFailurePhase::None
        || (result_is_transport_failure(result) && !context.disconnected);
    let disconnected = failure_phase == DeliveryFailurePhase::None
        && (context.disconnected
            || (result_is_disconnected(result) && client_disconnected_at_ns.is_some()));
    let cause = if disconnected && !context.disconnected {
        TerminalCause::Disconnected {
            requested_at_ns: client_disconnected_at_ns,
        }
    } else {
        active_terminal_cause(status, context, transport_failed)
    };
    TerminalObservation {
        completed_at_ns: scheduler_now_ns(),
        status,
        cause,
        disconnected,
        failed: context.failed || result_is_execution_failure(result),
        reclaimed: true,
        delivery_status: terminal_delivery_status(result, terminal_claimed, failure_phase),
    }
}

fn result_is_transport_failure(result: &Result<(), Box<dyn Error>>) -> bool {
    result
        .as_ref()
        .is_err_and(|error| is_transport_failure(error.as_ref()))
}

fn result_is_execution_failure(result: &Result<(), Box<dyn Error>>) -> bool {
    result
        .as_ref()
        .is_err_and(|error| !is_transport_failure(error.as_ref()))
}

fn result_is_disconnected(result: &Result<(), Box<dyn Error>>) -> bool {
    result
        .as_ref()
        .is_err_and(|error| is_disconnected(error.as_ref()))
}

fn active_terminal_cause(
    status: RequestStatus,
    context: &FinishContext,
    transport_failed: bool,
) -> TerminalCause {
    if transport_failed {
        return TerminalCause::TransportFailure;
    }
    if context.failed {
        return TerminalCause::ExecutionFailure;
    }
    if context.deadline_expired {
        return TerminalCause::DeadlineExpired {
            requested_at_ns: context.deadline_expired_at_ns,
        };
    }
    if context.client_cancelled {
        return TerminalCause::ClientCancellation {
            requested_at_ns: context.client_cancelled_at_ns,
        };
    }
    if context.disconnected {
        return TerminalCause::Disconnected {
            requested_at_ns: context.disconnected_at_ns,
        };
    }
    queued_terminal_cause(status)
}

fn terminal_delivery_status(
    result: &Result<(), Box<dyn Error>>,
    terminal_claimed: bool,
    failure_phase: DeliveryFailurePhase,
) -> DeliveryStatus {
    if failure_phase != DeliveryFailurePhase::None {
        delivery_status(failure_phase)
    } else if result.is_err() {
        DeliveryStatus::Failed
    } else if terminal_claimed {
        DeliveryStatus::Queued
    } else {
        DeliveryStatus::NotAttempted
    }
}

fn delivery_status(phase: DeliveryFailurePhase) -> DeliveryStatus {
    match phase {
        DeliveryFailurePhase::None => DeliveryStatus::Failed,
        DeliveryFailurePhase::EnqueueRejected => DeliveryStatus::EnqueueRejected,
        DeliveryFailurePhase::SocketWrite => DeliveryStatus::SocketWriteFailed,
    }
}

fn response_delivery_status(result: &io::Result<()>, stream: &OutputSink) -> DeliveryStatus {
    match stream.transport_telemetry().delivery_failure_phase {
        DeliveryFailurePhase::None if result.is_ok() => DeliveryStatus::Queued,
        DeliveryFailurePhase::None => DeliveryStatus::Failed,
        phase => delivery_status(phase),
    }
}

fn queued_output_delivery_status(output: &QueuedOutput) -> DeliveryStatus {
    let telemetry = output.stream.transport_telemetry();
    match telemetry.delivery_failure_phase {
        DeliveryFailurePhase::None if telemetry.queue_high_water_bytes > 0 => {
            DeliveryStatus::Queued
        }
        DeliveryFailurePhase::None => DeliveryStatus::NotAttempted,
        phase => delivery_status(phase),
    }
}

fn lease_pending_chat<B: Backend>(
    server: &mut Server<B>,
    pending: &mut PendingChat,
) -> Result<StoredSession<B>, io::Error> {
    match server.lease_session(&pending.plan) {
        Ok(stored) => Ok(stored),
        Err(error) => {
            let capacity = session_lease_capacity_error(error.as_ref());
            eprintln!("session preparation failed: {error}");
            write_session_lease_error_with_origin(
                &mut pending.stream,
                capacity,
                pending.plan.cors_origin.clone(),
            )?;
            Err(io::Error::other(SESSION_PREPARATION_ERROR))
        }
    }
}

fn restore_batch_tasks<B: Backend>(
    tasks: &mut BTreeMap<RequestId, ChatTask<B>>,
    states: &mut [ServerBatchState<B>],
) -> Result<(), io::Error> {
    let mut missing = None;
    for state in states {
        let Some(task) = state.task.take() else {
            missing.get_or_insert(state.dispatch.request_id);
            continue;
        };
        tasks.insert(state.dispatch.request_id, task);
    }
    missing.map_or(Ok(()), |request_id| {
        Err(invalid_data(format!(
            "batch task {} was already consumed",
            request_id.0
        )))
    })
}

fn session_lease_capacity_error(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(source) = current {
        if direct_session_capacity_error(source) {
            return true;
        }
        current = source.source();
    }
    false
}

fn direct_session_capacity_error(error: &(dyn Error + 'static)) -> bool {
    error.is::<SessionRecoveryCapacity>()
        || error.is::<SessionMemoryCapacity>()
        || matches!(
            error.downcast_ref::<MemoryError>(),
            Some(MemoryError::BudgetExceeded { .. })
        )
        || error
            .downcast_ref::<io::Error>()
            .is_some_and(io_session_capacity_error)
}

fn io_session_capacity_error(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|source| {
        source.is::<SessionRecoveryCapacity>()
            || source.is::<SessionMemoryCapacity>()
            || matches!(
                source.downcast_ref::<MemoryError>(),
                Some(MemoryError::BudgetExceeded { .. })
            )
    })
}

fn quantum_reached_eos(
    stop_hit: bool,
    tokens: &[u32],
    budget: usize,
    eos: Option<u32>,
    cancelled: bool,
) -> bool {
    stop_hit || tokens.last().copied() == eos || (!cancelled && tokens.len() < budget)
}

struct ServerBatchState<B: Backend> {
    dispatch: leone::scheduler::Dispatch,
    task: Option<ChatTask<B>>,
    tokens: Vec<u32>,
    budget: usize,
    cancelled: bool,
    eos: bool,
    prefill: Option<leone::service::PrefillProgress>,
}

impl<B: Backend> ServerBatchState<B> {
    fn new(dispatch: leone::scheduler::Dispatch, task: ChatTask<B>) -> Self {
        Self {
            dispatch,
            task: Some(task),
            tokens: Vec::with_capacity(dispatch.token_budget as usize),
            budget: dispatch.token_budget as usize,
            cancelled: false,
            eos: false,
            prefill: None,
        }
    }

    fn task_mut(&mut self) -> Result<&mut ChatTask<B>, io::Error> {
        self.task
            .as_mut()
            .ok_or_else(|| invalid_data("batch task was already consumed"))
    }

    fn runnable(&self) -> bool {
        !self.cancelled && !self.eos && self.tokens.len() < self.budget
    }

    fn decode_runnable(&self) -> bool {
        self.dispatch.kind == DispatchKind::Decode && self.runnable()
    }
}

fn execute_server_batch<B: Backend>(
    server: &mut Server<B>,
    trace: &mut ServiceTraceRecorder,
    states: &mut [ServerBatchState<B>],
) -> Result<Vec<QuantumOutput>, io::Error> {
    prepare_server_batch_headers(server, states)?;
    if states
        .iter()
        .any(|state| state.dispatch.kind == DispatchKind::Prefill)
    {
        return execute_mixed_server_batch(server, trace, states);
    }
    execute_decode_states(server, trace, states)?;
    states
        .iter_mut()
        .map(server_batch_output)
        .collect::<Result<Vec<_>, _>>()
}

fn execute_mixed_server_batch<B: Backend>(
    server: &mut Server<B>,
    trace: &mut ServiceTraceRecorder,
    states: &mut [ServerBatchState<B>],
) -> Result<Vec<QuantumOutput>, io::Error> {
    mark_closed_batch_peers(states)?;
    let prefill_ids = mixed_prefill_ids(states);
    let resident_decode_ids = mixed_decode_ids(states);
    begin_mixed_prefill_trace(trace, &prefill_ids);
    execute_mixed_prefill_states(server, trace, states, &resident_decode_ids)?;
    execute_decode_states(server, trace, states)?;
    end_mixed_prefill_trace(trace, states);
    mixed_batch_outputs(states)
}

fn begin_mixed_prefill_trace(trace: &mut ServiceTraceRecorder, request_ids: &[RequestId]) {
    for request_id in request_ids.iter().copied() {
        trace.begin_prefill(request_id);
    }
}

fn execute_mixed_prefill_states<B: Backend>(
    server: &mut Server<B>,
    trace: &mut ServiceTraceRecorder,
    states: &mut [ServerBatchState<B>],
    resident_decode_ids: &[RequestId],
) -> Result<(), io::Error> {
    for state in states
        .iter_mut()
        .filter(|state| state.runnable() && state.dispatch.kind == DispatchKind::Prefill)
    {
        execute_mixed_prefill(server, state)?;
        if let Some(progress) = state.prefill {
            trace.prefill_event(state.dispatch, progress, resident_decode_ids);
            if progress.ready || state.cancelled {
                trace.end_prefill(state.dispatch.request_id);
            }
        }
    }
    Ok(())
}

fn execute_decode_states<B: Backend>(
    server: &mut Server<B>,
    trace: &mut ServiceTraceRecorder,
    states: &mut [ServerBatchState<B>],
) -> Result<(), io::Error> {
    while states.iter().any(ServerBatchState::decode_runnable) {
        mark_closed_decode_peers(states)?;
        run_unready_batch_tasks(server, states)?;
        run_ready_batch_tasks(server, states)?;
    }
    let resident_request_ids = states
        .iter()
        .map(|state| state.dispatch.request_id)
        .collect::<Vec<_>>();
    record_decode_progress(trace, states, &resident_request_ids)?;
    Ok(())
}

fn end_mixed_prefill_trace<B: Backend>(
    trace: &mut ServiceTraceRecorder,
    states: &[ServerBatchState<B>],
) {
    for state in states.iter().filter(|state| {
        state.dispatch.kind == DispatchKind::Prefill
            && (state.cancelled || state.prefill.is_some_and(|progress| progress.ready))
    }) {
        trace.end_prefill(state.dispatch.request_id);
    }
}

fn mixed_batch_outputs<B: Backend>(
    states: &mut [ServerBatchState<B>],
) -> Result<Vec<QuantumOutput>, io::Error> {
    states
        .iter_mut()
        .map(server_batch_output)
        .collect::<Result<Vec<_>, _>>()
}

fn mixed_prefill_ids<B: Backend>(states: &[ServerBatchState<B>]) -> Vec<RequestId> {
    states
        .iter()
        .filter(|state| state.runnable() && state.dispatch.kind == DispatchKind::Prefill)
        .map(|state| state.dispatch.request_id)
        .collect()
}

fn mixed_decode_ids<B: Backend>(states: &[ServerBatchState<B>]) -> Vec<RequestId> {
    states
        .iter()
        .filter(|state| state.runnable() && state.dispatch.kind == DispatchKind::Decode)
        .map(|state| state.dispatch.request_id)
        .collect()
}

fn execute_mixed_prefill<B: Backend>(
    server: &mut Server<B>,
    state: &mut ServerBatchState<B>,
) -> Result<(), io::Error> {
    let budget = state.budget;
    let output = execute_task_prefill(server, state.task_mut()?, budget)?;
    state.prefill = output.prefill;
    state.cancelled = output.cancelled;
    Ok(())
}

fn record_decode_progress<B: Backend>(
    trace: &mut ServiceTraceRecorder,
    states: &mut [ServerBatchState<B>],
    resident_request_ids: &[RequestId],
) -> Result<(), io::Error> {
    for state in states
        .iter_mut()
        .filter(|state| state.dispatch.kind == DispatchKind::Decode)
    {
        let emitted = state.tokens.len();
        record_decode_telemetry(state.task_mut()?);
        trace.decode_event_with_residents(state.dispatch, emitted, resident_request_ids);
    }
    Ok(())
}

fn prepare_server_batch_headers<B: Backend>(
    server: &Server<B>,
    states: &mut [ServerBatchState<B>],
) -> Result<(), io::Error> {
    for state in states {
        if write_executor_headers(state.task_mut()?, &server.model_id).is_err() {
            state.cancelled = true;
        }
    }
    Ok(())
}

fn mark_closed_batch_peers<B: Backend>(
    states: &mut [ServerBatchState<B>],
) -> Result<(), io::Error> {
    for state in states.iter_mut().filter(|state| state.runnable()) {
        mark_closed_batch_peer(state)?;
    }
    Ok(())
}

fn mark_closed_decode_peers<B: Backend>(
    states: &mut [ServerBatchState<B>],
) -> Result<(), io::Error> {
    for state in states.iter_mut().filter(|state| state.decode_runnable()) {
        mark_closed_batch_peer(state)?;
    }
    Ok(())
}

fn mark_closed_batch_peer<B: Backend>(state: &mut ServerBatchState<B>) -> Result<(), io::Error> {
    let task = state.task_mut()?;
    if task.cancellation.is_cancelled() || task.stream.is_cancelled() {
        mark_transport_disconnect(task);
        state.cancelled = true;
    }
    Ok(())
}

fn run_unready_batch_tasks<B: Backend>(
    server: &mut Server<B>,
    states: &mut [ServerBatchState<B>],
) -> Result<(), io::Error> {
    for state in states.iter_mut().filter(|state| state.decode_runnable()) {
        let remaining = state.budget - state.tokens.len();
        let task = state
            .task
            .as_mut()
            .ok_or_else(|| invalid_data("batch task was already consumed"))?;
        if server.runtime.batch_session_ready(
            &task.stored.generation,
            &task.transcript,
            &task.options,
        ) {
            continue;
        }
        let budget = if task.stored.generation.is_empty() {
            1
        } else {
            remaining
        };
        let model_id = server.model_id.clone();
        let eos = server.runtime.model().tokenizer().eos_token();
        match execute_task_quantum(&mut server.runtime, task, budget, &model_id) {
            Ok((tokens, cancelled)) => {
                record_serial_batch_tokens(state, tokens, cancelled, budget, eos)?
            }
            Err(error) => {
                let _ = quantum_error(task, error);
                state.cancelled = true;
            }
        }
    }
    Ok(())
}

fn record_serial_batch_tokens<B: Backend>(
    state: &mut ServerBatchState<B>,
    tokens: Vec<u32>,
    cancelled: bool,
    requested: usize,
    eos: Option<u32>,
) -> Result<(), io::Error> {
    let stopped_early = tokens.len() < requested;
    let emitted_eos = tokens.last().copied() == eos;
    let count = tokens.len();
    let task = state
        .task
        .as_mut()
        .ok_or_else(|| invalid_data("batch task was already consumed"))?;
    task.remaining_tokens = task.remaining_tokens.saturating_sub(count);
    task.transcript.extend_from_slice(&tokens);
    task.tokens.extend_from_slice(&tokens);
    state.tokens.extend_from_slice(&tokens);
    state.cancelled = cancelled && !task.stop_hit;
    state.eos = task.stop_hit || (!cancelled && (emitted_eos || stopped_early));
    task.eos = state.eos;
    Ok(())
}

fn run_ready_batch_tasks<B: Backend>(
    server: &mut Server<B>,
    states: &mut [ServerBatchState<B>],
) -> Result<(), io::Error> {
    let ready = ready_batch_indices(server, states)?;
    if ready.is_empty() {
        return Ok(());
    }
    if ready.len() == 1 {
        return run_one_ready_task(server, &mut states[ready[0]]);
    }
    let generated = generate_ready_batch(server, states, &ready)?;
    match generated {
        Ok(tokens) => record_ready_batch_tokens(server, states, &ready, tokens),
        Err(error) => {
            fail_ready_batch(states, &ready, &error)?;
            Ok(())
        }
    }
}

fn generate_ready_batch<B: Backend>(
    server: &mut Server<B>,
    states: &mut [ServerBatchState<B>],
    ready: &[usize],
) -> Result<Result<Vec<GeneratedToken>, RuntimeError>, io::Error> {
    let mut inputs = Vec::with_capacity(ready.len());
    for (index, state) in states.iter_mut().enumerate() {
        if ready.binary_search(&index).is_err() {
            continue;
        }
        let task = state.task_mut()?;
        if let Err(error) = capture_reused_request_replay(&server.runtime, task) {
            return Ok(Err(error));
        }
        let ChatTask {
            stored,
            transcript,
            options,
            ..
        } = task;
        inputs.push(BatchSession {
            session: &mut stored.generation,
            transcript,
            options,
        });
    }
    Ok(server.runtime.generate_session_batch_token(&mut inputs))
}

fn ready_batch_indices<B: Backend>(
    server: &Server<B>,
    states: &mut [ServerBatchState<B>],
) -> Result<Vec<usize>, io::Error> {
    let mut ready = Vec::new();
    for (index, state) in states.iter_mut().enumerate() {
        if !state.decode_runnable() {
            continue;
        }
        let task = state.task_mut()?;
        if server.runtime.batch_session_ready(
            &task.stored.generation,
            &task.transcript,
            &task.options,
        ) {
            ready.push(index);
        }
    }
    Ok(ready)
}

fn run_one_ready_task<B: Backend>(
    server: &mut Server<B>,
    state: &mut ServerBatchState<B>,
) -> Result<(), io::Error> {
    let model_id = server.model_id.clone();
    let eos = server.runtime.model().tokenizer().eos_token();
    let task = state
        .task
        .as_mut()
        .ok_or_else(|| invalid_data("batch task was already consumed"))?;
    match execute_task_quantum(&mut server.runtime, task, 1, &model_id) {
        Ok((tokens, cancelled)) => record_serial_batch_tokens(state, tokens, cancelled, 1, eos),
        Err(error) => {
            let _ = quantum_error(task, error);
            state.cancelled = true;
            Ok(())
        }
    }
}

fn record_ready_batch_tokens<B: Backend>(
    server: &Server<B>,
    states: &mut [ServerBatchState<B>],
    ready: &[usize],
    tokens: Vec<leone::GeneratedToken>,
) -> Result<(), io::Error> {
    if tokens.len() != ready.len() {
        return Err(invalid_data("runtime batch output count differs"));
    }
    let eos = server.runtime.model().tokenizer().eos_token();
    let architecture = server.runtime.model().config().architecture;
    let header_token = server
        .runtime
        .model()
        .tokenizer()
        .token_id(LLAMA_HEADER_START)
        .ok();
    let engine_completion_ns = scheduler_now_ns();
    for (&index, token) in ready.iter().zip(tokens) {
        let tool_header_token = states[index].task.as_ref().and_then(|task| {
            official_llama_tool_header_token(&task.request, architecture, header_token)
        });
        record_ready_batch_token(
            &server.model_id,
            &mut states[index],
            token,
            eos,
            tool_header_token,
            engine_completion_ns,
        )?;
    }
    Ok(())
}

fn record_ready_batch_token<B: Backend>(
    model_id: &str,
    state: &mut ServerBatchState<B>,
    token: leone::GeneratedToken,
    eos: Option<u32>,
    tool_header_token: Option<u32>,
    engine_completion_ns: u64,
) -> Result<(), io::Error> {
    let task = state
        .task
        .as_mut()
        .ok_or_else(|| invalid_data("batch task was already consumed"))?;
    let stop_hit = Cell::new(task.stop_hit);
    let tool_header_stop = Cell::new(task.tool_header_stop);
    let mut context = ChatStreamContext {
        stream_mode: task.request.stream,
        tool_mode: effective_tool_mode(&task.request),
        tool_header_token,
        streamed: &mut task.streamed,
        stop: &mut task.stop,
        stop_hit: &stop_hit,
        tool_header_stop: &tool_header_stop,
        stream: &mut task.stream,
        token_boundaries_ns: &mut task.token_boundaries_ns,
        token_boundary_ns: Some(engine_completion_ns),
        model_id,
        completion_id: &task.completion_id,
        created: task.created,
        disconnected: &mut task.disconnected,
        disconnected_at_ns: &mut task.disconnected_at_ns,
    };
    if stream_quantum_token(&mut context, token.id, &token.bytes).is_err() {
        state.cancelled = true;
    }
    task.stop_hit = stop_hit.get();
    task.tool_header_stop = tool_header_stop.get();
    task.remaining_tokens = task.remaining_tokens.saturating_sub(1);
    task.transcript.push(token.id);
    task.tokens.push(token.id);
    state.tokens.push(token.id);
    state.eos = task.stop_hit || Some(token.id) == eos;
    task.eos = state.eos;
    Ok(())
}

fn fail_ready_batch<B: Backend>(
    states: &mut [ServerBatchState<B>],
    ready: &[usize],
    error: &RuntimeError,
) -> Result<(), io::Error> {
    for &index in ready {
        let state = &mut states[index];
        let task = state.task_mut()?;
        record_task_failure(task, error);
        state.cancelled = true;
    }
    Ok(())
}

fn server_batch_output<B: Backend>(
    state: &mut ServerBatchState<B>,
) -> Result<QuantumOutput, io::Error> {
    let task = state
        .task
        .as_ref()
        .ok_or_else(|| invalid_data("batch task was already consumed"))?;
    Ok(QuantumOutput {
        tokens: std::mem::take(&mut state.tokens),
        eos: state.eos || task.remaining_tokens == 0,
        cancelled: state.cancelled,
        prefill: batch_prefill_progress(state.dispatch.kind, state.prefill.take(), state.cancelled),
    })
}

fn batch_prefill_progress(
    kind: DispatchKind,
    progress: Option<leone::service::PrefillProgress>,
    cancelled: bool,
) -> Option<leone::service::PrefillProgress> {
    if kind == DispatchKind::Prefill && cancelled && progress.is_none() {
        return QuantumOutput::prefill(0, false, true).prefill;
    }
    progress
}

fn write_executor_headers<B: Backend>(
    task: &mut ChatTask<B>,
    model_id: &str,
) -> Result<(), io::Error> {
    if !task.request.stream || task.headers_written {
        return Ok(());
    }
    task.stream.mark_streaming();
    if let Err(error) = write_chat_headers_io(
        &mut task.stream,
        &task.session_id,
        &task.completion_id,
        task.created,
        model_id,
        task.cors_origin.as_deref(),
    ) {
        mark_transport_disconnect(task);
        return Err(error);
    }
    task.headers_written = true;
    Ok(())
}

fn execute_task_quantum<B: Backend>(
    runtime: &mut Runtime<B>,
    task: &mut ChatTask<B>,
    budget: usize,
    model_id: &str,
) -> Result<(Vec<u32>, bool), RuntimeError> {
    if task.pending_prefill.is_some() {
        return Err(RuntimeError::PrefillPending);
    }
    capture_reused_request_replay(runtime, task)?;
    let mut options = task.options.clone();
    options.max_tokens = budget;
    let architecture = runtime.model().config().architecture;
    let header_token = runtime
        .model()
        .tokenizer()
        .token_id(LLAMA_HEADER_START)
        .ok();
    let tool_header_token =
        official_llama_tool_header_token(&task.request, architecture, header_token);
    let generation = &mut task.stored.generation;
    let transcript = &task.transcript;
    let cancellation = task.cancellation.clone();
    let stream_cancelled = task.stream.clone();
    let stop_hit = Cell::new(task.stop_hit);
    let tool_header_stop = Cell::new(task.tool_header_stop);
    let mut stream_context = ChatStreamContext {
        stream_mode: task.request.stream,
        tool_mode: effective_tool_mode(&task.request),
        tool_header_token,
        streamed: &mut task.streamed,
        stop: &mut task.stop,
        stop_hit: &stop_hit,
        tool_header_stop: &tool_header_stop,
        stream: &mut task.stream,
        token_boundaries_ns: &mut task.token_boundaries_ns,
        token_boundary_ns: None,
        model_id,
        completion_id: &task.completion_id,
        created: task.created,
        disconnected: &mut task.disconnected,
        disconnected_at_ns: &mut task.disconnected_at_ns,
    };
    let result = runtime.generate_session_tokens_with_stop(
        generation,
        transcript,
        options,
        |token| stream_quantum_token(&mut stream_context, token.id, &token.bytes),
        || cancellation.is_cancelled() || stream_cancelled.is_cancelled(),
        || stop_hit.get(),
    )?;
    task.stop_hit = stop_hit.get();
    task.tool_header_stop = tool_header_stop.get();
    task.request_replay.get_or_insert(generation.last_replay());
    Ok((
        result.tokens,
        result.termination == GenerationTermination::Cancelled,
    ))
}

fn capture_reused_request_replay<B: Backend>(
    runtime: &Runtime<B>,
    task: &mut ChatTask<B>,
) -> Result<(), RuntimeError> {
    if task.request_replay.is_none() {
        let replay = runtime.plan_session_replay(
            &task.stored.generation,
            &task.prompt_tokens,
            &task.options,
        )?;
        if replay.reused_tokens == task.prompt_tokens.len() {
            task.request_replay = Some(replay);
        }
    }
    Ok(())
}

fn execute_task_prefill<B: Backend>(
    server: &mut Server<B>,
    task: &mut ChatTask<B>,
    budget: usize,
) -> Result<QuantumOutput, io::Error> {
    let model_id = server.model_id.clone();
    if write_executor_headers(task, &model_id).is_err() {
        return Ok(quantum_cancelled(task));
    }
    let Some(budget) = NonZeroUsize::new(budget) else {
        return Err(invalid_data("prefill budget must be nonzero"));
    };
    if task.cancellation.is_cancelled() || task.stream.is_cancelled() {
        mark_transport_disconnect(task);
        return Ok(quantum_cancelled(task));
    }
    let pending = match take_or_begin_prefill(server, task) {
        Ok(pending) => pending,
        Err(error) => return Ok(prefill_error(task, error)),
    };
    let progress = match advance_task_prefill(server, task, pending, budget) {
        Ok(progress) => progress,
        Err(error) => return Ok(prefill_error(task, error)),
    };
    let before = task.prefill_processed;
    record_prefill_telemetry(task, DispatchKind::Prefill, &progress);
    finish_task_prefill(server, task, progress, before)
}

fn take_or_begin_prefill<B: Backend>(
    server: &mut Server<B>,
    task: &mut ChatTask<B>,
) -> Result<leone::PendingPrefill<B>, RuntimeError> {
    task.pending_prefill.take().map_or_else(
        || {
            server.runtime.begin_prefill(
                &mut task.stored.generation,
                &task.transcript,
                task.options.clone(),
            )
        },
        Ok,
    )
}

fn advance_task_prefill<B: Backend>(
    server: &mut Server<B>,
    task: &mut ChatTask<B>,
    pending: leone::PendingPrefill<B>,
    budget: NonZeroUsize,
) -> Result<leone::PrefillProgress<B>, RuntimeError> {
    server.runtime.advance_prefill(pending, budget, || {
        task.cancellation.is_cancelled() || task.stream.is_cancelled()
    })
}

fn finish_task_prefill<B: Backend>(
    server: &mut Server<B>,
    task: &mut ChatTask<B>,
    progress: leone::PrefillProgress<B>,
    before: usize,
) -> Result<QuantumOutput, io::Error> {
    match progress {
        leone::PrefillProgress::Pending(pending) => {
            let processed = pending.processed_tokens().saturating_sub(before);
            task.pending_prefill = Some(pending);
            Ok(QuantumOutput::prefill(
                u32::try_from(processed).unwrap_or(u32::MAX),
                false,
                false,
            ))
        }
        leone::PrefillProgress::Ready(ready) => {
            let processed = ready.processed_tokens().saturating_sub(before);
            if let Err(error) = server
                .runtime
                .finish_prefill(ready, &mut task.stored.generation)
            {
                return Ok(prefill_error(task, error));
            }
            task.request_replay = Some(task.stored.generation.last_replay());
            Ok(QuantumOutput::prefill(
                u32::try_from(processed).unwrap_or(u32::MAX),
                true,
                false,
            ))
        }
        leone::PrefillProgress::Cancelled(cancelled) => Ok(QuantumOutput::prefill(
            u32::try_from(cancelled.processed_tokens().saturating_sub(before)).unwrap_or(u32::MAX),
            false,
            true,
        )),
    }
}

fn stream_quantum_token(
    context: &mut ChatStreamContext<'_>,
    token_id: u32,
    bytes: &[u8],
) -> Result<(), RuntimeError> {
    let token_index = context.token_boundaries_ns.len();
    let engine_boundary_ns = context.token_boundary_ns.unwrap_or_else(scheduler_now_ns);
    context.token_boundaries_ns.push(engine_boundary_ns);
    if context.stream_mode {
        emit_token_boundary(context, token_index, engine_boundary_ns)?;
    }
    if context.tool_header_token == Some(token_id) {
        context.stop_hit.set(true);
        context.tool_header_stop.set(true);
        return Ok(());
    }
    if !context.stream_mode {
        return Ok(());
    }
    let safe = context.stop.push(bytes);
    context.stop_hit.set(context.stop.hit());
    if !context.stream_mode || context.tool_mode {
        return Ok(());
    }
    emit_stream_bytes(context, &safe)?;
    Ok(())
}

fn emit_token_boundary(
    context: &mut ChatStreamContext<'_>,
    token_index: usize,
    engine_boundary_ns: u64,
) -> Result<(), RuntimeError> {
    let token_index = u64::try_from(token_index)
        .map_err(|_| RuntimeError::token_callback("token index does not fit u64"))?;
    write_sse(
        context.stream,
        &json!({
            "id": context.completion_id,
            "object": "chat.completion.chunk",
            "created": context.created,
            "model": context.model_id,
            "choices": [],
            "usage": null,
            "leone_telemetry": {
                "token_index": token_index,
                "engine_token_boundary_ns": engine_boundary_ns,
            }
        }),
    )
    .map_err(|error| RuntimeError::token_callback(error.to_string()))
}

fn flush_stream_pending(context: &mut ChatStreamContext<'_>) -> Result<(), RuntimeError> {
    let pending = context.stop.finish();
    if !context.stream_mode || context.tool_mode {
        return Ok(());
    }
    emit_stream_bytes(context, &pending)
}

fn flush_task_pending<B: Backend>(
    task: &mut ChatTask<B>,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    let stop_hit = Cell::new(task.stop_hit);
    let tool_header_stop = Cell::new(task.tool_header_stop);
    let mut context = ChatStreamContext {
        stream_mode: task.request.stream,
        tool_mode: effective_tool_mode(&task.request),
        tool_header_token: None,
        streamed: &mut task.streamed,
        stop: &mut task.stop,
        stop_hit: &stop_hit,
        tool_header_stop: &tool_header_stop,
        stream: &mut task.stream,
        token_boundaries_ns: &mut task.token_boundaries_ns,
        token_boundary_ns: None,
        model_id,
        completion_id: &task.completion_id,
        created: task.created,
        disconnected: &mut task.disconnected,
        disconnected_at_ns: &mut task.disconnected_at_ns,
    };
    let result = flush_stream_pending(&mut context);
    task.stop_hit = stop_hit.get();
    task.tool_header_stop = tool_header_stop.get();
    result.map_err(|error| -> Box<dyn Error> { error.into() })
}

fn emit_stream_bytes(
    context: &mut ChatStreamContext<'_>,
    bytes: &[u8],
) -> Result<(), RuntimeError> {
    for content in context.streamed.push(bytes) {
        if write_sse(
            context.stream,
            &json!({
                "id": context.completion_id,
                "object": "chat.completion.chunk",
                "created": context.created,
                "model": context.model_id,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}],
                "usage": null
            }),
        )
        .is_err()
        {
            let telemetry = context.stream.transport_telemetry();
            if telemetry.delivery_failure_phase == DeliveryFailurePhase::None {
                if let Some(at_ns) = telemetry.client_disconnected_at_ns {
                    *context.disconnected = true;
                    *context.disconnected_at_ns = Some(at_ns);
                }
            }
            return Err(RuntimeError::token_callback("client disconnected"));
        }
    }
    Ok(())
}

fn mark_transport_disconnect<B: Backend>(task: &mut ChatTask<B>) {
    if task.client_cancelled || task.deadline_expired {
        return;
    }
    let telemetry = task.stream.transport_telemetry();
    if telemetry.delivery_failure_phase != DeliveryFailurePhase::None {
        return;
    }
    if let Some(at_ns) = telemetry.client_disconnected_at_ns {
        task.disconnected = true;
        task.disconnected_at_ns = Some(at_ns);
    }
}

fn quantum_error<B: Backend>(task: &mut ChatTask<B>, error: RuntimeError) -> QuantumOutput {
    record_task_failure(task, &error);
    QuantumOutput::decode(Vec::new(), false, true)
}

fn quantum_cancelled<B: Backend>(_task: &mut ChatTask<B>) -> QuantumOutput {
    QuantumOutput::prefill(0, false, true)
}

fn record_task_failure<B: Backend>(task: &mut ChatTask<B>, error: &RuntimeError) {
    task.failure_effect = error.session_failure_effect();
    if !task.disconnected {
        task.failure = Some(error.to_string());
    }
}

fn prefill_error<B: Backend>(task: &mut ChatTask<B>, error: RuntimeError) -> QuantumOutput {
    record_task_failure(task, &error);
    QuantumOutput::prefill(0, false, true)
}

fn record_phase<B: Backend>(task: &mut ChatTask<B>, kind: DispatchKind) {
    const MAX_PHASE_TRACE: usize = 64;
    if task.phase_trace.len() < MAX_PHASE_TRACE {
        task.phase_trace.push(kind);
    }
}

fn record_decode_telemetry<B: Backend>(task: &mut ChatTask<B>) {
    record_phase(task, DispatchKind::Decode);
    task.decode_quanta = task.decode_quanta.saturating_add(1);
}

fn record_prefill_telemetry<B: Backend>(
    task: &mut ChatTask<B>,
    kind: DispatchKind,
    progress: &leone::PrefillProgress<B>,
) {
    record_phase(task, kind);
    task.prefill_chunks = task.prefill_chunks.saturating_add(1);
    let processed = match progress {
        leone::PrefillProgress::Pending(pending) => pending.processed_tokens(),
        leone::PrefillProgress::Ready(ready) => ready.processed_tokens(),
        leone::PrefillProgress::Cancelled(cancelled) => cancelled.processed_tokens(),
    };
    let delta = processed.saturating_sub(task.prefill_processed);
    task.prefill_processed = processed;
    task.prefill_tokens = task
        .prefill_tokens
        .saturating_add(u64::try_from(delta).unwrap_or(u64::MAX));
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    mark_process_start();
    run_inner(arguments, None)
}

pub(crate) fn preflight(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    validate_remote_access(&arguments)?;
    Ok(())
}

pub(crate) fn mark_process_start() {
    let _ = PROCESS_START_NS.set(process_start_unix_ns());
}

fn process_start_unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .unwrap_or(0)
}

pub(crate) fn run_verified(
    arguments: &[String],
    artifact: &crate::registry::VerifiedArtifact,
) -> Result<(), Box<dyn Error>> {
    mark_process_start();
    run_inner(arguments, Some(artifact))
}

fn run_inner(
    arguments: &[String],
    trusted_artifact: Option<&crate::registry::VerifiedArtifact>,
) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    if let Some(artifact) = trusted_artifact {
        if arguments.model != artifact.path() {
            return Err(invalid_data("verified model path does not match server model").into());
        }
    }
    validate_remote_access(&arguments)?;
    let interrupted = install_interrupt_handler()?;
    let (signing_key, model_sha256) = load_server_identity(
        &arguments,
        trusted_artifact.map(|artifact| artifact.sha256()),
    )?;
    serve_backend(arguments, signing_key, model_sha256, interrupted)
}

fn validate_remote_access(arguments: &ServeArgs) -> Result<(), Box<dyn Error>> {
    if !arguments.allow_remote && !arguments.bind.ip().is_loopback() {
        return Err(
            invalid_data("serve refuses a non-loopback address without --allow-remote").into(),
        );
    }
    Ok(())
}

fn install_interrupt_handler() -> Result<Arc<AtomicBool>, Box<dyn Error>> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let handler = Arc::clone(&interrupted);
    ctrlc::set_handler(move || handler.store(true, Ordering::Relaxed))?;
    Ok(interrupted)
}

fn load_server_identity(
    arguments: &ServeArgs,
    trusted_sha256: Option<&str>,
) -> Result<(SigningKey, String), Box<dyn Error>> {
    let signing_key = load_or_create_key(&arguments.signing_key)?;
    let model_sha256 = match trusted_sha256 {
        Some(value) => value.to_owned(),
        None => sha256_file(&arguments.model)?,
    };
    Ok((signing_key, model_sha256))
}

fn running_service_identity(model_sha256: &str) -> Result<ServiceIdentity, Box<dyn Error>> {
    let executable_sha256 = sha256_file(std::env::current_exe()?)?;
    let source_id = format!(
        "leone-cli:{}:{}",
        env!("LEONE_SOURCE_COMMIT"),
        env!("LEONE_SOURCE_DIRTY")
    );
    ServiceIdentity::new(
        source_id,
        executable_sha256,
        model_sha256.to_owned(),
        *PROCESS_START_NS.get_or_init(process_start_unix_ns),
    )
    .ok_or_else(|| invalid_data("running service identity is invalid").into())
}

fn serve_backend(
    arguments: ServeArgs,
    signing_key: SigningKey,
    model_sha256: String,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    match arguments.backend {
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => serve(
            CudaBackend::new(0)?,
            BackendChoice::Cuda,
            arguments,
            signing_key,
            model_sha256,
            interrupted,
        ),
        #[cfg(not(feature = "cuda"))]
        BackendChoice::Cuda => validate_compiled(BackendChoice::Cuda).map_err(Into::into),
        #[cfg(feature = "metal")]
        BackendChoice::Metal => serve(
            MetalBackend::new()?,
            BackendChoice::Metal,
            arguments,
            signing_key,
            model_sha256,
            interrupted,
        ),
        #[cfg(not(feature = "metal"))]
        BackendChoice::Metal => validate_compiled(BackendChoice::Metal).map_err(Into::into),
        BackendChoice::Cpu => serve(
            CpuBackend::new(),
            BackendChoice::Cpu,
            arguments,
            signing_key,
            model_sha256,
            interrupted,
        ),
    }
}

fn observe_backend_capacity<B: Backend>(
    backend: &mut B,
    choice: BackendChoice,
) -> Result<BackendCapacityObservation, Box<dyn Error>> {
    match choice {
        BackendChoice::Cpu => Ok(BackendCapacityObservation::Cpu),
        BackendChoice::Cuda => match backend.memory_capacity()? {
            MemoryCapacity::Limited {
                available_bytes,
                total_bytes,
            } => Ok(BackendCapacityObservation::Discrete {
                total_bytes,
                available_bytes,
            }),
            MemoryCapacity::Unbounded => Ok(BackendCapacityObservation::Unavailable {
                reason: ObservationUnavailable::SourceUnavailable,
            }),
        },
        BackendChoice::Metal => match backend.memory_capacity()? {
            MemoryCapacity::Limited {
                available_bytes,
                total_bytes,
            } => Ok(BackendCapacityObservation::Recommended {
                working_set_bytes: total_bytes,
                current_allocated_bytes: total_bytes.saturating_sub(available_bytes),
                unified_memory: true,
            }),
            MemoryCapacity::Unbounded => Ok(BackendCapacityObservation::Unavailable {
                reason: ObservationUnavailable::SourceUnavailable,
            }),
        },
    }
}

fn policy_memory_topology(choice: BackendChoice) -> PolicyMemoryTopology {
    match choice {
        BackendChoice::Cuda => PolicyMemoryTopology::Discrete,
        BackendChoice::Cpu => PolicyMemoryTopology::Cpu,
        BackendChoice::Metal => PolicyMemoryTopology::Unified,
    }
}

fn prepare_service_memory<B: Backend>(
    backend: &mut B,
    choice: BackendChoice,
    arguments: ServiceBudgetArgs,
) -> Result<ServerMemory, Box<dyn Error>> {
    let topology = policy_memory_topology(choice);
    let inputs = ServiceMemoryInputs {
        topology,
        backend: observe_backend_capacity(backend, choice)?,
        host: observe_host_memory(),
        backend_owned_bytes: backend.memory_accounting().live_bytes,
        host_owned_bytes: 0,
    };
    let policy = resolve_policy(arguments, inputs)
        .map_err(|error| invalid_data(format!("service memory policy is invalid: {error}")))?;
    let host = install_service_memory_trackers(backend, policy)?;
    Ok(ServerMemory::with_topology(policy, topology, host))
}

fn install_service_memory_trackers<B: Backend>(
    backend: &mut B,
    policy: ServiceMemoryPolicy,
) -> Result<HostMemoryLedger, BackendError> {
    let backend_budget = MemoryBudget::Bytes(policy.pools().combined_bytes());
    let host = match policy.shared_budget() {
        Some(shared_budget) => {
            let root = MemoryTrackerRoot::new(shared_budget);
            backend.set_memory_tracker(MemoryTracker::child(backend_budget, root.clone()))?;
            HostMemoryLedger::child(policy.host_budget(), root)
        }
        None => {
            backend.set_memory_budget(backend_budget)?;
            HostMemoryLedger::new(policy.host_budget())
        }
    };
    Ok(host)
}

pub fn verify_response_receipt(path: &Path) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let receipt = ResponseReceipt::from_json(&bytes)?;
    println!(
        "response receipt {} claim and signature agree",
        receipt.claim.receipt_id
    );
    println!("embedded signer: {}", receipt.public_key_ed25519);
    println!("model: {}", receipt.claim.model_sha256);
    println!("transcript: {}", receipt.claim.transcript_sha256);
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn run_session_gate(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse_session_gate(arguments)?;
    let SessionGateSetup {
        mut runtime,
        prompt_tokens,
        policies,
        cuts,
        tokens,
        seed,
    } = prepare_session_gate(arguments)?;
    let (comparisons, mismatches) =
        run_session_policies(&mut runtime, &prompt_tokens, cuts, tokens, seed, policies)?;
    let cancellation_leaked = session_gate_cancellation(&mut runtime, &prompt_tokens, tokens)?;
    println!("comparisons={comparisons}");
    println!("mismatches={mismatches}");
    println!("cancellation_leaked={cancellation_leaked}");
    if mismatches != 0 || cancellation_leaked {
        return Err(invalid_data("session replay gate failed").into());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct SessionPolicy {
    name: &'static str,
    mirostat: Option<MirostatConfig>,
}

#[cfg(feature = "cuda")]
struct SessionGateSetup {
    runtime: Runtime<CudaBackend>,
    prompt_tokens: Vec<u32>,
    policies: [SessionPolicy; 2],
    cuts: usize,
    tokens: usize,
    seed: u64,
}

#[cfg(feature = "cuda")]
fn prepare_session_gate(arguments: SessionGateArgs) -> Result<SessionGateSetup, Box<dyn Error>> {
    let model = arguments
        .model
        .ok_or_else(|| invalid_data("verify session requires -m <gguf>"))?;
    let runtime = Runtime::load(CudaBackend::new(0)?, model)?;
    let prompt =
        "Explain why exact replay matters for a local inference server. Use short sentences.";
    let prompt_tokens = runtime.model().tokenizer().encode(prompt)?;
    let policies = session_gate_policies()?;
    Ok(SessionGateSetup {
        runtime,
        prompt_tokens,
        policies,
        cuts: arguments.cuts,
        tokens: arguments.tokens,
        seed: arguments.seed,
    })
}

#[cfg(feature = "cuda")]
struct SessionGateArgs {
    model: Option<PathBuf>,
    cuts: usize,
    tokens: usize,
    seed: u64,
}

#[cfg(feature = "cuda")]
fn parse_session_gate(arguments: &[String]) -> Result<SessionGateArgs, Box<dyn Error>> {
    let mut parsed = SessionGateArgs {
        model: None,
        cuts: 32,
        tokens: 64,
        seed: 0x4c65_6f6e_652d_7633_u64,
    };
    let mut index = 0;
    while index < arguments.len() {
        if parse_session_model(&mut parsed, arguments, &mut index)?
            || parse_session_counts(&mut parsed, arguments, &mut index)?
            || parse_session_seed(&mut parsed, arguments, &mut index)?
        {
            index += 1;
            continue;
        }
        return Err(invalid_data(format!(
            "verify session argument is invalid: {}",
            arguments[index]
        ))
        .into());
    }
    if parsed.tokens < 2 {
        return Err(invalid_data("verify session needs at least two tokens").into());
    }
    Ok(parsed)
}

#[cfg(feature = "cuda")]
fn parse_session_model(
    parsed: &mut SessionGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if !matches!(arguments[*index].as_str(), "-m" | "--model") {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

#[cfg(feature = "cuda")]
fn parse_session_counts(
    parsed: &mut SessionGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] == "--cuts" {
        parsed.cuts = nonzero_usize(flag_value(arguments, index)?, "cuts")?;
        return Ok(true);
    }
    if arguments[*index] == "--tokens" {
        parsed.tokens = nonzero_usize(flag_value(arguments, index)?, "tokens")?;
        return Ok(true);
    }
    Ok(false)
}

#[cfg(feature = "cuda")]
fn parse_session_seed(
    parsed: &mut SessionGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--seed" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.seed = value
        .parse()
        .map_err(|_| invalid_data(format!("seed is invalid: {value}")))?;
    Ok(true)
}

#[cfg(feature = "cuda")]
fn session_gate_policies() -> Result<[SessionPolicy; 2], Box<dyn Error>> {
    Ok([
        SessionPolicy {
            name: "seeded",
            mirostat: None,
        },
        SessionPolicy {
            name: "mirostat-v2",
            mirostat: Some(MirostatConfig::new(5.0, 0.1).map_err(|error| {
                invalid_data(format!("Mirostat gate configuration failed: {error}"))
            })?),
        },
    ])
}

#[cfg(feature = "cuda")]
struct SessionReplaySummary {
    class: String,
    cached: usize,
    reused: usize,
    replayed: usize,
    computed: usize,
}

#[cfg(feature = "cuda")]
struct SessionCutEvidence {
    live_tokens: Vec<u32>,
    restored_tokens: Vec<u32>,
    live_replay: SessionReplaySummary,
    restored_replay: SessionReplaySummary,
}

#[cfg(feature = "cuda")]
fn run_session_policies(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    cuts: usize,
    tokens: usize,
    seed: u64,
    policies: [SessionPolicy; 2],
) -> Result<(usize, usize), Box<dyn Error>> {
    let mut comparisons = 0;
    let mut mismatches = 0;
    for SessionPolicy { name, mirostat } in policies {
        let (policy_comparisons, policy_mismatches) =
            run_session_policy(runtime, prompt_tokens, cuts, tokens, seed, name, mirostat)?;
        comparisons += policy_comparisons;
        mismatches += policy_mismatches;
    }
    Ok((comparisons, mismatches))
}

#[cfg(feature = "cuda")]
fn run_session_policy(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    cuts: usize,
    tokens: usize,
    seed: u64,
    policy: &str,
    mirostat: Option<MirostatConfig>,
) -> Result<(usize, usize), Box<dyn Error>> {
    let mut baseline_options = GenerateOptions::greedy(tokens);
    baseline_options.decode_execution = DecodeExecution::Eager;
    baseline_options.sampler = Sampler::temperature(1.0);
    baseline_options.seed = seed;
    baseline_options.mirostat = mirostat;
    let baseline =
        runtime.generate_tokens(prompt_tokens, baseline_options, |_| Ok(()), || false)?;
    if baseline.tokens.len() < 2 {
        return Err(invalid_data("session gate baseline stopped before two tokens").into());
    }
    let mut comparisons = 0;
    let mut live_mismatches = 0;
    let mut restored_mismatches = 0;
    let mut draw = seed;
    for _ in 0..cuts {
        draw = draw
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .rotate_left(17)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let cut = 1 + draw as usize % (baseline.tokens.len() - 1);
        let (live_mismatch, restored_mismatch) = session_gate_cut(
            runtime,
            prompt_tokens,
            &baseline.tokens,
            cut,
            seed,
            mirostat,
            policy,
        )?;
        comparisons += 2;
        live_mismatches += usize::from(live_mismatch);
        restored_mismatches += usize::from(restored_mismatch);
    }
    println!(
        "policy={policy} cuts={cuts} live_mismatches={} restored_mismatches={} baseline_tokens={} transcript={}",
        live_mismatches,
        restored_mismatches,
        baseline.tokens.len(),
        token_stream_sha256(&baseline.tokens)
    );
    Ok((comparisons, live_mismatches + restored_mismatches))
}

#[cfg(feature = "cuda")]
fn session_gate_cut(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    baseline: &[u32],
    cut: usize,
    seed: u64,
    mirostat: Option<MirostatConfig>,
    policy: &str,
) -> Result<(bool, bool), Box<dyn Error>> {
    let evidence =
        session_gate_cut_evidence(runtime, prompt_tokens, baseline.len(), cut, seed, mirostat)?;
    let live_mismatch = evidence.live_tokens != baseline;
    let restored_mismatch = evidence.restored_tokens != baseline;
    print_session_mismatch(live_mismatch, "live", policy, cut, &evidence.live_replay);
    print_session_mismatch(
        restored_mismatch,
        "restored",
        policy,
        cut,
        &evidence.restored_replay,
    );
    Ok((live_mismatch, restored_mismatch))
}

#[cfg(feature = "cuda")]
fn session_gate_cut_evidence(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    limit: usize,
    cut: usize,
    seed: u64,
    mirostat: Option<MirostatConfig>,
) -> Result<SessionCutEvidence, Box<dyn Error>> {
    let mut prefix_options = GenerateOptions::greedy(cut);
    prefix_options.decode_execution = DecodeExecution::Eager;
    prefix_options.sampler = Sampler::temperature(1.0);
    prefix_options.seed = seed;
    prefix_options.mirostat = mirostat;
    let mut live = GenerationSession::new();
    let prefix = runtime.generate_session_tokens(
        &mut live,
        prompt_tokens,
        prefix_options,
        |_| Ok(()),
        || false,
    )?;
    let checkpoint = live.checkpoint();
    let mut continuation_prompt = prompt_tokens.to_vec();
    continuation_prompt.extend_from_slice(&prefix.tokens);
    let mut tail_options = GenerateOptions::greedy(limit - cut);
    tail_options.decode_execution = DecodeExecution::Eager;
    tail_options.sampler = Sampler::temperature(1.0);
    tail_options.seed = seed;
    tail_options.mirostat = mirostat;
    let live_tail = runtime.generate_session_tokens(
        &mut live,
        &continuation_prompt,
        tail_options.clone(),
        |_| Ok(()),
        || false,
    )?;
    let mut restored = GenerationSession::new();
    restored.restore(checkpoint);
    let restored_tail = runtime.generate_session_tokens(
        &mut restored,
        &continuation_prompt,
        tail_options,
        |_| Ok(()),
        || false,
    )?;
    let mut live_tokens = prefix.tokens.clone();
    live_tokens.extend_from_slice(&live_tail.tokens);
    let mut restored_tokens = prefix.tokens;
    restored_tokens.extend_from_slice(&restored_tail.tokens);
    let live_replay = live.last_replay();
    let restored_replay = restored.last_replay();
    Ok(SessionCutEvidence {
        live_tokens,
        restored_tokens,
        live_replay: SessionReplaySummary {
            class: reuse_class_name(live_replay.reuse_class).to_owned(),
            cached: live_replay.cached_tokens,
            reused: live_replay.reused_tokens,
            replayed: live_replay.replayed_tokens,
            computed: live_replay.computed_tokens,
        },
        restored_replay: SessionReplaySummary {
            class: reuse_class_name(restored_replay.reuse_class).to_owned(),
            cached: restored_replay.cached_tokens,
            reused: restored_replay.reused_tokens,
            replayed: restored_replay.replayed_tokens,
            computed: restored_replay.computed_tokens,
        },
    })
}

#[cfg(feature = "cuda")]
fn print_session_mismatch(
    mismatch: bool,
    kind: &str,
    policy: &str,
    cut: usize,
    replay: &SessionReplaySummary,
) {
    if mismatch {
        println!(
            "mismatch={kind} policy={policy} cut={cut} class={} cached={} reused={} replayed={} computed={}",
            replay.class, replay.cached, replay.reused, replay.replayed, replay.computed,
        );
    }
}

#[cfg(feature = "cuda")]
fn session_gate_cancellation(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    tokens: usize,
) -> Result<bool, Box<dyn Error>> {
    let mut cancelled_session = GenerationSession::new();
    let emitted = std::cell::Cell::new(0_usize);
    let mut cancellation_options = GenerateOptions::greedy(tokens);
    cancellation_options.decode_execution = DecodeExecution::Eager;
    let cancelled = runtime.generate_session_tokens(
        &mut cancelled_session,
        prompt_tokens,
        cancellation_options,
        |_| {
            emitted.set(emitted.get() + 1);
            Ok(())
        },
        || emitted.get() >= 3,
    )?;
    Ok(!cancelled.stats.cancelled || !cancelled_session.is_empty())
}

#[cfg(feature = "cuda")]
pub fn run_scheduler_gate(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse_scheduler_gate(arguments)?;
    let SchedulerGateSetup {
        runtime,
        prompt_tokens,
        options,
        isolated,
        policy,
        tokens,
    } = prepare_scheduler_gate(arguments)?;
    let executor = RuntimeQuantumExecutor::new(LeoneRuntimeDriver::new(runtime));
    let mut service = ScheduledService::new(policy, executor)?;
    admit_scheduler_requests(&mut service, &prompt_tokens, &options, tokens)?;
    finish_scheduler_requests(&mut service)?;
    let mismatches = compare_scheduler_requests(&service, &isolated)?;
    println!("comparisons={}", isolated.len());
    println!("mismatches={mismatches}");
    if mismatches != 0 {
        return Err(invalid_data("scheduler model gate failed").into());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct SchedulerGateSetup {
    runtime: Runtime<CudaBackend>,
    prompt_tokens: Vec<Vec<u32>>,
    options: Vec<GenerateOptions>,
    isolated: Vec<Vec<u32>>,
    policy: SchedulerPolicy,
    tokens: usize,
}

#[cfg(feature = "cuda")]
fn prepare_scheduler_gate(
    arguments: SchedulerGateArgs,
) -> Result<SchedulerGateSetup, Box<dyn Error>> {
    let model = arguments
        .model
        .ok_or_else(|| invalid_data("verify scheduler requires -m <gguf>"))?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, model)?;
    let SchedulerGateWorkload {
        prompt_tokens,
        options,
    } = scheduler_gate_workload(&mut runtime, arguments.tokens, arguments.seed)?;
    let context = runtime.model().config().context_length;
    validate_scheduler_context(&prompt_tokens, arguments.tokens, context)?;
    let isolated = scheduler_isolated(&mut runtime, &prompt_tokens, &options)?;
    let policy = scheduler_gate_policy(
        &runtime,
        prompt_tokens.len(),
        arguments.tokens,
        context,
        arguments.quantum,
    )?;
    Ok(SchedulerGateSetup {
        runtime,
        prompt_tokens,
        options,
        isolated,
        policy,
        tokens: arguments.tokens,
    })
}

#[cfg(feature = "cuda")]
struct SchedulerGateArgs {
    model: Option<PathBuf>,
    tokens: usize,
    quantum: u32,
    seed: u64,
}

#[cfg(feature = "cuda")]
fn parse_scheduler_gate(arguments: &[String]) -> Result<SchedulerGateArgs, Box<dyn Error>> {
    let mut parsed = SchedulerGateArgs {
        model: None,
        tokens: 32,
        quantum: 4,
        seed: 0x4c65_6f6e_652d_7631_u64,
    };
    let mut index = 0;
    while index < arguments.len() {
        if parse_scheduler_model(&mut parsed, arguments, &mut index)?
            || parse_scheduler_counts(&mut parsed, arguments, &mut index)?
            || parse_scheduler_seed(&mut parsed, arguments, &mut index)?
        {
            index += 1;
            continue;
        }
        return Err(invalid_data(format!(
            "verify scheduler argument is invalid: {}",
            arguments[index]
        ))
        .into());
    }
    Ok(parsed)
}

#[cfg(feature = "cuda")]
fn parse_scheduler_model(
    parsed: &mut SchedulerGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if !matches!(arguments[*index].as_str(), "-m" | "--model") {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

#[cfg(feature = "cuda")]
fn parse_scheduler_counts(
    parsed: &mut SchedulerGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] == "--tokens" {
        parsed.tokens = nonzero_usize(flag_value(arguments, index)?, "tokens")?;
        return Ok(true);
    }
    if arguments[*index] == "--quantum" {
        let value = nonzero_usize(flag_value(arguments, index)?, "quantum")?;
        parsed.quantum =
            u32::try_from(value).map_err(|_| invalid_data("quantum does not fit u32"))?;
        return Ok(true);
    }
    Ok(false)
}

#[cfg(feature = "cuda")]
fn parse_scheduler_seed(
    parsed: &mut SchedulerGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--seed" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.seed = value
        .parse()
        .map_err(|_| invalid_data(format!("seed is invalid: {value}")))?;
    Ok(true)
}

#[cfg(feature = "cuda")]
struct SchedulerGateWorkload {
    prompt_tokens: Vec<Vec<u32>>,
    options: Vec<GenerateOptions>,
}

#[cfg(feature = "cuda")]
fn scheduler_gate_workload(
    runtime: &mut Runtime<CudaBackend>,
    tokens: usize,
    seed: u64,
) -> Result<SchedulerGateWorkload, Box<dyn Error>> {
    let prompts = [
        "Explain one invariant of exact local inference in two sentences.",
        "List three reasons to bound a server work quantum.",
        "Write a short definition of reproducible performance evidence.",
    ];
    let prompt_tokens = prompts
        .iter()
        .map(|prompt| runtime.model().tokenizer().encode(prompt))
        .collect::<Result<Vec<_>, _>>()?;
    let mut options = vec![GenerateOptions::greedy(tokens); prompts.len()];
    for option in &mut options {
        option.decode_execution = DecodeExecution::Eager;
    }
    options[1].sampler = Sampler::temperature(0.8);
    options[1].seed = seed;
    options[2].sampler = Sampler::temperature(1.0);
    options[2].seed = seed ^ 0x9e37_79b9_7f4a_7c15;
    options[2].mirostat = Some(
        MirostatConfig::new(5.0, 0.1)
            .map_err(|error| invalid_data(format!("Mirostat configuration failed: {error}")))?,
    );
    Ok(SchedulerGateWorkload {
        prompt_tokens,
        options,
    })
}

#[cfg(feature = "cuda")]
fn validate_scheduler_context(
    prompts: &[Vec<u32>],
    tokens: usize,
    context: usize,
) -> Result<(), Box<dyn Error>> {
    for prompt in prompts {
        if prompt.len().saturating_add(tokens) > context {
            return Err(invalid_data("scheduler gate workload exceeds model context").into());
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn scheduler_isolated(
    runtime: &mut Runtime<CudaBackend>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let mut isolated = Vec::with_capacity(prompts.len());
    for (prompt, options) in prompts.iter().zip(options) {
        let mut session = GenerationSession::new();
        isolated.push(
            runtime
                .generate_session_tokens(
                    &mut session,
                    prompt,
                    options.clone(),
                    |_| Ok(()),
                    || false,
                )?
                .tokens,
        );
    }
    Ok(isolated)
}

#[cfg(feature = "cuda")]
fn scheduler_gate_policy(
    runtime: &Runtime<CudaBackend>,
    request_count: usize,
    tokens: usize,
    context: usize,
    quantum: u32,
) -> Result<SchedulerPolicy, Box<dyn Error>> {
    let config = runtime.model().config();
    let (kv_bytes_per_token, request_count) = scheduler_gate_dimensions(config, request_count)?;
    let context_tokens = host_u64(context)?;
    let output_tokens = host_u64(tokens)?;
    let max_reserved_kv_bytes =
        scheduler_gate_capacity(kv_bytes_per_token, context_tokens, request_count)?;
    Ok(SchedulerPolicy {
        max_active_requests: request_count,
        max_queued_requests: request_count,
        max_batch_requests: request_count,
        max_reserved_kv_bytes,
        kv_bytes_per_token,
        kv_page_tokens: 16,
        max_prompt_tokens: context_tokens,
        max_output_tokens: output_tokens,
        service_quantum_tokens: quantum,
        prefill_chunk_tokens: u32::try_from(leone::DEFAULT_PREFILL_CHUNK_TOKENS)
            .map_err(|_| invalid_data("prefill chunk does not fit the scheduler"))?,
        urgent_window_ns: 0,
        max_prefix_credit_tokens: 0,
    })
}

#[cfg(feature = "cuda")]
fn scheduler_gate_dimensions(
    config: &leone::ModelConfig,
    request_count: usize,
) -> Result<(u64, u32), Box<dyn Error>> {
    let layers = host_u64(config.n_layer)?;
    let kv_heads = host_u64(config.n_head_kv)?;
    let head_dim = host_u64(config.head_dim)?;
    let request_count =
        u32::try_from(request_count).map_err(|_| invalid_data("request count does not fit u32"))?;
    let kv_bytes_per_token = layers
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| invalid_data("scheduler gate KV bound overflowed"))?;
    Ok((kv_bytes_per_token, request_count))
}

#[cfg(feature = "cuda")]
fn scheduler_gate_capacity(
    kv_bytes_per_token: u64,
    context_tokens: u64,
    request_count: u32,
) -> Result<u64, Box<dyn Error>> {
    kv_bytes_per_token
        .checked_mul(context_tokens)
        .and_then(|value| value.checked_mul(u64::from(request_count)))
        .ok_or_else(|| invalid_data("scheduler gate KV capacity overflowed").into())
}

#[cfg(feature = "cuda")]
fn admit_scheduler_requests(
    service: &mut ScheduledService<RuntimeQuantumExecutor<LeoneRuntimeDriver<CudaBackend>>>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
    tokens: usize,
) -> Result<(), Box<dyn Error>> {
    for (offset, (prompt, options)) in prompts.iter().zip(options).enumerate() {
        let request_id = RequestId(u64::try_from(offset)? + 1);
        let request = ScheduledGenerationRequest::new(prompt.clone(), options.clone());
        let spec = RequestSpec {
            id: request_id,
            arrival_ns: 0,
            prompt_tokens: host_u64(prompt.len())?,
            prefix_reused_tokens: 0,
            max_output_tokens: host_u64(tokens)?,
            priority: 1,
            deadline_ns: None,
        };
        if !matches!(
            service.admit(spec, &request, 0)?,
            AdmissionOutcome::Admitted { .. }
        ) {
            return Err(invalid_data("scheduler gate admission failed").into());
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn finish_scheduler_requests(
    service: &mut ScheduledService<RuntimeQuantumExecutor<LeoneRuntimeDriver<CudaBackend>>>,
) -> Result<(), Box<dyn Error>> {
    let mut now_ns = 1_u64;
    while service.has_runnable_requests() {
        service.tick_batch(now_ns)?;
        now_ns = now_ns.saturating_add(1);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn compare_scheduler_requests(
    service: &ScheduledService<RuntimeQuantumExecutor<LeoneRuntimeDriver<CudaBackend>>>,
    isolated: &[Vec<u32>],
) -> Result<usize, Box<dyn Error>> {
    let mut mismatches = 0;
    for (offset, expected) in isolated.iter().enumerate() {
        let request_id = RequestId(u64::try_from(offset)? + 1);
        let concurrent = &service
            .executor()
            .completed(request_id)
            .ok_or_else(|| invalid_data("scheduler gate lost a completed request"))?
            .tokens;
        let matches = concurrent == expected;
        mismatches += usize::from(!matches);
        println!(
            "request={} matches={} tokens={} transcript={}",
            request_id.0,
            matches,
            concurrent.len(),
            token_stream_sha256(concurrent)
        );
    }
    Ok(mismatches)
}

fn serve<B: Backend>(
    mut backend: B,
    choice: BackendChoice,
    arguments: ServeArgs,
    signing_key: SigningKey,
    model_sha256: String,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let memory = prepare_service_memory(
        &mut backend,
        choice,
        ServiceBudgetArgs {
            memory: arguments.memory_budget,
            host: arguments.host_memory_budget,
            kv_reservation: arguments.kv_reservation_budget,
        },
    )?;
    let setup = build_serve_setup(backend, arguments, signing_key, model_sha256, memory)?;
    let policy = server_scheduler_policy(&setup.server, setup.max_sessions, setup.batch_size)?;
    let transport_settings = transport_config(&setup);
    let server = initialize_server_metrics(setup.server, &policy)?;
    let trace = service_trace(&server);
    let bind = setup.bind;
    let mut service = ScheduledService::new(
        policy,
        ServerExecutor {
            server,
            tasks: BTreeMap::new(),
            wire_request_ids: BTreeMap::new(),
            queued_outputs: BTreeMap::new(),
            terminal_outputs: VecDeque::new(),
            trace,
        },
    )?;
    let transport = Transport::bind(
        bind,
        transport_settings,
        Arc::clone(&interrupted),
        read_request,
        request_client_identity,
    )?;
    run_serve_loop(&mut service, transport, interrupted)
}

fn initialize_server_metrics<B: Backend>(
    mut server: Server<B>,
    policy: &SchedulerPolicy,
) -> Result<Server<B>, Box<dyn Error>> {
    let workload_epoch = format!(
        "{}:{}",
        server.identity.source_id, server.identity.process_instance_id
    );
    server.metrics = ServerMetrics::with_scheduler_capacity(
        usize::try_from(policy.max_active_requests)
            .map_err(|_| invalid_data("active request limit does not fit usize"))?,
        usize::try_from(policy.max_queued_requests)
            .map_err(|_| invalid_data("queued request limit does not fit usize"))?,
        workload_epoch,
    )?;
    server.metrics.bind_identity(&server.identity);
    sample_server_metrics(&mut server, 0);
    Ok(server)
}

fn service_trace<B: Backend>(server: &Server<B>) -> ServiceTraceRecorder {
    let trace_topology = metrics_memory_topology(server.memory.topology);
    let trace_host_tracker = Some(server.memory.host.tracker());
    let workload_epoch = format!(
        "{}:{}",
        server.identity.source_id, server.identity.process_instance_id
    );
    ServiceTraceRecorder::with_identity(
        &server.runtime,
        trace_topology,
        trace_host_tracker,
        &server.identity.source_id,
        &server.identity.process_instance_id,
        &workload_epoch,
    )
}

struct ServeSetup<B: Backend> {
    server: Server<B>,
    bind: SocketAddr,
    max_sessions: usize,
    batch_size: usize,
    max_connections: usize,
    max_connections_per_client: usize,
    max_pending_requests: usize,
    max_output_bytes: usize,
    request_timeout: Duration,
    trusted_proxy_ips: Vec<IpAddr>,
}

struct SessionPersistence {
    store: Option<SessionStore>,
    archives: Vec<LoadedSessionArchive>,
}

struct PersistedCacheTables {
    entries: BoundedTable<String, PersistedArchiveEntry>,
    allocation: Option<MemoryAllocation>,
}

struct ServeBuildInputs<B: Backend> {
    runtime: Runtime<B>,
    arguments: ServeArgs,
    signing_key: SigningKey,
    model_sha256: String,
    memory: ServerMemory,
    #[cfg(feature = "cuda")]
    execution_plan: Option<ExecutionPlanSelection>,
    prefill_chunk_tokens: usize,
    context_limit: usize,
    persistence: SessionPersistence,
    batch_size: usize,
    kv_cache_dtype: KvCacheDtype,
}

fn build_serve_setup<B: Backend>(
    backend: B,
    arguments: ServeArgs,
    signing_key: SigningKey,
    model_sha256: String,
    mut memory: ServerMemory,
) -> Result<ServeSetup<B>, Box<dyn Error>> {
    let batch_size = resolve_backend_batch_size(arguments.batch_size, backend.max_batch_size())?;
    let execution_plan = load_serve_plan(&arguments, backend.name())?;
    let kv_cache_dtype = effective_serve_kv(
        arguments.kv_cache_dtype,
        arguments.kv_cache_dtype_explicit,
        execution_plan,
    )?;
    let prefill_chunk_tokens =
        configured_prefill_chunk(arguments.prefill_chunk_tokens, execution_plan);
    let staging = leone::HostStaging::from_tracker(memory.host.tracker());
    let runtime = Runtime::load_with_host_staging(backend, &arguments.model, &staging)?;
    let context_limit = serve_context_limit(
        arguments.context_limit,
        runtime.model().config().context_length,
    )?;
    memory.reserve_metrics_metadata(
        arguments.sessions,
        MAX_QUEUED_REQUESTS,
        context_limit,
        arguments.max_output_bytes,
    )?;
    let persistence = load_session_persistence(&arguments, context_limit, memory.host.clone())?;
    finish_serve_setup(ServeBuildInputs {
        runtime,
        arguments,
        signing_key,
        model_sha256,
        memory,
        #[cfg(feature = "cuda")]
        execution_plan,
        prefill_chunk_tokens,
        context_limit,
        persistence,
        batch_size,
        kv_cache_dtype,
    })
}

fn finish_serve_setup<B: Backend>(
    inputs: ServeBuildInputs<B>,
) -> Result<ServeSetup<B>, Box<dyn Error>> {
    let ServeBuildInputs {
        runtime,
        arguments,
        signing_key,
        model_sha256,
        memory,
        #[cfg(feature = "cuda")]
        execution_plan,
        prefill_chunk_tokens,
        context_limit,
        persistence,
        batch_size,
        kv_cache_dtype,
    } = inputs;
    let SessionPersistence {
        store: session_store,
        archives,
    } = persistence;
    let model_id = arguments
        .model
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("leone")
        .to_owned();
    let clock = archives
        .iter()
        .map(|archive| archive.archive.last_used)
        .max()
        .unwrap_or(0);
    let max_sessions = arguments.sessions;
    let bind = arguments.bind;
    let persisted_cache_limit = arguments
        .sessions
        .checked_add(arguments.hibernated_sessions)
        .ok_or_else(|| invalid_data("session cache limit overflowed"))?;
    let persisted_tables = allocate_persisted_cache_tables(&memory, persisted_cache_limit)?;
    let server = complete_server_setup(
        Server {
            runtime,
            model_id,
            model_sha256: model_sha256.clone(),
            sessions: HashMap::new(),
            hibernated: HashMap::new(),
            persisted: persisted_tables.entries,
            _persisted_table_allocation: persisted_tables.allocation,
            persisted_cache_limit,
            session_store,
            max_sessions,
            max_hibernated_sessions: arguments.hibernated_sessions,
            clock,
            kv_cache_dtype,
            #[cfg(feature = "cuda")]
            execution_plan,
            prefill_chunk_tokens,
            context_limit,
            receipts: arguments.receipts,
            signing_key,
            memory,
            cors_origins: arguments.cors_origins,
            proxy_origin: arguments.proxy_origin,
            metrics: ServerMetrics::new()?,
            identity: running_service_identity(&model_sha256)?,
            #[cfg(test)]
            test_finalize_delay: None,
        },
        archives,
    )?;
    Ok(ServeSetup {
        server,
        bind,
        max_sessions,
        batch_size,
        max_connections: arguments.max_connections,
        max_connections_per_client: arguments.max_connections_per_client,
        max_pending_requests: arguments.max_pending_requests,
        max_output_bytes: arguments.max_output_bytes,
        request_timeout: Duration::from_millis(arguments.request_timeout_ms),
        trusted_proxy_ips: arguments.trusted_proxy_ips,
    })
}

fn complete_server_setup<B: Backend>(
    mut server: Server<B>,
    archives: Vec<LoadedSessionArchive>,
) -> Result<Server<B>, Box<dyn Error>> {
    install_persisted_archives(&mut server, archives)?;
    if let Some(store) = server.session_store.as_mut() {
        store.startup_archive_transition.take();
    }
    server.memory.logical_kv_reservation = server.memory.resolve_logical_kv(
        &server.runtime,
        server.kv_cache_dtype,
        server.context_limit,
        server.max_sessions,
    )?;
    Ok(server)
}

fn resolve_backend_batch_size(
    requested: Option<usize>,
    supported: NonZeroUsize,
) -> Result<usize, io::Error> {
    match requested {
        Some(requested) => {
            validate_backend_batch_size(requested, supported)?;
            Ok(requested)
        }
        None => Ok(DEFAULT_SERVE_BATCH_SIZE.min(supported.get())),
    }
}

#[cfg(feature = "cuda")]
fn effective_serve_kv(
    requested: KvCacheDtype,
    explicit: bool,
    plan: Option<ExecutionPlanSelection>,
) -> Result<KvCacheDtype, io::Error> {
    let Some(plan) = plan else {
        return Ok(requested);
    };
    let planned = plan.kv_dtype();
    if explicit && requested != planned {
        return Err(invalid_data(format!(
            "--kv {:?} conflicts with execution plan KV type {:?}",
            requested, planned
        )));
    }
    Ok(planned)
}

#[cfg(not(feature = "cuda"))]
fn effective_serve_kv(
    requested: KvCacheDtype,
    _explicit: bool,
    _plan: Option<ExecutionPlanSelection>,
) -> Result<KvCacheDtype, io::Error> {
    Ok(requested)
}

fn validate_backend_batch_size(requested: usize, supported: NonZeroUsize) -> Result<(), io::Error> {
    if requested > supported.get() {
        return Err(invalid_data(format!(
            "batch size {requested} exceeds backend maximum {}",
            supported.get()
        )));
    }
    Ok(())
}

fn install_persisted_archives<B: Backend>(
    server: &mut Server<B>,
    archives: Vec<LoadedSessionArchive>,
) -> Result<(), Box<dyn Error>> {
    if archives.len() > server.persisted_cache_limit {
        return Err(invalid_data("persisted archive count exceeds its cache limit").into());
    }
    for loaded in archives {
        server.cache_persisted_archive_with_allocation(
            loaded.session_id,
            loaded.archive,
            loaded.allocation,
        )?;
    }
    Ok(())
}

fn allocate_persisted_cache_tables(
    memory: &ServerMemory,
    limit: usize,
) -> Result<PersistedCacheTables, Box<dyn Error>> {
    let allocation = if limit == 0 {
        None
    } else {
        Some(memory.host.allocate(persisted_cache_table_bound(limit)?)?)
    };
    let entries = BoundedTable::with_capacity(limit).map_err(|error| {
        invalid_data(format!(
            "persisted archive table allocation failed: {error}"
        ))
    })?;
    Ok(PersistedCacheTables {
        entries,
        allocation,
    })
}

#[cfg(feature = "cuda")]
fn configured_prefill_chunk(
    explicit: Option<usize>,
    plan: Option<ExecutionPlanSelection>,
) -> usize {
    explicit
        .or_else(|| plan.map(|selection| selection.prefill_chunk_tokens))
        .unwrap_or(DEFAULT_PREFILL_CHUNK_TOKENS)
}

#[cfg(not(feature = "cuda"))]
fn configured_prefill_chunk(
    explicit: Option<usize>,
    _plan: Option<ExecutionPlanSelection>,
) -> usize {
    explicit.unwrap_or(DEFAULT_PREFILL_CHUNK_TOKENS)
}

#[cfg(feature = "cuda")]
fn load_serve_plan(
    arguments: &ServeArgs,
    backend_name: &str,
) -> Result<Option<ExecutionPlanSelection>, Box<dyn Error>> {
    crate::execution_plan::load_optional(arguments.plan.as_deref(), &arguments.model, backend_name)
}

#[cfg(not(feature = "cuda"))]
fn load_serve_plan(
    arguments: &ServeArgs,
    _backend_name: &str,
) -> Result<Option<ExecutionPlanSelection>, Box<dyn Error>> {
    if arguments.plan.is_some() {
        return Err(invalid_data("--plan requires a CUDA-enabled build").into());
    }
    Ok(None)
}

fn load_session_persistence(
    arguments: &ServeArgs,
    context_limit: usize,
    host: HostMemoryLedger,
) -> Result<SessionPersistence, Box<dyn Error>> {
    match arguments.session_store.clone() {
        Some(path) => {
            let cache_limit = arguments
                .sessions
                .checked_add(arguments.hibernated_sessions)
                .ok_or_else(|| invalid_data("session cache limit overflowed"))?;
            let archive_limit = session_archive_json_bound(context_limit)?;
            let (store, archives, _) = SessionStore::open_with_archive_limit_and_host(
                path,
                cache_limit,
                archive_limit,
                host,
            )?;
            Ok(SessionPersistence {
                store: Some(store),
                archives,
            })
        }
        None => Ok(SessionPersistence {
            store: None,
            archives: Vec::new(),
        }),
    }
}

fn transport_config<B: Backend>(setup: &ServeSetup<B>) -> crate::transport::Config {
    crate::transport::Config {
        max_connections: setup.max_connections,
        max_connections_per_client: setup.max_connections_per_client,
        max_pending_requests: setup.max_pending_requests,
        max_output_bytes: setup.max_output_bytes,
        read_timeout: setup.request_timeout,
        write_timeout: setup.request_timeout,
        trusted_proxy_ips: setup.trusted_proxy_ips.clone(),
    }
}

fn session_archive_json_bound(token_count: usize) -> Result<u64, Box<dyn Error>> {
    leone::SessionArchive::max_json_len(token_count)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid_data("session archive JSON size overflowed").into())
}

/// Bounds the host bytes one persist holds: the checkpoint clone, the serialization buffer,
/// and scratch for the digest and file names.
fn session_archive_persist_bound(token_count: usize) -> Result<u64, Box<dyn Error>> {
    let checkpoint_bytes = u64::try_from(token_count)
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| invalid_data("session archive checkpoint size overflowed"))?;
    checkpoint_bytes
        .checked_add(session_archive_json_bound(token_count)?)
        .and_then(|bytes| bytes.checked_add(SESSION_ARCHIVE_FIXED_BYTES))
        .ok_or_else(|| invalid_data("session archive storage size overflowed").into())
}

fn session_archive_restore_bound(serialized_bytes: u64) -> Result<u64, Box<dyn Error>> {
    let decoded_tokens = serialized_bytes
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| invalid_data("session archive token storage size overflowed"))?;
    let replay_copies = decoded_tokens
        .checked_mul(2)
        .ok_or_else(|| invalid_data("session archive replay size overflowed"))?;
    SESSION_ARCHIVE_FIXED_BYTES
        .checked_add(replay_copies)
        .ok_or_else(|| invalid_data("session archive restore size overflowed").into())
}

fn stored_archive_metadata_bound(
    session_id: &str,
    archive: &StoredArchive,
) -> Result<u64, Box<dyn Error>> {
    let digest = u64::try_from(archive.blob_sha256.len())
        .map_err(|_| invalid_data("session digest length does not fit a byte bound"))?;
    stored_archive_metadata_bound_parts(session_id, digest)
}

fn stored_archive_metadata_bound_for_id(session_id: &str) -> Result<u64, Box<dyn Error>> {
    stored_archive_metadata_bound_parts(session_id, 64)
}

fn stored_archive_metadata_bound_parts(
    session_id: &str,
    digest_bytes: u64,
) -> Result<u64, Box<dyn Error>> {
    let session_id = u64::try_from(session_id.len())
        .map_err(|_| invalid_data("session ID length does not fit a byte bound"))?;
    SESSION_ARCHIVE_FIXED_BYTES
        .checked_add(STORED_ARCHIVE_METADATA_FIXED_BYTES)
        .and_then(|bytes| bytes.checked_add(session_id))
        .and_then(|bytes| bytes.checked_add(digest_bytes))
        .ok_or_else(|| invalid_data("session archive metadata size overflowed").into())
}

fn startup_archive_transition_bound(count: usize) -> Result<u64, Box<dyn Error>> {
    let count = u64::try_from(count)
        .map_err(|_| invalid_data("session archive count does not fit a byte bound"))?;
    let vector_entry = u64::try_from(std::mem::size_of::<LoadedSessionArchive>())
        .map_err(|_| invalid_data("session archive vector entry does not fit a byte bound"))?;
    Ok(checked_bound_mul(
        count,
        vector_entry,
        "session archive vector size overflowed",
    )?)
}

fn run_serve_loop<B: Backend>(
    service: &mut ChatService<B>,
    transport: Transport<HttpRequest>,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let local_addr = transport.local_addr();
    println!("Leone serves http://{local_addr}");
    println!("OpenAI base URL: http://{local_addr}/v1");
    println!("execution: scheduled");
    while !interrupted.load(Ordering::Relaxed) {
        let accepted = accept_serve_connection(&transport, service)?;
        let ticked = tick_serve_requests(service)?;
        service.executor_mut().drain_transport_observations();
        service.prune_terminal();
        let progressed = accepted || ticked;
        if !progressed {
            thread::sleep(Duration::from_millis(2));
        }
    }
    Ok(())
}

fn accept_serve_connection<B: Backend>(
    transport: &Transport<HttpRequest>,
    service: &mut ChatService<B>,
) -> Result<bool, Box<dyn Error>> {
    let Some(incoming) = transport
        .try_recv()
        .map_err(|_| invalid_data("transport stopped"))?
    else {
        return Ok(false);
    };
    if let Err(error) = admit_incoming_request(service, incoming) {
        eprintln!("request admission failed: {error}");
    }
    Ok(true)
}

fn admit_incoming_request<B: Backend>(
    service: &mut ChatService<B>,
    incoming: Incoming<HttpRequest>,
) -> Result<(), Box<dyn Error>> {
    let Incoming {
        client_id,
        peer_addr,
        request_id,
        received_at_ns,
        request,
        output,
        cancellation,
        deadline,
        ..
    } = incoming;
    if request.method == "OPTIONS" {
        return write_options_response(service, output, &request);
    }
    if request.method == "GET" {
        return write_get_response(service, output, &request);
    }
    if request.method == "POST" {
        if let Some(wire_request_id) = cancel_request_id(&request.path) {
            let origin = response_origin(&service.executor().server, &request);
            cancel_scheduled_request(service, output, wire_request_id, origin)?;
            return Ok(());
        }
    }
    if request.method != "POST" || request.path != "/v1/chat/completions" {
        let mut output = output;
        write_error_with_origin(
            &mut output,
            404,
            "route not found",
            response_origin(&service.executor().server, &request),
        )?;
        return Ok(());
    }
    admit_scheduled_chat(
        service,
        output,
        RequestId(request_id),
        received_at_ns,
        deadline,
        cancellation,
        &request,
    )
    .map_err(|error| {
        eprintln!("client {client_id} at {peer_addr} request {request_id} failed: {error}");
        error
    })
}

fn write_options_response<B: Backend>(
    service: &ChatService<B>,
    mut output: OutputSink,
    request: &HttpRequest,
) -> Result<(), Box<dyn Error>> {
    write_empty_with_origin(
        &mut output,
        204,
        "No Content",
        response_origin(&service.executor().server, request),
    )?;
    Ok(())
}

fn write_get_response<B: Backend>(
    service: &mut ChatService<B>,
    mut output: OutputSink,
    request: &HttpRequest,
) -> Result<(), Box<dyn Error>> {
    let origin = response_origin(&service.executor().server, request);
    handle_scheduled_get(service, &mut output, &request.path, origin)
}

fn cancel_request_id(path: &str) -> Option<&str> {
    path.strip_prefix("/debug/service-requests/")
        .and_then(|path| path.strip_suffix("/cancel"))
        .filter(|value| !value.is_empty() && !value.contains('/'))
}

fn tick_serve_requests<B: Backend>(service: &mut ChatService<B>) -> Result<bool, Box<dyn Error>> {
    cancel_unresponsive_requests(service)?;
    if service.has_runnable_requests() {
        if let Err(error) = service.tick_batch(scheduler_now_ns()) {
            eprintln!("scheduled request failed: {error}");
        }
        let reserved = service.scheduler().reserved_kv_bytes();
        service.executor_mut().sample_trace("tick", reserved);
        sample_server_metrics(&mut service.executor_mut().server, reserved);
        return Ok(true);
    }
    Ok(false)
}

fn cancel_unresponsive_requests<B: Backend>(
    service: &mut ChatService<B>,
) -> Result<(), Box<dyn Error>> {
    let now = std::time::Instant::now();
    let expired = expired_request_ids(service, now);
    for request_id in expired {
        service.executor_mut().expire(request_id);
        if let Err(error) = service.cancel(request_id) {
            eprintln!(
                "request deadline cleanup failed for {}: {error}",
                request_id.0
            );
        }
    }
    for request_id in cancelled_request_ids(service) {
        cancel_unresponsive_request(service, request_id);
    }
    Ok(())
}

fn expired_request_ids<B: Backend>(
    service: &ChatService<B>,
    now: std::time::Instant,
) -> Vec<RequestId> {
    service
        .executor()
        .tasks
        .iter()
        .filter_map(|(id, task)| (now >= task.deadline && !task.deadline_expired).then_some(*id))
        .collect()
}

fn cancelled_request_ids<B: Backend>(service: &ChatService<B>) -> Vec<RequestId> {
    service
        .executor()
        .tasks
        .iter()
        .filter_map(|(id, task)| {
            (task.cancellation.is_cancelled() && !task.deadline_expired).then_some(*id)
        })
        .collect()
}

fn cancel_unresponsive_request<B: Backend>(service: &mut ChatService<B>, request_id: RequestId) {
    let client_cancelled = service
        .executor()
        .tasks
        .get(&request_id)
        .is_some_and(|task| task.client_cancelled);
    if !client_cancelled {
        service.executor_mut().cancel(request_id);
    }
    if let Err(error) = service.cancel(request_id) {
        eprintln!("request cancellation failed for {}: {error}", request_id.0);
    }
}

fn cancel_scheduled_request<B: Backend>(
    service: &mut ChatService<B>,
    mut stream: OutputSink,
    wire_request_id: &str,
    origin: Option<String>,
) -> Result<(), Box<dyn Error>> {
    let Some(request_id) = service.executor().request_id_by_wire(wire_request_id) else {
        write_error_with_origin(&mut stream, 404, "request ID is not active", origin)?;
        return Ok(());
    };
    if !service.executor_mut().cancel_requested(request_id) {
        write_error_with_origin(&mut stream, 404, "request ID is not active", origin)?;
        return Ok(());
    }
    match service.cancel(request_id) {
        Ok(status) => {
            write_json_with_origin(
                &mut stream,
                202,
                &json!({
                    "request_id": wire_request_id,
                    "status": status,
                    "reclaimed": status.is_terminal()
                }),
                origin.as_deref(),
            )?;
        }
        Err(error) => {
            write_error_with_origin(&mut stream, 409, &error.to_string(), origin)?;
        }
    }
    Ok(())
}

fn server_scheduler_policy<B: Backend>(
    server: &Server<B>,
    sessions: usize,
    batch_size: usize,
) -> Result<SchedulerPolicy, Box<dyn Error>> {
    let config = server.runtime.model().config();
    let kv_bytes_per_token = server.kv_cache_dtype.bytes_per_token(config)?;
    let max_active_requests = u32::try_from(sessions)
        .map_err(|_| invalid_data("session limit does not fit the scheduler"))?;
    let max_batch_requests = u32::try_from(batch_size)
        .map_err(|_| invalid_data("batch size does not fit the scheduler"))?
        .min(max_active_requests);
    let context_tokens = host_u64(server.context_limit)?;
    let max_reserved_kv_bytes = server.memory.logical_kv_reservation;
    if max_reserved_kv_bytes == 0 {
        return Err(invalid_data("logical KV reservation is not configured").into());
    }
    Ok(SchedulerPolicy {
        max_active_requests,
        max_queued_requests: MAX_QUEUED_REQUESTS_U32,
        max_batch_requests,
        max_reserved_kv_bytes,
        kv_bytes_per_token,
        kv_page_tokens: 16,
        max_prompt_tokens: context_tokens,
        max_output_tokens: context_tokens,
        service_quantum_tokens: 4,
        prefill_chunk_tokens: server_prefill_chunk(server)?,
        urgent_window_ns: 5_000_000,
        max_prefix_credit_tokens: context_tokens,
    }
    .validate()?)
}

fn server_prefill_chunk<B: Backend>(server: &Server<B>) -> Result<u32, Box<dyn Error>> {
    u32::try_from(server.prefill_chunk_tokens)
        .map_err(|_| invalid_data("prefill chunk does not fit the scheduler").into())
}

fn scheduler_now_ns() -> u64 {
    crate::clock::now_ns()
}

fn metrics_memory_topology(topology: PolicyMemoryTopology) -> MetricsMemoryTopology {
    match topology {
        PolicyMemoryTopology::Discrete => MetricsMemoryTopology::Discrete,
        PolicyMemoryTopology::Unified => MetricsMemoryTopology::UnifiedParent,
        PolicyMemoryTopology::Cpu => MetricsMemoryTopology::CpuParent,
    }
}

fn metrics_metadata_bytes(
    max_active_requests: usize,
    max_queued_requests: usize,
    max_output_tokens: usize,
    max_output_bytes: usize,
) -> Result<u64, Box<dyn Error>> {
    let max_in_flight = metrics_in_flight_bound(max_active_requests, max_queued_requests)?;
    let buffers = metrics_buffer_bytes(max_in_flight, max_output_tokens, max_output_bytes)?;
    let histories = metrics_history_bytes(max_in_flight)?;
    let trace = trace_metadata_bytes(max_active_requests)?;
    let total = metadata_sum([
        buffers.0,
        buffers.1,
        buffers.2,
        histories.0,
        histories.1,
        trace,
    ])?;
    host_u64(total)
        .map_err(|error| invalid_data(format!("metrics metadata bound is invalid: {error}")))
        .map_err(Into::into)
}

fn trace_metadata_bytes(max_active_requests: usize) -> Result<usize, Box<dyn Error>> {
    let request_ids = metadata_product(
        MAX_SERVICE_TRACE_EVENTS,
        max_active_requests,
        std::mem::size_of::<u64>() * 3,
        "trace request identity",
    )?;
    let event_storage = metadata_product(
        MAX_SERVICE_TRACE_EVENTS,
        std::mem::size_of::<ServiceTraceEvent>(),
        1,
        "trace event",
    )?;
    let memory_storage = metadata_product(
        MAX_MEMORY_SAMPLES,
        std::mem::size_of::<TraceMemorySample>() + TRACE_TEXT_BYTES,
        1,
        "trace memory sample",
    )?;
    let class_storage = trace_class_metadata_bytes()?;
    let active_prefill = metadata_product(
        max_active_requests,
        std::mem::size_of::<RequestId>() + TRACE_MAP_ENTRY_HEADROOM,
        1,
        "trace active prefill",
    )?;
    let identities = TRACE_TEXT_BYTES
        .checked_mul(3)
        .ok_or_else(|| invalid_data("trace identity metadata bound overflowed"))?;
    let total = metadata_sum([
        request_ids,
        event_storage,
        memory_storage,
        class_storage,
        active_prefill,
        identities,
    ])?;
    total
        .checked_mul(2)
        .ok_or_else(|| invalid_data("trace metadata bound overflowed").into())
}

fn trace_class_metadata_bytes() -> Result<usize, Box<dyn Error>> {
    let class_entry = std::mem::size_of::<(String, TraceMemoryClass)>()
        .checked_add(TRACE_TEXT_BYTES)
        .and_then(|bytes| bytes.checked_add(TRACE_MAP_ENTRY_HEADROOM))
        .ok_or_else(|| invalid_data("trace memory class metadata bound overflowed"))?;
    metadata_product(
        MAX_MEMORY_SAMPLES,
        MemoryClass::ALL.len(),
        class_entry,
        "trace memory class",
    )
}

fn metrics_in_flight_bound(
    max_active_requests: usize,
    max_queued_requests: usize,
) -> Result<usize, Box<dyn Error>> {
    max_active_requests
        .checked_add(max_queued_requests)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid_data("metrics request metadata bound overflowed").into())
}

fn metrics_buffer_bytes(
    max_in_flight: usize,
    max_output_tokens: usize,
    max_output_bytes: usize,
) -> Result<(usize, usize, usize), Box<dyn Error>> {
    let output_slots = max_in_flight
        .checked_add(MAX_TERMINAL_OUTPUTS)
        .ok_or_else(|| invalid_data("metrics output slot bound overflowed"))?;
    Ok((
        metadata_product(max_in_flight, max_output_tokens, size_of::<u64>(), "token")?,
        metadata_product(
            output_slots,
            crate::transport::output_metadata_bytes(),
            1,
            "pressure",
        )?,
        metadata_product(output_slots, max_output_bytes, 1, "output")?,
    ))
}

fn metrics_history_bytes(max_in_flight: usize) -> Result<(usize, usize), Box<dyn Error>> {
    let terminal = metadata_product(
        MAX_TERMINAL_OUTPUTS,
        size_of::<TerminalOutput>(),
        1,
        "terminal",
    )?;
    let collector = collector_metadata_bytes(max_in_flight)
        .map_err(|error| invalid_data(format!("collector metadata bound is invalid: {error}")))?;
    let wire = metrics_response_bytes()
        .checked_mul(2)
        .ok_or_else(|| invalid_data("metrics response bound overflowed"))?;
    let retained = retained_metadata_bytes();
    Ok((terminal, metadata_sum([retained, collector, wire])?))
}

fn metadata_product(
    left: usize,
    right: usize,
    scale: usize,
    label: &str,
) -> Result<usize, Box<dyn Error>> {
    left.checked_mul(right)
        .and_then(|value| value.checked_mul(scale))
        .ok_or_else(|| invalid_data(format!("metrics {label} metadata bound overflowed")).into())
}

fn metadata_sum(values: impl IntoIterator<Item = usize>) -> Result<usize, Box<dyn Error>> {
    values.into_iter().try_fold(0_usize, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| invalid_data("metrics metadata bound overflowed").into())
    })
}

fn memory_runtime_error(error: leone::MemoryError) -> RuntimeError {
    RuntimeError::Backend(BackendError::Memory(error))
}

fn sample_server_metrics<B: Backend>(server: &mut Server<B>, reserved_bytes: u64) {
    let parent = server.memory.parent_observation(&server.runtime);
    server.metrics.sample_runtime_with_parent(
        &server.runtime,
        scheduler_now_ns(),
        reserved_bytes,
        parent,
    );
}

fn deadline_ns(deadline: std::time::Instant) -> u64 {
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    scheduler_now_ns().saturating_add(u64::try_from(remaining.as_nanos()).unwrap_or(u64::MAX))
}

fn handle_scheduled_get<B: Backend>(
    service: &mut ChatService<B>,
    stream: &mut dyn Write,
    path: &str,
    origin: Option<String>,
) -> Result<(), Box<dyn Error>> {
    if handle_scheduled_debug_get(service, stream, path)? {
        return Ok(());
    }
    handle_scheduled_basic_get(service, stream, path, origin)
}

fn handle_scheduled_basic_get<B: Backend>(
    service: &mut ChatService<B>,
    stream: &mut dyn Write,
    path: &str,
    origin: Option<String>,
) -> Result<(), Box<dyn Error>> {
    match path {
        "/health" => {
            write_json_with_origin(stream, 200, &json!({"status": "ok"}), origin.as_deref())?
        }
        "/v1/models" => write_scheduled_models(service, stream, origin.as_deref())?,
        "/debug/service-trace" => write_scheduled_trace(service, stream, origin.as_deref())?,
        _ => write_error_with_origin(stream, 404, "route not found", origin)?,
    }
    Ok(())
}

fn handle_scheduled_debug_get<B: Backend>(
    service: &mut ChatService<B>,
    stream: &mut dyn Write,
    path: &str,
) -> Result<bool, Box<dyn Error>> {
    if path.starts_with("/debug/service-requests/") {
        write_scheduled_terminal(service, stream, path)?;
        return Ok(true);
    }
    match path {
        "/debug/service-metrics" => write_scheduled_metrics(service, stream)?,
        "/debug/service-identity" => write_scheduled_identity(service, stream)?,
        "/debug/service-capabilities" => write_scheduled_capabilities(stream)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn write_scheduled_metrics<B: Backend>(
    service: &mut ChatService<B>,
    stream: &mut dyn Write,
) -> Result<(), io::Error> {
    service.executor_mut().drain_transport_observations();
    service.executor().server.metrics.write_response(stream)
}

fn write_scheduled_identity<B: Backend>(
    service: &ChatService<B>,
    stream: &mut dyn Write,
) -> Result<(), io::Error> {
    stream.write_all(&identity_response(&service.executor().server.identity)?)
}

fn write_scheduled_capabilities(stream: &mut dyn Write) -> Result<(), io::Error> {
    write_json_with_origin(
        stream,
        200,
        &json!({
            "schema_version": 1,
            "branch_methods": ["leone_fork_session"]
        }),
        None,
    )
}

fn write_scheduled_terminal<B: Backend>(
    service: &mut ChatService<B>,
    stream: &mut dyn Write,
    path: &str,
) -> Result<(), io::Error> {
    service.executor_mut().drain_transport_observations();
    let request_id = path
        .strip_prefix("/debug/service-requests/")
        .unwrap_or_default();
    stream.write_all(
        &service
            .executor()
            .server
            .metrics
            .terminal_response_bytes(request_id)?,
    )
}

fn write_scheduled_models<B: Backend>(
    service: &ChatService<B>,
    stream: &mut dyn Write,
    origin: Option<&str>,
) -> Result<(), io::Error> {
    write_json_with_origin(
        stream,
        200,
        &json!({
            "object": "list",
            "data": [{
                "id": service.executor().server.model_id,
                "object": "model",
                "owned_by": "local"
            }]
        }),
        origin,
    )
}

fn write_scheduled_trace<B: Backend>(
    service: &mut ChatService<B>,
    stream: &mut dyn Write,
    origin: Option<&str>,
) -> Result<(), io::Error> {
    let reserved = service.scheduler().reserved_kv_bytes();
    let executor = service.executor_mut();
    let response = executor.trace.response(&executor.server.runtime, reserved);
    write_json_with_origin(stream, 200, &response, origin)
}

fn admit_scheduled_chat<B: Backend>(
    service: &mut ChatService<B>,
    mut stream: OutputSink,
    request_id: RequestId,
    received_at_ns: u64,
    deadline: std::time::Instant,
    cancellation: Cancellation,
    http: &HttpRequest,
) -> Result<(), Box<dyn Error>> {
    service
        .executor_mut()
        .server
        .metrics
        .request_started(request_id, received_at_ns);
    let wire_request_id = new_completion_id();
    let prepared = prepare_chat_plan(&service.executor().server, http, wire_request_id.clone());
    let plan = match prepared {
        Ok(plan) => plan,
        Err(error) => {
            service
                .executor_mut()
                .server
                .metrics
                .request_finished_with_wire_id(
                    request_id,
                    &wire_request_id,
                    TerminalObservation {
                        completed_at_ns: scheduler_now_ns(),
                        status: RequestStatus::Rejected,
                        cause: TerminalCause::Rejected,
                        disconnected: false,
                        failed: false,
                        reclaimed: true,
                        delivery_status: DeliveryStatus::NotAttempted,
                    },
                );
            let result = write_error_with_origin(
                &mut stream,
                400,
                &error.to_string(),
                response_origin(&service.executor().server, http),
            );
            service
                .executor_mut()
                .server
                .metrics
                .update_delivery_status(
                    &wire_request_id,
                    response_delivery_status(&result, &stream),
                );
            service.executor_mut().retain_terminal_output(
                request_id,
                wire_request_id,
                stream.clone(),
            );
            result?;
            return Ok(());
        }
    };
    if active_session_exists(service, &plan.session_id) {
        let mut stream = stream;
        let wire_request_id = plan.completion_id.clone();
        service
            .executor_mut()
            .server
            .metrics
            .request_finished_with_wire_id(
                request_id,
                &wire_request_id,
                TerminalObservation {
                    completed_at_ns: scheduler_now_ns(),
                    status: RequestStatus::Rejected,
                    cause: TerminalCause::Rejected,
                    disconnected: false,
                    failed: false,
                    reclaimed: true,
                    delivery_status: DeliveryStatus::NotAttempted,
                },
            );
        let result = write_error_with_origin(
            &mut stream,
            409,
            "the requested Leone session is already running",
            plan.cors_origin.clone(),
        );
        service
            .executor_mut()
            .server
            .metrics
            .update_delivery_status(&wire_request_id, response_delivery_status(&result, &stream));
        service
            .executor_mut()
            .retain_terminal_output(request_id, wire_request_id, stream.clone());
        result?;
        return Ok(());
    }
    let spec = scheduled_chat_spec(request_id, &plan, deadline)?;
    let wire_request_id = plan.completion_id.clone();
    service
        .executor_mut()
        .wire_request_ids
        .insert(wire_request_id.clone(), request_id);
    service
        .executor_mut()
        .register_queued_output(request_id, wire_request_id, stream.clone());
    let admission = ChatAdmission(RefCell::new(Some(PendingChat {
        plan,
        stream,
        cancellation,
        deadline,
    })));
    admit_pending_chat(service, spec, &admission)
}

#[cfg(test)]
fn scheduled_request_or_error<T>(
    stream: &mut dyn Write,
    result: Result<T, Box<dyn Error>>,
    origin: Option<String>,
) -> Result<Option<T>, Box<dyn Error>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            write_error_with_origin(stream, 400, &error.to_string(), origin)?;
            Ok(None)
        }
    }
}

fn prefix_source_for_plan<B: Backend>(server: &Server<B>, plan: &ChatPlan) -> PrefixSource {
    match &plan.source {
        ChatPlanSource::Continue => prefix_source_for_session(server, &plan.session_id),
        ChatPlanSource::PrefixReuse { source_id } => prefix_source_for_session(server, source_id),
        ChatPlanSource::ExplicitFork { .. } => PrefixSource::ForkParent,
    }
}

fn record_chat_plan_metrics<B: Backend>(
    server: &mut Server<B>,
    plan: &ChatPlan,
    prefix_source: PrefixSource,
    fork_source: Option<ForkSource>,
) {
    match &plan.source {
        ChatPlanSource::Continue | ChatPlanSource::PrefixReuse { .. } => server
            .metrics
            .prefix_reuse_observed(prefix_source, plan.prefix_reused_tokens),
        ChatPlanSource::ExplicitFork { .. } => {
            let source = fork_source.unwrap_or(ForkSource::Unavailable);
            let fork_prefix_source = if source == ForkSource::Unavailable {
                PrefixSource::Unavailable
            } else {
                PrefixSource::ForkParent
            };
            server
                .metrics
                .prefix_reuse_observed(fork_prefix_source, plan.prefix_reused_tokens);
        }
    }
}

fn prefix_source_for_session<B: Backend>(server: &Server<B>, session_id: &str) -> PrefixSource {
    if server.sessions.contains_key(session_id) {
        PrefixSource::LiveSession
    } else if server.hibernated.contains_key(session_id) {
        PrefixSource::HibernatedSession
    } else if server.persisted.contains_key(session_id) {
        PrefixSource::PersistedSession
    } else {
        PrefixSource::NewSession
    }
}

fn fork_source_for_session<B: Backend>(server: &Server<B>, session_id: &str) -> ForkSource {
    match prefix_source_for_session(server, session_id) {
        PrefixSource::LiveSession => ForkSource::LiveSession,
        PrefixSource::HibernatedSession => ForkSource::HibernatedSession,
        PrefixSource::PersistedSession => ForkSource::PersistedSession,
        PrefixSource::NewSession | PrefixSource::ForkParent | PrefixSource::Unavailable => {
            ForkSource::Unavailable
        }
    }
}

fn active_session_exists<B: Backend>(service: &ChatService<B>, session_id: &str) -> bool {
    service
        .executor()
        .tasks
        .values()
        .any(|active| active.session_id == session_id)
}

fn scheduled_chat_spec(
    request_id: RequestId,
    plan: &ChatPlan,
    deadline: std::time::Instant,
) -> Result<RequestSpec, Box<dyn Error>> {
    Ok(RequestSpec {
        id: request_id,
        arrival_ns: scheduler_now_ns(),
        prompt_tokens: host_u64(plan.prompt_tokens.len())?,
        prefix_reused_tokens: host_u64(plan.prefix_reused_tokens)?,
        max_output_tokens: host_u64(plan.options.max_tokens)?,
        priority: 1,
        deadline_ns: Some(deadline_ns(deadline)),
    })
}

fn admit_pending_chat<B: Backend>(
    service: &mut ChatService<B>,
    spec: RequestSpec,
    admission: &ChatAdmission,
) -> Result<(), Box<dyn Error>> {
    let request_id = spec.id;
    let outcome = match service.admit(spec, admission, scheduler_now_ns()) {
        Ok(outcome) => outcome,
        Err(error) => return finish_admission_error(service, request_id, error),
    };
    match outcome {
        AdmissionOutcome::Admitted { .. } => {}
        AdmissionOutcome::Rejected { reason } => {
            finish_rejected_admission(service, request_id, admission, reason)?;
        }
    }
    Ok(())
}

fn finish_admission_error<B: Backend>(
    service: &mut ChatService<B>,
    request_id: RequestId,
    error: leone::service::ServiceError<io::Error>,
) -> Result<(), Box<dyn Error>> {
    let output = service.executor_mut().take_queued_output(request_id);
    let wire_request_id = output.as_ref().map_or_else(
        || request_id.0.to_string(),
        |value| value.wire_request_id.clone(),
    );
    service
        .executor_mut()
        .wire_request_ids
        .remove(&wire_request_id);
    let delivery_status = output
        .as_ref()
        .map(queued_output_delivery_status)
        .unwrap_or(DeliveryStatus::NotAttempted);
    service
        .executor_mut()
        .server
        .metrics
        .request_finished_with_wire_id(
            request_id,
            &wire_request_id,
            TerminalObservation {
                completed_at_ns: scheduler_now_ns(),
                status: RequestStatus::Rejected,
                cause: TerminalCause::AdmissionFailure,
                disconnected: false,
                failed: true,
                reclaimed: true,
                delivery_status,
            },
        );
    if let Some(output) = output {
        service.executor_mut().retain_terminal_output(
            request_id,
            output.wire_request_id,
            output.stream,
        );
    }
    Err(error.into())
}

fn finish_rejected_admission<B: Backend>(
    service: &mut ChatService<B>,
    request_id: RequestId,
    admission: &ChatAdmission,
    reason: AdmissionReject,
) -> Result<(), Box<dyn Error>> {
    let mut pending = admission
        .0
        .borrow_mut()
        .take()
        .expect("rejected admission does not create executor state");
    let wire_request_id = pending.plan.completion_id.clone();
    service
        .executor_mut()
        .server
        .metrics
        .request_finished_with_wire_id(
            request_id,
            &wire_request_id,
            TerminalObservation {
                completed_at_ns: scheduler_now_ns(),
                status: RequestStatus::Rejected,
                cause: TerminalCause::Rejected,
                disconnected: false,
                failed: false,
                reclaimed: true,
                delivery_status: DeliveryStatus::NotAttempted,
            },
        );
    let result = write_admission_rejection_with_origin(
        &mut pending.stream,
        reason,
        pending.plan.cors_origin.as_deref(),
    );
    service
        .executor_mut()
        .server
        .metrics
        .update_delivery_status(
            &wire_request_id,
            response_delivery_status(&result, &pending.stream),
        );
    if let Some(output) = service.executor_mut().take_queued_output(request_id) {
        service
            .executor_mut()
            .wire_request_ids
            .remove(&output.wire_request_id);
        service.executor_mut().retain_terminal_output(
            request_id,
            output.wire_request_id,
            output.stream,
        );
    }
    result?;
    Ok(())
}

fn prepare_chat_plan<B: Backend>(
    server: &Server<B>,
    http: &HttpRequest,
    completion_id: String,
) -> Result<ChatPlan, Box<dyn Error>> {
    let (request, stop_sequences, prompt_tokens) = parse_chat_plan_input(server, http)?;
    let options = chat_plan_options(server, &request, &prompt_tokens)?;
    let requested_session = requested_chat_session(&request, http);
    let (session_id, source) = chat_plan_session(
        server,
        requested_session,
        request.leone_fork_session.as_deref(),
        &prompt_tokens,
        &options,
    )?;
    let prefix_reused_tokens =
        scheduler_prefix_reused_tokens(server, &session_id, &source, &prompt_tokens, &options)?;
    let request_sha256 = sha256_bytes(&http.body);
    let created = unix_seconds()?;
    Ok(ChatPlan {
        request,
        stop_sequences,
        prompt_tokens,
        options,
        session_id,
        source,
        prefix_reused_tokens,
        request_sha256,
        created,
        completion_id,
        cors_origin: response_origin(server, http),
    })
}

fn new_completion_id() -> String {
    format!("chatcmpl-{}", Uuid::new_v4().simple())
}

fn scheduler_prefix_reused_tokens<B: Backend>(
    server: &Server<B>,
    session_id: &str,
    source: &ChatPlanSource,
    prompt_tokens: &[u32],
    options: &GenerateOptions,
) -> Result<usize, RuntimeError> {
    let id = match source {
        ChatPlanSource::Continue => session_id,
        ChatPlanSource::ExplicitFork { parent_id } => parent_id,
        ChatPlanSource::PrefixReuse { source_id } => source_id,
    };
    if server.session_discard_pending(id) {
        return Ok(0);
    }
    let Some(stored) = server.sessions.get(id) else {
        // Host wake and archive replay receive no physical credit before leasing.
        return Ok(0);
    };
    server
        .runtime
        .reusable_prefill_tokens(&stored.generation, prompt_tokens, options)
}

fn parse_chat_plan_input<B: Backend>(
    server: &Server<B>,
    http: &HttpRequest,
) -> Result<ChatPlanInput, Box<dyn Error>> {
    reject_duplicate_json_keys(&http.body)?;
    let request: ChatRequest = serde_json::from_slice(&http.body)?;
    validate_session_identifiers(&request, http)?;
    validate_chat_request(&request, &server.model_id)?;
    let stop_sequences = chat_stop_sequences(&request)?;
    let prompt_tokens = chat_tokens(
        server.runtime.model().tokenizer(),
        server.runtime.model().config().architecture,
        &request.messages,
        request.tools.as_deref(),
        request.tool_choice.as_ref(),
        request.leone_template,
    )?;
    Ok((request, stop_sequences, prompt_tokens))
}

fn chat_stop_sequences(request: &ChatRequest) -> Result<Vec<Vec<u8>>, io::Error> {
    request_stop_sequences(request)
}

fn official_llama_tool_header_token(
    request: &ChatRequest,
    architecture: leone::ModelArchitecture,
    header_token: Option<u32>,
) -> Option<u32> {
    if request.leone_template == ChatTemplateMode::Official
        && architecture == leone::ModelArchitecture::Llama
        && effective_tool_mode(request)
    {
        header_token
    } else {
        None
    }
}

fn chat_plan_options<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    prompt_tokens: &[u32],
) -> Result<GenerateOptions, Box<dyn Error>> {
    let mut options = chat_plan_base_options(server, request, prompt_tokens)?;
    configure_chat_plan_options(server, request, &mut options)?;
    Ok(options)
}

fn chat_plan_base_options<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    prompt_tokens: &[u32],
) -> Result<GenerateOptions, Box<dyn Error>> {
    let max_tokens = chat_output_budget(request, prompt_tokens.len(), server.context_limit)?;
    let mut options = GenerateOptions::greedy(max_tokens);
    options.decode_execution = crate::decode_execution_for_runtime(&server.runtime, false);
    Ok(options)
}

fn serve_context_limit(requested: Option<usize>, model_limit: usize) -> Result<usize, io::Error> {
    let limit = requested.unwrap_or(model_limit);
    if limit == 0 || limit > model_limit {
        return Err(invalid_data(
            "context limit must be nonzero and at most the model context",
        ));
    }
    Ok(limit)
}

fn chat_output_budget(
    request: &ChatRequest,
    prompt_tokens: usize,
    capacity: usize,
) -> Result<usize, io::Error> {
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 {
        return Err(invalid_data("max_tokens must be nonzero"));
    }
    let remaining = capacity
        .checked_sub(prompt_tokens)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid_data("the chat prompt exceeds the configured context limit"))?;
    if max_tokens > remaining {
        return Err(invalid_data(
            "requested output exceeds the configured context limit",
        ));
    }
    Ok(max_tokens)
}

fn configure_chat_plan_options<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    options: &mut GenerateOptions,
) -> Result<(), Box<dyn Error>> {
    options.kv_cache_dtype = server.kv_cache_dtype;
    options.seed = request.seed.unwrap_or(0);
    options.sampler = sampler(request)?;
    options.penalties = penalties(request)?;
    options.mirostat = mirostat(request)?;
    options.output_constraint = response_constraint(request.response_format.as_ref())?;
    options.speculation = if options.output_constraint.is_some() {
        Speculation::Disabled
    } else {
        speculation(request)?
    };
    let header_token = server
        .runtime
        .model()
        .tokenizer()
        .token_id(LLAMA_HEADER_START)
        .ok();
    if !chat_stop_sequences(request)?.is_empty()
        || official_llama_tool_header_token(
            request,
            server.runtime.model().config().architecture,
            header_token,
        )
        .is_some()
    {
        options.speculation = Speculation::Disabled;
    }
    apply_chat_execution_plan(server, options);
    options.prefill_chunk_tokens = server.prefill_chunk_tokens;
    Ok(())
}

#[cfg(feature = "cuda")]
fn apply_chat_execution_plan<B: Backend>(server: &Server<B>, options: &mut GenerateOptions) {
    if let Some(plan) = server.execution_plan {
        plan.apply(options);
    }
}

#[cfg(not(feature = "cuda"))]
fn apply_chat_execution_plan<B: Backend>(_server: &Server<B>, _options: &mut GenerateOptions) {}

fn requested_chat_session<'a>(request: &'a ChatRequest, http: &'a HttpRequest) -> Option<&'a str> {
    request
        .leone_session
        .as_deref()
        .or_else(|| http.headers.get("x-leone-session").map(String::as_str))
}

fn validate_session_identifiers(
    request: &ChatRequest,
    http: &HttpRequest,
) -> Result<(), io::Error> {
    let header_session = http.headers.get("x-leone-session").map(String::as_str);
    for (name, value) in [
        ("leone_session", request.leone_session.as_deref()),
        ("leone_fork_session", request.leone_fork_session.as_deref()),
        ("x-leone-session", header_session),
    ] {
        if let Some(value) = value {
            validate_session_identifier(name, value)?;
        }
    }
    if let (Some(body), Some(header)) = (request.leone_session.as_deref(), header_session) {
        if body != header {
            return Err(invalid_data(
                "leone_session and x-leone-session must match when both are set",
            ));
        }
    }
    Ok(())
}

fn validate_session_identifier(name: &str, value: &str) -> Result<(), io::Error> {
    if value.is_empty() {
        return Err(invalid_data(format!("{name} must not be empty")));
    }
    if value.len() > MAX_SESSION_ID_BYTES {
        return Err(invalid_data(format!(
            "{name} exceeds the {MAX_SESSION_ID_BYTES}-byte limit"
        )));
    }
    if value
        .bytes()
        .any(|byte| !(0x21..=0x7e).contains(&byte) || byte == b',')
    {
        return Err(invalid_data(format!(
            "{name} must contain visible ASCII characters without commas"
        )));
    }
    Ok(())
}

fn chat_plan_session<B: Backend>(
    server: &Server<B>,
    requested_session: Option<&str>,
    fork_parent: Option<&str>,
    prompt_tokens: &[u32],
    options: &GenerateOptions,
) -> Result<(String, ChatPlanSource), Box<dyn Error>> {
    if let Some(parent_id) = fork_parent {
        return fork_chat_session(server, requested_session, parent_id);
    }
    if let Some(session_id) = requested_session {
        return Ok((session_id.to_owned(), ChatPlanSource::Continue));
    }
    let session_id = Uuid::new_v4().to_string();
    let source = server
        .select_session(prompt_tokens, options)?
        .map(|source_id| ChatPlanSource::PrefixReuse { source_id })
        .unwrap_or(ChatPlanSource::Continue);
    Ok((session_id, source))
}

fn fork_chat_session<B: Backend>(
    server: &Server<B>,
    requested_session: Option<&str>,
    parent_id: &str,
) -> Result<(String, ChatPlanSource), Box<dyn Error>> {
    if parent_id.is_empty() {
        return Err(invalid_data("leone_fork_session must not be empty").into());
    }
    let session_id = requested_session
        .map(str::to_owned)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    if session_id == parent_id {
        return Err(invalid_data("a fork must use a new session identifier").into());
    }
    if server.session_discard_pending(parent_id) {
        return Err(invalid_data("the fork parent session cleanup is pending").into());
    }
    if server_has_session_identity(server, &session_id)? {
        return Err(invalid_data("a fork must not replace an existing session").into());
    }
    if !server_has_session_identity(server, parent_id)? {
        return Err(invalid_data("the fork parent session does not exist").into());
    }
    Ok((
        session_id,
        ChatPlanSource::ExplicitFork {
            parent_id: parent_id.to_owned(),
        },
    ))
}

fn server_has_session_identity<B: Backend>(
    server: &Server<B>,
    session_id: &str,
) -> Result<bool, Box<dyn Error>> {
    Ok(server.sessions.contains_key(session_id)
        || server.hibernated.contains_key(session_id)
        || server.has_persisted_session(session_id)?)
}

struct ChatGeneration {
    prompt_tokens: Vec<u32>,
    tokens: Vec<u32>,
    request_replay: Option<leone::SessionReplay>,
    cancelled: bool,
    stop_hit: bool,
    tool_header_stop: bool,
    eos: bool,
    prefill_chunks: u64,
    prefill_tokens: u64,
    decode_quanta: u64,
    phase_trace: Vec<DispatchKind>,
}

#[derive(Debug, Serialize)]
struct ChatTelemetry {
    prefill_chunks: u64,
    prefill_tokens: u64,
    decode_quanta: u64,
    phase_trace: Vec<DispatchKind>,
}

impl ChatGeneration {
    fn telemetry(&self) -> ChatTelemetry {
        ChatTelemetry {
            prefill_chunks: self.prefill_chunks,
            prefill_tokens: self.prefill_tokens,
            decode_quanta: self.decode_quanta,
            phase_trace: self.phase_trace.clone(),
        }
    }
}

struct ChatResponseDetails {
    decoded_content: Option<String>,
    generated_tool_calls: Option<Vec<GeneratedToolCall>>,
    finish_reason: String,
}

#[derive(Debug)]
struct InvalidGeneratedResponse(String);

impl std::fmt::Display for InvalidGeneratedResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for InvalidGeneratedResponse {}

#[derive(Clone, Copy)]
struct ResponseWireState {
    request_stream: bool,
    started: bool,
}

struct ChatStreamContext<'a> {
    stream_mode: bool,
    tool_mode: bool,
    tool_header_token: Option<u32>,
    streamed: &'a mut Utf8Stream,
    stop: &'a mut StopMatcher,
    stop_hit: &'a Cell<bool>,
    tool_header_stop: &'a Cell<bool>,
    stream: &'a mut OutputSink,
    token_boundaries_ns: &'a mut Vec<u64>,
    token_boundary_ns: Option<u64>,
    model_id: &'a str,
    completion_id: &'a str,
    created: u64,
    disconnected: &'a mut bool,
    disconnected_at_ns: &'a mut Option<u64>,
}

struct ResponseWriter<'a, W: ?Sized> {
    stream: &'a mut W,
    started: &'a mut bool,
}

impl<'a, W: Write + ?Sized> ResponseWriter<'a, W> {
    fn new(stream: &'a mut W, started: &'a mut bool) -> Self {
        Self { stream, started }
    }
}

impl<W: Write + ?Sized> Write for ResponseWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let result = self.stream.write(bytes);
        if result.as_ref().is_ok_and(|written| *written > 0) {
            *self.started = true;
        }
        result
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

struct ChatResponseContext<'a> {
    request: &'a ChatRequest,
    session_id: String,
    completion_id: String,
    created: u64,
    request_sha256: String,
    seed: u64,
    model_id: String,
    cors_origin: Option<String>,
}

fn write_chat_headers_io<W: Write + ?Sized>(
    stream: &mut W,
    session_id: &str,
    completion_id: &str,
    created: u64,
    model_id: &str,
    origin: Option<&str>,
) -> io::Result<()> {
    write_stream_headers_origin(stream, session_id, origin)?;
    write_sse(
        stream,
        &json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
            "usage": null
        }),
    )?;
    Ok(())
}

fn finish_chat_response<B: Backend, W: Write + ?Sized>(
    server: &mut Server<B>,
    stream: &mut W,
    context: ChatResponseContext<'_>,
    streamed: &mut Utf8Stream,
    stored: &mut Option<StoredSession<B>>,
    generated: ChatGeneration,
) -> Result<(), Box<dyn Error>> {
    let details = chat_response_details(server, context.request, &generated)?;
    let replay = generated
        .request_replay
        .ok_or_else(|| invalid_data("the request has no committed prefill record"))?;
    let session = response_session_record(
        &context.session_id,
        reuse_class_name(replay.reuse_class),
        replay.cached_tokens,
        replay.reused_tokens,
        replay.replayed_tokens,
        replay.computed_tokens,
    )?;
    let receipt = sign_chat_receipt(
        server,
        &context.request_sha256,
        context.seed,
        &generated,
        &details,
        session,
    )?;
    #[cfg(test)]
    if let Some(delay) = server.test_finalize_delay {
        thread::sleep(delay);
    }
    persist_chat_session(server, &context.session_id, stored, generated.cancelled)?;
    if let Err(error) = write_response_receipt(&server.receipts, &receipt) {
        discard_response_session(server, &context.session_id);
        return Err(error.into());
    }
    write_chat_response(stream, &context, streamed, &generated, &details, &receipt)
}

fn chat_response_details<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    generated: &ChatGeneration,
) -> Result<ChatResponseDetails, Box<dyn Error>> {
    let decoded_content = decoded_chat_content(server, request, generated)?;
    let generated_tool_calls = decoded_content
        .as_deref()
        .map(|content| {
            parse_generated_tool_calls(
                content,
                request.tools.as_deref(),
                request.tool_choice.as_ref(),
            )
        })
        .transpose()
        .map_err(|error| Box::new(InvalidGeneratedResponse(error.to_string())) as Box<dyn Error>)?
        .flatten();
    validate_generated_tool_choice(request.tool_choice.as_ref(), generated_tool_calls.is_some())
        .map_err(|error| Box::new(InvalidGeneratedResponse(error.to_string())) as Box<dyn Error>)?;
    let finish_reason = chat_finish_reason(generated, generated_tool_calls.is_some());
    Ok(ChatResponseDetails {
        decoded_content,
        generated_tool_calls,
        finish_reason: finish_reason.to_owned(),
    })
}

fn decoded_chat_content<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    generated: &ChatGeneration,
) -> Result<Option<String>, Box<dyn Error>> {
    if request.stream && request.tools.is_none() {
        return Ok(None);
    }
    let content = decode_chat_tokens_lossy(
        server.runtime.model().tokenizer(),
        generated_content_tokens(generated),
    )?;
    let sequences = chat_stop_sequences(request)?;
    Ok(Some(truncate_at_stop(&content, &sequences).0))
}

fn generated_content_tokens(generated: &ChatGeneration) -> &[u32] {
    if generated.tool_header_stop {
        generated
            .tokens
            .get(..generated.tokens.len().saturating_sub(1))
            .unwrap_or(&[])
    } else {
        &generated.tokens
    }
}

fn decode_chat_tokens_lossy(
    tokenizer: &leone::Tokenizer,
    tokens: &[u32],
) -> Result<String, Box<dyn Error>> {
    let mut bytes = Vec::new();
    for token in tokens {
        bytes.extend(tokenizer.token_bytes(*token)?);
    }
    Ok(decode_utf8_lossy(&bytes))
}

fn decode_utf8_lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn chat_finish_reason(generated: &ChatGeneration, has_tool_calls: bool) -> &'static str {
    if generated.stop_hit || generated.eos {
        return if has_tool_calls { "tool_calls" } else { "stop" };
    }
    if generated.cancelled {
        return "cancelled";
    }
    if has_tool_calls {
        "tool_calls"
    } else {
        "length"
    }
}

fn response_session_record(
    session_id: &str,
    reuse_class: &str,
    cached_tokens: usize,
    reused_tokens: usize,
    replayed_tokens: usize,
    computed_tokens: usize,
) -> Result<SessionReplayRecord, Box<dyn Error>> {
    Ok(SessionReplayRecord {
        session_id: session_id.to_owned(),
        reuse_class: reuse_class.to_owned(),
        cached_tokens: host_u64(cached_tokens)?,
        reused_tokens: host_u64(reused_tokens)?,
        replayed_tokens: host_u64(replayed_tokens)?,
        computed_tokens: host_u64(computed_tokens)?,
    })
}

fn sign_chat_receipt<B: Backend>(
    server: &Server<B>,
    request_sha256: &str,
    seed: u64,
    generated: &ChatGeneration,
    details: &ChatResponseDetails,
    session: SessionReplayRecord,
) -> Result<ResponseReceipt, Box<dyn Error>> {
    let response_tokens_sha256 = token_stream_sha256(&generated.tokens);
    let mut transcript = generated.prompt_tokens.clone();
    transcript.extend_from_slice(&generated.tokens);
    Ok(ResponseReceipt::sign(
        ResponseClaim {
            schema_version: RESPONSE_SCHEMA_VERSION,
            receipt_id: Uuid::new_v4(),
            created_utc: Utc::now(),
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
            model_sha256: server.model_sha256.clone(),
            request_sha256: request_sha256.to_owned(),
            prompt_tokens_sha256: token_stream_sha256(&generated.prompt_tokens),
            response_tokens_sha256,
            transcript_sha256: token_stream_sha256(&transcript),
            seed,
            prompt_tokens: host_u64(generated.prompt_tokens.len())?,
            generated_tokens: host_u64(generated.tokens.len())?,
            finish_reason: details.finish_reason.clone(),
            cancelled: generated.cancelled,
            session,
        },
        &server.signing_key,
    )?)
}

fn persist_chat_session<B: Backend>(
    server: &mut Server<B>,
    session_id: &str,
    stored: &mut Option<StoredSession<B>>,
    cancelled: bool,
) -> Result<(), Box<dyn Error>> {
    if cancelled {
        return server.discard_stored_session(
            session_id,
            stored.as_mut().expect("response session is present"),
        );
    }
    server.clock = server.clock.saturating_add(1);
    let stored_session = stored.as_mut().expect("response session is present");
    stored_session.last_used = server.clock;
    if let Err(error) = persist_session_archive(server, session_id, stored_session) {
        return Err(
            unchanged_runtime_error(RuntimeError::Backend(BackendError::operation(
                "persist session archive",
                error,
            )))
            .into(),
        );
    }
    if server.session_store.is_some() {
        if let Some(mut lease) = stored_session.reference_lease.take() {
            if let Err(error) = lease.retire() {
                let mut reference_lease = Some(lease);
                if let Err(retain_error) =
                    server.retain_reference_recovery(session_id, &mut reference_lease)
                {
                    stored_session.reference_lease = reference_lease;
                    server.sessions.insert(
                        session_id.to_owned(),
                        stored.take().expect("response session is present"),
                    );
                    return Err(retain_error);
                }
                server.sessions.insert(
                    session_id.to_owned(),
                    stored.take().expect("response session is present"),
                );
                return Err(session_persistence_error(
                    SessionFailureEffect::Unchanged,
                    "retire superseded persisted session reference",
                    error,
                )
                .into());
            }
        }
    }
    server
        .insert_session(
            session_id.to_owned(),
            stored.take().expect("response session is present"),
        )
        .map_err(|error| -> Box<dyn Error> { invalid_data(error.to_string()).into() })
}

fn persist_session_archive<B: Backend>(
    server: &mut Server<B>,
    session_id: &str,
    stored: &StoredSession<B>,
) -> Result<(), Box<dyn Error>> {
    let Some(store) = server.session_store.as_ref() else {
        return Ok(());
    };
    let metadata_allocation = reserve_session_archive_metadata(server, session_id)?;
    let persist_allocation = reserve_session_archive_storage(server, stored)?;
    let archive = store.persist(
        session_id,
        &server.model_sha256,
        stored.generation.checkpoint(),
        stored.last_used,
    )?;
    drop(persist_allocation);
    server.cache_persisted_archive_with_allocation(
        session_id.to_owned(),
        archive,
        metadata_allocation,
    )?;
    Ok(())
}

fn reserve_session_archive_metadata<B: Backend>(
    server: &Server<B>,
    session_id: &str,
) -> Result<MemoryAllocation, Box<dyn Error>> {
    server
        .memory
        .reserve_host(stored_archive_metadata_bound_for_id(session_id)?)?
        .commit()
        .map_err(|error| {
            unchanged_runtime_error(RuntimeError::Backend(BackendError::Memory(error))).into()
        })
}

fn reserve_session_archive_storage<B: Backend>(
    server: &Server<B>,
    stored: &StoredSession<B>,
) -> Result<MemoryAllocation, Box<dyn Error>> {
    server
        .memory
        .reserve_host(session_archive_persist_bound(
            stored.generation.evaluated_tokens().len(),
        )?)?
        .commit()
        .map_err(|error| {
            unchanged_runtime_error(RuntimeError::Backend(BackendError::Memory(error))).into()
        })
}

fn write_chat_response<W: Write + ?Sized>(
    stream: &mut W,
    context: &ChatResponseContext<'_>,
    streamed: &mut Utf8Stream,
    generated: &ChatGeneration,
    details: &ChatResponseDetails,
    receipt: &ResponseReceipt,
) -> Result<(), Box<dyn Error>> {
    if context.request.stream {
        return write_stream_chat_response(stream, context, streamed, generated, details, receipt);
    }
    write_nonstream_chat_response(stream, context, generated, details, receipt)
}

fn write_stream_chat_response<W: Write + ?Sized>(
    stream: &mut W,
    context: &ChatResponseContext<'_>,
    streamed: &mut Utf8Stream,
    generated: &ChatGeneration,
    details: &ChatResponseDetails,
    receipt: &ResponseReceipt,
) -> Result<(), Box<dyn Error>> {
    write_stream_pending(stream, context, streamed)?;
    write_stream_tool_payload(stream, context, details)?;
    write_stream_terminal(stream, context, generated, details, receipt)?;
    write_chunk(stream, b"data: [DONE]\n\n")?;
    finish_chunks(stream)?;
    Ok(())
}

fn write_stream_pending<W: Write + ?Sized>(
    stream: &mut W,
    context: &ChatResponseContext<'_>,
    streamed: &mut Utf8Stream,
) -> Result<(), Box<dyn Error>> {
    for content in streamed.finish() {
        write_sse(
            stream,
            &json!({
                "id": context.completion_id,
                "object": "chat.completion.chunk",
                "created": context.created,
                "model": context.model_id,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}],
                "usage": null
            }),
        )?;
    }
    Ok(())
}

fn write_stream_tool_payload<W: Write + ?Sized>(
    stream: &mut W,
    context: &ChatResponseContext<'_>,
    details: &ChatResponseDetails,
) -> Result<(), Box<dyn Error>> {
    if let Some(tool_calls) = details.generated_tool_calls.as_ref() {
        for (index, tool_call) in tool_calls.iter().enumerate() {
            write_sse(stream, &tool_call_chunk(context, index, tool_call))?;
        }
        return Ok(());
    }
    write_stream_tool_fallback(stream, context, details)
}

fn write_stream_terminal<W: Write + ?Sized>(
    stream: &mut W,
    context: &ChatResponseContext<'_>,
    generated: &ChatGeneration,
    details: &ChatResponseDetails,
    receipt: &ResponseReceipt,
) -> Result<(), Box<dyn Error>> {
    write_sse(
        stream,
        &json!({
            "id": context.completion_id,
            "object": "chat.completion.chunk",
            "created": context.created,
            "model": context.model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": details.finish_reason}],
            "usage": null,
            "leone_telemetry": generated.telemetry(),
            "leone_receipt": receipt
        }),
    )?;
    if stream_usage_requested(context.request) {
        write_sse(
            stream,
            &json!({
                "id": context.completion_id,
                "object": "chat.completion.chunk",
                "created": context.created,
                "model": context.model_id,
                "choices": [],
                "usage": usage_value(generated)
            }),
        )?;
    }
    Ok(())
}

fn stream_usage_requested(request: &ChatRequest) -> bool {
    request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false)
}

fn usage_value(generated: &ChatGeneration) -> Value {
    json!({
        "prompt_tokens": generated.prompt_tokens.len(),
        "completion_tokens": generated.tokens.len(),
        "total_tokens": generated.prompt_tokens.len() + generated.tokens.len()
    })
}

fn write_stream_tool_fallback<W: Write + ?Sized>(
    stream: &mut W,
    context: &ChatResponseContext<'_>,
    details: &ChatResponseDetails,
) -> Result<(), Box<dyn Error>> {
    if !effective_tool_mode(context.request) {
        return Ok(());
    }
    let Some(content) = details.decoded_content.as_deref() else {
        return Ok(());
    };
    if content.is_empty() {
        return Ok(());
    }
    write_sse(
        stream,
        &json!({
            "id": context.completion_id,
            "object": "chat.completion.chunk",
            "created": context.created,
            "model": context.model_id,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}],
            "usage": null
        }),
    )?;
    Ok(())
}

fn write_nonstream_chat_response<W: Write + ?Sized>(
    stream: &mut W,
    context: &ChatResponseContext<'_>,
    generated: &ChatGeneration,
    details: &ChatResponseDetails,
    receipt: &ResponseReceipt,
) -> Result<(), Box<dyn Error>> {
    let message = match details.generated_tool_calls.as_ref() {
        Some(tool_calls) => json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls
        }),
        None => json!({
            "role": "assistant",
            "content": details
                .decoded_content
                .as_ref()
                .expect("non-streaming content is decoded")
        }),
    };
    Ok(write_json_with_session_origin(
        stream,
        200,
        &context.session_id,
        &json!({
            "id": context.completion_id,
            "object": "chat.completion",
            "created": context.created,
            "model": context.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": details.finish_reason
            }],
            "usage": {
                "prompt_tokens": generated.prompt_tokens.len(),
                "completion_tokens": generated.tokens.len(),
                "total_tokens": generated.prompt_tokens.len() + generated.tokens.len()
            },
            "leone_telemetry": generated.telemetry(),
            "leone_receipt": receipt
        }),
        context.cors_origin.as_deref(),
    )?)
}

fn tool_call_chunk(
    context: &ChatResponseContext<'_>,
    index: usize,
    tool_call: &GeneratedToolCall,
) -> Value {
    json!({
        "id": context.completion_id,
        "object": "chat.completion.chunk",
        "created": context.created,
        "model": context.model_id,
        "choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": index,
            "id": tool_call.id,
            "type": tool_call.kind,
            "function": tool_call.function
        }]}, "finish_reason": null}],
        "usage": null
    })
}

impl<B: Backend> Server<B> {
    fn resident_metadata_requirement(
        &self,
        generation: &GenerationSession<B>,
        context_tokens: usize,
    ) -> Result<u64, RuntimeError> {
        let bound = self.runtime.resident_metadata_bound(context_tokens)?;
        if generation.evaluated_tokens().is_empty() {
            return Ok(bound);
        }
        Ok(bound.max(self.runtime.hibernation_metadata_bytes(generation)?))
    }

    fn allocate_resident_metadata(
        &self,
        context_tokens: usize,
    ) -> Result<ResidentMetadataLease, RuntimeError> {
        let bytes = self.runtime.resident_metadata_bound(context_tokens)?;
        let allocation = self
            .memory
            .reserve_host(bytes)?
            .commit()
            .map_err(|error| RuntimeError::Backend(BackendError::Memory(error)))?;
        Ok(ResidentMetadataLease::single(allocation))
    }

    fn ensure_resident_metadata(
        &self,
        stored: &mut StoredSession<B>,
        context_tokens: usize,
    ) -> Result<(), RuntimeError> {
        let required = self.resident_metadata_requirement(&stored.generation, context_tokens)?;
        let current = stored
            .metadata_allocation
            .as_ref()
            .map_or(0, ResidentMetadataLease::bytes);
        if required <= current {
            return Ok(());
        }
        let additional = required
            .checked_sub(current)
            .ok_or(RuntimeError::SizeOverflow)?;
        let replacement = self
            .memory
            .reserve_host(additional)?
            .commit()
            .map_err(|error| RuntimeError::Backend(BackendError::Memory(error)))?;
        match stored.metadata_allocation.as_mut() {
            Some(lease) => lease.append(replacement),
            None => {
                stored.metadata_allocation = Some(ResidentMetadataLease::single(replacement));
                Ok(())
            }
        }
    }

    fn discard_session(&mut self, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.retire_resident_session(session_id)?;
        let held_lease = self.take_session_reference_lease(session_id);
        self.discard_session_with_reference(session_id, held_lease)
    }

    fn retire_resident_session(&mut self, session_id: &str) -> Result<(), Box<dyn Error>> {
        if let Some(stored) = self.sessions.get_mut(session_id) {
            if !stored.generation.is_empty() {
                self.runtime.discard_session(&mut stored.generation)?;
            }
        }
        Ok(())
    }

    fn discard_stored_session(
        &mut self,
        session_id: &str,
        stored: &mut StoredSession<B>,
    ) -> Result<(), Box<dyn Error>> {
        self.runtime.discard_session(&mut stored.generation)?;
        let held_lease = stored.reference_lease.take();
        self.discard_session_with_reference(session_id, held_lease)
    }

    fn discard_session_with_reference(
        &mut self,
        session_id: &str,
        held_lease: Option<SessionReferenceLease>,
    ) -> Result<(), Box<dyn Error>> {
        let reference_lease = if held_lease.is_some() {
            held_lease
        } else {
            match self.current_reference_lease_for_discard(session_id) {
                Ok(lease) => lease,
                Err(error) => {
                    self.remove_session_state(session_id)?;
                    return Err(error);
                }
            }
        };
        self.discard_leased_session(session_id, reference_lease)
    }

    fn current_reference_lease_for_discard(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionReferenceLease>, Box<dyn Error>> {
        let Some(store) = self.session_store.as_ref() else {
            return Ok(None);
        };
        let owner_held =
            self.sessions.contains_key(session_id) || self.hibernated.contains_key(session_id);
        store
            .lease_reference_for_discard(session_id, owner_held)
            .map_err(|error| {
                session_persistence_error(
                    SessionFailureEffect::Quarantine,
                    "lease persisted session reference for discard",
                    error,
                )
                .into()
            })
    }

    fn retain_reference_recovery(
        &self,
        session_id: &str,
        reference_lease: &mut Option<SessionReferenceLease>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(store) = self.session_store.as_ref() {
            store
                .retain_recovery_lease(session_id, reference_lease)
                .map_err(|error| {
                    session_persistence_error(
                        SessionFailureEffect::Unchanged,
                        "retain persisted session reference",
                        error,
                    )
                })?;
        }
        Ok(())
    }

    fn take_session_reference_lease(&mut self, session_id: &str) -> Option<SessionReferenceLease> {
        if let Some(stored) = self.sessions.get_mut(session_id) {
            if stored.reference_lease.is_some() {
                return stored.reference_lease.take();
            }
        }
        self.hibernated
            .get_mut(session_id)
            .and_then(|stored| stored.reference_lease.take())
    }

    fn discard_leased_session(
        &mut self,
        session_id: &str,
        mut reference_lease: Option<SessionReferenceLease>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(lease) = reference_lease.as_mut() {
            if let Err(error) = lease.discard() {
                let lease = reference_lease.take().expect("lease exists");
                if lease.has_recovery_slot() {
                    if let Some(store) = self.session_store.as_ref() {
                        let mut lease = Some(lease);
                        if let Err(retain_error) =
                            store.retain_recovery_lease(session_id, &mut lease)
                        {
                            let retained = self.retain_owner_held_discard(
                                session_id,
                                lease.take().expect("recovery lease is present"),
                            )?;
                            assert!(retained, "a recovery lease has a bounded owner");
                            return Err(retain_error.into());
                        }
                    } else {
                        unreachable!("a recovery lease requires a session store");
                    }
                    self.remove_session_state(session_id)?;
                } else {
                    let retained = self.retain_owner_held_discard(session_id, lease)?;
                    assert!(retained, "a no-slot discard lease has a bounded owner");
                }
                return Err(session_persistence_error(
                    SessionFailureEffect::Quarantine,
                    "quarantine persisted session reference",
                    error,
                )
                .into());
            }
        }
        self.remove_session_state(session_id)?;
        Ok(())
    }

    fn retain_owner_held_discard(
        &mut self,
        session_id: &str,
        lease: SessionReferenceLease,
    ) -> Result<bool, Box<dyn Error>> {
        if let Some(stored) = self.sessions.get_mut(session_id) {
            stored.reference_lease = Some(lease);
            self.runtime.discard_session(&mut stored.generation)?;
        } else if let Some(stored) = self.hibernated.get_mut(session_id) {
            stored.reference_lease = Some(lease);
        } else {
            return Ok(false);
        }
        self.persisted.remove(session_id);
        Ok(true)
    }

    fn session_discard_pending(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .and_then(|stored| stored.reference_lease.as_ref())
            .or_else(|| {
                self.hibernated
                    .get(session_id)
                    .and_then(|stored| stored.reference_lease.as_ref())
            })
            .is_some_and(SessionReferenceLease::discard_pending)
    }

    fn require_session_cleanup_complete(
        &self,
        session_id: &str,
        message: &'static str,
    ) -> Result<(), Box<dyn Error>> {
        if self.session_discard_pending(session_id) {
            return Err(invalid_data(message).into());
        }
        Ok(())
    }

    fn remove_session_state(&mut self, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.retire_resident_session(session_id)?;
        self.sessions.remove(session_id);
        let removed_hibernation = self.hibernated.remove(session_id);
        if let Some(removed) = removed_hibernation {
            release_hibernation(removed);
        }
        self.persisted.remove(session_id);
        Ok(())
    }

    #[cfg(test)]
    fn cache_persisted_archive(
        &mut self,
        session_id: String,
        archive: StoredArchive,
    ) -> Result<(), Box<dyn Error>> {
        if self.persisted_cache_limit == 0 {
            return Ok(());
        }
        let bytes = stored_archive_metadata_bound(&session_id, &archive)?;
        let allocation = self
            .memory
            .reserve_host(bytes)?
            .commit()
            .map_err(|error| RuntimeError::Backend(BackendError::Memory(error)))?;
        self.cache_persisted_archive_with_allocation(session_id, archive, allocation)
    }

    fn cache_persisted_archive_with_allocation(
        &mut self,
        session_id: String,
        archive: StoredArchive,
        allocation: MemoryAllocation,
    ) -> Result<(), Box<dyn Error>> {
        if self.persisted_cache_limit == 0 {
            drop(allocation);
            return Ok(());
        }
        let is_replacement = self.persisted.contains_key(&session_id);
        if !is_replacement && self.persisted.len() == self.persisted_cache_limit {
            let oldest = self
                .persisted
                .iter()
                .min_by_key(|(_, archive)| archive.last_used)
                .map(|(session_id, _)| session_id.clone())
                .expect("a full persisted cache has an entry");
            self.persisted.remove(&oldest);
        }
        if !self.persisted.can_insert(&session_id) {
            drop(allocation);
            return Err(invalid_data("persisted archive table is full").into());
        }
        let entry = PersistedArchiveEntry {
            archive,
            _allocation: allocation,
        };
        if let Some(previous) = self
            .persisted
            .insert(session_id, entry)
            .map_err(|(_, _)| invalid_data("persisted archive table is full"))?
        {
            drop(previous);
        }
        Ok(())
    }

    fn persisted_metadata(
        &mut self,
        session_id: &str,
    ) -> Result<Option<StoredArchive>, Box<dyn Error>> {
        if self.session_discard_pending(session_id) {
            return Err(invalid_data("session cleanup is pending").into());
        }
        if let Some(archive) = self.persisted.get(session_id) {
            return Ok(Some(archive.archive.clone()));
        }
        let Some(store) = self.session_store.as_ref() else {
            return Ok(None);
        };
        let Some(loaded) = store.load_metadata_with_allocation(session_id)? else {
            return Ok(None);
        };
        let archive = loaded.archive.clone();
        self.cache_persisted_archive_with_allocation(
            loaded.session_id,
            loaded.archive,
            loaded.allocation,
        )?;
        Ok(Some(archive))
    }

    fn has_persisted_session(&self, session_id: &str) -> Result<bool, Box<dyn Error>> {
        if self.session_discard_pending(session_id) {
            return Err(invalid_data("session cleanup is pending").into());
        }
        if self.persisted.contains_key(session_id) {
            return Ok(true);
        }
        match self.session_store.as_ref() {
            Some(store) => store.has_reference(session_id),
            None => Ok(false),
        }
    }

    fn load_persisted_archive(
        &self,
        session_id: &str,
        stored: &StoredArchive,
        reference_lease: Option<&SessionReferenceLease>,
        serialized_bytes: u64,
    ) -> Result<leone::SessionArchive, Box<dyn Error>> {
        let store = self
            .session_store
            .as_ref()
            .ok_or_else(|| invalid_data("the session store is unavailable"))?;
        let archive = store.load_archive_with_limit(
            session_id,
            stored,
            reference_lease,
            &self.model_sha256,
            serialized_bytes,
        )?;
        if archive.evaluated_tokens().len() > self.context_limit {
            return Err(invalid_data("session archive exceeds the model context limit").into());
        }
        Ok(archive)
    }

    fn reserve_persisted_archive(
        &self,
        stored: &StoredArchive,
    ) -> Result<PersistedArchiveReservation, Box<dyn Error>> {
        let store = self
            .session_store
            .as_ref()
            .ok_or_else(|| invalid_data("the session store is unavailable"))?;
        let serialized_bytes = store.archive_size(stored)?;
        let bytes = session_archive_restore_bound(serialized_bytes)?;
        let allocation = self
            .memory
            .reserve_host(bytes)?
            .commit()
            .map_err(|error| RuntimeError::Backend(BackendError::Memory(error)))?;
        Ok(PersistedArchiveReservation {
            _allocation: allocation,
            serialized_bytes,
        })
    }

    fn attach_session_reference(
        &self,
        session_id: &str,
        stored: &mut StoredSession<B>,
    ) -> Result<(), Box<dyn Error>> {
        if stored.reference_lease.is_none() {
            stored.reference_lease = self.lease_session_reference(session_id)?;
        }
        Ok(())
    }

    fn lease_session(&mut self, plan: &ChatPlan) -> Result<StoredSession<B>, Box<dyn Error>> {
        let context_tokens = request_metadata_context(plan)?;
        match &plan.source {
            ChatPlanSource::ExplicitFork { parent_id } => {
                return self.lease_available_fork_session(parent_id, context_tokens);
            }
            ChatPlanSource::PrefixReuse { source_id } => {
                return self.lease_prefix_reuse(source_id, context_tokens);
            }
            ChatPlanSource::Continue => {}
        }
        self.lease_continue_session(&plan.session_id, context_tokens)
    }

    fn lease_continue_session(
        &mut self,
        session_id: &str,
        context_tokens: usize,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        if self.session_discard_pending(session_id) {
            self.discard_session(session_id)?;
            return self.new_cold_session(context_tokens);
        }
        if let Some(stored) = self.sessions.remove(session_id) {
            return self.attach_resident_session(session_id, stored, context_tokens);
        }
        if self.hibernated.contains_key(session_id) {
            let stored = self.wake_hibernated_session(session_id, context_tokens)?;
            return self.attach_resident_session(session_id, stored, context_tokens);
        }
        if let Some(archive) = self.persisted_metadata(session_id)? {
            return self.restore_persisted_continuation(session_id, archive, context_tokens);
        }
        self.new_cold_session(context_tokens)
    }

    fn new_cold_session(&self, context_tokens: usize) -> Result<StoredSession<B>, Box<dyn Error>> {
        Ok(StoredSession {
            generation: GenerationSession::new(),
            metadata_allocation: Some(self.allocate_resident_metadata(context_tokens)?),
            reference_lease: None,
            last_used: 0,
        })
    }

    fn restore_persisted_continuation(
        &mut self,
        session_id: &str,
        archive: StoredArchive,
        context_tokens: usize,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        let mut reference_lease = self.lease_session_reference(session_id)?;
        let (checkpoint, archive_reservation) =
            match self.load_persisted_checkpoint(session_id, &archive, reference_lease.as_ref()) {
                Ok(value) => value,
                Err(error) => {
                    return Err(restore_reference_after_load_error(
                        self,
                        session_id,
                        &mut reference_lease,
                        error,
                    ));
                }
            };
        let metadata_context = context_tokens.max(checkpoint.evaluated_tokens().len());
        let metadata_allocation = match self.allocate_resident_metadata(metadata_context) {
            Ok(allocation) => allocation,
            Err(error) => {
                drop(checkpoint);
                drop(archive_reservation);
                return Err(restore_reference_after_load_error(
                    self,
                    session_id,
                    &mut reference_lease,
                    error.into(),
                ));
            }
        };
        let mut generation = GenerationSession::new();
        generation.restore(checkpoint);
        drop(archive_reservation);
        Ok(StoredSession {
            generation,
            metadata_allocation: Some(metadata_allocation),
            reference_lease,
            last_used: archive.last_used,
        })
    }

    fn attach_resident_session(
        &mut self,
        session_id: &str,
        mut stored: StoredSession<B>,
        context_tokens: usize,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        match self
            .attach_session_reference(session_id, &mut stored)
            .and_then(|()| {
                self.ensure_resident_metadata(&mut stored, context_tokens)
                    .map_err(Into::into)
            }) {
            Ok(()) => Ok(stored),
            Err(error) => {
                self.sessions.insert(session_id.to_owned(), stored);
                Err(error)
            }
        }
    }

    fn lease_prefix_reuse(
        &mut self,
        source_id: &str,
        context_tokens: usize,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        self.require_session_cleanup_complete(
            source_id,
            "the prefix reuse source cleanup is pending",
        )?;
        let source_context = self
            .sessions
            .get(source_id)
            .ok_or_else(|| invalid_data("the prefix reuse source is no longer resident"))?
            .generation
            .evaluated_tokens()
            .len();
        let metadata_allocation =
            self.allocate_resident_metadata(context_tokens.max(source_context))?;
        let generation = {
            let source = self
                .sessions
                .get(source_id)
                .ok_or_else(|| invalid_data("the prefix reuse source is no longer resident"))?;
            self.runtime.reuse_prefix_session(&source.generation)?
        };
        Ok(StoredSession {
            generation,
            metadata_allocation: Some(metadata_allocation),
            reference_lease: None,
            last_used: 0,
        })
    }

    fn lease_fork_session(
        &mut self,
        parent_id: &str,
        context_tokens: usize,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        self.wake_fork_parent(parent_id, context_tokens)?;
        let (generation, metadata_allocation) = match self.sessions.get(parent_id) {
            Some(parent) => {
                let parent_context = parent.generation.evaluated_tokens().len();
                let metadata_allocation =
                    self.allocate_resident_metadata(context_tokens.max(parent_context))?;
                let generation = self.runtime.fork_session(&parent.generation)?;
                (generation, metadata_allocation)
            }
            None => {
                let parent = self
                    .persisted_metadata(parent_id)?
                    .ok_or_else(|| invalid_data("the fork parent session does not exist"))?;
                self.restore_persisted_fork(parent_id, &parent, context_tokens)?
            }
        };
        Ok(StoredSession {
            generation,
            metadata_allocation: Some(metadata_allocation),
            reference_lease: None,
            last_used: 0,
        })
    }

    fn lease_available_fork_session(
        &mut self,
        parent_id: &str,
        context_tokens: usize,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        self.require_session_cleanup_complete(
            parent_id,
            "the fork parent session cleanup is pending",
        )?;
        self.lease_fork_session(parent_id, context_tokens)
    }

    fn restore_persisted_fork(
        &mut self,
        parent_id: &str,
        parent: &StoredArchive,
        context_tokens: usize,
    ) -> Result<(GenerationSession<B>, ResidentMetadataLease), Box<dyn Error>> {
        let mut reference_lease = self.lease_session_reference(parent_id)?;
        let (checkpoint, archive_reservation) =
            match self.load_persisted_checkpoint(parent_id, parent, reference_lease.as_ref()) {
                Ok(value) => value,
                Err(error) => {
                    return Err(restore_reference_after_load_error(
                        self,
                        parent_id,
                        &mut reference_lease,
                        error,
                    ));
                }
            };
        if let Err(error) = restore_loaded_reference(&mut reference_lease) {
            drop(checkpoint);
            drop(archive_reservation);
            self.retain_reference_recovery(parent_id, &mut reference_lease)?;
            return Err(error);
        }
        let metadata_context = context_tokens.max(checkpoint.evaluated_tokens().len());
        let metadata = self.allocate_resident_metadata(metadata_context)?;
        let mut generation = GenerationSession::<B>::new();
        generation.restore(checkpoint);
        drop(archive_reservation);
        Ok((generation, metadata))
    }

    fn wake_fork_parent(
        &mut self,
        parent_id: &str,
        context_tokens: usize,
    ) -> Result<(), Box<dyn Error>> {
        if self.hibernated.contains_key(parent_id) {
            let parent = self.wake_hibernated_session(parent_id, context_tokens)?;
            self.clock = self.clock.saturating_add(1);
            self.insert_session(
                parent_id.to_owned(),
                StoredSession {
                    generation: parent.generation,
                    metadata_allocation: parent.metadata_allocation,
                    reference_lease: parent.reference_lease,
                    last_used: self.clock,
                },
            )?;
        }
        Ok(())
    }

    fn wake_hibernated_session(
        &mut self,
        session_id: &str,
        context_tokens: usize,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        if !self.hibernated.contains_key(session_id) {
            return Err(invalid_data("the hibernated session does not exist").into());
        }
        let reservation = self.wake_metadata_reservation(session_id, context_tokens)?;
        let reference_lease = self.take_hibernation_reference(session_id)?;
        match self.wake_generation(session_id) {
            Ok((generation, last_used)) => self.finish_wake(
                session_id,
                generation,
                last_used,
                reservation,
                reference_lease,
            ),
            Err(error) => self.finish_wake_error(session_id, error, reservation, reference_lease),
        }
    }

    fn take_hibernation_reference(
        &mut self,
        session_id: &str,
    ) -> Result<Option<SessionReferenceLease>, Box<dyn Error>> {
        if let Some(stored) = self.hibernated.get_mut(session_id) {
            if stored.reference_lease.is_some() {
                return Ok(stored.reference_lease.take());
            }
        }
        self.lease_session_reference(session_id)
    }

    fn retain_hibernation_reference(
        &mut self,
        session_id: &str,
        reference_lease: Option<SessionReferenceLease>,
    ) {
        if let Some(reference_lease) = reference_lease {
            if let Some(stored) = self.hibernated.get_mut(session_id) {
                stored.reference_lease = Some(reference_lease);
            }
        }
    }

    fn wake_metadata_reservation(
        &self,
        session_id: &str,
        context_tokens: usize,
    ) -> Result<MemoryReservation, Box<dyn Error>> {
        let stored = self.hibernated.get(session_id).ok_or_else(|| {
            RuntimeError::Backend(BackendError::operation(
                "wake hibernated session",
                "the hibernated session disappeared",
            ))
        })?;
        let bytes = self
            .runtime
            .wake_metadata_bytes(&stored.generation)?
            .max(self.runtime.resident_metadata_bound(context_tokens)?);
        self.memory.reserve_host(bytes).map_err(Into::into)
    }

    fn lease_session_reference(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionReferenceLease>, Box<dyn Error>> {
        let Some(store) = self.session_store.as_ref() else {
            return Ok(None);
        };
        store
            .lease_reference(session_id)
            .map_err(session_reference_lease_error)
    }

    fn wake_generation(
        &mut self,
        session_id: &str,
    ) -> Result<(GenerationSession<B>, u64), RuntimeError> {
        let stored = self.hibernated.get(session_id).ok_or_else(|| {
            RuntimeError::Backend(BackendError::operation(
                "wake hibernated session",
                "the hibernated session disappeared",
            ))
        })?;
        let last_used = stored.last_used;
        let generation = self.runtime.wake_session(&stored.generation)?;
        Ok((generation, last_used))
    }

    fn finish_wake(
        &mut self,
        session_id: &str,
        generation: GenerationSession<B>,
        last_used: u64,
        reservation: MemoryReservation,
        mut reference_lease: Option<SessionReferenceLease>,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        if let Some(lease) = reference_lease.as_mut() {
            if let Err(error) = lease.restore() {
                self.retain_hibernation_reference(session_id, reference_lease);
                drop(generation);
                drop(reservation);
                return Err(session_persistence_error(
                    SessionFailureEffect::Unchanged,
                    "restore persisted session reference",
                    error,
                )
                .into());
            }
        }
        let metadata_allocation = reservation
            .commit()
            .map_err(|error| RuntimeError::Backend(BackendError::Memory(error)))?;
        let stored = self
            .hibernated
            .remove(session_id)
            .ok_or_else(|| invalid_data("the hibernated session disappeared"))?;
        release_hibernation(stored);
        Ok(StoredSession {
            generation,
            metadata_allocation: Some(ResidentMetadataLease::single(metadata_allocation)),
            reference_lease: None,
            last_used,
        })
    }

    fn finish_wake_error(
        &mut self,
        session_id: &str,
        error: RuntimeError,
        reservation: MemoryReservation,
        mut reference_lease: Option<SessionReferenceLease>,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        drop(reservation);
        if error.session_failure_effect() == Some(SessionFailureEffect::Quarantine) {
            self.discard_leased_session(session_id, reference_lease)?;
            return Err(error.into());
        }
        if let Err(persistence_error) = restore_loaded_reference(&mut reference_lease) {
            self.retain_hibernation_reference(session_id, reference_lease);
            return Err(persistence_error);
        }
        Err(error.into())
    }

    fn load_persisted_checkpoint(
        &self,
        session_id: &str,
        stored: &StoredArchive,
        reference_lease: Option<&SessionReferenceLease>,
    ) -> Result<(leone::GenerationCheckpoint, PersistedArchiveReservation), Box<dyn Error>> {
        let archive_reservation = self.reserve_persisted_archive(stored)?;
        let archive = self.load_persisted_archive(
            session_id,
            stored,
            reference_lease,
            archive_reservation.serialized_bytes,
        )?;
        let checkpoint = archive.checkpoint()?;
        Ok((checkpoint, archive_reservation))
    }

    fn finish_scheduled_chat(
        &mut self,
        task: ChatTask<B>,
        status: RequestStatus,
    ) -> Result<(), Box<dyn Error>> {
        if status == RequestStatus::DeadlineExpired || task.deadline_expired {
            return self.finish_deadline_task(task);
        }
        if task.disconnected {
            return self.finish_disconnected_task(task);
        }
        if task.failure.is_some() {
            return self.finish_failed_task(task);
        }
        self.finish_generated_task(task, status)
    }

    fn finish_deadline_task(&mut self, mut task: ChatTask<B>) -> Result<(), Box<dyn Error>> {
        if let Err(error) = self.discard_stored_session(&task.session_id, &mut task.stored) {
            eprintln!(
                "request deadline session cleanup failed for {}: {error}",
                task.session_id
            );
        }
        if !task.stream.begin_terminal_response() {
            return Ok(());
        }
        write_stream_error(
            &mut task.stream,
            task.headers_written,
            408,
            "the request deadline expired",
            task.cors_origin,
        )
        .map_err(Into::into)
    }

    fn finish_disconnected_task(&mut self, mut task: ChatTask<B>) -> Result<(), Box<dyn Error>> {
        self.discard_stored_session(&task.session_id, &mut task.stored)?;
        Ok(())
    }

    fn finish_failed_task(&mut self, mut task: ChatTask<B>) -> Result<(), Box<dyn Error>> {
        let error = task.failure.take().expect("failure was checked");
        if task.failure_effect == Some(SessionFailureEffect::Unchanged) {
            if let Err(retain_error) =
                self.retain_failed_session(task.session_id.clone(), task.stored)
            {
                eprintln!(
                    "failed request session retention failed for {}: {retain_error}",
                    task.session_id
                );
            }
        } else if let Err(cleanup_error) =
            self.discard_stored_session(&task.session_id, &mut task.stored)
        {
            eprintln!(
                "failed request session cleanup failed for {}: {cleanup_error}",
                task.session_id
            );
        }
        if !task.stream.begin_terminal_response() {
            return Ok(());
        }
        write_stream_error(
            &mut task.stream,
            task.headers_written,
            500,
            &error,
            task.cors_origin,
        )
        .map_err(Into::into)
    }

    fn finish_generated_task(
        &mut self,
        task: ChatTask<B>,
        status: RequestStatus,
    ) -> Result<(), Box<dyn Error>> {
        let mut task = task;
        if let Err(error) = flush_task_pending(&mut task, &self.model_id) {
            eprintln!("response stream flush failed: {error}");
            task.disconnected = true;
            return self.finish_disconnected_task(task);
        }
        self.finish_generated_task_after_flush(task, status)
    }

    fn finish_generated_task_after_flush(
        &mut self,
        task: ChatTask<B>,
        status: RequestStatus,
    ) -> Result<(), Box<dyn Error>> {
        let cancelled = status != RequestStatus::Finished;
        let ChatTask {
            request,
            prompt_tokens,
            tokens,
            options,
            session_id,
            stored,
            mut stream,
            request_sha256,
            created,
            completion_id,
            cors_origin,
            mut streamed,
            stop_hit,
            tool_header_stop,
            headers_written,
            eos,
            prefill_chunks,
            prefill_tokens,
            request_replay,
            decode_quanta,
            phase_trace,
            ..
        } = task;
        let mut response_started = headers_written;
        let generated = ChatGeneration {
            prompt_tokens,
            tokens,
            cancelled,
            stop_hit,
            tool_header_stop,
            eos,
            prefill_chunks,
            prefill_tokens,
            request_replay,
            decode_quanta,
            phase_trace,
        };
        let mut stored = Some(stored);
        if stream.is_cancelled() || !stream.begin_terminal_response() {
            if let Err(error) = self.discard_stored_session(
                &session_id,
                stored.as_mut().expect("response session is present"),
            ) {
                eprintln!("response session cleanup failed for {session_id}: {error}");
            }
            return Ok(());
        }
        let response_session_id = session_id.clone();
        let model_id = self.model_id.clone();
        let error_origin = cors_origin.clone();
        let response = {
            let mut response_stream = ResponseWriter::new(&mut stream, &mut response_started);
            finish_chat_response(
                self,
                &mut response_stream,
                ChatResponseContext {
                    request: &request,
                    session_id,
                    completion_id,
                    created,
                    request_sha256,
                    seed: options.seed,
                    model_id,
                    cors_origin,
                },
                &mut streamed,
                &mut stored,
                generated,
            )
        };
        self.finish_generated_response(
            response,
            ResponseWireState {
                request_stream: request.stream,
                started: response_started,
            },
            &response_session_id,
            &mut stored,
            &mut stream,
            error_origin,
        )
    }

    fn finish_generated_response(
        &mut self,
        response: Result<(), Box<dyn Error>>,
        wire: ResponseWireState,
        response_session_id: &str,
        stored: &mut Option<StoredSession<B>>,
        stream: &mut OutputSink,
        error_origin: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        match response {
            Ok(()) => Ok(()),
            Err(error) => self.finish_generated_response_error(
                error,
                wire,
                response_session_id,
                stored,
                stream,
                error_origin,
            ),
        }
    }

    fn finish_generated_response_error(
        &mut self,
        error: Box<dyn Error>,
        wire: ResponseWireState,
        response_session_id: &str,
        stored: &mut Option<StoredSession<B>>,
        stream: &mut OutputSink,
        error_origin: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        eprintln!("response finalization failed: {error}");
        self.resolve_response_session_error(response_session_id, stored, error.as_ref());
        let request_error = error.downcast_ref::<InvalidGeneratedResponse>().is_some();
        let status = if request_error { 400 } else { 500 };
        if !wire.request_stream && wire.started {
            stream.cancel();
            return Err(error);
        }
        let message = if request_error {
            error.to_string()
        } else {
            RESPONSE_FINALIZATION_ERROR.to_owned()
        };
        let started = wire.started;
        let mut response_started = wire.started;
        let mut response_stream = ResponseWriter::new(stream, &mut response_started);
        write_stream_error(
            &mut response_stream,
            started,
            status,
            &message,
            error_origin,
        )
        .map_err(Into::into)
    }

    fn resolve_response_session_error(
        &mut self,
        session_id: &str,
        stored: &mut Option<StoredSession<B>>,
        error: &(dyn Error + 'static),
    ) {
        if preserves_session_on_failure(error) {
            if let Some(stored) = stored.take() {
                if let Err(retain_error) = self.retain_failed_session(session_id.to_owned(), stored)
                {
                    eprintln!("response session retention failed for {session_id}: {retain_error}");
                }
            }
        } else if let Some(stored) = stored.as_mut() {
            if let Err(cleanup_error) = self.discard_stored_session(session_id, stored) {
                eprintln!("response session cleanup failed for {session_id}: {cleanup_error}");
            }
        } else {
            discard_response_session(self, session_id);
        }
    }

    fn select_session(
        &self,
        prompt: &[u32],
        options: &GenerateOptions,
    ) -> Result<Option<String>, RuntimeError> {
        let best = self.sessions.iter().try_fold(
            None,
            |best: Option<(&String, usize, u64)>,
             (id, session)|
             -> Result<Option<(&String, usize, u64)>, RuntimeError> {
                if session
                    .reference_lease
                    .as_ref()
                    .is_some_and(SessionReferenceLease::discard_pending)
                {
                    return Ok(best);
                }
                let reused =
                    self.runtime
                        .reusable_prefill_tokens(&session.generation, prompt, options)?;
                if reused == 0 {
                    return Ok(best);
                }
                let candidate = (reused, session.last_used, id.as_str());
                let replace = best.is_none_or(|(best_id, best_reused, best_last_used)| {
                    prefix_candidate_is_better(
                        candidate,
                        (best_reused, best_last_used, best_id.as_str()),
                    )
                });
                Ok(if replace {
                    Some((id, reused, session.last_used))
                } else {
                    best
                })
            },
        )?;
        Ok(best.map(|(id, _, _)| id.clone()))
    }

    fn insert_session(
        &mut self,
        id: String,
        mut session: StoredSession<B>,
    ) -> Result<(), RuntimeError> {
        let context_tokens = session.generation.evaluated_tokens().len();
        let metadata_result = self.ensure_resident_metadata(&mut session, context_tokens);
        self.sessions.insert(id, session);
        metadata_result?;
        self.hibernate_until_capacity()?;
        self.evict_hibernated_until_limit()?;
        Ok(())
    }

    fn retain_failed_session(
        &mut self,
        id: String,
        session: StoredSession<B>,
    ) -> Result<(), RuntimeError> {
        self.retain_failed_session_with_sync(id, session, &sync_directory)
    }

    fn retain_failed_session_with_sync<F>(
        &mut self,
        id: String,
        mut session: StoredSession<B>,
        sync: &F,
    ) -> Result<(), RuntimeError>
    where
        F: Fn(&Path) -> io::Result<()>,
    {
        self.persisted.remove(&id);
        if let Err(error) =
            restore_hibernation_reference_with_sync(&mut session.reference_lease, sync)
        {
            self.sessions.insert(id, session);
            return Err(error);
        }
        self.clock = self.clock.saturating_add(1);
        session.last_used = self.clock;
        self.insert_session(id, session)
    }

    fn hibernate_until_capacity(&mut self) -> Result<(), RuntimeError> {
        while self.sessions.len() > self.max_sessions {
            match self.hibernate_oldest() {
                Ok(()) => {}
                Err(error) if memory_budget_exceeded(&error) => {
                    if !self.discard_oldest_hibernated()? {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn evict_hibernated_until_limit(&mut self) -> Result<(), RuntimeError> {
        while self.hibernated.len() > self.max_hibernated_sessions {
            if !self.discard_oldest_hibernated()? {
                break;
            }
        }
        Ok(())
    }

    fn discard_oldest_hibernated(&mut self) -> Result<bool, RuntimeError> {
        let Some(oldest) = self
            .hibernated
            .iter()
            .min_by_key(|(_, session)| session.last_used)
            .map(|(id, _)| id.clone())
        else {
            return Ok(false);
        };
        if self.retry_pending_session_cleanup(&oldest)? {
            return Ok(true);
        }
        if let Some(stored) = self.hibernated.get_mut(&oldest) {
            restore_hibernation_reference(&mut stored.reference_lease)?;
        }
        let removed = self.hibernated.remove(&oldest);
        if let Some(removed) = removed {
            release_hibernation(removed);
        }
        Ok(true)
    }

    fn hibernate_oldest(&mut self) -> Result<(), RuntimeError> {
        let Some(oldest) = self
            .sessions
            .iter()
            .min_by_key(|(_, session)| session.last_used)
            .map(|(id, _)| id.clone())
        else {
            return Ok(());
        };
        if self.retry_pending_session_cleanup(&oldest)? {
            return Ok(());
        }
        if self.discard_idle_session(&oldest)? {
            return Ok(());
        }
        let Some(stored) = self.sessions.remove(&oldest) else {
            return Ok(());
        };
        self.hibernate_stored(oldest, stored)
    }

    fn retry_pending_session_cleanup(&mut self, session_id: &str) -> Result<bool, RuntimeError> {
        if !self.session_discard_pending(session_id) {
            return Ok(false);
        }
        self.discard_session(session_id).map_err(|error| {
            RuntimeError::Backend(BackendError::operation(
                "retry pending session cleanup",
                error,
            ))
        })?;
        Ok(true)
    }

    fn hibernate_stored(
        &mut self,
        oldest: String,
        mut stored: StoredSession<B>,
    ) -> Result<(), RuntimeError> {
        let (reservation, metadata_bytes) = match self.prepare_hibernation(&oldest, &mut stored) {
            Ok(reservation) => reservation,
            Err(error) => {
                self.sessions.insert(oldest, stored);
                return Err(error);
            }
        };
        let captured = self.capture_hibernated(oldest.clone(), stored, reservation);
        let (generation, host_allocation, last_used, mut reference_lease) = match captured {
            Ok(captured) => captured,
            Err(error) => {
                self.restore_failed_hibernation_reference(&oldest, &error)?;
                return Err(error);
            }
        };
        self.store_hibernated(
            oldest,
            generation,
            host_allocation,
            metadata_bytes,
            last_used,
            &mut reference_lease,
        )?;
        Ok(())
    }

    fn store_hibernated(
        &mut self,
        session_id: String,
        generation: HibernatedSession,
        host_allocation: MemoryAllocation,
        metadata_bytes: u64,
        last_used: u64,
        reference_lease: &mut Option<SessionReferenceLease>,
    ) -> Result<(), RuntimeError> {
        if let Err(error) = restore_hibernation_reference(reference_lease) {
            self.hibernated.insert(
                session_id,
                StoredHibernation {
                    generation,
                    host_allocation,
                    metadata_bytes,
                    reference_lease: reference_lease.take(),
                    last_used,
                },
            );
            return Err(error);
        }
        self.hibernated.insert(
            session_id,
            StoredHibernation {
                generation,
                host_allocation,
                metadata_bytes,
                reference_lease: None,
                last_used,
            },
        );
        Ok(())
    }

    fn prepare_hibernation(
        &self,
        session_id: &str,
        stored: &mut StoredSession<B>,
    ) -> Result<(MemoryReservation, u64), RuntimeError> {
        let result = self.hibernation_reservation(&stored.generation)?;
        if stored.reference_lease.is_none() {
            stored.reference_lease = self.lease_hibernation_reference(session_id)?;
        }
        Ok(result)
    }

    fn restore_failed_hibernation_reference(
        &mut self,
        session_id: &str,
        error: &RuntimeError,
    ) -> Result<(), RuntimeError> {
        if error.session_failure_effect() == Some(SessionFailureEffect::Quarantine) {
            return Ok(());
        }
        if let Some(stored) = self.sessions.get_mut(session_id) {
            restore_hibernation_reference(&mut stored.reference_lease)?;
        }
        Ok(())
    }

    fn hibernation_reservation(
        &self,
        generation: &GenerationSession<B>,
    ) -> Result<(MemoryReservation, u64), RuntimeError> {
        let (bytes, metadata_bytes) = hibernation_host_bytes(&self.runtime, generation)?;
        let reservation = self.memory.reserve_host(bytes)?;
        Ok((reservation, metadata_bytes))
    }

    fn lease_hibernation_reference(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionReferenceLease>, RuntimeError> {
        let Some(store) = self.session_store.as_ref() else {
            return Ok(None);
        };
        store.lease_reference(session_id).map_err(|error| {
            session_persistence_error(
                SessionFailureEffect::Unchanged,
                "lease persisted session reference",
                error,
            )
        })
    }

    fn capture_hibernated(
        &mut self,
        oldest: String,
        mut stored: StoredSession<B>,
        reservation: MemoryReservation,
    ) -> Result<
        (
            HibernatedSession,
            MemoryAllocation,
            u64,
            Option<SessionReferenceLease>,
        ),
        RuntimeError,
    > {
        let generation = match self.runtime.hibernate_session(&mut stored.generation) {
            Ok(generation) => generation,
            Err(error) => {
                drop(reservation);
                if error.session_failure_effect() != Some(SessionFailureEffect::Quarantine) {
                    self.sessions.insert(oldest, stored);
                    return Err(error);
                }
                let reference_lease = stored.reference_lease.take();
                self.discard_leased_session(&oldest, reference_lease)
                    .map_err(|persistence_error| {
                        RuntimeError::Backend(BackendError::operation(
                            "quarantine persisted session",
                            persistence_error,
                        ))
                    })?;
                self.runtime.discard_session(&mut stored.generation)?;
                return Err(error);
            }
        };
        let host_allocation = reservation
            .commit()
            .map_err(|error| RuntimeError::Backend(BackendError::Memory(error)))?;
        let reference_lease = stored.reference_lease.take();
        Ok((
            generation,
            host_allocation,
            stored.last_used,
            reference_lease,
        ))
    }

    fn discard_idle_session(&mut self, session_id: &str) -> Result<bool, RuntimeError> {
        #[cfg(target_os = "macos")]
        {
            // A host copy uses the same unified pool on macOS, so replay needs a persisted checkpoint.
            let persisted = self.has_persisted_session(session_id).map_err(|error| {
                RuntimeError::SessionFailure {
                    effect: SessionFailureEffect::Unchanged,
                    source: Box::new(RuntimeError::Backend(BackendError::operation(
                        "inspect persisted session before retirement",
                        error,
                    ))),
                }
            })?;
            if !persisted {
                return Ok(false);
            }
            if self.sessions.contains_key(session_id) {
                self.retire_resident_session(session_id).map_err(|error| {
                    RuntimeError::Backend(BackendError::operation(
                        "retire idle session graph",
                        error,
                    ))
                })?;
            }
            Ok(self.sessions.remove(session_id).is_some())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = session_id;
            Ok(false)
        }
    }
}

fn hibernation_host_bytes<B: Backend>(
    runtime: &Runtime<B>,
    source: &GenerationSession<B>,
) -> Result<(u64, u64), RuntimeError> {
    let payload_bytes = runtime.hibernation_bytes(source)?;
    let metadata_bytes = runtime.hibernation_metadata_bytes(source)?;
    let total_bytes = payload_bytes
        .checked_add(metadata_bytes)
        .ok_or(RuntimeError::SizeOverflow)?;
    Ok((total_bytes, metadata_bytes))
}

fn session_persistence_error(
    effect: SessionFailureEffect,
    operation: &'static str,
    error: io::Error,
) -> RuntimeError {
    RuntimeError::SessionFailure {
        effect,
        source: Box::new(RuntimeError::Backend(BackendError::operation(
            operation, error,
        ))),
    }
}

fn session_reference_lease_error(error: io::Error) -> Box<dyn Error> {
    if error
        .get_ref()
        .is_some_and(|source| source.is::<SessionRecoveryCapacity>())
    {
        return Box::new(SessionRecoveryCapacity);
    }
    if io_session_capacity_error(&error) {
        return Box::new(SessionMemoryCapacity);
    }
    Box::new(session_persistence_error(
        SessionFailureEffect::Unchanged,
        "lease persisted session reference",
        error,
    ))
}

fn unchanged_runtime_error(error: RuntimeError) -> RuntimeError {
    RuntimeError::SessionFailure {
        effect: SessionFailureEffect::Unchanged,
        source: Box::new(error),
    }
}

fn restore_hibernation_reference(
    reference_lease: &mut Option<SessionReferenceLease>,
) -> Result<(), RuntimeError> {
    restore_hibernation_reference_with_sync(reference_lease, &sync_directory)
}

fn restore_hibernation_reference_with_sync<F>(
    reference_lease: &mut Option<SessionReferenceLease>,
    sync: &F,
) -> Result<(), RuntimeError>
where
    F: Fn(&Path) -> io::Result<()>,
{
    let Some(lease) = reference_lease.as_mut() else {
        return Ok(());
    };
    lease.restore_with_sync(sync).map_err(|error| {
        session_persistence_error(
            SessionFailureEffect::Unchanged,
            "restore persisted session reference",
            error,
        )
    })?;
    reference_lease.take();
    Ok(())
}

fn restore_loaded_reference(
    reference_lease: &mut Option<SessionReferenceLease>,
) -> Result<(), Box<dyn Error>> {
    if let Some(lease) = reference_lease.as_mut() {
        lease.restore().map_err(|error| {
            session_persistence_error(
                SessionFailureEffect::Unchanged,
                "restore persisted session reference",
                error,
            )
        })?;
    }
    reference_lease.take();
    Ok(())
}

fn restore_reference_after_load_error<B: Backend>(
    server: &Server<B>,
    session_id: &str,
    reference_lease: &mut Option<SessionReferenceLease>,
    error: Box<dyn Error>,
) -> Box<dyn Error> {
    if let Err(persistence_error) = restore_loaded_reference(reference_lease) {
        if let Err(retain_error) = server.retain_reference_recovery(session_id, reference_lease) {
            return retain_error;
        }
        return persistence_error;
    }
    error
}

fn memory_budget_exceeded(error: &RuntimeError) -> bool {
    matches!(
        error,
        RuntimeError::Backend(BackendError::Memory(MemoryError::BudgetExceeded { .. }))
    )
}

fn release_hibernation(stored: StoredHibernation) {
    debug_assert_eq!(
        stored
            .generation
            .record()
            .host_bytes
            .checked_add(stored.metadata_bytes),
        Some(stored.host_allocation.bytes())
    );
    drop(stored);
}

fn discard_response_session<B: Backend>(server: &mut Server<B>, session_id: &str) {
    if let Err(error) = server.discard_session(session_id) {
        eprintln!("response session cleanup failed for {session_id}: {error}");
    }
}

fn preserves_session_on_failure(error: &(dyn Error + 'static)) -> bool {
    error
        .downcast_ref::<RuntimeError>()
        .and_then(RuntimeError::session_failure_effect)
        == Some(SessionFailureEffect::Unchanged)
}

fn validate_chat_request(request: &ChatRequest, model_id: &str) -> Result<(), io::Error> {
    validate_chat_identity(request, model_id)?;
    validate_messages(&request.messages)?;
    validate_chat_tools(request)?;
    validate_chat_sampling(request)?;
    validate_stream_options(request)?;
    sampler(request)?;
    penalties(request)?;
    validate_chat_stop(request)?;
    Ok(())
}

fn validate_chat_stop(request: &ChatRequest) -> Result<(), io::Error> {
    let stop_sequences = request_stop_sequences(request)?;
    if request.tools.is_some() && !stop_sequences.is_empty() {
        return Err(invalid_data("stop cannot be combined with tools"));
    }
    Ok(())
}

fn effective_tool_mode(request: &ChatRequest) -> bool {
    request.tools.as_deref().is_some_and(|tools| {
        !tools.is_empty() && request.tool_choice.as_ref().and_then(Value::as_str) != Some("none")
    })
}

fn validate_messages(messages: &[Message]) -> Result<(), io::Error> {
    let mut pending_calls = BTreeMap::new();
    let mut seen_call_ids = std::collections::BTreeSet::new();
    for message in messages {
        let role = validated_message_role(&message.role)?;
        if role == "tool" {
            validate_tool_message_order(message, &mut pending_calls)?;
        } else {
            if !pending_calls.is_empty() {
                return Err(invalid_data(
                    "assistant tool calls require matching tool messages before the next message",
                ));
            }
            validate_message(message)?;
            if role == "assistant" {
                register_assistant_tool_calls(message, &mut pending_calls, &mut seen_call_ids)?;
            }
        }
    }
    if !pending_calls.is_empty() {
        return Err(invalid_data(
            "assistant tool calls require matching tool messages",
        ));
    }
    Ok(())
}

fn register_assistant_tool_calls(
    message: &Message,
    pending_calls: &mut BTreeMap<String, String>,
    seen_call_ids: &mut std::collections::BTreeSet<String>,
) -> Result<(), io::Error> {
    let Some(calls) = message.tool_calls.as_ref() else {
        return Ok(());
    };
    for call in calls {
        if !seen_call_ids.insert(call.id.clone()) {
            return Err(invalid_data("assistant tool call ids must be unique"));
        }
        pending_calls.insert(call.id.clone(), call.function.name.clone());
    }
    Ok(())
}

fn validate_tool_message_order(
    message: &Message,
    pending_calls: &mut BTreeMap<String, String>,
) -> Result<(), io::Error> {
    validate_message(message)?;
    let id = message
        .tool_call_id
        .as_deref()
        .ok_or_else(|| invalid_data("a tool message requires tool_call_id"))?;
    let Some(expected_name) = pending_calls.remove(id) else {
        return Err(invalid_data(
            "tool message tool_call_id has no matching call",
        ));
    };
    if message
        .name
        .as_deref()
        .is_some_and(|name| name != expected_name)
    {
        return Err(invalid_data("tool message name does not match tool call"));
    }
    Ok(())
}

fn validate_message_response_fields(message: &Message) -> Result<(), io::Error> {
    for (field, value) in [
        ("refusal", message.refusal.as_ref()),
        ("annotations", message.annotations.as_ref()),
        ("audio", message.audio.as_ref()),
        ("function_call", message.function_call.as_ref()),
    ] {
        if value.is_some_and(|value| !value.is_null()) {
            return Err(invalid_data(format!(
                "message.{field} is response-only and must be null"
            )));
        }
    }
    Ok(())
}

fn validate_message(message: &Message) -> Result<(), io::Error> {
    let role = validated_message_role(&message.role)?;
    validate_message_response_fields(message)?;
    validate_message_fields(message, role)?;
    match role {
        "assistant" => validate_assistant_message(message),
        "tool" => validate_tool_message(message),
        _ => Ok(()),
    }
}

fn validated_message_role(role: &str) -> Result<&'static str, io::Error> {
    match role {
        "system" | "developer" => Ok("system"),
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        "tool" => Ok("tool"),
        _ => Err(invalid_data("message role is not supported")),
    }
}

fn validate_message_fields(message: &Message, role: &str) -> Result<(), io::Error> {
    if message.name.is_some() && role != "tool" {
        return Err(invalid_data(
            "message name is supported only for tool messages",
        ));
    }
    if message.tool_calls.is_some() && role != "assistant" {
        return Err(invalid_data(
            "tool_calls are supported only for assistant messages",
        ));
    }
    if message.tool_call_id.is_some() && role != "tool" {
        return Err(invalid_data(
            "tool_call_id is supported only for tool messages",
        ));
    }
    Ok(())
}

fn validate_assistant_message(message: &Message) -> Result<(), io::Error> {
    let Some(calls) = &message.tool_calls else {
        return if message.content.is_null() {
            Err(invalid_data(
                "assistant content is required without tool_calls",
            ))
        } else {
            Ok(())
        };
    };
    if calls.is_empty() {
        return Err(invalid_data("assistant tool_calls must not be empty"));
    }
    for call in calls {
        validate_request_tool_call(call)?;
    }
    Ok(())
}

fn validate_tool_message(message: &Message) -> Result<(), io::Error> {
    if message.tool_call_id.as_deref().is_none_or(str::is_empty) {
        return Err(invalid_data("a tool message requires tool_call_id"));
    }
    Ok(())
}

fn validate_request_tool_call(call: &RequestToolCall) -> Result<(), io::Error> {
    if call.id.is_empty() {
        return Err(invalid_data("assistant tool calls require id"));
    }
    if call.kind != "function" {
        return Err(invalid_data("assistant tool calls must be functions"));
    }
    validate_function_name(&call.function.name)?;
    parse_request_tool_arguments(&call.function.arguments)
        .map(|_| ())
        .map_err(|error| invalid_data(error.to_string()))
}

fn validate_function_name(name: &str) -> Result<(), io::Error> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(invalid_data(
            "tool function names must use 1 to 64 ASCII letters, digits, underscores, or hyphens",
        ));
    }
    Ok(())
}

fn validate_stream_options(request: &ChatRequest) -> Result<(), io::Error> {
    if let Some(options) = &request.stream_options {
        if !request.stream {
            return Err(invalid_data("stream_options requires stream=true"));
        }
        let _ = options.include_usage;
    }
    Ok(())
}

fn validate_chat_identity(request: &ChatRequest, model_id: &str) -> Result<(), io::Error> {
    if request.model != model_id && request.model != "leone" {
        return Err(invalid_data(format!(
            "model {:?} is not served; use {model_id:?}",
            request.model
        )));
    }
    if request.messages.is_empty() {
        return Err(invalid_data("messages must not be empty"));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(invalid_data("the stable subset supports only n=1"));
    }
    Ok(())
}

fn validate_chat_tools(request: &ChatRequest) -> Result<(), io::Error> {
    validate_tools(request)?;
    let response_constraint = response_constraint(request.response_format.as_ref())?;
    if response_constraint == Some(OutputConstraint::JsonObject) && effective_tool_mode(request) {
        return Err(invalid_data(
            "response_format json_object cannot be combined with function tools",
        ));
    }
    if request.logprobs.unwrap_or(false) || request.top_logprobs.is_some() {
        return Err(invalid_data("logprobs are not in the stable subset"));
    }
    Ok(())
}

fn validate_chat_sampling(request: &ChatRequest) -> Result<(), io::Error> {
    if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
        return Err(invalid_data(
            "set max_tokens or max_completion_tokens, not both",
        ));
    }
    if request.mirostat_tau.is_some() != request.mirostat_eta.is_some() {
        return Err(invalid_data(
            "mirostat_tau and mirostat_eta must be set together",
        ));
    }
    if request.mirostat_tau.is_some() && request.draft_tokens.is_some() {
        return Err(invalid_data(
            "Mirostat cannot be combined with draft_tokens",
        ));
    }
    if request.mirostat_tau.is_some() && request.adaptive_speculation == Some(true) {
        return Err(invalid_data(
            "Mirostat cannot be combined with adaptive_speculation",
        ));
    }
    Ok(())
}

fn validate_tools(request: &ChatRequest) -> Result<(), io::Error> {
    let Some(tools) = request.tools.as_deref() else {
        if request.tool_choice.is_some() {
            return Err(invalid_data("tool_choice requires tools"));
        }
        return Ok(());
    };
    let names = validate_tool_definitions(tools)?;
    validate_tool_choice(request.tool_choice.as_ref(), &names)
}

fn validate_tool_definitions(
    tools: &[ToolDefinition],
) -> Result<std::collections::BTreeSet<&str>, io::Error> {
    let mut names = std::collections::BTreeSet::new();
    for tool in tools {
        if tool.kind != "function" {
            return Err(invalid_data("every tool type must be function"));
        }
        validate_function_name(&tool.function.name)?;
        if !names.insert(tool.function.name.as_str()) {
            return Err(invalid_data(
                "tool function names must be nonempty and unique",
            ));
        }
        if tool
            .function
            .parameters
            .as_ref()
            .is_some_and(|value| !value.is_object())
        {
            return Err(invalid_data(
                "tool function parameters must be one JSON object",
            ));
        }
        if tool.function.strict == Some(true) {
            return Err(invalid_data(
                "tool function strict=true is unsupported by the stable subset",
            ));
        }
    }
    Ok(names)
}

fn validate_tool_choice(
    choice: Option<&Value>,
    names: &std::collections::BTreeSet<&str>,
) -> Result<(), io::Error> {
    let Some(choice) = choice else {
        return Ok(());
    };
    match choice {
        Value::String(value) => validate_tool_choice_string(value)?,
        Value::Object(object) => validate_named_tool_choice(object, names)?,
        _ => return Err(invalid_data("tool_choice is invalid")),
    }
    if names.is_empty() && tool_choice_needs_function(choice) {
        return Err(invalid_data(
            "tool_choice requires at least one function tool",
        ));
    }
    Ok(())
}

fn validate_tool_choice_string(value: &str) -> Result<(), io::Error> {
    if matches!(value, "auto" | "none" | "required") {
        Ok(())
    } else {
        Err(invalid_data("tool_choice is invalid"))
    }
}

fn validate_named_tool_choice(
    object: &serde_json::Map<String, Value>,
    names: &std::collections::BTreeSet<&str>,
) -> Result<(), io::Error> {
    reject_unknown_keys(object, &["type", "function"], "tool_choice")?;
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_data("named tool_choice requires function.name"))?;
    reject_unknown_keys(function, &["name"], "tool_choice.function")?;
    let name = object
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_data("named tool_choice requires function.name"))?;
    if object.get("type").and_then(Value::as_str) != Some("function") || !names.contains(name) {
        return Err(invalid_data(
            "named tool_choice does not match a function tool",
        ));
    }
    Ok(())
}

fn reject_unknown_keys(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    field: &str,
) -> Result<(), io::Error> {
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid_data(format!("{field} contains an unknown field")));
    }
    Ok(())
}

fn tool_choice_needs_function(choice: &Value) -> bool {
    matches!(choice, Value::String(value) if value == "required")
        || matches!(choice, Value::Object(_))
}

fn response_constraint(value: Option<&Value>) -> Result<Option<OutputConstraint>, io::Error> {
    let Some(value) = value else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid_data("response_format must be one object"))?;
    reject_unknown_keys(object, &["type"], "response_format")?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_data("response_format.type must be one string"))?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(OutputConstraint::JsonObject)),
        _ => Err(invalid_data(format!(
            "response_format.type {kind:?} is not supported; use text or json_object"
        ))),
    }
}

fn sampler(request: &ChatRequest) -> Result<Sampler, io::Error> {
    let temperature = request.temperature.unwrap_or(1.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(invalid_data("temperature must be finite and in [0, 2]"));
    }
    let mut sampler = Sampler {
        temperature: if temperature == 0.0 {
            Temperature::Greedy
        } else {
            Temperature::Scaled(temperature)
        },
        truncations: Vec::new(),
    };
    if let Some(value) = request.top_k {
        let count =
            NonZeroUsize::new(value).ok_or_else(|| invalid_data("top_k must be nonzero"))?;
        sampler.truncations.push(Truncation::TopK(count));
    }
    append_probability_truncations(&mut sampler, request);
    sampler
        .validate()
        .map_err(|error| invalid_data(error.to_string()))?;
    Ok(sampler)
}

fn append_probability_truncations(sampler: &mut Sampler, request: &ChatRequest) {
    let stages = [
        request.top_p.map(Truncation::TopP),
        request.min_p.map(Truncation::MinP),
        request.top_a.map(Truncation::TopA),
        request.tfs_z.map(Truncation::TailFree),
        request.typical_p.map(Truncation::Typical),
    ];
    sampler.truncations.extend(stages.into_iter().flatten());
}

fn penalties(request: &ChatRequest) -> Result<Penalties, io::Error> {
    let presence = request.presence_penalty.unwrap_or(0.0);
    let frequency = request.frequency_penalty.unwrap_or(0.0);
    let repetition = request.repetition_penalty.unwrap_or(1.0);
    validate_openai_penalty(presence, "presence_penalty")?;
    validate_openai_penalty(frequency, "frequency_penalty")?;
    if !repetition.is_finite() || repetition <= 0.0 {
        return Err(invalid_data(
            "repetition_penalty must be finite and greater than zero",
        ));
    }
    let mut penalties = Penalties::none();
    penalties.presence = presence;
    penalties.frequency = frequency;
    penalties.repetition = repetition;
    penalties.window = PenaltyWindow::from_optional(request.repetition_window);
    Ok(penalties)
}

fn validate_openai_penalty(value: f64, name: &str) -> Result<(), io::Error> {
    if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
        return Err(invalid_data(format!(
            "{name} must be finite and in [-2, 2]"
        )));
    }
    Ok(())
}

fn mirostat(request: &ChatRequest) -> Result<Option<MirostatConfig>, io::Error> {
    request
        .mirostat_tau
        .zip(request.mirostat_eta)
        .map(|(target, learning)| {
            MirostatConfig::new(target, learning).map_err(|error| invalid_data(error.to_string()))
        })
        .transpose()
}

fn speculation(request: &ChatRequest) -> Result<Speculation, io::Error> {
    if request.draft_tokens.is_some() && request.adaptive_speculation == Some(true) {
        return Err(invalid_data(
            "draft_tokens and adaptive_speculation cannot be combined",
        ));
    }
    if request.mirostat_tau.is_some() {
        return Ok(Speculation::Disabled);
    }
    let Some(width) = request.draft_tokens else {
        if request.adaptive_speculation != Some(true) {
            return Ok(Speculation::Disabled);
        }
        return Ok(Speculation::Adaptive(AdaptiveDrafter::new(
            AdaptiveDrafterConfig::default(),
        )));
    };
    let proposal = NonZeroUsize::new(width)
        .filter(|value| value.get() <= 7)
        .ok_or_else(|| invalid_data("draft_tokens must be in [1, 7]"))?;
    Ok(Speculation::Suffix(
        SuffixDrafter::new(
            NonZeroUsize::new(16).expect("sixteen is nonzero"),
            NonZeroUsize::new(4).expect("four is nonzero"),
            proposal,
        )
        .map_err(|error| invalid_data(error.to_string()))?,
    ))
}

fn chat_tokens(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
    template: ChatTemplateMode,
) -> Result<Vec<u32>, RuntimeError> {
    if template == ChatTemplateMode::Official {
        if architecture == leone::ModelArchitecture::Llama {
            validate_official_llama_tool_history(messages)?;
        }
        return chat_tokens_official(tokenizer, architecture, messages, tools, tool_choice);
    }
    let mut tokens = Vec::new();
    append_chat_prefix(tokenizer, architecture, &mut tokens)?;
    let tool_prompt = tool_system_prompt(tools, tool_choice)?;
    append_chat_messages_with_tools(
        tokenizer,
        architecture,
        messages,
        tool_prompt.as_deref(),
        &mut tokens,
    )?;
    append_chat_suffix_if_needed(tokenizer, architecture, messages, &mut tokens)?;
    Ok(tokens)
}

fn validate_official_llama_tool_history(messages: &[Message]) -> Result<(), RuntimeError> {
    if messages.iter().any(|message| {
        message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| calls.len() != 1)
    }) {
        return Err(RuntimeError::token_callback(
            "official Llama templates support one tool call per assistant message",
        ));
    }
    Ok(())
}

fn chat_tokens_official(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
) -> Result<Vec<u32>, RuntimeError> {
    let rendered = match architecture {
        leone::ModelArchitecture::Qwen3 => {
            chat_tokens_official_qwen(tokenizer, messages, tools, tool_choice)
        }
        leone::ModelArchitecture::Llama => {
            chat_tokens_official_llama(tokenizer, messages, tools, tool_choice)
        }
    }?;
    canonicalize_template_tokens(tokenizer, architecture, &rendered)
}

fn canonicalize_template_tokens(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    tokens: &[u32],
) -> Result<Vec<u32>, RuntimeError> {
    let markers = template_markers(architecture);
    let rendered = template_text_from_tokens(tokenizer, markers, tokens)?;
    encode_template_text(tokenizer, markers, &rendered)
}

fn template_markers(architecture: leone::ModelArchitecture) -> &'static [&'static str] {
    match architecture {
        leone::ModelArchitecture::Qwen3 => &QWEN_TEMPLATE_MARKERS,
        leone::ModelArchitecture::Llama => &LLAMA_TEMPLATE_MARKERS,
    }
}

fn template_text_from_tokens(
    tokenizer: &leone::Tokenizer,
    markers: &[&str],
    tokens: &[u32],
) -> Result<String, RuntimeError> {
    let mut rendered = String::new();
    for token in tokens {
        if let Some(marker) = markers
            .iter()
            .find(|marker| tokenizer.token_id(marker).ok() == Some(*token))
        {
            rendered.push_str(marker);
        } else {
            rendered.push_str(
                &String::from_utf8(tokenizer.token_bytes(*token)?)
                    .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
            );
        }
    }
    Ok(rendered)
}

fn encode_template_text(
    tokenizer: &leone::Tokenizer,
    markers: &[&str],
    mut rendered: &str,
) -> Result<Vec<u32>, RuntimeError> {
    let mut tokens = Vec::new();
    while !rendered.is_empty() {
        let next = markers
            .iter()
            .filter_map(|marker| rendered.find(marker).map(|offset| (offset, *marker)))
            .min_by_key(|(offset, marker)| (*offset, marker.len()));
        let Some((offset, marker)) = next else {
            tokens.extend(tokenizer.encode_piece(rendered)?);
            break;
        };
        if offset > 0 {
            tokens.extend(tokenizer.encode_piece(&rendered[..offset])?);
        }
        tokens.push(tokenizer.token_id(marker)?);
        rendered = &rendered[offset + marker.len()..];
    }
    Ok(tokens)
}

fn official_tools<'a>(
    tools: Option<&'a [ToolDefinition]>,
    tool_choice: Option<&Value>,
) -> Option<&'a [ToolDefinition]> {
    tools.filter(|tools| !tools.is_empty() && tool_choice.and_then(Value::as_str) != Some("none"))
}

fn chat_tokens_official_qwen(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
) -> Result<Vec<u32>, RuntimeError> {
    let mut tokens = Vec::new();
    let tools = official_tools(tools, tool_choice);
    let index = append_official_qwen_prefix(tokenizer, messages, tools, tool_choice, &mut tokens)?;
    append_official_qwen_messages(tokenizer, messages, index, &mut tokens)?;
    append_official_qwen_suffix(tokenizer, messages, &mut tokens)?;
    Ok(tokens)
}

fn append_official_qwen_prefix(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
    tokens: &mut Vec<u32>,
) -> Result<usize, RuntimeError> {
    if let Some(tools) = tools {
        append_qwen_tool_system(tokenizer, messages, tools, tool_choice, tokens)?;
        return Ok(usize::from(
            messages
                .first()
                .is_some_and(|message| message.role == "system"),
        ));
    }
    if messages
        .first()
        .is_some_and(|message| message.role == "system")
    {
        append_regular_chat_message(
            tokenizer,
            leone::ModelArchitecture::Qwen3,
            &messages[0],
            tokens,
        )?;
        return Ok(1);
    }
    Ok(0)
}

fn append_official_qwen_suffix(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    if messages
        .last()
        .is_some_and(|message| message.role == "assistant")
    {
        return Ok(());
    }
    append_chat_suffix(tokenizer, leone::ModelArchitecture::Qwen3, tokens)
}

fn append_qwen_tool_system(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    tools: &[ToolDefinition],
    tool_choice: Option<&Value>,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let mut prompt = String::new();
    if let Some(message) = messages.first().filter(|message| message.role == "system") {
        prompt.push_str(&message_text(&message.content)?);
        prompt.push_str("\n\n");
    }
    prompt.push_str(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
    );
    prompt.push_str(
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        prompt.push('\n');
        prompt
            .push_str(&spaced_json_value(&serde_json::to_value(tool).map_err(
                |error| RuntimeError::token_callback(error.to_string()),
            )?)?);
    }
    prompt.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    append_tool_choice_instruction(&mut prompt, tool_choice)?;
    append_qwen_message(tokenizer, tokens, "system", &prompt)
}

fn append_official_qwen_messages(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    mut index: usize,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    while index < messages.len() {
        index = append_official_qwen_message(tokenizer, messages, index, tokens)?;
    }
    Ok(())
}

fn append_official_qwen_message(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    index: usize,
    tokens: &mut Vec<u32>,
) -> Result<usize, RuntimeError> {
    let message = &messages[index];
    let role = message_role(message)?;
    if role == "tool" {
        return append_official_qwen_tool_group(tokenizer, messages, index, tokens);
    }
    if role == "assistant" {
        append_official_qwen_assistant_message(
            tokenizer,
            message,
            index + 1 == messages.len(),
            tokens,
        )?;
    } else {
        let content = message_text(&message.content)?;
        append_qwen_message(tokenizer, tokens, role, &content)?;
    }
    Ok(index + 1)
}

fn append_official_qwen_assistant_message(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    continuation: bool,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    if continuation && message.tool_calls.is_none() {
        return append_official_qwen_continuation(tokenizer, message, tokens);
    }
    if message.tool_calls.is_some() {
        return append_official_qwen_tool_calls(tokenizer, message, tokens);
    }
    let content = message_text(&message.content)?;
    append_qwen_message(tokenizer, tokens, "assistant", &content)
}

fn append_official_qwen_tool_group(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    mut index: usize,
    tokens: &mut Vec<u32>,
) -> Result<usize, RuntimeError> {
    tokens.push(tokenizer.token_id(IM_START)?);
    tokens.extend(tokenizer.encode_piece("user")?);
    while index < messages.len() && message_role(&messages[index])? == "tool" {
        append_official_qwen_tool_result(tokenizer, &messages[index], tokens)?;
        index += 1;
    }
    tokens.push(tokenizer.token_id(IM_END)?);
    tokens.extend(tokenizer.encode_piece("\n")?);
    Ok(index)
}

fn append_official_qwen_tool_result(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let content = message_text(&message.content)?;
    tokens.extend(tokenizer.encode_piece("\n<tool_response>\n")?);
    append_qwen_content(tokenizer, tokens, &content)?;
    tokens.extend(tokenizer.encode_piece("\n</tool_response>")?);
    Ok(())
}

fn append_official_qwen_continuation(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let content = message_text(&message.content)?;
    tokens.push(tokenizer.token_id(IM_START)?);
    tokens.extend(tokenizer.encode_piece("assistant\n")?);
    tokens.push(tokenizer.token_id("<think>")?);
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    tokens.push(tokenizer.token_id("</think>")?);
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    append_qwen_content(tokenizer, tokens, content.trim_start_matches('\n'))
}

fn append_official_qwen_tool_calls(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let content = if message.content.is_null() {
        String::new()
    } else {
        message_text(&message.content)?
    };
    let rendered =
        render_official_tool_calls(&content, message.tool_calls.as_deref().unwrap_or(&[]))?;
    append_qwen_message(tokenizer, tokens, "assistant", &rendered)
}

fn chat_tokens_official_llama(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
) -> Result<Vec<u32>, RuntimeError> {
    let mut tokens = Vec::new();
    let tools = official_tools(tools, tool_choice);
    append_llama_official_system(tokenizer, messages, tools, &mut tokens)?;
    let mut index = usize::from(
        messages
            .first()
            .is_some_and(|message| message.role == "system"),
    );
    if tools.is_some() {
        index = append_llama_official_first_user(
            tokenizer,
            messages,
            index,
            tools,
            tool_choice,
            &mut tokens,
        )?;
    }
    append_official_llama_messages(tokenizer, messages, index, &mut tokens)?;
    if messages
        .last()
        .is_none_or(|message| message.role != "assistant")
    {
        append_llama_generation_prompt(tokenizer, &mut tokens)?;
    }
    Ok(tokens)
}

fn append_llama_official_system(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    append_llama_header(tokenizer, tokens, "system")?;
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    if tools.is_some() {
        tokens.extend(tokenizer.encode_piece("Environment: ipython\n")?);
    }
    tokens.extend(tokenizer.encode_piece("Cutting Knowledge Date: December 2023\n")?);
    tokens.extend(tokenizer.encode_piece("Today Date: 26 Jul 2024\n\n")?);
    append_llama_official_system_message(tokenizer, messages, tokens)?;
    tokens.push(tokenizer.token_id(LLAMA_EOT)?);
    Ok(())
}

fn append_llama_official_system_message(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    if let Some(message) = messages.first().filter(|message| message.role == "system") {
        tokens.extend(tokenizer.encode_piece(message_text(&message.content)?.trim())?);
    }
    Ok(())
}

fn append_llama_official_first_user(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    index: usize,
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
    tokens: &mut Vec<u32>,
) -> Result<usize, RuntimeError> {
    let Some(tools) = tools else {
        return Ok(index);
    };
    let Some(message) = messages.get(index).filter(|message| message.role == "user") else {
        return Err(RuntimeError::token_callback(
            "official Llama templates require a user message when tools are present",
        ));
    };
    append_llama_header(tokenizer, tokens, "user")?;
    append_llama_official_tool_instruction(tokenizer, tools, tool_choice, tokens)?;
    tokens.extend(tokenizer.encode_piece(message_text(&message.content)?.trim())?);
    tokens.push(tokenizer.token_id(LLAMA_EOT)?);
    Ok(index + 1)
}

fn append_llama_official_tool_instruction(
    tokenizer: &leone::Tokenizer,
    tools: &[ToolDefinition],
    tool_choice: Option<&Value>,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    tokens.extend(tokenizer.encode_piece("\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n")?);
    tokens.extend(tokenizer.encode_piece("Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n")?);
    for tool in tools
        .iter()
        .filter(|tool| named_tool_choice(tool_choice).is_none_or(|name| tool.function.name == name))
    {
        tokens.extend(tokenizer.encode_piece(&pretty_tool_definition(tool)?)?);
        tokens.extend(tokenizer.encode_piece("\n\n")?);
    }
    Ok(())
}

fn append_official_llama_messages(
    tokenizer: &leone::Tokenizer,
    messages: &[Message],
    mut index: usize,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    while index < messages.len() {
        append_official_llama_message(
            tokenizer,
            &messages[index],
            index + 1 == messages.len(),
            tokens,
        )?;
        index += 1;
    }
    Ok(())
}

fn append_official_llama_message(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    continuation: bool,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let role = message_role(message)?;
    if role == "assistant" && message.tool_calls.is_some() {
        append_llama_official_tool_call(tokenizer, message, tokens)
    } else if role == "tool" {
        append_llama_official_tool_result(tokenizer, message, tokens)
    } else if continuation && role == "assistant" {
        append_llama_official_continuation(tokenizer, message, tokens)
    } else {
        append_llama_message(
            tokenizer,
            tokens,
            role,
            message_text(&message.content)?.trim(),
        )
    }
}

fn append_llama_official_continuation(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    append_llama_header(tokenizer, tokens, "assistant")?;
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    tokens.extend(tokenizer.encode_piece(message_text(&message.content)?.trim())?);
    Ok(())
}

fn append_llama_official_tool_call(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let calls = message.tool_calls.as_deref().unwrap_or(&[]);
    if calls.len() != 1 {
        return Err(RuntimeError::token_callback(
            "official Llama templates support one tool call",
        ));
    }
    let call = &calls[0];
    let arguments = parse_request_tool_arguments(&call.function.arguments)?;
    let rendered = format!(
        "{{\"name\": {}, \"parameters\": {}}}",
        serde_json::to_string(&call.function.name)
            .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
        spaced_json_value(&arguments)?
    );
    append_llama_header(tokenizer, tokens, "assistant")?;
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    tokens.extend(tokenizer.encode_piece(&rendered)?);
    tokens.push(tokenizer.token_id(LLAMA_EOT)?);
    Ok(())
}

fn append_llama_official_tool_result(
    tokenizer: &leone::Tokenizer,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    append_llama_header(tokenizer, tokens, "ipython")?;
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    let content = message_text(&message.content)?;
    let quoted = serde_json::to_string(&content)
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    tokens.extend(tokenizer.encode_piece(&quoted)?);
    tokens.push(tokenizer.token_id(LLAMA_EOT)?);
    Ok(())
}

fn append_llama_header(
    tokenizer: &leone::Tokenizer,
    tokens: &mut Vec<u32>,
    role: &str,
) -> Result<(), RuntimeError> {
    tokens.push(tokenizer.token_id(LLAMA_HEADER_START)?);
    tokens.extend(tokenizer.encode_piece(role)?);
    tokens.push(tokenizer.token_id(LLAMA_HEADER_END)?);
    Ok(())
}

fn append_llama_generation_prompt(
    tokenizer: &leone::Tokenizer,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    append_llama_header(tokenizer, tokens, "assistant")?;
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    Ok(())
}

fn pretty_tool_definition(tool: &ToolDefinition) -> Result<String, RuntimeError> {
    let value = serde_json::to_value(tool)
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    pretty_official_json(&value)
}

fn pretty_official_json(value: &Value) -> Result<String, RuntimeError> {
    let mut rendered = String::new();
    append_pretty_official_json(value, 0, &mut rendered)?;
    Ok(rendered)
}

fn append_pretty_official_json(
    value: &Value,
    level: usize,
    rendered: &mut String,
) -> Result<(), RuntimeError> {
    match value {
        Value::Object(object) => append_pretty_official_object(object, level, rendered)?,
        Value::Array(values) => append_pretty_official_array(values, level, rendered)?,
        _ => append_pretty_official_scalar(value, rendered)?,
    }
    Ok(())
}

fn append_pretty_official_object(
    object: &serde_json::Map<String, Value>,
    level: usize,
    rendered: &mut String,
) -> Result<(), RuntimeError> {
    let mut keys = object.keys().collect::<Vec<_>>();
    keys.sort_by_key(|key| official_json_key_order(key));
    rendered.push('{');
    for (index, key) in keys.iter().enumerate() {
        if index == 0 {
            rendered.push('\n');
        }
        append_json_indent(rendered, level + 1);
        rendered.push_str(
            &serde_json::to_string(key)
                .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
        );
        rendered.push_str(": ");
        append_pretty_official_json(&object[*key], level + 1, rendered)?;
        if index + 1 < keys.len() {
            rendered.push_str(",\n");
        } else {
            rendered.push('\n');
        }
    }
    if !keys.is_empty() {
        append_json_indent(rendered, level);
    }
    rendered.push('}');
    Ok(())
}

fn append_pretty_official_array(
    values: &[Value],
    level: usize,
    rendered: &mut String,
) -> Result<(), RuntimeError> {
    rendered.push('[');
    for (index, value) in values.iter().enumerate() {
        if index == 0 {
            rendered.push('\n');
        }
        append_json_indent(rendered, level + 1);
        append_pretty_official_json(value, level + 1, rendered)?;
        if index + 1 < values.len() {
            rendered.push_str(",\n");
        } else {
            rendered.push('\n');
        }
    }
    if !values.is_empty() {
        append_json_indent(rendered, level);
    }
    rendered.push(']');
    Ok(())
}

fn append_pretty_official_scalar(value: &Value, rendered: &mut String) -> Result<(), RuntimeError> {
    rendered.push_str(
        &serde_json::to_string(value)
            .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
    );
    Ok(())
}

fn append_json_indent(rendered: &mut String, level: usize) {
    for _ in 0..level {
        rendered.push_str("    ");
    }
}

fn spaced_json_value(value: &Value) -> Result<String, RuntimeError> {
    Ok(spaced_json_text(&official_json(value)?))
}

fn official_json(value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::Object(object) => official_json_object(object),
        Value::Array(values) => official_json_array(values),
        _ => official_json_scalar(value),
    }
}

fn official_json_object(object: &serde_json::Map<String, Value>) -> Result<String, RuntimeError> {
    let mut keys = object.keys().collect::<Vec<_>>();
    keys.sort_by_key(|key| official_json_key_order(key));
    let fields = keys
        .iter()
        .map(|key| {
            Ok(format!(
                "{}:{}",
                serde_json::to_string(key)
                    .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
                official_json(&object[*key])?
            ))
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    Ok(format!("{{{}}}", fields.join(",")))
}

fn official_json_array(values: &[Value]) -> Result<String, RuntimeError> {
    let values = values
        .iter()
        .map(official_json)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(format!("[{}]", values.join(",")))
}

fn official_json_scalar(value: &Value) -> Result<String, RuntimeError> {
    serde_json::to_string(value).map_err(|error| RuntimeError::token_callback(error.to_string()))
}

fn official_json_key_order(key: &str) -> (u8, &str) {
    let rank = match key {
        "type" => 0,
        "function" => 1,
        "name" => 0,
        "description" => 1,
        "parameters" => 2,
        "properties" => 1,
        "required" => 2,
        _ => 3,
    };
    (rank, key)
}

fn append_chat_suffix_if_needed(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let continues_assistant = messages
        .last()
        .is_some_and(|message| matches!(message_role(message), Ok("assistant")));
    if continues_assistant {
        return Ok(());
    }
    append_chat_suffix(tokenizer, architecture, tokens)
}

fn append_chat_prefix(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    if architecture == leone::ModelArchitecture::Llama {
        tokens.push(tokenizer.token_id(LLAMA_BEGIN)?);
    }
    Ok(())
}

fn append_chat_messages(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let mut index = 0;
    while index < messages.len() {
        index = append_chat_message_at(tokenizer, architecture, messages, index, tokens)?;
    }
    Ok(())
}

fn append_chat_message_at(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    index: usize,
    tokens: &mut Vec<u32>,
) -> Result<usize, RuntimeError> {
    if message_role(&messages[index])? == "tool" {
        return append_tool_message_group(tokenizer, architecture, messages, tokens, index);
    }
    let message = &messages[index];
    if index + 1 == messages.len()
        && message_role(message)? == "assistant"
        && message.tool_calls.is_none()
    {
        append_assistant_continuation(tokenizer, architecture, message, tokens)?;
    } else {
        append_regular_chat_message(tokenizer, architecture, message, tokens)?;
    }
    Ok(index + 1)
}

fn append_assistant_continuation(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let content = render_message_content(message, "assistant")?;
    match architecture {
        leone::ModelArchitecture::Qwen3 => {
            append_qwen_assistant_continuation(tokenizer, &content, tokens)?
        }
        leone::ModelArchitecture::Llama => {
            append_llama_assistant_continuation(tokenizer, &content, tokens)?
        }
    }
    Ok(())
}

fn append_qwen_assistant_continuation(
    tokenizer: &leone::Tokenizer,
    content: &str,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    tokens.push(tokenizer.token_id(IM_START)?);
    tokens.extend(tokenizer.encode_piece("assistant\n")?);
    append_qwen_content(tokenizer, tokens, content)
}

fn append_llama_assistant_continuation(
    tokenizer: &leone::Tokenizer,
    content: &str,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    tokens.push(tokenizer.token_id(LLAMA_HEADER_START)?);
    tokens.extend(tokenizer.encode_piece("assistant")?);
    tokens.push(tokenizer.token_id(LLAMA_HEADER_END)?);
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    tokens.extend(tokenizer.encode_piece(content)?);
    Ok(())
}

fn append_tool_message_group(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tokens: &mut Vec<u32>,
    mut index: usize,
) -> Result<usize, RuntimeError> {
    let mut results = Vec::new();
    while index < messages.len() && message_role(&messages[index])? == "tool" {
        results.push(render_message_content(&messages[index], "tool")?);
        index += 1;
    }
    let content = results.join("\n");
    append_chat_message(tokenizer, architecture, tokens, "user", &content)?;
    Ok(index)
}

fn append_regular_chat_message(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    message: &Message,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let role = message_role(message)?;
    if message.name.is_some() && role != "tool" {
        return Err(RuntimeError::token_callback(
            "message names are supported only for tool results",
        ));
    }
    let rendered = render_message_content(message, role)?;
    append_chat_message(tokenizer, architecture, tokens, role, &rendered)
}

fn append_chat_messages_with_tools(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tool_prompt: Option<&str>,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    let Some(tool_prompt) = tool_prompt else {
        return append_chat_messages(tokenizer, architecture, messages, tokens);
    };
    if let Some(first) = messages.first() {
        if message_role(first)? == "system" {
            if first.name.is_some() {
                return Err(RuntimeError::token_callback(
                    "message names are supported only for tool results",
                ));
            }
            let content = render_message_content(first, "system")?;
            let combined = format!("{content}\n\n{tool_prompt}");
            append_chat_message(tokenizer, architecture, tokens, "system", &combined)?;
            return append_chat_messages(tokenizer, architecture, &messages[1..], tokens);
        }
    }
    append_chat_message(tokenizer, architecture, tokens, "system", tool_prompt)?;
    append_chat_messages(tokenizer, architecture, messages, tokens)
}

fn message_role(message: &Message) -> Result<&'static str, RuntimeError> {
    match message.role.as_str() {
        "system" | "developer" => Ok("system"),
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        "tool" => Ok("tool"),
        _ => Err(RuntimeError::token_callback(format!(
            "message role {:?} is not supported",
            message.role
        ))),
    }
}

fn render_message_content(message: &Message, role: &str) -> Result<String, RuntimeError> {
    let content =
        if role == "assistant" && message.tool_calls.is_some() && message.content.is_null() {
            String::new()
        } else {
            message_text(&message.content)?
        };
    if role == "tool" {
        if message.tool_call_id.as_deref().unwrap_or("").is_empty() {
            return Err(RuntimeError::token_callback(
                "a tool message requires tool_call_id",
            ));
        }
        return Ok(format!("<tool_response>\n{content}\n</tool_response>"));
    }
    if role == "assistant" && message.tool_calls.is_some() {
        return render_request_tool_calls(&content, message.tool_calls.as_deref().unwrap_or(&[]));
    }
    Ok(content)
}

fn append_chat_suffix(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    match architecture {
        leone::ModelArchitecture::Qwen3 => {
            tokens.push(tokenizer.token_id(IM_START)?);
            tokens.extend(tokenizer.encode_piece("assistant")?);
            tokens.extend(tokenizer.encode_piece("\n")?);
        }
        leone::ModelArchitecture::Llama => {
            tokens.push(tokenizer.token_id(LLAMA_HEADER_START)?);
            tokens.extend(tokenizer.encode_piece("assistant")?);
            tokens.push(tokenizer.token_id(LLAMA_HEADER_END)?);
            tokens.extend(tokenizer.encode_piece("\n\n")?);
        }
    }
    Ok(())
}

fn message_text(content: &Value) -> Result<String, RuntimeError> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_owned());
    }
    let parts: Vec<ContentPart> = serde_json::from_value(content.clone())
        .map_err(|_| RuntimeError::token_callback("message content parts are invalid"))?;
    let mut text = String::new();
    for part in parts {
        match part {
            ContentPart::Text { text: part } => text.push_str(&part),
            ContentPart::ImageUrl { image_url } => {
                let input = leone::ImageInput::new(image_url.url, image_url.detail.as_deref())
                    .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
                return Err(RuntimeError::token_callback(format!(
                    "this model has no vision backend ({:?}, {:?})",
                    input.source(),
                    input.detail()
                )));
            }
        }
    }
    Ok(text)
}

fn append_chat_message(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    tokens: &mut Vec<u32>,
    role: &str,
    content: &str,
) -> Result<(), RuntimeError> {
    match architecture {
        leone::ModelArchitecture::Qwen3 => append_qwen_message(tokenizer, tokens, role, content)?,
        leone::ModelArchitecture::Llama => append_llama_message(tokenizer, tokens, role, content)?,
    }
    Ok(())
}

fn append_qwen_message(
    tokenizer: &leone::Tokenizer,
    tokens: &mut Vec<u32>,
    role: &str,
    content: &str,
) -> Result<(), RuntimeError> {
    tokens.push(tokenizer.token_id(IM_START)?);
    tokens.extend(tokenizer.encode_piece(role)?);
    tokens.extend(tokenizer.encode_piece("\n")?);
    append_qwen_content(tokenizer, tokens, content)?;
    tokens.push(tokenizer.token_id(IM_END)?);
    tokens.extend(tokenizer.encode_piece("\n")?);
    Ok(())
}

const QWEN_MARKERS: [&str; 6] = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
];

fn append_qwen_content(
    tokenizer: &leone::Tokenizer,
    tokens: &mut Vec<u32>,
    content: &str,
) -> Result<(), RuntimeError> {
    let mut remaining = content;
    while !remaining.is_empty() {
        let Some((offset, marker)) = QWEN_MARKERS
            .iter()
            .filter_map(|marker| remaining.find(marker).map(|offset| (offset, *marker)))
            .min_by_key(|(offset, _)| *offset)
        else {
            tokens.extend(tokenizer.encode_piece(remaining)?);
            break;
        };
        if offset > 0 {
            tokens.extend(tokenizer.encode_piece(&remaining[..offset])?);
        }
        tokens.push(tokenizer.token_id(marker)?);
        remaining = &remaining[offset + marker.len()..];
    }
    Ok(())
}

fn append_llama_message(
    tokenizer: &leone::Tokenizer,
    tokens: &mut Vec<u32>,
    role: &str,
    content: &str,
) -> Result<(), RuntimeError> {
    tokens.push(tokenizer.token_id(LLAMA_HEADER_START)?);
    tokens.extend(tokenizer.encode_piece(role)?);
    tokens.push(tokenizer.token_id(LLAMA_HEADER_END)?);
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    tokens.extend(tokenizer.encode_piece(content)?);
    tokens.push(tokenizer.token_id(LLAMA_EOT)?);
    Ok(())
}

fn tool_system_prompt(
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
) -> Result<Option<String>, RuntimeError> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    if tools.is_empty() || tool_choice.and_then(Value::as_str) == Some("none") {
        return Ok(None);
    }
    let definitions = tools
        .iter()
        .map(render_tool_definition)
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    let mut prompt = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
    );
    prompt.push_str(
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    prompt.push('\n');
    prompt.push_str(&definitions);
    prompt.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    append_tool_choice_instruction(&mut prompt, tool_choice)?;
    Ok(Some(prompt))
}

fn render_tool_definition(tool: &ToolDefinition) -> Result<String, RuntimeError> {
    let value = serde_json::to_value(tool)
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    spaced_json(&value)
}

fn spaced_json(value: &Value) -> Result<String, RuntimeError> {
    let compact = serde_json::to_string(value)
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    Ok(spaced_json_text(&compact))
}

fn spaced_json_text(compact: &str) -> String {
    let mut rendered = String::with_capacity(compact.len() + compact.len() / 8);
    let mut in_string = false;
    let mut escaped = false;
    for byte in compact.bytes() {
        append_spaced_json_byte(&mut rendered, &mut in_string, &mut escaped, byte);
    }
    rendered
}

fn append_spaced_json_byte(
    rendered: &mut String,
    in_string: &mut bool,
    escaped: &mut bool,
    byte: u8,
) {
    if *in_string {
        rendered.push(char::from(byte));
        update_json_string_state(in_string, escaped, byte);
        return;
    }
    match byte {
        b'"' => {
            *in_string = true;
            rendered.push('"');
        }
        b':' => rendered.push_str(": "),
        b',' => rendered.push_str(", "),
        byte => rendered.push(char::from(byte)),
    }
}

fn update_json_string_state(in_string: &mut bool, escaped: &mut bool, byte: u8) {
    if *escaped {
        *escaped = false;
    } else if byte == b'\\' {
        *escaped = true;
    } else if byte == b'"' {
        *in_string = false;
    }
}

fn append_tool_choice_instruction(
    prompt: &mut String,
    choice: Option<&Value>,
) -> Result<(), RuntimeError> {
    match choice {
        Some(Value::String(value)) if value == "required" => {
            prompt.push_str("\n\nYou must call one of the provided functions.");
        }
        Some(Value::Object(object)) => append_named_tool_instruction(prompt, object)?,
        _ => {}
    }
    Ok(())
}

fn append_named_tool_instruction(
    prompt: &mut String,
    choice: &serde_json::Map<String, Value>,
) -> Result<(), RuntimeError> {
    let name = choice
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .ok_or_else(|| RuntimeError::token_callback("named tool_choice requires function.name"))?;
    prompt.push_str("\n\nYou must call only ");
    prompt.push_str(
        &serde_json::to_string(name)
            .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
    );
    prompt.push('.');
    Ok(())
}

fn render_request_tool_calls(
    content: &str,
    calls: &[RequestToolCall],
) -> Result<String, RuntimeError> {
    let mut rendered = content.to_owned();
    for call in calls {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(&render_request_tool_call(call)?);
    }
    Ok(rendered)
}

fn render_request_tool_call(call: &RequestToolCall) -> Result<String, RuntimeError> {
    if call.kind != "function" || call.function.name.is_empty() {
        return Err(RuntimeError::token_callback(
            "assistant tool calls must name one function",
        ));
    }
    let arguments = parse_request_tool_arguments(&call.function.arguments)?;
    let name = serde_json::to_string(&call.function.name)
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    let arguments = serde_json::to_string(&arguments)
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    Ok(format!(
        "<tool_call>\n{{\"name\": {name}, \"arguments\": {arguments}}}\n</tool_call>"
    ))
}

fn render_official_tool_calls(
    content: &str,
    calls: &[RequestToolCall],
) -> Result<String, RuntimeError> {
    let mut rendered = content.to_owned();
    for call in calls {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        if call.kind != "function" || call.function.name.is_empty() {
            return Err(RuntimeError::token_callback(
                "assistant tool calls must name one function",
            ));
        }
        let arguments = parse_request_tool_arguments(&call.function.arguments)?;
        rendered.push_str("<tool_call>\n{\"name\": ");
        rendered.push_str(
            &serde_json::to_string(&call.function.name)
                .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
        );
        rendered.push_str(", \"arguments\": ");
        rendered.push_str(&spaced_json(&arguments)?);
        rendered.push_str("}\n</tool_call>");
    }
    Ok(rendered)
}

fn parse_request_tool_arguments(arguments: &str) -> Result<Value, RuntimeError> {
    reject_duplicate_json_keys(arguments.as_bytes())
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    let value: Value = serde_json::from_str(arguments)
        .map_err(|_| RuntimeError::token_callback("tool call arguments must be JSON"))?;
    if !value.is_object() {
        return Err(RuntimeError::token_callback(
            "tool call arguments must be a JSON object",
        ));
    }
    Ok(value)
}

fn parse_generated_tool_calls(
    content: &str,
    offered: Option<&[ToolDefinition]>,
    choice: Option<&Value>,
) -> Result<Option<Vec<GeneratedToolCall>>, io::Error> {
    let mut remaining = content.trim();
    if remaining.starts_with('{') {
        if !native_tool_mode(offered, choice) {
            return Ok(None);
        }
        return parse_native_generated_tool_call(remaining, offered, choice);
    }
    if remaining.starts_with("<tool_call>") {
        let mut calls = Vec::new();
        while !remaining.is_empty() {
            let (call, rest) = parse_generated_tool_call(remaining, offered, choice)?;
            calls.push(call);
            remaining = rest;
        }
        return Ok(Some(calls));
    }
    Ok(None)
}

fn native_tool_mode(offered: Option<&[ToolDefinition]>, choice: Option<&Value>) -> bool {
    offered.is_some_and(|tools| !tools.is_empty()) && choice.and_then(Value::as_str) != Some("none")
}

fn parse_native_generated_tool_call(
    content: &str,
    offered: Option<&[ToolDefinition]>,
    choice: Option<&Value>,
) -> Result<Option<Vec<GeneratedToolCall>>, io::Error> {
    let forced = choice.is_some_and(tool_choice_needs_function);
    let Some(value) = parse_native_generated_value(content, forced)? else {
        return Ok(None);
    };
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    if !native_tool_envelope_is_actionable(object, forced)? {
        return Ok(None);
    }
    let name = generated_tool_name(&value, offered, choice)?;
    let arguments = generated_tool_arguments(&value)?;
    Ok(Some(vec![generated_tool_call(name, arguments)?]))
}

fn parse_native_generated_value(content: &str, forced: bool) -> Result<Option<Value>, io::Error> {
    if let Err(error) = reject_duplicate_json_keys(content.as_bytes()) {
        if forced {
            return Err(invalid_data(format!(
                "tool call output has invalid JSON: {error}"
            )));
        }
        return Ok(None);
    }
    match serde_json::from_str(content) {
        Ok(value) => Ok(Some(value)),
        Err(_error) if !forced => Ok(None),
        Err(error) => Err(invalid_data(format!(
            "tool call output is invalid JSON: {error}"
        ))),
    }
}

fn native_tool_envelope_is_actionable(
    object: &serde_json::Map<String, Value>,
    forced: bool,
) -> Result<bool, io::Error> {
    let has_name = object
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|name| !name.is_empty());
    let has_arguments = object.contains_key("arguments") || object.contains_key("parameters");
    if !has_name || !has_arguments {
        return reject_native_tool_envelope(
            forced,
            "forced tool choice requires one name and one arguments object",
        );
    }
    if object
        .keys()
        .any(|key| !matches!(key.as_str(), "name" | "arguments" | "parameters"))
    {
        return reject_native_tool_envelope(forced, "tool call output contains an unknown field");
    }
    Ok(true)
}

fn reject_native_tool_envelope(forced: bool, message: &str) -> Result<bool, io::Error> {
    if forced {
        Err(invalid_data(message))
    } else {
        Ok(false)
    }
}

fn parse_generated_tool_call<'a>(
    remaining: &'a str,
    offered: Option<&[ToolDefinition]>,
    choice: Option<&Value>,
) -> Result<(GeneratedToolCall, &'a str), io::Error> {
    let body = remaining
        .strip_prefix("<tool_call>")
        .ok_or_else(|| invalid_data("tool call output contains text outside its tags"))?;
    let end = body
        .find("</tool_call>")
        .ok_or_else(|| invalid_data("tool call output is missing </tool_call>"))?;
    let (name, arguments) = parse_generated_tool_fields(body, end, offered, choice)?;
    let rest = body[end + "</tool_call>".len()..].trim();
    let call = generated_tool_call(name, arguments)?;
    Ok((call, rest))
}

fn generated_tool_call(name: String, arguments: Value) -> Result<GeneratedToolCall, io::Error> {
    Ok(GeneratedToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        kind: "function",
        function: GeneratedToolCallFunction {
            name,
            arguments: serde_json::to_string(&arguments).map_err(invalid_json)?,
        },
    })
}

fn parse_generated_tool_fields(
    body: &str,
    end: usize,
    offered: Option<&[ToolDefinition]>,
    choice: Option<&Value>,
) -> Result<(String, Value), io::Error> {
    let value = parse_generated_tool_value(body, end)?;
    let name = generated_tool_name(&value, offered, choice)?;
    let arguments = generated_tool_arguments(&value)?;
    Ok((name, arguments))
}

fn parse_generated_tool_value(body: &str, end: usize) -> Result<Value, io::Error> {
    let value = body[..end].trim();
    reject_duplicate_json_keys(value.as_bytes())?;
    serde_json::from_str(value).map_err(invalid_json)
}

fn generated_tool_name(
    value: &Value,
    offered: Option<&[ToolDefinition]>,
    choice: Option<&Value>,
) -> Result<String, io::Error> {
    let name = generated_tool_name_field(value)?;
    validate_generated_tool_name(name, offered, choice)?;
    Ok(name.to_owned())
}

fn generated_tool_name_field(value: &Value) -> Result<&str, io::Error> {
    value
        .as_object()
        .ok_or_else(|| invalid_data("tool call output must be one JSON object"))?
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid_data("tool call output requires one function name"))
}

fn validate_generated_tool_name(
    name: &str,
    offered: Option<&[ToolDefinition]>,
    choice: Option<&Value>,
) -> Result<(), io::Error> {
    if !offered.is_some_and(|tools| tools.iter().any(|tool| tool.function.name == name)) {
        return Err(invalid_data(format!(
            "tool call output names an unoffered function: {name}"
        )));
    }
    validate_named_generated_tool_name(name, choice)
}

fn validate_named_generated_tool_name(name: &str, choice: Option<&Value>) -> Result<(), io::Error> {
    if let Some(selected) = named_tool_choice(choice) {
        if selected != name {
            return Err(invalid_data(format!(
                "tool call output does not match named tool_choice: {selected}"
            )));
        }
    }
    Ok(())
}

fn named_tool_choice(choice: Option<&Value>) -> Option<&str> {
    choice
        .and_then(Value::as_object)
        .and_then(|object| object.get("function"))
        .and_then(Value::as_object)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
}

fn validate_generated_tool_choice(
    choice: Option<&Value>,
    has_tool_calls: bool,
) -> Result<(), io::Error> {
    match choice {
        Some(Value::String(value)) if value == "none" && has_tool_calls => {
            Err(invalid_data("tool_choice none forbids tool calls"))
        }
        Some(Value::String(value)) if value == "required" && !has_tool_calls => {
            Err(invalid_data("tool_choice required needs a tool call"))
        }
        Some(Value::Object(_)) if !has_tool_calls => {
            Err(invalid_data("named tool_choice needs a tool call"))
        }
        _ => Ok(()),
    }
}

fn generated_tool_arguments(value: &Value) -> Result<Value, io::Error> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_data("tool call output must be one JSON object"))?;
    if object
        .keys()
        .any(|key| !matches!(key.as_str(), "name" | "arguments" | "parameters"))
    {
        return Err(invalid_data("tool call output contains an unknown field"));
    }
    let mut fields = object
        .iter()
        .filter(|(key, _)| matches!(key.as_str(), "arguments" | "parameters"));
    let Some((_, arguments)) = fields.next() else {
        return Err(invalid_data(
            "tool call output requires arguments or parameters",
        ));
    };
    if fields.next().is_some() {
        return Err(invalid_data(
            "tool call output must use one arguments field",
        ));
    }
    if !arguments.is_object() {
        return Err(invalid_data("tool call arguments must be one JSON object"));
    }
    Ok(arguments.clone())
}

fn read_request(
    stream: &mut TcpStream,
    deadline: std::time::Instant,
) -> Result<Option<HttpRequest>, io::Error> {
    let Some((mut bytes, header_end)) = read_request_headers(stream, deadline)? else {
        return Ok(None);
    };
    let (method, path, headers) = parse_request_head(&bytes[..header_end])?;
    let content_length = request_content_length(&headers)?;
    if content_length > MAX_BODY_BYTES {
        return Err(invalid_input("HTTP body exceeds 4 MiB"));
    }
    read_request_body(stream, &mut bytes, header_end, content_length, deadline)?;
    Ok(Some(HttpRequest {
        method,
        path,
        headers,
        body: bytes[header_end..header_end + content_length].to_vec(),
    }))
}

fn reject_duplicate_json_keys(body: &[u8]) -> Result<(), io::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    Deserializer::deserialize_any(&mut deserializer, DuplicateJsonVisitor::new())
        .map_err(invalid_json)?;
    deserializer.end().map_err(invalid_json)
}

struct DuplicateJsonVisitor {
    nodes: Rc<Cell<usize>>,
    depth: usize,
}

impl DuplicateJsonVisitor {
    fn new() -> Self {
        Self {
            nodes: Rc::new(Cell::new(0)),
            depth: 0,
        }
    }

    fn claim<E: de::Error>(&self) -> Result<(), E> {
        let nodes = self.nodes.get();
        if nodes >= MAX_JSON_NODES {
            return Err(E::custom("JSON input exceeds the node limit"));
        }
        self.nodes.set(nodes + 1);
        Ok(())
    }

    fn child(&self) -> Self {
        Self {
            nodes: Rc::clone(&self.nodes),
            depth: self.depth + 1,
        }
    }
}

impl<'de> Visitor<'de> for DuplicateJsonVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("one JSON value")
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_str<E: de::Error>(self, _: &str) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_string<E: de::Error>(self, _: String) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        self.claim()?;
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        self.claim()?;
        if self.depth >= MAX_JSON_DEPTH {
            return Err(de::Error::custom("JSON input exceeds the depth limit"));
        }
        while sequence
            .next_element_seed(DuplicateJsonSeed {
                visitor: self.child(),
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        self.claim()?;
        if self.depth >= MAX_JSON_DEPTH {
            return Err(de::Error::custom("JSON input exceeds the depth limit"));
        }
        let mut keys = std::collections::BTreeSet::new();
        loop {
            self.claim()?;
            let Some(key) = map.next_key::<String>()? else {
                break;
            };
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON field {key:?}")));
            }
            map.next_value_seed(DuplicateJsonSeed {
                visitor: self.child(),
            })?;
        }
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

struct DuplicateJsonSeed {
    visitor: DuplicateJsonVisitor,
}

impl<'de> DeserializeSeed<'de> for DuplicateJsonSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self.visitor)
    }
}

fn read_request_headers(
    stream: &mut TcpStream,
    deadline: std::time::Instant,
) -> Result<Option<(Vec<u8>, usize)>, io::Error> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8192];
    let header_end = loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "request deadline expired",
            ));
        }
        stream.set_read_timeout(Some(remaining))?;
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Ok(None);
        }
        bytes.extend_from_slice(&buffer[..read]);
        if bytes.len() > MAX_HEADER_BYTES {
            return Err(invalid_data("HTTP headers exceed 64 KiB"));
        }
        if let Some(end) = find_bytes(&bytes, b"\r\n\r\n") {
            break end + 4;
        }
    };
    Ok(Some((bytes, header_end)))
}

fn parse_request_head(
    bytes: &[u8],
) -> Result<(String, String, BTreeMap<String, String>), io::Error> {
    let head =
        std::str::from_utf8(bytes).map_err(|_| invalid_data("HTTP headers are not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let (method, path) = parse_request_line(
        lines
            .next()
            .ok_or_else(|| invalid_data("HTTP request line is missing"))?,
    )?;
    let headers = parse_request_headers(lines)?;
    Ok((method, path, headers))
}

fn parse_request_line(line: &str) -> Result<(String, String), io::Error> {
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| invalid_data("HTTP method is missing"))?
        .to_owned();
    let path = parts
        .next()
        .ok_or_else(|| invalid_data("HTTP path is missing"))?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    Ok((method, path))
}

fn parse_request_headers<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> Result<BTreeMap<String, String>, io::Error> {
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| invalid_data("HTTP header is malformed"))?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "x-leone-session" && headers.contains_key(&name) {
            return Err(invalid_data("x-leone-session must appear once"));
        }
        headers
            .entry(name)
            .and_modify(|current| {
                current.push(',');
                current.push_str(value);
            })
            .or_insert_with(|| value.to_owned());
    }
    Ok(headers)
}

fn request_content_length(headers: &BTreeMap<String, String>) -> Result<usize, io::Error> {
    headers
        .get("content-length")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| invalid_data("Content-Length is invalid"))
        })
        .transpose()
        .map(|length| length.unwrap_or(0))
}

fn read_request_body(
    stream: &mut TcpStream,
    bytes: &mut Vec<u8>,
    header_end: usize,
    content_length: usize,
    deadline: std::time::Instant,
) -> Result<(), io::Error> {
    let mut buffer = [0_u8; 8192];
    while bytes.len() - header_end < content_length {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "request deadline expired",
            ));
        }
        stream.set_read_timeout(Some(remaining))?;
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "HTTP body ended early",
            ));
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    Ok(())
}

fn write_json_with_origin<W: Write + ?Sized, T: Serialize>(
    stream: &mut W,
    status: u16,
    value: &T,
    origin: Option<&str>,
) -> io::Result<()> {
    write_json_with_session_origin(stream, status, "", value, origin)
}

fn write_json_with_session_origin<W: Write + ?Sized, T: Serialize>(
    stream: &mut W,
    status: u16,
    session: &str,
    value: &T,
    origin: Option<&str>,
) -> io::Result<()> {
    let body = serde_json::to_vec(value).map_err(invalid_json)?;
    let reason = response_reason(status);
    let mut response = Vec::new();
    write!(
        &mut response,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    )?;
    write_cors_headers(&mut response, origin)?;
    if !session.is_empty() {
        write!(&mut response, "X-Leone-Session: {session}\r\n")?;
    }
    write!(&mut response, "\r\n")?;
    response.extend_from_slice(&body);
    stream.write_all(&response)
}

fn response_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        _ => "Error",
    }
}

fn write_error_with_origin<W: Write + ?Sized>(
    stream: &mut W,
    status: u16,
    message: &str,
    origin: Option<String>,
) -> io::Result<()> {
    write_json_with_origin(
        stream,
        status,
        &json!({"error": {"message": message, "type": error_type_for_status(status)}}),
        origin.as_deref(),
    )
}

fn write_session_lease_error_with_origin<W: Write + ?Sized>(
    stream: &mut W,
    capacity: bool,
    origin: Option<String>,
) -> io::Result<()> {
    let (status, error) = if capacity {
        (
            429,
            json!({
                "message": "the server cannot admit this request",
                "type": "server_overloaded",
                "code": "session-capacity"
            }),
        )
    } else {
        (
            500,
            json!({
                "message": SESSION_PREPARATION_ERROR,
                "type": "server_error"
            }),
        )
    };
    write_json_with_origin(stream, status, &json!({"error": error}), origin.as_deref())
}

fn error_type_for_status(status: u16) -> &'static str {
    if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    }
}

#[cfg(test)]
fn write_admission_rejection<W: Write + ?Sized>(
    stream: &mut W,
    reason: AdmissionReject,
) -> io::Result<()> {
    write_admission_rejection_with_origin(stream, reason, None)
}

fn write_admission_rejection_with_origin<W: Write + ?Sized>(
    stream: &mut W,
    reason: AdmissionReject,
    origin: Option<&str>,
) -> io::Result<()> {
    let code = match reason {
        AdmissionReject::ActiveLimit => "active-limit",
        AdmissionReject::QueueLimit => "queue-limit",
        AdmissionReject::KvCapacity => "kv-capacity",
        AdmissionReject::PromptLimit => "prompt-limit",
        AdmissionReject::OutputLimit => "output-limit",
        AdmissionReject::InvalidPrefix => "invalid-prefix",
        AdmissionReject::InvalidPriority => "invalid-priority",
        AdmissionReject::ExpiredDeadline => "expired-deadline",
    };
    write_json_with_origin(
        stream,
        429,
        &json!({
            "error": {
                "message": "the server cannot admit this request",
                "type": "server_overloaded",
                "code": code
            }
        }),
        origin,
    )
}

fn write_empty_with_origin<W: Write + ?Sized>(
    stream: &mut W,
    status: u16,
    reason: &str,
    origin: Option<String>,
) -> io::Result<()> {
    let mut response = Vec::new();
    write!(
        &mut response,
        "HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\n"
    )?;
    write_cors_headers(&mut response, origin.as_deref())?;
    write!(
        &mut response,
        "Access-Control-Allow-Headers: authorization, content-type, x-leone-session\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n"
    )?;
    stream.write_all(&response)
}

fn write_stream_headers_origin<W: Write + ?Sized>(
    stream: &mut W,
    session: &str,
    origin: Option<&str>,
) -> io::Result<()> {
    let mut response = Vec::new();
    write!(
        &mut response,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n"
    )?;
    write_cors_headers(&mut response, origin)?;
    write!(&mut response, "X-Leone-Session: {session}\r\n\r\n")?;
    stream.write_all(&response)
}

fn write_cors_headers<W: Write + ?Sized>(stream: &mut W, origin: Option<&str>) -> io::Result<()> {
    if let Some(origin) = origin {
        write!(
            stream,
            "Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\nAccess-Control-Expose-Headers: X-Leone-Session\r\n"
        )?;
    }
    Ok(())
}

fn response_origin<B: Backend>(server: &Server<B>, request: &HttpRequest) -> Option<String> {
    let origin = request.headers.get("origin")?;
    if server.cors_origins.iter().any(|allowed| allowed == origin)
        || server.proxy_origin.as_deref() == Some(origin)
    {
        Some(origin.clone())
    } else {
        None
    }
}

fn request_client_identity(
    request: &HttpRequest,
    peer_addr: SocketAddr,
    trusted_proxy_ips: &[IpAddr],
) -> IpAddr {
    let peer = canonical_ip(peer_addr.ip());
    let trusted = trusted_proxy_ips
        .iter()
        .copied()
        .map(canonical_ip)
        .collect::<Vec<_>>();
    if !trusted.contains(&peer) {
        return peer;
    }
    let Some(forwarded) = request.headers.get("x-forwarded-for") else {
        return peer;
    };
    for value in forwarded.split(',').rev() {
        let Ok(identity) = value.trim().parse::<IpAddr>() else {
            return peer;
        };
        let identity = canonical_ip(identity);
        if !trusted.contains(&identity) {
            return identity;
        }
    }
    peer
}

fn write_stream_error<W: Write + ?Sized>(
    stream: &mut W,
    headers_written: bool,
    status: u16,
    message: &str,
    origin: Option<String>,
) -> io::Result<()> {
    if headers_written {
        write_sse(
            stream,
            &json!({
                "error": {
                    "message": message,
                    "type": error_type_for_status(status)
                }
            }),
        )?;
        write_chunk(stream, b"data: [DONE]\n\n")?;
        return finish_chunks(stream);
    }
    write_error_with_origin(stream, status, message, origin)
}

fn write_sse<W: Write + ?Sized>(stream: &mut W, value: &Value) -> io::Result<()> {
    let mut payload = b"data: ".to_vec();
    payload.extend(serde_json::to_vec(value).map_err(invalid_json)?);
    payload.extend_from_slice(b"\n\n");
    write_chunk(stream, &payload)
}

fn write_chunk<W: Write + ?Sized>(stream: &mut W, payload: &[u8]) -> io::Result<()> {
    let mut frame = Vec::with_capacity(payload.len().saturating_add(32));
    write!(&mut frame, "{:x}\r\n", payload.len())?;
    frame.extend_from_slice(payload);
    frame.extend_from_slice(b"\r\n");
    stream.write_all(&frame)?;
    stream.flush()
}

fn finish_chunks<W: Write + ?Sized>(stream: &mut W) -> io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

fn parse(arguments: &[String]) -> Result<ServeArgs, io::Error> {
    let mut parsed = ServeArgsBuilder::new()?;
    let mut index = 0;
    while index < arguments.len() {
        if parse_serve_argument(&mut parsed, arguments, &mut index)? {
            index += 1;
            continue;
        }
        return Err(invalid_data(format!(
            "serve argument is invalid: {}",
            arguments[index]
        )));
    }
    parsed.finish()
}

struct ServeArgsBuilder {
    model: Option<PathBuf>,
    bind: SocketAddr,
    sessions: usize,
    batch_size: Option<usize>,
    hibernated_sessions: usize,
    backend: BackendChoice,
    kv_cache_dtype: KvCacheDtype,
    kv_cache_dtype_explicit: bool,
    receipts: PathBuf,
    signing_key: Option<PathBuf>,
    session_store: Option<PathBuf>,
    allow_remote: bool,
    plan: Option<PathBuf>,
    prefill_chunk_tokens: Option<usize>,
    context_limit: Option<usize>,
    memory_budget: BudgetRequest,
    host_memory_budget: BudgetRequest,
    kv_reservation_budget: BudgetRequest,
    max_connections: usize,
    max_connections_per_client: usize,
    max_pending_requests: usize,
    max_output_bytes: usize,
    request_timeout_ms: u64,
    cors_origins: Vec<String>,
    proxy_origin: Option<String>,
    trusted_proxy_ips: Vec<IpAddr>,
}

impl ServeArgsBuilder {
    fn new() -> Result<Self, io::Error> {
        Ok(Self {
            model: None,
            bind: DEFAULT_BIND
                .parse::<SocketAddr>()
                .expect("default bind is valid"),
            sessions: 2,
            batch_size: None,
            hibernated_sessions: 8,
            backend: default_backend(),
            kv_cache_dtype: KvCacheDtype::F16,
            kv_cache_dtype_explicit: false,
            receipts: PathBuf::from("receipts"),
            signing_key: None,
            session_store: None,
            allow_remote: false,
            plan: None,
            prefill_chunk_tokens: None,
            context_limit: None,
            memory_budget: BudgetRequest::Auto,
            host_memory_budget: BudgetRequest::Auto,
            kv_reservation_budget: BudgetRequest::Auto,
            max_connections: 64,
            max_connections_per_client: 4,
            max_pending_requests: 64,
            max_output_bytes: 256 * 1024,
            request_timeout_ms: 30_000,
            cors_origins: Vec::new(),
            proxy_origin: None,
            trusted_proxy_ips: Vec::new(),
        })
    }

    fn finish(self) -> Result<ServeArgs, io::Error> {
        validate_serve_backend_options(self.backend, self.plan.is_some())?;
        let signing_key = match self.signing_key {
            Some(path) => path,
            None => default_key_path()?,
        };
        Ok(ServeArgs {
            model: self
                .model
                .ok_or_else(|| invalid_data("serve requires -m <gguf>"))?,
            bind: self.bind,
            sessions: self.sessions,
            batch_size: self.batch_size,
            hibernated_sessions: self.hibernated_sessions,
            backend: self.backend,
            kv_cache_dtype: self.kv_cache_dtype,
            kv_cache_dtype_explicit: self.kv_cache_dtype_explicit,
            receipts: self.receipts,
            signing_key,
            session_store: self.session_store,
            allow_remote: self.allow_remote,
            plan: self.plan,
            prefill_chunk_tokens: self.prefill_chunk_tokens,
            context_limit: self.context_limit,
            memory_budget: self.memory_budget,
            host_memory_budget: self.host_memory_budget,
            kv_reservation_budget: self.kv_reservation_budget,
            max_connections: self.max_connections,
            max_connections_per_client: self.max_connections_per_client,
            max_pending_requests: self.max_pending_requests,
            max_output_bytes: self.max_output_bytes,
            request_timeout_ms: self.request_timeout_ms,
            cors_origins: self.cors_origins,
            proxy_origin: self.proxy_origin,
            trusted_proxy_ips: self.trusted_proxy_ips,
        })
    }
}

fn validate_serve_backend_options(backend: BackendChoice, has_plan: bool) -> Result<(), io::Error> {
    if has_plan && backend != BackendChoice::Cuda {
        return Err(invalid_data("--plan requires the CUDA backend"));
    }
    validate_compiled(backend)
}

type ServeArgumentParser =
    fn(&mut ServeArgsBuilder, &[String], &mut usize) -> Result<bool, io::Error>;

fn parse_serve_argument(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let parsers: &[ServeArgumentParser] = &[
        parse_serve_model,
        parse_serve_bind,
        parse_serve_sessions,
        parse_serve_batch_size,
        parse_serve_hibernated_sessions,
        parse_serve_backend,
        parse_serve_kv,
        parse_serve_receipts,
        parse_serve_signing_key,
        parse_serve_session_store,
        parse_serve_allow_remote,
        parse_serve_plan,
        parse_serve_prefill_chunk,
        parse_serve_context_limit,
        parse_serve_memory_budget,
        parse_serve_host_memory_budget,
        parse_serve_kv_reservation_budget,
        parse_serve_max_connections,
        parse_serve_max_connections_per_client,
        parse_serve_max_pending_requests,
        parse_serve_max_output_bytes,
        parse_serve_request_timeout,
        parse_serve_cors_origin,
        parse_serve_proxy_origin,
        parse_serve_trusted_proxy,
    ];
    for parser in parsers {
        if parser(parsed, arguments, index)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn parse_serve_model(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if !matches!(arguments[*index].as_str(), "-m" | "--model") {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_serve_bind(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--bind" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.bind = value
        .parse()
        .map_err(|_| invalid_data(format!("bind address is invalid: {value}")))?;
    Ok(true)
}

fn parse_serve_sessions(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--sessions" {
        return Ok(false);
    }
    parsed.sessions = parse_positive_serve_value(flag_value(arguments, index)?, "sessions")?;
    Ok(true)
}

fn parse_serve_batch_size(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--batch-size" {
        return Ok(false);
    }
    parsed.batch_size = Some(parse_positive_serve_value(
        flag_value(arguments, index)?,
        "batch size",
    )?);
    Ok(true)
}

fn parse_serve_hibernated_sessions(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--hibernated-sessions" {
        return Ok(false);
    }
    parsed.hibernated_sessions =
        parse_positive_serve_value(flag_value(arguments, index)?, "hibernated sessions")?;
    Ok(true)
}

fn parse_positive_serve_value(value: &str, name: &str) -> Result<usize, io::Error> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data(format!("{name} must be nonzero")))
}

fn parse_serve_backend(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--backend" {
        return Ok(false);
    }
    parsed.backend = match flag_value(arguments, index)? {
        "cuda" => BackendChoice::Cuda,
        "cpu" => BackendChoice::Cpu,
        "metal" => BackendChoice::Metal,
        value => return Err(invalid_data(format!("backend is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_serve_kv(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--kv" {
        return Ok(false);
    }
    parsed.kv_cache_dtype = match flag_value(arguments, index)? {
        "q8" => KvCacheDtype::Q8,
        "f16" => KvCacheDtype::F16,
        "f32" => KvCacheDtype::F32,
        value => return Err(invalid_data(format!("KV dtype is invalid: {value}"))),
    };
    parsed.kv_cache_dtype_explicit = true;
    Ok(true)
}

fn parse_serve_receipts(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--receipt-dir" {
        return Ok(false);
    }
    parsed.receipts = PathBuf::from(flag_value(arguments, index)?);
    Ok(true)
}

fn parse_serve_signing_key(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--signing-key" {
        return Ok(false);
    }
    parsed.signing_key = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_serve_session_store(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--session-store" {
        return Ok(false);
    }
    parsed.session_store = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_serve_allow_remote(
    parsed: &mut ServeArgsBuilder,
    _arguments: &[String],
    _index: &mut usize,
) -> Result<bool, io::Error> {
    if _arguments[*_index] != "--allow-remote" {
        return Ok(false);
    }
    parsed.allow_remote = true;
    Ok(true)
}

fn parse_serve_plan(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--plan" {
        return Ok(false);
    }
    parsed.plan = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_serve_prefill_chunk(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--prefill-chunk" {
        return Ok(false);
    }
    parsed.prefill_chunk_tokens = Some(parse_positive_serve_value(
        flag_value(arguments, index)?,
        "prefill chunk",
    )?);
    Ok(true)
}

fn parse_serve_context_limit(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--context-limit" {
        return Ok(false);
    }
    parsed.context_limit = Some(parse_positive_serve_value(
        flag_value(arguments, index)?,
        "context limit",
    )?);
    Ok(true)
}

fn parse_serve_memory_budget(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--memory-budget-bytes" {
        return Ok(false);
    }
    parsed.memory_budget = BudgetRequest::parse(flag_value(arguments, index)?)
        .map_err(|error| invalid_data(format!("memory budget is invalid: {error}")))?;
    Ok(true)
}

fn parse_serve_host_memory_budget(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--host-memory-budget-bytes" {
        return Ok(false);
    }
    parsed.host_memory_budget = BudgetRequest::parse(flag_value(arguments, index)?)
        .map_err(|error| invalid_data(format!("host memory budget is invalid: {error}")))?;
    Ok(true)
}

fn parse_serve_kv_reservation_budget(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--kv-reservation-budget-bytes" {
        return Ok(false);
    }
    parsed.kv_reservation_budget = BudgetRequest::parse(flag_value(arguments, index)?)
        .map_err(|error| invalid_data(format!("KV reservation budget is invalid: {error}")))?;
    Ok(true)
}

fn parse_serve_max_connections(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--max-connections" {
        return Ok(false);
    }
    parsed.max_connections =
        parse_positive_serve_value(flag_value(arguments, index)?, "max connections")?;
    Ok(true)
}

fn parse_serve_max_connections_per_client(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--max-connections-per-client" {
        return Ok(false);
    }
    parsed.max_connections_per_client =
        parse_positive_serve_value(flag_value(arguments, index)?, "max connections per client")?;
    Ok(true)
}

fn parse_serve_max_pending_requests(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--max-pending-requests" {
        return Ok(false);
    }
    parsed.max_pending_requests =
        parse_positive_serve_value(flag_value(arguments, index)?, "max pending requests")?;
    Ok(true)
}

fn parse_serve_max_output_bytes(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--max-output-bytes" {
        return Ok(false);
    }
    parsed.max_output_bytes =
        parse_positive_serve_value(flag_value(arguments, index)?, "max output bytes")?;
    Ok(true)
}

fn parse_serve_request_timeout(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--request-timeout-ms" {
        return Ok(false);
    }
    parsed.request_timeout_ms = flag_value(arguments, index)?
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data("request timeout must be nonzero"))?;
    Ok(true)
}

fn parse_serve_cors_origin(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--cors-origin" {
        return Ok(false);
    }
    let origin = explicit_origin(flag_value(arguments, index)?, "CORS origin")?;
    parsed.cors_origins.push(origin);
    Ok(true)
}

fn parse_serve_proxy_origin(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--proxy-origin" {
        return Ok(false);
    }
    let origin = explicit_origin(flag_value(arguments, index)?, "proxy origin")?;
    parsed.proxy_origin = Some(origin);
    Ok(true)
}

fn parse_serve_trusted_proxy(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--trusted-proxy" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    let address = value
        .parse()
        .map_err(|_| invalid_data(format!("trusted proxy IP is invalid: {value}")))?;
    parsed.trusted_proxy_ips.push(address);
    Ok(true)
}

fn explicit_origin(value: &str, label: &str) -> Result<String, io::Error> {
    if value.is_empty() || value == "*" || value.bytes().any(|byte| byte <= b' ' || byte == 0x7f) {
        return Err(invalid_data(format!("{label} must be explicit")));
    }
    Ok(value.to_owned())
}

fn load_or_create_key(path: &Path) -> Result<SigningKey, io::Error> {
    if path.exists() {
        return read_signing_key(path);
    }
    create_signing_key(path)
}

fn read_signing_key(path: &Path) -> Result<SigningKey, io::Error> {
    let bytes = fs::read(path)?;
    let seed: [u8; 32] = bytes
        .try_into()
        .map_err(|_| invalid_data("Ed25519 key file must contain exactly 32 bytes"))?;
    Ok(SigningKey::from_bytes(&seed))
}

fn create_signing_key(path: &Path) -> Result<SigningKey, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut seed = [0_u8; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut seed)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)?.write_all(&seed)?;
    Ok(SigningKey::from_bytes(&seed))
}

fn default_key_path() -> Result<PathBuf, io::Error> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| invalid_data("HOME is not set; pass --signing-key"))?;
    Ok(PathBuf::from(home).join(".config/leone/response.key"))
}

fn prefix_candidate_is_better(
    candidate: (usize, u64, &str),
    incumbent: (usize, u64, &str),
) -> bool {
    candidate.0 > incumbent.0
        || (candidate.0 == incumbent.0
            && (candidate.1 > incumbent.1
                || (candidate.1 == incumbent.1 && candidate.2 < incumbent.2)))
}

fn reuse_class_name(class: SessionReuseClass) -> &'static str {
    match class {
        SessionReuseClass::Cold => "cold",
        SessionReuseClass::ExactRepeat => "exact-repeat",
        SessionReuseClass::AppendOnly => "append-only",
        SessionReuseClass::ArbitraryBranch => "arbitrary-branch",
        SessionReuseClass::RestoreReplay => "restore-replay",
        SessionReuseClass::DeviceFork => "device-fork",
        SessionReuseClass::HostWake => "host-wake",
    }
}

fn host_u64(value: usize) -> Result<u64, io::Error> {
    u64::try_from(value).map_err(|_| invalid_data("token count exceeds u64"))
}

fn unix_seconds() -> Result<u64, io::Error> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| invalid_data("system clock is before 1970"))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(feature = "cuda")]
fn nonzero_usize(value: &str, field: &str) -> Result<usize, io::Error> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data(format!("{field} must be nonzero: {value}")))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_json(error: serde_json::Error) -> io::Error {
    invalid_data(error.to_string())
}

fn request_stop_sequences(request: &ChatRequest) -> Result<Vec<Vec<u8>>, io::Error> {
    let Some(value) = request.stop.as_ref() else {
        return Ok(Vec::new());
    };
    let values = stop_values(value)?;
    if values.iter().any(String::is_empty) {
        return Err(invalid_data("stop sequences must not be empty"));
    }
    Ok(values.into_iter().map(String::into_bytes).collect())
}

const MAX_STOP_BYTES: usize = 4096;

fn stop_values(value: &Value) -> Result<Vec<String>, io::Error> {
    let values = parse_stop_values(value)?;
    validate_stop_bytes(&values)?;
    Ok(values)
}

fn parse_stop_values(value: &Value) -> Result<Vec<String>, io::Error> {
    match value {
        Value::String(sequence) => Ok(vec![sequence.clone()]),
        Value::Array(values) => {
            if values.len() > 4 {
                return Err(invalid_data("stop supports at most four sequences"));
            }
            Ok(values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid_data("stop array entries must be strings"))
                })
                .collect::<Result<Vec<_>, _>>()?)
        }
        _ => Err(invalid_data(
            "stop must be one string or an array of strings",
        )),
    }
}

fn validate_stop_bytes(values: &[String]) -> Result<(), io::Error> {
    values
        .iter()
        .try_fold(0_usize, |total, value| {
            if value.is_empty() {
                return Err(invalid_data("stop sequences must be nonempty"));
            }
            total
                .checked_add(value.len())
                .filter(|total| *total <= MAX_STOP_BYTES)
                .ok_or_else(|| invalid_data("stop sequences exceed 4096 bytes"))
        })
        .map(|_| ())
}

fn truncate_at_stop(content: &str, sequences: &[Vec<u8>]) -> (String, bool) {
    let Some(position) = find_stop_position(content.as_bytes(), sequences) else {
        return (content.to_owned(), false);
    };
    (content[..position].to_owned(), true)
}

fn find_stop_position(bytes: &[u8], sequences: &[Vec<u8>]) -> Option<usize> {
    sequences
        .iter()
        .filter_map(|sequence| {
            bytes
                .windows(sequence.len())
                .position(|window| window == sequence)
        })
        .min()
}

#[cfg(test)]
#[path = "../../leone/tests/common/cpu_fixture.rs"]
mod cpu_fixture_support;

#[cfg(test)]
mod session_store_tests {
    use crate::service_memory::{HostMemoryObservation, HostMemorySemantics, HostMemorySource};
    use std::net::TcpListener;
    use std::num::NonZeroU64;

    #[test]
    fn cancelled_prefill_has_explicit_zero_progress() {
        let output =
            super::batch_prefill_progress(super::DispatchKind::Prefill, None, true).unwrap();
        assert_eq!(output.processed_tokens, 0);
        assert!(!output.ready);
        assert!(super::batch_prefill_progress(super::DispatchKind::Decode, None, true).is_none());
    }

    #[test]
    fn context_limits_reject_output_overflow() {
        let request: super::ChatRequest =
            serde_json::from_str(r#"{"model":"test","messages":[],"max_tokens":4}"#).unwrap();
        assert_eq!(super::chat_output_budget(&request, 5, 8).unwrap(), 4);
        assert!(super::chat_output_budget(&request, 6, 8).is_err());
        assert!(super::chat_output_budget(&request, 9, 8).is_err());
        assert_eq!(super::serve_context_limit(Some(8), 16).unwrap(), 8);
        assert!(super::serve_context_limit(Some(17), 16).is_err());
        assert!(super::serve_context_limit(Some(0), 16).is_err());
    }

    use super::*;

    #[derive(serde::Deserialize)]
    struct TemplateFixtureDocument {
        schema_version: String,
        contract: String,
        generator: TemplateFixtureGenerator,
        architectures: BTreeMap<String, TemplateFixtureArchitecture>,
        cases: Vec<TemplateFixture>,
    }

    #[derive(serde::Deserialize)]
    struct TemplateFixtureGenerator {
        path: String,
        engine: String,
        llama_cpp_commit: Option<String>,
        source: String,
        date_string: String,
    }

    #[derive(serde::Deserialize)]
    struct TemplateFixtureArchitecture {
        model_source: String,
        model_sha256: String,
        template_source: String,
        template_sha256: String,
        embedded_template_sha256: String,
    }

    #[derive(serde::Deserialize)]
    struct TemplateFixture {
        name: String,
        architecture: String,
        messages: Vec<Message>,
        #[serde(default)]
        tools: Option<Vec<ToolDefinition>>,
        #[serde(default)]
        tool_choice: Option<Value>,
        expected_tokens: Vec<u32>,
        token_count: usize,
        rendered_sha256: String,
    }

    #[derive(serde::Deserialize)]
    struct LegacyTemplateFixtureDocument {
        cases: Vec<LegacyTemplateFixture>,
    }

    #[derive(serde::Deserialize)]
    struct LegacyTemplateFixture {
        name: String,
        architecture: String,
        messages: Vec<Message>,
        #[serde(default)]
        tools: Option<Vec<ToolDefinition>>,
        #[serde(default)]
        tool_choice: Option<Value>,
        expected_tokens: Vec<u32>,
        token_count: usize,
    }

    #[test]
    fn chat_rejects_unknown_controls_and_invalid_sampling_before_execution() {
        let base = json!({"model": "leone", "messages": [{"role": "user", "content": "Hello"}]});
        let mut unknown = base.clone();
        unknown["ignored_control"] = json!(true);
        assert!(serde_json::from_value::<ChatRequest>(unknown).is_err());
        for (field, value) in [("top_p", 0.0), ("min_p", 1.1), ("repetition_penalty", -1.0)] {
            let mut body = base.clone();
            body[field] = json!(value);
            let request: ChatRequest = serde_json::from_value(body).unwrap();
            assert!(validate_chat_request(&request, "leone").is_err(), "{field}");
        }
    }

    #[test]
    fn nested_json_duplicate_fields_are_rejected_before_deserialization() {
        assert!(reject_duplicate_json_keys(
            br#"{"response_format":{"type":"text","type":"json_object"}}"#
        )
        .is_err());
        assert!(reject_duplicate_json_keys(
            br#"{"tool_choice":{"type":"function","function":{"name":"weather"}}}"#
        )
        .is_ok());
    }

    #[test]
    fn chat_template_mode_is_explicit_and_defaults_to_legacy() {
        let base = json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let request: ChatRequest = serde_json::from_value(base.clone()).expect("default mode");
        assert_eq!(request.leone_template, ChatTemplateMode::Legacy);
        let mut official = base;
        official["leone_template"] = json!("official");
        let request: ChatRequest = serde_json::from_value(official).expect("official mode");
        assert_eq!(request.leone_template, ChatTemplateMode::Official);
        let mut invalid = json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "leone_template": "other"
        });
        assert!(serde_json::from_value::<ChatRequest>(invalid.take()).is_err());
    }

    #[test]
    fn session_identifiers_are_bounded_and_header_safe() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "leone_session": "client-session",
            "leone_fork_session": "parent-session",
            "user": "ordinary-user"
        }))
        .expect("session request");
        let mut headers = BTreeMap::new();
        headers.insert("x-leone-session".to_owned(), "client-session".to_owned());
        let http = HttpRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: headers.clone(),
            body: Vec::new(),
        };
        validate_session_identifiers(&request, &http).expect("valid identifiers");
        assert_eq!(
            requested_chat_session(&request, &http),
            Some("client-session")
        );
        headers.insert("x-leone-session".to_owned(), "header-session".to_owned());
        let conflicting = HttpRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers,
            body: Vec::new(),
        };
        assert!(validate_session_identifiers(&request, &conflicting).is_err());
        let user_only: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "user": "ordinary-user"
        }))
        .expect("user metadata request");
        let no_session_http = HttpRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: BTreeMap::new(),
            body: Vec::new(),
        };
        assert_eq!(requested_chat_session(&user_only, &no_session_http), None);

        let mut injected = request;
        injected.leone_session = Some("ok\r\nX-Injected: yes".to_owned());
        assert!(validate_session_identifiers(&injected, &http).is_err());
        injected.leone_session = Some("x".repeat(MAX_SESSION_ID_BYTES + 1));
        assert!(validate_session_identifiers(&injected, &http).is_err());

        let mut empty_header = http;
        injected.leone_session = Some("valid-session".to_owned());
        empty_header
            .headers
            .insert("x-leone-session".to_owned(), String::new());
        assert!(validate_session_identifiers(&injected, &empty_header).is_err());
        injected.leone_session = Some(" padded".to_owned());
        assert!(validate_session_identifiers(&injected, &no_session_http).is_err());
        injected.leone_session = Some("comma,id".to_owned());
        assert!(validate_session_identifiers(&injected, &no_session_http).is_err());
    }

    #[test]
    fn chat_sampling_controls_reach_the_runtime_policy() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone", "messages": [{"role": "user", "content": "Hello"}],
            "top_k": 8, "top_p": 0.9, "min_p": 0.1, "top_a": 0.2,
            "tfs_z": 0.8, "typical_p": 0.7,
            "repetition_penalty": 1.2, "repetition_window": 64
        }))
        .unwrap();
        let observed = sampler(&request).unwrap();
        assert_eq!(
            observed.truncations,
            vec![
                Truncation::TopK(NonZeroUsize::new(8).unwrap()),
                Truncation::TopP(0.9),
                Truncation::MinP(0.1),
                Truncation::TopA(0.2),
                Truncation::TailFree(0.8),
                Truncation::Typical(0.7),
            ]
        );
        let observed = penalties(&request).unwrap();
        assert_eq!(observed.repetition, 1.2);
        assert_eq!(observed.window, PenaltyWindow::from_size(64));
    }

    #[test]
    fn openai_penalties_accept_only_the_documented_range() {
        for field in ["presence_penalty", "frequency_penalty"] {
            for value in [-2.0, 2.0] {
                let request: ChatRequest = serde_json::from_value(json!({
                    "model": "leone",
                    "messages": [{"role": "user", "content": "Hello"}],
                    field: value
                }))
                .expect("penalty request");
                validate_chat_request(&request, "leone").expect("boundary penalty");
            }
            for value in [-2.01, 2.01] {
                let request: ChatRequest = serde_json::from_value(json!({
                    "model": "leone",
                    "messages": [{"role": "user", "content": "Hello"}],
                    field: value
                }))
                .expect("penalty request");
                assert!(validate_chat_request(&request, "leone").is_err());
            }
        }
    }

    #[test]
    fn chat_template_fixtures_pin_independent_sources() {
        let fixture: TemplateFixtureDocument = serde_json::from_str(include_str!(
            "../../../fixtures/openai-chat-template-tokens.json"
        ))
        .expect("template fixtures");
        assert_eq!(
            fixture.schema_version,
            "leone.openai-chat-template-fixtures.v3"
        );
        assert_eq!(fixture.contract, "official-llama.cpp");
        assert_eq!(
            fixture.generator.path,
            "scripts/generate-openai-chat-template-fixtures.py"
        );
        assert!(fixture.generator.engine.contains("/apply-template"));
        assert_eq!(
            fixture.generator.llama_cpp_commit.as_deref(),
            Some(include_str!("../../../external/PINNED").trim())
        );
        assert!(fixture.generator.source.contains("tokenizer_config.json"));
        assert_eq!(fixture.generator.date_string, "26 Jul 2024");
        assert_eq!(fixture.cases.len(), 9);
        assert!(fixture
            .cases
            .iter()
            .any(|case| case.name == "llama_tool_history"));
        for case in &fixture.cases {
            assert!(fixture.architectures.contains_key(&case.architecture));
            assert!(!case.name.is_empty());
            assert_eq!(case.expected_tokens.len(), case.token_count);
            assert!(!case.expected_tokens.is_empty());
            assert_eq!(case.rendered_sha256.len(), 64);
        }
        for architecture in ["qwen3", "llama3"] {
            let source = fixture
                .architectures
                .get(architecture)
                .expect("architecture provenance");
            assert!(source.model_source.starts_with("https://huggingface.co/"));
            assert!(source.template_source.starts_with("https://"));
            let (model, template, embedded) = match architecture {
                "qwen3" => (
                    "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
                    "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                    "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361",
                ),
                "llama3" => (
                    "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146",
                    "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
                    "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
                ),
                _ => unreachable!(),
            };
            assert_eq!(source.model_sha256, model);
            assert_eq!(source.template_sha256, template);
            assert_eq!(source.embedded_template_sha256, embedded);
        }
    }

    #[test]
    #[ignore = "requires the pinned GGUF fixture models"]
    fn leone_renderer_matches_its_pinned_tokenizers() {
        let model_dir = fixture_model_dir();
        let qwen = load_fixture_tokenizer(
            &model_dir.join("Qwen3-8B-Q4_K_M.gguf"),
            leone::ModelArchitecture::Qwen3,
        );
        let llama = load_fixture_tokenizer(
            &model_dir.join("Llama-3.2-1B-Instruct-f16.gguf"),
            leone::ModelArchitecture::Llama,
        );
        let fixture: TemplateFixtureDocument = serde_json::from_str(include_str!(
            "../../../fixtures/openai-chat-template-leone.json"
        ))
        .expect("template fixtures");
        assert_eq!(fixture.contract, "leone-v0.4-legacy-renderer");
        for case in fixture.cases {
            let (tokenizer, architecture) = match case.architecture.as_str() {
                "qwen3" => (&qwen, leone::ModelArchitecture::Qwen3),
                "llama3" => (&llama, leone::ModelArchitecture::Llama),
                architecture => panic!("unknown fixture architecture {architecture}"),
            };
            let actual = chat_tokens(
                tokenizer,
                architecture,
                &case.messages,
                case.tools.as_deref(),
                case.tool_choice.as_ref(),
                ChatTemplateMode::Legacy,
            )
            .expect("render template");
            assert_eq!(actual, case.expected_tokens, "{} template", case.name);
        }
    }

    #[test]
    #[ignore = "requires the pinned GGUF fixture models"]
    fn legacy_renderer_matches_independent_reasoning_history() {
        let model_dir = fixture_model_dir();
        let qwen = load_fixture_tokenizer(
            &model_dir.join("Qwen3-8B-Q4_K_M.gguf"),
            leone::ModelArchitecture::Qwen3,
        );
        let fixture: LegacyTemplateFixtureDocument = serde_json::from_str(include_str!(
            "../../../fixtures/openai-chat-template-legacy-reasoning.json"
        ))
        .expect("template fixtures");
        for name in ["qwen_reasoning_history", "qwen_reasoning_markers_history"] {
            let case = fixture
                .cases
                .iter()
                .find(|case| case.name == name)
                .unwrap_or_else(|| panic!("missing {name} fixture"));
            assert_eq!(case.architecture, "qwen3", "{name} architecture");
            let actual = chat_tokens(
                &qwen,
                leone::ModelArchitecture::Qwen3,
                &case.messages,
                case.tools.as_deref(),
                case.tool_choice.as_ref(),
                ChatTemplateMode::Legacy,
            )
            .expect("render reasoning history");
            assert_eq!(actual.len(), case.token_count, "{name} count");
            assert_eq!(actual, case.expected_tokens, "{name} template");
        }
    }

    #[test]
    #[ignore = "requires the pinned GGUF fixture models"]
    fn official_renderer_matches_independent_fixture() {
        let model_dir = fixture_model_dir();
        let qwen = load_fixture_tokenizer(
            &model_dir.join("Qwen3-8B-Q4_K_M.gguf"),
            leone::ModelArchitecture::Qwen3,
        );
        let llama = load_fixture_tokenizer(
            &model_dir.join("Llama-3.2-1B-Instruct-f16.gguf"),
            leone::ModelArchitecture::Llama,
        );
        let fixture: TemplateFixtureDocument = serde_json::from_str(include_str!(
            "../../../fixtures/openai-chat-template-tokens.json"
        ))
        .expect("template fixtures");
        for case in fixture.cases {
            let (tokenizer, architecture) = match case.architecture.as_str() {
                "qwen3" => (&qwen, leone::ModelArchitecture::Qwen3),
                "llama3" => (&llama, leone::ModelArchitecture::Llama),
                architecture => panic!("unknown fixture architecture {architecture}"),
            };
            let actual = chat_tokens(
                tokenizer,
                architecture,
                &case.messages,
                case.tools.as_deref(),
                case.tool_choice.as_ref(),
                ChatTemplateMode::Official,
            )
            .expect("render official template");
            assert_eq!(actual, case.expected_tokens, "{} template", case.name);
        }
    }

    fn fixture_model_dir() -> PathBuf {
        std::env::var_os("LEONE_CHAT_FIXTURE_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models")
            })
    }

    fn load_fixture_tokenizer(
        path: &Path,
        architecture: leone::ModelArchitecture,
    ) -> leone::Tokenizer {
        let gguf = leone_gguf::Gguf::open(path).expect("fixture model");
        let model_architecture = match gguf.metadata().get("general.architecture") {
            Some(leone_gguf::MetadataValue::String(value)) => value,
            _ => panic!("fixture model architecture metadata"),
        };
        let expected = match architecture {
            leone::ModelArchitecture::Qwen3 => "qwen3",
            leone::ModelArchitecture::Llama => "llama",
        };
        assert_eq!(model_architecture, expected);
        leone::Tokenizer::from_metadata(gguf.metadata()).expect("fixture tokenizer")
    }

    const MODEL: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn cpu_fixture() -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("CPU GGUF fixture");
        file.write_all(&super::cpu_fixture_support::bytes())
            .expect("write CPU GGUF fixture");
        file
    }

    fn cpu_options(max_tokens: usize) -> GenerateOptions {
        let mut options = GenerateOptions::greedy(max_tokens);
        options.decode_execution = leone::DecodeExecution::Eager;
        options.prefill_chunk_tokens = 4;
        options
    }

    fn install_test_archives<B: Backend>(
        server: &mut Server<B>,
        archives: Vec<LoadedSessionArchive>,
    ) {
        for loaded in archives {
            server
                .persisted
                .insert(
                    loaded.session_id,
                    PersistedArchiveEntry {
                        archive: loaded.archive,
                        _allocation: loaded.allocation,
                    },
                )
                .expect("persisted archive table capacity");
        }
    }

    fn test_server_memory() -> ServerMemory {
        let limit = NonZeroU64::new(1 << 40).expect("test memory limit");
        let policy = resolve_policy(
            ServiceBudgetArgs {
                memory: BudgetRequest::Bytes(limit),
                host: BudgetRequest::Bytes(limit),
                kv_reservation: BudgetRequest::Bytes(limit),
            },
            ServiceMemoryInputs {
                topology: PolicyMemoryTopology::Cpu,
                backend: BackendCapacityObservation::Cpu,
                host: HostMemoryObservation::Available {
                    total_bytes: 1 << 50,
                    available_bytes: 1 << 50,
                    source: HostMemorySource::LinuxMeminfo,
                    semantics: HostMemorySemantics::KernelAvailableEstimate,
                },
                backend_owned_bytes: 0,
                host_owned_bytes: 0,
            },
        )
        .expect("test memory policy");
        let mut memory = ServerMemory::with_topology(
            policy,
            PolicyMemoryTopology::Cpu,
            HostMemoryLedger::new(policy.host_budget()),
        );
        memory.logical_kv_reservation = 1 << 20;
        memory
    }

    fn shared_test_server_memory(limit: NonZeroU64) -> ServerMemory {
        let policy = resolve_policy(
            ServiceBudgetArgs {
                memory: BudgetRequest::Bytes(limit),
                host: BudgetRequest::Bytes(limit),
                kv_reservation: BudgetRequest::Auto,
            },
            ServiceMemoryInputs {
                topology: PolicyMemoryTopology::Cpu,
                backend: BackendCapacityObservation::Cpu,
                host: HostMemoryObservation::Available {
                    total_bytes: limit.get(),
                    available_bytes: limit.get(),
                    source: HostMemorySource::LinuxMeminfo,
                    semantics: HostMemorySemantics::KernelAvailableEstimate,
                },
                backend_owned_bytes: 0,
                host_owned_bytes: 0,
            },
        )
        .expect("shared test memory policy");
        let root = MemoryTrackerRoot::new(policy.shared_budget().expect("shared budget"));
        let host = HostMemoryLedger::child(policy.host_budget(), root);
        ServerMemory::with_topology(policy, PolicyMemoryTopology::Cpu, host)
    }

    fn fill_host_budget(server: &Server<CpuBackend>) -> MemoryAllocation {
        let accounting = server.memory.host.snapshot();
        let MemoryBudget::Bytes(limit) = accounting.budget else {
            panic!("test host budget is finite");
        };
        let owned = accounting
            .live_bytes
            .checked_add(accounting.reserved_bytes)
            .expect("host accounting sum");
        server
            .memory
            .host
            .allocate(
                limit
                    .get()
                    .checked_sub(owned)
                    .expect("available host budget"),
            )
            .expect("fill host budget")
    }

    fn cpu_server(model: &Path) -> Server<CpuBackend> {
        let memory = test_server_memory();
        let persisted_tables =
            allocate_persisted_cache_tables(&memory, 16).expect("persisted cache tables");
        Server {
            runtime: Runtime::load(CpuBackend::new(), model).expect("load CPU fixture"),
            model_id: "fixture".to_owned(),
            model_sha256: MODEL.to_owned(),
            sessions: HashMap::new(),
            hibernated: HashMap::new(),
            persisted: persisted_tables.entries,
            _persisted_table_allocation: persisted_tables.allocation,
            persisted_cache_limit: 16,
            session_store: None,
            max_sessions: 8,
            max_hibernated_sessions: 8,
            clock: 0,
            kv_cache_dtype: KvCacheDtype::F16,
            #[cfg(feature = "cuda")]
            execution_plan: None,
            prefill_chunk_tokens: 4,
            context_limit: 64,
            receipts: PathBuf::new(),
            signing_key: SigningKey::from_bytes(&[0; 32]),
            memory,
            cors_origins: Vec::new(),
            proxy_origin: None,
            metrics: ServerMetrics::new().expect("test metrics"),
            identity: ServiceIdentity::new("test-service", MODEL, MODEL, 1).expect("test identity"),
            test_finalize_delay: None,
        }
    }

    fn resident_cpu_source(server: &mut Server<CpuBackend>, id: &str, options: GenerateOptions) {
        let prompt = [1, 2, 3, 4];
        let mut generation = GenerationSession::new();
        server
            .runtime
            .generate_session_tokens(&mut generation, &prompt, options, |_| Ok(()), || false)
            .expect("generate CPU fixture source");
        let mut stored = StoredSession {
            generation,
            metadata_allocation: None,
            reference_lease: None,
            last_used: 1,
        };
        let context_tokens = stored.generation.evaluated_tokens().len();
        server
            .ensure_resident_metadata(&mut stored, context_tokens)
            .expect("resident metadata lease");
        server.sessions.insert(id.to_owned(), stored);
    }

    fn populated_cpu_checkpoint() -> leone::GenerationCheckpoint {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        resident_cpu_source(&mut server, "checkpoint", cpu_options(1));
        server
            .sessions
            .remove("checkpoint")
            .expect("populated checkpoint")
            .generation
            .checkpoint()
    }

    fn persisted_cpu_server_default(label: &str) -> (PathBuf, Server<CpuBackend>) {
        let root = std::env::temp_dir().join(format!("leone-{label}-{}", Uuid::new_v4()));
        let (store, persisted, clock) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist session");
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        server.session_store = Some(store);
        install_test_archives(&mut server, persisted);
        server.clock = clock;
        (root, server)
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn idle_retirement_keeps_resident_state_when_checkpoint_lookup_fails() {
        let (root, mut server) = persisted_cpu_server_default("idle-lookup-failure");
        resident_cpu_source(&mut server, "session", cpu_options(1));
        let expected = server.sessions["session"]
            .generation
            .evaluated_tokens()
            .to_vec();
        let reference = server
            .session_store
            .as_ref()
            .expect("store")
            .reference_path("session");
        fs::write(reference, b"invalid reference").expect("corrupt reference");

        let error = server
            .discard_idle_session("session")
            .expect_err("lookup fails");
        assert_eq!(
            error.session_failure_effect(),
            Some(SessionFailureEffect::Unchanged)
        );
        assert_eq!(
            server.sessions["session"].generation.evaluated_tokens(),
            expected
        );
        drop(server);
        fs::remove_dir_all(root).expect("remove store");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn idle_retirement_requires_a_persisted_checkpoint() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        resident_cpu_source(&mut server, "session", cpu_options(1));
        assert!(!server
            .discard_idle_session("session")
            .expect("lookup completes"));
        assert!(server.sessions.contains_key("session"));
    }

    fn assert_session_absent_after_restart(root: PathBuf, server: Server<CpuBackend>) {
        drop(server);
        let (_, reopened, _) = SessionStore::open(root.clone(), 8).expect("restart store");
        assert!(reopened.is_empty());
        assert_eq!(
            fs::read_dir(root.join("quarantine"))
                .expect("quarantine directory")
                .count(),
            0
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    fn test_chat_plan(
        session_id: &str,
        source: ChatPlanSource,
        prompt_tokens: &[u32],
        options: GenerateOptions,
    ) -> ChatPlan {
        ChatPlan {
            request: serde_json::from_value(json!({
                "model": "fixture",
                "messages": []
            }))
            .expect("fixture request"),
            stop_sequences: Vec::new(),
            prompt_tokens: prompt_tokens.to_vec(),
            options,
            session_id: session_id.to_owned(),
            source,
            prefix_reused_tokens: 0,
            request_sha256: String::new(),
            created: 0,
            completion_id: String::new(),
            cors_origin: None,
        }
    }

    #[test]
    fn identical_sessions_share_one_immutable_blob() {
        let root = std::env::temp_dir().join(format!("leone-session-store-{}", Uuid::new_v4()));
        let (store, loaded, _) = SessionStore::open(root.clone(), 8).expect("new store");
        assert!(loaded.is_empty());
        let checkpoint = GenerationSession::<CpuBackend>::new().checkpoint();

        store
            .persist("left", MODEL, checkpoint.clone(), 1)
            .expect("left session");
        store
            .persist("right", MODEL, checkpoint, 2)
            .expect("right session");

        assert_eq!(fs::read_dir(root.join("blobs")).expect("blobs").count(), 1);
        let (_, reopened, _) = SessionStore::open(root.clone(), 8).expect("reopen store");
        assert_eq!(reopened.len(), 2);
        fs::remove_dir_all(root).expect("remove test store");
    }

    fn archive_json(ids: impl Iterator<Item = u32>, tokens: usize) -> Vec<u8> {
        let ids: Vec<String> = ids.map(|id| id.to_string()).collect();
        format!(
            r#"{{"schema_version":1,"model_sha256":"{MODEL}","evaluated_tokens":[{}],"prefill_boundary":{tokens},"mirostat":null}}"#,
            ids.join(",")
        )
        .into_bytes()
    }

    #[test]
    fn persist_bound_covers_the_checkpoint_clone_and_serializer_capacity() {
        let tokens = 150_000;
        let cases = [
            archive_json((0..tokens as u32).map(|index| 100_000 + index), tokens),
            archive_json(std::iter::repeat_n(u32::MAX, tokens), tokens),
        ];
        for json in cases {
            let archive = leone::SessionArchive::from_json(&json, MODEL).expect("archive");
            let blob = archive.to_json().expect("JSON");
            let peak = 4 * tokens + blob.capacity();

            let bound = session_archive_persist_bound(tokens).expect("bound");
            assert!(peak as u64 <= bound, "peak {peak} exceeds bound {bound}");
            assert!(blob.len() as u64 <= session_archive_json_bound(tokens).unwrap());
        }
    }

    #[test]
    fn metal_choice_reads_unbounded_capacity_as_unavailable() {
        let observation = observe_backend_capacity(&mut CpuBackend::new(), BackendChoice::Metal)
            .expect("observe");

        assert_eq!(
            observation,
            BackendCapacityObservation::Unavailable {
                reason: ObservationUnavailable::SourceUnavailable
            }
        );
    }

    #[test]
    fn post_rename_sync_failure_retains_a_recoverable_reference_lease() {
        let root = std::env::temp_dir().join(format!("leone-session-recovery-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let checkpoint = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("recover", MODEL, checkpoint, 1)
            .expect("persist session");
        let refs = root.join("refs");
        let removed = std::cell::Cell::new(false);
        let fail_sync = |path: &Path| {
            if path == refs && !removed.replace(true) {
                fs::remove_dir_all(&refs).expect("remove refs directory");
            }
            Err(io::Error::other("injected directory sync failure"))
        };
        assert!(store
            .lease_reference_with_sync("recover", &fail_sync)
            .is_err());
        fs::create_dir_all(&refs).expect("recreate refs directory");
        let mut lease = store
            .lease_reference("recover")
            .expect("recovery lookup")
            .expect("retained recovery lease");
        assert_eq!(lease.state, SessionReferenceLeaseState::Restore);
        lease.restore().expect("restore recovered reference");
        assert!(
            regular_reference_exists(&store.reference_path("recover")).expect("reference check")
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn reference_recovery_capacity_blocks_a_second_lease() {
        let root =
            std::env::temp_dir().join(format!("leone-session-recovery-cap-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 1).expect("new store");
        let checkpoint = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("left", MODEL, checkpoint.clone(), 1)
            .expect("left session");
        store
            .persist("right", MODEL, checkpoint, 2)
            .expect("right session");
        let mut left = store
            .lease_reference("left")
            .expect("left lease")
            .expect("left reference");
        assert!(store.lease_reference("right").is_err());
        assert!(regular_reference_exists(&store.reference_path("right")).expect("right reference"));
        left.restore().expect("restore left");
        drop(left);
        let mut right = store
            .lease_reference("right")
            .expect("right lease")
            .expect("right reference");
        right.restore().expect("restore right");
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn batch_lookup_failure_retains_the_task_held_reference() {
        let (root, server) = persisted_cpu_server("session-batch-lookup", 8);
        let reference = server
            .session_store
            .as_ref()
            .expect("session store")
            .reference_path("session");
        let (transport, client, incoming) = test_server_incoming();
        let mut task = test_chat_task(incoming, false, "session");
        task.stored.reference_lease = server
            .lease_session_reference("session")
            .expect("lease task reference");
        let trace = ServiceTraceRecorder::new(
            &server.runtime,
            MetricsMemoryTopology::BackendChildFallback,
            None,
        );
        let mut executor = ServerExecutor {
            server,
            tasks: BTreeMap::from([(RequestId(1), task)]),
            wire_request_ids: BTreeMap::new(),
            queued_outputs: BTreeMap::new(),
            terminal_outputs: VecDeque::new(),
            trace,
        };
        let dispatch = |request_id| Dispatch {
            request_id: RequestId(request_id),
            token_budget: 1,
            dispatch_ns: 0,
            kind: DispatchKind::Decode,
        };

        let error = executor
            .execute_batch(&[dispatch(1), dispatch(2)])
            .expect_err("reject unknown batch request");
        assert_eq!(error.to_string(), "unknown request 2");
        let mut task = executor
            .tasks
            .remove(&RequestId(1))
            .expect("restore prior batch task");
        let mut lease = task
            .stored
            .reference_lease
            .take()
            .expect("retain task reference lease");
        lease.restore().expect("restore task reference");
        assert!(regular_reference_exists(&reference).expect("active reference"));
        assert_eq!(
            fs::read_dir(root.join("quarantine"))
                .expect("quarantine directory")
                .count(),
            0
        );

        drop(lease);
        drop(task);
        drop(client);
        drop(transport);
        drop(executor);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn session_store_error_returns_a_private_server_error() {
        let (root, mut server) = persisted_cpu_server("session-private-error", 8);
        let reference = server
            .session_store
            .as_ref()
            .expect("session store")
            .reference_path("session");
        let mut value: Value = serde_json::from_slice(
            &fs::read(&reference).expect("read persisted session reference"),
        )
        .expect("parse persisted session reference");
        value["schema_version"] = json!(u32::MAX);
        fs::write(
            &reference,
            serde_json::to_vec(&value).expect("encode corrupt reference"),
        )
        .expect("write corrupt reference");
        let (transport, mut client, incoming) = test_server_incoming();
        let mut pending = test_pending_chat(incoming, "session");

        let error = match lease_pending_chat(&mut server, &mut pending) {
            Ok(_) => panic!("corrupt persisted session was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), SESSION_PREPARATION_ERROR);
        drop(pending);
        let response = read_test_response(&mut client);
        assert!(response.starts_with("HTTP/1.1 500 Error\r\n"));
        assert!(response.contains("\"type\":\"server_error\""));
        assert!(response.contains(SESSION_PREPARATION_ERROR));
        assert!(!response.contains(&root.display().to_string()));
        assert!(!response.contains("unsupported session reference schema"));

        drop(transport);
        drop(server);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn session_recovery_capacity_returns_a_typed_overload() {
        let (root, mut server) = persisted_cpu_server("session-capacity-error", 1);
        let held = {
            let store = server.session_store.as_ref().expect("session store");
            store
                .persist(
                    "held",
                    MODEL,
                    GenerationSession::<CpuBackend>::new().checkpoint(),
                    2,
                )
                .expect("persist held session");
            store
                .lease_reference("held")
                .expect("lease held reference")
                .expect("held reference")
        };
        let (transport, mut client, incoming) = test_server_incoming();
        let mut pending = test_pending_chat(incoming, "session");

        let error = match lease_pending_chat(&mut server, &mut pending) {
            Ok(_) => panic!("exhausted recovery capacity was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), SESSION_PREPARATION_ERROR);
        drop(pending);
        let response = read_test_response(&mut client);
        assert!(response.starts_with("HTTP/1.1 429 Error\r\n"));
        assert!(response.contains("\"type\":\"server_overloaded\""));
        assert!(response.contains("\"code\":\"session-capacity\""));
        assert!(!response.contains("session reference recovery capacity is full"));

        drop(transport);
        drop(held);
        drop(server);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn session_recovery_memory_returns_a_typed_overload() {
        let root =
            std::env::temp_dir().join(format!("leone-session-memory-capacity-{}", Uuid::new_v4()));
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        let (store, _, clock) = SessionStore::open_with_archive_limit_and_host(
            root.clone(),
            1,
            DEFAULT_SESSION_ARCHIVE_BYTES,
            server.memory.host.clone(),
        )
        .expect("new store");
        store
            .persist(
                "session",
                MODEL,
                GenerationSession::<CpuBackend>::new().checkpoint(),
                1,
            )
            .expect("persist session");
        server.session_store = Some(store);
        server.clock = clock;
        let host_capacity = fill_host_budget(&server);
        let (transport, mut client, incoming) = test_server_incoming();
        let mut pending = test_pending_chat(incoming, "session");

        let error = match lease_pending_chat(&mut server, &mut pending) {
            Ok(_) => panic!("exhausted session memory was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), SESSION_PREPARATION_ERROR);
        drop(pending);
        let response = read_test_response(&mut client);
        assert!(response.starts_with("HTTP/1.1 429 Error\r\n"));
        assert!(response.contains("\"type\":\"server_overloaded\""));
        assert!(response.contains("\"code\":\"session-capacity\""));
        assert!(!response.contains("allocation of"));

        drop(transport);
        drop(host_capacity);
        drop(server);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn recovery_table_bound_covers_growth_thresholds() {
        let entry = u64::try_from(std::mem::size_of::<(String, SessionReferenceLease)>())
            .expect("recovery table entry size");
        let cases = [1_usize, 4, 8, 15, 29];
        assert_eq!(recovery_table_bound(0).expect("empty table bound"), 0);
        for limit in cases {
            let expected = u64::try_from(limit)
                .expect("recovery table limit")
                .checked_mul(entry)
                .and_then(|bytes| bytes.checked_add(64))
                .expect("recovery table bound");
            assert_eq!(
                recovery_table_bound(limit).expect("recovery table bound"),
                expected
            );

            let table = RecoveryTable::with_capacity(limit).expect("reserve recovery table");
            assert_eq!(table.capacity(), limit);
        }
    }

    #[test]
    fn bounded_table_full_insert_returns_the_unstored_entry() {
        let mut table = BoundedTable::with_capacity(1).expect("bounded table");
        table.insert("stored".to_owned(), 1).expect("first entry");
        let (key, value) = table
            .insert("rejected".to_owned(), 2)
            .expect_err("full table");
        assert_eq!(key, "rejected");
        assert_eq!(value, 2);
        assert_eq!(table.len(), 1);
        assert_eq!(table.capacity(), 1);
    }

    #[test]
    fn recovery_table_capacity_stays_fixed_through_lease_churn() {
        let root = std::env::temp_dir().join(format!(
            "leone-session-recovery-table-churn-{}",
            Uuid::new_v4()
        ));
        let (store, _, _) = SessionStore::open(root.clone(), 16).expect("new store");
        let checkpoint = GenerationSession::<CpuBackend>::new().checkpoint();
        let session_ids: Vec<String> = (0..16).map(|index| format!("session-{index}")).collect();
        for (index, session_id) in session_ids.iter().enumerate() {
            store
                .persist(session_id, MODEL, checkpoint.clone(), index as u64)
                .expect("persist session");
        }
        for session_id in &session_ids {
            let lease = store
                .lease_reference(session_id)
                .expect("lease session")
                .expect("session reference");
            let mut lease = Some(lease);
            store
                .retain_recovery_lease(session_id, &mut lease)
                .expect("retain recovery lease");
        }
        let initial_capacity = store
            .recovery
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .capacity();
        for index in 0..128 {
            let session_id = &session_ids[index % session_ids.len()];
            let lease = store
                .take_recovery_lease(session_id)
                .expect("take recovery lease");
            let mut lease = Some(lease);
            store
                .retain_recovery_lease(session_id, &mut lease)
                .expect("retain churned recovery lease");
        }
        let final_capacity = store
            .recovery
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .capacity();
        assert_eq!(final_capacity, initial_capacity);
        drop(store);
        fs::remove_dir_all(root).expect("remove recovery table root");
    }

    #[test]
    fn persisted_cache_tables_hold_their_capacity_charge() {
        let cases = [1_usize, 3, 8, 15];
        for limit in cases {
            let memory = test_server_memory();
            let tables = allocate_persisted_cache_tables(&memory, limit)
                .expect("allocate persisted cache tables");
            assert_eq!(tables.entries.capacity(), limit);
            assert_eq!(
                memory.host.snapshot().live_bytes,
                persisted_cache_table_bound(limit).expect("persisted table bound")
            );
            drop(tables);
            assert_eq!(memory.host.snapshot().live_bytes, 0);
        }
    }

    #[test]
    fn reference_recovery_lease_charges_host_bytes() {
        let root =
            std::env::temp_dir().join(format!("leone-session-recovery-bytes-{}", Uuid::new_v4()));
        let lease_bytes = recovery_lease_bound_for_root(&root).expect("recovery lease bound");
        let table_bytes = recovery_table_bound(8).expect("recovery table bound");
        let host_limit = table_bytes
            .checked_add(lease_bytes)
            .expect("recovery host limit");
        let host = HostMemoryLedger::new(NonZeroU64::new(host_limit).expect("recovery host limit"));
        let (store, _, _) = SessionStore::open_with_archive_limit_and_host(
            root.clone(),
            8,
            DEFAULT_SESSION_ARCHIVE_BYTES,
            host.clone(),
        )
        .expect("new store");
        let checkpoint = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("left", MODEL, checkpoint.clone(), 1)
            .expect("left session");
        store
            .persist("right", MODEL, checkpoint, 2)
            .expect("right session");
        let left = store
            .lease_reference("left")
            .expect("left lease")
            .expect("left reference");
        assert!(host.snapshot().live_bytes >= table_bytes + lease_bytes);
        assert!(store.lease_reference("right").is_err());
        drop(left);
        assert_eq!(host.snapshot().live_bytes, table_bytes);
        drop(store);
        assert_eq!(host.snapshot().live_bytes, 0);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn bounded_reference_read_charges_and_releases_host_bytes() {
        let root = std::env::temp_dir().join(format!("leone-session-read-{}", Uuid::new_v4()));
        let table_bytes = recovery_table_bound(8).expect("recovery table bound");
        let archive = StoredArchive {
            blob_sha256: sha256_bytes(b"reference metadata"),
            last_used: 1,
        };
        let metadata_bytes =
            stored_archive_metadata_bound("reference", &archive).expect("archive metadata bound");
        let host_limit = table_bytes
            .checked_add(MAX_SESSION_REFERENCE_BYTES)
            .and_then(|bytes| bytes.checked_add(SESSION_REFERENCE_PARSE_BYTES))
            .and_then(|bytes| bytes.checked_add(metadata_bytes))
            .expect("reference read host limit");
        let host =
            HostMemoryLedger::new(NonZeroU64::new(host_limit).expect("reference read host limit"));
        let (store, _, _) = SessionStore::open_with_archive_limit_and_host(
            root.clone(),
            8,
            DEFAULT_SESSION_ARCHIVE_BYTES,
            host.clone(),
        )
        .expect("new store");
        store
            .persist(
                "reference",
                MODEL,
                GenerationSession::<CpuBackend>::new().checkpoint(),
                1,
            )
            .expect("persist reference");
        store
            .load_metadata("reference")
            .expect("load reference")
            .expect("reference metadata");
        let accounting = host.snapshot();
        assert_eq!(accounting.live_bytes, table_bytes);
        assert!(
            accounting.peak_live_bytes
                >= table_bytes + MAX_SESSION_REFERENCE_BYTES + SESSION_REFERENCE_PARSE_BYTES
        );
        drop(store);
        assert_eq!(host.snapshot().live_bytes, 0);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn reference_parse_reserves_its_peak_before_deserialization() {
        let root = std::env::temp_dir().join(format!("leone-session-parse-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("reference parse root");
        let session_id = "parse";
        let path = root.join(format!("{}.json", sha256_bytes(session_id.as_bytes())));
        let mut reference = SessionReference {
            schema_version: SessionStore::REFERENCE_SCHEMA_VERSION,
            session_id: String::new(),
            blob_sha256: sha256_bytes(b"parse"),
            last_used: 1,
        };
        let fixed_json = serde_json::to_vec(&reference).expect("reference JSON");
        let oversized_id_bytes = usize::try_from(MAX_SESSION_REFERENCE_BYTES)
            .expect("reference size")
            .checked_sub(fixed_json.len())
            .and_then(|bytes| bytes.checked_sub(1))
            .expect("near-limit session ID");
        reference.session_id = "x".repeat(oversized_id_bytes);
        let oversized = serde_json::to_vec(&reference).expect("oversized reference JSON");
        assert_eq!(
            oversized.len(),
            usize::try_from(MAX_SESSION_REFERENCE_BYTES).expect("reference size") - 1
        );
        fs::write(&path, oversized).expect("write oversized reference");

        let insufficient = MAX_SESSION_REFERENCE_BYTES
            .checked_add(SESSION_REFERENCE_PARSE_BYTES)
            .and_then(|bytes| bytes.checked_sub(1))
            .expect("insufficient parse budget");
        let host = HostMemoryLedger::new(NonZeroU64::new(insufficient).expect("parse budget"));
        let error = read_stored_session_reference_charged(&path, &host)
            .expect_err("reject missing parse reservation");
        assert!(io_session_capacity_error(&error));
        assert_eq!(host.snapshot().live_bytes, 0);

        let exact = insufficient.checked_add(1).expect("exact parse budget");
        let host = HostMemoryLedger::new(NonZeroU64::new(exact).expect("parse budget"));
        let error = read_stored_session_reference_charged(&path, &host)
            .expect_err("reject oversized session ID");
        assert!(error.to_string().contains(&format!(
            "stored session_id exceeds the {MAX_SESSION_ID_BYTES}-byte limit"
        )));
        assert_eq!(host.snapshot().live_bytes, 0);
        assert!(host.snapshot().peak_live_bytes >= exact);

        reference.session_id = session_id.to_owned();
        fs::write(
            &path,
            serde_json::to_vec(&reference).expect("reference JSON"),
        )
        .expect("write reference");
        let loaded = read_stored_session_reference_charged(&path, &host).expect("parse reference");
        assert_eq!(host.snapshot().live_bytes, SESSION_REFERENCE_PARSE_BYTES);
        drop(loaded);
        assert_eq!(host.snapshot().live_bytes, 0);
        fs::remove_dir_all(root).expect("remove reference parse root");
    }

    #[test]
    fn existing_blob_verification_charges_its_read_buffer() {
        let root = std::env::temp_dir().join(format!("leone-session-blob-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("blob test root");
        let blob = br#"{"schema_version":1}"#;
        let host = HostMemoryLedger::new(NonZeroU64::new(blob.len() as u64).expect("blob length"));
        let path = root.join("blob.json");
        write_session_blob(&path, blob, &host).expect("write blob");
        write_session_blob(&path, blob, &host).expect("verify blob");
        let accounting = host.snapshot();
        assert_eq!(accounting.live_bytes, 0);
        assert!(accounting.peak_live_bytes >= blob.len() as u64);
        fs::remove_dir_all(root).expect("remove blob test root");
    }

    #[test]
    fn startup_archive_metadata_has_a_host_lease() {
        let root =
            std::env::temp_dir().join(format!("leone-session-startup-memory-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let stored = store
            .persist(
                "startup",
                MODEL,
                GenerationSession::<CpuBackend>::new().checkpoint(),
                1,
            )
            .expect("persist startup archive");
        drop(store);

        let transition = startup_archive_transition_bound(1).expect("startup transition bound");
        let metadata_bytes = stored_archive_metadata_bound("startup", &stored)
            .expect("startup archive metadata bound");
        let table_bytes = recovery_table_bound(8).expect("recovery table bound");
        let host_limit = MAX_SESSION_REFERENCE_BYTES
            .checked_add(table_bytes)
            .and_then(|bytes| bytes.checked_add(SESSION_REFERENCE_PARSE_BYTES))
            .and_then(|bytes| bytes.checked_add(transition))
            .and_then(|bytes| bytes.checked_add(metadata_bytes))
            .expect("startup host limit");
        let metadata =
            HostMemoryLedger::new(NonZeroU64::new(host_limit).expect("startup host limit"));
        let (store, archives, _) = SessionStore::open_with_archive_limit_and_host(
            root.clone(),
            8,
            DEFAULT_SESSION_ARCHIVE_BYTES,
            metadata.clone(),
        )
        .expect("reload startup archive");
        assert_eq!(archives.len(), 1);
        assert!(store.startup_archive_transition.is_some());
        assert!(metadata.snapshot().live_bytes >= transition + metadata_bytes);
        drop(store);
        drop(archives);
        assert_eq!(metadata.snapshot().live_bytes, 0);
        fs::remove_dir_all(root).expect("remove startup store");
    }

    #[test]
    fn oversized_archive_is_rejected_before_parse() {
        let root =
            std::env::temp_dir().join(format!("leone-session-archive-limit-{}", Uuid::new_v4()));
        let (store, _, _) =
            SessionStore::open_with_archive_limit(root.clone(), 8, 1).expect("new bounded store");
        let checkpoint = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("bounded", MODEL, checkpoint, 1)
            .expect("persist archive");
        let stored = store
            .load_metadata("bounded")
            .expect("archive metadata")
            .expect("stored archive");
        assert!(store.load_archive("bounded", &stored, None, MODEL).is_err());
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn response_format_maps_json_object_to_the_runtime_constraint() {
        assert_eq!(
            response_constraint(Some(&json!({"type": "json_object"})))
                .expect("supported response format"),
            Some(OutputConstraint::JsonObject)
        );
        assert!(response_constraint(Some(&json!({"type": "json_schema"}))).is_err());
    }

    #[test]
    fn tool_prompt_and_response_use_typed_function_calls() {
        let tools = vec![ToolDefinition {
            kind: "function".to_owned(),
            function: ToolFunction {
                name: "weather".to_owned(),
                description: Some("Read the weather".to_owned()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
                strict: None,
            },
        }];
        let prompt = tool_system_prompt(Some(&tools), Some(&json!("required")))
            .expect("tool prompt")
            .expect("enabled tools");
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("<tool_call>\n{\"name\": <function-name>"));
        assert!(prompt.contains("\"name\": \"weather\""));

        let calls = parse_generated_tool_calls(
            r#"<tool_call>{"name":"weather","arguments":{"city":"Oslo"}}</tool_call>"#,
            Some(&tools),
            Some(&json!("required")),
        )
        .expect("valid generated call")
        .expect("tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Oslo"}"#);
        let native = parse_generated_tool_calls(
            r#"{"name":"weather","parameters":{"city":"Oslo"}}"#,
            Some(&tools),
            Some(&json!({"type": "function", "function": {"name": "weather"}})),
        )
        .expect("native generated call")
        .expect("native tool call");
        assert_eq!(native[0].function.name, "weather");
        assert_eq!(native[0].function.arguments, r#"{"city":"Oslo"}"#);
        let ordinary = parse_generated_tool_calls(
            r#"{"name":"Alice","city":"Oslo"}"#,
            Some(&tools),
            Some(&json!("auto")),
        )
        .expect("ordinary JSON content");
        assert!(ordinary.is_none());
        let truncated = parse_generated_tool_calls(
            r#"{"name":"Alice","city":"Oslo""#,
            Some(&tools),
            Some(&json!("auto")),
        )
        .expect("truncated ordinary JSON content");
        assert!(truncated.is_none());
        assert!(parse_generated_tool_calls(
            r#"{"name":"weather","arguments":{},"parameters":{}}"#,
            Some(&tools),
            Some(&json!({"type":"function", "function":{"name":"weather"}})),
        )
        .is_err());
        assert!(parse_generated_tool_calls(
            r#"{"name":"bad","name":"weather","arguments":{}}"#,
            Some(&tools),
            Some(&json!("required")),
        )
        .is_err());
        assert!(parse_generated_tool_calls(
            r#"<tool_call>{"name":"weather","arguments":{},"arguments":{}}</tool_call>"#,
            Some(&tools),
            Some(&json!("required")),
        )
        .is_err());
        assert!(parse_generated_tool_calls(
            r#"<tool_call>{"name":"weather","arguments":{},"extra":1}</tool_call>"#,
            Some(&tools),
            Some(&json!("required")),
        )
        .is_err());
        assert!(parse_generated_tool_calls(
            r#"<tool_call>{"name":"unknown","arguments":{}}</tool_call>"#,
            Some(&tools),
            Some(&json!("auto")),
        )
        .is_err());
        let rendered = render_request_tool_calls(
            "",
            &[RequestToolCall {
                id: "call_request".to_owned(),
                kind: "function".to_owned(),
                function: RequestToolCallFunction {
                    name: "weather".to_owned(),
                    arguments: r#"{"city":"Oslo"}"#.to_owned(),
                },
            }],
        )
        .expect("render request tool call");
        assert_eq!(
            rendered,
            "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\":\"Oslo\"}}\n</tool_call>"
        );

        let assistant = Message {
            role: "assistant".to_owned(),
            content: Value::Null,
            name: None,
            tool_calls: Some(vec![RequestToolCall {
                id: "call_request".to_owned(),
                kind: "function".to_owned(),
                function: RequestToolCallFunction {
                    name: "weather".to_owned(),
                    arguments: r#"{"city":"Oslo"}"#.to_owned(),
                },
            }]),
            tool_call_id: None,
            refusal: None,
            annotations: None,
            audio: None,
            function_call: None,
        };
        assert_eq!(
            render_message_content(&assistant, "assistant").expect("assistant tool message"),
            rendered
        );
        let escaped = render_request_tool_calls(
            "",
            &[RequestToolCall {
                id: "call_escaped".to_owned(),
                kind: "function".to_owned(),
                function: RequestToolCallFunction {
                    name: "weather\"quoted".to_owned(),
                    arguments: "{}".to_owned(),
                },
            }],
        )
        .expect("escape request tool name");
        assert!(escaped.contains("weather\\\"quoted"));
        serde_json::from_str::<Value>(
            escaped
                .strip_prefix("<tool_call>\n")
                .and_then(|value| value.strip_suffix("\n</tool_call>"))
                .expect("tool wrapper"),
        )
        .expect("escaped tool JSON");
    }

    #[test]
    fn stop_matcher_handles_utf8_and_token_boundaries() {
        let mut matcher = StopMatcher::new(vec!["🙂END".as_bytes().to_vec()]);
        let first = matcher.push("prefix🙂".as_bytes());
        assert_eq!(String::from_utf8(first).expect("UTF-8 prefix"), "prefix");
        let second = matcher.push("ENDsuffix".as_bytes());
        assert!(second.is_empty());
        assert!(matcher.hit());
        assert!(matcher.finish().is_empty());
    }

    #[test]
    fn stop_matcher_prefers_the_earliest_overlapping_match() {
        let mut matcher = StopMatcher::new(vec![b"ABCDE".to_vec(), b"ABC".to_vec()]);
        assert!(matcher.push(b"AB").is_empty());
        assert!(matcher.push(b"CDEx").is_empty());
        assert!(matcher.hit());
        assert!(matcher.finish().is_empty());

        let mut trailing = StopMatcher::new(vec![b"END".to_vec()]);
        assert_eq!(trailing.push(b"prefixENDtail"), b"prefix");
        assert!(trailing.hit());
        assert!(trailing.push(b"ignored").is_empty());
    }

    #[test]
    fn stop_sequences_accept_string_and_bounded_arrays() {
        let string: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "END"
        }))
        .expect("string stop");
        assert_eq!(
            request_stop_sequences(&string).expect("stop sequence"),
            vec![b"END".to_vec()]
        );

        let array: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["END", "DONE"]
        }))
        .expect("array stop");
        assert_eq!(
            request_stop_sequences(&array)
                .expect("stop sequences")
                .len(),
            2
        );

        let mut too_many = array;
        too_many.stop = Some(json!(["1", "2", "3", "4", "5"]));
        assert!(request_stop_sequences(&too_many).is_err());

        let mut empty = string;
        empty.stop = Some(json!(""));
        assert!(request_stop_sequences(&empty).is_err());
    }

    #[test]
    fn stop_truncation_reports_a_terminal_match() {
        let (content, matched) =
            truncate_at_stop("before-END-after", &[b"END".to_vec(), b"DONE".to_vec()]);
        assert_eq!(content, "before-");
        assert!(matched);
        let (content, matched) = truncate_at_stop("without marker", &[b"END".to_vec()]);
        assert_eq!(content, "without marker");
        assert!(!matched);
    }

    #[test]
    fn official_llama_tool_stop_is_token_scoped() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Call weather"}],
            "leone_template": "official",
            "tools": [{"type": "function", "function": {"name": "weather"}}]
        }))
        .expect("tool request");
        assert_eq!(
            official_llama_tool_header_token(&request, leone::ModelArchitecture::Llama, Some(41)),
            Some(41)
        );
        assert_eq!(
            official_llama_tool_header_token(&request, leone::ModelArchitecture::Qwen3, Some(41)),
            None
        );
        let auto_request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Call weather"}],
            "leone_template": "official",
            "tools": [{"type": "function", "function": {"name": "weather"}}],
            "tool_choice": "auto"
        }))
        .expect("auto tool request");
        assert_eq!(
            official_llama_tool_header_token(
                &auto_request,
                leone::ModelArchitecture::Llama,
                Some(41)
            ),
            Some(41)
        );
    }

    #[test]
    fn tool_header_token_stays_in_usage_but_not_content() {
        let generation = ChatGeneration {
            prompt_tokens: vec![1, 2],
            tokens: vec![7, 41],
            request_replay: Some(leone::SessionReplay {
                reuse_class: leone::SessionReuseClass::Cold,
                cached_tokens: 0,
                reused_tokens: 0,
                replayed_tokens: 0,
                computed_tokens: 2,
            }),
            cancelled: false,
            stop_hit: true,
            tool_header_stop: true,
            eos: true,
            prefill_chunks: 0,
            prefill_tokens: 0,
            decode_quanta: 1,
            phase_trace: vec![DispatchKind::Decode],
        };
        assert_eq!(generated_content_tokens(&generation), &[7]);
        assert_eq!(generation.tokens.len(), 2);
    }

    #[test]
    fn finish_reason_reports_stop_before_cancellation() {
        let generation = ChatGeneration {
            prompt_tokens: Vec::new(),
            tokens: vec![7],
            request_replay: Some(leone::SessionReplay {
                reuse_class: leone::SessionReuseClass::Cold,
                cached_tokens: 0,
                reused_tokens: 0,
                replayed_tokens: 0,
                computed_tokens: 0,
            }),
            cancelled: false,
            stop_hit: true,
            tool_header_stop: false,
            eos: true,
            prefill_chunks: 0,
            prefill_tokens: 0,
            decode_quanta: 1,
            phase_trace: vec![DispatchKind::Decode],
        };
        assert_eq!(chat_finish_reason(&generation, false), "stop");

        let mut cancelled = generation;
        cancelled.stop_hit = false;
        cancelled.eos = false;
        cancelled.cancelled = true;
        assert_eq!(chat_finish_reason(&cancelled, false), "cancelled");
        assert_eq!(chat_finish_reason(&cancelled, true), "cancelled");
    }

    #[test]
    fn stream_tool_chunks_use_openai_tool_delta_fields() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Call weather"}],
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "weather"}}]
        }))
        .expect("tool request");
        let context = ChatResponseContext {
            request: &request,
            session_id: "session".to_owned(),
            completion_id: "chatcmpl-test".to_owned(),
            created: 1,
            request_sha256: "request".to_owned(),
            seed: 0,
            model_id: "leone".to_owned(),
            cors_origin: None,
        };
        let call = GeneratedToolCall {
            id: "call_test".to_owned(),
            kind: "function",
            function: GeneratedToolCallFunction {
                name: "weather".to_owned(),
                arguments: "{\"city\":\"Oslo\"}".to_owned(),
            },
        };
        let chunk = tool_call_chunk(&context, 0, &call);
        assert_eq!(chunk["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(
            chunk["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_test"
        );
        assert_eq!(
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
    }

    #[test]
    fn stream_usage_flag_controls_usage_events() {
        let false_request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "stream_options": {"include_usage": false}
        }))
        .expect("stream options");
        assert!(!stream_usage_requested(&false_request));
        let absent_request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .expect("stream options absent");
        assert!(!stream_usage_requested(&absent_request));
        let null_request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "stream_options": {"include_usage": null}
        }))
        .expect("null stream option");
        assert!(!stream_usage_requested(&null_request));
        let true_request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .expect("stream options true");
        assert!(stream_usage_requested(&true_request));
        validate_chat_request(&false_request, "leone").expect("false include_usage is valid");
    }

    fn usage_stream_wire(request: ChatRequest) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind usage listener");
        let mut client = TcpStream::connect(listener.local_addr().expect("usage listener address"))
            .expect("connect usage client");
        let (mut server, _) = listener.accept().expect("accept usage client");
        write_chat_headers_io(&mut server, "session", "chatcmpl-usage", 1, "leone", None)
            .expect("write usage headers");
        let context = ChatResponseContext {
            request: &request,
            session_id: "session".to_owned(),
            completion_id: "chatcmpl-usage".to_owned(),
            created: 1,
            request_sha256: "0".repeat(64),
            seed: 0,
            model_id: "leone".to_owned(),
            cors_origin: None,
        };
        let generated = ChatGeneration {
            prompt_tokens: vec![1],
            tokens: vec![2],
            request_replay: Some(leone::SessionReplay {
                reuse_class: leone::SessionReuseClass::Cold,
                cached_tokens: 0,
                reused_tokens: 0,
                replayed_tokens: 0,
                computed_tokens: 1,
            }),
            cancelled: false,
            stop_hit: false,
            tool_header_stop: false,
            eos: true,
            prefill_chunks: 0,
            prefill_tokens: 0,
            decode_quanta: 1,
            phase_trace: vec![DispatchKind::Decode],
        };
        let details = ChatResponseDetails {
            decoded_content: Some("answer".to_owned()),
            generated_tool_calls: None,
            finish_reason: "stop".to_owned(),
        };
        let receipt = ResponseReceipt::sign(
            ResponseClaim {
                schema_version: RESPONSE_SCHEMA_VERSION,
                receipt_id: Uuid::new_v4(),
                created_utc: Utc::now(),
                engine_version: "test".to_owned(),
                model_sha256: "1".repeat(64),
                request_sha256: "2".repeat(64),
                prompt_tokens_sha256: "3".repeat(64),
                response_tokens_sha256: "4".repeat(64),
                transcript_sha256: "5".repeat(64),
                seed: 0,
                prompt_tokens: 1,
                generated_tokens: 1,
                finish_reason: "stop".to_owned(),
                cancelled: false,
                session: SessionReplayRecord {
                    session_id: "session".to_owned(),
                    reuse_class: "new".to_owned(),
                    cached_tokens: 0,
                    reused_tokens: 0,
                    replayed_tokens: 0,
                    computed_tokens: 1,
                },
            },
            &SigningKey::from_bytes(&[7; 32]),
        )
        .expect("sign usage receipt");
        let mut streamed = Utf8Stream::default();
        streamed.push(b"answer");
        write_stream_chat_response(
            &mut server,
            &context,
            &mut streamed,
            &generated,
            &details,
            &receipt,
        )
        .expect("write usage response");
        drop(server);
        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("read usage response");
        response
    }

    #[test]
    fn stream_usage_wire_matches_the_requested_flag() {
        let request = |include_usage| {
            serde_json::from_value(json!({
                "model": "leone",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true,
                "stream_options": {"include_usage": include_usage}
            }))
            .expect("usage request")
        };
        let with_usage = usage_stream_wire(request(true));
        let without_usage = usage_stream_wire(request(false));
        let default_usage = usage_stream_wire(
            serde_json::from_value(json!({
                "model": "leone",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true
            }))
            .expect("default usage request"),
        );
        let choices = with_usage.find("\"choices\":[]").expect("usage choices");
        let usage = with_usage
            .find("\"usage\":{\"completion_tokens\"")
            .expect("usage field");
        assert!(choices < usage);
        assert!(usage < with_usage.find("[DONE]").expect("usage done"));
        assert!(with_usage.contains("\"usage\":null"));
        assert!(with_usage.contains("\"usage\":{\"completion_tokens\""));
        assert!(without_usage.contains("\"usage\":null"));
        assert!(default_usage.contains("\"usage\":null"));
        assert!(!without_usage.contains("\"usage\":{\"completion_tokens\""));
        assert!(!default_usage.contains("\"usage\":{\"completion_tokens\""));
    }

    #[test]
    fn pending_utf8_stream_chunks_include_null_usage() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .expect("stream request");
        let context = ChatResponseContext {
            request: &request,
            session_id: "session".to_owned(),
            completion_id: "chatcmpl-pending".to_owned(),
            created: 1,
            request_sha256: String::new(),
            seed: 0,
            model_id: "leone".to_owned(),
            cors_origin: None,
        };
        let mut streamed = Utf8Stream::default();
        assert!(streamed.push(&[0xf0]).is_empty());
        let mut output = Vec::new();
        write_stream_pending(&mut output, &context, &mut streamed).expect("pending chunk");
        let output = String::from_utf8(output).expect("SSE UTF-8");
        assert!(output.contains("\"content\":\"�\""));
        assert!(output.contains("\"usage\":null"));
    }

    #[test]
    fn nonstream_utf8_decode_replaces_incomplete_bytes() {
        assert_eq!(decode_utf8_lossy(&[0xc3]), "�");
        assert_eq!(decode_utf8_lossy("Ålesund".as_bytes()), "Ålesund");
    }

    #[test]
    fn tool_choice_and_history_fields_are_strictly_enforced() {
        let base = json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Call weather"}],
            "tools": [{"type": "function", "function": {"name": "weather"}},
                       {"type": "function", "function": {"name": "time"}}]
        });
        let mut none = base.clone();
        none["tool_choice"] = json!("none");
        let none: ChatRequest = serde_json::from_value(none).expect("none choice");
        validate_chat_request(&none, "leone").expect("none choice");
        assert!(!effective_tool_mode(&none));
        assert!(validate_generated_tool_choice(Some(&json!("none")), true).is_err());

        let mut required = base.clone();
        required["tool_choice"] = json!("required");
        let required: ChatRequest = serde_json::from_value(required).expect("required choice");
        validate_chat_request(&required, "leone").expect("required choice");
        assert!(validate_generated_tool_choice(Some(&json!("required")), false).is_err());

        let mut named = base;
        named["tool_choice"] = json!({"type":"function", "function":{"name":"weather"}});
        let named: ChatRequest = serde_json::from_value(named).expect("named choice");
        validate_chat_request(&named, "leone").expect("named choice");
        assert!(parse_generated_tool_calls(
            r#"<tool_call>{"name":"time","arguments":{}}</tool_call>"#,
            named.tools.as_deref(),
            named.tool_choice.as_ref(),
        )
        .is_err());
        let mut unknown_choice = named;
        unknown_choice.tool_choice = Some(json!({
            "type": "function",
            "function": {"name": "weather", "unexpected": true}
        }));
        assert!(validate_chat_request(&unknown_choice, "leone").is_err());
        assert!(response_constraint(Some(&json!({
            "type": "text",
            "unexpected": true
        })))
        .is_err());

        let json_response: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}],
            "response_format": {"type": "json_object"}
        }))
        .expect("JSON response request");
        assert!(validate_chat_request(&json_response, "leone").is_err());

        assert!(
            parse_generated_tool_calls(r#"{"name":"Alice","city":"Oslo"}"#, None, None,)
                .expect("ordinary JSON response")
                .is_none()
        );
        assert!(parse_generated_tool_calls(
            r#"{"name":"Alice","city":"Oslo"}"#,
            Some(&[]),
            Some(&json!("none")),
        )
        .expect("ordinary JSON with disabled tools")
        .is_none());

        let empty_required: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Call"}],
            "tools": [],
            "tool_choice": "required"
        }))
        .expect("empty required request");
        assert!(validate_chat_request(&empty_required, "leone").is_err());

        let assistant: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "weather", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "name": "weather", "content": "{}"}
            ]
        }))
        .expect("assistant tool history without content");
        validate_chat_request(&assistant, "leone").expect("assistant tool history");

        let mut invalid_role = assistant;
        invalid_role.messages[0].role = "user".to_owned();
        assert!(validate_chat_request(&invalid_role, "leone").is_err());
        assert!(serde_json::from_value::<ChatRequest>(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "x", "unexpected": true}]
        }))
        .is_err());
    }

    #[test]
    fn tool_history_rejects_orphans_duplicates_and_missing_results() {
        let call = json!({
            "id": "call_1", "type": "function",
            "function": {"name": "weather", "arguments": "{}"}
        });
        let assistant = json!({"role": "assistant", "tool_calls": [call]});
        let valid = json!({
            "model": "leone",
            "messages": [assistant.clone(), {
                "role": "tool", "tool_call_id": "call_1", "name": "weather", "content": "{}"
            }]
        });
        let request: ChatRequest = serde_json::from_value(valid).expect("valid tool history");
        validate_chat_request(&request, "leone").expect("matching tool result");

        for messages in [
            json!([{"role": "tool", "tool_call_id": "call_1", "content": "{}"}]),
            json!([assistant.clone()]),
            json!([assistant.clone(), {
                "role": "tool", "tool_call_id": "call_other", "content": "{}"
            }]),
            json!([{
                "role": "assistant", "tool_calls": [call.clone(), call.clone()]
            }]),
            json!([assistant.clone(), {
                "role": "tool", "tool_call_id": "call_1", "name": "time", "content": "{}"
            }]),
            json!([assistant.clone(), {
                "role": "tool", "tool_call_id": "call_1", "name": "weather", "content": "{}"
            }, assistant.clone(), {
                "role": "tool", "tool_call_id": "call_1", "name": "weather", "content": "{}"
            }]),
        ] {
            let request: ChatRequest = serde_json::from_value(json!({
                "model": "leone",
                "messages": messages
            }))
            .expect("well-formed tool history");
            assert!(validate_chat_request(&request, "leone").is_err());
        }
    }

    #[test]
    fn tool_history_rejects_non_object_and_duplicate_arguments() {
        for arguments in ["[]", r#"{"city":"Oslo","city":"Bergen"}"#] {
            let request: ChatRequest = serde_json::from_value(json!({
                "model": "leone",
                "messages": [{
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": arguments}
                    }]
                }, {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "{}"
                }]
            }))
            .expect("well-formed encoded tool history");
            assert!(validate_chat_request(&request, "leone").is_err());
        }
    }

    #[test]
    fn official_llama_history_rejects_multiple_tool_calls() {
        let messages: Vec<Message> = serde_json::from_value(json!([{
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "weather", "arguments": "{}"
                }},
                {"id": "call_2", "type": "function", "function": {
                    "name": "time", "arguments": "{}"
                }}
            ]
        }]))
        .expect("tool history");
        assert!(validate_official_llama_tool_history(&messages).is_err());
        let single: Vec<Message> = serde_json::from_value(json!([{
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "weather", "arguments": "{}"
            }}]
        }]))
        .expect("single tool history");
        validate_official_llama_tool_history(&single).expect("single tool call");
    }

    #[test]
    fn sdk_response_null_fields_round_trip_and_strict_tools_fail_explicitly() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{
                "role": "assistant", "content": "answer",
                "refusal": null, "annotations": null, "audio": null,
                "function_call": null
            }],
            "tools": [{"type": "function", "function": {
                "name": "weather", "strict": false
            }}]
        }))
        .expect("OpenAI SDK response fields");
        validate_chat_request(&request, "leone").expect("nullable response fields");

        let mut invalid = request;
        invalid.messages[0].refusal = Some(json!("refused"));
        assert!(validate_chat_request(&invalid, "leone").is_err());

        let strict: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Call weather"}],
            "tools": [{"type": "function", "function": {
                "name": "weather", "strict": true
            }}]
        }))
        .expect("strict tool request");
        let error = validate_chat_request(&strict, "leone").expect_err("strict rejection");
        assert!(error.to_string().contains("strict=true is unsupported"));
    }

    #[test]
    fn stop_bytes_are_bounded_and_tool_stops_are_rejected() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "x".repeat(MAX_STOP_BYTES + 1)
        }))
        .expect("long stop request");
        assert!(request_stop_sequences(&request).is_err());
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Call weather"}],
            "stop": "</tool_call>",
            "tools": [{"type": "function", "function": {"name": "weather"}}]
        }))
        .expect("tool stop request");
        assert!(validate_chat_request(&request, "leone").is_err());
    }

    #[test]
    fn stream_errors_finish_the_existing_sse_response() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let mut client = TcpStream::connect(listener.local_addr().expect("listener address"))
            .expect("connect test client");
        let (mut server, _) = listener.accept().expect("accept test client");
        write_chat_headers_io(&mut server, "session", "chatcmpl-test", 1, "leone", None)
            .expect("headers");
        write_stream_error(&mut server, true, 200, "malformed tool call", None)
            .expect("stream error");
        drop(server);
        let mut response = String::new();
        client.read_to_string(&mut response).expect("read stream");
        assert_eq!(response.matches("HTTP/1.1 200").count(), 1);
        assert!(response.contains("malformed tool call"));
        assert!(response.contains("data: [DONE]\n\n"));
    }

    #[test]
    fn stream_headers_return_one_validated_session_id() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind header listener");
        let mut client = TcpStream::connect(listener.local_addr().expect("header address"))
            .expect("connect header client");
        let (mut server, _) = listener.accept().expect("accept header client");
        write_chat_headers_io(
            &mut server,
            "client-session",
            "chatcmpl-header",
            1,
            "leone",
            Some("https://client.example"),
        )
        .expect("headers");
        drop(server);
        let mut response = String::new();
        client.read_to_string(&mut response).expect("read headers");
        assert_eq!(response.matches("X-Leone-Session:").count(), 1);
        assert!(response.contains("X-Leone-Session: client-session\r\n"));
        assert!(response.contains("Access-Control-Expose-Headers: X-Leone-Session\r\n"));
    }

    #[test]
    fn failed_blob_write_does_not_publish_a_truncated_canonical_file() {
        let root =
            std::env::temp_dir().join(format!("leone-session-blob-write-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let blob = b"complete session archive";
        let path = root
            .join("blobs")
            .join(format!("{}.json", sha256_bytes(blob)));
        let fail_write = |mut file: File, bytes: &[u8]| {
            file.write_all(&bytes[..bytes.len() / 2])?;
            Err(io::Error::other("injected blob write failure"))
        };
        assert!(write_session_blob_with(&path, blob, &fail_write, &sync_directory).is_err());
        assert!(!path.exists());
        assert_eq!(
            fs::read_dir(root.join("blobs"))
                .expect("blob directory")
                .count(),
            0
        );

        let sync_calls = std::cell::Cell::new(0);
        let count_sync = |directory: &Path| {
            assert_eq!(directory, root.join("blobs"));
            sync_calls.set(sync_calls.get() + 1);
            sync_directory(directory)
        };
        write_session_blob_with(&path, blob, &write_new_session_blob, &count_sync)
            .expect("publish complete blob");
        assert_eq!(sync_calls.get(), 1);
        assert_eq!(fs::read(path).expect("read canonical blob"), blob);
        drop(store);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn session_reference_rejects_a_blob_path_before_io() {
        let root = std::env::temp_dir().join(format!("leone-session-blob-path-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let session_id = "session";
        let path = store.reference_path(session_id);
        let reference = SessionReference {
            schema_version: SessionStore::REFERENCE_SCHEMA_VERSION,
            session_id: session_id.to_owned(),
            blob_sha256: "../outside".to_owned(),
            last_used: 1,
        };
        fs::write(
            &path,
            serde_json::to_vec(&reference).expect("reference JSON"),
        )
        .expect("write invalid reference");
        let error = store
            .load_metadata(session_id)
            .expect_err("reject blob path");
        assert!(error.to_string().contains("blob digest is invalid"));
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn existing_blob_retries_a_failed_directory_sync() {
        let root = std::env::temp_dir().join(format!("leone-session-blob-sync-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let blob = b"durable session archive";
        let path = root
            .join("blobs")
            .join(format!("{}.json", sha256_bytes(blob)));
        let fail_sync = |_directory: &Path| Err(io::Error::other("injected blob sync failure"));
        assert!(write_session_blob_with(&path, blob, &write_new_session_blob, &fail_sync).is_err());
        assert_eq!(fs::read(&path).expect("read linked blob"), blob);

        let sync_calls = std::cell::Cell::new(0);
        let retry_sync = |directory: &Path| {
            assert_eq!(directory, root.join("blobs"));
            sync_calls.set(sync_calls.get() + 1);
            sync_directory(directory)
        };
        write_session_blob_with(&path, blob, &write_new_session_blob, &retry_sync)
            .expect("retry blob directory sync");
        assert_eq!(sync_calls.get(), 1);
        drop(store);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn recovery_keeps_a_published_successor_before_supersede_intent() {
        let root = std::env::temp_dir().join(format!("leone-session-successor-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist predecessor");
        let lease = store
            .lease_reference("session")
            .expect("lease predecessor")
            .expect("persisted predecessor");
        let successor = populated_cpu_checkpoint();
        let expected_tokens = successor.evaluated_tokens().to_vec();
        let published = store
            .persist("session", MODEL, successor, 2)
            .expect("publish successor");

        drop(lease);
        drop(store);
        let (reopened_store, reopened, _) =
            SessionStore::open(root.clone(), 8).expect("recover successor");
        let recovered = reopened
            .iter()
            .find(|archive| archive.session_id == "session")
            .expect("recovered successor");
        assert_eq!(recovered.archive.blob_sha256, published.blob_sha256);
        let archive = reopened_store
            .load_archive("session", &recovered.archive, None, MODEL)
            .expect("load recovered successor");
        assert_eq!(
            archive
                .checkpoint()
                .expect("successor checkpoint")
                .evaluated_tokens(),
            expected_tokens
        );
        assert_eq!(
            fs::read_dir(root.join("quarantine"))
                .expect("quarantine directory")
                .count(),
            0
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn recovery_lookup_leases_the_published_successor() {
        let root =
            std::env::temp_dir().join(format!("leone-session-successor-lookup-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist predecessor");
        let lease = store
            .lease_reference("session")
            .expect("lease predecessor")
            .expect("persisted predecessor");
        let published = store
            .persist("session", MODEL, populated_cpu_checkpoint(), 2)
            .expect("publish successor");
        let mut lease = Some(lease);
        store
            .retain_recovery_lease("session", &mut lease)
            .expect("retain predecessor recovery lease");

        let mut successor_lease = store
            .lease_reference("session")
            .expect("resolve recovery lease")
            .expect("lease successor");
        let successor_reference = read_session_reference(successor_lease.active_path(), "session")
            .expect("read successor reference");
        assert_eq!(successor_reference.blob_sha256, published.blob_sha256);
        successor_lease.restore().expect("restore successor");
        drop(successor_lease);
        drop(store);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn superseded_cleanup_failure_preserves_the_published_successor() {
        let root = std::env::temp_dir().join(format!(
            "leone-session-successor-cleanup-{}",
            Uuid::new_v4()
        ));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist predecessor");
        let mut lease = store
            .lease_reference("session")
            .expect("lease predecessor")
            .expect("persisted predecessor");
        let successor = populated_cpu_checkpoint();
        let expected_tokens = successor.evaluated_tokens().to_vec();
        let published = store
            .persist("session", MODEL, successor, 2)
            .expect("publish successor");
        let failed_path = lease.quarantine.superseded.clone();
        let fail_cleanup = |path: &Path| {
            if path == failed_path {
                Err(io::Error::other("injected superseded cleanup failure"))
            } else {
                fs::remove_file(path)
            }
        };
        assert!(lease.retire_with_remove(&fail_cleanup).is_err());
        assert_eq!(lease.state, SessionReferenceLeaseState::Superseded);

        drop(lease);
        drop(store);
        let (reopened_store, reopened, _) =
            SessionStore::open(root.clone(), 8).expect("recover successor");
        let recovered = reopened
            .iter()
            .find(|archive| archive.session_id == "session")
            .expect("recovered successor");
        assert_eq!(recovered.archive.blob_sha256, published.blob_sha256);
        let archive = reopened_store
            .load_archive("session", &recovered.archive, None, MODEL)
            .expect("load recovered successor");
        assert_eq!(
            archive
                .checkpoint()
                .expect("successor checkpoint")
                .evaluated_tokens(),
            expected_tokens
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn discard_cleanup_failure_never_restores_the_identity() {
        let root = std::env::temp_dir().join(format!("leone-session-discard-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist session");
        let mut lease = store
            .lease_reference("session")
            .expect("lease session")
            .expect("persisted session");
        let failed_path = lease.quarantine.discard.clone();
        let fail_cleanup = |path: &Path| {
            if path == failed_path {
                Err(io::Error::other("injected discard cleanup failure"))
            } else {
                fs::remove_file(path)
            }
        };
        assert!(lease.discard_with_remove(&fail_cleanup).is_err());
        assert_eq!(lease.state, SessionReferenceLeaseState::Discard);
        assert!(!lease.namespace_sync_pending);

        let key = sha256_bytes(b"session");
        let stale = root
            .join("quarantine")
            .join(format!("{key}-{}.restore.json", Uuid::new_v4().simple()));
        fs::write(&stale, b"invalid stale restore").expect("write stale restore");
        sync_directory(&root.join("quarantine")).expect("sync stale restore");

        drop(lease);
        drop(store);
        let (_, reopened, _) = SessionStore::open(root.clone(), 8).expect("recover discard");
        assert!(reopened.is_empty());
        assert_eq!(
            fs::read_dir(root.join("quarantine"))
                .expect("quarantine directory")
                .count(),
            0
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn restored_lease_retries_pending_directory_syncs() {
        let root =
            std::env::temp_dir().join(format!("leone-session-sync-retry-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist session");
        let mut lease = store
            .lease_reference("session")
            .expect("lease session")
            .expect("persisted session");
        let failed_calls = std::cell::Cell::new(0);
        let fail_first_sync = |_path: &Path| {
            failed_calls.set(failed_calls.get() + 1);
            Err(io::Error::other("injected restore sync failure"))
        };
        assert!(lease.restore_with_sync(&fail_first_sync).is_err());
        assert_eq!(lease.state, SessionReferenceLeaseState::Active);
        assert!(lease.namespace_sync_pending);

        let retry_calls = std::cell::Cell::new(0);
        let retry_sync = |_path: &Path| {
            retry_calls.set(retry_calls.get() + 1);
            Ok(())
        };
        lease
            .restore_with_sync(&retry_sync)
            .expect("retry namespace sync");
        assert_eq!(retry_calls.get(), 2);
        assert!(!lease.namespace_sync_pending);
        drop(lease);
        drop(store);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn zero_repetition_window_disables_counting_penalties() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "repetition_penalty": 1.2,
            "repetition_window": 0
        }))
        .expect("penalty request");
        assert_eq!(
            penalties(&request).expect("penalties").window,
            PenaltyWindow::Disabled
        );
    }

    #[test]
    fn mirostat_rejects_adaptive_speculation() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [{"role": "user", "content": "Hello"}],
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "adaptive_speculation": true
        }))
        .expect("sampling request");
        let error = validate_chat_sampling(&request).expect_err("conflicting controls");
        assert!(error
            .to_string()
            .contains("Mirostat cannot be combined with adaptive_speculation"));
    }

    #[test]
    fn wake_restore_retries_published_successor_cleanup() {
        let root = std::env::temp_dir().join(format!(
            "leone-session-successor-restore-{}",
            Uuid::new_v4()
        ));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist predecessor");
        let mut reference_lease = store.lease_reference("session").expect("lease predecessor");
        let published = store
            .persist("session", MODEL, populated_cpu_checkpoint(), 2)
            .expect("publish successor");

        let sync_calls = std::cell::Cell::new(0);
        let fail_sync = |_directory: &Path| {
            sync_calls.set(sync_calls.get() + 1);
            Err(io::Error::other("injected successor cleanup sync failure"))
        };
        assert!(reference_lease
            .as_mut()
            .expect("predecessor lease")
            .restore_with_sync(&fail_sync)
            .is_err());
        assert_eq!(sync_calls.get(), 1);
        assert_eq!(
            reference_lease
                .as_ref()
                .expect("retained predecessor lease")
                .state,
            SessionReferenceLeaseState::Superseded
        );
        restore_hibernation_reference(&mut reference_lease)
            .expect("retry successor cleanup through wake restoration");
        assert!(reference_lease.is_none());
        let current = store
            .load_metadata("session")
            .expect("load current reference")
            .expect("published successor");
        assert_eq!(current.blob_sha256, published.blob_sha256);
        assert_eq!(
            fs::read_dir(root.join("quarantine"))
                .expect("quarantine directory")
                .count(),
            0
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn failed_successor_publication_invalidates_predecessor_metadata() {
        let (root, mut server) = persisted_cpu_server_default("session-successor-cache");
        let predecessor = server
            .persisted_metadata("session")
            .expect("load predecessor metadata")
            .expect("cached predecessor")
            .blob_sha256;
        resident_cpu_source(&mut server, "session", cpu_options(1));
        let mut stored = server.sessions.remove("session").expect("resident session");
        stored.reference_lease = server
            .lease_session_reference("session")
            .expect("lease predecessor reference");
        let expected_tokens = stored.generation.evaluated_tokens().to_vec();
        let sync_calls = std::cell::Cell::new(0);
        let fail_reference_sync = |_directory: &Path| {
            sync_calls.set(sync_calls.get() + 1);
            Err(io::Error::other("injected reference sync failure"))
        };
        server
            .session_store
            .as_ref()
            .expect("session store")
            .persist_with_reference_sync(
                "session",
                MODEL,
                stored.generation.checkpoint(),
                2,
                &fail_reference_sync,
            )
            .expect_err("fail after publishing successor reference");
        assert_eq!(sync_calls.get(), 1);
        let successor = server
            .session_store
            .as_ref()
            .expect("session store")
            .load_metadata("session")
            .expect("load successor metadata")
            .expect("published successor");
        assert_ne!(successor.blob_sha256, predecessor);

        let cleanup_sync_calls = std::cell::Cell::new(0);
        let fail_cleanup_sync = |_directory: &Path| {
            cleanup_sync_calls.set(cleanup_sync_calls.get() + 1);
            Err(io::Error::other("injected successor cleanup sync failure"))
        };
        server
            .retain_failed_session_with_sync("session".to_owned(), stored, &fail_cleanup_sync)
            .expect_err("retain owner after successor cleanup failure");
        assert_eq!(cleanup_sync_calls.get(), 1);
        assert!(!server.persisted.contains_key("session"));
        let stored = server
            .sessions
            .remove("session")
            .expect("retained session after cleanup failure");
        assert!(stored
            .reference_lease
            .as_ref()
            .is_some_and(|lease| { lease.state == SessionReferenceLeaseState::Superseded }));
        server
            .retain_failed_session("session".to_owned(), stored)
            .expect("retry successor cleanup");
        drop(server.sessions.remove("session").expect("retained session"));
        let metadata = server
            .persisted_metadata("session")
            .expect("reload successor metadata")
            .expect("successor metadata");
        assert_eq!(metadata.blob_sha256, successor.blob_sha256);
        let mut restored = server
            .restore_persisted_continuation("session", metadata, server.context_limit)
            .expect("restore published successor");
        assert!(!restored.generation.is_empty());
        restore_hibernation_reference(&mut restored.reference_lease)
            .expect("restore successor reference");
        drop(restored);
        drop(server);

        let (reopened_store, reopened, _) =
            SessionStore::open(root.clone(), 8).expect("restart store");
        let metadata = reopened
            .iter()
            .find(|archive| archive.session_id == "session")
            .expect("restarted successor");
        let archive = reopened_store
            .load_archive("session", &metadata.archive, None, MODEL)
            .expect("load restarted successor");
        assert_eq!(
            archive
                .checkpoint()
                .expect("successor checkpoint")
                .evaluated_tokens(),
            expected_tokens
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn successor_without_its_blob_does_not_replace_recovery_evidence() {
        let root =
            std::env::temp_dir().join(format!("leone-session-successor-blob-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        store
            .persist(
                "session",
                MODEL,
                GenerationSession::<CpuBackend>::new().checkpoint(),
                1,
            )
            .expect("persist predecessor");
        let mut lease = store
            .lease_reference("session")
            .expect("lease predecessor")
            .expect("persisted predecessor");
        write_session_reference(&store, "session", &"0".repeat(64), 2)
            .expect("publish malformed successor");

        assert!(lease.restore().is_err());
        assert!(lease.quarantine.restore.exists());
        assert!(store.reference_path("session").exists());
        drop(lease);
        drop(store);
        assert!(SessionStore::open(root.clone(), 8).is_err());
        assert_eq!(
            fs::read_dir(root.join("quarantine"))
                .expect("quarantine directory")
                .count(),
            1
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn content_parts_preserve_text_and_type_image_failures() {
        assert_eq!(
            message_text(&json!([
                {"type": "text", "text": "one"},
                {"type": "text", "text": " two"}
            ]))
            .expect("text parts"),
            "one two"
        );
        let error = message_text(&json!([{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}
        }]))
        .expect_err("no vision backend");
        assert!(error.to_string().contains("no vision backend"));
    }

    #[test]
    fn scheduled_request_errors_use_the_openai_shape() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let mut client = TcpStream::connect(listener.local_addr().expect("listener address"))
            .expect("connect test client");
        let (mut server, _) = listener.accept().expect("accept test client");

        let result = scheduled_request_or_error::<()>(
            &mut server,
            Err(invalid_data("invalid request").into()),
            None,
        )
        .expect("write error response");
        assert!(result.is_none());
        drop(server);

        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("read error response");
        assert!(response.starts_with("HTTP/1.1 400 Error\r\n"));
        assert!(response.contains("\"type\":\"invalid_request_error\""));
        assert!(response.contains("\"message\":\"invalid request\""));
    }

    #[test]
    fn admission_rejections_use_typed_openai_errors() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let mut client = TcpStream::connect(listener.local_addr().expect("listener address"))
            .expect("connect test client");
        let (mut server, _) = listener.accept().expect("accept test client");

        write_admission_rejection(&mut server, AdmissionReject::KvCapacity)
            .expect("write rejection");
        drop(server);

        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("read rejection");
        assert!(response.starts_with("HTTP/1.1 429 Error\r\n"));
        assert!(response.contains("\"type\":\"server_overloaded\""));
        assert!(response.contains("\"code\":\"kv-capacity\""));
    }

    #[test]
    fn implicit_prefix_selection_prefers_positive_credit() {
        assert!(super::prefix_candidate_is_better(
            (4, 1, "resident"),
            (0, 99, "incompatible")
        ),);
        assert!(!super::prefix_candidate_is_better(
            (0, 100, "wrong-dtype"),
            (1, 0, "resident")
        ));
    }

    #[test]
    fn failed_discard_intent_blocks_in_process_restore() {
        let root =
            std::env::temp_dir().join(format!("leone-session-discard-intent-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("persist session");
        let mut lease = store
            .lease_reference("session")
            .expect("lease session")
            .expect("persisted session");
        let blocked_path = lease.quarantine.discard.clone();
        fs::create_dir(&blocked_path).expect("block discard rename");
        assert!(lease.discard().is_err());
        assert!(lease.discard_requested);
        assert_eq!(lease.state, SessionReferenceLeaseState::Restore);
        let mut lease = Some(lease);
        store
            .retain_recovery_lease("session", &mut lease)
            .expect("retain pending discard lease");

        assert!(store.has_reference("session").is_err());
        assert!(store.lease_reference("session").is_err());
        fs::remove_dir(blocked_path).expect("unblock discard rename");
        assert!(store
            .lease_reference("session")
            .expect("finish pending discard")
            .is_none());
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn implicit_prefix_selection_has_a_stable_identifier_tie_break() {
        assert!(super::prefix_candidate_is_better(
            (4, 7, "a-session"),
            (4, 7, "z-session")
        ));
        assert!(!super::prefix_candidate_is_better(
            (4, 7, "z-session"),
            (4, 7, "a-session")
        ));
        assert!(super::prefix_candidate_is_better(
            (4, 8, "z-session"),
            (4, 7, "a-session")
        ));
    }

    #[test]
    fn cpu_prefix_selection_propagates_request_validation_errors() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        resident_cpu_source(&mut server, "source", cpu_options(1));
        assert!(matches!(
            server.select_session(&[], &cpu_options(1)),
            Err(RuntimeError::EmptyPrompt)
        ));
    }

    #[test]
    fn openai_user_does_not_become_a_session_identity() {
        let request: super::ChatRequest = serde_json::from_value(json!({
            "model": "leone",
            "messages": [],
            "user": "caller"
        }))
        .expect("chat request");
        let http = super::HttpRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: BTreeMap::new(),
            body: Vec::new(),
        };
        assert_eq!(super::requested_chat_session(&request, &http), None);
    }

    #[test]
    fn recovery_requarantines_an_active_reference_after_sync_failure() {
        let root =
            std::env::temp_dir().join(format!("leone-session-active-recovery-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        store
            .persist(
                "session",
                MODEL,
                GenerationSession::<CpuBackend>::new().checkpoint(),
                1,
            )
            .expect("persist session");
        let fail_sync = |_directory: &Path| Err(io::Error::other("injected sync failure"));
        assert!(store
            .lease_reference_with_sync("session", &fail_sync)
            .is_err());

        let retry_calls = std::cell::Cell::new(0);
        let retry_sync = |_directory: &Path| {
            retry_calls.set(retry_calls.get() + 1);
            Ok(())
        };
        let mut lease = store
            .lease_reference_with_sync("session", &retry_sync)
            .expect("retry active recovery")
            .expect("recovered reference");
        assert_eq!(retry_calls.get(), 4);
        assert_eq!(lease.state, SessionReferenceLeaseState::Restore);
        assert!(!lease.namespace_sync_pending);
        assert!(!store.reference_path("session").exists());
        assert!(lease.quarantine.restore.exists());
        lease.restore().expect("restore recovered reference");
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn metadata_lookup_reads_a_recovery_owned_reference() {
        let root = std::env::temp_dir().join(format!(
            "leone-session-recovery-metadata-{}",
            Uuid::new_v4()
        ));
        let (store, _, _) = SessionStore::open(root.clone(), 8).expect("new store");
        let persisted = store
            .persist(
                "session",
                MODEL,
                GenerationSession::<CpuBackend>::new().checkpoint(),
                7,
            )
            .expect("persist session");
        let lease = store
            .lease_reference("session")
            .expect("lease session")
            .expect("persisted reference");
        let mut lease = Some(lease);
        store
            .retain_recovery_lease("session", &mut lease)
            .expect("retain recovery lease");

        let metadata = store
            .load_metadata("session")
            .expect("load recovery metadata")
            .expect("recovery metadata");
        assert_eq!(metadata.blob_sha256, persisted.blob_sha256);
        assert_eq!(metadata.last_used, 7);
        assert!(!store.reference_path("session").exists());
        let mut recovered = store
            .lease_reference("session")
            .expect("lease retained recovery reference")
            .expect("retained recovery reference");
        recovered.restore().expect("restore retained reference");
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn cpu_prefix_reuse_preserves_source_and_clears_child_controller() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        let mut source_options = cpu_options(1);
        source_options.mirostat = Some(MirostatConfig::new(5.0, 0.1).expect("Mirostat config"));
        resident_cpu_source(&mut server, "source", source_options);
        let prompt = [1, 2, 3, 5];
        let options = cpu_options(1);

        let (first_id, first_source) =
            super::chat_plan_session(&server, None, None, &prompt, &options)
                .expect("implicit prefix plan");
        let (second_id, second_source) =
            super::chat_plan_session(&server, None, None, &prompt, &options)
                .expect("second implicit prefix plan");
        assert_ne!(first_id, second_id);
        assert!(
            matches!(first_source, ChatPlanSource::PrefixReuse { ref source_id } if source_id == "source")
        );
        assert!(
            matches!(second_source, ChatPlanSource::PrefixReuse { ref source_id } if source_id == "source")
        );
        let credit = super::scheduler_prefix_reused_tokens(
            &server,
            &first_id,
            &first_source,
            &prompt,
            &options,
        )
        .expect("prefix credit");
        assert!(credit > 0);

        let source_tokens = server
            .sessions
            .get("source")
            .expect("source resident")
            .generation
            .evaluated_tokens()
            .to_vec();
        let mut child = server
            .lease_session(&test_chat_plan(
                &first_id,
                first_source,
                &prompt,
                options.clone(),
            ))
            .expect("prefix lease");
        let source_checkpoint = server
            .sessions
            .get("source")
            .expect("source remains resident")
            .generation
            .checkpoint();
        assert!(source_checkpoint.mirostat().is_some());
        assert!(child.generation.checkpoint().mirostat().is_none());
        assert!(child.metadata_allocation.is_some());
        assert!(server
            .sessions
            .get("source")
            .and_then(|stored| stored.metadata_allocation.as_ref())
            .is_some());
        server
            .runtime
            .generate_session_tokens(
                &mut child.generation,
                &prompt,
                options,
                |_| Ok(()),
                || false,
            )
            .expect("continue child");
        assert_eq!(child.generation.last_replay().reused_tokens, credit);
        assert!(server
            .sessions
            .get("source")
            .expect("source remains after child continuation")
            .generation
            .checkpoint()
            .mirostat()
            .is_some());
        assert_eq!(
            server
                .sessions
                .get("source")
                .expect("source remains after child continuation")
                .generation
                .evaluated_tokens(),
            source_tokens
        );
    }

    struct QueuedChat {
        _transport: Transport<HttpRequest>,
        client: TcpStream,
        request_id: RequestId,
        cancellation: Cancellation,
    }

    fn cpu_chat_service(root: &Path, model: &Path) -> ChatService<CpuBackend> {
        let mut server = cpu_server(model);
        server.receipts = root.join("receipts");
        let trace = ServiceTraceRecorder::new(
            &server.runtime,
            MetricsMemoryTopology::BackendChildFallback,
            None,
        );
        let executor = ServerExecutor {
            server,
            tasks: BTreeMap::new(),
            wire_request_ids: BTreeMap::new(),
            queued_outputs: BTreeMap::new(),
            terminal_outputs: VecDeque::new(),
            trace,
        };
        let policy = server_scheduler_policy(&executor.server, 4, 1).expect("scheduler policy");
        ScheduledService::new(policy, executor).expect("service")
    }

    fn admit_planned_chat(
        service: &mut ChatService<CpuBackend>,
        requested_session: Option<&str>,
        fork_parent: Option<&str>,
        prompt: &[u32],
        options: &GenerateOptions,
    ) -> QueuedChat {
        let (transport, client, incoming) =
            test_server_incoming_with(Duration::from_secs(5), 1 << 20);
        let request_id = RequestId(incoming.request_id);
        let wire = format!("chatcmpl-queued-{}", request_id.0);
        let executor = service.executor_mut();
        let server = &mut executor.server;
        server
            .metrics
            .request_started(request_id, incoming.received_at_ns);
        let (session_id, source) =
            chat_plan_session(server, requested_session, fork_parent, prompt, options)
                .expect("chat plan");
        let credit = scheduler_prefix_reused_tokens(server, &session_id, &source, prompt, options)
            .expect("plan credit");
        let mut plan = test_chat_plan(&session_id, source, prompt, options.clone());
        plan.prefix_reused_tokens = credit;
        plan.completion_id = wire.clone();
        plan.request_sha256 = sha256_bytes(wire.as_bytes());
        let spec = scheduled_chat_spec(request_id, &plan, incoming.deadline).expect("chat spec");
        executor.wire_request_ids.insert(wire.clone(), request_id);
        executor.register_queued_output(request_id, wire, incoming.output.clone());
        let cancellation = incoming.cancellation.clone();
        let admission = ChatAdmission(RefCell::new(Some(PendingChat {
            plan,
            stream: incoming.output,
            cancellation: incoming.cancellation,
            deadline: incoming.deadline,
        })));
        admit_pending_chat(service, spec, &admission).expect("admit chat");
        QueuedChat {
            _transport: transport,
            client,
            request_id,
            cancellation,
        }
    }

    fn run_queued_chat(service: &mut ChatService<CpuBackend>, chat: &mut QueuedChat) -> String {
        while service.has_runnable_requests() {
            service.tick_batch(scheduler_now_ns()).expect("tick");
        }
        read_test_response(&mut chat.client)
    }

    fn response_receipt(response: &str) -> ResponseReceipt {
        let (_, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        let value: Value = serde_json::from_str(body).expect("response JSON");
        serde_json::from_value(value["leone_receipt"].clone()).expect("response receipt")
    }

    fn metrics_json(server: &Server<CpuBackend>) -> Value {
        let mut bytes = Vec::new();
        server
            .metrics
            .write_response(&mut bytes)
            .expect("metrics response");
        let text = String::from_utf8(bytes).expect("metrics text");
        let (_, body) = text.split_once("\r\n\r\n").expect("metrics body");
        serde_json::from_str(body).expect("metrics JSON")
    }

    fn isolated_cpu_tokens(
        runtime: &mut Runtime<CpuBackend>,
        prompt: &[u32],
        options: &GenerateOptions,
    ) -> Vec<u32> {
        runtime
            .generate_session_tokens(
                &mut GenerationSession::new(),
                prompt,
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("isolated CPU oracle")
            .tokens
    }

    #[test]
    fn queued_prefix_child_matches_the_isolated_oracle_after_its_source_moves() {
        let moves: [fn(&mut Server<CpuBackend>); 2] = [
            |server| {
                server.max_sessions = 0;
                server.hibernate_until_capacity().expect("hibernate source");
                assert!(server.hibernated.contains_key("source"));
            },
            |server| server.discard_session("source").expect("discard source"),
        ];
        for move_source in moves {
            let fixture = cpu_fixture();
            let root = test_server_root("prefix-queue");
            let mut service = cpu_chat_service(&root, fixture.path());
            let host_baseline = service.executor().server.memory.host.snapshot().live_bytes;
            resident_cpu_source(&mut service.executor_mut().server, "source", cpu_options(1));
            let prompt = [1, 2, 3, 5];
            let options = cpu_options(2);

            let mut chat = admit_planned_chat(&mut service, None, None, &prompt, &options);
            let executor = service.executor();
            let task = &executor.tasks[&chat.request_id];
            let credit = task.planned_prefix_tokens;
            assert!(credit > 0, "the plan names the live source");
            assert!(executor.server.sessions.contains_key("source"));
            assert!(task.stored.metadata_allocation.is_some());

            move_source(&mut service.executor_mut().server);
            assert!(!service.executor().server.sessions.contains_key("source"));
            let receipt = response_receipt(&run_queued_chat(&mut service, &mut chat));

            receipt.verify().expect("signed response");
            let session = &receipt.claim.session;
            assert_eq!(session.reused_tokens, credit as u64);
            assert_eq!(session.replayed_tokens, 0);
            assert_eq!(session.computed_tokens, (prompt.len() - credit) as u64);
            let oracle = isolated_cpu_tokens(
                &mut service.executor_mut().server.runtime,
                &prompt,
                &options,
            );
            assert_eq!(
                receipt.claim.response_tokens_sha256,
                token_stream_sha256(&oracle)
            );
            let metrics = metrics_json(&service.executor().server);
            assert_eq!(metrics["prefix_reuse_sources"]["reused_tokens"], credit);
            let realized = &metrics["realized_prompt_tokens"];
            assert_eq!(realized["requests"], 1);
            assert_eq!(realized["prompt_tokens"], prompt.len());
            assert_eq!(realized["reused_tokens"], credit);
            assert_eq!(realized["computed_tokens"], prompt.len() - credit);
            assert_eq!(realized["credit_shortfall_tokens"], 0);

            let server = &mut service.executor_mut().server;
            server.sessions.clear();
            server.hibernated.clear();
            service.prune_terminal();
            assert_eq!(
                service.executor().server.memory.host.snapshot().live_bytes,
                host_baseline
            );
            drop(chat);
            remove_test_server_root(root);
        }
    }

    #[test]
    fn cancelled_prefill_keeps_its_planned_credit_and_realizes_no_reuse() {
        let fixture = cpu_fixture();
        let root = test_server_root("prefix-cancel");
        let mut service = cpu_chat_service(&root, fixture.path());
        resident_cpu_source(&mut service.executor_mut().server, "source", cpu_options(1));
        let prompt = [1, 2, 3, 5, 6, 7, 8, 9, 10];
        let options = cpu_options(1);
        let mut chat = admit_planned_chat(&mut service, None, None, &prompt, &options);
        let credit = service.executor().tasks[&chat.request_id].planned_prefix_tokens;

        service.tick_batch(scheduler_now_ns()).expect("first chunk");
        let task = &service.executor().tasks[&chat.request_id];
        assert!(task.pending_prefill.is_some(), "prefill is mid-flight");
        assert!(task.request_replay.is_none());
        chat.cancellation.cancel();
        let response = run_queued_chat(&mut service, &mut chat);

        assert!(!response.contains("\"leone_receipt\""));
        let metrics = metrics_json(&service.executor().server);
        assert_eq!(metrics["prefix_reuse_sources"]["reused_tokens"], credit);
        assert!(credit > 0);
        let realized = &metrics["realized_prompt_tokens"];
        assert_eq!(realized["requests"], 0);
        assert_eq!(realized["reused_tokens"], 0);
        assert_eq!(realized["planned_credit_tokens"], 0);
        remove_test_server_root(root);
    }

    #[test]
    fn fork_of_a_hibernated_parent_realizes_reuse_the_plan_did_not_credit() {
        let fixture = cpu_fixture();
        let root = test_server_root("fork-realized");
        let mut service = cpu_chat_service(&root, fixture.path());
        let server = &mut service.executor_mut().server;
        resident_cpu_source(server, "parent", cpu_options(1));
        server.max_sessions = 0;
        server.hibernate_until_capacity().expect("hibernate parent");
        server.max_sessions = 8;
        let prompt = [1, 2, 3, 4, 5];
        let options = cpu_options(1);

        let mut chat = admit_planned_chat(
            &mut service,
            Some("child"),
            Some("parent"),
            &prompt,
            &options,
        );
        let task = &service.executor().tasks[&chat.request_id];
        assert_eq!(
            task.planned_prefix_tokens, 0,
            "host wake has no plan credit"
        );
        let receipt = response_receipt(&run_queued_chat(&mut service, &mut chat));

        let reused = receipt.claim.session.reused_tokens;
        assert!(reused > 0);
        let metrics = metrics_json(&service.executor().server);
        assert_eq!(metrics["prefix_reuse_sources"]["reused_tokens"], 0);
        let realized = &metrics["realized_prompt_tokens"];
        assert_eq!(realized["reused_tokens"], reused);
        assert_eq!(realized["planned_credit_tokens"], 0);
        assert_eq!(realized["credit_shortfall_tokens"], 0);
        remove_test_server_root(root);
    }

    #[test]
    fn resident_metadata_lease_covers_later_session_growth() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        resident_cpu_source(&mut server, "source", cpu_options(1));
        let mut stored = server.sessions.remove("source").expect("source resident");
        assert!(stored.metadata_allocation.is_some());
        stored.metadata_allocation = Some(ResidentMetadataLease::single(
            server.memory.host.allocate(1).expect("small lease"),
        ));
        let prompt = [1, 2, 3, 4, 5, 6];
        server
            .runtime
            .generate_session_tokens(
                &mut stored.generation,
                &prompt,
                cpu_options(4),
                |_| Ok(()),
                || false,
            )
            .expect("grow resident session");
        let required = server
            .resident_metadata_requirement(&stored.generation, prompt.len())
            .expect("metadata requirement");
        server
            .ensure_resident_metadata(&mut stored, prompt.len())
            .expect("grow metadata lease");
        let grown = stored
            .metadata_allocation
            .as_ref()
            .expect("grown metadata lease")
            .bytes();
        assert_eq!(grown, required);
        server.sessions.insert("source".to_owned(), stored);
    }

    #[test]
    fn cold_session_lease_uses_request_context_metadata_bound() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        let prompt = [1, 2, 3];
        let options = cpu_options(1);
        let request_context = prompt.len() + options.max_tokens;
        let full_context = server
            .runtime
            .resident_metadata_bound(server.context_limit)
            .expect("full metadata bound");
        let request_bound = server
            .runtime
            .resident_metadata_bound(request_context)
            .expect("request metadata bound");
        assert!(request_bound < full_context);
        let accounting = server.memory.host.snapshot();
        let owned = accounting
            .live_bytes
            .checked_add(accounting.reserved_bytes)
            .expect("host accounting sum");
        server
            .memory
            .host
            .set_limit(NonZeroU64::new(owned + request_bound).expect("finite test limit"))
            .expect("lower host limit");
        assert!(server.memory.host.allocate(full_context).is_err());

        let plan = test_chat_plan("cold", ChatPlanSource::Continue, &prompt, options);
        let stored = server.lease_session(&plan).expect("request metadata fits");
        assert_eq!(
            stored
                .metadata_allocation
                .as_ref()
                .expect("metadata allocation")
                .bytes(),
            request_bound
        );
    }

    #[test]
    fn persisted_archive_cache_evicts_oldest_metadata() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        server.persisted_cache_limit = 3;
        let persisted_tables = allocate_persisted_cache_tables(&server.memory, 3)
            .expect("three-entry persisted cache");
        server.persisted = persisted_tables.entries;
        server._persisted_table_allocation = persisted_tables.allocation;
        let archive_capacity = server.persisted.capacity();
        let before = server.memory.host.snapshot().live_bytes;
        for index in 0..5 {
            server
                .cache_persisted_archive(
                    format!("session-{index}"),
                    StoredArchive {
                        blob_sha256: format!("digest-{index}"),
                        last_used: index,
                    },
                )
                .expect("cache persisted archive");
        }
        assert!(server.memory.host.snapshot().live_bytes > before);
        assert_eq!(server.persisted.capacity(), archive_capacity);
        assert_eq!(server.persisted.len(), 3);
        assert!(!server.persisted.contains_key("session-0"));
        assert!(!server.persisted.contains_key("session-1"));
        assert!(server.persisted.contains_key("session-2"));
        assert!(server.persisted.contains_key("session-3"));
        assert!(server.persisted.contains_key("session-4"));
    }

    #[test]
    fn fork_admission_checks_uncached_durable_identities() {
        let root =
            std::env::temp_dir().join(format!("leone-session-fork-index-{}", Uuid::new_v4()));
        let (store, _, _) = SessionStore::open(root.clone(), 1).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("parent", MODEL, empty.clone(), 1)
            .expect("persist parent");
        store
            .persist("destination", MODEL, empty.clone(), 2)
            .expect("persist destination");
        store
            .persist("cached", MODEL, empty, 3)
            .expect("persist cached session");
        drop(store);
        let (store, persisted, clock) = SessionStore::open(root.clone(), 1).expect("reopen store");
        assert_eq!(persisted.len(), 1);
        assert!(persisted
            .iter()
            .any(|archive| archive.session_id == "cached"));

        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        server.session_store = Some(store);
        install_test_archives(&mut server, persisted);
        server.clock = clock;
        let (_, source) =
            fork_chat_session(&server, Some("child"), "parent").expect("fork uncached parent");
        assert!(matches!(
            source,
            ChatPlanSource::ExplicitFork { ref parent_id } if parent_id == "parent"
        ));
        let error = match fork_chat_session(&server, Some("destination"), "parent") {
            Ok(_) => panic!("uncached destination replaced"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("must not replace an existing session"));
        drop(server);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn persisted_update_retires_old_lease_before_hibernation() {
        let root =
            std::env::temp_dir().join(format!("leone-session-update-hibernate-{}", Uuid::new_v4()));
        let (store, persisted, clock) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("old session");
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        server.session_store = Some(store);
        install_test_archives(&mut server, persisted);
        server.clock = clock;
        resident_cpu_source(&mut server, "session", cpu_options(1));
        let mut stored = server.sessions.remove("session").expect("resident session");
        stored.reference_lease = server
            .lease_session_reference("session")
            .expect("lease old reference");
        assert!(stored
            .reference_lease
            .as_ref()
            .is_some_and(|lease| { lease.state == SessionReferenceLeaseState::Restore }));

        let mut stored = Some(stored);
        persist_chat_session(&mut server, "session", &mut stored, false).expect("publish update");
        let published = server
            .persisted
            .get("session")
            .expect("published metadata")
            .blob_sha256
            .clone();
        assert!(server
            .sessions
            .get("session")
            .is_some_and(|session| session.reference_lease.is_none()));

        let stored = server.sessions.remove("session").expect("updated session");
        server
            .hibernate_stored("session".to_owned(), stored)
            .expect("hibernate updated session");
        let current = server
            .session_store
            .as_ref()
            .expect("session store")
            .load_metadata("session")
            .expect("load published metadata")
            .expect("published reference");
        assert_eq!(current.blob_sha256, published);
        let (_, reopened, _) = SessionStore::open(root.clone(), 8).expect("restart store");
        assert_eq!(
            reopened
                .iter()
                .find(|archive| archive.session_id == "session")
                .map(|archive| &archive.archive.blob_sha256),
            Some(&published)
        );
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn persisted_update_cancellation_discards_current_reference() {
        let root =
            std::env::temp_dir().join(format!("leone-session-update-cancel-{}", Uuid::new_v4()));
        let (store, persisted, clock) = SessionStore::open(root.clone(), 8).expect("new store");
        let empty = GenerationSession::<CpuBackend>::new().checkpoint();
        store
            .persist("session", MODEL, empty, 1)
            .expect("old session");
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        server.session_store = Some(store);
        install_test_archives(&mut server, persisted);
        server.clock = clock;
        resident_cpu_source(&mut server, "session", cpu_options(1));
        let mut stored = server.sessions.remove("session").expect("resident session");
        stored.reference_lease = server
            .lease_session_reference("session")
            .expect("lease old reference");
        let mut stored = Some(stored);
        persist_chat_session(&mut server, "session", &mut stored, false).expect("publish update");

        server
            .discard_session("session")
            .expect("cancel updated session");
        assert!(server
            .session_store
            .as_ref()
            .expect("session store")
            .load_metadata("session")
            .expect("load cancelled metadata")
            .is_none());
        let quarantine_entries = fs::read_dir(root.join("quarantine"))
            .expect("quarantine directory")
            .count();
        assert_eq!(quarantine_entries, 0);
        let (_, reopened, _) = SessionStore::open(root.clone(), 8).expect("restart store");
        assert!(reopened.is_empty());
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn owner_held_discard_retries_without_a_recovery_slot() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        resident_cpu_source(&mut server, "session", cpu_options(1));
        let prompt = server
            .sessions
            .get("session")
            .expect("resident session")
            .generation
            .evaluated_tokens()
            .to_vec();
        assert_eq!(
            server
                .select_session(&prompt, &cpu_options(1))
                .expect("select source")
                .as_deref(),
            Some("session")
        );

        let root = std::env::temp_dir().join(format!(
            "leone-session-owner-held-discard-{}",
            Uuid::new_v4()
        ));
        let (store, _, _) = SessionStore::open(root.clone(), 1).expect("new store");
        let checkpoint = server
            .sessions
            .get("session")
            .expect("resident session")
            .generation
            .checkpoint();
        store
            .persist("session", MODEL, checkpoint, 1)
            .expect("persist session");
        let recovery_slot = store
            .acquire_recovery_slot()
            .expect("fill recovery capacity");
        server.session_store = Some(store);

        fs::remove_dir(root.join("quarantine")).expect("remove quarantine directory");
        fs::write(root.join("quarantine"), b"blocked").expect("block quarantine path");
        server
            .discard_session("session")
            .expect_err("retain failed owner-held discard");
        assert!(server.session_discard_pending("session"));
        assert!(server
            .session_store
            .as_ref()
            .expect("session store")
            .reference_path("session")
            .exists());
        assert!(server
            .select_session(&prompt, &cpu_options(1))
            .expect("skip pending source")
            .is_none());
        assert_eq!(
            scheduler_prefix_reused_tokens(
                &server,
                "session",
                &ChatPlanSource::Continue,
                &prompt,
                &cpu_options(1),
            )
            .expect("pending prefix credit"),
            0
        );
        assert!(server.persisted_metadata("session").is_err());
        assert!(fork_chat_session(&server, Some("child"), "session").is_err());
        assert!(server
            .lease_prefix_reuse("session", server.context_limit)
            .is_err());
        assert!(server
            .lease_available_fork_session("session", server.context_limit)
            .is_err());
        assert!(server
            .lease_continue_session("session", server.context_limit)
            .is_err());
        assert!(server.session_discard_pending("session"));

        fs::remove_file(root.join("quarantine")).expect("remove quarantine blocker");
        fs::create_dir(root.join("quarantine")).expect("restore quarantine directory");
        let cold = server
            .lease_continue_session("session", server.context_limit)
            .expect("retry terminal cleanup");
        assert!(cold.generation.is_empty());
        assert!(!server.sessions.contains_key("session"));
        assert!(!server.hibernated.contains_key("session"));
        drop(cold);
        drop(recovery_slot);
        assert_session_absent_after_restart(root, server);
    }

    #[test]
    fn hibernated_eviction_retries_an_owner_held_discard() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        resident_cpu_source(&mut server, "session", cpu_options(1));
        let root = std::env::temp_dir().join(format!(
            "leone-session-hibernated-discard-{}",
            Uuid::new_v4()
        ));
        let (store, _, _) = SessionStore::open(root.clone(), 1).expect("new store");
        let checkpoint = server
            .sessions
            .get("session")
            .expect("resident session")
            .generation
            .checkpoint();
        store
            .persist("session", MODEL, checkpoint, 1)
            .expect("persist session");
        server.session_store = Some(store);
        let stored = server.sessions.remove("session").expect("resident session");
        server
            .hibernate_stored("session".to_owned(), stored)
            .expect("hibernate session");
        assert!(server.hibernated.contains_key("session"));

        let recovery_slot = server
            .session_store
            .as_ref()
            .expect("session store")
            .acquire_recovery_slot()
            .expect("fill recovery capacity");
        fs::remove_dir(root.join("quarantine")).expect("remove quarantine directory");
        fs::write(root.join("quarantine"), b"blocked").expect("block quarantine path");
        server
            .discard_session("session")
            .expect_err("retain failed hibernated discard");
        assert!(server.session_discard_pending("session"));
        assert!(server.hibernated.contains_key("session"));

        fs::remove_file(root.join("quarantine")).expect("remove quarantine blocker");
        fs::create_dir(root.join("quarantine")).expect("restore quarantine directory");
        assert!(server
            .discard_oldest_hibernated()
            .expect("retry hibernated cleanup"));
        assert!(!server.hibernated.contains_key("session"));
        drop(recovery_slot);
        assert_session_absent_after_restart(root, server);
    }

    #[test]
    fn deadline_discards_the_task_held_reference() {
        let (root, mut server) = persisted_cpu_server_default("session-deadline");
        let (_transport, client, incoming) = test_server_incoming();
        let mut task = test_chat_task(incoming, false, "session");
        task.stored.reference_lease = server
            .lease_session_reference("session")
            .expect("lease task reference");
        server
            .finish_deadline_task(task)
            .expect("finish deadline task");
        drop(client);
        assert_session_absent_after_restart(root, server);
    }

    #[test]
    fn disconnect_discards_the_task_held_reference() {
        let (root, mut server) = persisted_cpu_server_default("session-disconnect");
        let (_transport, client, incoming) = test_server_incoming();
        let mut task = test_chat_task(incoming, false, "session");
        task.stored.reference_lease = server
            .lease_session_reference("session")
            .expect("lease task reference");
        server
            .finish_disconnected_task(task)
            .expect("finish disconnected task");
        drop(client);
        assert_session_absent_after_restart(root, server);
    }

    #[test]
    fn quarantine_discards_the_task_held_reference() {
        let (root, mut server) = persisted_cpu_server_default("session-quarantine");
        let (_transport, client, incoming) = test_server_incoming();
        let mut task = test_chat_task(incoming, false, "session");
        task.stored.reference_lease = server
            .lease_session_reference("session")
            .expect("lease task reference");
        task.failure = Some("injected session failure".to_owned());
        task.failure_effect = Some(SessionFailureEffect::Quarantine);
        server
            .finish_failed_task(task)
            .expect("finish quarantined task");
        drop(client);
        assert_session_absent_after_restart(root, server);
    }

    #[test]
    fn cancelled_persist_discards_the_task_held_reference() {
        let (root, mut server) = persisted_cpu_server_default("session-cancelled-persist");
        let mut stored = Some(StoredSession {
            generation: GenerationSession::new(),
            metadata_allocation: None,
            reference_lease: server
                .lease_session_reference("session")
                .expect("lease task reference"),
            last_used: 0,
        });
        persist_chat_session(&mut server, "session", &mut stored, true)
            .expect("cancel persisted task");
        assert_session_absent_after_restart(root, server);
    }

    #[test]
    fn failed_persist_retains_the_task_checkpoint_and_source_reference() {
        let (root, mut server) = persisted_cpu_server_default("session-persist-failure");
        let mut generation = GenerationSession::new();
        server
            .runtime
            .generate_session_tokens(
                &mut generation,
                &[1, 2, 3, 4],
                cpu_options(1),
                |_| Ok(()),
                || false,
            )
            .expect("generate task checkpoint");
        let expected_tokens = generation.evaluated_tokens().to_vec();
        let mut stored = Some(StoredSession {
            generation,
            metadata_allocation: None,
            reference_lease: server
                .lease_session_reference("session")
                .expect("lease task reference"),
            last_used: 0,
        });
        let blocker = fill_host_budget(&server);
        let error = persist_chat_session(&mut server, "session", &mut stored, false)
            .expect_err("reject persistence allocation");
        assert!(preserves_session_on_failure(error.as_ref()));
        assert!(stored.is_some());
        drop(blocker);

        server.resolve_response_session_error("session", &mut stored, error.as_ref());
        assert!(stored.is_none());
        assert_eq!(
            server
                .sessions
                .get("session")
                .expect("retained task session")
                .generation
                .evaluated_tokens(),
            expected_tokens
        );
        assert!(server
            .session_store
            .as_ref()
            .expect("session store")
            .load_metadata("session")
            .expect("load source reference")
            .is_some());
        drop(server);
        let (_, reopened, _) = SessionStore::open(root.clone(), 8).expect("restart store");
        assert!(reopened
            .iter()
            .any(|archive| archive.session_id == "session"));
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn unchanged_prefill_failure_retains_a_replay_only_session_under_pressure() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        let mut source_options = cpu_options(1);
        source_options.mirostat = Some(MirostatConfig::new(5.0, 0.1).expect("Mirostat config"));
        resident_cpu_source(&mut server, "source", source_options);
        let checkpoint = server
            .sessions
            .remove("source")
            .expect("source session")
            .generation
            .checkpoint();
        let expected_tokens = checkpoint.evaluated_tokens().to_vec();
        assert!(!expected_tokens.is_empty());
        assert!(checkpoint.mirostat().is_some());

        let root =
            std::env::temp_dir().join(format!("leone-session-replay-retention-{}", Uuid::new_v4()));
        let (store, persisted, clock) = SessionStore::open(root.clone(), 8).expect("new store");
        store
            .persist("session", MODEL, checkpoint, 1)
            .expect("persist replay source");
        server.session_store = Some(store);
        install_test_archives(&mut server, persisted);
        server.clock = clock;
        let archive = server
            .persisted_metadata("session")
            .expect("load persisted metadata")
            .expect("persisted session");
        let restored = server
            .restore_persisted_continuation("session", archive, server.context_limit)
            .expect("restore replay-only continuation");

        let (_transport, client, incoming) = test_server_incoming();
        let mut task = test_chat_task(incoming, false, "session");
        task.stored = restored;
        task.prompt_tokens = vec![u32::MAX];
        task.transcript = task.prompt_tokens.clone();
        let mut continuation_options = cpu_options(1);
        continuation_options.mirostat =
            Some(MirostatConfig::new(5.0, 0.1).expect("Mirostat config"));
        task.options = continuation_options.clone();
        execute_task_prefill(&mut server, &mut task, 1).expect("run rejected prefill");
        assert_eq!(task.failure_effect, Some(SessionFailureEffect::Unchanged));
        assert!(task.stored.generation.checkpoint().mirostat().is_some());

        let blocker = fill_host_budget(&server);
        server
            .finish_failed_task(task)
            .expect("retain failed replay-only session");
        let mut retained = server
            .sessions
            .remove("session")
            .expect("retained replay-only session");
        assert!(retained.generation.checkpoint().mirostat().is_some());
        server
            .runtime
            .generate_session_tokens(
                &mut retained.generation,
                &expected_tokens,
                continuation_options,
                |_| Ok(()),
                || false,
            )
            .expect("retry retained continuation");
        assert!(retained.generation.checkpoint().mirostat().is_some());
        drop(blocker);
        drop(client);
        drop(server);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn metadata_growth_failure_keeps_the_session_owner() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        resident_cpu_source(&mut server, "session", cpu_options(1));
        let mut stored = server.sessions.remove("session").expect("resident session");
        let expected_tokens = stored.generation.evaluated_tokens().to_vec();
        stored.metadata_allocation = Some(ResidentMetadataLease::single(
            server
                .memory
                .host
                .allocate(1)
                .expect("small metadata lease"),
        ));
        let blocker = fill_host_budget(&server);

        assert!(server.insert_session("session".to_owned(), stored).is_err());
        assert_eq!(
            server
                .sessions
                .get("session")
                .expect("retained session owner")
                .generation
                .evaluated_tokens(),
            expected_tokens
        );
        drop(blocker);
    }

    #[test]
    fn scheduler_uses_resolved_logical_kv_budget() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        server.memory.logical_kv_reservation = 1 << 20;
        let policy = server_scheduler_policy(&server, 8, 4).expect("scheduler policy");
        assert_eq!(policy.max_reserved_kv_bytes, 1 << 20);
    }

    #[test]
    fn persisted_table_charge_reduces_shared_logical_kv() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        let limit = NonZeroU64::new(1 << 30).expect("shared memory limit");
        server.memory = shared_test_server_memory(limit);
        let unconstrained = server
            .memory
            .resolve_logical_kv(
                &server.runtime,
                server.kv_cache_dtype,
                server.context_limit,
                server.max_sessions,
            )
            .expect("unconstrained logical KV");
        let table_bytes = persisted_cache_table_bound(3).expect("persisted table bound");
        assert!(unconstrained > table_bytes);
        let blocker_bytes = limit
            .get()
            .checked_sub(unconstrained)
            .expect("logical KV fits shared memory");
        let blocker = server
            .memory
            .host
            .allocate(blocker_bytes)
            .expect("constrain shared memory");
        assert_eq!(
            server
                .memory
                .resolve_logical_kv(
                    &server.runtime,
                    server.kv_cache_dtype,
                    server.context_limit,
                    server.max_sessions,
                )
                .expect("logical KV before cache"),
            unconstrained
        );

        let tables =
            allocate_persisted_cache_tables(&server.memory, 3).expect("persisted cache tables");
        assert_eq!(
            server
                .memory
                .resolve_logical_kv(
                    &server.runtime,
                    server.kv_cache_dtype,
                    server.context_limit,
                    server.max_sessions,
                )
                .expect("logical KV after cache"),
            unconstrained - table_bytes
        );
        drop(tables);
        drop(blocker);
    }

    #[test]
    fn streamed_errors_keep_one_response_shape_after_headers() {
        let mut response = Vec::new();
        write_stream_error(&mut response, true, 400, "runtime failed", None).expect("stream error");
        let response = String::from_utf8(response).expect("UTF-8 response");
        assert!(!response.starts_with("HTTP/1.1"));
        assert!(response.contains("\"message\":\"runtime failed\""));
        assert!(response.contains("\"type\":\"invalid_request_error\""));
        assert!(response.contains("data: [DONE]"));
        assert!(response.ends_with("0\r\n\r\n"));
    }

    #[test]
    fn server_errors_use_server_error_type() {
        let mut response = Vec::new();
        write_error_with_origin(&mut response, 500, "receipt failed", None).expect("server error");
        let response = String::from_utf8(response).expect("UTF-8 response");
        assert!(response.contains("\"type\":\"server_error\""));
    }

    #[test]
    fn streamed_error_after_headers_has_one_wire_response() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let mut client = TcpStream::connect(listener.local_addr().expect("listener address"))
            .expect("connect test client");
        let (mut server, _) = listener.accept().expect("accept test client");
        write_stream_headers_origin(&mut server, "session", Some("https://client.example"))
            .expect("stream headers");
        write_stream_error(&mut server, true, 500, "receipt failed", None).expect("stream error");
        drop(server);

        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("read stream response");
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert_eq!(response.matches("HTTP/1.1").count(), 1);
        assert!(response.contains("Access-Control-Expose-Headers: X-Leone-Session\r\n"));
        assert!(response.contains("receipt failed"));
        assert!(response.contains("data: [DONE]"));
        assert!(response.ends_with("0\r\n\r\n"));
    }

    #[test]
    fn streamed_errors_before_headers_use_http_error_response() {
        let mut response = Vec::new();
        write_stream_error(
            &mut response,
            false,
            408,
            "the request deadline expired",
            Some("https://client.example".to_owned()),
        )
        .expect("HTTP error");
        let response = String::from_utf8(response).expect("UTF-8 response");
        assert!(response.starts_with("HTTP/1.1 408 Error\r\n"));
        assert!(response.contains("Access-Control-Allow-Origin: https://client.example\r\n"));
        assert!(response.contains("Access-Control-Expose-Headers: X-Leone-Session\r\n"));
        assert!(!response.contains("data: [DONE]"));
    }

    #[test]
    fn forwarded_identity_requires_explicit_trusted_proxy() {
        let mut headers = BTreeMap::new();
        headers.insert(
            "x-forwarded-for".to_owned(),
            "198.51.100.9, 192.0.2.44, 192.0.2.1".to_owned(),
        );
        let request = HttpRequest {
            method: "GET".to_owned(),
            path: "/health".to_owned(),
            headers,
            body: Vec::new(),
        };
        let peer = "127.0.0.1:8080".parse().expect("peer");
        assert_eq!(request_client_identity(&request, peer, &[]), peer.ip());
        assert_eq!(
            request_client_identity(&request, peer, &[peer.ip(), "192.0.2.1".parse().unwrap()],),
            "192.0.2.44".parse::<IpAddr>().expect("forwarded identity")
        );
        assert_eq!(request_client_identity(&request, peer, &[]), peer.ip());
    }

    #[test]
    fn mapped_proxy_identity_uses_one_client_quota_key() {
        let request = HttpRequest {
            method: "GET".to_owned(),
            path: "/health".to_owned(),
            headers: BTreeMap::from([("x-forwarded-for".to_owned(), "192.0.2.44".to_owned())]),
            body: Vec::new(),
        };
        let peer = "[::ffff:127.0.0.1]:8080".parse().expect("mapped peer");
        assert_eq!(
            request_client_identity(&request, peer, &["127.0.0.1".parse().unwrap()]),
            "192.0.2.44".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn invalid_forwarded_identity_falls_back_to_proxy() {
        let request = HttpRequest {
            method: "GET".to_owned(),
            path: "/health".to_owned(),
            headers: BTreeMap::from([(
                "x-forwarded-for".to_owned(),
                "198.51.100.9, 192.0.2.44:8080".to_owned(),
            )]),
            body: Vec::new(),
        };
        let peer = "127.0.0.1:8080".parse().expect("peer");
        assert_eq!(
            request_client_identity(&request, peer, &[peer.ip()]),
            peer.ip()
        );
    }

    #[test]
    fn repeated_forwarded_headers_are_combined_in_order() {
        let headers = parse_request_headers(
            ["X-Forwarded-For: 192.0.2.44", "X-Forwarded-For: 192.0.2.1"].into_iter(),
        )
        .expect("headers");
        assert_eq!(
            headers.get("x-forwarded-for").map(String::as_str),
            Some("192.0.2.44,192.0.2.1")
        );
    }

    #[test]
    fn repeated_session_headers_are_rejected_before_folding() {
        let error = parse_request_headers(
            ["X-Leone-Session: first", "X-Leone-Session: second"].into_iter(),
        )
        .expect_err("repeated session header");
        assert!(error
            .to_string()
            .contains("x-leone-session must appear once"));
    }

    #[test]
    fn trusted_proxy_transport_uses_rightmost_untrusted_hop() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            crate::transport::Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: vec!["127.0.0.1".parse().expect("proxy")],
            },
            Arc::clone(&stop),
            read_request,
            request_client_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(
                b"GET /health HTTP/1.1\r\nX-Forwarded-For: 198.51.100.9, 192.0.2.44, 127.0.0.1\r\n\r\n",
            )
            .expect("request");
        let incoming = receive_transport_request(&transport);
        assert_eq!(
            incoming.client_id,
            "192.0.2.44".parse::<IpAddr>().expect("client identity")
        );
        drop(incoming);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn cpu_prefix_reuse_rejects_changed_dtype_and_continuation_keeps_identity() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        let mut source_options = cpu_options(1);
        source_options.mirostat = Some(MirostatConfig::new(5.0, 0.1).expect("Mirostat config"));
        resident_cpu_source(&mut server, "source", source_options);
        let prompt = [1, 2, 3, 5];
        let mut incompatible = cpu_options(1);
        incompatible.kv_cache_dtype = KvCacheDtype::Q8;
        assert_eq!(
            super::scheduler_prefix_reused_tokens(
                &server,
                "source",
                &ChatPlanSource::Continue,
                &prompt,
                &incompatible,
            )
            .expect("incompatible prefix credit"),
            0
        );
        let (cold_id, cold_source) =
            super::chat_plan_session(&server, None, None, &prompt, &incompatible)
                .expect("cold implicit plan");
        assert!(matches!(cold_source, ChatPlanSource::Continue));
        assert_ne!(cold_id, "source");

        let continued = server
            .lease_session(&test_chat_plan(
                "source",
                ChatPlanSource::Continue,
                &prompt,
                cpu_options(1),
            ))
            .expect("explicit continuation");
        assert!(continued.generation.checkpoint().mirostat().is_some());
        assert!(!server.sessions.contains_key("source"));
    }

    #[test]
    fn cpu_explicit_fork_keeps_parent_and_request_controls() {
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        let mut source_options = cpu_options(1);
        source_options.mirostat = Some(MirostatConfig::new(5.0, 0.1).expect("Mirostat config"));
        resident_cpu_source(&mut server, "source", source_options);
        let prompt = [1, 2, 3, 5];
        let (child_id, source) = super::chat_plan_session(
            &server,
            Some("fork"),
            Some("source"),
            &prompt,
            &cpu_options(1),
        )
        .expect("explicit fork plan");
        assert_eq!(child_id, "fork");
        assert!(
            matches!(source, ChatPlanSource::ExplicitFork { ref parent_id } if parent_id == "source")
        );
        let child = server
            .lease_session(&test_chat_plan(&child_id, source, &prompt, cpu_options(1)))
            .expect("explicit fork lease");
        assert!(server.sessions.contains_key("source"));
        assert!(child.generation.checkpoint().mirostat().is_some());
    }

    #[test]
    fn oversized_request_returns_typed_payload_error() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            crate::transport::Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_request,
            request_client_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 4194305\r\n\r\n")
            .expect("request");
        client
            .write_all(b"partial body still uploading")
            .expect("body");
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        assert!(response.starts_with("HTTP/1.1 413 Payload Too Large"));
        assert!(response.contains("transport-body-too-large"));
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn expired_stream_preserves_http_and_chunk_boundaries() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            crate::transport::Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 4096,
                read_timeout: Duration::from_millis(50),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_request,
            request_client_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /stream HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_transport_request(&transport);
        let mut output = incoming.output.clone();
        output.mark_streaming();
        let header = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
        let role = b"data: role\n\n";
        let error = b"data: {\"error\":{\"message\":\"the request deadline expired\"}}\n\n";
        let done = b"data: [DONE]\n\n";
        let chunk = |payload: &[u8]| {
            let mut frame = Vec::new();
            write!(&mut frame, "{:x}\r\n", payload.len()).expect("chunk size");
            frame.extend_from_slice(payload);
            frame.extend_from_slice(b"\r\n");
            frame
        };
        output.write_all(header).expect("headers");
        output.write_all(&chunk(role)).expect("role chunk");
        assert!(output.begin_terminal_response());
        thread::sleep(Duration::from_millis(75));
        incoming.cancellation.cancel();
        output.write_all(&chunk(error)).expect("error chunk");
        output.write_all(&chunk(done)).expect("done chunk");
        output.write_all(b"0\r\n\r\n").expect("end chunk");
        drop(output);
        drop(incoming);
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = Vec::new();
        client.read_to_end(&mut response).expect("response");
        let mut expected = header.to_vec();
        expected.extend_from_slice(&chunk(role));
        expected.extend_from_slice(&chunk(error));
        expected.extend_from_slice(&chunk(done));
        expected.extend_from_slice(b"0\r\n\r\n");
        assert_eq!(response, expected);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn cold_response_preserves_prefill_replay_and_counts_one_decode_dispatch() {
        let root = test_server_root("request-replay");
        let model = root.join("model.gguf");
        write_test_server_model(&model);
        let mut executor = test_server_executor(&model, &root.join("receipts"), None);
        let (transport, mut client, incoming) =
            test_server_incoming_with(Duration::from_secs(5), 1 << 20);
        let mut task = test_chat_task(incoming, false, "cold");
        task.prompt_tokens = vec![0, 1, 0, 1];
        task.transcript = task.prompt_tokens.clone();
        task.request_replay = None;
        task.tokens.clear();
        task.remaining_tokens = 1;
        task.eos = false;
        executor.tasks.insert(RequestId(1), task);
        let output = executor
            .execute_prefill(&Dispatch {
                request_id: RequestId(1),
                token_budget: 4,
                dispatch_ns: 0,
                kind: DispatchKind::Prefill,
            })
            .expect("cold prefill");
        assert!(output.prefill.expect("prefill result").ready);
        let decoded = executor.execute(RequestId(1), 1).expect("decode dispatch");
        assert_eq!(decoded.tokens.len(), 1);
        let task = &executor.tasks[&RequestId(1)];
        assert_eq!(task.decode_quanta, 1);
        assert_eq!(
            task.phase_trace,
            [DispatchKind::Prefill, DispatchKind::Decode]
        );
        assert_eq!(task.stored.generation.last_replay().reused_tokens, 4);
        assert_eq!(
            task.request_replay.expect("request replay").computed_tokens,
            4
        );
        executor
            .finish(RequestId(1), RequestStatus::Finished)
            .expect("response");
        let response = read_test_response(&mut client);
        let (_, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        let response: Value = serde_json::from_str(body).expect("response JSON");
        let receipt: ResponseReceipt =
            serde_json::from_value(response["leone_receipt"].clone()).expect("response receipt");
        receipt.verify().expect("signed response");
        assert_eq!(receipt.claim.session.reuse_class, "cold");
        assert_eq!(receipt.claim.session.computed_tokens, 4);
        assert_eq!(receipt.claim.session.reused_tokens, 0);
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn fully_reused_requests_keep_their_class_without_a_prefill_dispatch() {
        for class in [
            SessionReuseClass::ExactRepeat,
            SessionReuseClass::DeviceFork,
            SessionReuseClass::HostWake,
        ] {
            let root = test_server_root("reused-request");
            let model = root.join("model.gguf");
            write_test_server_model(&model);
            let mut executor = test_server_executor(&model, &root.join("receipts"), None);
            let prompt = [0, 1, 0, 1];
            let mut source = GenerationSession::new();
            executor
                .server
                .runtime
                .generate_session_tokens(
                    &mut source,
                    &prompt,
                    GenerateOptions::greedy(1),
                    |_| Ok(()),
                    || false,
                )
                .expect("source generation");
            let generation = match class {
                SessionReuseClass::DeviceFork => {
                    executor.server.runtime.fork_session(&source).expect("fork")
                }
                SessionReuseClass::HostWake => {
                    let snapshot = executor
                        .server
                        .runtime
                        .hibernate_session(&mut source)
                        .expect("hibernate");
                    executor
                        .server
                        .runtime
                        .wake_session(&snapshot)
                        .expect("wake")
                }
                _ => source,
            };
            let (transport, mut client, incoming) =
                test_server_incoming_with(Duration::from_secs(5), 1 << 20);
            let mut task = test_chat_task(incoming, false, "reused");
            task.stored.generation = generation;
            task.prompt_tokens = prompt.to_vec();
            task.transcript = prompt.to_vec();
            task.request_replay = None;
            task.tokens.clear();
            task.remaining_tokens = 1;
            task.eos = false;
            executor.tasks.insert(RequestId(1), task);
            assert_eq!(
                executor
                    .execute(RequestId(1), 1)
                    .expect("decode")
                    .tokens
                    .len(),
                1
            );
            let task = &executor.tasks[&RequestId(1)];
            assert_eq!(task.phase_trace, [DispatchKind::Decode]);
            assert_eq!(
                task.request_replay.expect("request replay").reuse_class,
                class
            );
            executor
                .finish(RequestId(1), RequestStatus::Finished)
                .expect("response");
            let response = read_test_response(&mut client);
            let (_, body) = response.split_once("\r\n\r\n").expect("HTTP response");
            let response: Value = serde_json::from_str(body).expect("response JSON");
            let receipt: ResponseReceipt =
                serde_json::from_value(response["leone_receipt"].clone())
                    .expect("response receipt");
            receipt.verify().expect("signed response");
            assert_eq!(receipt.claim.session.reuse_class, reuse_class_name(class));
            assert_eq!(receipt.claim.session.computed_tokens, 0);
            assert_eq!(receipt.claim.session.reused_tokens, 4);
            drop(transport);
            remove_test_server_root(root);
        }
    }

    #[test]
    fn uncommitted_prefill_cannot_publish_a_success_receipt() {
        let root = test_server_root("uncommitted-prefill");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let mut executor = test_server_executor(&model, &receipts, None);
        let (transport, mut client, incoming) =
            test_server_incoming_with(Duration::from_secs(5), 1 << 20);
        let mut task = test_chat_task(incoming, false, "uncommitted");
        task.request_replay = None;
        executor.tasks.insert(RequestId(1), task);
        executor
            .finish(RequestId(1), RequestStatus::Cancelled)
            .expect("typed response");
        let response = read_test_response(&mut client);
        assert!(!response.contains("\"leone_receipt\""));
        assert!(response.contains("\"error\""));
        assert!(!executor.server.sessions.contains_key("uncommitted"));
        assert!(!receipts.exists());
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn server_completion_claim_prevents_transport_timeout_response() {
        let root = test_server_root("completion-claim");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let mut executor =
            test_server_executor(&model, &receipts, Some(Duration::from_millis(300)));
        let (transport, mut client, incoming) =
            test_server_incoming_with(Duration::from_millis(200), 1 << 20);
        let task = test_chat_task(incoming, false, "completion-claim");

        executor.tasks.insert(RequestId(1), task);
        executor
            .finish(RequestId(1), RequestStatus::Finished)
            .expect("server completion");
        thread::sleep(Duration::from_millis(350));
        let response = read_test_response(&mut client);
        assert_eq!(response.matches("HTTP/1.1").count(), 1);
        assert!(!response.contains("408 Request Timeout"));
        assert!(response.contains("\"leone_receipt\""));
        assert!(executor.server.sessions.contains_key("completion-claim"));
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn writer_owned_deadline_discards_late_server_completion() {
        let root = test_server_root("writer-deadline");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let mut executor = test_server_executor(&model, &receipts, None);
        let (transport, mut client, incoming) = test_server_incoming();
        thread::sleep(Duration::from_millis(350));
        let task = test_chat_task(incoming, false, "writer-deadline");

        executor.tasks.insert(RequestId(1), task);
        executor
            .finish(RequestId(1), RequestStatus::Finished)
            .expect("late completion is request-local");
        let response = read_test_response(&mut client);
        assert_eq!(response.matches("HTTP/1.1").count(), 1);
        assert!(response.starts_with("HTTP/1.1 408 Request Timeout\r\n"));
        assert!(!executor.server.sessions.contains_key("writer-deadline"));
        assert!(!receipts.exists());
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn delayed_persistence_and_receipt_after_grace_discards_session() {
        let root = test_server_root("completion-delay");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        let session_store_root = root.join("sessions");
        write_test_server_model(&model);
        let mut executor =
            test_server_executor(&model, &receipts, Some(Duration::from_millis(350)));
        let (session_store, persisted, _) =
            SessionStore::open(session_store_root.clone(), 8).expect("session store");
        executor.server.session_store = Some(session_store);
        install_test_archives(&mut executor.server, persisted);
        let (transport, mut client, incoming) = test_server_incoming();
        let task = test_chat_task(incoming, false, "completion-delay");

        executor.tasks.insert(RequestId(1), task);
        executor
            .finish(RequestId(1), RequestStatus::Finished)
            .expect("disconnected completion is request-local");
        let response = read_test_response(&mut client);
        assert!(response.is_empty());
        assert!(!executor.server.sessions.contains_key("completion-delay"));
        let receipt_count = fs::read_dir(&receipts)
            .expect("receipt directory")
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
            .count();
        assert_eq!(receipt_count, 1);
        let reference_count = fs::read_dir(session_store_root.join("refs"))
            .expect("session references")
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
            .count();
        assert_eq!(reference_count, 0);
        let blob_count = fs::read_dir(session_store_root.join("blobs"))
            .expect("session blobs")
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
            .count();
        assert_eq!(blob_count, 1);
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn server_stream_completion_has_one_terminal_sequence_after_slow_read() {
        let root = test_server_root("stream-completion");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let mut executor = test_server_executor(&model, &receipts, None);
        let (transport, mut client, incoming) =
            test_server_incoming_with(Duration::from_secs(1), 16 << 20);
        let mut task = test_chat_task(incoming, true, "stream-completion");
        task.streamed.pending = vec![b'x'; 8 << 20];

        executor.tasks.insert(RequestId(1), task);
        executor
            .finish(RequestId(1), RequestStatus::Finished)
            .expect("stream completion");
        thread::sleep(Duration::from_millis(100));
        let response = read_test_response(&mut client);
        assert_eq!(response.matches("HTTP/1.1").count(), 1);
        assert_eq!(response.matches("data: [DONE]").count(), 1);
        assert_eq!(response.matches("0\r\n\r\n").count(), 1);
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn service_metrics_endpoint_uses_complete_bounded_response() {
        let root = test_server_root("metrics-endpoint");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let executor = test_server_executor(&model, &receipts, None);
        let policy = server_scheduler_policy(&executor.server, 4, 1).expect("scheduler policy");
        let mut service = ScheduledService::new(policy, executor).expect("service");
        let (transport, mut client, mut incoming) = test_server_incoming();
        incoming.request.path = "/debug/service-metrics".to_owned();
        admit_incoming_request(&mut service, incoming).expect("metrics request");
        let response = read_test_response(&mut client);
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("\"token_timing_definition\""));
        assert!(response.contains("\"collection_losses\""));
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("metrics response body");
        let value: serde_json::Value = serde_json::from_str(body).expect("metrics JSON");
        assert!(value["requests"]["values"].is_array());
        assert!(value["terminal_requests"].is_null());
        assert!(value["source_id"].is_string());
        assert!(value["process_instance_id"].is_string());
        assert!(value["physical_tracker_ledger"].is_string());
        assert_eq!(value["clock_domain"], "caller_monotonic_ns");
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn service_capabilities_endpoint_reports_fork_method() {
        let root = test_server_root("capabilities-endpoint");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let executor = test_server_executor(&model, &receipts, None);
        let policy = server_scheduler_policy(&executor.server, 4, 1).expect("scheduler policy");
        let mut service = ScheduledService::new(policy, executor).expect("service");
        let (transport, mut client, mut incoming) = test_server_incoming();
        incoming.request.path = "/debug/service-capabilities".to_owned();
        admit_incoming_request(&mut service, incoming).expect("capabilities request");
        let response = read_test_response(&mut client);
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("capabilities response body");
        let value: serde_json::Value = serde_json::from_str(body).expect("capabilities JSON");
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["branch_methods"], json!(["leone_fork_session"]));
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn cancellation_endpoint_records_client_terminal_state_and_reclaims_request() {
        let root = test_server_root("cancel-endpoint");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let executor = test_server_executor(&model, &receipts, None);
        let policy = server_scheduler_policy(&executor.server, 4, 1).expect("scheduler policy");
        let mut service = ScheduledService::new(policy, executor).expect("service");
        let (transport, mut request_client, incoming) =
            test_server_incoming_with(Duration::from_millis(500), 1 << 20);
        let request_id = RequestId(incoming.request_id);
        let wire_request_id = "chatcmpl-cancel-test".to_owned();
        service
            .executor_mut()
            .server
            .metrics
            .request_started(request_id, incoming.received_at_ns);
        let plan = ChatPlan {
            request: serde_json::from_value(json!({"model": "tiny", "messages": []}))
                .expect("chat request"),
            prompt_tokens: vec![1],
            options: GenerateOptions::greedy(1),
            session_id: "cancel-endpoint".to_owned(),
            source: ChatPlanSource::Continue,
            prefix_reused_tokens: 0,
            request_sha256: sha256_bytes(b"cancel-test"),
            created: 1,
            completion_id: wire_request_id.clone(),
            cors_origin: None,
            stop_sequences: Vec::new(),
        };
        let spec = RequestSpec {
            id: request_id,
            arrival_ns: scheduler_now_ns(),
            prompt_tokens: 1,
            prefix_reused_tokens: 0,
            max_output_tokens: 1,
            priority: 1,
            deadline_ns: Some(deadline_ns(incoming.deadline)),
        };
        service
            .executor_mut()
            .wire_request_ids
            .insert(wire_request_id.clone(), request_id);
        service.executor_mut().register_queued_output(
            request_id,
            wire_request_id.clone(),
            incoming.output.clone(),
        );
        let admission = ChatAdmission(RefCell::new(Some(PendingChat {
            plan,
            stream: incoming.output,
            cancellation: incoming.cancellation,
            deadline: incoming.deadline,
        })));
        service
            .admit(spec, &admission, scheduler_now_ns())
            .expect("chat admission");
        let cancel_path = format!("/debug/service-requests/{wire_request_id}/cancel");
        let (mut cancel_client, cancel_incoming) =
            send_test_request(&transport, "POST", &cancel_path, &[]);
        admit_incoming_request(&mut service, cancel_incoming).expect("cancel request");
        let cancel_response = read_test_response(&mut cancel_client);
        assert!(cancel_response.starts_with("HTTP/1.1 202 Accepted\r\n"));
        assert!(cancel_response.contains("\"reclaimed\":true"));
        let request_response = read_test_response(&mut request_client);
        assert_eq!(request_response.matches("HTTP/1.1").count(), 1);
        let terminal = service
            .executor()
            .server
            .metrics
            .terminal_status_by_id(&wire_request_id)
            .expect("terminal metrics");
        assert_eq!(terminal.outcome, "cancelled");
        assert!(terminal.reclaimed);
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn serve_batch_size_is_bounded_and_nonzero() {
        let defaults = parse(&["-m".to_owned(), "model.gguf".to_owned()]).expect("defaults");
        assert_eq!(defaults.batch_size, None);

        let explicit = parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--batch-size".to_owned(),
            "3".to_owned(),
        ])
        .expect("explicit batch size");
        assert_eq!(explicit.batch_size, Some(3));

        assert!(parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--batch-size".to_owned(),
            "0".to_owned(),
        ])
        .is_err());
    }

    #[test]
    fn serve_batch_size_rejects_backend_overflow_before_dispatch() {
        let cuda_limit = NonZeroUsize::new(8).expect("8 is nonzero");
        assert!(validate_backend_batch_size(8, cuda_limit).is_ok());
        assert!(validate_backend_batch_size(9, cuda_limit).is_err());
        let cpu_limit = NonZeroUsize::new(1).expect("1 is nonzero");
        assert!(validate_backend_batch_size(2, cpu_limit).is_err());
    }

    #[test]
    fn serve_batch_size_defaults_to_backend_capacity() {
        let cuda_limit = NonZeroUsize::new(8).expect("8 is nonzero");
        assert_eq!(
            resolve_backend_batch_size(None, cuda_limit).expect("cuda default"),
            8
        );
        let cpu_limit = NonZeroUsize::new(1).expect("1 is nonzero");
        assert_eq!(
            resolve_backend_batch_size(None, cpu_limit).expect("cpu default"),
            1
        );
        assert!(resolve_backend_batch_size(Some(8), cpu_limit).is_err());
        assert_eq!(
            resolve_backend_batch_size(Some(1), cpu_limit).expect("explicit cpu size"),
            1
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn serve_plan_kv_type_rejects_explicit_conflict() {
        let plan = crate::execution_plan::PlanSelection {
            decode: crate::execution_plan::PlanDecode::Eager,
            kv: crate::execution_plan::PlanKv::F16,
            prefill_chunk_tokens: 128,
        };
        assert!(effective_serve_kv(KvCacheDtype::Q8, true, Some(plan)).is_err());
        assert_eq!(
            effective_serve_kv(KvCacheDtype::Q8, false, Some(plan)).expect("planned KV"),
            KvCacheDtype::F16
        );
    }

    #[test]
    fn serve_prefill_chunk_override_is_positive() {
        let parsed = parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--prefill-chunk".to_owned(),
            "128".to_owned(),
        ])
        .expect("prefill chunk");
        assert_eq!(parsed.prefill_chunk_tokens, Some(128));
        assert!(parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--prefill-chunk".to_owned(),
            "0".to_owned(),
        ])
        .is_err());
    }

    #[test]
    fn service_memory_budgets_accept_auto_and_positive_bytes() {
        let parsed = parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--memory-budget-bytes".to_owned(),
            "auto".to_owned(),
            "--host-memory-budget-bytes".to_owned(),
            "4096".to_owned(),
            "--kv-reservation-budget-bytes".to_owned(),
            "2048".to_owned(),
        ])
        .expect("service memory budgets");
        assert_eq!(parsed.memory_budget, BudgetRequest::Auto);
        assert_eq!(
            parsed.host_memory_budget,
            BudgetRequest::Bytes(std::num::NonZeroU64::new(4096).unwrap())
        );
        assert_eq!(
            parsed.kv_reservation_budget,
            BudgetRequest::Bytes(std::num::NonZeroU64::new(2048).unwrap())
        );
        assert!(parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--memory-budget-bytes".to_owned(),
            "0".to_owned(),
        ])
        .is_err());
    }

    fn explicit_service_memory_policy(
        topology: PolicyMemoryTopology,
        backend: BackendCapacityObservation,
        backend_owned_bytes: u64,
    ) -> ServiceMemoryPolicy {
        let limit = NonZeroU64::new(64).expect("test memory limit");
        resolve_policy(
            ServiceBudgetArgs {
                memory: BudgetRequest::Bytes(limit),
                host: BudgetRequest::Bytes(limit),
                kv_reservation: BudgetRequest::Auto,
            },
            ServiceMemoryInputs {
                topology,
                backend,
                host: HostMemoryObservation::Available {
                    total_bytes: 1 << 20,
                    available_bytes: 1 << 19,
                    source: HostMemorySource::LinuxMeminfo,
                    semantics: HostMemorySemantics::KernelAvailableEstimate,
                },
                backend_owned_bytes,
                host_owned_bytes: 0,
            },
        )
        .expect("test service memory policy")
    }

    #[test]
    fn discrete_service_budget_updates_existing_backend_tracker() {
        let mut backend = CpuBackend::new();
        let buffer = backend
            .allocate(leone::BufferLayout::f32(4).expect("test buffer layout"))
            .expect("test backend allocation");
        let before = backend.memory_accounting();
        let root_before = backend.memory_tracker_root();
        let policy = explicit_service_memory_policy(
            PolicyMemoryTopology::Discrete,
            BackendCapacityObservation::Discrete {
                total_bytes: 1 << 20,
                available_bytes: 1 << 19,
            },
            before.live_bytes,
        );
        let host = install_service_memory_trackers(&mut backend, policy)
            .expect("install discrete service memory");
        let budget = MemoryBudget::limited(64).expect("test budget");
        let mut expected = before.clone();
        expected.budget = budget;
        assert_eq!(backend.memory_accounting(), expected);
        assert_eq!(
            backend.memory_tracker_root().owned_bytes(),
            root_before.owned_bytes()
        );
        assert_eq!(backend.memory_tracker_root().budget(), budget);
        assert_eq!(root_before.budget(), budget);
        assert_eq!(host.snapshot().budget, budget);

        let unchanged = backend.memory_accounting();
        assert!(matches!(
            backend.set_memory_budget(MemoryBudget::limited(8).expect("small budget")),
            Err(BackendError::Memory(MemoryError::BudgetBelowOwned { .. }))
        ));
        assert_eq!(backend.memory_accounting(), unchanged);
        drop((host, buffer));
        assert_eq!(root_before.owned_bytes(), 0);
    }

    #[test]
    fn shared_service_budget_keeps_backend_and_host_on_one_root() {
        let mut backend = CpuBackend::new();
        let policy = explicit_service_memory_policy(
            PolicyMemoryTopology::Cpu,
            BackendCapacityObservation::Cpu,
            0,
        );
        let host = install_service_memory_trackers(&mut backend, policy)
            .expect("install shared service memory");
        let budget = MemoryBudget::limited(64).expect("test budget");
        assert_eq!(backend.memory_tracker_root().budget(), budget);
        assert_eq!(host.root().budget(), budget);

        let backend_buffer = backend
            .allocate(leone::BufferLayout::f32(4).expect("test buffer layout"))
            .expect("test backend allocation");
        let host_allocation = host.allocate(16).expect("test host allocation");
        assert_eq!(backend.memory_accounting().live_bytes, 16);
        assert_eq!(host.snapshot().live_bytes, 16);
        assert_eq!(backend.memory_tracker_root().owned_bytes(), 32);
        drop((backend_buffer, host_allocation, host));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn cuda_startup_memory_preparation_keeps_constructor_tracker() {
        let mut backend = CudaBackend::new(0).expect("create CUDA backend");
        let memory = prepare_service_memory(
            &mut backend,
            BackendChoice::Cuda,
            ServiceBudgetArgs::default(),
        )
        .expect("prepare CUDA service memory");
        assert_eq!(
            backend.memory_tracker_root().owned_bytes(),
            backend.memory_accounting().live_bytes
        );
        drop(memory);
    }

    #[test]
    fn metrics_metadata_bound_grows_with_each_retained_class() {
        let base = metrics_metadata_bytes(1, 1, 1, 1).expect("base metadata bound");
        let more_tokens = metrics_metadata_bytes(1, 1, 2, 1).expect("token metadata bound");
        let more_requests = metrics_metadata_bytes(2, 1, 1, 1).expect("request metadata bound");
        let more_output = metrics_metadata_bytes(1, 1, 1, 2).expect("output metadata bound");
        assert!(more_tokens > base);
        assert!(more_requests > base);
        assert!(more_output > base);
    }

    #[test]
    fn failed_lease_retains_wire_identity_and_removes_mapping() {
        let root = test_server_root("metrics-failed-lease-identity");
        let model = root.join("model.gguf");
        let receipts = root.join("receipts");
        write_test_server_model(&model);
        let executor = test_server_executor(&model, &receipts, None);
        let policy = server_scheduler_policy(&executor.server, 4, 1).expect("policy");
        let mut service = ScheduledService::new(policy, executor).expect("service");
        let (transport, _client, incoming) = test_server_incoming();
        let id = RequestId(incoming.request_id);
        let wire = "chatcmpl-failed-lease".to_owned();
        service
            .executor_mut()
            .server
            .metrics
            .request_started(id, incoming.received_at_ns);
        let plan = ChatPlan {
            request: serde_json::from_value(json!({"model":"tiny","messages":[]}))
                .expect("request"),
            prompt_tokens: vec![1],
            options: GenerateOptions::greedy(1),
            session_id: "failed-child".to_owned(),
            source: ChatPlanSource::ExplicitFork {
                parent_id: "missing-parent".to_owned(),
            },
            prefix_reused_tokens: 0,
            request_sha256: sha256_bytes(b"failed-lease"),
            created: 1,
            completion_id: wire.clone(),
            cors_origin: None,
            stop_sequences: Vec::new(),
        };
        let spec = RequestSpec {
            id,
            arrival_ns: scheduler_now_ns(),
            prompt_tokens: 1,
            prefix_reused_tokens: 0,
            max_output_tokens: 1,
            priority: 1,
            deadline_ns: Some(deadline_ns(incoming.deadline)),
        };
        service
            .executor_mut()
            .wire_request_ids
            .insert(wire.clone(), id);
        service
            .executor_mut()
            .register_queued_output(id, wire.clone(), incoming.output.clone());
        let admission = ChatAdmission(RefCell::new(Some(PendingChat {
            plan,
            stream: incoming.output,
            cancellation: incoming.cancellation,
            deadline: incoming.deadline,
        })));
        assert!(admit_pending_chat(&mut service, spec, &admission).is_err());
        let executor = service.executor();
        let terminal = executor
            .server
            .metrics
            .terminal_status_by_id(&wire)
            .expect("wire terminal");
        assert_eq!(terminal.request_id, wire);
        assert_eq!(terminal.numeric_request_id, id.0);
        assert_eq!(terminal.delivery_status, DeliveryStatus::Queued);
        assert!(!executor.wire_request_ids.contains_key(&wire));
        assert!(executor
            .server
            .metrics
            .terminal_status_by_id(&id.0.to_string())
            .is_some());
        drop(transport);
        remove_test_server_root(root);
    }

    #[test]
    fn trace_memory_keeps_child_parent_and_host_ledgers_separate() {
        let root = leone::MemoryTrackerRoot::new(MemoryBudget::limited(1_024).unwrap());
        let child = MemoryTracker::child(MemoryBudget::limited(512).unwrap(), root.clone());
        let host = MemoryTracker::child(MemoryBudget::limited(512).unwrap(), root.clone());
        let _child_allocation = child
            .allocate(MemoryClass::ModelWeight, 32)
            .expect("child allocation");
        let _host_allocation = host
            .allocate(MemoryClass::ContractBuffer, 16)
            .expect("host allocation");
        let trace = trace_memory_snapshot(
            child.snapshot(),
            root.snapshot(),
            Some(host.snapshot()),
            MetricsMemoryTopology::CpuParent,
        );
        assert_eq!(trace.live_bytes, 32);
        assert_eq!(trace.parent.live_bytes, 48);
        assert_eq!(trace.host.expect("host ledger").live_bytes, 16);
        assert_eq!(trace.parent.topology, MetricsMemoryTopology::CpuParent);
    }

    #[test]
    fn transport_rejection_is_not_recorded_as_client_disconnect() {
        let context = FinishContext {
            client_cancelled: false,
            client_cancelled_at_ns: None,
            deadline_expired: false,
            deadline_expired_at_ns: None,
            disconnected: false,
            disconnected_at_ns: None,
            failed: false,
        };
        let result: Result<(), Box<dyn Error>> = Err(Box::new(io::Error::new(
            io::ErrorKind::WouldBlock,
            "response queue is full",
        )));
        let observation = finish_observation(
            RequestStatus::Cancelled,
            &context,
            &result,
            false,
            DeliveryFailurePhase::EnqueueRejected,
            None,
        );
        assert_eq!(observation.cause, TerminalCause::TransportFailure);
        assert!(!observation.disconnected);
        assert_eq!(observation.delivery_status, DeliveryStatus::EnqueueRejected);
    }

    #[test]
    fn late_transport_failure_overrides_earlier_disconnect_observation() {
        let context = FinishContext {
            client_cancelled: false,
            client_cancelled_at_ns: None,
            deadline_expired: false,
            deadline_expired_at_ns: None,
            disconnected: true,
            disconnected_at_ns: Some(12),
            failed: false,
        };
        let result: Result<(), Box<dyn Error>> = Err(Box::new(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "socket writer failed",
        )));
        let observation = finish_observation(
            RequestStatus::Finished,
            &context,
            &result,
            true,
            DeliveryFailurePhase::SocketWrite,
            Some(12),
        );
        assert_eq!(observation.cause, TerminalCause::TransportFailure);
        assert!(!observation.disconnected);
        assert_eq!(
            observation.delivery_status,
            DeliveryStatus::SocketWriteFailed
        );
    }

    #[test]
    fn queued_terminal_cause_keeps_deadline_and_disconnect_timestamps() {
        assert_eq!(
            queued_terminal_cause_with_timestamps(
                RequestStatus::DeadlineExpired,
                None,
                Some(55),
                None
            ),
            TerminalCause::DeadlineExpired {
                requested_at_ns: Some(55)
            }
        );
        assert_eq!(
            queued_terminal_cause_with_timestamps(RequestStatus::Cancelled, None, None, Some(66)),
            TerminalCause::Disconnected {
                requested_at_ns: Some(66)
            }
        );
        assert_eq!(
            queued_terminal_cause_with_timestamps(
                RequestStatus::Cancelled,
                Some(77),
                Some(88),
                None
            ),
            TerminalCause::DeadlineExpired {
                requested_at_ns: Some(88)
            }
        );
    }

    #[test]
    fn transport_limits_and_origin_policy_are_explicit() {
        let defaults = parse(&["-m".to_owned(), "model.gguf".to_owned()]).expect("defaults");
        assert_eq!(defaults.max_connections, 64);
        assert_eq!(defaults.max_connections_per_client, 4);
        assert_eq!(defaults.max_pending_requests, 64);
        assert!(defaults.cors_origins.is_empty());
        assert!(defaults.proxy_origin.is_none());
        assert!(defaults.trusted_proxy_ips.is_empty());

        let configured = parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--max-connections".to_owned(),
            "4".to_owned(),
            "--max-connections-per-client".to_owned(),
            "2".to_owned(),
            "--max-pending-requests".to_owned(),
            "3".to_owned(),
            "--max-output-bytes".to_owned(),
            "4096".to_owned(),
            "--request-timeout-ms".to_owned(),
            "1200".to_owned(),
            "--cors-origin".to_owned(),
            "https://client.example".to_owned(),
            "--proxy-origin".to_owned(),
            "https://proxy.example".to_owned(),
            "--trusted-proxy".to_owned(),
            "127.0.0.1".to_owned(),
        ])
        .expect("configured transport");
        assert_eq!(configured.max_connections, 4);
        assert_eq!(configured.max_connections_per_client, 2);
        assert_eq!(configured.max_pending_requests, 3);
        assert_eq!(configured.max_output_bytes, 4096);
        assert_eq!(configured.request_timeout_ms, 1200);
        assert_eq!(
            configured.cors_origins,
            vec!["https://client.example".to_owned()]
        );
        assert_eq!(
            configured.proxy_origin.as_deref(),
            Some("https://proxy.example")
        );
        assert_eq!(
            configured.trusted_proxy_ips,
            vec!["127.0.0.1".parse::<IpAddr>().unwrap()]
        );
        assert!(parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--cors-origin".to_owned(),
            "*".to_owned(),
        ])
        .is_err());
        assert!(parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--proxy-origin".to_owned(),
            "https://proxy.example\r\nX-Injected: true".to_owned(),
        ])
        .is_err());
    }

    #[test]
    fn service_trace_orders_prefill_and_resident_decode_events() {
        let mut trace = ServiceTraceRecorder {
            source_id: "source".to_owned(),
            process_instance_id: "process".to_owned(),
            workload_epoch: "epoch".to_owned(),
            events: Vec::new(),
            memory_samples: Vec::new(),
            active_prefill: BTreeMap::new(),
            logical_reserved_kv_bytes: 0,
            dropped_events: 0,
            dropped_memory_samples: 0,
            memory_topology: MetricsMemoryTopology::BackendChildFallback,
            host_tracker: None,
        };
        let prefill = Dispatch {
            request_id: RequestId(1),
            token_budget: 4,
            dispatch_ns: 0,
            kind: DispatchKind::Prefill,
        };
        let decode = Dispatch {
            request_id: RequestId(2),
            token_budget: 2,
            dispatch_ns: 0,
            kind: DispatchKind::Decode,
        };
        trace.begin_prefill(prefill.request_id);
        trace.prefill_event(
            prefill,
            leone::service::PrefillProgress {
                processed_tokens: 4,
                ready: false,
            },
            &[],
        );
        trace.decode_event_with_residents(decode, 2, &[RequestId(2), RequestId(3)]);
        trace.prefill_event(
            prefill,
            leone::service::PrefillProgress {
                processed_tokens: 4,
                ready: true,
            },
            &[],
        );
        trace.end_prefill(prefill.request_id);
        let count = trace.events.len();
        trace.decode_event(decode, 2);
        assert_eq!(trace.events.len(), count);
        let body = serde_json::to_value(ServiceTraceResponse {
            schema_version: SERVICE_TRACE_SCHEMA,
            clock_domain: PRESSURE_CLOCK_DOMAIN,
            source_id: trace.source_id.clone(),
            process_instance_id: trace.process_instance_id.clone(),
            workload_epoch: trace.workload_epoch.clone(),
            events: trace.events,
            memory_samples: Vec::new(),
            logical_reserved_kv_bytes: 0,
            dropped_events: trace.dropped_events,
            dropped_memory_samples: trace.dropped_memory_samples,
        })
        .expect("trace JSON");
        let events = body["events"].as_array().expect("events");
        assert_eq!(events[0]["kind"], "prefill_chunk");
        assert_eq!(events[1]["kind"], "resident_decode_progress");
        assert_eq!(events[2]["kind"], "prefill_chunk");
        assert_eq!(events[1]["during_prefill_request_id"], 1);
        assert_eq!(events[1]["emitted_tokens"], 2);
        assert_eq!(events[1]["resident_request_ids"], json!([2, 3]));
        assert_eq!(body["workload_epoch"], "epoch");
        assert_eq!(body["process_instance_id"], "process");
    }

    #[test]
    fn service_trace_keeps_multiple_prefill_ids_after_cancel() {
        let mut trace = ServiceTraceRecorder {
            source_id: "source".to_owned(),
            process_instance_id: "process".to_owned(),
            workload_epoch: "epoch".to_owned(),
            events: Vec::new(),
            memory_samples: Vec::new(),
            active_prefill: BTreeMap::new(),
            logical_reserved_kv_bytes: 0,
            dropped_events: 0,
            dropped_memory_samples: 0,
            memory_topology: MetricsMemoryTopology::BackendChildFallback,
            host_tracker: None,
        };
        let first_prefill = RequestId(7);
        let cancelled_prefill = RequestId(8);
        let decode = Dispatch {
            request_id: RequestId(9),
            token_budget: 1,
            dispatch_ns: 0,
            kind: DispatchKind::Decode,
        };
        trace.begin_prefill(first_prefill);
        trace.begin_prefill(cancelled_prefill);
        trace.decode_event(decode, 1);
        trace.end_prefill(cancelled_prefill);
        trace.decode_event(decode, 1);

        let body = serde_json::to_value(ServiceTraceResponse {
            schema_version: SERVICE_TRACE_SCHEMA,
            clock_domain: PRESSURE_CLOCK_DOMAIN,
            source_id: trace.source_id.clone(),
            process_instance_id: trace.process_instance_id.clone(),
            workload_epoch: trace.workload_epoch.clone(),
            events: trace.events,
            memory_samples: Vec::new(),
            logical_reserved_kv_bytes: 0,
            dropped_events: trace.dropped_events,
            dropped_memory_samples: trace.dropped_memory_samples,
        })
        .expect("trace JSON");
        let events = body["events"].as_array().expect("events");
        assert_eq!(events.len(), 2);
        assert!(events[0]["during_prefill_request_id"].is_null());
        assert_eq!(events[0]["during_prefill_request_ids"], json!([7, 8]));
        assert_eq!(events[1]["during_prefill_request_id"], 7);
        assert_eq!(events[1]["during_prefill_request_ids"], json!([7]));
    }

    struct TestTensor {
        name: &'static str,
        shape: Vec<u64>,
        dtype: u32,
        data: Vec<u8>,
    }

    fn test_server_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("leone-server-{label}-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("test server root");
        root
    }

    fn remove_test_server_root(root: PathBuf) {
        fs::remove_dir_all(root).expect("remove test server root");
    }

    fn test_server_executor(
        model: &Path,
        receipts: &Path,
        finalize_delay: Option<Duration>,
    ) -> ServerExecutor<CpuBackend> {
        let runtime = Runtime::load(CpuBackend::new(), model).expect("test server model");
        let memory = test_server_memory();
        let persisted_tables =
            allocate_persisted_cache_tables(&memory, 16).expect("persisted cache tables");
        let trace =
            ServiceTraceRecorder::new(&runtime, MetricsMemoryTopology::BackendChildFallback, None);
        ServerExecutor {
            server: Server {
                runtime,
                model_id: "tiny".to_owned(),
                model_sha256: MODEL.to_owned(),
                sessions: HashMap::new(),
                hibernated: HashMap::new(),
                persisted: persisted_tables.entries,
                _persisted_table_allocation: persisted_tables.allocation,
                persisted_cache_limit: 16,
                session_store: None,
                max_sessions: 4,
                max_hibernated_sessions: 4,
                clock: 0,
                kv_cache_dtype: KvCacheDtype::F16,
                #[cfg(feature = "cuda")]
                execution_plan: None,
                prefill_chunk_tokens: 4,
                context_limit: 16,
                receipts: receipts.to_owned(),
                signing_key: SigningKey::from_bytes(&[7; 32]),
                memory,
                cors_origins: Vec::new(),
                proxy_origin: None,
                metrics: ServerMetrics::new().expect("test metrics"),
                identity: ServiceIdentity::new("test-service", MODEL, MODEL, 1)
                    .expect("test identity"),
                test_finalize_delay: finalize_delay,
            },
            tasks: BTreeMap::new(),
            wire_request_ids: BTreeMap::new(),
            queued_outputs: BTreeMap::new(),
            terminal_outputs: VecDeque::new(),
            trace,
        }
    }

    fn test_server_incoming() -> (Transport<HttpRequest>, TcpStream, Incoming<HttpRequest>) {
        test_server_incoming_with(Duration::from_millis(50), 1 << 20)
    }

    fn test_server_incoming_with(
        read_timeout: Duration,
        max_output_bytes: usize,
    ) -> (Transport<HttpRequest>, TcpStream, Incoming<HttpRequest>) {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            crate::transport::Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes,
                read_timeout,
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            stop,
            read_request,
            request_client_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /test HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_transport_request(&transport);
        (transport, client, incoming)
    }

    fn send_test_request(
        transport: &Transport<HttpRequest>,
        method: &str,
        path: &str,
        body: &[u8],
    ) -> (TcpStream, Incoming<HttpRequest>) {
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        client.write_all(request.as_bytes()).expect("request");
        client.write_all(body).expect("request body");
        let incoming = receive_transport_request(transport);
        (client, incoming)
    }

    fn test_chat_task(
        incoming: Incoming<HttpRequest>,
        stream: bool,
        session_id: &str,
    ) -> ChatTask<CpuBackend> {
        if stream {
            let mut output = incoming.output.clone();
            output.mark_streaming();
            output
                .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                .expect("stream headers");
        }
        ChatTask {
            request: ChatRequest {
                model: "tiny".to_owned(),
                messages: Vec::new(),
                leone_template: ChatTemplateMode::Legacy,
                stream,
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                top_k: None,
                top_a: None,
                tfs_z: None,
                typical_p: None,
                repetition_penalty: None,
                repetition_window: None,
                stream_options: None,
                min_p: None,
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
                n: None,
                stop: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                logprobs: None,
                top_logprobs: None,
                _user: None,
                leone_session: None,
                leone_fork_session: None,
                mirostat_tau: None,
                mirostat_eta: None,
                draft_tokens: None,
                adaptive_speculation: None,
            },
            prompt_tokens: Vec::new(),
            transcript: Vec::new(),
            options: GenerateOptions::greedy(1),
            pending_prefill: None,
            request_replay: Some(leone::SessionReplay::default()),
            planned_prefix_tokens: 0,
            remaining_tokens: 0,
            session_id: session_id.to_owned(),
            stored: StoredSession {
                generation: GenerationSession::new(),
                metadata_allocation: None,
                reference_lease: None,
                last_used: 0,
            },
            stream: incoming.output,
            cancellation: incoming.cancellation,
            deadline: incoming.deadline,
            deadline_expired: false,
            deadline_expired_at_ns: None,
            request_sha256: sha256_bytes(b"test-request"),
            created: 1,
            completion_id: "chatcmpl-test".to_owned(),
            cors_origin: None,
            streamed: Utf8Stream::default(),
            stop: StopMatcher::new(Vec::new()),
            stop_hit: false,
            tool_header_stop: false,
            tokens: vec![0],
            eos: true,
            disconnected: false,
            disconnected_at_ns: None,
            failure: None,
            failure_effect: None,
            headers_written: stream,
            prefill_chunks: 0,
            prefill_tokens: 0,
            prefill_processed: 0,
            decode_quanta: 0,
            phase_trace: Vec::new(),
            token_boundaries_ns: Vec::new(),
            client_cancelled: false,
            client_cancelled_at_ns: None,
        }
    }

    fn test_pending_chat(incoming: Incoming<HttpRequest>, session_id: &str) -> PendingChat {
        let Incoming {
            output,
            cancellation,
            deadline,
            ..
        } = incoming;
        PendingChat {
            plan: test_chat_plan(session_id, ChatPlanSource::Continue, &[], cpu_options(1)),
            stream: output,
            cancellation,
            deadline,
        }
    }

    fn persisted_cpu_server(label: &str, recovery_limit: usize) -> (PathBuf, Server<CpuBackend>) {
        let root = std::env::temp_dir().join(format!("leone-{label}-{}", Uuid::new_v4()));
        let (store, _, clock) =
            SessionStore::open(root.clone(), recovery_limit).expect("new store");
        store
            .persist(
                "session",
                MODEL,
                GenerationSession::<CpuBackend>::new().checkpoint(),
                1,
            )
            .expect("persist session");
        let fixture = cpu_fixture();
        let mut server = cpu_server(fixture.path());
        server.session_store = Some(store);
        server.clock = clock;
        (root, server)
    }

    fn read_test_response(client: &mut TcpStream) -> String {
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("response timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        response
    }

    fn write_test_server_model(path: &Path) {
        let metadata = test_server_metadata();
        let tensors = test_server_tensors();
        let mut file = Vec::new();
        file.extend_from_slice(b"GGUF");
        put_u32(&mut file, 3);
        put_u64(&mut file, tensors.len() as u64);
        put_u64(&mut file, metadata.len() as u64);
        for entry in metadata {
            put_string(&mut file, entry.0);
            put_u32(&mut file, entry.1);
            file.extend_from_slice(&entry.2);
        }
        let mut offset = 0_u64;
        for tensor in &tensors {
            put_string(&mut file, tensor.name);
            put_u32(&mut file, tensor.shape.len() as u32);
            for dimension in &tensor.shape {
                put_u64(&mut file, *dimension);
            }
            put_u32(&mut file, tensor.dtype);
            put_u64(&mut file, offset);
            offset += tensor.data.len() as u64;
        }
        while file.len() % 32 != 0 {
            file.push(0);
        }
        for tensor in tensors {
            file.extend_from_slice(&tensor.data);
        }
        fs::write(path, file).expect("test model");
    }

    fn test_server_metadata() -> Vec<(&'static str, u32, Vec<u8>)> {
        let tokens = std::iter::once("a".to_owned())
            .chain(std::iter::once("b".to_owned()))
            .chain((0..254).map(|index| format!("tok{index}")))
            .collect();
        vec![
            ("general.architecture", 8, gguf_string("llama")),
            ("general.alignment", 4, gguf_u32(32)),
            ("llama.block_count", 4, gguf_u32(1)),
            ("llama.attention.head_count", 4, gguf_u32(1)),
            ("llama.attention.head_count_kv", 4, gguf_u32(1)),
            ("llama.embedding_length", 4, gguf_u32(256)),
            ("llama.feed_forward_length", 4, gguf_u32(256)),
            ("llama.vocab_size", 4, gguf_u32(256)),
            ("llama.context_length", 4, gguf_u32(16)),
            ("llama.attention.key_length", 4, gguf_u32(256)),
            ("llama.rope.freq_base", 6, gguf_f32(10_000.0)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                6,
                gguf_f32(0.00001),
            ),
            ("tokenizer.ggml.model", 8, gguf_string("gpt2")),
            ("tokenizer.ggml.pre", 8, gguf_string("llama-bpe")),
            ("tokenizer.ggml.tokens", 9, gguf_string_array(tokens)),
            ("tokenizer.ggml.token_type", 9, gguf_i32_array(256, 1)),
            ("tokenizer.ggml.merges", 9, gguf_string_array(Vec::new())),
        ]
    }

    fn test_server_tensors() -> Vec<TestTensor> {
        let quant_names = [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let norm_names = [
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ];
        let quant = vec![0; 144 * 256];
        let norm = (0..256)
            .flat_map(|_| 1.0_f32.to_le_bytes())
            .collect::<Vec<_>>();
        quant_names
            .into_iter()
            .map(|name| TestTensor {
                name,
                shape: vec![256, 256],
                dtype: 12,
                data: quant.clone(),
            })
            .chain(norm_names.into_iter().map(|name| TestTensor {
                name,
                shape: vec![256],
                dtype: 0,
                data: norm.clone(),
            }))
            .collect()
    }

    fn gguf_string(value: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_string(&mut bytes, value);
        bytes
    }

    fn gguf_string_array(values: Vec<String>) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 8);
        put_u64(&mut bytes, values.len() as u64);
        for value in values {
            put_string(&mut bytes, &value);
        }
        bytes
    }

    fn gguf_i32_array(length: usize, value: i32) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 5);
        put_u64(&mut bytes, length as u64);
        for _ in 0..length {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn gguf_u32(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn gguf_f32(value: f32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        put_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn put_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn receive_transport_request(transport: &Transport<HttpRequest>) -> Incoming<HttpRequest> {
        for _ in 0..100 {
            if let Some(request) = transport.try_recv().expect("transport receive") {
                return request;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("request did not arrive");
    }
}

fn is_disconnected(error: &(dyn Error + 'static)) -> bool {
    error
        .downcast_ref::<io::Error>()
        .map(|error| {
            matches!(
                error.kind(),
                io::ErrorKind::BrokenPipe
                    | io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
            )
        })
        .unwrap_or(false)
}

#[derive(Debug)]
struct StopMatcher {
    sequences: Vec<Vec<u8>>,
    pending: Vec<u8>,
    matched: bool,
}

impl StopMatcher {
    fn new(sequences: Vec<Vec<u8>>) -> Self {
        Self {
            sequences,
            pending: Vec::new(),
            matched: false,
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Vec<u8> {
        if self.matched {
            return Vec::new();
        }
        self.pending.extend_from_slice(bytes);
        if let Some(position) = find_stop_position(&self.pending, &self.sequences) {
            self.matched = true;
            let safe = self.pending[..position].to_vec();
            self.pending.clear();
            return safe;
        }
        let hold = self
            .sequences
            .iter()
            .map(|sequence| suffix_prefix_length(&self.pending, sequence))
            .max()
            .unwrap_or(0);
        let safe_len = self.pending.len().saturating_sub(hold);
        self.pending.drain(..safe_len).collect()
    }

    fn finish(&mut self) -> Vec<u8> {
        if self.matched {
            self.pending.clear();
            Vec::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }

    const fn hit(&self) -> bool {
        self.matched
    }
}

fn suffix_prefix_length(bytes: &[u8], sequence: &[u8]) -> usize {
    (1..sequence.len())
        .rev()
        .find(|length| bytes.ends_with(&sequence[..*length]))
        .unwrap_or(0)
}

fn is_transport_failure(error: &(dyn Error + 'static)) -> bool {
    error.downcast_ref::<io::Error>().is_some_and(|error| {
        matches!(
            error.kind(),
            io::ErrorKind::BrokenPipe
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::ConnectionAborted
                | io::ErrorKind::TimedOut
                | io::ErrorKind::WouldBlock
                | io::ErrorKind::WriteZero
        )
    })
}

#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        self.take(false)
    }

    fn finish(&mut self) -> Vec<String> {
        self.take(true)
    }

    fn take(&mut self, final_chunk: bool) -> Vec<String> {
        let mut output = Vec::new();
        while take_utf8_piece(&mut self.pending, final_chunk, &mut output) {}
        output
    }
}

fn take_utf8_piece(pending: &mut Vec<u8>, final_chunk: bool, output: &mut Vec<String>) -> bool {
    match std::str::from_utf8(pending) {
        Ok(text) => {
            if !text.is_empty() {
                output.push(text.to_owned());
                pending.clear();
            }
            false
        }
        Err(error) if error.valid_up_to() > 0 => {
            let count = error.valid_up_to();
            let text = String::from_utf8(pending.drain(..count).collect())
                .expect("validated UTF-8 prefix");
            output.push(text);
            true
        }
        Err(error) if error.error_len().is_some() => {
            let count = error.error_len().expect("checked above");
            pending.drain(..count);
            output.push("\u{fffd}".to_owned());
            true
        }
        Err(_) if final_chunk => {
            output.push(String::from_utf8_lossy(pending).into_owned());
            pending.clear();
            false
        }
        Err(_) => false,
    }
}
