//! Adapts backend-neutral service metrics to the HTTP server.

use leone::scheduler::{RequestId, RequestStatus};
use leone::service_metrics::BoundedHistory;
use leone::{
    Backend, LatencySummary, Measurement, MemoryClass, MetricsConfig, PhysicalBytesSample,
    ProcessMemorySample, RequestControlSample, Runtime, SchedulerReservationSample, ServiceMetrics,
    SessionReplay, SystemMemorySample,
};
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
#[cfg(target_os = "linux")]
use std::fs;
use std::io::{self, Write};
use uuid::Uuid;

const MAX_METRICS_WIRE_BYTES: usize = 512 * 1024;
const MAX_HTTP_RESPONSE_BYTES: usize = MAX_METRICS_WIRE_BYTES + 2 * 1024;
const MAX_TERMINAL_HISTORY: usize = 256;
const MAX_SLOW_CLIENT_INTERVALS: usize = 256;
const MAX_RETAINED_TEXT_BYTES: usize = 64;
const MAX_WORKLOAD_EPOCH_BYTES: usize = 128;
const OS_MEMORY_SAMPLE_INTERVAL_NS: u64 = 100_000_000;
pub(crate) const PRESSURE_CLOCK_DOMAIN: &str = "caller_monotonic_ns";
const TOKEN_TIMING_DEFINITION: &str =
    "engine token boundary timestamps define TTFT and ITL; SSE chunks are transport records; the request start is a declared server lifecycle boundary";
const PRESSURE_CLOCK_DEFINITION: &str =
    "socket pressure intervals and service trace sibling events use caller_monotonic_ns from one process-relative monotonic clock origin; client receive clocks require an explicit mapping";
const REALIZED_PROMPT_DEFINITION: &str = concat!(
    "realized_prompt_tokens sums each request's recorded SessionReplay prompt token counts when the request ends. ",
    "Full prompt reuse records a replay at the first decode dispatch, without prefill. ",
    "In-flight requests and requests without a recorded replay add nothing. ",
    "prefix_reuse_sources.reused_tokens counts plan-time credit for requests that leased a session. ",
    "planned_credit_tokens and credit_shortfall_tokens cover only requests counted in realized_prompt_tokens. ",
    "Reused tokens can exceed planned credit after a host wake or archive restore."
);
const PHYSICAL_PEAK_DEFINITION: &str =
    "physical_tracker_peak_bytes is the selected parent ledger peak_live_bytes; backend class samples are separate";
const FALLBACK_PEAK_DEFINITION: &str =
    "physical_tracker_peak_bytes is the backend child peak_live_bytes because the service did not supply a parent ledger";
const ADMITTED_PEAK_DEFINITION: &str =
    "peak_live_and_reserved_bytes is the selected parent ledger peak of live plus reserved bytes; it is an admission bound, not RSS";
#[cfg(target_os = "linux")]
const MEMORY_OBSERVATION_DEFINITION: &str =
    "process memory uses /proc/self/status VmRSS and VmSize; system available memory uses /proc/meminfo MemAvailable; system used memory is MemTotal minus MemAvailable; missing fields remain unavailable";
#[cfg(all(feature = "metal", target_os = "macos"))]
const MEMORY_OBSERVATION_DEFINITION: &str =
    "process memory uses task_info, system total memory uses sysctl hw.memsize, and system available memory estimates free_count plus external_page_count minus speculative_count (file-backed non-swap pages) from one native bridge call; those reads are not atomic; system used memory is the validated snapshot estimate; virtual bytes do not count as memory savings; missing fields remain unavailable";
#[cfg(not(any(target_os = "linux", all(feature = "metal", target_os = "macos"))))]
const MEMORY_OBSERVATION_DEFINITION: &str =
    "process and system host memory are unavailable on this target; missing fields remain unavailable";
const UNAVAILABLE_BODY: &[u8] = br#"{"error":{"type":"metrics_unavailable","code":"metrics_wire_limit","message":"the service metrics snapshot exceeds its wire bound","status":"unavailable"}}"#;

const METRICS_REQUEST_HISTORY: usize = 256;
const METRICS_LATENCY_SAMPLES: usize = 4096;
const LATENCY_HISTORY_COUNT: usize = 3;
const LATENCY_SUMMARY_SCRATCH_COUNT: usize = 3;
const METRICS_OBSERVATION_HISTORY: usize = 256;
const METRICS_OUTCOME_KINDS: usize = 64;
const METRICS_TEXT_BYTES: usize = 64;
const MAP_ENTRY_HEADROOM: usize = std::mem::size_of::<usize>() * 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefixSource {
    NewSession,
    LiveSession,
    HibernatedSession,
    PersistedSession,
    ForkParent,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForkSource {
    LiveSession,
    HibernatedSession,
    PersistedSession,
    Unavailable,
}

/// Identifies the ledger topology used for the parent memory peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MemoryTopology {
    /// Backend and host allocations use independent ledgers.
    Discrete,
    /// Backend and host allocations share one parent ledger.
    UnifiedParent,
    /// CPU and host allocations share one parent ledger.
    CpuParent,
    /// The parent ledger was not supplied by the service integration.
    BackendChildFallback,
}

/// Supplies parent-ledger peaks without coupling the adapter to server memory types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParentMemoryObservation {
    pub(crate) topology: MemoryTopology,
    pub(crate) peak_live_bytes: Option<u64>,
    pub(crate) peak_live_and_reserved_bytes: Option<u64>,
}

impl ParentMemoryObservation {
    /// Creates one parent observation from a memory-ledger snapshot.
    pub(crate) const fn new(
        topology: MemoryTopology,
        peak_live_bytes: Option<u64>,
        peak_live_and_reserved_bytes: Option<u64>,
    ) -> Self {
        Self {
            topology,
            peak_live_bytes,
            peak_live_and_reserved_bytes,
        }
    }
}

/// Records the cause of a terminal request state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalCause {
    /// The engine produced a terminal response.
    Completed,
    /// The client sent an explicit cancellation request.
    ClientCancellation { requested_at_ns: Option<u64> },
    /// The service deadline expired.
    DeadlineExpired { requested_at_ns: Option<u64> },
    /// The client connection ended while the request was active.
    Disconnected { requested_at_ns: Option<u64> },
    /// Admission rejected the request before execution.
    Rejected,
    /// Execution or finalization failed.
    ExecutionFailure,
    /// Admission or lease setup failed.
    AdmissionFailure,
    /// The transport failed while the request was terminal.
    TransportFailure,
    /// The caller has not supplied a structured cause yet.
    Unclassified,
}

/// Records the strongest response-delivery state observed by the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DeliveryStatus {
    /// No response bytes were submitted to the transport.
    NotAttempted,
    /// The transport accepted the response bytes for delivery.
    Queued,
    /// The transport rejected the response bytes or failed while writing.
    Failed,
    /// The transport rejected response bytes before enqueue.
    EnqueueRejected,
    /// The socket writer failed after enqueue.
    SocketWriteFailed,
}

/// Carries the terminal facts observed by the server lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TerminalObservation {
    pub(crate) completed_at_ns: u64,
    pub(crate) status: RequestStatus,
    pub(crate) cause: TerminalCause,
    pub(crate) disconnected: bool,
    pub(crate) failed: bool,
    pub(crate) reclaimed: bool,
    pub(crate) delivery_status: DeliveryStatus,
}

/// Identifies one measured output pressure event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlowClientCause {
    /// A socket write stayed blocked for the measured interval.
    WriteBlocked,
    /// The output queue crossed its measured high-water mark.
    OutputQueueHighWater,
}

impl SlowClientCause {
    fn name(self) -> &'static str {
        match self {
            Self::WriteBlocked => "write_blocked",
            Self::OutputQueueHighWater => "output_queue_high_water",
        }
    }
}

/// Holds the identity reported by the process that served the study.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ServiceIdentity {
    pub(crate) source_id: String,
    pub(crate) executable_sha256: String,
    pub(crate) model_sha256: String,
    pub(crate) process_start_ns: u64,
    pub(crate) process_start_clock: &'static str,
    pub(crate) process_instance_id: String,
}

impl ServiceIdentity {
    /// Validates the identity fields used to bind a receipt to one process.
    pub(crate) fn new(
        source_id: impl Into<String>,
        executable_sha256: impl Into<String>,
        model_sha256: impl Into<String>,
        process_start_ns: u64,
    ) -> Option<Self> {
        let identity = Self {
            source_id: source_id.into(),
            executable_sha256: executable_sha256.into(),
            model_sha256: model_sha256.into(),
            process_start_ns,
            process_start_clock: "unix_epoch_ns",
            process_instance_id: Uuid::new_v4().to_string(),
        };
        (valid_identity_text(&identity.source_id)
            && valid_digest(&identity.executable_sha256)
            && valid_digest(&identity.model_sha256)
            && identity.process_start_ns != 0
            && valid_identity_text(&identity.process_instance_id))
        .then_some(identity)
    }
}

#[derive(Debug, Default, Serialize)]
struct PrefixSourceCounts {
    new_session: u64,
    live_session: u64,
    hibernated_session: u64,
    persisted_session: u64,
    fork_parent: u64,
    unavailable: u64,
    reused_tokens: u64,
}

/// Prompt token counts from recorded replays of requests that ended.
///
/// A request records its replay after prefill or when decode reuses the full prompt.
/// `planned_credit_tokens` is the plan-time credit of the same requests.
/// `credit_shortfall_tokens` sums `planned - reused` where that difference is positive.
#[derive(Debug, Default, Serialize)]
struct RealizedPromptCounts {
    requests: u64,
    prompt_tokens: u64,
    reused_tokens: u64,
    replayed_tokens: u64,
    computed_tokens: u64,
    planned_credit_tokens: u64,
    credit_shortfall_tokens: u64,
}

#[derive(Debug, Default, Serialize)]
struct ForkSourceCounts {
    live_session: u64,
    hibernated_session: u64,
    persisted_session: u64,
    unavailable: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TerminalStatus {
    pub(crate) request_id: String,
    pub(crate) numeric_request_id: u64,
    #[serde(skip)]
    pub(crate) wire_request_id: Option<String>,
    pub(crate) outcome: String,
    pub(crate) reclaimed: bool,
    pub(crate) terminal_at_ns: u64,
    pub(crate) delivery_status: DeliveryStatus,
}

#[derive(Debug, Clone, Serialize)]
struct SlowClientInterval {
    numeric_request_id: u64,
    #[serde(rename = "request_id")]
    wire_request_id: Option<String>,
    start_ns: u64,
    end_ns: u64,
    reason: String,
}

#[derive(Debug, Serialize)]
struct SlowClientIntervalHistory<'a> {
    values: &'a VecDeque<SlowClientInterval>,
    capacity: usize,
    sample_count: u64,
    dropped_count: u64,
}

#[derive(Debug, Serialize)]
struct RequestHistory<'a> {
    values: &'a VecDeque<TerminalStatus>,
    capacity: usize,
    sample_count: u64,
    dropped_count: u64,
}

#[derive(Debug, Default, Serialize)]
struct CollectionLosses {
    request_started: u64,
    request_finished: u64,
    token_boundaries: u64,
    outcomes: u64,
    cancellation: u64,
    slow_client: u64,
    fork_latency: u64,
    physical_bytes: u64,
    scheduler_reservations: u64,
    process_memory: u64,
    system_memory: u64,
    terminal_outputs: u64,
    total: u64,
    overflowed: bool,
}

#[derive(Debug, Clone, Copy)]
enum LossKind {
    RequestStarted,
    RequestFinished,
    TokenBoundaries,
    Cancellation,
    SlowClient,
    ForkLatency,
    PhysicalBytes,
    SchedulerReservations,
    ProcessMemory,
    SystemMemory,
    TerminalOutputs,
}

#[derive(Debug, Serialize)]
struct MetricsWire<'a> {
    schema_version: u32,
    request_count: u64,
    active_request_count: u64,
    active_request_capacity: u64,
    outcomes: &'a BTreeMap<String, u64>,
    requests: RequestHistory<'a>,
    ttft: &'a LatencySummary,
    inter_token_latency: &'a LatencySummary,
    fork_latency: &'a LatencySummary,
    cancellation: &'a BoundedHistory<RequestControlSample>,
    slow_client: &'a BoundedHistory<RequestControlSample>,
    physical_bytes: &'a BoundedHistory<PhysicalBytesSample>,
    scheduler_reservations: &'a BoundedHistory<SchedulerReservationSample>,
    process_memory: &'a BoundedHistory<ProcessMemorySample>,
    system_memory: &'a BoundedHistory<SystemMemorySample>,
    physical_bytes_peak_by_class: &'a BTreeMap<String, u64>,
    scheduler_reservation_peak_bytes: &'a Measurement<u64>,
    process_memory_peak: &'a ProcessMemorySample,
    system_memory_peak: &'a SystemMemorySample,
    quantile_definition: &'static str,
    clock_domain: &'static str,
    scheduler_reservation_definition: &'static str,
    prefix_reuse_sources: &'a PrefixSourceCounts,
    fork_sources: &'a ForkSourceCounts,
    realized_prompt_tokens: &'a RealizedPromptCounts,
    realized_prompt_definition: &'static str,
    slow_client_intervals: SlowClientIntervalHistory<'a>,
    physical_tracker_peak_bytes: &'a Measurement<u64>,
    peak_live_and_reserved_bytes: &'a Measurement<u64>,
    memory_topology: MemoryTopology,
    physical_peak_definition: &'a str,
    admitted_peak_definition: &'static str,
    memory_observation_definition: &'static str,
    token_timing_definition: &'static str,
    pressure_clock_definition: &'static str,
    collection_errors: u64,
    collection_losses: &'a CollectionLosses,
    counter_overflowed: bool,
    memory_topology_conflict: bool,
    degraded: bool,
    terminal_transport_disconnects: u64,
    terminal_execution_failures: u64,
    workload_epoch: Option<&'a str>,
    source_id: &'a str,
    process_instance_id: &'a str,
    physical_tracker_ledger: &'static str,
    wire_limit_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct ServerMetrics {
    collector: ServiceMetrics,
    prefix_sources: PrefixSourceCounts,
    fork_sources: ForkSourceCounts,
    realized_prompts: RealizedPromptCounts,
    terminal_requests: VecDeque<TerminalStatus>,
    terminal_sample_count: u64,
    terminal_dropped_count: u64,
    slow_client_intervals: VecDeque<SlowClientInterval>,
    slow_client_interval_sample_count: u64,
    slow_client_interval_dropped_count: u64,
    physical_tracker_peak_bytes: Measurement<u64>,
    peak_live_and_reserved_bytes: Measurement<u64>,
    memory_topology: MemoryTopology,
    workload_epoch: Option<String>,
    source_id: String,
    process_instance_id: String,
    last_os_sample_ns: Option<u64>,
    collection_losses: CollectionLosses,
    collection_errors: u64,
    counter_overflowed: bool,
    memory_topology_conflict: bool,
    transport_disconnects: u64,
    execution_failures: u64,
}

/// Returns the bounded retained string and entry storage for server metrics history.
pub(crate) fn retained_metadata_bytes() -> usize {
    let terminal_entry = std::mem::size_of::<TerminalStatus>()
        + (MAX_RETAINED_TEXT_BYTES * 3)
        + MAX_RETAINED_TEXT_BYTES;
    let interval_entry = std::mem::size_of::<SlowClientInterval>()
        + (MAX_RETAINED_TEXT_BYTES * 2)
        + MAX_RETAINED_TEXT_BYTES;
    (MAX_TERMINAL_HISTORY * terminal_entry)
        + (MAX_SLOW_CLIENT_INTERVALS * interval_entry)
        + MAX_WORKLOAD_EPOCH_BYTES
}

/// Returns the bounded collector storage and snapshot clone budget.
pub(crate) fn collector_metadata_bytes(
    max_in_flight_requests: usize,
) -> Result<usize, leone::MetricsError> {
    let retained = collector_retained_bytes()?;
    let latency_scratch = latency_summary_scratch_bytes()?;
    let active = checked_product(
        max_in_flight_requests,
        std::mem::size_of::<u64>() * 4 + bounded_string_bytes() + MAP_ENTRY_HEADROOM,
    )?;
    let maps = checked_sum([
        checked_product(
            METRICS_OUTCOME_KINDS,
            bounded_string_bytes() + std::mem::size_of::<u64>() + MAP_ENTRY_HEADROOM,
        )?,
        checked_product(
            MemoryClass::ALL.len(),
            bounded_string_bytes() + std::mem::size_of::<u64>() + MAP_ENTRY_HEADROOM,
        )?,
    ])?;
    checked_sum([
        doubled(retained)?,
        active,
        doubled(maps)?,
        latency_scratch,
        checked_product(8, METRICS_TEXT_BYTES)?,
    ])
}

fn collector_retained_bytes() -> Result<usize, leone::MetricsError> {
    let request = checked_product(
        METRICS_REQUEST_HISTORY,
        bounded_slot_bytes::<leone::RequestMetric>(4),
    )?;
    let latency = latency_history_bytes(std::mem::size_of::<u64>())?;
    let controls = doubled(checked_product(
        METRICS_OBSERVATION_HISTORY,
        bounded_slot_bytes::<leone::RequestControlSample>(3),
    )?)?;
    let observations = collector_observation_bytes()?;
    checked_sum([request, latency, controls, observations])
}

fn latency_history_bytes(element_size: usize) -> Result<usize, leone::MetricsError> {
    checked_product(
        LATENCY_HISTORY_COUNT,
        checked_product(METRICS_LATENCY_SAMPLES, element_size)?,
    )
}

fn latency_summary_scratch_bytes() -> Result<usize, leone::MetricsError> {
    checked_product(
        LATENCY_SUMMARY_SCRATCH_COUNT,
        checked_product(METRICS_LATENCY_SAMPLES, std::mem::size_of::<f64>())?,
    )
}

fn collector_observation_bytes() -> Result<usize, leone::MetricsError> {
    checked_sum([
        checked_product(METRICS_OBSERVATION_HISTORY, physical_slot_bytes())?,
        checked_product(
            METRICS_OBSERVATION_HISTORY,
            bounded_slot_bytes::<leone::SchedulerReservationSample>(1),
        )?,
        checked_product(
            METRICS_OBSERVATION_HISTORY,
            bounded_slot_bytes::<leone::ProcessMemorySample>(2),
        )?,
        checked_product(
            METRICS_OBSERVATION_HISTORY,
            bounded_slot_bytes::<leone::SystemMemorySample>(2),
        )?,
    ])
}

/// Returns the maximum response allocation used by the metrics endpoint.
pub(crate) const fn metrics_response_bytes() -> usize {
    MAX_HTTP_RESPONSE_BYTES
}

fn bounded_slot_bytes<T>(string_fields: usize) -> usize {
    std::mem::size_of::<T>() + string_fields * bounded_string_bytes() + MAP_ENTRY_HEADROOM
}

fn physical_slot_bytes() -> usize {
    std::mem::size_of::<leone::PhysicalBytesSample>()
        + MemoryClass::ALL.len()
            * (bounded_string_bytes() + std::mem::size_of::<u64>() + MAP_ENTRY_HEADROOM)
}

const fn bounded_string_bytes() -> usize {
    METRICS_TEXT_BYTES
}

fn checked_product(left: usize, right: usize) -> Result<usize, leone::MetricsError> {
    left.checked_mul(right)
        .ok_or(leone::MetricsError::CounterOverflow)
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize, leone::MetricsError> {
    values.into_iter().try_fold(0_usize, |sum, value| {
        sum.checked_add(value)
            .ok_or(leone::MetricsError::CounterOverflow)
    })
}

fn doubled(value: usize) -> Result<usize, leone::MetricsError> {
    value
        .checked_mul(2)
        .ok_or(leone::MetricsError::CounterOverflow)
}

impl ServerMetrics {
    pub(crate) fn new() -> Result<Self, leone::MetricsError> {
        Self::with_capacity_and_epoch(256, None)
    }

    /// Creates an adapter whose active bound matches the service admission bound.
    pub(crate) fn with_capacity(
        max_in_flight_requests: usize,
        workload_epoch: impl Into<String>,
    ) -> Result<Self, leone::MetricsError> {
        let epoch = workload_epoch.into();
        if epoch.is_empty() || epoch.len() > MAX_WORKLOAD_EPOCH_BYTES {
            return Err(leone::MetricsError::FieldTooLong {
                field: "workload_epoch",
                max_bytes: MAX_WORKLOAD_EPOCH_BYTES,
            });
        }
        Self::with_capacity_and_epoch(max_in_flight_requests, Some(epoch))
    }

    /// Creates an adapter for the scheduler's active and queued request bound.
    pub(crate) fn with_scheduler_capacity(
        max_active_requests: usize,
        max_queued_requests: usize,
        workload_epoch: impl Into<String>,
    ) -> Result<Self, leone::MetricsError> {
        let max_in_flight_requests = max_active_requests
            .checked_add(max_queued_requests)
            .and_then(|value| value.checked_add(1))
            .ok_or(leone::MetricsError::CounterOverflow)?;
        Self::with_capacity(max_in_flight_requests, workload_epoch)
    }

    fn with_capacity_and_epoch(
        max_in_flight_requests: usize,
        workload_epoch: Option<String>,
    ) -> Result<Self, leone::MetricsError> {
        let collector = ServiceMetrics::new(Self::collector_config(max_in_flight_requests))?;
        Ok(Self {
            collector,
            prefix_sources: PrefixSourceCounts::default(),
            fork_sources: ForkSourceCounts::default(),
            realized_prompts: RealizedPromptCounts::default(),
            terminal_requests: VecDeque::with_capacity(MAX_TERMINAL_HISTORY),
            terminal_sample_count: 0,
            terminal_dropped_count: 0,
            slow_client_intervals: VecDeque::with_capacity(MAX_SLOW_CLIENT_INTERVALS),
            slow_client_interval_sample_count: 0,
            slow_client_interval_dropped_count: 0,
            physical_tracker_peak_bytes: Measurement::unavailable("memory_tracker_unavailable"),
            peak_live_and_reserved_bytes: Measurement::unavailable(
                "parent_memory_peak_unavailable",
            ),
            memory_topology: MemoryTopology::BackendChildFallback,
            workload_epoch,
            source_id: String::new(),
            process_instance_id: String::new(),
            last_os_sample_ns: None,
            collection_losses: CollectionLosses::default(),
            collection_errors: 0,
            counter_overflowed: false,
            memory_topology_conflict: false,
            transport_disconnects: 0,
            execution_failures: 0,
        })
    }

    pub(crate) fn bind_identity(&mut self, identity: &ServiceIdentity) {
        self.source_id = identity.source_id.clone();
        self.process_instance_id = identity.process_instance_id.clone();
    }

    fn collector_config(max_in_flight_requests: usize) -> MetricsConfig {
        MetricsConfig {
            max_in_flight_requests,
            max_request_history: METRICS_REQUEST_HISTORY,
            max_latency_samples: METRICS_LATENCY_SAMPLES,
            max_observation_history: METRICS_OBSERVATION_HISTORY,
            max_outcome_kinds: METRICS_OUTCOME_KINDS,
            max_identifier_bytes: METRICS_TEXT_BYTES,
            max_reason_bytes: METRICS_TEXT_BYTES,
            max_memory_classes: MemoryClass::ALL.len(),
        }
    }

    pub(crate) fn request_started(&mut self, request_id: RequestId, at_ns: u64) {
        let result = self
            .collector
            .request_started(request_id.0.to_string(), at_ns);
        self.record(LossKind::RequestStarted, result);
    }

    /// Records a terminal request and retains the response header ID for lookup.
    pub(crate) fn request_finished_with_wire_id(
        &mut self,
        request_id: RequestId,
        wire_request_id: &str,
        terminal: TerminalObservation,
    ) {
        self.finish_request(request_id, Some(wire_request_id), terminal);
    }

    fn finish_request(
        &mut self,
        request_id: RequestId,
        wire_request_id: Option<&str>,
        terminal: TerminalObservation,
    ) {
        let numeric_request_id = request_id.0;
        let request_id = request_id.0.to_string();
        let outcome = request_outcome_for_cause(
            terminal.status,
            terminal.cause,
            terminal.disconnected,
            terminal.failed,
        );
        let result =
            self.collector
                .request_finished(&request_id, Some(terminal.completed_at_ns), outcome);
        self.record(LossKind::RequestFinished, result);
        self.record_terminal(
            numeric_request_id,
            wire_request_id,
            outcome,
            terminal.completed_at_ns,
            terminal.reclaimed,
            terminal.delivery_status,
        );
        if terminal.disconnected {
            increment_counter(
                &mut self.transport_disconnects,
                &mut self.counter_overflowed,
            );
        }
        if terminal.failed {
            increment_counter(&mut self.execution_failures, &mut self.counter_overflowed);
        }
        self.record_cancellation(&request_id, terminal.completed_at_ns, terminal.cause);
    }

    fn record_cancellation(
        &mut self,
        request_id: &str,
        completed_at_ns: u64,
        cause: TerminalCause,
    ) {
        let (requested_at_ns, reason) = match cause {
            TerminalCause::ClientCancellation { requested_at_ns } => {
                (requested_at_ns, "client_cancellation")
            }
            TerminalCause::DeadlineExpired { requested_at_ns } => {
                (requested_at_ns, "deadline_expired")
            }
            TerminalCause::Disconnected { requested_at_ns } => {
                (requested_at_ns, "client_disconnected")
            }
            _ => return,
        };
        let result = match requested_at_ns {
            Some(requested_at_ns) => self.collector.cancellation_observed(
                request_id.to_owned(),
                requested_at_ns,
                Some(completed_at_ns),
                reason,
            ),
            None => self
                .collector
                .cancellation_unavailable(request_id.to_owned(), reason),
        };
        self.record(LossKind::Cancellation, result);
    }

    /// Records one measured output pressure event.
    pub(crate) fn slow_client_observed(
        &mut self,
        request_id: RequestId,
        blocked_ns: u64,
        cause: SlowClientCause,
    ) {
        let result = self.collector.slow_client_observed(
            request_id.0.to_string(),
            (blocked_ns != 0).then_some(blocked_ns),
            cause.name(),
        );
        self.record(LossKind::SlowClient, result);
    }

    pub(crate) fn slow_client_interval_observed(
        &mut self,
        request_id: RequestId,
        wire_request_id: &str,
        start_ns: u64,
        end_ns: u64,
        cause: SlowClientCause,
    ) {
        increment_counter(
            &mut self.slow_client_interval_sample_count,
            &mut self.counter_overflowed,
        );
        if self.slow_client_intervals.len() == MAX_SLOW_CLIENT_INTERVALS {
            self.slow_client_intervals.pop_front();
            increment_counter(
                &mut self.slow_client_interval_dropped_count,
                &mut self.counter_overflowed,
            );
        }
        self.slow_client_intervals.push_back(SlowClientInterval {
            numeric_request_id: request_id.0,
            wire_request_id: valid_request_id(wire_request_id).then(|| wire_request_id.to_owned()),
            start_ns,
            end_ns,
            reason: cause.name().to_owned(),
        });
    }

    pub(crate) fn terminal_output_dropped(&mut self) {
        checked_add_counter(&mut self.collection_errors, 1, &mut self.counter_overflowed);
        self.collection_losses.record_count(
            LossKind::TerminalOutputs,
            1,
            &mut self.counter_overflowed,
        );
    }

    pub(crate) fn transport_observation_dropped_count(&mut self, count: u64) {
        checked_add_counter(
            &mut self.collection_errors,
            count,
            &mut self.counter_overflowed,
        );
        self.collection_losses.record_count(
            LossKind::SlowClient,
            count,
            &mut self.counter_overflowed,
        );
    }

    /// Retains emitted engine boundaries when a later callback or write fails.
    pub(crate) fn token_boundaries_partial(
        &mut self,
        request_id: RequestId,
        timestamps_ns: &[u64],
        token_count: usize,
        fallback_at_ns: u64,
    ) {
        let request_id = request_id.0.to_string();
        // A callback can retain boundaries before a later callback or write fails.
        // Those boundaries remain engine observations even when the task has no
        // matching completed token vector.
        for at_ns in timestamps_ns {
            let result = self.collector.token_observed(&request_id, *at_ns);
            self.record(LossKind::TokenBoundaries, result);
        }
        if timestamps_ns.is_empty() && token_count != 0 {
            let result = self.collector.token_batch_observed(
                &request_id,
                fallback_at_ns,
                token_count as u64,
            );
            self.record(LossKind::TokenBoundaries, result);
        }
    }

    pub(crate) fn prefix_reuse_observed(&mut self, source: PrefixSource, reused_tokens: usize) {
        increment_prefix_source(
            &mut self.prefix_sources,
            source,
            &mut self.counter_overflowed,
        );
        add_token_counter(
            &mut self.prefix_sources.reused_tokens,
            reused_tokens,
            &mut self.counter_overflowed,
        );
    }

    /// Adds one request's recorded prompt token counts when it ends.
    ///
    /// `planned_credit_tokens` is the plan-time credit the request carried.
    pub(crate) fn prefill_realized(
        &mut self,
        prompt_tokens: usize,
        planned_credit_tokens: usize,
        replay: SessionReplay,
    ) {
        let overflowed = &mut self.counter_overflowed;
        let counts = &mut self.realized_prompts;
        checked_add_counter(&mut counts.requests, 1, overflowed);
        for (counter, tokens) in [
            (&mut counts.prompt_tokens, prompt_tokens),
            (&mut counts.reused_tokens, replay.reused_tokens),
            (&mut counts.replayed_tokens, replay.replayed_tokens),
            (&mut counts.computed_tokens, replay.computed_tokens),
            (&mut counts.planned_credit_tokens, planned_credit_tokens),
            (
                &mut counts.credit_shortfall_tokens,
                planned_credit_tokens.saturating_sub(replay.reused_tokens),
            ),
        ] {
            add_token_counter(counter, tokens, overflowed);
        }
    }

    pub(crate) fn fork_observed(&mut self, source: ForkSource, latency_ns: u64) {
        increment_fork_source(&mut self.fork_sources, source, &mut self.counter_overflowed);
        let result = self.collector.fork_observed(latency_ns);
        self.record(LossKind::ForkLatency, result);
    }

    /// Samples runtime bytes and parent-ledger peaks supplied by the service.
    pub(crate) fn sample_runtime_with_parent<B: Backend>(
        &mut self,
        runtime: &Runtime<B>,
        at_ns: u64,
        reserved_bytes: u64,
        parent: ParentMemoryObservation,
    ) {
        let accounting = runtime.backend().memory_accounting();
        let bytes_by_class = MemoryClass::ALL
            .into_iter()
            .map(|class| (class.name().to_owned(), accounting.class(class).live_bytes))
            .collect::<BTreeMap<_, _>>();
        let result = self
            .collector
            .physical_bytes_observed(at_ns, bytes_by_class);
        self.record(LossKind::PhysicalBytes, result);
        let result = self
            .collector
            .scheduler_reservation_observed(at_ns, None, reserved_bytes);
        self.record(LossKind::SchedulerReservations, result);
        if self.should_sample_os(at_ns) {
            self.last_os_sample_ns = Some(at_ns);
            let observation = observe_os_memory();
            let result = self.collector.process_memory_observed(
                at_ns,
                observation.process_resident_bytes,
                observation.process_virtual_bytes,
            );
            self.record(LossKind::ProcessMemory, result);
            let result = self.collector.system_memory_observed(
                at_ns,
                observation.system_used_bytes,
                observation.system_available_bytes,
            );
            self.record(LossKind::SystemMemory, result);
        }
        self.observe_parent_peak(parent);
    }

    fn should_sample_os(&self, at_ns: u64) -> bool {
        self.last_os_sample_ns.is_none_or(|last| {
            at_ns
                .checked_sub(last)
                .is_some_and(|elapsed| elapsed >= OS_MEMORY_SAMPLE_INTERVAL_NS)
        })
    }

    fn observe_parent_peak(&mut self, parent: ParentMemoryObservation) {
        if self.memory_topology == MemoryTopology::BackendChildFallback
            || self.memory_topology == parent.topology
        {
            self.memory_topology = parent.topology;
        } else {
            self.memory_topology_conflict = true;
        }
        update_peak_measurement(
            &mut self.physical_tracker_peak_bytes,
            parent.peak_live_bytes,
            "parent_memory_peak_unavailable",
        );
        update_peak_measurement(
            &mut self.peak_live_and_reserved_bytes,
            parent.peak_live_and_reserved_bytes,
            "parent_memory_peak_unavailable",
        );
    }

    pub(crate) fn write_response<W: Write + ?Sized>(&self, stream: &mut W) -> io::Result<()> {
        stream.write_all(&self.response_bytes()?)
    }

    pub(crate) fn response_bytes(&self) -> io::Result<Vec<u8>> {
        let (status, body) = self.body();
        http_response(status, &body)
    }

    /// Returns one retained terminal state by its numeric or wire request ID.
    pub(crate) fn terminal_status_by_id(&self, request_id: &str) -> Option<TerminalStatus> {
        let numeric_request_id = request_id.parse::<u64>().ok();
        self.terminal_requests
            .iter()
            .rev()
            .find(|item| {
                item.request_id == request_id
                    || item.wire_request_id.as_deref() == Some(request_id)
                    || numeric_request_id == Some(item.numeric_request_id)
            })
            .cloned()
    }

    /// Builds one complete response for the terminal request status endpoint.
    pub(crate) fn terminal_response_bytes(&self, request_id: &str) -> io::Result<Vec<u8>> {
        let Some(status) = self.terminal_status_by_id(request_id) else {
            let body = br#"{"error":"request terminal state unavailable"}"#;
            return http_response(404, body);
        };
        let body = serialize_bounded(&status).map_err(|()| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "terminal request state exceeds wire bound",
            )
        })?;
        http_response(200, &body)
    }

    /// Marks a queued response as failed after a later transport write error.
    pub(crate) fn mark_delivery_failed(
        &mut self,
        wire_request_id: &str,
        phase: DeliveryStatus,
    ) -> bool {
        self.update_delivery_status(wire_request_id, phase)
    }

    pub(crate) fn update_delivery_status(
        &mut self,
        wire_request_id: &str,
        delivery_status: DeliveryStatus,
    ) -> bool {
        let Some(status) = self
            .terminal_requests
            .iter_mut()
            .rev()
            .find(|item| item.wire_request_id.as_deref() == Some(wire_request_id))
        else {
            return false;
        };
        status.delivery_status = delivery_status;
        true
    }

    fn record_terminal(
        &mut self,
        numeric_request_id: u64,
        wire_request_id: Option<&str>,
        outcome: &str,
        at_ns: u64,
        reclaimed: bool,
        delivery_status: DeliveryStatus,
    ) {
        increment_counter(
            &mut self.terminal_sample_count,
            &mut self.counter_overflowed,
        );
        if self.terminal_requests.len() == MAX_TERMINAL_HISTORY {
            self.terminal_requests.pop_front();
            increment_counter(
                &mut self.terminal_dropped_count,
                &mut self.counter_overflowed,
            );
        }
        let wire_request_id = wire_request_id
            .filter(|value| valid_request_id(value))
            .map(str::to_owned);
        let request_id = wire_request_id
            .clone()
            .unwrap_or_else(|| numeric_request_id.to_string());
        self.terminal_requests.push_back(TerminalStatus {
            request_id,
            numeric_request_id,
            wire_request_id,
            outcome: outcome.to_owned(),
            reclaimed,
            terminal_at_ns: at_ns,
            delivery_status,
        });
    }

    fn body(&self) -> (u16, Vec<u8>) {
        let snapshot = self.collector.snapshot();
        let wire = MetricsWire {
            schema_version: snapshot.schema_version,
            request_count: snapshot.request_count,
            active_request_count: snapshot.active_request_count,
            active_request_capacity: snapshot.active_request_capacity,
            outcomes: &snapshot.outcomes,
            requests: RequestHistory {
                values: &self.terminal_requests,
                capacity: MAX_TERMINAL_HISTORY,
                sample_count: self.terminal_sample_count,
                dropped_count: self.terminal_dropped_count,
            },
            ttft: &snapshot.ttft,
            inter_token_latency: &snapshot.inter_token_latency,
            fork_latency: &snapshot.fork_latency,
            cancellation: &snapshot.cancellation,
            slow_client: &snapshot.slow_client,
            physical_bytes: &snapshot.physical_bytes,
            scheduler_reservations: &snapshot.scheduler_reservations,
            process_memory: &snapshot.process_memory,
            system_memory: &snapshot.system_memory,
            physical_bytes_peak_by_class: &snapshot.physical_bytes_peak_by_class,
            scheduler_reservation_peak_bytes: &snapshot.scheduler_reservation_peak_bytes,
            process_memory_peak: &snapshot.process_memory_peak,
            system_memory_peak: &snapshot.system_memory_peak,
            quantile_definition: snapshot.quantile_definition,
            clock_domain: snapshot.clock_domain,
            scheduler_reservation_definition: snapshot.scheduler_reservation_definition,
            prefix_reuse_sources: &self.prefix_sources,
            fork_sources: &self.fork_sources,
            realized_prompt_tokens: &self.realized_prompts,
            realized_prompt_definition: REALIZED_PROMPT_DEFINITION,
            slow_client_intervals: SlowClientIntervalHistory {
                values: &self.slow_client_intervals,
                capacity: MAX_SLOW_CLIENT_INTERVALS,
                sample_count: self.slow_client_interval_sample_count,
                dropped_count: self.slow_client_interval_dropped_count,
            },
            physical_tracker_peak_bytes: &self.physical_tracker_peak_bytes,
            peak_live_and_reserved_bytes: &self.peak_live_and_reserved_bytes,
            memory_topology: self.memory_topology,
            physical_peak_definition: physical_peak_definition(self.memory_topology),
            admitted_peak_definition: ADMITTED_PEAK_DEFINITION,
            memory_observation_definition: MEMORY_OBSERVATION_DEFINITION,
            token_timing_definition: TOKEN_TIMING_DEFINITION,
            pressure_clock_definition: PRESSURE_CLOCK_DEFINITION,
            collection_errors: self.collection_errors,
            collection_losses: &self.collection_losses,
            counter_overflowed: self.counter_overflowed,
            memory_topology_conflict: self.memory_topology_conflict,
            degraded: self.counter_overflowed
                || self.memory_topology_conflict
                || self.collection_errors != 0,
            terminal_transport_disconnects: self.transport_disconnects,
            terminal_execution_failures: self.execution_failures,
            workload_epoch: self.workload_epoch.as_deref(),
            source_id: &self.source_id,
            process_instance_id: &self.process_instance_id,
            physical_tracker_ledger: physical_tracker_ledger(self.memory_topology),
            wire_limit_bytes: MAX_METRICS_WIRE_BYTES,
        };
        match serialize_bounded(&wire) {
            Ok(body) => (200, body),
            Err(()) => (503, UNAVAILABLE_BODY.to_vec()),
        }
    }

    fn record<T>(&mut self, kind: LossKind, result: Result<T, leone::MetricsError>) {
        if result.is_err() {
            increment_counter(&mut self.collection_errors, &mut self.counter_overflowed);
            self.collection_losses
                .record_count(kind, 1, &mut self.counter_overflowed);
        }
    }
}

impl CollectionLosses {
    fn record_count(&mut self, kind: LossKind, count: u64, overflowed: &mut bool) {
        let counter = if kind.is_control() {
            self.control_counter(kind)
        } else {
            self.memory_counter(kind)
        };
        checked_add_counter(counter, count, overflowed);
        checked_add_counter(&mut self.total, count, overflowed);
        self.overflowed |= *overflowed;
    }

    fn control_counter(&mut self, kind: LossKind) -> &mut u64 {
        match kind {
            LossKind::RequestStarted => &mut self.request_started,
            LossKind::RequestFinished => &mut self.request_finished,
            LossKind::TokenBoundaries => &mut self.token_boundaries,
            LossKind::Cancellation => &mut self.cancellation,
            LossKind::SlowClient => &mut self.slow_client,
            _ => unreachable!("control loss kind invariant"),
        }
    }

    fn memory_counter(&mut self, kind: LossKind) -> &mut u64 {
        match kind {
            LossKind::ForkLatency => &mut self.fork_latency,
            LossKind::PhysicalBytes => &mut self.physical_bytes,
            LossKind::SchedulerReservations => &mut self.scheduler_reservations,
            LossKind::ProcessMemory => &mut self.process_memory,
            LossKind::SystemMemory => &mut self.system_memory,
            LossKind::TerminalOutputs => &mut self.terminal_outputs,
            _ => unreachable!("memory loss kind invariant"),
        }
    }
}

impl LossKind {
    fn is_control(self) -> bool {
        matches!(
            self,
            Self::RequestStarted
                | Self::RequestFinished
                | Self::TokenBoundaries
                | Self::Cancellation
                | Self::SlowClient
        )
    }
}

#[cfg(test)]
fn request_outcome(status: RequestStatus, disconnected: bool, failed: bool) -> &'static str {
    request_outcome_for_cause(status, TerminalCause::Unclassified, disconnected, failed)
}

fn physical_peak_definition(topology: MemoryTopology) -> &'static str {
    if topology == MemoryTopology::BackendChildFallback {
        FALLBACK_PEAK_DEFINITION
    } else {
        PHYSICAL_PEAK_DEFINITION
    }
}

fn physical_tracker_ledger(topology: MemoryTopology) -> &'static str {
    match topology {
        MemoryTopology::Discrete => "backend_memory_tracker_root",
        MemoryTopology::UnifiedParent | MemoryTopology::CpuParent => "parent_memory_tracker_root",
        MemoryTopology::BackendChildFallback => "backend_child_tracker",
    }
}

fn request_outcome_for_cause(
    status: RequestStatus,
    cause: TerminalCause,
    disconnected: bool,
    failed: bool,
) -> &'static str {
    if failed {
        return "execution_error";
    }
    if disconnected {
        return "client_disconnected";
    }
    cause_outcome(cause).unwrap_or_else(|| status_outcome(status))
}

fn cause_outcome(cause: TerminalCause) -> Option<&'static str> {
    match cause {
        TerminalCause::ClientCancellation { .. } => Some("cancelled"),
        TerminalCause::DeadlineExpired { .. } => Some("deadline_expired"),
        TerminalCause::Disconnected { .. } => Some("client_disconnected"),
        TerminalCause::Rejected => Some("rejected"),
        TerminalCause::ExecutionFailure | TerminalCause::AdmissionFailure => {
            Some("execution_error")
        }
        TerminalCause::TransportFailure => Some("transport_error"),
        TerminalCause::Completed | TerminalCause::Unclassified => None,
    }
}

fn status_outcome(status: RequestStatus) -> &'static str {
    match status {
        RequestStatus::Finished => "finished",
        RequestStatus::Cancelled => "cancelled",
        RequestStatus::DeadlineExpired => "deadline_expired",
        RequestStatus::Rejected => "rejected",
        RequestStatus::Queued | RequestStatus::Running => "incomplete",
    }
}

fn increment_prefix_source(
    counts: &mut PrefixSourceCounts,
    source: PrefixSource,
    overflowed: &mut bool,
) {
    let counter = match source {
        PrefixSource::NewSession => &mut counts.new_session,
        PrefixSource::LiveSession => &mut counts.live_session,
        PrefixSource::HibernatedSession => &mut counts.hibernated_session,
        PrefixSource::PersistedSession => &mut counts.persisted_session,
        PrefixSource::ForkParent => &mut counts.fork_parent,
        PrefixSource::Unavailable => &mut counts.unavailable,
    };
    checked_add_counter(counter, 1, overflowed);
}

fn increment_fork_source(counts: &mut ForkSourceCounts, source: ForkSource, overflowed: &mut bool) {
    let counter = match source {
        ForkSource::LiveSession => &mut counts.live_session,
        ForkSource::HibernatedSession => &mut counts.hibernated_session,
        ForkSource::PersistedSession => &mut counts.persisted_session,
        ForkSource::Unavailable => &mut counts.unavailable,
    };
    checked_add_counter(counter, 1, overflowed);
}

fn increment_counter(counter: &mut u64, overflowed: &mut bool) {
    checked_add_counter(counter, 1, overflowed);
}

fn add_token_counter(counter: &mut u64, tokens: usize, overflowed: &mut bool) {
    let tokens = u64::try_from(tokens).unwrap_or_else(|_| {
        *overflowed = true;
        u64::MAX
    });
    checked_add_counter(counter, tokens, overflowed);
}

fn checked_add_counter(counter: &mut u64, value: u64, overflowed: &mut bool) {
    match counter.checked_add(value) {
        Some(next) => *counter = next,
        None => {
            *counter = u64::MAX;
            *overflowed = true;
        }
    }
}

fn update_peak_measurement(
    measurement: &mut Measurement<u64>,
    value: Option<u64>,
    unavailable_reason: &'static str,
) {
    let Some(value) = value else {
        if measurement.value().is_none() {
            *measurement = Measurement::unavailable(unavailable_reason);
        }
        return;
    };
    if measurement.value().is_none_or(|current| value > *current) {
        *measurement = Measurement::observed(value);
    }
}

fn valid_identity_text(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256
}

fn valid_request_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= 64
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub(crate) fn identity_response(identity: &ServiceIdentity) -> io::Result<Vec<u8>> {
    let body = serialize_bounded(identity).map_err(|()| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "service identity exceeds wire bound",
        )
    })?;
    http_response(200, &body)
}

fn serialize_bounded<T: Serialize>(value: &T) -> Result<Vec<u8>, ()> {
    let mut buffer = BoundedBuffer::new(MAX_METRICS_WIRE_BYTES);
    serde_json::to_writer(&mut buffer, value).map_err(|_| ())?;
    Ok(buffer.into_inner())
}

fn http_response(status: u16, body: &[u8]) -> io::Result<Vec<u8>> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Service Unavailable",
    };
    let mut buffer = BoundedBuffer::new(MAX_HTTP_RESPONSE_BYTES);
    write!(
        buffer,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: authorization, content-type, x-leone-session\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n",
        body.len()
    )?;
    buffer.write_all(body)?;
    Ok(buffer.into_inner())
}

struct BoundedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedBuffer {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self.bytes.len().checked_add(bytes.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::WriteZero, "metrics wire bound overflowed")
        })?;
        if next > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "metrics wire bound exceeded",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct OsMemoryObservation {
    process_resident_bytes: Option<u64>,
    process_virtual_bytes: Option<u64>,
    system_used_bytes: Option<u64>,
    system_available_bytes: Option<u64>,
}

#[cfg(target_os = "linux")]
fn observe_os_memory() -> OsMemoryObservation {
    let (process_resident_bytes, process_virtual_bytes) = process_memory();
    let (system_used_bytes, system_available_bytes) = system_memory();
    OsMemoryObservation {
        process_resident_bytes,
        process_virtual_bytes,
        system_used_bytes,
        system_available_bytes,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
/// Samples process and system memory with one macOS bridge call.
///
/// Available system bytes estimate free plus file-backed non-swap pages minus
/// speculative pages from one `host_statistics64` snapshot. Virtual process
/// bytes remain address-space telemetry and do not count as memory savings. The
/// task, sysctl, and host-statistics reads are not atomic.
fn observe_os_memory() -> OsMemoryObservation {
    let Ok(snapshot) = leone_metal::host_memory_snapshot() else {
        return OsMemoryObservation {
            process_resident_bytes: None,
            process_virtual_bytes: None,
            system_used_bytes: None,
            system_available_bytes: None,
        };
    };
    OsMemoryObservation {
        process_resident_bytes: snapshot.process_resident_bytes,
        process_virtual_bytes: snapshot.process_virtual_bytes,
        system_used_bytes: snapshot.system_used_bytes,
        system_available_bytes: snapshot.system_available_bytes,
    }
}

#[cfg(not(any(target_os = "linux", all(feature = "metal", target_os = "macos"))))]
fn observe_os_memory() -> OsMemoryObservation {
    OsMemoryObservation {
        process_resident_bytes: None,
        process_virtual_bytes: None,
        system_used_bytes: None,
        system_available_bytes: None,
    }
}

#[cfg(target_os = "linux")]
fn process_memory() -> (Option<u64>, Option<u64>) {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    (
        proc_kib_field(&status, "VmRSS:"),
        proc_kib_field(&status, "VmSize:"),
    )
}

#[cfg(target_os = "linux")]
fn system_memory() -> (Option<u64>, Option<u64>) {
    let Ok(meminfo) = fs::read_to_string("/proc/meminfo") else {
        return (None, None);
    };
    let total = proc_kib_field(&meminfo, "MemTotal:");
    let available = proc_kib_field(&meminfo, "MemAvailable:");
    system_memory_fields(total, available)
}

#[cfg(any(target_os = "linux", test))]
fn system_memory_fields(total: Option<u64>, available: Option<u64>) -> (Option<u64>, Option<u64>) {
    let used = total
        .zip(available)
        .and_then(|(total, available)| total.checked_sub(available));
    (used, available)
}

#[cfg(any(target_os = "linux", test))]
fn proc_kib_field(contents: &str, field: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let rest = line.strip_prefix(field)?.trim_start();
        let mut fields = rest.split_whitespace();
        let value = fields.next()?.parse::<u64>().ok()?;
        if fields.next()? != "kB" {
            return None;
        }
        value.checked_mul(1024)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_buffer_rejects_without_partial_append() {
        let mut buffer = BoundedBuffer::new(3);
        buffer.write_all(b"abc").unwrap();
        assert!(buffer.write_all(b"d").is_err());
        assert_eq!(buffer.into_inner(), b"abc");
    }

    #[test]
    fn collector_budget_covers_latency_histories_and_quantile_scratch() {
        let request = checked_product(
            METRICS_REQUEST_HISTORY,
            bounded_slot_bytes::<leone::RequestMetric>(4),
        )
        .unwrap();
        let controls = doubled(
            checked_product(
                METRICS_OBSERVATION_HISTORY,
                bounded_slot_bytes::<leone::RequestControlSample>(3),
            )
            .unwrap(),
        )
        .unwrap();
        let observations = collector_observation_bytes().unwrap();
        let expected_retained = checked_sum([
            request,
            latency_history_bytes(std::mem::size_of::<u64>()).unwrap(),
            controls,
            observations,
        ])
        .unwrap();
        assert_eq!(collector_retained_bytes().unwrap(), expected_retained);
        assert_eq!(
            latency_history_bytes(std::mem::size_of::<u64>()).unwrap(),
            METRICS_LATENCY_SAMPLES * LATENCY_HISTORY_COUNT * std::mem::size_of::<u64>()
        );
        assert_eq!(
            latency_summary_scratch_bytes().unwrap(),
            METRICS_LATENCY_SAMPLES * LATENCY_SUMMARY_SCRATCH_COUNT * std::mem::size_of::<f64>()
        );

        let active = checked_product(
            1,
            std::mem::size_of::<u64>() * 4 + bounded_string_bytes() + MAP_ENTRY_HEADROOM,
        )
        .unwrap();
        let maps = checked_sum([
            checked_product(
                METRICS_OUTCOME_KINDS,
                bounded_string_bytes() + std::mem::size_of::<u64>() + MAP_ENTRY_HEADROOM,
            )
            .unwrap(),
            checked_product(
                MemoryClass::ALL.len(),
                bounded_string_bytes() + std::mem::size_of::<u64>() + MAP_ENTRY_HEADROOM,
            )
            .unwrap(),
        ])
        .unwrap();
        let expected_budget = checked_sum([
            doubled(expected_retained).unwrap(),
            active,
            doubled(maps).unwrap(),
            latency_summary_scratch_bytes().unwrap(),
            checked_product(8, METRICS_TEXT_BYTES).unwrap(),
        ])
        .unwrap();
        assert_eq!(collector_metadata_bytes(1).unwrap(), expected_budget);
    }

    #[test]
    fn process_fields_parse_kib() {
        let contents = "VmRSS:       12 kB\nVmSize: 34 kB\n";
        assert_eq!(proc_kib_field(contents, "VmRSS:"), Some(12 * 1024));
        assert_eq!(proc_kib_field(contents, "VmSize:"), Some(34 * 1024));
        assert_eq!(proc_kib_field(contents, "VmPeak:"), None);
        assert_eq!(proc_kib_field("VmRSS: 12 MB\n", "VmRSS:"), None);
    }

    #[test]
    fn os_memory_sources_share_checked_total_available_contract() {
        assert_eq!(
            system_memory_fields(Some(100), Some(40)),
            (Some(60), Some(40))
        );
        assert_eq!(
            system_memory_fields(Some(100), Some(120)),
            (None, Some(120))
        );
        assert_eq!(system_memory_fields(None, Some(40)), (None, Some(40)));
    }

    #[test]
    fn metrics_body_declares_memory_observation_definition() {
        let metrics = ServerMetrics::new().unwrap();
        let body: serde_json::Value = serde_json::from_slice(&metrics.body().1).unwrap();
        assert_eq!(
            body["memory_observation_definition"],
            MEMORY_OBSERVATION_DEFINITION
        );
    }

    #[test]
    fn request_outcomes_preserve_disconnect_and_failure() {
        assert_eq!(
            request_outcome(RequestStatus::Finished, true, false),
            "client_disconnected"
        );
        assert_eq!(
            request_outcome(RequestStatus::Finished, true, true),
            "execution_error"
        );
        assert_eq!(
            request_outcome(RequestStatus::DeadlineExpired, false, false),
            "deadline_expired"
        );
    }

    #[test]
    fn realized_prompt_counts_stay_separate_from_planned_credit() {
        let mut metrics = ServerMetrics::new().unwrap();
        metrics.prefix_reuse_observed(PrefixSource::LiveSession, 6);
        let replay = |reused_tokens, computed_tokens| SessionReplay {
            reused_tokens,
            computed_tokens,
            ..SessionReplay::default()
        };
        metrics.prefill_realized(8, 6, replay(4, 4));
        metrics.prefill_realized(5, 0, replay(5, 0));
        metrics.prefill_realized(3, 2, replay(2, 1));

        let (status, body) = metrics.body();
        assert_eq!(status, 200);
        assert!(body.len() < MAX_METRICS_WIRE_BYTES);
        let wire: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(wire["prefix_reuse_sources"]["reused_tokens"], 6);
        let realized = &wire["realized_prompt_tokens"];
        assert_eq!(realized["requests"], 3);
        assert_eq!(realized["prompt_tokens"], 16);
        assert_eq!(realized["reused_tokens"], 11);
        assert_eq!(realized["replayed_tokens"], 0);
        assert_eq!(realized["computed_tokens"], 5);
        assert_eq!(realized["planned_credit_tokens"], 8);
        assert_eq!(realized["credit_shortfall_tokens"], 2);
        assert!(wire["realized_prompt_definition"].is_string());
        assert!(!metrics.counter_overflowed);
    }

    #[test]
    fn realized_prompt_counters_saturate_and_flag_overflow() {
        let mut metrics = ServerMetrics::new().unwrap();
        metrics.realized_prompts.reused_tokens = u64::MAX - 1;
        let replay = SessionReplay {
            reused_tokens: 2,
            ..SessionReplay::default()
        };
        metrics.prefill_realized(2, 2, replay);
        assert_eq!(metrics.realized_prompts.reused_tokens, u64::MAX);
        assert_eq!(metrics.realized_prompts.requests, 1);
        assert!(metrics.counter_overflowed);
    }

    #[test]
    fn response_buffer_contains_one_complete_json_body() {
        let mut metrics = ServerMetrics::new().unwrap();
        let request_id = RequestId(7);
        metrics.request_started(request_id, 10);
        metrics.token_boundaries_partial(request_id, &[20, 30], 2, 30);
        metrics.request_finished_with_wire_id(
            request_id,
            "7",
            TerminalObservation {
                completed_at_ns: 40,
                status: RequestStatus::Finished,
                cause: TerminalCause::Completed,
                disconnected: false,
                failed: false,
                reclaimed: true,
                delivery_status: DeliveryStatus::Queued,
            },
        );
        let mut response = Vec::new();
        metrics.write_response(&mut response).unwrap();
        let body_start = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("HTTP response has header terminator")
            + 4;
        let headers = &response[..body_start];
        assert!(std::str::from_utf8(headers)
            .expect("headers are UTF-8")
            .contains("HTTP/1.1 200 OK"));
        let body = &response[body_start..];
        serde_json::from_slice::<serde_json::Value>(body).expect("body is one JSON value");
        assert!(!body.is_empty());
    }

    #[test]
    fn partial_token_callbacks_keep_boundaries_before_task_failure() {
        let mut metrics = ServerMetrics::new().unwrap();
        let request_id = RequestId(11);
        metrics.request_started(request_id, 10);
        metrics.token_boundaries_partial(request_id, &[20, 30], 1, 30);
        let snapshot = metrics.collector.snapshot();
        assert_eq!(snapshot.ttft.sample_count, 1);
        assert_eq!(snapshot.inter_token_latency.sample_count, 1);
        assert_eq!(
            snapshot
                .inter_token_latency
                .samples
                .values()
                .copied()
                .collect::<Vec<_>>(),
            [10]
        );
    }

    #[test]
    fn terminal_cause_controls_cancellation_and_reclaim() {
        let mut metrics = ServerMetrics::with_capacity(4, "epoch-1").unwrap();
        let request_id = RequestId(8);
        metrics.request_started(request_id, 10);
        metrics.request_finished_with_wire_id(
            request_id,
            "session-wire-id",
            TerminalObservation {
                completed_at_ns: 30,
                status: RequestStatus::Cancelled,
                cause: TerminalCause::ClientCancellation {
                    requested_at_ns: Some(20),
                },
                disconnected: false,
                failed: false,
                reclaimed: true,
                delivery_status: DeliveryStatus::Queued,
            },
        );
        let terminal = metrics
            .terminal_status_by_id("session-wire-id")
            .expect("terminal state");
        assert_eq!(terminal.request_id, "session-wire-id");
        assert_eq!(terminal.numeric_request_id, 8);
        assert_eq!(terminal.wire_request_id.as_deref(), Some("session-wire-id"));
        assert_eq!(
            metrics
                .terminal_status_by_id("8")
                .expect("numeric terminal state")
                .wire_request_id
                .as_deref(),
            Some("session-wire-id")
        );
        assert_eq!(terminal.outcome, "cancelled");
        assert!(terminal.reclaimed);
        assert_eq!(terminal.delivery_status, DeliveryStatus::Queued);
        assert!(metrics.mark_delivery_failed("session-wire-id", DeliveryStatus::SocketWriteFailed));
        assert_eq!(
            metrics
                .terminal_status_by_id("session-wire-id")
                .expect("updated terminal state")
                .delivery_status,
            DeliveryStatus::SocketWriteFailed
        );
        let terminal_response = metrics.terminal_response_bytes("session-wire-id").unwrap();
        assert!(String::from_utf8(terminal_response)
            .unwrap()
            .contains("\"reclaimed\":true"));
        let body = metrics.body().1;
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["workload_epoch"], "epoch-1");
        assert_eq!(value["requests"]["values"][0]["reclaimed"], true);
        assert_eq!(
            value["requests"]["values"][0]["request_id"],
            "session-wire-id"
        );
        assert_eq!(value["requests"]["values"][0]["numeric_request_id"], 8);
        assert!(value["requests"]["values"][0]["wire_request_id"].is_null());
        assert!(value["terminal_requests"].is_null());
        assert_eq!(value["slow_client"]["values"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn terminal_output_eviction_has_its_own_collection_loss() {
        let mut metrics = ServerMetrics::new().unwrap();
        metrics.terminal_output_dropped();
        let value: serde_json::Value = serde_json::from_slice(&metrics.body().1).unwrap();
        assert_eq!(value["collection_losses"]["terminal_outputs"], 1);
        assert_eq!(value["collection_losses"]["slow_client"], 0);
    }

    #[test]
    fn scheduler_capacity_includes_active_queue_and_admission_slot() {
        let metrics = ServerMetrics::with_scheduler_capacity(4, 2, "epoch-2").unwrap();
        assert_eq!(metrics.collector.snapshot().active_request_capacity, 7);
    }

    #[test]
    fn internal_cancel_does_not_create_cancellation_or_slow_client_samples() {
        let mut metrics = ServerMetrics::new().unwrap();
        let request_id = RequestId(9);
        metrics.request_started(request_id, 10);
        metrics.request_finished_with_wire_id(
            request_id,
            "9",
            TerminalObservation {
                completed_at_ns: 30,
                status: RequestStatus::Cancelled,
                cause: TerminalCause::ExecutionFailure,
                disconnected: true,
                failed: true,
                reclaimed: false,
                delivery_status: DeliveryStatus::NotAttempted,
            },
        );
        let snapshot = metrics.collector.snapshot();
        assert!(snapshot.cancellation.values().next().is_none());
        assert!(snapshot.slow_client.values().next().is_none());
        assert_eq!(
            snapshot.requests.values().next().unwrap().outcome,
            "execution_error"
        );
    }

    #[test]
    fn measured_slow_client_event_keeps_typed_reason() {
        let mut metrics = ServerMetrics::new().unwrap();
        metrics.slow_client_observed(RequestId(10), 25, SlowClientCause::WriteBlocked);
        metrics.slow_client_observed(RequestId(11), 40, SlowClientCause::OutputQueueHighWater);
        let snapshot = metrics.collector.snapshot();
        let values = snapshot.slow_client.values().collect::<Vec<_>>();
        assert_eq!(values[0].reason, "write_blocked");
        assert_eq!(values[1].reason, "output_queue_high_water");
        assert_eq!(values[0].latency_ns.value(), Some(&25));
        metrics.slow_client_observed(RequestId(12), 0, SlowClientCause::WriteBlocked);
        let zero_status = metrics
            .collector
            .snapshot()
            .slow_client
            .values()
            .last()
            .unwrap()
            .latency_ns
            .status();
        assert_eq!(zero_status, leone::MeasurementStatus::Unavailable);
    }

    #[test]
    fn slow_client_intervals_keep_shared_clock_boundaries_and_join_ids() {
        let mut metrics = ServerMetrics::new().unwrap();
        metrics.slow_client_interval_observed(
            RequestId(12),
            "chatcmpl-12",
            100,
            140,
            SlowClientCause::WriteBlocked,
        );
        let body = metrics.body().1;
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let interval = &value["slow_client_intervals"]["values"][0];
        assert_eq!(interval["numeric_request_id"], 12);
        assert_eq!(interval["request_id"], "chatcmpl-12");
        assert!(interval["wire_request_id"].is_null());
        assert_eq!(interval["start_ns"], 100);
        assert_eq!(interval["end_ns"], 140);
    }

    #[test]
    fn parent_peak_is_separate_from_backend_classes() {
        let mut metrics = ServerMetrics::new().unwrap();
        metrics.observe_parent_peak(ParentMemoryObservation::new(
            MemoryTopology::CpuParent,
            Some(100),
            Some(140),
        ));
        metrics.observe_parent_peak(ParentMemoryObservation::new(
            MemoryTopology::CpuParent,
            Some(80),
            Some(120),
        ));
        assert_eq!(metrics.physical_tracker_peak_bytes.value(), Some(&100));
        assert_eq!(metrics.peak_live_and_reserved_bytes.value(), Some(&140));
        assert_eq!(metrics.memory_topology, MemoryTopology::CpuParent);
    }

    #[test]
    fn identity_requires_real_lowercase_digests_and_disables_cache() {
        let digest = "a".repeat(64);
        let identity =
            ServiceIdentity::new("source/build", &digest, &digest, 17).expect("valid identity");
        let second_identity =
            ServiceIdentity::new("source/build", &digest, &digest, 17).expect("valid identity");
        assert_ne!(
            identity.process_instance_id,
            second_identity.process_instance_id
        );
        assert!(ServiceIdentity::new("source/build", "A", &digest, 17).is_none());
        let response = identity_response(&identity).unwrap();
        let text = String::from_utf8(response).unwrap();
        assert!(text.contains("\"process_start_ns\":17"));
        assert!(text.contains("\"process_start_clock\":\"unix_epoch_ns\""));
        assert!(text.contains("\"process_instance_id\":"));
        assert!(text.contains("Cache-Control: no-store"));
    }

    #[test]
    fn all_structured_cause_and_topology_variants_are_constructible() {
        let topologies = [
            MemoryTopology::Discrete,
            MemoryTopology::UnifiedParent,
            MemoryTopology::CpuParent,
            MemoryTopology::BackendChildFallback,
        ];
        assert_eq!(topologies.len(), 4);
        let causes = [
            TerminalCause::Completed,
            TerminalCause::ClientCancellation {
                requested_at_ns: None,
            },
            TerminalCause::DeadlineExpired {
                requested_at_ns: None,
            },
            TerminalCause::Disconnected {
                requested_at_ns: None,
            },
            TerminalCause::Rejected,
            TerminalCause::ExecutionFailure,
            TerminalCause::AdmissionFailure,
            TerminalCause::TransportFailure,
            TerminalCause::Unclassified,
        ];
        assert_eq!(causes.len(), 9);
    }
}
