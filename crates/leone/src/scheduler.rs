use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// The stable identity of one scheduled request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub u64);

/// Fixed admission and service limits for one scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerPolicy {
    pub max_active_requests: u32,
    pub max_queued_requests: u32,
    pub max_reserved_kv_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub max_prompt_tokens: u64,
    pub max_output_tokens: u64,
    pub service_quantum_tokens: u32,
    pub urgent_window_ns: u64,
    pub max_prefix_credit_tokens: u64,
}

impl SchedulerPolicy {
    pub fn validate(self) -> Result<Self, SchedulerError> {
        if self.max_active_requests == 0 || self.max_queued_requests == 0 {
            return Err(SchedulerError::InvalidPolicy(
                "request limits must be nonzero",
            ));
        }
        if self.max_reserved_kv_bytes == 0 || self.kv_bytes_per_token == 0 {
            return Err(SchedulerError::InvalidPolicy(
                "KV byte limits must be nonzero",
            ));
        }
        if self.max_prompt_tokens == 0 || self.max_output_tokens == 0 {
            return Err(SchedulerError::InvalidPolicy(
                "token limits must be nonzero",
            ));
        }
        if self.service_quantum_tokens == 0 {
            return Err(SchedulerError::InvalidPolicy(
                "service quantum must be nonzero",
            ));
        }
        Ok(self)
    }
}

/// One bounded request submitted to the scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestSpec {
    pub id: RequestId,
    pub arrival_ns: u64,
    pub prompt_tokens: u64,
    pub prefix_reused_tokens: u64,
    pub max_output_tokens: u64,
    pub priority: u16,
    pub deadline_ns: Option<u64>,
}

/// A typed overload or request-bound rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdmissionReject {
    ActiveLimit,
    QueueLimit,
    KvCapacity,
    PromptLimit,
    OutputLimit,
    InvalidPrefix,
    InvalidPriority,
    ExpiredDeadline,
}

/// The result of one admission attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum AdmissionOutcome {
    Admitted { reserved_kv_bytes: u64 },
    Rejected { reason: AdmissionReject },
}

/// The terminal or runnable state of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RequestStatus {
    Queued,
    Running,
    Finished,
    Cancelled,
    DeadlineExpired,
    Rejected,
}

impl RequestStatus {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Finished | Self::Cancelled | Self::DeadlineExpired | Self::Rejected
        )
    }
}

/// One resident transaction selected for execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dispatch {
    pub request_id: RequestId,
    pub token_budget: u32,
    pub dispatch_ns: u64,
}

/// The result committed at one transaction boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantumCompletion {
    pub request_id: RequestId,
    pub emitted_tokens: u32,
    pub status: RequestStatus,
}

/// Aggregate scheduler counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub admission_attempts: u64,
    pub admitted_requests: u64,
    pub rejected_requests: u64,
    pub dispatched_quanta: u64,
    pub emitted_tokens: u64,
    pub completed_requests: u64,
    pub cancelled_requests: u64,
    pub expired_requests: u64,
    pub max_queue_depth: u32,
    pub max_reserved_kv_bytes: u64,
    pub max_cancellation_overshoot_tokens: u32,
}

/// An invalid scheduler transition or configuration.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("invalid scheduler policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("request {0:?} already exists")]
    DuplicateRequest(RequestId),
    #[error("request {0:?} does not exist")]
    UnknownRequest(RequestId),
    #[error("request {0:?} has not arrived")]
    FutureArrival(RequestId),
    #[error("request {0:?} is not the running request")]
    NotRunning(RequestId),
    #[error("request {id:?} emitted {emitted} tokens with a budget of {budget}")]
    QuantumExceeded {
        id: RequestId,
        emitted: u32,
        budget: u32,
    },
    #[error("a scheduler transaction is already running")]
    DispatchInFlight,
    #[error("scheduler byte or token accounting overflowed")]
    AccountingOverflow,
    #[error("service trace schema {0} is not supported")]
    TraceSchema(u32),
    #[error("service trace is invalid: {0}")]
    InvalidTrace(String),
    #[error("service trace JSON failed: {0}")]
    Json(String),
}

#[derive(Debug, Clone)]
struct RequestState {
    spec: RequestSpec,
    status: RequestStatus,
    reserved_kv_bytes: u64,
    remaining_tokens: u64,
    emitted_tokens: u64,
    virtual_runtime: u128,
    sequence: u64,
    cancellation_requested: bool,
}

/// Schedules batch-1 resident quanta under reserved-KV admission.
#[derive(Debug)]
pub struct ContinuousScheduler {
    policy: SchedulerPolicy,
    requests: BTreeMap<RequestId, RequestState>,
    running: Option<Dispatch>,
    reserved_kv_bytes: u64,
    next_sequence: u64,
    stats: SchedulerStats,
}

impl ContinuousScheduler {
    pub fn new(policy: SchedulerPolicy) -> Result<Self, SchedulerError> {
        Ok(Self {
            policy: policy.validate()?,
            requests: BTreeMap::new(),
            running: None,
            reserved_kv_bytes: 0,
            next_sequence: 0,
            stats: SchedulerStats::default(),
        })
    }

    pub fn admit(
        &mut self,
        spec: RequestSpec,
        now_ns: u64,
    ) -> Result<AdmissionOutcome, SchedulerError> {
        if self.requests.contains_key(&spec.id) {
            return Err(SchedulerError::DuplicateRequest(spec.id));
        }
        if spec.arrival_ns > now_ns {
            return Err(SchedulerError::FutureArrival(spec.id));
        }
        self.stats.admission_attempts += 1;
        let rejection = self.admission_rejection(&spec, now_ns)?;
        let reservation = self.insert_request(spec, rejection.as_ref())?;
        if let Some(reason) = rejection {
            self.stats.rejected_requests += 1;
            return Ok(AdmissionOutcome::Rejected { reason });
        }
        self.reserved_kv_bytes = self
            .reserved_kv_bytes
            .checked_add(reservation)
            .ok_or(SchedulerError::AccountingOverflow)?;
        self.stats.admitted_requests += 1;
        self.update_high_watermarks();
        Ok(AdmissionOutcome::Admitted {
            reserved_kv_bytes: reservation,
        })
    }

    pub fn dispatch(&mut self, now_ns: u64) -> Result<Option<Dispatch>, SchedulerError> {
        if self.running.is_some() {
            return Err(SchedulerError::DispatchInFlight);
        }
        self.expire_deadlines(now_ns);
        let selected = self
            .requests
            .values()
            .filter(|state| state.status == RequestStatus::Queued)
            .min_by(|left, right| compare_requests(left, right, now_ns, self.policy))
            .map(|state| state.spec.id);
        let Some(request_id) = selected else {
            return Ok(None);
        };
        let state = self
            .requests
            .get_mut(&request_id)
            .expect("selected request exists");
        let token_budget = u32::try_from(
            state
                .remaining_tokens
                .min(u64::from(self.policy.service_quantum_tokens)),
        )
        .map_err(|_| SchedulerError::AccountingOverflow)?;
        state.status = RequestStatus::Running;
        let dispatch = Dispatch {
            request_id,
            token_budget,
            dispatch_ns: now_ns,
        };
        self.running = Some(dispatch);
        self.stats.dispatched_quanta += 1;
        Ok(Some(dispatch))
    }

    pub fn complete_quantum(
        &mut self,
        request_id: RequestId,
        emitted_tokens: u32,
        eos: bool,
    ) -> Result<QuantumCompletion, SchedulerError> {
        let dispatch = self.running.ok_or(SchedulerError::NotRunning(request_id))?;
        if dispatch.request_id != request_id {
            return Err(SchedulerError::NotRunning(request_id));
        }
        if emitted_tokens > dispatch.token_budget {
            return Err(SchedulerError::QuantumExceeded {
                id: request_id,
                emitted: emitted_tokens,
                budget: dispatch.token_budget,
            });
        }
        let state = self
            .requests
            .get_mut(&request_id)
            .ok_or(SchedulerError::UnknownRequest(request_id))?;
        apply_quantum_state(state, &mut self.stats, emitted_tokens, eos)?;
        if state.status.is_terminal() {
            self.reserved_kv_bytes -= state.reserved_kv_bytes;
            state.reserved_kv_bytes = 0;
        }
        self.running = None;
        Ok(QuantumCompletion {
            request_id,
            emitted_tokens,
            status: state.status,
        })
    }

    fn insert_request(
        &mut self,
        spec: RequestSpec,
        rejection: Option<&AdmissionReject>,
    ) -> Result<u64, SchedulerError> {
        let reservation = if rejection.is_none() {
            self.reservation_bytes(&spec)?
        } else {
            0
        };
        let status = if rejection.is_none() {
            RequestStatus::Queued
        } else {
            RequestStatus::Rejected
        };
        let base_runtime = self
            .requests
            .values()
            .filter(|state| !state.status.is_terminal())
            .map(|state| state.virtual_runtime)
            .min()
            .unwrap_or(0);
        // 1024 keeps virtual runtime moving when priority is large.
        let credit = u128::from(
            spec.prefix_reused_tokens
                .min(self.policy.max_prefix_credit_tokens),
        )
        .saturating_mul(1_024)
        .checked_div(u128::from(spec.priority.max(1)))
        .unwrap_or(0);
        let id = spec.id;
        self.requests.insert(
            id,
            RequestState {
                remaining_tokens: spec.max_output_tokens,
                spec,
                status,
                reserved_kv_bytes: reservation,
                emitted_tokens: 0,
                virtual_runtime: base_runtime.saturating_sub(credit),
                sequence: self.next_sequence,
                cancellation_requested: false,
            },
        );
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SchedulerError::AccountingOverflow)?;
        Ok(reservation)
    }

    /// Aborts one running quantum and releases its reservation.
    ///
    /// The executor calls this when it cannot commit a quantum. No emitted
    /// token is recorded.
    pub fn abort_quantum(
        &mut self,
        request_id: RequestId,
    ) -> Result<RequestStatus, SchedulerError> {
        let dispatch = self.running.ok_or(SchedulerError::NotRunning(request_id))?;
        if dispatch.request_id != request_id {
            return Err(SchedulerError::NotRunning(request_id));
        }
        let state = self
            .requests
            .get_mut(&request_id)
            .ok_or(SchedulerError::UnknownRequest(request_id))?;
        state.status = RequestStatus::Cancelled;
        state.cancellation_requested = false;
        self.reserved_kv_bytes -= state.reserved_kv_bytes;
        state.reserved_kv_bytes = 0;
        self.stats.cancelled_requests += 1;
        self.running = None;
        Ok(state.status)
    }

    pub fn cancel(&mut self, request_id: RequestId) -> Result<RequestStatus, SchedulerError> {
        let state = self
            .requests
            .get_mut(&request_id)
            .ok_or(SchedulerError::UnknownRequest(request_id))?;
        if state.status == RequestStatus::Queued {
            state.status = RequestStatus::Cancelled;
            self.reserved_kv_bytes -= state.reserved_kv_bytes;
            state.reserved_kv_bytes = 0;
            self.stats.cancelled_requests += 1;
        } else if state.status == RequestStatus::Running {
            state.cancellation_requested = true;
        }
        Ok(state.status)
    }

    pub fn status(&self, id: RequestId) -> Option<RequestStatus> {
        self.requests.get(&id).map(|state| state.status)
    }
    pub fn emitted_tokens(&self, id: RequestId) -> Option<u64> {
        self.requests.get(&id).map(|state| state.emitted_tokens)
    }
    pub const fn reserved_kv_bytes(&self) -> u64 {
        self.reserved_kv_bytes
    }
    pub const fn stats(&self) -> SchedulerStats {
        self.stats
    }
    pub fn has_runnable_requests(&self) -> bool {
        self.requests
            .values()
            .any(|state| matches!(state.status, RequestStatus::Queued | RequestStatus::Running))
    }

    fn admission_rejection(
        &self,
        spec: &RequestSpec,
        now_ns: u64,
    ) -> Result<Option<AdmissionReject>, SchedulerError> {
        if let Some(reason) = self.basic_admission_rejection(spec, now_ns) {
            return Ok(Some(reason));
        }
        let (active, queued) = self.request_counts();
        if active >= usize::try_from(self.policy.max_active_requests).unwrap_or(usize::MAX) {
            return Ok(Some(AdmissionReject::ActiveLimit));
        }
        if queued >= usize::try_from(self.policy.max_queued_requests).unwrap_or(usize::MAX) {
            return Ok(Some(AdmissionReject::QueueLimit));
        }
        let reservation = self.reservation_bytes(spec)?;
        if self
            .reserved_kv_bytes
            .checked_add(reservation)
            .ok_or(SchedulerError::AccountingOverflow)?
            > self.policy.max_reserved_kv_bytes
        {
            return Ok(Some(AdmissionReject::KvCapacity));
        }
        Ok(None)
    }

    fn basic_admission_rejection(
        &self,
        spec: &RequestSpec,
        now_ns: u64,
    ) -> Option<AdmissionReject> {
        if spec.prefix_reused_tokens > spec.prompt_tokens {
            return Some(AdmissionReject::InvalidPrefix);
        }
        if spec.prompt_tokens > self.policy.max_prompt_tokens {
            return Some(AdmissionReject::PromptLimit);
        }
        if spec.max_output_tokens == 0 || spec.max_output_tokens > self.policy.max_output_tokens {
            return Some(AdmissionReject::OutputLimit);
        }
        // Virtual runtime divides by priority, so zero is undefined. The
        // allowed range is 1 through 100.
        if spec.priority == 0 || spec.priority > 100 {
            return Some(AdmissionReject::InvalidPriority);
        }
        spec.deadline_ns
            .is_some_and(|deadline| deadline <= now_ns)
            .then_some(AdmissionReject::ExpiredDeadline)
    }

    fn request_counts(&self) -> (usize, usize) {
        let active = self
            .requests
            .values()
            .filter(|state| !state.status.is_terminal())
            .count();
        let queued = self
            .requests
            .values()
            .filter(|state| state.status == RequestStatus::Queued)
            .count();
        (active, queued)
    }

    fn reservation_bytes(&self, spec: &RequestSpec) -> Result<u64, SchedulerError> {
        // Prefix tokens already occupy KV, so they are not reserved again.
        spec.prompt_tokens
            .checked_sub(spec.prefix_reused_tokens)
            .and_then(|tokens| tokens.checked_add(spec.max_output_tokens))
            .and_then(|tokens| tokens.checked_mul(self.policy.kv_bytes_per_token))
            .ok_or(SchedulerError::AccountingOverflow)
    }

    fn expire_deadlines(&mut self, now_ns: u64) {
        let ids: Vec<_> = self
            .requests
            .values()
            .filter(|state| {
                state.status == RequestStatus::Queued
                    && state
                        .spec
                        .deadline_ns
                        .is_some_and(|deadline| deadline <= now_ns)
            })
            .map(|state| state.spec.id)
            .collect();
        for id in ids {
            let state = self.requests.get_mut(&id).expect("expired request exists");
            state.status = RequestStatus::DeadlineExpired;
            self.reserved_kv_bytes -= state.reserved_kv_bytes;
            state.reserved_kv_bytes = 0;
            self.stats.expired_requests += 1;
        }
    }

    fn update_high_watermarks(&mut self) {
        let depth = self
            .requests
            .values()
            .filter(|state| state.status == RequestStatus::Queued)
            .count();
        self.stats.max_queue_depth = self
            .stats
            .max_queue_depth
            .max(u32::try_from(depth).unwrap_or(u32::MAX));
        self.stats.max_reserved_kv_bytes =
            self.stats.max_reserved_kv_bytes.max(self.reserved_kv_bytes);
    }
}

// Deadlines inside urgent_window_ns preempt virtual-runtime order.
fn compare_requests(
    left: &RequestState,
    right: &RequestState,
    now_ns: u64,
    policy: SchedulerPolicy,
) -> Ordering {
    let limit = now_ns.saturating_add(policy.urgent_window_ns);
    let left_urgent = left
        .spec
        .deadline_ns
        .is_some_and(|deadline| deadline <= limit);
    let right_urgent = right
        .spec
        .deadline_ns
        .is_some_and(|deadline| deadline <= limit);
    right_urgent.cmp(&left_urgent).then_with(|| {
        if left_urgent && right_urgent {
            left.spec
                .deadline_ns
                .cmp(&right.spec.deadline_ns)
                .then_with(|| left.virtual_runtime.cmp(&right.virtual_runtime))
                .then_with(|| left.sequence.cmp(&right.sequence))
        } else {
            left.virtual_runtime
                .cmp(&right.virtual_runtime)
                .then_with(|| right.spec.priority.cmp(&left.spec.priority))
                .then_with(|| left.sequence.cmp(&right.sequence))
        }
    })
}

/// One request and its isolated token oracle in a service trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceTraceRequest {
    pub spec: RequestSpec,
    pub isolated_tokens: Vec<u32>,
    pub cancel_at_ns: Option<u64>,
}

/// A frozen concurrent service trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceTrace {
    pub schema_version: u32,
    pub policy: SchedulerPolicy,
    pub token_duration_ns: u64,
    pub requests: Vec<ServiceTraceRequest>,
}

/// One event emitted by a service trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "kebab-case")]
pub enum ServiceEvent {
    Admission {
        at_ns: u64,
        request_id: RequestId,
        outcome: AdmissionOutcome,
    },
    Dispatch {
        at_ns: u64,
        request_id: RequestId,
        token_budget: u32,
    },
    Cancel {
        at_ns: u64,
        request_id: RequestId,
        running: bool,
    },
    Complete {
        at_ns: u64,
        request_id: RequestId,
        emitted_tokens: u32,
        status: RequestStatus,
        reserved_kv_bytes: u64,
    },
}

/// One request result from a service trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceTraceResult {
    pub request_id: RequestId,
    pub status: RequestStatus,
    pub output_tokens: Vec<u32>,
}

/// The deterministic report from a concurrent service trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceTraceReport {
    pub schema_version: u32,
    pub fixture_sha256: String,
    pub events: Vec<ServiceEvent>,
    pub results: Vec<ServiceTraceResult>,
    pub stats: SchedulerStats,
    pub oracle_verified: bool,
}

struct ServiceTraceExecution {
    events: Vec<ServiceEvent>,
    outputs: BTreeMap<RequestId, Vec<u32>>,
}

/// Runs a frozen concurrent trace against isolated token streams.
pub fn run_service_trace_json(bytes: &[u8]) -> Result<ServiceTraceReport, SchedulerError> {
    let trace = decode_service_trace(bytes)?;
    let (ordered, sources) = order_service_trace(&trace)?;
    let mut scheduler = ContinuousScheduler::new(trace.policy)?;
    let ServiceTraceExecution {
        events,
        mut outputs,
    } = execute_service_trace(&trace, &mut scheduler, &ordered, &sources)?;
    let results = service_trace_results(&sources, &mut outputs, &scheduler);
    let mut report = ServiceTraceReport {
        schema_version: 1,
        fixture_sha256: digest_hex(bytes),
        events,
        results,
        stats: scheduler.stats(),
        oracle_verified: false,
    };
    verify_service_trace(&trace, &report)?;
    report.oracle_verified = true;
    Ok(report)
}

fn decode_service_trace(bytes: &[u8]) -> Result<ServiceTrace, SchedulerError> {
    let trace: ServiceTrace =
        serde_json::from_slice(bytes).map_err(|error| SchedulerError::Json(error.to_string()))?;
    if trace.schema_version != 1 {
        return Err(SchedulerError::TraceSchema(trace.schema_version));
    }
    if trace.token_duration_ns == 0 {
        return Err(SchedulerError::InvalidTrace(
            "token duration must be nonzero".into(),
        ));
    }
    Ok(trace)
}

fn order_service_trace(
    trace: &ServiceTrace,
) -> Result<
    (
        Vec<ServiceTraceRequest>,
        BTreeMap<RequestId, ServiceTraceRequest>,
    ),
    SchedulerError,
> {
    let mut ordered = trace.requests.clone();
    ordered.sort_by_key(|request| (request.spec.arrival_ns, request.spec.id));
    let sources: BTreeMap<_, _> = ordered
        .iter()
        .map(|request| (request.spec.id, request.clone()))
        .collect();
    if sources.len() != ordered.len() {
        return Err(SchedulerError::InvalidTrace(
            "request IDs must be unique".into(),
        ));
    }
    Ok((ordered, sources))
}

fn execute_service_trace(
    trace: &ServiceTrace,
    scheduler: &mut ContinuousScheduler,
    ordered: &[ServiceTraceRequest],
    sources: &BTreeMap<RequestId, ServiceTraceRequest>,
) -> Result<ServiceTraceExecution, SchedulerError> {
    let mut next = 0_usize;
    let mut now = ordered.first().map_or(0, |request| request.spec.arrival_ns);
    let mut cancellations = BTreeSet::new();
    let mut outputs: BTreeMap<RequestId, Vec<u32>> = BTreeMap::new();
    let mut events = Vec::new();
    while next < ordered.len() || scheduler.has_runnable_requests() {
        admit_trace_arrivals(
            ordered,
            &mut next,
            now,
            scheduler,
            &mut outputs,
            &mut events,
        )?;
        apply_cancellations(sources, scheduler, &mut cancellations, &mut events, now)?;
        if let Some(dispatch) = scheduler.dispatch(now)? {
            now = complete_trace_quantum(
                trace,
                scheduler,
                sources,
                &mut cancellations,
                &mut outputs,
                &mut events,
                dispatch,
                now,
            )?;
        } else if next < ordered.len() {
            now = now.max(ordered[next].spec.arrival_ns);
        } else {
            break;
        }
    }
    Ok(ServiceTraceExecution { events, outputs })
}

fn admit_trace_arrivals(
    ordered: &[ServiceTraceRequest],
    next: &mut usize,
    now: u64,
    scheduler: &mut ContinuousScheduler,
    outputs: &mut BTreeMap<RequestId, Vec<u32>>,
    events: &mut Vec<ServiceEvent>,
) -> Result<(), SchedulerError> {
    while *next < ordered.len() && ordered[*next].spec.arrival_ns <= now {
        let spec = ordered[*next].spec.clone();
        let outcome = scheduler.admit(spec.clone(), now)?;
        events.push(ServiceEvent::Admission {
            at_ns: now,
            request_id: spec.id,
            outcome,
        });
        outputs.entry(spec.id).or_default();
        *next += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn complete_trace_quantum(
    trace: &ServiceTrace,
    scheduler: &mut ContinuousScheduler,
    sources: &BTreeMap<RequestId, ServiceTraceRequest>,
    cancellations: &mut BTreeSet<RequestId>,
    outputs: &mut BTreeMap<RequestId, Vec<u32>>,
    events: &mut Vec<ServiceEvent>,
    dispatch: Dispatch,
    now: u64,
) -> Result<u64, SchedulerError> {
    events.push(ServiceEvent::Dispatch {
        at_ns: now,
        request_id: dispatch.request_id,
        token_budget: dispatch.token_budget,
    });
    let (source, start, count, end_ns) = trace_quantum_bounds(
        trace,
        scheduler,
        sources,
        dispatch.request_id,
        dispatch.token_budget,
        now,
    )?;
    apply_cancellations(sources, scheduler, cancellations, events, end_ns)?;
    outputs
        .get_mut(&dispatch.request_id)
        .expect("output exists")
        .extend_from_slice(&source.isolated_tokens[start..start + count]);
    let completion = scheduler.complete_quantum(
        dispatch.request_id,
        u32::try_from(count).map_err(|_| SchedulerError::AccountingOverflow)?,
        start + count == source.isolated_tokens.len(),
    )?;
    events.push(ServiceEvent::Complete {
        at_ns: end_ns,
        request_id: completion.request_id,
        emitted_tokens: completion.emitted_tokens,
        status: completion.status,
        reserved_kv_bytes: scheduler.reserved_kv_bytes(),
    });
    Ok(end_ns)
}

fn trace_quantum_bounds<'a>(
    trace: &ServiceTrace,
    scheduler: &ContinuousScheduler,
    sources: &'a BTreeMap<RequestId, ServiceTraceRequest>,
    request_id: RequestId,
    token_budget: u32,
    now: u64,
) -> Result<(&'a ServiceTraceRequest, usize, usize, u64), SchedulerError> {
    let source = sources
        .get(&request_id)
        .ok_or(SchedulerError::UnknownRequest(request_id))?;
    let start = trace_emitted_start(scheduler, request_id)?;
    let count = trace_quantum_count(token_budget, source.isolated_tokens.len(), start)?;
    let duration = u64::try_from(count)
        .map_err(|_| SchedulerError::AccountingOverflow)?
        .checked_mul(trace.token_duration_ns)
        .ok_or(SchedulerError::AccountingOverflow)?;
    let end_ns = now
        .checked_add(duration)
        .ok_or(SchedulerError::AccountingOverflow)?;
    Ok((source, start, count, end_ns))
}

fn trace_emitted_start(
    scheduler: &ContinuousScheduler,
    request_id: RequestId,
) -> Result<usize, SchedulerError> {
    usize::try_from(
        scheduler
            .emitted_tokens(request_id)
            .ok_or(SchedulerError::UnknownRequest(request_id))?,
    )
    .map_err(|_| SchedulerError::AccountingOverflow)
}

fn trace_quantum_count(
    token_budget: u32,
    isolated_tokens: usize,
    start: usize,
) -> Result<usize, SchedulerError> {
    Ok(usize::try_from(token_budget)
        .map_err(|_| SchedulerError::AccountingOverflow)?
        .min(isolated_tokens.saturating_sub(start)))
}

fn service_trace_results(
    sources: &BTreeMap<RequestId, ServiceTraceRequest>,
    outputs: &mut BTreeMap<RequestId, Vec<u32>>,
    scheduler: &ContinuousScheduler,
) -> Vec<ServiceTraceResult> {
    sources
        .keys()
        .map(|id| ServiceTraceResult {
            request_id: *id,
            status: scheduler.status(*id).unwrap_or(RequestStatus::Rejected),
            output_tokens: outputs.remove(id).unwrap_or_default(),
        })
        .collect()
}

fn apply_cancellations(
    sources: &BTreeMap<RequestId, ServiceTraceRequest>,
    scheduler: &mut ContinuousScheduler,
    done: &mut BTreeSet<RequestId>,
    events: &mut Vec<ServiceEvent>,
    through_ns: u64,
) -> Result<(), SchedulerError> {
    for source in sources.values() {
        let Some(at_ns) = source.cancel_at_ns else {
            continue;
        };
        if at_ns > through_ns || done.contains(&source.spec.id) {
            continue;
        }
        let Some(status) = scheduler.status(source.spec.id) else {
            continue;
        };
        if status.is_terminal() {
            done.insert(source.spec.id);
            continue;
        }
        let running = status == RequestStatus::Running;
        scheduler.cancel(source.spec.id)?;
        done.insert(source.spec.id);
        events.push(ServiceEvent::Cancel {
            at_ns,
            request_id: source.spec.id,
            running,
        });
    }
    Ok(())
}

/// Independently checks capacity, quantum, cancellation, and transcript events.
pub fn verify_service_trace(
    trace: &ServiceTrace,
    report: &ServiceTraceReport,
) -> Result<(), SchedulerError> {
    let sources: BTreeMap<_, _> = trace
        .requests
        .iter()
        .map(|request| (request.spec.id, request))
        .collect();
    let mut replay = TraceReplay::default();
    for event in &report.events {
        replay.apply_event(event, trace.policy.max_reserved_kv_bytes)?;
    }
    if replay.running.is_some() || replay.reserved != 0 {
        return Err(SchedulerError::InvalidTrace(
            "trace ends with live state".into(),
        ));
    }
    verify_trace_results(trace, report, &sources, &replay.emitted)?;
    if report.stats.max_cancellation_overshoot_tokens > trace.policy.service_quantum_tokens {
        return Err(SchedulerError::InvalidTrace(
            "cancellation bound exceeded".into(),
        ));
    }
    Ok(())
}

fn apply_quantum_state(
    state: &mut RequestState,
    stats: &mut SchedulerStats,
    emitted_tokens: u32,
    eos: bool,
) -> Result<(), SchedulerError> {
    state.remaining_tokens -= u64::from(emitted_tokens);
    state.emitted_tokens = state
        .emitted_tokens
        .checked_add(u64::from(emitted_tokens))
        .ok_or(SchedulerError::AccountingOverflow)?;
    state.virtual_runtime = state
        .virtual_runtime
        .checked_add(
            u128::from(emitted_tokens)
                .saturating_mul(1_024)
                .div_ceil(u128::from(state.spec.priority)),
        )
        .ok_or(SchedulerError::AccountingOverflow)?;
    stats.emitted_tokens = stats
        .emitted_tokens
        .checked_add(u64::from(emitted_tokens))
        .ok_or(SchedulerError::AccountingOverflow)?;
    if state.cancellation_requested {
        state.status = RequestStatus::Cancelled;
        stats.cancelled_requests += 1;
        // Cancellation applies at the quantum boundary, so overshoot stays within this quantum.
        stats.max_cancellation_overshoot_tokens =
            stats.max_cancellation_overshoot_tokens.max(emitted_tokens);
    } else if eos || state.remaining_tokens == 0 {
        state.status = RequestStatus::Finished;
        stats.completed_requests += 1;
    } else {
        state.status = RequestStatus::Queued;
    }
    Ok(())
}

#[derive(Default)]
struct TraceReplay {
    reservations: BTreeMap<RequestId, u64>,
    reserved: u64,
    running: Option<(RequestId, u32, bool)>,
    emitted: BTreeMap<RequestId, u64>,
}

impl TraceReplay {
    fn apply_event(&mut self, event: &ServiceEvent, capacity: u64) -> Result<(), SchedulerError> {
        self.apply_trace_event(event, capacity)?;
        if self.reserved > capacity {
            return Err(SchedulerError::InvalidTrace("KV capacity exceeded".into()));
        }
        Ok(())
    }

    fn apply_trace_event(
        &mut self,
        event: &ServiceEvent,
        capacity: u64,
    ) -> Result<(), SchedulerError> {
        match *event {
            ServiceEvent::Admission {
                request_id,
                outcome: AdmissionOutcome::Admitted { reserved_kv_bytes },
                ..
            } => self.admit(request_id, reserved_kv_bytes)?,
            ServiceEvent::Admission { .. } => {}
            ServiceEvent::Dispatch {
                request_id,
                token_budget,
                ..
            } => self.dispatch(request_id, token_budget, capacity)?,
            ServiceEvent::Cancel {
                request_id,
                running,
                ..
            } => self.cancel(request_id, running)?,
            ServiceEvent::Complete {
                request_id,
                emitted_tokens,
                status,
                reserved_kv_bytes,
                ..
            } => self.complete(request_id, emitted_tokens, status, reserved_kv_bytes)?,
        }
        Ok(())
    }

    fn admit(&mut self, request_id: RequestId, bytes: u64) -> Result<(), SchedulerError> {
        if self.reservations.insert(request_id, bytes).is_some() {
            return Err(SchedulerError::InvalidTrace("duplicate admission".into()));
        }
        self.reserved = self
            .reserved
            .checked_add(bytes)
            .ok_or(SchedulerError::AccountingOverflow)?;
        Ok(())
    }

    fn dispatch(
        &mut self,
        request_id: RequestId,
        token_budget: u32,
        quantum: u64,
    ) -> Result<(), SchedulerError> {
        if self.running.is_some()
            || !self.reservations.contains_key(&request_id)
            || token_budget == 0
            || u64::from(token_budget) > quantum
        {
            return Err(SchedulerError::InvalidTrace("invalid dispatch".into()));
        }
        self.running = Some((request_id, token_budget, false));
        Ok(())
    }

    fn cancel(&mut self, request_id: RequestId, running: bool) -> Result<(), SchedulerError> {
        if running {
            let (active, budget, _) = self.running.ok_or_else(|| {
                SchedulerError::InvalidTrace("running cancellation without dispatch".into())
            })?;
            if active != request_id {
                return Err(SchedulerError::InvalidTrace(
                    "cancellation names another request".into(),
                ));
            }
            self.running = Some((active, budget, true));
        } else if let Some(bytes) = self.reservations.remove(&request_id) {
            self.reserved -= bytes;
        }
        Ok(())
    }

    fn complete(
        &mut self,
        request_id: RequestId,
        emitted_tokens: u32,
        status: RequestStatus,
        reserved_kv_bytes: u64,
    ) -> Result<(), SchedulerError> {
        let (active, budget, cancelled) = self
            .running
            .take()
            .ok_or_else(|| SchedulerError::InvalidTrace("completion without dispatch".into()))?;
        if active != request_id
            || emitted_tokens > budget
            || (cancelled && status != RequestStatus::Cancelled)
        {
            return Err(SchedulerError::InvalidTrace(
                "completion violates dispatch".into(),
            ));
        }
        *self.emitted.entry(request_id).or_default() += u64::from(emitted_tokens);
        if status.is_terminal() {
            if let Some(bytes) = self.reservations.remove(&request_id) {
                self.reserved -= bytes;
            }
        }
        if self.reserved != reserved_kv_bytes {
            return Err(SchedulerError::InvalidTrace(
                "KV event accounting differs".into(),
            ));
        }
        Ok(())
    }
}

fn verify_trace_results(
    _trace: &ServiceTrace,
    report: &ServiceTraceReport,
    sources: &BTreeMap<RequestId, &ServiceTraceRequest>,
    emitted: &BTreeMap<RequestId, u64>,
) -> Result<(), SchedulerError> {
    for result in &report.results {
        verify_trace_result(result, sources, emitted)?;
    }
    Ok(())
}

fn verify_trace_result(
    result: &ServiceTraceResult,
    sources: &BTreeMap<RequestId, &ServiceTraceRequest>,
    emitted: &BTreeMap<RequestId, u64>,
) -> Result<(), SchedulerError> {
    let source = sources
        .get(&result.request_id)
        .ok_or_else(|| SchedulerError::InvalidTrace("unknown result".into()))?;
    if !source.isolated_tokens.starts_with(&result.output_tokens) {
        return Err(SchedulerError::InvalidTrace(
            "transcript differs from isolated oracle".into(),
        ));
    }
    if result.status == RequestStatus::Finished && result.output_tokens != source.isolated_tokens {
        return Err(SchedulerError::InvalidTrace(
            "finished transcript is partial".into(),
        ));
    }
    if result.status != RequestStatus::Rejected
        && Some(
            u64::try_from(result.output_tokens.len())
                .map_err(|_| SchedulerError::AccountingOverflow)?,
        ) != emitted.get(&result.request_id).copied()
    {
        return Err(SchedulerError::InvalidTrace(
            "token count differs from event replay".into(),
        ));
    }
    Ok(())
}

fn digest_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_trace_matches_isolated_transcripts() {
        let report =
            run_service_trace_json(include_bytes!("../../../fixtures/service-trace.json")).unwrap();
        assert!(report.oracle_verified);
        assert!(report.stats.admitted_requests >= 3);
        assert!(report.stats.rejected_requests >= 1);
        assert!(report.stats.cancelled_requests >= 1);
    }

    #[test]
    fn prefix_reuse_reduces_the_exact_kv_charge() {
        let mut scheduler = ContinuousScheduler::new(policy()).unwrap();
        let outcome = scheduler
            .admit(
                RequestSpec {
                    id: RequestId(1),
                    arrival_ns: 0,
                    prompt_tokens: 100,
                    prefix_reused_tokens: 80,
                    max_output_tokens: 10,
                    priority: 1,
                    deadline_ns: None,
                },
                0,
            )
            .unwrap();
        assert_eq!(
            outcome,
            AdmissionOutcome::Admitted {
                reserved_kv_bytes: 30 * 64
            }
        );
    }

    #[test]
    fn running_cancellation_is_bounded_by_one_quantum() {
        let mut scheduler = ContinuousScheduler::new(policy()).unwrap();
        scheduler.admit(spec(1), 0).unwrap();
        let dispatch = scheduler.dispatch(0).unwrap().unwrap();
        scheduler.cancel(RequestId(1)).unwrap();
        let completion = scheduler
            .complete_quantum(RequestId(1), dispatch.token_budget, false)
            .unwrap();
        assert_eq!(completion.status, RequestStatus::Cancelled);
        assert_eq!(
            scheduler.stats().max_cancellation_overshoot_tokens,
            policy().service_quantum_tokens
        );
        assert_eq!(scheduler.reserved_kv_bytes(), 0);
    }

    #[test]
    fn aborted_quantum_releases_the_scheduler_transaction() {
        let mut scheduler = ContinuousScheduler::new(policy()).unwrap();
        scheduler.admit(spec(1), 0).unwrap();
        scheduler.dispatch(0).unwrap().unwrap();
        assert_eq!(
            scheduler.abort_quantum(RequestId(1)).unwrap(),
            RequestStatus::Cancelled
        );
        assert_eq!(scheduler.reserved_kv_bytes(), 0);
        assert!(!scheduler.has_runnable_requests());
        assert_eq!(scheduler.dispatch(1).unwrap(), None);
    }

    #[test]
    fn overload_is_typed_and_does_not_reserve_bytes() {
        let mut selected = policy();
        selected.max_active_requests = 1;
        let mut scheduler = ContinuousScheduler::new(selected).unwrap();
        assert!(matches!(
            scheduler.admit(spec(1), 0).unwrap(),
            AdmissionOutcome::Admitted { .. }
        ));
        let reserved = scheduler.reserved_kv_bytes();
        assert_eq!(
            scheduler.admit(spec(2), 0).unwrap(),
            AdmissionOutcome::Rejected {
                reason: AdmissionReject::ActiveLimit
            }
        );
        assert_eq!(scheduler.reserved_kv_bytes(), reserved);
    }

    #[test]
    fn event_oracle_rejects_a_mutated_transcript() {
        let bytes = include_bytes!("../../../fixtures/service-trace.json");
        let trace: ServiceTrace = serde_json::from_slice(bytes).unwrap();
        let mut report = run_service_trace_json(bytes).unwrap();
        let result = report
            .results
            .iter_mut()
            .find(|result| result.status == RequestStatus::Finished)
            .unwrap();
        result.output_tokens[0] ^= 1;
        assert!(verify_service_trace(&trace, &report).is_err());
    }

    fn policy() -> SchedulerPolicy {
        SchedulerPolicy {
            max_active_requests: 8,
            max_queued_requests: 8,
            max_reserved_kv_bytes: 1 << 20,
            kv_bytes_per_token: 64,
            max_prompt_tokens: 1_024,
            max_output_tokens: 128,
            service_quantum_tokens: 4,
            urgent_window_ns: 10_000,
            max_prefix_credit_tokens: 256,
        }
    }
    fn spec(id: u64) -> RequestSpec {
        RequestSpec {
            id: RequestId(id),
            arrival_ns: 0,
            prompt_tokens: 32,
            prefix_reused_tokens: 0,
            max_output_tokens: 20,
            priority: 1,
            deadline_ns: None,
        }
    }
}
