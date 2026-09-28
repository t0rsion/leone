//! Collects bounded, backend-neutral service measurements.
//!
//! Callers supply timestamps from one process-monotonic clock domain. The
//! snapshot names that domain so a receipt cannot mix client and engine clocks.

use std::collections::{BTreeMap, VecDeque};

use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Identifies the schema written by [`ServiceMetrics::snapshot`].
pub const SERVICE_METRICS_SCHEMA_VERSION: u32 = 1;
const DEFAULT_HISTORY_CAPACITY: usize = 1024;
const DEFAULT_LATENCY_CAPACITY: usize = 4096;
const DEFAULT_OUTCOME_KINDS: usize = 64;
const DEFAULT_IDENTIFIER_BYTES: usize = 256;
const DEFAULT_REASON_BYTES: usize = 128;
const DEFAULT_MEMORY_CLASSES: usize = 32;
const MAX_ACTIVE_REQUESTS: usize = 65_536;
const MAX_OUTCOME_KINDS: usize = 4_096;
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_MEMORY_CLASSES: usize = 256;
const MAX_TOTAL_HISTORY_ENTRIES: usize = 1_000_000;
const QUANTILE_DEFINITION: &str =
    "linear interpolation between adjacent sorted observations at h=(n-1)p";
const CLOCK_DOMAIN: &str = "caller_monotonic_ns";
const SCHEDULER_RESERVATION_DEFINITION: &str =
    "reserved_bytes is the total scheduler reservation at each sample";

/// States whether a measurement contains an observed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementStatus {
    /// The value was observed by the service.
    Observed,
    /// The service could not observe the value.
    Unavailable,
    /// The value does not apply to this record.
    NotApplicable,
}

/// Stores a measurement without turning an absent value into zero.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement<T> {
    value: Option<T>,
    status: MeasurementStatus,
    reason: Option<String>,
}

impl<T> Measurement<T> {
    /// Creates an observed measurement.
    pub fn observed(value: T) -> Self {
        Self {
            value: Some(value),
            status: MeasurementStatus::Observed,
            reason: None,
        }
    }

    /// Creates a missing measurement with an explicit reason.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            value: None,
            status: MeasurementStatus::Unavailable,
            reason: Some(bound_reason(reason.into())),
        }
    }

    /// Creates a measurement that does not apply to the record.
    pub fn not_applicable(reason: impl Into<String>) -> Self {
        Self {
            value: None,
            status: MeasurementStatus::NotApplicable,
            reason: Some(bound_reason(reason.into())),
        }
    }

    /// Returns the measurement state.
    pub fn status(&self) -> MeasurementStatus {
        self.status
    }

    /// Returns the value when the measurement was observed.
    pub fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Returns the explicit absence reason.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

impl<'de, T> Deserialize<'de> for Measurement<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire<T> {
            value: Option<T>,
            status: MeasurementStatus,
            reason: Option<String>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let valid = match wire.status {
            MeasurementStatus::Observed => wire.value.is_some() && wire.reason.is_none(),
            MeasurementStatus::Unavailable | MeasurementStatus::NotApplicable => {
                wire.value.is_none()
                    && wire.reason.as_ref().is_some_and(|reason| {
                        !reason.is_empty() && reason.len() <= DEFAULT_REASON_BYTES
                    })
            }
        };
        if !valid {
            return Err(serde::de::Error::custom(
                "measurement value, status, and reason disagree",
            ));
        }
        Ok(Self {
            value: wire.value,
            status: wire.status,
            reason: wire.reason,
        })
    }
}

/// Reports an invalid bounded collector operation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MetricsError {
    #[error("{field} must be nonzero")]
    ZeroCapacity { field: &'static str },
    #[error("request {0} is already active")]
    DuplicateRequest(String),
    #[error("request {0} is not active")]
    UnknownRequest(String),
    #[error("request {request_id} timestamp {at_ns} precedes {previous_ns}")]
    TimestampReversed {
        request_id: String,
        at_ns: u64,
        previous_ns: u64,
    },
    #[error("active request capacity is full")]
    RequestCapacity,
    #[error("{field} is longer than {max_bytes} bytes")]
    FieldTooLong {
        field: &'static str,
        max_bytes: usize,
    },
    #[error("{field} has too many entries")]
    EntryCapacity { field: &'static str },
    #[error("{field} timestamp {at_ns} precedes {previous_ns}")]
    ObservationTimestampReversed {
        field: &'static str,
        at_ns: u64,
        previous_ns: u64,
    },
    #[error("{field} could not reserve bounded history capacity")]
    AllocationFailed { field: &'static str },
    #[error("metrics counter overflowed")]
    CounterOverflow,
    #[error("{field} exceeds the maximum bound of {max}")]
    CapacityTooLarge { field: &'static str, max: usize },
    #[error("token batch count must be nonzero")]
    EmptyTokenBatch,
}

/// Sets the bounds for retained request and sample history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    pub max_in_flight_requests: usize,
    pub max_request_history: usize,
    pub max_latency_samples: usize,
    pub max_observation_history: usize,
    pub max_outcome_kinds: usize,
    pub max_identifier_bytes: usize,
    pub max_reason_bytes: usize,
    pub max_memory_classes: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            max_in_flight_requests: DEFAULT_HISTORY_CAPACITY,
            max_request_history: DEFAULT_HISTORY_CAPACITY,
            max_latency_samples: DEFAULT_LATENCY_CAPACITY,
            max_observation_history: DEFAULT_HISTORY_CAPACITY,
            max_outcome_kinds: DEFAULT_OUTCOME_KINDS,
            max_identifier_bytes: DEFAULT_IDENTIFIER_BYTES,
            max_reason_bytes: DEFAULT_REASON_BYTES,
            max_memory_classes: DEFAULT_MEMORY_CLASSES,
        }
    }
}

impl MetricsConfig {
    fn validate(self) -> Result<Self, MetricsError> {
        let fields = [
            (self.max_in_flight_requests, "max_in_flight_requests"),
            (self.max_request_history, "max_request_history"),
            (self.max_latency_samples, "max_latency_samples"),
            (self.max_observation_history, "max_observation_history"),
            (self.max_outcome_kinds, "max_outcome_kinds"),
            (self.max_identifier_bytes, "max_identifier_bytes"),
            (self.max_reason_bytes, "max_reason_bytes"),
            (self.max_memory_classes, "max_memory_classes"),
        ];
        if let Some((_, field)) = fields.into_iter().find(|(value, _)| *value == 0) {
            return Err(MetricsError::ZeroCapacity { field });
        }
        validate_capacity(
            self.max_in_flight_requests,
            MAX_ACTIVE_REQUESTS,
            "max_in_flight_requests",
        )?;
        validate_capacity(
            self.max_outcome_kinds,
            MAX_OUTCOME_KINDS,
            "max_outcome_kinds",
        )?;
        validate_capacity(
            self.max_identifier_bytes,
            MAX_TEXT_BYTES,
            "max_identifier_bytes",
        )?;
        validate_capacity(self.max_reason_bytes, MAX_TEXT_BYTES, "max_reason_bytes")?;
        validate_capacity(
            self.max_memory_classes,
            MAX_MEMORY_CLASSES,
            "max_memory_classes",
        )?;
        validate_total_history(self)?;
        Ok(self)
    }
}

fn validate_capacity(value: usize, max: usize, field: &'static str) -> Result<(), MetricsError> {
    if value > max {
        return Err(MetricsError::CapacityTooLarge { field, max });
    }
    Ok(())
}

fn validate_total_history(config: MetricsConfig) -> Result<(), MetricsError> {
    let total = config
        .max_request_history
        .checked_add(config.max_latency_samples.saturating_mul(3))
        .and_then(|value| value.checked_add(config.max_observation_history.saturating_mul(6)))
        .ok_or(MetricsError::AllocationFailed {
            field: "total history capacity",
        })?;
    if total > MAX_TOTAL_HISTORY_ENTRIES {
        return Err(MetricsError::AllocationFailed {
            field: "total history capacity",
        });
    }
    Ok(())
}

/// Retains the newest values while exposing every eviction in the snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedHistory<T> {
    capacity: usize,
    values: VecDeque<T>,
    dropped_count: u64,
    total_count: u64,
}

impl<T> BoundedHistory<T> {
    /// Creates an empty history with a fixed positive capacity.
    pub fn new(capacity: usize) -> Result<Self, MetricsError> {
        if capacity == 0 {
            return Err(MetricsError::ZeroCapacity {
                field: "history capacity",
            });
        }
        let mut values = VecDeque::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| MetricsError::AllocationFailed {
                field: "history capacity",
            })?;
        Ok(Self {
            capacity,
            values,
            dropped_count: 0,
            total_count: 0,
        })
    }

    /// Appends a value and reports whether the oldest value was evicted.
    pub fn push(&mut self, value: T) -> Result<bool, MetricsError> {
        self.can_push()?;
        Ok(self.push_validated(value))
    }

    fn push_validated(&mut self, value: T) -> bool {
        let evicted = self.values.len() == self.capacity;
        if evicted {
            self.values.pop_front();
        }
        self.values.push_back(value);
        self.dropped_count += u64::from(evicted);
        self.total_count += 1;
        evicted
    }

    fn can_push(&self) -> Result<(), MetricsError> {
        self.total_count
            .checked_add(1)
            .ok_or(MetricsError::CounterOverflow)?;
        if self.values.len() == self.capacity {
            self.dropped_count
                .checked_add(1)
                .ok_or(MetricsError::CounterOverflow)?;
        }
        Ok(())
    }

    /// Returns the fixed retention capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns retained values in timestamp or insertion order.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    /// Returns the number of retained values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether no values remain retained.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the number of values evicted since creation.
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }

    /// Returns the count of all accepted values, including evicted values.
    pub fn sample_count(&self) -> u64 {
        self.total_count
    }

    pub fn is_truncated(&self) -> bool {
        self.dropped_count != 0
    }
}

impl<T: Serialize> Serialize for BoundedHistory<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("BoundedHistory", 4)?;
        state.serialize_field(
            "capacity",
            &u64::try_from(self.capacity).map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field("values", &self.values)?;
        state.serialize_field("dropped_count", &self.dropped_count)?;
        state.serialize_field("sample_count", &self.total_count)?;
        state.end()
    }
}

/// Returns the linear-interpolation quantile for retained observations.
fn linear_quantile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    if ordered.len() == 1 {
        return ordered.first().copied();
    }
    let position = (ordered.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(ordered.len() - 1);
    let fraction = position - lower as f64;
    Some(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)
}

/// Summarizes one bounded latency series.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatencySummary {
    pub sample_count: u64,
    pub retained_sample_count: u64,
    pub dropped_sample_count: u64,
    pub p95_ns: Measurement<f64>,
    pub samples: BoundedHistory<u64>,
}

impl LatencySummary {
    fn from_history(history: &BoundedHistory<u64>) -> Self {
        let p95_ns = if history.values.is_empty() {
            Measurement::unavailable("no_samples")
        } else if history.is_truncated() {
            Measurement::unavailable("latency_history_truncated")
        } else {
            let values: Vec<_> = history.values.iter().map(|value| *value as f64).collect();
            match linear_quantile(&values, 0.95) {
                Some(value) => Measurement::observed(value),
                None => Measurement::unavailable("quantile_unavailable"),
            }
        };
        Self {
            sample_count: history.total_count,
            retained_sample_count: history.values.len() as u64,
            dropped_sample_count: history.dropped_count,
            p95_ns,
            samples: history.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct ActiveRequest {
    started_ns: u64,
    first_token_ns: Option<u64>,
    last_token_ns: Option<u64>,
    token_count: u64,
}

/// Stores one completed request outcome and its request-level timings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestMetric {
    pub request_id: String,
    pub outcome: String,
    pub token_count: u64,
    pub ttft_ns: Measurement<u64>,
    pub completion_latency_ns: Measurement<u64>,
}

/// Stores one physical allocation snapshot by backend-neutral class name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalBytesSample {
    pub at_ns: u64,
    pub bytes_by_class: BTreeMap<String, u64>,
}

/// Stores one scheduler reservation snapshot separately from physical bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerReservationSample {
    pub at_ns: u64,
    pub request_id: Option<String>,
    pub reserved_bytes: u64,
}

/// Stores process memory fields with explicit unavailable values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessMemorySample {
    pub at_ns: u64,
    pub resident_bytes: Measurement<u64>,
    pub virtual_bytes: Measurement<u64>,
}

/// Stores system memory fields with explicit unavailable values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemMemorySample {
    pub at_ns: u64,
    pub used_bytes: Measurement<u64>,
    pub available_bytes: Measurement<u64>,
}

/// Stores cancellation and slow-client observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestControlSample {
    pub request_id: String,
    pub latency_ns: Measurement<u64>,
    pub reason: String,
}

/// A bounded service metrics snapshot suitable for a structured receipt.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ServiceMetricsSnapshot {
    pub schema_version: u32,
    pub request_count: u64,
    pub active_request_count: u64,
    pub active_request_capacity: u64,
    pub outcomes: BTreeMap<String, u64>,
    pub requests: BoundedHistory<RequestMetric>,
    pub ttft: LatencySummary,
    pub inter_token_latency: LatencySummary,
    pub fork_latency: LatencySummary,
    pub cancellation: BoundedHistory<RequestControlSample>,
    pub slow_client: BoundedHistory<RequestControlSample>,
    pub physical_bytes: BoundedHistory<PhysicalBytesSample>,
    pub scheduler_reservations: BoundedHistory<SchedulerReservationSample>,
    pub process_memory: BoundedHistory<ProcessMemorySample>,
    pub system_memory: BoundedHistory<SystemMemorySample>,
    pub physical_bytes_peak_by_class: BTreeMap<String, u64>,
    pub scheduler_reservation_peak_bytes: Measurement<u64>,
    /// The sample with the most resident bytes, or most virtual bytes if no resident value was observed.
    pub process_memory_peak: ProcessMemorySample,
    /// The sample with the most used bytes, or least available bytes if no used value was observed.
    pub system_memory_peak: SystemMemorySample,
    pub quantile_definition: &'static str,
    pub clock_domain: &'static str,
    pub scheduler_reservation_definition: &'static str,
}

/// Collects bounded service metrics without depending on a transport backend.
#[derive(Debug)]
pub struct ServiceMetrics {
    max_in_flight_requests: usize,
    max_outcome_kinds: usize,
    max_identifier_bytes: usize,
    max_reason_bytes: usize,
    max_memory_classes: usize,
    active: BTreeMap<String, ActiveRequest>,
    request_count: u64,
    outcomes: BTreeMap<String, u64>,
    requests: BoundedHistory<RequestMetric>,
    ttft: BoundedHistory<u64>,
    inter_token_latency: BoundedHistory<u64>,
    fork_latency: BoundedHistory<u64>,
    cancellation: BoundedHistory<RequestControlSample>,
    slow_client: BoundedHistory<RequestControlSample>,
    physical_bytes: BoundedHistory<PhysicalBytesSample>,
    scheduler_reservations: BoundedHistory<SchedulerReservationSample>,
    process_memory: BoundedHistory<ProcessMemorySample>,
    system_memory: BoundedHistory<SystemMemorySample>,
    last_physical_bytes_ns: Option<u64>,
    last_scheduler_reservation_ns: Option<u64>,
    last_process_memory_ns: Option<u64>,
    last_system_memory_ns: Option<u64>,
    physical_bytes_peak_by_class: BTreeMap<String, u64>,
    scheduler_reservation_peak_bytes: Option<u64>,
    process_memory_peak: ProcessMemorySample,
    system_memory_peak: SystemMemorySample,
}

type MetricsHistories = (
    BoundedHistory<RequestMetric>,
    BoundedHistory<u64>,
    BoundedHistory<u64>,
    BoundedHistory<u64>,
    BoundedHistory<RequestControlSample>,
    BoundedHistory<RequestControlSample>,
    BoundedHistory<PhysicalBytesSample>,
    BoundedHistory<SchedulerReservationSample>,
    BoundedHistory<ProcessMemorySample>,
    BoundedHistory<SystemMemorySample>,
);
type LatencyHistories = (
    BoundedHistory<u64>,
    BoundedHistory<u64>,
    BoundedHistory<u64>,
);
type ObservationHistories = (
    BoundedHistory<RequestControlSample>,
    BoundedHistory<RequestControlSample>,
    BoundedHistory<PhysicalBytesSample>,
    BoundedHistory<SchedulerReservationSample>,
    BoundedHistory<ProcessMemorySample>,
    BoundedHistory<SystemMemorySample>,
);

impl ServiceMetrics {
    /// Creates a collector with fixed bounds for active requests and history.
    pub fn new(config: MetricsConfig) -> Result<Self, MetricsError> {
        let config = config.validate()?;
        let (
            requests,
            ttft,
            inter_token_latency,
            fork_latency,
            cancellation,
            slow_client,
            physical_bytes,
            scheduler_reservations,
            process_memory,
            system_memory,
        ) = histories(config)?;
        Ok(Self {
            max_in_flight_requests: config.max_in_flight_requests,
            max_outcome_kinds: config.max_outcome_kinds,
            max_identifier_bytes: config.max_identifier_bytes,
            max_reason_bytes: config.max_reason_bytes,
            max_memory_classes: config.max_memory_classes,
            active: BTreeMap::new(),
            request_count: 0,
            outcomes: BTreeMap::new(),
            requests,
            ttft,
            inter_token_latency,
            fork_latency,
            cancellation,
            slow_client,
            physical_bytes,
            scheduler_reservations,
            process_memory,
            system_memory,
            last_physical_bytes_ns: None,
            last_scheduler_reservation_ns: None,
            last_process_memory_ns: None,
            last_system_memory_ns: None,
            physical_bytes_peak_by_class: BTreeMap::new(),
            scheduler_reservation_peak_bytes: None,
            process_memory_peak: empty_process_memory(),
            system_memory_peak: empty_system_memory(),
        })
    }

    /// Starts timing for one request while enforcing the active bound.
    pub fn request_started(
        &mut self,
        request_id: impl Into<String>,
        at_ns: u64,
    ) -> Result<(), MetricsError> {
        let request_id = request_id.into();
        validate_text(&request_id, self.max_identifier_bytes, "request_id")?;
        if self.active.contains_key(&request_id) {
            return Err(MetricsError::DuplicateRequest(request_id));
        }
        if self.active.len() >= self.max_in_flight_requests {
            return Err(MetricsError::RequestCapacity);
        }
        self.active.insert(
            request_id,
            ActiveRequest {
                started_ns: at_ns,
                first_token_ns: None,
                last_token_ns: None,
                token_count: 0,
            },
        );
        Ok(())
    }

    /// Records an outcome that has no request timing, such as admission rejection.
    pub fn outcome_observed(&mut self, outcome: impl Into<String>) -> Result<(), MetricsError> {
        let outcome = outcome.into();
        self.preflight_outcome(&outcome)?;
        let next_request_count = self
            .request_count
            .checked_add(1)
            .ok_or(MetricsError::CounterOverflow)?;
        let count = self.outcomes.entry(outcome).or_default();
        *count = count.checked_add(1).ok_or(MetricsError::CounterOverflow)?;
        self.request_count = next_request_count;
        Ok(())
    }

    /// Returns the number of requests with active timing state.
    pub fn active_request_count(&self) -> usize {
        self.active.len()
    }

    /// Records one observed token boundary and updates TTFT and ITL samples.
    pub fn token_observed(&mut self, request_id: &str, at_ns: u64) -> Result<(), MetricsError> {
        validate_text(request_id, self.max_identifier_bytes, "request_id")?;
        let state = active_request(&self.active, request_id)?;
        let (was_first, interval) = token_delta(request_id, state, at_ns)?;
        record_latency_sample(
            was_first,
            interval,
            &mut self.ttft,
            &mut self.inter_token_latency,
        )?;
        let state = self
            .active
            .get_mut(request_id)
            .ok_or_else(|| MetricsError::UnknownRequest(request_id.to_owned()))?;
        commit_token_delta(state, at_ns);
        Ok(())
    }

    /// Records a token batch without inventing boundaries between its tokens.
    pub fn token_batch_observed(
        &mut self,
        request_id: &str,
        at_ns: u64,
        token_count: u64,
    ) -> Result<(), MetricsError> {
        if token_count == 0 {
            return Err(MetricsError::EmptyTokenBatch);
        }
        validate_text(request_id, self.max_identifier_bytes, "request_id")?;
        let state = active_request(&self.active, request_id)?;
        let (first, ttft_ns, next_count) =
            token_batch_delta(request_id, state, at_ns, token_count)?;
        if first {
            self.ttft.can_push()?;
        }
        let state = self
            .active
            .get_mut(request_id)
            .ok_or_else(|| MetricsError::UnknownRequest(request_id.to_owned()))?;
        commit_token_batch(state, at_ns, next_count, first);
        if first {
            self.ttft.push_validated(ttft_ns);
        }
        Ok(())
    }

    /// Completes one request and records its outcome and request-level metrics.
    pub fn request_finished(
        &mut self,
        request_id: &str,
        completed_at_ns: Option<u64>,
        outcome: impl Into<String>,
    ) -> Result<(), MetricsError> {
        validate_text(request_id, self.max_identifier_bytes, "request_id")?;
        let state = active_request(&self.active, request_id)?;
        let state_copy = state.clone();
        let outcome = outcome.into();
        self.preflight_outcome(&outcome)?;
        let completion_latency_ns = completion_latency(request_id, state, completed_at_ns)?;
        self.requests.can_push()?;
        let ttft_ns = request_ttft(&state_copy);
        self.commit_outcome(outcome.clone());
        self.requests.push_validated(RequestMetric {
            request_id: request_id.to_owned(),
            outcome,
            token_count: state_copy.token_count,
            ttft_ns,
            completion_latency_ns,
        });
        self.active.remove(request_id);
        Ok(())
    }

    /// Records one measured device or host session fork latency.
    pub fn fork_observed(&mut self, latency_ns: u64) -> Result<(), MetricsError> {
        self.fork_latency.can_push()?;
        self.fork_latency.push_validated(latency_ns);
        Ok(())
    }

    /// Records cancellation latency. Missing completion timing stays explicit.
    pub fn cancellation_observed(
        &mut self,
        request_id: impl Into<String>,
        requested_at_ns: u64,
        completed_at_ns: Option<u64>,
        reason: impl Into<String>,
    ) -> Result<(), MetricsError> {
        let request_id = request_id.into();
        let reason = reason.into();
        validate_text(&request_id, self.max_identifier_bytes, "request_id")?;
        validate_text(&reason, self.max_reason_bytes, "reason")?;
        let latency_ns = match completed_at_ns {
            Some(at_ns) if at_ns >= requested_at_ns => {
                Measurement::observed(at_ns - requested_at_ns)
            }
            Some(at_ns) => {
                return Err(MetricsError::TimestampReversed {
                    request_id,
                    at_ns,
                    previous_ns: requested_at_ns,
                });
            }
            None => Measurement::unavailable("cancellation_completion_unavailable"),
        };
        self.cancellation.can_push()?;
        self.cancellation.push_validated(RequestControlSample {
            request_id,
            latency_ns,
            reason,
        });
        Ok(())
    }

    /// Records cancellation when the request timestamp is unavailable.
    pub fn cancellation_unavailable(
        &mut self,
        request_id: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), MetricsError> {
        let request_id = request_id.into();
        let reason = reason.into();
        validate_text(&request_id, self.max_identifier_bytes, "request_id")?;
        validate_text(&reason, self.max_reason_bytes, "reason")?;
        self.cancellation.can_push()?;
        self.cancellation.push_validated(RequestControlSample {
            request_id,
            latency_ns: Measurement::unavailable("cancellation_request_time_unavailable"),
            reason,
        });
        Ok(())
    }

    /// Records a slow-client observation. Missing drain timing stays explicit.
    pub fn slow_client_observed(
        &mut self,
        request_id: impl Into<String>,
        blocked_ns: Option<u64>,
        reason: impl Into<String>,
    ) -> Result<(), MetricsError> {
        let request_id = request_id.into();
        let reason = reason.into();
        validate_text(&request_id, self.max_identifier_bytes, "request_id")?;
        validate_text(&reason, self.max_reason_bytes, "reason")?;
        self.slow_client.can_push()?;
        self.slow_client.push_validated(RequestControlSample {
            request_id,
            latency_ns: blocked_ns
                .map(Measurement::observed)
                .unwrap_or_else(|| Measurement::unavailable("slow_client_timing_unavailable")),
            reason,
        });
        Ok(())
    }

    /// Records backend-neutral physical bytes by allocation class.
    pub fn physical_bytes_observed(
        &mut self,
        at_ns: u64,
        bytes_by_class: BTreeMap<String, u64>,
    ) -> Result<(), MetricsError> {
        self.validate_observation_timestamp("physical_bytes", at_ns, self.last_physical_bytes_ns)?;
        self.validate_memory_classes(&bytes_by_class)?;
        self.physical_bytes.can_push()?;
        for (class, bytes) in &bytes_by_class {
            let peak = self
                .physical_bytes_peak_by_class
                .entry(class.clone())
                .or_insert(0);
            *peak = (*peak).max(*bytes);
        }
        self.physical_bytes.push_validated(PhysicalBytesSample {
            at_ns,
            bytes_by_class,
        });
        self.last_physical_bytes_ns = Some(at_ns);
        Ok(())
    }

    /// Records scheduler reservation bytes independently of physical bytes.
    pub fn scheduler_reservation_observed(
        &mut self,
        at_ns: u64,
        request_id: Option<String>,
        reserved_bytes: u64,
    ) -> Result<(), MetricsError> {
        self.validate_observation_timestamp(
            "scheduler_reservation",
            at_ns,
            self.last_scheduler_reservation_ns,
        )?;
        if let Some(request_id) = request_id.as_deref() {
            validate_text(request_id, self.max_identifier_bytes, "request_id")?;
        }
        self.scheduler_reservations.can_push()?;
        self.scheduler_reservation_peak_bytes = Some(
            self.scheduler_reservation_peak_bytes
                .unwrap_or(0)
                .max(reserved_bytes),
        );
        self.scheduler_reservations
            .push_validated(SchedulerReservationSample {
                at_ns,
                request_id,
                reserved_bytes,
            });
        self.last_scheduler_reservation_ns = Some(at_ns);
        Ok(())
    }

    /// Records process memory. `None` means the source did not report a field.
    pub fn process_memory_observed(
        &mut self,
        at_ns: u64,
        resident_bytes: Option<u64>,
        virtual_bytes: Option<u64>,
    ) -> Result<(), MetricsError> {
        self.validate_observation_timestamp("process_memory", at_ns, self.last_process_memory_ns)?;
        self.process_memory.can_push()?;
        let sample = ProcessMemorySample {
            at_ns,
            resident_bytes: optional_measurement(resident_bytes, "process_resident_unavailable"),
            virtual_bytes: optional_measurement(virtual_bytes, "process_virtual_unavailable"),
        };
        self.process_memory_peak = peak_process_memory(&self.process_memory_peak, &sample);
        self.process_memory
            .push_validated(ProcessMemorySample { ..sample });
        self.last_process_memory_ns = Some(at_ns);
        Ok(())
    }

    /// Records system memory. `None` means the source did not report a field.
    pub fn system_memory_observed(
        &mut self,
        at_ns: u64,
        used_bytes: Option<u64>,
        available_bytes: Option<u64>,
    ) -> Result<(), MetricsError> {
        self.validate_observation_timestamp("system_memory", at_ns, self.last_system_memory_ns)?;
        self.system_memory.can_push()?;
        let sample = SystemMemorySample {
            at_ns,
            used_bytes: optional_measurement(used_bytes, "system_used_unavailable"),
            available_bytes: optional_measurement(available_bytes, "system_available_unavailable"),
        };
        self.system_memory_peak = peak_system_memory(&self.system_memory_peak, &sample);
        self.system_memory
            .push_validated(SystemMemorySample { ..sample });
        self.last_system_memory_ns = Some(at_ns);
        Ok(())
    }

    /// Builds a bounded structured snapshot and reports retained-history loss.
    pub fn snapshot(&self) -> ServiceMetricsSnapshot {
        ServiceMetricsSnapshot {
            schema_version: SERVICE_METRICS_SCHEMA_VERSION,
            request_count: self.request_count,
            active_request_count: self.active.len() as u64,
            active_request_capacity: self.max_in_flight_requests as u64,
            outcomes: self.outcomes.clone(),
            requests: self.requests.clone(),
            ttft: LatencySummary::from_history(&self.ttft),
            inter_token_latency: LatencySummary::from_history(&self.inter_token_latency),
            fork_latency: LatencySummary::from_history(&self.fork_latency),
            cancellation: self.cancellation.clone(),
            slow_client: self.slow_client.clone(),
            physical_bytes: self.physical_bytes.clone(),
            scheduler_reservations: self.scheduler_reservations.clone(),
            process_memory: self.process_memory.clone(),
            system_memory: self.system_memory.clone(),
            physical_bytes_peak_by_class: self.physical_bytes_peak_by_class.clone(),
            scheduler_reservation_peak_bytes: self
                .scheduler_reservation_peak_bytes
                .map(Measurement::observed)
                .unwrap_or_else(|| Measurement::unavailable("scheduler_reservation_unavailable")),
            process_memory_peak: self.process_memory_peak.clone(),
            system_memory_peak: self.system_memory_peak.clone(),
            quantile_definition: QUANTILE_DEFINITION,
            clock_domain: CLOCK_DOMAIN,
            scheduler_reservation_definition: SCHEDULER_RESERVATION_DEFINITION,
        }
    }

    fn preflight_outcome(&self, outcome: &str) -> Result<(), MetricsError> {
        validate_text(outcome, self.max_reason_bytes, "outcome")?;
        self.request_count
            .checked_add(1)
            .ok_or(MetricsError::CounterOverflow)?;
        match self.outcomes.get(outcome) {
            Some(count) => {
                count.checked_add(1).ok_or(MetricsError::CounterOverflow)?;
            }
            None if self.outcomes.len() >= self.max_outcome_kinds => {
                return Err(MetricsError::EntryCapacity { field: "outcomes" });
            }
            None => {}
        }
        Ok(())
    }

    fn commit_outcome(&mut self, outcome: String) {
        let count = self.outcomes.entry(outcome).or_default();
        *count += 1;
        self.request_count += 1;
    }

    fn validate_memory_classes(&self, values: &BTreeMap<String, u64>) -> Result<(), MetricsError> {
        if values.len() > self.max_memory_classes {
            return Err(MetricsError::EntryCapacity {
                field: "physical_bytes_by_class",
            });
        }
        let new_classes = values
            .keys()
            .filter(|class| !self.physical_bytes_peak_by_class.contains_key(*class))
            .count();
        if self.physical_bytes_peak_by_class.len() + new_classes > self.max_memory_classes {
            return Err(MetricsError::EntryCapacity {
                field: "physical_bytes_peak_by_class",
            });
        }
        for class in values.keys() {
            validate_text(class, self.max_reason_bytes, "memory_class")?;
        }
        Ok(())
    }

    fn validate_observation_timestamp(
        &self,
        field: &'static str,
        at_ns: u64,
        previous_ns: Option<u64>,
    ) -> Result<(), MetricsError> {
        if let Some(previous_ns) = previous_ns {
            if at_ns < previous_ns {
                return Err(MetricsError::ObservationTimestampReversed {
                    field,
                    at_ns,
                    previous_ns,
                });
            }
        }
        Ok(())
    }
}

fn optional_measurement(value: Option<u64>, reason: &'static str) -> Measurement<u64> {
    value
        .map(Measurement::observed)
        .unwrap_or_else(|| Measurement::unavailable(reason))
}

fn bound_reason(reason: String) -> String {
    if reason.len() <= DEFAULT_REASON_BYTES {
        reason
    } else {
        "reason_exceeds_bound".to_owned()
    }
}

fn validate_text(value: &str, max_bytes: usize, field: &'static str) -> Result<(), MetricsError> {
    if value.len() > max_bytes {
        return Err(MetricsError::FieldTooLong { field, max_bytes });
    }
    Ok(())
}

fn history<T>(capacity: usize) -> Result<BoundedHistory<T>, MetricsError> {
    BoundedHistory::new(capacity)
}

fn active_request<'a>(
    active: &'a BTreeMap<String, ActiveRequest>,
    request_id: &str,
) -> Result<&'a ActiveRequest, MetricsError> {
    active
        .get(request_id)
        .ok_or_else(|| MetricsError::UnknownRequest(request_id.to_owned()))
}

fn histories(config: MetricsConfig) -> Result<MetricsHistories, MetricsError> {
    let requests = history(config.max_request_history)?;
    let (ttft, inter_token_latency, fork_latency) = latency_histories(config)?;
    let (
        cancellation,
        slow_client,
        physical_bytes,
        scheduler_reservations,
        process_memory,
        system_memory,
    ) = observation_histories(config)?;
    Ok((
        requests,
        ttft,
        inter_token_latency,
        fork_latency,
        cancellation,
        slow_client,
        physical_bytes,
        scheduler_reservations,
        process_memory,
        system_memory,
    ))
}

fn latency_histories(config: MetricsConfig) -> Result<LatencyHistories, MetricsError> {
    Ok((
        history(config.max_latency_samples)?,
        history(config.max_latency_samples)?,
        history(config.max_latency_samples)?,
    ))
}

fn observation_histories(config: MetricsConfig) -> Result<ObservationHistories, MetricsError> {
    Ok((
        history(config.max_observation_history)?,
        history(config.max_observation_history)?,
        history(config.max_observation_history)?,
        history(config.max_observation_history)?,
        history(config.max_observation_history)?,
        history(config.max_observation_history)?,
    ))
}

fn empty_process_memory() -> ProcessMemorySample {
    ProcessMemorySample {
        at_ns: 0,
        resident_bytes: Measurement::unavailable("process_resident_unavailable"),
        virtual_bytes: Measurement::unavailable("process_virtual_unavailable"),
    }
}

fn empty_system_memory() -> SystemMemorySample {
    SystemMemorySample {
        at_ns: 0,
        used_bytes: Measurement::unavailable("system_used_unavailable"),
        available_bytes: Measurement::unavailable("system_available_unavailable"),
    }
}

fn token_delta(
    request_id: &str,
    state: &ActiveRequest,
    at_ns: u64,
) -> Result<(bool, u64), MetricsError> {
    let was_first = state.last_token_ns.is_none();
    let previous_ns = state.last_token_ns.unwrap_or(state.started_ns);
    if at_ns < previous_ns {
        return Err(MetricsError::TimestampReversed {
            request_id: request_id.to_owned(),
            at_ns,
            previous_ns,
        });
    }
    state
        .token_count
        .checked_add(1)
        .ok_or(MetricsError::CounterOverflow)?;
    Ok((was_first, at_ns - previous_ns))
}

fn token_batch_delta(
    request_id: &str,
    state: &ActiveRequest,
    at_ns: u64,
    token_count: u64,
) -> Result<(bool, u64, u64), MetricsError> {
    let previous_ns = state.last_token_ns.unwrap_or(state.started_ns);
    if at_ns < previous_ns {
        return Err(MetricsError::TimestampReversed {
            request_id: request_id.to_owned(),
            at_ns,
            previous_ns,
        });
    }
    let next_count = state
        .token_count
        .checked_add(token_count)
        .ok_or(MetricsError::CounterOverflow)?;
    Ok((
        state.last_token_ns.is_none(),
        at_ns - state.started_ns,
        next_count,
    ))
}

fn record_latency_sample(
    first: bool,
    interval: u64,
    ttft: &mut BoundedHistory<u64>,
    inter_token_latency: &mut BoundedHistory<u64>,
) -> Result<(), MetricsError> {
    let history = if first { ttft } else { inter_token_latency };
    history.can_push()?;
    history.push_validated(interval);
    Ok(())
}

fn commit_token_delta(state: &mut ActiveRequest, at_ns: u64) {
    if state.last_token_ns.is_none() {
        state.first_token_ns = Some(at_ns);
    }
    state.last_token_ns = Some(at_ns);
    state.token_count += 1;
}

fn commit_token_batch(state: &mut ActiveRequest, at_ns: u64, next_count: u64, first: bool) {
    if first {
        state.first_token_ns = Some(at_ns);
    }
    state.last_token_ns = Some(at_ns);
    state.token_count = next_count;
}

fn completion_latency(
    request_id: &str,
    state: &ActiveRequest,
    completed_at_ns: Option<u64>,
) -> Result<Measurement<u64>, MetricsError> {
    let minimum_ns = state.last_token_ns.unwrap_or(state.started_ns);
    match completed_at_ns {
        Some(at_ns) if at_ns >= minimum_ns => Ok(Measurement::observed(at_ns - state.started_ns)),
        Some(at_ns) => Err(MetricsError::TimestampReversed {
            request_id: request_id.to_owned(),
            at_ns,
            previous_ns: minimum_ns,
        }),
        None => Ok(Measurement::unavailable("completion_timestamp_unavailable")),
    }
}

fn request_ttft(state: &ActiveRequest) -> Measurement<u64> {
    state
        .first_token_ns
        .map(|at_ns| Measurement::observed(at_ns - state.started_ns))
        .unwrap_or_else(|| Measurement::unavailable("no_token_observed"))
}

fn peak_process_memory(
    current: &ProcessMemorySample,
    candidate: &ProcessMemorySample,
) -> ProcessMemorySample {
    let replace = match (
        current.resident_bytes.value(),
        candidate.resident_bytes.value(),
    ) {
        (None, None) => candidate.virtual_bytes.value() > current.virtual_bytes.value(),
        (current, candidate) => candidate > current,
    };
    if replace {
        candidate.clone()
    } else {
        current.clone()
    }
}

fn peak_system_memory(
    current: &SystemMemorySample,
    candidate: &SystemMemorySample,
) -> SystemMemorySample {
    let replace = match (current.used_bytes.value(), candidate.used_bytes.value()) {
        (None, None) => match (
            current.available_bytes.value(),
            candidate.available_bytes.value(),
        ) {
            (_, None) => false,
            (None, Some(_)) => true,
            (Some(current), Some(candidate)) => candidate < current,
        },
        (current, candidate) => candidate > current,
    };
    if replace {
        candidate.clone()
    } else {
        current.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MetricsConfig {
        MetricsConfig {
            max_in_flight_requests: 2,
            max_request_history: 2,
            max_latency_samples: 3,
            max_observation_history: 2,
            max_outcome_kinds: 4,
            max_identifier_bytes: 32,
            max_reason_bytes: 32,
            max_memory_classes: 4,
        }
    }

    #[test]
    fn linear_quantile_uses_interpolation() {
        assert_eq!(linear_quantile(&[1.0, 2.0, 4.0, 8.0], 0.5), Some(3.0));
        assert_eq!(linear_quantile(&[], 0.95), None);
    }

    #[test]
    fn metrics_keep_outcomes_and_missing_values_explicit() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics.request_started("done", 10).unwrap();
        metrics.token_observed("done", 20).unwrap();
        metrics.token_observed("done", 30).unwrap();
        metrics
            .request_finished("done", Some(40), "completed")
            .unwrap();
        metrics.request_started("cancelled", 50).unwrap();
        metrics
            .request_finished("cancelled", None, "cancelled")
            .unwrap();
        metrics.outcome_observed("rejected").unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.outcomes["completed"], 1);
        assert_eq!(snapshot.outcomes["cancelled"], 1);
        assert_eq!(snapshot.outcomes["rejected"], 1);
        assert_eq!(
            snapshot
                .requests
                .values()
                .nth(1)
                .expect("request retained")
                .ttft_ns
                .status(),
            MeasurementStatus::Unavailable
        );
        assert_eq!(
            snapshot
                .requests
                .values()
                .nth(1)
                .expect("request retained")
                .completion_latency_ns
                .status(),
            MeasurementStatus::Unavailable
        );
        assert_eq!(snapshot.ttft.p95_ns.value(), Some(&10.0));
    }

    #[test]
    fn bounded_history_reports_evictions() {
        let mut history = BoundedHistory::new(2).unwrap();
        history.push(1).unwrap();
        history.push(2).unwrap();
        assert!(history.push(3).unwrap());
        assert_eq!(history.dropped_count(), 1);
        assert_eq!(history.values().copied().collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn physical_and_scheduler_bytes_are_separate() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics
            .physical_bytes_observed(1, BTreeMap::from([(String::from("kv_cache"), 64)]))
            .unwrap();
        metrics
            .scheduler_reservation_observed(1, Some(String::from("r")), 128)
            .unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(
            snapshot
                .physical_bytes
                .values()
                .next()
                .expect("sample retained")
                .bytes_by_class["kv_cache"],
            64
        );
        assert_eq!(
            snapshot
                .scheduler_reservations
                .values()
                .next()
                .expect("sample retained")
                .reserved_bytes,
            128
        );
    }

    #[test]
    fn reversed_token_timestamp_is_rejected() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics.request_started("r", 10).unwrap();
        metrics.token_observed("r", 20).unwrap();
        assert!(matches!(
            metrics.token_observed("r", 19),
            Err(MetricsError::TimestampReversed { .. })
        ));
    }

    #[test]
    fn token_batch_records_count_and_ttft_without_itl_boundaries() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics.request_started("r", 10).unwrap();
        metrics.token_batch_observed("r", 20, 4).unwrap();
        metrics.token_batch_observed("r", 30, 2).unwrap();
        metrics
            .request_finished("r", Some(40), "completed")
            .unwrap();
        let snapshot = metrics.snapshot();
        let request = snapshot.requests.values().next().expect("request retained");
        assert_eq!(request.token_count, 6);
        assert_eq!(
            snapshot.ttft.samples.values().copied().collect::<Vec<_>>(),
            vec![10]
        );
        assert!(snapshot.inter_token_latency.samples.is_empty());
    }

    #[test]
    fn cancellation_without_request_time_is_explicitly_unavailable() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics
            .cancellation_unavailable("r", "disconnect_not_timestamped")
            .unwrap();
        let snapshot = metrics.snapshot();
        let sample = snapshot
            .cancellation
            .values()
            .next()
            .expect("cancellation retained");
        assert_eq!(sample.latency_ns.status(), MeasurementStatus::Unavailable);
    }

    #[test]
    fn config_rejects_unbounded_map_and_text_limits() {
        let mut oversized = config();
        oversized.max_outcome_kinds = usize::MAX;
        assert!(matches!(
            ServiceMetrics::new(oversized),
            Err(MetricsError::CapacityTooLarge { .. })
        ));
        let mut oversized_text = config();
        oversized_text.max_reason_bytes = usize::MAX;
        assert!(matches!(
            ServiceMetrics::new(oversized_text),
            Err(MetricsError::CapacityTooLarge { .. })
        ));
    }

    #[test]
    fn completion_before_last_token_is_rejected_without_removing_request() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics.request_started("r", 10).unwrap();
        metrics.token_observed("r", 100).unwrap();
        assert!(matches!(
            metrics.request_finished("r", Some(20), "completed"),
            Err(MetricsError::TimestampReversed { .. })
        ));
        assert_eq!(metrics.active_request_count(), 1);
    }

    #[test]
    fn observation_series_reject_decreasing_timestamps() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics
            .physical_bytes_observed(100, BTreeMap::new())
            .unwrap();
        assert!(matches!(
            metrics.physical_bytes_observed(99, BTreeMap::new()),
            Err(MetricsError::ObservationTimestampReversed { .. })
        ));
    }

    #[test]
    fn bounded_history_preflights_counter_before_eviction() {
        let mut history = BoundedHistory::new(1).unwrap();
        history.push(7).unwrap();
        history.dropped_count = u64::MAX;
        assert!(matches!(
            history.push(8),
            Err(MetricsError::CounterOverflow)
        ));
        assert_eq!(history.values().copied().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn huge_history_capacity_returns_an_error() {
        assert!(BoundedHistory::<u8>::new(usize::MAX).is_err());
    }

    #[test]
    fn invalid_measurement_shape_is_rejected() {
        let invalid = r#"{"value":7,"status":"unavailable","reason":null}"#;
        assert!(serde_json::from_str::<Measurement<u64>>(invalid).is_err());
    }

    #[test]
    fn latency_samples_and_peaks_survive_history_eviction() {
        let mut metrics = ServiceMetrics::new(MetricsConfig {
            max_in_flight_requests: 2,
            max_request_history: 2,
            max_latency_samples: 1,
            max_observation_history: 1,
            max_outcome_kinds: 4,
            max_identifier_bytes: 32,
            max_reason_bytes: 32,
            max_memory_classes: 4,
        })
        .unwrap();
        metrics.request_started("a", 0).unwrap();
        metrics.token_observed("a", 10).unwrap();
        metrics
            .request_finished("a", Some(10), "completed")
            .unwrap();
        metrics.request_started("b", 20).unwrap();
        metrics.token_observed("b", 30).unwrap();
        metrics
            .request_finished("b", Some(30), "completed")
            .unwrap();
        metrics
            .physical_bytes_observed(1, BTreeMap::from([(String::from("kv"), 10)]))
            .unwrap();
        metrics
            .physical_bytes_observed(2, BTreeMap::from([(String::from("kv"), 20)]))
            .unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.ttft.sample_count, 2);
        assert_eq!(
            snapshot.ttft.samples.values().copied().collect::<Vec<_>>(),
            vec![10]
        );
        assert_eq!(snapshot.physical_bytes_peak_by_class["kv"], 20);
    }

    #[test]
    fn memory_peak_timestamp_stays_with_latest_peak_update() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics
            .process_memory_observed(10, Some(100), Some(200))
            .unwrap();
        metrics
            .process_memory_observed(20, Some(50), Some(150))
            .unwrap();
        metrics
            .system_memory_observed(10, Some(300), Some(400))
            .unwrap();
        metrics
            .system_memory_observed(20, Some(250), Some(350))
            .unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.process_memory_peak.at_ns, 10);
        assert_eq!(snapshot.system_memory_peak.at_ns, 10);
    }

    #[test]
    fn memory_peak_keeps_one_observed_sample() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics
            .process_memory_observed(10, Some(100), Some(200))
            .unwrap();
        metrics
            .process_memory_observed(20, Some(50), Some(300))
            .unwrap();
        metrics
            .system_memory_observed(30, Some(200), Some(800))
            .unwrap();
        metrics
            .system_memory_observed(40, Some(100), Some(900))
            .unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.process_memory_peak.at_ns, 10);
        assert_eq!(
            snapshot.process_memory_peak.resident_bytes.value(),
            Some(&100)
        );
        assert_eq!(
            snapshot.process_memory_peak.virtual_bytes.value(),
            Some(&200)
        );
        assert_eq!(snapshot.system_memory_peak.at_ns, 30);
        assert_eq!(snapshot.system_memory_peak.used_bytes.value(), Some(&200));
        assert_eq!(
            snapshot.system_memory_peak.available_bytes.value(),
            Some(&800)
        );
    }

    #[test]
    fn process_peak_never_compares_resident_bytes_with_virtual_bytes() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics
            .process_memory_observed(10, None, Some(1_000))
            .unwrap();
        metrics
            .process_memory_observed(20, Some(100), Some(2_000))
            .unwrap();
        let peak = metrics.snapshot().process_memory_peak;
        assert_eq!(peak.at_ns, 20);
        assert_eq!(peak.resident_bytes.value(), Some(&100));
        assert_eq!(peak.virtual_bytes.value(), Some(&2_000));

        metrics
            .process_memory_observed(30, None, Some(3_000))
            .unwrap();
        metrics.process_memory_observed(40, None, None).unwrap();
        assert_eq!(metrics.snapshot().process_memory_peak, peak);
    }

    #[test]
    fn process_peak_uses_virtual_bytes_only_without_resident_observations() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics.process_memory_observed(10, None, None).unwrap();
        metrics
            .process_memory_observed(20, None, Some(200))
            .unwrap();
        metrics
            .process_memory_observed(30, None, Some(100))
            .unwrap();
        let peak = metrics.snapshot().process_memory_peak;
        assert_eq!(peak.at_ns, 20);
        assert_eq!(peak.resident_bytes.status(), MeasurementStatus::Unavailable);
        assert_eq!(peak.virtual_bytes.value(), Some(&200));
    }

    #[test]
    fn system_peak_prefers_used_bytes_over_available_only_observations() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics
            .system_memory_observed(10, None, Some(1_000))
            .unwrap();
        metrics
            .system_memory_observed(20, Some(100), Some(2_000))
            .unwrap();
        let peak = metrics.snapshot().system_memory_peak;
        assert_eq!(peak.at_ns, 20);
        assert_eq!(peak.used_bytes.value(), Some(&100));
        assert_eq!(peak.available_bytes.value(), Some(&2_000));

        metrics.system_memory_observed(30, None, Some(1)).unwrap();
        metrics.system_memory_observed(40, None, None).unwrap();
        assert_eq!(metrics.snapshot().system_memory_peak, peak);
    }

    #[test]
    fn system_peak_selects_least_available_bytes_when_used_is_unavailable() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics.system_memory_observed(10, None, None).unwrap();
        metrics.system_memory_observed(20, None, Some(200)).unwrap();
        metrics.system_memory_observed(30, None, Some(100)).unwrap();
        metrics.system_memory_observed(40, None, Some(300)).unwrap();
        let peak = metrics.snapshot().system_memory_peak;
        assert_eq!(peak.at_ns, 30);
        assert_eq!(peak.used_bytes.status(), MeasurementStatus::Unavailable);
        assert_eq!(peak.available_bytes.value(), Some(&100));
    }

    #[test]
    fn serialized_latency_history_reproduces_sample_count_and_values() {
        let mut metrics = ServiceMetrics::new(config()).unwrap();
        metrics.request_started("r", 0).unwrap();
        metrics.token_observed("r", 10).unwrap();
        metrics.token_observed("r", 20).unwrap();
        let value = serde_json::to_value(metrics.snapshot()).unwrap();
        let history = &value["ttft"]["samples"];
        assert_eq!(history["sample_count"], 1);
        assert_eq!(history["values"], serde_json::json!([10]));
        assert_eq!(value["quantile_definition"], QUANTILE_DEFINITION);
        assert_eq!(value["clock_domain"], CLOCK_DOMAIN);
    }
}
