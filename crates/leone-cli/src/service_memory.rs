use leone::{
    MemoryAccounting, MemoryAllocation, MemoryBudget, MemoryClass, MemoryError, MemoryReservation,
    MemoryTracker, MemoryTrackerRoot,
};
use std::num::NonZeroU64;
#[cfg(target_os = "macos")]
use std::process::Command;
use thiserror::Error;

/// Numerator for the automatic four-fifths capacity policy.
pub const AUTO_HEADROOM_NUMERATOR: u64 = 4;
/// Denominator for the automatic four-fifths capacity policy.
pub const AUTO_HEADROOM_DENOMINATOR: u64 = 5;

/// Reserves a finite service limit automatically or uses an explicit byte count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetRequest {
    Auto,
    Bytes(NonZeroU64),
}

impl BudgetRequest {
    /// Parses `auto` or a positive decimal byte count.
    pub fn parse(value: &str) -> Result<Self, BudgetRequestError> {
        if value == "auto" {
            return Ok(Self::Auto);
        }
        let bytes = value
            .parse::<u64>()
            .map_err(|_| BudgetRequestError::InvalidNumber)?;
        NonZeroU64::new(bytes)
            .map(Self::Bytes)
            .ok_or(BudgetRequestError::Zero)
    }

    fn is_explicit(self) -> bool {
        matches!(self, Self::Bytes(_))
    }
}

/// Reports an invalid finite service limit request.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BudgetRequestError {
    #[error("budget request is not a positive decimal byte count")]
    InvalidNumber,
    #[error("budget request must be nonzero")]
    Zero,
}

/// Service limits that are resolved before model loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceBudgetArgs {
    pub memory: BudgetRequest,
    pub host: BudgetRequest,
    pub kv_reservation: BudgetRequest,
}

impl Default for ServiceBudgetArgs {
    fn default() -> Self {
        Self {
            memory: BudgetRequest::Auto,
            host: BudgetRequest::Auto,
            kv_reservation: BudgetRequest::Auto,
        }
    }
}

/// Describes the physical memory topology used by the service policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTopology {
    Discrete,
    Unified,
    Cpu,
}

/// Reports an unavailable capacity observation without inventing a byte count.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ObservationUnavailable {
    #[error("the platform has no supported memory observation")]
    UnsupportedPlatform,
    #[error("the memory observation source is unavailable")]
    SourceUnavailable,
    #[error("the memory observation source returned malformed data")]
    Malformed,
    #[error("the memory observation is internally inconsistent")]
    Inconsistent,
}

/// Reports a host observation or its typed unavailable state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemoryObservation {
    Available {
        total_bytes: u64,
        available_bytes: u64,
        source: HostMemorySource,
        semantics: HostMemorySemantics,
    },
    Unavailable {
        reason: ObservationUnavailable,
    },
}

/// Identifies the operating system source for a host observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemorySource {
    LinuxMeminfo,
    MacVmStat,
}

/// Describes how an operating system produced the host capacity estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemorySemantics {
    /// Linux `MemAvailable`, which is a kernel estimate rather than a free-page count.
    KernelAvailableEstimate,
    /// macOS free and file-backed pages from one `vm_stat` snapshot.
    ReclaimablePagesEstimate,
    /// macOS free pages when file-backed pages are absent from `vm_stat`.
    FreePagesOnly,
}

/// Reports backend capacity without calling guidance free memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendCapacityObservation {
    Discrete {
        total_bytes: u64,
        available_bytes: u64,
    },
    Recommended {
        working_set_bytes: u64,
        current_allocated_bytes: u64,
        unified_memory: bool,
    },
    Cpu,
    Unavailable {
        reason: ObservationUnavailable,
    },
}

/// Inputs collected before service policy resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceMemoryInputs {
    pub topology: MemoryTopology,
    pub backend: BackendCapacityObservation,
    pub host: HostMemoryObservation,
    pub backend_owned_bytes: u64,
    pub host_owned_bytes: u64,
}

/// Resolved finite backend and host limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceMemoryPools {
    Discrete {
        backend_bytes: NonZeroU64,
        host_bytes: NonZeroU64,
    },
    Shared {
        combined_bytes: NonZeroU64,
        host_bytes: NonZeroU64,
    },
}

impl ServiceMemoryPools {
    /// Returns the limit that includes backend and host storage for this pool.
    pub fn combined_bytes(self) -> NonZeroU64 {
        match self {
            Self::Discrete { backend_bytes, .. } => backend_bytes,
            Self::Shared { combined_bytes, .. } => combined_bytes,
        }
    }

    /// Returns the independent host storage limit.
    pub fn host_bytes(self) -> NonZeroU64 {
        match self {
            Self::Discrete { host_bytes, .. } | Self::Shared { host_bytes, .. } => host_bytes,
        }
    }
}

/// Records which service settings replaced automatic policy values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetOverrides {
    pub memory: bool,
    pub host: bool,
    pub kv_reservation: bool,
}

/// A finite policy suitable for pre-load backend configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceMemoryPolicy {
    pools: ServiceMemoryPools,
    kv_reservation: BudgetRequest,
    overrides: BudgetOverrides,
    backend_observation: BackendCapacityObservation,
    host_observation: HostMemoryObservation,
}

impl ServiceMemoryPolicy {
    /// Returns the resolved pool limits.
    pub fn pools(self) -> ServiceMemoryPools {
        self.pools
    }

    /// Returns the host limit used by a host allocation ledger.
    pub fn host_budget(self) -> NonZeroU64 {
        self.pools.host_bytes()
    }

    /// Returns the combined parent budget for unified and CPU pools.
    ///
    /// The caller creates one shared tracker root from this value before loading.
    pub fn shared_budget(self) -> Option<MemoryBudget> {
        match self.pools {
            ServiceMemoryPools::Discrete { .. } => None,
            ServiceMemoryPools::Shared { combined_bytes, .. } => {
                Some(MemoryBudget::Bytes(combined_bytes))
            }
        }
    }

    /// Returns the logical KV reservation request for post-load resolution.
    pub fn kv_reservation(self) -> BudgetRequest {
        self.kv_reservation
    }

    /// Returns override flags for receipt construction.
    pub fn overrides(self) -> BudgetOverrides {
        self.overrides
    }

    /// Returns the observations used to resolve this policy.
    pub fn observations(self) -> (BackendCapacityObservation, HostMemoryObservation) {
        (self.backend_observation, self.host_observation)
    }

    /// Computes a diagnostic remainder from current host lease counters.
    ///
    /// Shared pools enforce the combined limit through the tracker root.
    pub fn effective_backend_budget(
        self,
        host_owned_bytes: u64,
        host_reserved_bytes: u64,
    ) -> Result<MemoryBudget, MemoryPolicyError> {
        match self.pools {
            ServiceMemoryPools::Discrete { backend_bytes, .. } => {
                limited_budget(backend_bytes, "backend")
            }
            ServiceMemoryPools::Shared { combined_bytes, .. } => {
                let host_total =
                    checked_add(host_owned_bytes, host_reserved_bytes, "host allocations")?;
                let remaining = combined_bytes.get().checked_sub(host_total).ok_or(
                    MemoryPolicyError::CombinedBelowHost {
                        combined: combined_bytes.get(),
                        host: host_total,
                    },
                )?;
                let remaining = positive(remaining, "effective backend")?;
                limited_budget(remaining, "effective backend")
            }
        }
    }
}

/// Resolves finite service limits from observations and requests.
pub fn resolve_policy(
    args: ServiceBudgetArgs,
    inputs: ServiceMemoryInputs,
) -> Result<ServiceMemoryPolicy, MemoryPolicyError> {
    let host = host_capacity(inputs.host)?;
    let combined_owned = checked_add(
        inputs.backend_owned_bytes,
        inputs.host_owned_bytes,
        "owned service memory",
    )?;
    let host_limit = resolve_limit(args.host, host, inputs.host_owned_bytes, "host")?;
    let result = match inputs.topology {
        MemoryTopology::Discrete => resolve_discrete(args, inputs, host_limit),
        MemoryTopology::Unified => resolve_unified(args, inputs, host, host_limit, combined_owned),
        MemoryTopology::Cpu => resolve_cpu(args, inputs, host, host_limit, combined_owned),
    }?;
    if result.pools.host_bytes().get() < inputs.host_owned_bytes {
        return Err(MemoryPolicyError::HostBelowOwned {
            requested: result.pools.host_bytes().get(),
            owned: inputs.host_owned_bytes,
        });
    }
    Ok(result)
}

/// Resolves a logical KV reservation after model loading.
pub fn resolve_kv_reservation(
    request: BudgetRequest,
    bytes_per_token: u64,
    context_limit: u64,
    max_active_requests: u64,
    backend_limit: u64,
    owned_non_kv_bytes: u64,
    transient_headroom_bytes: u64,
) -> Result<NonZeroU64, MemoryPolicyError> {
    let logical = checked_mul(bytes_per_token, context_limit, "KV reservation")?;
    let logical = checked_mul(logical, max_active_requests, "KV reservation")?;
    let non_kv = checked_add(
        owned_non_kv_bytes,
        transient_headroom_bytes,
        "non-KV allocation",
    )?;
    let physical = backend_limit
        .checked_sub(non_kv)
        .ok_or(MemoryPolicyError::ZeroCapacity {
            field: "physical KV reservation",
        })?;
    let maximum = logical.min(physical);
    let maximum = positive(maximum, "KV reservation")?;
    match request {
        BudgetRequest::Auto => Ok(maximum),
        BudgetRequest::Bytes(requested) if requested > maximum => {
            Err(MemoryPolicyError::KvReservationExceedsLimit {
                requested: requested.get(),
                limit: maximum.get(),
            })
        }
        BudgetRequest::Bytes(requested) => Ok(requested),
    }
}

/// Captures host memory for one service process without treating mapped bytes as resident.
pub fn observe_host_memory() -> HostMemoryObservation {
    #[cfg(target_os = "linux")]
    {
        return observe_linux_memory();
    }
    #[cfg(target_os = "macos")]
    {
        return observe_macos_memory();
    }
    #[allow(unreachable_code)]
    HostMemoryObservation::Unavailable {
        reason: ObservationUnavailable::UnsupportedPlatform,
    }
}

/// Tracks host snapshots and staging through the checked allocation ledger.
#[derive(Debug, Clone)]
pub struct HostMemoryLedger {
    tracker: MemoryTracker,
}

impl HostMemoryLedger {
    /// Creates a finite host ledger with the backend reservation and rollback rules.
    pub fn new(limit: NonZeroU64) -> Self {
        Self::from_tracker(MemoryTracker::new(MemoryBudget::Bytes(limit)))
    }

    /// Creates a host child ledger under one shared physical parent.
    pub fn child(limit: NonZeroU64, root: MemoryTrackerRoot) -> Self {
        Self::from_tracker(MemoryTracker::child(MemoryBudget::Bytes(limit), root))
    }

    /// Wraps an existing host tracker without creating another quota.
    pub fn from_tracker(tracker: MemoryTracker) -> Self {
        Self { tracker }
    }

    /// Returns the tracker used by host staging and snapshots.
    pub fn tracker(&self) -> MemoryTracker {
        self.tracker.clone()
    }

    /// Returns the shared parent charged by this host ledger.
    pub fn root(&self) -> MemoryTrackerRoot {
        self.tracker.root()
    }

    /// Changes the host limit when owned and pending bytes fit.
    pub fn set_limit(&self, limit: NonZeroU64) -> Result<(), MemoryError> {
        self.tracker.set_budget(MemoryBudget::Bytes(limit))
    }

    /// Reserves bytes before host storage construction begins.
    pub fn reserve(&self, bytes: u64) -> Result<MemoryReservation, MemoryError> {
        self.tracker.reserve(MemoryClass::ContractBuffer, bytes)
    }

    /// Allocates one committed host lease through the checked ledger.
    pub fn allocate(&self, bytes: u64) -> Result<MemoryAllocation, MemoryError> {
        self.tracker.allocate(MemoryClass::ContractBuffer, bytes)
    }

    /// Returns the host allocation accounting snapshot.
    pub fn snapshot(&self) -> MemoryAccounting {
        self.tracker.snapshot()
    }
}

/// Reports an invalid or unavailable service memory policy input.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPolicyError {
    #[error("{pool} capacity is unavailable: {reason}")]
    MissingObservation {
        pool: &'static str,
        reason: ObservationUnavailable,
    },
    #[error("{pool} capacity observation is invalid: {reason}")]
    InvalidObservation {
        pool: &'static str,
        reason: ObservationInvalid,
    },
    #[error("{field} arithmetic overflows")]
    Overflow { field: &'static str },
    #[error("{field} resolves to zero")]
    ZeroCapacity { field: &'static str },
    #[error("{pool} owns {owned} bytes but the observation totals {total} bytes")]
    BaselineExceedsObserved {
        pool: &'static str,
        owned: u64,
        total: u64,
    },
    #[error("{field} requests {requested} bytes above observed capacity {maximum}")]
    ExplicitExceedsObserved {
        field: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("{field} requests {requested} bytes below {owned} owned bytes")]
    ExplicitBelowOwned {
        field: &'static str,
        requested: u64,
        owned: u64,
    },
    #[error("{field} resolves to {limit} bytes below {owned} owned bytes")]
    ResolvedBelowOwned {
        field: &'static str,
        limit: u64,
        owned: u64,
    },
    #[error("host limit {requested} exceeds shared limit {combined}")]
    HostExceedsCombined { requested: u64, combined: u64 },
    #[error("host limit {requested} is below {owned} owned bytes")]
    HostBelowOwned { requested: u64, owned: u64 },
    #[error("shared limit {combined} is below {host} host bytes")]
    CombinedBelowHost { combined: u64, host: u64 },
    #[error("requested KV reservation {requested} exceeds the allowed limit {limit}")]
    KvReservationExceedsLimit { requested: u64, limit: u64 },
    #[error("backend capacity observation does not match the selected topology")]
    TopologyMismatch,
}

/// Explains why a known capacity observation cannot be used.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ObservationInvalid {
    #[error("total bytes are zero")]
    ZeroTotal,
    #[error("available bytes exceed total bytes")]
    AvailableExceedsTotal,
    #[error("working-set guidance is zero")]
    ZeroWorkingSet,
    #[error("working-set guidance is not marked unified")]
    NotUnified,
}

#[derive(Debug, Clone, Copy)]
enum CapacityState {
    Known {
        total_bytes: u64,
        available_bytes: u64,
    },
    Unavailable {
        reason: ObservationUnavailable,
    },
}

#[derive(Debug, Clone, Copy)]
struct Recommendation {
    working_set_bytes: u64,
}

fn resolve_discrete(
    args: ServiceBudgetArgs,
    inputs: ServiceMemoryInputs,
    host_limit: NonZeroU64,
) -> Result<ServiceMemoryPolicy, MemoryPolicyError> {
    let backend = discrete_capacity(inputs.backend)?;
    let backend_limit = resolve_limit(args.memory, backend, inputs.backend_owned_bytes, "backend")?;
    let pools = ServiceMemoryPools::Discrete {
        backend_bytes: backend_limit,
        host_bytes: host_limit,
    };
    Ok(make_policy(args, pools, inputs.backend, inputs.host))
}

fn resolve_unified(
    args: ServiceBudgetArgs,
    inputs: ServiceMemoryInputs,
    host: CapacityState,
    host_limit: NonZeroU64,
    combined_owned: u64,
) -> Result<ServiceMemoryPolicy, MemoryPolicyError> {
    let recommendation = unified_recommendation(inputs.backend)?;
    let combined = resolve_combined(args.memory, host, recommendation, combined_owned)?;
    let host_limit = cap_host_limit(args.host, host_limit, combined)?;
    let pools = ServiceMemoryPools::Shared {
        combined_bytes: combined,
        host_bytes: host_limit,
    };
    Ok(make_policy(args, pools, inputs.backend, inputs.host))
}

fn resolve_cpu(
    args: ServiceBudgetArgs,
    inputs: ServiceMemoryInputs,
    host: CapacityState,
    host_limit: NonZeroU64,
    combined_owned: u64,
) -> Result<ServiceMemoryPolicy, MemoryPolicyError> {
    if !matches!(inputs.backend, BackendCapacityObservation::Cpu) {
        return Err(MemoryPolicyError::TopologyMismatch);
    }
    let combined = resolve_limit(args.memory, host, combined_owned, "backend")?;
    ensure_at_least(combined, combined_owned, "backend")?;
    let host_limit = cap_host_limit(args.host, host_limit, combined)?;
    let pools = ServiceMemoryPools::Shared {
        combined_bytes: combined,
        host_bytes: host_limit,
    };
    Ok(make_policy(args, pools, inputs.backend, inputs.host))
}

fn make_policy(
    args: ServiceBudgetArgs,
    pools: ServiceMemoryPools,
    backend_observation: BackendCapacityObservation,
    host_observation: HostMemoryObservation,
) -> ServiceMemoryPolicy {
    ServiceMemoryPolicy {
        pools,
        kv_reservation: args.kv_reservation,
        overrides: BudgetOverrides {
            memory: args.memory.is_explicit(),
            host: args.host.is_explicit(),
            kv_reservation: args.kv_reservation.is_explicit(),
        },
        backend_observation,
        host_observation,
    }
}

fn resolve_limit(
    request: BudgetRequest,
    capacity: CapacityState,
    owned: u64,
    field: &'static str,
) -> Result<NonZeroU64, MemoryPolicyError> {
    match request {
        BudgetRequest::Auto => resolve_automatic(capacity, owned, field),
        BudgetRequest::Bytes(requested) => resolve_explicit(requested, capacity, owned, field),
    }
}

fn resolve_automatic(
    capacity: CapacityState,
    owned: u64,
    field: &'static str,
) -> Result<NonZeroU64, MemoryPolicyError> {
    let CapacityState::Known {
        total_bytes,
        available_bytes,
    } = capacity
    else {
        let CapacityState::Unavailable { reason } = capacity else {
            unreachable!("capacity state match is exhaustive")
        };
        return Err(MemoryPolicyError::MissingObservation {
            pool: field,
            reason,
        });
    };
    let maximum = observed_maximum(total_bytes, available_bytes, owned, field)?;
    let headroom = four_fifths(available_bytes, field)?;
    let automatic = checked_add(owned, headroom, field)?.min(maximum);
    ensure_at_least(positive(automatic, field)?, owned, field)
}

fn resolve_explicit(
    requested: NonZeroU64,
    capacity: CapacityState,
    owned: u64,
    field: &'static str,
) -> Result<NonZeroU64, MemoryPolicyError> {
    if requested.get() < owned {
        return Err(MemoryPolicyError::ExplicitBelowOwned {
            field,
            requested: requested.get(),
            owned,
        });
    }
    if let CapacityState::Known {
        total_bytes,
        available_bytes,
    } = capacity
    {
        let maximum = observed_maximum(total_bytes, available_bytes, owned, field)?;
        if requested.get() > maximum {
            return Err(MemoryPolicyError::ExplicitExceedsObserved {
                field,
                requested: requested.get(),
                maximum,
            });
        }
    }
    Ok(requested)
}

fn observed_maximum(
    total_bytes: u64,
    available_bytes: u64,
    owned: u64,
    field: &'static str,
) -> Result<u64, MemoryPolicyError> {
    if owned > total_bytes {
        return Err(MemoryPolicyError::BaselineExceedsObserved {
            pool: field,
            owned,
            total: total_bytes,
        });
    }
    Ok(total_bytes.min(checked_add(owned, available_bytes, field)?))
}

fn resolve_combined(
    request: BudgetRequest,
    host: CapacityState,
    recommendation: Option<Recommendation>,
    owned: u64,
) -> Result<NonZeroU64, MemoryPolicyError> {
    match request {
        request @ BudgetRequest::Bytes(_) => resolve_limit(request, host, owned, "backend"),
        BudgetRequest::Auto => {
            let recommendation = recommendation.ok_or(MemoryPolicyError::MissingObservation {
                pool: "backend",
                reason: ObservationUnavailable::SourceUnavailable,
            })?;
            let host = match host {
                CapacityState::Known {
                    total_bytes,
                    available_bytes,
                    ..
                } => CapacityState::Known {
                    total_bytes,
                    available_bytes,
                },
                CapacityState::Unavailable { reason } => {
                    return Err(MemoryPolicyError::MissingObservation {
                        pool: "host",
                        reason,
                    });
                }
            };
            let guidance = four_fifths(recommendation.working_set_bytes, "backend")?;
            let host_derived = resolve_automatic(host, owned, "backend")?;
            positive(guidance.min(host_derived.get()), "backend")
                .and_then(|limit| ensure_at_least(limit, owned, "backend"))
        }
    }
}

fn cap_host_limit(
    request: BudgetRequest,
    host: NonZeroU64,
    combined: NonZeroU64,
) -> Result<NonZeroU64, MemoryPolicyError> {
    if host <= combined {
        return Ok(host);
    }
    if request.is_explicit() {
        return Err(MemoryPolicyError::HostExceedsCombined {
            requested: host.get(),
            combined: combined.get(),
        });
    }
    Ok(combined)
}

fn host_capacity(observation: HostMemoryObservation) -> Result<CapacityState, MemoryPolicyError> {
    match observation {
        HostMemoryObservation::Available {
            total_bytes,
            available_bytes,
            ..
        } => checked_capacity("host", total_bytes, available_bytes),
        HostMemoryObservation::Unavailable { reason } => Ok(CapacityState::Unavailable { reason }),
    }
}

fn discrete_capacity(
    observation: BackendCapacityObservation,
) -> Result<CapacityState, MemoryPolicyError> {
    match observation {
        BackendCapacityObservation::Discrete {
            total_bytes,
            available_bytes,
        } => checked_capacity("backend", total_bytes, available_bytes),
        BackendCapacityObservation::Unavailable { reason } => {
            Ok(CapacityState::Unavailable { reason })
        }
        _ => Err(MemoryPolicyError::TopologyMismatch),
    }
}

fn unified_recommendation(
    observation: BackendCapacityObservation,
) -> Result<Option<Recommendation>, MemoryPolicyError> {
    match observation {
        BackendCapacityObservation::Recommended {
            working_set_bytes,
            current_allocated_bytes: _,
            unified_memory,
        } => {
            if !unified_memory {
                return Err(MemoryPolicyError::InvalidObservation {
                    pool: "backend",
                    reason: ObservationInvalid::NotUnified,
                });
            }
            if working_set_bytes == 0 {
                return Err(MemoryPolicyError::InvalidObservation {
                    pool: "backend",
                    reason: ObservationInvalid::ZeroWorkingSet,
                });
            }
            Ok(Some(Recommendation { working_set_bytes }))
        }
        BackendCapacityObservation::Unavailable { .. } => Ok(None),
        _ => Err(MemoryPolicyError::TopologyMismatch),
    }
}

fn checked_capacity(
    pool: &'static str,
    total_bytes: u64,
    available_bytes: u64,
) -> Result<CapacityState, MemoryPolicyError> {
    if total_bytes == 0 {
        return Err(MemoryPolicyError::InvalidObservation {
            pool,
            reason: ObservationInvalid::ZeroTotal,
        });
    }
    if available_bytes > total_bytes {
        return Err(MemoryPolicyError::InvalidObservation {
            pool,
            reason: ObservationInvalid::AvailableExceedsTotal,
        });
    }
    Ok(CapacityState::Known {
        total_bytes,
        available_bytes,
    })
}

fn ensure_at_least(
    limit: NonZeroU64,
    owned: u64,
    field: &'static str,
) -> Result<NonZeroU64, MemoryPolicyError> {
    if limit.get() < owned {
        return Err(MemoryPolicyError::ResolvedBelowOwned {
            field,
            limit: limit.get(),
            owned,
        });
    }
    Ok(limit)
}

fn limited_budget(
    limit: NonZeroU64,
    field: &'static str,
) -> Result<MemoryBudget, MemoryPolicyError> {
    MemoryBudget::limited(limit.get()).map_err(|_| MemoryPolicyError::ZeroCapacity { field })
}

fn positive(value: u64, field: &'static str) -> Result<NonZeroU64, MemoryPolicyError> {
    NonZeroU64::new(value).ok_or(MemoryPolicyError::ZeroCapacity { field })
}

fn checked_add(left: u64, right: u64, field: &'static str) -> Result<u64, MemoryPolicyError> {
    left.checked_add(right)
        .ok_or(MemoryPolicyError::Overflow { field })
}

fn checked_mul(left: u64, right: u64, field: &'static str) -> Result<u64, MemoryPolicyError> {
    left.checked_mul(right)
        .ok_or(MemoryPolicyError::Overflow { field })
}

fn four_fifths(value: u64, field: &'static str) -> Result<u64, MemoryPolicyError> {
    let quotient = value / AUTO_HEADROOM_DENOMINATOR;
    let remainder = value % AUTO_HEADROOM_DENOMINATOR;
    quotient
        .checked_mul(AUTO_HEADROOM_NUMERATOR)
        .and_then(|value| {
            value.checked_add(remainder * AUTO_HEADROOM_NUMERATOR / AUTO_HEADROOM_DENOMINATOR)
        })
        .ok_or(MemoryPolicyError::Overflow { field })
}

#[cfg(target_os = "linux")]
fn observe_linux_memory() -> HostMemoryObservation {
    match std::fs::read_to_string("/proc/meminfo") {
        Ok(contents) => parse_linux_meminfo(&contents),
        Err(_) => unavailable_source(),
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_meminfo(contents: &str) -> HostMemoryObservation {
    let total = parse_meminfo_kib(contents, "MemTotal:");
    let available = parse_meminfo_kib(contents, "MemAvailable:");
    match (total, available) {
        (Some(total), Some(available)) => {
            match (total.checked_mul(1024), available.checked_mul(1024)) {
                (Some(total_bytes), Some(available_bytes)) => HostMemoryObservation::Available {
                    total_bytes,
                    available_bytes,
                    source: HostMemorySource::LinuxMeminfo,
                    semantics: HostMemorySemantics::KernelAvailableEstimate,
                },
                _ => unavailable_malformed(),
            }
        }
        _ => unavailable_malformed(),
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(contents: &str, key: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let mut fields = line.strip_prefix(key)?.split_whitespace();
        let value = fields.next()?;
        if fields.next()? != "kB" {
            return None;
        }
        value.parse().ok()
    })
}

#[cfg(target_os = "macos")]
fn observe_macos_memory() -> HostMemoryObservation {
    let total = command_u64("sysctl", &["-n", "hw.memsize"]);
    let Ok(output) = Command::new("vm_stat").output() else {
        return unavailable_source();
    };
    if !output.status.success() {
        return unavailable_source();
    }
    let Ok(contents) = String::from_utf8(output.stdout) else {
        return unavailable_malformed();
    };
    let Some(page_size) = parse_vm_stat_page_size(&contents) else {
        return unavailable_malformed();
    };
    let Some((free_pages, file_backed_pages)) = parse_vm_stat(&contents) else {
        return unavailable_malformed();
    };
    let Some(total) = total else {
        return unavailable_source();
    };
    let Some(available_pages) = free_pages.checked_add(file_backed_pages.unwrap_or(0)) else {
        return unavailable_malformed();
    };
    let Some(available) = available_pages.checked_mul(page_size) else {
        return unavailable_malformed();
    };
    if available > total {
        return unavailable_inconsistent();
    }
    let semantics = if file_backed_pages.is_some() {
        HostMemorySemantics::ReclaimablePagesEstimate
    } else {
        HostMemorySemantics::FreePagesOnly
    };
    HostMemoryObservation::Available {
        total_bytes: total,
        available_bytes: available,
        source: HostMemorySource::MacVmStat,
        semantics,
    }
}

#[cfg(target_os = "macos")]
fn parse_vm_stat(contents: &str) -> Option<(u64, Option<u64>)> {
    let free = parse_vm_stat_pages(contents, "Pages free:")?;
    let file_backed = parse_vm_stat_pages(contents, "File-backed pages:");
    Some((free, file_backed))
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_page_size(contents: &str) -> Option<u64> {
    let prefix = "page size of ";
    let value = contents.lines().find_map(|line| {
        let start = line.find(prefix)? + prefix.len();
        line[start..].split_whitespace().next()
    })?;
    value.parse().ok()
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_pages(contents: &str, label: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let value = line.strip_prefix(label)?.trim().trim_end_matches('.');
        value.parse().ok()
    })
}

#[cfg(target_os = "macos")]
fn command_u64(command: &str, args: &[&str]) -> Option<u64> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

fn unavailable_source() -> HostMemoryObservation {
    HostMemoryObservation::Unavailable {
        reason: ObservationUnavailable::SourceUnavailable,
    }
}

fn unavailable_malformed() -> HostMemoryObservation {
    HostMemoryObservation::Unavailable {
        reason: ObservationUnavailable::Malformed,
    }
}

#[cfg(target_os = "macos")]
fn unavailable_inconsistent() -> HostMemoryObservation {
    HostMemoryObservation::Unavailable {
        reason: ObservationUnavailable::Inconsistent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).expect("test value is nonzero")
    }

    fn host(available_bytes: u64) -> HostMemoryObservation {
        HostMemoryObservation::Available {
            total_bytes: available_bytes.max(1),
            available_bytes,
            source: HostMemorySource::LinuxMeminfo,
            semantics: HostMemorySemantics::KernelAvailableEstimate,
        }
    }

    fn discrete_inputs(
        backend: BackendCapacityObservation,
        backend_owned_bytes: u64,
        host_owned_bytes: u64,
    ) -> ServiceMemoryInputs {
        ServiceMemoryInputs {
            topology: MemoryTopology::Discrete,
            backend,
            host: host(200),
            backend_owned_bytes,
            host_owned_bytes,
        }
    }

    #[test]
    fn requests_are_finite_and_parse_auto() {
        assert_eq!(BudgetRequest::parse("auto"), Ok(BudgetRequest::Auto));
        assert_eq!(BudgetRequest::parse("32"), Ok(BudgetRequest::Bytes(nz(32))));
        assert_eq!(BudgetRequest::parse("0"), Err(BudgetRequestError::Zero));
        assert_eq!(
            BudgetRequest::parse("none"),
            Err(BudgetRequestError::InvalidNumber)
        );
    }

    #[test]
    fn four_fifths_handles_large_values_without_multiplication_overflow() {
        let value = u64::MAX - 1;
        let expected = (value / AUTO_HEADROOM_DENOMINATOR) * AUTO_HEADROOM_NUMERATOR
            + (value % AUTO_HEADROOM_DENOMINATOR) * AUTO_HEADROOM_NUMERATOR
                / AUTO_HEADROOM_DENOMINATOR;
        assert_eq!(four_fifths(value, "test"), Ok(expected));
    }

    #[test]
    fn unavailable_backend_requires_an_explicit_finite_limit() {
        let inputs = discrete_inputs(
            BackendCapacityObservation::Unavailable {
                reason: ObservationUnavailable::SourceUnavailable,
            },
            0,
            0,
        );
        let error = resolve_policy(ServiceBudgetArgs::default(), inputs).unwrap_err();
        assert_eq!(
            error,
            MemoryPolicyError::MissingObservation {
                pool: "backend",
                reason: ObservationUnavailable::SourceUnavailable,
            }
        );
        let args = ServiceBudgetArgs {
            memory: BudgetRequest::Bytes(nz(64)),
            ..ServiceBudgetArgs::default()
        };
        let policy = resolve_policy(args, inputs).unwrap();
        assert_eq!(policy.pools().combined_bytes(), nz(64));
    }

    #[test]
    fn unavailable_host_requires_an_explicit_host_limit() {
        let inputs = ServiceMemoryInputs {
            topology: MemoryTopology::Discrete,
            backend: BackendCapacityObservation::Discrete {
                total_bytes: 200,
                available_bytes: 100,
            },
            host: HostMemoryObservation::Unavailable {
                reason: ObservationUnavailable::UnsupportedPlatform,
            },
            backend_owned_bytes: 0,
            host_owned_bytes: 0,
        };
        let policy = resolve_policy(ServiceBudgetArgs::default(), inputs).unwrap_err();
        assert_eq!(
            policy,
            MemoryPolicyError::MissingObservation {
                pool: "host",
                reason: ObservationUnavailable::UnsupportedPlatform,
            }
        );
        let args = ServiceBudgetArgs {
            host: BudgetRequest::Bytes(nz(32)),
            ..ServiceBudgetArgs::default()
        };
        let policy = resolve_policy(args, inputs).unwrap();
        assert_eq!(policy.pools().host_bytes(), nz(32));
    }

    #[test]
    fn explicit_backend_override_is_checked_against_observed_capacity() {
        let inputs = discrete_inputs(
            BackendCapacityObservation::Discrete {
                total_bytes: 1_000,
                available_bytes: 100,
            },
            10,
            20,
        );
        let args = ServiceBudgetArgs {
            memory: BudgetRequest::Bytes(nz(50)),
            ..ServiceBudgetArgs::default()
        };
        let policy = resolve_policy(args, inputs).unwrap();
        assert_eq!(policy.pools().combined_bytes(), nz(50));
        assert!(policy.overrides().memory);
        assert_eq!(
            policy.effective_backend_budget(20, 0),
            Ok(MemoryBudget::Bytes(nz(50)))
        );
    }

    #[test]
    fn observed_total_caps_auto_and_explicit_limits() {
        let inputs = discrete_inputs(
            BackendCapacityObservation::Discrete {
                total_bytes: 100,
                available_bytes: 60,
            },
            80,
            0,
        );
        let policy = resolve_policy(ServiceBudgetArgs::default(), inputs).unwrap();
        assert_eq!(policy.pools().combined_bytes(), nz(100));
        let args = ServiceBudgetArgs {
            memory: BudgetRequest::Bytes(nz(120)),
            ..ServiceBudgetArgs::default()
        };
        assert_eq!(
            resolve_policy(args, inputs),
            Err(MemoryPolicyError::ExplicitExceedsObserved {
                field: "backend",
                requested: 120,
                maximum: 100,
            })
        );
    }

    #[test]
    fn shared_policy_caps_host_limit_and_allows_guidance_override() {
        let inputs = ServiceMemoryInputs {
            topology: MemoryTopology::Unified,
            backend: BackendCapacityObservation::Recommended {
                working_set_bytes: 500,
                current_allocated_bytes: 10,
                unified_memory: true,
            },
            host: host(1_000),
            backend_owned_bytes: 20,
            host_owned_bytes: 30,
        };
        let auto = resolve_policy(ServiceBudgetArgs::default(), inputs).unwrap();
        assert_eq!(
            auto.pools(),
            ServiceMemoryPools::Shared {
                combined_bytes: nz(400),
                host_bytes: nz(400),
            }
        );
        assert_eq!(
            auto.effective_backend_budget(30, 0),
            Ok(MemoryBudget::Bytes(nz(370)))
        );
        let args = ServiceBudgetArgs {
            memory: BudgetRequest::Bytes(nz(700)),
            ..ServiceBudgetArgs::default()
        };
        let override_policy = resolve_policy(args, inputs).unwrap();
        assert_eq!(override_policy.pools().combined_bytes(), nz(700));
        assert_eq!(override_policy.pools().host_bytes(), nz(700));
    }

    #[test]
    fn shared_policy_builds_one_parent_for_host_and_backend_children() {
        let inputs = ServiceMemoryInputs {
            topology: MemoryTopology::Unified,
            backend: BackendCapacityObservation::Recommended {
                working_set_bytes: 500,
                current_allocated_bytes: 10,
                unified_memory: true,
            },
            host: host(1_000),
            backend_owned_bytes: 20,
            host_owned_bytes: 30,
        };
        let policy = resolve_policy(ServiceBudgetArgs::default(), inputs).unwrap();
        let root = MemoryTrackerRoot::new(policy.shared_budget().unwrap());
        let host = HostMemoryLedger::child(policy.host_budget(), root.clone());
        let device = MemoryTracker::child(
            MemoryBudget::Bytes(policy.pools().combined_bytes()),
            root.clone(),
        );
        let host_allocation = host.allocate(200).unwrap();
        let device_allocation = device.allocate(MemoryClass::ModelWeight, 200).unwrap();
        assert!(matches!(
            host.reserve(1),
            Err(MemoryError::BudgetExceeded { budget: 400, .. })
        ));
        drop(device_allocation);
        drop(host_allocation);
        assert_eq!(root.owned_bytes(), 0);
    }

    #[test]
    fn working_set_guidance_is_advisory_for_explicit_limits() {
        let inputs = ServiceMemoryInputs {
            topology: MemoryTopology::Unified,
            backend: BackendCapacityObservation::Recommended {
                working_set_bytes: 50,
                current_allocated_bytes: 80,
                unified_memory: true,
            },
            host: host(1_000),
            backend_owned_bytes: 10,
            host_owned_bytes: 20,
        };
        let args = ServiceBudgetArgs {
            memory: BudgetRequest::Bytes(nz(100)),
            ..ServiceBudgetArgs::default()
        };
        let policy = resolve_policy(args, inputs).unwrap();
        assert_eq!(policy.pools().combined_bytes(), nz(100));
    }

    #[test]
    fn automatic_guidance_below_owned_bytes_is_rejected() {
        let inputs = ServiceMemoryInputs {
            topology: MemoryTopology::Unified,
            backend: BackendCapacityObservation::Recommended {
                working_set_bytes: 50,
                current_allocated_bytes: 80,
                unified_memory: true,
            },
            host: host(1_000),
            backend_owned_bytes: 50,
            host_owned_bytes: 20,
        };
        assert_eq!(
            resolve_policy(ServiceBudgetArgs::default(), inputs),
            Err(MemoryPolicyError::ResolvedBelowOwned {
                field: "backend",
                limit: 40,
                owned: 70,
            })
        );
    }

    #[test]
    fn cpu_combined_limit_is_independent_of_host_subquota() {
        let inputs = ServiceMemoryInputs {
            topology: MemoryTopology::Cpu,
            backend: BackendCapacityObservation::Cpu,
            host: HostMemoryObservation::Available {
                total_bytes: 1_000,
                available_bytes: 900,
                source: HostMemorySource::LinuxMeminfo,
                semantics: HostMemorySemantics::KernelAvailableEstimate,
            },
            backend_owned_bytes: 100,
            host_owned_bytes: 20,
        };
        let args = ServiceBudgetArgs {
            host: BudgetRequest::Bytes(nz(30)),
            ..ServiceBudgetArgs::default()
        };
        let policy = resolve_policy(args, inputs).unwrap();
        assert_eq!(policy.pools().combined_bytes(), nz(840));
        assert_eq!(policy.pools().host_bytes(), nz(30));
    }

    #[test]
    fn policy_rejects_checked_baseline_overflow() {
        let inputs = discrete_inputs(
            BackendCapacityObservation::Discrete {
                total_bytes: u64::MAX,
                available_bytes: 1,
            },
            u64::MAX,
            0,
        );
        let error = resolve_policy(ServiceBudgetArgs::default(), inputs).unwrap_err();
        assert_eq!(error, MemoryPolicyError::Overflow { field: "backend" });
    }

    #[test]
    fn kv_reservation_cannot_enlarge_physical_capacity() {
        let error = resolve_kv_reservation(BudgetRequest::Bytes(nz(50)), 4, 10, 2, 100, 60, 30)
            .unwrap_err();
        assert_eq!(
            error,
            MemoryPolicyError::KvReservationExceedsLimit {
                requested: 50,
                limit: 10,
            }
        );
    }

    #[test]
    fn kv_reservation_overflow_is_typed() {
        assert_eq!(
            resolve_kv_reservation(BudgetRequest::Auto, u64::MAX, 2, 1, u64::MAX, 0, 0),
            Err(MemoryPolicyError::Overflow {
                field: "KV reservation",
            })
        );
    }

    #[test]
    fn host_copy_denial_rolls_back_pending_bytes() {
        let ledger = HostMemoryLedger::new(nz(64));
        let source = ledger.allocate(40).unwrap();
        let destination = ledger.reserve(24).unwrap();
        assert!(matches!(
            ledger.reserve(1),
            Err(MemoryError::BudgetExceeded { .. })
        ));
        drop(destination);
        assert_eq!(ledger.snapshot().reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_bytes, 40);
        let retry = ledger.allocate(24).unwrap();
        assert_eq!(ledger.snapshot().live_bytes, 64);
        drop(retry);
        drop(source);
        assert_eq!(ledger.snapshot().live_bytes, 0);
    }

    #[test]
    fn host_limit_change_rejects_owned_bytes_atomically() {
        let ledger = HostMemoryLedger::new(nz(64));
        let allocation = ledger.allocate(32).unwrap();
        assert_eq!(
            ledger.set_limit(nz(16)),
            Err(MemoryError::BudgetBelowOwned {
                budget: 16,
                owned: 32,
                reserved: 0,
            })
        );
        assert_eq!(ledger.snapshot().budget, MemoryBudget::Bytes(nz(64)));
        drop(allocation);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_meminfo_parser_never_uses_mapped_bytes() {
        let observation = parse_linux_meminfo(
            "MemTotal:       100 kB\nMemAvailable:    40 kB\nMapped:          99 kB\n",
        );
        assert_eq!(
            observation,
            HostMemoryObservation::Available {
                total_bytes: 102_400,
                available_bytes: 40_960,
                source: HostMemorySource::LinuxMeminfo,
                semantics: HostMemorySemantics::KernelAvailableEstimate,
            }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mac_vm_stat_parser_records_page_size_and_file_backed_estimate() {
        let contents = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 10.\nPages speculative: 3.\nFile-backed pages: 20.\n";
        assert_eq!(parse_vm_stat_page_size(contents), Some(4096));
        assert_eq!(parse_vm_stat(contents), Some((10, Some(20))));
    }
}
