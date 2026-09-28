use std::collections::BTreeMap;
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;

mod host;
mod kv;
pub use host::HostStaging;
pub use kv::{AttentionDecodeRow, KvReadSpan, KvReadView, KvWriteSpan};

const K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q8_KV_BLOCK_ELEMENTS: usize = 32;
const Q8_KV_BLOCK_BYTES: usize = 34;

/// An error returned when a backend contract cannot be satisfied.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BackendError {
    #[error("backend operation {operation} failed: {message}")]
    Operation {
        operation: &'static str,
        message: String,
    },
    #[error("memory {0}")]
    Memory(#[from] MemoryError),
    #[error("{field} must be nonzero")]
    Zero { field: &'static str },
    #[error("{field} must be divisible by {divisor}, found {value}")]
    NotDivisible {
        field: &'static str,
        value: usize,
        divisor: usize,
    },
    #[error("{field} overflows the host size")]
    SizeOverflow { field: &'static str },
    #[error("n_head {n_head} is not divisible by n_head_kv {n_head_kv}")]
    InvalidGqa { n_head: usize, n_head_kv: usize },
    #[error("{name} has {actual} elements, expected {expected}")]
    SizeMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field} must be finite and greater than zero, found {value}")]
    InvalidPositiveFloat { field: &'static str, value: f32 },
    #[error("row {row} is outside a matrix with {rows} rows")]
    RowOutOfBounds { row: usize, rows: usize },
    #[error("position {position} is outside a context with capacity {max_context}")]
    PositionOutOfBounds { position: usize, max_context: usize },
    #[error("position {position} is outside KV span [{start}, {end})")]
    PositionOutsideSpan {
        position: usize,
        start: usize,
        end: usize,
    },
}

/// Selects the maximum bytes that one tracker may own or reserve.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryBudget {
    #[default]
    Unlimited,
    Bytes(NonZeroU64),
}

impl MemoryBudget {
    /// Creates a finite budget and rejects zero.
    pub fn limited(bytes: u64) -> Result<Self, MemoryError> {
        NonZeroU64::new(bytes)
            .map(Self::Bytes)
            .ok_or(MemoryError::ZeroBudget)
    }
}

/// Reports a failed checked memory reservation or budget change.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MemoryError {
    #[error("allocation bytes must be nonzero")]
    ZeroBytes,
    #[error("memory budget must be nonzero")]
    ZeroBudget,
    #[error(
        "allocation of {requested} bytes exceeds budget {budget}; {owned} bytes are owned and {reserved} bytes are reserved"
    )]
    BudgetExceeded {
        requested: u64,
        budget: u64,
        owned: u64,
        reserved: u64,
    },
    #[error("memory budget {budget} is below {owned} owned bytes and {reserved} reserved bytes")]
    BudgetBelowOwned {
        budget: u64,
        owned: u64,
        reserved: u64,
    },
    #[error("memory accounting byte count overflows")]
    ByteOverflow,
    #[error("memory accounting allocation count overflows")]
    AllocationCountOverflow,
    #[error("memory reservation state is invalid")]
    InvalidReservation,
    #[error("memory tracker has {owned} owned and {reserved} reserved bytes")]
    TrackerInUse { owned: u64, reserved: u64 },
}

/// Selects one physical allocation class reported by a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MemoryClass {
    ContractBuffer = 0,
    ModelWeight = 1,
    RepackedWeight = 2,
    Activation = 3,
    KvCache = 4,
    BackendScratch = 5,
    PrefillScratch = 6,
    GraphBuffer = 7,
}

impl MemoryClass {
    /// Lists every class that can appear in a memory snapshot.
    pub const ALL: [Self; 8] = [
        Self::ContractBuffer,
        Self::ModelWeight,
        Self::RepackedWeight,
        Self::Activation,
        Self::KvCache,
        Self::BackendScratch,
        Self::PrefillScratch,
        Self::GraphBuffer,
    ];

    /// Returns the stable receipt name for this class.
    pub const fn name(self) -> &'static str {
        match self {
            Self::ContractBuffer => "contract_buffer",
            Self::ModelWeight => "model_weight",
            Self::RepackedWeight => "repacked_weight",
            Self::Activation => "activation",
            Self::KvCache => "kv_cache",
            Self::BackendScratch => "backend_scratch",
            Self::PrefillScratch => "prefill_scratch",
            Self::GraphBuffer => "graph_buffer",
        }
    }

    fn from_index(index: u8) -> Self {
        match index {
            0 => Self::ContractBuffer,
            1 => Self::ModelWeight,
            2 => Self::RepackedWeight,
            3 => Self::Activation,
            4 => Self::KvCache,
            5 => Self::BackendScratch,
            6 => Self::PrefillScratch,
            7 => Self::GraphBuffer,
            _ => unreachable!("memory class index invariant"),
        }
    }
}

/// Counts allocations attributed to one physical memory class.
///
/// Reclassification transfers an allocation's count and live bytes. Earlier
/// peaks stay recorded in the original class. Frees accrue to the final class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryClassStats {
    pub live_bytes: u64,
    pub peak_live_bytes: u64,
    pub live_allocations: u64,
    pub peak_live_allocations: u64,
    pub allocations: u64,
    pub frees: u64,
}

/// Counts backend objects whose byte size is owned by an external library.
///
/// Their byte sizes stay separate from tracked allocation bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UntrackedMemory {
    pub graph_objects: u64,
    pub library_handles: u64,
    pub execution_streams: u64,
    pub execution_events: u64,
}

impl UntrackedMemory {
    pub fn object_count(self) -> u64 {
        self.graph_objects
            .checked_add(self.library_handles)
            .and_then(|count| count.checked_add(self.execution_streams))
            .and_then(|count| count.checked_add(self.execution_events))
            .expect("untracked object count overflow")
    }
}

/// A physical allocation snapshot with live bytes, high-water marks, and counts.
///
/// `live_bytes` covers allocations made through the backend's checked allocator.
/// `reserved_bytes` covers bytes held by reservations, including backend calls
/// in progress and conservative bounds retained across calls.
/// `peak_owned_and_reserved_bytes` records the highest admitted sum of live and
/// reserved bytes for this ownership scope. It measures accounting, not RSS.
/// `budget` bounds `live_bytes + reserved_bytes`.
/// External library objects are reported in `untracked` when their byte sizes
/// are not available through the backend contract.
/// Counters follow ownership release. They cannot confirm device cleanup after
/// an external library or driver rejects a destructor operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryAccounting {
    pub live_bytes: u64,
    pub reserved_bytes: u64,
    pub peak_live_bytes: u64,
    pub peak_owned_and_reserved_bytes: u64,
    pub live_allocations: u64,
    pub peak_live_allocations: u64,
    pub allocations: u64,
    pub frees: u64,
    pub classes: BTreeMap<MemoryClass, MemoryClassStats>,
    pub untracked: UntrackedMemory,
    pub budget: MemoryBudget,
}

pub(crate) fn reserve_rope_host_bytes(
    staging: &HostStaging,
    head_dim: usize,
) -> Result<Option<MemoryReservation>, BackendError> {
    let pairs = head_dim / 2;
    if pairs == 0 {
        return Ok(None);
    }
    let pairs = u64::try_from(pairs).map_err(|_| BackendError::SizeOverflow {
        field: "RoPE inverse frequency count",
    })?;
    let element_bytes =
        u64::try_from(std::mem::size_of::<f64>()).map_err(|_| BackendError::SizeOverflow {
            field: "RoPE inverse frequency element size",
        })?;
    let bytes = pairs
        .checked_mul(element_bytes)
        .ok_or(BackendError::SizeOverflow {
            field: "RoPE inverse frequency bytes",
        })?;
    staging.reserve(bytes).map(Some).map_err(BackendError::from)
}

impl Default for MemoryAccounting {
    fn default() -> Self {
        Self {
            live_bytes: 0,
            reserved_bytes: 0,
            peak_live_bytes: 0,
            peak_owned_and_reserved_bytes: 0,
            live_allocations: 0,
            peak_live_allocations: 0,
            allocations: 0,
            frees: 0,
            classes: MemoryClass::ALL
                .into_iter()
                .map(|class| (class, MemoryClassStats::default()))
                .collect(),
            untracked: UntrackedMemory::default(),
            budget: MemoryBudget::Unlimited,
        }
    }
}

impl MemoryAccounting {
    /// Returns the snapshot for one allocation class.
    pub fn class(&self, class: MemoryClass) -> MemoryClassStats {
        self.classes.get(&class).copied().unwrap_or_default()
    }

    /// Adds counts for external-library objects without assigning them byte sizes.
    pub fn with_untracked(mut self, untracked: UntrackedMemory) -> Self {
        self.untracked = untracked;
        self
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct MemoryCounters {
    live_bytes: u64,
    peak_live_bytes: u64,
    live_allocations: u64,
    peak_live_allocations: u64,
    allocations: u64,
    frees: u64,
}

impl MemoryCounters {
    fn snapshot(self) -> MemoryClassStats {
        MemoryClassStats {
            live_bytes: self.live_bytes,
            peak_live_bytes: self.peak_live_bytes,
            live_allocations: self.live_allocations,
            peak_live_allocations: self.peak_live_allocations,
            allocations: self.allocations,
            frees: self.frees,
        }
    }
}

#[derive(Debug, Default)]
struct MemoryRootState {
    total: MemoryCounters,
    budget: MemoryBudget,
    reserved_bytes: u64,
    reserved_allocations: u64,
    peak_owned_and_reserved_bytes: u64,
}

/// Owns the physical byte ceiling shared by sibling trackers.
#[derive(Debug, Clone)]
pub struct MemoryTrackerRoot {
    state: Arc<Mutex<MemoryRootState>>,
}

impl Default for MemoryTrackerRoot {
    fn default() -> Self {
        Self::new(MemoryBudget::Unlimited)
    }
}

impl MemoryTrackerRoot {
    /// Creates a parent with the supplied combined owned-byte budget.
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            state: Arc::new(Mutex::new(MemoryRootState {
                budget,
                ..MemoryRootState::default()
            })),
        }
    }

    /// Returns the combined parent budget.
    pub fn budget(&self) -> MemoryBudget {
        lock_tracker(&self.state).budget
    }

    /// Changes the parent budget when every child allocation and reservation fits.
    pub fn set_budget(&self, budget: MemoryBudget) -> Result<(), MemoryError> {
        let mut state = lock_tracker(&self.state);
        check_budget_change(budget, state.total.live_bytes, state.reserved_bytes)?;
        state.budget = budget;
        Ok(())
    }

    /// Returns bytes owned by all child trackers.
    pub fn owned_bytes(&self) -> u64 {
        lock_tracker(&self.state).total.live_bytes
    }

    /// Returns bytes reserved by all child trackers.
    pub fn reserved_bytes(&self) -> u64 {
        lock_tracker(&self.state).reserved_bytes
    }

    /// Returns combined parent accounting without child class attribution.
    pub fn snapshot(&self) -> MemoryAccounting {
        root_snapshot(&lock_tracker(&self.state))
    }
}

#[derive(Debug, Default)]
struct MemoryTrackerState {
    total: MemoryCounters,
    classes: BTreeMap<MemoryClass, MemoryCounters>,
    reserved_classes: BTreeMap<MemoryClass, u64>,
    budget: MemoryBudget,
    reserved_bytes: u64,
    reserved_allocations: u64,
    peak_owned_and_reserved_bytes: u64,
}

/// Tracks exact bytes owned by one backend or host child allocation pool.
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    state: Arc<Mutex<MemoryTrackerState>>,
    shared: MemoryTrackerRoot,
    root_owner: bool,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new(MemoryBudget::Unlimited)
    }
}

impl MemoryTracker {
    /// Creates an independent tracker with the supplied owned-byte budget.
    pub fn new(budget: MemoryBudget) -> Self {
        Self::from_parts(budget, MemoryTrackerRoot::new(budget), true)
    }

    /// Creates a child quota that charges the supplied shared parent.
    pub fn child(budget: MemoryBudget, shared: MemoryTrackerRoot) -> Self {
        Self::from_parts(budget, shared, false)
    }

    fn from_parts(budget: MemoryBudget, shared: MemoryTrackerRoot, root_owner: bool) -> Self {
        Self {
            state: Arc::new(Mutex::new(MemoryTrackerState {
                budget,
                ..MemoryTrackerState::default()
            })),
            shared,
            root_owner,
        }
    }

    /// Returns the shared parent used by this tracker.
    pub fn root(&self) -> MemoryTrackerRoot {
        self.shared.clone()
    }

    /// Returns the tracker budget.
    pub fn budget(&self) -> MemoryBudget {
        if self.root_owner {
            lock_tracker(&self.shared.state).budget
        } else {
            lock_tracker(&self.state).budget
        }
    }

    /// Changes this child quota when all owned and pending bytes fit.
    pub fn set_budget(&self, budget: MemoryBudget) -> Result<(), MemoryError> {
        let mut shared = lock_tracker(&self.shared.state);
        let mut state = lock_tracker(&self.state);
        check_budget_change(budget, state.total.live_bytes, state.reserved_bytes)?;
        if self.root_owner {
            check_budget_change(budget, shared.total.live_bytes, shared.reserved_bytes)?;
            shared.budget = budget;
        }
        state.budget = budget;
        Ok(())
    }

    /// Returns true when this child owns and reserves no bytes.
    pub fn is_empty(&self) -> bool {
        let state = lock_tracker(&self.state);
        state.total.live_bytes == 0 && state.reserved_bytes == 0
    }

    /// Reserves bytes before a backend performs its physical allocation.
    pub fn reserve(
        &self,
        class: MemoryClass,
        bytes: u64,
    ) -> Result<MemoryReservation, MemoryError> {
        if bytes == 0 {
            return Err(MemoryError::ZeroBytes);
        }
        let mut shared = lock_tracker(&self.shared.state);
        let mut state = lock_tracker(&self.state);
        let (shared_reserved_bytes, shared_reserved_allocations, shared_accounted) =
            checked_shared_reservation(&shared, bytes)?;
        let budget = if self.root_owner {
            shared.budget
        } else {
            state.budget
        };
        let (reserved_bytes, reserved_allocations, class_reservations, accounted) =
            checked_reservation(&state, budget, class, bytes)?;
        if self.root_owner {
            state.budget = shared.budget;
        }
        shared.reserved_bytes = shared_reserved_bytes;
        shared.reserved_allocations = shared_reserved_allocations;
        shared.peak_owned_and_reserved_bytes =
            shared.peak_owned_and_reserved_bytes.max(shared_accounted);
        state.reserved_bytes = reserved_bytes;
        state.reserved_allocations = reserved_allocations;
        state.reserved_classes.insert(class, class_reservations);
        state.peak_owned_and_reserved_bytes = state.peak_owned_and_reserved_bytes.max(accounted);
        Ok(MemoryReservation {
            tracker: self.clone(),
            class,
            bytes,
            committed: false,
        })
    }

    /// Records one allocation through the checked reservation path.
    pub fn allocate(
        &self,
        class: MemoryClass,
        bytes: u64,
    ) -> Result<MemoryAllocation, MemoryError> {
        self.reserve(class, bytes)?.commit()
    }

    /// Returns the bytes currently owned by allocations in this child.
    pub fn owned_bytes(&self) -> u64 {
        lock_tracker(&self.state).total.live_bytes
    }

    /// Returns bytes reserved by allocations in this child.
    pub fn reserved_bytes(&self) -> u64 {
        lock_tracker(&self.state).reserved_bytes
    }

    /// Returns a snapshot of this child's allocation classes.
    pub fn snapshot(&self) -> MemoryAccounting {
        if self.root_owner {
            let shared = lock_tracker(&self.shared.state);
            let state = lock_tracker(&self.state);
            let mut snapshot = tracker_snapshot(&state);
            snapshot.budget = shared.budget;
            snapshot
        } else {
            let state = lock_tracker(&self.state);
            tracker_snapshot(&state)
        }
    }
}

/// Holds a checked reservation until a physical allocation succeeds.
#[must_use]
pub struct MemoryReservation {
    tracker: MemoryTracker,
    class: MemoryClass,
    bytes: u64,
    committed: bool,
}

impl fmt::Debug for MemoryReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryReservation")
            .field("bytes", &self.bytes)
            .field("class", &self.class)
            .field("committed", &self.committed)
            .finish()
    }
}

impl MemoryReservation {
    /// Returns the reserved physical byte count.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Commits the reservation as one owned allocation.
    pub fn commit(mut self) -> Result<MemoryAllocation, MemoryError> {
        let mut shared = lock_tracker(&self.tracker.shared.state);
        let mut state = lock_tracker(&self.tracker.state);
        let shared_other_reservations = checked_shared_commit(&shared, self.bytes)?;
        let class_reservations = state
            .reserved_classes
            .get(&self.class)
            .copied()
            .unwrap_or(0);
        if state.reserved_bytes < self.bytes
            || state.reserved_allocations == 0
            || class_reservations == 0
        {
            return Err(MemoryError::InvalidReservation);
        }
        let other_reservations = state.reserved_allocations - 1;
        checked_counter_capacity(&state.total, self.bytes, other_reservations)?;
        let class_counters = state.classes.get(&self.class).copied().unwrap_or_default();
        checked_counter_capacity(&class_counters, self.bytes, class_reservations - 1)?;
        shared.reserved_bytes -= self.bytes;
        shared.reserved_allocations = shared_other_reservations;
        record_allocate(&mut shared.total, self.bytes);
        state.reserved_bytes -= self.bytes;
        state.reserved_allocations -= 1;
        decrement_reserved_class(&mut state.reserved_classes, self.class);
        record_allocate(&mut state.total, self.bytes);
        record_allocate(state.classes.entry(self.class).or_default(), self.bytes);
        let record = AllocationRecord {
            identity: state.total.allocations,
            tracker: self.tracker.clone(),
            class: AtomicU8::new(self.class as u8),
            bytes: self.bytes,
        };
        self.committed = true;
        Ok(MemoryAllocation { record })
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let mut shared = lock_tracker(&self.tracker.shared.state);
        let mut state = lock_tracker(&self.tracker.state);
        shared.reserved_bytes = shared
            .reserved_bytes
            .checked_sub(self.bytes)
            .expect("shared memory reservation byte underflow");
        shared.reserved_allocations = shared
            .reserved_allocations
            .checked_sub(1)
            .expect("shared memory reservation count underflow");
        state.reserved_bytes = state
            .reserved_bytes
            .checked_sub(self.bytes)
            .expect("memory reservation byte underflow");
        state.reserved_allocations = state
            .reserved_allocations
            .checked_sub(1)
            .expect("memory reservation count underflow");
        decrement_reserved_class(&mut state.reserved_classes, self.class);
    }
}

/// Drop-tracked ownership of one backend allocation.
pub struct MemoryAllocation {
    record: AllocationRecord,
}

#[derive(Debug)]
struct AllocationRecord {
    identity: u64,
    tracker: MemoryTracker,
    class: AtomicU8,
    bytes: u64,
}

impl fmt::Debug for MemoryAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryAllocation")
            .field("bytes", &self.record.bytes)
            .field("class", &self.class())
            .finish()
    }
}

impl MemoryAllocation {
    /// Returns an identity that is never reused within the allocation tracker.
    pub fn identity(&self) -> u64 {
        self.record.identity
    }

    /// Returns the exact physical byte count assigned to this allocation.
    pub fn bytes(&self) -> u64 {
        self.record.bytes
    }

    /// Returns the current allocation class.
    pub fn class(&self) -> MemoryClass {
        MemoryClass::from_index(self.record.class.load(Ordering::Acquire))
    }

    /// Moves live bytes to another class without changing allocation identity.
    pub fn reclassify(&self, class: MemoryClass) {
        let _shared = lock_tracker(&self.record.tracker.shared.state);
        let mut state = lock_tracker(&self.record.tracker.state);
        let previous = MemoryClass::from_index(self.record.class.load(Ordering::Acquire));
        if previous == class {
            return;
        }
        self.record.class.store(class as u8, Ordering::Release);
        move_live(&mut state, previous, class, self.record.bytes);
    }

    /// Reserves an independent allocation with the same class and byte count.
    pub fn reserve_duplicate(&self) -> Result<MemoryReservation, MemoryError> {
        self.record.tracker.reserve(self.class(), self.record.bytes)
    }

    /// Creates an independent allocation through the checked reservation path.
    pub fn duplicate(&self) -> Result<Self, MemoryError> {
        self.record
            .tracker
            .allocate(self.class(), self.record.bytes)
    }
}

impl Drop for MemoryAllocation {
    fn drop(&mut self) {
        let mut shared = lock_tracker(&self.record.tracker.shared.state);
        let mut state = lock_tracker(&self.record.tracker.state);
        let class = MemoryClass::from_index(self.record.class.load(Ordering::Acquire));
        record_free(&mut state.total, self.record.bytes);
        record_free(state.classes.entry(class).or_default(), self.record.bytes);
        record_free(&mut shared.total, self.record.bytes);
    }
}

fn lock_tracker<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn check_budget_change(budget: MemoryBudget, owned: u64, reserved: u64) -> Result<(), MemoryError> {
    let accounted = owned
        .checked_add(reserved)
        .ok_or(MemoryError::ByteOverflow)?;
    if let MemoryBudget::Bytes(limit) = budget {
        if accounted > limit.get() {
            return Err(MemoryError::BudgetBelowOwned {
                budget: limit.get(),
                owned,
                reserved,
            });
        }
    }
    Ok(())
}

fn root_snapshot(state: &MemoryRootState) -> MemoryAccounting {
    let mut snapshot = MemoryAccounting {
        live_bytes: state.total.live_bytes,
        reserved_bytes: state.reserved_bytes,
        peak_live_bytes: state.total.peak_live_bytes,
        peak_owned_and_reserved_bytes: state.peak_owned_and_reserved_bytes,
        live_allocations: state.total.live_allocations,
        peak_live_allocations: state.total.peak_live_allocations,
        allocations: state.total.allocations,
        frees: state.total.frees,
        budget: state.budget,
        ..MemoryAccounting::default()
    };
    snapshot.classes = MemoryClass::ALL
        .into_iter()
        .map(|class| (class, MemoryClassStats::default()))
        .collect();
    snapshot
}

fn tracker_snapshot(state: &MemoryTrackerState) -> MemoryAccounting {
    let classes = MemoryClass::ALL
        .into_iter()
        .map(|class| {
            (
                class,
                state
                    .classes
                    .get(&class)
                    .copied()
                    .unwrap_or_default()
                    .snapshot(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    MemoryAccounting {
        live_bytes: state.total.live_bytes,
        reserved_bytes: state.reserved_bytes,
        peak_live_bytes: state.total.peak_live_bytes,
        peak_owned_and_reserved_bytes: state.peak_owned_and_reserved_bytes,
        live_allocations: state.total.live_allocations,
        peak_live_allocations: state.total.peak_live_allocations,
        allocations: state.total.allocations,
        frees: state.total.frees,
        classes,
        untracked: UntrackedMemory::default(),
        budget: state.budget,
    }
}

fn checked_shared_reservation(
    state: &MemoryRootState,
    bytes: u64,
) -> Result<(u64, u64, u64), MemoryError> {
    let accounted = state
        .total
        .live_bytes
        .checked_add(state.reserved_bytes)
        .and_then(|value| value.checked_add(bytes))
        .ok_or(MemoryError::ByteOverflow)?;
    if let MemoryBudget::Bytes(limit) = state.budget {
        if accounted > limit.get() {
            return Err(MemoryError::BudgetExceeded {
                requested: bytes,
                budget: limit.get(),
                owned: state.total.live_bytes,
                reserved: state.reserved_bytes,
            });
        }
    }
    checked_counter_capacity(&state.total, bytes, state.reserved_allocations)?;
    let reserved_bytes = state
        .reserved_bytes
        .checked_add(bytes)
        .ok_or(MemoryError::ByteOverflow)?;
    let reserved_allocations = state
        .reserved_allocations
        .checked_add(1)
        .ok_or(MemoryError::AllocationCountOverflow)?;
    Ok((reserved_bytes, reserved_allocations, accounted))
}

fn checked_shared_commit(state: &MemoryRootState, bytes: u64) -> Result<u64, MemoryError> {
    if state.reserved_bytes < bytes || state.reserved_allocations == 0 {
        return Err(MemoryError::InvalidReservation);
    }
    let other_reservations = state.reserved_allocations - 1;
    checked_counter_capacity(&state.total, bytes, other_reservations)?;
    Ok(other_reservations)
}

fn checked_counter_capacity(
    counters: &MemoryCounters,
    bytes: u64,
    pending_allocations: u64,
) -> Result<(), MemoryError> {
    counters
        .live_bytes
        .checked_add(bytes)
        .ok_or(MemoryError::ByteOverflow)?;
    counters
        .live_allocations
        .checked_add(pending_allocations)
        .and_then(|value| value.checked_add(1))
        .ok_or(MemoryError::AllocationCountOverflow)?;
    counters
        .allocations
        .checked_add(pending_allocations)
        .and_then(|value| value.checked_add(1))
        .ok_or(MemoryError::AllocationCountOverflow)?;
    Ok(())
}

fn checked_reservation(
    state: &MemoryTrackerState,
    budget: MemoryBudget,
    class: MemoryClass,
    bytes: u64,
) -> Result<(u64, u64, u64, u64), MemoryError> {
    let accounted = state
        .total
        .live_bytes
        .checked_add(state.reserved_bytes)
        .and_then(|value| value.checked_add(bytes))
        .ok_or(MemoryError::ByteOverflow)?;
    if let MemoryBudget::Bytes(limit) = budget {
        let limit = limit.get();
        if accounted > limit {
            return Err(MemoryError::BudgetExceeded {
                requested: bytes,
                budget: limit,
                owned: state.total.live_bytes,
                reserved: state.reserved_bytes,
            });
        }
    }
    checked_counter_capacity(&state.total, bytes, state.reserved_allocations)?;
    let class_reservations = state.reserved_classes.get(&class).copied().unwrap_or(0);
    let class_counters = state.classes.get(&class).copied().unwrap_or_default();
    checked_counter_capacity(&class_counters, bytes, class_reservations)?;
    let reserved_bytes = state
        .reserved_bytes
        .checked_add(bytes)
        .ok_or(MemoryError::ByteOverflow)?;
    let reserved_allocations = state
        .reserved_allocations
        .checked_add(1)
        .ok_or(MemoryError::AllocationCountOverflow)?;
    let class_reservations = class_reservations
        .checked_add(1)
        .ok_or(MemoryError::AllocationCountOverflow)?;
    Ok((
        reserved_bytes,
        reserved_allocations,
        class_reservations,
        accounted,
    ))
}

fn decrement_reserved_class(classes: &mut BTreeMap<MemoryClass, u64>, class: MemoryClass) {
    let count = classes
        .get_mut(&class)
        .expect("reserved memory class invariant");
    *count = count
        .checked_sub(1)
        .expect("reserved memory class underflow");
    if *count == 0 {
        classes.remove(&class);
    }
}

fn record_allocate(counters: &mut MemoryCounters, bytes: u64) {
    counters.live_bytes = counters
        .live_bytes
        .checked_add(bytes)
        .expect("memory accounting byte overflow");
    counters.live_allocations = counters
        .live_allocations
        .checked_add(1)
        .expect("memory accounting allocation overflow");
    counters.allocations = counters
        .allocations
        .checked_add(1)
        .expect("memory accounting allocation overflow");
    counters.peak_live_bytes = counters.peak_live_bytes.max(counters.live_bytes);
    counters.peak_live_allocations = counters
        .peak_live_allocations
        .max(counters.live_allocations);
}

fn record_free(counters: &mut MemoryCounters, bytes: u64) {
    counters.live_bytes = counters
        .live_bytes
        .checked_sub(bytes)
        .expect("memory accounting byte underflow");
    counters.live_allocations = counters
        .live_allocations
        .checked_sub(1)
        .expect("memory accounting allocation underflow");
    counters.frees = counters
        .frees
        .checked_add(1)
        .expect("memory accounting free overflow");
}

fn move_live(state: &mut MemoryTrackerState, previous: MemoryClass, next: MemoryClass, bytes: u64) {
    let previous_counters = state.classes.entry(previous).or_default();
    previous_counters.live_bytes = previous_counters
        .live_bytes
        .checked_sub(bytes)
        .expect("memory accounting class byte underflow");
    previous_counters.live_allocations = previous_counters
        .live_allocations
        .checked_sub(1)
        .expect("memory accounting class allocation underflow");
    previous_counters.allocations = previous_counters
        .allocations
        .checked_sub(1)
        .expect("memory accounting class allocation underflow");
    let next_counters = state.classes.entry(next).or_default();
    next_counters.live_bytes = next_counters
        .live_bytes
        .checked_add(bytes)
        .expect("memory accounting class byte overflow");
    next_counters.live_allocations = next_counters
        .live_allocations
        .checked_add(1)
        .expect("memory accounting class allocation overflow");
    next_counters.allocations = next_counters
        .allocations
        .checked_add(1)
        .expect("memory accounting class allocation overflow");
    next_counters.peak_live_bytes = next_counters.peak_live_bytes.max(next_counters.live_bytes);
    next_counters.peak_live_allocations = next_counters
        .peak_live_allocations
        .max(next_counters.live_allocations);
}

impl BackendError {
    /// Wraps an implementation error without exposing its concrete type.
    pub fn operation(operation: &'static str, error: impl fmt::Display) -> Self {
        Self::Operation {
            operation,
            message: error.to_string(),
        }
    }
}

/// The scalar or quantized storage held by an opaque backend buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferStorage {
    F16,
    F32,
    U32,
    Q8Kv,
    Q4K,
    Q6K,
}

/// A checked buffer layout with logical element and physical byte counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferLayout {
    storage: BufferStorage,
    elements: usize,
    bytes: usize,
}

/// Exact physical bytes and layout for one portable backend buffer snapshot.
///
/// The host-owned bytes are outside `MemoryAccounting` and `MemoryBudget` until restored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSnapshot {
    layout: BufferLayout,
    bytes: Vec<u8>,
}

impl BufferSnapshot {
    /// Creates a snapshot only when its physical byte count matches the layout.
    pub fn new(layout: BufferLayout, bytes: Vec<u8>) -> Result<Self, BackendError> {
        if bytes.len() != layout.bytes() {
            return Err(BackendError::SizeMismatch {
                name: "snapshot bytes",
                expected: layout.bytes(),
                actual: bytes.len(),
            });
        }
        Ok(Self { layout, bytes })
    }

    pub const fn layout(&self) -> BufferLayout {
        self.layout
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl BufferLayout {
    /// Checks a dense f16 buffer.
    pub fn f16(elements: usize) -> Result<Self, BackendError> {
        Self::dense(BufferStorage::F16, elements, 2)
    }

    /// Checks a dense f32 buffer.
    pub fn f32(elements: usize) -> Result<Self, BackendError> {
        Self::dense(BufferStorage::F32, elements, 4)
    }

    /// Checks a dense u32 buffer.
    pub fn u32(elements: usize) -> Result<Self, BackendError> {
        Self::dense(BufferStorage::U32, elements, 4)
    }

    /// Checks q8_0 KV storage with one FP16 scale per 32 values.
    pub fn q8_kv(elements: usize) -> Result<Self, BackendError> {
        nonzero("Q8 KV elements", elements)?;
        divisible("Q8 KV elements", elements, Q8_KV_BLOCK_ELEMENTS)?;
        let bytes = (elements / Q8_KV_BLOCK_ELEMENTS)
            .checked_mul(Q8_KV_BLOCK_BYTES)
            .ok_or(BackendError::SizeOverflow {
                field: "Q8 KV buffer bytes",
            })?;
        Ok(Self {
            storage: BufferStorage::Q8Kv,
            elements,
            bytes,
        })
    }

    /// Checks a K-quant buffer with complete 256-value blocks.
    pub fn quantized(elements: usize, format: QuantFormat) -> Result<Self, BackendError> {
        nonzero("quantized elements", elements)?;
        divisible("quantized elements", elements, K_BLOCK_ELEMENTS)?;
        let blocks = elements / K_BLOCK_ELEMENTS;
        let bytes = blocks
            .checked_mul(format.block_bytes())
            .ok_or(BackendError::SizeOverflow {
                field: "quantized buffer bytes",
            })?;
        Ok(Self {
            storage: format.storage(),
            elements,
            bytes,
        })
    }

    fn dense(
        storage: BufferStorage,
        elements: usize,
        element_bytes: usize,
    ) -> Result<Self, BackendError> {
        nonzero("buffer elements", elements)?;
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or(BackendError::SizeOverflow {
                field: "dense buffer bytes",
            })?;
        Ok(Self {
            storage,
            elements,
            bytes,
        })
    }

    pub const fn storage(self) -> BufferStorage {
        self.storage
    }

    pub const fn elements(self) -> usize {
        self.elements
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// A K-quant format supported by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantFormat {
    Q4K,
    Q6K,
}

impl QuantFormat {
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => Q4_K_BLOCK_BYTES,
            Self::Q6K => Q6_K_BLOCK_BYTES,
        }
    }

    pub const fn storage(self) -> BufferStorage {
        match self {
            Self::Q4K => BufferStorage::Q4K,
            Self::Q6K => BufferStorage::Q6K,
        }
    }
}

/// A checked row-major quantized matrix shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuantMatrix {
    rows: usize,
    columns: usize,
    format: QuantFormat,
}

impl QuantMatrix {
    /// Checks `[rows][columns]` storage with complete row blocks.
    pub fn new(rows: usize, columns: usize, format: QuantFormat) -> Result<Self, BackendError> {
        nonzero("matrix rows", rows)?;
        nonzero("matrix columns", columns)?;
        divisible("matrix columns", columns, K_BLOCK_ELEMENTS)?;
        rows.checked_mul(columns)
            .ok_or(BackendError::SizeOverflow {
                field: "matrix elements",
            })?;
        Ok(Self {
            rows,
            columns,
            format,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> usize {
        self.columns
    }

    pub const fn format(self) -> QuantFormat {
        self.format
    }

    pub fn layout(self) -> Result<BufferLayout, BackendError> {
        BufferLayout::quantized(
            self.rows
                .checked_mul(self.columns)
                .ok_or(BackendError::SizeOverflow {
                    field: "matrix elements",
                })?,
            self.format,
        )
    }

    pub fn row_bytes(self) -> Result<usize, BackendError> {
        (self.columns / K_BLOCK_ELEMENTS)
            .checked_mul(self.format.block_bytes())
            .ok_or(BackendError::SizeOverflow {
                field: "matrix row bytes",
            })
    }
}

/// A checked dense row shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorShape {
    rows: usize,
    columns: usize,
}

impl VectorShape {
    /// Checks `[rows][columns]` dense storage.
    pub fn new(rows: usize, columns: usize) -> Result<Self, BackendError> {
        nonzero("vector rows", rows)?;
        nonzero("vector columns", columns)?;
        rows.checked_mul(columns)
            .ok_or(BackendError::SizeOverflow {
                field: "vector elements",
            })?;
        Ok(Self { rows, columns })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> usize {
        self.columns
    }

    pub fn elements(self) -> Result<usize, BackendError> {
        self.rows
            .checked_mul(self.columns)
            .ok_or(BackendError::SizeOverflow {
                field: "vector elements",
            })
    }
}

/// A checked GPT-NeoX RoPE shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RopeShape {
    tokens: usize,
    heads: usize,
    head_dim: usize,
}

/// Selects how RoPE coordinates form complex pairs within one head.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RopePairing {
    /// Pairs coordinate `i` with `i + head_dim / 2`.
    #[default]
    HalfSplit,
    /// Pairs coordinates `2i` and `2i + 1`.
    Adjacent,
}

impl RopeShape {
    /// Checks `[tokens][heads][head_dim]` half-pair storage.
    pub fn new(tokens: usize, heads: usize, head_dim: usize) -> Result<Self, BackendError> {
        nonzero("RoPE tokens", tokens)?;
        nonzero("RoPE heads", heads)?;
        nonzero("RoPE head_dim", head_dim)?;
        divisible("RoPE head_dim", head_dim, 2)?;
        tokens
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or(BackendError::SizeOverflow {
                field: "RoPE elements",
            })?;
        Ok(Self {
            tokens,
            heads,
            head_dim,
        })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn heads(self) -> usize {
        self.heads
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub fn elements(self) -> Result<usize, BackendError> {
        self.tokens
            .checked_mul(self.heads)
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or(BackendError::SizeOverflow {
                field: "RoPE elements",
            })
    }
}

/// A checked batch-1 GQA attention and KV layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttentionShape {
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    max_context: usize,
}

/// The implementation selected for prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillMethod {
    /// Evaluates one prompt token through the decode path at a time.
    SequentialDecode,
    /// Evaluates position blocks through native GPU kernels.
    ChunkedGpu,
    /// Evaluates position blocks through FP16 cuBLASLt matrix products.
    ChunkedCublasLtFp16,
    /// Evaluates large position blocks through bounded cuBLASLt attention tiles.
    TiledCublasLtFp16,
    /// Reuses retained prompt state without evaluating a prefill segment.
    Reused,
}

impl PrefillMethod {
    /// Returns the stable receipt value for this method.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SequentialDecode => "sequential-decode",
            Self::ChunkedGpu => "chunked-gpu",
            Self::ChunkedCublasLtFp16 => "chunked-cublaslt-fp16",
            Self::TiledCublasLtFp16 => "tiled-cublaslt-fp16",
            Self::Reused => "reused",
        }
    }
}

/// Selects the numerical contract for chunked prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillNumerics {
    /// Uses the backend's default prefill arithmetic.
    BackendPreferred,
    /// Requires bitwise equivalence with repeated decode on supported device,
    /// build, model, KV, and operator shapes.
    DecodeEquivalent,
}

/// A checked workspace plan for one chunked prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    chunk_tokens: usize,
    context_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_embd: usize,
    n_ff: usize,
    max_matrix_rows: usize,
    numerics: PrefillNumerics,
}

impl PrefillPlan {
    /// Checks the largest position block and layer dimensions used by prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_tokens: usize,
        context_tokens: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        n_embd: usize,
        n_ff: usize,
        max_matrix_rows: usize,
    ) -> Result<Self, BackendError> {
        validate_prefill_nonzero(
            chunk_tokens,
            context_tokens,
            n_head,
            n_head_kv,
            head_dim,
            n_embd,
            n_ff,
            max_matrix_rows,
        )?;
        validate_prefill_shapes(
            chunk_tokens,
            context_tokens,
            n_head,
            n_head_kv,
            n_embd,
            n_ff,
            max_matrix_rows,
        )?;
        Ok(Self {
            chunk_tokens,
            context_tokens,
            n_head,
            n_head_kv,
            head_dim,
            n_embd,
            n_ff,
            max_matrix_rows,
            numerics: PrefillNumerics::BackendPreferred,
        })
    }

    /// Sets the numerical contract for this prefill plan.
    ///
    /// The caller must match this contract to the KV cache dtype because the
    /// plan does not store that dtype.
    pub const fn with_numerics(mut self, numerics: PrefillNumerics) -> Self {
        self.numerics = numerics;
        self
    }

    pub const fn chunk_tokens(self) -> usize {
        self.chunk_tokens
    }

    pub const fn context_tokens(self) -> usize {
        self.context_tokens
    }

    pub const fn n_head(self) -> usize {
        self.n_head
    }

    pub const fn n_head_kv(self) -> usize {
        self.n_head_kv
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn n_embd(self) -> usize {
        self.n_embd
    }

    pub const fn n_ff(self) -> usize {
        self.n_ff
    }

    pub const fn max_matrix_rows(self) -> usize {
        self.max_matrix_rows
    }

    /// Returns the numerical contract for this prefill plan.
    pub const fn numerics(self) -> PrefillNumerics {
        self.numerics
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_prefill_nonzero(
    chunk_tokens: usize,
    context_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_embd: usize,
    n_ff: usize,
    max_matrix_rows: usize,
) -> Result<(), BackendError> {
    nonzero("prefill chunk tokens", chunk_tokens)?;
    nonzero("prefill context tokens", context_tokens)?;
    nonzero("prefill n_head", n_head)?;
    nonzero("prefill n_head_kv", n_head_kv)?;
    nonzero("prefill head_dim", head_dim)?;
    nonzero("prefill n_embd", n_embd)?;
    nonzero("prefill n_ff", n_ff)?;
    nonzero("prefill max matrix rows", max_matrix_rows)
}

#[allow(clippy::too_many_arguments)]
fn validate_prefill_shapes(
    chunk_tokens: usize,
    context_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    n_embd: usize,
    n_ff: usize,
    max_matrix_rows: usize,
) -> Result<(), BackendError> {
    if chunk_tokens > context_tokens {
        return Err(BackendError::SizeMismatch {
            name: "prefill chunk context",
            expected: context_tokens,
            actual: chunk_tokens,
        });
    }
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(BackendError::InvalidGqa { n_head, n_head_kv });
    }
    n_embd
        .checked_mul(max_matrix_rows)
        .ok_or(BackendError::SizeOverflow {
            field: "prefill weight elements",
        })?;
    chunk_tokens
        .checked_mul(n_ff.max(n_embd))
        .ok_or(BackendError::SizeOverflow {
            field: "prefill activation elements",
        })?;
    n_head
        .checked_mul(chunk_tokens)
        .and_then(|value| value.checked_mul(context_tokens))
        .ok_or(BackendError::SizeOverflow {
            field: "prefill attention elements",
        })?;
    Ok(())
}

/// Device scratch reserved for one chunked prefill plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillWorkspace {
    pub dequantized_weight_bytes: u64,
    pub converted_activation_bytes: u64,
    pub attention_bytes: u64,
    pub cublaslt_bytes: u64,
    pub batch_activation_bytes: u64,
    pub total_bytes: u64,
}

impl AttentionShape {
    /// Checks Q `[n_head][head_dim]` and KV `[n_head_kv][max_context][head_dim]`.
    pub fn new(
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        nonzero("n_head", n_head)?;
        nonzero("n_head_kv", n_head_kv)?;
        nonzero("head_dim", head_dim)?;
        nonzero("max_context", max_context)?;
        if !n_head.is_multiple_of(n_head_kv) {
            return Err(BackendError::InvalidGqa { n_head, n_head_kv });
        }
        n_head
            .checked_mul(head_dim)
            .ok_or(BackendError::SizeOverflow {
                field: "query elements",
            })?;
        n_head_kv
            .checked_mul(max_context)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or(BackendError::SizeOverflow {
                field: "KV elements",
            })?;
        Ok(Self {
            n_head,
            n_head_kv,
            head_dim,
            max_context,
        })
    }

    pub const fn n_head(self) -> usize {
        self.n_head
    }

    pub const fn n_head_kv(self) -> usize {
        self.n_head_kv
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn max_context(self) -> usize {
        self.max_context
    }

    pub fn query_elements(self) -> Result<usize, BackendError> {
        self.n_head
            .checked_mul(self.head_dim)
            .ok_or(BackendError::SizeOverflow {
                field: "query elements",
            })
    }

    pub fn projected_kv_elements(self) -> Result<usize, BackendError> {
        self.n_head_kv
            .checked_mul(self.head_dim)
            .ok_or(BackendError::SizeOverflow {
                field: "projected KV elements",
            })
    }

    pub fn cache_elements(self) -> Result<usize, BackendError> {
        self.n_head_kv
            .checked_mul(self.max_context)
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or(BackendError::SizeOverflow {
                field: "KV elements",
            })
    }
}

/// The memory capacity reported by a backend driver.
///
/// This describes free and total device memory. `MemoryBudget` bounds Leone's
/// owned allocations separately and does not claim that the driver has space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCapacity {
    Limited {
        available_bytes: u64,
        total_bytes: u64,
    },
    Unbounded,
}

/// One-time lossless relayout work performed while importing model weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelImportMetrics {
    pub lossless_repack_source_bytes: u64,
    pub lossless_repack_duration: Duration,
}

/// Where an operation reads its decode position from.
///
/// `Host` carries the value. `Device` points at a `u32` the GPU increments
/// inside a captured graph. The host does not read that value and does not
/// synchronize. A backend that cannot consume a device position calls
/// [`Backend::resolve_position`].
pub enum Position<'a, T> {
    Host(usize),
    Device(&'a T),
}

impl<T> Copy for Position<'_, T> {}

impl<T> Clone for Position<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The repeatability guarantee for one backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Determinism {
    /// Fixed launch and reduction order gives identical bits on the same device type.
    FixedOrder,
}

/// Returns the power-of-two context bucket used by decode graphs.
///
/// Buckets start at 512 positions. A backend rebuilds its graph when decode
/// crosses a bucket boundary.
pub fn decode_graph_bucket(context_length: usize) -> Result<usize, BackendError> {
    context_length
        .max(512)
        .checked_next_power_of_two()
        .ok_or(BackendError::SizeOverflow {
            field: "decode graph context bucket",
        })
}

/// One operation class in a batch-1 decode evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecodeOp {
    Embed,
    QkvGemv,
    QkNorm,
    Rope,
    KvAppend,
    Attention,
    OutputGemv,
    FfnGemv,
    SwiGlu,
    Norm,
    LmHeadGemv,
    Argmax,
}

impl DecodeOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed
            | Self::QkvGemv
            | Self::QkNorm
            | Self::Rope
            | Self::KvAppend
            | Self::Attention => self.attention_name(),
            Self::OutputGemv
            | Self::FfnGemv
            | Self::SwiGlu
            | Self::Norm
            | Self::LmHeadGemv
            | Self::Argmax => self.output_name(),
        }
    }

    const fn attention_name(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::QkvGemv => "qkv gemv",
            Self::QkNorm => "qknorm",
            Self::Rope => "rope",
            Self::KvAppend => "kv append",
            Self::Attention => "attention",
            _ => unreachable!(),
        }
    }

    const fn output_name(self) -> &'static str {
        match self {
            Self::OutputGemv => "o gemv",
            Self::FfnGemv => "ffn gemvs",
            Self::SwiGlu => "swiglu",
            Self::Norm => "norms",
            Self::LmHeadGemv => "lm_head gemv",
            Self::Argmax => "argmax",
            _ => unreachable!(),
        }
    }

    pub const fn all() -> [Self; 12] {
        [
            Self::Embed,
            Self::QkvGemv,
            Self::QkNorm,
            Self::Rope,
            Self::KvAppend,
            Self::Attention,
            Self::OutputGemv,
            Self::FfnGemv,
            Self::SwiGlu,
            Self::Norm,
            Self::LmHeadGemv,
            Self::Argmax,
        ]
    }
}

/// CUDA-event timing for one quantized matrix shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemvProfile {
    pub calls: usize,
    pub gpu_duration: Duration,
}

/// GPU time and host traffic collected over steady-state decode evaluations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeProfile {
    pub steps: usize,
    pub wall_duration: Duration,
    pub gpu_duration_by_op: BTreeMap<DecodeOp, Duration>,
    pub gemv_by_shape: BTreeMap<QuantMatrix, GemvProfile>,
    pub kernel_launches: usize,
    pub h2d_copies: usize,
    pub d2h_copies: usize,
    pub stream_synchronizations: usize,
}

impl DecodeProfile {
    pub fn gpu_duration(&self) -> Duration {
        self.gpu_duration_by_op.values().copied().sum()
    }

    pub fn host_gap(&self) -> Duration {
        self.wall_duration.saturating_sub(self.gpu_duration())
    }
}

fn unsupported_kv_layout(message: &'static str) -> BackendError {
    BackendError::operation("use KV span layout", message)
}

fn validate_span_append_position<T>(
    target: &KvWriteSpan<'_, T>,
    shape: AttentionShape,
    position: Position<'_, T>,
) -> Result<Option<usize>, BackendError> {
    match position {
        Position::Host(position) => {
            if position >= shape.max_context() {
                return Err(BackendError::PositionOutOfBounds {
                    position,
                    max_context: shape.max_context(),
                });
            }
            Ok(Some(target.local_position(position)?))
        }
        Position::Device(_) if target.logical_start() != 0 => Err(unsupported_kv_layout(
            "device-position append requires a zero-start span",
        )),
        Position::Device(_) if target.capacity_token_count() > shape.max_context() => {
            Err(BackendError::PositionOutOfBounds {
                position: target.capacity_token_count(),
                max_context: shape.max_context(),
            })
        }
        Position::Device(_) => Ok(None),
    }
}

/// Runs the operations required by Qwen3 decode and chunked prefill.
///
/// Buffers are opaque to the runtime. Dense values use `f16`, `f32`, or `u32`.
/// Uploaded dense bytes use little-endian encoding. Quantized uploads use GGUF
/// bytes at the contract boundary. A backend may repack them inside its opaque
/// buffer. Every operation rejects wrong storage or shape.
///
/// GEMV reads a row-major `[rows][columns]` matrix and one `columns` vector.
/// Prefill GEMM reads the same quantized matrix and a row-major
/// `[tokens][columns]` input. It writes `[tokens][rows]`. Chunked KV append
/// reads `[tokens][head_kv][head_dim]`. Prefill attention reads queries in
/// `[tokens][head][head_dim]` order and applies a causal mask through the end
/// of the current position block.
///
/// RMSNorm uses `sqrt(mean(x*x) + epsilon)` for each row. The residual form
/// normalizes `left + right`. RoPE rotates GPT-NeoX half pairs at one scalar
/// position. SwiGLU computes `silu(gate) * up`. Attention reads contiguous
/// f16 or f32 KV as `[head_kv][max_context][head_dim]`. It accumulates in f32
/// and uses causal positions `0..context_length`. Argmax returns the lowest
/// index on a finite-value tie.
///
/// Span attention uses `capacity_tokens` as each physical row stride. A read
/// view supplies an address bound, not an initialization claim. The caller
/// initializes every causal prefix before dispatch. `AttentionShape::max_context`
/// remains the logical context and reduction bound.
///
/// Backends may use different intermediate number formats. The CUDA Q4_K and
/// Q6_K GEMV paths quantize each 32-value activation block to signed q8_1 with
/// an `f16` scale. The scalar CPU path keeps `f32` activations. Logit KLD
/// against the scalar path is expected to have magnitude near `1e-3` over a
/// causal prefix. Differential tests define the accepted operation-level error.
///
/// A decode graph covers work from the device input token through argmax and
/// device position increment. The token D2H copy stays outside the graph.
/// Graph cache lengths use `decode_graph_bucket`. A backend rebuilds the graph
/// before a position exceeds its bucket. Device-position operations read one
/// `u32` scalar. CPU backends may report no graph support and execute eagerly.
///
/// `FixedOrder` means repeated calls with the same inputs return identical bits
/// on the same backend and device type. Results may differ across backends.
pub trait Backend {
    type Buffer: fmt::Debug;

    fn name(&self) -> &'static str;
    fn determinism(&self) -> Determinism;
    /// Returns the largest decode batch this backend can execute in one pass.
    fn max_batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(1).expect("one is nonzero")
    }
    fn prefill_method(&self) -> PrefillMethod {
        PrefillMethod::SequentialDecode
    }

    /// Returns true when chunked prefill can write and read `q8` KV storage.
    fn q8_prefill_supported(&self) -> bool {
        false
    }
    /// Returns true when this backend can satisfy decode-equivalent warm prefill
    /// for its supported device, build, model, KV, and operator shapes.
    ///
    /// `prepare_prefill` rejects a `DecodeEquivalent` plan when this returns
    /// false. The default is false.
    fn decode_equivalent_prefill_supported(&self) -> bool {
        false
    }
    /// Returns bytes owned by tracked allocations and external object counts.
    fn memory_accounting(&self) -> MemoryAccounting;
    /// Returns the parent ledger charged by this backend's tracker.
    fn memory_tracker_root(&self) -> MemoryTrackerRoot;
    /// Reassigns a live allocation. Class peaks retain its earlier attribution.
    fn classify_buffer(
        &mut self,
        buffer: &Self::Buffer,
        class: MemoryClass,
    ) -> Result<(), BackendError>;
    /// Installs the child tracker before any backend allocation begins.
    ///
    /// The tracker may share a parent with host staging. Implementations reject
    /// replacement after their current tracker owns or reserves bytes.
    fn set_memory_tracker(&mut self, tracker: MemoryTracker) -> Result<(), BackendError>;
    /// Sets the owned-allocation budget shared by backend buffers and scratch.
    fn set_memory_budget(&mut self, budget: MemoryBudget) -> Result<(), BackendError>;
    /// Allocates directly in one physical memory class.
    fn allocate_classified(
        &mut self,
        layout: BufferLayout,
        class: MemoryClass,
    ) -> Result<Self::Buffer, BackendError>;
    fn memory_capacity(&mut self) -> Result<MemoryCapacity, BackendError>;
    fn model_import_metrics(&self) -> ModelImportMetrics {
        ModelImportMetrics::default()
    }
    fn allocate(&mut self, layout: BufferLayout) -> Result<Self::Buffer, BackendError>;
    fn upload(&mut self, layout: BufferLayout, bytes: &[u8]) -> Result<Self::Buffer, BackendError>;
    /// Uploads model bytes while charging backend-specific host staging.
    fn upload_with_host_staging(
        &mut self,
        layout: BufferLayout,
        bytes: &[u8],
        _staging: &HostStaging,
    ) -> Result<Self::Buffer, BackendError> {
        self.upload(layout, bytes)
    }
    /// Allocates an independent buffer with the exact contents of `source`.
    /// The allocation inherits the source's memory class.
    ///
    /// The copy is ordered with other backend work. A backend may enqueue it,
    /// so the caller must synchronize before timing completion or using the
    /// result from another execution context.
    fn clone_buffer(&mut self, source: &Self::Buffer) -> Result<Self::Buffer, BackendError>;
    /// Copies one buffer into portable host-owned physical bytes.
    fn download_buffer(&mut self, source: &Self::Buffer) -> Result<BufferSnapshot, BackendError>;
    /// Restores physical bytes without applying model-import transformations.
    fn restore_buffer(&mut self, source: &BufferSnapshot) -> Result<Self::Buffer, BackendError>;
    /// Restores physical bytes directly into one allocation class.
    fn restore_buffer_classified(
        &mut self,
        source: &BufferSnapshot,
        class: MemoryClass,
    ) -> Result<Self::Buffer, BackendError>;
    fn configure_rope(
        &mut self,
        _head_dim: usize,
        _theta: f32,
        _frequency_factors: Option<&[f32]>,
        _pairing: RopePairing,
    ) -> Result<(), BackendError> {
        Ok(())
    }
    /// Configures RoPE while charging temporary host frequency storage.
    ///
    /// A backend that retains its host frequency table overrides this method
    /// and keeps the committed host allocation with that table.
    fn configure_rope_with_host_staging(
        &mut self,
        head_dim: usize,
        theta: f32,
        frequency_factors: Option<&[f32]>,
        pairing: RopePairing,
        staging: &HostStaging,
    ) -> Result<(), BackendError> {
        let reservation = reserve_rope_host_bytes(staging, head_dim)?;
        self.configure_rope(head_dim, theta, frequency_factors, pairing)?;
        if let Some(reservation) = reservation {
            let _allocation = reservation.commit()?;
        }
        Ok(())
    }
    /// Reads a device position back to the host.
    ///
    /// A backend that consumes a device position directly overrides the
    /// operation instead of calling this. Calling it inside a captured graph
    /// forces a synchronization. The CUDA backend does not call it there.
    fn resolve_position(
        &mut self,
        position: Position<'_, Self::Buffer>,
    ) -> Result<usize, BackendError> {
        match position {
            Position::Host(value) => Ok(value),
            Position::Device(buffer) => {
                let mut value = [0_u32];
                self.read_u32(buffer, &mut value)?;
                Ok(value[0] as usize)
            }
        }
    }
    fn prepare_rope(&mut self, _position: Position<'_, Self::Buffer>) -> Result<(), BackendError> {
        Ok(())
    }
    fn write_u32(&mut self, buffer: &mut Self::Buffer, values: &[u32]) -> Result<(), BackendError>;
    fn read_u32(&mut self, buffer: &Self::Buffer, values: &mut [u32]) -> Result<(), BackendError>;
    fn read_f16(&mut self, buffer: &Self::Buffer, values: &mut [u16]) -> Result<(), BackendError>;
    fn read_f32(&mut self, buffer: &Self::Buffer, values: &mut [f32]) -> Result<(), BackendError>;
    /// Validates a prefill plan and allocates backend-owned workspace.
    ///
    /// The default implementation rejects `DecodeEquivalent` unless the
    /// backend advertises that contract through
    /// [`Backend::decode_equivalent_prefill_supported`]. Overrides retain this
    /// validation before allocating workspace.
    fn prepare_prefill(&mut self, plan: PrefillPlan) -> Result<PrefillWorkspace, BackendError> {
        if plan.numerics() == PrefillNumerics::DecodeEquivalent
            && !self.decode_equivalent_prefill_supported()
        {
            return Err(BackendError::operation(
                "prepare decode-equivalent prefill",
                "the backend does not support decode-equivalent prefill",
            ));
        }
        Ok(PrefillWorkspace::default())
    }
    fn prefill_gemm(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError>;
    /// Returns true when the backend can verify several decode positions
    /// against one weight stream without changing any position's logits.
    fn verify_supported(&self) -> bool {
        false
    }
    /// Applies one quantized matrix to position-major activation rows.
    fn verify_gemv(
        &mut self,
        _weights: &Self::Buffer,
        _input: &Self::Buffer,
        _output: &mut Self::Buffer,
        _shape: QuantMatrix,
        _positions: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::operation(
            "run verifier GEMV",
            "the backend does not support batched verification",
        ))
    }
    /// Applies three matrices to the same position-major activation rows.
    #[allow(clippy::too_many_arguments)]
    fn verify_gemv_triple(
        &mut self,
        first_weights: &Self::Buffer,
        second_weights: &Self::Buffer,
        third_weights: &Self::Buffer,
        input: &Self::Buffer,
        first_output: &mut Self::Buffer,
        second_output: &mut Self::Buffer,
        third_output: &mut Self::Buffer,
        first_shape: QuantMatrix,
        second_shape: QuantMatrix,
        third_shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.verify_gemv(first_weights, input, first_output, first_shape, positions)?;
        self.verify_gemv(
            second_weights,
            input,
            second_output,
            second_shape,
            positions,
        )?;
        self.verify_gemv(third_weights, input, third_output, third_shape, positions)
    }
    /// Applies two matrices to the same position-major activation rows.
    #[allow(clippy::too_many_arguments)]
    fn verify_gemv_pair(
        &mut self,
        first_weights: &Self::Buffer,
        second_weights: &Self::Buffer,
        input: &Self::Buffer,
        first_output: &mut Self::Buffer,
        second_output: &mut Self::Buffer,
        first_shape: QuantMatrix,
        second_shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.verify_gemv(first_weights, input, first_output, first_shape, positions)?;
        self.verify_gemv(
            second_weights,
            input,
            second_output,
            second_shape,
            positions,
        )
    }
    /// Applies one quantized matrix to verifier rows prepared by the prior operation.
    #[allow(clippy::too_many_arguments)]
    fn verify_gemv_residual_prepared(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.verify_gemv_residual(weights, input, residual, output, shape, positions)
    }
    /// Applies SwiGLU and prepares its verifier rows for a following GEMV.
    fn verify_swiglu(
        &mut self,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        output: &mut Self::Buffer,
        _columns: usize,
        _positions: usize,
    ) -> Result<(), BackendError> {
        self.swiglu(gate, up, output)
    }
    /// Applies one quantized matrix and a position-major residual.
    #[allow(clippy::too_many_arguments)]
    fn verify_gemv_residual(
        &mut self,
        _weights: &Self::Buffer,
        _input: &Self::Buffer,
        _residual: &Self::Buffer,
        _output: &mut Self::Buffer,
        _shape: QuantMatrix,
        _positions: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::operation(
            "run verifier residual GEMV",
            "the backend does not support batched verification",
        ))
    }
    fn gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn gemv_residual(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn gemv_pair(
        &mut self,
        first_weights: &Self::Buffer,
        first_shape: QuantMatrix,
        second_weights: &Self::Buffer,
        second_shape: QuantMatrix,
        input: &Self::Buffer,
        first_output: &mut Self::Buffer,
        second_output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        self.gemv(first_weights, input, first_output, first_shape)?;
        self.gemv(second_weights, input, second_output, second_shape)
    }
    #[allow(clippy::too_many_arguments)]
    fn gemv_pair_swiglu(
        &mut self,
        gate_weights: &Self::Buffer,
        gate_shape: QuantMatrix,
        up_weights: &Self::Buffer,
        up_shape: QuantMatrix,
        input: &Self::Buffer,
        gate: &mut Self::Buffer,
        up: &mut Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        self.gemv_pair(
            gate_weights,
            gate_shape,
            up_weights,
            up_shape,
            input,
            gate,
            up,
        )?;
        self.swiglu(gate, up, output)
    }
    #[allow(clippy::too_many_arguments)]
    fn qkv_gemv(
        &mut self,
        query_weights: &Self::Buffer,
        query_shape: QuantMatrix,
        key_weights: &Self::Buffer,
        key_shape: QuantMatrix,
        value_weights: &Self::Buffer,
        value_shape: QuantMatrix,
        input: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
    ) -> Result<(), BackendError>;
    fn rms_norm(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError>;
    fn prefill_rms_norm(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        self.rms_norm(input, weight, output, shape, epsilon)
    }
    /// Normalizes and rotates position-major rows for chunked prefill.
    #[allow(clippy::too_many_arguments)]
    fn prefill_rms_norm_rope(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        rope_shape: RopeShape,
        start_position: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        self.prefill_rms_norm(input, weight, output, shape, epsilon)?;
        self.rope(output, start_position, rope_shape, theta)
    }
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_rope(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        position: Position<'_, Self::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(
        &mut self,
        query: &Self::Buffer,
        query_weight: &Self::Buffer,
        query_output: &mut Self::Buffer,
        query_shape: VectorShape,
        key: &Self::Buffer,
        key_weight: &Self::Buffer,
        key_output: &mut Self::Buffer,
        key_shape: VectorShape,
        position: Position<'_, Self::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        self.rms_norm_rope(
            query,
            query_weight,
            query_output,
            query_shape,
            position,
            epsilon,
            theta,
        )?;
        self.rms_norm_rope(
            key, key_weight, key_output, key_shape, position, epsilon, theta,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_kv_append(
        &mut self,
        query: &Self::Buffer,
        query_weight: &Self::Buffer,
        query_output: &mut Self::Buffer,
        query_shape: VectorShape,
        key: &Self::Buffer,
        key_weight: &Self::Buffer,
        key_output: &mut Self::Buffer,
        key_shape: VectorShape,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
        attention_shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        self.qk_norm_rope(
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            position,
            epsilon,
            theta,
        )?;
        self.kv_append(
            key_output,
            value,
            key_cache,
            value_cache,
            attention_shape,
            position,
        )
    }
    /// Normalizes and rotates one QK row, then appends its KV to a span.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_kv_append_span(
        &mut self,
        query: &Self::Buffer,
        query_weight: &Self::Buffer,
        query_output: &mut Self::Buffer,
        query_shape: VectorShape,
        key: &Self::Buffer,
        key_weight: &Self::Buffer,
        key_output: &mut Self::Buffer,
        key_shape: VectorShape,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        attention_shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        let _physical_shape = AttentionShape::new(
            attention_shape.n_head(),
            attention_shape.n_head_kv(),
            attention_shape.head_dim(),
            target.capacity_token_count(),
        )?;
        validate_span_append_position(&target, attention_shape, position)?;
        self.qk_norm_rope(
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            position,
            epsilon,
            theta,
        )?;
        self.kv_append_span(key_output, value, target, attention_shape, position)
    }
    /// Normalizes and rotates position-major QK rows, then appends their KV.
    #[allow(clippy::too_many_arguments)]
    fn verify_qk_norm_rope_kv_append(
        &mut self,
        _query: &Self::Buffer,
        _query_weight: &Self::Buffer,
        _query_output: &mut Self::Buffer,
        _query_shape: VectorShape,
        _key: &Self::Buffer,
        _key_weight: &Self::Buffer,
        _key_output: &mut Self::Buffer,
        _key_shape: VectorShape,
        _value: &Self::Buffer,
        _key_cache: &mut Self::Buffer,
        _value_cache: &mut Self::Buffer,
        _attention_shape: AttentionShape,
        _start_position: usize,
        _positions: usize,
        _epsilon: f32,
        _theta: f32,
    ) -> Result<(), BackendError> {
        Err(BackendError::operation(
            "run verifier QK normalization",
            "the backend does not support batched verification",
        ))
    }
    /// Normalizes and rotates verifier rows, then appends their KV to a span.
    #[allow(clippy::too_many_arguments)]
    fn verify_qk_norm_rope_kv_append_span(
        &mut self,
        query: &Self::Buffer,
        query_weight: &Self::Buffer,
        query_output: &mut Self::Buffer,
        query_shape: VectorShape,
        key: &Self::Buffer,
        key_weight: &Self::Buffer,
        key_output: &mut Self::Buffer,
        key_shape: VectorShape,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        attention_shape: AttentionShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        if target.logical_start() != 0
            || target.capacity_token_count() != attention_shape.max_context()
        {
            return Err(unsupported_kv_layout(
                "verifier fused append requires one contiguous span",
            ));
        }
        if positions == 0 {
            return Err(BackendError::Zero {
                field: "verifier KV append positions",
            });
        }
        target.local_range(start_position, positions)?;
        let (key_cache, value_cache, _, _) = target.into_parts();
        self.verify_qk_norm_rope_kv_append(
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            value,
            key_cache,
            value_cache,
            attention_shape,
            start_position,
            positions,
            epsilon,
            theta,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_residual(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_residual_store(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        residual: &mut Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError>;
    fn rope(
        &mut self,
        values: &mut Self::Buffer,
        position: usize,
        shape: RopeShape,
        theta: f32,
    ) -> Result<(), BackendError>;
    /// Applies RoPE at a host or backend-resident decode position.
    fn rope_position(
        &mut self,
        values: &mut Self::Buffer,
        position: Position<'_, Self::Buffer>,
        shape: RopeShape,
        theta: f32,
    ) -> Result<(), BackendError> {
        let position = self.resolve_position(position)?;
        self.rope(values, position, shape, theta)
    }
    fn swiglu(
        &mut self,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError>;
    fn residual_add(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn kv_append(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn kv_append_chunk(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError>;
    /// Appends one projected KV row to a physical cache span.
    #[allow(clippy::too_many_arguments)]
    fn kv_append_span(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        let physical_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            target.capacity_token_count(),
        )?;
        let local_position = validate_span_append_position(&target, shape, position)?;
        let (key_cache, value_cache, _, _) = target.into_parts();
        let position = local_position.map_or(position, Position::Host);
        self.kv_append(key, value, key_cache, value_cache, physical_shape, position)
    }

    /// Appends projected KV rows to one physical cache span.
    #[allow(clippy::too_many_arguments)]
    fn kv_append_chunk_span(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        if tokens == 0 {
            return Err(BackendError::Zero {
                field: "KV append tokens",
            });
        }
        let end_position =
            start_position
                .checked_add(tokens)
                .ok_or(BackendError::SizeOverflow {
                    field: "KV append end position",
                })?;
        if end_position > shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position: end_position,
                max_context: shape.max_context(),
            });
        }
        let local_range = target.local_range(start_position, tokens)?;
        let physical_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            target.capacity_token_count(),
        )?;
        let (key_cache, value_cache, _, _) = target.into_parts();
        self.kv_append_chunk(
            key,
            value,
            key_cache,
            value_cache,
            physical_shape,
            local_range.start,
            tokens,
        )
    }

    /// Prepares one mutable KV span for a later append.
    ///
    /// The default implementation keeps no backend state. Backends with
    /// replayable graphs may retain device descriptors and allocation pins.
    /// The backend pins the target allocation through the graph lifetime.
    fn prepare_kv_write_span(
        &mut self,
        _target: KvWriteSpan<'_, Self::Buffer>,
        _shape: AttentionShape,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Prepares an immutable KV view for a later attention operation.
    ///
    /// The default implementation keeps no backend state. A backend may copy
    /// descriptors to device storage, but it must not retain the borrowed
    /// view or its host span slice after this call returns.
    fn prepare_kv_read_view(
        &mut self,
        _cache: KvReadView<'_, Self::Buffer>,
        _shape: AttentionShape,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Prepares metadata and bounded workspace for batched decode attention.
    ///
    /// Query, KV, output, and device-position contents may be uninitialized.
    /// This method must not read those contents or change caller buffers.
    /// The caller prepares all layers after staging their KV append targets,
    /// before model execution or graph capture. A backend reserves descriptor
    /// and workspace allocations here, and retains allocation handles for any
    /// asynchronous work or graph that uses them. It must not retain host
    /// references. The default prepares each immutable KV view in row order.
    fn prepare_attention_decode_batch_spans(
        &mut self,
        rows: &[AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        for row in rows {
            self.prepare_kv_read_view(row.cache, row.shape)?;
        }
        Ok(())
    }

    /// Attends independent query rows over their immutable KV spans.
    ///
    /// The default calls [`Backend::attention_decode_spans`] in row order.
    /// Overrides may reuse reads from identical immutable allocations. They
    /// preserve each row's causal range and never share query-dependent state.
    /// A changed numerical path requires its own oracle and repeatability
    /// contract. Preparation does not authorize reading uninitialized tails.
    /// Outputs may be partial on error; callers quarantine all affected rows.
    fn attention_decode_batch_spans(
        &mut self,
        rows: &mut [AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        for row in rows {
            self.attention_decode_spans(row.query, row.cache, row.output, row.shape, row.position)?;
        }
        Ok(())
    }

    /// Retains dynamic buffers referenced by the next decode graph.
    ///
    /// The runtime calls this after graph metadata preparation and before
    /// [`Backend::begin_decode_graph`]. A backend keeps each buffer alive until
    /// the graph retires. The list contains every session position and
    /// activation buffer referenced by the capture. The default has no graph
    /// ownership because eager CPU execution keeps no device pointers.
    fn retain_decode_graph_buffers(
        &mut self,
        _buffers: &[&Self::Buffer],
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Starts a scope that prepares all KV descriptors for a graph replacement.
    ///
    /// The default implementation keeps no backend state. A backend with
    /// replayable graphs can stage a protected descriptor set in this scope.
    fn begin_kv_graph_preflight(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    /// Finishes the scope that prepares descriptors for a graph replacement.
    ///
    /// `keep` is true after every required view was prepared. The default
    /// implementation has no staged state to release.
    fn end_kv_graph_preflight(&mut self, _keep: bool) -> Result<(), BackendError> {
        Ok(())
    }

    /// Attends one query token at `position` over the cache before it.
    ///
    /// The attended context length is `position + 1` for both host and device
    /// positions.
    #[allow(clippy::too_many_arguments)]
    fn attention_decode(
        &mut self,
        query: &Self::Buffer,
        key_cache: &Self::Buffer,
        value_cache: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError>;
    /// Attends one query token over ordered immutable KV spans.
    #[allow(clippy::too_many_arguments)]
    fn attention_decode_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        let spans = cache.spans();
        if spans.len() != 1 {
            return Err(unsupported_kv_layout(
                "attention delegation requires one contiguous span",
            ));
        }
        let span = &spans[0];
        if span.logical_start() != 0 || span.capacity_token_count() != shape.max_context() {
            return Err(unsupported_kv_layout(
                "attention delegation requires a matching span capacity",
            ));
        }
        if matches!(position, Position::Device(_))
            && span.token_count() != span.capacity_token_count()
        {
            return Err(unsupported_kv_layout(
                "device-position attention requires a fully mapped span",
            ));
        }
        if let Position::Host(position) = position {
            let read_end = position.checked_add(1).ok_or(BackendError::SizeOverflow {
                field: "attention context length",
            })?;
            if read_end > cache.mapped_tokens() {
                return Err(BackendError::PositionOutOfBounds {
                    position: read_end,
                    max_context: cache.mapped_tokens(),
                });
            }
        }
        self.attention_decode(query, span.key(), span.value(), output, shape, position)
    }

    /// Reports whether verifier attention prepares rows for the following GEMV.
    fn verifier_attention_prepares_output(&self, _shape: AttentionShape) -> bool {
        false
    }
    /// Attends position-major queries at consecutive decode positions.
    #[allow(clippy::too_many_arguments)]
    fn verify_attention(
        &mut self,
        _query: &Self::Buffer,
        _key_cache: &Self::Buffer,
        _value_cache: &Self::Buffer,
        _output: &mut Self::Buffer,
        _shape: AttentionShape,
        _start_position: usize,
        _positions: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::operation(
            "run verifier attention",
            "the backend does not support batched verification",
        ))
    }
    /// Attends verifier queries over ordered immutable KV spans.
    #[allow(clippy::too_many_arguments)]
    fn verify_attention_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        positions: usize,
    ) -> Result<(), BackendError> {
        let spans = cache.spans();
        if spans.len() != 1 {
            return Err(unsupported_kv_layout(
                "verifier attention delegation requires one contiguous span",
            ));
        }
        let span = &spans[0];
        if span.logical_start() != 0 || span.capacity_token_count() != shape.max_context() {
            return Err(unsupported_kv_layout(
                "verifier attention delegation requires a matching span capacity",
            ));
        }
        if positions == 0 {
            return Err(BackendError::Zero {
                field: "verifier attention positions",
            });
        }
        let read_end = start_position
            .checked_add(positions)
            .ok_or(BackendError::SizeOverflow {
                field: "verifier attention end position",
            })?;
        if read_end > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: read_end,
                max_context: cache.mapped_tokens(),
            });
        }
        self.verify_attention(
            query,
            span.key(),
            span.value(),
            output,
            shape,
            start_position,
            positions,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill(
        &mut self,
        query: &Self::Buffer,
        key_cache: &Self::Buffer,
        value_cache: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError>;
    /// Attends causal prefill rows over ordered immutable KV spans.
    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        if tokens == 0 {
            return Err(BackendError::Zero {
                field: "prefill attention tokens",
            });
        }
        let spans = cache.spans();
        if spans.len() != 1 {
            return Err(unsupported_kv_layout(
                "prefill attention delegation requires one contiguous span",
            ));
        }
        let span = &spans[0];
        if span.logical_start() != 0 || span.capacity_token_count() != shape.max_context() {
            return Err(unsupported_kv_layout(
                "prefill attention delegation requires a matching span capacity",
            ));
        }
        let read_end = start_position
            .checked_add(tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "prefill attention end position",
            })?;
        if read_end > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: read_end,
                max_context: cache.mapped_tokens(),
            });
        }
        self.attention_prefill(
            query,
            span.key(),
            span.value(),
            output,
            shape,
            start_position,
            tokens,
        )
    }

    fn embed_gather(
        &mut self,
        table: &Self::Buffer,
        row: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError>;
    fn embed_gather_batch(
        &mut self,
        table: &Self::Buffer,
        rows: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError>;
    fn copy_f32_row(
        &mut self,
        input: &Self::Buffer,
        row: usize,
        columns: usize,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError>;
    /// Copies an exact FP32 vector into one matrix row.
    fn write_f32_row(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        row: usize,
        columns: usize,
    ) -> Result<(), BackendError>;
    fn argmax(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError>;
    fn synchronize(&mut self) -> Result<(), BackendError>;

    /// Returns true when the backend can capture and replay one decode graph.
    fn decode_graph_supported(&self) -> bool {
        false
    }

    /// Starts stream capture for one decode graph.
    fn begin_decode_graph(&mut self) -> Result<(), BackendError> {
        Err(BackendError::operation(
            "begin decode graph",
            "decode graphs are not supported",
        ))
    }

    /// Finishes capture and replaces the backend's current decode graph.
    fn end_decode_graph(&mut self) -> Result<(), BackendError> {
        Err(BackendError::operation(
            "end decode graph",
            "decode graphs are not supported",
        ))
    }

    /// Launches the backend's current decode graph once.
    fn replay_decode_graph(&mut self) -> Result<(), BackendError> {
        Err(BackendError::operation(
            "replay decode graph",
            "decode graphs are not supported",
        ))
    }

    /// Retires the current decode graph and any graph-owned KV descriptors.
    ///
    /// The backend fences the stream before releasing graph resources. A
    /// runtime calls this when a session is discarded, hibernated, or
    /// cancelled. The default implementation has no graph state.
    fn drop_decode_graph(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    /// Increments one device `u32` scalar in stream order.
    fn increment_u32(&mut self, buffer: &mut Self::Buffer) -> Result<(), BackendError> {
        let mut value = [0_u32];
        self.read_u32(buffer, &mut value)?;
        value[0] = value[0].checked_add(1).ok_or(BackendError::SizeOverflow {
            field: "device position",
        })?;
        self.write_u32(buffer, &value)
    }

    /// Starts optional decode profiling with storage for `operations` boundaries.
    fn begin_decode_profile(&mut self, _operations: usize) -> Result<(), BackendError> {
        Ok(())
    }

    /// Marks the operation that follows this stream boundary.
    fn profile_decode_op(&mut self, _op: DecodeOp) -> Result<(), BackendError> {
        Ok(())
    }

    /// Finishes optional decode profiling and returns backend measurements.
    fn end_decode_profile(
        &mut self,
        _steps: usize,
        _wall_duration: Duration,
    ) -> Result<Option<DecodeProfile>, BackendError> {
        Ok(None)
    }
}

fn nonzero(field: &'static str, value: usize) -> Result<(), BackendError> {
    if value == 0 {
        Err(BackendError::Zero { field })
    } else {
        Ok(())
    }
}

fn divisible(field: &'static str, value: usize, divisor: usize) -> Result<(), BackendError> {
    if value.is_multiple_of(divisor) {
        Ok(())
    } else {
        Err(BackendError::NotDivisible {
            field,
            value,
            divisor,
        })
    }
}

pub(crate) fn validate_positive(field: &'static str, value: f32) -> Result<(), BackendError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(BackendError::InvalidPositiveFloat { field, value })
    }
}

pub(crate) fn exact_len(
    name: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), BackendError> {
    if expected == actual {
        Ok(())
    } else {
        Err(BackendError::SizeMismatch {
            name,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn memory_tracker_returns_live_bytes_after_drop() {
        let tracker = MemoryTracker::default();
        let allocation = tracker.allocate(MemoryClass::KvCache, 128).unwrap();
        let duplicate = allocation.duplicate().unwrap();
        let live = tracker.snapshot();
        assert_eq!(live.live_bytes, 256);
        assert_eq!(live.class(MemoryClass::KvCache).live_allocations, 2);
        drop(duplicate);
        allocation.reclassify(MemoryClass::BackendScratch);
        let live = tracker.snapshot();
        assert_eq!(live.class(MemoryClass::KvCache).live_bytes, 0);
        assert_eq!(live.class(MemoryClass::BackendScratch).live_bytes, 128);
        assert_eq!(live.class(MemoryClass::BackendScratch).allocations, 1);
        assert_eq!(live.peak_live_bytes, 256);
        drop(allocation);
        let empty = tracker.snapshot();
        assert_eq!(empty.live_bytes, 0);
        assert_eq!(empty.frees, 2);
    }

    #[test]
    fn memory_budget_reserves_before_commit_and_releases_on_drop() {
        let tracker = MemoryTracker::new(MemoryBudget::limited(128).unwrap());
        let reservation = tracker.reserve(MemoryClass::BackendScratch, 96).unwrap();
        let pending = tracker.snapshot();
        assert_eq!(pending.live_bytes, 0);
        assert_eq!(pending.reserved_bytes, 96);
        assert_eq!(pending.peak_owned_and_reserved_bytes, 96);
        assert!(matches!(
            tracker.reserve(MemoryClass::KvCache, 33),
            Err(MemoryError::BudgetExceeded { .. })
        ));
        drop(reservation);
        let after_failed_allocation = tracker.snapshot();
        assert_eq!(after_failed_allocation.reserved_bytes, 0);
        assert_eq!(after_failed_allocation.peak_owned_and_reserved_bytes, 96);

        let allocation = tracker.allocate(MemoryClass::KvCache, 96).unwrap();
        let live = tracker.snapshot();
        assert_eq!(live.reserved_bytes, 0);
        assert_eq!(live.peak_owned_and_reserved_bytes, 96);
        assert!(matches!(
            tracker.set_budget(MemoryBudget::limited(64).unwrap()),
            Err(MemoryError::BudgetBelowOwned { .. })
        ));
        assert_eq!(tracker.budget(), MemoryBudget::limited(128).unwrap());
        assert!(matches!(
            allocation.duplicate(),
            Err(MemoryError::BudgetExceeded { .. })
        ));
        drop(allocation);
        tracker
            .set_budget(MemoryBudget::limited(64).unwrap())
            .unwrap();
        assert_eq!(tracker.budget(), MemoryBudget::limited(64).unwrap());
    }

    #[test]
    fn concurrent_reservations_share_one_budget() {
        let tracker = Arc::new(MemoryTracker::new(MemoryBudget::limited(128).unwrap()));
        let barrier = Arc::new(Barrier::new(4));
        let workers = (0..4)
            .map(|_| {
                let tracker = Arc::clone(&tracker);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    tracker.reserve(MemoryClass::KvCache, 64)
                })
            })
            .collect::<Vec<_>>();
        let reservations = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            reservations.iter().filter(|result| result.is_ok()).count(),
            2
        );
        assert_eq!(tracker.reserved_bytes(), 128);
        drop(reservations);
        assert_eq!(tracker.reserved_bytes(), 0);
    }

    #[test]
    fn shared_parent_rejects_combined_overlap() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(100).unwrap());
        let host = MemoryTracker::child(MemoryBudget::limited(60).unwrap(), root.clone());
        let device = MemoryTracker::child(MemoryBudget::limited(80).unwrap(), root.clone());
        let host_allocation = host.allocate(MemoryClass::ContractBuffer, 40).unwrap();
        let device_allocation = device.allocate(MemoryClass::ModelWeight, 40).unwrap();
        let staging = host.reserve(MemoryClass::ContractBuffer, 20).unwrap();
        let before_rejection = root.snapshot();
        assert!(matches!(
            device.reserve(MemoryClass::ModelWeight, 1),
            Err(MemoryError::BudgetExceeded {
                budget: 100,
                owned: 80,
                reserved: 20,
                ..
            })
        ));
        assert_eq!(root.snapshot(), before_rejection);
        assert_eq!(root.snapshot().reserved_bytes, 20);
        drop(staging);
        assert_eq!(root.snapshot().reserved_bytes, 0);
        drop(device_allocation);
        drop(host_allocation);
        assert_eq!(root.owned_bytes(), 0);
    }

    #[test]
    fn shared_parent_reservation_commit_and_drop_charge_once() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(64).unwrap());
        let child = MemoryTracker::child(MemoryBudget::limited(64).unwrap(), root.clone());
        let reservation = child.reserve(MemoryClass::KvCache, 32).unwrap();
        assert_eq!(child.snapshot().reserved_bytes, 32);
        assert_eq!(root.snapshot().reserved_bytes, 32);
        assert_eq!(child.snapshot().peak_owned_and_reserved_bytes, 32);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 32);
        let allocation = reservation.commit().unwrap();
        assert_eq!(child.snapshot().live_bytes, 32);
        assert_eq!(root.snapshot().live_bytes, 32);
        assert_eq!(root.snapshot().allocations, 1);
        assert_eq!(child.snapshot().peak_owned_and_reserved_bytes, 32);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 32);
        drop(allocation);
        assert_eq!(child.snapshot().live_bytes, 0);
        assert_eq!(root.snapshot().live_bytes, 0);
        assert_eq!(root.snapshot().frees, 1);
        assert_eq!(child.snapshot().peak_owned_and_reserved_bytes, 32);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 32);
    }

    #[test]
    fn shared_parent_failed_physical_allocation_rolls_back_reservation() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(64).unwrap());
        let child = MemoryTracker::child(MemoryBudget::limited(64).unwrap(), root.clone());
        let reservation = child.reserve(MemoryClass::BackendScratch, 48).unwrap();
        assert_eq!(root.reserved_bytes(), 48);
        drop(reservation);
        assert_eq!(child.snapshot().reserved_bytes, 0);
        assert_eq!(root.snapshot().reserved_bytes, 0);
        assert_eq!(root.snapshot().live_bytes, 0);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 48);
        let allocation = child.allocate(MemoryClass::BackendScratch, 64).unwrap();
        assert_eq!(root.owned_bytes(), 64);
        drop(allocation);
    }

    #[test]
    fn child_rejection_leaves_shared_parent_unchanged() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(128).unwrap());
        let child = MemoryTracker::child(MemoryBudget::limited(16).unwrap(), root.clone());
        assert!(matches!(
            child.reserve(MemoryClass::Activation, 17),
            Err(MemoryError::BudgetExceeded {
                budget: 16,
                owned: 0,
                reserved: 0,
                ..
            })
        ));
        assert_eq!(root.snapshot().live_bytes, 0);
        assert_eq!(root.snapshot().reserved_bytes, 0);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 0);
        assert_eq!(child.snapshot().reserved_bytes, 0);
        assert_eq!(child.snapshot().peak_owned_and_reserved_bytes, 0);
    }

    #[test]
    fn shared_parent_budget_change_includes_pending_siblings() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(128).unwrap());
        let host = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let device = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let host_allocation = host.allocate(MemoryClass::ContractBuffer, 80).unwrap();
        let device_reservation = device.reserve(MemoryClass::ModelWeight, 40).unwrap();
        assert!(matches!(
            root.set_budget(MemoryBudget::limited(100).unwrap()),
            Err(MemoryError::BudgetBelowOwned {
                budget: 100,
                owned: 80,
                reserved: 40,
            })
        ));
        assert_eq!(root.budget(), MemoryBudget::limited(128).unwrap());
        assert!(matches!(
            host.set_budget(MemoryBudget::limited(79).unwrap()),
            Err(MemoryError::BudgetBelowOwned {
                budget: 79,
                owned: 80,
                reserved: 0,
            })
        ));
        drop(device_reservation);
        root.set_budget(MemoryBudget::limited(80).unwrap()).unwrap();
        drop(host_allocation);
    }

    #[test]
    fn root_budget_updates_an_independent_tracker_view() {
        let tracker = MemoryTracker::new(MemoryBudget::limited(128).unwrap());
        let root = tracker.root();
        root.set_budget(MemoryBudget::limited(64).unwrap()).unwrap();
        assert_eq!(tracker.budget(), MemoryBudget::limited(64).unwrap());
        assert_eq!(
            tracker.snapshot().budget,
            MemoryBudget::limited(64).unwrap()
        );
    }

    #[test]
    fn concurrent_sibling_reservations_share_parent_atomically() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(64).unwrap());
        let host = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let device = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let barrier = Arc::new(Barrier::new(4));
        let workers = [host.clone(), device.clone(), host, device]
            .into_iter()
            .map(|tracker| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    tracker.reserve(MemoryClass::BackendScratch, 64)
                })
            })
            .collect::<Vec<_>>();
        let reservations = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            reservations.iter().filter(|result| result.is_ok()).count(),
            1
        );
        assert_eq!(root.reserved_bytes(), 64);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 64);
        drop(reservations);
        assert_eq!(root.reserved_bytes(), 0);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 64);
    }

    #[test]
    fn shared_peak_includes_staggered_reservations() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(128).unwrap());
        let host = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let device = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let host_reservation = host.reserve(MemoryClass::ContractBuffer, 64).unwrap();
        let device_reservation = device.reserve(MemoryClass::ModelWeight, 32).unwrap();

        let pending = root.snapshot();
        assert_eq!(pending.live_bytes, 0);
        assert_eq!(pending.reserved_bytes, 96);
        assert_eq!(pending.peak_owned_and_reserved_bytes, 96);
        assert_eq!(host.snapshot().peak_owned_and_reserved_bytes, 64);
        assert_eq!(device.snapshot().peak_owned_and_reserved_bytes, 32);

        let host_allocation = host_reservation.commit().unwrap();
        let committed = root.snapshot();
        assert_eq!(committed.live_bytes, 64);
        assert_eq!(committed.reserved_bytes, 32);
        assert_eq!(committed.peak_owned_and_reserved_bytes, 96);

        drop(device_reservation);
        assert_eq!(root.snapshot().live_bytes, 64);
        assert_eq!(root.snapshot().reserved_bytes, 0);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 96);
        drop(host_allocation);
        assert_eq!(root.snapshot().live_bytes, 0);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 96);
    }

    #[test]
    fn failed_reservation_leaves_owned_reserved_peak_unchanged() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(64).unwrap());
        let child = MemoryTracker::child(MemoryBudget::limited(64).unwrap(), root.clone());
        let allocation = child.allocate(MemoryClass::BackendScratch, 48).unwrap();
        let root_before = root.snapshot();
        let child_before = child.snapshot();

        assert!(matches!(
            child.reserve(MemoryClass::BackendScratch, 17),
            Err(MemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(root.snapshot(), root_before);
        assert_eq!(child.snapshot(), child_before);

        drop(allocation);
        assert_eq!(root.snapshot().live_bytes, 0);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 48);
        assert_eq!(child.snapshot().peak_owned_and_reserved_bytes, 48);
    }

    #[test]
    fn shared_peak_records_simultaneous_ownership() {
        let root = MemoryTrackerRoot::new(MemoryBudget::limited(128).unwrap());
        let host = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let device = MemoryTracker::child(MemoryBudget::limited(128).unwrap(), root.clone());
        let staging = host.allocate(MemoryClass::ContractBuffer, 64).unwrap();
        drop(staging);
        let weight = device.allocate(MemoryClass::ModelWeight, 32).unwrap();
        let snapshot = host.allocate(MemoryClass::ContractBuffer, 16).unwrap();

        assert_eq!(root.snapshot().live_bytes, 48);
        assert_eq!(root.snapshot().peak_live_bytes, 64);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 64);
        assert_eq!(host.snapshot().peak_live_bytes, 64);
        assert_eq!(host.snapshot().peak_owned_and_reserved_bytes, 64);
        assert_eq!(device.snapshot().peak_live_bytes, 32);
        assert_eq!(device.snapshot().peak_owned_and_reserved_bytes, 32);
        drop((weight, snapshot));
        assert_eq!(root.snapshot().live_bytes, 0);
        assert_eq!(root.snapshot().peak_live_bytes, 64);
        assert_eq!(root.snapshot().peak_owned_and_reserved_bytes, 64);
    }

    #[test]
    fn allocation_byte_overflow_is_rejected_without_state_change() {
        let tracker = MemoryTracker::default();
        let allocation = tracker
            .allocate(MemoryClass::ModelWeight, u64::MAX)
            .unwrap();
        assert!(matches!(
            tracker.reserve(MemoryClass::ModelWeight, 1),
            Err(MemoryError::ByteOverflow)
        ));
        assert_eq!(tracker.owned_bytes(), u64::MAX);
        assert_eq!(tracker.snapshot().peak_owned_and_reserved_bytes, u64::MAX);
        drop(allocation);
        assert_eq!(tracker.owned_bytes(), 0);
        assert_eq!(tracker.snapshot().peak_owned_and_reserved_bytes, u64::MAX);
    }

    #[test]
    fn budget_types_reject_zero_and_charge_one_commit() {
        assert_eq!(MemoryBudget::limited(0), Err(MemoryError::ZeroBudget));
        let tracker = MemoryTracker::new(MemoryBudget::limited(64).unwrap());
        let reservation = tracker.reserve(MemoryClass::KvCache, 64).unwrap();
        assert_eq!(tracker.owned_bytes(), 0);
        let allocation = reservation.commit().unwrap();
        let live = tracker.snapshot();
        assert_eq!(live.live_bytes, 64);
        assert_eq!(live.reserved_bytes, 0);
        assert_eq!(live.allocations, 1);
        drop(allocation);
        assert_eq!(tracker.snapshot().frees, 1);
    }

    #[test]
    fn qwen_shapes_have_checked_storage() {
        let matrix = QuantMatrix::new(151_936, 4_096, QuantFormat::Q6K).unwrap();
        assert_eq!(matrix.layout().unwrap().bytes(), 510_504_960);
        let attention = AttentionShape::new(32, 8, 128, 8_192).unwrap();
        assert_eq!(attention.query_elements().unwrap(), 4_096);
        assert_eq!(attention.cache_elements().unwrap(), 8_388_608);
    }

    #[test]
    fn partial_quantized_rows_are_rejected() {
        assert!(matches!(
            QuantMatrix::new(1, 255, QuantFormat::Q4K),
            Err(BackendError::NotDivisible { .. })
        ));
    }
}
