use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;

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
/// External library objects are reported in `untracked` when their byte sizes
/// are not available through the backend contract.
/// Counters follow ownership release. They cannot confirm device cleanup after
/// an external library or driver rejects a destructor operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryAccounting {
    pub live_bytes: u64,
    pub peak_live_bytes: u64,
    pub live_allocations: u64,
    pub peak_live_allocations: u64,
    pub allocations: u64,
    pub frees: u64,
    pub classes: BTreeMap<MemoryClass, MemoryClassStats>,
    pub untracked: UntrackedMemory,
}

impl Default for MemoryAccounting {
    fn default() -> Self {
        Self {
            live_bytes: 0,
            peak_live_bytes: 0,
            live_allocations: 0,
            peak_live_allocations: 0,
            allocations: 0,
            frees: 0,
            classes: MemoryClass::ALL
                .into_iter()
                .map(|class| (class, MemoryClassStats::default()))
                .collect(),
            untracked: UntrackedMemory::default(),
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
struct MemoryTrackerState {
    total: MemoryCounters,
    classes: BTreeMap<MemoryClass, MemoryCounters>,
}

/// Tracks exact bytes owned by backend allocations.
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    state: Arc<Mutex<MemoryTrackerState>>,
}

impl MemoryTracker {
    /// Records one allocation and returns its drop-tracked ownership token.
    pub fn allocate(&self, class: MemoryClass, bytes: u64) -> MemoryAllocation {
        let mut state = lock_tracker(&self.state);
        record_allocate(&mut state.total, bytes);
        record_allocate(state.classes.entry(class).or_default(), bytes);
        let record = AllocationRecord {
            identity: state.total.allocations,
            tracker: self.clone(),
            class: AtomicU8::new(class as u8),
            bytes,
        };
        MemoryAllocation { record }
    }

    /// Returns a snapshot of all tracked allocation classes.
    pub fn snapshot(&self) -> MemoryAccounting {
        let state = lock_tracker(&self.state);
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
            peak_live_bytes: state.total.peak_live_bytes,
            live_allocations: state.total.live_allocations,
            peak_live_allocations: state.total.peak_live_allocations,
            allocations: state.total.allocations,
            frees: state.total.frees,
            classes,
            untracked: UntrackedMemory::default(),
        }
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
        let mut state = lock_tracker(&self.record.tracker.state);
        let previous = MemoryClass::from_index(self.record.class.load(Ordering::Acquire));
        if previous == class {
            return;
        }
        self.record.class.store(class as u8, Ordering::Release);
        move_live(&mut state, previous, class, self.record.bytes);
    }

    /// Creates an independent allocation with the same class and byte count.
    pub fn duplicate(&self) -> Self {
        self.record
            .tracker
            .allocate(self.class(), self.record.bytes)
    }
}

impl Drop for MemoryAllocation {
    fn drop(&mut self) {
        let mut state = lock_tracker(&self.record.tracker.state);
        let class = MemoryClass::from_index(self.record.class.load(Ordering::Acquire));
        record_free(&mut state.total, self.record.bytes);
        record_free(state.classes.entry(class).or_default(), self.record.bytes);
    }
}

fn lock_tracker<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
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
    /// Evaluates position blocks through FP16 cuBLASLt matrix products.
    ChunkedCublasLtFp16,
    /// Evaluates large position blocks through bounded cuBLASLt attention tiles.
    TiledCublasLtFp16,
}

impl PrefillMethod {
    /// Returns the stable receipt value for this method.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SequentialDecode => "sequential-decode",
            Self::ChunkedCublasLtFp16 => "chunked-cublaslt-fp16",
            Self::TiledCublasLtFp16 => "tiled-cublaslt-fp16",
        }
    }
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
        })
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

/// The memory limit reported by a backend.
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
    fn prefill_method(&self) -> PrefillMethod {
        PrefillMethod::SequentialDecode
    }

    /// Returns true when chunked prefill can write and read `q8` KV storage.
    fn q8_prefill_supported(&self) -> bool {
        false
    }
    /// Returns bytes owned by tracked allocations and external object counts.
    fn memory_accounting(&self) -> MemoryAccounting;
    /// Reassigns a live allocation. Class peaks retain its earlier attribution.
    fn classify_buffer(
        &mut self,
        buffer: &Self::Buffer,
        class: MemoryClass,
    ) -> Result<(), BackendError>;
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
    fn prepare_prefill(&mut self, _plan: PrefillPlan) -> Result<PrefillWorkspace, BackendError> {
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

    #[test]
    fn memory_tracker_returns_live_bytes_after_drop() {
        let tracker = MemoryTracker::default();
        let allocation = tracker.allocate(MemoryClass::KvCache, 128);
        let duplicate = allocation.duplicate();
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
