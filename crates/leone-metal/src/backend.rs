use crate::ffi::{Context, CopyRegion, DeviceInfo, DispatchArgs};
use crate::SHADER_SOURCE;
use leone::backend::{
    MemoryAllocation, MemoryError, MemoryTracker, MemoryTrackerRoot, PrefillNumerics,
    UntrackedMemory,
};
use leone::MemoryReservation;
use leone::{
    AttentionDecodeRow, AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot,
    BufferStorage, DecodeOp, DecodeProfile, Determinism, KvReadSpan, KvReadView, KvWriteSpan,
    MemoryAccounting, MemoryBudget, MemoryCapacity, MemoryClass, ModelImportMetrics, Position,
    PrefillMethod, PrefillPlan, PrefillWorkspace, QuantFormat, QuantMatrix, RopePairing, RopeShape,
    VectorShape,
};
use std::collections::VecDeque;
use std::ffi::CString;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const OP_GEMV: u32 = 1;
const OP_GEMV_RESIDUAL: u32 = 2;
const OP_PREFILL_GEMM: u32 = 3;
const OP_RMS_NORM: u32 = 4;
const OP_RMS_NORM_ROPE: u32 = 5;
const OP_RMS_NORM_RESIDUAL: u32 = 6;
const OP_RMS_NORM_RESIDUAL_STORE: u32 = 7;
const OP_ROPE: u32 = 8;
const OP_SWIGLU: u32 = 9;
const OP_RESIDUAL_ADD: u32 = 10;
const OP_KV_APPEND: u32 = 11;
const OP_KV_APPEND_CHUNK: u32 = 12;
const OP_ATTENTION_DECODE: u32 = 13;
const OP_ATTENTION_PREFILL: u32 = 14;
const OP_EMBED_GATHER: u32 = 15;
const OP_EMBED_GATHER_BATCH: u32 = 16;
const OP_COPY_ROW: u32 = 17;
const OP_WRITE_ROW: u32 = 18;
const OP_ARGMAX: u32 = 19;
const OP_INCREMENT: u32 = 20;
const OP_ATTENTION_DECODE_SPANS: u32 = 21;
const OP_ATTENTION_PREFILL_SPANS: u32 = 22;
const OP_KV_COPY_SPAN: u32 = 23;
const OP_ATTENTION_BATCH_FIXED_TILE: u32 = 24;
const OP_ATTENTION_BATCH_SHARED: u32 = 25;
const OP_ATTENTION_BATCH_FIXED_SHARED: u32 = 26;
const OP_PREFILL_RMS_NORM_ROPE: u32 = 27;
const MAX_DIRECT_KV_SPANS: usize = 4;
const SPAN_DESCRIPTOR_WORDS: usize = 3;
const REDUCTION_LANES: usize = 256;
const MAX_KV_COPY_REGIONS: usize = 65_536;
const ATTENTION_TILE_TOKENS: usize = 32;
const ATTENTION_MAX_HEAD_DIM: usize = 128;
const ATTENTION_MAX_BATCH_ROWS: usize = 1_024;
const RESEARCH_BATCH_SIZE: usize = 8;
const PRODUCTION_BATCH_SIZE: usize = 1;
const PREFILL_TOKEN_TILE: usize = 4;
const PREFILL_TOKEN_TILE_THRESHOLD: usize = 9;
const PREFILL_MATRIX_TOKEN_TILE: usize = 16;
const PREFILL_MATRIX_ROW_TILE: usize = 32;
const PREFILL_MATRIX_K_TILE: usize = 32;

static NEXT_BACKEND_ID: AtomicU64 = AtomicU64::new(1);

/// Reports the limits and current allocation state of one Metal device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    pub max_buffer_length: u64,
    pub recommended_working_set: u64,
    pub current_allocated: u64,
    pub supports_simdgroup_matrix: bool,
}

/// Identifies the active Metal device and shader compilation inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceMetadata {
    pub registry_id: u64,
    pub device_name: String,
    pub architecture_name: String,
    pub os_version: String,
    pub compiler_version: Option<String>,
    pub shader_source_hash: String,
    pub fast_math_enabled: bool,
}

/// Selects one decode attention implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalAttentionPath {
    /// Dispatches span decode attention once per row.
    PerRow,
    /// Runs one query row per work item over a canonical tile table.
    FixedTilePerRow,
    /// Shares common prefix tile reads, then processes each private KV tail.
    SharedReadUnconstrained,
    /// Shares common prefix tile reads with canonical fixed reductions.
    SharedReadFixedReduction,
}

/// Counts batched decode dispatches and row groups for one Metal backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalAttentionBatchStats {
    /// Number of batch attention calls.
    pub dispatches: usize,
    /// Number of rows submitted to those calls.
    pub rows: usize,
    /// Number of row groups launched by the selected path.
    pub groups: usize,
    /// Number of row groups containing more than one row.
    pub multi_row_groups: usize,
}

impl MetalAttentionPath {
    /// Returns the path identifier used by research manifests.
    pub const fn name(self) -> &'static str {
        match self {
            Self::PerRow => "per_row",
            Self::FixedTilePerRow => "fixed_tile_per_row",
            Self::SharedReadUnconstrained => "shared_read_unconstrained",
            Self::SharedReadFixedReduction => "shared_read_fixed_reduction",
        }
    }

    const fn needs_workspace(self) -> bool {
        !matches!(self, Self::PerRow)
    }

    const fn is_shared(self) -> bool {
        matches!(
            self,
            Self::SharedReadUnconstrained | Self::SharedReadFixedReduction
        )
    }
}

impl From<DeviceInfo> for MetalDeviceInfo {
    fn from(info: DeviceInfo) -> Self {
        Self {
            max_buffer_length: info.max_buffer_length,
            recommended_working_set: info.recommended_working_set,
            current_allocated: info.current_allocated,
            supports_simdgroup_matrix: info.supports_simdgroup_matrix != 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillGemmMode {
    Scalar,
    TokenTile,
    SimdgroupMatrix,
}

impl PrefillGemmMode {
    const fn tile_tokens(self) -> usize {
        match self {
            Self::Scalar => 0,
            Self::TokenTile => PREFILL_TOKEN_TILE,
            Self::SimdgroupMatrix => PREFILL_MATRIX_TOKEN_TILE,
        }
    }

    const fn tile_rows(self) -> usize {
        match self {
            Self::Scalar | Self::TokenTile => 0,
            Self::SimdgroupMatrix => PREFILL_MATRIX_ROW_TILE,
        }
    }

    const fn k_tile(self) -> usize {
        match self {
            Self::Scalar | Self::TokenTile => 0,
            Self::SimdgroupMatrix => PREFILL_MATRIX_K_TILE,
        }
    }
}

#[derive(Clone, Copy)]
struct PrefillGemmDispatch {
    shape: QuantMatrix,
    tokens: usize,
    mode: PrefillGemmMode,
}

/// An opaque shared-storage `MTLBuffer` owned by the Metal backend.
pub struct MetalBuffer {
    layout: BufferLayout,
    handle: crate::ffi::Buffer,
    allocation: MemoryAllocation,
    owner: u64,
}

impl fmt::Debug for MetalBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalBuffer")
            .field("layout", &self.layout)
            .field("bytes", &self.layout.bytes())
            .field("class", &self.allocation.class())
            .finish()
    }
}

/// The native Apple-silicon implementation of the backend contract.
pub struct MetalBackend {
    context: Context,
    device: MetalDeviceInfo,
    owner: u64,
    memory: MemoryTracker,
    rope_frequencies: Option<MetalBuffer>,
    rope_pairing: RopePairing,
    attention_batch_stats: MetalAttentionBatchStats,
    batch_kv_gathers: Vec<RetainedKvGather>,
    attention_path: MetalAttentionPath,
    attention_batch_capacity: NonZeroUsize,
    attention_workspaces: Vec<BatchAttentionWorkspace>,
    batch_prefix_gathers: Vec<BatchAttentionGather>,
    prepared_batches: VecDeque<PreparedBatchAttention>,
    prefill_numerics: PrefillNumerics,
    span_descriptors: Option<MetalBuffer>,
    span_gather_key: Option<MetalBuffer>,
    span_gather_value: Option<MetalBuffer>,
}

struct VerifierShapes {
    query_batch: VectorShape,
    key_batch: VectorShape,
    query_rope: RopeShape,
    key_rope: RopeShape,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct KvSpanSignature {
    key_identity: u64,
    value_identity: u64,
    logical_start: usize,
    tokens: usize,
    capacity: usize,
}

struct RetainedKvGather {
    shape: AttentionShape,
    signature: Vec<KvSpanSignature>,
    regions: Vec<CopyRegion>,
    key: MetalBuffer,
    value: MetalBuffer,
}

struct BatchAttentionGather {
    shape: AttentionShape,
    rows: Vec<usize>,
    signatures: Vec<Vec<KvSpanSignature>>,
    shared_end: usize,
    row_offsets: Vec<usize>,
    gather_stride: usize,
    regions: Vec<CopyRegion>,
    key: MetalBuffer,
    value: MetalBuffer,
    tiles: Vec<u32>,
    groups: Vec<u32>,
    row_ids: Vec<u32>,
}

struct PreparedBatchAttention {
    workspaces: Vec<BatchAttentionWorkspace>,
    gathers: Vec<RetainedKvGather>,
    prefix_gathers: Vec<BatchAttentionGather>,
}

struct BatchAttentionWorkspace {
    shape: AttentionShape,
    rows: usize,
    tile_count: usize,
    query: MetalBuffer,
    output: MetalBuffer,
    positions: MetalBuffer,
    tiles: MetalBuffer,
    groups: MetalBuffer,
    row_ids: MetalBuffer,
}

struct AttentionBatchGroup {
    signature: Vec<KvSpanSignature>,
    rows: Vec<usize>,
}

struct AttentionShapeGroup {
    shape: AttentionShape,
    rows: Vec<usize>,
}

type AttentionDescriptorWords = (Vec<u32>, Vec<u32>, Vec<u32>);

impl fmt::Debug for MetalBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalBackend")
            .field("device", &self.device)
            .field("memory", &self.memory.snapshot())
            .finish()
    }
}

impl MetalBackend {
    /// Opens the default Apple GPU and compiles the checked Leone shader.
    pub fn new() -> Result<Self, BackendError> {
        Self::with_memory_budget(MemoryBudget::Unlimited)
    }

    /// Opens the default Apple GPU with an owned allocation budget.
    pub fn with_memory_budget(budget: MemoryBudget) -> Result<Self, BackendError> {
        let source = CString::new(SHADER_SOURCE)
            .map_err(|error| BackendError::operation("prepare Metal shader", error))?;
        let (context, device) = Context::new(&source)
            .map_err(|error| BackendError::operation("initialize Metal", error))?;
        let owner = NEXT_BACKEND_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| BackendError::SizeOverflow {
                field: "Metal backend identity",
            })?;
        Ok(Self {
            context,
            device: device.into(),
            owner,
            memory: MemoryTracker::new(budget),
            rope_frequencies: None,
            rope_pairing: RopePairing::HalfSplit,
            attention_batch_stats: MetalAttentionBatchStats::default(),
            batch_kv_gathers: Vec::new(),
            attention_path: MetalAttentionPath::PerRow,
            attention_batch_capacity: NonZeroUsize::new(PRODUCTION_BATCH_SIZE)
                .expect("production batch size is nonzero"),
            attention_workspaces: Vec::new(),
            batch_prefix_gathers: Vec::new(),
            prepared_batches: VecDeque::new(),
            prefill_numerics: PrefillNumerics::BackendPreferred,
            span_descriptors: None,
            span_gather_key: None,
            span_gather_value: None,
        })
    }

    /// Selects the explicit batch attention path used by subsequent prepares.
    pub fn set_attention_path(&mut self, path: MetalAttentionPath) {
        if self.attention_path != path {
            self.clear_batch_attention_resources();
            self.attention_path = path;
        }
    }

    /// Returns the configured batch attention path.
    pub const fn attention_path(&self) -> MetalAttentionPath {
        self.attention_path
    }

    /// Enables the bounded position-major verifier path used by the research driver.
    ///
    /// The research driver uses this path with speculative decode disabled.
    pub fn set_research_batch_size(&mut self) {
        self.attention_batch_capacity =
            NonZeroUsize::new(RESEARCH_BATCH_SIZE).expect("research batch size is nonzero");
    }

    fn check_research_positions(&self, positions: usize) -> Result<(), BackendError> {
        if !self.verify_supported() {
            return Err(BackendError::operation(
                "run Metal research matrix operation",
                "research batching is disabled",
            ));
        }
        check_nonzero_tokens(positions, "research positions")?;
        if positions > self.attention_batch_capacity.get() {
            return Err(BackendError::operation(
                "run Metal research matrix operation",
                "position count exceeds the research batch limit",
            ));
        }
        Ok(())
    }

    /// Returns batched decode dispatch counts since backend construction.
    pub const fn attention_batch_stats(&self) -> MetalAttentionBatchStats {
        self.attention_batch_stats
    }

    fn record_batch_dispatch(&mut self, rows: usize, groups: usize, multi_row_groups: usize) {
        self.attention_batch_stats.dispatches =
            self.attention_batch_stats.dispatches.saturating_add(1);
        self.attention_batch_stats.rows = self.attention_batch_stats.rows.saturating_add(rows);
        self.attention_batch_stats.groups =
            self.attention_batch_stats.groups.saturating_add(groups);
        self.attention_batch_stats.multi_row_groups = self
            .attention_batch_stats
            .multi_row_groups
            .saturating_add(multi_row_groups);
    }

    /// Returns device limits captured during backend initialization.
    pub const fn device_info(&self) -> MetalDeviceInfo {
        self.device
    }

    /// Returns device identity and the shader source hash used at startup.
    pub fn device_metadata(&self) -> Result<MetalDeviceMetadata, BackendError> {
        let metadata = self
            .context
            .device_metadata()
            .map_err(|error| BackendError::operation("read Metal metadata", error))?;
        Ok(MetalDeviceMetadata {
            registry_id: metadata.registry_id,
            device_name: metadata_text(&metadata.device_name, "Metal device name")?,
            architecture_name: metadata_text(
                &metadata.architecture_name,
                "Metal architecture name",
            )?,
            os_version: metadata_text(&metadata.os_version, "Metal OS version")?,
            compiler_version: metadata_optional_text(
                &metadata.compiler_version,
                "Metal compiler version",
            )?,
            shader_source_hash: metadata_text(
                &metadata.shader_source_hash,
                "Metal shader source hash",
            )?,
            fast_math_enabled: metadata.fast_math_enabled != 0,
        })
    }

    fn allocate_inner(
        &mut self,
        layout: BufferLayout,
        class: MemoryClass,
    ) -> Result<MetalBuffer, BackendError> {
        let bytes = layout.bytes();
        if u64::try_from(bytes).map_err(|_| BackendError::SizeOverflow {
            field: "Metal buffer bytes",
        })? > self.device.max_buffer_length
        {
            return Err(BackendError::operation(
                "allocate Metal buffer",
                "buffer exceeds MTLDevice maxBufferLength",
            ));
        }
        let tracked_bytes = u64::try_from(bytes).map_err(|_| BackendError::SizeOverflow {
            field: "Metal buffer bytes",
        })?;
        let reservation = self.memory.reserve(class, tracked_bytes)?;
        let handle = self
            .context
            .alloc(bytes)
            .map_err(|error| BackendError::operation("allocate Metal buffer", error))?;
        Ok(MetalBuffer {
            layout,
            handle,
            allocation: reservation.commit()?,
            owner: self.owner,
        })
    }

    fn allocate_precharged(
        &mut self,
        layouts: &[BufferLayout],
        class: MemoryClass,
    ) -> Result<Vec<MetalBuffer>, BackendError> {
        let reservations = self.reserve_layouts(layouts, class)?;
        self.materialize_reserved(reservations)
    }

    fn reserve_layouts(
        &self,
        layouts: &[BufferLayout],
        class: MemoryClass,
    ) -> Result<Vec<(BufferLayout, MemoryReservation)>, BackendError> {
        let mut reservations = Vec::new();
        reservations.try_reserve_exact(layouts.len()).map_err(|_| {
            BackendError::operation(
                "prepare Metal attention workspace",
                "descriptor allocation failed",
            )
        })?;
        for layout in layouts {
            self.check_buffer_limit(*layout)?;
            let bytes = u64::try_from(layout.bytes()).map_err(|_| BackendError::SizeOverflow {
                field: "Metal workspace bytes",
            })?;
            reservations.push((
                *layout,
                self.memory
                    .reserve(class, bytes)
                    .map_err(BackendError::from)?,
            ));
        }
        Ok(reservations)
    }

    fn materialize_reserved(
        &mut self,
        reservations: Vec<(BufferLayout, MemoryReservation)>,
    ) -> Result<Vec<MetalBuffer>, BackendError> {
        let mut buffers = Vec::new();
        buffers.try_reserve_exact(reservations.len()).map_err(|_| {
            BackendError::operation(
                "prepare Metal attention workspace",
                "buffer list allocation failed",
            )
        })?;
        for (layout, reservation) in reservations {
            let handle = self.context.alloc(layout.bytes()).map_err(|error| {
                BackendError::operation("allocate Metal attention workspace", error)
            })?;
            buffers.push(MetalBuffer {
                layout,
                handle,
                allocation: reservation.commit()?,
                owner: self.owner,
            });
        }
        Ok(buffers)
    }

    fn check_buffer_limit(&self, layout: BufferLayout) -> Result<(), BackendError> {
        if u64::try_from(layout.bytes()).map_err(|_| BackendError::SizeOverflow {
            field: "Metal workspace bytes",
        })? > self.device.max_buffer_length
        {
            return Err(BackendError::operation(
                "prepare Metal attention workspace",
                "buffer exceeds MTLDevice maxBufferLength",
            ));
        }
        Ok(())
    }

    fn allocate_attention_workspace(
        &mut self,
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
    ) -> Result<BatchAttentionWorkspace, BackendError> {
        let layouts = Self::attention_workspace_layouts(shape, rows, tile_count)?;
        let buffers = self.allocate_precharged(&layouts, MemoryClass::BackendScratch)?;
        Self::workspace_from_buffers(buffers, shape, rows, tile_count)
    }

    fn attention_workspace_layouts(
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
    ) -> Result<[BufferLayout; 6], BackendError> {
        let (query_elements, tile_words, group_words, row_words) =
            Self::attention_workspace_sizes(shape, rows, tile_count)?;
        Ok([
            Self::attention_f32_layout(query_elements)?,
            Self::attention_f32_layout(query_elements)?,
            Self::attention_u32_layout(rows)?,
            Self::attention_u32_layout(tile_words)?,
            Self::attention_u32_layout(group_words)?,
            Self::attention_u32_layout(row_words)?,
        ])
    }

    fn attention_workspace_sizes(
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
    ) -> Result<(usize, usize, usize, usize), BackendError> {
        let query_elements =
            batch_elements(rows, shape.query_elements()?, "batch attention query")?;
        let tile_words = shape
            .n_head()
            .checked_mul(tile_count)
            .and_then(|value| value.checked_mul(rows.checked_add(1)?))
            .and_then(|value| value.checked_mul(6))
            .ok_or(BackendError::SizeOverflow {
                field: "batch attention tile descriptors",
            })?;
        let group_words = shape
            .n_head()
            .checked_mul(6)
            .ok_or(BackendError::SizeOverflow {
                field: "batch attention group descriptors",
            })?;
        let row_words = rows
            .checked_mul(shape.n_head())
            .ok_or(BackendError::SizeOverflow {
                field: "batch attention row descriptors",
            })?;
        Self::check_workspace_index_ranges(query_elements, tile_words, group_words, row_words)?;
        Ok((query_elements, tile_words, group_words, row_words))
    }

    fn check_workspace_index_ranges(
        query_elements: usize,
        tile_words: usize,
        group_words: usize,
        row_words: usize,
    ) -> Result<(), BackendError> {
        check_u32_index_range(query_elements, "batch attention query scalar index")?;
        check_u32_index_range(tile_words, "batch attention tile descriptor index")?;
        check_u32_index_range(group_words, "batch attention group descriptor index")?;
        check_u32_index_range(row_words, "batch attention row descriptor index")
    }

    fn attention_f32_layout(elements: usize) -> Result<BufferLayout, BackendError> {
        BufferLayout::f32(elements)
    }

    fn attention_u32_layout(elements: usize) -> Result<BufferLayout, BackendError> {
        BufferLayout::u32(elements)
    }

    fn workspace_from_buffers(
        mut buffers: Vec<MetalBuffer>,
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
    ) -> Result<BatchAttentionWorkspace, BackendError> {
        let row_ids = Self::take_workspace_buffer(&mut buffers, "row descriptor is missing")?;
        let groups = Self::take_workspace_buffer(&mut buffers, "group descriptor is missing")?;
        let tiles = Self::take_workspace_buffer(&mut buffers, "tile descriptor is missing")?;
        let positions = Self::take_workspace_buffer(&mut buffers, "position staging is missing")?;
        let output = Self::take_workspace_buffer(&mut buffers, "output staging is missing")?;
        let query = Self::take_workspace_buffer(&mut buffers, "query staging is missing")?;
        Ok(BatchAttentionWorkspace {
            shape,
            rows,
            tile_count,
            query,
            output,
            positions,
            tiles,
            groups,
            row_ids,
        })
    }

    fn take_workspace_buffer(
        buffers: &mut Vec<MetalBuffer>,
        field: &'static str,
    ) -> Result<MetalBuffer, BackendError> {
        buffers.pop().ok_or(BackendError::operation(
            "prepare Metal attention workspace",
            field,
        ))
    }

    fn check_owner(
        &self,
        buffer: &MetalBuffer,
        operation: &'static str,
    ) -> Result<(), BackendError> {
        if buffer.owner != self.owner {
            return Err(BackendError::operation(
                operation,
                "Metal buffer belongs to another backend",
            ));
        }
        Ok(())
    }

    fn check_dispatch_owners(&self, buffers: &[Option<&MetalBuffer>]) -> Result<(), BackendError> {
        for buffer in buffers.iter().flatten() {
            self.check_owner(buffer, "dispatch Metal kernel")?;
        }
        Ok(())
    }

    fn dispatch(
        &self,
        args: DispatchArgs,
        buffers: [Option<&MetalBuffer>; 12],
    ) -> Result<(), BackendError> {
        self.check_dispatch_owners(&buffers)?;
        let handles = buffers.map(|buffer| buffer.map(|buffer| &buffer.handle));
        self.context
            .dispatch(&args, &handles)
            .map_err(|error| BackendError::operation("dispatch Metal kernel", error))
    }

    fn dispatch_sequence(
        &self,
        commands: &[(DispatchArgs, [Option<&MetalBuffer>; 12])],
    ) -> Result<(), BackendError> {
        if commands.is_empty() {
            return Err(BackendError::operation(
                "dispatch Metal kernel sequence",
                "the sequence has no commands",
            ));
        }
        for (_, buffers) in commands {
            self.check_dispatch_owners(buffers)?;
        }
        let args = commands.iter().map(|(args, _)| *args).collect::<Vec<_>>();
        let buffers = commands
            .iter()
            .map(|(_, buffers)| buffers.map(|buffer| buffer.map(|buffer| &buffer.handle)))
            .collect::<Vec<_>>();
        self.context
            .dispatch_sequence(&args, &buffers)
            .map_err(|error| BackendError::operation("dispatch Metal kernel sequence", error))
    }

    fn dispatch_span(
        &self,
        args: DispatchArgs,
        buffers: [Option<&MetalBuffer>; 13],
    ) -> Result<(), BackendError> {
        self.check_dispatch_owners(&buffers)?;
        let handles = buffers.map(|buffer| buffer.map(|buffer| &buffer.handle));
        self.context
            .dispatch(&args, &handles)
            .map_err(|error| BackendError::operation("dispatch Metal span kernel", error))
    }

    fn args(op: u32, threads: usize) -> Result<DispatchArgs, BackendError> {
        Ok(DispatchArgs {
            op,
            threads: as_u32("Metal dispatch threads", threads)?,
            ..DispatchArgs::default()
        })
    }

    fn rope_args(&self, mut args: DispatchArgs) -> Result<DispatchArgs, BackendError> {
        args.pairing = match self.rope_pairing {
            RopePairing::HalfSplit => 0,
            RopePairing::Adjacent => 1,
        };
        args.table_rows = as_u32(
            "RoPE frequency table rows",
            self.rope_frequencies
                .as_ref()
                .map_or(0, |buffer| buffer.layout.elements()),
        )?;
        Ok(args)
    }

    fn rope_binding(&self) -> Option<&MetalBuffer> {
        self.rope_frequencies.as_ref()
    }

    fn checked_position(
        &mut self,
        position: Position<'_, MetalBuffer>,
    ) -> Result<usize, BackendError> {
        self.resolve_position(position)
    }

    fn check_storage(
        buffer: &MetalBuffer,
        expected: BufferStorage,
        name: &'static str,
    ) -> Result<(), BackendError> {
        if buffer.layout.storage() == expected {
            Ok(())
        } else {
            Err(BackendError::operation(
                name,
                format!("expected {expected:?}, found {:?}", buffer.layout.storage()),
            ))
        }
    }

    fn check_elements(
        buffer: &MetalBuffer,
        expected: usize,
        name: &'static str,
    ) -> Result<(), BackendError> {
        if buffer.layout.elements() == expected {
            Ok(())
        } else {
            Err(BackendError::SizeMismatch {
                name,
                expected,
                actual: buffer.layout.elements(),
            })
        }
    }

    fn check_quant(
        buffer: &MetalBuffer,
        shape: QuantMatrix,
        name: &'static str,
    ) -> Result<(), BackendError> {
        let layout = shape.layout()?;
        let elements =
            shape
                .rows()
                .checked_mul(shape.columns())
                .ok_or(BackendError::SizeOverflow {
                    field: "quantized matrix elements",
                })?;
        as_u32("quantized matrix elements", elements)?;
        if buffer.layout == layout {
            Ok(())
        } else {
            Err(BackendError::operation(
                name,
                format!("expected {layout:?}, found {:?}", buffer.layout),
            ))
        }
    }

    fn quant_format(format: QuantFormat) -> u32 {
        match format {
            QuantFormat::Q4K => 0,
            QuantFormat::Q6K => 1,
        }
    }

    fn configure_quant_args(
        args: &mut DispatchArgs,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        args.rows = as_u32("matrix rows", shape.rows())?;
        args.columns = as_u32("matrix columns", shape.columns())?;
        args.format = Self::quant_format(shape.format());
        Ok(())
    }

    fn gemv_args(shape: QuantMatrix, op: u32) -> Result<DispatchArgs, BackendError> {
        let dispatch_threads = reduction_dispatch_threads(shape.rows(), "GEMV dispatch threads")?;
        let mut args = Self::args(op, dispatch_threads)?;
        Self::configure_quant_args(&mut args, shape)?;
        Ok(args)
    }

    fn as_f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }

    fn as_u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn check_kv_buffers(
        &self,
        key: &MetalBuffer,
        value: &MetalBuffer,
        key_cache: &MetalBuffer,
        value_cache: &MetalBuffer,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        Self::check_kv_storage(key, value, key_cache, value_cache)?;
        as_u32("KV projected elements", shape.projected_kv_elements()?)?;
        let cache_elements = shape.cache_elements()?;
        as_u32("KV cache elements", cache_elements)?;
        Self::check_cache_elements(
            key_cache,
            value_cache,
            cache_elements,
            "KV key cache",
            "KV value cache",
        )
    }

    fn check_attention_buffers(
        &self,
        query: &MetalBuffer,
        key_cache: &MetalBuffer,
        value_cache: &MetalBuffer,
        output: &MetalBuffer,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        Self::check_storage_pair(
            query,
            BufferStorage::F32,
            "attention query",
            output,
            BufferStorage::F32,
            "attention output",
        )?;
        Self::check_storage_pair(
            key_cache,
            BufferStorage::F16,
            "attention key cache",
            value_cache,
            BufferStorage::F16,
            "attention value cache",
        )?;
        let cache_elements = shape.cache_elements()?;
        as_u32("attention query elements", shape.query_elements()?)?;
        as_u32("attention cache elements", cache_elements)?;
        Self::check_cache_elements(
            key_cache,
            value_cache,
            cache_elements,
            "attention key cache",
            "attention value cache",
        )
    }

    fn check_kv_storage(
        key: &MetalBuffer,
        value: &MetalBuffer,
        key_cache: &MetalBuffer,
        value_cache: &MetalBuffer,
    ) -> Result<(), BackendError> {
        Self::check_storage_pair(
            key,
            BufferStorage::F32,
            "KV key",
            value,
            BufferStorage::F32,
            "KV value",
        )?;
        Self::check_storage_pair(
            key_cache,
            BufferStorage::F16,
            "KV key cache",
            value_cache,
            BufferStorage::F16,
            "KV value cache",
        )
    }

    fn check_storage_pair(
        first: &MetalBuffer,
        first_storage: BufferStorage,
        first_name: &'static str,
        second: &MetalBuffer,
        second_storage: BufferStorage,
        second_name: &'static str,
    ) -> Result<(), BackendError> {
        Self::check_storage(first, first_storage, first_name)?;
        Self::check_storage(second, second_storage, second_name)
    }

    fn check_cache_elements(
        key_cache: &MetalBuffer,
        value_cache: &MetalBuffer,
        elements: usize,
        key_name: &'static str,
        value_name: &'static str,
    ) -> Result<(), BackendError> {
        Self::check_elements(key_cache, elements, key_name)?;
        Self::check_elements(value_cache, elements, value_name)
    }

    fn check_embedding_buffers(
        &self,
        table: &MetalBuffer,
        rows: &MetalBuffer,
        output: &MetalBuffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        Self::check_quant(table, shape, "embedding table")?;
        Self::check_storage(rows, BufferStorage::U32, "embedding rows")?;
        Self::check_storage(output, BufferStorage::F32, "embedding output")?;
        Ok(())
    }

    fn check_rope_shape(&self, shape: RopeShape) -> Result<(), BackendError> {
        let expected = shape.head_dim() / 2;
        let actual = self
            .rope_frequencies
            .as_ref()
            .map_or(0, |frequencies| frequencies.layout.elements());
        if expected == actual {
            Ok(())
        } else {
            Err(BackendError::SizeMismatch {
                name: "RoPE inverse frequencies",
                expected,
                actual,
            })
        }
    }

    fn validate_embedding_rows(
        &mut self,
        rows: &MetalBuffer,
        table_rows: usize,
    ) -> Result<(), BackendError> {
        let elements = rows.layout.elements();
        let mut indices = Vec::new();
        indices.try_reserve_exact(elements).map_err(|error| {
            BackendError::operation(
                "validate embedding rows",
                format!("host allocation failed: {error}"),
            )
        })?;
        indices.resize(elements, 0);
        self.read_u32(rows, &mut indices)?;
        for value in indices {
            let row = usize::try_from(value)
                .map_err(|error| BackendError::operation("convert embedding row", error))?;
            if row >= table_rows {
                return Err(BackendError::RowOutOfBounds {
                    row,
                    rows: table_rows,
                });
            }
        }
        Ok(())
    }

    fn check_rms_buffers(
        &self,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        shape: VectorShape,
    ) -> Result<usize, BackendError> {
        let elements = shape.elements()?;
        Self::check_rms_storage(input, weight, output)?;
        Self::check_rms_lengths(input, weight, output, elements, shape.columns())?;
        Ok(elements)
    }

    fn check_rms_storage(
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<(), BackendError> {
        Self::check_storage_pair(
            input,
            BufferStorage::F32,
            "RMSNorm input",
            weight,
            BufferStorage::F32,
            "RMSNorm weight",
        )?;
        Self::check_storage(output, BufferStorage::F32, "RMSNorm output")
    }

    fn check_rms_lengths(
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        elements: usize,
        columns: usize,
    ) -> Result<(), BackendError> {
        Self::check_elements(input, elements, "RMSNorm input")?;
        Self::check_elements(output, elements, "RMSNorm output")?;
        Self::check_elements(weight, columns, "RMSNorm weight")
    }

    fn check_prefill_rms_rope_shape(
        shape: VectorShape,
        rope_shape: RopeShape,
    ) -> Result<(), BackendError> {
        let expected_rows =
            batch_elements(rope_shape.tokens(), rope_shape.heads(), "prefill RoPE rows")?;
        if shape.rows() != expected_rows {
            return Err(BackendError::SizeMismatch {
                name: "prefill RMSNorm RoPE rows",
                expected: expected_rows,
                actual: shape.rows(),
            });
        }
        if shape.columns() != rope_shape.head_dim() {
            return Err(BackendError::SizeMismatch {
                name: "prefill RMSNorm RoPE head_dim",
                expected: rope_shape.head_dim(),
                actual: shape.columns(),
            });
        }
        Ok(())
    }

    fn check_rms_residual_buffers(
        &self,
        left: &MetalBuffer,
        right: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        shape: VectorShape,
    ) -> Result<usize, BackendError> {
        let elements = Self::check_rms_residual_inputs(left, right, weight, shape)?;
        Self::check_storage(output, BufferStorage::F32, "RMSNorm residual output")?;
        Self::check_elements(output, elements, "RMSNorm residual output")?;
        Ok(elements)
    }

    fn check_rms_residual_store_buffers(
        &self,
        left: &MetalBuffer,
        right: &MetalBuffer,
        weight: &MetalBuffer,
        residual: &MetalBuffer,
        output: &MetalBuffer,
        shape: VectorShape,
    ) -> Result<usize, BackendError> {
        let elements = Self::check_rms_residual_inputs(left, right, weight, shape)?;
        Self::check_storage_pair(
            residual,
            BufferStorage::F32,
            "RMSNorm residual store",
            output,
            BufferStorage::F32,
            "RMSNorm residual output",
        )?;
        Self::check_elements(residual, elements, "RMSNorm residual store")?;
        Self::check_elements(output, elements, "RMSNorm residual output")?;
        Ok(elements)
    }

    fn check_rms_residual_inputs(
        left: &MetalBuffer,
        right: &MetalBuffer,
        weight: &MetalBuffer,
        shape: VectorShape,
    ) -> Result<usize, BackendError> {
        let elements = shape.elements()?;
        Self::check_storage_pair(
            left,
            BufferStorage::F32,
            "RMSNorm residual left",
            right,
            BufferStorage::F32,
            "RMSNorm residual right",
        )?;
        Self::check_storage(weight, BufferStorage::F32, "RMSNorm residual weight")?;
        Self::check_elements(left, elements, "RMSNorm residual left")?;
        Self::check_elements(right, elements, "RMSNorm residual right")?;
        Self::check_elements(weight, shape.columns(), "RMSNorm residual weight")?;
        Ok(elements)
    }

    fn rms_args(
        op: u32,
        shape: VectorShape,
        elements: usize,
        epsilon: f32,
    ) -> Result<DispatchArgs, BackendError> {
        as_u32("RMSNorm elements", elements)?;
        let mut args = Self::args(op, elements)?;
        args.rows = as_u32("RMSNorm rows", shape.rows())?;
        args.columns = as_u32("RMSNorm columns", shape.columns())?;
        args.epsilon = epsilon;
        Ok(args)
    }

    fn rms_norm_args(
        shape: VectorShape,
        elements: usize,
        epsilon: f32,
    ) -> Result<DispatchArgs, BackendError> {
        as_u32("RMSNorm elements", elements)?;
        let mut args = Self::args(
            OP_RMS_NORM,
            reduction_dispatch_threads(shape.rows(), "RMSNorm dispatch threads")?,
        )?;
        args.rows = as_u32("RMSNorm rows", shape.rows())?;
        args.columns = as_u32("RMSNorm columns", shape.columns())?;
        args.epsilon = epsilon;
        Ok(args)
    }

    fn rope_dispatch_args(
        &self,
        shape: RopeShape,
        elements: usize,
        position: usize,
        theta: f32,
    ) -> Result<DispatchArgs, BackendError> {
        let rows = shape
            .tokens()
            .checked_mul(shape.heads())
            .ok_or(BackendError::SizeOverflow { field: "RoPE rows" })?;
        as_u32("RoPE elements", elements)?;
        let mut args = Self::args(OP_ROPE, elements / 2)?;
        args.rows = as_u32("RoPE rows", rows)?;
        args.n_head = as_u32("RoPE heads", shape.heads())?;
        args.head_dim = as_u32("RoPE head_dim", shape.head_dim())?;
        args.position = as_u32("RoPE position", position)?;
        args.theta = theta;
        self.rope_args(args)
    }

    fn rms_rope_args(
        &self,
        shape: VectorShape,
        elements: usize,
        position: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::rms_args(OP_RMS_NORM_ROPE, shape, elements, epsilon)?;
        args.head_dim = as_u32("RMSNorm RoPE head_dim", shape.columns())?;
        args.position = as_u32("RoPE position", position)?;
        args.theta = theta;
        self.rope_args(args)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_rms_rope_args(
        &self,
        shape: VectorShape,
        rope_shape: RopeShape,
        elements: usize,
        start_position: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::rms_args(OP_PREFILL_RMS_NORM_ROPE, shape, elements, epsilon)?;
        args.tokens = as_u32("prefill RoPE tokens", rope_shape.tokens())?;
        args.n_head = as_u32("prefill RoPE heads", rope_shape.heads())?;
        args.head_dim = as_u32("prefill RoPE head_dim", rope_shape.head_dim())?;
        args.start_position = as_u32("prefill RoPE start position", start_position)?;
        args.theta = theta;
        self.rope_args(args)
    }

    fn kv_append_args(
        shape: AttentionShape,
        position: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::args(OP_KV_APPEND, shape.projected_kv_elements()?)?;
        args.n_head_kv = as_u32("KV heads", shape.n_head_kv())?;
        args.head_dim = as_u32("KV head_dim", shape.head_dim())?;
        args.max_context = as_u32("KV max context", shape.max_context())?;
        args.start_position = as_u32("KV position", position)?;
        Ok(args)
    }

    fn kv_append_chunk_args(
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
        elements: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::args(OP_KV_APPEND_CHUNK, elements)?;
        args.tokens = as_u32("KV chunk tokens", tokens)?;
        args.n_head_kv = as_u32("KV heads", shape.n_head_kv())?;
        args.head_dim = as_u32("KV head_dim", shape.head_dim())?;
        args.max_context = as_u32("KV max context", shape.max_context())?;
        args.start_position = as_u32("KV chunk start", start_position)?;
        Ok(args)
    }

    fn check_kv_inputs(
        &self,
        key: &MetalBuffer,
        value: &MetalBuffer,
        elements: usize,
        key_name: &'static str,
        value_name: &'static str,
    ) -> Result<(), BackendError> {
        Self::check_elements(key, elements, key_name)?;
        Self::check_elements(value, elements, value_name)
    }

    fn check_attention_io(
        query: &MetalBuffer,
        output: &MetalBuffer,
        elements: usize,
        query_name: &'static str,
        output_name: &'static str,
    ) -> Result<(), BackendError> {
        Self::check_elements(query, elements, query_name)?;
        Self::check_elements(output, elements, output_name)
    }

    fn check_span_cache(
        &self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        if cache.spans().is_empty() {
            return Err(BackendError::operation(
                "read Metal KV spans",
                "the view has no spans",
            ));
        }
        if cache.mapped_tokens() > shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position: cache.mapped_tokens(),
                max_context: shape.max_context(),
            });
        }
        for span in cache.spans() {
            self.check_span_cache_buffer(span, shape)?;
        }
        Ok(())
    }

    fn check_span_cache_buffer(
        &self,
        span: &leone::KvReadSpan<'_, MetalBuffer>,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        self.check_owner(span.key(), "read Metal KV spans")?;
        self.check_owner(span.value(), "read Metal KV spans")?;
        Self::check_storage(span.key(), BufferStorage::F16, "Metal KV key span")?;
        Self::check_storage(span.value(), BufferStorage::F16, "Metal KV value span")?;
        let elements = span
            .capacity_token_count()
            .checked_mul(shape.n_head_kv())
            .and_then(|value| value.checked_mul(shape.head_dim()))
            .ok_or(BackendError::SizeOverflow {
                field: "Metal KV span elements",
            })?;
        check_u32_index_range(elements, "Metal KV span scalar index")?;
        Self::check_elements(span.key(), elements, "Metal KV key span")?;
        Self::check_elements(span.value(), elements, "Metal KV value span")
    }

    fn check_span_write_buffers(
        key: &MetalBuffer,
        value: &MetalBuffer,
        shape: AttentionShape,
        capacity: usize,
    ) -> Result<AttentionShape, BackendError> {
        let physical_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            capacity,
        )?;
        Self::check_storage_pair(
            key,
            BufferStorage::F16,
            "Metal KV write key span",
            value,
            BufferStorage::F16,
            "Metal KV write value span",
        )?;
        let elements = physical_shape.cache_elements()?;
        check_u32_index_range(elements, "Metal KV write span scalar index")?;
        Self::check_cache_elements(
            key,
            value,
            elements,
            "Metal KV write key span",
            "Metal KV write value span",
        )?;
        Ok(physical_shape)
    }

    fn check_span_attention_io(
        query: &MetalBuffer,
        output: &MetalBuffer,
        shape: AttentionShape,
        tokens: usize,
    ) -> Result<usize, BackendError> {
        Self::check_storage_pair(
            query,
            BufferStorage::F32,
            "Metal span attention query",
            output,
            BufferStorage::F32,
            "Metal span attention output",
        )?;
        let elements = batch_elements(
            tokens,
            shape.query_elements()?,
            "Metal span attention elements",
        )?;
        Self::check_attention_io(
            query,
            output,
            elements,
            "Metal span attention query",
            "Metal span attention output",
        )?;
        as_u32("Metal span attention elements", elements)?;
        Ok(elements)
    }

    fn batched_vector_shape(
        shape: VectorShape,
        tokens: usize,
        field: &'static str,
    ) -> Result<VectorShape, BackendError> {
        check_nonzero_tokens(tokens, field)?;
        let rows = shape
            .rows()
            .checked_mul(tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "Metal verifier vector rows",
            })?;
        VectorShape::new(rows, shape.columns())
    }

    fn check_qk_shape(
        shape: VectorShape,
        expected_rows: usize,
        expected_columns: usize,
        name: &'static str,
    ) -> Result<(), BackendError> {
        if shape.rows() != expected_rows {
            return Err(BackendError::SizeMismatch {
                name,
                expected: expected_rows,
                actual: shape.rows(),
            });
        }
        if shape.columns() != expected_columns {
            return Err(BackendError::SizeMismatch {
                name,
                expected: expected_columns,
                actual: shape.columns(),
            });
        }
        Ok(())
    }

    fn verifier_shapes(
        &self,
        query_shape: VectorShape,
        key_shape: VectorShape,
        attention_shape: AttentionShape,
        positions: usize,
    ) -> Result<VerifierShapes, BackendError> {
        Self::check_qk_shape(
            query_shape,
            attention_shape.n_head(),
            attention_shape.head_dim(),
            "verifier query shape",
        )?;
        Self::check_qk_shape(
            key_shape,
            attention_shape.n_head_kv(),
            attention_shape.head_dim(),
            "verifier key shape",
        )?;
        let query_rope = RopeShape::new(positions, query_shape.rows(), query_shape.columns())?;
        let key_rope = RopeShape::new(positions, key_shape.rows(), key_shape.columns())?;
        self.check_rope_shape(query_rope)?;
        self.check_rope_shape(key_rope)?;
        Ok(VerifierShapes {
            query_batch: Self::batched_vector_shape(
                query_shape,
                positions,
                "verifier query positions",
            )?,
            key_batch: Self::batched_vector_shape(key_shape, positions, "verifier key positions")?,
            query_rope,
            key_rope,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn check_verifier_qk_buffers(
        &self,
        query: &MetalBuffer,
        query_weight: &MetalBuffer,
        query_output: &MetalBuffer,
        query_shape: VectorShape,
        key: &MetalBuffer,
        key_weight: &MetalBuffer,
        key_output: &MetalBuffer,
        key_shape: VectorShape,
    ) -> Result<(), BackendError> {
        let query_elements =
            self.check_rms_buffers(query, query_weight, query_output, query_shape)?;
        let key_elements = self.check_rms_buffers(key, key_weight, key_output, key_shape)?;
        as_u32("Metal verifier query elements", query_elements)?;
        as_u32("Metal verifier key elements", key_elements)?;
        Ok(())
    }

    fn check_verifier_value(
        value: &MetalBuffer,
        attention_shape: AttentionShape,
        positions: usize,
    ) -> Result<(), BackendError> {
        let elements = batch_elements(
            positions,
            attention_shape.projected_kv_elements()?,
            "Metal verifier value elements",
        )?;
        Self::check_storage(value, BufferStorage::F32, "Metal verifier value")?;
        Self::check_elements(value, elements, "Metal verifier value")
    }

    fn verifier_cache_parts(
        target: KvWriteSpan<'_, MetalBuffer>,
        attention_shape: AttentionShape,
        start_position: usize,
        positions: usize,
    ) -> Result<(&mut MetalBuffer, &mut MetalBuffer, usize, AttentionShape), BackendError> {
        let local_range = target.local_range(start_position, positions)?;
        let capacity = target.capacity_token_count();
        let (key_cache, value_cache, _, _) = target.into_parts();
        let physical_shape =
            Self::check_span_write_buffers(key_cache, value_cache, attention_shape, capacity)?;
        Ok((key_cache, value_cache, local_range.start, physical_shape))
    }

    #[allow(clippy::too_many_arguments)]
    fn check_verifier_kv_buffers(
        &self,
        key_output: &MetalBuffer,
        value: &MetalBuffer,
        key_cache: &MetalBuffer,
        value_cache: &MetalBuffer,
        shape: AttentionShape,
        local_start: usize,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.check_kv_buffers(key_output, value, key_cache, value_cache, shape)?;
        let elements = batch_elements(
            positions,
            shape.projected_kv_elements()?,
            "Metal verifier KV elements",
        )?;
        self.check_kv_inputs(
            key_output,
            value,
            elements,
            "Metal verifier key",
            "Metal verifier value",
        )?;
        Self::kv_append_chunk_args(shape, local_start, positions, elements)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn check_verifier_parameters(
        &self,
        query_shape: VectorShape,
        key_shape: VectorShape,
        attention_shape: AttentionShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<VerifierShapes, BackendError> {
        check_nonzero_tokens(positions, "verifier KV append positions")?;
        check_context_range(start_position, positions, attention_shape.max_context())?;
        check_rope_position(start_position, positions)?;
        validate_positive("epsilon", epsilon)?;
        validate_positive("RoPE theta", theta)?;
        self.verifier_shapes(query_shape, key_shape, attention_shape, positions)
    }

    fn check_direct_span_count(cache: &KvReadView<'_, MetalBuffer>) -> Result<usize, BackendError> {
        let count = cache.spans().len();
        if count > MAX_DIRECT_KV_SPANS {
            return Err(BackendError::operation(
                "dispatch Metal span attention",
                format!("{count} KV spans exceed the {MAX_DIRECT_KV_SPANS}-span limit"),
            ));
        }
        Ok(count)
    }

    fn kv_copy_span_args(
        shape: AttentionShape,
        logical_start: usize,
        tokens: usize,
        source_capacity: usize,
        destination_capacity: usize,
        elements: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::args(OP_KV_COPY_SPAN, elements)?;
        args.start_position = as_u32("Metal KV copy logical start", logical_start)?;
        args.tokens = as_u32("Metal KV copy tokens", tokens)?;
        args.max_context = as_u32("Metal KV copy destination capacity", destination_capacity)?;
        args.n_head_kv = as_u32("Metal KV copy KV heads", shape.n_head_kv())?;
        args.head_dim = as_u32("Metal KV copy head_dim", shape.head_dim())?;
        args.row = as_u32("Metal KV copy source capacity", source_capacity)?;
        Ok(args)
    }

    fn ensure_span_gather_capacity(
        &mut self,
        shape: AttentionShape,
        capacity: usize,
    ) -> Result<(), BackendError> {
        let gather_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            capacity,
        )?;
        let elements = gather_shape.cache_elements()?;
        let needs_replace = self
            .span_gather_key
            .as_ref()
            .is_none_or(|buffer| buffer.layout.elements() != elements)
            || self
                .span_gather_value
                .as_ref()
                .is_none_or(|buffer| buffer.layout.elements() != elements);
        if needs_replace {
            self.span_gather_key = None;
            self.span_gather_value = None;
            let layout = BufferLayout::f16(elements)?;
            let key = self.allocate_classified(layout, MemoryClass::BackendScratch)?;
            let value = self.allocate_classified(layout, MemoryClass::BackendScratch)?;
            self.span_gather_key = Some(key);
            self.span_gather_value = Some(value);
        }
        Ok(())
    }

    fn dispatch_kv_copy_span(
        &self,
        source: &MetalBuffer,
        destination: &MetalBuffer,
        args: DispatchArgs,
    ) -> Result<(), BackendError> {
        self.dispatch(
            args,
            [
                None,
                None,
                None,
                None,
                None,
                Some(source),
                Some(destination),
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn check_gather_span_inputs(
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
        visible_end: usize,
    ) -> Result<(), BackendError> {
        let projected = shape.projected_kv_elements()?;
        for span in cache.spans() {
            Self::check_gather_span_input(span, projected, visible_end)?;
        }
        Ok(())
    }

    fn check_gather_span_input(
        span: &leone::KvReadSpan<'_, MetalBuffer>,
        projected: usize,
        visible_end: usize,
    ) -> Result<(), BackendError> {
        let copy_tokens = Self::gather_copy_tokens(span, visible_end)?;
        let elements = batch_elements(copy_tokens, projected, "Metal KV span copy elements")?;
        let source_elements = batch_elements(
            span.capacity_token_count(),
            projected,
            "Metal KV span source elements",
        )?;
        as_u32("Metal KV span copy scalar index", elements)?;
        check_u32_index_range(source_elements, "Metal KV span source scalar index")?;
        as_u32("Metal KV span copy logical end", span.mapped_end())?;
        as_u32("Metal KV span copy capacity", span.capacity_token_count())?;
        Ok(())
    }

    fn check_gather_visible_end(
        cache: &KvReadView<'_, MetalBuffer>,
        visible_end: usize,
    ) -> Result<(), BackendError> {
        if visible_end == 0 || visible_end > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: visible_end,
                max_context: cache.mapped_tokens(),
            });
        }
        Ok(())
    }

    fn gather_copy_tokens(
        span: &leone::KvReadSpan<'_, MetalBuffer>,
        visible_end: usize,
    ) -> Result<usize, BackendError> {
        if span.logical_start() >= visible_end {
            return Ok(0);
        }
        span.mapped_end()
            .min(visible_end)
            .checked_sub(span.logical_start())
            .ok_or(BackendError::SizeOverflow {
                field: "Metal KV gather visible span end",
            })
    }

    fn copy_gather_span(
        &self,
        shape: AttentionShape,
        visible_end: usize,
        span: &leone::KvReadSpan<'_, MetalBuffer>,
        key_destination: &MetalBuffer,
        value_destination: &MetalBuffer,
    ) -> Result<(), BackendError> {
        let tokens = Self::gather_copy_tokens(span, visible_end)?;
        if tokens == 0 {
            return Ok(());
        }
        let elements = batch_elements(
            tokens,
            shape.projected_kv_elements()?,
            "Metal KV span copy elements",
        )?;
        let args = Self::kv_copy_span_args(
            shape,
            span.logical_start(),
            tokens,
            span.capacity_token_count(),
            visible_end,
            elements,
        )?;
        self.dispatch_kv_copy_span(span.key(), key_destination, args)?;
        self.dispatch_kv_copy_span(span.value(), value_destination, args)
    }

    fn prepare_span_gather(
        &mut self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
        visible_end: usize,
    ) -> Result<AttentionShape, BackendError> {
        self.check_span_cache(cache, shape)?;
        Self::check_gather_visible_end(cache, visible_end)?;
        let gather_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            visible_end,
        )?;
        check_u32_index_range(
            gather_shape.cache_elements()?,
            "Metal gathered KV scalar index",
        )?;
        Self::check_gather_span_inputs(cache, shape, visible_end)?;
        self.ensure_span_gather_capacity(shape, visible_end)?;
        Ok(gather_shape)
    }

    fn copy_span_gather(
        &self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
        visible_end: usize,
    ) -> Result<(), BackendError> {
        let key_destination = self.span_gather_key.as_ref().ok_or_else(|| {
            BackendError::operation("prepare Metal KV gather", "key scratch missing")
        })?;
        let value_destination = self.span_gather_value.as_ref().ok_or_else(|| {
            BackendError::operation("prepare Metal KV gather", "value scratch missing")
        })?;
        for span in cache.spans() {
            self.copy_gather_span(shape, visible_end, span, key_destination, value_destination)?;
        }
        Ok(())
    }

    fn gather_span_cache(
        &mut self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
        visible_end: usize,
    ) -> Result<AttentionShape, BackendError> {
        let gather_shape = self.prepare_span_gather(cache, shape, visible_end)?;
        self.copy_span_gather(cache, shape, visible_end)?;
        Ok(gather_shape)
    }

    fn prepare_span_descriptors(
        &mut self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
    ) -> Result<usize, BackendError> {
        self.check_span_cache(cache, shape)?;
        let count = Self::check_direct_span_count(cache)?;
        let words = count
            .checked_mul(SPAN_DESCRIPTOR_WORDS)
            .ok_or(BackendError::SizeOverflow {
                field: "Metal KV span descriptors",
            })?;
        let mut descriptors = [0_u32; MAX_DIRECT_KV_SPANS * SPAN_DESCRIPTOR_WORDS];
        Self::fill_span_descriptor_values(cache, &mut descriptors[..words])?;
        self.ensure_span_descriptor_capacity(words)?;
        self.write_span_descriptors(&descriptors[..words])?;
        Ok(count)
    }

    fn fill_span_descriptor_values(
        cache: &KvReadView<'_, MetalBuffer>,
        descriptors: &mut [u32],
    ) -> Result<(), BackendError> {
        for (index, span) in cache.spans().iter().enumerate() {
            let offset = index * SPAN_DESCRIPTOR_WORDS;
            descriptors[offset] = as_u32("Metal KV span logical start", span.logical_start())?;
            descriptors[offset + 1] = as_u32("Metal KV span tokens", span.token_count())?;
            descriptors[offset + 2] =
                as_u32("Metal KV span capacity", span.capacity_token_count())?;
        }
        Ok(())
    }

    fn ensure_span_descriptor_capacity(&mut self, words: usize) -> Result<(), BackendError> {
        let desired_words = words.max(SPAN_DESCRIPTOR_WORDS);
        let replace = self
            .span_descriptors
            .as_ref()
            .is_none_or(|buffer| buffer.layout.elements() != desired_words);
        if replace {
            self.span_descriptors = None;
            self.span_descriptors = Some(self.allocate_classified(
                BufferLayout::u32(desired_words)?,
                MemoryClass::BackendScratch,
            )?);
        }
        Ok(())
    }

    fn write_span_descriptors(&self, descriptors: &[u32]) -> Result<(), BackendError> {
        let descriptor = self.span_descriptors.as_ref().ok_or_else(|| {
            BackendError::operation("prepare Metal KV spans", "descriptor missing")
        })?;
        descriptor
            .handle
            .write(&Self::as_u32_bytes(descriptors))
            .map_err(|error| BackendError::operation("write Metal KV spans", error))
    }

    fn span_dispatch_buffers<'a>(
        cache: &'a KvReadView<'a, MetalBuffer>,
        descriptor: &'a MetalBuffer,
        query: &'a MetalBuffer,
        output: &'a MetalBuffer,
    ) -> Result<[Option<&'a MetalBuffer>; 13], BackendError> {
        let count = Self::check_direct_span_count(cache)?;
        let mut buffers = [None; 13];
        buffers[1] = Some(query);
        buffers[2] = Some(output);
        buffers[3] = Some(descriptor);
        for (index, span) in cache.spans().iter().enumerate() {
            buffers[5 + index * 2] = Some(span.key());
            buffers[6 + index * 2] = Some(span.value());
        }
        if count == 0 {
            return Err(BackendError::operation(
                "dispatch Metal span attention",
                "the view has no spans",
            ));
        }
        Ok(buffers)
    }

    fn check_span_mapped_context(
        cache: &KvReadView<'_, MetalBuffer>,
        start_position: usize,
        tokens: usize,
        field: &'static str,
    ) -> Result<(), BackendError> {
        let context_end = start_position
            .checked_add(tokens)
            .ok_or(BackendError::SizeOverflow { field })?;
        if context_end > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: context_end,
                max_context: cache.mapped_tokens(),
            });
        }
        Ok(())
    }

    fn dispatch_gathered_decode(
        &mut self,
        query: &MetalBuffer,
        cache: KvReadView<'_, MetalBuffer>,
        output: &MetalBuffer,
        shape: AttentionShape,
        position: usize,
    ) -> Result<(), BackendError> {
        self.check_owner(query, "read Metal span attention query")?;
        self.check_owner(output, "write Metal span attention output")?;
        let visible_end = position.checked_add(1).ok_or(BackendError::SizeOverflow {
            field: "Metal gathered decode visible end",
        })?;
        let gather_shape = self.gather_span_cache(&cache, shape, visible_end)?;
        let key = self.span_gather_key.as_ref().ok_or_else(|| {
            BackendError::operation("dispatch Metal span attention", "key scratch missing")
        })?;
        let value = self.span_gather_value.as_ref().ok_or_else(|| {
            BackendError::operation("dispatch Metal span attention", "value scratch missing")
        })?;
        self.attention_decode_contiguous(query, key, value, output, gather_shape, position)
    }

    fn dispatch_direct_decode(
        &mut self,
        query: &MetalBuffer,
        cache: KvReadView<'_, MetalBuffer>,
        output: &MetalBuffer,
        shape: AttentionShape,
        position: usize,
    ) -> Result<(), BackendError> {
        self.check_owner(query, "read Metal span attention query")?;
        self.check_owner(output, "write Metal span attention output")?;
        let span_count = self.prepare_span_descriptors(&cache, shape)?;
        let descriptor = self.span_descriptors.as_ref().ok_or_else(|| {
            BackendError::operation("dispatch Metal span attention", "descriptor missing")
        })?;
        let buffers = Self::span_dispatch_buffers(&cache, descriptor, query, output)?;
        let args = Self::attention_decode_span_args(shape, position, span_count)?;
        self.dispatch_span(args, buffers)
    }

    fn dispatch_attention_decode_spans(
        &mut self,
        query: &MetalBuffer,
        cache: KvReadView<'_, MetalBuffer>,
        output: &MetalBuffer,
        shape: AttentionShape,
        position: usize,
    ) -> Result<(), BackendError> {
        if cache.spans().len() > MAX_DIRECT_KV_SPANS {
            return self.dispatch_gathered_decode(query, cache, output, shape, position);
        }
        self.dispatch_direct_decode(query, cache, output, shape, position)
    }

    fn dispatch_gathered_prefill(
        &mut self,
        query: &MetalBuffer,
        cache: KvReadView<'_, MetalBuffer>,
        output: &MetalBuffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        Self::check_span_attention_io(query, output, shape, tokens)?;
        self.check_owner(query, "read Metal span attention query")?;
        self.check_owner(output, "write Metal span attention output")?;
        let visible_end = start_position
            .checked_add(tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "Metal gathered prefill visible end",
            })?;
        let gather_shape = self.gather_span_cache(&cache, shape, visible_end)?;
        let key = self.span_gather_key.as_ref().ok_or_else(|| {
            BackendError::operation("dispatch Metal span attention", "key scratch missing")
        })?;
        let value = self.span_gather_value.as_ref().ok_or_else(|| {
            BackendError::operation("dispatch Metal span attention", "value scratch missing")
        })?;
        self.attention_prefill_contiguous(
            query,
            key,
            value,
            output,
            gather_shape,
            start_position,
            tokens,
        )
    }

    fn dispatch_direct_prefill(
        &mut self,
        query: &MetalBuffer,
        cache: KvReadView<'_, MetalBuffer>,
        output: &MetalBuffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let _ = Self::check_span_attention_io(query, output, shape, tokens)?;
        self.check_owner(query, "read Metal span attention query")?;
        self.check_owner(output, "write Metal span attention output")?;
        let span_count = self.prepare_span_descriptors(&cache, shape)?;
        let descriptor = self.span_descriptors.as_ref().ok_or_else(|| {
            BackendError::operation("dispatch Metal span attention", "descriptor missing")
        })?;
        let buffers = Self::span_dispatch_buffers(&cache, descriptor, query, output)?;
        let args = Self::attention_prefill_span_args(shape, start_position, tokens, span_count)?;
        self.dispatch_span(args, buffers)
    }

    fn dispatch_attention_prefill_spans(
        &mut self,
        query: &MetalBuffer,
        cache: KvReadView<'_, MetalBuffer>,
        output: &MetalBuffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        if cache.spans().len() > MAX_DIRECT_KV_SPANS {
            return self.dispatch_gathered_prefill(
                query,
                cache,
                output,
                shape,
                start_position,
                tokens,
            );
        }
        self.dispatch_direct_prefill(query, cache, output, shape, start_position, tokens)
    }

    fn refresh_span_gather(
        context: &Context,
        cache: &KvReadView<'_, MetalBuffer>,
        gather: &mut RetainedKvGather,
        shape: AttentionShape,
        causal_end: usize,
    ) -> Result<(), BackendError> {
        let signature = Self::span_signatures(cache, shape)?;
        if !Self::same_span_storage(&gather.signature, &signature) {
            return Err(BackendError::operation(
                "refresh Metal KV spans",
                "the prepared KV source changed",
            ));
        }
        if causal_end == 0 || causal_end > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: causal_end,
                max_context: cache.mapped_tokens(),
            });
        }
        gather.regions.clear();
        Self::append_cache_range_copy_regions(
            &mut gather.regions,
            cache,
            0,
            causal_end,
            0,
            shape.max_context(),
            shape,
            &gather.key,
            &gather.value,
        )?;
        context
            .copy_regions(&gather.regions)
            .map_err(|error| BackendError::operation("gather Metal KV spans", error))
    }

    fn attention_decode_args(
        shape: AttentionShape,
        position: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let (threads, tiled) = attention_dispatch_threads(shape, 1, "attention dispatch threads")?;
        let mut args = Self::args(OP_ATTENTION_DECODE, threads)?;
        args.position = as_u32("attention position", position)?;
        args.max_context = as_u32("attention max context", shape.max_context())?;
        args.n_head = as_u32("attention heads", shape.n_head())?;
        args.n_head_kv = as_u32("attention KV heads", shape.n_head_kv())?;
        args.head_dim = as_u32("attention head_dim", shape.head_dim())?;
        args.pairing = u32::from(tiled);
        Ok(args)
    }

    fn attention_decode_span_args(
        shape: AttentionShape,
        position: usize,
        span_count: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::attention_decode_args(shape, position)?;
        args.op = OP_ATTENTION_DECODE_SPANS;
        args.row = as_u32("Metal KV span count", span_count)?;
        Ok(args)
    }

    fn attention_prefill_args(
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let (threads, tiled) =
            attention_dispatch_threads(shape, tokens, "attention prefill dispatch threads")?;
        let mut args = Self::args(OP_ATTENTION_PREFILL, threads)?;
        args.tokens = as_u32("attention prefill tokens", tokens)?;
        args.start_position = as_u32("attention prefill start", start_position)?;
        args.max_context = as_u32("attention max context", shape.max_context())?;
        args.n_head = as_u32("attention heads", shape.n_head())?;
        args.n_head_kv = as_u32("attention KV heads", shape.n_head_kv())?;
        args.head_dim = as_u32("attention head_dim", shape.head_dim())?;
        args.pairing = u32::from(tiled);
        Ok(args)
    }

    fn attention_prefill_span_args(
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
        span_count: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::attention_prefill_args(shape, start_position, tokens)?;
        args.op = OP_ATTENTION_PREFILL_SPANS;
        args.row = as_u32("Metal KV span count", span_count)?;
        Ok(args)
    }

    fn check_batch_attention_rows(
        &self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
    ) -> Result<Vec<AttentionShapeGroup>, BackendError> {
        rows.first().ok_or(BackendError::Zero {
            field: "Metal batch attention rows",
        })?;
        if rows.len() > ATTENTION_MAX_BATCH_ROWS {
            return Err(BackendError::operation(
                "prepare Metal batch attention",
                "the batch row limit was exceeded",
            ));
        }
        for row in rows {
            self.check_batch_attention_row(row)?;
        }
        Self::shape_groups(rows)
    }

    fn shape_groups(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
    ) -> Result<Vec<AttentionShapeGroup>, BackendError> {
        let mut groups = Vec::new();
        groups.try_reserve(rows.len()).map_err(|_| {
            BackendError::operation(
                "prepare Metal batch attention",
                "shape descriptor allocation failed",
            )
        })?;
        for (row_index, row) in rows.iter().enumerate() {
            if let Some(group) = groups
                .iter_mut()
                .find(|group: &&mut AttentionShapeGroup| group.shape == row.shape)
            {
                group.rows.push(row_index);
            } else {
                groups.push(AttentionShapeGroup {
                    shape: row.shape,
                    rows: vec![row_index],
                });
            }
        }
        Ok(groups)
    }

    fn check_batch_attention_row(
        &self,
        row: &AttentionDecodeRow<'_, MetalBuffer>,
    ) -> Result<(), BackendError> {
        self.check_batch_row_buffers(row)?;
        self.check_batch_row_shape(row.shape)?;
        Self::check_batch_position(self, row)
    }

    fn check_batch_row_buffers(
        &self,
        row: &AttentionDecodeRow<'_, MetalBuffer>,
    ) -> Result<(), BackendError> {
        self.check_owner(row.query, "prepare Metal batch query")?;
        self.check_owner(row.output, "prepare Metal batch output")?;
        Self::check_storage(row.query, BufferStorage::F32, "batch attention query")?;
        Self::check_storage(row.output, BufferStorage::F32, "batch attention output")?;
        let elements = row.shape.query_elements()?;
        Self::check_elements(row.query, elements, "batch attention query")?;
        Self::check_elements(row.output, elements, "batch attention output")
    }

    fn check_batch_row_shape(&self, shape: AttentionShape) -> Result<(), BackendError> {
        if self.attention_path.needs_workspace() && shape.head_dim() > ATTENTION_MAX_HEAD_DIM {
            return Err(BackendError::operation(
                "prepare Metal batch attention",
                "the four-path kernels require head_dim at most 128",
            ));
        }
        Ok(())
    }

    fn check_batch_position(
        backend: &Self,
        row: &AttentionDecodeRow<'_, MetalBuffer>,
    ) -> Result<(), BackendError> {
        match row.position {
            Position::Host(position) => {
                check_context_position(position, row.shape.max_context())?;
                let end = position.checked_add(1).ok_or(BackendError::SizeOverflow {
                    field: "Metal batch attention context",
                })?;
                if end > row.cache.mapped_tokens() {
                    return Err(BackendError::PositionOutOfBounds {
                        position: end,
                        max_context: row.cache.mapped_tokens(),
                    });
                }
            }
            Position::Device(buffer) => {
                backend.check_owner(buffer, "prepare Metal batch position")?;
                Self::check_storage(buffer, BufferStorage::U32, "batch attention position")?;
                Self::check_elements(buffer, 1, "batch attention position")?;
                if row.cache.mapped_tokens() == 0 {
                    return Err(BackendError::Zero {
                        field: "batch attention mapped tokens",
                    });
                }
            }
        }
        Ok(())
    }

    fn allocate_attention_workspaces(
        &mut self,
        groups: &[AttentionShapeGroup],
    ) -> Result<Vec<BatchAttentionWorkspace>, BackendError> {
        if !self.attention_path.needs_workspace() {
            return Ok(Vec::new());
        }
        let mut workspaces = Vec::new();
        workspaces.try_reserve_exact(groups.len()).map_err(|_| {
            BackendError::operation(
                "prepare Metal attention workspace",
                "workspace list allocation failed",
            )
        })?;
        for group in groups {
            let tile_count = Self::attention_tile_count(group.shape)?;
            workspaces.push(self.allocate_attention_workspace(
                group.shape,
                group.rows.len(),
                tile_count,
            )?);
        }
        Ok(workspaces)
    }

    fn allocate_batch_gathers(
        &mut self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_groups: &[AttentionShapeGroup],
    ) -> Result<Vec<RetainedKvGather>, BackendError> {
        let mut gathers = Vec::new();
        for shape_group in shape_groups {
            let cache_groups = Self::cache_groups(rows, &shape_group.rows, shape_group.shape)?;
            for cache_group in cache_groups {
                let row = rows
                    .get(cache_group.rows[0])
                    .ok_or(BackendError::operation(
                        "prepare Metal batch attention",
                        "cache group row is outside the batch",
                    ))?;
                self.check_span_cache(&row.cache, shape_group.shape)?;
                gathers.push(self.allocate_span_gather(
                    &row.cache,
                    shape_group.shape,
                    cache_group.signature,
                )?);
            }
        }
        Ok(gathers)
    }

    fn allocate_prefix_gather(
        &mut self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
    ) -> Result<BatchAttentionGather, BackendError> {
        self.check_prefix_caches(rows, selected, shape)?;
        let common_end = Self::common_prefix_end(rows, selected)?;
        let shared_end = common_end / ATTENTION_TILE_TOKENS * ATTENTION_TILE_TOKENS;
        let (row_offsets, stride) = Self::prefix_row_layout(rows, selected, shared_end)?;
        let (key, value) = self.allocate_packed_cache(shape, stride)?;
        let regions = Self::packed_regions(
            rows,
            selected,
            shape,
            shared_end,
            &row_offsets,
            stride.max(1),
            &key,
            &value,
        )?;
        let signatures = selected
            .iter()
            .map(|index| Self::span_signatures(&rows[*index].cache, shape))
            .collect::<Result<Vec<_>, _>>()?;
        let reverse = self.attention_path == MetalAttentionPath::SharedReadUnconstrained;
        let (tiles, groups, row_ids) = Self::prefix_descriptor_words(
            rows,
            selected,
            shape,
            shared_end,
            &row_offsets,
            reverse,
        )?;
        Ok(BatchAttentionGather {
            shape,
            rows: selected.to_vec(),
            signatures,
            shared_end,
            row_offsets,
            gather_stride: stride.max(1),
            regions,
            key,
            value,
            tiles,
            groups,
            row_ids,
        })
    }

    fn check_prefix_caches(
        &self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        for index in selected.iter().copied() {
            let row = rows.get(index).ok_or(BackendError::operation(
                "prepare Metal shared attention",
                "row descriptor index is outside the batch",
            ))?;
            self.check_span_cache(&row.cache, shape)?;
        }
        Ok(())
    }

    fn prefix_row_layout(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shared_end: usize,
    ) -> Result<(Vec<usize>, usize), BackendError> {
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(selected.len()).map_err(|_| {
            BackendError::operation(
                "prepare Metal shared attention",
                "row offset allocation failed",
            )
        })?;
        let mut stride = shared_end;
        for index in selected.iter().copied() {
            offsets.push(stride);
            let mapped = rows
                .get(index)
                .ok_or(BackendError::operation(
                    "prepare Metal shared attention",
                    "row descriptor index is outside the batch",
                ))?
                .cache
                .mapped_tokens();
            stride = stride
                .checked_add(mapped.saturating_sub(shared_end))
                .ok_or(BackendError::SizeOverflow {
                    field: "Metal packed KV cache stride",
                })?;
        }
        Ok((offsets, stride))
    }

    fn allocate_prefix_gathers(
        &mut self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_groups: &[AttentionShapeGroup],
    ) -> Result<Vec<BatchAttentionGather>, BackendError> {
        let mut gathers = Vec::new();
        gathers.try_reserve_exact(shape_groups.len()).map_err(|_| {
            BackendError::operation(
                "prepare Metal shared attention",
                "prefix gather list allocation failed",
            )
        })?;
        for group in shape_groups {
            gathers.push(self.allocate_prefix_gather(rows, &group.rows, group.shape)?);
        }
        Ok(gathers)
    }

    fn prepare_batch_attention_resources(
        &mut self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_groups: &[AttentionShapeGroup],
    ) -> Result<PreparedBatchAttention, BackendError> {
        let workspaces = self.allocate_attention_workspaces(shape_groups)?;
        let (gathers, prefix_gathers) = match self.attention_path {
            MetalAttentionPath::PerRow => (Vec::new(), Vec::new()),
            MetalAttentionPath::FixedTilePerRow => {
                (self.allocate_batch_gathers(rows, shape_groups)?, Vec::new())
            }
            MetalAttentionPath::SharedReadUnconstrained
            | MetalAttentionPath::SharedReadFixedReduction => (
                Vec::new(),
                self.allocate_prefix_gathers(rows, shape_groups)?,
            ),
        };
        Ok(PreparedBatchAttention {
            workspaces,
            gathers,
            prefix_gathers,
        })
    }

    fn cache_groups(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
    ) -> Result<Vec<AttentionBatchGroup>, BackendError> {
        let mut groups = Vec::new();
        for row_index in selected.iter().copied() {
            let row = rows.get(row_index).ok_or(BackendError::operation(
                "prepare Metal batch attention",
                "row descriptor index is outside the batch",
            ))?;
            let signature = Self::span_signatures(&row.cache, shape)?;
            if let Some(group) = groups
                .iter_mut()
                .find(|group: &&mut AttentionBatchGroup| group.signature == signature)
            {
                group.rows.push(row_index);
            } else {
                groups.push(AttentionBatchGroup {
                    signature,
                    rows: vec![row_index],
                });
            }
        }
        Ok(groups)
    }

    fn tile_words(shape: AttentionShape, reverse: bool) -> Result<Vec<u32>, BackendError> {
        let tile_count = Self::attention_tile_count(shape)?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(
                tile_count
                    .checked_mul(6)
                    .ok_or(BackendError::SizeOverflow {
                        field: "batch attention tile descriptors",
                    })?,
            )
            .map_err(|_| {
                BackendError::operation(
                    "prepare Metal attention tiles",
                    "descriptor allocation failed",
                )
            })?;
        for tile in Self::tile_order(tile_count, reverse) {
            words.extend(Self::tile_descriptor(shape, tile)?);
        }
        Ok(words)
    }

    fn attention_tile_count(shape: AttentionShape) -> Result<usize, BackendError> {
        Ok(shape
            .max_context()
            .checked_add(ATTENTION_TILE_TOKENS - 1)
            .ok_or(BackendError::SizeOverflow {
                field: "batch attention tile count",
            })?
            / ATTENTION_TILE_TOKENS)
    }

    fn tile_order(tile_count: usize, reverse: bool) -> Vec<usize> {
        if reverse {
            (0..tile_count).rev().collect()
        } else {
            (0..tile_count).collect()
        }
    }

    fn tile_descriptor(shape: AttentionShape, tile: usize) -> Result<[u32; 6], BackendError> {
        let start = tile
            .checked_mul(ATTENTION_TILE_TOKENS)
            .ok_or(BackendError::SizeOverflow {
                field: "batch attention tile start",
            })?;
        let valid = ATTENTION_TILE_TOKENS.min(shape.max_context() - start);
        let offset = start
            .checked_mul(shape.head_dim())
            .ok_or(BackendError::SizeOverflow {
                field: "batch attention tile offset",
            })?;
        Ok([
            as_u32("batch attention tile start", start)?,
            as_u32("batch attention tile tokens", valid)?,
            as_u32("batch attention key offset", offset)?,
            as_u32("batch attention value offset", offset)?,
            0,
            0,
        ])
    }

    fn shared_descriptor_words(
        shape: AttentionShape,
        rows: usize,
        reverse: bool,
    ) -> Result<AttentionDescriptorWords, BackendError> {
        if rows > 256 {
            return Err(BackendError::operation(
                "dispatch Metal shared attention",
                "one shared group supports at most 256 rows",
            ));
        }
        let base_tiles = Self::tile_words(shape, reverse)?;
        let tile_count = base_tiles.len() / 6;
        let tiles = Self::repeat_tiles(&base_tiles, shape.n_head())?;
        let groups = Self::shared_group_words(shape, rows, tile_count)?;
        let row_ids = Self::shared_row_ids(rows)?;
        Ok((tiles, groups, row_ids))
    }

    fn repeat_tiles(base: &[u32], heads: usize) -> Result<Vec<u32>, BackendError> {
        let mut tiles = Vec::new();
        tiles
            .try_reserve_exact(
                base.len()
                    .checked_mul(heads)
                    .ok_or(BackendError::SizeOverflow {
                        field: "batch attention tile descriptors",
                    })?,
            )
            .map_err(|_| {
                BackendError::operation(
                    "prepare Metal attention tiles",
                    "descriptor allocation failed",
                )
            })?;
        for _ in 0..heads {
            tiles.extend(base);
        }
        Ok(tiles)
    }

    fn shared_group_words(
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
    ) -> Result<Vec<u32>, BackendError> {
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(
                shape
                    .n_head()
                    .checked_mul(6)
                    .ok_or(BackendError::SizeOverflow {
                        field: "batch attention group descriptors",
                    })?,
            )
            .map_err(|_| {
                BackendError::operation(
                    "prepare Metal attention groups",
                    "descriptor allocation failed",
                )
            })?;
        for query_head in 0..shape.n_head() {
            groups.extend([
                0,
                as_u32("batch attention group rows", rows)?,
                as_u32("batch attention query head", query_head)?,
                as_u32(
                    "batch attention KV head",
                    query_head / (shape.n_head() / shape.n_head_kv()),
                )?,
                as_u32("batch attention tile offset", query_head * tile_count)?,
                as_u32("batch attention tile count", tile_count)?,
            ]);
        }
        Ok(groups)
    }

    fn shared_row_ids(rows: usize) -> Result<Vec<u32>, BackendError> {
        (0..rows)
            .map(|row| as_u32("batch attention row", row))
            .collect()
    }

    fn prefix_tile_schedule(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
        shared_end: usize,
        row_offsets: &[usize],
    ) -> Result<Vec<[u32; 6]>, BackendError> {
        let mut schedule = Self::shared_prefix_tiles(shape, shared_end, selected.len())?;
        let tail_capacity = Self::attention_tile_count(shape)?
            .checked_mul(selected.len())
            .ok_or(BackendError::SizeOverflow {
                field: "shared tail tile descriptors",
            })?;
        schedule.try_reserve(tail_capacity).map_err(|_| {
            BackendError::operation(
                "prepare Metal shared attention",
                "tail descriptor allocation failed",
            )
        })?;
        for (local, index) in selected.iter().copied().enumerate() {
            schedule.extend(Self::private_tail_tiles(
                rows,
                index,
                local,
                shape,
                shared_end,
                row_offsets[local],
            )?);
        }
        Ok(schedule)
    }

    fn shared_prefix_tiles(
        shape: AttentionShape,
        shared_end: usize,
        row_count: usize,
    ) -> Result<Vec<[u32; 6]>, BackendError> {
        let tile_count = shared_end.checked_add(ATTENTION_TILE_TOKENS - 1).ok_or(
            BackendError::SizeOverflow {
                field: "shared prefix tile count",
            },
        )? / ATTENTION_TILE_TOKENS;
        let mut tiles = Vec::new();
        tiles.try_reserve_exact(tile_count).map_err(|_| {
            BackendError::operation(
                "prepare Metal shared attention",
                "tile descriptor allocation failed",
            )
        })?;
        let mut start = 0;
        while start < shared_end {
            let valid = (shared_end - start).min(ATTENTION_TILE_TOKENS);
            tiles.push(Self::shared_prefix_tile(shape, start, valid, row_count)?);
            start = start.checked_add(valid).ok_or(BackendError::SizeOverflow {
                field: "shared tile schedule",
            })?;
        }
        Ok(tiles)
    }

    fn shared_prefix_tile(
        shape: AttentionShape,
        start: usize,
        valid: usize,
        row_count: usize,
    ) -> Result<[u32; 6], BackendError> {
        let offset = start
            .checked_mul(shape.head_dim())
            .ok_or(BackendError::SizeOverflow {
                field: "shared tile offset",
            })?;
        Ok([
            as_u32("shared tile start", start)?,
            as_u32("shared tile tokens", valid)?,
            as_u32("shared tile key offset", offset)?,
            as_u32("shared tile value offset", offset)?,
            0,
            as_u32("shared tile row count", row_count)?,
        ])
    }

    fn private_tail_tiles(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        index: usize,
        local: usize,
        shape: AttentionShape,
        shared_end: usize,
        row_offset: usize,
    ) -> Result<Vec<[u32; 6]>, BackendError> {
        let mapped = rows
            .get(index)
            .ok_or(BackendError::operation(
                "prepare Metal shared attention",
                "row descriptor index is outside the batch",
            ))?
            .cache
            .mapped_tokens();
        let tile_count = mapped
            .saturating_sub(shared_end)
            .checked_add(ATTENTION_TILE_TOKENS - 1)
            .ok_or(BackendError::SizeOverflow {
                field: "shared tail tile count",
            })?
            / ATTENTION_TILE_TOKENS;
        let mut tiles = Vec::new();
        tiles.try_reserve_exact(tile_count).map_err(|_| {
            BackendError::operation(
                "prepare Metal shared attention",
                "tail descriptor allocation failed",
            )
        })?;
        let mut tail = shared_end;
        while tail < mapped {
            let valid = (mapped - tail).min(ATTENTION_TILE_TOKENS);
            tiles.push(Self::private_tail_tile(
                shape, tail, valid, local, row_offset, shared_end,
            )?);
            tail = tail.checked_add(valid).ok_or(BackendError::SizeOverflow {
                field: "shared tail tile schedule",
            })?;
        }
        Ok(tiles)
    }

    fn private_tail_tile(
        shape: AttentionShape,
        tail: usize,
        valid: usize,
        local: usize,
        row_offset: usize,
        shared_end: usize,
    ) -> Result<[u32; 6], BackendError> {
        let destination =
            row_offset
                .checked_add(tail - shared_end)
                .ok_or(BackendError::SizeOverflow {
                    field: "shared tail tile offset",
                })?;
        let offset =
            destination
                .checked_mul(shape.head_dim())
                .ok_or(BackendError::SizeOverflow {
                    field: "shared tail tile elements",
                })?;
        Ok([
            as_u32("shared tail tile start", tail)?,
            as_u32("shared tail tile tokens", valid)?,
            as_u32("shared tail key offset", offset)?,
            as_u32("shared tail value offset", offset)?,
            as_u32("shared tail row offset", local)?,
            1,
        ])
    }

    fn prefix_descriptor_words(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
        shared_end: usize,
        row_offsets: &[usize],
        reverse: bool,
    ) -> Result<AttentionDescriptorWords, BackendError> {
        if selected.len() > 256 {
            return Err(BackendError::operation(
                "prepare Metal shared attention",
                "one shared group supports at most 256 rows",
            ));
        }
        let mut schedule =
            Self::prefix_tile_schedule(rows, selected, shape, shared_end, row_offsets)?;
        if reverse {
            schedule.reverse();
        }
        let tile_count = schedule.len();
        let tiles = Self::repeat_prefix_tiles(&schedule, shape.n_head())?;
        let groups = Self::prefix_groups(shape, selected.len(), tile_count)?;
        let row_ids = Self::prefix_row_ids(selected.len(), shape.n_head())?;
        Ok((tiles, groups, row_ids))
    }

    fn repeat_prefix_tiles(schedule: &[[u32; 6]], heads: usize) -> Result<Vec<u32>, BackendError> {
        let words = schedule
            .len()
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(6))
            .ok_or(BackendError::SizeOverflow {
                field: "shared tile descriptors",
            })?;
        let mut tiles = Vec::new();
        tiles.try_reserve_exact(words).map_err(|_| {
            BackendError::operation(
                "prepare Metal shared attention",
                "tile descriptor allocation failed",
            )
        })?;
        for _ in 0..heads {
            for tile in schedule {
                tiles.extend(tile);
            }
        }
        Ok(tiles)
    }

    fn prefix_groups(
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
    ) -> Result<Vec<u32>, BackendError> {
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(
                shape
                    .n_head()
                    .checked_mul(6)
                    .ok_or(BackendError::SizeOverflow {
                        field: "shared group descriptors",
                    })?,
            )
            .map_err(|_| {
                BackendError::operation(
                    "prepare Metal shared attention",
                    "group descriptor allocation failed",
                )
            })?;
        for query_head in 0..shape.n_head() {
            groups.extend(Self::prefix_group(shape, rows, tile_count, query_head)?);
        }
        Ok(groups)
    }

    fn prefix_group(
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
        query_head: usize,
    ) -> Result<[u32; 6], BackendError> {
        let row_offset = query_head
            .checked_mul(rows)
            .ok_or(BackendError::SizeOverflow {
                field: "shared group row offset",
            })?;
        let tile_offset = query_head
            .checked_mul(tile_count)
            .ok_or(BackendError::SizeOverflow {
                field: "shared group tile offset",
            })?;
        Ok([
            as_u32("shared group row offset", row_offset)?,
            as_u32("shared group rows", rows)?,
            as_u32("shared query head", query_head)?,
            as_u32(
                "shared KV head",
                query_head / (shape.n_head() / shape.n_head_kv()),
            )?,
            as_u32("shared group tile offset", tile_offset)?,
            as_u32("shared group tile count", tile_count)?,
        ])
    }

    fn prefix_row_ids(rows: usize, heads: usize) -> Result<Vec<u32>, BackendError> {
        let mut row_ids = Vec::new();
        row_ids
            .try_reserve_exact(rows.checked_mul(heads).ok_or(BackendError::SizeOverflow {
                field: "shared row descriptors",
            })?)
            .map_err(|_| {
                BackendError::operation(
                    "prepare Metal shared attention",
                    "row descriptor allocation failed",
                )
            })?;
        for _ in 0..heads {
            for index in 0..rows {
                row_ids.push(as_u32("shared row", index)?);
            }
        }
        Ok(row_ids)
    }

    fn write_words(buffer: &MetalBuffer, values: &[u32]) -> Result<(), BackendError> {
        if buffer.layout.storage() != BufferStorage::U32 {
            return Err(BackendError::operation(
                "write Metal attention descriptors",
                "descriptor buffer is not u32",
            ));
        }
        if buffer.layout.elements() < values.len() {
            return Err(BackendError::SizeMismatch {
                name: "Metal attention descriptors",
                expected: values.len(),
                actual: buffer.layout.elements(),
            });
        }
        buffer
            .handle
            .write(&Self::as_u32_bytes(values))
            .map_err(|error| BackendError::operation("write Metal attention descriptors", error))
    }

    fn stage_batch_rows(
        &self,
        workspace: &BatchAttentionWorkspace,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
    ) -> Result<Vec<usize>, BackendError> {
        self.stage_query_copies(workspace, rows, selected)?;
        let (positions, device_copies) = Self::position_stage(workspace, rows, selected)?;
        Self::write_words(&workspace.positions, &positions)?;
        if !device_copies.is_empty() {
            self.context.copy_regions(&device_copies).map_err(|error| {
                BackendError::operation("stage Metal attention positions", error)
            })?;
        }
        let positions = Self::read_staged_positions(workspace, selected.len())?;
        Self::validate_staged_positions(rows, selected, &positions)?;
        Ok(positions)
    }

    fn stage_query_copies(
        &self,
        workspace: &BatchAttentionWorkspace,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
    ) -> Result<(), BackendError> {
        let row_bytes = Self::batch_row_bytes(workspace.shape)?;
        let mut copies = Vec::new();
        copies.try_reserve_exact(selected.len()).map_err(|_| {
            BackendError::operation(
                "stage Metal attention queries",
                "copy descriptor allocation failed",
            )
        })?;
        for (local, index) in selected.iter().copied().enumerate() {
            copies.push(CopyRegion::new(
                rows[index].query.handle.raw(),
                workspace.query.handle.raw(),
                0,
                local
                    .checked_mul(row_bytes)
                    .ok_or(BackendError::SizeOverflow {
                        field: "batch attention query staging offset",
                    })?,
                row_bytes,
            ));
        }
        self.context
            .copy_regions(&copies)
            .map_err(|error| BackendError::operation("stage Metal attention queries", error))
    }

    fn batch_row_bytes(shape: AttentionShape) -> Result<usize, BackendError> {
        shape
            .query_elements()?
            .checked_mul(4)
            .ok_or(BackendError::SizeOverflow {
                field: "batch attention row bytes",
            })
    }

    fn position_stage(
        workspace: &BatchAttentionWorkspace,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
    ) -> Result<(Vec<u32>, Vec<CopyRegion>), BackendError> {
        let mut positions = vec![0_u32; selected.len()];
        let mut position_copies = Vec::new();
        for (local, index) in selected.iter().copied().enumerate() {
            match rows[index].position {
                Position::Host(position) => {
                    positions[local] = as_u32("batch attention position", position)?
                }
                Position::Device(buffer) => position_copies.push(CopyRegion::new(
                    buffer.handle.raw(),
                    workspace.positions.handle.raw(),
                    0,
                    local.checked_mul(4).ok_or(BackendError::SizeOverflow {
                        field: "batch attention position offset",
                    })?,
                    4,
                )),
            }
        }
        Ok((positions, position_copies))
    }

    fn read_staged_positions(
        workspace: &BatchAttentionWorkspace,
        count: usize,
    ) -> Result<Vec<usize>, BackendError> {
        let bytes_len = count.checked_mul(4).ok_or(BackendError::SizeOverflow {
            field: "batch attention position bytes",
        })?;
        let mut bytes = vec![0_u8; bytes_len];
        workspace
            .positions
            .handle
            .read(&mut bytes)
            .map_err(|error| BackendError::operation("read Metal attention positions", error))?;
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                usize::try_from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .map_err(|_| BackendError::SizeOverflow {
                        field: "batch attention position",
                    })
            })
            .collect()
    }

    fn validate_staged_positions(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        positions: &[usize],
    ) -> Result<(), BackendError> {
        if positions.len() != selected.len() {
            return Err(BackendError::SizeMismatch {
                name: "batch attention positions",
                expected: selected.len(),
                actual: positions.len(),
            });
        }
        for (local, index) in selected.iter().copied().enumerate() {
            let row = rows.get(index).ok_or(BackendError::operation(
                "stage Metal attention positions",
                "row descriptor index is outside the batch",
            ))?;
            let position = positions[local];
            check_context_position(position, row.shape.max_context())?;
            let end = position.checked_add(1).ok_or(BackendError::SizeOverflow {
                field: "Metal batch attention context",
            })?;
            if end > row.cache.mapped_tokens() {
                return Err(BackendError::PositionOutOfBounds {
                    position: end,
                    max_context: row.cache.mapped_tokens(),
                });
            }
        }
        Ok(())
    }

    fn copy_batch_outputs(
        &self,
        workspace: &BatchAttentionWorkspace,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
    ) -> Result<(), BackendError> {
        let row_bytes = Self::batch_row_bytes(workspace.shape)?;
        let mut copies = Vec::new();
        copies.try_reserve_exact(selected.len()).map_err(|_| {
            BackendError::operation(
                "store Metal attention outputs",
                "copy descriptor allocation failed",
            )
        })?;
        for (local, index) in selected.iter().copied().enumerate() {
            copies.push(CopyRegion::new(
                workspace.output.handle.raw(),
                rows[index].output.handle.raw(),
                local
                    .checked_mul(row_bytes)
                    .ok_or(BackendError::SizeOverflow {
                        field: "batch attention output staging offset",
                    })?,
                0,
                row_bytes,
            ));
        }
        self.context
            .copy_regions(&copies)
            .map_err(|error| BackendError::operation("store Metal attention outputs", error))
    }

    fn batch_attention_args(
        shape: AttentionShape,
        path: MetalAttentionPath,
        rows: usize,
        tile_count: usize,
        groups: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let op = Self::batch_attention_op(path)?;
        let threads = Self::batch_attention_threads(path, rows, groups, shape)?;
        let mut args = Self::args(op, threads)?;
        Self::set_batch_attention_dimensions(&mut args, shape, rows, tile_count, groups)?;
        Self::set_batch_threadgroup_bytes(&mut args, path, shape)?;
        Ok(args)
    }

    fn batch_attention_op(path: MetalAttentionPath) -> Result<u32, BackendError> {
        match path {
            MetalAttentionPath::FixedTilePerRow => Ok(OP_ATTENTION_BATCH_FIXED_TILE),
            MetalAttentionPath::SharedReadUnconstrained => Ok(OP_ATTENTION_BATCH_SHARED),
            MetalAttentionPath::SharedReadFixedReduction => Ok(OP_ATTENTION_BATCH_FIXED_SHARED),
            MetalAttentionPath::PerRow => Err(BackendError::operation(
                "dispatch Metal batch attention",
                "per-row path has no batch descriptor dispatch",
            )),
        }
    }

    fn batch_attention_threads(
        path: MetalAttentionPath,
        rows: usize,
        groups: usize,
        shape: AttentionShape,
    ) -> Result<usize, BackendError> {
        match path {
            MetalAttentionPath::FixedTilePerRow => {
                batch_elements(rows, shape.n_head(), "batch attention threads")
            }
            MetalAttentionPath::SharedReadUnconstrained
            | MetalAttentionPath::SharedReadFixedReduction => {
                batch_elements(groups, 256, "batch attention shared threads")
            }
            MetalAttentionPath::PerRow => Ok(1),
        }
    }

    fn set_batch_attention_dimensions(
        args: &mut DispatchArgs,
        shape: AttentionShape,
        rows: usize,
        tile_count: usize,
        groups: usize,
    ) -> Result<(), BackendError> {
        Self::set_batch_shape_dimensions(args, shape)?;
        Self::set_batch_descriptor_dimensions(args, rows, tile_count, groups)
    }

    fn set_batch_shape_dimensions(
        args: &mut DispatchArgs,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        args.max_context = as_u32("batch attention max context", shape.max_context())?;
        args.gather_stride = args.max_context;
        args.n_head = as_u32("batch attention heads", shape.n_head())?;
        args.n_head_kv = as_u32("batch attention KV heads", shape.n_head_kv())?;
        args.head_dim = as_u32("batch attention head_dim", shape.head_dim())?;
        Ok(())
    }

    fn set_batch_descriptor_dimensions(
        args: &mut DispatchArgs,
        rows: usize,
        tile_count: usize,
        groups: usize,
    ) -> Result<(), BackendError> {
        args.rows = as_u32("batch attention rows", rows)?;
        args.tokens = as_u32("batch attention tile count", tile_count)?;
        args.batch_rows = as_u32("batch attention rows", rows)?;
        args.tile_tokens = as_u32("batch attention tile tokens", ATTENTION_TILE_TOKENS)?;
        args.tile_count = as_u32("batch attention tile count", tile_count)?;
        args.group_count = as_u32("batch attention groups", groups)?;
        Ok(())
    }

    fn set_batch_threadgroup_bytes(
        args: &mut DispatchArgs,
        path: MetalAttentionPath,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        if matches!(
            path,
            MetalAttentionPath::SharedReadUnconstrained
                | MetalAttentionPath::SharedReadFixedReduction
        ) {
            args.threadgroup_bytes = as_u32(
                "batch attention threadgroup bytes",
                ATTENTION_TILE_TOKENS
                    .checked_mul(shape.head_dim())
                    .and_then(|elements| elements.checked_mul(2 * 2))
                    .ok_or(BackendError::SizeOverflow {
                        field: "batch attention threadgroup bytes",
                    })?,
            )?;
        }
        Ok(())
    }

    fn dispatch_batch_group(
        &mut self,
        workspace: &BatchAttentionWorkspace,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        let positions = self.stage_batch_rows(workspace, rows, selected)?;
        let causal_end = positions
            .iter()
            .copied()
            .max()
            .and_then(|position| position.checked_add(1))
            .ok_or(BackendError::Zero {
                field: "batch attention positions",
            })?;
        let gather_index =
            self.refresh_batch_gather(&rows[selected[0]].cache, shape, causal_end)?;
        let gather = &self.batch_kv_gathers[gather_index];
        let (tiles, groups, row_ids) = self.batch_descriptor_words(shape, selected.len())?;
        let group_count = self.write_batch_descriptors(workspace, &tiles, &groups, &row_ids)?;
        let args = Self::batch_attention_args(
            shape,
            self.attention_path,
            selected.len(),
            workspace.tile_count,
            group_count,
        )?;
        self.dispatch_batch_kernel(workspace, gather, args)?;
        self.copy_batch_outputs(workspace, rows, selected)
    }

    fn write_batch_descriptors(
        &self,
        workspace: &BatchAttentionWorkspace,
        tiles: &[u32],
        groups: &[u32],
        row_ids: &[u32],
    ) -> Result<usize, BackendError> {
        Self::write_words(&workspace.tiles, tiles)?;
        if groups.is_empty() {
            return Ok(0);
        }
        Self::write_words(&workspace.groups, groups)?;
        Self::write_words(&workspace.row_ids, row_ids)?;
        Ok(groups.len() / 6)
    }

    fn batch_descriptor_words(
        &self,
        shape: AttentionShape,
        rows: usize,
    ) -> Result<AttentionDescriptorWords, BackendError> {
        match self.attention_path {
            MetalAttentionPath::SharedReadUnconstrained => {
                Self::shared_descriptor_words(shape, rows, true)
            }
            MetalAttentionPath::SharedReadFixedReduction => {
                Self::shared_descriptor_words(shape, rows, false)
            }
            MetalAttentionPath::FixedTilePerRow => {
                Ok((Self::tile_words(shape, false)?, Vec::new(), Vec::new()))
            }
            MetalAttentionPath::PerRow => Err(BackendError::operation(
                "dispatch Metal batch attention",
                "per-row path has no batch descriptor dispatch",
            )),
        }
    }

    fn dispatch_batch_kernel(
        &self,
        workspace: &BatchAttentionWorkspace,
        gather: &RetainedKvGather,
        args: DispatchArgs,
    ) -> Result<(), BackendError> {
        self.dispatch_batch_buffers(workspace, &gather.key, &gather.value, args)
    }

    fn dispatch_batch_buffers(
        &self,
        workspace: &BatchAttentionWorkspace,
        key: &MetalBuffer,
        value: &MetalBuffer,
        args: DispatchArgs,
    ) -> Result<(), BackendError> {
        self.dispatch(
            args,
            [
                None,
                Some(&workspace.query),
                Some(&workspace.output),
                None,
                None,
                Some(key),
                Some(value),
                None,
                Some(&workspace.tiles),
                Some(&workspace.groups),
                Some(&workspace.row_ids),
                Some(&workspace.positions),
            ],
        )
    }

    fn attention_decode_contiguous(
        &self,
        query: &MetalBuffer,
        key_cache: &MetalBuffer,
        value_cache: &MetalBuffer,
        output: &MetalBuffer,
        shape: AttentionShape,
        position: usize,
    ) -> Result<(), BackendError> {
        check_context_position(position, shape.max_context())?;
        self.check_attention_buffers(query, key_cache, value_cache, output, shape)?;
        let elements = shape.query_elements()?;
        Self::check_attention_io(
            query,
            output,
            elements,
            "attention query",
            "attention output",
        )?;
        let args = Self::attention_decode_args(shape, position)?;
        self.dispatch(
            args,
            [
                None,
                Some(query),
                Some(output),
                None,
                None,
                Some(key_cache),
                Some(value_cache),
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_contiguous(
        &self,
        query: &MetalBuffer,
        key_cache: &MetalBuffer,
        value_cache: &MetalBuffer,
        output: &MetalBuffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        check_nonzero_tokens(tokens, "attention prefill tokens")?;
        check_context_range(start_position, tokens, shape.max_context())?;
        self.check_attention_buffers(query, key_cache, value_cache, output, shape)?;
        let elements = batch_elements(
            tokens,
            shape.query_elements()?,
            "attention prefill elements",
        )?;
        Self::check_attention_io(
            query,
            output,
            elements,
            "attention prefill query",
            "attention prefill output",
        )?;
        let args = Self::attention_prefill_args(shape, start_position, tokens)?;
        self.dispatch(
            args,
            [
                None,
                Some(query),
                Some(output),
                None,
                None,
                Some(key_cache),
                Some(value_cache),
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn check_embedding_batch_buffers(
        &mut self,
        table: &MetalBuffer,
        rows: &MetalBuffer,
        output: &MetalBuffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<usize, BackendError> {
        self.check_embedding_buffers(table, rows, output, shape)?;
        Self::check_elements(rows, tokens, "embedding rows")?;
        self.validate_embedding_rows(rows, shape.rows())?;
        let elements = batch_elements(tokens, shape.columns(), "embedding output elements")?;
        Self::check_elements(output, elements, "embedding output")?;
        Ok(elements)
    }

    fn embedding_batch_args(
        shape: QuantMatrix,
        tokens: usize,
        elements: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let mut args = Self::args(OP_EMBED_GATHER_BATCH, elements)?;
        Self::configure_quant_args(&mut args, shape)?;
        args.tokens = as_u32("embedding tokens", tokens)?;
        args.table_rows = as_u32("embedding rows", shape.rows())?;
        Ok(args)
    }

    fn prefill_gemm_args(
        &self,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        dispatch: PrefillGemmDispatch,
    ) -> Result<DispatchArgs, BackendError> {
        let dispatch_threads = self.prefill_dispatch_threads(weights, input, output, dispatch)?;
        Self::configured_prefill_args(dispatch, dispatch_threads)
    }

    fn prefill_dispatch_threads(
        &self,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        dispatch: PrefillGemmDispatch,
    ) -> Result<usize, BackendError> {
        let PrefillGemmDispatch {
            shape,
            tokens,
            mode,
        } = dispatch;
        check_nonzero_tokens(tokens, "prefill tokens")?;
        self.check_prefill_gemm_buffers(weights, input, output, shape, tokens)?;
        let dispatch_groups = prefill_dispatch_groups(shape, tokens, mode)?;
        reduction_dispatch_threads(dispatch_groups, "prefill dispatch threads")
    }

    fn configured_prefill_args(
        dispatch: PrefillGemmDispatch,
        dispatch_threads: usize,
    ) -> Result<DispatchArgs, BackendError> {
        let PrefillGemmDispatch {
            shape,
            tokens,
            mode,
        } = dispatch;
        let mut args = Self::args(OP_PREFILL_GEMM, dispatch_threads)?;
        Self::configure_quant_args(&mut args, shape)?;
        args.tokens = as_u32("prefill tokens", tokens)?;
        args.tile_tokens = as_u32("prefill token tile", mode.tile_tokens())?;
        args.prefill_tile_rows = as_u32("prefill row tile", mode.tile_rows())?;
        args.prefill_k_tile = as_u32("prefill K tile", mode.k_tile())?;
        Ok(args)
    }

    fn dispatch_prefill_gemm(
        &mut self,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        output: &mut MetalBuffer,
        dispatch: PrefillGemmDispatch,
    ) -> Result<(), BackendError> {
        let args = self.prefill_gemm_args(weights, input, output, dispatch)?;
        self.dispatch(
            args,
            [
                Some(weights),
                Some(input),
                Some(output),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn prefill_gemm_mode(&self, tokens: usize) -> PrefillGemmMode {
        select_prefill_gemm_mode_with_numerics(
            self.device.supports_simdgroup_matrix,
            tokens,
            self.prefill_numerics,
        )
    }

    fn check_decode_equivalent_prefill_plan(plan: PrefillPlan) -> Result<(), BackendError> {
        if plan.head_dim() > ATTENTION_MAX_HEAD_DIM || !plan.head_dim().is_multiple_of(2) {
            return Err(BackendError::operation(
                "prepare Metal prefill",
                format!(
                    "decode-equivalent mode requires an even head_dim at most {ATTENTION_MAX_HEAD_DIM}"
                ),
            ));
        }
        let expected_n_embd =
            plan.n_head()
                .checked_mul(plan.head_dim())
                .ok_or(BackendError::SizeOverflow {
                    field: "decode-equivalent embedding width",
                })?;
        if plan.n_embd() != expected_n_embd {
            return Err(BackendError::SizeMismatch {
                name: "decode-equivalent embedding width",
                expected: expected_n_embd,
                actual: plan.n_embd(),
            });
        }
        Ok(())
    }

    fn check_prefill_gemm_buffers(
        &self,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError> {
        Self::check_quant(weights, shape, "prefill weights")?;
        Self::check_storage_pair(
            input,
            BufferStorage::F32,
            "prefill input",
            output,
            BufferStorage::F32,
            "prefill output",
        )?;
        let input_elements = batch_elements(tokens, shape.columns(), "prefill input elements")?;
        check_u32_index_range(input_elements, "prefill input scalar index")?;
        Self::check_elements(input, input_elements, "prefill input")?;
        let output_elements = batch_elements(tokens, shape.rows(), "prefill output elements")?;
        check_u32_index_range(output_elements, "prefill output scalar index")?;
        Self::check_elements(output, output_elements, "prefill output")
    }

    fn upload_rope_frequencies(&mut self, inverse: &[f32]) -> Result<MetalBuffer, BackendError> {
        let layout = BufferLayout::f32(inverse.len())?;
        let buffer = self.allocate_classified(layout, MemoryClass::BackendScratch)?;
        buffer
            .handle
            .write(&Self::as_f32_bytes(inverse))
            .map_err(|error| BackendError::operation("upload RoPE frequencies", error))?;
        Ok(buffer)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_verifier_qk_norm_rope(
        &mut self,
        query: &MetalBuffer,
        query_weight: &MetalBuffer,
        query_output: &mut MetalBuffer,
        key: &MetalBuffer,
        key_weight: &MetalBuffer,
        key_output: &mut MetalBuffer,
        start_position: usize,
        shapes: VerifierShapes,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        self.rms_norm(
            query,
            query_weight,
            query_output,
            shapes.query_batch,
            epsilon,
        )?;
        self.rms_norm(key, key_weight, key_output, shapes.key_batch, epsilon)?;
        self.rope(query_output, start_position, shapes.query_rope, theta)?;
        self.rope(key_output, start_position, shapes.key_rope, theta)
    }

    fn dispatch_batch_per_row(
        &mut self,
        rows: &mut [AttentionDecodeRow<'_, MetalBuffer>],
    ) -> Result<(), BackendError> {
        for row in rows {
            self.attention_decode_spans(row.query, row.cache, row.output, row.shape, row.position)?;
        }
        Ok(())
    }

    fn batch_group_stats(
        &self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_groups: &[AttentionShapeGroup],
    ) -> Result<(usize, usize), BackendError> {
        let mut groups = 0_usize;
        let mut multi_row_groups = 0_usize;
        for shape_group in shape_groups {
            let head_groups = shape_group.shape.n_head();
            if self.attention_path.is_shared() {
                groups = groups.saturating_add(head_groups);
                if shape_group.rows.len() > 1 {
                    multi_row_groups = multi_row_groups.saturating_add(head_groups);
                }
                continue;
            }
            for group in Self::cache_groups(rows, &shape_group.rows, shape_group.shape)? {
                groups = groups.saturating_add(head_groups);
                if group.rows.len() > 1 {
                    multi_row_groups = multi_row_groups.saturating_add(head_groups);
                }
            }
        }
        Ok((groups, multi_row_groups))
    }

    fn dispatch_batch_groups(
        &mut self,
        workspace: &BatchAttentionWorkspace,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_group: &AttentionShapeGroup,
    ) -> Result<(), BackendError> {
        if workspace.shape != shape_group.shape || workspace.rows < shape_group.rows.len() {
            return Err(BackendError::operation(
                "dispatch Metal batch attention",
                "prepared workspace does not match the shape group",
            ));
        }
        if self.attention_path.is_shared() {
            self.dispatch_shared_batch_group(workspace, rows, shape_group)?;
        } else {
            let groups = Self::cache_groups(rows, &shape_group.rows, shape_group.shape)?;
            for group in &groups {
                self.dispatch_batch_group(workspace, rows, &group.rows, shape_group.shape)?;
            }
        }
        Ok(())
    }

    fn dispatch_shared_batch_group(
        &mut self,
        workspace: &BatchAttentionWorkspace,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_group: &AttentionShapeGroup,
    ) -> Result<(), BackendError> {
        let positions = self.stage_batch_rows(workspace, rows, &shape_group.rows)?;
        let gather_index = self.refresh_prefix_gather(rows, shape_group, &positions)?;
        let gather = &self.batch_prefix_gathers[gather_index];
        let group_count = self.write_batch_descriptors(
            workspace,
            &gather.tiles,
            &gather.groups,
            &gather.row_ids,
        )?;
        let stride =
            shape_group
                .shape
                .n_head()
                .checked_mul(6)
                .ok_or(BackendError::SizeOverflow {
                    field: "shared tile descriptor stride",
                })?;
        let tile_count =
            gather
                .tiles
                .len()
                .checked_div(stride)
                .ok_or(BackendError::SizeOverflow {
                    field: "shared tile descriptor count",
                })?;
        let mut args = Self::batch_attention_args(
            shape_group.shape,
            self.attention_path,
            shape_group.rows.len(),
            tile_count,
            group_count,
        )?;
        args.gather_stride = as_u32("shared KV gather stride", gather.gather_stride)?;
        self.dispatch_batch_buffers(workspace, &gather.key, &gather.value, args)?;
        self.copy_batch_outputs(workspace, rows, &shape_group.rows)
    }

    fn dispatch_batch_shape_groups(
        &mut self,
        workspaces: &[BatchAttentionWorkspace],
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_groups: &[AttentionShapeGroup],
    ) -> Result<(), BackendError> {
        for shape_group in shape_groups {
            let workspace = workspaces
                .iter()
                .find(|workspace| workspace.shape == shape_group.shape)
                .ok_or_else(|| {
                    BackendError::operation(
                        "dispatch Metal batch attention",
                        "prepared workspace is missing a shape group",
                    )
                })?;
            self.dispatch_batch_groups(workspace, rows, shape_group)?;
        }
        Ok(())
    }
}

impl MetalBackend {
    fn allocate_packed_cache(
        &mut self,
        shape: AttentionShape,
        stride: usize,
    ) -> Result<(MetalBuffer, MetalBuffer), BackendError> {
        let elements = Self::packed_cache_elements(shape, stride.max(1))?;
        let layout = BufferLayout::f16(elements)?;
        let buffers = self.allocate_precharged(&[layout, layout], MemoryClass::BackendScratch)?;
        let mut buffers = buffers.into_iter();
        let key = buffers.next().ok_or(BackendError::operation(
            "prepare Metal packed KV cache",
            "key buffer is missing",
        ))?;
        let value = buffers.next().ok_or(BackendError::operation(
            "prepare Metal packed KV cache",
            "value buffer is missing",
        ))?;
        Ok((key, value))
    }

    fn allocate_span_cache(&mut self, shape: AttentionShape) -> Result<MetalBuffer, BackendError> {
        let elements = shape.cache_elements()?;
        check_u32_index_range(elements, "Metal span gather scalar index")?;
        let layout = BufferLayout::f16(elements)?;
        self.allocate_classified(layout, MemoryClass::BackendScratch)
    }

    fn allocate_span_gather(
        &mut self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
        signature: Vec<KvSpanSignature>,
    ) -> Result<RetainedKvGather, BackendError> {
        let key = self.allocate_span_cache(shape)?;
        let value = self.allocate_span_cache(shape)?;
        let regions = Self::build_span_copy_regions(cache, &key, &value, shape)?;
        Ok(RetainedKvGather {
            shape,
            signature,
            regions,
            key,
            value,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn append_cache_range_copy_regions(
        regions: &mut Vec<CopyRegion>,
        cache: &KvReadView<'_, MetalBuffer>,
        range_start: usize,
        range_end: usize,
        destination_start: usize,
        destination_stride: usize,
        shape: AttentionShape,
        key_destination: &MetalBuffer,
        value_destination: &MetalBuffer,
    ) -> Result<(), BackendError> {
        if range_start > range_end || range_end > cache.mapped_tokens() {
            return Err(BackendError::operation(
                "prepare Metal KV span copies",
                "cache copy range is outside mapped tokens",
            ));
        }
        for span in cache.spans() {
            let start = range_start.max(span.logical_start());
            let end = range_end.min(span.mapped_end());
            if start < end {
                let destination = destination_start.checked_add(start - range_start).ok_or(
                    BackendError::SizeOverflow {
                        field: "Metal packed KV destination offset",
                    },
                )?;
                Self::append_span_range_copy_regions(
                    regions,
                    span,
                    key_destination,
                    value_destination,
                    shape,
                    start - span.logical_start(),
                    end - start,
                    destination,
                    destination_stride,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_prefix_regions(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
        shared_end: usize,
        row_offsets: &[usize],
        stride: usize,
        ends: &[usize],
        regions: &mut Vec<CopyRegion>,
        key: &MetalBuffer,
        value: &MetalBuffer,
    ) -> Result<(), BackendError> {
        let common_end = ends.iter().copied().max().unwrap_or(0).min(shared_end);
        let first = selected[0];
        Self::append_cache_range_copy_regions(
            regions,
            &rows[first].cache,
            0,
            common_end,
            0,
            stride,
            shape,
            key,
            value,
        )?;
        for (local, index) in selected.iter().copied().enumerate() {
            let start = shared_end.min(ends[local]);
            if start < ends[local] {
                Self::append_cache_range_copy_regions(
                    regions,
                    &rows[index].cache,
                    start,
                    ends[local],
                    row_offsets[local],
                    stride,
                    shape,
                    key,
                    value,
                )?;
            }
        }
        Ok(())
    }

    fn append_span_copy_regions(
        regions: &mut Vec<CopyRegion>,
        span: &KvReadSpan<'_, MetalBuffer>,
        key_destination: &MetalBuffer,
        value_destination: &MetalBuffer,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        Self::append_span_range_copy_regions(
            regions,
            span,
            key_destination,
            value_destination,
            shape,
            0,
            span.token_count(),
            span.logical_start(),
            shape.max_context(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn append_span_range_copy_regions(
        regions: &mut Vec<CopyRegion>,
        span: &KvReadSpan<'_, MetalBuffer>,
        key_destination: &MetalBuffer,
        value_destination: &MetalBuffer,
        shape: AttentionShape,
        source_start: usize,
        tokens: usize,
        destination_start: usize,
        destination_stride: usize,
    ) -> Result<(), BackendError> {
        let source_end = source_start
            .checked_add(tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "Metal KV span source range",
            })?;
        if source_end > span.token_count() {
            return Err(BackendError::operation(
                "prepare Metal KV span copies",
                "copy range exceeds the mapped span",
            ));
        }
        let bytes = tokens
            .checked_mul(shape.head_dim())
            .and_then(|value| value.checked_mul(2))
            .ok_or(BackendError::SizeOverflow {
                field: "Metal KV span copy bytes",
            })?;
        for head in 0..shape.n_head_kv() {
            let source_offset = Self::span_copy_offset(
                head,
                span.capacity_token_count(),
                shape.head_dim(),
                source_start,
            )?;
            let destination_offset = Self::span_copy_offset(
                head,
                destination_stride,
                shape.head_dim(),
                destination_start,
            )?;
            regions.push(CopyRegion::new(
                span.key().handle.raw(),
                key_destination.handle.raw(),
                source_offset,
                destination_offset,
                bytes,
            ));
            regions.push(CopyRegion::new(
                span.value().handle.raw(),
                value_destination.handle.raw(),
                source_offset,
                destination_offset,
                bytes,
            ));
        }
        Ok(())
    }

    fn batch_gather_index(
        &self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
    ) -> Result<usize, BackendError> {
        self.check_span_cache(cache, shape)?;
        let signature = Self::span_signatures(cache, shape)?;
        self.batch_kv_gathers
            .iter()
            .position(|gather| {
                gather.shape == shape
                    && (gather.signature == signature
                        || Self::same_span_storage(&gather.signature, &signature))
            })
            .ok_or_else(|| {
                BackendError::operation(
                    "dispatch Metal batch attention",
                    "KV span gather was not precharged for this batch",
                )
            })
    }

    fn build_span_copy_regions(
        cache: &KvReadView<'_, MetalBuffer>,
        key_destination: &MetalBuffer,
        value_destination: &MetalBuffer,
        shape: AttentionShape,
    ) -> Result<Vec<CopyRegion>, BackendError> {
        let region_count = Self::span_copy_region_count(cache, shape)?;
        if region_count > MAX_KV_COPY_REGIONS {
            return Err(BackendError::operation(
                "prepare Metal KV span copies",
                format!("{region_count} copy regions exceed the bounded descriptor limit"),
            ));
        }
        let mut regions = Vec::new();
        regions.try_reserve(region_count).map_err(|_| {
            BackendError::operation(
                "prepare Metal KV span copies",
                "descriptor allocation failed",
            )
        })?;
        for span in cache.spans() {
            Self::append_span_copy_regions(
                &mut regions,
                span,
                key_destination,
                value_destination,
                shape,
            )?;
        }
        Ok(regions)
    }

    fn check_prefix_signature(
        &self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_group: &AttentionShapeGroup,
        gather: &BatchAttentionGather,
        local: usize,
        index: usize,
    ) -> Result<(), BackendError> {
        let row = rows.get(index).ok_or(BackendError::operation(
            "dispatch Metal shared attention",
            "row descriptor index is outside the batch",
        ))?;
        self.check_span_cache(&row.cache, shape_group.shape)?;
        if !Self::same_prefix_storage(&row.cache, &gather.signatures[local]) {
            return Err(BackendError::operation(
                "refresh Metal shared attention",
                "the prepared KV source changed",
            ));
        }
        Ok(())
    }

    fn clear_batch_attention_resources(&mut self) {
        self.batch_kv_gathers.clear();
        self.attention_workspaces.clear();
        self.batch_prefix_gathers.clear();
        self.prepared_batches.clear();
    }

    fn common_prefix_end(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
    ) -> Result<usize, BackendError> {
        let first = selected.first().copied().ok_or(BackendError::Zero {
            field: "Metal shared attention rows",
        })?;
        let limit = Self::common_prefix_limit(rows, selected, first)?;
        let mut position = 0;
        while position < limit {
            match Self::common_prefix_step(rows, selected, first, position, limit)? {
                Some(next) => position = next,
                None => return Ok(position),
            }
        }
        Ok(limit)
    }

    fn common_prefix_limit(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        first: usize,
    ) -> Result<usize, BackendError> {
        let mut limit = rows
            .get(first)
            .ok_or(BackendError::operation(
                "prepare Metal shared attention",
                "row descriptor index is outside the batch",
            ))?
            .cache
            .mapped_tokens();
        for index in selected.iter().copied().skip(1) {
            limit = limit.min(
                rows.get(index)
                    .ok_or(BackendError::operation(
                        "prepare Metal shared attention",
                        "row descriptor index is outside the batch",
                    ))?
                    .cache
                    .mapped_tokens(),
            );
        }
        Ok(limit)
    }

    fn common_prefix_step(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        first: usize,
        position: usize,
        limit: usize,
    ) -> Result<Option<usize>, BackendError> {
        let reference = rows[first]
            .cache
            .span_for_position(position)
            .map(|(span, _)| span)
            .ok_or(BackendError::operation(
                "prepare Metal shared attention",
                "the first cache has a gap in mapped tokens",
            ))?;
        let mut next = reference.mapped_end().min(limit);
        for index in selected.iter().copied().skip(1) {
            let span = rows[index]
                .cache
                .span_for_position(position)
                .map(|(span, _)| span)
                .ok_or(BackendError::operation(
                    "prepare Metal shared attention",
                    "a cache has a gap in mapped tokens",
                ))?;
            if !Self::same_span_source(reference, span) {
                return Ok(None);
            }
            next = next.min(span.mapped_end());
        }
        if next <= position {
            return Err(BackendError::operation(
                "prepare Metal shared attention",
                "the shared cache span does not advance",
            ));
        }
        Ok(Some(next))
    }

    fn packed_cache_elements(shape: AttentionShape, stride: usize) -> Result<usize, BackendError> {
        let elements = stride
            .checked_mul(shape.n_head_kv())
            .and_then(|value| value.checked_mul(shape.head_dim()))
            .ok_or(BackendError::SizeOverflow {
                field: "Metal packed KV cache elements",
            })?;
        check_u32_index_range(elements, "Metal packed KV scalar index")?;
        Ok(elements)
    }

    fn packed_region_estimate(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
    ) -> Result<usize, BackendError> {
        let first = selected.first().copied().ok_or(BackendError::Zero {
            field: "Metal shared attention rows",
        })?;
        let mut total = 0_usize;
        for index in selected.iter().copied() {
            let row = rows.get(index).ok_or(BackendError::operation(
                "prepare Metal shared attention",
                "row descriptor index is outside the batch",
            ))?;
            total = total
                .checked_add(Self::span_copy_region_count(&row.cache, shape)?)
                .ok_or(BackendError::SizeOverflow {
                    field: "Metal packed KV copy regions",
                })?;
        }
        total = total
            .checked_add(Self::span_copy_region_count(&rows[first].cache, shape)?)
            .ok_or(BackendError::SizeOverflow {
                field: "Metal packed KV copy regions",
            })?;
        if total > MAX_KV_COPY_REGIONS {
            return Err(BackendError::operation(
                "prepare Metal shared attention",
                format!("{total} copy regions exceed the bounded descriptor limit"),
            ));
        }
        Ok(total)
    }

    #[allow(clippy::too_many_arguments)]
    fn packed_regions(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
        shared_end: usize,
        row_offsets: &[usize],
        stride: usize,
        key: &MetalBuffer,
        value: &MetalBuffer,
    ) -> Result<Vec<CopyRegion>, BackendError> {
        let capacity = Self::packed_region_estimate(rows, selected, shape)?;
        let mut regions = Vec::new();
        regions.try_reserve_exact(capacity).map_err(|_| {
            BackendError::operation(
                "prepare Metal shared attention",
                "copy descriptor allocation failed",
            )
        })?;
        let first = selected.first().copied().ok_or(BackendError::Zero {
            field: "Metal shared attention rows",
        })?;
        let common = rows.get(first).ok_or(BackendError::operation(
            "prepare Metal shared attention",
            "row descriptor index is outside the batch",
        ))?;
        Self::append_cache_range_copy_regions(
            &mut regions,
            &common.cache,
            0,
            shared_end,
            0,
            stride,
            shape,
            key,
            value,
        )?;
        for (local, index) in selected.iter().copied().enumerate() {
            let row = rows.get(index).ok_or(BackendError::operation(
                "prepare Metal shared attention",
                "row descriptor index is outside the batch",
            ))?;
            Self::append_cache_range_copy_regions(
                &mut regions,
                &row.cache,
                shared_end,
                row.cache.mapped_tokens(),
                row_offsets[local],
                stride,
                shape,
                key,
                value,
            )?;
        }
        Ok(regions)
    }

    fn prefix_causal_ends(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
        positions: &[usize],
    ) -> Result<Vec<usize>, BackendError> {
        let mut ends = Vec::new();
        ends.try_reserve_exact(positions.len()).map_err(|_| {
            BackendError::operation(
                "refresh Metal shared attention",
                "position descriptor allocation failed",
            )
        })?;
        for (local, index) in selected.iter().copied().enumerate() {
            let row = rows.get(index).ok_or(BackendError::operation(
                "refresh Metal shared attention",
                "row descriptor index is outside the batch",
            ))?;
            let end = positions[local]
                .checked_add(1)
                .ok_or(BackendError::SizeOverflow {
                    field: "shared attention causal end",
                })?;
            if end > row.cache.mapped_tokens() || end > shape.max_context() {
                return Err(BackendError::PositionOutOfBounds {
                    position: end,
                    max_context: row.cache.mapped_tokens().min(shape.max_context()),
                });
            }
            ends.push(end);
        }
        Ok(ends)
    }

    fn refresh_batch_gather(
        &mut self,
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
        causal_end: usize,
    ) -> Result<usize, BackendError> {
        let index = self.batch_gather_index(cache, shape)?;
        let gather = &mut self.batch_kv_gathers[index];
        Self::refresh_span_gather(&self.context, cache, gather, shape, causal_end)?;
        Ok(index)
    }

    fn refresh_prefix_gather(
        &mut self,
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        shape_group: &AttentionShapeGroup,
        positions: &[usize],
    ) -> Result<usize, BackendError> {
        let index = self
            .batch_prefix_gathers
            .iter()
            .position(|gather| gather.shape == shape_group.shape && gather.rows == shape_group.rows)
            .ok_or_else(|| {
                BackendError::operation(
                    "dispatch Metal shared attention",
                    "prefix gather was not precharged for this batch",
                )
            })?;
        let gather = &self.batch_prefix_gathers[index];
        for (local, index) in shape_group.rows.iter().copied().enumerate() {
            self.check_prefix_signature(rows, shape_group, gather, local, index)?;
        }
        let gather = &mut self.batch_prefix_gathers[index];
        Self::rewrite_prefix_regions(
            rows,
            &shape_group.rows,
            shape_group.shape,
            gather.shared_end,
            &gather.row_offsets,
            gather.gather_stride,
            positions,
            &mut gather.regions,
            &gather.key,
            &gather.value,
        )?;
        self.context
            .copy_regions(&gather.regions)
            .map_err(|error| BackendError::operation("gather Metal shared KV spans", error))?;
        Ok(index)
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite_prefix_regions(
        rows: &[AttentionDecodeRow<'_, MetalBuffer>],
        selected: &[usize],
        shape: AttentionShape,
        shared_end: usize,
        row_offsets: &[usize],
        stride: usize,
        positions: &[usize],
        regions: &mut Vec<CopyRegion>,
        key: &MetalBuffer,
        value: &MetalBuffer,
    ) -> Result<(), BackendError> {
        if selected.is_empty() || selected.len() != positions.len() {
            return Err(BackendError::SizeMismatch {
                name: "shared attention positions",
                expected: selected.len(),
                actual: positions.len(),
            });
        }
        if row_offsets.len() != selected.len() {
            return Err(BackendError::SizeMismatch {
                name: "shared attention row offsets",
                expected: selected.len(),
                actual: row_offsets.len(),
            });
        }
        let ends = Self::prefix_causal_ends(rows, selected, shape, positions)?;
        regions.clear();
        Self::append_prefix_regions(
            rows,
            selected,
            shape,
            shared_end,
            row_offsets,
            stride,
            &ends,
            regions,
            key,
            value,
        )?;
        if regions.is_empty() {
            return Err(BackendError::Zero {
                field: "shared attention copy regions",
            });
        }
        Ok(())
    }

    fn same_prefix_storage(
        cache: &KvReadView<'_, MetalBuffer>,
        expected: &[KvSpanSignature],
    ) -> bool {
        cache.spans().len() == expected.len()
            && cache.spans().iter().zip(expected).all(|(span, signature)| {
                span.key().allocation.identity() == signature.key_identity
                    && span.value().allocation.identity() == signature.value_identity
                    && span.logical_start() == signature.logical_start
                    && span.token_count() == signature.tokens
                    && span.capacity_token_count() == signature.capacity
            })
    }

    fn same_span_source(
        left: &KvReadSpan<'_, MetalBuffer>,
        right: &KvReadSpan<'_, MetalBuffer>,
    ) -> bool {
        left.key().allocation.identity() == right.key().allocation.identity()
            && left.value().allocation.identity() == right.value().allocation.identity()
            && left.logical_start() == right.logical_start()
            && left.token_count() == right.token_count()
            && left.capacity_token_count() == right.capacity_token_count()
    }

    fn same_span_storage(left: &[KvSpanSignature], right: &[KvSpanSignature]) -> bool {
        left.len() == right.len()
            && left.iter().zip(right).all(|(left, right)| {
                left.key_identity == right.key_identity
                    && left.value_identity == right.value_identity
                    && left.logical_start == right.logical_start
                    && left.tokens == right.tokens
                    && left.capacity == right.capacity
            })
    }

    fn span_copy_offset(
        head: usize,
        tokens: usize,
        head_dim: usize,
        token_offset: usize,
    ) -> Result<usize, BackendError> {
        head.checked_mul(tokens)
            .and_then(|value| value.checked_add(token_offset))
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(2))
            .ok_or(BackendError::SizeOverflow {
                field: "Metal KV span byte offset",
            })
    }

    fn span_copy_region_count(
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
    ) -> Result<usize, BackendError> {
        cache
            .spans()
            .len()
            .checked_mul(shape.n_head_kv())
            .and_then(|value| value.checked_mul(2))
            .ok_or(BackendError::SizeOverflow {
                field: "Metal KV span copy regions",
            })
    }

    fn span_signatures(
        cache: &KvReadView<'_, MetalBuffer>,
        shape: AttentionShape,
    ) -> Result<Vec<KvSpanSignature>, BackendError> {
        let region_count = Self::span_copy_region_count(cache, shape)?;
        if region_count > MAX_KV_COPY_REGIONS {
            return Err(BackendError::operation(
                "prepare Metal KV spans",
                format!("{region_count} copy regions exceed the bounded descriptor limit"),
            ));
        }
        let mut signatures = Vec::new();
        signatures.try_reserve(cache.spans().len()).map_err(|_| {
            BackendError::operation("prepare Metal KV spans", "descriptor allocation failed")
        })?;
        for span in cache.spans() {
            signatures.push(KvSpanSignature {
                key_identity: span.key().allocation.identity(),
                value_identity: span.value().allocation.identity(),
                logical_start: span.logical_start(),
                tokens: span.token_count(),
                capacity: span.capacity_token_count(),
            });
        }
        Ok(signatures)
    }
}

impl Backend for MetalBackend {
    type Buffer = MetalBuffer;

    fn name(&self) -> &'static str {
        "metal"
    }

    fn determinism(&self) -> Determinism {
        Determinism::FixedOrder
    }

    fn max_batch_size(&self) -> NonZeroUsize {
        self.attention_batch_capacity
    }

    fn prefill_method(&self) -> PrefillMethod {
        PrefillMethod::ChunkedGpu
    }

    fn q8_prefill_supported(&self) -> bool {
        false
    }

    fn decode_equivalent_prefill_supported(&self) -> bool {
        true
    }

    fn memory_accounting(&self) -> MemoryAccounting {
        self.memory.snapshot().with_untracked(UntrackedMemory {
            execution_streams: 1,
            ..UntrackedMemory::default()
        })
    }

    fn memory_tracker_root(&self) -> MemoryTrackerRoot {
        self.memory.root()
    }

    fn classify_buffer(
        &mut self,
        buffer: &Self::Buffer,
        class: MemoryClass,
    ) -> Result<(), BackendError> {
        self.check_owner(buffer, "classify Metal buffer")?;
        buffer.allocation.reclassify(class);
        Ok(())
    }

    fn set_memory_tracker(&mut self, tracker: MemoryTracker) -> Result<(), BackendError> {
        if !self.memory.is_empty() {
            return Err(MemoryError::TrackerInUse {
                owned: self.memory.owned_bytes(),
                reserved: self.memory.reserved_bytes(),
            }
            .into());
        }
        self.memory = tracker;
        Ok(())
    }

    fn set_memory_budget(&mut self, budget: MemoryBudget) -> Result<(), BackendError> {
        self.memory.set_budget(budget)?;
        Ok(())
    }

    fn allocate_classified(
        &mut self,
        layout: BufferLayout,
        class: MemoryClass,
    ) -> Result<Self::Buffer, BackendError> {
        self.allocate_inner(layout, class)
    }

    fn memory_capacity(&mut self) -> Result<MemoryCapacity, BackendError> {
        if let Ok(observed) = self.context.device_info() {
            self.device = observed.into();
        }
        if self.device.recommended_working_set == 0 {
            return Ok(MemoryCapacity::Unbounded);
        }
        let tracked = self.memory.snapshot().live_bytes;
        Ok(MemoryCapacity::Limited {
            available_bytes: self
                .device
                .recommended_working_set
                .saturating_sub(self.device.current_allocated.max(tracked)),
            total_bytes: self.device.recommended_working_set,
        })
    }

    fn model_import_metrics(&self) -> ModelImportMetrics {
        ModelImportMetrics::default()
    }

    fn allocate(&mut self, layout: BufferLayout) -> Result<Self::Buffer, BackendError> {
        self.allocate_classified(layout, MemoryClass::ContractBuffer)
    }

    fn upload(&mut self, layout: BufferLayout, bytes: &[u8]) -> Result<Self::Buffer, BackendError> {
        if bytes.len() != layout.bytes() {
            return Err(BackendError::SizeMismatch {
                name: "uploaded bytes",
                expected: layout.bytes(),
                actual: bytes.len(),
            });
        }
        let buffer = self.allocate_classified(layout, MemoryClass::ModelWeight)?;
        buffer
            .handle
            .write(bytes)
            .map_err(|error| BackendError::operation("upload Metal buffer", error))?;
        Ok(buffer)
    }

    fn clone_buffer(&mut self, source: &Self::Buffer) -> Result<Self::Buffer, BackendError> {
        self.check_owner(source, "clone Metal buffer")?;
        let destination = self.allocate_classified(source.layout, source.allocation.class())?;
        self.context
            .copy(&source.handle, &destination.handle, source.layout.bytes())
            .map_err(|error| BackendError::operation("clone Metal buffer", error))?;
        Ok(destination)
    }

    fn download_buffer(&mut self, source: &Self::Buffer) -> Result<BufferSnapshot, BackendError> {
        self.check_owner(source, "download Metal buffer")?;
        let byte_count = source.layout.bytes();
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_count).map_err(|error| {
            BackendError::operation(
                "download Metal buffer",
                format!("host allocation failed: {error}"),
            )
        })?;
        bytes.resize(byte_count, 0);
        source
            .handle
            .read(&mut bytes)
            .map_err(|error| BackendError::operation("download Metal buffer", error))?;
        BufferSnapshot::new(source.layout, bytes)
    }

    fn restore_buffer(&mut self, source: &BufferSnapshot) -> Result<Self::Buffer, BackendError> {
        self.restore_buffer_classified(source, MemoryClass::ContractBuffer)
    }

    fn restore_buffer_classified(
        &mut self,
        source: &BufferSnapshot,
        class: MemoryClass,
    ) -> Result<Self::Buffer, BackendError> {
        let buffer = self.allocate_classified(source.layout(), class)?;
        buffer
            .handle
            .write(source.bytes())
            .map_err(|error| BackendError::operation("restore Metal buffer", error))?;
        Ok(buffer)
    }

    fn configure_rope(
        &mut self,
        head_dim: usize,
        theta: f32,
        frequency_factors: Option<&[f32]>,
        pairing: RopePairing,
    ) -> Result<(), BackendError> {
        let inverse = build_rope_frequencies(head_dim, theta, frequency_factors)?;
        let buffer = self.upload_rope_frequencies(&inverse)?;
        self.rope_frequencies = Some(buffer);
        self.rope_pairing = pairing;
        Ok(())
    }

    fn prepare_rope(&mut self, _position: Position<'_, Self::Buffer>) -> Result<(), BackendError> {
        Ok(())
    }

    fn write_u32(&mut self, buffer: &mut Self::Buffer, values: &[u32]) -> Result<(), BackendError> {
        self.check_owner(buffer, "write u32")?;
        Self::check_storage(buffer, BufferStorage::U32, "write u32")?;
        Self::check_elements(buffer, values.len(), "u32 write")?;
        buffer
            .handle
            .write(&Self::as_u32_bytes(values))
            .map_err(|error| BackendError::operation("write Metal u32", error))
    }

    fn read_u32(&mut self, buffer: &Self::Buffer, values: &mut [u32]) -> Result<(), BackendError> {
        self.check_owner(buffer, "read u32")?;
        Self::check_storage(buffer, BufferStorage::U32, "read u32")?;
        Self::check_elements(buffer, values.len(), "u32 read")?;
        buffer
            .handle
            .read_u32(values)
            .map_err(|error| BackendError::operation("read Metal u32", error))?;
        Ok(())
    }

    fn read_f16(&mut self, buffer: &Self::Buffer, values: &mut [u16]) -> Result<(), BackendError> {
        self.check_owner(buffer, "read f16")?;
        Self::check_storage(buffer, BufferStorage::F16, "read f16")?;
        Self::check_elements(buffer, values.len(), "f16 read")?;
        buffer
            .handle
            .read_f16(values)
            .map_err(|error| BackendError::operation("read Metal f16", error))?;
        Ok(())
    }

    fn read_f32(&mut self, buffer: &Self::Buffer, values: &mut [f32]) -> Result<(), BackendError> {
        self.check_owner(buffer, "read f32")?;
        Self::check_storage(buffer, BufferStorage::F32, "read f32")?;
        Self::check_elements(buffer, values.len(), "f32 read")?;
        buffer
            .handle
            .read_f32(values)
            .map_err(|error| BackendError::operation("read Metal f32", error))?;
        Ok(())
    }

    fn prepare_prefill(&mut self, plan: PrefillPlan) -> Result<PrefillWorkspace, BackendError> {
        self.prefill_numerics = PrefillNumerics::BackendPreferred;
        let numerics = plan.numerics();
        if numerics == PrefillNumerics::DecodeEquivalent {
            Self::check_decode_equivalent_prefill_plan(plan)?;
        }
        let _ =
            plan.chunk_tokens()
                .checked_mul(plan.n_embd())
                .ok_or(BackendError::SizeOverflow {
                    field: "Metal prefill activation elements",
                })?;
        self.prefill_numerics = numerics;
        Ok(PrefillWorkspace::default())
    }

    fn prefill_gemm(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let mode = self.prefill_gemm_mode(tokens);
        self.dispatch_prefill_gemm(
            weights,
            input,
            output,
            PrefillGemmDispatch {
                shape,
                tokens,
                mode,
            },
        )
    }

    fn verify_supported(&self) -> bool {
        self.attention_batch_capacity.get() == RESEARCH_BATCH_SIZE
    }

    fn verify_gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.check_research_positions(positions)?;
        self.dispatch_prefill_gemm(
            weights,
            input,
            output,
            PrefillGemmDispatch {
                shape,
                tokens: positions,
                mode: PrefillGemmMode::Scalar,
            },
        )
    }

    fn verify_gemv_residual(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.check_research_positions(positions)?;
        self.check_owner(residual, "research GEMV residual")?;
        Self::check_storage(residual, BufferStorage::F32, "research GEMV residual")?;
        let elements = batch_elements(positions, shape.rows(), "research GEMV outputs")?;
        Self::check_elements(residual, elements, "research GEMV residual")?;
        let args = Self::args(OP_RESIDUAL_ADD, elements)?;
        self.dispatch_prefill_gemm(
            weights,
            input,
            output,
            PrefillGemmDispatch {
                shape,
                tokens: positions,
                mode: PrefillGemmMode::Scalar,
            },
        )?;
        // Each invocation reads and writes one element. `dispatch` waits for
        // the product command buffer before this aliased dispatch.
        self.dispatch(
            args,
            [
                None,
                Some(output),
                Some(output),
                Some(residual),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        check_gemv_buffers(weights, input, output, shape)?;
        let args = Self::gemv_args(shape, OP_GEMV)?;
        self.dispatch(
            args,
            [
                Some(weights),
                Some(input),
                Some(output),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn gemv_residual(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        Self::check_storage(residual, BufferStorage::F32, "GEMV residual")?;
        Self::check_elements(residual, shape.rows(), "GEMV residual")?;
        check_gemv_buffers(weights, input, output, shape)?;
        let args = Self::gemv_args(shape, OP_GEMV_RESIDUAL)?;
        self.dispatch(
            args,
            [
                Some(weights),
                Some(input),
                Some(output),
                Some(residual),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

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
    ) -> Result<(), BackendError> {
        check_gemv_buffers(query_weights, input, query, query_shape)?;
        check_gemv_buffers(key_weights, input, key, key_shape)?;
        check_gemv_buffers(value_weights, input, value, value_shape)?;
        let commands = [
            (
                Self::gemv_args(query_shape, OP_GEMV)?,
                [
                    Some(query_weights),
                    Some(input),
                    Some(query),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            ),
            (
                Self::gemv_args(key_shape, OP_GEMV)?,
                [
                    Some(key_weights),
                    Some(input),
                    Some(key),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            ),
            (
                Self::gemv_args(value_shape, OP_GEMV)?,
                [
                    Some(value_weights),
                    Some(input),
                    Some(value),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            ),
        ];
        self.dispatch_sequence(&commands)
    }

    fn rms_norm(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        validate_positive("epsilon", epsilon)?;
        let elements = self.check_rms_buffers(input, weight, output, shape)?;
        let args = Self::rms_norm_args(shape, elements, epsilon)?;
        self.dispatch(
            args,
            [
                None,
                Some(input),
                Some(output),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(weight),
                None,
            ],
        )
    }

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
        if self.prefill_numerics != PrefillNumerics::DecodeEquivalent {
            self.prefill_rms_norm(input, weight, output, shape, epsilon)?;
            return self.rope(output, start_position, rope_shape, theta);
        }
        validate_positive("epsilon", epsilon)?;
        validate_positive("RoPE theta", theta)?;
        check_rope_position(start_position, rope_shape.tokens())?;
        Self::check_prefill_rms_rope_shape(shape, rope_shape)?;
        self.check_rope_shape(rope_shape)?;
        let elements = self.check_rms_buffers(input, weight, output, shape)?;
        let args = self.prefill_rms_rope_args(
            shape,
            rope_shape,
            elements,
            start_position,
            epsilon,
            theta,
        )?;
        self.dispatch(
            args,
            [
                None,
                Some(input),
                Some(output),
                None,
                None,
                None,
                None,
                None,
                None,
                self.rope_binding(),
                Some(weight),
                None,
            ],
        )
    }

    fn rms_norm_rope(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        position: Position<'_, Self::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        validate_positive("epsilon", epsilon)?;
        validate_positive("RoPE theta", theta)?;
        let position = self.checked_position(position)?;
        check_rope_position(position, 1)?;
        let rope_shape = RopeShape::new(1, shape.rows(), shape.columns())?;
        self.check_rope_shape(rope_shape)?;
        let elements = self.check_rms_buffers(input, weight, output, shape)?;
        let args = self.rms_rope_args(shape, elements, position, epsilon, theta)?;
        self.dispatch(
            args,
            [
                None,
                Some(input),
                Some(output),
                None,
                None,
                None,
                None,
                None,
                None,
                self.rope_binding(),
                Some(weight),
                None,
            ],
        )
    }

    fn rms_norm_residual(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        validate_positive("epsilon", epsilon)?;
        let elements = self.check_rms_residual_buffers(left, right, weight, output, shape)?;
        let args = Self::rms_args(OP_RMS_NORM_RESIDUAL, shape, elements, epsilon)?;
        self.dispatch(
            args,
            [
                None,
                Some(left),
                Some(output),
                Some(right),
                None,
                None,
                None,
                None,
                None,
                None,
                Some(weight),
                None,
            ],
        )
    }

    fn rms_norm_residual_store(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        weight: &Self::Buffer,
        residual: &mut Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        validate_positive("epsilon", epsilon)?;
        let elements =
            self.check_rms_residual_store_buffers(left, right, weight, residual, output, shape)?;
        let args = Self::rms_args(OP_RMS_NORM_RESIDUAL_STORE, shape, elements, epsilon)?;
        self.dispatch(
            args,
            [
                None,
                Some(left),
                Some(output),
                Some(right),
                Some(residual),
                None,
                None,
                None,
                None,
                None,
                Some(weight),
                None,
            ],
        )
    }

    fn rope(
        &mut self,
        values: &mut Self::Buffer,
        position: usize,
        shape: RopeShape,
        theta: f32,
    ) -> Result<(), BackendError> {
        validate_positive("RoPE theta", theta)?;
        Self::check_storage(values, BufferStorage::F32, "RoPE values")?;
        self.check_rope_shape(shape)?;
        let elements = shape.elements()?;
        check_rope_position(position, shape.tokens())?;
        Self::check_elements(values, elements, "RoPE values")?;
        let args = self.rope_dispatch_args(shape, elements, position, theta)?;
        self.dispatch(
            args,
            [
                None,
                None,
                Some(values),
                None,
                None,
                None,
                None,
                None,
                None,
                self.rope_binding(),
                None,
                None,
            ],
        )
    }

    fn swiglu(
        &mut self,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        Self::check_storage(gate, BufferStorage::F32, "SwiGLU gate")?;
        Self::check_storage(up, BufferStorage::F32, "SwiGLU up")?;
        Self::check_storage(output, BufferStorage::F32, "SwiGLU output")?;
        Self::check_elements(up, gate.layout.elements(), "SwiGLU up")?;
        Self::check_elements(output, gate.layout.elements(), "SwiGLU output")?;
        let args = Self::args(OP_SWIGLU, gate.layout.elements())?;
        self.dispatch(
            args,
            [
                None,
                Some(gate),
                Some(output),
                Some(up),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn residual_add(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        Self::check_storage(left, BufferStorage::F32, "residual left")?;
        Self::check_storage(right, BufferStorage::F32, "residual right")?;
        Self::check_storage(output, BufferStorage::F32, "residual output")?;
        Self::check_elements(right, left.layout.elements(), "residual right")?;
        Self::check_elements(output, left.layout.elements(), "residual output")?;
        let args = Self::args(OP_RESIDUAL_ADD, left.layout.elements())?;
        self.dispatch(
            args,
            [
                None,
                Some(left),
                Some(output),
                Some(right),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn kv_append(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        let position = self.checked_position(position)?;
        check_context_position(position, shape.max_context())?;
        self.check_kv_buffers(key, value, key_cache, value_cache, shape)?;
        let projected = shape.projected_kv_elements()?;
        self.check_kv_inputs(key, value, projected, "KV key", "KV value")?;
        let args = Self::kv_append_args(shape, position)?;
        self.dispatch(
            args,
            [
                None,
                Some(key),
                None,
                Some(value),
                None,
                None,
                Some(key_cache),
                Some(value_cache),
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn kv_append_chunk(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        key_cache: &mut Self::Buffer,
        value_cache: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        check_nonzero_tokens(tokens, "KV chunk tokens")?;
        check_context_range(start_position, tokens, shape.max_context())?;
        self.check_kv_buffers(key, value, key_cache, value_cache, shape)?;
        let elements = batch_elements(tokens, shape.projected_kv_elements()?, "KV chunk elements")?;
        self.check_kv_inputs(key, value, elements, "KV chunk key", "KV chunk value")?;
        let args = Self::kv_append_chunk_args(shape, start_position, tokens, elements)?;
        self.dispatch(
            args,
            [
                None,
                Some(key),
                None,
                Some(value),
                None,
                None,
                Some(key_cache),
                Some(value_cache),
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn kv_append_span(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        let position = self.checked_position(position)?;
        check_context_position(position, shape.max_context())?;
        let local_position = target.local_position(position)?;
        let physical_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            target.capacity_token_count(),
        )?;
        let (key_cache, value_cache, _, _) = target.into_parts();
        self.check_owner(key_cache, "write Metal KV span")?;
        self.check_owner(value_cache, "write Metal KV span")?;
        self.kv_append(
            key,
            value,
            key_cache,
            value_cache,
            physical_shape,
            Position::Host(local_position),
        )
    }

    fn kv_append_chunk_span(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        check_nonzero_tokens(tokens, "KV append tokens")?;
        check_context_range(start_position, tokens, shape.max_context())?;
        let local_range = target.local_range(start_position, tokens)?;
        let physical_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            target.capacity_token_count(),
        )?;
        let (key_cache, value_cache, _, _) = target.into_parts();
        self.check_owner(key_cache, "write Metal KV span")?;
        self.check_owner(value_cache, "write Metal KV span")?;
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

    fn prepare_attention_decode_batch_spans(
        &mut self,
        rows: &[AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        let result = self.check_batch_attention_rows(rows).and_then(|groups| {
            let prepared = self.prepare_batch_attention_resources(rows, &groups)?;
            self.prepared_batches.try_reserve(1).map_err(|_| {
                BackendError::operation(
                    "prepare Metal batch attention",
                    "prepared batch queue allocation failed",
                )
            })?;
            self.prepared_batches.push_back(prepared);
            Ok(())
        });
        if result.is_err() {
            self.clear_batch_attention_resources();
        }
        result
    }

    fn attention_decode_batch_spans(
        &mut self,
        rows: &mut [AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        let shape_groups = match self.check_batch_attention_rows(rows) {
            Ok(groups) => groups,
            Err(error) => {
                self.clear_batch_attention_resources();
                return Err(error);
            }
        };
        let (groups, multi_row_groups) = if self.attention_path == MetalAttentionPath::PerRow {
            (0, 0)
        } else {
            self.batch_group_stats(rows, &shape_groups)?
        };
        let mut pending = std::mem::take(&mut self.prepared_batches);
        let mut prepared = match pending.pop_front() {
            Some(prepared) => prepared,
            None => {
                self.clear_batch_attention_resources();
                return Err(BackendError::operation(
                    "dispatch Metal batch attention",
                    "no prepared batch resources are available",
                ));
            }
        };
        std::mem::swap(&mut self.attention_workspaces, &mut prepared.workspaces);
        std::mem::swap(&mut self.batch_kv_gathers, &mut prepared.gathers);
        std::mem::swap(&mut self.batch_prefix_gathers, &mut prepared.prefix_gathers);
        let result = if self.attention_path == MetalAttentionPath::PerRow {
            self.dispatch_batch_per_row(rows)
        } else {
            let workspaces = std::mem::take(&mut self.attention_workspaces);
            let result = self.dispatch_batch_shape_groups(&workspaces, rows, &shape_groups);
            self.attention_workspaces = workspaces;
            result
        };
        std::mem::swap(&mut self.attention_workspaces, &mut prepared.workspaces);
        std::mem::swap(&mut self.batch_kv_gathers, &mut prepared.gathers);
        std::mem::swap(&mut self.batch_prefix_gathers, &mut prepared.prefix_gathers);
        if result.is_ok() {
            self.record_batch_dispatch(rows.len(), groups, multi_row_groups);
            self.prepared_batches = pending;
        } else {
            self.clear_batch_attention_resources();
        }
        result
    }

    fn attention_decode(
        &mut self,
        query: &Self::Buffer,
        key_cache: &Self::Buffer,
        value_cache: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        let position = self.checked_position(position)?;
        self.attention_decode_contiguous(query, key_cache, value_cache, output, shape, position)
    }

    fn attention_decode_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        let position = self.checked_position(position)?;
        check_context_position(position, shape.max_context())?;
        Self::check_span_mapped_context(&cache, position, 1, "Metal span attention context")?;
        Self::check_span_attention_io(query, output, shape, 1)?;
        self.dispatch_attention_decode_spans(query, cache, output, shape, position)
    }

    fn attention_prefill(
        &mut self,
        query: &Self::Buffer,
        key_cache: &Self::Buffer,
        value_cache: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        self.attention_prefill_contiguous(
            query,
            key_cache,
            value_cache,
            output,
            shape,
            start_position,
            tokens,
        )
    }

    fn attention_prefill_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        check_nonzero_tokens(tokens, "prefill attention tokens")?;
        check_context_range(start_position, tokens, shape.max_context())?;
        Self::check_span_mapped_context(
            &cache,
            start_position,
            tokens,
            "Metal span prefill attention context",
        )?;
        self.dispatch_attention_prefill_spans(query, cache, output, shape, start_position, tokens)
    }

    fn verify_attention_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        positions: usize,
    ) -> Result<(), BackendError> {
        check_nonzero_tokens(positions, "verifier attention positions")?;
        check_context_range(start_position, positions, shape.max_context())?;
        Self::check_span_mapped_context(
            &cache,
            start_position,
            positions,
            "Metal verifier attention context",
        )?;
        self.dispatch_attention_prefill_spans(
            query,
            cache,
            output,
            shape,
            start_position,
            positions,
        )
    }

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
        let shapes = self.check_verifier_parameters(
            query_shape,
            key_shape,
            attention_shape,
            start_position,
            positions,
            epsilon,
            theta,
        )?;
        self.check_verifier_qk_buffers(
            query,
            query_weight,
            query_output,
            shapes.query_batch,
            key,
            key_weight,
            key_output,
            shapes.key_batch,
        )?;
        Self::check_verifier_value(value, attention_shape, positions)?;
        let (key_cache, value_cache, local_start, physical_shape) =
            Self::verifier_cache_parts(target, attention_shape, start_position, positions)?;
        self.check_owner(key_cache, "write Metal verifier KV span")?;
        self.check_owner(value_cache, "write Metal verifier KV span")?;
        self.check_verifier_kv_buffers(
            key_output,
            value,
            key_cache,
            value_cache,
            physical_shape,
            local_start,
            positions,
        )?;

        self.run_verifier_qk_norm_rope(
            query,
            query_weight,
            query_output,
            key,
            key_weight,
            key_output,
            start_position,
            shapes,
            epsilon,
            theta,
        )?;
        self.kv_append_chunk(
            key_output,
            value,
            key_cache,
            value_cache,
            physical_shape,
            local_start,
            positions,
        )
    }

    fn embed_gather(
        &mut self,
        table: &Self::Buffer,
        row: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        self.check_embedding_buffers(table, row, output, shape)?;
        Self::check_elements(row, 1, "embedding row")?;
        Self::check_elements(output, shape.columns(), "embedding output")?;
        self.validate_embedding_rows(row, shape.rows())?;
        let mut args = Self::args(OP_EMBED_GATHER, shape.columns())?;
        Self::configure_quant_args(&mut args, shape)?;
        args.table_rows = as_u32("embedding rows", shape.rows())?;
        self.dispatch(
            args,
            [
                Some(table),
                None,
                Some(output),
                None,
                None,
                None,
                None,
                None,
                Some(row),
                None,
                None,
                None,
            ],
        )
    }

    fn embed_gather_batch(
        &mut self,
        table: &Self::Buffer,
        rows: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError> {
        check_nonzero_tokens(tokens, "embedding tokens")?;
        let elements = self.check_embedding_batch_buffers(table, rows, output, shape, tokens)?;
        let args = Self::embedding_batch_args(shape, tokens, elements)?;
        self.dispatch(
            args,
            [
                Some(table),
                None,
                Some(output),
                None,
                None,
                None,
                None,
                None,
                Some(rows),
                None,
                None,
                None,
            ],
        )
    }

    fn copy_f32_row(
        &mut self,
        input: &Self::Buffer,
        row: usize,
        columns: usize,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        Self::check_storage(input, BufferStorage::F32, "copy f32 row input")?;
        Self::check_storage(output, BufferStorage::F32, "copy f32 row output")?;
        if columns == 0 {
            return Err(BackendError::Zero {
                field: "row columns",
            });
        }
        Self::check_elements(output, columns, "copied f32 row output")?;
        let rows = input.layout.elements() / columns;
        if row >= rows {
            return Err(BackendError::RowOutOfBounds { row, rows });
        }
        check_row_span(row, columns, "copied row index")?;
        let mut args = Self::args(OP_COPY_ROW, columns)?;
        args.row = as_u32("copied row", row)?;
        args.columns = as_u32("row columns", columns)?;
        self.dispatch(
            args,
            [
                None,
                Some(input),
                Some(output),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn write_f32_row(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        row: usize,
        columns: usize,
    ) -> Result<(), BackendError> {
        Self::check_storage(input, BufferStorage::F32, "write f32 row input")?;
        Self::check_storage(output, BufferStorage::F32, "write f32 row output")?;
        if columns == 0 {
            return Err(BackendError::Zero {
                field: "row columns",
            });
        }
        Self::check_elements(input, columns, "written f32 row input")?;
        let rows = output.layout.elements() / columns;
        if row >= rows {
            return Err(BackendError::RowOutOfBounds { row, rows });
        }
        check_row_span(row, columns, "written row index")?;
        let mut args = Self::args(OP_WRITE_ROW, columns)?;
        args.row = as_u32("written row", row)?;
        args.columns = as_u32("row columns", columns)?;
        self.dispatch(
            args,
            [
                None,
                Some(input),
                Some(output),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
    }

    fn argmax(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        Self::check_storage(input, BufferStorage::F32, "argmax input")?;
        Self::check_storage(output, BufferStorage::U32, "argmax output")?;
        Self::check_elements(output, 1, "argmax output")?;
        if input.layout.elements() == 0 {
            return Err(BackendError::Zero {
                field: "argmax input",
            });
        }
        let mut args = Self::args(OP_ARGMAX, REDUCTION_LANES)?;
        args.columns = as_u32("argmax elements", input.layout.elements())?;
        self.dispatch(
            args,
            [
                None,
                Some(input),
                None,
                None,
                None,
                None,
                None,
                None,
                Some(output),
                None,
                None,
                None,
            ],
        )
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        self.context
            .synchronize()
            .map_err(|error| BackendError::operation("synchronize Metal", error))?;
        self.span_descriptors = None;
        self.span_gather_key = None;
        self.span_gather_value = None;
        Ok(())
    }

    fn increment_u32(&mut self, buffer: &mut Self::Buffer) -> Result<(), BackendError> {
        Self::check_storage(buffer, BufferStorage::U32, "increment device position")?;
        Self::check_elements(buffer, 1, "increment device position")?;
        let mut value = [0_u32];
        self.read_u32(buffer, &mut value)?;
        value[0].checked_add(1).ok_or(BackendError::SizeOverflow {
            field: "device position",
        })?;
        let args = Self::args(OP_INCREMENT, 1)?;
        self.dispatch(
            args,
            [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(buffer),
                None,
                None,
                None,
            ],
        )
    }

    fn drop_decode_graph(&mut self) -> Result<(), BackendError> {
        self.context
            .synchronize()
            .map_err(|error| BackendError::operation("retire Metal decode graph", error))?;
        self.clear_batch_attention_resources();
        Ok(())
    }

    fn decode_graph_supported(&self) -> bool {
        false
    }

    fn begin_decode_profile(&mut self, _operations: usize) -> Result<(), BackendError> {
        Ok(())
    }

    fn profile_decode_op(&mut self, _op: DecodeOp) -> Result<(), BackendError> {
        Ok(())
    }

    fn end_decode_profile(
        &mut self,
        _steps: usize,
        _wall_duration: Duration,
    ) -> Result<Option<DecodeProfile>, BackendError> {
        Ok(None)
    }
}

fn as_u32(field: &'static str, value: usize) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(|_| BackendError::SizeOverflow { field })
}

fn check_gemv_buffers(
    weights: &MetalBuffer,
    input: &MetalBuffer,
    output: &MetalBuffer,
    shape: QuantMatrix,
) -> Result<(), BackendError> {
    MetalBackend::check_quant(weights, shape, "GEMV weights")?;
    MetalBackend::check_storage(input, BufferStorage::F32, "GEMV input")?;
    MetalBackend::check_storage(output, BufferStorage::F32, "GEMV output")?;
    MetalBackend::check_elements(input, shape.columns(), "GEMV input")?;
    MetalBackend::check_elements(output, shape.rows(), "GEMV output")
}

fn metadata_text(bytes: &[u8], field: &'static str) -> Result<String, BackendError> {
    let length = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    if length == 0 {
        return Err(BackendError::operation(
            field,
            "Metal returned an empty metadata field",
        ));
    }
    String::from_utf8(bytes[..length].to_vec())
        .map_err(|error| BackendError::operation(field, error))
}

fn metadata_optional_text(
    bytes: &[u8],
    field: &'static str,
) -> Result<Option<String>, BackendError> {
    let length = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    if length == 0 {
        Ok(None)
    } else {
        metadata_text(bytes, field).map(Some)
    }
}

fn batch_elements(count: usize, width: usize, field: &'static str) -> Result<usize, BackendError> {
    count
        .checked_mul(width)
        .ok_or(BackendError::SizeOverflow { field })
}

fn prefill_dispatch_groups(
    shape: QuantMatrix,
    tokens: usize,
    mode: PrefillGemmMode,
) -> Result<usize, BackendError> {
    match mode {
        PrefillGemmMode::Scalar => batch_elements(tokens, shape.rows(), "prefill dispatch outputs"),
        PrefillGemmMode::TokenTile => {
            let token_tiles = tokens.div_ceil(PREFILL_TOKEN_TILE);
            batch_elements(token_tiles, shape.rows(), "prefill dispatch groups")
        }
        PrefillGemmMode::SimdgroupMatrix => {
            let row_tiles = shape.rows().div_ceil(PREFILL_MATRIX_ROW_TILE);
            let token_tiles = tokens.div_ceil(PREFILL_MATRIX_TOKEN_TILE);
            batch_elements(token_tiles, row_tiles, "prefill dispatch groups")
        }
    }
}

fn select_prefill_gemm_mode(supports_simdgroup_matrix: bool, tokens: usize) -> PrefillGemmMode {
    if supports_simdgroup_matrix && tokens >= PREFILL_MATRIX_TOKEN_TILE {
        PrefillGemmMode::SimdgroupMatrix
    } else if tokens >= PREFILL_TOKEN_TILE_THRESHOLD {
        PrefillGemmMode::TokenTile
    } else {
        PrefillGemmMode::Scalar
    }
}

fn select_prefill_gemm_mode_with_numerics(
    supports_simdgroup_matrix: bool,
    tokens: usize,
    numerics: PrefillNumerics,
) -> PrefillGemmMode {
    if numerics == PrefillNumerics::DecodeEquivalent {
        return if tokens >= PREFILL_TOKEN_TILE_THRESHOLD {
            PrefillGemmMode::TokenTile
        } else {
            PrefillGemmMode::Scalar
        };
    }
    select_prefill_gemm_mode(supports_simdgroup_matrix, tokens)
}

fn attention_dispatch_threads(
    shape: AttentionShape,
    tokens: usize,
    field: &'static str,
) -> Result<(usize, bool), BackendError> {
    let tiled = shape.head_dim() <= REDUCTION_LANES;
    let outputs = if tiled {
        batch_elements(tokens, shape.n_head(), field)?
    } else {
        batch_elements(tokens, shape.query_elements()?, field)?
    };
    let threads = if tiled {
        reduction_dispatch_threads(outputs, field)?
    } else {
        as_u32(field, outputs)?;
        outputs
    };
    Ok((threads, tiled))
}

fn reduction_dispatch_threads(outputs: usize, field: &'static str) -> Result<usize, BackendError> {
    let threads = batch_elements(outputs, REDUCTION_LANES, field)?;
    as_u32(field, threads)?;
    Ok(threads)
}

fn check_u32_index_range(elements: usize, field: &'static str) -> Result<(), BackendError> {
    let last = elements
        .checked_sub(1)
        .ok_or(BackendError::Zero { field })?;
    as_u32(field, last).map(|_| ())
}

fn check_nonzero_tokens(tokens: usize, field: &'static str) -> Result<(), BackendError> {
    if tokens == 0 {
        return Err(BackendError::Zero { field });
    }
    Ok(())
}

fn check_context_position(position: usize, max_context: usize) -> Result<(), BackendError> {
    if position >= max_context {
        return Err(BackendError::PositionOutOfBounds {
            position,
            max_context,
        });
    }
    Ok(())
}

fn check_context_range(
    start_position: usize,
    tokens: usize,
    max_context: usize,
) -> Result<(), BackendError> {
    let end = start_position
        .checked_add(tokens)
        .ok_or(BackendError::SizeOverflow {
            field: "Metal context range",
        })?;
    if end > max_context {
        return Err(BackendError::PositionOutOfBounds {
            position: end,
            max_context,
        });
    }
    Ok(())
}

fn check_rope_position(position: usize, tokens: usize) -> Result<(), BackendError> {
    check_nonzero_tokens(tokens, "RoPE tokens")?;
    let last_position = position
        .checked_add(tokens - 1)
        .ok_or(BackendError::SizeOverflow {
            field: "RoPE final position",
        })?;
    as_u32("RoPE final position", last_position).map(|_| ())
}

fn check_row_span(row: usize, columns: usize, field: &'static str) -> Result<(), BackendError> {
    let last_element = row
        .checked_mul(columns)
        .and_then(|start| start.checked_add(columns.saturating_sub(1)))
        .ok_or(BackendError::SizeOverflow { field })?;
    as_u32(field, last_element).map(|_| ())
}

fn build_rope_frequencies(
    head_dim: usize,
    theta: f32,
    frequency_factors: Option<&[f32]>,
) -> Result<Vec<f32>, BackendError> {
    validate_rope_head_dim(head_dim)?;
    validate_positive("RoPE theta", theta)?;
    let half = head_dim / 2;
    validate_rope_factors(frequency_factors, half)?;
    (0..half)
        .map(|pair| {
            let factor = frequency_factors.map_or(1.0, |factors| factors[pair]);
            validate_positive("RoPE frequency factor", factor)?;
            Ok(theta.powf(-2.0 * pair as f32 / head_dim as f32) / factor)
        })
        .collect()
}

fn validate_rope_head_dim(head_dim: usize) -> Result<(), BackendError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(BackendError::NotDivisible {
            field: "RoPE head_dim",
            value: head_dim,
            divisor: 2,
        });
    }
    Ok(())
}

fn validate_rope_factors(factors: Option<&[f32]>, expected: usize) -> Result<(), BackendError> {
    if let Some(factors) = factors {
        if factors.len() != expected {
            return Err(BackendError::SizeMismatch {
                name: "RoPE frequency factors",
                expected,
                actual: factors.len(),
            });
        }
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f32) -> Result<(), BackendError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(BackendError::InvalidPositiveFloat { field, value })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use super::PrefillGemmDispatch;
    use super::{
        check_u32_index_range, prefill_dispatch_groups, select_prefill_gemm_mode,
        select_prefill_gemm_mode_with_numerics, AttentionShape, MetalAttentionPath, MetalBackend,
        PrefillGemmMode, QuantMatrix, PREFILL_MATRIX_K_TILE, PREFILL_MATRIX_ROW_TILE,
        PREFILL_MATRIX_TOKEN_TILE, PREFILL_TOKEN_TILE, PRODUCTION_BATCH_SIZE, REDUCTION_LANES,
        RESEARCH_BATCH_SIZE, SHADER_SOURCE,
    };
    #[cfg(target_os = "macos")]
    use half::f16;
    use leone::backend::PrefillNumerics;
    #[cfg(target_os = "macos")]
    use leone::{Backend, BufferLayout, CpuBackend};
    use leone::{BackendError, PrefillPlan, QuantFormat, VectorShape};

    #[test]
    fn attention_paths_keep_production_row_kernel_distinct() {
        assert!(!MetalAttentionPath::PerRow.needs_workspace());
        assert!(MetalAttentionPath::FixedTilePerRow.needs_workspace());
        assert!(MetalAttentionPath::SharedReadUnconstrained.is_shared());
        assert!(MetalAttentionPath::SharedReadFixedReduction.is_shared());
        assert_eq!(PRODUCTION_BATCH_SIZE, 1);
        assert_eq!(RESEARCH_BATCH_SIZE, 8);
    }

    #[test]
    fn prefill_scalar_index_range_rejects_shader_index_wrap() {
        if usize::BITS <= u32::BITS {
            return;
        }
        let maximum = usize::try_from(u32::MAX).expect("u32 fits in usize");
        assert!(check_u32_index_range(maximum + 1, "prefill input scalar index").is_ok());
        assert!(matches!(
            check_u32_index_range(maximum + 2, "prefill input scalar index"),
            Err(BackendError::SizeOverflow {
                field: "prefill input scalar index"
            })
        ));
    }

    #[test]
    fn simdgroup_prefill_dispatch_tiles_rows_and_tokens() {
        let shape = QuantMatrix::new(33, 256, QuantFormat::Q4K).expect("matrix shape");
        let groups = prefill_dispatch_groups(shape, 17, PrefillGemmMode::SimdgroupMatrix)
            .expect("matrix dispatch groups");
        assert_eq!(groups, 4);
        assert_eq!(PREFILL_MATRIX_ROW_TILE, 32);
        assert_eq!(PREFILL_MATRIX_TOKEN_TILE, 16);
    }

    #[test]
    fn scalar_prefill_dispatch_keeps_one_group_per_output() {
        let shape = QuantMatrix::new(3, 256, QuantFormat::Q6K).expect("matrix shape");
        let groups = prefill_dispatch_groups(shape, 8, PrefillGemmMode::Scalar)
            .expect("scalar dispatch groups");
        assert_eq!(groups, 24);
    }

    #[test]
    fn prefill_mode_selects_all_production_boundaries() {
        assert_eq!(select_prefill_gemm_mode(false, 8), PrefillGemmMode::Scalar);
        assert_eq!(select_prefill_gemm_mode(true, 8), PrefillGemmMode::Scalar);
        assert_eq!(
            select_prefill_gemm_mode(false, 9),
            PrefillGemmMode::TokenTile
        );
        assert_eq!(
            select_prefill_gemm_mode(true, 9),
            PrefillGemmMode::TokenTile
        );
        assert_eq!(
            select_prefill_gemm_mode(false, 15),
            PrefillGemmMode::TokenTile
        );
        assert_eq!(
            select_prefill_gemm_mode(true, 15),
            PrefillGemmMode::TokenTile
        );
        assert_eq!(
            select_prefill_gemm_mode(false, 16),
            PrefillGemmMode::TokenTile
        );
        assert_eq!(
            select_prefill_gemm_mode(true, 16),
            PrefillGemmMode::SimdgroupMatrix
        );
        assert_eq!(
            select_prefill_gemm_mode(false, 17),
            PrefillGemmMode::TokenTile
        );
        assert_eq!(
            select_prefill_gemm_mode(true, 17),
            PrefillGemmMode::SimdgroupMatrix
        );
        assert_eq!(
            PrefillGemmMode::SimdgroupMatrix.tile_rows(),
            PREFILL_MATRIX_ROW_TILE
        );
        assert_eq!(
            PrefillGemmMode::SimdgroupMatrix.k_tile(),
            PREFILL_MATRIX_K_TILE
        );
    }

    #[test]
    fn decode_equivalent_prefill_mode_avoids_matrix_selection() {
        for supports_matrix in [false, true] {
            assert_eq!(
                select_prefill_gemm_mode_with_numerics(
                    supports_matrix,
                    8,
                    PrefillNumerics::DecodeEquivalent,
                ),
                PrefillGemmMode::Scalar
            );
            assert_eq!(
                select_prefill_gemm_mode_with_numerics(
                    supports_matrix,
                    9,
                    PrefillNumerics::DecodeEquivalent,
                ),
                PrefillGemmMode::TokenTile
            );
            assert_eq!(
                select_prefill_gemm_mode_with_numerics(
                    supports_matrix,
                    16,
                    PrefillNumerics::DecodeEquivalent,
                ),
                PrefillGemmMode::TokenTile
            );
        }
    }

    #[test]
    fn decode_equivalent_prefill_plan_rejects_nonstandard_shapes() {
        let wide =
            PrefillPlan::new(8, 8, 32, 8, 256, 8_192, 32_768, 32_768).expect("wide prefill plan");
        assert!(MetalBackend::check_decode_equivalent_prefill_plan(wide).is_err());
        let mismatched = PrefillPlan::new(8, 8, 32, 8, 128, 4_095, 32_768, 32_768)
            .expect("mismatched prefill plan");
        assert!(MetalBackend::check_decode_equivalent_prefill_plan(mismatched).is_err());
    }

    #[test]
    fn prefill_shader_constants_match_host_selection() {
        let source = std::str::from_utf8(SHADER_SOURCE).expect("Metal source is UTF-8");
        for declaration in [
            "#define LEONE_PREFILL_TOKEN_TILE 4",
            "#define PREFILL_MATRIX_TILE_TOKENS 16",
            "#define PREFILL_MATRIX_TILE_ROWS 32",
            "#define PREFILL_MATRIX_K_TILE 32",
            "#define PREFILL_MATRIX_DIM 8",
        ] {
            assert!(source.contains(declaration), "missing {declaration}");
        }
        assert!(source.contains("args.tile_tokens == PREFILL_MATRIX_TILE_TOKENS"));
        assert!(source.contains("args.tile_tokens == LEONE_PREFILL_TOKEN_TILE"));
        assert!(source.contains("/ args.prefill_tile_rows + 1"));
        assert!(source.contains("+= args.prefill_k_tile"));
    }

    #[test]
    fn token_tile_prefill_dispatch_covers_token_tail() {
        let shape = QuantMatrix::new(33, 256, QuantFormat::Q4K).expect("matrix shape");
        let groups = prefill_dispatch_groups(shape, 15, PrefillGemmMode::TokenTile)
            .expect("token tile dispatch groups");
        assert_eq!(groups, 33 * 4);
        assert_eq!(PrefillGemmMode::TokenTile.tile_tokens(), PREFILL_TOKEN_TILE);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn forced_token_tile_at_large_tokens_matches_cpu_gemv() {
        let mut metal = MetalBackend::new().expect("native Metal backend");
        assert!(
            metal.device_info().supports_simdgroup_matrix,
            "the selected Metal device does not support SIMD-group matrix operations"
        );
        let rows = 3;
        let columns = 256;
        let mut cpu = CpuBackend::new();

        for tokens in [128, 129] {
            let input_values = (0..tokens * columns)
                .map(|index| (index % 23) as f32 / 11.0 - 1.0)
                .collect::<Vec<_>>();
            let input_bytes = backend_test_f32_bytes(&input_values);
            for format in [QuantFormat::Q4K, QuantFormat::Q6K] {
                let shape = QuantMatrix::new(rows, columns, format).expect("test matrix shape");
                let weight_bytes = token_tile_test_weights(format, rows, columns);
                let weight_layout = shape.layout().expect("test weight layout");
                let metal_weights = metal
                    .upload(weight_layout, &weight_bytes)
                    .expect("Metal weights");
                let metal_input = metal
                    .upload(
                        BufferLayout::f32(input_values.len()).expect("Metal input layout"),
                        &input_bytes,
                    )
                    .expect("Metal input");
                let metal_output = metal
                    .upload(
                        BufferLayout::f32(tokens * rows).expect("Metal output layout"),
                        &backend_test_f32_bytes(&vec![91.0; tokens * rows]),
                    )
                    .expect("Metal output");
                let dispatch = PrefillGemmDispatch {
                    shape,
                    tokens,
                    mode: PrefillGemmMode::TokenTile,
                };
                let args = metal
                    .prefill_gemm_args(&metal_weights, &metal_input, &metal_output, dispatch)
                    .expect("token tile arguments");
                assert_eq!(args.tile_tokens, PREFILL_TOKEN_TILE as u32);
                assert_eq!(args.prefill_tile_rows, 0);
                assert_eq!(args.prefill_k_tile, 0);
                assert_eq!(
                    args.threads,
                    (tokens.div_ceil(PREFILL_TOKEN_TILE) * rows * 256) as u32
                );
                metal
                    .dispatch(
                        args,
                        [
                            Some(&metal_weights),
                            Some(&metal_input),
                            Some(&metal_output),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        ],
                    )
                    .expect("forced token tile dispatch");
                let mut actual = vec![0.0; tokens * rows];
                metal
                    .read_f32(&metal_output, &mut actual)
                    .expect("read token tile output");

                let cpu_weights = cpu
                    .upload(weight_layout, &weight_bytes)
                    .expect("CPU weights");
                let mut expected = vec![0.0; tokens * rows];
                for token in 0..tokens {
                    let start = token * columns;
                    let cpu_input = cpu
                        .upload(
                            BufferLayout::f32(columns).expect("CPU input layout"),
                            &backend_test_f32_bytes(&input_values[start..start + columns]),
                        )
                        .expect("CPU input");
                    let mut cpu_output = cpu
                        .allocate(BufferLayout::f32(rows).expect("CPU output layout"))
                        .expect("CPU output");
                    cpu.gemv(&cpu_weights, &cpu_input, &mut cpu_output, shape)
                        .expect("CPU GEMV");
                    cpu.read_f32(&cpu_output, &mut expected[token * rows..(token + 1) * rows])
                        .expect("read CPU GEMV");
                }
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(actual.is_finite(), "nonfinite token tile output at {index}");
                    let tolerance = 1e-4 * expected.abs().max(1.0);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "token tile output differs at {index}: actual={actual:e}, expected={expected:e}"
                    );
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn backend_test_f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }

    #[cfg(target_os = "macos")]
    fn token_tile_test_weights(format: QuantFormat, rows: usize, columns: usize) -> Vec<u8> {
        let block_bytes = format.block_bytes();
        let mut bytes = vec![0_u8; rows * (columns / 256) * block_bytes];
        for (block_index, block) in bytes.chunks_exact_mut(block_bytes).enumerate() {
            match format {
                QuantFormat::Q4K => fill_token_tile_q4_block(block, block_index),
                QuantFormat::Q6K => fill_token_tile_q6_block(block, block_index),
            }
        }
        bytes
    }

    #[cfg(target_os = "macos")]
    fn fill_token_tile_q4_block(block: &mut [u8], block_index: usize) {
        set_token_tile_f16(block, 0, 0.25);
        set_token_tile_f16(block, 2, 0.125);
        block[4..16].copy_from_slice(&[
            0xc1, 0x82, 0xe3, 0x44, 0xf5, 0xa6, 0xd7, 0x68, 0xb9, 0xca, 0xeb, 0x9c,
        ]);
        for (index, byte) in block[16..].iter_mut().enumerate() {
            *byte = (index * 11 + block_index * 7 + 3) as u8;
        }
    }

    #[cfg(target_os = "macos")]
    fn fill_token_tile_q6_block(block: &mut [u8], block_index: usize) {
        for (index, byte) in block[..128].iter_mut().enumerate() {
            *byte = (index * 29 + block_index * 17 + 3) as u8;
        }
        for (index, byte) in block[128..192].iter_mut().enumerate() {
            *byte = (index * 53 + block_index * 31 + 0xa5) as u8;
        }
        block[192..208].copy_from_slice(&[
            0x81, 0x7f, 0x82, 0x03, 0xfe, 0x05, 0x86, 0x09, 0x0a, 0xf0, 0x0c, 0x8d, 0x10, 0x91,
            0x12, 0x13,
        ]);
        set_token_tile_f16(block, 208, 0.125);
    }

    #[cfg(target_os = "macos")]
    fn set_token_tile_f16(bytes: &mut [u8], offset: usize, value: f32) {
        bytes[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }

    #[test]
    fn batch_workspace_rejects_descriptor_index_wrap() {
        let shape = AttentionShape::new(2, 2, 128, 16_777_248).expect("valid shape");
        let tile_count = MetalBackend::attention_tile_count(shape).expect("tile count");
        let error = MetalBackend::attention_workspace_sizes(shape, 1_024, tile_count)
            .expect_err("descriptor index must fit Metal uint");
        assert!(matches!(
            error,
            BackendError::SizeOverflow {
                field: "batch attention tile descriptor index"
            }
        ));
    }

    #[test]
    fn packed_cache_rejects_scalar_index_wrap() {
        let shape = AttentionShape::new(2, 2, 128, 16_777_248).expect("valid shape");
        let error = MetalBackend::packed_cache_elements(shape, shape.max_context())
            .expect_err("packed KV index must fit Metal uint");
        assert!(matches!(
            error,
            BackendError::SizeOverflow {
                field: "Metal packed KV scalar index"
            }
        ));
    }

    #[test]
    fn plain_rms_norm_uses_one_fixed_group_per_row() {
        let shape = VectorShape::new(3, 257).expect("RMSNorm shape");
        let args = MetalBackend::rms_norm_args(shape, shape.elements().unwrap(), 1e-5)
            .expect("RMSNorm dispatch arguments");
        assert_eq!(
            args.threads,
            u32::try_from(3 * REDUCTION_LANES).expect("thread count fits")
        );
        assert_eq!(args.rows, 3);
        assert_eq!(args.columns, 257);
    }

    #[test]
    fn plain_rms_norm_rejects_row_group_index_wrap() {
        if usize::BITS <= u32::BITS {
            return;
        }
        let maximum_rows = usize::try_from(u32::MAX).expect("u32 fits in usize") / REDUCTION_LANES;
        let shape = VectorShape::new(maximum_rows + 1, 1).expect("valid RMSNorm shape");
        let error = MetalBackend::rms_norm_args(shape, shape.elements().unwrap(), 1e-5)
            .expect_err("RMSNorm thread count must fit Metal uint");
        assert!(matches!(
            error,
            BackendError::SizeOverflow {
                field: "RMSNorm dispatch threads"
            }
        ));
    }

    #[test]
    fn plain_rms_norm_accepts_the_maximum_scalar_index() {
        if usize::BITS <= u32::BITS {
            return;
        }
        let maximum = usize::try_from(u32::MAX).expect("u32 fits in usize");
        let shape = VectorShape::new(1, maximum).expect("valid RMSNorm shape");
        let args = MetalBackend::rms_norm_args(shape, shape.elements().unwrap(), 1e-5)
            .expect("maximum RMSNorm scalar index fits");
        assert_eq!(args.columns, u32::MAX);
    }
}
