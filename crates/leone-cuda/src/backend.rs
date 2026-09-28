use crate::cuda::{
    clear_cuda_last_error, prefill_growth_recovery, rope_at_frequencies,
    rope_at_frequencies_device_position, DeviceAllocationKeepalive, PrefillGrowthRecovery,
    SPAN_ERROR_WORDS,
};
use crate::{
    argmax, attention_decode, attention_decode_batch_spans_f16, attention_decode_device_position,
    attention_decode_f16, attention_decode_f16_device_position, attention_decode_q8,
    attention_decode_q8_device_position, attention_decode_spans,
    attention_decode_spans_device_position, attention_decode_spans_f16,
    attention_decode_spans_f16_device_position, attention_decode_spans_q8,
    attention_decode_spans_q8_device_position, attention_prefill_f16, attention_prefill_f32,
    attention_prefill_q8, attention_prefill_spans_f16, attention_prefill_spans_f32,
    attention_prefill_spans_q8, copy_f32_row, embedding_gather_batch,
    embedding_gather_q4_k_device_row, embedding_gather_q6_k_device_row, gemv_pair_q4_k,
    gemv_pair_swiglu_q4_k, gemv_q4_k, gemv_q4_k_residual, gemv_q6_k, gemv_q6_k_residual,
    increment_u32_scalar, kv_append, kv_append_chunk, kv_append_chunk_f16, kv_append_chunk_q8,
    kv_append_chunk_span, kv_append_chunk_span_f16, kv_append_chunk_span_q8,
    kv_append_device_position, kv_append_f16, kv_append_f16_device_position, kv_append_q8,
    kv_append_q8_device_position, kv_append_span, kv_append_span_device_position,
    kv_append_span_f16, kv_append_span_f16_device_position, kv_append_span_q8,
    kv_append_span_q8_device_position, kv_span_allocation, kv_span_owners, prefill_gemm,
    qk_norm_rope, qk_norm_rope_kv_append, qk_norm_rope_kv_append_device_position,
    qk_norm_rope_kv_append_f16, qk_norm_rope_kv_append_f16_device_position,
    qk_norm_rope_kv_append_span, qk_norm_rope_kv_append_span_device_position,
    qk_norm_rope_kv_append_span_f16, qk_norm_rope_kv_append_span_f16_device_position, qkv_gemv,
    repack_q4_k, residual_add, rms_norm, rms_norm_q8_parallel, rms_norm_residual,
    rms_norm_residual_store, rms_norm_rope, rms_norm_rope_device_position, swiglu,
    verify_attention_spans, verify_attention_spans_f16, verify_qk_norm_rope_kv_append_span,
    verify_qk_norm_rope_kv_append_span_f16, write_f32_row, write_u32_scalar, ArgmaxScratch,
    AttentionScratch, BatchDecodeGroup, BatchDecodeRow, Context, CublasLt, DeviceBuffer, Event,
    GemvScratch, Graph, KvSpanAllocation, KvSpanDescriptor, KvSpanOwners, KvSpanTable,
    PrefillScratch, RopeScratch, Stream, PREPARED_ATTENTION_HEAD_DIM,
};
use leone::backend::{
    HostStaging, KvReadView, KvWriteSpan, MemoryAccounting, MemoryAllocation, MemoryClass,
    MemoryReservation, UntrackedMemory,
};
use leone::{
    AttentionDecodeRow, AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot,
    BufferStorage, DecodeOp, DecodeProfile, Determinism, GemvProfile, MemoryBudget, MemoryCapacity,
    MemoryTracker, MemoryTrackerRoot, ModelImportMetrics, Position, PrefillMethod, PrefillNumerics,
    PrefillPlan, PrefillWorkspace, QuantFormat, QuantMatrix, RopePairing, RopeShape, VectorShape,
};
use std::collections::{btree_map::Entry, BTreeMap};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

const MAX_GRAPH_SPAN_TABLES: usize = 512;
const MAX_EAGER_SPAN_TABLES: usize = MAX_GRAPH_SPAN_TABLES;
// The limit covers one batch workspace, eight row workspaces, and eight positions.
const MAX_DECODE_GRAPH_BUFFERS: usize = 256;
const MAX_BATCH_DECODE_WORKSPACES: usize = MAX_GRAPH_SPAN_TABLES;
const BATCH_DECODE_TILE_TOKENS: usize = 32;
const BATCH_DECODE_MAX_ROWS: usize = 32;
const RESEARCH_BATCH_SIZE: usize = 8;

/// Selects the opt-in CUDA batched attention path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaAttentionBatchPath {
    /// Runs the existing per-row span kernel for every row.
    PerRow,
    /// Runs the native fixed tile kernel with one group per row.
    FixedTilePerRow,
    /// Reuses shared-prefix tile loads with schedule-dependent tile order.
    SharedReadUnconstrained,
    /// Reuses shared-prefix tile loads with one absolute reduction order.
    SharedReadFixedReduction,
}

/// Counts batched decode dispatches and row groups for one CUDA backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaAttentionBatchStats {
    /// Number of batch attention calls.
    pub dispatches: usize,
    /// Number of rows submitted to those calls.
    pub rows: usize,
    /// Number of native row groups launched.
    pub groups: usize,
    /// Number of native groups containing more than one row.
    pub multi_row_groups: usize,
}

impl CudaAttentionBatchPath {
    const fn uses_native_kernel(self) -> bool {
        !matches!(self, Self::PerRow)
    }

    const fn uses_shared_reads(self) -> bool {
        matches!(
            self,
            Self::SharedReadUnconstrained | Self::SharedReadFixedReduction
        )
    }

    const fn fixed_reduction(self) -> bool {
        matches!(self, Self::FixedTilePerRow | Self::SharedReadFixedReduction)
    }

    const fn max_batch_size(self) -> usize {
        match self {
            Self::PerRow
            | Self::FixedTilePerRow
            | Self::SharedReadUnconstrained
            | Self::SharedReadFixedReduction => RESEARCH_BATCH_SIZE,
        }
    }
}

#[derive(Debug)]
struct SpanTableInput {
    descriptors: Vec<KvSpanDescriptor>,
    allocations: Vec<KvSpanAllocation>,
    owners: Vec<KvSpanOwners>,
    mapped_tokens: usize,
}

#[derive(Debug)]
struct BatchDecodeWorkspace {
    rows: DeviceBuffer<BatchDecodeRow>,
    spans: DeviceBuffer<KvSpanDescriptor>,
    row_ids: DeviceBuffer<u32>,
    groups: DeviceBuffer<BatchDecodeGroup>,
    host_rows: Vec<BatchDecodeRow>,
    host_spans: Vec<KvSpanDescriptor>,
    host_row_ids: Vec<u32>,
    host_groups: Vec<BatchDecodeGroup>,
    allocations: Vec<KvSpanAllocation>,
    owners: Vec<KvSpanOwners>,
}

#[derive(Debug)]
struct BatchDecodeHostPlan {
    rows: Vec<BatchDecodeRow>,
    spans: Vec<KvSpanDescriptor>,
    row_ids: Vec<u32>,
    groups: Vec<BatchDecodeGroup>,
    allocations: Vec<KvSpanAllocation>,
    owners: Vec<KvSpanOwners>,
}

struct BatchDecodeRowsInput {
    rows: Vec<BatchDecodeRow>,
    spans: Vec<KvSpanDescriptor>,
    allocations: Vec<KvSpanAllocation>,
    owners: Vec<KvSpanOwners>,
    ranges: Vec<(usize, usize)>,
}

fn batch_u32(value: usize, field: &'static str) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(|_| BackendError::SizeOverflow { field })
}

fn rows_use_host_positions(rows: &[AttentionDecodeRow<'_, CudaBuffer>]) -> bool {
    rows.iter()
        .all(|row| matches!(row.position, Position::Host(_)))
}

fn attention_decode_batch_scalar(
    backend: &mut CudaBackend,
    rows: &mut [AttentionDecodeRow<'_, CudaBuffer>],
) -> Result<(), BackendError> {
    for row in &mut *rows {
        backend.attention_decode_spans(
            row.query,
            row.cache,
            row.output,
            row.shape,
            row.position,
        )?;
    }
    backend.record_batch_dispatch(rows.len(), 0, 0);
    Ok(())
}

fn same_span_layout(left: &KvSpanDescriptor, right: &KvSpanDescriptor) -> bool {
    left.key == right.key
        && left.value == right.value
        && left.logical_start == right.logical_start
        && left.mapped_tokens == right.mapped_tokens
        && left.capacity_tokens == right.capacity_tokens
}

fn common_span_count(first: &[KvSpanDescriptor], candidate: &[KvSpanDescriptor]) -> usize {
    first
        .iter()
        .zip(candidate)
        .take_while(|(left, right)| same_span_layout(left, right))
        .count()
}

fn batch_position_descriptor(
    position: Position<'_, CudaBuffer>,
    shape: AttentionShape,
) -> Result<(*const u32, u32, u32), BackendError> {
    match position {
        Position::Host(position) => {
            let end = position.checked_add(1).ok_or(BackendError::SizeOverflow {
                field: "batch attention context",
            })?;
            if end > shape.max_context() {
                return Err(BackendError::PositionOutOfBounds {
                    position: end,
                    max_context: shape.max_context(),
                });
            }
            Ok((
                std::ptr::null(),
                batch_u32(position, "batch attention position")?,
                0,
            ))
        }
        Position::Device(position) => {
            let buffer = position.u32()?;
            if buffer.len() != 1 {
                return Err(BackendError::SizeMismatch {
                    name: "batch attention device position",
                    expected: 1,
                    actual: buffer.len(),
                });
            }
            Ok((buffer.const_ptr(), 0, 1))
        }
    }
}

fn check_batch_row_devices(
    row: &AttentionDecodeRow<'_, CudaBuffer>,
    device: i32,
) -> Result<(), BackendError> {
    if cuda_storage_device(&row.query.storage) != device
        || cuda_storage_device(&row.output.storage) != device
    {
        return Err(BackendError::operation(
            "prepare batched attention",
            "query and output buffers use a different CUDA device",
        ));
    }
    if let Position::Device(position) = row.position {
        if position.u32()?.device() != device {
            return Err(BackendError::operation(
                "prepare batched attention",
                "device position uses a different CUDA device",
            ));
        }
    }
    Ok(())
}

fn check_batch_span_devices(input: &SpanTableInput, device: i32) -> Result<(), BackendError> {
    if input
        .allocations
        .iter()
        .any(|allocation| allocation.device != device)
    {
        return Err(BackendError::operation(
            "prepare batched attention",
            "KV buffers use a different CUDA device",
        ));
    }
    Ok(())
}

fn batch_row_pointers(
    row: &AttentionDecodeRow<'_, CudaBuffer>,
) -> Result<(*const f32, *mut f32), BackendError> {
    Ok((
        row.query.f32()?.const_ptr(),
        row.output.f32()?.const_ptr().cast_mut(),
    ))
}

fn batch_row_descriptor(
    row: &AttentionDecodeRow<'_, CudaBuffer>,
    shape: AttentionShape,
    device: i32,
    span_offset: usize,
    span_count: usize,
) -> Result<BatchDecodeRow, BackendError> {
    check_batch_row_devices(row, device)?;
    let query_elements = shape.query_elements()?;
    let expected_layout = BufferLayout::f32(query_elements)?;
    if row.query.layout != expected_layout || row.output.layout != expected_layout {
        return Err(BackendError::operation(
            "prepare batched attention",
            "query and output layouts do not match attention shape",
        ));
    }
    let (position, host_position, device_position) =
        batch_position_descriptor(row.position, shape)?;
    let (query, output) = batch_row_pointers(row)?;
    Ok(BatchDecodeRow {
        query,
        output,
        position,
        host_position,
        spans_offset: batch_u32(span_offset, "batch attention span offset")?,
        span_count: batch_u32(span_count, "batch attention span count")?,
        device_position,
    })
}

fn prepare_batch_row(
    row: &AttentionDecodeRow<'_, CudaBuffer>,
    shape: AttentionShape,
    device: i32,
    span_offset: usize,
) -> Result<(BatchDecodeRow, SpanTableInput), BackendError> {
    if row.shape != shape {
        return Err(BackendError::operation(
            "prepare batched attention",
            "all rows must use one attention shape",
        ));
    }
    let cache_kind = span_cache_kind(row.cache, row.shape)?;
    if cache_kind != BufferStorage::F16 {
        return Err(BackendError::operation(
            "prepare batched shared attention",
            "the native path requires FP16 KV spans",
        ));
    }
    validate_span_read_position(row.cache, shape, row.position)?;
    let input = read_span_descriptors(row.cache, row.shape)?;
    check_batch_span_devices(&input, device)?;
    let descriptor =
        batch_row_descriptor(row, shape, device, span_offset, input.descriptors.len())?;
    Ok((descriptor, input))
}

fn collect_batch_rows(
    rows: &[AttentionDecodeRow<'_, CudaBuffer>],
    shape: AttentionShape,
    device: i32,
) -> Result<BatchDecodeRowsInput, BackendError> {
    let mut plan_rows = Vec::with_capacity(rows.len());
    let mut all_spans = Vec::new();
    let mut allocations = Vec::new();
    let mut owners = Vec::new();
    let mut row_span_ranges = Vec::with_capacity(rows.len());
    for row in rows {
        let span_offset = all_spans.len();
        let (descriptor, input) = prepare_batch_row(row, shape, device, span_offset)?;
        all_spans.extend_from_slice(&input.descriptors);
        allocations.extend_from_slice(&input.allocations);
        owners.extend_from_slice(&input.owners);
        row_span_ranges.push((span_offset, input.descriptors.len()));
        plan_rows.push(descriptor);
    }
    Ok(BatchDecodeRowsInput {
        rows: plan_rows,
        spans: all_spans,
        allocations,
        owners,
        ranges: row_span_ranges,
    })
}

fn span_fully_visible(span: &KvSpanDescriptor, row: &BatchDecodeRow) -> bool {
    span.logical_start
        .checked_add(span.mapped_tokens)
        .is_some_and(|end| end <= (row.host_position as usize).saturating_add(1))
}

fn collect_group_members(
    seed: usize,
    rows: &[BatchDecodeRow],
    allow_shared_reads: bool,
    assigned: &mut [bool],
    all_spans: &[KvSpanDescriptor],
    row_span_ranges: &[(usize, usize)],
) -> Vec<usize> {
    if !allow_shared_reads {
        assigned[seed] = true;
        return vec![seed];
    }
    let (seed_offset, seed_count) = row_span_ranges[seed];
    let seed_spans = &all_spans[seed_offset..seed_offset + seed_count];
    let Some(seed_span) = seed_spans.first() else {
        assigned[seed] = true;
        return vec![seed];
    };
    if !span_fully_visible(seed_span, &rows[seed]) {
        assigned[seed] = true;
        return vec![seed];
    }
    let mut members = Vec::new();
    for candidate in seed..rows.len() {
        if assigned[candidate] || members.len() == BATCH_DECODE_MAX_ROWS {
            continue;
        }
        let (offset, count) = row_span_ranges[candidate];
        let candidate_spans = &all_spans[offset..offset + count];
        if seed_spans
            .first()
            .zip(candidate_spans.first())
            .is_some_and(|(left, right)| {
                same_span_layout(left, right)
                    && span_fully_visible(seed_span, &rows[seed])
                    && span_fully_visible(right, &rows[candidate])
            })
        {
            assigned[candidate] = true;
            members.push(candidate);
        }
    }
    members
}

fn append_group_descriptors(
    groups: &mut Vec<BatchDecodeGroup>,
    row_offset: usize,
    row_count: usize,
    shared_span_offset: usize,
    shared_span_count: usize,
    shape: AttentionShape,
) -> Result<(), BackendError> {
    for query_head in 0..shape.n_head() {
        groups.push(BatchDecodeGroup {
            row_offset: batch_u32(row_offset, "batch attention row offset")?,
            row_count: batch_u32(row_count, "batch attention row count")?,
            shared_span_offset: batch_u32(
                shared_span_offset,
                "batch attention shared span offset",
            )?,
            shared_span_count: batch_u32(shared_span_count, "batch attention shared span count")?,
            query_head: batch_u32(query_head, "batch attention query head")?,
            kv_head: batch_u32(
                query_head * shape.n_head_kv() / shape.n_head(),
                "batch attention KV head",
            )?,
        });
    }
    Ok(())
}

fn append_batch_groups(
    rows: &[BatchDecodeRow],
    shape: AttentionShape,
    allow_shared_reads: bool,
    all_spans: &[KvSpanDescriptor],
    row_span_ranges: &[(usize, usize)],
) -> Result<(Vec<u32>, Vec<BatchDecodeGroup>), BackendError> {
    let rows_len = rows.len();
    let mut row_ids = Vec::new();
    let mut groups = Vec::new();
    let mut assigned = vec![false; rows_len];
    for seed in 0..rows_len {
        if assigned[seed] {
            continue;
        }
        let members = collect_group_members(
            seed,
            rows,
            allow_shared_reads,
            &mut assigned,
            all_spans,
            row_span_ranges,
        );
        let (seed_offset, seed_count) = row_span_ranges[seed];
        let seed_spans = &all_spans[seed_offset..seed_offset + seed_count];
        let shared_count = members
            .iter()
            .map(|member| {
                let (offset, count) = row_span_ranges[*member];
                let candidate_spans = &all_spans[offset..offset + count];
                common_span_count(seed_spans, candidate_spans)
                    .min(visible_span_count(seed_spans, &rows[seed]))
                    .min(visible_span_count(candidate_spans, &rows[*member]))
            })
            .min()
            .unwrap_or(0);
        let row_offset = row_ids.len();
        row_ids.extend(members.iter().map(|member| *member as u32));
        append_group_descriptors(
            &mut groups,
            row_offset,
            members.len(),
            seed_offset,
            shared_count,
            shape,
        )?;
    }
    Ok((row_ids, groups))
}

fn visible_span_count(spans: &[KvSpanDescriptor], row: &BatchDecodeRow) -> usize {
    spans
        .iter()
        .take_while(|span| span_fully_visible(span, row))
        .count()
}

fn build_batch_decode_plan(
    rows: &[AttentionDecodeRow<'_, CudaBuffer>],
    device: i32,
    allow_shared_reads: bool,
) -> Result<(BatchDecodeHostPlan, AttentionShape), BackendError> {
    let first = rows.first().ok_or(BackendError::Zero {
        field: "batch attention rows",
    })?;
    let shape = first.shape;
    let input = collect_batch_rows(rows, shape, device)?;
    let (row_ids, groups) = append_batch_groups(
        &input.rows,
        shape,
        allow_shared_reads,
        &input.spans,
        &input.ranges,
    )?;
    if input.spans.is_empty() || groups.is_empty() {
        return Err(BackendError::Zero {
            field: "batch attention descriptors",
        });
    }
    Ok((
        BatchDecodeHostPlan {
            rows: input.rows,
            spans: input.spans,
            row_ids,
            groups,
            allocations: input.allocations,
            owners: input.owners,
        },
        shape,
    ))
}

impl BatchDecodeWorkspace {
    fn new(context: &Context, plan: BatchDecodeHostPlan) -> Result<Self, BackendError> {
        let workspace = Self {
            rows: context
                .alloc_class(plan.rows.len(), MemoryClass::BackendScratch)
                .map_err(|error| cuda_error("allocate batch attention rows", error))?,
            spans: context
                .alloc_class(plan.spans.len(), MemoryClass::BackendScratch)
                .map_err(|error| cuda_error("allocate batch attention spans", error))?,
            row_ids: context
                .alloc_class(plan.row_ids.len(), MemoryClass::BackendScratch)
                .map_err(|error| cuda_error("allocate batch attention row ids", error))?,
            groups: context
                .alloc_class(plan.groups.len(), MemoryClass::BackendScratch)
                .map_err(|error| cuda_error("allocate batch attention groups", error))?,
            host_rows: plan.rows,
            host_spans: plan.spans,
            host_row_ids: plan.row_ids,
            host_groups: plan.groups,
            allocations: plan.allocations,
            owners: plan.owners,
        };
        workspace.upload()
    }

    fn upload(mut self) -> Result<Self, BackendError> {
        self.rows
            .copy_from(&self.host_rows)
            .map_err(|error| cuda_error("upload batch attention rows", error))?;
        self.spans
            .copy_from(&self.host_spans)
            .map_err(|error| cuda_error("upload batch attention spans", error))?;
        self.row_ids
            .copy_from(&self.host_row_ids)
            .map_err(|error| cuda_error("upload batch attention row ids", error))?;
        self.groups
            .copy_from(&self.host_groups)
            .map_err(|error| cuda_error("upload batch attention groups", error))?;
        Ok(self)
    }

    fn matches(&self, plan: &BatchDecodeHostPlan) -> bool {
        self.host_rows == plan.rows
            && self.host_spans == plan.spans
            && self.host_row_ids == plan.row_ids
            && self.host_groups == plan.groups
            && self.allocations == plan.allocations
            && self
                .owners
                .iter()
                .all(|owner| owner.key.is_live() && owner.value.is_live())
    }
}

/// An opaque buffer owned by the CUDA backend.
#[derive(Debug)]
pub struct CudaBuffer {
    layout: BufferLayout,
    storage: CudaStorage,
}

impl CudaBuffer {
    fn memory_class(&self) -> MemoryClass {
        match &self.storage {
            CudaStorage::Bytes(buffer) => buffer.memory_class(),
            CudaStorage::F16(buffer) => buffer.memory_class(),
            CudaStorage::F32(buffer) => buffer.memory_class(),
            CudaStorage::U32(buffer) => buffer.memory_class(),
        }
    }

    fn reclassify(&self, class: MemoryClass) {
        match &self.storage {
            CudaStorage::Bytes(buffer) => buffer.reclassify(class),
            CudaStorage::F16(buffer) => buffer.reclassify(class),
            CudaStorage::F32(buffer) => buffer.reclassify(class),
            CudaStorage::U32(buffer) => buffer.reclassify(class),
        }
    }

    fn graph_keepalive(&self) -> DeviceAllocationKeepalive {
        match &self.storage {
            CudaStorage::Bytes(buffer) => buffer.keepalive(),
            CudaStorage::F16(buffer) => buffer.keepalive(),
            CudaStorage::F32(buffer) => buffer.keepalive(),
            CudaStorage::U32(buffer) => buffer.keepalive(),
        }
    }
}

fn cuda_storage_device(storage: &CudaStorage) -> i32 {
    match storage {
        CudaStorage::Bytes(buffer) => buffer.device(),
        CudaStorage::F16(buffer) => buffer.device(),
        CudaStorage::F32(buffer) => buffer.device(),
        CudaStorage::U32(buffer) => buffer.device(),
    }
}

fn span_descriptor(
    key: &CudaBuffer,
    value: &CudaBuffer,
    logical_start: usize,
    mapped_tokens: usize,
    capacity_tokens: usize,
    shape: AttentionShape,
) -> Result<(KvSpanDescriptor, KvSpanAllocation, KvSpanOwners), BackendError> {
    validate_span_descriptor_range(
        logical_start,
        mapped_tokens,
        capacity_tokens,
        shape,
        key.layout.elements(),
        value.layout.elements(),
    )?;
    let storage = span_storage(key, value)?;
    let descriptor = span_descriptor_for_storage(
        key,
        value,
        storage,
        logical_start,
        mapped_tokens,
        capacity_tokens,
    )?;
    let allocations = span_allocation_for_storage(key, value, storage)?;
    let owners = span_owners_for_storage(key, value, storage)?;
    Ok((descriptor, allocations, owners))
}

fn validate_span_descriptor_range(
    logical_start: usize,
    mapped_tokens: usize,
    capacity_tokens: usize,
    shape: AttentionShape,
    key_elements: usize,
    value_elements: usize,
) -> Result<(), BackendError> {
    if mapped_tokens == 0 || capacity_tokens == 0 || mapped_tokens > capacity_tokens {
        return Err(BackendError::operation(
            "build KV span descriptor",
            "mapped tokens must fit a nonempty span capacity",
        ));
    }
    let physical_end =
        logical_start
            .checked_add(capacity_tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "KV span physical end",
            })?;
    if physical_end > shape.max_context() {
        return Err(BackendError::PositionOutOfBounds {
            position: physical_end,
            max_context: shape.max_context(),
        });
    }
    let elements = capacity_tokens
        .checked_mul(shape.projected_kv_elements()?)
        .ok_or(BackendError::SizeOverflow {
            field: "KV span physical elements",
        })?;
    if key_elements != elements || value_elements != elements {
        return Err(BackendError::SizeMismatch {
            name: "KV span physical elements",
            expected: elements,
            actual: key_elements.min(value_elements),
        });
    }
    Ok(())
}

fn span_storage(key: &CudaBuffer, value: &CudaBuffer) -> Result<BufferStorage, BackendError> {
    if cuda_storage_device(&key.storage) != cuda_storage_device(&value.storage) {
        return Err(BackendError::operation(
            "build KV span descriptor",
            "key and value buffers use different CUDA devices",
        ));
    }
    let key_storage = key.layout.storage();
    let value_storage = value.layout.storage();
    if key_storage != value_storage {
        return Err(BackendError::operation(
            "build KV span descriptor",
            "key and value buffers must use matching FP32, FP16, or q8 KV storage",
        ));
    }
    match key_storage {
        BufferStorage::F32 | BufferStorage::F16 | BufferStorage::Q8Kv => Ok(key_storage),
        _ => Err(BackendError::operation(
            "build KV span descriptor",
            "key and value buffers must use matching FP32, FP16, or q8 KV storage",
        )),
    }
}

fn span_descriptor_for_storage(
    key: &CudaBuffer,
    value: &CudaBuffer,
    storage: BufferStorage,
    logical_start: usize,
    mapped_tokens: usize,
    capacity_tokens: usize,
) -> Result<KvSpanDescriptor, BackendError> {
    match (storage, &key.storage, &value.storage) {
        (BufferStorage::F32, CudaStorage::F32(key), CudaStorage::F32(value)) => {
            Ok(crate::cuda::kv_span_descriptor(
                key,
                value,
                logical_start,
                mapped_tokens,
                capacity_tokens,
            ))
        }
        (BufferStorage::F16, CudaStorage::F16(key), CudaStorage::F16(value)) => {
            Ok(crate::cuda::kv_span_descriptor(
                key,
                value,
                logical_start,
                mapped_tokens,
                capacity_tokens,
            ))
        }
        (BufferStorage::Q8Kv, CudaStorage::Bytes(key), CudaStorage::Bytes(value)) => {
            Ok(crate::cuda::kv_span_descriptor(
                key,
                value,
                logical_start,
                mapped_tokens,
                capacity_tokens,
            ))
        }
        _ => Err(BackendError::operation(
            "build KV span descriptor",
            "storage variant does not match KV span storage",
        )),
    }
}

fn span_allocation_for_storage(
    key: &CudaBuffer,
    value: &CudaBuffer,
    storage: BufferStorage,
) -> Result<KvSpanAllocation, BackendError> {
    match (storage, &key.storage, &value.storage) {
        (BufferStorage::F32, CudaStorage::F32(key), CudaStorage::F32(value)) => {
            Ok(kv_span_allocation(key, value))
        }
        (BufferStorage::F16, CudaStorage::F16(key), CudaStorage::F16(value)) => {
            Ok(kv_span_allocation(key, value))
        }
        (BufferStorage::Q8Kv, CudaStorage::Bytes(key), CudaStorage::Bytes(value)) => {
            Ok(kv_span_allocation(key, value))
        }
        _ => Err(BackendError::operation(
            "build KV span descriptor",
            "storage variant does not match KV span storage",
        )),
    }
}

fn span_owners_for_storage(
    key: &CudaBuffer,
    value: &CudaBuffer,
    storage: BufferStorage,
) -> Result<KvSpanOwners, BackendError> {
    match (storage, &key.storage, &value.storage) {
        (BufferStorage::F32, CudaStorage::F32(key), CudaStorage::F32(value)) => {
            Ok(kv_span_owners(key, value))
        }
        (BufferStorage::F16, CudaStorage::F16(key), CudaStorage::F16(value)) => {
            Ok(kv_span_owners(key, value))
        }
        (BufferStorage::Q8Kv, CudaStorage::Bytes(key), CudaStorage::Bytes(value)) => {
            Ok(kv_span_owners(key, value))
        }
        _ => Err(BackendError::operation(
            "build KV span descriptor",
            "storage variant does not match KV span storage",
        )),
    }
}

fn read_span_descriptors(
    cache: KvReadView<'_, CudaBuffer>,
    shape: AttentionShape,
) -> Result<SpanTableInput, BackendError> {
    if cache.spans().is_empty() {
        return Err(BackendError::Zero {
            field: "KV read spans",
        });
    }
    let mut descriptors = Vec::with_capacity(cache.spans().len());
    let mut allocations = Vec::with_capacity(cache.spans().len());
    let mut owners = Vec::with_capacity(cache.spans().len());
    for span in cache.spans() {
        let (descriptor, allocation, owner) = span_descriptor(
            span.key(),
            span.value(),
            span.logical_start(),
            span.token_count(),
            span.capacity_token_count(),
            shape,
        )?;
        descriptors.push(descriptor);
        allocations.push(allocation);
        owners.push(owner);
    }
    Ok(SpanTableInput {
        descriptors,
        allocations,
        owners,
        mapped_tokens: cache.mapped_tokens(),
    })
}

fn write_span_descriptor(
    target: KvWriteSpan<'_, CudaBuffer>,
    shape: AttentionShape,
) -> Result<
    (
        KvSpanDescriptor,
        KvSpanAllocation,
        KvSpanOwners,
        &mut CudaBuffer,
        &mut CudaBuffer,
    ),
    BackendError,
> {
    let (key, value, logical_start, capacity) = target.into_parts();
    let capacity = capacity.get();
    let (descriptor, allocation, owners) =
        span_descriptor(key, value, logical_start, capacity, capacity, shape)?;
    Ok((descriptor, allocation, owners, key, value))
}

fn matching_span_storage(
    operation: &'static str,
    key: &CudaBuffer,
    value: &CudaBuffer,
) -> Result<BufferStorage, BackendError> {
    let storage = key.layout.storage();
    if value.layout.storage() != storage {
        return Err(BackendError::operation(
            operation,
            "key and value cache storage differs",
        ));
    }
    match storage {
        BufferStorage::F32 | BufferStorage::F16 | BufferStorage::Q8Kv => Ok(storage),
        _ => Err(storage_error(operation, storage)),
    }
}

fn validate_span_host_position(
    target: &KvWriteSpan<'_, CudaBuffer>,
    shape: AttentionShape,
    position: Position<'_, CudaBuffer>,
) -> Result<(), BackendError> {
    if let Position::Host(position) = position {
        if position >= shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position,
                max_context: shape.max_context(),
            });
        }
        let span_end = target
            .logical_start()
            .checked_add(target.capacity_token_count())
            .ok_or(BackendError::SizeOverflow {
                field: "KV span logical end",
            })?;
        if position < target.logical_start() || position >= span_end {
            return Err(BackendError::PositionOutsideSpan {
                position,
                start: target.logical_start(),
                end: span_end,
            });
        }
    }
    Ok(())
}

fn validate_span_range(
    target: &KvWriteSpan<'_, CudaBuffer>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    field: &'static str,
) -> Result<usize, BackendError> {
    if tokens == 0 {
        return Err(BackendError::Zero { field });
    }
    let end_position = start_position
        .checked_add(tokens)
        .ok_or(BackendError::SizeOverflow {
            field: "KV span append end position",
        })?;
    if end_position > shape.max_context() {
        return Err(BackendError::PositionOutOfBounds {
            position: end_position,
            max_context: shape.max_context(),
        });
    }
    target.local_range(start_position, tokens)?;
    Ok(end_position)
}

fn validate_q8_span_shape(
    storage: BufferStorage,
    shape: AttentionShape,
) -> Result<(), BackendError> {
    if storage == BufferStorage::Q8Kv && !shape.head_dim().is_multiple_of(32) {
        return Err(BackendError::NotDivisible {
            field: "q8 KV head dimension",
            value: shape.head_dim(),
            divisor: 32,
        });
    }
    Ok(())
}

fn validate_fused_kv_value(
    value: &CudaBuffer,
    shape: AttentionShape,
    device: i32,
) -> Result<(), BackendError> {
    if value.layout.storage() != BufferStorage::F32 {
        return Err(storage_error(
            "launch fused QK KV append",
            value.layout.storage(),
        ));
    }
    let expected = shape.projected_kv_elements()?;
    if value.layout.elements() != expected {
        return Err(BackendError::SizeMismatch {
            name: "fused QK KV value",
            expected,
            actual: value.layout.elements(),
        });
    }
    if cuda_storage_device(&value.storage) != device {
        return Err(BackendError::operation(
            "launch fused QK KV append",
            "value uses a different CUDA device",
        ));
    }
    Ok(())
}

fn validate_fused_cache_buffer(
    cache: &CudaBuffer,
    name: &'static str,
    expected: usize,
    device: i32,
) -> Result<(), BackendError> {
    if cache.layout.elements() != expected {
        return Err(BackendError::SizeMismatch {
            name,
            expected,
            actual: cache.layout.elements(),
        });
    }
    if cuda_storage_device(&cache.storage) != device {
        return Err(BackendError::operation(
            "launch fused QK KV append",
            "KV cache uses a different CUDA device",
        ));
    }
    Ok(())
}

fn validate_fused_kv_cache_layout(
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    shape: AttentionShape,
    device: i32,
) -> Result<BufferStorage, BackendError> {
    let storage = matching_span_storage("launch fused QK KV append", key_cache, value_cache)?;
    validate_q8_span_shape(storage, shape)?;
    let expected = shape.cache_elements()?;
    validate_fused_cache_buffer(key_cache, "fused QK key cache", expected, device)?;
    validate_fused_cache_buffer(value_cache, "fused QK value cache", expected, device)?;
    Ok(storage)
}

fn validate_fused_kv_position(
    position: Position<'_, CudaBuffer>,
    shape: AttentionShape,
    device: i32,
) -> Result<(), BackendError> {
    match position {
        Position::Host(position) if position >= shape.max_context() => {
            Err(BackendError::PositionOutOfBounds {
                position,
                max_context: shape.max_context(),
            })
        }
        Position::Device(position) => {
            let position = position.u32()?;
            if position.len() != 1 {
                return Err(BackendError::SizeMismatch {
                    name: "fused QK KV position",
                    expected: 1,
                    actual: position.len(),
                });
            }
            if position.device() != device {
                return Err(BackendError::operation(
                    "launch fused QK KV append",
                    "position uses a different CUDA device",
                ));
            }
            Ok(())
        }
        Position::Host(_) => Ok(()),
    }
}

fn validate_fused_kv_cache(
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    shape: AttentionShape,
    position: Position<'_, CudaBuffer>,
    device: i32,
) -> Result<BufferStorage, BackendError> {
    let storage = validate_fused_kv_cache_layout(key_cache, value_cache, shape, device)?;
    validate_fused_kv_position(position, shape, device)?;
    Ok(storage)
}

fn validate_fused_qk_shape(
    query_shape: VectorShape,
    key_shape: VectorShape,
    attention_shape: AttentionShape,
) -> Result<(), BackendError> {
    if query_shape.rows() != attention_shape.n_head() {
        return Err(BackendError::SizeMismatch {
            name: "fused QK query rows",
            expected: attention_shape.n_head(),
            actual: query_shape.rows(),
        });
    }
    if key_shape.rows() != attention_shape.n_head_kv() {
        return Err(BackendError::SizeMismatch {
            name: "fused QK key rows",
            expected: attention_shape.n_head_kv(),
            actual: key_shape.rows(),
        });
    }
    if query_shape.columns() != attention_shape.head_dim() {
        return Err(BackendError::SizeMismatch {
            name: "fused QK query columns",
            expected: attention_shape.head_dim(),
            actual: query_shape.columns(),
        });
    }
    if key_shape.columns() != attention_shape.head_dim() {
        return Err(BackendError::SizeMismatch {
            name: "fused QK key columns",
            expected: attention_shape.head_dim(),
            actual: key_shape.columns(),
        });
    }
    Ok(())
}

#[derive(Debug)]
enum CudaStorage {
    Bytes(DeviceBuffer<u8>),
    F16(DeviceBuffer<u16>),
    F32(DeviceBuffer<f32>),
    U32(DeviceBuffer<u32>),
}

struct PrefillGrowthFailure {
    error: BackendError,
    recovery: Option<PrefillGrowthRecovery>,
}

impl CudaBuffer {
    fn f16(&self) -> Result<&DeviceBuffer<u16>, BackendError> {
        match &self.storage {
            CudaStorage::F16(buffer) => Ok(buffer),
            _ => Err(storage_error("read f16", self.layout.storage())),
        }
    }

    fn f16_mut(&mut self) -> Result<&mut DeviceBuffer<u16>, BackendError> {
        match &mut self.storage {
            CudaStorage::F16(buffer) => Ok(buffer),
            _ => Err(storage_error("write f16", self.layout.storage())),
        }
    }

    fn bytes(&self) -> Result<&DeviceBuffer<u8>, BackendError> {
        match &self.storage {
            CudaStorage::Bytes(buffer) => Ok(buffer),
            _ => Err(storage_error("read quantized bytes", self.layout.storage())),
        }
    }

    fn bytes_mut(&mut self) -> Result<&mut DeviceBuffer<u8>, BackendError> {
        match &mut self.storage {
            CudaStorage::Bytes(buffer) => Ok(buffer),
            _ => Err(storage_error(
                "write quantized bytes",
                self.layout.storage(),
            )),
        }
    }

    fn f32(&self) -> Result<&DeviceBuffer<f32>, BackendError> {
        match &self.storage {
            CudaStorage::F32(buffer) => Ok(buffer),
            _ => Err(storage_error("read f32", self.layout.storage())),
        }
    }

    fn f32_mut(&mut self) -> Result<&mut DeviceBuffer<f32>, BackendError> {
        match &mut self.storage {
            CudaStorage::F32(buffer) => Ok(buffer),
            _ => Err(storage_error("write f32", self.layout.storage())),
        }
    }

    fn u32(&self) -> Result<&DeviceBuffer<u32>, BackendError> {
        match &self.storage {
            CudaStorage::U32(buffer) => Ok(buffer),
            _ => Err(storage_error("read u32", self.layout.storage())),
        }
    }

    fn u32_mut(&mut self) -> Result<&mut DeviceBuffer<u32>, BackendError> {
        match &mut self.storage {
            CudaStorage::U32(buffer) => Ok(buffer),
            _ => Err(storage_error("write u32", self.layout.storage())),
        }
    }
}

/// The SM89 CUDA implementation of the backend contract.
#[derive(Debug)]
pub struct CudaBackend {
    context: Context,
    stream: Stream,
    gemv_scratch: BTreeMap<usize, GemvScratch>,
    verify_gemv_scratch: BTreeMap<(usize, usize), GemvScratch>,
    prepared_gemv_inputs: BTreeMap<usize, u64>,
    attention_scratch: BTreeMap<AttentionShape, AttentionScratch>,
    verify_attention_scratch: BTreeMap<(AttentionShape, usize), AttentionScratch>,
    argmax_scratch: BTreeMap<usize, ArgmaxScratch>,
    rope_scratch: Option<RopeScratch>,
    cublaslt: Option<CublasLt>,
    prefill_scratch: Option<PrefillScratch>,
    prefill_plan: Option<PrefillPlan>,
    eager_span_tables: Vec<KvSpanTable>,
    pending_graph_span_tables: Vec<KvSpanTable>,
    graph_span_tables: Vec<KvSpanTable>,
    graph_capture_active: bool,
    graph_preflight_active: bool,
    pending_graph_buffer_owners: Vec<DeviceAllocationKeepalive>,
    capture_graph_buffer_owners: Vec<DeviceAllocationKeepalive>,
    span_error: DeviceBuffer<u32>,
    span_error_max_context: Option<usize>,
    decode_graph_span_max_context: Option<usize>,
    attention_batch_path: CudaAttentionBatchPath,
    attention_batch_stats: CudaAttentionBatchStats,
    batch_decode_workspaces: Vec<BatchDecodeWorkspace>,
    decode_profiler: Option<CudaDecodeProfiler>,
    decode_graph: Option<Graph>,
    untracked: UntrackedMemory,
    q4_repack_duration: Duration,
    q4_repack_source_bytes: u64,
}

#[derive(Debug)]
struct ProfileOperation {
    class: DecodeOp,
    gemvs: [Option<QuantMatrix>; 3],
}

#[derive(Debug)]
struct CudaDecodeProfiler {
    events: Vec<Event>,
    operations: Vec<ProfileOperation>,
    kernel_launches: usize,
    h2d_copies: usize,
    d2h_copies: usize,
    stream_synchronizations: usize,
}

struct DecodeProfileCollections {
    gpu_duration_by_op: BTreeMap<DecodeOp, Duration>,
    gemv_by_shape: BTreeMap<QuantMatrix, GemvProfile>,
}

impl CudaBackend {
    /// Initializes one CUDA device and a nonblocking stream.
    pub fn new(device: i32) -> Result<Self, BackendError> {
        Self::with_memory_budget_and_attention_path(
            device,
            MemoryBudget::Unlimited,
            CudaAttentionBatchPath::PerRow,
        )
    }

    /// Initializes one CUDA device with an owned-allocation budget.
    pub fn with_memory_budget(device: i32, budget: MemoryBudget) -> Result<Self, BackendError> {
        Self::with_memory_budget_and_attention_path(device, budget, CudaAttentionBatchPath::PerRow)
    }

    /// Initializes one CUDA device with a caller-owned allocation tracker.
    pub fn with_memory_tracker(device: i32, tracker: MemoryTracker) -> Result<Self, BackendError> {
        Self::with_memory_tracker_and_attention_path(
            device,
            tracker,
            CudaAttentionBatchPath::PerRow,
        )
    }

    /// Initializes one CUDA device with a selected batched attention path.
    pub fn with_attention_batch_path(
        device: i32,
        path: CudaAttentionBatchPath,
    ) -> Result<Self, BackendError> {
        Self::with_memory_budget_and_attention_path(device, MemoryBudget::Unlimited, path)
    }

    /// Initializes one CUDA device with an allocation budget and attention path.
    pub fn with_memory_budget_and_attention_path(
        device: i32,
        budget: MemoryBudget,
        path: CudaAttentionBatchPath,
    ) -> Result<Self, BackendError> {
        Self::with_memory_tracker_and_attention_path(device, MemoryTracker::new(budget), path)
    }

    fn with_memory_tracker_and_attention_path(
        device: i32,
        tracker: MemoryTracker,
        attention_batch_path: CudaAttentionBatchPath,
    ) -> Result<Self, BackendError> {
        let context = Context::with_memory_tracker(device, tracker)
            .map_err(|error| cuda_error("initialize CUDA", error))?;
        let stream = Stream::new(&context).map_err(|error| cuda_error("create stream", error))?;
        let span_error = context
            .copy_to_device_class(&[0_u32; SPAN_ERROR_WORDS], MemoryClass::BackendScratch)
            .map_err(|error| cuda_error("allocate CUDA span error state", error))?;
        Ok(Self {
            context,
            stream,
            gemv_scratch: BTreeMap::new(),
            verify_gemv_scratch: BTreeMap::new(),
            prepared_gemv_inputs: BTreeMap::new(),
            attention_scratch: BTreeMap::new(),
            verify_attention_scratch: BTreeMap::new(),
            argmax_scratch: BTreeMap::new(),
            rope_scratch: None,
            cublaslt: None,
            prefill_scratch: None,
            prefill_plan: None,
            eager_span_tables: Vec::new(),
            pending_graph_span_tables: Vec::new(),
            graph_span_tables: Vec::new(),
            graph_capture_active: false,
            graph_preflight_active: false,
            pending_graph_buffer_owners: Vec::new(),
            capture_graph_buffer_owners: Vec::new(),
            span_error,
            span_error_max_context: None,
            decode_graph_span_max_context: None,
            attention_batch_path,
            attention_batch_stats: CudaAttentionBatchStats::default(),
            batch_decode_workspaces: Vec::new(),
            decode_profiler: None,
            decode_graph: None,
            untracked: UntrackedMemory {
                execution_streams: 1,
                ..UntrackedMemory::default()
            },
            q4_repack_duration: Duration::ZERO,
            q4_repack_source_bytes: 0,
        })
    }

    /// Returns identity and capacity from the device owned by this backend.
    pub fn device_info(&self) -> Result<crate::cuda::CudaDeviceInfo, BackendError> {
        self.context
            .device_info()
            .map_err(|error| cuda_error("query CUDA device information", error))
    }

    /// Returns the immutable batched attention path selected at construction.
    pub const fn attention_batch_path(&self) -> CudaAttentionBatchPath {
        self.attention_batch_path
    }

    /// Returns batched decode dispatch counts since construction.
    pub const fn attention_batch_stats(&self) -> CudaAttentionBatchStats {
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

    fn profile_launches(&mut self, launches: usize) {
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.kernel_launches += launches;
        }
    }

    fn profile_gemv(&mut self, shape: QuantMatrix) -> Result<(), BackendError> {
        if let Some(profiler) = &mut self.decode_profiler {
            let operation = profiler.operations.last_mut().ok_or_else(|| {
                BackendError::operation("profile GEMV", "missing operation boundary")
            })?;
            let slot = operation
                .gemvs
                .iter_mut()
                .find(|slot| slot.is_none())
                .ok_or_else(|| BackendError::operation("profile GEMV", "too many matrices"))?;
            *slot = Some(shape);
        }
        Ok(())
    }

    fn take_prepared_input(
        &mut self,
        columns: usize,
        input: &CudaBuffer,
    ) -> Result<bool, BackendError> {
        let identity = input.f32()?.identity();
        Ok(self.prepared_gemv_inputs.remove(&columns) == Some(identity))
    }

    fn mark_prepared_input(
        &mut self,
        columns: usize,
        input: &CudaBuffer,
    ) -> Result<(), BackendError> {
        self.prepared_gemv_inputs
            .insert(columns, input.f32()?.identity());
        Ok(())
    }

    fn prepare_prefill_scratch(
        &mut self,
        plan: PrefillPlan,
    ) -> Result<PrefillWorkspace, BackendError> {
        let can_reuse = self
            .prefill_scratch
            .as_ref()
            .is_some_and(|scratch| scratch.can_reuse(&plan));
        if can_reuse {
            match self.grow_prefill_scratch(plan) {
                Ok(usage) => Ok(usage),
                Err(failure) if failure.recovery.is_some() => {
                    self.retry_prefill_scratch(plan, failure)
                }
                Err(failure) => Err(failure.error),
            }
        } else {
            self.allocate_prefill_scratch(plan)
        }
    }

    fn validate_prefill_numerics(&self, plan: PrefillPlan) -> Result<(), BackendError> {
        if plan.numerics() == PrefillNumerics::DecodeEquivalent
            && !self.decode_equivalent_prefill_supported()
        {
            return Err(BackendError::operation(
                "prepare decode-equivalent prefill",
                "the CUDA backend does not support decode-equivalent prefill",
            ));
        }
        Ok(())
    }

    fn grow_prefill_scratch(
        &mut self,
        plan: PrefillPlan,
    ) -> std::result::Result<PrefillWorkspace, PrefillGrowthFailure> {
        let scratch = self
            .prefill_scratch
            .as_mut()
            .ok_or_else(|| PrefillGrowthFailure {
                error: BackendError::operation("reuse prefill workspace", "scratch is missing"),
                recovery: None,
            })?;
        if let Err(error) = scratch.ensure_capacity(&self.context, plan) {
            return Err(PrefillGrowthFailure {
                recovery: prefill_growth_recovery(&error),
                error: cuda_error("grow prefill workspace", error),
            });
        }
        Ok(scratch.usage())
    }

    fn retry_prefill_scratch(
        &mut self,
        plan: PrefillPlan,
        failure: PrefillGrowthFailure,
    ) -> Result<PrefillWorkspace, BackendError> {
        let growth_error = failure.error.to_string();
        // Only prefill scratch is cleared. Session buffers stay live for the retry.
        self.stream.synchronize().map_err(|error| {
            BackendError::operation(
                "retry prefill workspace",
                format!("{growth_error}; fence failed: {error}"),
            )
        })?;
        if failure.recovery == Some(PrefillGrowthRecovery::RuntimeAllocation) {
            clear_cuda_last_error();
        }
        self.allocate_prefill_scratch(plan)
            .map_err(|error| match error {
                BackendError::Memory(_) => error,
                error => BackendError::operation(
                    "retry prefill workspace",
                    format!("{growth_error}; allocation failed: {error}"),
                ),
            })
    }

    fn arm_span_error(&mut self, max_context: usize) {
        self.span_error_max_context = Some(max_context);
        if self.graph_capture_active {
            self.decode_graph_span_max_context = Some(max_context);
        }
    }

    fn ensure_span_execution_allowed(&self) -> Result<(), BackendError> {
        if self.graph_preflight_active {
            return Err(BackendError::operation(
                "execute KV span operation",
                "descriptor preflight is active",
            ));
        }
        Ok(())
    }

    fn retain_graph_buffer_owners(&mut self, buffers: &[&CudaBuffer]) -> Result<(), BackendError> {
        self.validate_graph_buffer_devices(buffers)?;
        let new_count = self.count_new_buffer_owners(buffers)?;
        self.reserve_graph_owner_capacity(new_count)?;
        for buffer in buffers {
            self.push_graph_owner(buffer.graph_keepalive());
        }
        Ok(())
    }

    fn validate_graph_buffer_devices(&self, buffers: &[&CudaBuffer]) -> Result<(), BackendError> {
        for buffer in buffers {
            if cuda_storage_device(&buffer.storage) != self.context.device() {
                return Err(BackendError::operation(
                    "retain decode graph buffers",
                    "buffer uses a different CUDA device",
                ));
            }
        }
        Ok(())
    }

    fn count_new_buffer_owners(&self, buffers: &[&CudaBuffer]) -> Result<usize, BackendError> {
        let mut new_count = 0_usize;
        for (index, buffer) in buffers.iter().enumerate() {
            let candidate = buffer.graph_keepalive();
            if self.graph_owner_is_pending(&candidate)
                || Self::buffer_owner_precedes(buffers, index, &candidate)
            {
                continue;
            }
            new_count = new_count.checked_add(1).ok_or(BackendError::SizeOverflow {
                field: "decode graph buffer owners",
            })?;
        }
        Ok(new_count)
    }

    fn buffer_owner_precedes(
        buffers: &[&CudaBuffer],
        index: usize,
        candidate: &DeviceAllocationKeepalive,
    ) -> bool {
        buffers[..index]
            .iter()
            .map(|buffer| buffer.graph_keepalive())
            .any(|owner| owner.same_allocation(candidate))
    }

    fn graph_owner_is_pending(&self, candidate: &DeviceAllocationKeepalive) -> bool {
        self.pending_graph_buffer_owners
            .iter()
            .any(|owner| owner.same_allocation(candidate))
    }

    fn reserve_graph_owner_capacity(&mut self, new_count: usize) -> Result<(), BackendError> {
        let additional =
            Self::graph_owner_reservation(self.pending_graph_buffer_owners.len(), new_count)?;
        if additional > 0 {
            self.pending_graph_buffer_owners
                .try_reserve_exact(additional)
                .map_err(|error| BackendError::operation("retain decode graph buffers", error))?;
        }
        Ok(())
    }

    fn graph_owner_reservation(
        current_len: usize,
        new_count: usize,
    ) -> Result<usize, BackendError> {
        let total = current_len
            .checked_add(new_count)
            .ok_or(BackendError::SizeOverflow {
                field: "decode graph buffer owners",
            })?;
        if total > MAX_DECODE_GRAPH_BUFFERS {
            return Err(Self::graph_owner_limit_error());
        }
        Ok(total.saturating_sub(current_len))
    }

    fn push_graph_owner(&mut self, candidate: DeviceAllocationKeepalive) {
        if !self.graph_owner_is_pending(&candidate) {
            self.pending_graph_buffer_owners.push(candidate);
        }
    }

    fn stage_graph_owner<T, F>(
        pending: &[T],
        candidates: &mut [Option<T>; MAX_DECODE_GRAPH_BUFFERS],
        count: &mut usize,
        candidate: T,
        same: F,
    ) -> bool
    where
        F: Fn(&T, &T) -> bool + Copy,
    {
        if pending.iter().any(|owner| same(owner, &candidate))
            || candidates[..*count]
                .iter()
                .flatten()
                .any(|owner| same(owner, &candidate))
        {
            return false;
        }
        if pending.len() >= MAX_DECODE_GRAPH_BUFFERS
            || *count >= MAX_DECODE_GRAPH_BUFFERS - pending.len()
        {
            return true;
        }
        candidates[*count] = Some(candidate);
        *count += 1;
        false
    }

    fn stage_backend_graph_dependencies(
        &self,
        candidates: &mut [Option<DeviceAllocationKeepalive>; MAX_DECODE_GRAPH_BUFFERS],
        count: &mut usize,
    ) -> Result<(), BackendError> {
        self.stage_gemv_graph_dependencies(candidates, count)?;
        self.stage_attention_graph_dependencies(candidates, count)?;
        self.stage_rope_graph_dependencies(candidates, count)?;
        self.stage_argmax_graph_dependencies(candidates, count)?;
        let pending = self.pending_graph_buffer_owners.as_slice();
        if Self::stage_graph_owner(
            pending,
            candidates,
            count,
            self.span_error.keepalive(),
            DeviceAllocationKeepalive::same_allocation,
        ) {
            return Err(Self::graph_owner_limit_error());
        }
        Ok(())
    }

    fn stage_gemv_graph_dependencies(
        &self,
        candidates: &mut [Option<DeviceAllocationKeepalive>; MAX_DECODE_GRAPH_BUFFERS],
        count: &mut usize,
    ) -> Result<(), BackendError> {
        let pending = self.pending_graph_buffer_owners.as_slice();
        let mut limit = false;
        for scratch in self.gemv_scratch.values() {
            scratch.for_each_graph_owner(|owner| {
                if !limit {
                    limit = Self::stage_graph_owner(
                        pending,
                        candidates,
                        count,
                        owner,
                        DeviceAllocationKeepalive::same_allocation,
                    );
                }
            });
            if limit {
                return Err(Self::graph_owner_limit_error());
            }
        }
        Ok(())
    }

    fn stage_attention_graph_dependencies(
        &self,
        candidates: &mut [Option<DeviceAllocationKeepalive>; MAX_DECODE_GRAPH_BUFFERS],
        count: &mut usize,
    ) -> Result<(), BackendError> {
        let pending = self.pending_graph_buffer_owners.as_slice();
        let mut limit = false;
        for scratch in self.attention_scratch.values() {
            scratch.for_each_graph_owner(|owner| {
                if !limit {
                    limit = Self::stage_graph_owner(
                        pending,
                        candidates,
                        count,
                        owner,
                        DeviceAllocationKeepalive::same_allocation,
                    );
                }
            });
            if limit {
                return Err(Self::graph_owner_limit_error());
            }
        }
        Ok(())
    }

    fn stage_rope_graph_dependencies(
        &self,
        candidates: &mut [Option<DeviceAllocationKeepalive>; MAX_DECODE_GRAPH_BUFFERS],
        count: &mut usize,
    ) -> Result<(), BackendError> {
        let Some(scratch) = self.rope_scratch.as_ref() else {
            return Ok(());
        };
        let pending = self.pending_graph_buffer_owners.as_slice();
        let mut limit = false;
        scratch.for_each_graph_owner(|owner| {
            if !limit {
                limit = Self::stage_graph_owner(
                    pending,
                    candidates,
                    count,
                    owner,
                    DeviceAllocationKeepalive::same_allocation,
                );
            }
        });
        if limit {
            return Err(Self::graph_owner_limit_error());
        }
        Ok(())
    }

    fn stage_argmax_graph_dependencies(
        &self,
        candidates: &mut [Option<DeviceAllocationKeepalive>; MAX_DECODE_GRAPH_BUFFERS],
        count: &mut usize,
    ) -> Result<(), BackendError> {
        let pending = self.pending_graph_buffer_owners.as_slice();
        let mut limit = false;
        for scratch in self.argmax_scratch.values() {
            scratch.for_each_graph_owner(|owner| {
                if !limit {
                    limit = Self::stage_graph_owner(
                        pending,
                        candidates,
                        count,
                        owner,
                        DeviceAllocationKeepalive::same_allocation,
                    );
                }
            });
            if limit {
                return Err(Self::graph_owner_limit_error());
            }
        }
        Ok(())
    }

    fn retain_backend_graph_dependencies(&mut self) -> Result<(), BackendError> {
        let mut candidates = std::array::from_fn(|_| None);
        let mut count = 0;
        self.stage_backend_graph_dependencies(&mut candidates, &mut count)?;
        self.reserve_graph_owner_capacity(count)?;
        for candidate in candidates.iter_mut().take(count) {
            if let Some(candidate) = candidate.take() {
                self.pending_graph_buffer_owners.push(candidate);
            }
        }
        Ok(())
    }

    fn graph_owner_limit_error() -> BackendError {
        BackendError::operation(
            "retain decode graph buffers",
            "captured buffer retention limit reached",
        )
    }

    fn enqueue_span_error(
        &mut self,
        status: &mut [u32; SPAN_ERROR_WORDS],
    ) -> Result<bool, BackendError> {
        if self.span_error_max_context.is_none() {
            return Ok(false);
        }
        self.span_error
            .copy_to_async(&self.stream, status)
            .map_err(|error| cuda_error("read CUDA span error state", error))?;
        Ok(true)
    }

    fn finish_span_error(&mut self, status: [u32; SPAN_ERROR_WORDS]) -> Result<(), BackendError> {
        let Some(max_context) = self.span_error_max_context else {
            return Ok(());
        };
        if status[0] == 0 {
            self.span_error_max_context = None;
            return Ok(());
        }
        self.span_error
            .copy_from(&[0_u32; SPAN_ERROR_WORDS])
            .map_err(|error| cuda_error("clear CUDA span error state", error))?;
        self.span_error_max_context = None;
        Err(span_error_from_status(status, max_context))
    }

    fn check_span_error(&mut self) -> Result<(), BackendError> {
        let mut status = [0_u32; SPAN_ERROR_WORDS];
        if !self.enqueue_span_error(&mut status)? {
            return Ok(());
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("finish CUDA span error read", error))?;
        self.finish_span_error(status)
    }

    fn clear_span_error(&mut self) -> Result<(), BackendError> {
        self.span_error
            .copy_from(&[0_u32; SPAN_ERROR_WORDS])
            .map_err(|error| cuda_error("clear CUDA span error state", error))?;
        self.span_error_max_context = None;
        Ok(())
    }

    fn allocate_prefill_scratch(
        &mut self,
        plan: PrefillPlan,
    ) -> Result<PrefillWorkspace, BackendError> {
        self.prefill_scratch = None;
        let scratch = PrefillScratch::new(&self.context, plan)
            .map_err(|error| cuda_error("allocate prefill workspace", error))?;
        let usage = scratch.usage();
        self.prefill_scratch = Some(scratch);
        Ok(usage)
    }

    fn prepare_batch_decode_workspace(
        &mut self,
        rows: &[AttentionDecodeRow<'_, CudaBuffer>],
        allow_shared_reads: bool,
    ) -> Result<(), BackendError> {
        let (plan, _) = build_batch_decode_plan(rows, self.context.device(), allow_shared_reads)?;
        if self.batch_workspace_index(&plan).is_some() {
            return Ok(());
        }
        if self.graph_capture_active {
            return Err(BackendError::operation(
                "prepare batched attention",
                "graph capture requires the prepared batch layout",
            ));
        }
        self.install_batch_workspace(plan)
    }

    fn batch_workspace_index(&self, plan: &BatchDecodeHostPlan) -> Option<usize> {
        self.batch_decode_workspaces
            .iter()
            .position(|workspace| workspace.matches(plan))
    }

    fn fence_batch_workspace(&self) -> Result<(), BackendError> {
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("fence batch attention workspace", error))
    }

    fn install_batch_workspace(&mut self, plan: BatchDecodeHostPlan) -> Result<(), BackendError> {
        if self.batch_decode_workspaces.len() == MAX_BATCH_DECODE_WORKSPACES {
            return Err(BackendError::operation(
                "prepare batched attention",
                "batch workspace retention limit reached",
            ));
        }
        self.fence_batch_workspace()?;
        self.batch_decode_workspaces
            .push(BatchDecodeWorkspace::new(&self.context, plan)?);
        Ok(())
    }

    fn batch_decode_workspace(
        &self,
        rows: &[AttentionDecodeRow<'_, CudaBuffer>],
        allow_shared_reads: bool,
    ) -> Result<(AttentionShape, &BatchDecodeWorkspace), BackendError> {
        let (plan, shape) =
            build_batch_decode_plan(rows, self.context.device(), allow_shared_reads)?;
        let index = self.batch_workspace_index(&plan).ok_or_else(|| {
            BackendError::operation(
                "run batched attention",
                "batch attention workspace is not prepared",
            )
        })?;
        let workspace = self.batch_decode_workspaces.get(index).ok_or_else(|| {
            BackendError::operation(
                "run batched attention",
                "batch attention workspace disappeared after preflight",
            )
        })?;
        Ok((shape, workspace))
    }

    fn retain_span_table(
        &mut self,
        descriptors: &[KvSpanDescriptor],
        allocations: &[KvSpanAllocation],
        owners: &[KvSpanOwners],
    ) -> Result<usize, BackendError> {
        self.validate_span_table_devices(allocations)?;
        if self.graph_capture_active {
            return self.graph_span_table_index(descriptors, allocations);
        }
        if self.graph_preflight_active {
            if let Some(index) = self
                .pending_graph_span_tables
                .iter()
                .position(|table| table.matches(descriptors, allocations))
            {
                return Ok(index);
            }
            return self.allocate_pending_graph_span_table(descriptors, allocations, owners);
        }
        if let Some(index) = self
            .eager_span_tables
            .iter()
            .position(|table| table.matches(descriptors, allocations))
        {
            return Ok(index);
        }
        self.allocate_eager_span_table(descriptors, allocations, owners)
    }

    fn validate_span_table_devices(
        &self,
        allocations: &[KvSpanAllocation],
    ) -> Result<(), BackendError> {
        if allocations
            .iter()
            .any(|allocation| allocation.device != self.context.device())
        {
            return Err(BackendError::operation(
                "retain KV span table",
                "KV buffers use a different CUDA device",
            ));
        }
        Ok(())
    }

    fn graph_span_table_index(
        &self,
        descriptors: &[KvSpanDescriptor],
        allocations: &[KvSpanAllocation],
    ) -> Result<usize, BackendError> {
        self.graph_span_tables
            .iter()
            .position(|table| table.matches(descriptors, allocations))
            .ok_or_else(|| {
                BackendError::operation(
                    "retain KV span table",
                    "span table was not prepared before graph capture",
                )
            })
    }

    fn allocate_eager_span_table(
        &mut self,
        descriptors: &[KvSpanDescriptor],
        allocations: &[KvSpanAllocation],
        owners: &[KvSpanOwners],
    ) -> Result<usize, BackendError> {
        if self.eager_span_tables.len() >= MAX_EAGER_SPAN_TABLES {
            self.stream
                .synchronize()
                .map_err(|error| cuda_error("retire KV span tables", error))?;
            self.eager_span_tables.clear();
        }
        let table = KvSpanTable::new(&self.context, descriptors, allocations, owners)
            .map_err(|error| cuda_error("allocate KV span table", error))?;
        self.eager_span_tables.push(table);
        Ok(self.eager_span_tables.len() - 1)
    }

    fn allocate_pending_graph_span_table(
        &mut self,
        descriptors: &[KvSpanDescriptor],
        allocations: &[KvSpanAllocation],
        owners: &[KvSpanOwners],
    ) -> Result<usize, BackendError> {
        if self.pending_graph_span_tables.len() >= MAX_GRAPH_SPAN_TABLES {
            return Err(BackendError::operation(
                "prepare decode graph",
                "descriptor table retention limit reached",
            ));
        }
        let table = KvSpanTable::new(&self.context, descriptors, allocations, owners)
            .map_err(|error| cuda_error("allocate KV graph span table", error))?;
        self.pending_graph_span_tables.push(table);
        Ok(self.pending_graph_span_tables.len() - 1)
    }

    fn retain_span_table_input(&mut self, input: &SpanTableInput) -> Result<usize, BackendError> {
        self.retain_span_table(&input.descriptors, &input.allocations, &input.owners)
    }

    fn has_retained_span_table(&self, input: &SpanTableInput) -> bool {
        if self.graph_preflight_active {
            return self
                .pending_graph_span_tables
                .iter()
                .any(|table| table.matches(&input.descriptors, &input.allocations));
        }
        self.eager_span_tables
            .iter()
            .chain(&self.graph_span_tables)
            .any(|table| table.matches(&input.descriptors, &input.allocations))
    }

    fn retained_span_table(
        &self,
        table_index: usize,
    ) -> Result<&DeviceBuffer<KvSpanDescriptor>, BackendError> {
        let tables = if self.graph_capture_active {
            &self.graph_span_tables
        } else {
            &self.eager_span_tables
        };
        tables
            .get(table_index)
            .map(KvSpanTable::device)
            .ok_or_else(|| BackendError::operation("find retained KV span table", "missing entry"))
    }

    fn validate_graph_table_count(&self, staged_tables: bool) -> Result<(), BackendError> {
        let table_count = if staged_tables {
            self.pending_graph_span_tables.len()
        } else {
            self.eager_span_tables.len()
        };
        if table_count > MAX_GRAPH_SPAN_TABLES {
            return Err(BackendError::operation(
                "prepare decode graph",
                "descriptor table retention limit reached",
            ));
        }
        Ok(())
    }

    fn retire_decode_graph(&mut self) -> Result<(), BackendError> {
        if self.decode_graph.is_some() || !self.graph_span_tables.is_empty() {
            self.stream
                .synchronize()
                .map_err(|error| cuda_error("prepare decode graph replacement", error))?;
        }
        if self.decode_graph.take().is_some() {
            self.untracked.graph_objects = self
                .untracked
                .graph_objects
                .checked_sub(1)
                .expect("graph object count underflow");
        }
        self.graph_span_tables.clear();
        self.decode_graph_span_max_context = None;
        Ok(())
    }

    fn stage_decode_graph_tables(&mut self, staged_tables: bool) {
        if staged_tables {
            self.eager_span_tables.clear();
            self.graph_span_tables = std::mem::take(&mut self.pending_graph_span_tables);
        } else {
            self.graph_span_tables = std::mem::take(&mut self.eager_span_tables);
        }
    }

    fn start_graph_capture(&mut self) -> Result<(), BackendError> {
        self.graph_capture_active = true;
        match self.stream.begin_graph_capture() {
            Ok(()) => Ok(()),
            Err(error) => {
                self.graph_capture_active = false;
                self.capture_graph_buffer_owners.clear();
                self.eager_span_tables = std::mem::take(&mut self.graph_span_tables);
                Err(cuda_error("begin decode graph", error))
            }
        }
    }

    fn retire_eager_decode_workspaces(&mut self) {
        self.eager_span_tables.clear();
        self.batch_decode_workspaces.clear();
    }

    fn dispatch_native_attention_decode_batch(
        &mut self,
        rows: &mut [AttentionDecodeRow<'_, CudaBuffer>],
    ) -> Result<(), BackendError> {
        let view_rows = rows
            .iter_mut()
            .map(|row| AttentionDecodeRow {
                query: row.query,
                cache: row.cache,
                output: row.output,
                shape: row.shape,
                position: row.position,
            })
            .collect::<Vec<_>>();
        let (shape, workspace) =
            self.batch_decode_workspace(&view_rows, self.attention_batch_path.uses_shared_reads())?;
        let cuda_shape = attention_shape(shape)?;
        attention_decode_batch_spans_f16(
            &self.stream,
            &workspace.rows,
            &workspace.spans,
            &workspace.row_ids,
            &workspace.groups,
            cuda_shape,
            BATCH_DECODE_TILE_TOKENS,
            self.attention_batch_path.fixed_reduction(),
        )
        .map_err(|error| cuda_error("launch batched span attention", error))?;
        let multi_row_groups = workspace
            .host_groups
            .iter()
            .filter(|group| group.row_count > 1)
            .count();
        self.record_batch_dispatch(rows.len(), workspace.host_groups.len(), multi_row_groups);
        self.profile_launches(1);
        Ok(())
    }
}

impl Backend for CudaBackend {
    type Buffer = CudaBuffer;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn determinism(&self) -> Determinism {
        Determinism::FixedOrder
    }

    fn max_batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.attention_batch_path.max_batch_size())
            .expect("CUDA batch size is nonzero")
    }

    /// Reports direct `cudaMalloc` bytes and counts CUDA library objects whose
    /// byte sizes are not exposed by the runtime API.
    fn memory_accounting(&self) -> MemoryAccounting {
        self.context
            .memory_accounting()
            .with_untracked(self.untracked)
    }

    fn memory_tracker_root(&self) -> MemoryTrackerRoot {
        self.context.memory_tracker_root()
    }

    fn classify_buffer(
        &mut self,
        buffer: &Self::Buffer,
        class: MemoryClass,
    ) -> Result<(), BackendError> {
        buffer.reclassify(class);
        Ok(())
    }

    fn set_memory_tracker(&mut self, tracker: MemoryTracker) -> Result<(), BackendError> {
        self.context
            .set_memory_tracker(tracker)
            .map_err(|error| cuda_error("set memory tracker", error))
    }

    fn set_memory_budget(&mut self, budget: MemoryBudget) -> Result<(), BackendError> {
        self.context
            .set_memory_budget(budget)
            .map_err(|error| cuda_error("set memory budget", error))
    }

    fn prefill_method(&self) -> PrefillMethod {
        PrefillMethod::TiledCublasLtFp16
    }

    fn q8_prefill_supported(&self) -> bool {
        true
    }

    fn memory_capacity(&mut self) -> Result<MemoryCapacity, BackendError> {
        let (available_bytes, total_bytes) = self
            .context
            .memory_info()
            .map_err(|error| cuda_error("query CUDA memory", error))?;
        Ok(MemoryCapacity::Limited {
            available_bytes: u64::try_from(available_bytes).map_err(|_| {
                BackendError::SizeOverflow {
                    field: "available backend memory",
                }
            })?,
            total_bytes: u64::try_from(total_bytes).map_err(|_| BackendError::SizeOverflow {
                field: "total backend memory",
            })?,
        })
    }

    fn model_import_metrics(&self) -> ModelImportMetrics {
        ModelImportMetrics {
            lossless_repack_source_bytes: self.q4_repack_source_bytes,
            lossless_repack_duration: self.q4_repack_duration,
        }
    }

    fn configure_rope(
        &mut self,
        head_dim: usize,
        theta: f32,
        frequency_factors: Option<&[f32]>,
        pairing: RopePairing,
    ) -> Result<(), BackendError> {
        self.rope_scratch = Some(
            RopeScratch::new(
                &self.context,
                head_dim,
                theta,
                frequency_factors,
                pairing == RopePairing::Adjacent,
            )
            .map_err(|error| cuda_error("configure RoPE", error))?,
        );
        Ok(())
    }

    fn prepare_rope(&mut self, position: Position<'_, Self::Buffer>) -> Result<(), BackendError> {
        self.profile_launches(1);
        let scratch = self
            .rope_scratch
            .as_mut()
            .ok_or_else(|| BackendError::operation("prepare RoPE", "RoPE is not configured"))?;
        match position {
            Position::Host(position) => scratch
                .prepare(&self.stream, position)
                .map_err(|error| cuda_error("prepare RoPE", error)),
            Position::Device(position) => scratch
                .prepare_device_position(&self.stream, position.u32()?)
                .map_err(|error| cuda_error("prepare device-position RoPE", error)),
        }
    }

    fn allocate(&mut self, layout: BufferLayout) -> Result<Self::Buffer, BackendError> {
        self.allocate_classified(layout, MemoryClass::ContractBuffer)
    }

    fn allocate_classified(
        &mut self,
        layout: BufferLayout,
        class: MemoryClass,
    ) -> Result<Self::Buffer, BackendError> {
        let storage = allocate_storage(&self.context, layout, class)?;
        Ok(CudaBuffer { layout, storage })
    }

    fn upload(&mut self, layout: BufferLayout, bytes: &[u8]) -> Result<Self::Buffer, BackendError> {
        upload_backend(self, layout, bytes, None)
    }

    fn upload_with_host_staging(
        &mut self,
        layout: BufferLayout,
        bytes: &[u8],
        staging: &HostStaging,
    ) -> Result<Self::Buffer, BackendError> {
        upload_backend(self, layout, bytes, Some(staging))
    }

    fn clone_buffer(&mut self, source: &Self::Buffer) -> Result<Self::Buffer, BackendError> {
        let storage = allocate_storage(&self.context, source.layout, source.memory_class())?;
        let mut destination = CudaBuffer {
            layout: source.layout,
            storage,
        };
        clone_storage(&self.stream, &source.storage, &mut destination.storage)?;
        Ok(destination)
    }

    fn download_buffer(&mut self, source: &Self::Buffer) -> Result<BufferSnapshot, BackendError> {
        // The synchronous copy uses CUDA's default stream. Fence Leone's
        // nonblocking stream first so queued device writes reach the snapshot.
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("synchronize before download", error))?;
        let bytes = download_storage(&source.storage, source.layout)?;
        self.check_span_error()?;
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
        let layout = source.layout();
        let mut destination = self.allocate_classified(layout, class)?;
        restore_storage(&self.stream, &mut destination.storage, source.bytes())?;
        Ok(destination)
    }

    fn write_u32(&mut self, buffer: &mut Self::Buffer, values: &[u32]) -> Result<(), BackendError> {
        if values.len() == 1 && buffer.layout.elements() == 1 {
            self.profile_launches(1);
            return write_u32_scalar(&self.stream, buffer.u32_mut()?, values[0])
                .map_err(|error| cuda_error("write u32 scalar", error));
        }
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.h2d_copies += 1;
            profiler.stream_synchronizations += 1;
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("fence u32 write", error))?;
        buffer
            .u32_mut()?
            .copy_from(values)
            .map_err(|error| cuda_error("write u32 buffer", error))
    }

    fn read_u32(&mut self, buffer: &Self::Buffer, values: &mut [u32]) -> Result<(), BackendError> {
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.d2h_copies += 1;
        }
        buffer
            .u32()?
            .copy_to_async(&self.stream, values)
            .map_err(|error| cuda_error("enqueue u32 read", error))?;
        let mut span_status = [0_u32; SPAN_ERROR_WORDS];
        let check_span = self.enqueue_span_error(&mut span_status)?;
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.stream_synchronizations += 1;
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("finish u32 read", error))?;
        if check_span {
            self.finish_span_error(span_status)
        } else {
            Ok(())
        }
    }

    fn read_f16(&mut self, buffer: &Self::Buffer, values: &mut [u16]) -> Result<(), BackendError> {
        buffer
            .f16()?
            .copy_to_async(&self.stream, values)
            .map_err(|error| cuda_error("enqueue f16 read", error))?;
        let mut span_status = [0_u32; SPAN_ERROR_WORDS];
        let check_span = self.enqueue_span_error(&mut span_status)?;
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("finish f16 read", error))?;
        if check_span {
            self.finish_span_error(span_status)
        } else {
            Ok(())
        }
    }

    fn read_f32(&mut self, buffer: &Self::Buffer, values: &mut [f32]) -> Result<(), BackendError> {
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.d2h_copies += 1;
        }
        buffer
            .f32()?
            .copy_to_async(&self.stream, values)
            .map_err(|error| cuda_error("enqueue f32 read", error))?;
        let mut span_status = [0_u32; SPAN_ERROR_WORDS];
        let check_span = self.enqueue_span_error(&mut span_status)?;
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.stream_synchronizations += 1;
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("finish f32 read", error))?;
        self.retire_eager_decode_workspaces();
        if check_span {
            self.finish_span_error(span_status)
        } else {
            Ok(())
        }
    }

    fn prepare_prefill(&mut self, plan: PrefillPlan) -> Result<PrefillWorkspace, BackendError> {
        self.validate_prefill_numerics(plan)?;
        if self.prefill_plan == Some(plan) {
            return self
                .prefill_scratch
                .as_ref()
                .map(PrefillScratch::usage)
                .ok_or_else(|| {
                    BackendError::operation("reuse prefill workspace", "scratch is missing")
                });
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("fence prefill workspace replacement", error))?;
        self.prefill_plan = None;
        if self.cublaslt.is_none() {
            self.cublaslt = Some(
                CublasLt::new(&self.context)
                    .map_err(|error| cuda_error("create cuBLASLt handle", error))?,
            );
            self.untracked.library_handles = self
                .untracked
                .library_handles
                .checked_add(1)
                .expect("library handle count overflow");
        }
        let usage = self.prepare_prefill_scratch(plan)?;
        self.prefill_plan = Some(plan);
        Ok(usage)
    }

    fn prefill_gemm(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError> {
        check_layout("prefill GEMM weights", shape.layout()?, weights.layout)?;
        let cuda_shape = quant_shape(shape)?;
        ensure_prefill_gemm_scratch(self, shape.columns(), cuda_shape)?;
        launch_prefill_gemm(self, weights, input, output, cuda_shape, tokens)
    }

    fn verify_supported(&self) -> bool {
        true
    }

    fn verify_gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.profile_launches(2);
        self.profile_gemv(shape)?;
        check_layout("verifier GEMV weights", shape.layout()?, weights.layout)?;
        let scratch = ensure_verify_gemv_scratch(
            &self.context,
            &mut self.verify_gemv_scratch,
            shape.columns(),
            positions,
        )?;
        crate::cuda::verify_gemv(
            &self.stream,
            weights.bytes()?,
            input.f32()?,
            None,
            output.f32_mut()?,
            scratch,
            quant_shape(shape)?,
            positions,
        )
        .map_err(|error| cuda_error("launch verifier GEMV", error))
    }

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
        self.profile_launches(2);
        profile_gemv_shapes(self, [first_shape, second_shape, third_shape])?;
        validate_verify_gemv_layouts(&[
            ("first verifier GEMV weights", first_weights, first_shape),
            ("second verifier GEMV weights", second_weights, second_shape),
            ("third verifier GEMV weights", third_weights, third_shape),
        ])?;
        let scratch = ensure_verify_gemv_scratch(
            &self.context,
            &mut self.verify_gemv_scratch,
            first_shape.columns(),
            positions,
        )?;
        let (first_cuda_shape, second_cuda_shape, third_cuda_shape) =
            verify_gemv_cuda_shapes3(first_shape, second_shape, third_shape)?;
        launch_verify_gemv_triple(
            &self.stream,
            first_weights,
            second_weights,
            third_weights,
            input,
            first_output,
            second_output,
            third_output,
            scratch,
            first_cuda_shape,
            second_cuda_shape,
            third_cuda_shape,
            positions,
        )
    }

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
        self.profile_launches(2);
        profile_gemv_shapes(self, [first_shape, second_shape])?;
        validate_verify_gemv_layouts(&[
            ("first verifier GEMV weights", first_weights, first_shape),
            ("second verifier GEMV weights", second_weights, second_shape),
        ])?;
        let scratch = ensure_verify_gemv_scratch(
            &self.context,
            &mut self.verify_gemv_scratch,
            first_shape.columns(),
            positions,
        )?;
        let (first_cuda_shape, second_cuda_shape) =
            verify_gemv_cuda_shapes2(first_shape, second_shape)?;
        launch_verify_gemv_pair(
            &self.stream,
            first_weights,
            second_weights,
            input,
            first_output,
            second_output,
            scratch,
            first_cuda_shape,
            second_cuda_shape,
            positions,
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
        self.profile_launches(2);
        self.profile_gemv(shape)?;
        check_layout(
            "verifier residual GEMV weights",
            shape.layout()?,
            weights.layout,
        )?;
        let scratch = ensure_verify_gemv_scratch(
            &self.context,
            &mut self.verify_gemv_scratch,
            shape.columns(),
            positions,
        )?;
        map_cuda_result(
            crate::cuda::verify_gemv(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                Some(residual.f32()?),
                output.f32_mut()?,
                scratch,
                quant_shape(shape)?,
                positions,
            ),
            "launch verifier residual GEMV",
        )
    }

    fn verify_gemv_residual_prepared(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        self.profile_gemv(shape)?;
        check_layout(
            "prepared verifier GEMV weights",
            shape.layout()?,
            weights.layout,
        )?;
        let scratch = prepared_verify_gemv_scratch(
            &mut self.verify_gemv_scratch,
            shape.columns(),
            positions,
        )?;
        map_cuda_result(
            crate::cuda::verify_gemv_prepared(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                Some(residual.f32()?),
                output.f32_mut()?,
                scratch,
                quant_shape(shape)?,
                positions,
            ),
            "launch prepared verifier residual GEMV",
        )
    }

    fn verify_swiglu(
        &mut self,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        output: &mut Self::Buffer,
        columns: usize,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        let key = (columns, positions);
        if !self.verify_gemv_scratch.contains_key(&key) {
            let scratch = GemvScratch::new_multi(&self.context, columns, positions)
                .map_err(|error| cuda_error("allocate verifier SwiGLU scratch", error))?;
            self.verify_gemv_scratch.insert(key, scratch);
        }
        let scratch = self.verify_gemv_scratch.get_mut(&key).ok_or_else(|| {
            BackendError::operation("find verifier SwiGLU scratch", "missing entry")
        })?;
        crate::cuda::verify_swiglu_q8(
            &self.stream,
            gate.f32()?,
            up.f32()?,
            output.f32_mut()?,
            scratch,
            columns,
            positions,
        )
        .map_err(|error| cuda_error("launch verifier SwiGLU", error))
    }

    fn gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        self.profile_gemv(shape)?;
        let input_prepared = self.take_prepared_input(shape.columns(), input)?;
        self.profile_launches(if input_prepared { 1 } else { 2 });
        check_layout("GEMV weights", shape.layout()?, weights.layout)?;
        let cuda_shape = quant_shape(shape)?;
        let scratch = ensure_gemv_scratch(
            &self.context,
            &mut self.gemv_scratch,
            shape.columns(),
            cuda_shape,
            "GEMV",
        )?;
        launch_gemv(
            GemvLaunch {
                stream: &self.stream,
                weights,
                input,
                output,
                scratch,
                shape: cuda_shape,
                input_prepared,
            },
            shape.format(),
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
        self.profile_gemv(shape)?;
        check_layout("residual GEMV weights", shape.layout()?, weights.layout)?;
        let input_prepared = self.take_prepared_input(shape.columns(), input)?;
        self.profile_launches(if input_prepared { 1 } else { 2 });
        let cuda_shape = quant_shape(shape)?;
        let scratch = ensure_gemv_scratch(
            &self.context,
            &mut self.gemv_scratch,
            shape.columns(),
            cuda_shape,
            "residual GEMV",
        )?;
        launch_gemv_residual(
            GemvResidualLaunch {
                stream: &self.stream,
                weights,
                input,
                residual,
                output,
                scratch,
                shape: cuda_shape,
                input_prepared,
            },
            shape.format(),
        )
    }

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
        if first_shape.format() != QuantFormat::Q4K || second_shape.format() != QuantFormat::Q4K {
            self.gemv(first_weights, input, first_output, first_shape)?;
            return self.gemv(second_weights, input, second_output, second_shape);
        }
        run_gemv_pair_q4(
            self,
            first_weights,
            first_shape,
            second_weights,
            second_shape,
            input,
            first_output,
            second_output,
        )
    }

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
        if !can_fuse_gemv_pair_swiglu(gate_shape, up_shape) {
            self.gemv_pair(
                gate_weights,
                gate_shape,
                up_weights,
                up_shape,
                input,
                gate,
                up,
            )?;
            return self.swiglu(gate, up, output);
        }
        run_gemv_pair_swiglu(
            self,
            gate_weights,
            gate_shape,
            up_weights,
            up_shape,
            input,
            gate,
            up,
            output,
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
        profile_gemv_shapes(self, [query_shape, key_shape, value_shape])?;
        let input_prepared = self.take_prepared_input(query_shape.columns(), input)?;
        self.profile_launches(if input_prepared { 1 } else { 2 });
        validate_qkv_layouts(
            query_weights,
            query_shape,
            key_weights,
            key_shape,
            value_weights,
            value_shape,
        )?;
        let (query_cuda_shape, key_cuda_shape, value_cuda_shape) =
            qkv_cuda_shapes(query_shape, key_shape, value_shape)?;
        let scratch = ensure_gemv_scratch(
            &self.context,
            &mut self.gemv_scratch,
            query_shape.columns(),
            query_cuda_shape,
            "find QKV GEMV scratch",
        )?;
        launch_qkv_gemv(
            &self.stream,
            query_weights,
            query_cuda_shape,
            key_weights,
            key_cuda_shape,
            value_weights,
            value_cuda_shape,
            input,
            query,
            key,
            value,
            scratch,
            input_prepared,
        )
    }

    fn rms_norm(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        let output_key = shape.columns();
        ensure_rms_norm_scratch(&self.context, &mut self.gemv_scratch, output_key)?;
        let scratch = rms_norm_scratch(&mut self.gemv_scratch, output_key)?;
        map_cuda_result(
            rms_norm_q8_parallel(
                &self.stream,
                input.f32()?,
                weight.f32()?,
                output.f32_mut()?,
                scratch,
                vector_shape(shape)?,
                epsilon,
            ),
            "launch parallel RMSNorm q8_1",
        )?;
        self.mark_prepared_input(output_key, output)
    }

    fn prefill_rms_norm(
        &mut self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: VectorShape,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        rms_norm(
            &self.stream,
            input.f32()?,
            weight.f32()?,
            output.f32_mut()?,
            vector_shape(shape)?,
            epsilon,
        )
        .map_err(|error| cuda_error("launch prefill RMSNorm", error))
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
        self.profile_launches(1);
        match position {
            Position::Host(position) => launch_rms_norm_rope_host(
                RmsNormRopeLaunch {
                    stream: &self.stream,
                    input,
                    weight,
                    output,
                    shape,
                    epsilon,
                    theta,
                },
                position,
            ),
            Position::Device(position) => launch_rms_norm_rope_device(
                RmsNormRopeLaunch {
                    stream: &self.stream,
                    input,
                    weight,
                    output,
                    shape,
                    epsilon,
                    theta,
                },
                position,
            ),
        }
    }

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
        _position: Position<'_, Self::Buffer>,
        epsilon: f32,
        _theta: f32,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        let scratch = configured_rope_scratch(self)?;
        let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
        let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
        map_cuda_result(
            qk_norm_rope(
                &self.stream,
                query.input,
                query.weight,
                query.output,
                query.shape,
                key.input,
                key.weight,
                key.output,
                key.shape,
                scratch,
                epsilon,
            ),
            "launch QK RMSNorm RoPE",
        )
    }

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
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
        epsilon: f32,
        theta: f32,
    ) -> Result<(), BackendError> {
        validate_fused_qk_shape(query_shape, key_shape, shape)?;
        validate_fused_kv_value(value, shape, self.context.device())?;
        let cache_kind = validate_fused_kv_cache(
            key_cache,
            value_cache,
            shape,
            position,
            self.context.device(),
        )?;
        if cache_kind == BufferStorage::Q8Kv {
            return qk_norm_rope_q8_fallback(
                self,
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
                shape,
                position,
                epsilon,
                theta,
            );
        }
        self.profile_launches(1);
        let scratch = self.rope_scratch.as_ref().ok_or_else(|| {
            BackendError::operation("launch QK RMSNorm RoPE", "RoPE is not configured")
        })?;
        let shape = attention_shape(shape)?;
        launch_qk_norm_rope_kv_append(
            &self.stream,
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
            shape,
            position,
            scratch,
            epsilon,
        )
    }

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
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
        epsilon: f32,
        _theta: f32,
    ) -> Result<(), BackendError> {
        self.ensure_span_execution_allowed()?;
        validate_fused_qk_shape(query_shape, key_shape, shape)?;
        validate_fused_kv_value(value, shape, self.context.device())?;
        let (cuda_shape, cache_kind, table_index, error_max_context) =
            prepare_fused_span_append(self, target, shape, position)?;
        if matches!(position, Position::Device(_)) {
            self.arm_span_error(error_max_context);
        }
        self.profile_launches(1);
        if cache_kind == BufferStorage::Q8Kv {
            return launch_q8_fused_span(
                self,
                query,
                query_weight,
                query_output,
                query_shape,
                key,
                key_weight,
                key_output,
                key_shape,
                value,
                table_index,
                cuda_shape,
                position,
                epsilon,
                _theta,
            );
        }
        launch_non_q8_fused_span(
            self,
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            value,
            table_index,
            cuda_shape,
            cache_kind,
            position,
            epsilon,
        )
    }

    fn verify_qk_norm_rope_kv_append(
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
        shape: AttentionShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        _theta: f32,
    ) -> Result<(), BackendError> {
        self.profile_launches(2);
        let scratch = self.rope_scratch.as_mut().ok_or_else(|| {
            BackendError::operation("launch verifier QK RMSNorm RoPE", "RoPE is not configured")
        })?;
        let shape = attention_shape(shape)?;
        launch_verify_qk_norm_rope_kv_append(
            &self.stream,
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
            shape,
            start_position,
            positions,
            scratch,
            epsilon,
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
        shape: AttentionShape,
        start_position: usize,
        positions: usize,
        epsilon: f32,
        _theta: f32,
    ) -> Result<(), BackendError> {
        self.ensure_span_execution_allowed()?;
        let (cuda_shape, cache_kind, table_index) =
            prepare_verify_qk_span(self, target, shape, start_position, positions)?;
        self.profile_launches(2);
        launch_backend_verify_span_qk(
            self,
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            value,
            cuda_shape,
            cache_kind,
            start_position,
            positions,
            table_index,
            epsilon,
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
        self.profile_launches(1);
        rms_norm_residual(
            &self.stream,
            left.f32()?,
            right.f32()?,
            weight.f32()?,
            output.f32_mut()?,
            vector_shape(shape)?,
            epsilon,
        )
        .map_err(|error| cuda_error("launch residual RMSNorm", error))
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
        self.profile_launches(1);
        rms_norm_residual_store(
            &self.stream,
            left.f32()?,
            right.f32()?,
            weight.f32()?,
            residual.f32_mut()?,
            output.f32_mut()?,
            vector_shape(shape)?,
            epsilon,
        )
        .map_err(|error| cuda_error("launch stored residual RMSNorm", error))
    }

    fn rope(
        &mut self,
        values: &mut Self::Buffer,
        position: usize,
        shape: RopeShape,
        _theta: f32,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        let scratch = self.rope_scratch.as_ref().ok_or_else(|| {
            BackendError::operation("launch configured RoPE", "RoPE is not configured")
        })?;
        rope_at_frequencies(
            &self.stream,
            values.f32_mut()?,
            position,
            rope_shape(shape)?,
            scratch,
        )
        .map_err(|error| cuda_error("launch RoPE", error))
    }

    fn rope_position(
        &mut self,
        values: &mut Self::Buffer,
        position: Position<'_, Self::Buffer>,
        shape: RopeShape,
        _theta: f32,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        let scratch = configured_rope_scratch(self)?;
        let shape = rope_shape(shape)?;
        let result = match position {
            Position::Host(position) => {
                rope_at_frequencies(&self.stream, values.f32_mut()?, position, shape, scratch)
            }
            Position::Device(position) => rope_at_frequencies_device_position(
                &self.stream,
                values.f32_mut()?,
                position.u32()?,
                shape,
                scratch,
            ),
        };
        map_cuda_result(result, "launch positioned RoPE")
    }

    fn swiglu(
        &mut self,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        swiglu(&self.stream, gate.f32()?, up.f32()?, output.f32_mut()?)
            .map_err(|error| cuda_error("launch SwiGLU", error))
    }

    fn residual_add(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        residual_add(&self.stream, left.f32()?, right.f32()?, output.f32_mut()?)
            .map_err(|error| cuda_error("launch residual add", error))
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
        self.profile_launches(1);
        let shape = attention_shape(shape)?;
        launch_kv_append(
            &self.stream,
            key,
            value,
            key_cache,
            value_cache,
            shape,
            position,
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
        let shape = attention_shape(shape)?;
        launch_kv_append_chunk(KvAppendChunk {
            stream: &self.stream,
            key,
            value,
            key_cache,
            value_cache,
            shape,
            start_position,
            tokens,
        })
    }

    fn kv_append_span(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        self.ensure_span_execution_allowed()?;
        let logical_shape = shape;
        let cuda_shape = attention_shape(logical_shape)?;
        validate_span_host_position(&target, logical_shape, position)?;
        let (descriptor, allocation, owners, key_cache, value_cache) =
            write_span_descriptor(target, logical_shape)?;
        if matches!(position, Position::Device(_)) {
            self.arm_span_error(shape.max_context());
        }
        let cache_kind = matching_span_storage("append KV span", key_cache, value_cache)?;
        validate_q8_span_shape(cache_kind, logical_shape)?;
        let table_index = self.retain_span_table(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&allocation),
            std::slice::from_ref(&owners),
        )?;
        self.profile_launches(1);
        let (table, span_error) = retained_span_table_with_error(
            self.graph_capture_active,
            &self.graph_span_tables,
            &self.eager_span_tables,
            &mut self.span_error,
            table_index,
        )?;
        launch_kv_span(
            &self.stream,
            key,
            value,
            table,
            cuda_shape,
            cache_kind,
            position,
            span_error,
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
        self.ensure_span_execution_allowed()?;
        let logical_shape = shape;
        let cuda_shape = attention_shape(logical_shape)?;
        validate_span_range(
            &target,
            logical_shape,
            start_position,
            tokens,
            "KV span append tokens",
        )?;
        let (descriptor, allocation, owners, key_cache, value_cache) =
            write_span_descriptor(target, logical_shape)?;
        let cache_kind = matching_span_storage("append KV span chunk", key_cache, value_cache)?;
        validate_q8_span_shape(cache_kind, logical_shape)?;
        let table_index = self.retain_span_table(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&allocation),
            std::slice::from_ref(&owners),
        )?;
        self.profile_launches(1);
        let table = self.retained_span_table(table_index)?;
        launch_kv_chunk_span(
            &self.stream,
            key,
            value,
            table,
            cuda_shape,
            cache_kind,
            start_position,
            tokens,
        )
    }

    fn prepare_kv_write_span(
        &mut self,
        target: KvWriteSpan<'_, Self::Buffer>,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        let (descriptor, allocation, owners, key_cache, value_cache) =
            write_span_descriptor(target, shape)?;
        let cache_kind = matching_span_storage("prepare KV write span", key_cache, value_cache)?;
        validate_q8_span_shape(cache_kind, shape)?;
        self.retain_span_table(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&allocation),
            std::slice::from_ref(&owners),
        )?;
        Ok(())
    }

    fn prepare_kv_read_view(
        &mut self,
        cache: KvReadView<'_, Self::Buffer>,
        shape: AttentionShape,
    ) -> Result<(), BackendError> {
        let _ = span_cache_kind(cache, shape)?;
        let table_input = read_span_descriptors(cache, shape)?;
        if self.has_retained_span_table(&table_input) {
            return Ok(());
        }
        self.retain_span_table_input(&table_input)?;
        Ok(())
    }

    fn prepare_attention_decode_batch_spans(
        &mut self,
        rows: &[AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        if rows.is_empty() {
            return Err(BackendError::Zero {
                field: "batch attention rows",
            });
        }
        if self.attention_batch_path.uses_native_kernel() && rows_use_host_positions(rows) {
            self.prepare_batch_decode_workspace(rows, self.attention_batch_path.uses_shared_reads())
        } else {
            for row in rows {
                let (cuda_shape, _, _, _) =
                    prepare_attention_decode_span(self, row.cache, row.shape, row.position)?;
                ensure_attention_scratch(self, row.shape, cuda_shape)?;
                let prepared = row.shape.head_dim() == PREPARED_ATTENTION_HEAD_DIM;
                let output_key = row.shape.query_elements()?;
                ensure_attention_output_scratch(self, prepared, output_key)?;
            }
            Ok(())
        }
    }

    fn attention_decode_batch_spans(
        &mut self,
        rows: &mut [AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        if rows.is_empty() {
            return Err(BackendError::Zero {
                field: "batch attention rows",
            });
        }
        self.ensure_span_execution_allowed()?;
        if !self.attention_batch_path.uses_native_kernel() || !rows_use_host_positions(rows) {
            return attention_decode_batch_scalar(self, rows);
        }
        self.dispatch_native_attention_decode_batch(rows)
    }

    fn retain_decode_graph_buffers(
        &mut self,
        buffers: &[&Self::Buffer],
    ) -> Result<(), BackendError> {
        self.retain_graph_buffer_owners(buffers)
    }

    fn begin_kv_graph_preflight(&mut self) -> Result<(), BackendError> {
        if self.graph_capture_active || self.graph_preflight_active {
            return Err(BackendError::operation(
                "prepare decode graph",
                "descriptor preflight is already active",
            ));
        }
        self.pending_graph_span_tables.clear();
        self.pending_graph_buffer_owners.clear();
        self.graph_preflight_active = true;
        Ok(())
    }

    fn end_kv_graph_preflight(&mut self, keep: bool) -> Result<(), BackendError> {
        if !self.graph_preflight_active {
            return Err(BackendError::operation(
                "finish decode graph preparation",
                "descriptor preflight is not active",
            ));
        }
        self.graph_preflight_active = false;
        if !keep {
            self.pending_graph_span_tables.clear();
        }
        Ok(())
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
        self.profile_launches(2);
        let cuda_shape = attention_shape(shape)?;
        ensure_attention_scratch(self, shape, cuda_shape)?;
        // Only head dimension 128 fills the q8_1 epilogue scratch for the output projection.
        let prepared = shape.head_dim() == PREPARED_ATTENTION_HEAD_DIM;
        let output_key = shape.query_elements()?;
        ensure_attention_output_scratch(self, prepared, output_key)?;
        run_attention_decode(
            self,
            query,
            key_cache,
            value_cache,
            output,
            shape,
            output_key,
            prepared,
            cuda_shape,
            position,
        )
    }

    fn attention_decode_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        self.ensure_span_execution_allowed()?;
        let (cuda_shape, cache_kind, table_index, mapped_tokens) =
            prepare_attention_decode_span(self, cache, shape, position)?;
        if matches!(position, Position::Device(_)) {
            self.arm_span_error(shape.max_context());
        }
        self.profile_launches(2);
        ensure_attention_scratch(self, shape, cuda_shape)?;
        let prepared = shape.head_dim() == PREPARED_ATTENTION_HEAD_DIM;
        let output_key = shape.query_elements()?;
        ensure_attention_output_scratch(self, prepared, output_key)?;
        run_backend_attention_decode_span(
            self,
            query,
            output,
            shape,
            output_key,
            prepared,
            cuda_shape,
            cache_kind,
            table_index,
            position,
            mapped_tokens,
        )?;
        if prepared {
            self.mark_prepared_input(output_key, output)?;
        }
        Ok(())
    }

    fn verifier_attention_prepares_output(&self, shape: AttentionShape) -> bool {
        attention_shape(shape)
            .map(crate::cuda::verifier_attention_prepares_output)
            .unwrap_or(false)
    }

    fn verify_attention(
        &mut self,
        query: &Self::Buffer,
        key_cache: &Self::Buffer,
        value_cache: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        start_position: usize,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.profile_launches(2);
        let cuda_shape = attention_shape(shape)?;
        ensure_verify_attention_scratch(self, shape, cuda_shape, positions)?;
        let prepares_output = crate::cuda::verifier_attention_prepares_output(cuda_shape);
        prepare_verify_attention_output(self, prepares_output, shape, positions)?;
        let scratch_key = (shape, positions);
        let scratch = self
            .verify_attention_scratch
            .get_mut(&scratch_key)
            .ok_or_else(|| {
                BackendError::operation("find verifier attention scratch", "missing entry")
            })?;
        let prepared_output = if prepares_output {
            let output_key = shape.query_elements()?;
            self.verify_gemv_scratch
                .get_mut(&(output_key, positions))
                .map(Some)
                .ok_or_else(|| {
                    BackendError::operation("find verifier attention q8_1 scratch", "missing entry")
                })?
        } else {
            None
        };
        launch_verify_attention(
            &self.stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            cuda_shape,
            start_position,
            positions,
        )
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
        self.ensure_span_execution_allowed()?;
        validate_span_attention_range(
            shape,
            cache.mapped_tokens(),
            start_position,
            positions,
            "span verifier positions",
        )?;
        let cuda_shape = attention_shape(shape)?;
        let cache_kind = span_cache_kind(cache, shape)?;
        let table_input = read_span_descriptors(cache, shape)?;
        let table_index = self.retain_span_table_input(&table_input)?;
        self.profile_launches(2);
        ensure_verify_attention_scratch(self, shape, cuda_shape, positions)?;
        let prepares_output = crate::cuda::verifier_attention_prepares_output(cuda_shape);
        prepare_verify_attention_output(self, prepares_output, shape, positions)?;
        run_backend_verify_attention_span(
            self,
            query,
            output,
            shape,
            positions,
            prepares_output,
            cuda_shape,
            cache_kind,
            table_index,
            start_position,
        )
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
        let cuda_shape = attention_shape(shape)?;
        ensure_attention_scratch(self, shape, cuda_shape)?;
        let handle = self.cublaslt.as_ref().ok_or_else(|| {
            BackendError::operation("run prefill attention", "cuBLASLt is not prepared")
        })?;
        let scratch = self.prefill_scratch.as_mut().ok_or_else(|| {
            BackendError::operation("run prefill attention", "prefill workspace is not prepared")
        })?;
        launch_attention_prefill(
            handle,
            &self.stream,
            query,
            key_cache,
            value_cache,
            output,
            cuda_shape,
            start_position,
            tokens,
            scratch,
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
        self.ensure_span_execution_allowed()?;
        validate_span_attention_range(
            shape,
            cache.mapped_tokens(),
            start_position,
            tokens,
            "span prefill tokens",
        )?;
        let cuda_shape = attention_shape(shape)?;
        let cache_kind = span_cache_kind(cache, shape)?;
        let table_input = read_span_descriptors(cache, shape)?;
        let table_index = self.retain_span_table_input(&table_input)?;
        self.profile_launches(1);
        ensure_attention_scratch(self, shape, cuda_shape)?;
        run_backend_attention_prefill_span(
            self,
            query,
            output,
            cuda_shape,
            start_position,
            tokens,
            cache_kind,
            table_index,
        )
    }

    fn embed_gather(
        &mut self,
        table: &Self::Buffer,
        row: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        launch_embed_gather(&self.stream, table, row, output, shape)
    }

    fn embed_gather_batch(
        &mut self,
        table: &Self::Buffer,
        rows: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError> {
        check_layout("prefill embedding table", shape.layout()?, table.layout)?;
        embedding_gather_batch(
            &self.stream,
            table.bytes()?,
            rows.u32()?,
            output.f32_mut()?,
            quant_shape(shape)?,
            tokens,
        )
        .map_err(|error| cuda_error("launch prefill embedding gather", error))
    }

    fn copy_f32_row(
        &mut self,
        input: &Self::Buffer,
        row: usize,
        columns: usize,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        copy_f32_row(&self.stream, input.f32()?, row, columns, output.f32_mut()?)
            .map_err(|error| cuda_error("copy prefill output row", error))
    }

    fn write_f32_row(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        row: usize,
        columns: usize,
    ) -> Result<(), BackendError> {
        write_f32_row(&self.stream, input.f32()?, output.f32_mut()?, row, columns)
            .map_err(|error| cuda_error("write batch input row", error))
    }

    fn argmax(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        self.profile_launches(2);
        let elements = input.layout.elements();
        if !self.argmax_scratch.contains_key(&elements) {
            let scratch = ArgmaxScratch::new(&self.context, elements)
                .map_err(|error| cuda_error("allocate argmax scratch", error))?;
            self.argmax_scratch.insert(elements, scratch);
        }
        let scratch = self
            .argmax_scratch
            .get_mut(&elements)
            .ok_or_else(|| BackendError::operation("find argmax scratch", "missing entry"))?;
        argmax(&self.stream, input.f32()?, output.u32_mut()?, scratch)
            .map_err(|error| cuda_error("launch argmax", error))
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.stream_synchronizations += 1;
        }
        let mut span_status = [0_u32; SPAN_ERROR_WORDS];
        let check_span = self.enqueue_span_error(&mut span_status)?;
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("synchronize CUDA", error))?;
        self.retire_eager_decode_workspaces();
        if check_span {
            self.finish_span_error(span_status)
        } else {
            Ok(())
        }
    }

    fn decode_graph_supported(&self) -> bool {
        !self.attention_batch_path.uses_native_kernel()
    }

    fn begin_decode_graph(&mut self) -> Result<(), BackendError> {
        if !self.decode_graph_supported() {
            return Err(BackendError::operation(
                "begin decode graph",
                "native batched attention uses eager workspaces",
            ));
        }
        if self.graph_preflight_active {
            self.pending_graph_buffer_owners.clear();
            return Err(BackendError::operation(
                "prepare decode graph",
                "descriptor preflight is still active",
            ));
        }
        let staged_tables = !self.pending_graph_span_tables.is_empty();
        if let Err(error) = self.validate_graph_table_count(staged_tables) {
            self.pending_graph_buffer_owners.clear();
            return Err(error);
        }
        if self.pending_graph_buffer_owners.is_empty() {
            return Err(BackendError::operation(
                "prepare decode graph",
                "dynamic graph buffers were not retained",
            ));
        }
        if let Err(error) = self.retire_decode_graph() {
            self.pending_graph_buffer_owners.clear();
            return Err(error);
        }
        if let Err(error) = self.retain_backend_graph_dependencies() {
            self.pending_graph_buffer_owners.clear();
            return Err(error);
        }
        self.capture_graph_buffer_owners = std::mem::take(&mut self.pending_graph_buffer_owners);
        self.stage_decode_graph_tables(staged_tables);
        self.start_graph_capture()
    }

    fn end_decode_graph(&mut self) -> Result<(), BackendError> {
        if !self.graph_capture_active {
            return Err(BackendError::operation(
                "end decode graph",
                "decode graph capture is not active",
            ));
        }
        let owners = std::mem::take(&mut self.capture_graph_buffer_owners);
        let graph = match self.stream.end_graph_capture_with_owners(owners) {
            Ok(graph) => graph,
            Err(error) => {
                self.graph_capture_active = false;
                self.graph_span_tables.clear();
                return Err(cuda_error("end decode graph", error));
            }
        };
        self.graph_capture_active = false;
        self.decode_graph = Some(graph);
        self.untracked.graph_objects = self
            .untracked
            .graph_objects
            .checked_add(1)
            .expect("graph object count overflow");
        Ok(())
    }

    fn replay_decode_graph(&mut self) -> Result<(), BackendError> {
        if let Some(max_context) = self.decode_graph_span_max_context {
            self.arm_span_error(max_context);
        }
        let graph = self.decode_graph.as_ref().ok_or_else(|| {
            BackendError::operation("replay decode graph", "no graph has been captured")
        })?;
        graph
            .launch(&self.stream)
            .map_err(|error| cuda_error("replay decode graph", error))
    }

    fn drop_decode_graph(&mut self) -> Result<(), BackendError> {
        if self.graph_capture_active {
            return Err(BackendError::operation(
                "drop decode graph",
                "decode graph capture is active",
            ));
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("retire decode graph", error))?;
        if self.decode_graph.take().is_some() {
            self.untracked.graph_objects = self
                .untracked
                .graph_objects
                .checked_sub(1)
                .expect("graph object count underflow");
        }
        self.capture_graph_buffer_owners.clear();
        self.pending_graph_buffer_owners.clear();
        self.graph_span_tables.clear();
        self.pending_graph_span_tables.clear();
        self.eager_span_tables.clear();
        self.batch_decode_workspaces.clear();
        self.decode_graph_span_max_context = None;
        self.graph_preflight_active = false;
        self.clear_span_error()
    }

    fn increment_u32(&mut self, buffer: &mut Self::Buffer) -> Result<(), BackendError> {
        self.profile_launches(1);
        increment_u32_scalar(&self.stream, buffer.u32_mut()?)
            .map_err(|error| cuda_error("increment u32", error))
    }

    fn begin_decode_profile(&mut self, operations: usize) -> Result<(), BackendError> {
        if self.decode_profiler.is_some() {
            return Err(BackendError::operation(
                "begin decode profile",
                "a decode profile is already active",
            ));
        }
        let event_count = operations
            .checked_add(1)
            .ok_or(BackendError::SizeOverflow {
                field: "decode profile event count",
            })?;
        let mut events = Vec::with_capacity(event_count);
        for _ in 0..event_count {
            events.push(
                Event::new(&self.context)
                    .map_err(|error| cuda_error("create decode profile event", error))?,
            );
        }
        self.untracked.execution_events = self
            .untracked
            .execution_events
            .checked_add(
                u64::try_from(event_count).map_err(|_| BackendError::SizeOverflow {
                    field: "decode profile event count",
                })?,
            )
            .ok_or(BackendError::SizeOverflow {
                field: "tracked decode profile events",
            })?;
        self.decode_profiler = Some(CudaDecodeProfiler {
            events,
            operations: Vec::with_capacity(operations),
            kernel_launches: 0,
            h2d_copies: 0,
            d2h_copies: 0,
            stream_synchronizations: 0,
        });
        Ok(())
    }

    fn profile_decode_op(&mut self, op: DecodeOp) -> Result<(), BackendError> {
        let Some(profiler) = &mut self.decode_profiler else {
            return Ok(());
        };
        let event = profiler
            .events
            .get_mut(profiler.operations.len())
            .ok_or_else(|| {
                BackendError::operation("record decode profile", "operation capacity exceeded")
            })?;
        event
            .record(&self.stream)
            .map_err(|error| cuda_error("record decode profile event", error))?;
        profiler.operations.push(ProfileOperation {
            class: op,
            gemvs: [None; 3],
        });
        Ok(())
    }

    fn end_decode_profile(
        &mut self,
        steps: usize,
        wall_duration: Duration,
    ) -> Result<Option<DecodeProfile>, BackendError> {
        let Some(mut profiler) = self.decode_profiler.take() else {
            return Ok(None);
        };
        self.untracked.execution_events = self
            .untracked
            .execution_events
            .checked_sub(u64::try_from(profiler.events.len()).map_err(|_| {
                BackendError::SizeOverflow {
                    field: "decode profile event count",
                }
            })?)
            .ok_or(BackendError::SizeOverflow {
                field: "tracked decode profile events",
            })?;
        let operation_count = profiler.operations.len();
        finish_decode_profile(&self.stream, &mut profiler, operation_count)?;
        let DecodeProfileCollections {
            gpu_duration_by_op,
            gemv_by_shape,
        } = collect_decode_profile(&profiler)?;
        Ok(Some(DecodeProfile {
            steps,
            wall_duration,
            gpu_duration_by_op,
            gemv_by_shape,
            kernel_launches: profiler.kernel_launches,
            h2d_copies: profiler.h2d_copies,
            d2h_copies: profiler.d2h_copies,
            stream_synchronizations: profiler.stream_synchronizations,
        }))
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        if self.graph_capture_active {
            let owners = std::mem::take(&mut self.capture_graph_buffer_owners);
            let graph = self.stream.end_graph_capture_with_owners(owners).ok();
            self.graph_capture_active = false;
            drop(graph);
        }
        let _ = self.stream.synchronize();
        self.decode_graph.take();
        self.capture_graph_buffer_owners.clear();
        self.pending_graph_buffer_owners.clear();
        self.graph_span_tables.clear();
        self.pending_graph_span_tables.clear();
        self.eager_span_tables.clear();
    }
}

fn finish_decode_profile(
    stream: &Stream,
    profiler: &mut CudaDecodeProfiler,
    operation_count: usize,
) -> Result<(), BackendError> {
    if operation_count == 0 {
        return Ok(());
    }
    let end = profiler.events.get_mut(operation_count).ok_or_else(|| {
        BackendError::operation("finish decode profile", "final event is missing")
    })?;
    end.record(stream)
        .map_err(|error| cuda_error("record final decode profile event", error))?;
    end.synchronize()
        .map_err(|error| cuda_error("wait for decode profile", error))
}

fn collect_decode_profile(
    profiler: &CudaDecodeProfiler,
) -> Result<DecodeProfileCollections, BackendError> {
    let mut gpu_duration_by_op = BTreeMap::new();
    let mut gemv_by_shape = BTreeMap::new();
    for (index, operation) in profiler.operations.iter().enumerate() {
        let duration = decode_operation_duration(profiler, index)?;
        *gpu_duration_by_op.entry(operation.class).or_default() += duration;
        collect_operation_gemvs(operation, duration, &mut gemv_by_shape)?;
    }
    Ok(DecodeProfileCollections {
        gpu_duration_by_op,
        gemv_by_shape,
    })
}

fn decode_operation_duration(
    profiler: &CudaDecodeProfiler,
    index: usize,
) -> Result<Duration, BackendError> {
    let milliseconds = Event::elapsed_ms(&profiler.events[index], &profiler.events[index + 1])
        .map_err(|error| cuda_error("measure decode profile event", error))?;
    Ok(Duration::from_secs_f64(f64::from(milliseconds) / 1_000.0))
}

fn collect_operation_gemvs(
    operation: &ProfileOperation,
    duration: Duration,
    gemv_by_shape: &mut BTreeMap<QuantMatrix, GemvProfile>,
) -> Result<(), BackendError> {
    let total_bytes = operation
        .gemvs
        .iter()
        .flatten()
        .try_fold(0_u64, |total, shape| {
            u64::try_from(shape.layout()?.bytes())
                .ok()
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or(BackendError::SizeOverflow {
                    field: "profiled GEMV bytes",
                })
        })?;
    for shape in operation.gemvs.iter().flatten() {
        let shape_bytes =
            u64::try_from(shape.layout()?.bytes()).map_err(|_| BackendError::SizeOverflow {
                field: "profiled GEMV bytes",
            })?;
        let share = if total_bytes == 0 {
            Duration::ZERO
        } else {
            duration.mul_f64(shape_bytes as f64 / total_bytes as f64)
        };
        let entry = gemv_by_shape.entry(*shape).or_insert(GemvProfile {
            calls: 0,
            gpu_duration: Duration::ZERO,
        });
        entry.calls += 1;
        entry.gpu_duration += share;
    }
    Ok(())
}

fn launch_kv_append(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: Position<'_, CudaBuffer>,
) -> Result<(), BackendError> {
    match (key_cache.layout.storage(), position) {
        (BufferStorage::F32, Position::Host(position)) => {
            launch_kv_append_f32_host(stream, key, value, key_cache, value_cache, shape, position)
        }
        (BufferStorage::F16, Position::Host(position)) => {
            launch_kv_append_f16_host(stream, key, value, key_cache, value_cache, shape, position)
        }
        (BufferStorage::F32, Position::Device(position)) => {
            launch_kv_append_f32_device(stream, key, value, key_cache, value_cache, shape, position)
        }
        (BufferStorage::F16, Position::Device(position)) => {
            launch_kv_append_f16_device(stream, key, value, key_cache, value_cache, shape, position)
        }
        (BufferStorage::Q8Kv, Position::Host(position)) => {
            launch_kv_append_q8_host(stream, key, value, key_cache, value_cache, shape, position)
        }
        (BufferStorage::Q8Kv, Position::Device(position)) => {
            launch_kv_append_q8_device(stream, key, value, key_cache, value_cache, shape, position)
        }
        (storage, _) => Err(storage_error("write KV cache", storage)),
    }
}

fn launch_kv_append_f32_host(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    kv_append(
        stream,
        key.f32()?,
        value.f32()?,
        key_cache.f32_mut()?,
        value_cache.f32_mut()?,
        shape,
        position,
    )
    .map_err(|error| cuda_error("launch KV append", error))
}

fn launch_kv_append_f16_host(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    kv_append_f16(
        stream,
        key.f32()?,
        value.f32()?,
        key_cache.f16_mut()?,
        value_cache.f16_mut()?,
        shape,
        position,
    )
    .map_err(|error| cuda_error("launch KV append", error))
}

fn launch_kv_append_f32_device(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
) -> Result<(), BackendError> {
    kv_append_device_position(
        stream,
        key.f32()?,
        value.f32()?,
        key_cache.f32_mut()?,
        value_cache.f32_mut()?,
        shape,
        position.u32()?,
    )
    .map_err(|error| cuda_error("launch KV append", error))
}

fn launch_kv_append_f16_device(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
) -> Result<(), BackendError> {
    kv_append_f16_device_position(
        stream,
        key.f32()?,
        value.f32()?,
        key_cache.f16_mut()?,
        value_cache.f16_mut()?,
        shape,
        position.u32()?,
    )
    .map_err(|error| cuda_error("launch KV append", error))
}

fn launch_kv_append_q8_host(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    kv_append_q8(
        stream,
        key.f32()?,
        value.f32()?,
        key_cache.bytes_mut()?,
        value_cache.bytes_mut()?,
        shape,
        position,
    )
    .map_err(|error| cuda_error("launch KV append", error))
}

fn launch_kv_append_q8_device(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
) -> Result<(), BackendError> {
    kv_append_q8_device_position(
        stream,
        key.f32()?,
        value.f32()?,
        key_cache.bytes_mut()?,
        value_cache.bytes_mut()?,
        shape,
        position.u32()?,
    )
    .map_err(|error| cuda_error("launch KV append", error))
}

struct KvAppendChunk<'a> {
    stream: &'a Stream,
    key: &'a CudaBuffer,
    value: &'a CudaBuffer,
    key_cache: &'a mut CudaBuffer,
    value_cache: &'a mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    tokens: usize,
}

struct AttentionDecodeArgs<'a> {
    stream: &'a Stream,
    query: &'a CudaBuffer,
    key_cache: &'a CudaBuffer,
    value_cache: &'a CudaBuffer,
    output: &'a mut CudaBuffer,
    scratch: &'a mut AttentionScratch,
    prepared_output: Option<&'a mut GemvScratch>,
    shape: crate::AttentionShape,
}

struct GemvLaunch<'a> {
    stream: &'a Stream,
    weights: &'a CudaBuffer,
    input: &'a CudaBuffer,
    output: &'a mut CudaBuffer,
    scratch: &'a mut GemvScratch,
    shape: crate::QuantizedMatrixShape,
    input_prepared: bool,
}

struct GemvResidualLaunch<'a> {
    stream: &'a Stream,
    weights: &'a CudaBuffer,
    input: &'a CudaBuffer,
    residual: &'a CudaBuffer,
    output: &'a mut CudaBuffer,
    scratch: &'a mut GemvScratch,
    shape: crate::QuantizedMatrixShape,
    input_prepared: bool,
}

struct RmsNormRopeLaunch<'a> {
    stream: &'a Stream,
    input: &'a CudaBuffer,
    weight: &'a CudaBuffer,
    output: &'a mut CudaBuffer,
    shape: VectorShape,
    epsilon: f32,
    theta: f32,
}

fn launch_kv_append_chunk(operation: KvAppendChunk<'_>) -> Result<(), BackendError> {
    let storage = operation.key_cache.layout.storage();
    match storage {
        BufferStorage::F32 => launch_kv_append_chunk_f32(operation),
        BufferStorage::F16 => launch_kv_append_chunk_f16(operation),
        BufferStorage::Q8Kv => launch_kv_append_chunk_q8(operation),
        storage => Err(storage_error("write prefill KV cache", storage)),
    }
}

fn launch_kv_append_chunk_f32(operation: KvAppendChunk<'_>) -> Result<(), BackendError> {
    kv_append_chunk(
        operation.stream,
        operation.key.f32()?,
        operation.value.f32()?,
        operation.key_cache.f32_mut()?,
        operation.value_cache.f32_mut()?,
        operation.shape,
        operation.start_position,
        operation.tokens,
    )
    .map_err(|error| cuda_error("launch prefill KV append", error))
}

fn launch_kv_append_chunk_f16(operation: KvAppendChunk<'_>) -> Result<(), BackendError> {
    kv_append_chunk_f16(
        operation.stream,
        operation.key.f32()?,
        operation.value.f32()?,
        operation.key_cache.f16_mut()?,
        operation.value_cache.f16_mut()?,
        operation.shape,
        operation.start_position,
        operation.tokens,
    )
    .map_err(|error| cuda_error("launch prefill KV append", error))
}

fn launch_kv_append_chunk_q8(operation: KvAppendChunk<'_>) -> Result<(), BackendError> {
    kv_append_chunk_q8(
        operation.stream,
        operation.key.f32()?,
        operation.value.f32()?,
        operation.key_cache.bytes_mut()?,
        operation.value_cache.bytes_mut()?,
        operation.shape,
        operation.start_position,
        operation.tokens,
    )
    .map_err(|error| cuda_error("launch prefill KV append", error))
}

fn ensure_attention_scratch(
    backend: &mut CudaBackend,
    shape: AttentionShape,
    cuda_shape: crate::AttentionShape,
) -> Result<(), BackendError> {
    if !backend.attention_scratch.contains_key(&shape) {
        let scratch = AttentionScratch::new(&backend.context, cuda_shape)
            .map_err(|error| cuda_error("allocate attention scratch", error))?;
        backend.attention_scratch.insert(shape, scratch);
    }
    Ok(())
}

fn ensure_attention_output_scratch(
    backend: &mut CudaBackend,
    prepared: bool,
    output_key: usize,
) -> Result<(), BackendError> {
    if prepared && !backend.gemv_scratch.contains_key(&output_key) {
        let output_shape = crate::QuantizedMatrixShape::new(1, output_key, crate::QuantFormat::Q4K)
            .map_err(|error| cuda_error("check attention q8_1 scratch", error))?;
        let scratch = GemvScratch::new(&backend.context, output_shape)
            .map_err(|error| cuda_error("allocate attention q8_1 scratch", error))?;
        backend.gemv_scratch.insert(output_key, scratch);
    }
    Ok(())
}

fn ensure_verify_attention_scratch(
    backend: &mut CudaBackend,
    shape: AttentionShape,
    cuda_shape: crate::AttentionShape,
    positions: usize,
) -> Result<(), BackendError> {
    let key = (shape, positions);
    if !backend.verify_attention_scratch.contains_key(&key) {
        let scratch = AttentionScratch::new_multi(&backend.context, cuda_shape, positions)
            .map_err(|error| cuda_error("allocate verifier attention scratch", error))?;
        backend.verify_attention_scratch.insert(key, scratch);
    }
    Ok(())
}

fn prepare_verify_attention_output(
    backend: &mut CudaBackend,
    prepares_output: bool,
    shape: AttentionShape,
    positions: usize,
) -> Result<(), BackendError> {
    if !prepares_output {
        return Ok(());
    }
    let output_key = shape.query_elements()?;
    let gemv_key = (output_key, positions);
    if !backend.verify_gemv_scratch.contains_key(&gemv_key) {
        let gemv_scratch = GemvScratch::new_multi(&backend.context, output_key, positions)
            .map_err(|error| cuda_error("allocate verifier attention q8_1 scratch", error))?;
        backend.verify_gemv_scratch.insert(gemv_key, gemv_scratch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_attention(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<(), BackendError> {
    match key_cache.layout.storage() {
        BufferStorage::F32 => launch_verify_attention_f32(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            start_position,
            positions,
        ),
        BufferStorage::F16 => launch_verify_attention_f16(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            start_position,
            positions,
        ),
        storage => Err(storage_error("read verifier KV cache", storage)),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_attention_f32(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<(), BackendError> {
    crate::cuda::verify_attention(
        stream,
        query.f32()?,
        key_cache.f32()?,
        value_cache.f32()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        start_position,
        positions,
    )
    .map_err(|error| cuda_error("launch verifier attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_attention_f16(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<(), BackendError> {
    crate::cuda::verify_attention_f16(
        stream,
        query.f32()?,
        key_cache.f16()?,
        value_cache.f16()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        start_position,
        positions,
    )
    .map_err(|error| cuda_error("launch verifier attention", error))
}

fn launch_q8_span_append(
    stream: &Stream,
    key_output: &CudaBuffer,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    position: Position<'_, CudaBuffer>,
    error: &mut DeviceBuffer<u32>,
) -> Result<(), BackendError> {
    let result = match position {
        Position::Host(position) => kv_append_span_q8(
            stream,
            key_output.f32()?,
            value.f32()?,
            table,
            shape,
            position,
        ),
        Position::Device(position) => kv_append_span_q8_device_position(
            stream,
            key_output.f32()?,
            value.f32()?,
            table,
            shape,
            position.u32()?,
            error,
        ),
    };
    result.map_err(|error| cuda_error("launch span fused q8 KV append", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_span_qk_append(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: Position<'_, CudaBuffer>,
    scratch: &RopeScratch,
    epsilon: f32,
    error: &mut DeviceBuffer<u32>,
) -> Result<(), BackendError> {
    let value = value.f32()?;
    match position {
        Position::Host(position) => launch_span_qk_host(
            stream,
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            value,
            table,
            shape,
            cache_kind,
            position,
            scratch,
            epsilon,
        ),
        Position::Device(position) => launch_span_qk_device(
            stream,
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            value,
            table,
            shape,
            cache_kind,
            position.u32()?,
            scratch,
            epsilon,
            error,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_span_qk_host(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: usize,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let result = match cache_kind {
        BufferStorage::F32 => qk_norm_rope_kv_append_span(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            value,
            table,
            shape,
            position,
            scratch,
            epsilon,
        ),
        BufferStorage::F16 => qk_norm_rope_kv_append_span_f16(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            value,
            table,
            shape,
            position,
            scratch,
            epsilon,
        ),
        _ => unreachable!("span fused QK storage is checked"),
    };
    result.map_err(|error| cuda_error("launch span fused QK append", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_span_qk_device(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: &DeviceBuffer<u32>,
    scratch: &RopeScratch,
    epsilon: f32,
    error: &mut DeviceBuffer<u32>,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let result = match cache_kind {
        BufferStorage::F32 => qk_norm_rope_kv_append_span_device_position(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            value,
            table,
            shape,
            position,
            scratch,
            epsilon,
            error,
        ),
        BufferStorage::F16 => qk_norm_rope_kv_append_span_f16_device_position(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            value,
            table,
            shape,
            position,
            scratch,
            epsilon,
            error,
        ),
        _ => unreachable!("span fused QK storage is checked"),
    };
    result.map_err(|error| cuda_error("launch device-position span fused QK append", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_backend_verify_span_qk(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    start_position: usize,
    positions: usize,
    table_index: usize,
    epsilon: f32,
) -> Result<(), BackendError> {
    let scratch = backend.rope_scratch.as_mut().ok_or_else(|| {
        BackendError::operation(
            "launch span verifier fused QK append",
            "RoPE is not configured",
        )
    })?;
    let table = if backend.graph_capture_active {
        backend
            .graph_span_tables
            .get(table_index)
            .map(KvSpanTable::device)
            .ok_or_else(|| {
                BackendError::operation("find retained KV span table", "missing entry")
            })?
    } else {
        backend
            .eager_span_tables
            .get(table_index)
            .map(KvSpanTable::device)
            .ok_or_else(|| {
                BackendError::operation("find retained KV span table", "missing entry")
            })?
    };
    launch_verify_span_qk(
        &backend.stream,
        query,
        query_weight,
        query_output,
        query_shape,
        key,
        key_weight,
        key_output,
        key_shape,
        value,
        table,
        shape,
        cache_kind,
        start_position,
        positions,
        scratch,
        epsilon,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_span_qk(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    match cache_kind {
        BufferStorage::F32 => launch_verify_span_qk_f32(
            stream,
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            value,
            table,
            shape,
            start_position,
            positions,
            scratch,
            epsilon,
        ),
        BufferStorage::F16 => launch_verify_span_qk_f16(
            stream,
            query,
            query_weight,
            query_output,
            query_shape,
            key,
            key_weight,
            key_output,
            key_shape,
            value,
            table,
            shape,
            start_position,
            positions,
            scratch,
            epsilon,
        ),
        _ => unreachable!("span verifier QK storage is checked"),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_span_qk_f32(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let result = verify_qk_norm_rope_kv_append_span(
        stream,
        query.input,
        query.weight,
        query.output,
        query.shape,
        key.input,
        key.weight,
        key.output,
        key.shape,
        value.f32()?,
        table,
        shape,
        start_position,
        positions,
        scratch,
        epsilon,
    );
    result.map_err(|error| cuda_error("launch span verifier fused QK append", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_span_qk_f16(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let result = verify_qk_norm_rope_kv_append_span_f16(
        stream,
        query.input,
        query.weight,
        query.output,
        query.shape,
        key.input,
        key.weight,
        key.output,
        key.shape,
        value.f32()?,
        table,
        shape,
        start_position,
        positions,
        scratch,
        epsilon,
    );
    result.map_err(|error| cuda_error("launch span verifier fused QK append", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_span(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: Position<'_, CudaBuffer>,
    error: &mut DeviceBuffer<u32>,
) -> Result<(), BackendError> {
    match position {
        Position::Host(position) => {
            launch_kv_span_host(stream, key, value, table, shape, cache_kind, position)
        }
        Position::Device(position) => launch_kv_span_device(
            stream, key, value, table, shape, cache_kind, position, error,
        ),
    }
}

fn launch_kv_span_host(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: usize,
) -> Result<(), BackendError> {
    let key = key.f32()?;
    let value = value.f32()?;
    let result = match cache_kind {
        BufferStorage::F32 => kv_append_span(stream, key, value, table, shape, position),
        BufferStorage::F16 => kv_append_span_f16(stream, key, value, table, shape, position),
        BufferStorage::Q8Kv => kv_append_span_q8(stream, key, value, table, shape, position),
        _ => unreachable!("KV span storage is checked"),
    };
    result.map_err(|error| cuda_error("append KV span", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_span_device(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: &CudaBuffer,
    error: &mut DeviceBuffer<u32>,
) -> Result<(), BackendError> {
    let key = key.f32()?;
    let value = value.f32()?;
    let position = position.u32()?;
    let result = match cache_kind {
        BufferStorage::F32 => {
            kv_append_span_device_position(stream, key, value, table, shape, position, error)
        }
        BufferStorage::F16 => {
            kv_append_span_f16_device_position(stream, key, value, table, shape, position, error)
        }
        BufferStorage::Q8Kv => {
            kv_append_span_q8_device_position(stream, key, value, table, shape, position, error)
        }
        _ => unreachable!("KV span storage is checked"),
    };
    result.map_err(|error| cuda_error("append KV span", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_chunk_span(
    stream: &Stream,
    key: &CudaBuffer,
    value: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    start_position: usize,
    tokens: usize,
) -> Result<(), BackendError> {
    let key = key.f32()?;
    let value = value.f32()?;
    let result = match cache_kind {
        BufferStorage::F32 => {
            kv_append_chunk_span(stream, key, value, table, shape, start_position, tokens)
        }
        BufferStorage::F16 => {
            kv_append_chunk_span_f16(stream, key, value, table, shape, start_position, tokens)
        }
        BufferStorage::Q8Kv => {
            kv_append_chunk_span_q8(stream, key, value, table, shape, start_position, tokens)
        }
        _ => unreachable!("KV span storage is checked"),
    };
    result.map_err(|error| cuda_error("append KV span chunk", error))
}

#[allow(clippy::too_many_arguments)]
fn run_verify_attention_spans(
    stream: &Stream,
    query: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    start_position: usize,
    positions: usize,
) -> Result<(), BackendError> {
    let result = match cache_kind {
        BufferStorage::F32 => verify_attention_spans(
            stream,
            query.f32()?,
            table,
            output.f32_mut()?,
            scratch,
            prepared_output,
            shape,
            start_position,
            positions,
        ),
        BufferStorage::F16 => verify_attention_spans_f16(
            stream,
            query.f32()?,
            table,
            output.f32_mut()?,
            scratch,
            prepared_output,
            shape,
            start_position,
            positions,
        ),
        _ => unreachable!("span verifier storage is checked before launch"),
    };
    result.map_err(|error| cuda_error("launch span verifier attention", error))
}

fn launch_attention_decode(
    args: AttentionDecodeArgs<'_>,
    position: Position<'_, CudaBuffer>,
) -> Result<(), BackendError> {
    let AttentionDecodeArgs {
        stream,
        query,
        key_cache,
        value_cache,
        output,
        scratch,
        prepared_output,
        shape,
    } = args;
    match (key_cache.layout.storage(), position) {
        (BufferStorage::F32, Position::Host(position)) => launch_attention_decode_f32_host(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            position,
        ),
        (BufferStorage::F16, Position::Host(position)) => launch_attention_decode_f16_host(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            position,
        ),
        (BufferStorage::F32, Position::Device(position)) => launch_attention_decode_f32_device(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            position,
        ),
        (BufferStorage::F16, Position::Device(position)) => launch_attention_decode_f16_device(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            position,
        ),
        (BufferStorage::Q8Kv, Position::Host(position)) => launch_attention_decode_q8_host(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            position,
        ),
        (BufferStorage::Q8Kv, Position::Device(position)) => launch_attention_decode_q8_device(
            stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape,
            position,
        ),
        (storage, _) => Err(storage_error("read KV cache", storage)),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_decode_f32_host(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    let context_length = attention_context_length(position)?;
    attention_decode(
        stream,
        query.f32()?,
        key_cache.f32()?,
        value_cache.f32()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        context_length,
    )
    .map_err(|error| cuda_error("launch attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_decode_f16_host(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    let context_length = attention_context_length(position)?;
    attention_decode_f16(
        stream,
        query.f32()?,
        key_cache.f16()?,
        value_cache.f16()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        context_length,
    )
    .map_err(|error| cuda_error("launch attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_decode_f32_device(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
) -> Result<(), BackendError> {
    attention_decode_device_position(
        stream,
        query.f32()?,
        key_cache.f32()?,
        value_cache.f32()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        position.u32()?,
    )
    .map_err(|error| cuda_error("launch attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_decode_f16_device(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
) -> Result<(), BackendError> {
    attention_decode_f16_device_position(
        stream,
        query.f32()?,
        key_cache.f16()?,
        value_cache.f16()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        position.u32()?,
    )
    .map_err(|error| cuda_error("launch attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_decode_q8_host(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    let context_length = attention_context_length(position)?;
    attention_decode_q8(
        stream,
        query.f32()?,
        key_cache.bytes()?,
        value_cache.bytes()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        context_length,
    )
    .map_err(|error| cuda_error("launch attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_decode_q8_device(
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
) -> Result<(), BackendError> {
    attention_decode_q8_device_position(
        stream,
        query.f32()?,
        key_cache.bytes()?,
        value_cache.bytes()?,
        output.f32_mut()?,
        scratch,
        prepared_output,
        shape,
        position.u32()?,
    )
    .map_err(|error| cuda_error("launch attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_prefill(
    handle: &CublasLt,
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<(), BackendError> {
    match key_cache.layout.storage() {
        BufferStorage::F32 => launch_attention_prefill_f32(
            handle,
            stream,
            query,
            key_cache,
            value_cache,
            output,
            shape,
            start_position,
            tokens,
            scratch,
        ),
        BufferStorage::F16 => launch_attention_prefill_f16(
            handle,
            stream,
            query,
            key_cache,
            value_cache,
            output,
            shape,
            start_position,
            tokens,
            scratch,
        ),
        BufferStorage::Q8Kv => launch_attention_prefill_q8(
            handle,
            stream,
            query,
            key_cache,
            value_cache,
            output,
            shape,
            start_position,
            tokens,
            scratch,
        ),
        storage => Err(storage_error("read prefill KV cache", storage)),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_prefill_f32(
    handle: &CublasLt,
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<(), BackendError> {
    attention_prefill_f32(
        handle,
        stream,
        query.f32()?,
        key_cache.f32()?,
        value_cache.f32()?,
        output.f32_mut()?,
        shape,
        start_position,
        tokens,
        scratch,
    )
    .map_err(|error| cuda_error("run prefill attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_prefill_f16(
    handle: &CublasLt,
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<(), BackendError> {
    attention_prefill_f16(
        handle,
        stream,
        query.f32()?,
        key_cache.f16()?,
        value_cache.f16()?,
        output.f32_mut()?,
        shape,
        start_position,
        tokens,
        scratch,
    )
    .map_err(|error| cuda_error("run prefill attention", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_prefill_q8(
    handle: &CublasLt,
    stream: &Stream,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<(), BackendError> {
    attention_prefill_q8(
        handle,
        stream,
        query.f32()?,
        key_cache.bytes()?,
        value_cache.bytes()?,
        output.f32_mut()?,
        shape,
        start_position,
        tokens,
        scratch,
    )
    .map_err(|error| cuda_error("run prefill attention", error))
}

fn upload_backend(
    backend: &mut CudaBackend,
    layout: BufferLayout,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaBuffer, BackendError> {
    if bytes.len() != layout.bytes() {
        return Err(BackendError::SizeMismatch {
            name: "uploaded bytes",
            expected: layout.bytes(),
            actual: bytes.len(),
        });
    }
    let storage = upload_storage(backend, layout.storage(), bytes, staging)?;
    Ok(CudaBuffer { layout, storage })
}

#[allow(clippy::too_many_arguments)]
fn run_backend_attention_prefill_span(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    tokens: usize,
    cache_kind: BufferStorage,
    table_index: usize,
) -> Result<(), BackendError> {
    let handle = backend.cublaslt.as_ref().ok_or_else(|| {
        BackendError::operation("run span prefill attention", "cuBLASLt is not prepared")
    })?;
    let table = if backend.graph_capture_active {
        backend
            .graph_span_tables
            .get(table_index)
            .map(KvSpanTable::device)
            .ok_or_else(|| {
                BackendError::operation("find retained KV span table", "missing entry")
            })?
    } else {
        backend
            .eager_span_tables
            .get(table_index)
            .map(KvSpanTable::device)
            .ok_or_else(|| {
                BackendError::operation("find retained KV span table", "missing entry")
            })?
    };
    let scratch = backend.prefill_scratch.as_mut().ok_or_else(|| {
        BackendError::operation(
            "run span prefill attention",
            "prefill workspace is not prepared",
        )
    })?;
    run_attention_prefill_spans(
        handle,
        &backend.stream,
        query,
        table,
        output,
        shape,
        start_position,
        tokens,
        scratch,
        cache_kind,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_attention_prefill_spans(
    handle: &CublasLt,
    stream: &Stream,
    query: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    output: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
    cache_kind: BufferStorage,
) -> Result<(), BackendError> {
    let query = query.f32()?;
    let output = output.f32_mut()?;
    let result = match cache_kind {
        BufferStorage::F32 => attention_prefill_spans_f32(
            handle,
            stream,
            query,
            table,
            output,
            shape,
            start_position,
            tokens,
            scratch,
        ),
        BufferStorage::F16 => attention_prefill_spans_f16(
            handle,
            stream,
            query,
            table,
            output,
            shape,
            start_position,
            tokens,
            scratch,
        ),
        BufferStorage::Q8Kv => attention_prefill_spans_q8(
            handle,
            stream,
            query,
            table,
            output,
            shape,
            start_position,
            tokens,
            scratch,
        ),
        _ => unreachable!("span prefill storage is checked before launch"),
    };
    result.map_err(|error| cuda_error("run span prefill attention", error))
}

fn upload_storage(
    backend: &mut CudaBackend,
    storage: BufferStorage,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaStorage, BackendError> {
    match storage {
        BufferStorage::F16 | BufferStorage::F32 | BufferStorage::U32 => {
            upload_dense_storage(backend, storage, bytes, staging)
        }
        BufferStorage::Q4K | BufferStorage::Q8Kv | BufferStorage::Q6K => {
            upload_quantized_storage(backend, storage, bytes, staging)
        }
    }
}

fn ensure_gemv_scratch<'a>(
    context: &Context,
    scratches: &'a mut BTreeMap<usize, GemvScratch>,
    columns: usize,
    shape: crate::QuantizedMatrixShape,
    operation: &'static str,
) -> Result<&'a mut GemvScratch, BackendError> {
    if let Entry::Vacant(entry) = scratches.entry(columns) {
        let scratch = GemvScratch::new(context, shape)
            .map_err(|error| cuda_error("allocate GEMV scratch", error))?;
        entry.insert(scratch);
    }
    scratches
        .get_mut(&columns)
        .ok_or_else(|| BackendError::operation(operation, "missing entry"))
}

fn launch_gemv(operation: GemvLaunch<'_>, format: QuantFormat) -> Result<(), BackendError> {
    let GemvLaunch {
        stream,
        weights,
        input,
        output,
        scratch,
        shape,
        input_prepared,
    } = operation;
    match format {
        QuantFormat::Q4K => launch_gemv_q4(
            stream,
            weights,
            input,
            output,
            scratch,
            shape,
            input_prepared,
        ),
        QuantFormat::Q6K => launch_gemv_q6(
            stream,
            weights,
            input,
            output,
            scratch,
            shape,
            input_prepared,
        ),
    }
}

fn launch_gemv_q4(
    stream: &Stream,
    weights: &CudaBuffer,
    input: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    shape: crate::QuantizedMatrixShape,
    input_prepared: bool,
) -> Result<(), BackendError> {
    let result = if input_prepared {
        gemv_q4_k_residual(
            stream,
            weights.bytes()?,
            input.f32()?,
            None,
            output.f32_mut()?,
            scratch,
            shape,
            true,
        )
    } else {
        gemv_q4_k(
            stream,
            weights.bytes()?,
            input.f32()?,
            output.f32_mut()?,
            scratch,
            shape,
        )
    };
    result.map_err(|error| cuda_error("launch GEMV", error))
}

fn launch_gemv_q6(
    stream: &Stream,
    weights: &CudaBuffer,
    input: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    shape: crate::QuantizedMatrixShape,
    input_prepared: bool,
) -> Result<(), BackendError> {
    let result = if input_prepared {
        gemv_q6_k_residual(
            stream,
            weights.bytes()?,
            input.f32()?,
            None,
            output.f32_mut()?,
            scratch,
            shape,
            true,
        )
    } else {
        gemv_q6_k(
            stream,
            weights.bytes()?,
            input.f32()?,
            output.f32_mut()?,
            scratch,
            shape,
        )
    };
    result.map_err(|error| cuda_error("launch GEMV", error))
}

fn launch_gemv_residual(
    operation: GemvResidualLaunch<'_>,
    format: QuantFormat,
) -> Result<(), BackendError> {
    let GemvResidualLaunch {
        stream,
        weights,
        input,
        residual,
        output,
        scratch,
        shape,
        input_prepared,
    } = operation;
    match format {
        QuantFormat::Q4K => launch_gemv_residual_q4(
            stream,
            weights,
            input,
            residual,
            output,
            scratch,
            shape,
            input_prepared,
        ),
        QuantFormat::Q6K => launch_gemv_residual_q6(
            stream,
            weights,
            input,
            residual,
            output,
            scratch,
            shape,
            input_prepared,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv_residual_q4(
    stream: &Stream,
    weights: &CudaBuffer,
    input: &CudaBuffer,
    residual: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    shape: crate::QuantizedMatrixShape,
    input_prepared: bool,
) -> Result<(), BackendError> {
    map_cuda_result(
        gemv_q4_k_residual(
            stream,
            weights.bytes()?,
            input.f32()?,
            Some(residual.f32()?),
            output.f32_mut()?,
            scratch,
            shape,
            input_prepared,
        ),
        "launch residual GEMV",
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv_residual_q6(
    stream: &Stream,
    weights: &CudaBuffer,
    input: &CudaBuffer,
    residual: &CudaBuffer,
    output: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    shape: crate::QuantizedMatrixShape,
    input_prepared: bool,
) -> Result<(), BackendError> {
    map_cuda_result(
        gemv_q6_k_residual(
            stream,
            weights.bytes()?,
            input.f32()?,
            Some(residual.f32()?),
            output.f32_mut()?,
            scratch,
            shape,
            input_prepared,
        ),
        "launch residual GEMV",
    )
}

fn validate_gemv_pair_layouts(
    first_weights: &CudaBuffer,
    first_shape: QuantMatrix,
    second_weights: &CudaBuffer,
    second_shape: QuantMatrix,
) -> Result<(), BackendError> {
    check_layout(
        "first paired GEMV weights",
        first_shape.layout()?,
        first_weights.layout,
    )?;
    check_layout(
        "second paired GEMV weights",
        second_shape.layout()?,
        second_weights.layout,
    )
}

fn pair_cuda_shapes(
    first_shape: QuantMatrix,
    second_shape: QuantMatrix,
) -> Result<(crate::QuantizedMatrixShape, crate::QuantizedMatrixShape), BackendError> {
    Ok((quant_shape(first_shape)?, quant_shape(second_shape)?))
}

#[allow(clippy::too_many_arguments)]
fn run_gemv_pair_q4(
    backend: &mut CudaBackend,
    first_weights: &CudaBuffer,
    first_shape: QuantMatrix,
    second_weights: &CudaBuffer,
    second_shape: QuantMatrix,
    input: &CudaBuffer,
    first_output: &mut CudaBuffer,
    second_output: &mut CudaBuffer,
) -> Result<(), BackendError> {
    backend.profile_gemv(first_shape)?;
    backend.profile_gemv(second_shape)?;
    let input_prepared = backend.take_prepared_input(first_shape.columns(), input)?;
    backend.profile_launches(if input_prepared { 1 } else { 2 });
    validate_gemv_pair_layouts(first_weights, first_shape, second_weights, second_shape)?;
    let (first_cuda_shape, second_cuda_shape) = pair_cuda_shapes(first_shape, second_shape)?;
    let scratch = ensure_gemv_scratch(
        &backend.context,
        &mut backend.gemv_scratch,
        first_shape.columns(),
        first_cuda_shape,
        "find paired GEMV scratch",
    )?;
    launch_gemv_pair_q4(
        &backend.stream,
        first_weights,
        first_cuda_shape,
        second_weights,
        second_cuda_shape,
        input,
        first_output,
        second_output,
        scratch,
        input_prepared,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv_pair_q4(
    stream: &Stream,
    first_weights: &CudaBuffer,
    first_shape: crate::QuantizedMatrixShape,
    second_weights: &CudaBuffer,
    second_shape: crate::QuantizedMatrixShape,
    input: &CudaBuffer,
    first_output: &mut CudaBuffer,
    second_output: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    input_prepared: bool,
) -> Result<(), BackendError> {
    gemv_pair_q4_k(
        stream,
        first_weights.bytes()?,
        first_shape,
        second_weights.bytes()?,
        second_shape,
        input.f32()?,
        first_output.f32_mut()?,
        second_output.f32_mut()?,
        scratch,
        input_prepared,
    )
    .map_err(|error| cuda_error("launch paired GEMV", error))
}

fn can_fuse_gemv_pair_swiglu(gate: QuantMatrix, up: QuantMatrix) -> bool {
    gate.format() == QuantFormat::Q4K
        && up.format() == QuantFormat::Q4K
        && gate.rows() == up.rows()
        && gate.columns() == up.columns()
        && gate.rows().is_multiple_of(32)
        && gate.columns() != gate.rows()
}

fn profile_gemv_shapes<const N: usize>(
    backend: &mut CudaBackend,
    shapes: [QuantMatrix; N],
) -> Result<(), BackendError> {
    for shape in shapes {
        backend.profile_gemv(shape)?;
    }
    Ok(())
}

fn validate_verify_gemv_layouts(
    layouts: &[(&'static str, &CudaBuffer, QuantMatrix)],
) -> Result<(), BackendError> {
    for (name, weights, shape) in layouts {
        check_layout(name, shape.layout()?, weights.layout)?;
    }
    Ok(())
}

fn ensure_verify_gemv_scratch<'a>(
    context: &Context,
    scratches: &'a mut BTreeMap<(usize, usize), GemvScratch>,
    columns: usize,
    positions: usize,
) -> Result<&'a mut GemvScratch, BackendError> {
    let key = (columns, positions);
    if let Entry::Vacant(entry) = scratches.entry(key) {
        let scratch = GemvScratch::new_multi(context, columns, positions)
            .map_err(|error| cuda_error("allocate verifier GEMV scratch", error))?;
        entry.insert(scratch);
    }
    scratches
        .get_mut(&key)
        .ok_or_else(|| BackendError::operation("find verifier GEMV scratch", "missing entry"))
}

fn prepared_verify_gemv_scratch(
    scratches: &mut BTreeMap<(usize, usize), GemvScratch>,
    columns: usize,
    positions: usize,
) -> Result<&mut GemvScratch, BackendError> {
    scratches.get_mut(&(columns, positions)).ok_or_else(|| {
        BackendError::operation("find prepared verifier GEMV scratch", "missing entry")
    })
}

fn verify_gemv_cuda_shapes3(
    first: QuantMatrix,
    second: QuantMatrix,
    third: QuantMatrix,
) -> Result<
    (
        crate::QuantizedMatrixShape,
        crate::QuantizedMatrixShape,
        crate::QuantizedMatrixShape,
    ),
    BackendError,
> {
    Ok((
        quant_shape(first)?,
        quant_shape(second)?,
        quant_shape(third)?,
    ))
}

fn verify_gemv_cuda_shapes2(
    first: QuantMatrix,
    second: QuantMatrix,
) -> Result<(crate::QuantizedMatrixShape, crate::QuantizedMatrixShape), BackendError> {
    Ok((quant_shape(first)?, quant_shape(second)?))
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_gemv_triple(
    stream: &Stream,
    first_weights: &CudaBuffer,
    second_weights: &CudaBuffer,
    third_weights: &CudaBuffer,
    input: &CudaBuffer,
    first_output: &mut CudaBuffer,
    second_output: &mut CudaBuffer,
    third_output: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    first_shape: crate::QuantizedMatrixShape,
    second_shape: crate::QuantizedMatrixShape,
    third_shape: crate::QuantizedMatrixShape,
    positions: usize,
) -> Result<(), BackendError> {
    crate::cuda::verify_gemv_triple(
        stream,
        first_weights.bytes()?,
        second_weights.bytes()?,
        third_weights.bytes()?,
        input.f32()?,
        first_output.f32_mut()?,
        second_output.f32_mut()?,
        third_output.f32_mut()?,
        scratch,
        first_shape,
        second_shape,
        third_shape,
        positions,
    )
    .map_err(|error| cuda_error("launch verifier GEMV triple", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_gemv_pair(
    stream: &Stream,
    first_weights: &CudaBuffer,
    second_weights: &CudaBuffer,
    input: &CudaBuffer,
    first_output: &mut CudaBuffer,
    second_output: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    first_shape: crate::QuantizedMatrixShape,
    second_shape: crate::QuantizedMatrixShape,
    positions: usize,
) -> Result<(), BackendError> {
    crate::cuda::verify_gemv_pair(
        stream,
        first_weights.bytes()?,
        second_weights.bytes()?,
        input.f32()?,
        first_output.f32_mut()?,
        second_output.f32_mut()?,
        scratch,
        first_shape,
        second_shape,
        positions,
    )
    .map_err(|error| cuda_error("launch verifier GEMV pair", error))
}

#[allow(clippy::too_many_arguments)]
fn validate_qkv_layouts(
    query_weights: &CudaBuffer,
    query_shape: QuantMatrix,
    key_weights: &CudaBuffer,
    key_shape: QuantMatrix,
    value_weights: &CudaBuffer,
    value_shape: QuantMatrix,
) -> Result<(), BackendError> {
    check_layout("query weights", query_shape.layout()?, query_weights.layout)?;
    check_layout("key weights", key_shape.layout()?, key_weights.layout)?;
    check_layout("value weights", value_shape.layout()?, value_weights.layout)
}

fn qkv_cuda_shapes(
    query_shape: QuantMatrix,
    key_shape: QuantMatrix,
    value_shape: QuantMatrix,
) -> Result<
    (
        crate::QuantizedMatrixShape,
        crate::QuantizedMatrixShape,
        crate::QuantizedMatrixShape,
    ),
    BackendError,
> {
    Ok((
        quant_shape(query_shape)?,
        quant_shape(key_shape)?,
        quant_shape(value_shape)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn launch_qkv_gemv(
    stream: &Stream,
    query_weights: &CudaBuffer,
    query_shape: crate::QuantizedMatrixShape,
    key_weights: &CudaBuffer,
    key_shape: crate::QuantizedMatrixShape,
    value_weights: &CudaBuffer,
    value_shape: crate::QuantizedMatrixShape,
    input: &CudaBuffer,
    query: &mut CudaBuffer,
    key: &mut CudaBuffer,
    value: &mut CudaBuffer,
    scratch: &mut GemvScratch,
    input_prepared: bool,
) -> Result<(), BackendError> {
    qkv_gemv(
        stream,
        query_weights.bytes()?,
        query_shape,
        key_weights.bytes()?,
        key_shape,
        value_weights.bytes()?,
        value_shape,
        input.f32()?,
        query.f32_mut()?,
        key.f32_mut()?,
        value.f32_mut()?,
        scratch,
        input_prepared,
    )
    .map_err(|error| cuda_error("launch QKV GEMV", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_non_q8_fused_span(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    table_index: usize,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: Position<'_, CudaBuffer>,
    epsilon: f32,
) -> Result<(), BackendError> {
    let (table, span_error) = retained_span_table_with_error(
        backend.graph_capture_active,
        &backend.graph_span_tables,
        &backend.eager_span_tables,
        &mut backend.span_error,
        table_index,
    )?;
    let scratch = backend.rope_scratch.as_ref().ok_or_else(|| {
        BackendError::operation("launch span fused QK append", "RoPE is not configured")
    })?;
    launch_span_qk_append(
        &backend.stream,
        query,
        query_weight,
        query_output,
        query_shape,
        key,
        key_weight,
        key_output,
        key_shape,
        value,
        table,
        shape,
        cache_kind,
        position,
        scratch,
        epsilon,
        span_error,
    )
}

fn prepare_verify_qk_span(
    backend: &mut CudaBackend,
    target: KvWriteSpan<'_, CudaBuffer>,
    shape: AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<(crate::AttentionShape, BufferStorage, usize), BackendError> {
    let cuda_shape = attention_shape(shape)?;
    validate_span_range(
        &target,
        shape,
        start_position,
        positions,
        "verifier KV append positions",
    )?;
    let (descriptor, allocation, owners, key_cache, value_cache) =
        write_span_descriptor(target, shape)?;
    let cache_kind = matching_span_storage(
        "launch span verifier fused QK append",
        key_cache,
        value_cache,
    )?;
    if cache_kind == BufferStorage::Q8Kv {
        return Err(storage_error(
            "launch span verifier fused QK append",
            cache_kind,
        ));
    }
    let table_index = backend.retain_span_table(
        std::slice::from_ref(&descriptor),
        std::slice::from_ref(&allocation),
        std::slice::from_ref(&owners),
    )?;
    Ok((cuda_shape, cache_kind, table_index))
}

fn prepare_fused_span_append(
    backend: &mut CudaBackend,
    target: KvWriteSpan<'_, CudaBuffer>,
    shape: AttentionShape,
    position: Position<'_, CudaBuffer>,
) -> Result<(crate::AttentionShape, BufferStorage, usize, usize), BackendError> {
    let cuda_shape = attention_shape(shape)?;
    validate_span_host_position(&target, shape, position)?;
    let (descriptor, allocation, owners, key_cache, value_cache) =
        write_span_descriptor(target, shape)?;
    let cache_kind = matching_span_storage("launch span fused QK append", key_cache, value_cache)?;
    validate_q8_span_shape(cache_kind, shape)?;
    let table_index = backend.retain_span_table(
        std::slice::from_ref(&descriptor),
        std::slice::from_ref(&allocation),
        std::slice::from_ref(&owners),
    )?;
    Ok((cuda_shape, cache_kind, table_index, shape.max_context()))
}

#[allow(clippy::too_many_arguments)]
fn launch_q8_fused_span(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    table_index: usize,
    shape: crate::AttentionShape,
    position: Position<'_, CudaBuffer>,
    epsilon: f32,
    theta: f32,
) -> Result<(), BackendError> {
    backend.qk_norm_rope(
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
    backend.profile_launches(1);
    let (table, span_error) = retained_span_table_with_error(
        backend.graph_capture_active,
        &backend.graph_span_tables,
        &backend.eager_span_tables,
        &mut backend.span_error,
        table_index,
    )?;
    launch_q8_span_append(
        &backend.stream,
        key_output,
        value,
        table,
        shape,
        position,
        span_error,
    )
}

#[allow(clippy::too_many_arguments)]
fn qk_norm_rope_q8_fallback(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: AttentionShape,
    position: Position<'_, CudaBuffer>,
    epsilon: f32,
    theta: f32,
) -> Result<(), BackendError> {
    backend.qk_norm_rope(
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
    backend.kv_append(key_output, value, key_cache, value_cache, shape, position)
}

#[allow(clippy::too_many_arguments)]
fn launch_qk_norm_rope_kv_append(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: Position<'_, CudaBuffer>,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    match (key_cache.layout.storage(), position) {
        (BufferStorage::F32, Position::Host(position)) => launch_qk_norm_rope_kv_append_f32_host(
            stream,
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
            shape,
            position,
            scratch,
            epsilon,
        ),
        (BufferStorage::F16, Position::Host(position)) => launch_qk_norm_rope_kv_append_f16_host(
            stream,
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
            shape,
            position,
            scratch,
            epsilon,
        ),
        (BufferStorage::F32, Position::Device(position)) => {
            launch_qk_norm_rope_kv_append_f32_device(
                stream,
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
                shape,
                position,
                scratch,
                epsilon,
            )
        }
        (BufferStorage::F16, Position::Device(position)) => {
            launch_qk_norm_rope_kv_append_f16_device(
                stream,
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
                shape,
                position,
                scratch,
                epsilon,
            )
        }
        (storage, _) => Err(storage_error("write KV cache", storage)),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_qk_norm_rope_kv_append_f32_host(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: usize,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let cache = qk_kv_f32_inputs(value, key_cache, value_cache)?;
    map_cuda_result(
        qk_norm_rope_kv_append(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            cache.value,
            cache.key_cache,
            cache.value_cache,
            shape,
            position,
            scratch,
            epsilon,
        ),
        "launch QK RMSNorm RoPE with KV append",
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_qk_norm_rope_kv_append_f16_host(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: usize,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let cache = qk_kv_f16_inputs(value, key_cache, value_cache)?;
    map_cuda_result(
        qk_norm_rope_kv_append_f16(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            cache.value,
            cache.key_cache,
            cache.value_cache,
            shape,
            position,
            scratch,
            epsilon,
        ),
        "launch QK RMSNorm RoPE with KV append",
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_qk_norm_rope_kv_append_f32_device(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let cache = qk_kv_f32_inputs(value, key_cache, value_cache)?;
    let position = position.u32()?;
    map_cuda_result(
        qk_norm_rope_kv_append_device_position(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            cache.value,
            cache.key_cache,
            cache.value_cache,
            shape,
            position,
            scratch,
            epsilon,
        ),
        "launch QK RMSNorm RoPE with KV append",
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_qk_norm_rope_kv_append_f16_device(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    position: &CudaBuffer,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let cache = qk_kv_f16_inputs(value, key_cache, value_cache)?;
    let position = position.u32()?;
    map_cuda_result(
        qk_norm_rope_kv_append_f16_device_position(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            cache.value,
            cache.key_cache,
            cache.value_cache,
            shape,
            position,
            scratch,
            epsilon,
        ),
        "launch QK RMSNorm RoPE with KV append",
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_qk_norm_rope_kv_append(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    match key_cache.layout.storage() {
        BufferStorage::F32 => launch_verify_qk_norm_rope_kv_append_f32(
            stream,
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
            shape,
            start_position,
            positions,
            scratch,
            epsilon,
        ),
        BufferStorage::F16 => launch_verify_qk_norm_rope_kv_append_f16(
            stream,
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
            shape,
            start_position,
            positions,
            scratch,
            epsilon,
        ),
        storage => Err(storage_error("write verifier KV cache", storage)),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_qk_norm_rope_kv_append_f32(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let cache = qk_kv_f32_inputs(value, key_cache, value_cache)?;
    map_cuda_result(
        crate::cuda::verify_qk_norm_rope_kv_append(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            cache.value,
            cache.key_cache,
            cache.value_cache,
            shape,
            start_position,
            positions,
            scratch,
            epsilon,
        ),
        "launch verifier QK RMSNorm RoPE",
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_qk_norm_rope_kv_append_f16(
    stream: &Stream,
    query: &CudaBuffer,
    query_weight: &CudaBuffer,
    query_output: &mut CudaBuffer,
    query_shape: VectorShape,
    key: &CudaBuffer,
    key_weight: &CudaBuffer,
    key_output: &mut CudaBuffer,
    key_shape: VectorShape,
    value: &CudaBuffer,
    key_cache: &mut CudaBuffer,
    value_cache: &mut CudaBuffer,
    shape: crate::AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<(), BackendError> {
    let query = qk_norm_inputs(query, query_weight, query_output, query_shape)?;
    let key = qk_norm_inputs(key, key_weight, key_output, key_shape)?;
    let cache = qk_kv_f16_inputs(value, key_cache, value_cache)?;
    map_cuda_result(
        crate::cuda::verify_qk_norm_rope_kv_append_f16(
            stream,
            query.input,
            query.weight,
            query.output,
            query.shape,
            key.input,
            key.weight,
            key.output,
            key.shape,
            cache.value,
            cache.key_cache,
            cache.value_cache,
            shape,
            start_position,
            positions,
            scratch,
            epsilon,
        ),
        "launch verifier QK RMSNorm RoPE",
    )
}

fn validate_fused_gemv_layouts(
    gate_weights: &CudaBuffer,
    gate_shape: QuantMatrix,
    up_weights: &CudaBuffer,
    up_shape: QuantMatrix,
) -> Result<(), BackendError> {
    check_layout(
        "fused gate GEMV weights",
        gate_shape.layout()?,
        gate_weights.layout,
    )?;
    check_layout(
        "fused up GEMV weights",
        up_shape.layout()?,
        up_weights.layout,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_gemv_pair_swiglu(
    backend: &mut CudaBackend,
    gate_weights: &CudaBuffer,
    gate_shape: QuantMatrix,
    up_weights: &CudaBuffer,
    up_shape: QuantMatrix,
    input: &CudaBuffer,
    gate: &mut CudaBuffer,
    up: &mut CudaBuffer,
    output: &mut CudaBuffer,
) -> Result<(), BackendError> {
    backend.profile_gemv(gate_shape)?;
    backend.profile_gemv(up_shape)?;
    validate_fused_gemv_layouts(gate_weights, gate_shape, up_weights, up_shape)?;
    let (gate_cuda_shape, up_cuda_shape) = pair_cuda_shapes(gate_shape, up_shape)?;
    let input_key = gate_shape.columns();
    let output_key = gate_shape.rows();
    let input_prepared = backend.take_prepared_input(input_key, input)?;
    backend.profile_launches(if input_prepared { 1 } else { 2 });
    ensure_gemv_scratch(
        &backend.context,
        &mut backend.gemv_scratch,
        input_key,
        gate_cuda_shape,
        "find fused input scratch",
    )?;
    let mut output_scratch = take_fused_output_scratch(backend, output_key)?;
    let result = launch_gemv_pair_swiglu_with_scratch(
        backend,
        gate_weights,
        gate_cuda_shape,
        up_weights,
        up_cuda_shape,
        input,
        gate,
        up,
        output,
        input_key,
        &mut output_scratch,
        input_prepared,
    );
    backend.gemv_scratch.insert(output_key, output_scratch);
    result?;
    backend.mark_prepared_input(output_key, output)
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv_pair_swiglu_with_scratch(
    backend: &mut CudaBackend,
    gate_weights: &CudaBuffer,
    gate_shape: crate::QuantizedMatrixShape,
    up_weights: &CudaBuffer,
    up_shape: crate::QuantizedMatrixShape,
    input: &CudaBuffer,
    gate: &mut CudaBuffer,
    up: &mut CudaBuffer,
    output: &mut CudaBuffer,
    input_key: usize,
    output_scratch: &mut GemvScratch,
    input_prepared: bool,
) -> Result<(), BackendError> {
    let input_scratch = backend
        .gemv_scratch
        .get_mut(&input_key)
        .ok_or_else(|| BackendError::operation("find fused input scratch", "missing entry"))?;
    launch_fused_gemv_pair_swiglu(
        &backend.stream,
        gate_weights,
        gate_shape,
        up_weights,
        up_shape,
        input,
        gate,
        up,
        output,
        input_scratch,
        output_scratch,
        input_prepared,
    )
}

fn take_fused_output_scratch(
    backend: &mut CudaBackend,
    output_key: usize,
) -> Result<GemvScratch, BackendError> {
    if !backend.gemv_scratch.contains_key(&output_key) {
        let output_shape = crate::QuantizedMatrixShape::new(1, output_key, crate::QuantFormat::Q4K)
            .map_err(|error| cuda_error("check fused output scratch", error))?;
        let scratch = GemvScratch::new(&backend.context, output_shape)
            .map_err(|error| cuda_error("allocate fused output scratch", error))?;
        backend.gemv_scratch.insert(output_key, scratch);
    }
    backend
        .gemv_scratch
        .remove(&output_key)
        .ok_or_else(|| BackendError::operation("take fused output scratch", "missing entry"))
}

#[allow(clippy::too_many_arguments)]
fn launch_fused_gemv_pair_swiglu(
    stream: &Stream,
    gate_weights: &CudaBuffer,
    gate_shape: crate::QuantizedMatrixShape,
    up_weights: &CudaBuffer,
    up_shape: crate::QuantizedMatrixShape,
    input: &CudaBuffer,
    gate: &mut CudaBuffer,
    up: &mut CudaBuffer,
    output: &mut CudaBuffer,
    input_scratch: &mut GemvScratch,
    output_scratch: &mut GemvScratch,
    input_prepared: bool,
) -> Result<(), BackendError> {
    gemv_pair_swiglu_q4_k(
        stream,
        gate_weights.bytes()?,
        gate_shape,
        up_weights.bytes()?,
        up_shape,
        input.f32()?,
        gate.f32_mut()?,
        up.f32_mut()?,
        output.f32_mut()?,
        input_scratch,
        output_scratch,
        input_prepared,
    )
    .map_err(|error| cuda_error("launch fused gate, up, and SwiGLU GEMV", error))
}

struct QkNormInputs<'a> {
    input: &'a DeviceBuffer<f32>,
    weight: &'a DeviceBuffer<f32>,
    output: &'a mut DeviceBuffer<f32>,
    shape: crate::VectorShape,
}

struct QkKvF32Inputs<'a> {
    value: &'a DeviceBuffer<f32>,
    key_cache: &'a mut DeviceBuffer<f32>,
    value_cache: &'a mut DeviceBuffer<f32>,
}

struct QkKvF16Inputs<'a> {
    value: &'a DeviceBuffer<f32>,
    key_cache: &'a mut DeviceBuffer<u16>,
    value_cache: &'a mut DeviceBuffer<u16>,
}

struct EmbeddingGatherInputs<'a> {
    table: &'a DeviceBuffer<u8>,
    row: &'a DeviceBuffer<u32>,
    output: &'a mut DeviceBuffer<f32>,
}

fn qk_norm_inputs<'a>(
    input: &'a CudaBuffer,
    weight: &'a CudaBuffer,
    output: &'a mut CudaBuffer,
    shape: VectorShape,
) -> Result<QkNormInputs<'a>, BackendError> {
    Ok(QkNormInputs {
        input: input.f32()?,
        weight: weight.f32()?,
        output: output.f32_mut()?,
        shape: vector_shape(shape)?,
    })
}

fn qk_kv_f32_inputs<'a>(
    value: &'a CudaBuffer,
    key_cache: &'a mut CudaBuffer,
    value_cache: &'a mut CudaBuffer,
) -> Result<QkKvF32Inputs<'a>, BackendError> {
    Ok(QkKvF32Inputs {
        value: value.f32()?,
        key_cache: key_cache.f32_mut()?,
        value_cache: value_cache.f32_mut()?,
    })
}

fn qk_kv_f16_inputs<'a>(
    value: &'a CudaBuffer,
    key_cache: &'a mut CudaBuffer,
    value_cache: &'a mut CudaBuffer,
) -> Result<QkKvF16Inputs<'a>, BackendError> {
    Ok(QkKvF16Inputs {
        value: value.f32()?,
        key_cache: key_cache.f16_mut()?,
        value_cache: value_cache.f16_mut()?,
    })
}

fn embedding_gather_inputs<'a>(
    table: &'a CudaBuffer,
    row: &'a CudaBuffer,
    output: &'a mut CudaBuffer,
) -> Result<EmbeddingGatherInputs<'a>, BackendError> {
    Ok(EmbeddingGatherInputs {
        table: table.bytes()?,
        row: row.u32()?,
        output: output.f32_mut()?,
    })
}

fn launch_prefill_gemm(
    backend: &mut CudaBackend,
    weights: &CudaBuffer,
    input: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: crate::QuantizedMatrixShape,
    tokens: usize,
) -> Result<(), BackendError> {
    let handle = backend
        .cublaslt
        .as_ref()
        .ok_or_else(|| BackendError::operation("run prefill GEMM", "cuBLASLt is not prepared"))?;
    let scratch = backend.prefill_scratch.as_mut().ok_or_else(|| {
        BackendError::operation("run prefill GEMM", "prefill workspace is not prepared")
    })?;
    map_cuda_result(
        prefill_gemm(
            handle,
            &backend.stream,
            weights.bytes()?,
            input.f32()?,
            output.f32_mut()?,
            shape,
            tokens,
            scratch,
        ),
        "run prefill GEMM",
    )
}

fn ensure_rms_norm_scratch(
    context: &Context,
    scratches: &mut BTreeMap<usize, GemvScratch>,
    output_key: usize,
) -> Result<(), BackendError> {
    if let Entry::Vacant(entry) = scratches.entry(output_key) {
        let output_shape = map_cuda_result(
            crate::QuantizedMatrixShape::new(1, output_key, crate::QuantFormat::Q4K),
            "check RMSNorm q8_1 scratch",
        )?;
        let scratch = map_cuda_result(
            GemvScratch::new(context, output_shape),
            "allocate RMSNorm q8_1 scratch",
        )?;
        entry.insert(scratch);
    }
    Ok(())
}

fn rms_norm_scratch(
    scratches: &mut BTreeMap<usize, GemvScratch>,
    output_key: usize,
) -> Result<&mut GemvScratch, BackendError> {
    scratches
        .get_mut(&output_key)
        .ok_or_else(|| BackendError::operation("find RMSNorm q8_1 scratch", "missing entry"))
}

fn launch_rms_norm_rope_host(
    operation: RmsNormRopeLaunch<'_>,
    position: usize,
) -> Result<(), BackendError> {
    let RmsNormRopeLaunch {
        stream,
        input,
        weight,
        output,
        shape,
        epsilon,
        theta,
    } = operation;
    map_cuda_result(
        rms_norm_rope(
            stream,
            input.f32()?,
            weight.f32()?,
            output.f32_mut()?,
            vector_shape(shape)?,
            position,
            epsilon,
            theta,
        ),
        "launch RMSNorm RoPE",
    )
}

fn launch_rms_norm_rope_device(
    operation: RmsNormRopeLaunch<'_>,
    position: &CudaBuffer,
) -> Result<(), BackendError> {
    let RmsNormRopeLaunch {
        stream,
        input,
        weight,
        output,
        shape,
        epsilon,
        theta,
    } = operation;
    map_cuda_result(
        rms_norm_rope_device_position(
            stream,
            input.f32()?,
            weight.f32()?,
            output.f32_mut()?,
            vector_shape(shape)?,
            position.u32()?,
            epsilon,
            theta,
        ),
        "launch device-position RMSNorm RoPE",
    )
}

fn configured_rope_scratch(backend: &CudaBackend) -> Result<&RopeScratch, BackendError> {
    backend
        .rope_scratch
        .as_ref()
        .ok_or_else(|| BackendError::operation("launch QK RMSNorm RoPE", "RoPE is not configured"))
}

fn span_error_from_status(status: [u32; SPAN_ERROR_WORDS], max_context: usize) -> BackendError {
    let (position, start, end, kind) = match decode_span_error_status(status) {
        Ok(values) => values,
        Err(error) => return error,
    };
    if kind == 1 {
        return BackendError::PositionOutsideSpan {
            position,
            start,
            end,
        };
    }
    if kind == 0 {
        return BackendError::PositionOutOfBounds {
            position: position.checked_add(1).unwrap_or(position),
            max_context: end,
        };
    }
    if start == 0 && start < end {
        return BackendError::PositionOutOfBounds {
            position: position.checked_add(1).unwrap_or(position),
            max_context: end,
        };
    }
    if start < end {
        return BackendError::PositionOutsideSpan {
            position,
            start,
            end,
        };
    }
    BackendError::PositionOutOfBounds {
        position,
        max_context,
    }
}

fn decode_span_error_status(
    status: [u32; SPAN_ERROR_WORDS],
) -> Result<(usize, usize, usize, u32), BackendError> {
    let position = usize::try_from(status[1]).map_err(|_| BackendError::SizeOverflow {
        field: "CUDA span error position",
    })?;
    let start = u64::from(status[2]) | (u64::from(status[3]) << 32);
    let start = usize::try_from(start).map_err(|_| BackendError::SizeOverflow {
        field: "CUDA span error start",
    })?;
    let end = u64::from(status[4]) | (u64::from(status[5]) << 32);
    let end = usize::try_from(end).map_err(|_| BackendError::SizeOverflow {
        field: "CUDA span error end",
    })?;
    Ok((position, start, end, status[6]))
}

fn span_cache_kind(
    cache: KvReadView<'_, CudaBuffer>,
    shape: AttentionShape,
) -> Result<BufferStorage, BackendError> {
    let first = cache.spans().first().ok_or(BackendError::Zero {
        field: "KV read spans",
    })?;
    let storage = first.key().layout.storage();
    if first.value().layout.storage() != storage {
        return Err(BackendError::operation(
            "read KV spans",
            "key and value cache storage differs",
        ));
    }
    for span in cache.spans().iter().skip(1) {
        if span.key().layout.storage() != storage || span.value().layout.storage() != storage {
            return Err(BackendError::operation(
                "read KV spans",
                "all spans must use one cache storage",
            ));
        }
    }
    validate_q8_span_shape(storage, shape)?;
    match storage {
        BufferStorage::F32 | BufferStorage::F16 | BufferStorage::Q8Kv => Ok(storage),
        storage => Err(storage_error("read KV spans", storage)),
    }
}

fn validate_span_read_position(
    cache: KvReadView<'_, CudaBuffer>,
    shape: AttentionShape,
    position: Position<'_, CudaBuffer>,
) -> Result<(), BackendError> {
    match position {
        Position::Host(position) => {
            let end = position.checked_add(1).ok_or(BackendError::SizeOverflow {
                field: "KV span attention context length",
            })?;
            if end > shape.max_context() || end > cache.mapped_tokens() {
                return Err(BackendError::PositionOutOfBounds {
                    position: end,
                    max_context: shape.max_context().min(cache.mapped_tokens()),
                });
            }
        }
        Position::Device(position) => {
            position.u32()?;
            let mapped_tail = cache
                .spans()
                .last()
                .is_some_and(|span| span.token_count() == span.capacity_token_count());
            if !mapped_tail {
                return Err(BackendError::operation(
                    "read KV spans",
                    "device-position attention requires a fully mapped tail",
                ));
            }
        }
    }
    Ok(())
}

fn prepare_attention_decode_span(
    backend: &mut CudaBackend,
    cache: KvReadView<'_, CudaBuffer>,
    shape: AttentionShape,
    position: Position<'_, CudaBuffer>,
) -> Result<(crate::AttentionShape, BufferStorage, usize, usize), BackendError> {
    let cuda_shape = attention_shape(shape)?;
    validate_span_read_position(cache, shape, position)?;
    let cache_kind = span_cache_kind(cache, shape)?;
    let table_input = read_span_descriptors(cache, shape)?;
    let table_index = backend.retain_span_table_input(&table_input)?;
    Ok((
        cuda_shape,
        cache_kind,
        table_index,
        table_input.mapped_tokens,
    ))
}

fn validate_span_attention_range(
    shape: AttentionShape,
    mapped_tokens: usize,
    start_position: usize,
    positions: usize,
    field: &'static str,
) -> Result<(), BackendError> {
    if positions == 0 {
        return Err(BackendError::Zero { field });
    }
    let end_position = start_position
        .checked_add(positions)
        .ok_or(BackendError::SizeOverflow {
            field: "span attention end position",
        })?;
    if end_position > shape.max_context() || end_position > mapped_tokens {
        return Err(BackendError::PositionOutOfBounds {
            position: end_position,
            max_context: shape.max_context().min(mapped_tokens),
        });
    }
    Ok(())
}

fn span_table_from_parts<'a>(
    graph_capture_active: bool,
    graph_tables: &'a [KvSpanTable],
    eager_tables: &'a [KvSpanTable],
    table_index: usize,
) -> Result<&'a DeviceBuffer<KvSpanDescriptor>, BackendError> {
    let tables = if graph_capture_active {
        graph_tables
    } else {
        eager_tables
    };
    tables
        .get(table_index)
        .map(KvSpanTable::device)
        .ok_or_else(|| BackendError::operation("find retained KV span table", "missing entry"))
}

fn retained_span_table_with_error<'a>(
    graph_capture_active: bool,
    graph_span_tables: &'a [KvSpanTable],
    eager_span_tables: &'a [KvSpanTable],
    span_error: &'a mut DeviceBuffer<u32>,
    table_index: usize,
) -> Result<
    (
        &'a DeviceBuffer<KvSpanDescriptor>,
        &'a mut DeviceBuffer<u32>,
    ),
    BackendError,
> {
    let tables = if graph_capture_active {
        graph_span_tables
    } else {
        eager_span_tables
    };
    let table = tables
        .get(table_index)
        .map(KvSpanTable::device)
        .ok_or_else(|| BackendError::operation("find retained KV span table", "missing entry"))?;
    Ok((table, span_error))
}

#[allow(clippy::too_many_arguments)]
fn run_backend_verify_attention_span(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: AttentionShape,
    positions: usize,
    prepares_output: bool,
    cuda_shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    table_index: usize,
    start_position: usize,
) -> Result<(), BackendError> {
    let CudaBackend {
        stream,
        verify_gemv_scratch,
        verify_attention_scratch,
        eager_span_tables,
        graph_span_tables,
        graph_capture_active,
        ..
    } = backend;
    let scratch = verify_attention_scratch
        .get_mut(&(shape, positions))
        .ok_or_else(|| {
            BackendError::operation("find span verifier attention scratch", "missing entry")
        })?;
    let prepared_output = if prepares_output {
        let output_key = shape.query_elements()?;
        verify_gemv_scratch
            .get_mut(&(output_key, positions))
            .map(Some)
            .ok_or_else(|| {
                BackendError::operation(
                    "find span verifier attention q8_1 scratch",
                    "missing entry",
                )
            })?
    } else {
        None
    };
    let table = span_table_from_parts(
        *graph_capture_active,
        graph_span_tables,
        eager_span_tables,
        table_index,
    )?;
    run_verify_attention_spans(
        stream,
        query,
        table,
        output,
        scratch,
        prepared_output,
        cuda_shape,
        cache_kind,
        start_position,
        positions,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_backend_attention_decode_span(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: AttentionShape,
    output_key: usize,
    prepared: bool,
    cuda_shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    table_index: usize,
    position: Position<'_, CudaBuffer>,
    mapped_tokens: usize,
) -> Result<(), BackendError> {
    let CudaBackend {
        stream,
        gemv_scratch,
        attention_scratch,
        eager_span_tables,
        graph_span_tables,
        graph_capture_active,
        span_error,
        ..
    } = backend;
    let scratch = attention_scratch
        .get_mut(&shape)
        .ok_or_else(|| BackendError::operation("find span attention scratch", "missing entry"))?;
    let prepared_output = if prepared {
        gemv_scratch.get_mut(&output_key)
    } else {
        None
    };
    let table = span_table_from_parts(
        *graph_capture_active,
        graph_span_tables,
        eager_span_tables,
        table_index,
    )?;
    run_attention_decode_spans(
        stream,
        query,
        table,
        output,
        scratch,
        prepared_output,
        cuda_shape,
        cache_kind,
        position,
        mapped_tokens,
        span_error,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_attention_decode_spans(
    stream: &Stream,
    query: &CudaBuffer,
    table: &DeviceBuffer<KvSpanDescriptor>,
    output: &mut CudaBuffer,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    cuda_shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: Position<'_, CudaBuffer>,
    mapped_tokens: usize,
    span_error: &mut DeviceBuffer<u32>,
) -> Result<(), BackendError> {
    match position {
        Position::Host(position) => run_attention_decode_span_host(
            stream,
            query.f32()?,
            table,
            output.f32_mut()?,
            scratch,
            prepared_output,
            cuda_shape,
            cache_kind,
            position,
        ),
        Position::Device(position) => run_attention_decode_span_device(
            stream,
            query.f32()?,
            table,
            output.f32_mut()?,
            scratch,
            prepared_output,
            cuda_shape,
            cache_kind,
            position.u32()?,
            mapped_tokens,
            span_error,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_attention_decode_span_host(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    table: &DeviceBuffer<KvSpanDescriptor>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: usize,
) -> Result<(), BackendError> {
    let context_length = attention_context_length(position)?;
    let result = match cache_kind {
        BufferStorage::F32 => attention_decode_spans(
            stream,
            query,
            table,
            output,
            scratch,
            prepared_output,
            shape,
            context_length,
        ),
        BufferStorage::F16 => attention_decode_spans_f16(
            stream,
            query,
            table,
            output,
            scratch,
            prepared_output,
            shape,
            context_length,
        ),
        BufferStorage::Q8Kv => attention_decode_spans_q8(
            stream,
            query,
            table,
            output,
            scratch,
            prepared_output,
            shape,
            context_length,
        ),
        _ => unreachable!("span attention storage is checked before launch"),
    };
    result.map_err(|error| cuda_error("launch span attention", error))
}

#[allow(clippy::too_many_arguments)]
fn run_attention_decode_span_device(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    table: &DeviceBuffer<KvSpanDescriptor>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: crate::AttentionShape,
    cache_kind: BufferStorage,
    position: &DeviceBuffer<u32>,
    mapped_tokens: usize,
    error: &mut DeviceBuffer<u32>,
) -> Result<(), BackendError> {
    let result = match cache_kind {
        BufferStorage::F32 => attention_decode_spans_device_position(
            stream,
            query,
            table,
            output,
            scratch,
            prepared_output,
            shape,
            position,
            mapped_tokens,
            error,
        ),
        BufferStorage::F16 => attention_decode_spans_f16_device_position(
            stream,
            query,
            table,
            output,
            scratch,
            prepared_output,
            shape,
            position,
            mapped_tokens,
            error,
        ),
        BufferStorage::Q8Kv => attention_decode_spans_q8_device_position(
            stream,
            query,
            table,
            output,
            scratch,
            prepared_output,
            shape,
            position,
            mapped_tokens,
            error,
        ),
        _ => unreachable!("span attention storage is checked before launch"),
    };
    result.map_err(|error| cuda_error("launch span attention", error))
}

#[allow(clippy::too_many_arguments)]
fn run_attention_decode(
    backend: &mut CudaBackend,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: AttentionShape,
    output_key: usize,
    prepared: bool,
    cuda_shape: crate::AttentionShape,
    position: Position<'_, CudaBuffer>,
) -> Result<(), BackendError> {
    let scratch = backend
        .attention_scratch
        .get_mut(&shape)
        .ok_or_else(|| BackendError::operation("find attention scratch", "missing entry"))?;
    let prepared_output = match backend.gemv_scratch.get_mut(&output_key) {
        Some(scratch) if prepared => Some(scratch),
        _ => None,
    };
    launch_attention_decode(
        AttentionDecodeArgs {
            stream: &backend.stream,
            query,
            key_cache,
            value_cache,
            output,
            scratch,
            prepared_output,
            shape: cuda_shape,
        },
        position,
    )?;
    if prepared {
        backend.mark_prepared_input(output_key, output)?;
    }
    Ok(())
}

fn launch_embed_gather(
    stream: &Stream,
    table: &CudaBuffer,
    row: &CudaBuffer,
    output: &mut CudaBuffer,
    shape: QuantMatrix,
) -> Result<(), BackendError> {
    check_layout("embedding table", shape.layout()?, table.layout)?;
    let cuda_shape = quant_shape(shape)?;
    let inputs = embedding_gather_inputs(table, row, output)?;
    let result = match shape.format() {
        QuantFormat::Q4K => embedding_gather_q4_k_device_row(
            stream,
            inputs.table,
            inputs.row,
            inputs.output,
            cuda_shape,
        ),
        QuantFormat::Q6K => embedding_gather_q6_k_device_row(
            stream,
            inputs.table,
            inputs.row,
            inputs.output,
            cuda_shape,
        ),
    };
    map_cuda_result(result, "launch embedding gather")
}

fn ensure_prefill_gemm_scratch(
    backend: &mut CudaBackend,
    columns: usize,
    shape: crate::QuantizedMatrixShape,
) -> Result<(), BackendError> {
    if !backend.gemv_scratch.contains_key(&columns) {
        let scratch = GemvScratch::new(&backend.context, shape)
            .map_err(|error| cuda_error("allocate future decode GEMV scratch", error))?;
        backend.gemv_scratch.insert(columns, scratch);
    }
    Ok(())
}

fn allocate_storage(
    context: &Context,
    layout: BufferLayout,
    class: MemoryClass,
) -> Result<CudaStorage, BackendError> {
    match layout.storage() {
        BufferStorage::F16 | BufferStorage::F32 | BufferStorage::U32 => {
            allocate_dense_storage(context, layout, class)
        }
        BufferStorage::Q8Kv | BufferStorage::Q4K | BufferStorage::Q6K => {
            allocate_quantized_storage(context, layout, class)
        }
    }
}

fn allocate_dense_storage(
    context: &Context,
    layout: BufferLayout,
    class: MemoryClass,
) -> Result<CudaStorage, BackendError> {
    match layout.storage() {
        BufferStorage::F16 => map_cuda_result(
            context.alloc_class(layout.elements(), class),
            "allocate f16 buffer",
        )
        .map(CudaStorage::F16),
        BufferStorage::F32 => map_cuda_result(
            context.alloc_class(layout.elements(), class),
            "allocate f32 buffer",
        )
        .map(CudaStorage::F32),
        BufferStorage::U32 => map_cuda_result(
            context.alloc_class(layout.elements(), class),
            "allocate u32 buffer",
        )
        .map(CudaStorage::U32),
        _ => unreachable!("allocate_dense_storage receives dense storage"),
    }
}

fn allocate_quantized_storage(
    context: &Context,
    layout: BufferLayout,
    class: MemoryClass,
) -> Result<CudaStorage, BackendError> {
    map_cuda_result(
        context.alloc_class(layout.bytes(), class),
        "allocate quantized buffer",
    )
    .map(CudaStorage::Bytes)
}

fn clone_storage(
    stream: &Stream,
    source: &CudaStorage,
    destination: &mut CudaStorage,
) -> Result<(), BackendError> {
    match (source, destination) {
        (CudaStorage::Bytes(source), CudaStorage::Bytes(destination)) => destination
            .copy_from_device_async(stream, source)
            .map_err(|error| cuda_error("clone byte buffer", error)),
        (CudaStorage::F16(source), CudaStorage::F16(destination)) => destination
            .copy_from_device_async(stream, source)
            .map_err(|error| cuda_error("clone f16 buffer", error)),
        (CudaStorage::F32(source), CudaStorage::F32(destination)) => destination
            .copy_from_device_async(stream, source)
            .map_err(|error| cuda_error("clone f32 buffer", error)),
        (CudaStorage::U32(source), CudaStorage::U32(destination)) => destination
            .copy_from_device_async(stream, source)
            .map_err(|error| cuda_error("clone u32 buffer", error)),
        _ => Err(BackendError::operation(
            "clone buffer",
            "allocated storage does not match its source",
        )),
    }
}

fn download_storage(storage: &CudaStorage, layout: BufferLayout) -> Result<Vec<u8>, BackendError> {
    if cfg!(target_endian = "big") {
        return Err(BackendError::operation(
            "download buffer",
            "raw CUDA snapshots require a little-endian host",
        ));
    }
    match storage {
        CudaStorage::Bytes(buffer) => download_raw(buffer, layout, "download byte snapshot"),
        CudaStorage::F16(buffer) => download_raw(buffer, layout, "download f16 snapshot"),
        CudaStorage::F32(buffer) => download_raw(buffer, layout, "download f32 snapshot"),
        CudaStorage::U32(buffer) => download_raw(buffer, layout, "download u32 snapshot"),
    }
}

fn download_raw<T: crate::cuda::DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    layout: BufferLayout,
    operation: &'static str,
) -> Result<Vec<u8>, BackendError> {
    let bytes = layout.bytes();
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes)
        .map_err(|error| BackendError::operation(operation, error))?;
    values.resize(bytes, 0);
    map_cuda_result(buffer.copy_bytes_to(&mut values), operation)?;
    Ok(values)
}

fn restore_storage(
    stream: &Stream,
    storage: &mut CudaStorage,
    bytes: &[u8],
) -> Result<(), BackendError> {
    match storage {
        CudaStorage::Bytes(buffer) => buffer
            .copy_bytes_from_async(stream, bytes)
            .map_err(|error| cuda_error("restore byte snapshot", error)),
        CudaStorage::F16(buffer) => buffer
            .copy_bytes_from_async(stream, bytes)
            .map_err(|error| cuda_error("restore f16 snapshot", error)),
        CudaStorage::F32(buffer) => buffer
            .copy_bytes_from_async(stream, bytes)
            .map_err(|error| cuda_error("restore f32 snapshot", error)),
        CudaStorage::U32(buffer) => buffer
            .copy_bytes_from_async(stream, bytes)
            .map_err(|error| cuda_error("restore u32 snapshot", error)),
    }
}

fn upload_dense_storage(
    backend: &mut CudaBackend,
    storage: BufferStorage,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaStorage, BackendError> {
    match storage {
        BufferStorage::F16 => upload_f16_storage(backend, bytes, staging),
        BufferStorage::F32 => upload_f32_storage(backend, bytes, staging),
        BufferStorage::U32 => upload_u32_storage(backend, bytes, staging),
        _ => unreachable!("upload_dense_storage receives dense storage"),
    }
}

fn upload_f16_storage(
    backend: &CudaBackend,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaStorage, BackendError> {
    let reservation = reserve_host_bytes(staging, bytes.len())?;
    let values = parse_f16(bytes)?;
    let staged = StagedHost::commit(reservation, values)?;
    map_cuda_result(
        backend
            .context
            .copy_to_device_class(&staged.value, MemoryClass::ModelWeight),
        "upload f16 buffer",
    )
    .map(CudaStorage::F16)
}

fn upload_f32_storage(
    backend: &CudaBackend,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaStorage, BackendError> {
    let reservation = reserve_host_bytes(staging, bytes.len())?;
    let values = parse_f32(bytes)?;
    let staged = StagedHost::commit(reservation, values)?;
    map_cuda_result(
        backend
            .context
            .copy_to_device_class(&staged.value, MemoryClass::ModelWeight),
        "upload f32 buffer",
    )
    .map(CudaStorage::F32)
}

fn upload_u32_storage(
    backend: &CudaBackend,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaStorage, BackendError> {
    let reservation = reserve_host_bytes(staging, bytes.len())?;
    let values = parse_u32(bytes)?;
    let staged = StagedHost::commit(reservation, values)?;
    map_cuda_result(
        backend
            .context
            .copy_to_device_class(&staged.value, MemoryClass::ModelWeight),
        "upload u32 buffer",
    )
    .map(CudaStorage::U32)
}

fn upload_quantized_storage(
    backend: &mut CudaBackend,
    storage: BufferStorage,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaStorage, BackendError> {
    match storage {
        BufferStorage::Q4K => upload_q4_k_storage(backend, bytes, staging),
        BufferStorage::Q8Kv | BufferStorage::Q6K => Ok(CudaStorage::Bytes(
            backend
                .context
                .copy_to_device_class(bytes, MemoryClass::ModelWeight)
                .map_err(|error| cuda_error("upload quantized buffer", error))?,
        )),
        _ => unreachable!("upload_quantized_storage receives quantized storage"),
    }
}

fn upload_q4_k_storage(
    backend: &mut CudaBackend,
    bytes: &[u8],
    staging: Option<&HostStaging>,
) -> Result<CudaStorage, BackendError> {
    let started = Instant::now();
    let reservation = reserve_host_bytes(staging, bytes.len())?;
    let repacked = repack_q4_k(bytes)
        .map_err(|error| BackendError::operation("repack Q4_K weights", error.to_string()))?;
    let staged = StagedHost::commit(reservation, repacked)?;
    backend.q4_repack_duration += started.elapsed();
    backend.q4_repack_source_bytes = backend
        .q4_repack_source_bytes
        .checked_add(
            u64::try_from(bytes.len()).map_err(|_| BackendError::SizeOverflow {
                field: "Q4_K repack source bytes",
            })?,
        )
        .ok_or(BackendError::SizeOverflow {
            field: "Q4_K repack source bytes",
        })?;
    Ok(CudaStorage::Bytes(
        backend
            .context
            .copy_to_device_class(&staged.value, MemoryClass::RepackedWeight)
            .map_err(|error| cuda_error("upload repacked Q4_K buffer", error))?,
    ))
}

struct StagedHost<T> {
    value: T,
    _allocation: Option<MemoryAllocation>,
}

impl<T> StagedHost<T> {
    fn commit(reservation: Option<MemoryReservation>, value: T) -> Result<Self, BackendError> {
        let allocation = match reservation {
            Some(reservation) => match reservation.commit() {
                Ok(allocation) => Some(allocation),
                Err(error) => {
                    drop(value);
                    return Err(error.into());
                }
            },
            None => None,
        };
        Ok(Self {
            value,
            _allocation: allocation,
        })
    }
}

fn reserve_host_bytes(
    staging: Option<&HostStaging>,
    bytes: usize,
) -> Result<Option<MemoryReservation>, BackendError> {
    let Some(staging) = staging else {
        return Ok(None);
    };
    let bytes = u64::try_from(bytes).map_err(|_| BackendError::SizeOverflow {
        field: "host staging bytes",
    })?;
    staging.reserve(bytes).map(Some).map_err(BackendError::from)
}

fn quant_shape(shape: QuantMatrix) -> Result<crate::QuantizedMatrixShape, BackendError> {
    let format = match shape.format() {
        QuantFormat::Q4K => crate::QuantFormat::Q4K,
        QuantFormat::Q6K => crate::QuantFormat::Q6K,
    };
    crate::QuantizedMatrixShape::new(shape.rows(), shape.columns(), format)
        .map_err(|error| cuda_error("check CUDA matrix shape", error))
}

fn vector_shape(shape: VectorShape) -> Result<crate::VectorShape, BackendError> {
    crate::VectorShape::new(shape.rows(), shape.columns())
        .map_err(|error| cuda_error("check CUDA vector shape", error))
}

fn rope_shape(shape: RopeShape) -> Result<crate::RopeShape, BackendError> {
    crate::RopeShape::new(shape.tokens(), shape.heads(), shape.head_dim())
        .map_err(|error| cuda_error("check CUDA RoPE shape", error))
}

/// Returns `position + 1`, the decode attention context length.
fn attention_context_length(position: usize) -> Result<usize, BackendError> {
    position.checked_add(1).ok_or(BackendError::SizeOverflow {
        field: "attention context length",
    })
}

fn attention_shape(shape: AttentionShape) -> Result<crate::AttentionShape, BackendError> {
    crate::AttentionShape::new(
        shape.n_head(),
        shape.n_head_kv(),
        shape.head_dim(),
        shape.max_context(),
    )
    .map_err(|error| cuda_error("check CUDA attention shape", error))
}

fn parse_f32(bytes: &[u8]) -> Result<Vec<f32>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 4)
        .map_err(|error| BackendError::operation("parse f32 upload", error))?;
    values.extend(
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]])),
    );
    Ok(values)
}

fn parse_f16(bytes: &[u8]) -> Result<Vec<u16>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 2)
        .map_err(|error| BackendError::operation("parse f16 upload", error))?;
    values.extend(
        bytes
            .chunks_exact(2)
            .map(|value| u16::from_le_bytes([value[0], value[1]])),
    );
    Ok(values)
}

fn parse_u32(bytes: &[u8]) -> Result<Vec<u32>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 4)
        .map_err(|error| BackendError::operation("parse u32 upload", error))?;
    values.extend(
        bytes
            .chunks_exact(4)
            .map(|value| u32::from_le_bytes([value[0], value[1], value[2], value[3]])),
    );
    Ok(values)
}

fn check_layout(
    name: &'static str,
    expected: BufferLayout,
    actual: BufferLayout,
) -> Result<(), BackendError> {
    if expected == actual {
        Ok(())
    } else {
        Err(BackendError::Operation {
            operation: "check buffer layout",
            message: format!("{name} has {actual:?}, expected {expected:?}"),
        })
    }
}

fn storage_error(operation: &'static str, storage: BufferStorage) -> BackendError {
    BackendError::Operation {
        operation,
        message: format!("buffer storage is {storage:?}"),
    }
}

fn cuda_error(operation: &'static str, error: crate::Error) -> BackendError {
    match error {
        crate::Error::Memory(error) => BackendError::Memory(error),
        error => BackendError::operation(operation, error),
    }
}

fn map_cuda_result<T>(
    result: crate::Result<T>,
    operation: &'static str,
) -> Result<T, BackendError> {
    result.map_err(|error| cuda_error(operation, error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_attention_paths_keep_eight_row_runtime_limit() {
        for path in [
            CudaAttentionBatchPath::PerRow,
            CudaAttentionBatchPath::FixedTilePerRow,
            CudaAttentionBatchPath::SharedReadUnconstrained,
            CudaAttentionBatchPath::SharedReadFixedReduction,
        ] {
            assert_eq!(path.max_batch_size(), RESEARCH_BATCH_SIZE);
        }
    }

    #[test]
    fn memory_budget_errors_remain_typed_at_backend_boundary() {
        let memory = leone::MemoryError::BudgetExceeded {
            requested: 8,
            budget: 16,
            owned: 16,
            reserved: 0,
        };
        let error = cuda_error("allocate buffer", crate::Error::Memory(memory.clone()));
        assert_eq!(error, BackendError::Memory(memory));
    }

    #[test]
    fn graph_owner_staging_deduplicates_duplicate_candidates() {
        let pending: [usize; 0] = [];
        let mut candidates = std::array::from_fn(|_| None);
        let mut count = 0;
        let same = |left: &usize, right: &usize| left == right;
        assert!(!CudaBackend::stage_graph_owner(
            &pending,
            &mut candidates,
            &mut count,
            7,
            same,
        ));
        assert!(!CudaBackend::stage_graph_owner(
            &pending,
            &mut candidates,
            &mut count,
            7,
            same,
        ));
        assert_eq!(count, 1);
        assert_eq!(candidates[0], Some(7));
    }

    #[test]
    fn graph_owner_staging_rejects_distinct_owner_at_limit() {
        let pending: [usize; MAX_DECODE_GRAPH_BUFFERS] = std::array::from_fn(|index| index);
        let mut candidates = std::array::from_fn(|_| None);
        let mut count = 0;
        let same = |left: &usize, right: &usize| left == right;
        assert!(!CudaBackend::stage_graph_owner(
            &pending,
            &mut candidates,
            &mut count,
            0,
            same,
        ));
        assert!(CudaBackend::stage_graph_owner(
            &pending,
            &mut candidates,
            &mut count,
            MAX_DECODE_GRAPH_BUFFERS,
            same,
        ));
        assert_eq!(count, 0);
        assert!(candidates.iter().all(Option::is_none));
    }

    #[test]
    fn graph_owner_staging_fills_only_bounded_slots() {
        let pending: [usize; MAX_DECODE_GRAPH_BUFFERS - 1] = std::array::from_fn(|index| index);
        let mut candidates = std::array::from_fn(|_| None);
        let mut count = 0;
        let same = |left: &usize, right: &usize| left == right;
        assert!(!CudaBackend::stage_graph_owner(
            &pending,
            &mut candidates,
            &mut count,
            MAX_DECODE_GRAPH_BUFFERS - 1,
            same,
        ));
        assert!(CudaBackend::stage_graph_owner(
            &pending,
            &mut candidates,
            &mut count,
            MAX_DECODE_GRAPH_BUFFERS,
            same,
        ));
        assert_eq!(count, 1);
        assert_eq!(candidates[0], Some(MAX_DECODE_GRAPH_BUFFERS - 1));
        assert!(candidates[1..].iter().all(Option::is_none));
    }

    #[test]
    fn graph_owner_reservation_uses_vec_length_not_capacity() {
        let mut owners = Vec::with_capacity(200);
        owners.resize(100, 0_usize);
        let additional = CudaBackend::graph_owner_reservation(owners.len(), 120).unwrap();
        assert_eq!(additional, 120);
        owners.try_reserve_exact(additional).unwrap();
        assert!(owners.capacity() >= 220);
    }
}
