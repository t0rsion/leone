use crate::cuda::{rope_at_frequencies, rope_at_frequencies_device_position};
use crate::{
    argmax, attention_decode, attention_decode_device_position, attention_decode_f16,
    attention_decode_f16_device_position, attention_decode_q8, attention_decode_q8_device_position,
    attention_prefill_f16, attention_prefill_f32, attention_prefill_q8, copy_f32_row,
    embedding_gather_batch, embedding_gather_q4_k_device_row, embedding_gather_q6_k_device_row,
    gemv_pair_q4_k, gemv_pair_swiglu_q4_k, gemv_q4_k, gemv_q4_k_residual, gemv_q6_k,
    gemv_q6_k_residual, increment_u32_scalar, kv_append, kv_append_chunk, kv_append_chunk_f16,
    kv_append_chunk_q8, kv_append_device_position, kv_append_f16, kv_append_f16_device_position,
    kv_append_q8, kv_append_q8_device_position, prefill_gemm, qk_norm_rope, qk_norm_rope_kv_append,
    qk_norm_rope_kv_append_device_position, qk_norm_rope_kv_append_f16,
    qk_norm_rope_kv_append_f16_device_position, qkv_gemv, repack_q4_k, residual_add, rms_norm,
    rms_norm_q8_parallel, rms_norm_residual, rms_norm_residual_store, rms_norm_rope,
    rms_norm_rope_device_position, swiglu, write_f32_row, write_u32_scalar, ArgmaxScratch,
    AttentionScratch, Context, CublasLt, DeviceBuffer, Event, GemvScratch, Graph, PrefillScratch,
    RopeScratch, Stream, PREPARED_ATTENTION_HEAD_DIM,
};
use leone::backend::{MemoryAccounting, MemoryClass, UntrackedMemory};
use leone::{
    AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot, BufferStorage, DecodeOp,
    DecodeProfile, Determinism, GemvProfile, MemoryCapacity, ModelImportMetrics, Position,
    PrefillMethod, PrefillPlan, PrefillWorkspace, QuantFormat, QuantMatrix, RopePairing, RopeShape,
    VectorShape,
};
use std::collections::{btree_map::Entry, BTreeMap};
use std::time::{Duration, Instant};

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
}

#[derive(Debug)]
enum CudaStorage {
    Bytes(DeviceBuffer<u8>),
    F16(DeviceBuffer<u16>),
    F32(DeviceBuffer<f32>),
    U32(DeviceBuffer<u32>),
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
        let context = Context::new(device).map_err(|error| cuda_error("initialize CUDA", error))?;
        let stream = Stream::new(&context).map_err(|error| cuda_error("create stream", error))?;
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
}

impl Backend for CudaBackend {
    type Buffer = CudaBuffer;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn determinism(&self) -> Determinism {
        Determinism::FixedOrder
    }

    /// Reports direct `cudaMalloc` bytes and counts CUDA library objects whose
    /// byte sizes are not exposed by the runtime API.
    fn memory_accounting(&self) -> MemoryAccounting {
        self.context
            .memory_accounting()
            .with_untracked(self.untracked)
    }

    fn classify_buffer(
        &mut self,
        buffer: &Self::Buffer,
        class: MemoryClass,
    ) -> Result<(), BackendError> {
        buffer.reclassify(class);
        Ok(())
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
        if bytes.len() != layout.bytes() {
            return Err(BackendError::SizeMismatch {
                name: "uploaded bytes",
                expected: layout.bytes(),
                actual: bytes.len(),
            });
        }
        let storage = upload_storage(self, layout.storage(), bytes)?;
        Ok(CudaBuffer { layout, storage })
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
        let bytes = download_storage(&source.storage, source.layout)?;
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
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.stream_synchronizations += 1;
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("finish u32 read", error))
    }

    fn read_f16(&mut self, buffer: &Self::Buffer, values: &mut [u16]) -> Result<(), BackendError> {
        buffer
            .f16()?
            .copy_to_async(&self.stream, values)
            .map_err(|error| cuda_error("enqueue f16 read", error))?;
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("finish f16 read", error))
    }

    fn read_f32(&mut self, buffer: &Self::Buffer, values: &mut [f32]) -> Result<(), BackendError> {
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.d2h_copies += 1;
        }
        buffer
            .f32()?
            .copy_to_async(&self.stream, values)
            .map_err(|error| cuda_error("enqueue f32 read", error))?;
        if let Some(profiler) = &mut self.decode_profiler {
            profiler.stream_synchronizations += 1;
        }
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("finish f32 read", error))
    }

    fn prepare_prefill(&mut self, plan: PrefillPlan) -> Result<PrefillWorkspace, BackendError> {
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
        self.prefill_plan = None;
        self.prefill_scratch = None;
        let scratch = PrefillScratch::new(&self.context, plan)
            .map_err(|error| cuda_error("allocate prefill workspace", error))?;
        let usage = scratch.usage();
        self.prefill_scratch = Some(scratch);
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
        if key_cache.layout.storage() == BufferStorage::Q8Kv {
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
        self.stream
            .synchronize()
            .map_err(|error| cuda_error("synchronize CUDA", error))
    }

    fn decode_graph_supported(&self) -> bool {
        true
    }

    fn begin_decode_graph(&mut self) -> Result<(), BackendError> {
        if self.decode_graph.take().is_some() {
            self.untracked.graph_objects = self
                .untracked
                .graph_objects
                .checked_sub(1)
                .expect("graph object count underflow");
        }
        self.stream
            .begin_graph_capture()
            .map_err(|error| cuda_error("begin decode graph", error))
    }

    fn end_decode_graph(&mut self) -> Result<(), BackendError> {
        let graph = self
            .stream
            .end_graph_capture()
            .map_err(|error| cuda_error("end decode graph", error))?;
        self.decode_graph = Some(graph);
        self.untracked.graph_objects = self
            .untracked
            .graph_objects
            .checked_add(1)
            .expect("graph object count overflow");
        Ok(())
    }

    fn replay_decode_graph(&mut self) -> Result<(), BackendError> {
        let graph = self.decode_graph.as_ref().ok_or_else(|| {
            BackendError::operation("replay decode graph", "no graph has been captured")
        })?;
        graph
            .launch(&self.stream)
            .map_err(|error| cuda_error("replay decode graph", error))
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

fn upload_storage(
    backend: &mut CudaBackend,
    storage: BufferStorage,
    bytes: &[u8],
) -> Result<CudaStorage, BackendError> {
    match storage {
        BufferStorage::F16 | BufferStorage::F32 | BufferStorage::U32 => {
            upload_dense_storage(backend, storage, bytes)
        }
        BufferStorage::Q4K | BufferStorage::Q8Kv | BufferStorage::Q6K => {
            upload_quantized_storage(backend, storage, bytes)
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
    launch_gemv_pair_swiglu_with_scratch(
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
    )?;
    backend.gemv_scratch.insert(output_key, output_scratch);
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
    match storage {
        CudaStorage::Bytes(buffer) => download_bytes(buffer, layout),
        CudaStorage::F16(buffer) => download_f16(buffer, layout),
        CudaStorage::F32(buffer) => download_f32(buffer, layout),
        CudaStorage::U32(buffer) => download_u32(buffer, layout),
    }
}

fn download_bytes(
    buffer: &DeviceBuffer<u8>,
    layout: BufferLayout,
) -> Result<Vec<u8>, BackendError> {
    let mut values = vec![0_u8; layout.bytes()];
    map_cuda_result(buffer.copy_to(&mut values), "download byte snapshot")?;
    Ok(values)
}

fn download_f16(buffer: &DeviceBuffer<u16>, layout: BufferLayout) -> Result<Vec<u8>, BackendError> {
    let mut values = vec![0_u16; layout.elements()];
    map_cuda_result(buffer.copy_to(&mut values), "download f16 snapshot")?;
    Ok(values.into_iter().flat_map(u16::to_le_bytes).collect())
}

fn download_f32(buffer: &DeviceBuffer<f32>, layout: BufferLayout) -> Result<Vec<u8>, BackendError> {
    let mut values = vec![0_f32; layout.elements()];
    map_cuda_result(buffer.copy_to(&mut values), "download f32 snapshot")?;
    Ok(values
        .into_iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect())
}

fn download_u32(buffer: &DeviceBuffer<u32>, layout: BufferLayout) -> Result<Vec<u8>, BackendError> {
    let mut values = vec![0_u32; layout.elements()];
    map_cuda_result(buffer.copy_to(&mut values), "download u32 snapshot")?;
    Ok(values.into_iter().flat_map(u32::to_le_bytes).collect())
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
) -> Result<CudaStorage, BackendError> {
    match storage {
        BufferStorage::F16 => upload_f16_storage(backend, bytes),
        BufferStorage::F32 => upload_f32_storage(backend, bytes),
        BufferStorage::U32 => upload_u32_storage(backend, bytes),
        _ => unreachable!("upload_dense_storage receives dense storage"),
    }
}

fn upload_f16_storage(backend: &CudaBackend, bytes: &[u8]) -> Result<CudaStorage, BackendError> {
    map_cuda_result(
        backend
            .context
            .copy_to_device_class(&parse_f16(bytes), MemoryClass::ModelWeight),
        "upload f16 buffer",
    )
    .map(CudaStorage::F16)
}

fn upload_f32_storage(backend: &CudaBackend, bytes: &[u8]) -> Result<CudaStorage, BackendError> {
    map_cuda_result(
        backend
            .context
            .copy_to_device_class(&parse_f32(bytes), MemoryClass::ModelWeight),
        "upload f32 buffer",
    )
    .map(CudaStorage::F32)
}

fn upload_u32_storage(backend: &CudaBackend, bytes: &[u8]) -> Result<CudaStorage, BackendError> {
    map_cuda_result(
        backend
            .context
            .copy_to_device_class(&parse_u32(bytes), MemoryClass::ModelWeight),
        "upload u32 buffer",
    )
    .map(CudaStorage::U32)
}

fn upload_quantized_storage(
    backend: &mut CudaBackend,
    storage: BufferStorage,
    bytes: &[u8],
) -> Result<CudaStorage, BackendError> {
    match storage {
        BufferStorage::Q4K => upload_q4_k_storage(backend, bytes),
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
) -> Result<CudaStorage, BackendError> {
    let started = Instant::now();
    let repacked = repack_q4_k(bytes)
        .map_err(|error| BackendError::operation("repack Q4_K weights", error.to_string()))?;
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
            .copy_to_device_class(&repacked, MemoryClass::RepackedWeight)
            .map_err(|error| cuda_error("upload repacked Q4_K buffer", error))?,
    ))
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

fn parse_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect()
}

fn parse_f16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|value| u16::from_le_bytes([value[0], value[1]]))
        .collect()
}

fn parse_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|value| u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect()
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

fn cuda_error(operation: &'static str, error: impl std::fmt::Display) -> BackendError {
    BackendError::operation(operation, error)
}

fn map_cuda_result<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    operation: &'static str,
) -> Result<T, BackendError> {
    result.map_err(|error| cuda_error(operation, error))
}
