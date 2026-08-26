use crate::cuda::rope_at_frequencies;
use crate::{
    argmax, attention_decode, attention_decode_device_position, attention_decode_f16,
    attention_decode_f16_device_position, attention_decode_q8, attention_decode_q8_device_position,
    attention_prefill_f16, attention_prefill_f32, copy_f32_row, embedding_gather_batch,
    embedding_gather_q4_k_device_row, embedding_gather_q6_k_device_row, gemv_pair_q4_k,
    gemv_pair_swiglu_q4_k, gemv_q4_k, gemv_q4_k_residual, gemv_q6_k, gemv_q6_k_residual,
    increment_u32_scalar, kv_append, kv_append_chunk, kv_append_chunk_f16,
    kv_append_device_position, kv_append_f16, kv_append_f16_device_position, kv_append_q8,
    kv_append_q8_device_position, prefill_gemm, qk_norm_rope, qk_norm_rope_kv_append,
    qk_norm_rope_kv_append_device_position, qk_norm_rope_kv_append_f16,
    qk_norm_rope_kv_append_f16_device_position, qkv_gemv, repack_q4_k, residual_add, rms_norm,
    rms_norm_q8_parallel, rms_norm_residual, rms_norm_residual_store, rms_norm_rope,
    rms_norm_rope_device_position, swiglu, write_u32_scalar, ArgmaxScratch, AttentionScratch,
    Context, CublasLt, DeviceBuffer, Event, GemvScratch, Graph, PrefillScratch, RopeScratch,
    Stream, PREPARED_ATTENTION_HEAD_DIM,
};
use leone::{
    AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot, BufferStorage, DecodeOp,
    DecodeProfile, Determinism, GemvProfile, MemoryCapacity, ModelImportMetrics, Position,
    PrefillMethod, PrefillPlan, PrefillWorkspace, QuantFormat, QuantMatrix, RopePairing, RopeShape,
    VectorShape,
};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// An opaque buffer owned by the CUDA backend.
#[derive(Debug)]
pub struct CudaBuffer {
    layout: BufferLayout,
    storage: CudaStorage,
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
    prepared_gemv_inputs: BTreeMap<usize, usize>,
    attention_scratch: BTreeMap<AttentionShape, AttentionScratch>,
    verify_attention_scratch: BTreeMap<(AttentionShape, usize), AttentionScratch>,
    argmax_scratch: BTreeMap<usize, ArgmaxScratch>,
    rope_scratch: Option<RopeScratch>,
    cublaslt: Option<CublasLt>,
    prefill_scratch: Option<PrefillScratch>,
    prefill_plan: Option<PrefillPlan>,
    decode_profiler: Option<CudaDecodeProfiler>,
    decode_graph: Option<Graph>,
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

    fn prefill_method(&self) -> PrefillMethod {
        PrefillMethod::TiledCublasLtFp16
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
        let storage = match layout.storage() {
            BufferStorage::F16 => CudaStorage::F16(
                self.context
                    .alloc(layout.elements())
                    .map_err(|error| cuda_error("allocate f16 buffer", error))?,
            ),
            BufferStorage::F32 => CudaStorage::F32(
                self.context
                    .alloc(layout.elements())
                    .map_err(|error| cuda_error("allocate f32 buffer", error))?,
            ),
            BufferStorage::U32 => CudaStorage::U32(
                self.context
                    .alloc(layout.elements())
                    .map_err(|error| cuda_error("allocate u32 buffer", error))?,
            ),
            BufferStorage::Q8Kv | BufferStorage::Q4K | BufferStorage::Q6K => CudaStorage::Bytes(
                self.context
                    .alloc(layout.bytes())
                    .map_err(|error| cuda_error("allocate quantized buffer", error))?,
            ),
        };
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
        let storage = match layout.storage() {
            BufferStorage::F16 => CudaStorage::F16(
                self.context
                    .copy_to_device(&parse_f16(bytes))
                    .map_err(|error| cuda_error("upload f16 buffer", error))?,
            ),
            BufferStorage::F32 => CudaStorage::F32(
                self.context
                    .copy_to_device(&parse_f32(bytes))
                    .map_err(|error| cuda_error("upload f32 buffer", error))?,
            ),
            BufferStorage::U32 => CudaStorage::U32(
                self.context
                    .copy_to_device(&parse_u32(bytes))
                    .map_err(|error| cuda_error("upload u32 buffer", error))?,
            ),
            BufferStorage::Q4K => {
                let started = Instant::now();
                let repacked = repack_q4_k(bytes).map_err(|error| {
                    BackendError::operation("repack Q4_K weights", error.to_string())
                })?;
                self.q4_repack_duration += started.elapsed();
                self.q4_repack_source_bytes = self
                    .q4_repack_source_bytes
                    .checked_add(u64::try_from(bytes.len()).map_err(|_| {
                        BackendError::SizeOverflow {
                            field: "Q4_K repack source bytes",
                        }
                    })?)
                    .ok_or(BackendError::SizeOverflow {
                        field: "Q4_K repack source bytes",
                    })?;
                CudaStorage::Bytes(
                    self.context
                        .copy_to_device(&repacked)
                        .map_err(|error| cuda_error("upload repacked Q4_K buffer", error))?,
                )
            }
            BufferStorage::Q8Kv | BufferStorage::Q6K => CudaStorage::Bytes(
                self.context
                    .copy_to_device(bytes)
                    .map_err(|error| cuda_error("upload quantized buffer", error))?,
            ),
        };
        Ok(CudaBuffer { layout, storage })
    }

    fn clone_buffer(&mut self, source: &Self::Buffer) -> Result<Self::Buffer, BackendError> {
        let mut destination = self.allocate(source.layout)?;
        match (&source.storage, &mut destination.storage) {
            (CudaStorage::Bytes(source), CudaStorage::Bytes(destination)) => destination
                .copy_from_device_async(&self.stream, source)
                .map_err(|error| cuda_error("clone byte buffer", error))?,
            (CudaStorage::F16(source), CudaStorage::F16(destination)) => destination
                .copy_from_device_async(&self.stream, source)
                .map_err(|error| cuda_error("clone f16 buffer", error))?,
            (CudaStorage::F32(source), CudaStorage::F32(destination)) => destination
                .copy_from_device_async(&self.stream, source)
                .map_err(|error| cuda_error("clone f32 buffer", error))?,
            (CudaStorage::U32(source), CudaStorage::U32(destination)) => destination
                .copy_from_device_async(&self.stream, source)
                .map_err(|error| cuda_error("clone u32 buffer", error))?,
            _ => {
                return Err(BackendError::operation(
                    "clone buffer",
                    "allocated storage does not match its source",
                ))
            }
        }
        Ok(destination)
    }

    fn download_buffer(&mut self, source: &Self::Buffer) -> Result<BufferSnapshot, BackendError> {
        let bytes = match &source.storage {
            CudaStorage::Bytes(buffer) => {
                let mut values = vec![0_u8; source.layout.bytes()];
                buffer
                    .copy_to(&mut values)
                    .map_err(|error| cuda_error("download byte snapshot", error))?;
                values
            }
            CudaStorage::F16(buffer) => {
                let mut values = vec![0_u16; source.layout.elements()];
                buffer
                    .copy_to(&mut values)
                    .map_err(|error| cuda_error("download f16 snapshot", error))?;
                values.into_iter().flat_map(u16::to_le_bytes).collect()
            }
            CudaStorage::F32(buffer) => {
                let mut values = vec![0_f32; source.layout.elements()];
                buffer
                    .copy_to(&mut values)
                    .map_err(|error| cuda_error("download f32 snapshot", error))?;
                values
                    .into_iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect()
            }
            CudaStorage::U32(buffer) => {
                let mut values = vec![0_u32; source.layout.elements()];
                buffer
                    .copy_to(&mut values)
                    .map_err(|error| cuda_error("download u32 snapshot", error))?;
                values.into_iter().flat_map(u32::to_le_bytes).collect()
            }
        };
        BufferSnapshot::new(source.layout, bytes)
    }

    fn restore_buffer(&mut self, source: &BufferSnapshot) -> Result<Self::Buffer, BackendError> {
        let layout = source.layout();
        let mut destination = self.allocate(layout)?;
        match &mut destination.storage {
            CudaStorage::Bytes(buffer) => buffer
                .copy_bytes_from_async(&self.stream, source.bytes())
                .map_err(|error| cuda_error("restore byte snapshot", error))?,
            CudaStorage::F16(buffer) => buffer
                .copy_bytes_from_async(&self.stream, source.bytes())
                .map_err(|error| cuda_error("restore f16 snapshot", error))?,
            CudaStorage::F32(buffer) => buffer
                .copy_bytes_from_async(&self.stream, source.bytes())
                .map_err(|error| cuda_error("restore f32 snapshot", error))?,
            CudaStorage::U32(buffer) => buffer
                .copy_bytes_from_async(&self.stream, source.bytes())
                .map_err(|error| cuda_error("restore u32 snapshot", error))?,
        }
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
        }
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
        let scratch_key = shape.columns();
        if !self.gemv_scratch.contains_key(&scratch_key) {
            let scratch = GemvScratch::new(&self.context, cuda_shape)
                .map_err(|error| cuda_error("allocate future decode GEMV scratch", error))?;
            self.gemv_scratch.insert(scratch_key, scratch);
        }
        let handle = self.cublaslt.as_ref().ok_or_else(|| {
            BackendError::operation("run prefill GEMM", "cuBLASLt is not prepared")
        })?;
        let scratch = self.prefill_scratch.as_mut().ok_or_else(|| {
            BackendError::operation("run prefill GEMM", "prefill workspace is not prepared")
        })?;
        prefill_gemm(
            handle,
            &self.stream,
            weights.bytes()?,
            input.f32()?,
            output.f32_mut()?,
            cuda_shape,
            tokens,
            scratch,
        )
        .map_err(|error| cuda_error("run prefill GEMM", error))
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
        let key = (shape.columns(), positions);
        if !self.verify_gemv_scratch.contains_key(&key) {
            let scratch = GemvScratch::new_multi(&self.context, shape.columns(), positions)
                .map_err(|error| cuda_error("allocate verifier GEMV scratch", error))?;
            self.verify_gemv_scratch.insert(key, scratch);
        }
        let scratch = self.verify_gemv_scratch.get_mut(&key).ok_or_else(|| {
            BackendError::operation("find verifier GEMV scratch", "missing entry")
        })?;
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
        for shape in [first_shape, second_shape, third_shape] {
            self.profile_gemv(shape)?;
        }
        check_layout(
            "first verifier GEMV weights",
            first_shape.layout()?,
            first_weights.layout,
        )?;
        check_layout(
            "second verifier GEMV weights",
            second_shape.layout()?,
            second_weights.layout,
        )?;
        check_layout(
            "third verifier GEMV weights",
            third_shape.layout()?,
            third_weights.layout,
        )?;
        let key = (first_shape.columns(), positions);
        if !self.verify_gemv_scratch.contains_key(&key) {
            let scratch = GemvScratch::new_multi(&self.context, first_shape.columns(), positions)
                .map_err(|error| cuda_error("allocate verifier GEMV scratch", error))?;
            self.verify_gemv_scratch.insert(key, scratch);
        }
        let scratch = self.verify_gemv_scratch.get_mut(&key).ok_or_else(|| {
            BackendError::operation("find verifier GEMV scratch", "missing entry")
        })?;
        crate::cuda::verify_gemv_triple(
            &self.stream,
            first_weights.bytes()?,
            second_weights.bytes()?,
            third_weights.bytes()?,
            input.f32()?,
            first_output.f32_mut()?,
            second_output.f32_mut()?,
            third_output.f32_mut()?,
            scratch,
            quant_shape(first_shape)?,
            quant_shape(second_shape)?,
            quant_shape(third_shape)?,
            positions,
        )
        .map_err(|error| cuda_error("launch verifier GEMV triple", error))
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
        self.profile_gemv(first_shape)?;
        self.profile_gemv(second_shape)?;
        check_layout(
            "first verifier GEMV weights",
            first_shape.layout()?,
            first_weights.layout,
        )?;
        check_layout(
            "second verifier GEMV weights",
            second_shape.layout()?,
            second_weights.layout,
        )?;
        let key = (first_shape.columns(), positions);
        if !self.verify_gemv_scratch.contains_key(&key) {
            let scratch = GemvScratch::new_multi(&self.context, first_shape.columns(), positions)
                .map_err(|error| cuda_error("allocate verifier GEMV scratch", error))?;
            self.verify_gemv_scratch.insert(key, scratch);
        }
        let scratch = self.verify_gemv_scratch.get_mut(&key).ok_or_else(|| {
            BackendError::operation("find verifier GEMV scratch", "missing entry")
        })?;
        crate::cuda::verify_gemv_pair(
            &self.stream,
            first_weights.bytes()?,
            second_weights.bytes()?,
            input.f32()?,
            first_output.f32_mut()?,
            second_output.f32_mut()?,
            scratch,
            quant_shape(first_shape)?,
            quant_shape(second_shape)?,
            positions,
        )
        .map_err(|error| cuda_error("launch verifier GEMV pair", error))
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
        let key = (shape.columns(), positions);
        if !self.verify_gemv_scratch.contains_key(&key) {
            let scratch = GemvScratch::new_multi(&self.context, shape.columns(), positions)
                .map_err(|error| cuda_error("allocate verifier GEMV scratch", error))?;
            self.verify_gemv_scratch.insert(key, scratch);
        }
        let scratch = self.verify_gemv_scratch.get_mut(&key).ok_or_else(|| {
            BackendError::operation("find verifier GEMV scratch", "missing entry")
        })?;
        crate::cuda::verify_gemv(
            &self.stream,
            weights.bytes()?,
            input.f32()?,
            Some(residual.f32()?),
            output.f32_mut()?,
            scratch,
            quant_shape(shape)?,
            positions,
        )
        .map_err(|error| cuda_error("launch verifier residual GEMV", error))
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
        let key = (shape.columns(), positions);
        let scratch = self.verify_gemv_scratch.get_mut(&key).ok_or_else(|| {
            BackendError::operation("find prepared verifier GEMV scratch", "missing entry")
        })?;
        crate::cuda::verify_gemv_prepared(
            &self.stream,
            weights.bytes()?,
            input.f32()?,
            Some(residual.f32()?),
            output.f32_mut()?,
            scratch,
            quant_shape(shape)?,
            positions,
        )
        .map_err(|error| cuda_error("launch prepared verifier residual GEMV", error))
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
        let scratch_key = shape.columns();
        if !self.gemv_scratch.contains_key(&scratch_key) {
            let scratch = GemvScratch::new(&self.context, cuda_shape)
                .map_err(|error| cuda_error("allocate GEMV scratch", error))?;
            self.gemv_scratch.insert(scratch_key, scratch);
        }
        let scratch = self
            .gemv_scratch
            .get_mut(&scratch_key)
            .ok_or_else(|| BackendError::operation("find GEMV scratch", "missing entry"))?;
        let result = match (shape.format(), input_prepared) {
            (QuantFormat::Q4K, true) => gemv_q4_k_residual(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                None,
                output.f32_mut()?,
                scratch,
                cuda_shape,
                true,
            ),
            (QuantFormat::Q4K, false) => gemv_q4_k(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                output.f32_mut()?,
                scratch,
                cuda_shape,
            ),
            (QuantFormat::Q6K, true) => gemv_q6_k_residual(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                None,
                output.f32_mut()?,
                scratch,
                cuda_shape,
                true,
            ),
            (QuantFormat::Q6K, false) => gemv_q6_k(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                output.f32_mut()?,
                scratch,
                cuda_shape,
            ),
        };
        result.map_err(|error| cuda_error("launch GEMV", error))
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
        let scratch_key = shape.columns();
        if !self.gemv_scratch.contains_key(&scratch_key) {
            let scratch = GemvScratch::new(&self.context, cuda_shape)
                .map_err(|error| cuda_error("allocate residual GEMV scratch", error))?;
            self.gemv_scratch.insert(scratch_key, scratch);
        }
        let scratch = self.gemv_scratch.get_mut(&scratch_key).ok_or_else(|| {
            BackendError::operation("find residual GEMV scratch", "missing entry")
        })?;
        match shape.format() {
            QuantFormat::Q4K => gemv_q4_k_residual(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                Some(residual.f32()?),
                output.f32_mut()?,
                scratch,
                cuda_shape,
                input_prepared,
            ),
            QuantFormat::Q6K => gemv_q6_k_residual(
                &self.stream,
                weights.bytes()?,
                input.f32()?,
                Some(residual.f32()?),
                output.f32_mut()?,
                scratch,
                cuda_shape,
                input_prepared,
            ),
        }
        .map_err(|error| cuda_error("launch residual GEMV", error))
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
        self.profile_gemv(first_shape)?;
        self.profile_gemv(second_shape)?;
        let input_prepared = self.take_prepared_input(first_shape.columns(), input)?;
        self.profile_launches(if input_prepared { 1 } else { 2 });
        check_layout(
            "first paired GEMV weights",
            first_shape.layout()?,
            first_weights.layout,
        )?;
        check_layout(
            "second paired GEMV weights",
            second_shape.layout()?,
            second_weights.layout,
        )?;
        let first_cuda_shape = quant_shape(first_shape)?;
        let second_cuda_shape = quant_shape(second_shape)?;
        let scratch_key = first_shape.columns();
        if !self.gemv_scratch.contains_key(&scratch_key) {
            let scratch = GemvScratch::new(&self.context, first_cuda_shape)
                .map_err(|error| cuda_error("allocate paired GEMV scratch", error))?;
            self.gemv_scratch.insert(scratch_key, scratch);
        }
        let scratch = self
            .gemv_scratch
            .get_mut(&scratch_key)
            .ok_or_else(|| BackendError::operation("find paired GEMV scratch", "missing entry"))?;
        gemv_pair_q4_k(
            &self.stream,
            first_weights.bytes()?,
            first_cuda_shape,
            second_weights.bytes()?,
            second_cuda_shape,
            input.f32()?,
            first_output.f32_mut()?,
            second_output.f32_mut()?,
            scratch,
            input_prepared,
        )
        .map_err(|error| cuda_error("launch paired GEMV", error))
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
        if gate_shape.format() != QuantFormat::Q4K
            || up_shape.format() != QuantFormat::Q4K
            || gate_shape.rows() != up_shape.rows()
            || gate_shape.columns() != up_shape.columns()
            || !gate_shape.rows().is_multiple_of(32)
            || gate_shape.columns() == gate_shape.rows()
        {
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
        self.profile_gemv(gate_shape)?;
        self.profile_gemv(up_shape)?;
        check_layout(
            "fused gate GEMV weights",
            gate_shape.layout()?,
            gate_weights.layout,
        )?;
        check_layout(
            "fused up GEMV weights",
            up_shape.layout()?,
            up_weights.layout,
        )?;
        let gate_cuda_shape = quant_shape(gate_shape)?;
        let up_cuda_shape = quant_shape(up_shape)?;
        let input_key = gate_shape.columns();
        let output_key = gate_shape.rows();
        let input_prepared = self.take_prepared_input(input_key, input)?;
        self.profile_launches(if input_prepared { 1 } else { 2 });
        if !self.gemv_scratch.contains_key(&input_key) {
            let scratch = GemvScratch::new(&self.context, gate_cuda_shape)
                .map_err(|error| cuda_error("allocate fused input scratch", error))?;
            self.gemv_scratch.insert(input_key, scratch);
        }
        if !self.gemv_scratch.contains_key(&output_key) {
            let output_shape =
                crate::QuantizedMatrixShape::new(1, output_key, crate::QuantFormat::Q4K)
                    .map_err(|error| cuda_error("check fused output scratch", error))?;
            let scratch = GemvScratch::new(&self.context, output_shape)
                .map_err(|error| cuda_error("allocate fused output scratch", error))?;
            self.gemv_scratch.insert(output_key, scratch);
        }
        let mut output_scratch = self
            .gemv_scratch
            .remove(&output_key)
            .ok_or_else(|| BackendError::operation("take fused output scratch", "missing entry"))?;
        let result = {
            let input_scratch = self.gemv_scratch.get_mut(&input_key).ok_or_else(|| {
                BackendError::operation("find fused input scratch", "missing entry")
            })?;
            gemv_pair_swiglu_q4_k(
                &self.stream,
                gate_weights.bytes()?,
                gate_cuda_shape,
                up_weights.bytes()?,
                up_cuda_shape,
                input.f32()?,
                gate.f32_mut()?,
                up.f32_mut()?,
                output.f32_mut()?,
                input_scratch,
                &mut output_scratch,
                input_prepared,
            )
            .map_err(|error| cuda_error("launch fused gate, up, and SwiGLU GEMV", error))
        };
        self.gemv_scratch.insert(output_key, output_scratch);
        result?;
        self.mark_prepared_input(output_key, output)
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
        self.profile_gemv(query_shape)?;
        self.profile_gemv(key_shape)?;
        self.profile_gemv(value_shape)?;
        let input_prepared = self.take_prepared_input(query_shape.columns(), input)?;
        self.profile_launches(if input_prepared { 1 } else { 2 });
        check_layout("query weights", query_shape.layout()?, query_weights.layout)?;
        check_layout("key weights", key_shape.layout()?, key_weights.layout)?;
        check_layout("value weights", value_shape.layout()?, value_weights.layout)?;
        let query_cuda_shape = quant_shape(query_shape)?;
        let scratch_key = query_shape.columns();
        if !self.gemv_scratch.contains_key(&scratch_key) {
            let scratch = GemvScratch::new(&self.context, query_cuda_shape)
                .map_err(|error| cuda_error("allocate QKV GEMV scratch", error))?;
            self.gemv_scratch.insert(scratch_key, scratch);
        }
        let scratch = self
            .gemv_scratch
            .get_mut(&scratch_key)
            .ok_or_else(|| BackendError::operation("find QKV GEMV scratch", "missing entry"))?;
        qkv_gemv(
            &self.stream,
            query_weights.bytes()?,
            query_cuda_shape,
            key_weights.bytes()?,
            quant_shape(key_shape)?,
            value_weights.bytes()?,
            quant_shape(value_shape)?,
            input.f32()?,
            query.f32_mut()?,
            key.f32_mut()?,
            value.f32_mut()?,
            scratch,
            input_prepared,
        )
        .map_err(|error| cuda_error("launch QKV GEMV", error))
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
        if !self.gemv_scratch.contains_key(&output_key) {
            let output_shape =
                crate::QuantizedMatrixShape::new(1, output_key, crate::QuantFormat::Q4K)
                    .map_err(|error| cuda_error("check RMSNorm q8_1 scratch", error))?;
            let scratch = GemvScratch::new(&self.context, output_shape)
                .map_err(|error| cuda_error("allocate RMSNorm q8_1 scratch", error))?;
            self.gemv_scratch.insert(output_key, scratch);
        }
        let scratch = self
            .gemv_scratch
            .get_mut(&output_key)
            .ok_or_else(|| BackendError::operation("find RMSNorm q8_1 scratch", "missing entry"))?;
        rms_norm_q8_parallel(
            &self.stream,
            input.f32()?,
            weight.f32()?,
            output.f32_mut()?,
            scratch,
            vector_shape(shape)?,
            epsilon,
        )
        .map_err(|error| cuda_error("launch parallel RMSNorm q8_1", error))?;
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
            Position::Host(position) => rms_norm_rope(
                &self.stream,
                input.f32()?,
                weight.f32()?,
                output.f32_mut()?,
                vector_shape(shape)?,
                position,
                epsilon,
                theta,
            )
            .map_err(|error| cuda_error("launch RMSNorm RoPE", error)),
            Position::Device(position) => rms_norm_rope_device_position(
                &self.stream,
                input.f32()?,
                weight.f32()?,
                output.f32_mut()?,
                vector_shape(shape)?,
                position.u32()?,
                epsilon,
                theta,
            )
            .map_err(|error| cuda_error("launch device-position RMSNorm RoPE", error)),
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
        let scratch = self.rope_scratch.as_ref().ok_or_else(|| {
            BackendError::operation("launch QK RMSNorm RoPE", "RoPE is not configured")
        })?;
        qk_norm_rope(
            &self.stream,
            query.f32()?,
            query_weight.f32()?,
            query_output.f32_mut()?,
            vector_shape(query_shape)?,
            key.f32()?,
            key_weight.f32()?,
            key_output.f32_mut()?,
            vector_shape(key_shape)?,
            scratch,
            epsilon,
        )
        .map_err(|error| cuda_error("launch QK RMSNorm RoPE", error))
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
            <Self as Backend>::qk_norm_rope(
                self,
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
            return <Self as Backend>::kv_append(
                self,
                key_output,
                value,
                key_cache,
                value_cache,
                shape,
                position,
            );
        }
        self.profile_launches(1);
        let scratch = self.rope_scratch.as_ref().ok_or_else(|| {
            BackendError::operation("launch QK RMSNorm RoPE", "RoPE is not configured")
        })?;
        let shape = attention_shape(shape)?;
        let result = match (key_cache.layout.storage(), position) {
            (BufferStorage::F32, Position::Host(position)) => qk_norm_rope_kv_append(
                &self.stream,
                query.f32()?,
                query_weight.f32()?,
                query_output.f32_mut()?,
                vector_shape(query_shape)?,
                key.f32()?,
                key_weight.f32()?,
                key_output.f32_mut()?,
                vector_shape(key_shape)?,
                value.f32()?,
                key_cache.f32_mut()?,
                value_cache.f32_mut()?,
                shape,
                position,
                scratch,
                epsilon,
            ),
            (BufferStorage::F16, Position::Host(position)) => qk_norm_rope_kv_append_f16(
                &self.stream,
                query.f32()?,
                query_weight.f32()?,
                query_output.f32_mut()?,
                vector_shape(query_shape)?,
                key.f32()?,
                key_weight.f32()?,
                key_output.f32_mut()?,
                vector_shape(key_shape)?,
                value.f32()?,
                key_cache.f16_mut()?,
                value_cache.f16_mut()?,
                shape,
                position,
                scratch,
                epsilon,
            ),
            (BufferStorage::F32, Position::Device(position)) => {
                qk_norm_rope_kv_append_device_position(
                    &self.stream,
                    query.f32()?,
                    query_weight.f32()?,
                    query_output.f32_mut()?,
                    vector_shape(query_shape)?,
                    key.f32()?,
                    key_weight.f32()?,
                    key_output.f32_mut()?,
                    vector_shape(key_shape)?,
                    value.f32()?,
                    key_cache.f32_mut()?,
                    value_cache.f32_mut()?,
                    shape,
                    position.u32()?,
                    scratch,
                    epsilon,
                )
            }
            (BufferStorage::F16, Position::Device(position)) => {
                qk_norm_rope_kv_append_f16_device_position(
                    &self.stream,
                    query.f32()?,
                    query_weight.f32()?,
                    query_output.f32_mut()?,
                    vector_shape(query_shape)?,
                    key.f32()?,
                    key_weight.f32()?,
                    key_output.f32_mut()?,
                    vector_shape(key_shape)?,
                    value.f32()?,
                    key_cache.f16_mut()?,
                    value_cache.f16_mut()?,
                    shape,
                    position.u32()?,
                    scratch,
                    epsilon,
                )
            }
            (storage, _) => return Err(storage_error("write KV cache", storage)),
        };
        result.map_err(|error| cuda_error("launch QK RMSNorm RoPE with KV append", error))
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
        let result = match key_cache.layout.storage() {
            BufferStorage::F32 => crate::cuda::verify_qk_norm_rope_kv_append(
                &self.stream,
                query.f32()?,
                query_weight.f32()?,
                query_output.f32_mut()?,
                vector_shape(query_shape)?,
                key.f32()?,
                key_weight.f32()?,
                key_output.f32_mut()?,
                vector_shape(key_shape)?,
                value.f32()?,
                key_cache.f32_mut()?,
                value_cache.f32_mut()?,
                shape,
                start_position,
                positions,
                scratch,
                epsilon,
            ),
            BufferStorage::F16 => crate::cuda::verify_qk_norm_rope_kv_append_f16(
                &self.stream,
                query.f32()?,
                query_weight.f32()?,
                query_output.f32_mut()?,
                vector_shape(query_shape)?,
                key.f32()?,
                key_weight.f32()?,
                key_output.f32_mut()?,
                vector_shape(key_shape)?,
                value.f32()?,
                key_cache.f16_mut()?,
                value_cache.f16_mut()?,
                shape,
                start_position,
                positions,
                scratch,
                epsilon,
            ),
            storage => return Err(storage_error("write verifier KV cache", storage)),
        };
        result.map_err(|error| cuda_error("launch verifier QK RMSNorm RoPE", error))
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
        let result = match (key_cache.layout.storage(), position) {
            (BufferStorage::F32, Position::Host(position)) => kv_append(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.f32_mut()?,
                value_cache.f32_mut()?,
                shape,
                position,
            ),
            (BufferStorage::F16, Position::Host(position)) => kv_append_f16(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.f16_mut()?,
                value_cache.f16_mut()?,
                shape,
                position,
            ),
            (BufferStorage::F32, Position::Device(position)) => kv_append_device_position(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.f32_mut()?,
                value_cache.f32_mut()?,
                shape,
                position.u32()?,
            ),
            (BufferStorage::F16, Position::Device(position)) => kv_append_f16_device_position(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.f16_mut()?,
                value_cache.f16_mut()?,
                shape,
                position.u32()?,
            ),
            (BufferStorage::Q8Kv, Position::Host(position)) => kv_append_q8(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.bytes_mut()?,
                value_cache.bytes_mut()?,
                shape,
                position,
            ),
            (BufferStorage::Q8Kv, Position::Device(position)) => kv_append_q8_device_position(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.bytes_mut()?,
                value_cache.bytes_mut()?,
                shape,
                position.u32()?,
            ),
            (storage, _) => return Err(storage_error("write KV cache", storage)),
        };
        result.map_err(|error| cuda_error("launch KV append", error))
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
        let result = match key_cache.layout.storage() {
            BufferStorage::F32 => kv_append_chunk(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.f32_mut()?,
                value_cache.f32_mut()?,
                shape,
                start_position,
                tokens,
            ),
            BufferStorage::F16 => kv_append_chunk_f16(
                &self.stream,
                key.f32()?,
                value.f32()?,
                key_cache.f16_mut()?,
                value_cache.f16_mut()?,
                shape,
                start_position,
                tokens,
            ),
            storage => return Err(storage_error("write prefill KV cache", storage)),
        };
        result.map_err(|error| cuda_error("launch prefill KV append", error))
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
        if !self.attention_scratch.contains_key(&shape) {
            let scratch = AttentionScratch::new(&self.context, cuda_shape)
                .map_err(|error| cuda_error("allocate attention scratch", error))?;
            self.attention_scratch.insert(shape, scratch);
        }
        // The fused q8_1 epilogue covers only head_dim 128. Another head
        // dimension must not mark a prepared input. The kernel does not fill
        // the scratch, so the output projection would read the previous RMSNorm
        // activation.
        let prepared = shape.head_dim() == PREPARED_ATTENTION_HEAD_DIM;
        let output_key = shape.query_elements()?;
        if prepared && !self.gemv_scratch.contains_key(&output_key) {
            let output_shape =
                crate::QuantizedMatrixShape::new(1, output_key, crate::QuantFormat::Q4K)
                    .map_err(|error| cuda_error("check attention q8_1 scratch", error))?;
            let scratch = GemvScratch::new(&self.context, output_shape)
                .map_err(|error| cuda_error("allocate attention q8_1 scratch", error))?;
            self.gemv_scratch.insert(output_key, scratch);
        }
        let scratch = self
            .attention_scratch
            .get_mut(&shape)
            .ok_or_else(|| BackendError::operation("find attention scratch", "missing entry"))?;
        let prepared_output = match self.gemv_scratch.get_mut(&output_key) {
            Some(scratch) if prepared => Some(scratch),
            _ => None,
        };
        // Device-held positions stay on device. Context length is `position + 1`.
        let result = match (key_cache.layout.storage(), position) {
            (BufferStorage::F32, Position::Host(position)) => {
                let context_length = attention_context_length(position)?;
                attention_decode(
                    &self.stream,
                    query.f32()?,
                    key_cache.f32()?,
                    value_cache.f32()?,
                    output.f32_mut()?,
                    scratch,
                    prepared_output,
                    cuda_shape,
                    context_length,
                )
            }
            (BufferStorage::F16, Position::Host(position)) => {
                let context_length = attention_context_length(position)?;
                attention_decode_f16(
                    &self.stream,
                    query.f32()?,
                    key_cache.f16()?,
                    value_cache.f16()?,
                    output.f32_mut()?,
                    scratch,
                    prepared_output,
                    cuda_shape,
                    context_length,
                )
            }
            (BufferStorage::F32, Position::Device(position)) => attention_decode_device_position(
                &self.stream,
                query.f32()?,
                key_cache.f32()?,
                value_cache.f32()?,
                output.f32_mut()?,
                scratch,
                prepared_output,
                cuda_shape,
                position.u32()?,
            ),
            (BufferStorage::F16, Position::Device(position)) => {
                attention_decode_f16_device_position(
                    &self.stream,
                    query.f32()?,
                    key_cache.f16()?,
                    value_cache.f16()?,
                    output.f32_mut()?,
                    scratch,
                    prepared_output,
                    cuda_shape,
                    position.u32()?,
                )
            }
            (BufferStorage::Q8Kv, Position::Host(position)) => {
                let context_length = attention_context_length(position)?;
                attention_decode_q8(
                    &self.stream,
                    query.f32()?,
                    key_cache.bytes()?,
                    value_cache.bytes()?,
                    output.f32_mut()?,
                    scratch,
                    prepared_output,
                    cuda_shape,
                    context_length,
                )
            }
            (BufferStorage::Q8Kv, Position::Device(position)) => {
                attention_decode_q8_device_position(
                    &self.stream,
                    query.f32()?,
                    key_cache.bytes()?,
                    value_cache.bytes()?,
                    output.f32_mut()?,
                    scratch,
                    prepared_output,
                    cuda_shape,
                    position.u32()?,
                )
            }
            (storage, _) => return Err(storage_error("read KV cache", storage)),
        };
        result.map_err(|error| cuda_error("launch attention", error))?;
        if prepared {
            self.mark_prepared_input(output_key, output)?;
        }
        Ok(())
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
        let scratch_key = (shape, positions);
        if !self.verify_attention_scratch.contains_key(&scratch_key) {
            let scratch = AttentionScratch::new_multi(&self.context, cuda_shape, positions)
                .map_err(|error| cuda_error("allocate verifier attention scratch", error))?;
            self.verify_attention_scratch.insert(scratch_key, scratch);
        }
        let scratch = self
            .verify_attention_scratch
            .get_mut(&scratch_key)
            .ok_or_else(|| {
                BackendError::operation("find verifier attention scratch", "missing entry")
            })?;
        let output_key = shape.query_elements()?;
        let gemv_key = (output_key, positions);
        if !self.verify_gemv_scratch.contains_key(&gemv_key) {
            let gemv_scratch = GemvScratch::new_multi(&self.context, output_key, positions)
                .map_err(|error| cuda_error("allocate verifier attention q8_1 scratch", error))?;
            self.verify_gemv_scratch.insert(gemv_key, gemv_scratch);
        }
        let prepared_output = self.verify_gemv_scratch.get_mut(&gemv_key).ok_or_else(|| {
            BackendError::operation("find verifier attention q8_1 scratch", "missing entry")
        })?;
        let result = match key_cache.layout.storage() {
            BufferStorage::F32 => crate::cuda::verify_attention(
                &self.stream,
                query.f32()?,
                key_cache.f32()?,
                value_cache.f32()?,
                output.f32_mut()?,
                scratch,
                Some(prepared_output),
                cuda_shape,
                start_position,
                positions,
            ),
            BufferStorage::F16 => crate::cuda::verify_attention_f16(
                &self.stream,
                query.f32()?,
                key_cache.f16()?,
                value_cache.f16()?,
                output.f32_mut()?,
                scratch,
                Some(prepared_output),
                cuda_shape,
                start_position,
                positions,
            ),
            storage => return Err(storage_error("read verifier KV cache", storage)),
        };
        result.map_err(|error| cuda_error("launch verifier attention", error))
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
        if !self.attention_scratch.contains_key(&shape) {
            let scratch = AttentionScratch::new(&self.context, cuda_shape)
                .map_err(|error| cuda_error("allocate future decode attention scratch", error))?;
            self.attention_scratch.insert(shape, scratch);
        }
        let handle = self.cublaslt.as_ref().ok_or_else(|| {
            BackendError::operation("run prefill attention", "cuBLASLt is not prepared")
        })?;
        let scratch = self.prefill_scratch.as_mut().ok_or_else(|| {
            BackendError::operation("run prefill attention", "prefill workspace is not prepared")
        })?;
        let result = match key_cache.layout.storage() {
            BufferStorage::F32 => attention_prefill_f32(
                handle,
                &self.stream,
                query.f32()?,
                key_cache.f32()?,
                value_cache.f32()?,
                output.f32_mut()?,
                cuda_shape,
                start_position,
                tokens,
                scratch,
            ),
            BufferStorage::F16 => attention_prefill_f16(
                handle,
                &self.stream,
                query.f32()?,
                key_cache.f16()?,
                value_cache.f16()?,
                output.f32_mut()?,
                cuda_shape,
                start_position,
                tokens,
                scratch,
            ),
            storage => return Err(storage_error("read prefill KV cache", storage)),
        };
        result.map_err(|error| cuda_error("run prefill attention", error))
    }

    fn embed_gather(
        &mut self,
        table: &Self::Buffer,
        row: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        self.profile_launches(1);
        check_layout("embedding table", shape.layout()?, table.layout)?;
        let cuda_shape = quant_shape(shape)?;
        let result = match shape.format() {
            QuantFormat::Q4K => embedding_gather_q4_k_device_row(
                &self.stream,
                table.bytes()?,
                row.u32()?,
                output.f32_mut()?,
                cuda_shape,
            ),
            QuantFormat::Q6K => embedding_gather_q6_k_device_row(
                &self.stream,
                table.bytes()?,
                row.u32()?,
                output.f32_mut()?,
                cuda_shape,
            ),
        };
        result.map_err(|error| cuda_error("launch embedding gather", error))
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
        self.decode_graph = None;
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
        let operation_count = profiler.operations.len();
        if operation_count > 0 {
            let end = profiler.events.get_mut(operation_count).ok_or_else(|| {
                BackendError::operation("finish decode profile", "final event is missing")
            })?;
            end.record(&self.stream)
                .map_err(|error| cuda_error("record final decode profile event", error))?;
            end.synchronize()
                .map_err(|error| cuda_error("wait for decode profile", error))?;
        }

        let mut gpu_duration_by_op = BTreeMap::new();
        let mut gemv_by_shape = BTreeMap::new();
        for (index, operation) in profiler.operations.iter().enumerate() {
            let milliseconds =
                Event::elapsed_ms(&profiler.events[index], &profiler.events[index + 1])
                    .map_err(|error| cuda_error("measure decode profile event", error))?;
            let duration = Duration::from_secs_f64(f64::from(milliseconds) / 1_000.0);
            *gpu_duration_by_op.entry(operation.class).or_default() += duration;
            let total_bytes =
                operation
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
                let shape_bytes = u64::try_from(shape.layout()?.bytes()).map_err(|_| {
                    BackendError::SizeOverflow {
                        field: "profiled GEMV bytes",
                    }
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
        }
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
