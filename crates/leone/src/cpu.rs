use crate::backend::{
    exact_len, reserve_rope_host_bytes, validate_positive, HostStaging, MemoryAccounting,
    MemoryAllocation, MemoryBudget, MemoryClass, MemoryError, MemoryTracker, MemoryTrackerRoot,
};
#[cfg(test)]
use crate::PrefillMethod;
use crate::{
    AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot, BufferStorage,
    Determinism, KvReadView, KvWriteSpan, MemoryCapacity, Position, QuantFormat, QuantMatrix,
    RopePairing, RopeShape, VectorShape,
};
use half::f16;
use leone_gguf::ref_dequant;
use rayon::prelude::*;

/// An opaque buffer owned by the scalar CPU backend.
#[derive(Debug)]
pub struct CpuBuffer {
    layout: BufferLayout,
    storage: CpuStorage,
    allocation: MemoryAllocation,
}

#[derive(Debug)]
enum CpuStorage {
    Bytes(Vec<u8>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl CpuBuffer {
    #[cfg(test)]
    pub(crate) fn allocation_identity(&self) -> u64 {
        self.allocation.identity()
    }

    fn f16_mut(&mut self) -> Result<&mut [f16], BackendError> {
        match &mut self.storage {
            CpuStorage::F16(values) => Ok(values),
            _ => Err(storage_error("write f16", self.layout.storage())),
        }
    }

    fn f32(&self) -> Result<&[f32], BackendError> {
        match &self.storage {
            CpuStorage::F32(values) => Ok(values),
            _ => Err(storage_error("read f32", self.layout.storage())),
        }
    }

    fn f32_mut(&mut self) -> Result<&mut [f32], BackendError> {
        match &mut self.storage {
            CpuStorage::F32(values) => Ok(values),
            _ => Err(storage_error("write f32", self.layout.storage())),
        }
    }

    fn u32(&self) -> Result<&[u32], BackendError> {
        match &self.storage {
            CpuStorage::U32(values) => Ok(values),
            _ => Err(storage_error("read u32", self.layout.storage())),
        }
    }

    fn u32_mut(&mut self) -> Result<&mut [u32], BackendError> {
        match &mut self.storage {
            CpuStorage::U32(values) => Ok(values),
            _ => Err(storage_error("write u32", self.layout.storage())),
        }
    }

    fn bytes(&self) -> Result<&[u8], BackendError> {
        match &self.storage {
            CpuStorage::Bytes(values) => Ok(values),
            _ => Err(storage_error("read quantized bytes", self.layout.storage())),
        }
    }

    fn bytes_mut(&mut self) -> Result<&mut [u8], BackendError> {
        match &mut self.storage {
            CpuStorage::Bytes(values) => Ok(values),
            _ => Err(storage_error(
                "write quantized bytes",
                self.layout.storage(),
            )),
        }
    }
}

/// The scalar reference implementation of the backend contract.
#[derive(Debug, Default)]
pub struct CpuBackend {
    q8_1_activations: bool,
    rope_inverse_frequencies: Vec<f64>,
    rope_host_allocation: Option<MemoryAllocation>,
    rope_pairing: RopePairing,
    memory: MemoryTracker,
    #[cfg(test)]
    allocation_fail_after: Option<usize>,
    #[cfg(test)]
    reference_verify: bool,
    #[cfg(test)]
    pub(crate) attention_probe: Option<AttentionProbe>,
    #[cfg(test)]
    pub(crate) completion_probe: Option<CompletionProbe>,
    #[cfg(test)]
    pub(crate) preflight_probe: Option<PreflightProbe>,
    #[cfg(test)]
    pub(crate) graph_capture: bool,
    #[cfg(test)]
    test_batch_size: Option<std::num::NonZeroUsize>,
    #[cfg(test)]
    pub(crate) verification_probe: Option<VerificationProbe>,
    #[cfg(test)]
    prefill_routing_stub: bool,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct CompletionProbe {
    pub(crate) calls: usize,
    pub(crate) fail: bool,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct VerificationProbe {
    pub(crate) fail_qkv: bool,
    pub(crate) fail_ffn: bool,
    pub(crate) fail_drop: bool,
    pub(crate) fail_drop_forever: bool,
    pub(crate) drop_graph_calls: usize,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct AttentionProbe {
    pub(crate) events: Vec<AttentionEvent>,
    pub(crate) prepare_fail_after: Option<usize>,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct PreflightProbe {
    pub(crate) end_keep_calls: usize,
    pub(crate) fail_end_keep: bool,
    pub(crate) begin_live_bytes: Option<u64>,
    pub(crate) begin_live_allocations: Option<u64>,
    pub(crate) drop_calls: usize,
    pub(crate) graph_buffer: Option<CpuBuffer>,
    pub(crate) fail_drop_before_release: bool,
    pub(crate) fail_drop_after_release: bool,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AttentionEvent {
    Prepare { rows: usize, device_positions: bool },
    Append,
    Dispatch { rows: usize },
    Scatter,
}

impl CpuBackend {
    /// Creates a scalar CPU backend with no declared memory limit.
    pub fn new() -> Self {
        Self::with_memory_budget(MemoryBudget::Unlimited)
    }

    /// Creates a scalar CPU backend with an owned-allocation budget.
    pub fn with_memory_budget(budget: MemoryBudget) -> Self {
        Self::with_memory_tracker(MemoryTracker::new(budget))
    }

    /// Creates a scalar CPU backend with a caller-owned allocation tracker.
    pub fn with_memory_tracker(memory: MemoryTracker) -> Self {
        Self {
            q8_1_activations: false,
            rope_inverse_frequencies: Vec::new(),
            rope_host_allocation: None,
            rope_pairing: RopePairing::HalfSplit,
            memory,
            #[cfg(test)]
            allocation_fail_after: None,
            #[cfg(test)]
            reference_verify: false,
            #[cfg(test)]
            attention_probe: None,
            #[cfg(test)]
            completion_probe: None,
            #[cfg(test)]
            preflight_probe: None,
            #[cfg(test)]
            graph_capture: false,
            #[cfg(test)]
            test_batch_size: None,
            #[cfg(test)]
            verification_probe: None,
            #[cfg(test)]
            prefill_routing_stub: false,
        }
    }

    /// Creates a diagnostic CPU backend that emulates CUDA q8_1 activations.
    ///
    /// The quantizer uses 32-value blocks and stores each scale as `f16`.
    /// This mode isolates activation quantization from other backend differences.
    pub fn with_q8_1_activations() -> Self {
        Self {
            q8_1_activations: true,
            rope_inverse_frequencies: Vec::new(),
            rope_host_allocation: None,
            rope_pairing: RopePairing::HalfSplit,
            memory: MemoryTracker::default(),
            #[cfg(test)]
            allocation_fail_after: None,
            #[cfg(test)]
            reference_verify: false,
            #[cfg(test)]
            attention_probe: None,
            #[cfg(test)]
            completion_probe: None,
            #[cfg(test)]
            preflight_probe: None,
            #[cfg(test)]
            graph_capture: false,
            #[cfg(test)]
            test_batch_size: None,
            #[cfg(test)]
            verification_probe: None,
            #[cfg(test)]
            prefill_routing_stub: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn fail_allocations_after(&mut self, allocations: usize) {
        self.allocation_fail_after = Some(allocations);
    }

    #[cfg(test)]
    pub(crate) fn clear_allocation_failure(&mut self) {
        self.allocation_fail_after = None;
    }

    #[cfg(test)]
    pub(crate) fn enable_reference_verify(&mut self) {
        self.reference_verify = true;
    }

    #[cfg(test)]
    pub(crate) fn set_test_batch_size(&mut self, size: usize) {
        self.test_batch_size = std::num::NonZeroUsize::new(size);
    }

    #[cfg(test)]
    pub(crate) fn enable_prefill_routing_stub(&mut self) {
        self.prefill_routing_stub = true;
    }

    #[cfg(test)]
    fn record_attention_event(&mut self, event: AttentionEvent) {
        if let Some(probe) = self.attention_probe.as_mut() {
            probe.events.push(event);
        }
    }

    #[cfg(test)]
    fn fail_attention_preparation(&mut self) -> Result<(), BackendError> {
        let remaining = self
            .attention_probe
            .as_mut()
            .and_then(|probe| probe.prepare_fail_after.as_mut());
        if let Some(remaining) = remaining {
            if *remaining == 0 {
                return Err(BackendError::operation(
                    "prepare CPU batch attention",
                    "injected preparation failure",
                ));
            }
            *remaining -= 1;
        }
        Ok(())
    }

    #[cfg(test)]
    fn fail_qkv_verification(&mut self) -> Result<(), BackendError> {
        if let Some(probe) = self.verification_probe.as_mut() {
            if probe.fail_qkv {
                probe.fail_qkv = false;
                return Err(BackendError::operation(
                    "run CPU batch QKV",
                    "injected QKV failure",
                ));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn fail_ffn_verification(&mut self) -> Result<(), BackendError> {
        if let Some(probe) = self.verification_probe.as_mut() {
            if probe.fail_ffn {
                probe.fail_ffn = false;
                return Err(BackendError::operation(
                    "run CPU batch FFN",
                    "injected FFN failure",
                ));
            }
        }
        Ok(())
    }
}

impl Backend for CpuBackend {
    #[cfg(test)]
    fn begin_kv_graph_preflight(&mut self) -> Result<(), BackendError> {
        if let Some(probe) = self.preflight_probe.as_mut() {
            let memory = self.memory.snapshot();
            probe.begin_live_bytes = Some(memory.live_bytes);
            probe.begin_live_allocations = Some(memory.live_allocations);
        }
        Ok(())
    }

    #[cfg(test)]
    fn end_kv_graph_preflight(&mut self, keep: bool) -> Result<(), BackendError> {
        if let Some(probe) = self.preflight_probe.as_mut() {
            if keep {
                probe.end_keep_calls += 1;
                if probe.fail_end_keep {
                    return Err(BackendError::operation(
                        "finish CPU KV graph preflight",
                        "injected finalization failure",
                    ));
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn decode_graph_supported(&self) -> bool {
        self.graph_capture
    }

    #[cfg(test)]
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
        self.fail_qkv_verification()?;
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

    #[cfg(test)]
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
        self.fail_ffn_verification()?;
        self.verify_gemv(first_weights, input, first_output, first_shape, positions)?;
        self.verify_gemv(
            second_weights,
            input,
            second_output,
            second_shape,
            positions,
        )
    }

    #[cfg(test)]
    fn begin_decode_graph(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    #[cfg(test)]
    fn end_decode_graph(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    #[cfg(test)]
    fn replay_decode_graph(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    #[cfg(test)]
    fn drop_decode_graph(&mut self) -> Result<(), BackendError> {
        if let Some(probe) = self.preflight_probe.as_mut() {
            if probe.fail_drop_before_release {
                return Err(BackendError::operation(
                    "drop CPU decode graph",
                    "injected pre-release failure",
                ));
            }
            probe.drop_calls += 1;
            probe.graph_buffer.take();
            if probe.fail_drop_after_release {
                return Err(BackendError::operation(
                    "drop CPU decode graph",
                    "injected post-release failure",
                ));
            }
        }
        if let Some(probe) = self.verification_probe.as_mut() {
            probe.drop_graph_calls += 1;
            if probe.fail_drop || probe.fail_drop_forever {
                probe.fail_drop = false;
                return Err(BackendError::operation(
                    "drop CPU decode graph",
                    "injected graph retirement failure",
                ));
            }
        }
        Ok(())
    }

    fn configure_rope(
        &mut self,
        head_dim: usize,
        theta: f32,
        frequency_factors: Option<&[f32]>,
        pairing: RopePairing,
    ) -> Result<(), BackendError> {
        let inverse = build_rope_inverse_frequencies(head_dim, theta, frequency_factors)?;
        self.rope_inverse_frequencies = inverse;
        self.rope_host_allocation = None;
        self.rope_pairing = pairing;
        Ok(())
    }

    fn configure_rope_with_host_staging(
        &mut self,
        head_dim: usize,
        theta: f32,
        frequency_factors: Option<&[f32]>,
        pairing: RopePairing,
        staging: &HostStaging,
    ) -> Result<(), BackendError> {
        let reservation = reserve_rope_host_bytes(staging, head_dim)?;
        let inverse = build_rope_inverse_frequencies(head_dim, theta, frequency_factors)?;
        let allocation = reservation
            .map(|reservation| reservation.commit())
            .transpose()?;
        self.rope_inverse_frequencies = inverse;
        self.rope_host_allocation = allocation;
        self.rope_pairing = pairing;
        Ok(())
    }

    type Buffer = CpuBuffer;

    fn name(&self) -> &'static str {
        if self.q8_1_activations {
            "cpu-q8_1"
        } else {
            "cpu"
        }
    }

    fn max_batch_size(&self) -> std::num::NonZeroUsize {
        #[cfg(test)]
        if let Some(size) = self.test_batch_size {
            return size;
        }
        std::num::NonZeroUsize::new(1).expect("one is nonzero")
    }

    fn determinism(&self) -> Determinism {
        Determinism::FixedOrder
    }

    #[cfg(test)]
    fn prefill_method(&self) -> PrefillMethod {
        if self.prefill_routing_stub {
            PrefillMethod::ChunkedGpu
        } else {
            PrefillMethod::SequentialDecode
        }
    }

    #[cfg(test)]
    fn decode_equivalent_prefill_supported(&self) -> bool {
        self.prefill_routing_stub
    }

    #[cfg(test)]
    fn verify_supported(&self) -> bool {
        self.reference_verify
    }

    #[cfg(test)]
    fn verify_gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.prefill_gemm(weights, input, output, shape, positions)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn verify_gemv_residual(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        positions: usize,
    ) -> Result<(), BackendError> {
        self.prefill_gemm(weights, input, output, shape, positions)?;
        let residual = residual.f32()?;
        let output = output.f32_mut()?;
        exact_len("verifier residual output", residual.len(), output.len())?;
        for (output, residual) in output.iter_mut().zip(residual) {
            *output += residual;
        }
        Ok(())
    }

    fn memory_accounting(&self) -> MemoryAccounting {
        self.memory.snapshot()
    }

    fn memory_tracker_root(&self) -> MemoryTrackerRoot {
        self.memory.root()
    }

    fn classify_buffer(
        &mut self,
        buffer: &Self::Buffer,
        class: MemoryClass,
    ) -> Result<(), BackendError> {
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

    fn memory_capacity(&mut self) -> Result<MemoryCapacity, BackendError> {
        Ok(MemoryCapacity::Unbounded)
    }

    fn allocate(&mut self, layout: BufferLayout) -> Result<Self::Buffer, BackendError> {
        self.allocate_classified(layout, MemoryClass::ContractBuffer)
    }

    fn allocate_classified(
        &mut self,
        layout: BufferLayout,
        class: MemoryClass,
    ) -> Result<Self::Buffer, BackendError> {
        #[cfg(test)]
        if let Some(remaining) = self.allocation_fail_after.as_mut() {
            if *remaining == 0 {
                return Err(BackendError::operation(
                    "allocate CPU buffer",
                    "injected allocation failure",
                ));
            }
            *remaining -= 1;
        }
        let bytes = u64::try_from(layout.bytes()).map_err(|_| BackendError::SizeOverflow {
            field: "CPU buffer bytes",
        })?;
        let reservation = self.memory.reserve(class, bytes)?;
        let storage = allocate_cpu_storage(layout)?;
        let allocation = reservation.commit()?;
        Ok(CpuBuffer {
            layout,
            storage,
            allocation,
        })
    }

    fn upload(&mut self, layout: BufferLayout, bytes: &[u8]) -> Result<Self::Buffer, BackendError> {
        exact_len("uploaded bytes", layout.bytes(), bytes.len())?;
        let tracked_bytes =
            u64::try_from(layout.bytes()).map_err(|_| BackendError::SizeOverflow {
                field: "CPU buffer bytes",
            })?;
        let reservation = self
            .memory
            .reserve(MemoryClass::ModelWeight, tracked_bytes)?;
        let storage = parse_cpu_storage(layout, bytes)?;
        let allocation = reservation.commit()?;
        Ok(CpuBuffer {
            layout,
            storage,
            allocation,
        })
    }

    fn clone_buffer(&mut self, source: &Self::Buffer) -> Result<Self::Buffer, BackendError> {
        let class = source.allocation.class();
        let bytes =
            u64::try_from(source.layout.bytes()).map_err(|_| BackendError::SizeOverflow {
                field: "CPU buffer bytes",
            })?;
        let reservation = self.memory.reserve(class, bytes)?;
        let storage = clone_cpu_storage(&source.storage)?;
        let allocation = reservation.commit()?;
        Ok(CpuBuffer {
            layout: source.layout,
            storage,
            allocation,
        })
    }

    fn download_buffer(&mut self, source: &Self::Buffer) -> Result<BufferSnapshot, BackendError> {
        let bytes = encode_cpu_storage(&source.storage, source.layout.bytes())?;
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
        let bytes = u64::try_from(layout.bytes()).map_err(|_| BackendError::SizeOverflow {
            field: "CPU buffer bytes",
        })?;
        let reservation = self.memory.reserve(class, bytes)?;
        let storage = parse_cpu_storage(layout, source.bytes())?;
        let allocation = reservation.commit()?;
        Ok(CpuBuffer {
            layout,
            storage,
            allocation,
        })
    }

    fn write_u32(&mut self, buffer: &mut Self::Buffer, values: &[u32]) -> Result<(), BackendError> {
        let destination = buffer.u32_mut()?;
        exact_len("u32 write", destination.len(), values.len())?;
        destination.copy_from_slice(values);
        Ok(())
    }

    fn read_u32(&mut self, buffer: &Self::Buffer, values: &mut [u32]) -> Result<(), BackendError> {
        let source = buffer.u32()?;
        exact_len("u32 read", source.len(), values.len())?;
        values.copy_from_slice(source);
        Ok(())
    }

    fn read_f16(&mut self, buffer: &Self::Buffer, values: &mut [u16]) -> Result<(), BackendError> {
        let source = match &buffer.storage {
            CpuStorage::F16(source) => source,
            _ => return Err(storage_error("read f16", buffer.layout.storage())),
        };
        exact_len("f16 read", source.len(), values.len())?;
        for (destination, source) in values.iter_mut().zip(source) {
            *destination = source.to_bits();
        }
        Ok(())
    }

    fn read_f32(&mut self, buffer: &Self::Buffer, values: &mut [f32]) -> Result<(), BackendError> {
        let source = buffer.f32()?;
        exact_len("f32 read", source.len(), values.len())?;
        values.copy_from_slice(source);
        Ok(())
    }

    fn prefill_gemm(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let (input, output) = prefill_gemm_buffers(weights, input, output, shape, tokens)?;
        let quantized_input = self.q8_1_activations.then(|| emulate_q8_1(input));
        let input = quantized_input.as_deref().unwrap_or(input);
        gemm_rows(input, output, weights.bytes()?, shape, shape.row_bytes()?)
    }

    fn gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        let (input, output) = gemv_buffers(weights, input, output, shape)?;
        let quantized_input = self.q8_1_activations.then(|| emulate_q8_1(input));
        let input = quantized_input.as_deref().unwrap_or(input);
        gemv_rows(input, output, weights.bytes()?, shape, shape.row_bytes()?)?;
        Ok(())
    }

    fn gemv_residual(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        self.gemv(weights, input, output, shape)?;
        let residual = residual.f32()?;
        let output = output.f32_mut()?;
        exact_len("GEMV residual", output.len(), residual.len())?;
        for (output, residual) in output.iter_mut().zip(residual) {
            *output += residual;
        }
        Ok(())
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
        self.gemv(query_weights, input, query, query_shape)?;
        self.gemv(key_weights, input, key, key_shape)?;
        self.gemv(value_weights, input, value, value_shape)
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
        let input = input.f32()?;
        let weight = weight.f32()?;
        let output = output.f32_mut()?;
        let elements = shape.elements()?;
        exact_len("RMSNorm input", elements, input.len())?;
        exact_len("RMSNorm weight", shape.columns(), weight.len())?;
        exact_len("RMSNorm output", elements, output.len())?;
        rms_norm_rows(input, weight, output, shape, epsilon);
        Ok(())
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
        let position = self.resolve_position(position)?;
        self.rms_norm(input, weight, output, shape, epsilon)?;
        self.rope(
            output,
            position,
            RopeShape::new(1, shape.rows(), shape.columns())?,
            theta,
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
        let RmsResidualBuffers {
            left,
            right,
            weight,
            output,
        } = rms_residual_buffers(left, right, weight, output)?;
        validate_rms_residual_lengths(left, right, weight, output, shape)?;
        rms_residual_rows(left, right, weight, output, shape, epsilon);
        Ok(())
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
        let RmsResidualStoreBuffers {
            left,
            right,
            weight,
            residual,
            output,
        } = rms_residual_store_buffers(left, right, weight, residual, output)?;
        validate_rms_residual_store_lengths(left, right, weight, residual, output, shape)?;
        rms_residual_store_rows(left, right, weight, residual, output, shape, epsilon);
        Ok(())
    }

    fn rope(
        &mut self,
        values: &mut Self::Buffer,
        position: usize,
        shape: RopeShape,
        _theta: f32,
    ) -> Result<(), BackendError> {
        let values = values.f32_mut()?;
        exact_len("RoPE values", shape.elements()?, values.len())?;
        let half = shape.head_dim() / 2;
        exact_len(
            "RoPE inverse frequencies",
            half,
            self.rope_inverse_frequencies.len(),
        )?;
        rope_values(
            values,
            position,
            shape,
            half,
            &self.rope_inverse_frequencies,
            self.rope_pairing,
        )?;
        Ok(())
    }

    fn swiglu(
        &mut self,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        let gate = gate.f32()?;
        let up = up.f32()?;
        let output = output.f32_mut()?;
        exact_len("SwiGLU up", gate.len(), up.len())?;
        exact_len("SwiGLU output", gate.len(), output.len())?;
        for ((destination, gate), up) in output.iter_mut().zip(gate).zip(up) {
            *destination = (*gate / (1.0 + (-*gate).exp())) * *up;
        }
        Ok(())
    }

    fn residual_add(
        &mut self,
        left: &Self::Buffer,
        right: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        let left = left.f32()?;
        let right = right.f32()?;
        let output = output.f32_mut()?;
        exact_len("residual right", left.len(), right.len())?;
        exact_len("residual output", left.len(), output.len())?;
        for ((destination, left), right) in output.iter_mut().zip(left).zip(right) {
            *destination = left + right;
        }
        Ok(())
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
        let position = self.resolve_position(position)?;
        validate_kv_position(position, shape)?;
        let (key, value, cached) = kv_append_buffers(key, value, shape)?;
        append_kv_by_storage(key, value, key_cache, value_cache, cached, shape, position)?;
        Ok(())
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
        validate_kv_chunk_position(start_position, tokens, shape)?;
        let (key, value, cached) = kv_append_chunk_buffers(key, value, shape, tokens)?;
        let spec = KvChunkSpec {
            cached,
            shape,
            start_position,
            tokens,
        };
        match key_cache.layout.storage() {
            BufferStorage::F32 => append_kv_chunk_f32(key, value, key_cache, value_cache, spec)?,
            BufferStorage::F16 => append_kv_chunk_f16(key, value, key_cache, value_cache, spec)?,
            BufferStorage::Q8Kv => append_kv_chunk_q8(key, value, key_cache, value_cache, spec)?,
            storage => return Err(storage_error("write prefill KV cache", storage)),
        }
        Ok(())
    }

    fn kv_append_span(
        &mut self,
        key: &Self::Buffer,
        value: &Self::Buffer,
        target: KvWriteSpan<'_, Self::Buffer>,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        #[cfg(test)]
        self.record_attention_event(AttentionEvent::Append);
        let position = self.resolve_position(position)?;
        validate_kv_position(position, shape)?;
        let local_position = target.local_position(position)?;
        let physical_shape = physical_attention_shape(shape, target.capacity_token_count())?;
        let (key, value, _) = kv_append_buffers(key, value, shape)?;
        let (key_cache, value_cache, _, _) = target.into_parts();
        validate_kv_span_buffers(key_cache, value_cache, physical_shape)?;
        append_kv_by_storage(
            key,
            value,
            key_cache,
            value_cache,
            physical_shape.cache_elements()?,
            physical_shape,
            local_position,
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
        if tokens == 0 {
            return Err(BackendError::Zero {
                field: "KV append tokens",
            });
        }
        let (physical_shape, local_start) =
            prepare_kv_chunk_span(&target, shape, start_position, tokens)?;
        let (key, value, _) = kv_append_chunk_buffers(key, value, shape, tokens)?;
        let (key_cache, value_cache, _, _) = target.into_parts();
        validate_kv_span_buffers(key_cache, value_cache, physical_shape)?;
        let spec = KvChunkSpec {
            cached: physical_shape.cache_elements()?,
            shape: physical_shape,
            start_position: local_start,
            tokens,
        };
        match key_cache.layout.storage() {
            BufferStorage::F32 => append_kv_chunk_f32(key, value, key_cache, value_cache, spec),
            BufferStorage::F16 => append_kv_chunk_f16(key, value, key_cache, value_cache, spec),
            BufferStorage::Q8Kv => append_kv_chunk_q8(key, value, key_cache, value_cache, spec),
            storage => Err(storage_error("write span KV cache", storage)),
        }
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
        let context_length =
            self.resolve_position(position)?
                .checked_add(1)
                .ok_or(BackendError::SizeOverflow {
                    field: "attention context length",
                })?;
        if !(1..=shape.max_context()).contains(&context_length) {
            return Err(BackendError::PositionOutOfBounds {
                position: context_length,
                max_context: shape.max_context(),
            });
        }
        let query = query.f32()?;
        let output = output.f32_mut()?;
        validate_attention_buffers(query, output, key_cache, value_cache, shape)?;
        if key_cache.layout.storage() != value_cache.layout.storage() {
            return Err(BackendError::operation(
                "read KV cache",
                "key and value storage differ",
            ));
        }
        attention_decode_rows(query, key_cache, value_cache, output, shape, context_length)?;
        Ok(())
    }

    #[cfg(test)]
    fn prepare_attention_decode_batch_spans(
        &mut self,
        rows: &[crate::AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        self.record_attention_event(AttentionEvent::Prepare {
            rows: rows.len(),
            device_positions: rows
                .iter()
                .all(|row| matches!(row.position, Position::Device(_))),
        });
        self.fail_attention_preparation()?;
        for row in rows {
            self.prepare_kv_read_view(row.cache, row.shape)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn attention_decode_batch_spans(
        &mut self,
        rows: &mut [crate::AttentionDecodeRow<'_, Self::Buffer>],
    ) -> Result<(), BackendError> {
        self.record_attention_event(AttentionEvent::Dispatch { rows: rows.len() });
        for row in rows {
            self.attention_decode_spans(row.query, row.cache, row.output, row.shape, row.position)?;
        }
        Ok(())
    }

    fn attention_decode_spans(
        &mut self,
        query: &Self::Buffer,
        cache: KvReadView<'_, Self::Buffer>,
        output: &mut Self::Buffer,
        shape: AttentionShape,
        position: Position<'_, Self::Buffer>,
    ) -> Result<(), BackendError> {
        let position = self.resolve_position(position)?;
        let context_length = position.checked_add(1).ok_or(BackendError::SizeOverflow {
            field: "attention context length",
        })?;
        if context_length > shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position: context_length,
                max_context: shape.max_context(),
            });
        }
        if context_length > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: context_length,
                max_context: cache.mapped_tokens(),
            });
        }
        let query = query.f32()?;
        let output = output.f32_mut()?;
        validate_span_attention_buffers(query, output, &cache, shape, 1)?;
        attention_decode_span_rows(query, &cache, output, shape, context_length)
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
        let end_position =
            start_position
                .checked_add(tokens)
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill attention end position",
                })?;
        if end_position > shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position: end_position,
                max_context: shape.max_context(),
            });
        }
        let query = query.f32()?;
        let output = output.f32_mut()?;
        validate_prefill_attention_buffers(query, output, key_cache, value_cache, shape, tokens)?;
        if key_cache.layout.storage() != value_cache.layout.storage() {
            return Err(BackendError::operation(
                "read prefill KV cache",
                "key and value storage differ",
            ));
        }
        attention_prefill_rows(
            query,
            key_cache,
            value_cache,
            output,
            shape,
            start_position,
            tokens,
        )?;
        Ok(())
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
        if tokens == 0 {
            return Err(BackendError::Zero {
                field: "prefill attention tokens",
            });
        }
        let end_position =
            start_position
                .checked_add(tokens)
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill attention end position",
                })?;
        if end_position > shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position: end_position,
                max_context: shape.max_context(),
            });
        }
        if end_position > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: end_position,
                max_context: cache.mapped_tokens(),
            });
        }
        let query = query.f32()?;
        let output = output.f32_mut()?;
        validate_span_attention_buffers(query, output, &cache, shape, tokens)?;
        attention_prefill_span_rows(query, &cache, output, shape, start_position, tokens)
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
        if positions == 0 {
            return Err(BackendError::Zero {
                field: "verifier attention positions",
            });
        }
        let end_position =
            start_position
                .checked_add(positions)
                .ok_or(BackendError::SizeOverflow {
                    field: "verifier attention end position",
                })?;
        if end_position > shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position: end_position,
                max_context: shape.max_context(),
            });
        }
        if end_position > cache.mapped_tokens() {
            return Err(BackendError::PositionOutOfBounds {
                position: end_position,
                max_context: cache.mapped_tokens(),
            });
        }
        let query = query.f32()?;
        let output = output.f32_mut()?;
        validate_span_attention_buffers(query, output, &cache, shape, positions)?;
        attention_prefill_span_rows(query, &cache, output, shape, start_position, positions)
    }

    fn embed_gather(
        &mut self,
        table: &Self::Buffer,
        row: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        let rows = row.u32()?;
        let row = embedding_row(rows, shape.rows())?;
        check_layout("embedding table", shape.layout()?, table.layout)?;
        let table = table.bytes()?;
        let output = output.f32_mut()?;
        exact_len("embedding output", shape.columns(), output.len())?;
        let row_bytes = shape.row_bytes()?;
        let start = row * row_bytes;
        let decoded = dequant(
            &table[start..start + row_bytes],
            shape.columns(),
            shape.format(),
        )?;
        output.copy_from_slice(&decoded);
        Ok(())
    }

    fn embed_gather_batch(
        &mut self,
        table: &Self::Buffer,
        rows: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let EmbedBatchBuffers {
            rows,
            table,
            output,
            row_bytes,
        } = embed_batch_buffers(table, rows, output, shape, tokens)?;
        embed_rows_batch(rows, table, output, shape, row_bytes, tokens)?;
        Ok(())
    }

    fn copy_f32_row(
        &mut self,
        input: &Self::Buffer,
        row: usize,
        columns: usize,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        let input = input.f32()?;
        let output = output.f32_mut()?;
        exact_len("copied f32 row output", columns, output.len())?;
        let start = row.checked_mul(columns).ok_or(BackendError::SizeOverflow {
            field: "copied f32 row offset",
        })?;
        let end = start
            .checked_add(columns)
            .ok_or(BackendError::SizeOverflow {
                field: "copied f32 row end",
            })?;
        if end > input.len() {
            return Err(BackendError::RowOutOfBounds {
                row,
                rows: input.len() / columns,
            });
        }
        output.copy_from_slice(&input[start..end]);
        Ok(())
    }

    fn write_f32_row(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        row: usize,
        columns: usize,
    ) -> Result<(), BackendError> {
        #[cfg(test)]
        self.record_attention_event(AttentionEvent::Scatter);
        let input = input.f32()?;
        exact_len("written f32 row input", columns, input.len())?;
        let output = output.f32_mut()?;
        let start = row.checked_mul(columns).ok_or(BackendError::SizeOverflow {
            field: "written f32 row offset",
        })?;
        let end = start
            .checked_add(columns)
            .ok_or(BackendError::SizeOverflow {
                field: "written f32 row end",
            })?;
        let rows = output.len() / columns;
        let target = output
            .get_mut(start..end)
            .ok_or(BackendError::RowOutOfBounds { row, rows })?;
        target.copy_from_slice(input);
        Ok(())
    }

    fn argmax(
        &mut self,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<(), BackendError> {
        let input = input.f32()?;
        let output = output.u32_mut()?;
        exact_len("argmax output", 1, output.len())?;
        let mut best_index = 0_u32;
        let mut best_value = f32::NEG_INFINITY;
        let mut found = false;
        for (index, value) in input.iter().copied().enumerate() {
            if !value.is_nan() && (!found || value > best_value) {
                best_index = u32::try_from(index)
                    .map_err(|error| BackendError::operation("convert argmax index", error))?;
                best_value = value;
                found = true;
            }
        }
        output[0] = best_index;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        #[cfg(test)]
        if let Some(probe) = self.completion_probe.as_mut() {
            probe.calls += 1;
            if probe.fail {
                return Err(BackendError::operation(
                    "synchronize CPU backend",
                    "injected completion failure",
                ));
            }
        }
        Ok(())
    }
}

fn gemm_rows(
    input: &[f32],
    output: &mut [f32],
    weights: &[u8],
    shape: QuantMatrix,
    row_bytes: usize,
) -> Result<(), BackendError> {
    output
        .par_chunks_exact_mut(shape.rows())
        .enumerate()
        .try_for_each(|(token, output_row)| {
            let input_row = &input[token * shape.columns()..(token + 1) * shape.columns()];
            for (row, destination) in output_row.iter_mut().enumerate() {
                let row_start = row * row_bytes;
                let decoded = dequant(
                    &weights[row_start..row_start + row_bytes],
                    shape.columns(),
                    shape.format(),
                )?;
                *destination = decoded
                    .iter()
                    .zip(input_row)
                    .fold(0.0_f32, |sum, (weight, value)| weight.mul_add(*value, sum));
            }
            Ok::<(), BackendError>(())
        })
}

fn prefill_gemm_buffers<'a>(
    weights: &CpuBuffer,
    input: &'a CpuBuffer,
    output: &'a mut CpuBuffer,
    shape: QuantMatrix,
    tokens: usize,
) -> Result<(&'a [f32], &'a mut [f32]), BackendError> {
    check_layout("prefill GEMM weights", shape.layout()?, weights.layout)?;
    let input = input.f32()?;
    let output = output.f32_mut()?;
    let input_elements = tokens
        .checked_mul(shape.columns())
        .ok_or(BackendError::SizeOverflow {
            field: "prefill GEMM input elements",
        })?;
    let output_elements = tokens
        .checked_mul(shape.rows())
        .ok_or(BackendError::SizeOverflow {
            field: "prefill GEMM output elements",
        })?;
    exact_len("prefill GEMM input", input_elements, input.len())?;
    exact_len("prefill GEMM output", output_elements, output.len())?;
    Ok((input, output))
}

fn gemv_buffers<'a>(
    weights: &CpuBuffer,
    input: &'a CpuBuffer,
    output: &'a mut CpuBuffer,
    shape: QuantMatrix,
) -> Result<(&'a [f32], &'a mut [f32]), BackendError> {
    check_layout("GEMV weights", shape.layout()?, weights.layout)?;
    let input = input.f32()?;
    let output = output.f32_mut()?;
    exact_len("GEMV input", shape.columns(), input.len())?;
    exact_len("GEMV output", shape.rows(), output.len())?;
    Ok((input, output))
}

struct RmsResidualBuffers<'a> {
    left: &'a [f32],
    right: &'a [f32],
    weight: &'a [f32],
    output: &'a mut [f32],
}

struct RmsResidualStoreBuffers<'a> {
    left: &'a [f32],
    right: &'a [f32],
    weight: &'a [f32],
    residual: &'a mut [f32],
    output: &'a mut [f32],
}

struct EmbedBatchBuffers<'a> {
    rows: &'a [u32],
    table: &'a [u8],
    output: &'a mut [f32],
    row_bytes: usize,
}

fn rms_residual_buffers<'a>(
    left: &'a CpuBuffer,
    right: &'a CpuBuffer,
    weight: &'a CpuBuffer,
    output: &'a mut CpuBuffer,
) -> Result<RmsResidualBuffers<'a>, BackendError> {
    Ok(RmsResidualBuffers {
        left: left.f32()?,
        right: right.f32()?,
        weight: weight.f32()?,
        output: output.f32_mut()?,
    })
}

fn validate_rms_residual_lengths(
    left: &[f32],
    right: &[f32],
    weight: &[f32],
    output: &[f32],
    shape: VectorShape,
) -> Result<(), BackendError> {
    let elements = shape.elements()?;
    exact_len("residual left", elements, left.len())?;
    exact_len("residual right", elements, right.len())?;
    exact_len("RMSNorm weight", shape.columns(), weight.len())?;
    exact_len("RMSNorm output", elements, output.len())?;
    Ok(())
}

fn rms_residual_store_buffers<'a>(
    left: &'a CpuBuffer,
    right: &'a CpuBuffer,
    weight: &'a CpuBuffer,
    residual: &'a mut CpuBuffer,
    output: &'a mut CpuBuffer,
) -> Result<RmsResidualStoreBuffers<'a>, BackendError> {
    Ok(RmsResidualStoreBuffers {
        left: left.f32()?,
        right: right.f32()?,
        weight: weight.f32()?,
        residual: residual.f32_mut()?,
        output: output.f32_mut()?,
    })
}

fn validate_rms_residual_store_lengths(
    left: &[f32],
    right: &[f32],
    weight: &[f32],
    residual: &[f32],
    output: &[f32],
    shape: VectorShape,
) -> Result<(), BackendError> {
    let elements = shape.elements()?;
    exact_len("residual left", elements, left.len())?;
    exact_len("residual right", elements, right.len())?;
    exact_len("RMSNorm weight", shape.columns(), weight.len())?;
    exact_len("stored residual", elements, residual.len())?;
    exact_len("RMSNorm output", elements, output.len())?;
    Ok(())
}

fn validate_kv_position(position: usize, shape: AttentionShape) -> Result<(), BackendError> {
    if position >= shape.max_context() {
        return Err(BackendError::PositionOutOfBounds {
            position,
            max_context: shape.max_context(),
        });
    }
    Ok(())
}

fn kv_append_buffers<'a>(
    key: &'a CpuBuffer,
    value: &'a CpuBuffer,
    shape: AttentionShape,
) -> Result<(&'a [f32], &'a [f32], usize), BackendError> {
    let key = key.f32()?;
    let value = value.f32()?;
    let projected = shape.projected_kv_elements()?;
    let cached = shape.cache_elements()?;
    exact_len("projected key", projected, key.len())?;
    exact_len("projected value", projected, value.len())?;
    Ok((key, value, cached))
}

fn validate_kv_chunk_position(
    start_position: usize,
    tokens: usize,
    shape: AttentionShape,
) -> Result<(), BackendError> {
    let end_position = start_position
        .checked_add(tokens)
        .ok_or(BackendError::SizeOverflow {
            field: "prefill KV end position",
        })?;
    if end_position > shape.max_context() {
        return Err(BackendError::PositionOutOfBounds {
            position: end_position,
            max_context: shape.max_context(),
        });
    }
    Ok(())
}

fn kv_append_chunk_buffers<'a>(
    key: &'a CpuBuffer,
    value: &'a CpuBuffer,
    shape: AttentionShape,
    tokens: usize,
) -> Result<(&'a [f32], &'a [f32], usize), BackendError> {
    let key = key.f32()?;
    let value = value.f32()?;
    let projected =
        shape
            .projected_kv_elements()?
            .checked_mul(tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "prefill projected KV elements",
            })?;
    exact_len("prefill projected key", projected, key.len())?;
    exact_len("prefill projected value", projected, value.len())?;
    let cached = shape.cache_elements()?;
    Ok((key, value, cached))
}

fn embed_batch_buffers<'a>(
    table: &'a CpuBuffer,
    rows: &'a CpuBuffer,
    output: &'a mut CpuBuffer,
    shape: QuantMatrix,
    tokens: usize,
) -> Result<EmbedBatchBuffers<'a>, BackendError> {
    let rows = rows.u32()?;
    exact_len("prefill embedding rows", tokens, rows.len())?;
    check_layout("prefill embedding table", shape.layout()?, table.layout)?;
    let table = table.bytes()?;
    let output = output.f32_mut()?;
    let output_elements =
        tokens
            .checked_mul(shape.columns())
            .ok_or(BackendError::SizeOverflow {
                field: "prefill embedding output elements",
            })?;
    exact_len("prefill embedding output", output_elements, output.len())?;
    let row_bytes = shape.row_bytes()?;
    Ok(EmbedBatchBuffers {
        rows,
        table,
        output,
        row_bytes,
    })
}

fn gemv_rows(
    input: &[f32],
    output: &mut [f32],
    weights: &[u8],
    shape: QuantMatrix,
    row_bytes: usize,
) -> Result<(), BackendError> {
    weights
        .par_chunks_exact(row_bytes)
        .zip(output.par_iter_mut())
        .try_for_each(|(row_data, destination)| {
            let decoded = dequant(row_data, shape.columns(), shape.format())?;
            *destination = decoded
                .iter()
                .zip(input)
                .fold(0.0_f32, |sum, (weight, value)| weight.mul_add(*value, sum));
            Ok::<(), BackendError>(())
        })
}

fn rms_norm_rows(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    shape: VectorShape,
    epsilon: f32,
) {
    for (input_row, output_row) in input
        .chunks_exact(shape.columns())
        .zip(output.chunks_exact_mut(shape.columns()))
    {
        let square_sum = input_row
            .iter()
            .fold(0.0_f32, |sum, value| value.mul_add(*value, sum));
        let inverse_rms = (square_sum / shape.columns() as f32 + epsilon)
            .sqrt()
            .recip();
        for ((destination, value), scale) in output_row.iter_mut().zip(input_row).zip(weight) {
            *destination = *value * *scale * inverse_rms;
        }
    }
}

fn rms_residual_rows(
    left: &[f32],
    right: &[f32],
    weight: &[f32],
    output: &mut [f32],
    shape: VectorShape,
    epsilon: f32,
) {
    for row in 0..shape.rows() {
        let start = row * shape.columns();
        let end = start + shape.columns();
        let square_sum =
            left[start..end]
                .iter()
                .zip(&right[start..end])
                .fold(0.0_f32, |sum, (left, right)| {
                    let value = left + right;
                    value.mul_add(value, sum)
                });
        let inverse_rms = (square_sum / shape.columns() as f32 + epsilon)
            .sqrt()
            .recip();
        for column in 0..shape.columns() {
            output[start + column] =
                (left[start + column] + right[start + column]) * weight[column] * inverse_rms;
        }
    }
}

fn rms_residual_store_rows(
    left: &[f32],
    right: &[f32],
    weight: &[f32],
    residual: &mut [f32],
    output: &mut [f32],
    shape: VectorShape,
    epsilon: f32,
) {
    for row in 0..shape.rows() {
        let start = row * shape.columns();
        let end = start + shape.columns();
        let mut square_sum = 0.0_f32;
        for column in start..end {
            let value = left[column] + right[column];
            residual[column] = value;
            square_sum = value.mul_add(value, square_sum);
        }
        let inverse_rms = (square_sum / shape.columns() as f32 + epsilon)
            .sqrt()
            .recip();
        for column in start..end {
            output[column] = residual[column] * weight[column - start] * inverse_rms;
        }
    }
}

fn build_rope_inverse_frequencies(
    head_dim: usize,
    theta: f32,
    frequency_factors: Option<&[f32]>,
) -> Result<Vec<f64>, BackendError> {
    validate_positive("RoPE theta", theta)?;
    let half = head_dim / 2;
    if let Some(factors) = frequency_factors {
        exact_len("RoPE frequency factors", half, factors.len())?;
        for &factor in factors {
            validate_positive("RoPE frequency factor", factor)?;
        }
    }
    let mut inverse = Vec::new();
    inverse
        .try_reserve_exact(half)
        .map_err(|error| BackendError::operation("allocate RoPE inverse frequencies", error))?;
    for pair in 0..half {
        let factor = frequency_factors
            .map(|factors| f64::from(factors[pair]))
            .unwrap_or(1.0);
        inverse.push(f64::from(theta).powf(-2.0 * pair as f64 / head_dim as f64) / factor);
    }
    Ok(inverse)
}

fn rope_values(
    values: &mut [f32],
    position: usize,
    shape: RopeShape,
    half: usize,
    frequencies: &[f64],
    pairing: RopePairing,
) -> Result<(), BackendError> {
    for token in 0..shape.tokens() {
        for head in 0..shape.heads() {
            let base = (token * shape.heads() + head) * shape.head_dim();
            for (pair, frequency) in frequencies.iter().take(half).copied().enumerate() {
                let token_position =
                    position
                        .checked_add(token)
                        .ok_or(BackendError::SizeOverflow {
                            field: "RoPE position",
                        })?;
                let angle = token_position as f64 * frequency;
                let (sine, cosine) = angle.sin_cos();
                let (first_index, second_index) = match pairing {
                    RopePairing::HalfSplit => (base + pair, base + pair + half),
                    RopePairing::Adjacent => (base + pair * 2, base + pair * 2 + 1),
                };
                let first = f64::from(values[first_index]);
                let second = f64::from(values[second_index]);
                values[first_index] = (first * cosine - second * sine) as f32;
                values[second_index] = (first * sine + second * cosine) as f32;
            }
        }
    }
    Ok(())
}

fn append_kv_by_storage(
    key: &[f32],
    value: &[f32],
    key_cache: &mut CpuBuffer,
    value_cache: &mut CpuBuffer,
    cached: usize,
    shape: AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    match key_cache.layout.storage() {
        BufferStorage::F32 => {
            append_kv_f32(key, value, key_cache, value_cache, cached, shape, position)
        }
        BufferStorage::F16 => {
            append_kv_f16(key, value, key_cache, value_cache, cached, shape, position)
        }
        BufferStorage::Q8Kv => {
            append_kv_q8(key, value, key_cache, value_cache, cached, shape, position)
        }
        storage => Err(storage_error("write KV cache", storage)),
    }
}

fn append_kv_f32(
    key: &[f32],
    value: &[f32],
    key_cache: &mut CpuBuffer,
    value_cache: &mut CpuBuffer,
    cached: usize,
    shape: AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    let key_cache = key_cache.f32_mut()?;
    let value_cache = value_cache.f32_mut()?;
    exact_len("key cache", cached, key_cache.len())?;
    exact_len("value cache", cached, value_cache.len())?;
    for head in 0..shape.n_head_kv() {
        let source = head * shape.head_dim();
        let target = (head * shape.max_context() + position) * shape.head_dim();
        key_cache[target..target + shape.head_dim()]
            .copy_from_slice(&key[source..source + shape.head_dim()]);
        value_cache[target..target + shape.head_dim()]
            .copy_from_slice(&value[source..source + shape.head_dim()]);
    }
    Ok(())
}

fn append_kv_f16(
    key: &[f32],
    value: &[f32],
    key_cache: &mut CpuBuffer,
    value_cache: &mut CpuBuffer,
    cached: usize,
    shape: AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    let key_cache = key_cache.f16_mut()?;
    let value_cache = value_cache.f16_mut()?;
    exact_len("key cache", cached, key_cache.len())?;
    exact_len("value cache", cached, value_cache.len())?;
    for head in 0..shape.n_head_kv() {
        let source = head * shape.head_dim();
        let target = (head * shape.max_context() + position) * shape.head_dim();
        for dimension in 0..shape.head_dim() {
            key_cache[target + dimension] = f16::from_f32(key[source + dimension]);
            value_cache[target + dimension] = f16::from_f32(value[source + dimension]);
        }
    }
    Ok(())
}

fn append_kv_q8(
    key: &[f32],
    value: &[f32],
    key_cache: &mut CpuBuffer,
    value_cache: &mut CpuBuffer,
    cached: usize,
    shape: AttentionShape,
    position: usize,
) -> Result<(), BackendError> {
    let key_cache = key_cache.bytes_mut()?;
    let value_cache = value_cache.bytes_mut()?;
    let bytes = cached / 32 * 34;
    exact_len("Q8 key cache", bytes, key_cache.len())?;
    exact_len("Q8 value cache", bytes, value_cache.len())?;
    for head in 0..shape.n_head_kv() {
        let source = head * shape.head_dim();
        let target = (head * shape.max_context() + position) * shape.head_dim();
        q8_kv_store(key_cache, target, &key[source..source + shape.head_dim()]);
        q8_kv_store(
            value_cache,
            target,
            &value[source..source + shape.head_dim()],
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct KvChunkSpec {
    cached: usize,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
}

fn append_kv_chunk_f32(
    key: &[f32],
    value: &[f32],
    key_cache: &mut CpuBuffer,
    value_cache: &mut CpuBuffer,
    spec: KvChunkSpec,
) -> Result<(), BackendError> {
    let KvChunkSpec {
        cached,
        shape,
        start_position,
        tokens,
    } = spec;
    let key_cache = key_cache.f32_mut()?;
    let value_cache = value_cache.f32_mut()?;
    exact_len("key cache", cached, key_cache.len())?;
    exact_len("value cache", cached, value_cache.len())?;
    for token in 0..tokens {
        for head in 0..shape.n_head_kv() {
            let source = (token * shape.n_head_kv() + head) * shape.head_dim();
            let target = (head * shape.max_context() + start_position + token) * shape.head_dim();
            key_cache[target..target + shape.head_dim()]
                .copy_from_slice(&key[source..source + shape.head_dim()]);
            value_cache[target..target + shape.head_dim()]
                .copy_from_slice(&value[source..source + shape.head_dim()]);
        }
    }
    Ok(())
}

fn append_kv_chunk_f16(
    key: &[f32],
    value: &[f32],
    key_cache: &mut CpuBuffer,
    value_cache: &mut CpuBuffer,
    spec: KvChunkSpec,
) -> Result<(), BackendError> {
    let KvChunkSpec {
        cached,
        shape,
        start_position,
        tokens,
    } = spec;
    let key_cache = key_cache.f16_mut()?;
    let value_cache = value_cache.f16_mut()?;
    exact_len("key cache", cached, key_cache.len())?;
    exact_len("value cache", cached, value_cache.len())?;
    for token in 0..tokens {
        for head in 0..shape.n_head_kv() {
            let source = (token * shape.n_head_kv() + head) * shape.head_dim();
            let target = (head * shape.max_context() + start_position + token) * shape.head_dim();
            for dimension in 0..shape.head_dim() {
                key_cache[target + dimension] = f16::from_f32(key[source + dimension]);
                value_cache[target + dimension] = f16::from_f32(value[source + dimension]);
            }
        }
    }
    Ok(())
}

fn append_kv_chunk_q8(
    key: &[f32],
    value: &[f32],
    key_cache: &mut CpuBuffer,
    value_cache: &mut CpuBuffer,
    spec: KvChunkSpec,
) -> Result<(), BackendError> {
    let KvChunkSpec {
        cached,
        shape,
        start_position,
        tokens,
    } = spec;
    let key_cache = key_cache.bytes_mut()?;
    let value_cache = value_cache.bytes_mut()?;
    let bytes = cached / 32 * 34;
    exact_len("Q8 key cache", bytes, key_cache.len())?;
    exact_len("Q8 value cache", bytes, value_cache.len())?;
    for token in 0..tokens {
        for head in 0..shape.n_head_kv() {
            let source = (token * shape.n_head_kv() + head) * shape.head_dim();
            let target = (head * shape.max_context() + start_position + token) * shape.head_dim();
            q8_kv_store(key_cache, target, &key[source..source + shape.head_dim()]);
            q8_kv_store(
                value_cache,
                target,
                &value[source..source + shape.head_dim()],
            );
        }
    }
    Ok(())
}

fn validate_attention_buffers(
    query: &[f32],
    output: &[f32],
    key_cache: &CpuBuffer,
    value_cache: &CpuBuffer,
    shape: AttentionShape,
) -> Result<(), BackendError> {
    let query_elements = shape.query_elements()?;
    let cache_elements = shape.cache_elements()?;
    exact_len("attention query", query_elements, query.len())?;
    exact_len(
        "attention key cache",
        cache_elements,
        key_cache.layout.elements(),
    )?;
    exact_len(
        "attention value cache",
        cache_elements,
        value_cache.layout.elements(),
    )?;
    exact_len("attention output", query_elements, output.len())
}

fn validate_prefill_attention_buffers(
    query: &[f32],
    output: &[f32],
    key_cache: &CpuBuffer,
    value_cache: &CpuBuffer,
    shape: AttentionShape,
    tokens: usize,
) -> Result<(), BackendError> {
    let block_elements =
        tokens
            .checked_mul(shape.query_elements()?)
            .ok_or(BackendError::SizeOverflow {
                field: "prefill attention block elements",
            })?;
    exact_len("prefill attention query", block_elements, query.len())?;
    exact_len("prefill attention output", block_elements, output.len())?;
    let cache_elements = shape.cache_elements()?;
    exact_len(
        "prefill attention key cache",
        cache_elements,
        key_cache.layout.elements(),
    )?;
    exact_len(
        "prefill attention value cache",
        cache_elements,
        value_cache.layout.elements(),
    )
}

fn physical_attention_shape(
    shape: AttentionShape,
    capacity_tokens: usize,
) -> Result<AttentionShape, BackendError> {
    AttentionShape::new(
        shape.n_head(),
        shape.n_head_kv(),
        shape.head_dim(),
        capacity_tokens,
    )
}

fn prepare_kv_chunk_span(
    target: &KvWriteSpan<'_, CpuBuffer>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Result<(AttentionShape, usize), BackendError> {
    validate_kv_chunk_position(start_position, tokens, shape)?;
    let local_start = target.local_range(start_position, tokens)?.start;
    let physical_shape = physical_attention_shape(shape, target.capacity_token_count())?;
    Ok((physical_shape, local_start))
}

fn validate_kv_span_buffers(
    key_cache: &CpuBuffer,
    value_cache: &CpuBuffer,
    shape: AttentionShape,
) -> Result<(), BackendError> {
    if key_cache.layout.storage() != value_cache.layout.storage() {
        return Err(BackendError::operation(
            "write KV span",
            "key and value storage differ",
        ));
    }
    let expected = expected_kv_span_layout(key_cache.layout.storage(), shape)?;
    check_layout("KV span key", expected, key_cache.layout)?;
    check_layout("KV span value", expected, value_cache.layout)
}

fn expected_kv_span_layout(
    storage: BufferStorage,
    shape: AttentionShape,
) -> Result<BufferLayout, BackendError> {
    let elements = shape.cache_elements()?;
    match storage {
        BufferStorage::F32 => BufferLayout::f32(elements),
        BufferStorage::F16 => BufferLayout::f16(elements),
        BufferStorage::Q8Kv if !shape.head_dim().is_multiple_of(32) => {
            Err(BackendError::NotDivisible {
                field: "KV head_dim",
                value: shape.head_dim(),
                divisor: 32,
            })
        }
        BufferStorage::Q8Kv => BufferLayout::q8_kv(elements),
        storage => Err(storage_error("write KV span", storage)),
    }
}

fn validate_span_attention_buffers(
    query: &[f32],
    output: &[f32],
    cache: &KvReadView<'_, CpuBuffer>,
    shape: AttentionShape,
    tokens: usize,
) -> Result<(), BackendError> {
    let query_elements =
        tokens
            .checked_mul(shape.query_elements()?)
            .ok_or(BackendError::SizeOverflow {
                field: "span attention query elements",
            })?;
    exact_len("span attention query", query_elements, query.len())?;
    exact_len("span attention output", query_elements, output.len())?;
    validate_span_cache_buffers(cache, shape)?;
    Ok(())
}

fn validate_span_cache_buffers(
    cache: &KvReadView<'_, CpuBuffer>,
    shape: AttentionShape,
) -> Result<(), BackendError> {
    let first = cache
        .spans()
        .first()
        .ok_or_else(|| BackendError::operation("read KV spans", "the view has no spans"))?;
    let storage = first.key().layout.storage();
    validate_kv_span_buffers(
        first.key(),
        first.value(),
        physical_attention_shape(shape, first.capacity_token_count())?,
    )?;
    for span in cache.spans().iter().skip(1) {
        if span.key().layout.storage() != storage {
            return Err(BackendError::operation(
                "read KV spans",
                "span storage differs",
            ));
        }
        validate_kv_span_buffers(
            span.key(),
            span.value(),
            physical_attention_shape(shape, span.capacity_token_count())?,
        )?;
    }
    Ok(())
}

fn cache_value(
    buffer: &CpuBuffer,
    index: usize,
    operation: &'static str,
) -> Result<f32, BackendError> {
    match &buffer.storage {
        CpuStorage::F32(values) => Ok(values[index]),
        CpuStorage::F16(values) => Ok(values[index].to_f32()),
        CpuStorage::Bytes(values) if buffer.layout.storage() == BufferStorage::Q8Kv => {
            Ok(q8_kv_load(values, index))
        }
        _ => Err(storage_error(operation, buffer.layout.storage())),
    }
}

fn span_cache_value(
    cache: &KvReadView<'_, CpuBuffer>,
    kv_head: usize,
    position: usize,
    dimension: usize,
    shape: AttentionShape,
    key: bool,
) -> Result<f32, BackendError> {
    let (span, local_position) =
        cache
            .span_for_position(position)
            .ok_or(BackendError::PositionOutOfBounds {
                position,
                max_context: cache.mapped_tokens(),
            })?;
    let row = kv_head
        .checked_mul(span.capacity_token_count())
        .and_then(|value| value.checked_add(local_position))
        .ok_or(BackendError::SizeOverflow {
            field: "KV span row index",
        })?;
    let index = row
        .checked_mul(shape.head_dim())
        .and_then(|value| value.checked_add(dimension))
        .ok_or(BackendError::SizeOverflow {
            field: "KV span element index",
        })?;
    let buffer = if key { span.key() } else { span.value() };
    cache_value(
        buffer,
        index,
        if key {
            "read span attention key cache"
        } else {
            "read span attention value cache"
        },
    )
}

fn attention_decode_span_rows(
    query: &[f32],
    cache: &KvReadView<'_, CpuBuffer>,
    output: &mut [f32],
    shape: AttentionShape,
    context_length: usize,
) -> Result<(), BackendError> {
    let group_size = shape.n_head() / shape.n_head_kv();
    let scale = (shape.head_dim() as f32).sqrt().recip();
    let mut numerator = vec![0.0_f32; shape.head_dim()];
    for query_head in 0..shape.n_head() {
        numerator.fill(0.0);
        let query_base = query_head * shape.head_dim();
        let kv_head = query_head / group_size;
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        for position in 0..context_length {
            let mut dot = 0.0_f32;
            for dimension in 0..shape.head_dim() {
                dot = query[query_base + dimension].mul_add(
                    span_cache_value(cache, kv_head, position, dimension, shape, true)?,
                    dot,
                );
            }
            let score = dot * scale;
            let next_max = running_max.max(score);
            let previous_scale = if running_sum == 0.0 {
                0.0
            } else {
                (running_max - next_max).exp()
            };
            let score_scale = (score - next_max).exp();
            running_sum = running_sum * previous_scale + score_scale;
            for (dimension, numerator) in numerator.iter_mut().enumerate() {
                *numerator = *numerator * previous_scale
                    + score_scale
                        * span_cache_value(cache, kv_head, position, dimension, shape, false)?;
            }
            running_max = next_max;
        }
        for (destination, numerator) in output[query_base..query_base + shape.head_dim()]
            .iter_mut()
            .zip(&numerator)
        {
            *destination = *numerator / running_sum;
        }
    }
    Ok(())
}

fn attention_prefill_span_rows(
    query: &[f32],
    cache: &KvReadView<'_, CpuBuffer>,
    output: &mut [f32],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Result<(), BackendError> {
    let group_size = shape.n_head() / shape.n_head_kv();
    let scale = (shape.head_dim() as f32).sqrt().recip();
    let mut numerator = vec![0.0_f32; shape.head_dim()];
    for token in 0..tokens {
        let context_length = start_position + token + 1;
        for query_head in 0..shape.n_head() {
            numerator.fill(0.0);
            let query_base = (token * shape.n_head() + query_head) * shape.head_dim();
            let kv_head = query_head / group_size;
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0_f32;
            for position in 0..context_length {
                let mut dot = 0.0_f32;
                for dimension in 0..shape.head_dim() {
                    dot = query[query_base + dimension].mul_add(
                        span_cache_value(cache, kv_head, position, dimension, shape, true)?,
                        dot,
                    );
                }
                let score = dot * scale;
                let next_max = running_max.max(score);
                let previous_scale = if running_sum == 0.0 {
                    0.0
                } else {
                    (running_max - next_max).exp()
                };
                let score_scale = (score - next_max).exp();
                running_sum = running_sum * previous_scale + score_scale;
                for (dimension, numerator) in numerator.iter_mut().enumerate() {
                    *numerator = *numerator * previous_scale
                        + score_scale
                            * span_cache_value(cache, kv_head, position, dimension, shape, false)?;
                }
                running_max = next_max;
            }
            for (destination, numerator) in output[query_base..query_base + shape.head_dim()]
                .iter_mut()
                .zip(&numerator)
            {
                *destination = *numerator / running_sum;
            }
        }
    }
    Ok(())
}

fn attention_decode_rows(
    query: &[f32],
    key_cache: &CpuBuffer,
    value_cache: &CpuBuffer,
    output: &mut [f32],
    shape: AttentionShape,
    context_length: usize,
) -> Result<(), BackendError> {
    let group_size = shape.n_head() / shape.n_head_kv();
    let scale = (shape.head_dim() as f32).sqrt().recip();
    let mut numerator = vec![0.0_f32; shape.head_dim()];
    for query_head in 0..shape.n_head() {
        numerator.fill(0.0);
        let query_base = query_head * shape.head_dim();
        let kv_head = query_head / group_size;
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        for position in 0..context_length {
            let cache_base = (kv_head * shape.max_context() + position) * shape.head_dim();
            let mut dot = 0.0_f32;
            for dimension in 0..shape.head_dim() {
                dot = query[query_base + dimension].mul_add(
                    cache_value(
                        key_cache,
                        cache_base + dimension,
                        "read attention key cache",
                    )?,
                    dot,
                );
            }
            let score = dot * scale;
            let next_max = running_max.max(score);
            let previous_scale = if running_sum == 0.0 {
                0.0
            } else {
                (running_max - next_max).exp()
            };
            let score_scale = (score - next_max).exp();
            running_sum = running_sum * previous_scale + score_scale;
            for (dimension, numerator) in numerator.iter_mut().enumerate() {
                *numerator = *numerator * previous_scale
                    + score_scale
                        * cache_value(
                            value_cache,
                            cache_base + dimension,
                            "read attention value cache",
                        )?;
            }
            running_max = next_max;
        }
        for (destination, numerator) in output[query_base..query_base + shape.head_dim()]
            .iter_mut()
            .zip(&numerator)
        {
            *destination = *numerator / running_sum;
        }
    }
    Ok(())
}

fn attention_prefill_rows(
    query: &[f32],
    key_cache: &CpuBuffer,
    value_cache: &CpuBuffer,
    output: &mut [f32],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Result<(), BackendError> {
    let group_size = shape.n_head() / shape.n_head_kv();
    let scale = (shape.head_dim() as f32).sqrt().recip();
    let mut numerator = vec![0.0_f32; shape.head_dim()];
    for token in 0..tokens {
        let context_length = start_position + token + 1;
        for query_head in 0..shape.n_head() {
            numerator.fill(0.0);
            let query_base = (token * shape.n_head() + query_head) * shape.head_dim();
            let kv_head = query_head / group_size;
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0_f32;
            for position in 0..context_length {
                let cache_base = (kv_head * shape.max_context() + position) * shape.head_dim();
                let mut dot = 0.0_f32;
                for dimension in 0..shape.head_dim() {
                    dot = query[query_base + dimension].mul_add(
                        cache_value(
                            key_cache,
                            cache_base + dimension,
                            "read prefill attention key cache",
                        )?,
                        dot,
                    );
                }
                let score = dot * scale;
                let next_max = running_max.max(score);
                let previous_scale = if running_sum == 0.0 {
                    0.0
                } else {
                    (running_max - next_max).exp()
                };
                let score_scale = (score - next_max).exp();
                running_sum = running_sum * previous_scale + score_scale;
                for (dimension, numerator) in numerator.iter_mut().enumerate() {
                    *numerator = *numerator * previous_scale
                        + score_scale
                            * cache_value(
                                value_cache,
                                cache_base + dimension,
                                "read prefill attention value cache",
                            )?;
                }
                running_max = next_max;
            }
            for (destination, numerator) in output[query_base..query_base + shape.head_dim()]
                .iter_mut()
                .zip(&numerator)
            {
                *destination = *numerator / running_sum;
            }
        }
    }
    Ok(())
}

fn embedding_row(rows: &[u32], row_count: usize) -> Result<usize, BackendError> {
    exact_len("embedding row", 1, rows.len())?;
    let row = usize::try_from(rows[0]).map_err(|_| BackendError::SizeOverflow {
        field: "embedding row",
    })?;
    if row >= row_count {
        return Err(BackendError::RowOutOfBounds {
            row,
            rows: row_count,
        });
    }
    Ok(row)
}

fn embed_rows_batch(
    rows: &[u32],
    table: &[u8],
    output: &mut [f32],
    shape: QuantMatrix,
    row_bytes: usize,
    tokens: usize,
) -> Result<(), BackendError> {
    for (token, row) in rows.iter().copied().enumerate().take(tokens) {
        let row = usize::try_from(row).map_err(|_| BackendError::SizeOverflow {
            field: "prefill embedding row",
        })?;
        if row >= shape.rows() {
            return Err(BackendError::RowOutOfBounds {
                row,
                rows: shape.rows(),
            });
        }
        let start = row * row_bytes;
        let decoded = dequant(
            &table[start..start + row_bytes],
            shape.columns(),
            shape.format(),
        )?;
        let output_start = token * shape.columns();
        output[output_start..output_start + shape.columns()].copy_from_slice(&decoded);
    }
    Ok(())
}

fn zeroed<T: Default + Clone>(len: usize, operation: &'static str) -> Result<Vec<T>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| BackendError::operation(operation, error))?;
    values.resize(len, T::default());
    Ok(values)
}

fn clone_values<T: Copy>(values: &[T], operation: &'static str) -> Result<Vec<T>, BackendError> {
    let mut clone = Vec::new();
    clone
        .try_reserve_exact(values.len())
        .map_err(|error| BackendError::operation(operation, error))?;
    clone.extend_from_slice(values);
    Ok(clone)
}

fn clone_cpu_storage(storage: &CpuStorage) -> Result<CpuStorage, BackendError> {
    match storage {
        CpuStorage::Bytes(values) => Ok(CpuStorage::Bytes(clone_values(
            values,
            "clone quantized bytes",
        )?)),
        CpuStorage::F16(values) => Ok(CpuStorage::F16(clone_values(values, "clone f16")?)),
        CpuStorage::F32(values) => Ok(CpuStorage::F32(clone_values(values, "clone f32")?)),
        CpuStorage::U32(values) => Ok(CpuStorage::U32(clone_values(values, "clone u32")?)),
    }
}

fn encode_values<T, const N: usize, F: FnMut(&T) -> [u8; N]>(
    values: &[T],
    bytes: usize,
    mut encode: F,
) -> Result<Vec<u8>, BackendError> {
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(bytes)
        .map_err(|error| BackendError::operation("encode CPU buffer", error))?;
    for value in values {
        encoded.extend_from_slice(&encode(value));
    }
    Ok(encoded)
}

fn parse_cpu_storage(layout: BufferLayout, bytes: &[u8]) -> Result<CpuStorage, BackendError> {
    match layout.storage() {
        BufferStorage::F16 => Ok(CpuStorage::F16(parse_f16(bytes)?)),
        BufferStorage::F32 => Ok(CpuStorage::F32(parse_f32(bytes)?)),
        BufferStorage::U32 => Ok(CpuStorage::U32(parse_u32(bytes)?)),
        BufferStorage::Q8Kv | BufferStorage::Q4K | BufferStorage::Q6K => Ok(CpuStorage::Bytes(
            clone_values(bytes, "copy quantized bytes")?,
        )),
    }
}

fn encode_cpu_storage(storage: &CpuStorage, bytes: usize) -> Result<Vec<u8>, BackendError> {
    match storage {
        CpuStorage::Bytes(values) => clone_values(values, "download quantized bytes"),
        CpuStorage::F16(values) => {
            encode_values(values, bytes, |value| value.to_bits().to_le_bytes())
        }
        CpuStorage::F32(values) => {
            encode_values(values, bytes, |value| value.to_bits().to_le_bytes())
        }
        CpuStorage::U32(values) => encode_values(values, bytes, |value| value.to_le_bytes()),
    }
}

fn parse_f16(bytes: &[u8]) -> Result<Vec<f16>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 2)
        .map_err(|error| BackendError::operation("parse f16", error))?;
    values.extend(
        bytes
            .chunks_exact(2)
            .map(|value| f16::from_bits(u16::from_le_bytes([value[0], value[1]]))),
    );
    Ok(values)
}

fn parse_f32(bytes: &[u8]) -> Result<Vec<f32>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 4)
        .map_err(|error| BackendError::operation("parse f32", error))?;
    values.extend(
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]])),
    );
    Ok(values)
}

fn parse_u32(bytes: &[u8]) -> Result<Vec<u32>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 4)
        .map_err(|error| BackendError::operation("parse u32", error))?;
    values.extend(
        bytes
            .chunks_exact(4)
            .map(|value| u32::from_le_bytes([value[0], value[1], value[2], value[3]])),
    );
    Ok(values)
}

fn dequant(bytes: &[u8], elements: usize, format: QuantFormat) -> Result<Vec<f32>, BackendError> {
    match format {
        QuantFormat::Q4K => ref_dequant::q4_k::dequant_row(bytes, elements),
        QuantFormat::Q6K => ref_dequant::q6_k::dequant_row(bytes, elements),
    }
    .map_err(|error| BackendError::operation("dequantize matrix row", error))
}

fn emulate_q8_1(input: &[f32]) -> Vec<f32> {
    input
        .chunks(32)
        .flat_map(|block| {
            let maximum = block
                .iter()
                .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
            let scale = maximum / 127.0;
            let stored_scale = f16::from_f32(scale).to_f32();
            block.iter().map(move |value| {
                let quantized = if maximum == 0.0 {
                    0.0
                } else {
                    (*value / scale).round_ties_even()
                };
                quantized * stored_scale
            })
        })
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

fn allocate_cpu_storage(layout: BufferLayout) -> Result<CpuStorage, BackendError> {
    match layout.storage() {
        BufferStorage::F16 => Ok(CpuStorage::F16(zeroed(layout.elements(), "allocate f16")?)),
        BufferStorage::F32 => Ok(CpuStorage::F32(zeroed(layout.elements(), "allocate f32")?)),
        BufferStorage::U32 => Ok(CpuStorage::U32(zeroed(layout.elements(), "allocate u32")?)),
        BufferStorage::Q8Kv | BufferStorage::Q4K | BufferStorage::Q6K => Ok(CpuStorage::Bytes(
            zeroed(layout.bytes(), "allocate quantized bytes")?,
        )),
    }
}

fn q8_kv_store(cache: &mut [u8], logical_start: usize, values: &[f32]) {
    for (block_offset, block) in values.chunks_exact(32).enumerate() {
        let maximum = block
            .iter()
            .fold(0.0_f32, |value, next| value.max(next.abs()));
        let scale = if maximum == 0.0 { 0.0 } else { maximum / 127.0 };
        let block_index = logical_start / 32 + block_offset;
        let byte_start = block_index * 34;
        cache[byte_start..byte_start + 2]
            .copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        let stored_scale = f16::from_f32(scale).to_f32();
        for (code, value) in cache[byte_start + 2..byte_start + 34].iter_mut().zip(block) {
            let quantized = if stored_scale == 0.0 {
                0
            } else {
                (*value / stored_scale).round().clamp(-127.0, 127.0) as i8
            };
            *code = quantized as u8;
        }
    }
}

fn q8_kv_load(cache: &[u8], logical_index: usize) -> f32 {
    let byte_start = (logical_index / 32) * 34;
    let scale = f16::from_bits(u16::from_le_bytes([
        cache[byte_start],
        cache[byte_start + 1],
    ]))
    .to_f32();
    let code = cache[byte_start + 2 + logical_index % 32] as i8;
    f32::from(code) * scale
}

#[cfg(test)]
mod tests {
    use crate::KvReadSpan;

    use super::*;

    #[test]
    fn scalar_backend_declares_single_row_decode() {
        assert_eq!(CpuBackend::new().max_batch_size().get(), 1);
    }

    #[test]
    fn classified_restore_and_clone_preserve_peaks_and_ownership() {
        let mut backend = CpuBackend::new();
        let layout = BufferLayout::f32(4).unwrap();
        let buffer = backend
            .allocate_classified(layout, MemoryClass::KvCache)
            .unwrap();
        let snapshot = backend.download_buffer(&buffer).unwrap();
        let restored = backend
            .restore_buffer_classified(&snapshot, MemoryClass::KvCache)
            .unwrap();
        let cloned = backend.clone_buffer(&buffer).unwrap();
        assert_ne!(buffer.allocation.identity(), cloned.allocation.identity());
        let live = backend.memory_accounting();
        assert_eq!(live.class(MemoryClass::KvCache).live_bytes, 48);
        assert_eq!(live.class(MemoryClass::ContractBuffer).peak_live_bytes, 0);
        assert_eq!(live.class(MemoryClass::ModelWeight).peak_live_bytes, 0);
        drop((buffer, restored, cloned));
        let empty = backend.memory_accounting();
        assert_eq!(empty.live_bytes, 0);
        assert_eq!(empty.class(MemoryClass::KvCache).frees, 3);
    }

    #[test]
    fn memory_accounting_tracks_cpu_buffer_lifetime() {
        let mut backend = CpuBackend::new();
        let buffer = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
        let live = backend.memory_accounting();
        assert_eq!(live.live_bytes, 16);
        assert_eq!(live.live_allocations, 1);
        drop(buffer);
        let empty = backend.memory_accounting();
        assert_eq!(empty.live_bytes, 0);
        assert_eq!(empty.frees, 1);
    }

    #[test]
    fn memory_budget_rejects_cpu_buffer_before_storage_allocation() {
        let mut backend = CpuBackend::with_memory_budget(MemoryBudget::limited(16).unwrap());
        let error = backend.allocate(BufferLayout::f32(5).unwrap()).unwrap_err();
        assert!(matches!(
            error,
            BackendError::Memory(crate::backend::MemoryError::BudgetExceeded { .. })
        ));
        let memory = backend.memory_accounting();
        assert_eq!(memory.live_bytes, 0);
        assert_eq!(memory.reserved_bytes, 0);
    }

    #[test]
    fn memory_tracker_injection_shares_parent_and_rejects_live_swap() {
        let root = crate::MemoryTrackerRoot::new(MemoryBudget::limited(64).unwrap());
        let tracker = MemoryTracker::child(MemoryBudget::limited(64).unwrap(), root.clone());
        let replacement = MemoryTracker::child(MemoryBudget::limited(64).unwrap(), root);
        let mut backend = CpuBackend::new();
        backend.set_memory_tracker(tracker).unwrap();
        let buffer = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
        assert_eq!(backend.memory_accounting().live_bytes, 16);
        assert_eq!(backend.memory_tracker_root().owned_bytes(), 16);
        assert!(matches!(
            backend.set_memory_tracker(replacement.clone()),
            Err(BackendError::Memory(MemoryError::TrackerInUse {
                owned: 16,
                reserved: 0,
            }))
        ));
        drop(buffer);
        backend.set_memory_tracker(replacement).unwrap();
    }

    #[test]
    fn host_staged_rope_retains_cpu_table_charge() {
        let staging = HostStaging::new(MemoryBudget::limited(64).unwrap());
        let mut backend = CpuBackend::new();
        backend
            .configure_rope_with_host_staging(16, 10_000.0, None, RopePairing::HalfSplit, &staging)
            .unwrap();
        assert_eq!(backend.rope_inverse_frequencies.len(), 8);
        assert_eq!(staging.snapshot().live_bytes, 64);
        drop(backend);
        assert_eq!(staging.snapshot().live_bytes, 0);
    }

    #[test]
    fn host_staged_rope_denial_keeps_cpu_table_unchanged() {
        let staging = HostStaging::new(MemoryBudget::limited(63).unwrap());
        let mut backend = CpuBackend::new();
        let error = backend
            .configure_rope_with_host_staging(16, 10_000.0, None, RopePairing::HalfSplit, &staging)
            .unwrap_err();
        assert!(matches!(
            error,
            BackendError::Memory(MemoryError::BudgetExceeded { .. })
        ));
        assert!(backend.rope_inverse_frequencies.is_empty());
        assert_eq!(staging.snapshot().live_bytes, 0);
        assert_eq!(staging.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn rms_norm_and_argmax_use_the_contract() {
        let mut backend = CpuBackend::new();
        let input = backend
            .upload(
                BufferLayout::f32(4).unwrap(),
                &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            )
            .unwrap();
        let weight = backend
            .upload(BufferLayout::f32(2).unwrap(), &f32_bytes(&[1.0, 0.5]))
            .unwrap();
        let mut normalized = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
        backend
            .rms_norm(
                &input,
                &weight,
                &mut normalized,
                VectorShape::new(2, 2).unwrap(),
                1e-6,
            )
            .unwrap();
        let mut token = backend.allocate(BufferLayout::u32(1).unwrap()).unwrap();
        backend.argmax(&normalized, &mut token).unwrap();
        let mut observed = [0];
        backend.read_u32(&token, &mut observed).unwrap();
        assert_eq!(observed, [2]);
    }

    #[test]
    fn kv_append_uses_head_major_cache_rows() {
        let mut backend = CpuBackend::new();
        let key = backend
            .upload(
                BufferLayout::f32(4).unwrap(),
                &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            )
            .unwrap();
        let value = backend
            .upload(
                BufferLayout::f32(4).unwrap(),
                &f32_bytes(&[5.0, 6.0, 7.0, 8.0]),
            )
            .unwrap();
        let shape = AttentionShape::new(4, 2, 2, 3).unwrap();
        let mut key_cache = backend
            .allocate(BufferLayout::f32(shape.cache_elements().unwrap()).unwrap())
            .unwrap();
        let mut value_cache = backend
            .allocate(BufferLayout::f32(shape.cache_elements().unwrap()).unwrap())
            .unwrap();
        backend
            .kv_append(
                &key,
                &value,
                &mut key_cache,
                &mut value_cache,
                shape,
                Position::Host(1),
            )
            .unwrap();
        let mut observed = vec![0.0; shape.cache_elements().unwrap()];
        backend.read_f32(&key_cache, &mut observed).unwrap();
        assert_eq!(
            observed,
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        );
    }

    #[test]
    fn prefill_attention_is_causal_within_the_position_block() {
        let mut backend = CpuBackend::new();
        let shape = AttentionShape::new(2, 1, 2, 2).unwrap();
        let key = backend
            .upload(
                BufferLayout::f32(4).unwrap(),
                &f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
            )
            .unwrap();
        let value = backend
            .upload(
                BufferLayout::f32(4).unwrap(),
                &f32_bytes(&[2.0, 4.0, 6.0, 8.0]),
            )
            .unwrap();
        let query = backend
            .upload(
                BufferLayout::f32(8).unwrap(),
                &f32_bytes(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            )
            .unwrap();
        let mut key_cache = backend
            .allocate(BufferLayout::f32(shape.cache_elements().unwrap()).unwrap())
            .unwrap();
        let mut value_cache = backend
            .allocate(BufferLayout::f32(shape.cache_elements().unwrap()).unwrap())
            .unwrap();
        backend
            .kv_append_chunk(&key, &value, &mut key_cache, &mut value_cache, shape, 0, 2)
            .unwrap();
        let mut output = backend.allocate(BufferLayout::f32(8).unwrap()).unwrap();
        backend
            .attention_prefill(&query, &key_cache, &value_cache, &mut output, shape, 0, 2)
            .unwrap();
        let mut observed = vec![0.0; 8];
        backend.read_f32(&output, &mut observed).unwrap();
        assert_eq!(&observed[..4], &[2.0, 4.0, 2.0, 4.0]);
        let high = (1.0_f32 / 2.0_f32.sqrt()).exp();
        let denominator = high + 1.0;
        let expected = [
            (high * 2.0 + 6.0) / denominator,
            (high * 4.0 + 8.0) / denominator,
            (2.0 + high * 6.0) / denominator,
            (4.0 + high * 8.0) / denominator,
        ];
        for (actual, expected) in observed[4..].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn q8_1_emulation_uses_block_scale_and_round_to_even() {
        let mut input = vec![0.0; 32];
        input[0] = 1.0;
        input[1] = -1.0;
        input[2] = 0.5 / 127.0;
        let observed = emulate_q8_1(&input);
        let scale = f16::from_f32(1.0 / 127.0).to_f32();
        assert_eq!(observed[0], 127.0 * scale);
        assert_eq!(observed[1], -127.0 * scale);
        assert_eq!(observed[2], 0.0);
    }

    #[test]
    fn q8_kv_decoder_exhausts_every_block_code() {
        let scale = f16::from_f32(0.125);
        for raw_code in u8::MIN..=u8::MAX {
            let mut block = [0_u8; 34];
            block[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
            block[2..].fill(raw_code);
            let expected = f32::from(raw_code as i8) * scale.to_f32();
            for logical_index in 0..32 {
                assert_eq!(q8_kv_load(&block, logical_index), expected);
            }
        }
    }

    #[test]
    fn span_attention_matches_contiguous_for_all_kv_storages() {
        for storage in [BufferStorage::F32, BufferStorage::F16, BufferStorage::Q8Kv] {
            run_span_attention_case(storage);
        }
    }

    fn run_span_attention_case(storage: BufferStorage) {
        let mut backend = CpuBackend::new();
        let shape = AttentionShape::new(2, 1, 32, 8).unwrap();
        let projected = shape.projected_kv_elements().unwrap();
        let key_values: Vec<f32> = (0..shape.max_context() * projected)
            .map(|index| (index as f32 - 17.0) * 0.03125)
            .collect();
        let value_values: Vec<f32> = (0..shape.max_context() * projected)
            .map(|index| (index as f32 + 3.0) * 0.017)
            .collect();
        let key = backend
            .upload(
                BufferLayout::f32(key_values.len()).unwrap(),
                &f32_bytes(&key_values),
            )
            .unwrap();
        let value = backend
            .upload(
                BufferLayout::f32(value_values.len()).unwrap(),
                &f32_bytes(&value_values),
            )
            .unwrap();
        let contiguous_key_layout = test_cache_layout(shape, storage, shape.max_context());
        let mut contiguous_key = backend.allocate(contiguous_key_layout).unwrap();
        let mut contiguous_value = backend.allocate(contiguous_key_layout).unwrap();
        backend
            .kv_append_chunk(
                &key,
                &value,
                &mut contiguous_key,
                &mut contiguous_value,
                shape,
                0,
                shape.max_context(),
            )
            .unwrap();

        let boundaries = [0, 1, 3, 8];
        let mut segments = Vec::new();
        for window in boundaries.windows(2) {
            let start = window[0];
            let tokens = window[1] - start;
            let layout = test_cache_layout(shape, storage, tokens);
            let mut segment_key = backend.allocate(layout).unwrap();
            let mut segment_value = backend.allocate(layout).unwrap();
            let key_chunk = backend
                .upload(
                    BufferLayout::f32(tokens * projected).unwrap(),
                    &f32_bytes(&key_values[start * projected..window[1] * projected]),
                )
                .unwrap();
            let value_chunk = backend
                .upload(
                    BufferLayout::f32(tokens * projected).unwrap(),
                    &f32_bytes(&value_values[start * projected..window[1] * projected]),
                )
                .unwrap();
            backend
                .kv_append_chunk_span(
                    &key_chunk,
                    &value_chunk,
                    KvWriteSpan::new(&mut segment_key, &mut segment_value, start, tokens).unwrap(),
                    shape,
                    start,
                    tokens,
                )
                .unwrap();
            segments.push((segment_key, segment_value, start, tokens));
        }
        let contiguous_spans = [KvReadSpan::new(
            &contiguous_key,
            &contiguous_value,
            0,
            shape.max_context(),
            shape.max_context(),
        )
        .unwrap()];
        let segmented_spans: Vec<_> = segments
            .iter()
            .map(|(key, value, start, tokens)| {
                KvReadSpan::new(key, value, *start, *tokens, *tokens).unwrap()
            })
            .collect();
        let contiguous_view = KvReadView::new(&contiguous_spans).unwrap();
        let segmented_view = KvReadView::new(&segmented_spans).unwrap();
        let query_values: Vec<f32> = (0..shape.query_elements().unwrap())
            .map(|index| (index as f32 - 11.0) * 0.013)
            .collect();
        let query = backend
            .upload(
                BufferLayout::f32(query_values.len()).unwrap(),
                &f32_bytes(&query_values),
            )
            .unwrap();
        let mut contiguous_output = backend
            .allocate(BufferLayout::f32(shape.query_elements().unwrap()).unwrap())
            .unwrap();
        let mut segmented_output = backend
            .allocate(BufferLayout::f32(shape.query_elements().unwrap()).unwrap())
            .unwrap();
        backend
            .attention_decode_spans(
                &query,
                contiguous_view,
                &mut contiguous_output,
                shape,
                Position::Host(7),
            )
            .unwrap();
        backend
            .attention_decode_spans(
                &query,
                segmented_view,
                &mut segmented_output,
                shape,
                Position::Host(7),
            )
            .unwrap();
        let mut contiguous_values = vec![0.0; shape.query_elements().unwrap()];
        let mut segmented_values = vec![0.0; shape.query_elements().unwrap()];
        backend
            .read_f32(&contiguous_output, &mut contiguous_values)
            .unwrap();
        backend
            .read_f32(&segmented_output, &mut segmented_values)
            .unwrap();
        assert_eq!(contiguous_values, segmented_values);

        let prefill_tokens = 3;
        let prefill_query_values: Vec<f32> = (0..prefill_tokens * shape.query_elements().unwrap())
            .map(|index| (index as f32 + 5.0) * 0.009)
            .collect();
        let prefill_query = backend
            .upload(
                BufferLayout::f32(prefill_query_values.len()).unwrap(),
                &f32_bytes(&prefill_query_values),
            )
            .unwrap();
        let prefill_output_layout =
            BufferLayout::f32(prefill_tokens * shape.query_elements().unwrap()).unwrap();
        let mut contiguous_prefill = backend.allocate(prefill_output_layout).unwrap();
        let mut segmented_prefill = backend.allocate(prefill_output_layout).unwrap();
        let contiguous_view = KvReadView::new(&contiguous_spans).unwrap();
        let segmented_view = KvReadView::new(&segmented_spans).unwrap();
        backend
            .attention_prefill_spans(
                &prefill_query,
                contiguous_view,
                &mut contiguous_prefill,
                shape,
                5,
                prefill_tokens,
            )
            .unwrap();
        backend
            .attention_prefill_spans(
                &prefill_query,
                segmented_view,
                &mut segmented_prefill,
                shape,
                5,
                prefill_tokens,
            )
            .unwrap();
        let mut contiguous_prefill_values =
            vec![0.0; prefill_tokens * shape.query_elements().unwrap()];
        let mut segmented_prefill_values =
            vec![0.0; prefill_tokens * shape.query_elements().unwrap()];
        backend
            .read_f32(&contiguous_prefill, &mut contiguous_prefill_values)
            .unwrap();
        backend
            .read_f32(&segmented_prefill, &mut segmented_prefill_values)
            .unwrap();
        assert_eq!(contiguous_prefill_values, segmented_prefill_values);
    }

    fn test_cache_layout(
        shape: AttentionShape,
        storage: BufferStorage,
        capacity: usize,
    ) -> BufferLayout {
        let elements = shape.n_head_kv() * capacity * shape.head_dim();
        match storage {
            BufferStorage::F32 => BufferLayout::f32(elements).unwrap(),
            BufferStorage::F16 => BufferLayout::f16(elements).unwrap(),
            BufferStorage::Q8Kv => BufferLayout::q8_kv(elements).unwrap(),
            _ => unreachable!(),
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}
