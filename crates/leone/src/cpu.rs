use crate::backend::{exact_len, validate_positive};
use crate::{
    AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot, BufferStorage,
    Determinism, MemoryCapacity, Position, QuantFormat, QuantMatrix, RopePairing, RopeShape,
    VectorShape,
};
use half::f16;
use leone_gguf::ref_dequant;
use rayon::prelude::*;

/// An opaque buffer owned by the scalar CPU backend.
#[derive(Debug, Clone)]
pub struct CpuBuffer {
    layout: BufferLayout,
    storage: CpuStorage,
}

#[derive(Debug, Clone)]
enum CpuStorage {
    Bytes(Vec<u8>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl CpuBuffer {
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
    rope_pairing: RopePairing,
}

impl CpuBackend {
    /// Creates a scalar CPU backend with no declared memory limit.
    pub const fn new() -> Self {
        Self {
            q8_1_activations: false,
            rope_inverse_frequencies: Vec::new(),
            rope_pairing: RopePairing::HalfSplit,
        }
    }

    /// Creates a diagnostic CPU backend that emulates CUDA q8_1 activations.
    ///
    /// The quantizer uses 32-value blocks and stores each scale as `f16`.
    /// This mode isolates activation quantization from other backend differences.
    pub const fn with_q8_1_activations() -> Self {
        Self {
            q8_1_activations: true,
            rope_inverse_frequencies: Vec::new(),
            rope_pairing: RopePairing::HalfSplit,
        }
    }
}

impl Backend for CpuBackend {
    fn configure_rope(
        &mut self,
        head_dim: usize,
        theta: f32,
        frequency_factors: Option<&[f32]>,
        pairing: RopePairing,
    ) -> Result<(), BackendError> {
        validate_positive("RoPE theta", theta)?;
        let half = head_dim / 2;
        if let Some(factors) = frequency_factors {
            exact_len("RoPE frequency factors", half, factors.len())?;
            for &factor in factors {
                validate_positive("RoPE frequency factor", factor)?;
            }
        }
        self.rope_inverse_frequencies = (0..half)
            .map(|pair| {
                let factor = frequency_factors
                    .map(|factors| f64::from(factors[pair]))
                    .unwrap_or(1.0);
                f64::from(theta).powf(-2.0 * pair as f64 / head_dim as f64) / factor
            })
            .collect();
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

    fn determinism(&self) -> Determinism {
        Determinism::FixedOrder
    }

    fn memory_capacity(&mut self) -> Result<MemoryCapacity, BackendError> {
        Ok(MemoryCapacity::Unbounded)
    }

    fn allocate(&mut self, layout: BufferLayout) -> Result<Self::Buffer, BackendError> {
        let storage = match layout.storage() {
            BufferStorage::F16 => CpuStorage::F16(zeroed(layout.elements(), "allocate f16")?),
            BufferStorage::F32 => CpuStorage::F32(zeroed(layout.elements(), "allocate f32")?),
            BufferStorage::U32 => CpuStorage::U32(zeroed(layout.elements(), "allocate u32")?),
            BufferStorage::Q8Kv | BufferStorage::Q4K | BufferStorage::Q6K => {
                CpuStorage::Bytes(zeroed(layout.bytes(), "allocate quantized bytes")?)
            }
        };
        Ok(CpuBuffer { layout, storage })
    }

    fn upload(&mut self, layout: BufferLayout, bytes: &[u8]) -> Result<Self::Buffer, BackendError> {
        exact_len("uploaded bytes", layout.bytes(), bytes.len())?;
        let storage = match layout.storage() {
            BufferStorage::F16 => CpuStorage::F16(parse_f16(bytes)),
            BufferStorage::F32 => CpuStorage::F32(parse_f32(bytes)),
            BufferStorage::U32 => CpuStorage::U32(parse_u32(bytes)),
            BufferStorage::Q8Kv | BufferStorage::Q4K | BufferStorage::Q6K => {
                CpuStorage::Bytes(bytes.to_vec())
            }
        };
        Ok(CpuBuffer { layout, storage })
    }

    fn clone_buffer(&mut self, source: &Self::Buffer) -> Result<Self::Buffer, BackendError> {
        Ok(source.clone())
    }

    fn download_buffer(&mut self, source: &Self::Buffer) -> Result<BufferSnapshot, BackendError> {
        let bytes = match &source.storage {
            CpuStorage::Bytes(values) => values.clone(),
            CpuStorage::F16(values) => values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
            CpuStorage::F32(values) => values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
            CpuStorage::U32(values) => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        };
        BufferSnapshot::new(source.layout, bytes)
    }

    fn restore_buffer(&mut self, source: &BufferSnapshot) -> Result<Self::Buffer, BackendError> {
        self.upload(source.layout(), source.bytes())
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
        check_layout("prefill GEMM weights", shape.layout()?, weights.layout)?;
        let input = input.f32()?;
        let output = output.f32_mut()?;
        let input_elements =
            tokens
                .checked_mul(shape.columns())
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill GEMM input elements",
                })?;
        let output_elements =
            tokens
                .checked_mul(shape.rows())
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill GEMM output elements",
                })?;
        exact_len("prefill GEMM input", input_elements, input.len())?;
        exact_len("prefill GEMM output", output_elements, output.len())?;
        let quantized_input = self.q8_1_activations.then(|| emulate_q8_1(input));
        let input = quantized_input.as_deref().unwrap_or(input);
        let weights = weights.bytes()?;
        let row_bytes = shape.row_bytes()?;
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

    fn gemv(
        &mut self,
        weights: &Self::Buffer,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        check_layout("GEMV weights", shape.layout()?, weights.layout)?;
        let input = input.f32()?;
        let output = output.f32_mut()?;
        exact_len("GEMV input", shape.columns(), input.len())?;
        exact_len("GEMV output", shape.rows(), output.len())?;
        let quantized_input = self.q8_1_activations.then(|| emulate_q8_1(input));
        let input = quantized_input.as_deref().unwrap_or(input);
        let weights = weights.bytes()?;
        let row_bytes = shape.row_bytes()?;
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
            })?;
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
        let left = left.f32()?;
        let right = right.f32()?;
        let weight = weight.f32()?;
        let output = output.f32_mut()?;
        let elements = shape.elements()?;
        exact_len("residual left", elements, left.len())?;
        exact_len("residual right", elements, right.len())?;
        exact_len("RMSNorm weight", shape.columns(), weight.len())?;
        exact_len("RMSNorm output", elements, output.len())?;
        for row in 0..shape.rows() {
            let start = row * shape.columns();
            let end = start + shape.columns();
            let square_sum = left[start..end].iter().zip(&right[start..end]).fold(
                0.0_f32,
                |sum, (left, right)| {
                    let value = left + right;
                    value.mul_add(value, sum)
                },
            );
            let inverse_rms = (square_sum / shape.columns() as f32 + epsilon)
                .sqrt()
                .recip();
            for column in 0..shape.columns() {
                output[start + column] =
                    (left[start + column] + right[start + column]) * weight[column] * inverse_rms;
            }
        }
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
        let left = left.f32()?;
        let right = right.f32()?;
        let weight = weight.f32()?;
        let residual = residual.f32_mut()?;
        let output = output.f32_mut()?;
        let elements = shape.elements()?;
        exact_len("residual left", elements, left.len())?;
        exact_len("residual right", elements, right.len())?;
        exact_len("RMSNorm weight", shape.columns(), weight.len())?;
        exact_len("stored residual", elements, residual.len())?;
        exact_len("RMSNorm output", elements, output.len())?;
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
        for token in 0..shape.tokens() {
            for head in 0..shape.heads() {
                let base = (token * shape.heads() + head) * shape.head_dim();
                for pair in 0..half {
                    let token_position =
                        position
                            .checked_add(token)
                            .ok_or(BackendError::SizeOverflow {
                                field: "RoPE position",
                            })?;
                    let angle = token_position as f64 * self.rope_inverse_frequencies[pair];
                    let (sine, cosine) = angle.sin_cos();
                    let (first_index, second_index) = match self.rope_pairing {
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
        if position >= shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position,
                max_context: shape.max_context(),
            });
        }
        let key = key.f32()?;
        let value = value.f32()?;
        let projected = shape.projected_kv_elements()?;
        let cached = shape.cache_elements()?;
        exact_len("projected key", projected, key.len())?;
        exact_len("projected value", projected, value.len())?;
        match key_cache.layout.storage() {
            BufferStorage::F32 => {
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
            }
            BufferStorage::F16 => {
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
            }
            BufferStorage::Q8Kv => {
                let key_cache = key_cache.bytes_mut()?;
                let value_cache = value_cache.bytes_mut()?;
                exact_len("Q8 key cache", cached / 32 * 34, key_cache.len())?;
                exact_len("Q8 value cache", cached / 32 * 34, value_cache.len())?;
                for head in 0..shape.n_head_kv() {
                    let source = head * shape.head_dim();
                    let target = (head * shape.max_context() + position) * shape.head_dim();
                    q8_kv_store(
                        &mut key_cache[..],
                        target,
                        &key[source..source + shape.head_dim()],
                    );
                    q8_kv_store(
                        &mut value_cache[..],
                        target,
                        &value[source..source + shape.head_dim()],
                    );
                }
            }
            storage => return Err(storage_error("write KV cache", storage)),
        }
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
        let end_position =
            start_position
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
        let key = key.f32()?;
        let value = value.f32()?;
        let projected = shape.projected_kv_elements()?.checked_mul(tokens).ok_or(
            BackendError::SizeOverflow {
                field: "prefill projected KV elements",
            },
        )?;
        exact_len("prefill projected key", projected, key.len())?;
        exact_len("prefill projected value", projected, value.len())?;
        let cached = shape.cache_elements()?;
        match key_cache.layout.storage() {
            BufferStorage::F32 => {
                let key_cache = key_cache.f32_mut()?;
                let value_cache = value_cache.f32_mut()?;
                exact_len("key cache", cached, key_cache.len())?;
                exact_len("value cache", cached, value_cache.len())?;
                for token in 0..tokens {
                    for head in 0..shape.n_head_kv() {
                        let source = (token * shape.n_head_kv() + head) * shape.head_dim();
                        let target = (head * shape.max_context() + start_position + token)
                            * shape.head_dim();
                        key_cache[target..target + shape.head_dim()]
                            .copy_from_slice(&key[source..source + shape.head_dim()]);
                        value_cache[target..target + shape.head_dim()]
                            .copy_from_slice(&value[source..source + shape.head_dim()]);
                    }
                }
            }
            BufferStorage::F16 => {
                let key_cache = key_cache.f16_mut()?;
                let value_cache = value_cache.f16_mut()?;
                exact_len("key cache", cached, key_cache.len())?;
                exact_len("value cache", cached, value_cache.len())?;
                for token in 0..tokens {
                    for head in 0..shape.n_head_kv() {
                        let source = (token * shape.n_head_kv() + head) * shape.head_dim();
                        let target = (head * shape.max_context() + start_position + token)
                            * shape.head_dim();
                        for dimension in 0..shape.head_dim() {
                            key_cache[target + dimension] = f16::from_f32(key[source + dimension]);
                            value_cache[target + dimension] =
                                f16::from_f32(value[source + dimension]);
                        }
                    }
                }
            }
            BufferStorage::Q8Kv => {
                let key_cache = key_cache.bytes_mut()?;
                let value_cache = value_cache.bytes_mut()?;
                exact_len("Q8 key cache", cached / 32 * 34, key_cache.len())?;
                exact_len("Q8 value cache", cached / 32 * 34, value_cache.len())?;
                for token in 0..tokens {
                    for head in 0..shape.n_head_kv() {
                        let source = (token * shape.n_head_kv() + head) * shape.head_dim();
                        let target = (head * shape.max_context() + start_position + token)
                            * shape.head_dim();
                        q8_kv_store(
                            &mut key_cache[..],
                            target,
                            &key[source..source + shape.head_dim()],
                        );
                        q8_kv_store(
                            &mut value_cache[..],
                            target,
                            &value[source..source + shape.head_dim()],
                        );
                    }
                }
            }
            storage => return Err(storage_error("write prefill KV cache", storage)),
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
        exact_len("attention query", shape.query_elements()?, query.len())?;
        exact_len(
            "attention key cache",
            shape.cache_elements()?,
            key_cache.layout.elements(),
        )?;
        exact_len(
            "attention value cache",
            shape.cache_elements()?,
            value_cache.layout.elements(),
        )?;
        if key_cache.layout.storage() != value_cache.layout.storage() {
            return Err(BackendError::operation(
                "read KV cache",
                "key and value storage differ",
            ));
        }
        let key_at = |index: usize| -> Result<f32, BackendError> {
            match &key_cache.storage {
                CpuStorage::F32(values) => Ok(values[index]),
                CpuStorage::F16(values) => Ok(values[index].to_f32()),
                CpuStorage::Bytes(values) if key_cache.layout.storage() == BufferStorage::Q8Kv => {
                    Ok(q8_kv_load(values, index))
                }
                _ => Err(storage_error(
                    "read attention key cache",
                    key_cache.layout.storage(),
                )),
            }
        };
        let value_at = |index: usize| -> Result<f32, BackendError> {
            match &value_cache.storage {
                CpuStorage::F32(values) => Ok(values[index]),
                CpuStorage::F16(values) => Ok(values[index].to_f32()),
                CpuStorage::Bytes(values)
                    if value_cache.layout.storage() == BufferStorage::Q8Kv =>
                {
                    Ok(q8_kv_load(values, index))
                }
                _ => Err(storage_error(
                    "read attention value cache",
                    value_cache.layout.storage(),
                )),
            }
        };
        exact_len("attention output", shape.query_elements()?, output.len())?;
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
                    dot =
                        query[query_base + dimension].mul_add(key_at(cache_base + dimension)?, dot);
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
                        + score_scale * value_at(cache_base + dimension)?;
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
        let block_elements =
            tokens
                .checked_mul(shape.query_elements()?)
                .ok_or(BackendError::SizeOverflow {
                    field: "prefill attention block elements",
                })?;
        exact_len("prefill attention query", block_elements, query.len())?;
        exact_len("prefill attention output", block_elements, output.len())?;
        exact_len(
            "prefill attention key cache",
            shape.cache_elements()?,
            key_cache.layout.elements(),
        )?;
        exact_len(
            "prefill attention value cache",
            shape.cache_elements()?,
            value_cache.layout.elements(),
        )?;
        if key_cache.layout.storage() != value_cache.layout.storage() {
            return Err(BackendError::operation(
                "read prefill KV cache",
                "key and value storage differ",
            ));
        }
        let key_at = |index: usize| -> Result<f32, BackendError> {
            match &key_cache.storage {
                CpuStorage::F32(values) => Ok(values[index]),
                CpuStorage::F16(values) => Ok(values[index].to_f32()),
                CpuStorage::Bytes(values) if key_cache.layout.storage() == BufferStorage::Q8Kv => {
                    Ok(q8_kv_load(values, index))
                }
                _ => Err(storage_error(
                    "read prefill attention key cache",
                    key_cache.layout.storage(),
                )),
            }
        };
        let value_at = |index: usize| -> Result<f32, BackendError> {
            match &value_cache.storage {
                CpuStorage::F32(values) => Ok(values[index]),
                CpuStorage::F16(values) => Ok(values[index].to_f32()),
                CpuStorage::Bytes(values)
                    if value_cache.layout.storage() == BufferStorage::Q8Kv =>
                {
                    Ok(q8_kv_load(values, index))
                }
                _ => Err(storage_error(
                    "read prefill attention value cache",
                    value_cache.layout.storage(),
                )),
            }
        };
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
                        dot = query[query_base + dimension]
                            .mul_add(key_at(cache_base + dimension)?, dot);
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
                            + score_scale * value_at(cache_base + dimension)?;
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

    fn embed_gather(
        &mut self,
        table: &Self::Buffer,
        row: &Self::Buffer,
        output: &mut Self::Buffer,
        shape: QuantMatrix,
    ) -> Result<(), BackendError> {
        let rows = row.u32()?;
        exact_len("embedding row", 1, rows.len())?;
        let row = usize::try_from(rows[0]).map_err(|_| BackendError::SizeOverflow {
            field: "embedding row",
        })?;
        if row >= shape.rows() {
            return Err(BackendError::RowOutOfBounds {
                row,
                rows: shape.rows(),
            });
        }
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
        for (token, row) in rows.iter().copied().enumerate() {
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
        Ok(())
    }
}

fn zeroed<T: Default + Clone>(len: usize, operation: &'static str) -> Result<Vec<T>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| BackendError::operation(operation, error))?;
    values.resize(len, T::default());
    Ok(values)
}

fn parse_f16(bytes: &[u8]) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .map(|value| f16::from_bits(u16::from_le_bytes([value[0], value[1]])))
        .collect()
}

fn parse_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect()
}

fn parse_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|value| u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect()
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
    use super::*;

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

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}
