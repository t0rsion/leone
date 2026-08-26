use std::collections::BTreeMap;
use std::fmt;
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

/// A K-quant format supported by the v0.1 runtime.
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
        nonzero("prefill chunk tokens", chunk_tokens)?;
        nonzero("prefill context tokens", context_tokens)?;
        nonzero("prefill n_head", n_head)?;
        nonzero("prefill n_head_kv", n_head_kv)?;
        nonzero("prefill head_dim", head_dim)?;
        nonzero("prefill n_embd", n_embd)?;
        nonzero("prefill n_ff", n_ff)?;
        nonzero("prefill max matrix rows", max_matrix_rows)?;
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
            Self::Embed => "embed",
            Self::QkvGemv => "qkv gemv",
            Self::QkNorm => "qknorm",
            Self::Rope => "rope",
            Self::KvAppend => "kv append",
            Self::Attention => "attention",
            Self::OutputGemv => "o gemv",
            Self::FfnGemv => "ffn gemvs",
            Self::SwiGlu => "swiglu",
            Self::Norm => "norms",
            Self::LmHeadGemv => "lm_head gemv",
            Self::Argmax => "argmax",
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
    fn memory_capacity(&mut self) -> Result<MemoryCapacity, BackendError>;
    fn model_import_metrics(&self) -> ModelImportMetrics {
        ModelImportMetrics::default()
    }
    fn allocate(&mut self, layout: BufferLayout) -> Result<Self::Buffer, BackendError>;
    fn upload(&mut self, layout: BufferLayout, bytes: &[u8]) -> Result<Self::Buffer, BackendError>;
    /// Allocates an independent buffer with the exact contents of `source`.
    ///
    /// The copy is ordered with other backend work. A backend may enqueue it,
    /// so the caller must synchronize before timing completion or using the
    /// result from another execution context.
    fn clone_buffer(&mut self, source: &Self::Buffer) -> Result<Self::Buffer, BackendError>;
    /// Copies one buffer into portable host-owned physical bytes.
    fn download_buffer(&mut self, source: &Self::Buffer) -> Result<BufferSnapshot, BackendError>;
    /// Restores physical bytes without applying model-import transformations.
    fn restore_buffer(&mut self, source: &BufferSnapshot) -> Result<Self::Buffer, BackendError>;
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
