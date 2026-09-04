use crate::ffi;
use crate::shape::{argmax_blocks, validate_element_grid};
use crate::{
    AttentionShape, Error, QuantFormat, QuantizedMatrixShape, Result, RopeShape, VectorShape,
};
use std::ffi::{c_void, CStr};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A plain scalar type that CUDA may copy by value.
pub trait DeviceCopy: sealed::Sealed + Copy + 'static {}

impl DeviceCopy for u8 {}
impl DeviceCopy for u16 {}
impl DeviceCopy for u32 {}
impl DeviceCopy for f32 {}
impl DeviceCopy for f64 {}

/// Selects and initializes one CUDA runtime device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context {
    device: i32,
}

impl Context {
    /// Initializes the CUDA runtime on `device`.
    pub fn new(device: i32) -> Result<Self> {
        activate_device(device)?;
        // SAFETY: The C wrapper takes no pointers and initializes the current device.
        check(unsafe { ffi::ie_cuda_initialize() }, "initialize")?;
        Ok(Self { device })
    }

    /// Returns the CUDA runtime device index.
    pub const fn device(self) -> i32 {
        self.device
    }

    /// Returns the fixed decode attention split count for one graph bucket.
    pub fn attention_split_count(&self, shape: AttentionShape, f16_cache: bool) -> Result<usize> {
        activate_device(self.device)?;
        let mut split_count = 0;
        // SAFETY: `split_count` is a live host value. The shape has checked
        // nonzero dimensions, and the format flag is either zero or one.
        check(
            unsafe {
                ffi::ie_attention_split_count(
                    i32::from(f16_cache),
                    shape.n_head(),
                    shape.max_context(),
                    &mut split_count,
                )
            },
            "query attention split count",
        )?;
        Ok(split_count)
    }

    /// Returns the current free and total device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        activate_device(self.device)?;
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        // SAFETY: Both output pointers refer to live host `usize` values.
        check(
            unsafe { ffi::ie_cuda_mem_get_info(&mut free_bytes, &mut total_bytes) },
            "query device memory",
        )?;
        Ok((free_bytes, total_bytes))
    }

    /// Allocates an uninitialized device buffer with a typed element count.
    pub fn alloc<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        if len == 0 {
            return Err(Error::Zero {
                field: "device buffer length",
            });
        }
        let bytes = byte_len::<T>(len, "device buffer bytes")?;
        activate_device(self.device)?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid output pointer, and `bytes` is nonzero.
        check(
            unsafe { ffi::ie_cuda_malloc(&mut raw, bytes) },
            "allocate device memory",
        )?;
        let pointer = NonNull::new(raw).ok_or_else(|| Error::Runtime {
            operation: "allocate device memory",
            code: -1,
            message: "CUDA returned a null pointer".to_owned(),
        })?;
        Ok(DeviceBuffer {
            pointer,
            len,
            bytes,
            device: self.device,
            marker: PhantomData,
        })
    }

    /// Allocates a buffer and copies all host elements into it.
    pub fn copy_to_device<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut buffer = self.alloc(values.len())?;
        buffer.copy_from(values)?;
        Ok(buffer)
    }
}

/// An owned cuBLASLt handle for FP16 prefill matrix products.
#[derive(Debug)]
pub struct CublasLt {
    raw: NonNull<c_void>,
    device: i32,
}

impl CublasLt {
    /// Creates one cuBLASLt handle on the context device.
    pub fn new(context: &Context) -> Result<Self> {
        activate_device(context.device)?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid output pointer for one cuBLASLt handle.
        check_cublas(
            unsafe { ffi::ie_cublaslt_create(&mut raw) },
            "create cuBLASLt handle",
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::Runtime {
            operation: "create cuBLASLt handle",
            code: -1,
            message: "cuBLASLt returned a null handle".to_owned(),
        })?;
        Ok(Self {
            raw,
            device: context.device,
        })
    }

    fn raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }
}

impl Drop for CublasLt {
    fn drop(&mut self) {
        // SAFETY: `raw` is a live handle released exactly once here.
        unsafe {
            let _ = ffi::ie_cuda_set_device(self.device);
            let _ = ffi::ie_cublaslt_destroy(self.raw.as_ptr());
        }
    }
}

/// An owned device allocation with a fixed scalar type and element count.
#[derive(Debug)]
pub struct DeviceBuffer<T: DeviceCopy> {
    pointer: NonNull<c_void>,
    len: usize,
    bytes: usize,
    device: i32,
    marker: PhantomData<T>,
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Returns the typed element count.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Always false. `alloc` rejects a zero-length buffer.
    pub const fn is_empty(&self) -> bool {
        false
    }

    pub(crate) fn identity(&self) -> usize {
        self.pointer.as_ptr() as usize
    }

    /// Copies an exact host slice into the device allocation.
    pub fn copy_from(&mut self, values: &[T]) -> Result<()> {
        exact_len("host source", self.len, values.len())?;
        activate_device(self.device)?;
        // SAFETY: Both pointers cover `self.bytes`. The synchronous copy ends
        // before `values` can be released or changed.
        check(
            unsafe {
                ffi::ie_cuda_copy_h2d(
                    self.pointer.as_ptr(),
                    values.as_ptr().cast::<c_void>(),
                    self.bytes,
                )
            },
            "copy host to device",
        )
    }

    /// Enqueues exact physical host bytes into this allocation.
    ///
    /// The caller keeps `bytes` live and unchanged until `stream` completes.
    pub fn copy_bytes_from_async(&mut self, stream: &Stream, bytes: &[u8]) -> Result<()> {
        exact_len("host source bytes", self.bytes, bytes.len())?;
        same_device(self.device, stream.device)?;
        activate_device(self.device)?;
        // SAFETY: Both pointers cover `self.bytes`. The caller keeps the host
        // slice live until the stream completes.
        check(
            unsafe {
                ffi::ie_cuda_copy_h2d_async(
                    self.pointer.as_ptr(),
                    bytes.as_ptr().cast::<c_void>(),
                    self.bytes,
                    stream.raw(),
                )
            },
            "enqueue host bytes to device",
        )
    }

    /// Copies the complete device allocation into an exact host slice.
    pub fn copy_to(&self, values: &mut [T]) -> Result<()> {
        exact_len("host destination", self.len, values.len())?;
        activate_device(self.device)?;
        // SAFETY: Both pointers cover `self.bytes`. The synchronous copy ends
        // before this function returns the initialized host slice.
        check(
            unsafe {
                ffi::ie_cuda_copy_d2h(
                    values.as_mut_ptr().cast::<c_void>(),
                    self.pointer.as_ptr(),
                    self.bytes,
                )
            },
            "copy device to host",
        )
    }

    /// Enqueues a complete device-to-host copy on `stream`.
    ///
    /// The caller keeps `values` live and unchanged until the stream completes.
    pub fn copy_to_async(&self, stream: &Stream, values: &mut [T]) -> Result<()> {
        exact_len("host destination", self.len, values.len())?;
        same_device(self.device, stream.device)?;
        activate_device(self.device)?;
        // SAFETY: Both pointers cover `self.bytes`. The caller keeps the host
        // slice live until the stream completes.
        check(
            unsafe {
                ffi::ie_cuda_copy_d2h_async(
                    values.as_mut_ptr().cast::<c_void>(),
                    self.pointer.as_ptr(),
                    self.bytes,
                    stream.raw(),
                )
            },
            "enqueue device to host copy",
        )
    }

    /// Enqueues a complete copy from another device allocation.
    pub fn copy_from_device_async(&mut self, stream: &Stream, source: &Self) -> Result<()> {
        exact_len("device source", self.len, source.len)?;
        same_device(self.device, source.device)?;
        same_device(self.device, stream.device)?;
        activate_device(self.device)?;
        // SAFETY: Both live allocations cover `self.bytes`. The shared stream
        // orders the copy before either allocation is reused.
        check(
            unsafe {
                ffi::ie_cuda_copy_d2d_async(
                    self.pointer.as_ptr(),
                    source.pointer.as_ptr(),
                    self.bytes,
                    stream.raw(),
                )
            },
            "enqueue device to device copy",
        )
    }

    fn const_ptr(&self) -> *const T {
        self.pointer.as_ptr().cast::<T>()
    }

    fn mut_ptr(&mut self) -> *mut T {
        self.pointer.as_ptr().cast::<T>()
    }
}

impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: The pointer came from `ie_cuda_malloc` and is freed once here.
        unsafe {
            let _ = ffi::ie_cuda_set_device(self.device);
            let _ = ffi::ie_cuda_free(self.pointer.as_ptr());
        }
    }
}

/// An owned CUDA stream for ordered asynchronous kernel launches.
#[derive(Debug)]
pub struct Stream {
    raw: NonNull<c_void>,
    device: i32,
}

impl Stream {
    /// Creates a nonblocking stream on the context device.
    pub fn new(context: &Context) -> Result<Self> {
        activate_device(context.device)?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid output pointer for the created stream.
        check(
            unsafe { ffi::ie_cuda_stream_create(&mut raw) },
            "create stream",
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::Runtime {
            operation: "create stream",
            code: -1,
            message: "CUDA returned a null stream".to_owned(),
        })?;
        Ok(Self {
            raw,
            device: context.device,
        })
    }

    /// Waits until every prior operation in this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        activate_device(self.device)?;
        // SAFETY: `raw` is a live stream owned by this value.
        check(
            unsafe { ffi::ie_cuda_stream_synchronize(self.raw.as_ptr()) },
            "synchronize stream",
        )
    }

    /// Starts thread-local CUDA graph capture on this stream.
    pub fn begin_graph_capture(&self) -> Result<()> {
        activate_device(self.device)?;
        // SAFETY: `raw` is a live nonblocking stream on the current device.
        check(
            unsafe { ffi::ie_cuda_graph_capture_begin(self.raw.as_ptr()) },
            "begin graph capture",
        )
    }

    /// Finishes capture and instantiates an executable graph.
    pub fn end_graph_capture(&self) -> Result<Graph> {
        activate_device(self.device)?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` receives one graph captured on this live stream.
        check(
            unsafe { ffi::ie_cuda_graph_capture_end(self.raw.as_ptr(), &mut raw) },
            "end graph capture",
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::Runtime {
            operation: "end graph capture",
            code: -1,
            message: "CUDA returned a null graph".to_owned(),
        })?;
        Ok(Graph {
            raw,
            device: self.device,
        })
    }

    fn raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }
}

/// An executable CUDA graph owned by one device.
#[derive(Debug)]
pub struct Graph {
    raw: NonNull<c_void>,
    device: i32,
}

impl Graph {
    /// Enqueues one graph replay on `stream`.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        same_device(self.device, stream.device)?;
        activate_device(self.device)?;
        // SAFETY: The graph and stream are live and use the current device.
        check(
            unsafe { ffi::ie_cuda_graph_launch(self.raw.as_ptr(), stream.raw()) },
            "launch graph",
        )
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: `raw` is a live executable graph released once here.
        unsafe {
            let _ = ffi::ie_cuda_set_device(self.device);
            let _ = ffi::ie_cuda_graph_destroy(self.raw.as_ptr());
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: `raw` is a live stream and this destructor releases it once.
        unsafe {
            let _ = ffi::ie_cuda_set_device(self.device);
            let _ = ffi::ie_cuda_stream_destroy(self.raw.as_ptr());
        }
    }
}

/// An owned CUDA event for stream completion and elapsed time queries.
#[derive(Debug)]
pub struct Event {
    raw: NonNull<c_void>,
    device: i32,
}

impl Event {
    /// Creates an event with CUDA timing enabled.
    pub fn new(context: &Context) -> Result<Self> {
        activate_device(context.device)?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid output pointer for the created event.
        check(
            unsafe { ffi::ie_cuda_event_create(&mut raw) },
            "create event",
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::Runtime {
            operation: "create event",
            code: -1,
            message: "CUDA returned a null event".to_owned(),
        })?;
        Ok(Self {
            raw,
            device: context.device,
        })
    }

    /// Records this event after all earlier work in `stream`.
    pub fn record(&mut self, stream: &Stream) -> Result<()> {
        same_device(self.device, stream.device)?;
        activate_device(self.device)?;
        // SAFETY: The event and stream are live and belong to the current device.
        check(
            unsafe { ffi::ie_cuda_event_record(self.raw.as_ptr(), stream.raw()) },
            "record event",
        )
    }

    /// Waits until this event completes.
    pub fn synchronize(&self) -> Result<()> {
        activate_device(self.device)?;
        // SAFETY: `raw` is a live event owned by this value.
        check(
            unsafe { ffi::ie_cuda_event_synchronize(self.raw.as_ptr()) },
            "synchronize event",
        )
    }

    /// Returns elapsed milliseconds between two completed event records.
    pub fn elapsed_ms(start: &Self, end: &Self) -> Result<f32> {
        same_device(start.device, end.device)?;
        activate_device(start.device)?;
        let mut milliseconds = 0.0;
        // SAFETY: Both events are live, and `milliseconds` is a valid output.
        check(
            unsafe {
                ffi::ie_cuda_event_elapsed_ms(
                    &mut milliseconds,
                    start.raw.as_ptr(),
                    end.raw.as_ptr(),
                )
            },
            "measure event elapsed time",
        )?;
        Ok(milliseconds)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `raw` is a live event and this destructor releases it once.
        unsafe {
            let _ = ffi::ie_cuda_set_device(self.device);
            let _ = ffi::ie_cuda_event_destroy(self.raw.as_ptr());
        }
    }
}

/// Head dimension covered by the fused attention q8_1 epilogue.
///
/// Another head dimension runs attention without the epilogue. The caller
/// must not treat the scratch as a prepared GEMV input.
pub const PREPARED_ATTENTION_HEAD_DIM: usize = 128;

const Q8_1_BLOCK_ELEMENTS: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;

/// Device scratch for one q8_1 activation vector.
#[derive(Debug)]
pub struct GemvScratch {
    quantized_input: DeviceBuffer<u8>,
    quantized_sums: DeviceBuffer<u32>,
    epilogue_ready: DeviceBuffer<u32>,
    columns: usize,
}

const CUBLASLT_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const PREFILL_ATTENTION_TILE_TOKENS: usize = 1024;

/// Reusable device storage for one bounded chunked prefill plan.
#[derive(Debug)]
pub struct PrefillScratch {
    dequantized_weights: DeviceBuffer<u16>,
    converted_input: DeviceBuffer<u16>,
    converted_query: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    probabilities: DeviceBuffer<u16>,
    head_output: DeviceBuffer<f32>,
    converted_kv: DeviceBuffer<u16>,
    cublaslt_workspace: DeviceBuffer<u8>,
    plan: leone::PrefillPlan,
    usage: leone::PrefillWorkspace,
}

struct PrefillAllocation {
    weight_elements: usize,
    input_elements: usize,
    query_elements: usize,
    attention_elements: usize,
    compact_kv_elements: usize,
    usage: leone::PrefillWorkspace,
}

struct PrefillDimensions {
    weight_elements: usize,
    input_elements: usize,
    query_elements: usize,
    attention_elements: usize,
    compact_kv_elements: usize,
}

impl PrefillScratch {
    /// Allocates the largest layer, activation, and attention buffers in `plan`.
    pub fn new(context: &Context, plan: leone::PrefillPlan) -> Result<Self> {
        let allocation = prefill_allocation(&plan)?;
        Ok(Self {
            dequantized_weights: context.alloc(allocation.weight_elements)?,
            converted_input: context.alloc(allocation.input_elements)?,
            converted_query: context.alloc(allocation.query_elements)?,
            scores: context.alloc(allocation.attention_elements)?,
            probabilities: context.alloc(allocation.attention_elements)?,
            head_output: context.alloc(allocation.query_elements)?,
            converted_kv: context.alloc(allocation.compact_kv_elements)?,
            cublaslt_workspace: context.alloc(CUBLASLT_WORKSPACE_BYTES)?,
            plan,
            usage: allocation.usage,
        })
    }

    /// Returns the checked byte budget for this allocation.
    pub const fn usage(&self) -> leone::PrefillWorkspace {
        self.usage
    }
}

fn prefill_allocation(plan: &leone::PrefillPlan) -> Result<PrefillAllocation> {
    let dimensions = prefill_dimensions(plan)?;
    let usage = prefill_usage(&dimensions)?;
    Ok(PrefillAllocation {
        weight_elements: dimensions.weight_elements,
        input_elements: dimensions.input_elements,
        query_elements: dimensions.query_elements,
        attention_elements: dimensions.attention_elements,
        compact_kv_elements: dimensions.compact_kv_elements,
        usage,
    })
}

fn prefill_dimensions(plan: &leone::PrefillPlan) -> Result<PrefillDimensions> {
    let weight_elements = checked_product(
        plan.n_embd(),
        plan.max_matrix_rows(),
        "prefill dequantized weight elements",
    )?;
    let input_elements = checked_product(
        plan.chunk_tokens(),
        plan.n_embd().max(plan.n_ff()),
        "prefill converted activation elements",
    )?;
    let attention_tokens = plan.chunk_tokens().min(PREFILL_ATTENTION_TILE_TOKENS);
    let query_elements = checked_product3(
        plan.n_head(),
        attention_tokens,
        plan.head_dim(),
        "prefill converted query elements",
    )?;
    let attention_elements = checked_product3(
        plan.n_head(),
        attention_tokens,
        plan.context_tokens(),
        "prefill attention score elements",
    )?;
    let compact_kv_elements = checked_product3(
        plan.n_head_kv(),
        plan.context_tokens(),
        plan.head_dim(),
        "prefill compact KV elements",
    )?
    .checked_mul(2)
    .ok_or(Error::SizeOverflow {
        field: "prefill compact KV elements",
    })?;
    Ok(PrefillDimensions {
        weight_elements,
        input_elements,
        query_elements,
        attention_elements,
        compact_kv_elements,
    })
}

fn prefill_usage(dimensions: &PrefillDimensions) -> Result<leone::PrefillWorkspace> {
    let dequantized_weight_bytes = bytes_u64(dimensions.weight_elements, 2)?;
    let converted_activation_bytes = bytes_u64(dimensions.input_elements, 2)?;
    let attention_bytes = prefill_attention_bytes(
        dimensions.query_elements,
        dimensions.attention_elements,
        dimensions.compact_kv_elements,
    )?;
    let cublaslt_bytes =
        u64::try_from(CUBLASLT_WORKSPACE_BYTES).map_err(|_| Error::SizeOverflow {
            field: "cuBLASLt workspace bytes",
        })?;
    let total_bytes = dequantized_weight_bytes
        .checked_add(converted_activation_bytes)
        .and_then(|value| value.checked_add(attention_bytes))
        .and_then(|value| value.checked_add(cublaslt_bytes))
        .ok_or(Error::SizeOverflow {
            field: "prefill workspace bytes",
        })?;
    Ok(leone::PrefillWorkspace {
        dequantized_weight_bytes,
        converted_activation_bytes,
        attention_bytes,
        cublaslt_bytes,
        batch_activation_bytes: 0,
        total_bytes,
    })
}

fn prefill_attention_bytes(
    query_elements: usize,
    attention_elements: usize,
    compact_kv_elements: usize,
) -> Result<u64> {
    bytes_u64(query_elements, 2)?
        .checked_add(bytes_u64(attention_elements, 4)?)
        .and_then(|value| value.checked_add(bytes_u64(attention_elements, 2).ok()?))
        .and_then(|value| value.checked_add(bytes_u64(query_elements, 4).ok()?))
        .and_then(|value| value.checked_add(bytes_u64(compact_kv_elements, 2).ok()?))
        .ok_or(Error::SizeOverflow {
            field: "prefill attention bytes",
        })
}

/// Dequantizes one complete K-quant matrix into row-major FP16 storage.
pub fn dequantize_k_f16(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    shape: QuantizedMatrixShape,
) -> Result<()> {
    exact_len("prefill dequant weights", shape.bytes(), weights.len())?;
    let elements = checked_product(shape.rows(), shape.columns(), "prefill dequant elements")?;
    exact_len("prefill dequant output", elements, output.len())?;
    same_devices(stream.device, &[weights.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: The checked quantized buffer covers every complete 256-value
    // block. The FP16 output covers one value per logical matrix element.
    check(
        unsafe {
            ffi::ie_launch_dequant_k_f16(
                weights.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                i32::from(shape.format() == QuantFormat::Q4K),
                stream.raw(),
            )
        },
        "launch prefill weight dequantization",
    )
}

/// Runs one quantized-weight prefill GEMM through cuBLASLt.
#[allow(clippy::too_many_arguments)]
pub fn prefill_gemm(
    handle: &CublasLt,
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<()> {
    validate_prefill_gemm(
        handle, stream, weights, input, output, shape, tokens, scratch,
    )?;
    activate_device(stream.device)?;
    // SAFETY: All matrix ranges and reusable scratch capacities were checked.
    // cuBLASLt consumes them in stream order and does not retain host pointers.
    check_cublas(
        unsafe {
            ffi::ie_cublaslt_prefill_gemm(
                handle.raw(),
                weights.const_ptr(),
                input.const_ptr(),
                output.mut_ptr(),
                scratch.dequantized_weights.mut_ptr(),
                scratch.converted_input.mut_ptr(),
                shape.rows(),
                shape.columns(),
                tokens,
                i32::from(shape.format() == QuantFormat::Q4K),
                scratch.cublaslt_workspace.mut_ptr(),
                scratch.cublaslt_workspace.len(),
                stream.raw(),
            )
        },
        "run prefill GEMM",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_prefill_gemm(
    handle: &CublasLt,
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    tokens: usize,
    scratch: &PrefillScratch,
) -> Result<()> {
    validate_prefill_gemm_lengths(weights, input, output, shape, tokens)?;
    validate_prefill_gemm_scratch(shape, tokens, scratch)?;
    validate_prefill_gemm_devices(handle, stream, weights, input, output, scratch)
}

fn validate_prefill_gemm_lengths(
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    tokens: usize,
) -> Result<()> {
    exact_len("prefill GEMM weights", shape.bytes(), weights.len())?;
    exact_len(
        "prefill GEMM input",
        checked_product(tokens, shape.columns(), "prefill GEMM input")?,
        input.len(),
    )?;
    exact_len(
        "prefill GEMM output",
        checked_product(tokens, shape.rows(), "prefill GEMM output")?,
        output.len(),
    )
}

fn validate_prefill_gemm_scratch(
    shape: QuantizedMatrixShape,
    tokens: usize,
    scratch: &PrefillScratch,
) -> Result<()> {
    if tokens > scratch.plan.chunk_tokens()
        || shape
            .rows()
            .checked_mul(shape.columns())
            .ok_or(Error::SizeOverflow {
                field: "prefill GEMM weight elements",
            })?
            > scratch.dequantized_weights.len()
        || tokens
            .checked_mul(shape.columns())
            .ok_or(Error::SizeOverflow {
                field: "prefill GEMM converted input elements",
            })?
            > scratch.converted_input.len()
    {
        return Err(Error::SizeMismatch {
            name: "prefill GEMM scratch plan",
            expected: scratch.plan.chunk_tokens(),
            actual: tokens,
        });
    }
    Ok(())
}

fn validate_prefill_gemm_devices(
    handle: &CublasLt,
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    scratch: &PrefillScratch,
) -> Result<()> {
    same_devices(
        stream.device,
        &[
            handle.device,
            weights.device,
            input.device,
            output.device,
            scratch.dequantized_weights.device,
            scratch.converted_input.device,
            scratch.cublaslt_workspace.device,
        ],
    )
}

/// Appends one token-major position block to FP32 KV storage.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_chunk(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Result<()> {
    check_prefill_kv(
        stream,
        key,
        value,
        key_cache,
        value_cache,
        shape,
        start_position,
        tokens,
        shape.cache_elements(),
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked source block and head-major cache cover the launch.
    check(
        unsafe {
            ffi::ie_launch_kv_append_chunk(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                start_position,
                tokens,
                stream.raw(),
            )
        },
        "launch prefill FP32 KV append",
    )
}

/// Appends one token-major position block to FP16 KV storage.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_chunk_f16(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u16>,
    value_cache: &mut DeviceBuffer<u16>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Result<()> {
    check_prefill_kv(
        stream,
        key,
        value,
        key_cache,
        value_cache,
        shape,
        start_position,
        tokens,
        shape.cache_elements(),
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked source block and FP16 cache cover the launch.
    check(
        unsafe {
            ffi::ie_launch_kv_append_chunk_f16(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                start_position,
                tokens,
                stream.raw(),
            )
        },
        "launch prefill FP16 KV append",
    )
}

/// Appends one token-major position block to `q8` KV storage.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_chunk_q8(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u8>,
    value_cache: &mut DeviceBuffer<u8>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Result<()> {
    let cache_bytes =
        checked_product(shape.cache_elements() / 32, 34, "q8 prefill KV cache bytes")?;
    check_prefill_kv(
        stream,
        key,
        value,
        key_cache,
        value_cache,
        shape,
        start_position,
        tokens,
        cache_bytes,
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked token-major inputs and q8 cache cover every warp.
    check(
        unsafe {
            ffi::ie_launch_kv_append_chunk_q8(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                start_position,
                tokens,
                stream.raw(),
            )
        },
        "launch prefill q8 KV append",
    )
}

#[allow(clippy::too_many_arguments)]
fn check_prefill_kv<T: DeviceCopy>(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    cache_elements: usize,
) -> Result<()> {
    let end = start_position
        .checked_add(tokens)
        .ok_or(Error::SizeOverflow {
            field: "prefill KV end position",
        })?;
    if end > shape.max_context() {
        return Err(Error::ContextLength {
            context_length: end,
            max_context: shape.max_context(),
        });
    }
    let block_elements = checked_product(
        tokens,
        shape.projected_kv_elements()?,
        "prefill projected KV elements",
    )?;
    exact_len("prefill projected key", block_elements, key.len())?;
    exact_len("prefill projected value", block_elements, value.len())?;
    exact_len("prefill key cache", cache_elements, key_cache.len())?;
    exact_len("prefill value cache", cache_elements, value_cache.len())?;
    same_devices(
        stream.device,
        &[
            key.device,
            value.device,
            key_cache.device,
            value_cache.device,
        ],
    )
}

/// Runs causal position-block attention with an FP16 KV cache.
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_f16(
    handle: &CublasLt,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u16>,
    value_cache: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<()> {
    check_prefill_attention(
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
        shape.cache_elements(),
    )?;
    activate_device(stream.device)?;
    // SAFETY: Every query, cache, output, and scratch range is checked. The
    // wrapper bounds each QK and PV operation to one fixed query tile.
    check_cublas(
        unsafe {
            ffi::ie_cublaslt_attention_prefill_f16(
                handle.raw(),
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.converted_query.mut_ptr(),
                scratch.scores.mut_ptr(),
                scratch.probabilities.mut_ptr(),
                scratch.head_output.mut_ptr(),
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                start_position,
                tokens,
                scratch.cublaslt_workspace.mut_ptr(),
                scratch.cublaslt_workspace.len(),
                stream.raw(),
            )
        },
        "run tiled FP16 prefill attention",
    )
}

/// Runs causal position-block attention with an FP32 KV cache.
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_f32(
    handle: &CublasLt,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<()> {
    check_prefill_attention(
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
        shape.cache_elements(),
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked conversion buffer holds the largest compact FP16 K
    // and V copies needed by any query tile in this plan.
    check_cublas(
        unsafe {
            ffi::ie_cublaslt_attention_prefill_f32(
                handle.raw(),
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.converted_query.mut_ptr(),
                scratch.scores.mut_ptr(),
                scratch.probabilities.mut_ptr(),
                scratch.head_output.mut_ptr(),
                scratch.converted_kv.mut_ptr(),
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                start_position,
                tokens,
                scratch.cublaslt_workspace.mut_ptr(),
                scratch.cublaslt_workspace.len(),
                stream.raw(),
            )
        },
        "run tiled FP32-cache prefill attention",
    )
}

/// Runs causal position-block attention with a `q8` KV cache.
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_q8(
    handle: &CublasLt,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> Result<()> {
    let cache_bytes = checked_product(
        shape.cache_elements() / 32,
        34,
        "q8 prefill attention cache bytes",
    )?;
    check_prefill_attention(
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
        cache_bytes,
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked q8 cache and compact conversion scratch cover the
    // current context. The C++ wrapper bounds every tiled matmul.
    check_cublas(
        unsafe {
            ffi::ie_cublaslt_attention_prefill_q8(
                handle.raw(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                query.const_ptr(),
                output.mut_ptr(),
                scratch.converted_query.mut_ptr(),
                scratch.scores.mut_ptr(),
                scratch.probabilities.mut_ptr(),
                scratch.head_output.mut_ptr(),
                scratch.converted_kv.mut_ptr(),
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                start_position,
                tokens,
                scratch.cublaslt_workspace.mut_ptr(),
                scratch.cublaslt_workspace.len(),
                stream.raw(),
            )
        },
        "run tiled q8-cache prefill attention",
    )
}

#[allow(clippy::too_many_arguments)]
fn check_prefill_attention<T: DeviceCopy>(
    handle: &CublasLt,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    output: &DeviceBuffer<f32>,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &PrefillScratch,
    cache_elements: usize,
) -> Result<()> {
    let query_elements = validate_prefill_attention_shape(scratch, shape, start_position, tokens)?;
    validate_prefill_attention_lengths(
        query,
        key_cache,
        value_cache,
        output,
        query_elements,
        cache_elements,
    )?;
    same_devices(
        stream.device,
        &[
            handle.device,
            query.device,
            key_cache.device,
            value_cache.device,
            output.device,
            scratch.converted_query.device,
            scratch.scores.device,
            scratch.probabilities.device,
            scratch.head_output.device,
            scratch.converted_kv.device,
            scratch.cublaslt_workspace.device,
        ],
    )
}

fn validate_prefill_attention_shape(
    scratch: &PrefillScratch,
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Result<usize> {
    let context_length = start_position
        .checked_add(tokens)
        .ok_or(Error::SizeOverflow {
            field: "prefill attention context length",
        })?;
    if context_length > shape.max_context()
        || context_length > scratch.plan.context_tokens()
        || tokens > scratch.plan.chunk_tokens()
    {
        return Err(Error::ContextLength {
            context_length,
            max_context: shape.max_context().min(scratch.plan.context_tokens()),
        });
    }
    if shape.n_head() != scratch.plan.n_head()
        || shape.n_head_kv() != scratch.plan.n_head_kv()
        || shape.head_dim() != scratch.plan.head_dim()
    {
        return Err(Error::SizeMismatch {
            name: "prefill attention plan",
            expected: scratch.plan.n_head() * scratch.plan.head_dim(),
            actual: shape.n_head() * shape.head_dim(),
        });
    }
    checked_product(
        tokens,
        shape.query_elements(),
        "prefill attention query elements",
    )
}

fn validate_prefill_attention_lengths<T: DeviceCopy>(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    output: &DeviceBuffer<f32>,
    query_elements: usize,
    cache_elements: usize,
) -> Result<()> {
    exact_len("prefill attention query", query_elements, query.len())?;
    exact_len("prefill attention output", query_elements, output.len())?;
    exact_len(
        "prefill attention key cache",
        cache_elements,
        key_cache.len(),
    )?;
    exact_len(
        "prefill attention value cache",
        cache_elements,
        value_cache.len(),
    )
}

/// Gathers one quantized embedding row per prompt token.
pub fn embedding_gather_batch(
    stream: &Stream,
    table: &DeviceBuffer<u8>,
    rows: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    tokens: usize,
) -> Result<()> {
    exact_len("prefill embedding table", shape.bytes(), table.len())?;
    exact_len("prefill embedding rows", tokens, rows.len())?;
    exact_len(
        "prefill embedding output",
        checked_product(tokens, shape.columns(), "prefill embedding output")?,
        output.len(),
    )?;
    same_devices(stream.device, &[table.device, rows.device, output.device])?;
    activate_device(stream.device)?;
    let code = match shape.format() {
        QuantFormat::Q4K => {
            // SAFETY: The checked table, row list, and output cover all tokens.
            unsafe {
                ffi::ie_launch_embedding_q4_k_batch(
                    table.const_ptr(),
                    rows.const_ptr(),
                    output.mut_ptr(),
                    shape.rows(),
                    shape.columns(),
                    tokens,
                    stream.raw(),
                )
            }
        }
        QuantFormat::Q6K => {
            // SAFETY: The checked table, row list, and output cover all tokens.
            unsafe {
                ffi::ie_launch_embedding_q6_k_batch(
                    table.const_ptr(),
                    rows.const_ptr(),
                    output.mut_ptr(),
                    shape.rows(),
                    shape.columns(),
                    tokens,
                    stream.raw(),
                )
            }
        }
    };
    check(code, "launch prefill embedding gather")
}

/// Copies one FP32 matrix row into an exact output vector.
pub fn copy_f32_row(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    row: usize,
    columns: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    exact_len("copied FP32 row output", columns, output.len())?;
    let rows = input.len() / columns;
    if row >= rows || !input.len().is_multiple_of(columns) {
        return Err(Error::RowOutOfBounds { row, rows });
    }
    same_devices(stream.device, &[input.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: `row * columns..(row + 1) * columns` lies in the checked input.
    check(
        unsafe {
            ffi::ie_launch_copy_f32_row(
                input.const_ptr(),
                output.mut_ptr(),
                row,
                columns,
                stream.raw(),
            )
        },
        "launch FP32 row copy",
    )
}

/// Selects a Q4_K counter-free probe geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Q4KProbeGeometry {
    LoadsOnlyFourWarpsOneRow = 0,
    TwoWarpsOneRow = 1,
    ThreeWarpsOneRow = 2,
    FourWarpsOneRow = 3,
    TwoWarpsTwoRows = 4,
    ThreeWarpsTwoRows = 5,
    FourWarpsTwoRows = 6,
    TwoWarpsFourRows = 7,
    ThreeWarpsFourRows = 8,
    FourWarpsFourRows = 9,
    ProductionFourWarpsOneRow = 10,
    WideFourWarpsOneRow = 11,
    TwoRowsPerWarpGroup = 12,
    RowInterleavedFourWarpsOneRow = 13,
    SplitKTwoCtasPerRow = 14,
    GgufFourWarpsOneRow = 15,
    TwoBlockIlpFourWarpsOneRow = 16,
    OneWarpFourRows = 17,
    GgufWideFourWarpsOneRow = 18,
    GgufWideOneWarpFourRows = 19,
}

impl GemvScratch {
    /// Allocates q8_1 storage for consecutive verifier positions.
    pub fn new_multi(context: &Context, columns: usize, positions: usize) -> Result<Self> {
        if positions == 0 || positions > 8 {
            return Err(Error::TooLarge {
                field: "verifier positions",
                value: positions,
                maximum: 8,
            });
        }
        if !columns.is_multiple_of(Q8_1_BLOCK_ELEMENTS) {
            return Err(Error::NotDivisible {
                field: "verifier columns",
                value: columns,
                divisor: Q8_1_BLOCK_ELEMENTS,
            });
        }
        let blocks = checked_product(
            columns / Q8_1_BLOCK_ELEMENTS,
            positions,
            "verifier q8_1 blocks",
        )?;
        Ok(Self {
            quantized_input: context.alloc(checked_product(
                blocks,
                Q8_1_BLOCK_BYTES,
                "verifier q8_1 bytes",
            )?)?,
            quantized_sums: context.alloc(blocks)?,
            epilogue_ready: context.alloc(1)?,
            columns,
        })
    }

    /// Allocates one 36-byte q8_1 block per 32 input values.
    pub fn new(context: &Context, shape: QuantizedMatrixShape) -> Result<Self> {
        let blocks =
            shape
                .columns()
                .checked_div(Q8_1_BLOCK_ELEMENTS)
                .ok_or(Error::SizeOverflow {
                    field: "q8_1 activation scratch blocks",
                })?;
        let bytes = blocks
            .checked_mul(Q8_1_BLOCK_BYTES)
            .ok_or(Error::SizeOverflow {
                field: "q8_1 activation scratch bytes",
            })?;
        Ok(Self {
            quantized_input: context.alloc(bytes)?,
            quantized_sums: context.alloc(blocks)?,
            epilogue_ready: context.copy_to_device(&vec![0_u32; blocks])?,
            columns: shape.columns(),
        })
    }
}

/// Quantizes one activation for repeated Q4_K probe launches.
pub fn prepare_q4_k_probe(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
) -> Result<()> {
    quantize_q8_1(stream, input, scratch)
}

/// Launches one Q4_K geometry probe against a selected matrix in a weight ring.
pub fn launch_q4_k_probe(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_set: usize,
    geometry: Q4KProbeGeometry,
) -> Result<()> {
    if shape.format() != QuantFormat::Q4K {
        return Err(Error::SizeMismatch {
            name: "Q4_K probe format marker",
            expected: QuantFormat::Q4K.block_bytes(),
            actual: shape.format().block_bytes(),
        });
    }
    let output_required = if geometry == Q4KProbeGeometry::SplitKTwoCtasPerRow {
        shape.rows().checked_mul(2).ok_or(Error::SizeOverflow {
            field: "Q4_K split probe output elements",
        })?
    } else {
        shape.rows()
    };
    if output.len() < output_required {
        return Err(Error::SizeMismatch {
            name: "Q4_K probe output",
            expected: output_required,
            actual: output.len(),
        });
    }
    if scratch.columns != shape.columns() {
        return Err(Error::SizeMismatch {
            name: "Q4_K probe q8_1 scratch columns",
            expected: shape.columns(),
            actual: scratch.columns,
        });
    }
    let required = shape
        .bytes()
        .checked_mul(weight_set + 1)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K probe weight ring bytes",
        })?;
    if weights.len() < required {
        return Err(Error::SizeMismatch {
            name: "Q4_K probe weight ring",
            expected: required,
            actual: weights.len(),
        });
    }
    same_devices(
        stream.device,
        &[
            weights.device,
            output.device,
            scratch.quantized_input.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The selected complete matrix, q8_1 input, and output cover the
    // checked launch geometry.
    check(
        unsafe {
            ffi::ie_launch_q4_k_gemv_probe(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                weight_set,
                geometry as i32,
                stream.raw(),
            )
        },
        "launch Q4_K GEMV probe",
    )
}

/// Streams every Q4_K matrix in a weight ring through one persistent launch.
pub fn launch_q4_k_ring_probe(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<()> {
    if shape.format() != QuantFormat::Q4K {
        return Err(Error::SizeMismatch {
            name: "Q4_K ring probe format marker",
            expected: QuantFormat::Q4K.block_bytes(),
            actual: shape.format().block_bytes(),
        });
    }
    if weight_sets == 0 {
        return Err(Error::Zero {
            field: "Q4_K ring probe weight sets",
        });
    }
    let output_required = shape
        .rows()
        .checked_mul(weight_sets)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K ring probe output elements",
        })?;
    if output.len() < output_required {
        return Err(Error::SizeMismatch {
            name: "Q4_K ring probe output",
            expected: output_required,
            actual: output.len(),
        });
    }
    if scratch.columns != shape.columns() {
        return Err(Error::SizeMismatch {
            name: "Q4_K ring probe q8_1 scratch columns",
            expected: shape.columns(),
            actual: scratch.columns,
        });
    }
    let required = shape
        .bytes()
        .checked_mul(weight_sets)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K ring probe weight bytes",
        })?;
    if weights.len() < required {
        return Err(Error::SizeMismatch {
            name: "Q4_K ring probe weight ring",
            expected: required,
            actual: weights.len(),
        });
    }
    same_devices(
        stream.device,
        &[
            weights.device,
            output.device,
            scratch.quantized_input.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked ring contains `weight_sets` complete matrices. The
    // output has one element for every matrix row.
    check(
        unsafe {
            ffi::ie_launch_q4_k_gemv_ring_probe(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                weight_sets,
                stream.raw(),
            )
        },
        "launch persistent Q4_K ring probe",
    )
}

/// Launches adjacent 4096-row Q4_K probes with an L2 apron between them.
#[allow(clippy::too_many_arguments)]
pub fn launch_q4_k_apron_pair_probe(
    stream: &Stream,
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    third_weights: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_set: usize,
    apron_bytes: usize,
) -> Result<()> {
    validate_apron_probe_inputs(ApronProbeInputs {
        first_weights,
        second_weights,
        third_weights,
        output,
        scratch,
        shape,
        weight_set,
        apron_bytes,
    })?;
    same_devices(
        stream.device,
        &[
            first_weights.device,
            second_weights.device,
            third_weights.device,
            output.device,
            scratch.quantized_input.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: Both selected matrices, the q8_1 input, and the output cover
    // the checked fixed probe geometry.
    check(
        unsafe {
            ffi::ie_launch_q4_k_apron_pair_probe(
                first_weights.const_ptr(),
                second_weights.const_ptr(),
                third_weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                weight_set,
                apron_bytes,
                stream.raw(),
            )
        },
        "launch Q4_K apron pair probe",
    )
}

struct ApronProbeInputs<'a> {
    first_weights: &'a DeviceBuffer<u8>,
    second_weights: &'a DeviceBuffer<u8>,
    third_weights: &'a DeviceBuffer<u8>,
    output: &'a DeviceBuffer<f32>,
    scratch: &'a GemvScratch,
    shape: QuantizedMatrixShape,
    weight_set: usize,
    apron_bytes: usize,
}

fn validate_apron_probe_inputs(inputs: ApronProbeInputs<'_>) -> Result<QuantizedMatrixShape> {
    let next_shape = validate_apron_shape(inputs.shape, inputs.apron_bytes)?;
    validate_apron_output(inputs.output, inputs.scratch, inputs.shape, next_shape)?;
    validate_apron_weight_rings(
        inputs.first_weights,
        inputs.second_weights,
        inputs.third_weights,
        inputs.shape,
        next_shape,
        inputs.weight_set,
    )?;
    Ok(next_shape)
}

fn validate_apron_shape(
    shape: QuantizedMatrixShape,
    apron_bytes: usize,
) -> Result<QuantizedMatrixShape> {
    const MAX_APRON_BYTES: usize = 4 * 1024 * 1024;
    if shape.format() != QuantFormat::Q4K || shape.rows() != 4096 || shape.columns() != 4096 {
        return Err(Error::SizeMismatch {
            name: "Q4_K apron probe matrix elements",
            expected: 4096 * 4096,
            actual: shape.rows() * shape.columns(),
        });
    }
    if apron_bytes > MAX_APRON_BYTES || !apron_bytes.is_multiple_of(16) {
        return Err(Error::SizeMismatch {
            name: "Q4_K apron probe bytes",
            expected: MAX_APRON_BYTES,
            actual: apron_bytes,
        });
    }
    QuantizedMatrixShape::new(12288, 4096, QuantFormat::Q4K)
}

fn validate_apron_output(
    output: &DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    next_shape: QuantizedMatrixShape,
) -> Result<()> {
    let output_required = next_shape
        .rows()
        .checked_mul(2)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K apron probe output elements",
        })?;
    if output.len() < output_required {
        return Err(Error::SizeMismatch {
            name: "Q4_K apron probe output",
            expected: output_required,
            actual: output.len(),
        });
    }
    if scratch.columns != shape.columns() {
        return Err(Error::SizeMismatch {
            name: "Q4_K apron probe q8_1 scratch columns",
            expected: shape.columns(),
            actual: scratch.columns,
        });
    }
    Ok(())
}

fn validate_apron_weight_rings(
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    third_weights: &DeviceBuffer<u8>,
    shape: QuantizedMatrixShape,
    next_shape: QuantizedMatrixShape,
    weight_set: usize,
) -> Result<()> {
    let first_required = shape
        .bytes()
        .checked_mul(weight_set + 1)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K apron probe weight ring bytes",
        })?;
    let next_required =
        next_shape
            .bytes()
            .checked_mul(weight_set + 1)
            .ok_or(Error::SizeOverflow {
                field: "Q4_K apron next weight ring bytes",
            })?;
    for (name, weights, required) in [
        (
            "Q4_K apron first weight ring",
            first_weights,
            first_required,
        ),
        (
            "Q4_K apron second weight ring",
            second_weights,
            next_required,
        ),
        ("Q4_K apron third weight ring", third_weights, next_required),
    ] {
        if weights.len() < required {
            return Err(Error::SizeMismatch {
                name,
                expected: required,
                actual: weights.len(),
            });
        }
    }
    Ok(())
}

/// Device scratch for partial online-softmax states.
#[derive(Debug)]
pub struct AttentionScratch {
    partial_max: DeviceBuffer<f32>,
    partial_sum: DeviceBuffer<f32>,
    partial_output: DeviceBuffer<f32>,
    shape: AttentionShape,
    positions: usize,
}

impl AttentionScratch {
    /// Allocates one partial state per 64-position tile, capped at 64 splits.
    pub fn new(context: &Context, shape: AttentionShape) -> Result<Self> {
        Self::new_multi(context, shape, 1)
    }

    /// Allocates independent partial states for consecutive verifier queries.
    pub fn new_multi(context: &Context, shape: AttentionShape, positions: usize) -> Result<Self> {
        let partial_rows = checked_product(
            shape.partial_rows()?,
            positions,
            "verifier attention partial rows",
        )?;
        let partial_elements = checked_product(
            shape.partial_elements()?,
            positions,
            "verifier attention partial elements",
        )?;
        Ok(Self {
            partial_max: context.alloc(partial_rows)?,
            partial_sum: context.alloc(partial_rows)?,
            partial_output: context.alloc(partial_elements)?,
            shape,
            positions,
        })
    }
}

/// Device storage for one model's RoPE frequencies and current FP32 table.
#[derive(Debug)]
pub struct RopeScratch {
    inverse_frequencies: DeviceBuffer<f64>,
    table: DeviceBuffer<f32>,
    head_dim: usize,
    adjacent_pairs: bool,
}

impl RopeScratch {
    /// Precomputes one inverse frequency per GPT-NeoX half pair.
    pub fn new(
        context: &Context,
        head_dim: usize,
        theta: f32,
        frequency_factors: Option<&[f32]>,
        adjacent_pairs: bool,
    ) -> Result<Self> {
        let pairs = validate_rope_parameters(head_dim, theta, frequency_factors)?;
        let inverse_frequencies =
            rope_inverse_frequencies(pairs, head_dim, theta, frequency_factors);
        let table_elements = checked_product(head_dim, 8, "verifier RoPE table elements")?;
        Ok(Self {
            inverse_frequencies: context.copy_to_device(&inverse_frequencies)?,
            table: context.alloc(table_elements)?,
            head_dim,
            adjacent_pairs,
        })
    }

    /// Fills the FP32 sine and cosine table for one host position.
    pub fn prepare(&mut self, stream: &Stream, position: usize) -> Result<()> {
        same_devices(
            stream.device,
            &[self.inverse_frequencies.device, self.table.device],
        )?;
        activate_device(stream.device)?;
        let pairs = self.head_dim / 2;
        // SAFETY: The input and output buffers each cover `pairs` frequency
        // entries. The table stores one adjacent cosine and sine pair.
        check(
            unsafe {
                ffi::ie_launch_prepare_rope_table(
                    self.inverse_frequencies.const_ptr(),
                    self.table.mut_ptr(),
                    pairs,
                    position,
                    stream.raw(),
                )
            },
            "prepare RoPE table",
        )
    }

    /// Fills the FP32 sine and cosine table from one device position.
    pub fn prepare_device_position(
        &mut self,
        stream: &Stream,
        position: &DeviceBuffer<u32>,
    ) -> Result<()> {
        exact_len("RoPE position", 1, position.len())?;
        same_devices(
            stream.device,
            &[
                self.inverse_frequencies.device,
                self.table.device,
                position.device,
            ],
        )?;
        activate_device(stream.device)?;
        let pairs = self.head_dim / 2;
        // SAFETY: The position contains one live scalar. Both table buffers
        // remain live through stream execution.
        check(
            unsafe {
                ffi::ie_launch_prepare_rope_table_device_position(
                    self.inverse_frequencies.const_ptr(),
                    self.table.mut_ptr(),
                    pairs,
                    position.const_ptr(),
                    stream.raw(),
                )
            },
            "prepare device-position RoPE table",
        )
    }

    fn check(&self, stream: &Stream, head_dim: usize) -> Result<()> {
        if self.head_dim != head_dim {
            return Err(Error::SizeMismatch {
                name: "RoPE scratch head dimension",
                expected: head_dim,
                actual: self.head_dim,
            });
        }
        same_device(stream.device, self.table.device)
    }
}

fn validate_rope_parameters(
    head_dim: usize,
    theta: f32,
    frequency_factors: Option<&[f32]>,
) -> Result<usize> {
    validate_positive("RoPE theta", theta)?;
    if head_dim == 0 {
        return Err(Error::Zero {
            field: "RoPE head dimension",
        });
    }
    if !head_dim.is_multiple_of(2) {
        return Err(Error::NotDivisible {
            field: "RoPE head dimension",
            value: head_dim,
            divisor: 2,
        });
    }
    if head_dim > 512 {
        return Err(Error::TooLarge {
            field: "RoPE head dimension",
            value: head_dim,
            maximum: 512,
        });
    }
    let pairs = head_dim / 2;
    if let Some(factors) = frequency_factors {
        exact_len("RoPE frequency factors", pairs, factors.len())?;
        for &factor in factors {
            validate_positive("RoPE frequency factor", factor)?;
        }
    }
    Ok(pairs)
}

fn rope_inverse_frequencies(
    pairs: usize,
    head_dim: usize,
    theta: f32,
    frequency_factors: Option<&[f32]>,
) -> Vec<f64> {
    (0..pairs)
        .map(|pair| {
            let factor = frequency_factors
                .map(|factors| f64::from(factors[pair]))
                .unwrap_or(1.0);
            f64::from(theta).powf(-2.0 * pair as f64 / head_dim as f64) / factor
        })
        .collect()
}

/// Device scratch for a two-stage deterministic argmax.
#[derive(Debug)]
pub struct ArgmaxScratch {
    partial_values: DeviceBuffer<f32>,
    partial_indices: DeviceBuffer<u32>,
    elements: usize,
}

impl ArgmaxScratch {
    /// Allocates one partial pair per 1,024 input values.
    pub fn new(context: &Context, elements: usize) -> Result<Self> {
        let blocks = argmax_blocks(elements)?;
        Ok(Self {
            partial_values: context.alloc(blocks)?,
            partial_indices: context.alloc(blocks)?,
            elements,
        })
    }
}

/// Launches Q4_K by q8_1 row-major GEMV.
pub fn gemv_q4_k(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
) -> Result<()> {
    check_gemv(
        stream,
        weights,
        input,
        output,
        scratch,
        shape,
        QuantFormat::Q4K,
    )?;
    quantize_q8_1(stream, input, scratch)?;
    activate_device(stream.device)?;
    // SAFETY: The checked matrix and q8_1 scratch cover the input and output.
    check(
        unsafe {
            ffi::ie_launch_q4_k_gemv(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                scratch.quantized_sums.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                stream.raw(),
            )
        },
        "launch Q4_K GEMV",
    )
}

/// Launches Q4_K GEMV and adds an optional residual in its epilogue.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_k_residual(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    residual: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
    input_prepared: bool,
) -> Result<()> {
    check_gemv(
        stream,
        weights,
        input,
        output,
        scratch,
        shape,
        QuantFormat::Q4K,
    )?;
    if let Some(residual) = residual {
        exact_len("Q4_K GEMV residual", shape.rows(), residual.len())?;
        same_device(stream.device, residual.device)?;
    }
    if !input_prepared {
        quantize_q8_1(stream, input, scratch)?;
    }
    activate_device(stream.device)?;
    let code = if let Some(residual) = residual {
        // SAFETY: The checked matrix, q8_1 scratch, residual, and output cover
        // the launch dimensions.
        unsafe {
            ffi::ie_launch_q4_k_gemv_residual(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                residual.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                stream.raw(),
            )
        }
    } else {
        // SAFETY: The checked matrix and q8_1 scratch cover the launch ranges.
        unsafe {
            ffi::ie_launch_q4_k_gemv(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                scratch.quantized_sums.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                stream.raw(),
            )
        }
    };
    check(code, "launch Q4_K GEMV epilogue")
}

/// Launches Q6_K by q8_1 row-major GEMV.
pub fn gemv_q6_k(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
) -> Result<()> {
    check_gemv(
        stream,
        weights,
        input,
        output,
        scratch,
        shape,
        QuantFormat::Q6K,
    )?;
    quantize_q8_1(stream, input, scratch)?;
    activate_device(stream.device)?;
    // SAFETY: The checked matrix and q8_1 scratch cover the input and output.
    check(
        unsafe {
            ffi::ie_launch_q6_k_gemv(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                stream.raw(),
            )
        },
        "launch Q6_K GEMV",
    )
}

/// Launches Q6_K GEMV and adds an optional residual in its epilogue.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q6_k_residual(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    residual: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
    input_prepared: bool,
) -> Result<()> {
    check_gemv(
        stream,
        weights,
        input,
        output,
        scratch,
        shape,
        QuantFormat::Q6K,
    )?;
    if let Some(residual) = residual {
        exact_len("Q6_K GEMV residual", shape.rows(), residual.len())?;
        same_device(stream.device, residual.device)?;
    }
    if !input_prepared {
        quantize_q8_1(stream, input, scratch)?;
    }
    activate_device(stream.device)?;
    let code = if let Some(residual) = residual {
        // SAFETY: The checked matrix, q8_1 scratch, residual, and output cover
        // the launch dimensions.
        unsafe {
            ffi::ie_launch_q6_k_gemv_residual(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                residual.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                stream.raw(),
            )
        }
    } else {
        // SAFETY: The checked matrix and q8_1 scratch cover the launch.
        unsafe {
            ffi::ie_launch_q6_k_gemv(
                weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                stream.raw(),
            )
        }
    };
    check(code, "launch Q6_K GEMV epilogue")
}

/// Applies one Q4_K or Q6_K matrix to consecutive verifier positions.
#[allow(clippy::too_many_arguments)]
pub fn verify_gemv(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    residual: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
    positions: usize,
) -> Result<()> {
    verify_gemv_inner(
        stream, weights, input, residual, output, scratch, shape, positions, false,
    )
}

/// Applies a verifier matrix to q8_1 rows prepared in `scratch`.
#[allow(clippy::too_many_arguments)]
pub fn verify_gemv_prepared(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    residual: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
    positions: usize,
) -> Result<()> {
    verify_gemv_inner(
        stream, weights, input, residual, output, scratch, shape, positions, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_gemv_inner(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    residual: Option<&DeviceBuffer<f32>>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
    positions: usize,
    input_prepared: bool,
) -> Result<()> {
    let input_elements = validate_verify_gemv_inputs(
        stream, weights, input, residual, output, scratch, shape, positions,
    )?;
    if let Some(residual) = residual {
        same_device(stream.device, residual.device)?;
    }
    activate_device(stream.device)?;
    if !input_prepared {
        quantize_verifier_input(stream, input, scratch, input_elements)?;
    }
    let residual = residual.map_or(ptr::null(), DeviceBuffer::const_ptr);
    check(
        launch_verify_gemv(stream, weights, output, scratch, shape, positions, residual),
        "launch verifier GEMV",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_verify_gemv_inputs(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    residual: Option<&DeviceBuffer<f32>>,
    output: &DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    positions: usize,
) -> Result<usize> {
    validate_verifier_positions(positions)?;
    let input_elements = checked_product(shape.columns(), positions, "verifier input elements")?;
    let output_elements = checked_product(shape.rows(), positions, "verifier output elements")?;
    validate_verifier_lengths(
        weights,
        input,
        output,
        residual,
        shape,
        input_elements,
        output_elements,
    )?;
    if scratch.columns != shape.columns() {
        return Err(Error::SizeMismatch {
            name: "verifier q8_1 scratch columns",
            expected: shape.columns(),
            actual: scratch.columns,
        });
    }
    same_devices(
        stream.device,
        &[
            weights.device,
            input.device,
            output.device,
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    Ok(input_elements)
}

fn validate_verifier_positions(positions: usize) -> Result<()> {
    if positions == 0 || positions > 8 {
        return Err(Error::TooLarge {
            field: "verifier positions",
            value: positions,
            maximum: 8,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_verifier_lengths(
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    residual: Option<&DeviceBuffer<f32>>,
    shape: QuantizedMatrixShape,
    input_elements: usize,
    output_elements: usize,
) -> Result<()> {
    exact_len("verifier weights", shape.bytes(), weights.len())?;
    exact_len("verifier input", input_elements, input.len())?;
    exact_len("verifier output", output_elements, output.len())?;
    if let Some(residual) = residual {
        exact_len("verifier residual", output_elements, residual.len())?;
    }
    Ok(())
}

fn quantize_verifier_input(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    input_elements: usize,
) -> Result<()> {
    check(
        // SAFETY: The checked dense input and q8_1 buffers cover all
        // position-major activation rows.
        unsafe {
            ffi::ie_launch_quantize_q8_1(
                input.const_ptr(),
                scratch.quantized_input.mut_ptr(),
                scratch.quantized_sums.mut_ptr(),
                input_elements,
                stream.raw(),
            )
        },
        "quantize verifier activations",
    )
}

fn launch_verify_gemv(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    positions: usize,
    residual: *const f32,
) -> i32 {
    match shape.format() {
        QuantFormat::Q4K => {
            // SAFETY: The matrix, q8_1 rows, residual, and output match the
            // checked position-major dimensions.
            unsafe {
                ffi::ie_launch_q4_k_gemv_multi(
                    weights.const_ptr(),
                    scratch.quantized_input.const_ptr(),
                    residual,
                    output.mut_ptr(),
                    shape.rows(),
                    shape.columns(),
                    positions,
                    stream.raw(),
                )
            }
        }
        QuantFormat::Q6K => {
            // SAFETY: The matrix, q8_1 rows, residual, and output match the
            // checked position-major dimensions.
            unsafe {
                ffi::ie_launch_q6_k_gemv_multi(
                    weights.const_ptr(),
                    scratch.quantized_input.const_ptr(),
                    residual,
                    output.mut_ptr(),
                    shape.rows(),
                    shape.columns(),
                    positions,
                    stream.raw(),
                )
            }
        }
    }
}

/// Applies three verifier projections through one combined output-row grid.
#[allow(clippy::too_many_arguments)]
pub fn verify_gemv_triple(
    stream: &Stream,
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    third_weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    first_output: &mut DeviceBuffer<f32>,
    second_output: &mut DeviceBuffer<f32>,
    third_output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    first_shape: QuantizedMatrixShape,
    second_shape: QuantizedMatrixShape,
    third_shape: QuantizedMatrixShape,
    positions: usize,
) -> Result<()> {
    verify_gemv_group(
        stream,
        first_weights,
        second_weights,
        Some(third_weights),
        input,
        first_output,
        second_output,
        Some(third_output),
        scratch,
        first_shape,
        second_shape,
        Some(third_shape),
        positions,
    )
}

/// Applies two verifier projections through one combined output-row grid.
#[allow(clippy::too_many_arguments)]
pub fn verify_gemv_pair(
    stream: &Stream,
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    first_output: &mut DeviceBuffer<f32>,
    second_output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    first_shape: QuantizedMatrixShape,
    second_shape: QuantizedMatrixShape,
    positions: usize,
) -> Result<()> {
    verify_gemv_group(
        stream,
        first_weights,
        second_weights,
        None,
        input,
        first_output,
        second_output,
        None,
        scratch,
        first_shape,
        second_shape,
        None,
        positions,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_gemv_group(
    stream: &Stream,
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    third_weights: Option<&DeviceBuffer<u8>>,
    input: &DeviceBuffer<f32>,
    first_output: &mut DeviceBuffer<f32>,
    second_output: &mut DeviceBuffer<f32>,
    third_output: Option<&mut DeviceBuffer<f32>>,
    scratch: &mut GemvScratch,
    first_shape: QuantizedMatrixShape,
    second_shape: QuantizedMatrixShape,
    third_shape: Option<QuantizedMatrixShape>,
    positions: usize,
) -> Result<()> {
    let input_elements = validate_verify_gemv_group(
        first_weights,
        second_weights,
        input,
        first_output,
        second_output,
        scratch,
        first_shape,
        second_shape,
        third_shape,
        positions,
    )?;
    let third =
        verify_gemv_group_third(stream, third_weights, third_output, third_shape, positions)?;
    same_devices(
        stream.device,
        &[
            first_weights.device,
            second_weights.device,
            input.device,
            first_output.device,
            second_output.device,
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    activate_device(stream.device)?;
    check(
        // SAFETY: The checked dense input and q8_1 buffers cover all rows.
        unsafe {
            ffi::ie_launch_quantize_q8_1(
                input.const_ptr(),
                scratch.quantized_input.mut_ptr(),
                scratch.quantized_sums.mut_ptr(),
                input_elements,
                stream.raw(),
            )
        },
        "quantize verifier projection group input",
    )?;
    check(
        launch_verify_gemv_group(
            stream,
            first_weights,
            second_weights,
            third,
            first_output,
            second_output,
            scratch,
            first_shape,
            second_shape,
            positions,
        ),
        "launch verifier projection group",
    )
}

struct VerifyGemvGroupThird {
    weights: *const u8,
    output: *mut f32,
    rows: usize,
    q4: i32,
}

#[allow(clippy::too_many_arguments)]
fn validate_verify_gemv_group(
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    first_output: &DeviceBuffer<f32>,
    second_output: &DeviceBuffer<f32>,
    scratch: &GemvScratch,
    first_shape: QuantizedMatrixShape,
    second_shape: QuantizedMatrixShape,
    third_shape: Option<QuantizedMatrixShape>,
    positions: usize,
) -> Result<usize> {
    validate_verify_gemv_group_shapes(first_shape, second_shape, third_shape, positions)?;
    let input_elements = checked_product(
        first_shape.columns(),
        positions,
        "verifier projection group input elements",
    )?;
    validate_verify_gemv_group_lengths(
        first_weights,
        second_weights,
        input,
        first_output,
        second_output,
        first_shape,
        second_shape,
        positions,
        input_elements,
    )?;
    if scratch.columns != first_shape.columns() {
        return Err(Error::SizeMismatch {
            name: "verifier projection group scratch columns",
            expected: first_shape.columns(),
            actual: scratch.columns,
        });
    }
    Ok(input_elements)
}

fn validate_verify_gemv_group_shapes(
    first_shape: QuantizedMatrixShape,
    second_shape: QuantizedMatrixShape,
    third_shape: Option<QuantizedMatrixShape>,
    positions: usize,
) -> Result<()> {
    if positions == 0 || positions > 8 {
        return Err(Error::TooLarge {
            field: "verifier positions",
            value: positions,
            maximum: 8,
        });
    }
    if second_shape.columns() != first_shape.columns()
        || third_shape.is_some_and(|shape| shape.columns() != first_shape.columns())
    {
        return Err(Error::SizeMismatch {
            name: "verifier projection group columns",
            expected: first_shape.columns(),
            actual: second_shape.columns(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_verify_gemv_group_lengths(
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    first_output: &DeviceBuffer<f32>,
    second_output: &DeviceBuffer<f32>,
    first_shape: QuantizedMatrixShape,
    second_shape: QuantizedMatrixShape,
    positions: usize,
    input_elements: usize,
) -> Result<()> {
    exact_len(
        "first verifier group weights",
        first_shape.bytes(),
        first_weights.len(),
    )?;
    exact_len(
        "second verifier group weights",
        second_shape.bytes(),
        second_weights.len(),
    )?;
    exact_len(
        "verifier projection group input",
        input_elements,
        input.len(),
    )?;
    exact_len(
        "first verifier group output",
        checked_product(first_shape.rows(), positions, "first verifier group output")?,
        first_output.len(),
    )?;
    exact_len(
        "second verifier group output",
        checked_product(
            second_shape.rows(),
            positions,
            "second verifier group output",
        )?,
        second_output.len(),
    )?;
    Ok(())
}

fn verify_gemv_group_third(
    stream: &Stream,
    third_weights: Option<&DeviceBuffer<u8>>,
    third_output: Option<&mut DeviceBuffer<f32>>,
    third_shape: Option<QuantizedMatrixShape>,
    positions: usize,
) -> Result<VerifyGemvGroupThird> {
    match (third_weights, third_output, third_shape) {
        (Some(weights), Some(output), Some(shape)) => {
            exact_len("third verifier group weights", shape.bytes(), weights.len())?;
            exact_len(
                "third verifier group output",
                checked_product(shape.rows(), positions, "third verifier group output")?,
                output.len(),
            )?;
            same_devices(stream.device, &[weights.device, output.device])?;
            Ok(VerifyGemvGroupThird {
                weights: weights.const_ptr(),
                output: output.mut_ptr(),
                rows: shape.rows(),
                q4: i32::from(shape.format() == QuantFormat::Q4K),
            })
        }
        (None, None, None) => Ok(VerifyGemvGroupThird {
            weights: ptr::null(),
            output: ptr::null_mut(),
            rows: 0,
            q4: 0,
        }),
        _ => Err(Error::SizeMismatch {
            name: "third verifier projection",
            expected: 0,
            actual: 1,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_verify_gemv_group(
    stream: &Stream,
    first_weights: &DeviceBuffer<u8>,
    second_weights: &DeviceBuffer<u8>,
    third: VerifyGemvGroupThird,
    first_output: &mut DeviceBuffer<f32>,
    second_output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    first_shape: QuantizedMatrixShape,
    second_shape: QuantizedMatrixShape,
    positions: usize,
) -> i32 {
    unsafe {
        ffi::ie_launch_quant_gemv_group_multi(
            first_weights.const_ptr(),
            second_weights.const_ptr(),
            third.weights,
            scratch.quantized_input.const_ptr(),
            first_output.mut_ptr(),
            second_output.mut_ptr(),
            third.output,
            first_shape.rows(),
            second_shape.rows(),
            third.rows,
            first_shape.columns(),
            i32::from(first_shape.format() == QuantFormat::Q4K),
            i32::from(second_shape.format() == QuantFormat::Q4K),
            third.q4,
            positions,
            stream.raw(),
        )
    }
}

/// Launches two Q4_K projections after one q8_1 activation quantization.
#[allow(clippy::too_many_arguments)]
pub fn gemv_pair_q4_k(
    stream: &Stream,
    first_weights: &DeviceBuffer<u8>,
    first_shape: QuantizedMatrixShape,
    second_weights: &DeviceBuffer<u8>,
    second_shape: QuantizedMatrixShape,
    input: &DeviceBuffer<f32>,
    first_output: &mut DeviceBuffer<f32>,
    second_output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    input_prepared: bool,
) -> Result<()> {
    if first_shape.columns() != second_shape.columns() {
        return Err(Error::SizeMismatch {
            name: "paired GEMV columns",
            expected: first_shape.columns(),
            actual: second_shape.columns(),
        });
    }
    check_gemv(
        stream,
        first_weights,
        input,
        first_output,
        scratch,
        first_shape,
        QuantFormat::Q4K,
    )?;
    check_gemv(
        stream,
        second_weights,
        input,
        second_output,
        scratch,
        second_shape,
        QuantFormat::Q4K,
    )?;
    if !input_prepared {
        quantize_q8_1(stream, input, scratch)?;
    }
    activate_device(stream.device)?;
    // SAFETY: Both checked matrices share the checked q8_1 input. Each output
    // covers its matrix row count.
    check(
        unsafe {
            ffi::ie_launch_q4_k_gemv_pair(
                first_weights.const_ptr(),
                second_weights.const_ptr(),
                scratch.quantized_input.const_ptr(),
                scratch.quantized_sums.const_ptr(),
                first_output.mut_ptr(),
                second_output.mut_ptr(),
                first_shape.rows(),
                second_shape.rows(),
                first_shape.columns(),
                stream.raw(),
            )
        },
        "launch paired Q4_K GEMV",
    )
}

/// Fuses equal-row Q4_K gate and up projections with SwiGLU and q8_1 output.
#[allow(clippy::too_many_arguments)]
pub fn gemv_pair_swiglu_q4_k(
    stream: &Stream,
    gate_weights: &DeviceBuffer<u8>,
    gate_shape: QuantizedMatrixShape,
    up_weights: &DeviceBuffer<u8>,
    up_shape: QuantizedMatrixShape,
    input: &DeviceBuffer<f32>,
    gate_output: &mut DeviceBuffer<f32>,
    up_output: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    input_scratch: &mut GemvScratch,
    output_scratch: &mut GemvScratch,
    input_prepared: bool,
) -> Result<()> {
    validate_gemv_pair_swiglu(
        stream,
        gate_weights,
        gate_shape,
        up_weights,
        up_shape,
        input,
        gate_output,
        up_output,
        output,
        input_scratch,
        output_scratch,
    )?;
    if !input_prepared {
        quantize_q8_1(stream, input, input_scratch)?;
    }
    activate_device(stream.device)?;
    // SAFETY: Both checked matrices share the checked q8_1 input. Dense and
    // quantized outputs cover every equal output row.
    check(
        unsafe {
            ffi::ie_launch_q4_k_gemv_swiglu(
                gate_weights.const_ptr(),
                up_weights.const_ptr(),
                input_scratch.quantized_input.const_ptr(),
                gate_output.mut_ptr(),
                up_output.mut_ptr(),
                output.mut_ptr(),
                output_scratch.quantized_input.mut_ptr(),
                output_scratch.quantized_sums.mut_ptr(),
                output_scratch.epilogue_ready.mut_ptr(),
                gate_shape.rows(),
                gate_shape.columns(),
                stream.raw(),
            )
        },
        "launch fused Q4_K SwiGLU GEMV",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_gemv_pair_swiglu(
    stream: &Stream,
    gate_weights: &DeviceBuffer<u8>,
    gate_shape: QuantizedMatrixShape,
    up_weights: &DeviceBuffer<u8>,
    up_shape: QuantizedMatrixShape,
    input: &DeviceBuffer<f32>,
    gate_output: &DeviceBuffer<f32>,
    up_output: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    input_scratch: &GemvScratch,
    output_scratch: &GemvScratch,
) -> Result<()> {
    if gate_shape.rows() != up_shape.rows() {
        return Err(Error::SizeMismatch {
            name: "fused SwiGLU up rows",
            expected: gate_shape.rows(),
            actual: up_shape.rows(),
        });
    }
    if gate_shape.columns() != up_shape.columns() {
        return Err(Error::SizeMismatch {
            name: "fused SwiGLU up columns",
            expected: gate_shape.columns(),
            actual: up_shape.columns(),
        });
    }
    if !gate_shape.rows().is_multiple_of(Q8_1_BLOCK_ELEMENTS) {
        return Err(Error::NotDivisible {
            field: "fused SwiGLU rows",
            value: gate_shape.rows(),
            divisor: Q8_1_BLOCK_ELEMENTS,
        });
    }
    check_gemv(
        stream,
        gate_weights,
        input,
        gate_output,
        input_scratch,
        gate_shape,
        QuantFormat::Q4K,
    )?;
    check_gemv(
        stream,
        up_weights,
        input,
        up_output,
        input_scratch,
        up_shape,
        QuantFormat::Q4K,
    )?;
    exact_len("fused SwiGLU output", gate_shape.rows(), output.len())?;
    if output_scratch.columns != gate_shape.rows() {
        return Err(Error::SizeMismatch {
            name: "fused SwiGLU output q8_1 scratch columns",
            expected: gate_shape.rows(),
            actual: output_scratch.columns,
        });
    }
    same_devices(
        stream.device,
        &[
            output.device,
            output_scratch.quantized_input.device,
            output_scratch.quantized_sums.device,
            output_scratch.epilogue_ready.device,
        ],
    )
}

/// Launches three quantized projections that share one input.
#[allow(clippy::too_many_arguments)]
pub fn qkv_gemv(
    stream: &Stream,
    query_weights: &DeviceBuffer<u8>,
    query_shape: QuantizedMatrixShape,
    key_weights: &DeviceBuffer<u8>,
    key_shape: QuantizedMatrixShape,
    value_weights: &DeviceBuffer<u8>,
    value_shape: QuantizedMatrixShape,
    input: &DeviceBuffer<f32>,
    query: &mut DeviceBuffer<f32>,
    key: &mut DeviceBuffer<f32>,
    value: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    input_prepared: bool,
) -> Result<()> {
    validate_qkv_gemv(
        stream,
        query_weights,
        query_shape,
        key_weights,
        key_shape,
        value_weights,
        value_shape,
        input,
        query,
        key,
        value,
        scratch,
    )?;
    if !input_prepared {
        quantize_q8_1(stream, input, scratch)?;
    }
    activate_device(stream.device)?;
    check(
        launch_qkv_gemv(
            stream,
            query_weights,
            key_weights,
            value_weights,
            input,
            query,
            key,
            value,
            scratch,
            query_shape,
            key_shape,
            value_shape,
        ),
        "launch QKV GEMV",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_qkv_gemv(
    stream: &Stream,
    query_weights: &DeviceBuffer<u8>,
    query_shape: QuantizedMatrixShape,
    key_weights: &DeviceBuffer<u8>,
    key_shape: QuantizedMatrixShape,
    value_weights: &DeviceBuffer<u8>,
    value_shape: QuantizedMatrixShape,
    input: &DeviceBuffer<f32>,
    query: &DeviceBuffer<f32>,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    scratch: &GemvScratch,
) -> Result<()> {
    validate_qkv_shapes(query_shape, key_shape, value_shape)?;
    validate_qkv_buffers(
        query_weights,
        query_shape,
        key_weights,
        key_shape,
        value_weights,
        value_shape,
        input,
        query,
        key,
        value,
    )?;
    if scratch.columns != query_shape.columns() {
        return Err(Error::SizeMismatch {
            name: "QKV q8_1 scratch columns",
            expected: query_shape.columns(),
            actual: scratch.columns,
        });
    }
    same_devices(
        stream.device,
        &[
            query_weights.device,
            key_weights.device,
            value_weights.device,
            input.device,
            query.device,
            key.device,
            value.device,
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    Ok(())
}

fn validate_qkv_shapes(
    query_shape: QuantizedMatrixShape,
    key_shape: QuantizedMatrixShape,
    value_shape: QuantizedMatrixShape,
) -> Result<()> {
    if query_shape.columns() != key_shape.columns()
        || query_shape.columns() != value_shape.columns()
    {
        return Err(Error::SizeMismatch {
            name: "QKV shared columns",
            expected: query_shape.columns(),
            actual: key_shape.columns().min(value_shape.columns()),
        });
    }
    query_shape
        .rows()
        .checked_add(key_shape.rows())
        .and_then(|rows| rows.checked_add(value_shape.rows()))
        .ok_or(Error::SizeOverflow {
            field: "QKV output rows",
        })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_qkv_buffers(
    query_weights: &DeviceBuffer<u8>,
    query_shape: QuantizedMatrixShape,
    key_weights: &DeviceBuffer<u8>,
    key_shape: QuantizedMatrixShape,
    value_weights: &DeviceBuffer<u8>,
    value_shape: QuantizedMatrixShape,
    input: &DeviceBuffer<f32>,
    query: &DeviceBuffer<f32>,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
) -> Result<()> {
    exact_len("query weights", query_shape.bytes(), query_weights.len())?;
    exact_len("key weights", key_shape.bytes(), key_weights.len())?;
    exact_len("value weights", value_shape.bytes(), value_weights.len())?;
    exact_len("QKV input", query_shape.columns(), input.len())?;
    exact_len("query output", query_shape.rows(), query.len())?;
    exact_len("key output", key_shape.rows(), key.len())?;
    exact_len("value output", value_shape.rows(), value.len())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_qkv_gemv(
    stream: &Stream,
    query_weights: &DeviceBuffer<u8>,
    key_weights: &DeviceBuffer<u8>,
    value_weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    query: &mut DeviceBuffer<f32>,
    key: &mut DeviceBuffer<f32>,
    value: &mut DeviceBuffer<f32>,
    scratch: &GemvScratch,
    query_shape: QuantizedMatrixShape,
    key_shape: QuantizedMatrixShape,
    value_shape: QuantizedMatrixShape,
) -> i32 {
    let is_q4 = |format| i32::from(format == QuantFormat::Q4K);
    // SAFETY: The three checked matrices share the checked input width. Each
    // output covers its matrix row count.
    unsafe {
        ffi::ie_launch_qkv_gemv(
            query_weights.const_ptr(),
            key_weights.const_ptr(),
            value_weights.const_ptr(),
            input.const_ptr(),
            scratch.quantized_input.const_ptr(),
            scratch.quantized_sums.const_ptr(),
            query.mut_ptr(),
            key.mut_ptr(),
            value.mut_ptr(),
            query_shape.rows(),
            key_shape.rows(),
            value_shape.rows(),
            query_shape.columns(),
            is_q4(query_shape.format()),
            is_q4(key_shape.format()),
            is_q4(value_shape.format()),
            stream.raw(),
        )
    }
}

fn check_gemv(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    format: QuantFormat,
) -> Result<()> {
    if shape.format() != format {
        return Err(Error::SizeMismatch {
            name: "quant format marker",
            expected: format.block_bytes(),
            actual: shape.format().block_bytes(),
        });
    }
    exact_len("weights", shape.bytes(), weights.len())?;
    exact_len("GEMV input", shape.columns(), input.len())?;
    exact_len("GEMV output", shape.output_elements(), output.len())?;
    if scratch.columns != shape.columns() {
        return Err(Error::SizeMismatch {
            name: "GEMV q8_1 scratch columns",
            expected: shape.columns(),
            actual: scratch.columns,
        });
    }
    same_devices(
        stream.device,
        &[
            weights.device,
            input.device,
            output.device,
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    Ok(())
}

fn quantize_q8_1(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
) -> Result<()> {
    exact_len("q8_1 activation input", scratch.columns, input.len())?;
    same_devices(
        stream.device,
        &[
            input.device,
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: Scratch contains one 36-byte block for every 32 checked values.
    check(
        unsafe {
            ffi::ie_launch_quantize_q8_1(
                input.const_ptr(),
                scratch.quantized_input.mut_ptr(),
                scratch.quantized_sums.mut_ptr(),
                scratch.columns,
                stream.raw(),
            )
        },
        "launch q8_1 activation quantization",
    )
}

/// Launches RMSNorm over dense rows.
pub fn rms_norm(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: VectorShape,
    epsilon: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    exact_len("RMSNorm input", shape.elements(), input.len())?;
    exact_len("RMSNorm weight", shape.columns(), weight.len())?;
    exact_len("RMSNorm output", shape.elements(), output.len())?;
    same_devices(stream.device, &[input.device, weight.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: The checked row shape covers each input and output range.
    check(
        unsafe {
            ffi::ie_launch_rms_norm(
                input.const_ptr(),
                weight.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                epsilon,
                stream.raw(),
            )
        },
        "launch RMSNorm",
    )
}

/// Applies RMSNorm and emits the same rows as q8_1 GEMV input.
pub fn rms_norm_q8_parallel(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: VectorShape,
    epsilon: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    if shape.rows() != 1 {
        return Err(Error::SizeMismatch {
            name: "parallel RMSNorm q8_1 rows",
            expected: 1,
            actual: shape.rows(),
        });
    }
    if !shape.columns().is_multiple_of(256) {
        return Err(Error::NotDivisible {
            field: "parallel RMSNorm q8_1 columns",
            value: shape.columns(),
            divisor: 256,
        });
    }
    exact_len("parallel RMSNorm q8_1 input", shape.elements(), input.len())?;
    exact_len(
        "parallel RMSNorm q8_1 weight",
        shape.columns(),
        weight.len(),
    )?;
    exact_len(
        "parallel RMSNorm q8_1 output",
        shape.elements(),
        output.len(),
    )?;
    if scratch.columns != shape.columns() {
        return Err(Error::SizeMismatch {
            name: "parallel RMSNorm q8_1 scratch columns",
            expected: shape.columns(),
            actual: scratch.columns,
        });
    }
    same_devices(
        stream.device,
        &[
            input.device,
            weight.device,
            output.device,
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked row covers the dense output. Each 256-column block
    // emits eight disjoint q8_1 blocks.
    check(
        unsafe {
            ffi::ie_launch_rms_norm_q8_parallel(
                input.const_ptr(),
                weight.const_ptr(),
                output.mut_ptr(),
                scratch.quantized_input.mut_ptr(),
                scratch.quantized_sums.mut_ptr(),
                shape.rows(),
                shape.columns(),
                epsilon,
                stream.raw(),
            )
        },
        "launch parallel RMSNorm with q8_1 output",
    )
}

/// Applies RMSNorm and GPT-NeoX RoPE to one token of head rows.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_rope(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: VectorShape,
    position: usize,
    epsilon: f32,
    theta: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    validate_positive("theta", theta)?;
    if !shape.columns().is_multiple_of(2) {
        return Err(Error::NotDivisible {
            field: "RMSNorm RoPE columns",
            value: shape.columns(),
            divisor: 2,
        });
    }
    if shape.columns() > 512 {
        return Err(Error::TooLarge {
            field: "RMSNorm RoPE columns",
            value: shape.columns(),
            maximum: 512,
        });
    }
    exact_len("RMSNorm RoPE input", shape.elements(), input.len())?;
    exact_len("RMSNorm RoPE weight", shape.columns(), weight.len())?;
    exact_len("RMSNorm RoPE output", shape.elements(), output.len())?;
    same_devices(stream.device, &[input.device, weight.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: Each row is one head. The checked even column count covers all
    // normalized half-pairs.
    check(
        unsafe {
            ffi::ie_launch_rms_norm_rope(
                input.const_ptr(),
                weight.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                position,
                epsilon,
                theta,
                stream.raw(),
            )
        },
        "launch RMSNorm RoPE",
    )
}

/// Applies RMSNorm and GPT-NeoX RoPE at one device-held position.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_rope_device_position(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: VectorShape,
    position: &DeviceBuffer<u32>,
    epsilon: f32,
    theta: f32,
) -> Result<()> {
    validate_rms_norm_rope_device_position(
        stream, input, weight, output, shape, position, epsilon, theta,
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked even row shape covers each pair. Position contains
    // one device scalar that remains live through stream execution.
    check(
        unsafe {
            ffi::ie_launch_rms_norm_rope_device_position(
                input.const_ptr(),
                weight.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                position.const_ptr(),
                epsilon,
                theta,
                stream.raw(),
            )
        },
        "launch device-position RMSNorm RoPE",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_rms_norm_rope_device_position(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    shape: VectorShape,
    position: &DeviceBuffer<u32>,
    epsilon: f32,
    theta: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    validate_positive("theta", theta)?;
    if !shape.columns().is_multiple_of(2) {
        return Err(Error::NotDivisible {
            field: "RMSNorm RoPE columns",
            value: shape.columns(),
            divisor: 2,
        });
    }
    if shape.columns() > 512 {
        return Err(Error::TooLarge {
            field: "RMSNorm RoPE columns",
            value: shape.columns(),
            maximum: 512,
        });
    }
    exact_len("RMSNorm RoPE input", shape.elements(), input.len())?;
    exact_len("RMSNorm RoPE weight", shape.columns(), weight.len())?;
    exact_len("RMSNorm RoPE output", shape.elements(), output.len())?;
    exact_len("RMSNorm RoPE position", 1, position.len())?;
    same_devices(
        stream.device,
        &[input.device, weight.device, output.device, position.device],
    )
}

/// Applies RMSNorm and RoPE to all query and key heads in one launch.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<()> {
    check_qk_norm_rope(
        stream,
        query,
        query_weight,
        query_output,
        query_shape,
        key,
        key_weight,
        key_output,
        key_shape,
        scratch,
        epsilon,
    )?;
    activate_device(stream.device)?;
    // SAFETY: Both checked row sets use the same even head dimension. Each
    // output covers its input row count.
    check(
        unsafe {
            ffi::ie_launch_qk_norm_rope(
                query.const_ptr(),
                query_weight.const_ptr(),
                query_output.mut_ptr(),
                query_shape.rows(),
                key.const_ptr(),
                key_weight.const_ptr(),
                key_output.mut_ptr(),
                key_shape.rows(),
                query_shape.columns(),
                scratch.table.const_ptr(),
                epsilon,
                stream.raw(),
            )
        },
        "launch QK RMSNorm RoPE",
    )
}

/// Applies QK normalization and RoPE, then appends to f32 KV caches.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_kv_append(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
    attention_shape: AttentionShape,
    position: usize,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<()> {
    check_qk_norm_rope_kv_append(
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
        attention_shape,
        Some(position),
        None,
        scratch,
        epsilon,
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked QK rows and KV buffers match the head-major cache.
    check(
        unsafe {
            ffi::ie_launch_qk_norm_rope_kv_append(
                query.const_ptr(),
                query_weight.const_ptr(),
                query_output.mut_ptr(),
                query_shape.rows(),
                key.const_ptr(),
                key_weight.const_ptr(),
                key_output.mut_ptr(),
                key_shape.rows(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                query_shape.columns(),
                attention_shape.max_context(),
                position,
                scratch.table.const_ptr(),
                epsilon,
                stream.raw(),
            )
        },
        "launch QK RMSNorm RoPE with KV append",
    )
}

/// Applies QK normalization and RoPE, then appends to f16 KV caches.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_kv_append_f16(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u16>,
    value_cache: &mut DeviceBuffer<u16>,
    attention_shape: AttentionShape,
    position: usize,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<()> {
    check_qk_norm_rope_kv_append(
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
        attention_shape,
        Some(position),
        None,
        scratch,
        epsilon,
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked QK rows and f16 KV buffers match the cache.
    check(
        unsafe {
            ffi::ie_launch_qk_norm_rope_kv_append_f16(
                query.const_ptr(),
                query_weight.const_ptr(),
                query_output.mut_ptr(),
                query_shape.rows(),
                key.const_ptr(),
                key_weight.const_ptr(),
                key_output.mut_ptr(),
                key_shape.rows(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                query_shape.columns(),
                attention_shape.max_context(),
                position,
                scratch.table.const_ptr(),
                epsilon,
                stream.raw(),
            )
        },
        "launch QK RMSNorm RoPE with f16 KV append",
    )
}

/// Applies QK normalization and RoPE, then appends at a device position.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_kv_append_device_position(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
    attention_shape: AttentionShape,
    position: &DeviceBuffer<u32>,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<()> {
    check_qk_norm_rope_kv_append(
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
        attention_shape,
        None,
        Some(position),
        scratch,
        epsilon,
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked device position remains live through execution.
    check(
        unsafe {
            ffi::ie_launch_qk_norm_rope_kv_append_device_position(
                query.const_ptr(),
                query_weight.const_ptr(),
                query_output.mut_ptr(),
                query_shape.rows(),
                key.const_ptr(),
                key_weight.const_ptr(),
                key_output.mut_ptr(),
                key_shape.rows(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                query_shape.columns(),
                attention_shape.max_context(),
                position.const_ptr(),
                scratch.table.const_ptr(),
                epsilon,
                stream.raw(),
            )
        },
        "launch device-position QK RMSNorm RoPE with KV append",
    )
}

/// Applies QK normalization and RoPE, then appends to f16 at a device position.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_kv_append_f16_device_position(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u16>,
    value_cache: &mut DeviceBuffer<u16>,
    attention_shape: AttentionShape,
    position: &DeviceBuffer<u32>,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<()> {
    check_qk_norm_rope_kv_append(
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
        attention_shape,
        None,
        Some(position),
        scratch,
        epsilon,
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked device position and f16 caches remain live.
    check(
        unsafe {
            ffi::ie_launch_qk_norm_rope_kv_append_f16_device_position(
                query.const_ptr(),
                query_weight.const_ptr(),
                query_output.mut_ptr(),
                query_shape.rows(),
                key.const_ptr(),
                key_weight.const_ptr(),
                key_output.mut_ptr(),
                key_shape.rows(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                query_shape.columns(),
                attention_shape.max_context(),
                position.const_ptr(),
                scratch.table.const_ptr(),
                epsilon,
                stream.raw(),
            )
        },
        "launch device-position QK RMSNorm RoPE with f16 KV append",
    )
}

/// Applies decode-identical QK normalization and KV append at consecutive positions.
#[allow(clippy::too_many_arguments)]
pub fn verify_qk_norm_rope_kv_append(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
    attention_shape: AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<()> {
    verify_qk_norm_rope_kv_append_inner(
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
        attention_shape,
        start_position,
        positions,
        scratch,
        epsilon,
        ffi::ie_launch_verify_qk_norm_rope_kv_append,
    )
}

/// Applies decode-identical QK normalization and f16 KV append at consecutive positions.
#[allow(clippy::too_many_arguments)]
pub fn verify_qk_norm_rope_kv_append_f16(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u16>,
    value_cache: &mut DeviceBuffer<u16>,
    attention_shape: AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<()> {
    verify_qk_norm_rope_kv_append_inner(
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
        attention_shape,
        start_position,
        positions,
        scratch,
        epsilon,
        ffi::ie_launch_verify_qk_norm_rope_kv_append_f16,
    )
}

type VerifyQkLaunch<Cache> = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *mut f32,
    usize,
    *const f32,
    *const f32,
    *mut f32,
    usize,
    *const f32,
    *mut Cache,
    *mut Cache,
    usize,
    usize,
    usize,
    usize,
    *const f64,
    *mut f32,
    f32,
    *mut c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn verify_qk_norm_rope_kv_append_inner<Cache: DeviceCopy>(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &mut DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &mut DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<Cache>,
    value_cache: &mut DeviceBuffer<Cache>,
    attention_shape: AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
    launch: VerifyQkLaunch<Cache>,
) -> Result<()> {
    validate_verify_qk_norm_rope_kv_append(
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
        attention_shape,
        start_position,
        positions,
        scratch,
        epsilon,
    )?;
    activate_device(stream.device)?;
    // SAFETY: Each dense position row, cache, and RoPE scratch range was
    // checked above. The launcher serializes positions on this stream.
    check(
        unsafe {
            launch(
                query.const_ptr(),
                query_weight.const_ptr(),
                query_output.mut_ptr(),
                query_shape.rows(),
                key.const_ptr(),
                key_weight.const_ptr(),
                key_output.mut_ptr(),
                key_shape.rows(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                query_shape.columns(),
                attention_shape.max_context(),
                start_position,
                positions,
                scratch.inverse_frequencies.const_ptr(),
                scratch.table.mut_ptr(),
                epsilon,
                stream.raw(),
            )
        },
        "launch verifier QK RMSNorm RoPE with KV append",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_verify_qk_norm_rope_kv_append<Cache: DeviceCopy>(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<Cache>,
    value_cache: &DeviceBuffer<Cache>,
    attention_shape: AttentionShape,
    start_position: usize,
    positions: usize,
    scratch: &mut RopeScratch,
    epsilon: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    validate_verify_qk_positions(attention_shape, start_position, positions)?;
    let (query_elements, key_elements) = verify_qk_elements(query_shape, key_shape, positions)?;
    validate_verify_qk_lengths(
        query,
        query_weight,
        query_output,
        query_shape,
        query_elements,
        key,
        key_weight,
        key_output,
        key_shape,
        key_elements,
        value,
        key_cache,
        value_cache,
        attention_shape,
    )?;
    scratch.check(stream, query_shape.columns())?;
    same_devices(
        stream.device,
        &[
            query.device,
            query_weight.device,
            query_output.device,
            key.device,
            key_weight.device,
            key_output.device,
            value.device,
            key_cache.device,
            value_cache.device,
            scratch.inverse_frequencies.device,
            scratch.table.device,
        ],
    )
}

fn validate_verify_qk_positions(
    attention_shape: AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<()> {
    if positions == 0 || positions > 8 {
        return Err(Error::TooLarge {
            field: "verifier positions",
            value: positions,
            maximum: 8,
        });
    }
    let end = start_position
        .checked_add(positions)
        .ok_or(Error::SizeOverflow {
            field: "verifier KV end position",
        })?;
    if end > attention_shape.max_context() {
        return Err(Error::ContextLength {
            context_length: end,
            max_context: attention_shape.max_context(),
        });
    }
    Ok(())
}

fn verify_qk_elements(
    query_shape: VectorShape,
    key_shape: VectorShape,
    positions: usize,
) -> Result<(usize, usize)> {
    Ok((
        checked_product(query_shape.elements(), positions, "verifier query elements")?,
        checked_product(key_shape.elements(), positions, "verifier key elements")?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_verify_qk_lengths<Cache: DeviceCopy>(
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &DeviceBuffer<f32>,
    query_shape: VectorShape,
    query_elements: usize,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &DeviceBuffer<f32>,
    key_shape: VectorShape,
    key_elements: usize,
    value: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<Cache>,
    value_cache: &DeviceBuffer<Cache>,
    attention_shape: AttentionShape,
) -> Result<()> {
    exact_len("verifier query", query_elements, query.len())?;
    exact_len("verifier query output", query_elements, query_output.len())?;
    exact_len("verifier key", key_elements, key.len())?;
    exact_len("verifier key output", key_elements, key_output.len())?;
    exact_len("verifier value", key_elements, value.len())?;
    exact_len(
        "verifier query weight",
        query_shape.columns(),
        query_weight.len(),
    )?;
    exact_len("verifier key weight", key_shape.columns(), key_weight.len())?;
    exact_len(
        "verifier key cache",
        attention_shape.cache_elements(),
        key_cache.len(),
    )?;
    exact_len(
        "verifier value cache",
        attention_shape.cache_elements(),
        value_cache.len(),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_qk_norm_rope_kv_append<T: DeviceCopy>(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &DeviceBuffer<f32>,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    attention_shape: AttentionShape,
    host_position: Option<usize>,
    device_position: Option<&DeviceBuffer<u32>>,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<()> {
    check_qk_norm_rope(
        stream,
        query,
        query_weight,
        query_output,
        query_shape,
        key,
        key_weight,
        key_output,
        key_shape,
        scratch,
        epsilon,
    )?;
    validate_qk_norm_rope_kv_append_tail(
        stream,
        key_shape,
        value,
        key_cache,
        value_cache,
        attention_shape,
        host_position,
        device_position,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_qk_norm_rope_kv_append_tail<T: DeviceCopy>(
    stream: &Stream,
    key_shape: VectorShape,
    value: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    attention_shape: AttentionShape,
    host_position: Option<usize>,
    device_position: Option<&DeviceBuffer<u32>>,
) -> Result<()> {
    validate_kv_append_shape(key_shape, attention_shape)?;
    validate_kv_append_position(stream, attention_shape, host_position, device_position)?;
    validate_kv_append_buffers(value, key_cache, value_cache, attention_shape)?;
    same_devices(
        stream.device,
        &[value.device, key_cache.device, value_cache.device],
    )
}

fn validate_kv_append_shape(key_shape: VectorShape, attention_shape: AttentionShape) -> Result<()> {
    if key_shape.rows() != attention_shape.n_head_kv()
        || key_shape.columns() != attention_shape.head_dim()
    {
        return Err(Error::SizeMismatch {
            name: "QK rows for KV append",
            expected: attention_shape.projected_kv_elements()?,
            actual: key_shape.elements(),
        });
    }
    Ok(())
}

fn validate_kv_append_position(
    stream: &Stream,
    attention_shape: AttentionShape,
    host_position: Option<usize>,
    device_position: Option<&DeviceBuffer<u32>>,
) -> Result<()> {
    if let Some(position) = host_position {
        if position >= attention_shape.max_context() {
            return Err(Error::ContextLength {
                context_length: position + 1,
                max_context: attention_shape.max_context(),
            });
        }
    }
    if let Some(position) = device_position {
        exact_len("KV position", 1, position.len())?;
        same_device(stream.device, position.device)?;
    }
    Ok(())
}

fn validate_kv_append_buffers<T: DeviceCopy>(
    value: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    attention_shape: AttentionShape,
) -> Result<()> {
    exact_len(
        "projected value",
        attention_shape.projected_kv_elements()?,
        value.len(),
    )?;
    exact_len(
        "key cache",
        attention_shape.cache_elements(),
        key_cache.len(),
    )?;
    exact_len(
        "value cache",
        attention_shape.cache_elements(),
        value_cache.len(),
    )
}

#[allow(clippy::too_many_arguments)]
fn check_qk_norm_rope(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &DeviceBuffer<f32>,
    key_shape: VectorShape,
    scratch: &RopeScratch,
    epsilon: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    validate_qk_norm_rope_shapes(query_shape, key_shape)?;
    validate_qk_norm_rope_buffers(
        stream,
        query,
        query_weight,
        query_output,
        query_shape,
        key,
        key_weight,
        key_output,
        key_shape,
        scratch,
    )
}

fn validate_qk_norm_rope_shapes(query_shape: VectorShape, key_shape: VectorShape) -> Result<()> {
    if query_shape.columns() != key_shape.columns() {
        return Err(Error::SizeMismatch {
            name: "QK RMSNorm RoPE columns",
            expected: query_shape.columns(),
            actual: key_shape.columns(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_qk_norm_rope_buffers(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    query_weight: &DeviceBuffer<f32>,
    query_output: &DeviceBuffer<f32>,
    query_shape: VectorShape,
    key: &DeviceBuffer<f32>,
    key_weight: &DeviceBuffer<f32>,
    key_output: &DeviceBuffer<f32>,
    key_shape: VectorShape,
    scratch: &RopeScratch,
) -> Result<()> {
    if !query_shape.columns().is_multiple_of(2) {
        return Err(Error::NotDivisible {
            field: "QK RMSNorm RoPE columns",
            value: query_shape.columns(),
            divisor: 2,
        });
    }
    if query_shape.columns() > 512 {
        return Err(Error::TooLarge {
            field: "QK RMSNorm RoPE columns",
            value: query_shape.columns(),
            maximum: 512,
        });
    }
    exact_len("query norm input", query_shape.elements(), query.len())?;
    exact_len(
        "query norm weight",
        query_shape.columns(),
        query_weight.len(),
    )?;
    exact_len(
        "query norm output",
        query_shape.elements(),
        query_output.len(),
    )?;
    exact_len("key norm input", key_shape.elements(), key.len())?;
    exact_len("key norm weight", key_shape.columns(), key_weight.len())?;
    exact_len("key norm output", key_shape.elements(), key_output.len())?;
    scratch.check(stream, query_shape.columns())?;
    same_devices(
        stream.device,
        &[
            query.device,
            query_weight.device,
            query_output.device,
            key.device,
            key_weight.device,
            key_output.device,
        ],
    )
}

/// Launches fused residual addition and RMSNorm over dense rows.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual(
    stream: &Stream,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: VectorShape,
    epsilon: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    exact_len("residual left", shape.elements(), left.len())?;
    exact_len("residual right", shape.elements(), right.len())?;
    exact_len("RMSNorm weight", shape.columns(), weight.len())?;
    exact_len("RMSNorm output", shape.elements(), output.len())?;
    same_devices(
        stream.device,
        &[left.device, right.device, weight.device, output.device],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked row shape covers all four device ranges.
    check(
        unsafe {
            ffi::ie_launch_rms_norm_residual(
                left.const_ptr(),
                right.const_ptr(),
                weight.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                epsilon,
                stream.raw(),
            )
        },
        "launch residual RMSNorm",
    )
}

/// Adds and stores a residual while normalizing it.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual_store(
    stream: &Stream,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    residual: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: VectorShape,
    epsilon: f32,
) -> Result<()> {
    validate_positive("epsilon", epsilon)?;
    exact_len("residual left", shape.elements(), left.len())?;
    exact_len("residual right", shape.elements(), right.len())?;
    exact_len("RMSNorm weight", shape.columns(), weight.len())?;
    exact_len("stored residual", shape.elements(), residual.len())?;
    exact_len("RMSNorm output", shape.elements(), output.len())?;
    same_devices(
        stream.device,
        &[
            left.device,
            right.device,
            weight.device,
            residual.device,
            output.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked row shape covers all five device ranges.
    check(
        unsafe {
            ffi::ie_launch_rms_norm_residual_store(
                left.const_ptr(),
                right.const_ptr(),
                weight.const_ptr(),
                residual.mut_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                epsilon,
                stream.raw(),
            )
        },
        "launch stored residual RMSNorm",
    )
}

/// Applies GPT-NeoX half-pair RoPE in place.
pub fn rope_neox(
    stream: &Stream,
    values: &mut DeviceBuffer<f32>,
    positions: &DeviceBuffer<u32>,
    shape: RopeShape,
    theta: f32,
) -> Result<()> {
    validate_positive("theta", theta)?;
    exact_len("RoPE values", shape.elements(), values.len())?;
    exact_len("RoPE positions", shape.tokens(), positions.len())?;
    same_devices(stream.device, &[values.device, positions.device])?;
    activate_device(stream.device)?;
    // SAFETY: Every half-pair lies inside the checked value shape. Positions
    // contains one entry for every token.
    check(
        unsafe {
            ffi::ie_launch_rope_neox(
                values.mut_ptr(),
                positions.const_ptr(),
                shape.tokens(),
                shape.heads(),
                shape.head_dim(),
                theta,
                stream.raw(),
            )
        },
        "launch RoPE",
    )
}

/// Applies GPT-NeoX half-pair RoPE at consecutive scalar positions.
pub fn rope_neox_at(
    stream: &Stream,
    values: &mut DeviceBuffer<f32>,
    position: usize,
    shape: RopeShape,
    theta: f32,
) -> Result<()> {
    validate_positive("theta", theta)?;
    exact_len("RoPE values", shape.elements(), values.len())?;
    same_device(stream.device, values.device)?;
    position
        .checked_add(shape.tokens() - 1)
        .ok_or(Error::SizeOverflow {
            field: "RoPE position",
        })?;
    activate_device(stream.device)?;
    // SAFETY: Every half-pair lies inside the checked value shape. Consecutive
    // positions start at the checked scalar `position`.
    check(
        unsafe {
            ffi::ie_launch_rope_neox_at(
                values.mut_ptr(),
                position,
                shape.tokens(),
                shape.heads(),
                shape.head_dim(),
                theta,
                stream.raw(),
            )
        },
        "launch scalar-position RoPE",
    )
}

/// Applies GPT-NeoX half-pair RoPE with configured inverse frequencies.
pub fn rope_at_frequencies(
    stream: &Stream,
    values: &mut DeviceBuffer<f32>,
    position: usize,
    shape: RopeShape,
    scratch: &RopeScratch,
) -> Result<()> {
    exact_len("RoPE values", shape.elements(), values.len())?;
    exact_len(
        "RoPE inverse frequencies",
        shape.head_dim() / 2,
        scratch.inverse_frequencies.len(),
    )?;
    same_devices(
        stream.device,
        &[values.device, scratch.inverse_frequencies.device],
    )?;
    position
        .checked_add(shape.tokens() - 1)
        .ok_or(Error::SizeOverflow {
            field: "RoPE position",
        })?;
    activate_device(stream.device)?;
    // SAFETY: The checked values cover every half-pair. The frequency buffer
    // contains one entry for each pair in a head.
    check(
        unsafe {
            ffi::ie_launch_rope_at_frequencies(
                values.mut_ptr(),
                position,
                shape.tokens(),
                shape.heads(),
                shape.head_dim(),
                scratch.inverse_frequencies.const_ptr(),
                scratch.adjacent_pairs,
                stream.raw(),
            )
        },
        "launch configured RoPE",
    )
}

/// Applies configured RoPE at one device-resident starting position.
pub fn rope_at_frequencies_device_position(
    stream: &Stream,
    values: &mut DeviceBuffer<f32>,
    position: &DeviceBuffer<u32>,
    shape: RopeShape,
    scratch: &RopeScratch,
) -> Result<()> {
    exact_len("RoPE values", shape.elements(), values.len())?;
    exact_len("RoPE device position", 1, position.len())?;
    exact_len(
        "RoPE inverse frequencies",
        shape.head_dim() / 2,
        scratch.inverse_frequencies.len(),
    )?;
    same_devices(
        stream.device,
        &[
            values.device,
            position.device,
            scratch.inverse_frequencies.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The checked values cover every pair. The scalar position and
    // inverse-frequency buffers remain live through the stream launch.
    check(
        unsafe {
            ffi::ie_launch_rope_at_frequencies_device_position(
                values.mut_ptr(),
                position.const_ptr(),
                shape.tokens(),
                shape.heads(),
                shape.head_dim(),
                scratch.inverse_frequencies.const_ptr(),
                scratch.adjacent_pairs,
                stream.raw(),
            )
        },
        "launch device-position configured RoPE",
    )
}

/// Launches fused SiLU multiplication over `gate` and `up`.
pub fn swiglu(
    stream: &Stream,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let elements = gate.len();
    validate_element_grid(elements, "SwiGLU grid blocks")?;
    exact_len("SwiGLU up", elements, up.len())?;
    exact_len("SwiGLU output", elements, output.len())?;
    same_devices(stream.device, &[gate.device, up.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: All device buffers contain exactly `elements` values.
    check(
        unsafe {
            ffi::ie_launch_swiglu(
                gate.const_ptr(),
                up.const_ptr(),
                output.mut_ptr(),
                elements,
                stream.raw(),
            )
        },
        "launch SwiGLU",
    )
}

/// Applies SwiGLU and emits the same vector as q8_1 GEMV input.
pub fn swiglu_q8(
    stream: &Stream,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
) -> Result<()> {
    let elements = gate.len();
    validate_element_grid(elements, "SwiGLU q8_1 grid blocks")?;
    if !elements.is_multiple_of(256) {
        return Err(Error::NotDivisible {
            field: "SwiGLU q8_1 elements",
            value: elements,
            divisor: 256,
        });
    }
    exact_len("SwiGLU q8_1 up", elements, up.len())?;
    exact_len("SwiGLU q8_1 output", elements, output.len())?;
    let blocks = elements / Q8_1_BLOCK_ELEMENTS;
    exact_len(
        "SwiGLU q8_1 scratch bytes",
        checked_product(blocks, Q8_1_BLOCK_BYTES, "SwiGLU q8_1 scratch bytes")?,
        scratch.quantized_input.len(),
    )?;
    exact_len(
        "SwiGLU q8_1 scratch sums",
        blocks,
        scratch.quantized_sums.len(),
    )?;
    same_devices(
        stream.device,
        &[
            gate.device,
            up.device,
            output.device,
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The exact dense vectors and q8_1 scratch cover `elements`.
    check(
        unsafe {
            ffi::ie_launch_swiglu_q8(
                gate.const_ptr(),
                up.const_ptr(),
                output.mut_ptr(),
                scratch.quantized_input.mut_ptr(),
                scratch.quantized_sums.mut_ptr(),
                elements,
                stream.raw(),
            )
        },
        "launch SwiGLU with q8_1 output",
    )
}

/// Applies SwiGLU and emits consecutive verifier rows as q8_1 GEMV input.
pub fn verify_swiglu_q8(
    stream: &Stream,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    columns: usize,
    positions: usize,
) -> Result<()> {
    let elements = checked_product(columns, positions, "verifier SwiGLU elements")?;
    exact_len("verifier SwiGLU gate", elements, gate.len())?;
    exact_len("verifier SwiGLU up", elements, up.len())?;
    exact_len("verifier SwiGLU output", elements, output.len())?;
    if scratch.columns != columns {
        return Err(Error::SizeMismatch {
            name: "verifier SwiGLU q8_1 scratch columns",
            expected: columns,
            actual: scratch.columns,
        });
    }
    swiglu_q8(stream, gate, up, output, scratch)
}

/// Adds two dense device buffers element by element.
pub fn residual_add(
    stream: &Stream,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let elements = left.len();
    validate_element_grid(elements, "residual grid blocks")?;
    exact_len("residual right", elements, right.len())?;
    exact_len("residual output", elements, output.len())?;
    same_devices(stream.device, &[left.device, right.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: All device buffers contain exactly `elements` values.
    check(
        unsafe {
            ffi::ie_launch_residual_add(
                left.const_ptr(),
                right.const_ptr(),
                output.mut_ptr(),
                elements,
                stream.raw(),
            )
        },
        "launch residual add",
    )
}

/// Writes one scalar on the ordered stream.
pub fn write_u32_scalar(stream: &Stream, output: &mut DeviceBuffer<u32>, value: u32) -> Result<()> {
    exact_len("u32 scalar output", 1, output.len())?;
    same_device(stream.device, output.device)?;
    activate_device(stream.device)?;
    // SAFETY: `output` contains one writable `u32` on the stream device.
    check(
        unsafe { ffi::ie_launch_write_u32(output.mut_ptr(), value, stream.raw()) },
        "launch u32 scalar write",
    )
}

/// Increments one device scalar on the ordered stream.
pub fn increment_u32_scalar(stream: &Stream, output: &mut DeviceBuffer<u32>) -> Result<()> {
    exact_len("u32 scalar output", 1, output.len())?;
    same_device(stream.device, output.device)?;
    activate_device(stream.device)?;
    // SAFETY: `output` contains one writable `u32` on the stream device.
    check(
        unsafe { ffi::ie_launch_increment_u32(output.mut_ptr(), stream.raw()) },
        "launch u32 scalar increment",
    )
}

/// Appends one projected key and value row to a head-major KV cache.
#[allow(clippy::too_many_arguments)]
pub fn kv_append(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
    shape: AttentionShape,
    position: usize,
) -> Result<()> {
    if position >= shape.max_context() {
        return Err(Error::ContextLength {
            context_length: position + 1,
            max_context: shape.max_context(),
        });
    }
    let projected = shape.projected_kv_elements()?;
    exact_len("projected key", projected, key.len())?;
    exact_len("projected value", projected, value.len())?;
    exact_len("key cache", shape.cache_elements(), key_cache.len())?;
    exact_len("value cache", shape.cache_elements(), value_cache.len())?;
    same_devices(
        stream.device,
        &[
            key.device,
            value.device,
            key_cache.device,
            value_cache.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The projected rows and cache match the checked head-major shape.
    check(
        unsafe {
            ffi::ie_launch_kv_append(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position,
                stream.raw(),
            )
        },
        "launch KV append",
    )
}

/// Converts and appends one projected KV row to f16 caches.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_f16(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u16>,
    value_cache: &mut DeviceBuffer<u16>,
    shape: AttentionShape,
    position: usize,
) -> Result<()> {
    if position >= shape.max_context() {
        return Err(Error::ContextLength {
            context_length: position + 1,
            max_context: shape.max_context(),
        });
    }
    let projected = shape.projected_kv_elements()?;
    exact_len("projected key", projected, key.len())?;
    exact_len("projected value", projected, value.len())?;
    exact_len("key cache", shape.cache_elements(), key_cache.len())?;
    exact_len("value cache", shape.cache_elements(), value_cache.len())?;
    same_devices(
        stream.device,
        &[
            key.device,
            value.device,
            key_cache.device,
            value_cache.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The projected f32 rows and f16 caches match the checked shape.
    check(
        unsafe {
            ffi::ie_launch_kv_append_f16(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position,
                stream.raw(),
            )
        },
        "launch f16 KV append",
    )
}

/// Appends one projected KV row at a device-held position.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_device_position(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
    shape: AttentionShape,
    position: &DeviceBuffer<u32>,
) -> Result<()> {
    let projected = shape.projected_kv_elements()?;
    exact_len("projected key", projected, key.len())?;
    exact_len("projected value", projected, value.len())?;
    exact_len("key cache", shape.cache_elements(), key_cache.len())?;
    exact_len("value cache", shape.cache_elements(), value_cache.len())?;
    exact_len("KV position", 1, position.len())?;
    same_devices(
        stream.device,
        &[
            key.device,
            value.device,
            key_cache.device,
            value_cache.device,
            position.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The projected rows and caches match the checked layout. The
    // runtime keeps the device position below `max_context`.
    check(
        unsafe {
            ffi::ie_launch_kv_append_device_position(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position.const_ptr(),
                stream.raw(),
            )
        },
        "launch device-position KV append",
    )
}

/// Converts and appends one projected KV row at a device-held position.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_f16_device_position(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u16>,
    value_cache: &mut DeviceBuffer<u16>,
    shape: AttentionShape,
    position: &DeviceBuffer<u32>,
) -> Result<()> {
    let projected = shape.projected_kv_elements()?;
    exact_len("projected key", projected, key.len())?;
    exact_len("projected value", projected, value.len())?;
    exact_len("key cache", shape.cache_elements(), key_cache.len())?;
    exact_len("value cache", shape.cache_elements(), value_cache.len())?;
    exact_len("KV position", 1, position.len())?;
    same_devices(
        stream.device,
        &[
            key.device,
            value.device,
            key_cache.device,
            value_cache.device,
            position.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: The f32 rows and f16 caches match the checked layout. The
    // runtime keeps the device position below `max_context`.
    check(
        unsafe {
            ffi::ie_launch_kv_append_f16_device_position(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position.const_ptr(),
                stream.raw(),
            )
        },
        "launch device-position f16 KV append",
    )
}

/// Quantizes and appends one projected KV row to q8_0 caches.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_q8(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u8>,
    value_cache: &mut DeviceBuffer<u8>,
    shape: AttentionShape,
    position: usize,
) -> Result<()> {
    if position >= shape.max_context() {
        return Err(Error::ContextLength {
            context_length: position + 1,
            max_context: shape.max_context(),
        });
    }
    check_q8_kv_buffers(stream, key, value, key_cache, value_cache, shape)?;
    activate_device(stream.device)?;
    // SAFETY: The projected rows and q8 caches match the checked layout.
    check(
        unsafe {
            ffi::ie_launch_kv_append_q8(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position,
                stream.raw(),
            )
        },
        "launch q8 KV append",
    )
}

/// Quantizes and appends KV at a device-held position.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_q8_device_position(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<u8>,
    value_cache: &mut DeviceBuffer<u8>,
    shape: AttentionShape,
    position: &DeviceBuffer<u32>,
) -> Result<()> {
    check_q8_kv_buffers(stream, key, value, key_cache, value_cache, shape)?;
    exact_len("KV position", 1, position.len())?;
    same_device(stream.device, position.device)?;
    activate_device(stream.device)?;
    // SAFETY: The runtime keeps the device position inside the checked cache.
    check(
        unsafe {
            ffi::ie_launch_kv_append_q8_device_position(
                key.const_ptr(),
                value.const_ptr(),
                key_cache.mut_ptr(),
                value_cache.mut_ptr(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position.const_ptr(),
                stream.raw(),
            )
        },
        "launch device-position q8 KV append",
    )
}

fn check_q8_kv_buffers(
    stream: &Stream,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    shape: AttentionShape,
) -> Result<()> {
    exact_len("projected key", shape.projected_kv_elements()?, key.len())?;
    exact_len(
        "projected value",
        shape.projected_kv_elements()?,
        value.len(),
    )?;
    let cache_bytes = shape.cache_elements() / 32 * 34;
    exact_len("q8 key cache", cache_bytes, key_cache.len())?;
    exact_len("q8 value cache", cache_bytes, value_cache.len())?;
    same_devices(
        stream.device,
        &[
            key.device,
            value.device,
            key_cache.device,
            value_cache.device,
        ],
    )
}

/// Dequantizes one Q4_K embedding row directly from the device table.
pub fn embedding_gather_q4_k(
    stream: &Stream,
    table: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    row: usize,
) -> Result<()> {
    embedding_gather(
        stream,
        table,
        output,
        shape,
        row,
        QuantFormat::Q4K,
        ffi::ie_launch_embedding_q4_k,
    )
}

/// Dequantizes one Q6_K embedding row directly from the device table.
pub fn embedding_gather_q6_k(
    stream: &Stream,
    table: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    row: usize,
) -> Result<()> {
    embedding_gather(
        stream,
        table,
        output,
        shape,
        row,
        QuantFormat::Q6K,
        ffi::ie_launch_embedding_q6_k,
    )
}

/// Dequantizes one Q4_K embedding row selected by a device `u32`.
pub fn embedding_gather_q4_k_device_row(
    stream: &Stream,
    table: &DeviceBuffer<u8>,
    row: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
) -> Result<()> {
    embedding_gather_device_row(
        stream,
        table,
        row,
        output,
        shape,
        QuantFormat::Q4K,
        ffi::ie_launch_embedding_q4_k_device_row,
    )
}

/// Dequantizes one Q6_K embedding row selected by a device `u32`.
pub fn embedding_gather_q6_k_device_row(
    stream: &Stream,
    table: &DeviceBuffer<u8>,
    row: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
) -> Result<()> {
    embedding_gather_device_row(
        stream,
        table,
        row,
        output,
        shape,
        QuantFormat::Q6K,
        ffi::ie_launch_embedding_q6_k_device_row,
    )
}

type DeviceEmbeddingLaunch =
    unsafe extern "C" fn(*const u8, *const u32, *mut f32, usize, usize, *mut c_void) -> i32;

#[allow(clippy::too_many_arguments)]
fn embedding_gather_device_row(
    stream: &Stream,
    table: &DeviceBuffer<u8>,
    row: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    format: QuantFormat,
    launch: DeviceEmbeddingLaunch,
) -> Result<()> {
    if shape.format() != format {
        return Err(Error::SizeMismatch {
            name: "quant format marker",
            expected: format.block_bytes(),
            actual: shape.format().block_bytes(),
        });
    }
    exact_len("embedding table", shape.bytes(), table.len())?;
    exact_len("embedding row", 1, row.len())?;
    exact_len("embedding output", shape.columns(), output.len())?;
    same_devices(stream.device, &[table.device, row.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: The row buffer contains one tokenizer or argmax result. The
    // kernel checks it against `shape.rows()` before reading the table.
    check(
        unsafe {
            launch(
                table.const_ptr(),
                row.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                stream.raw(),
            )
        },
        "launch device-row embedding gather",
    )
}

type EmbeddingLaunch =
    unsafe extern "C" fn(*const u8, *mut f32, usize, usize, usize, *mut c_void) -> i32;

#[allow(clippy::too_many_arguments)]
fn embedding_gather(
    stream: &Stream,
    table: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    row: usize,
    format: QuantFormat,
    launch: EmbeddingLaunch,
) -> Result<()> {
    if shape.format() != format {
        return Err(Error::SizeMismatch {
            name: "quant format marker",
            expected: format.block_bytes(),
            actual: shape.format().block_bytes(),
        });
    }
    if row >= shape.rows() {
        return Err(Error::RowOutOfBounds {
            row,
            rows: shape.rows(),
        });
    }
    exact_len("embedding table", shape.bytes(), table.len())?;
    exact_len("embedding output", shape.columns(), output.len())?;
    same_devices(stream.device, &[table.device, output.device])?;
    activate_device(stream.device)?;
    // SAFETY: `row` is in bounds, and the table has the exact checked storage.
    check(
        unsafe {
            launch(
                table.const_ptr(),
                output.mut_ptr(),
                shape.rows(),
                shape.columns(),
                row,
                stream.raw(),
            )
        },
        "launch embedding gather",
    )
}

/// Runs batch-1 GQA decode attention over a contiguous f32 KV cache.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    context_length: usize,
) -> Result<()> {
    if !(1..=shape.max_context()).contains(&context_length) {
        return Err(Error::ContextLength {
            context_length,
            max_context: shape.max_context(),
        });
    }
    if scratch.shape != shape {
        return Err(Error::SizeMismatch {
            name: "attention scratch query elements",
            expected: shape.query_elements(),
            actual: scratch.shape.query_elements(),
        });
    }
    exact_len("attention query", shape.query_elements(), query.len())?;
    exact_len(
        "attention key cache",
        shape.cache_elements(),
        key_cache.len(),
    )?;
    exact_len(
        "attention value cache",
        shape.cache_elements(),
        value_cache.len(),
    )?;
    exact_len("attention output", shape.query_elements(), output.len())?;
    same_devices(
        stream.device,
        &[
            query.device,
            key_cache.device,
            value_cache.device,
            output.device,
            scratch.partial_max.device,
            scratch.partial_sum.device,
            scratch.partial_output.device,
        ],
    )?;
    let (quantized_output, quantized_sums) = attention_q8_outputs(stream, prepared_output, shape)?;
    activate_device(stream.device)?;
    // SAFETY: Q, K, V, output, and partial buffers match the checked layout.
    // The context length includes the current token and is within the cache.
    check(
        unsafe {
            ffi::ie_launch_attention_decode(
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.partial_max.mut_ptr(),
                scratch.partial_sum.mut_ptr(),
                scratch.partial_output.mut_ptr(),
                quantized_output,
                quantized_sums,
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                context_length,
                stream.raw(),
            )
        },
        "launch decode attention",
    )
}

/// Runs batch-1 GQA decode attention over f16 KV caches.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_f16(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u16>,
    value_cache: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    context_length: usize,
) -> Result<()> {
    if !(1..=shape.max_context()).contains(&context_length) {
        return Err(Error::ContextLength {
            context_length,
            max_context: shape.max_context(),
        });
    }
    check_attention_f16(
        stream,
        query,
        key_cache,
        value_cache,
        output,
        scratch,
        shape,
    )?;
    let (quantized_output, quantized_sums) = attention_q8_outputs(stream, prepared_output, shape)?;
    activate_device(stream.device)?;
    // SAFETY: All buffers match the checked layout and context length.
    check(
        unsafe {
            ffi::ie_launch_attention_decode_f16(
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.partial_max.mut_ptr(),
                scratch.partial_sum.mut_ptr(),
                scratch.partial_output.mut_ptr(),
                quantized_output,
                quantized_sums,
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                context_length,
                stream.raw(),
            )
        },
        "launch f16 decode attention",
    )
}

/// Runs decode attention through a device-held current position.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_device_position(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    position: &DeviceBuffer<u32>,
) -> Result<()> {
    if scratch.shape != shape {
        return Err(Error::SizeMismatch {
            name: "attention scratch query elements",
            expected: shape.query_elements(),
            actual: scratch.shape.query_elements(),
        });
    }
    exact_len("attention query", shape.query_elements(), query.len())?;
    exact_len(
        "attention key cache",
        shape.cache_elements(),
        key_cache.len(),
    )?;
    exact_len(
        "attention value cache",
        shape.cache_elements(),
        value_cache.len(),
    )?;
    exact_len("attention output", shape.query_elements(), output.len())?;
    exact_len("attention position", 1, position.len())?;
    same_devices(
        stream.device,
        &[
            query.device,
            key_cache.device,
            value_cache.device,
            output.device,
            scratch.partial_max.device,
            scratch.partial_sum.device,
            scratch.partial_output.device,
            position.device,
        ],
    )?;
    let (quantized_output, quantized_sums) = attention_q8_outputs(stream, prepared_output, shape)?;
    activate_device(stream.device)?;
    // SAFETY: Every buffer matches the checked layout. The runtime keeps the
    // device position below `max_context`.
    check(
        unsafe {
            ffi::ie_launch_attention_decode_device_position(
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.partial_max.mut_ptr(),
                scratch.partial_sum.mut_ptr(),
                scratch.partial_output.mut_ptr(),
                quantized_output,
                quantized_sums,
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position.const_ptr(),
                stream.raw(),
            )
        },
        "launch device-position decode attention",
    )
}

/// Runs f16 KV decode attention through a device-held position.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_f16_device_position(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u16>,
    value_cache: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    position: &DeviceBuffer<u32>,
) -> Result<()> {
    check_attention_f16(
        stream,
        query,
        key_cache,
        value_cache,
        output,
        scratch,
        shape,
    )?;
    exact_len("attention position", 1, position.len())?;
    same_device(stream.device, position.device)?;
    let (quantized_output, quantized_sums) = attention_q8_outputs(stream, prepared_output, shape)?;
    activate_device(stream.device)?;
    // SAFETY: Every buffer matches the checked layout. The runtime keeps the
    // device position below `max_context`.
    check(
        unsafe {
            ffi::ie_launch_attention_decode_f16_device_position(
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.partial_max.mut_ptr(),
                scratch.partial_sum.mut_ptr(),
                scratch.partial_output.mut_ptr(),
                quantized_output,
                quantized_sums,
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position.const_ptr(),
                stream.raw(),
            )
        },
        "launch device-position f16 decode attention",
    )
}

/// Runs decode attention directly over q8_0 KV caches.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_q8(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    context_length: usize,
) -> Result<()> {
    if !(1..=shape.max_context()).contains(&context_length) {
        return Err(Error::ContextLength {
            context_length,
            max_context: shape.max_context(),
        });
    }
    check_attention_q8(
        stream,
        query,
        key_cache,
        value_cache,
        output,
        scratch,
        shape,
    )?;
    let (quantized_output, quantized_sums) = attention_q8_outputs(stream, prepared_output, shape)?;
    activate_device(stream.device)?;
    // SAFETY: Every buffer and the live context match the checked shape.
    check(
        unsafe {
            ffi::ie_launch_attention_decode_q8(
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.partial_max.mut_ptr(),
                scratch.partial_sum.mut_ptr(),
                scratch.partial_output.mut_ptr(),
                quantized_output,
                quantized_sums,
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                context_length,
                stream.raw(),
            )
        },
        "launch q8 decode attention",
    )
}

/// Runs q8_0 KV attention through a device-held position.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_q8_device_position(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    position: &DeviceBuffer<u32>,
) -> Result<()> {
    check_attention_q8(
        stream,
        query,
        key_cache,
        value_cache,
        output,
        scratch,
        shape,
    )?;
    exact_len("attention position", 1, position.len())?;
    same_device(stream.device, position.device)?;
    let (quantized_output, quantized_sums) = attention_q8_outputs(stream, prepared_output, shape)?;
    activate_device(stream.device)?;
    // SAFETY: Every buffer matches the checked layout and device.
    check(
        unsafe {
            ffi::ie_launch_attention_decode_q8_device_position(
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.partial_max.mut_ptr(),
                scratch.partial_sum.mut_ptr(),
                scratch.partial_output.mut_ptr(),
                quantized_output,
                quantized_sums,
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                position.const_ptr(),
                stream.raw(),
            )
        },
        "launch device-position q8 decode attention",
    )
}

fn check_attention_q8(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    output: &DeviceBuffer<f32>,
    scratch: &AttentionScratch,
    shape: AttentionShape,
) -> Result<()> {
    if scratch.shape != shape {
        return Err(Error::SizeMismatch {
            name: "attention scratch query elements",
            expected: shape.query_elements(),
            actual: scratch.shape.query_elements(),
        });
    }
    exact_len("attention query", shape.query_elements(), query.len())?;
    let cache_bytes = shape.cache_elements() / 32 * 34;
    exact_len("q8 attention key cache", cache_bytes, key_cache.len())?;
    exact_len("q8 attention value cache", cache_bytes, value_cache.len())?;
    exact_len("attention output", shape.query_elements(), output.len())?;
    same_devices(
        stream.device,
        &[
            query.device,
            key_cache.device,
            value_cache.device,
            output.device,
            scratch.partial_max.device,
            scratch.partial_sum.device,
            scratch.partial_output.device,
        ],
    )
}

/// Runs decode-identical attention for consecutive verifier queries.
#[allow(clippy::too_many_arguments)]
pub fn verify_attention(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<()> {
    verify_attention_inner(
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
        ffi::ie_launch_verify_attention,
    )
}

/// Runs decode-identical attention over f16 KV for consecutive queries.
#[allow(clippy::too_many_arguments)]
pub fn verify_attention_f16(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u16>,
    value_cache: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<()> {
    verify_attention_inner(
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
        ffi::ie_launch_verify_attention_f16,
    )
}

type VerifyAttentionLaunch<Cache> = unsafe extern "C" fn(
    *const f32,
    *const Cache,
    *const Cache,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut u8,
    *mut u32,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    *mut c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn verify_attention_inner<Cache: DeviceCopy>(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<Cache>,
    value_cache: &DeviceBuffer<Cache>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    start_position: usize,
    positions: usize,
    launch: VerifyAttentionLaunch<Cache>,
) -> Result<()> {
    validate_verify_attention(
        stream,
        query,
        key_cache,
        value_cache,
        output,
        scratch,
        shape,
        start_position,
        positions,
    )?;
    let (quantized_output, quantized_sums) =
        attention_q8_outputs_multi(stream, prepared_output, shape, positions)?;
    activate_device(stream.device)?;
    // SAFETY: Each query position has a distinct checked scratch range. The
    // launch reads and writes only the buffers and ranges validated above.
    check(
        unsafe {
            launch(
                query.const_ptr(),
                key_cache.const_ptr(),
                value_cache.const_ptr(),
                output.mut_ptr(),
                scratch.partial_max.mut_ptr(),
                scratch.partial_sum.mut_ptr(),
                scratch.partial_output.mut_ptr(),
                quantized_output,
                quantized_sums,
                shape.n_head(),
                shape.n_head_kv(),
                shape.head_dim(),
                shape.max_context(),
                start_position,
                positions,
                stream.raw(),
            )
        },
        "launch verifier attention",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_verify_attention<Cache: DeviceCopy>(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<Cache>,
    value_cache: &DeviceBuffer<Cache>,
    output: &DeviceBuffer<f32>,
    scratch: &AttentionScratch,
    shape: AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<usize> {
    let query_elements = checked_product(
        shape.query_elements(),
        positions,
        "verifier attention elements",
    )?;
    validate_verify_attention_lengths(
        query,
        key_cache,
        value_cache,
        output,
        query_elements,
        shape.cache_elements(),
    )?;
    validate_verify_attention_position(shape, start_position, positions)?;
    if scratch.shape != shape || scratch.positions != positions {
        return Err(Error::SizeMismatch {
            name: "verifier attention scratch query elements",
            expected: query_elements,
            actual: scratch.shape.query_elements() * scratch.positions,
        });
    }
    same_devices(
        stream.device,
        &[
            query.device,
            key_cache.device,
            value_cache.device,
            output.device,
            scratch.partial_max.device,
            scratch.partial_sum.device,
            scratch.partial_output.device,
        ],
    )?;
    Ok(query_elements)
}

fn validate_verify_attention_lengths<Cache: DeviceCopy>(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<Cache>,
    value_cache: &DeviceBuffer<Cache>,
    output: &DeviceBuffer<f32>,
    query_elements: usize,
    cache_elements: usize,
) -> Result<()> {
    exact_len("verifier attention query", query_elements, query.len())?;
    exact_len("verifier attention output", query_elements, output.len())?;
    exact_len(
        "verifier attention key cache",
        cache_elements,
        key_cache.len(),
    )?;
    exact_len(
        "verifier attention value cache",
        cache_elements,
        value_cache.len(),
    )
}

fn validate_verify_attention_position(
    shape: AttentionShape,
    start_position: usize,
    positions: usize,
) -> Result<()> {
    if start_position
        .checked_add(positions)
        .ok_or(Error::SizeOverflow {
            field: "verifier attention end position",
        })?
        > shape.max_context()
    {
        return Err(Error::ContextLength {
            context_length: start_position + positions,
            max_context: shape.max_context(),
        });
    }
    Ok(())
}

fn attention_q8_outputs_multi(
    stream: &Stream,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
    positions: usize,
) -> Result<(*mut u8, *mut u32)> {
    let Some(scratch) = prepared_output else {
        return Ok((ptr::null_mut(), ptr::null_mut()));
    };
    if shape.head_dim() != PREPARED_ATTENTION_HEAD_DIM {
        return Err(Error::SizeMismatch {
            name: "prepared verifier attention head dimension",
            expected: PREPARED_ATTENTION_HEAD_DIM,
            actual: shape.head_dim(),
        });
    }
    if scratch.columns != shape.query_elements() {
        return Err(Error::SizeMismatch {
            name: "prepared verifier attention q8_1 columns",
            expected: shape.query_elements(),
            actual: scratch.columns,
        });
    }
    let blocks = checked_product(
        shape.query_elements() / Q8_1_BLOCK_ELEMENTS,
        positions,
        "prepared verifier attention q8_1 blocks",
    )?;
    exact_len(
        "prepared verifier attention q8_1 bytes",
        checked_product(
            blocks,
            Q8_1_BLOCK_BYTES,
            "prepared verifier attention q8_1 bytes",
        )?,
        scratch.quantized_input.len(),
    )?;
    exact_len(
        "prepared verifier attention q8_1 sums",
        blocks,
        scratch.quantized_sums.len(),
    )?;
    same_devices(
        stream.device,
        &[
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    Ok((
        scratch.quantized_input.mut_ptr(),
        scratch.quantized_sums.mut_ptr().cast(),
    ))
}

pub(crate) fn verifier_attention_prepares_output(shape: AttentionShape) -> bool {
    shape.head_dim() == PREPARED_ATTENTION_HEAD_DIM
        && shape.query_elements().is_multiple_of(Q8_1_BLOCK_ELEMENTS)
}

fn attention_q8_outputs(
    stream: &Stream,
    prepared_output: Option<&mut GemvScratch>,
    shape: AttentionShape,
) -> Result<(*mut u8, *mut u32)> {
    let Some(scratch) = prepared_output else {
        return Ok((ptr::null_mut(), ptr::null_mut()));
    };
    if shape.head_dim() != PREPARED_ATTENTION_HEAD_DIM {
        return Err(Error::SizeMismatch {
            name: "prepared attention head dimension",
            expected: PREPARED_ATTENTION_HEAD_DIM,
            actual: shape.head_dim(),
        });
    }
    if scratch.columns != shape.query_elements() {
        return Err(Error::SizeMismatch {
            name: "prepared attention q8_1 columns",
            expected: shape.query_elements(),
            actual: scratch.columns,
        });
    }
    same_devices(
        stream.device,
        &[
            scratch.quantized_input.device,
            scratch.quantized_sums.device,
        ],
    )?;
    Ok((
        scratch.quantized_input.mut_ptr(),
        scratch.quantized_sums.mut_ptr(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn check_attention_f16(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u16>,
    value_cache: &DeviceBuffer<u16>,
    output: &DeviceBuffer<f32>,
    scratch: &AttentionScratch,
    shape: AttentionShape,
) -> Result<()> {
    if scratch.shape != shape {
        return Err(Error::SizeMismatch {
            name: "attention scratch query elements",
            expected: shape.query_elements(),
            actual: scratch.shape.query_elements(),
        });
    }
    exact_len("attention query", shape.query_elements(), query.len())?;
    exact_len(
        "attention key cache",
        shape.cache_elements(),
        key_cache.len(),
    )?;
    exact_len(
        "attention value cache",
        shape.cache_elements(),
        value_cache.len(),
    )?;
    exact_len("attention output", shape.query_elements(), output.len())?;
    same_devices(
        stream.device,
        &[
            query.device,
            key_cache.device,
            value_cache.device,
            output.device,
            scratch.partial_max.device,
            scratch.partial_sum.device,
            scratch.partial_output.device,
        ],
    )
}

/// Finds the lowest index with the largest non-NaN value.
pub fn argmax(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<u32>,
    scratch: &mut ArgmaxScratch,
) -> Result<()> {
    if scratch.elements != input.len() {
        return Err(Error::SizeMismatch {
            name: "argmax scratch elements",
            expected: input.len(),
            actual: scratch.elements,
        });
    }
    exact_len("argmax output", 1, output.len())?;
    same_devices(
        stream.device,
        &[
            input.device,
            output.device,
            scratch.partial_values.device,
            scratch.partial_indices.device,
        ],
    )?;
    activate_device(stream.device)?;
    // SAFETY: Scratch was sized from this exact input length, and output has
    // one `u32` element.
    check(
        unsafe {
            ffi::ie_launch_argmax(
                input.const_ptr(),
                input.len(),
                output.mut_ptr(),
                scratch.partial_values.mut_ptr(),
                scratch.partial_indices.mut_ptr(),
                stream.raw(),
            )
        },
        "launch argmax",
    )
}

fn byte_len<T>(len: usize, field: &'static str) -> Result<usize> {
    len.checked_mul(mem::size_of::<T>())
        .ok_or(Error::SizeOverflow { field })
}

fn checked_product(left: usize, right: usize, field: &'static str) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::SizeOverflow { field })
}

fn checked_product3(
    first: usize,
    second: usize,
    third: usize,
    field: &'static str,
) -> Result<usize> {
    first
        .checked_mul(second)
        .and_then(|value| value.checked_mul(third))
        .ok_or(Error::SizeOverflow { field })
}

fn bytes_u64(elements: usize, element_bytes: usize) -> Result<u64> {
    elements
        .checked_mul(element_bytes)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(Error::SizeOverflow {
            field: "prefill workspace bytes",
        })
}

fn exact_len(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::SizeMismatch {
            name,
            expected,
            actual,
        })
    }
}

fn validate_positive(field: &'static str, value: f32) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidPositiveFloat { field, value })
    }
}

fn same_device(left: i32, right: i32) -> Result<()> {
    if left == right {
        Ok(())
    } else {
        Err(Error::DeviceMismatch)
    }
}

fn same_devices(device: i32, others: &[i32]) -> Result<()> {
    if others.iter().all(|other| *other == device) {
        Ok(())
    } else {
        Err(Error::DeviceMismatch)
    }
}

fn activate_device(device: i32) -> Result<()> {
    // SAFETY: The call takes one integer device index and no pointers.
    check(unsafe { ffi::ie_cuda_set_device(device) }, "set device")
}

fn check(code: i32, operation: &'static str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: CUDA returns a process-lifetime, null-terminated error string.
    let pointer = unsafe { ffi::ie_cuda_error_string(code) };
    let message = if pointer.is_null() {
        "unknown CUDA error".to_owned()
    } else {
        // SAFETY: The non-null pointer is a CUDA-owned, null-terminated string.
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error::Runtime {
        operation,
        code,
        message,
    })
}

fn check_cublas(code: i32, operation: &'static str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: cuBLAS returns a process-lifetime, null-terminated status string.
    let pointer = unsafe { ffi::ie_cublaslt_error_string(code) };
    let message = if pointer.is_null() {
        "unknown cuBLASLt error".to_owned()
    } else {
        // SAFETY: The non-null pointer is library-owned and null-terminated.
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error::Runtime {
        operation,
        code,
        message,
    })
}
