#![deny(unsafe_code)]

//! Runs the Leone CUDA kernels through checked Rust wrappers.
//!
//! The crate targets SM89 and uses the CUDA runtime API. The build does not
//! enable `--use_fast_math`. Q4_K uploads are losslessly repacked into the
//! decode layout. Q4_K GEMV uses q8_1 activations and one fixed warp reduction
//! per row. Attention uses fixed split reductions. Neither path uses atomics.
//!
//! Chunked prefill dequantizes one quantized matrix at a time into reusable
//! FP16 storage. cuBLASLt reads FP16 activations and weights, accumulates with
//! `CUBLAS_COMPUTE_32F`, and writes FP32 outputs. FP32 output keeps RMSNorm and
//! elementwise buffers common with decode. Attention folds each grouped-query
//! head group into the GEMM row dimension. The extra rounding occurs only at
//! the tensor-core inputs and the declared FP16 KV cache.

mod backend;
#[allow(unsafe_code)]
mod cuda;
mod error;
#[allow(unsafe_code)]
mod ffi;
mod repack;
mod shape;

pub use backend::{CudaBackend, CudaBuffer};
pub use cuda::{
    argmax, attention_decode, attention_decode_device_position, attention_decode_f16,
    attention_decode_f16_device_position, attention_decode_q8, attention_decode_q8_device_position,
    attention_prefill_f16, attention_prefill_f32, attention_prefill_q8, copy_f32_row,
    dequantize_k_f16, embedding_gather_batch, embedding_gather_q4_k,
    embedding_gather_q4_k_device_row, embedding_gather_q6_k, embedding_gather_q6_k_device_row,
    gemv_pair_q4_k, gemv_pair_swiglu_q4_k, gemv_q4_k, gemv_q4_k_residual, gemv_q6_k,
    gemv_q6_k_residual, increment_u32_scalar, kv_append, kv_append_chunk, kv_append_chunk_f16,
    kv_append_chunk_q8, kv_append_device_position, kv_append_f16, kv_append_f16_device_position,
    kv_append_q8, kv_append_q8_device_position, launch_q4_k_apron_pair_probe, launch_q4_k_probe,
    launch_q4_k_ring_probe, prefill_gemm, prepare_q4_k_probe, qk_norm_rope, qk_norm_rope_kv_append,
    qk_norm_rope_kv_append_device_position, qk_norm_rope_kv_append_f16,
    qk_norm_rope_kv_append_f16_device_position, qkv_gemv, residual_add, rms_norm,
    rms_norm_q8_parallel, rms_norm_residual, rms_norm_residual_store, rms_norm_rope,
    rms_norm_rope_device_position, rope_at_frequencies, rope_at_frequencies_device_position,
    rope_neox, rope_neox_at, swiglu, swiglu_q8, verify_gemv, write_u32_scalar, ArgmaxScratch,
    AttentionScratch, Context, CublasLt, DeviceBuffer, DeviceCopy, Event, GemvScratch, Graph,
    PrefillScratch, Q4KProbeGeometry, RopeScratch, Stream, PREPARED_ATTENTION_HEAD_DIM,
};
pub use error::{Error, Result};
pub use repack::{
    dequantize_q4_k, repack_q4_k, Q4_K_BLOCK_ELEMENTS, Q4_K_CODE_BYTES, Q4_K_GGUF_BLOCK_BYTES,
    Q4_K_METADATA_BYTES,
};
pub use shape::{
    AttentionShape, QuantFormat, QuantizedMatrixShape, RopeShape, VectorShape,
    ARGMAX_ITEMS_PER_THREAD, ATTENTION_BLOCK_SIZE, ATTENTION_KV_TILE, ATTENTION_SPLIT_KV_MAX,
    CUDA_BLOCK_SIZE,
};
