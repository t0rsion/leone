use half::f16;
use leone::{
    BatchSession, DecodeExecution, GenerateOptions, GenerationSession, KvCacheDtype, PrefillPlan,
    Runtime, Sampler,
};
use leone_cuda::{
    argmax, attention_decode, attention_decode_f16, attention_decode_q8, attention_prefill_f16,
    attention_prefill_q8, dequantize_k_f16, embedding_gather_q4_k, embedding_gather_q6_k,
    gemv_q4_k, gemv_q4_k_residual, gemv_q6_k, gemv_q6_k_residual, kv_append_chunk_q8,
    kv_append_f16, kv_append_q8, qk_norm_rope, qk_norm_rope_kv_append_f16, qkv_gemv, repack_q4_k,
    residual_add, rms_norm, rms_norm_q8_parallel, rms_norm_residual, rms_norm_residual_store,
    rms_norm_rope, rope_at_frequencies, rope_neox, swiglu, swiglu_q8, ArgmaxScratch,
    AttentionScratch, Context, CublasLt, CudaBackend, DeviceBuffer, GemvScratch, PrefillScratch,
    QuantFormat, QuantizedMatrixShape, RopeScratch, RopeShape, Stream, VectorShape,
};
use leone_gguf::ref_dequant;
use leone_gguf::{GgmlType, Gguf};
use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use std::error::Error;
use std::path::PathBuf;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct PrefillAttentionDeviceBuffers {
    query: DeviceBuffer<f32>,
    keys: DeviceBuffer<u16>,
    values: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
}

struct ProductionPrefillInputs {
    shape: leone_cuda::AttentionShape,
    plan: PrefillPlan,
    query: Vec<f32>,
    keys: Vec<u16>,
    values: Vec<u16>,
}

struct QkvDeviceOutputs {
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    scratch: GemvScratch,
}

struct RmsNormResults {
    plain: Vec<f32>,
    residual: Vec<f32>,
    stored: Vec<f32>,
    stored_norm: Vec<f32>,
}

struct RmsNormDeviceBuffers {
    left: DeviceBuffer<f32>,
    right: DeviceBuffer<f32>,
    weight: DeviceBuffer<f32>,
    plain: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    stored: DeviceBuffer<f32>,
    stored_norm: DeviceBuffer<f32>,
}

struct ParallelRmsResults {
    standalone: Vec<f32>,
    parallel: Vec<f32>,
    standalone_projection: Vec<f32>,
    parallel_projection: Vec<f32>,
}

struct ParallelRmsBuffers {
    input: DeviceBuffer<f32>,
    weight: DeviceBuffer<f32>,
    gemv_weights: DeviceBuffer<u8>,
    standalone: DeviceBuffer<f32>,
    parallel: DeviceBuffer<f32>,
    standalone_projection: DeviceBuffer<f32>,
    parallel_projection: DeviceBuffer<f32>,
    standalone_scratch: GemvScratch,
    parallel_scratch: GemvScratch,
}

struct ParallelRmsOutputBuffers {
    standalone: DeviceBuffer<f32>,
    parallel: DeviceBuffer<f32>,
    standalone_projection: DeviceBuffer<f32>,
    parallel_projection: DeviceBuffer<f32>,
    standalone_scratch: GemvScratch,
    parallel_scratch: GemvScratch,
}

struct QkNormResults {
    query_actual: Vec<f32>,
    key_actual: Vec<f32>,
    fused_query: Vec<f32>,
    fused_key: Vec<f32>,
    key_cache: Vec<u16>,
    value_cache: Vec<u16>,
    fused_key_cache: Vec<u16>,
    fused_value_cache: Vec<u16>,
}

struct QkNormInputBuffers {
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    query_weight: DeviceBuffer<f32>,
    key_weight: DeviceBuffer<f32>,
}

struct QkNormOutputBuffers {
    query_output: DeviceBuffer<f32>,
    key_output: DeviceBuffer<f32>,
    fused_query_output: DeviceBuffer<f32>,
    fused_key_output: DeviceBuffer<f32>,
    key_cache: DeviceBuffer<u16>,
    value_cache: DeviceBuffer<u16>,
    fused_key_cache: DeviceBuffer<u16>,
    fused_value_cache: DeviceBuffer<u16>,
}

struct SwigluAddBuffers {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    swiglu_output: DeviceBuffer<f32>,
    add_output: DeviceBuffer<f32>,
}

struct GemvEpilogueBuffers {
    weights: DeviceBuffer<u8>,
    input: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    projection: DeviceBuffer<f32>,
    composed: DeviceBuffer<f32>,
    fused: DeviceBuffer<f32>,
    composed_scratch: GemvScratch,
    fused_scratch: GemvScratch,
}

struct SwigluQ8Buffers {
    weights: DeviceBuffer<u8>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    standalone: DeviceBuffer<f32>,
    fused: DeviceBuffer<f32>,
    standalone_projection: DeviceBuffer<f32>,
    fused_projection: DeviceBuffer<f32>,
    standalone_scratch: GemvScratch,
    fused_scratch: GemvScratch,
}

struct SwigluQ8Outputs {
    standalone: DeviceBuffer<f32>,
    fused: DeviceBuffer<f32>,
    standalone_projection: DeviceBuffer<f32>,
    fused_projection: DeviceBuffer<f32>,
    standalone_scratch: GemvScratch,
    fused_scratch: GemvScratch,
}

struct F16KvResults {
    actual: Vec<f32>,
    prepared_output: Vec<f32>,
    standalone_projection: Vec<f32>,
    prepared_projection: Vec<f32>,
}

struct F16DownstreamBuffers {
    weights: DeviceBuffer<u8>,
    prepared_output: DeviceBuffer<f32>,
    standalone_projection: DeviceBuffer<f32>,
    prepared_projection: DeviceBuffer<f32>,
    prepared_scratch: GemvScratch,
    standalone_scratch: GemvScratch,
}

struct Q8ChunkedPrefillResults {
    chunk_key_bytes: Vec<u8>,
    chunk_value_bytes: Vec<u8>,
    scalar_key_bytes: Vec<u8>,
    scalar_value_bytes: Vec<u8>,
    actual: Vec<f32>,
}

struct Q8ChunkedCache {
    chunk_key_bytes: Vec<u8>,
    chunk_value_bytes: Vec<u8>,
    scalar_key_bytes: Vec<u8>,
    scalar_value_bytes: Vec<u8>,
    chunk_keys: DeviceBuffer<u8>,
    chunk_values: DeviceBuffer<u8>,
}

struct Q8ChunkedCacheBuffers {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    chunk_keys: DeviceBuffer<u8>,
    chunk_values: DeviceBuffer<u8>,
    scalar_keys: DeviceBuffer<u8>,
    scalar_values: DeviceBuffer<u8>,
}

struct GemvBuffers {
    weights: DeviceBuffer<u8>,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    scratch: GemvScratch,
}

fn assert_f32_bitwise_equal(left: &[f32], right: &[f32], label: &str) {
    assert_eq!(left.len(), right.len(), "{label} length");
    if let Some((index, (left_value, right_value))) = left
        .iter()
        .zip(right)
        .enumerate()
        .find(|(_, (left_value, right_value))| left_value.to_bits() != right_value.to_bits())
    {
        let mismatch_count = left
            .iter()
            .zip(right)
            .filter(|(left_value, right_value)| left_value.to_bits() != right_value.to_bits())
            .count();
        panic!(
            "{label} changed at {index}: {left_value:e} != {right_value:e}; {mismatch_count} mismatches"
        );
    }
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn prefill_dequant_matches_scalar_block_decoders() -> TestResult {
    for (format, seed) in [
        (QuantFormat::Q4K, 0x7072_6566_7134_u64),
        (QuantFormat::Q6K, 0x7072_6566_7136_u64),
    ] {
        prefill_dequant_case(format, seed)?;
    }
    Ok(())
}

fn prefill_dequant_case(format: QuantFormat, seed: u64) -> TestResult {
    let shape = QuantizedMatrixShape::new(3, 512, format)?;
    let source = quantized_bytes(shape, seed);
    let observed = run_prefill_dequant_device(&source, shape)?;
    let expected = prefill_dequant_expected(&source, shape, format)?;
    assert_eq!(observed, expected, "{format:?} FP16 dequantization");
    Ok(())
}

fn run_prefill_dequant_device(source: &[u8], shape: QuantizedMatrixShape) -> TestResult<Vec<u16>> {
    let device_source = device_weight_bytes(source, shape)?;
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let d_source = context.copy_to_device(&device_source)?;
    let mut d_output = context.alloc(shape.rows() * shape.columns())?;
    dequantize_k_f16(&stream, &d_source, &mut d_output, shape)?;
    stream.synchronize()?;
    let mut observed = vec![0_u16; shape.rows() * shape.columns()];
    d_output.copy_to(&mut observed)?;
    Ok(observed)
}

fn prefill_dequant_expected(
    source: &[u8],
    shape: QuantizedMatrixShape,
    format: QuantFormat,
) -> TestResult<Vec<u16>> {
    let mut expected = Vec::with_capacity(shape.rows() * shape.columns());
    for row in source.chunks_exact(shape.row_bytes()) {
        let decoded = match format {
            QuantFormat::Q4K => ref_dequant::q4_k::dequant_row(row, shape.columns())?,
            QuantFormat::Q6K => ref_dequant::q6_k::dequant_row(row, shape.columns())?,
        };
        expected.extend(
            decoded
                .into_iter()
                .map(|value| f16::from_f32(value).to_bits()),
        );
    }
    Ok(expected)
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn prefill_gemm_matches_fp16_input_and_weight_oracle() -> TestResult {
    let shape = QuantizedMatrixShape::new(37, 512, QuantFormat::Q4K)?;
    let tokens = 7;
    let weights = quantized_bytes(shape, 0x7072_6566_6765_6d6d);
    let input = random_f32(tokens * shape.columns(), 0x7072_6566_696e_7074, -0.25..0.25);
    let (actual, repeated) = run_prefill_gemm_device(&weights, &input, shape, tokens)?;
    assert_f32_bitwise_equal(&actual, &repeated, "prefill GEMM repeat");
    let expected = prefill_gemm_expected(&weights, &input, shape, tokens)?;
    let errors = assert_close("prefill GEMM", &actual, &expected, 2e-2, 2e-3);
    eprintln!("prefill GEMM {errors}");
    Ok(())
}

fn run_prefill_gemm_device(
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
    tokens: usize,
) -> TestResult<(Vec<f32>, Vec<f32>)> {
    let device_weights = device_weight_bytes(weights, shape)?;
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let plan = PrefillPlan::new(tokens, tokens, 4, 2, 32, 512, 512, 512)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let d_weights = context.copy_to_device(&device_weights)?;
    let d_input = context.copy_to_device(input)?;
    let mut d_output = context.alloc(tokens * shape.rows())?;
    prefill_gemm_repeated(
        &handle,
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        shape,
        tokens,
        &mut scratch,
    )
}

#[allow(clippy::too_many_arguments)]
fn prefill_gemm_repeated(
    handle: &CublasLt,
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> TestResult<(Vec<f32>, Vec<f32>)> {
    leone_cuda::prefill_gemm(
        handle, stream, weights, input, output, shape, tokens, scratch,
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; tokens * shape.rows()];
    output.copy_to(&mut actual)?;
    leone_cuda::prefill_gemm(
        handle, stream, weights, input, output, shape, tokens, scratch,
    )?;
    stream.synchronize()?;
    let mut repeated = vec![0.0; actual.len()];
    output.copy_to(&mut repeated)?;
    Ok((actual, repeated))
}

fn prefill_gemm_expected(
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
    tokens: usize,
) -> TestResult<Vec<f64>> {
    let decoded = (0..shape.rows())
        .map(|row| {
            let start = row * shape.row_bytes();
            dequant_row(
                shape.format(),
                &weights[start..start + shape.row_bytes()],
                shape.columns(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut expected = vec![0.0_f64; tokens * shape.rows()];
    for token in 0..tokens {
        for row in 0..shape.rows() {
            expected[token * shape.rows() + row] = decoded[row]
                .iter()
                .zip(&input[token * shape.columns()..(token + 1) * shape.columns()])
                .map(|(weight, input)| {
                    f64::from(f16::from_f32(*weight).to_f32())
                        * f64::from(f16::from_f32(*input).to_f32())
                })
                .sum();
        }
    }
    Ok(expected)
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn real_prefill_gemm_is_bitwise_deterministic() -> TestResult {
    let (first, second) = run_real_prefill_gemm()?;
    assert_f32_bitwise_equal(&first, &second, "real prefill GEMM repeat");
    Ok(())
}

fn run_real_prefill_gemm() -> TestResult<(Vec<f32>, Vec<f32>)> {
    let (shape, weights) = real_prefill_weights()?;
    let tokens = 512;
    let input = random_f32(tokens * shape.columns(), 0x7072_6566_7265_616c, -0.25..0.25);
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let plan = PrefillPlan::new(tokens, tokens, 32, 8, 128, 4_096, 12_288, 12_288)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let (d_weights, d_input, mut d_output) =
        real_prefill_buffers(&context, &weights, &input, shape, tokens)?;
    prefill_gemm_repeated(
        &handle,
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        shape,
        tokens,
        &mut scratch,
    )
}

fn real_prefill_buffers(
    context: &Context,
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
    tokens: usize,
) -> TestResult<(DeviceBuffer<u8>, DeviceBuffer<f32>, DeviceBuffer<f32>)> {
    let device_weights = device_weight_bytes(weights, shape)?;
    Ok((
        context.copy_to_device(&device_weights)?,
        context.copy_to_device(input)?,
        context.alloc(tokens * shape.rows())?,
    ))
}

fn real_prefill_weights() -> TestResult<(QuantizedMatrixShape, Vec<u8>)> {
    let gguf = Gguf::open(model_path())?;
    let tensor_name = "blk.0.ffn_gate.weight";
    let shape = QuantizedMatrixShape::new(12_288, 4_096, QuantFormat::Q4K)?;
    let weights = gguf.tensor_data(tensor_name)?;
    Ok((shape, weights))
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn prefill_attention_matches_causal_fp16_oracle() -> TestResult {
    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 32;
    let max_context = 1030;
    let start_position = 2;
    let tokens = 1025;
    let shape = leone_cuda::AttentionShape::new(n_head, n_head_kv, head_dim, max_context)?;
    let plan = PrefillPlan::new(
        tokens,
        start_position + tokens,
        n_head,
        n_head_kv,
        head_dim,
        128,
        256,
        256,
    )?;
    let query = random_f32(
        tokens * shape.query_elements(),
        0x7072_6566_6174_746e,
        -0.5..0.5,
    );
    let cache_elements = shape.cache_elements();
    let keys = random_f32(cache_elements, 0x7072_6566_6b65_7973, -0.5..0.5)
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let values = random_f32(cache_elements, 0x7072_6566_7661_6c73, -0.5..0.5)
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let (actual, repeated) =
        run_prefill_attention_device(&query, &keys, &values, shape, start_position, tokens, plan)?;
    assert_f32_bitwise_equal(&actual, &repeated, "prefill attention repeat");
    let keys = keys
        .into_iter()
        .map(|bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let values = values
        .into_iter()
        .map(|bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = prefill_attention_oracle(&query, &keys, &values, shape, start_position, tokens);
    let errors = assert_close("prefill attention", &actual, &expected, 2e-3, 2e-2);
    eprintln!("prefill attention {errors}");
    Ok(())
}

fn run_prefill_attention_device(
    query: &[f32],
    keys: &[u16],
    values: &[u16],
    shape: leone_cuda::AttentionShape,
    start_position: usize,
    tokens: usize,
    plan: PrefillPlan,
) -> TestResult<(Vec<f32>, Vec<f32>)> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let PrefillAttentionDeviceBuffers {
        query: d_query,
        keys: d_keys,
        values: d_values,
        output: mut d_output,
    } = prefill_attention_device_buffers(&context, query, keys, values, shape, tokens)?;
    prefill_attention_repeated(
        &handle,
        &stream,
        &d_query,
        &d_keys,
        &d_values,
        &mut d_output,
        shape,
        start_position,
        tokens,
        &mut scratch,
    )
}

fn prefill_attention_device_buffers(
    context: &Context,
    query: &[f32],
    keys: &[u16],
    values: &[u16],
    shape: leone_cuda::AttentionShape,
    tokens: usize,
) -> TestResult<PrefillAttentionDeviceBuffers> {
    Ok(PrefillAttentionDeviceBuffers {
        query: context.copy_to_device(query)?,
        keys: context.copy_to_device(keys)?,
        values: context.copy_to_device(values)?,
        output: context.alloc(tokens * shape.query_elements())?,
    })
}

#[allow(clippy::too_many_arguments)]
fn prefill_attention_repeated(
    handle: &CublasLt,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    keys: &DeviceBuffer<u16>,
    values: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    shape: leone_cuda::AttentionShape,
    start_position: usize,
    tokens: usize,
    scratch: &mut PrefillScratch,
) -> TestResult<(Vec<f32>, Vec<f32>)> {
    attention_prefill_f16(
        handle,
        stream,
        query,
        keys,
        values,
        output,
        shape,
        start_position,
        tokens,
        scratch,
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; tokens * shape.query_elements()];
    output.copy_to(&mut actual)?;
    attention_prefill_f16(
        handle,
        stream,
        query,
        keys,
        values,
        output,
        shape,
        start_position,
        tokens,
        scratch,
    )?;
    stream.synchronize()?;
    let mut repeated = vec![0.0; actual.len()];
    output.copy_to(&mut repeated)?;
    Ok((actual, repeated))
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn production_prefill_attention_is_bitwise_deterministic() -> TestResult {
    let (first, second) = run_production_prefill_attention()?;
    assert_f32_bitwise_equal(&first, &second, "production prefill attention repeat");
    Ok(())
}

fn run_production_prefill_attention() -> TestResult<(Vec<f32>, Vec<f32>)> {
    let tokens = 512;
    let ProductionPrefillInputs {
        shape,
        plan,
        query,
        keys,
        values,
    } = production_prefill_inputs(tokens)?;
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let PrefillAttentionDeviceBuffers {
        query: d_query,
        keys: d_keys,
        values: d_values,
        output: mut d_output,
    } = prefill_attention_device_buffers(&context, &query, &keys, &values, shape, tokens)?;
    prefill_attention_repeated(
        &handle,
        &stream,
        &d_query,
        &d_keys,
        &d_values,
        &mut d_output,
        shape,
        0,
        tokens,
        &mut scratch,
    )
}

fn production_prefill_inputs(tokens: usize) -> TestResult<ProductionPrefillInputs> {
    let shape = leone_cuda::AttentionShape::new(32, 8, 128, tokens)?;
    let plan = PrefillPlan::new(tokens, tokens, 32, 8, 128, 4_096, 12_288, 12_288)?;
    let query = random_f32(
        tokens * shape.query_elements(),
        0x7072_6f64_6174_746e,
        -0.5..0.5,
    );
    let keys = random_f32(shape.cache_elements(), 0x7072_6f64_6b65_7973, -0.5..0.5)
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect();
    let values = random_f32(shape.cache_elements(), 0x7072_6f64_7661_6c73, -0.5..0.5)
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect();
    Ok(ProductionPrefillInputs {
        shape,
        plan,
        query,
        keys,
        values,
    })
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn chunked_prefill_characterizes_kv_and_greedy_tokens() -> TestResult {
    const KV_MAX_ABS_TOLERANCE: f32 = 0.5;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, model_path())?;
    let prompt = vec![1_u32; 512];
    let result = runtime.characterize_prefill(&prompt, 32, 512)?;
    for layer in &result.layers {
        eprintln!(
            "layer {:02}: key abs={:.6e} rel={:.6e}; value abs={:.6e} rel={:.6e}",
            layer.layer,
            layer.key_max_abs,
            layer.key_max_rel,
            layer.value_max_abs,
            layer.value_max_rel,
        );
        assert!(layer.key_max_abs.is_finite());
        assert!(layer.key_max_rel.is_finite());
        assert!(layer.value_max_abs.is_finite());
        assert!(layer.value_max_rel.is_finite());
        assert!(
            layer.key_max_abs <= KV_MAX_ABS_TOLERANCE,
            "layer {} key max absolute error exceeds {KV_MAX_ABS_TOLERANCE}",
            layer.layer,
        );
        assert!(
            layer.value_max_abs <= KV_MAX_ABS_TOLERANCE,
            "layer {} value max absolute error exceeds {KV_MAX_ABS_TOLERANCE}",
            layer.layer,
        );
    }
    for layer in &result.repeated_layers {
        assert_eq!(layer.key_max_abs, 0.0, "repeat layer {} key", layer.layer);
        assert_eq!(
            layer.value_max_abs, 0.0,
            "repeat layer {} value",
            layer.layer
        );
    }
    eprintln!("repeated chunked prefill KV matches bitwise across all layers");
    let token_matches = result
        .sequential_tokens
        .iter()
        .zip(&result.chunked_tokens)
        .take_while(|(left, right)| left == right)
        .count();
    eprintln!(
        "sequential vs chunked greedy prefix: {token_matches}/{}",
        result.sequential_tokens.len()
    );
    assert_eq!(
        result.sequential_tokens, result.chunked_tokens,
        "chunked prefill changed the 32-token greedy continuation"
    );
    assert_eq!(
        result.chunked_tokens, result.repeated_chunked_tokens,
        "chunked prefill continuation changed between identical runs"
    );
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn chunked_prefill_cancels_between_position_blocks() -> TestResult {
    let mut runtime = Runtime::load(CudaBackend::new(0)?, model_path())?;
    let prompt = " one".repeat(leone::DEFAULT_PREFILL_CHUNK_TOKENS + 1_024);
    let mut cancellation_checks = 0;
    let result = runtime.generate(
        &prompt,
        GenerateOptions::greedy(1),
        |_| panic!("cancelled prefill must not emit a token"),
        || {
            cancellation_checks += 1;
            cancellation_checks == 2
        },
    )?;
    assert!(result.stats.cancelled);
    assert_eq!(result.stats.emitted_tokens, 0);
    assert_eq!(
        result.stats.prompt_tokens,
        leone::DEFAULT_PREFILL_CHUNK_TOKENS
    );
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn q4_k_gemv_matches_f64_oracle() -> TestResult {
    // q8_1 activation rounding dominates this synthetic case. Random block
    // codes also cause output cancellation, so the bound includes an absolute
    // term. Real model tensors have a separate relative-error gate below.
    let shape = QuantizedMatrixShape::new(37, 1_024, QuantFormat::Q4K)?;
    let weights = quantized_bytes(shape, 0x7134_5f67_656d_7634);
    let input = random_f32(shape.columns(), 0x7134_5f69_6e70_7574, -0.5..0.5);
    let expected = gemv_oracle(&weights, &input, shape)?;
    let actual = run_gemv(&weights, &input, shape)?;
    let errors = assert_close("Q4_K GEMV", &actual, &expected, 1.0, 2e-2);
    eprintln!("Q4_K GEMV {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn q6_k_gemv_matches_f64_oracle() -> TestResult {
    // q8_1 activation rounding dominates this synthetic case. Random block
    // codes also cause output cancellation, so the bound includes an absolute
    // term. Real model tensors have a separate relative-error gate below.
    let shape = QuantizedMatrixShape::new(37, 1_024, QuantFormat::Q6K)?;
    let weights = quantized_bytes(shape, 0x7136_5f67_656d_7636);
    let input = random_f32(shape.columns(), 0x7136_5f69_6e70_7574, -0.25..0.25);
    let expected = gemv_oracle(&weights, &input, shape)?;
    let actual = run_gemv(&weights, &input, shape)?;
    let errors = assert_close("Q6_K GEMV", &actual, &expected, 1.0, 2e-2);
    eprintln!("Q6_K GEMV {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn multi_gemv_matches_single_gemv_bitwise() -> TestResult {
    for (format, columns, seed) in [
        (QuantFormat::Q4K, 4_096, 0x6d75_6c74_695f_7134),
        (QuantFormat::Q4K, 12_288, 0x6d75_6c74_695f_3438),
        (QuantFormat::Q4K, 1_024, 0x6d75_6c74_695f_7130),
        (QuantFormat::Q6K, 4_096, 0x6d75_6c74_695f_7136),
    ] {
        multi_gemv_case(format, columns, seed)?;
    }
    Ok(())
}

fn multi_gemv_case(format: QuantFormat, columns: usize, seed: u64) -> TestResult {
    let shape = QuantizedMatrixShape::new(37, columns, format)?;
    let weights = quantized_bytes(shape, seed);
    let device_weights = device_weight_bytes(&weights, shape)?;
    for positions in 1..=8 {
        multi_gemv_position(
            &weights,
            &device_weights,
            shape,
            columns,
            seed,
            format,
            positions,
        )?;
    }
    Ok(())
}

fn multi_gemv_position(
    weights: &[u8],
    device_weights: &[u8],
    shape: QuantizedMatrixShape,
    columns: usize,
    seed: u64,
    format: QuantFormat,
    positions: usize,
) -> TestResult {
    let input = random_f32(positions * columns, seed ^ positions as u64, -0.25..0.25);
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let d_weights = context.copy_to_device(device_weights)?;
    let d_input = context.copy_to_device(&input)?;
    let mut d_output = context.alloc(positions * shape.rows())?;
    let mut scratch = GemvScratch::new_multi(&context, columns, positions)?;
    let actual = run_multi_gemv(
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        &mut scratch,
        shape,
        positions,
    )?;
    compare_multi_gemv(&actual, weights, &input, shape, positions, format)
}

#[allow(clippy::too_many_arguments)]
fn run_multi_gemv(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
    positions: usize,
) -> TestResult<Vec<f32>> {
    leone_cuda::verify_gemv(
        stream, weights, input, None, output, scratch, shape, positions,
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; positions * shape.rows()];
    output.copy_to(&mut actual)?;
    Ok(actual)
}

fn compare_multi_gemv(
    actual: &[f32],
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
    positions: usize,
    format: QuantFormat,
) -> TestResult {
    for position in 0..positions {
        let expected = run_gemv(
            weights,
            &input[position * shape.columns()..(position + 1) * shape.columns()],
            shape,
        )?;
        assert_f32_bitwise_equal(
            &actual[position * shape.rows()..(position + 1) * shape.rows()],
            &expected,
            &format!("{format:?} width {positions} position {position}"),
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn verify_logits_match_sequential_decode_bitwise() -> TestResult {
    let mut runtime = Runtime::load(CudaBackend::new(0)?, model_path())?;
    for dtype in [KvCacheDtype::F16, KvCacheDtype::F32] {
        for depth in [32, 512] {
            let prompt = vec![1_u32; depth];
            for width in 1..=8 {
                let input = vec![1_u32; width];
                let result = runtime.characterize_verify(&prompt, &input, dtype)?;
                assert_eq!(
                    result.mismatching_floats, 0,
                    "{dtype:?} depth {depth} width {width}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn qkv_gemv_matches_three_f64_oracles() -> TestResult {
    // All projections use q8_1 activations. Random block codes need an
    // absolute term near zero.
    let query_shape = QuantizedMatrixShape::new(37, 1_024, QuantFormat::Q4K)?;
    let key_shape = QuantizedMatrixShape::new(11, 1_024, QuantFormat::Q6K)?;
    let value_shape = QuantizedMatrixShape::new(11, 1_024, QuantFormat::Q4K)?;
    let query_weights = quantized_bytes(query_shape, 0x716b_765f_7177_6774);
    let key_weights = quantized_bytes(key_shape, 0x716b_765f_6b77_6774);
    let value_weights = quantized_bytes(value_shape, 0x716b_765f_7677_6774);
    let input = random_f32(1_024, 0x716b_765f_696e_7074, -0.25..0.25);
    let (query, key, value) = run_qkv_gemv_device(
        &query_weights,
        query_shape,
        &key_weights,
        key_shape,
        &value_weights,
        value_shape,
        &input,
    )?;
    let query_expected = gemv_oracle(&query_weights, &input, query_shape)?;
    let key_expected = gemv_oracle(&key_weights, &input, key_shape)?;
    let value_expected = gemv_oracle(&value_weights, &input, value_shape)?;
    let query_errors = assert_close("QKV query", &query, &query_expected, 1.0, 2e-2);
    let key_errors = assert_close("QKV key", &key, &key_expected, 1.0, 2e-2);
    let value_errors = assert_close("QKV value", &value, &value_expected, 1.0, 2e-2);
    eprintln!("QKV query {query_errors}; key {key_errors}; value {value_errors}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_qkv_gemv_device(
    query_weights: &[u8],
    query_shape: QuantizedMatrixShape,
    key_weights: &[u8],
    key_shape: QuantizedMatrixShape,
    value_weights: &[u8],
    value_shape: QuantizedMatrixShape,
    input: &[f32],
) -> TestResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let (d_query_weights, d_key_weights, d_value_weights) = qkv_device_weights(
        &context,
        query_weights,
        query_shape,
        key_weights,
        value_weights,
        value_shape,
    )?;
    let d_input = context.copy_to_device(input)?;
    let QkvDeviceOutputs {
        query: mut d_query,
        key: mut d_key,
        value: mut d_value,
        mut scratch,
    } = qkv_device_outputs(&context, query_shape, key_shape, value_shape)?;
    run_qkv_gemv_once(
        &stream,
        &d_query_weights,
        query_shape,
        &d_key_weights,
        key_shape,
        &d_value_weights,
        value_shape,
        &d_input,
        &mut d_query,
        &mut d_key,
        &mut d_value,
        &mut scratch,
    )?;
    stream.synchronize()?;
    copy_qkv_outputs(
        &d_query,
        &d_key,
        &d_value,
        query_shape,
        key_shape,
        value_shape,
    )
}

fn copy_qkv_outputs(
    query: &DeviceBuffer<f32>,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    query_shape: QuantizedMatrixShape,
    key_shape: QuantizedMatrixShape,
    value_shape: QuantizedMatrixShape,
) -> TestResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    Ok((
        copy_f32_buffer(query, query_shape.rows())?,
        copy_f32_buffer(key, key_shape.rows())?,
        copy_f32_buffer(value, value_shape.rows())?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn qkv_device_weights(
    context: &Context,
    query_weights: &[u8],
    query_shape: QuantizedMatrixShape,
    key_weights: &[u8],
    value_weights: &[u8],
    value_shape: QuantizedMatrixShape,
) -> TestResult<(DeviceBuffer<u8>, DeviceBuffer<u8>, DeviceBuffer<u8>)> {
    let query_device_weights = device_weight_bytes(query_weights, query_shape)?;
    let value_device_weights = device_weight_bytes(value_weights, value_shape)?;
    Ok((
        context.copy_to_device(&query_device_weights)?,
        context.copy_to_device(key_weights)?,
        context.copy_to_device(&value_device_weights)?,
    ))
}

fn qkv_device_outputs(
    context: &Context,
    query_shape: QuantizedMatrixShape,
    key_shape: QuantizedMatrixShape,
    value_shape: QuantizedMatrixShape,
) -> TestResult<QkvDeviceOutputs> {
    Ok(QkvDeviceOutputs {
        query: context.alloc(query_shape.rows())?,
        key: context.alloc(key_shape.rows())?,
        value: context.alloc(value_shape.rows())?,
        scratch: GemvScratch::new(context, query_shape)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_qkv_gemv_once(
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
) -> TestResult {
    qkv_gemv(
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
        false,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn real_q4_k_4096_square_gemv_matches_spot_oracle() -> TestResult {
    // Model codes, not random blocks. Spot rows use a 1e-3 absolute and 1e-2
    // relative bound. Relative error is also gated for references above 1e-2.
    let gguf = Gguf::open(model_path())?;
    let tensor_name = "blk.0.attn_output.weight";
    let tensor = gguf.tensor(tensor_name).ok_or("missing attention output")?;
    assert_eq!(tensor.dtype, GgmlType::Q4_K);
    assert_eq!(tensor.shape, [4_096, 4_096]);
    let shape = QuantizedMatrixShape::new(4_096, 4_096, QuantFormat::Q4K)?;
    let weights = gguf.tensor_data(tensor_name)?;
    let input = random_f32(shape.columns(), 0x7265_616c_5f71_346b, -0.1..0.1);
    let actual = run_gemv(&weights, &input, shape)?;
    let rows = [0, 1, 2_047, 4_095];
    let expected = gemv_spot_oracle(&weights, &input, shape, &rows)?;
    let observed = rows.map(|row| actual[row]);
    let errors = assert_close("real Q4_K GEMV", &observed, &expected, 1e-3, 1e-2);
    assert!(errors.max_abs <= 1e-3, "real Q4_K {errors}");
    assert!(errors.max_rel_above_floor <= 1e-2, "real Q4_K {errors}");
    eprintln!("real Q4_K 4096x4096 GEMV {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn real_q4_k_production_shapes_match_spot_oracles() -> TestResult {
    let gguf = Gguf::open(model_path())?;
    let cases = [
        ("blk.0.attn_k.weight", 1_024, 4_096, 0x7134_5f31_u64),
        ("blk.0.attn_output.weight", 4_096, 4_096, 0x7134_5f32),
        ("blk.0.ffn_gate.weight", 12_288, 4_096, 0x7134_5f33),
        ("blk.4.ffn_down.weight", 4_096, 12_288, 0x7134_5f34),
    ];
    for (tensor_name, rows, columns, seed) in cases {
        let tensor = gguf.tensor(tensor_name).ok_or("missing Q4_K tensor")?;
        assert_eq!(tensor.dtype, GgmlType::Q4_K);
        assert_eq!(tensor.shape, [columns as u64, rows as u64]);
        let shape = QuantizedMatrixShape::new(rows, columns, QuantFormat::Q4K)?;
        let weights = gguf.tensor_data(tensor_name)?;
        let input = random_f32(columns, seed, -0.1..0.1);
        let actual = run_gemv(&weights, &input, shape)?;
        let spot_rows = [0, 1, rows / 3, rows / 2, rows - 2, rows - 1];
        let expected = gemv_spot_oracle(&weights, &input, shape, &spot_rows)?;
        let observed = spot_rows.map(|row| actual[row]);
        let errors = assert_close(tensor_name, &observed, &expected, 5e-3, 1e-2);
        assert!(errors.relative_l_inf() <= 1e-2, "{tensor_name} {errors}");
        eprintln!("real Q4_K {rows}x{columns} GEMV {errors}");
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn real_q6_k_vocab_gemv_matches_spot_oracle() -> TestResult {
    // Full 151,936 by 4,096 production grid. Three rows use a 2e-3 absolute
    // and 1e-3 relative oracle bound.
    let gguf = Gguf::open(model_path())?;
    let tensor_name = "output.weight";
    let tensor = gguf.tensor(tensor_name).ok_or("missing output weight")?;
    assert_eq!(tensor.dtype, GgmlType::Q6_K);
    assert_eq!(tensor.shape, [4_096, 151_936]);
    let shape = QuantizedMatrixShape::new(151_936, 4_096, QuantFormat::Q6K)?;
    let weights = gguf.tensor_data(tensor_name)?;
    let input = random_f32(shape.columns(), 0x7265_616c_5f71_366b, -0.05..0.05);
    let actual = run_gemv(&weights, &input, shape)?;
    let rows = [0, 75_968, 151_935];
    let expected = gemv_spot_oracle(&weights, &input, shape, &rows)?;
    let observed = rows.map(|row| actual[row]);
    let errors = assert_close("real Q6_K vocab GEMV", &observed, &expected, 2e-3, 1e-3);
    eprintln!("real Q6_K 151936x4096 GEMV {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn rms_norm_variants_match_f64_oracle() -> TestResult {
    // The f32 tree reduction uses a 3e-6 absolute and 2e-5 relative bound
    // against the sequential f64 sum.
    let shape = VectorShape::new(3, 4_096)?;
    let left = random_f32(shape.elements(), 0x726d_736e_5f6c_6566, -1.0..1.0);
    let right = random_f32(shape.elements(), 0x726d_736e_5f72_6967, -0.5..0.5);
    let weight = random_f32(shape.columns(), 0x726d_736e_5f77_6768, 0.5..1.5);
    let epsilon = 1e-6_f32;
    let RmsNormResults {
        plain,
        residual,
        stored,
        stored_norm,
    } = run_rms_norm_variants(&left, &right, &weight, shape, epsilon)?;
    let plain_expected = rms_oracle(&left, None, &weight, shape, epsilon);
    let residual_expected = rms_oracle(&left, Some(&right), &weight, shape, epsilon);
    let plain_errors = assert_close("RMSNorm", &plain, &plain_expected, 3e-6, 2e-5);
    let residual_errors = assert_close(
        "residual RMSNorm",
        &residual,
        &residual_expected,
        3e-6,
        2e-5,
    );
    let stored_errors = assert_close(
        "stored residual RMSNorm",
        &stored_norm,
        &residual_expected,
        3e-6,
        2e-5,
    );
    for (index, ((left, right), actual)) in left.iter().zip(&right).zip(&stored).enumerate() {
        assert_eq!(
            actual.to_bits(),
            (left + right).to_bits(),
            "stored residual {index}"
        );
    }
    eprintln!("RMSNorm {plain_errors}; residual RMSNorm {residual_errors}; stored {stored_errors}");
    Ok(())
}

fn run_rms_norm_variants(
    left: &[f32],
    right: &[f32],
    weight: &[f32],
    shape: VectorShape,
    epsilon: f32,
) -> TestResult<RmsNormResults> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let RmsNormDeviceBuffers {
        left: d_left,
        right: d_right,
        weight: d_weight,
        plain: mut d_plain,
        residual: mut d_residual,
        stored: mut d_stored,
        stored_norm: mut d_stored_norm,
    } = rms_norm_device_buffers(&context, left, right, weight, shape)?;
    run_rms_norm_variants_once(
        &stream,
        &d_left,
        &d_right,
        &d_weight,
        &mut d_plain,
        &mut d_residual,
        &mut d_stored,
        &mut d_stored_norm,
        shape,
        epsilon,
    )?;
    stream.synchronize()?;
    let mut plain = vec![0.0; shape.elements()];
    let mut residual = vec![0.0; shape.elements()];
    let mut stored = vec![0.0; shape.elements()];
    let mut stored_norm = vec![0.0; shape.elements()];
    d_plain.copy_to(&mut plain)?;
    d_residual.copy_to(&mut residual)?;
    d_stored.copy_to(&mut stored)?;
    d_stored_norm.copy_to(&mut stored_norm)?;
    Ok(RmsNormResults {
        plain,
        residual,
        stored,
        stored_norm,
    })
}

fn rms_norm_device_buffers(
    context: &Context,
    left: &[f32],
    right: &[f32],
    weight: &[f32],
    shape: VectorShape,
) -> TestResult<RmsNormDeviceBuffers> {
    Ok(RmsNormDeviceBuffers {
        left: context.copy_to_device(left)?,
        right: context.copy_to_device(right)?,
        weight: context.copy_to_device(weight)?,
        plain: context.alloc(shape.elements())?,
        residual: context.alloc(shape.elements())?,
        stored: context.alloc(shape.elements())?,
        stored_norm: context.alloc(shape.elements())?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_rms_norm_variants_once(
    stream: &Stream,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    plain: &mut DeviceBuffer<f32>,
    residual: &mut DeviceBuffer<f32>,
    stored: &mut DeviceBuffer<f32>,
    stored_norm: &mut DeviceBuffer<f32>,
    shape: VectorShape,
    epsilon: f32,
) -> TestResult {
    rms_norm(stream, left, weight, plain, shape, epsilon)?;
    rms_norm_residual(stream, left, right, weight, residual, shape, epsilon)?;
    rms_norm_residual_store(
        stream,
        left,
        right,
        weight,
        stored,
        stored_norm,
        shape,
        epsilon,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn parallel_rms_norm_q8_matches_standalone_path_bitwise() -> TestResult {
    parallel_rms_case()
}

fn parallel_rms_case() -> TestResult {
    let vector_shape = VectorShape::new(1, 4_096)?;
    let gemv_shape = QuantizedMatrixShape::new(67, 4_096, QuantFormat::Q4K)?;
    let input = random_f32(vector_shape.elements(), 0x726d_735f_7138_696e, -1.0..1.0);
    let weight = random_f32(vector_shape.columns(), 0x726d_735f_7138_7767, 0.5..1.5);
    let gemv_weights = quantized_bytes(gemv_shape, 0x726d_735f_7138_6776);
    let ParallelRmsResults {
        standalone,
        parallel,
        standalone_projection,
        parallel_projection,
    } = run_parallel_rms_device(&input, &weight, &gemv_weights, vector_shape, gemv_shape)?;
    for (index, (actual, expected)) in parallel.iter().zip(&standalone).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "RMSNorm value {index}"
        );
    }
    for (index, (actual, expected)) in parallel_projection
        .iter()
        .zip(&standalone_projection)
        .enumerate()
    {
        assert_eq!(actual.to_bits(), expected.to_bits(), "GEMV row {index}");
    }
    eprintln!("parallel RMSNorm q8_1 and downstream GEMV match bitwise");
    Ok(())
}

fn run_parallel_rms_device(
    input: &[f32],
    weight: &[f32],
    gemv_weights: &[u8],
    vector_shape: VectorShape,
    gemv_shape: QuantizedMatrixShape,
) -> TestResult<ParallelRmsResults> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let ParallelRmsBuffers {
        input: d_input,
        weight: d_weight,
        gemv_weights: d_gemv_weights,
        standalone: mut d_standalone,
        parallel: mut d_parallel,
        standalone_projection: mut d_standalone_projection,
        parallel_projection: mut d_parallel_projection,
        mut standalone_scratch,
        mut parallel_scratch,
    } = parallel_rms_buffers(
        &context,
        input,
        weight,
        gemv_weights,
        vector_shape,
        gemv_shape,
    )?;
    run_parallel_rms_once(
        &stream,
        &d_input,
        &d_weight,
        &d_gemv_weights,
        &mut d_standalone,
        &mut d_parallel,
        &mut d_standalone_projection,
        &mut d_parallel_projection,
        &mut standalone_scratch,
        &mut parallel_scratch,
        vector_shape,
        gemv_shape,
    )?;
    stream.synchronize()?;
    let mut standalone = vec![0.0; vector_shape.elements()];
    let mut parallel = vec![0.0; vector_shape.elements()];
    d_standalone.copy_to(&mut standalone)?;
    d_parallel.copy_to(&mut parallel)?;
    let mut standalone_projection = vec![0.0; gemv_shape.rows()];
    let mut parallel_projection = vec![0.0; gemv_shape.rows()];
    d_standalone_projection.copy_to(&mut standalone_projection)?;
    d_parallel_projection.copy_to(&mut parallel_projection)?;
    Ok(ParallelRmsResults {
        standalone,
        parallel,
        standalone_projection,
        parallel_projection,
    })
}

#[allow(clippy::too_many_arguments)]
fn parallel_rms_buffers(
    context: &Context,
    input: &[f32],
    weight: &[f32],
    gemv_weights: &[u8],
    vector_shape: VectorShape,
    gemv_shape: QuantizedMatrixShape,
) -> TestResult<ParallelRmsBuffers> {
    let (d_input, d_weight, d_gemv_weights) =
        parallel_rms_input_buffers(context, input, weight, gemv_weights, gemv_shape)?;
    let ParallelRmsOutputBuffers {
        standalone: d_standalone,
        parallel: d_parallel,
        standalone_projection: d_standalone_projection,
        parallel_projection: d_parallel_projection,
        standalone_scratch,
        parallel_scratch,
    } = parallel_rms_output_buffers(context, vector_shape, gemv_shape)?;
    Ok(ParallelRmsBuffers {
        input: d_input,
        weight: d_weight,
        gemv_weights: d_gemv_weights,
        standalone: d_standalone,
        parallel: d_parallel,
        standalone_projection: d_standalone_projection,
        parallel_projection: d_parallel_projection,
        standalone_scratch,
        parallel_scratch,
    })
}

fn parallel_rms_input_buffers(
    context: &Context,
    input: &[f32],
    weight: &[f32],
    gemv_weights: &[u8],
    gemv_shape: QuantizedMatrixShape,
) -> TestResult<(DeviceBuffer<f32>, DeviceBuffer<f32>, DeviceBuffer<u8>)> {
    let gemv_device_weights = device_weight_bytes(gemv_weights, gemv_shape)?;
    Ok((
        context.copy_to_device(input)?,
        context.copy_to_device(weight)?,
        context.copy_to_device(&gemv_device_weights)?,
    ))
}

fn parallel_rms_output_buffers(
    context: &Context,
    vector_shape: VectorShape,
    gemv_shape: QuantizedMatrixShape,
) -> TestResult<ParallelRmsOutputBuffers> {
    Ok(ParallelRmsOutputBuffers {
        standalone: context.alloc(vector_shape.elements())?,
        parallel: context.alloc(vector_shape.elements())?,
        standalone_projection: context.alloc(gemv_shape.rows())?,
        parallel_projection: context.alloc(gemv_shape.rows())?,
        standalone_scratch: GemvScratch::new(context, gemv_shape)?,
        parallel_scratch: GemvScratch::new(context, gemv_shape)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_parallel_rms_once(
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    gemv_weights: &DeviceBuffer<u8>,
    standalone: &mut DeviceBuffer<f32>,
    parallel: &mut DeviceBuffer<f32>,
    standalone_projection: &mut DeviceBuffer<f32>,
    parallel_projection: &mut DeviceBuffer<f32>,
    standalone_scratch: &mut GemvScratch,
    parallel_scratch: &mut GemvScratch,
    vector_shape: VectorShape,
    gemv_shape: QuantizedMatrixShape,
) -> TestResult {
    rms_norm(stream, input, weight, standalone, vector_shape, 1e-6)?;
    rms_norm_q8_parallel(
        stream,
        input,
        weight,
        parallel,
        parallel_scratch,
        vector_shape,
        1e-6,
    )?;
    gemv_q4_k(
        stream,
        gemv_weights,
        standalone,
        standalone_projection,
        standalone_scratch,
        gemv_shape,
    )?;
    gemv_q4_k_residual(
        stream,
        gemv_weights,
        parallel,
        None,
        parallel_projection,
        parallel_scratch,
        gemv_shape,
        true,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn rope_neox_matches_f64_oracle() -> TestResult {
    // CUDA and the oracle form the phase in f64, then CUDA rounds the rotated
    // value to f32. The bound is 5e-7 absolute plus 5e-7 relative.
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = RopeShape::new(4, 6, 128)?;
    let input = random_f32(shape.elements(), 0x726f_7065_5f6e_656f, -1.0..1.0);
    let positions = [0_u32, 17, 2_048, 8_000];
    let theta = 1_000_000.0_f32;
    let mut values = context.copy_to_device(&input)?;
    let positions = context.copy_to_device(&positions)?;
    rope_neox(&stream, &mut values, &positions, shape, theta)?;
    stream.synchronize()?;
    let mut actual = vec![0.0; shape.elements()];
    values.copy_to(&mut actual)?;
    let expected = rope_oracle(&input, &[0, 17, 2_048, 8_000], shape, theta);
    let errors = assert_close("RoPE", &actual, &expected, 5e-7, 5e-7);
    eprintln!("RoPE {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn rope_adjacent_with_factors_matches_f64_oracle() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = RopeShape::new(4, 6, 64)?;
    let input = random_f32(shape.elements(), 0x6c6c_616d_615f_726f, -1.0..1.0);
    let factors = (0..shape.head_dim() / 2)
        .map(|pair| 1.0 + pair as f32 / 16.0)
        .collect::<Vec<_>>();
    let theta = 500_000.0_f32;
    let position = 17;
    let scratch = RopeScratch::new(&context, shape.head_dim(), theta, Some(&factors), true)?;
    let mut values = context.copy_to_device(&input)?;
    rope_at_frequencies(&stream, &mut values, position, shape, &scratch)?;
    stream.synchronize()?;
    let mut actual = vec![0.0; shape.elements()];
    values.copy_to(&mut actual)?;
    let expected = rope_adjacent_oracle(&input, position, shape, theta, &factors);
    let errors = assert_close("adjacent RoPE", &actual, &expected, 5e-7, 5e-7);
    eprintln!("adjacent RoPE {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn fused_rms_norm_rope_matches_composed_oracle() -> TestResult {
    let shape = VectorShape::new(6, 128)?;
    let rope_shape = RopeShape::new(1, shape.rows(), shape.columns())?;
    let input = random_f32(shape.elements(), 0x6675_7365_645f_726d, -1.0..1.0);
    let weight = random_f32(shape.columns(), 0x6675_7365_645f_7774, 0.5..1.5);
    let position = 2_048;
    let epsilon = 1e-6_f32;
    let theta = 1_000_000.0_f32;
    let actual = run_fused_rms_norm_rope(&input, &weight, shape, position, epsilon, theta)?;
    let normalized = rms_oracle(&input, None, &weight, shape, epsilon)
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let expected = rope_oracle(&normalized, &[position as u32], rope_shape, theta);
    let errors = assert_close("fused RMSNorm RoPE", &actual, &expected, 5e-6, 2e-5);
    eprintln!("fused RMSNorm RoPE {errors}");
    Ok(())
}

fn run_fused_rms_norm_rope(
    input: &[f32],
    weight: &[f32],
    shape: VectorShape,
    position: usize,
    epsilon: f32,
    theta: f32,
) -> TestResult<Vec<f32>> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let d_input = context.copy_to_device(input)?;
    let d_weight = context.copy_to_device(weight)?;
    let mut d_output = context.alloc(shape.elements())?;
    rms_norm_rope(
        &stream,
        &d_input,
        &d_weight,
        &mut d_output,
        shape,
        position,
        epsilon,
        theta,
    )?;
    stream.synchronize()?;
    copy_f32_buffer(&d_output, shape.elements())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn qk_norm_rope_matches_two_composed_oracles() -> TestResult {
    let query_shape = VectorShape::new(32, 128)?;
    let key_shape = VectorShape::new(8, 128)?;
    let query = random_f32(query_shape.elements(), 0x716b_5f71_7565_7279, -1.0..1.0);
    let key = random_f32(key_shape.elements(), 0x716b_5f6b_6579_5f5f, -1.0..1.0);
    let value = random_f32(key_shape.elements(), 0x716b_5f76_616c_7565, -1.0..1.0);
    let query_weight = random_f32(128, 0x716b_5f71_7765_6967, 0.5..1.5);
    let key_weight = random_f32(128, 0x716b_5f6b_7765_6967, 0.5..1.5);
    let position = 2_048;
    let epsilon = 1e-6_f32;
    let theta = 1_000_000.0_f32;
    let attention_shape = leone_cuda::AttentionShape::new(32, 8, 128, 4_096)?;
    let outputs = run_qk_norm_rope_device(
        &query,
        &key,
        &value,
        &query_weight,
        &key_weight,
        query_shape,
        key_shape,
        attention_shape,
        position,
        epsilon,
        theta,
    )?;
    let QkNormResults {
        query_actual,
        key_actual,
        fused_query,
        fused_key,
        key_cache,
        value_cache,
        fused_key_cache,
        fused_value_cache,
    } = outputs;
    assert_eq!(query_actual, fused_query);
    assert_eq!(key_actual, fused_key);
    assert_eq!(key_cache, fused_key_cache);
    assert_eq!(value_cache, fused_value_cache);
    let query_expected =
        rms_rope_oracle(&query, &query_weight, query_shape, position, epsilon, theta)?;
    let key_expected = rms_rope_oracle(&key, &key_weight, key_shape, position, epsilon, theta)?;
    let query_errors = assert_close(
        "batched query RMSNorm RoPE",
        &query_actual,
        &query_expected,
        5e-6,
        2e-5,
    );
    let key_errors = assert_close(
        "batched key RMSNorm RoPE",
        &key_actual,
        &key_expected,
        5e-6,
        2e-5,
    );
    eprintln!("batched query {query_errors}; batched key {key_errors}");
    Ok(())
}

struct QkNormBuffers {
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    query_weight: DeviceBuffer<f32>,
    key_weight: DeviceBuffer<f32>,
    query_output: DeviceBuffer<f32>,
    key_output: DeviceBuffer<f32>,
    fused_query_output: DeviceBuffer<f32>,
    fused_key_output: DeviceBuffer<f32>,
    key_cache: DeviceBuffer<u16>,
    value_cache: DeviceBuffer<u16>,
    fused_key_cache: DeviceBuffer<u16>,
    fused_value_cache: DeviceBuffer<u16>,
}

#[allow(clippy::too_many_arguments)]
fn run_qk_norm_rope_device(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_weight: &[f32],
    key_weight: &[f32],
    query_shape: VectorShape,
    key_shape: VectorShape,
    attention_shape: leone_cuda::AttentionShape,
    position: usize,
    epsilon: f32,
    theta: f32,
) -> TestResult<QkNormResults> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let mut buffers = qk_norm_buffers(
        &context,
        query,
        key,
        value,
        query_weight,
        key_weight,
        query_shape,
        key_shape,
        attention_shape,
    )?;
    let mut rope_scratch = RopeScratch::new(&context, query_shape.columns(), theta, None, false)?;
    run_qk_norm_rope_launch(
        &stream,
        &mut buffers,
        &mut rope_scratch,
        query_shape,
        key_shape,
        attention_shape,
        position,
        epsilon,
    )?;
    stream.synchronize()?;
    copy_qk_norm_outputs(&buffers, query_shape, key_shape, attention_shape)
}

#[allow(clippy::too_many_arguments)]
fn qk_norm_buffers(
    context: &Context,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_weight: &[f32],
    key_weight: &[f32],
    query_shape: VectorShape,
    key_shape: VectorShape,
    attention_shape: leone_cuda::AttentionShape,
) -> TestResult<QkNormBuffers> {
    let QkNormInputBuffers {
        query,
        key,
        value,
        query_weight,
        key_weight,
    } = qk_norm_input_buffers(context, query, key, value, query_weight, key_weight)?;
    let QkNormOutputBuffers {
        query_output,
        key_output,
        fused_query_output,
        fused_key_output,
        key_cache,
        value_cache,
        fused_key_cache,
        fused_value_cache,
    } = qk_norm_output_buffers(context, query_shape, key_shape, attention_shape)?;
    Ok(QkNormBuffers {
        query,
        key,
        value,
        query_weight,
        key_weight,
        query_output,
        key_output,
        fused_query_output,
        fused_key_output,
        key_cache,
        value_cache,
        fused_key_cache,
        fused_value_cache,
    })
}

fn qk_norm_input_buffers(
    context: &Context,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_weight: &[f32],
    key_weight: &[f32],
) -> TestResult<QkNormInputBuffers> {
    Ok(QkNormInputBuffers {
        query: context.copy_to_device(query)?,
        key: context.copy_to_device(key)?,
        value: context.copy_to_device(value)?,
        query_weight: context.copy_to_device(query_weight)?,
        key_weight: context.copy_to_device(key_weight)?,
    })
}

fn qk_norm_output_buffers(
    context: &Context,
    query_shape: VectorShape,
    key_shape: VectorShape,
    attention_shape: leone_cuda::AttentionShape,
) -> TestResult<QkNormOutputBuffers> {
    let empty_cache = vec![0_u16; attention_shape.cache_elements()];
    Ok(QkNormOutputBuffers {
        query_output: context.alloc(query_shape.elements())?,
        key_output: context.alloc(key_shape.elements())?,
        fused_query_output: context.alloc(query_shape.elements())?,
        fused_key_output: context.alloc(key_shape.elements())?,
        key_cache: context.copy_to_device(&empty_cache)?,
        value_cache: context.copy_to_device(&empty_cache)?,
        fused_key_cache: context.copy_to_device(&empty_cache)?,
        fused_value_cache: context.copy_to_device(&empty_cache)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_qk_norm_rope_launch(
    stream: &Stream,
    buffers: &mut QkNormBuffers,
    rope_scratch: &mut RopeScratch,
    query_shape: VectorShape,
    key_shape: VectorShape,
    attention_shape: leone_cuda::AttentionShape,
    position: usize,
    epsilon: f32,
) -> TestResult {
    rope_scratch.prepare(stream, position)?;
    qk_norm_rope(
        stream,
        &buffers.query,
        &buffers.query_weight,
        &mut buffers.query_output,
        query_shape,
        &buffers.key,
        &buffers.key_weight,
        &mut buffers.key_output,
        key_shape,
        rope_scratch,
        epsilon,
    )?;
    kv_append_f16(
        stream,
        &buffers.key_output,
        &buffers.value,
        &mut buffers.key_cache,
        &mut buffers.value_cache,
        attention_shape,
        position,
    )?;
    qk_norm_rope_kv_append_f16(
        stream,
        &buffers.query,
        &buffers.query_weight,
        &mut buffers.fused_query_output,
        query_shape,
        &buffers.key,
        &buffers.key_weight,
        &mut buffers.fused_key_output,
        key_shape,
        &buffers.value,
        &mut buffers.fused_key_cache,
        &mut buffers.fused_value_cache,
        attention_shape,
        position,
        rope_scratch,
        epsilon,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn copy_qk_norm_outputs(
    buffers: &QkNormBuffers,
    query_shape: VectorShape,
    key_shape: VectorShape,
    attention_shape: leone_cuda::AttentionShape,
) -> TestResult<QkNormResults> {
    let query_actual = copy_f32_buffer(&buffers.query_output, query_shape.elements())?;
    let key_actual = copy_f32_buffer(&buffers.key_output, key_shape.elements())?;
    let fused_query = copy_f32_buffer(&buffers.fused_query_output, query_shape.elements())?;
    let fused_key = copy_f32_buffer(&buffers.fused_key_output, key_shape.elements())?;
    let mut key_cache = vec![0_u16; attention_shape.cache_elements()];
    let mut value_cache = vec![0_u16; attention_shape.cache_elements()];
    let mut fused_key_cache = vec![0_u16; attention_shape.cache_elements()];
    let mut fused_value_cache = vec![0_u16; attention_shape.cache_elements()];
    buffers.key_cache.copy_to(&mut key_cache)?;
    buffers.value_cache.copy_to(&mut value_cache)?;
    buffers.fused_key_cache.copy_to(&mut fused_key_cache)?;
    buffers.fused_value_cache.copy_to(&mut fused_value_cache)?;
    Ok(QkNormResults {
        query_actual,
        key_actual,
        fused_query,
        fused_key,
        key_cache,
        value_cache,
        fused_key_cache,
        fused_value_cache,
    })
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn swiglu_and_residual_add_match_oracles() -> TestResult {
    // CUDA and the f64 SwiGLU oracle differ only in the transcendental and
    // final rounding. The bound is 2e-6 absolute plus 3e-6 relative.
    let gate = random_f32(12_289, 0x7377_6967_6c75_6761, -8.0..8.0);
    let up = random_f32(12_289, 0x7377_6967_6c75_7570, -2.0..2.0);
    let (actual, added) = run_swiglu_add_device(&gate, &up)?;
    let expected = swiglu_expected(&gate, &up);
    let errors = assert_close("SwiGLU", &actual, &expected, 2e-6, 3e-6);
    for (index, ((left, right), result)) in gate.iter().zip(&up).zip(&added).enumerate() {
        assert_eq!(
            result.to_bits(),
            (left + right).to_bits(),
            "add index {index}"
        );
    }
    eprintln!("SwiGLU {errors}; residual add max error 0");
    Ok(())
}

fn run_swiglu_add_device(gate: &[f32], up: &[f32]) -> TestResult<(Vec<f32>, Vec<f32>)> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let SwigluAddBuffers {
        gate: d_gate,
        up: d_up,
        swiglu_output: mut d_swiglu,
        add_output: mut d_add,
    } = swiglu_add_buffers(&context, gate, up)?;
    swiglu_add_launch(&stream, &d_gate, &d_up, &mut d_swiglu, &mut d_add)?;
    stream.synchronize()?;
    let mut actual = vec![0.0; gate.len()];
    let mut added = vec![0.0; gate.len()];
    d_swiglu.copy_to(&mut actual)?;
    d_add.copy_to(&mut added)?;
    Ok((actual, added))
}

fn swiglu_add_buffers(context: &Context, gate: &[f32], up: &[f32]) -> TestResult<SwigluAddBuffers> {
    Ok(SwigluAddBuffers {
        gate: context.copy_to_device(gate)?,
        up: context.copy_to_device(up)?,
        swiglu_output: context.alloc(gate.len())?,
        add_output: context.alloc(gate.len())?,
    })
}

fn swiglu_add_launch(
    stream: &Stream,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    swiglu_output: &mut DeviceBuffer<f32>,
    add_output: &mut DeviceBuffer<f32>,
) -> TestResult {
    swiglu(stream, gate, up, swiglu_output)?;
    residual_add(stream, gate, up, add_output)?;
    Ok(())
}

fn swiglu_expected(gate: &[f32], up: &[f32]) -> Vec<f64> {
    gate.iter()
        .zip(up)
        .map(|(gate, up)| {
            let value = f64::from(*gate);
            (value / (1.0 + (-value).exp()) * f64::from(*up)) as f32 as f64
        })
        .collect()
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn gemv_epilogues_match_composed_kernels_bitwise() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    for (format, seed) in [
        (QuantFormat::Q4K, 0x6570_696c_6f67_7134),
        (QuantFormat::Q6K, 0x6570_696c_6f67_7136),
    ] {
        gemv_epilogue_case(&context, &stream, format, seed)?;
    }
    eprintln!("Q4_K and Q6_K residual epilogues match composed kernels bitwise");
    Ok(())
}

fn gemv_epilogue_case(
    context: &Context,
    stream: &Stream,
    format: QuantFormat,
    seed: u64,
) -> TestResult {
    let shape = QuantizedMatrixShape::new(67, 1_024, format)?;
    let weights = quantized_bytes(shape, seed);
    let input = random_f32(shape.columns(), seed ^ 0x1111, -0.25..0.25);
    let residual = random_f32(shape.rows(), seed ^ 0x2222, -0.5..0.5);
    let GemvEpilogueBuffers {
        weights: d_weights,
        input: d_input,
        residual: d_residual,
        projection: mut d_projection,
        composed: mut d_composed,
        fused: mut d_fused,
        mut composed_scratch,
        mut fused_scratch,
    } = gemv_epilogue_buffers(context, &weights, &input, &residual, shape)?;
    run_gemv_epilogue_launch(
        stream,
        format,
        &d_weights,
        &d_input,
        &d_residual,
        &mut d_projection,
        &mut d_composed,
        &mut d_fused,
        &mut composed_scratch,
        &mut fused_scratch,
        shape,
    )?;
    compare_gemv_epilogue(stream, &d_composed, &d_fused, shape, format)
}

#[allow(clippy::too_many_arguments)]
fn gemv_epilogue_buffers(
    context: &Context,
    weights: &[u8],
    input: &[f32],
    residual: &[f32],
    shape: QuantizedMatrixShape,
) -> TestResult<GemvEpilogueBuffers> {
    let device_weights = device_weight_bytes(weights, shape)?;
    Ok(GemvEpilogueBuffers {
        weights: context.copy_to_device(&device_weights)?,
        input: context.copy_to_device(input)?,
        residual: context.copy_to_device(residual)?,
        projection: context.alloc(shape.rows())?,
        composed: context.alloc(shape.rows())?,
        fused: context.alloc(shape.rows())?,
        composed_scratch: GemvScratch::new(context, shape)?,
        fused_scratch: GemvScratch::new(context, shape)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_gemv_epilogue_launch(
    stream: &Stream,
    format: QuantFormat,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    projection: &mut DeviceBuffer<f32>,
    composed: &mut DeviceBuffer<f32>,
    fused: &mut DeviceBuffer<f32>,
    composed_scratch: &mut GemvScratch,
    fused_scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
) -> TestResult {
    match format {
        QuantFormat::Q4K => {
            gemv_q4_k(stream, weights, input, projection, composed_scratch, shape)?;
            gemv_q4_k_residual(
                stream,
                weights,
                input,
                Some(residual),
                fused,
                fused_scratch,
                shape,
                false,
            )?;
        }
        QuantFormat::Q6K => {
            gemv_q6_k(stream, weights, input, projection, composed_scratch, shape)?;
            gemv_q6_k_residual(
                stream,
                weights,
                input,
                Some(residual),
                fused,
                fused_scratch,
                shape,
                false,
            )?;
        }
    }
    residual_add(stream, projection, residual, composed)?;
    Ok(())
}

fn compare_gemv_epilogue(
    stream: &Stream,
    composed: &DeviceBuffer<f32>,
    fused: &DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    format: QuantFormat,
) -> TestResult {
    stream.synchronize()?;
    let mut composed_values = vec![0.0; shape.rows()];
    let mut fused_values = vec![0.0; shape.rows()];
    composed.copy_to(&mut composed_values)?;
    fused.copy_to(&mut fused_values)?;
    for (index, (actual, expected)) in fused_values.iter().zip(&composed_values).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{format:?} row {index}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn swiglu_q8_epilogue_matches_standalone_path_bitwise() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = QuantizedMatrixShape::new(67, 1_024, QuantFormat::Q4K)?;
    let weights = quantized_bytes(shape, 0x7377_7138_5f77_6768);
    let gate = random_f32(shape.columns(), 0x7377_7138_5f67_6174, -8.0..8.0);
    let up = random_f32(shape.columns(), 0x7377_7138_5f75_705f, -2.0..2.0);
    swiglu_q8_case(&context, &stream, &weights, &gate, &up, shape)?;
    eprintln!("SwiGLU q8_1 epilogue and downstream GEMV match bitwise");
    Ok(())
}

fn swiglu_q8_case(
    context: &Context,
    stream: &Stream,
    weights: &[u8],
    gate: &[f32],
    up: &[f32],
    shape: QuantizedMatrixShape,
) -> TestResult {
    let SwigluQ8Buffers {
        weights: d_weights,
        gate: d_gate,
        up: d_up,
        standalone: mut d_standalone,
        fused: mut d_fused,
        standalone_projection: mut d_standalone_projection,
        fused_projection: mut d_fused_projection,
        mut standalone_scratch,
        mut fused_scratch,
    } = swiglu_q8_buffers(context, weights, gate, up, shape)?;
    run_swiglu_q8_launch(
        stream,
        &d_weights,
        &d_gate,
        &d_up,
        &mut d_standalone,
        &mut d_fused,
        &mut d_standalone_projection,
        &mut d_fused_projection,
        &mut standalone_scratch,
        &mut fused_scratch,
        shape,
    )?;
    compare_swiglu_q8(
        stream,
        &d_standalone,
        &d_fused,
        &d_standalone_projection,
        &d_fused_projection,
        shape,
    )
}

#[allow(clippy::too_many_arguments)]
fn swiglu_q8_buffers(
    context: &Context,
    weights: &[u8],
    gate: &[f32],
    up: &[f32],
    shape: QuantizedMatrixShape,
) -> TestResult<SwigluQ8Buffers> {
    let (d_weights, d_gate, d_up) = swiglu_q8_inputs(context, weights, gate, up, shape)?;
    let SwigluQ8Outputs {
        standalone: d_standalone,
        fused: d_fused,
        standalone_projection: d_standalone_projection,
        fused_projection: d_fused_projection,
        standalone_scratch,
        fused_scratch,
    } = swiglu_q8_outputs(context, shape)?;
    Ok(SwigluQ8Buffers {
        weights: d_weights,
        gate: d_gate,
        up: d_up,
        standalone: d_standalone,
        fused: d_fused,
        standalone_projection: d_standalone_projection,
        fused_projection: d_fused_projection,
        standalone_scratch,
        fused_scratch,
    })
}

fn swiglu_q8_inputs(
    context: &Context,
    weights: &[u8],
    gate: &[f32],
    up: &[f32],
    shape: QuantizedMatrixShape,
) -> TestResult<(DeviceBuffer<u8>, DeviceBuffer<f32>, DeviceBuffer<f32>)> {
    let device_weights = device_weight_bytes(weights, shape)?;
    Ok((
        context.copy_to_device(&device_weights)?,
        context.copy_to_device(gate)?,
        context.copy_to_device(up)?,
    ))
}

fn swiglu_q8_outputs(
    context: &Context,
    shape: QuantizedMatrixShape,
) -> TestResult<SwigluQ8Outputs> {
    Ok(SwigluQ8Outputs {
        standalone: context.alloc(shape.columns())?,
        fused: context.alloc(shape.columns())?,
        standalone_projection: context.alloc(shape.rows())?,
        fused_projection: context.alloc(shape.rows())?,
        standalone_scratch: GemvScratch::new(context, shape)?,
        fused_scratch: GemvScratch::new(context, shape)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_swiglu_q8_launch(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    standalone: &mut DeviceBuffer<f32>,
    fused: &mut DeviceBuffer<f32>,
    standalone_projection: &mut DeviceBuffer<f32>,
    fused_projection: &mut DeviceBuffer<f32>,
    standalone_scratch: &mut GemvScratch,
    fused_scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
) -> TestResult {
    swiglu(stream, gate, up, standalone)?;
    swiglu_q8(stream, gate, up, fused, fused_scratch)?;
    gemv_q4_k(
        stream,
        weights,
        standalone,
        standalone_projection,
        standalone_scratch,
        shape,
    )?;
    gemv_q4_k_residual(
        stream,
        weights,
        fused,
        None,
        fused_projection,
        fused_scratch,
        shape,
        true,
    )?;
    Ok(())
}

fn compare_swiglu_q8(
    stream: &Stream,
    standalone: &DeviceBuffer<f32>,
    fused: &DeviceBuffer<f32>,
    standalone_projection: &DeviceBuffer<f32>,
    fused_projection: &DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
) -> TestResult {
    stream.synchronize()?;
    let mut standalone_values = vec![0.0; shape.columns()];
    let mut fused_values = vec![0.0; shape.columns()];
    standalone.copy_to(&mut standalone_values)?;
    fused.copy_to(&mut fused_values)?;
    let mut standalone_rows = vec![0.0; shape.rows()];
    let mut fused_rows = vec![0.0; shape.rows()];
    standalone_projection.copy_to(&mut standalone_rows)?;
    fused_projection.copy_to(&mut fused_rows)?;
    compare_swiglu_vectors(
        &standalone_values,
        &fused_values,
        &standalone_rows,
        &fused_rows,
    )
}

fn compare_swiglu_vectors(
    standalone: &[f32],
    fused: &[f32],
    standalone_projection: &[f32],
    fused_projection: &[f32],
) -> TestResult {
    for (index, (actual, expected)) in fused.iter().zip(standalone).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "SwiGLU value {index}");
    }
    for (index, (actual, expected)) in fused_projection
        .iter()
        .zip(standalone_projection)
        .enumerate()
    {
        assert_eq!(actual.to_bits(), expected.to_bits(), "GEMV row {index}");
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn quantized_embedding_gathers_match_scalar_dequant() -> TestResult {
    // Gather performs the same f32 dequantization as the scalar decoder. The
    // 2e-6 bound permits one FMA contraction in Q4_K.
    for (format, seed) in [
        (QuantFormat::Q4K, 0x656d_6265_645f_7134),
        (QuantFormat::Q6K, 0x656d_6265_645f_7136),
    ] {
        embedding_gather_case(format, seed)?;
    }
    Ok(())
}

fn embedding_gather_case(format: QuantFormat, seed: u64) -> TestResult {
    let shape = QuantizedMatrixShape::new(11, 4_096, format)?;
    let table = quantized_bytes(shape, seed);
    let row = 7;
    let actual = run_embedding_gather_device(&table, shape, row, format)?;
    let expected = embedding_gather_expected(&table, shape, format, row)?;
    let errors = assert_close("embedding gather", &actual, &expected, 2e-6, 2e-6);
    eprintln!("{format:?} embedding gather {errors}");
    Ok(())
}

fn run_embedding_gather_device(
    table: &[u8],
    shape: QuantizedMatrixShape,
    row: usize,
    format: QuantFormat,
) -> TestResult<Vec<f32>> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let (d_table, mut d_output) = embedding_gather_buffers(&context, table, shape)?;
    match format {
        QuantFormat::Q4K => embedding_gather_q4_k(&stream, &d_table, &mut d_output, shape, row)?,
        QuantFormat::Q6K => embedding_gather_q6_k(&stream, &d_table, &mut d_output, shape, row)?,
    }
    stream.synchronize()?;
    copy_f32_buffer(&d_output, shape.columns())
}

fn embedding_gather_buffers(
    context: &Context,
    table: &[u8],
    shape: QuantizedMatrixShape,
) -> TestResult<(DeviceBuffer<u8>, DeviceBuffer<f32>)> {
    let device_table = device_weight_bytes(table, shape)?;
    Ok((
        context.copy_to_device(&device_table)?,
        context.alloc(shape.columns())?,
    ))
}

fn embedding_gather_expected(
    table: &[u8],
    shape: QuantizedMatrixShape,
    format: QuantFormat,
    row: usize,
) -> TestResult<Vec<f64>> {
    let row_start = row * shape.row_bytes();
    Ok(dequant_row(
        format,
        &table[row_start..row_start + shape.row_bytes()],
        shape.columns(),
    )?
    .into_iter()
    .map(f64::from)
    .collect())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn attention_decode_matches_f64_oracle_at_split_boundaries() -> TestResult {
    // Online f32 softmax changes the addition order from the brute-force f64
    // oracle. The bound is 3e-5 absolute plus 3e-4 relative.
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = leone_cuda::AttentionShape::new(4, 2, 128, 8_000)?;
    let query = random_f32(shape.query_elements(), 0x6174_746e_5f71_7565, -0.5..0.5);
    let keys = random_f32(shape.cache_elements(), 0x6174_746e_5f6b_6579, -0.5..0.5);
    let values = random_f32(shape.cache_elements(), 0x6174_746e_5f76_616c, -0.5..0.5);
    let d_query = context.copy_to_device(&query)?;
    let d_keys = context.copy_to_device(&keys)?;
    let d_values = context.copy_to_device(&values)?;
    let mut d_output = context.alloc(shape.query_elements())?;
    let mut scratch = AttentionScratch::new(&context, shape)?;
    check_attention_decode_cases(
        &stream,
        &d_query,
        &d_keys,
        &d_values,
        &mut d_output,
        &mut scratch,
        &query,
        &keys,
        &values,
        shape,
        &[1, 17, 256, 2_048, 8_000],
        "decode attention",
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_attention_decode_cases(
    stream: &Stream,
    query_device: &DeviceBuffer<f32>,
    keys_device: &DeviceBuffer<f32>,
    values_device: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    context_lengths: &[usize],
    label: &str,
) -> TestResult {
    for &context_length in context_lengths {
        attention_decode(
            stream,
            query_device,
            keys_device,
            values_device,
            output,
            scratch,
            None,
            shape,
            context_length,
        )?;
        stream.synchronize()?;
        let actual = copy_f32_buffer(output, shape.query_elements())?;
        let expected = attention_oracle(query, keys, values, shape, context_length);
        let errors = assert_close(label, &actual, &expected, 3e-5, 3e-4);
        eprintln!("attention context {context_length}: {errors}");
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn f16_attention_covers_strided_tile_tail() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = leone_cuda::AttentionShape::new(2, 1, 128, 16_384)?;
    let query = random_f32(shape.query_elements(), 0x7461_696c_7175_6572, -0.5..0.5);
    let keys = rounded_f16_values(shape.cache_elements(), 0x7461_696c_6b65_7973);
    let values = rounded_f16_values(shape.cache_elements(), 0x7461_696c_7661_6c73);
    let mut buffers = f16_decode_buffers(&context, &query, &keys, &values)?;
    let mut scratch = AttentionScratch::new(&context, shape)?;
    let splits = context.attention_split_count(shape, true)?;
    // The graph bucket selects the 128-thread kernel even when a live split
    // fits the 192-position tile. Its tail needs a second strided iteration.
    assert!(shape.max_context().div_ceil(splits) > 192);
    for count in [127, 128, 129, 191, 192, 193, 255] {
        check_f16_tile_case(
            &stream,
            &mut buffers,
            &mut scratch,
            &query,
            &keys,
            &values,
            shape,
            count * splits,
        )?;
    }
    Ok(())
}

fn rounded_f16_values(elements: usize, seed: u64) -> Vec<f32> {
    random_f32(elements, seed, -0.5..0.5)
        .into_iter()
        .map(|value| f16::from_f32(value).to_f32())
        .collect()
}

fn f16_decode_buffers(
    context: &Context,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
) -> TestResult<PrefillAttentionDeviceBuffers> {
    let key_bits = keys
        .iter()
        .map(|value| f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let value_bits = values
        .iter()
        .map(|value| f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    Ok(PrefillAttentionDeviceBuffers {
        query: context.copy_to_device(query)?,
        keys: context.copy_to_device(&key_bits)?,
        values: context.copy_to_device(&value_bits)?,
        output: context.alloc(query.len())?,
    })
}

#[allow(clippy::too_many_arguments)]
fn check_f16_tile_case(
    stream: &Stream,
    buffers: &mut PrefillAttentionDeviceBuffers,
    scratch: &mut AttentionScratch,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    context_length: usize,
) -> TestResult {
    attention_decode_f16(
        stream,
        &buffers.query,
        &buffers.keys,
        &buffers.values,
        &mut buffers.output,
        scratch,
        None,
        shape,
        context_length,
    )?;
    stream.synchronize()?;
    let actual = copy_f32_buffer(&buffers.output, shape.query_elements())?;
    let expected = attention_oracle(query, keys, values, shape, context_length);
    assert_close("f16 attention tile tail", &actual, &expected, 3e-5, 3e-4);
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn attention_split_counts_follow_graph_buckets() -> TestResult {
    let context = Context::new(0)?;
    for max_context in [512, 1_024, 2_048, 4_096, 8_192] {
        let shape = leone_cuda::AttentionShape::new(32, 8, 128, max_context)?;
        let f32_splits = context.attention_split_count(shape, false)?;
        let f16_splits = context.attention_split_count(shape, true)?;
        eprintln!("attention bucket {max_context}: f32={f32_splits}, f16={f16_splits}");
        assert!((1..=leone_cuda::ATTENTION_SPLIT_KV_MAX).contains(&f32_splits));
        assert!((1..=leone_cuda::ATTENTION_SPLIT_KV_MAX).contains(&f16_splits));
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn f16_kv_append_and_attention_match_rounded_oracle() -> TestResult {
    let shape = leone_cuda::AttentionShape::new(4, 2, 128, 17)?;
    let query = random_f32(shape.query_elements(), 0x6631_365f_7175_6572, -0.5..0.5);
    let keys = random_f32(shape.cache_elements(), 0x6631_365f_6b65_7973, -0.5..0.5);
    let values = random_f32(shape.cache_elements(), 0x6631_365f_7661_6c73, -0.5..0.5);
    let F16KvResults {
        actual,
        prepared_output,
        standalone_projection,
        prepared_projection,
    } = run_f16_kv_device(&query, &keys, &values, shape)?;
    let rounded_keys = keys
        .iter()
        .map(|value| f16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let rounded_values = values
        .iter()
        .map(|value| f16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let expected = attention_oracle(
        &query,
        &rounded_keys,
        &rounded_values,
        shape,
        shape.max_context(),
    );
    let errors = assert_close("f16 KV attention", &actual, &expected, 3e-5, 3e-4);
    for (index, (prepared, standalone)) in prepared_output.iter().zip(&actual).enumerate() {
        assert_eq!(
            prepared.to_bits(),
            standalone.to_bits(),
            "prepared attention value {index}"
        );
    }
    for (index, (prepared, standalone)) in prepared_projection
        .iter()
        .zip(&standalone_projection)
        .enumerate()
    {
        assert_eq!(
            prepared.to_bits(),
            standalone.to_bits(),
            "prepared attention GEMV row {index}"
        );
    }
    eprintln!("f16 KV attention {errors}; q8_1 epilogue matches bitwise");
    Ok(())
}

fn run_f16_kv_device(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
) -> TestResult<F16KvResults> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let (d_query, mut d_keys, mut d_values) = f16_kv_buffers(&context, query, shape)?;
    append_f16_kv_positions(
        &context,
        &stream,
        keys,
        values,
        &mut d_keys,
        &mut d_values,
        shape,
    )?;
    stream.synchronize()?;
    let (mut d_output, mut attention_scratch, actual) =
        run_f16_initial_attention(&context, &stream, &d_query, &d_keys, &d_values, shape)?;
    let (prepared_output, standalone_projection, prepared_projection) =
        run_f16_downstream_attention(
            &context,
            &stream,
            &d_query,
            &d_keys,
            &d_values,
            &mut d_output,
            &mut attention_scratch,
            shape,
        )?;
    Ok(F16KvResults {
        actual,
        prepared_output,
        standalone_projection,
        prepared_projection,
    })
}

fn run_f16_initial_attention(
    context: &Context,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    keys: &DeviceBuffer<u16>,
    values: &DeviceBuffer<u16>,
    shape: leone_cuda::AttentionShape,
) -> TestResult<(DeviceBuffer<f32>, AttentionScratch, Vec<f32>)> {
    let mut output = context.alloc(shape.query_elements())?;
    let mut scratch = AttentionScratch::new(context, shape)?;
    run_f16_attention_into(
        stream,
        query,
        keys,
        values,
        &mut output,
        &mut scratch,
        None,
        shape,
    )?;
    stream.synchronize()?;
    let actual = copy_f32_buffer(&output, shape.query_elements())?;
    Ok((output, scratch, actual))
}

#[allow(clippy::too_many_arguments)]
fn run_f16_downstream_attention(
    context: &Context,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    keys: &DeviceBuffer<u16>,
    values: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    attention_scratch: &mut AttentionScratch,
    shape: leone_cuda::AttentionShape,
) -> TestResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let downstream_shape = QuantizedMatrixShape::new(37, shape.query_elements(), QuantFormat::Q4K)?;
    let weights = quantized_bytes(downstream_shape, 0x6174_746e_5f71_385f);
    let F16DownstreamBuffers {
        weights: d_weights,
        prepared_output: mut d_prepared_output,
        standalone_projection: mut d_standalone_projection,
        prepared_projection: mut d_prepared_projection,
        mut prepared_scratch,
        mut standalone_scratch,
    } = f16_downstream_buffers(context, &weights, downstream_shape, shape.query_elements())?;
    run_f16_attention_into(
        stream,
        query,
        keys,
        values,
        &mut d_prepared_output,
        attention_scratch,
        Some(&mut prepared_scratch),
        shape,
    )?;
    run_f16_downstream_projections(
        stream,
        &d_weights,
        output,
        &mut d_standalone_projection,
        &mut standalone_scratch,
        &d_prepared_output,
        &mut d_prepared_projection,
        &mut prepared_scratch,
        downstream_shape,
    )?;
    stream.synchronize()?;
    let prepared_output = copy_f32_buffer(&d_prepared_output, shape.query_elements())?;
    let standalone_projection = copy_f32_buffer(&d_standalone_projection, downstream_shape.rows())?;
    let prepared_projection = copy_f32_buffer(&d_prepared_projection, downstream_shape.rows())?;
    Ok((prepared_output, standalone_projection, prepared_projection))
}

fn f16_kv_buffers(
    context: &Context,
    query: &[f32],
    shape: leone_cuda::AttentionShape,
) -> TestResult<(DeviceBuffer<f32>, DeviceBuffer<u16>, DeviceBuffer<u16>)> {
    Ok((
        context.copy_to_device(query)?,
        context.alloc(shape.cache_elements())?,
        context.alloc(shape.cache_elements())?,
    ))
}

fn append_f16_kv_positions(
    context: &Context,
    stream: &Stream,
    keys: &[f32],
    values: &[f32],
    d_keys: &mut DeviceBuffer<u16>,
    d_values: &mut DeviceBuffer<u16>,
    shape: leone_cuda::AttentionShape,
) -> TestResult {
    for position in 0..shape.max_context() {
        let (projected_key, projected_value) =
            projected_kv_position(keys, values, shape, position)?;
        let d_key = context.copy_to_device(&projected_key)?;
        let d_value = context.copy_to_device(&projected_value)?;
        kv_append_f16(stream, &d_key, &d_value, d_keys, d_values, shape, position)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_f16_attention_into(
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    keys: &DeviceBuffer<u16>,
    values: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut AttentionScratch,
    prepared: Option<&mut GemvScratch>,
    shape: leone_cuda::AttentionShape,
) -> TestResult {
    attention_decode_f16(
        stream,
        query,
        keys,
        values,
        output,
        scratch,
        prepared,
        shape,
        shape.max_context(),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn f16_downstream_buffers(
    context: &Context,
    weights: &[u8],
    shape: QuantizedMatrixShape,
    query_elements: usize,
) -> TestResult<F16DownstreamBuffers> {
    let device_weights = device_weight_bytes(weights, shape)?;
    Ok(F16DownstreamBuffers {
        weights: context.copy_to_device(&device_weights)?,
        prepared_output: context.alloc(query_elements)?,
        standalone_projection: context.alloc(shape.rows())?,
        prepared_projection: context.alloc(shape.rows())?,
        prepared_scratch: GemvScratch::new(context, shape)?,
        standalone_scratch: GemvScratch::new(context, shape)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_f16_downstream_projections(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    standalone_input: &DeviceBuffer<f32>,
    standalone_output: &mut DeviceBuffer<f32>,
    standalone_scratch: &mut GemvScratch,
    prepared_input: &DeviceBuffer<f32>,
    prepared_output: &mut DeviceBuffer<f32>,
    prepared_scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
) -> TestResult {
    gemv_q4_k(
        stream,
        weights,
        standalone_input,
        standalone_output,
        standalone_scratch,
        shape,
    )?;
    gemv_q4_k_residual(
        stream,
        weights,
        prepared_input,
        None,
        prepared_output,
        prepared_scratch,
        shape,
        true,
    )?;
    Ok(())
}

fn copy_f32_buffer(buffer: &DeviceBuffer<f32>, len: usize) -> TestResult<Vec<f32>> {
    let mut values = vec![0.0; len];
    buffer.copy_to(&mut values)?;
    Ok(values)
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn q8_kv_append_and_attention_match_scalar_oracle() -> TestResult {
    let shape = leone_cuda::AttentionShape::new(4, 2, 128, 17)?;
    let query = random_f32(shape.query_elements(), 0x7138_5f71_7565_7279, -0.5..0.5);
    let keys = random_f32(shape.cache_elements(), 0x7138_5f6b_6579_7300, -0.5..0.5);
    let values = random_f32(shape.cache_elements(), 0x7138_5f76_616c_7565, -0.5..0.5);
    let cache_bytes = shape.cache_elements() / 32 * 34;
    let expected_keys = q8_kv_encode(&keys, shape);
    let expected_values = q8_kv_encode(&values, shape);
    let (actual_keys, actual_values, actual) =
        run_q8_kv_device(&query, &keys, &values, shape, cache_bytes)?;
    assert_eq!(actual_keys, expected_keys, "q8 key cache bytes");
    assert_eq!(actual_values, expected_values, "q8 value cache bytes");
    let decoded_keys = q8_kv_decode(&expected_keys, shape);
    let decoded_values = q8_kv_decode(&expected_values, shape);
    let expected = attention_oracle(
        &query,
        &decoded_keys,
        &decoded_values,
        shape,
        shape.max_context(),
    );
    let errors = assert_close("q8 KV attention", &actual, &expected, 3e-5, 3e-4);
    eprintln!("q8 KV attention {errors}");
    Ok(())
}

fn run_q8_kv_device(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    cache_bytes: usize,
) -> TestResult<(Vec<u8>, Vec<u8>, Vec<f32>)> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let (d_query, mut d_keys, mut d_values) = q8_kv_device_buffers(&context, query, cache_bytes)?;
    append_q8_kv_positions(
        &context,
        &stream,
        keys,
        values,
        &mut d_keys,
        &mut d_values,
        shape,
    )?;
    stream.synchronize()?;
    let mut actual_keys = vec![0_u8; cache_bytes];
    let mut actual_values = vec![0_u8; cache_bytes];
    d_keys.copy_to(&mut actual_keys)?;
    d_values.copy_to(&mut actual_values)?;
    let actual = run_q8_attention(&context, &stream, &d_query, &d_keys, &d_values, shape)?;
    Ok((actual_keys, actual_values, actual))
}

fn q8_kv_device_buffers(
    context: &Context,
    query: &[f32],
    cache_bytes: usize,
) -> TestResult<(DeviceBuffer<f32>, DeviceBuffer<u8>, DeviceBuffer<u8>)> {
    Ok((
        context.copy_to_device(query)?,
        context.alloc::<u8>(cache_bytes)?,
        context.alloc::<u8>(cache_bytes)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn append_q8_kv_positions(
    context: &Context,
    stream: &Stream,
    keys: &[f32],
    values: &[f32],
    d_keys: &mut DeviceBuffer<u8>,
    d_values: &mut DeviceBuffer<u8>,
    shape: leone_cuda::AttentionShape,
) -> TestResult {
    for position in 0..shape.max_context() {
        let (projected_key, projected_value) =
            projected_kv_position(keys, values, shape, position)?;
        let d_key = context.copy_to_device(&projected_key)?;
        let d_value = context.copy_to_device(&projected_value)?;
        kv_append_q8(stream, &d_key, &d_value, d_keys, d_values, shape, position)?;
    }
    Ok(())
}

fn projected_kv_position(
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    position: usize,
) -> TestResult<(Vec<f32>, Vec<f32>)> {
    let mut projected_key = Vec::with_capacity(shape.projected_kv_elements()?);
    let mut projected_value = Vec::with_capacity(shape.projected_kv_elements()?);
    for head in 0..shape.n_head_kv() {
        let base = (head * shape.max_context() + position) * shape.head_dim();
        projected_key.extend_from_slice(&keys[base..base + shape.head_dim()]);
        projected_value.extend_from_slice(&values[base..base + shape.head_dim()]);
    }
    Ok((projected_key, projected_value))
}

fn run_q8_attention(
    context: &Context,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    keys: &DeviceBuffer<u8>,
    values: &DeviceBuffer<u8>,
    shape: leone_cuda::AttentionShape,
) -> TestResult<Vec<f32>> {
    let mut d_output = context.alloc(shape.query_elements())?;
    let mut scratch = AttentionScratch::new(context, shape)?;
    attention_decode_q8(
        stream,
        query,
        keys,
        values,
        &mut d_output,
        &mut scratch,
        None,
        shape,
        shape.max_context(),
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; shape.query_elements()];
    d_output.copy_to(&mut actual)?;
    Ok(actual)
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn q8_chunked_prefill_matches_scalar_cache_and_attention_oracles() -> TestResult {
    let tokens = 17;
    let shape = leone_cuda::AttentionShape::new(4, 2, 128, 33)?;
    let projected = shape.projected_kv_elements()?;
    let keys = random_f32(tokens * projected, 0x7138_5f63_6875_6e6b, -0.5..0.5);
    let values = random_f32(tokens * projected, 0x7138_5f70_7265_6669, -0.5..0.5);
    let query = random_f32(
        tokens * shape.query_elements(),
        0x7138_5f61_7474_6e00,
        -0.5..0.5,
    );
    let cache_bytes = shape.cache_elements() / 32 * 34;
    let Q8ChunkedPrefillResults {
        chunk_key_bytes,
        chunk_value_bytes,
        scalar_key_bytes,
        scalar_value_bytes,
        actual,
    } = run_q8_chunked_prefill_device(
        &query,
        &keys,
        &values,
        shape,
        tokens,
        projected,
        cache_bytes,
    )?;
    assert_eq!(chunk_key_bytes, scalar_key_bytes, "chunked q8 key cache");
    assert_eq!(
        chunk_value_bytes, scalar_value_bytes,
        "chunked q8 value cache"
    );

    let decoded_keys = q8_kv_decode(&chunk_key_bytes, shape)
        .into_iter()
        .map(|value| f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let decoded_values = q8_kv_decode(&chunk_value_bytes, shape)
        .into_iter()
        .map(|value| f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let expected =
        prefill_attention_oracle(&query, &decoded_keys, &decoded_values, shape, 0, tokens);
    let errors = assert_close("q8 chunked prefill", &actual, &expected, 4e-4, 4e-4);
    eprintln!("q8 chunked prefill {errors}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_q8_chunked_prefill_device(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    tokens: usize,
    projected: usize,
    cache_bytes: usize,
) -> TestResult<Q8ChunkedPrefillResults> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let Q8ChunkedCache {
        chunk_key_bytes,
        chunk_value_bytes,
        scalar_key_bytes,
        scalar_value_bytes,
        chunk_keys,
        chunk_values,
    } = run_q8_chunked_cache(
        &context,
        &stream,
        keys,
        values,
        shape,
        tokens,
        projected,
        cache_bytes,
    )?;
    let actual = run_q8_prefill_attention(
        &context,
        &stream,
        &handle,
        query,
        &chunk_keys,
        &chunk_values,
        shape,
        tokens,
    )?;
    Ok(Q8ChunkedPrefillResults {
        chunk_key_bytes,
        chunk_value_bytes,
        scalar_key_bytes,
        scalar_value_bytes,
        actual,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_q8_chunked_cache(
    context: &Context,
    stream: &Stream,
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    tokens: usize,
    projected: usize,
    cache_bytes: usize,
) -> TestResult<Q8ChunkedCache> {
    let Q8ChunkedCacheBuffers {
        key: d_key,
        value: d_value,
        mut chunk_keys,
        mut chunk_values,
        mut scalar_keys,
        mut scalar_values,
    } = q8_chunked_cache_buffers(context, keys, values, cache_bytes)?;
    kv_append_chunk_q8(
        stream,
        &d_key,
        &d_value,
        &mut chunk_keys,
        &mut chunk_values,
        shape,
        0,
        tokens,
    )?;
    append_q8_scalar_positions(
        context,
        stream,
        keys,
        values,
        &mut scalar_keys,
        &mut scalar_values,
        shape,
        tokens,
        projected,
    )?;
    stream.synchronize()?;
    let (chunk_key_bytes, chunk_value_bytes) =
        copy_q8_cache_pair(&chunk_keys, &chunk_values, cache_bytes)?;
    let (scalar_key_bytes, scalar_value_bytes) =
        copy_q8_cache_pair(&scalar_keys, &scalar_values, cache_bytes)?;
    Ok(Q8ChunkedCache {
        chunk_key_bytes,
        chunk_value_bytes,
        scalar_key_bytes,
        scalar_value_bytes,
        chunk_keys,
        chunk_values,
    })
}

fn q8_chunked_cache_buffers(
    context: &Context,
    keys: &[f32],
    values: &[f32],
    cache_bytes: usize,
) -> TestResult<Q8ChunkedCacheBuffers> {
    let empty = vec![0_u8; cache_bytes];
    Ok(Q8ChunkedCacheBuffers {
        key: context.copy_to_device(keys)?,
        value: context.copy_to_device(values)?,
        chunk_keys: context.copy_to_device(&empty)?,
        chunk_values: context.copy_to_device(&empty)?,
        scalar_keys: context.copy_to_device(&empty)?,
        scalar_values: context.copy_to_device(&empty)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_q8_scalar_positions(
    context: &Context,
    stream: &Stream,
    keys: &[f32],
    values: &[f32],
    scalar_keys: &mut DeviceBuffer<u8>,
    scalar_values: &mut DeviceBuffer<u8>,
    shape: leone_cuda::AttentionShape,
    tokens: usize,
    projected: usize,
) -> TestResult {
    for position in 0..tokens {
        let start = position * projected;
        let end = start + projected;
        let key = context.copy_to_device(&keys[start..end])?;
        let value = context.copy_to_device(&values[start..end])?;
        kv_append_q8(
            stream,
            &key,
            &value,
            scalar_keys,
            scalar_values,
            shape,
            position,
        )?;
    }
    Ok(())
}

fn copy_q8_cache_pair(
    keys: &DeviceBuffer<u8>,
    values: &DeviceBuffer<u8>,
    cache_bytes: usize,
) -> TestResult<(Vec<u8>, Vec<u8>)> {
    let mut key_bytes = vec![0_u8; cache_bytes];
    let mut value_bytes = vec![0_u8; cache_bytes];
    keys.copy_to(&mut key_bytes)?;
    values.copy_to(&mut value_bytes)?;
    Ok((key_bytes, value_bytes))
}

#[allow(clippy::too_many_arguments)]
fn run_q8_prefill_attention(
    context: &Context,
    stream: &Stream,
    handle: &CublasLt,
    query: &[f32],
    keys: &DeviceBuffer<u8>,
    values: &DeviceBuffer<u8>,
    shape: leone_cuda::AttentionShape,
    tokens: usize,
) -> TestResult<Vec<f32>> {
    let d_query = context.copy_to_device(query)?;
    let mut d_output = context.alloc(tokens * shape.query_elements())?;
    let plan = PrefillPlan::new(tokens, shape.max_context(), 4, 2, 128, 512, 512, 512)?;
    let mut scratch = PrefillScratch::new(context, plan)?;
    attention_prefill_q8(
        handle,
        stream,
        &d_query,
        keys,
        values,
        &mut d_output,
        shape,
        0,
        tokens,
        &mut scratch,
    )?;
    stream.synchronize()?;
    copy_f32_buffer(&d_output, tokens * shape.query_elements())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn argmax_matches_greedy_oracle_over_real_vocab() -> TestResult {
    let mut values = random_f32(151_936, 0x6172_676d_6178_766f, -10.0..10.0);
    values[0] = f32::NAN;
    values[17] = 100.0;
    values[149_000] = 100.0;
    let actual = run_argmax(&values)?;
    let expected = argmax_expected(&values).ok_or("argmax oracle has no finite values")?;
    assert_eq!(actual, expected);
    eprintln!("argmax index {}, error 0", actual);
    Ok(())
}

fn run_argmax(values: &[f32]) -> TestResult<u32> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let input = context.copy_to_device(values)?;
    let mut output = context.alloc(1)?;
    let mut scratch = ArgmaxScratch::new(&context, values.len())?;
    argmax(&stream, &input, &mut output, &mut scratch)?;
    stream.synchronize()?;
    let mut actual = [0_u32];
    output.copy_to(&mut actual)?;
    Ok(actual[0])
}

fn argmax_expected(values: &[f32]) -> Option<u32> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(index, _)| index as u32)
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn fixed_reductions_are_bitwise_deterministic() -> TestResult {
    let shape = QuantizedMatrixShape::new(4_096, 4_096, QuantFormat::Q4K)?;
    let weights = quantized_bytes(shape, 0x6465_7465_726d_5f77);
    let input = random_f32(shape.columns(), 0x6465_7465_726d_5f78, -0.25..0.25);
    let first = run_gemv(&weights, &input, shape)?;
    let second = run_gemv(&weights, &input, shape)?;
    for (index, (left, right)) in first.iter().zip(&second).enumerate() {
        assert_eq!(left.to_bits(), right.to_bits(), "GEMV output {index}");
    }
    eprintln!("determinism: 4096 GEMV outputs match bitwise");
    Ok(())
}

fn run_gemv(
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let device_weights = device_weight_bytes(weights, shape)?;
    let GemvBuffers {
        weights: d_weights,
        input: d_input,
        output: mut d_output,
        mut scratch,
    } = run_gemv_buffers(&context, &device_weights, input, shape)?;
    run_gemv_launch(
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        &mut scratch,
        shape,
    )?;
    stream.synchronize()?;
    let mut output = vec![0.0; shape.rows()];
    d_output.copy_to(&mut output)?;
    Ok(output)
}

fn run_gemv_buffers(
    context: &Context,
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
) -> TestResult<GemvBuffers> {
    Ok(GemvBuffers {
        weights: context.copy_to_device(weights)?,
        input: context.copy_to_device(input)?,
        output: context.alloc(shape.rows())?,
        scratch: GemvScratch::new(context, shape)?,
    })
}

fn run_gemv_launch(
    stream: &Stream,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut GemvScratch,
    shape: QuantizedMatrixShape,
) -> TestResult {
    match shape.format() {
        QuantFormat::Q4K => gemv_q4_k(stream, weights, input, output, scratch, shape)?,
        QuantFormat::Q6K => gemv_q6_k(stream, weights, input, output, scratch, shape)?,
    }
    Ok(())
}

fn quantized_bytes(shape: QuantizedMatrixShape, seed: u64) -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut bytes = vec![0; shape.bytes()];
    rng.fill_bytes(&mut bytes);
    for row in bytes.chunks_exact_mut(shape.row_bytes()) {
        for block in row.chunks_exact_mut(shape.format().block_bytes()) {
            match shape.format() {
                QuantFormat::Q4K => {
                    set_half(block, 0, rng.random_range(0.001_f32..0.05));
                    set_half(block, 2, rng.random_range(0.001_f32..0.05));
                }
                QuantFormat::Q6K => {
                    set_half(block, 208, rng.random_range(0.001_f32..0.05));
                }
            }
        }
    }
    bytes
}

fn device_weight_bytes(
    bytes: &[u8],
    shape: QuantizedMatrixShape,
) -> Result<Vec<u8>, Box<dyn Error>> {
    match shape.format() {
        QuantFormat::Q4K => Ok(repack_q4_k(bytes)?),
        QuantFormat::Q6K => Ok(bytes.to_vec()),
    }
}

fn random_f32(elements: usize, seed: u64, range: std::ops::Range<f32>) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..elements)
        .map(|_| rng.random_range(range.clone()))
        .collect()
}

fn gemv_oracle(
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
) -> Result<Vec<f64>, Box<dyn Error>> {
    (0..shape.rows())
        .map(|row| gemv_row_oracle(weights, input, shape, row))
        .collect()
}

fn gemv_spot_oracle(
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
    rows: &[usize],
) -> Result<Vec<f64>, Box<dyn Error>> {
    rows.iter()
        .map(|row| gemv_row_oracle(weights, input, shape, *row))
        .collect()
}

fn gemv_row_oracle(
    weights: &[u8],
    input: &[f32],
    shape: QuantizedMatrixShape,
    row: usize,
) -> Result<f64, Box<dyn Error>> {
    let start = row * shape.row_bytes();
    let values = dequant_row(
        shape.format(),
        &weights[start..start + shape.row_bytes()],
        shape.columns(),
    )?;
    Ok(values
        .iter()
        .zip(input)
        .map(|(weight, input)| f64::from(*weight) * f64::from(*input))
        .sum())
}

fn dequant_row(
    format: QuantFormat,
    bytes: &[u8],
    elements: usize,
) -> ref_dequant::Result<Vec<f32>> {
    match format {
        QuantFormat::Q4K => ref_dequant::q4_k::dequant_row(bytes, elements),
        QuantFormat::Q6K => ref_dequant::q6_k::dequant_row(bytes, elements),
    }
}

fn rms_oracle(
    left: &[f32],
    right: Option<&[f32]>,
    weight: &[f32],
    shape: VectorShape,
    epsilon: f32,
) -> Vec<f64> {
    let mut output = Vec::with_capacity(shape.elements());
    for row in 0..shape.rows() {
        let base = row * shape.columns();
        let square_sum = (0..shape.columns())
            .map(|column| {
                let mut value = f64::from(left[base + column]);
                if let Some(right) = right {
                    value += f64::from(right[base + column]);
                }
                value * value
            })
            .sum::<f64>();
        let inverse = 1.0 / (square_sum / shape.columns() as f64 + f64::from(epsilon)).sqrt();
        for column in 0..shape.columns() {
            let mut value = f64::from(left[base + column]);
            if let Some(right) = right {
                value += f64::from(right[base + column]);
            }
            output.push(value * f64::from(weight[column]) * inverse);
        }
    }
    output
}

fn rms_rope_oracle(
    input: &[f32],
    weight: &[f32],
    shape: VectorShape,
    position: usize,
    epsilon: f32,
    theta: f32,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let normalized = rms_oracle(input, None, weight, shape, epsilon)
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let rope_shape = RopeShape::new(1, shape.rows(), shape.columns())?;
    Ok(rope_oracle(
        &normalized,
        &[position as u32],
        rope_shape,
        theta,
    ))
}

fn rope_oracle(input: &[f32], positions: &[u32], shape: RopeShape, theta: f32) -> Vec<f64> {
    let mut output = input.iter().copied().map(f64::from).collect::<Vec<_>>();
    let half = shape.head_dim() / 2;
    for (token, position) in positions.iter().copied().enumerate() {
        for head in 0..shape.heads() {
            let base = (token * shape.heads() + head) * shape.head_dim();
            for pair in 0..half {
                let angle = f64::from(position)
                    * f64::from(theta).powf(-2.0 * pair as f64 / shape.head_dim() as f64);
                let (sine, cosine) = angle.sin_cos();
                let first = f64::from(input[base + pair]);
                let second = f64::from(input[base + pair + half]);
                output[base + pair] = first * cosine - second * sine;
                output[base + pair + half] = first * sine + second * cosine;
            }
        }
    }
    output
}

fn rope_adjacent_oracle(
    input: &[f32],
    position: usize,
    shape: RopeShape,
    theta: f32,
    factors: &[f32],
) -> Vec<f64> {
    let mut output = input.iter().copied().map(f64::from).collect::<Vec<_>>();
    let half = shape.head_dim() / 2;
    for token in 0..shape.tokens() {
        for head in 0..shape.heads() {
            let base = (token * shape.heads() + head) * shape.head_dim();
            for (pair, &factor) in factors.iter().take(half).enumerate() {
                let inverse = f64::from(theta).powf(-2.0 * pair as f64 / shape.head_dim() as f64)
                    / f64::from(factor);
                let angle = (position + token) as f64 * inverse;
                let (sine, cosine) = angle.sin_cos();
                let first_index = base + pair * 2;
                let second_index = first_index + 1;
                let first = f64::from(input[first_index]);
                let second = f64::from(input[second_index]);
                output[first_index] = first * cosine - second * sine;
                output[second_index] = first * sine + second * cosine;
            }
        }
    }
    output
}

fn attention_oracle(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    context_length: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; shape.query_elements()];
    let group_size = shape.n_head() / shape.n_head_kv();
    let scale = 1.0 / (shape.head_dim() as f64).sqrt();
    for query_head in 0..shape.n_head() {
        let kv_head = query_head / group_size;
        let scores = (0..context_length)
            .map(|position| {
                let cache_base = (kv_head * shape.max_context() + position) * shape.head_dim();
                let dot = (0..shape.head_dim())
                    .map(|dimension| {
                        f64::from(query[query_head * shape.head_dim() + dimension])
                            * f64::from(keys[cache_base + dimension])
                    })
                    .sum::<f64>();
                dot * scale
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .sum::<f64>();
        for dimension in 0..shape.head_dim() {
            let numerator = scores
                .iter()
                .enumerate()
                .map(|(position, score)| {
                    let cache_index =
                        (kv_head * shape.max_context() + position) * shape.head_dim() + dimension;
                    (score - maximum).exp() * f64::from(values[cache_index])
                })
                .sum::<f64>();
            output[query_head * shape.head_dim() + dimension] = numerator / denominator;
        }
    }
    output
}

fn q8_kv_encode(values: &[f32], shape: leone_cuda::AttentionShape) -> Vec<u8> {
    assert_eq!(values.len(), shape.cache_elements());
    let blocks_per_row = shape.head_dim() / 32;
    let mut encoded = vec![0_u8; shape.cache_elements() / 32 * 34];
    for head in 0..shape.n_head_kv() {
        for position in 0..shape.max_context() {
            for block in 0..blocks_per_row {
                let logical_start =
                    (head * shape.max_context() + position) * shape.head_dim() + block * 32;
                let source = &values[logical_start..logical_start + 32];
                let maximum = source
                    .iter()
                    .fold(0.0_f32, |current, value| current.max(value.abs()));
                let scale = f16::from_f32(maximum / 127.0);
                let stored_scale = scale.to_f32();
                let byte_start =
                    ((head * shape.max_context() + position) * blocks_per_row + block) * 34;
                encoded[byte_start..byte_start + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
                for (code, value) in encoded[byte_start + 2..byte_start + 34]
                    .iter_mut()
                    .zip(source)
                {
                    let quantized = if stored_scale == 0.0 {
                        0
                    } else {
                        (*value / stored_scale)
                            .round_ties_even()
                            .clamp(-127.0, 127.0) as i8
                    };
                    *code = quantized as u8;
                }
            }
        }
    }
    encoded
}

fn q8_kv_decode(encoded: &[u8], shape: leone_cuda::AttentionShape) -> Vec<f32> {
    assert_eq!(encoded.len(), shape.cache_elements() / 32 * 34);
    (0..shape.cache_elements())
        .map(|logical_index| {
            let block_start = (logical_index / 32) * 34;
            let scale = f16::from_bits(u16::from_le_bytes([
                encoded[block_start],
                encoded[block_start + 1],
            ]))
            .to_f32();
            let code = encoded[block_start + 2 + logical_index % 32] as i8;
            scale * f32::from(code)
        })
        .collect()
}

fn prefill_attention_oracle(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: leone_cuda::AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Vec<f64> {
    let mut output = Vec::with_capacity(tokens * shape.query_elements());
    for token in 0..tokens {
        let base = token * shape.query_elements();
        let rounded_query = query[base..base + shape.query_elements()]
            .iter()
            .map(|value| f16::from_f32(*value).to_f32())
            .collect::<Vec<_>>();
        output.extend(attention_oracle(
            &rounded_query,
            keys,
            values,
            shape,
            start_position + token + 1,
        ));
    }
    output
}

#[derive(Debug, Clone, Copy)]
struct ErrorStats {
    max_abs: f64,
    max_rel: f64,
    max_rel_above_floor: f64,
    max_reference_abs: f64,
}

impl ErrorStats {
    fn relative_l_inf(self) -> f64 {
        if self.max_reference_abs == 0.0 {
            self.max_abs
        } else {
            self.max_abs / self.max_reference_abs
        }
    }
}

impl std::fmt::Display for ErrorStats {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "max_abs={:.6e}, rel_l_inf={:.6e}, max_rel={:.6e}, max_rel_ref_ge_1e-2={:.6e}",
            self.max_abs,
            self.relative_l_inf(),
            self.max_rel,
            self.max_rel_above_floor
        )
    }
}

fn assert_close(
    name: &str,
    actual: &[f32],
    expected: &[f64],
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> ErrorStats {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    let mut stats = ErrorStats {
        max_abs: 0.0,
        max_rel: 0.0,
        max_rel_above_floor: 0.0,
        max_reference_abs: 0.0,
    };
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(*actual);
        let absolute = (actual - expected).abs();
        let relative = if *expected == 0.0 {
            if absolute == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            absolute / expected.abs()
        };
        stats.max_abs = stats.max_abs.max(absolute);
        stats.max_rel = stats.max_rel.max(relative);
        stats.max_reference_abs = stats.max_reference_abs.max(expected.abs());
        if expected.abs() >= 1e-2 {
            stats.max_rel_above_floor = stats.max_rel_above_floor.max(relative);
        }
        let bound = absolute_tolerance + relative_tolerance * expected.abs();
        assert!(
            absolute <= bound,
            "{name} differs at {index}: actual={actual:.9e}, expected={expected:.9e}, abs={absolute:.9e}, bound={bound:.9e}"
        );
    }
    stats
}

fn set_half(block: &mut [u8], offset: usize, value: f32) {
    block[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("crate is under workspace/crates")
        .join("models/Qwen3-8B-Q4_K_M.gguf")
}

fn llama_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("crate is under workspace/crates")
        .join("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
}

struct BatchOracleState<B: leone::Backend> {
    session: GenerationSession<B>,
    transcript: Vec<u32>,
    options: GenerateOptions,
    output: Vec<u32>,
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn request_batch_matches_isolated_greedy_and_stochastic_streams() -> TestResult {
    request_batch_model_oracle(model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Llama 3.2 Q4_K_M model"]
fn llama_request_batch_matches_isolated_streams() -> TestResult {
    request_batch_model_oracle(llama_model_path())
}

fn request_batch_model_oracle(path: PathBuf) -> TestResult {
    let mut runtime = Runtime::load(CudaBackend::new(0)?, path)?;
    let prompts = [
        "Define an invariant in one sentence.",
        "Name two properties of exact sampling.",
        "Explain bounded admission briefly.",
    ];
    let mut options = vec![GenerateOptions::greedy(12); prompts.len()];
    for option in &mut options {
        option.decode_execution = DecodeExecution::Eager;
    }
    options[1].sampler = Sampler::temperature(0.8);
    options[1].seed = 41;
    options[2].sampler = Sampler::temperature(1.1);
    options[2].seed = 97;
    let prompt_tokens = prompts
        .iter()
        .map(|prompt| runtime.model().tokenizer().encode(prompt))
        .collect::<Result<Vec<_>, _>>()?;
    let expected = isolated_batch_oracle(&mut runtime, &prompt_tokens, &options)?;
    let actual = run_request_batch(&mut runtime, &prompt_tokens, &options, &expected)?;
    assert_eq!(actual, expected);
    Ok(())
}

fn isolated_batch_oracle<B: leone::Backend>(
    runtime: &mut Runtime<B>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
) -> TestResult<Vec<Vec<u32>>> {
    let mut expected = Vec::with_capacity(prompts.len());
    for (prompt, option) in prompts.iter().zip(options) {
        let mut session = GenerationSession::new();
        let result = runtime.generate_session_tokens(
            &mut session,
            prompt,
            option.clone(),
            |_| Ok(()),
            || false,
        )?;
        expected.push(result.tokens);
    }
    Ok(expected)
}

fn run_request_batch<B: leone::Backend>(
    runtime: &mut Runtime<B>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
    expected: &[Vec<u32>],
) -> TestResult<Vec<Vec<u32>>> {
    let mut states = prompts
        .iter()
        .zip(options)
        .map(|(prompt, option)| BatchOracleState {
            session: GenerationSession::new(),
            transcript: prompt.clone(),
            options: option.clone(),
            output: Vec::new(),
        })
        .collect::<Vec<_>>();
    initialize_batch(runtime, &mut states)?;
    while states
        .iter()
        .zip(expected)
        .any(|(state, expected)| state.output.len() < expected.len())
    {
        advance_batch(runtime, &mut states, expected)?;
    }
    Ok(states.into_iter().map(|state| state.output).collect())
}

fn initialize_batch<B: leone::Backend>(
    runtime: &mut Runtime<B>,
    states: &mut [BatchOracleState<B>],
) -> TestResult {
    for state in states {
        let mut first = state.options.clone();
        first.max_tokens = 1;
        let result = runtime.generate_session_tokens(
            &mut state.session,
            &state.transcript,
            first,
            |_| Ok(()),
            || false,
        )?;
        state.transcript.extend_from_slice(&result.tokens);
        state.output.extend(result.tokens);
    }
    Ok(())
}

fn advance_batch<B: leone::Backend>(
    runtime: &mut Runtime<B>,
    states: &mut [BatchOracleState<B>],
    expected: &[Vec<u32>],
) -> TestResult {
    let active = states
        .iter()
        .zip(expected)
        .map(|(state, expected)| state.output.len() < expected.len())
        .collect::<Vec<_>>();
    let mut inputs = states
        .iter_mut()
        .zip(&active)
        .filter(|(_, active)| **active)
        .map(|(state, _)| BatchSession {
            session: &mut state.session,
            transcript: &state.transcript,
            options: &state.options,
        })
        .collect::<Vec<_>>();
    let tokens = runtime.generate_session_batch_token(&mut inputs)?;
    drop(inputs);
    let mut tokens = tokens.into_iter();
    for (state, active) in states.iter_mut().zip(active) {
        if active {
            let token = tokens.next().expect("one token per active session");
            state.transcript.push(token.id);
            state.output.push(token.id);
        }
    }
    assert!(tokens.next().is_none());
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn attention_decode_runs_without_the_prepared_epilogue_at_head_dim_64() -> TestResult {
    // The fused q8_1 epilogue covers only head_dim 128. The backend runs any
    // other head dimension without it.
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = leone_cuda::AttentionShape::new(4, 2, 64, 1_024)?;
    let query = random_f32(shape.query_elements(), 0x6864_3634_5f71_7565, -0.5..0.5);
    let keys = random_f32(shape.cache_elements(), 0x6864_3634_5f6b_6579, -0.5..0.5);
    let values = random_f32(shape.cache_elements(), 0x6864_3634_5f76_616c, -0.5..0.5);
    let d_query = context.copy_to_device(&query)?;
    let d_keys = context.copy_to_device(&keys)?;
    let d_values = context.copy_to_device(&values)?;
    let mut d_output = context.alloc(shape.query_elements())?;
    let mut scratch = AttentionScratch::new(&context, shape)?;
    check_attention_decode_cases(
        &stream,
        &d_query,
        &d_keys,
        &d_values,
        &mut d_output,
        &mut scratch,
        &query,
        &keys,
        &values,
        shape,
        &[1, 33, 512, 1_024],
        "head_dim 64 attention",
    )?;
    Ok(())
}
