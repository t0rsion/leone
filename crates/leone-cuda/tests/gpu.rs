use half::f16;
use leone::{GenerateOptions, KvCacheDtype, PrefillPlan, Runtime};
use leone_cuda::{
    argmax, attention_decode, attention_decode_f16, attention_decode_q8, attention_prefill_f16,
    dequantize_k_f16, embedding_gather_q4_k, embedding_gather_q6_k, gemv_q4_k, gemv_q4_k_residual,
    gemv_q6_k, gemv_q6_k_residual, kv_append_f16, kv_append_q8, qk_norm_rope,
    qk_norm_rope_kv_append_f16, qkv_gemv, repack_q4_k, residual_add, rms_norm,
    rms_norm_q8_parallel, rms_norm_residual, rms_norm_residual_store, rms_norm_rope,
    rope_at_frequencies, rope_neox, swiglu, swiglu_q8, ArgmaxScratch, AttentionScratch, Context,
    CublasLt, CudaBackend, GemvScratch, PrefillScratch, QuantFormat, QuantizedMatrixShape,
    RopeScratch, RopeShape, Stream, VectorShape,
};
use leone_gguf::ref_dequant;
use leone_gguf::{GgmlType, Gguf};
use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use std::error::Error;
use std::path::PathBuf;

type TestResult = Result<(), Box<dyn Error>>;

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
        let shape = QuantizedMatrixShape::new(3, 512, format)?;
        let source = quantized_bytes(shape, seed);
        let device_source = device_weight_bytes(&source, shape)?;
        let context = Context::new(0)?;
        let stream = Stream::new(&context)?;
        let d_source = context.copy_to_device(&device_source)?;
        let mut d_output = context.alloc(shape.rows() * shape.columns())?;
        dequantize_k_f16(&stream, &d_source, &mut d_output, shape)?;
        stream.synchronize()?;
        let mut observed = vec![0_u16; shape.rows() * shape.columns()];
        d_output.copy_to(&mut observed)?;
        let mut expected = Vec::with_capacity(observed.len());
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
        assert_eq!(observed, expected, "{format:?} FP16 dequantization");
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn prefill_gemm_matches_fp16_input_and_weight_oracle() -> TestResult {
    let shape = QuantizedMatrixShape::new(37, 512, QuantFormat::Q4K)?;
    let tokens = 7;
    let weights = quantized_bytes(shape, 0x7072_6566_6765_6d6d);
    let device_weights = device_weight_bytes(&weights, shape)?;
    let input = random_f32(tokens * shape.columns(), 0x7072_6566_696e_7074, -0.25..0.25);
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let plan = PrefillPlan::new(tokens, tokens, 4, 2, 32, 512, 512, 512)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let d_weights = context.copy_to_device(&device_weights)?;
    let d_input = context.copy_to_device(&input)?;
    let mut d_output = context.alloc(tokens * shape.rows())?;
    leone_cuda::prefill_gemm(
        &handle,
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        shape,
        tokens,
        &mut scratch,
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; tokens * shape.rows()];
    d_output.copy_to(&mut actual)?;
    leone_cuda::prefill_gemm(
        &handle,
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        shape,
        tokens,
        &mut scratch,
    )?;
    stream.synchronize()?;
    let mut repeated = vec![0.0; actual.len()];
    d_output.copy_to(&mut repeated)?;
    assert_f32_bitwise_equal(&actual, &repeated, "prefill GEMM repeat");
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
    let mut expected = vec![0.0_f64; actual.len()];
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
    let errors = assert_close("prefill GEMM", &actual, &expected, 2e-2, 2e-3);
    eprintln!("prefill GEMM {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn real_prefill_gemm_is_bitwise_deterministic() -> TestResult {
    let gguf = Gguf::open(model_path())?;
    let tensor_name = "blk.0.ffn_gate.weight";
    let shape = QuantizedMatrixShape::new(12_288, 4_096, QuantFormat::Q4K)?;
    let weights = gguf.tensor_data(tensor_name)?;
    let device_weights = device_weight_bytes(&weights, shape)?;
    let tokens = 512;
    let input = random_f32(tokens * shape.columns(), 0x7072_6566_7265_616c, -0.25..0.25);
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let plan = PrefillPlan::new(tokens, tokens, 32, 8, 128, 4_096, 12_288, 12_288)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let d_weights = context.copy_to_device(&device_weights)?;
    let d_input = context.copy_to_device(&input)?;
    let mut d_output = context.alloc(tokens * shape.rows())?;
    leone_cuda::prefill_gemm(
        &handle,
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        shape,
        tokens,
        &mut scratch,
    )?;
    stream.synchronize()?;
    let mut first = vec![0.0; tokens * shape.rows()];
    d_output.copy_to(&mut first)?;
    leone_cuda::prefill_gemm(
        &handle,
        &stream,
        &d_weights,
        &d_input,
        &mut d_output,
        shape,
        tokens,
        &mut scratch,
    )?;
    stream.synchronize()?;
    let mut second = vec![0.0; first.len()];
    d_output.copy_to(&mut second)?;
    assert_f32_bitwise_equal(&first, &second, "real prefill GEMM repeat");
    Ok(())
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
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let d_query = context.copy_to_device(&query)?;
    let d_keys = context.copy_to_device(&keys)?;
    let d_values = context.copy_to_device(&values)?;
    let mut d_output = context.alloc(tokens * shape.query_elements())?;
    attention_prefill_f16(
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
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; tokens * shape.query_elements()];
    d_output.copy_to(&mut actual)?;
    attention_prefill_f16(
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
    )?;
    stream.synchronize()?;
    let mut repeated = vec![0.0; actual.len()];
    d_output.copy_to(&mut repeated)?;
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

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn production_prefill_attention_is_bitwise_deterministic() -> TestResult {
    let tokens = 512;
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
        .collect::<Vec<_>>();
    let values = random_f32(shape.cache_elements(), 0x7072_6f64_7661_6c73, -0.5..0.5)
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let mut scratch = PrefillScratch::new(&context, plan)?;
    let d_query = context.copy_to_device(&query)?;
    let d_keys = context.copy_to_device(&keys)?;
    let d_values = context.copy_to_device(&values)?;
    let mut d_output = context.alloc(tokens * shape.query_elements())?;
    attention_prefill_f16(
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
    )?;
    stream.synchronize()?;
    let mut first = vec![0.0; tokens * shape.query_elements()];
    d_output.copy_to(&mut first)?;
    attention_prefill_f16(
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
    )?;
    stream.synchronize()?;
    let mut second = vec![0.0; first.len()];
    d_output.copy_to(&mut second)?;
    assert_f32_bitwise_equal(&first, &second, "production prefill attention repeat");
    Ok(())
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
        let shape = QuantizedMatrixShape::new(37, columns, format)?;
        let weights = quantized_bytes(shape, seed);
        let device_weights = device_weight_bytes(&weights, shape)?;
        for positions in 1..=8 {
            let input = random_f32(positions * columns, seed ^ positions as u64, -0.25..0.25);
            let context = Context::new(0)?;
            let stream = Stream::new(&context)?;
            let d_weights = context.copy_to_device(&device_weights)?;
            let d_input = context.copy_to_device(&input)?;
            let mut d_output = context.alloc(positions * shape.rows())?;
            let mut scratch = GemvScratch::new_multi(&context, columns, positions)?;
            leone_cuda::verify_gemv(
                &stream,
                &d_weights,
                &d_input,
                None,
                &mut d_output,
                &mut scratch,
                shape,
                positions,
            )?;
            stream.synchronize()?;
            let mut actual = vec![0.0; positions * shape.rows()];
            d_output.copy_to(&mut actual)?;
            for position in 0..positions {
                let expected = run_gemv(
                    &weights,
                    &input[position * columns..(position + 1) * columns],
                    shape,
                )?;
                assert_f32_bitwise_equal(
                    &actual[position * shape.rows()..(position + 1) * shape.rows()],
                    &expected,
                    &format!("{format:?} width {positions} position {position}"),
                );
            }
        }
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
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let query_shape = QuantizedMatrixShape::new(37, 1_024, QuantFormat::Q4K)?;
    let key_shape = QuantizedMatrixShape::new(11, 1_024, QuantFormat::Q6K)?;
    let value_shape = QuantizedMatrixShape::new(11, 1_024, QuantFormat::Q4K)?;
    let query_weights = quantized_bytes(query_shape, 0x716b_765f_7177_6774);
    let key_weights = quantized_bytes(key_shape, 0x716b_765f_6b77_6774);
    let value_weights = quantized_bytes(value_shape, 0x716b_765f_7677_6774);
    let input = random_f32(1_024, 0x716b_765f_696e_7074, -0.25..0.25);
    let query_device_weights = device_weight_bytes(&query_weights, query_shape)?;
    let value_device_weights = device_weight_bytes(&value_weights, value_shape)?;
    let d_query_weights = context.copy_to_device(&query_device_weights)?;
    let d_key_weights = context.copy_to_device(&key_weights)?;
    let d_value_weights = context.copy_to_device(&value_device_weights)?;
    let d_input = context.copy_to_device(&input)?;
    let mut d_query = context.alloc(query_shape.rows())?;
    let mut d_key = context.alloc(key_shape.rows())?;
    let mut d_value = context.alloc(value_shape.rows())?;
    let mut scratch = GemvScratch::new(&context, query_shape)?;
    qkv_gemv(
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
        false,
    )?;
    stream.synchronize()?;
    let mut query = vec![0.0; query_shape.rows()];
    let mut key = vec![0.0; key_shape.rows()];
    let mut value = vec![0.0; value_shape.rows()];
    d_query.copy_to(&mut query)?;
    d_key.copy_to(&mut key)?;
    d_value.copy_to(&mut value)?;
    let query_expected = gemv_oracle(&query_weights, &input, query_shape)?;
    let key_expected = gemv_oracle(&key_weights, &input, key_shape)?;
    let value_expected = gemv_oracle(&value_weights, &input, value_shape)?;
    let query_errors = assert_close("QKV query", &query, &query_expected, 1.0, 2e-2);
    let key_errors = assert_close("QKV key", &key, &key_expected, 1.0, 2e-2);
    let value_errors = assert_close("QKV value", &value, &value_expected, 1.0, 2e-2);
    eprintln!("QKV query {query_errors}; key {key_errors}; value {value_errors}");
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
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = VectorShape::new(3, 4_096)?;
    let left = random_f32(shape.elements(), 0x726d_736e_5f6c_6566, -1.0..1.0);
    let right = random_f32(shape.elements(), 0x726d_736e_5f72_6967, -0.5..0.5);
    let weight = random_f32(shape.columns(), 0x726d_736e_5f77_6768, 0.5..1.5);
    let epsilon = 1e-6_f32;
    let d_left = context.copy_to_device(&left)?;
    let d_right = context.copy_to_device(&right)?;
    let d_weight = context.copy_to_device(&weight)?;
    let mut d_plain = context.alloc(shape.elements())?;
    let mut d_residual = context.alloc(shape.elements())?;
    let mut d_stored = context.alloc(shape.elements())?;
    let mut d_stored_norm = context.alloc(shape.elements())?;
    rms_norm(&stream, &d_left, &d_weight, &mut d_plain, shape, epsilon)?;
    rms_norm_residual(
        &stream,
        &d_left,
        &d_right,
        &d_weight,
        &mut d_residual,
        shape,
        epsilon,
    )?;
    rms_norm_residual_store(
        &stream,
        &d_left,
        &d_right,
        &d_weight,
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

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn parallel_rms_norm_q8_matches_standalone_path_bitwise() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let vector_shape = VectorShape::new(1, 4_096)?;
    let gemv_shape = QuantizedMatrixShape::new(67, 4_096, QuantFormat::Q4K)?;
    let input = random_f32(vector_shape.elements(), 0x726d_735f_7138_696e, -1.0..1.0);
    let weight = random_f32(vector_shape.columns(), 0x726d_735f_7138_7767, 0.5..1.5);
    let gemv_weights = quantized_bytes(gemv_shape, 0x726d_735f_7138_6776);
    let d_input = context.copy_to_device(&input)?;
    let d_weight = context.copy_to_device(&weight)?;
    let gemv_device_weights = device_weight_bytes(&gemv_weights, gemv_shape)?;
    let d_gemv_weights = context.copy_to_device(&gemv_device_weights)?;
    let mut d_standalone = context.alloc(vector_shape.elements())?;
    let mut d_parallel = context.alloc(vector_shape.elements())?;
    let mut d_standalone_projection = context.alloc(gemv_shape.rows())?;
    let mut d_parallel_projection = context.alloc(gemv_shape.rows())?;
    let mut standalone_scratch = GemvScratch::new(&context, gemv_shape)?;
    let mut parallel_scratch = GemvScratch::new(&context, gemv_shape)?;
    rms_norm(
        &stream,
        &d_input,
        &d_weight,
        &mut d_standalone,
        vector_shape,
        1e-6,
    )?;
    rms_norm_q8_parallel(
        &stream,
        &d_input,
        &d_weight,
        &mut d_parallel,
        &mut parallel_scratch,
        vector_shape,
        1e-6,
    )?;
    gemv_q4_k(
        &stream,
        &d_gemv_weights,
        &d_standalone,
        &mut d_standalone_projection,
        &mut standalone_scratch,
        gemv_shape,
    )?;
    gemv_q4_k_residual(
        &stream,
        &d_gemv_weights,
        &d_parallel,
        None,
        &mut d_parallel_projection,
        &mut parallel_scratch,
        gemv_shape,
        true,
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
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = VectorShape::new(6, 128)?;
    let rope_shape = RopeShape::new(1, shape.rows(), shape.columns())?;
    let input = random_f32(shape.elements(), 0x6675_7365_645f_726d, -1.0..1.0);
    let weight = random_f32(shape.columns(), 0x6675_7365_645f_7774, 0.5..1.5);
    let position = 2_048;
    let epsilon = 1e-6_f32;
    let theta = 1_000_000.0_f32;
    let d_input = context.copy_to_device(&input)?;
    let d_weight = context.copy_to_device(&weight)?;
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
    let mut actual = vec![0.0; shape.elements()];
    d_output.copy_to(&mut actual)?;
    let normalized = rms_oracle(&input, None, &weight, shape, epsilon)
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let expected = rope_oracle(&normalized, &[position as u32], rope_shape, theta);
    let errors = assert_close("fused RMSNorm RoPE", &actual, &expected, 5e-6, 2e-5);
    eprintln!("fused RMSNorm RoPE {errors}");
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn qk_norm_rope_matches_two_composed_oracles() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
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
    let d_query = context.copy_to_device(&query)?;
    let d_key = context.copy_to_device(&key)?;
    let d_value = context.copy_to_device(&value)?;
    let d_query_weight = context.copy_to_device(&query_weight)?;
    let d_key_weight = context.copy_to_device(&key_weight)?;
    let mut d_query_output = context.alloc(query_shape.elements())?;
    let mut d_key_output = context.alloc(key_shape.elements())?;
    let mut rope_scratch = RopeScratch::new(&context, query_shape.columns(), theta, None, false)?;
    rope_scratch.prepare(&stream, position)?;
    qk_norm_rope(
        &stream,
        &d_query,
        &d_query_weight,
        &mut d_query_output,
        query_shape,
        &d_key,
        &d_key_weight,
        &mut d_key_output,
        key_shape,
        &rope_scratch,
        epsilon,
    )?;
    let attention_shape = leone_cuda::AttentionShape::new(32, 8, 128, 4_096)?;
    let empty_cache = vec![0_u16; attention_shape.cache_elements()];
    let mut d_key_cache = context.copy_to_device(&empty_cache)?;
    let mut d_value_cache = context.copy_to_device(&empty_cache)?;
    kv_append_f16(
        &stream,
        &d_key_output,
        &d_value,
        &mut d_key_cache,
        &mut d_value_cache,
        attention_shape,
        position,
    )?;
    let mut d_fused_query_output = context.alloc(query_shape.elements())?;
    let mut d_fused_key_output = context.alloc(key_shape.elements())?;
    let mut d_fused_key_cache = context.copy_to_device(&empty_cache)?;
    let mut d_fused_value_cache = context.copy_to_device(&empty_cache)?;
    qk_norm_rope_kv_append_f16(
        &stream,
        &d_query,
        &d_query_weight,
        &mut d_fused_query_output,
        query_shape,
        &d_key,
        &d_key_weight,
        &mut d_fused_key_output,
        key_shape,
        &d_value,
        &mut d_fused_key_cache,
        &mut d_fused_value_cache,
        attention_shape,
        position,
        &rope_scratch,
        epsilon,
    )?;
    stream.synchronize()?;
    let mut query_actual = vec![0.0; query_shape.elements()];
    let mut key_actual = vec![0.0; key_shape.elements()];
    d_query_output.copy_to(&mut query_actual)?;
    d_key_output.copy_to(&mut key_actual)?;
    let mut fused_query = vec![0.0; query_shape.elements()];
    let mut fused_key = vec![0.0; key_shape.elements()];
    d_fused_query_output.copy_to(&mut fused_query)?;
    d_fused_key_output.copy_to(&mut fused_key)?;
    assert_eq!(query_actual, fused_query);
    assert_eq!(key_actual, fused_key);
    let mut key_cache = vec![0_u16; attention_shape.cache_elements()];
    let mut value_cache = vec![0_u16; attention_shape.cache_elements()];
    let mut fused_key_cache = vec![0_u16; attention_shape.cache_elements()];
    let mut fused_value_cache = vec![0_u16; attention_shape.cache_elements()];
    d_key_cache.copy_to(&mut key_cache)?;
    d_value_cache.copy_to(&mut value_cache)?;
    d_fused_key_cache.copy_to(&mut fused_key_cache)?;
    d_fused_value_cache.copy_to(&mut fused_value_cache)?;
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

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn swiglu_and_residual_add_match_oracles() -> TestResult {
    // CUDA and the f64 SwiGLU oracle differ only in the transcendental and
    // final rounding. The bound is 2e-6 absolute plus 3e-6 relative.
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let gate = random_f32(12_289, 0x7377_6967_6c75_6761, -8.0..8.0);
    let up = random_f32(12_289, 0x7377_6967_6c75_7570, -2.0..2.0);
    let d_gate = context.copy_to_device(&gate)?;
    let d_up = context.copy_to_device(&up)?;
    let mut d_swiglu = context.alloc(gate.len())?;
    let mut d_add = context.alloc(gate.len())?;
    swiglu(&stream, &d_gate, &d_up, &mut d_swiglu)?;
    residual_add(&stream, &d_gate, &d_up, &mut d_add)?;
    stream.synchronize()?;
    let mut actual = vec![0.0; gate.len()];
    d_swiglu.copy_to(&mut actual)?;
    let expected = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| {
            let value = f64::from(*gate);
            (value / (1.0 + (-value).exp()) * f64::from(*up)) as f32 as f64
        })
        .collect::<Vec<_>>();
    let errors = assert_close("SwiGLU", &actual, &expected, 2e-6, 3e-6);
    let mut added = vec![0.0; gate.len()];
    d_add.copy_to(&mut added)?;
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

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn gemv_epilogues_match_composed_kernels_bitwise() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    for (format, seed) in [
        (QuantFormat::Q4K, 0x6570_696c_6f67_7134),
        (QuantFormat::Q6K, 0x6570_696c_6f67_7136),
    ] {
        let shape = QuantizedMatrixShape::new(67, 1_024, format)?;
        let weights = quantized_bytes(shape, seed);
        let input = random_f32(shape.columns(), seed ^ 0x1111, -0.25..0.25);
        let residual = random_f32(shape.rows(), seed ^ 0x2222, -0.5..0.5);
        let device_weights = device_weight_bytes(&weights, shape)?;
        let d_weights = context.copy_to_device(&device_weights)?;
        let d_input = context.copy_to_device(&input)?;
        let d_residual = context.copy_to_device(&residual)?;
        let mut d_projection = context.alloc(shape.rows())?;
        let mut d_composed = context.alloc(shape.rows())?;
        let mut d_fused = context.alloc(shape.rows())?;
        let mut composed_scratch = GemvScratch::new(&context, shape)?;
        let mut fused_scratch = GemvScratch::new(&context, shape)?;
        match format {
            QuantFormat::Q4K => {
                gemv_q4_k(
                    &stream,
                    &d_weights,
                    &d_input,
                    &mut d_projection,
                    &mut composed_scratch,
                    shape,
                )?;
                gemv_q4_k_residual(
                    &stream,
                    &d_weights,
                    &d_input,
                    Some(&d_residual),
                    &mut d_fused,
                    &mut fused_scratch,
                    shape,
                    false,
                )?;
            }
            QuantFormat::Q6K => {
                gemv_q6_k(
                    &stream,
                    &d_weights,
                    &d_input,
                    &mut d_projection,
                    &mut composed_scratch,
                    shape,
                )?;
                gemv_q6_k_residual(
                    &stream,
                    &d_weights,
                    &d_input,
                    Some(&d_residual),
                    &mut d_fused,
                    &mut fused_scratch,
                    shape,
                    false,
                )?;
            }
        }
        residual_add(&stream, &d_projection, &d_residual, &mut d_composed)?;
        stream.synchronize()?;
        let mut composed = vec![0.0; shape.rows()];
        let mut fused = vec![0.0; shape.rows()];
        d_composed.copy_to(&mut composed)?;
        d_fused.copy_to(&mut fused)?;
        for (index, (actual, expected)) in fused.iter().zip(&composed).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{format:?} row {index}"
            );
        }
    }
    eprintln!("Q4_K and Q6_K residual epilogues match composed kernels bitwise");
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
    let device_weights = device_weight_bytes(&weights, shape)?;
    let d_weights = context.copy_to_device(&device_weights)?;
    let d_gate = context.copy_to_device(&gate)?;
    let d_up = context.copy_to_device(&up)?;
    let mut d_standalone = context.alloc(shape.columns())?;
    let mut d_fused = context.alloc(shape.columns())?;
    let mut d_standalone_projection = context.alloc(shape.rows())?;
    let mut d_fused_projection = context.alloc(shape.rows())?;
    let mut standalone_scratch = GemvScratch::new(&context, shape)?;
    let mut fused_scratch = GemvScratch::new(&context, shape)?;
    swiglu(&stream, &d_gate, &d_up, &mut d_standalone)?;
    swiglu_q8(&stream, &d_gate, &d_up, &mut d_fused, &mut fused_scratch)?;
    gemv_q4_k(
        &stream,
        &d_weights,
        &d_standalone,
        &mut d_standalone_projection,
        &mut standalone_scratch,
        shape,
    )?;
    gemv_q4_k_residual(
        &stream,
        &d_weights,
        &d_fused,
        None,
        &mut d_fused_projection,
        &mut fused_scratch,
        shape,
        true,
    )?;
    stream.synchronize()?;
    let mut standalone = vec![0.0; shape.columns()];
    let mut fused = vec![0.0; shape.columns()];
    d_standalone.copy_to(&mut standalone)?;
    d_fused.copy_to(&mut fused)?;
    let mut standalone_projection = vec![0.0; shape.rows()];
    let mut fused_projection = vec![0.0; shape.rows()];
    d_standalone_projection.copy_to(&mut standalone_projection)?;
    d_fused_projection.copy_to(&mut fused_projection)?;
    for (index, (actual, expected)) in fused.iter().zip(&standalone).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "SwiGLU value {index}");
    }
    for (index, (actual, expected)) in fused_projection
        .iter()
        .zip(&standalone_projection)
        .enumerate()
    {
        assert_eq!(actual.to_bits(), expected.to_bits(), "GEMV row {index}");
    }
    eprintln!("SwiGLU q8_1 epilogue and downstream GEMV match bitwise");
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
        let context = Context::new(0)?;
        let stream = Stream::new(&context)?;
        let shape = QuantizedMatrixShape::new(11, 4_096, format)?;
        let table = quantized_bytes(shape, seed);
        let row = 7;
        let device_table = device_weight_bytes(&table, shape)?;
        let d_table = context.copy_to_device(&device_table)?;
        let mut d_output = context.alloc(shape.columns())?;
        match format {
            QuantFormat::Q4K => {
                embedding_gather_q4_k(&stream, &d_table, &mut d_output, shape, row)?
            }
            QuantFormat::Q6K => {
                embedding_gather_q6_k(&stream, &d_table, &mut d_output, shape, row)?
            }
        }
        stream.synchronize()?;
        let mut actual = vec![0.0; shape.columns()];
        d_output.copy_to(&mut actual)?;
        let row_start = row * shape.row_bytes();
        let expected = dequant_row(
            format,
            &table[row_start..row_start + shape.row_bytes()],
            shape.columns(),
        )?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
        let errors = assert_close("embedding gather", &actual, &expected, 2e-6, 2e-6);
        eprintln!("{format:?} embedding gather {errors}");
    }
    Ok(())
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
    for context_length in [1, 17, 256, 2_048, 8_000] {
        attention_decode(
            &stream,
            &d_query,
            &d_keys,
            &d_values,
            &mut d_output,
            &mut scratch,
            None,
            shape,
            context_length,
        )?;
        stream.synchronize()?;
        let mut actual = vec![0.0; shape.query_elements()];
        d_output.copy_to(&mut actual)?;
        let expected = attention_oracle(&query, &keys, &values, shape, context_length);
        let errors = assert_close("decode attention", &actual, &expected, 3e-5, 3e-4);
        eprintln!("attention context {context_length}: {errors}");
    }
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
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = leone_cuda::AttentionShape::new(4, 2, 128, 17)?;
    let query = random_f32(shape.query_elements(), 0x6631_365f_7175_6572, -0.5..0.5);
    let keys = random_f32(shape.cache_elements(), 0x6631_365f_6b65_7973, -0.5..0.5);
    let values = random_f32(shape.cache_elements(), 0x6631_365f_7661_6c73, -0.5..0.5);
    let d_query = context.copy_to_device(&query)?;
    let mut d_keys = context.alloc::<u16>(shape.cache_elements())?;
    let mut d_values = context.alloc::<u16>(shape.cache_elements())?;
    for position in 0..shape.max_context() {
        let mut projected_key = Vec::with_capacity(shape.projected_kv_elements()?);
        let mut projected_value = Vec::with_capacity(shape.projected_kv_elements()?);
        for head in 0..shape.n_head_kv() {
            let base = (head * shape.max_context() + position) * shape.head_dim();
            projected_key.extend_from_slice(&keys[base..base + shape.head_dim()]);
            projected_value.extend_from_slice(&values[base..base + shape.head_dim()]);
        }
        let d_key = context.copy_to_device(&projected_key)?;
        let d_value = context.copy_to_device(&projected_value)?;
        kv_append_f16(
            &stream,
            &d_key,
            &d_value,
            &mut d_keys,
            &mut d_values,
            shape,
            position,
        )?;
    }
    let mut d_output = context.alloc(shape.query_elements())?;
    let mut scratch = AttentionScratch::new(&context, shape)?;
    attention_decode_f16(
        &stream,
        &d_query,
        &d_keys,
        &d_values,
        &mut d_output,
        &mut scratch,
        None,
        shape,
        shape.max_context(),
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; shape.query_elements()];
    d_output.copy_to(&mut actual)?;
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
    let downstream_shape = QuantizedMatrixShape::new(37, shape.query_elements(), QuantFormat::Q4K)?;
    let weights = quantized_bytes(downstream_shape, 0x6174_746e_5f71_385f);
    let device_weights = device_weight_bytes(&weights, downstream_shape)?;
    let d_weights = context.copy_to_device(&device_weights)?;
    let mut d_prepared_output = context.alloc(shape.query_elements())?;
    let mut prepared_scratch = GemvScratch::new(&context, downstream_shape)?;
    attention_decode_f16(
        &stream,
        &d_query,
        &d_keys,
        &d_values,
        &mut d_prepared_output,
        &mut scratch,
        Some(&mut prepared_scratch),
        shape,
        shape.max_context(),
    )?;
    let mut d_standalone_projection = context.alloc(downstream_shape.rows())?;
    let mut d_prepared_projection = context.alloc(downstream_shape.rows())?;
    let mut standalone_scratch = GemvScratch::new(&context, downstream_shape)?;
    gemv_q4_k(
        &stream,
        &d_weights,
        &d_output,
        &mut d_standalone_projection,
        &mut standalone_scratch,
        downstream_shape,
    )?;
    gemv_q4_k_residual(
        &stream,
        &d_weights,
        &d_prepared_output,
        None,
        &mut d_prepared_projection,
        &mut prepared_scratch,
        downstream_shape,
        true,
    )?;
    stream.synchronize()?;
    let mut prepared_output = vec![0.0; shape.query_elements()];
    d_prepared_output.copy_to(&mut prepared_output)?;
    let mut standalone_projection = vec![0.0; downstream_shape.rows()];
    let mut prepared_projection = vec![0.0; downstream_shape.rows()];
    d_standalone_projection.copy_to(&mut standalone_projection)?;
    d_prepared_projection.copy_to(&mut prepared_projection)?;
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

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn q8_kv_append_and_attention_match_scalar_oracle() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let shape = leone_cuda::AttentionShape::new(4, 2, 128, 17)?;
    let query = random_f32(shape.query_elements(), 0x7138_5f71_7565_7279, -0.5..0.5);
    let keys = random_f32(shape.cache_elements(), 0x7138_5f6b_6579_7300, -0.5..0.5);
    let values = random_f32(shape.cache_elements(), 0x7138_5f76_616c_7565, -0.5..0.5);
    let d_query = context.copy_to_device(&query)?;
    let cache_bytes = shape.cache_elements() / 32 * 34;
    let mut d_keys = context.alloc::<u8>(cache_bytes)?;
    let mut d_values = context.alloc::<u8>(cache_bytes)?;
    for position in 0..shape.max_context() {
        let mut projected_key = Vec::with_capacity(shape.projected_kv_elements()?);
        let mut projected_value = Vec::with_capacity(shape.projected_kv_elements()?);
        for head in 0..shape.n_head_kv() {
            let base = (head * shape.max_context() + position) * shape.head_dim();
            projected_key.extend_from_slice(&keys[base..base + shape.head_dim()]);
            projected_value.extend_from_slice(&values[base..base + shape.head_dim()]);
        }
        let d_key = context.copy_to_device(&projected_key)?;
        let d_value = context.copy_to_device(&projected_value)?;
        kv_append_q8(
            &stream,
            &d_key,
            &d_value,
            &mut d_keys,
            &mut d_values,
            shape,
            position,
        )?;
    }
    stream.synchronize()?;
    let expected_keys = q8_kv_encode(&keys, shape);
    let expected_values = q8_kv_encode(&values, shape);
    let mut actual_keys = vec![0_u8; cache_bytes];
    let mut actual_values = vec![0_u8; cache_bytes];
    d_keys.copy_to(&mut actual_keys)?;
    d_values.copy_to(&mut actual_values)?;
    assert_eq!(actual_keys, expected_keys, "q8 key cache bytes");
    assert_eq!(actual_values, expected_values, "q8 value cache bytes");

    let mut d_output = context.alloc(shape.query_elements())?;
    let mut scratch = AttentionScratch::new(&context, shape)?;
    attention_decode_q8(
        &stream,
        &d_query,
        &d_keys,
        &d_values,
        &mut d_output,
        &mut scratch,
        None,
        shape,
        shape.max_context(),
    )?;
    stream.synchronize()?;
    let mut actual = vec![0.0; shape.query_elements()];
    d_output.copy_to(&mut actual)?;
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

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn argmax_matches_greedy_oracle_over_real_vocab() -> TestResult {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let mut values = random_f32(151_936, 0x6172_676d_6178_766f, -10.0..10.0);
    values[0] = f32::NAN;
    values[17] = 100.0;
    values[149_000] = 100.0;
    let input = context.copy_to_device(&values)?;
    let mut output = context.alloc(1)?;
    let mut scratch = ArgmaxScratch::new(&context, values.len())?;
    argmax(&stream, &input, &mut output, &mut scratch)?;
    stream.synchronize()?;
    let mut actual = [0_u32];
    output.copy_to(&mut actual)?;
    let expected = values
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(index, _)| index as u32)
        .ok_or("argmax oracle has no finite values")?;
    assert_eq!(actual[0], expected);
    eprintln!("argmax index {}, error 0", actual[0]);
    Ok(())
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
    let d_weights = context.copy_to_device(&device_weights)?;
    let d_input = context.copy_to_device(input)?;
    let mut d_output = context.alloc(shape.rows())?;
    let mut scratch = GemvScratch::new(&context, shape)?;
    match shape.format() {
        QuantFormat::Q4K => gemv_q4_k(
            &stream,
            &d_weights,
            &d_input,
            &mut d_output,
            &mut scratch,
            shape,
        )?,
        QuantFormat::Q6K => gemv_q6_k(
            &stream,
            &d_weights,
            &d_input,
            &mut d_output,
            &mut scratch,
            shape,
        )?,
    }
    stream.synchronize()?;
    let mut output = vec![0.0; shape.rows()];
    d_output.copy_to(&mut output)?;
    Ok(output)
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
    for context_length in [1, 33, 512, 1_024] {
        attention_decode(
            &stream,
            &d_query,
            &d_keys,
            &d_values,
            &mut d_output,
            &mut scratch,
            None,
            shape,
            context_length,
        )?;
        stream.synchronize()?;
        let mut actual = vec![0.0; shape.query_elements()];
        d_output.copy_to(&mut actual)?;
        let expected = attention_oracle(&query, &keys, &values, shape, context_length);
        let errors = assert_close("head_dim 64 attention", &actual, &expected, 3e-5, 3e-4);
        eprintln!("head_dim 64 context {context_length}: {errors}");
    }
    Ok(())
}
