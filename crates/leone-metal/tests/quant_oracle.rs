mod oracle_support;

use leone::{Backend, BackendError, BufferLayout, CpuBackend, QuantFormat, QuantMatrix};
#[cfg(target_os = "macos")]
use leone_gguf::{GgmlType, Gguf};
#[cfg(target_os = "macos")]
use std::path::PathBuf;

use oracle_support::{
    assert_close, decode_rows, matrix_dot, matrix_gemm, metal_backend, read_f32, seeded_values,
    set_f16, upload_f32, upload_u32,
};

const ROWS: usize = 3;
const COLUMNS: usize = 512;
const TOKENS: usize = 3;
const MODEL_COLUMNS: usize = 4_096;
const MODEL_ROWS: usize = 3;
const MODEL_TOKENS: [usize; 3] = [1, 3, 4];
const MODEL_QUERY_ROWS: usize = 4_096;
const MODEL_KV_ROWS: usize = 1_024;
const REDUCTION_LANES: usize = 256;
const SIMDGROUP_MATRIX_DIM: usize = 8;
const FP32_UNIT_ROUND: f64 = 5.960_464_477_539_063e-8;

#[test]
fn research_matrix_rows_match_fp64_and_individual_decode() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    assert!(!metal.verify_supported());
    assert_eq!(metal.max_batch_size().get(), 1);
    metal.set_research_batch_size();
    assert!(metal.verify_supported());
    for format in [QuantFormat::Q4K, QuantFormat::Q6K] {
        for positions in [1, 3, 8] {
            assert_research_matrix(&mut metal, format, positions);
        }
    }
}

fn assert_research_matrix(
    metal: &mut leone_metal::MetalBackend,
    format: QuantFormat,
    positions: usize,
) {
    let shape = QuantMatrix::new(MODEL_ROWS, MODEL_COLUMNS, format).expect("matrix shape");
    let bytes = quantized_matrix(format, shape.rows(), shape.columns());
    let decoded = decode_rows(format, &bytes, shape.rows(), shape.columns());
    let weights = metal.upload(shape.layout().unwrap(), &bytes).unwrap();
    let input_values = adversarial_batch(&cancellation_input(&decoded[..MODEL_COLUMNS]), positions);
    let residual_values = seeded_values(positions * shape.rows(), 0x5245_5349);
    let input = upload_f32(metal, &input_values);
    let residual = upload_f32(metal, &residual_values);
    let mut output = metal
        .allocate(BufferLayout::f32(residual_values.len()).unwrap())
        .unwrap();
    metal
        .verify_gemv(&weights, &input, &mut output, shape, positions)
        .expect("research GEMV");
    let actual = read_f32(metal, &output, residual_values.len());
    let poison = vec![91.0; residual_values.len()];
    let mut poisoned_output = upload_f32(metal, &poison);
    metal
        .verify_gemv_residual(
            &weights,
            &input,
            &residual,
            &mut poisoned_output,
            shape,
            positions,
        )
        .expect("research residual GEMV");
    let actual_residual = read_f32(metal, &poisoned_output, residual_values.len());
    for position in 0..positions {
        let start = position * shape.rows();
        let row_input = &input_values[position * MODEL_COLUMNS..(position + 1) * MODEL_COLUMNS];
        let row_residual = &residual_values[start..start + shape.rows()];
        let row_actual = &actual[start..start + shape.rows()];
        let row_actual_residual = &actual_residual[start..start + shape.rows()];
        assert_research_row_oracle(
            &decoded,
            row_input,
            row_residual,
            row_actual,
            row_actual_residual,
        );
        assert_research_row_decode(
            metal,
            &weights,
            shape,
            row_input,
            row_residual,
            row_actual,
            row_actual_residual,
        );
    }
}

fn assert_research_row_oracle(
    decoded: &[f32],
    input: &[f32],
    residual: &[f32],
    actual: &[f32],
    actual_residual: &[f32],
) {
    for (row, weights) in decoded.chunks_exact(input.len()).enumerate() {
        assert_reduction_value("research GEMV", actual[row], weights, input, None);
        assert_reduction_value(
            "research residual GEMV",
            actual_residual[row],
            weights,
            input,
            Some(residual[row]),
        );
    }
}

fn assert_research_row_decode(
    metal: &mut leone_metal::MetalBackend,
    weights: &<leone_metal::MetalBackend as Backend>::Buffer,
    shape: QuantMatrix,
    input: &[f32],
    residual: &[f32],
    actual: &[f32],
    actual_residual: &[f32],
) {
    let input = upload_f32(metal, input);
    let residual = upload_f32(metal, residual);
    let mut output = metal
        .allocate(BufferLayout::f32(shape.rows()).unwrap())
        .unwrap();
    metal.gemv(weights, &input, &mut output, shape).unwrap();
    let isolated = read_f32(metal, &output, shape.rows());
    assert_eq!(float_bits(actual), float_bits(&isolated));
    metal
        .gemv_residual(weights, &input, &residual, &mut output, shape)
        .unwrap();
    let isolated_residual = read_f32(metal, &output, shape.rows());
    assert_eq!(float_bits(actual_residual), float_bits(&isolated_residual));
}

fn float_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn research_matrix_invalid_inputs_preserve_output() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = QuantMatrix::new(ROWS, COLUMNS, QuantFormat::Q4K).unwrap();
    let weights = metal
        .upload(shape.layout().unwrap(), &q4_matrix(ROWS, COLUMNS / 256))
        .unwrap();
    let input = upload_f32(&mut metal, &seeded_values(COLUMNS, 0x494e_5054));
    let sentinel = vec![37.0; ROWS];
    let mut output = upload_f32(&mut metal, &sentinel);
    assert!(metal
        .verify_gemv(&weights, &input, &mut output, shape, 1)
        .is_err());
    metal.set_research_batch_size();
    let zero_error = metal
        .verify_gemv(&weights, &input, &mut output, shape, 0)
        .expect_err("zero verifier positions must be rejected");
    assert_eq!(
        zero_error,
        BackendError::Zero {
            field: "research positions"
        }
    );

    let over_positions = 9;
    let over_input = upload_f32(
        &mut metal,
        &seeded_values(over_positions * COLUMNS, 0x4f56_4552),
    );
    let over_sentinel = vec![37.0; over_positions * ROWS];
    let mut over_output = upload_f32(&mut metal, &over_sentinel);
    let over_error = metal
        .verify_gemv(
            &weights,
            &over_input,
            &mut over_output,
            shape,
            over_positions,
        )
        .expect_err("verifier positions above the research width must be rejected");
    assert_eq!(
        over_error,
        BackendError::Operation {
            operation: "run Metal research matrix operation",
            message: "position count exceeds the research batch limit".to_owned(),
        }
    );
    assert_eq!(
        read_f32(&mut metal, &over_output, over_positions * ROWS),
        over_sentinel
    );
    let short_residual = upload_f32(&mut metal, &[1.0]);
    assert!(metal
        .verify_gemv_residual(&weights, &input, &short_residual, &mut output, shape, 1)
        .is_err());
    let actual = read_f32(&mut metal, &output, ROWS);
    assert_eq!(actual, sentinel);
}

#[test]
fn model_width_reductions_match_fp64_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    for format in [QuantFormat::Q4K, QuantFormat::Q6K] {
        let shape = QuantMatrix::new(MODEL_ROWS, MODEL_COLUMNS, format).expect("matrix shape");
        let weights_bytes = quantized_matrix(format, MODEL_ROWS, MODEL_COLUMNS);
        let weights = metal
            .upload(shape.layout().expect("weight layout"), &weights_bytes)
            .expect("Metal weights");
        let decoded = decode_rows(format, &weights_bytes, MODEL_ROWS, MODEL_COLUMNS);
        let input_values = cancellation_input(&decoded[..MODEL_COLUMNS]);
        assert_model_width_gemv(&mut metal, &weights, shape, &decoded, &input_values);
        assert_model_width_residual(&mut metal, &weights, shape, &decoded, &input_values);
        for tokens in MODEL_TOKENS {
            assert_model_width_prefill(
                &mut metal,
                &weights,
                shape,
                &decoded,
                &input_values,
                tokens,
            );
        }
    }
}

#[test]
fn distinct_qkv_model_projection_widths_match_fp64_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    for format in [QuantFormat::Q4K, QuantFormat::Q6K] {
        let query_shape =
            QuantMatrix::new(MODEL_QUERY_ROWS, MODEL_COLUMNS, format).expect("query shape");
        let kv_shape = QuantMatrix::new(MODEL_KV_ROWS, MODEL_COLUMNS, format).expect("KV shape");
        let query_bytes = distinct_quantized_matrix(format, MODEL_QUERY_ROWS, MODEL_COLUMNS, 0x11);
        let key_bytes = distinct_quantized_matrix(format, MODEL_KV_ROWS, MODEL_COLUMNS, 0x73);
        let value_bytes = distinct_quantized_matrix(format, MODEL_KV_ROWS, MODEL_COLUMNS, 0xd9);
        let query_weights = metal
            .upload(query_shape.layout().expect("query layout"), &query_bytes)
            .expect("query weights");
        let key_weights = metal
            .upload(kv_shape.layout().expect("key layout"), &key_bytes)
            .expect("key weights");
        let value_weights = metal
            .upload(kv_shape.layout().expect("value layout"), &value_bytes)
            .expect("value weights");
        let query_decoded = decode_rows(format, &query_bytes, 1, MODEL_COLUMNS);
        let input_values = cancellation_input(&query_decoded);
        let input = upload_f32(&mut metal, &input_values);
        let mut query = metal
            .allocate(BufferLayout::f32(MODEL_QUERY_ROWS).expect("query output layout"))
            .expect("query output");
        let mut key = metal
            .allocate(BufferLayout::f32(MODEL_KV_ROWS).expect("key output layout"))
            .expect("key output");
        let mut value = metal
            .allocate(BufferLayout::f32(MODEL_KV_ROWS).expect("value output layout"))
            .expect("value output");
        metal
            .qkv_gemv(
                &query_weights,
                query_shape,
                &key_weights,
                kv_shape,
                &value_weights,
                kv_shape,
                &input,
                &mut query,
                &mut key,
                &mut value,
            )
            .expect("QKV GEMV");
        let query_actual = read_f32(&mut metal, &query, MODEL_QUERY_ROWS);
        let key_actual = read_f32(&mut metal, &key, MODEL_KV_ROWS);
        let value_actual = read_f32(&mut metal, &value, MODEL_KV_ROWS);
        assert_selected_rows(
            "Metal query projection",
            &query_actual,
            &query_bytes,
            format,
            &input_values,
            &[0, MODEL_QUERY_ROWS / 2, MODEL_QUERY_ROWS - 1],
            None,
        );
        assert_selected_rows(
            "Metal key projection",
            &key_actual,
            &key_bytes,
            format,
            &input_values,
            &[0, MODEL_KV_ROWS / 2, MODEL_KV_ROWS - 1],
            None,
        );
        assert_selected_rows(
            "Metal value projection",
            &value_actual,
            &value_bytes,
            format,
            &input_values,
            &[0, MODEL_KV_ROWS / 2, MODEL_KV_ROWS - 1],
            None,
        );
        let residual_values = seeded_values(MODEL_QUERY_ROWS, 0x5245_5349);
        let residual = upload_f32(&mut metal, &residual_values);
        let mut residual_output = metal
            .allocate(BufferLayout::f32(MODEL_QUERY_ROWS).expect("residual output layout"))
            .expect("residual output");
        metal
            .gemv_residual(
                &query_weights,
                &input,
                &residual,
                &mut residual_output,
                query_shape,
            )
            .expect("residual GEMV");
        let residual_actual = read_f32(&mut metal, &residual_output, MODEL_QUERY_ROWS);
        assert_selected_rows(
            "Metal residual projection",
            &residual_actual,
            &query_bytes,
            format,
            &input_values,
            &[0, MODEL_QUERY_ROWS - 1],
            Some(&residual_values),
        );
    }
}

fn assert_model_width_gemv(
    metal: &mut leone_metal::MetalBackend,
    weights: &<leone_metal::MetalBackend as Backend>::Buffer,
    shape: QuantMatrix,
    decoded: &[f32],
    input_values: &[f32],
) {
    let input = upload_f32(metal, input_values);
    let mut output = metal
        .allocate(BufferLayout::f32(shape.rows()).expect("GEMV output layout"))
        .expect("GEMV output");
    metal
        .gemv(weights, &input, &mut output, shape)
        .expect("model-width GEMV");
    let actual = read_f32(metal, &output, shape.rows());
    assert_selected_decoded_rows(
        "Metal model-width GEMV",
        &actual,
        decoded,
        shape.rows(),
        input_values,
        &(0..shape.rows()).collect::<Vec<_>>(),
        None,
    );
}

fn assert_model_width_residual(
    metal: &mut leone_metal::MetalBackend,
    weights: &<leone_metal::MetalBackend as Backend>::Buffer,
    shape: QuantMatrix,
    decoded: &[f32],
    input_values: &[f32],
) {
    let input = upload_f32(metal, input_values);
    let residual_values = seeded_values(shape.rows(), 0x5245_5349);
    let residual = upload_f32(metal, &residual_values);
    let mut output = metal
        .allocate(BufferLayout::f32(shape.rows()).expect("residual output layout"))
        .expect("residual output");
    metal
        .gemv_residual(weights, &input, &residual, &mut output, shape)
        .expect("model-width residual GEMV");
    let actual = read_f32(metal, &output, shape.rows());
    let rows = (0..shape.rows()).collect::<Vec<_>>();
    assert_selected_decoded_rows(
        "Metal model-width residual GEMV",
        &actual,
        decoded,
        shape.rows(),
        input_values,
        &rows,
        Some(&residual_values),
    );
}

fn assert_model_width_prefill(
    metal: &mut leone_metal::MetalBackend,
    weights: &<leone_metal::MetalBackend as Backend>::Buffer,
    shape: QuantMatrix,
    decoded: &[f32],
    base_input: &[f32],
    tokens: usize,
) {
    let input_values = adversarial_batch(base_input, tokens);
    let input = upload_f32(metal, &input_values);
    let mut output = metal
        .allocate(BufferLayout::f32(tokens * shape.rows()).expect("GEMM output layout"))
        .expect("GEMM output");
    metal
        .prefill_gemm(weights, &input, &mut output, shape, tokens)
        .expect("model-width prefill GEMM");
    let actual = read_f32(metal, &output, tokens * shape.rows());
    let rows = (0..shape.rows()).collect::<Vec<_>>();
    for token in 0..tokens {
        let token_input = &input_values[token * base_input.len()..(token + 1) * base_input.len()];
        let token_actual = &actual[token * shape.rows()..(token + 1) * shape.rows()];
        assert_selected_decoded_rows(
            "Metal model-width prefill GEMM",
            token_actual,
            decoded,
            shape.rows(),
            token_input,
            &rows,
            None,
        );
    }
}

fn assert_selected_rows(
    name: &str,
    actual: &[f32],
    weights_bytes: &[u8],
    format: QuantFormat,
    input: &[f32],
    selected_rows: &[usize],
    residual: Option<&[f32]>,
) {
    let columns = input.len();
    let row_bytes = columns / 256 * format.block_bytes();
    for &row in selected_rows {
        let start = row * row_bytes;
        let decoded = decode_rows(format, &weights_bytes[start..start + row_bytes], 1, columns);
        let residual_value = residual.map(|values| values[row]);
        assert_reduction_value(
            &format!("{name} row {row}"),
            actual[row],
            &decoded,
            input,
            residual_value,
        );
    }
}

fn assert_selected_decoded_rows(
    name: &str,
    actual: &[f32],
    decoded: &[f32],
    rows: usize,
    input: &[f32],
    selected_rows: &[usize],
    residual: Option<&[f32]>,
) {
    let columns = input.len();
    for &row in selected_rows {
        let weights = &decoded[row * columns..(row + 1) * columns];
        let residual_value = residual.map(|values| values[row]);
        assert_reduction_value(
            &format!("{name} row {row}"),
            actual[row],
            weights,
            input,
            residual_value,
        );
    }
    assert_eq!(actual.len(), rows, "{name} output rows");
}

fn assert_reduction_value(
    name: &str,
    actual: f32,
    weights: &[f32],
    input: &[f32],
    residual: Option<f32>,
) {
    assert!(actual.is_finite(), "{name} is not finite: {actual}");
    let mut expected = fp64_dot(weights, input);
    if let Some(value) = residual {
        expected += f64::from(value);
    }
    let error = (f64::from(actual) - expected).abs();
    let bound = fp32_reduction_bound(weights, input, residual.is_some());
    assert!(
        error <= bound,
        "{name} differs: actual={actual:e}, expected={expected:e}, error={error:e}, bound={bound:e}"
    );
}

fn fp64_dot(weights: &[f32], input: &[f32]) -> f64 {
    weights
        .iter()
        .zip(input)
        .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
        .sum()
}

// The bound covers FP32 products, lane sums, tree reduction, residual addition, and store rounding.
fn fp32_reduction_bound(weights: &[f32], input: &[f32], residual: bool) -> f64 {
    let terms_per_lane = weights.len().div_ceil(REDUCTION_LANES);
    let depth = 2 + terms_per_lane + REDUCTION_LANES.ilog2() as usize + usize::from(residual);
    let depth = depth as f64;
    let gamma = (depth * FP32_UNIT_ROUND) / (1.0 - depth * FP32_UNIT_ROUND);
    let absolute_terms = weights
        .iter()
        .zip(input)
        .map(|(weight, value)| f64::from(*weight).abs() * f64::from(*value).abs())
        .sum::<f64>();
    gamma * absolute_terms + f64::EPSILON
}

fn cancellation_input(weights: &[f32]) -> Vec<f32> {
    let mut input = (0..weights.len())
        .map(|index| if index.is_multiple_of(2) { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let Some((pivot, &weight)) = weights
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
    else {
        return input;
    };
    if weight == 0.0 {
        return input;
    }
    let without_pivot = weights
        .iter()
        .zip(&input)
        .enumerate()
        .filter(|(index, _)| *index != pivot)
        .map(|(_, (weight, value))| f64::from(*weight) * f64::from(*value))
        .sum::<f64>();
    let adjusted = (-without_pivot / f64::from(weight)) as f32;
    if adjusted.is_finite() {
        input[pivot] = adjusted;
    }
    input
}

fn adversarial_batch(base: &[f32], tokens: usize) -> Vec<f32> {
    (0..tokens)
        .flat_map(|token| {
            let scale = 2.0_f32.powi(token as i32);
            base.iter().map(move |value| {
                if token.is_multiple_of(2) {
                    *value * scale
                } else {
                    -*value * scale
                }
            })
        })
        .collect()
}

#[test]
fn quantized_gemv_gemm_and_embedding_match_scalar_reference() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    for (format, weights_bytes) in quantized_cases() {
        let shape = QuantMatrix::new(ROWS, COLUMNS, format).expect("matrix shape");
        let layout = shape.layout().expect("weight layout");
        let cpu_weights = cpu.upload(layout, &weights_bytes).expect("CPU weights");
        let metal_weights = metal.upload(layout, &weights_bytes).expect("Metal weights");
        let decoded = decode_rows(format, &weights_bytes, ROWS, COLUMNS);

        let input = seeded_values(COLUMNS, 0x4d45_5441);
        let cpu_input = upload_f32(&mut cpu, &input);
        let metal_input = upload_f32(&mut metal, &input);
        let mut cpu_output = cpu
            .allocate(BufferLayout::f32(ROWS).expect("CPU output layout"))
            .expect("CPU output");
        let mut metal_output = metal
            .allocate(BufferLayout::f32(ROWS).expect("Metal output layout"))
            .expect("Metal output");
        cpu.gemv(&cpu_weights, &cpu_input, &mut cpu_output, shape)
            .expect("CPU GEMV");
        metal
            .gemv(&metal_weights, &metal_input, &mut metal_output, shape)
            .expect("Metal GEMV");
        let expected = matrix_dot(&decoded, ROWS, COLUMNS, &input);
        let cpu_values = read_f32(&mut cpu, &cpu_output, ROWS);
        let metal_values = read_f32(&mut metal, &metal_output, ROWS);
        assert_close("CPU GEMV reference", &cpu_values, &expected, 1e-4, 1e-6);
        assert_close("Metal GEMV reference", &metal_values, &expected, 0.05, 3e-4);

        let gemm_input = seeded_values(TOKENS * COLUMNS, 0x4745_4d4d);
        let cpu_gemm_input = upload_f32(&mut cpu, &gemm_input);
        let metal_gemm_input = upload_f32(&mut metal, &gemm_input);
        let mut cpu_gemm_output = cpu
            .allocate(BufferLayout::f32(TOKENS * ROWS).expect("CPU GEMM output layout"))
            .expect("CPU GEMM output");
        let mut metal_gemm_output = metal
            .allocate(BufferLayout::f32(TOKENS * ROWS).expect("Metal GEMM output layout"))
            .expect("Metal GEMM output");
        cpu.prefill_gemm(
            &cpu_weights,
            &cpu_gemm_input,
            &mut cpu_gemm_output,
            shape,
            TOKENS,
        )
        .expect("CPU GEMM");
        metal
            .prefill_gemm(
                &metal_weights,
                &metal_gemm_input,
                &mut metal_gemm_output,
                shape,
                TOKENS,
            )
            .expect("Metal GEMM");
        let expected_gemm = matrix_gemm(&decoded, ROWS, COLUMNS, &gemm_input, TOKENS);
        let cpu_gemm_values = read_f32(&mut cpu, &cpu_gemm_output, TOKENS * ROWS);
        let metal_gemm_values = read_f32(&mut metal, &metal_gemm_output, TOKENS * ROWS);
        assert_close(
            "CPU GEMM reference",
            &cpu_gemm_values,
            &expected_gemm,
            1e-4,
            1e-6,
        );
        assert_close(
            "Metal GEMM reference",
            &metal_gemm_values,
            &expected_gemm,
            0.05,
            3e-4,
        );

        let residual = seeded_values(ROWS, 0x5245_5349);
        let cpu_residual = upload_f32(&mut cpu, &residual);
        let metal_residual = upload_f32(&mut metal, &residual);
        let mut cpu_residual_output = cpu
            .allocate(BufferLayout::f32(ROWS).expect("CPU residual layout"))
            .expect("CPU residual output");
        let mut metal_residual_output = metal
            .allocate(BufferLayout::f32(ROWS).expect("Metal residual layout"))
            .expect("Metal residual output");
        cpu.gemv_residual(
            &cpu_weights,
            &cpu_input,
            &cpu_residual,
            &mut cpu_residual_output,
            shape,
        )
        .expect("CPU GEMV residual");
        metal
            .gemv_residual(
                &metal_weights,
                &metal_input,
                &metal_residual,
                &mut metal_residual_output,
                shape,
            )
            .expect("Metal GEMV residual");
        let expected_residual = expected
            .iter()
            .zip(&residual)
            .map(|(value, residual)| value + residual)
            .collect::<Vec<_>>();
        let cpu_residual_values = read_f32(&mut cpu, &cpu_residual_output, ROWS);
        let metal_residual_values = read_f32(&mut metal, &metal_residual_output, ROWS);
        assert_close(
            "CPU GEMV residual reference",
            &cpu_residual_values,
            &expected_residual,
            1e-4,
            1e-6,
        );
        assert_close(
            "Metal GEMV residual reference",
            &metal_residual_values,
            &expected_residual,
            0.05,
            3e-4,
        );

        let rows = [2_u32, 0, 1];
        let single_row = [rows[0]];
        let cpu_single_row = upload_u32(&mut cpu, &single_row);
        let metal_single_row = upload_u32(&mut metal, &single_row);
        let cpu_rows = upload_u32(&mut cpu, &rows);
        let metal_rows = upload_u32(&mut metal, &rows);
        let mut cpu_embedding = cpu
            .allocate(BufferLayout::f32(COLUMNS).expect("CPU embedding layout"))
            .expect("CPU embedding output");
        let mut metal_embedding = metal
            .allocate(BufferLayout::f32(COLUMNS).expect("Metal embedding layout"))
            .expect("Metal embedding output");
        let mut cpu_batch = cpu
            .allocate(BufferLayout::f32(TOKENS * COLUMNS).expect("CPU batch layout"))
            .expect("CPU batch output");
        let mut metal_batch = metal
            .allocate(BufferLayout::f32(TOKENS * COLUMNS).expect("Metal batch layout"))
            .expect("Metal batch output");
        cpu.embed_gather(&cpu_weights, &cpu_single_row, &mut cpu_embedding, shape)
            .expect("CPU embedding gather");
        metal
            .embed_gather(
                &metal_weights,
                &metal_single_row,
                &mut metal_embedding,
                shape,
            )
            .expect("Metal embedding gather");
        cpu.embed_gather_batch(&cpu_weights, &cpu_rows, &mut cpu_batch, shape, TOKENS)
            .expect("CPU embedding batch");
        metal
            .embed_gather_batch(&metal_weights, &metal_rows, &mut metal_batch, shape, TOKENS)
            .expect("Metal embedding batch");
        let expected_embedding = decoded[COLUMNS * 2..COLUMNS * 3].to_vec();
        let expected_batch = rows
            .iter()
            .flat_map(|row| decoded[*row as usize * COLUMNS..(*row as usize + 1) * COLUMNS].iter())
            .copied()
            .collect::<Vec<_>>();
        let cpu_embedding_values = read_f32(&mut cpu, &cpu_embedding, COLUMNS);
        let metal_embedding_values = read_f32(&mut metal, &metal_embedding, COLUMNS);
        let cpu_batch_values = read_f32(&mut cpu, &cpu_batch, TOKENS * COLUMNS);
        let metal_batch_values = read_f32(&mut metal, &metal_batch, TOKENS * COLUMNS);
        assert_close(
            "CPU embedding reference",
            &cpu_embedding_values,
            &expected_embedding,
            1e-4,
            1e-6,
        );
        assert_close(
            "Metal embedding reference",
            &metal_embedding_values,
            &expected_embedding,
            1e-4,
            1e-6,
        );
        assert_close(
            "CPU embedding batch reference",
            &cpu_batch_values,
            &expected_batch,
            1e-4,
            1e-6,
        );
        assert_close(
            "Metal embedding batch reference",
            &metal_batch_values,
            &expected_batch,
            1e-4,
            1e-6,
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a Q4_K or Q6_K GGUF model and a Metal GPU"]
fn fullmodel_quantized_tensor_rows_match_scalar_reference() {
    let model = std::env::var_os("LEONE_METAL_MODEL")
        .map(PathBuf::from)
        .expect("LEONE_METAL_MODEL must name the GGUF model for this ignored gate");
    assert!(
        model.is_file(),
        "Metal model path is not a file: {}",
        model.display()
    );
    let Some(mut metal) = metal_backend() else {
        panic!("the fullmodel Metal gate requires macOS with a Metal device");
    };
    let mut cpu = CpuBackend::new();
    let gguf = Gguf::open(&model).expect("Metal model opens as GGUF");
    for name in [
        "token_embd.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
    ] {
        assert_fullmodel_tensor_rows(&gguf, name, &mut cpu, &mut metal);
    }
}

#[cfg(target_os = "macos")]
fn assert_fullmodel_tensor_rows(
    gguf: &Gguf,
    name: &str,
    cpu: &mut CpuBackend,
    metal: &mut leone_metal::MetalBackend,
) {
    let tensor = gguf
        .tensor(name)
        .unwrap_or_else(|| panic!("Metal model has no {name}"));
    let format = match tensor.dtype {
        GgmlType::Q4_K => QuantFormat::Q4K,
        GgmlType::Q6_K => QuantFormat::Q6K,
        dtype => panic!("{name} uses unsupported type {dtype}"),
    };
    let columns = usize::try_from(tensor.shape[0]).expect("tensor columns fit usize");
    let rows = tensor
        .shape
        .iter()
        .skip(1)
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .expect("tensor rows fit usize");
    assert!(rows >= 2, "{name} has too few rows");
    let shape = QuantMatrix::new(1, columns, format).expect("tensor row shape");
    let row_bytes = shape.row_bytes().expect("tensor row bytes");
    for row in [0, rows - 1] {
        let row_offset = u64::try_from(row * row_bytes).expect("row offset fits u64");
        let bytes = gguf
            .tensor_range(name, row_offset, row_bytes as u64)
            .expect("tensor row is readable");
        let decoded = decode_rows(format, &bytes, 1, columns);
        let input = seeded_values(columns, row as u32 ^ 0x4d4f_444c);
        let cpu_weights = cpu
            .upload(shape.layout().expect("CPU row layout"), &bytes)
            .expect("CPU model row");
        let metal_weights = metal
            .upload(shape.layout().expect("Metal row layout"), &bytes)
            .expect("Metal model row");
        let cpu_input = upload_f32(cpu, &input);
        let metal_input = upload_f32(metal, &input);
        let mut cpu_output = cpu
            .allocate(BufferLayout::f32(1).expect("CPU output layout"))
            .expect("CPU output");
        let mut metal_output = metal
            .allocate(BufferLayout::f32(1).expect("Metal output layout"))
            .expect("Metal output");
        cpu.gemv(&cpu_weights, &cpu_input, &mut cpu_output, shape)
            .expect("CPU model row GEMV");
        metal
            .gemv(&metal_weights, &metal_input, &mut metal_output, shape)
            .expect("Metal model row GEMV");
        let expected = matrix_dot(&decoded, 1, columns, &input);
        let cpu_values = read_f32(cpu, &cpu_output, 1);
        let metal_values = read_f32(metal, &metal_output, 1);
        assert_close(
            &format!("CPU {name} row"),
            &cpu_values,
            &expected,
            1e-4,
            1e-6,
        );
        assert_reduction_value(
            &format!("Metal {name} row"),
            metal_values[0],
            &decoded,
            &input,
            None,
        );
    }
}

#[test]
fn prefill_gemm_matches_fp64_oracle_at_token_boundaries() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    for rows in [1, 5] {
        for (format, salt) in [(QuantFormat::Q4K, 0x21), (QuantFormat::Q6K, 0x8f)] {
            for columns in [256, 512, 768, 4_096] {
                for tokens in [8, 9, 10, 11, 12, 13, 15] {
                    assert_prefill_case(&mut metal, format, rows, columns, tokens, &[], salt);
                }
            }
        }
    }
}

#[test]
fn prefill_gemm_matches_individual_gemv_at_token_boundaries() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    for (format, salt) in [(QuantFormat::Q4K, 0x21), (QuantFormat::Q6K, 0x8f)] {
        for tokens in [8, 9, 10, 11, 12, 13, 15] {
            let token_indices = individual_token_indices(tokens);
            assert_prefill_case(&mut metal, format, 5, 4_096, tokens, &token_indices, salt);
        }
    }
}

fn assert_prefill_case(
    metal: &mut leone_metal::MetalBackend,
    format: QuantFormat,
    rows: usize,
    columns: usize,
    tokens: usize,
    exact_tokens: &[usize],
    salt: u8,
) {
    assert!(
        tokens < 16,
        "the scalar reduction bound and bitwise GEMV check apply below the 16-token matrix tile"
    );
    let shape = QuantMatrix::new(rows, columns, format).expect("matrix shape");
    let bytes = distinct_quantized_matrix(format, rows, columns, salt);
    let decoded = decode_rows(format, &bytes, rows, columns);
    let weights = metal.upload(shape.layout().unwrap(), &bytes).unwrap();
    let input_values = bounded_input(&decoded[..columns], tokens);
    let input = upload_f32(metal, &input_values);
    let poison = vec![91.0; tokens * rows];
    let mut output = upload_f32(metal, &poison);
    metal
        .prefill_gemm(&weights, &input, &mut output, shape, tokens)
        .expect("prefill GEMM");
    let actual = read_f32(metal, &output, tokens * rows);
    for (token, input_row) in input_values.chunks_exact(columns).enumerate() {
        for (row, weights_row) in decoded.chunks_exact(columns).enumerate() {
            let output_index = token * rows + row;
            assert_reduction_value(
                &format!("{format:?} prefill {tokens}x{columns} row {row} token {token}"),
                actual[output_index],
                weights_row,
                input_row,
                None,
            );
        }
    }
    assert_prefill_matches_individual_gemv(
        metal,
        &weights,
        shape,
        &input_values,
        &actual,
        exact_tokens,
    );
}

fn assert_prefill_matches_individual_gemv(
    metal: &mut leone_metal::MetalBackend,
    weights: &<leone_metal::MetalBackend as Backend>::Buffer,
    shape: QuantMatrix,
    input_values: &[f32],
    actual: &[f32],
    exact_tokens: &[usize],
) {
    for &token in exact_tokens {
        let input_start = token * shape.columns();
        let token_input = &input_values[input_start..input_start + shape.columns()];
        let input = upload_f32(metal, token_input);
        let mut output = metal
            .allocate(BufferLayout::f32(shape.rows()).unwrap())
            .unwrap();
        metal
            .gemv(weights, &input, &mut output, shape)
            .expect("individual GEMV");
        let isolated = read_f32(metal, &output, shape.rows());
        let batch_start = token * shape.rows();
        assert_eq!(
            float_bits(&actual[batch_start..batch_start + shape.rows()]),
            float_bits(&isolated),
            "batched output differs from GEMV at token {token}"
        );
    }
}

fn individual_token_indices(tokens: usize) -> Vec<usize> {
    vec![0, 1, tokens - 1]
}

fn bounded_input(weights: &[f32], tokens: usize) -> Vec<f32> {
    let base = cancellation_input(weights);
    let mut values = Vec::with_capacity(tokens * weights.len());
    for token in 0..tokens {
        let sign = if token.is_multiple_of(2) { 1.0 } else { -1.0 };
        let scale = 1.0 + token as f32 / 32.0;
        for (column, value) in base.iter().enumerate() {
            let tail = input_perturbation(column, weights.len());
            values.push(sign * scale * (*value + tail));
        }
    }
    values
}

fn input_perturbation(column: usize, columns: usize) -> f32 {
    match column {
        0 => 0.125,
        255 => -0.25,
        256 if columns > 256 => 0.5,
        column if column + 1 == columns => 0.75,
        _ => 0.0,
    }
}

#[test]
#[ignore = "requires native Metal SIMD-group matrix support"]
fn simdgroup_prefill_matrix_matches_fp64_at_valid_tiles() {
    let Some(mut metal) = metal_backend() else {
        panic!("native Metal backend is required for the SIMD-group matrix oracle");
    };
    assert!(
        metal.device_info().supports_simdgroup_matrix,
        "the selected Metal device does not support the SIMD-group matrix prototype"
    );
    let cases = [
        (1, 16, 256),
        (31, 17, 512),
        (32, 31, 768),
        (33, 32, 4_096),
        (33, 33, 256),
        (33, 128, 4_096),
    ];
    for (format, salt) in [(QuantFormat::Q4K, 0x21), (QuantFormat::Q6K, 0x8f)] {
        for &(rows, tokens, columns) in &cases {
            assert_simdgroup_prefill_case(&mut metal, format, rows, columns, tokens, salt);
        }
    }
}

#[test]
#[ignore = "requires native Metal SIMD-group matrix support"]
fn simdgroup_prefill_matrix_is_bitwise_stable_for_fixed_shape() {
    let Some(mut metal) = metal_backend() else {
        panic!("native Metal backend is required for the SIMD-group matrix stability test");
    };
    assert!(
        metal.device_info().supports_simdgroup_matrix,
        "the selected Metal device does not support the SIMD-group matrix prototype"
    );
    let format = QuantFormat::Q6K;
    let rows = 33;
    let columns = 768;
    let tokens = 32;
    let shape = QuantMatrix::new(rows, columns, format).expect("matrix shape");
    let bytes = distinct_quantized_matrix(format, rows, columns, 0xa1);
    let decoded = decode_rows(format, &bytes, rows, columns);
    let input_values = bounded_input(&decoded[..columns], tokens);
    let weights = metal.upload(shape.layout().expect("weight layout"), &bytes);
    let weights = weights.expect("Metal weights");
    let mut first: Option<Vec<f32>> = None;
    for _ in 0..3 {
        let input = upload_f32(&mut metal, &input_values);
        let mut output = upload_f32(&mut metal, &vec![91.0; tokens * rows]);
        metal
            .prefill_gemm(&weights, &input, &mut output, shape, tokens)
            .expect("prefill GEMM");
        let actual = read_f32(&mut metal, &output, tokens * rows);
        if let Some(expected) = &first {
            assert_eq!(float_bits(&actual), float_bits(expected));
        } else {
            first = Some(actual);
        }
    }
}

fn assert_simdgroup_prefill_case(
    metal: &mut leone_metal::MetalBackend,
    format: QuantFormat,
    rows: usize,
    columns: usize,
    tokens: usize,
    salt: u8,
) {
    let shape = QuantMatrix::new(rows, columns, format).expect("matrix shape");
    let bytes = distinct_quantized_matrix(format, rows, columns, salt);
    let decoded = decode_rows(format, &bytes, rows, columns);
    let weights = metal.upload(shape.layout().expect("weight layout"), &bytes);
    let weights = weights.expect("Metal weights");
    let input_values = bounded_input(&decoded[..columns], tokens);
    assert!(input_values.iter().all(|value| value.is_finite()));
    let input = upload_f32(metal, &input_values);
    let mut output = upload_f32(metal, &vec![91.0; tokens * rows]);
    metal
        .prefill_gemm(&weights, &input, &mut output, shape, tokens)
        .expect("prefill GEMM");
    let actual = read_f32(metal, &output, tokens * rows);
    for (token, input_row) in input_values.chunks_exact(columns).enumerate() {
        for (row, weights_row) in decoded.chunks_exact(columns).enumerate() {
            let output_index = token * rows + row;
            assert_simdgroup_matrix_value(
                &format!("{format:?} prefill {tokens}x{columns} row {row} token {token}"),
                actual[output_index],
                weights_row,
                input_row,
            );
        }
    }
}

fn assert_simdgroup_matrix_value(name: &str, actual: f32, weights: &[f32], input: &[f32]) {
    assert!(actual.is_finite(), "{name} is not finite: {actual}");
    let expected = fp64_dot(weights, input);
    let error = (f64::from(actual) - expected).abs();
    let bound = fp32_simdgroup_matrix_bound(weights, input);
    assert!(
        error <= bound,
        "{name} differs: actual={actual:e}, expected={expected:e}, error={error:e}, bound={bound:e}"
    );
}

fn fp32_simdgroup_matrix_bound(weights: &[f32], input: &[f32]) -> f64 {
    let matrix_steps = weights.len().div_ceil(SIMDGROUP_MATRIX_DIM);
    // This envelope assumes finite inputs with FP32 product and accumulation rounding.
    // Its depth stays conservative because the matrix intrinsic's internal order is unspecified.
    let depth = 2 * weights.len() + matrix_steps + 2;
    let depth = depth as f64;
    let gamma = (depth * FP32_UNIT_ROUND) / (1.0 - depth * FP32_UNIT_ROUND);
    let absolute_terms = weights
        .iter()
        .zip(input)
        .map(|(weight, value)| f64::from(*weight).abs() * f64::from(*value).abs())
        .sum::<f64>();
    gamma * absolute_terms + f64::EPSILON
}

fn quantized_cases() -> [(QuantFormat, Vec<u8>); 2] {
    [
        (QuantFormat::Q4K, q4_matrix(ROWS, COLUMNS / 256)),
        (QuantFormat::Q6K, q6_matrix(ROWS, COLUMNS / 256)),
    ]
}

fn quantized_matrix(format: QuantFormat, rows: usize, columns: usize) -> Vec<u8> {
    assert!(columns.is_multiple_of(256));
    match format {
        QuantFormat::Q4K => q4_matrix(rows, columns / 256),
        QuantFormat::Q6K => q6_matrix(rows, columns / 256),
    }
}

fn distinct_quantized_matrix(
    format: QuantFormat,
    rows: usize,
    columns: usize,
    salt: u8,
) -> Vec<u8> {
    let mut matrix = quantized_matrix(format, rows, columns);
    for (block_index, block) in matrix.chunks_exact_mut(format.block_bytes()).enumerate() {
        let offset = salt.wrapping_add(block_index as u8);
        let payload = match format {
            QuantFormat::Q4K => &mut block[16..],
            QuantFormat::Q6K => &mut block[..192],
        };
        for byte in payload {
            *byte = byte.wrapping_add(offset);
        }
    }
    matrix
}

fn q4_matrix(rows: usize, blocks_per_row: usize) -> Vec<u8> {
    let row_bytes = blocks_per_row * 144;
    let mut matrix = vec![0_u8; rows * row_bytes];
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let start = row * row_bytes + block * 144;
            set_f16(
                &mut matrix[start..start + 144],
                0,
                0.25 + row as f32 * 0.03125,
            );
            set_f16(
                &mut matrix[start..start + 144],
                2,
                0.125 + block as f32 * 0.015625,
            );
            let scales = [
                0xc1, 0x82, 0xe3, 0x44, 0xf5, 0xa6, 0xd7, 0x68, 0xb9, 0xca, 0xeb, 0x9c,
            ];
            matrix[start + 4..start + 16].copy_from_slice(&scales);
            for index in 0..128 {
                let low = (index * 7 + row * 3 + block) as u8 & 0x0f;
                let high = (index * 11 + row * 5 + block * 3 + 1) as u8 & 0x0f;
                matrix[start + 16 + index] = low | (high << 4);
            }
        }
    }
    matrix
}

fn q6_matrix(rows: usize, blocks_per_row: usize) -> Vec<u8> {
    let row_bytes = blocks_per_row * 210;
    let mut matrix = vec![0_u8; rows * row_bytes];
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let start = row * row_bytes + block * 210;
            for index in 0..128 {
                matrix[start + index] = (index * 29 + row * 17 + block * 7 + 3) as u8;
            }
            for index in 0..64 {
                matrix[start + 128 + index] = (index * 53 + row * 31 + block * 19 + 0xa5) as u8;
            }
            let scales = [
                0x81, 0x7f, 0x82, 0x03, 0xfe, 0x05, 0x86, 0x09, 0x0a, 0xf0, 0x0c, 0x8d, 0x10, 0x91,
                0x12, 0x13,
            ];
            matrix[start + 192..start + 208].copy_from_slice(&scales);
            set_f16(
                &mut matrix[start..start + 210],
                208,
                0.125 + row as f32 * 0.015625,
            );
        }
    }
    matrix
}
