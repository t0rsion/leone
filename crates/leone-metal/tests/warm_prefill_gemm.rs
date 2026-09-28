mod oracle_support;
mod warm_prefill_support;

use leone::{Backend, BufferLayout, QuantFormat, QuantMatrix};
use oracle_support::{decode_rows, metal_backend, read_f32, upload_f32};
use warm_prefill_support::{
    f32_bits, f32_gemm_error_bound, f64_gemm, f64_gemm_abs, finite_values,
    prepare_decode_equivalent, quantized_matrix,
};

#[test]
fn decode_equivalent_gemm_matches_repeated_gemv_at_production_tail_sizes() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    assert!(metal.decode_equivalent_prefill_supported());

    let cases = [
        (256, 128),
        (256, 129),
        (512, 128),
        (512, 129),
        (2_048, 7),
        (2_048, 8),
        (2_048, 9),
        (2_048, 15),
        (2_048, 16),
        (2_048, 17),
        (2_048, 22),
        (4_096, 127),
        (4_096, 128),
        (4_096, 129),
        (8_192, 127),
        (8_192, 128),
        (8_192, 129),
    ];
    for format in [QuantFormat::Q4K, QuantFormat::Q6K] {
        for &(columns, tokens) in &cases {
            assert_gemm_case(&mut metal, format, columns, tokens);
        }
    }
}

fn assert_gemm_case(
    metal: &mut leone_metal::MetalBackend,
    format: QuantFormat,
    columns: usize,
    tokens: usize,
) {
    let rows = 3;
    let shape = QuantMatrix::new(rows, columns, format).expect("GEMM shape");
    let bytes = quantized_matrix(format, rows, columns, 0x41 ^ tokens as u8);
    let decoded = decode_rows(format, &bytes, rows, columns);
    let inputs = finite_values(tokens * columns, 0x4745_4d56 ^ columns as u32);
    let weights = metal
        .upload(shape.layout().expect("weight layout"), &bytes)
        .expect("quantized weights");
    let input = upload_f32(metal, &inputs);
    let mut output = upload_f32(metal, &vec![f32::from_bits(0x7fc0_1234); tokens * rows]);

    let heads = columns / 64;
    prepare_decode_equivalent(metal, tokens, 512, heads, heads, 64, columns, rows, rows);
    metal
        .prefill_gemm(&weights, &input, &mut output, shape, tokens)
        .expect("decode-equivalent prefill GEMM");
    let actual = read_f32(metal, &output, tokens * rows);
    let expected = f64_gemm(&decoded, rows, columns, &inputs);
    let abs_products = f64_gemm_abs(&decoded, rows, columns, &inputs);

    let mut repeated = Vec::with_capacity(actual.len());
    for token in 0..tokens {
        let token_input = upload_f32(metal, &inputs[token * columns..(token + 1) * columns]);
        let mut token_output = metal
            .allocate(BufferLayout::f32(rows).expect("GEMV output layout"))
            .expect("GEMV output");
        metal
            .gemv(&weights, &token_input, &mut token_output, shape)
            .expect("repeated GEMV");
        repeated.extend(read_f32(metal, &token_output, rows));
    }
    assert_eq!(
        f32_bits(&actual),
        f32_bits(&repeated),
        "{format:?} {columns}-column prefill differs from repeated GEMV at {tokens} tokens"
    );
    assert_fp64_reference(format, columns, tokens, &actual, &expected, &abs_products);
}

fn assert_fp64_reference(
    format: QuantFormat,
    columns: usize,
    tokens: usize,
    actual: &[f32],
    expected: &[f64],
    abs_products: &[f64],
) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), abs_products.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{format:?} output {index} is not finite"
        );
        let error = (f64::from(actual) - expected).abs();
        let bound = f32_gemm_error_bound(abs_products[index], columns);
        assert!(
            error <= bound,
            "{format:?} {columns}-column {tokens}-token output {index} differs: actual={actual:e}, expected={expected:e}, error={error:e}, sum_abs={sum_abs:e}, bound={bound:e}",
            sum_abs = abs_products[index]
        );
    }
}
