mod oracle_support;
mod warm_prefill_support;

use leone::{Backend, BufferLayout, Position, RopePairing, RopeShape, VectorShape};
use oracle_support::{metal_backend, read_f32, upload_f32};
use warm_prefill_support::{f32_bits, f64_rms_norm_rope, finite_values, prepare_decode_equivalent};

#[test]
fn decode_equivalent_qk_prefill_matches_fused_decode_norm_rope() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    assert!(metal.decode_equivalent_prefill_supported());

    for pairing in [RopePairing::HalfSplit, RopePairing::Adjacent] {
        for start_position in [3, 8_193] {
            for head_dim in [64, 128] {
                for heads in [1, 2] {
                    assert_qk_case(&mut metal, pairing, start_position, head_dim, heads);
                }
            }
        }
    }
}

fn assert_qk_case(
    metal: &mut leone_metal::MetalBackend,
    pairing: RopePairing,
    start_position: usize,
    head_dim: usize,
    heads: usize,
) {
    let tokens = 3;
    let shape = VectorShape::new(tokens * heads, head_dim).expect("QK vector shape");
    let rope_shape = RopeShape::new(tokens, heads, head_dim).expect("QK RoPE shape");
    let input_values = finite_values(shape.elements().expect("QK elements"), 0x514b_4e4f);
    let weights = finite_values(head_dim, 0x5745_4947)
        .into_iter()
        .map(|value| value.abs() + 0.25)
        .collect::<Vec<_>>();
    let factors = (0..head_dim / 2)
        .map(|index| 0.75 + (index % 7) as f32 * 0.125)
        .collect::<Vec<_>>();
    let theta = 10_000.0;
    let epsilon = 1e-5;
    metal
        .configure_rope(head_dim, theta, Some(&factors), pairing)
        .expect("configure QK RoPE");
    prepare_decode_equivalent(
        metal,
        tokens,
        start_position + tokens + 1,
        heads,
        heads,
        head_dim,
        heads * head_dim,
        heads * head_dim,
        heads * head_dim,
    );

    let input = upload_f32(metal, &input_values);
    let weight = upload_f32(metal, &weights);
    let mut batched = upload_f32(
        metal,
        &vec![f32::from_bits(0x7fc0_1256); shape.elements().unwrap()],
    );
    metal
        .prefill_rms_norm_rope(
            &input,
            &weight,
            &mut batched,
            shape,
            rope_shape,
            start_position,
            epsilon,
            theta,
        )
        .expect("decode-equivalent QK prefill");
    let actual = read_f32(metal, &batched, shape.elements().unwrap());
    let mut repeated = Vec::with_capacity(actual.len());
    let row_elements = heads * head_dim;
    for token in 0..tokens {
        let token_input = upload_f32(
            metal,
            &input_values[token * row_elements..(token + 1) * row_elements],
        );
        let mut token_output = metal
            .allocate(BufferLayout::f32(row_elements).expect("QK decode output layout"))
            .expect("QK decode output");
        metal
            .rms_norm_rope(
                &token_input,
                &weight,
                &mut token_output,
                VectorShape::new(heads, head_dim).expect("QK decode shape"),
                Position::Host(start_position + token),
                epsilon,
                theta,
            )
            .expect("fused decode RMSNorm RoPE");
        repeated.extend(read_f32(metal, &token_output, row_elements));
    }
    assert_eq!(
        f32_bits(&actual),
        f32_bits(&repeated),
        "QK output differs for {pairing:?}, start {start_position}, head_dim {head_dim}, heads {heads}"
    );

    let expected = f64_rms_norm_rope(
        &input_values,
        &weights,
        tokens,
        heads,
        head_dim,
        start_position,
        epsilon,
        theta,
        &factors,
        pairing,
    );
    assert_fp64_qk_reference(&actual, &expected, pairing, start_position, head_dim, heads);
}

fn assert_fp64_qk_reference(
    actual: &[f32],
    expected: &[f32],
    pairing: RopePairing,
    start_position: usize,
    head_dim: usize,
    heads: usize,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let bound = 5e-3 + 3e-3 * expected.abs();
        assert!(
            actual.is_finite() && error <= bound,
            "QK FP64 oracle differs for {pairing:?}, start {start_position}, head_dim {head_dim}, heads {heads}, index {index}: actual={actual:e}, expected={expected:e}, error={error:e}, bound={bound:e}"
        );
    }
}
