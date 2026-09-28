mod oracle_support;
mod warm_prefill_support;

use leone::{AttentionShape, Backend, Position, RopePairing, RopeShape, VectorShape};
use oracle_support::{metal_backend, read_f32, upload_f32};
use warm_prefill_support::{
    allocate_f16_cache, f32_bits, finite_values, prepare_decode_equivalent, read_f16_bits,
};

#[test]
fn decode_equivalent_qk_prefill_preserves_f16_kv_bits() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    assert!(metal.decode_equivalent_prefill_supported());

    for pairing in [RopePairing::HalfSplit, RopePairing::Adjacent] {
        for start_position in [2, 8_193] {
            for head_dim in [64, 128] {
                assert_f16_kv_case(&mut metal, pairing, start_position, head_dim);
            }
        }
    }
}

fn assert_f16_kv_case(
    metal: &mut leone_metal::MetalBackend,
    pairing: RopePairing,
    start_position: usize,
    head_dim: usize,
) {
    let tokens = 3;
    let heads = 2;
    let context = start_position + tokens + 2;
    let vector_shape = VectorShape::new(tokens * heads, head_dim).expect("key vector shape");
    let rope_shape = RopeShape::new(tokens, heads, head_dim).expect("key RoPE shape");
    let attention_shape =
        AttentionShape::new(heads, heads, head_dim, context).expect("KV attention shape");
    let projected = attention_shape
        .projected_kv_elements()
        .expect("projected KV elements");
    let input_values = finite_values(vector_shape.elements().unwrap(), 0x4b45_5951);
    let weights = finite_values(head_dim, 0x4b45_5754)
        .into_iter()
        .map(|value| value.abs() + 0.25)
        .collect::<Vec<_>>();
    let values = finite_values(tokens * projected, 0x5641_4c55)
        .into_iter()
        .map(|value| value / 8.0)
        .collect::<Vec<_>>();
    let factors = (0..head_dim / 2)
        .map(|index| 0.75 + (index % 7) as f32 * 0.125)
        .collect::<Vec<_>>();
    let theta = 10_000.0;
    let epsilon = 1e-5;
    metal
        .configure_rope(head_dim, theta, Some(&factors), pairing)
        .expect("configure key RoPE");
    prepare_decode_equivalent(
        metal,
        tokens,
        context,
        heads,
        heads,
        head_dim,
        heads * head_dim,
        heads * head_dim,
        heads * head_dim,
    );

    let input = upload_f32(metal, &input_values);
    let weight = upload_f32(metal, &weights);
    let mut batched_key = upload_f32(
        metal,
        &vec![f32::from_bits(0x7fc0_1278); input_values.len()],
    );
    metal
        .prefill_rms_norm_rope(
            &input,
            &weight,
            &mut batched_key,
            vector_shape,
            rope_shape,
            start_position,
            epsilon,
            theta,
        )
        .expect("decode-equivalent key prefill");
    let actual_key = read_f32(metal, &batched_key, input_values.len());

    let mut control_key = Vec::with_capacity(actual_key.len());
    for token in 0..tokens {
        let token_input = upload_f32(
            metal,
            &input_values[token * heads * head_dim..(token + 1) * heads * head_dim],
        );
        let mut token_output = metal
            .allocate(leone::BufferLayout::f32(heads * head_dim).expect("key decode output layout"))
            .expect("key decode output");
        metal
            .rms_norm_rope(
                &token_input,
                &weight,
                &mut token_output,
                VectorShape::new(heads, head_dim).expect("key decode shape"),
                Position::Host(start_position + token),
                epsilon,
                theta,
            )
            .expect("fused decode key RMSNorm RoPE");
        control_key.extend(read_f32(metal, &token_output, heads * head_dim));
    }
    assert_eq!(f32_bits(&actual_key), f32_bits(&control_key));

    let mut actual_key_cache = allocate_f16_cache(metal, attention_shape.cache_elements().unwrap());
    let mut actual_value_cache =
        allocate_f16_cache(metal, attention_shape.cache_elements().unwrap());
    let mut control_key_cache =
        allocate_f16_cache(metal, attention_shape.cache_elements().unwrap());
    let mut control_value_cache =
        allocate_f16_cache(metal, attention_shape.cache_elements().unwrap());
    let value_input = upload_f32(metal, &values);
    metal
        .kv_append_chunk(
            &batched_key,
            &value_input,
            &mut actual_key_cache,
            &mut actual_value_cache,
            attention_shape,
            start_position,
            tokens,
        )
        .expect("batched F16 KV append");
    for token in 0..tokens {
        let key_row = upload_f32(
            metal,
            &control_key[token * projected..(token + 1) * projected],
        );
        let value_row = upload_f32(metal, &values[token * projected..(token + 1) * projected]);
        metal
            .kv_append(
                &key_row,
                &value_row,
                &mut control_key_cache,
                &mut control_value_cache,
                attention_shape,
                Position::Host(start_position + token),
            )
            .expect("repeated F16 KV append");
    }
    assert_eq!(
        read_f16_bits(
            metal,
            &actual_key_cache,
            attention_shape.cache_elements().unwrap()
        ),
        read_f16_bits(
            metal,
            &control_key_cache,
            attention_shape.cache_elements().unwrap()
        ),
        "F16 key cache differs for {pairing:?}, start {start_position}, head_dim {head_dim}"
    );
    assert_eq!(
        read_f16_bits(
            metal,
            &actual_value_cache,
            attention_shape.cache_elements().unwrap()
        ),
        read_f16_bits(
            metal,
            &control_value_cache,
            attention_shape.cache_elements().unwrap()
        ),
        "F16 value cache differs for {pairing:?}, start {start_position}, head_dim {head_dim}"
    );
}
