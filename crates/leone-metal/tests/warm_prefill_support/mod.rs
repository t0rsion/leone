#![allow(dead_code)]

use half::f16;
use leone::{
    AttentionShape, Backend, BufferLayout, PrefillNumerics, PrefillPlan, QuantFormat, RopePairing,
};
use leone_metal::MetalBackend;

#[allow(clippy::too_many_arguments)]
pub fn prepare_decode_equivalent(
    backend: &mut MetalBackend,
    chunk_tokens: usize,
    context_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_embd: usize,
    n_ff: usize,
    max_matrix_rows: usize,
) {
    let plan = PrefillPlan::new(
        chunk_tokens,
        context_tokens,
        n_head,
        n_head_kv,
        head_dim,
        n_embd,
        n_ff,
        max_matrix_rows,
    )
    .expect("decode-equivalent plan")
    .with_numerics(PrefillNumerics::DecodeEquivalent);
    assert_eq!(plan.numerics(), PrefillNumerics::DecodeEquivalent);
    backend
        .prepare_prefill(plan)
        .expect("prepare decode-equivalent prefill");
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_backend_preferred(
    backend: &mut MetalBackend,
    chunk_tokens: usize,
    context_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_embd: usize,
    n_ff: usize,
    max_matrix_rows: usize,
) {
    let plan = PrefillPlan::new(
        chunk_tokens,
        context_tokens,
        n_head,
        n_head_kv,
        head_dim,
        n_embd,
        n_ff,
        max_matrix_rows,
    )
    .expect("backend-preferred plan");
    assert_eq!(plan.numerics(), PrefillNumerics::BackendPreferred);
    backend
        .prepare_prefill(plan)
        .expect("prepare backend-preferred prefill");
}

pub fn quantized_matrix(format: QuantFormat, rows: usize, columns: usize, salt: u8) -> Vec<u8> {
    assert!(columns.is_multiple_of(256));
    let row_bytes = columns / 256 * format.block_bytes();
    let mut matrix = vec![0_u8; rows * row_bytes];
    for (index, byte) in matrix.iter_mut().enumerate() {
        let block = index / format.block_bytes();
        *byte = salt
            .wrapping_add((index as u8).wrapping_mul(29))
            .wrapping_add(block as u8);
    }
    for block in matrix.chunks_exact_mut(format.block_bytes()) {
        match format {
            QuantFormat::Q4K => {
                block[0..2].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&f16::from_f32(0.125).to_bits().to_le_bytes());
            }
            QuantFormat::Q6K => {
                block[208..210].copy_from_slice(&f16::from_f32(0.125).to_bits().to_le_bytes());
            }
        }
    }
    matrix
}

pub fn f64_gemm(decoded: &[f32], rows: usize, columns: usize, input: &[f32]) -> Vec<f64> {
    assert_eq!(decoded.len(), rows * columns);
    assert_eq!(input.len() % columns, 0);
    input
        .chunks_exact(columns)
        .flat_map(|row| {
            decoded.chunks_exact(columns).map(move |weights| {
                weights
                    .iter()
                    .zip(row)
                    .fold(0.0_f64, |sum, (weight, value)| {
                        sum + f64::from(*weight) * f64::from(*value)
                    })
            })
        })
        .collect()
}

pub fn f64_gemm_abs(decoded: &[f32], rows: usize, columns: usize, input: &[f32]) -> Vec<f64> {
    assert_eq!(decoded.len(), rows * columns);
    assert_eq!(input.len() % columns, 0);
    input
        .chunks_exact(columns)
        .flat_map(|row| {
            decoded.chunks_exact(columns).map(move |weights| {
                weights
                    .iter()
                    .zip(row)
                    .map(|(weight, value)| (f64::from(*weight) * f64::from(*value)).abs())
                    .sum()
            })
        })
        .collect()
}

pub fn f32_gemm_error_bound(abs_products: f64, columns: usize) -> f64 {
    // TokenTile rounds one product and one accumulation per lane term, then eight tree levels.
    let lane_terms = columns.div_ceil(256);
    let operations = 2 * lane_terms + 8;
    let unit_roundoff = f64::from(f32::EPSILON) * 0.5;
    let order = operations as f64 * unit_roundoff;
    abs_products * (order / (1.0 - order))
}

pub fn finite_values(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let exponent = (index % 17) as i32 - 8;
            let magnitude = 2.0_f32.powi(exponent) * (1.0 + (index % 7) as f32 * 0.03125);
            let sign = if (index ^ seed as usize).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            sign * magnitude
        })
        .collect()
}

pub fn attention_values(len: usize, seed: u32) -> Vec<f32> {
    finite_values(len, seed)
        .into_iter()
        .map(|value| value / 8.0)
        .collect()
}

pub fn attention_stress_values(len: usize, seed: u32) -> Vec<f32> {
    const MAGNITUDES: [f32; 8] = [
        65_504.0,
        65_504.0,
        f32::from_bits(0x3880_0000),
        f32::from_bits(0x3880_0000),
        f32::from_bits(0x3380_0000),
        f32::from_bits(0x3380_0000),
        0.5,
        0.5,
    ];
    (0..len)
        .map(|index| {
            let phase = (index / MAGNITUDES.len() + seed as usize) % 7;
            let magnitude = MAGNITUDES[(index % MAGNITUDES.len() + phase) % MAGNITUDES.len()];
            let sign = if ((index / 2 + index / MAGNITUDES.len()) ^ seed as usize).is_multiple_of(2)
            {
                1.0
            } else {
                -1.0
            };
            sign * magnitude
        })
        .collect()
}

pub fn attention_stress_query_values(len: usize, seed: u32) -> Vec<f32> {
    attention_values(len, seed)
        .into_iter()
        .map(|value| value * 1.0e-8)
        .collect()
}

pub fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub fn f64_attention_oracle(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Vec<f32> {
    let query_elements = shape.query_elements().expect("attention query elements");
    let mut output = Vec::with_capacity(tokens * query_elements);
    for token in 0..tokens {
        let context = start_position + token + 1;
        for head in 0..shape.n_head() {
            let kv_head = head / (shape.n_head() / shape.n_head_kv());
            let query_start = token * query_elements + head * shape.head_dim();
            let scores = (0..context)
                .map(|position| {
                    let key_start = (kv_head * shape.max_context() + position) * shape.head_dim();
                    (0..shape.head_dim())
                        .map(|column| {
                            f64::from(query[query_start + column])
                                * f64::from(keys[key_start + column])
                        })
                        .sum::<f64>()
                        / (shape.head_dim() as f64).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f64>();
            for column in 0..shape.head_dim() {
                let value = (0..context)
                    .map(|position| {
                        let index =
                            (kv_head * shape.max_context() + position) * shape.head_dim() + column;
                        weights[position] * f64::from(values[index])
                    })
                    .sum::<f64>()
                    / denominator;
                output.push(value as f32);
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub fn f64_rms_norm_rope(
    input: &[f32],
    weight: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    start_position: usize,
    epsilon: f32,
    theta: f32,
    factors: &[f32],
    pairing: RopePairing,
) -> Vec<f32> {
    let half = head_dim / 2;
    assert_eq!(input.len(), tokens * heads * head_dim);
    assert_eq!(weight.len(), head_dim);
    assert_eq!(factors.len(), half);
    let mut output = vec![0.0; input.len()];
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            let row = &input[base..base + head_dim];
            let square_sum = row.iter().fold(0.0_f64, |sum, value| {
                sum + f64::from(*value) * f64::from(*value)
            });
            let inverse = (square_sum / head_dim as f64 + f64::from(epsilon))
                .sqrt()
                .recip();
            for column in 0..head_dim {
                output[base + column] =
                    (f64::from(row[column]) * f64::from(weight[column]) * inverse) as f32;
            }
            let position = start_position + token;
            for (pair, factor) in factors.iter().enumerate() {
                // Metal stores f32 frequencies and forms the angle in f32.
                let exponent = -2.0_f32 * pair as f32 / head_dim as f32;
                let frequency = theta.powf(exponent) / *factor;
                let angle = position as f32 * frequency;
                let (sine, cosine) = f64::from(angle).sin_cos();
                let (first, second) = match pairing {
                    RopePairing::HalfSplit => (base + pair, base + pair + half),
                    RopePairing::Adjacent => (base + pair * 2, base + pair * 2 + 1),
                };
                let first_value = f64::from(output[first]);
                let second_value = f64::from(output[second]);
                output[first] = (first_value * cosine - second_value * sine) as f32;
                output[second] = (first_value * sine + second_value * cosine) as f32;
            }
        }
    }
    output
}

pub fn read_f16_bits(
    backend: &mut MetalBackend,
    buffer: &<MetalBackend as Backend>::Buffer,
    len: usize,
) -> Vec<u16> {
    let mut values = vec![0_u16; len];
    backend
        .read_f16(buffer, &mut values)
        .expect("read F16 bits");
    values
}

pub fn allocate_f16_cache(
    backend: &mut MetalBackend,
    elements: usize,
) -> <MetalBackend as Backend>::Buffer {
    allocate_f16_cache_with_value(backend, elements, 0.0)
}

pub fn allocate_f16_cache_with_value(
    backend: &mut MetalBackend,
    elements: usize,
    value: f32,
) -> <MetalBackend as Backend>::Buffer {
    let layout = BufferLayout::f16(elements).expect("F16 cache layout");
    let bits = f16::from_f32(value).to_bits().to_le_bytes();
    let bytes = bits.repeat(elements);
    backend.upload(layout, &bytes).expect("F16 cache")
}
