//! Defines the lossless Q4_K layout used by the CUDA decode path.
//!
//! Each GGUF superblock contains 16 metadata bytes followed by 128 code bytes.
//! The decode layout stores every code region first, then every metadata region.
//! Within one code region, four bytes from positions `0..16` sit beside the
//! matching four bytes from positions `16..32`. One aligned 8-byte load supplies
//! both `dp4a` chain words for that lane.
//! Four 24-bit metadata records hold the two scales and two minima for each
//! 64-value group. The records contain the same 6-bit integers as GGUF.
//!
//! The repack does not change a code, scale, minimum, `d`, or `dmin` value. The
//! scalar decoder multiplies `d * scale` and `dmin * minimum` in `f32`, then
//! applies the code with the same expression order as the GGUF decoder. Derived
//! scale products are not stored because narrowing them would change values.
//! The CPU backend can keep GGUF storage because backend buffers are opaque.

use crate::{Error, Result};
use half::f16;

pub const Q4_K_BLOCK_ELEMENTS: usize = 256;
pub const Q4_K_GGUF_BLOCK_BYTES: usize = 144;
pub const Q4_K_CODE_BYTES: usize = 128;
pub const Q4_K_METADATA_BYTES: usize = 16;

/// Reorders complete GGUF Q4_K blocks into the CUDA decode layout.
pub fn repack_q4_k(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() {
        return Err(Error::Zero {
            field: "Q4_K repack bytes",
        });
    }
    if !bytes.len().is_multiple_of(Q4_K_GGUF_BLOCK_BYTES) {
        return Err(Error::NotDivisible {
            field: "Q4_K repack bytes",
            value: bytes.len(),
            divisor: Q4_K_GGUF_BLOCK_BYTES,
        });
    }
    let blocks = bytes.len() / Q4_K_GGUF_BLOCK_BYTES;
    let codes_bytes = blocks
        .checked_mul(Q4_K_CODE_BYTES)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K repacked code bytes",
        })?;
    let mut output = vec![0_u8; bytes.len()];
    for (block_index, source) in bytes.chunks_exact(Q4_K_GGUF_BLOCK_BYTES).enumerate() {
        let code_target =
            &mut output[block_index * Q4_K_CODE_BYTES..(block_index + 1) * Q4_K_CODE_BYTES];
        for group in 0..4 {
            let source_group = &source[Q4_K_METADATA_BYTES + group * 32..];
            let target_group = &mut code_target[group * 32..];
            for lane_item in 0..4 {
                let source_offset = lane_item * 4;
                let target_offset = lane_item * 8;
                target_group[target_offset..target_offset + 4]
                    .copy_from_slice(&source_group[source_offset..source_offset + 4]);
                target_group[target_offset + 4..target_offset + 8]
                    .copy_from_slice(&source_group[source_offset + 16..source_offset + 20]);
            }
        }
        let metadata_target = codes_bytes + block_index * Q4_K_METADATA_BYTES;
        let target = &mut output[metadata_target..metadata_target + Q4_K_METADATA_BYTES];
        target[..4].copy_from_slice(&source[..4]);
        for group in 0..4 {
            let (scale0, minimum0) = scale_min(group * 2, &source[4..16]);
            let (scale1, minimum1) = scale_min(group * 2 + 1, &source[4..16]);
            let parameters = u32::from(scale0)
                | (u32::from(scale1) << 6)
                | (u32::from(minimum0) << 12)
                | (u32::from(minimum1) << 18);
            target[4 + group] = parameters as u8;
            target[8 + group] = (parameters >> 8) as u8;
            target[12 + group] = (parameters >> 16) as u8;
        }
    }
    Ok(output)
}

/// Dequantizes complete blocks from the CUDA Q4_K decode layout.
pub fn dequantize_q4_k(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    if !n_elems.is_multiple_of(Q4_K_BLOCK_ELEMENTS) {
        return Err(Error::NotDivisible {
            field: "Q4_K dequantized elements",
            value: n_elems,
            divisor: Q4_K_BLOCK_ELEMENTS,
        });
    }
    let blocks = n_elems / Q4_K_BLOCK_ELEMENTS;
    let expected = blocks
        .checked_mul(Q4_K_GGUF_BLOCK_BYTES)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K repacked bytes",
        })?;
    if bytes.len() != expected {
        return Err(Error::SizeMismatch {
            name: "Q4_K repacked bytes",
            expected,
            actual: bytes.len(),
        });
    }
    let codes_bytes = blocks
        .checked_mul(Q4_K_CODE_BYTES)
        .ok_or(Error::SizeOverflow {
            field: "Q4_K repacked code bytes",
        })?;
    let (codes, metadata) = bytes.split_at(codes_bytes);
    let mut output = Vec::with_capacity(n_elems);
    for block in 0..blocks {
        dequantize_block(
            &codes[block * Q4_K_CODE_BYTES..(block + 1) * Q4_K_CODE_BYTES],
            &metadata[block * Q4_K_METADATA_BYTES..(block + 1) * Q4_K_METADATA_BYTES],
            &mut output,
        );
    }
    Ok(output)
}

fn dequantize_block(codes: &[u8], metadata: &[u8], output: &mut Vec<f32>) {
    let d = half(metadata, 0);
    let dmin = half(metadata, 2);
    for group in 0..4 {
        let parameters = group_parameters(metadata, group);
        let scale0 = (parameters & 63) as u8;
        let scale1 = ((parameters >> 6) & 63) as u8;
        let min0 = ((parameters >> 12) & 63) as u8;
        let min1 = ((parameters >> 18) & 63) as u8;
        let d0 = d * f32::from(scale0);
        let d1 = d * f32::from(scale1);
        let m0 = dmin * f32::from(min0);
        let m1 = dmin * f32::from(min1);
        let group_codes = &codes[group * 32..group * 32 + 32];
        for original_index in 0..32 {
            let quant = group_codes[striped_index(original_index)];
            output.push(d0 * f32::from(quant & 0x0f) - m0);
        }
        for original_index in 0..32 {
            let quant = group_codes[striped_index(original_index)];
            output.push(d1 * f32::from(quant >> 4) - m1);
        }
    }
}

const fn striped_index(original_index: usize) -> usize {
    let half = original_index / 16;
    let within = original_index % 16;
    (within / 4) * 8 + half * 4 + within % 4
}

fn half(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn group_parameters(metadata: &[u8], group: usize) -> u32 {
    u32::from(metadata[4 + group])
        | (u32::from(metadata[8 + group]) << 8)
        | (u32::from(metadata[12 + group]) << 16)
}

fn scale_min(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 63, packed[index + 4] & 63)
    } else {
        (
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leone_gguf::ref_dequant;
    use leone_gguf::{GgmlType, Gguf};
    use rand::rngs::SmallRng;
    use rand::{Rng, RngCore, SeedableRng};
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    #[test]
    fn exhaustive_edge_pattern_blocks_are_bit_identical() {
        let mut base = base_block();
        for code in 0_u8..=u8::MAX {
            base[16..].fill(code);
            assert_block_equivalent(&base);
        }

        base = base_block();
        for metadata_index in 4..16 {
            for code in 0_u8..=u8::MAX {
                base[metadata_index] = code;
                assert_block_equivalent(&base);
            }
            base[metadata_index] = base_block()[metadata_index];
        }

        let finite_half_edges = [
            0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff,
        ];
        for d in finite_half_edges {
            for dmin in finite_half_edges {
                let mut block = base_block();
                block[0..2].copy_from_slice(&d.to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_le_bytes());
                assert_block_equivalent(&block);
            }
        }
    }

    #[test]
    fn one_thousand_seeded_random_blocks_are_bit_identical() {
        let mut rng = SmallRng::seed_from_u64(0x7134_6b5f_7265_706b);
        for _ in 0..1_000 {
            let mut block = vec![0_u8; Q4_K_GGUF_BLOCK_BYTES];
            rng.fill_bytes(&mut block);
            set_half(&mut block, 0, rng.random_range(-4.0_f32..4.0));
            set_half(&mut block, 2, rng.random_range(-4.0_f32..4.0));
            assert_block_equivalent(&block);
        }
    }

    #[test]
    #[ignore = "requires the Qwen3 Q4_K_M model"]
    fn five_named_real_tensors_have_identical_dequantized_hashes() {
        let gguf = Gguf::open(model_path()).expect("model opens");
        let names = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
        ];
        for name in names {
            let source = gguf.tensor_data(name).expect("tensor bytes are readable");
            assert_eq!(
                gguf.tensor(name).expect("tensor exists").dtype,
                GgmlType::Q4_K,
                "tensor {name}"
            );
            let repacked = repack_q4_k(&source).expect("tensor repacks");
            let blocks = source.len() / Q4_K_GGUF_BLOCK_BYTES;
            let codes_bytes = blocks * Q4_K_CODE_BYTES;
            let (codes, metadata) = repacked.split_at(codes_bytes);
            let mut gguf_hash = Sha256::new();
            let mut repacked_hash = Sha256::new();
            for block in 0..blocks {
                let source_start = block * Q4_K_GGUF_BLOCK_BYTES;
                let expected = ref_dequant::q4_k::dequant_row(
                    &source[source_start..source_start + Q4_K_GGUF_BLOCK_BYTES],
                    Q4_K_BLOCK_ELEMENTS,
                )
                .expect("GGUF block decodes");
                let mut actual = Vec::with_capacity(Q4_K_BLOCK_ELEMENTS);
                dequantize_block(
                    &codes[block * Q4_K_CODE_BYTES..(block + 1) * Q4_K_CODE_BYTES],
                    &metadata[block * Q4_K_METADATA_BYTES..(block + 1) * Q4_K_METADATA_BYTES],
                    &mut actual,
                );
                assert_bits_equal(&expected, &actual);
                for (left, right) in expected.iter().zip(actual) {
                    gguf_hash.update(left.to_bits().to_le_bytes());
                    repacked_hash.update(right.to_bits().to_le_bytes());
                }
            }
            let gguf_digest = format!("{:x}", gguf_hash.finalize());
            let repacked_digest = format!("{:x}", repacked_hash.finalize());
            assert_eq!(gguf_digest, repacked_digest, "tensor {name}");
            eprintln!("{name}: {gguf_digest}");
        }
    }

    fn assert_block_equivalent(block: &[u8]) {
        let expected =
            ref_dequant::q4_k::dequant_row(block, Q4_K_BLOCK_ELEMENTS).expect("GGUF block decodes");
        let repacked = repack_q4_k(block).expect("block repacks");
        let actual =
            dequantize_q4_k(&repacked, Q4_K_BLOCK_ELEMENTS).expect("repacked block decodes");
        assert_bits_equal(&expected, &actual);
    }

    fn assert_bits_equal(expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len());
        for (index, (left, right)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "dequantized value {index}: {left:e} != {right:e}"
            );
        }
    }

    fn base_block() -> Vec<u8> {
        let mut block = vec![0x5a_u8; Q4_K_GGUF_BLOCK_BYTES];
        set_half(&mut block, 0, 0.03125);
        set_half(&mut block, 2, 0.015625);
        block
    }

    fn set_half(block: &mut [u8], offset: usize, value: f32) {
        block[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3-8B-Q4_K_M.gguf")
    }
}
