//! Provides exact scalar dequantizers for GGUF import validation.
//!
//! The layouts follow `block_q5_0`, `block_q8_0`, `block_q4_K`,
//! `block_q5_K`, and `block_q6_K` in llama.cpp `ggml-common.h`.

use half::f16 as F16;
use thiserror::Error;

const K_BLOCK_ELEMENTS: usize = 256;

/// An invalid scalar dequantizer input.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DequantError {
    #[error("element count {n_elems} is not divisible by block size {block_elems}")]
    ElementCount { n_elems: usize, block_elems: usize },
    #[error("input has {actual} bytes, expected {expected}")]
    ByteLength { actual: usize, expected: usize },
    #[error("dequantized output length overflows the host size")]
    LengthOverflow,
}

/// A scalar dequantizer result.
pub type Result<T> = std::result::Result<T, DequantError>;

/// Dequantizes little-endian F32 rows.
pub mod f32 {
    /// Dequantizes one row into scalar `f32` values.
    pub fn dequant_row(bytes: &[u8], n_elems: usize) -> super::Result<Vec<f32>> {
        super::dequant_f32(bytes, n_elems)
    }
}

/// Dequantizes little-endian F16 rows.
pub mod f16 {
    /// Dequantizes one row into scalar `f32` values.
    pub fn dequant_row(bytes: &[u8], n_elems: usize) -> super::Result<Vec<f32>> {
        super::dequant_f16(bytes, n_elems)
    }
}

/// Dequantizes `block_q5_0` rows.
pub mod q5_0 {
    /// Dequantizes one row into scalar `f32` values.
    pub fn dequant_row(bytes: &[u8], n_elems: usize) -> super::Result<Vec<f32>> {
        super::dequant_q5_0(bytes, n_elems)
    }
}

/// Dequantizes `block_q8_0` rows.
pub mod q8_0 {
    /// Dequantizes one row into scalar `f32` values.
    pub fn dequant_row(bytes: &[u8], n_elems: usize) -> super::Result<Vec<f32>> {
        super::dequant_q8_0(bytes, n_elems)
    }
}

/// Dequantizes `block_q4_K` rows.
pub mod q4_k {
    /// Dequantizes one row into scalar `f32` values.
    pub fn dequant_row(bytes: &[u8], n_elems: usize) -> super::Result<Vec<f32>> {
        super::dequant_q4_k(bytes, n_elems)
    }
}

/// Dequantizes `block_q5_K` rows.
pub mod q5_k {
    /// Dequantizes one row into scalar `f32` values.
    pub fn dequant_row(bytes: &[u8], n_elems: usize) -> super::Result<Vec<f32>> {
        super::dequant_q5_k(bytes, n_elems)
    }
}

/// Dequantizes `block_q6_K` rows.
pub mod q6_k {
    /// Dequantizes one row into scalar `f32` values.
    pub fn dequant_row(bytes: &[u8], n_elems: usize) -> super::Result<Vec<f32>> {
        super::dequant_q6_k(bytes, n_elems)
    }
}

fn dequant_f32(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    validate(bytes, n_elems, 1, 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect())
}

fn dequant_f16(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    validate(bytes, n_elems, 1, 2)?;
    Ok(bytes.chunks_exact(2).map(|value| half(value, 0)).collect())
}

fn dequant_q5_0(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 22;
    validate(bytes, n_elems, 32, BLOCK_BYTES)?;
    let mut output = Vec::with_capacity(n_elems);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = half(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        for (j, quant) in qs.iter().copied().enumerate() {
            let high0 = (((qh >> j) << 4) & 0x10) as u8;
            let high1 = ((qh >> (j + 12)) & 0x10) as u8;
            output.push((((quant & 0x0f) | high0) as i32 - 16) as f32 * d);
            output.push((((quant >> 4) | high1) as i32 - 16) as f32 * d);
        }
        reorder_halves(&mut output, 32);
    }
    Ok(output)
}

fn dequant_q8_0(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 34;
    validate(bytes, n_elems, 32, BLOCK_BYTES)?;
    let mut output = Vec::with_capacity(n_elems);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = half(block, 0);
        output.extend(block[2..].iter().map(|quant| f32::from(*quant as i8) * d));
    }
    Ok(output)
}

fn dequant_q4_k(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 144;
    validate(bytes, n_elems, K_BLOCK_ELEMENTS, BLOCK_BYTES)?;
    let mut output = Vec::with_capacity(n_elems);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = half(block, 0);
        let dmin = half(block, 2);
        let scales = &block[4..16];
        let quants = &block[16..144];
        let mut scale_index = 0;
        for chunk in quants.chunks_exact(32) {
            let (scale0, min0) = scale_min(scale_index, scales);
            let (scale1, min1) = scale_min(scale_index + 1, scales);
            let d0 = d * f32::from(scale0);
            let d1 = d * f32::from(scale1);
            let m0 = dmin * f32::from(min0);
            let m1 = dmin * f32::from(min1);
            output.extend(chunk.iter().map(|quant| d0 * f32::from(quant & 0x0f) - m0));
            output.extend(chunk.iter().map(|quant| d1 * f32::from(quant >> 4) - m1));
            scale_index += 2;
        }
    }
    Ok(output)
}

fn dequant_q5_k(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 176;
    validate(bytes, n_elems, K_BLOCK_ELEMENTS, BLOCK_BYTES)?;
    let mut output = Vec::with_capacity(n_elems);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = half(block, 0);
        let dmin = half(block, 2);
        let scales = &block[4..16];
        let high = &block[16..48];
        let low = &block[48..176];
        let mut scale_index = 0;
        let mut high0 = 1_u8;
        let mut high1 = 2_u8;
        for chunk in low.chunks_exact(32) {
            let (scale0, min0) = scale_min(scale_index, scales);
            let (scale1, min1) = scale_min(scale_index + 1, scales);
            let d0 = d * f32::from(scale0);
            let d1 = d * f32::from(scale1);
            let m0 = dmin * f32::from(min0);
            let m1 = dmin * f32::from(min1);
            for (index, quant) in chunk.iter().copied().enumerate() {
                let value = (quant & 0x0f) + if high[index] & high0 != 0 { 16 } else { 0 };
                output.push(d0 * f32::from(value) - m0);
            }
            for (index, quant) in chunk.iter().copied().enumerate() {
                let value = (quant >> 4) + if high[index] & high1 != 0 { 16 } else { 0 };
                output.push(d1 * f32::from(value) - m1);
            }
            scale_index += 2;
            high0 <<= 2;
            high1 <<= 2;
        }
    }
    Ok(output)
}

fn dequant_q6_k(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 210;
    validate(bytes, n_elems, K_BLOCK_ELEMENTS, BLOCK_BYTES)?;
    let mut output = vec![0.0; n_elems];
    for (block_index, block) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let low = &block[0..128];
        let high = &block[128..192];
        let scales = &block[192..208];
        let d = half(block, 208);
        let output_base = block_index * K_BLOCK_ELEMENTS;
        for half_index in 0..2 {
            let low = &low[half_index * 64..];
            let high = &high[half_index * 32..];
            let scales = &scales[half_index * 8..];
            let base = output_base + half_index * 128;
            for index in 0..32 {
                let scale_index = index / 16;
                let q0 = ((low[index] & 0x0f) | ((high[index] & 3) << 4)) as i8 - 32;
                let q1 = ((low[index + 32] & 0x0f) | (((high[index] >> 2) & 3) << 4)) as i8 - 32;
                let q2 = ((low[index] >> 4) | (((high[index] >> 4) & 3) << 4)) as i8 - 32;
                let q3 = ((low[index + 32] >> 4) | (((high[index] >> 6) & 3) << 4)) as i8 - 32;
                output[base + index] = d * f32::from(scales[scale_index] as i8) * f32::from(q0);
                output[base + index + 32] =
                    d * f32::from(scales[scale_index + 2] as i8) * f32::from(q1);
                output[base + index + 64] =
                    d * f32::from(scales[scale_index + 4] as i8) * f32::from(q2);
                output[base + index + 96] =
                    d * f32::from(scales[scale_index + 6] as i8) * f32::from(q3);
            }
        }
    }
    Ok(output)
}

fn half(bytes: &[u8], offset: usize) -> f32 {
    F16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
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

fn validate(bytes: &[u8], n_elems: usize, block_elems: usize, block_bytes: usize) -> Result<()> {
    if !n_elems.is_multiple_of(block_elems) {
        return Err(DequantError::ElementCount {
            n_elems,
            block_elems,
        });
    }
    let expected = n_elems
        .checked_div(block_elems)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(DequantError::LengthOverflow)?;
    if bytes.len() != expected {
        return Err(DequantError::ByteLength {
            actual: bytes.len(),
            expected,
        });
    }
    Ok(())
}

fn reorder_halves(values: &mut [f32], block_elems: usize) {
    let start = values.len() - block_elems;
    let block = &mut values[start..];
    let mut reordered = vec![0.0; block_elems];
    for index in 0..block_elems / 2 {
        reordered[index] = block[index * 2];
        reordered[index + block_elems / 2] = block[index * 2 + 1];
    }
    block.copy_from_slice(&reordered);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, RngCore, SeedableRng};

    type Decoder = fn(&[u8], usize) -> Result<Vec<f32>>;

    #[derive(Clone, Copy)]
    struct Format {
        block_bytes: usize,
        block_elems: usize,
        d_offset: usize,
        decoder: Decoder,
    }

    const FORMATS: [Format; 5] = [
        Format {
            block_bytes: 22,
            block_elems: 32,
            d_offset: 0,
            decoder: dequant_q5_0,
        },
        Format {
            block_bytes: 34,
            block_elems: 32,
            d_offset: 0,
            decoder: dequant_q8_0,
        },
        Format {
            block_bytes: 144,
            block_elems: 256,
            d_offset: 0,
            decoder: dequant_q4_k,
        },
        Format {
            block_bytes: 176,
            block_elems: 256,
            d_offset: 0,
            decoder: dequant_q5_k,
        },
        Format {
            block_bytes: 210,
            block_elems: 256,
            d_offset: 208,
            decoder: dequant_q6_k,
        },
    ];

    #[test]
    fn rejects_partial_blocks_without_panicking() {
        for format in FORMATS {
            assert!(matches!(
                (format.decoder)(&[0], format.block_elems),
                Err(DequantError::ByteLength { .. })
            ));
            assert!(matches!(
                (format.decoder)(&[], format.block_elems - 1),
                Err(DequantError::ElementCount { .. })
            ));
        }
    }

    #[test]
    fn edge_patterns_have_finite_outputs() {
        for format in FORMATS {
            let patterns = edge_patterns(format);
            for block in patterns {
                let output = (format.decoder)(&block, format.block_elems).unwrap();
                assert_eq!(output.len(), format.block_elems);
                assert!(output.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn every_payload_byte_code_has_finite_outputs() {
        for format in FORMATS {
            for byte in 0_u8..=u8::MAX {
                let mut block = vec![byte; format.block_bytes];
                set_half(&mut block, format.d_offset, 1.0);
                if matches!(format.block_bytes, 144 | 176) {
                    set_half(&mut block, 2, 1.0);
                }
                let output = (format.decoder)(&block, format.block_elems).unwrap();
                assert!(output.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn one_thousand_seeded_random_blocks_per_format_are_finite() {
        let mut rng = SmallRng::seed_from_u64(0x6767_7566_7633);
        for format in FORMATS {
            for _ in 0..1_000 {
                let mut block = vec![0; format.block_bytes];
                rng.fill_bytes(&mut block);
                set_half(&mut block, format.d_offset, rng.random_range(-4.0_f32..4.0));
                if matches!(format.block_bytes, 144 | 176) {
                    set_half(&mut block, 2, rng.random_range(-4.0_f32..4.0));
                }
                let output = (format.decoder)(&block, format.block_elems).unwrap();
                assert!(output.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn q8_quant_negation_negates_output() {
        let mut block = vec![0; 34];
        set_half(&mut block, 0, 0.5);
        for (index, value) in block[2..].iter_mut().enumerate() {
            *value = (index as i8 - 16) as u8;
        }
        let positive = dequant_q8_0(&block, 32).unwrap();
        for value in &mut block[2..] {
            *value = (-(*value as i8)) as u8;
        }
        let negative = dequant_q8_0(&block, 32).unwrap();
        for (left, right) in positive.iter().zip(negative) {
            assert_eq!(*left, -right);
        }
    }

    fn edge_patterns(format: Format) -> Vec<Vec<u8>> {
        let zero = vec![0; format.block_bytes];
        let mut max_scales = vec![u8::MAX; format.block_bytes];
        set_half(&mut max_scales, format.d_offset, 1.0);
        if matches!(format.block_bytes, 144 | 176) {
            set_half(&mut max_scales, 2, 1.0);
        }
        let mut alternating = (0..format.block_bytes)
            .map(|index| if index % 2 == 0 { 0xaa } else { 0x55 })
            .collect::<Vec<_>>();
        set_half(&mut alternating, format.d_offset, 1.0);
        if matches!(format.block_bytes, 144 | 176) {
            set_half(&mut alternating, 2, 1.0);
        }
        let mut subnormal = vec![0xff; format.block_bytes];
        subnormal[format.d_offset..format.d_offset + 2].copy_from_slice(&1_u16.to_le_bytes());
        if matches!(format.block_bytes, 144 | 176) {
            subnormal[2..4].copy_from_slice(&1_u16.to_le_bytes());
        }
        vec![zero, max_scales, alternating, subnormal]
    }

    fn set_half(block: &mut [u8], offset: usize, value: f32) {
        block[offset..offset + 2].copy_from_slice(&F16::from_f32(value).to_bits().to_le_bytes());
    }

    // The tests below pin exact outputs for crafted blocks. Each expected
    // value is computed by hand from the llama.cpp block layout, not captured
    // from this implementation. A wrong nibble order, misplaced high bit, or
    // wrong scale stride fails here instead of producing a finite wrong
    // number. `tests/differential.rs` is the layout oracle. These tests pin
    // the arithmetic on top of it.

    #[test]
    fn q8_0_scales_signed_quants() {
        let mut block = vec![0; 34];
        set_half(&mut block, 0, 0.5);
        for index in 0..32 {
            block[2 + index] = index as u8;
        }
        block[2] = (-128_i8) as u8;
        let output = dequant_q8_0(&block, 32).unwrap();
        assert_eq!(output[0], -64.0);
        assert_eq!(output[1], 0.5);
        assert_eq!(output[31], 15.5);
    }

    #[test]
    fn q5_0_places_the_fifth_bit_and_deinterleaves() {
        let mut block = vec![0; 22];
        set_half(&mut block, 0, 1.0);
        // Bit 0 lifts the low nibble of byte 0; bit 16 lifts its high nibble.
        block[2..6].copy_from_slice(&0x0001_0001_u32.to_le_bytes());
        block[6] = 0x00;
        block[7] = 0x0f;
        let output = dequant_q5_0(&block, 32).unwrap();
        assert_eq!(output[0], 0.0);
        assert_eq!(output[16], 0.0);
        assert_eq!(output[1], -1.0);
        assert_eq!(output[17], -16.0);
        assert_eq!(output[2], -16.0);
    }

    #[test]
    fn q4_k_orders_nibbles_and_applies_the_packed_minimum() {
        let mut block = vec![0; 144];
        set_half(&mut block, 0, 1.0);
        set_half(&mut block, 2, 1.0);
        block[4] = 1;
        block[5] = 2;
        block[8] = 3;
        for index in 0..32 {
            block[16 + index] = 0x21;
        }
        let output = dequant_q4_k(&block, 256).unwrap();
        // Low nibbles first, scaled by scales[0] and offset by mins[0].
        assert_eq!(output[0], 1.0 * 1.0 * 1.0 - 1.0 * 3.0);
        assert_eq!(output[31], -2.0);
        // High nibbles next, scaled by scales[1] with a zero minimum.
        assert_eq!(output[32], 2.0 * 2.0);
        assert_eq!(output[63], 4.0);
    }

    #[test]
    fn q5_k_advances_its_high_bit_masks_per_chunk() {
        let mut block = vec![0; 176];
        set_half(&mut block, 0, 1.0);
        set_half(&mut block, 2, 0.0);
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        block[16] = 0b0000_0111;
        block[17] = 0b0000_0001;
        let output = dequant_q5_k(&block, 256).unwrap();
        assert_eq!(output[0], 16.0);
        assert_eq!(output[32], 16.0);
        assert_eq!(output[1], 16.0);
        assert_eq!(output[33], 0.0);
        // The second chunk reads masks 4 and 8 from the same high byte.
        assert_eq!(output[64], 16.0);
        assert_eq!(output[96], 0.0);
    }

    #[test]
    fn q6_k_splits_four_quants_from_one_high_byte() {
        let mut block = vec![0; 210];
        set_half(&mut block, 208, 1.0);
        for index in 0..8 {
            block[192 + index] = 1;
        }
        block[194] = 2;
        block[0] = 0x10;
        block[32] = 0x00;
        block[128] = 0b1110_0100;
        let output = dequant_q6_k(&block, 256).unwrap();
        assert_eq!(output[0], -32.0);
        assert_eq!(output[32], -32.0);
        assert_eq!(output[64], 1.0);
        assert_eq!(output[96], 16.0);
    }

    #[test]
    fn f16_decodes_known_bit_patterns() {
        let block = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x00];
        let output = dequant_f16(&block, 3).unwrap();
        assert_eq!(output, [1.0, -2.0, 0.0]);
    }
}
