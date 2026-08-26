//! Splits fixed-width target codes into portable base and refinement planes.

use thiserror::Error;

const BASE_BITS: u8 = 2;

/// A supported complete target-code width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressiveWidth {
    /// Four-bit target codes with a two-bit refinement.
    Q4,
    /// Six-bit target codes with a four-bit refinement.
    Q6,
}

impl ProgressiveWidth {
    pub const fn target_bits(self) -> u8 {
        match self {
            Self::Q4 => 4,
            Self::Q6 => 6,
        }
    }

    pub const fn refinement_bits(self) -> u8 {
        self.target_bits() - BASE_BITS
    }

    const fn maximum_code(self) -> u8 {
        (1_u8 << self.target_bits()) - 1
    }
}

/// An invalid progressive-code payload or value.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProgressiveError {
    #[error("code {code} at index {index} exceeds the {bits}-bit target range")]
    CodeOutOfRange { index: usize, code: u8, bits: u8 },
    #[error("progressive code count overflowed")]
    SizeOverflow,
    #[error("the {plane} plane has {actual} bytes, expected {expected}")]
    PlaneSize {
        plane: &'static str,
        expected: usize,
        actual: usize,
    },
}

/// Complete target codes stored as a two-bit base plane and a refinement plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressiveCodes {
    width: ProgressiveWidth,
    codes: usize,
    base: Vec<u8>,
    refinement: Vec<u8>,
}

impl ProgressiveCodes {
    /// Splits complete unsigned target codes without changing their values.
    pub fn encode(width: ProgressiveWidth, codes: &[u8]) -> Result<Self, ProgressiveError> {
        for (index, code) in codes.iter().copied().enumerate() {
            if code > width.maximum_code() {
                return Err(ProgressiveError::CodeOutOfRange {
                    index,
                    code,
                    bits: width.target_bits(),
                });
            }
        }
        let refinement_bits = width.refinement_bits();
        let refinement_mask = (1_u8 << refinement_bits) - 1;
        let base_codes = codes
            .iter()
            .map(|code| code >> refinement_bits)
            .collect::<Vec<_>>();
        let refinement_codes = codes
            .iter()
            .map(|code| code & refinement_mask)
            .collect::<Vec<_>>();
        Ok(Self {
            width,
            codes: codes.len(),
            base: pack(&base_codes, BASE_BITS)?,
            refinement: pack(&refinement_codes, refinement_bits)?,
        })
    }

    /// Reconstructs every complete target code.
    pub fn decode_complete(&self) -> Result<Vec<u8>, ProgressiveError> {
        self.validate_plane_sizes()?;
        let base = unpack(&self.base, BASE_BITS, self.codes);
        let refinement_bits = self.width.refinement_bits();
        let refinement = unpack(&self.refinement, refinement_bits, self.codes);
        Ok(base
            .into_iter()
            .zip(refinement)
            .map(|(base, refinement)| (base << refinement_bits) | refinement)
            .collect())
    }

    /// Returns the two-bit base codes used by an approximate view.
    pub fn decode_base(&self) -> Result<Vec<u8>, ProgressiveError> {
        self.validate_plane_sizes()?;
        Ok(unpack(&self.base, BASE_BITS, self.codes))
    }

    pub const fn width(&self) -> ProgressiveWidth {
        self.width
    }

    pub const fn len(&self) -> usize {
        self.codes
    }

    pub const fn is_empty(&self) -> bool {
        self.codes == 0
    }

    pub fn base_bytes(&self) -> &[u8] {
        &self.base
    }

    pub fn refinement_bytes(&self) -> &[u8] {
        &self.refinement
    }

    /// Returns stored bytes, including terminal-byte padding in both planes.
    pub fn stored_bytes(&self) -> usize {
        self.base.len() + self.refinement.len()
    }

    fn validate_plane_sizes(&self) -> Result<(), ProgressiveError> {
        let base = packed_len(self.codes, BASE_BITS)?;
        if self.base.len() != base {
            return Err(ProgressiveError::PlaneSize {
                plane: "base",
                expected: base,
                actual: self.base.len(),
            });
        }
        let refinement = packed_len(self.codes, self.width.refinement_bits())?;
        if self.refinement.len() != refinement {
            return Err(ProgressiveError::PlaneSize {
                plane: "refinement",
                expected: refinement,
                actual: self.refinement.len(),
            });
        }
        Ok(())
    }
}

/// Exhaustive scalar reconstruction counts for one format gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressiveOracleStats {
    pub q4_codes: u64,
    pub q4_mismatches: u64,
    pub q6_codes: u64,
    pub q6_mismatches: u64,
}

/// Compares every Q4 and Q6 code to direct shift-and-mask reconstruction.
pub fn exhaustive_progressive_oracle() -> Result<ProgressiveOracleStats, ProgressiveError> {
    let q4 = (0..=ProgressiveWidth::Q4.maximum_code()).collect::<Vec<_>>();
    let q6 = (0..=ProgressiveWidth::Q6.maximum_code()).collect::<Vec<_>>();
    let q4_decoded = ProgressiveCodes::encode(ProgressiveWidth::Q4, &q4)?.decode_complete()?;
    let q6_decoded = ProgressiveCodes::encode(ProgressiveWidth::Q6, &q6)?.decode_complete()?;
    Ok(ProgressiveOracleStats {
        q4_codes: q4.len() as u64,
        q4_mismatches: q4.iter().zip(q4_decoded).filter(|(a, b)| **a != *b).count() as u64,
        q6_codes: q6.len() as u64,
        q6_mismatches: q6.iter().zip(q6_decoded).filter(|(a, b)| **a != *b).count() as u64,
    })
}

fn packed_len(values: usize, bits: u8) -> Result<usize, ProgressiveError> {
    values
        .checked_mul(bits as usize)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or(ProgressiveError::SizeOverflow)
}

fn pack(values: &[u8], bits: u8) -> Result<Vec<u8>, ProgressiveError> {
    let mut output = vec![0_u8; packed_len(values.len(), bits)?];
    for (index, value) in values.iter().copied().enumerate() {
        let start = index * bits as usize;
        for bit in 0..bits as usize {
            if value & (1_u8 << bit) != 0 {
                output[(start + bit) / 8] |= 1_u8 << ((start + bit) % 8);
            }
        }
    }
    Ok(output)
}

fn unpack(bytes: &[u8], bits: u8, values: usize) -> Vec<u8> {
    (0..values)
        .map(|index| {
            let start = index * bits as usize;
            (0..bits as usize).fold(0_u8, |value, bit| {
                let source = (bytes[(start + bit) / 8] >> ((start + bit) % 8)) & 1;
                value | (source << bit)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_target_codes_reconstruct() {
        let result = exhaustive_progressive_oracle().unwrap();
        assert_eq!(result.q4_codes, 16);
        assert_eq!(result.q4_mismatches, 0);
        assert_eq!(result.q6_codes, 64);
        assert_eq!(result.q6_mismatches, 0);
    }

    #[test]
    fn terminal_bytes_preserve_partial_planes() {
        let values = [0, 15, 3, 12, 7];
        let codes = ProgressiveCodes::encode(ProgressiveWidth::Q4, &values).unwrap();
        assert_eq!(codes.decode_complete().unwrap(), values);
        assert_eq!(codes.stored_bytes(), 4);
    }

    #[test]
    fn out_of_range_code_is_typed() {
        assert!(matches!(
            ProgressiveCodes::encode(ProgressiveWidth::Q4, &[16]),
            Err(ProgressiveError::CodeOutOfRange { .. })
        ));
    }
}
