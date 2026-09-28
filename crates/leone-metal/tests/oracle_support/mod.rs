#![allow(dead_code)]

use half::f16;
use leone::{Backend, BufferLayout, QuantFormat};
use leone_gguf::ref_dequant;
use leone_metal::MetalBackend;

pub fn metal_backend() -> Option<MetalBackend> {
    match MetalBackend::new() {
        Ok(backend) => Some(backend),
        Err(error) => {
            #[cfg(target_os = "macos")]
            panic!("Metal initialization failed on macOS: {error}");
            #[cfg(not(target_os = "macos"))]
            {
                let _ = error;
                None
            }
        }
    }
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

pub fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn upload_f32<B: Backend>(backend: &mut B, values: &[f32]) -> B::Buffer {
    let layout = BufferLayout::f32(values.len()).expect("f32 layout");
    backend
        .upload(layout, &f32_bytes(values))
        .expect("upload f32 values")
}

pub fn upload_u32<B: Backend>(backend: &mut B, values: &[u32]) -> B::Buffer {
    let layout = BufferLayout::u32(values.len()).expect("u32 layout");
    backend
        .upload(layout, &u32_bytes(values))
        .expect("upload u32 values")
}

pub fn read_f32<B: Backend>(backend: &mut B, buffer: &B::Buffer, len: usize) -> Vec<f32> {
    let mut values = vec![0.0; len];
    backend
        .read_f32(buffer, &mut values)
        .expect("read f32 values");
    values
}

pub fn read_f16<B: Backend>(backend: &mut B, buffer: &B::Buffer, len: usize) -> Vec<f32> {
    let mut bits = vec![0_u16; len];
    backend
        .read_f16(buffer, &mut bits)
        .expect("read f16 values");
    bits.into_iter()
        .map(f16::from_bits)
        .map(f32::from)
        .collect()
}

pub fn seeded_values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_215.0;
            (unit * 2.0 - 1.0) * (1.0 + (index % 5) as f32 * 0.07)
        })
        .collect()
}

pub fn set_f16(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}

pub fn decode_rows(format: QuantFormat, bytes: &[u8], rows: usize, columns: usize) -> Vec<f32> {
    let row_bytes = columns / 256 * format.block_bytes();
    (0..rows)
        .flat_map(|row| {
            let start = row * row_bytes;
            let row_bytes = &bytes[start..start + row_bytes];
            match format {
                QuantFormat::Q4K => ref_dequant::q4_k::dequant_row(row_bytes, columns),
                QuantFormat::Q6K => ref_dequant::q6_k::dequant_row(row_bytes, columns),
            }
            .expect("reference row decodes")
        })
        .collect()
}

pub fn assert_close(name: &str, actual: &[f32], expected: &[f32], absolute: f32, relative: f32) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{name} at {index} is not finite: {actual}"
        );
        assert!(
            expected.is_finite(),
            "{name} oracle at {index} is not finite: {expected}"
        );
        let error = (*actual - *expected).abs();
        let bound = absolute + relative * expected.abs();
        assert!(
            error <= bound,
            "{name} differs at {index}: actual={actual:e}, expected={expected:e}, error={error:e}, bound={bound:e}"
        );
    }
}

pub fn matrix_dot(decoded: &[f32], rows: usize, columns: usize, input: &[f32]) -> Vec<f32> {
    assert_eq!(decoded.len(), rows * columns);
    assert_eq!(input.len(), columns);
    decoded
        .chunks_exact(columns)
        .map(|row| {
            row.iter()
                .zip(input)
                .fold(0.0_f32, |sum, (weight, value)| weight.mul_add(*value, sum))
        })
        .collect()
}

pub fn matrix_gemm(
    decoded: &[f32],
    rows: usize,
    columns: usize,
    input: &[f32],
    tokens: usize,
) -> Vec<f32> {
    assert_eq!(decoded.len(), rows * columns);
    assert_eq!(input.len(), tokens * columns);
    input
        .chunks_exact(columns)
        .flat_map(|input_row| matrix_dot(decoded, rows, columns, input_row))
        .collect()
}
