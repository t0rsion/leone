#![cfg(feature = "differential")]

use half::f16;
use leone_gguf::ref_dequant;
use leone_gguf::{GgmlType, Gguf};
use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

type Decoder = fn(&[u8], usize) -> ref_dequant::Result<Vec<f32>>;

#[derive(Clone, Copy)]
struct Format {
    name: &'static str,
    dtype: GgmlType,
    block_bytes: usize,
    block_elems: usize,
    d_offset: usize,
    decoder: Decoder,
}

const FORMATS: [Format; 7] = [
    Format {
        name: "f32",
        dtype: GgmlType::F32,
        block_bytes: 4,
        block_elems: 1,
        d_offset: 0,
        decoder: ref_dequant::f32::dequant_row,
    },
    Format {
        name: "f16",
        dtype: GgmlType::F16,
        block_bytes: 2,
        block_elems: 1,
        d_offset: 0,
        decoder: ref_dequant::f16::dequant_row,
    },
    Format {
        name: "q5_0",
        dtype: GgmlType::Q5_0,
        block_bytes: 22,
        block_elems: 32,
        d_offset: 0,
        decoder: ref_dequant::q5_0::dequant_row,
    },
    Format {
        name: "q8_0",
        dtype: GgmlType::Q8_0,
        block_bytes: 34,
        block_elems: 32,
        d_offset: 0,
        decoder: ref_dequant::q8_0::dequant_row,
    },
    Format {
        name: "q4_K",
        dtype: GgmlType::Q4_K,
        block_bytes: 144,
        block_elems: 256,
        d_offset: 0,
        decoder: ref_dequant::q4_k::dequant_row,
    },
    Format {
        name: "q5_K",
        dtype: GgmlType::Q5_K,
        block_bytes: 176,
        block_elems: 256,
        d_offset: 0,
        decoder: ref_dequant::q5_k::dequant_row,
    },
    Format {
        name: "q6_K",
        dtype: GgmlType::Q6_K,
        block_bytes: 210,
        block_elems: 256,
        d_offset: 208,
        decoder: ref_dequant::q6_k::dequant_row,
    },
];

#[test]
#[ignore = "requires external/shim and the pinned llama.cpp build"]
fn random_blocks_match_llama_cpp() -> Result<(), Box<dyn Error>> {
    let mut rng = SmallRng::seed_from_u64(0x6c6c_616d_612d_6767);
    for format in FORMATS {
        let block_count = if format.block_elems == 1 { 257 } else { 1_000 };
        let mut bytes = vec![0; block_count * format.block_bytes];
        rng.fill_bytes(&mut bytes);
        if format.block_elems == 1 {
            make_plain_values_finite(format, &mut bytes, &mut rng);
        } else {
            for block in bytes.chunks_exact_mut(format.block_bytes) {
                set_half(block, format.d_offset, rng.random_range(-4.0_f32..4.0));
                if matches!(format.dtype, GgmlType::Q4_K | GgmlType::Q5_K) {
                    set_half(block, 2, rng.random_range(-4.0_f32..4.0));
                }
            }
        }
        let n_elems = block_count * format.block_elems;
        let rust = (format.decoder)(&bytes, n_elems)?;
        let oracle = shim(format.name, &bytes, n_elems)?;
        compare_rows(format.name, &rust, &oracle)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires the Qwen3 model and external/shim"]
fn named_tensor_edge_rows_match_llama_cpp() -> Result<(), Box<dyn Error>> {
    let gguf = Gguf::open(workspace().join("models/Qwen3-8B-Q4_K_M.gguf"))?;
    let names = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
    ];
    for name in names {
        let tensor = gguf
            .tensor(name)
            .ok_or_else(|| format!("model has no tensor named {name}"))?;
        let format = FORMATS
            .iter()
            .find(|format| format.dtype == tensor.dtype)
            .ok_or_else(|| format!("tensor {name} uses unsupported type {}", tensor.dtype))?;
        let row_elems = usize::try_from(tensor.shape[0])?;
        let row_bytes = row_elems / format.block_elems * format.block_bytes;
        let row_count = tensor
            .shape
            .iter()
            .skip(1)
            .try_fold(1_u64, |value, dimension| value.checked_mul(*dimension))
            .ok_or("row count overflow")?;
        for (edge, row) in [("first", 0), ("last", row_count - 1)] {
            let bytes = gguf.tensor_range(name, row * row_bytes as u64, row_bytes as u64)?;
            let rust = (format.decoder)(&bytes, row_elems)?;
            let oracle = shim(format.name, &bytes, row_elems)?;
            compare_rows(&format!("{name} {edge} row"), &rust, &oracle)?;
        }
    }
    Ok(())
}

fn shim(format: &str, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut child = Command::new(workspace().join("external/shim/build/ggml-dequant"))
        .args([format, &n_elems.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("shim stdin is unavailable")?
        .write_all(bytes)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("shim failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    if output.stdout.len() != n_elems * 4 {
        return Err(format!(
            "shim returned {} bytes, expected {}",
            output.stdout.len(),
            n_elems * 4
        )
        .into());
    }
    Ok(output
        .stdout
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn compare_rows(name: &str, rust: &[f32], oracle: &[f32]) -> Result<(), Box<dyn Error>> {
    if rust.len() != oracle.len() {
        return Err(format!("{name} output lengths differ").into());
    }
    for (index, (left, right)) in rust.iter().zip(oracle).enumerate() {
        if left.to_bits() != right.to_bits() && ulp_distance(*left, *right) > 1 {
            return Err(format!(
                "{name} differs at {index}: Rust {left:e} ({:08x}), llama.cpp {right:e} ({:08x})",
                left.to_bits(),
                right.to_bits()
            )
            .into());
        }
    }
    Ok(())
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    ordered_bits(left).abs_diff(ordered_bits(right))
}

fn ordered_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn make_plain_values_finite(format: Format, bytes: &mut [u8], rng: &mut SmallRng) {
    if format.dtype == GgmlType::F32 {
        for value in bytes.chunks_exact_mut(4) {
            value.copy_from_slice(&rng.random_range(-10.0_f32..10.0).to_le_bytes());
        }
    } else {
        for value in bytes.chunks_exact_mut(2) {
            value.copy_from_slice(
                &f16::from_f32(rng.random_range(-10.0_f32..10.0))
                    .to_bits()
                    .to_le_bytes(),
            );
        }
    }
}

fn set_half(block: &mut [u8], offset: usize, value: f32) {
    block[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("crate is under workspace/crates")
        .to_owned()
}
