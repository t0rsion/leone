//! Reports import and execution support without treating metadata as runtime support.

use crate::{ref_dequant, GgmlType};
use thiserror::Error;

/// The structural class of one model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureClass {
    DenseGqa,
    MixtureOfExperts,
    VisionLanguage,
    HybridState,
    Unknown,
}

impl ArchitectureClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::DenseGqa => "dense-gqa",
            Self::MixtureOfExperts => "mixture-of-experts",
            Self::VisionLanguage => "vision-language",
            Self::HybridState => "hybrid-state",
            Self::Unknown => "unknown",
        }
    }
}

/// Support levels for one architecture name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureSupport {
    pub architecture: String,
    pub class: ArchitectureClass,
    pub metadata_probe: bool,
    pub text_runtime: bool,
    pub vision_runtime: bool,
}

/// Returns the capability record for a GGUF architecture name.
pub fn architecture_support(architecture: &str) -> ArchitectureSupport {
    let class = match architecture {
        "qwen3" | "qwen2" | "llama" | "llama2" | "llama3" | "mistral" => {
            ArchitectureClass::DenseGqa
        }
        "qwen3moe" | "mixtral" | "deepseek2" => ArchitectureClass::MixtureOfExperts,
        "qwen3vl" | "llava" | "gemma3" => ArchitectureClass::VisionLanguage,
        "qwen3next" | "mamba" => ArchitectureClass::HybridState,
        _ => ArchitectureClass::Unknown,
    };
    ArchitectureSupport {
        architecture: architecture.to_owned(),
        class,
        metadata_probe: class != ArchitectureClass::Unknown,
        text_runtime: matches!(architecture, "qwen3" | "llama" | "llama2" | "llama3"),
        vision_runtime: false,
    }
}

/// Import and runtime support for one quantized tensor format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationSupport {
    pub block_values: u64,
    pub block_bytes: u64,
    pub scalar_oracle: bool,
    pub cpu_runtime: bool,
    pub cuda_runtime: bool,
}

/// Returns support for one quantized type that has a scalar decoder.
pub fn quantization_support(dtype: GgmlType) -> Option<QuantizationSupport> {
    let (block_values, block_bytes) = dtype.block_layout()?;
    let scalar_oracle = matches!(
        dtype,
        GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0
    );
    scalar_oracle.then_some(QuantizationSupport {
        block_values,
        block_bytes,
        scalar_oracle,
        cpu_runtime: matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K),
        cuda_runtime: matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K),
    })
}

/// A tensor block that cannot be imported by the scalar oracle.
#[derive(Debug, Error)]
pub enum BlockImportError {
    #[error("quantized tensor type {0} has no scalar importer")]
    Unsupported(GgmlType),
    #[error("{dtype} block has {actual} bytes; expected {expected}")]
    BlockBytes {
        dtype: GgmlType,
        expected: usize,
        actual: usize,
    },
    #[error(transparent)]
    Decode(#[from] ref_dequant::DequantError),
}

/// Decodes one complete storage block through the scalar importer.
pub fn decode_quantized_block(dtype: GgmlType, block: &[u8]) -> Result<Vec<f32>, BlockImportError> {
    let support = quantization_support(dtype).ok_or(BlockImportError::Unsupported(dtype))?;
    let expected =
        usize::try_from(support.block_bytes).map_err(|_| BlockImportError::Unsupported(dtype))?;
    if block.len() != expected {
        return Err(BlockImportError::BlockBytes {
            dtype,
            expected,
            actual: block.len(),
        });
    }
    let values =
        usize::try_from(support.block_values).map_err(|_| BlockImportError::Unsupported(dtype))?;
    decode_supported_block(dtype, block, values)
}

fn decode_supported_block(
    dtype: GgmlType,
    block: &[u8],
    values: usize,
) -> Result<Vec<f32>, BlockImportError> {
    match dtype {
        GgmlType::Q4_K => ref_dequant::q4_k::dequant_row(block, values),
        GgmlType::Q5_0 => ref_dequant::q5_0::dequant_row(block, values),
        GgmlType::Q5_K => ref_dequant::q5_k::dequant_row(block, values),
        GgmlType::Q6_K => ref_dequant::q6_k::dequant_row(block, values),
        GgmlType::Q8_0 => ref_dequant::q8_0::dequant_row(block, values),
        _ => return Err(BlockImportError::Unsupported(dtype)),
    }
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn every_claimed_scalar_importer_decodes_one_complete_block() {
        for dtype in [
            GgmlType::Q4_K,
            GgmlType::Q5_0,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            let support = quantization_support(dtype).expect("claimed support");
            let block = vec![0_u8; support.block_bytes as usize];
            let values = decode_quantized_block(dtype, &block).expect("complete block");
            assert_eq!(values.len(), support.block_values as usize);
            assert!(values.into_iter().all(f32::is_finite));
        }
    }

    #[test]
    fn support_distinguishes_probe_and_runtime_architectures() {
        let qwen = architecture_support("qwen3");
        assert!(qwen.metadata_probe && qwen.text_runtime);
        let llama = architecture_support("llama3");
        assert!(llama.metadata_probe && llama.text_runtime);
        let moe = architecture_support("qwen3moe");
        assert_eq!(moe.class, ArchitectureClass::MixtureOfExperts);
        assert!(moe.metadata_probe && !moe.text_runtime);
        let vision = architecture_support("qwen3vl");
        assert_eq!(vision.class, ArchitectureClass::VisionLanguage);
        assert!(!vision.vision_runtime);
        assert!(!architecture_support("future-unknown").metadata_probe);
    }

    #[test]
    fn support_matches_the_frozen_fixture() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../fixtures/model-contracts.json"))
                .expect("model contract fixture");
        for entry in fixture["architectures"].as_array().expect("architectures") {
            let expected = architecture_support(entry["name"].as_str().expect("name"));
            assert_eq!(
                expected.class.name(),
                entry["class"].as_str().expect("class")
            );
            assert_eq!(
                expected.metadata_probe,
                entry["metadata_probe"].as_bool().expect("metadata probe")
            );
            assert_eq!(
                expected.text_runtime,
                entry["text_runtime"].as_bool().expect("text runtime")
            );
            assert_eq!(
                expected.vision_runtime,
                entry["vision_runtime"].as_bool().expect("vision runtime")
            );
        }
        for entry in fixture["quantizations"].as_array().expect("quantizations") {
            let dtype = match entry["name"].as_str().expect("name") {
                "Q4_K" => GgmlType::Q4_K,
                "Q5_0" => GgmlType::Q5_0,
                "Q5_K" => GgmlType::Q5_K,
                "Q6_K" => GgmlType::Q6_K,
                "Q8_0" => GgmlType::Q8_0,
                name => panic!("unknown fixture dtype {name}"),
            };
            let expected = quantization_support(dtype).expect("quantization support");
            assert_eq!(
                expected.block_values,
                entry["block_values"].as_u64().expect("block values")
            );
            assert_eq!(
                expected.block_bytes,
                entry["block_bytes"].as_u64().expect("block bytes")
            );
            assert_eq!(
                expected.scalar_oracle,
                entry["scalar_oracle"].as_bool().expect("scalar oracle")
            );
            assert_eq!(
                expected.cpu_runtime,
                entry["cpu_runtime"].as_bool().expect("CPU runtime")
            );
            assert_eq!(
                expected.cuda_runtime,
                entry["cuda_runtime"].as_bool().expect("CUDA runtime")
            );
        }
    }
}
