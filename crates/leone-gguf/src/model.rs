//! Extracts the pure-GQA model fields needed by the v0.1 runtime.

use crate::{MetadataArray, MetadataValue};
use std::collections::BTreeMap;
use thiserror::Error;

/// The tokenizer metadata checked at model import time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerMetadata {
    pub model: String,
    pub token_count: u64,
}

/// Optional RoPE scaling fields stored by llama.cpp GGUF writers.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScaling {
    pub kind: Option<String>,
    pub factor: Option<f64>,
    pub original_context_length: Option<u64>,
    pub finetuned: Option<bool>,
    pub attention_factor: Option<f64>,
    pub yarn_log_multiplier: Option<f64>,
    pub yarn_ext_factor: Option<f64>,
    pub yarn_beta_fast: Option<f64>,
    pub yarn_beta_slow: Option<f64>,
}

/// The validated pure-GQA fields needed by the v0.1 runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_layer: u64,
    pub n_head: u64,
    pub n_head_kv: u64,
    pub n_embd: u64,
    pub n_ff: u64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub rms_eps: f64,
    pub vocab_size: u64,
    pub context_length: u64,
    pub head_dim: Option<u64>,
    pub tokenizer: TokenizerMetadata,
}

/// An error returned while extracting the supported model schema.
#[derive(Debug, Error, PartialEq)]
pub enum ModelError {
    #[error("GGUF metadata key {0:?} is missing")]
    Missing(String),
    #[error("GGUF metadata key {key:?} has the wrong type, expected {expected}")]
    WrongType { key: String, expected: &'static str },
    #[error(
        "GGUF architecture {0:?} is not supported; expected qwen3 or a llama family architecture"
    )]
    UnsupportedArchitecture(String),
    #[error("GGUF field {field} must be nonzero")]
    Zero { field: &'static str },
    #[error("GGUF field {field} must be finite and greater than zero, found {value}")]
    InvalidFloat { field: &'static str, value: f64 },
    #[error("GGUF n_head {n_head} is not divisible by n_head_kv {n_head_kv}")]
    InvalidGqa { n_head: u64, n_head_kv: u64 },
    #[error("GGUF n_embd {n_embd} is not divisible by n_head {n_head}")]
    InvalidEmbedding { n_embd: u64, n_head: u64 },
    #[error("GGUF vocab size {vocab_size} does not match tokenizer token count {token_count}")]
    VocabMismatch { vocab_size: u64, token_count: u64 },
}

impl ModelConfig {
    /// Extracts and validates a pure-GQA Qwen3 or llama family configuration.
    pub fn from_metadata(metadata: &BTreeMap<String, MetadataValue>) -> Result<Self, ModelError> {
        let architecture = required_string(metadata, "general.architecture")?.to_owned();
        if architecture != "qwen3" && !architecture.starts_with("llama") {
            return Err(ModelError::UnsupportedArchitecture(architecture));
        }
        let prefix = &architecture;
        let n_layer = required_u64(metadata, &key(prefix, "block_count"))?;
        let n_head = required_u64(metadata, &key(prefix, "attention.head_count"))?;
        let n_head_kv = required_u64(metadata, &key(prefix, "attention.head_count_kv"))?;
        let n_embd = required_u64(metadata, &key(prefix, "embedding_length"))?;
        let n_ff = required_u64(metadata, &key(prefix, "feed_forward_length"))?;
        let rope_theta = required_f64(metadata, &key(prefix, "rope.freq_base"))?;
        let rms_eps = required_f64(metadata, &key(prefix, "attention.layer_norm_rms_epsilon"))?;
        let context_length = required_u64(metadata, &key(prefix, "context_length"))?;
        let head_dim = optional_u64(metadata, &key(prefix, "attention.key_length"))?;

        for (field, value) in [
            ("n_layer", n_layer),
            ("n_head", n_head),
            ("n_head_kv", n_head_kv),
            ("n_embd", n_embd),
            ("n_ff", n_ff),
            ("context_length", context_length),
        ] {
            if value == 0 {
                return Err(ModelError::Zero { field });
            }
        }
        if n_head % n_head_kv != 0 {
            return Err(ModelError::InvalidGqa { n_head, n_head_kv });
        }
        if n_embd % n_head != 0 {
            return Err(ModelError::InvalidEmbedding { n_embd, n_head });
        }
        if head_dim == Some(0) {
            return Err(ModelError::Zero { field: "head_dim" });
        }
        for (field, value) in [("rope_theta", rope_theta), ("rms_eps", rms_eps)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ModelError::InvalidFloat { field, value });
            }
        }

        let tokenizer_model = required_string(metadata, "tokenizer.ggml.model")?.to_owned();
        let token_count = match metadata.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(MetadataArray::String(tokens))) => tokens.len() as u64,
            Some(_) => {
                return Err(ModelError::WrongType {
                    key: "tokenizer.ggml.tokens".to_owned(),
                    expected: "string array",
                });
            }
            None => return Err(ModelError::Missing("tokenizer.ggml.tokens".to_owned())),
        };
        if token_count == 0 {
            return Err(ModelError::Zero {
                field: "tokenizer token count",
            });
        }
        let vocab_size = optional_u64(metadata, &key(prefix, "vocab_size"))?.unwrap_or(token_count);
        if vocab_size != token_count {
            return Err(ModelError::VocabMismatch {
                vocab_size,
                token_count,
            });
        }

        let rope_scaling = rope_scaling(metadata, prefix)?;
        Ok(Self {
            architecture,
            n_layer,
            n_head,
            n_head_kv,
            n_embd,
            n_ff,
            rope_theta,
            rope_scaling,
            rms_eps,
            vocab_size,
            context_length,
            head_dim,
            tokenizer: TokenizerMetadata {
                model: tokenizer_model,
                token_count,
            },
        })
    }
}

fn rope_scaling(
    metadata: &BTreeMap<String, MetadataValue>,
    architecture: &str,
) -> Result<Option<RopeScaling>, ModelError> {
    let scaling = RopeScaling {
        kind: optional_string(metadata, &key(architecture, "rope.scaling.type"))?
            .map(str::to_owned),
        factor: optional_f64(metadata, &key(architecture, "rope.scaling.factor"))?,
        original_context_length: optional_u64(
            metadata,
            &key(architecture, "rope.scaling.original_context_length"),
        )?,
        finetuned: optional_bool(metadata, &key(architecture, "rope.scaling.finetuned"))?,
        attention_factor: optional_f64(metadata, &key(architecture, "rope.scaling.attn_factor"))?,
        yarn_log_multiplier: optional_f64(
            metadata,
            &key(architecture, "rope.scaling.yarn_log_multiplier"),
        )?,
        yarn_ext_factor: optional_f64(
            metadata,
            &key(architecture, "rope.scaling.yarn_ext_factor"),
        )?,
        yarn_beta_fast: optional_f64(metadata, &key(architecture, "rope.scaling.yarn_beta_fast"))?,
        yarn_beta_slow: optional_f64(metadata, &key(architecture, "rope.scaling.yarn_beta_slow"))?,
    };
    if scaling.kind.is_none()
        && scaling.factor.is_none()
        && scaling.original_context_length.is_none()
        && scaling.finetuned.is_none()
        && scaling.attention_factor.is_none()
        && scaling.yarn_log_multiplier.is_none()
        && scaling.yarn_ext_factor.is_none()
        && scaling.yarn_beta_fast.is_none()
        && scaling.yarn_beta_slow.is_none()
    {
        Ok(None)
    } else {
        Ok(Some(scaling))
    }
}

fn key(architecture: &str, suffix: &str) -> String {
    format!("{architecture}.{suffix}")
}

fn required_u64(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<u64, ModelError> {
    optional_u64(metadata, key)?.ok_or_else(|| ModelError::Missing(key.to_owned()))
}

fn optional_u64(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<u64>, ModelError> {
    metadata
        .get(key)
        .map(|value| {
            value.as_u64().ok_or_else(|| ModelError::WrongType {
                key: key.to_owned(),
                expected: "unsigned integer",
            })
        })
        .transpose()
}

fn required_f64(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<f64, ModelError> {
    optional_f64(metadata, key)?.ok_or_else(|| ModelError::Missing(key.to_owned()))
}

fn optional_f64(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f64>, ModelError> {
    metadata
        .get(key)
        .map(|value| {
            value.as_f64().ok_or_else(|| ModelError::WrongType {
                key: key.to_owned(),
                expected: "floating point",
            })
        })
        .transpose()
}

fn required_string<'a>(
    metadata: &'a BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a str, ModelError> {
    optional_string(metadata, key)?.ok_or_else(|| ModelError::Missing(key.to_owned()))
}

fn optional_string<'a>(
    metadata: &'a BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<&'a str>, ModelError> {
    metadata
        .get(key)
        .map(|value| {
            value.as_str().ok_or_else(|| ModelError::WrongType {
                key: key.to_owned(),
                expected: "string",
            })
        })
        .transpose()
}

fn optional_bool(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<bool>, ModelError> {
    metadata
        .get(key)
        .map(|value| match value {
            MetadataValue::Bool(value) => Ok(*value),
            _ => Err(ModelError::WrongType {
                key: key.to_owned(),
                expected: "boolean",
            }),
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_metadata() -> BTreeMap<String, MetadataValue> {
        [
            (
                "general.architecture",
                MetadataValue::String("qwen3".to_owned()),
            ),
            ("qwen3.block_count", MetadataValue::Uint32(36)),
            ("qwen3.attention.head_count", MetadataValue::Uint32(32)),
            ("qwen3.attention.head_count_kv", MetadataValue::Uint32(8)),
            ("qwen3.embedding_length", MetadataValue::Uint32(4096)),
            ("qwen3.feed_forward_length", MetadataValue::Uint32(12288)),
            ("qwen3.rope.freq_base", MetadataValue::Float32(1_000_000.0)),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                MetadataValue::Float32(0.000_001),
            ),
            ("qwen3.context_length", MetadataValue::Uint32(40_960)),
            (
                "tokenizer.ggml.model",
                MetadataValue::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens",
                MetadataValue::Array(MetadataArray::String(vec!["a".to_owned(), "b".to_owned()])),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
    }

    #[test]
    fn extracts_valid_qwen3_metadata() {
        let config = ModelConfig::from_metadata(&valid_metadata()).unwrap();
        assert_eq!(config.n_layer, 36);
        assert_eq!(config.n_head_kv, 8);
        assert_eq!(config.vocab_size, 2);
        assert_eq!(config.tokenizer.model, "gpt2");
    }

    #[test]
    fn names_the_rejected_architecture() {
        let mut metadata = valid_metadata();
        metadata.insert(
            "general.architecture".to_owned(),
            MetadataValue::String("mistral".to_owned()),
        );
        assert_eq!(
            ModelConfig::from_metadata(&metadata),
            Err(ModelError::UnsupportedArchitecture("mistral".to_owned()))
        );
    }

    #[test]
    fn rejects_non_integral_gqa_groups() {
        let mut metadata = valid_metadata();
        metadata.insert(
            "qwen3.attention.head_count_kv".to_owned(),
            MetadataValue::Uint32(7),
        );
        assert_eq!(
            ModelConfig::from_metadata(&metadata),
            Err(ModelError::InvalidGqa {
                n_head: 32,
                n_head_kv: 7,
            })
        );
    }

    #[test]
    fn requires_tokenizer_metadata() {
        let mut metadata = valid_metadata();
        metadata.remove("tokenizer.ggml.tokens");
        assert_eq!(
            ModelConfig::from_metadata(&metadata),
            Err(ModelError::Missing("tokenizer.ggml.tokens".to_owned()))
        );
    }
}
