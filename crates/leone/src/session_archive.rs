//! Defines the persistent, model-bound replay format for generation sessions.

use crate::{GenerationCheckpoint, MirostatConfig, MirostatState, SamplerError};
use leone_receipt::sha256_bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Persistent session archive schema version.
pub const SESSION_ARCHIVE_SCHEMA_VERSION: u32 = 1;

/// A validated session replay archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionArchive {
    schema_version: u32,
    model_sha256: String,
    evaluated_tokens: Vec<u32>,
    prefill_boundary: usize,
    mirostat: Option<MirostatArchive>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MirostatArchive {
    target_surprise: f64,
    learning_rate: f64,
    maximum_surprise: f64,
}

/// An invalid or incompatible persistent session archive.
#[derive(Debug, Error)]
pub enum SessionArchiveError {
    #[error("session archive JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported session archive schema {found}; expected {expected}")]
    Schema { found: u32, expected: u32 },
    #[error("session archive model hash is not a lowercase SHA-256 digest")]
    ModelHash,
    #[error("session archive belongs to model {found}, not {expected}")]
    ModelMismatch { found: String, expected: String },
    #[error("session archive prefill boundary {boundary} exceeds {tokens} evaluated tokens")]
    PrefillBoundary { boundary: usize, tokens: usize },
    #[error(transparent)]
    Sampler(#[from] SamplerError),
}

impl SessionArchive {
    /// Captures the replay state and binds it to one model file.
    pub fn new(
        model_sha256: impl Into<String>,
        checkpoint: &GenerationCheckpoint,
    ) -> Result<Self, SessionArchiveError> {
        let mirostat = checkpoint.mirostat().map(|state| MirostatArchive {
            target_surprise: state.config().target_surprise(),
            learning_rate: state.config().learning_rate(),
            maximum_surprise: state.maximum_surprise(),
        });
        let archive = Self {
            schema_version: SESSION_ARCHIVE_SCHEMA_VERSION,
            model_sha256: model_sha256.into(),
            evaluated_tokens: checkpoint.evaluated_tokens().to_vec(),
            prefill_boundary: checkpoint.prefill_boundary(),
            mirostat,
        };
        archive.validate(None)?;
        Ok(archive)
    }

    /// Parses and validates an archive for the expected model.
    pub fn from_json(bytes: &[u8], expected_model: &str) -> Result<Self, SessionArchiveError> {
        let archive: Self = serde_json::from_slice(bytes)?;
        archive.validate(Some(expected_model))?;
        Ok(archive)
    }

    /// Serializes the canonical archive bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, SessionArchiveError> {
        self.validate(None)?;
        Ok(serde_json::to_vec(self)?)
    }

    /// Returns the SHA-256 of the canonical archive bytes.
    pub fn sha256(&self) -> Result<String, SessionArchiveError> {
        Ok(sha256_bytes(&self.to_json()?))
    }

    /// Reconstructs the runtime checkpoint after validation.
    pub fn checkpoint(&self) -> Result<GenerationCheckpoint, SessionArchiveError> {
        self.validate(None)?;
        let mirostat = self
            .mirostat
            .map(|state| {
                let config = MirostatConfig::new(state.target_surprise, state.learning_rate)?;
                MirostatState::from_parts(config, state.maximum_surprise)
            })
            .transpose()?;
        Ok(GenerationCheckpoint::from_parts(
            self.evaluated_tokens.clone(),
            self.prefill_boundary,
            mirostat,
        ))
    }

    pub fn evaluated_tokens(&self) -> &[u32] {
        &self.evaluated_tokens
    }

    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }

    fn validate(&self, expected_model: Option<&str>) -> Result<(), SessionArchiveError> {
        self.validate_schema()?;
        self.validate_model_hash()?;
        self.validate_expected_model(expected_model)?;
        self.validate_prefill_boundary()?;
        self.validate_mirostat()?;
        Ok(())
    }

    fn validate_schema(&self) -> Result<(), SessionArchiveError> {
        if self.schema_version != SESSION_ARCHIVE_SCHEMA_VERSION {
            return Err(SessionArchiveError::Schema {
                found: self.schema_version,
                expected: SESSION_ARCHIVE_SCHEMA_VERSION,
            });
        }
        Ok(())
    }

    fn validate_model_hash(&self) -> Result<(), SessionArchiveError> {
        if self.model_sha256.len() != 64
            || !self
                .model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(SessionArchiveError::ModelHash);
        }
        Ok(())
    }

    fn validate_expected_model(
        &self,
        expected_model: Option<&str>,
    ) -> Result<(), SessionArchiveError> {
        if let Some(expected) = expected_model {
            if self.model_sha256 != expected {
                return Err(SessionArchiveError::ModelMismatch {
                    found: self.model_sha256.clone(),
                    expected: expected.to_owned(),
                });
            }
        }
        Ok(())
    }

    fn validate_prefill_boundary(&self) -> Result<(), SessionArchiveError> {
        if self.prefill_boundary > self.evaluated_tokens.len() {
            return Err(SessionArchiveError::PrefillBoundary {
                boundary: self.prefill_boundary,
                tokens: self.evaluated_tokens.len(),
            });
        }
        Ok(())
    }

    fn validate_mirostat(&self) -> Result<(), SessionArchiveError> {
        if let Some(state) = self.mirostat {
            let config = MirostatConfig::new(state.target_surprise, state.learning_rate)?;
            MirostatState::from_parts(config, state.maximum_surprise)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn archive_round_trip_preserves_replay_state() {
        let config = MirostatConfig::new(5.0, 0.1).expect("valid configuration");
        let checkpoint = GenerationCheckpoint::from_parts(
            vec![1, 2, 3],
            2,
            Some(MirostatState::from_parts(config, 7.5).expect("valid state")),
        );
        let archive = SessionArchive::new(MODEL, &checkpoint).expect("valid archive");
        let parsed = SessionArchive::from_json(&archive.to_json().expect("JSON"), MODEL)
            .expect("valid archive JSON");

        assert_eq!(parsed.checkpoint().expect("checkpoint"), checkpoint);
        assert_eq!(parsed.sha256().expect("digest").len(), 64);
    }

    #[test]
    fn archive_rejects_a_different_model() {
        let checkpoint = GenerationCheckpoint::from_parts(vec![1], 1, None);
        let archive = SessionArchive::new(MODEL, &checkpoint).expect("valid archive");
        let other = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

        assert!(matches!(
            SessionArchive::from_json(&archive.to_json().expect("JSON"), other),
            Err(SessionArchiveError::ModelMismatch { .. })
        ));
    }

    #[test]
    fn identical_checkpoints_have_one_content_identity() {
        let checkpoint = GenerationCheckpoint::from_parts(vec![4, 5, 6], 3, None);
        let left = SessionArchive::new(MODEL, &checkpoint).expect("left archive");
        let right = SessionArchive::new(MODEL, &checkpoint).expect("right archive");

        assert_eq!(
            left.sha256().expect("left digest"),
            right.sha256().expect("right digest")
        );
    }
}
