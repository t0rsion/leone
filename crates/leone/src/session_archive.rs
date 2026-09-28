//! Defines the persistent, model-bound replay format for generation sessions.

use crate::{GenerationCheckpoint, MirostatConfig, MirostatState, SamplerError};
use leone_receipt::sha256_bytes;
use serde::{Deserialize, Serialize};
use std::io;
use thiserror::Error;

/// Persistent session archive schema version.
pub const SESSION_ARCHIVE_SCHEMA_VERSION: u32 = 1;

const JSON_ENVELOPE: &str = r#"{"schema_version":,"model_sha256":"","evaluated_tokens":[],"prefill_boundary":,"mirostat":}"#;
const JSON_MIROSTAT: &str = r#"{"target_surprise":,"learning_rate":,"maximum_surprise":}"#;
const MODEL_HASH_BYTES: usize = 64;
const SCHEMA_JSON_BYTES: usize = SESSION_ARCHIVE_SCHEMA_VERSION.ilog10() as usize + 1;
const TOKEN_JSON_BYTES: usize = u32::MAX.ilog10() as usize + 1;
// Sign, 17 significant digits, a point, and `e-308`.
const F64_JSON_BYTES: usize = 24;

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
    #[error("session archive serialization buffer cannot be allocated")]
    Buffer,
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
        Self::from_checkpoint(model_sha256, checkpoint.clone())
    }

    /// Captures an owned replay checkpoint without copying its token vector.
    pub fn from_checkpoint(
        model_sha256: impl Into<String>,
        checkpoint: GenerationCheckpoint,
    ) -> Result<Self, SessionArchiveError> {
        let (evaluated_tokens, prefill_boundary, mirostat) = checkpoint.into_parts();
        let mirostat = mirostat.map(|state| MirostatArchive {
            target_surprise: state.config().target_surprise(),
            learning_rate: state.config().learning_rate(),
            maximum_surprise: state.maximum_surprise(),
        });
        let archive = Self {
            schema_version: SESSION_ARCHIVE_SCHEMA_VERSION,
            model_sha256: model_sha256.into(),
            evaluated_tokens,
            prefill_boundary,
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

    /// Returns an upper bound on the canonical JSON length of a valid archive with
    /// `token_count` tokens.
    ///
    /// Counts each token id and Mirostat float at its widest text. A valid prefill boundary
    /// is at most `token_count`, so its width is that of `token_count`. Returns `None` when
    /// the bound overflows `usize`.
    pub fn max_json_len(token_count: usize) -> Option<usize> {
        let boundary = token_count
            .checked_ilog10()
            .map_or(1, |digits| digits as usize + 1);
        let fixed = JSON_ENVELOPE.len() + JSON_MIROSTAT.len() + MODEL_HASH_BYTES;
        let numbers = SCHEMA_JSON_BYTES + boundary + 3 * F64_JSON_BYTES;
        let tokens = token_count.checked_mul(TOKEN_JSON_BYTES + 1)?;
        fixed.checked_add(numbers)?.checked_add(tokens)
    }

    /// Serializes the canonical archive bytes.
    ///
    /// Reserves `max_json_len` bytes once and never reallocates. The returned capacity is
    /// that bound, not the serialized length.
    pub fn to_json(&self) -> Result<Vec<u8>, SessionArchiveError> {
        self.validate(None)?;
        let bound =
            Self::max_json_len(self.evaluated_tokens.len()).ok_or(SessionArchiveError::Buffer)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(bound)
            .map_err(|_| SessionArchiveError::Buffer)?;
        let mut writer = FixedWriter(buffer);
        serde_json::to_writer(&mut writer, self)?;
        Ok(writer.0)
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

/// Appends within the reserved capacity and fails instead of reallocating.
struct FixedWriter(Vec<u8>);

impl io::Write for FixedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.0.capacity() - self.0.len() {
            return Err(io::ErrorKind::WriteZero.into());
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
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
    fn owned_checkpoint_reuses_its_token_vector() {
        let checkpoint = GenerationCheckpoint::from_parts(vec![1, 2, 3], 2, None);
        let archive = SessionArchive::from_checkpoint(MODEL, checkpoint).expect("archive");
        assert_eq!(archive.evaluated_tokens(), [1, 2, 3]);
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

    fn worst_case_archive(tokens: usize) -> SessionArchive {
        let widest = 2.2250738585072014e-308;
        let config = MirostatConfig::new(widest, widest).expect("valid configuration");
        let mirostat = MirostatState::from_parts(config, widest).expect("valid state");
        let checkpoint =
            GenerationCheckpoint::from_parts(vec![u32::MAX; tokens], tokens, Some(mirostat));
        SessionArchive::from_checkpoint(MODEL, checkpoint).expect("archive")
    }

    #[test]
    fn json_bound_covers_the_widest_archive_within_a_small_margin() {
        for tokens in [0, 1, 9, 10, 1_000] {
            let archive = worst_case_archive(tokens);
            let length = serde_json::to_vec(&archive).expect("JSON").len();
            let bound = SessionArchive::max_json_len(tokens).expect("bound");

            assert!(length <= bound, "{tokens} tokens: {length} > {bound}");
            assert!(bound - length <= 4, "{tokens} tokens: {bound} vs {length}");
        }
    }

    #[test]
    fn serialization_reserves_the_bound_and_keeps_the_canonical_bytes() {
        let tokens = 150_000;
        let ids = (0..tokens as u32).map(|index| 100_000 + index).collect();
        let checkpoint = GenerationCheckpoint::from_parts(ids, tokens, None);
        let archive = SessionArchive::from_checkpoint(MODEL, checkpoint).expect("archive");

        let blob = archive.to_json().expect("JSON");

        assert_eq!(blob, serde_json::to_vec(&archive).expect("reference JSON"));
        assert_eq!(
            blob.capacity(),
            SessionArchive::max_json_len(tokens).unwrap()
        );
    }

    #[test]
    fn json_bound_rejects_overflow() {
        assert_eq!(SessionArchive::max_json_len(usize::MAX), None);
    }

    #[test]
    fn fixed_writer_refuses_to_grow() {
        use std::io::Write;
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(4).expect("reserve");
        let capacity = buffer.capacity();
        let mut writer = FixedWriter(buffer);

        writer.write_all(&vec![0; capacity]).expect("fits");
        assert!(writer.write_all(&[0]).is_err());
        assert_eq!(writer.0.capacity(), capacity);
    }
}
