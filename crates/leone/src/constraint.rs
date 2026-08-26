//! Applies byte-exact output constraints before token sampling.

use crate::{Tokenizer, TokenizerError};
use serde_json::Value;
use thiserror::Error;

/// A supported output-language constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputConstraint {
    /// Emits one JSON object with optional surrounding whitespace.
    JsonObject,
}

/// An error returned when an output constraint cannot admit a token.
#[derive(Debug, Error)]
pub enum ConstraintError {
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error("the output constraint admits no vocabulary token")]
    NoValidToken,
    #[error("the selected token violates the output constraint")]
    InvalidTransition,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct JsonObjectConstraint {
    bytes: Vec<u8>,
}

impl JsonObjectConstraint {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn mask(
        &self,
        logits: &mut [f32],
        tokenizer: &Tokenizer,
    ) -> Result<(), ConstraintError> {
        let eos = tokenizer.eos_token();
        let mut allowed = 0_usize;
        for (index, logit) in logits.iter_mut().enumerate() {
            let token = u32::try_from(index).map_err(|_| ConstraintError::NoValidToken)?;
            let admitted = if Some(token) == eos {
                self.is_complete()
            } else {
                let bytes = tokenizer.token_bytes(token)?;
                !bytes.is_empty() && self.allows_bytes(&bytes)
            };
            if admitted && logit.is_finite() {
                allowed += 1;
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }
        if allowed == 0 {
            Err(ConstraintError::NoValidToken)
        } else {
            Ok(())
        }
    }

    pub(crate) fn accept(
        &mut self,
        token: u32,
        tokenizer: &Tokenizer,
    ) -> Result<(), ConstraintError> {
        if Some(token) == tokenizer.eos_token() {
            return self
                .is_complete()
                .then_some(())
                .ok_or(ConstraintError::InvalidTransition);
        }
        let bytes = tokenizer.token_bytes(token)?;
        if bytes.is_empty() || !self.allows_bytes(&bytes) {
            return Err(ConstraintError::InvalidTransition);
        }
        self.bytes.extend_from_slice(&bytes);
        Ok(())
    }

    fn allows_bytes(&self, bytes: &[u8]) -> bool {
        let mut candidate = Vec::with_capacity(self.bytes.len() + bytes.len());
        candidate.extend_from_slice(&self.bytes);
        candidate.extend_from_slice(bytes);
        valid_object_prefix(&candidate)
    }

    fn is_complete(&self) -> bool {
        matches!(
            serde_json::from_slice::<Value>(&self.bytes),
            Ok(Value::Object(_))
        )
    }
}

fn valid_object_prefix(bytes: &[u8]) -> bool {
    let Some(first) = bytes
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
    else {
        return true;
    };
    if first != b'{' {
        return false;
    }
    match serde_json::from_slice::<Value>(bytes) {
        Ok(Value::Object(_)) => true,
        Ok(_) => false,
        Err(error) => error.is_eof(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_prefixes_remain_admissible() {
        for prefix in [
            b"".as_slice(),
            b"  ",
            b"{",
            br#"{"key""#,
            br#"{"key":[1,true,null,{"nested":"value"}]}"#,
            br#"{"unicode":"\u20ac"}"#,
        ] {
            assert!(valid_object_prefix(prefix), "{prefix:?}");
        }
    }

    #[test]
    fn non_objects_and_broken_syntax_are_rejected() {
        for prefix in [
            b"[".as_slice(),
            b"true",
            br#"{"key":]"#,
            br#"{"key":01"#,
            br#"{"key":"\x""#,
        ] {
            assert!(!valid_object_prefix(prefix), "{prefix:?}");
        }
    }

    #[test]
    fn completion_requires_one_object() {
        let mut constraint = JsonObjectConstraint::new();
        assert!(!constraint.is_complete());
        constraint.bytes.extend_from_slice(br#"{"ok":true}"#);
        assert!(constraint.is_complete());
        constraint.bytes.push(b'x');
        assert!(!constraint.is_complete());
    }
}
