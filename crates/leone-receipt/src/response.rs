use crate::{Error, Result, RESPONSE_SCHEMA_VERSION};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Reuse and replay counts bound into one served response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionReplayRecord {
    pub session_id: String,
    pub reuse_class: String,
    pub cached_tokens: u64,
    pub reused_tokens: u64,
    pub replayed_tokens: u64,
    pub computed_tokens: u64,
}

/// The canonical fields signed for one served response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseClaim {
    pub schema_version: u32,
    pub receipt_id: Uuid,
    pub created_utc: DateTime<Utc>,
    pub engine_version: String,
    pub model_sha256: String,
    pub request_sha256: String,
    pub prompt_tokens_sha256: String,
    pub response_tokens_sha256: String,
    pub transcript_sha256: String,
    pub seed: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub finish_reason: String,
    pub cancelled: bool,
    pub session: SessionReplayRecord,
}

/// An Ed25519 signature and its canonical response claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseReceipt {
    pub claim: ResponseClaim,
    pub public_key_ed25519: String,
    pub signature_ed25519: String,
}

impl ResponseReceipt {
    /// Signs one checked claim with a local Ed25519 key.
    pub fn sign(claim: ResponseClaim, key: &SigningKey) -> Result<Self> {
        validate_claim(&claim)?;
        let payload = serde_json::to_vec(&claim)?;
        let signature = key.sign(&payload);
        Ok(Self {
            claim,
            public_key_ed25519: encode_hex(key.verifying_key().as_bytes()),
            signature_ed25519: encode_hex(&signature.to_bytes()),
        })
    }

    /// Verifies the schema and signature against the embedded public key.
    ///
    /// This proves internal consistency, not signer identity.
    /// [`Self::verify_with_public_key`] checks a signer trusted out of band.
    pub fn verify(&self) -> Result<()> {
        validate_claim(&self.claim)?;
        let public_key = decode_array::<32>(&self.public_key_ed25519, "public key")?;
        let signature = decode_array::<64>(&self.signature_ed25519, "signature")?;
        let key = VerifyingKey::from_bytes(&public_key)
            .map_err(|_| Error::validation("response public key is invalid"))?;
        let signature = Signature::from_bytes(&signature);
        let payload = serde_json::to_vec(&self.claim)?;
        key.verify(&payload, &signature)
            .map_err(|_| Error::validation("response signature does not verify"))
    }

    /// Verifies the receipt against one trusted Ed25519 public key.
    pub fn verify_with_public_key(&self, expected: &[u8; 32]) -> Result<()> {
        self.verify()?;
        let embedded = decode_array::<32>(&self.public_key_ed25519, "public key")?;
        if embedded != *expected {
            return Err(Error::validation(
                "response signer does not match the trusted public key",
            ));
        }
        Ok(())
    }

    /// Parses a receipt and verifies it against its embedded public key.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let receipt: Self = serde_json::from_slice(bytes)?;
        receipt.verify()?;
        Ok(receipt)
    }

    pub fn to_json(&self) -> Result<Vec<u8>> {
        self.verify()?;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

fn validate_claim(claim: &ResponseClaim) -> Result<()> {
    if claim.schema_version != RESPONSE_SCHEMA_VERSION {
        return Err(Error::validation(format!(
            "response schema version is {}, expected {RESPONSE_SCHEMA_VERSION}",
            claim.schema_version
        )));
    }
    for (name, digest) in [
        ("model", &claim.model_sha256),
        ("request", &claim.request_sha256),
        ("prompt tokens", &claim.prompt_tokens_sha256),
        ("response tokens", &claim.response_tokens_sha256),
        ("transcript", &claim.transcript_sha256),
    ] {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::validation(format!("{name} SHA-256 is invalid")));
        }
    }
    if claim.finish_reason.is_empty() {
        return Err(Error::validation("response finish reason is empty"));
    }
    if claim.session.session_id.is_empty() || claim.session.reuse_class.is_empty() {
        return Err(Error::validation("response session record is incomplete"));
    }
    Ok(())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 15)]));
    }
    output
}

fn decode_array<const N: usize>(value: &str, field: &str) -> Result<[u8; N]> {
    if value.len() != N * 2 {
        return Err(Error::validation(format!(
            "response {field} length is invalid"
        )));
    }
    if !value.is_ascii() {
        return Err(Error::validation(format!(
            "response {field} is not hexadecimal"
        )));
    }
    let mut output = [0_u8; N];
    for (index, destination) in output.iter_mut().enumerate() {
        let start = index * 2;
        *destination = u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|_| Error::validation(format!("response {field} is not hexadecimal")))?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_response() -> ResponseReceipt {
        let claim = ResponseClaim {
            schema_version: RESPONSE_SCHEMA_VERSION,
            receipt_id: Uuid::nil(),
            created_utc: "2026-09-01T00:00:00Z".parse().unwrap(),
            engine_version: "fixture".to_owned(),
            model_sha256: "a".repeat(64),
            request_sha256: "b".repeat(64),
            prompt_tokens_sha256: "c".repeat(64),
            response_tokens_sha256: "d".repeat(64),
            transcript_sha256: "e".repeat(64),
            seed: 0,
            prompt_tokens: 1,
            generated_tokens: 1,
            finish_reason: "length".to_owned(),
            cancelled: false,
            session: SessionReplayRecord {
                session_id: "fixture".to_owned(),
                reuse_class: "cold".to_owned(),
                cached_tokens: 0,
                reused_tokens: 0,
                replayed_tokens: 0,
                computed_tokens: 1,
            },
        };
        ResponseReceipt::sign(claim, &SigningKey::from_bytes(&[7; 32])).unwrap()
    }

    #[test]
    fn response_json_rejects_non_ascii_hex_at_each_byte_offset() {
        let receipt = signed_response();
        let bytes = receipt.to_json().unwrap();
        assert_eq!(ResponseReceipt::from_json(&bytes).unwrap(), receipt);
        let original = serde_json::to_value(receipt).unwrap();
        for (field, length) in [("public_key_ed25519", 64), ("signature_ed25519", 128)] {
            for offset in 0..=length - 3 {
                let mut malformed = original.clone();
                malformed[field] = format!(
                    "{}\u{20ac}{}",
                    "a".repeat(offset),
                    "a".repeat(length - offset - 3)
                )
                .into();
                let bytes = serde_json::to_vec(&malformed).unwrap();
                let error = ResponseReceipt::from_json(&bytes).unwrap_err();
                assert!(error.to_string().contains("is not hexadecimal"));
            }
        }
    }
}
