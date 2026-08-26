#![deny(unsafe_code)]

//! Receipt schemas, validation, and storage.

mod error;
mod response;
mod schema;
mod storage;
mod validate;

pub use error::{Error, Result};
pub use response::{ResponseClaim, ResponseReceipt, SessionReplayRecord};
pub use schema::{
    ArtifactRef, BatchInvarianceMetric, Corpus, DeterminismClaim, DurationSummary, Engine,
    EngineRef, GpuClocksMhz, KldMetric, Machine, Metrics, ModelArtifact, Oracle, PrefillMethod,
    QualityReceipt, QualitySummary, RateSummary, ReductionOrder, Roofline, RuntimeReceipt,
    RuntimeResults, SamplerRecord, SpeculationRecord, Subject, TensorClass, UsableBar, Workload,
};
pub use storage::{write_quality_receipt, write_response_receipt, write_runtime_receipt};
pub use validate::{summarize_duration_samples_ms, summarize_samples, validate, ValidateReceipt};

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// The current runtime receipt schema version.
pub const RUNTIME_SCHEMA_VERSION: u32 = 9;

/// The current quality receipt schema version.
pub const QUALITY_SCHEMA_VERSION: u32 = 3;

/// The current signed response receipt schema version.
pub const RESPONSE_SCHEMA_VERSION: u32 = 1;

/// The required KLD definition for quality receipts.
pub const KLD_DEFINITION: &str =
    "mean over scored positions of KL(P_oracle || P_subject) in nats, full softmax over full vocab";

/// The byte accounting used as the eta denominator in runtime schema v3.
pub const ROOFLINE_DENOMINATOR_DEFINITION: &str = "Quantized matrix bytes and non-quantized weight bytes read once per decode evaluation, KV cache bytes at the average live depth and declared dtype, and one embedding row; excludes re-reads, metadata, and storage padding.";

/// Returns the lowercase SHA-256 digest of a byte slice.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Returns the lowercase SHA-256 digest of a file.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests;
