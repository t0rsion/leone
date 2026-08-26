use crate::schema::filename_timestamp;
use crate::{QualityReceipt, ResponseReceipt, Result, RuntimeReceipt};
use chrono::Datelike;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INDEX_HEADER: &str =
    "# Receipt index\n\n| Date | Kind | Receipt ID | Headline |\n|---|---|---|---|\n";

/// Writes a runtime receipt and appends its decode result to `INDEX.md`.
pub fn write_runtime_receipt(
    receipts_dir: impl AsRef<Path>,
    receipt: &RuntimeReceipt,
) -> Result<PathBuf> {
    let headline = format!("{:.3} decode tok/s", receipt.results.decode_tok_s.median);
    write_receipt(
        receipts_dir.as_ref(),
        "runtime",
        receipt.receipt_id,
        receipt.created_utc,
        &headline,
        &receipt.to_json()?,
    )
}

/// Writes a quality receipt and appends its mean KLD to `INDEX.md`.
pub fn write_quality_receipt(
    receipts_dir: impl AsRef<Path>,
    receipt: &QualityReceipt,
) -> Result<PathBuf> {
    let headline = match (&receipt.metrics, &receipt.batch_invariance) {
        (Some(metrics), None) => format!("{:.6} mean KLD nats", metrics.kld.mean),
        (None, Some(metric)) => format!(
            "{} mismatching floats over {} comparisons",
            metric.mismatching_floats, metric.compared_floats
        ),
        _ => "invalid quality metric selection".to_owned(),
    };
    write_receipt(
        receipts_dir.as_ref(),
        "quality",
        receipt.receipt_id,
        receipt.created_utc,
        &headline,
        &receipt.to_json()?,
    )
}

/// Writes a signed served-response receipt and appends it to `INDEX.md`.
pub fn write_response_receipt(
    receipts_dir: impl AsRef<Path>,
    receipt: &ResponseReceipt,
) -> Result<PathBuf> {
    write_receipt(
        receipts_dir.as_ref(),
        "response",
        receipt.claim.receipt_id,
        receipt.claim.created_utc,
        &format!("{} served tokens", receipt.claim.generated_tokens),
        &receipt.to_json()?,
    )
}

fn write_receipt(
    receipts_dir: &Path,
    kind: &str,
    receipt_id: uuid::Uuid,
    created_utc: chrono::DateTime<chrono::Utc>,
    headline: &str,
    json: &[u8],
) -> Result<PathBuf> {
    fs::create_dir_all(receipts_dir)?;
    let short_id = &receipt_id.simple().to_string()[..8];
    let filename = format!("{}-{kind}-{short_id}.json", filename_timestamp(created_utc));
    let path = receipts_dir.join(filename);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    file.write_all(json)?;

    let index_path = receipts_dir.join("INDEX.md");
    let needs_header = index_path
        .metadata()
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(true);
    let mut index = OpenOptions::new()
        .create(true)
        .append(true)
        .open(index_path)?;
    if needs_header {
        index.write_all(INDEX_HEADER.as_bytes())?;
    }
    writeln!(
        index,
        "| {:04}-{:02}-{:02} | {kind} | `{receipt_id}` | {headline} |",
        created_utc.year(),
        created_utc.month(),
        created_utc.day()
    )?;
    Ok(path)
}
