use chrono::Utc;
use leone::{KvCacheDtype, Runtime};
use leone_cuda::CudaBackend;
use leone_receipt::{
    sha256_file, write_quality_receipt, ArtifactRef, BatchInvarianceMetric, Corpus, EngineRef,
    Oracle, QualityReceipt, Subject, QUALITY_SCHEMA_VERSION,
};
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug)]
struct VerifyArgs {
    model: PathBuf,
    tokens: PathBuf,
    widths: Vec<usize>,
    depths: Vec<usize>,
    commit: String,
    receipt: bool,
}

pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let tokens = read_tokens(&arguments.tokens)?;
    let required = arguments
        .depths
        .iter()
        .copied()
        .max()
        .and_then(|depth| {
            arguments
                .widths
                .iter()
                .copied()
                .max()
                .and_then(|width| depth.checked_add(width))
        })
        .ok_or_else(|| invalid("verifier token count overflowed"))?;
    if tokens.len() < required {
        return Err(invalid(format!(
            "verifier needs {required} tokens, but {} were provided",
            tokens.len()
        ))
        .into());
    }
    let mut runtime = Runtime::load(CudaBackend::new(0)?, &arguments.model)?;
    let mut compared = 0_u64;
    let mut mismatches = 0_u64;
    for depth in arguments.depths.iter().copied() {
        for width in arguments.widths.iter().copied() {
            let result = runtime.characterize_verify(
                &tokens[..depth],
                &tokens[depth..depth + width],
                KvCacheDtype::F16,
            )?;
            compared = compared
                .checked_add(result.compared_floats as u64)
                .ok_or_else(|| invalid("verifier comparison count overflowed"))?;
            mismatches = mismatches
                .checked_add(result.mismatching_floats as u64)
                .ok_or_else(|| invalid("verifier mismatch count overflowed"))?;
            println!(
                "depth {depth}, width {width}: {} mismatches over {} floats",
                result.mismatching_floats, result.compared_floats
            );
        }
    }
    if mismatches != 0 {
        return Err(invalid(format!(
            "batch invariance failed with {mismatches} mismatching floats"
        ))
        .into());
    }
    if arguments.receipt {
        let model_sha = sha256_file(&arguments.model)?;
        let token_sha = sha256_file(&arguments.tokens)?;
        let receipt = QualityReceipt {
            schema_version: QUALITY_SCHEMA_VERSION,
            receipt_id: Uuid::new_v4(),
            created_utc: Utc::now(),
            corpus: Corpus {
                name: arguments
                    .tokens
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("verifier-tokens")
                    .to_owned(),
                sha256: token_sha,
                n_prompts: arguments.depths.len() as u64,
                n_tokens_scored: arguments
                    .depths
                    .len()
                    .checked_mul(arguments.widths.iter().sum())
                    .ok_or_else(|| invalid("verifier scored token count overflowed"))?
                    as u64,
            },
            oracle: Oracle {
                description: "sequential 1-position decode on the same artifact".to_owned(),
                artifact_sha256: model_sha.clone(),
                engine: EngineRef {
                    name: "leone sequential decode".to_owned(),
                    git_commit: arguments.commit.clone(),
                },
                dtype: "Q4_K_M weights, FP16 KV".to_owned(),
            },
            subject: Subject {
                model_artifact: ArtifactRef {
                    sha256: model_sha,
                    path: arguments.model.display().to_string(),
                },
                logits_artifact: None,
                engine: EngineRef {
                    name: "leone position-major verifier".to_owned(),
                    git_commit: arguments.commit,
                },
            },
            metrics: None,
            batch_invariance: Some(BatchInvarianceMetric {
                definition: "raw f32 logits are equal by to_bits at every compared position"
                    .to_owned(),
                compared_floats: compared,
                mismatching_floats: mismatches,
                widths: arguments
                    .widths
                    .into_iter()
                    .map(|value| value as u64)
                    .collect(),
                context_depths: arguments
                    .depths
                    .into_iter()
                    .map(|value| value as u64)
                    .collect(),
                kv_cache: "f16".to_owned(),
            }),
            sample_count: compared,
        };
        let path = write_quality_receipt(Path::new("receipts"), &receipt)?;
        println!("quality receipt: {}", path.display());
    }
    Ok(())
}

fn parse(arguments: &[String]) -> Result<VerifyArgs, io::Error> {
    let mut model = None;
    let mut tokens = None;
    let mut widths = vec![1, 2, 4, 8];
    let mut depths = vec![32, 512];
    let mut commit = None;
    let mut receipt = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-m" | "--model" => model = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--tokens" => tokens = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--widths" => widths = parse_list(value(arguments, &mut index)?, "width", 8)?,
            "--depths" => depths = parse_list(value(arguments, &mut index)?, "depth", usize::MAX)?,
            "--commit" => commit = Some(value(arguments, &mut index)?.to_owned()),
            "--receipt" => receipt = true,
            value => return Err(invalid(format!("verify argument is invalid: {value}"))),
        }
        index += 1;
    }
    Ok(VerifyArgs {
        model: model.ok_or_else(|| invalid("verify requires -m <gguf>"))?,
        tokens: tokens.ok_or_else(|| invalid("verify requires --tokens <u32le>"))?,
        widths,
        depths,
        commit: commit.ok_or_else(|| invalid("verify requires --commit <git-sha>"))?,
        receipt,
    })
}

fn parse_list(value: &str, field: &str, maximum: usize) -> Result<Vec<usize>, io::Error> {
    let values = value
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .ok()
                .filter(|value| (1..=maximum).contains(value))
                .ok_or_else(|| invalid(format!("{field} is invalid: {item}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(invalid(format!("{field} list is empty")));
    }
    Ok(values)
}

fn read_tokens(path: &Path) -> Result<Vec<u32>, io::Error> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid("token file size must be a multiple of four bytes"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid("command flag is missing its value"))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
