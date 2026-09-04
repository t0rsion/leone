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

#[derive(Debug, Clone)]
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
    let tokens = load_verify_tokens(&arguments)?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, &arguments.model)?;
    let (compared, mismatches) = compare_widths(&mut runtime, &tokens, &arguments)?;
    ensure_invariance(mismatches)?;
    write_receipt_if_requested(&arguments, compared, mismatches)?;
    Ok(())
}

fn load_verify_tokens(arguments: &VerifyArgs) -> Result<Vec<u32>, Box<dyn Error>> {
    let tokens = read_tokens(&arguments.tokens)?;
    let required = required_tokens(arguments)?;
    if tokens.len() < required {
        return Err(invalid(format!(
            "verifier needs {required} tokens, but {} were provided",
            tokens.len()
        ))
        .into());
    }
    Ok(tokens)
}

fn ensure_invariance(mismatches: u64) -> Result<(), Box<dyn Error>> {
    if mismatches != 0 {
        return Err(invalid(format!(
            "batch invariance failed with {mismatches} mismatching floats"
        ))
        .into());
    }
    Ok(())
}

fn write_receipt_if_requested(
    arguments: &VerifyArgs,
    compared: u64,
    mismatches: u64,
) -> Result<(), Box<dyn Error>> {
    if arguments.receipt {
        write_receipt(arguments.clone(), compared, mismatches)?;
    }
    Ok(())
}

fn required_tokens(arguments: &VerifyArgs) -> Result<usize, io::Error> {
    arguments
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
        .ok_or_else(|| invalid("verifier token count overflowed"))
}

fn compare_widths(
    runtime: &mut Runtime<CudaBackend>,
    tokens: &[u32],
    arguments: &VerifyArgs,
) -> Result<(u64, u64), io::Error> {
    let mut compared = 0_u64;
    let mut mismatches = 0_u64;
    for depth in arguments.depths.iter().copied() {
        for width in arguments.widths.iter().copied() {
            let result = runtime
                .characterize_verify(
                    &tokens[..depth],
                    &tokens[depth..depth + width],
                    KvCacheDtype::F16,
                )
                .map_err(|error| invalid(error.to_string()))?;
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
    Ok((compared, mismatches))
}

fn write_receipt(
    arguments: VerifyArgs,
    compared: u64,
    mismatches: u64,
) -> Result<(), Box<dyn Error>> {
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
            definition: "raw f32 logits are equal by to_bits at every compared position".to_owned(),
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
    Ok(())
}

fn parse(arguments: &[String]) -> Result<VerifyArgs, io::Error> {
    let mut parsed = VerifyBuilder::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_paths(flag, arguments, &mut index, &mut parsed)?
            || parse_ranges(flag, arguments, &mut index, &mut parsed)?
            || parse_commit(flag, arguments, &mut index, &mut parsed)?;
        if !handled {
            return Err(invalid(format!("verify argument is invalid: {flag}")));
        }
        index += 1;
    }
    parsed.finish()
}

#[derive(Default)]
struct VerifyBuilder {
    model: Option<PathBuf>,
    tokens: Option<PathBuf>,
    widths: Option<Vec<usize>>,
    depths: Option<Vec<usize>>,
    commit: Option<String>,
    receipt: bool,
}

impl VerifyBuilder {
    fn finish(self) -> Result<VerifyArgs, io::Error> {
        Ok(VerifyArgs {
            model: self
                .model
                .ok_or_else(|| invalid("verify requires -m <gguf>"))?,
            tokens: self
                .tokens
                .ok_or_else(|| invalid("verify requires --tokens <u32le>"))?,
            widths: self.widths.unwrap_or_else(|| vec![1, 2, 4, 8]),
            depths: self.depths.unwrap_or_else(|| vec![32, 512]),
            commit: self
                .commit
                .ok_or_else(|| invalid("verify requires --commit <git-sha>"))?,
            receipt: self.receipt,
        })
    }
}

fn parse_paths(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut VerifyBuilder,
) -> Result<bool, io::Error> {
    match flag {
        "-m" | "--model" => {
            parsed.model = Some(PathBuf::from(value(arguments, index)?));
            Ok(true)
        }
        "--tokens" => {
            parsed.tokens = Some(PathBuf::from(value(arguments, index)?));
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_ranges(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut VerifyBuilder,
) -> Result<bool, io::Error> {
    match flag {
        "--widths" => {
            parsed.widths = Some(parse_list(value(arguments, index)?, "width", 8)?);
            Ok(true)
        }
        "--depths" => {
            parsed.depths = Some(parse_list(value(arguments, index)?, "depth", usize::MAX)?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_commit(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut VerifyBuilder,
) -> Result<bool, io::Error> {
    match flag {
        "--commit" => {
            parsed.commit = Some(value(arguments, index)?.to_owned());
            Ok(true)
        }
        "--receipt" => {
            parsed.receipt = true;
            Ok(true)
        }
        _ => Ok(false),
    }
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
