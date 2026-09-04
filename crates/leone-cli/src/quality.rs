use chrono::Utc;
use leone_gguf::model::ModelConfig;
use leone_gguf::Gguf;
use leone_receipt::{
    sha256_file, write_quality_receipt, ArtifactRef, Corpus, EngineRef, KldMetric, Metrics, Oracle,
    QualityReceipt, Subject, KLD_DEFINITION, QUALITY_SCHEMA_VERSION,
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug)]
struct QualityArgs {
    oracle: PathBuf,
    subject: PathBuf,
    corpus: PathBuf,
    tokens: PathBuf,
    oracle_model: PathBuf,
    subject_model: PathBuf,
    oracle_engine: String,
    oracle_commit: String,
    oracle_dtype: String,
    subject_engine: String,
    subject_commit: String,
    receipt: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct KldStats {
    mean: f64,
    p50: f64,
    p99: f64,
    max: f64,
    top1_agreement: f64,
}

pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let (sample_count, stats) = measure_quality(&arguments)?;
    print_quality_stats(sample_count, stats);
    if arguments.receipt {
        write_receipt(arguments, sample_count, stats)?;
    }
    Ok(())
}

fn measure_quality(arguments: &QualityArgs) -> Result<(usize, KldStats), Box<dyn Error>> {
    let tokens = read_tokens(&arguments.tokens)?;
    let sample_count = tokens
        .len()
        .checked_sub(1)
        .ok_or_else(|| invalid("quality needs at least two tokens"))?;
    let oracle_model = load_model_config(&arguments.oracle_model)?;
    let subject_model = load_model_config(&arguments.subject_model)?;
    if oracle_model.vocab_size != subject_model.vocab_size {
        return Err(invalid(format!(
            "oracle vocab size {} differs from subject vocab size {}",
            oracle_model.vocab_size, subject_model.vocab_size
        ))
        .into());
    }
    let vocab_size = usize::try_from(subject_model.vocab_size)?;
    let stats = measure(
        &arguments.oracle,
        &arguments.subject,
        sample_count,
        vocab_size,
    )?;
    Ok((sample_count, stats))
}

fn print_quality_stats(sample_count: usize, stats: KldStats) {
    println!("samples: {sample_count}");
    println!("KLD mean: {:.9} nats", stats.mean);
    println!("KLD p50:  {:.9} nats", stats.p50);
    println!("KLD p99:  {:.9} nats", stats.p99);
    println!("KLD max:  {:.9} nats", stats.max);
    println!("top1 agreement: {:.9}", stats.top1_agreement);
}

fn load_model_config(path: &Path) -> Result<ModelConfig, Box<dyn Error>> {
    let gguf = Gguf::open(path)?;
    Ok(ModelConfig::from_metadata(gguf.metadata())?)
}

fn write_receipt(
    arguments: QualityArgs,
    sample_count: usize,
    stats: KldStats,
) -> Result<(), Box<dyn Error>> {
    let root = std::env::current_dir()?;
    let oracle_model_sha256 = sha256_file(&arguments.oracle_model)?;
    let receipt = QualityReceipt {
        schema_version: QUALITY_SCHEMA_VERSION,
        receipt_id: Uuid::new_v4(),
        created_utc: Utc::now(),
        corpus: Corpus {
            name: "leone-v0.1-research-corpus".to_owned(),
            sha256: sha256_file(&arguments.corpus)?,
            n_prompts: 1,
            n_tokens_scored: u64::try_from(sample_count)?,
        },
        oracle: Oracle {
            description: format!(
                "{} {} execution of {} (model SHA-256 {oracle_model_sha256}); full-vocabulary logits at {}",
                arguments.oracle_engine,
                arguments.oracle_dtype,
                arguments.oracle_model.display(),
                arguments.oracle.display(),
            ),
            artifact_sha256: sha256_file(&arguments.oracle)?,
            engine: EngineRef {
                name: arguments.oracle_engine,
                git_commit: arguments.oracle_commit,
            },
            dtype: arguments.oracle_dtype,
        },
        subject: Subject {
            model_artifact: ArtifactRef {
                sha256: sha256_file(&arguments.subject_model)?,
                path: arguments.subject_model.display().to_string(),
            },
            logits_artifact: Some(ArtifactRef {
                sha256: sha256_file(&arguments.subject)?,
                path: arguments.subject.display().to_string(),
            }),
            engine: EngineRef {
                name: arguments.subject_engine,
                git_commit: arguments.subject_commit,
            },
        },
        metrics: Some(Metrics {
            kld: KldMetric {
                mean: stats.mean,
                p50: Some(stats.p50),
                p99: stats.p99,
                max: Some(stats.max),
                definition: KLD_DEFINITION.to_owned(),
            },
            top1_agreement: stats.top1_agreement,
        }),
        batch_invariance: None,
        sample_count: u64::try_from(sample_count)?,
    };
    let path = write_quality_receipt(root.join("receipts"), &receipt)?;
    println!("receipt: {}", path.display());
    println!("quality receipt: {}", receipt.receipt_id);
    Ok(())
}

fn parse(arguments: &[String]) -> Result<QualityArgs, io::Error> {
    let mut parsed = QualityBuilder::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_artifact_option(flag, arguments, &mut index, &mut parsed)?
            || parse_model_option(flag, arguments, &mut index, &mut parsed)?
            || parse_engine_option(flag, arguments, &mut index, &mut parsed)?
            || parse_quality_flag(flag, &mut parsed);
        if !handled {
            return Err(invalid(format!("quality argument is invalid: {flag}")));
        }
        index += 1;
    }
    parsed.finish()
}

#[derive(Default)]
struct QualityBuilder {
    oracle: Option<PathBuf>,
    subject: Option<PathBuf>,
    corpus: Option<PathBuf>,
    tokens: Option<PathBuf>,
    oracle_model: Option<PathBuf>,
    subject_model: Option<PathBuf>,
    oracle_engine: Option<String>,
    oracle_commit: Option<String>,
    oracle_dtype: Option<String>,
    subject_engine: Option<String>,
    subject_commit: Option<String>,
    receipt: bool,
}

impl QualityBuilder {
    fn finish(self) -> Result<QualityArgs, io::Error> {
        let QualityBuilder {
            oracle,
            subject,
            corpus,
            tokens,
            oracle_model,
            subject_model,
            oracle_engine,
            oracle_commit,
            oracle_dtype,
            subject_engine,
            subject_commit,
            receipt,
        } = self;
        let paths = quality_paths(oracle, subject, corpus, tokens, oracle_model, subject_model)?;
        let engines = quality_engines(
            oracle_engine,
            oracle_commit,
            oracle_dtype,
            subject_engine,
            subject_commit,
        )?;
        Ok(QualityArgs {
            oracle: paths.oracle,
            subject: paths.subject,
            corpus: paths.corpus,
            tokens: paths.tokens,
            oracle_model: paths.oracle_model,
            subject_model: paths.subject_model,
            oracle_engine: engines.oracle_engine,
            oracle_commit: engines.oracle_commit,
            oracle_dtype: engines.oracle_dtype,
            subject_engine: engines.subject_engine,
            subject_commit: engines.subject_commit,
            receipt,
        })
    }
}

struct QualityPaths {
    oracle: PathBuf,
    subject: PathBuf,
    corpus: PathBuf,
    tokens: PathBuf,
    oracle_model: PathBuf,
    subject_model: PathBuf,
}

fn quality_paths(
    oracle: Option<PathBuf>,
    subject: Option<PathBuf>,
    corpus: Option<PathBuf>,
    tokens: Option<PathBuf>,
    oracle_model: Option<PathBuf>,
    subject_model: Option<PathBuf>,
) -> Result<QualityPaths, io::Error> {
    Ok(QualityPaths {
        oracle: required(oracle, "--oracle")?,
        subject: required(subject, "--subject")?,
        corpus: required(corpus, "--corpus")?,
        tokens: required(tokens, "--tokens")?,
        oracle_model: required(oracle_model, "--oracle-model")?,
        subject_model: required(subject_model, "--subject-model")?,
    })
}

struct QualityEngines {
    oracle_engine: String,
    oracle_commit: String,
    oracle_dtype: String,
    subject_engine: String,
    subject_commit: String,
}

fn quality_engines(
    oracle_engine: Option<String>,
    oracle_commit: Option<String>,
    oracle_dtype: Option<String>,
    subject_engine: Option<String>,
    subject_commit: Option<String>,
) -> Result<QualityEngines, io::Error> {
    Ok(QualityEngines {
        oracle_engine: required(oracle_engine, "--oracle-engine")?,
        oracle_commit: required(oracle_commit, "--oracle-commit")?,
        oracle_dtype: required(oracle_dtype, "--oracle-dtype")?,
        subject_engine: required(subject_engine, "--subject-engine")?,
        subject_commit: required(subject_commit, "--subject-commit")?,
    })
}

fn parse_artifact_option(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut QualityBuilder,
) -> Result<bool, io::Error> {
    let target = match flag {
        "--oracle" => &mut parsed.oracle,
        "--subject" => &mut parsed.subject,
        "--corpus" => &mut parsed.corpus,
        "--tokens" => &mut parsed.tokens,
        _ => return Ok(false),
    };
    *target = Some(PathBuf::from(value(arguments, index)?));
    Ok(true)
}

fn parse_model_option(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut QualityBuilder,
) -> Result<bool, io::Error> {
    let target = match flag {
        "--oracle-model" => &mut parsed.oracle_model,
        "--subject-model" => &mut parsed.subject_model,
        _ => return Ok(false),
    };
    *target = Some(PathBuf::from(value(arguments, index)?));
    Ok(true)
}

fn parse_engine_option(
    flag: &str,
    arguments: &[String],
    index: &mut usize,
    parsed: &mut QualityBuilder,
) -> Result<bool, io::Error> {
    let target = match flag {
        "--oracle-engine" => &mut parsed.oracle_engine,
        "--oracle-commit" => &mut parsed.oracle_commit,
        "--oracle-dtype" => &mut parsed.oracle_dtype,
        "--subject-engine" => &mut parsed.subject_engine,
        "--subject-commit" => &mut parsed.subject_commit,
        _ => return Ok(false),
    };
    *target = Some(value(arguments, index)?.to_owned());
    Ok(true)
}

fn parse_quality_flag(flag: &str, parsed: &mut QualityBuilder) -> bool {
    if flag == "--receipt" {
        parsed.receipt = true;
        true
    } else {
        false
    }
}

fn measure(
    oracle_path: &Path,
    subject_path: &Path,
    sample_count: usize,
    vocab_size: usize,
) -> Result<KldStats, io::Error> {
    let row_bytes = measurement_row_bytes(sample_count, vocab_size)?;
    let expected_bytes = expected_artifact_bytes(sample_count, row_bytes)?;
    check_artifact_size(oracle_path, expected_bytes, "oracle")?;
    check_artifact_size(subject_path, expected_bytes, "subject")?;
    let (mut values, top1_matches) = measure_rows(
        oracle_path,
        subject_path,
        sample_count,
        vocab_size,
        row_bytes,
    )?;
    values.sort_by(f64::total_cmp);
    let sum: f64 = values.iter().sum();
    Ok(KldStats {
        mean: sum / sample_count as f64,
        p50: percentile(&values, 0.50),
        p99: percentile(&values, 0.99),
        max: values[sample_count - 1],
        top1_agreement: top1_matches as f64 / sample_count as f64,
    })
}

fn measurement_row_bytes(sample_count: usize, vocab_size: usize) -> Result<usize, io::Error> {
    if sample_count == 0 || vocab_size == 0 {
        return Err(invalid("KLD dimensions must be nonzero"));
    }
    vocab_size
        .checked_mul(4)
        .ok_or_else(|| invalid("logit row size overflowed"))
}

fn expected_artifact_bytes(sample_count: usize, row_bytes: usize) -> Result<u64, io::Error> {
    sample_count
        .checked_mul(row_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("logit artifact size overflowed"))
}

fn measure_rows(
    oracle_path: &Path,
    subject_path: &Path,
    sample_count: usize,
    vocab_size: usize,
    row_bytes: usize,
) -> Result<(Vec<f64>, usize), io::Error> {
    let mut oracle = BufReader::new(File::open(oracle_path)?);
    let mut subject = BufReader::new(File::open(subject_path)?);
    let mut oracle_bytes = vec![0_u8; row_bytes];
    let mut subject_bytes = vec![0_u8; row_bytes];
    let mut oracle_logits = vec![0.0_f32; vocab_size];
    let mut subject_logits = vec![0.0_f32; vocab_size];
    let mut values = Vec::with_capacity(sample_count);
    let mut top1_matches = 0_usize;
    for position in 0..sample_count {
        let (kld, top1_match) = measure_row(
            &mut oracle,
            &mut subject,
            &mut oracle_bytes,
            &mut subject_bytes,
            &mut oracle_logits,
            &mut subject_logits,
        )?;
        values.push(kld);
        if top1_match {
            top1_matches += 1;
        }
        if (position + 1) % 128 == 0 {
            eprintln!(
                "computed KLD for {}/{} positions",
                position + 1,
                sample_count
            );
        }
    }
    Ok((values, top1_matches))
}

fn measure_row(
    oracle: &mut BufReader<File>,
    subject: &mut BufReader<File>,
    oracle_bytes: &mut [u8],
    subject_bytes: &mut [u8],
    oracle_logits: &mut [f32],
    subject_logits: &mut [f32],
) -> Result<(f64, bool), io::Error> {
    oracle.read_exact(oracle_bytes)?;
    subject.read_exact(subject_bytes)?;
    decode_f32(oracle_bytes, oracle_logits);
    decode_f32(subject_bytes, subject_logits);
    let kld = position_kld(oracle_logits, subject_logits)?;
    let top1_match = argmax(oracle_logits)? == argmax(subject_logits)?;
    Ok((kld, top1_match))
}

fn position_kld(oracle: &[f32], subject: &[f32]) -> Result<f64, io::Error> {
    let oracle_lse = logsumexp(oracle)?;
    let subject_lse = logsumexp(subject)?;
    let mut kld = 0.0_f64;
    for (&oracle_logit, &subject_logit) in oracle.iter().zip(subject) {
        if oracle_logit == f32::NEG_INFINITY {
            continue;
        }
        if !oracle_logit.is_finite() || !subject_logit.is_finite() {
            return Err(invalid("a scored logit is not finite"));
        }
        let log_p = f64::from(oracle_logit) - oracle_lse;
        let log_q = f64::from(subject_logit) - subject_lse;
        kld += log_p.exp() * (log_p - log_q);
    }
    if (-1e-12..=0.0).contains(&kld) {
        Ok(0.0)
    } else if !kld.is_finite() || kld < 0.0 {
        Err(invalid(format!("computed invalid KLD value {kld}")))
    } else {
        Ok(kld)
    }
}

fn logsumexp(values: &[f32]) -> Result<f64, io::Error> {
    if values.iter().any(|value| value.is_nan()) {
        return Err(invalid("logit row contains NaN"));
    }
    let maximum = values
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .ok_or_else(|| invalid("logit row is empty"))?;
    if !maximum.is_finite() {
        return Err(invalid("logit row has no finite value"));
    }
    let sum: f64 = values
        .iter()
        .map(|value| (f64::from(*value) - f64::from(maximum)).exp())
        .sum();
    Ok(f64::from(maximum) + sum.ln())
}

fn argmax(values: &[f32]) -> Result<usize, io::Error> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| invalid("logit row contains only NaN"))
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let rank = percentile * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn decode_f32(bytes: &[u8], output: &mut [f32]) {
    for (encoded, value) in bytes.chunks_exact(4).zip(output) {
        *value = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
    }
}

fn check_artifact_size(path: &Path, expected: u64, name: &str) -> Result<(), io::Error> {
    let actual = fs::metadata(path)?.len();
    if actual != expected {
        return Err(invalid(format!(
            "{name} artifact has {actual} bytes, expected {expected}"
        )));
    }
    Ok(())
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

fn required<T>(value: Option<T>, flag: &str) -> Result<T, io::Error> {
    value.ok_or_else(|| invalid(format!("quality requires {flag}")))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn artifact(path: &Path, rows: &[[f32; 3]]) {
        let mut file = File::create(path).unwrap();
        for row in rows {
            for value in row {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn self_kld_is_exactly_zero() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("self.f32");
        artifact(&path, &[[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]);
        let stats = measure(&path, &path, 2, 3).unwrap();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.top1_agreement, 1.0);
    }

    #[test]
    fn constant_logit_shift_does_not_change_distribution() {
        let directory = tempfile::tempdir().unwrap();
        let oracle = directory.path().join("oracle.f32");
        let subject = directory.path().join("subject.f32");
        artifact(&oracle, &[[1.0, 2.0, 3.0]]);
        artifact(&subject, &[[11.0, 12.0, 13.0]]);
        let stats = measure(&oracle, &subject, 1, 3).unwrap();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.top1_agreement, 1.0);
    }

    #[test]
    fn artifact_shape_must_match_tokens_and_vocab() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("short.f32");
        artifact(&path, &[[1.0, 2.0, 3.0]]);
        assert!(measure(&path, &path, 2, 3).is_err());
    }
}
