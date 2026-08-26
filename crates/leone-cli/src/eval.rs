use leone::{Backend, CpuBackend, KvCacheDtype, Runtime, RuntimeError, Tokenizer};
use leone_cuda::CudaBackend;
use leone_gguf::Gguf;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const DEFAULT_WINDOW_TOKENS: usize = 512;

#[derive(Debug, Clone, Copy)]
enum BackendChoice {
    Cuda,
    Cpu,
    CpuQ8_1,
}

#[derive(Debug)]
enum TokenSource {
    Corpus { path: PathBuf, limit: usize },
    Binary(PathBuf),
}

#[derive(Debug, Clone, Copy)]
enum PositionScope {
    All,
    First(usize),
}

#[derive(Debug)]
struct EvalArgs {
    model: PathBuf,
    source: TokenSource,
    tokens_out: Option<PathBuf>,
    logits: Option<PathBuf>,
    backend: BackendChoice,
    window_tokens: usize,
    positions: PositionScope,
    prefill_chunk: Option<usize>,
}

pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let mut tokens = match &arguments.source {
        TokenSource::Corpus { path, limit } => tokenize(&arguments.model, path, *limit)?,
        TokenSource::Binary(path) => read_tokens(path)?,
    };
    if let PositionScope::First(positions) = arguments.positions {
        let required = positions
            .checked_add(1)
            .ok_or_else(|| invalid("position limit overflowed"))?;
        if tokens.len() < required {
            return Err(invalid(format!(
                "position limit {positions} needs {required} tokens, but the file has {}",
                tokens.len()
            ))
            .into());
        }
        tokens.truncate(required);
    }
    if let Some(path) = &arguments.tokens_out {
        write_tokens(path, &tokens)?;
        println!("tokens: {} ({})", path.display(), tokens.len());
    }
    if let Some(path) = &arguments.logits {
        match arguments.backend {
            BackendChoice::Cuda => dump_logits(CudaBackend::new(0)?, &arguments, &tokens, path)?,
            BackendChoice::Cpu => dump_logits(CpuBackend::new(), &arguments, &tokens, path)?,
            BackendChoice::CpuQ8_1 => dump_logits(
                CpuBackend::with_q8_1_activations(),
                &arguments,
                &tokens,
                path,
            )?,
        }
    }
    Ok(())
}

fn parse(arguments: &[String]) -> Result<EvalArgs, io::Error> {
    let mut model = None;
    let mut corpus = None;
    let mut tokens = None;
    let mut token_limit = None;
    let mut tokens_out = None;
    let mut logits = None;
    let mut backend = BackendChoice::Cuda;
    let mut window_tokens = DEFAULT_WINDOW_TOKENS;
    let mut position_limit = None;
    let mut prefill_chunk = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-m" | "--model" => model = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--corpus" => corpus = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--tokens" => tokens = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--token-limit" => {
                token_limit = Some(parse_positive(
                    value(arguments, &mut index)?,
                    "token limit",
                )?);
            }
            "--tokens-out" => tokens_out = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--logits" => logits = Some(PathBuf::from(value(arguments, &mut index)?)),
            "--backend" => {
                backend = match value(arguments, &mut index)? {
                    "cuda" => BackendChoice::Cuda,
                    "cpu" => BackendChoice::Cpu,
                    "cpu-q8_1" => BackendChoice::CpuQ8_1,
                    value => return Err(invalid(format!("backend is invalid: {value}"))),
                };
            }
            "--window" => {
                window_tokens = parse_positive(value(arguments, &mut index)?, "window")?;
            }
            "--position-limit" => {
                position_limit = Some(parse_positive(
                    value(arguments, &mut index)?,
                    "position limit",
                )?);
            }
            "--prefill-chunk" => {
                prefill_chunk = Some(parse_positive(
                    value(arguments, &mut index)?,
                    "prefill chunk",
                )?);
            }
            value => return Err(invalid(format!("eval argument is invalid: {value}"))),
        }
        index += 1;
    }
    let source = match (corpus, tokens, token_limit) {
        (Some(path), None, Some(limit)) => TokenSource::Corpus { path, limit },
        (Some(_), None, None) => return Err(invalid("eval --corpus requires --token-limit")),
        (None, Some(path), None) => TokenSource::Binary(path),
        (None, Some(_), Some(_)) => {
            return Err(invalid("eval --token-limit is only valid with --corpus"));
        }
        (Some(_), Some(_), _) => {
            return Err(invalid(
                "eval accepts either --corpus or --tokens, not both",
            ));
        }
        (None, None, _) => return Err(invalid("eval requires --corpus or --tokens")),
    };
    if tokens_out.is_none() && logits.is_none() {
        return Err(invalid("eval requires --tokens-out or --logits"));
    }
    if matches!(&source, TokenSource::Corpus { .. }) && position_limit.is_some() {
        return Err(invalid(
            "eval --position-limit is only valid with a binary token input",
        ));
    }
    Ok(EvalArgs {
        model: model.ok_or_else(|| invalid("eval requires -m <gguf>"))?,
        source,
        tokens_out,
        logits,
        backend,
        window_tokens,
        positions: position_limit.map_or(PositionScope::All, PositionScope::First),
        prefill_chunk,
    })
}

fn tokenize(model: &Path, corpus: &Path, limit: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    let gguf = Gguf::open(model)?;
    let tokenizer = Tokenizer::from_metadata(gguf.metadata())?;
    let encoded = tokenizer.encode(&fs::read_to_string(corpus)?)?;
    if encoded.len() < limit {
        return Err(invalid(format!(
            "corpus produced {} tokens, fewer than the required {limit}",
            encoded.len()
        ))
        .into());
    }
    Ok(encoded.into_iter().take(limit).collect())
}

fn dump_logits<B: Backend>(
    backend: B,
    arguments: &EvalArgs,
    tokens: &[u32],
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut output = BufWriter::new(File::create(path)?);
    let mut runtime = Runtime::load(backend, &arguments.model)?;
    let mut write_row = |position: usize, logits: &[f32]| {
        for logit in logits {
            output
                .write_all(&logit.to_le_bytes())
                .map_err(RuntimeError::logit_callback)?;
        }
        if position.is_multiple_of(128) {
            eprintln!("scored position {position}/{}", tokens.len() - 1);
        }
        Ok(())
    };
    let scored = if let Some(chunk) = arguments.prefill_chunk {
        runtime.evaluate_logits_chunked(
            tokens,
            arguments.window_tokens,
            chunk,
            KvCacheDtype::F16,
            &mut write_row,
        )?
    } else {
        runtime.evaluate_logits(
            tokens,
            arguments.window_tokens,
            KvCacheDtype::F16,
            &mut write_row,
        )?
    };
    output.flush()?;
    println!(
        "logits: {} ({scored} positions, vocab {})",
        path.display(),
        runtime.model().config().vocab_size
    );
    Ok(())
}

fn read_tokens(path: &Path) -> Result<Vec<u32>, io::Error> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid("token file size must be a multiple of four bytes"));
    }
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();
    if tokens.len() < 2 {
        return Err(invalid("token file must contain at least two tokens"));
    }
    Ok(tokens)
}

fn write_tokens(path: &Path, tokens: &[u32]) -> Result<(), io::Error> {
    let mut output = BufWriter::new(File::create(path)?);
    for token in tokens {
        output.write_all(&token.to_le_bytes())?;
    }
    output.flush()
}

fn value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid("command flag is missing its value"))
}

fn parse_positive(value: &str, field: &str) -> Result<usize, io::Error> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{field} must be a positive integer")))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_source_requires_an_explicit_limit() {
        let arguments = vec![
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--corpus".to_owned(),
            "corpus.txt".to_owned(),
            "--tokens-out".to_owned(),
            "tokens.bin".to_owned(),
        ];
        assert!(parse(&arguments).is_err());
    }

    #[test]
    fn token_source_rejects_a_limit() {
        let arguments = vec![
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "--tokens".to_owned(),
            "tokens.bin".to_owned(),
            "--token-limit".to_owned(),
            "64".to_owned(),
            "--logits".to_owned(),
            "logits.f32".to_owned(),
        ];
        assert!(parse(&arguments).is_err());
    }
}
