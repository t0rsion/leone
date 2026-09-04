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
    let tokens = load_tokens(&arguments)?;
    if let Some(path) = &arguments.tokens_out {
        write_token_output(path, &tokens)?;
    }
    if let Some(path) = &arguments.logits {
        dump_logits_for_backend(&arguments, &tokens, path)?;
    }
    Ok(())
}

fn load_tokens(arguments: &EvalArgs) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut tokens = match &arguments.source {
        TokenSource::Corpus { path, limit } => tokenize(&arguments.model, path, *limit)?,
        TokenSource::Binary(path) => read_tokens(path)?,
    };
    apply_position_limit(&mut tokens, arguments.positions)?;
    Ok(tokens)
}

fn apply_position_limit(tokens: &mut Vec<u32>, scope: PositionScope) -> Result<(), io::Error> {
    let PositionScope::First(positions) = scope else {
        return Ok(());
    };
    let required = positions
        .checked_add(1)
        .ok_or_else(|| invalid("position limit overflowed"))?;
    if tokens.len() < required {
        return Err(invalid(format!(
            "position limit {positions} needs {required} tokens, but the file has {}",
            tokens.len()
        )));
    }
    tokens.truncate(required);
    Ok(())
}

fn write_token_output(path: &Path, tokens: &[u32]) -> Result<(), io::Error> {
    write_tokens(path, tokens)?;
    println!("tokens: {} ({})", path.display(), tokens.len());
    Ok(())
}

fn dump_logits_for_backend(
    arguments: &EvalArgs,
    tokens: &[u32],
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    match arguments.backend {
        BackendChoice::Cuda => dump_logits(CudaBackend::new(0)?, arguments, tokens, path),
        BackendChoice::Cpu => dump_logits(CpuBackend::new(), arguments, tokens, path),
        BackendChoice::CpuQ8_1 => {
            dump_logits(CpuBackend::with_q8_1_activations(), arguments, tokens, path)
        }
    }
}

fn parse(arguments: &[String]) -> Result<EvalArgs, io::Error> {
    let mut parsed = EvalBuilder::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let handled = parse_eval_paths(&mut parsed, arguments, &mut index)?
            || parse_eval_backend(&mut parsed, arguments, &mut index)?
            || parse_eval_limits(&mut parsed, arguments, &mut index)?;
        if !handled {
            return Err(invalid(format!("eval argument is invalid: {flag}")));
        }
        index += 1;
    }
    parsed.finish()
}

#[derive(Debug)]
struct EvalBuilder {
    model: Option<PathBuf>,
    corpus: Option<PathBuf>,
    tokens: Option<PathBuf>,
    token_limit: Option<usize>,
    tokens_out: Option<PathBuf>,
    logits: Option<PathBuf>,
    backend: BackendChoice,
    window_tokens: usize,
    position_limit: Option<usize>,
    prefill_chunk: Option<usize>,
}

impl Default for EvalBuilder {
    fn default() -> Self {
        Self {
            model: None,
            corpus: None,
            tokens: None,
            token_limit: None,
            tokens_out: None,
            logits: None,
            backend: BackendChoice::Cuda,
            window_tokens: DEFAULT_WINDOW_TOKENS,
            position_limit: None,
            prefill_chunk: None,
        }
    }
}

impl EvalBuilder {
    fn finish(self) -> Result<EvalArgs, io::Error> {
        let EvalBuilder {
            model,
            corpus,
            tokens,
            token_limit,
            tokens_out,
            logits,
            backend,
            window_tokens,
            position_limit,
            prefill_chunk,
        } = self;
        let source = eval_source(corpus, tokens, token_limit)?;
        validate_eval_outputs(&tokens_out, &logits)?;
        validate_eval_position(&source, position_limit)?;
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
}

fn eval_source(
    corpus: Option<PathBuf>,
    tokens: Option<PathBuf>,
    token_limit: Option<usize>,
) -> Result<TokenSource, io::Error> {
    match (corpus, tokens, token_limit) {
        (Some(path), None, Some(limit)) => Ok(TokenSource::Corpus { path, limit }),
        (Some(_), None, None) => Err(invalid("eval --corpus requires --token-limit")),
        (None, Some(path), None) => Ok(TokenSource::Binary(path)),
        (None, Some(_), Some(_)) => Err(invalid("eval --token-limit is only valid with --corpus")),
        (Some(_), Some(_), _) => Err(invalid(
            "eval accepts either --corpus or --tokens, not both",
        )),
        (None, None, _) => Err(invalid("eval requires --corpus or --tokens")),
    }
}

fn validate_eval_outputs(
    tokens_out: &Option<PathBuf>,
    logits: &Option<PathBuf>,
) -> Result<(), io::Error> {
    if tokens_out.is_none() && logits.is_none() {
        return Err(invalid("eval requires --tokens-out or --logits"));
    }
    Ok(())
}

fn validate_eval_position(
    source: &TokenSource,
    position_limit: Option<usize>,
) -> Result<(), io::Error> {
    if matches!(source, TokenSource::Corpus { .. }) && position_limit.is_some() {
        return Err(invalid(
            "eval --position-limit is only valid with a binary token input",
        ));
    }
    Ok(())
}

fn parse_eval_paths(
    parsed: &mut EvalBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_eval_source_paths(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_eval_output_paths(parsed, arguments, index)
}

fn parse_eval_source_paths(
    parsed: &mut EvalBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "-m" | "--model" => parsed.model = Some(PathBuf::from(value(arguments, index)?)),
        "--corpus" => parsed.corpus = Some(PathBuf::from(value(arguments, index)?)),
        "--tokens" => parsed.tokens = Some(PathBuf::from(value(arguments, index)?)),
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_eval_output_paths(
    parsed: &mut EvalBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--tokens-out" => parsed.tokens_out = Some(PathBuf::from(value(arguments, index)?)),
        "--logits" => parsed.logits = Some(PathBuf::from(value(arguments, index)?)),
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_eval_backend(
    parsed: &mut EvalBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--backend" {
        return Ok(false);
    }
    parsed.backend = match value(arguments, index)? {
        "cuda" => BackendChoice::Cuda,
        "cpu" => BackendChoice::Cpu,
        "cpu-q8_1" => BackendChoice::CpuQ8_1,
        value => return Err(invalid(format!("backend is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_eval_limits(
    parsed: &mut EvalBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_eval_count_limits(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_eval_position_limits(parsed, arguments, index)
}

fn parse_eval_count_limits(
    parsed: &mut EvalBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--token-limit" => {
            parsed.token_limit = Some(parse_positive(value(arguments, index)?, "token limit")?);
        }
        "--window" => {
            parsed.window_tokens = parse_positive(value(arguments, index)?, "window")?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_eval_position_limits(
    parsed: &mut EvalBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--position-limit" => {
            parsed.position_limit =
                Some(parse_positive(value(arguments, index)?, "position limit")?);
        }
        "--prefill-chunk" => {
            parsed.prefill_chunk = Some(parse_positive(value(arguments, index)?, "prefill chunk")?);
        }
        _ => return Ok(false),
    }
    Ok(true)
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
    let token_count = tokens.len();
    let mut write_row = |position: usize, logits: &[f32]| {
        write_logit_row(&mut output, position, token_count, logits)
    };
    let scored = evaluate_logits(&mut runtime, arguments, tokens, &mut write_row)?;
    output.flush()?;
    println!(
        "logits: {} ({scored} positions, vocab {})",
        path.display(),
        runtime.model().config().vocab_size
    );
    Ok(())
}

fn evaluate_logits<B: Backend>(
    runtime: &mut Runtime<B>,
    arguments: &EvalArgs,
    tokens: &[u32],
    write_row: &mut impl FnMut(usize, &[f32]) -> Result<(), RuntimeError>,
) -> Result<usize, RuntimeError> {
    match arguments.prefill_chunk {
        Some(chunk) => runtime.evaluate_logits_chunked(
            tokens,
            arguments.window_tokens,
            chunk,
            KvCacheDtype::F16,
            write_row,
        ),
        None => runtime.evaluate_logits(
            tokens,
            arguments.window_tokens,
            KvCacheDtype::F16,
            write_row,
        ),
    }
}

fn write_logit_row(
    output: &mut BufWriter<File>,
    position: usize,
    token_count: usize,
    logits: &[f32],
) -> Result<(), RuntimeError> {
    for logit in logits {
        output
            .write_all(&logit.to_le_bytes())
            .map_err(RuntimeError::logit_callback)?;
    }
    if position.is_multiple_of(128) {
        eprintln!("scored position {position}/{}", token_count - 1);
    }
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
