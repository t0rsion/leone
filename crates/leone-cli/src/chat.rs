use leone::{
    AdaptiveDrafter, AdaptiveDrafterConfig, Backend, CpuBackend, DecodeExecution,
    DecodeProfileMode, GenerateOptions, KvCacheDtype, LogitCapture, Penalties, Runtime,
    RuntimeError, Sampler, Speculation, SuffixDrafter, Temperature, Tokenizer, Truncation,
};
use leone_cuda::CudaBackend;
use std::error::Error;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const DEFAULT_MAX_TOKENS: usize = 512;
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const LLAMA_BEGIN: &str = "<|begin_of_text|>";
const LLAMA_HEADER_START: &str = "<|start_header_id|>";
const LLAMA_HEADER_END: &str = "<|end_header_id|>";
const LLAMA_EOT: &str = "<|eot_id|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cuda,
    Cpu,
}

#[derive(Debug, PartialEq)]
struct ChatArgs {
    model: PathBuf,
    tokens: usize,
    backend: BackendChoice,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    sampler: Sampler,
    penalties: Penalties,
    seed: u64,
    speculation: Speculation,
    plan: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
}

impl Role {
    const fn name(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Turn {
    role: Role,
    content: String,
}

enum PromptPart<'a> {
    Special(&'static str),
    Text(&'a str),
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    let interrupted = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&interrupted);
    ctrlc::set_handler(move || handler_flag.store(true, Ordering::Relaxed))?;
    match arguments.backend {
        BackendChoice::Cuda => run_chat(CudaBackend::new(0)?, arguments, interrupted),
        BackendChoice::Cpu => run_chat(CpuBackend::new(), arguments, interrupted),
    }
}

fn parse(arguments: &[String]) -> Result<ChatArgs, io::Error> {
    let mut parsed = ChatBuilder::default();
    parse_chat_arguments(arguments, &mut parsed)?;
    parsed.finish()
}

fn parse_chat_arguments(arguments: &[String], parsed: &mut ChatBuilder) -> Result<(), io::Error> {
    let mut index = 0;
    while index < arguments.len() {
        if parse_chat_argument(parsed, arguments, &mut index)? {
            index += 1;
            continue;
        }
        let flag = arguments[index].as_str();
        return Err(invalid_data(format!("chat argument is invalid: {flag}")));
    }
    Ok(())
}

fn parse_chat_argument(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_paths(parsed, arguments, index)? || parse_chat_runtime(parsed, arguments, index)?
    {
        return Ok(true);
    }
    if parse_chat_sampling(parsed, arguments, index)?
        || parse_chat_penalties(parsed, arguments, index)?
    {
        return Ok(true);
    }
    parse_chat_speculation(parsed, arguments, index)
}

#[derive(Debug)]
struct ChatBuilder {
    model: Option<PathBuf>,
    tokens: usize,
    backend: BackendChoice,
    eager_decode: bool,
    kv_cache_dtype: KvCacheDtype,
    temperature: Temperature,
    truncations: Vec<Truncation>,
    penalties: Penalties,
    seed: u64,
    draft: Option<NonZeroUsize>,
    adaptive_draft: bool,
    plan: Option<PathBuf>,
}

impl Default for ChatBuilder {
    fn default() -> Self {
        Self {
            model: None,
            tokens: DEFAULT_MAX_TOKENS,
            backend: BackendChoice::Cuda,
            eager_decode: false,
            kv_cache_dtype: KvCacheDtype::F16,
            temperature: Temperature::Greedy,
            truncations: Vec::new(),
            penalties: Penalties::none(),
            seed: 0,
            draft: None,
            adaptive_draft: true,
            plan: None,
        }
    }
}

impl ChatBuilder {
    fn finish(self) -> Result<ChatArgs, io::Error> {
        Ok(ChatArgs {
            model: self
                .model
                .ok_or_else(|| invalid_data("chat requires -m <gguf>"))?,
            tokens: self.tokens,
            backend: self.backend,
            eager_decode: self.eager_decode,
            kv_cache_dtype: self.kv_cache_dtype,
            sampler: Sampler {
                temperature: self.temperature,
                truncations: self.truncations,
            },
            penalties: self.penalties,
            seed: self.seed,
            speculation: chat_speculation(self.draft, self.adaptive_draft)?,
            plan: self.plan,
        })
    }
}

fn parse_chat_paths(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_model(parsed, arguments, index)? {
        return Ok(true);
    }
    if parse_chat_tokens(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_chat_plan(parsed, arguments, index)
}

fn parse_chat_model(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "-m" && arguments[*index] != "--model" {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_chat_tokens(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "-n" && arguments[*index] != "--tokens" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.tokens = value
        .parse()
        .map_err(|_| invalid_data(format!("chat token count is invalid: {value}")))?;
    if parsed.tokens == 0 {
        return Err(invalid_data("chat token count must be nonzero"));
    }
    Ok(true)
}

fn parse_chat_plan(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--plan" {
        return Ok(false);
    }
    parsed.plan = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_chat_runtime(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_backend(parsed, arguments, index)? {
        return Ok(true);
    }
    if arguments[*index] == "--eager-decode" {
        parsed.eager_decode = true;
        return Ok(true);
    }
    parse_chat_kv(parsed, arguments, index)
}

fn parse_chat_backend(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--backend" {
        return Ok(false);
    }
    parsed.backend = match flag_value(arguments, index)? {
        "cuda" => BackendChoice::Cuda,
        "cpu" => BackendChoice::Cpu,
        value => return Err(invalid_data(format!("backend is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_chat_kv(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--kv" {
        return Ok(false);
    }
    parsed.kv_cache_dtype = match flag_value(arguments, index)? {
        "q8" => KvCacheDtype::Q8,
        "f16" => KvCacheDtype::F16,
        "f32" => KvCacheDtype::F32,
        value => return Err(invalid_data(format!("KV dtype is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_chat_sampling(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_sampling_control(parsed, arguments, index)? {
        return Ok(true);
    }
    if parse_chat_top_truncation(parsed, arguments, index)? {
        return Ok(true);
    }
    if parse_chat_tail_truncation(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_chat_raw_truncation(parsed, arguments, index)
}

fn parse_chat_sampling_control(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--temp" | "--temperature" => {
            parsed.temperature = Temperature::Scaled(parse_f64(arguments, index, "--temp")?);
        }
        "--seed" => {
            let value = flag_value(arguments, index)?;
            parsed.seed = value
                .parse()
                .map_err(|_| invalid_data(format!("seed is invalid: {value}")))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_chat_top_truncation(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_top_k(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_chat_top_mass(parsed, arguments, index)
}

fn parse_chat_top_k(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--top-k" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    let count = value
        .parse::<usize>()
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| invalid_data(format!("top-k is invalid: {value}")))?;
    parsed.truncations.push(Truncation::TopK(count));
    Ok(true)
}

fn parse_chat_top_mass(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let truncation = match arguments[*index].as_str() {
        "--top-p" => Truncation::TopP(parse_f64(arguments, index, "--top-p")?),
        "--min-p" => Truncation::MinP(parse_f64(arguments, index, "--min-p")?),
        "--top-a" => Truncation::TopA(parse_f64(arguments, index, "--top-a")?),
        _ => return Ok(false),
    };
    parsed.truncations.push(truncation);
    Ok(true)
}

fn parse_chat_tail_truncation(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let truncation = match arguments[*index].as_str() {
        "--tfs" => Truncation::TailFree(parse_f64(arguments, index, "--tfs")?),
        "--typical" => Truncation::Typical(parse_f64(arguments, index, "--typical")?),
        "--epsilon" => Truncation::Epsilon(parse_f64(arguments, index, "--epsilon")?),
        "--eta" => Truncation::Eta(parse_f64(arguments, index, "--eta")?),
        _ => return Ok(false),
    };
    parsed.truncations.push(truncation);
    Ok(true)
}

fn parse_chat_raw_truncation(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let truncation = match arguments[*index].as_str() {
        "--min-k" => Truncation::MinK(parse_f64(arguments, index, "--min-k")?),
        "--top-n-sigma" => Truncation::TopNSigma(parse_f64(arguments, index, "--top-n-sigma")?),
        _ => return Ok(false),
    };
    parsed.truncations.push(truncation);
    Ok(true)
}

fn parse_chat_penalties(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_basic_penalty(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_chat_dry_penalty(parsed, arguments, index)
}

fn parse_chat_basic_penalty(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_penalty_values(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_chat_penalty_window(parsed, arguments, index)
}

fn parse_chat_penalty_values(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--repeat-penalty" => {
            parsed.penalties.repetition = parse_f64(arguments, index, "--repeat-penalty")?;
        }
        "--presence-penalty" => {
            parsed.penalties.presence = parse_f64(arguments, index, "--presence-penalty")?;
        }
        "--frequency-penalty" => {
            parsed.penalties.frequency = parse_f64(arguments, index, "--frequency-penalty")?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_chat_penalty_window(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--penalty-window" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.penalties.window = value
        .parse()
        .map_err(|_| invalid_data(format!("penalty window is invalid: {value}")))?;
    Ok(true)
}

fn parse_chat_dry_penalty(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if parse_chat_dry_floats(parsed, arguments, index)? {
        return Ok(true);
    }
    parse_chat_dry_sizes(parsed, arguments, index)
}

fn parse_chat_dry_floats(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--dry-multiplier" => {
            parsed.penalties.dry.multiplier = parse_f64(arguments, index, "--dry-multiplier")?;
        }
        "--dry-base" => {
            parsed.penalties.dry.base = parse_f64(arguments, index, "--dry-base")?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_chat_dry_sizes(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--dry-allowed-length" => {
            let value = flag_value(arguments, index)?;
            parsed.penalties.dry.allowed_length = value
                .parse()
                .map_err(|_| invalid_data(format!("DRY allowed length is invalid: {value}")))?;
        }
        "--dry-window" => {
            let value = flag_value(arguments, index)?;
            parsed.penalties.dry.window = value
                .parse()
                .map_err(|_| invalid_data(format!("DRY window is invalid: {value}")))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_chat_speculation(
    parsed: &mut ChatBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "--draft" => {
            let value = flag_value(arguments, index)?;
            parsed.draft = Some(
                value
                    .parse::<usize>()
                    .ok()
                    .and_then(NonZeroUsize::new)
                    .ok_or_else(|| invalid_data(format!("draft length is invalid: {value}")))?,
            );
        }
        "--no-draft" => parsed.adaptive_draft = false,
        _ => return Ok(false),
    }
    Ok(true)
}

fn chat_speculation(
    draft: Option<NonZeroUsize>,
    adaptive_draft: bool,
) -> Result<Speculation, io::Error> {
    match draft {
        None if adaptive_draft => Ok(Speculation::Adaptive(AdaptiveDrafter::new(
            AdaptiveDrafterConfig::default(),
        ))),
        None => Ok(Speculation::Disabled),
        Some(proposal) => SuffixDrafter::new(
            NonZeroUsize::new(16).expect("sixteen is nonzero"),
            NonZeroUsize::new(4).expect("four is nonzero"),
            proposal,
        )
        .map(Speculation::Suffix)
        .map_err(|error| invalid_data(error.to_string())),
    }
}

fn parse_f64(arguments: &[String], index: &mut usize, flag: &str) -> Result<f64, io::Error> {
    let value = flag_value(arguments, index)?;
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid_data(format!("{flag} is invalid: {value}")))?;
    if !parsed.is_finite() {
        return Err(invalid_data(format!("{flag} must be finite: {value}")));
    }
    Ok(parsed)
}

fn flag_value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid_data("command flag is missing its value"))
}

fn run_chat<B: Backend>(
    backend: B,
    arguments: ChatArgs,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let backend_name = backend.name();
    let mut runtime = Runtime::load(backend, &arguments.model)?;
    let execution_plan = crate::execution_plan::load_optional(
        arguments.plan.as_deref(),
        &arguments.model,
        backend_name,
    )?;
    let mut stdout = io::stdout().lock();
    let architecture = runtime.model().config().architecture;
    writeln!(
        stdout,
        "Leone {} chat. Use /clear to reset or /exit to quit.",
        architecture.name()
    )?;
    run_chat_loop(
        &mut runtime,
        &arguments,
        |options| {
            if let Some(plan) = execution_plan.as_ref() {
                plan.apply(options);
            }
        },
        &interrupted,
        architecture,
        &mut stdout,
    )
}

fn run_chat_loop<B: Backend, W: Write, F: Fn(&mut GenerateOptions)>(
    runtime: &mut Runtime<B>,
    arguments: &ChatArgs,
    apply_plan: F,
    interrupted: &Arc<AtomicBool>,
    architecture: leone::ModelArchitecture,
    stdout: &mut W,
) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut turns = Vec::new();
    let mut input = String::new();
    loop {
        let Some(message) = read_chat_input(&stdin, stdout, &mut input)? else {
            break;
        };
        match classify_chat_input(message, &mut turns, stdout)? {
            ChatInput::Continue => continue,
            ChatInput::Exit => break,
            ChatInput::Message(message) => run_chat_turn(
                runtime,
                ChatTurnContext {
                    arguments,
                    interrupted,
                    architecture,
                    turns: &mut turns,
                    message: &message,
                },
                |options| apply_plan(options),
                stdout,
            )?,
        }
    }
    Ok(())
}

enum ChatInput {
    Continue,
    Exit,
    Message(String),
}

struct ChatTurnContext<'a> {
    arguments: &'a ChatArgs,
    interrupted: &'a Arc<AtomicBool>,
    architecture: leone::ModelArchitecture,
    turns: &'a mut Vec<Turn>,
    message: &'a str,
}

fn read_chat_input<W: Write>(
    stdin: &io::Stdin,
    stdout: &mut W,
    input: &mut String,
) -> Result<Option<String>, Box<dyn Error>> {
    write!(stdout, "You: ")?;
    stdout.flush()?;
    input.clear();
    if stdin.read_line(input)? == 0 {
        writeln!(stdout)?;
        return Ok(None);
    }
    Ok(Some(input.trim_end_matches(['\r', '\n']).to_owned()))
}

fn classify_chat_input<W: Write>(
    message: String,
    turns: &mut Vec<Turn>,
    stdout: &mut W,
) -> Result<ChatInput, Box<dyn Error>> {
    match message.as_str() {
        "" => Ok(ChatInput::Continue),
        "/exit" => Ok(ChatInput::Exit),
        "/clear" => {
            turns.clear();
            writeln!(stdout, "Conversation cleared.")?;
            Ok(ChatInput::Continue)
        }
        _ => Ok(ChatInput::Message(message)),
    }
}

fn run_chat_turn<B: Backend, W: Write, F: FnOnce(&mut GenerateOptions)>(
    runtime: &mut Runtime<B>,
    context: ChatTurnContext<'_>,
    apply_plan: F,
    stdout: &mut W,
) -> Result<(), Box<dyn Error>> {
    let ChatTurnContext {
        arguments,
        interrupted,
        architecture,
        turns,
        message,
    } = context;
    turns.push(Turn {
        role: Role::User,
        content: message.to_owned(),
    });
    let capacity = runtime.model().config().context_length;
    let Some(prompt_tokens) = prompt_for_turn(runtime, architecture, turns, capacity, stdout)?
    else {
        turns.pop();
        return Ok(());
    };
    let remaining = capacity
        .checked_sub(prompt_tokens.len())
        .and_then(|positions| positions.checked_add(1))
        .ok_or_else(|| invalid_data("chat context length overflowed"))?;
    let mut options = chat_options(runtime, arguments, remaining);
    apply_plan(&mut options);
    let result = generate_chat_response(runtime, &prompt_tokens, options, interrupted, stdout)?;
    let response = runtime.model().tokenizer().decode(&result.tokens)?;
    if result.stats.cancelled && response.is_empty() {
        turns.pop();
        return Ok(());
    }
    turns.push(Turn {
        role: Role::Assistant,
        content: response,
    });
    Ok(())
}

fn prompt_for_turn<B: Backend, W: Write>(
    runtime: &Runtime<B>,
    architecture: leone::ModelArchitecture,
    turns: &[Turn],
    capacity: usize,
    stdout: &mut W,
) -> Result<Option<Vec<u32>>, Box<dyn Error>> {
    let prompt_tokens = chat_prompt_tokens(runtime.model().tokenizer(), architecture, turns)?;
    if prompt_tokens.len() > capacity {
        writeln!(
            stdout,
            "Conversation needs {} positions but the model supports {capacity}. Use /clear.",
            prompt_tokens.len()
        )?;
        return Ok(None);
    }
    Ok(Some(prompt_tokens))
}

fn chat_options<B: Backend>(
    runtime: &Runtime<B>,
    arguments: &ChatArgs,
    remaining: usize,
) -> GenerateOptions {
    let mut options = GenerateOptions::greedy(arguments.tokens.min(remaining));
    options.logit_capture = LogitCapture::Disabled;
    options.decode_profile = DecodeProfileMode::Disabled;
    options.decode_execution = if arguments.eager_decode
        || !runtime
            .model()
            .config()
            .architecture
            .decode_graph_supported()
    {
        DecodeExecution::Eager
    } else {
        DecodeExecution::Graph
    };
    options.kv_cache_dtype = arguments.kv_cache_dtype;
    options.sampler = arguments.sampler.clone();
    options.penalties = arguments.penalties.clone();
    options.seed = arguments.seed;
    options.speculation = arguments.speculation;
    options
}

fn generate_chat_response<B: Backend, W: Write>(
    runtime: &mut Runtime<B>,
    prompt_tokens: &[u32],
    options: GenerateOptions,
    interrupted: &Arc<AtomicBool>,
    stdout: &mut W,
) -> Result<leone::GenerationResult, Box<dyn Error>> {
    interrupted.store(false, Ordering::Relaxed);
    write!(stdout, "Assistant: ")?;
    stdout.flush()?;
    let result = runtime.generate_tokens(
        prompt_tokens,
        options,
        |token| {
            stdout
                .write_all(&token.bytes)
                .and_then(|()| stdout.flush())
                .map_err(RuntimeError::token_callback)
        },
        || interrupted.load(Ordering::Relaxed),
    )?;
    writeln!(stdout)?;
    Ok(result)
}

fn chat_prompt_tokens(
    tokenizer: &Tokenizer,
    architecture: leone::ModelArchitecture,
    turns: &[Turn],
) -> Result<Vec<u32>, RuntimeError> {
    let parts = match architecture {
        leone::ModelArchitecture::Qwen3 => qwen3_prompt_parts(turns),
        leone::ModelArchitecture::Llama => llama3_prompt_parts(turns),
    };
    prompt_tokens(tokenizer, parts)
}

fn prompt_tokens(
    tokenizer: &Tokenizer,
    parts: Vec<PromptPart<'_>>,
) -> Result<Vec<u32>, RuntimeError> {
    let mut tokens = Vec::new();
    for part in parts {
        match part {
            PromptPart::Special(value) => tokens.push(tokenizer.token_id(value)?),
            PromptPart::Text(value) => tokens.extend(tokenizer.encode_piece(value)?),
        }
    }
    Ok(tokens)
}

fn llama3_prompt_parts(turns: &[Turn]) -> Vec<PromptPart<'_>> {
    let mut parts = Vec::with_capacity(turns.len() * 7 + 5);
    parts.push(PromptPart::Special(LLAMA_BEGIN));
    for turn in turns {
        parts.push(PromptPart::Special(LLAMA_HEADER_START));
        parts.push(PromptPart::Text(turn.role.name()));
        parts.push(PromptPart::Special(LLAMA_HEADER_END));
        parts.push(PromptPart::Text("\n\n"));
        parts.push(PromptPart::Text(&turn.content));
        parts.push(PromptPart::Special(LLAMA_EOT));
    }
    parts.push(PromptPart::Special(LLAMA_HEADER_START));
    parts.push(PromptPart::Text(Role::Assistant.name()));
    parts.push(PromptPart::Special(LLAMA_HEADER_END));
    parts.push(PromptPart::Text("\n\n"));
    parts
}

fn qwen3_prompt_parts(turns: &[Turn]) -> Vec<PromptPart<'_>> {
    let mut parts = Vec::with_capacity(turns.len() * 6 + 3);
    for turn in turns {
        parts.push(PromptPart::Special(IM_START));
        parts.push(PromptPart::Text(turn.role.name()));
        parts.push(PromptPart::Text("\n"));
        parts.push(PromptPart::Text(&turn.content));
        parts.push(PromptPart::Special(IM_END));
        parts.push(PromptPart::Text("\n"));
    }
    parts.push(PromptPart::Special(IM_START));
    parts.push(PromptPart::Text(Role::Assistant.name()));
    parts.push(PromptPart::Text("\n"));
    parts
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_the_qwen3_no_tools_template() {
        let turns = [
            Turn {
                role: Role::User,
                content: "Hello".to_owned(),
            },
            Turn {
                role: Role::Assistant,
                content: "Hi".to_owned(),
            },
            Turn {
                role: Role::User,
                content: "Continue".to_owned(),
            },
        ];
        let rendered = qwen3_prompt_parts(&turns)
            .into_iter()
            .map(|part| match part {
                PromptPart::Special(value) | PromptPart::Text(value) => value,
            })
            .collect::<String>();
        assert_eq!(
            rendered,
            "<|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\nHi<|im_end|>\n\
             <|im_start|>user\nContinue<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn renders_the_llama3_no_tools_template() {
        let turns = [Turn {
            role: Role::User,
            content: "Hello".to_owned(),
        }];
        let rendered = llama3_prompt_parts(&turns)
            .into_iter()
            .map(|part| match part {
                PromptPart::Special(value) | PromptPart::Text(value) => value,
            })
            .collect::<String>();
        assert_eq!(
            rendered,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
             Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn parses_chat_defaults() {
        let parsed = parse(&["-m".to_owned(), "model.gguf".to_owned()]).unwrap();
        assert_eq!(parsed.model, PathBuf::from("model.gguf"));
        assert_eq!(parsed.tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(parsed.backend, BackendChoice::Cuda);
        assert_eq!(parsed.kv_cache_dtype, KvCacheDtype::F16);
        assert_eq!(parsed.sampler, Sampler::greedy());
        assert_eq!(parsed.penalties, Penalties::none());
        assert_eq!(parsed.seed, 0);
        assert_eq!(
            parsed.speculation,
            Speculation::Adaptive(AdaptiveDrafter::new(AdaptiveDrafterConfig::default()))
        );
        assert!(!parsed.eager_decode);
    }

    #[test]
    fn rejects_zero_generation_length() {
        assert!(parse(&[
            "-m".to_owned(),
            "model.gguf".to_owned(),
            "-n".to_owned(),
            "0".to_owned(),
        ])
        .is_err());
    }
}
