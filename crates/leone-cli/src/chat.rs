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
    let mut model = None;
    let mut tokens = DEFAULT_MAX_TOKENS;
    let mut backend = BackendChoice::Cuda;
    let mut eager_decode = false;
    let mut kv_cache_dtype = KvCacheDtype::F16;
    let mut temperature = Temperature::Greedy;
    let mut truncations = Vec::new();
    let mut penalties = Penalties::none();
    let mut seed = 0_u64;
    let mut draft = None;
    let mut adaptive_draft = true;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-m" | "--model" => {
                model = Some(PathBuf::from(flag_value(arguments, &mut index)?));
            }
            "-n" | "--tokens" => {
                let value = flag_value(arguments, &mut index)?;
                tokens = value
                    .parse()
                    .map_err(|_| invalid_data(format!("chat token count is invalid: {value}")))?;
                if tokens == 0 {
                    return Err(invalid_data("chat token count must be nonzero"));
                }
            }
            "--backend" => {
                backend = match flag_value(arguments, &mut index)? {
                    "cuda" => BackendChoice::Cuda,
                    "cpu" => BackendChoice::Cpu,
                    value => return Err(invalid_data(format!("backend is invalid: {value}"))),
                };
            }
            "--eager-decode" => eager_decode = true,
            "--kv" => {
                kv_cache_dtype = match flag_value(arguments, &mut index)? {
                    "q8" => KvCacheDtype::Q8,
                    "f16" => KvCacheDtype::F16,
                    "f32" => KvCacheDtype::F32,
                    value => return Err(invalid_data(format!("KV dtype is invalid: {value}"))),
                };
            }
            "--temp" | "--temperature" => {
                temperature = Temperature::Scaled(parse_f64(arguments, &mut index, "--temp")?);
            }
            "--seed" => {
                let value = flag_value(arguments, &mut index)?;
                seed = value
                    .parse()
                    .map_err(|_| invalid_data(format!("seed is invalid: {value}")))?;
            }
            "--top-k" => {
                let value = flag_value(arguments, &mut index)?;
                let count = value
                    .parse::<usize>()
                    .ok()
                    .and_then(NonZeroUsize::new)
                    .ok_or_else(|| invalid_data(format!("top-k is invalid: {value}")))?;
                truncations.push(Truncation::TopK(count));
            }
            "--top-p" => truncations.push(Truncation::TopP(parse_f64(
                arguments, &mut index, "--top-p",
            )?)),
            "--min-p" => truncations.push(Truncation::MinP(parse_f64(
                arguments, &mut index, "--min-p",
            )?)),
            "--top-a" => truncations.push(Truncation::TopA(parse_f64(
                arguments, &mut index, "--top-a",
            )?)),
            "--tfs" => truncations.push(Truncation::TailFree(parse_f64(
                arguments, &mut index, "--tfs",
            )?)),
            "--typical" => truncations.push(Truncation::Typical(parse_f64(
                arguments,
                &mut index,
                "--typical",
            )?)),
            "--epsilon" => truncations.push(Truncation::Epsilon(parse_f64(
                arguments,
                &mut index,
                "--epsilon",
            )?)),
            "--eta" => {
                truncations.push(Truncation::Eta(parse_f64(arguments, &mut index, "--eta")?))
            }
            "--min-k" => truncations.push(Truncation::MinK(parse_f64(
                arguments, &mut index, "--min-k",
            )?)),
            "--top-n-sigma" => truncations.push(Truncation::TopNSigma(parse_f64(
                arguments,
                &mut index,
                "--top-n-sigma",
            )?)),
            "--repeat-penalty" => {
                penalties.repetition = parse_f64(arguments, &mut index, "--repeat-penalty")?;
            }
            "--presence-penalty" => {
                penalties.presence = parse_f64(arguments, &mut index, "--presence-penalty")?;
            }
            "--frequency-penalty" => {
                penalties.frequency = parse_f64(arguments, &mut index, "--frequency-penalty")?;
            }
            "--penalty-window" => {
                let value = flag_value(arguments, &mut index)?;
                penalties.window = value
                    .parse()
                    .map_err(|_| invalid_data(format!("penalty window is invalid: {value}")))?;
            }
            "--dry-multiplier" => {
                penalties.dry.multiplier = parse_f64(arguments, &mut index, "--dry-multiplier")?;
            }
            "--dry-base" => {
                penalties.dry.base = parse_f64(arguments, &mut index, "--dry-base")?;
            }
            "--dry-allowed-length" => {
                let value = flag_value(arguments, &mut index)?;
                penalties.dry.allowed_length = value
                    .parse()
                    .map_err(|_| invalid_data(format!("DRY allowed length is invalid: {value}")))?;
            }
            "--dry-window" => {
                let value = flag_value(arguments, &mut index)?;
                penalties.dry.window = value
                    .parse()
                    .map_err(|_| invalid_data(format!("DRY window is invalid: {value}")))?;
            }
            "--draft" => {
                let value = flag_value(arguments, &mut index)?;
                draft = Some(
                    value
                        .parse::<usize>()
                        .ok()
                        .and_then(NonZeroUsize::new)
                        .ok_or_else(|| invalid_data(format!("draft length is invalid: {value}")))?,
                );
            }
            "--no-draft" => adaptive_draft = false,
            value => return Err(invalid_data(format!("chat argument is invalid: {value}"))),
        }
        index += 1;
    }
    Ok(ChatArgs {
        model: model.ok_or_else(|| invalid_data("chat requires -m <gguf>"))?,
        tokens,
        backend,
        eager_decode,
        kv_cache_dtype,
        sampler: Sampler {
            temperature,
            truncations,
        },
        penalties,
        seed,
        speculation: match draft {
            None if adaptive_draft => {
                Speculation::Adaptive(AdaptiveDrafter::new(AdaptiveDrafterConfig::default()))
            }
            None => Speculation::Disabled,
            Some(proposal) => Speculation::Suffix(
                SuffixDrafter::new(
                    NonZeroUsize::new(16).expect("sixteen is nonzero"),
                    NonZeroUsize::new(4).expect("four is nonzero"),
                    proposal,
                )
                .map_err(|error| invalid_data(error.to_string()))?,
            ),
        },
    })
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
    let mut runtime = Runtime::load(backend, &arguments.model)?;
    let mut turns = Vec::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut input = String::new();
    let architecture = runtime.model().config().architecture;
    writeln!(
        stdout,
        "Leone {} chat. Use /clear to reset or /exit to quit.",
        architecture.name()
    )?;
    loop {
        write!(stdout, "You: ")?;
        stdout.flush()?;
        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            writeln!(stdout)?;
            break;
        }
        let message = input.trim_end_matches(['\r', '\n']);
        match message {
            "" => continue,
            "/exit" => break,
            "/clear" => {
                turns.clear();
                writeln!(stdout, "Conversation cleared.")?;
                continue;
            }
            _ => {}
        }

        turns.push(Turn {
            role: Role::User,
            content: message.to_owned(),
        });
        let prompt_tokens = chat_prompt_tokens(runtime.model().tokenizer(), architecture, &turns)?;
        let capacity = runtime.model().config().context_length;
        if prompt_tokens.len() > capacity {
            turns.pop();
            writeln!(
                stdout,
                "Conversation needs {} positions but the model supports {capacity}. Use /clear.",
                prompt_tokens.len()
            )?;
            continue;
        }
        let remaining = capacity
            .checked_sub(prompt_tokens.len())
            .and_then(|positions| positions.checked_add(1))
            .ok_or_else(|| invalid_data("chat context length overflowed"))?;
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
        interrupted.store(false, Ordering::Relaxed);
        write!(stdout, "Assistant: ")?;
        stdout.flush()?;
        let result = runtime.generate_tokens(
            &prompt_tokens,
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
        let response = runtime.model().tokenizer().decode(&result.tokens)?;
        if result.stats.cancelled && response.is_empty() {
            turns.pop();
            continue;
        }
        turns.push(Turn {
            role: Role::Assistant,
            content: response,
        });
    }
    Ok(())
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
