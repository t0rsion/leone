#![cfg(target_os = "macos")]

use leone::{
    Backend, DecodeExecution, GenerateOptions, GenerationResult, GenerationSession, LogitCapture,
    PrefillMethod, PrefillNumerics, PrefillProgress, Runtime, SessionReuseClass,
};
use leone_metal::MetalBackend;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const QWEN_MODEL: &str = "Qwen3-8B-Q4_K_M.gguf";
const LLAMA_MODEL: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";
const COMPATIBILITY_JSON: &str = include_str!("../../../packaging/compatibility.json");
const PREFIX_TOKENS: usize = 384;
const SUFFIX_TOKENS: usize = 129;
const CHUNK_TOKENS: usize = 128;
const CONTINUATION_TOKENS: usize = 4;
const LIFECYCLE_OUTPUT_TOKENS: usize = 2;

#[test]
#[ignore = "requires LEONE_METAL_MODEL=Qwen3-8B-Q4_K_M.gguf and an Apple GPU"]
fn qwen_metal_warm_prefill_fullmodel_bitwise() -> TestResult {
    run_warm_prefill(QWEN_MODEL, "qwen3")
}

#[test]
#[ignore = "requires LEONE_METAL_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf and an Apple GPU"]
fn llama_metal_warm_prefill_fullmodel_bitwise() -> TestResult {
    run_warm_prefill(LLAMA_MODEL, "llama")
}

#[test]
#[ignore = "requires LEONE_METAL_MODEL=Qwen3-8B-Q4_K_M.gguf and an Apple GPU"]
fn qwen_metal_warm_prefill_session_lifecycle() -> TestResult {
    run_warm_prefill_lifecycle(QWEN_MODEL, "qwen3")
}

#[test]
#[ignore = "requires LEONE_METAL_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf and an Apple GPU"]
fn llama_metal_warm_prefill_session_lifecycle() -> TestResult {
    run_warm_prefill_lifecycle(LLAMA_MODEL, "llama")
}

fn run_warm_prefill(expected_model: &str, expected_architecture: &str) -> TestResult {
    let (mut runtime, prefix, suffix) = load_warm_model(expected_model, expected_architecture)?;
    let result = runtime.characterize_warm_prefill(
        &prefix,
        &suffix,
        NonZeroUsize::new(CHUNK_TOKENS).expect("chunk is nonzero"),
        NonZeroUsize::new(CONTINUATION_TOKENS).expect("continuation is nonzero"),
    )?;
    assert_eq!(result.prefix_tokens, PREFIX_TOKENS);
    assert_eq!(result.suffix_tokens, SUFFIX_TOKENS);
    assert_eq!(result.chunk_tokens, CHUNK_TOKENS);
    assert_eq!(result.continuation_tokens, CONTINUATION_TOKENS);
    assert_eq!(result.numerics, PrefillNumerics::DecodeEquivalent);
    assert_eq!(result.control_method, PrefillMethod::SequentialDecode);
    assert_eq!(result.subject_method, runtime.prefill_method());
    assert_ne!(result.subject_method, PrefillMethod::SequentialDecode);
    assert_eq!(result.subject_full_chunks, 1);
    assert_eq!(result.subject_tail_tokens, 1);
    assert_eq!(result.sequential_tokens, result.subject_tokens);
    let prefix_positions = prefix.len();
    let suffix_positions = prefix_positions + suffix.len();
    let continued_positions = suffix_positions + CONTINUATION_TOKENS - 1;
    assert_bitwise_zero(
        result.prefix_kv,
        kv_cardinality(&runtime, prefix_positions),
        "prefix KV",
    );
    assert_bitwise_zero(result.prefix_logits, runtime.vocab_size(), "prefix logits");
    assert_bitwise_zero(
        result.suffix_kv,
        kv_cardinality(&runtime, suffix_positions),
        "suffix KV",
    );
    assert_bitwise_zero(
        result.continued_kv,
        kv_cardinality(&runtime, continued_positions),
        "continued KV",
    );
    assert_bitwise_zero(
        result.suffix_logits,
        suffix.len() * runtime.vocab_size(),
        "suffix logits",
    );
    assert_bitwise_zero(
        result.continuation_logits,
        CONTINUATION_TOKENS * runtime.vocab_size(),
        "continuation logits",
    );
    Ok(())
}

fn run_warm_prefill_lifecycle(expected_model: &str, expected_architecture: &str) -> TestResult {
    let (mut runtime, prefix, suffix) = load_warm_model(expected_model, expected_architecture)?;
    warm_session_lifecycle(&mut runtime, &prefix, &suffix)
}

fn load_warm_model(
    expected_model: &str,
    expected_architecture: &str,
) -> TestResult<(Runtime<MetalBackend>, Vec<u32>, Vec<u32>)> {
    let model = model_path(expected_model)?;
    let digest = model_sha256(&model)?;
    assert_eq!(digest, gated_model_sha256(expected_architecture)?);
    let runtime = Runtime::load(MetalBackend::new()?, &model)?;
    assert_eq!(
        runtime.model().config().architecture.name(),
        expected_architecture
    );
    assert!(runtime.backend().decode_equivalent_prefill_supported());
    let prefix = warm_tokens(&runtime, PREFIX_TOKENS, "shared prefix")?;
    let suffix = warm_tokens(&runtime, SUFFIX_TOKENS, "warm suffix")?;
    assert!(prefix.len() + suffix.len() > 512);
    Ok((runtime, prefix, suffix))
}

fn warm_session_lifecycle(
    runtime: &mut Runtime<MetalBackend>,
    prefix: &[u32],
    suffix: &[u32],
) -> TestResult {
    let prompt = append_tokens(prefix, suffix);
    let options = lifecycle_options(runtime.vocab_size())?;
    let mut live = GenerationSession::new();
    complete_prefill(runtime, &mut live, prefix, options.clone())?;
    let warm = continue_session(runtime, &mut live, &prompt, &options)?;
    assert_eq!(warm.tokens.len(), LIFECYCLE_OUTPUT_TOKENS);
    assert_eq!(warm.stats.prefill_method, PrefillMethod::ChunkedGpu);
    assert!(warm.stats.prefill_workspace.batch_activation_bytes > 0);
    assert_eq!(
        live.last_replay().reuse_class,
        SessionReuseClass::AppendOnly
    );
    assert_eq!(live.last_replay().reused_tokens, prefix.len());

    check_fork_lifecycle(runtime, &mut live, &options)?;
    check_wake_lifecycle(runtime, &mut live, &options)?;
    check_restore_lifecycle(runtime, &mut live, &options)?;
    runtime.discard_session(&mut live)?;
    paused_resumed_prefill(runtime, prefix, &prompt, &options)
}

fn check_fork_lifecycle(
    runtime: &mut Runtime<MetalBackend>,
    live: &mut GenerationSession<MetalBackend>,
    options: &GenerateOptions,
) -> TestResult {
    let history = live.evaluated_tokens().to_vec();
    let mut fork = runtime.fork_session(live)?;
    assert_eq!(
        fork.last_replay().reuse_class,
        SessionReuseClass::DeviceFork
    );
    let live_result = continue_session(runtime, live, &history, options)?;
    let fork_result = continue_session(runtime, &mut fork, &history, options)?;
    assert_results_bitwise(&live_result, &fork_result, runtime.vocab_size(), "fork");
    runtime.discard_session(&mut fork)?;
    Ok(())
}

fn check_wake_lifecycle(
    runtime: &mut Runtime<MetalBackend>,
    live: &mut GenerationSession<MetalBackend>,
    options: &GenerateOptions,
) -> TestResult {
    let history = live.evaluated_tokens().to_vec();
    let mut hibernating = runtime.fork_session(live)?;
    let hibernated = runtime.hibernate_session(&mut hibernating)?;
    let mut woken = runtime.wake_session(&hibernated)?;
    assert_eq!(woken.last_replay().reuse_class, SessionReuseClass::HostWake);
    let live_result = continue_session(runtime, live, &history, options)?;
    let wake_result = continue_session(runtime, &mut woken, &history, options)?;
    assert_results_bitwise(&live_result, &wake_result, runtime.vocab_size(), "wake");
    runtime.discard_session(&mut woken)?;
    drop(hibernated);
    Ok(())
}

fn check_restore_lifecycle(
    runtime: &mut Runtime<MetalBackend>,
    live: &mut GenerationSession<MetalBackend>,
    options: &GenerateOptions,
) -> TestResult {
    let history = live.evaluated_tokens().to_vec();
    let checkpoint = live.checkpoint();
    let mut restored = GenerationSession::new();
    restored.restore(checkpoint);
    let live_result = continue_session(runtime, live, &history, options)?;
    let restored_result = continue_session(runtime, &mut restored, &history, options)?;
    assert_eq!(
        restored.last_replay().reuse_class,
        SessionReuseClass::RestoreReplay
    );
    assert_results_bitwise(
        &live_result,
        &restored_result,
        runtime.vocab_size(),
        "restore",
    );
    runtime.discard_session(&mut restored)?;
    Ok(())
}

fn append_tokens(prefix: &[u32], suffix: &[u32]) -> Vec<u32> {
    let mut prompt = Vec::with_capacity(prefix.len() + suffix.len());
    prompt.extend_from_slice(prefix);
    prompt.extend_from_slice(suffix);
    prompt
}

fn lifecycle_options(vocab_size: usize) -> TestResult<GenerateOptions> {
    let capture = NonZeroUsize::new(vocab_size)
        .ok_or_else(|| io::Error::other("model vocabulary is empty"))?;
    Ok(GenerateOptions {
        prefill_chunk_tokens: CHUNK_TOKENS,
        logit_capture: LogitCapture::Top(capture),
        decode_execution: DecodeExecution::Eager,
        ..GenerateOptions::greedy(LIFECYCLE_OUTPUT_TOKENS)
    })
}

fn kv_cardinality(runtime: &Runtime<MetalBackend>, positions: usize) -> usize {
    let config = runtime.model().config();
    2 * config.n_layer * config.n_head_kv * config.head_dim * positions
}

fn complete_prefill(
    runtime: &mut Runtime<MetalBackend>,
    session: &mut GenerationSession<MetalBackend>,
    prompt: &[u32],
    options: GenerateOptions,
) -> TestResult {
    let mut pending = runtime.begin_prefill(session, prompt, options)?;
    loop {
        let budget = pending.minimum_budget();
        match runtime.advance_prefill(pending, budget, || false)? {
            PrefillProgress::Pending(next) => pending = next,
            PrefillProgress::Ready(ready) => {
                runtime.finish_prefill(ready, session)?;
                return Ok(());
            }
            PrefillProgress::Cancelled(_) => {
                return Err(io::Error::other("prefill was cancelled").into());
            }
        }
    }
}

fn continue_session(
    runtime: &mut Runtime<MetalBackend>,
    session: &mut GenerationSession<MetalBackend>,
    prompt: &[u32],
    options: &GenerateOptions,
) -> TestResult<GenerationResult> {
    runtime
        .generate_session_tokens(session, prompt, options.clone(), |_| Ok(()), || false)
        .map_err(Into::into)
}

fn assert_results_bitwise(
    reference: &GenerationResult,
    actual: &GenerationResult,
    vocab_size: usize,
    label: &str,
) {
    assert_eq!(
        actual.prompt_tokens, reference.prompt_tokens,
        "{label} prompt"
    );
    assert_eq!(actual.tokens, reference.tokens, "{label} tokens");
    assert!(!reference.logits.is_empty(), "{label} has no logit rows");
    assert_eq!(
        actual.logits.len(),
        reference.logits.len(),
        "{label} logit rows"
    );
    for (row_index, (expected, observed)) in reference.logits.iter().zip(&actual.logits).enumerate()
    {
        assert_eq!(
            observed.input_position, expected.input_position,
            "{label} row {row_index}"
        );
        assert_eq!(
            observed.sampled_token, expected.sampled_token,
            "{label} row {row_index}"
        );
        assert_eq!(
            observed.top.len(),
            vocab_size,
            "{label} row {row_index} is not full vocabulary"
        );
        assert_eq!(
            expected.top.len(),
            vocab_size,
            "reference row {row_index} is not full vocabulary"
        );
        assert_eq!(
            observed.top.len(),
            expected.top.len(),
            "{label} row {row_index} width"
        );
        for (logit_index, (expected_logit, observed_logit)) in
            expected.top.iter().zip(&observed.top).enumerate()
        {
            assert!(
                expected_logit.value.is_finite(),
                "{label} expected non-finite logit"
            );
            assert!(
                observed_logit.value.is_finite(),
                "{label} observed non-finite logit"
            );
            assert_eq!(
                observed_logit.token, expected_logit.token,
                "{label} logit {row_index}:{logit_index}"
            );
            assert_eq!(
                observed_logit.value.to_bits(),
                expected_logit.value.to_bits(),
                "{label} logit {row_index}:{logit_index}",
            );
        }
    }
}

fn paused_resumed_prefill(
    runtime: &mut Runtime<MetalBackend>,
    prefix: &[u32],
    prompt: &[u32],
    options: &GenerateOptions,
) -> TestResult {
    let reference = uninterrupted_warm_result(runtime, prefix, prompt, options)?;
    let (mut paused, pending) = begin_paused_warm_prefill(runtime, prefix, prompt, options)?;
    assert!(paused.is_empty());
    run_unrelated_cold_generation(runtime, prompt, options)?;
    let ready = finish_pending_prefill(runtime, pending)?;
    assert_eq!(ready.prompt_tokens(), prompt);
    runtime.finish_prefill(ready, &mut paused)?;
    let resumed = continue_session(runtime, &mut paused, prompt, options)?;
    assert_results_bitwise(&reference, &resumed, runtime.vocab_size(), "paused resume");
    assert_eq!(paused.last_replay().reused_tokens, prompt.len());
    runtime.discard_session(&mut paused)?;
    Ok(())
}

fn uninterrupted_warm_result(
    runtime: &mut Runtime<MetalBackend>,
    prefix: &[u32],
    prompt: &[u32],
    options: &GenerateOptions,
) -> TestResult<GenerationResult> {
    let mut session = GenerationSession::new();
    complete_prefill(runtime, &mut session, prefix, options.clone())?;
    let result = continue_session(runtime, &mut session, prompt, options)?;
    runtime.discard_session(&mut session)?;
    Ok(result)
}

fn begin_paused_warm_prefill(
    runtime: &mut Runtime<MetalBackend>,
    prefix: &[u32],
    prompt: &[u32],
    options: &GenerateOptions,
) -> TestResult<(
    GenerationSession<MetalBackend>,
    leone::PendingPrefill<MetalBackend>,
)> {
    let mut paused = GenerationSession::new();
    complete_prefill(runtime, &mut paused, prefix, options.clone())?;
    let pending = runtime.begin_prefill(&mut paused, prompt, options.clone())?;
    let budget = pending.minimum_budget();
    let pending = match runtime.advance_prefill(pending, budget, || false)? {
        PrefillProgress::Pending(pending) => pending,
        PrefillProgress::Ready(_) => {
            return Err(io::Error::other("warm prefill did not pause").into());
        }
        PrefillProgress::Cancelled(_) => {
            return Err(io::Error::other("warm prefill was cancelled").into());
        }
    };
    Ok((paused, pending))
}

fn run_unrelated_cold_generation(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
    options: &GenerateOptions,
) -> TestResult {
    let _cold = runtime.generate_tokens(prompt, options.clone(), |_| Ok(()), || false)?;
    Ok(())
}

fn finish_pending_prefill(
    runtime: &mut Runtime<MetalBackend>,
    mut pending: leone::PendingPrefill<MetalBackend>,
) -> TestResult<leone::ReadyPrefill<MetalBackend>> {
    loop {
        let budget = pending.minimum_budget();
        match runtime.advance_prefill(pending, budget, || false)? {
            PrefillProgress::Pending(next) => pending = next,
            PrefillProgress::Ready(ready) => return Ok(ready),
            PrefillProgress::Cancelled(_) => {
                return Err(io::Error::other("resumed prefill was cancelled").into());
            }
        }
    }
}

fn warm_tokens(runtime: &Runtime<MetalBackend>, count: usize, label: &str) -> TestResult<Vec<u32>> {
    let text = format!("A deterministic {label} exercises decode-equivalent warm prefill. ")
        .repeat(count / 8 + 32);
    let mut tokens = runtime.model().tokenizer().encode(&text)?;
    if tokens.len() < count {
        return Err(io::Error::other(format!("{label} did not reach {count} tokens")).into());
    }
    tokens.truncate(count);
    Ok(tokens)
}

fn model_path(expected_model: &str) -> TestResult<PathBuf> {
    let path = std::env::var_os("LEONE_METAL_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("LEONE_METAL_MODEL is required"))?;
    if !path.is_file() {
        return Err(io::Error::other(format!(
            "LEONE_METAL_MODEL is not a file: {}",
            path.display()
        ))
        .into());
    }
    if path.file_name().and_then(|name| name.to_str()) != Some(expected_model) {
        return Err(
            io::Error::other(format!("expected {expected_model}, got {}", path.display())).into(),
        );
    }
    Ok(path)
}

fn model_sha256(path: &Path) -> TestResult<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Deserialize)]
struct CompatibilityManifest {
    quality_gated_models: Vec<QualityGatedModel>,
}

#[derive(Deserialize)]
struct QualityGatedModel {
    architecture: String,
    model_sha256: String,
}

fn gated_model_sha256(architecture: &str) -> TestResult<String> {
    let manifest: CompatibilityManifest = serde_json::from_str(COMPATIBILITY_JSON)?;
    manifest
        .quality_gated_models
        .into_iter()
        .find(|model| model.architecture == architecture)
        .map(|model| model.model_sha256)
        .ok_or_else(|| io::Error::other(format!("no gated {architecture} model")).into())
}

fn assert_bitwise_zero(comparison: leone::PrefillBitwiseComparison, expected: usize, label: &str) {
    assert_eq!(comparison.compared, expected, "{label} cardinality");
    assert_eq!(comparison.mismatching, 0, "{label} changed bits");
}
