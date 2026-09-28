#![cfg(target_os = "macos")]

use leone::backend::{MemoryAccounting, MemoryClass};
use leone::{
    Backend, DecodeExecution, GenerateOptions, GenerationResult, GenerationSession, LogitCapture,
    PendingPrefill, PrefillProgress, ReadyPrefill, Runtime, SessionReuseClass,
};
use leone_metal::MetalBackend;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const QWEN_MODEL: &str = "Qwen3-8B-Q4_K_M.gguf";
const LLAMA_MODEL: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";
const COMPATIBILITY_JSON: &str = include_str!("../../../packaging/compatibility.json");
const FIXED_PROMPT_TOKENS: usize = 32;
const FIXED_CONTEXT_TOKENS: usize = 64;
const PREFILL_CHUNK_TOKENS: usize = 5;
const PREFIX_OUTPUT_TOKENS: usize = 2;
const CONTINUATION_OUTPUT_TOKENS: usize = 3;
const BRANCH_OUTPUT_TOKENS: usize = 3;
const LOGIT_TOLERANCE: f32 = 5e-3;

#[test]
#[ignore = "requires LEONE_METAL_MODEL=Qwen3-8B-Q4_K_M.gguf and an Apple GPU"]
fn qwen_metal_session_lifecycle() -> TestResult {
    run_lifecycle(QWEN_MODEL)
}

#[test]
#[ignore = "requires LEONE_METAL_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf and an Apple GPU"]
fn llama_metal_session_lifecycle() -> TestResult {
    run_lifecycle(LLAMA_MODEL)
}

fn run_lifecycle(expected_model: &str) -> TestResult {
    let model = model_path(expected_model)?;
    let model_digest = model_sha256(&model)?;
    let expected_digest = gated_model_sha256(expected_architecture(expected_model))?;
    assert_eq!(
        model_digest, expected_digest,
        "model SHA does not match packaging gate"
    );
    let mut runtime = Runtime::load(MetalBackend::new()?, &model)?;
    println!(
        "metal lifecycle model_sha256={model_digest} config={}",
        config_identity(&runtime)
    );
    assert_eq!(
        runtime.model().config().architecture.name(),
        expected_architecture(expected_model),
        "model architecture does not match the lifecycle gate"
    );
    let prompt = fixed_prompt(&runtime)?;
    run_lifecycle_checks(&mut runtime, &prompt)
}

fn expected_architecture(model_name: &str) -> &'static str {
    if model_name == QWEN_MODEL {
        "qwen3"
    } else {
        "llama"
    }
}

fn run_lifecycle_checks(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    assert!(prompt.len() > PREFILL_CHUNK_TOKENS);
    assert!(prompt.len() + PREFIX_OUTPUT_TOKENS + 1 + BRANCH_OUTPUT_TOKENS <= FIXED_CONTEXT_TOKENS);
    exact_repeat(runtime, prompt)?;
    resumable_prefill(runtime, prompt)?;
    cancel_decode_with_neighbor(runtime, prompt)?;
    warm_continuation(runtime, prompt)?;
    fork_branches(runtime, prompt)?;
    hibernate_wake(runtime, prompt)?;
    Ok(())
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
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("compatibility manifest has no gated {architecture} model"),
            )
            .into()
        })
}

fn model_path(expected_model: &str) -> TestResult<PathBuf> {
    let path = std::env::var_os("LEONE_METAL_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "LEONE_METAL_MODEL is required"))?;
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("LEONE_METAL_MODEL is not a file: {}", path.display()),
        )
        .into());
    }
    let actual_model = path.file_name().and_then(|name| name.to_str());
    if actual_model != Some(expected_model) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected {expected_model}, got {}", path.display()),
        )
        .into());
    }
    Ok(path)
}

fn fixed_prompt(runtime: &Runtime<MetalBackend>) -> TestResult<Vec<u32>> {
    let text =
        "A fixed context checks exact session state across Metal lifecycle paths. ".repeat(16);
    let mut tokens = runtime.model().tokenizer().encode(&text)?;
    if tokens.len() < FIXED_PROMPT_TOKENS {
        return Err(io::Error::other("lifecycle prompt did not reach the fixed length").into());
    }
    tokens.truncate(FIXED_PROMPT_TOKENS);
    Ok(tokens)
}

fn options(vocab_size: usize, max_tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::greedy(max_tokens);
    options.prefill_chunk_tokens = PREFILL_CHUNK_TOKENS;
    options.decode_execution = DecodeExecution::Eager;
    options.logit_capture =
        LogitCapture::Top(NonZeroUsize::new(vocab_size).expect("nonzero vocabulary"));
    options
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

fn config_identity(runtime: &Runtime<MetalBackend>) -> String {
    let config = runtime.model().config();
    format!(
        "arch={} layers={} heads={} kv_heads={} embd={} ff={} head_dim={} vocab={} context={} rope_theta_bits={:08x} rms_epsilon_bits={:08x}",
        config.architecture.name(),
        config.n_layer,
        config.n_head,
        config.n_head_kv,
        config.n_embd,
        config.n_ff,
        config.head_dim,
        config.vocab_size,
        config.context_length,
        config.rope_theta.to_bits(),
        config.rms_epsilon.to_bits(),
    )
}

fn exact_repeat(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    let baseline = runtime.backend().memory_accounting();
    let first = runtime.generate_tokens(
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    let second = runtime.generate_tokens(
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    assert_results_bitwise(
        &first,
        &second,
        runtime.vocab_size(),
        "same-history cold repeat",
    );
    assert_clean(runtime, &baseline, "exact repeat cleanup");
    Ok(())
}

fn resumable_prefill(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    let baseline = runtime.backend().memory_accounting();
    cancel_prefill(runtime, prompt, &baseline)?;
    paired_resumable_prefill(runtime, prompt)?;
    let resumed = resumed_prefill(runtime, prompt)?;
    let uninterrupted = uninterrupted_prefill(runtime, prompt)?;
    assert_results_bitwise(
        &resumed,
        &uninterrupted,
        runtime.vocab_size(),
        "same-history resumable versus uninterrupted prefill",
    );
    let reference = one_chunk_reference(runtime, prompt)?;
    assert_results_close(&resumed, &reference, "different prefill chunks");
    assert_clean(runtime, &baseline, "one chunk cleanup");
    Ok(())
}

fn paired_resumable_prefill(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    let baseline = runtime.backend().memory_accounting();
    let (mut subject, mut control) = paired_prefill_sessions(runtime, prompt)?;
    let (subject_result, control_result) =
        paired_prefill_results(runtime, &mut subject, &mut control, prompt)?;
    assert_results_bitwise(
        &subject_result,
        &control_result,
        runtime.vocab_size(),
        "same-history resumable prefill",
    );
    subject.invalidate();
    control.invalidate();
    assert_clean(runtime, &baseline, "paired resumable prefill cleanup");
    Ok(())
}

fn paired_prefill_sessions(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
) -> TestResult<(
    GenerationSession<MetalBackend>,
    GenerationSession<MetalBackend>,
)> {
    let mut subject = GenerationSession::new();
    let mut control = GenerationSession::new();
    let subject_pending = runtime.begin_prefill(
        &mut subject,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
    )?;
    let control_pending = runtime.begin_prefill(
        &mut control,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
    )?;
    let subject_ready = complete_prefill(runtime, &subject, subject_pending)?;
    let control_ready = complete_prefill(runtime, &control, control_pending)?;
    runtime.finish_prefill(subject_ready, &mut subject)?;
    runtime.finish_prefill(control_ready, &mut control)?;
    Ok((subject, control))
}

fn paired_prefill_results(
    runtime: &mut Runtime<MetalBackend>,
    subject: &mut GenerationSession<MetalBackend>,
    control: &mut GenerationSession<MetalBackend>,
    prompt: &[u32],
) -> TestResult<(GenerationResult, GenerationResult)> {
    let subject_result = runtime.generate_session_tokens(
        subject,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    let control_result = runtime.generate_session_tokens(
        control,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    Ok((subject_result, control_result))
}

fn cancel_prefill(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
    baseline: &MemoryAccounting,
) -> TestResult {
    let chunk = NonZeroUsize::new(PREFILL_CHUNK_TOKENS).expect("nonzero prefill chunk");
    let mut cancelled_session = GenerationSession::new();
    let pending = runtime.begin_prefill(
        &mut cancelled_session,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
    )?;
    let pending = pending_progress(runtime.advance_prefill(pending, chunk, || false)?)?;
    assert_eq!(pending.processed_tokens(), PREFILL_CHUNK_TOKENS);
    expect_cancelled_prefill(
        runtime,
        pending,
        chunk,
        || true,
        PREFILL_CHUNK_TOKENS,
        "prefill cancellation was ignored",
    )?;
    assert!(
        cancelled_session.is_empty(),
        "cancelled prefill retained session state"
    );

    let mut immediate_session = GenerationSession::new();
    let pending = runtime.begin_prefill(
        &mut immediate_session,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
    )?;
    expect_cancelled_prefill(
        runtime,
        pending,
        chunk,
        || true,
        0,
        "immediate prefill cancellation was ignored",
    )?;
    assert!(
        immediate_session.is_empty(),
        "immediate cancellation retained session state"
    );
    assert_clean(runtime, baseline, "cancelled prefill cleanup");
    Ok(())
}

fn expect_cancelled_prefill<C>(
    runtime: &mut Runtime<MetalBackend>,
    pending: PendingPrefill<MetalBackend>,
    chunk: NonZeroUsize,
    mut cancelled: C,
    expected_tokens: usize,
    error_message: &'static str,
) -> TestResult
where
    C: FnMut() -> bool,
{
    let progress = runtime.advance_prefill(pending, chunk, &mut cancelled)?;
    match progress {
        PrefillProgress::Cancelled(record) => {
            assert_eq!(record.processed_tokens(), expected_tokens)
        }
        PrefillProgress::Pending(_) | PrefillProgress::Ready(_) => {
            return Err(io::Error::other(error_message).into())
        }
    }
    Ok(())
}

fn resumed_prefill(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
) -> TestResult<GenerationResult> {
    let mut session = GenerationSession::new();
    let pending = runtime.begin_prefill(
        &mut session,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
    )?;
    let ready = complete_prefill(runtime, &session, pending)?;
    assert_eq!(ready.prompt_tokens(), prompt);
    assert_eq!(ready.processed_tokens(), prompt.len());
    runtime.finish_prefill(ready, &mut session)?;
    let resumed = runtime.generate_session_tokens(
        &mut session,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    assert_eq!(
        session.last_replay().reuse_class,
        SessionReuseClass::ExactRepeat
    );
    assert_eq!(session.last_replay().reused_tokens, prompt.len());
    session.invalidate();
    Ok(resumed)
}

fn one_chunk_reference(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
) -> TestResult<GenerationResult> {
    let mut one_chunk = options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS);
    one_chunk.prefill_chunk_tokens = prompt.len();
    Ok(runtime.generate_tokens(prompt, one_chunk, |_| Ok(()), || false)?)
}

fn uninterrupted_prefill(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
) -> TestResult<GenerationResult> {
    Ok(runtime.generate_tokens(
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?)
}

fn cancel_decode_with_neighbor(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    let baseline = runtime.backend().memory_accounting();
    let mut subject = GenerationSession::new();
    let subject_prefix = generate_prefix(runtime, &mut subject, prompt)?;
    let mut neighbor = GenerationSession::new();
    let neighbor_prefix = generate_prefix(runtime, &mut neighbor, prompt)?;
    let mut control = GenerationSession::new();
    let control_prefix = generate_prefix(runtime, &mut control, prompt)?;
    assert_results_bitwise(
        &neighbor_prefix,
        &control_prefix,
        runtime.vocab_size(),
        "same-history cancellation neighbor prefix",
    );
    let continuation_prompt = append_tokens(prompt, &subject_prefix.tokens);
    let before_cancel = runtime.backend().memory_accounting();
    let cancel_requested = Cell::new(false);
    let cancelled = runtime.generate_session_tokens(
        &mut subject,
        &continuation_prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| {
            cancel_requested.set(true);
            Ok(())
        },
        || cancel_requested.get(),
    )?;
    assert!(cancelled.stats.cancelled, "decode cancellation was ignored");
    assert!(
        !cancelled.tokens.is_empty(),
        "decode cancellation happened before a token"
    );
    assert!(
        subject.is_empty(),
        "cancelled decode retained session state"
    );
    let after_cancel = runtime.backend().memory_accounting();
    assert_session_allocations_drop(&before_cancel, &after_cancel, "cancelled decode");
    let neighbor_result = generate_continuation(runtime, &mut neighbor, &continuation_prompt)?;
    let control_result = generate_continuation(runtime, &mut control, &continuation_prompt)?;
    assert_results_bitwise(
        &neighbor_result,
        &control_result,
        runtime.vocab_size(),
        "same-history cancellation neighbor continuation",
    );
    neighbor.invalidate();
    control.invalidate();
    assert_clean(runtime, &baseline, "cancelled decode cleanup");
    Ok(())
}

fn complete_prefill(
    runtime: &mut Runtime<MetalBackend>,
    session: &GenerationSession<MetalBackend>,
    pending: PendingPrefill<MetalBackend>,
) -> TestResult<ReadyPrefill<MetalBackend>> {
    let mut progress = PrefillProgress::Pending(pending);
    let mut previous = 0;
    loop {
        let pending = pending_progress(progress)?;
        let before = pending.processed_tokens();
        assert_eq!(before, previous, "prefill progress regressed");
        let budget = NonZeroUsize::new(pending.options().prefill_chunk_tokens)
            .expect("nonzero prefill chunk");
        progress = runtime.advance_prefill(pending, budget, || false)?;
        assert!(session.is_empty(), "pending prefill exposed session state");
        update_prefill_progress(&progress, before, &mut previous);
        if let PrefillProgress::Ready(ready) = progress {
            return Ok(ready);
        }
    }
}

fn pending_progress(
    progress: PrefillProgress<MetalBackend>,
) -> TestResult<PendingPrefill<MetalBackend>> {
    match progress {
        PrefillProgress::Pending(pending) => Ok(pending),
        PrefillProgress::Ready(_) => Err(io::Error::other("prefill returned ready twice").into()),
        PrefillProgress::Cancelled(_) => {
            Err(io::Error::other("uncancelled prefill was cancelled").into())
        }
    }
}

fn update_prefill_progress(
    progress: &PrefillProgress<MetalBackend>,
    before: usize,
    previous: &mut usize,
) {
    if let PrefillProgress::Pending(next) = progress {
        assert!(next.processed_tokens() > before, "prefill made no progress");
        *previous = next.processed_tokens();
    }
    if let PrefillProgress::Ready(ready) = progress {
        assert!(
            ready.processed_tokens() > *previous,
            "prefill skipped its tail"
        );
    }
}

fn warm_continuation(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    let baseline = runtime.backend().memory_accounting();
    let (mut subject, mut control, continuation_prompt) = warm_pair_prefix(runtime, prompt)?;
    let (subject_result, _control_result) =
        warm_pair_results(runtime, &mut subject, &mut control, &continuation_prompt)?;
    assert!(subject.last_replay().reused_tokens >= prompt.len());
    assert!(control.last_replay().reused_tokens >= prompt.len());
    subject.invalidate();
    control.invalidate();
    let reference = runtime.generate_tokens(
        &continuation_prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    assert_results_close(&subject_result, &reference, "warm continuation cold path");
    assert_clean(runtime, &baseline, "warm continuation cleanup");
    Ok(())
}

fn warm_pair_prefix(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
) -> TestResult<(
    GenerationSession<MetalBackend>,
    GenerationSession<MetalBackend>,
    Vec<u32>,
)> {
    let mut subject = GenerationSession::new();
    let mut control = GenerationSession::new();
    let subject_prefix = generate_prefix(runtime, &mut subject, prompt)?;
    let control_prefix = generate_prefix(runtime, &mut control, prompt)?;
    assert_results_bitwise(
        &subject_prefix,
        &control_prefix,
        runtime.vocab_size(),
        "same-history warm prefix",
    );
    let continuation_prompt = append_tokens(prompt, &subject_prefix.tokens);
    Ok((subject, control, continuation_prompt))
}

fn warm_pair_results(
    runtime: &mut Runtime<MetalBackend>,
    subject: &mut GenerationSession<MetalBackend>,
    control: &mut GenerationSession<MetalBackend>,
    continuation_prompt: &[u32],
) -> TestResult<(GenerationResult, GenerationResult)> {
    let subject_result = generate_continuation(runtime, subject, continuation_prompt)?;
    let control_result = generate_continuation(runtime, control, continuation_prompt)?;
    assert_results_bitwise(
        &subject_result,
        &control_result,
        runtime.vocab_size(),
        "same-history warm continuation",
    );
    assert!(subject.last_replay().reused_tokens > 0);
    assert!(control.last_replay().reused_tokens > 0);
    Ok((subject_result, control_result))
}

fn fork_branches(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    let baseline = runtime.backend().memory_accounting();
    let (mut branches, branch_base) = start_fork_branches(runtime, prompt)?;
    let vocab = runtime.model().config().vocab_size;
    let prompts = branch_prompts(&branch_base, vocab);
    let results = generate_fork_traces(runtime, &mut branches, &prompts)?;
    let controls = independent_fork_results(runtime, prompt, &branches.prefix, &prompts)?;
    for (index, (trace, control)) in results.iter().zip(&controls).enumerate() {
        assert_eq!(trace.len(), control.len(), "fork branch {index} step count");
        for (step, (result, expected)) in trace.iter().zip(control).enumerate() {
            assert_results_bitwise(
                result,
                expected,
                runtime.vocab_size(),
                &format!("fork branch {index} step {step} control"),
            );
        }
    }
    assert_traces_diverge(&results, "fork branches");
    let before_drop = runtime.backend().memory_accounting();
    assert_session_allocations_rise(&baseline, &before_drop, "fork branches");
    drop_fork_branches(branches);
    let after_drop = runtime.backend().memory_accounting();
    assert_clean(runtime, &baseline, "fork branch cleanup");
    assert!(
        after_drop.frees > before_drop.frees,
        "dropping fork branches did not record frees"
    );
    Ok(())
}

struct ForkBranches {
    prefix: GenerationResult,
    parent: GenerationSession<MetalBackend>,
    child_a: GenerationSession<MetalBackend>,
    child_b: GenerationSession<MetalBackend>,
    grandchild: GenerationSession<MetalBackend>,
}

fn start_fork_branches(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
) -> TestResult<(ForkBranches, Vec<u32>)> {
    let mut parent = GenerationSession::new();
    let prefix = runtime.generate_session_tokens(
        &mut parent,
        prompt,
        options(runtime.vocab_size(), PREFIX_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    assert!(!prefix.tokens.is_empty(), "fork prefix emitted no tokens");
    let child_a = runtime.fork_session(&parent)?;
    let child_b = runtime.fork_session(&parent)?;
    let grandchild = runtime.fork_session(&child_a)?;
    let branch_base = append_tokens(prompt, &prefix.tokens);
    let branches = ForkBranches {
        prefix,
        parent,
        child_a,
        child_b,
        grandchild,
    };
    Ok((branches, branch_base))
}

fn branch_prompts(base: &[u32], vocab: usize) -> Vec<Vec<u32>> {
    (1..=4)
        .map(|marker| branch_prompt(base, marker, vocab))
        .collect()
}

fn generate_fork_traces(
    runtime: &mut Runtime<MetalBackend>,
    branches: &mut ForkBranches,
    prompts: &[Vec<u32>],
) -> TestResult<[Vec<GenerationResult>; 4]> {
    let mut histories = prompts.to_vec();
    let mut traces: [Vec<GenerationResult>; 4] =
        std::array::from_fn(|_| Vec::with_capacity(BRANCH_OUTPUT_TOKENS));
    for _ in 0..BRANCH_OUTPUT_TOKENS {
        let parent_history = branches.parent.evaluated_tokens().to_vec();
        for index in [1_usize, 2, 3] {
            let result = generate_fork_quantum(runtime, branches, index, &histories[index])?;
            append_result_tokens(&mut histories[index], &result);
            traces[index].push(result);
        }
        if traces[1].len() == 1 {
            assert_fork_reuse(branches);
        }
        assert_eq!(
            branches.parent.evaluated_tokens(),
            parent_history,
            "child branches changed parent state"
        );
        let result = generate_fork_quantum(runtime, branches, 0, &histories[0])?;
        append_result_tokens(&mut histories[0], &result);
        traces[0].push(result);
    }
    Ok(traces)
}

fn independent_fork_results(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
    expected_prefix: &GenerationResult,
    prompts: &[Vec<u32>],
) -> TestResult<[Vec<GenerationResult>; 4]> {
    let mut controls = build_control_forks(runtime, prompt, expected_prefix)?;
    let traces = generate_control_traces(runtime, &mut controls, prompts)?;
    drop_control_forks(&mut controls);
    Ok(traces)
}

struct ControlForks {
    active: [GenerationSession<MetalBackend>; 4],
    owners: Vec<GenerationSession<MetalBackend>>,
}

fn build_control_forks(
    runtime: &mut Runtime<MetalBackend>,
    prompt: &[u32],
    expected_prefix: &GenerationResult,
) -> TestResult<ControlForks> {
    let mut roots = Vec::with_capacity(4);
    for _ in 0..4 {
        let mut root = GenerationSession::new();
        let prefix = generate_prefix(runtime, &mut root, prompt)?;
        assert_results_bitwise(
            &prefix,
            expected_prefix,
            runtime.vocab_size(),
            "independent fork prefix",
        );
        roots.push(root);
    }
    let mut active: [GenerationSession<MetalBackend>; 4] =
        std::array::from_fn(|_| GenerationSession::new());
    let mut owners = Vec::with_capacity(4);
    active[0] = roots.remove(0);
    for active_child in &mut active[1..=2] {
        let root = roots.remove(0);
        let child = runtime.fork_session(&root)?;
        owners.push(root);
        *active_child = child;
    }
    let root = roots.remove(0);
    let child = runtime.fork_session(&root)?;
    let grandchild = runtime.fork_session(&child)?;
    owners.push(root);
    owners.push(child);
    active[3] = grandchild;
    Ok(ControlForks { active, owners })
}

fn generate_control_traces(
    runtime: &mut Runtime<MetalBackend>,
    controls: &mut ControlForks,
    prompts: &[Vec<u32>],
) -> TestResult<[Vec<GenerationResult>; 4]> {
    let mut histories = prompts.to_vec();
    let mut traces: [Vec<GenerationResult>; 4] =
        std::array::from_fn(|_| Vec::with_capacity(BRANCH_OUTPUT_TOKENS));
    for _ in 0..BRANCH_OUTPUT_TOKENS {
        for index in [1_usize, 2, 3, 0] {
            let result =
                generate_branch_quantum(runtime, &mut controls.active[index], &histories[index])?;
            append_result_tokens(&mut histories[index], &result);
            traces[index].push(result);
        }
    }
    Ok(traces)
}

fn drop_control_forks(controls: &mut ControlForks) {
    for session in &mut controls.active {
        session.invalidate();
    }
    for session in &mut controls.owners {
        session.invalidate();
    }
}

fn assert_fork_reuse(branches: &ForkBranches) {
    for session in [&branches.child_a, &branches.child_b, &branches.grandchild] {
        assert_eq!(
            session.last_replay().reuse_class,
            SessionReuseClass::DeviceFork
        );
    }
}

fn drop_fork_branches(mut branches: ForkBranches) {
    branches.parent.invalidate();
    branches.child_a.invalidate();
    branches.child_b.invalidate();
    branches.grandchild.invalidate();
}

fn generate_fork_quantum(
    runtime: &mut Runtime<MetalBackend>,
    branches: &mut ForkBranches,
    index: usize,
    prompt: &[u32],
) -> TestResult<GenerationResult> {
    match index {
        0 => generate_branch_quantum(runtime, &mut branches.parent, prompt),
        1 => generate_branch_quantum(runtime, &mut branches.child_a, prompt),
        2 => generate_branch_quantum(runtime, &mut branches.child_b, prompt),
        3 => generate_branch_quantum(runtime, &mut branches.grandchild, prompt),
        _ => Err(io::Error::other("fork branch index changed").into()),
    }
}

fn generate_branch_quantum(
    runtime: &mut Runtime<MetalBackend>,
    session: &mut GenerationSession<MetalBackend>,
    prompt: &[u32],
) -> TestResult<GenerationResult> {
    let result = runtime.generate_session_tokens(
        session,
        prompt,
        options(runtime.vocab_size(), 1),
        |_| Ok(()),
        || false,
    )?;
    assert_eq!(result.tokens.len(), 1, "branch quantum emitted no token");
    Ok(result)
}

fn append_result_tokens(history: &mut Vec<u32>, result: &GenerationResult) {
    history.extend_from_slice(&result.tokens);
}

fn generate_prefix(
    runtime: &mut Runtime<MetalBackend>,
    session: &mut GenerationSession<MetalBackend>,
    prompt: &[u32],
) -> TestResult<GenerationResult> {
    Ok(runtime.generate_session_tokens(
        session,
        prompt,
        options(runtime.vocab_size(), PREFIX_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?)
}

fn generate_continuation(
    runtime: &mut Runtime<MetalBackend>,
    session: &mut GenerationSession<MetalBackend>,
    prompt: &[u32],
) -> TestResult<GenerationResult> {
    Ok(runtime.generate_session_tokens(
        session,
        prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?)
}

fn hibernate_wake(runtime: &mut Runtime<MetalBackend>, prompt: &[u32]) -> TestResult {
    let baseline = runtime.backend().memory_accounting();
    let mut source = GenerationSession::new();
    let source_prefix = generate_prefix(runtime, &mut source, prompt)?;
    let mut control = GenerationSession::new();
    let control_prefix = generate_prefix(runtime, &mut control, prompt)?;
    assert_results_bitwise(
        &source_prefix,
        &control_prefix,
        runtime.vocab_size(),
        "same-history hibernate prefix",
    );
    let continuation_prompt = append_tokens(prompt, &source_prefix.tokens);
    let before_hibernate = runtime.backend().memory_accounting();
    let hibernated = runtime.hibernate_session(&mut source)?;
    let after_hibernate = check_hibernation(&source, &hibernated, runtime, &before_hibernate)?;
    let subject_result =
        wake_continuation(runtime, &hibernated, &continuation_prompt, &after_hibernate)?;
    let control_result = generate_continuation(runtime, &mut control, &continuation_prompt)?;
    assert_results_bitwise(
        &subject_result,
        &control_result,
        runtime.vocab_size(),
        "same-history host wake",
    );
    control.invalidate();
    let reference = runtime.generate_tokens(
        &continuation_prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    assert_results_close(&subject_result, &reference, "host wake cold path");
    assert_clean(runtime, &baseline, "host wake cleanup");
    Ok(())
}

fn check_hibernation(
    source: &GenerationSession<MetalBackend>,
    hibernated: &leone::HibernatedSession,
    runtime: &Runtime<MetalBackend>,
    before: &MemoryAccounting,
) -> TestResult<MemoryAccounting> {
    let record = hibernated.record();
    assert!(source.is_empty(), "hibernate retained device session state");
    assert!(record.host_bytes > 0, "hibernate recorded no host bytes");
    let after = runtime.backend().memory_accounting();
    assert_session_allocations_drop(before, &after, "hibernate");
    Ok(after)
}

fn wake_continuation(
    runtime: &mut Runtime<MetalBackend>,
    hibernated: &leone::HibernatedSession,
    continuation_prompt: &[u32],
    after_hibernate: &MemoryAccounting,
) -> TestResult<GenerationResult> {
    let mut woken = runtime.wake_session(hibernated)?;
    assert_eq!(woken.last_replay().reuse_class, SessionReuseClass::HostWake);
    let after_wake = runtime.backend().memory_accounting();
    assert_session_allocations_rise(after_hibernate, &after_wake, "wake");
    let continuation = runtime.generate_session_tokens(
        &mut woken,
        continuation_prompt,
        options(runtime.vocab_size(), CONTINUATION_OUTPUT_TOKENS),
        |_| Ok(()),
        || false,
    )?;
    assert_eq!(woken.last_replay().reuse_class, SessionReuseClass::HostWake);
    woken.invalidate();
    Ok(continuation)
}

fn append_tokens(prefix: &[u32], suffix: &[u32]) -> Vec<u32> {
    let mut tokens = prefix.to_vec();
    tokens.extend_from_slice(suffix);
    tokens
}

fn branch_prompt(base: &[u32], marker: u32, vocab_size: usize) -> Vec<u32> {
    let vocab = u32::try_from(vocab_size).expect("vocabulary fits u32");
    let mut prompt = base.to_vec();
    let previous = prompt.last().copied().unwrap_or_default();
    prompt.push(previous.wrapping_add(marker) % vocab);
    prompt
}

fn assert_results_bitwise(
    actual: &GenerationResult,
    expected: &GenerationResult,
    vocab_size: usize,
    label: &str,
) {
    assert_eq!(actual.tokens, expected.tokens, "{label} sampled tokens");
    assert_eq!(
        actual.logits.len(),
        expected.logits.len(),
        "{label} logit count"
    );
    assert!(!actual.logits.is_empty(), "{label} did not capture logits");
    for (index, (left, right)) in actual.logits.iter().zip(&expected.logits).enumerate() {
        assert_eq!(
            left.input_position, right.input_position,
            "{label} position {index}"
        );
        assert_eq!(
            left.sampled_token, right.sampled_token,
            "{label} sampled {index}"
        );
        let left = normalized_logits(left);
        let right = normalized_logits(right);
        assert_eq!(left.len(), vocab_size, "{label} left vocabulary {index}");
        assert_eq!(right.len(), vocab_size, "{label} right vocabulary {index}");
        for (rank, (left, right)) in left.iter().zip(&right).enumerate() {
            assert_eq!(left.token, right.token, "{label} token {index}:{rank}");
            assert_eq!(
                left.value.to_bits(),
                right.value.to_bits(),
                "{label} value bits {index}:{rank}"
            );
        }
    }
}

fn normalized_logits(snapshot: &leone::LogitSnapshot) -> Vec<leone::Logit> {
    let mut logits = snapshot.top.clone();
    logits.sort_unstable_by_key(|logit| logit.token);
    logits
}

fn assert_results_close(actual: &GenerationResult, expected: &GenerationResult, label: &str) {
    assert_eq!(
        actual.logits.len(),
        expected.logits.len(),
        "{label} logit count"
    );
    assert!(!actual.logits.is_empty(), "{label} did not capture logits");
    for (index, (left, right)) in actual.logits.iter().zip(&expected.logits).enumerate() {
        assert_eq!(
            left.input_position, right.input_position,
            "{label} position {index}"
        );
        let left = normalized_logits(left);
        let right = normalized_logits(right);
        assert_eq!(left.len(), right.len(), "{label} top count {index}");
        for (rank, (left, right)) in left.iter().zip(&right).enumerate() {
            assert_eq!(left.token, right.token, "{label} token {index}:{rank}");
            let scale = left.value.abs().max(right.value.abs()).max(1.0);
            let difference = (left.value - right.value).abs();
            assert!(
                difference <= LOGIT_TOLERANCE * scale,
                "{label} value {index}:{rank} differs by {difference}"
            );
        }
    }
}

fn assert_traces_diverge(traces: &[Vec<GenerationResult>; 4], label: &str) {
    let first = traces.first().expect("branch trace");
    for (index, candidate) in traces.iter().enumerate().skip(1) {
        assert!(
            traces_differ(first, candidate),
            "{label} branch {index} matched the parent at every quantum"
        );
    }
}

fn traces_differ(left: &[GenerationResult], right: &[GenerationResult]) -> bool {
    left.iter()
        .zip(right)
        .any(|(left, right)| results_differ(left, right))
}

fn results_differ(left: &GenerationResult, right: &GenerationResult) -> bool {
    left.tokens != right.tokens
        || left.logits.iter().zip(&right.logits).any(|(left, right)| {
            normalized_logits(left)
                .iter()
                .zip(normalized_logits(right).iter())
                .any(|(left, right)| {
                    left.token != right.token || left.value.to_bits() != right.value.to_bits()
                })
        })
}

fn assert_session_allocations_rise(
    baseline: &MemoryAccounting,
    active: &MemoryAccounting,
    label: &str,
) {
    for class in [MemoryClass::KvCache, MemoryClass::Activation] {
        assert!(
            active.class(class).live_bytes > baseline.class(class).live_bytes,
            "{label} {class:?} bytes did not rise"
        );
    }
}

fn assert_session_allocations_drop(
    before: &MemoryAccounting,
    after: &MemoryAccounting,
    label: &str,
) {
    for class in [MemoryClass::KvCache, MemoryClass::Activation] {
        assert!(
            after.class(class).live_bytes < before.class(class).live_bytes,
            "{label} {class:?} bytes did not fall"
        );
    }
}

fn assert_clean(runtime: &Runtime<MetalBackend>, baseline: &MemoryAccounting, label: &str) {
    let observed = runtime.backend().memory_accounting();
    assert_eq!(
        observed.live_allocations, baseline.live_allocations,
        "{label} allocations"
    );
    for class in [MemoryClass::KvCache, MemoryClass::Activation] {
        assert_eq!(
            observed.class(class).live_bytes,
            baseline.class(class).live_bytes,
            "{label} {class:?} bytes"
        );
    }
}
