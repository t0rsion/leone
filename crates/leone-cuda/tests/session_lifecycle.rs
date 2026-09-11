use leone::backend::{MemoryAccounting, MemoryClass};
use leone::{
    Backend, BatchSession, DecodeExecution, GenerateOptions, GenerationSession, KvCacheDtype,
    PendingPrefill, PrefillProgress, ReadyPrefill, Runtime, Sampler,
};
use leone_cuda::CudaBackend;
use std::cell::Cell;
use std::error::Error;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const PREFILL_CHUNK_TOKENS: usize = 4;
const GRAPH_STEPS: usize = 32;
const BOUNDARY_PROMPT_TOKENS: usize = 500;
const BOUNDARY_OUTPUT_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug)]
struct MatrixCase {
    dtype: KvCacheDtype,
    stochastic: bool,
}

#[derive(Debug)]
struct ChurnState {
    session: GenerationSession<CudaBackend>,
    transcript: Vec<u32>,
    output: Vec<u32>,
    options: GenerateOptions,
    arrival_step: usize,
}

struct ForkRun {
    source: GenerationSession<CudaBackend>,
    child: GenerationSession<CudaBackend>,
    parent_prompt: Vec<u32>,
    child_prompt: Vec<u32>,
    parent_tokens: Vec<u32>,
    child_tokens: Vec<u32>,
}

struct WakeRun {
    session: GenerationSession<CudaBackend>,
    prompt: Vec<u32>,
    tokens: Vec<u32>,
}

struct GraphRun {
    outputs: Vec<Vec<u32>>,
    live_memory: MemoryAccounting,
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn qwen_graph_churn_matches_isolated_matrix() -> TestResult {
    run_graph_churn_matrix(&model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Llama 3.2 Q4_K_M model"]
fn llama_graph_churn_matches_isolated_matrix() -> TestResult {
    run_graph_churn_matrix(&llama_model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn qwen_session_lifecycle_matches_independent_continuations() -> TestResult {
    run_lifecycle_matrix(&model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Llama 3.2 Q4_K_M model"]
fn llama_session_lifecycle_matches_independent_continuations() -> TestResult {
    run_lifecycle_matrix(&llama_model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn qwen_prefill_suspend_cancel_and_drop_matrix() -> TestResult {
    run_prefill_matrix(&model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Llama 3.2 Q4_K_M model"]
fn llama_prefill_suspend_cancel_and_drop_matrix() -> TestResult {
    run_prefill_matrix(&llama_model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn qwen_context_boundary_prefill_matches_reference() -> TestResult {
    context_boundary_case(&model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Llama 3.2 Q4_K_M model"]
fn llama_context_boundary_prefill_matches_reference() -> TestResult {
    context_boundary_case(&llama_model_path())
}

fn run_graph_churn_matrix(path: &Path) -> TestResult {
    for case in matrix_cases() {
        graph_churn_case(path, case)?;
    }
    Ok(())
}

fn graph_churn_case(path: &Path, case: MatrixCase) -> TestResult {
    let prompts = churn_prompts();
    let mut reference = Runtime::load(CudaBackend::new(0)?, path)?;
    let prompt_tokens = encode_prompts(&reference, &prompts)?;
    let options = churn_options(case);
    let expected = isolated_transcripts(&mut reference, &prompt_tokens, &options)?;
    drop(reference);

    let mut batched = Runtime::load(CudaBackend::new(0)?, path)?;
    let baseline = batched.backend().memory_accounting();
    let run = run_staggered_graph(&mut batched, &prompt_tokens, &options, &expected)?;
    assert_session_allocations_rise(&baseline, &run.live_memory, "graph churn sessions");
    assert_eq!(run.outputs, expected, "graph churn case {case:?}");
    assert_session_allocations_match(&batched, &baseline, "graph churn cleanup");
    Ok(())
}

fn run_staggered_graph(
    runtime: &mut Runtime<CudaBackend>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
    expected: &[Vec<u32>],
) -> TestResult<GraphRun> {
    let mut states = prompts
        .iter()
        .zip(options)
        .enumerate()
        .map(|(index, (prompt, options))| ChurnState {
            session: GenerationSession::new(),
            transcript: prompt.clone(),
            output: Vec::new(),
            options: options.clone(),
            arrival_step: index,
        })
        .collect::<Vec<_>>();

    for step in 0..GRAPH_STEPS {
        initialize_arrivals(runtime, &mut states, expected, step)?;
        let active = active_indices(&states, expected, step);
        if active.is_empty() {
            if all_finished(&states, expected) {
                break;
            }
            continue;
        }
        advance_active(runtime, &mut states, &active)?;
    }
    assert!(
        all_finished(&states, expected),
        "graph churn did not finish"
    );
    let live_memory = runtime.backend().memory_accounting();
    let outputs = states.iter().map(|state| state.output.clone()).collect();
    Ok(GraphRun {
        outputs,
        live_memory,
    })
}

fn initialize_arrivals(
    runtime: &mut Runtime<CudaBackend>,
    states: &mut [ChurnState],
    expected: &[Vec<u32>],
    step: usize,
) -> TestResult {
    for (index, state) in states.iter_mut().enumerate() {
        if state.arrival_step != step || !state.output.is_empty() {
            continue;
        }
        let mut first_options = state.options.clone();
        first_options.max_tokens = 1;
        let result = runtime.generate_session_tokens(
            &mut state.session,
            &state.transcript,
            first_options,
            |_| Ok(()),
            || false,
        )?;
        assert_eq!(
            result.tokens.len(),
            1,
            "arrival {index} did not emit one token"
        );
        state.transcript.extend_from_slice(&result.tokens);
        state.output.extend(result.tokens);
        assert_eq!(state.output[0], expected[index][0], "arrival {index} token");
    }
    Ok(())
}

fn active_indices(states: &[ChurnState], expected: &[Vec<u32>], step: usize) -> Vec<usize> {
    states
        .iter()
        .enumerate()
        .filter(|(_, state)| state.arrival_step <= step)
        .filter(|(index, state)| state.output.len() < expected[*index].len())
        .map(|(index, _)| index)
        .collect()
}

fn all_finished(states: &[ChurnState], expected: &[Vec<u32>]) -> bool {
    states
        .iter()
        .zip(expected)
        .all(|(state, expected)| state.output == *expected)
}

fn advance_active(
    runtime: &mut Runtime<CudaBackend>,
    states: &mut [ChurnState],
    active: &[usize],
) -> TestResult {
    let mut inputs = states
        .iter_mut()
        .enumerate()
        .filter(|(index, _)| active.contains(index))
        .map(|(_, state)| BatchSession {
            session: &mut state.session,
            transcript: &state.transcript,
            options: &state.options,
        })
        .collect::<Vec<_>>();
    let tokens = runtime.generate_session_batch_token(&mut inputs)?;
    drop(inputs);
    assert_eq!(tokens.len(), active.len(), "batch token count");
    for (index, token) in active.iter().zip(tokens) {
        states[*index].transcript.push(token.id);
        states[*index].output.push(token.id);
    }
    Ok(())
}

fn run_lifecycle_matrix(path: &Path) -> TestResult {
    for case in matrix_cases() {
        lifecycle_case(path, case)?;
    }
    Ok(())
}

fn lifecycle_case(path: &Path, case: MatrixCase) -> TestResult {
    let mut runtime = Runtime::load(CudaBackend::new(0)?, path)?;
    let baseline = runtime.backend().memory_accounting();
    let prompt_tokens = runtime.model().tokenizer().encode(lifecycle_prompt())?;
    let fork = run_fork(&mut runtime, &prompt_tokens, case)?;
    let wake = run_hibernation(&mut runtime, &prompt_tokens, case, &baseline)?;
    let live_memory = runtime.backend().memory_accounting();
    assert_session_allocations_rise(&baseline, &live_memory, "session lifecycle");
    let continuation_options = lifecycle_options(case, 5);
    let actual_parent = fork.parent_tokens.clone();
    let actual_child = fork.child_tokens.clone();
    let actual_wake = wake.tokens.clone();
    let parent_prompt = fork.parent_prompt.clone();
    let child_prompt = fork.child_prompt.clone();
    let wake_prompt = wake.prompt.clone();
    let mut source = fork.source;
    let mut child = fork.child;
    let mut woken = wake.session;
    source.invalidate();
    child.invalidate();
    woken.invalidate();
    assert_session_allocations_match(&runtime, &baseline, "session lifecycle cleanup");
    drop(runtime);
    compare_lifecycle_reference(
        path,
        &parent_prompt,
        &child_prompt,
        &wake_prompt,
        continuation_options,
        (&actual_parent, &actual_child, &actual_wake),
        case,
    )
}

fn run_fork(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    case: MatrixCase,
) -> TestResult<ForkRun> {
    let (mut source, parent_prompt, child_prompt) = fork_prefix(runtime, prompt_tokens, case)?;
    let mut child = runtime.fork_session(&source)?;
    let (parent_tokens, child_tokens) = fork_continuations(
        runtime,
        &mut source,
        &mut child,
        &parent_prompt,
        &child_prompt,
        case,
    )?;
    Ok(ForkRun {
        source,
        child,
        parent_prompt,
        child_prompt,
        parent_tokens,
        child_tokens,
    })
}

fn fork_prefix(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    case: MatrixCase,
) -> TestResult<(GenerationSession<CudaBackend>, Vec<u32>, Vec<u32>)> {
    let mut source = GenerationSession::new();
    let prefix = runtime.generate_session_tokens(
        &mut source,
        prompt_tokens,
        lifecycle_options(case, 3),
        |_| Ok(()),
        || false,
    )?;
    assert!(!source.is_empty(), "prefix session has no live state");
    let mut parent_prompt = prompt_tokens.to_vec();
    parent_prompt.extend_from_slice(&prefix.tokens);
    let child_prompt = append_branch_token(&parent_prompt, runtime.vocab_size());
    Ok((source, parent_prompt, child_prompt))
}

fn fork_continuations(
    runtime: &mut Runtime<CudaBackend>,
    source: &mut GenerationSession<CudaBackend>,
    child: &mut GenerationSession<CudaBackend>,
    parent_prompt: &[u32],
    child_prompt: &[u32],
    case: MatrixCase,
) -> TestResult<(Vec<u32>, Vec<u32>)> {
    let options = lifecycle_options(case, 5);
    let parent = runtime.generate_session_tokens(
        source,
        parent_prompt,
        options.clone(),
        |_| Ok(()),
        || false,
    )?;
    let child_result =
        runtime.generate_session_tokens(child, child_prompt, options, |_| Ok(()), || false)?;
    assert_eq!(
        child.last_replay().reuse_class,
        leone::SessionReuseClass::DeviceFork
    );
    assert!(child.last_fork().is_some(), "fork record was dropped");
    Ok((parent.tokens, child_result.tokens))
}

fn compare_lifecycle_reference(
    path: &Path,
    parent_prompt: &[u32],
    child_prompt: &[u32],
    wake_prompt: &[u32],
    options: GenerateOptions,
    actual: (&[u32], &[u32], &[u32]),
    case: MatrixCase,
) -> TestResult {
    let mut reference = Runtime::load(CudaBackend::new(0)?, path)?;
    let expected_parent = reference_continuation(&mut reference, parent_prompt, options.clone())?;
    let expected_child = reference_continuation(&mut reference, child_prompt, options.clone())?;
    let expected_wake = reference_continuation(&mut reference, wake_prompt, options)?;
    assert_eq!(actual.0, expected_parent, "parent continuation {case:?}");
    assert_eq!(actual.1, expected_child, "fork continuation {case:?}");
    assert_eq!(actual.2, expected_wake, "wake continuation {case:?}");
    Ok(())
}

fn run_hibernation(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    case: MatrixCase,
    baseline: &MemoryAccounting,
) -> TestResult<WakeRun> {
    let mut hibernating = GenerationSession::new();
    let prefix = runtime.generate_session_tokens(
        &mut hibernating,
        prompt_tokens,
        lifecycle_options(case, 3),
        |_| Ok(()),
        || false,
    )?;
    let before_hibernate = runtime.backend().memory_accounting();
    assert_session_allocations_rise(baseline, &before_hibernate, "hibernate source");
    let hibernated = runtime.hibernate_session(&mut hibernating)?;
    assert!(hibernating.is_empty(), "hibernate retained device state");
    assert!(
        hibernated.record().host_bytes > 0,
        "hibernate copied no bytes"
    );
    let after_hibernate = runtime.backend().memory_accounting();
    assert_session_allocations_drop(
        &before_hibernate,
        &after_hibernate,
        "hibernate source release",
    );
    let mut woken = runtime.wake_session(hibernated)?;
    let mut wake_prompt = prompt_tokens.to_vec();
    wake_prompt.extend_from_slice(&prefix.tokens);
    let wake = runtime.generate_session_tokens(
        &mut woken,
        &wake_prompt,
        lifecycle_options(case, 5),
        |_| Ok(()),
        || false,
    )?;
    assert_eq!(
        woken.last_replay().reuse_class,
        leone::SessionReuseClass::HostWake
    );
    Ok(WakeRun {
        session: woken,
        prompt: wake_prompt,
        tokens: wake.tokens,
    })
}

fn reference_continuation(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    options: GenerateOptions,
) -> TestResult<Vec<u32>> {
    let mut session = GenerationSession::new();
    let base = runtime.model().tokenizer().encode(lifecycle_prompt())?;
    let mut prefix_options = options.clone();
    prefix_options.max_tokens = 3;
    let prefix = runtime.generate_session_tokens(
        &mut session,
        &base,
        prefix_options,
        |_| Ok(()),
        || false,
    )?;
    let mut evaluated_prompt = base;
    evaluated_prompt.extend_from_slice(&prefix.tokens);
    assert!(
        prompt.starts_with(&evaluated_prompt),
        "independent prefix differs"
    );
    Ok(runtime
        .generate_session_tokens(&mut session, prompt, options, |_| Ok(()), || false)?
        .tokens)
}

fn reference_generation(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    options: GenerateOptions,
) -> TestResult<Vec<u32>> {
    let mut session = GenerationSession::new();
    Ok(runtime
        .generate_session_tokens(&mut session, prompt, options, |_| Ok(()), || false)?
        .tokens)
}

fn assert_session_allocations_rise(
    baseline: &MemoryAccounting,
    active: &MemoryAccounting,
    label: &str,
) {
    for class in [MemoryClass::KvCache, MemoryClass::Activation] {
        let before = baseline.class(class).live_bytes;
        let during = active.class(class).live_bytes;
        assert!(
            during > before,
            "{label} {class:?} bytes did not rise: {before} to {during}"
        );
    }
}

fn assert_session_allocations_drop(
    before: &MemoryAccounting,
    after: &MemoryAccounting,
    label: &str,
) {
    for class in [MemoryClass::KvCache, MemoryClass::Activation] {
        let held = before.class(class).live_bytes;
        let remaining = after.class(class).live_bytes;
        assert!(
            remaining < held,
            "{label} {class:?} bytes did not fall: {held} to {remaining}"
        );
    }
}

fn assert_session_allocations_match(
    runtime: &Runtime<CudaBackend>,
    baseline: &MemoryAccounting,
    label: &str,
) {
    let observed = runtime.backend().memory_accounting();
    assert!(
        observed.peak_live_bytes >= observed.live_bytes,
        "{label} peak bytes fell below live bytes"
    );
    for class in [MemoryClass::KvCache, MemoryClass::Activation] {
        assert_eq!(
            observed.class(class).live_bytes,
            baseline.class(class).live_bytes,
            "{label} {class:?} bytes"
        );
    }
}

fn run_prefill_matrix(path: &Path) -> TestResult {
    for dtype in [KvCacheDtype::F16, KvCacheDtype::Q8] {
        prefill_case(path, dtype)?;
    }
    Ok(())
}

fn prefill_case(path: &Path, dtype: KvCacheDtype) -> TestResult {
    prefill_ready_case(path, dtype)?;
    prefill_cancel_case(path, dtype)
}

fn prefill_ready_case(path: &Path, dtype: KvCacheDtype) -> TestResult {
    let mut runtime = Runtime::load(CudaBackend::new(0)?, path)?;
    let prompt = runtime.model().tokenizer().encode(long_prompt())?;
    assert!(
        prompt.len() > PREFILL_CHUNK_TOKENS,
        "prefill fixture has one chunk"
    );
    let options = prefill_options(dtype);
    let baseline = runtime.backend().memory_accounting();
    let mut session = GenerationSession::new();
    check_dropped_prefill(&mut runtime, &mut session, &prompt, &options, &baseline)?;
    let actual = finish_resumable_prefill(
        &mut runtime,
        &mut session,
        &prompt,
        options.clone(),
        &baseline,
    )?;
    session.invalidate();
    assert_session_allocations_match(&runtime, &baseline, "ready prefill cleanup");
    drop(runtime);

    let mut reference = Runtime::load(CudaBackend::new(0)?, path)?;
    let expected = reference_generation(&mut reference, &prompt, options)?;
    assert_eq!(actual, expected, "resumed prefill continuation {dtype:?}");
    Ok(())
}

fn check_dropped_prefill(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    prompt: &[u32],
    options: &GenerateOptions,
    baseline: &MemoryAccounting,
) -> TestResult {
    let dropped = runtime.begin_prefill(session, prompt, options.clone())?;
    assert!(session.is_empty(), "pending prefill populated its session");
    let dropped_memory = runtime.backend().memory_accounting();
    assert_session_allocations_rise(baseline, &dropped_memory, "dropped prefill");
    drop(dropped);
    assert!(session.is_empty(), "dropped prefill retained session state");
    assert_session_allocations_match(runtime, baseline, "dropped prefill cleanup");
    Ok(())
}

fn finish_resumable_prefill(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    prompt: &[u32],
    options: GenerateOptions,
    baseline: &MemoryAccounting,
) -> TestResult<Vec<u32>> {
    let pending = runtime.begin_prefill(session, prompt, options.clone())?;
    let pending_memory = runtime.backend().memory_accounting();
    assert_session_allocations_rise(baseline, &pending_memory, "pending prefill");
    let ready = drive_prefill(runtime, session, pending)?;
    assert_eq!(ready.prompt_tokens(), prompt);
    assert_eq!(ready.processed_tokens(), prompt.len());
    runtime.finish_prefill(ready, session)?;
    assert!(!session.is_empty(), "finished prefill did not commit state");
    let ready_memory = runtime.backend().memory_accounting();
    assert_session_allocations_rise(baseline, &ready_memory, "ready prefill");
    let result = runtime.generate_session_tokens(session, prompt, options, |_| Ok(()), || false)?;
    assert_eq!(
        session.last_replay().reuse_class,
        leone::SessionReuseClass::ExactRepeat
    );
    assert_eq!(session.last_replay().reused_tokens, prompt.len());
    Ok(result.tokens)
}

fn prefill_cancel_case(path: &Path, dtype: KvCacheDtype) -> TestResult {
    let mut runtime = Runtime::load(CudaBackend::new(0)?, path)?;
    let prompt = runtime.model().tokenizer().encode(long_prompt())?;
    let baseline = runtime.backend().memory_accounting();
    let mut cancelled_session = GenerationSession::new();
    let cancelled_pending =
        runtime.begin_prefill(&mut cancelled_session, &prompt, prefill_options(dtype))?;
    assert!(
        cancelled_session.is_empty(),
        "cancelled prefill populated session"
    );
    let pending_memory = runtime.backend().memory_accounting();
    assert_session_allocations_rise(&baseline, &pending_memory, "cancelled prefill");
    let checks = Cell::new(0_usize);
    let cancelled = runtime.advance_prefill(
        cancelled_pending,
        NonZeroUsize::new(PREFILL_CHUNK_TOKENS).expect("prefill chunk is nonzero"),
        || {
            let next = checks.get() + 1;
            checks.set(next);
            next >= 2
        },
    )?;
    match cancelled {
        PrefillProgress::Cancelled(record) => {
            assert_eq!(record.processed_tokens(), PREFILL_CHUNK_TOKENS);
        }
        PrefillProgress::Pending(_) | PrefillProgress::Ready(_) => {
            return Err(std::io::Error::other("prefill cancellation was ignored").into())
        }
    }
    assert!(checks.get() >= 2, "prefill did not yield to cancellation");
    assert!(
        cancelled_session.is_empty(),
        "cancelled prefill retained session state"
    );
    assert_session_allocations_match(&runtime, &baseline, "cancelled prefill cleanup");
    check_resumed_cancellation(&mut runtime, &prompt, prefill_options(dtype), &baseline)
}

fn check_resumed_cancellation(
    runtime: &mut Runtime<CudaBackend>,
    prompt: &[u32],
    options: GenerateOptions,
    baseline: &MemoryAccounting,
) -> TestResult {
    let mut session = GenerationSession::new();
    let pending = runtime.begin_prefill(&mut session, prompt, options.clone())?;
    let pending = advance_pending_once(runtime, pending)?;
    assert_eq!(pending.processed_tokens(), PREFILL_CHUNK_TOKENS);
    let budget = pending.minimum_budget();
    let cancelled = runtime.advance_prefill(pending, budget, || true)?;
    let PrefillProgress::Cancelled(record) = cancelled else {
        return Err(std::io::Error::other("resumed prefill ignored cancellation").into());
    };
    assert_eq!(record.processed_tokens(), PREFILL_CHUNK_TOKENS);
    assert_session_allocations_match(runtime, baseline, "resumed cancellation");
    let next = runtime.begin_prefill(&mut session, prompt, options)?;
    let ready = complete_pending(runtime, next)?;
    assert_eq!(ready.processed_tokens(), prompt.len());
    drop(ready);
    assert_session_allocations_match(runtime, baseline, "fresh prefill after cancellation");
    Ok(())
}

fn prefill_options(dtype: KvCacheDtype) -> GenerateOptions {
    let mut options = lifecycle_options(
        MatrixCase {
            dtype,
            stochastic: false,
        },
        5,
    );
    options.prefill_chunk_tokens = PREFILL_CHUNK_TOKENS;
    options
}

fn pending_from_progress(
    progress: PrefillProgress<CudaBackend>,
) -> TestResult<PendingPrefill<CudaBackend>> {
    match progress {
        PrefillProgress::Pending(pending) => Ok(pending),
        PrefillProgress::Ready(_) => {
            Err(std::io::Error::other("prefill completed before its prompt").into())
        }
        PrefillProgress::Cancelled(_) => {
            Err(std::io::Error::other("prefill cancelled without a request").into())
        }
    }
}

fn context_boundary_case(path: &Path) -> TestResult {
    let (prompt, options, expected) = context_boundary_reference(path)?;
    context_boundary_actual(path, &prompt, options, &expected)
}

fn context_boundary_reference(path: &Path) -> TestResult<(Vec<u32>, GenerateOptions, Vec<u32>)> {
    let mut reference = Runtime::load(CudaBackend::new(0)?, path)?;
    let seed = reference.model().tokenizer().encode("Boundary")?[0];
    let prompt = vec![seed; BOUNDARY_PROMPT_TOKENS];
    let options = lifecycle_options(
        MatrixCase {
            dtype: KvCacheDtype::F16,
            stochastic: false,
        },
        BOUNDARY_OUTPUT_TOKENS,
    );
    let expected = reference_generation(&mut reference, &prompt, options.clone())?;
    drop(reference);
    Ok((prompt, options, expected))
}

fn context_boundary_actual(
    path: &Path,
    prompt: &[u32],
    options: GenerateOptions,
    expected: &[u32],
) -> TestResult {
    let mut runtime = Runtime::load(CudaBackend::new(0)?, path)?;
    let baseline = runtime.backend().memory_accounting();
    let mut session = GenerationSession::new();
    let pending = runtime.begin_prefill(&mut session, prompt, options.clone())?;
    let pending_memory = runtime.backend().memory_accounting();
    assert_session_allocations_rise(&baseline, &pending_memory, "context-boundary pending");
    let ready = drive_prefill(&mut runtime, &mut session, pending)?;
    runtime.finish_prefill(ready, &mut session)?;
    let ready_memory = runtime.backend().memory_accounting();
    assert_session_allocations_rise(&baseline, &ready_memory, "context-boundary ready");
    let actual = decode_in_quanta(&mut runtime, &mut session, prompt, options)?;
    assert_eq!(actual, expected, "context-boundary continuation");
    assert!(actual.len() <= BOUNDARY_OUTPUT_TOKENS);
    session.invalidate();
    assert_session_allocations_match(&runtime, &baseline, "context boundary cleanup");
    Ok(())
}

fn decode_in_quanta(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    prompt: &[u32],
    options: GenerateOptions,
) -> TestResult<Vec<u32>> {
    let mut transcript = prompt.to_vec();
    let mut output = Vec::new();
    while output.len() < options.max_tokens {
        let mut quantum = options.clone();
        quantum.max_tokens = 4.min(options.max_tokens - output.len());
        let result =
            runtime.generate_session_tokens(session, &transcript, quantum, |_| Ok(()), || false)?;
        assert!(session.last_replay().reused_tokens >= prompt.len());
        if result.tokens.is_empty() {
            break;
        }
        transcript.extend_from_slice(&result.tokens);
        output.extend_from_slice(&result.tokens);
    }
    Ok(output)
}

fn drive_prefill(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    pending: PendingPrefill<CudaBackend>,
) -> TestResult<ReadyPrefill<CudaBackend>> {
    let mut progress = PrefillProgress::Pending(pending);
    let mut previous = 0_usize;
    let mut advances = 0_usize;
    loop {
        let pending = pending_from_progress(progress)?;
        let before = pending.processed_tokens();
        assert_eq!(before, previous, "prefill progress regressed");
        let chunk = pending.options().prefill_chunk_tokens;
        let budget = NonZeroUsize::new(chunk).expect("prefill chunk is nonzero");
        let budget_tokens = budget.get();
        progress = runtime.advance_prefill(pending, budget, || false)?;
        advances += 1;
        assert!(session.is_empty(), "pending prefill exposed session state");
        match progress {
            PrefillProgress::Pending(next) => {
                assert!(next.processed_tokens() > before, "prefill made no progress");
                assert!(
                    next.processed_tokens() - before <= budget_tokens,
                    "prefill exceeded its chunk budget"
                );
                previous = next.processed_tokens();
                progress = PrefillProgress::Pending(next);
            }
            PrefillProgress::Ready(ready) => {
                assert!(advances > 1, "prefill did not yield between chunks");
                return Ok(ready);
            }
            PrefillProgress::Cancelled(_) => {
                return Err(std::io::Error::other("uncancelled prefill was cancelled").into())
            }
        }
    }
}

fn matrix_cases() -> [MatrixCase; 4] {
    [
        MatrixCase {
            dtype: KvCacheDtype::F16,
            stochastic: false,
        },
        MatrixCase {
            dtype: KvCacheDtype::F16,
            stochastic: true,
        },
        MatrixCase {
            dtype: KvCacheDtype::Q8,
            stochastic: false,
        },
        MatrixCase {
            dtype: KvCacheDtype::Q8,
            stochastic: true,
        },
    ]
}

fn churn_options(case: MatrixCase) -> Vec<GenerateOptions> {
    [5, 7, 9, 6]
        .into_iter()
        .enumerate()
        .map(|(index, tokens)| {
            let mut options = lifecycle_options(case, tokens);
            options.seed = 0x6c65_6f6e_6500_0000_u64 + index as u64;
            options
        })
        .collect()
}

fn lifecycle_options(case: MatrixCase, max_tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::greedy(max_tokens);
    options.decode_execution = DecodeExecution::Graph;
    options.prefill_chunk_tokens = PREFILL_CHUNK_TOKENS;
    options.kv_cache_dtype = case.dtype;
    if case.stochastic {
        options.sampler = Sampler::temperature(0.8);
        options.seed = 0x5345_5353_494f_4e31;
    }
    options
}

fn isolated_transcripts(
    runtime: &mut Runtime<CudaBackend>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
) -> TestResult<Vec<Vec<u32>>> {
    prompts
        .iter()
        .zip(options)
        .map(|(prompt, options)| reference_generation(runtime, prompt, options.clone()))
        .collect()
}

fn encode_prompts(runtime: &Runtime<CudaBackend>, prompts: &[&str]) -> TestResult<Vec<Vec<u32>>> {
    prompts
        .iter()
        .map(|prompt| Ok(runtime.model().tokenizer().encode(prompt)?))
        .collect()
}

fn append_branch_token(prompt: &[u32], vocab_size: usize) -> Vec<u32> {
    let mut branch = prompt.to_vec();
    let next = (usize::try_from(*prompt.last().expect("branch prompt is nonempty")).unwrap() + 1)
        % vocab_size;
    branch.push(u32::try_from(next).expect("model vocabulary fits u32"));
    branch
}

fn churn_prompts() -> [&'static str; 4] {
    [
        "State one invariant for a bounded decode scheduler.",
        "Explain why a sampler needs a fixed seed in one sentence.",
        "Describe one reason to split prompt prefill into chunks.",
        "Give one rule for releasing cancelled session state.",
    ]
}

fn lifecycle_prompt() -> &'static str {
    "Describe how an exact session keeps its token transcript across a fork."
}

fn long_prompt() -> &'static str {
    "A bounded prefill processes this prompt in several fixed chunks before decode starts. \
     The cancellation point must occur after one chunk and before the next chunk."
}

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("crate is under workspace/crates")
        .join("models/Qwen3-8B-Q4_K_M.gguf")
}

fn llama_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("crate is under workspace/crates")
        .join("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
}

#[derive(Debug, Clone, Copy)]
enum PrefillOrigin {
    Append,
    Repeat,
    Restore,
    Fork,
    Wake,
    MirostatBranch,
    MirostatChange,
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Llama Q4_K_M model"]
fn llama_resumable_prefill_state_transitions_match_direct() -> TestResult {
    prefill_transition_matrix(&llama_model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn qwen_resumable_prefill_state_transitions_match_direct() -> TestResult {
    prefill_transition_matrix(&model_path())
}

fn prefill_transition_matrix(path: &Path) -> TestResult {
    let origins = [
        PrefillOrigin::Append,
        PrefillOrigin::Repeat,
        PrefillOrigin::Restore,
        PrefillOrigin::Fork,
        PrefillOrigin::Wake,
        PrefillOrigin::MirostatBranch,
        PrefillOrigin::MirostatChange,
    ];
    for dtype in [KvCacheDtype::F16, KvCacheDtype::Q8] {
        let mut reference = Runtime::load(CudaBackend::new(0)?, path)?;
        let expected = origins
            .iter()
            .map(|origin| transition_case(&mut reference, dtype, *origin, false))
            .collect::<TestResult<Vec<_>>>()?;
        drop(reference);
        let mut runtime = Runtime::load(CudaBackend::new(0)?, path)?;
        for (origin, expected) in origins.iter().zip(expected) {
            let observed = transition_case(&mut runtime, dtype, *origin, true)?;
            assert_eq!(observed, expected, "{origin:?} {dtype:?}");
        }
    }
    Ok(())
}

fn transition_options(dtype: KvCacheDtype, origin: PrefillOrigin) -> TestResult<GenerateOptions> {
    let mut options = lifecycle_options(
        MatrixCase {
            dtype,
            stochastic: true,
        },
        3,
    );
    if matches!(
        origin,
        PrefillOrigin::MirostatBranch | PrefillOrigin::MirostatChange
    ) {
        options.mirostat = Some(leone::MirostatConfig::new(4.0, 0.2)?);
    }
    Ok(options)
}

fn transition_case(
    runtime: &mut Runtime<CudaBackend>,
    dtype: KvCacheDtype,
    origin: PrefillOrigin,
    resumable: bool,
) -> TestResult<(Vec<u32>, leone::GenerationCheckpoint)> {
    let mut options = transition_options(dtype, origin)?;
    let (mut session, mut prompt) = transition_prefix(runtime, &options)?;
    prepare_transition(
        runtime,
        &mut session,
        &mut prompt,
        &mut options,
        origin,
        resumable,
    )?;
    if resumable {
        apply_resumable_transition(runtime, &mut session, &prompt, &options, origin)?;
    }
    let generated =
        runtime.generate_session_tokens(&mut session, &prompt, options, |_| Ok(()), || false)?;
    if resumable {
        assert_eq!(
            session.last_replay().reuse_class,
            leone::SessionReuseClass::ExactRepeat
        );
        assert_eq!(session.last_replay().reused_tokens, prompt.len());
        check_transition_records(&session, origin);
    }
    Ok((generated.tokens, session.checkpoint()))
}

fn transition_prefix(
    runtime: &mut Runtime<CudaBackend>,
    options: &GenerateOptions,
) -> TestResult<(GenerationSession<CudaBackend>, Vec<u32>)> {
    let mut session = GenerationSession::new();
    let mut prompt = runtime.model().tokenizer().encode(lifecycle_prompt())?;
    let prefix = runtime.generate_session_tokens(
        &mut session,
        &prompt,
        options.clone(),
        |_| Ok(()),
        || false,
    )?;
    prompt.extend_from_slice(&prefix.tokens);
    Ok((session, prompt))
}

fn apply_resumable_transition(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    prompt: &[u32],
    options: &GenerateOptions,
    origin: PrefillOrigin,
) -> TestResult {
    check_prefill_validation_preserves_session(runtime, session, prompt, options)?;
    let credited = runtime.reusable_prefill_tokens(session, prompt, options)?;
    let pending = runtime.begin_prefill(session, prompt, options.clone())?;
    let ready = complete_pending(runtime, pending)?;
    assert_eq!(ready.processed_tokens(), prompt.len() - credited);
    runtime.finish_prefill(ready, session)?;
    check_transition_records(session, origin);
    Ok(())
}

fn prepare_transition(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    prompt: &mut Vec<u32>,
    options: &mut GenerateOptions,
    origin: PrefillOrigin,
    resumable: bool,
) -> TestResult {
    prepare_transition_prompt(session, prompt, options, origin, runtime.vocab_size())?;
    if resumable {
        prepare_transition_device(runtime, session, origin)?;
    }
    Ok(())
}

fn prepare_transition_prompt(
    session: &mut GenerationSession<CudaBackend>,
    prompt: &mut Vec<u32>,
    options: &mut GenerateOptions,
    origin: PrefillOrigin,
    vocab: usize,
) -> TestResult {
    match origin {
        PrefillOrigin::Append => *prompt = append_branch_token(prompt, vocab),
        PrefillOrigin::Repeat => *prompt = session.evaluated_tokens().to_vec(),
        PrefillOrigin::Restore => session.restore(session.checkpoint()),
        PrefillOrigin::MirostatBranch => prompt[1] = (prompt[1] + 1) % vocab as u32,
        PrefillOrigin::MirostatChange => {
            options.mirostat = Some(leone::MirostatConfig::new(2.0, 0.1)?)
        }
        _ => {}
    }
    Ok(())
}

fn prepare_transition_device(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    origin: PrefillOrigin,
) -> TestResult {
    match origin {
        PrefillOrigin::Fork => *session = runtime.fork_session(session)?,
        PrefillOrigin::Wake => {
            let host = runtime.hibernate_session(session)?;
            *session = runtime.wake_session(host)?;
        }
        _ => {}
    }
    Ok(())
}

fn check_transition_records(session: &GenerationSession<CudaBackend>, origin: PrefillOrigin) {
    match origin {
        PrefillOrigin::Fork => assert!(session.last_fork().is_some()),
        PrefillOrigin::Wake => assert!(session.last_hibernation().is_some()),
        _ => {}
    }
}

fn check_prefill_validation_preserves_session(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    prompt: &[u32],
    options: &GenerateOptions,
) -> TestResult {
    let before = session.checkpoint();
    let memory = runtime.backend().memory_accounting();
    let mut invalid = options.clone();
    invalid.max_tokens = 0;
    assert!(matches!(
        runtime.begin_prefill(session, prompt, invalid),
        Err(leone::RuntimeError::ZeroGeneration)
    ));
    assert_eq!(session.checkpoint(), before);
    assert_eq!(
        runtime.backend().memory_accounting().live_bytes,
        memory.live_bytes
    );
    Ok(())
}

fn complete_pending(
    runtime: &mut Runtime<CudaBackend>,
    mut pending: PendingPrefill<CudaBackend>,
) -> TestResult<ReadyPrefill<CudaBackend>> {
    loop {
        let budget = pending.minimum_budget();
        match runtime.advance_prefill(pending, budget, || false)? {
            PrefillProgress::Pending(next) => pending = next,
            PrefillProgress::Ready(ready) => return Ok(ready),
            PrefillProgress::Cancelled(_) => {
                return Err(std::io::Error::other("unexpected cancellation").into())
            }
        }
    }
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Llama Q4_K_M model"]
fn llama_pending_prefills_interleave_with_graph_decode() -> TestResult {
    interleaved_prefill_matrix(&llama_model_path())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU and the Qwen3 Q4_K_M model"]
fn qwen_pending_prefills_interleave_with_graph_decode() -> TestResult {
    interleaved_prefill_matrix(&model_path())
}

fn interleaved_prefill_matrix(path: &Path) -> TestResult {
    for dtype in [KvCacheDtype::F16, KvCacheDtype::Q8] {
        let mut reference = Runtime::load(CudaBackend::new(0)?, path)?;
        let second = format!("{} {}", lifecycle_prompt(), long_prompt());
        let prompts = encode_prompts(&reference, &[long_prompt(), &second, churn_prompts()[0]])?;
        let mut options = [
            prefill_options(dtype),
            prefill_options(dtype),
            prefill_options(dtype),
        ];
        options[1].prefill_chunk_tokens = 7;
        let expected = isolated_transcripts(&mut reference, &prompts, &options)?;
        drop(reference);
        let mut runtime = Runtime::load(CudaBackend::new(0)?, path)?;
        let actual = interleaved_prefill_case(&mut runtime, &prompts, &options)?;
        assert_eq!(actual, expected, "interleaved prefill {dtype:?}");
    }
    Ok(())
}

fn interleaved_prefill_case(
    runtime: &mut Runtime<CudaBackend>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions; 3],
) -> TestResult<Vec<Vec<u32>>> {
    let mut resident = GenerationSession::new();
    let mut initial = options[2].clone();
    initial.max_tokens = 3;
    let prefix = runtime.generate_session_tokens(
        &mut resident,
        &prompts[2],
        initial,
        |_| Ok(()),
        || false,
    )?;
    let mut transcript = prompts[2].clone();
    transcript.extend_from_slice(&prefix.tokens);
    let mut resident_tokens = prefix.tokens;
    let mut left = GenerationSession::new();
    let mut right = GenerationSession::new();
    let mut pending_left = runtime.begin_prefill(&mut left, &prompts[0], options[0].clone())?;
    let mut pending_right = runtime.begin_prefill(&mut right, &prompts[1], options[1].clone())?;
    (pending_left, pending_right) = interleave_two_steps(
        runtime,
        pending_left,
        pending_right,
        &mut resident,
        &mut transcript,
        &mut resident_tokens,
        &options[2],
    )?;
    let left_tokens = finish_pending_tokens(runtime, &mut left, pending_left, &options[0])?;
    let right_tokens = finish_pending_tokens(runtime, &mut right, pending_right, &options[1])?;
    Ok(vec![left_tokens, right_tokens, resident_tokens])
}

fn interleave_two_steps(
    runtime: &mut Runtime<CudaBackend>,
    mut pending_left: PendingPrefill<CudaBackend>,
    mut pending_right: PendingPrefill<CudaBackend>,
    resident: &mut GenerationSession<CudaBackend>,
    transcript: &mut Vec<u32>,
    resident_tokens: &mut Vec<u32>,
    options: &GenerateOptions,
) -> TestResult<(PendingPrefill<CudaBackend>, PendingPrefill<CudaBackend>)> {
    for _ in 0..2 {
        pending_left = advance_pending_once(runtime, pending_left)?;
        pending_right = advance_pending_once(runtime, pending_right)?;
        let tokens = runtime.generate_session_batch_token(&mut [BatchSession {
            session: resident,
            transcript,
            options,
        }])?;
        transcript.push(tokens[0].id);
        resident_tokens.push(tokens[0].id);
    }
    Ok((pending_left, pending_right))
}

fn advance_pending_once(
    runtime: &mut Runtime<CudaBackend>,
    pending: PendingPrefill<CudaBackend>,
) -> TestResult<PendingPrefill<CudaBackend>> {
    let budget = pending.minimum_budget();
    pending_from_progress(runtime.advance_prefill(pending, budget, || false)?)
}

fn finish_pending_tokens(
    runtime: &mut Runtime<CudaBackend>,
    session: &mut GenerationSession<CudaBackend>,
    pending: PendingPrefill<CudaBackend>,
    options: &GenerateOptions,
) -> TestResult<Vec<u32>> {
    let ready = complete_pending(runtime, pending)?;
    let prompt = ready.prompt_tokens().to_vec();
    runtime.finish_prefill(ready, session)?;
    let result =
        runtime.generate_session_tokens(session, &prompt, options.clone(), |_| Ok(()), || false)?;
    assert_eq!(session.last_replay().reused_tokens, prompt.len());
    Ok(result.tokens)
}
