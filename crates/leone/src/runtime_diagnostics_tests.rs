use super::batch_attention_tests::queued_runtime;
use super::tests::{cycle_options, cycle_runtime};
use super::*;
use crate::cpu::CompletionProbe;
use crate::{CpuBackend, CpuBuffer};

fn seed_session(
    runtime: &mut Runtime<CpuBackend>,
    prompt: &[u32],
) -> GenerationSession<CpuBackend> {
    let mut session = GenerationSession::new();
    runtime
        .generate_session_tokens(
            &mut session,
            prompt,
            cycle_options(1, false),
            |_| Ok(()),
            || false,
        )
        .unwrap();
    session
}

pub(super) fn assert_cycle_row(
    runtime: &mut Runtime<CpuBackend>,
    session: &GenerationSession<CpuBackend>,
) -> Vec<f32> {
    let mut output = vec![f32::NAN; runtime.vocab_size()];
    let position = runtime.read_session_logits(session, &mut output).unwrap();
    assert_eq!(position, session.evaluated_tokens().len());
    let next = (session.evaluated_tokens().last().unwrap() + 1) % 4;
    let nonzero = (1.0 / (1.0 / 256.0 + 1e-6_f64).sqrt()) as f32;
    for (index, value) in output.iter().copied().enumerate() {
        let expected = if index == next as usize { nonzero } else { 0.0 };
        assert!(value.is_finite());
        assert!((value - expected).abs() < 1e-5, "logit {index}: {value}");
    }
    output
}

fn assert_unavailable(runtime: &mut Runtime<CpuBackend>, session: &GenerationSession<CpuBackend>) {
    let mut output = vec![37.0; runtime.vocab_size()];
    assert!(matches!(
        runtime.read_session_logits(session, &mut output),
        Err(RuntimeError::SessionLogitsUnavailable)
    ));
    assert!(output.iter().all(|value| *value == 37.0));
}

fn completion_calls(runtime: &Runtime<CpuBackend>) -> usize {
    runtime.backend.completion_probe.as_ref().unwrap().calls
}

fn assert_f32_bits(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "logit {index} differs at the bit level"
        );
    }
}

fn read_logits(
    runtime: &mut Runtime<CpuBackend>,
    activations: &Activations<CpuBackend>,
) -> Vec<f32> {
    let mut logits = vec![0.0; runtime.vocab_size()];
    runtime
        .backend
        .read_f32(&activations.logits, &mut logits)
        .unwrap();
    logits
}

fn read_kv_snapshots(
    runtime: &mut Runtime<CpuBackend>,
    state: &KvState<CpuBackend>,
) -> Vec<BufferSnapshot> {
    let snapshot = state.snapshot(&mut runtime.backend).unwrap();
    let mut buffers = Vec::new();
    for segment in snapshot.segments {
        for (key, value) in segment.layers {
            buffers.extend([key, value]);
        }
    }
    buffers
}

fn assert_kv_bits(expected: &[BufferSnapshot], actual: &[BufferSnapshot]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.layout(),
            actual.layout(),
            "KV buffer {index} layout"
        );
        assert_eq!(
            expected.bytes(),
            actual.bytes(),
            "KV buffer {index} differs"
        );
    }
}

fn sequential_control(
    runtime: &mut Runtime<CpuBackend>,
    prompt: &[u32],
    dtype: KvCacheDtype,
) -> (Vec<BufferSnapshot>, Vec<f32>) {
    let context = decode_graph_bucket(prompt.len()).unwrap();
    let attention_shape = AttentionShape::new(
        runtime.model.config.n_head,
        runtime.model.config.n_head_kv,
        runtime.model.config.head_dim,
        context,
    )
    .unwrap();
    let mut state = KvState::new(
        &mut runtime.backend,
        runtime.model.config.n_layer,
        attention_shape,
        dtype,
    )
    .unwrap();
    let mut activations = Activations::new(&mut runtime.backend, &runtime.model.config).unwrap();
    for token in prompt.iter().copied() {
        runtime
            .backend
            .write_u32(&mut activations.sampled, &[token])
            .unwrap();
        runtime
            .forward(&mut state, &mut activations, attention_shape)
            .unwrap();
    }
    runtime.backend.synchronize().unwrap();
    let kv = read_kv_snapshots(runtime, &state);
    let logits = read_logits(runtime, &activations);
    (kv, logits)
}

fn sequential_control_rows(
    runtime: &mut Runtime<CpuBackend>,
    prompt: &[u32],
    dtype: KvCacheDtype,
) -> Vec<Vec<f32>> {
    (1..=prompt.len())
        .map(|end| sequential_control(runtime, &prompt[..end], dtype).1)
        .collect()
}

fn run_sequential_prefill_case(prompt: &[u32], dtype: KvCacheDtype) {
    let mut control = queued_runtime();
    let expected_rows = sequential_control_rows(&mut control, prompt, dtype);
    let (expected_kv, expected_prompt_logits) = sequential_control(&mut control, prompt, dtype);

    let mut runtime = queued_runtime();
    let options = GenerateOptions {
        kv_cache_dtype: dtype,
        ..cycle_options(1, false)
    };
    let mut session = GenerationSession::new();
    let mut pending = runtime
        .begin_prefill(&mut session, prompt, options)
        .unwrap();
    let ready = loop {
        match runtime
            .advance_prefill(pending, NonZeroUsize::new(3).unwrap(), || false)
            .unwrap()
        {
            PrefillProgress::Pending(next) => {
                let processed = next.processed_tokens();
                let actual = read_logits(&mut runtime, &next.activations);
                assert_f32_bits(&expected_rows[processed - 1], &actual);
                pending = next;
            }
            PrefillProgress::Ready(ready) => {
                let actual = read_logits(&mut runtime, &ready.activations);
                assert_f32_bits(&expected_prompt_logits, &actual);
                break ready;
            }
            PrefillProgress::Cancelled(_) => panic!("prefill was not cancelled"),
        }
    };
    runtime.finish_prefill(ready, &mut session).unwrap();

    let actual_kv = read_kv_snapshots(&mut runtime, session.state.as_ref().unwrap());
    assert_kv_bits(&expected_kv, &actual_kv);
    let mut prompt_logits = vec![0.0; runtime.vocab_size()];
    runtime
        .read_session_logits(&session, &mut prompt_logits)
        .unwrap();
    assert_f32_bits(&expected_prompt_logits, &prompt_logits);

    let result = runtime
        .generate_session_tokens(
            &mut session,
            prompt,
            GenerateOptions {
                kv_cache_dtype: dtype,
                ..cycle_options(2, false)
            },
            |_| Ok(()),
            || false,
        )
        .unwrap();
    let decode_prompt = [prompt, &result.tokens[..1]].concat();
    let (expected_decode_kv, expected_decode_logits) =
        sequential_control(&mut control, &decode_prompt, dtype);
    let actual_decode_kv = read_kv_snapshots(&mut runtime, session.state.as_ref().unwrap());
    assert_kv_bits(&expected_decode_kv, &actual_decode_kv);
    let mut decode_logits = vec![0.0; runtime.vocab_size()];
    runtime
        .read_session_logits(&session, &mut decode_logits)
        .unwrap();
    assert_f32_bits(&expected_decode_logits, &decode_logits);
}

#[test]
fn diagnostic_read_returns_raw_batch_rows_before_penalties() {
    let mut runtime = cycle_runtime();
    runtime.backend.enable_reference_verify();
    let mut first = seed_session(&mut runtime, &[0, 1, 2, 3]);
    let mut second = seed_session(&mut runtime, &[0, 1, 3, 2]);
    let options = GenerateOptions {
        penalties: Penalties {
            repetition: 2.0,
            presence: 2.0,
            frequency: 1.0,
            ..Penalties::none()
        },
        ..cycle_options(1, false)
    };
    for (first_tokens, second_tokens) in [
        (&[0, 1, 2, 3, 0][..], &[0, 1, 3, 2, 3][..]),
        (&[0, 1, 2, 3, 0, 1][..], &[0, 1, 3, 2, 3, 0][..]),
    ] {
        runtime
            .generate_session_batch_token(&mut [
                BatchSession {
                    session: &mut first,
                    transcript: first_tokens,
                    options: &options,
                },
                BatchSession {
                    session: &mut second,
                    transcript: second_tokens,
                    options: &options,
                },
            ])
            .unwrap();
        let first_row = assert_cycle_row(&mut runtime, &first);
        let second_row = assert_cycle_row(&mut runtime, &second);
        assert_ne!(first_row, second_row);
        let mut penalized = first_row.clone();
        options
            .penalties
            .apply(&mut penalized, first_tokens)
            .unwrap();
        assert_ne!(first_row, penalized);
    }
}

#[test]
fn diagnostic_read_waits_and_allocates_no_backend_storage() {
    let mut runtime = cycle_runtime();
    let session = seed_session(&mut runtime, &[0, 1, 2]);
    runtime.backend.completion_probe = Some(CompletionProbe::default());
    let before = runtime.backend.memory_accounting();
    assert_cycle_row(&mut runtime, &session);
    assert_eq!(completion_calls(&runtime), 1);
    assert_eq!(runtime.backend.memory_accounting(), before);
    runtime.synchronize().unwrap();
    assert_eq!(completion_calls(&runtime), 2);
}

#[test]
fn diagnostic_completion_failure_preserves_output_and_reports_quarantine() {
    let mut runtime = cycle_runtime();
    let session = seed_session(&mut runtime, &[0, 1, 2]);
    runtime.backend.completion_probe = Some(CompletionProbe {
        calls: 0,
        fail: true,
    });
    let mut output = vec![37.0; runtime.vocab_size()];
    let error = runtime
        .read_session_logits(&session, &mut output)
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert!(output.iter().all(|value| *value == 37.0));
    assert_eq!(completion_calls(&runtime), 1);
    let error = runtime.synchronize().unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert_eq!(completion_calls(&runtime), 2);
}

#[test]
fn diagnostic_read_rejects_unavailable_and_stale_rows() {
    let mut runtime = cycle_runtime();
    let mut session = GenerationSession::new();
    assert_unavailable(&mut runtime, &session);
    session.restore_tokens(vec![0, 1, 2], None);
    assert_unavailable(&mut runtime, &session);
    session = seed_session(&mut runtime, &[0, 1, 2]);
    session.retained_logits_position = Some(2);
    assert_unavailable(&mut runtime, &session);
    session.retained_logits_position = Some(3);
    session.state.as_mut().unwrap().position = 2;
    assert_unavailable(&mut runtime, &session);
    session.invalidate();
    assert_unavailable(&mut runtime, &session);
}

#[test]
fn diagnostic_read_rejects_wrong_output_sizes_before_completion() {
    let mut runtime = cycle_runtime();
    let session = seed_session(&mut runtime, &[0, 1, 2]);
    runtime.backend.completion_probe = Some(CompletionProbe::default());
    for length in [0, 255, 257] {
        let mut output = vec![37.0; length];
        assert!(matches!(
            runtime.read_session_logits(&session, &mut output),
            Err(RuntimeError::Backend(BackendError::SizeMismatch {
                name: "session logit output", expected: 256, actual,
            })) if actual == length
        ));
        assert!(output.iter().all(|value| *value == 37.0));
    }
    assert_eq!(completion_calls(&runtime), 0);
    assert_cycle_row(&mut runtime, &session);
}

#[test]
fn diagnostic_read_checks_vocabulary_byte_overflow() {
    let mut runtime = cycle_runtime();
    let session = seed_session(&mut runtime, &[0, 1, 2]);
    runtime.model.config.vocab_size = usize::MAX;
    runtime.backend.completion_probe = Some(CompletionProbe::default());
    let mut output = [37.0; 256];
    assert!(matches!(
        runtime.read_session_logits(&session, &mut output),
        Err(RuntimeError::Backend(BackendError::SizeOverflow { .. }))
    ));
    assert_eq!(output, [37.0; 256]);
    assert_eq!(completion_calls(&runtime), 0);
}

fn replace_logits(
    runtime: &mut Runtime<CpuBackend>,
    session: &mut GenerationSession<CpuBackend>,
    layout: BufferLayout,
) {
    let logits: CpuBuffer = runtime.backend.allocate(layout).unwrap();
    session.activations.as_mut().unwrap().logits = logits;
}

#[test]
fn diagnostic_read_checks_retained_storage_type_and_size() {
    let mut runtime = cycle_runtime();
    let mut session = seed_session(&mut runtime, &[0, 1, 2]);
    for layout in [
        BufferLayout::f32(255).unwrap(),
        BufferLayout::f32(257).unwrap(),
        BufferLayout::f16(512).unwrap(),
    ] {
        replace_logits(&mut runtime, &mut session, layout);
        let mut output = [37.0; 256];
        assert!(runtime.read_session_logits(&session, &mut output).is_err());
        assert_eq!(output, [37.0; 256]);
    }
}

#[test]
fn diagnostic_read_waits_for_resumable_prefill_commit() {
    let mut runtime = cycle_runtime();
    let mut session = seed_session(&mut runtime, &[0, 1, 2, 3]);
    let prompt = [0, 1, 2, 3, 0, 1, 2];
    let mut pending = runtime
        .begin_prefill(&mut session, &prompt, cycle_options(1, false))
        .unwrap();
    assert_unavailable(&mut runtime, &session);
    let ready = loop {
        match runtime
            .advance_prefill(pending, NonZeroUsize::new(1).unwrap(), || false)
            .unwrap()
        {
            PrefillProgress::Pending(next) => {
                pending = next;
                assert_unavailable(&mut runtime, &session);
            }
            PrefillProgress::Ready(ready) => break ready,
            PrefillProgress::Cancelled(_) => panic!("prefill was not cancelled"),
        }
    };
    assert_unavailable(&mut runtime, &session);
    runtime.finish_prefill(ready, &mut session).unwrap();
    assert_eq!(session.evaluated_tokens(), prompt);
    assert_cycle_row(&mut runtime, &session);
}

#[test]
fn sequential_prefill_preserves_logits_and_kv_continuity() {
    let prompt = [0, 1, 2, 3, 0, 1, 2];
    for dtype in [KvCacheDtype::F16, KvCacheDtype::Q8, KvCacheDtype::F32] {
        run_sequential_prefill_case(&prompt, dtype);
    }
}

#[test]
fn sequential_prefill_cancellation_discards_partial_state() {
    let prompt = [0, 1, 2, 3, 0, 1, 2];
    let mut cancelled_runtime = queued_runtime();
    let mut cancelled_session = GenerationSession::new();
    let pending = cancelled_runtime
        .begin_prefill(&mut cancelled_session, &prompt, cycle_options(1, false))
        .unwrap();
    let calls = std::cell::Cell::new(0_usize);
    let progress = cancelled_runtime
        .advance_prefill(pending, NonZeroUsize::new(2).unwrap(), || {
            let previous = calls.get();
            calls.set(previous + 1);
            previous > 0
        })
        .unwrap();
    assert!(matches!(progress, PrefillProgress::Cancelled(_)));
    assert!(cancelled_session.is_empty());
}

#[test]
fn diagnostic_read_survives_fork_hibernation_and_exact_repeat() {
    let mut runtime = cycle_runtime();
    let mut source = seed_session(&mut runtime, &[0, 1, 2]);
    let expected = assert_cycle_row(&mut runtime, &source);
    let child = runtime.fork_session(&source).unwrap();
    assert_eq!(assert_cycle_row(&mut runtime, &child), expected);
    let snapshot = runtime.hibernate_session(&mut source).unwrap();
    assert_unavailable(&mut runtime, &source);
    let mut restored = runtime.wake_session(&snapshot).unwrap();
    assert_eq!(assert_cycle_row(&mut runtime, &restored), expected);
    runtime
        .generate_session_tokens(
            &mut restored,
            &[0, 1, 2],
            cycle_options(1, false),
            |_| Ok(()),
            || false,
        )
        .unwrap();
    assert_eq!(assert_cycle_row(&mut runtime, &restored), expected);
}

#[test]
fn diagnostic_read_preserves_validity_after_unchanged_rejection() {
    let mut runtime = cycle_runtime();
    runtime.backend.enable_reference_verify();
    let mut session = seed_session(&mut runtime, &[0, 1, 2]);
    let before = assert_cycle_row(&mut runtime, &session);
    let options = cycle_options(1, false);
    let error = runtime
        .generate_session_batch_token(&mut [BatchSession {
            session: &mut session,
            transcript: &[0, 1, 3, 0],
            options: &options,
        }])
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Unchanged)
    );
    assert_eq!(assert_cycle_row(&mut runtime, &session), before);
}

#[test]
fn diagnostic_read_preserves_validity_after_warm_allocation_rejection() {
    let mut runtime = cycle_runtime();
    let mut session = seed_session(&mut runtime, &[0, 1, 2, 3]);
    let before = assert_cycle_row(&mut runtime, &session);
    runtime.backend.fail_allocations_after(0);
    let error = runtime
        .generate_session_tokens(
            &mut session,
            &[0, 1, 2, 3, 0],
            cycle_options(8, true),
            |_| Ok(()),
            || false,
        )
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Unchanged)
    );
    assert_eq!(assert_cycle_row(&mut runtime, &session), before);
}

#[test]
fn diagnostic_read_rejects_every_row_after_partial_batch_failure() {
    let mut runtime = cycle_runtime();
    runtime.backend.enable_reference_verify();
    let mut first = seed_session(&mut runtime, &[0, 1, 2]);
    let mut second = runtime.fork_session(&first).unwrap();
    let options = cycle_options(1, false);
    let mut invalid = options.clone();
    invalid.penalties.presence = f64::NAN;
    let error = runtime
        .generate_session_batch_token(&mut [
            BatchSession {
                session: &mut first,
                transcript: &[0, 1, 2, 3],
                options: &options,
            },
            BatchSession {
                session: &mut second,
                transcript: &[0, 1, 2, 3],
                options: &invalid,
            },
        ])
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert_eq!(first.evaluated_tokens(), &[0, 1, 2, 3]);
    assert_eq!(second.evaluated_tokens(), &[0, 1, 2]);
    assert_unavailable(&mut runtime, &first);
    assert_unavailable(&mut runtime, &second);
    let snapshot = runtime.hibernate_session(&mut first).unwrap();
    let restored = runtime.wake_session(&snapshot).unwrap();
    assert_unavailable(&mut runtime, &restored);
}

#[test]
fn diagnostic_read_rejects_cancelled_generation() {
    let mut runtime = cycle_runtime();
    let mut session = seed_session(&mut runtime, &[0, 1, 2]);
    let emitted = std::cell::Cell::new(false);
    let result = runtime
        .generate_session_tokens(
            &mut session,
            &[0, 1, 2, 3],
            cycle_options(3, false),
            |_| {
                emitted.set(true);
                Ok(())
            },
            || emitted.get(),
        )
        .unwrap();
    assert_eq!(result.termination, GenerationTermination::Cancelled);
    assert_unavailable(&mut runtime, &session);
}

#[test]
fn diagnostic_read_rejects_failed_generation() {
    let mut runtime = cycle_runtime();
    let mut session = seed_session(&mut runtime, &[0, 1, 2]);
    let error = runtime
        .generate_session_tokens(
            &mut session,
            &[0, 1, 2, 3],
            cycle_options(3, false),
            |_| Err(RuntimeError::token_callback("injected failure")),
            || false,
        )
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert_unavailable(&mut runtime, &session);
}
