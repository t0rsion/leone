use super::*;
use crate::cpu::{AttentionEvent, AttentionProbe, PreflightProbe, VerificationProbe};
use crate::model::{OutputWeight, QuantWeight, StagedVec};
use crate::runtime_service::{
    DriverPrefillProgress, GenerationQuantum, LeoneRuntimeDriver, LeoneRuntimeDriverError,
    PausableGenerationDriver, RuntimeQuantumExecutor, ScheduledGenerationRequest,
};
use crate::scheduler::{DispatchKind, RequestId, RequestSpec, RequestStatus, SchedulerPolicy};
use crate::service::ScheduledService;
use crate::{CpuBackend, CpuBuffer, ModelConfig, QuantFormat, QuantMatrix};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};

fn upload_values(backend: &mut CpuBackend, values: &[f32]) -> CpuBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    backend
        .upload(BufferLayout::f32(values.len()).unwrap(), &bytes)
        .unwrap()
}

fn test_weight(backend: &mut CpuBackend, rows: usize, columns: usize) -> QuantWeight<CpuBackend> {
    let shape = QuantMatrix::new(rows, columns, QuantFormat::Q4K).unwrap();
    QuantWeight {
        buffer: backend.allocate(shape.layout().unwrap()).unwrap(),
        shape,
    }
}

fn test_layer(
    backend: &mut CpuBackend,
    config: &ModelConfig,
    normalized: bool,
) -> DenseLayer<CpuBackend> {
    let qk_norm = if normalized {
        QkNorm::Rms {
            query: upload_values(backend, &vec![1.5; config.head_dim]),
            key: upload_values(backend, &vec![0.75; config.head_dim]),
        }
    } else {
        QkNorm::Identity
    };
    let columns = config.n_head_kv * config.head_dim;
    DenseLayer {
        attention_norm: upload_values(backend, &vec![1.0; config.n_embd]),
        query: test_weight(backend, config.n_embd, config.n_embd),
        key: test_weight(backend, columns, config.n_embd),
        value: test_weight(backend, columns, config.n_embd),
        qk_norm,
        attention_output: test_weight(backend, config.n_embd, config.n_embd),
        ffn_norm: upload_values(backend, &vec![1.0; config.n_embd]),
        ffn_gate: test_weight(backend, config.n_ff, config.n_embd),
        ffn_up: test_weight(backend, config.n_ff, config.n_embd),
        ffn_down: test_weight(backend, config.n_embd, config.n_ff),
    }
}

pub(super) fn attention_runtime(layers: usize, normalized: bool) -> Runtime<CpuBackend> {
    let mut runtime = super::tests::cpu_toy_runtime();
    runtime.backend.set_test_batch_size(2);
    runtime.model.config.n_layer = layers;
    runtime.model.config.n_head_kv = 2;
    let config = &runtime.model.config;
    let weights = (0..layers)
        .map(|_| test_layer(&mut runtime.backend, config, normalized))
        .collect();
    runtime.model.weights.layers = StagedVec::from_test_values(weights);
    runtime
        .backend
        .configure_rope(
            config.head_dim,
            config.rope_theta,
            None,
            crate::RopePairing::HalfSplit,
        )
        .unwrap();
    runtime.backend.enable_reference_verify();
    runtime
}

fn nonzero_weight(
    backend: &mut CpuBackend,
    rows: usize,
    columns: usize,
    seed: usize,
) -> QuantWeight<CpuBackend> {
    let shape = QuantMatrix::new(rows, columns, QuantFormat::Q4K).unwrap();
    let layout = shape.layout().unwrap();
    let mut bytes = vec![0_u8; layout.bytes()];
    for (block_index, block) in bytes.chunks_exact_mut(144).enumerate() {
        block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[4] = 1;
        block[16 + (block_index + seed) % 128] = 1;
    }
    QuantWeight {
        buffer: backend.upload(layout, &bytes).unwrap(),
        shape,
    }
}

pub(super) fn queued_runtime() -> Runtime<CpuBackend> {
    let mut runtime = super::tests::cpu_toy_runtime();
    runtime.model.config.n_layer = 1;
    runtime.model.config.n_head_kv = 2;
    let config = runtime.model.config.clone();
    let embedding_shape = runtime.model.weights.token_embedding.shape;
    runtime.model.weights.token_embedding = nonzero_weight(
        &mut runtime.backend,
        embedding_shape.rows(),
        embedding_shape.columns(),
        0,
    );
    runtime.model.weights.output = OutputWeight::Separate(nonzero_weight(
        &mut runtime.backend,
        config.vocab_size,
        config.n_embd,
        3,
    ));
    runtime.model.weights.output_norm =
        upload_values(&mut runtime.backend, &vec![1.0; config.n_embd]);
    let columns = config.n_head_kv * config.head_dim;
    let layer = DenseLayer {
        attention_norm: upload_values(&mut runtime.backend, &vec![1.0; config.n_embd]),
        query: nonzero_weight(&mut runtime.backend, config.n_embd, config.n_embd, 5),
        key: nonzero_weight(&mut runtime.backend, columns, config.n_embd, 7),
        value: nonzero_weight(&mut runtime.backend, columns, config.n_embd, 11),
        qk_norm: QkNorm::Identity,
        attention_output: nonzero_weight(&mut runtime.backend, config.n_embd, config.n_embd, 13),
        ffn_norm: upload_values(&mut runtime.backend, &vec![1.0; config.n_embd]),
        ffn_gate: nonzero_weight(&mut runtime.backend, config.n_ff, config.n_embd, 17),
        ffn_up: nonzero_weight(&mut runtime.backend, config.n_ff, config.n_embd, 19),
        ffn_down: nonzero_weight(&mut runtime.backend, config.n_embd, config.n_ff, 23),
    };
    runtime.model.weights.layers = StagedVec::from_test_values(vec![layer]);
    runtime
        .backend
        .configure_rope(
            config.head_dim,
            config.rope_theta,
            None,
            crate::RopePairing::HalfSplit,
        )
        .unwrap();
    runtime.backend.enable_reference_verify();
    runtime
}

fn prefix_session(runtime: &mut Runtime<CpuBackend>) -> GenerationSession<CpuBackend> {
    prefix_session_with_context(runtime, 128)
}

fn prefix_session_with_context(
    runtime: &mut Runtime<CpuBackend>,
    context_tokens: usize,
) -> GenerationSession<CpuBackend> {
    let config = &runtime.model.config;
    let shape = AttentionShape::new(
        config.n_head,
        config.n_head_kv,
        config.head_dim,
        context_tokens,
    )
    .unwrap();
    let mut state = KvState::new(
        &mut runtime.backend,
        config.n_layer,
        shape,
        KvCacheDtype::F16,
    )
    .unwrap();
    let range = state
        .prepare_append(&mut runtime.backend, config.n_layer, 3, 3)
        .unwrap();
    let elements = 3 * shape.projected_kv_elements().unwrap();
    let values: Vec<f32> = (0..elements)
        .map(|index| (index % 31) as f32 / 31.0 - 0.5)
        .collect();
    let key = upload_values(&mut runtime.backend, &values);
    let value = upload_values(
        &mut runtime.backend,
        &values.iter().map(|value| value * -0.5).collect::<Vec<_>>(),
    );
    for layer in 0..config.n_layer {
        runtime
            .backend
            .kv_append_chunk_span(
                &key,
                &value,
                state.write_span(layer, range).unwrap(),
                shape,
                0,
                3,
            )
            .unwrap();
    }
    state.commit_append(range).unwrap();
    let mut session = GenerationSession::new();
    session.activations = Some(Activations::new(&mut runtime.backend, config).unwrap());
    session.state = Some(state);
    session.evaluated_tokens = vec![0, 1, 2];
    session
}

fn inputs<'a>(
    sessions: &'a mut [GenerationSession<CpuBackend>],
    options: &'a GenerateOptions,
) -> Vec<BatchSession<'a, CpuBackend>> {
    sessions
        .iter_mut()
        .map(|session| BatchSession {
            session,
            transcript: &[0, 1, 2, 3],
            options,
        })
        .collect()
}

fn session_bytes(
    backend: &mut CpuBackend,
    session: &GenerationSession<CpuBackend>,
) -> Vec<BufferSnapshot> {
    let state = session.state.as_ref().unwrap().snapshot(backend).unwrap();
    let mut buffers = vec![state.device_position];
    for segment in state.segments {
        for (key, value) in segment.layers {
            buffers.extend([key, value]);
        }
    }
    buffers.extend(
        HibernatedActivations::capture(backend, session.activations.as_ref().unwrap())
            .unwrap()
            .buffers,
    );
    buffers
}

fn session_kv_bytes(
    backend: &mut CpuBackend,
    session: &GenerationSession<CpuBackend>,
) -> Vec<BufferSnapshot> {
    let state = session.state.as_ref().unwrap().snapshot(backend).unwrap();
    let mut buffers = Vec::new();
    for segment in state.segments {
        for (key, value) in segment.layers {
            buffers.extend([key, value]);
        }
    }
    buffers
}

struct BatchObservation {
    tokens: Vec<u32>,
    evaluated_tokens: Vec<Vec<u32>>,
    logits: Vec<Vec<f32>>,
}

fn batch_logits(
    runtime: &mut Runtime<CpuBackend>,
    sessions: &[GenerationSession<CpuBackend>],
) -> Vec<Vec<f32>> {
    let mut logits = Vec::with_capacity(sessions.len());
    for session in sessions {
        let mut row = vec![0.0; runtime.vocab_size()];
        runtime
            .read_session_logits(session, &mut row)
            .expect("batch session retains logits");
        logits.push(row);
    }
    logits
}

fn single_row_observation() -> BatchObservation {
    let mut runtime = attention_runtime(2, true);
    let mut sessions = [prefix_session(&mut runtime), prefix_session(&mut runtime)];
    let options = GenerateOptions::greedy(1);
    let mut tokens = Vec::with_capacity(sessions.len());
    for session in &mut sessions {
        let result = runtime
            .generate_session_tokens(
                session,
                &[0, 1, 2, 3],
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("clean single-row decode succeeds");
        tokens.extend(result.tokens);
    }
    let evaluated_tokens = sessions
        .iter()
        .map(|session| session.evaluated_tokens.clone())
        .collect();
    let logits = batch_logits(&mut runtime, &sessions);
    BatchObservation {
        tokens,
        evaluated_tokens,
        logits,
    }
}

fn snapshots_have_nonzero_bytes(snapshots: &[BufferSnapshot]) -> bool {
    snapshots
        .iter()
        .any(|snapshot| snapshot.bytes().iter().any(|byte| *byte != 0))
}

struct FinishedSibling {
    evaluated_tokens: Vec<u32>,
    emitted_tokens: Vec<u32>,
    bytes: Vec<BufferSnapshot>,
    logits: Vec<f32>,
}

struct QueuedSession {
    runtime: GenerationSession<CpuBackend>,
    emitted_tokens: Vec<u32>,
}

struct SharedPrefixDriver {
    inner: LeoneRuntimeDriver<CpuBackend>,
    source: GenerationSession<CpuBackend>,
    finished: Vec<FinishedSibling>,
    source_after: Option<Vec<BufferSnapshot>>,
    source_kv_after: Option<u64>,
}

impl SharedPrefixDriver {
    fn new(runtime: Runtime<CpuBackend>, source: GenerationSession<CpuBackend>) -> Self {
        Self {
            inner: LeoneRuntimeDriver::new(runtime),
            source,
            finished: Vec::new(),
            source_after: None,
            source_kv_after: None,
        }
    }

    fn snapshot_source(&mut self) {
        let runtime = self.inner.runtime_mut();
        self.source_after = Some(session_bytes(&mut runtime.backend, &self.source));
        self.source_kv_after = Some(
            runtime
                .backend
                .memory_accounting()
                .class(MemoryClass::KvCache)
                .live_bytes,
        );
    }
}

impl PausableGenerationDriver for SharedPrefixDriver {
    type Session = QueuedSession;
    type PendingPrefill = PendingPrefill<CpuBackend>;
    type ReadyPrefill = ReadyPrefill<CpuBackend>;
    type Error = LeoneRuntimeDriverError;

    fn start_session(&mut self) -> Self::Session {
        let runtime = self
            .inner
            .runtime_mut()
            .reuse_prefix_session(&self.source)
            .expect("shared prefix fork succeeds");
        QueuedSession {
            runtime,
            emitted_tokens: Vec::new(),
        }
    }

    fn begin_prefill(
        &mut self,
        session: &mut Self::Session,
        transcript: &[u32],
        options: GenerateOptions,
    ) -> Result<Self::PendingPrefill, Self::Error> {
        self.inner
            .runtime_mut()
            .begin_prefill(&mut session.runtime, transcript, options)
            .map_err(Into::into)
    }

    fn advance_prefill(
        &mut self,
        pending: Self::PendingPrefill,
        budget: NonZeroUsize,
        cancelled: &AtomicBool,
    ) -> Result<DriverPrefillProgress<Self::PendingPrefill, Self::ReadyPrefill>, Self::Error> {
        let before = pending.processed_tokens();
        let progress = self
            .inner
            .runtime_mut()
            .advance_prefill(pending, budget, || cancelled.load(Ordering::Acquire))
            .map_err(LeoneRuntimeDriverError::from)?;
        Ok(match progress {
            PrefillProgress::Pending(pending) => DriverPrefillProgress::Pending {
                processed_tokens: pending.processed_tokens().saturating_sub(before),
                pending,
            },
            PrefillProgress::Ready(ready) => DriverPrefillProgress::Ready {
                processed_tokens: ready.processed_tokens().saturating_sub(before),
                ready,
            },
            PrefillProgress::Cancelled(cancelled) => DriverPrefillProgress::Cancelled {
                processed_tokens: cancelled.processed_tokens().saturating_sub(before),
            },
        })
    }

    fn finish_prefill(
        &mut self,
        ready: Self::ReadyPrefill,
        session: &mut Self::Session,
    ) -> Result<(), Self::Error> {
        self.inner
            .runtime_mut()
            .finish_prefill(ready, &mut session.runtime)
            .map_err(Into::into)
    }

    fn generate_quantum(
        &mut self,
        session: &mut Self::Session,
        transcript: &[u32],
        options: GenerateOptions,
        cancelled: &AtomicBool,
    ) -> Result<GenerationQuantum, Self::Error> {
        let quantum = PausableGenerationDriver::generate_quantum(
            &mut self.inner,
            &mut session.runtime,
            transcript,
            options,
            cancelled,
        )?;
        session.emitted_tokens.extend_from_slice(&quantum.tokens);
        Ok(quantum)
    }

    fn finish_session(
        &mut self,
        session: &mut Self::Session,
        status: RequestStatus,
    ) -> Result<(), Self::Error> {
        if status == RequestStatus::Finished {
            let evaluated_tokens = session.runtime.evaluated_tokens.clone();
            let (bytes, logits) = {
                let runtime = self.inner.runtime_mut();
                let mut logits = vec![0.0; runtime.vocab_size()];
                runtime
                    .read_session_logits(&session.runtime, &mut logits)
                    .expect("finished sibling retains raw logits");
                let bytes = session_bytes(&mut runtime.backend, &session.runtime);
                (bytes, logits)
            };
            self.finished.push(FinishedSibling {
                evaluated_tokens,
                emitted_tokens: session.emitted_tokens.clone(),
                bytes,
                logits,
            });
        }
        PausableGenerationDriver::finish_session(&mut self.inner, &mut session.runtime, status)?;
        if status == RequestStatus::Finished {
            session.runtime.invalidate();
        }
        self.snapshot_source();
        Ok(())
    }
}

fn queued_options() -> GenerateOptions {
    GenerateOptions {
        prefill_chunk_tokens: 2,
        decode_execution: DecodeExecution::Eager,
        ..GenerateOptions::greedy(2)
    }
}

fn actual_prefix_session(runtime: &mut Runtime<CpuBackend>) -> GenerationSession<CpuBackend> {
    let mut session = GenerationSession::new();
    let pending = runtime
        .begin_prefill(&mut session, &[0, 1, 2], queued_options())
        .expect("source prefill starts");
    let pending = match runtime
        .advance_prefill(pending, NonZeroUsize::new(2).unwrap(), || false)
        .expect("source prefill first chunk succeeds")
    {
        PrefillProgress::Pending(pending) => {
            assert_eq!(pending.processed_tokens(), 2);
            pending
        }
        PrefillProgress::Ready(_) => panic!("source prefill completed before its second chunk"),
        PrefillProgress::Cancelled(_) => panic!("source prefill was cancelled"),
    };
    let ready = match runtime
        .advance_prefill(pending, NonZeroUsize::new(2).unwrap(), || false)
        .expect("source prefill final chunk succeeds")
    {
        PrefillProgress::Ready(ready) => {
            assert_eq!(ready.processed_tokens(), 3);
            ready
        }
        PrefillProgress::Pending(_) => panic!("source prefill did not complete"),
        PrefillProgress::Cancelled(_) => panic!("source prefill was cancelled"),
    };
    runtime
        .finish_prefill(ready, &mut session)
        .expect("source prefill commits");
    session
}

fn queued_policy() -> SchedulerPolicy {
    SchedulerPolicy {
        max_active_requests: 3,
        max_queued_requests: 3,
        max_batch_requests: 2,
        max_reserved_kv_bytes: 1 << 20,
        kv_bytes_per_token: 8,
        kv_page_tokens: 4,
        max_prompt_tokens: 16,
        max_output_tokens: 2,
        service_quantum_tokens: 1,
        prefill_chunk_tokens: 2,
        urgent_window_ns: 0,
        max_prefix_credit_tokens: 3,
    }
}

fn isolated_sibling(prompt: &[u32]) -> FinishedSibling {
    let mut runtime = queued_runtime();
    let source = actual_prefix_session(&mut runtime);
    let mut session = runtime
        .reuse_prefix_session(&source)
        .expect("isolated prefix fork succeeds");
    let mut options = queued_options();
    options.max_tokens = 1;
    let mut emitted_tokens = Vec::new();
    let mut transcript = prompt.to_vec();
    for _ in 0..2 {
        let result = runtime
            .generate_session_tokens(
                &mut session,
                &transcript,
                options.clone(),
                |_| Ok(()),
                || false,
            )
            .expect("isolated sibling succeeds");
        transcript.extend_from_slice(&result.tokens);
        emitted_tokens.extend(result.tokens);
    }
    let mut logits = vec![0.0; runtime.vocab_size()];
    runtime
        .read_session_logits(&session, &mut logits)
        .expect("isolated sibling retains raw logits");
    FinishedSibling {
        evaluated_tokens: session.evaluated_tokens.clone(),
        emitted_tokens,
        bytes: session_bytes(&mut runtime.backend, &session),
        logits,
    }
}

fn finished_for<'a>(captures: &'a [FinishedSibling], prompt: &[u32]) -> &'a FinishedSibling {
    captures
        .iter()
        .find(|capture| capture.evaluated_tokens.starts_with(prompt))
        .expect("finished sibling capture")
}

#[test]
fn preparation_failure_keeps_committed_rows_and_clears_replay() {
    let mut runtime = attention_runtime(2, true);
    let source = prefix_session(&mut runtime);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    let before: Vec<_> = sessions
        .iter()
        .map(|session| session_bytes(&mut runtime.backend, session))
        .collect();
    let kv_bytes = runtime
        .backend
        .memory_accounting()
        .class(MemoryClass::KvCache)
        .live_bytes;
    runtime.batch_graph_signature = Some(vec![99]);
    for session in &mut sessions {
        let state = session.state.as_mut().unwrap();
        state.graph_revision = Some(crate::kv::KvGraphRevision {
            layout: state.revision(),
            generation: 0,
        });
    }
    runtime.backend.attention_probe = Some(AttentionProbe {
        prepare_fail_after: Some(1),
        ..AttentionProbe::default()
    });
    let options = GenerateOptions::greedy(1);
    let error = runtime
        .generate_session_batch_token(&mut inputs(&mut sessions, &options))
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert!(error.to_string().contains("injected preparation failure"));
    assert!(runtime.batch_graph_signature.is_none());
    for (session, original) in sessions.iter().zip(before) {
        assert_eq!(session.evaluated_tokens, [0, 1, 2]);
        assert_eq!(session.state.as_ref().unwrap().position, 3);
        assert!(session.state.as_ref().unwrap().graph_revision.is_none());
        assert_eq!(session_bytes(&mut runtime.backend, session), original);
    }
    assert_eq!(
        runtime
            .backend
            .memory_accounting()
            .class(MemoryClass::KvCache)
            .live_bytes,
        kv_bytes
    );
    assert_eq!(
        runtime.backend.attention_probe.as_ref().unwrap().events,
        vec![
            AttentionEvent::Prepare {
                rows: 2,
                device_positions: false
            },
            AttentionEvent::Prepare {
                rows: 2,
                device_positions: false
            },
        ]
    );
}

fn batch_failure_recovery(fail_qkv: bool, expected_error: &str) {
    let mut runtime = attention_runtime(2, true);
    let source = prefix_session(&mut runtime);
    let source_kv = runtime
        .backend
        .memory_accounting()
        .class(MemoryClass::KvCache);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    let options = GenerateOptions::greedy(1);
    let before_failure = runtime
        .backend
        .memory_accounting()
        .class(MemoryClass::KvCache);
    let before_evaluated = sessions
        .iter()
        .map(|session| session.evaluated_tokens.clone())
        .collect::<Vec<_>>();
    runtime.backend.verification_probe = Some(VerificationProbe {
        fail_qkv,
        fail_ffn: !fail_qkv,
        ..VerificationProbe::default()
    });
    let error = runtime
        .generate_session_batch_token(&mut inputs(&mut sessions, &options))
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert!(error.to_string().contains(expected_error));
    assert_eq!(
        runtime
            .backend
            .verification_probe
            .as_ref()
            .unwrap()
            .drop_graph_calls,
        1
    );
    assert!(runtime.batch_graph_signature.is_none());
    assert_eq!(
        sessions
            .iter()
            .map(|session| session.evaluated_tokens.clone())
            .collect::<Vec<_>>(),
        before_evaluated
    );
    assert!(sessions
        .iter()
        .all(|session| session.state.as_ref().unwrap().position == 3));
    assert!(
        runtime
            .backend
            .memory_accounting()
            .class(MemoryClass::KvCache)
            .live_bytes
            >= before_failure.live_bytes
    );

    for session in &mut sessions {
        runtime
            .discard_session(session)
            .expect("quarantined session cleanup succeeds");
    }
    assert!(sessions.iter().all(GenerationSession::is_empty));
    assert_eq!(
        runtime
            .backend
            .memory_accounting()
            .class(MemoryClass::KvCache)
            .live_bytes,
        source_kv.live_bytes
    );
    assert_eq!(
        runtime
            .backend
            .verification_probe
            .as_ref()
            .unwrap()
            .drop_graph_calls,
        3
    );

    let mut retry_sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    let retry_tokens = runtime
        .generate_session_batch_token(&mut inputs(&mut retry_sessions, &options))
        .expect("fresh batch readmission succeeds");
    let retry_evaluated = retry_sessions
        .iter()
        .map(|session| session.evaluated_tokens.clone())
        .collect::<Vec<_>>();
    let retry_token_ids = retry_tokens
        .iter()
        .map(|token| token.id)
        .collect::<Vec<_>>();
    let retry_logits = batch_logits(&mut runtime, &retry_sessions);
    let clean = single_row_observation();
    assert_eq!(retry_token_ids, clean.tokens);
    assert_eq!(retry_evaluated, clean.evaluated_tokens);
    assert_eq!(retry_logits, clean.logits);
}

#[test]
fn qkv_failure_retires_batch_resources_before_fresh_readmission() {
    batch_failure_recovery(true, "injected QKV failure");
}

#[test]
fn ffn_failure_retires_batch_resources_before_fresh_readmission() {
    batch_failure_recovery(false, "injected FFN failure");
}

#[test]
fn failed_batch_retirement_keeps_pending_owners_until_discard() {
    let mut runtime = attention_runtime(2, true);
    let source = prefix_session(&mut runtime);
    let source_kv = runtime
        .backend
        .memory_accounting()
        .class(MemoryClass::KvCache);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    let before_failure = runtime
        .backend
        .memory_accounting()
        .class(MemoryClass::KvCache);
    runtime.backend.verification_probe = Some(VerificationProbe {
        fail_qkv: true,
        fail_drop: true,
        ..VerificationProbe::default()
    });
    let error = runtime
        .generate_session_batch_token(&mut inputs(&mut sessions, &GenerateOptions::greedy(1)))
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert!(error
        .to_string()
        .contains("injected graph retirement failure"));
    assert!(
        runtime
            .backend
            .memory_accounting()
            .class(MemoryClass::KvCache)
            .live_bytes
            > before_failure.live_bytes
    );
    assert_eq!(
        runtime
            .backend
            .verification_probe
            .as_ref()
            .unwrap()
            .drop_graph_calls,
        1
    );

    for session in &mut sessions {
        runtime
            .discard_session(session)
            .expect("terminal cleanup retries graph retirement");
    }
    assert!(sessions.iter().all(GenerationSession::is_empty));
    assert_eq!(
        runtime
            .backend
            .memory_accounting()
            .class(MemoryClass::KvCache)
            .live_bytes,
        source_kv.live_bytes
    );
    assert_eq!(
        runtime
            .backend
            .verification_probe
            .as_ref()
            .unwrap()
            .drop_graph_calls,
        3
    );
}

#[test]
fn persistent_batch_retirement_failure_keeps_sessions_until_retry() {
    let mut runtime = attention_runtime(2, true);
    let source = prefix_session(&mut runtime);
    let source_kv = runtime
        .backend
        .memory_accounting()
        .class(MemoryClass::KvCache);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    runtime.backend.verification_probe = Some(VerificationProbe {
        fail_qkv: true,
        fail_drop_forever: true,
        ..VerificationProbe::default()
    });
    let error = runtime
        .generate_session_batch_token(&mut inputs(&mut sessions, &GenerateOptions::greedy(1)))
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    for session in &mut sessions {
        assert!(runtime.discard_session(session).is_err());
    }
    assert!(sessions.iter().all(|session| !session.is_empty()));
    assert_eq!(
        runtime
            .backend
            .verification_probe
            .as_ref()
            .unwrap()
            .drop_graph_calls,
        3
    );
    runtime
        .backend
        .verification_probe
        .as_mut()
        .unwrap()
        .fail_drop_forever = false;
    for session in &mut sessions {
        runtime
            .discard_session(session)
            .expect("terminal cleanup succeeds after retirement recovers");
    }
    assert!(sessions.iter().all(GenerationSession::is_empty));
    assert_eq!(
        runtime
            .backend
            .memory_accounting()
            .class(MemoryClass::KvCache)
            .live_bytes,
        source_kv.live_bytes
    );
}

#[test]
fn failed_retirement_keeps_preflight_pending_allocations() {
    let mut runtime = attention_runtime(1, false);
    let source = prefix_session(&mut runtime);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    for session in &mut sessions {
        session
            .state
            .as_mut()
            .unwrap()
            .prepare_append(&mut runtime.backend, 1, 1, 1)
            .unwrap();
    }
    runtime.backend.preflight_probe = Some(PreflightProbe {
        fail_end_keep: true,
        ..PreflightProbe::default()
    });
    runtime.backend.verification_probe = Some(VerificationProbe {
        fail_drop: true,
        ..VerificationProbe::default()
    });
    runtime.backend.graph_capture = true;
    let generation = runtime.decode_graph_generation;
    let error = runtime
        .generate_session_batch_token(&mut inputs(&mut sessions, &GenerateOptions::greedy(1)))
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("injected graph retirement failure"));
    assert_eq!(
        runtime
            .backend
            .verification_probe
            .as_ref()
            .unwrap()
            .drop_graph_calls,
        1
    );
    assert_eq!(runtime.decode_graph_generation, generation + 1);
    for session in &sessions {
        assert_eq!(
            session.state.as_ref().unwrap().read_spans(0).unwrap().len(),
            3,
            "failed retirement must retain the new empty capture tail"
        );
    }
    for session in &mut sessions {
        runtime.discard_session(session).unwrap();
    }
}

#[test]
fn failed_retirement_keeps_attention_preparation_pending_allocations() {
    let mut runtime = attention_runtime(2, true);
    let source = prefix_session(&mut runtime);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    runtime.backend.attention_probe = Some(AttentionProbe {
        prepare_fail_after: Some(1),
        ..AttentionProbe::default()
    });
    runtime.backend.verification_probe = Some(VerificationProbe {
        fail_drop: true,
        ..VerificationProbe::default()
    });
    let error = runtime
        .generate_session_batch_token(&mut inputs(&mut sessions, &GenerateOptions::greedy(1)))
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("injected graph retirement failure"));
    assert_eq!(
        runtime
            .backend
            .verification_probe
            .as_ref()
            .unwrap()
            .drop_graph_calls,
        1
    );
    for session in &sessions {
        assert_eq!(
            session.state.as_ref().unwrap().read_spans(0).unwrap().len(),
            2,
            "failed retirement must retain the pending append tail"
        );
    }
    for session in &mut sessions {
        runtime.discard_session(session).unwrap();
    }
}

#[test]
fn queued_prefix_siblings_resume_prefill_cancel_one_and_match_isolated_cpu() {
    let prompts = [
        vec![0, 1, 2, 3, 4, 5, 6],
        vec![0, 1, 2, 7, 8, 9, 10],
        vec![0, 1, 2, 11, 12, 13, 14],
    ];
    let mut runtime = queued_runtime();
    let source = actual_prefix_session(&mut runtime);
    let source_tokens = source.evaluated_tokens.clone();
    let source_kv = session_kv_bytes(&mut runtime.backend, &source);
    assert!(snapshots_have_nonzero_bytes(&source_kv));
    let source_bytes = session_bytes(&mut runtime.backend, &source);
    let source_kv_bytes = runtime
        .backend
        .memory_accounting()
        .class(MemoryClass::KvCache)
        .live_bytes;
    let requests: Vec<_> = prompts
        .iter()
        .map(|prompt| ScheduledGenerationRequest::new(prompt.clone(), queued_options()))
        .collect();
    let driver = SharedPrefixDriver::new(runtime, source);
    let executor = RuntimeQuantumExecutor::new(driver);
    let mut service =
        ScheduledService::new(queued_policy(), executor).expect("queued CPU policy is valid");
    for (index, request) in requests.iter().enumerate() {
        let id = RequestId(u64::try_from(index + 1).expect("request index"));
        let prompt_tokens = u64::try_from(request.prompt_tokens.len()).expect("prompt length");
        let outcome = service
            .admit(
                RequestSpec {
                    id,
                    arrival_ns: 0,
                    prompt_tokens,
                    prefix_reused_tokens: 3,
                    max_output_tokens: 2,
                    priority: 1,
                    deadline_ns: None,
                },
                request,
                0,
            )
            .expect("sibling admission succeeds");
        assert!(matches!(
            outcome,
            crate::scheduler::AdmissionOutcome::Admitted { .. }
        ));
    }

    let first = service.tick_batch(0).expect("first prefill batch succeeds");
    assert_eq!(first.len(), 2);
    assert!(first
        .iter()
        .all(|quantum| quantum.kind == DispatchKind::Prefill));
    assert!(first.iter().all(|quantum| quantum.progress_tokens == 2));
    assert_eq!(service.status(RequestId(3)), Some(RequestStatus::Queued));
    requests[1].cancel();

    let mut quantums = first;
    for now in 1..32 {
        quantums.extend(
            service
                .tick_batch(now)
                .expect("queued sibling tick succeeds"),
        );
        if !service.has_runnable_requests() {
            break;
        }
    }
    assert!(!service.has_runnable_requests());
    assert_eq!(service.status(RequestId(1)), Some(RequestStatus::Finished));
    assert_eq!(service.status(RequestId(2)), Some(RequestStatus::Cancelled));
    assert_eq!(service.status(RequestId(3)), Some(RequestStatus::Finished));
    assert!(quantums.iter().any(|quantum| {
        quantum.completion.request_id == RequestId(2)
            && quantum.kind == DispatchKind::Prefill
            && quantum.completion.status == RequestStatus::Cancelled
    }));
    for id in [RequestId(1), RequestId(3)] {
        assert!(
            quantums
                .iter()
                .filter(|quantum| {
                    quantum.completion.request_id == id && quantum.kind == DispatchKind::Prefill
                })
                .count()
                >= 2
        );
    }
    assert!(service
        .output(RequestId(2))
        .is_some_and(|tokens| tokens.is_empty()));
    let oracle_a = isolated_sibling(&prompts[0]);
    let oracle_c = isolated_sibling(&prompts[2]);
    let captures = &service.executor().driver().finished;
    assert_eq!(captures.len(), 2);
    let capture_a = finished_for(captures, &prompts[0]);
    let capture_c = finished_for(captures, &prompts[2]);
    assert_ne!(capture_a.logits, capture_c.logits);
    for (id, prompt, oracle) in [
        (RequestId(1), &prompts[0], &oracle_a),
        (RequestId(3), &prompts[2], &oracle_c),
    ] {
        let capture = finished_for(captures, prompt);
        let completed = &service
            .executor()
            .completed(id)
            .expect("surviving completion")
            .tokens;
        assert_eq!(service.output(id), Some(completed.as_slice()));
        assert_eq!(completed, &oracle.emitted_tokens);
        assert_eq!(capture.evaluated_tokens, oracle.evaluated_tokens);
        assert_eq!(capture.bytes, oracle.bytes);
        assert_eq!(capture.logits, oracle.logits);
    }
    assert!(oracle_a.logits.iter().any(|logit| *logit != 0.0));
    assert_ne!(
        finished_for(captures, &prompts[0]).evaluated_tokens,
        finished_for(captures, &prompts[2]).evaluated_tokens
    );
    let driver = service.executor().driver();
    assert_eq!(driver.source.evaluated_tokens, source_tokens);
    assert_eq!(driver.source_after.as_ref(), Some(&source_bytes));
    assert_eq!(driver.source_kv_after, Some(source_kv_bytes));
    assert_eq!(service.executor().active_len(), 0);
    assert_eq!(
        service
            .executor()
            .completed(RequestId(2))
            .expect("cancelled completion")
            .tokens,
        Vec::<u32>::new()
    );
}

fn projected_rows(runtime: &mut Runtime<CpuBackend>) -> VerifyActivations<CpuBackend> {
    let config = &runtime.model.config;
    let mut batch = VerifyActivations::new(&mut runtime.backend, config, 2).unwrap();
    let queries: Vec<_> = (0..2 * config.n_embd)
        .map(|index| (index % 17) as f32 / 17.0 - 0.5)
        .collect();
    let keys: Vec<_> = (0..2 * config.n_head_kv * config.head_dim)
        .map(|index| (index % 23) as f32 / 23.0 - 0.5)
        .collect();
    batch.query = upload_values(&mut runtime.backend, &queries);
    batch.key = upload_values(&mut runtime.backend, &keys);
    batch.value = upload_values(
        &mut runtime.backend,
        &keys.iter().map(|key| key * 2.0).collect::<Vec<_>>(),
    );
    batch
}

fn scalar_outputs(
    runtime: &mut Runtime<CpuBackend>,
    sessions: &[GenerationSession<CpuBackend>],
    normalized: bool,
) -> Vec<u32> {
    let mut outputs = Vec::new();
    for session in sessions {
        let state = session.state.as_ref().unwrap();
        let single = session.activations.as_ref().unwrap();
        let query = if normalized {
            &single.query_norm
        } else {
            &single.query
        };
        let spans = state.read_spans(0).unwrap();
        let mut output = runtime
            .backend
            .allocate(BufferLayout::f32(state.shape.query_elements().unwrap()).unwrap())
            .unwrap();
        runtime
            .backend
            .attention_decode_spans(
                query,
                KvReadView::new(&spans).unwrap(),
                &mut output,
                state.shape,
                Position::Host(3),
            )
            .unwrap();
        let mut row = vec![0.0; state.shape.query_elements().unwrap()];
        runtime.backend.read_f32(&output, &mut row).unwrap();
        outputs.extend(row.into_iter().map(f32::to_bits));
    }
    outputs
}

fn run_attention_phases(normalized: bool, device_positions: bool) {
    let mut runtime = attention_runtime(1, normalized);
    let source = prefix_session(&mut runtime);
    let source_bytes = session_bytes(&mut runtime.backend, &source);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    let mut batch = projected_rows(&mut runtime);
    let positions = [3_u32, 3].map(|position| {
        runtime
            .backend
            .upload(BufferLayout::u32(1).unwrap(), &position.to_le_bytes())
            .unwrap()
    });
    let capture = device_positions.then_some(positions.as_slice());
    let options = GenerateOptions::greedy(1);
    let config = runtime.model.config.clone();
    let mut rows = inputs(&mut sessions, &options);
    let ranges = runtime.prepare_batch_appends(&mut rows, 1).unwrap();
    runtime.backend.attention_probe = Some(AttentionProbe::default());
    runtime
        .prepare_batch_attention_views(&mut rows, capture)
        .unwrap();
    Runtime::<CpuBackend>::forward_batch_attention_rows(
        &mut runtime.backend,
        &runtime.model.weights.layers[0],
        &mut rows,
        &mut batch,
        0,
        VectorShape::new(config.n_head, config.head_dim).unwrap(),
        VectorShape::new(config.n_head_kv, config.head_dim).unwrap(),
        config.rms_epsilon,
        config.rope_theta,
        config.n_embd,
        capture,
        &ranges,
    )
    .unwrap();
    assert_eq!(
        runtime.backend.attention_probe.as_ref().unwrap().events,
        vec![
            AttentionEvent::Prepare {
                rows: 2,
                device_positions
            },
            AttentionEvent::Append,
            AttentionEvent::Append,
            AttentionEvent::Dispatch { rows: 2 },
            AttentionEvent::Scatter,
            AttentionEvent::Scatter,
        ]
    );
    let mut actual = vec![0.0; 2 * config.n_embd];
    runtime
        .backend
        .read_f32(&batch.attention, &mut actual)
        .unwrap();
    let expected = scalar_outputs(&mut runtime, &sessions, normalized);
    assert_eq!(
        actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(session_bytes(&mut runtime.backend, &source), source_bytes);
}

#[test]
fn append_dispatch_scatter_preserves_each_rows_scalar_attention() {
    run_attention_phases(false, false);
    run_attention_phases(true, false);
    run_attention_phases(false, true);
}

#[test]
fn capture_preparation_failure_occurs_before_capture_or_row_writes() {
    let mut runtime = attention_runtime(1, false);
    let source = prefix_session(&mut runtime);
    let before = session_bytes(&mut runtime.backend, &source);
    let mut sessions = [source];
    let options = GenerateOptions::greedy(1);
    let mut rows = inputs(&mut sessions, &options);
    let ranges = runtime.prepare_batch_appends(&mut rows, 1).unwrap();
    let mut batch = VerifyActivations::new(&mut runtime.backend, &runtime.model.config, 1).unwrap();
    let shapes = runtime.batch_vector_shapes(1).unwrap();
    let mut positions = [runtime
        .backend
        .allocate(BufferLayout::u32(1).unwrap())
        .unwrap()];
    runtime.backend.attention_probe = Some(AttentionProbe {
        prepare_fail_after: Some(0),
        ..AttentionProbe::default()
    });
    let error = runtime
        .capture_batch_forward(&mut rows, &mut batch, shapes, &ranges, &mut positions)
        .unwrap_err();
    assert!(error.to_string().contains("injected preparation failure"));
    Runtime::<CpuBackend>::discard_batch_pending(&mut rows).unwrap();
    drop(rows);
    assert_eq!(session_bytes(&mut runtime.backend, &sessions[0]), before);
    assert_eq!(
        runtime.backend.attention_probe.as_ref().unwrap().events,
        vec![AttentionEvent::Prepare {
            rows: 1,
            device_positions: true
        },]
    );
}

#[test]
fn context_bucket_growth_preserves_cpu_kv_prefix_and_logits() {
    let mut runtime = attention_runtime(1, false);
    runtime.model.config.context_length = 2_048;
    let options = GenerateOptions {
        decode_execution: DecodeExecution::Eager,
        ..GenerateOptions::greedy(1)
    };
    let initial_prompt = (0..480).map(|index| index as u32 % 256).collect::<Vec<_>>();
    let mut session = GenerationSession::new();
    runtime
        .generate_session_tokens(
            &mut session,
            &initial_prompt,
            options.clone(),
            |_| Ok(()),
            || false,
        )
        .expect("initial CPU generation succeeds");
    let state = session.state.as_ref().expect("initial state is retained");
    assert_eq!(state.shape.max_context(), 512);
    assert_eq!(state.position, 480);
    let before = state
        .snapshot(&mut runtime.backend)
        .expect("initial KV snapshot succeeds");
    let before_memory = runtime.backend.memory_accounting();
    let old_graph_generation = runtime.decode_graph_generation;
    let initial_revision = session.state.as_ref().unwrap().revision();
    session.state.as_mut().unwrap().graph_revision = Some(crate::kv::KvGraphRevision {
        layout: initial_revision,
        generation: old_graph_generation,
    });
    runtime.batch_graph_signature = Some(vec![old_graph_generation]);
    let mut grown_prompt = session.evaluated_tokens.clone();
    grown_prompt.extend((0..500).map(|index| (index as u32 + 1) % 256));

    let grown = runtime
        .generate_session_tokens(
            &mut session,
            &grown_prompt,
            options.clone(),
            |_| Ok(()),
            || false,
        )
        .expect("CPU growth generation succeeds");

    assert_eq!(
        session.last_replay().reuse_class,
        SessionReuseClass::AppendOnly
    );
    assert_eq!(session.last_replay().reused_tokens, 480);
    assert_eq!(session.last_replay().replayed_tokens, 0);
    assert_eq!(session.last_replay().computed_tokens, 500);
    let state = session.state.as_ref().expect("grown state is retained");
    assert_eq!(state.shape.max_context(), 1_024);
    assert_eq!(state.position, 980);
    assert!(runtime.decode_graph_generation > old_graph_generation);
    assert!(runtime.batch_graph_signature.is_none());
    assert!(state.graph_revision.is_none());
    let after = state
        .snapshot(&mut runtime.backend)
        .expect("grown KV snapshot succeeds");
    assert_eq!(after.shape.max_context(), 1_024);
    assert!(after.segments.len() > before.segments.len());
    for (before_segment, after_segment) in before.segments.iter().zip(&after.segments) {
        assert_eq!(before_segment.logical_start, after_segment.logical_start);
        assert_eq!(
            before_segment.committed_tokens,
            after_segment.committed_tokens
        );
        assert_eq!(
            before_segment.capacity_tokens,
            after_segment.capacity_tokens
        );
        assert_eq!(before_segment.layers, after_segment.layers);
    }
    assert!(
        runtime
            .backend
            .memory_accounting()
            .class(MemoryClass::KvCache)
            .live_bytes
            > before_memory.class(MemoryClass::KvCache).live_bytes
    );

    let mut expected_runtime = attention_runtime(1, false);
    expected_runtime.model.config.context_length = 2_048;
    let mut expected_session = GenerationSession::new();
    let expected = expected_runtime
        .generate_session_tokens(
            &mut expected_session,
            &grown_prompt,
            options,
            |_| Ok(()),
            || false,
        )
        .expect("cold larger-bucket CPU oracle succeeds");
    assert_eq!(grown.tokens, expected.tokens);
    let mut actual_logits = vec![0.0; runtime.vocab_size()];
    let mut expected_logits = vec![0.0; expected_runtime.vocab_size()];
    runtime
        .read_session_logits(&session, &mut actual_logits)
        .expect("retained logits are readable");
    expected_runtime
        .read_session_logits(&expected_session, &mut expected_logits)
        .expect("oracle logits are readable");
    assert_eq!(
        actual_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn context_growth_retires_pending_batch_graph_and_recaptures_sibling() {
    let mut runtime = attention_runtime(1, false);
    runtime.model.config.context_length = 2_048;
    runtime.backend.graph_capture = true;
    let config = runtime.model.config.clone();
    let make_session = |runtime: &mut Runtime<CpuBackend>| {
        let shape = AttentionShape::new(config.n_head, config.n_head_kv, config.head_dim, 512)
            .expect("initial graph shape is valid");
        let mut state = KvState::new(
            &mut runtime.backend,
            config.n_layer,
            shape,
            KvCacheDtype::F16,
        )
        .expect("initial graph state allocates");
        let range = state
            .prepare_append(&mut runtime.backend, config.n_layer, 480, 480)
            .expect("initial graph prefix allocates");
        state
            .commit_append(range)
            .expect("initial graph prefix commits");
        let mut session = GenerationSession::new();
        session.state = Some(state);
        session.activations = Some(
            Activations::new(&mut runtime.backend, &config).expect("graph activations allocate"),
        );
        session.evaluated_tokens = (0..480).map(|index| index as u32 % 256).collect();
        session
    };
    let mut sessions = [make_session(&mut runtime), make_session(&mut runtime)];
    let options = GenerateOptions::greedy(1);
    let mut first_transcript = sessions[0].evaluated_tokens.clone();
    first_transcript.push(0);
    let mut second_transcript = sessions[1].evaluated_tokens.clone();
    second_transcript.push(0);
    {
        let (first_session, second_session) = sessions.split_at_mut(1);
        let mut rows = [
            BatchSession {
                session: &mut first_session[0],
                transcript: &first_transcript,
                options: &options,
            },
            BatchSession {
                session: &mut second_session[0],
                transcript: &second_transcript,
                options: &options,
            },
        ];
        runtime
            .generate_session_batch_token(&mut rows)
            .expect("batch graph capture succeeds");
    }
    assert!(runtime.batch_graph_signature.is_some());
    assert!(sessions[0]
        .state
        .as_ref()
        .expect("first state")
        .has_pending_append());
    assert!(sessions[1]
        .state
        .as_ref()
        .expect("second state")
        .has_pending_append());
    let old_generation = runtime.decode_graph_generation;
    let reused_tokens = sessions[0].evaluated_tokens.len();
    let mut grown_prompt = sessions[0].evaluated_tokens.clone();
    grown_prompt.extend((0..500).map(|index| (index as u32 + 1) % 256));
    runtime
        .generate_session_tokens(
            &mut sessions[0],
            &grown_prompt,
            GenerateOptions::greedy(1),
            |_| Ok(()),
            || false,
        )
        .expect("growth retires the captured graph");
    assert_eq!(
        sessions[0].last_replay().reuse_class,
        SessionReuseClass::AppendOnly
    );
    assert_eq!(sessions[0].last_replay().reused_tokens, reused_tokens);
    assert_eq!(sessions[0].last_replay().replayed_tokens, 0);
    assert_eq!(
        sessions[0]
            .state
            .as_ref()
            .expect("grown state")
            .shape
            .max_context(),
        1_024
    );
    assert!(runtime.decode_graph_generation > old_generation);
    assert!(runtime.batch_graph_signature.is_none());
    assert!(sessions[1]
        .state
        .as_ref()
        .expect("sibling state")
        .has_pending_append());

    let mut sibling_transcript = sessions[1].evaluated_tokens.clone();
    sibling_transcript.push(0);
    {
        let mut sibling_row = [BatchSession {
            session: &mut sessions[1],
            transcript: &sibling_transcript,
            options: &options,
        }];
        runtime
            .generate_session_batch_token(&mut sibling_row)
            .expect("sibling recaptures after retirement");
    }
    assert!(runtime.batch_graph_signature.is_some());
    assert!(sessions[1]
        .state
        .as_ref()
        .expect("recaptured sibling state")
        .has_pending_append());
}

#[test]
fn failed_graph_preflight_restores_positions_and_allows_retry() {
    let mut runtime = attention_runtime(1, false);
    let source = prefix_session(&mut runtime);
    let mut sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    let original_positions: Vec<_> = sessions
        .iter()
        .map(|session| {
            session
                .state
                .as_ref()
                .unwrap()
                .device_position
                .allocation_identity()
        })
        .collect();
    runtime.backend.preflight_probe = Some(PreflightProbe {
        fail_end_keep: true,
        ..PreflightProbe::default()
    });
    runtime.backend.graph_capture = true;
    let options = GenerateOptions::greedy(1);
    let error = runtime
        .generate_session_batch_token(&mut inputs(&mut sessions, &options))
        .unwrap_err();
    assert_eq!(
        error.session_failure_effect(),
        Some(SessionFailureEffect::Quarantine)
    );
    assert!(error.to_string().contains("injected finalization failure"));
    let probe = runtime.backend.preflight_probe.as_ref().unwrap();
    assert_eq!(probe.end_keep_calls, 1);
    assert_eq!(probe.drop_calls, 1);
    assert_eq!(
        runtime.backend.memory_accounting().live_bytes,
        probe.begin_live_bytes.unwrap()
    );
    assert_eq!(
        runtime.backend.memory_accounting().live_allocations,
        probe.begin_live_allocations.unwrap()
    );
    let restored_positions: Vec<_> = sessions
        .iter()
        .map(|session| {
            session
                .state
                .as_ref()
                .unwrap()
                .device_position
                .allocation_identity()
        })
        .collect();
    assert_eq!(restored_positions, original_positions);
    assert!(runtime.batch_graph_signature.is_none());

    for session in &mut sessions {
        runtime.discard_session(session).unwrap();
    }
    runtime
        .backend
        .preflight_probe
        .as_mut()
        .unwrap()
        .fail_end_keep = false;
    let mut retry_sessions = [
        runtime.fork_session(&source).unwrap(),
        runtime.fork_session(&source).unwrap(),
    ];
    let mut retry_rows = inputs(&mut retry_sessions, &options);
    let before_retry = runtime.backend.memory_accounting();
    let (_, _, retry_positions) = runtime
        .prepare_batch_graph_capture(&mut retry_rows)
        .unwrap();
    runtime.restore_batch_capture_positions(&mut retry_rows, retry_positions);
    for row in &mut retry_rows {
        row.session
            .state
            .as_mut()
            .unwrap()
            .discard_pending()
            .unwrap();
    }
    drop(retry_rows);
    assert_eq!(
        runtime
            .backend
            .preflight_probe
            .as_ref()
            .unwrap()
            .end_keep_calls,
        2
    );
    assert_eq!(
        runtime.backend.memory_accounting().live_bytes,
        before_retry.live_bytes
    );
    assert_eq!(
        runtime.backend.memory_accounting().live_allocations,
        before_retry.live_allocations
    );
    for session in &mut retry_sessions {
        runtime.discard_session(session).unwrap();
    }
}
