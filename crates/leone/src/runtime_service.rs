//! Runs bounded generation quanta against retained runtime sessions.

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use thiserror::Error;

use crate::backend::Backend;
use crate::runtime::{GenerateOptions, GeneratedToken, GenerationSession, Runtime, RuntimeError};
use crate::scheduler::{RequestId, RequestStatus};
use crate::service::{QuantumExecutor, QuantumOutput};

/// The input and cancellation state for one scheduled generation.
#[derive(Clone, Debug)]
pub struct ScheduledGenerationRequest {
    /// Prompt tokens at admission.
    pub prompt_tokens: Vec<u32>,
    /// `max_tokens` is the total request limit, not the per-quantum bound.
    pub options: GenerateOptions,
    cancelled: Arc<AtomicBool>,
}

impl ScheduledGenerationRequest {
    pub fn new(prompt_tokens: Vec<u32>, options: GenerateOptions) -> Self {
        Self {
            prompt_tokens,
            options,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Requests cancellation at the next runtime callback or quantum boundary.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns true after any clone requests cancellation.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// The output of one bounded call to a pausable generation driver.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationQuantum {
    /// Token identifiers in emission order.
    pub tokens: Vec<u32>,
    /// Exact byte pieces reported by the tokenizer callback.
    pub pieces: Vec<Vec<u8>>,
    /// True when the uncancelled call emitted EOS or stopped before the bound.
    pub eos: bool,
    pub cancelled: bool,
}

/// Runs retained generation state for one bounded token quantum.
pub trait PausableGenerationDriver {
    type Session;
    type Error: Error + Send + Sync + 'static;

    /// Creates empty retained state after admission succeeds.
    fn start_session(&mut self) -> Self::Session;

    /// Generates at most `options.max_tokens` tokens from the transcript.
    fn generate_quantum(
        &mut self,
        session: &mut Self::Session,
        transcript: &[u32],
        options: GenerateOptions,
        cancelled: &AtomicBool,
    ) -> Result<GenerationQuantum, Self::Error>;

    /// Releases or invalidates retained state after a terminal status.
    fn finish_session(&mut self, session: &mut Self::Session, status: RequestStatus);
}

/// Adapts `Runtime` retained sessions to bounded generation calls.
pub struct LeoneRuntimeDriver<B: Backend> {
    runtime: Runtime<B>,
    eos_token: Option<u32>,
}

impl<B: Backend> LeoneRuntimeDriver<B> {
    /// Wraps a loaded runtime and records its tokenizer EOS identifier.
    pub fn new(runtime: Runtime<B>) -> Self {
        let eos_token = runtime.model().tokenizer().eos_token();
        Self { runtime, eos_token }
    }

    pub const fn runtime(&self) -> &Runtime<B> {
        &self.runtime
    }

    pub const fn runtime_mut(&mut self) -> &mut Runtime<B> {
        &mut self.runtime
    }

    pub fn into_runtime(self) -> Runtime<B> {
        self.runtime
    }
}

/// An invariant failure or runtime error during scheduled execution.
#[derive(Debug, Error)]
pub enum LeoneRuntimeDriverError {
    #[error("runtime token callback disagreed with the generation result")]
    CallbackMismatch,
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

impl<B: Backend> PausableGenerationDriver for LeoneRuntimeDriver<B> {
    type Session = GenerationSession<B>;
    type Error = LeoneRuntimeDriverError;

    fn start_session(&mut self) -> Self::Session {
        GenerationSession::new()
    }

    fn generate_quantum(
        &mut self,
        session: &mut Self::Session,
        transcript: &[u32],
        options: GenerateOptions,
        cancelled: &AtomicBool,
    ) -> Result<GenerationQuantum, Self::Error> {
        let token_budget = options.max_tokens;
        let mut callback_tokens = Vec::with_capacity(token_budget);
        let mut pieces = Vec::with_capacity(token_budget);
        let result = self.runtime.generate_session_tokens(
            session,
            transcript,
            options,
            |token: &GeneratedToken| {
                callback_tokens.push(token.id);
                pieces.push(token.bytes.clone());
                Ok(())
            },
            || cancelled.load(Ordering::Acquire),
        )?;
        // Pieces come from the callback and tokens from the result.
        if callback_tokens != result.tokens {
            return Err(LeoneRuntimeDriverError::CallbackMismatch);
        }

        let was_cancelled = result.stats.cancelled || cancelled.load(Ordering::Acquire);
        let emitted_eos = result
            .tokens
            .last()
            .is_some_and(|token| Some(*token) == self.eos_token);
        let stopped_early = result.tokens.len() < token_budget;
        Ok(GenerationQuantum {
            tokens: result.tokens,
            pieces,
            eos: !was_cancelled && (emitted_eos || stopped_early),
            cancelled: was_cancelled,
        })
    }

    fn finish_session(&mut self, session: &mut Self::Session, status: RequestStatus) {
        // Finished sessions may retain KV. Other terminal statuses drop it.
        if status != RequestStatus::Finished {
            session.invalidate();
        }
    }
}

struct RequestState<S> {
    session: S,
    transcript: Vec<u32>,
    options: GenerateOptions,
    remaining_tokens: usize,
    emitted_tokens: Vec<u32>,
    emitted_pieces: Vec<Vec<u8>>,
    cancelled: Arc<AtomicBool>,
}

/// The exact terminal transcript retained for a scheduled request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletedGeneration {
    pub status: RequestStatus,
    /// Token identifiers in emission order.
    pub tokens: Vec<u32>,
    /// Exact byte pieces in emission order.
    pub pieces: Vec<Vec<u8>>,
}

/// Executes scheduler selections with one retained session per admitted request.
pub struct RuntimeQuantumExecutor<D: PausableGenerationDriver> {
    driver: D,
    active: BTreeMap<RequestId, RequestState<D::Session>>,
    completed: BTreeMap<RequestId, CompletedGeneration>,
}

impl<D: PausableGenerationDriver> RuntimeQuantumExecutor<D> {
    pub fn new(driver: D) -> Self {
        Self {
            driver,
            active: BTreeMap::new(),
            completed: BTreeMap::new(),
        }
    }

    pub const fn driver(&self) -> &D {
        &self.driver
    }

    /// Returns the exact terminal transcript for a request.
    pub fn completed(&self, request_id: RequestId) -> Option<&CompletedGeneration> {
        self.completed.get(&request_id)
    }

    /// Returns the number of admitted requests with retained state.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }
}

/// An invariant or driver failure during scheduled generation.
#[derive(Debug, Error)]
pub enum RuntimeQuantumExecutorError {
    #[error("executor state already exists for request {0}")]
    DuplicateRequest(u64),
    #[error("executor state does not exist for request {0}")]
    UnknownRequest(u64),
    #[error("driver emitted {emitted} tokens for a {budget}-token quantum")]
    QuantumOverflow { budget: usize, emitted: usize },
    #[error("generation driver failed: {0}")]
    Driver(#[source] Box<dyn Error + Send + Sync>),
}

impl<D: PausableGenerationDriver> QuantumExecutor for RuntimeQuantumExecutor<D> {
    type Request = ScheduledGenerationRequest;
    type Error = RuntimeQuantumExecutorError;

    fn begin(&mut self, request_id: RequestId, request: &Self::Request) -> Result<(), Self::Error> {
        if self.active.contains_key(&request_id) {
            return Err(RuntimeQuantumExecutorError::DuplicateRequest(request_id.0));
        }
        self.completed.remove(&request_id);
        self.active.insert(
            request_id,
            RequestState {
                session: self.driver.start_session(),
                transcript: request.prompt_tokens.clone(),
                options: request.options.clone(),
                remaining_tokens: request.options.max_tokens,
                emitted_tokens: Vec::with_capacity(request.options.max_tokens),
                emitted_pieces: Vec::with_capacity(request.options.max_tokens),
                cancelled: Arc::clone(&request.cancelled),
            },
        );
        Ok(())
    }

    fn execute(
        &mut self,
        request_id: RequestId,
        token_budget: u32,
    ) -> Result<QuantumOutput, Self::Error> {
        let state = self
            .active
            .get_mut(&request_id)
            .ok_or(RuntimeQuantumExecutorError::UnknownRequest(request_id.0))?;
        // The scheduler quantum cannot exceed the remaining request limit.
        let budget = usize::try_from(token_budget)
            .expect("u32 fits usize")
            .min(state.remaining_tokens);
        let mut options = state.options.clone();
        options.max_tokens = budget;
        let quantum = self
            .driver
            .generate_quantum(
                &mut state.session,
                &state.transcript,
                options,
                &state.cancelled,
            )
            .map_err(|error| RuntimeQuantumExecutorError::Driver(Box::new(error)))?;
        if quantum.tokens.len() > budget {
            return Err(RuntimeQuantumExecutorError::QuantumOverflow {
                budget,
                emitted: quantum.tokens.len(),
            });
        }

        state.remaining_tokens -= quantum.tokens.len();
        state.transcript.extend_from_slice(&quantum.tokens);
        state.emitted_tokens.extend_from_slice(&quantum.tokens);
        state.emitted_pieces.extend(quantum.pieces);
        // Remaining == 0 ends the request even if the model did not emit EOS.
        Ok(QuantumOutput {
            tokens: quantum.tokens,
            eos: quantum.eos || state.remaining_tokens == 0,
            cancelled: quantum.cancelled,
        })
    }

    fn finish(&mut self, request_id: RequestId, status: RequestStatus) -> Result<(), Self::Error> {
        let mut state = self
            .active
            .remove(&request_id)
            .ok_or(RuntimeQuantumExecutorError::UnknownRequest(request_id.0))?;
        self.driver.finish_session(&mut state.session, status);
        self.completed.insert(
            request_id,
            CompletedGeneration {
                status,
                tokens: state.emitted_tokens,
                pieces: state.emitted_pieces,
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use crate::scheduler::{RequestSpec, SchedulerPolicy};
    use crate::service::ScheduledService;

    use super::*;

    #[derive(Debug, Default)]
    struct FakeDriver;

    impl PausableGenerationDriver for FakeDriver {
        type Session = ();
        type Error = Infallible;

        fn start_session(&mut self) -> Self::Session {}

        fn generate_quantum(
            &mut self,
            _session: &mut Self::Session,
            transcript: &[u32],
            options: GenerateOptions,
            cancelled: &AtomicBool,
        ) -> Result<GenerationQuantum, Self::Error> {
            let cancelled = cancelled.load(Ordering::Acquire);
            let tokens = if cancelled {
                Vec::new()
            } else {
                (0..options.max_tokens)
                    .map(|offset| {
                        transcript[0]
                            .wrapping_mul(31)
                            .wrapping_add(u32::try_from(transcript.len() + offset).unwrap())
                    })
                    .collect::<Vec<_>>()
            };
            let pieces = tokens
                .iter()
                .map(|token| token.to_le_bytes().to_vec())
                .collect();
            Ok(GenerationQuantum {
                tokens,
                pieces,
                eos: false,
                cancelled,
            })
        }

        fn finish_session(&mut self, _session: &mut Self::Session, _status: RequestStatus) {}
    }

    fn policy() -> SchedulerPolicy {
        SchedulerPolicy {
            max_active_requests: 2,
            max_queued_requests: 4,
            max_reserved_kv_bytes: 1 << 20,
            kv_bytes_per_token: 8,
            max_prompt_tokens: 64,
            max_output_tokens: 64,
            service_quantum_tokens: 2,
            urgent_window_ns: 0,
            max_prefix_credit_tokens: 0,
        }
    }

    fn spec(id: u64, prompt_tokens: usize, output_tokens: usize) -> RequestSpec {
        RequestSpec {
            id: RequestId(id),
            arrival_ns: 0,
            prompt_tokens: u64::try_from(prompt_tokens).unwrap(),
            prefix_reused_tokens: 0,
            max_output_tokens: u64::try_from(output_tokens).unwrap(),
            priority: 1,
            deadline_ns: None,
        }
    }

    fn run(requests: &[(u64, Vec<u32>, usize)]) -> BTreeMap<RequestId, Vec<u32>> {
        let mut service =
            ScheduledService::new(policy(), RuntimeQuantumExecutor::new(FakeDriver)).unwrap();
        for (id, prompt, max_tokens) in requests {
            let request = ScheduledGenerationRequest::new(
                prompt.clone(),
                GenerateOptions::greedy(*max_tokens),
            );
            service
                .admit(spec(*id, prompt.len(), *max_tokens), &request, *id)
                .unwrap();
        }
        let mut now = 100;
        while service.has_runnable_requests() {
            service.tick(now).unwrap();
            now += 1;
        }
        requests
            .iter()
            .map(|(id, _, _)| {
                let request_id = RequestId(*id);
                (
                    request_id,
                    service
                        .executor()
                        .completed(request_id)
                        .unwrap()
                        .tokens
                        .clone(),
                )
            })
            .collect()
    }

    #[test]
    fn interleaved_requests_match_isolated_execution() {
        let requests = vec![(1, vec![11, 12], 7), (2, vec![23, 24, 25], 5)];
        let concurrent = run(&requests);
        for request in &requests {
            let isolated = run(std::slice::from_ref(request));
            let id = RequestId(request.0);
            assert_eq!(concurrent[&id], isolated[&id]);
        }
    }

    #[test]
    fn cancellation_releases_state_at_the_next_quantum() {
        let mut service =
            ScheduledService::new(policy(), RuntimeQuantumExecutor::new(FakeDriver)).unwrap();
        let request = ScheduledGenerationRequest::new(vec![7, 8], GenerateOptions::greedy(12));
        service.admit(spec(9, 2, 12), &request, 0).unwrap();
        service.tick(1).unwrap();
        request.cancel();
        service.tick(2).unwrap();

        assert_eq!(service.status(RequestId(9)), Some(RequestStatus::Cancelled));
        assert_eq!(service.executor().active_len(), 0);
        assert_eq!(
            service
                .executor()
                .completed(RequestId(9))
                .unwrap()
                .tokens
                .len(),
            2
        );
    }
}
