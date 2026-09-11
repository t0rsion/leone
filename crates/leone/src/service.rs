use crate::scheduler::{
    AdmissionOutcome, ContinuousScheduler, Dispatch, DispatchKind, QuantumCompletion, RequestId,
    RequestSpec, RequestStatus, SchedulerError, SchedulerPolicy,
};
use std::collections::BTreeMap;
use std::error::Error;
use thiserror::Error;

/// The tokens and terminal state returned by one executor quantum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantumOutput {
    pub tokens: Vec<u32>,
    pub eos: bool,
    pub cancelled: bool,
    /// Progress returned by a resumable prefill call.
    pub prefill: Option<PrefillProgress>,
}

/// Reports prompt positions evaluated by one prefill dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillProgress {
    pub processed_tokens: u32,
    pub ready: bool,
}

impl QuantumOutput {
    /// Creates output from a decode dispatch.
    pub fn decode(tokens: Vec<u32>, eos: bool, cancelled: bool) -> Self {
        Self {
            tokens,
            eos,
            cancelled,
            prefill: None,
        }
    }

    /// Creates progress from a prefill dispatch.
    pub fn prefill(processed_tokens: u32, ready: bool, cancelled: bool) -> Self {
        Self {
            tokens: Vec::new(),
            eos: false,
            cancelled,
            prefill: Some(PrefillProgress {
                processed_tokens,
                ready,
            }),
        }
    }
}

/// Executes bounded inference quanta selected by the service scheduler.
pub trait QuantumExecutor {
    type Request;
    type Error: Error + Send + Sync + 'static;

    /// Creates executor state after the scheduler admits a request.
    fn begin(&mut self, request_id: RequestId, request: &Self::Request) -> Result<(), Self::Error>;

    /// Executes at most `token_budget` tokens for one admitted request.
    fn execute(
        &mut self,
        request_id: RequestId,
        token_budget: u32,
    ) -> Result<QuantumOutput, Self::Error>;

    /// Executes the selected phase for one scheduler dispatch.
    ///
    /// Executors that support resumable prefill inspect `dispatch.kind`.
    /// Existing executors use the decode-compatible default.
    fn execute_dispatch(&mut self, dispatch: &Dispatch) -> Result<QuantumOutput, Self::Error> {
        self.execute(dispatch.request_id, dispatch.token_budget)
    }

    /// Executes one scheduler-selected request set in dispatch order.
    ///
    /// The default preserves compatibility for executors without a batch path.
    fn execute_batch(
        &mut self,
        dispatches: &[crate::scheduler::Dispatch],
    ) -> Result<Vec<QuantumOutput>, Self::Error> {
        dispatches
            .iter()
            .map(|dispatch| self.execute_dispatch(dispatch))
            .collect()
    }

    /// Releases executor state after any terminal scheduler result.
    fn finish(&mut self, request_id: RequestId, status: RequestStatus) -> Result<(), Self::Error>;
}

/// An invalid transition across the scheduler and executor boundary.
#[derive(Debug, Error)]
pub enum ServiceError<E: Error + Send + Sync + 'static> {
    #[error("scheduler failed: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("quantum executor failed: {0}")]
    Executor(E),
    #[error("executor returned {emitted} tokens with a budget of {budget}")]
    ExecutorBudget { emitted: usize, budget: u32 },
    #[error("executor returned {actual} outputs for a batch of {expected}")]
    ExecutorBatchSize { expected: usize, actual: usize },
    #[error("prefill dispatch {request_id:?} emitted {emitted} generated tokens")]
    PrefillOutput {
        request_id: RequestId,
        emitted: usize,
    },
    #[error("prefill dispatch {request_id:?} returned no progress")]
    PrefillNoProgress { request_id: RequestId },
    #[error("prefill dispatch {request_id:?} omitted progress")]
    PrefillProgressMissing { request_id: RequestId },
    #[error(
        "prefill dispatch {request_id:?} progressed {progress} tokens with a budget of {budget}"
    )]
    PrefillProgressOverBudget {
        request_id: RequestId,
        progress: u32,
        budget: u32,
    },
    #[error("decode dispatch {request_id:?} returned prefill progress")]
    DecodePrefillProgress { request_id: RequestId },
    #[error("decode dispatch {request_id:?} made no progress")]
    NoProgress { request_id: RequestId },
}

/// The committed result of one scheduled executor quantum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceQuantum {
    pub completion: QuantumCompletion,
    pub kind: DispatchKind,
    pub progress_tokens: u32,
    pub tokens: Vec<u32>,
}

/// Connects bounded admission and scheduling to a quantum executor.
#[derive(Debug)]
pub struct ScheduledService<E: QuantumExecutor> {
    scheduler: ContinuousScheduler,
    executor: E,
    outputs: BTreeMap<RequestId, Vec<u32>>,
}

impl<E: QuantumExecutor> ScheduledService<E> {
    pub fn new(policy: SchedulerPolicy, executor: E) -> Result<Self, SchedulerError> {
        Ok(Self {
            scheduler: ContinuousScheduler::new(policy)?,
            executor,
            outputs: BTreeMap::new(),
        })
    }

    /// Admits a request and creates executor state only after admission passes.
    pub fn admit(
        &mut self,
        spec: RequestSpec,
        request: &E::Request,
        now_ns: u64,
    ) -> Result<AdmissionOutcome, ServiceError<E::Error>> {
        let request_id = spec.id;
        let outcome = self.scheduler.admit(spec, now_ns)?;
        if matches!(outcome, AdmissionOutcome::Admitted { .. }) {
            if let Err(error) = self.executor.begin(request_id, request) {
                // Admission reserved KV. Release it if the executor cannot start.
                self.scheduler.cancel(request_id)?;
                return Err(ServiceError::Executor(error));
            }
            self.outputs.insert(request_id, Vec::new());
        }
        Ok(outcome)
    }

    /// Executes one scheduler-selected quantum.
    pub fn tick(&mut self, now_ns: u64) -> Result<Option<ServiceQuantum>, ServiceError<E::Error>> {
        Ok(self.tick_batch_with_limit(now_ns, 1)?.into_iter().next())
    }

    /// Executes one scheduler-selected request batch.
    pub fn tick_batch(
        &mut self,
        now_ns: u64,
    ) -> Result<Vec<ServiceQuantum>, ServiceError<E::Error>> {
        let limit = self.scheduler.policy().max_batch_requests;
        self.tick_batch_with_limit(now_ns, limit)
    }

    fn tick_batch_with_limit(
        &mut self,
        now_ns: u64,
        limit: u32,
    ) -> Result<Vec<ServiceQuantum>, ServiceError<E::Error>> {
        let dispatches = self.scheduler.dispatch_batch(now_ns, limit)?;
        if dispatches.is_empty() {
            return Ok(Vec::new());
        }
        let outputs = self.execute_batch_outputs(&dispatches)?;
        self.commit_batch(dispatches, outputs)
    }

    fn execute_batch_outputs(
        &mut self,
        dispatches: &[crate::scheduler::Dispatch],
    ) -> Result<Vec<QuantumOutput>, ServiceError<E::Error>> {
        let outputs = match self.executor.execute_batch(dispatches) {
            Ok(outputs) => outputs,
            Err(error) => {
                return self.abort_with_error(dispatches, ServiceError::Executor(error));
            }
        };
        if outputs.len() != dispatches.len() {
            return self.abort_with_error(
                dispatches,
                ServiceError::ExecutorBatchSize {
                    expected: dispatches.len(),
                    actual: outputs.len(),
                },
            );
        }
        if let Some((emitted, budget)) = over_budget_output(dispatches, &outputs) {
            return self
                .abort_with_error(dispatches, ServiceError::ExecutorBudget { emitted, budget });
        }
        if let Some(error) = invalid_output_phase(dispatches, &outputs) {
            return self.abort_with_error(dispatches, error);
        }
        Ok(outputs)
    }

    fn abort_with_error<T>(
        &mut self,
        dispatches: &[crate::scheduler::Dispatch],
        error: ServiceError<E::Error>,
    ) -> Result<T, ServiceError<E::Error>> {
        self.abort_batch(dispatches)?;
        Err(error)
    }

    fn abort_batch(
        &mut self,
        dispatches: &[crate::scheduler::Dispatch],
    ) -> Result<(), ServiceError<E::Error>> {
        let mut first_error = None;
        for dispatch in dispatches {
            let status = self.scheduler.abort_quantum(dispatch.request_id)?;
            if let Err(error) = self.executor.finish(dispatch.request_id, status) {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(ServiceError::Executor(error)),
            None => Ok(()),
        }
    }

    fn commit_batch(
        &mut self,
        dispatches: Vec<crate::scheduler::Dispatch>,
        outputs: Vec<QuantumOutput>,
    ) -> Result<Vec<ServiceQuantum>, ServiceError<E::Error>> {
        let mut committed = Vec::with_capacity(dispatches.len());
        for (index, (dispatch, output)) in dispatches.iter().copied().zip(outputs).enumerate() {
            match self.commit_quantum(dispatch, output) {
                Ok(quantum) => committed.push(quantum),
                Err(error) => {
                    self.abort_running_batch(&dispatches[index..]);
                    return Err(error);
                }
            }
        }
        Ok(committed)
    }

    fn abort_running_batch(&mut self, dispatches: &[crate::scheduler::Dispatch]) {
        for dispatch in dispatches {
            if self.scheduler.status(dispatch.request_id) != Some(RequestStatus::Running) {
                continue;
            }
            let Ok(status) = self.scheduler.abort_quantum(dispatch.request_id) else {
                continue;
            };
            let _ = self.executor.finish(dispatch.request_id, status);
        }
    }

    fn commit_quantum(
        &mut self,
        dispatch: crate::scheduler::Dispatch,
        output: QuantumOutput,
    ) -> Result<ServiceQuantum, ServiceError<E::Error>> {
        if output.tokens.len() > usize::try_from(dispatch.token_budget).unwrap_or(usize::MAX) {
            return self.reject_quantum(
                dispatch,
                ServiceError::ExecutorBudget {
                    emitted: output.tokens.len(),
                    budget: dispatch.token_budget,
                },
            );
        }
        if let Some(error) = invalid_output_for_dispatch(dispatch, &output) {
            return self.reject_quantum(dispatch, error);
        }
        if output.cancelled {
            self.scheduler.cancel(dispatch.request_id)?;
        }
        let emitted_tokens = u32::try_from(output.tokens.len())
            .expect("an output within a u32 token budget fits u32");
        let progress_tokens = quantum_progress(dispatch, &output);
        let prefill_ready = output.prefill.is_some_and(|progress| progress.ready);
        let completion = self.scheduler.complete_quantum_with_progress(
            dispatch.request_id,
            emitted_tokens,
            progress_tokens,
            prefill_ready,
            output.eos,
        )?;
        self.outputs
            .get_mut(&dispatch.request_id)
            .expect("an admitted request has an output")
            .extend_from_slice(&output.tokens);
        if completion.status.is_terminal() {
            self.executor
                .finish(dispatch.request_id, completion.status)
                .map_err(ServiceError::Executor)?;
        }
        Ok(ServiceQuantum {
            completion,
            kind: dispatch.kind,
            progress_tokens: quantum_progress(dispatch, &output),
            tokens: output.tokens,
        })
    }

    fn reject_quantum<T>(
        &mut self,
        dispatch: crate::scheduler::Dispatch,
        error: ServiceError<E::Error>,
    ) -> Result<T, ServiceError<E::Error>> {
        let status = self.scheduler.abort_quantum(dispatch.request_id)?;
        self.executor
            .finish(dispatch.request_id, status)
            .map_err(ServiceError::Executor)?;
        Err(error)
    }

    /// Cancels a request.
    ///
    /// A queued request releases executor state immediately. A running request
    /// stops at the next quantum boundary.
    pub fn cancel(
        &mut self,
        request_id: RequestId,
    ) -> Result<RequestStatus, ServiceError<E::Error>> {
        let before = self
            .scheduler
            .status(request_id)
            .ok_or(SchedulerError::UnknownRequest(request_id))?;
        let status = self.scheduler.cancel(request_id)?;
        if before == RequestStatus::Queued && status == RequestStatus::Cancelled {
            self.executor
                .finish(request_id, status)
                .map_err(ServiceError::Executor)?;
        }
        Ok(status)
    }

    /// Removes terminal scheduler records and their retained output tokens.
    ///
    /// Callers must consume terminal results before pruning.
    pub fn prune_terminal(&mut self) {
        self.outputs.retain(|id, _| {
            self.scheduler
                .status(*id)
                .is_some_and(|status| !status.is_terminal())
        });
        self.scheduler.prune_terminal();
    }

    pub fn output(&self, request_id: RequestId) -> Option<&[u32]> {
        self.outputs.get(&request_id).map(Vec::as_slice)
    }

    pub fn status(&self, request_id: RequestId) -> Option<RequestStatus> {
        self.scheduler.status(request_id)
    }

    pub fn has_runnable_requests(&self) -> bool {
        self.scheduler.has_runnable_requests()
    }

    pub fn scheduler(&self) -> &ContinuousScheduler {
        &self.scheduler
    }

    pub fn executor(&self) -> &E {
        &self.executor
    }

    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }
}

fn over_budget_output(
    dispatches: &[crate::scheduler::Dispatch],
    outputs: &[QuantumOutput],
) -> Option<(usize, u32)> {
    dispatches
        .iter()
        .zip(outputs)
        .find(|(dispatch, output)| {
            output.tokens.len() > usize::try_from(dispatch.token_budget).unwrap_or(usize::MAX)
        })
        .map(|(dispatch, output)| (output.tokens.len(), dispatch.token_budget))
}

fn invalid_output_phase<E: Error + Send + Sync + 'static>(
    dispatches: &[Dispatch],
    outputs: &[QuantumOutput],
) -> Option<ServiceError<E>> {
    dispatches
        .iter()
        .zip(outputs)
        .find_map(|(dispatch, output)| invalid_output_for_dispatch(*dispatch, output))
}

fn invalid_output_for_dispatch<E: Error + Send + Sync + 'static>(
    dispatch: Dispatch,
    output: &QuantumOutput,
) -> Option<ServiceError<E>> {
    match dispatch.kind {
        DispatchKind::Prefill => invalid_prefill_output(dispatch, output),
        DispatchKind::Decode => invalid_decode_output(dispatch, output),
    }
}

fn invalid_prefill_output<E: Error + Send + Sync + 'static>(
    dispatch: Dispatch,
    output: &QuantumOutput,
) -> Option<ServiceError<E>> {
    let Some(progress) = output.prefill else {
        return Some(ServiceError::PrefillProgressMissing {
            request_id: dispatch.request_id,
        });
    };
    if !output.tokens.is_empty() {
        return Some(ServiceError::PrefillOutput {
            request_id: dispatch.request_id,
            emitted: output.tokens.len(),
        });
    }
    if progress.processed_tokens > dispatch.token_budget {
        return Some(ServiceError::PrefillProgressOverBudget {
            request_id: dispatch.request_id,
            progress: progress.processed_tokens,
            budget: dispatch.token_budget,
        });
    }
    if progress.processed_tokens == 0 && !progress.ready && !output.cancelled {
        return Some(ServiceError::PrefillNoProgress {
            request_id: dispatch.request_id,
        });
    }
    None
}

fn invalid_decode_output<E: Error + Send + Sync + 'static>(
    dispatch: Dispatch,
    output: &QuantumOutput,
) -> Option<ServiceError<E>> {
    if output.prefill.is_some() {
        return Some(ServiceError::DecodePrefillProgress {
            request_id: dispatch.request_id,
        });
    }
    if output.tokens.is_empty() && !output.eos && !output.cancelled {
        return Some(ServiceError::NoProgress {
            request_id: dispatch.request_id,
        });
    }
    None
}

fn quantum_progress(dispatch: Dispatch, output: &QuantumOutput) -> u32 {
    match dispatch.kind {
        DispatchKind::Prefill => output
            .prefill
            .map_or(0, |progress| progress.processed_tokens),
        DispatchKind::Decode => u32::try_from(output.tokens.len()).unwrap_or(u32::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    #[test]
    fn terminal_pruning_preserves_live_requests_and_statistics() {
        let executor = IsolatedExecutor::new(BTreeMap::from([(RequestId(1), vec![1])]));
        let mut service = ScheduledService::new(policy(), executor).unwrap();
        service.admit(spec(1), &RequestId(1), 0).unwrap();
        service.prune_terminal();
        assert!(service.status(RequestId(1)).is_some());
        while service.has_runnable_requests() {
            service.tick(0).unwrap();
        }
        let stats = service.scheduler().stats();
        service.prune_terminal();
        assert!(service.status(RequestId(1)).is_none());
        assert!(service.output(RequestId(1)).is_none());
        assert_eq!(service.scheduler().stats(), stats);
    }

    #[test]
    fn scheduled_executor_matches_every_isolated_stream() {
        let streams = BTreeMap::from([
            (RequestId(1), vec![1, 2, 3, 4, 5, 6, 7]),
            (RequestId(2), vec![10, 11, 12, 13, 14]),
            (RequestId(3), vec![20, 21, 22, 23, 24, 25]),
        ]);
        let executor = IsolatedExecutor::new(streams.clone());
        let mut service = ScheduledService::new(policy(), executor).unwrap();
        for id in 1..=3 {
            service.admit(spec(id), &RequestId(id), 0).unwrap();
        }
        let mut now = 0;
        while service.has_runnable_requests() {
            service.tick(now).unwrap().unwrap();
            now += 1;
        }
        for (id, expected) in streams {
            assert_eq!(service.status(id), Some(RequestStatus::Finished));
            assert_eq!(service.output(id), Some(expected.as_slice()));
        }
        assert_eq!(service.executor().finished.len(), 3);
    }

    #[test]
    fn one_tick_commits_a_selected_request_batch() {
        let streams = BTreeMap::from([
            (RequestId(1), vec![1, 2]),
            (RequestId(2), vec![3, 4]),
            (RequestId(3), vec![5, 6]),
        ]);
        let executor = IsolatedExecutor::new(streams);
        let mut service = ScheduledService::new(policy(), executor).unwrap();
        for id in 1..=3 {
            service.admit(spec(id), &RequestId(id), 0).unwrap();
        }
        let completed = service.tick_batch(0).unwrap();
        assert_eq!(completed.len(), 3);
        assert_eq!(service.scheduler().stats().dispatched_batches, 1);
        assert_eq!(service.scheduler().stats().max_batch_width, 3);
    }

    #[test]
    fn executor_state_is_not_created_for_overload() {
        let mut selected = policy();
        selected.max_active_requests = 1;
        let executor = IsolatedExecutor::new(BTreeMap::from([
            (RequestId(1), vec![1]),
            (RequestId(2), vec![2]),
        ]));
        let mut service = ScheduledService::new(selected, executor).unwrap();
        assert!(matches!(
            service.admit(spec(1), &RequestId(1), 0).unwrap(),
            AdmissionOutcome::Admitted { .. }
        ));
        assert!(matches!(
            service.admit(spec(2), &RequestId(2), 0).unwrap(),
            AdmissionOutcome::Rejected { .. }
        ));
        assert_eq!(service.executor().positions.len(), 1);
    }

    #[test]
    fn executor_cancellation_stops_at_the_quantum_boundary() {
        let executor =
            IsolatedExecutor::new(BTreeMap::from([(RequestId(1), vec![1, 2, 3, 4, 5, 6])]));
        let mut service = ScheduledService::new(policy(), executor).unwrap();
        let mut request = spec(1);
        request.prefix_reused_tokens = request.prompt_tokens;
        service.admit(request, &RequestId(1), 0).unwrap();
        service.executor.cancel_next = Some(RequestId(1));
        let quantum = service.tick(0).unwrap().unwrap();
        assert_eq!(quantum.completion.status, RequestStatus::Cancelled);
        assert_eq!(quantum.tokens.len(), 2);
        assert_eq!(service.scheduler().reserved_kv_bytes(), 0);
    }

    #[test]
    fn malformed_batch_aborts_every_selected_request() {
        for fault in [BatchFault::MissingOutput, BatchFault::OverBudget] {
            let streams =
                BTreeMap::from([(RequestId(1), vec![1, 2, 3]), (RequestId(2), vec![4, 5, 6])]);
            let mut executor = IsolatedExecutor::new(streams);
            executor.batch_fault = Some(fault);
            let mut service = ScheduledService::new(policy(), executor).unwrap();
            for id in 1..=2 {
                service.admit(spec(id), &RequestId(id), 0).unwrap();
            }

            assert!(service.tick_batch(0).is_err());
            assert_eq!(service.status(RequestId(1)), Some(RequestStatus::Cancelled));
            assert_eq!(service.status(RequestId(2)), Some(RequestStatus::Cancelled));
            assert_eq!(service.scheduler().reserved_kv_bytes(), 0);
        }
    }

    #[test]
    fn prefill_progress_interleaves_with_decode_and_stays_nonterminal() {
        let executor = PrefillExecutor::new();
        let mut service = ScheduledService::new(policy(), executor).unwrap();
        service
            .admit(
                RequestSpec {
                    id: RequestId(1),
                    arrival_ns: 0,
                    prompt_tokens: 0,
                    prefix_reused_tokens: 0,
                    max_output_tokens: 4,
                    priority: 1,
                    deadline_ns: None,
                },
                &PrefillRequest::new(0, vec![1, 2, 3, 4]),
                0,
            )
            .unwrap();
        service
            .admit(
                RequestSpec {
                    id: RequestId(2),
                    arrival_ns: 0,
                    prompt_tokens: 6,
                    prefix_reused_tokens: 0,
                    max_output_tokens: 2,
                    priority: 1,
                    deadline_ns: None,
                },
                &PrefillRequest::new(6, vec![9, 10]),
                0,
            )
            .unwrap();

        let first = service.tick_batch(0).unwrap();
        assert_eq!(first.len(), 2);
        assert!(first.iter().any(|quantum| {
            quantum.completion.request_id == RequestId(2)
                && quantum.kind == DispatchKind::Prefill
                && quantum.tokens.is_empty()
                && quantum.progress_tokens == 4
                && quantum.completion.status == RequestStatus::Queued
        }));
        assert!(first.iter().any(|quantum| {
            quantum.completion.request_id == RequestId(1) && quantum.kind == DispatchKind::Decode
        }));

        let second = service.tick_batch(1).unwrap();
        assert!(second.iter().any(|quantum| {
            quantum.completion.request_id == RequestId(2)
                && quantum.kind == DispatchKind::Prefill
                && quantum.progress_tokens == 2
        }));
        assert_eq!(service.scheduler().stats().prefill_chunks, 2);
        assert_eq!(service.scheduler().stats().prefill_tokens, 6);
    }

    #[test]
    fn malformed_prefill_transition_cancels_the_selected_request() {
        let mut executor = PrefillExecutor::new();
        executor.malformed = true;
        let mut service = ScheduledService::new(policy(), executor).unwrap();
        service
            .admit(
                RequestSpec {
                    id: RequestId(1),
                    arrival_ns: 0,
                    prompt_tokens: 2,
                    prefix_reused_tokens: 0,
                    max_output_tokens: 2,
                    priority: 1,
                    deadline_ns: None,
                },
                &PrefillRequest::new(2, vec![1, 2]),
                0,
            )
            .unwrap();
        assert!(matches!(
            service.tick(0),
            Err(ServiceError::PrefillOutput {
                request_id: RequestId(1),
                ..
            })
        ));
        assert_eq!(service.status(RequestId(1)), Some(RequestStatus::Cancelled));
        assert_eq!(service.scheduler().reserved_kv_bytes(), 0);
    }

    #[test]
    fn prefill_cancellation_releases_reservation_without_output() {
        let mut executor = PrefillExecutor::new();
        executor.cancel_next = Some(RequestId(1));
        let mut service = ScheduledService::new(policy(), executor).unwrap();
        service
            .admit(
                RequestSpec {
                    id: RequestId(1),
                    arrival_ns: 0,
                    prompt_tokens: 4,
                    prefix_reused_tokens: 0,
                    max_output_tokens: 2,
                    priority: 1,
                    deadline_ns: None,
                },
                &PrefillRequest::new(4, vec![1, 2]),
                0,
            )
            .unwrap();
        let quantum = service.tick(0).unwrap().unwrap();
        assert_eq!(quantum.completion.status, RequestStatus::Cancelled);
        assert!(quantum.tokens.is_empty());
        assert_eq!(service.scheduler().reserved_kv_bytes(), 0);
    }

    #[derive(Debug)]
    struct IsolatedError;

    impl fmt::Display for IsolatedError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("isolated executor failed")
        }
    }

    impl Error for IsolatedError {}

    #[derive(Debug, Clone, Copy)]
    enum BatchFault {
        MissingOutput,
        OverBudget,
    }

    #[derive(Debug)]
    struct IsolatedExecutor {
        streams: BTreeMap<RequestId, Vec<u32>>,
        positions: BTreeMap<RequestId, usize>,
        finished: BTreeMap<RequestId, RequestStatus>,
        cancel_next: Option<RequestId>,
        batch_fault: Option<BatchFault>,
    }

    #[derive(Debug, Clone)]
    struct PrefillRequest {
        prompt_tokens: u32,
        output: Vec<u32>,
    }

    impl PrefillRequest {
        fn new(prompt_tokens: u32, output: Vec<u32>) -> Self {
            Self {
                prompt_tokens,
                output,
            }
        }
    }

    #[derive(Debug)]
    struct PrefillExecutor {
        output: BTreeMap<RequestId, Vec<u32>>,
        positions: BTreeMap<RequestId, usize>,
        pending: BTreeMap<RequestId, u32>,
        cancel_next: Option<RequestId>,
        malformed: bool,
    }

    impl PrefillExecutor {
        fn new() -> Self {
            Self {
                output: BTreeMap::new(),
                positions: BTreeMap::new(),
                pending: BTreeMap::new(),
                cancel_next: None,
                malformed: false,
            }
        }
    }

    impl QuantumExecutor for PrefillExecutor {
        type Request = PrefillRequest;
        type Error = IsolatedError;

        fn begin(
            &mut self,
            request_id: RequestId,
            request: &Self::Request,
        ) -> Result<(), Self::Error> {
            self.output.insert(request_id, request.output.clone());
            self.positions.insert(request_id, 0);
            self.pending.insert(request_id, request.prompt_tokens);
            Ok(())
        }

        fn execute(
            &mut self,
            request_id: RequestId,
            token_budget: u32,
        ) -> Result<QuantumOutput, Self::Error> {
            let output = self.output.get(&request_id).ok_or(IsolatedError)?;
            let position = self.positions.get_mut(&request_id).ok_or(IsolatedError)?;
            let budget = usize::try_from(token_budget).unwrap_or(usize::MAX);
            let count = budget.min(output.len().saturating_sub(*position));
            let tokens = output[*position..*position + count].to_vec();
            *position += count;
            Ok(QuantumOutput::decode(
                tokens,
                *position == output.len(),
                false,
            ))
        }

        fn execute_dispatch(
            &mut self,
            dispatch: &crate::scheduler::Dispatch,
        ) -> Result<QuantumOutput, Self::Error> {
            if dispatch.kind == DispatchKind::Prefill {
                let pending = self
                    .pending
                    .get_mut(&dispatch.request_id)
                    .ok_or(IsolatedError)?;
                let processed = (*pending).min(dispatch.token_budget);
                *pending -= processed;
                let cancelled = self.cancel_next == Some(dispatch.request_id);
                self.cancel_next = None;
                if self.malformed {
                    return Ok(QuantumOutput {
                        tokens: vec![u32::MAX],
                        eos: false,
                        cancelled: false,
                        prefill: Some(PrefillProgress {
                            processed_tokens: dispatch.token_budget,
                            ready: false,
                        }),
                    });
                }
                return Ok(QuantumOutput::prefill(processed, *pending == 0, cancelled));
            }
            self.execute(dispatch.request_id, dispatch.token_budget)
        }

        fn finish(
            &mut self,
            request_id: RequestId,
            _status: RequestStatus,
        ) -> Result<(), Self::Error> {
            self.pending.remove(&request_id);
            Ok(())
        }
    }

    impl IsolatedExecutor {
        fn new(streams: BTreeMap<RequestId, Vec<u32>>) -> Self {
            Self {
                streams,
                positions: BTreeMap::new(),
                finished: BTreeMap::new(),
                cancel_next: None,
                batch_fault: None,
            }
        }
    }

    impl QuantumExecutor for IsolatedExecutor {
        type Request = RequestId;
        type Error = IsolatedError;

        fn begin(
            &mut self,
            request_id: RequestId,
            request: &Self::Request,
        ) -> Result<(), Self::Error> {
            if request_id != *request || !self.streams.contains_key(&request_id) {
                return Err(IsolatedError);
            }
            self.positions.insert(request_id, 0);
            Ok(())
        }

        fn execute(
            &mut self,
            request_id: RequestId,
            token_budget: u32,
        ) -> Result<QuantumOutput, Self::Error> {
            let stream = self.streams.get(&request_id).ok_or(IsolatedError)?;
            let position = self.positions.get_mut(&request_id).ok_or(IsolatedError)?;
            let count = usize::try_from(token_budget)
                .unwrap_or(usize::MAX)
                .min(stream.len() - *position);
            let tokens = stream[*position..*position + count].to_vec();
            *position += count;
            let cancelled = self.cancel_next == Some(request_id);
            self.cancel_next = None;
            Ok(QuantumOutput {
                tokens,
                eos: *position == stream.len(),
                cancelled,
                prefill: None,
            })
        }

        fn execute_dispatch(
            &mut self,
            dispatch: &crate::scheduler::Dispatch,
        ) -> Result<QuantumOutput, Self::Error> {
            if dispatch.kind == DispatchKind::Prefill {
                let cancelled = self.cancel_next == Some(dispatch.request_id);
                self.cancel_next = None;
                return Ok(QuantumOutput::prefill(
                    dispatch.token_budget,
                    true,
                    cancelled,
                ));
            }
            self.execute(dispatch.request_id, dispatch.token_budget)
        }

        fn execute_batch(
            &mut self,
            dispatches: &[crate::scheduler::Dispatch],
        ) -> Result<Vec<QuantumOutput>, Self::Error> {
            let mut outputs = Vec::with_capacity(dispatches.len());
            for dispatch in dispatches {
                outputs.push(self.execute_dispatch(dispatch)?);
            }
            match self.batch_fault {
                Some(BatchFault::MissingOutput) => {
                    outputs.pop();
                }
                Some(BatchFault::OverBudget) => {
                    if let Some(output) = outputs.first_mut() {
                        output.tokens.push(u32::MAX);
                    }
                }
                None => {}
            }
            Ok(outputs)
        }

        fn finish(
            &mut self,
            request_id: RequestId,
            status: RequestStatus,
        ) -> Result<(), Self::Error> {
            self.finished.insert(request_id, status);
            Ok(())
        }
    }

    fn policy() -> SchedulerPolicy {
        SchedulerPolicy {
            max_active_requests: 4,
            max_queued_requests: 4,
            max_batch_requests: 4,
            max_reserved_kv_bytes: 1 << 20,
            kv_bytes_per_token: 64,
            kv_page_tokens: 16,
            max_prompt_tokens: 1_024,
            max_output_tokens: 64,
            service_quantum_tokens: 2,
            prefill_chunk_tokens: 4,
            urgent_window_ns: 0,
            max_prefix_credit_tokens: 64,
        }
    }

    fn spec(id: u64) -> RequestSpec {
        RequestSpec {
            id: RequestId(id),
            arrival_ns: 0,
            prompt_tokens: 8,
            prefix_reused_tokens: 0,
            max_output_tokens: 16,
            priority: 1,
            deadline_ns: None,
        }
    }
}
