use crate::scheduler::{
    AdmissionOutcome, ContinuousScheduler, QuantumCompletion, RequestId, RequestSpec,
    RequestStatus, SchedulerError, SchedulerPolicy,
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
}

/// The committed result of one scheduled executor quantum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceQuantum {
    pub completion: QuantumCompletion,
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
        let Some(dispatch) = self.scheduler.dispatch(now_ns)? else {
            return Ok(None);
        };
        let output = match self
            .executor
            .execute(dispatch.request_id, dispatch.token_budget)
        {
            Ok(output) => output,
            Err(error) => {
                // Abort rather than complete. No tokens were committed.
                let status = self.scheduler.abort_quantum(dispatch.request_id)?;
                if let Err(cleanup) = self.executor.finish(dispatch.request_id, status) {
                    return Err(ServiceError::Executor(cleanup));
                }
                return Err(ServiceError::Executor(error));
            }
        };
        if output.tokens.len() > usize::try_from(dispatch.token_budget).unwrap_or(usize::MAX) {
            // An over-budget quantum cannot be committed.
            let status = self.scheduler.abort_quantum(dispatch.request_id)?;
            self.executor
                .finish(dispatch.request_id, status)
                .map_err(ServiceError::Executor)?;
            return Err(ServiceError::ExecutorBudget {
                emitted: output.tokens.len(),
                budget: dispatch.token_budget,
            });
        }
        if output.cancelled {
            self.scheduler.cancel(dispatch.request_id)?;
        }
        let emitted_tokens = u32::try_from(output.tokens.len())
            .expect("an output within a u32 token budget fits u32");
        let completion =
            self.scheduler
                .complete_quantum(dispatch.request_id, emitted_tokens, output.eos)?;
        self.outputs
            .get_mut(&dispatch.request_id)
            .expect("an admitted request has an output")
            .extend_from_slice(&output.tokens);
        if completion.status.is_terminal() {
            self.executor
                .finish(dispatch.request_id, completion.status)
                .map_err(ServiceError::Executor)?;
        }
        Ok(Some(ServiceQuantum {
            completion,
            tokens: output.tokens,
        }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

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
        service.admit(spec(1), &RequestId(1), 0).unwrap();
        service.executor.cancel_next = Some(RequestId(1));
        let quantum = service.tick(0).unwrap().unwrap();
        assert_eq!(quantum.completion.status, RequestStatus::Cancelled);
        assert_eq!(quantum.tokens.len(), 2);
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

    #[derive(Debug)]
    struct IsolatedExecutor {
        streams: BTreeMap<RequestId, Vec<u32>>,
        positions: BTreeMap<RequestId, usize>,
        finished: BTreeMap<RequestId, RequestStatus>,
        cancel_next: Option<RequestId>,
    }

    impl IsolatedExecutor {
        fn new(streams: BTreeMap<RequestId, Vec<u32>>) -> Self {
            Self {
                streams,
                positions: BTreeMap::new(),
                finished: BTreeMap::new(),
                cancel_next: None,
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
            })
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
            max_reserved_kv_bytes: 1 << 20,
            kv_bytes_per_token: 64,
            max_prompt_tokens: 1_024,
            max_output_tokens: 64,
            service_quantum_tokens: 2,
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
