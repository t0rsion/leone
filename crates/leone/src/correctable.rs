//! Selects distribution-valued history proposals from measured exact rounds.

use crate::{select_draft, Distribution, SamplerError, SamplerRng};
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;
use thiserror::Error;

pub const CORRECTABLE_PLAN_COUNT: usize = 4;
const WIDTH_SLOTS: usize = 9;

/// One proposal model available to the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectablePlan {
    /// Longest repeated suffix of at least four tokens.
    Suffix,
    /// Count every token in committed history.
    Unigram,
    /// Count continuations after the latest one-token context.
    Bigram,
    /// Count continuations after the latest three-token context.
    Fourgram,
}

impl CorrectablePlan {
    pub const ALL: [Self; CORRECTABLE_PLAN_COUNT] =
        [Self::Suffix, Self::Unigram, Self::Bigram, Self::Fourgram];

    pub const fn index(self) -> usize {
        match self {
            Self::Suffix => 0,
            Self::Unigram => 1,
            Self::Bigram => 2,
            Self::Fourgram => 3,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Suffix => "suffix",
            Self::Unigram => "unigram",
            Self::Bigram => "bigram",
            Self::Fourgram => "fourgram",
        }
    }

    const fn preferred_positions(self) -> usize {
        match self {
            Self::Unigram => 2,
            Self::Bigram => 4,
            Self::Suffix | Self::Fourgram => 8,
        }
    }
}

/// One sampled token and the distribution that proposed it.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectableStep {
    pub token: u32,
    pub distribution: Distribution,
}

/// One conditional proposal sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectableProposal {
    pub plan: CorrectablePlan,
    pub steps: Vec<CorrectableStep>,
}

impl CorrectableProposal {
    pub fn tokens(&self) -> Vec<u32> {
        self.steps.iter().map(|step| step.token).collect()
    }

    pub fn distributions(&self) -> Vec<Distribution> {
        self.steps
            .iter()
            .map(|step| step.distribution.clone())
            .collect()
    }
}

/// An invalid correctable proposal or controller observation.
#[derive(Debug, Error, PartialEq)]
pub enum CorrectableError {
    #[error("vocabulary size must be nonzero")]
    EmptyVocabulary,
    #[error("maximum proposal tokens must be in [1, 7]")]
    ProposalWidth,
    #[error("minimum speedup must be finite and greater than one")]
    MinimumSpeedup,
    #[error("maximum regret fraction must be finite and in [0, 1)")]
    MaximumRegret,
    #[error("EWMA weight must be finite and in (0, 1]")]
    EwmaWeight,
    #[error("verifier positions must be in [1, 8]")]
    VerifierPositions,
    #[error("correctable accounting overflowed")]
    CountOverflow,
    #[error(transparent)]
    Sampler(#[from] SamplerError),
}

/// Builds categorical proposals from committed token history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrectableDrafter {
    vocab_size: NonZeroUsize,
    maximum_proposal_tokens: NonZeroUsize,
}

impl CorrectableDrafter {
    pub fn new(
        vocab_size: usize,
        maximum_proposal_tokens: NonZeroUsize,
    ) -> Result<Self, CorrectableError> {
        let vocab_size = NonZeroUsize::new(vocab_size).ok_or(CorrectableError::EmptyVocabulary)?;
        if maximum_proposal_tokens.get() > WIDTH_SLOTS - 2 {
            return Err(CorrectableError::ProposalWidth);
        }
        Ok(Self {
            vocab_size,
            maximum_proposal_tokens,
        })
    }

    pub fn maximum_verifier_positions(&self) -> usize {
        self.maximum_proposal_tokens.get() + 1
    }

    /// Returns plans with at least one distribution for the current history.
    pub fn available_plans(&self, history: &[u32]) -> [bool; CORRECTABLE_PLAN_COUNT] {
        let mut available = [false; CORRECTABLE_PLAN_COUNT];
        for plan in CorrectablePlan::ALL {
            available[plan.index()] = self.plan_available(plan, history);
        }
        available
    }

    fn plan_available(&self, plan: CorrectablePlan, history: &[u32]) -> bool {
        match plan {
            CorrectablePlan::Unigram => history
                .iter()
                .any(|token| (*token as usize) < self.vocab_size.get()),
            CorrectablePlan::Bigram => Self::order_available(history, 1),
            CorrectablePlan::Fourgram => Self::order_available(history, 3),
            CorrectablePlan::Suffix => {
                let maximum = history.len().saturating_sub(1).min(64);
                (4..=maximum)
                    .rev()
                    .any(|order| Self::order_available(history, order))
            }
        }
    }

    fn order_available(history: &[u32], order: usize) -> bool {
        if history.len() <= order || order == 0 {
            return false;
        }
        let suffix = &history[history.len() - order..];
        (order..history.len()).any(|end| &history[end - order..end] == suffix)
    }

    /// Samples at most `tokens` conditional proposal steps.
    pub fn propose(
        &self,
        plan: CorrectablePlan,
        history: &[u32],
        tokens: usize,
        rng: SamplerRng,
        position: u64,
    ) -> Result<Option<CorrectableProposal>, CorrectableError> {
        let tokens = tokens.min(self.maximum_proposal_tokens.get());
        if tokens == 0 {
            return Ok(None);
        }
        let mut working = history.to_vec();
        let mut steps = Vec::with_capacity(tokens);
        for offset in 0..tokens {
            let Some(distribution) = self.next_distribution(plan, &working) else {
                break;
            };
            let draw_position = position
                .checked_add(offset as u64)
                .ok_or(CorrectableError::CountOverflow)?;
            let token = select_draft(&distribution, rng, draw_position)?;
            working.push(token);
            steps.push(CorrectableStep {
                token,
                distribution,
            });
        }
        Ok((!steps.is_empty()).then_some(CorrectableProposal { plan, steps }))
    }

    fn next_distribution(&self, plan: CorrectablePlan, history: &[u32]) -> Option<Distribution> {
        match plan {
            CorrectablePlan::Unigram => self.distribution_for_order(history, 0),
            CorrectablePlan::Bigram => self.distribution_for_order(history, 1),
            CorrectablePlan::Fourgram => self.distribution_for_order(history, 3),
            CorrectablePlan::Suffix => {
                let maximum = history.len().saturating_sub(1).min(64);
                (4..=maximum)
                    .rev()
                    .find_map(|order| self.distribution_for_order(history, order))
            }
        }
    }

    fn distribution_for_order(&self, history: &[u32], order: usize) -> Option<Distribution> {
        if history.is_empty() || history.len() <= order {
            return None;
        }
        let mut counts = vec![0_u64; self.vocab_size.get()];
        let mut total = 0_u64;
        if order == 0 {
            for token in history.iter().copied() {
                let count = counts.get_mut(token as usize)?;
                *count = count.checked_add(1)?;
                total = total.checked_add(1)?;
            }
        } else {
            let suffix = &history[history.len() - order..];
            for end in order..history.len() {
                if &history[end - order..end] != suffix {
                    continue;
                }
                let token = history[end];
                let count = counts.get_mut(token as usize)?;
                *count = count.checked_add(1)?;
                total = total.checked_add(1)?;
            }
        }
        if total == 0 {
            return None;
        }
        let probabilities = counts
            .into_iter()
            .map(|count| count as f64 / total as f64)
            .collect();
        Distribution::from_probabilities(probabilities).ok()
    }
}

/// Configures measured selection over correctable plans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectableControllerConfig {
    plain_warmup_rounds: NonZeroU64,
    minimum_speedup: f64,
    maximum_regret_fraction: f64,
    ewma_weight: f64,
}

impl CorrectableControllerConfig {
    pub fn new(
        plain_warmup_rounds: NonZeroU64,
        minimum_speedup: f64,
        maximum_regret_fraction: f64,
        ewma_weight: f64,
    ) -> Result<Self, CorrectableError> {
        if !minimum_speedup.is_finite() || minimum_speedup <= 1.0 {
            return Err(CorrectableError::MinimumSpeedup);
        }
        if !maximum_regret_fraction.is_finite() || !(0.0..1.0).contains(&maximum_regret_fraction) {
            return Err(CorrectableError::MaximumRegret);
        }
        if !ewma_weight.is_finite() || ewma_weight <= 0.0 || ewma_weight > 1.0 {
            return Err(CorrectableError::EwmaWeight);
        }
        Ok(Self {
            plain_warmup_rounds,
            minimum_speedup,
            maximum_regret_fraction,
            ewma_weight,
        })
    }
}

impl Default for CorrectableControllerConfig {
    fn default() -> Self {
        Self::new(
            NonZeroU64::new(8).expect("eight is nonzero"),
            1.10,
            0.03,
            0.25,
        )
        .expect("the default correctable policy is valid")
    }
}

/// Why the controller selected plain or correctable execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectableReason {
    NoProposal,
    PlainWarmup,
    PlanProbe,
    BestMeasuredPlan,
    BelowSpeedupGate,
    RegretLimit,
}

/// One action selected before a target row is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectableDecision {
    Plain {
        reason: CorrectableReason,
    },
    Speculate {
        plan: CorrectablePlan,
        verifier_positions: NonZeroUsize,
        reason: CorrectableReason,
    },
}

/// One completed action observed after exact correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectableObservation {
    pub plan: Option<CorrectablePlan>,
    pub verifier_positions: NonZeroUsize,
    pub emitted_tokens: NonZeroUsize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub overlap_sum: f64,
    pub overlap_proposals: usize,
    pub wall_duration: Duration,
    pub controller_duration: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
struct PlanMeasurement {
    rounds: u64,
    nanoseconds_per_token: f64,
}

/// Aggregate counts recorded in runtime receipt schema v9.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CorrectableControllerStats {
    pub plain_rounds: u64,
    pub speculative_rounds: u64,
    pub below_gate_rounds: u64,
    pub regret_limited_rounds: u64,
    pub plan_rounds: [u64; CORRECTABLE_PLAN_COUNT],
    pub width_rounds: [u64; WIDTH_SLOTS],
    pub proposed_tokens: u64,
    pub accepted_tokens: u64,
    pub overlap_sum: f64,
    pub overlap_proposals: u64,
    pub controller_duration: Duration,
}

/// Selects a proposal plan from measured complete-round costs.
#[derive(Debug, Clone)]
pub struct CorrectableController {
    config: CorrectableControllerConfig,
    plain: PlanMeasurement,
    plans: [PlanMeasurement; CORRECTABLE_PLAN_COUNT],
    cumulative_speculative_ns: f64,
    cumulative_plain_equivalent_ns: f64,
    regret_limited: bool,
    stats: CorrectableControllerStats,
}

impl CorrectableController {
    pub fn new(config: CorrectableControllerConfig) -> Self {
        Self {
            config,
            plain: PlanMeasurement::default(),
            plans: [PlanMeasurement::default(); CORRECTABLE_PLAN_COUNT],
            cumulative_speculative_ns: 0.0,
            cumulative_plain_equivalent_ns: 0.0,
            regret_limited: false,
            stats: CorrectableControllerStats::default(),
        }
    }

    pub fn decide(
        &mut self,
        available: [bool; CORRECTABLE_PLAN_COUNT],
        maximum_positions: usize,
    ) -> Result<CorrectableDecision, CorrectableError> {
        if !(1..WIDTH_SLOTS).contains(&maximum_positions) {
            return Err(CorrectableError::VerifierPositions);
        }
        if !available.iter().any(|value| *value) {
            return Ok(CorrectableDecision::Plain {
                reason: CorrectableReason::NoProposal,
            });
        }
        if self.plain.rounds < self.config.plain_warmup_rounds.get() {
            return Ok(CorrectableDecision::Plain {
                reason: CorrectableReason::PlainWarmup,
            });
        }
        for plan in CorrectablePlan::ALL {
            if available[plan.index()] && self.plans[plan.index()].rounds == 0 {
                return Ok(Self::speculate(
                    plan,
                    maximum_positions,
                    CorrectableReason::PlanProbe,
                ));
            }
        }
        if self.regret_limited {
            self.stats.regret_limited_rounds = self
                .stats
                .regret_limited_rounds
                .checked_add(1)
                .ok_or(CorrectableError::CountOverflow)?;
            return Ok(CorrectableDecision::Plain {
                reason: CorrectableReason::RegretLimit,
            });
        }
        let best = CorrectablePlan::ALL
            .into_iter()
            .filter(|plan| available[plan.index()] && self.plans[plan.index()].rounds != 0)
            .map(|plan| {
                let speedup = self.plain.nanoseconds_per_token
                    / self.plans[plan.index()].nanoseconds_per_token;
                (plan, speedup)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1));
        match best {
            Some((plan, speedup)) if speedup >= self.config.minimum_speedup => Ok(Self::speculate(
                plan,
                maximum_positions,
                CorrectableReason::BestMeasuredPlan,
            )),
            _ => {
                self.stats.below_gate_rounds = self
                    .stats
                    .below_gate_rounds
                    .checked_add(1)
                    .ok_or(CorrectableError::CountOverflow)?;
                Ok(CorrectableDecision::Plain {
                    reason: CorrectableReason::BelowSpeedupGate,
                })
            }
        }
    }

    pub fn observe(&mut self, observation: CorrectableObservation) -> Result<(), CorrectableError> {
        let width = observation.verifier_positions.get();
        if width >= WIDTH_SLOTS || !observation.overlap_sum.is_finite() {
            return Err(CorrectableError::VerifierPositions);
        }
        let duration_ns = observation.wall_duration.as_secs_f64() * 1e9;
        let ns_per_token = duration_ns / observation.emitted_tokens.get() as f64;
        self.stats.controller_duration += observation.controller_duration;
        self.stats.width_rounds[width] = self.stats.width_rounds[width]
            .checked_add(1)
            .ok_or(CorrectableError::CountOverflow)?;
        let Some(plan) = observation.plan else {
            Self::update(&mut self.plain, ns_per_token, self.config.ewma_weight);
            self.stats.plain_rounds = self
                .stats
                .plain_rounds
                .checked_add(1)
                .ok_or(CorrectableError::CountOverflow)?;
            return Ok(());
        };
        Self::update(
            &mut self.plans[plan.index()],
            ns_per_token,
            self.config.ewma_weight,
        );
        self.stats.speculative_rounds = self
            .stats
            .speculative_rounds
            .checked_add(1)
            .ok_or(CorrectableError::CountOverflow)?;
        self.stats.plan_rounds[plan.index()] = self.stats.plan_rounds[plan.index()]
            .checked_add(1)
            .ok_or(CorrectableError::CountOverflow)?;
        self.stats.proposed_tokens = self
            .stats
            .proposed_tokens
            .checked_add(observation.proposed_tokens as u64)
            .ok_or(CorrectableError::CountOverflow)?;
        self.stats.accepted_tokens = self
            .stats
            .accepted_tokens
            .checked_add(observation.accepted_tokens as u64)
            .ok_or(CorrectableError::CountOverflow)?;
        self.stats.overlap_sum += observation.overlap_sum;
        self.stats.overlap_proposals = self
            .stats
            .overlap_proposals
            .checked_add(observation.overlap_proposals as u64)
            .ok_or(CorrectableError::CountOverflow)?;
        self.cumulative_speculative_ns += duration_ns;
        self.cumulative_plain_equivalent_ns +=
            self.plain.nanoseconds_per_token * observation.emitted_tokens.get() as f64;
        self.regret_limited = self.cumulative_speculative_ns
            > self.cumulative_plain_equivalent_ns * (1.0 + self.config.maximum_regret_fraction);
        Ok(())
    }

    pub const fn stats(&self) -> CorrectableControllerStats {
        self.stats
    }

    fn speculate(
        plan: CorrectablePlan,
        maximum_positions: usize,
        reason: CorrectableReason,
    ) -> CorrectableDecision {
        let positions = plan.preferred_positions().min(maximum_positions).max(2);
        CorrectableDecision::Speculate {
            plan,
            verifier_positions: NonZeroUsize::new(positions).expect("positions are nonzero"),
            reason,
        }
    }

    fn update(measurement: &mut PlanMeasurement, value: f64, weight: f64) {
        measurement.nanoseconds_per_token = if measurement.rounds == 0 {
            value
        } else {
            weight * value + (1.0 - weight) * measurement.nanoseconds_per_token
        };
        measurement.rounds += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drafter() -> CorrectableDrafter {
        CorrectableDrafter::new(16, NonZeroUsize::new(7).unwrap()).unwrap()
    }

    #[test]
    fn proposal_records_the_distribution_that_sampled_each_token() {
        let history = [1, 2, 3, 1, 2, 4, 1, 2];
        let proposal = drafter()
            .propose(CorrectablePlan::Bigram, &history, 2, SamplerRng::new(7), 10)
            .unwrap()
            .unwrap();
        assert!(!proposal.steps.is_empty());
        for step in proposal.steps {
            assert!(step.distribution.probability(step.token).unwrap() > 0.0);
        }
    }

    #[test]
    fn unavailable_plan_is_never_selected() {
        let mut controller = CorrectableController::new(
            CorrectableControllerConfig::new(NonZeroU64::new(1).unwrap(), 1.1, 0.03, 0.25).unwrap(),
        );
        controller
            .observe(CorrectableObservation {
                plan: None,
                verifier_positions: NonZeroUsize::new(1).unwrap(),
                emitted_tokens: NonZeroUsize::new(1).unwrap(),
                proposed_tokens: 0,
                accepted_tokens: 0,
                overlap_sum: 0.0,
                overlap_proposals: 0,
                wall_duration: Duration::from_millis(10),
                controller_duration: Duration::ZERO,
            })
            .unwrap();
        let decision = controller.decide([false, false, true, false], 8).unwrap();
        assert!(matches!(
            decision,
            CorrectableDecision::Speculate {
                plan: CorrectablePlan::Bigram,
                ..
            }
        ));
    }
}
