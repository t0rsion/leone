//! Training-free proposals and measured control for exact speculative decode.

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;
use thiserror::Error;

const WIDTH_SLOTS: usize = 9;

/// An invalid adaptive drafting policy or observation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AdaptiveError {
    #[error("maximum proposal tokens must be in [1, 7]")]
    ProposalWidth,
    #[error("minimum suffix tokens must not exceed maximum suffix tokens")]
    SuffixRange,
    #[error("minimum speedup must be finite and greater than one")]
    MinimumSpeedup,
    #[error("maximum regret fraction must be finite and in [0, 1)")]
    MaximumRegret,
    #[error("EWMA weight must be finite and in (0, 1]")]
    EwmaWeight,
    #[error("verifier positions must be in [1, 8]")]
    VerifierPositions,
    #[error("adaptive accounting overflowed")]
    CountOverflow,
}

/// Selects the token-history method that produced a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalSource {
    /// Reuse the continuation after the longest repeated suffix.
    Suffix,
    /// Reuse the most frequent prior continuation after the current token.
    TokenRecycling,
}

/// A point-mass proposal that the target model must verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveProposal {
    /// Proposed tokens in decode order.
    pub tokens: Vec<u32>,
    /// History method that produced `tokens`.
    pub source: ProposalSource,
}

/// Configures training-free proposal search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveDrafterConfig {
    /// Maximum number of proposed tokens in one decode round.
    max_proposal_tokens: NonZeroUsize,
    /// Minimum repeated suffix length, in tokens.
    minimum_suffix_tokens: NonZeroUsize,
    /// Maximum suffix length examined, in tokens.
    maximum_suffix_tokens: NonZeroUsize,
}

impl AdaptiveDrafterConfig {
    /// Creates one checked training-free drafting policy.
    pub fn new(
        max_proposal_tokens: NonZeroUsize,
        minimum_suffix_tokens: NonZeroUsize,
        maximum_suffix_tokens: NonZeroUsize,
    ) -> Result<Self, AdaptiveError> {
        if max_proposal_tokens.get() > WIDTH_SLOTS - 2 {
            return Err(AdaptiveError::ProposalWidth);
        }
        if minimum_suffix_tokens > maximum_suffix_tokens {
            return Err(AdaptiveError::SuffixRange);
        }
        Ok(Self {
            max_proposal_tokens,
            minimum_suffix_tokens,
            maximum_suffix_tokens,
        })
    }
}

impl Default for AdaptiveDrafterConfig {
    fn default() -> Self {
        Self::new(
            NonZeroUsize::new(7).expect("seven is nonzero"),
            NonZeroUsize::new(4).expect("four is nonzero"),
            NonZeroUsize::new(64).expect("64 is nonzero"),
        )
        .expect("the default drafting policy is valid")
    }
}

/// Produces suffix and token-recycling proposals from accepted history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveDrafter {
    config: AdaptiveDrafterConfig,
}

impl AdaptiveDrafter {
    /// Creates a drafter with bounded proposal and suffix lengths.
    pub fn new(config: AdaptiveDrafterConfig) -> Self {
        Self { config }
    }

    /// Returns the maximum verifier positions required by this drafter.
    pub fn maximum_verifier_positions(&self) -> usize {
        self.config.max_proposal_tokens.get() + 1
    }

    /// Produces one proposal. Suffix matching takes priority over recycling.
    pub fn propose(&self, history: &[u32]) -> Option<AdaptiveProposal> {
        self.suffix_proposal(history)
            .or_else(|| self.recycling_proposal(history))
    }

    fn suffix_proposal(&self, history: &[u32]) -> Option<AdaptiveProposal> {
        let minimum = self.config.minimum_suffix_tokens.get();
        if history.len() <= minimum {
            return None;
        }

        let maximum = self
            .config
            .maximum_suffix_tokens
            .get()
            .min(history.len() - 1);
        for suffix_len in (minimum..=maximum).rev() {
            let suffix_start = history.len() - suffix_len;
            let suffix = &history[suffix_start..];
            for prior_end in (suffix_len..suffix_start).rev() {
                if &history[prior_end - suffix_len..prior_end] != suffix {
                    continue;
                }

                let proposal_end =
                    (prior_end + self.config.max_proposal_tokens.get()).min(history.len());
                if proposal_end == prior_end {
                    continue;
                }
                return Some(AdaptiveProposal {
                    tokens: history[prior_end..proposal_end].to_vec(),
                    source: ProposalSource::Suffix,
                });
            }
        }
        None
    }

    fn recycling_proposal(&self, history: &[u32]) -> Option<AdaptiveProposal> {
        let (&current, prefix) = history.split_last()?;
        let history_end = history.len() - 1;
        let mut tokens = Vec::with_capacity(self.config.max_proposal_tokens.get());

        while tokens.len() < self.config.max_proposal_tokens.get() {
            let mut votes: BTreeMap<u32, (u64, usize)> = BTreeMap::new();
            for (position, &token) in prefix.iter().enumerate() {
                if token != current {
                    continue;
                }
                let continuation = position + 1;
                let next = continuation + tokens.len();
                if next >= history_end || history[continuation..next] != tokens {
                    continue;
                }
                let entry = votes.entry(history[next]).or_insert((0, position));
                entry.0 += 1;
                entry.1 = entry.1.max(position);
            }
            let Some(token) = votes
                .into_iter()
                .max_by(|(left_token, left_score), (right_token, right_score)| {
                    left_score
                        .0
                        .cmp(&right_score.0)
                        .then_with(|| left_score.1.cmp(&right_score.1))
                        .then_with(|| right_token.cmp(left_token))
                })
                .map(|(token, _)| token)
            else {
                break;
            };
            tokens.push(token);
        }

        if tokens.is_empty() {
            return None;
        }

        Some(AdaptiveProposal {
            tokens,
            source: ProposalSource::TokenRecycling,
        })
    }
}

/// Configures the measured adaptive controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveControllerConfig {
    /// Plain rounds measured before the first speculative probe.
    plain_warmup_rounds: NonZeroU64,
    /// Minimum measured speedup required to keep speculating.
    minimum_speedup: f64,
    /// Maximum cumulative speculative loss relative to plain decode.
    maximum_regret_fraction: f64,
    /// Weight assigned to the newest duration sample.
    ewma_weight: f64,
}

impl AdaptiveControllerConfig {
    /// Creates one checked measured-control policy.
    pub fn new(
        plain_warmup_rounds: NonZeroU64,
        minimum_speedup: f64,
        maximum_regret_fraction: f64,
        ewma_weight: f64,
    ) -> Result<Self, AdaptiveError> {
        if !minimum_speedup.is_finite() || minimum_speedup <= 1.0 {
            return Err(AdaptiveError::MinimumSpeedup);
        }
        if !maximum_regret_fraction.is_finite() || !(0.0..1.0).contains(&maximum_regret_fraction) {
            return Err(AdaptiveError::MaximumRegret);
        }
        if !ewma_weight.is_finite() || ewma_weight <= 0.0 || ewma_weight > 1.0 {
            return Err(AdaptiveError::EwmaWeight);
        }
        Ok(Self {
            plain_warmup_rounds,
            minimum_speedup,
            maximum_regret_fraction,
            ewma_weight,
        })
    }
}

impl Default for AdaptiveControllerConfig {
    fn default() -> Self {
        Self::new(
            NonZeroU64::new(8).expect("eight is nonzero"),
            1.10,
            0.03,
            0.25,
        )
        .expect("the default controller policy is valid")
    }
}

/// Explains why the controller selected plain or speculative decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveReason {
    /// The token history produced no proposal.
    NoProposal,
    /// The controller is measuring its plain-decode baseline.
    PlainWarmup,
    /// The controller is measuring an unobserved verifier width.
    WidthProbe,
    /// A measured width cleared the speedup gate.
    BestMeasuredWidth,
    /// No measured width cleared the speedup gate.
    BelowSpeedupGate,
    /// Cumulative loss reached the regret limit.
    RegretLimit,
}

/// One decode-mode decision for the next round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveDecision {
    /// One ordinary target-model decode step.
    Plain { reason: AdaptiveReason },
    /// Verification of `verifier_positions - 1` proposed tokens in one round.
    Speculate {
        verifier_positions: NonZeroUsize,
        reason: AdaptiveReason,
    },
}

/// One completed decode round observed by the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveObservation {
    /// Verifier positions used by the round. Plain decode uses one.
    pub verifier_positions: NonZeroUsize,
    /// Tokens committed by the round.
    pub emitted_tokens: NonZeroUsize,
    /// Complete wall time for the round.
    pub wall_duration: Duration,
    /// Time spent selecting the round mode.
    pub controller_duration: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
struct WidthMeasurement {
    rounds: u64,
    nanoseconds_per_token: f64,
}

/// Aggregate controller counters for receipts and diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdaptiveControllerStats {
    /// Completed plain rounds.
    pub plain_rounds: u64,
    /// Completed speculative rounds.
    pub speculative_rounds: u64,
    /// Decisions that stayed plain because the speedup gate failed.
    pub below_gate_rounds: u64,
    /// Decisions that stayed plain because the regret limit was reached.
    pub regret_limited_rounds: u64,
    /// Proposals produced by suffix matching.
    pub suffix_proposals: u64,
    /// Proposals produced by token recycling.
    pub recycling_proposals: u64,
    /// Total controller time.
    pub controller_duration: Duration,
    /// Completed rounds indexed by verifier position count.
    pub width_rounds: [u64; WIDTH_SLOTS],
}

/// Selects verifier widths from measured complete-round durations.
#[derive(Debug, Clone)]
pub struct AdaptiveController {
    config: AdaptiveControllerConfig,
    plain: WidthMeasurement,
    widths: [WidthMeasurement; WIDTH_SLOTS],
    cumulative_speculative_ns: f64,
    cumulative_plain_equivalent_ns: f64,
    regret_limited: bool,
    stats: AdaptiveControllerStats,
}

impl AdaptiveController {
    /// Creates an empty controller.
    pub fn new(config: AdaptiveControllerConfig) -> Self {
        Self {
            config,
            plain: WidthMeasurement::default(),
            widths: [WidthMeasurement::default(); WIDTH_SLOTS],
            cumulative_speculative_ns: 0.0,
            cumulative_plain_equivalent_ns: 0.0,
            regret_limited: false,
            stats: AdaptiveControllerStats::default(),
        }
    }

    /// Chooses a mode for a proposal of `proposal_tokens` tokens.
    pub fn decide(&mut self, proposal_tokens: usize) -> AdaptiveDecision {
        if proposal_tokens == 0 {
            return AdaptiveDecision::Plain {
                reason: AdaptiveReason::NoProposal,
            };
        }
        if self.plain.rounds < self.config.plain_warmup_rounds.get() {
            return AdaptiveDecision::Plain {
                reason: AdaptiveReason::PlainWarmup,
            };
        }
        let available_positions = (proposal_tokens + 1).min(WIDTH_SLOTS - 1);
        for width in [2, 4, 6, 8] {
            if width <= available_positions && self.widths[width].rounds == 0 {
                return AdaptiveDecision::Speculate {
                    verifier_positions: NonZeroUsize::new(width).expect("probe widths are nonzero"),
                    reason: AdaptiveReason::WidthProbe,
                };
            }
        }
        if self.regret_limited {
            self.stats.regret_limited_rounds += 1;
            return AdaptiveDecision::Plain {
                reason: AdaptiveReason::RegretLimit,
            };
        }

        let best = (2..=available_positions)
            .filter(|&width| self.widths[width].rounds != 0)
            .map(|width| {
                let speedup =
                    self.plain.nanoseconds_per_token / self.widths[width].nanoseconds_per_token;
                (width, speedup)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1));

        match best {
            Some((width, speedup)) if speedup >= self.config.minimum_speedup => {
                AdaptiveDecision::Speculate {
                    verifier_positions: NonZeroUsize::new(width)
                        .expect("measured widths are nonzero"),
                    reason: AdaptiveReason::BestMeasuredWidth,
                }
            }
            _ => {
                self.stats.below_gate_rounds += 1;
                AdaptiveDecision::Plain {
                    reason: AdaptiveReason::BelowSpeedupGate,
                }
            }
        }
    }

    /// Records one complete decode round.
    pub fn observe(&mut self, observation: AdaptiveObservation) -> Result<(), AdaptiveError> {
        let width = observation.verifier_positions.get();
        if width >= WIDTH_SLOTS {
            return Err(AdaptiveError::VerifierPositions);
        }
        let duration_ns = observation.wall_duration.as_secs_f64() * 1e9;
        let ns_per_token = duration_ns / observation.emitted_tokens.get() as f64;
        self.stats.controller_duration += observation.controller_duration;
        self.stats.width_rounds[width] = self.stats.width_rounds[width]
            .checked_add(1)
            .ok_or(AdaptiveError::CountOverflow)?;

        if width == 1 {
            Self::update_measurement(&mut self.plain, ns_per_token, self.config.ewma_weight);
            self.stats.plain_rounds = self
                .stats
                .plain_rounds
                .checked_add(1)
                .ok_or(AdaptiveError::CountOverflow)?;
            return Ok(());
        }

        Self::update_measurement(
            &mut self.widths[width],
            ns_per_token,
            self.config.ewma_weight,
        );
        self.stats.speculative_rounds = self
            .stats
            .speculative_rounds
            .checked_add(1)
            .ok_or(AdaptiveError::CountOverflow)?;
        self.cumulative_speculative_ns += duration_ns;
        self.cumulative_plain_equivalent_ns +=
            self.plain.nanoseconds_per_token * observation.emitted_tokens.get() as f64;
        self.regret_limited = self.cumulative_speculative_ns
            > self.cumulative_plain_equivalent_ns * (1.0 + self.config.maximum_regret_fraction);
        Ok(())
    }

    /// Records the proposal source selected for one round.
    pub fn record_source(&mut self, source: ProposalSource) -> Result<(), AdaptiveError> {
        match source {
            ProposalSource::Suffix => {
                self.stats.suffix_proposals = self
                    .stats
                    .suffix_proposals
                    .checked_add(1)
                    .ok_or(AdaptiveError::CountOverflow)?;
            }
            ProposalSource::TokenRecycling => {
                self.stats.recycling_proposals = self
                    .stats
                    .recycling_proposals
                    .checked_add(1)
                    .ok_or(AdaptiveError::CountOverflow)?;
            }
        }
        Ok(())
    }

    /// Returns receipt counters accumulated by this controller.
    pub fn stats(&self) -> AdaptiveControllerStats {
        self.stats
    }

    fn update_measurement(measurement: &mut WidthMeasurement, value: f64, weight: f64) {
        measurement.nanoseconds_per_token = if measurement.rounds == 0 {
            value
        } else {
            weight * value + (1.0 - weight) * measurement.nanoseconds_per_token
        };
        measurement.rounds += 1;
    }
}
