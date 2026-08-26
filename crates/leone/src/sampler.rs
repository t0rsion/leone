//! Provides exact sampling and speculative verification.
//!
//! Every distribution here is computed in `f64`. Sampling costs one pass over
//! the vocabulary, which is negligible beside the weight stream a decode step
//! reads, so precision is cheaper than the risk of a wrong token.
//!
//! The speculative rule follows Leviathan et al. and Chen et al. Accept a
//! drafted token with probability `min(1, p(x) / q(x))`. On rejection, sample
//! from `normalize(max(p - q, 0))`. The emitted distribution then equals the
//! target distribution exactly, for any drafter.

use std::num::NonZeroUsize;
use thiserror::Error;

/// An invalid sampler configuration or input.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SamplerError {
    #[error("logit row is empty")]
    EmptyRow,
    #[error("logit rows differ: target has {target} entries, draft has {draft}")]
    RowMismatch { target: usize, draft: usize },
    #[error("token {token} is outside a vocabulary of {vocab}")]
    TokenOutOfRange { token: u32, vocab: usize },
    #[error("temperature must be finite and greater than zero")]
    Temperature,
    #[error("the {stage} parameter is outside its valid range")]
    Truncation { stage: &'static str },
    #[error("every logit is non-finite, so no token can be sampled")]
    AllNaN,
    #[error("probabilities must be finite, nonnegative, and have positive mass")]
    InvalidDistribution,
    #[error("the Mirostat target must be finite and greater than zero")]
    MirostatTarget,
    #[error("the Mirostat state must be finite and greater than zero")]
    MirostatState,
    #[error("the Mirostat learning rate must be finite and in (0, 1]")]
    MirostatLearningRate,
}

/// How logits are scaled before sampling.
///
/// `Greedy` is not temperature zero. It is a separate variant because dividing
/// by zero has no meaning, and because greedy needs no random draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Temperature {
    Greedy,
    Scaled(f64),
}

/// The purpose of one random draw.
///
/// The stream is keyed by purpose as well as position, so the acceptance draw
/// and the categorical draw at one position never collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Draw {
    /// The uniform value compared against an acceptance ratio.
    Accept,
    /// The uniform value that selects a token from a distribution.
    Select,
    /// The independent uniform value that samples a proposal distribution.
    Draft,
}

/// A counter-based random stream for sampling.
///
/// A draw is a pure function of the seed, the position, and the purpose.
/// A rejected draft does not shift later positions, so speculative and
/// non-speculative decode produce the same tokens for the same seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerRng {
    seed: u64,
}

impl SamplerRng {
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Returns a uniform value in `[0, 1)` for one position and purpose.
    pub const fn uniform(self, position: u64, draw: Draw) -> f64 {
        let purpose = match draw {
            Draw::Accept => 0x9e37_79b9_7f4a_7c15,
            Draw::Select => 0xbf58_476d_1ce4_e5b9,
            Draw::Draft => 0x94d0_49bb_1331_11eb,
        };
        let mixed = splitmix64(self.seed ^ splitmix64(position ^ purpose));
        // 53 bits is the full mantissa of an f64, so every representable value
        // in [0, 1) is reachable and none is favored.
        (mixed >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

/// The fixed parameters for Mirostat v2.
///
/// `target_surprise` is measured in nats. `learning_rate` is in `(0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirostatConfig {
    target_surprise: f64,
    learning_rate: f64,
}

impl MirostatConfig {
    /// Checks one Mirostat v2 controller configuration.
    pub fn new(target_surprise: f64, learning_rate: f64) -> Result<Self, SamplerError> {
        if !target_surprise.is_finite() || target_surprise <= 0.0 {
            return Err(SamplerError::MirostatTarget);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(SamplerError::MirostatLearningRate);
        }
        Ok(Self {
            target_surprise,
            learning_rate,
        })
    }

    pub const fn target_surprise(self) -> f64 {
        self.target_surprise
    }

    pub const fn learning_rate(self) -> f64 {
        self.learning_rate
    }
}

/// The evolving scalar carried by a Mirostat v2 session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirostatState {
    config: MirostatConfig,
    maximum_surprise: f64,
}

impl MirostatState {
    /// Starts Mirostat v2 at twice the target surprise.
    pub fn new(config: MirostatConfig) -> Self {
        Self {
            config,
            maximum_surprise: 2.0 * config.target_surprise,
        }
    }

    /// Restores a checked Mirostat v2 state.
    pub fn from_parts(config: MirostatConfig, maximum_surprise: f64) -> Result<Self, SamplerError> {
        if !maximum_surprise.is_finite() || maximum_surprise <= 0.0 {
            return Err(SamplerError::MirostatState);
        }
        Ok(Self {
            config,
            maximum_surprise,
        })
    }

    pub const fn config(self) -> MirostatConfig {
        self.config
    }

    pub const fn maximum_surprise(self) -> f64 {
        self.maximum_surprise
    }
}

const fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// One truncation stage in a sampling pipeline.
///
/// Every stage keeps the highest-probability token, so a pipeline can never
/// empty the candidate set. Stages run in the order the sampler lists them,
/// and each one sees the distribution the previous stage left, renormalized.
/// That rule matters for `Epsilon`, `Eta`, and `Typical`, whose thresholds are
/// absolute or entropy-based; the purely relative stages are unaffected by it.
///
/// `TopNSigma` and `MinK` run on the raw logits before temperature, which is
/// what makes them temperature-invariant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Truncation {
    /// Keeps the `k` highest-probability tokens.
    TopK(NonZeroUsize),
    /// Keeps the shortest descending prefix whose mass reaches `p`.
    ///
    /// Nucleus sampling, Holtzman et al. 2019.
    TopP(f64),
    /// Keeps tokens with probability at least `min_p` times the highest.
    ///
    /// Min-p, Nguyen et al.
    MinP(f64),
    /// Keeps tokens with probability at least `a` times the highest squared.
    ///
    /// Top-a. The square makes the cut sharper than min-p when the model is
    /// confident.
    TopA(f64),
    /// Keeps the head before the tail's second derivative flattens.
    ///
    /// Tail-free sampling. Takes first differences of the sorted
    /// probabilities, then absolute second differences, normalizes those to
    /// sum to one, and keeps the prefix whose cumulative share stays under
    /// `z`.
    TailFree(f64),
    /// Keeps the tokens whose surprise is closest to the distribution's
    /// entropy, until their mass reaches `p`.
    ///
    /// Locally typical sampling, Meister et al. Scores each token by
    /// `abs(-ln(p) - H)` and takes the lowest scores first, so it can drop the
    /// most probable token when that token is atypically certain.
    Typical(f64),
    /// Keeps tokens with probability at least `epsilon`.
    Epsilon(f64),
    /// Keeps tokens with probability at least `min(eta, sqrt(eta) * exp(-H))`.
    ///
    /// Eta sampling. The entropy term relaxes the floor when the model is
    /// uncertain.
    Eta(f64),
    /// Keeps the top `k` raw logits, with `k` found from the sharpest
    /// rank-weighted drop.
    ///
    /// Min-k, Ding et al. Runs before temperature and is exactly temperature
    /// invariant. `tau` sets the flat-row fallback and defaults to 3.0 in the
    /// reference implementation.
    MinK(f64),
    /// Keeps tokens whose raw logit is within `n` standard deviations of the
    /// highest raw logit.
    ///
    /// Top-n-sigma. Runs before temperature, so raising temperature changes
    /// how the survivors are weighted but not which tokens survive.
    TopNSigma(f64),
}

impl Truncation {
    fn validate(self) -> Result<(), SamplerError> {
        let (value, kind) = match self {
            Self::TopK(_) => return Ok(()),
            Self::TopP(value) => (value, Bound::UnitInterval),
            Self::MinP(value) => (value, Bound::UnitInterval),
            Self::TopA(value) => (value, Bound::UnitInterval),
            Self::TailFree(value) => (value, Bound::UnitInterval),
            Self::Typical(value) => (value, Bound::UnitInterval),
            Self::Epsilon(value) => (value, Bound::UnitInterval),
            Self::Eta(value) => (value, Bound::UnitInterval),
            Self::MinK(value) => (value, Bound::NonNegative),
            Self::TopNSigma(value) => (value, Bound::NonNegative),
        };
        let ok = match kind {
            Bound::UnitInterval => value.is_finite() && value > 0.0 && value <= 1.0,
            Bound::NonNegative => value.is_finite() && value >= 0.0,
        };
        if ok {
            Ok(())
        } else {
            Err(SamplerError::Truncation { stage: self.name() })
        }
    }

    /// Returns the stage name used in errors and receipts.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TopK(_) => "top-k",
            Self::TopP(_) => "top-p",
            Self::MinP(_) => "min-p",
            Self::TopA(_) => "top-a",
            Self::TailFree(_) => "tail-free",
            Self::Typical(_) => "typical",
            Self::Epsilon(_) => "epsilon",
            Self::Eta(_) => "eta",
            Self::MinK(_) => "min-k",
            Self::TopNSigma(_) => "top-n-sigma",
        }
    }
}

enum Bound {
    UnitInterval,
    NonNegative,
}

/// One sampling policy: a temperature and an ordered truncation pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampler {
    pub temperature: Temperature,
    pub truncations: Vec<Truncation>,
}

impl Sampler {
    /// Returns the policy that always takes the highest logit.
    pub const fn greedy() -> Self {
        Self {
            temperature: Temperature::Greedy,
            truncations: Vec::new(),
        }
    }

    /// Returns a policy with one temperature and no truncation.
    pub const fn temperature(value: f64) -> Self {
        Self {
            temperature: Temperature::Scaled(value),
            truncations: Vec::new(),
        }
    }

    /// Appends one truncation stage.
    pub fn then(mut self, truncation: Truncation) -> Self {
        self.truncations.push(truncation);
        self
    }

    fn validate(&self) -> Result<(), SamplerError> {
        match self.temperature {
            Temperature::Greedy => {}
            Temperature::Scaled(value) => {
                if !value.is_finite() || value <= 0.0 {
                    return Err(SamplerError::Temperature);
                }
            }
        }
        for truncation in &self.truncations {
            truncation.validate()?;
        }
        Ok(())
    }
}

/// A sampling distribution over one vocabulary.
///
/// Entries outside the eligible set are exactly zero, and the eligible entries
/// sum to one. Storing the whole row keeps the speculative residual simple.
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution {
    probabilities: Vec<f64>,
}

impl Distribution {
    /// Builds and normalizes one explicitly supplied categorical distribution.
    pub fn from_probabilities(mut probabilities: Vec<f64>) -> Result<Self, SamplerError> {
        if probabilities.is_empty()
            || probabilities
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(SamplerError::InvalidDistribution);
        }
        let total = probabilities.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(SamplerError::InvalidDistribution);
        }
        for probability in &mut probabilities {
            *probability /= total;
        }
        Ok(Self { probabilities })
    }

    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }

    /// Returns the probability of one token.
    pub fn probability(&self, token: u32) -> Result<f64, SamplerError> {
        self.probabilities
            .get(token as usize)
            .copied()
            .ok_or(SamplerError::TokenOutOfRange {
                token,
                vocab: self.probabilities.len(),
            })
    }
}

impl MirostatState {
    /// Selects one token and updates the controller from its observed surprise.
    pub fn select(
        &mut self,
        target: &Distribution,
        rng: SamplerRng,
        position: u64,
    ) -> Result<u32, SamplerError> {
        if target.is_empty() {
            return Err(SamplerError::EmptyRow);
        }
        let mut probabilities = vec![0.0; target.len()];
        let mut highest = None;
        let mut total = 0.0;
        for (token, probability) in target.probabilities().iter().copied().enumerate() {
            if probability <= 0.0 {
                continue;
            }
            if highest.map(|(_, best)| probability > best).unwrap_or(true) {
                highest = Some((token, probability));
            }
            if -probability.ln() <= self.maximum_surprise {
                probabilities[token] = probability;
                total += probability;
            }
        }
        if total == 0.0 {
            let (token, probability) = highest.ok_or(SamplerError::AllNaN)?;
            probabilities[token] = probability;
            total = probability;
        }
        for probability in &mut probabilities {
            *probability /= total;
        }
        let filtered = Distribution { probabilities };
        let token = select(&filtered, rng, position)?;
        let observed = -target.probability(token)?.ln();
        self.maximum_surprise -=
            self.config.learning_rate * (observed - self.config.target_surprise);
        self.maximum_surprise = self.maximum_surprise.max(f64::EPSILON);
        Ok(token)
    }
}

/// Builds the distribution one sampler produces from one logit row.
///
/// Non-finite logits are excluded. A row without a finite value is an error.
pub fn distribution(logits: &[f32], sampler: &Sampler) -> Result<Distribution, SamplerError> {
    sampler.validate()?;
    if logits.is_empty() {
        return Err(SamplerError::EmptyRow);
    }
    let mut probabilities = vec![0.0_f64; logits.len()];

    if let Temperature::Greedy = sampler.temperature {
        let best = argmax(logits).ok_or(SamplerError::AllNaN)?;
        probabilities[best] = 1.0;
        return Ok(Distribution { probabilities });
    }
    let Temperature::Scaled(temperature) = sampler.temperature else {
        unreachable!("greedy returned above")
    };

    // Order by descending logit, breaking ties toward the lower token id so
    // the eligible set matches the greedy tie rule and the FP64 oracle.
    // Only finite logits are candidates. A NaN carries no information, and a
    // negative infinity has exactly zero probability while still poisoning the
    // mean and standard deviation that top-n-sigma needs.
    let mut order: Vec<usize> = (0..logits.len())
        .filter(|index| logits[*index].is_finite())
        .collect();
    if order.is_empty() {
        return Err(SamplerError::AllNaN);
    }
    order.sort_by(|left, right| {
        f64::from(logits[*right])
            .total_cmp(&f64::from(logits[*left]))
            .then_with(|| left.cmp(right))
    });

    // Top-n-sigma and min-k run on the raw logits, before temperature.
    // Applying them here is what makes the surviving set independent of
    // temperature.
    for truncation in &sampler.truncations {
        let keep = match truncation {
            Truncation::TopNSigma(sigmas) => top_n_sigma_keep(logits, &order, *sigmas),
            Truncation::MinK(tau) => min_k_keep(logits, &order, *tau),
            _ => continue,
        };
        order.truncate(keep.max(1));
    }

    // Temperature is positive, so it preserves the order and the maximum stays
    // first. Subtracting it before exponentiating keeps the sum finite.
    let highest = f64::from(logits[order[0]]) / temperature;
    let mut kept = Vec::with_capacity(order.len());
    let mut total = 0.0_f64;
    for index in &order {
        let weight = (f64::from(logits[*index]) / temperature - highest).exp();
        total += weight;
        kept.push(weight);
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(SamplerError::AllNaN);
    }
    for weight in &mut kept {
        *weight /= total;
    }

    for truncation in &sampler.truncations {
        let keep = match truncation {
            Truncation::TopNSigma(_) | Truncation::MinK(_) => kept.len(),
            Truncation::TopK(count) => count.get().min(kept.len()),
            Truncation::TopP(mass) => prefix_reaching_mass(&kept, *mass),
            Truncation::MinP(fraction) => threshold_keep(&kept, fraction * kept[0]),
            Truncation::TopA(fraction) => threshold_keep(&kept, fraction * kept[0] * kept[0]),
            Truncation::Epsilon(floor) => threshold_keep(&kept, *floor),
            Truncation::Eta(floor) => {
                let entropy = entropy(&kept);
                threshold_keep(&kept, floor.min(floor.sqrt() * (-entropy).exp()))
            }
            Truncation::TailFree(share) => tail_free_keep(&kept, *share),
            Truncation::Typical(mass) => {
                // Typical sampling reorders the candidates, so it rewrites the
                // working set rather than truncating it in place.
                let selected = typical_select(&kept, *mass);
                let mut next_order = Vec::with_capacity(selected.len());
                let mut next_kept = Vec::with_capacity(selected.len());
                for slot in selected {
                    next_order.push(order[slot]);
                    next_kept.push(kept[slot]);
                }
                order = next_order;
                kept = next_kept;
                renormalize(&mut kept)?;
                continue;
            }
        };
        let keep = keep.max(1);
        order.truncate(keep);
        kept.truncate(keep);
        renormalize(&mut kept)?;
    }

    for (slot, index) in order.iter().enumerate() {
        probabilities[*index] = kept[slot];
    }
    Ok(Distribution { probabilities })
}

fn renormalize(kept: &mut [f64]) -> Result<(), SamplerError> {
    let total: f64 = kept.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Err(SamplerError::AllNaN);
    }
    for value in kept {
        *value /= total;
    }
    Ok(())
}

/// Returns the entropy of a normalized distribution in nats.
fn entropy(kept: &[f64]) -> f64 {
    kept.iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum()
}

/// Returns how many leading entries have probability at least `floor`.
///
/// The input is sorted descending, so the survivors are always a prefix.
fn threshold_keep(kept: &[f64], floor: f64) -> usize {
    kept.iter().take_while(|value| **value >= floor).count()
}

/// Returns the shortest prefix whose cumulative mass reaches `mass`.
fn prefix_reaching_mass(kept: &[f64], mass: f64) -> usize {
    let mut covered = 0.0_f64;
    for (count, value) in kept.iter().enumerate() {
        covered += value;
        if covered >= mass {
            return count + 1;
        }
    }
    kept.len()
}

/// Returns how many leading raw logits lie within `sigmas` of the highest.
///
/// The standard deviation is the population figure over the surviving
/// candidates, which is what llama.cpp computes. The paper's reference code
/// uses the sample figure over the whole vocabulary instead. At a vocabulary
/// this size the two differ by about four parts in a million, which no decode
/// run would notice but an exhaustive oracle would.
fn top_n_sigma_keep(logits: &[f32], order: &[usize], sigmas: f64) -> usize {
    if order.len() < 2 {
        return order.len();
    }
    let count = order.len() as f64;
    let mean: f64 = order
        .iter()
        .map(|index| f64::from(logits[*index]))
        .sum::<f64>()
        / count;
    let variance: f64 = order
        .iter()
        .map(|index| {
            let value = f64::from(logits[*index]) - mean;
            value * value
        })
        .sum::<f64>()
        / count;
    let deviation = variance.sqrt();
    if !deviation.is_finite() {
        return order.len();
    }
    let cut = f64::from(logits[order[0]]) - sigmas * deviation;
    order
        .iter()
        .take_while(|index| f64::from(logits[**index]) >= cut)
        .count()
        .max(1)
}

/// Returns the tail-free prefix length for share `z`.
///
/// Position one is always kept and the last position is always dropped, which
/// is what the surviving reference implementation does. A row too short to
/// have a second difference keeps every candidate, because the criterion is
/// undefined there rather than maximally aggressive.
fn tail_free_keep(kept: &[f64], share: f64) -> usize {
    let count = kept.len();
    if count < 3 {
        return count;
    }
    let mut second = Vec::with_capacity(count - 2);
    let mut total = 0.0_f64;
    for slot in 0..count - 2 {
        let value = (kept[slot] - 2.0 * kept[slot + 1] + kept[slot + 2]).abs();
        total += value;
        second.push(value);
    }
    if total <= 0.0 || !total.is_finite() {
        return count;
    }
    let mut covered = 0.0_f64;
    let mut keep = 1;
    for value in second {
        covered += value / total;
        // The comparison keeps equality, so a share of exactly `z` survives.
        if covered <= share {
            keep += 1;
        } else {
            break;
        }
    }
    keep
}

/// Returns how many leading raw logits Min-k keeps for `tau`.
///
/// Min-k, Ding et al. Finds the sharpest relative drop in the sorted logits,
/// weighting each gap by its rank so an early cliff wins, and falls back to a
/// rank proportional to the logit range when the row is nearly flat. Like
/// top-n-sigma it runs before temperature and is exactly temperature
/// invariant, because scaling every logit scales the gaps and the range alike.
fn min_k_keep(logits: &[f32], order: &[usize], tau: f64) -> usize {
    if order.len() < 2 {
        return order.len();
    }
    let highest = f64::from(logits[order[0]]);
    let lowest = f64::from(logits[order[order.len() - 1]]);
    let range = highest - lowest + 1e-8;
    if !range.is_finite() || range <= 0.0 {
        return order.len();
    }
    let mut cliff = 1;
    let mut best = f64::NEG_INFINITY;
    for rank in 1..order.len() {
        let gap = f64::from(logits[order[rank - 1]]) - f64::from(logits[order[rank]]);
        let weighted = (gap / range) / rank as f64;
        // Ties go to the lowest rank, so compare strictly.
        if weighted > best {
            best = weighted;
            cliff = rank;
        }
    }
    let fallback = (tau / range).floor();
    let fallback = if fallback.is_finite() && fallback >= 0.0 {
        (fallback as usize).min(order.len())
    } else {
        order.len()
    };
    cliff.max(fallback).clamp(1, order.len())
}

/// Returns the candidate slots locally typical sampling keeps, in descending
/// probability order.
fn typical_select(kept: &[f64], mass: f64) -> Vec<usize> {
    let target = entropy(kept);
    let mut scored: Vec<(usize, f64)> = kept
        .iter()
        .enumerate()
        .map(|(slot, value)| (slot, ((-value.ln()) - target).abs()))
        .collect();
    scored.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut covered = 0.0_f64;
    let mut selected = Vec::new();
    for (slot, _) in scored {
        selected.push(slot);
        covered += kept[slot];
        if covered >= mass {
            break;
        }
    }
    // Restore descending probability order so the rest of the pipeline and the
    // token scan see the same convention as every other stage.
    selected.sort_unstable();
    selected
}

/// Returns the index of the highest logit, taking the lowest index on a tie.
///
/// Non-finite entries are skipped. Returns `None` without a finite entry.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            continue;
        }
        match best {
            Some((_, current)) if value <= current => {}
            _ => best = Some((index, value)),
        }
    }
    best.map(|(index, _)| index)
}

/// Draws one token from a distribution.
///
/// The scan is over the whole row in token order, so the same uniform value
/// always selects the same token.
pub fn select(
    distribution: &Distribution,
    rng: SamplerRng,
    position: u64,
) -> Result<u32, SamplerError> {
    select_with(distribution, rng.uniform(position, Draw::Select))
}

/// Draws one proposal token from a stream independent of target sampling.
pub fn select_draft(
    distribution: &Distribution,
    rng: SamplerRng,
    position: u64,
) -> Result<u32, SamplerError> {
    select_with(distribution, rng.uniform(position, Draw::Draft))
}

/// Returns the total-variation distance between two categorical distributions.
pub fn total_variation(left: &Distribution, right: &Distribution) -> Result<f64, SamplerError> {
    if left.len() != right.len() {
        return Err(SamplerError::RowMismatch {
            target: left.len(),
            draft: right.len(),
        });
    }
    Ok(0.5
        * left
            .probabilities()
            .iter()
            .zip(right.probabilities())
            .map(|(left, right)| (left - right).abs())
            .sum::<f64>())
}

fn select_with(distribution: &Distribution, uniform: f64) -> Result<u32, SamplerError> {
    let mut cumulative = 0.0_f64;
    let mut last_eligible = None;
    for (index, probability) in distribution.probabilities.iter().copied().enumerate() {
        if probability <= 0.0 {
            continue;
        }
        last_eligible = Some(index);
        cumulative += probability;
        if uniform < cumulative {
            return Ok(index as u32);
        }
    }
    // Rounding can leave the cumulative sum slightly below the uniform value.
    // Falling back to the last eligible token keeps the draw inside the
    // support instead of failing.
    last_eligible
        .map(|index| index as u32)
        .ok_or(SamplerError::AllNaN)
}

/// The outcome of verifying one drafted token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The drafted token is kept.
    Accept,
    /// The drafted token is rejected and this token replaces it.
    Reject { token: u32 },
}

/// Verifies one drafted token against the target distribution.
///
/// `target` is the distribution the full model produces at this position and
/// `draft` is the distribution the drafter sampled from. The emitted
/// distribution equals `target` exactly, whatever the drafter proposed.
///
/// A rejection ends the drafted run: later drafted tokens were conditioned on
/// a token that is no longer there.
pub fn verify(
    target: &Distribution,
    draft: &Distribution,
    drafted: u32,
    rng: SamplerRng,
    position: u64,
) -> Result<Verdict, SamplerError> {
    if target.len() != draft.len() {
        return Err(SamplerError::RowMismatch {
            target: target.len(),
            draft: draft.len(),
        });
    }
    let target_probability = target.probability(drafted)?;
    let draft_probability = draft.probability(drafted)?;

    // A token the drafter could not have produced cannot be accepted. The
    // ratio is undefined, so this is an outright rejection.
    let accepted = if draft_probability > 0.0 {
        let ratio = target_probability / draft_probability;
        ratio >= 1.0 || rng.uniform(position, Draw::Accept) < ratio
    } else {
        false
    };
    if accepted {
        return Ok(Verdict::Accept);
    }

    let residual = correction_residual(target, draft)?;
    let token = select_with(&residual, rng.uniform(position, Draw::Select))?;
    Ok(Verdict::Reject { token })
}

/// Returns `normalize(max(target - draft, 0))`.
///
/// When the two distributions coincide, `max(target - draft, 0)` is zero.
/// That can only happen if the drafted token had zero target probability and
/// zero draft probability, so falling back to the target keeps the result in
/// support.
pub fn correction_residual(
    target: &Distribution,
    draft: &Distribution,
) -> Result<Distribution, SamplerError> {
    if target.len() != draft.len() {
        return Err(SamplerError::RowMismatch {
            target: target.len(),
            draft: draft.len(),
        });
    }
    let mut probabilities = Vec::with_capacity(target.len());
    let mut total = 0.0_f64;
    for (left, right) in target
        .probabilities
        .iter()
        .copied()
        .zip(draft.probabilities.iter().copied())
    {
        let value = (left - right).max(0.0);
        total += value;
        probabilities.push(value);
    }
    if total <= 0.0 || !total.is_finite() {
        return Ok(target.clone());
    }
    for value in &mut probabilities {
        *value /= total;
    }
    Ok(Distribution { probabilities })
}

#[cfg(test)]
mod tests;
