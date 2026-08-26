//! Checks the sampler pipeline against a slow FP64 oracle.
//!
//! The oracle is written to be obviously correct rather than fast. It computes
//! the full softmax before any truncation, builds the descending order by
//! repeated selection instead of sorting, sorts typicality by insertion, and
//! materializes an explicit cumulative array. It shares only the
//! specification with the production sampler: the tie rule, the token-order
//! scan, the renormalization between stages, and the meaning of each stage.

use super::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn oracle_entropy(kept: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    for value in kept {
        if *value > 0.0 {
            total -= value * value.ln();
        }
    }
    total
}

fn oracle_normalize(kept: &mut [f64]) -> Result<(), SamplerError> {
    let mut total = 0.0_f64;
    for value in kept.iter() {
        total += *value;
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(SamplerError::AllNaN);
    }
    for value in kept {
        *value /= total;
    }
    Ok(())
}

fn oracle_truncate(order: &mut Vec<usize>, kept: &mut Vec<f64>, count: usize) {
    let count = count.clamp(1, kept.len());
    order.truncate(count);
    kept.truncate(count);
}

/// Builds a distribution the slow, obvious way.
fn oracle_distribution(logits: &[f32], sampler: &Sampler) -> Result<Vec<f64>, SamplerError> {
    sampler.validate()?;
    if logits.is_empty() {
        return Err(SamplerError::EmptyRow);
    }
    let mut probabilities = vec![0.0_f64; logits.len()];

    let temperature = match sampler.temperature {
        Temperature::Greedy => {
            let mut best: Option<usize> = None;
            for index in 0..logits.len() {
                if !logits[index].is_finite() {
                    continue;
                }
                best = match best {
                    Some(current) if logits[current] >= logits[index] => Some(current),
                    _ => Some(index),
                };
            }
            probabilities[best.ok_or(SamplerError::AllNaN)?] = 1.0;
            return Ok(probabilities);
        }
        Temperature::Scaled(value) => value,
    };

    // Top-n-sigma and min-k cut the raw logits before temperature. The oracle
    // marks the losers NaN and then proceeds as if they had never been present.
    let mut row = logits.to_vec();
    for truncation in &sampler.truncations {
        if let Truncation::MinK(tau) = truncation {
            let mut present: Vec<f64> = row
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| f64::from(*value))
                .collect();
            if present.len() < 2 {
                continue;
            }
            present.sort_by(|left, right| right.total_cmp(left));
            let range = present[0] - present[present.len() - 1] + 1e-8;
            if !range.is_finite() || range <= 0.0 {
                continue;
            }
            let mut cliff = 1;
            let mut best = f64::NEG_INFINITY;
            for rank in 1..present.len() {
                let weighted = ((present[rank - 1] - present[rank]) / range) / rank as f64;
                if weighted > best {
                    best = weighted;
                    cliff = rank;
                }
            }
            let fallback = (tau / range).floor();
            let fallback = if fallback.is_finite() && fallback >= 0.0 {
                (fallback as usize).min(present.len())
            } else {
                present.len()
            };
            let keep = cliff.max(fallback).clamp(1, present.len());
            let floor = present[keep - 1];
            let mut remaining = keep;
            for value in &mut row {
                if !value.is_finite() {
                    continue;
                }
                if f64::from(*value) < floor || remaining == 0 {
                    *value = f32::NAN;
                } else {
                    remaining -= 1;
                }
            }
        }
        if let Truncation::TopNSigma(sigmas) = truncation {
            let present: Vec<f64> = row
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| f64::from(*value))
                .collect();
            if present.is_empty() {
                return Err(SamplerError::AllNaN);
            }
            let count = present.len() as f64;
            let mean = present.iter().sum::<f64>() / count;
            let variance = present
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / count;
            let peak = present.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let cut = peak - sigmas * variance.sqrt();
            if present.len() < 2 || !variance.sqrt().is_finite() {
                continue;
            }
            let mut survivors = 0;
            for value in &mut row {
                if value.is_finite() && f64::from(*value) < cut {
                    *value = f32::NAN;
                } else if value.is_finite() {
                    survivors += 1;
                }
            }
            let _ = survivors;
        }
    }
    let logits = row.as_slice();

    // Full softmax over every surviving entry, before any further truncation.
    let mut highest = f64::NEG_INFINITY;
    for value in logits {
        if value.is_finite() {
            highest = highest.max(f64::from(*value) / temperature);
        }
    }
    if !highest.is_finite() {
        return Err(SamplerError::AllNaN);
    }
    let mut full = vec![0.0_f64; logits.len()];
    let mut total = 0.0_f64;
    for (index, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        let weight = (f64::from(*value) / temperature - highest).exp();
        full[index] = weight;
        total += weight;
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(SamplerError::AllNaN);
    }
    for value in &mut full {
        *value /= total;
    }

    // Descending order by repeated selection, lowest token id on a tie.
    let mut remaining: Vec<usize> = (0..logits.len())
        .filter(|index| logits[*index].is_finite())
        .collect();
    let mut order = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let mut best = 0;
        for slot in 1..remaining.len() {
            let candidate = remaining[slot];
            let current = remaining[best];
            let left = f64::from(logits[candidate]);
            let right = f64::from(logits[current]);
            if left > right || (left == right && candidate < current) {
                best = slot;
            }
        }
        order.push(remaining.remove(best));
    }

    let mut kept: Vec<f64> = order.iter().map(|index| full[*index]).collect();
    oracle_normalize(&mut kept)?;

    for truncation in &sampler.truncations {
        match truncation {
            Truncation::TopNSigma(_) | Truncation::MinK(_) => {}
            Truncation::TopK(count) => oracle_truncate(&mut order, &mut kept, count.get()),
            Truncation::TopP(mass) => {
                let mut covered = 0.0_f64;
                let mut count = kept.len();
                for (slot, value) in kept.iter().enumerate() {
                    covered += *value;
                    if covered >= *mass {
                        count = slot + 1;
                        break;
                    }
                }
                oracle_truncate(&mut order, &mut kept, count);
            }
            Truncation::MinP(fraction) => {
                let floor = fraction * kept[0];
                let count = kept.iter().filter(|value| **value >= floor).count();
                oracle_truncate(&mut order, &mut kept, count);
            }
            Truncation::TopA(fraction) => {
                let floor = fraction * kept[0] * kept[0];
                let count = kept.iter().filter(|value| **value >= floor).count();
                oracle_truncate(&mut order, &mut kept, count);
            }
            Truncation::Epsilon(floor) => {
                let count = kept.iter().filter(|value| **value >= *floor).count();
                oracle_truncate(&mut order, &mut kept, count);
            }
            Truncation::Eta(floor) => {
                let bound = floor.min(floor.sqrt() * (-oracle_entropy(&kept)).exp());
                let count = kept.iter().filter(|value| **value >= bound).count();
                oracle_truncate(&mut order, &mut kept, count);
            }
            Truncation::TailFree(share) => {
                let total_count = kept.len();
                let count = if total_count < 3 {
                    total_count
                } else {
                    let mut second = Vec::new();
                    for slot in 1..total_count - 1 {
                        second.push((kept[slot - 1] - 2.0 * kept[slot] + kept[slot + 1]).abs());
                    }
                    let total: f64 = second.iter().sum();
                    if total <= 0.0 || !total.is_finite() {
                        total_count
                    } else {
                        // Position one is always kept; the last is always
                        // dropped; equality at the share is kept.
                        let mut covered = 0.0_f64;
                        let mut keep = 1;
                        for value in &second {
                            covered += value / total;
                            if covered <= *share {
                                keep += 1;
                            } else {
                                break;
                            }
                        }
                        keep
                    }
                };
                oracle_truncate(&mut order, &mut kept, count);
            }
            Truncation::Typical(mass) => {
                let target = oracle_entropy(&kept);
                let mut scored: Vec<(usize, f64)> = kept
                    .iter()
                    .enumerate()
                    .map(|(slot, value)| (slot, ((-value.ln()) - target).abs()))
                    .collect();
                for outer in 1..scored.len() {
                    let mut inner = outer;
                    while inner > 0
                        && (scored[inner - 1].1 > scored[inner].1
                            || (scored[inner - 1].1 == scored[inner].1
                                && scored[inner - 1].0 > scored[inner].0))
                    {
                        scored.swap(inner - 1, inner);
                        inner -= 1;
                    }
                }
                let mut covered = 0.0_f64;
                let mut chosen = Vec::new();
                for (slot, _) in &scored {
                    chosen.push(*slot);
                    covered += kept[*slot];
                    if covered >= *mass {
                        break;
                    }
                }
                chosen.sort_unstable();
                order = chosen.iter().map(|slot| order[*slot]).collect();
                kept = chosen.iter().map(|slot| kept[*slot]).collect();
                oracle_normalize(&mut kept)?;
                continue;
            }
        }
        oracle_normalize(&mut kept)?;
    }

    for (slot, index) in order.iter().enumerate() {
        probabilities[*index] = kept[slot];
    }
    Ok(probabilities)
}

/// Selects a token from an explicit cumulative array, in token order.
fn oracle_select(probabilities: &[f64], uniform: f64) -> Option<u32> {
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut running = 0.0_f64;
    for value in probabilities {
        running += *value;
        cumulative.push(running);
    }
    let mut last = None;
    for (index, value) in probabilities.iter().enumerate() {
        if *value > 0.0 {
            last = Some(index as u32);
        }
    }
    for (index, bound) in cumulative.iter().enumerate() {
        if probabilities[index] > 0.0 && uniform < *bound {
            return Some(index as u32);
        }
    }
    last
}

fn random_truncation(rng: &mut SmallRng, vocab: usize) -> Truncation {
    match rng.random_range(0..10) {
        0 => Truncation::TopK(
            NonZeroUsize::new(rng.random_range(1..=vocab)).expect("range starts at one"),
        ),
        1 => Truncation::TopP(rng.random_range(0.01_f64..1.0)),
        2 => Truncation::MinP(rng.random_range(0.01_f64..1.0)),
        3 => Truncation::TopA(rng.random_range(0.01_f64..1.0)),
        4 => Truncation::TailFree(rng.random_range(0.01_f64..1.0)),
        5 => Truncation::Typical(rng.random_range(0.01_f64..1.0)),
        6 => Truncation::Epsilon(rng.random_range(0.001_f64..0.5)),
        7 => Truncation::Eta(rng.random_range(0.001_f64..0.5)),
        8 => Truncation::MinK(rng.random_range(0.0_f64..8.0)),
        _ => Truncation::TopNSigma(rng.random_range(0.0_f64..4.0)),
    }
}

fn random_sampler(rng: &mut SmallRng, vocab: usize) -> Sampler {
    let temperature = if rng.random_bool(0.15) {
        Temperature::Greedy
    } else {
        Temperature::Scaled(rng.random_range(0.05_f64..3.0))
    };
    let stages = rng.random_range(0..4);
    let truncations = (0..stages).map(|_| random_truncation(rng, vocab)).collect();
    Sampler {
        temperature,
        truncations,
    }
}

fn random_logits(rng: &mut SmallRng, vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|_| {
            if rng.random_bool(0.02) {
                // Exercise the NaN and infinity paths a real logit row can
                // contain after a masked or saturated matrix multiply.
                [f32::NAN, f32::NEG_INFINITY, 0.0][rng.random_range(0..3)]
            } else {
                rng.random_range(-30.0_f32..30.0)
            }
        })
        .collect()
}

/// Runs the differential and returns the compared and mismatched counts.
fn differential(cases: usize, seed: u64) -> (usize, usize) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut mismatches = 0;
    let mut compared = 0;
    for _ in 0..cases {
        let vocab = rng.random_range(1..=64);
        let logits = random_logits(&mut rng, vocab);
        let sampler = random_sampler(&mut rng, vocab);
        let stream = SamplerRng::new(rng.random());
        let position = rng.random_range(0..1_000_u64);

        match (
            distribution(&logits, &sampler),
            oracle_distribution(&logits, &sampler),
        ) {
            (Err(left), Err(right)) => {
                if left != right {
                    mismatches += 1;
                }
            }
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => mismatches += 1,
            (Ok(produced), Ok(expected)) => {
                compared += 1;
                let worst = produced
                    .probabilities()
                    .iter()
                    .zip(expected.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                if worst > 1e-12 {
                    mismatches += 1;
                    continue;
                }
                let uniform = stream.uniform(position, Draw::Select);
                if select_with(&produced, uniform).ok() != oracle_select(&expected, uniform) {
                    mismatches += 1;
                }
            }
        }
    }
    (compared, mismatches)
}

#[test]
fn sampler_matches_the_fp64_oracle_over_one_hundred_thousand_cases() {
    let (compared, mismatches) = differential(100_000, 0x5361_6d70_6c65_7231);
    assert_eq!(mismatches, 0, "compared {compared} distributions");
}

#[test]
#[ignore = "the full sampler gate, ten million cases"]
fn sampler_matches_the_fp64_oracle_over_ten_million_cases() {
    let mut compared = 0;
    let mut mismatches = 0;
    // Ten independent seeds, so one unlucky stream cannot hide a class of
    // inputs the sampler mishandles.
    for shard in 0..10_u64 {
        let (shard_compared, shard_mismatches) =
            differential(1_000_000, 0x5361_6d70_6c65_7200 + shard);
        compared += shard_compared;
        mismatches += shard_mismatches;
    }
    println!(
        "sampler differential: 10000000 cases, {compared} distributions compared, {mismatches} mismatches"
    );
    assert_eq!(mismatches, 0);
}

#[test]
fn greedy_takes_the_lowest_token_on_a_tie() {
    let logits = [1.0, 4.0, 4.0, f32::NAN];
    let produced = distribution(&logits, &Sampler::greedy()).unwrap();
    assert_eq!(produced.probability(1).unwrap(), 1.0);
    assert_eq!(produced.probability(2).unwrap(), 0.0);
    assert_eq!(argmax(&logits), Some(1));
}

#[test]
fn a_row_without_finite_logits_is_an_error_not_a_token() {
    let logits = [f32::NAN, f32::NEG_INFINITY];
    assert_eq!(
        distribution(&logits, &Sampler::greedy()),
        Err(SamplerError::AllNaN)
    );
}

#[test]
fn top_k_keeps_exactly_that_many_tokens() {
    let logits = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let sampler = Sampler::temperature(1.0).then(Truncation::TopK(NonZeroUsize::new(2).unwrap()));
    let produced = distribution(&logits, &sampler).unwrap();
    assert_eq!(eligible(&produced), 2);
    assert!((produced.probabilities().iter().sum::<f64>() - 1.0).abs() < 1e-15);
}

#[test]
fn min_p_scales_its_floor_with_confidence() {
    let peaked = [10.0_f32, 0.0, 0.0, 0.0];
    let flat = [1.0_f32, 1.0, 1.0, 1.0];
    let sampler = Sampler::temperature(1.0).then(Truncation::MinP(0.5));
    assert_eq!(eligible(&distribution(&peaked, &sampler).unwrap()), 1);
    assert_eq!(eligible(&distribution(&flat, &sampler).unwrap()), 4);
}

#[test]
fn top_n_sigma_does_not_depend_on_temperature() {
    // The surviving set is fixed by the raw logits. Temperature reweights
    // survivors and never readmits a dropped token.
    let logits = [8.0_f32, 7.5, 2.0, 1.0, 0.5];
    let cold = Sampler::temperature(0.1).then(Truncation::TopNSigma(1.0));
    let hot = Sampler::temperature(1000.0).then(Truncation::TopNSigma(1.0));
    let cold = distribution(&logits, &cold).unwrap();
    let hot = distribution(&logits, &hot).unwrap();
    let support = |produced: &Distribution| {
        produced
            .probabilities()
            .iter()
            .map(|value| *value > 0.0)
            .collect::<Vec<_>>()
    };
    assert_eq!(support(&cold), support(&hot));
    assert!(eligible(&cold) < logits.len());
}

#[test]
fn epsilon_and_eta_keep_at_least_the_peak() {
    let logits = [5.0_f32, 0.0, -5.0];
    for stage in [Truncation::Epsilon(1.0), Truncation::Eta(1.0)] {
        let sampler = Sampler::temperature(1.0).then(stage);
        let produced = distribution(&logits, &sampler).unwrap();
        assert_eq!(produced.probability(0).unwrap(), 1.0);
    }
}

#[test]
fn typical_sampling_keeps_the_least_surprising_tokens() {
    // Locally typical sampling ranks by how close a token's surprise is to the
    // distribution's entropy, not by probability.
    let logits = [4.0_f32, 2.0, 1.5, 1.0, 0.5, 0.0];
    let mass = 0.5;
    let sampler = Sampler::temperature(1.0).then(Truncation::Typical(mass));
    let produced = distribution(&logits, &sampler).unwrap();

    let plain = distribution(&logits, &Sampler::temperature(1.0)).unwrap();
    let entropy: f64 = plain
        .probabilities()
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum();
    let mut ranked: Vec<(usize, f64)> = plain
        .probabilities()
        .iter()
        .enumerate()
        .map(|(token, value)| (token, ((-value.ln()) - entropy).abs()))
        .collect();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut covered = 0.0_f64;
    let mut wanted = Vec::new();
    for (token, _) in ranked {
        wanted.push(token);
        covered += plain.probabilities()[token];
        if covered >= mass {
            break;
        }
    }
    wanted.sort_unstable();

    let kept: Vec<usize> = produced
        .probabilities()
        .iter()
        .enumerate()
        .filter(|(_, value)| **value > 0.0)
        .map(|(token, _)| token)
        .collect();
    assert_eq!(kept, wanted);
}

#[test]
fn every_stage_keeps_a_usable_distribution() {
    let logits = [3.0_f32, 2.0, 1.0, 0.0, -1.0];
    let stages = [
        Truncation::TopK(NonZeroUsize::new(1).unwrap()),
        Truncation::TopP(0.01),
        Truncation::MinP(1.0),
        Truncation::TopA(1.0),
        Truncation::TailFree(0.01),
        Truncation::Typical(0.01),
        Truncation::Epsilon(1.0),
        Truncation::Eta(1.0),
        Truncation::TopNSigma(0.0),
        Truncation::MinK(3.0),
    ];
    for stage in stages {
        let sampler = Sampler::temperature(1.0).then(stage);
        let produced = distribution(&logits, &sampler).unwrap();
        assert!(eligible(&produced) >= 1, "{} emptied the set", stage.name());
        let total: f64 = produced.probabilities().iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "{} left mass {total}",
            stage.name()
        );
    }
}

#[test]
fn an_invalid_policy_is_rejected() {
    let logits = [1.0_f32, 2.0];
    for temperature in [0.0, f64::NAN] {
        let sampler = Sampler::temperature(temperature);
        assert_eq!(
            distribution(&logits, &sampler),
            Err(SamplerError::Temperature)
        );
    }
    for stage in [
        Truncation::TopP(1.5),
        Truncation::MinP(0.0),
        Truncation::Eta(-1.0),
        Truncation::TopNSigma(-1.0),
    ] {
        let sampler = Sampler::temperature(1.0).then(stage);
        assert_eq!(
            distribution(&logits, &sampler),
            Err(SamplerError::Truncation {
                stage: stage.name()
            })
        );
    }
}

fn eligible(produced: &Distribution) -> usize {
    produced
        .probabilities()
        .iter()
        .filter(|value| **value > 0.0)
        .count()
}

#[test]
fn a_matching_draft_is_always_accepted() {
    let logits = [1.0_f32, 2.0, 3.0];
    let target = distribution(&logits, &Sampler::greedy()).unwrap();
    let draft = target.clone();
    let rng = SamplerRng::new(7);
    assert_eq!(verify(&target, &draft, 2, rng, 0).unwrap(), Verdict::Accept);
}

#[test]
fn a_draft_outside_the_target_support_is_rejected() {
    let target = distribution(&[0.0_f32, 10.0], &Sampler::greedy()).unwrap();
    let draft = distribution(&[10.0_f32, 0.0], &Sampler::greedy()).unwrap();
    let rng = SamplerRng::new(11);
    assert_eq!(
        verify(&target, &draft, 0, rng, 0).unwrap(),
        Verdict::Reject { token: 1 }
    );
}

#[test]
fn verification_reproduces_the_target_distribution() {
    // The exactness claim, checked empirically: emitted tokens follow the
    // target, not the draft.
    let target_logits = [0.0_f32, 1.0, 2.0, 3.0];
    let draft_logits = [3.0_f32, 2.0, 1.0, 0.0];
    let sampler = Sampler::temperature(1.0);
    let target = distribution(&target_logits, &sampler).unwrap();
    let draft = distribution(&draft_logits, &sampler).unwrap();

    let trials = 400_000_u64;
    let mut counts = [0_u64; 4];
    let rng = SamplerRng::new(0xa11c_e5ed);
    for trial in 0..trials {
        let drafted = select(&draft, rng, trial * 2).unwrap();
        let token = match verify(&target, &draft, drafted, rng, trial * 2 + 1).unwrap() {
            Verdict::Accept => drafted,
            Verdict::Reject { token } => token,
        };
        counts[token as usize] += 1;
    }
    for (token, count) in counts.iter().enumerate() {
        let observed = *count as f64 / trials as f64;
        let expected = target.probability(token as u32).unwrap();
        // Four standard deviations of a binomial at this trial count.
        let tolerance = 4.0 * (expected * (1.0 - expected) / trials as f64).sqrt();
        assert!(
            (observed - expected).abs() <= tolerance.max(1e-4),
            "token {token}: observed {observed}, expected {expected}"
        );
    }
}

#[test]
fn the_random_stream_depends_only_on_seed_position_and_purpose() {
    let rng = SamplerRng::new(1234);
    assert_eq!(rng.uniform(9, Draw::Select), rng.uniform(9, Draw::Select));
    assert_ne!(rng.uniform(9, Draw::Select), rng.uniform(9, Draw::Accept));
    assert_ne!(rng.uniform(9, Draw::Select), rng.uniform(10, Draw::Select));
    assert_ne!(
        rng.uniform(9, Draw::Select),
        SamplerRng::new(1235).uniform(9, Draw::Select)
    );
    for position in 0..10_000 {
        let value = rng.uniform(position, Draw::Select);
        assert!((0.0..1.0).contains(&value));
    }
}
