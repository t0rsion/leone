//! Adjusts a logit row from the decode history.
//!
//! Penalties run before the sampler. Truncation rewrites probabilities from
//! the row alone. A penalty can promote a token that truncation would
//! otherwise have removed, and it cannot be expressed as a filter over the
//! sorted distribution.
//!
//! There is no paper for these penalties. The specification follows llama.cpp.

use std::collections::BTreeMap;
use thiserror::Error;

/// An invalid penalty configuration.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PenaltyError {
    #[error("the repetition penalty must be finite and greater than zero")]
    Repetition,
    #[error("the presence penalty must be finite")]
    Presence,
    #[error("the frequency penalty must be finite")]
    Frequency,
    #[error("the DRY {field} is outside its valid range")]
    Dry { field: &'static str },
}

/// The repeated-suffix penalty, known as DRY.
///
/// For every token that could continue a repeat of earlier decode history,
/// the penalty is `multiplier * base.powi(match - allowed)`, applied only
/// once `match` reaches `allowed`. A short coincidental repeat costs almost
/// nothing. A long verbatim loop becomes unaffordable.
#[derive(Debug, Clone, PartialEq)]
pub struct Dry {
    pub multiplier: f64,
    pub base: f64,
    pub allowed_length: usize,
    /// How many recent tokens the search considers.
    pub window: usize,
    /// Tokens a repeat may not span, such as a newline or a turn marker.
    pub breakers: Vec<u32>,
}

impl Dry {
    /// Returns the reference defaults, with the penalty off.
    pub fn disabled() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            window: 64,
            breakers: Vec::new(),
        }
    }

    fn validate(&self) -> Result<(), PenaltyError> {
        if !self.multiplier.is_finite() || self.multiplier < 0.0 {
            return Err(PenaltyError::Dry {
                field: "multiplier",
            });
        }
        if !self.base.is_finite() || self.base <= 0.0 {
            return Err(PenaltyError::Dry { field: "base" });
        }
        if self.allowed_length == 0 {
            return Err(PenaltyError::Dry {
                field: "allowed length",
            });
        }
        Ok(())
    }

    fn is_active(&self) -> bool {
        self.multiplier > 0.0 && self.window > 0
    }
}

/// Everything applied to a logit row before the sampler sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Penalties {
    /// Divides positive logits and multiplies negative ones. 1.0 is off.
    pub repetition: f64,
    /// Subtracted once from any token already seen. 0.0 is off.
    pub presence: f64,
    /// Subtracted once per earlier occurrence. 0.0 is off.
    pub frequency: f64,
    /// How many recent tokens the three counting penalties consider.
    pub window: usize,
    pub dry: Dry,
}

impl Penalties {
    /// Returns penalties that change nothing.
    pub fn none() -> Self {
        Self {
            repetition: 1.0,
            presence: 0.0,
            frequency: 0.0,
            window: 64,
            dry: Dry::disabled(),
        }
    }

    /// Returns whether any penalty would change a logit.
    pub fn is_active(&self) -> bool {
        self.repetition != 1.0
            || self.presence != 0.0
            || self.frequency != 0.0
            || self.dry.is_active()
    }

    fn validate(&self) -> Result<(), PenaltyError> {
        if !self.repetition.is_finite() || self.repetition <= 0.0 {
            return Err(PenaltyError::Repetition);
        }
        if !self.presence.is_finite() {
            return Err(PenaltyError::Presence);
        }
        if !self.frequency.is_finite() {
            return Err(PenaltyError::Frequency);
        }
        self.dry.validate()
    }

    /// Rewrites `logits` in place from the decode history in `context`.
    ///
    /// `context` is the whole token history, prompt included. Only the last
    /// `window` tokens are considered, which is what bounds the cost.
    pub fn apply(&self, logits: &mut [f32], context: &[u32]) -> Result<(), PenaltyError> {
        self.validate()?;
        if !self.is_active() {
            return Ok(());
        }
        let recent = &context[context.len().saturating_sub(self.window)..];

        if self.repetition != 1.0 || self.presence != 0.0 || self.frequency != 0.0 {
            let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
            for token in recent {
                *counts.entry(*token).or_default() += 1;
            }
            for (token, count) in counts {
                let Some(logit) = logits.get_mut(token as usize) else {
                    continue;
                };
                let mut value = f64::from(*logit);
                // Dividing a positive logit and multiplying a negative one
                // moves both toward zero, so the penalty always discourages.
                if value > 0.0 {
                    value /= self.repetition;
                } else {
                    value *= self.repetition;
                }
                value -= self.presence;
                value -= self.frequency * f64::from(count);
                *logit = value as f32;
            }
        }

        if self.dry.is_active() {
            self.apply_dry(logits, context);
        }
        Ok(())
    }

    /// Penalizes every token that would extend a repeat of earlier text.
    fn apply_dry(&self, logits: &mut [f32], context: &[u32]) {
        let recent = &context[context.len().saturating_sub(self.dry.window)..];
        if recent.len() < 2 {
            return;
        }
        let penalties = self.dry_penalties(recent);
        for (token, penalty) in penalties {
            if let Some(logit) = logits.get_mut(token as usize) {
                *logit = (f64::from(*logit) - penalty) as f32;
            }
        }
    }

    fn dry_penalties(&self, recent: &[u32]) -> BTreeMap<u32, f64> {
        let last = recent.len() - 1;
        let mut penalties: BTreeMap<u32, f64> = BTreeMap::new();
        for start in 0..last {
            if recent[start] != recent[last] {
                continue;
            }
            let length = self.dry_match_length(recent, start, last);
            if length < self.dry.allowed_length {
                continue;
            }
            let continuation = recent[start + 1];
            if self.dry.breakers.contains(&continuation) {
                continue;
            }
            let exponent = (length - self.dry.allowed_length) as i32;
            let penalty = self.dry.multiplier * self.dry.base.powi(exponent);
            // A token reachable by several repeats takes the longest one.
            let slot = penalties.entry(continuation).or_insert(0.0);
            *slot = (*slot).max(penalty);
        }
        penalties
    }

    fn dry_match_length(&self, recent: &[u32], start: usize, last: usize) -> usize {
        let mut length = 1;
        while length <= start
            && length <= last
            && recent[start - length] == recent[last - length]
            && !self.dry.breakers.contains(&recent[last - length])
        {
            length += 1;
        }
        length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Applies the penalties the slow, obvious way.
    fn oracle(penalties: &Penalties, logits: &[f32], context: &[u32]) -> Vec<f32> {
        let mut out = logits.to_vec();
        if !penalties.is_active() {
            return out;
        }
        apply_oracle_repetition(penalties, logits, context, &mut out);
        apply_oracle_dry(penalties, context, &mut out);
        out
    }

    fn apply_oracle_repetition(
        penalties: &Penalties,
        logits: &[f32],
        context: &[u32],
        out: &mut [f32],
    ) {
        let start = context.len().saturating_sub(penalties.window);
        for (token, logit) in out.iter_mut().enumerate() {
            let count = context[start..]
                .iter()
                .filter(|entry| **entry as usize == token)
                .count() as u32;
            if count == 0 {
                continue;
            }
            let mut value = f64::from(logits[token]);
            if value > 0.0 {
                value /= penalties.repetition;
            } else {
                value *= penalties.repetition;
            }
            value -= penalties.presence;
            value -= penalties.frequency * f64::from(count);
            *logit = value as f32;
        }
    }

    fn apply_oracle_dry(penalties: &Penalties, context: &[u32], out: &mut [f32]) {
        if !penalties.dry.is_active() {
            return;
        }
        let dry_start = context.len().saturating_sub(penalties.dry.window);
        let recent = &context[dry_start..];
        if recent.len() < 2 {
            return;
        }
        let last = recent.len() - 1;
        let mut best = vec![0.0; out.len()];
        for candidate in 0..last {
            let Some(length) = oracle_dry_candidate_length(penalties, recent, candidate, last)
            else {
                continue;
            };
            let next = recent[candidate + 1] as usize;
            if penalties.dry.breakers.contains(&(next as u32)) || next >= out.len() {
                continue;
            }
            let penalty = (0..length - penalties.dry.allowed_length)
                .fold(penalties.dry.multiplier, |value, _| {
                    value * penalties.dry.base
                });
            if penalty > best[next] {
                best[next] = penalty;
            }
        }
        for (token, penalty) in best.iter().enumerate() {
            if *penalty > 0.0 {
                out[token] = (f64::from(out[token]) - penalty) as f32;
            }
        }
    }

    fn oracle_dry_candidate_length(
        penalties: &Penalties,
        recent: &[u32],
        candidate: usize,
        last: usize,
    ) -> Option<usize> {
        if recent[candidate] != recent[last] {
            return None;
        }
        let mut length = 1;
        while length <= candidate
            && length <= last
            && recent[candidate - length] == recent[last - length]
            && !penalties.dry.breakers.contains(&recent[last - length])
        {
            length += 1;
        }
        (length >= penalties.dry.allowed_length).then_some(length)
    }

    #[test]
    fn no_penalty_changes_nothing() {
        let logits = [1.0_f32, -2.0, 3.0];
        let mut out = logits;
        Penalties::none().apply(&mut out, &[0, 1, 2, 2]).unwrap();
        assert_eq!(out, logits);
    }

    #[test]
    fn repetition_moves_a_logit_toward_zero_from_either_side() {
        let mut out = [4.0_f32, -4.0, 4.0];
        let penalties = Penalties {
            repetition: 2.0,
            ..Penalties::none()
        };
        penalties.apply(&mut out, &[0, 1]).unwrap();
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], -8.0);
        assert_eq!(out[2], 4.0);
    }

    #[test]
    fn presence_applies_once_and_frequency_applies_per_occurrence() {
        let mut presence = [0.0_f32; 2];
        Penalties {
            presence: 1.0,
            ..Penalties::none()
        }
        .apply(&mut presence, &[0, 0, 0])
        .unwrap();
        assert_eq!(presence[0], -1.0);

        let mut frequency = [0.0_f32; 2];
        Penalties {
            frequency: 1.0,
            ..Penalties::none()
        }
        .apply(&mut frequency, &[0, 0, 0])
        .unwrap();
        assert_eq!(frequency[0], -3.0);
    }

    #[test]
    fn the_window_bounds_what_counts() {
        let mut out = [0.0_f32; 2];
        Penalties {
            frequency: 1.0,
            window: 2,
            ..Penalties::none()
        }
        .apply(&mut out, &[0, 0, 0, 0, 1])
        .unwrap();
        // Only the last two tokens count, and one of them is token 1.
        assert_eq!(out[0], -1.0);
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn dry_grows_exponentially_with_the_match() {
        let dry = Dry {
            multiplier: 1.0,
            base: 2.0,
            allowed_length: 2,
            window: 64,
            breakers: Vec::new(),
        };
        let penalties = Penalties {
            dry,
            ..Penalties::none()
        };
        // "1 2 3" appeared, and the context now ends "1 2 3", so token 4
        // continues a three-token repeat: 1.0 * 2^(3-2) = 2.
        let mut out = [0.0_f32; 8];
        penalties
            .apply(&mut out, &[1, 2, 3, 4, 9, 1, 2, 3])
            .unwrap();
        assert_eq!(out[4], -2.0);

        // "7 1 2 3" matches four tokens, so the exponent is two and the
        // penalty is 1.0 * 2^2.
        let mut longer = [0.0_f32; 8];
        penalties
            .apply(&mut longer, &[7, 1, 2, 3, 4, 9, 7, 1, 2, 3])
            .unwrap();
        assert_eq!(longer[4], -4.0);
    }

    #[test]
    fn dry_ignores_a_match_shorter_than_the_allowed_length() {
        let penalties = Penalties {
            dry: Dry {
                multiplier: 1.0,
                base: 2.0,
                allowed_length: 4,
                window: 64,
                breakers: Vec::new(),
            },
            ..Penalties::none()
        };
        let mut out = [0.0_f32; 8];
        penalties
            .apply(&mut out, &[1, 2, 3, 4, 9, 1, 2, 3])
            .unwrap();
        assert_eq!(out, [0.0_f32; 8]);
    }

    #[test]
    fn a_sequence_breaker_stops_a_repeat_from_spanning_it() {
        let penalties = Penalties {
            dry: Dry {
                multiplier: 1.0,
                base: 2.0,
                allowed_length: 2,
                window: 64,
                breakers: vec![9],
            },
            ..Penalties::none()
        };
        // The would-be match runs back through token 9, so it is cut short.
        let mut out = [0.0_f32; 8];
        penalties
            .apply(&mut out, &[1, 9, 2, 3, 5, 1, 9, 2])
            .unwrap();
        assert_eq!(out, [0.0_f32; 8]);
    }

    #[test]
    fn an_invalid_configuration_is_rejected() {
        let mut out = [0.0_f32; 2];
        for (penalties, expected) in [
            (
                Penalties {
                    repetition: 0.0,
                    ..Penalties::none()
                },
                PenaltyError::Repetition,
            ),
            (
                Penalties {
                    presence: f64::NAN,
                    ..Penalties::none()
                },
                PenaltyError::Presence,
            ),
            (
                Penalties {
                    dry: Dry {
                        base: 0.0,
                        ..Dry::disabled()
                    },
                    ..Penalties::none()
                },
                PenaltyError::Dry { field: "base" },
            ),
        ] {
            assert_eq!(penalties.apply(&mut out, &[0]), Err(expected));
        }
    }

    #[test]
    fn penalties_match_the_slow_oracle_over_random_cases() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(0x70656e_616c7479);
        for _ in 0..200_000 {
            let vocab = rng.random_range(1..=32_usize);
            let history = rng.random_range(1..=48_usize);
            let logits: Vec<f32> = (0..vocab)
                .map(|_| rng.random_range(-8.0_f32..8.0))
                .collect();
            let context: Vec<u32> = (0..history)
                .map(|_| rng.random_range(0..vocab as u32))
                .collect();
            let breakers = if rng.random_bool(0.3) {
                vec![rng.random_range(0..vocab as u32)]
            } else {
                Vec::new()
            };
            let penalties = Penalties {
                repetition: if rng.random_bool(0.5) {
                    1.0
                } else {
                    rng.random_range(0.5_f64..2.0)
                },
                presence: if rng.random_bool(0.5) {
                    0.0
                } else {
                    rng.random_range(-1.0_f64..1.0)
                },
                frequency: if rng.random_bool(0.5) {
                    0.0
                } else {
                    rng.random_range(-1.0_f64..1.0)
                },
                window: rng.random_range(1..=64),
                dry: Dry {
                    multiplier: if rng.random_bool(0.5) {
                        0.0
                    } else {
                        rng.random_range(0.1_f64..3.0)
                    },
                    base: rng.random_range(1.1_f64..2.5),
                    allowed_length: rng.random_range(1..=4),
                    window: rng.random_range(1..=64),
                    breakers,
                },
            };
            let expected = oracle(&penalties, &logits, &context);
            let mut produced = logits.clone();
            penalties.apply(&mut produced, &context).unwrap();
            assert_eq!(produced, expected, "penalties {penalties:?}");
        }
    }
}
