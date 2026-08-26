//! Proposes continuation tokens without running the model.
//!
//! Suffix drafting (prompt lookup) searches committed context for an earlier
//! occurrence of the current suffix and proposes whatever followed it. It
//! costs no matrix work, so a rejected draft costs only the verification it
//! triggered.
//!
//! Grounded prompts that quote the context accept more drafts than creative
//! continuation. The controller enables drafting by measurement rather than
//! by default.

use std::num::NonZeroUsize;
use thiserror::Error;

/// An invalid drafter configuration.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DrafterError {
    #[error("the shortest match ({shortest}) exceeds the longest ({longest})")]
    MatchRange { shortest: usize, longest: usize },
}

/// What a drafter proposed for one step.
///
/// `Nothing` is a real outcome, not a failure. Most steps on creative text
/// find no earlier occurrence of the current suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Draft {
    Nothing,
    Tokens(Vec<u32>),
}

impl Draft {
    pub fn tokens(&self) -> &[u32] {
        match self {
            Self::Nothing => &[],
            Self::Tokens(tokens) => tokens,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens().len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens().is_empty()
    }
}

/// Proposes tokens by matching the current suffix against earlier context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuffixDrafter {
    longest_match: NonZeroUsize,
    shortest_match: NonZeroUsize,
    proposal: NonZeroUsize,
}

impl SuffixDrafter {
    /// Creates a drafter that tries suffixes from `longest_match` down to
    /// `shortest_match` and proposes at most `proposal` tokens.
    pub fn new(
        longest_match: NonZeroUsize,
        shortest_match: NonZeroUsize,
        proposal: NonZeroUsize,
    ) -> Result<Self, DrafterError> {
        if shortest_match > longest_match {
            return Err(DrafterError::MatchRange {
                shortest: shortest_match.get(),
                longest: longest_match.get(),
            });
        }
        Ok(Self {
            longest_match,
            shortest_match,
            proposal,
        })
    }

    pub const fn proposal(&self) -> NonZeroUsize {
        self.proposal
    }

    /// Returns the tokens that followed an earlier occurrence of the current
    /// suffix.
    ///
    /// A longer match is tried first, because it is rarer and more likely to
    /// be accepted. Within one match length the search prefers the most recent
    /// occurrence that can supply a full proposal, and falls back to the most
    /// recent occurrence of any length. Preferring a full proposal matters on
    /// a repeating run, where the nearest occurrence sits one token back and
    /// could otherwise supply only one token.
    ///
    /// Matching is on token identifiers. Identical text can tokenize
    /// differently in different surroundings, so this misses some repeats a
    /// byte-level match would find.
    pub fn draft(&self, context: &[u32]) -> Draft {
        let wanted = self.proposal.get();
        let longest = self.longest_match.get().min(context.len());
        for length in (self.shortest_match.get()..=longest).rev() {
            let suffix_start = context.len() - length;
            let suffix = &context[suffix_start..];
            let mut fallback: Option<&[u32]> = None;
            // Scan backwards so the most recent occurrence is seen first. An
            // occurrence must begin before the suffix does, so there is always
            // at least one token after it.
            for start in (0..suffix_start).rev() {
                if &context[start..start + length] != suffix {
                    continue;
                }
                let follow = start + length;
                let end = (follow + wanted).min(context.len());
                let proposal = &context[follow..end];
                if proposal.len() == wanted {
                    return Draft::Tokens(proposal.to_vec());
                }
                fallback.get_or_insert(proposal);
            }
            if let Some(proposal) = fallback {
                return Draft::Tokens(proposal.to_vec());
            }
        }
        Draft::Nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drafter(longest: usize, shortest: usize, proposal: usize) -> SuffixDrafter {
        SuffixDrafter::new(
            NonZeroUsize::new(longest).unwrap(),
            NonZeroUsize::new(shortest).unwrap(),
            NonZeroUsize::new(proposal).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn an_inverted_match_range_is_rejected() {
        assert_eq!(
            SuffixDrafter::new(
                NonZeroUsize::new(2).unwrap(),
                NonZeroUsize::new(3).unwrap(),
                NonZeroUsize::new(4).unwrap(),
            ),
            Err(DrafterError::MatchRange {
                shortest: 3,
                longest: 2
            })
        );
    }

    #[test]
    fn a_repeated_phrase_proposes_its_continuation() {
        // "a b c d e" appeared earlier; the suffix "a b" should propose "c d".
        let context = [1, 2, 3, 4, 5, 9, 9, 1, 2];
        assert_eq!(drafter(4, 2, 2).draft(&context), Draft::Tokens(vec![3, 4]));
    }

    #[test]
    fn the_most_recent_occurrence_wins() {
        // "1 2" is followed by 3 early and by 7 later. Take the later one.
        let context = [1, 2, 3, 0, 1, 2, 7, 0, 1, 2];
        assert_eq!(drafter(2, 2, 1).draft(&context), Draft::Tokens(vec![7]));
    }

    #[test]
    fn a_longer_match_is_preferred_over_a_shorter_one() {
        // Suffix "5 1 2". The three-token match is older than the two-token
        // match, and it still wins because it is longer.
        let context = [5, 1, 2, 42, 0, 9, 1, 2, 77, 0, 5, 1, 2];
        assert_eq!(drafter(3, 2, 1).draft(&context), Draft::Tokens(vec![42]));
    }

    #[test]
    fn no_earlier_occurrence_proposes_nothing() {
        let context = [1, 2, 3, 4, 5];
        assert_eq!(drafter(3, 2, 2).draft(&context), Draft::Nothing);
        assert!(drafter(3, 2, 2).draft(&context).is_empty());
    }

    #[test]
    fn a_context_shorter_than_the_shortest_match_proposes_nothing() {
        assert_eq!(drafter(4, 3, 2).draft(&[7, 7]), Draft::Nothing);
        assert_eq!(drafter(4, 3, 2).draft(&[]), Draft::Nothing);
    }

    #[test]
    fn an_alternating_pattern_proposes_its_own_continuation() {
        // Context "1 2 1 2" makes "1 2" the natural next guess. Proposing
        // tokens that lie inside the matched suffix is correct: they exist,
        // and the draft is only a guess the verifier will check.
        let context = [1, 2, 1, 2];
        assert_eq!(drafter(2, 2, 2).draft(&context), Draft::Tokens(vec![1, 2]));
    }

    #[test]
    fn a_proposal_is_clamped_to_the_end_of_the_context() {
        let context = [1, 2, 9, 1, 2];
        assert_eq!(
            drafter(2, 2, 8).draft(&context),
            Draft::Tokens(vec![9, 1, 2])
        );
    }

    #[test]
    fn a_proposal_stops_at_the_requested_length() {
        let context = [1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2];
        assert_eq!(
            drafter(2, 2, 3).draft(&context),
            Draft::Tokens(vec![3, 4, 5])
        );
    }

    #[test]
    fn a_repeating_run_proposes_the_run() {
        let context = [4_u32; 32];
        assert_eq!(
            drafter(8, 2, 4).draft(&context),
            Draft::Tokens(vec![4, 4, 4, 4])
        );
    }
}
