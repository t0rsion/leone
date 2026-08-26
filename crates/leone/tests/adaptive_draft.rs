use leone::adaptive_draft::{AdaptiveDecision, AdaptiveObservation, AdaptiveReason};
use leone::{
    AdaptiveController, AdaptiveControllerConfig, AdaptiveDrafter, AdaptiveDrafterConfig,
    ProposalSource,
};
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test values are nonzero")
}

fn observation(width: usize, emitted: usize, milliseconds: u64) -> AdaptiveObservation {
    AdaptiveObservation {
        verifier_positions: nonzero(width),
        emitted_tokens: nonzero(emitted),
        wall_duration: Duration::from_millis(milliseconds),
        controller_duration: Duration::from_micros(2),
    }
}

fn controller_config(warmup_rounds: u64) -> AdaptiveControllerConfig {
    AdaptiveControllerConfig::new(
        NonZeroU64::new(warmup_rounds).expect("test warmup is nonzero"),
        1.10,
        0.03,
        0.25,
    )
    .expect("test controller policy is valid")
}

#[test]
fn suffix_matching_has_priority() {
    let drafter = AdaptiveDrafter::new(
        AdaptiveDrafterConfig::new(nonzero(3), nonzero(2), nonzero(8))
            .expect("test drafting policy is valid"),
    );
    let proposal = drafter
        .propose(&[1, 2, 3, 4, 9, 1, 2])
        .expect("the suffix repeats");
    assert_eq!(proposal.source, ProposalSource::Suffix);
    assert_eq!(proposal.tokens, [3, 4, 9]);
}

#[test]
fn recycling_supplies_a_proposal_without_a_suffix_match() {
    let drafter = AdaptiveDrafter::new(AdaptiveDrafterConfig::default());
    let proposal = drafter
        .propose(&[5, 7, 8, 9, 5])
        .expect("the current token appeared earlier");
    assert_eq!(proposal.source, ProposalSource::TokenRecycling);
    assert_eq!(proposal.tokens, [7, 8, 9]);
}

#[test]
fn recycling_votes_over_shared_continuation_prefixes() {
    let drafter = AdaptiveDrafter::new(AdaptiveDrafterConfig::default());
    let proposal = drafter
        .propose(&[5, 7, 8, 1, 5, 7, 8, 2, 5])
        .expect("two prior continuations share a prefix");
    assert_eq!(proposal.source, ProposalSource::TokenRecycling);
    assert_eq!(proposal.tokens, [7, 8, 2]);
}

#[test]
fn controller_measures_plain_decode_before_probing() {
    let mut controller = AdaptiveController::new(controller_config(2));
    assert_eq!(
        controller.decide(7),
        AdaptiveDecision::Plain {
            reason: AdaptiveReason::PlainWarmup
        }
    );
    controller.observe(observation(1, 1, 10)).unwrap();
    assert_eq!(
        controller.decide(7),
        AdaptiveDecision::Plain {
            reason: AdaptiveReason::PlainWarmup
        }
    );
    controller.observe(observation(1, 1, 10)).unwrap();
    assert_eq!(
        controller.decide(7),
        AdaptiveDecision::Speculate {
            verifier_positions: nonzero(2),
            reason: AdaptiveReason::WidthProbe
        }
    );
}

#[test]
fn controller_selects_the_fastest_measured_width() {
    let mut controller = AdaptiveController::new(controller_config(1));
    controller.observe(observation(1, 1, 10)).unwrap();
    controller.observe(observation(2, 2, 8)).unwrap();
    controller.observe(observation(4, 4, 12)).unwrap();
    controller.observe(observation(6, 6, 12)).unwrap();
    controller.observe(observation(8, 8, 12)).unwrap();
    assert_eq!(
        controller.decide(7),
        AdaptiveDecision::Speculate {
            verifier_positions: nonzero(8),
            reason: AdaptiveReason::BestMeasuredWidth
        }
    );
}

#[test]
fn regret_limit_returns_to_plain_decode() {
    let mut controller = AdaptiveController::new(controller_config(1));
    controller.observe(observation(1, 1, 1)).unwrap();
    controller.observe(observation(2, 1, 10)).unwrap();
    assert_eq!(
        controller.decide(1),
        AdaptiveDecision::Plain {
            reason: AdaptiveReason::RegretLimit
        }
    );
    assert_eq!(controller.stats().regret_limited_rounds, 1);
}

#[test]
fn preregistered_width_probes_complete_before_regret_fallback() {
    let mut controller = AdaptiveController::new(controller_config(1));
    controller.observe(observation(1, 1, 1)).unwrap();
    controller.observe(observation(2, 1, 10)).unwrap();
    assert_eq!(
        controller.decide(7),
        AdaptiveDecision::Speculate {
            verifier_positions: nonzero(4),
            reason: AdaptiveReason::WidthProbe
        }
    );
}

#[test]
fn no_proposal_never_starts_a_verify_pass() {
    let mut controller = AdaptiveController::new(controller_config(1));
    assert_eq!(
        controller.decide(0),
        AdaptiveDecision::Plain {
            reason: AdaptiveReason::NoProposal
        }
    );
}

#[test]
fn invalid_adaptive_policies_are_typed() {
    assert!(AdaptiveDrafterConfig::new(nonzero(8), nonzero(2), nonzero(8)).is_err());
    assert!(AdaptiveDrafterConfig::new(nonzero(3), nonzero(8), nonzero(2)).is_err());
    assert!(AdaptiveControllerConfig::new(NonZeroU64::new(1).unwrap(), 1.0, 0.03, 0.25,).is_err());
}

#[test]
fn observations_reject_unsupported_widths() {
    let mut controller = AdaptiveController::new(controller_config(1));
    assert!(controller.observe(observation(9, 1, 1)).is_err());
}
