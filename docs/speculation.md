# Exact speculation

Leone supports point and distribution proposals. Each form preserves the target
sampler distribution.

For a drafted token `x`, target distribution `p`, and draft distribution `q`,
accept with probability

```text
min(1, p(x) / q(x)).
```

If the token is rejected, sample from the residual

```text
normalize(max(p - q, 0)).
```

A proposal with draft probability 0 is never accepted. Target and draft rows
must have the same vocabulary size.

## Proposal controllers

The suffix drafter proposes tokens from repeated committed history. The adaptive
controller measures supported widths before selecting one. The correctable
controller records the distribution that sampled each proposal.

A controller may reduce work. It cannot change the sampling seed, draw purpose,
or target distribution. Cancellation commits no partial proposal.

## Evidence

The FP64 oracle checks reconstruction and deterministic verdicts. Runtime gates
compare plain and speculative token streams for identical seeds. A speed claim
requires an idle-GPU study and a generated receipt. Exactness alone does not
imply a speedup.
