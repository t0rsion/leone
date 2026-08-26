# Preregistered performance gate

`manifest.toml` fixes the public `v0.1.0` workload before measurement. It names
the model digest, machine, CPU masks, repetition count, comparator, and pass
bounds.

## Workload

- Prefill 512 tokens.
- Decode 128 tokens at batch 1.
- Start decode with 512 entries in the KV cache.
- Run one untimed warmup and five timed repetitions.
- Compare the medians.

The decode timer includes graph replay, token transfer, stream synchronization,
and detokenization. It excludes model load, state allocation, prefill, the
initial sample, and graph capture.

## Procedure

Build and test with `taskset -c 16-31`. Pin timings with `taskset -c 0-3,12-15`.
No other compute process may use the GPU.

Run the pinned llama.cpp comparator from `external/PINNED`. Then run Leone on
the same model and workload. Generate a runtime receipt for each engine.

Generate both quality receipts from one token corpus and one BF16 oracle. Link
each runtime receipt to its quality receipt.

## Gate

The performance ratio is

```text
median Leone decode tok/s / median llama.cpp decode tok/s.
```

The ratio must be at least `1.00`.

The quality bound is

```text
mean KLD(Leone Q4 || BF16) <= mean KLD(llama.cpp Q4 || BF16) + 0.02 nats.
```

A missing quality link prints `quality: unverified` and cannot pass. A failed
gate delays the release.

<!-- quality-results:start -->
## Measured v0.1 gate

### Performance

| Engine | Median decode (tok/s) | Receipt |
|---|---:|---|
| Leone | 172.565 | [`2026-08-26T16:59:47Z-runtime-67f91e44.json`](../receipts/2026-08-26T16:59:47Z-runtime-67f91e44.json) |
| llama.cpp | 168.317 | [`2026-08-26T17:01:03Z-runtime-2d188d61.json`](../receipts/2026-08-26T17:01:03Z-runtime-2d188d61.json) |

The performance gate is **pass**. The Leone to llama.cpp median ratio is 1.025238762. The required ratio is at least 1.00.

### Quality

| Subject | Mean KLD (nats) | p99 KLD (nats) | Top-1 agreement | Receipt |
|---|---:|---:|---:|---|
| Leone Q4_K_M | 0.035593219 | 0.279680019 | 0.884955752 | [`2026-08-26T16:59:34Z-quality-4e188eb5.json`](../receipts/2026-08-26T16:59:34Z-quality-4e188eb5.json) |
| llama.cpp Q4_K_M | 0.036080760 | 0.311306432 | 0.884955752 | [`2026-08-26T15:22:24Z-quality-d24c7c28.json`](../receipts/2026-08-26T15:22:24Z-quality-d24c7c28.json) |

The quality gate is **pass**. Leone mean KLD must not exceed 0.056080760 nats (llama.cpp mean KLD plus 0.02 nats).

The overall v0.1 evidence gate is **pass**. The performance gate and the quality gate must both pass.
<!-- quality-results:end -->

## Correctable inference

Correctable inference passes its oracle, exact-output, and runtime gates. The
generated record is [`correctable-20260826T170238Z.json`](../receipts/correctable-20260826T170238Z.json).
