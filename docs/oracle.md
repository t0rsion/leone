# Correctness oracles

A result ships only when an implementation-independent oracle can reject it.
The production path and oracle must not share the operation under test.

## Quantized blocks

Each supported GGUF block format has a scalar decoder. Tests enumerate every
payload byte code for the format and compare full decoded blocks. Partial blocks
return a typed error.

The CPU and CUDA import paths compare against the same scalar contract. Parser
recognition of a GGUF type does not make it a runtime format.

## CUDA kernels

Kernel tests use FP64 or composed scalar references. Tests cover production
shapes, boundary shapes, ties, cancellation points, and deterministic reruns.
Bitwise equality is required where the contract fixes reduction order. Other
operations state an absolute or relative error bound.

## Sampling

The sampling oracle computes each distribution in FP64. It materializes the
full categorical row and applies truncation in the documented order. Greedy
sampling chooses the lowest token identifier on a finite tie. A row without a
finite logit fails.

The full gate compares ten million seeded cases. For target distribution `p`
and draft distribution `q`, speculative verification checks that

```text
min(p, q) + (1 - sum(min(p, q))) * normalize(max(p - q, 0)) = p.
```

## Sessions and scheduling

A session gate compares uninterrupted generation with live replay and restored
replay at randomized cut points. Fork and hibernation gates compare each child
with an independently evaluated continuation. Cancellation must release the
child state.

The scheduler gate compares each interleaved transcript with isolated execution.
Admission, KV reservation, cancellation overshoot, and quantum size are explicit
bounds.

## Quality

A quality receipt binds four values:

- corpus SHA-256,
- BF16 oracle model SHA-256,
- KLD definition,
- sample count.

A runtime receipt references the quality receipt by identifier. Without that
link, output states `quality: unverified`.
