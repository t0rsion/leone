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
- oracle model SHA-256 and recorded dtype,
- KLD definition,
- sample count.

The batched-service study records the quality receipt identifier and digest.
The study fails if the quality model digest differs from the served model.

## Comparative quantized quality

`scripts/quality-concurrent-service.sh` binds one token stream to three runs:
the pinned llama.cpp BF16 or F16 oracle, pinned llama.cpp Q4 execution, and
Leone Q4 evaluation. The script records the model, token file, window,
executable, device, and executable build info in a comparison manifest.

The corpus is `corpus/quality-v03.txt`. Tokenization uses the Q4 subject model,
then reuses the resulting little-endian u32 file for every run. A 2,400-token
run with a 512-token window scores 2,399 rows across overlapping windows. Each
window starts with an empty context, matching the Leone eval contract. Leone
uses chunked prefill with 128-token chunks. Set
`LEONE_QUALITY_PREFILL_CHUNK` to compare another chunk size.

Run the Qwen comparison with:

```text
LEONE_QUALITY_WRITE_RECEIPTS=1 scripts/quality-concurrent-service.sh \
  models/Qwen3-8B-Q4_K_M.gguf \
  models/Qwen3-8B-BF16.gguf \
  corpus/quality-v03.txt 2400 512 quality/qwen3-v03.json cuda
```

The default oracle runs on the CPU. The Q4 llama.cpp and Leone runs use CUDA.
The comparison covers direct eval logits. It does not certify concurrent
server numerics. A server study must retain its own runtime receipt and use
the same model and token manifest. Run the Llama comparison by substituting
`models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` and
`models/Llama-3.2-1B-Instruct-f16.gguf` for the two model paths.
