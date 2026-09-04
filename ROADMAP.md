# Roadmap

## Current gate status

The v0.2.0 candidate passes its implementation, evidence, source hygiene,
dependency, packaging, and archive gates on the local RTX 4090. The evidence
summary is generated in
[docs/v0.2-release.md](docs/v0.2-release.md).

Evidence gates control each release. Dates only control priority.

## 0.1.0: coherent research preview

Gate:

- Dense Qwen3 and Llama 3 pass independent CPU and CUDA differentials.
- Quantized block decoders pass exhaustive scalar comparisons.
- Session replay, fork, hibernation, cancellation, and scheduling gates pass.
- Documentation states the exact API and model limits.
- Every measured claim has a generated receipt from the candidate commit.
- The source archive builds from a clean checkout.

## 0.2.0: measured local server

Implemented:

- Add chunked CUDA prefill for F16 and Q8 KV caches.
- Add Llama graph decode and an independent F16 quality oracle.
- Add proof-gated SM89 execution plans for Qwen3 8B and Llama 3.2 1B.
- Add a reproducible scheduled-server study.

Gate:

- Dense Qwen3 and Llama 3 pass independent CPU and CUDA differentials.
- Quantized block decoders pass exhaustive scalar comparisons.
- The sampler passes its ten-million-case FP64 differential.
- Prefill quality is linked to an independent oracle receipt.
- The release archives build twice with identical checksums.
- Extracted archives pass manifest, linkage, and execution checks.

## 0.3.0: batched service

Candidate work:

- Replace quantum interleaving with measured continuous batching.
- Add paged KV allocation and explicit backpressure.
- Preserve the documented OpenAI subset.

Gate:

- Concurrent transcripts match isolated execution.
- Cancellation overshoot and KV reservation remain bounded.
- Throughput and latency claims include workload traces and quality receipts.

## Later work

AMD, Metal, Vulkan, MoE, vision, and multi-GPU execution remain research
projects. A backend enters the roadmap only with an independent correctness
oracle and available test hardware.
