# Roadmap

Evidence gates control each release. Dates only control priority.

## 0.1.0: coherent research preview

Gate:

- Dense Qwen3 and Llama 3 pass independent CPU and CUDA differentials.
- Quantized block decoders pass exhaustive scalar comparisons.
- Session replay, fork, hibernation, cancellation, and scheduling gates pass.
- Documentation states the exact API and model limits.
- Every measured claim has a generated receipt from the candidate commit.
- The source archive builds from a clean checkout.

## 0.2.0: measured single-stream efficiency

Candidate work:

- Remove repeated RMSNorm reductions in grouped output projection.
- Reduce host synchronization in sampling and token upload.
- Cache stable CUDA launch and cuBLASLt metadata.
- Improve long-context prefill and `q8` KV execution.

Gate:

- Each optimization has an isolated baseline and candidate receipt.
- Numerical output stays within the existing differential bounds.
- Regressions outside the target workload are stated.

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
