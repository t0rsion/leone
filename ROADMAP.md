# Roadmap

Evidence gates control each release. Dates control priority.

## Current release: responsive concurrent sessions

Scope:

- Batch compatible decode rows into shared CUDA matrix operations.
- Yield between typed prefill chunks so resident decode can progress.
- Keep attention, sampling, cancellation, and session state separate.
- Account physical allocations separately from page-rounded KV admission.
- Test a named OpenAI client workflow with streaming and session forks.
- Compare mixed streaming workloads with pinned llama.cpp.

Gates:

- Resumable prefill matches uninterrupted execution with the same chunk sizes.
- Graph-mode Qwen3 and Llama streams match isolated Leone under matched settings.
- Fork, host wake, cancellation, and suspended prefill pass lifecycle checks.
- Tracked allocation classes remain bounded after repeated lifecycle operations.
- The five-run batching study passes its existing throughput and latency gates.
- The comparative study records all outcomes and complete input provenance.
- Both comparison engines have quality records against a common oracle independent of Leone.
- The client workflow and all static, CPU, CUDA, complexity, privacy,
  documentation, and archive gates pass on the candidate.

A comparative speed claim requires a measured advantage. A losing comparison
does not prevent publication of an otherwise passing research release.

## Next release: service hardening

- Replace page-rounded admission with a pooled physical KV page allocator.
- Reduce graph recapture when profiling shows a material cost.
- Batch compatible prefill work when measured workloads benefit.
- Add per-client quotas, request deadlines, and structured server metrics.
- Add an authenticated deployment profile behind a documented proxy contract.

## Next backend: Apple silicon

Metal is the next portability target because Apple silicon provides a clear
consumer-hardware test case. It needs a backend implementation, scalar
differentials, tokenizer parity, packaging, and measured hardware access.

Prepare a runtime build without CUDA before adding Metal. The first macOS package
targets Apple silicon. Intel Mac acceleration is outside that scope.

## Model coverage and later work

Select the next model family from concrete user workloads and available oracles.
One complete architecture takes priority over metadata recognition for many
architectures. A GPU-resident MoE implementation and CPU expert offload have
separate correctness and performance gates.

AMD, Vulkan, MoE, vision, and multi-GPU execution remain research projects. A
backend enters the release plan only with an independent correctness oracle and
available test hardware.

Windows packaging needs its own integration checks. Disk KV, new quantization,
and speculative batching require separate measured proposals.

## Stable public contracts

Stable APIs require tested upgrades and recovery, reproducible packages, and
a maintained hardware support matrix.

## Every release

Review the whole codebase before recording release evidence:

- Measure function complexity. Leave scores 1 through 5 alone. Review scores
  6 through 10 when the function changes. Refactor scores above 10, and split
  scores above 15. The automated ceiling is 10.
- Remove dead code, repeated logic, unnecessary wrappers, and abstraction leaks.
  Record the source LOC change. Preserve tests and numerical contracts.
- Run multiple prose passes using [the writing style](docs/writing-style.md).
  Keep comments that explain constraints, invariants, or reasons.
- After generating reports and packages, audit source and extracted artifacts
  for personal information and internal paths. Keep release history in the
  changelog. Preserve the machine metadata needed to reproduce evidence.

A smaller source count does not justify weaker checks or compressed formatting.
