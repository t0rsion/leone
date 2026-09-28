# Roadmap

Evidence gates control each release. Dates control priority.

## Current release: shared sessions on CUDA and Metal

Scope:

- Share immutable KV prefixes between concurrent sessions. Append to private
  state, and copy shared state only when a write requires it.
- Bound physical allocation ownership, including weights, loading peaks, KV,
  scratch capacity, retained pools, and host state.
- Select segmented storage or a page pool from measured waste and admission
  limits. A page pool is not a prerequisite for shared prefixes.
- Keep socket I/O off the inference thread. Bound connections and output
  queues, and test cancellation, deadlines, slow clients, and recovery.
- Keep compatible CUDA decode rows batched during mixed prefill ticks.
- Build without CUDA. Run native GPU chunked prefill and decode on Apple
  silicon through the same backend contract.
- Fix command help, model download recovery, diagnostics, and documented
  request semantics. Verify model templates against independent fixtures.
- Test a named OpenAI client workflow with streaming, tools, and session forks.

Release checks:

- Fix concrete correctness, memory, and responsiveness failures in normal use.
- Run focused independent oracles for changed kernels and runtime behavior.
  Reuse completed checks when their inputs are unchanged.
- Run the named Metal client workflow at the default request deadline.
- Record one simple performance run. Limit claims to that workload and device.
- Check the native builds, runtime archives, complexity, public prose, and PII.
- Keep receipt schemas append-only and the documented OpenAI subset stable.

Comparative service studies, repeated benchmarks, common-oracle quality
campaigns, and complete research evidence archives are optional research work.
Research tools remain experimental. Unmeasured results carry no speed or
quality claim.


## Performance and research

Profile prefill weight expansion, scratch allocation, graph recapture, attention,
logit transfers, and sampling before selecting kernel changes. Record current
single-stream and prefill baselines before measuring improvements.

The research experiment tests fixed-reduction shared-prefix attention at small
fan-out. It separates shared storage from shared KV reads and compares both
against per-row execution. Include unrelated prompts and short contexts.

Freeze workloads, numerical tolerances, and practical effect criteria before
evaluation. Scope invariance to declared inputs and execution history on one
backend. Separate estimated traffic from measured traffic. Compare prior art
before claiming a contribution. A negative research result does not waive a
product gate or justify a speed claim.

## Backend coverage

CUDA remains the measured NVIDIA backend. The macOS package targets tested
Apple silicon hardware. Intel Mac acceleration is outside this scope.

Metal requires scalar differentials, tokenizer parity, packaging, and measured
model fit. Host copies on unified memory do not
count as physical memory savings. CUDA graph and verification parity are not
requirements for the first Metal package.

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
  Keep comments that explain constraints, invariants, or reasons.
- After generating reports and packages, audit source and extracted artifacts
  for personal information and internal paths. Keep release history in the
  changelog. Preserve the machine metadata needed to reproduce evidence.

A smaller source count does not justify weaker checks or compressed formatting.
