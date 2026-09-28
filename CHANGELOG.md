# Changelog

## 0.4.0 - 2026-09-28

### Added

- Add native Metal kernels and scalar differential tests for Apple silicon.
- Share immutable KV segments between forked generation sessions.
- Add shared parent memory budgets for unified memory.
- Report macOS process memory and system memory estimates through the native bridge.
- Add exhaustive quantized block field tests against the scalar decoder.
- Add macOS package construction and offline archive checks.
- Include the current runtime source manifest in both native packages.
- Add an opt-in runtime driver for four shared-prefix attention paths.
  Its raw records report unverified quality until an independent oracle is attached.
- Freeze runtime numerical criteria from calibration and compare against a pinned
  CPU BF16 oracle. Timing labels flag differences for review and make no speed claim.
- Add `realized_prompt_tokens` to `/debug/service-metrics`. It sums the reused,
  replayed, and computed prompt tokens when requests end with a recorded replay.
  `prefix_reuse_sources.reused_tokens` stays the plan-time credit.

### Changed

- Tile Metal quantized prefill across tokens. Use FP32 SIMD-group matrix operations
  on supported devices, with a token-tile fallback.
- Use decode-equivalent chunked prefill for warm Metal sessions with F16 KV.
- Reduce Metal attention synchronization while preserving the reduction order.
- Batch Metal submissions and fence before host access or buffer release.
- Parallelize Metal argmax while preserving ties and non-finite value handling.
- Skip intermediate output projections during sequential prefill.
- Enable hardware SHA-256 for Apple silicon executables. Keep full-file identity checks.
- Reduce each plain Metal RMSNorm row in one threadgroup. The FP32 summation order changes.
- Make CUDA optional in the CLI build.
- Reuse bounded CUDA prefill scratch and retry allocation failures after release.
- Keep shared decode active during mixed prefill ticks.
- Select graph execution from the active backend's capabilities.
- Replace `AdaptiveObservation::emitted_tokens` with `produced_tokens` so a
  partial stop does not bias the measured cost of a speculative round.
- Remove the unused `CorrectableObservation::emitted_tokens` field.
  Correctable round costs use `produced_tokens`.
- Borrow `HibernatedSession` in `Runtime::wake_session`. Failed wake leaves the
  snapshot available for another attempt.
- Give automatic prefix reuse a fresh session ID and separate request state.
- Treat OpenAI `user` as caller metadata. Use `leone_session` or
  `X-Leone-Session` for explicit continuation.
- Replace `Penalties::window: usize` with typed `PenaltyWindow`. Zero disables
  counting penalties, and an omitted OpenAI window uses full history.
- Add the `official` chat template mode with pinned Llama and Qwen fixtures.
  Named and required tool choices filter definitions or add Leone instructions,
  so those requests extend the upstream templates.

### Fixed

- Restore blocking mode on accepted sockets before reading requests on macOS.
- Report prefill as `reused` when the request performs no prefill work.
- Reserve session metadata for admitted tokens instead of the full context limit.
- Tokenize Qwen reasoning markers in retained legacy chat history.
- Preserve CUDA constructor allocations when installing service memory budgets.
- Validate run options and compiled backends before downloading a model.
- Report model download progress and bounded, URL-redacted failures.
- Check the selected backend during every doctor invocation.
- Inspect metadata-only GGUF architectures without requiring a text runtime.
- Preserve verified model identity across downloads and runtime loading.
- Count decode telemetry only after successful dispatch.
- Bound session archive serialization capacity and recovered blob size by the
  canonical JSON schema. Return `SessionArchiveError::Buffer` if reservation fails.
- Derive Metal service capacity from the backend memory contract.
- Implement position-major Metal matrix operations for opt-in research batches.
- Include the embedded Metal shader identity in `--build-info` for quality checks.
- Hash tokenizer model files in bounded chunks and publish quality tasks without replacement.
- Preserve resident sessions when the macOS checkpoint lookup fails before retirement.
- Preserve failed batching studies and reject receipt replacement, stale binaries, and occupied ports.
- Require `ss` for batching studies and verify that the measured process owns the listener.
- Include batching samples, workload identity, and quality links in the release evidence gate.
- Reject archive installation on a mismatched host before changing the destination.
- Name the CUDA generation record and the Metal shader, doctor, and eval output
  files in the planned v0.4 quality records. Offline concurrent-study checks
  hash each recorded source file that the archive holds and no longer require the
  source crates.
- Add a branching study record for each backend to the planned v0.4 evidence. The
  release check runs canonical quality validation on the record each study binds,
  then the study harness offline. It pins the harness, the history helpers, and
  the shell wrappers in the source inputs.

## 0.3.0 - 2026-09-11

### Added

- Verify measured source contents independently of development history.

- Batch compatible request rows into shared CUDA decode operations.
- Add page-rounded KV admission and typed HTTP overload reasons.
- Add exact Qwen3 and Llama batched-to-isolated CUDA differentials.
- Add a five-run concurrent service study with a batch-limit-one baseline.
- Add resumable prefill and typed progress without emitted tokens.
- Track physical allocation lifetimes by buffer class.
- Add graph lifecycle checks for arrivals, cancellation, forks, and host wake.
- Add a mixed streaming comparison with pinned llama.cpp and a common oracle.
- Add an OpenAI Python client check for streaming, forks, and recovery.
- Add `--build-info` with source, target, profile, and dirty-tree status.

### Changed

- Disable adaptive speculation by default for server requests.
- Record comparative streaming evidence separately from the existing batching
  gate and execution-plan records.
- Apply whole-codebase complexity, abstraction, and prose review to each release.
- Audit personal paths and release references after package generation.
- Identify package contents through SHA-256 manifests. Remove duplicate package
  build records and release labels from the compatibility descriptor. Receipts
  retain each measurement's source identity.

### Removed

- Remove historical performance packets from the source tree.

## 0.2.0 - 2026-09-04

### Added

- Add chunked CUDA prefill for F16 and Q8 KV caches.
- Add proof-gated SM89 plan search, inspection, and loading.
- Add Llama graph decode and an independent llama.cpp F16 quality oracle.
- Add a reproducible live scheduled-server study.

### Changed

- Use typed prefill chunk sizes in generation and serving.
- Permit release plans to select graph mode, KV format, and prefill chunk size.

### Fixed

- Fall back to ordinary verifier GEMV when a model head width cannot prepare attention output.
- Resolve installed execution-plan receipts outside the current directory.
- Return an OpenAI-shaped `400` response for invalid scheduled chat requests.

## 0.1.0

First public research preview.

- Add dense Qwen3 and Llama 3 GGUF inference.
- Add CPU reference and handwritten CUDA backends.
- Add checked `Q4_K` and `Q6_K` import and execution.
- Add tiled prefill, graph-replayed decode, and typed KV cache storage.
- Add FP64 sampling oracles and exact speculative verification.
- Add persistent, hibernated, and forked sessions.
- Add bounded scheduling and an OpenAI-compatible chat subset.
- Add signed response receipts and linked runtime and quality receipts.

No earlier version was public.
