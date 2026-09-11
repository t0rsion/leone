# Changelog

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
