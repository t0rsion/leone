# Changelog

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

### Evidence

- See [docs/v0.2-release.md](docs/v0.2-release.md). The report is generated from checked-in receipts.

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
