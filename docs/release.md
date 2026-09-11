# Release gate

Run each gate on the candidate source tree. Keep the GPU idle except for the
command under test. Earlier measurements remain valid only when the runtime
and study inputs are unchanged. `receipts/source-inputs.json` records their
SHA-256 values. The gate checks file contents and additions or removals.

## Static and CPU gates

```sh
scripts/check-public-tree.sh
taskset -c 16-31 scripts/test-check-public-tree.sh
taskset -c 16-31 python3 -m unittest discover -s tests
taskset -c 16-31 cargo +1.92 test --workspace --locked
taskset -c 16-31 cargo +1.92 clippy --workspace --all-targets --locked -- -D warnings
taskset -c 16-31 cargo +1.92 fmt --all --check
taskset -c 16-31 scripts/check-rust-complexity.sh
taskset -c 16-31 cargo +1.92 deny check
RUSTDOCFLAGS='-D warnings' taskset -c 16-31 cargo +1.92 doc --workspace --no-deps --locked
```

The complexity gate requires `rust-code-analysis-cli 0.0.25`. It covers Rust,
Python, C, C++, and CUDA source, including tests and oracle adapters. Every
function must score at most 10. Review shell functions manually.

Before editing, preserve a source baseline. Set `LEONE_COMPLEXITY_BASELINE` to
that directory when running the complexity gate to report the LOC change.
Review duplication, dead code, unnecessary wrappers, and abstraction boundaries.
Run multiple prose passes using [the writing style](writing-style.md).

## CUDA gates

```sh
taskset -c 16-31 cargo +1.92 test --release --locked -- --ignored --test-threads=1
```

The ignored tests cover CUDA scalar differentials and exact batched-to-isolated
streams. Run the model-specific session, scheduler, fork, hibernation, and
correctable checks for each claimed model family.

## Batched-service gate

Pin the study to physical performance cores:

```sh
taskset -c 0-3,12-15 scripts/study-batched-service.sh \
  models/Qwen3-8B-Q4_K_M.gguf \
  plans/qwen3-8b-sm89.json \
  "$(jq -r '.executions.leone_q4.quality.receipt' receipts/quality-concurrent-service.json)" \
  receipts/batched-service-study.json
```

The study runs five repetitions. Every run must meet all conditions:

- Batched transcripts match the batch-limit-one baseline.
- Aggregate completion throughput is higher with batching.
- P95 completion latency is lower with batching.
- The server accepts a request after a forced client disconnect.

The checked-in study is one compact JSON record. Do not retain temporary HTTP
responses, server logs, or per-run files.

## Comparative and client gates

Run exploratory calibration before setting the frozen manifest's `freeze_status`
to `frozen`. Keep the workload fixed after that change.

Use a clean release build for quality and service measurements:

```sh
taskset -c 16-31 cargo +1.92 build --release --locked -p leone-cli
mkdir -p target/measured
cp target/release/leone target/measured/leone
LEONE_BINARY=target/measured/leone LEONE_QUALITY_WRITE_RECEIPTS=1 \
  taskset -c 16-31 scripts/quality-concurrent-service.sh \
  models/Qwen3-8B-Q4_K_M.gguf models/Qwen3-8B-BF16.gguf \
  corpus/quality-v03.txt 2400 512 receipts/quality-concurrent-service.json cuda
LEONE_BINARY=target/measured/leone LEONE_QUALITY_WRITE_RECEIPTS=1 \
  taskset -c 16-31 scripts/quality-concurrent-service.sh \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf models/Llama-3.2-1B-Instruct-f16.gguf \
  corpus/quality-v03.txt 2400 512 receipts/quality-llama-v03.json cuda
taskset -c 0-3,12-15 scripts/study-concurrent-service.sh \
  --manifest benchmarks/concurrent-service-frozen.json \
  --leone-binary target/measured/leone \
  --output receipts/concurrent-service-study.json
scripts/check-openai-client.sh target/measured/leone \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf receipts/openai-client-check.json
python3 scripts/check-release-evidence.py
python3 scripts/render-concurrent-evidence.py
python3 scripts/plot-concurrent-evidence.py
```

The frozen comparison requires complete outcomes, matching prompt token counts,
resident progress traces, physical memory samples, and linked quality records.
Both engines use the same model and chat template. llama.cpp may decode all slots;
Leone dispatches at most one or four requests. A comparative win is not a gate.

The OpenAI Python client environment is defined in [client check](client-workflow.md).
The plot uses the pinned package in `scripts/plot-requirements.txt`.
Generated receipts are append-only. Use a new output path for another run.

## Archive gate

```sh
LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh
```

Extract each archive into an empty directory. Check `MANIFEST.sha256` and the
packaged plans. Check the binary's build identity and dynamic linkage. Build both archives twice
from the same source tree. Their checksums must match.
Run `scripts/check-public-tree.sh <directory>` on each extracted archive after
all reports and packages are generated.

Package manifests identify files by SHA-256. Compatibility metadata describes
capabilities without repeating release labels. Receipt schema identifiers remain
versioned for decoding. The source manifest identifies matching reproduction
inputs after history changes. Dependency pins and protocol paths retain their
required identifiers. Audit these uses when adding machine metadata.

The full local gate is:

```sh
scripts/check-release.sh
```

The command does not tag, push, install outside a temporary directory, or
publish a crate.

## Decision

Ship only if every applicable gate passes. If a gate fails, fix the candidate
and rerun the full gate.
