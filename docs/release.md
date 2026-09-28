# Release checks

Run the required checks on the candidate source tree. Keep the GPU idle except
for the command under test. Earlier measurements remain valid only when the
runtime and study inputs are unchanged. `receipts/source-inputs-v04-prestudy.json`
records the runtime package inputs. Earlier evidence keeps its source manifests.
The source checks cover file contents and additions or removals.

Both runtime archives include this manifest. A prebuilt binary can retain its
original build identity after a source-history squash when the recorded input
contents and executable modes still match.

## Runtime release scope

The current scope covers focused product checks, native Linux CUDA and macOS
Metal builds, one simple performance run, one responsiveness check, complexity,
public-tree and PII checks, and runtime archive verification. Limit performance
claims to the measured workload and device. Keep
`packaging/release-evidence.v0.4.json` at `status: planned` while this scope is
in use.

Build the native binaries before the runtime archive checks:

```sh
taskset -c 16-31 cargo +1.92 build --release --locked -p leone-cli
cargo +1.92 build --release --locked -p leone-cli \
  --no-default-features --features metal
```

Use the [client workflow](client-workflow.md) for the responsiveness check.
It covers Metal streaming, tools, and session forks at the default deadline.
Reuse its completed result when the runtime inputs are unchanged.

For the simple Metal performance check, retain one generation and its runtime
footer:

```sh
target/release/leone generate --backend metal \
  -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p 'Explain how sharing an immutable prompt cache can help two independent conversations.' \
  -n 64 --seed 0 --debug-tokens generation-tokens.json \
  > generation.txt 2> generation.log
```

Record the device, model hash, and binary build information with the output.
The footer reports `quality: unverified`. This run makes no model-quality or
comparative performance claim.

The comparative service matrix, repeated batching studies, BF16 quality runs,
and complete common-oracle evidence archive remain optional research work.
Their receipts retain their recorded inputs and claims. They do not block a
runtime archive and do not make the evidence manifest complete.

## Source checks

Run regressions for the changed runtime paths and kernels. Reuse completed
checks when their inputs are unchanged. Run broader suites when a failure
shows that the affected scope is larger.

```sh
scripts/check-public-tree.sh
taskset -c 16-31 cargo +1.92 clippy --workspace --all-targets --locked -- -D warnings
taskset -c 16-31 cargo +1.92 fmt --all --check
taskset -c 16-31 scripts/check-rust-complexity.sh
RUSTDOCFLAGS='-D warnings' taskset -c 16-31 cargo +1.92 doc --workspace --no-deps --locked
```

The complexity gate requires `rust-code-analysis-cli 0.0.25`. It covers Rust,
Python, C, C++, and CUDA source, including tests and oracle adapters. Every
function must score at most 10. Review shell functions manually. Swift source
uses the pinned SwiftLint `0.65.1` `cyclomatic_complexity` rule. SwiftLint
counts `switch` cases in this check. Its metric is reported separately from
the Rust analyzer metric. The checker runs `swiftc -parse` before SwiftLint
to reject malformed files. The pinned portable binary comes from
<https://github.com/realm/SwiftLint/releases/download/0.65.1/portable_swiftlint.zip>
with SHA-256: `c1e429b0599cf1b516f369a2d9ec04eaf0e436f3c12b637df8851fa52ff694d0`.
Linux uses the static binary from
<https://github.com/realm/SwiftLint/releases/download/0.65.1/swiftlint_linux_amd64.zip>
with SHA-256: `caeed6f4a679c35539ffaf124f6c4ab4a8416917f7d8796279dc52b74026059d`.
Set `LEONE_SWIFTLINT_BIN` to the installed `swiftlint-static` path.
When Linux has no `swiftc`, the main gate runs the complexity rule and prints
`swift_syntax=unvalidated native_build_required=1`. Set
`LEONE_REQUIRE_SWIFT_SYNTAX=1` to reject that fallback. Release checks require
the native macOS syntax and complexity run.
Run `scripts/test-check-swift-complexity.sh` on macOS to exercise the Swift
fixtures. A gate run with tracked Swift files fails when the pinned SwiftLint
binary is unavailable.

Before editing, preserve a source baseline. Set `LEONE_COMPLEXITY_BASELINE` to
that directory when running the complexity gate to report the LOC change.
Review duplication, dead code, unnecessary wrappers, and abstraction boundaries.

## Optional backend and research evidence

The following sections describe optional evidence collection. They retain the
existing commands and receipt contracts for later research runs.

### CUDA evidence

```sh
taskset -c 16-31 cargo +1.92 test --release --locked -- --ignored --test-threads=1
```

The ignored tests cover CUDA scalar differentials and exact batched-to-isolated
streams. Run the model-specific session, scheduler, fork, hibernation, and
correctable checks for each claimed model family.

### Metal evidence

Run the Metal checks on Apple silicon:

```sh
cargo +1.92 build --release -p leone-cli --no-default-features --features metal
cargo +1.92 test --release -p leone-metal
LEONE_METAL_MODEL=models/Qwen3-8B-Q4_K_M.gguf \
  cargo +1.92 test --release -p leone-metal --test lifecycle qwen_metal_session_lifecycle -- --ignored --test-threads=1
LEONE_METAL_MODEL=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  cargo +1.92 test --release -p leone-metal --test lifecycle llama_metal_session_lifecycle -- --ignored --test-threads=1
```

The current `scripts/check-release.sh` command does not execute these Apple GPU
checks. The staged Metal quality workflow is documented in [the oracle contract](oracle.md).
The release manifest declares CUDA and Metal records. A Metal quality claim
requires checked-in Metal quality receipts and archive checks.

### Batched-service study

Pin the study to physical performance cores. The script exits with status 2,
before any repetition, when the output path exists:

```sh
study="receipts/$(date -u +%Y-%m-%dT%H-%M-%SZ)-batched-service-study.json"
taskset -c 0-3,12-15 scripts/study-batched-service.sh \
  models/Qwen3-8B-Q4_K_M.gguf \
  plans/qwen3-8b-sm89.json \
  "$(jq -r '.executions.leone_q4.quality.receipt' receipts/quality-concurrent-service.json)" \
  "$study"
scripts/render-release-evidence.sh docs/release-evidence.md "$study"
```

The study runs five repetitions. Every run must meet all conditions:

- Batched transcripts match the batch-limit-one baseline.
- Aggregate completion throughput is higher with batching.
- P95 completion latency is lower with batching.
- The server accepts a request after a forced client disconnect.

The study exits with status 2, before any server starts, unless
`LEONE_BINARY` (default `target/release/leone`) is a clean release CUDA build
whose source inputs equal the current tree. The study also exits with status 2
if a port it uses already has a listener. It exits with status 1 if a study
server exits before its measurements end. `ss` must report the listener on each
port under the process ID of the server the study started, so the binary must be
the listening process and not a wrapper. The receipt records the binary
SHA-256, its build information, and the GPU identity.

`LEONE_STUDY_SOURCE_MANIFEST` names a source manifest to check instead. A
manifest that lists evidence files must match the evidence files present, and
the run directories beside the output count as evidence when the output path
matches `receipts/v04-*`. `record-v04` creates its output file and refuses to
replace one that exists. To study against a manifest recorded before the study,
keep two manifests:

1. Record the first manifest at the frozen source commit under a name that does
   not match `receipts/v04-*`:

   ```sh
   python3 scripts/source_inputs.py record-v04 "$commit" \
     receipts/source-inputs-v04-prestudy.json
   ```

2. Run the study to a path outside `receipts/v04-*`, with the first manifest:

   ```sh
   LEONE_STUDY_SOURCE_MANIFEST=receipts/source-inputs-v04-prestudy.json \
     scripts/study-batched-service.sh MODEL PLAN QUALITY_RECEIPT \
     receipts/study-scratch.json
   ```

3. Move the receipt to `receipts/v04-linux-cuda-batched-service.json`.
4. Record the second manifest from the same commit. It lists the study and is
   the release manifest:

   ```sh
   python3 scripts/source_inputs.py record-v04 "$commit" \
     receipts/source-inputs-v04.json
   ```

The first manifest stays as recorded. The checker requires the study, binary,
and manifest commits to be equal.

The study writes one JSON record. Its `runs` field holds the five
unedited run records. Each run keeps the SHA-256 of every response and
transcript, the batch limit of each server, the SHA-256 of the prompt text, the
binary identity, and the GPU identity. The receipt does not retain signed
responses, and nothing checks a digest against a response body. Batch sizes
record command-line limits. Dispatch widths are unmeasured. After a failed
study, the completed run records, raw responses, and server
logs stay in `OUTPUT.runs.*`. A throughput or latency loss does not stop the
repetitions. The receipt stays at its path and the script exits with status 1.
Do not check in temporary HTTP responses, server logs, or per-run files.

Release evidence uses the Leone quality receipt that the CUDA quality comparison
embeds, `receipts/v04-linux-cuda-quality-comparison-qwen3-leone-quality.json`, as
the third argument. `scripts/check-batched-service.py` recomputes every ratio,
summary, and check from `runs`. It requires four clients, 64 tokens, five
repetitions, and the default prompt. It also requires the final source manifest,
the release binary hash, and a study that passes every condition above. It
rejects two byte-identical run records. It cannot detect distinct, consistent
forged runs.

### Branching-service study

The release evidence holds one branching study for each backend:
`receipts/v04-linux-cuda-branching-service.json` and
`receipts/v04-darwin-metal-branching-service.json`. Run the study with
`scripts/study-branching-service.sh`, as [the branching evidence
page](branching-service-evidence.md) describes. Each study binds the quality
comparison record that release validation checks for the same backend and model.
No second quality record or calibration-derived quality tolerance exists.

The release check does two things for each study. It runs `quality-stage-v1`
canonical validation on the exact record the study names, which recomputes the
quality metrics from the packaged samples. It then runs the study harness offline
in archive scope, with the packaged source manifest. The study harness only binds
the record. Its receipt says `quality_recomputation_status: "not_run_by_harness"`.
The check fails if the record's digest, backend, or model differs between the
study, the release record, and the packaged file, or if either quality sidecar is
not a packaged dependency.

The check also requires:

- The native executable hash of the quality record, not the statistics
  executable, equals the served Leone binary. It equals the binary of the
  same-backend client record.
- The source manifest pins the seven archive inputs: the harness, the two history
  helpers, the template generator, `scripts/linked_libraries.py`,
  `scripts/source_inputs.py`, and `fixtures/qwen3-legacy-chatml.jinja`.
- A history reexecution result that names the packaged receipt and the packaged
  expected pins, with a CPU-only server launch for every retained branch. The
  expected file must satisfy the history checker's schema and value rules. Its
  pins and the result's oracle identity must equal the oracle retained in every
  llama.cpp branch. The fields are the commit, executable, model, loaded
  library, template, policy, and vocabulary.
- Every Leone gate passes, and the Leone branch method is supported.

An offline archive cannot rerun a model, a server, or the tokenizer. The history
result is an unsigned record from collection. The check shows only
that it names this receipt and these pins. It does not show that the run
happened. The quality record is a common-model measurement from the evaluation
path, not the served schedule. The oracle shares the pinned llama.cpp code with
the peer engine. A complete manifest still requires a native study run.

### Comparative and client study

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
python3 scripts/check-release-evidence.py --leone-binary target/measured/leone \
  --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
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

## Required runtime archives

```sh
LEONE_BUILD_CPUSET=16-31 \
  scripts/package-release.sh --platform linux-x86_64 --runtime-only
```

On an Apple silicon runner, use:

```sh
scripts/package-release.sh --platform darwin-arm64 --runtime-only
```

The runtime-only commands build one runtime archive and skip the planned
evidence manifest. The default packaging command also stages an evidence
archive and requires a complete manifest.

Extract each archive into an empty directory. Check `MANIFEST.sha256` and the
packaged plans. Check the binary's build identity and dynamic linkage.
Run `scripts/check-public-tree.sh <directory>` on each extracted archive after
all reports and packages are generated.

Run `scripts/verify-release-archive.sh <archive>` for each runtime archive. It
checks the sidecar, manifest, package metadata, and public-tree rules. Evidence
archive verification is part of the deferred complete-evidence workflow. It
uses the trusted CPU-only `leone-receipt-verify` tool and does not compile code,
load models, query CUDA, or rerun a study.

Reproduction checks use `--mode reproduce` and explicit input paths when a
receipt records `<external>/...` artifact labels. Offline verification does not
inspect those paths.

Package manifests identify files by SHA-256. Compatibility metadata describes
capabilities without repeating release labels. Receipt schema identifiers remain
versioned for decoding. The source manifest identifies matching reproduction
inputs after history changes. Dependency pins and protocol paths retain their
required identifiers. Audit these uses when adding machine metadata.

## Deferred complete-evidence gate

The full local evidence gate is:

```sh
scripts/check-release.sh
```

The command does not tag, push, install outside a temporary directory, or
publish a crate. Its comparative, batching, quality, and archive evidence
checks remain deferred for the current runtime scope.

## Decision

Ship runtime archives after the required product, source, build, and archive
checks pass. Keep the evidence manifest planned until the optional
research records and their validators are complete.
