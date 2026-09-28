# Correctness oracles

A result ships only when an implementation-independent oracle can reject it.
The production path and oracle must not share the operation under test.
Script and fixture paths name files in a source checkout.

## Quantized blocks

Each supported GGUF block format has a scalar decoder. Tests enumerate every
payload byte code for the format and compare full decoded blocks. Partial blocks
return a typed error.

The CPU, CUDA, and Metal quantized import paths compare against the same scalar
contract. Parser recognition of a GGUF type does not make it a runtime format.

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

## Staged Metal quality

The staged Metal quality path does not load the BF16 oracle. Export the CPU
oracle and the shared token stream on a host with enough memory:

```text
scripts/export-quality-oracle.sh \
  models/Qwen3-8B-BF16.gguf \
  models/Qwen3-8B-Q4_K_M.gguf \
  "Q4_K - Medium" \
  corpus/quality-v03.txt target/quality-tokens.bin 512 target/qwen3-oracle
```

Existing CPU logits and their run manifest can be appended after the output
directory when the staged adapter artifact is present.

Copy the export directory to the Mac. Set `LLAMA_CPP_DIR` to a clean checkout
at the commit recorded in `external/PINNED`. Set `LEONE_BINARY` to a clean
release build that reports its Metal shader identity in `--build-info`.
Set `LEONE_METAL_SHADER_PATH` to that build's `crates/leone-metal/metal/leone.metal`
source file, then import both Metal subjects:

```text
LLAMA_CPP_DIR=/path/to/pinned/llama.cpp \
LEONE_BINARY=/path/to/release/leone \
LEONE_METAL_SHADER_PATH=/path/to/source/crates/leone-metal/metal/leone.metal \
LEONE_EXPECTED_SOURCE_COMMIT=<Leone commit> \
scripts/import-metal-quality.sh \
  target/qwen3-oracle models/Qwen3-8B-Q4_K_M.gguf target/qwen3-metal
```

The import runs `leone doctor --backend metal` and `leone eval --backend metal`
with the shared token stream and window. It also runs the pinned llama.cpp
Metal comparator. The stage records the model, token, window, executable,
source commit, device record, and output hash for each subject. The native
Leone record carries the clean release build metadata, F16 KV cache setting,
prefill path and chunk, and the source shader hash. Set
`LEONE_METAL_PREFILL_CHUNK` to record chunked prefill. The default records
sequential prefill for bring-up checks. Release comparisons use a declared
chunked prefill setting, such as `LEONE_METAL_PREFILL_CHUNK=128`, after the
native candidate covers that path. Doctor and eval text keep their final
newline in the recorded hash.
Oracle, comparator, native, and statistics source identities are recorded
separately. A matching export can be reused when its model, token, window,
corpus, and adapter hashes remain unchanged.

Copy the Metal stage back to the comparison host. The comparison command
checks model, corpus, token, adapter, and pinned llama.cpp hashes before it
writes two full precision linked quality receipts. Create each exact-answer
task from prompt bytes, needle bytes, and an independent token artifact from
the pinned `llama-tokenize` binary. Create the task before importing Metal.
The answer row must lie at the declared context depth. The helper rejects
windows below 4,096 tokens.

```text
scripts/write-quality-task.py \
  target/qwen3-oracle Qwen3-8B 4095 \
  target/qwen3-long-context-task.json \
  --prompt-file target/qwen3-prompt.txt \
  --needle-file target/qwen3-needle.txt \
  --tokenizer-executable /path/to/pinned/llama-tokenize \
  --tokenizer-model models/Qwen3-8B-BF16.gguf
```

The prompt and needle bytes must reconstruct the staged corpus. Run the same
task for the Llama stage with its own model and tokenizer identities. The checker compares
the answer against the pinned CPU oracle, the pinned llama.cpp Metal
comparator, and native Leone Metal logits.

The helper runs the supplied tokenizer and records its executable hash. The
llama.cpp checkout and commit are operator-attested provenance; the hash does
not prove the checkout source. The import producer needs only the expected native
source commit. The comparison producer below requires every trusted assignment
and exits with status 2 when one is missing. The native source commit and the
pinned llama.cpp oracle commit remain separate identities.

```text
LEONE_BINARY=/path/to/release/leone \
LEONE_RECEIPT_VERIFIER=/path/to/release/leone-receipt-verify \
LEONE_EXPECTED_SOURCE_COMMIT=<Leone commit> \
LEONE_EXPECTED_PLATFORM=darwin-arm64 \
LEONE_EXPECTED_TARGET=aarch64-apple-darwin \
LEONE_EXPECTED_STATISTICS_TARGET=x86_64-unknown-linux-gnu \
LEONE_EXPECTED_BACKEND=metal \
LEONE_EXPECTED_ADAPTER_SHA256=<adapter SHA-256> \
LEONE_EXPECTED_MODEL_FAMILY=Qwen3-8B \
LEONE_EXPECTED_MODEL_SHA256=<subject model SHA-256> \
LEONE_EXPECTED_ORACLE_MODEL_SHA256=<full-precision model SHA-256> \
LEONE_EXPECTED_CORPUS_SHA256=<corpus SHA-256> \
LEONE_QUALITY_GENERATION_RECORD=target/qwen3-metal-generation.json \
LEONE_EXPECTED_SAMPLE_CONTRACT=linspace-inclusive-v1:128+task-rows \
LEONE_EXPECTED_METRIC_FAMILY=kld \
scripts/compare-metal-quality.sh \
  target/qwen3-oracle target/qwen3-metal \
  models/Qwen3-8B-BF16.gguf models/Qwen3-8B-Q4_K_M.gguf \
  corpus/quality-v03.txt target/qwen3-long-context-task.json \
  target/qwen3-metal-quality.json
```

The command stores the task manifest and result beside the comparison. The
expected source commit binds the native Leone candidate. The native target is
Darwin ARM64, while the statistics target names the Linux host that runs the
comparison binary. The pinned llama.cpp commit remains the independent oracle
and comparator identity recorded in the stages. The generation record is
written after sample extraction and stays outside the comparison. Offline
validation rechecks every sampled task logit row and the exact answer token
from the packaged token stream.

The stage records use stable artifact names and SHA-256 values. They record the
selected backend and device identity. The comparison copies stage manifests,
token inputs, adapter source, and native build records beside the manifest. It
stores deterministic sampled logit rows with source row hashes. Full logits stay
generation-only artifacts. KLD compares the sampled rows for `llama.cpp-metal`
and `leone-metal` separately. Sampling uses 128 inclusive linear rows, plus
every exact-answer row. Each sample stores row-major little-endian `f32` bytes,
the source row index, and a SHA-256 row hash. The llama.cpp result is
comparator evidence. It does not certify the native Leone backend. The result
makes no cross-device bitwise claim and does not certify concurrent service
numerics.

Use `python3 scripts/validate-quality-stage.py comparison PATH VERIFIER` with the
trusted source, platform, target, backend, adapter, model, and metric options as
the offline validator. `VERIFIER` is `leone-receipt-verify`. The command dispatches
receipt parsing to `VERIFIER quality RECEIPT`, then checks source identities,
packaged stage files, sample row hashes, task rows, receipt bounds, and recomputed
KLD statistics.

```text
python3 scripts/validate-quality-stage.py comparison PATH VERIFIER \
  --source-commit <Leone commit> --platform darwin-arm64 \
  --target aarch64-apple-darwin \
  --statistics-target x86_64-unknown-linux-gnu --backend metal \
  --adapter-path research/oracle/llama_logits.cpp \
  --adapter-sha256 <adapter SHA-256> --model-family Qwen3-8B \
  --model-sha256 <subject model SHA-256> \
  --oracle-model-sha256 <full-precision model SHA-256> \
  --corpus-sha256 <corpus SHA-256> \
  --sample-manifest-sha256 <sample manifest SHA-256> \
  --sample-contract linspace-inclusive-v1:128+task-rows --metric-family kld
```

Read the sample hash from the caller-owned generation record after the
comparison producer completes. Keep that record with the release evidence.
The offline command rejects a package whose sample manifest differs from this
hash. Full logits remain generation-only artifacts.
