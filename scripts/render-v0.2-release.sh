#!/usr/bin/env bash
set -euo pipefail

output=${1:-docs/v0.2-release.md}
prefill=receipts/raw/v0.2-prefill-study.json
qwen_quality=receipts/2026-08-27T11:59:34Z-quality-e5c41c26.json
llama_quality=receipts/2026-08-27T12:21:08Z-quality-3e668ae3.json
server=receipts/raw/v0.2-live-server-study.json
correctable=receipts/correctable-20260827T124241Z.json
qwen_plan=plans/v0.2-qwen3-8b-sm89.json
llama_plan=plans/v0.2-llama3.2-1b-sm89.json
long_16k=receipts/2026-08-27T12:00:45Z-runtime-38be3405.json
long_32k=receipts/2026-08-27T12:02:38Z-runtime-694172f6.json

for input in "$prefill" "$qwen_quality" "$llama_quality" "$server" "$correctable" "$qwen_plan" "$llama_plan" "$long_16k" "$long_32k"; do
  if [[ ! -f $input ]]; then
    echo "release input does not exist: $input" >&2
    exit 2
  fi
done

baseline_commit=$(jq -er '.revisions.baseline' "$prefill")
candidate_commit=$(jq -er '.revisions.candidate' "$prefill")
if [[ ! $baseline_commit =~ ^[0-9a-f]{40}$ || ! $candidate_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "prefill study revisions must be full commit IDs" >&2
  exit 2
fi
for role in candidate-f16 candidate-q8; do
  receipt=$(jq -er --arg role "$role" '.inputs[] | select(.role == $role) | .path' "$prefill")
  expected_sha=$(jq -er --arg role "$role" '.inputs[] | select(.role == $role) | .sha256' "$prefill")
  if [[ $(sha256sum "$receipt" | cut -d' ' -f1) != "$expected_sha" ]]; then
    echo "prefill study input digest does not match: $receipt" >&2
    exit 2
  fi
  if [[ $(jq -er '.workload.engine.git_commit' "$receipt") != "$candidate_commit" ]]; then
    echo "prefill study candidate does not match: $receipt" >&2
    exit 2
  fi
done

f16_base=$(jq -r '.results.f16.baseline.ttft_ms | @text' "$prefill")
f16_candidate=$(jq -r '.results.f16.candidate.ttft_ms | @text' "$prefill")
f16_reduction=$(jq -r '.results.f16.ttft_reduction_percent | @text' "$prefill")
q8_base=$(jq -r '.results.q8.baseline.ttft_ms | @text' "$prefill")
q8_candidate=$(jq -r '.results.q8.candidate.ttft_ms | @text' "$prefill")
q8_speedup=$(jq -r '.results.q8.ttft_speedup | @text' "$prefill")
qwen_kld=$(jq -r '.metrics.kld.mean | @text' "$qwen_quality")
qwen_top1=$(jq -r '.metrics.top1_agreement * 100 | @text' "$qwen_quality")
llama_kld=$(jq -r '.metrics.kld.mean | @text' "$llama_quality")
llama_top1=$(jq -r '.metrics.top1_agreement * 100 | @text' "$llama_quality")
server_rate=$(jq -r '.scheduled.aggregate_completion_tok_s | @text' "$server")
server_p95=$(jq -r '.scheduled.total_ms.p95 | @text' "$server")
server_fairness=$(jq -r '.scheduled.latency_fairness_ratio | @text' "$server")
correctable_repeat=$(jq -r '.runtime.repeat_rich_geometric_mean_speedup | @text' "$correctable")
correctable_all=$(jq -r '.runtime.all_suite_geometric_mean_speedup | @text' "$correctable")
correctable_worst=$(jq -r '.runtime.worst_suite_speedup | @text' "$correctable")
ttft_16k=$(jq -r '.results.ttft_ms.median | @text' "$long_16k")
ttft_32k=$(jq -r '.results.ttft_ms.median | @text' "$long_32k")
workspace_16k=$(jq -r '.notes[] | select(startswith("Prefill workspace:")) | capture("Prefill workspace: (?<bytes>[0-9]+) bytes").bytes' "$long_16k")
workspace_32k=$(jq -r '.notes[] | select(startswith("Prefill workspace:")) | capture("Prefill workspace: (?<bytes>[0-9]+) bytes").bytes' "$long_32k")

mkdir -p "$(dirname "$output")"
cat >"$output" <<EOF
# Leone v0.2.0

Leone v0.2.0 makes the research engine usable as a measured local server. It adds chunked CUDA prefill, Llama graph execution, proof-gated execution plans, and a second independent quality oracle.

The primary target remains one NVIDIA RTX 4090 at batch 1. This release does not claim portability beyond SM89 and the tested GGUF models.

## Evidence summary

| Subject | Result | Evidence |
| --- | ---: | --- |
| Qwen 3 8B F16 KV, 4K TTFT | ${f16_base} ms to ${f16_candidate} ms (${f16_reduction}% lower) | [prefill study](../receipts/raw/v0.2-prefill-study.json) |
| Qwen 3 8B Q8 KV, 4K TTFT | ${q8_base} ms to ${q8_candidate} ms (${q8_speedup}x) | [prefill study](../receipts/raw/v0.2-prefill-study.json) |
| Qwen 3 8B Q4_K_M vs BF16 | mean KLD ${qwen_kld}, top-1 ${qwen_top1}% | [quality receipt](../receipts/2026-08-27T11:59:34Z-quality-e5c41c26.json) |
| Llama 3.2 1B Q4_K_M vs F16 | mean KLD ${llama_kld}, top-1 ${llama_top1}% | [quality receipt](../receipts/2026-08-27T12:21:08Z-quality-3e668ae3.json) |
| Four live Llama clients | ${server_rate} completion tok/s, ${server_p95} ms p95 completion, ${server_fairness} fairness ratio | [server study](../receipts/raw/v0.2-live-server-study.json) |
| Qwen correctable controller | ${correctable_repeat}x repeat-rich, ${correctable_all}x all-suite, ${correctable_worst}x worst suite | [correctable receipt](../receipts/correctable-20260827T124241Z.json) |

The server study uses four identical nine-token responses. It checks exact scheduled-to-isolated transcript agreement and recovery after a forced disconnect. It is not a long-response saturation study.

The correctable study runs on the required performance-core set. The failed E-core study remains in the evidence archive and does not satisfy the release gate.

## Long-context workspace

| Context | Median TTFT | Prefill workspace | Quality |
| --- | ---: | ---: | --- |
| 16K | ${ttft_16k} ms | ${workspace_16k} bytes | unverified |
| 32K | ${ttft_32k} ms | ${workspace_32k} bytes | unverified |

These runs establish memory behavior on the local RTX 4090. They do not reuse the 4K quality result.

## Execution plans

The tuner searches eager and graph decode, F16 and Q8 KV, and prefill chunk sizes. A candidate must preserve the exact transcript and improve median request time by at least 1%. It must also pass a one-sided sign test and memory and power limits.

The checked-in plans are:

- [Qwen 3 8B SM89](../plans/v0.2-qwen3-8b-sm89.json)
- [Llama 3.2 1B SM89](../plans/v0.2-llama3.2-1b-sm89.json)

Load a plan explicitly:

\`\`\`sh
leone chat -m models/Qwen3-8B-Q4_K_M.gguf --plan plans/v0.2-qwen3-8b-sm89.json
leone serve -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --plan plans/v0.2-llama3.2-1b-sm89.json
\`\`\`

Leone rejects a plan when its model hash, backend, compute capability, receipt hash, or selected candidate does not match.

## Reproduce the independent Llama oracle

Download the exact F16 GGUF whose SHA-256 appears in the quality receipt. Then run:

\`\`\`sh
taskset -c 16-31 scripts/build-llama-oracle.sh
taskset -c 16-31 scripts/run-llama-oracle.sh \\
  models/Llama-3.2-1B-Instruct-f16.gguf \\
  receipts/raw/v0.2-llama.tokens.bin \\
  receipts/raw/v0.2-llama-f16-oracle.f32 \\
  receipts/raw/v0.2-llama-oracle-manifest.json
\`\`\`

The adapter uses llama.cpp at the commit recorded in the manifest. It disables GPU model devices and writes row-major FP32 logits for the shared token IDs.

## Limits

- CUDA is the only accelerated backend.
- Qwen 3 and Llama 3.2 are the only release-gated architectures.
- The checked-in plans target SM89 and exact model hashes.
- The 16K and 32K studies report \`quality: unverified\`.
- The OpenAI-compatible server is a stable subset. Unsupported fields return an error.
- Model files and full logit artifacts are not part of the source archive.
EOF

echo "$output"
