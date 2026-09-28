#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [OUTPUT [STUDY_RECEIPT]]" >&2
  exit 2
fi

output=${1:-docs/release-evidence.md}
study=${2:-receipts/batched-service-study.json}

# The study path becomes a Markdown link target, so the accepted alphabet
# excludes spaces, parentheses, and backslashes.
require_repository_path() {
  local path=$1
  local dot_segment='(^|/)\.\.?(/|$)'
  if [[ ! $path =~ ^[A-Za-z0-9_][A-Za-z0-9._/:-]*$ || $path =~ $dot_segment || $path = *//* ]]; then
    echo "release input must be a repository-relative path: $path" >&2
    exit 2
  fi
}

require_repository_path "$study"
if [[ ! -f $study || -L $study ]]; then
  echo "release input does not exist: $study" >&2
  exit 2
fi

# The Limits text below names the model, client count, and GPU. A study that
# does not record them cannot carry that text. The historical default receipt
# predates the hardware record and is the only accepted exception.
historical=false
if [[ $study = receipts/batched-service-study.json ]]; then
  historical=true
fi
jq -e --argjson historical "$historical" '
  .schema_version == "leone.batched-service-study.v1" and
  (.quality.path | type == "string") and
  .model.path == "models/Qwen3-8B-Q4_K_M.gguf" and
  .workload.concurrent_clients == 4 and
  ((.hardware | type == "object" and (.gpu_name | type == "string" and test("RTX 4090"))) or
   ($historical and (has("hardware") | not))) and
  ([.workload.concurrent_clients, .workload.max_tokens, .workload.repetitions,
    .summary.aggregate_throughput_ratio.minimum,
    .summary.aggregate_throughput_ratio.median,
    .summary.aggregate_throughput_ratio.maximum,
    .summary.p95_completion_latency_ratio.minimum,
    .summary.p95_completion_latency_ratio.median,
    .summary.p95_completion_latency_ratio.maximum] | all(type == "number")) and
  .checks.every_transcript_matches and
  .checks.every_disconnect_recovers and
  .checks.every_throughput_sample_wins and
  .checks.every_p95_completion_sample_wins
' "$study" >/dev/null || {
  echo "release input is not a passing Qwen3 8B, four-client, RTX 4090 study: $study" >&2
  exit 2
}

# Historical receipts lack retained runs and keep their original rendering.
digest_limit='The study keeps response and transcript digests. It does not retain signed responses or verify their signatures.'
batch_limit='Batch sizes record command-line limits. Dispatch widths are unmeasured.'
retained_note=
if jq -e 'has("runs")' "$study" >/dev/null; then
  jq -e --arg digest "$digest_limit" --arg batch "$batch_limit" \
    '.limits | index($digest) != null and index($batch) != null' "$study" >/dev/null || {
    echo "release input lacks the digest or batch size limit: $study" >&2
    exit 2
  }
  retained_note=$'\n'"- $digest_limit"$'\n'"- $batch_limit"
fi

quality=$(jq -er ".quality.path" "$study")
require_repository_path "$quality"
if [[ ! -f $quality || -L $quality ]]; then
  echo "release input does not exist: $quality" >&2
  exit 2
fi

if [[ $output -ef $study || $output -ef $quality ]]; then
  echo "release output would replace an input: $output" >&2
  exit 2
fi

quality_sha256=$(sha256sum "$quality" | cut -d' ' -f1)
recorded_quality_sha256=$(jq -er '.quality.sha256' "$study")
if [[ $quality_sha256 != "$recorded_quality_sha256" ]]; then
  echo "service study quality digest does not match $quality" >&2
  exit 2
fi

model_sha256=$(jq -er '.model.sha256' "$study")
clients=$(jq -er '.workload.concurrent_clients' "$study")
max_tokens=$(jq -er '.workload.max_tokens' "$study")
repetitions=$(jq -er '.workload.repetitions' "$study")
throughput_min=$(jq -er '.summary.aggregate_throughput_ratio.minimum' "$study")
throughput_median=$(jq -er '.summary.aggregate_throughput_ratio.median' "$study")
throughput_max=$(jq -er '.summary.aggregate_throughput_ratio.maximum' "$study")
latency_min=$(jq -er '.summary.p95_completion_latency_ratio.minimum' "$study")
latency_median=$(jq -er '.summary.p95_completion_latency_ratio.median' "$study")
latency_max=$(jq -er '.summary.p95_completion_latency_ratio.maximum' "$study")
quality_id=$(jq -er '.receipt_id' "$quality")
quality_kld=$(jq -er '.metrics.kld.mean' "$quality")
quality_top1=$(jq -er '.metrics.top1_agreement' "$quality")

mkdir -p "$(dirname "$output")"
cat >"$output" <<EOF
# Batched-service evidence

Leone batches compatible decode rows. Attention, sampling,
cancellation, and retained sessions stay separate.
The [study receipt](../${study}) records the source and inputs.

## Tested workload

| Field | Value |
|---|---|
| Model SHA-256 | \`${model_sha256}\` |
| Concurrent clients | ${clients} |
| Maximum tokens per request | ${max_tokens} |
| Repetitions | ${repetitions} |
| Shared batch limit | ${clients} |
| Baseline batch limit | 1 |

## Results

| Ratio | Minimum | Median | Maximum | Required |
|---|---:|---:|---:|---:|
| Aggregate completion throughput | ${throughput_min} | ${throughput_median} | ${throughput_max} | greater than 1 |
| P95 completion latency | ${latency_min} | ${latency_median} | ${latency_max} | less than 1 |

All batched transcripts match the baseline. Every run accepts a request after
a forced client disconnect.

The linked quality record is \`${quality_id}\`. Its mean KLD is ${quality_kld}
nats, and its top-1 agreement is ${quality_top1}. The study verifies that the
quality record and served model have the same SHA-256 digest.

## Limits

- The result covers one Qwen3 8B model, one RTX 4090, four clients, and one
  deterministic prompt.
- The baseline accepts concurrent requests but executes at most one decode row
  per pass.
- Non-streaming time to first HTTP byte is not time to first generated token.
- KV pages are admission units. The runtime does not remap physical KV storage.${retained_note}
EOF

echo "$output"
