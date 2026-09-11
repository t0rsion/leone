#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

output=${1:-docs/release-evidence.md}
study=receipts/batched-service-study.json
quality=$(jq -er ".quality.path" "$study")

for input in "$study" "$quality"; do
  if [[ ! -f $input ]]; then
    echo "release input does not exist: $input" >&2
    exit 2
  fi
done

jq -e '
  .checks.every_transcript_matches and
  .checks.every_disconnect_recovers and
  .checks.every_throughput_sample_wins and
  .checks.every_p95_completion_sample_wins
' "$study" >/dev/null

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

The candidate batches compatible decode rows. Attention, sampling,
cancellation, and retained sessions stay separate.
The [study receipt](../receipts/batched-service-study.json) records the source and inputs.

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
- KV pages are admission units. The runtime does not remap physical KV storage.
EOF

echo "$output"
