#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 MODEL PLAN QUALITY_RECEIPT OUTPUT [CLIENTS] [MAX_TOKENS]" >&2
  exit 2
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "the service study requires a clean candidate commit" >&2
  exit 2
fi

model=$1
plan=$2
quality_receipt=$3
output=$4
clients=${5:-4}
max_tokens=${6:-64}
repetitions=${LEONE_STUDY_REPETITIONS:-5}
base_port=${LEONE_STUDY_PORT:-18100}
tmp=$(mktemp -d)

for input in "$model" "$plan" "$quality_receipt"; do
  if [[ $input = /* ]]; then
    echo "study inputs must use repository-relative paths: $input" >&2
    exit 2
  fi
done

cleanup() {
  rm -rf "$tmp"
}
trap cleanup EXIT

for repetition in $(seq 1 "$repetitions"); do
  LEONE_STUDY_PORT=$((base_port + repetition * 2)) \
    scripts/study-live-server.sh \
      "$model" "$plan" "$quality_receipt" "$tmp/$repetition.json" \
      "$clients" "$max_tokens"
done

mkdir -p "$(dirname "$output")"
jq -s '
  def median: sort | .[(length / 2 | floor)];
  . as $runs |
  ($runs | map(.checks.aggregate_throughput_ratio)) as $throughput_ratios |
  ($runs | map(.scheduled.total_ms.p95 / .serial.total_ms.p95)) as $latency_ratios |
  {
    schema_version: "leone.batched-service-study.v1",
    created_utc: $runs[-1].created_utc,
    source_commit: $runs[-1].source_commit,
    model: $runs[-1].model,
    plan: $runs[-1].plan,
    quality: $runs[-1].quality,
    workload: ($runs[-1].workload + {repetitions: ($runs | length)}),
    samples: ($runs | map({
      scheduled: {
        wall_ms: .scheduled.wall_ms,
        aggregate_completion_tok_s: .scheduled.aggregate_completion_tok_s,
        p95_completion_ms: .scheduled.total_ms.p95,
        transcript_sha256: .scheduled.requests[0].transcript_sha256
      },
      serial: {
        wall_ms: .serial.wall_ms,
        aggregate_completion_tok_s: .serial.aggregate_completion_tok_s,
        p95_completion_ms: .serial.total_ms.p95,
        transcript_sha256: .serial.requests[0].transcript_sha256
      },
      aggregate_throughput_ratio: .checks.aggregate_throughput_ratio,
      p95_completion_latency_ratio: (.scheduled.total_ms.p95 / .serial.total_ms.p95),
      transcript_matches: (.checks.scheduled_transcripts_agree and .checks.scheduled_matches_serial),
      disconnect_recovers: (.checks.disconnect.client_disconnected and (.checks.disconnect.recovery_http_code == 200))
    })),
    summary: {
      aggregate_throughput_ratio: {
        minimum: ($throughput_ratios | min),
        median: ($throughput_ratios | median),
        maximum: ($throughput_ratios | max)
      },
      p95_completion_latency_ratio: {
        minimum: ($latency_ratios | min),
        median: ($latency_ratios | median),
        maximum: ($latency_ratios | max)
      }
    },
    checks: {
      every_transcript_matches: ($runs | all(.checks.scheduled_transcripts_agree and .checks.scheduled_matches_serial)),
      every_disconnect_recovers: ($runs | all(.checks.disconnect.client_disconnected and (.checks.disconnect.recovery_http_code == 200))),
      every_throughput_sample_wins: ($throughput_ratios | all(. > 1)),
      every_p95_completion_sample_wins: ($latency_ratios | all(. < 1))
    },
    limits: [
      "The study covers one model, one GPU, one client count, and one prompt.",
      "The serial baseline accepts the same concurrent workload with a batch limit of one.",
      "The quality record covers the model and inference path. It does not certify this prompt alone."
    ]
  }
' "$tmp"/*.json >"$output"

jq -e '
  .checks.every_transcript_matches and
  .checks.every_disconnect_recovers and
  .checks.every_throughput_sample_wins and
  .checks.every_p95_completion_sample_wins
' "$output" >/dev/null

echo "$output"
