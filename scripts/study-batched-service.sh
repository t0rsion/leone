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
u32_max=4294967295
port_max=65535

for input in "$model" "$plan" "$quality_receipt"; do
  if [[ $input = /* ]]; then
    echo "study inputs must use repository-relative paths: $input" >&2
    exit 2
  fi
done

# The scheduler takes session and batch limits as u32. The server rejects a
# zero max_tokens. Each repetition uses two ports.
require_count() {
  local name=$1 value=$2 maximum=$3
  if [[ ! $value =~ ^[1-9][0-9]{0,9}$ ]] || ((value > maximum)); then
    echo "$name must be an integer from 1 to $maximum: $value" >&2
    exit 2
  fi
}
require_count CLIENTS "$clients" "$u32_max"
require_count MAX_TOKENS "$max_tokens" "$u32_max"
require_count LEONE_STUDY_PORT "$base_port" "$port_max"
require_count LEONE_STUDY_REPETITIONS "$repetitions" "$u32_max"
if ((base_port + repetitions * 2 + 1 > port_max)); then
  echo "LEONE_STUDY_PORT and LEONE_STUDY_REPETITIONS need ports above $port_max" >&2
  exit 2
fi

# The output stays unwritten until the complete receipt exists. Any existing
# path, including a dangling symlink or a directory, is a collision.
if [[ -z $output || $output = */ || -e $output || -L $output ]]; then
  echo "study output must be a new file path: $output" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"

# Run records and the composed receipt stay here until the study exits 0.
work=$(mktemp -d "$output.runs.XXXXXX")

finish() {
  local status=$?
  if [[ $status -ne 0 ]] && ! rmdir "$work" 2>/dev/null; then
    echo "measured records kept in $work" >&2
  elif [[ $status -eq 0 ]]; then
    rm -rf -- "$work"
  fi
  return "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

runs=()
for repetition in $(seq 1 "$repetitions"); do
  runs+=("$work/$repetition.json")
  status=0
  LEONE_STUDY_PORT=$((base_port + repetition * 2)) \
    LEONE_STUDY_GATE=collect \
    LEONE_STUDY_RAW_DIR="$work/$repetition.raw" \
    scripts/study-live-server.sh \
      "$model" "$plan" "$quality_receipt" "$work/$repetition.json" \
      "$clients" "$max_tokens" || status=$?
  if ((status != 0)); then
    echo "study repetition $repetition of $repetitions failed with status $status" >&2
    exit "$status"
  fi
done

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
      "The quality record covers the model and inference path. It does not certify this prompt alone.",
      "The study keeps response and transcript digests. It does not retain signed responses or verify their signatures.",
      "Batch sizes record command-line limits. Dispatch widths are unmeasured."
    ],
    runs: $runs,
    binary: ($runs | map(.binary) | unique | if length == 1 then .[0] else error("repetitions used different executables") end),
    hardware: ($runs | map(.hardware) | unique | if length == 1 then .[0] else error("repetitions used different GPUs") end)
  }
' "${runs[@]}" >"$work/receipt.json"

if ! ln -T -- "$work/receipt.json" "$output"; then
  echo "study output could not be created exclusively: $output" >&2
  exit 2
fi

status=0
jq -e '
  .checks.every_transcript_matches and
  .checks.every_disconnect_recovers and
  .checks.every_throughput_sample_wins and
  .checks.every_p95_completion_sample_wins
' "$output" >/dev/null || status=$?
if ((status != 0)); then
  echo "study receipt fails the performance gate: $output" >&2
  exit "$status"
fi

echo "$output"
