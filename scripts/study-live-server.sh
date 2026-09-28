#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 MODEL PLAN QUALITY_RECEIPT OUTPUT [CLIENTS] [MAX_TOKENS]" >&2
  exit 2
fi

model=$1
plan=$2
quality_receipt=$3
output=$4
clients=${5:-4}
max_tokens=${6:-64}
study_prompt=${LEONE_STUDY_PROMPT:-Write a detailed explanation of why deterministic scheduling matters for language model inference. Do not use a list.}
prompt_sha256=$(printf '%s' "$study_prompt" | sha256sum | cut -d' ' -f1)
binary=${LEONE_BINARY:-./target/release/leone}
base_port=${LEONE_STUDY_PORT:-18100}
gate=${LEONE_STUDY_GATE:-enforce}
raw_dir=${LEONE_STUDY_RAW_DIR:-}
source_manifest=${LEONE_STUDY_SOURCE_MANIFEST:-}
u32_max=4294967295
port_max=65535
server_pid=
tmp=
owns_tmp=0

# LEONE_STUDY_GATE=collect drops only the throughput condition from the exit
# status. Transcript, disconnect, and server errors still fail the run.
if [[ $gate != enforce && $gate != collect ]]; then
  echo "LEONE_STUDY_GATE must be enforce or collect: $gate" >&2
  exit 2
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "the service study requires a clean candidate commit" >&2
  exit 2
fi

for input in "$model" "$plan" "$quality_receipt"; do
  if [[ $input = /* ]]; then
    echo "study inputs must use repository-relative paths: $input" >&2
    exit 2
  fi
done

require_count() {
  local name=$1 value=$2 maximum=$3
  if [[ ! $value =~ ^[1-9][0-9]{0,9}$ ]] || ((value > maximum)); then
    echo "$name must be an integer from 1 to $maximum: $value" >&2
    exit 2
  fi
}
require_count CLIENTS "$clients" "$u32_max"
require_count MAX_TOKENS "$max_tokens" "$u32_max"
require_count LEONE_STUDY_PORT "$base_port" "$((port_max - 1))"

if [[ -z $output || $output = */ || -e $output || -L $output ]]; then
  echo "study output must be a new file path: $output" >&2
  exit 2
fi

# The binary must be a clean release CUDA build whose source inputs equal the
# current tree. HEAD alone says nothing about a prebuilt LEONE_BINARY.
if [[ ! -f $binary || ! -x $binary ]]; then
  echo "Leone executable is missing: $binary" >&2
  exit 2
fi
build_info=$("$binary" --build-info)
jq -e '
  .schema_version == "leone.build-info.v1" and
  (.source_commit | type == "string" and test("^[0-9a-f]{40}$")) and
  .source_tree_dirty == false and .provenance_unknown == false and
  .profile == "release" and .target == "x86_64-unknown-linux-gnu" and
  (.features | type == "string" and ((split(",") | index("cuda")) != null))
' <<<"$build_info" >/dev/null || {
  echo "the executable is not a clean release CUDA build with known provenance" >&2
  exit 2
}
source_check=(python3 scripts/source_inputs.py check "$(jq -er '.source_commit' <<<"$build_info")")
if [[ -n $source_manifest ]]; then
  source_check+=(--manifest "$source_manifest")
fi
"${source_check[@]}" >/dev/null || {
  echo "the executable does not match the current source inputs" >&2
  exit 2
}
binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)

# Exactly one visible GPU keeps the recorded identity unambiguous.
gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader,nounits) || {
  echo "nvidia-smi could not read the GPU identity" >&2
  exit 2
}
hardware=$(jq -Rn --arg gpu "$gpu" '
  ($gpu | split("\n") | map(select(length > 0))) as $lines |
  if ($lines | length) != 1 then error("expected one GPU") else
    ($lines[0] | split(", ")) as $field |
    if ($field | length) != 4 then error("unexpected GPU record") else
      {
        gpu_name: $field[0],
        driver_version: $field[1],
        memory_total_mib: ($field[2] | tonumber),
        compute_capability: $field[3]
      }
    end
  end
') || {
  echo "the study needs one GPU with a readable identity" >&2
  exit 2
}

# Responses, timings, server logs, and server receipts stay in this directory
# after any nonzero exit. A directory from LEONE_STUDY_RAW_DIR belongs to the
# caller and stays after success.
mkdir -p "$(dirname "$output")"
if [[ -n $raw_dir ]]; then
  mkdir -- "$raw_dir" || {
    echo "LEONE_STUDY_RAW_DIR must be a new directory: $raw_dir" >&2
    exit 2
  }
  tmp=$raw_dir
else
  tmp=$(mktemp -d "$output.raw.XXXXXX")
  owns_tmp=1
fi

finish() {
  local status=$?
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]] && ! rmdir "$tmp" 2>/dev/null; then
    echo "study records kept in $tmp" >&2
  elif [[ $status -eq 0 && $owns_tmp -eq 1 ]]; then
    rm -rf -- "$tmp"
  fi
  return "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

model_sha256=$(sha256sum "$model" | cut -d' ' -f1)
quality_model_sha256=$(jq -er '.subject.model_artifact.sha256' "$quality_receipt")
if [[ $quality_model_sha256 != "$model_sha256" ]]; then
  echo "quality receipt subject does not match the served model" >&2
  exit 2
fi

request_body=$(jq -nc --argjson max_tokens "$max_tokens" --arg prompt "$study_prompt" '{
  model: "leone",
  messages: [{role: "user", content: $prompt}],
  max_tokens: $max_tokens,
  temperature: 0,
  seed: 0,
  stream: false
}')

# The scheduled server batches up to one row per client. The serial baseline
# accepts the same clients with a batch limit of one.
batch_limit() {
  if [[ $1 == serial ]]; then
    echo 1
  else
    echo "$clients"
  fi
}

listeners() {
  ss -H -ltnp "sport = :$1" || {
    echo "ss could not list the listeners on port $1" >&2
    exit 2
  }
}

require_free_port() {
  local mode=$1 port=$2 found
  found=$(listeners "$port")
  if [[ -n $found ]]; then
    echo "$mode port $port already has a listener: $found" >&2
    exit 2
  fi
}

# The listener on the port must belong to the process started here. The
# process is the binary itself, not a wrapper, so a forking binary fails this
# check. A listener that another process holds never counts as ready, even
# when the new server is still alive and about to exit with EADDRINUSE.
start_server() {
  local mode=$1
  local port=$2
  local receipt_dir="$tmp/$mode-receipts" found
  require_free_port "$mode" "$port"
  mkdir -p "$receipt_dir"
  local batch_size
  batch_size=$(batch_limit "$mode")
  "$binary" serve -m "$model" --plan "$plan" \
    --bind "127.0.0.1:$port" --sessions "$clients" --batch-size "$batch_size" \
    --receipt-dir "$receipt_dir" >"$tmp/$mode-server.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 200); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$tmp/$mode-server.log" >&2
      echo "the $mode server exited before it listened on port $port" >&2
      exit 1
    fi
    found=$(listeners "$port")
    if [[ $found == *"pid=$server_pid,"* ]]; then
      return
    fi
    sleep 0.05
  done
  echo "$mode server did not become ready" >&2
  exit 1
}

stop_server() {
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "the study server exited before its measurements ended" >&2
    exit 1
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=
}

run_batch() {
  local mode=$1
  local port=$2
  local start_ns end_ns
  local pids=()
  start_ns=$(date +%s%N)
  for client in $(seq 1 "$clients"); do
    curl -fsS \
      -H 'content-type: application/json' \
      -d "$request_body" \
      -o "$tmp/$mode-$client.json" \
      -w '%{http_code}\t%{time_starttransfer}\t%{time_total}\n' \
      "http://127.0.0.1:$port/v1/chat/completions" \
      >"$tmp/$mode-$client.timing" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  end_ns=$(date +%s%N)

  for client in $(seq 1 "$clients"); do
    read -r http_code ttft_s total_s <"$tmp/$mode-$client.timing"
    jq -n \
      --argjson client "$client" \
      --argjson http_code "$http_code" \
      --argjson ttft_s "$ttft_s" \
      --argjson total_s "$total_s" \
      --arg response_sha256 "$(sha256sum "$tmp/$mode-$client.json" | cut -d' ' -f1)" \
      --arg transcript_sha256 "$(jq -er '.leone_receipt.claim.transcript_sha256' "$tmp/$mode-$client.json")" \
      --argjson completion_tokens "$(jq -er '.usage.completion_tokens' "$tmp/$mode-$client.json")" \
      '{client: $client, http_code: $http_code, ttft_ms: ($ttft_s * 1000), total_ms: ($total_s * 1000), completion_tokens: $completion_tokens, transcript_sha256: $transcript_sha256, response_sha256: $response_sha256}'
  done | jq -s \
    --arg mode "$mode" \
    --argjson batch_size "$(batch_limit "$mode")" \
    --argjson wall_ms "$(jq -n --argjson start "$start_ns" --argjson end "$end_ns" '($end - $start) / 1000000')" \
    --arg server_log_sha256 "$(sha256sum "$tmp/$mode-server.log" | cut -d' ' -f1)" \
    'def percentile($p): sort | .[((length - 1) * $p | floor)];
     . as $requests |
     ($requests | map(.total_ms)) as $totals |
     ($requests | map(.ttft_ms)) as $ttfts |
     {
       mode: $mode,
       batch_size: $batch_size,
       wall_ms: $wall_ms,
       aggregate_completion_tok_s: (($requests | map(.completion_tokens) | add) / ($wall_ms / 1000)),
       ttft_ms: {p50: ($ttfts | percentile(0.50)), p95: ($ttfts | percentile(0.95)), p99: ($ttfts | percentile(0.99))},
       total_ms: {p50: ($totals | percentile(0.50)), p95: ($totals | percentile(0.95)), p99: ($totals | percentile(0.99))},
       latency_fairness_ratio: (($totals | max) / ($totals | min)),
       server_log_sha256: $server_log_sha256,
       requests: $requests
     }' >"$tmp/$mode-summary.json"
}

run_disconnect_probe() {
  local port=$1
  local stream_body recovery_code
  stream_body=$(jq -nc '{
    model: "leone",
    messages: [{role: "user", content: "Count upward for as long as the response permits."}],
    max_tokens: 512,
    temperature: 0,
    seed: 0,
    stream: true
  }')
  set +e
  curl -sS --max-time 0.2 -H 'content-type: application/json' -d "$stream_body" \
    "http://127.0.0.1:$port/v1/chat/completions" >/dev/null 2>"$tmp/disconnect.stderr"
  disconnect_exit=$?
  set -e
  recovery_code=$(curl -sS -o "$tmp/recovery.json" -w '%{http_code}' \
    -H 'content-type: application/json' -d "$request_body" \
    "http://127.0.0.1:$port/v1/chat/completions")
  jq -n \
    --argjson curl_exit "$disconnect_exit" \
    --argjson recovery_http_code "$recovery_code" \
    --arg recovery_transcript_sha256 "$(jq -er '.leone_receipt.claim.transcript_sha256' "$tmp/recovery.json")" \
    '{curl_exit: $curl_exit, client_disconnected: ($curl_exit == 28), recovery_http_code: $recovery_http_code, recovery_transcript_sha256: $recovery_transcript_sha256}' \
    >"$tmp/disconnect-summary.json"
}

require_free_port scheduled "$base_port"
require_free_port serial "$((base_port + 1))"

start_server scheduled "$base_port"
run_batch scheduled "$base_port"
run_disconnect_probe "$base_port"
stop_server

start_server serial "$((base_port + 1))"
run_batch serial "$((base_port + 1))"
stop_server

if [[ $(sha256sum "$binary" | cut -d' ' -f1) != "$binary_sha256" ]]; then
  echo "the executable changed during the study" >&2
  exit 1
fi

jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_commit "$(git rev-parse HEAD)" \
  --arg model "$model" \
  --arg model_sha256 "$model_sha256" \
  --arg plan "$plan" \
  --arg plan_sha256 "$(sha256sum "$plan" | cut -d' ' -f1)" \
  --arg quality_receipt "$quality_receipt" \
  --arg quality_receipt_sha256 "$(sha256sum "$quality_receipt" | cut -d' ' -f1)" \
  --arg quality_receipt_id "$(jq -er '.receipt_id' "$quality_receipt")" \
  --argjson clients "$clients" \
  --argjson max_tokens "$max_tokens" \
  --arg prompt_sha256 "$prompt_sha256" \
  --arg binary_sha256 "$binary_sha256" \
  --argjson build_info "$build_info" \
  --argjson hardware "$hardware" \
  --slurpfile scheduled "$tmp/scheduled-summary.json" \
  --slurpfile serial "$tmp/serial-summary.json" \
  --slurpfile disconnect "$tmp/disconnect-summary.json" \
  '{
    schema_version: "leone.server-study.v2",
    created_utc: $created_utc,
    source_commit: $source_commit,
    model: {path: $model, sha256: $model_sha256},
    plan: {path: $plan, sha256: $plan_sha256},
    quality: {path: $quality_receipt, sha256: $quality_receipt_sha256, receipt_id: $quality_receipt_id},
    workload: {concurrent_clients: $clients, max_tokens: $max_tokens, temperature: 0, seed: 0, prompt_sha256: $prompt_sha256},
    scheduled: $scheduled[0],
    serial: $serial[0],
    checks: {
      scheduled_transcripts_agree: (($scheduled[0].requests | map(.transcript_sha256) | unique | length) == 1),
      scheduled_matches_serial: (($scheduled[0].requests | map(.transcript_sha256) | unique) == ($serial[0].requests | map(.transcript_sha256) | unique)),
      aggregate_throughput_ratio: ($scheduled[0].aggregate_completion_tok_s / $serial[0].aggregate_completion_tok_s),
      disconnect: $disconnect[0]
    },
    limits: [
      "This study measures one model, one GPU, simultaneous clients, and one deterministic prompt.",
      "curl reports time to the first HTTP response byte. This is not time to the first generated token for non-streaming responses.",
      "The serial baseline accepts the same concurrent clients with a batch limit of one.",
      "The study keeps response and transcript digests. It does not retain signed responses or verify their signatures.",
      "Batch sizes record command-line limits. Dispatch widths are unmeasured."
    ],
    binary: {role: "leone-cli", sha256: $binary_sha256, build_info: $build_info},
    hardware: $hardware
  }' >"$tmp/receipt.json"

if ! ln -T -- "$tmp/receipt.json" "$output"; then
  echo "study output could not be created exclusively: $output" >&2
  exit 2
fi

gate_filter='.checks.scheduled_transcripts_agree and .checks.scheduled_matches_serial and .checks.disconnect.client_disconnected and (.checks.disconnect.recovery_http_code == 200)'
if [[ $gate == enforce ]]; then
  gate_filter="$gate_filter and (.checks.aggregate_throughput_ratio > 1)"
fi
status=0
jq -e "$gate_filter" "$output" >/dev/null || status=$?
if ((status != 0)); then
  echo "study receipt fails the $gate gate: $output" >&2
  exit "$status"
fi
echo "$output"
