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
binary=${LEONE_BINARY:-./target/release/leone}
base_port=${LEONE_STUDY_PORT:-18100}
tmp=$(mktemp -d)
server_pid=

cleanup() {
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

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

start_server() {
  local mode=$1
  local port=$2
  local receipt_dir="$tmp/$mode-receipts"
  mkdir -p "$receipt_dir"
  local batch_size=$clients
  if [[ $mode == serial ]]; then
    batch_size=1
  fi
  "$binary" serve -m "$model" --plan "$plan" \
    --bind "127.0.0.1:$port" --sessions "$clients" --batch-size "$batch_size" \
    --receipt-dir "$receipt_dir" >"$tmp/$mode-server.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 200); do
    if nc -z 127.0.0.1 "$port" 2>/dev/null; then
      return
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$tmp/$mode-server.log" >&2
      exit 1
    fi
    sleep 0.05
  done
  echo "$mode server did not become ready" >&2
  exit 1
}

stop_server() {
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
    --argjson wall_ms "$(jq -n --argjson start "$start_ns" --argjson end "$end_ns" '($end - $start) / 1000000')" \
    --arg server_log_sha256 "$(sha256sum "$tmp/$mode-server.log" | cut -d' ' -f1)" \
    'def percentile($p): sort | .[((length - 1) * $p | floor)];
     . as $requests |
     ($requests | map(.total_ms)) as $totals |
     ($requests | map(.ttft_ms)) as $ttfts |
     {
       mode: $mode,
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

start_server scheduled "$base_port"
run_batch scheduled "$base_port"
run_disconnect_probe "$base_port"
stop_server

start_server serial "$((base_port + 1))"
run_batch serial "$((base_port + 1))"
stop_server

mkdir -p "$(dirname "$output")"
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
    workload: {concurrent_clients: $clients, max_tokens: $max_tokens, temperature: 0, seed: 0},
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
      "The serial baseline accepts the same concurrent clients with a batch limit of one."
    ]
  }' >"$output"

jq -e '.checks.scheduled_transcripts_agree and .checks.scheduled_matches_serial and (.checks.aggregate_throughput_ratio > 1) and .checks.disconnect.client_disconnected and (.checks.disconnect.recovery_http_code == 200)' "$output" >/dev/null
echo "$output"
