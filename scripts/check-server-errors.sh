#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BINARY MODEL PLAN" >&2
  exit 2
fi

binary=$(realpath "$1")
model=$(realpath "$2")
plan=$(realpath "$3")
scratch=$(mktemp -d)
server_pid=
active_pid=

cleanup() {
  if [[ -n $active_pid ]]; then
    kill "$active_pid" 2>/dev/null || true
    wait "$active_pid" 2>/dev/null || true
  fi
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$scratch"
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$scratch/receipts"
stdbuf -oL -eL "$binary" serve \
  -m "$model" \
  --plan "$plan" \
  --bind 127.0.0.1:0 \
  --sessions 1 \
  --batch-size 1 \
  --receipt-dir "$scratch/receipts" \
  >"$scratch/server.log" 2>&1 &
server_pid=$!

endpoint=
for _ in $(seq 1 600); do
  endpoint=$(sed -n 's/^Leone serves \(http:\/\/127\.0\.0\.1:[0-9][0-9]*\)$/\1/p' \
    "$scratch/server.log" | tail -1)
  if [[ -n $endpoint ]] && curl -fsS --max-time 2 "$endpoint/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$scratch/server.log" >&2
    exit 1
  fi
  sleep 0.1
done
if [[ -z $endpoint ]]; then
  cat "$scratch/server.log" >&2
  echo "server did not report its bound address" >&2
  exit 1
fi

check_error() {
  local name=$1
  local expected_status=$2
  local message_fragment=$3
  shift 3
  local response="$scratch/$name.json"
  local status
  status=$(curl -sS --max-time 10 -o "$response" -w '%{http_code}' "$@")
  if [[ $status != "$expected_status" ]]; then
    cat "$response" >&2
    echo "$name returned HTTP $status, expected $expected_status" >&2
    exit 1
  fi
  jq -e --arg fragment "$message_fragment" \
    '.error.type == "invalid_request_error" and
     (.error.message | type == "string" and contains($fragment))' \
    "$response" >/dev/null
}

check_error malformed-json 400 "EOF" \
  -H 'content-type: application/json' \
  --data-binary '{"model":' \
  "$endpoint/v1/chat/completions"
check_error unknown-model 400 "is not served" \
  -H 'content-type: application/json' \
  --data-binary '{"model":"not-served","messages":[{"role":"user","content":"test"}]}' \
  "$endpoint/v1/chat/completions"
check_error unknown-route 404 "route not found" "$endpoint/not-a-route"

active_response="$scratch/active-response.txt"
curl -sS -N --max-time 30 \
  -H 'content-type: application/json' \
  --data-binary '{"model":"leone","messages":[{"role":"user","content":"Count upward without stopping."}],"max_tokens":512,"temperature":0,"stream":true}' \
  "$endpoint/v1/chat/completions" >"$active_response" 2>/dev/null &
active_pid=$!
for _ in $(seq 1 100); do
  [[ -s $active_response ]] && break
  sleep 0.05
done
if [[ ! -s $active_response ]]; then
  echo "active request did not begin" >&2
  exit 1
fi

overload_response="$scratch/overload.json"
overload_status=$(curl -sS --max-time 10 -o "$overload_response" -w '%{http_code}' \
  -H 'content-type: application/json' \
  --data-binary '{"model":"leone","messages":[{"role":"user","content":"test"}],"max_tokens":16}' \
  "$endpoint/v1/chat/completions")
if [[ $overload_status != 429 ]]; then
  cat "$overload_response" >&2
  echo "overload returned HTTP $overload_status, expected 429" >&2
  exit 1
fi
jq -e '
  .error.message == "the server cannot admit this request" and
  .error.type == "server_overloaded" and
  .error.code == "active-limit"
' "$overload_response" >/dev/null

kill "$active_pid" 2>/dev/null || true
wait "$active_pid" 2>/dev/null || true
active_pid=
curl -fsS --max-time 2 "$endpoint/health" | jq -e '.status == "ok"' >/dev/null

echo "packaged scheduled HTTP and backpressure checks passed"
