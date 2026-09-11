#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BINARY MODEL OUTPUT" >&2
  exit 2
fi

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
binary=$1
model=$2
output=$3
python=${LEONE_CLIENT_PYTHON:-target/client-venv/bin/python}
port=${LEONE_CLIENT_PORT:-18220}
stage=$(mktemp -d)
server_pid=

cleanup() {
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$stage"
}
trap cleanup EXIT HUP INT TERM

if [[ ! -x $python ]]; then
  echo "create a client environment from scripts/client-requirements.txt" >&2
  exit 2
fi

"$binary" serve -m "$model" --bind "127.0.0.1:$port" \
  --sessions 8 --batch-size 4 --prefill-chunk 128 --receipt-dir "$stage/receipts" \
  --session-store "$stage/sessions" >"$stage/server.log" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$stage/server.log" >&2
    exit 1
  fi
  sleep 0.1
done
if [[ $ready != true ]]; then
  echo "the client test server did not become ready" >&2
  exit 1
fi

"$python" scripts/check-openai-client.py --binary "$binary" \
  --base-url "http://127.0.0.1:$port/v1" --output "$output"
