#!/usr/bin/env bash
set -euo pipefail
umask 077

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
if [[ -e $output || -L $output ]]; then
  echo "the output already exists; select a new receipt path" >&2
  exit 2
fi
server_log="${output}.server.log"
if [[ -e $server_log || -L $server_log ]]; then
  echo "the server log already exists; select a new receipt path" >&2
  exit 2
fi
mkdir -p "$(dirname -- "$output")"
stage=$(mktemp -d)
server_pid=
preserve_log() {
  if [[ ! -f $stage/server.log ]]; then
    return 0
  fi
  if [[ -e $server_log || -L $server_log ]]; then
    echo "cannot preserve server log: destination exists: $server_log" >&2
    return 1
  fi
  if ! (set -C; cat "$stage/server.log" >"$server_log"); then
    echo "cannot preserve server log; private stage kept at $stage" >&2
    return 1
  fi
  if ! chmod 600 "$server_log"; then
    echo "cannot secure server log; private stage kept at $stage" >&2
    return 1
  fi
}

cleanup() {
  local exit_code=$?
  local preserve_status=0
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if ! preserve_log; then
    preserve_status=1
    echo "server log was not removed; private stage kept at $stage" >&2
  fi
  if (( preserve_status == 0 )); then
    if ! rm -rf "$stage"; then
      preserve_status=1
      echo "private stage could not be removed; kept at $stage" >&2
    fi
  fi
  if (( preserve_status != 0 && exit_code == 0 )); then
    exit_code=1
  fi
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

build_info=$("$binary" --build-info)
if [[ -n ${LEONE_CLIENT_BACKEND:-} ]]; then
  backend=$LEONE_CLIENT_BACKEND
elif grep -q '"features":"[^"]*cuda' <<<"$build_info"; then
  backend=cuda
elif grep -q '"features":"[^"]*metal' <<<"$build_info"; then
  backend=metal
else
  backend=cpu
fi
kv=${LEONE_CLIENT_KV:-f16}
batch_size=${LEONE_CLIENT_BATCH_SIZE:-}

case "$backend" in
  cpu) default_batch_size=1 ;;
  cuda) default_batch_size=4 ;;
  metal) default_batch_size=1 ;;
  *) echo "LEONE_CLIENT_BACKEND must be cpu, cuda, or metal" >&2; exit 2 ;;
esac
if [[ -z $batch_size ]]; then
  batch_size=$default_batch_size
fi
if [[ ! $batch_size =~ ^[1-9][0-9]*$ || $batch_size -gt 8 ]]; then
  echo "LEONE_CLIENT_BATCH_SIZE must be an integer from 1 to 8" >&2
  exit 2
fi
case "$kv" in
  q8|f16|f32) ;;
  *) echo "LEONE_CLIENT_KV must be q8, f16, or f32" >&2; exit 2 ;;
esac
head -c 32 /dev/urandom >"$stage/response.key"

if [[ ! -x $python ]]; then
  echo "create a client environment from scripts/client-requirements.txt" >&2
  exit 2
fi

"$binary" serve -m "$model" --bind "127.0.0.1:$port" \
  --backend "$backend" --kv "$kv" --sessions 8 --batch-size "$batch_size" \
  --prefill-chunk 128 --signing-key "$stage/response.key" \
  --receipt-dir "$stage/receipts" --session-store "$stage/sessions" \
  >"$stage/server.log" 2>&1 &
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
  --model "$model" \
  --base-url "http://127.0.0.1:$port/v1" --output "$output" \
  --backend "$backend" --kv "$kv" --batch-size "$batch_size"
