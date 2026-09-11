#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

if [[ $# -lt 4 || $# -gt 10 ]]; then
  echo "usage: $0 MODEL TOKENS OUTPUT MANIFEST [WINDOW] [DEVICE] [THREADS] [N_BATCH] [N_UBATCH] [FLASH_ATTN]" >&2
  exit 2
fi

model=$1
tokens=$2
output=$3
manifest=$4
llama_dir=${LLAMA_CPP_DIR:-$root/external/llama.cpp}
oracle_binary=${LLAMA_ORACLE_BINARY:-$root/target/llama-logits-oracle}
window=${5:-${LLAMA_ORACLE_WINDOW:-}}
device=${6:-${LLAMA_ORACLE_DEVICE:-cpu}}
threads=${7:-${LLAMA_ORACLE_THREADS:-16}}
n_batch=${8:-${LLAMA_ORACLE_N_BATCH:-}}
n_ubatch=${9:-${LLAMA_ORACLE_N_UBATCH:-512}}
flash_attn=${10:-${LLAMA_ORACLE_FLASH_ATTN:-auto}}

if [[ $llama_dir != /* ]]; then
  llama_dir="$root/$llama_dir"
fi
if [[ $oracle_binary != /* ]]; then
  oracle_binary="$root/$oracle_binary"
fi
for input in "$model" "$tokens"; do
  if [[ ! -f $input ]]; then
    echo "error: required input is missing: $input" >&2
    exit 1
  fi
done
model=$(realpath "$model")
tokens=$(realpath "$tokens")
output=$(realpath -m "$output")
manifest=$(realpath -m "$manifest")
if [[ $output == "$model" || $output == "$tokens" || $manifest == "$model" || $manifest == "$tokens" || $output == "$manifest" ]]; then
  echo "error: output paths must differ from model, token, and each other" >&2
  exit 2
fi
if [[ ! -f "$llama_dir/.git/HEAD" ]]; then
  echo "error: pinned llama.cpp checkout is missing: $llama_dir" >&2
  exit 1
fi
if [[ ! -f "$root/external/PINNED" ]]; then
  echo "error: external/PINNED is missing" >&2
  exit 1
fi
pin=$(tr -d '[:space:]' <"$root/external/PINNED")
if [[ ! $pin =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: external/PINNED must contain one 40-character commit hash" >&2
  exit 1
fi
llama_commit=$(git -C "$llama_dir" rev-parse HEAD)
if [[ $llama_commit != "$pin" ]]; then
  echo "error: llama.cpp HEAD $llama_commit differs from external/PINNED $pin" >&2
  exit 1
fi
if [[ -n $(git -C "$llama_dir" status --porcelain --untracked-files=all) ]]; then
  echo "error: pinned llama.cpp checkout has local changes" >&2
  exit 1
fi

token_bytes=$(stat -c '%s' "$tokens")
if (( token_bytes < 8 || token_bytes % 4 != 0 )); then
  echo "error: token file must contain at least two little-endian u32 values" >&2
  exit 1
fi
token_count=$((token_bytes / 4))
if [[ -z $window ]]; then
  window=$token_count
fi
if [[ -z $n_batch ]]; then
  n_batch=$window
fi
if [[ $device != cpu && $device != cuda ]]; then
  echo "error: device must be cpu or cuda" >&2
  exit 2
fi
if [[ $flash_attn != auto && $flash_attn != on && $flash_attn != off ]]; then
  echo "error: flash attention must be auto, on, or off" >&2
  exit 2
fi
if ! [[ $window =~ ^[1-9][0-9]*$ && $threads =~ ^[1-9][0-9]*$ && $n_batch =~ ^[1-9][0-9]*$ && $n_ubatch =~ ^[1-9][0-9]*$ ]]; then
  echo "error: window, threads, n_batch, and n_ubatch must be positive integers" >&2
  exit 2
fi
if (( window < 2 || n_batch < window || n_ubatch > n_batch )); then
  echo "error: require window >= 2, n_batch >= window, and n_ubatch <= n_batch" >&2
  exit 2
fi

mkdir -p "$(dirname "$output")" "$(dirname "$manifest")"
scripts/build-llama-oracle.sh "$oracle_binary" >/dev/null
if [[ ! -x $oracle_binary ]]; then
  echo "error: oracle executable was not created: $oracle_binary" >&2
  exit 1
fi

oracle_log=${LLAMA_ORACLE_LOG:-}
if [[ -n $oracle_log ]]; then
  mkdir -p "$(dirname "$oracle_log")"
  run_info=$("$oracle_binary" "$model" "$tokens" "$output" "$window" "$threads" "$device" "$n_batch" "$n_ubatch" "$flash_attn" 2>"$oracle_log")
else
  run_info=$("$oracle_binary" "$model" "$tokens" "$output" "$window" "$threads" "$device" "$n_batch" "$n_ubatch" "$flash_attn")
fi
if ! jq -e '(.tokens | type == "number") and (.rows | type == "number") and (.vocab | type == "number") and (.window | type == "number")' <<<"$run_info" >/dev/null; then
  echo "error: oracle executable returned invalid run metadata" >&2
  exit 1
fi
if ! jq -e '
  (.stride | type == "number" and . > 0) and
  (.threads | type == "number" and . > 0) and
  (.n_batch | type == "number" and . > 0) and
  (.n_ubatch | type == "number" and . > 0) and
  (.context_capacity | type == "number" and . > 0) and
  (.effective_n_batch | type == "number" and . > 0) and
  (.effective_n_ubatch | type == "number" and . > 0) and
  (.device == "cpu" or .device == "cuda") and
  (.device_name | type == "string" and length > 0) and
  (.flash_attn | type == "string" and length > 0) and
  (.flash_attn_effective | type == "string" and length > 0) and
  (.offload_kqv | type == "boolean") and
  (.op_offload | type == "boolean") and
  (.model_context | type == "number" and . > 0) and
  (.model_size | type == "number" and . > 0) and
  (.model_ftype | type == "number") and
  (.model_ftype_name | type == "string" and length > 0) and
  (.architecture | type == "string" and length > 0) and
  (.tokenizer | type == "string" and length > 0)
' <<<"$run_info" >/dev/null; then
  echo "error: oracle executable omitted required model or execution metadata" >&2
  exit 1
fi
if [[ $(jq -er '.tokens' <<<"$run_info") != "$token_count" || $(jq -er '.window' <<<"$run_info") != "$window" ]]; then
  echo "error: oracle metadata does not match the requested token stream or window" >&2
  exit 1
fi
rows=$(jq -er '.rows' <<<"$run_info")
vocab=$(jq -er '.vocab' <<<"$run_info")
if [[ $rows != $((token_count - 1)) ]]; then
  echo "error: oracle emitted $rows rows, expected $((token_count - 1))" >&2
  exit 1
fi
expected_bytes=$((rows * vocab * 4))
actual_bytes=$(stat -c '%s' "$output")
if [[ $actual_bytes != "$expected_bytes" ]]; then
  echo "error: oracle logits have $actual_bytes bytes, expected $expected_bytes" >&2
  exit 1
fi

model_path=$model
tokens_path=$tokens
output_path=$output
manifest_path=$manifest
linked_libraries=$(python3 - "$oracle_binary" <<'PYLIB'
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

linked = subprocess.check_output(["ldd", sys.argv[1]], text=True)
if "not found" in linked:
    raise SystemExit("oracle has an unresolved shared library")
paths = sorted(set(re.findall(r"(?:=>\s+|^\s*)(/[^\s]+)", linked, flags=re.MULTILINE)))
records = []
for name in paths:
    path = Path(name)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    records.append({"path": str(path.resolve()), "sha256": sha})
print(json.dumps(records))
PYLIB
)

jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_commit "$(git rev-parse HEAD)" \
  --arg llama_commit "$llama_commit" \
  --arg adapter "research/oracle/llama_logits.cpp" \
  --arg adapter_sha256 "$(sha256sum research/oracle/llama_logits.cpp | cut -d' ' -f1)" \
  --arg executable "$oracle_binary" \
  --arg executable_sha256 "$(sha256sum "$oracle_binary" | cut -d' ' -f1)" \
  --argjson linked_libraries "$linked_libraries" \
  --arg model "$model_path" \
  --arg model_sha256 "$(sha256sum "$model_path" | cut -d' ' -f1)" \
  --arg tokens "$tokens_path" \
  --arg tokens_sha256 "$(sha256sum "$tokens_path" | cut -d' ' -f1)" \
  --arg logits "$output_path" \
  --arg logits_sha256 "$(sha256sum "$output_path" | cut -d' ' -f1)" \
  --argjson logits_bytes "$actual_bytes" \
  --argjson run_info "$run_info" \
  '{
    schema_version: "leone.llama-oracle.v2",
    created_utc: $created_utc,
    source_commit: $source_commit,
    engine: {name: "llama.cpp", git_commit: $llama_commit},
    adapter: {path: $adapter, sha256: $adapter_sha256},
    executable: {path: $executable, sha256: $executable_sha256, linked_libraries: $linked_libraries},
    model: {
      path: $model,
      sha256: $model_sha256,
      storage_type: $run_info.model_ftype_name,
      storage_type_id: $run_info.model_ftype,
      size_bytes: $run_info.model_size,
      context_tokens: $run_info.model_context,
      vocab_size: $run_info.vocab,
      architecture: $run_info.architecture,
      tokenizer: $run_info.tokenizer
    },
    input: {
      tokens: {path: $tokens, sha256: $tokens_sha256, count: $run_info.tokens, encoding: "u32le"},
      window_tokens: $run_info.window,
      stride_tokens: $run_info.stride,
      rows: $run_info.rows
    },
    execution: {
      device: $run_info.device,
      device_name: $run_info.device_name,
      threads: $run_info.threads,
      n_batch: $run_info.n_batch,
      n_ubatch: $run_info.n_ubatch,
      context_capacity: $run_info.context_capacity,
      effective_n_batch: $run_info.effective_n_batch,
      effective_n_ubatch: $run_info.effective_n_ubatch,
      flash_attention: $run_info.flash_attn,
      flash_attention_effective: $run_info.flash_attn_effective,
      offload_kqv: $run_info.offload_kqv,
      op_offload: $run_info.op_offload,
      arithmetic_dtype: "implementation-selected",
      logits_dtype: "f32"
    },
    logits: {path: $logits, sha256: $logits_sha256, bytes: $logits_bytes, encoding: "row-major-f32-le"}
  }' | jq --arg root "$root/" 'walk(if type == "string" then split($root) | join("") else . end)' >"$manifest_path"

echo "$manifest_path"
