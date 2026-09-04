#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL TOKENS OUTPUT MANIFEST" >&2
  exit 2
fi

model=$1
tokens=$2
output=$3
manifest=$4
llama_dir=${LLAMA_CPP_DIR:-external/llama.cpp}
oracle_binary=${LLAMA_ORACLE_BINARY:-target/llama-logits-oracle}

if [[ ! -x $oracle_binary ]]; then
  scripts/build-llama-oracle.sh "$oracle_binary" >/dev/null
fi

"$oracle_binary" "$model" "$tokens" "$output" "${LLAMA_ORACLE_THREADS:-16}"
mkdir -p "$(dirname "$manifest")"
jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg llama_commit "$(git -C "$llama_dir" rev-parse HEAD)" \
  --arg adapter "research/oracle/llama_logits.cpp" \
  --arg adapter_sha256 "$(sha256sum research/oracle/llama_logits.cpp | cut -d' ' -f1)" \
  --arg model "$model" \
  --arg model_sha256 "$(sha256sum "$model" | cut -d' ' -f1)" \
  --arg tokens "$tokens" \
  --arg tokens_sha256 "$(sha256sum "$tokens" | cut -d' ' -f1)" \
  --arg logits "$output" \
  --arg logits_sha256 "$(sha256sum "$output" | cut -d' ' -f1)" \
  '{
    schema_version: "leone.llama-oracle.v1",
    created_utc: $created_utc,
    engine: {name: "llama.cpp", git_commit: $llama_commit, device: "CPU"},
    adapter: {path: $adapter, sha256: $adapter_sha256},
    model: {path: $model, sha256: $model_sha256, storage_dtype: "f16"},
    tokens: {path: $tokens, sha256: $tokens_sha256, encoding: "u32le"},
    logits: {path: $logits, sha256: $logits_sha256, encoding: "row-major-f32"}
  }' >"$manifest"

echo "$manifest"
