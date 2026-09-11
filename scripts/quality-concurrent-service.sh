#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 SUBJECT_Q4 ORACLE_FULL CORPUS TOKEN_LIMIT WINDOW OUTPUT [LEONE_BACKEND]" >&2
  exit 2
fi

subject_model=$1
oracle_model=$2
corpus=$3
token_limit=$4
window=$5
manifest=$6
leone_backend=${7:-${LEONE_QUALITY_BACKEND:-cuda}}
binary=${LEONE_BINARY:-$root/target/release/leone}
leone_prefill_chunk=${LEONE_QUALITY_PREFILL_CHUNK:-128}
oracle_device=${LEONE_QUALITY_ORACLE_DEVICE:-cpu}
llama_device=${LEONE_QUALITY_LLAMA_DEVICE:-cuda}
oracle_threads=${LEONE_QUALITY_ORACLE_THREADS:-16}
llama_threads=${LEONE_QUALITY_LLAMA_THREADS:-16}
oracle_n_batch=${LEONE_QUALITY_ORACLE_N_BATCH:-$window}
llama_n_batch=${LEONE_QUALITY_LLAMA_N_BATCH:-$window}
oracle_flash=${LEONE_QUALITY_ORACLE_FLASH_ATTN:-off}
llama_flash=${LEONE_QUALITY_LLAMA_FLASH_ATTN:-auto}
write_receipts=${LEONE_QUALITY_WRITE_RECEIPTS:-0}
keep_artifacts=${LEONE_QUALITY_KEEP_ARTIFACTS:-0}

for input in "$subject_model" "$oracle_model" "$corpus"; do
  if [[ ! -f $input ]]; then
    echo "error: required input is missing: $input" >&2
    exit 1
  fi
done
subject_model_path=$(realpath "$subject_model")
oracle_model_path=$(realpath "$oracle_model")
corpus_path=$(realpath "$corpus")
manifest_path=$(realpath -m "$manifest")
if [[ -e $manifest ]]; then
  echo "error: refusing to replace existing quality manifest: $manifest" >&2
  exit 2
fi
if [[ $manifest_path == "$subject_model_path" || $manifest_path == "$oracle_model_path" || $manifest_path == "$corpus_path" ]]; then
  echo "error: output manifest must differ from model and corpus inputs" >&2
  exit 2
fi
if [[ $leone_backend != cuda && $leone_backend != cpu && $leone_backend != cpu-q8_1 ]]; then
  echo "error: Leone backend must be cuda, cpu, or cpu-q8_1" >&2
  exit 2
fi
if ! [[ $token_limit =~ ^[1-9][0-9]*$ && $window =~ ^[1-9][0-9]*$ ]]; then
  echo "error: token limit and window must be positive integers" >&2
  exit 2
fi
if (( token_limit < 2 || window < 2 || token_limit < window )); then
  echo "error: require token limit >= window >= 2" >&2
  exit 2
fi
if ! [[ $leone_prefill_chunk =~ ^[1-9][0-9]*$ ]]; then
  echo "error: LEONE_QUALITY_PREFILL_CHUNK must be a positive integer" >&2
  exit 2
fi
if (( leone_prefill_chunk > window )); then
  echo "error: LEONE_QUALITY_PREFILL_CHUNK must not exceed the evaluation window" >&2
  exit 2
fi
default_n_ubatch=$window
if (( default_n_ubatch > 512 )); then
  default_n_ubatch=512
fi
oracle_n_ubatch=${LEONE_QUALITY_ORACLE_N_UBATCH:-$default_n_ubatch}
llama_n_ubatch=${LEONE_QUALITY_LLAMA_N_UBATCH:-$default_n_ubatch}
if [[ $write_receipts != 0 && $write_receipts != 1 ]]; then
  echo "error: LEONE_QUALITY_WRITE_RECEIPTS must be 0 or 1" >&2
  exit 2
fi
if [[ $keep_artifacts != 0 && $keep_artifacts != 1 ]]; then
  echo "error: LEONE_QUALITY_KEEP_ARTIFACTS must be 0 or 1" >&2
  exit 2
fi
if [[ ! -x $binary ]]; then
  taskset -c "${LEONE_BUILD_CPUSET:-16-31}" cargo +1.92 build --release --locked -p leone-cli
fi
if [[ ! -x $binary ]]; then
  echo "error: Leone executable is missing: $binary" >&2
  exit 1
fi

source_commit=$(git rev-parse HEAD)
build_info=$("$binary" --build-info)
if ! jq -e '
  .schema_version == "leone.build-info.v1" and
  (.source_commit | type == "string" and length == 40) and
  (.source_tree_dirty | type == "boolean") and
  (.source_paths | type == "array" and all(.[]; type == "string")) and
  (.target | type == "string" and length > 0) and
  (.profile | type == "string" and length > 0)
' <<<"$build_info" >/dev/null; then
  echo "error: Leone executable returned invalid --build-info JSON" >&2
  exit 1
fi
binary_source_commit=$(jq -er '.source_commit' <<<"$build_info")
if [[ $binary_source_commit != "$source_commit" ]]; then
  echo "error: Leone executable was built from $binary_source_commit, expected $source_commit" >&2
  exit 1
fi

tmp=$(mktemp -d)
cleanup() {
  if [[ $keep_artifacts == 0 ]]; then
    rm -rf -- "$tmp"
  fi
}
trap cleanup EXIT HUP INT TERM

tokens="$tmp/tokens.bin"
oracle_logits="$tmp/oracle.f32"
llama_logits="$tmp/llama-q4.f32"
leone_logits="$tmp/leone-q4.f32"
oracle_manifest="$tmp/oracle.json"
llama_manifest="$tmp/llama.json"
token_stdout="$tmp/tokenize.stdout"
token_stderr="$tmp/tokenize.stderr"
leone_stdout="$tmp/leone-eval.stdout"
leone_stderr="$tmp/leone-eval.stderr"

if ! "$binary" eval \
  -m "$subject_model" \
  --corpus "$corpus" \
  --token-limit "$token_limit" \
  --tokens-out "$tokens" \
  >"$token_stdout" 2>"$token_stderr"; then
  cat "$token_stderr" >&2
  exit 1
fi
token_bytes=$(stat -c '%s' "$tokens")
if (( token_bytes != token_limit * 4 )); then
  echo "error: tokenizer emitted $((token_bytes / 4)) tokens, expected $token_limit" >&2
  exit 1
fi

if ! LLAMA_ORACLE_LOG="$tmp/oracle.log" scripts/run-llama-oracle.sh \
  "$oracle_model" "$tokens" "$oracle_logits" "$oracle_manifest" \
  "$window" "$oracle_device" "$oracle_threads" "$oracle_n_batch" "$oracle_n_ubatch" "$oracle_flash" \
  >"$tmp/oracle.stdout"; then
  if [[ -f $tmp/oracle.log ]]; then
    cat "$tmp/oracle.log" >&2
  fi
  exit 1
fi
if ! LLAMA_ORACLE_LOG="$tmp/llama.log" scripts/run-llama-oracle.sh \
  "$subject_model" "$tokens" "$llama_logits" "$llama_manifest" \
  "$window" "$llama_device" "$llama_threads" "$llama_n_batch" "$llama_n_ubatch" "$llama_flash" \
  >"$tmp/llama.stdout"; then
  if [[ -f $tmp/llama.log ]]; then
    cat "$tmp/llama.log" >&2
  fi
  exit 1
fi

if ! jq -e '
  .model.vocab_size == ($oracle[0].model.vocab_size) and
  .model.architecture == ($oracle[0].model.architecture) and
  .model.tokenizer == ($oracle[0].model.tokenizer) and
  .input.tokens.sha256 == ($oracle[0].input.tokens.sha256) and
  .input.tokens.encoding == ($oracle[0].input.tokens.encoding) and
  .input.tokens.count == ($oracle[0].input.tokens.count) and
  .input.window_tokens == ($oracle[0].input.window_tokens) and
  .input.rows == ($oracle[0].input.rows)
' --slurpfile oracle "$oracle_manifest" "$llama_manifest" >/dev/null; then
  echo "error: llama.cpp oracle and Q4 subject disagree on model or input provenance" >&2
  exit 1
fi
if [[ $(sha256sum "$subject_model" | cut -d' ' -f1) != "$(jq -er '.model.sha256' "$llama_manifest")" ]]; then
  echo "error: llama.cpp subject manifest does not bind the requested model" >&2
  exit 1
fi
if [[ $(sha256sum "$oracle_model" | cut -d' ' -f1) != "$(jq -er '.model.sha256' "$oracle_manifest")" ]]; then
  echo "error: llama.cpp oracle manifest does not bind the requested model" >&2
  exit 1
fi
oracle_storage=$(jq -er '.model.storage_type' "$oracle_manifest")
subject_storage=$(jq -er '.model.storage_type' "$llama_manifest")
case "$oracle_storage" in
  BF16|F16) ;;
  *) echo "error: oracle model storage type is $oracle_storage, expected BF16 or F16" >&2; exit 1 ;;
esac
if [[ $subject_storage != "Q4_K - Medium" ]]; then
  echo "error: llama.cpp subject storage type is $subject_storage, expected Q4_K - Medium" >&2
  exit 1
fi

if ! "$binary" eval \
  -m "$subject_model" \
  --tokens "$tokens" \
  --window "$window" \
  --prefill-chunk "$leone_prefill_chunk" \
  --backend "$leone_backend" \
  --logits "$leone_logits" \
  >"$leone_stdout" 2>"$leone_stderr"; then
  cat "$leone_stderr" >&2
  exit 1
fi
leone_rows=$((token_limit - 1))
leone_vocab=$(jq -er '.model.vocab_size' "$llama_manifest")
expected_leone_bytes=$((leone_rows * leone_vocab * 4))
actual_leone_bytes=$(stat -c '%s' "$leone_logits")
if [[ $actual_leone_bytes != "$expected_leone_bytes" ]]; then
  echo "error: Leone logits have $actual_leone_bytes bytes, expected $expected_leone_bytes" >&2
  exit 1
fi

oracle_manifest_sha256=$(sha256sum "$oracle_manifest" | cut -d' ' -f1)
llama_manifest_sha256=$(sha256sum "$llama_manifest" | cut -d' ' -f1)
oracle_manifest_body=$(cat "$oracle_manifest")
llama_manifest_body=$(cat "$llama_manifest")
corpus_sha256=$(sha256sum "$corpus" | cut -d' ' -f1)
tokens_sha256=$(sha256sum "$tokens" | cut -d' ' -f1)
subject_sha256=$(sha256sum "$subject_model" | cut -d' ' -f1)
oracle_sha256=$(sha256sum "$oracle_model" | cut -d' ' -f1)
binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)
leone_logits_sha256=$(sha256sum "$leone_logits" | cut -d' ' -f1)
oracle_commit=$(jq -er '.engine.git_commit' "$oracle_manifest")
oracle_dtype=$(jq -er '.model.storage_type | ascii_downcase' "$oracle_manifest")
leone_device_name=cpu
if [[ $leone_backend == cuda ]]; then
  leone_device_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i 0 2>/dev/null || true)
  if [[ -z $leone_device_name ]]; then
    leone_device_name=unreported-cuda-device
  fi
fi

run_quality() {
  local subject_logits_path=$1
  local subject_engine=$2
  local subject_commit=$3
  local quality_output=$4
  local quality_error=$5
  local receipt_args=()
  if [[ $write_receipts == 1 ]]; then
    receipt_args=(--receipt)
  fi
  if ! "$binary" quality \
    --oracle "$oracle_logits" \
    --subject "$subject_logits_path" \
    --corpus "$corpus" \
    --tokens "$tokens" \
    --oracle-model "$oracle_model" \
    --subject-model "$subject_model" \
    --oracle-engine "llama.cpp" \
    --oracle-commit "$oracle_commit" \
    --oracle-dtype "$oracle_dtype" \
    --subject-engine "$subject_engine" \
    --subject-commit "$subject_commit" \
    "${receipt_args[@]}" \
    >"$quality_output" 2>"$quality_error"; then
    cat "$quality_error" >&2
    exit 1
  fi
}

run_quality "$leone_logits" "leone-${leone_backend}-eval" "$source_commit" "$tmp/leone-quality.txt" "$tmp/leone-quality.err"
run_quality "$llama_logits" "llama.cpp-${llama_device}" "$oracle_commit" "$tmp/llama-quality.txt" "$tmp/llama-quality.err"

quality_json() {
  local path=$1
  local receipt_path=
  if [[ $write_receipts == 1 ]]; then
    receipt_path=$(awk '$1 == "receipt:" { print $2 }' "$path")
    if [[ -z $receipt_path || ! -f $receipt_path ]]; then
      echo "error: quality did not report a receipt path" >&2
      exit 1
    fi
    jq --arg receipt "$receipt_path" \
      --arg sha256 "$(sha256sum "$receipt_path" | cut -d' ' -f1)" '
      {sample_count: .corpus.n_tokens_scored,
       kld: .metrics.kld,
       top1_agreement: .metrics.top1_agreement,
       receipt: $receipt, receipt_sha256: $sha256}
    ' "$receipt_path"
    return
  fi
  local sample_count mean p50 p99 max top1
  sample_count=$(awk '$1 == "samples:" { print $2 }' "$path")
  mean=$(awk '$1 == "KLD" && $2 == "mean:" { print $3 }' "$path")
  p50=$(awk '$1 == "KLD" && $2 == "p50:" { print $3 }' "$path")
  p99=$(awk '$1 == "KLD" && $2 == "p99:" { print $3 }' "$path")
  max=$(awk '$1 == "KLD" && $2 == "max:" { print $3 }' "$path")
  top1=$(awk '$1 == "top1" && $2 == "agreement:" { print $3 }' "$path")
  if ! [[ $sample_count =~ ^[1-9][0-9]*$ && $mean =~ ^[0-9.eE+-]+$ && $p50 =~ ^[0-9.eE+-]+$ && $p99 =~ ^[0-9.eE+-]+$ && $max =~ ^[0-9.eE+-]+$ && $top1 =~ ^[0-9.eE+-]+$ ]]; then
    echo "error: quality output is missing a metric" >&2
    exit 1
  fi
  jq -n \
    --argjson sample_count "$sample_count" \
    --argjson mean "$mean" \
    --argjson p50 "$p50" \
    --argjson p99 "$p99" \
    --argjson max "$max" \
    --argjson top1 "$top1" \
    --arg receipt "$receipt_path" \
    '{sample_count: $sample_count, kld: {mean: $mean, p50: $p50, p99: $p99, max: $max}, top1_agreement: $top1} + (if $receipt == "" then {} else {receipt: $receipt} end)'
}

leone_quality=$(quality_json "$tmp/leone-quality.txt")
llama_quality=$(quality_json "$tmp/llama-quality.txt")
mkdir -p "$(dirname "$manifest")"
jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_commit "$source_commit" \
  --arg executable "$binary" \
  --arg executable_sha256 "$binary_sha256" \
  --argjson build_info "$build_info" \
  --arg subject_model "$subject_model" \
  --arg subject_sha256 "$subject_sha256" \
  --arg oracle_model "$oracle_model" \
  --arg oracle_sha256 "$oracle_sha256" \
  --arg corpus "$corpus" \
  --arg corpus_sha256 "$corpus_sha256" \
  --arg tokens "$tokens" \
  --arg tokens_sha256 "$tokens_sha256" \
  --argjson token_count "$token_limit" \
  --argjson window "$window" \
  --arg oracle_manifest "$oracle_manifest" \
  --arg oracle_manifest_sha256 "$oracle_manifest_sha256" \
  --argjson oracle_manifest_body "$oracle_manifest_body" \
  --arg llama_manifest "$llama_manifest" \
  --arg llama_manifest_sha256 "$llama_manifest_sha256" \
  --argjson llama_manifest_body "$llama_manifest_body" \
  --arg leone_logits_sha256 "$leone_logits_sha256" \
  --argjson leone_logits_bytes "$actual_leone_bytes" \
  --arg oracle_device "$oracle_device" \
  --arg llama_device "$llama_device" \
  --arg leone_backend "$leone_backend" \
  --arg leone_prefill_chunk "$leone_prefill_chunk" \
  --arg leone_device_name "$leone_device_name" \
  --argjson leone_quality "$leone_quality" \
  --argjson llama_quality "$llama_quality" \
  --argjson keep_artifacts "$keep_artifacts" \
  '{
    schema_version: "leone.quality-comparison.v1",
    created_utc: $created_utc,
    source_commit: $source_commit,
    executable: {path: $executable, sha256: $executable_sha256},
    build_info: $build_info,
    corpus: {path: $corpus, sha256: $corpus_sha256},
    input: {
      token_file: {path: $tokens, sha256: $tokens_sha256, count: $token_count, encoding: "u32le"},
      window_tokens: $window,
      stride_tokens: ($window - 1),
      scored_positions: ($token_count - 1)
    },
    model_family: {
      oracle: {
        path: $oracle_model,
        sha256: $oracle_sha256,
        manifest_path: $oracle_manifest,
        manifest_sha256: $oracle_manifest_sha256,
        manifest: $oracle_manifest_body
      },
      subject: {path: $subject_model, sha256: $subject_sha256}
    },
    executions: {
      oracle: {
        engine: "llama.cpp",
        device: $oracle_device,
        manifest_path: $oracle_manifest,
        manifest_sha256: $oracle_manifest_sha256,
        manifest: $oracle_manifest_body
      },
      llama_q4: {
        engine: "llama.cpp",
        device: $llama_device,
        manifest_path: $llama_manifest,
        manifest_sha256: $llama_manifest_sha256,
        manifest: $llama_manifest_body,
        quality: $llama_quality
      },
      leone_q4: {
        engine: "leone",
        path: "eval",
        backend: $leone_backend,
        device_name: $leone_device_name,
        arithmetic_dtype: "implementation-selected",
        kv_cache_dtype: "f16",
        prefill_path: "chunked",
        prefill_chunk_tokens: ($leone_prefill_chunk | tonumber),
        logits: {sha256: $leone_logits_sha256, bytes: $leone_logits_bytes, encoding: "row-major-f32-le"},
        quality: $leone_quality
      }
    },
    limits: [
      "The oracle is pinned llama.cpp BF16 or F16 execution, independent of Leone. The llama.cpp subject shares its implementation.",
      "The quality rows compare direct eval chunked-prefill artifacts. They do not certify concurrent server CUDA numerics.",
      "A CPU oracle supplies a common distributional reference. It does not certify the numeric path of a CUDA server.",
      "The service study must capture its own server path and retain its runtime receipt.",
      "KLD uses the full vocabulary and the declared overlapping window. A quality result does not cover every prompt or sampler."
    ],
    artifacts: {retained: ($keep_artifacts == 1)}
  }' | jq --arg root "$root/" 'walk(if type == "string" then split($root) | join("") else . end)' | (set -o noclobber; cat >"$manifest")

echo "$manifest"
