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
receipt_verifier=${LEONE_RECEIPT_VERIFIER:-$root/target/release/leone-receipt-verify}
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
comparison_version=${LEONE_QUALITY_COMPARISON_VERSION:-1}
quality_stage=${LEONE_QUALITY_STAGE:-aggregate}
oracle_stage=${LEONE_QUALITY_ORACLE_STAGE:-}
task_manifest=${LEONE_QUALITY_TASK_MANIFEST:-}
trusted_inputs=${LEONE_QUALITY_TRUSTED_INPUTS:-}
generation_record=${LEONE_QUALITY_GENERATION_RECORD:-}
collection_dir=${LEONE_QUALITY_COLLECTION_DIR:-}
statistics_platform=${LEONE_QUALITY_STATISTICS_PLATFORM:-linux-x86_64}
statistics_target=${LEONE_QUALITY_STATISTICS_TARGET:-}
requested_source_commit=${LEONE_QUALITY_SOURCE_COMMIT:-}

for input in "$subject_model" "$oracle_model" "$corpus"; do
  if [[ ! -f $input ]]; then
    echo "error: required input is missing: $input" >&2
    exit 1
  fi
done
subject_model_path=$(realpath "$subject_model")
oracle_model_path=$(realpath "$oracle_model")
corpus_path=$(realpath "$corpus")
subject_model_sha_before=$(sha256sum "$subject_model_path" | cut -d' ' -f1)
oracle_model_sha_before=$(sha256sum "$oracle_model_path" | cut -d' ' -f1)
manifest_path=$(realpath -m "$manifest")

public_path() {
  local path=$1
  local fallback=$2
  local resolved
  resolved=$(realpath -m "$path")
  case "$resolved" in
    "$root"/*) printf '%s\n' "${resolved#"$root/"}" ;;
    *) printf '%s\n' "$fallback" ;;
  esac
}

public_binary_path=$(public_path "$binary" "leone")
public_subject_model_path=$(public_path "$subject_model_path" "subject-model.gguf")
public_oracle_model_path=$(public_path "$oracle_model_path" "oracle-model.gguf")
public_corpus_path=$(public_path "$corpus_path" "corpus.txt")
public_tokens_path="tokens.u32le"
public_oracle_manifest_path="oracle-manifest.json"
public_llama_manifest_path="llama-manifest.json"
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
if [[ $comparison_version != 1 && $comparison_version != 2 ]]; then
  echo "error: LEONE_QUALITY_COMPARISON_VERSION must be 1 or 2" >&2
  exit 2
fi
if [[ $quality_stage != aggregate && $quality_stage != collect ]]; then
  echo "error: LEONE_QUALITY_STAGE must be aggregate or collect" >&2
  exit 2
fi
if [[ $quality_stage == collect && $comparison_version != 2 ]]; then
  echo "error: LEONE_QUALITY_STAGE=collect requires CUDA comparison v2" >&2
  exit 2
fi
if [[ ! -x $binary ]]; then
  taskset -c "${LEONE_BUILD_CPUSET:-16-31}" cargo +1.92 build --release --locked -p leone-cli
fi
if [[ ! -x $binary ]]; then
  echo "error: Leone executable is missing: $binary" >&2
  exit 1
fi

build_info=$("$binary" --build-info)
if ! jq -e '
  .schema_version == "leone.build-info.v1" and
  (.source_commit | type == "string" and length == 40) and
  (.source_tree_dirty | type == "boolean") and
  (.source_paths | type == "array" and all(.[]; type == "string")) and
  (.target | type == "string" and length > 0) and
  (.profile == "release")
' <<<"$build_info" >/dev/null; then
  echo "error: Leone executable returned invalid --build-info JSON" >&2
  exit 1
fi
binary_source_commit=$(jq -er '.source_commit' <<<"$build_info")
if [[ $(jq -er '.source_tree_dirty' <<<"$build_info") != false ]]; then
  echo "error: Leone executable was built from a dirty source tree" >&2
  exit 1
fi
source_commit=${requested_source_commit:-$binary_source_commit}
if ! [[ $source_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: LEONE_QUALITY_SOURCE_COMMIT is not a Git commit" >&2
  exit 2
fi
if [[ $binary_source_commit != "$source_commit" ]]; then
  echo "error: Leone executable was built from $binary_source_commit, expected $source_commit" >&2
  exit 1
fi
if ! git cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
  echo "error: Leone source commit is not present in this checkout: $source_commit" >&2
  exit 1
fi

tmp=$(mktemp -d)
cleanup() {
  if [[ $keep_artifacts == 0 ]]; then
    rm -rf -- "$tmp"
  fi
}
trap cleanup EXIT HUP INT TERM

tokens="$tmp/tokens.u32le"
oracle_logits="$tmp/oracle.f32"
llama_logits="$tmp/llama-q4.f32"
leone_logits="$tmp/leone-q4.f32"
oracle_manifest="$tmp/oracle.json"
llama_manifest="$tmp/llama.json"
token_stdout="$tmp/tokenize.stdout"
token_stderr="$tmp/tokenize.stderr"
leone_stdout="$tmp/leone-eval.stdout"
leone_stderr="$tmp/leone-eval.stderr"
leone_metadata="$tmp/leone-eval.json"

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
oracle_sha256=$(sha256sum "$oracle_model" | cut -d' ' -f1)
corpus_sha256=$(sha256sum "$corpus" | cut -d' ' -f1)
tokens_sha256=$(sha256sum "$tokens" | cut -d' ' -f1)

if [[ -n $oracle_stage ]]; then
  if [[ ! -f $oracle_stage ]]; then
    echo "error: LEONE_QUALITY_ORACLE_STAGE is missing: $oracle_stage" >&2
    exit 1
  fi
  oracle_stage=$(realpath "$oracle_stage")
  oracle_stage_dir=$(dirname "$oracle_stage")
  if ! python3 scripts/validate-quality-stage.py export "$oracle_stage_dir" >/dev/null; then
    echo "error: retained oracle stage failed validation" >&2
    exit 1
  fi
  if ! jq -e \
    --arg model_sha256 "$oracle_sha256" \
    --arg corpus_sha256 "$corpus_sha256" \
    --arg tokens_sha256 "$tokens_sha256" \
    --argjson token_count "$token_limit" \
    --argjson window "$window" \
    '.model.oracle.sha256 == $model_sha256 and
     .corpus.sha256 == $corpus_sha256 and
     .input.tokens.sha256 == $tokens_sha256 and
     .input.tokens.count == $token_count and
     .input.window_tokens == $window' \
    "$oracle_stage"; then
    echo "error: retained oracle stage does not match this model, corpus, token stream, or window" >&2
    exit 1
  fi
  cp "$oracle_stage_dir/oracle.f32" "$oracle_logits"
  jq -e '.oracle.manifest.body' "$oracle_stage" >"$oracle_manifest"
else
  if ! LLAMA_ORACLE_LOG="$tmp/oracle.log" scripts/run-llama-oracle.sh \
    "$oracle_model" "$tokens" "$oracle_logits" "$oracle_manifest" \
    "$window" "$oracle_device" "$oracle_threads" "$oracle_n_batch" "$oracle_n_ubatch" "$oracle_flash" \
    >"$tmp/oracle.stdout"; then
    if [[ -f $tmp/oracle.log ]]; then
      cat "$tmp/oracle.log" >&2
    fi
    exit 1
  fi
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
if [[ $(sha256sum "$subject_model_path" | cut -d' ' -f1) != "$subject_model_sha_before" || $(sha256sum "$oracle_model_path" | cut -d' ' -f1) != "$oracle_model_sha_before" ]]; then
  echo "error: a model changed during quality generation" >&2
  exit 1
fi

if ! "$binary" eval \
  -m "$subject_model" \
  --tokens "$tokens" \
  --window "$window" \
  --prefill-chunk "$leone_prefill_chunk" \
  --backend "$leone_backend" \
  --logits "$leone_logits" \
  --metadata "$leone_metadata" \
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
if [[ $(sha256sum "$subject_model_path" | cut -d' ' -f1) != "$subject_model_sha_before" || $(sha256sum "$oracle_model_path" | cut -d' ' -f1) != "$oracle_model_sha_before" ]]; then
  echo "error: a model changed during Leone evaluation" >&2
  exit 1
fi
leone_device_name=$(jq -er '.execution.device_name' "$leone_metadata")

oracle_manifest_sha256=$(sha256sum "$oracle_manifest" | cut -d' ' -f1)
llama_manifest_sha256=$(sha256sum "$llama_manifest" | cut -d' ' -f1)
oracle_manifest_body=$(cat "$oracle_manifest")
llama_manifest_body=$(cat "$llama_manifest")
corpus_sha256=$(sha256sum "$corpus" | cut -d' ' -f1)
tokens_sha256=$(sha256sum "$tokens" | cut -d' ' -f1)
subject_sha256=$(sha256sum "$subject_model" | cut -d' ' -f1)
oracle_sha256=$(sha256sum "$oracle_model" | cut -d' ' -f1)
binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)
oracle_logits_sha256=$(sha256sum "$oracle_logits" | cut -d' ' -f1)
llama_logits_sha256=$(sha256sum "$llama_logits" | cut -d' ' -f1)
leone_logits_sha256=$(sha256sum "$leone_logits" | cut -d' ' -f1)
leone_metadata_sha256=$(sha256sum "$leone_metadata" | cut -d' ' -f1)
oracle_commit=$(jq -er '.engine.git_commit' "$oracle_manifest")
oracle_dtype=$(jq -er '.model.storage_type | ascii_downcase' "$oracle_manifest")
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

if [[ $comparison_version != 2 ]]; then
  run_quality "$leone_logits" "leone-${leone_backend}-eval" "$source_commit" "$tmp/leone-quality.txt" "$tmp/leone-quality.err"
  run_quality "$llama_logits" "llama.cpp-${llama_device}" "$oracle_commit" "$tmp/llama-quality.txt" "$tmp/llama-quality.err"
fi

quality_json() {
  local path=$1
  local label=$2
  local receipt_path=
  if [[ $write_receipts == 1 ]]; then
    receipt_path=$(awk '$1 == "receipt:" { print $2 }' "$path")
    if [[ -z $receipt_path || ! -f $receipt_path ]]; then
      echo "error: quality did not report a receipt path" >&2
      exit 1
    fi
    local receipt_destination
    receipt_destination="$(dirname "$manifest")/$(basename "$manifest")-${label}-quality.json"
    mkdir -p "$(dirname "$receipt_destination")"
    if ! python3 - "$receipt_path" "$receipt_destination" <<'PY'
import os
import shutil
import sys

source, destination = sys.argv[1:]
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
try:
    with os.fdopen(descriptor, "wb") as target, open(source, "rb") as original:
        shutil.copyfileobj(original, target)
        target.flush()
        os.fsync(target.fileno())
except BaseException:
    try:
        os.unlink(destination)
    except FileNotFoundError:
        pass
    raise
PY
    then
      echo "error: refusing to replace quality receipt: $receipt_destination" >&2
      exit 2
    fi
    receipt_path=$(public_path "$receipt_destination" "${label}-quality.json")
    jq --arg receipt "$receipt_path" \
      --arg sha256 "$(sha256sum "$receipt_destination" | cut -d' ' -f1)" '
      {sample_count: .corpus.n_tokens_scored,
       kld: .metrics.kld,
       top1_agreement: .metrics.top1_agreement,
       receipt: $receipt, receipt_sha256: $sha256}
    ' "$receipt_destination"
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

publish_collection_file() {
  local source=$1
  local destination=$2
  python3 - "$source" "$destination" <<'PY'
import errno
import os
import shutil
import sys

source, destination = sys.argv[1:]
with open(source, "rb") as original:
    os.fsync(original.fileno())
try:
    os.link(source, destination)
except OSError as error:
    if error.errno != errno.EXDEV:
        raise
else:
    raise SystemExit(0)
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
try:
    with os.fdopen(descriptor, "wb") as target, open(source, "rb") as original:
        shutil.copyfileobj(original, target)
        target.flush()
        os.fsync(target.fileno())
except BaseException:
    try:
        os.unlink(destination)
    except FileNotFoundError:
        pass
    raise
PY
}

publish_collection_manifest() {
  local destination=$1
  local body=$2
  python3 - "$destination" "$body" <<'PY'
import json
import os
import sys
import tempfile

destination, encoded = sys.argv[1:]
descriptor, temporary_name = tempfile.mkstemp(dir=os.path.dirname(destination), prefix=".collection.", suffix=".tmp")
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        target.write(json.dumps(json.loads(encoded), indent=2, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.link(temporary_name, destination)
except FileExistsError:
    raise SystemExit(f"refusing to replace collection manifest: {destination}")
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
}

if [[ $comparison_version == 2 ]]; then
  if [[ $quality_stage == collect ]]; then
    if [[ $leone_backend != cuda ]]; then
      echo "error: CUDA collection requires --backend cuda" >&2
      exit 2
    fi
    if [[ -z $collection_dir ]]; then
      echo "error: CUDA collection requires LEONE_QUALITY_COLLECTION_DIR" >&2
      exit 2
    fi
    collection_path=$(realpath -m "$collection_dir")
    if [[ -e $collection_path ]]; then
      echo "error: refusing to replace collection directory: $collection_path" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$collection_path")"
    mkdir "$collection_path"
    build_info_collection="$tmp/build-info.json"
    printf '%s\n' "$build_info" >"$build_info_collection"
    for pair in \
      "$tokens tokens.u32le" \
      "$oracle_logits oracle.f32" \
      "$llama_logits llama-cpp-cuda.f32" \
      "$leone_logits leone-cuda.f32" \
      "$oracle_manifest oracle-manifest.json" \
      "$llama_manifest llama-manifest.json" \
      "$leone_metadata leone-manifest.json" \
      "$build_info_collection native-build-info.json"; do
      source=${pair% *}
      name=${pair##* }
      publish_collection_file "$source" "$collection_path/$name"
    done
    build_info_sha256=$(sha256sum "$build_info_collection" | cut -d' ' -f1)
    input_record=$(jq -c '.input | .tokens.path = "tokens.u32le"' "$llama_manifest")
    collection_body=$(jq -n \
      --argjson input "$input_record" \
      --arg subject_model_sha256 "$subject_sha256" \
      --arg oracle_model_sha256 "$oracle_sha256" \
      --arg corpus_sha256 "$corpus_sha256" \
      --arg tokens_sha256 "$tokens_sha256" \
      --arg oracle_logits_sha256 "$oracle_logits_sha256" \
      --arg llama_logits_sha256 "$llama_logits_sha256" \
      --arg leone_logits_sha256 "$leone_logits_sha256" \
      --arg oracle_manifest_sha256 "$oracle_manifest_sha256" \
      --arg llama_manifest_sha256 "$llama_manifest_sha256" \
      --arg leone_manifest_sha256 "$leone_metadata_sha256" \
      --arg build_info_sha256 "$build_info_sha256" \
      --argjson token_count "$token_limit" \
      --argjson token_bytes "$token_bytes" \
      --argjson oracle_bytes "$(stat -c '%s' "$oracle_logits")" \
      --argjson llama_bytes "$(stat -c '%s' "$llama_logits")" \
      --argjson leone_bytes "$actual_leone_bytes" \
      '{schema_version: "leone.quality-collection.v1", input: $input,
        models: {subject_sha256: $subject_model_sha256, oracle_sha256: $oracle_model_sha256,
                 corpus_sha256: $corpus_sha256},
        artifacts: {
          tokens: {path: "tokens.u32le", sha256: $tokens_sha256, bytes: $token_bytes, count: $token_count, encoding: "u32le"},
          oracle_logits: {path: "oracle.f32", sha256: $oracle_logits_sha256, bytes: $oracle_bytes, encoding: "row-major-f32-le"},
          llama_logits: {path: "llama-cpp-cuda.f32", sha256: $llama_logits_sha256, bytes: $llama_bytes, encoding: "row-major-f32-le"},
          leone_logits: {path: "leone-cuda.f32", sha256: $leone_logits_sha256, bytes: $leone_bytes, encoding: "row-major-f32-le"},
          oracle_manifest: {path: "oracle-manifest.json", sha256: $oracle_manifest_sha256},
          llama_manifest: {path: "llama-manifest.json", sha256: $llama_manifest_sha256},
          leone_manifest: {path: "leone-manifest.json", sha256: $leone_manifest_sha256},
          native_build_info: {path: "native-build-info.json", sha256: $build_info_sha256}
        },
        producer_manifests: {oracle: $oracle_manifest_sha256, llama_cpp: $llama_manifest_sha256,
                             leone: $leone_manifest_sha256}}')
    publish_collection_manifest "$collection_path/collection.json" "$collection_body"
    echo "collection: $collection_path/collection.json"
    exit 0
  fi
  if [[ $leone_backend != cuda || $write_receipts != 1 ]]; then
    echo "error: CUDA comparison v2 requires --backend cuda and LEONE_QUALITY_WRITE_RECEIPTS=1" >&2
    exit 2
  fi
  if [[ ! -x $receipt_verifier ]]; then
    echo "error: CUDA comparison v2 requires an executable receipt verifier: $receipt_verifier" >&2
    exit 1
  fi
  if [[ -z $task_manifest || ! -f $task_manifest ]]; then
    echo "error: CUDA comparison v2 requires LEONE_QUALITY_TASK_MANIFEST" >&2
    exit 2
  fi
  if [[ -z $trusted_inputs || ! -f $trusted_inputs ]]; then
    echo "error: CUDA comparison v2 requires LEONE_QUALITY_TRUSTED_INPUTS" >&2
    exit 2
  fi
  if [[ -z $generation_record || ! -f $generation_record ]]; then
    echo "error: CUDA comparison v2 requires LEONE_QUALITY_GENERATION_RECORD" >&2
    exit 2
  fi
  task_manifest=$(realpath "$task_manifest")
  build_info_path="$tmp/build-info.json"
  printf '%s\n' "$build_info" >"$build_info_path"
  target=$(jq -er '.target' <<<"$build_info")
  if [[ $target != x86_64-unknown-linux-gnu ]]; then
    echo "error: CUDA comparison v2 requires x86_64-unknown-linux-gnu" >&2
    exit 1
  fi
  statistics_target=${statistics_target:-$target}
  model_family=${LEONE_QUALITY_MODEL_FAMILY:-}
  if [[ -z $model_family ]]; then
    echo "error: CUDA comparison v2 requires LEONE_QUALITY_MODEL_FAMILY" >&2
    exit 2
  fi
  python3 scripts/write-cuda-quality-comparison.py \
    --output "$manifest" \
    --binary "$binary" \
    --receipt-verifier "$receipt_verifier" \
    --source-commit "$source_commit" \
    --build-info "$build_info_path" \
    --platform linux-x86_64 \
    --statistics-platform "$statistics_platform" \
    --target "$target" \
    --statistics-target "$statistics_target" \
    --trusted-inputs "$trusted_inputs" \
    --generation-record "$generation_record" \
    --model-family "$model_family" \
    --subject-model "$subject_model" \
    --oracle-model "$oracle_model" \
    --corpus "$corpus" \
    --tokens "$tokens" \
    --oracle-logits "$oracle_logits" \
    --llama-logits "$llama_logits" \
    --leone-logits "$leone_logits" \
    --oracle-manifest "$oracle_manifest" \
    --llama-manifest "$llama_manifest" \
    --leone-manifest "$leone_metadata" \
    --task-manifest "$task_manifest" \
    --leone-backend "$leone_backend"
  python3 scripts/validate-quality-stage.py comparison "$manifest" "$receipt_verifier" \
    --source-commit "$source_commit" \
    --platform linux-x86_64 \
    --statistics-platform "$statistics_platform" \
    --target "$target" \
    --statistics-target "$statistics_target" \
    --trusted-inputs "$trusted_inputs" \
    --generation-record "$generation_record" \
    --backend cuda \
    --adapter-path research/oracle/llama_logits.cpp \
    --adapter-sha256 "$(sha256sum "$root/research/oracle/llama_logits.cpp" | cut -d' ' -f1)" \
    --model-family "$model_family" \
    --model-sha256 "$subject_sha256" \
    --oracle-model-sha256 "$oracle_sha256" \
    --corpus-sha256 "$corpus_sha256" \
    --task-manifest-sha256 "$(sha256sum "$task_manifest" | cut -d' ' -f1)" \
    --sample-contract "linspace-inclusive-v1:128+task-rows" \
    --metric-family kld >/dev/null
  exit 0
fi
leone_quality=$(quality_json "$tmp/leone-quality.txt" leone)
llama_quality=$(quality_json "$tmp/llama-quality.txt" llama)
mkdir -p "$(dirname "$manifest")"
jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_commit "$source_commit" \
  --arg executable "$public_binary_path" \
  --arg executable_sha256 "$binary_sha256" \
  --argjson build_info "$build_info" \
  --arg subject_model "$public_subject_model_path" \
  --arg subject_sha256 "$subject_sha256" \
  --arg oracle_model "$public_oracle_model_path" \
  --arg oracle_sha256 "$oracle_sha256" \
  --arg corpus "$public_corpus_path" \
  --arg corpus_sha256 "$corpus_sha256" \
  --arg tokens "$public_tokens_path" \
  --arg tokens_sha256 "$tokens_sha256" \
  --argjson token_count "$token_limit" \
  --argjson window "$window" \
  --arg oracle_manifest "$public_oracle_manifest_path" \
  --arg oracle_manifest_sha256 "$oracle_manifest_sha256" \
  --argjson oracle_manifest_body "$oracle_manifest_body" \
  --arg llama_manifest "$public_llama_manifest_path" \
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
