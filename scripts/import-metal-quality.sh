#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ORACLE_STAGE SUBJECT_MODEL OUTPUT_DIR" >&2
  exit 2
fi

oracle_stage=$1
subject_model=$2
metal_stage=$3
leone_binary=${LEONE_BINARY:-$root/target/release/leone}
expected_native_source_commit=${LEONE_EXPECTED_SOURCE_COMMIT:-}
shader_path=${LEONE_METAL_SHADER_PATH:-}
native_prefill_chunk=${LEONE_METAL_PREFILL_CHUNK:-}

absolute_existing_path() {
  python3 - "$1" <<'PY'
import os
import sys

path = os.path.abspath(sys.argv[1])
if not os.path.isdir(path):
    raise SystemExit(f"stage directory is missing: {sys.argv[1]}")
print(path)
PY
}

absolute_file_path() {
  python3 - "$1" <<'PY'
import os
import sys

path = os.path.abspath(sys.argv[1])
if not os.path.isfile(path):
    raise SystemExit(f"required file is missing: {sys.argv[1]}")
print(path)
PY
}

file_size() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.stat(sys.argv[1]).st_size)
PY
}

file_sha256() {
  python3 - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

stable_name() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.basename(os.path.realpath(sys.argv[1])))
PY
}

oracle_stage=$(absolute_existing_path "$oracle_stage")
subject_model=$(absolute_file_path "$subject_model")
if [[ $leone_binary != /* ]]; then
  leone_binary="$root/$leone_binary"
fi
leone_binary=$(absolute_file_path "$leone_binary")
if [[ ! -x $leone_binary ]]; then
  echo "error: Leone executable is not executable: $leone_binary" >&2
  exit 1
fi
if ! [[ $expected_native_source_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: LEONE_EXPECTED_SOURCE_COMMIT must contain the expected 40-character Leone commit" >&2
  exit 2
fi
native_prefill_args=()
native_prefill_path=sequential
native_prefill_chunk_json=null
if [[ -n $native_prefill_chunk ]]; then
  if ! [[ $native_prefill_chunk =~ ^[1-9][0-9]*$ ]]; then
    echo "error: LEONE_METAL_PREFILL_CHUNK must be a positive integer" >&2
    exit 2
  fi
  native_prefill_args=(--prefill-chunk "$native_prefill_chunk")
  native_prefill_path=chunked
  native_prefill_chunk_json=$native_prefill_chunk
fi
metal_stage=$(python3 - "$metal_stage" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
)
if [[ -e $metal_stage ]]; then
  echo "error: output stage already exists: $metal_stage" >&2
  exit 2
fi
scripts/validate-quality-stage.py export "$oracle_stage" >/dev/null
expected_subject_sha256=$(jq -er '.model.subject.sha256' "$oracle_stage/oracle-stage.json")
actual_subject_sha256=$(file_sha256 "$subject_model")
if [[ $actual_subject_sha256 != "$expected_subject_sha256" ]]; then
  echo "error: subject model SHA-256 differs from the exported model" >&2
  exit 1
fi
leone_build_info=$("$leone_binary" --build-info)
if ! jq -e '
  .schema_version == "leone.build-info.v1" and
  (.source_commit | type == "string" and length == 40) and
  (.source_tree_dirty == false) and
  (.profile == "release")
' <<<"$leone_build_info" >/dev/null; then
  echo "error: Leone executable is not a clean release build" >&2
  exit 1
fi
if [[ $(jq -er '.source_commit' <<<"$leone_build_info") != "$expected_native_source_commit" ]]; then
  echo "error: Leone executable source commit differs from LEONE_EXPECTED_SOURCE_COMMIT" >&2
  exit 1
fi
shader_identity=$(jq -e '.metal_shader | objects' <<<"$leone_build_info") || {
  echo "error: Leone build info lacks the actual Metal shader identity" >&2
  exit 1
}
shader_name=$(jq -er '.name' <<<"$shader_identity")
shader_sha256=$(jq -er '.sha256' <<<"$shader_identity")
shader_bytes=$(jq -er '.bytes' <<<"$shader_identity")
if [[ $shader_path == "" ]]; then
  echo "error: LEONE_METAL_SHADER_PATH must identify the shader artifact reported by the executable" >&2
  exit 2
fi
if [[ $shader_path != /* ]]; then
  shader_path="$root/$shader_path"
fi
shader_path=$(absolute_file_path "$shader_path")
if [[ $(stable_name "$shader_path") != "$shader_name" || $(file_sha256 "$shader_path") != "$shader_sha256" || $(file_size "$shader_path") != "$shader_bytes" ]]; then
  echo "error: supplied Metal shader artifact differs from the executable build identity" >&2
  exit 1
fi

mkdir "$metal_stage"
remove_stage=1
native_tmp=
cleanup() {
  if [[ -n $native_tmp ]]; then
    rm -rf -- "$native_tmp"
  fi
  if (( remove_stage )); then
    rm -rf -- "$metal_stage"
  fi
}
trap cleanup EXIT HUP INT TERM
cp "$oracle_stage/oracle-stage.json" "$metal_stage/oracle-stage.json"
# APFS clones keep the oracle independent without duplicating its disk blocks.
if ! cp -c "$oracle_stage/oracle.f32" "$metal_stage/oracle.f32" 2>/dev/null; then
  cp "$oracle_stage/oracle.f32" "$metal_stage/oracle.f32"
fi
cp "$oracle_stage/tokens.u32le" "$metal_stage/tokens.u32le"
cp "$oracle_stage/llama_logits.cpp" "$metal_stage/llama_logits.cpp"
cp "$shader_path" "$metal_stage/$shader_name"

scripts/run-llama-oracle.sh \
  "$subject_model" "$metal_stage/tokens.u32le" "$metal_stage/metal.f32" "$metal_stage/metal.json" \
  "$(jq -er '.input.window_tokens' "$oracle_stage/oracle-stage.json")" metal >/dev/null
run_manifest=$(cat "$metal_stage/metal.json")
if [[ $(jq -er '.execution.device' "$metal_stage/metal.json") != metal ]]; then
  echo "error: comparator did not report a Metal execution" >&2
  exit 1
fi
backend_registry=$(jq -er '.execution.backend_registry' "$metal_stage/metal.json" | tr '[:lower:]' '[:upper:]')
if [[ $backend_registry != MTL && $backend_registry != METAL ]]; then
  echo "error: comparator did not report a Metal backend registry" >&2
  exit 1
fi

mkdir -p "$root/target"
native_tmp=$(mktemp -d "$root/target/metal-quality-native.XXXXXX")
ln -s "$subject_model" "$native_tmp/subject-model.gguf"
ln -s "$metal_stage/tokens.u32le" "$native_tmp/tokens.u32le"
if ! (
  cd "$native_tmp"
  "$leone_binary" doctor --backend metal >leone-doctor.txt 2>leone-doctor.stderr
); then
  cat "$native_tmp/leone-doctor.stderr" >&2
  exit 1
fi
platform_lines=$(grep -Ec '^platform:' "$native_tmp/leone-doctor.txt" || true)
darwin_platform_lines=$(grep -Ec '^platform: darwin arm64$' "$native_tmp/leone-doctor.txt" || true)
backend_lines=$(grep -Ec '^selected backend:' "$native_tmp/leone-doctor.txt" || true)
metal_backend_lines=$(grep -Ec '^selected backend: metal$' "$native_tmp/leone-doctor.txt" || true)
device_lines=$(grep -Ec '^metal device:' "$native_tmp/leone-doctor.txt" || true)
detected_device_lines=$(grep -Ec '^metal device: detected$' "$native_tmp/leone-doctor.txt" || true)
if [[ $platform_lines != 1 || $darwin_platform_lines != 1 ||
      $backend_lines != 1 || $metal_backend_lines != 1 ||
      $device_lines != 1 || $detected_device_lines != 1 ]]; then
  echo "error: Leone doctor did not confirm the native Metal device" >&2
  cat "$native_tmp/leone-doctor.txt" >&2
  exit 1
fi
if ! (
  cd "$native_tmp"
  "$leone_binary" eval \
    -m subject-model.gguf \
    --tokens tokens.u32le \
    --window "$(jq -er '.input.window_tokens' "$oracle_stage/oracle-stage.json")" \
    --backend metal \
    "${native_prefill_args[@]}" \
    --logits leone-metal.f32 \
    >leone-eval.stdout 2>leone-eval.stderr
); then
  cat "$native_tmp/leone-eval.stderr" >&2
  exit 1
fi
mv "$native_tmp/leone-metal.f32" "$metal_stage/leone-metal.f32"
cp "$native_tmp/leone-doctor.txt" "$metal_stage/leone-doctor.txt"
cp "$native_tmp/leone-eval.stdout" "$metal_stage/leone-eval.stdout"
native_rows=$(jq -er '.input.rows' "$oracle_stage/oracle-stage.json")
native_vocab=$(jq -er '.input.vocab_size' "$oracle_stage/oracle-stage.json")
expected_native_bytes=$((native_rows * native_vocab * 4))
native_bytes=$(file_size "$metal_stage/leone-metal.f32")
if [[ $native_bytes != "$expected_native_bytes" ]]; then
  echo "error: Leone Metal logits have $native_bytes bytes, expected $expected_native_bytes" >&2
  exit 1
fi

oracle_stage_manifest=$(cat "$oracle_stage/oracle-stage.json")
leone_manifest=$(jq -n \
  --arg source_commit "$(jq -er '.source_commit' <<<"$leone_build_info")" \
  --arg executable "$(stable_name "$leone_binary")" \
  --arg executable_sha256 "$(file_sha256 "$leone_binary")" \
  --argjson build_info "$leone_build_info" \
  --arg model "$(stable_name "$subject_model")" \
  --arg model_sha256 "$actual_subject_sha256" \
  --arg storage_type "$(jq -er '.model.storage_type' "$metal_stage/metal.json")" \
  --arg architecture "$(jq -er '.model.architecture' "$metal_stage/metal.json")" \
  --arg tokenizer "$(jq -er '.model.tokenizer' "$metal_stage/metal.json")" \
  --argjson model_vocab "$(jq -er '.model.vocab_size' "$metal_stage/metal.json")" \
  --arg tokens "tokens.u32le" \
  --arg token_sha256 "$(file_sha256 "$metal_stage/tokens.u32le")" \
  --argjson token_bytes "$(file_size "$metal_stage/tokens.u32le")" \
  --argjson token_count "$(jq -er '.input.tokens.count' "$oracle_stage/oracle-stage.json")" \
  --argjson window "$(jq -er '.input.window_tokens' "$oracle_stage/oracle-stage.json")" \
  --argjson stride "$(jq -er '.input.stride_tokens' "$oracle_stage/oracle-stage.json")" \
  --argjson rows "$native_rows" \
  --argjson vocab "$native_vocab" \
  --arg doctor_sha256 "$(file_sha256 "$metal_stage/leone-doctor.txt")" \
  --rawfile doctor_body "$metal_stage/leone-doctor.txt" \
  --arg stdout_sha256 "$(file_sha256 "$metal_stage/leone-eval.stdout")" \
  --rawfile stdout_body "$metal_stage/leone-eval.stdout" \
  --arg logits "leone-metal.f32" \
  --arg logits_sha256 "$(file_sha256 "$metal_stage/leone-metal.f32")" \
  --argjson logits_bytes "$native_bytes" \
  --arg shader_name "$shader_name" \
  --arg shader_sha256 "$shader_sha256" \
  --argjson shader_bytes "$shader_bytes" \
  --arg prefill_path "$native_prefill_path" \
  --argjson prefill_chunk "$native_prefill_chunk_json" \
  '{
    schema_version: "leone.native-metal-eval.v1",
    source_commit: $source_commit,
    executable: {name: $executable, sha256: $executable_sha256, build_info: $build_info},
    model: {path: $model, sha256: $model_sha256, storage_type: $storage_type, vocab_size: $model_vocab, architecture: $architecture, tokenizer: $tokenizer},
    input: {
      tokens: {path: $tokens, sha256: $token_sha256, bytes: $token_bytes, count: $token_count, encoding: "u32le"},
      window_tokens: $window,
      stride_tokens: $stride,
      rows: $rows,
      vocab_size: $vocab
    },
    execution: {
      engine: "leone",
      backend: "metal",
      backend_registry: "metal",
      device_type: "gpu",
      device: "metal",
      kv_cache_dtype: "F16",
      prefill: {path: $prefill_path, chunk_tokens: $prefill_chunk},
      shader: {name: $shader_name, sha256: $shader_sha256, bytes: $shader_bytes},
      machine: {path: "leone-doctor.txt", sha256: $doctor_sha256, body: $doctor_body},
      eval_stdout: {path: "leone-eval.stdout", sha256: $stdout_sha256, body: $stdout_body}
    },
    logits: {path: $logits, sha256: $logits_sha256, bytes: $logits_bytes, encoding: "row-major-f32-le"}
  }')
printf '%s\n' "$leone_manifest" >"$metal_stage/leone-metal.json"
leone_manifest_sha256=$(file_sha256 "$metal_stage/leone-metal.json")
jq -n \
  --arg source_commit "$(jq -er '.source_commit' "$metal_stage/metal.json")" \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg pin "$(cat external/PINNED)" \
  --arg adapter_sha256 "$(jq -er '.adapter.sha256' "$metal_stage/metal.json")" \
  --argjson adapter_bytes "$(file_size "$metal_stage/llama_logits.cpp")" \
  --arg oracle_stage_sha256 "$(file_sha256 "$oracle_stage/oracle-stage.json")" \
  --argjson oracle_stage "$oracle_stage_manifest" \
  --argjson oracle_model "$(jq -c '.model.oracle' "$oracle_stage/oracle-stage.json")" \
  --argjson subject_model_record "$(jq -c '.model.subject' "$oracle_stage/oracle-stage.json")" \
  --arg subject_storage_type "$(jq -er '.model.storage_type' "$metal_stage/metal.json")" \
  --arg subject_architecture "$(jq -er '.model.architecture' "$metal_stage/metal.json")" \
  --arg subject_tokenizer "$(jq -er '.model.tokenizer' "$metal_stage/metal.json")" \
  --argjson subject_vocab "$(jq -er '.model.vocab_size' "$metal_stage/metal.json")" \
  --argjson corpus "$(jq -c '.corpus' "$oracle_stage/oracle-stage.json")" \
  --argjson input "$(jq -c '.input' "$oracle_stage/oracle-stage.json")" \
  --arg token_sha256 "$(file_sha256 "$metal_stage/tokens.u32le")" \
  --argjson token_bytes "$(file_size "$metal_stage/tokens.u32le")" \
  --arg oracle_logits_sha256 "$(file_sha256 "$metal_stage/oracle.f32")" \
  --argjson oracle_logits_bytes "$(file_size "$metal_stage/oracle.f32")" \
  --arg metal_manifest_sha256 "$(file_sha256 "$metal_stage/metal.json")" \
  --argjson metal_manifest "$run_manifest" \
  --arg metal_logits_sha256 "$(file_sha256 "$metal_stage/metal.f32")" \
  --argjson metal_logits_bytes "$(file_size "$metal_stage/metal.f32")" \
  --argjson leone_manifest "$leone_manifest" \
  --arg leone_manifest_sha256 "$leone_manifest_sha256" \
  --arg native_source_commit "$(jq -er '.source_commit' <<<"$leone_manifest")" \
  --arg leone_logits_sha256 "$(file_sha256 "$metal_stage/leone-metal.f32")" \
  --argjson leone_logits_bytes "$(file_size "$metal_stage/leone-metal.f32")" \
  '{
    schema_version: "leone.quality-stage.v1",
    stage: "metal-subject",
    created_utc: $created_utc,
    source_commit: $source_commit,
    native_source_commit: $native_source_commit,
    expected_native_source_commit: $native_source_commit,
    engine: {name: "llama.cpp", git_commit: $pin},
    adapter: {path: "research/oracle/llama_logits.cpp", sha256: $adapter_sha256, artifact: {path: "llama_logits.cpp", sha256: $adapter_sha256, bytes: $adapter_bytes}},
    parent: {
      path: "oracle-stage.json",
      manifest_sha256: $oracle_stage_sha256,
      manifest: $oracle_stage
    },
    model: {oracle: $oracle_model, subject: ($subject_model_record + {storage_type: $subject_storage_type, vocab_size: $subject_vocab, architecture: $subject_architecture, tokenizer: $subject_tokenizer})},
    corpus: $corpus,
    input: $input,
    oracle: {
      logits: {path: "oracle.f32", sha256: $oracle_logits_sha256, bytes: $oracle_logits_bytes, encoding: "row-major-f32-le"}
    },
    subject: {
      manifest: {path: "metal.json", sha256: $metal_manifest_sha256, body: $metal_manifest},
      logits: {path: "metal.f32", sha256: $metal_logits_sha256, bytes: $metal_logits_bytes, encoding: "row-major-f32-le"}
    },
    native: {
      manifest: {path: "leone-metal.json", sha256: $leone_manifest_sha256, body: $leone_manifest},
      logits: {path: "leone-metal.f32", sha256: $leone_logits_sha256, bytes: $leone_logits_bytes, encoding: "row-major-f32-le"}
    },
    execution: $metal_manifest.execution,
    native_execution: $leone_manifest.execution,
    limits: [
      "The CPU full-precision oracle is compared with both the pinned llama.cpp Metal comparator and the native Leone Metal backend.",
      "The staged comparison uses separate numerical quality receipts. It makes no cross-device bitwise claim.",
      "The llama.cpp comparison is comparator evidence. It does not certify the native Leone backend.",
      "The native Leone stage does not certify concurrent service numerics or a shader smoke test."
    ]
  }' >"$metal_stage/metal-stage.json"
scripts/validate-quality-stage.py metal "$metal_stage"
remove_stage=0
trap - EXIT HUP INT TERM
echo "$metal_stage/metal-stage.json"
