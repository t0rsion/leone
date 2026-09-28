#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

if [[ $# -ne 7 ]]; then
  echo "usage: $0 ORACLE_STAGE METAL_STAGE ORACLE_MODEL SUBJECT_MODEL CORPUS TASK_MANIFEST OUTPUT_MANIFEST" >&2
  exit 2
fi

oracle_stage=$1
metal_stage=$2
oracle_model=$3
subject_model=$4
corpus=$5
task_manifest=$6
manifest=$7
binary=${LEONE_BINARY:-$root/target/release/leone}
receipt_verifier=${LEONE_RECEIPT_VERIFIER:-$root/target/release/leone-receipt-verify}
expected_native_source_commit=${LEONE_EXPECTED_SOURCE_COMMIT:-}
expected_platform=${LEONE_EXPECTED_PLATFORM:-}
expected_target=${LEONE_EXPECTED_TARGET:-}
expected_statistics_target=${LEONE_EXPECTED_STATISTICS_TARGET:-}
expected_backend=${LEONE_EXPECTED_BACKEND:-metal}
expected_adapter_sha256=${LEONE_EXPECTED_ADAPTER_SHA256:-}
expected_model_family=${LEONE_EXPECTED_MODEL_FAMILY:-}
expected_model_sha256=${LEONE_EXPECTED_MODEL_SHA256:-}
trusted_oracle_model_sha256=${LEONE_EXPECTED_ORACLE_MODEL_SHA256:-}
trusted_corpus_sha256=${LEONE_EXPECTED_CORPUS_SHA256:-}
expected_sample_manifest_sha256=${LEONE_EXPECTED_SAMPLE_MANIFEST_SHA256:-}
generation_record=${LEONE_QUALITY_GENERATION_RECORD:-}
expected_sample_contract=${LEONE_EXPECTED_SAMPLE_CONTRACT:-linspace-inclusive-v1:128+task-rows}
expected_metric_family=${LEONE_EXPECTED_METRIC_FAMILY:-kld}

absolute_existing_file() {
  python3 - "$1" <<'PY'
import os
import sys

path = os.path.abspath(sys.argv[1])
if not os.path.isfile(path):
    raise SystemExit(f"required input is missing: {sys.argv[1]}")
print(path)
PY
}

absolute_existing_dir() {
  python3 - "$1" <<'PY'
import os
import sys

path = os.path.abspath(sys.argv[1])
if not os.path.isdir(path):
    raise SystemExit(f"stage directory is missing: {sys.argv[1]}")
print(path)
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

publish_no_replace() {
  python3 - "$1" "$2" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
if not source.is_file() or source.is_symlink():
    raise SystemExit(f"publication source is not a regular file: {source}")
if target.exists() or target.is_symlink():
    raise SystemExit(f"refusing to replace existing publication: {target}")
if not target.parent.is_dir() or target.parent.is_symlink():
    raise SystemExit(f"publication parent is not a regular directory: {target.parent}")
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
temporary = Path(temporary_name)
try:
    with source.open("rb") as original, os.fdopen(descriptor, "wb") as copy:
        shutil.copyfileobj(original, copy)
        copy.flush()
        os.fchmod(copy.fileno(), source.stat().st_mode & 0o777)
        os.fsync(copy.fileno())
        identity = os.fstat(copy.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise SystemExit(f"refusing to replace existing publication: {target}") from error
finally:
    temporary.unlink(missing_ok=True)
print(f"{identity.st_dev}:{identity.st_ino}")
PY
}

verify_stage_logits() {
  local stage=$1
  local manifest_name=$2
  local section=$3
  local label=$4
  local path expected actual
  path=$(jq -er ".$section.logits.path" "$stage/$manifest_name")
  expected=$(jq -er ".$section.logits.sha256" "$stage/$manifest_name")
  actual=$(file_sha256 "$stage/$path")
  if [[ $actual != "$expected" ]]; then
    echo "error: $label logits changed after stage capture" >&2
    exit 1
  fi
}

stable_name() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.basename(os.path.realpath(sys.argv[1])))
PY
}

oracle_stage=$(absolute_existing_dir "$oracle_stage")
metal_stage=$(absolute_existing_dir "$metal_stage")
oracle_model=$(absolute_existing_file "$oracle_model")
subject_model=$(absolute_existing_file "$subject_model")
corpus=$(absolute_existing_file "$corpus")
task_manifest=$(absolute_existing_file "$task_manifest")
binary=$(absolute_existing_file "$binary")
receipt_verifier=$(absolute_existing_file "$receipt_verifier")
if [[ ! -x $binary ]]; then
  echo "error: Leone executable is not executable: $binary" >&2
  exit 1
fi
if [[ ! -x $receipt_verifier ]]; then
  echo "error: standalone receipt verifier is not executable: $receipt_verifier" >&2
  exit 1
fi
if ! [[ $expected_native_source_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: LEONE_EXPECTED_SOURCE_COMMIT must contain the expected 40-character Leone commit" >&2
  exit 2
fi
if [[ -z $expected_platform || -z $expected_target || -z $expected_statistics_target || $expected_backend != metal || -z $expected_model_family || -z $expected_adapter_sha256 || -z $expected_model_sha256 || -z $trusted_oracle_model_sha256 || -z $trusted_corpus_sha256 || $expected_sample_contract != "linspace-inclusive-v1:128+task-rows" || $expected_metric_family != kld ]]; then
  echo "error: trusted native and statistics targets, source, backend, adapter, model, corpus, and sample inputs are required" >&2
  exit 2
fi
if ! [[ $expected_adapter_sha256 =~ ^[0-9a-f]{64}$ && $expected_model_sha256 =~ ^[0-9a-f]{64}$ && $trusted_oracle_model_sha256 =~ ^[0-9a-f]{64}$ && $trusted_corpus_sha256 =~ ^[0-9a-f]{64}$ ]]; then
  echo "error: trusted adapter, model, oracle, and corpus values must be lowercase SHA-256 digests" >&2
  exit 2
fi
if [[ -n $expected_sample_manifest_sha256 ]] && ! [[ $expected_sample_manifest_sha256 =~ ^[0-9a-f]{64}$ ]]; then
  echo "error: LEONE_EXPECTED_SAMPLE_MANIFEST_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if [[ $expected_adapter_sha256 != "$(file_sha256 "$root/research/oracle/llama_logits.cpp")" ]]; then
  echo "error: trusted adapter digest differs from research/oracle/llama_logits.cpp" >&2
  exit 1
fi
manifest=$(python3 - "$manifest" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
)
manifest_dir=$(dirname "$manifest")
if [[ -n $generation_record ]]; then
  generation_record=$(python3 - "$generation_record" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
if path.exists() or path.is_symlink():
    raise SystemExit(f"generation record already exists: {path}")
parent = path.parent
if not parent.is_dir() or parent.is_symlink():
    raise SystemExit(f"generation record parent is not a regular directory: {parent}")
print(os.path.abspath(path))
PY
  )
  if [[ $generation_record == "$manifest" ]]; then
    echo "error: generation record and comparison manifest must be different files" >&2
    exit 2
  fi
fi
case "$manifest_dir/" in
  "$oracle_stage/"*|"$metal_stage/"*)
    echo "error: comparison output must be outside the input stage directories" >&2
    exit 2
    ;;
esac
if [[ -e $manifest ]]; then
  echo "error: refusing to replace an existing comparison manifest: $manifest" >&2
  exit 2
fi
scripts/validate-quality-stage.py compare "$oracle_stage" "$metal_stage" >/dev/null

expected_oracle_model_sha256=$(jq -er '.model.oracle.sha256' "$oracle_stage/oracle-stage.json")
expected_subject_sha256=$(jq -er '.model.subject.sha256' "$oracle_stage/oracle-stage.json")
expected_corpus_sha256=$(jq -er '.corpus.sha256' "$oracle_stage/oracle-stage.json")
if [[ $expected_subject_sha256 != "$expected_model_sha256" ]]; then
  echo "error: trusted subject model digest differs from the exported model" >&2
  exit 1
fi
if [[ $(jq -er '.model.oracle.sha256' "$oracle_stage/oracle-stage.json") != "$trusted_oracle_model_sha256" ]]; then
  echo "error: trusted oracle model digest differs from the exported model" >&2
  exit 1
fi
if [[ $expected_corpus_sha256 != "$trusted_corpus_sha256" ]]; then
  echo "error: trusted corpus digest differs from the exported corpus" >&2
  exit 1
fi
if [[ $(file_sha256 "$oracle_model") != "$expected_oracle_model_sha256" ]]; then
  echo "error: supplied oracle model differs from the exported model" >&2
  exit 1
fi
if [[ $(file_sha256 "$subject_model") != "$expected_subject_sha256" ]]; then
  echo "error: supplied subject model differs from the exported model" >&2
  exit 1
fi
if [[ $(file_sha256 "$corpus") != "$expected_corpus_sha256" ]]; then
  echo "error: supplied corpus differs from the exported corpus" >&2
  exit 1
fi

build_info=$("$binary" --build-info)
if ! jq -e '
  .schema_version == "leone.build-info.v1" and
  (.source_commit | type == "string" and length == 40) and
  (.source_tree_dirty == false) and
  (.profile == "release")
' <<<"$build_info" >/dev/null; then
  echo "error: Leone executable returned invalid or dirty release metadata" >&2
  exit 1
fi
if [[ $(jq -er '.target' <<<"$build_info") != "$expected_statistics_target" ]]; then
  echo "error: statistics executable target differs from trusted statistics target" >&2
  exit 1
fi
statistics_source_commit=$(jq -er '.source_commit' <<<"$build_info")
if [[ $statistics_source_commit != "$expected_native_source_commit" ]]; then
  echo "error: statistics build source differs from the trusted Leone source" >&2
  exit 1
fi
native_source_commit=$(jq -er '.native_source_commit' "$metal_stage/metal-stage.json")
if [[ $native_source_commit != "$expected_native_source_commit" ]]; then
  echo "error: staged native source differs from LEONE_EXPECTED_SOURCE_COMMIT" >&2
  exit 1
fi

mkdir -p "$manifest_dir"
manifest_base=$(basename "${manifest%.*}")
llama_quality_name="$manifest_base-llama-quality.json"
leone_quality_name="$manifest_base-leone-quality.json"
oracle_stage_package_name="$manifest_base-oracle-stage"
metal_stage_package_name="$manifest_base-metal-stage"
samples_package_name="$manifest_base-samples"
task_manifest_name="$manifest_base-long-context-task.json"
oracle_stage_name="$oracle_stage_package_name/oracle-stage.json"
metal_stage_name="$metal_stage_package_name/metal-stage.json"
samples_manifest_name="$samples_package_name/sampling.json"
llama_quality_target="$manifest_dir/$llama_quality_name"
leone_quality_target="$manifest_dir/$leone_quality_name"
oracle_stage_package_target="$manifest_dir/$oracle_stage_package_name"
metal_stage_package_target="$manifest_dir/$metal_stage_package_name"
samples_package_target="$manifest_dir/$samples_package_name"
task_manifest_target="$manifest_dir/$task_manifest_name"
if [[ -e $llama_quality_target || -e $leone_quality_target || -e $oracle_stage_package_target || -e $metal_stage_package_target || -e $samples_package_target || -e $task_manifest_target ]]; then
  echo "error: refusing to replace an existing comparison artifact" >&2
  exit 2
fi

tmp=$(mktemp -d "$root/target/metal-quality-compare.XXXXXX")
publication_root="$tmp/public"
mkdir "$publication_root"
llama_quality_path="$publication_root/$llama_quality_name"
leone_quality_path="$publication_root/$leone_quality_name"
oracle_stage_package_path="$publication_root/$oracle_stage_package_name"
metal_stage_package_path="$publication_root/$metal_stage_package_name"
samples_package_path="$publication_root/$samples_package_name"
task_manifest_path="$publication_root/$task_manifest_name"
oracle_stage_path="$oracle_stage_package_path/oracle-stage.json"
metal_stage_path="$metal_stage_package_path/metal-stage.json"
manifest_tmp=
published=0
owned_file_paths=()
owned_file_identities=()
owned_directory_paths=()
owned_directory_identities=()
samples_package_tmp="$publication_root/$samples_package_name"
remove_owned_path() {
  python3 - "$1" "$2" "$3" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
device, inode = (int(value) for value in sys.argv[2].split(":", 1))
kind = sys.argv[3]
try:
    current = path.lstat()
except FileNotFoundError:
    raise SystemExit(0)
if (current.st_dev, current.st_ino) != (device, inode):
    raise SystemExit(0)
try:
    if kind == "directory":
        path.rmdir()
    else:
        path.unlink()
except OSError:
    pass
PY
}
record_owned_file() {
  owned_file_paths+=("$1")
  owned_file_identities+=("$2")
}
record_owned_directory() {
  owned_directory_paths+=("$1")
  owned_directory_identities+=("$2")
}
publish_owned_file() {
  local identity
  identity=$(publish_no_replace "$1" "$2")
  record_owned_file "$2" "$identity"
}
publish_owned_directory() {
  local source_dir=$1
  local target_dir=$2
  local identity source_file relative target_file
  if [[ -e $target_dir || -L $target_dir ]]; then
    echo "error: refusing to replace existing publication: $target_dir" >&2
    exit 1
  fi
  mkdir "$target_dir"
  identity=$(python3 - "$target_dir" <<'PY'
import os
import sys

record = os.lstat(sys.argv[1])
print(f"{record.st_dev}:{record.st_ino}")
PY
  )
  record_owned_directory "$target_dir" "$identity"
  while IFS= read -r -d '' source_file; do
    relative=${source_file#"$source_dir/"}
    target_file="$target_dir/$relative"
    publish_owned_file "$source_file" "$target_file"
  done < <(find "$source_dir" -type f -print0)
}
remove_owned_outputs() {
  local index
  for ((index=${#owned_file_paths[@]} - 1; index >= 0; index--)); do
    remove_owned_path "${owned_file_paths[index]}" "${owned_file_identities[index]}" file
  done
  for ((index=${#owned_directory_paths[@]} - 1; index >= 0; index--)); do
    remove_owned_path "${owned_directory_paths[index]}" "${owned_directory_identities[index]}" directory
  done
}
cleanup() {
  if (( published == 0 )); then
    remove_owned_outputs
  fi
  rm -rf -- "$tmp"
}
trap cleanup EXIT HUP INT TERM
ln -s "$oracle_model" "$tmp/oracle-model.gguf"
ln -s "$subject_model" "$tmp/subject-model.gguf"
ln -s "$corpus" "$tmp/corpus.txt"
python3 scripts/write-quality-samples.py \
  --oracle-stage "$oracle_stage" \
  --metal-stage "$metal_stage" \
  --task-manifest "$task_manifest" \
  --output-dir "$samples_package_tmp" \
  --path-prefix "$samples_package_name" >/dev/null
verify_stage_logits "$oracle_stage" oracle-stage.json oracle "oracle"
verify_stage_logits "$metal_stage" metal-stage.json subject "llama Metal"
verify_stage_logits "$metal_stage" metal-stage.json native "Leone Metal"
ln -s "$samples_package_tmp/sample.tokens.u32le" "$tmp/tokens.u32le"
ln -s "$samples_package_tmp/oracle.logits.bin" "$tmp/oracle.f32"
ln -s "$samples_package_tmp/llama-cpp-metal.logits.bin" "$tmp/metal.f32"
ln -s "$samples_package_tmp/leone-metal.logits.bin" "$tmp/leone-metal.f32"
task_result=$(python3 scripts/validate-quality-stage.py task "$task_manifest" "$oracle_stage" "$metal_stage" "$samples_package_tmp/sampling.json")
sample_manifest_sha256=$(file_sha256 "$samples_package_tmp/sampling.json")
if [[ -z $expected_sample_manifest_sha256 && -z $generation_record ]]; then
  echo "error: provide LEONE_EXPECTED_SAMPLE_MANIFEST_SHA256 or LEONE_QUALITY_GENERATION_RECORD" >&2
  exit 2
fi
if [[ -n $expected_sample_manifest_sha256 && $expected_sample_manifest_sha256 != "$sample_manifest_sha256" ]]; then
  echo "error: generated sample manifest differs from LEONE_EXPECTED_SAMPLE_MANIFEST_SHA256" >&2
  exit 1
fi
trusted_sample_manifest_sha256=${expected_sample_manifest_sha256:-$sample_manifest_sha256}
expected_oracle_sample_sha256=$(jq -er '.artifacts.logits.oracle.sha256' "$samples_package_tmp/sampling.json")
oracle_dtype=$(jq -er '.model.oracle.storage_type | ascii_downcase' "$oracle_stage/oracle-stage.json")
llama_pin=$(cat "$root/external/PINNED")

run_quality() {
  local subject_logits=$1
  local subject_engine=$2
  local subject_commit=$3
  local output=$4
  (
    cd "$tmp"
    "$binary" quality \
      --oracle oracle.f32 \
      --subject "$subject_logits" \
      --corpus corpus.txt \
      --tokens tokens.u32le \
      --oracle-model oracle-model.gguf \
      --subject-model subject-model.gguf \
      --oracle-engine llama.cpp \
      --oracle-commit "$llama_pin" \
      --oracle-dtype "$oracle_dtype" \
      --subject-engine "$subject_engine" \
      --subject-commit "$subject_commit" \
      --receipt >"$output"
  )
}

receipt_file() {
  local output=$1
  local path
  path=$(awk '$1 == "receipt:" { print $2 }' "$output")
  if [[ -z $path ]]; then
    echo "error: quality comparison did not produce a linked receipt" >&2
    exit 1
  fi
  if [[ $path != /* ]]; then
    path="$tmp/$path"
  fi
  if [[ ! -f $path ]]; then
    echo "error: quality receipt path is missing: $path" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

verify_quality_receipt_schema() {
  "$receipt_verifier" quality "$1" >/dev/null
}

verify_quality_artifact_link() {
  local receipt=$1
  local logits=$2
  if [[ $(jq -er '.oracle.artifact_sha256' "$receipt") != "$expected_oracle_sample_sha256" ]]; then
    echo "error: quality receipt oracle artifact differs" >&2
    exit 1
  fi
  if [[ $(jq -er '.subject.model_artifact.sha256' "$receipt") != "$expected_subject_sha256" ]]; then
    echo "error: quality receipt subject model differs" >&2
    exit 1
  fi
  if [[ $(jq -er '.subject.logits_artifact.sha256' "$receipt") != "$(file_sha256 "$logits")" ]]; then
    echo "error: quality receipt subject logits differ" >&2
    exit 1
  fi
}

run_quality metal.f32 llama.cpp-metal "$llama_pin" "$tmp/llama-quality.stdout"
llama_receipt=$(receipt_file "$tmp/llama-quality.stdout")
verify_quality_receipt_schema "$llama_receipt"
verify_quality_artifact_link "$llama_receipt" "$samples_package_tmp/llama-cpp-metal.logits.bin"
run_quality leone-metal.f32 leone-metal "$native_source_commit" "$tmp/leone-quality.stdout"
leone_receipt=$(receipt_file "$tmp/leone-quality.stdout")
verify_quality_receipt_schema "$leone_receipt"
verify_quality_artifact_link "$leone_receipt" "$samples_package_tmp/leone-metal.logits.bin"

cp "$llama_receipt" "$llama_quality_path"
cp "$leone_receipt" "$leone_quality_path"
cp "$task_manifest" "$task_manifest_path"
mkdir "$oracle_stage_package_path" "$metal_stage_package_path"
cp "$oracle_stage/oracle-stage.json" "$oracle_stage/oracle.json" "$oracle_stage/tokens.u32le" "$oracle_stage/llama_logits.cpp" "$oracle_stage_package_path/"
cp "$metal_stage/metal-stage.json" "$metal_stage/metal.json" "$metal_stage/leone-metal.json" "$metal_stage/oracle-stage.json" "$metal_stage/tokens.u32le" "$metal_stage/llama_logits.cpp" "$metal_stage_package_path/"
native_shader_name=$(jq -er '.native.manifest.body.execution.shader.name' "$metal_stage/metal-stage.json")
native_machine_name=$(jq -er '.native.manifest.body.execution.machine.path' "$metal_stage/metal-stage.json")
native_output_name=$(jq -er '.native.manifest.body.execution.eval_stdout.path' "$metal_stage/metal-stage.json")
cp "$metal_stage/$native_shader_name" "$metal_stage/$native_machine_name" "$metal_stage/$native_output_name" "$metal_stage_package_path/"

verify_stage_logits "$oracle_stage" oracle-stage.json oracle "oracle"
verify_stage_logits "$metal_stage" metal-stage.json subject "llama Metal"
verify_stage_logits "$metal_stage" metal-stage.json native "Leone Metal"
export_manifest=$(cat "$oracle_stage/oracle-stage.json")
metal_manifest=$(cat "$metal_stage/metal-stage.json")
llama_run=$(jq -c '.subject.manifest.body' "$metal_stage/metal-stage.json")
native_run=$(jq -c '.native.manifest.body' "$metal_stage/metal-stage.json")
manifest_tmp=$(mktemp "$publication_root/.${manifest_base}.json.XXXXXX")
jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_commit "$statistics_source_commit" \
  --arg executable "$(stable_name "$binary")" \
  --arg executable_sha256 "$(file_sha256 "$binary")" \
  --argjson build_info "$build_info" \
  --argjson export_manifest "$export_manifest" \
  --argjson metal_manifest "$metal_manifest" \
  --argjson llama_run "$llama_run" \
  --argjson native_run "$native_run" \
  --arg corpus "$(stable_name "$corpus")" \
  --arg oracle_stage_name "$oracle_stage_name" \
  --arg metal_stage_name "$metal_stage_name" \
  --arg samples_manifest_name "$samples_manifest_name" \
  --arg samples_manifest_sha256 "$(file_sha256 "$samples_package_tmp/sampling.json")" \
  --argjson samples_manifest "$(cat "$samples_package_tmp/sampling.json")" \
  --arg task_manifest_name "$task_manifest_name" \
  --arg task_manifest_sha256 "$(file_sha256 "$task_manifest_path")" \
  --argjson task_manifest "$(cat "$task_manifest_path")" \
  --argjson task_result "$task_result" \
  --arg llama_quality_receipt "$llama_quality_name" \
  --arg llama_quality_receipt_sha256 "$(file_sha256 "$llama_quality_path")" \
  --argjson llama_quality "$(cat "$llama_quality_path")" \
  --arg leone_quality_receipt "$leone_quality_name" \
  --arg leone_quality_receipt_sha256 "$(file_sha256 "$leone_quality_path")" \
  --argjson leone_quality "$(cat "$leone_quality_path")" \
  --arg native_source_commit "$native_source_commit" \
  --arg receipt_parser_name "$(stable_name "$receipt_verifier")" \
  --arg receipt_parser_sha256 "$(file_sha256 "$receipt_verifier")" \
  --arg trusted_source_commit "$expected_native_source_commit" \
  --arg trusted_platform "$expected_platform" \
  --arg trusted_target "$expected_target" \
  --arg trusted_statistics_target "$expected_statistics_target" \
  --arg trusted_backend "$expected_backend" \
  --arg trusted_adapter_path "research/oracle/llama_logits.cpp" \
  --arg trusted_adapter_sha256 "$expected_adapter_sha256" \
  --arg trusted_model_family "$expected_model_family" \
  --arg trusted_model_sha256 "$expected_model_sha256" \
  --arg trusted_oracle_model_sha256 "$trusted_oracle_model_sha256" \
  --arg trusted_corpus_sha256 "$trusted_corpus_sha256" \
  --arg trusted_sample_manifest_sha256 "$trusted_sample_manifest_sha256" \
  --arg trusted_sample_contract "$expected_sample_contract" \
  --arg trusted_metric_family "$expected_metric_family" \
  --arg llama_source_commit "$(jq -er '.source_commit' <<<"$llama_run")" \
  --arg llama_pin "$llama_pin" \
  --arg oracle_source_commit "$(jq -er '.source_commit' "$oracle_stage/oracle.json")" \
  '{
    schema_version: "leone.quality-cross-device.v2",
    created_utc: $created_utc,
    source_commit: $source_commit,
    executable: {name: $executable, sha256: $executable_sha256},
    build_info: $build_info,
    source_identities: {
      oracle: {source_commit: $oracle_source_commit},
      llama_cpp_metal: {source_commit: $llama_source_commit},
      leone_metal: {source_commit: $native_source_commit, expected_source_commit: $native_source_commit},
      statistics: {source_commit: $source_commit}
    },
    models: {
      oracle: $export_manifest.model.oracle,
      subject: $metal_manifest.model.subject
    },
    corpus: {name: $corpus, sha256: $export_manifest.corpus.sha256},
    input: $export_manifest.input,
    oracle: {
      engine: $export_manifest.engine,
      execution: $export_manifest.oracle.manifest.body.execution,
      logits: $export_manifest.oracle.logits
    },
    subjects: {
      llama_cpp: {
        engine: {name: "llama.cpp-metal", git_commit: $llama_pin},
        execution: $metal_manifest.execution,
        logits: $metal_manifest.subject.logits
      },
      leone: {
        engine: {name: "leone-metal", git_commit: $native_source_commit},
        execution: $metal_manifest.native_execution,
        logits: $metal_manifest.native.logits
      }
    },
    stages: {
      oracle_export: {path: $oracle_stage_name, sha256: "", body: $export_manifest},
      metal_subject: {path: $metal_stage_name, sha256: "", body: $metal_manifest}
    },
    tasks: {
      long_context: {
        manifest: {path: $task_manifest_name, sha256: $task_manifest_sha256, body: $task_manifest},
        result: $task_result
      }
    },
    samples: {
      manifest: {path: $samples_manifest_name, sha256: $samples_manifest_sha256, body: $samples_manifest}
    },
    quality: {
      llama_cpp: {path: $llama_quality_receipt, sha256: $llama_quality_receipt_sha256, receipt: $llama_quality},
      leone: {path: $leone_quality_receipt, sha256: $leone_quality_receipt_sha256, receipt: $leone_quality}
    },
    validation: {
      mode: "offline",
      packaged_stage_manifests: true,
      generation_rehashed_original_artifacts: true,
      statistics_executable: {name: $executable, sha256: $executable_sha256},
      receipt_parser: {name: $receipt_parser_name, sha256: $receipt_parser_sha256, interface: "quality"},
      trusted: {
        source_commit: $trusted_source_commit,
        platform: $trusted_platform,
        target: $trusted_target,
        statistics_target: $trusted_statistics_target,
        backend: $trusted_backend,
        adapter_path: $trusted_adapter_path,
        adapter_sha256: $trusted_adapter_sha256,
        model_family: $trusted_model_family,
        model_sha256: $trusted_model_sha256,
        oracle_model_sha256: $trusted_oracle_model_sha256,
        corpus_sha256: $trusted_corpus_sha256,
        sample_manifest_sha256: $trusted_sample_manifest_sha256,
        sample_contract: $trusted_sample_contract,
        metric_family: $trusted_metric_family
      }
    },
    limits: [
      "The CPU full-precision oracle is compared with the pinned llama.cpp Metal comparator and the native Leone Metal backend.",
      "The two subjects have separate numerical quality receipts over the shared u32le token stream.",
      "Quality receipts cover the deterministic sampled rows recorded in the packaged sample manifest.",
      "The package retains source logits hashes and sampled row hashes. Offline replay covers sampled rows only.",
      "The llama.cpp result is comparator evidence. It does not certify the native Leone backend.",
      "Offline validation rechecks packaged stage manifests and linked receipts. It does not rerun the model or logits.",
      "The result makes no cross-device bitwise claim and does not certify concurrent service numerics.",
      "A shader smoke test is not a quality receipt."
    ]
  }' | jq \
    --arg export_sha256 "$(file_sha256 "$oracle_stage_path")" \
    --arg metal_sha256 "$(file_sha256 "$metal_stage_path")" \
    '.stages.oracle_export.sha256 = $export_sha256 | .stages.metal_subject.sha256 = $metal_sha256' >"$manifest_tmp"
python3 scripts/validate-quality-stage.py comparison "$manifest_tmp" "$receipt_verifier" \
  --source-commit "$expected_native_source_commit" \
  --platform "$expected_platform" \
  --target "$expected_target" \
  --statistics-target "$expected_statistics_target" \
  --backend "$expected_backend" \
  --adapter-path research/oracle/llama_logits.cpp \
  --adapter-sha256 "$expected_adapter_sha256" \
  --model-family "$expected_model_family" \
  --model-sha256 "$expected_model_sha256" \
  --oracle-model-sha256 "$trusted_oracle_model_sha256" \
  --corpus-sha256 "$trusted_corpus_sha256" \
  --sample-manifest-sha256 "$trusted_sample_manifest_sha256" \
  --sample-contract "$expected_sample_contract" \
  --metric-family "$expected_metric_family" >/dev/null
publish_owned_file "$llama_quality_path" "$llama_quality_target"
publish_owned_file "$leone_quality_path" "$leone_quality_target"
publish_owned_file "$task_manifest_path" "$task_manifest_target"
publish_owned_directory "$oracle_stage_package_path" "$oracle_stage_package_target"
publish_owned_directory "$metal_stage_package_path" "$metal_stage_package_target"
publish_owned_directory "$samples_package_path" "$samples_package_target"
if [[ -n $generation_record ]]; then
  generation_identity=$(python3 scripts/write-metal-quality-generation.py "$manifest_tmp" "$generation_record" --identity)
  record_owned_file "$generation_record" "$generation_identity"
fi
publish_owned_file "$manifest_tmp" "$manifest"
manifest_tmp=
published=1
echo "$manifest"
