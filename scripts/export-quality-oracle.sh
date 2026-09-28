#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

if [[ $# -ne 7 && $# -ne 9 ]]; then
  echo "usage: $0 ORACLE_MODEL SUBJECT_MODEL SUBJECT_STORAGE_TYPE CORPUS TOKENS WINDOW OUTPUT_DIR [ORACLE_LOGITS ORACLE_MANIFEST]" >&2
  exit 2
fi

oracle_model=$1
subject_model=$2
subject_storage_type=$3
corpus=$4
tokens=$5
window=$6
stage=$7
oracle_logits_input=${8:-}
oracle_manifest_input=${9:-}

absolute_existing_path() {
  python3 - "$1" <<'PY'
import os
import sys

path = os.path.abspath(sys.argv[1])
if not os.path.isfile(path):
    raise SystemExit(f"required input is missing: {sys.argv[1]}")
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

oracle_model=$(absolute_existing_path "$oracle_model")
subject_model=$(absolute_existing_path "$subject_model")
corpus=$(absolute_existing_path "$corpus")
tokens=$(absolute_existing_path "$tokens")
if [[ -n $oracle_logits_input ]]; then
  oracle_logits_input=$(absolute_existing_path "$oracle_logits_input")
  oracle_manifest_input=$(absolute_existing_path "$oracle_manifest_input")
fi
if [[ -z $subject_storage_type ]]; then
  echo "error: subject storage type is required" >&2
  exit 2
fi
stage=$(python3 - "$stage" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
)
if [[ -e $stage ]]; then
  echo "error: output stage already exists: $stage" >&2
  exit 2
fi
if ! [[ $window =~ ^[1-9][0-9]*$ ]] || (( window < 2 )); then
  echo "error: window must be an integer of at least two" >&2
  exit 2
fi
token_bytes=$(file_size "$tokens")
if (( token_bytes < 8 || token_bytes % 4 != 0 )); then
  echo "error: token file must contain at least two little-endian u32 values" >&2
  exit 1
fi
token_count=$((token_bytes / 4))
if (( window >= token_count )); then
  echo "error: window exceeds scored rows" >&2
  exit 2
fi

mkdir "$stage"
remove_stage=1
cleanup() {
  if (( remove_stage )); then
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT HUP INT TERM
cp "$tokens" "$stage/tokens.u32le"
cp "$root/research/oracle/llama_logits.cpp" "$stage/llama_logits.cpp"
if [[ -n $oracle_logits_input ]]; then
  cp "$oracle_logits_input" "$stage/oracle.f32"
  python3 - "$oracle_manifest_input" "$stage/oracle.json" "$oracle_model" "$oracle_logits_input" "$tokens" "$window" "$root" "$corpus" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys

source = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[7])
corpus = Path(sys.argv[8])

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def require_hash(value, message):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, message)

require(source.get("schema_version") == "leone.llama-oracle.v2", "reused oracle manifest has the wrong schema")
require(isinstance(source.get("source_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", source["source_commit"]) is not None, "reused oracle manifest lacks source commit")
engine = source.get("engine")
require(isinstance(engine, dict) and engine.get("name") == "llama.cpp", "reused oracle manifest lacks llama.cpp provenance")
require(engine.get("git_commit") == Path(root / "external/PINNED").read_text().strip(), "reused oracle manifest llama pin differs")
adapter = source.get("adapter")
require(isinstance(adapter, dict) and adapter.get("path") == "research/oracle/llama_logits.cpp", "reused oracle manifest lacks adapter provenance")
require_hash(adapter.get("sha256"), "reused oracle adapter hash is invalid")
require(adapter["sha256"] == digest(root / adapter["path"]), "reused oracle adapter differs")
adapter_artifact = adapter.get("artifact")
require(isinstance(adapter_artifact, dict), "reused oracle adapter artifact is missing")
require(adapter_artifact.get("path") == "llama_logits.cpp", "reused oracle adapter artifact path differs")
require_hash(adapter_artifact.get("sha256"), "reused oracle adapter artifact hash is invalid")
require(adapter_artifact["sha256"] == adapter["sha256"], "reused oracle adapter artifact hash differs")
require(adapter_artifact.get("bytes") == (root / adapter["path"]).stat().st_size, "reused oracle adapter artifact size differs")
corpus_record = source.get("corpus")
require(isinstance(corpus_record, dict), "reused oracle manifest lacks corpus provenance")
require(isinstance(corpus_record.get("name"), str) and os.path.basename(corpus_record["name"]) == corpus_record["name"], "reused oracle corpus name is unstable")
require_hash(corpus_record.get("sha256"), "reused oracle corpus hash is invalid")
require(corpus_record["sha256"] == digest(corpus), "reused oracle corpus hash differs")
execution = source.get("execution")
require(isinstance(execution, dict), "reused oracle manifest lacks execution provenance")
for name in ("device", "backend_registry", "device_type", "device_name", "device_description", "device_id"):
    require(name in execution, f"reused oracle manifest lacks execution field: {name}")
require(isinstance(execution["device"], str) and isinstance(execution["backend_registry"], str), "reused oracle execution backend fields are invalid")
require(execution["device"] == "cpu" and execution["backend_registry"].lower() == "cpu", "reused oracle manifest is not a CPU oracle")
require(execution["device_type"] == "cpu", "reused oracle manifest device type differs")
require(isinstance(execution["device_name"], str) and execution["device_name"], "reused oracle device name is missing")
require(isinstance(execution["device_description"], str), "reused oracle device description is invalid")
require(execution["device_id"] is None or isinstance(execution["device_id"], str), "reused oracle device identifier is invalid")
model = source.get("model")
require(isinstance(model, dict), "reused oracle manifest lacks model provenance")
for name in ("path", "sha256", "storage_type"):
    require(name in model, f"reused oracle manifest lacks model field: {name}")
require(model["storage_type"] in ("BF16", "F16"), "reused oracle manifest is not full precision")
require_hash(model["sha256"], "reused oracle model hash is invalid")
require(model["sha256"] == digest(sys.argv[3]), "reused oracle model hash differs")
require(isinstance(model.get("vocab_size"), int) and not isinstance(model["vocab_size"], bool) and model["vocab_size"] > 0, "reused oracle model vocabulary is missing")
input_record = source.get("input")
require(isinstance(input_record, dict), "reused oracle manifest lacks input provenance")
tokens = input_record.get("tokens")
require(isinstance(tokens, dict), "reused oracle manifest lacks token provenance")
for name in ("path", "sha256", "bytes", "count", "encoding"):
    require(name in tokens, f"reused oracle manifest lacks token field: {name}")
require_hash(tokens["sha256"], "reused oracle token hash is invalid")
require(tokens["sha256"] == digest(sys.argv[5]), "reused oracle token hash differs")
require(tokens["encoding"] == "u32le", "reused oracle token encoding differs")
require(isinstance(tokens["count"], int) and not isinstance(tokens["count"], bool) and tokens["count"] >= 2, "reused oracle token count is invalid")
require(tokens["bytes"] == tokens["count"] * 4, "reused oracle token byte count differs")
require(tokens["count"] == os.stat(sys.argv[5]).st_size // 4 and os.stat(sys.argv[5]).st_size % 4 == 0, "reused oracle token count differs")
require(input_record.get("window_tokens") == int(sys.argv[6]), "reused oracle window differs")
require(isinstance(input_record.get("rows"), int) and not isinstance(input_record["rows"], bool) and input_record["rows"] > 0, "reused oracle row count is invalid")
require(input_record.get("rows") == tokens["count"] - 1, "reused oracle row count differs")
require(input_record["window_tokens"] <= input_record["rows"], "reused oracle window exceeds scored rows")
require(input_record.get("stride_tokens") == input_record["window_tokens"] - 1, "reused oracle stride differs")
require(input_record.get("vocab_size") == model.get("vocab_size"), "reused oracle vocabulary differs")
logits = source.get("logits")
require(isinstance(logits, dict), "reused oracle manifest lacks logits provenance")
for name in ("path", "sha256", "bytes", "encoding"):
    require(name in logits, f"reused oracle manifest lacks logits field: {name}")
require_hash(logits["sha256"], "reused oracle logits hash is invalid")
require(logits["sha256"] == digest(sys.argv[4]), "reused oracle logits hash differs")
require(isinstance(logits["bytes"], int) and not isinstance(logits["bytes"], bool) and logits["bytes"] > 0, "reused oracle logits byte count is invalid")
require(logits["bytes"] == os.stat(sys.argv[4]).st_size, "reused oracle logits byte count differs")
require(logits["encoding"] == "row-major-f32-le", "reused oracle logits encoding differs")
require(logits["bytes"] == input_record["rows"] * model["vocab_size"] * 4, "reused oracle logits shape differs")
executable = source.get("executable")
require(isinstance(executable, dict), "reused oracle manifest lacks executable provenance")
for name in ("path", "sha256"):
    require(name in executable, f"reused oracle manifest lacks executable field: {name}")
require_hash(executable["sha256"], "reused oracle executable hash is invalid")
libraries = executable.get("linked_libraries")
require(isinstance(libraries, list), "reused oracle manifest lacks linked library provenance")
source["executable"]["path"] = os.path.basename(source["executable"]["path"])
source["model"]["path"] = os.path.basename(sys.argv[3])
source["input"]["tokens"]["path"] = "tokens.u32le"
source["logits"]["path"] = "oracle.f32"
for library in libraries:
    require(isinstance(library, dict), "reused oracle linked library provenance is incomplete")
    library_name = library.get("path", library.get("name"))
    require(isinstance(library_name, str) and os.path.basename(library_name) == library_name, "reused oracle linked library name is unstable")
    require_hash(library.get("sha256"), "reused oracle linked library hash is invalid")
    if "path" in library:
        library["path"] = os.path.basename(library["path"])
Path(sys.argv[2]).write_text(json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n")
PY
else
  scripts/run-llama-oracle.sh \
    "$oracle_model" "$stage/tokens.u32le" "$stage/oracle.f32" "$stage/oracle.json" \
    "$window" cpu >/dev/null
fi
corpus_sha256=$(file_sha256 "$corpus")
corpus_name=$(stable_name "$corpus")
jq --arg corpus_name "$corpus_name" --arg corpus_sha256 "$corpus_sha256" \
  '.corpus = {name: $corpus_name, sha256: $corpus_sha256}' \
  "$stage/oracle.json" >"$stage/oracle.json.tmp"
mv "$stage/oracle.json.tmp" "$stage/oracle.json"
run_manifest=$(cat "$stage/oracle.json")
oracle_storage=$(jq -er '.model.storage_type' "$stage/oracle.json")
if [[ $oracle_storage != BF16 && $oracle_storage != F16 ]]; then
  echo "error: oracle model storage type is $oracle_storage, expected BF16 or F16" >&2
  exit 1
fi
if [[ $(jq -er '.execution.device' "$stage/oracle.json") != cpu ]]; then
  echo "error: export requires a CPU full-precision oracle" >&2
  exit 1
fi
if [[ $(jq -er '.input.window_tokens' "$stage/oracle.json") != "$window" ]]; then
  echo "error: exported oracle window differs from the requested window" >&2
  exit 1
fi
if [[ $(jq -er '.model.sha256' "$stage/oracle.json") != "$(file_sha256 "$oracle_model")" ]]; then
  echo "error: exported oracle manifest model differs from the requested model" >&2
  exit 1
fi
if [[ $(jq -er '.input.tokens.sha256' "$stage/oracle.json") != "$(file_sha256 "$stage/tokens.u32le")" ]]; then
  echo "error: exported oracle manifest tokens differ from the requested token stream" >&2
  exit 1
fi

jq -n \
  --arg source_commit "$(jq -er '.source_commit' "$stage/oracle.json")" \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg pin "$(cat external/PINNED)" \
  --arg adapter_sha256 "$(jq -er '.adapter.sha256' "$stage/oracle.json")" \
  --arg oracle_model "$(stable_name "$oracle_model")" \
  --argjson adapter "$(jq -c '.adapter' "$stage/oracle.json")" \
  --arg oracle_model_sha256 "$(file_sha256 "$oracle_model")" \
  --arg oracle_storage "$oracle_storage" \
  --argjson oracle_bytes "$(file_size "$oracle_model")" \
  --argjson oracle_vocab_size "$(jq -er '.model.vocab_size' "$stage/oracle.json")" \
  --arg oracle_architecture "$(jq -er '.model.architecture' "$stage/oracle.json")" \
  --arg oracle_tokenizer "$(jq -er '.model.tokenizer' "$stage/oracle.json")" \
  --arg subject_model "$(stable_name "$subject_model")" \
  --arg subject_storage_type "$subject_storage_type" \
  --arg subject_model_sha256 "$(file_sha256 "$subject_model")" \
  --argjson subject_bytes "$(file_size "$subject_model")" \
  --arg corpus "$(stable_name "$corpus")" \
  --arg corpus_sha256 "$corpus_sha256" \
  --argjson token_count "$token_count" \
  --argjson window "$window" \
  --argjson rows "$(jq -er '.rows // .input.rows' "$stage/oracle.json")" \
  --argjson vocab_size "$(jq -er '.vocab // .model.vocab_size' "$stage/oracle.json")" \
  --arg token_sha256 "$(file_sha256 "$stage/tokens.u32le")" \
  --argjson token_bytes "$(file_size "$stage/tokens.u32le")" \
  --arg oracle_manifest_sha256 "$(file_sha256 "$stage/oracle.json")" \
  --argjson oracle_manifest "$run_manifest" \
  --arg oracle_logits_sha256 "$(file_sha256 "$stage/oracle.f32")" \
  --argjson oracle_logits_bytes "$(file_size "$stage/oracle.f32")" \
  '{
    schema_version: "leone.quality-stage.v1",
    stage: "oracle-export",
    created_utc: $created_utc,
    source_commit: $source_commit,
    engine: {name: "llama.cpp", git_commit: $pin},
    adapter: $adapter,
    model: {
      oracle: {name: $oracle_model, sha256: $oracle_model_sha256, storage_type: $oracle_storage, bytes: $oracle_bytes, vocab_size: $oracle_vocab_size, architecture: $oracle_architecture, tokenizer: $oracle_tokenizer},
      subject: {name: $subject_model, sha256: $subject_model_sha256, storage_type: $subject_storage_type, bytes: $subject_bytes}
    },
    corpus: {name: $corpus, sha256: $corpus_sha256},
    input: {
      tokens: {path: "tokens.u32le", sha256: $token_sha256, bytes: $token_bytes, count: $token_count, encoding: "u32le"},
      window_tokens: $window,
      stride_tokens: ($window - 1),
      rows: $rows,
      vocab_size: $vocab_size
    },
    oracle: {
      manifest: {path: "oracle.json", sha256: $oracle_manifest_sha256, body: $oracle_manifest},
      logits: {path: "oracle.f32", sha256: $oracle_logits_sha256, bytes: $oracle_logits_bytes, encoding: "row-major-f32-le"}
    },
    limits: [
      "The CPU full-precision oracle is the common reference for the staged comparison.",
      "The staged comparison uses numerical quality metrics. It makes no cross-device bitwise claim.",
      "The stage does not certify concurrent service numerics."
    ]
  }' >"$stage/oracle-stage.json"
scripts/validate-quality-stage.py export "$stage"
remove_stage=0
trap - EXIT HUP INT TERM
echo "$stage/oracle-stage.json"
