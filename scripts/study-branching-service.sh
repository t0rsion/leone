#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

usage() {
  cat >&2 <<'USAGE'
usage: study-branching-service.sh STAGE [ARGUMENT...]

  build-leone BACKEND                       build target/release/leone (cuda or metal)
  build-llama BACKEND                       build the pinned llama-server (cuda or metal)
  source-manifest REVISION OUTPUT           record source inputs at REVISION
  check-quality RECORD VERIFIER [FLAG...]   canonical recomputation of the comparison record
  calibrate MANIFEST OUTPUT                 run the calibration study
  freeze TEMPLATE RECEIPT RECORD OUTPUT VERIFIER [FLAG...]
                                            check-quality, then write the frozen manifest
  frozen MANIFEST OUTPUT                    run the frozen study
  validate RECEIPT [repository|archive]     check a study receipt offline

Environment: LEONE_BRANCHING_SOURCE_MANIFEST binds a source manifest to a run
or a validation. LEONE_BRANCHING_SESSIONS, LEONE_BRANCHING_MAX_CONNECTIONS, and
LEONE_BRANCHING_CLIENT_CONNECTIONS set the Leone server limits. A run uses
LEONE_BRANCHING_SIGNING_KEY when set, or a private key under its temporary run
directory. The wrapper passes the key path to Leone and derives the trusted
public key for history tokenization.
USAGE
  exit 2
}

fail() {
  echo "error: $*" >&2
  exit 2
}

derive_signing_public_key() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

raw = Path(sys.argv[1]).read_bytes()
if len(raw) != 32:
    raise SystemExit("signing key must contain exactly 32 bytes")
public = Ed25519PrivateKey.from_private_bytes(raw).public_key()
print(public.public_bytes(Encoding.Raw, PublicFormat.Raw).hex())
PY
}

require_backend() {
  [[ $1 == cuda || $1 == metal ]] || fail "BACKEND must be cuda or metal: $1"
}

manifest_field() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
leone = next(engine for engine in manifest["engines"] if engine["kind"] == "leone")
values = {
    "backend": leone["backend"],
    "bind": leone["base_url"].split("://", 1)[1],
    "model": leone["model_artifact"],
    "plan": manifest["artifacts"].get("plan", {}).get("path", ""),
    "phase": manifest["phase"],
}
print(values[sys.argv[2]])
PY
}

# The study binds a comparison record by digest. It does not recompute KLD,
# top-1 agreement, or the needle argmax. This step does, with a verifier
# the caller names. It never reads an executable path from a receipt.
check_quality() {
  local record=$1 verifier=$2
  shift 2
  python3 scripts/validate-quality-stage.py comparison "$record" "$verifier" "$@"
  echo "canonical quality check passed: $record $(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$record")"
}

build_leone() {
  require_backend "$1"
  if [[ $1 == metal ]]; then
    cargo +1.92 build --release -p leone-cli --no-default-features --features metal
  else
    cargo +1.92 build --release -p leone-cli --features cuda
  fi
}

# fetch-llama-cpp.sh builds for CUDA only. A Metal build needs the pinned
# checkout to exist and to match external/PINNED.
build_llama() {
  require_backend "$1"
  if [[ $1 == cuda ]]; then
    scripts/fetch-llama-cpp.sh
    return
  fi
  local pin head
  pin=$(tr -d '[:space:]' <external/PINNED)
  head=$(git -C external/llama.cpp rev-parse HEAD)
  [[ $head == "$pin" ]] || fail "external/llama.cpp is at $head, external/PINNED is $pin"
  cmake -S external/llama.cpp -B external/llama.cpp/build -DGGML_METAL=ON -DGGML_CCACHE=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
  cmake --build external/llama.cpp/build --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}" --target llama-server
}

# One client IP opens five probe connections. The Leone default of four per
# client rejects the fifth. The harness does not observe these limits.
serve_leone() {
  local manifest=$1 work=$2 signing_key=$3 backend bind plan model
  backend=$(manifest_field "$manifest" backend)
  bind=$(manifest_field "$manifest" bind)
  model=$(manifest_field "$manifest" model)
  plan=$(manifest_field "$manifest" plan)
  local arguments=(serve -m "$model" --backend "$backend" --bind "$bind"
    --sessions "${LEONE_BRANCHING_SESSIONS:-8}"
    --max-connections "${LEONE_BRANCHING_MAX_CONNECTIONS:-16}"
    --max-connections-per-client "${LEONE_BRANCHING_CLIENT_CONNECTIONS:-8}"
    --receipt-dir "$work/receipts" --signing-key "$signing_key")
  if [[ $backend == metal ]]; then
    arguments+=(--batch-size 1)
  else
    arguments+=(--plan "$plan")
  fi
  if (exec 3<>"/dev/tcp/${bind%:*}/${bind#*:}") 2>/dev/null; then
    fail "something already listens on $bind"
  fi
  target/release/leone "${arguments[@]}" >"$work/leone-server.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 600); do
    kill -0 "$server_pid" 2>/dev/null || { cat "$work/leone-server.log" >&2; fail "the Leone server exited before it listened"; }
    if (exec 3<>"/dev/tcp/${bind%:*}/${bind#*:}") 2>/dev/null; then
      return
    fi
    sleep 0.1
  done
  fail "the Leone server did not listen on $bind"
}

source=()
source_arguments() {
  if [[ -n ${LEONE_BRANCHING_SOURCE_MANIFEST:-} ]]; then
    source=(--source-manifest "$LEONE_BRANCHING_SOURCE_MANIFEST")
  fi
}

preflight_tokenizer_inputs() {
  python3 - "$root" "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2])

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    print(f"error: cannot read branching manifest: {error}", file=sys.stderr)
    raise SystemExit(2)

engines = manifest.get("engines")
if not isinstance(engines, list):
    print("error: branching manifest engines must be a list", file=sys.stderr)
    raise SystemExit(2)

for engine in engines:
    engine_id = engine.get("id") if isinstance(engine, dict) else None
    declaration = engine.get("history_tokenization") if isinstance(engine, dict) else None
    if not isinstance(declaration, dict):
        print(f"error: {engine_id}: history tokenizer declaration is missing", file=sys.stderr)
        raise SystemExit(2)
    for field in ("producer_executable_path", "producer_model_artifact", "producer_template_file"):
        value = declaration.get(field)
        if not isinstance(value, str) or not value:
            print(f"error: {engine_id}: history tokenizer {field} is missing", file=sys.stderr)
            raise SystemExit(2)
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            print(f"error: {engine_id}: history tokenizer {field} escapes the study root", file=sys.stderr)
            raise SystemExit(2)
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            print(f"error: {engine_id}: history tokenizer {field} escapes the study root", file=sys.stderr)
            raise SystemExit(2)
        if not path.is_file():
            print(f"error: {engine_id}: history tokenizer input is missing: {field}", file=sys.stderr)
            raise SystemExit(2)
PY
}

run_study() {
  local manifest=$1 output=$2 phase=$3 work signing_key public_key
  [[ -n $output && ! -e $output && ! -L $output ]] || fail "output must be a new file path: $output"
  [[ $(manifest_field "$manifest" phase) == "$phase" ]] || fail "$manifest is not a $phase manifest"
  source_arguments
  preflight_tokenizer_inputs "$manifest"
  work=$(mktemp -d)
  signing_key=${LEONE_BRANCHING_SIGNING_KEY:-$work/response.key}
  server_pid=
  trap 'if [[ -n $server_pid ]]; then kill "$server_pid" 2>/dev/null || true; fi; rm -rf -- "$work"' EXIT
  serve_leone "$manifest" "$work" "$signing_key"
  export LEONE_BRANCHING_SIGNING_KEY="$signing_key"
  public_key=$(derive_signing_public_key "$signing_key")
  export LEONE_BRANCHING_TRUSTED_PUBLIC_KEY="$public_key"
  python3 scripts/study-branching-service.py --root "$root" --manifest "$manifest" \
    --mode "$phase" --run ${source[@]+"${source[@]}"} --output "$output"
}

stage=${1:-}
[[ -n $stage ]] || usage
shift
case $stage in
  build-leone) [[ $# -eq 1 ]] || usage; build_leone "$1" ;;
  build-llama) [[ $# -eq 1 ]] || usage; build_llama "$1" ;;
  source-manifest)
    [[ $# -eq 2 ]] || usage
    python3 scripts/source_inputs.py record-v04 "$1" "$2"
    ;;
  check-quality)
    [[ $# -ge 2 ]] || usage
    check_quality "$@"
    ;;
  calibrate) [[ $# -eq 2 ]] || usage; run_study "$1" "$2" calibration ;;
  frozen) [[ $# -eq 2 ]] || usage; run_study "$1" "$2" frozen ;;
  freeze)
    [[ $# -ge 5 ]] || usage
    template=$1 receipt=$2 record=$3 output=$4 verifier=$5
    shift 5
    check_quality "$record" "$verifier" "$@"
    python3 scripts/freeze-branching-manifest.py --root "$root" --template "$template" \
      --calibration-receipt "$receipt" --quality-record "$record" --output "$output"
    ;;
  validate)
    [[ $# -ge 1 && $# -le 2 ]] || usage
    scope=${2:-repository}
    if [[ $scope == archive && -z ${LEONE_BRANCHING_SOURCE_MANIFEST:-} ]]; then
      fail "archive validation requires LEONE_BRANCHING_SOURCE_MANIFEST"
    fi
    source_arguments
    python3 scripts/study-branching-service.py --root "$root" --validate-receipt "$1" \
      --source-scope "$scope" ${source[@]+"${source[@]}"}
    ;;
  *) usage ;;
esac
