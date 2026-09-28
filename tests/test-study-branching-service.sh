#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
work_root=$(mktemp -d "${TMPDIR:-/tmp}/leone-study-wrapper.XXXXXX")
trap 'rm -rf "$work_root"' EXIT HUP INT TERM

fixture_root="$work_root/repo"
tool_dir="$work_root/tools"
log_file="$work_root/python.log"
freeze_marker="$work_root/freeze.called"
mkdir -p "$fixture_root/scripts" "$tool_dir"
cp "$repo_root/scripts/study-branching-service.sh" "$fixture_root/scripts/"
mkdir -p "$fixture_root/target/release"

cat > "$fixture_root/target/release/leone" <<'STUB'
#!/usr/bin/env bash
touch "$LEONE_WRAPPER_SERVER_MARKER"
exit 0
STUB
chmod +x "$fixture_root/target/release/leone"

export LEONE_WRAPPER_LOG="$log_file"
export LEONE_WRAPPER_FREEZE_MARKER="$freeze_marker"
export LEONE_WRAPPER_SERVER_MARKER="$work_root/server.called"

cat > "$tool_dir/python3" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail

line=python3
for argument in "$@"; do
    line+="|$argument"
done
printf '%s\n' "$line" >> "$LEONE_WRAPPER_LOG"

case ${1:-} in
    scripts/validate-quality-stage.py)
        exit "${LEONE_WRAPPER_CHECK_STATUS:-0}"
        ;;
    scripts/freeze-branching-manifest.py)
        printf '%s\n' "$*" > "$LEONE_WRAPPER_FREEZE_MARKER"
        ;;
    scripts/study-branching-service.py)
        ;;
    -)
        if [[ ${3:-} == *.json ]]; then
            exec /usr/bin/python3 "$@"
        fi
        case ${3:-} in
            phase) printf 'calibration\n' ;;
            backend) printf 'cuda\n' ;;
            bind) printf '127.0.0.1:18800\n' ;;
            model) printf 'model.gguf\n' ;;
            plan) printf 'plan.json\n' ;;
            *) printf 'unexpected manifest field: %s\n' "${3:-}" >&2; exit 99 ;;
        esac
        ;;
    -c)
        printf '0%.0s' {1..64}
        printf '\n'
        ;;
    *)
        printf 'unexpected python invocation: %s\n' "$*" >&2
        exit 99
        ;;
esac
STUB
chmod +x "$tool_dir/python3"

run_wrapper() {
    (
        cd "$fixture_root"
        PATH="$tool_dir:$PATH" bash scripts/study-branching-service.sh "$@"
    )
}

if run_wrapper check-quality record.json 2>"$work_root/arity.err"; then
    printf '%s\n' 'check-quality accepted too few arguments' >&2
    exit 1
fi
grep -F 'usage: study-branching-service.sh' "$work_root/arity.err" >/dev/null
[[ ! -s "$log_file" ]]

run_wrapper check-quality record.json verifier --source-commit commit --backend cuda
grep -F 'python3|scripts/validate-quality-stage.py|comparison|record.json|verifier|--source-commit|commit|--backend|cuda' "$log_file" >/dev/null

rm -f "$freeze_marker"
: > "$log_file"
export LEONE_WRAPPER_CHECK_STATUS=31
if run_wrapper freeze template.json calibration.json record.json output.json verifier --backend cuda; then
    printf '%s\n' 'freeze continued after a failed quality check' >&2
    exit 1
fi
unset LEONE_WRAPPER_CHECK_STATUS
grep -F 'python3|scripts/validate-quality-stage.py|comparison|record.json|verifier|--backend|cuda' "$log_file" >/dev/null
if grep -F 'scripts/freeze-branching-manifest.py' "$log_file" >/dev/null; then
    printf '%s\n' 'freeze invoked its writer after a failed quality check' >&2
    exit 1
fi
[[ ! -e "$freeze_marker" ]]

: > "$log_file"
run_wrapper freeze template.json calibration.json record.json output.json verifier --backend cuda
grep -F 'python3|scripts/validate-quality-stage.py|comparison|record.json|verifier|--backend|cuda' "$log_file" >/dev/null
grep -F "python3|scripts/freeze-branching-manifest.py|--root|$fixture_root|--template|template.json|--calibration-receipt|calibration.json|--quality-record|record.json|--output|output.json" "$log_file" >/dev/null

: > "$log_file"
unset LEONE_BRANCHING_SOURCE_MANIFEST
if run_wrapper validate receipt.json archive 2>"$work_root/source.err"; then
    printf '%s\n' 'archive validation ran without a source manifest' >&2
    exit 1
fi
grep -F 'error: archive validation requires LEONE_BRANCHING_SOURCE_MANIFEST' "$work_root/source.err" >/dev/null
[[ ! -s "$log_file" ]]

if run_wrapper validate receipt.json archive extra 2>"$work_root/scope-arity.err"; then
    printf '%s\n' 'validate accepted too many arguments' >&2
    exit 1
fi
grep -F 'usage: study-branching-service.sh' "$work_root/scope-arity.err" >/dev/null

export LEONE_BRANCHING_SOURCE_MANIFEST=source-inputs.json
run_wrapper validate receipt.json archive
grep -F "python3|scripts/study-branching-service.py|--root|$fixture_root|--validate-receipt|receipt.json|--source-scope|archive|--source-manifest|source-inputs.json" "$log_file" >/dev/null

cat > "$fixture_root/calibration.json" <<'JSON'
{"phase":"calibration","engines":[{"id":"leone","history_tokenization":{"producer_executable_path":"missing/llama-server","producer_model_artifact":"missing/model.gguf","producer_template_file":"missing/template.jinja"}}]}
JSON
rm -f "$LEONE_WRAPPER_SERVER_MARKER"
if run_wrapper calibrate calibration.json output.json 2>"$work_root/preflight.err"; then
    printf '%s\n' 'calibrate accepted missing tokenizer inputs' >&2
    exit 1
fi
grep -F 'error: leone: history tokenizer input is missing' "$work_root/preflight.err" >/dev/null
[[ ! -e "$LEONE_WRAPPER_SERVER_MARKER" ]]

printf '%s\n' 'branching service wrapper tests passed'
