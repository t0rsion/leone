#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
work_root=$(mktemp -d "${TMPDIR:-/tmp}/leone-fetch-llama.XXXXXX")
trap 'rm -rf "$work_root"' EXIT HUP INT TERM

fixture_root="$work_root/repo"
tool_dir="$work_root/tools"
custom_dir="$fixture_root/custom/checkout"
build_dir="$custom_dir/build"
log_file="$work_root/tool.log"
pin=0123456789abcdef0123456789abcdef01234567

mkdir -p "$fixture_root/scripts" "$fixture_root/external/shim" "$custom_dir" "$tool_dir"
cp "$repo_root/scripts/check-quantized-differential.sh" "$fixture_root/scripts/"
cp "$repo_root/scripts/fetch-llama-cpp.sh" "$fixture_root/scripts/"
printf '%s\n' "$pin" > "$fixture_root/external/PINNED"
printf 'gitdir: %s\n' "$work_root/gitdir" > "$custom_dir/.git"

export LEONE_FIXTURE_LOG="$log_file"
export LEONE_FIXTURE_PIN="$pin"
export LEONE_FIXTURE_CUSTOM="$custom_dir"
export LEONE_FIXTURE_BUILD="$build_dir"

cat > "$tool_dir/git" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail

printf 'git %s\n' "$*" >> "$LEONE_FIXTURE_LOG"
if [[ ${1:-} == clone ]]; then
    echo "unexpected clone" >&2
    exit 91
fi
if [[ ${1:-} != -C ]]; then
    echo "unexpected git invocation" >&2
    exit 92
fi
shift 2
case ${1:-} in
    rev-parse)
        case ${2:-} in
            --git-dir) printf '.git\n' ;;
            --show-toplevel) printf '%s\n' "$LEONE_FIXTURE_CUSTOM" ;;
            HEAD) printf '%s\n' "$LEONE_FIXTURE_PIN" ;;
            *) echo "unexpected rev-parse request" >&2; exit 93 ;;
        esac
        ;;
    cat-file)
        ;;
    checkout)
        [[ ${2:-} == --detach && ${3:-} == "$LEONE_FIXTURE_PIN" ]] || exit 94
        ;;
    fetch)
        echo "unexpected fetch" >&2
        exit 95
        ;;
    *)
        echo "unexpected git command" >&2
        exit 96
        ;;
esac
STUB

cat > "$tool_dir/cmake" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail

printf 'cmake %s\n' "$*" >> "$LEONE_FIXTURE_LOG"
case ${1:-} in
    -S)
        [[ ${2:-} == "$LEONE_FIXTURE_CUSTOM" && ${3:-} == -B ]] || exit 101
        [[ ${4:-} == "$LEONE_FIXTURE_BUILD" ]] || exit 102
        mkdir -p "$LEONE_FIXTURE_BUILD/bin"
        : > "$LEONE_FIXTURE_BUILD/CMakeCache.txt"
        ;;
    --build)
        [[ ${2:-} == "$LEONE_FIXTURE_BUILD" ]] || exit 103
        ;;
    *)
        echo "unexpected cmake invocation" >&2
        exit 104
        ;;
esac
STUB

cat > "$tool_dir/make" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail

printf 'make %s\n' "$*" >> "$LEONE_FIXTURE_LOG"
[[ ${1:-} == -B && ${2:-} == -C && ${3:-} == external/shim ]] || exit 111
llama_dir=
build_dir=
for argument in "$@"; do
    case $argument in
        LLAMA_CPP_DIR=*) llama_dir=${argument#LLAMA_CPP_DIR=} ;;
        LLAMA_CPP_BUILD_DIR=*) build_dir=${argument#LLAMA_CPP_BUILD_DIR=} ;;
    esac
done
[[ $llama_dir == "$LEONE_FIXTURE_CUSTOM" ]] || exit 112
[[ $build_dir == "$LEONE_FIXTURE_BUILD" ]] || exit 113
STUB

cat > "$tool_dir/cargo" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail

printf 'cargo %s\n' "$*" >> "$LEONE_FIXTURE_LOG"
[[ $* == *"-p leone-gguf"* ]] || exit 121
STUB

chmod +x "$tool_dir"/*
default_before=$(find "$repo_root/external" -mindepth 1 -maxdepth 2 -print | sort)

(
    cd "$fixture_root"
    PATH="$tool_dir:$PATH" \
    CMAKE_BUILD_PARALLEL_LEVEL=3 \
    CARGO_BUILD_JOBS=2 \
    LLAMA_CPP_DIR=custom/checkout \
    LLAMA_CPP_BUILD_DIR=custom/checkout/build \
    bash scripts/check-quantized-differential.sh
)

default_after=$(find "$repo_root/external" -mindepth 1 -maxdepth 2 -print | sort)
[[ $default_before == "$default_after" ]] || {
    echo "default external tree changed" >&2
    exit 1
}
[[ ! -e "$fixture_root/external/llama.cpp" ]] || {
    echo "custom checkout touched the default fixture path" >&2
    exit 1
}
[[ $(cat "$fixture_root/external/PINNED") == "$pin" ]] || {
    echo "fixture PINNED changed" >&2
    exit 1
}
grep -F "git -C $custom_dir rev-parse --show-toplevel" "$log_file" >/dev/null
grep -F "cmake -S $custom_dir -B $build_dir" "$log_file" >/dev/null
rg -F "cmake --build $build_dir --parallel 3 --target ggml-base" "$log_file" >/dev/null
rg -F 'cargo +1.92 test --jobs 2 -p leone-gguf' "$log_file" >/dev/null
grep -F "make -B -C external/shim LLAMA_CPP_DIR=$custom_dir LLAMA_CPP_BUILD_DIR=$build_dir" "$log_file" >/dev/null

unsafe_root="$work_root/unversioned-repo"
unsafe_child="$unsafe_root/unversioned/child"
unsafe_build="$unsafe_child/build"
unsafe_tools="$work_root/unsafe-tools"
unsafe_log="$work_root/unsafe-tool.log"
mkdir -p "$unsafe_root/scripts" "$unsafe_root/external" "$unsafe_child" "$unsafe_tools"
cp "$repo_root/scripts/fetch-llama-cpp.sh" "$unsafe_root/scripts/"
printf 'fixture\n' > "$unsafe_root/marker"
git -C "$unsafe_root" init -q
git -C "$unsafe_root" config user.email fixture.invalid
git -C "$unsafe_root" config user.name fixture
git -C "$unsafe_root" add marker scripts/fetch-llama-cpp.sh
git -C "$unsafe_root" commit -qm fixture
unsafe_pin=$(git -C "$unsafe_root" rev-parse HEAD)
printf '%s\n' "$unsafe_pin" > "$unsafe_root/external/PINNED"
unsafe_branch_before=$(git -C "$unsafe_root" symbolic-ref --short HEAD)
cp "$tool_dir/cmake" "$unsafe_tools/cmake"
chmod +x "$unsafe_tools/cmake"
if (
    cd "$unsafe_root"
    PATH="$unsafe_tools:$PATH" \
    LEONE_FIXTURE_LOG="$unsafe_log" \
    LEONE_FIXTURE_CUSTOM="$unsafe_child" \
    LEONE_FIXTURE_BUILD="$unsafe_build" \
    LEONE_LLAMA_CPP_CPU_ONLY=1 \
    LLAMA_CPP_DIR=unversioned/child \
    LLAMA_CPP_BUILD_DIR=unversioned/child/build \
    bash scripts/fetch-llama-cpp.sh
); then
    echo "unversioned child was accepted as the parent repository" >&2
    exit 1
fi
unsafe_branch_after=$(git -C "$unsafe_root" symbolic-ref --short HEAD 2>/dev/null || printf '<detached>')
[[ $unsafe_branch_after == "$unsafe_branch_before" ]] || {
    echo "unversioned child changed the parent repository HEAD" >&2
    exit 1
}
[[ ! -e "$unsafe_child/.git" ]] || {
    echo "unversioned child was mutated" >&2
    exit 1
}
printf 'custom llama.cpp checkout fixture passed\n'
