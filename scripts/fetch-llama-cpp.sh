#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
external_dir="$repo_root/external"
llama_dir=${LLAMA_CPP_DIR:-$external_dir/llama.cpp}
if [[ $llama_dir != /* ]]; then
    llama_dir="$repo_root/$llama_dir"
fi
pin_file="$external_dir/PINNED"

mkdir -p "$external_dir"

if [[ -e "$llama_dir" ]]; then
    if [[ ! -f "$llama_dir/.git" && ! -d "$llama_dir/.git" ]]; then
        echo "error: llama.cpp checkout must contain .git: $llama_dir" >&2
        exit 1
    fi
else
    mkdir -p "$(dirname "$llama_dir")"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$llama_dir"
fi

canonical_llama_dir=$(CDPATH='' cd -- "$llama_dir" && pwd -P)
if ! git_root=$(git -C "$llama_dir" rev-parse --show-toplevel 2>/dev/null); then
    echo "error: llama.cpp checkout is not a worktree: $llama_dir" >&2
    exit 1
fi
canonical_git_root=$(CDPATH='' cd -- "$git_root" && pwd -P)
if [[ $canonical_git_root != "$canonical_llama_dir" ]]; then
    echo "error: llama.cpp checkout resolves outside LLAMA_CPP_DIR: $llama_dir" >&2
    exit 1
fi

# PINNED is the comparator pin. Later runs reuse it until the file is edited.
if [[ ! -f "$pin_file" ]]; then
    git -C "$llama_dir" rev-parse HEAD > "$pin_file"
fi

pin=$(tr -d '[:space:]' < "$pin_file")
if [[ ! "$pin" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: $pin_file must contain one 40-character commit hash" >&2
    exit 1
fi
if ! git -C "$llama_dir" cat-file -e "$pin^{commit}" 2>/dev/null; then
    git -C "$llama_dir" fetch --depth 1 origin "$pin"
fi
git -C "$llama_dir" checkout --detach "$pin"
checked_out_pin=$(git -C "$llama_dir" rev-parse HEAD)
if [[ $checked_out_pin != "$pin" ]]; then
    echo "error: llama.cpp HEAD $checked_out_pin differs from $pin_file $pin" >&2
    exit 1
fi
echo "llama.cpp pin: $pin"

build_dir=${LLAMA_CPP_BUILD_DIR:-$llama_dir/build}
if [[ $build_dir != /* ]]; then
    build_dir="$repo_root/$build_dir"
fi

run_build() {
    if [[ -n ${LEONE_BUILD_CPUSET:-} ]]; then
        taskset -c "$LEONE_BUILD_CPUSET" "$@"
    else
        "$@"
    fi
}

if [[ ${LEONE_LLAMA_CPP_CPU_ONLY:-0} == 1 ]]; then
    generator=()
    if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
        if command -v ninja >/dev/null 2>&1; then
            generator=(-G Ninja)
        else
            generator=(-G "Unix Makefiles")
        fi
    fi

    run_build cmake -S "$llama_dir" -B "$build_dir" \
        "${generator[@]}" \
        -DGGML_CUDA=OFF \
        -DGGML_CCACHE=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF

    run_build cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}" --target ggml-base
    exit 0
fi

cuda_compiler=/usr/bin/nvcc
if [[ ! -x "$cuda_compiler" && -x /opt/cuda/bin/nvcc ]]; then
    cuda_compiler=/opt/cuda/bin/nvcc
    echo "CUDA compiler fallback: $cuda_compiler"
fi
if [[ ! -x "$cuda_compiler" ]]; then
    echo "error: nvcc is not at /usr/bin/nvcc or /opt/cuda/bin/nvcc" >&2
    exit 1
fi

generator=()
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    if command -v ninja >/dev/null 2>&1; then
        generator=(-G Ninja)
    else
        generator=(-G "Unix Makefiles")
    fi
fi

run_build cmake -S "$llama_dir" -B "$build_dir" \
    "${generator[@]}" \
    -DGGML_CUDA=ON \
    -DGGML_CCACHE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF

run_build cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}" --target \
    llama-bench llama-cli llama-perplexity llama-server llama-tokenize
