#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
external_dir="$repo_root/external"
llama_dir="$external_dir/llama.cpp"
pin_file="$external_dir/PINNED"

mkdir -p "$external_dir"

if [[ ! -d "$llama_dir/.git" ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$llama_dir"
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
echo "llama.cpp pin: $pin"

cuda_compiler=/usr/bin/nvcc
if [[ ! -x "$cuda_compiler" && -x /opt/cuda/bin/nvcc ]]; then
    cuda_compiler=/opt/cuda/bin/nvcc
    echo "CUDA compiler fallback: $cuda_compiler"
fi
if [[ ! -x "$cuda_compiler" ]]; then
    echo "error: nvcc is not at /usr/bin/nvcc or /opt/cuda/bin/nvcc" >&2
    exit 1
fi

build_dir="$llama_dir/build"
generator=()
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    if command -v ninja >/dev/null 2>&1; then
        generator=(-G Ninja)
    else
        generator=(-G "Unix Makefiles")
    fi
fi

taskset -c 16-31 cmake -S "$llama_dir" -B "$build_dir" \
    "${generator[@]}" \
    -DGGML_CUDA=ON \
    -DGGML_CCACHE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF

taskset -c 16-31 cmake --build "$build_dir" --parallel 24 --target \
    llama-bench llama-cli llama-perplexity llama-server llama-tokenize
