#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
llama_dir=${LLAMA_CPP_DIR:-$root/external/llama.cpp}
output=${1:-$root/target/llama-logits-oracle}
build_cpuset=${LEONE_BUILD_CPUSET:-16-31}

if [[ $llama_dir != /* ]]; then
  llama_dir="$root/$llama_dir"
fi
if [[ $output != /* ]]; then
  output="$root/$output"
fi
library_dir="$llama_dir/build/bin"
if [[ ! -d "$llama_dir/.git" ]]; then
  echo "error: pinned llama.cpp checkout is missing: $llama_dir" >&2
  exit 1
fi
if [[ ! -f "$root/external/PINNED" ]]; then
  echo "error: external/PINNED is missing" >&2
  exit 1
fi
pin=$(tr -d '[:space:]' <"$root/external/PINNED")
if [[ ! $pin =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: external/PINNED must contain one 40-character commit hash" >&2
  exit 1
fi
llama_commit=$(git -C "$llama_dir" rev-parse HEAD)
if [[ $llama_commit != "$pin" ]]; then
  echo "error: llama.cpp HEAD $llama_commit differs from external/PINNED $pin" >&2
  exit 1
fi
if [[ -n $(git -C "$llama_dir" status --porcelain --untracked-files=all) ]]; then
  echo "error: pinned llama.cpp checkout has local changes" >&2
  exit 1
fi
for input in "$llama_dir/include/llama.h" "$llama_dir/ggml/include/ggml-backend.h" \
  "$library_dir/libllama.so" "$library_dir/libggml.so"; do
  if [[ ! -e $input ]]; then
    echo "error: llama.cpp build input is missing: $input" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$output")"
taskset -c "$build_cpuset" "${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$llama_dir/include" \
  -I"$llama_dir/ggml/include" \
  "$root/research/oracle/llama_logits.cpp" \
  -L"$library_dir" -Wl,-rpath,"$(realpath "$library_dir")" \
  -lllama -lggml -lggml-base -o "$output"

echo "$output"
