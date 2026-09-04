#!/usr/bin/env bash
set -euo pipefail

llama_dir=${LLAMA_CPP_DIR:-external/llama.cpp}
output=${1:-target/llama-logits-oracle}
library_dir="$llama_dir/build/bin"

mkdir -p "$(dirname "$output")"
${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$llama_dir/include" \
  -I"$llama_dir/ggml/include" \
  research/oracle/llama_logits.cpp \
  -L"$library_dir" -Wl,-rpath,"$(realpath "$library_dir")" \
  -lllama -o "$output"

echo "$output"
