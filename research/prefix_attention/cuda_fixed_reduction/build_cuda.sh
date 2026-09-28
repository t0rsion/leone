#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest_script="$script_dir/../generate_manifest.py"
nvcc_bin=${NVCC:-nvcc}
cuda_arch=${CUDA_ARCH:-sm_89}
output_path=${1:-/tmp/prefix_attention_cuda}

python3 "$manifest_script" --cpp-output "$script_dir/manifest_cases.h" --check

"$nvcc_bin" -std=c++17 -O3 -arch="$cuda_arch" \
  "$script_dir/operator.cu" "$script_dir/main.cu" \
  -o "$output_path"

strip --strip-all "$output_path"

printf '%s\n' "$output_path"
