#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest_script="$script_dir/../generate_manifest.py"
metallib_path=${1:-/tmp/prefix_attention.metallib}
runner_path=${2:-/tmp/prefix_attention_metal}
temporary=$(mktemp -d -t prefix_attention)
air_path="$temporary/prefix_attention.air"
manifest_swift="$temporary/manifest.swift"
main_swift="$temporary/main.swift"
trap 'rm -rf "$temporary"' EXIT

python3 "$manifest_script" --cpp-output "$script_dir/../cuda_fixed_reduction/manifest_cases.h" \
  --check --swift-output "$manifest_swift"

xcrun -sdk macosx metal -std=metal3.0 -c \
  "$script_dir/PrefixAttention.metal" -o "$air_path"
xcrun -sdk macosx metallib "$air_path" -o "$metallib_path"
cp "$script_dir/RunPrefixAttention.swift" "$main_swift"
swiftc -O "$manifest_swift" "$main_swift" \
  -framework Foundation -framework Metal -o "$runner_path"
printf '%s\n' "$metallib_path"
printf '%s\n' "$runner_path"
