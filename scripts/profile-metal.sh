#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo_root"
export CARGO_TARGET_DIR="$repo_root/target"

exec cargo +1.92 run --locked --release -p leone-metal --example metal_profile -- "$@"
