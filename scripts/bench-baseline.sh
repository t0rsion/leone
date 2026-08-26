#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench="$repo_root/external/llama.cpp/build/bin/llama-bench"
model="$repo_root/models/Qwen3-8B-Q4_K_M.gguf"
pin_file="$repo_root/external/PINNED"

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [quality-receipt-uuid]" >&2
    exit 2
fi

quality_args=()
if [[ $# -eq 1 ]]; then
    quality_args=(--quality-ref "$1")
fi

for required in "$bench" "$model" "$pin_file"; do
    if [[ ! -e "$required" ]]; then
        echo "error: required file is missing: $required" >&2
        exit 1
    fi
done

# A second compute process invalidates the preregistered timing condition.
active_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$active_pids" ]]; then
    echo "error: the GPU has an active compute process: $active_pids" >&2
    exit 1
fi

mkdir -p "$repo_root/receipts/raw"
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
raw="$repo_root/receipts/raw/$timestamp-llama-bench.json"

cd "$repo_root"
taskset -c 16-31 cargo +1.92 build -p leone-cli
env GGML_CUDA_GRAPH_OPT=1 taskset -c 0-3,12-15 "$bench" \
    -m models/Qwen3-8B-Q4_K_M.gguf \
    -p 0 \
    -n 128 \
    -d 512 \
    -r 5 \
    -o json \
    -ngl 99 > "$raw"

target/debug/leone receipt from-llama-bench "$raw" "${quality_args[@]}"
