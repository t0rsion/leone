#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:?usage: scripts/prefill-characterization.sh <receipt-path>}

if [[ -e "$output" ]]; then
    echo "characterization receipt already exists: $output" >&2
    exit 1
fi

mkdir -p "$(dirname "$output")"
{
    printf 'engine commit: '
    git -C "$root" rev-parse HEAD
    printf 'model SHA-256: '
    sha256sum "$root/models/Qwen3-8B-Q4_K_M.gguf" | cut -d' ' -f1
    printf 'test: chunked_prefill_characterizes_kv_and_greedy_tokens\n'
    printf 'KV tolerance: max absolute error per key or value layer <= 0.5\n'
    printf 'relative error denominator: max(abs(sequential), 1e-6)\n'
    taskset -c 16-31 cargo +1.92 test --release \
        --manifest-path "$root/Cargo.toml" \
        -p leone-cuda --test gpu \
        chunked_prefill_characterizes_kv_and_greedy_tokens \
        -- --ignored --nocapture
} 2>&1 | tee "$output"
