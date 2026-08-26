#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output="$root/corpus/v01.txt"

mkdir -p "$(dirname "$output")"
sed -sn '1,200p' \
    "$root/research/00-synthesis.md" \
    "$root/research/05-kernels-and-systems.md" > "$output"

(
    cd "$root/corpus"
    sha256sum v01.txt > SHA256SUMS
    cat SHA256SUMS
)
