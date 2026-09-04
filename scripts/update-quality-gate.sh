#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readme="$root/benchmarks/README.md"
top_readme="$root/README.md"
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -1)
release_root="https://github.com/t0rsion/leone/blob/v$version"

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <leone-quality> <llama-quality> <leone-runtime> <llama-runtime>" >&2
    exit 2
fi

leone=$1
llama=$2
leone_runtime=$3
llama_runtime=$4
for receipt in "$leone" "$llama" "$leone_runtime" "$llama_runtime"; do
    if [[ ! -f "$receipt" ]]; then
        echo "error: receipt is missing: $receipt" >&2
        exit 1
    fi
done

leone_mean=$(jq -er '.metrics.kld.mean' "$leone")
leone_p99=$(jq -er '.metrics.kld.p99' "$leone")
leone_top1=$(jq -er '.metrics.top1_agreement' "$leone")
llama_mean=$(jq -er '.metrics.kld.mean' "$llama")
llama_p99=$(jq -er '.metrics.kld.p99' "$llama")
llama_top1=$(jq -er '.metrics.top1_agreement' "$llama")
limit=$(awk -v baseline="$llama_mean" 'BEGIN { printf "%.9f", baseline + 0.02 }')
verdict=$(awk -v candidate="$leone_mean" -v maximum="$limit" \
    'BEGIN { print candidate <= maximum ? "pass" : "fail" }')
leone_name=$(basename "$leone")
llama_name=$(basename "$llama")
leone_runtime_rate=$(jq -er '.results.decode_tok_s.median' "$leone_runtime")
llama_runtime_rate=$(jq -er '.results.decode_tok_s.median' "$llama_runtime")
runtime_ratio=$(awk -v candidate="$leone_runtime_rate" -v baseline="$llama_runtime_rate" \
    'BEGIN { printf "%.9f", candidate / baseline }')
runtime_verdict=$(awk -v ratio="$runtime_ratio" \
    'BEGIN { print ratio >= 1.0 ? "pass" : "fail" }')
leone_runtime_name=$(basename "$leone_runtime")
llama_runtime_name=$(basename "$llama_runtime")
overall_verdict=fail
if [[ "$verdict" == pass && "$runtime_verdict" == pass ]]; then
    overall_verdict=pass
fi

block=$(mktemp)
headline=$(mktemp)
output=$(mktemp)
trap 'rm -f -- "$block" "$headline" "$output"' EXIT

{
    echo '<!-- quality-results:start -->'
    echo '## Measured v0.1 gate'
    echo
    echo '### Performance'
    echo
    echo '| Engine | Median decode (tok/s) | Receipt |'
    echo '|---|---:|---|'
    printf '| Leone | %.3f | [`%s`](../receipts/%s) |\n' \
        "$leone_runtime_rate" "$leone_runtime_name" "$leone_runtime_name"
    printf '| llama.cpp | %.3f | [`%s`](../receipts/%s) |\n' \
        "$llama_runtime_rate" "$llama_runtime_name" "$llama_runtime_name"
    echo
    printf 'The performance gate is **%s**. The Leone to llama.cpp median ratio is %.9f. The required ratio is at least 1.00.\n' \
        "$runtime_verdict" "$runtime_ratio"
    echo
    echo '### Quality'
    echo
    echo '| Subject | Mean KLD (nats) | p99 KLD (nats) | Top-1 agreement | Receipt |'
    echo '|---|---:|---:|---:|---|'
    printf '| Leone Q4_K_M | %.9f | %.9f | %.9f | [`%s`](../receipts/%s) |\n' \
        "$leone_mean" "$leone_p99" "$leone_top1" "$leone_name" "$leone_name"
    printf '| llama.cpp Q4_K_M | %.9f | %.9f | %.9f | [`%s`](../receipts/%s) |\n' \
        "$llama_mean" "$llama_p99" "$llama_top1" "$llama_name" "$llama_name"
    echo
    printf 'The quality gate is **%s**. Leone mean KLD must not exceed %.9f nats (llama.cpp mean KLD plus 0.02 nats).\n' \
        "$verdict" "$limit"
    echo
    printf 'The overall v0.1 evidence gate is **%s**. The performance gate and the quality gate must both pass.\n' \
        "$overall_verdict"
    echo '<!-- quality-results:end -->'
} > "$block"

{
    echo '<!-- gate-headline:start -->'
    echo '| Measurement | Leone | llama.cpp | Verdict |'
    echo '|---|---:|---:|---|'
    printf '| Decode tok/s at depth 512 | %.3f | %.3f | %s, ratio %.9f |\n' \
        "$leone_runtime_rate" "$llama_runtime_rate" "$runtime_verdict" "$runtime_ratio"
    printf '| Mean KLD vs BF16 oracle (nats) | %.9f | %.9f | %s, limit %.9f |\n' \
        "$leone_mean" "$llama_mean" "$verdict" "$limit"
    echo
    printf 'The v0.1 evidence gate is **%s**. Runtime receipts:\n' "$overall_verdict"
    printf '[`%s`](%s/receipts/%s)\n' \
        "$leone_runtime_name" "$release_root" "$leone_runtime_name"
    echo 'and'
    printf '[`%s`](%s/receipts/%s).\n' \
        "$llama_runtime_name" "$release_root" "$llama_runtime_name"
    echo 'Quality receipts:'
    printf '[`%s`](%s/receipts/%s)\n' \
        "$leone_name" "$release_root" "$leone_name"
    echo 'and'
    printf '[`%s`](%s/receipts/%s).\n' \
        "$llama_name" "$release_root" "$llama_name"
    echo 'The'
    printf '[benchmark report](%s/benchmarks/README.md) states the workload and gate.\n' \
        "$release_root"
    echo '<!-- gate-headline:end -->'
} > "$headline"

render() {
    local target=$1
    local replacement=$2
    local start=$3
    local end=$4
    awk -v replacement="$replacement" -v start="$start" -v end="$end" '
        BEGIN { inside = 0; replaced = 0 }
        $0 == start {
            while ((getline line < replacement) > 0) print line
            close(replacement)
            inside = 1
            replaced = 1
            next
        }
        $0 == end { inside = 0; next }
        !inside { print }
        END {
            if (!replaced) {
                print ""
                while ((getline line < replacement) > 0) print line
                close(replacement)
            }
        }
    ' "$target" > "$output"
    mv "$output" "$target"
    output=$(mktemp)
}

render "$top_readme" "$headline" \
    '<!-- gate-headline:start -->' '<!-- gate-headline:end -->'

awk -v replacement="$block" '
    BEGIN { inside = 0; replaced = 0 }
    /^<!-- quality-results:start -->$/ {
        while ((getline line < replacement) > 0) print line
        close(replacement)
        inside = 1
        replaced = 1
        next
    }
    /^<!-- quality-results:end -->$/ { inside = 0; next }
    !inside { print }
    END {
        if (!replaced) {
            print ""
            while ((getline line < replacement) > 0) print line
            close(replacement)
        }
    }
' "$readme" > "$output"
mv "$output" "$readme"
