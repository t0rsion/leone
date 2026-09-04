#!/usr/bin/env bash
set -euo pipefail

maximum=${LEONE_MAX_CYCLOMATIC:-10}
metrics=$(mktemp)
functions=$(mktemp)
trap 'rm -f "$metrics" "$functions"' EXIT HUP INT TERM

if ! command -v rust-code-analysis-cli >/dev/null 2>&1; then
  echo "rust-code-analysis-cli is required; install version 0.0.25" >&2
  exit 2
fi

rust-code-analysis-cli -p crates -m -O json --pr >"$metrics"

jq -r '
  .name as $file
  | ..
  | objects
  | select(.kind? == "function")
  | [$file, .start_line, .name, (.metrics.cyclomatic.sum // 0)]
  | @tsv
' "$metrics" | sort -t $'\t' -k4,4nr >"$functions"

awk -F '\t' '
  {
    total++
    if ($4 <= 5) low++
    else if ($4 <= 10) watch++
    else if ($4 <= 15) refactor++
    else split_count++
  }
  END {
    printf "functions=%d low_1_5=%d watch_6_10=%d refactor_11_15=%d split_16_plus=%d\n", total, low, watch, refactor, split_count
  }
' "$functions"

violations=$(awk -F '\t' -v maximum="$maximum" '$4 > maximum { count++ } END { print count + 0 }' "$functions")
if [[ $violations -ne 0 ]]; then
  echo "cyclomatic complexity exceeds $maximum in $violations functions:" >&2
  awk -F '\t' -v maximum="$maximum" '$4 > maximum { printf "%5.0f  %s:%s  %s\n", $4, $1, $2, $3 }' "$functions" >&2
  exit 1
fi

echo "cyclomatic complexity gate passed"
