#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

maximum=10
baseline_root=${LEONE_COMPLEXITY_BASELINE:-}
metrics=$(mktemp)
functions=$(mktemp)
sorted_functions=$(mktemp)
tracked=$(mktemp)
metrics_files=()
trap 'rm -f "$metrics" "$functions" "$sorted_functions" "$tracked" "${metrics_files[@]}"' EXIT HUP INT TERM

if ! command -v rust-code-analysis-cli >/dev/null 2>&1; then
  echo "rust-code-analysis-cli 0.0.25 is required" >&2
  exit 2
fi
if [[ "$(rust-code-analysis-cli --version 2>/dev/null)" != "rust-code-analysis-cli 0.0.25" ]]; then
  echo "rust-code-analysis-cli 0.0.25 is required" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 2
fi
if ! git ls-files --cached --others --exclude-standard -z >"$tracked"; then
  echo "git source inventory failed" >&2
  exit 2
fi
if [[ -n "$baseline_root" && ! -d "$baseline_root" ]]; then
  echo "complexity baseline does not exist: $baseline_root" >&2
  exit 2
fi

rust_files=()
python_files=()
c_files=()
cpp_files=()
cuda_files=()
shell_files=()
source_files=()
listed_count=0
excluded_external_count=0
while IFS= read -r -d '' file; do
  [[ -f "$file" ]] || continue
  listed_count=$((listed_count + 1))
  case "$file" in
    external/llama.cpp/*)
      excluded_external_count=$((excluded_external_count + 1))
      continue
      ;;
    *.rs) rust_files+=("$file"); source_files+=("$file") ;;
    *.py) python_files+=("$file"); source_files+=("$file") ;;
    *.c|*.h) c_files+=("$file"); source_files+=("$file") ;;
    *.cc|*.cpp|*.cxx|*.hh|*.hpp|*.hxx|*.inc|*.m|*.mm)
      cpp_files+=("$file")
      source_files+=("$file")
      ;;
    *.cu|*.cuh) cuda_files+=("$file"); source_files+=("$file") ;;
    *.sh) shell_files+=("$file"); ;;
  esac
done <"$tracked"

analyze_group() {
  local category=$1
  local language=$2
  shift 2
  local raw expected actual args
  if (( $# == 0 )); then
    return 0
  fi

  raw=$(mktemp)
  expected=$(mktemp)
  actual=$(mktemp)
  metrics_files+=("$raw" "$expected" "$actual")
  args=(--language-type "$language" -m -O json --pr)
  for file in "$@"; do
    args+=(-p "$file")
  done
  if ! rust-code-analysis-cli "${args[@]}" >"$raw"; then
    echo "rust-code-analysis-cli failed for $category sources" >&2
    return 1
  fi
  if ! jq -e -s \
    'all(.[]; (.name | type) == "string" and (.metrics.loc.sloc | type) == "number")' \
    "$raw" >/dev/null; then
    echo "rust-code-analysis-cli returned incomplete metrics for $category sources" >&2
    return 1
  fi
  if ! jq -e -s \
    'all(.[] | .. | objects | select(.kind? == "function");
      (.metrics.cyclomatic.sum | type) == "number")' \
    "$raw" >/dev/null; then
    echo "rust-code-analysis-cli returned incomplete function metrics for $category sources" >&2
    return 1
  fi

  printf '%s\n' "$@" | LC_ALL=C sort -u >"$expected"
  jq -r '.name' "$raw" | LC_ALL=C sort -u >"$actual"
  if ! cmp -s "$expected" "$actual"; then
    echo "rust-code-analysis-cli did not parse every tracked $category source" >&2
    comm -23 "$expected" "$actual" | sed 's/^/  missing: /' >&2
    comm -13 "$expected" "$actual" | sed 's/^/  unexpected: /' >&2
    return 1
  fi

  jq -r --arg category "$category" '
    [$category, .name, .metrics.loc.sloc,
      ([.. | objects | select(.kind? == "function")] | length)] | @tsv
  ' "$raw" >>"$metrics"
  jq -r --arg category "$category" '
    .name as $file
    | ..
    | objects
    | select(.kind? == "function")
    | [$file, (.start_line // 0), (.name // "<anonymous>"),
      .metrics.cyclomatic.sum, $category]
    | @tsv
  ' "$raw" >>"$functions"
}

analyze_group rust rust "${rust_files[@]}"
analyze_group python python "${python_files[@]}"
analyze_group c cpp "${c_files[@]}"
analyze_group cpp cpp "${cpp_files[@]}"
analyze_group cuda cpp "${cuda_files[@]}"

LC_ALL=C sort -t $'\t' -k4,4nr -k1,1 -k2,2n "$functions" >"$sorted_functions"

for category in rust python c cpp cuda; do
  awk -F '\t' -v category="$category" '
    $1 == category {
      files++
      sloc += $3
      functions += $4
    }
    END {
      printf "source category=%s files=%d functions=%d sloc=%.0f\n",
        category, files + 0, functions + 0, sloc + 0
    }
  ' "$metrics"
done

awk -F '\t' '
  {
    total++
    if ($4 <= 5) low++
    else if ($4 <= 10) watch++
    else if ($4 <= 15) refactor++
    else split_count++
  }
  END {
    printf "functions=%d low_1_5=%d watch_6_10=%d refactor_11_15=%d split_16_plus=%d\n",
      total, low, watch, refactor, split_count
  }
' "$sorted_functions"

echo "shell coverage=manual listed_files=${#shell_files[@]} parser=unavailable"
echo "coverage=parsed rust,python,c,cpp,cuda; external/llama.cpp excluded; shell manual"
other_count=$((listed_count - ${#source_files[@]} - ${#shell_files[@]} - excluded_external_count))
echo "coverage counts: listed=$listed_count parsed=${#source_files[@]} shell=${#shell_files[@]} other=$other_count excluded_external=$excluded_external_count"

if [[ -n "$baseline_root" ]]; then
  baseline_source_files=()
  while IFS= read -r -d '' file; do
    relative=${file#"$baseline_root"/}
    case "$relative" in
      external/llama.cpp/*) continue ;;
      *.rs|*.py|*.c|*.h|*.cc|*.cpp|*.cxx|*.cu|*.cuh|*.hh|*.hpp|*.hxx|*.inc|*.m|*.mm|*.sh)
        baseline_source_files+=("$file")
        ;;
    esac
  done < <(find "$baseline_root" -type f -print0)
  line_files=("${source_files[@]}" "${shell_files[@]}")
  current_lines=0
  baseline_lines=0
  baseline_files=0
  for file in "${line_files[@]}"; do
    current_lines=$((current_lines + $(wc -l <"$file")))
  done
  for file in "${baseline_source_files[@]}"; do
    baseline_lines=$((baseline_lines + $(wc -l <"$file")))
    baseline_files=$((baseline_files + 1))
  done
  echo "source lines: current=$current_lines baseline=$baseline_lines delta=$((current_lines - baseline_lines)) current_files=${#line_files[@]} baseline_files=$baseline_files"
fi

violations=$(awk -F '\t' -v maximum="$maximum" '$4 > maximum { count++ } END { print count + 0 }' "$functions")
if [[ $violations -ne 0 ]]; then
  echo "cyclomatic complexity exceeds $maximum in $violations functions:" >&2
  awk -F '\t' -v maximum="$maximum" \
    '$4 > maximum { printf "%5.0f  %s:%s  %s [%s]\n", $4, $1, $2, $3, $5 }' \
    "$sorted_functions" >&2
  exit 1
fi

echo "cyclomatic complexity gate passed (maximum=$maximum)"
