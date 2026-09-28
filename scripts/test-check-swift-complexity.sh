#!/usr/bin/env bash
set -euo pipefail

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
required_version=0.65.1
swiftlint_bin=${LEONE_SWIFTLINT_BIN:-}
if [[ -z "$swiftlint_bin" ]]; then
  swiftlint_bin=$(command -v swiftlint || true)
elif [[ "$swiftlint_bin" != */* ]]; then
  swiftlint_bin=$(command -v "$swiftlint_bin" || true)
fi
require_tool=0
if [[ "${LEONE_REQUIRE_SWIFTLINT:-0}" == 1 || "$(uname -s)" == Darwin ]]; then
  require_tool=1
fi
if [[ -z "$swiftlint_bin" || ! -x "$swiftlint_bin" ]] ||
  [[ "$("$swiftlint_bin" version 2>/dev/null | head -n 1 || true)" != "$required_version" ]]; then
  if (( require_tool == 1 )); then
    echo "Swift complexity fixtures require SwiftLint $required_version" >&2
    exit 2
  fi
  echo "Swift complexity fixtures skipped: SwiftLint $required_version is unavailable"
  exit 0
fi
syntax_validated=1
if ! command -v swiftc >/dev/null 2>&1; then
  if [[ "$(uname -s)" == Darwin || "${LEONE_REQUIRE_SWIFT_SYNTAX:-0}" == 1 ]]; then
    echo "Swift complexity fixtures require swiftc" >&2
    exit 2
  fi
  syntax_validated=0
  echo "Swift syntax fixtures deferred: native swiftc required" >&2
fi

run_checker() {
  if (( syntax_validated == 1 )); then
    "$repo/scripts/check-swift-complexity.sh" "$@"
  else
    LEONE_ALLOW_UNVALIDATED_SWIFT_SYNTAX=1 \
      "$repo/scripts/check-swift-complexity.sh" "$@"
  fi
}

temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT HUP INT TERM

cat >"$temporary/simple.swift" <<'EOF'
func simple(_ value: Int) -> Int {
  if value > 0 {
    return value
  }
  return 0
}
EOF
run_checker "$temporary/simple.swift" >/dev/null

{
  printf '%s\n' 'func boundary(_ value: Int) -> Int {'
  for branch in $(seq 1 10); do
    printf '  if value == %d { return %d }\n' "$branch" "$branch"
  done
  printf '%s\n' '  return 0' '}'
} >"$temporary/boundary.swift"
run_checker "$temporary/boundary.swift" >/dev/null

{
  printf '%s\n' 'func branchy(_ value: Int) -> Int {'
  for branch in $(seq 1 11); do
    printf '  if value == %d { return %d }\n' "$branch" "$branch"
  done
  printf '%s\n' '  return 0' '}'
} >"$temporary/branchy.swift"
if run_checker "$temporary/branchy.swift" >"$temporary/output" 2>&1; then
  cat "$temporary/output" >&2
  echo "expected the branch-heavy Swift fixture to fail" >&2
  exit 1
fi
grep -F 'cyclomatic_complexity' "$temporary/output" >/dev/null

{
  printf '%s\n' 'func switchCases(_ value: Int) -> Int {' '  switch value {'
  for branch in $(seq 1 10); do
    printf '  case %d: return %d\n' "$branch" "$branch"
  done
  printf '%s\n' '  default: return 0' '  }' '}'
} >"$temporary/switch-cases.swift"
if run_checker "$temporary/switch-cases.swift" >"$temporary/output" 2>&1; then
  cat "$temporary/output" >&2
  echo "expected switch cases to count toward complexity" >&2
  exit 1
fi
grep -F 'cyclomatic_complexity' "$temporary/output" >/dev/null

if run_checker "$temporary/missing.swift" >"$temporary/output" 2>&1; then
  cat "$temporary/output" >&2
  echo "expected a missing Swift file to fail" >&2
  exit 1
fi

printf '%s\n' 'func malformed( {' >"$temporary/malformed.swift"
if (( syntax_validated == 1 )); then
  if run_checker "$temporary/malformed.swift" >"$temporary/output" 2>&1; then
    cat "$temporary/output" >&2
    echo "expected a malformed Swift file to fail" >&2
    exit 1
  fi
else
  echo "Swift malformed-file fixture deferred: native swiftc required" >&2
fi

if (( syntax_validated == 1 )); then
  echo "Swift complexity fixtures passed"
else
  echo "Swift complexity fixtures passed; syntax fixtures deferred"
fi
