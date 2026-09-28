#!/usr/bin/env bash
set -euo pipefail

required_version=0.65.1
maximum=10
swiftlint_bin=${LEONE_SWIFTLINT_BIN:-}

if [[ -z "$swiftlint_bin" ]]; then
  swiftlint_bin=$(command -v swiftlint || true)
elif [[ "$swiftlint_bin" != */* ]]; then
  swiftlint_bin=$(command -v "$swiftlint_bin" || true)
fi

if (( $# == 0 )); then
  echo "usage: scripts/check-swift-complexity.sh FILE.swift ..." >&2
  exit 2
fi
if [[ -z "$swiftlint_bin" || ! -x "$swiftlint_bin" ]]; then
  echo "Swift files require SwiftLint $required_version; install the pinned tool" >&2
  exit 2
fi
actual_version=$("$swiftlint_bin" version 2>/dev/null | head -n 1 || true)
if [[ "$actual_version" != "$required_version" ]]; then
  echo "SwiftLint $required_version is required; found ${actual_version:-unknown}" >&2
  exit 2
fi

syntax_mode=validated
if ! command -v swiftc >/dev/null 2>&1; then
  allow_unvalidated=${LEONE_ALLOW_UNVALIDATED_SWIFT_SYNTAX:-0}
  require_syntax=${LEONE_REQUIRE_SWIFT_SYNTAX:-0}
  platform=$(uname -s)
  if [[ "$allow_unvalidated" != 1 || "$require_syntax" == 1 || "$platform" == Darwin ]]; then
    echo "Swift files require swiftc for syntax validation" >&2
    exit 2
  fi
  syntax_mode=unvalidated
fi

configuration=$(mktemp)
trap 'rm -f "$configuration"' EXIT HUP INT TERM
cat >"$configuration" <<EOF
only_rules:
  - cyclomatic_complexity
cyclomatic_complexity:
  warning: $maximum
  error: $maximum
  ignores_case_statements: false
EOF

status=0
for file in "$@"; do
  if [[ "$syntax_mode" == validated ]] && ! swiftc -parse "$file"; then
    status=1
    continue
  fi
  if ! "$swiftlint_bin" lint --quiet --no-cache --config "$configuration" \
    --reporter xcode "$file"; then
    status=1
  fi
done

if (( status != 0 )); then
  exit "$status"
fi
if [[ "$syntax_mode" == validated ]]; then
  printf 'swift complexity passed: files=%d tool=SwiftLint version=%s maximum=%d cases=counted swift_syntax=validated\n' \
    "$#" "$required_version" "$maximum"
else
  printf 'swift complexity passed: files=%d tool=SwiftLint version=%s maximum=%d cases=counted swift_syntax=unvalidated native_build_required=1\n' \
    "$#" "$required_version" "$maximum"
fi
