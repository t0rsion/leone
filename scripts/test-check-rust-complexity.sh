#!/usr/bin/env bash
set -euo pipefail

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
fixture="$tmp/fixture"
mkdir -p "$fixture/scripts" "$fixture/metal"
cp "$repo/scripts/check-rust-complexity.sh" "$fixture/scripts/"

{
  printf '%s\n' \
    '#include <metal_stdlib>' \
    'using namespace metal;' \
    '' \
    'kernel void simple_kernel(' \
    '    device uint *values [[buffer(0)]],' \
    '    uint index [[thread_position_in_grid]]) {' \
    '    values[index] += 1;' \
    '}'
} >"$fixture/metal/kernels.metal"

git -C "$fixture" init -q
git -C "$fixture" add scripts/check-rust-complexity.sh metal/kernels.metal
fixture_email=fixture
fixture_email+="@example.invalid"
git -C "$fixture" -c user.email="$fixture_email" -c user.name=fixture commit -qm fixture

output="$tmp/simple-output"
if ! (cd "$fixture" && scripts/check-rust-complexity.sh) >"$output" 2>&1; then
  cat "$output" >&2
  echo "expected the Metal fixture with one attributed kernel to pass" >&2
  exit 1
fi
grep -F 'source category=metal files=1 functions=1' "$output" >/dev/null
grep -F 'metal(cpp)' "$output" >/dev/null

{
  printf '%s\n' \
    '' \
    'kernel void branchy_kernel(' \
    '    device uint *values [[buffer(0)]],' \
    '    uint index [[thread_position_in_grid]]) {' \
    '    uint value = values[index];'
  for branch in $(seq 1 11); do
    printf '    if (value == %d) { value += 1; }\n' "$branch"
  done
  printf '%s\n' \
    '    values[index] = value;' \
    '}'
} >>"$fixture/metal/kernels.metal"
git -C "$fixture" add metal/kernels.metal

output="$tmp/branch-output"
if (cd "$fixture" && scripts/check-rust-complexity.sh) >"$output" 2>&1; then
  cat "$output" >&2
  echo "expected the branch-heavy Metal fixture to fail" >&2
  exit 1
fi
grep -F 'source category=metal files=1 functions=2' "$output" >/dev/null
grep -F 'branchy_kernel [metal]' "$output" >/dev/null

echo "Metal complexity fixtures passed"
