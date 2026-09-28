#!/usr/bin/env bash
set -euo pipefail

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
gate="$repo/scripts/check-public-tree.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

fail() {
  echo "$1" >&2
  exit 1
}

expect_fail() {
  local directory=$1
  local expected=$2
  local output="$tmp/output"
  if "$gate" "$directory" >"$output" 2>&1; then
    cat "$output" >&2
    fail "expected the public-tree gate to reject $directory"
  fi
  if ! rg -Fq -- "$expected" "$output"; then
    cat "$output" >&2
    fail "expected diagnostic: $expected"
  fi
}

expect_pass() {
  local directory=$1
  local output="$tmp/output"
  if ! "$gate" "$directory" >"$output" 2>&1; then
    cat "$output" >&2
    fail "expected the public-tree gate to accept $directory"
  fi
}

runtime="$tmp/accepted-runtime"
runtime_markdown=(
  README.md
  docs/branching-service-evidence.md
  docs/client-workflow.md
  docs/concurrent-service-evidence.md
  docs/memory-accounting.md
  docs/models.md
  docs/openai-api.md
  docs/oracle.md
  docs/release-candidate.md
  docs/release-evidence.md
  docs/release.md
  receipts/INDEX.md
)
for relative in "${runtime_markdown[@]}"; do
  mkdir -p "$runtime/$(dirname "$relative")"
  printf 'Qwen3-8B uses /v1. source https://example.com/evidence.\n' >"$runtime/$relative"
done
expect_pass "$runtime"

new_runtime() {
  cp -a "$runtime" "$1"
}

home_prefix=/ho
home_prefix+=me/
personal_path="${home_prefix}fixture-user"
new_runtime "$tmp/personal"
printf 'path %s\n' "$personal_path" >"$tmp/personal/README.md"
expect_fail "$tmp/personal" "an absolute personal path"

new_runtime "$tmp/gate-name"
mkdir -p "$tmp/gate-name/scripts"
printf 'path %s\n' "$personal_path" >"$tmp/gate-name/scripts/check-public-tree.py"
expect_fail "$tmp/gate-name" "an absolute personal path"

key_prefix=AK
key_prefix+=IA
new_runtime "$tmp/key"
printf 'key %s\n' "${key_prefix}1234567890123456" >"$tmp/key/key.txt"
expect_fail "$tmp/key" "a private key or credential token"

email_user=fixture
email_domain=example.com
new_runtime "$tmp/email"
printf 'mail %s@%s\n' "$email_user" "$email_domain" >"$tmp/email/mail.txt"
expect_fail "$tmp/email" "an email address"

record_prefix=Sub
record_prefix+=agent
new_runtime "$tmp/record"
printf '%s id\n' "$record_prefix" >"$tmp/record/record.txt"
expect_fail "$tmp/record" "an internal agent record"

new_runtime "$tmp/binary"
printf 'x\0%s\0y\n' "$personal_path" >"$tmp/binary/model.bin"
expect_fail "$tmp/binary" "an absolute personal path"

version=v
version+=9.8.7
new_runtime "$tmp/version"
printf 'release %s\n' "$version" >"$tmp/version/README.md"
expect_fail "$tmp/version" "a narrative version identifier"

commit_hash=deadbeef
commit_hash+=deadbeef
commit_hash+=deadbeef
commit_hash+=deadbeef
commit_hash+=deadbeef
new_runtime "$tmp/commit"
printf 'commit %s\n' "$commit_hash" >"$tmp/commit/README.md"
expect_fail "$tmp/commit" "a narrative commit or revision hash"

new_runtime "$tmp/branch"
printf 'branch: feature/private-a\n' >"$tmp/branch/README.md"
expect_fail "$tmp/branch" "a narrative branch identifier"

new_runtime "$tmp/comment"
printf '// release %s\n' "$version" >"$tmp/comment/comment.rs"
expect_fail "$tmp/comment" "a narrative version identifier"

new_runtime "$tmp/block-comment"
printf '/*\nrelease %s\n*/\n' "$version" >"$tmp/block-comment/comment.c"
expect_fail "$tmp/block-comment" "a narrative version identifier"

new_runtime "$tmp/star-block-comment"
printf '/*\n* release %s\n*/\n' "$version" >"$tmp/star-block-comment/comment.c"
expect_fail "$tmp/star-block-comment" "a narrative version identifier"

new_runtime "$tmp/pointer-assignment"
printf '*hash *= 1099511628211ULL;\n' >"$tmp/pointer-assignment/digest.c"
expect_pass "$tmp/pointer-assignment"

evidence="$tmp/accepted-evidence"
mkdir -p "$evidence/receipts"
printf 'source_commit: %s\n' "$commit_hash" >"$evidence/README.md"
printf 'receipt index\n' >"$evidence/receipts/INDEX.md"
expect_pass "$evidence"

unexpected_archive="$tmp/unexpected-archive"
cp -a "$runtime" "$unexpected_archive"
printf 'internal notes\n' >"$unexpected_archive/docs/internal.md"
expect_fail "$unexpected_archive" "unexpected Markdown file: docs/internal.md"

new_runtime "$tmp/docstring"
printf '"""release %s"""\n' "$version" >"$tmp/docstring/doc.py"
expect_fail "$tmp/docstring" "a narrative version identifier"

for private_name in AGENTS CLAUDE PLAN; do
  new_runtime "$tmp/forbidden-$private_name"
  printf 'local notes\n' >"$tmp/forbidden-$private_name/$private_name.md"
  expect_fail "$tmp/forbidden-$private_name" "extracted tree contains a forbidden path"
done

new_runtime "$tmp/symlink"
printf 'outside\n' >"$tmp/symlink/outside.txt"
ln -s outside.txt "$tmp/symlink/link.txt"
expect_fail "$tmp/symlink" "link.txt (symlink)"

source_repo="$tmp/source-repo"
mkdir -p "$source_repo/scripts" "$source_repo/target"
cp "$gate" "$repo/scripts/check-public-tree.py" "$source_repo/scripts/"
printf 'target/\nlocal/\n' >"$source_repo/.gitignore"
source_markdown=(
  README.md
  CHANGELOG.md
  ROADMAP.md
  benchmarks/README.md
  docs/branching-service-evidence.md
  docs/client-workflow.md
  docs/concurrent-service-evidence.md
  docs/memory-accounting.md
  docs/models.md
  docs/openai-api.md
  docs/oracle.md
  docs/release-candidate.md
  docs/release-evidence.md
  docs/release.md
  docs/speculation.md
  packaging/EVIDENCE.md
  packaging/README.md
  receipts/INDEX.md
  research/prefix_attention/ORACLE_CONTRACT.md
  research/prefix_attention/README.md
)
for relative in "${source_markdown[@]}"; do
  mkdir -p "$source_repo/$(dirname "$relative")"
  printf 'public source fixture\n' >"$source_repo/$relative"
done
git -C "$source_repo" init -q
git -C "$source_repo" add .gitignore scripts "${source_markdown[@]}"
git_user=fixture
git_domain=example.invalid
git -C "$source_repo" -c "user.email=${git_user}@${git_domain}" -c user.name=fixture commit -qm fixture

mkdir -p "$source_repo/native"
cat >"$source_repo/native/bridge.m" <<EOF
#import <Foundation/Foundation.h>
// source_commit: $commit_hash
void bridge_fixture(void) {}
EOF
cat >"$source_repo/native/kernel.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;

kernel void kernel_fixture(device float *values [[buffer(0)]], uint index [[thread_position_in_grid]]) {
  values[index] = values[index];
}
EOF
printf '// source fixture\n' >"$source_repo/native/bridge.mm"
printf '// source fixture\n' >"$source_repo/native/runner.swift"
git -C "$source_repo" add native
git -C "$source_repo" -c "user.email=${git_user}@${git_domain}" -c user.name=fixture commit -qm native-fixture
if ! (cd "$source_repo" && scripts/check-public-tree.sh) >"$tmp/source-accepted-output" 2>&1; then
  cat "$tmp/source-accepted-output" >&2
  fail "expected the exact tracked Markdown and native source set to pass"
fi

expect_source_fail() {
  local expected=$1
  local output="$tmp/source-output"
  if (cd "$source_repo" && scripts/check-public-tree.sh) >"$output" 2>&1; then
    cat "$output" >&2
    fail "expected the source gate to reject its fixture"
  fi
  if ! rg -Fq -- "$expected" "$output"; then
    cat "$output" >&2
    fail "expected source diagnostic: $expected"
  fi
}

for source in bridge.m bridge.mm runner.swift; do
  printf '// release %s\n' "$version" >"$source_repo/native/$source"
  expect_source_fail "native/$source:1: a narrative version identifier"
  git -C "$source_repo" checkout -q -- "native/$source"
done

comment_open='/* '
branch_word=branch
branch_value=feature/private-a
printf '%s%s: %s */\n' "$comment_open" "$branch_word" "$branch_value" >"$source_repo/native/kernel.metal"
expect_source_fail "native/kernel.metal:1: a narrative branch identifier"
git -C "$source_repo" checkout -q -- native/kernel.metal

printf 'path %s\n' "$personal_path" >"$source_repo/CHANGELOG.md"
expect_source_fail "CHANGELOG.md:1: an absolute personal path"
git -C "$source_repo" checkout -q -- CHANGELOG.md

mkdir -p "$source_repo/docs"
printf 'internal notes\n' >"$source_repo/docs/unexpected.md"
git -C "$source_repo" add docs/unexpected.md
if (cd "$source_repo" && scripts/check-public-tree.sh) >"$tmp/tracked-output" 2>&1; then
  cat "$tmp/tracked-output" >&2
  fail "expected an unexpected tracked Markdown file to fail"
fi
git -C "$source_repo" rm -q --cached docs/unexpected.md
rm -f "$source_repo/docs/unexpected.md"

source_fixture="$source_repo/source.md"
ignored_fixture="$source_repo/target/.public-tree-ignored.md"
printf 'path %s\n' "$personal_path" >"$source_fixture"
if (cd "$source_repo" && scripts/check-public-tree.sh) >"$tmp/source-output" 2>&1; then
  cat "$tmp/source-output" >&2
  fail "expected the source gate to reject an untracked nonignored file"
fi
rm -f "$source_fixture"

printf 'path %s\n' "$personal_path" >"$ignored_fixture"
mkdir -p "$source_repo/local"
printf 'path %s\n' "$personal_path" >"$source_repo/local/.public-tree-ignored.md"
if ! (cd "$source_repo" && scripts/check-public-tree.sh) >"$tmp/ignored-output" 2>&1; then
  cat "$tmp/ignored-output" >&2
  fail "expected an ignored file to stay outside the source scan"
fi

echo "check-public-tree fixtures passed"
