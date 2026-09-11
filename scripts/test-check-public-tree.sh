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
  local output="$tmp/output"
  if "$gate" "$directory" >"$output" 2>&1; then
    cat "$output" >&2
    fail "expected the public-tree gate to reject $directory"
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

home_prefix=/ho
home_prefix+=me/
personal_path="${home_prefix}fixture-user"
mkdir "$tmp/personal"
printf 'path %s\n' "$personal_path" >"$tmp/personal/path.md"
expect_fail "$tmp/personal"

mkdir -p "$tmp/gate-name/scripts"
printf 'path %s\n' "$personal_path" >"$tmp/gate-name/scripts/check-public-tree.py"
expect_fail "$tmp/gate-name"

key_prefix=AK
key_prefix+=IA
mkdir "$tmp/key"
printf 'key %s\n' "${key_prefix}1234567890123456" >"$tmp/key/key.txt"
expect_fail "$tmp/key"

email_user=fixture
email_domain=example.com
mkdir "$tmp/email"
printf 'mail %s@%s\n' "$email_user" "$email_domain" >"$tmp/email/mail.txt"
expect_fail "$tmp/email"

record_prefix=Sub
record_prefix+=agent
mkdir "$tmp/record"
printf '%s id\n' "$record_prefix" >"$tmp/record/record.txt"
expect_fail "$tmp/record"

mkdir "$tmp/binary"
printf 'x\0%s\0y\n' "$personal_path" >"$tmp/binary/model.bin"
expect_fail "$tmp/binary"

version=v
version+=9.8.7
mkdir "$tmp/version"
printf 'release %s\n' "$version" >"$tmp/version/version.md"
expect_fail "$tmp/version"

commit_hash=deadbeef
commit_hash+=deadbeef
commit_hash+=deadbeef
commit_hash+=deadbeef
commit_hash+=deadbeef
mkdir "$tmp/commit"
printf 'commit %s\n' "$commit_hash" >"$tmp/commit/commit.md"
expect_fail "$tmp/commit"

mkdir "$tmp/branch"
printf 'branch: feature/private-a\n' >"$tmp/branch/branch.md"
expect_fail "$tmp/branch"

mkdir "$tmp/comment"
printf '// release %s\n' "$version" >"$tmp/comment/comment.rs"
expect_fail "$tmp/comment"

mkdir "$tmp/block-comment"
printf '/*\nrelease %s\n*/\n' "$version" >"$tmp/block-comment/comment.c"
expect_fail "$tmp/block-comment"

mkdir "$tmp/allowed-only"
printf 'Qwen3-8B uses /v1. Transactions commit state. Session branches stay independent.\n' >"$tmp/allowed-only/allowed.md"
printf 'cargo %s\n' "$version" >"$tmp/allowed-only/toolchain.md"
printf 'source_commit: %s\n' "$commit_hash" >"$tmp/allowed-only/receipt.md"
printf 'release %s\n' "$version" >"$tmp/allowed-only/CHANGELOG.md"
expect_pass "$tmp/allowed-only"

mkdir "$tmp/url-only"
printf 'source https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0\n' >"$tmp/url-only/source.md"
expect_pass "$tmp/url-only"

mkdir "$tmp/changelog-pii"
printf 'path %s\n' "$personal_path" >"$tmp/changelog-pii/CHANGELOG.md"
expect_fail "$tmp/changelog-pii"

mkdir "$tmp/docstring"
printf '"""release %s"""\n' "$version" >"$tmp/docstring/doc.py"
expect_fail "$tmp/docstring"

for private_name in AGENTS CLAUDE PLAN; do
  mkdir "$tmp/forbidden-$private_name"
  printf 'local notes\n' >"$tmp/forbidden-$private_name/$private_name.md"
  expect_fail "$tmp/forbidden-$private_name"
done

mkdir "$tmp/symlink"
printf 'outside\n' >"$tmp/symlink/outside.txt"
ln -s outside.txt "$tmp/symlink/link.txt"
expect_fail "$tmp/symlink"

source_repo="$tmp/source-repo"
mkdir -p "$source_repo/scripts" "$source_repo/target"
cp "$gate" "$repo/scripts/check-public-tree.py" "$source_repo/scripts/"
printf 'target/\n' >"$source_repo/.gitignore"
git -C "$source_repo" init -q
git -C "$source_repo" add .gitignore scripts
git_user=fixture
git_domain=example.invalid
git -C "$source_repo" -c "user.email=${git_user}@${git_domain}" -c user.name=fixture commit -qm fixture
source_fixture="$source_repo/source.md"
ignored_fixture="$source_repo/target/.public-tree-ignored.md"
printf 'path %s\n' "$personal_path" >"$source_fixture"
if (cd "$source_repo" && scripts/check-public-tree.sh) >"$tmp/source-output" 2>&1; then
  cat "$tmp/source-output" >&2
  fail "expected the source gate to reject an untracked nonignored file"
fi
rm -f "$source_fixture"

printf 'path %s\n' "$personal_path" >"$ignored_fixture"
if ! (cd "$source_repo" && scripts/check-public-tree.sh) >"$tmp/ignored-output" 2>&1; then
  cat "$tmp/ignored-output" >&2
  fail "expected an ignored file to stay outside the source scan"
fi

echo "check-public-tree fixtures passed"
