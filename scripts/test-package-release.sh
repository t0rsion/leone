#!/usr/bin/env bash
set -euo pipefail

source_repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
if ! git -C "$source_repo" diff --quiet || ! git -C "$source_repo" diff --cached --quiet; then
    echo "packaging fixtures require a clean source commit" >&2
    exit 2
fi
temporary=$(mktemp -d)
repo="$temporary/source"
cleanup() {
    git -C "$source_repo" worktree remove --force "$repo" >/dev/null 2>&1 || true
    rm -rf "$temporary"
}
trap cleanup EXIT HUP INT TERM
# The fake compiler writes target binaries, so it needs a disposable checkout.
git -C "$source_repo" worktree add --quiet --detach "$repo" HEAD
fixture="$repo/tests/fixtures"
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$repo/Cargo.toml" | head -1)
ln -s "$fixture/fake-cargo" "$temporary/cargo"

PATH="$temporary:$PATH" \
LEONE_FAKE_BINARY="$fixture/fake-leone" \
LEONE_FAKE_CARGO_LOG="$temporary/linux-cargo-args" \
LEONE_PACKAGE_DIST="$temporary/linux-dist" \
SOURCE_DATE_EPOCH=1700000000 \
CARGO="$temporary/cargo" \
    "$repo/scripts/package-release.sh" --platform linux-x86_64 --runtime-only >/dev/null

PATH="$temporary:$PATH" \
LEONE_FAKE_BINARY="$fixture/fake-leone" \
LEONE_FAKE_CARGO_LOG="$temporary/linux-cargo-args-second" \
LEONE_PACKAGE_DIST="$temporary/linux-dist-second" \
SOURCE_DATE_EPOCH=1700000000 \
CARGO="$temporary/cargo" \
    "$repo/scripts/package-release.sh" --platform linux-x86_64 --runtime-only >/dev/null

cmp "$temporary/linux-dist/leone-$version-linux-x86_64.tar.gz" \
    "$temporary/linux-dist-second/leone-$version-linux-x86_64.tar.gz"
cmp "$temporary/linux-dist/leone-$version-linux-x86_64.tar.gz.sha256" \
    "$temporary/linux-dist-second/leone-$version-linux-x86_64.tar.gz.sha256"

PATH="$temporary:$PATH" \
LEONE_FAKE_BINARY="$fixture/fake-leone" \
LEONE_FAKE_CARGO_LOG="$temporary/evidence-cargo-args" \
LEONE_PACKAGE_DIST="$temporary/evidence-dist" \
LEONE_RELEASE_EVIDENCE_MANIFEST=packaging/release-evidence.v0.3.json \
SOURCE_DATE_EPOCH=1700000000 \
CARGO="$temporary/cargo" \
    "$repo/scripts/package-release.sh" --platform linux-x86_64 >/dev/null
evidence_archive="$temporary/evidence-dist/leone-$version-evidence.tar.gz"
test -f "$evidence_archive"
evidence_extract="$temporary/evidence-extract"
mkdir "$evidence_extract"
tar -xzf "$evidence_archive" -C "$evidence_extract"
test -f "$evidence_extract/leone-$version-evidence/release-evidence.json"
test -f "$evidence_extract/leone-$version-evidence/README.md"

prebuilt="$temporary/prebuilt-leone"
commit=$(git -C "$repo" rev-parse HEAD)
cat >"$prebuilt" <<EOF
#!/bin/sh
set -eu
case "\${1:-}" in
    --version) printf 'leone fixture\\n' ;;
    --build-info) printf '{"schema_version":"leone.build-info.v1","source_commit":"$commit","source_tree_dirty":false,"provenance_unknown":false,"profile":"release","features":"cuda","target":"x86_64-unknown-linux-gnu"}\\n' ;;
    *) printf 'fixture binary\\n' ;;
esac
EOF
chmod 0755 "$prebuilt"
PATH="$temporary:$PATH" \
LEONE_RELEASE_BINARY="$prebuilt" \
LEONE_CARGO=/bin/false \
LEONE_PACKAGE_DIST="$temporary/prebuilt-dist" \
SOURCE_DATE_EPOCH=1700000000 \
    "$repo/scripts/package-release.sh" --platform linux-x86_64 --runtime-only >/dev/null
test ! -e "$temporary/cargo-should-not-run"

source_manifest=receipts/source-inputs-v04-prestudy.json
source_manifest_commit=$(jq -er '.source_commit' "$repo/$source_manifest")
manifest_prebuilt="$temporary/manifest-prebuilt-leone"
cat >"$manifest_prebuilt" <<EOF
#!/bin/sh
set -eu
case "\${1:-}" in
    --version) printf 'leone fixture\\n' ;;
    --build-info) printf '{"schema_version":"leone.build-info.v1","source_commit":"$source_manifest_commit","source_tree_dirty":false,"provenance_unknown":false,"profile":"release","features":"cuda","target":"x86_64-unknown-linux-gnu"}\\n' ;;
    *) printf 'fixture binary\\n' ;;
esac
EOF
chmod 0755 "$manifest_prebuilt"
LEONE_RELEASE_BINARY="$manifest_prebuilt" \
LEONE_RELEASE_SOURCE_MANIFEST="$source_manifest" \
LEONE_CARGO=/bin/false \
LEONE_PACKAGE_DIST="$temporary/manifest-prebuilt-dist" \
SOURCE_DATE_EPOCH=1700000000 \
    "$repo/scripts/package-release.sh" --platform linux-x86_64 --runtime-only >/dev/null
manifest_prebuilt_archive="$temporary/manifest-prebuilt-dist/leone-$version-linux-x86_64.tar.gz"
"$repo/scripts/verify-release-archive.sh" "$manifest_prebuilt_archive" >/dev/null
manifest_prebuilt_extract="$temporary/manifest-prebuilt-extract"
mkdir "$manifest_prebuilt_extract"
tar -xzf "$manifest_prebuilt_archive" -C "$manifest_prebuilt_extract"
cmp "$repo/$source_manifest" \
    "$manifest_prebuilt_extract/leone-$version-linux-x86_64/receipts/source-inputs-v04-prestudy.json"

bad_prebuilt="$temporary/bad-prebuilt-leone"
cat >"$bad_prebuilt" <<EOF
#!/bin/sh
set -eu
case "\${1:-}" in
    --build-info) printf '{"schema_version":"leone.build-info.v1","source_commit":"$commit","source_tree_dirty":false,"provenance_unknown":true,"profile":"release","features":"metal","target":"x86_64-unknown-linux-gnu"}\\n' ;;
    *) printf 'fixture binary\\n' ;;
esac
EOF
chmod 0755 "$bad_prebuilt"
if LEONE_RELEASE_BINARY="$bad_prebuilt" \
    LEONE_CARGO=/bin/false \
    LEONE_PACKAGE_DIST="$temporary/bad-prebuilt-dist" \
    SOURCE_DATE_EPOCH=1700000000 \
    "$repo/scripts/package-release.sh" --platform linux-x86_64 --runtime-only \
    >"$temporary/bad-prebuilt-output" 2>&1; then
    echo "invalid prebuilt provenance was accepted" >&2
    exit 1
fi
grep -F 'invalid release provenance' "$temporary/bad-prebuilt-output" >/dev/null

linux_archive="$temporary/linux-dist/leone-$version-linux-x86_64.tar.gz"
"$repo/scripts/verify-release-archive.sh" "$linux_archive" >/dev/null
linux_extract="$temporary/linux-extract"
mkdir "$linux_extract"
tar -xzf "$linux_archive" -C "$linux_extract"
linux_runtime="$linux_extract/leone-$version-linux-x86_64"
test -d "$linux_runtime/plans"
test -f "$linux_runtime/plans/llama3.2-1b-sm89.json"
test -f "$linux_runtime/plans/qwen3-8b-sm89.json"
cmp "$repo/receipts/source-inputs-v04-prestudy.json" \
    "$linux_runtime/receipts/source-inputs-v04-prestudy.json"
grep -F '"backend": "cuda"' "$linux_runtime/package-info.json" >/dev/null
grep -F '"target": "x86_64-unknown-linux-gnu"' "$linux_runtime/package-info.json" >/dev/null

PATH="$temporary:$PATH" \
LEONE_FAKE_BINARY="$fixture/fake-leone" \
LEONE_FAKE_CARGO_LOG="$temporary/cargo-args" \
LEONE_PACKAGE_DIST="$temporary/dist" \
SOURCE_DATE_EPOCH=1700000000 \
CARGO="$temporary/cargo" \
LEONE_RELEASE_EVIDENCE_MANIFEST=packaging/release-evidence.v0.4.json \
    "$repo/scripts/package-release.sh" --platform darwin-arm64 --runtime-only >/dev/null

PATH="$temporary:$PATH" \
LEONE_FAKE_BINARY="$fixture/fake-leone" \
LEONE_FAKE_CARGO_LOG="$temporary/cargo-args-second" \
LEONE_PACKAGE_DIST="$temporary/dist-second" \
SOURCE_DATE_EPOCH=1700000000 \
CARGO="$temporary/cargo" \
LEONE_RELEASE_EVIDENCE_MANIFEST=packaging/release-evidence.v0.4.json \
    "$repo/scripts/package-release.sh" --platform darwin-arm64 --runtime-only >/dev/null

cmp "$temporary/dist/leone-$version-darwin-arm64.tar.gz" \
    "$temporary/dist-second/leone-$version-darwin-arm64.tar.gz"
cmp "$temporary/dist/leone-$version-darwin-arm64.tar.gz.sha256" \
    "$temporary/dist-second/leone-$version-darwin-arm64.tar.gz.sha256"

if PATH="$temporary:$PATH" \
    LEONE_FAKE_BINARY="$fixture/fake-leone" \
    LEONE_FAKE_CARGO_LOG="$temporary/cargo-args-incomplete" \
    LEONE_PACKAGE_DIST="$temporary/incomplete-dist" \
    LEONE_RELEASE_EVIDENCE_MANIFEST=packaging/release-evidence.v0.4.json \
    "$repo/scripts/package-release.sh" --platform darwin-arm64 >/dev/null 2>&1; then
    echo "incomplete v0.4 evidence manifest was accepted" >&2
    exit 1
fi

grep -Fx -- '--target' "$temporary/cargo-args" >/dev/null
grep -Fx -- 'aarch64-apple-darwin' "$temporary/cargo-args" >/dev/null
grep -Fx -- '--no-default-features' "$temporary/cargo-args" >/dev/null
grep -Fx -- '--features' "$temporary/cargo-args" >/dev/null
grep -Fx -- 'metal' "$temporary/cargo-args" >/dev/null

archive="$temporary/dist/leone-$version-darwin-arm64.tar.gz"
"$repo/scripts/verify-release-archive.sh" "$archive" >/dev/null
extract="$temporary/extract"
mkdir "$extract"
tar -xzf "$archive" -C "$extract"
runtime="$extract/leone-$version-darwin-arm64"
cmp "$repo/receipts/source-inputs-v04-prestudy.json" \
    "$runtime/receipts/source-inputs-v04-prestudy.json"
grep -F '"backend": "metal"' "$runtime/package-info.json" >/dev/null
grep -F '"target": "aarch64-apple-darwin"' "$runtime/package-info.json" >/dev/null
grep -F '"signature": null' "$runtime/package-info.json" >/dev/null
test ! -d "$runtime/plans"
host=$(uname -s):$(uname -m)
case "$host" in
    Linux:x86_64)
        native_runtime=$linux_runtime
        foreign_runtime=$runtime
        ;;
    Darwin:arm64|Darwin:aarch64)
        native_runtime=$runtime
        foreign_runtime=$linux_runtime
        ;;
    *)
        echo "packaging fixture does not support host $host" >&2
        exit 2
        ;;
esac
PREFIX="$temporary/install" "$native_runtime/install.sh" >/dev/null
test -x "$temporary/install/bin/leone"
test "$("$temporary/install/bin/leone" --version)" = 'leone fixture'

foreign_prefix="$temporary/foreign-install"
mkdir "$foreign_prefix"
printf 'sentinel\n' >"$foreign_prefix/sentinel"
if PREFIX="$foreign_prefix" "$foreign_runtime/install.sh" \
    >"$temporary/foreign-install-output" 2>&1; then
    echo "foreign archive was installed" >&2
    exit 1
fi
grep -F 'does not match host' "$temporary/foreign-install-output" >/dev/null
test "$(cat "$foreign_prefix/sentinel")" = sentinel
test ! -e "$foreign_prefix/bin"

dirty_marker="$repo/tests/fixtures/.package-release-dirty"
printf 'dirty\n' >"$dirty_marker"
if PATH="$temporary:$PATH" \
    LEONE_FAKE_BINARY="$fixture/fake-leone" \
    LEONE_PACKAGE_DIST="$temporary/dirty-dist" \
    CARGO="$temporary/cargo" \
    "$repo/scripts/package-release.sh" --platform darwin-arm64 --runtime-only \
    >"$temporary/dirty-output" 2>&1; then
    echo "dirty source tree was accepted" >&2
    exit 1
fi
grep -F 'requires a clean source tree' "$temporary/dirty-output" >/dev/null
rm -f "$dirty_marker"

echo "Native packaging fixtures passed"
