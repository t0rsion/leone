#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
commit=$(git rev-parse HEAD)
binary_archive="dist/leone-$version-linux-x86_64.tar.gz"
evidence_archive="dist/leone-$version-evidence.tar.gz"
study=receipts/batched-service-study.json

if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
  echo "tracked files differ from $commit" >&2
  exit 2
fi

source_commit=$(jq -er '.source_commit' "$study")
python3 scripts/source_inputs.py check "$source_commit"

scripts/check-public-tree.sh
taskset -c 16-31 scripts/test-check-public-tree.sh
python3 scripts/check-release-evidence.py
taskset -c 16-31 python3 -m unittest discover -s tests
taskset -c 16-31 cargo +1.92 fmt --all --check
taskset -c 16-31 scripts/check-rust-complexity.sh
taskset -c 16-31 cargo +1.92 clippy --workspace --all-targets --locked -- -D warnings
taskset -c 16-31 cargo +1.92 deny check
RUSTDOCFLAGS="-D warnings" taskset -c 16-31 cargo +1.92 doc --workspace --no-deps --locked
taskset -c 16-31 cargo +1.92 test --workspace --locked
taskset -c 16-31 cargo +1.92 test --release --locked -- --ignored --test-threads=1

for model in \
  models/Qwen3-8B-Q4_K_M.gguf \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf; do
  taskset -c 16-31 target/release/leone verify session -m "$model"
  taskset -c 16-31 target/release/leone verify scheduler -m "$model"
  taskset -c 16-31 target/release/leone verify fork -m "$model" --exact-only
  taskset -c 16-31 target/release/leone verify hibernate -m "$model" --exact-only
done

rendered=$(mktemp)
stage=$(mktemp -d)
trap 'rm -f "$rendered"; rm -rf "$stage"' EXIT HUP INT TERM
scripts/render-release-evidence.sh "$rendered" >/dev/null
cmp docs/release-evidence.md "$rendered"
python3 scripts/render-concurrent-evidence.py "$rendered" >/dev/null
cmp docs/concurrent-service-evidence.md "$rendered"
python3 scripts/plot-concurrent-evidence.py "$stage/concurrent-service.svg" >/dev/null
cmp docs/concurrent-service.svg "$stage/concurrent-service.svg"

LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh >/dev/null
(cd dist && sha256sum -c "$(basename "$binary_archive").sha256")
(cd dist && sha256sum -c "$(basename "$evidence_archive").sha256")

tar -xzf "$binary_archive" -C "$stage"
tar -xzf "$evidence_archive" -C "$stage"
binary_root="$stage/leone-$version-linux-x86_64"
evidence_root="$stage/leone-$version-evidence"

scripts/check-public-tree.sh "$binary_root"
scripts/check-public-tree.sh "$evidence_root"

(cd "$binary_root" && sha256sum -c MANIFEST.sha256 >/dev/null)
(cd "$evidence_root" && sha256sum -c MANIFEST.sha256 >/dev/null)

for archive in "$binary_archive" "$evidence_archive"; do
  if tar -tzf "$archive" | rg -q '\.(gguf|f16|f32|pem|key)$'; then
    echo "release archive contains a model, logit artifact, or key: $archive" >&2
    exit 2
  fi
done

binary="$binary_root/bin/leone"
[[ $("$binary" --version) == "leone $version" ]]
"$binary" --build-info | jq -e --arg commit "$commit" --arg version "$version" '
  .source_commit == $commit and .version == $version and
  .source_tree_dirty == false and .profile == "release"
' >/dev/null
if strings "$binary" | rg -F -q "$root"; then
  echo "release binary contains the workspace path" >&2
  exit 2
fi
if readelf -d "$binary" | rg -q 'RPATH|RUNPATH'; then
  echo "release binary contains a runtime search path" >&2
  exit 2
fi
if ldd "$binary" | rg -q 'not found'; then
  echo "release binary has an unresolved library" >&2
  exit 2
fi

"$binary" doctor -m models/Qwen3-8B-Q4_K_M.gguf >/dev/null
"$binary" doctor -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf >/dev/null
scripts/check-server-errors.sh \
  "$binary" \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  "$binary_root/plans/llama3.2-1b-sm89.json"
taskset -c 16-31 scripts/check-openai-client.sh \
  "$binary" models/Llama-3.2-1B-Instruct-Q4_K_M.gguf "$stage/client-check.json"

install_root="$stage/install"
PREFIX="$install_root" "$binary_root/install.sh" >/dev/null
(
  cd "$stage"
  "$root/scripts/check-server-errors.sh" \
    "$install_root/bin/leone" \
    "$root/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "$install_root/share/leone/plans/llama3.2-1b-sm89.json"
)

cp "$binary_archive.sha256" "$stage/binary-first.sha256"
cp "$evidence_archive.sha256" "$stage/evidence-first.sha256"
LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh >/dev/null
cmp "$stage/binary-first.sha256" "$binary_archive.sha256"
cmp "$stage/evidence-first.sha256" "$evidence_archive.sha256"

printf '%s candidate %s passed publication checks\n' "$version" "$commit"
