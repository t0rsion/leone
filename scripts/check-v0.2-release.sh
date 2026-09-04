#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
commit=$(git rev-parse HEAD)
runtime="dist/leone-$version-linux-x86_64.tar.gz"
evidence="dist/leone-$version-evidence.tar.gz"

if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
  echo "error: tracked files differ from $commit" >&2
  exit 2
fi

scripts/check-public-tree.sh
cargo +1.92 fmt --all --check
scripts/check-rust-complexity.sh
cargo +1.92 clippy --workspace --all-targets -- -D warnings
cargo +1.92 deny check
RUSTDOCFLAGS="-D warnings" cargo +1.92 doc --workspace --no-deps --lib
taskset -c 16-31 cargo +1.92 test --workspace
taskset -c 16-31 cargo +1.92 test --release -- --ignored --test-threads=1

rendered=$(mktemp)
stage=$(mktemp -d)
trap 'rm -f "$rendered"; rm -rf "$stage"' EXIT HUP INT TERM
scripts/render-v0.2-release.sh "$rendered" >/dev/null
cmp docs/v0.2-release.md "$rendered"

LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh >/dev/null
(cd dist && sha256sum -c "$(basename "$runtime").sha256")
(cd dist && sha256sum -c "$(basename "$evidence").sha256")

tar -xzf "$runtime" -C "$stage"
tar -xzf "$evidence" -C "$stage"
runtime_root="$stage/leone-$version-linux-x86_64"
evidence_root="$stage/leone-$version-evidence"

expected=$(printf 'version=%s\ncommit=%s\n' "$version" "$commit")
[[ $(cat "$runtime_root/BUILD-INFO") == "$expected" ]]
[[ $(cat "$evidence_root/BUILD-INFO") == "$expected" ]]
(cd "$runtime_root" && sha256sum -c MANIFEST.sha256 >/dev/null)
(cd "$evidence_root" && sha256sum -c MANIFEST.sha256 >/dev/null)

if tar -tzf "$runtime" | rg -q '\.(gguf|f16|f32)$'; then
  echo "error: runtime archive contains a model or full-logit artifact" >&2
  exit 2
fi
if tar -tzf "$evidence" | rg -q '\.(gguf|f16|f32)$'; then
  echo "error: evidence archive contains a model or full-logit artifact" >&2
  exit 2
fi

[[ $("$runtime_root/bin/leone" --version) == "leone $version" ]]
binary="$runtime_root/bin/leone"
if strings "$binary" | rg -F -q "$root"; then
  echo "error: release binary contains the workspace path" >&2
  exit 2
fi
if readelf -d "$binary" | rg -q 'RPATH|RUNPATH'; then
  echo "error: release binary contains a runtime search path" >&2
  exit 2
fi
if ldd "$binary" | rg -q 'not found'; then
  echo "error: release binary has an unresolved library" >&2
  exit 2
fi
"$runtime_root/bin/leone" doctor -m models/Qwen3-8B-Q4_K_M.gguf >/dev/null
"$runtime_root/bin/leone" doctor -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf >/dev/null
scripts/check-server-errors.sh \
  "$binary" \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  "$runtime_root/plans/v0.2-llama3.2-1b-sm89.json"

install_root="$stage/install"
PREFIX="$install_root" "$runtime_root/install.sh" >/dev/null
(
  cd "$stage"
  "$root/scripts/check-server-errors.sh" \
    "$install_root/bin/leone" \
    "$root/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "$install_root/share/leone/plans/v0.2-llama3.2-1b-sm89.json"
)

cp "$runtime.sha256" "$stage/runtime-first.sha256"
cp "$evidence.sha256" "$stage/evidence-first.sha256"
LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh >/dev/null
cmp "$stage/runtime-first.sha256" "$runtime.sha256"
cmp "$stage/evidence-first.sha256" "$evidence.sha256"

printf 'v%s candidate %s passed publication checks\n' "$version" "$commit"
