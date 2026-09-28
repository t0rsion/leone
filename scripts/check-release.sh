#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
commit=$(git rev-parse HEAD)
binary_archive="dist/leone-$version-linux-x86_64.tar.gz"
evidence_archive="dist/leone-$version-evidence.tar.gz"
release_manifest=${LEONE_RELEASE_EVIDENCE_MANIFEST:-$(python3 scripts/release-evidence-manifest.py select --root "$root")}

if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
  echo "tracked files differ from $commit" >&2
  exit 2
fi

release_line=$(jq -er '.release_line' "$release_manifest")
source_manifest=""
if [[ $release_line == v0.4 ]]; then
  source_manifest=$(jq -er '.source_manifest' "$release_manifest")
fi
if [[ $release_line == v0.3 ]]; then
  study=receipts/batched-service-study.json
  source_commit=$(jq -er '.source_commit' "$study")
  python3 scripts/source_inputs.py check "$source_commit"
fi

scripts/check-public-tree.sh
taskset -c 16-31 scripts/test-check-public-tree.sh
taskset -c 16-31 tests/test-fetch-llama-cpp.sh
receipt_validator_args=()
if [[ -n ${LEONE_TRUSTED_RECEIPT_VALIDATOR:-} ]]; then
  receipt_validator_args+=(--trusted-receipt-validator "$LEONE_TRUSTED_RECEIPT_VALIDATOR")
fi
if [[ $release_line == v0.4 ]]; then
  if ((${#receipt_validator_args[@]} == 0)); then
    receipt_validator_args=(--trusted-receipt-validator target/release/leone-receipt-verify)
  fi
  [[ -x ${receipt_validator_args[1]} && ! -L ${receipt_validator_args[1]} ]] || {
    echo "v0.4 verification requires a trusted CPU receipt validator" >&2
    exit 2
  }
  python3 scripts/check-release-evidence.py \
    --root "$root" \
    --manifest "$release_manifest" \
    --offline \
    "${receipt_validator_args[@]}"
else
  python3 scripts/check-release-evidence.py \
    --manifest "$release_manifest" \
    "${receipt_validator_args[@]}"
fi
taskset -c 16-31 python3 -m unittest discover -s tests
scripts/check-quantized-differential.sh
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
if [[ $release_line == v0.4 ]]; then
  batching_study=$(jq -er '
    [.backend_requirements[] | select(.backend == "cuda") | .records[] | select(.role == "batching") | .path]
    | if length == 1 then .[0] else error("v0.4 evidence needs one CUDA batching record") end
  ' "$release_manifest")
  scripts/render-release-evidence.sh "$rendered" "$batching_study" >/dev/null
else
  scripts/render-release-evidence.sh "$rendered" >/dev/null
fi
cmp docs/release-evidence.md "$rendered"
python3 scripts/render-concurrent-evidence.py "$rendered" >/dev/null
cmp docs/concurrent-service-evidence.md "$rendered"
python3 scripts/plot-concurrent-evidence.py "$stage/concurrent-service.svg" >/dev/null
cmp docs/concurrent-service.svg "$stage/concurrent-service.svg"

if [[ $release_line == v0.4 ]]; then
  release_binary=${LEONE_RELEASE_BINARY:-target/release/leone}
  [[ -f $release_binary && ! -L $release_binary ]] || {
    echo "v0.4 packaging requires the measured binary at $release_binary" >&2
    exit 2
  }
  LEONE_BUILD_CPUSET=16-31 \
  LEONE_RELEASE_BINARY="$release_binary" \
    scripts/package-release.sh >/dev/null
else
  release_binary=target/release/leone
  LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh >/dev/null
fi
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

if [[ $release_line == v0.4 ]]; then
  python3 scripts/check-release-evidence.py \
    --root "$evidence_root" \
    --manifest "$evidence_root/release-evidence.json" \
    --offline \
    "${receipt_validator_args[@]}"
fi

for archive in "$binary_archive" "$evidence_archive"; do
  if tar -tzf "$archive" | rg -q '\.(gguf|f16|pem|key)$'; then
    echo "release archive contains a model or key: $archive" >&2
    exit 2
  fi
done
if tar -tzf "$binary_archive" | rg -qi '\.f32$'; then
  echo "runtime archive contains an f32 artifact" >&2
  exit 2
fi

while IFS= read -r path; do
  relative=${path#"$evidence_root/"}
  if ! jq -e --arg path "$relative" '
    . as $manifest
    | select(any($manifest.files[]; .destination == $path))
    | $manifest.backend_requirements[]
    | select(.backend == "cuda")
    | .records[]
    | select(.role == "quality")
    | (.path | sub("\\.json$"; "") + "-cuda-samples/") as $root
    | select($path | startswith($root))
    | select(.artifact_files | index($path))
    | select($path | ltrimstr($root) | IN("oracle.f32", "llama_cpp.f32", "leone.f32"))
  ' "$evidence_root/release-evidence.json" >/dev/null; then
    echo "evidence archive contains an undeclared f32 artifact: $relative" >&2
    exit 2
  fi
done < <(find "$evidence_root" -type f -iname '*.f32' -print)

binary="$binary_root/bin/leone"
if [[ $release_line == v0.4 ]]; then
  expected_binary_sha256=$(jq -er \
    --arg platform "linux-x86_64" \
    --arg target "x86_64-unknown-linux-gnu" \
    --arg backend "cuda" \
    '.shared_records[] | select(.role == "client" and .platform == $platform and .target == $target and .backend == $backend) | .binary_sha256' \
    "$evidence_root/release-evidence.json")
  actual_binary_sha256=$(packaging/checksum.sh "$binary" | awk '{print $1}')
  [[ $actual_binary_sha256 == "$expected_binary_sha256" ]] || {
    echo "runtime binary differs from Linux CUDA client evidence" >&2
    exit 2
  }
fi
[[ $("$binary" --version) == "leone $version" ]]
build_info=$("$binary" --build-info)
build_commit=$(jq -er '.source_commit' <<<"$build_info")
if [[ $release_line == v0.4 ]]; then
  python3 scripts/source_inputs.py check "$build_commit" --manifest "$source_manifest"
else
  [[ $build_commit == "$commit" ]]
fi
jq -e --arg commit "$build_commit" --arg version "$version" --arg feature "$(jq -er '.backend' "$binary_root/package-info.json")" '
  .source_commit == $commit and .version == $version and
  .source_tree_dirty == false and .provenance_unknown == false and
  .profile == "release" and
  (.features | type == "string" and ((split(",") | index($feature)) != null))
' <<<"$build_info" >/dev/null
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
if [[ $release_line == v0.4 ]]; then
  LEONE_BUILD_CPUSET=16-31 LEONE_RELEASE_BINARY="$release_binary" \
    scripts/package-release.sh >/dev/null
else
  LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh >/dev/null
fi
cmp "$stage/binary-first.sha256" "$binary_archive.sha256"
cmp "$stage/evidence-first.sha256" "$evidence_archive.sha256"

printf '%s candidate %s passed publication checks\n' "$version" "$commit"
