#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -1)

if [ -z "$version" ]; then
    echo "error: workspace version is missing" >&2
    exit 1
fi

case "$(uname -s):$(uname -m)" in
    Linux:x86_64) platform=linux-x86_64 ;;
    *)
        echo "error: release packaging supports Linux x86_64 only" >&2
        exit 1
        ;;
esac

name="leone-$version-$platform"
evidence="leone-$version-evidence"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT HUP INT TERM
path_remaps="--remap-path-prefix=$root=/source/leone"
path_remaps="$path_remaps --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo"
path_remaps="$path_remaps --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=/rustup"
release_rustflags="${RUSTFLAGS:+$RUSTFLAGS }$path_remaps"

cd "$root"
if [ -n "${LEONE_BUILD_CPUSET:-}" ]; then
    RUSTFLAGS="$release_rustflags" taskset -c "$LEONE_BUILD_CPUSET" \
        cargo +1.92 build --release -p leone-cli --locked
else
    RUSTFLAGS="$release_rustflags" cargo +1.92 build --release -p leone-cli --locked
fi

mkdir -p "$stage/$name/bin" "$stage/$name/docs" "$stage/$name/plans" "$stage/$name/receipts"
mkdir -p "$stage/$evidence/plans" "$stage/$evidence/receipts" "$stage/$evidence/research/oracle" "$stage/$evidence/scripts" "$stage/$evidence/benchmarks" "$stage/$evidence/corpus" dist
install -m 0755 target/release/leone "$stage/$name/bin/leone"
install -m 0755 packaging/install.sh "$stage/$name/install.sh"
cp packaging/README.md "$stage/$name/README.md"
cp packaging/compatibility.json "$stage/$name/compatibility.json"
cp deny.toml "$stage/$evidence/deny.toml"
cp packaging/EVIDENCE.md "$stage/$evidence/README.md"
cp docs/openai-api.md docs/release.md docs/release-evidence.md \
    docs/release-candidate.md docs/client-workflow.md docs/memory-accounting.md docs/concurrent-service-evidence.md docs/concurrent-service.svg "$stage/$name/docs/"
cp plans/*.json "$stage/$name/plans/"
cp plans/*.json "$stage/$evidence/plans/"
cp receipts/INDEX.md "$stage/$name/receipts/"
cp receipts/INDEX.md "$stage/$evidence/receipts/"
for receipt in receipts/*.json; do
    [ -f "$receipt" ] || continue
    cp "$receipt" "$stage/$name/receipts/"
    cp "$receipt" "$stage/$evidence/receipts/"
done
cp research/oracle/llama_logits.cpp "$stage/$evidence/research/oracle/"
cp scripts/build-llama-oracle.sh scripts/run-llama-oracle.sh \
    scripts/study-live-server.sh scripts/study-batched-service.sh \
    scripts/study-concurrent-service.py scripts/study-concurrent-service.sh \
    scripts/quality-concurrent-service.sh scripts/check-openai-client.py \
    scripts/check-openai-client.sh scripts/client-requirements.txt \
    scripts/check-release-evidence.py scripts/source_inputs.py scripts/render-concurrent-evidence.py scripts/plot-concurrent-evidence.py \
    scripts/plot-requirements.txt \
    scripts/render-release-evidence.sh "$stage/$evidence/scripts/"
cp benchmarks/concurrent-service-*.json "$stage/$evidence/benchmarks/"
cp corpus/*.txt corpus/SHA256SUMS "$stage/$evidence/corpus/"
cp scripts/check-public-tree.sh scripts/check-public-tree.py \
    scripts/test-check-public-tree.sh scripts/check-rust-complexity.sh \
    scripts/check-server-errors.sh scripts/check-release.sh \
    "$stage/$evidence/scripts/"
for license in LICENSE*; do
    [ -f "$license" ] || continue
    cp "$license" "$stage/$name/"
    cp "$license" "$stage/$evidence/"
done

epoch=${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}

for artifact in "$name" "$evidence"; do
    manifest="$stage/$artifact.MANIFEST.sha256"
    (
        cd "$stage/$artifact"
        find . -type f ! -name MANIFEST.sha256 -print0 | LC_ALL=C sort -z | \
            xargs -0 sha256sum >"$manifest"
    )
    mv "$manifest" "$stage/$artifact/MANIFEST.sha256"
    tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" \
        -C "$stage" -cf - "$artifact" | gzip -n -9 > "dist/$artifact.tar.gz"
    (cd dist && sha256sum "$artifact.tar.gz" > "$artifact.tar.gz.sha256")
done

printf '%s\n' "$root/dist/$name.tar.gz" "$root/dist/$evidence.tar.gz"
