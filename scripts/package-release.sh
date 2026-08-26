#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
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

mkdir -p "$stage/$name/bin" "$stage/$name/docs" "$stage/$name/receipts"
mkdir -p "$stage/$evidence/receipts" dist
install -m 0755 target/release/leone "$stage/$name/bin/leone"
install -m 0755 packaging/install.sh "$stage/$name/install.sh"
cp packaging/README.md "$stage/$name/README.md"
cp packaging/compatibility.json "$stage/$name/compatibility.json"
cp docs/openai-api.md docs/release.md "$stage/$name/docs/"
cp receipts/INDEX.md "$stage/$name/receipts/"
cp receipts/INDEX.md "$stage/$evidence/receipts/"
for receipt in receipts/*.json; do
    [ -f "$receipt" ] || continue
    cp "$receipt" "$stage/$name/receipts/"
    cp "$receipt" "$stage/$evidence/receipts/"
done
for license in LICENSE*; do
    [ -f "$license" ] || continue
    cp "$license" "$stage/$name/"
done

commit=$(git rev-parse HEAD)
printf 'version=%s\ncommit=%s\n' "$version" "$commit" > "$stage/$name/BUILD-INFO"
printf 'version=%s\ncommit=%s\n' "$version" "$commit" > "$stage/$evidence/BUILD-INFO"
epoch=${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}

for artifact in "$name" "$evidence"; do
    tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" \
        -C "$stage" -cf - "$artifact" | gzip -n -9 > "dist/$artifact.tar.gz"
    (cd dist && sha256sum "$artifact.tar.gz" > "$artifact.tar.gz.sha256")
done

printf '%s\n' "$root/dist/$name.tar.gz" "$root/dist/$evidence.tar.gz"
