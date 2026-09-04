#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
path_remaps="--remap-path-prefix=$root=/source/leone"
path_remaps="$path_remaps --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo"
path_remaps="$path_remaps --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=/rustup"
release_rustflags="${RUSTFLAGS:+$RUSTFLAGS }$path_remaps"

cd "$root"
mkdir -p dist/pypi
if [ -n "${LEONE_BUILD_CPUSET:-}" ]; then
    RUSTUP_TOOLCHAIN=1.92 RUSTFLAGS="$release_rustflags" \
        taskset -c "$LEONE_BUILD_CPUSET" maturin build --release --locked --out dist/pypi
else
    RUSTUP_TOOLCHAIN=1.92 RUSTFLAGS="$release_rustflags" \
        maturin build --release --locked --out dist/pypi
fi
