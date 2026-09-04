# Release gate

Run this gate on the exact candidate commit. Keep the GPU idle except for the
command under test.

## Static and CPU gates

```sh
scripts/check-public-tree.sh
taskset -c 16-31 cargo +1.92 test --workspace
cargo +1.92 clippy --workspace --all-targets -- -D warnings
cargo +1.92 fmt --all --check
scripts/check-rust-complexity.sh
cargo +1.92 deny check
RUSTDOCFLAGS='-D warnings' cargo +1.92 doc --workspace --no-deps --lib
```

## GPU gates

```sh
taskset -c 16-31 cargo +1.92 test --release -- --ignored --test-threads=1
```

Run the model-specific session, scheduler, fork, hibernation, and correctable
verification commands for each claimed model family. Record the model SHA-256
and candidate commit.

Use `--exact-only` for fork and hibernation checks on secondary model families.
Apply a speed gate only to the stated model and idle hardware target.

## Measurement gate

Pin timing studies to physical performance cores:

```sh
taskset -c 0-3,12-15 ./target/release/leone bench \
  --prefill-context 4096 \
  --prefill-chunk 2048 \
  --receipt
```

A runtime receipt must reference a quality receipt for the same model and
workload. Otherwise the result states `quality: unverified`.

Never edit a recorded measurement. Replace an invalid run with a new receipt.
Retain the failed candidate record in the evidence archive.

## Archive gate

```sh
LEONE_BUILD_CPUSET=16-31 scripts/package-release.sh
```

Extract the archive into an empty directory. From that directory, run
`bin/leone` and `bin/leone doctor -m <model.gguf>`. Compare the
archive with its SHA-256 file.

From each extracted archive root, run `sha256sum -c MANIFEST.sha256`.
Reject a binary with a workspace path, runtime search path, or unresolved
library. Build both archives twice from the same commit. Their checksums must
match.

Tag the tested commit as `v<version>`. The release workflow builds the Linux
x86_64 CUDA archive and creates the GitHub release.

## Decision

Ship only if every applicable gate passes. If a gate fails, fix the release
or narrow the stated scope before rerunning the full gate.
