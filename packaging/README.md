# Leone binary archive

The release includes these runtime archives:

| Archive | Backend | Runtime requirement |
|---|---|---|
| `leone-VERSION-linux-x86_64.tar.gz` | CUDA | NVIDIA CUDA and cuBLAS |
| `leone-VERSION-darwin-arm64.tar.gz` | Metal | Apple silicon and macOS |

The Darwin archive targets Apple silicon. Intel macOS and Windows packages are
outside this release. The archive carries `package-info.json` and
`environment.json` with its target, backend, toolchain, path remaps, and archive
timestamp.

Packaging resolves `cargo` from `PATH`. Set `CARGO` or `LEONE_CARGO` to an
executable path when a noninteractive environment does not expose the tool.

Before inference, run `bin/leone doctor -m <model.gguf>`.

Install for one user with:

```sh
./install.sh
```

Set `PREFIX` to choose another destination. The default is `$HOME/.local`.
The installer copies the binary, plans, and receipts. It does not change shell
startup files.

Each runtime archive contains `MANIFEST.sha256`. Verify it with the portable
helper under `tools/checksum.sh`. From a source checkout, verify an archive and
its extracted manifest with:

```sh
scripts/verify-release-archive.sh leone-VERSION-linux-x86_64.tar.gz
```

Evidence archives, when provided, use the same verifier. Build the
CPU-only `leone-receipt-verify` tool from the trusted source checkout and pass
it with `--trusted-receipt-validator`. Offline verification checks bundled
content hashes, receipt links, source input records, response signatures, and
receipt structure. It does not compile code, load models, query CUDA, or rerun a
study.

Use `scripts/check-release-evidence.py --mode reproduce` from a source checkout
when local models, binaries, the pinned llama.cpp tree, and the recorded driver
libraries are available. That mode checks local inputs against recorded hashes.
When a receipt uses `<external>/...` labels, pass the matching paths with
`--model`, `--plan`, `--leone-binary`, and `--llama-binary`.

Archives are unsigned and have no notarization record. `MANIFEST.sha256` and
the adjacent `.sha256` file identify content. They do not identify a publisher.

The Linux archive includes proof-gated CUDA plans under `plans/`. The Darwin
archive omits those plans because they target NVIDIA SM89. A plan works only
with its exact model SHA-256 and tested compute capability.
