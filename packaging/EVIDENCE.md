# Leone evidence archive

This archive contains generated receipts, workload manifests, corpora, execution
plans, source input records, and reproduction scripts. `MANIFEST.sha256` identifies
each packaged file by content. Each measurement records its own source and inputs.

Run `scripts/verify-release-archive.sh` on the evidence archive for offline
verification. The verifier checks archive and file hashes, receipt links, source
input records, and recorded summaries. It does not require Git, models, native
runtime binaries, CUDA libraries, a GPU, or a network connection.

Offline verification checks what the archive contains. It does not reproduce a
measurement. Reproduction uses `scripts/check-release-evidence.py --mode reproduce`
from a source checkout with the models, pinned llama.cpp checkout, recorded
runtime files, and documented hardware environment. Receipts with
`<external>/...` artifact labels require those paths as explicit command-line
inputs.

The evidence archive does not include models, full logits, signing keys, source
crates, or native runtime binaries. Client records retain the SHA-256 of the
matching native runtime package. The runtime archive carries the corresponding
bytes, and release checks compare the hashes. A source input record identifies
files by SHA-256. Offline checks hash each declared source input included in the
archive. External source inputs remain declarations.

Each quality comparison record names the files its canonical producer
writes. A CUDA record names `statistics_platform`, the external
`generation_record` (a release input under `receipts/v04-*`), and the packaged
copy `generation-record.json` in its artifact root. A Metal record names the
embedded shader `leone.metal`, `leone-doctor.txt`, and `leone-eval.stdout` in its
`-metal-stage` root. The planned manifest carries no hash for the generation
record or the task manifest. A complete manifest requires both hashes, and the
dispatcher rejects a recorded artifact that is missing from the archive.

The evidence manifest defines one branching study record for each backend. It
names the study manifest,
the calibration manifest and receipt, the history reexecution pins and result, and
the quality comparison record the study binds. The planned manifest carries no
hash for a study. A complete manifest requires the binary hash of the same-backend
client, and the dispatcher rejects a study whose record digest, backend, or model
differs from the release record. Both quality sidecars of that record are
dependencies of the study.

Build the CPU-only receipt validator from the clean release checkout before
verification:

```sh
taskset -c 16-31 cargo +1.92 build --release -p leone-receipt-verify --locked
scripts/verify-release-archive.sh \
  leone-VERSION-evidence.tar.gz \
  --trusted-root "$PWD" \
  --trusted-receipt-validator target/release/leone-receipt-verify
```

The checker passes each receipt's platform, target, backend, model family, and
source manifest to that validator. It also passes each retained response receipt
to the trusted CPU tool. It runs the checkout copy after checking the archive
hash. It never runs a validator or native runtime binary copied from the archive.

The adjacent `.sha256` file and `MANIFEST.sha256` identify content. They do not
identify a publisher. Archives have no signing or notarization record.
