# Leone evidence archive

This archive contains generated receipts, workload manifests, corpora, execution
plans, and reproduction scripts. `MANIFEST.sha256` identifies each packaged file
by content. Each measurement records its own source and inputs.

`receipts/source-inputs.json` identifies the measured source files by SHA-256.
A checkout matching these inputs can reproduce the study without development history.

Reproduction requires the models named
by SHA-256, the pinned llama.cpp checkout, and the documented CUDA environment.
The archive does not include models, full logits, signing keys, or source crates.
The scripts run from the source checkout, not from this archive alone.

Start with `receipts/INDEX.md`. The source checkout's `docs/release.md` lists the
quality, streaming, client, and publication gates. The OpenAI client and plot
package versions are pinned in the two requirements files under `scripts/`.

`MANIFEST.sha256` covers the archive files. It checks file integrity, not publisher
identity. Response signatures require a trusted signer public key for identity.
