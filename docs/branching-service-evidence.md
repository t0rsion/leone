# Branching service evidence

The branching study compares Leone session forks with llama.cpp slot copies
in one server workload on CUDA and Metal.

All commands go through `scripts/study-branching-service.sh`. Paths are relative
to the repository root. No receipt from this workflow exists until you run it.

## What the receipt claims

- Both engines complete the frozen prompts, and each supported service metric stays
  inside the bound derived from the service calibration run.
- Both engines have a row in one comparison record against one BF16 or F16 oracle
  that Leone does not produce. The record covers one model artifact and one token
  stream. The manifest declares the policy `canonical_v2_common_oracle` before any
  run: accept the record after canonical recomputation, and report both rows.
- The receipt binds the record by path and SHA-256, and states its own limits.

Limits:

- The policy sets no KLD or top-1 bound. The KLD, top-1 agreement, and needle
  argmax of each row are measured results in the record. A llama.cpp row that
  differs from the Leone row is a result, not a failure.
- The study harness does not recompute quality. It checks the record structure,
  the model and corpus joins, and the sidecar bytes. The receipt carries
  `quality_recomputation_status: "not_run_by_harness"`. The `check-quality` stage
  recomputes KLD, top-1 agreement, and the needle argmax from retained logits. It
  runs before `freeze`. Release validation (`quality-stage-v1`) must also pass for
  the same record digest.
- The quality record is a common-model observation from the evaluation path. It is
  not a measurement of the served schedule. The receipt carries
  `quality_path_status: "eval_path_not_served_path"`.
- Leone quality and Leone serving use the same native executable bytes. That does
  not show they run the same code path. The statistics producer of the record
  runs on its own target and is not compared with the server.
- llama.cpp quality comes from the `llama_logits` adapter. The server is
  `llama-server`. The two executables differ. The receipt joins them by
  `external/PINNED` and by the SHA-256 of `libllama`, `libggml`, `libggml-base`,
  `libggml-cpu`, and the active backend library.
- The library hashes come from `ldd` (Linux) or a walk over `otool -L` (Darwin).
  They describe resolved files. They are not a map of the running process
  (`loaded_library_status: "resolved_linkage_not_process_map"`).
- The oracle shares the pinned llama.cpp implementation with the peer.
- Tokenizing history the same way in both engines does not show that a branch
  computes the same numbers as an isolated run.
- The stream, lifecycle, memory, cancel, sibling, and overlap checks are separate
  gates. The quality record does not replace them.
- Each engine row declares the same independent llama.cpp tokenizer binary, model,
  legacy template, and source commit. Leone history evidence uses that tokenizer
  identity while its signed response remains bound to the Leone serving process.

## Inputs

You need these before the calibration run:

- The model at the path the manifest names (`models/Qwen3-8B-Q4_K_M.gguf`).
- On CUDA, the plan the manifest names.
- One comparison record from the canonical collectors. CUDA records use
  `leone.quality-comparison.v2`. Metal records use `leone.quality-cross-device.v2`.
  The record carries the long-context task and its retained samples.
- A `leone-receipt-verify` binary that you build and pass on the command line. The
  workflow never runs a path it reads from a receipt.

The workflow does not rerun the quality collectors. Collect the record once with
`scripts/quality-concurrent-service.sh` (CUDA) or `scripts/compare-metal-quality.sh`
(Metal), then pass it in.

The wrapper passes Leone an explicit signing-key path. Set
`LEONE_BRANCHING_SIGNING_KEY` to a private 32-byte Ed25519 seed when calibration
and frozen runs must share one key. Without it, each run creates a private key in
its temporary work directory. The wrapper derives the public key from that file;
the harness checks the derived key against every signed history receipt. It never
accepts the response public key as the trust configuration.

## Manifests

| Backend | Calibration | Frozen template |
|---|---|---|
| CUDA | `benchmarks/branching-service-calibration.json` | `benchmarks/branching-service-frozen.json` |
| Metal | `benchmarks/branching-service-calibration-metal.json` | `benchmarks/branching-service-frozen-metal.json` |

The frozen templates carry no thresholds. `freeze` writes a new manifest and
refuses to overwrite one.

## Run

1. Build both servers.

   ```sh
   scripts/study-branching-service.sh build-leone cuda
   scripts/study-branching-service.sh build-llama cuda
   ```

   Use `metal` for both on the Mac. `build-llama metal` needs the checkout in
   `external/llama.cpp` to sit at the commit in `external/PINNED`.

2. Check the comparison record by canonical recomputation. The flags after the
   verifier are the ones `scripts/validate-quality-stage.py comparison` takes.

   ```sh
   scripts/study-branching-service.sh check-quality RECORD target/release/leone-receipt-verify \
     --source-commit COMMIT --backend BACKEND ...
   ```

   The stage prints `canonical quality check passed` and the record digest, or
   fails with the validator error.

3. Record the source inputs, then bind them to the runs.

   ```sh
   scripts/study-branching-service.sh source-manifest COMMIT receipts/branching-source.json
   export LEONE_BRANCHING_SOURCE_MANIFEST=receipts/branching-source.json
   ```

4. Run the calibration study. The script starts the Leone server and the harness
   starts `llama-server`.

   ```sh
   scripts/study-branching-service.sh calibrate benchmarks/branching-service-calibration.json \
     receipts/branching-calibration.json
   ```

5. Freeze. The stage repeats the `check-quality` step, then writes a new manifest.
   Every service threshold derives from the calibration receipt by a fixed rule.
   The quality record is bound by path and SHA-256 under the policy the template
   declares. The stage derives no quality bound and refuses to overwrite a file.

   ```sh
   scripts/study-branching-service.sh freeze benchmarks/branching-service-frozen.json \
     receipts/branching-calibration.json RECORD benchmarks/branching-service-frozen-run.json \
     target/release/leone-receipt-verify --source-commit COMMIT --backend BACKEND ...
   ```

6. Run the frozen study and validate the receipt.

   ```sh
   scripts/study-branching-service.sh frozen benchmarks/branching-service-frozen-run.json \
     receipts/branching-frozen.json
   scripts/study-branching-service.sh validate receipts/branching-frozen.json
   ```

To validate from a release archive, which omits `crates`, pass the scope:
`validate RECEIPT archive`. Set `LEONE_BRANCHING_SOURCE_MANIFEST` first. The
wrapper rejects archive validation without that binding. The archive check
needs the harness, the history-tokenization scripts, `scripts/linked_libraries.py`,
`scripts/source_inputs.py`, and `fixtures/qwen3-legacy-chatml.jinja`.

## Server limits

The probe schedule opens five connections from one client address. The Leone default
of four per client rejects the fifth. The script starts Leone with `--sessions 8`,
`--max-connections 16`, and `--max-connections-per-client 8`. Set
`LEONE_BRANCHING_SESSIONS`, `LEONE_BRANCHING_MAX_CONNECTIONS`, and
`LEONE_BRANCHING_CLIENT_CONNECTIONS` to change them. The harness does not observe
these values, so the receipt does not record them.

## Metal

Leone runs on Metal with `--backend metal --batch-size 1`. No Metal plan exists.
The Metal manifests pass `--device MTL0` to `llama-server`. No native run has
verified that device name. Check it against `llama-server --list-devices` on the Mac
before you trust a Metal receipt.

## Retained results

Both rows of the comparison record stay in the record. A row that shows lower
quality than the other is reported as measured. Nothing deletes it, and no speed
claim from the study drops the model, the task, or this quality scope.
