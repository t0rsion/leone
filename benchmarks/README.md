# Preregistered batched-service gate

`manifest.toml` fixes the model digest, machine, CPU masks, repetition count,
workload, and pass bounds before measurement.

`single-stream-prefill-v1.toml` is the separate batch-1 manifest used by
`leone bench`. The batched-service manifest remains unchanged.

## Workload

- Start four simultaneous non-streaming requests.
- Emit at most 64 tokens for each request.
- Run one server with a batch limit of four.
- Run the baseline server with a batch limit of one.
- Repeat the comparison five times.

Both servers accept the same concurrent request set. The batch limit changes;
the prompt, sampling settings, session count, model, and execution plan do not.

## Procedure

Build and test with `taskset -c 16-31`. Pin timings with
`taskset -c 0-3,12-15`. No other compute process may use the GPU.

```sh
study="receipts/$(date -u +%Y-%m-%dT%H-%M-%SZ)-batched-service-study.json"
taskset -c 0-3,12-15 scripts/study-batched-service.sh \
  models/Qwen3-8B-Q4_K_M.gguf \
  plans/qwen3-8b-sm89.json \
  "$(jq -r '.executions.leone_q4.quality.receipt' receipts/quality-concurrent-service.json)" \
  "$study"
```

The script checks that the quality record names the served model digest. It
exits with status 2, before any repetition, when the output path exists. After
a passing study it stores one aggregate JSON file, which keeps the run records,
and removes temporary responses and logs. After a failure it keeps the completed run
records, raw responses, and server logs in `OUTPUT.runs.*`. A receipt that
fails the gate stays at its path. The script still runs every repetition, then
exits with status 1.

The script reads `LEONE_BINARY` (default `target/release/leone`). It exits with
status 2 unless `--build-info` reports a clean release CUDA build whose source
inputs equal the current tree. The receipt records the binary SHA-256, the GPU
identity, the prompt SHA-256, the batch limit passed to each server, and the five
run records. Dispatch widths are unmeasured. Run records keep response digests,
not signed responses.

## Gate

Every repetition must satisfy all four conditions:

- Batched transcripts match the batch-limit-one transcripts.
- Aggregate completion throughput ratio is greater than `1.00`.
- P95 completion latency ratio is less than `1.00`.
- A new request completes after a forced client disconnect.

The result applies to one model, one GPU, four clients, and one deterministic
prompt. It does not establish performance for other workloads.

## Branching study manifests

`branching-service-calibration.json` and `branching-service-frozen.json` declare
the CUDA branching study. The `-metal` pair declares the Metal study. Each frozen
manifest names the quality policy `canonical_v2_common_oracle`. Neither carries a
quality tolerance. Store study receipts under `receipts/`. Run them with
`scripts/study-branching-service.sh`.
