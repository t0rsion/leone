# Preregistered batched-service gate

`manifest.toml` fixes the model digest, machine, CPU masks, repetition count,
workload, and pass bounds before measurement.

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
taskset -c 0-3,12-15 scripts/study-batched-service.sh \
  models/Qwen3-8B-Q4_K_M.gguf \
  plans/qwen3-8b-sm89.json \
  receipts/quality-qwen3-8b.json \
  receipts/batched-service-study.json
```

The script checks that the quality record names the served model digest. It
stores one aggregate JSON file and removes temporary responses and logs.

## Gate

Every repetition must satisfy all four conditions:

- Batched transcripts match the batch-limit-one transcripts.
- Aggregate completion throughput ratio is greater than `1.00`.
- P95 completion latency ratio is less than `1.00`.
- A new request completes after a forced client disconnect.

The result applies to one model, one GPU, four clients, and one deterministic
prompt. It does not establish performance for other workloads.
