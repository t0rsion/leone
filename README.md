# Leone

Leone is a research LLM inference engine for consumer GPUs. It uses a Rust
runtime and handwritten CUDA kernels. The first target is batch-1 decode on an
NVIDIA RTX 4090.

`v0.1.0` is a research preview. It is not a general-purpose inference server.
It collects the private prototype series into one public release.

## What ships

- Dense Qwen3 and Llama 3 text inference from GGUF.
- `Q4_K` and `Q6_K` weights on the CPU and CUDA backends.
- `f32`, `f16`, and `q8` KV cache storage.
- Tiled prefill and graph-replayed CUDA decode where the model permits it.
- Greedy and stochastic sampling with an FP64 CPU oracle.
- Exact speculative verification and adaptive proposal controllers.
- Persistent, hibernated, and forked generation sessions.
- A bounded batch-1 scheduler with typed admission failures.
- An OpenAI-compatible chat subset with streaming and tool calls.
- Signed response receipts and reproducible runtime receipts.

Each kernel has a scalar reference path. Each performance claim must name a
receipt under `receipts/`.

See [docs/oracle.md](docs/oracle.md) for the evidence contract and
[docs/speculation.md](docs/speculation.md) for the correction rule.

## Scope

| Area | `v0.1.0` |
|---|---|
| Primary GPU | NVIDIA SM89 |
| Correctness target | SM89 and SM120 |
| Model families | Dense Qwen3 and Llama 3 |
| Weight formats | `Q4_K`, `Q6_K` |
| Workload | Batch-1 prefill and decode |
| Server | Local HTTP, OpenAI chat subset |
| Other vendors | Not implemented |
| Vision and MoE | Metadata probe only |

Run `leone doctor -m model.gguf` before inference. The command reports the
architecture, tensor formats, backend support, and model limits without loading
all weights.

## Install

v0.1 packages Linux x86_64 only. It requires a supported NVIDIA GPU, CUDA, and
cuBLAS. Windows and macOS packages are not available.

The GitHub release also contains a binary archive, its checksum, dynamic-linkage
output, and the evidence bundle.

## Build

You need Rust 1.92, CUDA, cuBLAS, and a supported NVIDIA GPU.

```sh
cargo +1.92 build --release -p leone-cli
./target/release/leone doctor -m model.gguf
```

The CUDA build uses the architectures selected by the build script. The release
gate tests SM89 on the local RTX 4090. SM120 is a correctness and non-regression
target, not a measured performance target.

## Generate text

```sh
./target/release/leone generate \
  -m model.gguf \
  -p 'State one invariant of exact speculative decoding.' \
  -n 64
```

Use `--backend cpu` for the scalar runtime. Use `--kv q8` to reduce KV storage.
Run `leone generate --help` for sampling and receipt options.

## Serve chat completions

```sh
./target/release/leone serve \
  -m model.gguf \
  --bind 127.0.0.1:8080 \
  --session-store .leone-sessions
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"leone","messages":[{"role":"user","content":"Define KV reuse."}]}'
```

The server binds to loopback by default. A non-loopback address requires
`--allow-remote`. The server does not provide authentication or TLS.

See [docs/openai-api.md](docs/openai-api.md) for the exact API subset.

## Evidence

Correctness gates compare the CUDA backend with independent CPU oracles.
Runtime receipts record the commit, model digest, hardware, clocks, workload,
and raw sample summary. Quality receipts bind the corpus and BF16 oracle.

```sh
taskset -c 16-31 cargo +1.92 test --workspace
taskset -c 16-31 cargo +1.92 test --release -- --ignored
```

Timing runs use physical performance cores and an otherwise idle GPU:

```sh
taskset -c 0-3,12-15 ./target/release/leone bench \
  -m model.gguf \
  --receipt
```

<!-- BEGIN GENERATED RESULTS -->
Current measured results are indexed in `receipts/INDEX.md`. Do not infer a
performance claim from an unindexed file.
<!-- END GENERATED RESULTS -->

## Limits

- The runtime executes one resident decode stream at a time.
- The scheduler interleaves bounded quanta. It does not batch matrix work.
- Prefill is tiled but remains a secondary optimization target.
- The OpenAI API is a documented subset, not a drop-in implementation.
- Response receipt self-verification proves internal consistency. Identity
  requires a match between the embedded signer and a trusted public key.
- Backward compatibility is not promised before 1.0, except for versioned
  receipt schemas and the documented server subset.

## Release gate

[docs/release.md](docs/release.md) defines the public gate. A failed gate delays
the release. It does not weaken the claim.

<!-- gate-headline:start -->
| Measurement | Leone | llama.cpp | Verdict |
|---|---:|---:|---|
| Decode tok/s at depth 512 | 172.565 | 168.317 | pass, ratio 1.025238762 |
| Mean KLD vs BF16 oracle (nats) | 0.035593219 | 0.036080760 | pass, limit 0.056080760 |

The v0.1 evidence gate is **pass**. Runtime receipts: [`2026-08-26T16:59:47Z-runtime-67f91e44.json`](receipts/2026-08-26T16:59:47Z-runtime-67f91e44.json) and [`2026-08-26T17:01:03Z-runtime-2d188d61.json`](receipts/2026-08-26T17:01:03Z-runtime-2d188d61.json). Quality receipts: [`2026-08-26T16:59:34Z-quality-4e188eb5.json`](receipts/2026-08-26T16:59:34Z-quality-4e188eb5.json) and [`2026-08-26T15:22:24Z-quality-d24c7c28.json`](receipts/2026-08-26T15:22:24Z-quality-d24c7c28.json). The full table is in [`benchmarks/README.md`](benchmarks/README.md).
<!-- gate-headline:end -->

Correctable inference passes its independent oracle, exact-output, and runtime
gates. See [`correctable-20260826T170238Z.json`](receipts/correctable-20260826T170238Z.json).

## License

The project is dual-licensed under Apache-2.0 and MIT.
