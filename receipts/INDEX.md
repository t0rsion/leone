# Evidence index

The checked-in evidence is linked below.
[Source inputs](./source-inputs.json) records the measured file hashes and executable modes.

- [Frozen streaming comparison](./concurrent-service-study.json): mixed requests, resident progress, memory, and all outcomes.
- [Internal batching gate](./batched-service-study.json): repeated batch-limit-one and shared-batch comparisons.
- [Qwen3 common-oracle comparison](./quality-concurrent-service.json): Leone and llama.cpp Q4 against the same BF16 reference.
- [Llama common-oracle comparison](./quality-llama-v03.json): Leone and llama.cpp Q4 against the same F16 reference.
- [OpenAI Python client workflow](./openai-client-check.json): forks, context growth, reset isolation, and recovery.
- [Metal RMSNorm profiles](./metal-rms-reduction/): paired operation measurements before and after the parallel row reduction.
  Durations include host dispatch and backend wait. Quality: unverified.

The comparison manifests link the full precision quality records:

- [Qwen3, leone_q4](./2026-09-11T18:18:33Z-quality-db8f6bb8.json)
- [Qwen3, llama_q4](./2026-09-11T18:18:52Z-quality-d76766b1.json)
- [Llama, leone_q4](./2026-09-11T18:20:26Z-quality-3e839ef9.json)
- [Llama, llama_q4](./2026-09-11T18:20:36Z-quality-b5b405af.json)

The timestamped plan-search records authenticate the checked-in SM89 plans.
`quality-qwen3-8b.json` is the historical quality reference for the earlier service study.

Model files, full logits, temporary HTTP responses, server logs, and signing
keys remain outside the public tree.

To reproduce a Metal operation profile, check out `source.git_commit` from its
runtime record. Match the recorded hardware and build settings. On Apple silicon, run:

```sh
cargo +1.92 run --locked --release -p leone-metal --example metal_profile -- \
  --repetitions 16 --receipt-dir receipts
cargo +1.92 test --locked --release -p leone-metal --test rms_oracle
```

The operation profiles do not measure end-to-end decode speed.
