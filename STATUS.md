# Status

## Local release candidate

The candidate adds continuous decode batching, resumable prefill, physical
allocation accounting, page-rounded KV admission, and typed overload responses.
Qwen3 and Llama batch differentials compare each request with isolated execution. The live service study compares the same
concurrent request set with batch limits greater than one and equal to one.

The candidate stays local until every command in [docs/release.md](docs/release.md)
passes on the final source tree.

## Release claim

Leone runs dense Qwen3 and Llama 3 text models from GGUF on CPU and CUDA. The
measured target is SM89. The server provides the subset in
[docs/openai-api.md](docs/openai-api.md).

## Known limits

- NVIDIA is the only accelerated backend.
- Vision, MoE, and hybrid architectures do not execute.
- KV pages bound admission but do not remap physical KV storage.
- Explicit speculation runs outside the shared request batch.
- Authentication, TLS, distributed execution, and multi-GPU execution are out
  of scope.
