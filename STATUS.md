# Status

## v0.2.0

The local candidate includes Qwen and Llama execution plans, independent
quality receipts, a live scheduled-server study, and chunked Q8 prefill. Its
exact-source publication gate passes on the local RTX 4090.

The v0.2 release follows the public v0.1 line. Local research branches do not
define the public release history.

## Release claim

Leone runs dense Qwen3 and Llama 3 text models from GGUF on its CPU and CUDA
backends. The measured target is batch-1 decode on SM89. The server provides
the subset in `docs/openai-api.md`.

The release is ready only when every command in `docs/release.md` passes on the
candidate commit. Runtime and quality claims require current receipts.

## Known limits

- NVIDIA is the only accelerated backend.
- Vision, MoE, and hybrid architectures do not execute.
- The scheduler interleaves bounded quanta. It does not batch matrix work.
- Authentication, TLS, distributed execution, and multi-GPU execution are out
  of scope.
- Performance work without a preregistered study and generated receipt does not
  enter the release claim.
