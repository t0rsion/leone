# LLM inference engines, August 2026

*A survey for someone building a new research inference engine targeting consumer hardware.*

Compiled 2026-08-18. Two independent research tracks were used and cross-checked: (A) direct web retrieval of repositories, release feeds, and documentation; (B) an offline architectural analysis with no network access, which contributed reasoning rather than facts. Where the two disagree, the web evidence wins and the disagreement is noted. Those disagreements show which pieces of 2024-25 conventional wisdom have expired.

---

## 1. Summary

**The consolidation is over.** Text Generation Inference was archived on 2026-03-21 and its README now points users at vLLM, SGLang, llama.cpp, or MLX. DeepSpeed-MII has had no commit since 2025-06-30. ipex-llm was archived with a security notice. AutoGPTQ and AutoAWQ are both archived. ExLlamaV2 is archived in favor of ExLlamaV3. MLC-LLM's last six months of commits are TVM-refactor plumbing with no feature work. The field narrowed to six live engines: llama.cpp/ggml, vLLM, SGLang, TensorRT-LLM, MLX, and ExLlamaV3, plus FlashInfer underneath three of them.

**The compile-versus-capture argument flipped.** SGLang shipped *Breakable CUDA Graph*, which captures graphs with explicit eager breaks via an `@eager_on_graph` decorator and no compiler at all. On gpt-oss-120b prefill across 4×GB300: full capture 1.93× over eager, BCG 1.70×, `torch.compile` piecewise 1.45×, and BCG builds graphs 3.8-5.2× faster, because "torch.compile accounts for 78-86% of the setup time." It needs about a quarter of the code. vLLM went the opposite way, hardening `-O2` (`FULL_AND_PIECEWISE`) as the default and adding JIT warmup to hide first-request stalls. For a research engine with fast iteration and short-lived processes, the capture-with-breaks approach is the better bet.

**Trellis/vector quantization crossed from research into production on both CPU and GPU simultaneously.** ExLlamaV3 (2026-07-14) ships QTIP-derived tail-biting trellises with a fused Viterbi quantizer kernel, hours on one 4090 for a 70B model, against AQLM's ~720 A100-hours. Independently, ik_llama.cpp shipped `IQ1_KT`/`IQ2_KT`/`IQ3_KT`/`IQ4_KT`, integer-base trellises designed for CPU decode speed. The generic "invent a better low-bit codebook" opportunity is now closed.

**MLX stopped being an Apple project.** MLX 0.32.1 has a working CUDA backend with quantized matmul/matvec kernels for SM80, cuDNN SDPA with attention sinks, CUDA Hadamard transforms, FSDP, and Windows support. Meanwhile Modular's MAX extended Apple Silicon GPU serving back to M1. The "portable low-bit kernel IR" gap is being closed from two directions at once.

**Ollama reversed course.** Ollama did *not* replace llama.cpp with a Go engine. Its `runner/runner.go` now accepts exactly one engine flag (`--mlx-engine`); the Go GGML runner path was **removed**. GGUF inference is delegated to upstream `llama-server` as a subprocess, pinned by a repo-root `LLAMA_CPP_VERSION` file (currently `b10434`) with a `llama/compat/` patch layer described in-tree as "temporary… during the transition to llama-server." Ollama's own engine now exists only for Apple Silicon, written in Go against MLX.

### Implication

At batch 1-4 on one consumer GPU, decode is a weight-bandwidth problem and prefill is a compute/attention problem. The datacenter engines are optimizing a different objective and their scheduler sophistication is unused at batch one. The open opportunity is a **quantization/runtime co-design compiler**: one checkpoint that carries or cheaply derives both a GEMV-shaped decode layout and a tensor-core-shaped prefill layout, with bit allocation chosen by *measured kernel latency* rather than bits-per-weight, verified by an evaluation harness that reports crossover surfaces rather than single winning points.

---

## 2. A quantitative anchor: the roofline efficiency curve

Before the survey, one measurement that anchors the rest. llama.cpp's community CUDA and Apple scoreboards report Llama-2-7B Q4_0 `tg128` (batch-1 decode). The GGUF is 3.83 GB. Dividing device memory bandwidth by that gives an absolute upper bound on tokens/sec, since every weight must be read once per token.

| Device | BW (GB/s) | Roofline tok/s | Measured tok/s | Efficiency |
|---|---:|---:|---:|---:|
| M1 (8-core) | 68 | 17.8 | 14.2 | **80%** |
| M4 (10-core) | 120 | 31.3 | 24.1 | **77%** |
| RTX 4080 | 717 | 187 | 142.5 | **76%** |
| RTX 5080 | 960 | 251 | 182.0 | **73%** |
| M4 Pro (20-core) | 273 | 71 | 50.7 | **71%** |
| RTX 4090 | 1008 | 263 | 186.2 | **71%** |
| RTX 3080 | 760 | 198 | 139.7 | **70%** |
| RTX 3090 | 936 | 244 | 158.2 | **65%** |
| M3 Max (40-core) | 400 | 104 | 66.3 | **64%** |
| RTX 5090 | 1792 | 468 | 290.0 | **62%** |
| M4 Max (40-core) | 546 | 143 | 83.1 | **58%** |
| M2 Ultra | 800 | 209 | 94.3 | **45%** |
| M3 Ultra | 819 | 214 | 92.1 | **43%** |
| A100 80GB | 2039 | 532 | 190.9 | **36%** |
| H100 SXM | 3350 | 875 | 267.8 | **31%** |

*(Community-submitted, mixed driver/clock conditions, flash attention off. Treat as a trend, not a leaderboard.)*

The trend is monotone and it is the most useful single fact in this document: **roofline efficiency falls as bandwidth rises.** Small chips hit 76-80%. Mid consumer GPUs hit 70-76%. A 5090 gets 62%. An H100, nearly twice a 5090's bandwidth, delivers *less absolute throughput than a 5090* at batch one.

The mechanism is the wave problem. Take a 4096×4096 W4 layer: 8 MiB of nibbles plus ~256 KB of FP16 scales, about 8.7 MB moved, 33.6 MFLOP performed, roughly 3.8 FLOP/byte. If a CTA covers 128 output rows, that is 32 CTAs on a 128-SM 4090: one quarter of a single wave, with 96 SMs idle. Saturating DRAM needs enough independent in-flight loads to cover hundreds of cycles of latency. Bigger GPUs have more SMs and more latency to hide, and a GEMV does not supply enough parallelism to fill them; split-K and narrower row tiles buy some of it back at the cost of reduction traffic.

Two consequences for a consumer engine. First, the 4090/5090 class is the efficient range for batch-one work, not a compromise. You get a higher fraction of your hardware than a datacenter card does. Second, the remaining 25-40% is real, addressable engineering: CTA count, tail tiles, scale traffic, kernel gaps, split-K reduction, and L2/TLB behavior. An engine that reliably reached 85% of roofline on a 4090 would be roughly 20% faster than llama.cpp today, which is a publishable result on its own.

---

## 3. The llama.cpp / ggml ecosystem

### 3.1 llama.cpp itself

C/C++ with per-ISA SIMD and a `ggml-backend` abstraction separating devices, buffer types, allocation, graph execution, cross-backend copies, and graph partitioning. Nightly build `b10488` as of 2026-08-18; 124.5k stars. Semantic versioning started the same day and is described upstream as work in progress.

**Packaging changed.** There is now a `llama.app` site and a single `llama` binary with git-style subcommands: `serve`, `cli`, `completion`, `bench`, `batched-bench`, `fit-params`, `quantize`, `perplexity`, `download`, `update`. Tools moved to `tools/{cli,completion,server,...}`.

**The KV API was rewritten.** `llama_kv_self_*` is gone; the public surface is `llama_memory_t` with `llama_memory_seq_rm/cp/keep/add/div` and `llama_memory_seq_pos_min/max`. Unified KV is a context parameter (`kv_unified`) and a server flag `-kvu`, enabled by default when slot count is automatic. `llama_context_params` also gained `n_rs_seq` (recurrent-state snapshots for rollback), `ctx_type` (e.g. MTP), and an **in-context backend sampler chain** (`samplers`, `n_samplers`): device-side sampling, exposed as `-bs/--backend-sampling`.

Important nuance: **`--kv-unified` is not PagedAttention.** It is a shared arena of KV cells with position and sequence ownership metadata, not a logical-page-to-physical-frame translation with block tables. Sequences share or copy state, unused cells are reclaimed, the cache can be shifted or defragmented, and attention receives slot/position information plus a mask. This is a different design point and worth studying on its own terms.

**Graph reuse is pervasive and real.** `src/llama-graph.h` defines `llm_graph_result::update()` and `can_reuse(const llm_graph_params&)` overrides on roughly eighteen input classes plus `allow_reuse()` topology matching. If the ubatch shape matches, only input tensors get re-pointed and the graph relaunches. CUDA graphs sit underneath with `cudaGraphExecUpdate` fast-path re-instantiation keyed by a `graph_key`.

**MoE and placement.** `-cmoe/--cpu-moe` puts all expert weights on CPU; `-ncmoe/--n-cpu-moe N` does the first N layers; `-ot/--override-tensor <regex>=<buffer type>` gives arbitrary per-tensor placement. Draft-model variants exist for all three.

**Auto-fit is new and matters for consumer hardware.** `-fit on|off` defaults to *on* and auto-adjusts unset arguments to fit device memory, with `-fitt` (per-device MiB margin, default 1024) and `-fitc` (minimum context, default 4096). `-ngl` now accepts `auto`/`all` and defaults to `auto`. There is a standalone `llama-fit-params` tool.

**Multi-GPU changed shape.** `-sm {none,layer,row,tensor}`: `layer` is default (pipeline parallel), **`row` is deprecated**, and **`tensor` is new and experimental**, true tensor parallelism splitting weights *and* KV via a "meta device," requiring `-fa on` and unquantized KV, incompatible with `--fit`, and unimplemented for a long list of architectures (Grok, DeepSeek2, GLM-DSA, Nemotron-H, Jamba, Mamba/Mamba2, RWKV, BitNet, T5, Gemma-3n).

**Speculative decoding was rewritten.** `--spec-type` takes a comma-separated list of `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`. The old `--draft`/`--draft-max` are gone. EAGLE-3 checkpoints import from SpecForge, vLLM, and AngelSlim via `convert_hf_to_gguf.py --target-model-dir`. `ngram-mod`, a roughly 16 MB rolling-hash pool shared across all server slots, is the most interesting draft-free trick here for consumer use, because it costs almost nothing and needs no second model in VRAM. `/metrics` now reports `llamacpp:spec_decode_num_draft_tokens_total`, `..._accepted_tokens_total`, and per-position acceptance, which is exactly the telemetry an adaptive speculation policy would need.

**Caching tiers.** `--cache-prompt` (on), `-cram/--cache-ram` (8192 MiB host-RAM tier), `--cache-reuse N` (KV-shifting-based partial prefix reuse), `--cache-idle-slots`. Context shift is now **off** by default. SWA uses a windowed cache by default (`--swa-full` to disable) with context checkpoints (`-ctxcp`, default 32).

**Server surface, 2026.** OpenAI chat/responses/embeddings, **Anthropic Messages API**, rerank, infill. A **router server** for multi-model serving (`--models-dir`, `--models-max` default 4, `--models-autoload`). **Agent mode** (`-ag`, `--tools read_file,file_glob_search,grep_search,exec_shell_command,write_file,edit_file`, `--tools-runtime docker:/podman:/ssh:`) and MCP client support. `--jinja` is on by default. Reasoning controls (`--reasoning-effort` up to `xhigh`/`max`, `--reasoning-budget`). A new **adaptive-p** sampler.

**Quantization types.** `GGML_TYPE_COUNT = 43`. Beyond the familiar Q4_0…Q8_0, Q2_K…Q6_K, and IQ1/IQ2/IQ3/IQ4 families: **TQ1_0 (1.69 bpw)** and **TQ2_0 (2.06 bpw)** ternary, **MXFP4 (39)**, **NVFP4 (40)** with E4M3 scales, **Q1_0 (1.125 bpw)**, **Q2_0 (2.25 bpw, group 64)**, and `MXFP4_MOE` as a quantize target. The ISA-specific repacked types (`Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8`, `IQ4_NL_4_4/4_8/8_8`) were **removed from GGUF files** and replaced by runtime repacking (`--repack`, on by default). KV cache types: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`.

One gap worth flagging: `LLAMA_FTYPE_MOSTLY_NVFP4` exists as a type, but NVFP4 is not in `llama-quantize`'s option list nor in `convert_hf_to_gguf.py --outtype`. The upstream *producer* path for NVFP4 GGUFs is unclear even though community NVFP4 GGUFs circulate.

**Backends.** BLAS, BLIS, CANN, CUDA, HIP, Hexagon (in progress), IBM zDNN, MUSA, Metal, OpenCL (Adreno), OpenVINO (in progress), RPC, SYCL, VirtGPU/APIR, Vulkan, WebGPU, ZenDNN. Multiple backends coexist in one binary with runtime `--device` selection. The generated `docs/ops.md` matrix shows CPU and CUDA complete, Vulkan and SYCL close behind. Practical ranking on the same hardware (Ryzen AI Max+ 395): **ROCm is roughly 2× Vulkan** for decode (94.7-98.5 vs 48.8-50.2 tok/s) with clearly better TTFT.

**Weaknesses.** Quant kernel quality varies enormously across CPU/CUDA/Metal/Vulkan/SYCL for the same nominal format. Continuous batching is real but far less sophisticated than vLLM/SGLang. Tensor parallelism is experimental and architecture-limited. New architectures land first as unoptimized operator compositions. Build-number releases make reproducibility awkward.

### 3.2 ik_llama.cpp

Ikawrakow's fork, diverged from upstream since August 2024 and no longer tracking it. Active: 3,056 stars, pushed 2026-08-15. **CPU (AVX2+/NEON+) and CUDA (Turing+) only**. ROCm/Vulkan/Metal issues are explicitly not accepted.

The value here is that it is the best available laboratory for "how much performance is hidden in the exact packing of 2-5 bit weights."

**Quant types** use IDs above 100 to avoid upstream collisions: Q6_0, IQ1_BN, IQ2_BN, IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K, IQ4_KS, IQ2_KS, IQ4_KSS, Q8_KV, IQ5_KS, IQ3_KS, IQ2_KL, and the **trellis family IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT** built on an integer-base trellis chosen for CPU decode throughput. There is a whole parallel space of **row-interleaved `_R4`/`_R8` repacked variants** (IDs 200-399): `Q4_0_R8`, `Q8_0_R8`, `Q2_K_R4` … `Q6_K_R4`, `IQ4_XS_R8`, `MXFP4_R8`, `Q8_K_R16`, which is precisely the "same numeric format, different execution layout" idea a new engine should generalize.

**Distinctive flags.** `-mla/--mla-use {0..3}` default 3 (MLA for DeepSeek-class models), `-fmoe` fused MoE (on by default), `-rtr` runtime repacking, `-ser/--smart-expert-reduction Kmin,t` (described in-tree as "basically REAP from the command line"), `-amb/--attention-max-batch`, `-gr/--graph-reuse`, `-sm graph` (exclusive: splits tensors *and* the compute graph across heterogeneous GPUs) with `-grt/--graph-reduce-type {q8_0,bf16,f16,f32}`, `-op/--offload-policy` for per-`ggml_op` GPU offload control, `-khad`/`-vhad` Hadamard-transformed K/V cache, `--defer-experts` (lazy expert mmap fault-in), `--custom-q "regex=type,..."` with `--dry-run`, and `llama-sweep-bench`. It also has FlashMLA-3 on CPU and CUDA, fused delta-net (shipped before upstream), MTP, DFlash, and DSpark.

**A caution the README itself raises:** do not use `-rtr` for hybrid CPU/GPU MoE, because repacked k-quants have no CUDA row-interleaved kernels and those matmuls get pinned to CPU.

**A caution the community measured.** Qwen3.6-27B on an RTX 3090, full offload: `iq4_ks` at 14.68 GiB gives PPL 3.1341 and 40.68 tok/s decode; `iq4_kt` at 14.17 GiB gives PPL 3.1391 and 37.39 tok/s (−8.1%); `iq3_kt+iq4_kt` at 12.99 GiB gives PPL 3.1777 and 35.76 tok/s (−12.1%). The trellis quants lose on *both* axes for that model. Do not assume trellis dominates; measure per model.

### 3.3 Ollama, llamafile, koboldcpp, LM Studio

**Ollama** (178.9k stars) went back to llama.cpp, as described above. Its Apple Silicon path is its own Go engine over MLX (`x/mlxrunner`) with per-architecture models, a prefix cache, a cache trie, MTP, DFlash, and speculation. A 2026 GGUF performance release claims up to 20% faster on NVIDIA and enables Vulkan by default for AMD/Intel. Defaults worth knowing: `OLLAMA_NUM_PARALLEL=1`, flash attention **off**, KV cache f16, and context length auto-selected at 4k/32k/256k by VRAM tier (<24 / 24-48 / ≥48 GiB).

**llamafile** moved to `mozilla-ai/llamafile` and is alive again (0.10.5, 2026-08-03) with a rewritten build system tracking recent llama.cpp. Still Cosmopolitan Libc single-file executables; still capped at 4 GB on Windows.

**koboldcpp** (2026-08-16 snapshot) tracks upstream aggressively and adds DSpark + DFlash speculation, image/video generation, and a single-file distribution.

**LM Studio** ships dual runtimes, llama.cpp for GGUF everywhere, Apple MLX on Apple Silicon, swappable in the runtimes manager. Its 2026 surface includes an MCP client, publishable `model.yaml` definitions, speculative decoding, parallel requests, an `lms` CLI and daemon, Python/TypeScript SDKs, and LM Link for routing across devices. Good product, poor research substrate: the orchestration decisions are not inspectable.

---

## 4. The datacenter Python/CUDA engines

### 4.1 vLLM

Python/PyTorch control plane over CUDA, Triton, and CUTLASS kernels. The 2026-08-11 snapshot follows a roughly two-week release cadence. Baseline stack: PyTorch 2.13.0, Triton 3.7.1, FlashInfer 0.6.16.post3, Transformers 5.14.1.

**V0 is fully gone**, the V1 guide says so outright. Casualties of the migration: `best_of` sampling, per-request logits processors, GPU↔CPU KV swapping, request-level structured-output backends.

**Architecture.** Multi-process: an API server process (HTTP + input processing) feeds an EngineCore process (scheduler + KV cache manager, one per DP rank, running a busy loop) which drives GPU worker processes. A 4-GPU single-node deployment is six processes.

**Model Runner V2** is the 2026 core rewrite: a persistent-batch state table with GPU gathers instead of tensor reordering, GPU-native input preparation via Triton kernels (input_ids/positions/seq_lens built on device), async-first with no CPU-GPU syncs, and a `ModelState` abstraction that cut the largest runner file from 6,700 to under 1,300 lines. Reported +56% throughput on Qwen3-0.6B (25K vs 16K tok/s) and −6.3% TPOT with speculative decoding. Still opt-in behind `VLLM_USE_V2_MODEL_RUNNER=1`.

**Defaults.** Chunked prefill on whenever possible, decode prioritised. Prefix caching supported without explicit config. `torch.compile` at `-O2` (`FULL_AND_PIECEWISE` cudagraph mode) as default, artifacts cached under `~/.cache/vllm`. **Async scheduling on by default.** Dual batch overlap off by default. Sleep mode off by default.

**Attention backend selection is architecture-keyed.** Blackwell SM10.x prefers FlashInfer then FlashAttention; Ampere/Hopper prefer FlashAttention first. So FlashInfer is only the default on Blackwell. FlashAttention 4 on SM100 with FP8 KV and headdim-256 is present as of 2026-08-11. There are separate MLA backends and a sparse-attention category (block-sparse, used for MiniMax M3).

**Speculative decoding** now covers EAGLE, MTP, draft model, PARD (parallel draft), MLP speculators, n-gram, **suffix decoding** (dynamic depth, no extra model), hidden-state extraction, and an experimental custom-proposer backend, plus **dynamic speculative decoding** for fluctuating QPS and **adaptive verification** that sizes verification per request from drafter confidence. 2026 work: EAGLE 3.1, P-EAGLE, Speculators with online training, DSpark adaptive verification.

**KV connectors.** `NixlConnector` (UCX + GDS), `LMCacheConnectorV1`, `MooncakeConnector`, `MoRIIOConnector` (ROCm), `OffloadingConnector` (CPU), `FlexKVConnectorV1`, `MultiConnector`. Documentation is explicit that disaggregation does *not* improve aggregate throughput, only TTFT/ITL control.

**Quantization.** FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO. The well-optimized NVIDIA path is **Marlin**; the hardware matrix shows broad NVIDIA coverage and no AMD or Intel GPU support for it. MXFP4 is unsupported on Turing.

**The consumer-relevant data point.** vLLM on DGX Spark (GB10, 128 GB unified, sm_121) running Nemotron-3-Super-120B-A12B NVFP4 with `--max-num-seqs 4`, `--max-model-len 131072`: **22.7-23.7 tok/s decode**, TTFT 1.12 s to 3.85 s, prefill 1,636-1,877 tok/s, KV utilization under 5% single-user, and a ~25 s JIT cold start needing pre-warming. Official guidance is that 100-130B MoE NVFP4 with 10-15B active is the intended fit and dense models are a poor fit. **Sleep mode** is the other consumer-relevant feature: level 1 offloads weights to host RAM (wake 0.1-6 s), level 2 discards them; 18-200× faster model switching, five switches in 112.6 s versus 357.1 s.

**Weaknesses on consumer hardware.** A large Python/PyTorch stack, meaningful startup and compile warm-up, VRAM reserved for KV blocks and workspaces, CUDA-centric performance, throughput-shaped defaults, quant-format breadth that exceeds the set of well-tuned kernels, and CPU offload far less flexible than llama.cpp's.

### 4.2 SGLang

The 2026-08-08 snapshot follows the same two-week cadence, claiming deployment across more than 400,000 GPUs. Docs moved to `docs.sglang.io`.

**RadixAttention evolved into UnifiedRadixTree**, default for SWA, Mamba, and DSA models as of 2026-08-08, and **session-reference-aware** so agentic and RL-rollout workloads hold stable session references.

**HiCache** is a three-tier KV hierarchy: L1 GPU, L2 host memory (both instance-private), L3 cluster-shared storage over Mooncake/RDMA, DeepSeek 3FS, NIXL, AIBrix, or files. Flags: `--enable-hierarchical-cache`, `--hicache-ratio`, `--hicache-storage-backend`, with prefetch policies (`best_effort`, `wait_complete`, `timeout`) and write policies (`write_through`, `write_through_selective`, `write_back`).

**Breakable CUDA Graph** is the standout 2026 contribution and is described in section 1. Default for prefill and for DP attention as of 2026-08-08.

**Attention backends** are unusually numerous. MHA: FlashInfer, FA3, FA4, Triton, Torch SDPA, FlexAttention, TRTLLM MHA, Dual Chunk FA, HPC-Ops, AITER+Wave (ROCm), Ascend, Intel XPU, Intel AMX. MLA: FlashInfer MLA, FlashMLA, Cutlass MLA, TRTLLM MLA, CuteDSL MLA, TokenSpeed MLA, FA3, FA4, Triton, Ascend. Plus `--linear-attn-backend` for gated delta net, **DSA sparse attention** for DeepSeek-V3.2, and **HiSparse** hierarchical sparse attention. Defaults: Hopper → FA3; Blackwell B200 → TRTLLM MHA/MLA; otherwise FlashInfer with Triton fallback. Prefill and decode backends can be split independently (experimental).

**Speculative decoding** covers EAGLE-2, EAGLE-3, EAGLE-2+FR-Spec, MTP, DFlash (linear block verification), standalone draft, and n-gram, with **Spec V2 / overlap scheduler** default as of 2026-08-08. The published progression on LLaMA-3.1-8B / MT-Bench / 1×H100 is clean: baseline 158.34 tok/s → EAGLE-2 244.10 (1.54×) → EAGLE-3 373.25 (2.36×). The surveyed DSpark implementation reports confidence-driven drafting at 383.7 tok/s with accept length ~5, and **ReplaySSM Ring Spec-Verify** cutting speculative scratch memory from 11.5 GB to 1.8 GB. **IndexShare MTP** reuses the sparse-attention indexer top-k as the draft signal for up to 1.9× lower draft-step cost, an instance of reusing work that already exists.

Composability constraints are worth internalising as a design warning: DFlash is incompatible with DP attention, PP, and the overlap scheduler; n-gram disables the overlap scheduler and mixed chunked prefill; standalone drafting doesn't work with DP attention. Feature matrices in these engines are sparse, and that sparsity is itself a symptom of retrofitted design.

**Structured output**: XGrammar (default, JSON schema + regex + EBNF), llguidance (same three), outlines (schema + regex only). Docs recommend XGrammar.

### 4.3 TensorRT-LLM

**The TensorRT engine-build path is dead.** The PyTorch workflow is the stable default. `LLM(backend="tensorrt")` now raises `ValueError` and all TRT-specific CLIs and APIs are deleted. The August 2026 snapshot includes a C++ `KVCacheManagerV2` with paged vanilla attention and FlashInfer block reuse, plus a much richer sampler. **AutoDeploy** is a beta backend for PyTorch→TRT-LLM deployment.

**Consumer support is not a documented commitment.** The support matrix lists only datacenter parts and appears stale (it still cites TensorRT 10.11, an artifact of the removed path). Ada (SM89 = RTX 4090) and consumer Blackwell (SM120 = RTX 5090) are architecturally covered but never named, and NVFP4/MXFP4 are absent from that table despite being used heavily elsewhere in the project. The surveyed consumer-adjacent item is DGX Spark beta support. Documented known issues include torch.compile conflicting with CUDA graphs and MoE accuracy problems under certain quantizations.

For a consumer research engine, TRT-LLM is best treated as a source of kernel ideas rather than a baseline you can actually run.

### 4.4 FlashInfer: the shared substrate

This is the shared kernel library under the most live engines. As of 2026-08-12, used by SGLang, vLLM, TensorRT-LLM, MLC-LLM, LightLLM, LoRAX, and ScaleLLM. Kernel surface: attention (paged + ragged KV, decode/prefill/append, MLA, cascade attention, block-sparse, POD-attention for mixed batching), GEMM in BF16/FP8/FP4, MoE, sorting-free sampling, communication primitives, RoPE, norms. Delivery is JIT-first with optional AOT (`flashinfer-cubin`, `flashinfer-jit-cache`). Hardware span SM 7.5 (Turing) through SM 12.1 (Blackwell).

The architecturally important idea is the **plan/run split**: `plan` builds request/tile schedules, indirection arrays, workspace requirements, and kernel selection; `run` executes a graph-friendly specialized kernel. That separation is exactly what makes paged attention CUDA-graph-compatible, and it is worth copying.

As of that snapshot, MoE expert parallelism is production-ready with full CUDA-graph capture/replay and a fused single-launch quantize-and-stage hot path, the **SM12x (RTX 5090) fused-MoE kernels** have FP4 accuracy fixes, and the unified MoE API covers MXFP4 W4A8/W4A16.

### 4.5 The rest of the tier

**LMDeploy** (InternLM) is alive with two engines: TurboMind (hand-written CUDA) and a PyTorch engine. **The surveyed distribution ships prebuilt CUDA 12.8 wheels supporting RTX 50-series**, the surveyed datacenter-class engine that explicitly advertises consumer Blackwell. Claims 4-bit 2.4× faster than FP16, and MXFP4 at 1.5× vLLM on H800.

**TGI is archived** (2026-03-21, read-only) and redirects to vLLM/SGLang/llama.cpp/MLX.

**Aphrodite rebranded to "Sonar"** (2026-07-31), still a vLLM fork adding extra model/quant formats and sampling methods, with Apple Silicon support alongside CUDA/ROCm/CPU.

**LightLLM** is research-flavored and possibly slowing (last release September 2025), but two ideas are worth stealing: **Pre³**, deterministic-pushdown-automaton constrained decoding (ACL 2025 outstanding paper), and a past-future SLA-aware scheduler (ASPLOS'25).

**nano-vLLM** is about 1,200 lines of Python, MIT-licensed, with prefix caching, TP, torch.compile, and CUDA graphs. Its own benchmark on an RTX 4070 Laptop with Qwen3-0.6B and 256 sequences: 1,434.13 tok/s versus vLLM's 1,361.84, roughly 5% faster in 0.2% of the code. **This is the single best reference codebase for understanding a serving control path.**

**DeepSpeed-MII is dormant.** No commit since 2025-06-30, README news stopping at January 2024. No archive notice, but no development. Do not build on it.

**Orchestration** (Dynamo, llm-d, LMCache) sits strictly above the engines. Dynamo's own README says it "doesn't replace SGLang, TensorRT-LLM, or vLLM, it turns them into a coordinated multi-node inference system." llm-d joined CNCF as a sandbox project in March 2026. The surveyed LMCache distribution includes ROCm wheels; its **CacheBlend**, non-prefix KV reuse at arbitrary prompt positions, is the one idea from this layer that a single-GPU engine should care about.

None of this layer helps one GPU directly. The transferable lesson is that KV *location* deserves to be first-class metadata and that routing should consider cache reuse, not queue length. A consumer engine should keep that separation clean and not embed orchestration in the runtime.

---

## 5. The consumer specialists

### 5.1 ExLlamaV3 and EXL3

Shipped 2026-07-14 after a long alpha; still receiving roughly weekly releases as of 2026-08-11, authored personally by turboderp. **ExLlamaV2 is archived** and tabbyAPI dropped EXL2 (frozen on a checkpoint branch).

**EXL3 is a streamlined QTIP variant.** Procedural codebooks plus optimal tail-biting trellis structures, diverging from QTIP in how tensors are regularized and packed. Hessians are computed on the fly and a **fused Viterbi kernel** converts in a single pass, minutes for small models, a few hours for 70B+ on one 4090. That is the headline result: AQLM-class quality at roughly 1/1000th of the conversion cost. The GEMM kernel is Marlin-inspired and reaches approximately memory-bound latency at 4bpw on a 4090. Bitrate is a continuous `-b` target rather than an enum, with documented working points from ~1.6 to 8 bpw; Llama-3.1-70B is described as "coherent at 1.6 bpw" under 16 GB VRAM. Cache quantization is separately 2-8 bit, and as of July 2026 online cache quantization has **no latency cost** and often increases throughput on KV-heavy models. The default codebook changed from `mcg` to `mul1`.

A tail-biting trellis has identical initial and final state, which avoids explicit termination overhead while still allowing bounded independently-decodable blocks, that is how random access into a trellis-coded tensor is made tractable on GPU. The "1MAD"/"3INST" family of computed codebooks generates reconstruction values from a short sequence of integer mixing instructions rather than loading a table, trading scarce bandwidth for plentiful integer ALU.

**Speed.** The [reported implementation update](https://html.cafe/x16d85e50) delivered +14-66% decode over its earlier implementation on a 3090 and up to +109% on a 5090, via new cooperative GEMM and INT8 GEMV kernels. Examples on a 5090: Llama-3.1-8B 4bpw 209 tok/s, 2bpw 259; Qwen3.6-27B 3bpw 86; Gemma-4-12B 3.5bpw 134. The historical criticism, trellis decode is ALU-heavy and lagged EXL2 on Ampere, was explicitly targeted; improvements on Ada are described as "more modest since memory and ALU pipes are better balanced."

**Hardware:** CUDA 12.4+, SM80+ (Ampere/Ada/Blackwell). ROCm is the only item on the to-do list. Tensor and expert parallelism for multi-GPU, AVX512 CPU expert offload, and a second-tier CPU K/V cache.

Turboderp's own documentation: *"GGUF i-quants are abundant, and it's worth noting that they hold up well in comparison to SOTA formats."* He also expects stock QTIP to match or beat EXL3 on accuracy. Quality comparisons in the repo are published only as PNG charts (perplexity-vs-bpw, perplexity-vs-VRAM, KL-divergence-vs-bpw for Llama-3.1-8B), so numeric values could not be extracted. However, `eval/qbench.py` runs apples-to-apples KL-divergence against an HF BF16 reference across `transformers` / `exllamav3` / `llamacpp` engines with exact bpw accounting from shard headers. That is the harness to point a new engine at. It is the only cross-engine, cross-format quality tool found in this survey.

**tabbyAPI** is the OAI-compatible server, explicitly a hobby project "not meant to run on production servers."

### 5.2 MLX and mlx-lm

`mlx==0.32.1` (2026-08-18), 28k stars, the most actively developed project in this cluster.

**The CUDA backend is the headline.** QMM/QMV quantized kernels for SM80, cuDNN SDPA with attention sinks, CUDA FFT, CUDA Hadamard transform, FSDP, Windows CUDA + cuDNN. On the Apple side: **M5 "NAX" neural-accelerator kernels** (NAX attention, `MLX_SDPA_BLOCKS`) and **JACCL**, a low-latency RDMA-over-Thunderbolt collective library.

**Quantization modes** (`mx.quantize`), with defaults in bold:

| Mode | Group size | Bits | Scale type | Bias |
|---|---|---|---|---|
| `affine` | 32, **64**, 128 | 2,3,**4**,5,6,8 | as input | yes |
| `mxfp4` | **32** | **4** | e8m0 | no |
| `mxfp8` | **32** | **8** | e8m0 | no |
| `nvfp4` | **16** | **4** | e4m3 | no |

Plus mixed/per-layer precision predicates including for MLA models.

**Learned quantization is the real differentiator.** Four CLI tools: **DWQ** (`mlx_lm.dwq`) distils the non-quantized parameters including scales and biases against the FP teacher; **AWQ**; **GPTQ**; and **dynamic quantization** (`mlx_lm.dynamic_quant`), which estimates per-layer output sensitivity and assigns high/low bits to hit a `--target-bpw`. Methods cascade (dynamic → DWQ). The documented DWQ caveat is instructive: it is best at 2-4 bit and *poor* at 6-8 bit "because loss starts too low", at high bitrates the initial error is so small that calibration noise outweighs the correction.

**Measured quality/speed curve** (64 GB M4 Max, macOS 26.1, Qwen3-4B-Instruct-2507, MMLU-Pro / gen tok/s / GB):

| Format | MMLU-Pro | tok/s | GB |
|---|---:|---:|---:|
| bf16 | 64.05 | 52.5 | 9.02 |
| q8 | 63.85 | 86.9 | 5.25 |
| q6 | 63.53 | 104.7 | 4.25 |
| q5-g32 | 63.16 | 110.3 | 4.00 |
| q4-g32 | 61.46 | 126.0 | 3.60 |
| q4-g64 | 60.72 | 134.5 | 3.35 |

Read the shape rather than the numbers: 8-bit costs 0.20 points, 6-bit costs 0.52, 5-bit-g32 costs 0.89, and 4-bit costs 2.6-3.3. **Q6 is nearly free; 5-bit-g32 is the knee; 4-bit is where you start paying.** On Qwen3-30B-A3B the same curve holds (q8 72.46 at 33.5 GB, q4 70.71 at 18.2 GB).

Other features: speculative decoding (`--draft-model`, `--num-draft-tokens`, default 2), batch generation with `BatchKVCache`/`BatchRotatingKVCache`, prompt caching to safetensors, rotating fixed-size KV (`--max-kv-size`), and `mx.distributed` over MPI, RING (TCP, usually faster than MPI), JACCL (RDMA/Thunderbolt, required for tensor parallelism), or NCCL.

An MLX-vs-llama.cpp-Metal head-to-head could not be verified for 2026 and should not be asserted.

### 5.3 mistral.rs and candle

**mistral.rs** (2026-08-14 snapshot), 7,609 stars, built on candle. The interesting parts for a Rust engine builder: **ISQ** (in-situ quantization at load time, no offline step, with an executor and planning layer); **AFQ** (MLX's affine quantization, ported with CUDA and CPU backends, on Metal `2/3/4/6/8` resolve to AFQ2-8, on CUDA/CPU to Q2K/Q3K/Q4K/Q6K/Q8_0); **PagedAttention on CUDA *and* Apple Silicon**; **per-layer topology files** and "mixture of quant experts" for fine-grained bit allocation; and a complete serving surface (OpenAI `/v1`, Anthropic `/v1/messages`, MCP client and server, Prometheus metrics, `mistralrs tune` hardware-aware quant recommender, `mistralrs doctor`).

Its published self-reported benchmarks are worth reading for the *weakness*: it beats llama.cpp decisively on Q8 prefill (27,706 vs 11,992 tok/s, Gemma-4-E4B on B200) and beats vLLM on BF16 dense decode, but **loses badly on BF16 MoE prefill** (3,467 vs vLLM's 28,533). Vendor-reported, but the honest disclosure of the MoE prefill gap makes the rest more credible.

**candle** (20.9k stars, pushed 2026-08-15) remains a low-level tensor library, not an engine. Excellent Rust substrate; comparing "candle performance" to vLLM conflates a framework with a serving system.

### 5.4 KTransformers

Restructured in 2026. The monolithic framework is archived; the project now ships **`kt-kernel`**, a CPU MoE kernel library, and has been **upstreamed into SGLang** since October 2025. Strategic direction is "CPU MoE kernel library for other engines," not standalone server.

The canonical single-4090 numbers remain from the archived 2025 tutorial: DeepSeek-V3/R1 671B Q4_K_M on **14 GB VRAM + 382 GB DRAM** (4090D + Xeon Gold 6454S): prefill 54.21 tok/s at 32 cores, 74.4 dual-socket, **255-286 tok/s with the AMX kernel** in the 6-expert selective mode; decode 13.7 tok/s single-socket up to 16.8. Against llama.cpp's 10.31 prefill / 4.51 decode, that is up to 27.8× prefill and 3.0× decode.

Six auto-selected CPU backends (AMX / AVX512+BF16 / +VBMI / +VNNI / AVX512 / AVX2), plus AMD BLIS and ARM KML paths, an AVX512 native-precision FP8/BF16/RAWINT4 backend, NUMA-aware thread pools, hot-expert-on-GPU/cold-on-CPU scheduling, and a three-layer GPU-CPU-disk prefix cache. GPU side requires SM 8.0+.

### 5.5 Compiler stacks and alternative languages

**Modular MAX / Mojo.** Mojo 1.0 shipped in MAX 26.5.0 (2026-08-11), beginning stdlib API stability guarantees. The repo holds "over 450,000 lines of code from over 6,000 contributors" and is plausibly the largest open repository of CPU and GPU kernels. GPU programming APIs live in a top-level `max` package (`max.gpu.*`, `max.algorithm`, `max.layout`). GPU support spans NVIDIA, AMD, and Apple Silicon: Apple GPU serving covers most MAX models on M3+, and as of 2026-08-11 extends back to M1 with hardware-MMA flash-attention prefill on M5. Whether the Mojo *compiler front-end* is open source could not be confirmed; the evidence points to stdlib and kernels open, compiler proprietary. The "171% of vLLM throughput" claim is MAX 26.1 vs vLLM 0.10.1 on AMD MI355X, vendor-reported, six months stale, and on the hardware where vLLM's ROCm path is weakest. Not a defensible general claim. **The kernels are the asset here, not the server.**

**MLC-LLM / TVM.** 23,071 stars and pushed 2026-08-17, but the last six months of commits are entirely "adapt to TVM API refactor" plumbing. Docs still say 0.1.0 and © 2023-2025; the only GitHub release is `v0.1.dev0` from 2023. Breadth remains unmatched (Vulkan/ROCm/CUDA/Metal/OpenCL/WebGPU+WASM, iOS/Android) and **WebLLM** is the living downstream. Treat MLC as the reference for browser and mobile deployment, not a competitive server engine. A caution: a deep compiler refactor can legitimately suppress visible feature work, so "dead" is too strong.

**tinygrad** (33,462 stars, daily commits) repositioned as an end-to-end deep learning stack. Relevant as a hackable compiler with BEAM-search kernel autotuning and fully visible IR, not as a serving engine.

**ZML** (Zig + MLIR + OpenXLA + PJRT, 3,984 stars, daily commits) targets NVIDIA/AMD/Intel/TPU/Trainium. Its LLM example supports only Llama 3.1/3.2, Qwen 3.5, LFM 2.5, narrow, but the cleanest example of an ahead-of-time-compiled, non-Python, no-CUDA-toolchain inference stack.

**ONNX Runtime GenAI** 0.15.2 powers Foundry Local, Windows ML, and the VS Code AI Toolkit, across CPU/CUDA/DirectML/NvTensorRtRtx/OpenVINO/QNN/WebGPU. Multi-LoRA, continuous decoding, constrained decoding; speculative decoding still on the roadmap. This is the Windows/NPU story.

**OpenVINO GenAI** 2026.3.0.0 has C++/Python/Node.js APIs with no external dependencies across CPU/GPU/NPU, and two features worth stealing: **KV-cache token eviction** and **sparse attention prefill (Tri-shape and XAttention)**.

**ipex-llm is archived** with a security notice. Do not build on it.

### 5.6 Sparsity, ternary, and distributed-consumer research

**PowerInfer** moved to `Tiiny-AI/PowerInfer` after commercialization; last push 2026-05-11. The research core (hot/cold neuron placement by activation locality, 13.2 tok/s average on one 4090, 11.7× over llama.cpp on ReLU-sparse models) is unchanged since 2024. The pivot is to hardware, a "Pocket Lab" at CES 2026 running GPT-OSS-120B int4 locally at 20 tok/s. **Activation sparsity is now a niche idea; MoE displaced it.** SwiGLU models simply are not zero-sparse the way ReLU models were, and predictor overhead plus false negatives eat the margin.

**Fiddler** remains a systems-research prototype, but its scheduling insight is the durable one: compare *host compute time* against *PCIe transfer plus GPU compute* rather than assuming GPU execution is preferable.

**BitNet** (`microsoft/BitNet`, 40,103 stars, the most-starred repo in this cluster) is research-adjacent, not a general engine: it only runs models *trained* ternary, and only one general LLM exists (`BitNet-b1.58-2B-4T`, April 2025). 2026 activity moved to embeddings (BitNet-embedding-0.6B/270M, 1.3-2.3× F16 prefill speedup at 2 bits/weight) and CPU ASR. Claimed speedups 1.37-5.07× on ARM, 2.37-6.17× on x86, with 55-82% energy reduction. It is model/runtime co-design, not a drop-in for GGUF Q2.

**exo** has 46.9k stars, tensor parallelism claiming 1.8× on two devices and 3.2× on four, topology-aware auto-parallel, a macOS app, and day-0 RDMA-over-Thunderbolt-5 support with a claimed "99% reduction in latency between devices." **WebLLM** (18.6k stars) remains the WebGPU story with OpenAI-compatible APIs and q4f32/q4f16 model variants.

---

## 6. Quantization in 2026

The field bifurcated cleanly.

**Datacenter: microscaling FP formats won.** NVFP4 (E2M1 values, group 16, FP8 E4M3 block scale plus a tensor-level FP32 scale: 4.5 bpw) and MXFP4/MXFP8 (OCP MX, group 32, E8M0 power-of-two scale: 4.25 bpw) are the default for new checkpoints. NVFP4's finer blocks and non-power-of-two scales give better fidelity; MXFP4 is simpler and slightly smaller. `llm-compressor` 0.13.0 ships W8A8 int8/fp8, W4AFP8, NVFP4/MXFP4/MXFP8, **FP8 and NVFP4 attention/KV-cache quantization**, rotation methods (SpinQuant, QuIP), and **REAP expert pruning**. Structural MoE expert removal is a 2026 axis. The dominant production recipe is **NVFP4 on MoE layers plus FP8 on attention**, used by GLM-5.2, Hy3, and Kimi-K3 checkpoints. GPT-OSS confirms the pattern natively: its config carries `"quant_method": "mxfp4"` with `modules_to_not_convert` covering self-attention, the MoE router, embeddings, and lm_head.

**Consumer: three credible frontiers.**

1. **GGUF k-quants and i-quants with imatrix** remain the practical Pareto default, universal, abundant, CPU+GPU+Metal, and per turboderp's own assessment they "hold up well in comparison to SOTA formats."
2. **Trellis/vector quantization** (EXL3 on GPU, ik_llama.cpp's `_KT` family on CPU) is the accuracy frontier at 2-4 bpw and the surveyed method that stays coherent at 1.6-2 bpw. Cost: bespoke kernels, ALU-heavy decode, and, as the Qwen3.6-27B measurement showed, not always a win.
3. **Learned post-processing** (DWQ, AWQ, GPTQ, dynamic bit allocation) is the cheapest quality win at 3-4 bit and composes with any storage format.

**Declining:** AutoGPTQ archived (2025-04, superseded by `ModelCloud/GPTQModel`), AutoAWQ archived (2025-05, superseded by llm-compressor), HQQ moved to `dropbox/hqq` and semi-dormant, bitsandbytes alive but now primarily a QLoRA fine-tuning tool.

### The unifying view

All layer-wise PTQ methods minimize the same objective:

```
min_Ŵ  ‖(W − Ŵ)X‖²_F  =  min_Ŵ  tr((W − Ŵ) H (W − Ŵ)ᵀ),   H = E[xxᵀ]
```

They differ only in which degree of freedom they exploit. GPTQ/OBQ quantizes columns sequentially and uses a damped-Cholesky `H⁻¹` to compensate earlier errors into remaining weights. AWQ searches per-input-channel scaling `S`, quantizes `WS`, and pushes `S⁻¹` into activations. **The imatrix is exactly `diag(H)`**, from `E‖Ex‖² = tr(EHEᵀ)`, dropping off-diagonals gives `Σⱼ hⱼ‖E:,ⱼ‖²` with `hⱼ = E[xⱼ²]` accumulated over calibration tokens. It is a diagonal second-moment approximation, not an arbitrary importance heuristic. QuaRot inserts Hadamard rotations that absorb into adjacent operators; SpinQuant learns the rotation instead of accepting a random one; QTIP rotates *and then* applies a strong vector quantizer.

Incoherence processing (a randomized Hadamard transform) works because it makes coordinates more isotropic, so a scalar or block quantizer approximates the full-Hessian objective better. That is why it composes with the methods above.

**What this survey did not find optimized:** the objective is always quality under a *storage* budget. No method surveyed here minimizes `quality + λ·T_decode + μ·T_prefill + ν·memory` with measured kernel latency in the loop, uses different execution formats for prefill and decode, arranges transforms so they cancel across multiple adjacent operators, or conditions the bit allocation on where the tensor will actually run (CPU expert versus GPU attention).

---

## 7. KV cache and long context

For a 32-layer, 8-KV-head, head-dim-128 model, KV costs `32 × 8 × 128 × 2 × 2 = 128 KiB/token`. At 128K that is **16 GiB in FP16**, 8 GiB at Q8, 4 GiB at Q4. A 24 GB card can hold an 8B Q4 model plus 128K Q4 KV, barely, before workspace and graph memory.

**K and V need different quantization schemes.** Keys have persistent *channel* outliers: a few dimensions stay large across many tokens, so per-token scaling lets those outliers consume the entire range. Values vary more by token with less stable channel structure. The right scheme is asymmetric per-channel INT4/INT2 for K, per-token grouping for V, an FP16 residual window over recent tokens, and higher precision for attention sinks. RoPE complicates this because it mixes coordinate pairs, quantizing pre-RoPE K or using rotation-aware grouping helps.

**Sparse and compressed attention went mainstream in 2026.** DeepSeek-V4's scheme combines shared K/V vectors via inverse RoPE (2× reduction), c4a/c128a KV compression (~¼ and ~1/128), top-k sparse attention, and a 128-token sliding window, for roughly **8.7× total KV reduction**, a 1M-token sequence needs 9.62 GiB/layer in bf16 versus 83.9 GiB, with another 2× available from fp4/fp8. vLLM implements it with unified 256-token logical blocks across layers, the compressor state treated as a sliding-window cache, three shared pools to limit fragmentation, kernel fusions (compress+RMSNorm+RoPE; inverse-RoPE+fp8-quant), and multi-stream overlap of the indexer with KV compression, giving 5-6% end-to-end latency at low batch. SGLang has DSA backends and HiSparse; GLM-5.2's DSA cache-layer split cuts per-rank KV memory ~74%.

The critical distinction: **NSA/DSA/MoBA-class methods are trained into the model.** MLA likewise, its low-dimensional latent `c_t^KV` plus a separate RoPE-bearing key component cannot be losslessly retrofitted. Post-hoc eviction (H2O, SnapKV, StreamingLLM) remains workload-fragile: it can pass perplexity and fail multi-needle retrieval.

For an existing dense GQA model on 24 GB the practical winning stack is paged Q4/Q8 KV with K/V-specific schemes, an FP16 recent-and-sink window, FlashAttention-style prefill, and eviction only where the task tolerates approximation. For reliable 128K retrieval you want a model trained with GQA/MLA/native sparse attention.

---

## 8. What separates the fast engines

Decompose one token:

```
T_token = T_schedule + T_launch + T_weights + T_attention + T_collective + T_sample
```

At batch one, `T_weights` and launch/scheduler latency dominate. At batch 16-128, GEMM efficiency dominates. At very long context, KV reads dominate. A server that wins decisively at 64 concurrent requests can lose at batch one, and vice versa, this is why "fastest engine" is a category error.

**A realistic 4090 batch-1 budget** for an 8B Q4 model (≈4.5-5.0 GB): weight streaming with fused dequant/GEMV 5.5-7.5 ms; KV attention at ~4K context 0.3-0.9 ms; non-matmul ops 0.1-0.4 ms; graph replay and launch gaps 0.1-0.4 ms; scheduler and metadata 10-80 μs; GPU sampling 20-100 μs; detokenization 10-100 μs (usually overlapped). Total ≈6.2-9.3 ms, or 108-161 tok/s. Without CUDA graphs, 150-300 uncaptured launches at 3-8 μs of exposed gap each adds 0.5-2 ms. A Python scheduler that allocates tensors, builds lists, synchronizes on logits, and samples on CPU adds another 0.3-1.5 ms.

**Where a naive implementation loses 2×:** materializing W4 weights to FP16 before GEMM; separate dequant and GEMM kernels; too few CTAs in the GEMV; synchronizing after every layer; hundreds of uncaptured launches; copying full-vocabulary logits to host; reallocating contiguous KV; treating M=1 as a padded large-M GEMM; running partially-offloaded layers serially across PCIe.

**The GEMV-to-GEMM crossover.** W4 arithmetic intensity grows roughly as `4M` FLOP/byte. On a 4090 (≈330 TFLOP/s FP16 tensor, 1 TB/s), roofline balance lands near M≈82, but that is not the kernel crossover. Direct GEMV wins at M=1-4; specialized small-M Marlin/Machete wins at M≈1-32; conventional GEMM takes over around M=8-32; compute-bound behavior arrives near M=64-128. Note that "M" can be users, prefill tokens, *or speculative tokens*. Speculation pushes you across this boundary without adding users.

**Why the three kernel families differ.** ExLlama kernels are decode-first: direct or near-direct quantized GEMV, specialized bit extraction, format-specific packing, no padding of M to an MMA tile. Marlin treats quantized inference as a carefully scheduled tensor-core GEMM with pre-permuted MMA-friendly weights, fused dequantization inside the tiled loop, and work striped over N and K so small M still has parallelism, its achievement is keeping tensor cores useful much further into the small-M regime. Machete is a CUTLASS/CuTe generated mixed-input GEMM family, systematic and extensible but dependent on finding a tile schedule that doesn't waste most of the MMA tile at M=1.

**Fusions that pay:** RMSNorm + activation quantization; dequant + matmul; gate/up projection + SwiGLU; QKV projection; RoPE inside Q/K production; KV write + attention metadata update; sampling transforms + top-k. Over-fusion hurts through register pressure and shape-specific recompilation.

**Graph capture.** What breaks it: changing batch size, changing sequence length and block tables, dynamic allocation, host decisions about finished requests, variable speculative width, changing LoRAs or grammars, sampling that returns a token to host, attention kernels requesting variable workspace. The fix is preallocation plus a small lattice of shape buckets with masked inactive slots. Attention is excluded from piecewise capture because its work depends on per-request sequence lengths and page counts, but this is not fundamental. Capture attention too, if metadata buffers have stable addresses, workspaces are preallocated, sequence lengths are device values, and counts are bucketed. FlashInfer's plan/run split exists precisely to make this possible.

On Apple there is no CUDA Graph equivalent. The available mechanisms are precompiled `MTLComputePipelineState`s, one reusable command-buffer encoding pattern, argument buffers whose contents change without rebinding, `MTLIndirectCommandBuffer` for indirect dispatch, and heap-based stable resource allocation. The biggest practical win is fusing operators and submitting one command buffer per decode step.

**Paged KV mechanics.** The kernel does not translate per scalar: one lane loads a physical page ID, broadcasts it, and the warp consumes a contiguous tile within that page. The block table is tiny and stays in L1/L2. A good paged kernel lands within 0-10% of an equivalent contiguous kernel at medium-to-long context; poor layouts or tiny workloads lose 10-30%. Real page sizes are usually 16 or 32 tokens. To measure the penalty honestly, run the same kernel with an identity block table versus randomized physical pages at identical sequence lengths, that isolates address translation from allocator overhead.

**Speculative decoding rollback** is poorly documented. Verification writes KV for speculative nodes; treat it as a transaction. Reserve staging slots, run verification, find the accepted path, commit KV for accepted tokens, free the rejected suffix. The replacement token has logits but no committed KV. With paged KV, whole speculative pages can be remapped rather than copied; a partially filled tail page needs copy-on-write or a validity cursor. Tree verification is harder because accepted nodes can be non-contiguous. Under CUDA graphs you capture a fixed maximum tree size and mask unused nodes, or keep width buckets. A host readback to select the next graph reintroduces the synchronization you were avoiding.

Realistic consumer speedups: 0.9-1.5× for a small target with a separate draft, 1.3-2.3× for a large target with a good draft, 2-3× with excellent EAGLE/MTP acceptance. The SGLang numbers (1.54× EAGLE-2, 2.36× EAGLE-3 on 8B) sit right in that band and are the best-documented public figures.

---

## 9. Consumer hardware specifics

**Bandwidth is the budget.** RTX 4090: 24 GB, ~1.0 TB/s. RTX 5090: 32 GB, ~1.79 TB/s plus native Blackwell low-precision. Apple: M4 ~120 GB/s, M4 Pro ~273, M4 Max ~410 or 546 by configuration, M3 Ultra >800.

**Heterogeneous MoE, with numbers.** The decision rule for one expert:

```
CPU compute wins when:  B_e/BW_cpu + C_cpu  <  (B_e/R)/BW_pcie + B_e/BW_gpu + C_gpu
```

with `R` = reuse across the batch (`R=1` at batch one). Time to move 1 GB of weights: dual-channel DDR5-6000 ≈11 ms; 8-channel server DDR5 ≈2.5 ms; PCIe 4.0 x16 transfer plus GPU compute ≈41 ms; PCIe 5.0 ≈21 ms; GPU-resident ≈1 ms.

For DeepSeek-V3 (d_model 7168, expert intermediate 2048, 256 routed experts, 8 active), one expert is ≈44M weights ≈22 MB at W4, so 8 experts is ≈176 MB per MoE layer per token. Ideal bandwidth-only time per layer: 0.18 ms GPU-resident, 0.44 ms on 8-channel CPU, 1.96 ms on dual-channel, 3.5 ms PCIe-5 streaming, 7.0 ms PCIe-4 streaming: *before* GPU compute. Across ~60 MoE layers, dual-channel CPU expert bandwidth alone is ≈118 ms/token, about 8.5 tok/s before overhead.

**The conclusion is unambiguous: compute host-resident experts on the host. Never stream expert weights over PCIe at batch one.** KTransformers' system-level gain comes more from avoiding PCIe transfers and overlapping CPU/GPU than from AMX peak TOPS, because batch-one quantized expert execution is DRAM-bound.

Expert prefetching has a hard ceiling: one W4 expert is ~0.9 ms over PCIe 4, so eight is ~7 ms, and false positives are expensive. Top-8 recall may hit 70-95% on predictable workloads, but routers can shift sharply after attention and residual updates. Prediction is useful for caching one or two statistically hot experts, not for speculatively streaming all eight.

**Two GPUs without NVLink.** Every tensor-parallel layer needs a PCIe collective, and at batch one each collective carries a tiny activation but pays full fixed latency. Better placements: pipeline/layer split minimizing crossings; whole experts pinned per GPU for MoE; **target model on GPU 0 and draft model on GPU 1** (this avoids an all-reduce in every target layer, which naive TP does not); asymmetric placement by VRAM; second GPU for prefill only when prompts are long enough to amortize KV transfer.

**Apple specifics.** Unified memory removes PCIe copies and lets a 70B Q4 model run where a 24 GB card cannot, but CPU, GPU, media engines, and the OS share the same bandwidth, and swap is catastrophic for token latency. AMX is exposed mainly through Accelerate rather than a stable public ISA, and the ANE is mediated by Core ML with restricted operators and no convenient custom low-bit kernels; neither is a good foundation for an experimental engine. Metal *does* have SIMD-group shuffles and `simdgroup_matrix`, so it is not devoid of warp-level tools, but it gives less explicit control over bit-manipulation instructions, async global→shared pipelines, TMA-like bulk movement, and occupancy diagnostics.

**Recommendation: CUDA first, but design the IR and memory model so Metal stays credible.** Starting with both doubles kernel and debugging work and delays the actual contribution. M5's NAX accelerators make Metal more strategically important, but until Apple documents lower-level access, MLX is the sane interface.

---

## 10. Gaps and opportunities

Several 2025-era gaps closed during the past year and should be struck from any research plan.

**Closed:** inventing a better low-bit codebook (trellis shipped on both CPU and GPU); generic CPU/GPU MoE offload (kt-kernel, upstreamed into SGLang, plus llama.cpp's `-ncmoe`); generic host-RAM KV tiering (llama.cpp `--cache-ram`, SGLang HiCache, LMCache); generic prefix reuse (RadixAttention, APC, `--cache-reuse`); a portable general-purpose tensor IR (MLX, TVM, MAX, Triton, CUTLASS, ggml all occupy it).

**Still open, and worth the effort:**

**1. Rate-distortion-*latency* quantization.** Every quantizer minimizes reconstruction error under a bits budget and then hopes a kernel is fast. This survey found no method that puts measured kernel latency into the objective. A quantizer that benchmarks candidate formats on the actual target device and allocates bits to minimize `quality_loss + λ·T_decode + μ·T_prefill + ν·bytes` would be a new contribution, and the ik_llama.cpp result where `iq4_kt` loses to `iq4_ks` on *both* PPL and speed is direct evidence that the current objective is wrong.

**2. Dual-layout execution from one checkpoint.** Decode wants a GEMV-shaped layout; prefill wants tensor-core tiles. Every engine commits to one. ik_llama.cpp's `_R4`/`_R8` repacked variants and llama.cpp's runtime repacking are the closest existing work, but both are all-or-nothing at load time. Deriving the second layout cheaply, or storing a compact delta, without doubling model storage is unclaimed. This composes with (1) and is the strongest pairing.

**3. Transactional speculative KV pages.** A page allocator supporting branch-local staging, zero-copy commit of the accepted path, partial-page copy-on-write, and device-side accepted-length commit under fixed CUDA graphs. Every engine implements some version of this badly and none exposes it as a clean abstraction. The composability failures in SGLang's feature matrix (DFlash incompatible with DP attention, PP, and the overlap scheduler) are symptoms of exactly this missing abstraction.

**4. Quality-aware asymmetric KV tiering.** The generic version is claimed. What is not: different K and V schemes selected *per layer and per head* from online sensitivity estimates, asynchronous migration overlapped with attention, a guaranteed memory budget with a reported quality-risk score, and a rigorous comparison of recompute versus retain versus compress versus evict. The framing that matters is treating output-quality degradation as a runtime-managed resource with an explicit budget, not a dtype flag.

**5. Consumer topology optimization.** Automatically choose among second-GPU draft execution, TP, EP, CPU experts, and KV placement from *measured* PCIe and memory costs rather than static `-ngl`. The Fiddler inequality generalized and made online. This is well-scoped and consumer users would benefit immediately.

**6. Adaptive speculation as a control problem.** llama.cpp now exports per-position acceptance metrics and offers ten speculation strategies; vLLM has dynamic spec decode and adaptive verification; this survey found no engine that closes the loop. Choose the proposal mode per request *and per phase*: n-gram during code repetition, tiny draft during open-ended text, MTP where available, none when acceptance or verification economics turn negative. Measure acceptance, verification cost, KV length, and draft contention online.

**7. Energy and thermals.** Consumer GPUs and laptops throttle; datacenter engines ignore this and local products hide it. Sustained tokens/joule and p99 latency under thermal steady state are the metrics users actually experience. Power-limiting the GPU during memory-bound decode is nearly free.

**8. A credible cross-engine evaluation suite.** ExLlamaV3's `eval/qbench.py` runs KL-divergence against a BF16 reference across `transformers`/`exllamav3`/`llamacpp` with exact bpw accounting from shard headers. It is the only cross-engine quality tool found in this survey, and it comes from a single-maintainer hobby project.

---

## 11. Recommended architecture for a new engine

**Language and dependencies.** Rust for the server, scheduler, allocator, and model metadata; a thin C++/CUDA kernel ABI; CUTLASS/CuTe for conventional tensor-core GEMM; custom CUDA for batch-one GEMV, KV mutation, and irregular quantization; Triton for prototyping only; GGUF as an *importer* that gets repacked into your own execution layouts. Do not wrap ggml as the core. You inherit its graph and layouts, which is precisely what you want freedom to change. Do not write every GEMM yourself. **FlashInfer is the optional dependency with the largest kernel surface**, particularly its SM12x (RTX 5090) fused-MoE and FP4 kernels and its plan/run architecture.

**Minimum viable scope.** One Llama/GQA dense architecture, one DeepSeek-style MoE, RTX 4090/5090, batch 1-4, paged KV, graph capture, GGUF/AWQ import with internal repacking, and *one* new contribution. Metal, Vulkan, multimodality, LoRA multiplexing, and cluster serving are not MVP.

**Execution.** Separate compiled plans for prefill, decode, and speculative verification. Decode graphs bucketed for batch 1/2/4 and selected speculative widths. Graph capture with explicit eager breaks (SGLang's BCG approach) rather than a compiler, given the 3.8-5.2× faster build times and quarter-size codebase. GPU-resident sampling and grammar masks. Asynchronous token output with no device synchronization on the decode path. Target host overhead below 1-2% of token time.

**Two reference codebases to read first:** nano-vLLM (~1,200 lines, the whole control path is legible) for scheduling, and ExLlamaV3's kernels for what a batch-one quantized GEMV should look like.

**Traps to avoid.** Building a general compiler before demonstrating one execution idea. Inventing a 2-bit format without a faster consumption kernel. PCIe weight streaming at batch one. Unstructured sparsity without hardware-aligned blocks. Optimizing only average tokens/s. Claiming 128K support because allocation succeeded. Adding every model architecture. Python in the decode critical path. Duplicating a datacenter scheduler for one user. Comparing quantizations without downstream quality evaluation.

**Evaluation harness.** Record: exact model and artifact hashes; quantizer commit, calibration corpus, and effective bpw *by tensor class*; GPU clocks, power limit, driver, and kernel cache state; cold and warm TTFT; prompt throughput; p50/p95/p99 inter-token latency; batch 1/2/4; context 128/2K/8K/32K/128K; peak VRAM and RAM; PCIe bytes moved; physical and effective bandwidth (against the roofline in section 2); joules/token and thermal steady state; graph-capture and compilation time; speculative acceptance and wasted verified tokens. Use identical token streams and seeds across engines. Quality should include perplexity *plus* task suites sensitive to quantization and context, reasoning, code, multilingual, single- and multi-needle retrieval, long-context generation.

Publish the losing configurations and the crossover surfaces, not only the point where the new engine wins. In a field where every project reports vendor benchmarks on its own favorable hardware, and where no apples-to-apples vLLM-versus-SGLang 2026 comparison appears to exist publicly, that alone would be a contribution.

---

## 12. Cross-check notes: where the two research tracks disagreed

Worth recording, because these are the places 2025 intuitions have gone stale.

| Claim from the offline model | What the web shows |
|---|---|
| ExLlamaV2 is the NVIDIA consumer baseline | ExLlamaV2 archived; EXL3 shipped 2026-07-14 |
| Ollama is moving to a native Go engine | Reversed: Go GGML runner removed, back to `llama-server` subprocess; Go engine only for Apple/MLX |
| MLX is Apple-only | MLX has a CUDA backend with SM80 quantized kernels and Windows support |
| TGI is "maintenance-oriented" | Archived read-only, 2026-03-21 |
| TRT-LLM has both TRT and PyTorch backends | TRT backend removed entirely |
| `--split-mode row` is the multi-GPU option | `row` deprecated; experimental `tensor` mode is new |
| PowerInfer-style activation sparsity is promising | Dormant since 2025-05; MoE displaced it |
| Trellis quantization is a prime research direction | Already shipped in production on both CPU and GPU |

The offline track's *reasoning* held up well: the roofline argument, the wave-count analysis, the GEMV/GEMM crossover, the heterogeneous-MoE inequality, and the KV rollback transaction model all match the measured data. Its *facts* were roughly a year stale. Check derivations independently and verify factual claims against their sources.

---

## Sources

**llama.cpp / ggml**
- https://github.com/ggml-org/llama.cpp. README, releases, `include/llama.h`, `src/llama-graph.h`, `ggml/include/ggml.h`, `docs/build.md`, `docs/multi-gpu.md`, `docs/speculative.md`, `docs/ops.md`
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- https://github.com/ggml-org/llama.cpp/discussions, #23875 (unified binary), #27290 (RTX PRO 4000 Blackwell), #27154 (Ryzen AI Max+ 395, ROCm vs Vulkan), #27331 (tile-DAG CPU decode), #15013 (CUDA scoreboard), #4167 (Apple Silicon scoreboard), #27149 (MoE SSD streaming), #27219 (AMD HRX backend RFC), #20969 (TurboQuant)
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- https://llama.app/

**Forks and derivatives**
- https://github.com/ikawrakow/ik_llama.cpp. README, `docs/parameters.md`; discussion #2213 (Qwen3.6-27B quant comparison)
- https://github.com/ollama/ollama: `runner/runner.go`, `llm/llama_server.go`, `llama/compat/README.md`, `LLAMA_CPP_VERSION`; https://ollama.com/blog
- https://github.com/mozilla-ai/llamafile
- https://github.com/LostRuins/koboldcpp
- https://github.com/abetlen/llama-cpp-python
- https://lmstudio.ai/docs/app

**Datacenter engines**
- https://github.com/vllm-project/vllm, releases feed; https://docs.vllm.ai/en/latest/: `design/arch_overview.html`, `usage/v1_guide.html`, `design/attention_backends.html`, `configuration/optimization.html`, `features/speculative_decoding/`, `features/disagg_prefill.html`, `features/quantization/`
- https://vllm.ai/blog/2026-03-24-mrv2 (Model Runner V2); vLLM blog on DeepSeek-V4 attention (2026-04-24), FP8 KV/attention quantization (2026-04-22), sleep mode (2025-10-26), EAGLE 3.1 (2026-05-26), P-EAGLE (2026-03-13), DSpark (2026-08-14)
- https://github.com/sgl-project/sglang, releases; https://docs.sglang.io: `advanced_features/hicache_design`, `advanced_features/attention_backend`, `advanced_features/session_radix_cache`, speculative decoding docs
- https://lmsys.org/blog/2026-08-17-advanced-cuda-graph (Breakable CUDA Graph)
- https://github.com/NVIDIA/TensorRT-LLM, releases; https://nvidia.github.io/TensorRT-LLM/release-notes.html, `reference/support-matrix.html`
- https://github.com/flashinfer-ai/flashinfer
- https://github.com/InternLM/lmdeploy
- https://github.com/huggingface/text-generation-inference (archived 2026-03-21)
- https://github.com/aphrodite-engine/aphrodite-engine (renamed Sonar)
- https://github.com/ModelTC/LightLLM
- https://github.com/GeeeekExplorer/nano-vllm
- https://github.com/deepspeedai/DeepSpeed-MII
- https://github.com/ai-dynamo/dynamo · https://github.com/llm-d/llm-d · https://github.com/LMCache/LMCache

**Consumer specialists**
- https://github.com/turboderp-org/exllamav3. README, `doc/exl3.md`, `eval/qbench.py`, release notes (https://html.cafe/x16d85e50)
- https://github.com/turboderp-org/exllamav2 (archived) · https://github.com/theroyallab/tabbyAPI
- https://github.com/ml-explore/mlx · https://github.com/ml-explore/mlx-lm: `mlx_lm/LEARNED_QUANTS.md`, `mlx_lm/BENCHMARKS.md`, `python/src/ops.cpp`
- https://github.com/EricLBuehler/mistral.rs · https://ericlbuehler.github.io/mistral.rs/reference/quantization-types/
- https://github.com/huggingface/candle
- https://github.com/kvcache-ai/ktransformers · https://ktransformers.readthedocs.io/

**Compilers, alt-languages, edge**
- https://github.com/modular/modular (MAX 26.5.0, Mojo 1.0) · https://www.modular.com/max
- https://github.com/mlc-ai/mlc-llm · https://github.com/mlc-ai/web-llm
- https://github.com/tinygrad/tinygrad · https://github.com/zml/zml
- https://github.com/microsoft/onnxruntime-genai · https://github.com/openvinotoolkit/openvino.genai
- https://github.com/intel/ipex-llm (archived)
- https://github.com/exo-explore/exo

**Quantization**
- https://github.com/vllm-project/llm-compressor (0.13.0) · https://github.com/vllm-project/compressed-tensors
- https://github.com/ModelCloud/GPTQModel · https://github.com/dropbox/hqq · https://github.com/bitsandbytes-foundation/bitsandbytes · https://github.com/pytorch/ao
- https://github.com/microsoft/BitNet · https://arxiv.org/abs/2402.17764 (BitNet b1.58)
- https://arxiv.org/abs/2406.11235 (QTIP) · https://arxiv.org/abs/2402.04396 (QuIP#) · https://arxiv.org/abs/2211.10438 (SmoothQuant)
- https://arxiv.org/abs/2510.13999 (REAP expert pruning)
- https://huggingface.co/openai/gpt-oss-120b (native MXFP4 config)

**Papers and background**
- https://arxiv.org/abs/2309.06180 (PagedAttention) · https://arxiv.org/abs/2312.07104 (SGLang/RadixAttention)
- https://arxiv.org/abs/2312.12456 (PowerInfer) · https://arxiv.org/abs/2406.06282 (PowerInfer-2) · https://arxiv.org/abs/2402.07033 (Fiddler)
- https://github.com/Dao-AILab/flash-attention · https://github.com/NVIDIA/cutlass · https://github.com/IST-DASLab/marlin
- https://github.com/xlite-dev/Awesome-LLM-Inference (DeepSeek NSA/FlashMLA, prima.cpp, BitNet v2, GuidedQuant)

**Offline second-opinion track:** four rounds (survey, kernel/runtime mechanics, quantization internals and heterogeneous execution math, cross-check against web findings). No network access. Architectural reasoning and quantitative modeling, not facts. Roofline, wave-count, GEMV/GEMM-crossover, and heterogeneous-MoE analyses were independently consistent with the measured llama.cpp benchmark data in section 2.
