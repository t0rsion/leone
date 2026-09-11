# Kernels, Scheduling and Low-Level Systems Techniques for Fast LLM Inference

*Survey compiled 2026-08-18, oriented toward building a new research inference engine for consumer hardware (1-2 consumer GPUs, Apple Silicon, x86/ARM CPUs).*

---

## 0. The frame: two regimes, one roofline

Almost every technique below is explicable from one equation. For a decode step at batch size `B`, the GPU must read all model weights (plus the KV cache for the active sequences) from DRAM once, and does roughly `2 * P * B` FLOPs against `P` bytes-ish of weight traffic. Arithmetic intensity is therefore ~`B` FLOP/byte in fp16 and ~`4B` in 4-bit. Every modern accelerator has a ridge point far above that:

| Device | BW (GB/s) | Dense BF16 TFLOP/s | Ridge (FLOP/byte) |
|---|---|---|---|
| RTX 4090 | 1008 | ~165 (330 w/ sparsity) | ~165 |
| RTX 5090 | 1792 | ~210 dense BF16 (FP4 ~1.6 PFLOPS eff.) | ~120 |
| H100 SXM | 3350 | ~990 | ~295 |
| B200 | 8000 | ~2250 | ~280 |
| M4 Max | 546 | ~34 (GPU FP16) | ~60 |
| Ryzen AI Max+ 395 (Strix Halo) | ~256 | - | - |
| DGX Spark (GB10) | ~273 | - | - |
| Desktop DDR5-6000 dual channel | ~90 theoretical, 60-75 real | - | - |

So **decode at B=1 is a pure memory-bandwidth problem** and **prefill is a compute problem**. The upper bound for decode is `tok/s ≤ BW / bytes_read_per_token`. An 8B model at Q4_K_M is ~4.7 GB; on a 4090 (1008 GB/s) the ceiling is ~214 tok/s and llama.cpp realistically hits 120-150, i.e. 60-75% of theoretical. On a 5090 (1792 GB/s) the same model ceilings around ~375 tok/s and llama.cpp measures 185.9 tok/s for Qwen3-8B Q4_K_XL, i.e. **the consumer 4-bit decode path leaves 25-50% of bandwidth unused**. That gap is where a new engine wins, and §9.2 shows it is dequantization and fixed overhead, not bandwidth.

The two headline structural facts:

1. **You cannot beat the roofline, you can only approach it.** Approaching it means: fewer passes over weights, larger effective batch (speculation, MoE routing tricks), lower bytes/param (quantization), and eliminating the stalls that idle the memory pipeline (kernel boundaries, launch gaps, CPU stalls).
2. **At batch 1 on a consumer GPU the GPU is idle most of the time anyway.** A Llama-1B forward pass on H100 is ~1 ms of *actual* work and existing engines spend ~50% of bandwidth. Anything that fills those bubbles (megakernels, speculative decoding, async scheduling) converts directly into tokens.

---

## 1. Attention kernels

### 1.1 The FlashAttention lineage

**FA1** (2022) tiles and uses online softmax so `S = QK^T` is never materialized. **FA2** (2023) fixes work partitioning, parallelize over sequence length too, fewer non-matmul FLOPs, less shared-memory traffic between warps; still the fallback everywhere pre-Hopper and the basis of PyTorch SDPA. **FA3** (2024) adds Hopper machinery: producer/consumer **warp specialization**, **TMA** bulk async copies, async `wgmma`, and softmax/GEMM interleaving so exp2 work on the SFU overlaps tensor-core work.

**FA4** (arXiv 2603.05451, March 2026) targets Blackwell around a premise worth internalizing: *asymmetric hardware scaling*. Blackwell tensor cores handle 128×256×16 tiles (~4× Hopper's) but shared memory and the exponential unit did not scale proportionally. So FA4 (a) **software-emulates the exponential** instead of using the SFU, (b) uses **conditional softmax rescaling** to skip provably-unnecessary rescales, (c) uses **tensor memory (TMEM)** and **2-CTA MMA** to cut SMEM traffic and backward-pass atomics, and (d) is written **entirely in CuTe-DSL embedded in Python**, compiling 20-30× faster than C++ CUTLASS templates. Result: **1605-1613 TFLOP/s on B200 (~71% utilization), 1.3× cuDNN 9.13, 2.7× Triton**. Caveat for this audience: FA4 needs SM100/SM103 and CUDA 12.8+: **consumer Blackwell (SM120, RTX 5090) is not a target** and has no `tcgen05`/TMEM.

**The consumer lesson:** FA3/FA4 machinery is largely unavailable on SM89 (4090) and SM120 (5090), and a hand-written Ampere-style kernel gets very close. gau-nernst's from-scratch 5090 kernel, using only `cp.async`, `ldmatrix` and swizzled SMEM:

| Version | Technique | TFLOP/s | % of SOL |
|---|---|---|---|
| v1 | tiled + `cp.async` + `ldmatrix` | 142.9 | 68.2% |
| v2 | + XOR shared-memory swizzling | 181.1 | 86.5% |
| v3 | + 2-stage `cp.async` pipelining | 189.8 | 90.6% |
| v4 | + `ldmatrix.x4` (32B loads) | 194.3 | 92.8% |
| v5 | + drop redundant V double-buffering | 197.7 | **94.4%** |

Same machine: PyTorch SDPA 186.7 (89.1%), `flash-attn` 190.6 (91.0%), cuDNN 203.6 (97.2%). Two lessons: **swizzling alone is worth ~27%**, and **instruction issue rate, not memory, becomes the limiter** on modern consumer parts, prefer fewer, heavier instructions.

### 1.2 Decode attention: split-KV and GQA packing

At decode the query is one token, so the (batch × heads × query-blocks) grid collapses to (batch × heads): on a 4090 with 32 heads at batch 1 you occupy 32 of 128 SMs. **FlashDecoding** applies split-K along KV: compute partial (output, max, sumexp) per chunk, then combine via log-sum-exp in a second pass. Grid becomes `(batch, heads_kv, num_splits)`.

Split-KV is not universally a win: on GPUs where a subset of SMs already saturates DRAM, splitting only adds a reduction pass. Autotune `num_splits` against sequence length and batch. FlashInfer, FA3's split-KV path, and vLLM's Triton backend all do.

**GQA packing is the other half.** With Llama-3-8B's 32 Q heads / 8 KV heads, the 4 query heads sharing a KV head pack into the `M` dimension of an MMA, converting a GEMV into a small GEMM and letting decode use tensor cores at all. This is why GQA ratio is a template parameter in every serious decode kernel, llama.cpp's included.

### 1.3 FlashInfer

FlashInfer (MLSys 2025 best paper) is the closest thing to *the* attention kernel library, now largely NVIDIA-maintained. Ideas worth stealing:

- **Block-sparse row (BSR) as the universal KV format.** Paged KV, ragged KV, radix-tree prefix sharing and sliding-window are all block-sparse matrices with different block sizes, so one templated kernel family covers all of them. *Don't write a separate kernel per cache layout.*
- **JIT-compiled attention templates**. ALiBi, logit soft-cap, custom masks, on-the-fly RoPE as compile-time functors, not runtime branches.
- **Plan/run split**: a CPU-side `plan()` computes a load-balanced work partition for the ragged batch into pre-allocated device tensors; `run()` is CUDA-graph-capturable with fixed shapes. This is how you get variable-length batches *and* CUDA graphs.
- **Cascade attention** for shared prefixes (attend to prefix once with a multi-query kernel, then per-request suffix, merge by log-sum-exp, same primitive as split-KV), and **POD-Attention** fusing prefill+decode for chunked-prefill mixed batches.

Reported: 29-69% ITL reduction vs compiler baselines, 28-30% for long context, 13-17% end-to-end. Supports SM75-SM121, is CUDA-graph and `torch.compile` compatible, and is **vLLM's default attention backend on Blackwell** (FlashAttention remains default on Hopper).

### 1.4 PagedAttention, and the case against it

PagedAttention made KV non-contiguous, killing fragmentation and enabling copy-on-write prefix sharing, the enabling idea for continuous batching at scale. Costs: every attention kernel must walk a block table, plus per-step CPU block-manager work.

**vAttention** (ASPLOS 2025) argues this is unnecessary: use **CUDA virtual memory APIs** (`cuMemCreate`/`cuMemMap`/`cuMemAddressReserve`) to keep KV *virtually contiguous* while backing it with physical pages on demand, so unmodified attention kernels work as-is. Reported **up to 1.23× throughput** over PagedAttention variants of FA/FlashInfer, **up to 1.97× faster generation than vLLM**, 1.45-3.92× faster prompt processing. There's an open vLLM issue (#17612) to adopt it.

For a *new* consumer engine this is a genuine fork in the road. Serving one or two users, block tables buy almost nothing and cost real kernel complexity; VMM-backed contiguous KV lets you use *any* stock kernel including cuDNN and `flash-attn` directly. Caveats: CUDA VMM granularity is 2 MB, and equivalents vary (ROCm has VMM; Metal has heaps + sparse buffers; Vulkan has sparse binding).

### 1.5 MLA and FlashMLA

DeepSeek's Multi-head Latent Attention compresses KV to a 512-dim latent plus 64-dim decoupled RoPE: **656 bytes/token** (512B quantized latent + 16B scales + 128B RoPE) versus tens of KB for MHA. Decode is then effectively MQA over the latent: bandwidth-friendly, arithmetically awkward.

**FlashMLA** on H800 SXM5 / CUDA 12.8: dense decode **up to 3000 GB/s** memory-bound and **660 TFLOP/s** compute-bound; sparse decode (FP8 KV) **410 TFLOP/s**; sparse prefill **640 TFLOP/s** (B200: **1450**); dense MHA prefill on B200 1460 fwd / 1000 bwd. Paged KV, block size 64.

The **Hopper FP8 sparse deep-dive** (Sept 2025) is the best worked bottleneck example I found. Dequantizing FP8 KV on CUDA cores cost **~50 cycles/token** while the tensor-core MMA needed only **~34**, the kernel was *dequantization-bound*. Fix: launch CTAs in **clusters of 2** and exploit MQA's property that all query heads in a token attend to the same K head, so each CTA dequantizes half the K/V and shares it via **Distributed Shared Memory** (`st.async`). Result **250 → 410 TFLOP/s**. Generalizable lesson: **when you quantize the KV cache, check whether you moved the bottleneck from DRAM to the dequant ALU.**

### 1.6 Sparse attention as a kernel problem

- **NSA**, hardware-aligned trainable sparsity (compressed/selected/sliding branches); Triton kernels assign each GQA group to one SM to share the KV fetch. **9.0× fwd / 6.0× bwd at 64k**. **FSA** (arXiv 2508.18224) reorders the loops for **up to 3.5×** over the vanilla NSA kernel.
- **MoBA**, blockwise top-k routing of queries to KV blocks.
- **DSA (DeepSeek-V3.2)**, a **"lightning indexer"** (tiny FP8 scorer) ranks KV entries per query; top-k blocks go to real attention. O(L·k) instead of O(L²), **~3-6× cost reduction at 128k** with negligible quality loss. vLLM and SGLang both shipped day-0 support.

At short contexts this matters less than you'd think; at 64k+ where KV traffic dominates weight traffic it matters a lot. The kernel work is top-k selection without materializing scores, plus the gather of scattered KV blocks.

### 1.7 FlexAttention, SDPA, xformers

**FlexAttention** compiles `score_mod`/`mask_mod` Python functions into a **block-sparse Triton kernel**, with a `BlockMask` skipping fully-masked 128×128 blocks (a 1M-token mask costs only ~60 MB). Slightly slower than FA3 on plain causal attention, but a large win on exotic masks, document masking, sliding window + sink, PrefixLM, tree attention for speculative decoding. **Without `torch.compile` it is catastrophically slow**; the mods run eagerly elementwise. **PyTorch SDPA** is the sane Python default but gives no control over KV layout; **xformers** is mostly legacy for inference though it still covers head dims `flash-attn` doesn't.

### 1.8 How llama.cpp does it

The most-deployed consumer engine, so worth studying:
- CUDA FA has three families selected by compute capability and head dim: **`fattn-vec`** (CUDA-core, tuned for batch 1), a legacy **WMMA** family (Volta), and the modern **`fattn-mma-f16`** family that drops WMMA and issues **`mma.sync` PTX directly** (PR #11583) because WMMA's opaque fragment layout blocked optimization. Templated over head dim and GQA ratio.
- **Quantized KV is dequantized on the fly inside the attention kernel**, so KV quantization costs bandwidth savings without a separate pass.
- Metal has its own FA kernels on `simdgroup_matrix`; Vulkan gained FA via `NV_cooperative_matrix2` and now `KHR_cooperative_matrix`.
- Recurring practical caveat: FA on/off and KV quant type change which kernel path is selected and can swing decode throughput by tens of percent.

---

## 2. GEMM and GEMV for quantized decode

### 2.1 The decode GEMV problem

At batch 1 the "GEMM" is `[1, K] × [K, N]`, a GEMV. There is no data reuse on the weight matrix, so the *only* thing that matters is reading the quantized weights at full DRAM bandwidth and dequantizing them for free. Three requirements:

1. **Coalesced, wide loads.** Weights must be laid out so that consecutive lanes read consecutive bytes, hence weight *pre-packing/interleaving* at load time rather than at kernel time.
2. **Dequantization must cost ~nothing.** This is where the bit tricks live.
3. **The reduction must be cheap**, warp shuffles, not shared memory round-trips.

### 2.2 The LOP3 / `prmt` dequantization trick

The canonical trick (Marlin, AWQ, TensorRT-LLM, FasterTransformer all use variants): converting int4 → fp16 with `__int2half` is slow. Instead, note that an fp16 with exponent bits set to `0x64` (i.e. bit pattern `0x6400 | x`) numerically equals `1024.0 + x` for small integer `x`. So:

- Mask out a nibble and OR in the constant exponent in **one `lop3.b32` instruction** (a 3-input arbitrary LUT: `(a & b) | c` in a single op), producing two fp16s from one 32-bit register at a time.
- Then a single `fma`/`sub` with `-1024.0` (folded into the zero-point) recovers the integer value, and the group scale is applied in the same FMA.
- `prmt.b32` (byte permute) rearranges nibbles/bytes to fit the tensor-core fragment layout without shifts.

MARLIN (PPoPP 2025) additionally: **asynchronous global weight loads** (`cp.async`) into a **circular shared-memory queue**, **striped partitioning** of the weight matrix across SMs with dual pipelines so that both the memory and compute pipes stay full, and a **GPTQ-compatible pre-permuted layout** so the dequantized fragments land exactly where `mma` wants them. It targets batch 1-64 and sustains **close to 4× over FP16** in the memory-bound regime, degrading gracefully as batch grows (unlike naive int4 kernels which fall off a cliff past batch ~8 because they were only ever GEMV).

### 2.3 The kernel zoo, and which to copy

| Kernel | Target | Notes |
|---|---|---|
| **Marlin / GPTQ-Marlin / AWQ-Marlin** | Ampere/Ada, W4A16 | The reference for batch 1-32 mixed-input. ~4× FP16. |
| **Machete** | Hopper, W4A16 | Marlin rebuilt on CUTLASS 3.5.1 with `wgmma`+TMA; ~29-32% throughput gain over Marlin at ≥3 req/s. |
| **GemLite** (Mobius/PyTorch) | Triton, W4/W2/W1 A16 | Triton-based, easy to extend to odd bit widths; the pragmatic path if you're Python-first. |
| **CUTLASS / CuTe** | general | The substrate. `mixed_dtype` GEMM examples cover W4A16 on Hopper/Blackwell. |
| **DeepGEMM** | Hopper/Blackwell FP8/FP4/BF16 | JIT-compiled at runtime, per-128-channel scaling, normal + grouped (contiguous & masked layouts) for MoE. **1.4-2.7× CUTLASS** on dense, 1.1-1.3× on MoE grouped; up to **1550 TFLOP/s on H800**. |
| **ggml MMVQ / MMQ** | consumer, all ggml quants | See below. |
| **TileLang / Triton** | portable | 20-line GEMMs; TileLang now emits CUDA, HIP, **Metal 4 cooperative tensors**, and LLVM CPU. |

**ggml's design is the one most relevant to a consumer engine.** Two kernel families:
- **MMVQ** (matrix-vector quantized): batch 1: one warp per output row, each thread loads a quantized block, calls a fused `vec_dot_q*_q8_1` that dequantizes and dot-products in registers using `__dp4a` (int8 SIMD dot product) where available. The activation is quantized to **Q8_1** on the fly so the dot product happens in int8 and the block scales multiply out at the end.
- **MMQ** (matrix-matrix quantized): batch >1: tiles into shared memory, and on CC ≥ 7.5 uses **tensor-core `mma` on int8**, falling back to `dp4a` otherwise. Recent PRs add **NVFP4 `dp4a` and NVFP4 MMQ kernels** for Blackwell.

The general principle: **quantize the activations too (to int8 or FP8) so the inner product runs on integer/low-precision tensor cores**, then apply per-block scales in the epilogue. This is why `dp4a` and `mma.s8` matter far more than fp16 tensor cores for consumer quantized inference.

### 2.4 FP4/FP8 on Blackwell

RTX 5090, RTX PRO 6000, B200/B300 all have **native FP4 tensor cores**. Two competing formats:
- **NVFP4**: block size 16, FP8-E4M3 per-block scale + FP32 global scale.
- **MXFP4** (OCP): block size 32, E8M0 (power-of-two) scale.

NVFP4 is meaningfully more accurate (smaller blocks, finer scale encoding); MXFP4 is the portability choice because **AMD MI355X also supports it**, and it's what `gpt-oss` shipped in. Both run through the same FP4 tensor-core datapath at runtime; only the checkpoint metadata differs. Reported: **up to 3.13× throughput over BF16 on B200**, 3.93× on DGX Spark, ~2.5× for W4A4. In vLLM, dense NVFP4 loads directly on Blackwell; MoE NVFP4 needs `VLLM_USE_FLASHINFER_MOE_FP4=1`.

For a consumer engine this is the single biggest hardware-driven opportunity of the 5090 generation: **W4A4 halves both weight traffic and doubles tensor-core throughput**, which is how prefill gets substantially faster on consumer parts.

### 2.5 Blackwell/Hopper kernel machinery, briefly

If you do write datacenter-class kernels: **TMA** (async bulk copies with hardware descriptors, no address computation per thread, `cp.async.bulk.tensor`), **`wgmma`** (Hopper, warpgroup-wide async MMA reading operands from SMEM), **`tcgen05`** (Blackwell 5th-gen MMA, 2-4× `wgmma` throughput, accumulator lives in **Tensor Memory (TMEM)** freeing the register file, plus **2-SM/2-CTA MMA** where a pair of SMs cooperate), **hardware block-scaling** for MXFP8/NVFP4 (the scale application is in the MMA instruction), **mbarrier-based producer/consumer warp specialization**, and **cluster / Distributed Shared Memory** (CTAs in a cluster read each other's SMEM). Note: **none of `wgmma`, `tcgen05`, TMEM, or 2-CTA MMA exist on SM89 or SM120**; consumer Blackwell has FP4 MMA and TMA but is otherwise an Ampere-shaped programming model.

---

## 3. Fusion, MoE, and the megakernel frontier

### 3.1 Ordinary fusions (do these first)

The pointwise ops between GEMMs are pure memory traffic. Standard fusion set:
- **RMSNorm + quantize**: fuse the norm with the FP8/int8 activation quantization that follows, so the activation is written once. ~**7× over unfused RMSNorm** at hidden 16384, ~3× memory reduction.
- **RoPE fused into the QKV epilogue** (or into the attention kernel's Q/K load). ~**8× over a standalone RoPE kernel**.
- **SwiGLU / gated MLP**: fuse `silu(gate) * up` and, better, fuse both projections into one GEMM with a fused epilogue. (Fused gate+up as a single GEMM also halves the weight-read launch overhead.)
- **Residual add + next norm** fused into one pass.
- **Dequant + GEMM (SplitK)** for skinny shapes: **64-124% average, up to 295% peak speedup on H100**.

Whole-model effects from fusion suites like Liger/Unsloth-style kernels: **+42.8% throughput and −54.8% memory for Llama-3-8B at batch 64**, +25.5%/−56.8% for Qwen2, +27% for Mistral-7B.

At batch 1 the payoff is different but real: each unfused pointwise kernel is a ~2-5 µs launch plus a full round trip of the activation tensor, and a 32-layer model has dozens of them.

### 3.2 MoE

MoE decode is where consumer hardware gets interesting, because only ~1/8-1/32 of the weights are read per token, but they're *scattered*.

- **Fused MoE / grouped GEMM**: one kernel launch performs `E` independent GEMMs with variable `M` per expert. vLLM's `fused_moe` Triton kernel sorts tokens by expert, builds a block→expert map, and each block does a normal tiled GEMM against its expert's weights. PyTorch has a **persistent, cache-aware grouped-GEMM Triton kernel** that keeps a fixed number of CTAs resident and streams work groups to them, improving L2 reuse of expert weights.
- **DeepGEMM** provides *contiguous-layout* (tokens pre-permuted, expert boundaries aligned) and *masked-layout* (fixed max tokens per expert, mask off the rest. CUDA-graph-friendly since shapes are static) grouped GEMMs.
- **DeepEP** handles expert-parallel dispatch/combine all-to-all with NVSHMEM, overlapping communication with compute; largely irrelevant for single-node consumer but the *idea*, overlap the expert gather with the previous expert's GEMM, applies to CPU-offloaded experts too.
- **Routing** itself (top-k over 128-512 experts, per token) is a nontrivial kernel; it must be fused with the softmax/sigmoid and the histogram/scatter that builds the permutation.

For consumer: the winning pattern in 2026 is **hybrid CPU/GPU MoE**, attention + shared/dense weights on the GPU, sparse expert FFNs in system RAM computed on the CPU (llama.cpp `--n-cpu-moe`/`-ot`, KTransformers, kt-kernel). See §7.

### 3.3 Megakernels: the frontier for batch-1 latency

This is the most important recent idea for the consumer/low-latency case and it is deeply under-exploited.

Observation (Hazy Research, "No Bubbles", 2025): a Llama-1B forward pass is **~100 separate kernels**. Even with CUDA graphs, each boundary costs ~1.3-2.1 µs *and*, worse, imposes a **global barrier**, no thread block of kernel *n+1* may start until every block of kernel *n* has retired. That means the memory pipeline drains and refills ~100 times per token. Existing engines achieve ~50% of memory bandwidth on this workload.

Their megakernel fuses the whole forward pass into **one persistent kernel** with:
- An **on-GPU interpreter**: each SM executes a statically-planned sequence of coarse "instructions" (RMSNorm+QKV+RoPE, attention, O-proj, MLP up/gate, down, ...), 7 instruction types as CUDA templates.
- **Shared-memory paging**: 213 KB of SMEM split into 13 × 16 KiB pages that instructions explicitly request/release, so the *next* instruction can begin loading weights before the current one finishes.
- **Explicit dependency counters in global memory** instead of kernel-level barriers.

Results: **<1 ms per forward pass on H100 at 78% of memory bandwidth** (vs ~50% for others), **2.5× vLLM, 1.5× SGLang**; on B200, ~680 µs, **3.5× vLLM, 1.5× SGLang**. Their B200 breakdown is instructive, of ~600 µs, 250 µs is activation store/load + sync, 200 µs norms and matmuls, 80 µs setup, 40 µs weight loading (successfully pipelined), 30 µs barriers.

**Mirage Persistent Kernel (MPK)** (CMU/UW/Berkeley/NVIDIA/Tsinghua) automates this: a compiler lowers a tensor program into an **SM-level task graph** with dependencies at individual-SM granularity, and an in-kernel decentralized runtime schedules tasks inside one persistent kernel. Reported **1.0-1.7× over SGLang/vLLM on A100/H100**.

**Kog** did the same on AMD MI300X with a "monokernel": grid fixed to 256 CUs (`gridDim=(256,)`, `blockDim=(64,8)`), no kernel launches at all in the decode loop. Their measured constants are gold: **kernel launch ≈ 4.5 µs on MI300X**, HBM restart latency ≈ 0.5 µs, intermediate tensor round trip > 1 µs. They replaced atomic-counter synchronization with **NaN-sentinel polling** on the output buffers (consumers spin until real data appears), cutting sync latency from **7.6-7.9 µs to 0.8-0.9 µs**. Grid synchronization still accounts for ~35% of token time. They report 3000+ tok/s for a 2B model at batch 1 on an 8×MI300X node.

**Fleet** (arXiv 2604.15379) extends megakernel abstractions hierarchically for multi-die GPUs.

**Why this matters for a new engine:** on consumer hardware at batch 1, kernel-boundary bubbles and launch overhead are a first-order cost, and CUDA graphs only fix the *launch* half, not the *barrier* half. A megakernel-shaped design (persistent kernel + on-GPU instruction interpreter + fine-grained dependencies) is a differentiating architecture, and it is far more tractable for a single model family at batch 1 than for a general serving system.

---

## 4. Overheads: launches, Python, and the scheduling hot path

### 4.1 The numbers

- Kernel launch: **~1.3-2.1 µs** (NVIDIA, even with CUDA graphs amortization on the record side), **~4.5 µs** on MI300X.
- A Llama-8B decode step on an H100 is **~5 ms** of GPU time; the CPU work (scheduling, input prep, detokenization, streaming) in a naive engine is comparable or larger.
- Unoptimized engines can spend **~half their wall time on CPU**, per SGLang's analysis.

### 4.2 CUDA graphs

Record a whole decode step's launch sequence once, replay it with one API call. Replay cost is ~constant regardless of kernel count. The problem is **shape polymorphism**: graphs bake in pointers and shapes.

vLLM's solution set (a good template):
- Capture at a **discrete ladder of batch sizes** (~1 to 256, powers of two plus intermediates); pick the smallest captured graph ≥ actual batch and **pad with dummy sequences**; fall back to eager if nothing fits.
- **Modes**: `NONE`, `PIECEWISE` (attention eager, the rest graphed, needs piecewise `torch.compile` splitting on `splitting_ops`), `FULL` (whole step graphed), `FULL_DECODE_ONLY`, and the default **`FULL_AND_PIECEWISE`** (full graphs for uniform decode batches, piecewise for prefill/mixed): most performant, most memory.
- A **dispatcher** keyed on a `BatchDescriptor` (num tokens, num requests, uniformity, LoRA-ness) selects the graph.
- Attention backends declare capability: `ALWAYS` (FA3, Triton, mixed prefill/decode graphable), `UNIFORM_BATCH` (FA2, FlashInfer+TRTLLM), `UNIFORM_SINGLE_TOKEN_DECODE` (FlashInfer, Mamba), `NEVER`. Incompatible combinations auto-downgrade.
- Speculative decoding produces **uniform batches of query length `1 + num_spec_tokens`**, so full graphs still apply.

The key enabling trick for graph-capturing attention with variable KV lengths is the **plan/run split**: all length-dependent metadata (block tables, cu_seqlens, split-KV work assignment) lives in **pre-allocated device tensors written by the CPU before replay**, so the graph itself sees fixed shapes. Related: **persistent kernels with a fixed launch grid** are strictly better for graphs than variable grids (vLLM's Triton backend moved to persistent kernels for exactly this reason).

### 4.3 Getting scheduling off the hot path

Three generations of the idea:

1. **Multi-step scheduling** (vLLM 0.6): schedule once, run *n* decode steps on the GPU before returning to the scheduler. Cuts CPU work by *n*× but hurts TTFT for newly arriving requests and complicates stopping criteria.
2. **Zero-overhead / overlap scheduler** (SGLang, from NanoFlow): run the scheduler **one batch ahead**, preparing all metadata for step *k+1* while the GPU executes step *k*. The dependency problem, step *k+1*'s input tokens are step *k*'s outputs, which don't exist yet on the CPU, is resolved with **future token placeholders** (negative indices resolved on-GPU) plus careful CUDA event ordering. Verified with nsys: no GPU idle gaps across 5 consecutive decode batches. **1.1× vs previous SGLang, 1.3× vs SOTA baselines**, biggest wins on small models and large TP. (Worth noting: a 2026 SGLang issue, #19347, claims the overlap doesn't always materialize in practice, measure it yourself.)
3. **vLLM V1 (Jan 2025)**: isolated `EngineCore` process running only scheduler + executor, ZeroMQ IPC, tokenization/detokenization/multimodal preprocessing/streaming all in a separate process overlapping the core loop. Explicit design goal: "near-zero CPU overhead."

**vLLM Model Runner V2 (March 2026)** is the current state of the art and the clearest statement of the lesson: they redesigned the execution core around three pillars: **modularity** (persistent `ModelState` separated from per-step inputs), **GPU-native bookkeeping** (input preparation moved into **Triton kernels** rather than Python/CPU tensor manipulation), and **async-first** (device-resident preparation consumes GPU results directly, eliminating CPU↔GPU syncs in async/speculative decode). Results: **+56% throughput on Qwen3-0.6B on 1×GB200** (25K vs 16K tok/s) and **−6.3% TPOT** for speculative decoding on 4×GB200.

The general principle: **anything computed per-step on the CPU that depends on GPU results creates a sync point; move it to the GPU or make it speculative.** The 2026 endgame of this line is `Blink` (arXiv 2604.07609), which delegates the entire serving stack to GPU and SmartNIC to remove the CPU from the loop entirely.

**For a batch-1 consumer engine**: you do not need a scheduler at all in the steady state. What you need is (a) no Python on the hot path, (b) CUDA graphs or a megakernel, (c) sampling on-GPU so you never sync to read the token, and (d) the next step's input written by the GPU itself. Achieving "zero CPU→GPU sync per token" should be an explicit design invariant.

### 4.4 torch.compile / Inductor, and static vs dynamic shapes

- Inductor fuses pointwise/reduction chains into Triton kernels, cutting both traffic and launch count; combined with CUDA graphs it's multiplicative.
- **Dynamic shapes** are the enemy: mark batch/seq as dynamic and you get generic kernels and guard-check overhead; specialize and you get recompiles. vLLM's answer is piecewise compilation with a fixed set of captured sizes.
- **AOTInductor** compiles to a standalone `.so` you can load from C++, the practical way to ship a Python-authored kernel graph in a non-Python runtime.
- Compile time is a real cost (vLLM caches compilation artifacts; MAX claims 8 min vs TensorRT-LLM's 28 min per model version).

---

## 5. Batching and serving scheduling

### 5.1 Continuous batching (Orca, OSDI 2022)

The foundational idea: schedule at **iteration granularity**, not request granularity, after every decode step, finished requests leave and waiting requests join. Plus **selective batching**: batch the ops that can be batched (all the GEMMs, by flattening tokens into one long sequence) and run attention per-sequence since each has a different KV length. Orca reported **36.9× throughput over FasterTransformer at equal latency** on GPT-3 175B. Every engine since is a variation.

### 5.2 Chunked prefill

Split a long prompt into chunks of *N* tokens and co-schedule chunks with decode tokens in the same batch. Benefits: decode tokens ride along in a compute-bound batch nearly for free; no decode stalls behind a 100k-token prefill (which would spike ITL). Costs: the prefill is now split across several kernel invocations and re-reads KV; and you must pick a chunk size. Measured effect is workload-dependent: ~**20% throughput improvement on high/low-prompt low-decode workloads, ~5% on high-decode workloads**.

**Consumer relevance**: chunked prefill is how you keep an interactive session responsive while a long document is being ingested, and how you cap peak activation memory. Also the reason to have a *unified* prefill+decode attention kernel (FlashInfer's POD-Attention, vLLM's unified Triton kernel) rather than two.

### 5.3 Prefill/decode disaggregation (DistServe, Mooncake, Dynamo)

Run prefill and decode on *different* GPUs and ship the KV cache between them. The rationale: the two phases have opposite resource profiles and opposite SLOs (TTFT vs TPOT), and colocating forces one parallelism/batching configuration on both. Honest summary from the literature: **disaggregation does not increase raw throughput; it makes tail ITL controllable.** Chunked prefill can achieve similar tail control if you can find the right chunk size, which is hard and workload-dependent. Mooncake adds a **KV-cache-centric** view: a global paged KV store across the cluster with prefix-aware scheduling.

**This is essentially irrelevant on 1-2 consumer GPUs**, but the underlying insight is not: *the prefill and decode phases want different kernels, different batch shapes, and different memory layouts*, and an engine should be able to configure them separately even inside one device.

### 5.4 Prefix caching / RadixAttention

vLLM: hash 16-token blocks (hash = f(prev_block_hash, tokens, metadata)), look up in a `cached_block_hash_to_block` dict, reuse the KV blocks. SGLang **RadixAttention**: a radix tree over token sequences with LRU eviction, so *any* shared prefix (not only a fixed system prompt) is reused, and the cache is scheduler-aware (requests are ordered to maximize hits). Reported hit rates: **40-70% for RAG with a fixed corpus**, **20-40% for multi-turn chat**, **75-95% for agent workloads with fixed system prompt + tools**.

For a local/agentic consumer engine this is arguably the highest-ROI feature in the entire document: agent loops re-send the same 5-50k-token context every turn, and prefix caching converts a multi-second prefill into a memcpy.

### 5.5 Priority and admission

Beyond FCFS: priority queues with preemption (recompute vs swap-out), and **goodput**-oriented scheduling (count only requests meeting their SLO). Preemption-by-recompute is usually better than swapping KV to host over PCIe at consumer bandwidths.

### 5.6 What actually matters at batch 1 on consumer hardware

Ranked, roughly:
1. **Quantization** of weights (4-bit): directly divides the bytes-per-token.
2. **Eliminating CPU/launch/barrier bubbles** (CUDA graphs → megakernel).
3. **Speculative decoding**, the technique that raises arithmetic intensity without adding users. At batch 1 the GPU is idle, so verifying 4-8 draft tokens is nearly free. EAGLE-3 reports **3.0-6.5× over vanilla AR** with mean acceptance length ~2.77 across 11 domains (3.16 coding, 3.12 math, 3.11 RAG), and acceptance is **flat from 1k to 32k context** (2.64 → 2.63). Interacts well with CUDA graphs because the verify batch has uniform query length `1+k`.
4. **KV cache quantization** at long context, once KV traffic exceeds weight traffic, this is the roofline.
5. **Prefix caching** for agentic/multi-turn.
6. The remaining techniques in this document.

---

## 6. Sampling, constrained decoding, and the output path

### 6.1 The sampler is a real kernel, not an afterthought

At batch 1 a decode step may be 3-5 ms; a naive sampler touches a 128k-256k-element logits vector several times (softmax → sort → cumsum → mask → multinomial), each a separate launch. Sorting is O(V log V) with terrible access patterns and is almost entirely wasted work.

- **vLLM PR #15478**: replacing the full sort with a threshold-based `torch.topk` fast path took 500 ops at 128k vocab / batch 1024 on A100 from **11.571 s → 2.136 s (5.4×)**.
- **FlashInfer's sorting-free sampler** uses **dual-pivot rejection sampling**: maintain two pivots bounding the probability threshold, do inverse-transform sampling against the prefix sum, reject and shrink the range by ≥half per round, guaranteed **O(log(1/ε))** rounds, all fused into **one CUDA kernel**. Reported **>50% reduction in sampling time** vs vLLM's sorting path on H100. Supports `top_k_first` vs joint filter ordering, min-p, and chain speculative sampling.

Three traps to design around:
1. **Host syncs.** vLLM's `apply_top_k_only` avoids the sort but "involves a GPU→CPU sync which can be detrimental for async scheduling." Any sampler that reads a scalar back to host (e.g. the batch's max top-k) serializes your pipeline and can cost more than the sort you removed.
2. **Logprobs.** FlashInfer doesn't expose post-filter logits/logprobs, so engines keep a torch fallback path. If you promise logprobs, you need both.
3. **Penalties.** Repetition/presence/frequency penalties require a gather/scatter over each request's irregular generated-token history, they vectorize far worse than top-k and are a common profiler surprise. Keep per-request sampling params as **batched GPU tensors**, never a Python loop.

Design target: **one fused kernel** doing softmax + min-p/top-k/top-p + categorical sample with **zero host sync**, plus the sampled token written directly into the next step's input buffer.

### 6.2 Speculative decoding: the batch-1 superpower

Speculation trades compute (which you have) for memory-bandwidth-bound steps (which you don't). Hence the crucial asymmetry: **EAGLE-3 gives ~2.3× at batch 4 and roughly break-even at batch 32.** Speedups also *decrease* on higher-bandwidth GPUs. Consumer batch-1 is the best possible regime.

Acceptance lengths (accepted tokens per target forward):

| Method | Accepted / step |
|---|---|
| EAGLE-3 typical | 2.4-2.8 (coding 3.16, math 3.12, RAG 3.11) |
| EAGLE-3 on Qwen3-4B (math/code/chat) | 5.14 / 3.69 / 2.39 |
| DFlash | 5.40 / 4.40 / 3.07 |
| **DSpark (2026)** | **6.11 / 5.13 / 3.64** |

EAGLE-3's acceptance is **essentially flat from 1k to 32k context** (2.64 → 2.63), so it doesn't decay on long prompts. Strongest measured consumer datapoint: on an **M3 Max/64 GB**, Gemma4-E2B drafting for Gemma4-26B-A4B hit **206.5 tok/s vs 63.9 standalone (3.2×)** at a cost of ~9 GB extra memory.

**DSpark** (DeepSeek, open-sourced June 2026; in SGLang July 2026) is the current SOTA: a semi-autoregressive **block drafter** (5 draft layers, γ=5) plus **confidence-scheduled variable-length verification**, per-request draft length chosen by cumulative survival probability, so you stop verifying tokens that won't be accepted. Reported **60-85% faster per-user generation** at matched throughput with zero quality loss, best across batch 1-256, **383.7 tok/s at batch 1** on DeepSeek-V4-Pro (TP=8, B300). Verification overhead going from 4→16 draft tokens is only **0.2-1.3%**.

**The CUDA-graph interaction is the subtle part.** Variable-length verification normally defeats graph replay because you pad to full width and give the savings back. DSpark's answer is **ragged verification**: front-pack variable-length requests into one compact buffer and round up to the nearest *captured tier*, replaying a smaller graph. Build a **ladder of captured graphs**, not one graph.

Cheaper options that need no draft model and work well locally: **n-gram / prompt-lookup decoding** (copy continuations from the prompt, excellent for code editing, summarization, RAG), **Medusa** heads, and **MTP** heads shipped with the model (DeepSeek, GLM, Qwen3-Next).

### 6.3 Grammar-constrained decoding: free if you overlap it

Per-token mask computation cost is the metric:

| Library | Per-token mask | Compile/warmup |
|---|---|---|
| XGrammar | **<40 µs** typical JSON | >1000 ms |
| llguidance | **~50 µs** single core @128k vocab; p99 <1 ms | negligible (dynamic) |
| XGrammar-2 (2026) | ~333 µs in their harness (llguidance ~250 µs) | **~10 ms** |
| Outlines | precomputed | seconds-minutes, high memory |
| llama.cpp GBNF, LM-format-enforcer | slower | - |

A 128k-vocab mask is 128k bits = **16 KB**, trivial to transfer; the cost is computing it. llguidance's framing: "with 16 cores and a 10 ms forward pass, llguidance can handle batch sizes up to 3200 without slowing down the model."

**The overlap rule is the measured difference.** SqueezeBits compared published vLLM and SGLang configurations on H100 with Qwen3-8B/32B using the *same* grammar libraries: **SGLang shows minimal loss by overlapping CPU grammar work with GPU inference; vLLM shows a significant drop at batch ≥8.** Compute the mask for step *N+1* on a CPU thread while the GPU runs step *N* and constrained decoding is free.

**XGrammar-2** (2026, now default in vLLM/SGLang/TensorRT-LLM as of March 2026) targets *dynamic* schemas for agentic tool calling: **6× faster compilation**, 8.1× preprocessing via JIT, 99.6× reduction for repetition-heavy schemas, **7× end-to-end latency speedup** in function calling, and end-to-end within **6% of unconstrained**. Tested on RTX 5090 and B200.

### 6.4 Tokenizers

Baselines on GPT-2 (AMD EPYC 9565): **tiktoken 36.0 MB/s, HF tokenizers 24.8 MB/s** (tiktoken skips normalization). At 25 MB/s a 100 KB prompt tokenizes in ~4 ms, negligible next to a prefill. **Tokenization is not your bottleneck.** 2026 outliers exist (Gigatoken claims 24.53 GB/s on a 144-core EPYC, but that headline compares a parallel implementation to single-threaded baselines).

**Where it *does* matter**: **incremental detokenization during streaming**, which runs every decode step per request. A naive implementation that re-decodes the whole sequence each token is O(n²) over the generation. Keep a per-request incremental decoder with a pending-bytes buffer for partial UTF-8 and multi-token graphemes.

---

## 7. CPU backends

### 7.1 x86: AVX2 → AVX-512/VNNI → AMX

AVX-512 FP32 with two FMA units is 64 FLOP/cycle/core (~190 GFLOPS at 3 GHz); VNNI's `vpdpbusd` does 64 INT8 MACs/instruction; AMX's TMUL does **2048 INT8 or 1024 BF16 ops/cycle/core**. Intel quotes **8× INT8 and 16× BF16 over AVX-512 VNNI**, i.e. ~8 TOPS INT8 per core at 2 GHz, ~250 TOPS for a 32-core Sapphire Rapids socket.

Achieved: **KTransformers' AMX kernels sustain 21.3 TFLOPS on one Xeon socket, 3.9× PyTorch native** (~1/3 of BF16 peak); their microbenchmark shows **AMX 5.4 vs AVX-512 1.8 TFLOPS** (3×) at high arithmetic intensity. Crucially they **switch dynamically: AMX for prefill, AVX-512 for decode**, because `tileconfig`/tile-load overhead dominates at batch 1 and decode is bandwidth-bound anyway. `kt-kernel` ships six runtime-selected variants (AMX INT4/INT8, AVX512+BF16, AVX512+VBMI, AVX2, ARM KML, EPYC BLIS); **AVX2-only MoE landed 2026-03-26**. llama.cpp's AMX path (`amx/mmq.cpp`, TILE_M/N=16, TILE_K=32) covers Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS and is **GEMM-only. GEMV falls back to AVX-512 VNNI**. **llamafile/tinyBLAS** (upstreamed as `GGML_LLAMAFILE`) took Mistral-7B F16 prompt eval on an i9-14900K from **13 → 52 tok/s (4×)**.

### 7.2 ARM: dotprod → i8mm → SVE2 → SME2

KleidiAI dispatches **SME2 → I8MM → DotProd**. `sdot` does 16 INT8 MACs per 128-bit instruction; `smmla` (i8mm) does a 2×8×2 outer product = 32 MACs, ~4× the per-instruction work, which is why i8mm mostly helps *prefill*.

**SME on Apple M4 Pro** is now well characterized (arXiv 2512.21473): 512-bit streaming vector length, **ZA register = 4096 bytes**. Measured **2006 GFLOPS FP32/BF16/FP16 single-thread, 4195 peak; INT8 4010 GOPS single-thread → 8375 peak; FP64 only ~501 GFLOPS**. Two structural facts an engine must respect: there is **one SME unit per ~5 performance cores** (SME does not scale with core count), and four-register SME loads reach 900 GB/s while single-register loads manage 230-375 GB/s, with sustained bandwidth needing an ~8 MB working set. A cache-aware INT8 GEMM reached **94% of SMOPA peak**.

**SMEPilot** (arXiv 2606.16332) gets **up to 3.94× end-to-end over llama.cpp** on M4 Pro / Dimensity 9500 / KunPeng 920 (prefill 1.13-5.42×, decode up to 3.48×), with prefill FFN GEMM at **4.10-4.44 TFLOP/s on M4 Pro (6.06-7.43× vs GGML)**. Its key design finding: **SME and the NEON cores give near-additive matrix throughput but share memory bandwidth**, so you want shape-aware operator placement across both plus offline weight packing.

**The repacking number**: Snapdragon X Elite, 12 threads, 7B-class, plain Q4_0 gives **63-66 t/s pp512 / 19-20 t/s tg128**; the i8mm-repacked layout gives **169-178 pp512 / 23-24 tg128**. That is **2.8× on prefill, 1.2× on decode**. Repacking buys compute; decode isn't compute-limited.

### 7.3 llama.cpp CPU internals

The `ggml_type` enum has outgrown the classic set: **TQ1_0=34, TQ2_0=35, MXFP4=39, NVFP4=40, Q1_0=41, Q2_0=42** (the "1-bit quantization" of the April 2026 releases). The old `Q4_0_4_4`/`Q4_0_8_8` file types are gone, replaced by an **online repack path** (`repack.cpp`) exposed as extra buffer types; repack variants now cover q4_0, q4_K, q2_K, q5_K, q6_K, q8_0, q8_K, iq4_nl and mxfp4 in 4x4/4x8/8x8/16x1 interleavings, a big expansion over the ARM-only Q4_0 story of 2024. Arch kernels live under `arch/{arm,x86,riscv,powerpc,s390,wasm,loongarch}`.

The dequant+dot fusion is the `vec_dot` design: **weights stay block-quantized, activations are quantized to Q8_0/Q8_K on the fly**, and the kernel issues `vpdpbusd`/`sdot`/`smmla` directly on packed nibbles, applying per-block FP16 scales at the accumulator. Repack simply pre-interleaves 4 or 8 rows so one SIMD instruction feeds a tile.

**Threading is the weak point, and the fix is documented.** `ggml_barrier` is a global atomic spin barrier hit once per graph node. Arm's Neoverse N2 study on a 2×64-core box: baseline token generation peaked at **26.52 t/s at 32 threads and collapsed past 64 threads** as cross-NUMA atomics serialized. A **hierarchical barrier** (NUMA-local atomics, only the last thread per node syncs globally) gave **41.15 t/s at 40 threads, +55%**. The telemetry shows why, unoptimized at 128 threads, node 0 delivered **70.6 GB/s while node 1 delivered 0.2 GB/s**; optimized, both ran 72-74 GB/s. Same study: **96 threads best for prompt processing, 24 for token generation**, separate the two thread pools if designing fresh.

### 7.4 Lookup-table kernels for very low bit widths

**T-MAC** replaces mpGEMM with bit-serial `tbl`/`pshufb` lookups so cost scales *linearly with bit-width* rather than paying dequantization. Surface Laptop 7, BitNet-3B: **20 tok/s on one core, 48 on four** (~4-5× llama.cpp). Llama-2-7B W2: 9.3 → 28.4 tok/s (1 → 4 cores). Raspberry Pi 5, 3B BitNet: **11 tok/s**. Jetson AGX Orin, Llama-2-7B W2: T-MAC on CPU **15.62 tok/s at 10.4 W (0.66 J/token)** vs llama.cpp CPU 7.08 tok/s at 15 W (2.12 J/token) vs CUDA 20.03 tok/s at 30.8 W (1.54 J/token): *the CPU wins on energy per token.*

**bitnet.cpp** (ACL 2025) gives the cleanest comparison. i7-13700H: 700M model **125.4 (I2_S) / 127.0 (TL2) / 75.6 (TL1) / 76.3 (T-MAC) / 114.2 (TQ1_0)** tok/s; 7B **20.6 / 20.7 / 11.9 / 12.3 / 18.0**; 100B **1.65 / 1.69 / 0.75 / 0.73 / 1.48**. M2 Ultra: 700M 238 (I2_S), 100B 6.50/7.45. Overall **2.37-6.17× on x86, 1.37-5.07× on ARM**, perplexity 11.29-11.30 vs FP16 11.29 (lossless). Two lessons: **LUT wins on ARM/wide-SIMD but plain unpack+MAC (I2_S) is competitive or better on x86**, and **throughput saturates at ~4 threads** on a laptop because bandwidth is the wall. **Vec-LUT** (arXiv 2512.06443) fixes T-MAC's scalar-lookup weakness with a vectorized 1→N lookup across parallel tokens: **up to 4.2×**, now in llama.cpp.

### 7.5 Memory system and the batch-1 CPU roofline

An 8B at Q4_K_M is ~4.9 GB, so **tok/s ≈ effective BW / 4.9 GB**.

- **Dual-channel DDR5-6000**: 96 GB/s theoretical, real STREAM triad 60-75% → ~60-75 GB/s → **~12-15 tok/s ceiling**, matching observed 10-15 tok/s on i7/Ryzen 7 desktops and 11-12 on a 9950X.
- **DDR5 4800 → 6000 MT/s yields +20.3% (Mistral 7B) / +23.0% (Llama-3.1-8B)** decode speedup, near-linear in bandwidth, clean proof that **core count is nearly irrelevant** for CPU decode.
- **12-channel EPYC Turin DDR5-6000**: 576 GB/s theoretical, ~99% on pure reads but STREAM ADD/TRIAD only **348-356 GB/s (~62%)**. Plan with the triad number.

Arithmetic intensity at batch 1 with Q4 weights is ~**3.6 FLOP/byte**. The ridge for a 16-core AVX2 desktop (~1 TFLOPS, 70 GB/s) is ~14 FLOP/byte; for a 32-core AMX socket ~800 ops/byte. **You need batch ≈ 4-8 before AVX-512 matters and batch ≈ 200+ before AMX matters.** That single calculation justifies both the AMX-prefill/AVX-512-decode split and the 2.8×-vs-1.2× repack asymmetry.

Practical knobs: pin to physical cores (skip SMT siblings and E-cores); prefer first-touch node-local allocation with per-node thread pools over blanket `numactl --interleave=all`; 2 MB huge pages to cut TLB misses on multi-GB weight sweeps; software-prefetch one block ahead in `vec_dot`; keep activations resident so you stream only weights.

### 7.6 CPU+GPU hybrid MoE: the consumer big-model story

The dominant 2025/26 pattern: **attention and dense weights on GPU, MoE expert FFNs in host RAM on CPU** (llama.cpp `-ot ".ffn_.*_exps.=CPU"`, or `--n-cpu-moe N`).

- **KTransformers, DeepSeek-R1 671B on a single RTX 4090D (24 GB) + Xeon Gold 6454S + 1 TB DDR5-4800**: **prefill up to 286 tok/s, decode ~14 tok/s**; paper claims **4.62-19.74× prefill and 1.25-4.09× decode** overall. Their CPU kernels inside SGLang (8×L20, DeepSeek-R1-0528 FP8) give 227.9 tok/s total / 87.6 output, median ITL 299 ms: **prefill up to 20×, decode up to 4×** vs baseline.
- **NUMA matters enormously** because expert selection is data-dependent and unprefetchable: KTransformers reports **up to 63% decode gain from NUMA-aware placement** on dual-socket and **1.45× from expert deferral** (accuracy delta <0.5%).
- **ik_llama.cpp**, DeepSeek-R1 Q4_K_XL on Xeon 8480 + 8×48 GB DDR5-4800 + RTX Pro 6000: **~140-150 t/s prefill, ~15 t/s generation** stable to 32K (`-fa -fmoe -rtr -mla 1 -ctk q8_0`).
- **The consumer scaling curve** (gpt-oss-20b MXFP4, RX 7900 XT + Ryzen 9 5900X): **94 t/s fully on GPU → 60 (4 MoE layers on CPU) → 38 (8 layers) → 20 t/s (all MoE on CPU)**. Contrast Qwen3-235B fully resident on 2× RTX Pro 6000: >1000 t/s prefill, 50-58 t/s generation. **The entire hybrid delta is host DRAM bandwidth.**

### 7.7 2026 CPU papers worth reading

**Litespark Inference** (arXiv 2605.06485) is the most transferable: for ternary models, **store weights as INT8 rather than packed 2-bit** so `sdot`/`vpdpbusd` consume them directly, with per-row symmetric INT8 activation quant and precomputed column sums for zero-point correction. On M4: **20.4 vs 0.39 tok/s (52×) over PyTorch, TTFT 288 ms vs 2632 ms, 556 MB vs 7673 MB**. Versus bitnet.cpp v2 it's +1.19× prefill on Xeon, −1.26× on EPYC, decode within 5%, the ternary CPU field has converged and remaining wins are packing/layout, not instruction choice. Corollary: **don't over-pack low-bit weights**, below ~2 bits the unpack cost can exceed the bandwidth saved.

Also: **CAT-Q** (2606.26650, ternary quantization emitting TQ1_0/TQ2_0), **Arm codebook kernels** (2501.00032, groupwise non-uniform 4-bit, **3-3.2× prefill / 2× decode over llama.cpp** on Arm), and **PALUTE** (2606.08891) / **LUT-LLM** (2511.06174) continuing the LUT line into PIM and FPGA.

---

## 8. Non-NVIDIA GPU backends

### 8.1 Apple / Metal / MLX

**The M5 discontinuity.** M5 (late 2025) put **Neural Accelerators**, matrix units, inside each GPU core, exposed via Metal 4's `MetalPerformancePrimitives` tensor ops. Apple's own MLX measurements, base M5 vs base M4: **TTFT 3.33-4.06× faster** across Qwen 1.7B/8B/14B/30B-A3B and gpt-oss-20B, but decode only **1.19-1.27×**, tracking the bandwidth bump 120 → 153 GB/s (+28%). Exactly the roofline story: **prefill got a 4× hardware gift; decode is still pure bandwidth.** llama.cpp PR #16634 moved matmul onto MPP tensor ops. Mistral-7B on M5 **pp512 415.5 → 846.7 t/s (~2×)**, and it is **disabled on M4 and earlier**, where hand-written `simdgroup_matrix` kernels are still faster. A Metal backend therefore needs **two matmul paths**.

`ggml-metal` uses per-quant-type templated kernels whose names encode rows-per-simdgroup and simdgroups-per-threadgroup, dequantizing into threadgroup memory before the simdgroup matmul; Metal 4's tensor API lets you dequantize straight into registers, skipping the barrier.

**MLX** design points worth copying: unified memory (no `.to(device)`), **lazy graph construction** evaluated at `mx.eval()`, `mx.compile` for JIT fusion, and `mx.fast.metal_kernel` letting user Metal source participate in the lazy graph. Quantization is affine 2/3/4/5/6/8-bit with configurable group size (default **64**), plus MXFP4/MXFP8 (group 32) and NVFP4 (group 16); the Metal backend dispatches **qmv / qvm / qmm** by shape and alignment, accumulating in fp32 even for fp16 models. MLX's footprint beats GGUF by 7-13% (Qwen3-Coder-30B-A3B: 34.7 vs 40 GB).

**Serving on Apple, 2026**: `vllm-project/vllm-metal` is an official vLLM hardware plugin using MLX as compute backend; a **unified paged varlen Metal attention kernel** is the default. The vllm-mlx paper on **M4 Max (546 GB/s)**: Qwen3-0.6B 525 t/s, Qwen3-4B 159, Qwen3-8B 93.3, Nemotron-30B 121.8: **21-87% over llama.cpp**; continuous batching takes Qwen3-0.6B from 441 t/s (1 req) to **1642 t/s (16 concurrent)**. Ollama's MLX preview: Qwen3.5-35B-A3B NVFP4 at **1810 t/s prefill / 112 t/s decode** vs 1154/58 on GGML.

**Framework overhead is measurable, and this is the most useful evidence in this section.** **BaseRT** (arXiv 2607.00501) is a raw-Metal C++ runtime: zero-allocation decode loop, pre-allocated KV, fused RoPE/norm/attention, per-chip threadgroup-size tables, FlashAttention with online softmax. On **M4 Pro** (273 GB/s), tg128 Q4: Qwen3-0.6B **464.5** vs llama.cpp 297.4 (1.56×) vs MLX 343.6 (1.35×). But **on dense prefill all three converge within ±6%**, everyone saturates the matmul units. Gains concentrate in (a) small models where per-token dispatch overhead dominates and (b) **MoE prefill (1.78-1.81× over MLX/llama.cpp)**, where many small expert GEMMs expose launch overhead. *Runtime shape, not kernel math, is the remaining headroom.*

Rules of thumb (Q4): M1/M2 base ~12-17 t/s on 8B; M3 ~25; M4 Pro ~40-50; M4/M5 Max ~90; 70B Q4 needs ≥48 GB, ~12-18 t/s on M4 Max, **25-30 t/s on M3 Ultra (819 GB/s)**. llama.cpp Metal LLaMA-7B Q4_0: **M4 Max 885.7 pp / 83.1 tg**, M3 Max 759.7 / 66.3. **MLX distributed** now offers MPI, a TCP ring (often faster than MPI), NCCL, and **JACCL. RDMA over Thunderbolt 5** (macOS 26.2+).

### 8.2 AMD

**HipKittens** (arXiv 2511.08083) is the essential document for AMD kernel authors. CDNA3/4 constraints: 64-thread waves; **512 32-bit VGPRs per SIMD split 256 VGPR / 256 AGPR** with *static* allocation (so NVIDIA-style producer/consumer warp specialization wastes registers on the producer); MFMA shapes like 16×16×32; **8 XCD chiplets × 32 CUs each with a non-programmable 4 MB L2**. Their answer is two wave schedules: **8-wave ping-pong** and **4-wave interleave**, instead of warp specialization. MI355X: **BF16 GEMM 1610 TFLOPS vs AITER hand-written assembly 1169 (1.38×)**, FP8 GEMM 3222, GQA attention backward **1091 vs 272-384 (2.3-4×)**; Triton is 1.3-3.0× slower on GEMM. Chiplet-aware scheduling is decisive: 79% L2 / 93% LLC hit → **18.3 TB/s**, vs naive row-major 36%/76% → 10.7 TB/s.

**AITER** is AMD's cuDNN analogue (354+ precompiled assembly kernels; `VLLM_ROCM_USE_AITER=1`): MI300X block-scale GEMM 2×, fused MoE 3×, **MLA decode 17×, MHA prefill 14×**, DeepSeek-V3/R1 end-to-end 6485 → 13704 tok/s. vLLM's ROCm attention post documents 7 backends and 3-path routing; on Qwen3-235B, ROCM_AITER_FA is **2.7-4.4× faster than legacy ROCM_ATTN**, and a shuffled KV layout `[blocks, heads, head_dim//x, block_size, x]` adds **15-20% decode**. ROCm on MI300X/MI355X reaches ~90-95% of H100 vLLM throughput.

**Consumer AMD is different.** llama.cpp HIP (Llama-7B-Q4_0, pp512/tg128): MI300X 11476/233, **7900 XTX 3552/167**, 9070 XT 4904/97. Vulkan on the same cards: **7900 XTX 3532/191, 9070 XT 5036/137**. Vulkan **already beats ROCm on token generation for RDNA3/4 and on prefill for RDNA4**. vLLM has no native gfx1201 kernels and silently falls back to FP32 dequant, so llama.cpp+Vulkan (62 t/s) beat vLLM+ROCm (48 t/s) on a 9070 XT.

### 8.3 Vulkan as a default

Vulkan went from compatibility fallback to the recommended backend on non-NVIDIA consumer hardware. Mechanics: `vulkan-shaders-gen` cross-products quant types × extensions into SPIR-V variants; **`GL_KHR_cooperative_matrix`** (2023, broad support) for tiled matmul; **`GL_NV_cooperative_matrix2`** (Oct 2024) is richer and enables Vulkan flash attention, historically NVIDIA-only, but **Mesa 26.1 now advertises it on Intel**; **`GL_EXT_integer_dot_product`** powers `mul_mmq` integer matmul, which is what makes Adreno/Mali viable at all.

Reference sweep (7B Q4_0, pp512/tg128): RTX 5090 10382/264, 7900 XTX 3532/191, Intel Arc Pro B70 3379/112, **M3 Ultra via MoltenVK 1117/116** (far below native Metal), Ryzen AI Max+ 395 1289/54, Mali-G57 ~0.8 t/s. Against CUDA on NVIDIA, Vulkan still loses ~20-30% on average, it is the *portability* answer, not the NVIDIA answer.

### 8.4 Intel

Intel **archived IPEX-LLM in January 2026**. On Arc, Vulkan generally beats SYCL (~2×, and 40% above the IPEX-LLM SYCL build); B580 does ~504 t/s pp and ~30-42 t/s tg on 8B Q4_K_M, roughly RTX 3060 class at $250. Mesa 26.1's coopmat2 landing was transformative on Arc Pro B70 with Qwen3.6-35B-A3B: **tg128 37.8 → 76.0 t/s (2×)**, 8-stream aggregate **Vulkan 176 vs SYCL 100 t/s**. SYCL is catching up in spots (a Q8_0 reorder pushed bandwidth utilization **21% → 66%, a 3× throughput gain**) and DPAS/XMX intrinsics are replacing DP4A, but the SYCL backend still lacks flash attention and MMQ kernels.

### 8.5 WebGPU

**Subgroups shipped in Chrome 134** (Mar 2025; Google Meet measured **2.3-2.9×** on matrix-vector shaders). **Subgroup matrices have still not shipped in stable Chrome as of Chrome 152 (Aug 2026)**, that gap defines browser performance. LlamaWeb (arXiv 2605.20706, 16 devices / 8 vendors) quantifies it: hand-written WGSL with register tiling, templated device-specialized shaders and dequant fused into kernels across 23 GGUF formats achieves **decode 45-69% faster than WebLLM/Transformers.js, up to ~100 tok/s**, but **prefill 21-51% slower**, precisely because subgroup matrices are unavailable. Native CUDA/Metal beat WebGPU 2-10× on prefill; yet WebGPU beat HIP on AMD decode and SYCL on Arc prefill by 23%. Per-vendor tuning bought **41% average kernel speedup**; mobile GPUs manage 4-17 tok/s decode.

### 8.6 Cross-platform strategy

- **Per-backend hand kernels win, but only for the matmul/attention core**, budget ~8-15 kernels per backend, not hundreds. BaseRT's advantage evaporates on dense prefill; HipKittens shows AMD needs different *algorithms*, not ported NVIDIA patterns.
- **Vulkan is the highest-ROI single portable backend for consumer hardware**, beats ROCm on RDNA3/4 decode, beats SYCL on Arc, works on Windows with no vendor toolkit. Two weaknesses: MoltenVK on Apple leaves ~4-5× unused vs native Metal, and mobile drivers are a lottery.
- **Metal must be native and M5-aware** (two matmul paths).
- **A tile DSL is a credible middle path.** TileLang covers CUDA SM70-SM120, ROCm CDNA3/4 and RDNA3/3.5/4, **Metal 4 cooperative tensors**, WebGPU, and LLVM CPU, the widest span available. Expect ~70-80% of hand-tuned.
- **Runtime shape beats kernel micro-optimization for decode**, zero-allocation loops, pre-allocated KV, amortized sync, fusion. All backend-agnostic, worth 1.15-1.56× before you touch a shader.

---

## 9. Measurement: profiling, rooflines, benchmarks, and reference numbers

### 9.1 Profiling workflow

**Two-tool discipline: `nsys` for orientation, `ncu` for the 1-3 kernels it fingers.** The rule of thumb quoted in practice: most inference slowdowns come from 2-5 kernels a 10-minute profile would find.

vLLM's documented invocation is a good template:
```
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node <script>
# for a running server, add: --capture-range=cudaProfilerApi --capture-range-end repeat
```
Two flags that are easy to miss. `spawn` is required because nsys does not survive the default fork. **`--cuda-graph-trace=node` is essential**: without it a CUDA-graph engine shows one opaque blob instead of per-kernel timings, the failure mode you hit after graph capture is working.

`torch.profiler` (vLLM: `--profiler-config`, view at ui.perfetto.dev): **send only a few requests; flushing takes ~10 minutes for 100 requests on an H100.** Add cProfile+snakeviz for the Python scheduling path, which on a batch-1 engine is often where the time actually is.

`ncu --set full` is required for a roofline plot. Read metrics in this order: **SOL Memory % vs SM %** (a good decode kernel wants Memory % near 100 and SM % low; *both* low means you are launch- or latency-bound, not bandwidth-bound), then **DRAM throughput** against spec, then achieved occupancy. Don't chase occupancy on a bandwidth-bound GEMV; chase bytes moved and memory requests in flight.

**CUDA events vs wall clock: measure both, and the gap is your budget.** Events exclude launch and queueing overhead, precisely what you're hunting at batch 1. A large events-vs-wall delta *is* your launch and CPU scheduling cost and is the quantitative argument for CUDA graphs or a megakernel. Non-NVIDIA: `rocprof` (ROCm), Xcode Instruments / Metal System Trace + the Metal shader profiler (Apple), RGP (AMD consumer), `VK_EXT_calibrated_timestamps` (Vulkan).

### 9.2 Roofline

```
decode tok/s ≈ effective bandwidth (GB/s) / bytes read per token
bytes read per token ≈ active parameters × bytes/param + KV bytes touched
```
For MoE, use **active** parameters, that's why a 30B-A3B MoE outruns a dense 8B.

What real engines achieve against that ceiling (measured, primary sources):

| Setup | Measured | Roofline | Efficiency |
|---|---|---|---|
| Qwen3-8B **4-bit**, M-series (400 GB/s) | 64.9 tok/s | 87 | **75%** |
| Qwen3-8B **8-bit** | 40.0 | 46 | **87%** |
| Qwen3-8B **bf16** | 22.6 | 24 | **93%** |
| gpt-oss-20b, RTX 5090, tuned llama.cpp | 419 | ~520 | **~80%** |

**The efficiency gradient is the research opportunity.** Efficiency climbs 75% → 87% → 93% going 4-bit → 8-bit → bf16, because heavier weights amortize the *fixed* costs (dequant ALU work, launches, sampler, Python) over more bytes. **The 4-bit case leaves ~25% unused, and that gap is dequant-bound, not bandwidth-bound.** Closing it is the most defensible single research target in this document.

### 9.3 Benchmarks and metric hygiene

Definitions, stated precisely:
- **TTFT**, request send → first streamed output.
- **TPOT**, per request, `(e2e − TTFT) / (output_tokens − 1)`, then aggregated.
- **ITL**, time between consecutive *streamed outputs*. With standard decoding ITL ≈ TPOT, **but under speculative decoding one streamed output can contain several tokens, so they diverge**. Quoting ITL as TPOT under speculation is the most common measurement error in this field.
- **Goodput**, completed requests/s that also meet latency SLOs. Throughput without goodput is a vanity metric; you can always raise tok/s by making everyone wait.

Tools: **`llama-bench`** (`pp512`/`tg128`, the canonical prefill/decode split; `-fa 1`), **`vllm bench serve`** (TTFT/TPOT/ITL percentiles plus SLO goodput) and `benchmark_latency`, SGLang's `bench_serving`/`bench_one_batch`, and **LocalScore** (Apache-2.0, Llamafile-based, community DB by GPU; score 1000 = excellent, 250 = passable, <100 = poor UX).

**MLPerf Inference**: The September 2025 round added an **interactive scenario for Llama-3.1-405B: 12.5 TPS/user with a 4.5 s TTFT bound**, versus the server scenario's much laxer TTFT 6000 ms / TPOT 175 ms. That tightening is MLPerf conceding aggregate throughput measured the wrong thing. The 2026 round, AMD MI355X single node: Llama-2-70B **103,480 tok/s offline / 100,282 server / 73,608 interactive**, interactive costs ~29% of offline. **Llama-2-70B improved 4.4× offline / 4.8× server over the September 2025 round, primarily from FP4.**

### 9.4 Reference numbers to design against (2026)

**RTX 5090** (32 GB GDDR7, 1792 GB/s), llama.cpp `llama-bench`, Q4_K_XL, @4K ctx:

| Model | Size | pp tok/s | tg tok/s | Max ctx |
|---|---|---|---|---|
| Qwen3 8B | 4.78 GB | **10,406** | **185.9** | 131K |
| Qwen3 14B | 8.53 GB | 6,497 | 123.8 | 131K |
| **Qwen3 MoE 30B-A3B** | 16.47 GB | 6,630 | **234.3** | 147K |
| Qwen3 32B | 18.64 GB | 2,931 | 61.4 | 45K |
| GPT-OSS 20B | ~15 GB | 9,443 | 112.0 | 131K |

Two things to internalize. **The prefill:decode ratio is ~56:1 for the 8B**, that is the GEMM/GEMV intensity gap made concrete, and it is why prefill and decode need different kernels and different optimization budgets. And **the 30B MoE decodes at 234 tok/s, faster than the dense 8B despite being 3.4× larger**, because only ~3B params are active. *For a consumer research engine, sparse-MoE decode is where the wins are.*

**llama.cpp CUDA optimization work** (`GGML_CUDA_GRAPH_OPT=1`), showing what's still recoverable:

| Model | GPU | Baseline | Optimized | Gain |
|---|---|---|---|---|
| Qwen3 30B | 5090 | 246.96 | **352.06** | **+42.6%** |
| GPT-OSS 20B | 5090 | 329.23 | 419.14 | +27.3% |
| Qwen3 30B | 4090 | 198.39 | **271.04** | **+36.6%** |
| GPT-OSS 20B | 4090 | 232.05 | 271.99 | +17.2% |

The techniques are directly reusable: **GEMV fusion for gated activations** (`σ(W_gate·x) ⊙ W_up·x` as one kernel), **fused TopK-MoE expert selection**, **RMSNorm fused with the following mul/add**, and **concurrent CUDA streams for independent Q/K/V projections** (three independent GEMVs that a single stream serializes). Note the **5090 gains exceed the 4090's**, more bandwidth means fixed software overhead is a larger share of the step. *The faster the hardware, the more the software overhead matters.*

**Apple Silicon**, llama.cpp Metal, LLaMA-7B Q4_0 (pp512/tg128): **M4 Max 885.7 / 83.1**, M3 Max 759.7 / 66.3 (M4 Max +16.6% pp, +25.3% tg). Llama-3-8B Q4 on M4 Max ~75 tok/s. Note the **pp:tg ratio is only ~11:1** here versus the 5090's ~56:1. Apple has far less compute relative to bandwidth. **Prefill is the weak spot on Apple Silicon; decode is the weak spot on discrete NVIDIA.** Optimize per backend accordingly.

**Unified-memory boxes**: DGX Spark (GB10, ~273 GB/s) and Strix Halo / Ryzen AI Max+ 395 (~256 GB/s) land within 13% of each other on *decode* (gpt-oss-120B: 38.6 vs 34.1 tok/s) but **~5× apart on prefill** (~1700 vs ~340 tok/s): the same compute-vs-bandwidth split, now as a purchasing decision.

**Multi-GPU on consumer**: PCIe 4.0 is 64 GB/s vs NVLink 4.0's 900 GB/s. Dual-GPU tensor parallel on PCIe runs at roughly **0.70-0.75× scaling** (25-30% comm overhead); PCIe 5.0 roughly halves that to 6-7% for a dual-GPU 70B. llama.cpp's own docs are more sober than the blog coverage: `--split-mode layer` (pipeline, default) is best when the model doesn't fit and you want fast prefill and tolerate slow interconnect; `row` is **deprecated**; **`tensor` is experimental**, requires flash attention and non-quantized KV, is not implemented for MoE or state-space architectures, and works best on multiple NVIDIA GPUs with CUDA. Treat "3-4× from tensor parallelism" headlines with caution.

---

## 10. Language and runtime choices for a new engine

There is no single answer, but the design space has clarified a lot since 2024.

**C++/CUDA (llama.cpp, TensorRT-LLM, ggml).** Still the surveyed path to a zero-overhead decode loop with no runtime dependencies, a single binary, and control of every allocation. It's what you want if the engine must run on a user's laptop. Cost: you write and maintain every kernel per backend, and you have no ecosystem of pretrained model plumbing. llama.cpp is the existence proof that this is tractable, but note it has ~10 backends and years of contributor-hours in them.

**Rust.** `candle` (HuggingFace) and `mistral.rs` are the mature options; `mistral.rs` is competitive with llama.cpp on raw speed and supports GGUF/HF formats with GPTQ/AWQ/ISQ quantization. Rust buys you memory safety, no GC, fearless concurrency for the scheduler/server layer, and excellent CLI/server ergonomics, all real advantages for the *engine* half. It does **not** solve the kernel problem: you still write CUDA/Metal/HIP kernels in C, or bind to cuBLAS/cuDNN, or emit them from a DSL. The pragmatic Rust architecture is **Rust for the runtime, scheduler, server, tokenizer, memory management; C++/CUDA/Metal for kernels; a thin FFI seam**. The ecosystem gap versus llama.cpp is in model coverage, not speed.

**Python + Triton (vLLM, SGLang, PyTorch).** Fastest path to a *research* engine: you get autograd-free eager execution, `torch.compile`, Triton kernels that are ~70-90% of hand-tuned CUDA and portable to ROCm/Intel, FlashInfer/FlashAttention as libraries, and every model implementation for free. The costs are precisely the ones vLLM has spent three years engineering away: Python on the hot path, CPU-GPU syncs, dynamic shapes fighting CUDA graphs, and a large dependency surface. vLLM Model Runner V2's answer: **move input preparation into Triton kernels and treat async as a design constraint**, is the state of the art here and worth copying wholesale.

Notably, **FlashAttention-4 is written in CuTe-DSL embedded in Python** and compiles 20-30× faster than C++ CUTLASS templates. Python-hosted kernel DSLs are no longer a compromise for peak performance; they are where NVIDIA itself is going.

**Kernel DSLs, ranked by what they buy you:**
- **Triton**, the safe default. Portable NVIDIA/AMD/Intel; vLLM's Triton attention backend hit **100.7% of FA3 on H100 for long decode and ~5.8× the previous MI300 implementation from the same 800 lines** (versus FlashAttention-3's ~70,000). The counterexample: HipKittens measures Triton **1.3-3.0× slower than hand-tuned on AMD GEMM**. Warp specialization is landing (Hopper+).
- **Gluon / TLX**, lower-level Triton dialects exposing layouts, async copies, warp specialization explicitly.
- **Helion** (PyTorch): higher-level than Triton, compiles *to* Triton, autotunes more aggressively.
- **CuTe-DSL** (CUTLASS 4.x, Python): full control of layouts, TMEM, pipelines; the FA4 substrate; NVIDIA-only.
- **ThunderKittens**. C++ tile primitives; TK 2.0 (Jan 2026) added Blackwell, MXFP8, NVFP4. **HipKittens** ports the abstraction to CDNA3/4 and shows tile abstractions generalize but the *algorithms* must be rethought (ping-pong/interleave wave schedules instead of warp specialization, because AMD lacks dynamic register reallocation).
- **TileLang** (TVM-based): the widest hardware span: CUDA SM70-SM120, ROCm CDNA3/4 and RDNA3/3.5/4, **Metal 4 cooperative tensors**, WebGPU, LLVM CPU. ~20 lines for a GEMM. The best bet if you want one kernel source for Apple + AMD + NVIDIA.
- **Mojo/MAX**, the surveyed stack built entirely without CUDA, claiming portability across NVIDIA/AMD/Apple with a single kernel codebase and **1772 TFLOPS matmul on B200 (above cuBLAS)**. Compelling story; small ecosystem; verify claims yourself.

**Zig**: `zml` exists and Zig's comptime is attractive for generating specialized kernels and layouts, plus trivially good C interop and cross-compilation. But the ML ecosystem is thin enough that you'd be building on MLIR/XLA anyway.

**LLM-generated kernels are now a real tool**, not a curiosity: PyTorch's **KernelAgent** (March 2026) reports **100% correctness across all 250 KernelBench L1/L2/L3 tasks**, and the KernelBench-X / Kernel-Smith / FM-Agent line is active. Realistic use today: generate and autotune the long tail of pointwise/fusion kernels, hand-write the 5-10 that matter.

**Recommendation for a research engine targeting consumer hardware:** Rust or C++ for the runtime and scheduler with an explicit no-allocation, no-sync steady-state decode loop; **TileLang or hand-written kernels for the ~10 kernels that matter per backend** (quantized GEMV, quantized GEMM, attention prefill, attention decode, fused MoE, RMSNorm+quant, RoPE, fused SwiGLU, sampler, KV copy); Vulkan as the portability backend and native CUDA + native Metal as the two first-class ones; and a Python binding for research ergonomics that is *not* on the hot path.

---

## 11. A concrete build order

Ranked by (value at batch 1 on consumer hardware) ÷ (effort):

1. **Get the roofline instrumentation in first.** Report measured tok/s alongside `BW / bytes_per_token` and the efficiency ratio, per model, per quant, every run. You cannot optimize what you don't attribute.
2. **Zero-allocation, zero-sync steady-state decode loop.** Pre-allocated KV, pre-allocated buffers, sampled token written by the GPU into the next step's input buffer. Make "0 CPU↔GPU syncs per token" a test.
3. **Quantized GEMV kernel with fused dequant** using LOP3-style bit tricks and int8/`dp4a`/tensor-core inner products with on-the-fly activation quantization. Then measure, the 4-bit efficiency gap (75% vs 93% at bf16) lives here.
4. **CUDA graph capture with a tier ladder** (and the equivalent on Metal: `MTLIndirectCommandBuffer` / `MTLComputeCommandEncoder` reuse). Then measure the CUDA-events vs wall-clock gap.
5. **The measured fusions**: RMSNorm+quant, gated-MLP GEMV fusion, fused RoPE into QKV epilogue, fused TopK-MoE, residual+norm. Worth +17-43% on real hardware per llama.cpp's own numbers.
6. **Prefix caching** (radix tree, not exact-prefix): the single biggest end-to-end win for agentic and multi-turn local use, 20-95% hit rates.
7. **Speculative decoding** with an n-gram/prompt-lookup drafter first (free, no extra model), then EAGLE-3-style heads. This is the batch-1 regime where speculation pays maximally: 2.3× at batch 4, 3.2× measured at batch 1 on an M3 Max.
8. **Attention decode kernel with split-KV and GQA packing**, autotuned `num_splits`. Consider skipping paged KV entirely in favor of VMM-backed contiguous KV (vAttention) so stock kernels work.
9. **Fused sampler**, single launch, no host sync, rejection-sampling top-k/top-p.
10. **MoE-first**: sparse MoE decode is the standout consumer result (234 tok/s for 30B-A3B on a 5090, *faster than dense 8B*), and CPU-offloaded experts are how big models fit at all.
11. **Then, if you want a research contribution rather than a good engine: the megakernel.** Persistent kernel, on-GPU instruction interpreter, SMEM paging, dependency counters instead of kernel barriers. The measured headroom is real: 50% → 78% of memory bandwidth, 1.5-3.5× over vLLM/SGLang at batch 1. This survey found no strong megakernel for *quantized* models on *consumer* GPUs. That gap (megakernel × 4-bit weights × consumer SM89/SM120) is, as far as this survey can tell, unclaimed.

---

## Sources

**Attention kernels**
- FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling. https://arxiv.org/abs/2603.05451
- FlashAttention-4 gives the NVIDIA Blackwell platform its most optimized attention kernel yet (Lambda). https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet
- Writing Speed-of-Light Flash Attention for 5090 in CUDA C++. https://gau-nernst.github.io/fa-5090/
- FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. https://arxiv.org/abs/2501.01005
- FlashInfer (GitHub). https://github.com/flashinfer-ai/flashinfer
- Accelerating Self-Attentions for LLM Serving with FlashInfer. https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
- Run High-Performance LLM Inference Kernels from NVIDIA Using FlashInfer. https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/
- vLLM Triton Attention Backend Deep Dive. https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
- Attention Backend Feature Support (vLLM). https://docs.vllm.ai/en/latest/design/attention_backends/
- vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention. https://arxiv.org/abs/2405.04437 · https://github.com/microsoft/vattention
- FlashMLA. https://github.com/deepseek-ai/FlashMLA
- FlashMLA Hopper FP8 sparse deep dive. https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md
- DeepSeek-V3.2. https://arxiv.org/abs/2512.02556 · vLLM day-0. https://vllm.ai/blog/2025-09-29-deepseek-v3-2 · SGLang day-0. https://www.lmsys.org/blog/2025-09-29-deepseek-V32/
- Native Sparse Attention. https://arxiv.org/abs/2502.11089 · Triton impl. https://github.com/fla-org/native-sparse-attention
- FSA: An Alternative Efficient Implementation of Native Sparse Attention. https://arxiv.org/abs/2508.18224
- FlexAttention (PyTorch blog). https://pytorch.org/blog/flexattention/
- llama.cpp CUDA: use mma PTX instructions for FlashAttention (PR #11583). https://github.com/ggml-org/llama.cpp/pull/11583
- Flash Attention and Optimizations (llama.cpp DeepWiki). https://deepwiki.com/ggml-org/llama.cpp/8.2-flash-attention-and-optimizations
- FA3 kvcache + split kv + gqa parallelization (PR #1236). https://github.com/Dao-AILab/flash-attention/pull/1236

**GEMM / quantization kernels**
- MARLIN: Mixed-Precision Auto-Regressive Parallel Inference (PPoPP 2025). https://arxiv.org/abs/2408.11743 · https://research-explorer.ista.ac.at/download/19877/19883/2025_PPoPP_Frantar.pdf
- How Marlin pushes the boundaries of mixed-precision LLM inference. https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference
- Introducing Machete, a mixed-input GEMM kernel for Hopper. https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel
- DeepGEMM. https://github.com/deepseek-ai/DeepGEMM
- ggml CUDA backend / core operations. https://deepwiki.com/ggml-org/ggml/3.2-cuda-backend · https://deepwiki.com/ggml-org/ggml/3.2.1-cuda-core-operations
- ggml-cuda NVFP4 dp4a and MMQ kernels. https://github.com/ggml-org/llama.cpp/pull/20644 · https://github.com/ggml-org/llama.cpp/pull/21074
- Accelerating large language models with NVFP4 quantization. https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization
- CUTLASS Blackwell tutorials (tensor memory, block scaling). https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/ · https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/
- tcgen05 MMA Programming Guide. https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html
- The State of FP8 KV-Cache and Attention Quantization in vLLM. https://vllm.ai/blog/2026-04-22-fp8-kvcache
- Accelerating MoEs with a Triton Persistent Cache-Aware Grouped GEMM Kernel. https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- Fused MoE Kernel Features (vLLM). https://docs.vllm.ai/en/latest/design/moe_kernel_features/

**Megakernels / persistent kernels**
- Hazy Research, "No Bubbles" megakernel. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Compiling LLMs into a MegaKernel (Mirage Persistent Kernel). https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- Building a single-kernel, latency-optimized LLM inference engine on AMD MI300X (Kog). https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/
- Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs. https://arxiv.org/abs/2604.15379

**Scheduling / serving / overheads**
- Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI '22). https://www.usenix.org/system/files/osdi22-yu.pdf
- vLLM V1: A Major Upgrade to vLLM's Core Architecture. https://vllm.ai/blog/2025-01-27-v1-alpha-release
- Inside vLLM: Anatomy of a High-Throughput LLM Inference System. https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- Model Runner V2: A Modular and Faster Core for vLLM. https://vllm.ai/blog/2026-03-24-mrv2
- CUDA Graphs design (vLLM). https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md
- SGLang: Zero-Overhead Batch Scheduler. https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
- Zero-overhead batch scheduler notes. https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md
- SGLang issue #19347 (does the overlap actually happen?). https://github.com/sgl-project/sglang/issues/19347
- Disaggregated Prefilling (vLLM docs). https://docs.vllm.ai/en/stable/features/disagg_prefill/
- Beyond the Buzz: A Pragmatic Take on Inference Disaggregation. https://arxiv.org/abs/2506.05508
- vLLM Now Supports Qwen3-Next: Hybrid Architecture. https://vllm.ai/blog/2025-09-11-qwen3-next
- Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC. https://arxiv.org/abs/2604.07609

**Sampling / structured output / speculation**
- Sorting-Free GPU Kernels for LLM Sampling. https://flashinfer.ai/2025/03/10/sampling.html
- vLLM PR #15478 (faster top-k). https://github.com/vllm-project/vllm/pull/15478
- XGrammar. https://arxiv.org/abs/2411.15100 · XGrammar-2. https://arxiv.org/html/2601.04426v2
- llguidance. https://github.com/guidance-ai/llguidance
- Guided Decoding Performance on vLLM and SGLang (SqueezeBits). https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang
- EAGLE-3. https://arxiv.org/abs/2503.01840
- DSpark in SGLang. https://www.lmsys.org/blog/2026-07-06-dspark-sglang/ · https://arxiv.org/html/2607.05147v1

**CPU**
- KTransformers (SOSP '25). https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf · https://dl.acm.org/doi/10.1145/3731569.3764843
- Accelerating Hybrid Inference in SGLang with KTransformers CPU Kernels. https://www.lmsys.org/blog/2025-10-22-KTransformers/
- kt-kernel. https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel
- T-MAC. https://arxiv.org/abs/2407.00088 · https://github.com/microsoft/T-MAC/
- bitnet.cpp. https://arxiv.org/html/2502.11880v1 · https://aclanthology.org/2025.acl-long.457.pdf
- Vec-LUT. https://arxiv.org/abs/2512.06443
- Litespark Inference (ternary SIMD CPU). https://arxiv.org/abs/2605.06485
- Demystifying ARM SME to Optimize GEMM. https://arxiv.org/html/2512.21473v1
- SMEPilot. https://arxiv.org/html/2606.16332
- llamafile / tinyBLAS upstream (PR #6414). https://github.com/ggml-org/llama.cpp/pull/6414
- Cross-NUMA optimization in llama.cpp on Neoverse N2 (Arm). https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/introduce-the-cross-numa-problem-and-optimization-in-llama-cpp-with-llama3-model-running-in-neoverse-n2
- KleidiAI + SME2 in llama.cpp. https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/performance_llama_cpp_sme2/kleidiai_integration/
- ik_llama.cpp. https://github.com/ikawrakow/ik_llama.cpp
- Intel AMX technical brief. https://cdrdv2-public.intel.com/785250/Intel-AMXBrief-Final-3.17.pdf
- DDR5 speed and LLM inference. https://dev.to/maximsaplin/ddr5-speed-and-llm-inference-3cdn

**Non-NVIDIA GPUs**
- Exploring LLMs with MLX on M5 (Apple). https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- llama.cpp Metal tensor ops for M5 (PR #16634). https://github.com/ggml-org/llama.cpp/pull/16634
- BaseRT: raw-Metal runtime. https://arxiv.org/pdf/2607.00501
- Native LLM and MLLM Inference at Scale on Apple Silicon (vllm-mlx). https://arxiv.org/html/2601.19139v2 · https://github.com/vllm-project/vllm-metal
- MLX custom Metal kernels. https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html · quantization. https://deepwiki.com/ml-explore/mlx/7.1-quantization-api-and-modes
- MLX distributed (incl. JACCL over Thunderbolt 5). https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- HipKittens: Fast and Furious AMD Kernels. https://arxiv.org/abs/2511.08083
- vLLM ROCm attention backends. https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- AITER. https://hyper-accel.github.io/en/posts/rocm-aiter/
- llama.cpp Vulkan benchmarks. https://github.com/ggml-org/llama.cpp/discussions/10879 · ROCm benchmarks. https://github.com/ggml-org/llama.cpp/discussions/15021
- llama.cpp VK_KHR_cooperative_matrix (PR #10597). https://github.com/ggml-org/llama.cpp/pull/10597
- LlamaWeb (WebGPU). https://arxiv.org/html/2605.20706
- WebGPU release notes (Chrome). https://developer.chrome.com/blog/new-in-webgpu-146

**Tools, DSLs, measurement**
- ThunderKittens. https://arxiv.org/abs/2410.20399 · https://github.com/HazyResearch/ThunderKittens
- TileLang. https://github.com/tile-ai/tilelang
- Helion. https://github.com/pytorch/helion
- Warp Specialization in Triton: Design and Roadmap. https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/
- KernelAgent. https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/
- KernelBench-X. https://arxiv.org/html/2605.04956v1
- Modular MAX. https://www.modular.com/open-source/max
- mistral.rs. https://github.com/ericlbuehler/mistral.rs
- Profiling vLLM. https://docs.vllm.ai/en/stable/contributing/profiling/
- Optimizing Token Generation in llama.cpp's CUDA Backend. https://github.com/ggml-org/llama.cpp/discussions/17621
- llama.cpp multi-GPU docs. https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- llama.cpp Apple Silicon benchmarks. https://github.com/ggml-org/llama.cpp/discussions/4167
- RTX 5090 LLM benchmark results. https://www.hardware-corner.net/rtx-5090-llm-benchmarks/
- LocalScore. https://www.localscore.ai/blog
- AMD MLPerf Inference 2026 submission. https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
- Tested: AMD's Strix Halo vs Nvidia's DGX Spark. https://www.theregister.com/on-prem/2025/12/25/tested-amds-strix-halo-vs-nvidias-dgx-spark/2098514

---

### A note on sourcing

Numbers from vendor blogs, arXiv papers, and project repositories were fetched directly and are quoted as reported by their authors, they are not independently reproduced here. Community benchmark tables (llama.cpp discussions, Hardware Corner, llmcheck) are representative rather than controlled. Where a secondary source made a strong claim I checked the primary: for instance, blog coverage of "3-4× from llama.cpp tensor parallelism" is not supported by llama.cpp's own `docs/multi-gpu.md`, which describes tensor split mode as experimental, flash-attention-only, non-quantized-KV-only, and unimplemented for MoE. Treat that pattern as the norm: the 2026 SEO-blog layer around inference performance is dense and frequently wrong.
