# Research inference engine for consumer hardware

*Compiled 2026-08-18 from seven reports in this directory (`01` through `07`). An offline second-opinion track contributed roofline math, MoE-union arithmetic, and the KV-rollback model. Its facts were about a year stale. Every report reconciled claims against live sources. See §9.*

## Table of contents

1. Decode as a byte-moving bound
2. Speed techniques
3. Memory techniques
4. Speculative decoding
5. Quantization in 2026
6. Engines, hardware, and workloads
7. Open research opportunities
8. Recommended architecture and build order
9. Method notes
10. Index of the detailed reports

---

## 1. Decode as a byte-moving bound

Memory traffic bounds single-stream decode throughput:

```
tok/s  ≲  η · memory_bandwidth / bytes_touched_per_token
```

`bytes_touched_per_token` = active weights + KV read for the active sequences + any weights that had to cross a slow wire (PCIe, DDR). Prefill is the opposite regime: compute-bound GEMM. Every technique in this synthesis is either (a) fewer bytes per token, (b) a faster wire for those bytes, (c) more useful work per byte (batching, speculation), or (d) closing the gap between measured η and 1.

Two measured facts anchor how much headroom exists (report `01` §2, `04` §1, `05` §0):

- **η falls as hardware gets wider.** From llama.cpp's community tables (Llama-2-7B Q4_0, batch-1): M1 80%, RTX 4080 76%, RTX 4090 71%, RTX 5090 62%, M2/M3 Ultra ~44%, A100 36%, H100 31%. An H100 delivers *less* batch-1 throughput than a 5090. This is the wave problem: a GEMV over a 4096² W4 layer is ~32 CTAs on a 128-SM GPU. Consumer GPUs therefore realize a higher fraction of peak at batch 1, and 25-40% of roofline remains unused.
- **4-bit paths realize ~75% of roofline vs ~93% for bf16.** The gap is dequant cost and fixed overhead (kernel boundaries, launches, host syncs), not bandwidth. Concretely: RTX 5090, Qwen3-8B Q4_K_XL: 10,406 pp / 185.9 tg tok/s vs a ~375 tok/s ceiling.

Corollaries that keep recurring across all seven reports:

- **Sparse MoE reduces active weight traffic.** Qwen3-30B-A3B decodes at 234 tok/s on a 5090, faster than the dense 8B, because ~3B params are active. Sparse MoE is common among models people run in mid-2026: gpt-oss-120b, Qwen3.x MoEs, GLM-5.x, Kimi K2.x, DeepSeek V3.x/V4, MiniMax M2/M3, Gemma 4 MoE. Decode speed depends on active weight bytes and the bandwidth of the memory that holds them.
- **Prefill, not decode, is the consumer pain point.** 5090 prefill:decode ratio ≈56:1 vs M4 Max ≈11:1; Strix Halo 340 tok/s prefill vs DGX Spark 1700 on the same model despite similar decode; an 84 s cold vs 1.0 s warm prompt on a Mac (an 80× prompt-cache swing). Agentic and coding workloads re-prefill constantly.
- **The faster the hardware, the more the software overhead matters.** llama.cpp's own CUDA fusion work gained +36-43% on the 5090 vs +17-27% on the 4090.

---

## 2. Speed techniques

Ranked roughly by value ÷ effort at batch 1-4 on consumer hardware (`05` §5.6, §11):

1. **4-bit (or lower) weights** divide bytes/token. See §5.
2. **CUDA graphs, then a megakernel.** Kernel boundaries cost 1.3-4.5 µs each *and* act as global barriers. Existing engines waste ~50% of bandwidth on them at low batch. Fixes, in order: a zero-allocation zero-sync steady-state decode loop (sampled token written by GPU into next step's input); CUDA graph capture with a small lattice of shape buckets; SGLang's Breakable CUDA Graph (capture with explicit eager breaks), which beat `torch.compile` piecewise (1.70× vs 1.45× over eager) at ¼ the code and 3.8-5.2× faster build. At the frontier, the **megakernel** (Hazy Research: whole forward pass as one persistent kernel with an on-GPU interpreter, 78% of bandwidth, 2.5× vLLM at batch 1). This survey found no megakernel for quantized models on consumer SM89/SM120. That intersection is unclaimed.
3. **Speculative decoding** raises arithmetic intensity without adding users. See §4. Realistic consumer gains 1.3-2.5×. It can and does go *negative* if the engine cannot detect a losing config.
4. **Measured fusions**, worth +17-43%: gated-MLP GEMV fusion (`σ(W_gate x) ⊙ W_up x` as one kernel), fused TopK-MoE, RMSNorm+following mul/add, RMSNorm+activation-quant, RoPE in the QKV epilogue, residual+norm, concurrent streams for the three independent Q/K/V GEMVs, fused sampler (top-k/top-p/min-p, single launch, no host sync).
5. **Quantized GEMV done right**: fused dequant using LOP3/`prmt` bit tricks, int8 `dp4a`/tensor-core inner products with on-the-fly activation quant, enough CTAs (split-K, narrow row tiles), pre-permuted layouts for coalesced loads. Kernel families to study: ExLlama (decode-first direct GEMV), Marlin (tensor-core GEMM kept useful down to small M), Machete (CUTLASS mixed-input). GEMV wins at M=1-4, small-M Marlin/Machete at M≈1-32, plain GEMM past ~M=32. "M" can be users, prefill tokens, or speculative tokens.
6. **Attention decode**: split-KV/FlashDecoding with autotuned `num_splits`, GQA head packing; FlashInfer's plan/run split as the model for capture-friendly attention. On Blackwell, FlashAttention-4 (CuTe-DSL) and SageAttention3 are the baselines. Consider skipping PagedAttention for VMM-backed virtually-contiguous KV (vAttention) so stock kernels work unmodified.
7. **Prefix caching** (radix tree, not exact-prefix): 20-95% hit rates on agentic/multi-turn; production Copilot data shows ~90% within a turn, ~55% across turns. Persistent on-disk prompt cache is the largest measured user-facing win (`07`).
8. **Continuous batching + chunked prefill** even for a single user. Agents fan out many small requests; DGX Spark goes 33.5 → 862 tok/s from c=1 to c=256. Async/overlap scheduling to keep the CPU off the hot path (vLLM V1 / SGLang zero-overhead scheduler).
9. **Backend per phase**: on Strix Halo, Vulkan wins decode 1.5× (98.7% of measured bandwidth) but loses prefill 2.7× to HIP. Vulkan now beats ROCm on RDNA3/4 decode generally.
10. **CPU side**: AVX-512/VNNI/AMX (ktransformers), ARM i8mm/SME2, LUT-based low-bit matmul (T-MAC, Vec-LUT, now in llama.cpp), NUMA-local shards, thread pinning, hugepages. Dual-channel DDR5 is ~60-75 GB/s real; 12-channel EPYC ~500 GB/s.
11. **Grammar-constrained decoding is hidden if overlapped** (XGrammar/llguidance mask computation is well under token time).
12. **Power/thermals**: power-limiting during bandwidth-bound decode is nearly free; sustained tok/J and p99 under thermal steady state are what users experience.

---

## 3. Memory techniques

Four levers (`04` §0): quantization, sparsity (MoE), residency (compute where the bytes already live), and KV architecture.

**Offload.** The canonical 2026 rig is heterogeneous: attention + shared experts + embeddings on GPU, routed `ffn_*_exps` in system RAM computed on CPU (`--n-cpu-moe`, `-ot "exps=CPU"`; llama.cpp's `--fit` auto-places). Key quantitative facts:

- Offload is not a cliff but a steep line: `T(f)/T(0) = 1 + f(B_g/B_c − 1)`; with B_g/B_c ≈ 11, moving 12.5% of a model to CPU halves throughput. gpt-oss-120b on a 20 GB GPU: 94 → 60 → 20 tok/s as 0 → 4 → 36 MoE layers move to CPU (reproduced exactly by a two-constant cost model, ~4.25 GB touched/token).
- **Never stream expert weights over PCIe at batch 1.** Time to move 1 GB: dual-channel DDR5 ≈11 ms, 8-channel ≈2.5 ms, PCIe 5.0 ≈21 ms, PCIe 4.0 ≈41 ms, GPU-resident ≈1 ms. Compute host-resident experts on the host (ktransformers' real win).
- **Prefetch has a hard deadline**: a 24 MB W4 expert needs ~1 ms over PCIe 4; an attention block gives ~300 µs of lead. "99% router prediction accuracy" claims must be re-scored as deadline- and byte-weighted recall. Expert predictability is a *per-model* property that trades off against load balancing (arXiv:2505.16056). Profile before building a predictor.
- Batching and speculative verification are MoE infrastructure: they raise effective batch so expert-major GEMMs and cache reuse pay.

**Unified-memory boxes**: Apple (M4 Max 546 GB/s, M3 Ultra ~819 spec but ~44% η at batch 1), Strix Halo 128 GB (~212 GB/s measured; Qwen3-235B Q3 at 16 tok/s), DGX Spark (273 GB/s, gpt-oss-120b 60 tok/s decode, ~1950 prefill). Similar decode, 5× apart on prefill.

**Multi-GPU without NVLink**: PCIe TP scales ~0.70-0.75× on PCIe 4; layer/pipeline split or expert-parallel (each device owns disjoint experts) maps far better; put the draft model on GPU 1 to avoid per-layer collectives; auto-detect negotiated PCIe width. llama.cpp `--split-mode row` is deprecated, `tensor` experimental and unimplemented for MoE.

**KV cache**: paged blocks with GPU block tables, COW/refcounts, prefix hashing; KV quantization (q8_0 usually free, but not universally: Gemma-4-MoE q8_0 KLD 0.377 vs Qwen3.6 <0.04, so precision policy must be per-architecture, ideally measured at load; asymmetric K/V precision; ExLlama's Hadamard-rotated Q4 cache); eviction/compression (StreamingLLM/H2O/SnapKV/PyramidKV/KVzip). Those eviction methods fail on multi-turn and reasoning-faithfulness axes while aggregate accuracy looks fine (SCBench). MLA (68.6 KiB/token vs GQA's 320) and post-hoc MLA conversion (TransMLA); sliding-window ring buffers; cross-layer sharing; host/NVMe KV tiers (LMCache, Mooncake, SGLang HiCache, llama.cpp `--cache-ram`); Strata's finding that hierarchical KV goes I/O-bound on *layout fragmentation*, not bandwidth.

**The KV abstraction is already obsolete.** A single 2026 model mixes softmax KV, MLA latent KV, sliding-window KV, Mamba/DeltaNet recurrent state, conv state, per-layer-group RoPE/NoPE, sparse-attention indexer state. Memory planning, paging, eviction, offload, prefix caching, and speculative rollback must be per-state-kind. Forking a sequence *copies* recurrent state rather than refcounting blocks. Prefix caching and speculative rollback for recurrent state remain unsolved at production quality (MiniMax M2 reverted to full attention citing exactly this).

**Activation sparsity** (PowerInfer, Deja Vu, TEAL) is dormant as a product direction. MoE displaced it. 2026 mechanistic results (90% free sparsity, inter-layer neuron dependencies) and one real kernel (Celty) keep it alive as research.

---

## 4. Speculative decoding

**The contract.** Something proposes tokens; the target scores them in one batched pass; a verification rule (modified rejection sampling, provably lossless, acceptance = TV overlap) decides how many to commit. Draft models, MTP heads, EAGLE, n-gram lookup, and block-diffusion drafters are pluggable proposal backends. Build the contract, the exact sampler, KV rollback, and the instrumentation first (`03` §8, §11).

**State of the art as of 2026.** Autoregressive feature-level drafters (EAGLE-3, ~4-5 accepted tokens/round) gave way to **block-parallel / semi-autoregressive drafters**. DFlash (Feb 2026, >6× on Qwen3-4B/8B), then DFlare / Domino / DSpark / DominoTree / JetSpec / PCTree, producing a whole draft block in one pass with 9-11 accepted tokens/round on low-entropy tasks. Diffusion LMs' 2026 impact is as *drafters behind an exact AR verifier*, not as replacement decoders (Mercury/LLaDA2.0 remain separate execution modes). All three major engines ship these: llama.cpp's `common_speculative_type` now has DRAFT_SIMPLE / EAGLE3 / MTP / DFLASH / DSPARK plus five n-gram variants, chainable, with `--spec-*` flags; vLLM and SGLang register `dflash`/`dspark`.

**MTP heads pay where they exist.** Verified from configs: GLM-4.6 and DeepSeek-V3.2 have `num_nextn_predict_layers=1`; **Kimi-K2 has 0**; Qwen3-Next has no such field. Check per checkpoint. Community measurement: built-in MTP gives a clean 1.4-2.2× where separate drafts fail.

**Consumer-hardware constraints:**

1. **MoE breaks the "verification is free" argument.** γ draft tokens activate the *union* of their experts: `U(γ) = 128·[1 − (15/16)^γ]` → 6.45× expert traffic at γ=8 for a 128-expert/8-active model. With CPU-offloaded experts, γ=8 is a *net slowdown* (0.67×) and **γ=1 is optimal (1.80×)**. AcceptMoE-style expert restriction fixes it but is no longer lossless. On a 3090 with Qwen3.6-35B-A3B, 19 draft configs were *all* slower than the 135 tok/s baseline.
2. **Speculation frequently makes things slower.** "Lossless but Not Free" (2607.17283): 3 of 5 consumer configs decelerated; best case 1.61×; one failure because a quantized Metal backend ran "parallel" verification serially. Define `ρ_M = t_T(M)/t_T(1)`; ρ₈≈8 means the backend is serial and speculation cannot help.
3. **Expect 1.3-2.5× end to end**, with real outliers on repetitive/agentic text (prompt-lookup, suffix decoding, Oilbird-style tool-call retrieval 4.4×) and low-entropy math/code. Anything >4× needs inspection of baseline, bonus token, sampling mode, and target utilization.

**Recommended build order** (`03` §11): instrumented batched-verification baseline → exact linear sampler (bonus token, FP32 log-space acceptance, residual sampling, EOS/grammar correctness, paged-KV rollback as a transaction) → proposal registry with ordered fallthrough (suffix → hidden-state retrieval → native MTP → block drafter → external draft → none) → retrieval backends before any learned drafter → generic auxiliary-head loader → **~10 s startup autotuner + online EWMA controller** (enable only if `t_T(1)·τ_safe / C^p90 > 1.10`; disable/shorten when losing; re-explore every 32-128 blocks) → semi-AR block drafter → offload-aware (union-aware γ, grouped expert GEMMs) → ragged verification → fixed-template trees last.

**"Predictive coding".** Searched explicitly (`06` §7): **no paper connects predictive coding / free-energy to speculative decoding**, despite the near-exact correspondence (draft = prediction, verification = prediction error, accepted length = surprise). Adjacent 2026 work: FEP-derived temporal memory fixes MoE routing at domain transitions (2605.00604, unreplicated); Free Energy Heuristics for when-to-think (2606.15877); OasisKV using speculative lookahead as a KV prefetch oracle; SparDA forecasting next-layer KV needs; PixelPrune (classical residual predictive coding for token pruning). The unclaimed generalization: **treat every predictable quantity in the pipeline (next tokens, KV blocks, expert routes, output length, sparse indices, tool calls) as a prediction with an explicit correction budget**, controlled by one online estimator. Adjacent: token-level cascades (PyroDash) are structurally the convergence of routing and speculation.

---

## 5. Quantization in 2026

*(Summary from `01` §6 and `06` §5; see `02-quantization.md` for the full GGUF struct-level detail, EXL2/EXL3 internals, MLX/bitsandbytes formats, kernel derivations, and bit-width/recipe tables.)*

**Unifying view.** All layer-wise PTQ minimizes `‖(W−Ŵ)X‖²_F = tr((W−Ŵ)H(W−Ŵ)ᵀ)`, `H=E[xxᵀ]`. GPTQ = sequential column quantization with damped-Cholesky error feedback (provably Babai's nearest-plane); AWQ = per-channel scale search; **llama.cpp's imatrix is exactly `diag(H)`**; QuaRot/SpinQuant/FlatQuant = rotations that make coordinates isotropic so scalar/block quantizers approximate the full-Hessian objective; QTIP/EXL3 = rotate then trellis-code.

**Where the field is:**

- **Datacenter: microscaling FP won.** NVFP4 (group 16, E4M3 block scale, ~4.5 bpw) and MXFP4/MXFP8 (group 32, E8M0 scale, 4.25 bpw). Dominant recipe: NVFP4 on MoE experts + FP8 attention. gpt-oss ships MXFP4 natively; Kimi K2-Thinking ships QAT INT4; Nemotron 3 NVFP4. **Checkpoints now arrive quantized**, so dequant-fused GEMM is the common path. Two software tricks (Overflow-Aware Scaling, Macro Block Scaling; 2603.08713) close the MXFP4→NVFP4 gap to <1%; naive QuaRot on MXFP4 loses most of its benefit. Use block rotation (2511.04214).
- **Consumer: three frontiers.** (1) GGUF k-/i-quants with imatrix remain the practical Pareto default across CPU/GPU/Metal; (2) **trellis/vector quantization shipped in production on both GPU (ExLlamaV3, July 2026, fused Viterbi quantizer: hours on one 4090 vs AQLM's ~720 A100-hours) and CPU (ik_llama.cpp `IQn_KT`)**. It is the accuracy frontier at 2-4 bpw and the surveyed method that stays coherent at 1.6-2 bpw, but ALU-heavy and *not always a win* (ik's `iq4_kt` loses to `iq4_ks` on both PPL and speed on Qwen3.6-27B); (3) learned post-processing (DWQ/AWQ/GPTQ/dynamic per-tensor bit allocation à la Unsloth UD) is the cheapest quality win at 3-4 bit.
- **Bit-width choice**: ParetoQ finds a sharp learning transition between 2 and 3 bits. ≥3-bit PTQ works; ≤2-bit needs training. GSQ (2604.18556) gets VQ-class 2-3-bit accuracy with plain group-wise scalar weights so existing INT kernels work: the most engine-friendly 2026 result. GPTQv2 is strictly better than GPTQ at the same cost.
- **QAT existence proofs**: Gemma 3 QAT (~5k steps) makes Q4_0 near-lossless; BitNet Distillation converts off-the-shelf Qwen3 to 1.58-bit; BitNet v2 adds native 4-bit activations via an online Hadamard (same primitive QuaRot needs).
- **Activation quant**: W4A8KV4 (QServe/LiquidGEMM) is where dequant overhead disappears on current tensor cores; MXSens reaches W4A4KV4 without rotation.
- **CPU low-bit**: T-MAC / Vec-LUT / T-SAR lookup-table matmul; hand-tuned AVX-512/AMX still beats generic LUT by 2.2×.
- **MoE quantization** is hard (rarely-routed experts see no calibration): MoEQuant, AlphaQ (calibration-free per-expert bits), GEMQ (refit router after quant: the MLX "router picks wrong experts at Q4" complaint), Dynamic Expert Quantization as an online serving policy, REAP expert pruning.
- **Evaluation traps**: use KL-divergence vs a BF16 reference, not perplexity alone (ExLlamaV3's `eval/qbench.py` is the only cross-engine tool found); **low-bit reasoning models preserve accuracy but emit longer chains of thought** (2606.25519). Measure tokens. Factual recall survives compression while reasoning/agentic performance degrades 10-15% at 4-bit.
- **Quantizers surveyed here optimize storage, not latency.** Every method minimizes quality loss under a *storage* budget; none puts measured kernel latency in the objective, uses different execution formats for prefill vs decode, or conditions bit allocation on where a tensor will run (CPU expert vs GPU attention).

---

## 6. Engines, hardware, and workloads

**Engines (`01`).** Consolidation is over: TGI archived (2026-03), DeepSpeed-MII dormant, AutoGPTQ/AutoAWQ/ipex-llm archived, ExLlamaV2 archived for V3, TRT-LLM dropped its TensorRT backend, MLC-LLM on life support. Six live engines: **llama.cpp/ggml, vLLM (V1), SGLang, TensorRT-LLM, MLX, ExLlamaV3**, plus **FlashInfer** under three of them. Surprises: Ollama reversed and now shells out to upstream `llama-server` (its Go engine survives only for Apple/MLX); MLX has a working CUDA backend and Windows support; SGLang's Breakable CUDA Graph beat torch.compile; ktransformers' kt-kernel is upstreamed into SGLang; llama.cpp gained `--fit`, `--cache-ram`, `-ncmoe`, ten speculation modes, per-position acceptance metrics. Two codebases to read first: **nano-vLLM** (~1,200 lines, whole control path legible) and **ExLlamaV3's kernels**.

**Hardware (`07`).** RTX 5090 (32 GB, 1.79 TB/s), RTX PRO 6000 96 GB, used 3090s, Apple M4/M5 Max/Ultra, Strix Halo 128 GB, DGX Spark, Intel Arc B60, high-RAM EPYC boxes for MoE. Consumer multi-device is latency-bound (dual Arc over x4 slower than single; Strix Halo RPC no gain from Thunderbolt).

**Workloads.** Reasoning models (long decode, high length variance), agentic coding (many small parallel requests, ~55% cross-turn KV hit rate, constant re-prefill, tool calls), long context (32k-256k prompts). Session, not request, is the scheduling unit.

**Community pain points, ranked (`07` §5):** prompt processing on CPU/Apple/AMD; the VRAM cliff and hand-tuned `-ot` regexes; fragile prompt caching; backend fragmentation and regressions (ROCm/Vulkan/SYCL/CUDA-graph leaks); server-side multimodal and structured output half-finished; no real TP/EP in llama.cpp; ExLlamaV3 no ROCm; sampler semantics differ per engine; model support lag (new archs land in forks first).

**Architecture trends the engine must absorb (`06` §6, Trends):** sparse attention is now *in the checkpoint* (DeepSeek DSA, MiniMax MSA, GLM-5), so engines need a pluggable **indexer subsystem**; hybrid linear attention (Qwen3-Next, Kimi Linear, Nemotron-H) *and* its counter-trend (MiniMax M2 back to full attention because low-precision state, prefix caching, and speculation break); MLA; fine-grained MoE with shared experts; per-layer-group windows/RoPE/NoPE; MTP heads; VLMs; 1M+ context; small on-device models (Gemma 3n MatFormer/PLE).

---

## 7. Open research opportunities

Struck from the list because they closed in the last year: better low-bit codebooks (trellis shipped), generic CPU/GPU MoE offload, generic host-RAM KV tiering, generic prefix reuse, a portable tensor IR.

Open, roughly ranked by (novelty × consumer impact) ÷ effort. Each is a defensible thesis for a research engine:

1. **Rate-distortion-*latency* quantization + dual-layout execution from one checkpoint.** Allocate bits to minimize `quality_loss + λ·T_decode + μ·T_prefill + ν·bytes` with *measured* kernel latency on the target device, and carry/derive both a GEMV-shaped decode layout and a tensor-core prefill layout without doubling storage. Direct evidence the current objective is wrong: `iq4_kt` losing to `iq4_ks` on both axes. Extend to **placement-aware** bit allocation: a CPU-offloaded expert (RAM-bandwidth-bound) and a GPU-resident attention tensor deserve different bit policies within one checkpoint, yet placement (`-ot`) and quantization are decided by different tools at different times. Also make MoE **router protection the default**. Routers are the most quantization-fragile component across every independent finding. (`01` §10.1-2, `02` §17)
2. **Megakernel × 4-bit × consumer GPU.** Persistent kernel, on-GPU interpreter, SMEM paging, dependency counters instead of kernel barriers; measured headroom 50% → 78% of bandwidth. Unclaimed at this intersection. (`05` §3.3, §11)
3. **Cost-model-driven heterogeneous MoE scheduler.** Per-tensor bytes, per-device bandwidth, negotiated PCIe width, online expert activation frequency, deadline-aware prefetch with a kill switch, quantization as a *placement* dimension (Q2 transfer form / Q4 CPU form / Q4 GPU-resident form), expert-parallel across cheap interconnects, and continuous batching that coexists with CPU offload: the empty niche between llama.cpp and vLLM. (`04` §12, `07` §7)
4. **Adaptive speculation as a control problem**, per request *and per phase*, union-aware γ for MoE, serial-verification detection, exact-vs-constrained routing modes with excluded-mass telemetry. This survey found no engine that closes that loop. (`03` §11)
5. **Transactional speculative KV pages** as a clean abstraction: branch-local staging, zero-copy commit of the accepted path, partial-page COW, device-side accepted-length commit under fixed CUDA graphs; extended to recurrent state (unsolved). (`01` §10.3)
6. **Generalized state allocator**: KV / MLA latent / SWA ring / SSM state / conv state / indexer state as first-class kinds with per-kind paging, eviction, offload, prefix caching, and rollback; a unified budget shared across eviction, quantization, prompt compression, and agent memory under a rate-distortion objective (proposed, TriRoute, MoE-nD, but not built). (`04` §12, `06` Trends)
7. **Quality-aware asymmetric KV tiering**: per-layer/per-head K vs V schemes from online sensitivity, migration overlapped with attention, a reported quality-risk score, and rigorous recompute-vs-retain-vs-compress-vs-evict comparison. (`01` §10.4)
8. **Prefill on low-bandwidth devices**: chunked prefill, persistent cross-session prompt cache, disk-backed KV, prefill-on-GPU-while-decode-on-CPU pipelining, per-phase backend selection. Largest user-facing value on Apple/AMD/unified-memory boxes. (`07` §7)
9. **Predictive-coding unified predictor** (§4): every predictable quantity as a prediction with a correction budget. Unclaimed conceptual framing.
10. **A credible cross-engine evaluation suite**: KLD vs BF16 with exact bpw accounting, roofline efficiency reported every run, crossover surfaces rather than single winning points, energy and thermal steady state, published losing configurations. Arguably the field's biggest missing piece; no apples-to-apples vLLM-vs-SGLang 2026 comparison exists publicly. (`01` §11)

---

## 8. Recommended architecture and build order

Consensus across reports `01` §11, `04` §12, `05` §10-11, `03` §11:

**Stack.** Rust (or C++) for runtime, scheduler, allocator, server, tokenizer, with an explicit no-allocation, no-sync steady-state decode loop; a thin C ABI to hand-written CUDA (and later Metal) for the ~10 kernels that matter (quantized GEMV, quantized GEMM, attention prefill, attention decode, fused MoE, RMSNorm+quant, RoPE, fused SwiGLU, sampler, KV copy); TileLang or Triton for portability/prototyping (FA4 itself is CuTe-DSL in Python. Python-hosted kernel DSLs are no longer a compromise); FlashInfer as the optional dependency with the largest kernel surface (SM12x fused-MoE/FP4 kernels, plan/run architecture); GGUF/AWQ/EXL3 as *importers* repacked into your own execution layouts. Do **not** wrap ggml as the core. Python bindings off the hot path; Vulkan as the portability backend later. CUDA first, but design the IR and memory model so Metal stays credible.

**MVP scope.** One Llama/GQA dense architecture + one DeepSeek-style MoE; RTX 4090/5090; batch 1-4; paged (or VMM-contiguous) KV; graph capture with eager breaks; internal repacking; *one* new contribution from §7. Not MVP: Metal, Vulkan, multimodal, LoRA multiplexing, cluster serving, other architectures.

**Build order** (merged):

1. Roofline instrumentation first. Every run reports tok/s, `BW / bytes_per_token`, η, bytes/token per operator class, H2D/D2H bytes, KV bytes, active params, cache hit rate.
2. Zero-alloc / zero-sync decode loop; "0 CPU↔GPU syncs per token" as a test.
3. Quantized GEMV with fused dequant (LOP3, dp4a/tensor cores); measure the 4-bit gap.
4. CUDA graph capture with a bucket ladder; then the measured fusions.
5. Paged FP16/FP8 KV + continuous batching + chunked prefill; fused GPU sampler; grammar masks overlapped.
6. Prefix caching (radix), persistent on disk.
7. Speculative contract: batched verification + exact sampler + KV rollback + instrumentation → retrieval drafters → MTP/aux-head loader → autotuner → block drafter.
8. Tensor-class-aware placement (dense attn / shared experts / routed experts / embeddings / LM head / KV / recurrent state / scratch. `gpu_layers=N` is not expressive enough) → CPU-resident experts with real CPU kernels → GPU expert cache + byte telemetry → router-aware prefetch with kill switch → layer split across GPUs.
9. Generalized state kinds (SWA/MLA/cross-layer/SSM/DeltaNet), KV quant policy per architecture, hierarchical KV tiers; NVMe last.
10. Then the research contribution proper (megakernel, RD-latency quantizer, unified predictor, …).

**Traps.** Building a compiler before one execution idea; a 2-bit format without a faster kernel; PCIe weight streaming at batch 1; unstructured sparsity; optimizing only mean tok/s; claiming 128K because allocation succeeded; Python in the decode loop; duplicating a datacenter scheduler for one user; comparing quantizations without downstream (KLD/task) quality; comparing a fused speculative path against an unoptimized baseline; expecting seed-identical text under rejection sampling.

**Evaluation harness.** Record model/artifact hashes; quantizer commit, calibration corpus, effective bpw *by tensor class*; GPU clocks/power/driver; cold and warm TTFT; pp throughput; p50/p95/p99 ITL; batch 1/2/4; context 128/2K/8K/32K/128K; peak VRAM/RAM; PCIe bytes; physical and effective bandwidth vs roofline; J/token and thermal steady state; capture/compile time; speculative acceptance per depth and wasted verified tokens. Quality: perplexity + KLD + task suites sensitive to quantization and context (reasoning, code, multilingual, multi-needle, long generation), plus *tokens emitted* for reasoning models.

---

## 9. Method notes

- **Coverage.** ~500 sourced URLs across the seven reports; ~56k words. Reports `01`, `03`, `04`, `05`, and `07` rely on direct fetches of GitHub, arXiv, OpenAlex, Hugging Face `config.json` files, and vendor pages. Reddit is blocked to fetch; community numbers come from GitHub discussions and benchmark sites instead.
- **Offline second-opinion track.** Its reasoning held (roofline, wave counts, GEMV/GEMM crossover, MoE union arithmetic, transactional KV rollback). Its facts were about a year stale: it treated ExLlamaV2 as current, Ollama as going Go-native, MLX as Apple-only, TGI as alive, trellis quantization as an open opportunity, activation sparsity as promising, and `--split-mode row` as current. Every report tabulates the divergences and sided with live sources. Check derivations independently and verify factual claims against their sources.
- **Freshness caveat.** Many cited results carry 2606-2608 arXiv IDs (weeks-old preprints); treat reported speedups as author claims pending replication. Report `06` flags these individually.
- **Where reports disagree with each other**: `05` quotes EAGLE-3 paper speedups (3-6.5×) and a 3.2× M3 Max measurement, while `03` and `07` document consumer configurations that got *slower*. Both are right about their conditions. This synthesis takes the conservative position: 1.3-2.5× typical, self-tuning mandatory.

---

## 10. Index of the detailed reports

| File | Topic | Words | Highlights |
|---|---|---|---|
| `01-engine-landscape.md` | 30+ engines, what makes each fast, roofline-efficiency table, gaps, recommended architecture, offline-vs-web divergence table | ~9.7k | Ollama reversal; BCG > torch.compile; trellis shipped; η vs bandwidth curve |
| `02-quantization.md` | GGUF struct/format detail, EXL2/EXL3, MLX, bitsandbytes, FP8/NVFP4/MXFP4, kernels, QAT, KV/MoE quant, evaluation, bit-width breakpoints, recipes | ~9.5k | GGUF block structs; EXL3 Viterbi quantizer; MXFP4 block-rotation trap; ParetoQ 2-3 bit transition; MoE router fragility |
| `03-speculative-decoding.md` | Math, taxonomy, 2026 block drafters, MTP verification per checkpoint, MoE-offload union math, engine status from source, autotuner design | ~8.2k | γ=1 optimal under MoE offload; 3/5 consumer configs slower; build order |
| `04-memory-offload-kvcache.md` | Roofline, offload cost model, MoE expert offload, prefetch deadline, unified-memory boxes, KV/paged/eviction/MLA/SSM, allocator design, community recipes | ~8.5k | Offload line, 94→60→20 decomposition, Vulkan per-phase, state-kind allocator |
| `05-kernels-and-systems.md` | Attention/GEMM/GEMV kernels, LOP3 dequant, fusions, megakernels, CUDA graphs, scheduling, sampling, CPU/Metal/AMD/Vulkan backends, profiling, language choice | ~12.2k | +36-43% from fusion; megakernel gap; reference 5090/M4 numbers |
| `06-literature-2025-2026.md` | Annotated bibliography, ~120 papers by theme, top-15 must-reads, trends and open problems, predictive-coding search | ~11.3k | Sparse attention in-checkpoint; memory replaced compute; session as unit |
| `07-consumer-hardware-practice.md` | Hardware tiers with measured tok/s, software people use, techniques, models mid-2026, pain points, leaderboards | ~6.6k | MoE everywhere; `-ot exps=CPU`; prefill is the pain; per-model KV-quant sensitivity |
