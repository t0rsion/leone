# Memory Management, Offloading, KV-Cache, and Sparsity

*Survey for a new research inference engine — August 2026*

Cross-checked between live web/arXiv retrieval and a second model (GPT-5.6-sol). Where the two disagreed, live measurements win and the disagreement is noted. Every number that could be recomputed from first principles was recomputed.

---

## 0. The one-paragraph version

Single-stream decode is a **byte-moving problem, not a math problem**. Everything in this document is a technique for reducing the bytes a token must touch, or for moving those bytes on a faster wire. The four levers are: (1) **quantization** — fewer bytes per weight; (2) **sparsity** — read fewer weights, of which MoE is by far the most successful industrial instance; (3) **residency** — keep the hot bytes on the fastest memory you have and *compute where the bytes already live* rather than shipping them; (4) **KV architecture** — MLA, GQA, sliding windows, and hybrid SSM layers that make the per-token state small enough that context stops competing with weights. An engine that instruments bytes/token per operator class will make correct decisions automatically; one that thinks in FLOPs or in a single `gpu_layers=N` knob will not.

---

## 1. The roofline: decode is bandwidth-bound

### 1.1 Arithmetic intensity

For `y = Wx` with `W ∈ R^{m×n}` at batch 1: ~`2mn` FLOPs against `mn·b_w` bytes of weights, so

```
AI_GEMV ≈ 2 / b_w   FLOP/byte
```

| Weight format | bytes/weight | GEMV intensity |
|---|---:|---:|
| FP16/BF16 | 2 | 1.0 FLOP/B |
| INT8 | ~1.05 | ~1.9 FLOP/B |
| Q4_K_M (practical) | ~0.58–0.63 | ~3.2–3.4 FLOP/B |
| MXFP4 (4.25 bit incl. scales) | ~0.53 | ~3.8 FLOP/B |

An RTX 4090's machine balance is ~330 FLOP/B (330 TFLOP/s BF16 ÷ 1008 GB/s). Batch-1 GEMV supplies 1–4 FLOP/B. **The tensor cores are ~99% idle during decode.** With batch/prefill token count `B`, intensity becomes `≈ 2B/b_w`, so Q4 weights at `B=64` reach ~200 FLOP/B — this is why prefill is compute-bound and decode is not, and why *every* throughput trick in this document is really a trick for raising effective `B` or lowering bytes.

### 1.2 The decode ceiling

```
tok/s  ≤  BW_sustained / (W_active_bytes + KV_bytes_read + activations)
```

and the useful diagnostic in reverse:

```
BW_effective = tok/s × bytes_per_token
```

**Measured η (fraction of peak bandwidth actually achieved)**, computed from the llama.cpp community benchmark tables on Llama-2-7B Q4_0 (3.83 GB file):

| Device | Peak BW | tg128 (tok/s) | Implied BW | η |
|---|---:|---:|---:|---:|
| RTX 3060 12GB | 360 GB/s | 75.6 | 289 GB/s | **80%** |
| RTX 4090 | 1008 GB/s | 186.2 | 713 GB/s | **71%** |
| RTX 5090 | 1792 GB/s | 290.0 | 1111 GB/s | **62%** |
| M4 Pro | 273 GB/s | 50.7 | 194 GB/s | **71%** |
| M2 Max | 400 GB/s | 66.0 | 253 GB/s | **63%** |
| M4 Max | 546 GB/s | 83.1 | 318 GB/s | **58%** |
| M3 Ultra | 800 GB/s | 92.1 | 353 GB/s | **44%** |
| M2 Ultra | 800 GB/s | 94.3 | 361 GB/s | **45%** |

Two lessons an engine architect should internalize:

- **η is not a constant, and it falls as the machine gets wider.** A 7B model at batch 1 cannot fill an M3 Ultra's 800 GB/s or a 5090's 1792 GB/s — there isn't enough parallel work per layer to saturate the memory system, and per-layer launch/sync latency becomes a fixed cost. Apple's Ultra parts sit at 44–45%, essentially half their marketing number, while a modest RTX 3060 hits 80%.
- **Use η ≈ 0.65–0.75 for planning on discrete NVIDIA GPUs with medium/large models**, ~0.45–0.60 for Apple Ultra-class, and measure rather than assume for AMD (see §6.2 — one AMD backend hit 99% of achievable, another 65%, on the same silicon).

### 1.3 Where MoE changes everything

Capacity and bandwidth decouple:

```
storage      ~ P_total   × bytes_per_param
decode bytes ~ P_active  × bytes_per_param  +  always-active bytes
```

| Model | Total | Active/token | Practical Q4 file | Active bytes/token (est.) |
|---|---:|---:|---:|---:|
| DeepSeek-V3/R1/V3.1 | 671B | 37B | **384 GB** (Q4_K_XL, 4.5 bit) | ~20 GB |
| — Q2_K_XL | | | **251 GB** (2.71 bit) | ~12 GB |
| — TQ1_0 | | | **170 GB** (1.66 bit) | ~8 GB |
| Qwen3-235B-A22B | 235B | 22B | ~130–150 GB | ~12–13 GB |
| Qwen3-Coder-480B-A35B | 480B | 35B | ~270 GB | ~19 GB |
| Kimi K2 | ~1T | ~32B | ~580–620 GB | ~17 GB |
| GLM-4.5/4.6 | ~355B | ~32B | ~210 GB | ~17 GB |
| gpt-oss-120b | 117B | 5.1B | **63 GB** (MXFP4) | **~3.2 GB** |
| Llama 4 Scout | 109B | 17B | ~62 GB | ~10 GB |
| Llama 4 Maverick | 400B | 17B | ~230 GB | ~13 GB |

The DeepSeek quant sizes are the published Unsloth GGUF sizes, not estimates. Note the practical Q4 rate is **4.5 bits/param, not 4** — a common source of "why doesn't it fit" errors.

**Critical accounting rule.** Do not compute active bytes as `active_params × nominal_quant_bits`. The active route also reads: attention projections at every layer, the router, shared experts (selected with probability 1), norms, quantization scales, and — the one people forget — **the full LM head over the vocabulary, every token**. gpt-oss-120b's 201k-token vocab × 2880 hidden is 579M params, a meaningful slice of a 3.2 GB budget. Sum actual packed GGUF tensor bytes along one active route.

### 1.4 Prefill vs decode, and why offload asymmetry matters

Prefill reuses each weight tile across all prompt tokens; decode rereads everything per token. Loading a 1 GB layer over 25 GB/s PCIe costs 40 ms. For one decode token that is 40 ms/token. For a 1024-token prefill chunk it is 40 ms amortized over 1024 tokens. **Offload is nearly free for prefill and catastrophic for decode.** Design the two phases as different engines with different placement decisions.

---

## 2. CPU–GPU offload mechanics

### 2.1 The cost model

Let `M` = total weight bytes, `f` = fraction executed from CPU-resident memory, `B_g`/`B_c`/`B_p` = effective GPU / CPU-DRAM / PCIe bandwidth.

**Compute-where-resident** (the good scheme):
```
T = (1-f)M/B_g  +  fM/B_c  +  N_x·d·b_a/B_p  +  N_s·τ
```
**Stream weights to GPU** (the bad scheme), even with perfect double-buffering:
```
T ≳ (1-f)M/B_g  +  fM/B_p
```

The difference is structural: PCIe bytes are `O(L·d)` in the first (megabytes) and `O(f·M)` in the second (tens of gigabytes). For DeepSeek-V3 with 58 MoE layers, hidden 7168, BF16 activations: `2 × 58 × 7168 × 2 = 1.66 MB/token`, about **66 µs** at 25 GB/s. Negligible.

### 2.2 The offload cliff, quantified

42 GB Q4 70B, `B_g = 1008`, `B_c = 90` GB/s (dual-channel DDR5-6000, ~90 GB/s sustained of 96 theoretical):

| f (fraction on CPU) | tok/s, compute-resident | tok/s, weight-streaming @25 GB/s |
|---:|---:|---:|
| 0.00 | 24.0 | 24.0 |
| 0.125 | 10.6 | 3.9 |
| 0.25 | 6.76 | 2.22 |
| 0.50 | 3.93 | 1.16 |
| 0.75 | 2.77 | 0.79 |
| 1.00 | 2.14 | 0.60 |

The slowdown ratio is `T(f)/T(0) = 1 + f(B_g/B_c − 1)`. With `B_g/B_c ≈ 11`, **moving just 12.5% of the model to CPU more than halves throughput.** This is the "cliff" — it is not a cliff at all but a hyperbola whose steepest region is at small `f`. Users perceive it as a cliff because throughput is reciprocal latency.

If a user reports a *literal* 2× drop from `-ngl 40` → `-ngl 35` on an 80-layer model (Δf = 6%, model predicts −9%), the cause is discontinuous, not roofline: KV workspace spilled, a disproportionately large tensor moved, NUMA placement flipped, a fused GPU path was abandoned, or transfers fell back from pinned to pageable memory. Instrument for these.

### 2.3 CPU memory systems

| Config | Theoretical | STREAM-ish | Quantized GEMV achieved | 42 GB model ceiling |
|---|---:|---:|---:|---:|
| DDR5-6000 dual-channel | 96 GB/s | 70–90 | 40–70 | 1.0–1.7 tok/s |
| DDR5-4800 × 8ch (Xeon/Epyc) | 307 GB/s | 200–270 | 150–230 | 3.6–5.5 tok/s |
| DDR5-6400 × 12ch (Genoa/Turin) | 614 GB/s | 400–520 | 280–450 | 6.7–10.7 tok/s |

Quantized GEMV lands at **60–85% of STREAM** because, on top of the weight stream, it does scale/zero-point loads, bit unpacking and shuffles, int→float conversion, horizontal reduction, block tails, and per-layer barriers. Q2/Q3 formats can become *unpack-bound* — bytes fall faster than instruction cost, so the extra compression buys less than the byte count suggests. AMX helps prefill GEMMs dramatically and batch-1 GEMV barely at all.

**Two-socket NUMA is a trap.** `numactl --interleave=all` gives nominal aggregate bandwidth but, if the GPU hangs off socket 0, every socket-1 page traverses UPI/Infinity Fabric on the way out. Correct topology: first-touch each expert's weights on its owning socket, pin one thread pool per socket, assign whole experts (never tiles) to a socket, reduce gate-weighted outputs locally, and ship only one hidden-width vector per socket.

### 2.4 PCIe reality

| Gen | x4 | x8 | x16 |
|---|---:|---:|---:|
| Gen3 | 3.94 | 7.88 | 15.75 GB/s |
| Gen4 | 7.88 | 15.75 | 31.51 GB/s |
| Gen5 | 15.75 | 31.51 | 63.02 GB/s |

Sustained pinned H2D: Gen3 x16 ~11.5–13.5, Gen4 x16 ~24–28, Gen5 x16 ~45–56 GB/s. A 4090's VRAM is **40× faster than its own Gen4 x16 link**, not the commonly quoted 10–20×. Mandatory engineering: `cudaHostAlloc` staging buffers, `cudaMemcpyAsync` on a dedicated copy stream, ≥2 buffers (compute `i` while prefetching `i+1`), buffers allocated on the GPU-local NUMA node, and large contiguous transfers instead of thousands of small expert copies. Overlap only hides transfer when `T_compute(i) ≥ T_copy(i+1)` — a condition batch-1 quantized decode routinely violates.

**CUDA Unified Memory / HMM does not create bandwidth.** Oversubscribed weights are close to the worst case for demand paging: each token scans most active weights, pages migrate in, capacity pressure evicts them, the next token faults them back. `cudaMemPrefetchAsync` and `cudaMemAdviseSetPreferredLocation` make it *predictable*, not fast. Windows "shared GPU memory" is a paging fallback, not VRAM expansion.

### 2.5 The engines

- **llama.cpp** — `-ngl N` for layer granularity; `--override-tensor`/`-ot` with regex for tensor granularity, e.g. `-ot ".ffn_.*_exps.=CPU"` (all routed experts to CPU) or `-ot ".ffn_(up|down)_exps.=CPU"` (keep gate on GPU). The newer `--n-cpu-moe N` / `-ncmoe N` is a convenience wrapper for the common case. Both are confirmed present in current builds.
- **FlexGen** (arXiv:2303.06865) — LP search over weight/KV/activation placement across GPU/CPU/disk with a zig-zag block schedule. Ran OPT-175B on a 16 GB GPU. Its headline throughput depends entirely on large batches; it is not a latency design.
- **HF Accelerate** — `device_map="auto"` with `"disk"` offload via forward hooks. Makes things run; is not a token pipeline.
- **DeepSpeed ZeRO-Inference / ZeRO-Infinity** — pinned buffers, prefetch, layerwise streaming from CPU or NVMe. Throughput- and capacity-oriented.
- **ktransformers** — the reference implementation of "compute where resident": GPU attention + shared experts + dense path, CPU routed experts on AMX/AVX-512/VNNI INT4/INT8 kernels, NUMA-aware expert placement. Supports DeepSeek-V3/R1, Kimi K2/K2.5, Qwen3, GLM-5, MiniMax-M2/M3. Documented figure: DeepSeek-R1-0528 FP8 on 8×L20 + Xeon Gold 6454S at 227.85 tok/s aggregate / 87.58 tok/s output at 8-way concurrency; single-GPU DeepSeek inference in ~24 GB VRAM.

---

## 3. Streaming from NVMe

`mmap` in llama.cpp gives virtual mappings over the GGUF; first touch faults from storage, clean pages evict without writeback, and **after warm-up you are usually measuring the OS page cache, i.e. DRAM**. Always benchmark cold (post-`drop_caches`) and warm separately, or your "runs from disk" claim is a RAM claim.

The ceiling is brutal. A good PCIe 5 x4 NVMe does 12–14 GB/s sequential:

| Bytes/token | SSD ceiling |
|---:|---:|
| 42 GB (dense 70B Q4) | 0.33 tok/s |
| 20 GB (DeepSeek active) | 0.70 tok/s |
| 3.2 GB (gpt-oss active) | 4.4 tok/s |

Two PCIe 5 drives in RAID0 approach 24–28 GB/s if they don't share a chipset uplink; four might reach 45–55 GB/s with enough CPU lanes — still 5% of a 4090's VRAM. **GPUDirect Storage** (`nvidia-fs`) removes the CPU bounce buffer and reduces copies but is bounded by `min(BW_SSD, BW_PCIe_path)`, and its filesystem/driver requirements are far less uniform on consumer RTX than on certified DGX. `io_uring` + `O_DIRECT` + registered buffers + deep queues lower software overhead and are genuinely useful for **prefill pipelines, one-shot batch jobs, and expert-cache misses** — never for dense decode. **AirLLM** (layer-at-a-time loading) is a capacity demonstration, not an interactive system.

Verdict: **disk is capacity, not decode bandwidth.** The only architecture where NVMe streaming makes decode sense is a small-active-set MoE whose per-token expert working set is a few hundred MB and whose misses can be hidden behind CPU expert execution.

---

## 4. MoE expert offloading — the central topic

### 4.1 Three strategies, ranked

**(A) Compute non-resident experts on CPU.** Weights stay in DRAM, only activations cross PCIe, bounded by CPU DRAM bandwidth and quantized-GEMV kernel quality. *This is the correct default for batch 1.* ktransformers, llama.cpp `-ot`, and Fiddler all do this. Fiddler's key insight is exactly this: execute the missing expert on CPU rather than mandatorily swapping it in.

**(B) Cache experts in VRAM, transfer misses.** Governed entirely by hit rate. For a DeepSeek-class model — 640B routed params over 58 layers × 256 experts ≈ 43M params/expert ≈ 22–26 MB at Q4, top-8 → ~180–200 MB/layer, ~10–12 GB/token — an all-miss cache at 25 GB/s costs **0.4 s/token, i.e. 2.5 tok/s before any compute**. Strategy (B) only works with high hit rates or low-bit transfers.

**(C) Compress the cached/transferred copies.** 2/3-bit replicas for cold experts, higher precision for hot ones. Attacks the miss-byte term directly; adds quality risk.

Good engines combine all three and choose per expert, per layer, per token.

### 4.2 The prefetch deadline problem

This is the constraint that kills most naive designs. Layer *l*'s router consumes `h_l`, which does not exist until layer *l−1* completes. At that point the useful lead time is ~0. A predictor fired at layer entry gets roughly the duration of layer *l*'s attention sublayer:

| Lead time | Bytes movable @25 GB/s |
|---:|---:|
| 100 µs | 2.5 MB |
| 300 µs | 7.5 MB |
| 1 ms | 25 MB |
| 2 ms | 50 MB |

A single 24 MB DeepSeek expert needs 0.96 ms. **You cannot prefetch eight of them inside a 300 µs attention block.** Any paper claiming ">99% prefetch accuracy" solving offload has, at best, solved *which* experts — not *when*. Always evaluate predictors with **deadline-weighted, byte-weighted recall**: did the right bytes arrive before they were needed?

Consequences: predict ≥1 full layer ahead (Fate's adjacent-layer gate input, APEX's pre-attention issue), use previous-token routing (free, full-token lead time), transfer low-bit slices, or fall back to CPU.

### 4.3 Predictors and their real ceiling

For E=256, k=8, a random top-8 predictor has 3.1% recall. Realistic engineering targets for a rank-64 cross-layer MLP predictor (`7168→64→256` ≈ 475K weights/layer, ~0.95 MB at FP16, ~1M FLOPs/token/layer):
- top-8 recall 50–75%, top-16 recall 70–90%, <20–50 µs latency.

Useful signals, cheapest first: previous token's same-layer experts; previous router *distribution* blended with a global prior `p_pred = α·p_{t−1} + (1−α)·p_global`; prompt-phase expert histogram with decay; a small MLP on `h_{l−1}`; request/domain identity. Note that raw *expert-ID correlation across layers* is meaningless — expert 17 in layer *l−1* and expert 17 in layer *l* are unrelated parameters. The learnable mapping is `(h_{l−1}, r_{l−1}) → r_l`.

**The single most important paper here is the negative result:** *"Not All Models Suit Expert Offloading: On Local Routing Consistency of MoE Models"* (arXiv:2505.16056) profiles 20 MoE models and finds routing predictability varies enormously and trades off against load-balancing objectives. A model trained with aggressive load balancing has near-uniform routing and *cannot* be predicted. **Build the profiler before the predictor**, and gate strategy selection per model and per layer on measured consistency.

Cache policies, with honest failure modes:

| Policy | Strength | Failure mode |
|---|---|---|
| LRU | cheap, adapts to phase change | dies when reuse distance > capacity |
| LFU | exploits global skew | retains stale experts |
| LFU + aging | best simple default | more metadata |
| Router-probability prefetch | value-aware | calibration errors |
| Cost-aware / GreedyDual | accounts for size + CPU speed | scheduler complexity |

Sanity check: with uniform routing and C of E experts cached, hit rate is exactly `C/E`. 16 of 128 = 12.5%. Reported 70–95% hit rates require genuinely skewed routing *and* adequate capacity; never assume them.

### 4.4 Why batching rescues MoE

For near-uniform routing, batch `B` touches
```
D(B) = E · [1 − (1 − k/E)^B]
```
distinct experts. For E=256, k=8:

| B | assignments | distinct experts | intra-batch reuse |
|---:|---:|---:|---:|
| 1 | 8 | 8.0 | 0% |
| 4 | 32 | 30.5 | 4.6% |
| 8 | 64 | ~57 | ~11% |
| 32 | 256 | ~162 | ~37% |
| 128 | 1024 | ~252 | ~75% |

Each expert's weights are read once and serve every token assigned to it. This is why **continuous batching is not merely a serving feature — it is the mechanism by which heterogeneous MoE execution becomes efficient**, and why speculative decoding (which verifies `q` draft tokens at once, giving all `q` routes at a layer simultaneously) is a first-class MoE-offload technique rather than a separate optimization.

Layout matters: build `expert_offsets[E+1]`, `token_indices[B·k]`, `gate_weights[B·k]`, then run one GEMM per non-empty expert (**expert-major**). Token-major execution jumps across 8 expert matrices per token with zero weight reuse.

### 4.5 The 2025–2026 literature, triaged

Live arXiv retrieval surfaced ~30 relevant papers, most published after most models' training cutoffs. Verdicts for an engine architect:

**BUILD**
- **Routing-consistency profiler** (2505.16056). Prerequisite for everything else.
- **HybriMoE** (2504.05897) — dynamic intra-layer CPU-GPU scheduling, impact-driven prefetch, score-based caching; 1.33× prefill / 1.70× decode. The scheduling problem is tiny: for top-8 there are only `2^8 = 256` GPU/CPU partitions; enumerate them exhaustively per layer against `T(S) = max(GPU queue + Σ_S(t_gpu + t_copy), CPU queue + Σ_{¬S} t_cpu) + T_combine`.
- **CoX-MoE** (2605.17889) — coalesced expert execution with AMX CPU-GPU co-execution; claims 7.1× over FlexGen and 2.4× over MoE-Lightning. Baseline-sensitive, but the architecture (coalescing + AMX + co-execution) is right. Dispatch AMX only when an expert receives enough tokens to fill a tile; batch-1 AMX loses to AVX-512 VNNI GEMV.
- **Confidence-and-deadline-aware prefetch** (APEX 2608.11688, >99% overlap accuracy / 26% latency cut; Fate 2502.12224, adjacent-layer gate input, 99% hit rate, 4.5×/4.1×). Implement as: budget `C_copy = η_p·B_p·(T_deadline − T_now)`, sort candidates by `p_e·ΔT_critical(e)/s_e`, issue until budget exhausted. Let confidence control *what* is transferred — full expert at high confidence, high-bit slice at medium, CPU fallback at low.
- **Precision-lattice expert cache** (HOBBIT 2411.01433, 9.93×; Dynamic Expert Quantization 2511.15015, 2.73×). Give every expert a lattice of representations {absent, 2-bit, 3-bit, 4-bit, 8-bit} and solve an online multiple-choice knapsack with a shadow HBM price µ: `Score(e,q) = p_e·L_{e,q} − λ_Q·ΔQ_{e,q} − µ·s_{e,q}`, with hysteresis so routing noise doesn't cause thrashing. HOBBIT's "substitute a low-precision resident copy on miss" must be an explicit approximate mode.
- **Speculative × expert prefetch interfaces** (MoE-SpeQ 2511.14102 2.34×; SP-MoE 2510.10302 1.07–3.5×; AcceptMoE 2608.02989 2.06× with 73.6% traffic reduction; DraftExpert 2607.24434 1.45×). Build the hooks; keep the specific policy replaceable. Draft expert IDs are only meaningful if draft and target routing were explicitly aligned during training.

**WATCH**
- **SliceMoE** (2512.12990, bit-sliced caching, 2.37–2.85× energy). Worth checking the math: cache the high 2 bits of all 256 experts vs all 4 bits of 128 experts, equal budget `128S`. Full cache at uniform routing gives `h = 0.5`, so miss bytes per selected expert = `0.5S`. Bit-slice must fetch the low plane for *every* selection = `0.5S`. **They are exactly equal.** Bit slicing only wins when `h < 0.5` (guarantees every expert has a coarse resident form) or when you exploit `XW = 4·XW_hi + XW_lo` to compute the high plane *while transferring* the low plane. Full-expert caching strictly wins once `h > 0.5`. It also assumes the quant format decomposes into independent linear bit planes, which Q4_K and codebook formats do not.
- **OD-MoE** (2512.03927, cacheless, 99.94% prediction accuracy). Run the deadline inequality: `k·S_e/B_p ≤ T_lead` → 8 × 24 MB / 25 GB/s = **7.68 ms/layer required**. Perfect prediction 500 µs ahead still delivers nothing. Viable only for small experts, many-layer lookahead, low-bit slices, or CXL/NVLink-class links.
- **ReMoE** (2605.27081) — fine-tunes the *router* to prefer recently-used experts; 1.77–1.99× decode. Strongest argument for: it converts locality from a fragile heuristic into a trained property, after which plain LRU works and no predictor is needed. Strongest argument against: **it changes the model.** The router selects which nonlinear functions run; biasing it alters logits, benchmarks, safety behavior, and reproducibility. A general-purpose engine must execute supplied weights faithfully. Ship it as an explicit model-conversion pipeline producing a separately-identified artifact, never as a transparent optimization.
- **SMoE** (2508.18983) expert substitution, **Klotski** (2502.06888) expert-aware multi-batch pipelining, **MoE-Infinity** (request-level activation-aware prefetch), **MoE-Lightning** (CGOPipe overlapping pipeline).

**SKIP (for a pretrained-model engine)**
- **Mixture of Lookup Experts** (2503.15798) — reparameterizes FFN experts as lookup tables. Elegant, but it is a model architecture, not an inference optimization for existing DeepSeek/Qwen/Llama weights. Expose a generic expert-operator interface; don't restructure around it.

### 4.6 PowerInfer and sub-expert sparsity

**PowerInfer** (arXiv:2312.12456) works below expert granularity: offline profiling identifies persistently "hot" neurons that live on GPU, cold neurons stay on CPU, and a runtime predictor identifies contextually active ones. **PowerInfer-2** (2406.06282) extends this to phones, reporting ~11.7 tok/s for a 47B-class MoE. Both depend on the model being ReLU-compatible/sparse and on predictor accuracy. A 2026 follow-on, *"Uncovering Intra-expert Activation Sparsity"* (2605.08575), reports up to **90% intra-expert sparsity, 2.5× MoE-layer speedup, 1.2× end-to-end** — the modest end-to-end number versus the large kernel number is the recurring lesson of all sparsity work.

---

## 5. Activation and contextual sparsity

| Method | Idea | Status |
|---|---|---|
| Deja Vu (2310.17157) | predict important heads/neurons per context; ~80% contextual sparsity | foundational; needs custom kernels |
| PowerInfer | offline hot/cold + runtime prediction | works where ReLU-like |
| TEAL (2408.14690) | training-free magnitude thresholding, 40–50% sparsity | benefit only with byte-avoiding kernels |
| CATS | dynamic context-aware thresholds | variable-length work, scheduling cost |
| ProSparse (2402.13516) / ReLUfication | fine-tune toward exact zeros | needs training |
| Q-Sparse (2407.10969) | top-k + STE, trained sparse throughout | most promising: model + kernel share a contract |
| Prox (2607.27591, 2026) | training-free FFN sparsity from SwiGLU channel salience | **1.99× end-to-end decode at 70% FFN sparsity** |
| Celty (2608.01536, 2026) | SpMspV GPU kernel, dual weight+activation sparsity | 2.8× over cuBLAS; 5.3× at 70% dual-sparsity |
| ELAS (2605.03667, 2026) | squared-ReLU + 2:4 structured activation sparsity | trains into hardware-supported pattern |

**What survives at scale:** structured/block sparsity; coarse neuron or expert skipping; exact zeros created *during training*; static hot/cold placement refined contextually; sparsity used specifically to cut batch-1 bytes.

**What doesn't:** fine-grained unstructured masks; predictors nearly as expensive as the work skipped; sparsity below ~40–60% (overhead eats it); patterns requiring gathers that destroy coalescing; and any claim measured in FLOPs rather than bytes. NVIDIA's 2:4 sparsity accelerates one specific structured pattern and does nothing for arbitrary contextual zeros.

**Sparse attention is the exception that is clearly winning.** Because pages/blocks are coarse, and because at 128K–1M context attention traffic dominates weight traffic, block-sparse attention is now shipping in production models. **DeepSeek-V3.2-Exp** (685B) introduced **DeepSeek Sparse Attention (DSA)** — a lightning indexer plus top-k token selection layered on MLA, achieving fine-grained sparsity while matching V3.1-Terminus on benchmarks. The 2026 literature is dense with page-selection variants: **UNIQUE** (2605.27740, 11.4× attention-kernel / 5.3× end-to-end decode via per-page mean-key scoring), **LOCKS** (2607.24555, spectral per-page summaries, halves decode latency at 100K+ while attending ~2% of tokens), **Quest** (2406.10774), **COBS** (2607.09052), **Vortex** (2606.06453, programmable sparse-attention serving, 3.46× throughput). A caution from the same literature: *"Understanding Sparse Attention Selectivity via Counterfactual Evaluation"* (2608.01676) finds sparsification changes *which content influences output* in ways aggregate accuracy cannot detect.

---

## 6. Unified memory hardware

### 6.1 Apple

| Chip | Bandwidth | Max unified memory |
|---|---:|---:|
| M1 / Pro / Max / Ultra | 68 / 200 / 400 / 800 GB/s | 16 / 32 / 64 / 128 GB |
| M2 / Pro / Max / Ultra | 100 / 200 / 400 / 800 | 24 / 32 / 96 / 192 GB |
| M3 / Pro / Max / Ultra | 100 / 150 / 300–400 / 800–819 | 24 / 36 / 128 / **512 GB** |
| M4 / Pro / Max | 120 / 273 / 410–546 | 32 / 64 / 128 GB |

Metal's `recommendedMaxWorkingSetSize` is below physical RAM; `sysctl iogpu.wired_limit_mb` raises the GPU-wired ceiling and is how people fit a ~400 GB DeepSeek quant on a 512 GB M3 Ultra. Set it too high and the machine becomes unstable.

The M3 Ultra's 512 GB is unique in this class and is the only consumer-adjacent machine that holds DeepSeek Q4_K_XL (384 GB) fully resident. Community reports cluster in the **high-teens to low-20s tok/s** for generation. Note from §1.2 that Apple Ultra parts achieve only ~44% of peak bandwidth on small models — for a 20 GB active route at 800 GB/s and η=0.5, `400/20 = 20 tok/s`, which matches.

### 6.2 AMD Strix Halo / Ryzen AI Max+ 395 — *the most-corrected section*

Spec: 256-bit LPDDR5X-8000 → `8000 × 32 bytes = 256 GB/s` theoretical, up to 128 GB shared.

**Measured** (llm-tracker.info): GPU-internal ~**212 GB/s** (70–73% of theoretical), CPU→GPU copy only ~**84 GB/s**.

| Workload | Backend | pp (tok/s) | tg (tok/s) |
|---|---|---:|---:|
| Llama-2-7B Q4_0 | Vulkan+FA | 884 | 52.7 |
| Llama-2-7B Q4_0 | HIP+WMMA+FA | 344 | 50.9 |
| **Qwen3-235B-A22B Q3_K_XL** | HIP | 65.3 | **10.55** |
| **Qwen3-235B-A22B Q3_K_XL** | Vulkan | 23.8 | **16.09** |
| **Llama-4 Maverick 17B×128E Q4_K** | Vulkan | 57.9 | **16.30** |

Both models' Vulkan results imply ~13 GB/token: `16.09 × 13 = 209 GB/s`, **98.7% of the measured 212 GB/s ceiling**. The Vulkan Q3 GEMV path is essentially bandwidth-perfect. HIP at `10.55 × 13 = 137 GB/s` reaches only 65%.

Why does Vulkan win decode 1.53× and lose prefill 2.74×? They stress opposite things. Decode needs efficient quantized GEMV, low launch overhead, and coalesced dequant — llama.cpp's Vulkan shaders are extremely well tuned for fixed quant blocks. Prefill needs large GEMMs, where ROCm's rocBLAS/hipBLASLt and MFMA matrix instructions dominate and Vulkan shaders trail. **"ROCm vs Vulkan" has no single answer; a serious engine may rationally use different backends for prefill and decode.**

Practical note: Qwen3-235B at practical Q4 (~130–150 GB) does *not* fit 128 GB. Community recipes use Q2/Q3 (~80–110 GB), leaving room for runtime and KV.

### 6.3 NVIDIA DGX Spark / GB10

128 GB coherent LPDDR5X at 273 GB/s, Grace CPU + Blackwell GPU. Measured on llama.cpp:

| Model | pp (tok/s) | tg (tok/s) |
|---|---:|---:|
| gpt-oss-120b MXFP4 | 1956 (pp2048) | **60.57** (tg32) |
| gpt-oss-20b MXFP4 | 2008 | 60.85 |
| Qwen3-Coder-30B Q8_0 | 1654 | 44.26 |

Back-solve gpt-oss-120b: `60.57 × 3.2 GB = 194 GB/s = 71% of 273` — squarely in the normal η band, and an excellent independent confirmation of the ~3.2 GB/token active-route estimate. For a *dense* 70B Q4, `273/42 = 6.5 tok/s` absolute ceiling, ~4–5 realistic. Community disappointment comes entirely from comparing the FP4 TFLOPS headline against HBM GPUs while ignoring the 273 GB/s memory system. Spark is attractive for MoE with small active sets, fine-tuning capacity, large-batch prefill, and CUDA compatibility — not for dense decode.

One operational note from the same thread: kernel version mattered enormously; NVIDIA's 6.17.1 kernel with `NO_PAGE_MAPCOUNT` cut model load from ~104 s to ~22 s.

### 6.4 Intel Lunar Lake

LPDDR5X-8533 on a 128-bit path → `8533 × 16 = 136.5 GB/s`. Capacity tops out well below Strix Halo; suited to 7B–32B quantized models. Xe2 iGPU via Vulkan/SYCL/oneAPI.

---

## 7. Multi-GPU on consumer hardware

**Pipeline / layer split** (`--split-mode layer`, `--tensor-split`): each GPU owns whole layers, weights never move, one hidden activation crosses per stage boundary. Batch-1 latency is the sum of stage times; with multiple requests, stages overlap. **This is the right default on PCIe-only systems.**

**Tensor parallel** (`--split-mode row`, vLLM/SGLang/ExLlamaV2-V3 TP): each matrix splits across GPUs, requiring 1–2 all-reduces per layer. At hidden 8192, one BF16 activation is 16 KiB; 80 layers × 2 collectives = 2.5 MiB/token. The byte count is trivial. **The problem is 160 latency-sensitive collectives per token.** Without verified peer access, host-bounced collectives dominate. Ring all-reduce moves `2(P−1)/P · S` bytes per rank per collective — roughly 2× a one-way exchange, repeated every layer, so TP's total communication vastly exceeds PP's.

Hardware facts: RTX 4090 and 5090 have **no NVLink**; RTX 3090 does, and paired 3090s are meaningfully better for TP where software uses it. Many consumer boards run dual GPUs at Gen4 x8/x8 (15.75 GB/s each); chipset-attached slots are far worse. The tinygrad community's P2P driver patch for 4090-class cards enables peer access but is unsupported and driver-version-sensitive.

**Engine requirement:** discover negotiated PCIe width/generation at startup and refuse to place a hot TP partner behind a chipset x4 link. Use TP only when peer access is verified, topology is known, collectives are benchmarked, and batch/prefill throughput justifies it.

---

## 8. KV cache

### 8.1 The math

```
KV_bytes = 2 · L · N_kv · D_h · S · B_dtype
```

| Model | Per token | 32K | 128K |
|---|---:|---:|---:|
| Llama-2-70B (MHA, 80L × 64 heads × 128) | **2.5 MiB** | 80 GiB | 320 GiB |
| Llama-3-70B (GQA, 80L × 8 heads × 128) | **320 KiB** | 10 GiB | 40 GiB |
| DeepSeek-V3 (MLA, 61L × (512+64)) | **68.6 KiB** | 2.15 GiB | **8.58 GiB** |

GQA gives exactly 8× here. **MLA gives another 4.66× on top of Llama-3-scale GQA** — DeepSeek's 671B model has a *smaller* KV cache per token than an 8B GQA model would with the same layer count. This is not a footnote; it is why DeepSeek-class models are tractable at long context on consumer memory at all. Against DeepSeek's own would-be MHA baseline (128 heads × 128 dim) the ratio is `2·128·128/576 ≈ 57×`.

### 8.2 PagedAttention and block management

vLLM (arXiv:2309.06180) splits KV into fixed-token blocks with per-sequence logical→physical block tables, a free-block pool, reference counts, and copy-on-write. Only the last block has internal fragmentation: at block size 16, average tail waste is 7.5 tokens, i.e. `7.5/avg_seq_len` — 0.75% at 1000 tokens. Measured waste under 4%, 2–4× serving throughput over prior systems.

Copy-on-write: two sequences share physical blocks by refcount; on append to a *partially filled shared* block, allocate a new block, copy the partial contents, update that sequence's table, decrement the old refcount. Full shared blocks are never copied.

**vLLM V1** (Jan 2025) unified the prefill/decode scheduler, made chunked prefill a normal scheduling operation, integrated prefix caching with scheduling, and tightened compilation.

**Block sizing** trades tail waste against kernel indirection. Expected tail waste per live sequence ≈ `(T−1)/2 · q` where `q` is bytes/token:

| Schema | q/token | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| Llama-3-70B FP16 | 320 KiB | 1.09 MiB | 2.34 MiB | 4.84 MiB |
| Llama-2-70B FP16 | 2.5 MiB | 8.75 MiB | 18.75 MiB | 38.75 MiB |
| DeepSeek MLA BF16 | 68.6 KiB | 0.24 MiB | 0.50 MiB | 1.04 MiB |

Block *tables* are negligible (4 bytes/block). Choose block size **per schema** so a block bundle lands around 1–8 MiB. MLA's tiny per-token footprint makes 32- or 64-token blocks reasonable; MHA models want 8 or 16. Note MLA caches one latent per token per *layer* (not per head): `61 × 1152 B = 70,272 B/token`, and 1152 bytes is already 128-byte aligned, so the natural layout is convenient.

### 8.3 Prefix caching, hierarchy, and disaggregation

- **RadixAttention / SGLang** (2312.07104): radix tree over token prefixes, KV blocks on tree edges, scheduler orders requests by prefix-hit. Wins on system prompts, few-shot blocks, multi-turn, and agent trees.
- **LMCache**: externalizes KV across GPU / CPU / local disk / remote store; enables cross-instance prefix reuse when transfer beats recompute.
- **Mooncake**: prefill/decode disaggregation with a distributed KV store and cache-aware routing.

**KV offload economics differ fundamentally from weight offload.** Old KV is read only when attended to; dense weights are needed every token. With sparse/page-selected attention, hierarchical CPU/disk KV is genuinely viable where weight streaming never is.

**What to hash for prefix caching.** Chain-hash complete blocks: `H_i = BLAKE3(H_{i−1} ‖ token_ids_i ‖ position_ids_i ‖ cacheIdentity)`. `cacheIdentity` must include model weight revision, the exact LoRA/adapter stack (ids, order, scales, revisions), KV dtype and quant scheme, cache layout version, RoPE base/scaling/YaRN parameters, attention mask mode, prefix absolute-position offset, and any multimodal/soft-prompt embedding digest. It must **not** include temperature, top-p, seed, or repetition penalty — those affect future token selection, not the K/V of already-fixed tokens. Verify the full digest and token ids before reuse; a 64-bit tag is for lookup acceleration only.

### 8.4 KV quantization

| Format | vs FP16 | Behavior |
|---|---:|---|
| FP8 | 1/2 | usually low loss with calibrated scaling |
| INT8 / q8_0 | ~1/2 | near-lossless in most models |
| INT4 / q4_0-q4_1 | ~1/4 | model- and task-sensitive |
| KIVI 2-bit | ~1/8 | needs asymmetric handling + residual window |

**KIVI** (2402.02750) found keys and values have different outlier structure: quantize keys **per channel**, values **per token**, keep a recent full-precision residual window; ~2.6× lower peak memory. llama.cpp exposes `-ctk`/`--cache-type-k` and `-ctv`; keys are more sensitive than values, so `q8_0` keys with `q4_0`/`q4_1` values is a good asymmetric default. V-cache quantization requires Flash Attention compiled in. Validate on long-context retrieval, needle tests, and code — short chat evaluation will not detect the failure mode.

2026 work has pushed lower: **Output-Aware Rotation for INT2 KV** (2608.02691), **RotaryQuant** (2608.08081 — Walsh–Hadamard + SO(4) rotations for 3-bit KV, fitting a 120B MoE in 32 GB with near-zero perplexity loss), **RaBitQCache** (2606.31519, rotated binary quantization with proven error bounds).

### 8.5 Eviction and compression

| Method | Idea | Limitation |
|---|---|---|
| StreamingLLM (2309.17453) | keep 4 "attention sink" tokens + recent window | discarded facts are gone forever |
| H2O (2306.14048) | retain heavy hitters by accumulated attention | history-dependent bookkeeping |
| Scissorhands (2305.17118) | importance persists over time | approximate |
| TOVA (2401.06104) | evict lowest-attended token per step | per-step selection cost |
| SnapKV (2404.14469) | observation window selects prompt KV per head | prompt compression only |
| PyramidKV (2406.02069) | smaller budgets in compression-tolerant layers | needs a layer policy |
| Quest (2406.10774) | query-aware page selection via metadata | page metadata + search |
| KVzip | importance/reconstructability-based reusable compression | preprocessing |

**StreamingLLM is an infinite-*stream* mechanism, not infinite memory.** Four sinks plus a 4K window does not preserve a fact dropped 100K tokens ago. Sell it accordingly.

2026 additions worth tracking: **AnchorKV** (2608.02901, anchor+residual, 20× shrink at 99% accuracy on 70B without discarding tokens), **ResKV** (2607.29591, exact main cache + compact residual reconstructing omitted softmax contributions), **QEvict** (2608.05326, three tiers: full-precision / quantized-recoverable / deleted), **RestoreKV** (2608.01247, LoRA-adapted restoration pass using 0.4% params, lifting KVzip from 38.2→73.2 on RULER-4K at 5% budget), **vToken** (2608.13263, token-level virtualization + async repacking, 27–72% fewer retained blocks), **GraniKV** (2608.15584, asymmetric paging — contiguous HOT pool for shared prefix, token-level COLD pool for suffix, 2.16× throughput on multi-agent workloads), and, notably for evaluation design, **"Does Accuracy Equal Evidence?"** (2608.01631), which finds an *answer-evidence gap*: correct answers persist under compression while the underlying reasoning degrades. Do not validate KV compression on answer accuracy alone.

### 8.6 Architectural KV reduction and hybrid attention

- **MQA / GQA**: reduction = `N_q / N_kv`.
- **MLA**: cache compressed latent + decoupled RoPE key (§8.1).
- **CLA / YOCO**: adjacent layers share KV; an r-layer sharing group gives ~r× reduction.
- **Sliding window**: retain only W recent positions on local layers. Gemma 3 uses ~5 local layers per global layer; gpt-oss alternates dense and locally-banded layers; Mistral-family models use SWA in various configurations.

**Engines must allocate SWA layers as ring buffers**, `N_ring = ceil(W/T) + 1` blocks per sequence, with `slot = floor(pos/T) mod N_ring` and an *absolute logical-block generation* stored per slot so the kernel can reject a wrapped slot mistaken for old context. Allocating a full-context growing table for every sliding-window layer throws away the entire architectural saving — a real and common bug. Prefix sharing on local layers works cleanly only before the ring wraps; after that the local state is path-dependent.

**Chunked prefill** caps temporary activation memory, lets decode interleave with a long prompt, and gives the scheduler a stable token budget. The scheduler must preserve causal state across chunks and distinguish already-cached prefix tokens from newly computed ones.

---

## 9. Hybrid SSM / linear attention — constant-memory state

These replace some softmax layers with fixed-size recurrent state, independent of context length.

State size, worked: `d_inner = 8192`, `d_state = 16`, BF16 → `8192 × 16 × 2 = 256 KiB/layer`; across 64 layers ≈ **16 MiB/sequence**, constant at 128K or 1M. At `d_state = 128` it is ~128 MiB. Delta-rule/linear-attention layers hold per-head `d_k × d_v` matrices — "constant" does not mean "tiny", and at short context an SSM layer can use *more* memory than KV would.

Families: **Mamba-2** (2405.21060, state-space duality + chunk scans), **Jamba** (2403.19887), **Qwen3-Next** (Gated DeltaNet + attention hybrid), **Nemotron-H**, **Kimi Linear** (KDA + a minority of full-attention layers), **MiniMax-M1** (Lightning Attention + sparse full attention), **Falcon-H1**, **IBM Granite 4**.

**What an engine must do differently:**
1. A KV-only abstraction is insufficient. Declare per-layer state kinds: `FULL_KV`, `SLIDING_KV(window)`, `SHARED_KV(group)`, `MLA_LATENT`, `SSM_STATE`, `DELTA_MATRIX_STATE`, `NO_STATE`.
2. Prefill needs chunked parallel-scan kernels; decode needs in-place recurrent updates.
3. **Forking a sequence means copying recurrent state, not refcounting immutable blocks.** Beam search and speculative decoding must snapshot/rollback SSM state rather than sharing it.
4. Prefix caching can store checkpointed recurrent states at boundaries, but arbitrary branching is far less shareable than KV blocks.

2026 systems work is arriving: **DeltaLog** (2608.15533) represents recurrent state as a dense base plus a bounded log of recent compact updates, giving 1.86× kernel speedup and **7.83× reduction in state write traffic** — a direct hint that naive full-state rewrites per token are a real bottleneck. **MARCH** (2608.12435) periodically caches recurrent-state checkpoints as content-routed anchors, which is essentially prefix caching for SSMs. **UniPrefill** (2605.06221) accelerates prefill on hybrid architectures via block-wise dynamic sparsification.

---

## 10. Allocator design

### 10.1 PyTorch caching allocator

Small/large pools, block splitting, coalescing, stream/event safety. "Reserved" can vastly exceed "allocated" when differently-sized long- and short-lived tensors interleave. `empty_cache()` returns wholly-free segments but cannot compact live allocations. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps when sizes vary slightly across batches. **None of this substitutes for a KV-specific allocator.**

### 10.2 vAttention and CUDA VMM

CUDA VMM separates virtual reservation (`cuMemAddressReserve`), physical allocation (`cuMemCreate`), mapping (`cuMemMap`), and permissions (`cuMemSetAccess`). **vAttention** (2405.04437) reserves a contiguous virtual KV range per sequence and maps physical pages as it grows, so ordinary contiguous FlashAttention kernels work unmodified while the allocator stays paged underneath.

Granularity is **not universally 2 MiB** — query it with `cuMemGetAllocationGranularity(..., CU_MEM_ALLOC_GRANULARITY_MINIMUM)`. A100/H100-class environments commonly report ~2 MiB; some device/driver combinations expose 64 KiB. Never hard-code it. At 2 MiB granularity, thousands of short sequences waste substantial memory, map/unmap must be batched, and driver map latency (tens of µs) is unsuitable per-token unless amortized.

**Choose block-table paging** when you need cross-platform backends (CUDA/ROCm/Metal/Vulkan), prefix sharing, beam and speculative trees, many short sequences, KV offload tiers, mixed dtypes, or hybrid local/global/MLA layouts. **Choose VMM** when NVIDIA-only, existing contiguous kernels are strategically important, per-sequence allocations are large enough to amortize granularity, and branching is limited. A hybrid — VMM for large private decode sequences, explicit blocks for shared prefixes and short requests — is possible but only worth it after profiling proves indirection is the bottleneck.

### 10.3 Concrete recommendations

Compact 32-byte block descriptor: `{atomic<u32> refcount; u32 next_free; u16 pool_id; u16 used_tokens; u8 tier; u8 state; u16 flags; u64 last_access_epoch; u64 hash_tag;}`. Pools must be **homogeneous** in device, state kind, K/V dtype, layout, block_tokens, bytes/token/layer, layer group, and quant scheme — never mix FP16, FP8, Q4, MLA, and GQA blocks in one variable-size pool.

GPU block table as a flat `int32` array (`-1` = unmapped), one table per schema group, indexed `block_table_g[seq * stride_g + logical_block]`. Kernels compute `logical = pos / T; offset = pos % T; physical = table[...]`. Optimized layouts differ for K and V: K as `[block][kv_head][head_dim/vec][block_token][vec]` (transposed/vectorized for dot products), V as `[block][kv_head][block_token][head_dim]` (token-major for weighted accumulation).

Non-negotiables:
1. **Transactional allocation.** Reserve every block a scheduling quantum needs across *all* pools, or roll back. Otherwise a request gets K blocks and fails to get V or its MLA companion.
2. **Deferred free with GPU events.** Never reuse a block whose last kernel may still be running; queue `{pool, block, safe_after_event}` and reclaim from a poller.
3. **Low/high watermarks** triggering eviction or preemption *before* allocation fails.
4. **Swap-vs-recompute rule:** swap if `K/B_out + K/B_in + T_meta < R/PP_effective`. Example: 10 GiB KV at 25 GB/s each way ≈ 0.8 s round trip; 32K tokens at 2000 tok/s prefill = 16 s → swap. 512 uncached tokens = 0.26 s → discard and recompute.
5. **Instrument fragmentation**: physical used/free, reserved virtual, tail waste, shared bytes, evictable bytes, allocation latency, swap/prefetch bytes.
6. **Treat KV blocks as the admission-control resource.** Estimate worst-case blocks for the next quantum, not current usage.

---

## 11. Community recipes with real numbers

| Setup | Capacity | Reported |
|---|---|---|
| **DeepSeek-V3.1, 1×24 GB GPU + 128 GB RAM**, TQ1_0 (1.66 bit) | 170 GB file | runs; MoE offload via `-ot ".ffn_.*_exps.=CPU"` |
| **DeepSeek-V3.1, 1×24 GB GPU + 226 GB RAM+VRAM**, Q2_K_XL | 251 GB file | **~5 tok/s** (Unsloth's own guidance) |
| DeepSeek Q4_K_XL | 384 GB file | needs ~400 GB; M3 Ultra 512 GB or 512+ GB server |
| **DeepSeek-R1-0528 FP8, 8×L20 + Xeon 6454S**, ktransformers | — | **227.9 tok/s total / 87.6 tok/s output** @ 8-way concurrency |
| **Qwen3-235B-A22B Q3_K_XL, Strix Halo 128 GB** | ~100 GB | **pp 23.8 / tg 16.09** (Vulkan); pp 65.3 / tg 10.55 (HIP) |
| **Llama-4 Maverick Q4_K, Strix Halo** | — | **pp 57.9 / tg 16.30** (Vulkan) |
| **gpt-oss-120b MXFP4, DGX Spark 128 GB** | 63 GB | **pp2048 1956 / tg32 60.57** |
| **gpt-oss-120b, RX 7900 XT 20 GB + RAM** | 63 GB | **94 tok/s** default → **60** (4 MoE layers on CPU) → **20** (all MoE on CPU) |
| **gpt-oss-120b, RTX 3060 12 GB** + selective `-ot` | 63 GB | **75 tok/s** @2K ctx, **67** @16K, **53–56** @32K |
| DeepSeek Q4, M3 Ultra 512 GB | 384 GB | high-teens to low-20s tok/s |

**The 94 → 60 → 20 curve, decomposed.** Convert to latency: 10.64 → 16.67 → 50.0 ms/token. Assuming 36 MoE layers, `B_g_eff ≈ 400` GB/s and `B_c_eff ≈ 50` GB/s, solve `39.36 ms = A_e(1/50 − 1/400)` → routed-expert traffic `A_e ≈ 2.25 GB/token`, GPU expert time 5.63 ms, dense remainder 5.01 ms ≈ 2.0 GB — total ≈ **4.25 GB/token**. Check: all-CPU predicts `2.0/400 + 2.25/50 = 50 ms` = exactly 20 tok/s. Four-of-36 predicts 15.0 ms; observed 16.67 ms, the extra 1.7 ms being hybrid synchronization. The model reproduces all three points from two bandwidth constants.

**Reporting hygiene.** Two errors dominate community benchmarks: (1) a "384 GB DeepSeek Q4" claim that is actually a smaller IQ quant or leans on VRAM — 671B at 4.8 bits is ~403 GB before runtime overhead; (2) conflating pp and tg. `pp512=100, tg128=4` means interactive generation is 4 tok/s. A reproducible record needs: exact GGUF checksum, engine commit, quant type and file bytes, CPU/sockets/channels/DIMM speed, NUMA policy, GPU and **negotiated** PCIe width, exact CLI or tensor placement, context length, KV dtype, prompt/generated token counts, cold-vs-warm page cache, pp and tg separately, and peak RAM/VRAM.

---

## 12. Design implications for a new engine

**Priority order**, revised after the measured data:

1. **Exact active-route byte accounting**, generated from tensor metadata, not from parameter counts. Every benchmark reports model bytes/token, KV bytes/token, H2D/D2H bytes/token, effective GB/s per operator class, active params/token, cache hit rate, and useful overlap. This makes results portable across hardware and instantly exposes impossible claims.
2. **Tensor-class-aware placement.** Independent policies for dense attention, shared experts, routed experts, embeddings, LM head, KV, recurrent state, and scratch. `gpu_layers=N` is not expressive enough for any 2025+ model.
3. **Compute where resident.** Never auto-stage weight tensors over PCIe. Ship activations, not weights. Requires real CPU kernels: Q2/Q3/Q4 GEMV, grouped expert GEMM, AVX2/AVX-512/VNNI/AMX paths, NUMA-local shards.
4. **Separate decode and prefill calibration per backend.** The Strix Halo result (Vulkan wins decode 1.5×, loses prefill 2.7×) proves a single backend choice is wrong.
5. **Routing-consistency profiler per model and per layer**, gating which expert strategy runs where.
6. **Cost-aware expert scheduler.** For each expert know: resident device, compressed and expanded size, CPU time, GPU time, H2D time, predicted probability, reuse-distance distribution. Then exhaustively enumerate the ≤256 GPU/CPU partitions per layer.
7. **Deadline- and confidence-aware prefetch** with a byte budget derived from actual lead time — and an automatic kill switch when false-prefetch bytes exceed useful bytes.
8. **Native paged KV subsystem**, not framework tensors: fixed blocks, GPU block tables, COW/refcounts, prefix hashing with full cache identity, CPU/NVMe tiers, ring buffers for SWA, mixed dtypes, cache-aware routing.
9. **Generalized state types** beyond K/V, so MLA, CLA/YOCO, sliding windows, Mamba, and DeltaNet are first-class rather than retrofitted.
10. **Quantization as a placement dimension.** The same expert may have a high-precision CPU master, a Q4 CPU execution form, a Q2 transfer form, and a Q4 GPU-resident form. Mixed precision can turn a 40% cache hit rate into a 70% effective *byte* hit rate.
11. **Prefer pipeline/layer split on weak interconnects.** Auto-detect negotiated PCIe width and refuse pathological TP placements.
12. **Continuous batching and speculative verification as MoE infrastructure**, not just serving features — they are how you raise the effective batch that makes expert-major GEMMs and cache reuse work.

Suggested build order: resident dense Q4 decode with a roofline-valid bandwidth counter → paged FP16/FP8 KV + continuous batching → layer split across GPUs → CPU-resident dense layers → tensor-regex placement and CPU MoE experts → GPU expert cache with LRU + byte telemetry → router-aware prefetch with CPU-on-miss → prefix caching and hierarchical KV → SWA/MLA/cross-layer KV types → SSM/delta state allocator and chunk-scan kernels → NVMe tier last.

**The governing rule:** capacity may come from RAM or SSD, but every generated token still pays for the bytes it actually touches. Winning systems reduce those bytes — through quantization, active-parameter sparsity, expert residency, KV compression, architectural state reduction, and reuse — not through transparent oversubscription.

---

## Sources

### Systems and offload
- FlexGen — https://arxiv.org/abs/2303.06865
- DeepSpeed ZeRO-Inference — https://www.deepspeed.ai/2022/09/09/zero-inference.html
- ZeRO-Infinity — https://arxiv.org/abs/2104.07857
- HF Accelerate big-model inference — https://huggingface.co/docs/accelerate/usage_guides/big_modeling
- llama.cpp — https://github.com/ggml-org/llama.cpp
- ktransformers — https://github.com/kvcache-ai/ktransformers
- AirLLM — https://github.com/lyogavin/airllm
- NVIDIA GPUDirect Storage — https://docs.nvidia.com/gpudirect-storage/
- CUDA Unified Memory — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming
- PyTorch CUDA memory management — https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management

### MoE offloading (2024–2026)
- APEX: Adaptive Expert Prefetching for Memory-Efficient Edge MoE — https://arxiv.org/abs/2608.11688
- RotaryQuant: Fitting 120B MoE on Consumer Hardware — https://arxiv.org/abs/2608.08081
- AcceptMoE — https://arxiv.org/abs/2608.02989
- DraftExpert — https://arxiv.org/abs/2607.24434
- Beyond Uniform Experts: Cost-Aware Expert Execution — https://arxiv.org/abs/2606.29982
- SpecPrefetch — https://arxiv.org/abs/2607.24787
- ReMoE: Boosting Expert Reuse through Router Fine-Tuning — https://arxiv.org/abs/2605.27081
- CoX-MoE: Coalesced Expert Execution with AMX CPU-GPU Co-Execution — https://arxiv.org/abs/2605.17889
- SliceMoE: Bit-Sliced Expert Caching — https://arxiv.org/abs/2512.12990
- OD-MoE: On-Demand Expert Loading — https://arxiv.org/abs/2512.03927
- Dynamic Expert Quantization — https://arxiv.org/abs/2511.15015
- MoE-SpeQ — https://arxiv.org/abs/2511.14102
- Pre-Attention Expert Prediction and Prefetching — https://arxiv.org/abs/2511.10676
- In-depth Analysis on Caching and Pre-fetching in MoE Offloading — https://arxiv.org/abs/2511.05814
- ExpertFlow — https://arxiv.org/abs/2510.26730
- SP-MoE — https://arxiv.org/abs/2510.10302
- DuoServe-MoE — https://arxiv.org/abs/2509.07379
- Hiding Offloading Latency with Speculative Decoding — https://arxiv.org/abs/2508.21706
- SMoE: Expert Substitution — https://arxiv.org/abs/2508.18983
- Not All Models Suit Expert Offloading (local routing consistency) — https://arxiv.org/abs/2505.16056
- HybriMoE — https://arxiv.org/abs/2504.05897
- Mixture of Lookup Experts — https://arxiv.org/abs/2503.15798
- Fate: Cross-Layer Gate Prefetching — https://arxiv.org/abs/2502.12224
- Klotski — https://arxiv.org/abs/2502.06888
- Fine-Grained Expert Offloading — https://arxiv.org/abs/2502.05370
- HOBBIT — https://arxiv.org/abs/2411.01433
- Mixtral-offloading — https://github.com/dvmazur/mixtral-offloading

### Sparsity
- Deja Vu — https://arxiv.org/abs/2310.17157
- PowerInfer — https://arxiv.org/abs/2312.12456
- PowerInfer-2 — https://arxiv.org/abs/2406.06282
- TEAL — https://arxiv.org/abs/2408.14690
- ProSparse — https://arxiv.org/abs/2402.13516
- Q-Sparse — https://arxiv.org/abs/2407.10969
- Prox: Training-Free FFN Activation Sparsity — https://arxiv.org/abs/2607.27591
- Celty: SpMspV Dual-Sparse GPU Kernel — https://arxiv.org/abs/2608.01536
- Intra-expert Activation Sparsity — https://arxiv.org/abs/2605.08575
- ELAS: 2:4 Activation Sparsity — https://arxiv.org/abs/2605.03667
- UNIQUE: Universal Top-k Sparse Attention — https://arxiv.org/abs/2605.27740
- LOCKS: Page-Local Compact Key Summaries — https://arxiv.org/abs/2607.24555
- Quest — https://arxiv.org/abs/2406.10774
- COBS — https://arxiv.org/abs/2607.09052
- Vortex: Programmable Sparse Attention Serving — https://arxiv.org/abs/2606.06453
- Sparse Attention Selectivity via Counterfactual Evaluation — https://arxiv.org/abs/2608.01676

### KV cache
- PagedAttention / vLLM — https://arxiv.org/abs/2309.06180
- vLLM V1 — https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- SGLang / RadixAttention — https://arxiv.org/abs/2312.07104
- vAttention — https://arxiv.org/abs/2405.04437
- LMCache — https://github.com/LMCache/LMCache
- KIVI — https://arxiv.org/abs/2402.02750
- StreamingLLM — https://arxiv.org/abs/2309.17453
- H2O — https://arxiv.org/abs/2306.14048
- Scissorhands — https://arxiv.org/abs/2305.17118
- TOVA — https://arxiv.org/abs/2401.06104
- SnapKV — https://arxiv.org/abs/2404.14469
- PyramidKV — https://arxiv.org/abs/2406.02069
- GraniKV: Asymmetric Granularity KV Paging — https://arxiv.org/abs/2608.15584
- vToken: Token-Level Virtualization — https://arxiv.org/abs/2608.13263
- AnchorKV — https://arxiv.org/abs/2608.02901
- QEvict — https://arxiv.org/abs/2608.05326
- RestoreKV — https://arxiv.org/abs/2608.01247
- ResKV — https://arxiv.org/abs/2607.29591
- Output-Aware Rotation for INT2 KV — https://arxiv.org/abs/2608.02691
- RaBitQCache — https://arxiv.org/abs/2606.31519
- Does Accuracy Equal Evidence? (compression faithfulness) — https://arxiv.org/abs/2608.01631
- Runtime Observability for Heterogeneous Attention Memory — https://arxiv.org/abs/2608.05863

### Architectures
- DeepSeek-V2 / MLA — https://arxiv.org/abs/2405.04434
- DeepSeek-V3 — https://arxiv.org/abs/2412.19437
- DeepSeek-V3.2-Exp (DeepSeek Sparse Attention) — https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Mamba — https://arxiv.org/abs/2312.00752
- Mamba-2 — https://arxiv.org/abs/2405.21060
- Jamba — https://arxiv.org/abs/2403.19887
- Qwen3-Next — https://qwenlm.github.io/blog/qwen3-next/
- Qwen3 technical report — https://arxiv.org/abs/2505.09388
- DeltaLog: Deferred Materialization of Recurrent States — https://arxiv.org/abs/2608.15533
- MARCH: Content-Routed State Anchors — https://arxiv.org/abs/2608.12435
- UniPrefill (hybrid-architecture prefill) — https://arxiv.org/abs/2605.06221
- FlashAttention — https://arxiv.org/abs/2205.14135 · FlashAttention-2 — https://arxiv.org/abs/2307.08691

### Benchmarks and recipes
- llama.cpp Apple Silicon performance thread — https://github.com/ggml-org/llama.cpp/discussions/4167
- llama.cpp CUDA performance thread — https://github.com/ggml-org/llama.cpp/discussions/15013
- llama.cpp gpt-oss guide (MoE offload flags + benchmarks) — https://github.com/ggml-org/llama.cpp/discussions/15396
- llama.cpp DGX Spark benchmarks — https://github.com/ggml-org/llama.cpp/discussions/16578
- Strix Halo / Ryzen AI Max+ 395 benchmark compendium — https://llm-tracker.info/_TOORG/Strix-Halo
- Unsloth DeepSeek-V3.1 local run guide (quant sizes, flags, tok/s) — https://unsloth.ai/docs/models/tutorials/deepseek-v3.1-how-to-run-locally.md
- NVIDIA DGX Spark — https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- AMD Ryzen AI Max+ 395 — https://www.amd.com/en/products/processors/consumer/ryzen-ai.html
- Apple Mac technical specifications — https://support.apple.com/specs
- r/LocalLLaMA — https://www.reddit.com/r/LocalLLaMA/
