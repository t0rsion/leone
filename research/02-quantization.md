# Quantization and Compression for LLM Inference: A Survey for Engine Builders

*Compiled 2026-08-18. Companion to `00-synthesis.md` §5, `01-engine-landscape.md` §6, `06-literature-2025-2026.md` §5, and `07-consumer-hardware-practice.md` §3.1/3.2. This file goes deep on formats, kernels, and concrete recipes rather than re-deriving the unifying theory. That theory, unchanged here: almost every layer-wise PTQ method minimizes `‖(W−Ŵ)X‖²_F = tr((W−Ŵ)H(W−Ŵ)ᵀ)` with `H = E[xxᵀ]`; GPTQ solves it by sequential column quantization with Cholesky error feedback (provably Babai's nearest-plane algorithm), AWQ by per-channel scale search, llama.cpp's importance matrix is exactly `diag(H)`, and QuaRot/SpinQuant/QTIP/EXL3 rotate the coordinate system first so a scalar or trellis quantizer approximates the full-Hessian objective better.*

## Table of contents

1. The field in one paragraph
2. Weight-only PTQ methods (RTN → QTIP/EXL3, ParetoQ, GSQ, 2026 SOTA)
3. GGUF formats in depth
4. EXL2 and EXL3
5. MLX quantization
6. bitsandbytes: NF4 and LLM.int8()
7. FP8 and microscaling (NVFP4/MXFP4/MXFP8)
8. Activation quantization: W4A16 vs W4A8 vs W4A4 vs W8A8
9. Kernels: Marlin, Machete, GemLite, ggml mmq/mmvq, LOP3 tricks, CUTLASS, T-MAC/LUT
10. QAT and low-bit-native models
11. KV cache quantization
12. Hadamard/rotation tricks
13. MoE quantization
14. Evaluating quantization: PPL, KLD, agentic degradation
15. Where quality breaks by bit-width
16. Practical recipes: running 70B-1T models on consumer hardware
17. Implications for a new engine
18. Sources

---

## 1. The field in one paragraph

Quantization is the technique that most directly attacks the consumer bottleneck: decode at batch 1 is bandwidth-bound (§5 of `05-kernels-and-systems.md`), so halving bytes/parameter roughly doubles tokens/second *and* halves the memory footprint that decides whether a model fits at all. The field has split into three tracks that rarely cite each other's kernels: **GGUF k-/i-quants** (llama.cpp/ggml, the universal CPU+GPU+Metal default, imatrix-calibrated scalar and vector-ish quantization in a self-describing file format), **trellis/vector quantization** (QTIP → EXL3 on GPU, ik_llama.cpp's `_KT` family on CPU, the accuracy frontier below 4 bpw and the surveyed method that stays coherent below 2 bpw, at the cost of ALU-heavy decode), and **microscaling floating point** (MXFP4/NVFP4/MXFP8, what datacenter checkpoints now ship natively, e.g. gpt-oss, Kimi K2-Thinking, Nemotron 3). A fourth, increasingly important track is **QAT-native low-bit models** (BitNet, Gemma 3 QAT) that make the training-vs-PTQ boundary irrelevant. Across all four, the same finding recurs: no surveyed quantizer objective includes measured kernel latency, only storage-budget quality, which is exactly the gap §17 argues a new engine should close.

---

## 2. Weight-only PTQ methods

### 2.1 The baseline and the sequential-correction family

**RTN (round-to-nearest)**, per-tensor or per-group affine (`scale`, optional `zero_point`) quantization with no calibration. Degrades sharply below 4 bit because a handful of outlier channels dominate the group's dynamic range and the remaining channels get crushed into a few levels. Still the right default at 8 bit, where outliers barely matter.

**GPTQ** (Frantar et al., 2022) quantizes one column at a time and uses a damped inverse-Hessian (`H = 2XXᵀ + λI`, Cholesky-decomposed) to push each column's quantization error into the *not-yet-quantized* remaining columns, proven equivalent to Babai's nearest-plane lattice algorithm (`2507.18553`). One-shot, runs on a single GPU in under an hour for 7-13B models, W4A16 the default target. **GPTQv2** (`2504.02692`) fixes an inconsistency in the original calibration objective. It matches the *quantized* network's output at each step rather than the full-precision network's, at the same cost, with strictly better error.

**AWQ** (Lin et al., MLSys 2024) observes empirically that ~1% of weight channels are "salient" (large activation magnitude) and protects them not by keeping them in higher precision but by a *per-channel equivalent transformation*: scale up the salient input channels before quantizing the weight, scale the corresponding activations down by the inverse. A small grid search over the scale finds a near-optimal value without backprop or a reconstruction pass. Faster to produce than GPTQ, slightly more stable across domains since it doesn't overfit to the calibration set's Hessian, and the origin of the widely deployed AWQ/GPTQ-Marlin kernel pair.

**SmoothQuant** (Xiao et al., 2022) is AWQ's activation-quantization cousin: it migrates quantization difficulty from activations to weights via a per-channel smoothing factor `s`, dividing activations and multiplying weights, so that **both** weights and activations can go to INT8 (W8A8) without the activation outliers blowing up the quantization range. This is the mechanism behind TensorRT-LLM's and vLLM's SmoothQuant W8A8 paths.

**OmniQuant** (Shao et al., 2023) makes the transform *learnable* instead of grid-searched: **Learnable Weight Clipping (LWC)** optimizes per-channel clip thresholds, **Learnable Equivalent Transformation (LET)** generalizes AWQ/SmoothQuant's static scale into a gradient-descended one, both trained block-by-block (not end-to-end) so it's cheap enough to run on a single GPU while getting most of the benefit of full QAT. Works for weight-only and weight+activation quantization.

**HQQ** (Half-Quadratic Quantization, mobiusml) is calibration-free, no dataset, no forward passes through the model at all. It poses quantization as sparse-plus-quantized decomposition and solves it with half-quadratic splitting (alternating closed-form updates), producing a 4-bit or 3-bit quantized model in minutes rather than hours, competitive with GPTQ's calibrated result without needing calibration data. Good default when you don't trust your calibration set or need to quantize fast.

**SqueezeLLM** (Kim et al., 2023) departs from uniform grids entirely: **non-uniform quantization** via Fisher-information-weighted k-means clustering per channel (so the codebook levels concentrate where they reduce loss most), plus a **dense-and-sparse** split that pulls outlier weights out into an unquantized sparse matrix so the dense part can use a tight non-uniform codebook. Needs a LUT-based dequant kernel rather than a scale-multiply.

**SpQR** (Dettmers et al., 2023): Sparse-Quantized Representation, identifies "outlier" weights via a per-weight sensitivity proxy and keeps roughly 1% of weights in full precision inside a sparse side-matrix, quantizing the rest at 3-4 bit; reports near-lossless compression at ~4.63 bits average.

### 2.2 Incoherence processing and vector quantization: the sub-4-bit frontier

**QuIP** (Chee et al., NeurIPS 2023) introduced **incoherence processing**: multiply weights and Hessian by random orthogonal (Hadamard-derived) matrices before quantizing, which spreads outlier mass evenly across coordinates so a scalar quantizer no longer gets blindsided by a few huge values, then does adaptive (LDLQ) rounding. First method to make 2-bit weight-only quantization usable at all. **QuIP#** (Tseng et al., 2024) improves it with **E8 lattice codebooks** (denser, better-packing than a scalar grid), a fused fast Hadamard transform, and a light fine-tuning pass, near-lossless at 2 bits on several models, the direct ancestor of the codebook idea inside GGUF's `IQ*` types and of QTIP.

**AQLM** (Egiazarian et al., 2024): Additive Quantization for LLMs, represents each group of weights as a **sum of vectors drawn from multiple learned codebooks** (multi-codebook vector quantization, borrowed from the MCQ literature), then fine-tunes the codebooks and per-channel scales end-to-end against the calibration set. Reaches the best *quality* at 2-3 bit of the pre-trellis generation, but quantization is extremely expensive, on the order of **hundreds to ~720 A100-GPU-hours** for a 70B-class model, which is the reason EXL3's "hours on one 4090" framing (§4) reads as a big deal. **PV-Tuning** (`2405.14852`) generalizes the fine-tuning step: instead of only tuning scales after VQ, it tunes the discrete codebook assignments themselves via a continuous relaxation, recovering additional accuracy for any VQ-style quantizer including AQLM's.

**VPTQ** (`2409.17066`, Vector Post-Training Quantization) targets the same sub-2-bit regime as AQLM but with a much cheaper **second-order** (curvature-aware) codebook optimization per channel, closing most of the AQLM quality gap at a fraction of the quantization cost, closer to GPTQ-scale compute than AQLM-scale.

**QTIP** (`2406.11235`, NeurIPS 2024) is the method that made vector/lattice quantization *practically deployable*: incoherence-process the weights (as in QuIP#) so they're approximately i.i.d. Gaussian, then apply **trellis-coded quantization** with a **bitshift trellis**, a structured trellis whose transition function is a cheap bit-shift, which decouples codebook size (and thus quality) from bitrate (and thus decode cost) in a way lattice codebooks can't. This is the design EXL3 productizes (§4).

### 2.3 Bit-allocation and scaling-law results

**ParetoQ** (`2502.02631`, Meta, NeurIPS 2025) is the paper to read before picking a target bit-width. Training models at matched budgets across 1 / 1.58 / 2 / 3 / 4 bit, it finds a **sharp learning-dynamics transition between 2 and 3 bits**: at ≥3-bit the quantized model's weight distribution stays close to the pretrained one (PTQ-style correction suffices), at ≤2-bit the representation has to change substantially (needs QAT-style training, not post-hoc correction). It also finds 1.58/2/3-bit models dominate 4-bit on the accuracy-per-byte Pareto frontier when trained natively, and that roughly 90% full-precision pretraining + ~10% QAT fine-tuning is near budget-optimal. This transition is *the* reason 3-bit PTQ (GPTQ/AWQ/GGUF Q3_K) reliably works while 2-bit PTQ is fragile and 1.58-bit essentially requires BitNet-style training.

**GSQ** (`2604.18556`) is the most engine-friendly 2026 quantization result: it reaches VQ-class accuracy at 2-3 bit using **plain symmetric group-wise scalar** weights (i.e., a format any existing INT4/INT2 kernel can consume unmodified), by jointly optimizing grid assignment and scale via a Gumbel-Softmax relaxation instead of hand-designed heuristics. If it replicates broadly, it removes the main argument for trellis kernels at 2-3 bit: get the AQLM/QTIP-class accuracy without the bespoke ALU-heavy decode path.

### 2.4 What's new in the last few months (2026, lower confidence: fetched from an arXiv listing scan, not independently reproduced)

**SchurQuant** (`2608.15567`) reports a Schur-complement curvature analysis for groupwise discrete optimization, claiming +11.9pp accuracy over baseline at 2-bit on Qwen3-4B. **QUASAR** (`2608.13966`) is a QAT variant doing continuous loss-aware reconstruction via saliency-weighted least squares, +29% KL-divergence reduction at 2-bit. **FlashQuant** (`2608.15531`) fuses a dense-quantized GEMM path with a sparse-outlier path in one kernel for W4A16 decode, reporting 2.7-4.2× over a naive dequant-then-GEMM baseline, structurally similar to SpQR/SqueezeLLM's dense-and-sparse idea but pushed into the kernel rather than left as two passes. **BCJR-QAT** (`2605.10655`) replaces QTIP's non-differentiable Viterbi argmax with forward-backward sum-product decoding, opening a path to *train* on a trellis code while keeping the same fast inference decoder, relevant if trellis formats want a QAT story analogous to BitNet's. **CCQ** (`2507.07145`) reaches 2.0-2.75 bit with lookup-free encoding (avoiding VQ's codebook-bandwidth cost), demonstrated on DeepSeek-V3. Treat this whole paragraph as leads to verify, not settled results, the pattern of one paper per week claiming a new sub-3-bit SOTA has been constant through 2025-2026 and few have been independently reproduced outside their own repo.

---

## 3. GGUF formats in depth

GGUF (`ggml`/llama.cpp's file format) bundles metadata, tokenizer, and per-tensor quantized data into one file, self-describing enough that `llama-quantize` can mix quant types per tensor within a single file. All formats operate on fixed-size blocks that store one or more scale factors plus packed low-bit codes; the "K-quants" additionally group blocks into 256-element **super-blocks** that share higher-level scale/min metadata, amortizing metadata overhead at larger block counts.

### 3.1 Legacy formats (block size 32, no super-block)

```c
// Q4_0 - 4-bit, single per-block scale, no zero point (symmetric)
typedef struct { ggml_half d; uint8_t qs[16]; } block_q4_0;   // 18 B / 32 wt = 4.5 bpw

// Q4_1 - 4-bit, scale + min (asymmetric)
typedef struct { ggml_half d, m; uint8_t qs[16]; } block_q4_1; // 20 B / 32 wt = 5.0 bpw

// Q8_0 - 8-bit, single scale; the standard "quantize activations on the fly" target
typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;     // 34 B / 32 wt = 8.5 bpw
```
Q5_0/Q5_1 follow the same pattern with a 5-bit code plus 1 extra high-bit array. These are the oldest formats, still used because their dequant is the cheapest possible (one FMA per element) and because Gemma 3 QAT specifically targets Q4_0 (§10).

### 3.2 K-quants (super-block = 256 elements, 6-bit-packed scales)

The K-quants (Q2_K…Q6_K) split 256 weights into 16 sub-blocks of 16 (or for Q3_K/Q6_K, differently sized groups), give each sub-block its own **6-bit-quantized scale** (and for Q2_K/Q4_K/Q5_K, also a 6-bit-quantized *min*), then apply one FP16 **super-block scale** (`d`) and, for asymmetric variants, a super-block **min-scale** (`dmin`) that the sub-block scales/mins are relative to. This is a two-level quantization of the quantization parameters themselves, the sub-block scale doesn't need full FP16 precision because it's a small correction, so packing it into 6 bits barely costs quality but saves a lot of metadata bytes at 256-wide super-blocks.

```c
// Q4_K - the most-used K-quant
typedef struct {
    ggml_half d, dmin;              // super-block scale-of-scales, scale-of-mins
    uint8_t scales[K_SCALE_SIZE];   // 12 B: 8× 6-bit scales + 8× 6-bit mins, packed
    uint8_t qs[QK_K/2];             // 128 B: 256 4-bit codes
} block_q4_K;   // 144 B / 256 wt = 4.5 bpw

// Q6_K - highest-fidelity K-quant, used for output/embedding tensors
typedef struct {
    uint8_t ql[QK_K/2];    // 128 B low 4 bits
    uint8_t qh[QK_K/4];    // 64 B high 2 bits  → 6-bit codes total
    int8_t  scales[QK_K/16]; // 16 B per-sub-block 8-bit scale
    ggml_half d;            // 1 super-block scale
} block_q6_K;   // 210 B / 256 wt = 6.5625 bpw
```
Measured on Llama-3.1-8B (includes embedding/output tensors at their own type, hence the non-round numbers): Q5_K_S 5.5704 bpw, Q5_K_M 5.7036 bpw, Q6_K 6.5633 bpw, Q8_0 8.5008 bpw. Theoretical per-type: **Q2_K 2.5625, Q3_K 3.4375, Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625 bpw.**

### 3.3 I-quants (importance-matrix, codebook-based, sub-4-bit)

The `IQ*` family targets 1-4 bpw using ideas lifted directly from QuIP#/AQLM: instead of a scalar affine grid, low-bit sub-blocks index into a **fixed lattice/grid codebook** of near-optimal vectors (E8/D4-lattice-derived), so the decode step is a table lookup rather than an affine transform. `IQ2_XXS`/`IQ3_XXS` store 16-bit codeword indices (`uint16_t qs[QK_K/8]`) into these codebooks; `IQ4_NL`/`IQ4_XS` use a smaller **non-linear 16-level codebook** as a drop-in, non-superblock-dependent replacement for Q4_0 that is meaningfully better at the same bit count because the 16 levels aren't uniformly spaced. All I-quants are **only usable well when quantized with an importance matrix** (`imatrix`): a per-weight sensitivity file computed by running calibration text through the FP model and accumulating `E[x²]` per input channel (i.e. `diag(H)`, per the unifying theory above); without it the codebook search picks poorly and quality collapses.

```c
typedef struct { ggml_half d; uint16_t qs[QK_K/8]; } block_iq2_xxs;          // ~2.06 bpw
typedef struct { ggml_half d; uint8_t  qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s; // ~1.56 bpw
typedef struct { ggml_half d; uint8_t qs[16]; } block_iq4_nl;               // 4.5 bpw, QK4_NL=32
typedef struct { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs; // 4.25 bpw
```
Range: **IQ1_S ≈1.56, IQ1_M ≈1.75, IQ2_XXS ≈2.06, IQ2_XS/S/M ≈2.3-2.7, IQ3_XXS ≈3.06, IQ3_XS/S/M ≈3.3-3.7, IQ4_XS 4.25, IQ4_NL 4.5 bpw.** These are the formats that make sub-3-bit GGUF coherent at all, and are what Unsloth's `UD-Q2_K_XL`/1-bit dynamic quants build on top of for huge MoEs (§16).

### 3.4 Ternary types (TQ1_0, TQ2_0)

Added for BitNet-class {-1, 0, +1} models. `TQ1_0` packs weights in **base-3**: 5 ternary trits fit in one byte (`3⁵ = 243 < 256`), giving `(256 − 4·256/64)/5 ≈ 1.6875` bpw plus a small `qh` side-array and one FP16 scale, effectively **~1.69 bpw**, the tightest lossless-ish encoding of a ternary weight ggml ships. `TQ2_0` is simpler and slightly larger: two bits per trit with one wasted code, uniformly packed, **2.0625 bpw**, faster to unpack (no base-243 division) at a small size cost. Both need a native ternary-trained model (BitNet b1.58, §10): quantizing an ordinary FP16 model down to ternary post-hoc does not work.

### 3.5 How `llama-quantize` picks per-tensor types and the S/M/L mixes

`llama-quantize`'s named presets (e.g. `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_K_S`, `Q4_K_M`) are not a single uniform quant type applied everywhere, each preset is a **per-tensor-role policy table**. The general pattern documented across llama.cpp community writeups and the tool's own heuristics:

- The **base K-quant level** (the number in the name) is applied to the bulk of the FFN up/gate/down and attention Q/K weights.
- **`attn.wv` (value projection), `attn.wo` (output projection), and `ffn_down`** are widely regarded as more sensitive to quantization error (they sit closer to the residual stream / are harder to error-correct downstream) and get bumped to a **higher K-quant** than the nominal level, this is what differentiates `_S` (no bump, smallest), `_M` (bump these three), and `_L` (bump further / more tensors) within the same nominal bit family.
- **`output.weight` (the LM head) and `token_embd.weight`** are usually kept at **Q6_K or Q8_0** regardless of the overall preset, because they are a small fraction of total parameters but disproportionately affect measured perplexity, this is the single most consistent override across every preset.
- With `--imatrix`, the tool additionally uses the sensitivity data to decide, tensor by tensor, whether a given assignment would cause disproportionate error, and can fall back to a higher type for specific layers.
- `--tensor-type <regex>=<type>` lets a user override any of this manually, the mechanism Unsloth's UD quants and ik_llama.cpp's custom mixes are built on.

### 3.6 Unsloth Dynamic (UD) quants

Unsloth's Dynamic 2.0 methodology generalizes the "bump sensitive tensors" idea from a fixed three-tensor rule to a **per-layer, per-model search over every tensor**, evaluated not by perplexity but by **KL-divergence against the BF16 reference model's output distribution**, with a custom benchmarking harness built to reproduce official MMLU numbers before trusting comparisons. Concretely: "rather than modifying only select layers, we now dynamically adjust the quantization type of every possible layer," and the selected layers differ substantially model to model ("layers quantized in Gemma 3 differ significantly from those in Llama 4"). On Gemma 3 12B, `UD-Q3_K_XL` reaches **0.081 KLD** versus a plain imatrix `Q3_K_XL`'s **0.088 KLD**, smaller error at comparable size, i.e. it's a strictly better tensor-assignment policy than the static S/M/L heuristic, at the cost of a model-specific search pass instead of a fixed table. Reported concrete sizes for Kimi K2.7 Code: `UD-Q2_K_XL` 339 GB / PPL 2.4131, `UD-Q4_K_XL` 584 GB / PPL 1.8420, `UD-Q8_K_XL` 595 GB / PPL 1.8419 ("truly lossless" relative to full precision). Caveat: UD variants can be measurably *slower* to run than plain `Q4_K_M` at similar file size, because the mixed-type layout defeats some of llama.cpp's per-type kernel batching.

### 3.7 ik_llama.cpp's extended type zoo

ik_llama.cpp (a hard fork maintained for CPU/hybrid performance) ships a much larger type space than mainline:

- **`IQ*_K` family** (`IQ2_K`, `IQ3_K`, `IQ4_K`, `IQ5_K`, `IQ6_K`, plus `_KS`/`_KSS`/`_KL` sub-variants like `IQ4_KS`, `IQ2_KS`, `IQ4_KSS`, `IQ2_KL`): a redesigned super-block layout distinct from mainline's `IQ*` codebooks, tuned for CPU SIMD dequant throughput as much as for quality; these are ik_llama.cpp's own answer to "what would K-quants look like if you optimized jointly for size and CPU decode speed."
- **`IQ*_KT` trellis family** (`IQ1_KT`…`IQ4_KT`): CPU trellis quantization using, per the project's own description, "a novel, integer-base trellis, which allows to achieve reasonable CPU performance," i.e. the same QTIP/EXL3 idea (rotate + trellis-code) but with a trellis transition function cheap enough for CPU vector units rather than requiring GPU tensor cores. This is the CPU-side accuracy frontier below 3 bpw. It is **not always a win**: per the synthesis notes, `iq4_kt` measurably loses to `iq4_ks` on both PPL *and* speed on Qwen3.6-27B, trellis decode's extra ALU work isn't free, and when a scalar K-quant is already close to bandwidth-bound, adding compute only hurts.
- **`_R4`/`_R8` repacked types**, row-interleaved variants (4 or 8 rows packed together) of the base types, restructuring memory layout so one SIMD load/shuffle feeds a full dequant tile instead of needing per-row unpacking; the same idea as llama.cpp's mainline `mmq`/`mmvq` repack path (§9) but extended across ik's whole type zoo.

---

## 4. EXL2 and EXL3

**EXL2** (ExLlamaV2, turboderp) generalized GPTQ-style quantization to **fractional, mixed bits-per-weight**: a measurement pass computes the quantization error at several candidate bit levels per linear layer, then a global search allocates bits across layers to hit a target average bpw (e.g. 4.65) while minimizing total error, effectively a knapsack over per-layer error-vs-size curves. It renames/reshapes some tensors internally for its packed kernel layout, which is part of why EXL2 files aren't portable outside the ExLlama family.

**EXL3** (ExLlamaV3, shipped in July 2026) replaces this with a QTIP-derived pipeline: the README describes it as "a streamlined variant of QTIP from Cornell RelaxML." Its headline engineering contribution is a **fused Viterbi quantizer** that computes Hessians on the fly and does calibration + trellis quantization in a single pass, turning AQLM's "hundreds of A100-hours" into **minutes for small models, a few hours for 70B+ on one RTX 4090**. It **largely preserves the original tensor structure** (unlike EXL2), which the maintainers note could enable broader framework interoperability beyond ExLlamaV3 itself. Quality: **Llama-3.1-70B-EXL3 is reported coherent down to 1.6 bpw**, and with the output layer bumped to 3 bpw the model fits under 16 GB VRAM. On the kernel side, EXL3 uses a **Marlin-inspired GEMM kernel** that "achieves roughly memory-bound latency under optimal conditions (4 bpw, RTX 4090)," with the maintainers flagging that Ampere and low-bitrate cases still need optimization work, i.e. the trellis decode ALU cost is not yet fully hidden behind memory latency everywhere, echoing the ik_llama.cpp `_KT` caveat above. ExLlamaV3 also ships `eval/qbench.py`, described in the synthesis notes as the only cross-engine perplexity/KLD evaluation tool found in this survey, useful precisely because EXL3, GGUF, and datacenter FP4 formats otherwise have no shared benchmark harness.

The EXL2→EXL3 transition is a clean illustration of the field's direction: **mixed-bit scalar allocation (EXL2) lost to rotate-then-trellis-code (EXL3)** for the same reason QuIP# beat plain GPTQ at 2 bit, incoherence processing plus a strong code, not clever bit allocation alone, is what buys quality at very low bpw. The cost is the same everywhere trellis appears: decode needs real ALU work per weight (walking the trellis / evaluating the codebook), so it only wins when memory bandwidth, not compute, is the binding constraint, which is usually true at batch 1, and not reliably true once the block is small or the GPU generation is older (Ampere vs Ada/Blackwell).

---

## 5. MLX quantization

MLX (Apple's array framework, the base of `mlx-lm`) ships two families of quantized linear layers:

**Affine group quantization**, the default, and MLX's equivalent of RTN/GPTQ-style scalar quantization: weights are split into groups of size **32, 64, or 128**, and each group gets an independent `scale` and `bias` such that `value ≈ scale · quantized + bias`, with `scale = (max − min)/(2^bits − 1)`, `bias = min`, `quantized = round((value − bias)/scale)`. Supported bit-widths are **2, 3, 4, 5, 6, and 8**, a wider native range than GGUF's fixed set, because MLX generates the pack/unpack kernels rather than hand-writing one struct per width.

**Floating-point microscaling modes**. MLX also implements **MXFP4** and **MXFP8** (group size 32, shared power-of-two E8M0 exponent, no bias term) and **NVFP4** (group size 16, optional global FP32 scale on top of the per-block FP8 scale, also no bias) as first-class quantization modes, aligning MLX with the same OCP/NVIDIA microscaling formats used in datacenter checkpoints (§7): relevant as Apple Silicon (M5 generation) starts to expose hardware paths that reward these layouts.

**DWQ (distilled/dynamic weight quantization)**, shipped in `mlx-lm`, is a post-training refinement layered on top of affine group quantization rather than a new storage format: per an `mlx-community` model card, a DWQ model is produced "by distilling from the 6-bit to the 4-bit quantization" of the same model, i.e. the higher-bit quantized model is used as a teacher and the lower-bit target's quantization parameters (scales/zero-points, via a straight-through estimator) are optimized against the teacher's output distribution, rather than against a weight-reconstruction or activation-reconstruction error alone. This is the same instinct as Unsloth's KLD-driven layer selection and OmniQuant's learnable transforms, but implemented as gradient-based distillation directly on the deployed 4-bit weights rather than as a discrete layer/type search.

---

## 6. bitsandbytes: NF4 and LLM.int8()

**LLM.int8()** (Dettmers et al., 2022) is a **mixed-precision matmul**, not a stored weight format: at inference time, activation columns whose magnitude exceeds a threshold (empirically, "outliers with magnitude ≥6" recovers full performance) are routed through an **FP16 path**, while the remaining, well-behaved columns go through an **INT8 path** using per-row/per-column absmax scaling; the two partial results are summed and dequantized. The finding motivating this design: ordinary uniform INT8 quantization **fails specifically above ~6B parameters**, because large transformers develop *systematic* outlier feature dimensions present in every layer, a structural property of scale, not noise, so any scheme that quantizes those columns uniformly destroys them. With outlier extraction, the paper reports effectively zero measured degradation on OPT-175B and BLOOM-176B.

**NF4 (4-bit NormalFloat)**, introduced with QLoRA, is a different idea from GGUF/GPTQ/AWQ-style *affine* 4-bit: instead of a uniform grid, NF4's 16 quantization levels are the **quantiles of a standard normal distribution**, which is information-theoretically optimal *specifically because pretrained-then-frozen weight tensors are empirically close to zero-mean-normal*, an affine int4 grid wastes representational capacity on the flat tails where few weights live, NF4's levels concentrate near zero where the density is highest. QLoRA pairs this with **double quantization**: the per-block FP32 scale factors are themselves quantized (to 8-bit, with their own second-level block scale), shaving roughly 0.4 bit/parameter of pure metadata overhead, the same "quantize the quantization parameters" trick K-quants use at 256-wide super-blocks, just applied to bitsandbytes' typically 64-wide blocks. bitsandbytes today is used less for standalone inference (its dequant path is comparatively slow and its adoption for pure serving has been eclipsed by GGUF/AWQ/EXL3) and mostly survives as the **QLoRA fine-tuning** backend, where NF4 + double quantization is what lets a 65B model fine-tune on a single 48GB GPU.

---

## 7. FP8 and microscaling (NVFP4/MXFP4/MXFP8)

**FP8** comes in two IEEE-adjacent flavors used throughout inference: **E4M3** (4 exponent, 3 mantissa bits, more precision, less range, the default for weights/activations) and **E5M2** (5 exponent, 2 mantissa, more range, less precision, historically preferred for gradients). Both are natively supported by Hopper and Blackwell tensor cores, and are the safe, nearly-free 2× compression step from BF16 that most datacenter serving stacks apply by default before touching 4-bit formats.

**Microscaling (MX)** formats generalize per-tensor/per-channel scaling to **small shared groups**, standardized by the OCP MX spec: **MXFP4** is a 4-bit E2M1 float value with a **group of 32** sharing one **E8M0 (power-of-two) scale**: 4.25 bpw all-in. **MXFP8** is the 8-bit analog, also group-32/E8M0. **NVFP4** is NVIDIA's variant: still E2M1 4-bit values, but a **tighter group of 16** and an **FP8 E4M3 block scale** (not just a power of two) *plus* an optional tensor-level FP32 scale on top, the finer group and the non-power-of-two block scale give NVFP4 measurably better fidelity than MXFP4 at very slightly more metadata (4.5 vs 4.25 bpw). NVFP4 needs Blackwell; MXFP4 has broader hardware support (Hopper via software emulation, native on Blackwell) and is what gpt-oss ships.

Two 2026 results matter for anyone implementing these formats. First, **"Unveiling the Potential of Quantization with MXFP4"** (`2603.08713`) shows the accuracy gap between MXFP4 and NVFP4, naively ~10% relative, can be closed to **under 1%** with two pure-software techniques (Overflow-Aware Scaling and Macro Block Scaling), meaning the *hardware-support* argument for preferring NVFP4 over MXFP4 mostly evaporates if your quantizer is good. Second, **"Block Rotation is All You Need for MXFP4"** (`2511.04214`) found that naively porting QuaRot-style *global* Hadamard rotation onto MXFP4 loses most of the rotation's benefit, because MXFP4's power-of-two per-block scale interacts badly with a rotation that mixes values across block boundaries, you need a rotation matched to the block structure (block-local Hadamard), not a global one. Practically: **MXFP4 on MoE experts + FP8 (or MXFP8) on attention** is the dominant 2026 production recipe (GLM-5.2, Hy3, Kimi-K3), and gpt-oss's config literally carries `"quant_method": "mxfp4"` with `modules_to_not_convert` excluding self-attention, the MoE router, embeddings, and the LM head, i.e. even native-MXFP4 checkpoints keep the routing-sensitive and small-but-critical tensors out of 4-bit.

---

## 8. Activation quantization: W4A16 vs W4A8 vs W4A4 vs W8A8

The naming convention `WxAy` gives weight bits then activation bits (KV cache bits sometimes appended as `KVz`). The four regimes trade off differently:

- **W4A16** (weights 4-bit, activations stay FP16/BF16) is the consumer default. GGUF K-quants, AWQ, GPTQ-Marlin all live here. Decode at batch 1 is pure bandwidth savings on the weight read; the GEMV's activation vector is tiny regardless of its precision, so there's no compute-side reason to quantize activations at batch 1. The cost shows up at **larger batch / prefill**, where the GEMM becomes compute-bound and FP16 activations mean you're still doing full-precision tensor-core work per token.
- **W8A8** (SmoothQuant-style) gets both operands into INT8, so tensor cores run the matmul natively in int8 rather than dequantizing weights back to FP16 first, a real compute-side win at higher batch/prefill, and the standard TensorRT-LLM/vLLM datacenter path for models that don't need lower than 8-bit.
- **W4A8** is the awkward-sounding but well-motivated middle: **QServe** (`2405.04532`, MLSys 2025) is the canonical explanation of *why not W4A4*. On current tensor cores, once you've quantized weights to 4-bit, the dequantization overhead of getting them to match an 8-bit activation format essentially disappears (the dequant is cheap relative to the 8-bit MMA), while 4-bit activations require handling activation outliers that 8-bit tolerates for free. QServe's recipe is specifically **W4A8KV4**: 4-bit weights, 8-bit activations, 4-bit KV cache. **LiquidGEMM** (`2509.01229`) is a directly reusable W4A8 GEMM kernel design following the same logic.
- **W4A4** needs outlier suppression to work at all, either rotation (QuaRot/SpinQuant/FlatQuant, §12) or a format-aware trick. **MXSens** (`2607.17733`) reaches **W4A4KV4 without rotation**, skipping the online-Hadamard tax entirely, which matters because rotation is itself a real runtime cost (a matmul-shaped operation inserted into the hot path) that eats into the theoretical W4A4 speedup.

The practical rule of thumb an engine should encode: **activation quantization only pays off once you're compute-bound** (prefill, or decode at batch > ~8-16). Below that, quantizing activations adds a runtime cost (extra scale computation, sometimes a rotation) for no benefit, since the GEMV is DRAM-bound regardless of activation precision, this is the same gap flagged in §17 and in `00-synthesis.md` §5: surveyed methods do not condition bit allocation on where or how the tensor will run.

---

## 9. Kernels: making low-bit weights fast

*(Full derivation and measured numbers are in `05-kernels-and-systems.md` §2; this section summarizes the quantization-specific parts.)*

The core problem every W4A16 kernel solves: at batch 1, the matmul is a **GEMV** with zero weight reuse, so what matters is reading quantized weights at full DRAM bandwidth and dequantizing them overlapped with the memory fetch, not adding latency. The canonical trick, used by Marlin, AWQ's kernel, TensorRT-LLM, and FasterTransformer alike, is the **LOP3 int4→fp16 bit trick**: rather than an expensive `__int2half` conversion, note that an FP16 value with exponent bits fixed to `0x64` numerically equals `1024.0 + x` for small integer `x`, so converting a nibble to a usable float is a single 3-input LUT/permute instruction (`lop3.b32`/`prmt`) plus one subtract, not a real int-to-float conversion.

**Marlin** (PPoPP 2025) builds a full kernel around this: asynchronous global-memory loads (`cp.async`) into a circular shared-memory queue, striped weight-matrix partitioning across SMs with dual pipelines so memory and compute stay simultaneously busy, and, critically, a **GPTQ-compatible pre-permuted weight layout** computed offline so the dequantized fragments land exactly where the tensor-core `mma` instruction expects them, with zero runtime shuffle cost. It sustains close to the ideal 4× FP16 speedup from batch 1 up to ~64, where naive int4 GEMV kernels fall off a cliff past batch ~8.

**Machete** rebuilds the Marlin idea on CUTLASS 3.5.1 for Hopper, using `wgmma` and TMA bulk-copy instead of `cp.async`, reported **~29-32% throughput gain over Marlin at ≥3 req/s**, i.e. once the workload has enough concurrent requests to benefit from Hopper's async-copy and warp-specialization machinery.

**GemLite** (Mobius/PyTorch) is the Triton-first answer: a family of kernels (dense GEMM, split-K GEMM for batched decode, GEMV, and a newer "GEMV RevSplit-K" specifically tuned for batch-1) covering an unusually wide format matrix: **FP16×W8/W4/W2/W1, FP8×FP8, FP8×Wn, INT8×INT8, INT8×Wn, MXFPn×MXFPn, and NVFP4**, trading a little peak performance for being trivial to extend to odd bit-widths from Python. On Llama-3.1-8B (RTX PRO 6000), reported MXFP4 numbers: prefill 7.3ms vs FP16's 15.4ms, decode 4.84s vs FP16's 11.75s batch aggregate: "up to 7-8× faster prefill and 3-6× faster decode" vs default Torch AO kernels.

**CUTLASS mixed-input GEMM** is the underlying NVIDIA-maintained primitive Machete and much of TensorRT-LLM's W4A16/W8A16 path builds on, it formalizes "one operand at low bit-width, the other at high bit-width, dequantize during the mainloop" as a first-class template family rather than a hand-written kernel per model.

**On CPU/Metal/mobile, the story flips from bit-trick dequant to lookup tables.** GGUF's `mmq`/`mmvq` path quantizes activations to Q8_0/Q8_1 on the fly and issues `__dp4a`/`vpdpbusd`/`smmla` int8 dot products directly on packed nibbles, applying per-block FP16 scales at the accumulator, dequant and dot product fused into one instruction sequence, never materializing an FP16 weight. **T-MAC** and its successors (**Vec-LUT**, **T-SAR**) go further and replace dequantize-then-multiply with **bit-serial table lookups** (`tbl`/`pshufb`) entirely, so cost scales *linearly with bit-width* instead of paying a fixed dequant tax, the right model when the bit-width is very low (1-2 bit, BitNet-class) and ALU throughput, not bandwidth alone, becomes the limiter. **bitnet.cpp** is the shipping reference implementation of ternary CPU kernels built on this idea. The consistent empirical caveat across this whole kernel family (echoed by ik_llama.cpp's `_KT` trellis result and EXL3's Ampere caveat): **LUT/trellis/ALU-heavy formats only win when they're still memory-bound after adding the extra compute**, hand-tuned AVX-512/AMX microkernels still beat generic LUT kernels by ~2.2× when the comparison is apples-to-apples on the same hardware.

---

## 10. QAT and low-bit-native models

**BitNet b1.58** defines the ternary ({-1,0,+1}) no-multiplies kernel model, weights need only add/subtract, not multiply, which is the whole point of going this low. **BitNet b1.58 2B4T** is the reference open artifact for benchmarking ternary kernels against. **BitNet v2** adds **H-BitLinear**, an online Hadamard transform that enables *native 4-bit activations* (W1.58A4 rather than wasting the activation path on 8-bit when the weight side is already ternary): using the same Hadamard primitive QuaRot needs for its rotation, so a single fast-Hadamard kernel serves both lineages. **BitNet Distillation** removes the "must pretrain from scratch" blocker: it converts an off-the-shelf model (demonstrated on Qwen3) to 1.58-bit via SubLN insertion, a continual warm-up phase, and attention-map distillation from the FP teacher, meaning ternary is no longer only available for models trained ternary from step one.

**Gemma 3 QAT** (Google, no paper, blog + checkpoints) is the cleanest existence proof that a **modest QAT budget makes a GGUF-native legacy format near-lossless**: roughly 5,000 QAT steps, using the *non-quantized* checkpoint's output probabilities as the training target, gets Q4_0 (the cheapest-to-dequantize, oldest GGUF format) close enough to FP16 that essentially every consumer engine adopted the released Q4_0 checkpoint immediately. Gemma 3 27B goes from 54GB to 14.1GB with the accuracy hit largely eliminated by the QAT step. This is a strong argument that **checkpoint publishers doing a small amount of QAT work buys far more than any PTQ trick downstream**, and it's the same lesson gpt-oss (native MXFP4), Kimi K2-Thinking (native INT4 via QAT), and Nemotron 3 (native NVFP4) generalize: **checkpoints increasingly arrive already quantized**, which shifts the engine's job from "run a good PTQ pipeline" to "have a fast dequant-fused GEMM for whatever native format the checkpoint ships in", a real architectural implication (§17).

---

## 11. KV cache quantization

KV cache cost is `2 × n_layers × n_kv_heads × head_dim × bytes/elem` per token; at 128K context this is many GB even after GQA, so it quantizes for the same bandwidth reason weights do, and because in agentic/long-context workloads the KV cache, not the weights, is what stops fitting.

**K and V are not symmetric and shouldn't share a scheme.** Keys have *persistent channel* outliers, a few fixed dimensions stay large across nearly every token, so per-token scaling lets those outlier channels eat the whole quantization range; the fix is **per-channel (not per-token) scaling for K**. Values vary more per-token with less stable channel structure, so **per-token grouping works better for V**. **KIVI** and **KVQuant** are the reference papers establishing this asymmetric per-channel-K/per-token-V design, both landing around INT4/INT2 with an FP16 residual window for the most recent tokens (which are read repeatedly and are worth keeping exact) and higher precision for attention-sink tokens. RoPE complicates K quantization specifically because it rotates coordinate *pairs*, quantizing pre-RoPE keys, or using rotation-aware channel grouping, avoids smearing outlier structure across the rotation.

In production engines this shows up as simple flags: **llama.cpp's `-ctk`/`-ctv`** select per-tensor cache types independently: `q8_0` (halves cache, close to free quality-wise) and `q4_0`/`iq4_nl` (quarters it, real quality risk) are the common choices, and **quantized KV requires flash attention** to avoid becoming *slower* than FP16 KV, because the dequant is fused inside the attention kernel rather than done as a separate pass (§1.5 of `05-kernels-and-systems.md` covers a worked example where FP8 KV dequant, not the MMA itself, was the bottleneck: 50 cycles/token of dequant vs 34 of tensor-core work, until the kernel was restructured to share dequant work across query heads via distributed shared memory). **ExLlama's Q4/Q6/Q8 cache modes** additionally apply a **Hadamard rotation** to the cache before quantizing it, the same incoherence-processing idea as QuaRot/QuIP, applied specifically to make the K/V distribution friendlier to a scalar quantizer.

**Sensitivity is architecture-specific, and this is one of the more surprising 2026 measurements.** KLD numbers gathered across models: Qwen3.6-27B dense and Qwen3.6-35B-A3B MoE both tolerate q8_0 KV well (<0.04 KLD) and degrade moderately at q4_0 (0.087-0.117). Gemma 4 26B-A4B MoE is dramatically more fragile: **0.377 KLD at q8_0 and 1.088 at q4_0**, roughly 3.5× worse than its own dense 31B sibling at the same cache precision. The MoE router appears to *amplify* KV-cache quantization error rather than being independent of it, plausible mechanism: routing decisions are sensitive to small perturbations in the attended representation, so cache noise that a dense model's FFN would just absorb instead flips expert selection somewhere downstream. Practical consequence: **asymmetric `--cache-type-k q4_0 --cache-type-v q8_0` beats symmetric q4 in most measured cases**, and GQA/MQA models (fewer KV heads to protect) tolerate cache quantization better than MHA models in general, but the Gemma-vs-Qwen gap means **a new engine should not ship one global default**; it should measure per-architecture, ideally per-checkpoint.

**TurboQuant** and other transform-coding approaches to KV (e.g. `2608.14191`'s reverse-water-filling rate-distortion allocation, reported 5.8× compression near-lossless) reframe KV quantization as a classical signal-processing rate-distortion problem rather than a heuristic per-channel/per-token split, worth watching as a more principled successor to the KIVI/KVQuant heuristic, though not yet as widely deployed.

For sparse/compressed attention (MLA, NSA, DSA, covered in depth in `01`/`06`/`05`), the situation is qualitatively different: **these are trained into the model, not retrofittable.** MLA's low-rank latent KV cannot be losslessly reconstructed into an MHA cache and then quantized by the schemes above; it gets its own (much smaller, ~656 bytes/token per DeepSeek's public numbers) native representation that quantizes on its own terms.

---

## 12. Hadamard/rotation tricks

The unifying purpose of every rotation method: multiply activations and weights by a matrix `R` (with `RRᵀ = I`, so the linear layer's output is unchanged, `WX = (WRᵀ)(RX)`) chosen so that **outlier mass gets spread evenly across coordinates**, making the post-rotation distribution closer to isotropic Gaussian, exactly the condition under which a cheap scalar or block quantizer approximates the full-Hessian optimum well. Hadamard matrices are the default choice because a **Fast Hadamard Transform** costs `O(n log n)`, not `O(n²)`, so the rotation itself doesn't eat the speedup it enables.

- **QuaRot** inserts *random* Hadamard rotations that can be algebraically absorbed into adjacent weight matrices (so no extra runtime matmul survives after an offline fusion pass), enabling W4A4 with much smaller accuracy loss than unrotated W4A4.
- **SpinQuant** replaces the random rotation with a **learned** one (still orthogonal, but optimized rather than random), closing more of the accuracy gap at the same runtime cost.
- **FlatQuant** goes further still: **per-layer affine, Kronecker-decomposed** transforms (beyond orthogonal rotations) that directly optimize post-transform "flatness," reporting <1% degradation at W4A4 on Llama-3-70B, the accuracy leader in this family, at the cost of needing a fused per-layer transform kernel rather than one global Hadamard.
- **ReSpinQuant** recovers most of FlatQuant's expressivity while keeping the residual rotation fusible offline, so runtime cost stays at SpinQuant's level.
- **ParoQuant** swaps the dense Hadamard for independent **Givens rotations**, much cheaper and trivially parallelizable, a good fit specifically for decode-bound serving where every extra op matters.

**The MXFP4-specific finding is important enough to flag on its own**: **"Block Rotation is All You Need for MXFP4"** (`2511.04214`) shows that a *global* Hadamard rotation and MXFP4's *power-of-two per-block* scaling actively fight each other, the rotation mixes values across block boundaries in a way that breaks the block-local scale's assumptions, so **naively porting QuaRot to MXFP4 loses most of the intended benefit**. The fix is a rotation matched to the block structure (block-local Hadamard, or TORQ's two-level rotation matching MX's two-level scale hierarchy). This is a concrete, checkable engineering trap: *any rotation-based quantizer ported to a microscaling format needs to respect the format's block boundaries, not its bit-width alone.*

---

## 13. MoE quantization

MoE quantization is structurally harder than dense quantization because **calibration coverage is uneven by construction**: with a top-k router over hundreds of experts, most experts see a small, workload-dependent slice of calibration tokens, so naive per-tensor calibration statistics for rarely-routed experts are noisy or absent. Approaches split into a few families:

- **Coverage fixes**: **MoEQuant** (ICML 2025) uses expert-balanced sampling during calibration specifically to give rarely-routed experts enough tokens to compute reliable statistics.
- **Calibration-free**: **AlphaQ** allocates per-expert bit-widths using a heavy-tailed self-regularization signal computed from the weights alone, without needing calibration data to see each expert at all, useful when you can't guarantee coverage.
- **Router refit**: **GEMQ** re-fits the router *after* quantizing the experts, addressing a failure mode the community calls out concretely, quantization shifts each expert's effective function slightly, so the pre-quantization router's top-k choices are no longer optimal, and users report the router "picks wrong experts at Q4" on some MLX/GGUF quantizations that skip this step.
- **Serving-time policy**: **Dynamic Expert Quantization** treats per-expert precision as an *online* budget driven by observed routing traffic rather than a fixed offline choice, hot experts stay high-precision, cold ones get squeezed harder, adjustable as the workload shifts.
- **Storage-format-first**: **Tied Trit-Planes** optimizes the on-disk layout for SSD-streamed MoE serving specifically, where the bottleneck for huge sparse models (Kimi K2-class, 1T+ params) is disk/PCIe streaming bandwidth, not the GEMM.
- **Structural alternative, pruning instead of quantizing**: **REAP** (Router-weighted Expert Activation Pruning) removes whole experts rather than compressing them, ranking experts by a combination of router gate-values and average activation norm and dropping the least-contributing ones. At **50% expert removal**, REAP reports **near-lossless compression on code generation tasks** for Qwen3-Coder-480B and Kimi-K2, and a mean **1.9% accuracy drop** across a broader non-agentic benchmark set on smaller 20-30B-class MoEs (ERNIE-4.5-21B, Qwen3-30B, Mixtral-8x7B, GLM-4.5-Air, Llama-4-Scout-17B). Its stated advantage over *merging* experts (an older MoE-compression idea) is that pruning preserves the router's ability to modulate the surviving experts independently, avoiding "functional subspace collapse" that merging causes. **ExactMoE** (`2608.15383`, low confidence per §2.4's caveat) reports a related but opposite-direction approach, symmetric group-128 4-bit quantization applied to routed experts while keeping router/attention in BF16, claiming 99.23% accuracy retention, i.e. "quantize the experts, protect the router" as an alternative to REAP's "remove the weak experts, keep the router adjusting the rest."

The practical takeaway repeated across `01`, `06`, and `07`'s notes: **MoE routers appear to be the most quantization-sensitive component in the whole stack**, more sensitive than attention, more sensitive than the FFN weights themselves, showing up independently in the KV-cache Gemma-MoE amplification finding (§11) and in the router-refit and REAP results here. Any new engine's MoE support should treat the router as a protected tensor by default and make expert-level bit allocation a first-class, per-checkpoint-tunable knob rather than inheriting whatever bit-width the rest of the model uses.

---

## 14. Evaluating quantization: PPL, KLD, and agentic degradation

**Perplexity is a weak, easily-gamed metric for comparing quantization methods.** It's a single scalar aggregate over a token stream, insensitive to *which* tokens get worse, and vulnerable to calibration-set overfitting (Unsloth's own writeup flags Wikipedia-based calibration sets specifically for artificially inflating scores on similarity-matched test sets). Two better tools have become the de facto standard in 2026:

- **KL-divergence against the BF16 reference model's output distribution**, token by token over a held-out set, this is what Unsloth's UD quants, MLX's DWQ, and ExLlamaV3's `qbench` all optimize against or report, because it directly measures how much the quantized model's *next-token distribution* diverges from the original, which correlates much better with downstream task degradation than perplexity's aggregate loss.
- **Top-token agreement** (does the quantized model's argmax match the reference's argmax) as a cheap secondary check, catching cases where KLD is low on average but the model has flipped its answer on specific high-stakes tokens.

**ExLlamaV3's `eval/qbench.py`** is, per this survey's earlier notes, the only tool found that lets you compare quantization methods *across* engines (GGUF, EXL3, raw safetensors) on a common metric rather than each project reporting its own incomparable perplexity number on its own calibration set, a real gap for anyone trying to make an apples-to-apples format decision today.

**Two 2026 findings should change how anyone reads a "4-bit is fine" accuracy dashboard:**

1. **"Quantization Inflates Reasoning"** (`2606.25519`): low-bit reasoning models frequently **preserve final-answer accuracy while emitting substantially longer chains of thought** to get there. A dashboard that only tracks task accuracy will report the 4-bit model as equivalent to FP16, while cost-per-request (and latency) silently rises because the model is now thinking longer to compensate for degraded reasoning per step. **Measure tokens generated, not correctness alone.** QAT is reported as the best mitigation.
2. **"Can Compressed LLMs Truly Act?"** (`2505.19433`) and the related **UniComp** finding: 4-bit quantization preserves **workflow/plan generation** well (1-3% drop) but degrades **real agentic execution performance by 10-15%**, and more broadly, compression (pruning, quantization, and distillation alike) shows a consistent bias where **factual recall survives while multi-step reasoning degrades**. This is why RAG-style lookup workloads tolerate aggressive quantization far better than agentic coding/tool-use workloads, and it's a reason a "safe" 4-bit quant for chat can be a bad choice for an agent loop.

The engineering consequence: an evaluation suite for a new engine should default to **KLD + top-token-agreement against BF16**, and for any agentic/reasoning-heavy target workload, add a **token-length/step-count metric** alongside accuracy, accuracy alone will miss the two failure modes above.

---

## 15. Where quality breaks by bit-width

Synthesizing ParetoQ (§2.3), the GGUF I-quant/K-quant transition (§3), and the trellis-vs-scalar results (§4):

| Bit-width | What's viable | Why |
|---|---|---|
| **8-bit** | RTN is fine | Outliers small relative to 8-bit's range; W8A8 also viable for compute-bound cases |
| **4-6 bit** | Any calibrated PTQ (GPTQ/AWQ/K-quants); RTN degrades but is usable | Weight distribution stays close to pretrained; PTQ corrects residual error well |
| **3 bit** | Calibrated PTQ still works (GPTQ/AWQ/Q3_K/IQ3); RTN mostly breaks | Still "above" ParetoQ's transition-representation hasn't had to fundamentally change |
| **2-2.5 bit** | Needs vector/trellis quantization (QuIP#/AQLM/QTIP/EXL3/GSQ) or QAT; plain scalar PTQ degrades sharply | **ParetoQ's sharp transition**: weight representation must change substantially, which a reconstruction-error-minimizing PTQ pass alone can't achieve as reliably as it can at ≥3-bit |
| **1.6-1.75 bit** | Only trellis/lattice VQ (EXL3, QuIP#-derived): the frontier of "still coherent" | Needs both incoherence processing *and* a strong code; this is roughly EXL3's reported floor for a 70B model with a bumped output layer |
| **1.58 bit (ternary)** | Requires native QAT training (BitNet) or distillation into a ternary target (BitNet Distillation): PTQ does not work | No amount of post-hoc correction reaches this regime; the model has to be *trained* to have a ternary-friendly weight distribution |

The one-sentence version for an engine's default policy: **≥3-bit → any decent PTQ method is fine, pick for speed; 2-3 bit → you need vector/trellis quantization or you forgo real quality; below 2 bit → you need a QAT-native or QAT-distilled checkpoint, not a quantizer.**

---

## 16. Practical recipes: running 70B-1T models on consumer hardware

Figures below are gathered from community benchmark threads, `07-consumer-hardware-practice.md`'s hardware survey, and vendor/project pages; treat tok/s numbers as representative rather than precisely reproducible (hardware, prompt length, and llama.cpp version all move the number).

| Model class | Budget | Recommended quant | Approx. file size | Approx. decode t/s |
|---|---|---|---|---|
| **70B dense** (Llama-3.1-70B class) | 24 GB | EXL3 ~2.5-3 bpw or GGUF `IQ2_XS`/`IQ3_XXS`, CPU offload for overflow | ~22-26 GB | single digits-teens, offload-limited |
| **70B dense** | 48 GB | GGUF `Q4_K_M` or EXL3 4.0 bpw, fits mostly in VRAM | ~40 GB | 15-30 t/s on a fast card |
| **70B dense** | 96 GB (e.g. RTX PRO 6000) | `Q6_K`/`Q8_0` or EXL3 6 bpw, comfortable fit | ~55-70 GB | 30-50+ t/s |
| **120B MoE, ~5B active** (gpt-oss-120b) | 24 GB + 64 GB RAM, MoE offload | Native MXFP4 GGUF, `--n-cpu-moe` tuned to fit | ~59 GB total | 10-22 t/s (2×3090+64GB RAM), 40-56 t/s on Strix Halo 128GB, 33.5 t/s DGX Spark, 182 t/s on 3×4090 |
| **235B MoE (Qwen3-235B-A22B class)** | 48-96 GB + RAM offload | `UD-Q3_K_XL`/`IQ3_S` GGUF with `-ot exps=CPU` | ~90-110 GB | single digits to ~15 t/s depending on offload ratio |
| **405B dense (Llama-3.1-405B class)** | 128 GB unified (Mac Studio/Strix Halo) | `IQ2_XS`/`IQ2_M` or UD 2-bit dynamic | ~100-115 GB | low single digits, prefill-bound |
| **671B MoE (DeepSeek-V3-class)** | 128-192 GB | UD `Q2_K_XL`/`IQ2_XXS`, heavy CPU/RAM residency | ~170-220 GB | low single digits |
| **671B-1T MoE (Kimi K2-class)** | 512 GB (big EPYC box or 4× Strix Halo cluster) | UD 1-2 bit dynamic GGUF | ~310-350 GB | single digits, RAM-bandwidth-bound |
| **1T+ MoE, careful case** | 512 GB+ | UD-`Q4_K_XL` if budget allows (near-lossless per Unsloth's own PPL numbers) | ~580 GB | limited by RAM bandwidth, not compute |

The consistent shape across every row: **for dense models the limiter is "does it fit in fast memory at all," and the quant choice is a straightforward capacity/quality trade; for MoE models the limiter is CPU/RAM bandwidth for the offloaded experts**, so the quant choice interacts with `--n-cpu-moe`/`-ot` tensor placement (`07` §3.3) at least as much as with the nominal bit-width, a 4-bit MoE with a bad offload split can lose to a 3-bit MoE with a good one. KV-cache quantization (`-ctk`/`-ctv q8_0`, per §11's architecture-sensitivity caveat) is the other lever that determines whether long-context use of these models is possible at all on a given budget, independent of the weight quant.

---

## 17. Implications for a new engine

**Formats to support natively, in priority order:**
1. **GGUF** (all K-quant and I-quant types, plus native reading of `imatrix` files): it's the format the community's model zoo actually ships in, and skipping it means re-quantizing the zoo yourself.
2. **A trellis/VQ format**, either EXL3-compatible or GSQ-style plain-scalar-with-VQ-quality, because it is the surveyed method that stays coherent below 2.5 bpw and the accuracy gap to scalar quantization at 3-4 bpw is real, if narrowing (GSQ).
3. **Native microscaling FP** (MXFP4 at minimum, NVFP4 if targeting Blackwell): checkpoints increasingly *arrive* in these formats (gpt-oss, Kimi K2-Thinking, Nemotron 3), so "load and run the native format" beats "re-quantize on load" both for fidelity and for time-to-first-token.

**Kernel requirements this implies:**
- A **LOP3-style fused dequant** for every scalar INT format you support, dequant must be free (overlapped with the DRAM fetch), not a second pass; this is the single highest-value kernel investment per `05`'s roofline analysis (the "4-bit efficiency gap" is dequant-bound, not bandwidth-bound).
- A **block-structure-aware rotation path** if you support both a rotation method and a microscaling format, §12's MXFP4 finding means a generic global-Hadamard implementation will silently underperform on any block-scaled format unless the rotation respects block boundaries.
- **Per-format activation quantization that's conditional on batch size / compute-boundedness** (§8): don't quantize activations at batch 1 where it only adds overhead; do at batch/prefill where it's a real win. This is a scheduling decision, not a kernel decision alone, and no existing engine surfaced in this survey makes it dynamically.
- **A LUT/trellis decode path for CPU and for sub-3-bit GPU formats**, but with a runtime check (not a fixed assumption) of whether the ALU-heavy path actually beats the scalar path on the current hardware/block-size combination, the ik_llama.cpp `iq4_kt`-loses-to-`iq4_ks` and EXL3 Ampere caveats both say this is *not* a settled question per-format, it's per-(format, hardware) and should be measured, not assumed.

**What's open, worth building rather than borrowing:**
- **Latency-aware bit allocation.** Every method surveyed here. GGUF's imatrix, EXL2/EXL3's error search, GPTQ/AWQ's calibration, Unsloth's KLD search, minimizes *quality loss under a storage budget*. None of them puts *measured kernel latency* into the objective. A tensor that's cheap to dequantize at 3-bit but expensive at 2-bit-trellis should sometimes stay at 3-bit even if 2-bit-trellis has lower reconstruction error, if the trellis ALU cost isn't hidden by that tensor's memory-bandwidth headroom. This survey found no method that does this.
- **Placement-aware quantization.** This survey found no method that conditions bit allocation on *where* a tensor will execute. A CPU-offloaded MoE expert (RAM-bandwidth-bound, benefits from going lower-bit even at real ALU cost since RAM bandwidth is scarcer than compute) should plausibly get a different bit-width policy than a GPU-resident attention tensor (VRAM-bandwidth-bound but with cheap dequant ALU available), even within the same checkpoint. `--n-cpu-moe`/`-ot` placement and quantization are currently two completely separate decisions made by two different tools at two different times; unifying them is unexplored.
- **Per-checkpoint KV-cache sensitivity profiling as a first-class citizen.** The Gemma-vs-Qwen KLD gap (§11) means a responsible engine should measure, not assume, cache-quantization sensitivity per architecture (and ideally auto-select `-ctk`/`-ctv` accordingly), rather than shipping one global default the way most engines currently do.
- **Router-protection as a default, not an opt-in.** MoE routers are the most consistently quantization-fragile component across every independent finding in this survey (§13, §11). An engine's default MoE quantization policy should treat the router the way K-quants already treat `output.weight`/`token_embd`, protected by default, not something the user has to remember to exclude.
- **A shared cross-engine evaluation harness.** ExLlamaV3's `qbench` is the closest thing that exists and it's scoped to ExLlamaV3. A new engine that ships KLD/top-token-agreement/CoT-length evaluation as a first-class, format-agnostic tool would fill a real gap, right now, comparing a GGUF quant to an EXL3 quant to a native-MXFP4 checkpoint on equal footing requires building this yourself.

---

## 18. Sources

**PTQ methods**
- GPTQ. https://arxiv.org/abs/2210.17323
- GPTQv2. https://arxiv.org/abs/2504.02692
- The Geometry of LLM Quantization (GPTQ = Babai's algorithm). https://arxiv.org/abs/2507.18553
- AWQ. https://arxiv.org/abs/2306.00978
- SmoothQuant. https://arxiv.org/abs/2211.10438
- OmniQuant. https://arxiv.org/abs/2308.13137
- HQQ. https://mobiusml.github.io/hqq_blog/ · https://github.com/mobiusml/hqq
- SqueezeLLM. https://arxiv.org/abs/2306.07629
- SpQR. https://arxiv.org/abs/2306.03078
- QuIP. https://arxiv.org/abs/2307.13304
- QuIP#. https://arxiv.org/abs/2402.04396
- AQLM. https://arxiv.org/abs/2401.06118
- PV-Tuning. https://arxiv.org/abs/2405.14852
- VPTQ. https://arxiv.org/abs/2409.17066
- QTIP. https://arxiv.org/abs/2406.11235
- ParetoQ. https://arxiv.org/abs/2502.02631
- GSQ. https://arxiv.org/abs/2604.18556
- CCQ. https://arxiv.org/abs/2507.07145
- BiSCo-LLM. https://arxiv.org/abs/2607.08643
- BCJR-QAT. https://arxiv.org/abs/2605.10655
- SchurQuant (unverified, low confidence). https://arxiv.org/abs/2608.15567
- QUASAR (unverified, low confidence). https://arxiv.org/abs/2608.13966
- FlashQuant (unverified, low confidence). https://arxiv.org/abs/2608.15531
- ExactMoE (unverified, low confidence). https://arxiv.org/abs/2608.15383

**GGUF / llama.cpp / ik_llama.cpp**
- llama.cpp quantize tool README. https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- ggml-common.h (block struct definitions). https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
- ik_llama.cpp README. https://github.com/ikawrakow/ik_llama.cpp
- Unsloth Dynamic 2.0 GGUFs. https://unsloth.ai/blog/dynamic-v2 (also see docs.unsloth.ai)

**EXL2 / EXL3**
- ExLlamaV3. https://github.com/turboderp-org/exllamav3
- ExLlamaV2 (EXL2). https://github.com/turboderp/exllamav2

**MLX**
- MLX quantization API/modes (deepwiki). https://deepwiki.com/ml-explore/mlx/7.1-quantization-api-and-modes
- mlx-lm DWQ example model. https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit-DWQ

**bitsandbytes**
- Making LLMs even more accessible (LLM.int8 / NF4 background). https://huggingface.co/blog/hf-bitsandbytes-integration
- QLoRA (NF4, double quantization). https://arxiv.org/abs/2305.14314
- LLM.int8(). https://arxiv.org/abs/2208.07339

**FP8 / microscaling**
- Pretraining LLMs with NVFP4 (NVIDIA). https://arxiv.org/abs/2509.25149
- Pretraining with MXFP4 on Native FP4 Hardware. https://arxiv.org/abs/2605.09825
- Unveiling the Potential of Quantization with MXFP4. https://arxiv.org/abs/2603.08713
- Benchmarking PTQ under Microscaling Formats. https://arxiv.org/abs/2601.09555
- INT v.s. FP. https://arxiv.org/abs/2510.25602
- gpt-oss. https://arxiv.org/abs/2508.10925
- Block Rotation is All You Need for MXFP4. https://arxiv.org/abs/2511.04214
- TORQ. https://arxiv.org/abs/2605.19561

**Activation quantization / kernels**
- QServe. https://arxiv.org/abs/2405.04532
- LiquidGEMM. https://arxiv.org/abs/2509.01229
- MXSens. https://arxiv.org/abs/2607.17733
- Marlin (IST-DASLab). https://github.com/IST-DASLab/marlin · https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference
- Machete. https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel
- GemLite. https://github.com/mobiusml/gemlite
- T-MAC. https://arxiv.org/abs/2407.00088
- Vec-LUT. https://arxiv.org/abs/2512.06443
- bitnet.cpp. https://arxiv.org/abs/2502.11880

**QAT / low-bit-native models**
- BitNet b1.58. https://arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T. https://arxiv.org/abs/2504.12285
- BitNet v2. https://arxiv.org/abs/2504.18415
- BitNet Distillation. https://arxiv.org/abs/2510.13998
- Gemma 3 QAT (Google blog). https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-now-more-accessible/

**KV cache**
- KIVI. https://arxiv.org/abs/2402.02750
- KVQuant. https://arxiv.org/abs/2401.18079
- KV Cache Compression via Transform Coding (unverified, low confidence). https://arxiv.org/abs/2608.14191

**Rotation / Hadamard**
- QuaRot. https://arxiv.org/abs/2404.00456
- SpinQuant. https://arxiv.org/abs/2405.16406
- FlatQuant. https://arxiv.org/abs/2410.09426

**MoE quantization**
- MoEQuant. https://arxiv.org/abs/2505.03804
- AlphaQ. https://arxiv.org/abs/2606.04980
- GEMQ. https://arxiv.org/abs/2605.23078
- Dynamic Expert Quantization. https://arxiv.org/abs/2511.15015
- Tied Trit-Planes. https://arxiv.org/abs/2608.08910
- REAP. https://github.com/CerebrasResearch/reap

**Evaluation**
- Quantization Inflates Reasoning. https://arxiv.org/abs/2606.25519
- UniComp. https://arxiv.org/abs/2602.09130
- Can Compressed LLMs Truly Act?. https://arxiv.org/abs/2505.19433
- Quantization Hurts Reasoning?. https://arxiv.org/abs/2504.04823
- Give Me BF16 or Give Me Death?. https://arxiv.org/abs/2411.02355

**Cross-referenced internal reports**: `00-synthesis.md` §5, `01-engine-landscape.md` §6, `05-kernels-and-systems.md` §1.5/§2, `06-literature-2025-2026.md` §5, `07-consumer-hardware-practice.md` §3.1/§3.2/§4.
