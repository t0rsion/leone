# Speculative, Predictive and Parallel Decoding

*A survey for engine builders targeting consumer hardware — compiled 2026-08-18.*

**Method note.** This document cross-checks two independent sources: (A) primary-source verification done for this report — HuggingFace `config.json` files, engine source code on GitHub, arXiv abstracts, OpenAlex indexes; and (B) an extended technical dialogue with a second model (GPT-5.6-sol) covering the mathematics and systems analysis. Where the two disagreed, (A) wins and the disagreement is noted. Claims that neither source could verify are marked **[unverified]**.

---

## 0. Executive summary

Speculative decoding is not one technique. It is a *contract*: something proposes tokens, the target model scores them in one batched pass, and a verification rule decides how many to commit. Everything else — draft models, MTP heads, EAGLE, n-gram lookup, block diffusion — is a pluggable proposal backend behind that contract. Build the contract first.

The three facts that should drive a consumer engine's design:

1. **At batch 1 the target is memory-bandwidth-bound, so verifying γ tokens costs almost the same as generating 1** — for *dense* models. For MoE models this is false, and badly so: γ tokens activate the *union* of their experts. This single asymmetry determines nearly every design decision below.
2. **The 2026 state of the art moved away from EAGLE-style autoregressive drafters toward block-parallel / semi-autoregressive drafters** (DFlash → DFlare / Domino / DSpark), which produce a whole draft block in one forward pass. Accepted lengths went from ~4-5 tokens/round (EAGLE-3) to 9-11. All three major engines — vLLM, SGLang, llama.cpp — now ship these as first-class methods.
3. **Speculation frequently makes things slower.** A 2026 from-scratch study on consumer hardware found *three of five* configurations decelerated. The best case was 1.61×, not 6×. An engine that cannot detect and disable a losing configuration at runtime is worse than one with no speculation at all.

---

## 1. Foundations: the exact algorithm and why it is lossless

### 1.1 Modified rejection sampling

Two papers established the method independently: Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)), and Chen et al., *Accelerating LLM Decoding with Speculative Sampling* ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)).

Let `p` be the target's next-token distribution and `q` the draft's. Draw `x ~ q`, `u ~ U[0,1]`. Accept `x` iff

```
u ≤ min(1, p(x)/q(x))
```

On rejection, resample from the normalized positive residual:

```
r(y) = [p(y) - q(y)]₊ / Σ_z [p(z) - q(z)]₊        where [v]₊ = max(0, v)
```

### 1.2 The losslessness proof (three lines)

Mass reaching token `y` via the acceptance branch:

```
q(y) · min(1, p(y)/q(y)) = min(p(y), q(y))
```

Total rejection probability:

```
R = 1 - Σ_y min(p(y), q(y)) = Σ_y [p(y) - q(y)]₊
```

The residual branch therefore contributes exactly `R · r(y) = [p(y) - q(y)]₊`. Summing:

```
Pr(Y = y) = min(p(y), q(y)) + [p(y) - q(y)]₊ = p(y)   ∎
```

Applied conditionally at every accepted prefix, the whole sequence is distributed exactly as the target would produce it. **The proof holds for any normalized `p`** — which is why grammar masks, repetition penalties, and top-p filtering do not break it (§8.4), and why constrained MoE routing *does* (§6.3).

### 1.3 Acceptance rate is total-variation overlap

```
α = Σ_x q(x) · min(1, p(x)/q(x)) = Σ_x min(p(x), q(x)) = 1 - TV(p, q)
```

This is the single most useful identity in the field. Acceptance measures *distributional overlap*, not top-1 agreement. A draft that gets argmax right 90% of the time but is badly calibrated can have low α.

Special cases: at `T = 0` the target is a delta at `y*`, so a draft proposal is accepted iff it equals `y*` and α = q(y*); if both sides are greedy, verification degenerates to "accept the longest prefix where the argmaxes agree."

### 1.4 Expected committed tokens

With constant acceptance α and draft length γ, including the bonus token:

```
E[N] = Σ_{k=0}^{γ} α^k = (1 - α^{γ+1}) / (1 - α)
```

With position-dependent rates (the realistic model — acceptance decays with depth):

```
E[N] = 1 + Σ_{k=1}^{γ} Π_{i=1}^{k} α_i
```

### 1.5 Optimal draft length

Normalize a target decode step to cost 1 and let a draft step cost `c`. Under the idealized assumption that verifying γ positions costs one target step:

```
S(γ) = (1 - α^{γ+1}) / [(1 - α)(1 + cγ)]
```

Extending the block is worthwhile iff

```
α^{γ+1}(1 + cγ) > c · (1 - α^{γ+1})/(1 - α)
```

There is a closed form via the `W₋₁` branch of the Lambert W function, but **do not use it**. γ is small, measured costs are non-linear, and α decays with depth. Enumerate γ ∈ {1,2,4,8,16} against measured timings. The realistic objective an engine should optimize is

```
S(γ) = τ · t_T(1) / [ Σᵢ t_D,i + t_T(γ) + t_verify + t_sched ]
```

### 1.6 Tree generalization

SpecInfer ([arXiv:2305.09781](https://arxiv.org/abs/2305.09781)) verifies a whole token tree in one target pass. Exactness uses **recursive residual sampling**: start with residual `p₀ = p`; for candidate `xⱼ ~ qⱼ`, accept with `min(1, p_{j-1}(xⱼ)/qⱼ(xⱼ))`; on rejection update `pⱼ = norm([p_{j-1} - qⱼ]₊)` and try the next candidate. Since `min(p_{j-1}, qⱼ) + [p_{j-1} - qⱼ]₊ = p_{j-1}` at every level, the committed token has the correct conditional.

For a tree of depth D, `E[N] = Σ_{d=0}^{D} Pr(verification reaches depth d)`. Width raises each term but multiplies verified nodes, mask overhead and KV writes.

**Warning:** many published "top-k draft trees" use greedy verification or Medusa-style typical acceptance and are *not* exact samplers. Check for `[p-q]₊` in the code before calling an implementation lossless.

### 1.7 The bandwidth argument and where it breaks

A batch-1 decode step reads essentially all model weights to produce one token — arithmetic intensity ≈ 1. Verification converts GEMV into a small GEMM: the same weights, γ positions. So `t_T(γ) ≈ t_T(1)` until compute, attention, dequantization or capacity bites.

Roofline: `t_T(B,γ) ≈ max(weight_bytes/BW, FLOPs(B,γ)/C)`.

It stops helping when: ordinary decode already saturates tensor cores; the draft isn't cheap; KV-cache reads dominate and verification multiplies them; tree width wastes target evaluations; continuous batching keeps effective batch high; **or MoE verification fetches many extra experts.**

### 1.8 Metrics — define them in your results

- **Acceptance length τ** = committed tokens per target verification.
- **Draft acceptance rate** = accepted draft tokens / proposed draft tokens.
- **Block efficiency** = committed tokens / target positions evaluated (≈ τ/(γ+1) for a chain; divide by flattened node count for trees). This is the metric that catches wasted tree width.
- **Measured speedup** = baseline wall-clock / speculative wall-clock.

τ is not a speedup. A method with τ = 5 can be *slower* than baseline. Spec-Bench ([github.com/hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)) established this separation; SPEED-Bench ([arXiv:2604.09557](https://arxiv.org/abs/2604.09557), Feb 2026) extended it to production engines (vLLM, TensorRT-LLM) and throughput-oriented concurrency sweeps, and specifically quantifies how *synthetic inputs overestimate real-world throughput* and how optimal draft length is batch-size-dependent.

---

## 2. Quantitative model for a consumer GPU

Roofline arithmetic for a 32B dense model (64 layers, d=5120, 40 Q heads, 8 KV heads, head_dim=128), Q4_K_M ≈ 19 GB, on an RTX 4090 (1008 GB/s, ~165 TFLOP/s bf16) and 5090 (1792 GB/s, ~210 TFLOP/s).

KV bytes per token across all layers: `64 × 2 × 8 × 128 × 2 = 262 KB`. So 1.07 GB at 4k context, 8.59 GB at 32k.

Weight-only lower bound per pass: 19/1008 = **18.85 ms** (4090), 19/1792 = **10.60 ms** (5090).

| Context | γ | 4090, ideal KV reuse | 4090, KV re-read per query |
|---|---:|---:|---:|
| 4k | 1 | 19.9 ms | 19.9 ms |
| 4k | 8 | 19.9 ms | 27.4 ms |
| 4k | 16 | 19.9 ms | 35.9 ms |
| 32k | 1 | 27.4 ms | 27.4 ms |
| 32k | 8 | **27.4 ms** | **87.0 ms** |
| 32k | 16 | 27.4 ms | 155.2 ms |

**The compute crossover under ideal KV reuse is around γ ≈ 40-50 on a 4090.** So γ = 4 or 8 is essentially free *if the verification kernel is a prefill-style attention path with M = γ queries.* The 32k / no-reuse column is what happens if you call a single-query decode kernel γ times — a 3.2× penalty at γ=8 that erases the entire benefit. **This is the single most common implementation bug in the space**, and it is exactly what the consumer-hardware study (§9) observed on a quantized Metal backend.

### 2.1 The MoE expert-union calculation

For a 30B-A3B-style MoE (128 experts, top-8), assuming independent uniform routing, the expected number of distinct experts activated by γ tokens is

```
U(γ) = 128 · [1 - (15/16)^γ]
```

| γ | Distinct experts | Expert-read multiplier U/8 | Total weight-read multiplier* |
|---:|---:|---:|---:|
| 1 | 8.00 | 1.00× | 1.00× |
| 2 | 15.50 | 1.94× | 1.56× |
| 4 | 29.12 | 3.64× | 2.58× |
| 8 | 51.62 | 6.45× | 4.27× |
| 16 | 82.42 | 10.30× | 6.58× |

\* accounting for shared/dense weights being read once (≈1.2B shared, 28.8B routed).

Real routers are skewed and adjacent tokens correlate, which reduces the union — often substantially — but the shape of the curve is right. **Verification of a MoE target is not free.** Break-even τ for a MoE Q4 target at 4k context is ≈2.85 at γ=4 and ≈4.78 at γ=8 with a 0.5B draft, versus ≈1.10 and ≈1.20 for the dense target. This is why every 2026 MoE-speculation paper is about controlling the expert union.

### 2.2 Break-even τ (dense Q4 target, 4090, ideal KV reuse)

| Target | γ | 0.5B draft | EAGLE-style head | n-gram (free) |
|---|---:|---:|---:|---:|
| Dense Q4, 4k | 4 | 1.10 | 1.16 | 1.00 |
| Dense Q4, 4k | 8 | 1.20 | 1.32 | 1.00 |
| MoE Q4, 4k | 4 | 2.85 | 3.30 | 2.12 |
| MoE Q4, 4k | 8 | 4.78 | 5.66 | 3.31 |

For a dense target almost any nontrivial acceptance wins. For short-context MoE at γ=8 you need extraordinary acceptance.

---

## 3. Proposal backends: the taxonomy

### 3.1 Draft models

Pick: same tokenizer; same family/pretraining distribution; 1/10–1/30 the parameters; small enough to stay resident alongside target + KV. A 1B draft for a 70B target is a good ratio; a 7B draft for 70B usually is not, on one consumer GPU.

**DistillSpec** ([arXiv:2310.08461](https://arxiv.org/abs/2310.08461)) trains drafts for *agreement* rather than perplexity. Recipe: `L = λ_CE·CE(y,q) + λ_KD·T²·KL(p_T ‖ q_T)`, trained on target-generated (on-policy) prefixes at the deployment temperature, rolling the draft forward on its own tokens to expose depth-wise drift. Measure `1 - TV(pᵢ, qᵢ)` per depth, not perplexity.

**Cross-tokenizer**: emit text from the assistant, re-tokenize with the target tokenizer, align the byte span, verify. HuggingFace calls this *universal assisted generation*. Greedy cross-tokenizer assistance can stay lossless through target verification; exact *stochastic* rejection is much harder because `q` must be defined over target-token space. Pitfalls: normalization, byte fallback, leading-space conventions, one assistant token mapping to several target tokens.

### 3.2 Self-speculative / early exit

| Method | Idea | Systems problem |
|---|---|---|
| [Draft & Verify](https://arxiv.org/abs/2309.08168) | skip target layers when drafting | missing KV for skipped layers |
| [LayerSkip](https://arxiv.org/abs/2404.16710) | layer dropout + early-exit loss so intermediate layers can predict | needs a compatible checkpoint |
| [Kangaroo](https://arxiv.org/abs/2404.18911) | early-exit adapter + double early exiting | head latency, calibration |
| [SWIFT](https://arxiv.org/abs/2410.06916) | on-the-fly selection of a layer subset | irregular execution, cache layout |

Typical results 1.3–2.4× at batch 1. **The KV problem is the real cost**: if drafting exits at layer L_d, layers L_d+1…L produced no K/V for the proposed tokens, so verification must run them anyway. Prefix-layer early exit (reuse lower-layer KV, run only upper layers in verification) is the only clean design; non-contiguous layer skipping requires saved intermediates or recomputation. After verification, truncate *every* layer consistently — never let low-layer cache length diverge from upper-layer length.

### 3.3 Head-based / feature-based drafters

- **Medusa** ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774)) — K independent heads predicting t+1…t+K, tree attention, *typical acceptance* (`p(x) ≥ min(ε, δ·e^{-H(p)})`). ~2.2–2.8× but **lossy by default**.
- **Hydra** ([arXiv:2402.05109](https://arxiv.org/abs/2402.05109)) — sequentially dependent heads: models `q(x_{t+k} | h_t, x_{t+1:t+k-1})` instead of `q(x_{t+k} | h_t)`. Incremental gain over Medusa.
- **EAGLE** ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077)) — autoregression in *feature* space, predicting the next target hidden state from the previous feature plus the sampled token embedding; the key insight is that token sampling injects uncertainty a deterministic hidden-state predictor cannot ignore. ~2.7–3.5×.
- **EAGLE-2** ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858)) — context-aware *dynamic* draft trees; confidence-weighted expansion and pruning. 3.05–4.26×.
- **EAGLE-3** ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)) — "training-time test": fuses features from multiple target layers, trains on rolled-out draft trajectories, drops the feature-regression loss in favor of direct token prediction, and shows the drafter *scales* with training data. ~3–5×.
- **HASS** ([arXiv:2408.15766](https://arxiv.org/abs/2408.15766)) — harmonized self-distillation, aligning the drafter with its own rollout distribution (train/test mismatch fix).
- **GLIDE with a CaPE** ([arXiv:2402.02082](https://arxiv.org/abs/2402.02082)) — reuse target KV in the drafter + a proposal expansion head.
- **Clover** ([arXiv:2405.00263](https://arxiv.org/abs/2405.00263)) — regressive lightweight heads with sequential knowledge.
- **Sequoia** ([arXiv:2402.12374](https://arxiv.org/abs/2402.12374)) — hardware-aware optimal tree topology via dynamic programming.

**EAGLE-3 is no longer the frontier.** See §4.

### 3.4 Draft-free: n-gram, retrieval, suffix

These should be first-class backends — they cost almost nothing and can be spectacular.

| Method | Source | Wins on |
|---|---|---|
| Prompt lookup (PLD) / PLD+ | n-grams in the prompt | summarization, doc QA, extraction |
| [Lookahead decoding](https://arxiv.org/abs/2402.02057) | n-gram pool from Jacobi trajectories | repetitive/predictable generation |
| [REST](https://arxiv.org/abs/2311.08252) | external datastore retrieval | domain corpora |
| [Token Recycling](https://arxiv.org/abs/2408.08696) | graph of observed token transitions | long outputs |
| [SAM-Decoding](https://arxiv.org/abs/2411.10666) | suffix automaton over prompt + history | fast longest-suffix match |
| Arctic SuffixDecoding | reusable suffix continuations | agent loops, code edits |
| [Oilbird](https://arxiv.org/abs/2608.03839) (2026) | **re-keys the pool by the verifier's hidden state** | tool-calling traffic |

**Oilbird is the most interesting recent result here.** Diagnosing failures position-by-position across ten benchmarks, it finds that on dense tool-calling traffic *about half* of what the strongest exact-match drafter misses is already present in the pool but unreachable by exact lexical matching. Re-keying the same pool by the hidden state the verifier already computed at each committed token — free, no extra model — lifts accepted length 24–29% across three published drafters. **4.4× on API-Bank vs 3.9× for the best training-free baseline and 2.0× for EAGLE-3.** For an agentic workload this is the highest value-per-line-of-code in the entire survey.

---

## 4. The 2026 shift: block-parallel and semi-autoregressive drafters

This is the biggest change since EAGLE-3 and it is under-appreciated. The insight: an autoregressive drafter's cost grows with draft depth (γ sequential forwards), which caps γ. A *block diffusion* drafter emits the whole block in one forward pass, so drafting cost is decoupled from draft length.

**The lineage, all verified on arXiv:**

- **DFlash: Block Diffusion for Flash Speculative Decoding** ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036), Feb 2026). Lightweight block-diffusion draft model conditioned on context features extracted from the target. Reports **over 6× lossless acceleration, up to 2.5× higher speedup than EAGLE-3.**
- **DDTree** ([arXiv:2604.12989](https://arxiv.org/abs/2604.12989)) — builds a draft tree from DFlash's per-position marginals via best-first heap search under a node budget, verified with an ancestor-only attention mask.
- **DFlare** ([arXiv:2606.02091](https://arxiv.org/abs/2606.02091)) — fixes DFlash's conditioning bottleneck: each draft layer attends to its own learnable combination of a broad set of target layers. **5.52× on Qwen3-4B, 5.46× on Qwen3-8B, 3.91× on GPT-OSS-20B.** Code in Tencent's AngelSlim.
- **Domino** ([arXiv:2605.29707](https://arxiv.org/abs/2605.29707)) — *decouples causal modeling from autoregressive execution*: a parallel backbone produces preliminary block distributions, then a lightweight GRU-based "Domino head" refines them with prefix-dependent causal information. Up to 5.49× (Transformers) / 5.8× (SGLang).
- **DSpark** ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)) — semi-AR: parallel backbone + sequential Markov head + *confidence-scheduled verification length* tuned per request from estimated prefix-survival probability. Deployed in the DeepSeek-V4 serving system; reports **60–85% faster per-user generation vs the production MTP-1 baseline.**
- **DominoTree** ([arXiv:2607.08642](https://arxiv.org/abs/2607.08642)) — best-first tree scored by Domino's non-factorized path-dependent correction. **Up to 6.6× on Qwen3-4B, mean accepted length up to 10.7 tokens/round**, with a GPU-native CUDA-graph tree builder that is bit-identical to the Python reference.
- **JetSpec** ([arXiv:2606.18394](https://arxiv.org/abs/2606.18394)) — causal parallel draft head over fused frozen-target hidden states, so candidate-tree scores align with the target's autoregressive factorization. **Up to 9.64× on MATH-500, 4.58× on open-ended chat (H100).**
- **PCTree** ([arXiv:2608.02123](https://arxiv.org/abs/2608.02123)) — converts DSpark's linear block into a tree using the *pretrained* Markov head, no retraining. Qwen3-4B GSM8K at B=16: accepted length 9.41 → 11.16, speedup 6.14× → 6.60×.

### 4.1 The catch: entropy

**DBLAST** ([arXiv:2608.05448](https://arxiv.org/abs/2608.05448)) is the necessary counterweight. A factorized block proposal approximates `q(x_{1:K}|s) ≈ Π qᵢ(xᵢ|s)` while the truth is `Π p(xᵢ|s, x_{<i})`. At low entropy the missing dependencies don't matter — each position has one dominant continuation. At high entropy, early sampled choices reshape later conditionals, and **accepted draft length degrades as target sampling entropy rises.** DBLAST proposes a low-rank latent mixture over positions plus an acceptance-oriented training objective.

This means: **a causal correction head is not optional for a chat engine at T=0.7–1.0.** The DSpark/Domino Markov-or-GRU head is the minimum viable structure. It is also what makes exact rejection sampling *possible*, because an autoregressive correction exposes a tractable factorized proposal `q(x_{1:K}|s) = Π qᵢ(xᵢ|s, x_{<i})` — you can record every `qᵢ` and run the standard sampler. A pure diffusion procedure may not expose normalized per-position autoregressive `qᵢ` at all.

Cost of the correction head: a GRU of width d has ~6d² parameters, ~12d² FLOPs/step — negligible (31M FLOPs for 10 tokens at d=512). **The expensive part is the vocabulary projection**: at d=1024, V=150k, that's 307 MB in bf16, and reading it 10 times costs ~3.0 ms on a 4090 — a sixth of your entire target pass. Use a low-rank correction `ℓᵢ = zᵢ⁽⁰⁾ + A(Bhᵢ)` with r = 32–128 (≈19 MB at r=64), or restrict the correction to the backbone's top-M candidates.

### 4.2 A useful negative result

**PEFT-BD** ([arXiv:2607.12422](https://arxiv.org/abs/2607.12422)) tried a LoRA adapter on the same backbone as a block-diffusion drafter — no tokenizer mismatch, no second model, few trainable parameters. It obtained nontrivial accepted prefixes and **still produced no speedup**, because each step needed an adapter-enabled full-backbone draft pass plus an adapter-disabled full-backbone verify pass. *Parameter-efficient is not compute-efficient.* The drafter must be substantially cheaper to **execute**; long accepted prefixes cannot compensate.

---

## 5. Multi-Token Prediction

### 5.1 The training objective

Gloeckle et al., *Better & Faster LLMs via Multi-token Prediction* ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)):

```
L = -Σ_t Σ_{k=1}^{n} λ_k log p_k(x_{t+k} | x_{≤t})
```

Improves representation learning (especially on code) *and* provides native future-token heads for speculative inference.

### 5.2 DeepSeek's MTP module

From the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437). At prediction depth k:

```
uᵢᵏ = M_k · [ RMSNorm(hᵢᵏ⁻¹) ; RMSNorm(E(x_{i+k})) ]
hᵢᵏ = TransformerBlock_k(uᵢᵏ)
p_{i+k+1} = softmax(W_shared_head · hᵢᵏ)
```

Crucially it shares the base model's embedding and output head, and prediction depths are **sequential**, not independent as in Medusa — which is exactly what makes it usable as an autoregressive speculative drafter. DeepSeek reports ~85–90% acceptance for the additional token and ~1.8× speculative throughput.

**Caveat that bites people:** DeepSeek-V3 ships *one* trained next-token-prediction layer. Reusing it recursively for multiple speculative steps is possible but distribution drift grows fast. One MTP layer ≠ four trained future heads.

### 5.3 Which open models actually ship MTP — verified

Checked directly against `config.json` on HuggingFace (2026-08-18):

| Model | `num_nextn_predict_layers` | Verdict |
|---|---:|---|
| `zai-org/GLM-4.6` | **1** | MTP present |
| `deepseek-ai/DeepSeek-V3.2-Exp` | **1** | MTP present |
| `moonshotai/Kimi-K2-Instruct` | **0** | **No MTP** |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | *field absent* | see below |

Qwen3-Next's config carries no `nextn` field, yet vLLM registers a `qwen3_next_mtp` model type — so the MTP module ships as separate weights or under different tensor names rather than in the top-level config. **Never infer MTP support from a model name.** Inspect `num_nextn_predict_layers`, look for `mtp.*` / `nextn_predict_layers.*` tensors in the safetensors index, and confirm your quantization conversion retained them (GGUF/AWQ pipelines have historically dropped auxiliary modules silently).

### 5.4 What engines expose

**vLLM** (`vllm/config/speculative.py`, main branch) declares an `MTPModelTypes` literal covering: `deepseek_mtp`, `glm4_moe_mtp`, `glm4_moe_lite_mtp`, `glm_ocr_mtp`, `qwen3_next_mtp`, `qwen3_5_mtp`, `minimax_m3_mtp`, `kimi_k3_mtp`, `longcat_flash_mtp`, `ernie_mtp`, `nemotron_h_mtp`, `exaone_moe_mtp`, `exaone4_5_mtp`, `step3p5_mtp`, `gemma4_mtp`, `hy_v3_mtp`, `mimo_mtp`, `mimo_v2_mtp`, `pangu_ultra_moe_mtp`, `dots3_note_mtp`, `bailing_hybrid_mtp`, `bailing_hybrid_v3_mtp`, `inkling_mtp`. Note this is a *loader registry*: the presence of an enum entry proves a code path exists, not that public weights exist or that the path is fast.

Typical invocation:
```bash
vllm serve MODEL --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
vllm serve MODEL --speculative-config '{"method":"ngram","num_speculative_tokens":5,
                                        "prompt_lookup_min":2,"prompt_lookup_max":5}'
```

---

## 6. Systems interplay

### 6.1 Batching — when speculation stops paying

Speculation monetizes *unused arithmetic intensity*. Continuous batching consumes that same headroom. Rough regimes: batch 1–4 is the sweet spot; batch 8–16 is often marginal for 7B-class dense targets; large models stay bandwidth-bound longer; long contexts become KV-bound earlier. The scheduler rule: shorten or disable when

```
t_T(B, γ) / τ ≥ t_T(B, 1)
```

vLLM now encodes this structurally as `num_speculative_tokens_per_batch_size: list[(range_start, range_end, num_spec_tokens)]` — an explicit batch-size → draft-length schedule. Adaptive-length research: SpecDec++, AdaEAGLE ([arXiv:2412.18910](https://arxiv.org/abs/2412.18910)), DISCO, and **D-cut** ([arXiv:2607.14647](https://arxiv.org/abs/2607.14647)), which prunes verification depth *jointly across the batch* using a runtime cost model — improving average speedup from 1.26× to 1.65× under high concurrency and *restoring* acceleration in dense configurations where long-draft baselines were slower than plain AR decoding.

### 6.2 Quantized targets

Speculation stays exact **with respect to the deployed quantized target's distribution**, provided verification uses its actual logits. Effects cut both ways: smaller weights → more bandwidth headroom; dequantization → more compute per verified token; changed logits → worse agreement with a draft distilled from the FP16 target; an INT4 draft is cheaper but loses acceptance. Do the accept/reject arithmetic in FP32 and in log space (§8.5).

### 6.3 MoE and offload — the consumer-hardware crux

Consider Qwen3-235B-A22B at Q4 (~120 GB) on a 24 GB 4090 + 128 GB DDR5 (~80 GB/s) + PCIe 4.0 x16 (~25 GB/s effective), with attention and shared weights on GPU and experts in host RAM. Decomposing 235B total / 22B active with 128 experts top-8 gives ≈7.8B shared, 227B routed → `W_shared ≈ 3.98 GB`, `W_experts ≈ 116 GB`, and **7.25 GB of expert weights selected per token**.

```
PCIe:  7.25 / 25 = 290 ms/token   ← dominates
DRAM:  7.25 / 80 =  91 ms/token
                    → baseline ≈ 2.5–3.4 tok/s
```

Now speculate with γ=8. Under independent uniform routing U(8)=51.6 experts → 46.75 GB → **1.87 s per verification**, a 6.45× traffic multiplier. Even the theoretical max of 9 committed tokens gives only 9/6.45 = 1.40×. At α=0.8, τ=4.33 → **0.67× — a slowdown.** At α=0.9, τ=6.13 → 0.95×, still a loss.

| γ | τ (α=0.8) | Expert multiplier | Ideal transfer speedup |
|---:|---:|---:|---:|
| 1 | 1.80 | 1.00× | **1.80×** |
| 2 | 2.44 | 1.94× | 1.26× |
| 4 | 3.36 | 3.64× | 0.92× |
| 8 | 4.33 | 6.45× | 0.67× |

**γ=1 wins**, because the bonus token gives you a second token for one expert-streaming pass. This is a genuinely counterintuitive result and it is the most actionable number in this document for anyone doing MoE offload. Real routing is correlated and skewed, which improves matters — at an effective union of 24 experts, γ=8 reaches 1.44× at α=0.8 — but you must *measure* the union, not assume it.

**The 2026 answer is to control the union directly.** [AcceptMoE](https://arxiv.org/abs/2608.02989) is a verifier-side expert selector combining target-router scores with offline-estimated *commitment probabilities* (how likely each speculative position is to survive verification), self-sizing the eligible expert set per block, and — under offloading — **conditioning eligibility on cache residency instead of prefetching predicted routes.** Measured in SGLang at batch 1: **1.29× vs EAGLE-3 with experts in VRAM, 2.06× under physical expert offloading, ~7× less host-to-device traffic, mean accuracy −0.27 pp** across 12 model-task pairs.

[DraftExpert](https://arxiv.org/abs/2607.24434) attacks the same problem from the draft side: one lightweight accelerator-resident draft expert per layer, self-distilled from the frozen target with residual/logit/router-agreement signals, plus confidence-expansion truncation and target-expert prefetching. **1.45× average on DeepSeek-V2-Lite and Moonlight-16B-A3B across CPU-GPU and Flash-NPU offload; 84–87% draft acceptance; 86–88% prefetch hit rate.**

**Does constrained routing break losslessness?** Yes, and the distinction matters. Three cases:
1. *Residency affects only scheduling* — every originally-routed expert is eventually executed. Logits unchanged, speculation exact. 
2. *Eligibility is a causal deterministic function of the committed prefix* — this defines a modified model `p̃(x_t | x_{<t}, c_t)`, and rejection sampling can be exact **for that model**, not for the original checkpoint.
3. *Eligibility depends on router scores from all speculative positions* (the AcceptMoE-style case) — then the distribution used at position i depends on future draft candidates, so it is no longer a causal conditional `p̃(xᵢ | x_{<i})` at all. The output law is well-defined but depends on the drafter, on γ, and on tree layout. **Call this "approximate cache-conditioned inference", not "exact sampling from a modified target."** Change the speculative backend and outputs change at identical sampling settings.

For a consumer engine: expose `expert_policy = exact | residency_biased | transfer_budgeted`, default to exact for evaluation, and never enable restriction silently. Track excluded router mass per layer/block and force exact routing when it exceeds a threshold.

**CPU-compute alternative** (llama.cpp `-ot` / KTransformers: compute the expert FFN on the CPU with AMX/AVX-512 rather than moving weights): expert weights never cross PCIe. DRAM roofline is 7.25/80 = 91 ms → ~11 tok/s, realistically 6–10. Here speculation buys you *shared* weight reads and better GEMM shape, but it does **not** avoid γ× arithmetic (8 × 28.4 GFLOP = 227 GFLOP for γ=8 ≈ 114–227 ms at 1–2 effective TFLOP/s). γ=1–4 with batched per-expert GEMM is the useful range.

### 6.4 Long context

- **TriForce** ([arXiv:2404.11912](https://arxiv.org/abs/2404.11912)) — hierarchical speculation with a sparse-KV draft, up to ~2.3×.
- **MagicDec** ([arXiv:2408.11049](https://arxiv.org/abs/2408.11049)) — exploits the fact that long-context decode becomes *KV-bandwidth*-bound, so the classic latency/throughput tradeoff inverts.
- **Windowed-MTP** — "Removing the Full-Context Draft-KV Tax at Million-Token Context" (July 2026). **[title from OpenAlex index; abstract not retrieved, ID unverified]**

At very long context, reading the KV cache exceeds reading model weights, and a full-context draft stops being cheap. Sparse-KV drafts retrieve only salient blocks while the full target verifies against complete context.

---

## 7. Diffusion LMs as an alternative paradigm

A masked discrete diffusion LM iteratively denoises `x⁽ᴷ⁾ → … → x⁽⁰⁾`, updating many positions per iteration. Key models: **LLaDA** ([arXiv:2502.09992](https://arxiv.org/abs/2502.09992)), **Block Diffusion / BD3-LM** ([arXiv:2503.09573](https://arxiv.org/abs/2503.09573)), Dream 7B, Mercury (Inception Labs), Gemini Diffusion. As of Aug 2026 the field remains extremely active — LLaDA MoE v2 **[indexed Aug 2026; arXiv ID unverified]**, DiffusionGemma, Nemotron-Labs-Diffusion (a tri-mode model unifying AR, diffusion and self-speculation decoding), and serving systems like DiLaServe and Sangam ("serving diffusion LLMs with the AR stack") all appeared in 2026.

**Why KV caching breaks**: in a bidirectional diffusion block, changing token `x_j` changes the hidden state and K/V of every position attending to it, so cached entries go stale and each denoising iteration naïvely needs a full-sequence pass. Workarounds: block diffusion (freeze a finished block and cache it as an AR prefix), semi-autoregressive masks, confidence freezing, dLLM-Cache, and **Fast-dLLM** ([arXiv:2505.22618](https://arxiv.org/abs/2505.22618)). Several of these are lossy — a "stable" representation may have changed under later denoising.

**The honest verdict for a consumer engine:** the *biggest practical impact of diffusion LMs in 2026 is not as a replacement decoder — it is as a drafter.* DFlash/DFlare/DSpark take the parallel-generation property of block diffusion and put it behind an exact AR verifier, which sidesteps every quality, KV, and quantization problem while keeping the speed. Full diffusion-LM inference as an alternative model family remains a research bet: fewer strong open checkpoints, weaker quantization support, cache invalidation, variable completion lengths, less mature batching. Build the drafter, not the decoder.

---

## 8. Implementation guide for a new engine

### 8.1 The proposal interface

```
Proposal {
    token_ids[N]
    log_q[N]            // required for exact stochastic rejection; may be absent for greedy-only
    parent[N]           // -1 for a linear chain
    depth[N]
    position_id[N]      // = prefix_len + depth   ← NOT flattened index
    source_kind
    sampling_guarantee  // exact_target | greedy_exact | typical_lossy | approximate_residual
}
```

Backends register into an ordered chain and may decline cheaply. This is precisely the pattern llama.cpp now uses: a per-sequence `drafting` flag falls through implementations in order, and is cleared after the first implementation that produces a draft. Both vLLM and SGLang have converged on plugin registries too (SGLang has `spec_registry` with `register_algorithm` for custom algorithms).

### 8.2 KV rollback — linear case

Track two lengths: `committed_len` and `physical_written_len`. Write target K/V for all speculative positions during verification. If `a` draft tokens are accepted plus one replacement/bonus:

```
committed_len += a + 1
physical_written_len = committed_len
```

Rollback is a pointer update; bytes past the logical length may stay dirty. Paged attention complications: decrement refcounts only for pages wholly beyond the new end, preserve the partially-used terminal block, don't return pages while in-flight kernels reference them, and update block tables used by *captured graphs*.

**RoPE does not make rollback expensive** — keys are stored already-rotated for fixed positions. The danger is reusing a discarded slot with the wrong position or a stale block-table entry.

### 8.3 Trees, masks, and the compaction problem

Flatten depth-first. Node i attends to: all committed prefix positions, itself, and *precisely its ancestors*. Siblings must not see each other even when one precedes the other in flattened order.

```
M[i][j] = 0     if j ∈ Ancestors(i) ∪ {i}
        = -∞    otherwise
```

**Position IDs must be `prefix_len + depth`, never the flattened index.** Two siblings at the same depth are alternative values for the *same* sequence position and need the same RoPE phase. Get this wrong and: later siblings get wrong rotations; a copied accepted K/V entry is permanently rotated for the wrong position; the target logits used for acceptance are no longer the true target conditional — **which breaks the losslessness proof, not merely quality.**

Cost comparison for prefix=4096, N=64 tree nodes, 32 heads, head_dim=128, 64 layers:

| Approach | Prefix attention cost | Verdict |
|---|---|---|
| Dense logical mask + custom fused kernel | 275 GFLOP total ≈ 1.67 ms compute; 4.3 GB KV read ≈ 4.3 ms | mask itself is only ~33 KB as a bitmask; the problem is irregular tiling, not arithmetic |
| FlashAttention varlen with **full** prefix duplication per root-to-node path | 64 × 4.295 GB = **275 GB** ≈ 273 ms | unusable |
| varlen with a prefix-sharing/cascade kernel, duplicating only speculative ancestors | ~4.3 GB + a few hundred MB | viable |
| Chain-only (no tree) | standard causal kernel; extra triangular work = 2016 pairs/layer ≈ 33 MFLOP, <1% of prefix attention | **best engineering tradeoff for a first engine** |

**Commit/compaction.** After verification, the accepted path (say nodes 0 → 5 → 17 → 42) occupies physically unrelated scratch slots but must appear as logical positions L, L+1, L+2, L+3.

- With **token-granular indirection** (`logical_position → arbitrary physical slot`, SGLang's request-to-token map + token-to-KV pool style), commit is pure renumbering: `request_token_map[L+j] = scratch_slot[path[j]]`. No data moves.
- With **classic PagedAttention block-level indirection**, logical offset `i mod B` must map to the same offset inside the physical page, so arbitrary tree offsets cannot be expressed without block size 1, an extra token-level map, or a physical gather.

The gather is cheap enough that it is usually the right answer: for a GQA model at 256 KB KV/token, gathering 8 accepted tokens copies 2 MiB ≈ **2 µs of raw HBM traffic**, maybe tens of µs with launch overhead. Far simpler than making every attention kernel support arbitrary token-level indirection. Publish the destination block table only after copies complete; free rejected scratch slots after.

SGLang's `ragged_verify.py` / `RaggedVerifyLayout` suggests the current best direction: **variable draft and accepted lengths per request with cheap gather**, which at batch 1–4 is more valuable than arbitrary dynamic trees.

### 8.4 Sampling correctness — the edge cases you will hit

**Different filters on target and draft are fine.** Exactness requires only that `qᵢ` is the *actual* distribution the candidate was drawn from and `pᵢ` is the *actual* target distribution, both fully normalized after all transformations. Supports need not match; missing target mass is emitted by the residual. What is *not* exact: comparing unfiltered logits while sampling from filtered distributions, or clamping the ratio.

**Penalties (repetition/presence/frequency) do not break exactness** — they just make p and q history-dependent. But the target distribution at position i must apply penalties using the committed history *plus candidates 0…i-1*, because the target logits at i are conditioned on exactly that speculative prefix. In a tree, every node needs its own penalty state or a parent pointer with a delta representation.

**Grammar / JSON schema constraints do not break exactness either.** With allowed set `A(gᵢ)`:

```
p'ᵢ(x) = pᵢ(x)·1[x ∈ A(gᵢ)] / Σ_y pᵢ(y)·1[y ∈ A(gᵢ)]
```

Use `p'ᵢ` in acceptance and correction — the proof applies to any normalized target distribution. The draft *may* skip the grammar mask and stay exact (invalid proposals have `p'ᵢ(x)=0`, get rejected, residual emits a valid token) but acceptance will be terrible. Advance a speculative grammar state per proposed token; store one DFA state per tree node. A draft with a different tokenizer needs its own grammar transition logic — copying the target's token mask across vocabularies is invalid.

**Four things must roll back to the same accepted prefix**: KV logical length + block table, repetition-count state, grammar/DFA state, and RNG state.

**EOS mid-block**: if the target accepts a drafted EOS, commit and terminate — ignore later positions, no bonus. A drafted EOS after a rejection point is discarded (it was conditioned on an invalid prefix). If a grammar or min-length rule forbids EOS, `p(EOS)=0` and a drafted EOS is rejected with probability 1. The drafter should stop after EOS and pad the verification shape if a fixed graph requires it.

### 8.5 Numerics

Store draft logits in bf16 but do softmax and sampling in FP32, and retain FP32 `log q(xᵢ)`. **Storing only `log q(xᵢ)` is insufficient** — the residual needs every `qᵢ(v)`. For V=150k, γ=8, keeping bf16 draft logits costs 2.4 MB. Cheap.

Accept in log space to avoid overflow:
```
log u ≤ min(0, log p(x) - log q(x))
```
Build the residual stably:
```
log_residual = log_p + log1p(-exp(log_q - log_p))     // for positions where log_p > log_q
```
`expm1`/`log1p` matter when p ≈ q. Reduce in FP32, clamp tiny negative roundoff to zero, renormalize, sample by FP32 prefix sum or Gumbel-max. The residual sum `Z = Σ[p−q]₊` should equal the rejection probability; if Z rounds to zero while the acceptance branch says "reject", your implementation is inconsistent — recompute at higher precision rather than silently falling back to sampling from p (that is a small bias, not exactness).

### 8.6 The bonus token — quantify what you lose by dropping it

With α=0.8, γ=4: `P(all accepted) = 0.8⁴ = 40.96%` of steps get the bonus.

```
τ_with_bonus    = 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616
τ_without_bonus = 3.3616 - 0.4096                 = 2.9520
loss = 12.18%
```

Engines drop it because logits get aligned only to verification candidates, the sampler interface expects one output per input slot, or graph-captured shapes omit the final distribution. The bonus token's KV is simply not written yet — that's normal, it becomes an input on the next target call like any sampled token. **Reserve the output slot from day one.**

### 8.7 Lossy fast paths — label them

Offer them, never under the label "exact": greedy longest-prefix matching; typical acceptance; accept-if-target-rank ≤ k; accept-if `p(x) > ε`; approximate residual over target top-k. vLLM's own config now distinguishes `RejectionSampleMethod = standard | synthetic | block` and `DraftSampleMethod = greedy | probabilistic` — **an enum name is not a proof**; verify what each mode mathematically guarantees before advertising losslessness.

Two 2026 papers are worth reading here. [*Revisiting Lossy Verification*](https://arxiv.org/abs/2607.26627) shows many seemingly distinct lossy schemes collapse into two families — truncation-based and collaborative — and that truncation-based methods can degrade *below the true truncation-sampling baseline* through distributional distortion; for collaborative verification, controlling the overshoot of draft probability relative to target probability is the key principle. [*Approximate Speculative Decoding*](https://arxiv.org/abs/2608.03447) takes the opposite tack, replacing binary first-mismatch truncation with budgeted longest-prefix selection under a local logit-regret gate plus a persistent request-level regret budget, reducing exactly to standard greedy verification at budget zero: +3.05–15.26% throughput, 7.78% average across seven Qwen3-14B tasks.

### 8.8 CUDA graphs

Variable γ, tree width and batch conflict with fixed shapes. Options: capture buckets for (B, γ); capture a small set of tree *templates*; pad and mask unused slots; separate graphs for draft and verify; a max-shape graph with device-side valid lengths; eager execution for rare shapes. **Track graph hit rate** — a dynamic policy that raises τ but falls off captured graphs can lose net throughput. For a consumer engine, fixed tree templates usually beat arbitrary dynamic trees, because dynamic trees can erase EAGLE-2's theoretical advantage through graph misses and mask overhead.

---

## 9. Reality check: what actually happens on consumer hardware

The most important empirical paper for this audience is **"Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware"** ([arXiv:2607.17283](https://arxiv.org/abs/2607.17283), July 2026). A from-scratch device-agnostic (CUDA/MPS/CPU) implementation, five draft/target configurations on an Apple-silicon laptop, with distribution equivalence verified at three levels (χ²=162.5, dof=200, p=0.976 over ~9,200 tokens; exact greedy-sequence agreement).

Results:
- **Best configuration: 1.61× wall-clock at K=6.**
- Acceptance declined from 69.7% at K=1 to 37.8% at the optimum.
- **Three of five configurations were slower than baseline** — either because the draft failed to out-speed a small target, or because *the quantized Metal backend executed "parallel" verification serially*.

That last failure mode is the one to design against. Set `ρ_M = t_T(M)/t_T(1)`. If ρ₈ ≈ 1–2 you are in great shape; ρ₈ ≈ 3–4 means you need strong acceptance; **ρ₈ ≈ 8 means your backend is effectively serial and speculation cannot help at all.**

Against this, the paper-reported ranges:

| Method | Reported | Caveat |
|---|---:|---|
| Leviathan / Chen | 2–3× | model-pair specific |
| Draft & Verify | ~2× | layer selection |
| LayerSkip | 1.3–1.9× | needs trained checkpoint |
| Medusa | 2.2–2.8× | typically lossy acceptance |
| EAGLE-1 / -2 / -3 | 2.7–3.5× / 3.05–4.26× / 3–5× | batch-1 paper setups |
| REST | 1.6–2.4× | datastore-dependent |
| Lookahead | 1.3–1.8× | greedy/repetitive |
| DeepSeek MTP | ~1.8×, 85–90% 2nd-token acceptance | deployment-specific |
| DFlash / DFlare | >6× / 5.5× | Qwen3-4B/8B class, H100-era kernels |
| DominoTree / JetSpec | 6.6× / 9.64× (MATH-500) | low-entropy structured tasks |
| Oilbird (training-free) | 4.4× on API-Bank | tool-calling traffic specifically |

**A defensible consumer expectation is 1.3–2.5× end-to-end**, with genuine outliers on repetitive/agentic workloads (prompt-lookup and Oilbird-style retrieval) and on low-entropy math/code with a good block drafter. Treat anything above 4× as requiring inspection of the baseline, bonus-token handling, sampling mode and target kernel utilization. JetSpec's 9.64× is MATH-500; PCTree's 6.60× is Qwen3-**4B** at batch **16**. Neither is your workload.

---

## 10. Engine status, verified against source (Aug 2026)

### llama.cpp
`common/common.h` defines a full framework:
```c
enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,   // standalone draft model
    COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,      // multi-token prediction
    COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
    COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,   // DFlash + Markov head
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
};
```
This substantially changes the picture from 2025, when llama.cpp had draft-model speculation only. **MTP, EAGLE3, and block-diffusion drafters are all supported**, types can be **chained** (a per-sequence `drafting` flag falls through implementations until one produces a draft), and the type can be **inferred from GGUF metadata / HF sidecar files** in the draft repo (`common_speculative_types_from_gguf`).

Flags were renamed: `--spec-type`, `--spec-draft-model`/`-md`, `--spec-draft-n-max`/`--spec-draft-n-min`, `--spec-draft-p-min`, `--spec-draft-p-split`, `--spec-draft-hf`/`-hfd`, plus n-gram tuning (`--spec-ngram-map-k-size-n`, `--spec-ngram-mod-n-match`, …) and — notably for consumer offload — `--spec-draft-cpu-moe` / `-otd` to place the *draft's* expert tensors. The legacy `--draft-max`/`--draft-min` are removed with an explicit migration error. Defaults: `n_max=3`, `n_min=0`, `p_split=0.1`, `p_min=0.0`, `backend_sampling=true`. There is per-sequence state get/set and `common_speculative_print_stats`.

### vLLM (V1)
`SpeculativeMethod = ngram | medusa | mlp_speculator | draft_model | suffix | custom_class | EagleModelTypes | ngram_gpu | dspark`, where `EagleModelTypes = eagle | eagle3 | extract_hidden_states | MTPModelTypes | dflash`. Plus `RejectionSampleMethod = standard | synthetic | block`, `DraftSampleMethod = greedy | probabilistic`, and the batch-size draft-length schedule noted in §6.1. Historical gotchas to check: "no bonus token" paths, restricted architectures for EAGLE-3, pipeline-parallel incompatibility, and throughput regressions at high batch.

### SGLang
`SpeculativeAlgorithm = DFLASH | DSPARK | EAGLE | EAGLE3 | FROZEN_KV_MTP | STANDALONE | NGRAM | NONE`, plus a `spec_registry` plugin system (`register_algorithm`, `CustomSpecAlgo`, `WorkerFactory`, `ServerArgsValidator`) and `RaggedVerifyLayout`. SGLang has been the most aggressive tree/EAGLE implementation and its token-granular KV pool is what makes cheap tree commit possible. Knobs: `--speculative-algorithm`, `--speculative-draft-model-path`, `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens`. Reported serving gains typically 1.5–2.5×, below paper microbenchmarks — expected, since scheduling, graph padding and network overhead dilute model-only speedups.

### Others
TensorRT-LLM: draft-target, Medusa, ReDrafter, lookahead, EAGLE/EAGLE-3; strong fused kernels but build-time-fixed shapes and awkward consumer quantization support. HuggingFace Transformers: `assistant_model=`, `prompt_lookup_num_tokens=`, universal assisted generation — a good correctness reference, not a throughput target. ExLlamaV2 had draft-model speculation; ExLlamaV3 / mlx-lm / Ollama status **[unverified]** — check for `[p-q]₊` in the source before believing any of them is an exact stochastic sampler rather than greedy prefix matching.

---

## 11. Recommended build order

**Skip initially:** training your own EAGLE-3 head; arbitrary dynamic tree attention; independent-position block diffusion with no causal coupling; a large standalone 0.5–3B draft as the default path; a fixed global γ; silent approximate expert restriction; full diffusion-LM inference as a model family; supporting all 23 vLLM MTP enum entries before confirming public weights exist.

1. **Instrumented baseline with genuinely batched verification.** M>1 target forward using a *prefill-style* attention path. Timing decomposition, H2D byte counters, expert-union counters, graph hit tracking, and a regression test that proves verification is not silently serial. Build this before any drafter. The consumer-hardware paper is the argument.
2. **Exact linear speculative sampler.** Probabilistic standard rejection + greedy verification + bonus token + FP32/log-space acceptance + residual sampling + EOS/grammar/penalty correctness + paged-KV rollback. Keep the sampler completely independent of the proposal algorithm.
3. **Proposal registry with ordered fallthrough** (llama.cpp's chaining pattern): suffix → hidden-state retrieval → native MTP → block/semi-AR → external draft model → none. Backends decline cheaply.
4. **Free retrieval backends before any learned drafter.** n-gram hash, suffix automaton, generated-history cache, prompt lookup — then Oilbird-style hidden-state re-keying, since the verifier's last committed hidden state is already sitting in memory and is a much better semantic key than a token suffix.
5. **Generic auxiliary-head loader** rather than hard-coded "DeepSeek MTP": separate weight manifest, tensor-name/model-type registry, shared embedding/output-head references, quantization retention checks.
6. **Startup + online autotuner.** This belongs *before* EAGLE, trees, or diffusion drafting.
7. **Semi-autoregressive block drafter** (DSpark/Domino-shaped): parallel block backbone + compact causal correction (low-rank, r=32–128) exposing probabilistic `qᵢ`, confidence-scheduled length, linear/ragged verification, trained for both greedy and T=0.7–1.0 rollouts. A length-10 chain with high acceptance captures most of the 2026 benefit without any tree.
8. **Offload-aware execution**: grouped expert GEMMs across verification positions, async expert DMA, hot-expert cache, union-aware γ, exact *and* constrained routing modes with excluded-mass telemetry.
9. **Ragged verification** (variable draft/accepted length per request, token-level indirection or cheap gather, graph buckets with ragged valid ranges) — more valuable at batch 1–4 than dynamic trees.
10. **Fixed-template trees**, and only then arbitrary layouts.

### 11.1 A ~10-second startup autotuner

**Phase 1 (1.5 s) — capability and serial-verification detection.** Allocate synthetic KV at 4k (and one long bucket if configured). For M ∈ {1,2,4,8}: warm once, time 2–3 target forwards with CUDA events, record `t_target[M]`, `t_attention[M]`, `t_moe[M]`, `H2D_bytes[M]`, `expert_union[M]`, `graph_hit[M]`. Compute ρ_M.

**Phase 2 (1.5 s) — proposal backend timing.** For each available backend measure `t_propose[γ]` at γ ∈ {2,4,8}, VRAM cost, launch count, graph availability, whether full `q` is available, and which sampling guarantee it supports. Prune immediately if `t_D(γ) ≥ (γ+1)·t_T(1)`.

**Phase 3 (4.5 s) — acceptance calibration.** Two bundled prompts (one prose/chat, one code/repetition) or the first real request. Do **not** estimate α from realized Bernoulli events when p and q are available — compute `aᵢ = min(1, pᵢ(xᵢ)/qᵢ(xᵢ))` and use the conditional expectation directly:

```
τ_block = 1 + a₁ + a₁a₂ + … + Π aᵢ
```

**Decision rule.** With `C = t_D(γ) + t_T(γ) + t_sampling + t_KV + t_sched`, take a conservative lower bound `τ_L = max(1, τ̄ - 1.28·s_τ/√n)` and shrink toward 1 (`τ_safe = 1 + η(τ_L - 1)`, η ≈ 0.6–0.8) since n is tiny. Enable speculation only if

```
S_L = t_T(1)·τ_safe / C^p90  >  1.10
```

then pick `argmax_{b,γ} S_L` subject to VRAM, required sampling guarantee, and graph availability.

**Runtime safeguard.** Maintain EWMA statistics keyed by (batch bucket, context bucket, entropy bucket, backend, γ, structured-output on/off, expert policy). Disable or shorten when `[t_D + t_T(γ) + overhead] / τ̂ > 0.97·t_T(1)` for several consecutive blocks — the 3% margin prevents oscillation. Explore one alternative configuration every 32–128 blocks, since entropy and repetition change *within* a single answer.

**Fast special cases**: if suffix lookup finds a long exact continuation, use it before invoking any learned drafter; if the grammar allows exactly one token, draft it directly; if predicted expert union exceeds break-even, shorten γ; if ρ_M scales linearly with M, set γ=1 or disable.

### 11.2 Measurement discipline

Report: target model + exact quantization; drafter version + quantization; GPU/CPU/RAM bandwidth/PCIe generation; batch and scheduler policy; prompt/output/context lengths; temperature, top-p, top-k; γ, tree width, total verified nodes; per-depth acceptance αᵢ; τ; block efficiency; target/draft/verify/sampler time split; graph hit rate; tokens/s and inter-token-latency percentiles; peak VRAM and host-transfer bytes.

Rules: warm all kernels and graphs; compare against the *same* target kernels and scheduler (never an unoptimized baseline against a fused speculative path); use identical prompt/output-length distributions; **do not expect seed-identical text** — rejection sampling consumes RNG differently, so test distributions statistically; separate TTFT from decode; include low-acceptance cases, because tail behavior dominates agent workloads; measure PCIe bytes for offloaded inference; and always publish both τ *and* wall-clock.

---

## Sources

**Foundations**
- Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding* — https://arxiv.org/abs/2211.17192
- Chen et al. — *Accelerating LLM Decoding with Speculative Sampling* — https://arxiv.org/abs/2302.01318
- *SpecInfer: Tree-based Speculative Inference and Verification* — https://arxiv.org/abs/2305.09781
- *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding* — https://arxiv.org/abs/2402.12374
- Spec-Bench — https://github.com/hemingkx/Spec-Bench
- *SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding* — https://arxiv.org/abs/2604.09557

**Draft models and self-speculation**
- *DistillSpec* — https://arxiv.org/abs/2310.08461
- *Draft & Verify* — https://arxiv.org/abs/2309.08168
- *LayerSkip* — https://arxiv.org/abs/2404.16710
- *Kangaroo: Double Early Exiting* — https://arxiv.org/abs/2404.18911
- *SWIFT: On-the-Fly Self-Speculative Decoding* — https://arxiv.org/abs/2410.06916

**Head-based drafters**
- *Medusa* — https://arxiv.org/abs/2401.10774
- *Hydra: Sequentially-Dependent Draft Heads* — https://arxiv.org/abs/2402.05109
- *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — https://arxiv.org/abs/2401.15077
- *EAGLE-2: Dynamic Draft Trees* — https://arxiv.org/abs/2406.16858
- *EAGLE-3: Scaling up via Training-Time Test* — https://arxiv.org/abs/2503.01840
- *HASS: Learning Harmonized Representations for Speculative Sampling* — https://arxiv.org/abs/2408.15766
- *GliDe with a CaPE* — https://arxiv.org/abs/2402.02082
- *Clover* — https://arxiv.org/abs/2405.00263
- *AdaEAGLE* — https://arxiv.org/abs/2412.18910
- EAGLE repository — https://github.com/SafeAILab/EAGLE

**Draft-free / retrieval**
- *Lookahead Decoding* — https://arxiv.org/abs/2402.02057
- *REST: Retrieval-Based Speculative Decoding* — https://arxiv.org/abs/2311.08252
- *Token Recycling* — https://arxiv.org/abs/2408.08696
- *SAM Decoding: Speculative Decoding via Suffix Automaton* — https://arxiv.org/abs/2411.10666
- *Oilbird: Training-Free Speculative Decoding with Keys the Verifier Already Computes* — https://arxiv.org/abs/2608.03839

**Multi-token prediction**
- Gloeckle et al. — *Better & Faster LLMs via Multi-token Prediction* — https://arxiv.org/abs/2404.19737
- *DeepSeek-V3 Technical Report* — https://arxiv.org/abs/2412.19437
- *AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction* — https://arxiv.org/abs/2608.00434
- Config verification: https://huggingface.co/zai-org/GLM-4.6/raw/main/config.json , https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json , https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json , https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json

**Block-parallel / semi-autoregressive (2026 frontier)**
- *DFlash: Block Diffusion for Flash Speculative Decoding* — https://arxiv.org/abs/2602.06036
- *DDTree: Accelerating Speculative Decoding with Block Diffusion Draft Trees* — https://arxiv.org/abs/2604.12989
- *DFlare: Scaling Up Draft Capacity* — https://arxiv.org/abs/2606.02091 (code: https://github.com/Tencent/AngelSlim)
- *Domino: Decoupling Causal Modeling from Autoregressive Drafting* — https://arxiv.org/abs/2605.29707
- *DominoTree* — https://arxiv.org/abs/2607.08642
- *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* — https://arxiv.org/abs/2607.05147
- *JetSpec: Breaking the Scaling Ceiling with Parallel Tree Drafting* — https://arxiv.org/abs/2606.18394
- *PCTree: From Chains to Trees* — https://arxiv.org/abs/2608.02123
- *DBLAST: Dependent Block Drafting for Stochastic Speculative Decoding* — https://arxiv.org/abs/2608.05448
- *AngelSpec* — https://arxiv.org/abs/2607.25852
- *FlexDraft* — https://arxiv.org/abs/2605.20022
- *PEFT-BD (negative result)* — https://arxiv.org/abs/2607.12422

**Jacobi / parallel**
- *CLLMs: Consistency Large Language Models* — https://arxiv.org/abs/2403.00835

**Diffusion LMs**
- *LLaDA: Large Language Diffusion Models* — https://arxiv.org/abs/2502.09992
- *Block Diffusion (BD3-LM)* — https://arxiv.org/abs/2503.09573
- *Fast-dLLM* — https://arxiv.org/abs/2505.22618
- Gemini Diffusion — https://deepmind.google/models/gemini-diffusion/
- Inception Labs (Mercury) — https://www.inceptionlabs.ai/

**MoE, offload, long context, batching**
- *AcceptMoE* — https://arxiv.org/abs/2608.02989
- *DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE* — https://arxiv.org/abs/2607.24434
- *D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding* — https://arxiv.org/abs/2607.14647
- *TriForce* — https://arxiv.org/abs/2404.11912
- *MagicDec* — https://arxiv.org/abs/2408.11049
- PowerInfer — https://github.com/SJTU-IPADS/PowerInfer
- KTransformers — https://github.com/kvcache-ai/ktransformers

**Lossy verification**
- *Revisiting Lossy Verification in Speculative Decoding* — https://arxiv.org/abs/2607.26627
- *Approximate Speculative Decoding* — https://arxiv.org/abs/2608.03447

**Consumer-hardware reality check**
- *Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware* — https://arxiv.org/abs/2607.17283

**Engines (source verified 2026-08-18)**
- llama.cpp — https://github.com/ggml-org/llama.cpp (`common/common.h`, `common/speculative.h`, `common/arg.cpp`)
- vLLM — https://github.com/vllm-project/vllm (`vllm/config/speculative.py`); docs https://docs.vllm.ai/en/latest/features/spec_decode/
- SGLang — https://github.com/sgl-project/sglang (`python/sglang/srt/speculative/spec_info.py`); docs https://docs.sglang.ai/advanced_features/speculative_decoding.html
- TensorRT-LLM — https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html
- HuggingFace Transformers — https://huggingface.co/docs/transformers/generation_strategies#speculative-decoding
