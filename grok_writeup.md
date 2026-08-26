# Intelligence overlay — local LLM inference, 2026-08-21

*A field report to supplement `research/00-synthesis.md` and reports `01`–`07` (compiled 2026-08-18).*

---

## Document control

| Field | Value |
|---|---|
| **Title** | Intelligence overlay — local LLM inference |
| **File** | `grok_writeup.md` (repo root, sibling of `research/`) |
| **Type** | Time-boxed field overlay. Not a replacement for the Aug-18 corpus. Not an engine. |
| **Compiled** | 2026-08-21 (America/local session; crawl spanned ~one afternoon) |
| **Corpus it overlays** | `research/00-synthesis.md` + `01`–`07`, produced 2026-08-18 |
| **Collection window** | Primary: **2026-08-18 → 2026-08-21**. Background: ~June–August 2026 when needed to interpret the 3-day delta. |
| **As-of date for “current”** | 2026-08-21. Engine version pins and X posts older than ~48 h are labeled when used. |
| **Produced by** | Grok 4.6 (xAI) in the Grok Build TUI, driving 10 parallel general-purpose subagents plus parent-level X/web/GitHub fetches. |
| **Intended reader** | Someone who has read, or will read, `research/00-synthesis.md` and wants what shipped / what operators said in the three days after that freeze. |
| **Not for** | Capacity planning off a single tok/s tweet. Citing vendor 15× as a 4090 number. Treating this as independently reproduced benches. |
| **Suggested citation** | `inference_engines/grok_writeup.md` (2026-08-21 overlay on the 2026-08-18 research corpus). |
| **Stale after** | ~2–4 weeks for versions, model defaults, and open PRs. The *structural* claims (DFlash2 vs MTP is workload-dependent; 4-bit ≠ quality cliff on Qwen3.8; megakernel×4-bit×consumer still open) should last longer. |

### What this file is relative to the corpus

| Layer | Role | Freeze date |
|---|---|---|
| `research/00-synthesis.md` | Governing law, bags of tricks, open gaps, recommended stack | 2026-08-18 |
| `research/01`–`07` | Evidence behind the synthesis (~56k words, ~500 URLs) | 2026-08-18 |
| **This file** | Operator/SOTA overlay: what shipped, what people run, mood, gap motion | 2026-08-21 |

If this file and the corpus disagree on a *fact about the world as of Aug 18*, trust the corpus (it was cross-checked against live sources). If they disagree on *what happened Aug 18–21*, trust this file.

### How it was collected

Ten background subagents, each with web search, plus a parent pass over X and primary docs:

| # | Slice | Subagent id (session) |
|---|---|---|
| 1 | Engine landscape (llama.cpp, vLLM, SGLang, ExLlama, MLX, TRT-LLM, Ollama, …) | `01a02504-7ce8-7fe3-93cc-2280fc6a3d5f` |
| 2 | Quantization (GGUF, EXL3, NVFP4/MXFP4, Dynamic 3.0, KV) | `01a02504-7ce8-7fe3-93cc-2299a675086f` |
| 3 | Speculative decoding (DFlash2, MTP, DSpark, consumer negatives) | `01a02504-7ce8-7fe3-93cc-22ab78475d12` |
| 4 | Consumer hardware practice (5090, Strix Halo, Spark, Arc, power) | `01a02504-7ce8-7fe3-93cc-22bec7b41146` |
| 5 | Community sentiment (HN, LocalLLaMA, GitHub issues) | `01a02504-7ce8-7fe3-93cc-22c199ce90d5` |
| 6 | Kernels / megakernels / FA4 / FlashInfer / Metal | `01a02504-7ce8-7fe3-93cc-22d6e09da669` |
| 7 | Recent arXiv (2606–2608) and survey misses | `01a02504-7ce8-7fe3-93cc-22e8988c6e8d` |
| 8 | Model/architecture landscape (configs fetched live) | `01a02504-7ce8-7fe3-93cc-22f1abd1c40c` |
| 9 | MoE offload, KV hierarchy, prefix cache | `01a02504-7ce8-7fe3-93cc-230d4115a637` |
| 10 | Gap hunt / controversies / new products | `01a02504-7ce8-7fe3-93cc-23178e4d3d51` |

**Parent-level (not delegated), used to cross-check agents:**

- X semantic search and X keyword search (`Latest`), including queries for llama.cpp / vLLM / SGLang / ExLlama, DFlash / DSpark / MTP, Qwen3.8-27B, NVFP4 / EXL3 / Unsloth, 5090 / Strix Halo / Spark, NInfer / Escha / megakernel.
- `x_thread_fetch` on Fateev’s 4,800-task quant thread (`2090318703992717486`).
- Direct fetches: [Inco DFlash 2 blog](https://inco.ai/blog/dflash2/), [Escha-W2 model card](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2), [NInfer README](https://github.com/Neroued/ninfer), plus web search for Qwen3.8, DeepSeek-V4-Flash-on-Spark, Ada-MK.

Agents did not have X API access; X material in this file is from the parent pass unless an agent cited a tweet URL that was then kept.

### Source classes and how much to trust them

| Class | Examples | Trust | How used here |
|---|---|---|---|
| **A — Primary artifact** | Hugging Face `config.json`, GitHub README/PR/release, engine docs, Inco blog tables | High for “it exists / this is the API.” Medium for author tok/s. | Architecture tables, engine status, DFlash2 method. |
| **B — Independent multi-hour bench** | Fateev 4,800-task thread; Stellakjbk 5090 GSM8K A/B with seed/concurrency disclosed | High *for that stack and that suite*. Not a universal ranking. | Quality-vs-bits; DFlash2 vs MTP. |
| **C — Operator post (X / Reddit / HN)** | Single-box tok/s, “I tried DFlash2 on Vulkan,” HN quotes | Indicative. Hardware, quant, ctx, think-mode, and single-stream vs aggregate are often missing. | Sentiment, existence proofs, “this was slower.” |
| **D — Vendor / first-party blog** | LMSYS, NVIDIA 15×, SGLang DSpark 383 tok/s, Unsloth cards | Directionally useful. Baseline and hardware are often datacenter. | Labeled vendor. Never silently compared to a 4090. |
| **E — arXiv 2606–2608** | HiSparse, OasisKV, AcceptMoE, FreeToken, Ada-MK | Idea-grade. Speedups are author claims until a second implementation. | Gap motion, “steal this interface.” |
| **F — Search-index extract** | Reddit threads that did not return full HTML | Quote only when the extract included the sentence; otherwise paraphrase and link. | Sentiment section. |

**Hard collection limits**

- Reddit HTML is often blocked from this environment. LocalLLaMA quotes in §3 come from search-index extracts and pages that returned full text (PSA, ExLlama dual-3060, ik/LM Studio, ROCm-vs-Vulkan, MTP-buffer). If a Reddit quote matters to you, open the URL.
- X posts are a recency-weighted sample, not a census. Japanese, Chinese, and Spanish operator threads are over-represented in the 48-hour `Latest` scrape because that is who was posting Qwen3.8 numbers on Aug 21.
- No numbers in this file were reproduced on hardware we control. Community llama-bench tables in `research/07` remain the cleaner apples-to-apples set for *stock* llama.cpp.
- Subagents can stale-merge (e.g. an agent repeating an Aug-18 fact as if it were new). Parent pass preferred live fetches for DFlash2, Escha, NInfer, Fateev, and Qwen3.8 config.

### How to read numbers in this file

- **Single-stream decode** is what one user sees. **Aggregate** (`C=3`, `c=8`) is server throughput and will look 2–10× larger. Posts that omit this are tagged when caught.
- **Prefill (pp) and decode (tg)** are different regimes. Mixing them is the most common community error.
- **Thinking on/off and `reasoning_effort`** move tok/s and quality as much as quant. Fateev: 7.5× more reasoning tokens at `xhigh` for 0 extra pass@1 on Qwen3.8 Q4.
- **Acceptance length τ is not speedup.** A method with τ=5 can be slower than AR.
- **“Lossless”** means exact rejection sampling against the *deployed* target’s distribution, not bit-identical to BF16, and not “the blog said lossless.”

### Confidence tags used in the body

| Tag | Meaning |
|---|---|
| **Confirmed** | Two independent classes (e.g. HF config + engine source, or two operator A/Bs with disclosed setup). |
| **Vendor** | First-party number; hardware is usually not consumer. |
| **Indicative** | One operator post or one Reddit thread. |
| **Unreplicated** | arXiv 2608.* or a brand-new engine card. |
| **Open PR** | Not in a release binary; forks may already ship it. |

Untagged claims in the headline list (§0) are the compiler’s synthesis and inherit the caveats above.

### Changelog

| Date | What |
|---|---|
| 2026-08-21 | Initial overlay. Ten-agent crawl + parent X/web pass. |
| 2026-08-21 | Document-control / provenance block added (this section). |

---

## 0. Three-day headline

The Aug-18 world is intact: decode is bytes, MoE is the product, prefill is the pain, speculation is a contract with a kill switch. What changed is the *daily-driver model* and the *drafter people are starring*.

1. **Qwen3.8-27B** (released ~Aug 14) replaced Qwen3.6-27B as the 24 GB default. Same Gated-DeltaNet hybrid + MTP + vision stack, stronger coding/agent numbers, Apache-2.0. Unsloth GGUF downloads outran the official repo ~5.4× (3.56M vs 666k as of Aug 21).
2. **DFlash 2** (Inco AI, Aug 18) is the first material speculation move since DSpark. Path selector + local convolution on a still-parallel block. Vendor: 2.7–3.4× AR on Qwen3.8-27B at C=1 on H200. Consumer: 1.5–3×, often ≈ MTP, **dies vs MTP on long context and on Apple/Vulkan**.
3. **Unsloth Dynamic 3.0** (Aug 19–20) is the new GGUF default recipe. Divergence-300@32 is the eval they are selling because top-1 on one token is gamed.
4. **Two new consumer engines** appeared at the Qwen3.8 launch: **NInfer** (C++/CUDA, SM120-only, ReplaySSM, claimed ~200 tok/s on a 5090) and **Escha-W2** (2.47 bpw, custom SGLang kernels, 10.15 GB, quality table vs FP8 that independent users say is real *and* ~2× slower than Unsloth Q2 on a 5090).
5. **llama.cpp is competing with Ollama on UX** (`llama.app` / `llama serve` hit HN Aug 12, 364 points). Power users already treated Ollama as a tax. The product fight is now explicit.
6. **Nothing closed the ten research gaps.** Megakernel × 4-bit × 4090/5090 is *nudged* (Ada-MK on L20, AutoMegaKernel toys on SM120). Recurrent rollback is *half-closed* in vLLM (ReplaySSM). Predictive coding × speculation is still unclaimed.

---

## 1. What people are actually running (Aug 21)

### Decision tree, updated

| Situation | Engine | Mood |
|---|---|---|
| Single user, newest model, weird hardware | **llama.cpp from source** | Cultural default. “ffmpeg of AI.” |
| NVIDIA, model fits VRAM, quality-per-GB | **ExLlamaV3 + TabbyAPI** | Loud-positive this week. Dual-3060 Reddit: Qwen3.8 tg **95 vs llama.cpp 40**. |
| Concurrent users / agents / NVFP4 | **vLLM or SGLang** | “llama-server or vLLM if you’re serious.” |
| Apple Silicon | **MLX** (or Ollama `*:mlx`) | Gap vs llama.cpp now “within 10%.” Prefill still the Mac tax. |
| Hybrid CPU/GPU MoE, CUDA + lots of RAM | **ik_llama.cpp** | 15–22% faster. Vulkan abandoned. |
| DeepSeek/Kimi/GLM on 512 GB EPYC | **KTransformers / kt-kernel in SGLang** | Unchanged. |
| Don’t want to think | **Ollama / LM Studio** | Distribution win; social radioactive among people who know GGUF. |
| SM120, Qwen3.6/3.8 only, max tok/s | **NInfer** | New. Closed model set. 5090-only. |
| 2-bit quality experiment, Linux+NVIDIA | **Escha runtime** | New. Not GGUF. Fork of SGLang. |

HN, Aug 12 ([llama.app thread](https://news.ycombinator.com/item?id=49267928)):

> “At this point the options are llama-server or vLLM if you're serious about running things at your desk in the under 256GB RAM size class.”
> — walrus01

> “llama.cpp is like the ffmepg of AI, and one of the reasons I so greatly dislike ollama is that the latter completely obfuscates that they're a rebrand of the former.”
> — halyconWays

> “I guess overall it's the worst runtime I've seen so far, except for all the other runtimes out there...”
> — imrehg

X consensus is the same, slightly more operational: compile llama.cpp with the right `CMAKE_CUDA_ARCHITECTURES` (120 / 89 / 86); Ollama and LM Studio hide the flags you need.

### Default models

| Slot | Aug 18 | Aug 21 |
|---|---|---|
| 24 GB daily driver | Qwen3.6-27B | **Qwen3.8-27B** (dense hybrid, vision, 262K, built-in MTP) |
| 16 GB fast MoE | Qwen3.6-35B-A3B | Unchanged; still faster than dense 27B at same quality-per-watt on NVIDIA |
| 32 GB 5090 | 35B-A3B / Scout | Qwen3.8 Q6/Q8 or NVFP4; UD-Q6_K_XL ~100 t/s |
| 96 GB one card | gpt-oss-120b | Same + Qwen3.5-122B + MiniMax |
| 128 GB unified | gpt-oss-120b | Same + Qwen3.8 at 256k. Spark for prefill; Strix/Mac for silent capacity |
| Huge MoE | GLM-5.2, Kimi K2, DeepSeek V4 | V4-Flash 0731 on **one** Spark via EXL3 3.0 bpw is the new “it runs” recipe |

Qwen3.8-27B architecture (live `config.json`, Aug 21): 64-layer dense, 3:1 Gated DeltaNet + gated GQA, `mtp_num_hidden_layers=1`, 262K, vision tower. Drop-in for engines that already run Qwen3.6-27B.

---

## 2. Live numbers from X (last 48 hours)

These are not leaderboard numbers. They are what operators posted on 2026-08-20/21 (**class C — indicative**, except Fateev’s quality thread and Stellakjbk’s seed-disclosed A/B, which are **class B**). Hardware clocks, power limit, think-mode, and single-stream vs aggregate are often omitted. Status IDs are in §16.

### Qwen3.8-27B decode, single stream unless noted

| Hardware | Stack | tok/s | Notes | Who |
|---|---|---|---|---|
| RTX 5090 | SGLang + DFlash2, NVFP4, GSM8K, T=0, think off | **172 → 260** (+51%) | TTFT 80 ms MTP vs 190–200 ms DFlash2. At 8K ctx they **tie**, MTP slightly ahead | @Stellakjbk, A/B 128 GSM8K |
| RTX 5090 | SGLang NVFP4 + EAGLE/MTP, think off, **C=3 aggregate** | **~305 agg / ~102 per stream** | Throughput, not single-stream | @b0dre |
| RTX 5090 | llama.cpp Q5_K_M + MTP3, 128K, long-agent | **109.5 avg**, 89.6% accept, up to 106K ctx | MTP5 past the knee | @IgyutaeI83275 |
| RTX 5090 | llama.cpp Unsloth Q5, 256K | **75–85** | Hermes agent, not a power user | @PAguiar_NH |
| RTX 5090 | llama.cpp “out of the box” | **115** | No spec | @TailLatency |
| RTX 5090 | unnamed engine, NVFP4, **170K ctx + vision**, 29.5 GB | **110–150**, PP 3.3k @ 64K | “Not llama.cpp, not vLLM.” Long ctx is the point | @indesjyo |
| RTX 5090 | Escha-W2 2-bit | **84.3** (card says 82.6–87.1) | vs Unsloth UD-Q2 **168**. η ≈ 47% of BW ceiling | @bountyAIhunter |
| RTX 5090 | NInfer NVFP4 MTP3 C=1 | **143.8** committed (48.9% accept) | Official NInfer table. C=8: **766 agg** | NInfer README |
| RTX 5090 | llama.cpp Q4_K_M, power 80% | **58.6** on Qwen3-**32B** | −1.4% vs 100%, −106 W | Japanese operator, Aug 21 |
| RTX 6000 / PRO | llama.cpp, 4 prompts median | AR 47.4 → MTP 114.7 → DFlash 99.3 → **DFlash2 140.6** | One of the cleaner consumer A/Bs | r/LocalLLaMA via agents |
| RTX PRO 6000 | SGLang DFlash2 vs llama.cpp MTP | DFlash2 **150** vs MTP **50–60** | “50–60 for RTX 6000 is too low, you have an issue” | @DNesmrtelny |
| Dual 3090 | SGLang, MTP4 / DFlash2 / DSpark | DFlash2 “sweet spot”; DSpark 315 vs 248 on “count to 300”; MTP4 deepest KV (~500K) | Real tasks: DFlash2 ahead, **~30%** | @Tech2Wild |
| 5070 Ti + 5060 Ti | DFlash2 Q4_K_M | **62**, 2.33× AR, +24% vs MTP | Mid-range existence proof | @GuanceLim6277 |
| RTX 4090 | llama.cpp Unsloth GGUF, 213K, MTP | **64.9 decode / 1,107 prefill** | Long-ctx PoCs | @pongphat_c |
| Dual 3060 | ExLlamaV3 vs llama.cpp | tg **95 vs 40** (27B); **150 vs 84** (35B-A3B) | The post converting NVIDIA users this week | r/LocalLLaMA Aug 16 |
| Strix Halo | llama.cpp fork + DFlash2 | **41**, “nearly 4× without spec” | Laurent Zuijdwijk, Aug 21 | @laurent_zw |
| Strix Halo | stock vs MTP vs DFlash2 | 10.5–11.8 → MTP 24–36 → DFlash2 **21.1 at 32k** | Old DFlash collapsed at 32k | agent hardware report |
| M5 Max | oMLX DFlash2 | vendor **70 / 4.6×** | Best-case. Independent long-task: 80 pp / 30 decode | Inco blog + field |
| Intel Arc Pro B70 32 GB | Ollama Q4 vs vLLM GPTQ vs MTP4 | **12.5 / 33 / 65.5** | ~5.3× Ollama once stacked | @BullBoss5, Aug 21 |
| DGX Spark | llama.cpp Q4 | **11.6 tg / 837 pp** (day-0) | Saiyam Pathak | |
| DGX Spark | EXL3 3.0 bpw DeepSeek-V4-Flash | decode **~30** (was 26); prefill **53s → 26s** on 30k prompt | Prefill 2×. Context 1M → 384k | @Kris_Collo quoting @MiaAI_lab |
| RX 9070 XT Vulkan | DFlash2 vs Draft MTP | DFlash2 **9–23**, MTP **38–45** (77–81% accept) | “Usable speed never happened.” MTP wins | @LOREbeginning |

### The pattern those posts share

- **Peak tok/s is a workload.** DFlash2 260 on GSM8K think-off is real. SWE-Bench Verified on a PRO 6000: DSpark / DFlash2 / MTP all **~130** (@_thomasip). Code is *not* automatically a spec-decode free lunch once the agent is in a long repo.
- **TTFT vs decode is the DFlash2 tax.** MTP ~80 ms first token; DFlash2 ~190–200 ms. Short Q&A stays on MTP. Long structured answers flip.
- **Long context eats DFlash2’s lead.** Same 5090 A/B: at 8K in / 1K out, MTP 133.7 vs DFlash2 130.5.
- **Backend can nullify the algorithm.** SYCL MTP at 100% accept is **21% slower** than AR because dispatch is 100–500 µs vs ~5 µs CUDA. Vulkan DFlash2 on 9070 XT lost to MTP. Quantized Metal in “Lossless but Not Free” ran verification serially.
- **Aggregate ≠ single-stream.** 305 t/s at C=3 on a 5090 is 102 per user. Spark 33 → 862 at c=256 is the same curve. Ollama does not batch; that is now quantified.

---

## 3. Sentiment

### Ranked feelings

1. **llama.cpp is the default and the punching bag.** Loved for coverage and hybrid offload. Hated for “move fast, break things, rarely fix” on AMD, silent M5 tensor-core regression since Nov 2025 (`#27473`, filed Aug 21), `--fit` MoE footguns, and CLI flag churn.
2. **Ollama is socially radioactive among people who already know GGUF, and still the path of least resistance for “Claude Code on a local model tonight.”** StorageReview still names it “Best Overall.” HN: “Friends don't let friends use ollama.” Qwen3.8: 502k Ollama pulls in six days. Unsloth GGUF: 5.4× official downloads. Distribution ≠ engine quality.
3. **ExLlamaV3 is having a week on NVIDIA.** v1.0.0 (Jul 15) + Aug 16 dual-3060 numbers. “Significantly faster than llama.cpp and that’s crazyy.” Hard limit: no ROCm. TabbyAPI onboarding is still “a mission.”
4. **Speculation is a lottery, and the community has internalized that.** Adaptive MTP PR `#27210` (Aug 17) and DFlash2 PR `#27342` (106 reactions, Aug 18) are the constructive response. n-gram is almost nobody’s daily driver. PCTree wins on 8B and loses on 27B.
5. **Vulkan first on AMD, ROCm only for long-context prefill or TP.** This is now advice from people who *wrote* HIP kernels, not just frustrated users. HIP disaster still cited: RX 9070 XT Qwen 3.5 4B Q4_0, ROCm pp512 **50 vs Vulkan 3760**.
6. **Qwen 27B-class dense is the 24 GB religion.** 35B-A3B MoE is the 16 GB / partial-offload religion. DeepSeek V4-Flash is the “save up for 192 GB” religion. Everything 1T+ is a cluster or an API.
7. **Prefill / prompt cache is still the user-visible pain.** Decode bragging is for screenshots. Hybrid/SWA (Qwen3.5/3.6/3.8) still invalidates checkpoints. 80× TTFT on exact repeats is real and fragile.

### Quotes that actually capture the week

> “don't blame models for your choice of runtime.”
> — r/LocalLLaMA PSA

> “Vanilla llama.cpp leaves a lot of performance on the table. I'm reaching 120 t/s with a custom inference engine for a model that llama.cpp can barely run at 70 t/s.”
> — LoganDark, HN, on llama.app

> “MTP is ~21% slower than generating without speculation, despite 100% draft accuracy.”
> — llama.cpp `#23533` (SYCL / Arc B70)

> “I'd just recommend going with llama.cpp Vulkan and skipping ROCm completely.”
> — lhl, HN

> “DFlash2 is not a fixed 300 tok/s accelerator. What decides daily experience is how many draft tokens your workload actually accepts.”
> — @Stellakjbk, 5090 A/B, Aug 21 (paraphrase of a long measured thread)

> “I spent 67 hours of model time to find out how much dumber 4 bit really makes Qwen3.8-27B. … There is nothing to show. The gap between quants is smaller than the gap between reasoning presets.”
> — Alexey Fateev (@superalesha), 4,800 tasks, Aug 20

---

## 4. Engine landscape — what moved since Aug 18

### llama.cpp

Tip around **b10549** (Aug 21). Daily releases.

- **`--fit` is default-on** and still has MoE-specific bugs: granularity-128 caused a **~42% TG regression** on 8 GB MoE (`#25224`); MTP GGUFs under-counted NextN (`#26177`, ~10% TG on one model); fused GDN disabled when layer 0 is on CPU (`#27327`, still live). Expert hot-store `-ehs` (`#26563`) closed for redesign.
- **`--spec-type` is a list.** `draft-mtp,ngram-mod` is a free code-path win on code, ~−1% on prose, zero extra VRAM. DFlash2 is **PR `#27342`, not a release binary**. Tensor-split asserts; layer-split workaround drops 80+ t/s → ~18.
- **`--mmap/--no-mmap` deprecated for `--load-mode`.** Flag churn is the quiet complaint on X (`@repojournal`).
- **MoE expert cache is not merged.** RFC `#24528`: keep `MUL_MAT_ID` on CPU, dispatch hits to GPU. Independent: +25–72% TG on the right box, **−31% on a 1080 Ti**. Prefill can go backwards. Competing forks (Lidenburg, miltos22, ParmesanParty).
- **Security:** DEF CON 34, 10 llama.cpp vulns; two llama-server UAFs CVSS 9.2. Five of ten still unpatched as of a June audit.
- **llama.app / `llama serve -hf`** is an explicit grab for Ollama’s remaining advantage.

### vLLM

**v0.27.1** (Aug 11). PyTorch 2.13 + Triton 3.7.1 — breaking env.

- MRV2 is the production path. +56% on Qwen3-**0.6B** (do not quote that as a 70B number). GLM-4.7 MTP TPOT **−6.3%**.
- DFlash2 merged `#52816` (Aug 21). DSpark + DFlash already in 0.25+.
- Consumer: NVFP4 KV on SM120 still landing (`#46329`). Community backport: Qwen3.8-27B 262k on a 32 GB 5090 (fp8 KV could not).
- Draft-model spec in MRV2 still **open**. Windows community wheels exist.

### SGLang

Three design posts in the window: DSpark (Jul 6), Unified Radix Cache (Aug 11), Breakable CUDA Graph (Aug 17, now default prefill). BCG extracted to [meta-pytorch/breakable-cuda-graphs](https://github.com/meta-pytorch/breakable-cuda-graphs). HiCache docs updated Aug 21. DFlash2 treated as a DFLASH checkpoint.

Vendor-adjacent: GB300 disagg DSv4-Pro **~11,200 tok/s/GPU @ ~50 tok/s/user**, 5× Day-0. Do not plan a 5090 around that number.

### Others, compressed

| Engine | Status Aug 21 |
|---|---|
| **ExLlamaV3** | v1.4.2 (Aug 11). CPU offload exists. No ROCm. Dual-3060 community numbers are the conversion event. |
| **MLX** | 0.32.1 **Windows wheels** (Aug 18). CUDA extras opt-in. Docs still Darwin-default. WDDM quantized-prefill tax patched (+98% prefill on 5090 Win11). |
| **TensorRT-LLM** | **TRT backend deleted** (Jul 14, −92k LOC). Name is now a PyTorch serving stack. Edge-LLM still uses real TRT engines. Spheron “28 min compile” articles are describing the dead path. |
| **FlashInfer** | 0.6.17 (Aug 11). SM12x fused-MoE FP4 accuracy fix. MegaMoE EP production-ready. Treat NVFP4 MoE on SM120 as “should work now, measure.” |
| **Ollama** | v0.32.15. **All GGUF via llama-server subprocess.** MLX runner for Apple safetensors. ggrun bake-off: slower than raw `--fit` on large MoE. |
| **ktransformers** | kt-kernel 0.7.0 on PyPI (Aug 18). Product is a CPU MoE library inside SGLang, not a standalone server. |
| **mistral.rs** | v0.9.1/0.9.2. DFlash batched drafter. Maintainer “2.8× llama.cpp” is unreproduced. |
| **Modular MAX / Mojo** | Qualcomm closed the acquisition Jul 29. **Mojo 1.0** Aug 11. Compiler Apache-2.0. Homepage “171% of vLLM” still vs **vLLM 0.10.1**. Stale. |
| **LMDeploy** | v0.16.0 (Aug 19). CVE-2026-63764 SSRF 9.2 on ≤0.14. Confirm patch before assuming 0.16 is clean. |
| **NInfer** | New. C++/CUDA, **RTX 5090 only**, closed artifact set (Qwen3.6/3.8 27B ± NVFP4, 35B-A3B). ReplaySSM, PDL, CUDA graphs, MTP 1–5, DFlash on 35B only. Apache-2.0. This is the “specialized consumer engine” the corpus recommended, in miniature. |
| **Escha** | New. Mixed 2/3-bit (`escha` / 2.469 bpw), custom `escham_decode_gemv`, SGLang fork, Linux+NVIDIA sm_80+. Vision tower quantized *out*. Ampere needs `ESCHA_ROUTE=blackwell` for 1.72× at batch 1. Prefix cache does **not** help a growing conversation (recurrent-state boundary). |
| **FreeToken** | arXiv 2608.16157 (Aug 17). MIT/Berkeley. GLM-5.2 753B NVFP4 on **one** RTX PRO 6000 at 14.9 t/s. Bandwidth-adaptive expert residency. Code: FlashML-org/FreeToken. Offload serving, not “fits in 96 GB.” |
| **TileRT** | Pluggable decode engine behind vLLM’s KV connector. One in-flight request per decode node. The seam is the idea. |
| **ggrun / vllm.cpp / Tiny-vLLM** | Exact-VRAM MoE launcher; C++ vLLM-alike. Small, they exist. |

---

## 5. Quantization

### Dynamic 3.0 (after the corpus)

Unsloth Dynamic v3.0, Aug 19–20. Qwen3.8-27B GGUFs 6.19–31.46 GB. Claims **>10% top-1% at the same size** vs every other provider on Divergence-300@32. PTQ only. MTP stripped from quants ≤ `UD-Q2_K_XL`. `UD-IQ1_S` is 6.2 GB / ~72% top-1% — Unsloth says **do not use 1-bit for agents**. Independent token-level grading (@sudoingX, Aug 18): Unsloth UD-Q4 94.73% same-token vs BF16; AtomicChat AD-Q4 **95.39%**. UD is the download default, not automatically first at Q4.

### Fateev’s 67-hour 4-bit quality bench (the week’s best measurement)

@superalesha, 4×3090, Qwen3.8-27B, 4,800 tasks, 14.5M reasoning tokens, no token caps. Five stacks: FP8 vLLM, NVFP4 vLLM, AWQ INT4 vLLM, GGUF Q4_K_M llama.cpp, NInfer.

At `xhigh`, pass@1 on the full suite: **AWQ 90.0 / NVFP4 89.3 / GGUF 89.3 / FP8 88.7 / NInfer 88.0**. McNemar: statistical tie. The 4-bit quants scored *above* FP8. The only statistically real gap: **NVFP4 with reasoning OFF fell apart on HumanEval+ (13/30 vs FP8 30/30)**; turn reasoning to `low` and it is 90/90.

GGUF Q4_K_M at `low` and `xhigh` both 89.3%, with **86k vs 651k reasoning tokens** (7.5×). Cheat sheet he published: max quality = AWQ INT4 at xhigh; daily driver = **GGUF Q4_K_M at low**; never reasoning off (−8 to −12 points).

This is the corpus’s “Quantization Inflates Reasoning” paper, measured on the model everyone is downloading. **The knob is `reasoning_effort`, not bit-width, for this 27B at 4-bit.**

He is now grinding Unsloth Dynamic from Q4 down to 1-bit on 720 agentic/reasoning tasks. Place bets: does 2-bit survive?

### Escha-W2 vs Unsloth Q2 (the runtime column the launch skipped)

Escha card: 10.15 GB, ~100% of FP8 on Commonsense-6 / GPQA-Diamond / LiveCodeBench, 87.1 tok/s bs1 on a 5090. Independent (@bountyAIhunter):

| | Unsloth UD-Q2_K_XL | Escha-W2 |
|---|---|---|
| Size | 9.83 GB | 10.15 GB |
| llama.cpp / Ollama / vLLM | yes | **no** (custom SGLang wheel, torch 2.9, CUDA 12.8) |
| 41-task pass | 27/41 | **31/41** |
| 5090 single stream | **168 tok/s** | 84.3 |
| Vision | intact | quantized out |
| Time to first token | minutes of GGUF | 38 minutes of ABI pain |

Bandwidth ceiling on a 5090: 1792 / 10.15 ≈ 176 tok/s. Unsloth sits near it. Escha sits at ~48% — dequant of a mixed 2-to-4-bit layout. Ampere default route is 23.6 tok/s until `ESCHA_ROUTE=blackwell` (40.7, +73%). Someone ported native 2-bit decode into a llama.cpp fork (14.5 GB, ~43 t/s on a 3090, PPL match, MTP dropped).

**Lesson the corpus already had, now with a product:** VQ/trellis-class 2-bit quality is real; **kernels and distribution decide whether it is a daily driver.** Same story as EXL3 vs GGUF.

### What did not change

- `Q4_K_M` / `UD-Q4_K_XL` is still the practical floor. StorageReview Aug 19: below it, tool-call reliability falls faster than general quality.
- `iq4_kt` still loses to `iq4_ks` on CPU/hybrid. Trellis is not automatically a win.
- NVFP4 W4A4 on Blackwell wants **cute-DSL / FlashInfer, never Marlin** (2.5× slower). Spark without `--moe-backend flashinfer_b12x` is a trap.
- Mix-Quant (two paths: NVFP4 prefill, BF16 decode) is still the closest dual-layout existence proof. One checkpoint, two packed layouts: **still open**.
- APreQEL is still the only paper that puts llama-bench latency in a mixed-precision objective, and it is small models after K-quant types exist.
- REAP + NVFP4 on Qwen3.8-**2.4T** (Red Hat, Aug 20): GPQA Diamond 91.5 vs 92.6 at 25% experts pruned. Criterion still does not transfer across families.

---

## 6. Speculative decoding

### DFlash 2, technically

Inco AI (Z Lab spin-out), blog 2026-08-18. DFlash already ships in SGLang, vLLM, TRT-LLM, llama.cpp. NVIDIA “15×” is a **throughput-at-SLO** number on 8×B300, not batch-1. DFlash 2 keeps the one-pass block and adds:

1. **Path selector** (2.0M params, +0.6% cycle): top-16 candidates per position, pairwise bilinear score, greedy/sample walk. Recall@1 at pos 0 is 85.4%; Recall@16 is **99.5%**. The gap is selection, not coverage. Beats a DSpark-style GRU correction (77.8M, +9.6%) at 40× fewer params.
2. **Two-tap dynamic convolution** (16.5M / +3%, +0.7%): fights suffix decay. Five-layer + conv ≈ 15-layer DFlash. Within-block attention in late layers 9.4% → 0.5%.

Together +1.3% cycle latency, ~21% more accepted tokens vs DFlash. Qwen3.8-27B mean τ: MTP 4.28 / DSpark 3.62 / **DFlash2 4.80**.

At **concurrency 32**, MTP and DSpark go negative on several tasks. DFlash 2 is 1.01–1.45× — the only method that still helps, and barely on chat.

Engine status Aug 21: SGLang ships it as DFLASH. vLLM `#52816` merged. llama.cpp `#27342` open. Ollama `#17865`. oMLX fork.

### Ranking by deployed impact (not paper τ)

| Rank | Method | Consumer expectation |
|---|---|---|
| 1 | **Native MTP** | What people leave on. 1.4–2.2× dense. Often a *loss* on small-active MoE. |
| 2 | **DFlash / DFlash 2** | Industry default *drafter*. Consumer 1.5–3×, workload-dependent. Long ctx → MTP. Apple/Vulkan → often MTP. |
| 3 | **DSpark** | Production DeepSeek-V4 (60–85% faster *vs MTP-1*). Community vLLM ~1.5× over MTP-1. |
| 4 | **n-gram / suffix / Oilbird** | Free. Highest value-per-line for agents. Oilbird 4.4× on API-Bank. |
| 5 | **EAGLE-3 / P-EAGLE** | Baseline everyone reports against. P-EAGLE (`parallel_drafting: true` in vLLM) was a hole in `03`. 1.05–1.69× over EAGLE-3. Not in llama.cpp. |

### Negative results, still binding

- Lossless but Not Free: 3/5 Apple configs slower; best 1.61×.
- Qwen3.6-35B-A3B on 3090: 19 draft configs, all slower than 135 t/s.
- Q4_K_M creative: **−9%** with MTP. F16 creative: **+67%**. Bandwidth-starved loves spec; already-fast Q4 on high-entropy text does not.
- Offloaded MoE: γ=8 → 0.67×; **γ=1 is optimal (1.80×)**. AcceptMoE 2.06× under physical offload, not lossless.
- llama.cpp draft-n 16 **halves** Strix throughput; optimum 3–4.

**Nobody ships the corpus’s 10-second autotuner** (ρ_M serial-detection + EWMA disable). Adaptive MTP `#27210` is the first in-tree attempt to close that loop. Still not union-aware, still not per-phase.

**Predictive coding × speculation: still zero papers.** Rechecked Aug 21. Adjacent: OasisKV (lookahead as KV prefetch), SparDA, FEP-MoE routing, PixelPrune. The correspondence remains exact and unclaimed.

---

## 7. Hardware practice — delta

The Aug-18 structural facts hold. Three updates:

### Dense 27B + a drafter is the new 24 GB meta

Not only sparse MoE. Qwen3.8 stock ~40 t/s vs tuned **114 t/s / ~1,000 agg at 64 streams** on a **250 W 3090** (syv-ai recipe: vLLM + MTP + int8 GEMM). That is a 2.8× *stack* gap on the same card.

### Spark vs Strix Halo: prefill is software

Same ~270 GB/s class. Spark field notes: **~6,000 t/s prefill**, 27k prompt in 3.6 s. Strix Halo gpt-oss-120b pp 340–470. The 08-18 slogan “prefill is the pain” is **true on Apple/AMD and false on Spark once CUDA/FlashInfer kernels exist**. A new engine that treats unified-memory AMD/Apple as “Spark-class once GEMM is good” is the right bet.

DeepSeek V4-Flash 0731 on **one** Spark (EXL3 3.0 bpw, SparkInfer): decode barely moved (~26 → 30); **prefill 2×**. Agentic workloads are prefill-bound. Dual-Spark FP8 cluster is no longer required for that model if you accept 384k ctx and a community quant.

### Power limiting is still nearly free

3090 225–250 W is the LLM sweet spot (~0.42 vs 0.27 tok/s/W at 450 W). 5090 80% power: −1.4% speed, −106 W. Spark lock 2200 MHz: ~25% GPU power drop, decode holds, prefill −6%. Decode is bandwidth-bound; the extra watts buy clocks you are not using.

Laptop 5090 vs M5 Max (Wccftech, Aug 20): NVIDIA wins until VRAM fills. Qwen3-4B at 262k: laptop 5090 **25 t/s** vs M5 Max **181 t/s**. At 68k: 5090 **160 t/s**. Unified memory is a *capacity* weapon.

PRO 6000 street **~$8–16k** (roughly double original). 600 W vs Max-Q 300 W is a real SKU split (223 vs 119 t/s on the same Qwen3.8 NVFP4).

---

## 8. Kernels and the megakernel gap

**Megakernel × 4-bit × consumer 4090/5090 is still unclaimed** as a 7B+ production result.

| Work | Quant | GPU | Scale | Status |
|---|---|---|---|---|
| Ada-MK (2605.11581) | GPTQ W4A16 | L20 SM89 | Qwen 1.5–1.7B | Closed TRT-LLM plugin. +23.6% vs TRT-LLM at BS=1. Closest industrial hit. L20 ≠ 4090. |
| AutoMegaKernel (2606.09682) | W8A16; naive W4A16 | RTX 5090 SM120 | TinyLlama-1.1B | Open. Int4 is lossy (22% token agreement). Validator is the idea. |
| AlpinDale qwen_megakernel | **bf16 only** | 5090 SM120 | Qwen3-0.6B | **1036 tok/s**. Consumer SM120 megakernel exists; no 4-bit. |
| Hazy / MPK / Kog / Fleet | bf16 | H100/B200/MI300X | 1B–70B | Datacenter. |

SM120 programming model, restated because people still confuse it with B200: **TMA yes, FP4 MMA yes, no wgmma, no tcgen05, no TMEM, no 2-CTA MMA.** Ampere-shaped + FP4 + TMA. FA-4 is SM100/103 only. gau-nernst’s 5090 FA rewrite (CuTe-DSL + TMA + warp-spec, Aug 11) now **beats SDPA-cuDNN**. That is the consumer attention ceiling to beat, not FA-4.

FlashInfer 0.6.17 SM12x fused-MoE is the serving path to depend on. BCG is a library now. TileLang 0.1.13: SM120 NVF4 ~1527 TFLOPS + Metal 4 MPP.

Metal: M5 tensor cores have been **silently off in llama.cpp since Nov 2025** (`#27473`, filed today). Prefill tables in the Aug-18 Apple section may be eating that tax. BaseRT-M5: up to 6.4× prefill vs llama.cpp with NAX kernels.

NInfer’s interesting systems bits: **PDL** to hide latency, **ReplaySSM** for GDN + spec under concurrency (vLLM does not fully support this yet — NInfer claims), closed artifact so every kernel is specialized. That is the “MVP one architecture + one GPU” the corpus recommended, executed.

---

## 9. MoE, memory, KV

### Expert cache in llama.cpp: inverted design, not merged

Every 2025 attempt that moved `MUL_MAT_ID` to the GPU put misses on a synchronous PCIe path. Metal slot-pool was 2× slower than vanilla at 97–99% hit. The design that works: **keep MUL_MAT_ID on CPU; thread 0 ships hit rows as one GPU matvec; misses stay on CPU.** Fill is decode-only.

Independent gains +25–72% TG on 2–4×3090 / 5090 class; **−31% on 1080 Ti**. Prefill can lose 14–66%. Spec composes if the *draft* is not on GPU. Not upstream.

### Unified Radix Cache (SGLang, Aug 11)

One token-keyed tree; components FULL / SWA / MAMBA own reuse/COW/eviction. Day-0 for Qwen3.8 (69 GDN + 23 GQA). Recurrent/GDN: copy-on-write before mutate; checkpoints at prefill-chunk boundaries. Cascade eviction. L3 later-round hits ~98% on DSv4-Flash. This is the generalized-state allocator the corpus asked for, **inside one engine, not as a portable abstraction**.

Escha’s measurements on the same hybrid: radix **off by default** because without spec it disables the overlap scheduler, and on GDN it only hits **exact complete** repeats. A 116k shared prefix + 4k new: **0 cached**. Growing agent loops re-prefill. Budget ~68 s at 120k on a 4090. The 80× number is exact-repeat only.

### Strata / HiSparse / OasisKV

Unchanged and more confirmed. Hierarchical KV is I/O-bound on **layout fragmentation**. Sparse decode’s wall is **HBM residency of the selectable set**. Speculation as a KV prefetch oracle (OasisKV) is the coupling to steal.

### Prefix-cache security

PROMPTPEEK / EarlyBird / InputSnatch / NDSS 2026 Shadow in the Cache. vLLM `cache_salt`; KVGov HMAC per principal. Cold/cached TTFT ratio ~0.22 is a real side channel. A single-user engine can ignore this. Anything that shares a radix tree across apps on one box cannot. Cheap: salt the first-block hash.

### TransMLA caveat the corpus under-weighted

Conversion **hurts speculative acceptance**. Beyond KV Reconstruction (2607.27269): rebuild the drafter after GQA→MLA. Ant Ling-2.5-1T shipped TransMLA in production.

---

## 10. Architectures the engine must absorb (config-verified Aug 21)

Local winners are still **GQA ± SWA ± Gated DeltaNet**. Frontier self-host is **MLA or GQA + a trained indexer**.

MTP presence, re-checked:

| Model | MTP? |
|---|---|
| Qwen3.8-27B | **yes**, `mtp_num_hidden_layers=1` |
| Qwen3.6-27B / 35B-A3B | yes |
| Qwen3-Next-80B | yes in weights, field often absent |
| DeepSeek V3.2 / V4 | yes |
| GLM-4.6 / 5.2 | yes |
| Nemotron 3 Super / 3.5 Lightning | yes |
| **Kimi K2 / K2.7** | **no** (`num_nextn_predict_layers: 0`). They serve DFlash. |
| gpt-oss | no |
| Llama 4 | no |
| Gemma 4 | **separate assistant checkpoints**, not a field |
| MiniMax M2 | config yes; **weights often dropped** |

MiniMax M2 **did** revert to full GQA. Paper + LMSYS “No Free Lunch” blog + live `attn_type_list` all 1s. Three serving blockers: low-precision recurrent state, prefix cache, speculation. M3 switched to **learned sparse (MSA)** over full GQA — the 2026 consensus: keep softmax, don’t let it see the whole prefix.

DeepSeek V4 is **CSA/HCA**, not classic MLA: compress-4× then DSA, or compress-128× dense, plus 128-token window, plus **mHC** (4× residual streams). That last one breaks `x = x + f(x)`.

Gemma 4 E2B/E4B: PLE + nested MLP + cross-layer KV share. gpt-oss: 1:1 SWA-128 / full GQA, head_dim=64, **no shared expert**.

---

## 11. Papers the Aug-18 sweep under-weighted or missed

If implementing five things from Jun–Aug 2026:

1. **Indexer subsystem** — LiteTopK, LongCat (contiguity + cross-layer), Vortex (programmable sparse IR).
2. **Hierarchical KV** — HiSparse + OasisKV + Strata layouts.
3. **Speculation controller** — Lossless but Not Free (ρ_M), AcceptMoE, Oilbird first, DFlash2 selector not a 78M GRU.
4. **RD-aware KV with a meter** — WitCert, not a default compressor. Alignment Collapse (2606.09864): safety subspace 10²–10³× more fragile than PPL. Qeios mixed-precision KV (Aug 17): PPL and GSM8K **disagree** on key precision.
5. **Heterogeneous state** — ReplaySSM (cache SSM *inputs*, not snapshots). Nemotron 3 Ultra / V4 / DART as test models.

Other misses: ExpertPlex (tile-granularity persistent MoE across P/D), SiFAR (sync-free All-Reduce inside megakernels — collectives become the floor after fusion), FlashPrefill V2 (paged + continuous batching + FP8, posted Aug 20), CAKE (agent kernel IR, 2.05× FlashKDA), UnionSparse (index traffic is the SpMM bottleneck at W4A4 + sparsity), FleetSieve (stop profiling when the remaining gap cannot change the decision).

Skip: most 2608 “new KV eviction” papers; another 2-bit PTQ with +N pp on Qwen3-4B and no kernel.

---

## 12. Gap status vs `00-synthesis.md` §7

| # | Gap | Aug 21 |
|---|---|---|
| 1 | RD-**latency** quant + dual-layout from one checkpoint | **Open.** Mix-Quant is two machines. APreQEL is post-hoc on small models. Dynamic 3.0 is quality, not kernel latency. llama.cpp `#26079` (today) is per-HW MMVQ→MMQ *crossover*, not a quantizer objective. |
| 2 | Megakernel × 4-bit × SM89/SM120 | **Nudged.** Ada-MK W4A16 on L20; AMK toys on 5090; AlpinDale bf16 0.6B. 7B+ Q4_K/EXL3/NVFP4 on 4090/5090: unclaimed. |
| 3 | Cost-model heterogeneous MoE scheduler | **Open.** `--fit` / `-ncmoe` / `-ehs` are heuristics. FreeToken is the paper. Expert-cache RFC is the prototype. Niche between llama.cpp and vLLM still empty. |
| 4 | Adaptive speculation as a control problem | **Nudged.** Adaptive MTP PR, DSpark confidence, Nightjar, D-cut, LibraSpec. Nobody ships ρ_M + union-aware γ + per-phase EWMA. |
| 5 | Transactional speculative KV + recurrent rollback | **SSM half closed in vLLM (ReplaySSM).** Attention is still “truncate blocks.” TransKV unreplicated. |
| 6 | Generalized state allocator | **Open as a portable abstraction.** SGLang Unified Radix is the in-engine version. |
| 7 | Quality-aware asymmetric KV tiering | **Evidence, not a product.** Qeios + Gemma-vs-Qwen KLD. No online sensitivity → migration → risk score. |
| 8 | Prefill on low-bandwidth devices | **Open as an engine thesis.** Spark proves it is software. M5 NAX is a hardware gift llama.cpp has been leaving on the table for nine months. |
| 9 | Predictive-coding unified predictor | **Open.** Zero papers. OasisKV/SparDA/AcceptMoE are unnamed instances. |
| 10 | Cross-engine eval (KLD + η + J/token) | **Open.** Fateev’s 4,800-task run is the *kind* of thing. qbench.py is still ExLlama-scoped. No 2026 vLLM vs SGLang vs llama.cpp vs ExLlama table with losing configs. |

Strike nothing from the closed list. Add to the watch list: ReplaySSM, DFlash2 selector, FreeToken, NInfer as an existence proof of specialized consumer CUDA, Escha as an existence proof of 2-bit quality with a kernel tax, prefix-cache salting, Mix-Quant phase-aware precision.

---

## 13. New ideas worth stealing (not in the Aug-18 bag, or newly sharp)

1. **Choosing is cheaper than predicting.** DFlash2’s 2M-param selector vs DSpark’s 78M GRU. Recall@16 vs Recall@1 is the headroom diagram for every parallel drafter.
2. **Cache SSM inputs, not state.** ReplaySSM. Rollback is a ring-buffer pointer. Spec on hybrids stopped being a net loss in vLLM.
3. **Speculation as a prefetch oracle** (OasisKV for KV, AcceptMoE for experts). One predictor, two budgets.
4. **`--spec-type` as a fallthrough list.** MTP then n-gram is a free win on code. The corpus said this; llama.cpp now ships it.
5. **Schedule IR + validator** (AutoMegaKernel) if you ever megakernelize. Hand-writing one 8B Q4 megakernel loses the retargeting war. Marlin/Machete/FlashInfer as Layer-1 ops.
6. **Pluggable decode** (TileRT / vLLM connector). Even at batch 1–4, prefill pool / decode kernel / KV wire format as a seam.
7. **`reasoning_effort` is a first-class serving knob**, not a chat-template curiosity. Fateev: 7.5× tokens for 0 extra points. Willison: Qwen3.8 `xhigh` is 21 min vs 2 min. Default it to `low` locally.
8. **Prefix-cache salt** as MVP if anything is multi-tenant, including two apps on one box.
9. **ESCHA_ROUTE-style occupancy flags.** Same kernels, 1.72× at batch 1 on Ampere if you pick the geometry that fills SMs. Decode on hybrid-SSM is not purely bandwidth-bound at M=1 — Escha’s 3090 vs 4090 (93% BW, 61% decode even with the fix) is the exhibit.
10. **Working-set streaming for models that do not fit RAM** (Siliang, BigMoeOnEdge). Does not overturn PCIe-vs-DDR math for batch-1 experts that *do* fit RAM. Is the right primitive when they don’t. llama.cpp mmap is not that primitive.
11. **J/token at thermal steady state** as a headline metric. PELM: speculation + DVFS + variable verify depth, 52% energy cut vs DVFS-only.
12. **Cross-model KV transfer** (2608.03893). Highest-variance. If it replicates on Qwen3.x, size-tiered local stacks change.

---

## 14. Controversies that affect a new engine

**llama.cpp vs vLLM is a fit argument, not a speed argument.** Inside 24 GB, vLLM c1→c8 scales 3.9–5.4×; llama.cpp `-np 8` only 1.2–1.9×. Past 24 GB, llama.cpp *runs* gpt-oss-120b on a 3090+128 GB; **vLLM OOMs at ~22 GB used regardless of `--gpu-memory-utilization`.** That wall is reproduced.

**Ollama’s non-batching is now a number.** Spark c=8: Ollama ~42 vs vLLM 116–313 aggregate. Continuous batching for a *single user with parallel tool calls* is still the empty niche between llama.cpp and vLLM. 3090 114 → 1000 and Spark 33 → 862 are the same curve.

**EXL3 vs GGUF as the NVIDIA quality path.** Dual-3060 2× decode will convert people this month. Portability people will not move. AMD people cannot. A new engine that is CUDA-first can take EXL3/NVFP4 seriously without abandoning GGUF import.

**Which speculator on Qwen3.8 this month.** Adaptive MTP (coding, dGPU), DFlash2 (short/medium structured, NVIDIA CUDA), native MTP (Apple, Vulkan, long ctx, TTFT). There is no single default. That *is* the autotuner thesis.

**Quantization dissent is about which metric lies.** Fateev (pass@1 tie at 4-bit), Qeios (PPL vs GSM8K on KV keys), 2606.25519 (CoT inflation), Gemma-4-MoE q8_0 KLD 0.377. Anyone ranking recipes by WikiText PPL is participating.

**Megakernel hype.** 25× vs HF PyTorch on 0.8B is not a result. Hazy’s 78% of H100 bandwidth on bf16 1B is still the number to beat, and it is still not 4-bit consumer.

---

## 15. Implications for *this* repo’s engine

The Aug-18 recommended stack and build order still stand. Updates that change priority, not direction:

1. **MVP model is now Qwen3.8-27B + one DeepSeek-style MoE**, not “a Llama.” Hybrid GDN state is no longer optional if you want the daily driver.
2. **Instrument `reasoning_effort` and tokens-emitted from day one.** Fateev’s result will otherwise make every quant look equal.
3. **Ship MTP + n-gram fallthrough before DFlash2.** Then a DFlash2-*shaped* selector, not a 78M sequential head. Kill switch on ρ_M, small-active MoE, Q4+creative, high batch.
4. **ReplaySSM-style input ring before a unified allocator.** State kinds have different rollback physics; don’t pretend they don’t.
5. **Do not start with a megakernel.** If you go there, the thesis is 4-bit quality on SM89/SM120, schedule IR as the design, FlashInfer/Marlin as Layer-1. NInfer is the existence proof that *specializing* beats *generalizing* on one GPU — and the warning that a closed artifact set is how you get 200 tok/s press.
6. **Gemma-vs-Qwen KV probe + `cache_salt` in MVP**, not v2.
7. **The two still-unclaimed, still-high-value papers:** (a) latency-in-the-quantizer + dual-layout from one checkpoint; (b) predictive-coding controller over tokens / KV blocks / experts / tool calls. DFlash2’s selector is a special case of (b).
8. **Eval harness is still the field’s missing public good.** Fateev + qbench.py + roofline η + J/token + published losing configs. If this engine does one non-kernel thing that matters, do that.

---

## 16. Source index (this overlay)

Primary artifacts and posts actually used. Not a bibliography of everything the agents saw. Prefer these URLs over agent paraphrase if you need to cite.

### X posts fetched by the parent (status IDs)

| ID | Author | When (UTC) | Used for |
|---|---|---|---|
| 2090318703992717486 | @superalesha | 2026-08-20 06:03 | 4,800-task 4-bit quality bench (thread fetch) |
| 2090827925419343904 | @Stellakjbk | 2026-08-21 15:46 | 5090 DFlash2 vs MTP A/B |
| 2090811917241643291 | @bountyAIhunter | 2026-08-21 14:43 | Escha-W2 vs Unsloth Q2 runtime |
| 2090420361158336713 | @_thomasip | 2026-08-20 12:47 | SWE-Bench ~130 vs GSM8K 210–251 |
| 2090485014844600359 | @Tech2Wild | 2026-08-20 17:04 | Dual 3090 drafter bake-off |
| 2090823527830110686 | @BullBoss5 | 2026-08-21 15:29 | Arc B70 12.5 → 65.5 |
| 2090740272153928102 | @laurent_zw | 2026-08-21 09:58 | Strix Halo 41 t/s DFlash2 |
| 2090813632049561746 / 2090318703992717486 | @IgyutaeI83275 / Fateev replies | 2026-08-21 | MTP3 109.5 t/s at 128K |
| 2088310045901557909 | @repojournal | 2026-08-14 | Daily engine commit digest (sample) |

Open as `https://x.com/i/status/<ID>`. X `Latest` scrape is recency-weighted; it is not a complete firehose.

**X / operators (same posts as URLs)**
- https://x.com/superalesha/status/2090318703992717486 — 67-hour 4-bit quality bench
- https://x.com/Stellakjbk/status/2090827925419343904 — 5090 DFlash2 vs MTP A/B
- https://x.com/bountyAIhunter/status/2090811917241643291 — Escha-W2 vs Unsloth Q2 runtime column
- https://x.com/_thomasip/status/2090420361158336713 — SWE-Bench ~130 tok/s vs GSM8K 210–251
- https://x.com/Tech2Wild/status/2090485014844600359 — dual 3090 drafter bake-off
- https://x.com/BullBoss5/status/2090823527830110686 — Arc B70 12.5 → 65.5
- https://x.com/laurent_zw/status/2090740272153928102 — Strix Halo 41 t/s DFlash2
- https://x.com/repojournal — daily engine commit digest

**DFlash 2 / Qwen3.8**
- https://inco.ai/blog/dflash2/ (2026-08-18)
- https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- https://github.com/ggml-org/llama.cpp/pull/27342
- https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/

**New engines**
- https://github.com/Neroued/ninfer
- https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2
- https://arxiv.org/abs/2608.16157 — FreeToken
- https://vllm.ai/blog/2026-07-14-vllm-tilert-pd

**Engines / systems**
- https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/ — BCG
- https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/
- https://www.lmsys.org/blog/2026-07-06-dspark-sglang/
- https://vllm.ai/blog/2026-03-24-mrv2
- https://github.com/vllm-project/vllm/releases/tag/v0.27.0
- https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17
- https://tridao.me/blog/2026/replayssm/

**Community**
- https://news.ycombinator.com/item?id=49267928 — llama.app
- https://www.reddit.com/r/LocalLLaMA/comments/1vqh5s7/exllamav3_vs_llamacpp_2x_3060/
- https://sleepingrobots.com/dreams/stop-using-ollama/
- https://www.storagereview.com/best/local-llm-tools (2026-08-19)
- https://simonwillison.net/2026/Aug/16/qwen-38-27b/

**Quant / eval**
- https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
- https://unsloth.ai/docs/basics/nvfp4
- https://arxiv.org/abs/2606.25519 — Quantization Inflates Reasoning
- https://arxiv.org/abs/2606.09864 — Alignment Collapse under KV quant
- https://www.qeios.com/read/RGD04F — mixed-precision KV, PPL vs tasks

**Gaps / kernels**
- https://arxiv.org/abs/2605.11581 — Ada-MK
- https://arxiv.org/abs/2606.09682 — AutoMegaKernel
- https://arxiv.org/abs/2607.17283 — Lossless but Not Free
- https://arxiv.org/abs/2608.02989 — AcceptMoE
- https://arxiv.org/abs/2608.03839 — Oilbird
- https://arxiv.org/abs/2608.08097 — OasisKV
- https://arxiv.org/abs/2608.07009 — HiSparse
- https://github.com/RightNow-AI/AutoMegaKernel
- https://github.com/AlpinDale/qwen_megakernel

**llama.cpp PRs/issues this window**
- https://github.com/ggml-org/llama.cpp/issues/27473 — M5 tensor cores off since Nov 2025
- https://github.com/ggml-org/llama.cpp/issues/23533 — SYCL MTP slower at 100% accept
- https://github.com/ggml-org/llama.cpp/pull/27210 — adaptive MTP
- https://github.com/ggml-org/llama.cpp/discussions/24528 — MoE expert cache RFC
- https://github.com/ggml-org/llama.cpp/issues/25224 — `--fit` MoE 42% regression

---

---

**End of overlay.** Compiled 2026-08-21 by Grok 4.6. Does not replace `research/00-synthesis.md`. Numbers were not reproduced on hardware we control. Re-fetch the URLs above before citing a tok/s figure in a paper or a purchase decision.
