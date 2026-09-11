# How People Actually Run Large LLMs on Consumer Hardware (2025-2026)

*A practitioner's survey for engine builders. Compiled 2026-08-18 from r/LocalLLaMA-adjacent community writeups, llama.cpp GitHub discussions, Unsloth docs, vendor benchmarks and independent blogs. Numbers are as-reported; treat single-source figures as indicative rather than authoritative.*

---

## 0. Summary

1. **The workload has shifted from dense to sparse MoE.** Almost every model people care about in 2026 is MoE with a small active fraction: gpt-oss-120b (117B/5.1B), Qwen3.6-35B-A3B, Qwen3-Coder-Next-80B-A3B, GLM-5.x (744B/40B), Kimi K2.x (1T/32B), DeepSeek V4-Flash (284B/13B) and V4-Pro (1.6T/49B), MiniMax M2/M3, Gemma 4 26B-A4B. Token generation speed tracks **active parameters × memory bandwidth of wherever the experts live**, not total parameters.
2. **The canonical consumer setup is heterogeneous**: attention + shared experts + embeddings on GPU, routed expert FFNs in system RAM, executed on CPU. `--n-cpu-moe` / `-ot "exps=CPU"` is the placement that matters most.
3. **Prefill is the pain point, not decode.** Unified-memory boxes (Strix Halo, DGX Spark, Macs) and CPU-offload rigs generate tokens acceptably, but time-to-first-token on a 32k-256k prompt is the bottleneck. Agentic/coding workloads amplify this because they re-prefill constantly. Prompt cache reuse and cheap incremental prefill are worth more than 10% decode wins.
4. **Memory bandwidth is the single best predictor** of decode t/s: RTX 5090 1792 GB/s, RTX PRO 6000 ~1792 GB/s, 3090 936 GB/s, M5 Max ~600 GB/s, M4 Max 546 GB/s, Strix Halo ~215 GB/s measured (256 GB/s theoretical), DGX Spark 273 GB/s, DDR5 desktop dual-channel ~80-100 GB/s, 12-channel EPYC DDR5-5600 ~500 GB/s.
5. **The engine ecosystem is fragmented and buggy at the seams.** ROCm/Vulkan/SYCL/CUDA each break differently, multimodal server support is half-finished, tensor parallelism in llama.cpp is nascent, and quality-of-quantization is a live research topic. There is real room for a new engine that treats hybrid CPU/GPU MoE + long-context prefill as the *primary* case rather than a bolt-on.

---

## 1. Hardware tiers

### 1.1 Capacity and bandwidth

| Class | Device | Memory | Bandwidth | Street price (2026) | Notes |
|---|---|---|---|---|---|
| Flagship consumer GPU | RTX 5090 | 32 GB GDDR7 | 1792 GB/s | ~$2000-2800 | Best single-card decode; 32 GB caps model size |
| Prosumer GPU | RTX PRO 6000 Blackwell | 96 GB GDDR7 | ~1792 GB/s | ~$8000+ | One card can run models ≤120B |
| Prev-gen flagship | RTX 4090 | 24 GB | 1008 GB/s | ~$1600 used | Still very common |
| Workhorse used card | RTX 3090 | 24 GB | 936 GB/s | ~$600-800 used | **Most-submitted GPU on community benchmark sites**; NVLink pairs |
| Mid consumer | RTX 5080 / 5070 Ti | 16 GB | 960 / 896 GB/s | ~$1000 / $750 | Fine for ≤27B dense at Q4 |
| Budget-VRAM | RTX 4060 Ti 16 GB / 5060 Ti 16 GB | 16 GB | 288 / 448 GB/s | ~$450 | Cheap VRAM, weak bandwidth |
| Cheap used compute | AMD MI50 32 GB | 32 GB HBM2 | 1024 GB/s | ~$250 | ~68% of a 3090's decode for 15% of price |
| Cheap used compute | AMD MI60 / MI100 | 32 GB HBM2 | 1024 / 1229 GB/s | $300-600 | MI100 pp is much better than MI50 |
| Cheap used compute | Tesla P40 24 GB | 24 GB GDDR5X | 347 GB/s | ~$300 (rising) | No usable FP16; llama.cpp only; increasingly deprecated |
| Intel | Arc Pro B60 24 GB / B70 32 GB | 24 / 32 GB | 456 GB/s | ~$600 / $900 | Cheap VRAM per dollar, immature software |
| APU / unified | Ryzen AI Max+ 395 "Strix Halo" 128 GB | 128 GB LPDDR5X-8000 | ~215 GB/s measured | $1500-2500 box | Framework Desktop, GMKtec EVO-X2, HP ZBook G1a, Corsair AI 300 |
| Unified + CUDA | NVIDIA DGX Spark (GB10) | 128 GB LPDDR5X | 273 GB/s | ~$4699 | Excellent prefill, mediocre decode |
| Apple | M4 Pro / M4 Max / M5 Max / M5 Ultra | 24-192 GB | 273 / 546 / ~600 / ~1000+ GB/s | $2000-8000 | Mac Studio M3 Ultra 512 GB @ 819 GB/s is the large-model option |
| CPU server | 2× EPYC 9554/9654, 24× DDR5-5600 | 768 GB-1.5 TB | ~400-500 GB/s aggregate | $6-12k used | The only sane way to run 671B-1.6T at home |
| CPU desktop | Ryzen/Core + 64-192 GB DDR5 | 64-192 GB | 80-110 GB/s | - | Only viable paired with a GPU for MoE offload |

### 1.2 llama.cpp reference bench (Llama-2 7B Q4_0, pp512 / tg128)

These come from the canonical llama.cpp hardware discussions and are the cleanest apples-to-apples set available.

| GPU | Backend | pp512 t/s | tg128 t/s | pp512 +FA | tg128 +FA |
|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell | CUDA | 14855 | 274.2 | 16619 | 281.1 |
| RTX 5090 | CUDA | 14073 | 290.0 | 14970 | 300.4 |
| RTX 4090 | CUDA | 11993 | 186.2 | 14771 | 189.0 |
| RTX 5080 | CUDA | 8297 | 182.0 | 9488 | 184.7 |
| RTX 3090 | CUDA | 5175 | 158.2 | 5560 | 161.9 |
| RTX 4060 Ti | CUDA | 3395 | 63.9 | 3803 | 64.0 |
| RX 9070 XT | Vulkan | 5036 | 137.1 | - | - |
| RX 9070 XT | ROCm | 5055 | 101.3 | - | - |
| RX 7900 XTX | Vulkan | 3532 | 191.3 | - | - |
| RX 7900 XTX | ROCm | 3552 | 167.1 | - | - |
| Intel Arc Pro B70 | Vulkan | 3379 | 112.0 | - | - |
| Intel Arc A770 | Vulkan | 1074 | 52.6 | - | - |
| AMD MI100 | ROCm | 2733 | 110.5 | - | - |
| AMD MI50 32 GB | ROCm | 1057 | 99.0 | - | - |
| RTX 3080 | Vulkan vs CUDA | 1706 vs 4499 | - | - | - |

Reading of the table:
- **Flash attention is a free 5-25% on prefill** on CUDA and essentially never hurts there. On some RDNA2 AMD cards it *reduces* pp512, a real reported regression.
- **Vulkan on NVIDIA costs ~2.6× prefill** vs CUDA. Vulkan on AMD is at parity-or-better with ROCm for prefill and often *better* for decode on RDNA3 (191 vs 167 t/s on the 7900 XTX). This is why "Vulkan first, ROCm if you must" is the practical AMD advice in 2026.
- MI50 at $250 for 32 GB HBM2 remains the cheapest decode-per-dollar on the used market; its prefill is ~20% of a 3090's, which is exactly the shape of the "cheap card, slow prefill" problem.

### 1.3 Apple Silicon (llama.cpp Metal, LLaMA-7B)

| Chip | GPU cores | BW GB/s | F16 pp / tg | Q8_0 pp / tg | Q4_0 pp / tg |
|---|---|---|---|---|---|
| M2 (base) | 10 | 100 | 201 / 6.7 | - | - |
| M1 Pro | 16 | 200 | 302 / 12.8 | 270 / 22.3 | 266 / 36.4 |
| M2 Pro | 19 | 200 | 384 / 13.1 | - | 341 / 38.9 |
| M3 Pro | 18 | 150 | - | 345 / 17.5 | - |
| M1 Max | 32 | 400 | 600 / 23.0 | 537 / 40.2 | 530 / 61.2 |
| M2 Max | 38 | 400 | 756 / 24.7 | - | 671 / 66.0 |
| M3 Max | 40 | 300-400 | 779 / 25.1 | - | 760 / 66.3 |
| M4 Pro | 20 | 273 | - | - | 440 / 50.7 |
| M4 Max | 40 | 546 | 923 / 31.6 | - | 886 / 83.1 |
| M1 Ultra | 64 | 800 | 1169 / 37.0 | 1043 / 59.9 | 1030 / 83.7 |
| M2 Ultra | 76 | 800 | 1402 / 41.0 | - | 1238 / 94.3 |

Modern-model figures (Q4, community-reported):

| Chip / RAM | Model | Engine | tok/s |
|---|---|---|---|
| M5 Max 128 GB | 7B Q4 | MLX | 95-110 |
| M4 Max 128 GB | 7B Q4 | MLX | 75-90 |
| M5 Max 64 GB | Qwen3.6-27B Q4_K_M | llama.cpp | ~30 |
| M4 Max | Qwen3.6-35B-A3B Q4 | Ollama/LM Studio | 40-50 |
| M4 Max | Qwen3.6-35B-A3B Q4 | MLX-LM | 45-55 |
| M5 Max | Llama 4 Scout (MoE) | MLX | ~32 |
| 128 GB Mac | DeepSeek V4-Flash IQ2 | llama.cpp | 27-34 |
| 192 GB+ Ultra | DeepSeek V4-Flash 2-4 bit | llama.cpp | 26-37 |
| M2 Ultra 192 GB | gpt-oss-20b MXFP4 | llama.cpp | pp2048 2191 / tg 116 |
| M4 Max 36 GB | gpt-oss-20b MXFP4 | llama.cpp | pp2048 1277 / tg 92 |

M5 Max's move to ~600 GB/s plus GPU-side matmul accelerators yields roughly **+28% over M4 Max**. Prefill on Macs remains the weak spot: an M5 Max prefills ~350-450 t/s on a 4K prompt at 7B, i.e. a 100k-token prompt is minutes of waiting.

### 1.4 AMD Strix Halo (Ryzen AI Max+ 395, 128 GB)

The most-discussed new class of machine. Real numbers (llama.cpp, Vulkan RADV unless noted):

| Model | Quant | pp t/s | tg t/s |
|---|---|---|---|
| Qwen3-4B | Q8_0 | 853 | 44.6 |
| Qwen3-30B-A3B | IQ4_XS | - | 100.0 |
| Qwen3-30B-A3B | Q4_K_M | 1142 | 86.1 |
| Qwen3-Coder-30B-A3B | Q4_K_S | - | 98.5 |
| Qwen3-Coder-Next-80B-A3B | Q4_K_M | 531 | 42.7 |
| gpt-oss-20b | MXFP4 | 1233 | 68.5 |
| gpt-oss-120b | MXFP4 | 340-470 | 40.1-55.6 |
| Nemotron-3 Nano 30B-A3B | MXFP4 | 112 | 61.5 |
| MiniMax M2.5 | Q3_K_M | 156 | 32.8 |
| 70B dense | Q4-Q8 | - | 5-15 |
| Strix Halo iGPU (Llama-2 7B Q4_0, ROCm) | Q4_0 | 417 | 16.1 |

Key community findings:
- **Generation scales with active params**: ~3B active → 61-86 t/s; 5.1B active → ~53 t/s; ~10B active → 29-33 t/s; 22B active → ~17 t/s. Clean bandwidth-bound behavior.
- **Prefill is the deal-breaker**: gpt-oss-120b at ~340 t/s pp vs DGX Spark's ~1700 t/s, a 5× gap that widens with context.
- Required kernel/BIOS tuning: `amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=31457280`, UMA frame buffer 512 MB. Without this you cannot address the full 128 GB pool.
- Backend picture is messy: **Vulkan RADV** is the stable default and often the tg champion; **AMDVLK** is faster still on some quants but has a 2 GiB buffer allocation limit; **ROCm 7.x nightlies** win prompt processing on some models. One report has Vulkan degrading noticeably past ~4K context, requiring a ROCm switch.
- Ollama vs raw llama.cpp on the same box: 38.7 t/s vs 68.5 t/s on gpt-oss-20b, a 43% Ollama overhead penalty in that test.
- Clustering: 4× Framework Desktop over llama.cpp RPC is the documented way to run Kimi K2.5 (1T) at home. Community verdict on RPC is harsh: **latency-bound, not bandwidth-bound**, no improvement from Ethernet → Thunderbolt, and "we really need to add tensor parallelism to llama.cpp." `-dio` (direct I/O) is required to avoid load hangs on large models.

### 1.5 DGX Spark (GB10)

| Model | Engine | Single-stream decode | Aggregate @ c=256 |
|---|---|---|---|
| gpt-oss-120b MXFP4 | vLLM | 33.5 t/s | 862 t/s |
| Nemotron Super 49B NVFP4 | vLLM/NIM | 5.8 t/s | 695 t/s |
| Nemotron Nano 9B v2 NVFP4 | vLLM | 26.2 t/s | ~156 t/s (plateaus at c=8) |
| Llama-3.1-70B dense | - | 2.7 t/s | - |
| Qwen2.5-72B / Llama-3.2-90B | - | ~4.6 t/s | - |
| gpt-oss-20b MXFP4 | - | 49.7 t/s (pp 2053 t/s) | - |

The consensus characterization: **a strong prefill machine and a mediocre generation machine**: 128 GB behind a 273 GB/s bus. It is good at concurrency (120× throughput scaling from c=1 to c=256 on the 49B), which single-stream reviews miss. If you are building an engine, note that the same box looks terrible or excellent depending purely on whether you batch.

### 1.6 Intel Arc Pro B60/B70

| Model | Quant | Backend | pp t/s | tg t/s |
|---|---|---|---|---|
| Qwen3.5-27B | Q4_K_M | Vulkan (1× B70) | 238 | 10.1 |
| Qwen3.5-27B | Q3_K_M | Vulkan | 230 | 9.3 |
| Qwen3.5-27B | UD-Q6_K_XL | Vulkan | 217 | 4.9 |
| Gemma 4 31B | Q4_K_M | SYCL (1× B70) | 205 | 18.6 |
| Gemma 4 31B | UD-Q4_K_XL | SYCL | 203 | 16.3 |
| Gemma 4 31B | UD-Q6_K_XL | SYCL | 191 | 12.5 |
| Qwen3.5-27B FP8 | - | vLLM (B70+B60) | 1010 | 13.3 |

Notes: SYCL is up to 2× Vulkan on dense models but has a **state-leak bug on hybrid Mamba/attention models** (second response answers the first question). Dual-GPU over M.2 PCIe 4.0 x4 adapters is *slower for generation than a single card* on every quant, an 18-50% loss. vLLM FP8 is 5× llama.cpp on prefill. Standard `Q4_K_M` beat Unsloth `UD-Q4_K_XL` on generation here (10.1 vs 9.3), a useful reminder that dynamic quants trade speed for quality.

### 1.7 CPU and hybrid CPU/GPU servers

| Setup | Model | Engine | Prefill | Decode |
|---|---|---|---|---|
| 2× EPYC 9554 (128c), 1.5 TB DDR5-5600, 1× RTX 6000 Ada 48 GB, 126 threads | DeepSeek-R1 Q4_K_M | KTransformers | ~150 t/s | ~11 t/s |
| same, 254 threads (both sockets) | DeepSeek-R1 Q4_K_M | KTransformers | ~138 t/s | ~7.9 t/s |
| 2× EPYC, 384 GB DDR5 | DeepSeek-R1 671B IQ4_XS | llama.cpp CPU-only | - | 5-8 t/s |
| 512 GB RAM + 1× RTX 4090 | DeepSeek-R1 671B | KTransformers | up to 286 t/s | - |
| 2× RTX 3090 + 64 GB DDR4/DDR5 | gpt-oss-120b Q4_K_M, `--n-cpu-moe 26 --split-mode row` | llama.cpp | - | 10-22 t/s |
| RTX 3080 Ti 12 GB + 128 GB RAM | gpt-oss-120b | llama.cpp | - | 18-22 t/s |
| RTX 3060 12 GB | gpt-oss-20b MXFP4, `-ncmoe 2` | llama.cpp | pp2048 2230 | 31 (64 initial) |
| Ryzen 7 5700X + 32 GB DDR4 + GPU | Qwen3-30B-A3B | llama.cpp `--n-cpu-moe` | - | ~60 |
| 24 GB GPU, `--n-cpu-moe 99` | Qwen 35B-A3B @ 102k ctx | llama.cpp | 350-400 t/s | ~15 t/s |
| RTX PRO 6000 96 GB | Qwen3-Coder-Next-80B-A3B Q4_K, 4k ctx | llama.cpp | 2920 t/s | 85.6 t/s |
| same, 256k ctx | | | 1472 t/s | 61.0 t/s |
| 2× RTX 3090/4090, Q4_K_XL, `--fit`, 32k ctx | Qwen3-Coder-Next | llama.cpp | 1165 t/s | 33.2 t/s |
| 2× RTX 3090, Q3_K, 32k ctx | Qwen3-Coder-Next | llama.cpp | 1076 t/s | 70.9 t/s |
| 2× RTX 3090, Q3_K, 256k ctx | Qwen3-Coder-Next | llama.cpp | 535 t/s | 47.7 t/s |
| 3× RTX 4090, `--fit` | gpt-oss-120b | llama.cpp | 5499 t/s | 182 t/s |

The dual-socket finding is important and counterintuitive: going from one socket to two **loses 20-40% of decode** because of cross-NUMA traffic. `numactl --interleave=all` or single-socket binding is standard advice.

---

## 2. Software: what people run and why

| Tool | Who uses it | Why |
|---|---|---|
| **llama.cpp** (`llama-server`, `llama-bench`, `llama-sweep-bench`) | the substrate for nearly everyone | Best GGUF support, best hybrid CPU/GPU offload, most backends (CUDA/Metal/Vulkan/ROCm/SYCL/OpenCL/CANN/Hexagon), fastest model support. `--fit` (new) auto-solves layer placement |
| **ik_llama.cpp** (+ `firecoperana` fork) | serious MoE/big-model people | SOTA quants (`IQ*_K`, trellis `IQ*_KT`), FlashMLA for DeepSeek, fused MoE, `-mla 3`, `-fmoe`, `-rtr`, `-amb`, `--merge-qkv`, `-gr` graph reuse, `-sm graph` tensor parallel, `-grt q8_0` quantized inter-GPU transfer, and `llama-sweep-bench` |
| **Ollama** | default for "just make it work" and for coding-assistant integrations (Continue etc.) | One-line pull/run, model registry, now with a native MLX runner on Apple Silicon (since ~Mar 2026). Costs 20-45% throughput vs raw llama.cpp in some tests; limited tool-calling |
| **LM Studio** | GUI-first users, Mac users | Visual model browser, GPU offload sliders, MLX + llama.cpp engines, good for exploration |
| **koboldcpp** | roleplay/creative-writing crowd | Batteries-included single binary, rich sampler set, context shifting, image/TTS |
| **ExLlamaV3 + TabbyAPI** | 24-96 GB NVIDIA multi-GPU users who want max quality per GB | EXL3 (QTIP-derived trellis codebook) beats EXL2 badly below 4 bpw; Llama-3.1-70B coherent at 1.6 bpw; tensor-parallel *and* expert-parallel; 2-8 bit cache quant; speculative decoding. **No ROCm** |
| **vLLM / SGLang** | anyone serving >1 concurrent request; Arc/Blackwell FP8 paths | PagedAttention, continuous batching, real tensor parallelism. On an 8×3090 server, vLLM TP at 50 concurrent requests ≈ 800 t/s vs ~1 t/s for llama.cpp layer-split CPU-offload on DeepSeek-V2.5 236B BF16 |
| **KTransformers** | DeepSeek/Kimi/GLM at home on big-RAM boxes | AMX-optimized CPU expert kernels + GPU attention; claims up to 28× prefill and 3× decode vs llama.cpp. v0.6.3 (Jun 2026) supports DeepSeek-V3/R1/V4, Kimi K2/K2.5, GLM-5/5.2, Qwen3-MoE, Qwen3-Next, MiniMax |
| **MLX / MLX-LM** | Mac | 10-30% over llama.cpp on supported models, gap narrowing; model coverage lags |
| **oobabooga / Jan / LocalAI / GPT4All** | long tail | oobabooga as multi-backend playground; Jan for offline/no-telemetry; LocalAI for broadest format+modality coverage |
| **llmkube / Lemonade / infcore** | newer | k8s-native serving; AMD NPU path; OpenAI gateway with auth/RBAC merged into llama.cpp |

The typical decision tree people describe: *single user, weird hardware, newest model* → llama.cpp. *Single user, all-NVIDIA, want the best quality at a given VRAM* → ExLlamaV3/TabbyAPI. *Many users or agents* → vLLM/SGLang. *DeepSeek-class model with 512 GB RAM* → KTransformers or ik_llama.cpp. *Mac* → MLX or llama.cpp. *Want a product, not flags* → Ollama/LM Studio.

---

## 3. Techniques people actually use

### 3.1 Quantization selection

Rules of thumb that recur:

| VRAM/RAM budget vs model | Choice |
|---|---|
| Model fits comfortably | `Q6_K` / `Q8_0`, or EXL3 5-6 bpw |
| Model just barely fits | `Q4_K_M`-the universal default |
| Need to squeeze | `IQ4_XS`, `UD-Q4_K_XL`, EXL3 4.0 bpw |
| Aggressive | `UD-Q3_K_XL`, `IQ3_S`, EXL3 3.0-3.5 bpw ("closes the gap with FP16", 70B usable on one 24 GB card) |
| Desperate (huge MoE) | Unsloth `UD-Q2_K_XL` / 1-bit dynamic; ik_llama `IQ2_K`/`IQ2_KL`/trellis `IQ2_KT` |

**Unsloth Dynamic 2.0 (UD)** quants are the community default for big MoEs: per-layer selection of quant type to minimize KL divergence rather than uniform quantization. Reported to beat imatrix and QAT on MMLU and KLD. Concrete Kimi K2.7 Code figures: UD-Q2_K_XL 339 GB / PPL 2.4131, UD-Q4_K_XL 584 GB / PPL 1.8420, UD-Q8_K_XL 595 GB / PPL 1.8419 ("truly lossless"). Caveat from the Arc benchmarks: UD variants can be *slower* than plain `Q4_K_M` at similar size.

Native low-precision formats matter now too: **MXFP4** (gpt-oss, Nemotron) and **NVFP4** (Blackwell W4A4): Qwen3.6-27B NVFP4 decodes 125.9 t/s and 35B-A3B NVFP4 295.2 t/s on cute-DSL backends, ~2.5× the GGUF path on the same card.

### 3.2 KV cache quantization

`-ctk q8_0 -ctv q8_0` halves cache; `q4_0` quarters it. The 2026 measured picture is more nuanced than the old "q8 is free" folklore:

| Model | q8_0 KV KLD | q4_0 KV KLD |
|---|---|---|
| Qwen3.6-27B dense | <0.04 | 0.087-0.117 |
| Qwen3.6-35B-A3B MoE | <0.04 | 0.087-0.117 |
| Gemma 4 31B dense | 0.108 | - |
| Gemma 4 26B-A4B MoE | 0.377 | 1.088 |

So: **cache quantization sensitivity is architecture-specific**, and MoE routers appear to amplify it (Gemma's MoE is 3.5× worse than its dense sibling at the same cache precision). Asymmetric `--cache-type-k q4_0 --cache-type-v q8_0` is measurably better than symmetric q4 in most cases. GQA/MQA models tolerate it better. Quantized KV *requires* flash attention to not be slower than fp16 KV, because the kernels are fused.

### 3.3 MoE expert offload: the central trick

The strategy, stated cleanly: **VRAM gets attention, embeddings, norms, shared experts, and any routed experts that fit. RAM gets the sparse `ffn_*_exps` tensors.** Attention tensors are small, hot, and love the GPU; expert FFNs are huge, cold, and are the cheapest thing to exile.

```bash
llama-server -m model.gguf \
  -ngl 999 \                       # all layers to GPU first
  --n-cpu-moe 32 \                 # then pull N layers' experts back to CPU
  -c 32768 -fa 1 \
  -b 4096 -ub 4096 \               # large batches matter for MoE prefill
  -ctk q8_0 -ctv q8_0 \
  --threads 8 --no-mmap
```

Equivalent/finer-grained forms:
- `-ot "exps=CPU"` or `--cpu-moe`, all routed experts to CPU
- `-ot "\.ffn_.*_exps\.=CPU"`, universal MoE regex
- `-ot "blk\.(?:[0-9]|[1-7][0-9]|[8][0-7])\.ffn_.*_exps\.=CPU"`-layers 0-87 to CPU, rest stay on GPU
- `-ot "blk\.([0-9]|1[0-9])\.=CUDA0,blk\.(2[0-9]|3[0-9])\.=CUDA1,exps=CPU"`-multi-GPU + CPU experts
- `-mg N`, set the primary GPU for offload ops

Gotchas people repeatedly hit: `--n-cpu-moe` **counts down from the highest layer**, not up from zero. It only helps MoE, on a dense 70B there is no cold bulk to exile. Starting values: 12 GB card → 32; 16 GB → 20; 24 GB → often 0; 120B-class MoE → 40+. Tune by lowering until OOM, then back off one.

`GGML_OP_OFFLOAD_MIN_BATCH` overrides llama.cpp's default 32-token threshold for shipping a prefill op to the GPU. ik_llama.cpp already scales this threshold as `32 × total_experts / active_experts`, which is one of the main reasons it beats mainline on MoE prefill.

The new **`--fit`** machinery (on by default) does virtual test allocations and solves placement automatically, shrinking context first, then moving tensors out of VRAM, preferring to evict sparse MoE tensors over dense ones, and allowing single layers to straddle devices. Reported to reach 85-90% VRAM utilization; fitting costs 1.5-20 s depending on GPU count. This is the feature that finally made multi-GPU + MoE tractable for non-experts.

### 3.4 Speculative decoding

2026 flag names (renamed in the CLI rework): `--spec-draft-n-max N`, `--spec-draft-n-min N`, `--spec-type draft-mtp`, `-ngld 99` (draft fully on GPU).

| Pair | Hardware | Baseline | Speculative |
|---|---|---|---|
| Qwen3.6-27B + built-in MTP | A10G 24 GB | 25 t/s | 45 t/s (+78%) |
| Qwen3.6-27B + MTP | Apple Silicon | ~7 t/s | ~16 t/s |
| Gemma 4 + MTP | Apple Silicon MLX | 1.0× | ~1.9× |
| Llama-3.1-8B + Llama-3.2-1B draft | GPU | 1.0× | 1.83× at draft len 5 |
| Qwen2.5-14B + 0.5B draft (coding) | GPU | 1.0× | up to 2.5× |
| Qwen3.6-35B-A3B, 19 configs | RTX 3090 | 135.7 t/s | best 131.1 (**3% slower**), worst −15% |

The key lesson for an engine designer: **speculative decoding helps dense models and hurts small-active MoE models.** When decode is already 135 t/s because only 3B params are active, the draft's cost is not amortized, even at ~100% acceptance. Acceptance below ~60-70% is a net loss. Draft VRAM that pushes the target into CPU offload is catastrophic. The industry answer in 2026 is **built-in MTP heads** shipped with the model (Qwen3.6 MTP variants, Gemma 4 MTP drafters down to 78M for the E2B): 1.4-2.2× with guaranteed identical output distribution. An engine that supports MTP natively, and *auto-disables* speculation when measured acceptance × active-param ratio makes it unprofitable, would be strictly better than what exists.

### 3.5 Batching, threads, NUMA, memory mapping

- `-b 4096 -ub 4096` is the standard MoE prefill recommendation; `-ub 2048` for dense. Larger ubatch trades VRAM for prefill throughput.
- `--threads` = physical cores, not SMT threads. On hybrid Intel, pin to P-cores.
- Dual-socket: bind to one socket, or `numactl --interleave=all`. Naïve dual-socket costs 20-40% decode.
- `--no-mmap` is recommended when the model fits in RAM (avoids page-fault stalls on first pass, avoids double-counting). `--mlock` to prevent eviction. `-dio`/`--no-direct-io` matters for RPC and huge models; one report showed 14 vs 12 t/s on Kimi K2.5 from `--no-direct-io` alone.
- `--no-kv-offload` keeps the KV cache in system RAM, a last-resort context extender.

### 3.6 Multi-GPU

Three modes in practice:
1. **Layer split** (llama.cpp default `-sm layer`): trivially simple, no inter-GPU traffic, but sequential: no latency win, just capacity.
2. **Row split** (`--split-mode row --tensor-split a,b`): helps decode on well-connected pairs; commonly used on dual 3090s.
3. **True tensor parallel**, vLLM/SGLang/ExLlamaV3, and increasingly llama.cpp (an internal CUDA AllReduce kernel landed in 2026, single-phase with pipelined D2H, currently up to 2 GPUs; ik_llama has `-sm graph`).

PCIe matters far more than people expect. The Intel Arc dual-GPU-over-M.2-x4 result (dual *slower* than single for generation) and the Strix Halo RPC results (latency-bound, Thunderbolt no better than Ethernet) both say the same thing: **inter-device communication is the limiting factor for consumer multi-device rigs**, and the naive RPC/pipeline approaches waste it.

Power: 3090s are routinely power-limited to 250-280 W with <5% throughput loss (decode is bandwidth-bound), which is what makes 4-8 card rigs thermally and electrically feasible. Undervolting is standard on 4090/5090.

### 3.7 Prompt caching and context

- llama.cpp server slot save/restore + `--ctx-checkpoints N` (default 32; the Qwen3-Next-on-Mac writeup recommends **128** for multi-session stability).
- Measured impact is enormous: Qwen3-Next-80B on a 128 GB Mac, 60k-token prompt: **~84.2 s cold TTFT vs ~1.0 s hot**, an ~80× swing. For interleaved multi-slot workloads at 100k, ~2×.
- `--swa-full` bounds KV for sliding-window attention layers.
- Modern hybrid-attention models make long context cheap: Qwen3-Next's Gated-DeltaNet + SWA design costs ~25 GB of KV for 1M tokens, ~4× less than a pure transformer, and holds >1400 t/s prefill at 256k. Qwen3-Coder-Next's VRAM delta from 4k → 256k context is only ~7 GB at Q4.
- Broken prompt caching is a named community frustration ("Qwen models in llama.cpp" specifically).

---

## 4. Models people run (mid-2026)

*Version numbers move fast; this is an August 2026 snapshot and several entries come from secondary sources.*

| Model | Total / active | License | Where it runs |
|---|---|---|---|
| Gemma 4 E2B / E4B | 2.3B / ~4B eff | Gemma | Phones, 8 GB laptops; 261 t/s on M5 Max |
| Gemma 4 12B | 12B dense | Apache 2.0 | 12-16 GB VRAM, 21-50 t/s |
| Gemma 4 26B-A4B | 26B / 4B MoE | Gemma | 16 GB; ~40 t/s on 4070 Ti |
| Gemma 4 31B | 31B dense | Gemma | 24 GB @ Q4 (~18 GB); multimodal |
| Qwen3.6-27B | 27B dense + vision | Apache 2.0 | **The 24 GB default.** ~15-17 GB Q4, 262K ctx (1M YaRN), 77.2 SWE-bench V |
| Qwen3.6-35B-A3B | 35B / 3B MoE | Apache 2.0 | 16-24 GB; 220 t/s GGUF, 295 t/s NVFP4; Gated DeltaNet + 256 experts (8+1 active) |
| Qwen3-Coder-Next-80B-A3B | 80B / 3B | Apache 2.0 | 35-47 GB Q4; 2×3090 at 33-71 t/s; Strix Halo ~37-43 t/s |
| gpt-oss-20b | 21B / 3.6B, MXFP4 | Apache 2.0 | 16 GB; 162 t/s on 3090, 68 t/s on Strix Halo |
| gpt-oss-120b | 117B / 5.1B, MXFP4 | Apache 2.0 | **The MoE-offload poster child.** 59 GB; 10-22 t/s on 2×3090+64 GB, 40-56 t/s on Strix Halo, 33.5 t/s on DGX Spark, 182 t/s on 3×4090 |
| GLM-5 / 5.1 / 5.2 | ~744B / 40B | MIT | 128 GB+ / Mac Studio 512 GB / EPYC. 1M ctx. GLM-5.2 ranked #1 open-weights in the surveyed AA Intelligence Index ranking |
| Kimi K2.5 / K2.6 / K2.7 Code | 1T / 32B | Modified MIT | 310-350 GB at UD 1-2 bit; 4×Strix Halo cluster or EPYC box |
| DeepSeek V4-Flash | 284B / 13B | MIT | ~142-150 GB Q4; single 80 GB GPU quantized, or 2×48 GB; 27-37 t/s on 128-192 GB Macs |
| DeepSeek V4-Pro | 1.6T / 49B | MIT | 800 GB+; only big EPYC/1.5 TB boxes at 2-bit |
| MiniMax M2.x / M3 | MoE | - | M3 needs ~143 GB+; M2.5 Q3_K_M runs 32.8 t/s on Strix Halo |
| Nemotron 3 Nano/Super | 30B-A3B / 120B-A12B, NVFP4 | NVIDIA | Nano 61.5 t/s Strix Halo; Super has kernel bugs on GB10 |
| Llama 4 Scout | MoE, 10M ctx | Llama | ~32 t/s MLX on M5 Max |
| Qwen2.5-Coder 7B/14B/32B | dense | Apache 2.0 | Still the FIM/autocomplete pick-newer Qwen has no FIM |

**The structural point for engine design:** the models that people *want* (GLM-5.2, Kimi K2.x, DeepSeek V4) are 300 GB-1.6 TB, sparse, and long-context. The models that *fit* are 3B-active MoEs. Both cases reward the same engine properties: excellent sparse-expert scheduling across a memory hierarchy, cheap long-context attention, and prefill that doesn't fall over.

---

## 5. Pain points, bugs, and gaps

From llama.cpp weekly issue digests, forum threads and benchmark writeups:

**Performance**
- Prompt processing on CPU / Apple / AMD is the #1 complaint. "The prompt processing still isn't as quick as vLLM." CPU-only inference is described as unusable for coding assistants and agent loops precisely because of re-prefill.
- The VRAM cliff: "why does local AI fall off a cliff once VRAM runs out", mixed CPU/GPU is a discontinuity, not a gradient, and users report stalls as experts swap.
- Reported 2026 regressions: −7-9% on CUDA RTX 5070, −15-20% MTP throughput on Windows, −25-30% CPU decode with OpenBLAS, ROCm "decreased performance because of input layers on CPU."
- FA hurts prefill on some RDNA2 cards; RX 9070 non-XT beating 9070 XT in some configs.

**Correctness / stability**
- DeepSeek V4: assertion failures at large context, tensor reshape mismatches across multiple backends.
- Qwen3-VL: multimodal embedding broken from missing media markers.
- Gemma 4 official GGUFs aborted on duplicate vocab tokens.
- SYCL: segfaults with `-DGGML_SYCL_DEVICE_ARCH=xe2`; state-leak on hybrid models.
- Vulkan: crashes on AMD 8840U / Ryzen AI after extended token processing.
- VRAM leaks with CUDA Graphs; host memory exhaustion during MoE offload; KV cache reclamation failures.
- vLLM KV-cache allocation bugs affecting context limits; "Gemma 4 26B crashes roughly twice weekly."
- Security: llama-server web UI instruction injection via query parameters.

**Feature gaps**
- **Multimodal on the server is still WIP.** CLI is recommended over HTTP for reliable vision/audio. mtmd now covers images + audio (Qwen3-TTS, Ultravox, Voxtral, Qwen2.5-Omni) but server integration lags.
- **No general tensor parallelism in llama.cpp**, the AllReduce work is capped at 2 GPUs; RPC is latency-bound and community consensus is that it is the wrong abstraction.
- **Tool calling / structured output** is uneven: vLLM has production-grade parallel function calling; Ollama's is described as limited; grammar/JSON-schema enforcement quality varies by engine.
- **ExLlamaV3 has no ROCm.** Anyone on AMD is locked out of the best low-bpw quality format.
- **MLX model coverage lags**; MoE routers reportedly pick wrong experts at Q4 because "the router operates on weight distributions."
- Sampler semantics differ per engine; Unsloth ships per-model sampler recipes (e.g. Qwen3.6 thinking: `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`; non-thinking: `--temp 0.7 --top-p 0.8 --presence-penalty 1.5`) because defaults produce visibly worse output.
- Model support lag: new architectures land in forks (ik_llama.cpp, bati.cpp) weeks before mainline.

**Ergonomics**
- Setup difficulty is a recurring complaint: "most guides don't clarify what to install without abandoning the project after three hours."
- Model churn every 2-3 months makes hardware/infra investment feel risky.
- Backend selection requires downloading different binaries (HIP vs Vulkan); no runtime backend switching.

---

## 6. Benchmark aggregators and leaderboards

| Site | What it does |
|---|---|
| **llama.cpp GitHub discussions #4167 (Apple), #15013 (CUDA), #10879 (Vulkan), #15021 (ROCm)** | The canonical crowd-sourced `llama-bench` tables; Llama-2 7B Q4_0 pp512/tg128 as the common yardstick |
| **localmaxxing.com** | Community submissions; 611 RTX 3090 runs, 543 dual-3060 runs; most-tested models Qwen3.6-35B-A3B, Qwen3.6-27B |
| **local-bench.ai** | Judge-free quality + quant + VRAM + speed with reproducible run receipts |
| **localscore.ai** | Tiny (1B) / Small (8B) / Medium (14B) tiers across GPU, CPU+GPU, CPU-only; searchable by GPU |
| **LocalBench (companionintelligence/Local-Bench)** | Benchmarks your own Ollama / Strix Halo llama.cpp, ranks 40+ models against the Artificial Analysis Intelligence Index |
| **llmcheck.net** | Apple Silicon M1-M5, standardized 256-in/512-out Q4_K_M methodology |
| **kyuz0.github.io/amd-strix-halo-toolboxes** | Strix Halo backend grid (RADV / AMDVLK / ROCm 6.4.4 / 7.14 / TheRock nightlies) |
| **visorcraft/strix-halo-llm-perf** | Strix Halo single-host and RPC-cluster results |
| **ktransformers.net/en/benchmarks** | KTransformers vs llama.cpp prefill/decode leaderboard |
| **OpenBenchmarking.org llama.cpp suite** | Automated PTS runs (5090 159 / 4090 101 / 3090 91 index) |
| **Puget Systems Labs** | Controlled consumer-GPU and hybrid CPU/GPU studies |

---

## 7. What a new research inference engine could do better

Ranked by how much community pain it would remove:

1. **Make hybrid CPU/GPU MoE a first-class scheduler, not a flag.** Today users hand-tune `--n-cpu-moe`, `-ot` regexes, `-b/-ub`, and `GGML_OP_OFFLOAD_MIN_BATCH` by trial and error, and `--fit` only just started automating placement. An engine with a real cost model, per-tensor bytes, per-device bandwidth, PCIe cost, expert activation frequency measured online, could beat hand-tuning and eliminate the largest source of "why is my setup slow" threads.
2. **Attack prefill on low-bandwidth devices.** This is where every non-NVIDIA platform loses (Strix Halo 340 t/s vs Spark 1700 t/s on the same model; Macs at minutes-per-100k-prompt). Chunked prefill, aggressive prefix-cache reuse across sessions, disk-backed KV, and prefill-on-GPU-while-decode-on-CPU pipelining all matter more than decode micro-optimizations.
3. **Persistent prompt caching.** The 84 s → 1.0 s number is the largest measured win in this survey and it is currently fragile and model-dependent.
4. **Real tensor/expert parallelism over cheap interconnects.** Both the Intel dual-GPU-over-x4 result and the Strix Halo RPC result show consumer multi-device is latency-bound and current approaches waste it. Expert-parallel placement (each device owns a disjoint expert set, only routed tokens cross the link) maps far better onto MoE than layer-pipelining does.
5. **Adaptive speculation.** Support MTP heads natively; measure acceptance online; auto-disable when active-param ratio makes drafting unprofitable. The 3090/Qwen3.6-35B-A3B result (19 configs, all slower than baseline) is exactly the failure mode a self-tuning engine would avoid.
6. **Per-architecture KV-cache precision policy.** Given Gemma-4-MoE at q8_0 KLD 0.377 vs Qwen3.6 at <0.04, a static "q8 is fine" default is wrong. Measure KLD per model at load or ship a policy table; support asymmetric K/V precision by default.
7. **Backend uniformity.** One binary, runtime backend selection, and a conformance test suite that catches the class of bugs currently shipping (SYCL state leak, FA regressions on RDNA2, VRAM leaks with CUDA graphs, ROCm input-layers-on-CPU).
8. **Server-grade multimodal and structured output from day one**, the two gaps where llama.cpp explicitly tells users to fall back to the CLI or to another engine.
9. **Batching that helps single users too.** The DGX Spark data (33.5 → 862 t/s from c=1 to c=256) shows remaining headroom. Agentic workloads issue many parallel small requests; an engine that continuously batches *and* offloads MoE experts would occupy an empty niche between llama.cpp and vLLM.

---

## Sources

**llama.cpp / GitHub**
- Performance of llama.cpp on Apple Silicon. https://github.com/ggml-org/llama.cpp/discussions/4167
- Performance of llama.cpp on Nvidia CUDA. https://github.com/ggml-org/llama.cpp/discussions/15013
- Performance of llama.cpp with Vulkan. https://github.com/ggml-org/llama.cpp/discussions/10879
- Performance of llama.cpp on AMD ROCm (HIP). https://github.com/ggml-org/llama.cpp/discussions/15021
- Guide: running gpt-oss with llama.cpp. https://github.com/ggml-org/llama.cpp/discussions/15396
- Automation for GPU layers, tensor split, tensor overrides (`--fit`). https://github.com/ggml-org/llama.cpp/discussions/18049
- Speculative decoding potential for consumer GPUs. https://github.com/ggml-org/llama.cpp/discussions/10466
- MoE offload to a second (slower) GPU. https://github.com/ggml-org/llama.cpp/discussions/22183
- Optimize my llama.cpp. https://github.com/ggml-org/llama.cpp/discussions/21112
- 4-bit KV cache. https://github.com/ggml-org/llama.cpp/discussions/5932
- TurboQuant, extreme KV cache quantization. https://github.com/ggml-org/llama.cpp/discussions/20969
- Inference DeepSeek-V3 671B on CPU only. https://github.com/ggml-org/llama.cpp/discussions/11765
- mtmd multimodal README. https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd
- Weekly GitHub report for llama.cpp (Jul 13-20, 2026). https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-llamacpp-july-13-2026-4011/
- Weekly GitHub report for llama.cpp (May 4-11, 2026). https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-llamacpp-may-04-2026-may/
- FOSDEM 2026: multimodal support in llama.cpp. https://fosdem.org/2026/schedule/event/LRZJEH-llama-cpp-multimodal/

**ik_llama.cpp / ExLlama / KTransformers**
- ik_llama.cpp README. https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md
- ik_llama.cpp docs, hybrid CPU/GPU. https://ikawrakow-ik_llama-cpp.mintlify.app/inference/hybrid-cpu-gpu
- ik_llama.cpp CPU performance comparison. https://github.com/ikawrakow/ik_llama.cpp/discussions/164
- ExLlamaV3. https://github.com/turboderp-org/exllamav3 and https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md
- KTransformers. https://github.com/kvcache-ai/ktransformers , benchmarks https://ktransformers.net/en/benchmarks , SOSP'25 paper https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf

**Guides / tricks**
- Performant local MoE CPU inference with GPU acceleration in llama.cpp (Doctor-Shotgun, HF blog). https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
- Companion gist. https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0
- llama.cpp `--n-cpu-moe` guide. https://aliteq.com/run-big-moe-model-small-gpu-n-cpu-moe-guide
- Speculative decoding for local LLMs in 2026. https://runaihome.com/blog/speculative-decoding-llama-cpp-local-llm-setup-2026/
- KV cache quantization benchmark (Gemma 4 / Qwen 3.6 KLD). https://localbench.substack.com/p/kv-cache-quantization-benchmark
- Unsloth Dynamic 2.0 GGUFs. https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs and https://unsloth.ai/blog/dynamic-v2
- Unsloth: Qwen3.6 how to run locally. https://unsloth.ai/docs/models/qwen3.6
- Unsloth: Kimi K2.7 Code how to run locally. https://unsloth.ai/docs/models/kimi-k2.7-code
- Unsloth: DeepSeek-V4 how to run locally. https://unsloth.ai/docs/models/deepseek-v4
- Optimizing gpt-oss-120b local inference. https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/
- Running gpt-oss-120b on dual RTX 3090s. https://llmgarage.ai/gpt-oss-120b-dual-3090/
- Running GPT-OSS 120B on an RTX 3080 Ti 12 GB. https://github.crookster.org/running-gpt-oss-120b-on-rtx-3080-ti-12-gb-at-home/

**Hardware**
- AMD Strix Halo backend benchmark grid. https://kyuz0.github.io/amd-strix-halo-toolboxes/ and https://github.com/kyuz0/amd-strix-halo-toolboxes
- strix-halo-llm-perf. https://github.com/visorcraft/strix-halo-llm-perf
- Running local LLMs on a Strix Halo laptop. https://www.bogdanvarlamov.com/blog/local-llms-strix-halo/
- Ryzen AI Max+ 395 for local LLMs 2026. https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/
- AMD Ryzen AI Max+ 395 review (2026). https://www.hashtechwave.com/amd-strix-halo-review/
- AMD: how to run a one-trillion-parameter LLM locally (4× Framework cluster, Kimi K2.5). https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html
- Two-node Strix Halo cluster with llama.cpp RPC. https://community.frame.work/t/building-a-two-node-amd-strix-halo-cluster-for-llms-with-llama-cpp-rpc-minimax-m2-glm-4-6/77583
- Jeff Geerling: clustering four Framework mainboards. https://www.jeffgeerling.com/blog/2025/i-clustered-four-framework-mainboards-test-huge-llms/
- DGX Spark concurrency benchmark. https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/
- DGX Spark vs Strix Halo vs Mac Studio. https://www.compute-market.com/blog/dgx-spark-vs-strix-halo-local-ai-2026
- Intel Arc Pro B60 + B70 LLM benchmarks. https://bentech.substack.com/p/intel-arc-pro-b60-b70-llm-benchmarks
- vLLM on Intel Arc Pro B-series. https://vllm.ai/blog/2025-11-11-intel-arc-pro-b
- AMD MI50 32 GB: best AI card for beginners?. http://wtarreau.blogspot.com/2025/12/amd-radeon-instinct-mi50-32gb-best-ai-card.html
- llama-bench: Llama 3.1 8B on MI50 32 GB. https://ahelpme.com/ai/llamacpp-ai/llama-bench-the-llama-3-1-8b-and-amd-radeon-instinct-mi50-32gb/
- Puget Systems: LLM inference consumer GPU performance. https://www.pugetsystems.com/labs/articles/llm-inference-consumer-gpu-performance/
- Puget Systems: exploring hybrid CPU/GPU LLM inference. https://www.pugetsystems.com/labs/hpc/exploring-hybrid-cpu-gpu-llm-inference/
- What to buy for local LLMs (April 2026). https://julsimon.medium.com/what-to-buy-for-local-llms-april-2026-a4946a381a6a
- Local LLM tokens-per-second benchmarks 2026. https://presenc.ai/research/local-llm-tokens-per-second-benchmarks-2026
- Apple Silicon LLM benchmarks 2026 (M1-M5). https://llmcheck.net/benchmarks
- Best local LLMs for Mac in 2026. https://insiderllm.com/guides/best-local-llms-mac-2026/
- Stop using llama.cpp on multi-GPU rigs: the case for vLLM and tensor parallelism. https://thinksmart.life/research/posts/inference-engines-multi-gpu-llama-cpp-vllm/

**Models**
- Qwen3.6-35B-A3B. https://huggingface.co/Qwen/Qwen3.6-35B-A3B ; Qwen3.6-27B. https://huggingface.co/Qwen/Qwen3.6-27B
- Qwen3-Next-80B on 128 GB Apple Silicon (real-world). https://github.com/QwenLM/Qwen3.6/discussions/139
- Qwen3-Coder-Next hardware requirements. https://www.hardware-corner.net/qwen3-coder-next-hardware-requirements/
- DeepSeek-V4-Pro. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro ; V4-Flash. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- GLM-5 (mlabonne, HF blog). https://huggingface.co/blog/mlabonne/glm-5
- Kimi K2.5 GGUF. https://huggingface.co/unsloth/Kimi-K2.5-GGUF
- Local LLM 2026: every major model release + Ollama status. https://www.promptquorum.com/local-llms/local-llm-model-updates-2026
- Best local LLMs (August 2026) by VRAM tier. https://benchlm.ai/best/local-llm
- Best local coding models ranked by VRAM tier (2026). https://insiderllm.com/guides/best-local-coding-models-2026/
- Best local LLM 2026: what 300+ community builders actually run. https://bigguyonstuff.com/best-local-llm-2026-megathread-synthesis/

**Aggregators / tooling**
- localmaxxing. https://www.localmaxxing.com/en
- local-bench. https://local-bench.ai/
- LocalScore. https://www.localscore.ai/
- Local-Bench (GitHub). https://github.com/companionintelligence/Local-Bench
- Hosting LLMs: Ollama / LocalAI / Jan / LM Studio / vLLM comparison (2026). https://www.glukhov.org/llm-hosting/comparisons/hosting-llms-ollama-localai-jan-lmstudio-vllm-comparison/
- OpenBenchmarking llama.cpp suite. https://openbenchmarking.org/performance/test/pts/llama-cpp/
