# CUDA kernel design

The crate compiles one static CUDA library for SM89 with `-O3` and
`-lineinfo`. It does not use `--use_fast_math`. Kernels accumulate reductions
in `f32`. The compiler may use fused multiply-add because it performs one IEEE
rounding instead of two.

The CUDA backend losslessly repacks each Q4_K tensor at load. For `B`
superblocks, bytes `0..128 * B` hold codes. Bytes `128 * B..144 * B` hold
metadata.
One code block contains four 32-byte groups. Within a group, lane `i` stores
GGUF code bytes `4i..4i+4`, then bytes `16+4i..20+4i`. One aligned 8-byte load
supplies both `dp4a` chains for that lane.

Metadata bytes `0..4` keep the GGUF `d` and `dmin` f16 values. Each scale
group uses one 24-bit record: `scale0 | scale1 << 6 | min0 << 12 | min1 <<
18`. The low, middle, and high bytes for groups 0 through 3 occupy metadata
bytes `4..8`, `8..12`, and `12..16`. The 6-bit values are unchanged. The kernel
forms every product in `f32` with the GGUF arithmetic order. The scalar CPU
backend keeps the GGUF layout because backend buffers are opaque.

Q4_K GEMV uses four physical warps per output row. A separate kernel quantizes
the activation vector to q8_1 once per GEMV input. QKV shares one quantization.
Gate and up share one quantization through the backend's paired GEMV method.
Each q8_1 block stores 32 signed codes, an `f16` scale, and an `f16` source
sum. Q4_K computes each minimum correction from the same lane-local codes.

Sixteen lanes process one 256-value Q4_K block with `__dp4a`. One 128-thread
block covers eight Q4_K blocks per K iteration. The 4,096-column kernel
specializes 16 superblocks per row. The 12,288-column kernel specializes 48.
Each lane keeps one private accumulator across the full K loop. Warps 1 through
3 spill once to shared memory. Warp 0 combines the spills and runs one shuffle
reduction. QKV, gate, up, attention output, and FFN down use this same fixed
reduction order.

Q6_K GEMV quantizes the activation to q8_1 and uses `__dp4a`. Four warps handle
one output row. Each lane decodes one packed Q6_K code group and keeps a
private accumulator across its K blocks. The kernel reads weights directly
from device memory and does not stage them in shared memory. Neither GEMV path
materializes an `f32` weight matrix.

The backend can retain a q8_1 activation between producer and GEMV calls.
Attention writes its dense output and q8_1 blocks in the split-merge kernel.
SwiGLU does the same in one elementwise kernel. Those writes omit a standalone
quantization before the next GEMV.
RMSNorm launches one 256-thread block for every group of eight q8_1 blocks.
Each block repeats the same fixed RMS reduction, then writes a disjoint output
segment and its q8_1 blocks. The repeated reduction reads 16 KiB from L2. It
keeps all 128 q8_1 block reductions parallel and removes a second launch.

The attention-output, FFN-down, and Q6_K GEMV kernels can add a residual when
they write an output row. The CPU backend implements the same backend method
as a GEMV followed by an add. The CUDA epilogue uses the same `f32` addition
as the former elementwise kernel.

RMSNorm uses one 256-thread block per row. Cross-warp stages use shared memory.
The final warp uses shuffles and preserves a fixed reduction order. The
residual variant recomputes `left + right` after its square-sum reduction, so
it does not need a temporary vector. It uses `sqrtf`, not the approximate
reciprocal-square-root intrinsic.

Q and K normalization with RoPE uses one launch across all 32 query heads and
8 key heads. Other RoPE and SwiGLU work uses 256-thread elementwise grids.
RoPE uses GPT-NeoX half pairs and the unscaled frequency
`theta^(-2i/head_dim)`. RoPE forms its phase and rotation in `f64` to limit
long-position drift, then writes `f32`. SwiGLU uses the default precise `expf`
implementation.

Decode attention uses 128 threads and a register-resident online softmax.
Sixteen 8-lane groups process different KV positions. Each lane keeps 16 value
numerators for the 128-dimension path. The KV loop has no block barrier. Query,
key, and value segments use aligned 16-byte vectors.

The split count is fixed by the graph's power-of-two context bucket. The host
queries active blocks per SM for the kernel. It then searches 64-position
tiles for the highest one-wave utilization and caps the count at 64. The SM89
build selects 8, 16, 32, 36, and 36 F16 splits for the 512, 1,024, 2,048,
4,096, and 8,192 buckets. The F32 path selects 8, 16, 32, 32, and 32. The
difference follows from each compiled kernel's register count. Graph replay
preserves the selected launch geometry for every position in a bucket.

A second kernel combines partial states. Threads reduce the global maximum
and denominator in parallel. Each output-dimension thread then combines
splits in ascending order. The same kernel emits q8_1 blocks when its output
feeds a GEMV. The KV layout is
`[head_kv][max_context][head_dim]`, with `head_dim` contiguous. The query and
output layouts are `[head][head_dim]`. The context length includes the current
token, so decode is causal without a mask.

Embedding gather uses one 256-thread block and dequantizes only the selected
row. Argmax uses 256 threads and four values per thread. It reduces partial
pairs in a second kernel, ignores NaNs, and resolves ties to the lowest index.
All reductions avoid atomics and produce a fixed operation order.
