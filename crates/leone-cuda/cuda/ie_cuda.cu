#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda/atomic>
#include <math_constants.h>
#include <stddef.h>
#include <stdint.h>
#include <map>
#include <new>
#include <tuple>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kGemvWarpsPerBlock = 4;
constexpr int kAttentionThreads = 128;
constexpr int kAttentionTileThreads = 256;
constexpr int kAttentionKvTile = 192;
constexpr int kAttentionMaxSplitKv = 64;
constexpr int kArgmaxItemsPerThread = 4;
constexpr int kQuantBlockElements = 256;
constexpr int kQ8BlockElements = 32;
constexpr int kQ4CodeBytes = 128;
constexpr int kQ4MetadataBytes = 16;
constexpr int kQ6BlockBytes = 210;
constexpr int kQ4ProbePersistentBlocks = 1536;
constexpr int kQ4ApronBlocks = 512;
constexpr int kPrefillSoftmaxThreads = 256;

struct PrefillMatmulKey {
    size_t m;
    size_t n;
    size_t k;
    int batch_count;
    int64_t a_stride;
    int64_t b_stride;
    int64_t d_stride;
    bool transpose_b;

    bool operator<(const PrefillMatmulKey &other) const {
        return std::tie(m, n, k, batch_count, a_stride, b_stride, d_stride,
                        transpose_b) <
               std::tie(other.m, other.n, other.k, other.batch_count,
                        other.a_stride, other.b_stride, other.d_stride,
                        other.transpose_b);
    }
};

struct LeoneCublasLt {
    cublasLtHandle_t handle = nullptr;
    std::map<PrefillMatmulKey, cublasLtMatmulAlgo_t> algorithms;
};

struct __align__(4) Q8_1Block {
    __half2 ds;
    int8_t qs[kQ8BlockElements];
};

static_assert(sizeof(Q8_1Block) == 36);

struct __align__(2) Q8KVBlock {
    __half d;
    int8_t qs[kQ8BlockElements];
};

static_assert(sizeof(Q8KVBlock) == 34);

__device__ __forceinline__ float q8_kv_value(const void *cache,
                                              size_t index) {
    const Q8KVBlock &block =
        static_cast<const Q8KVBlock *>(cache)[index / kQ8BlockElements];
    return __half2float(block.d) *
           static_cast<float>(block.qs[index % kQ8BlockElements]);
}

struct DecodeGraph {
    cudaGraphExec_t executable;
};

__device__ __forceinline__ float half_from_bytes(const uint8_t *source) {
    __half_raw raw;
    raw.x = static_cast<uint16_t>(source[0]) |
            (static_cast<uint16_t>(source[1]) << 8);
    return __half2float(raw);
}

__device__ __forceinline__ float dequant_q4_repacked_value(
    const uint8_t *codes,
    const uint8_t *metadata,
    int index) {
    const float d = half_from_bytes(metadata);
    const float dmin = half_from_bytes(metadata + 2);
    const int group = index / 64;
    const int within = index % 64;
    const int half = within / 32;
    const int original = within % 32;
    const int striped = (original % 16) / 4 * 8 +
                        original / 16 * 4 + original % 4;
    const uint8_t packed = codes[group * 32 + striped];
    const int quant = half == 0 ? packed & 15 : packed >> 4;
    const uint32_t parameters =
        static_cast<uint32_t>(metadata[4 + group]) |
        (static_cast<uint32_t>(metadata[8 + group]) << 8) |
        (static_cast<uint32_t>(metadata[12 + group]) << 16);
    const int scale = static_cast<int>(
        (parameters >> (6 * half)) & 63);
    const int minimum = static_cast<int>(
        (parameters >> (12 + 6 * half)) & 63);
    return d * static_cast<float>(scale) * static_cast<float>(quant) -
           dmin * static_cast<float>(minimum);
}

__device__ __forceinline__ float dequant_q6_value(const uint8_t *block,
                                                   int index) {
    const int half = index / 128;
    const int within = index % 128;
    const int group = within / 32;
    const int lane = within % 32;
    const int low_index = half * 64 + lane + ((group & 1) == 0 ? 0 : 32);
    const uint8_t low_byte = block[low_index];
    const int low = group < 2 ? low_byte & 15 : low_byte >> 4;
    const uint8_t high_byte = block[128 + half * 32 + lane];
    const int high = (high_byte >> (group * 2)) & 3;
    const int quant = (low | (high << 4)) - 32;
    const int scale_index = half * 8 + lane / 16 + group * 2;
    const int scale = static_cast<int8_t>(block[192 + scale_index]);
    const float d = half_from_bytes(block + 208);
    return d * static_cast<float>(scale) * static_cast<float>(quant);
}

__device__ __forceinline__ void store_u32_bytes(uint8_t *destination,
                                                uint32_t value) {
    destination[0] = static_cast<uint8_t>(value);
    destination[1] = static_cast<uint8_t>(value >> 8);
    destination[2] = static_cast<uint8_t>(value >> 16);
    destination[3] = static_cast<uint8_t>(value >> 24);
}

__device__ __forceinline__ void load_q6_block(const uint8_t *source,
                                               uint8_t *destination) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(source);
    const int prefix = static_cast<int>((16 - (address & 15)) & 15);
    const int vector_count = (kQ6BlockBytes - prefix) / 16;
    const int vector_bytes = vector_count * 16;
    const int suffix = kQ6BlockBytes - prefix - vector_bytes;

    if (threadIdx.x < prefix) {
        destination[threadIdx.x] = source[threadIdx.x];
    }
    if (threadIdx.x < vector_count) {
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + prefix + threadIdx.x * 16);
        uint8_t *target = destination + prefix + threadIdx.x * 16;
        store_u32_bytes(target, value.x);
        store_u32_bytes(target + 4, value.y);
        store_u32_bytes(target + 8, value.z);
        store_u32_bytes(target + 12, value.w);
    }
    if (threadIdx.x < suffix) {
        const int offset = prefix + vector_bytes + threadIdx.x;
        destination[offset] = source[offset];
    }
}

__global__ void quantize_q8_1_warp(const float *__restrict__ input,
                                   Q8_1Block *__restrict__ output,
                                   int32_t *__restrict__ quantized_sums,
                                   size_t blocks) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t block = blockIdx.x * (kBlockThreads / 32) + warp;
    if (block >= blocks) {
        return;
    }

    const float value = input[block * kQ8BlockElements + lane];
    float maximum = fabsf(value);
    float sum = value;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum,
                        __shfl_down_sync(0xffffffff, maximum, offset));
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum / 127.0f;
    const int quant = maximum == 0.0f
                          ? 0
                          : __float2int_rn(value / scale);
    output[block].qs[lane] = static_cast<int8_t>(quant);
    int quantized_sum = quant;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        quantized_sum +=
            __shfl_down_sync(0xffffffff, quantized_sum, offset);
    }
    if (lane == 0) {
        output[block].ds = __floats2half2_rn(scale, sum);
        quantized_sums[block] = quantized_sum;
    }
}

__device__ __forceinline__ uint32_t q4_group_parameters(
    const uint8_t *__restrict__ metadata,
    int group) {
    return static_cast<uint32_t>(__ldcs(metadata + 4 + group)) |
           (static_cast<uint32_t>(__ldcs(metadata + 8 + group)) << 8) |
           (static_cast<uint32_t>(__ldcs(metadata + 12 + group)) << 16);
}

__device__ __forceinline__ float q4_q8_block_dot(
    const uint8_t *__restrict__ row_codes,
    const uint8_t *__restrict__ row_metadata,
    const Q8_1Block *__restrict__ input,
    size_t quant_block,
    int lane,
    int local,
    int q8_offset,
    int item) {
    const uint2 packed = __ldcs(reinterpret_cast<const uint2 *>(
        row_codes + quant_block * kQ4CodeBytes + local * 8));
    const uint8_t *metadata =
        row_metadata + quant_block * kQ4MetadataBytes;
    uint32_t parameters = 0;
    if (item == 0) {
        parameters = q4_group_parameters(metadata, q8_offset / 2);
    }
    parameters = __shfl_sync(0xffffffff, parameters, lane - item);

    uint32_t half_scales = 0;
    if (local == 0) {
        half_scales = __ldcs(
            reinterpret_cast<const uint32_t *>(metadata));
    }
    half_scales = __shfl_sync(0xffffffff, half_scales, lane & 16);
    __half_raw d_raw;
    __half_raw dmin_raw;
    d_raw.x = static_cast<uint16_t>(half_scales);
    dmin_raw.x = static_cast<uint16_t>(half_scales >> 16);
    const float d = __half2float(d_raw);
    const float dmin = __half2float(dmin_raw);

    float scaled_dot = 0.0f;
    float minimum_dot = 0.0f;
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        const size_t q8_block =
            quant_block * (kQuantBlockElements / kQ8BlockElements) +
            q8_offset + index;
        const Q8_1Block *q8 = input + q8_block;
        const int *values =
            reinterpret_cast<const int *>(q8->qs) + item;
        const int dot = __dp4a(
            static_cast<int>((packed.y >> (4 * index)) & 0x0f0f0f0f),
            values[4],
            __dp4a(static_cast<int>(
                       (packed.x >> (4 * index)) & 0x0f0f0f0f),
                   values[0], 0));
        const int input_sum = __dp4a(
            0x01010101, values[4],
            __dp4a(0x01010101, values[0], 0));
        const float input_scale = __low2float(q8->ds);
        const int scale = static_cast<int>(
            (parameters >> (6 * index)) & 63);
        const int minimum = static_cast<int>(
            (parameters >> (12 + 6 * index)) & 63);
        scaled_dot += input_scale * static_cast<float>(dot * scale);
        minimum_dot += input_scale * static_cast<float>(input_sum * minimum);
    }
    return d * scaled_dot - dmin * minimum_dot;
}

__device__ __forceinline__ uint4 load_l2_evict_last(
    const uint8_t *__restrict__ source) {
    uint4 value;
    asm volatile(
        "{\n"
        "   .reg .b64 policy;\n"
        "   createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n"
        "   ld.global.cg.L2::cache_hint.v4.u32 "
        "{%0, %1, %2, %3}, [%4], policy;\n"
        "}\n"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(source));
    return value;
}

template <int FixedBlocksPerRow, int WarpsPerRow>
__device__ __forceinline__ float q4_q8_row_dot_long_k(
    const uint8_t *__restrict__ row_codes,
    const uint8_t *__restrict__ row_metadata,
    const Q8_1Block *__restrict__ input,
    size_t columns,
    float *__restrict__ warp_partials) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    float sum = 0.0f;
    if constexpr (FixedBlocksPerRow > 0) {
#pragma unroll
        for (int quant_block = block_offset;
             quant_block < FixedBlocksPerRow;
             quant_block += WarpsPerRow * 2) {
            sum += q4_q8_block_dot(
                row_codes, row_metadata, input, quant_block, lane, local,
                q8_offset, item);
        }
    } else {
        const size_t blocks_per_row = columns / kQuantBlockElements;
        for (size_t quant_block = block_offset;
             quant_block < blocks_per_row;
             quant_block += WarpsPerRow * 2) {
            sum += q4_q8_block_dot(
                row_codes, row_metadata, input, quant_block, lane, local,
                q8_offset, item);
        }
    }
    if (warp > 0) {
        warp_partials[(warp - 1) * 32 + lane] = sum;
    }
    __syncthreads();
    if (warp > 0) {
        return sum;
    }
#pragma unroll
    for (int source_warp = 0; source_warp < WarpsPerRow - 1;
         ++source_warp) {
        sum += warp_partials[source_warp * 32 + lane];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

template <int Positions>
__device__ __forceinline__ void q4_q8_block_dots_multi(
    const uint8_t *__restrict__ row_codes,
    const uint8_t *__restrict__ row_metadata,
    const Q8_1Block *__restrict__ input,
    size_t input_blocks,
    size_t quant_block,
    int lane,
    int local,
    int q8_offset,
    int item,
    float (&sums)[Positions]) {
    const uint2 packed = __ldcs(reinterpret_cast<const uint2 *>(
        row_codes + quant_block * kQ4CodeBytes + local * 8));
    const uint8_t *metadata = row_metadata + quant_block * kQ4MetadataBytes;
    uint32_t parameters = 0;
    if (item == 0) {
        parameters = q4_group_parameters(metadata, q8_offset / 2);
    }
    parameters = __shfl_sync(0xffffffff, parameters, lane - item);
    uint32_t half_scales = 0;
    if (local == 0) {
        half_scales = __ldcs(reinterpret_cast<const uint32_t *>(metadata));
    }
    half_scales = __shfl_sync(0xffffffff, half_scales, lane & 16);
    __half_raw d_raw;
    __half_raw dmin_raw;
    d_raw.x = static_cast<uint16_t>(half_scales);
    dmin_raw.x = static_cast<uint16_t>(half_scales >> 16);
    const float d = __half2float(d_raw);
    const float dmin = __half2float(dmin_raw);

#pragma unroll
    for (int position = 0; position < Positions; ++position) {
        float scaled_dot = 0.0f;
        float minimum_dot = 0.0f;
#pragma unroll
        for (int index = 0; index < 2; ++index) {
            const size_t q8_block =
                quant_block * (kQuantBlockElements / kQ8BlockElements) +
                q8_offset + index;
            const Q8_1Block *q8 = input + position * input_blocks + q8_block;
            const int *values = reinterpret_cast<const int *>(q8->qs) + item;
            const int dot = __dp4a(
                static_cast<int>((packed.y >> (4 * index)) & 0x0f0f0f0f),
                values[4],
                __dp4a(static_cast<int>(
                           (packed.x >> (4 * index)) & 0x0f0f0f0f),
                       values[0], 0));
            const int input_sum = __dp4a(
                0x01010101, values[4],
                __dp4a(0x01010101, values[0], 0));
            const float input_scale = __low2float(q8->ds);
            const int scale = static_cast<int>((parameters >> (6 * index)) & 63);
            const int minimum =
                static_cast<int>((parameters >> (12 + 6 * index)) & 63);
            scaled_dot += input_scale * static_cast<float>(dot * scale);
            minimum_dot += input_scale * static_cast<float>(input_sum * minimum);
        }
        sums[position] += d * scaled_dot - dmin * minimum_dot;
    }
}

template <int FixedBlocksPerRow, int WarpsPerRow, int Positions>
__global__ void q4_q8_gemv_multi_warp(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    const float *__restrict__ residual,
    float *__restrict__ output,
    size_t rows,
    size_t columns) {
    __shared__ float warp_partials[Positions][WarpsPerRow - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    const size_t row = blockIdx.x;
    const size_t blocks_per_row = FixedBlocksPerRow > 0
                                      ? FixedBlocksPerRow
                                      : columns / kQuantBlockElements;
    const size_t input_blocks = columns / kQ8BlockElements;
    const uint8_t *metadata =
        weights + rows * blocks_per_row * kQ4CodeBytes;
    const uint8_t *row_codes =
        weights + row * blocks_per_row * kQ4CodeBytes;
    const uint8_t *row_metadata =
        metadata + row * blocks_per_row * kQ4MetadataBytes;
    float sums[Positions] = {};
    if constexpr (FixedBlocksPerRow > 0) {
#pragma unroll
        for (int quant_block = block_offset;
             quant_block < FixedBlocksPerRow;
             quant_block += WarpsPerRow * 2) {
            q4_q8_block_dots_multi(
                row_codes, row_metadata, input, input_blocks, quant_block,
                lane, local, q8_offset, item, sums);
        }
    } else {
        for (size_t quant_block = block_offset;
             quant_block < blocks_per_row;
             quant_block += WarpsPerRow * 2) {
            q4_q8_block_dots_multi(
                row_codes, row_metadata, input, input_blocks, quant_block,
                lane, local, q8_offset, item, sums);
        }
    }
#pragma unroll
    for (int position = 0; position < Positions; ++position) {
        if (warp > 0) {
            warp_partials[position][warp - 1][lane] = sums[position];
        }
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int position = 0; position < Positions; ++position) {
#pragma unroll
        for (int source_warp = 0; source_warp < WarpsPerRow - 1;
             ++source_warp) {
            sums[position] += warp_partials[position][source_warp][lane];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[position] +=
                __shfl_down_sync(0xffffffff, sums[position], offset);
        }
        if (lane == 0) {
            const size_t index = static_cast<size_t>(position) * rows + row;
            output[index] = residual == nullptr
                                ? sums[position]
                                : sums[position] + residual[index];
        }
    }
}

template <int FixedBlocksPerRow, int WarpsPerRow>
__device__ __forceinline__ float2 q4_q8_two_row_dots_long_k(
    const uint8_t *__restrict__ first_codes,
    const uint8_t *__restrict__ first_metadata,
    const uint8_t *__restrict__ second_codes,
    const uint8_t *__restrict__ second_metadata,
    const Q8_1Block *__restrict__ input,
    size_t columns,
    float *__restrict__ first_warp_partials,
    float *__restrict__ second_warp_partials) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    if constexpr (FixedBlocksPerRow > 0) {
#pragma unroll
        for (int quant_block = block_offset;
             quant_block < FixedBlocksPerRow;
             quant_block += WarpsPerRow * 2) {
            first_sum += q4_q8_block_dot(
                first_codes, first_metadata, input, quant_block, lane, local,
                q8_offset, item);
            second_sum += q4_q8_block_dot(
                second_codes, second_metadata, input, quant_block, lane,
                local, q8_offset, item);
        }
    } else {
        const size_t blocks_per_row = columns / kQuantBlockElements;
        for (size_t quant_block = block_offset;
             quant_block < blocks_per_row;
             quant_block += WarpsPerRow * 2) {
            first_sum += q4_q8_block_dot(
                first_codes, first_metadata, input, quant_block, lane, local,
                q8_offset, item);
            second_sum += q4_q8_block_dot(
                second_codes, second_metadata, input, quant_block, lane,
                local, q8_offset, item);
        }
    }
    if (warp > 0) {
        first_warp_partials[(warp - 1) * 32 + lane] = first_sum;
        second_warp_partials[(warp - 1) * 32 + lane] = second_sum;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int source_warp = 0; source_warp < WarpsPerRow - 1;
             ++source_warp) {
            first_sum += first_warp_partials[source_warp * 32 + lane];
            second_sum += second_warp_partials[source_warp * 32 + lane];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            first_sum += __shfl_down_sync(0xffffffff, first_sum, offset);
            second_sum += __shfl_down_sync(0xffffffff, second_sum, offset);
        }
    }
    return make_float2(first_sum, second_sum);
}

__device__ __forceinline__ uint32_t q4_q8_block_load_xor(
    const uint8_t *__restrict__ row_codes,
    const uint8_t *__restrict__ row_metadata,
    const Q8_1Block *__restrict__ input,
    size_t quant_block,
    int lane,
    int local,
    int q8_offset,
    int item) {
    const uint2 packed = *reinterpret_cast<const uint2 *>(
        row_codes + quant_block * kQ4CodeBytes + local * 8);
    const uint8_t *metadata =
        row_metadata + quant_block * kQ4MetadataBytes;
    uint32_t parameters = 0;
    if (item == 0) {
        parameters = q4_group_parameters(metadata, q8_offset / 2);
    }
    parameters = __shfl_sync(0xffffffff, parameters, lane - item);

    uint32_t half_scales = 0;
    if (local == 0) {
        half_scales = __ldcs(
            reinterpret_cast<const uint32_t *>(metadata));
    }
    half_scales = __shfl_sync(0xffffffff, half_scales, lane & 16);

    uint32_t loaded = packed.x ^ packed.y ^ parameters ^ half_scales;
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        const size_t q8_block =
            quant_block * (kQuantBlockElements / kQ8BlockElements) +
            q8_offset + index;
        const Q8_1Block *q8 = input + q8_block;
        const int *values = reinterpret_cast<const int *>(q8->qs) + item;
        const uint16_t scale =
            *reinterpret_cast<const uint16_t *>(&q8->ds);
        loaded ^= static_cast<uint32_t>(values[0]);
        loaded ^= static_cast<uint32_t>(values[4]);
        loaded ^= static_cast<uint32_t>(scale);
    }
    return loaded;
}

__device__ __forceinline__ float q4_q8_block_dot_wide(
    const uint8_t *__restrict__ row_codes,
    const uint8_t *__restrict__ row_metadata,
    const Q8_1Block *__restrict__ input,
    size_t quant_block,
    int lane,
    int local) {
    const uint4 packed = *reinterpret_cast<const uint4 *>(
        row_codes + quant_block * kQ4CodeBytes + local * 16);
    const uint8_t *metadata =
        row_metadata + quant_block * kQ4MetadataBytes;
    const int q8_offset = 2 * (local / 2);
    const int first_item = 2 * (local & 1);
    uint32_t parameters = 0;
    if ((local & 1) == 0) {
        parameters = q4_group_parameters(metadata, q8_offset / 2);
    }
    parameters = __shfl_sync(0xffffffff, parameters, lane - (local & 1));

    uint32_t half_scales = 0;
    if (local == 0) {
        half_scales = __ldcs(
            reinterpret_cast<const uint32_t *>(metadata));
    }
    half_scales = __shfl_sync(0xffffffff, half_scales, lane - local);
    __half_raw d_raw;
    __half_raw dmin_raw;
    d_raw.x = static_cast<uint16_t>(half_scales);
    dmin_raw.x = static_cast<uint16_t>(half_scales >> 16);
    const float d = __half2float(d_raw);
    const float dmin = __half2float(dmin_raw);

    float scaled_dot = 0.0f;
    float minimum_dot = 0.0f;
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        const size_t q8_block =
            quant_block * (kQuantBlockElements / kQ8BlockElements) +
            q8_offset + index;
        const Q8_1Block *q8 = input + q8_block;
        const int *values = reinterpret_cast<const int *>(q8->qs);
        const int first_low = values[first_item];
        const int second_low = values[first_item + 1];
        const int first_high = values[first_item + 4];
        const int second_high = values[first_item + 5];
        const int shift = 4 * index;
        const int first_dot = __dp4a(
            static_cast<int>((packed.y >> shift) & 0x0f0f0f0f),
            first_high,
            __dp4a(
                static_cast<int>((packed.x >> shift) & 0x0f0f0f0f),
                first_low, 0));
        const int second_dot = __dp4a(
            static_cast<int>((packed.w >> shift) & 0x0f0f0f0f),
            second_high,
            __dp4a(
                static_cast<int>((packed.z >> shift) & 0x0f0f0f0f),
                second_low, 0));
        const int first_sum = __dp4a(
            0x01010101, first_high,
            __dp4a(0x01010101, first_low, 0));
        const int second_sum = __dp4a(
            0x01010101, second_high,
            __dp4a(0x01010101, second_low, 0));
        const float input_scale = __low2float(q8->ds);
        const int scale = static_cast<int>(
            (parameters >> (6 * index)) & 63);
        const int minimum = static_cast<int>(
            (parameters >> (12 + 6 * index)) & 63);
        scaled_dot += input_scale *
                      static_cast<float>((first_dot + second_dot) * scale);
        minimum_dot += input_scale *
                       static_cast<float>((first_sum + second_sum) * minimum);
    }
    return d * scaled_dot - dmin * minimum_dot;
}

__global__ void q4_q8_gemv_wide_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output,
    size_t rows) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 8;
    const int local = thread % 8;
    constexpr int blocks_per_row = 16;
    const size_t row = blockIdx.x;
    const uint8_t *metadata =
        weights + rows * blocks_per_row * kQ4CodeBytes;
    const float sum = q4_q8_block_dot_wide(
        weights + row * blocks_per_row * kQ4CodeBytes,
        metadata + row * blocks_per_row * kQ4MetadataBytes,
        input, block_offset, lane, local);
    if (warp > 0) {
        warp_partials[warp - 1][lane] = sum;
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
    float reduced = sum;
#pragma unroll
    for (int source_warp = 0;
         source_warp < kGemvWarpsPerBlock - 1;
         ++source_warp) {
        reduced += warp_partials[source_warp][lane];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        reduced += __shfl_down_sync(0xffffffff, reduced, offset);
    }
    if (lane == 0) {
        output[row] = reduced;
    }
}

__global__ void q4_q8_gemv_two_rows_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output,
    size_t rows) {
    __shared__ float warp_partials[2][kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    constexpr int blocks_per_row = 16;
    const size_t first_row = blockIdx.x * 2;
    const size_t second_row = first_row + 1;
    const uint8_t *metadata =
        weights + rows * blocks_per_row * kQ4CodeBytes;
    const uint8_t *first_codes =
        weights + first_row * blocks_per_row * kQ4CodeBytes;
    const uint8_t *second_codes =
        weights + second_row * blocks_per_row * kQ4CodeBytes;
    const uint8_t *first_metadata =
        metadata + first_row * blocks_per_row * kQ4MetadataBytes;
    const uint8_t *second_metadata =
        metadata + second_row * blocks_per_row * kQ4MetadataBytes;

    float first_sum = q4_q8_block_dot(
        first_codes, first_metadata, input, block_offset, lane, local,
        q8_offset, item);
    float second_sum = q4_q8_block_dot(
        second_codes, second_metadata, input, block_offset, lane, local,
        q8_offset, item);
    first_sum += q4_q8_block_dot(
        first_codes, first_metadata, input, block_offset + 8, lane, local,
        q8_offset, item);
    second_sum += q4_q8_block_dot(
        second_codes, second_metadata, input, block_offset + 8, lane, local,
        q8_offset, item);
    if (warp > 0) {
        warp_partials[0][warp - 1][lane] = first_sum;
        warp_partials[1][warp - 1][lane] = second_sum;
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int source_warp = 0;
         source_warp < kGemvWarpsPerBlock - 1;
         ++source_warp) {
        first_sum += warp_partials[0][source_warp][lane];
        second_sum += warp_partials[1][source_warp][lane];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        first_sum += __shfl_down_sync(0xffffffff, first_sum, offset);
        second_sum += __shfl_down_sync(0xffffffff, second_sum, offset);
    }
    if (lane == 0) {
        output[first_row] = first_sum;
        output[second_row] = second_sum;
    }
}

__global__ void q4_q8_gemv_row_layout_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    constexpr int blocks_per_row = 16;
    constexpr int row_bytes =
        blocks_per_row * (kQ4CodeBytes + kQ4MetadataBytes);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t row = blockIdx.x;
    const uint8_t *row_weights = weights + row * row_bytes;
    const float sum = q4_q8_row_dot_long_k<
        blocks_per_row, kGemvWarpsPerBlock>(
        row_weights,
        row_weights + blocks_per_row * kQ4CodeBytes,
        input, 4096, &warp_partials[0][0]);
    if (warp == 0 && lane == 0) {
        output[row] = sum;
    }
}

__global__ void q4_q8_gemv_split_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ partials,
    size_t rows) {
    __shared__ float warp_partials[32];
    constexpr int blocks_per_row = 16;
    constexpr int warps_per_split = 2;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    const size_t row = blockIdx.x / 2;
    const int split = blockIdx.x & 1;
    const uint8_t *metadata =
        weights + rows * blocks_per_row * kQ4CodeBytes;
    const uint8_t *row_codes =
        weights + row * blocks_per_row * kQ4CodeBytes;
    const uint8_t *row_metadata =
        metadata + row * blocks_per_row * kQ4MetadataBytes;
    float sum = 0.0f;
#pragma unroll
    for (int quant_block = split * 8 + block_offset;
         quant_block < (split + 1) * 8;
         quant_block += warps_per_split * 2) {
        sum += q4_q8_block_dot(
            row_codes, row_metadata, input, quant_block, lane, local,
            q8_offset, item);
    }
    if (warp > 0) {
        warp_partials[lane] = sum;
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
    sum += warp_partials[lane];
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        partials[split * rows + row] = sum;
    }
}

__global__ void q4_q8_gemv_split_reduce_probe(
    float *__restrict__ partials,
    size_t rows) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        partials[row] += partials[rows + row];
    }
}

__device__ __forceinline__ float q4_q8_block_dot_gguf(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    size_t quant_block,
    int local) {
    const uint8_t *block = weights + quant_block * 144;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    const int *codes = reinterpret_cast<const int *>(
        block + 16 + 16 * q8_offset + 4 * item);
    const int first_codes = codes[0];
    const int second_codes = codes[4];
    const uint16_t *scales =
        reinterpret_cast<const uint16_t *>(block + 4);
    const int group = q8_offset / 2;
    uint16_t scale_bytes;
    uint16_t minimum_bytes;
    if (group < 2) {
        scale_bytes = scales[group] & 0x3f3f;
        minimum_bytes = scales[group + 2] & 0x3f3f;
    } else {
        scale_bytes = static_cast<uint16_t>(
            ((scales[group + 2] >> 0) & 0x0f0f) |
            ((scales[group - 2] & 0xc0c0) >> 2));
        minimum_bytes = static_cast<uint16_t>(
            ((scales[group + 2] >> 4) & 0x0f0f) |
            ((scales[group] & 0xc0c0) >> 2));
    }
    const uint32_t half_scales =
        *reinterpret_cast<const uint32_t *>(block);
    __half_raw d_raw;
    __half_raw dmin_raw;
    d_raw.x = static_cast<uint16_t>(half_scales);
    dmin_raw.x = static_cast<uint16_t>(half_scales >> 16);
    const float d = __half2float(d_raw);
    const float dmin = __half2float(dmin_raw);

    float scaled_dot = 0.0f;
    float minimum_dot = 0.0f;
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        const Q8_1Block *q8 = input + quant_block * 8 + q8_offset + index;
        const int *values = reinterpret_cast<const int *>(q8->qs) + item;
        const int codes0 =
            (first_codes >> (4 * index)) & 0x0f0f0f0f;
        const int codes1 =
            (second_codes >> (4 * index)) & 0x0f0f0f0f;
        const int dot = __dp4a(
            codes1, values[4], __dp4a(codes0, values[0], 0));
        const int input_sum = __dp4a(
            0x01010101, values[4],
            __dp4a(0x01010101, values[0], 0));
        const int scale = index == 0
                              ? scale_bytes & 0xff
                              : scale_bytes >> 8;
        const int minimum = index == 0
                                ? minimum_bytes & 0xff
                                : minimum_bytes >> 8;
        const float input_scale = __low2float(q8->ds);
        scaled_dot += input_scale * static_cast<float>(dot * scale);
        minimum_dot +=
            input_scale * static_cast<float>(input_sum * minimum);
    }
    return d * scaled_dot - dmin * minimum_dot;
}

__global__ void q4_q8_gemv_gguf_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    constexpr int blocks_per_row = 16;
    const size_t row = blockIdx.x;
    const uint8_t *row_weights =
        weights + row * blocks_per_row * 144;
    float sum = 0.0f;
#pragma unroll
    for (int quant_block = block_offset;
         quant_block < blocks_per_row;
         quant_block += kGemvWarpsPerBlock * 2) {
        sum += q4_q8_block_dot_gguf(
            row_weights, input, quant_block, local);
    }
    if (warp > 0) {
        warp_partials[warp - 1][lane] = sum;
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int source_warp = 0;
         source_warp < kGemvWarpsPerBlock - 1;
         ++source_warp) {
        sum += warp_partials[source_warp][lane];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

struct Q4Q8Loaded {
    uint2 packed;
    uint32_t parameters;
    uint32_t half_scales;
    int low[2];
    int high[2];
    uint16_t input_scale[2];
};

__device__ __forceinline__ Q4Q8Loaded q4_q8_load(
    const uint8_t *__restrict__ row_codes,
    const uint8_t *__restrict__ row_metadata,
    const Q8_1Block *__restrict__ input,
    size_t quant_block,
    int lane,
    int local,
    int q8_offset,
    int item) {
    Q4Q8Loaded loaded;
    loaded.packed = *reinterpret_cast<const uint2 *>(
        row_codes + quant_block * kQ4CodeBytes + local * 8);
    const uint8_t *metadata =
        row_metadata + quant_block * kQ4MetadataBytes;
    loaded.parameters = 0;
    if (item == 0) {
        loaded.parameters = q4_group_parameters(metadata, q8_offset / 2);
    }
    loaded.parameters = __shfl_sync(
        0xffffffff, loaded.parameters, lane - item);
    loaded.half_scales = 0;
    if (local == 0) {
        loaded.half_scales =
            *reinterpret_cast<const uint32_t *>(metadata);
    }
    loaded.half_scales = __shfl_sync(
        0xffffffff, loaded.half_scales, lane & 16);
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        const size_t q8_block =
            quant_block * (kQuantBlockElements / kQ8BlockElements) +
            q8_offset + index;
        const Q8_1Block *q8 = input + q8_block;
        const int *values = reinterpret_cast<const int *>(q8->qs) + item;
        loaded.low[index] = values[0];
        loaded.high[index] = values[4];
        loaded.input_scale[index] =
            *reinterpret_cast<const uint16_t *>(&q8->ds);
    }
    return loaded;
}

__device__ __forceinline__ float q4_q8_dot_loaded(
    const Q4Q8Loaded &loaded) {
    __half_raw d_raw;
    __half_raw dmin_raw;
    d_raw.x = static_cast<uint16_t>(loaded.half_scales);
    dmin_raw.x = static_cast<uint16_t>(loaded.half_scales >> 16);
    const float d = __half2float(d_raw);
    const float dmin = __half2float(dmin_raw);
    float scaled_dot = 0.0f;
    float minimum_dot = 0.0f;
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        const int dot = __dp4a(
            static_cast<int>((loaded.packed.y >> (4 * index)) &
                             0x0f0f0f0f),
            loaded.high[index],
            __dp4a(
                static_cast<int>((loaded.packed.x >> (4 * index)) &
                                 0x0f0f0f0f),
                loaded.low[index], 0));
        const int input_sum = __dp4a(
            0x01010101, loaded.high[index],
            __dp4a(0x01010101, loaded.low[index], 0));
        const int scale = static_cast<int>(
            (loaded.parameters >> (6 * index)) & 63);
        const int minimum = static_cast<int>(
            (loaded.parameters >> (12 + 6 * index)) & 63);
        __half_raw input_scale_raw;
        input_scale_raw.x = loaded.input_scale[index];
        const float input_scale = __half2float(input_scale_raw);
        scaled_dot += input_scale * static_cast<float>(dot * scale);
        minimum_dot +=
            input_scale * static_cast<float>(input_sum * minimum);
    }
    return d * scaled_dot - dmin * minimum_dot;
}

__global__ void q4_q8_gemv_two_block_ilp_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output,
    size_t rows) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    constexpr int blocks_per_row = 16;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    const size_t row = blockIdx.x;
    const uint8_t *metadata =
        weights + rows * blocks_per_row * kQ4CodeBytes;
    const uint8_t *row_codes =
        weights + row * blocks_per_row * kQ4CodeBytes;
    const uint8_t *row_metadata =
        metadata + row * blocks_per_row * kQ4MetadataBytes;
    const Q4Q8Loaded first = q4_q8_load(
        row_codes, row_metadata, input, block_offset, lane, local,
        q8_offset, item);
    const Q4Q8Loaded second = q4_q8_load(
        row_codes, row_metadata, input, block_offset + 8, lane, local,
        q8_offset, item);
    float sum = q4_q8_dot_loaded(first) + q4_q8_dot_loaded(second);
    if (warp > 0) {
        warp_partials[warp - 1][lane] = sum;
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int source_warp = 0;
         source_warp < kGemvWarpsPerBlock - 1;
         ++source_warp) {
        sum += warp_partials[source_warp][lane];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

__device__ __forceinline__ float q4_q8_block_dot_gguf_wide(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    size_t quant_block,
    int local) {
    const uint8_t *block = weights + quant_block * 144;
    const int q8_offset = 2 * (local / 2);
    const int first_item = 2 * (local & 1);
    const int2 first_codes = *reinterpret_cast<const int2 *>(
        block + 16 + 16 * q8_offset + 4 * first_item);
    const int2 second_codes = *reinterpret_cast<const int2 *>(
        block + 32 + 16 * q8_offset + 4 * first_item);
    const uint16_t *scales =
        reinterpret_cast<const uint16_t *>(block + 4);
    const int group = q8_offset / 2;
    uint16_t scale_bytes;
    uint16_t minimum_bytes;
    if (group < 2) {
        scale_bytes = scales[group] & 0x3f3f;
        minimum_bytes = scales[group + 2] & 0x3f3f;
    } else {
        scale_bytes = static_cast<uint16_t>(
            ((scales[group + 2] >> 0) & 0x0f0f) |
            ((scales[group - 2] & 0xc0c0) >> 2));
        minimum_bytes = static_cast<uint16_t>(
            ((scales[group + 2] >> 4) & 0x0f0f) |
            ((scales[group] & 0xc0c0) >> 2));
    }
    const uint32_t half_scales =
        *reinterpret_cast<const uint32_t *>(block);
    __half_raw d_raw;
    __half_raw dmin_raw;
    d_raw.x = static_cast<uint16_t>(half_scales);
    dmin_raw.x = static_cast<uint16_t>(half_scales >> 16);
    const float d = __half2float(d_raw);
    const float dmin = __half2float(dmin_raw);

    float scaled_dot = 0.0f;
    float minimum_dot = 0.0f;
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        const Q8_1Block *q8 = input + quant_block * 8 + q8_offset + index;
        const int *values = reinterpret_cast<const int *>(q8->qs);
        const int first_low = values[first_item];
        const int second_low = values[first_item + 1];
        const int first_high = values[first_item + 4];
        const int second_high = values[first_item + 5];
        const int shift = 4 * index;
        const int first_dot = __dp4a(
            (second_codes.x >> shift) & 0x0f0f0f0f,
            first_high,
            __dp4a(
                (first_codes.x >> shift) & 0x0f0f0f0f,
                first_low, 0));
        const int second_dot = __dp4a(
            (second_codes.y >> shift) & 0x0f0f0f0f,
            second_high,
            __dp4a(
                (first_codes.y >> shift) & 0x0f0f0f0f,
                second_low, 0));
        const int first_sum = __dp4a(
            0x01010101, first_high,
            __dp4a(0x01010101, first_low, 0));
        const int second_sum = __dp4a(
            0x01010101, second_high,
            __dp4a(0x01010101, second_low, 0));
        const int scale = index == 0
                              ? scale_bytes & 0xff
                              : scale_bytes >> 8;
        const int minimum = index == 0
                                ? minimum_bytes & 0xff
                                : minimum_bytes >> 8;
        const float input_scale = __low2float(q8->ds);
        scaled_dot += input_scale *
                      static_cast<float>((first_dot + second_dot) * scale);
        minimum_dot += input_scale *
                       static_cast<float>((first_sum + second_sum) * minimum);
    }
    return d * scaled_dot - dmin * minimum_dot;
}

__global__ void q4_q8_gemv_gguf_wide_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 8;
    const int local = thread % 8;
    constexpr int blocks_per_row = 16;
    const size_t row = blockIdx.x;
    const uint8_t *row_weights = weights + row * blocks_per_row * 144;
    float sum = q4_q8_block_dot_gguf_wide(
        row_weights, input, block_offset, local);
    if (warp > 0) {
        warp_partials[warp - 1][lane] = sum;
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int source_warp = 0;
         source_warp < kGemvWarpsPerBlock - 1;
         ++source_warp) {
        sum += warp_partials[source_warp][lane];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

__global__ void q4_q8_gemv_gguf_wide_one_warp_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output) {
    const int lane = threadIdx.x & 31;
    const int row_in_cta = threadIdx.x >> 5;
    const int block_offset = lane / 8;
    const int local = lane % 8;
    constexpr int blocks_per_row = 16;
    const size_t row = blockIdx.x * 4 + row_in_cta;
    const uint8_t *row_weights = weights + row * blocks_per_row * 144;
    float sum = 0.0f;
#pragma unroll
    for (int quant_block = block_offset;
         quant_block < blocks_per_row;
         quant_block += 4) {
        sum += q4_q8_block_dot_gguf_wide(
            row_weights, input, quant_block, local);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

template <int WarpsPerRow, int RowsPerCta, bool LoadsOnly>
__global__ void q4_q8_gemv_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t columns) {
    __shared__ float warp_partials[RowsPerCta]
                                      [WarpsPerRow > 1 ? WarpsPerRow - 1 : 1]
                                      [32];
    const int lane = threadIdx.x & 31;
    const int physical_warp = threadIdx.x >> 5;
    const int row_in_cta = physical_warp / WarpsPerRow;
    const int warp = physical_warp % WarpsPerRow;
    const int thread = warp * 32 + lane;
    const int block_offset = thread / 16;
    const int local = thread % 16;
    const int q8_offset = 2 * (local / 4);
    const int item = local % 4;
    const size_t row = blockIdx.x * RowsPerCta + row_in_cta;
    const size_t blocks_per_row = columns / kQuantBlockElements;
    const uint8_t *metadata =
        weights + rows * blocks_per_row * kQ4CodeBytes;
    float sum = 0.0f;
    uint32_t loaded = 0;
    if (row < rows) {
        const uint8_t *row_codes =
            weights + row * blocks_per_row * kQ4CodeBytes;
        const uint8_t *row_metadata =
            metadata + row * blocks_per_row * kQ4MetadataBytes;
        for (size_t quant_block = block_offset;
             quant_block < blocks_per_row;
             quant_block += WarpsPerRow * 2) {
            if constexpr (LoadsOnly) {
                loaded ^= q4_q8_block_load_xor(
                    row_codes, row_metadata, input, quant_block, lane,
                    local, q8_offset, item);
            } else {
                sum += q4_q8_block_dot(
                    row_codes, row_metadata, input, quant_block, lane,
                    local, q8_offset, item);
            }
        }
    }
    if constexpr (LoadsOnly) {
        sum = __uint_as_float(loaded);
    }
    if (warp > 0) {
        warp_partials[row_in_cta][warp - 1][lane] = sum;
    }
    __syncthreads();
    if (warp > 0 || row >= rows) {
        return;
    }
#pragma unroll
    for (int source_warp = 0; source_warp < WarpsPerRow - 1;
         ++source_warp) {
        if constexpr (LoadsOnly) {
            const uint32_t partial = __float_as_uint(
                warp_partials[row_in_cta][source_warp][lane]);
            sum = __uint_as_float(__float_as_uint(sum) ^ partial);
        } else {
            sum += warp_partials[row_in_cta][source_warp][lane];
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffff, sum, offset);
        if constexpr (LoadsOnly) {
            sum = __uint_as_float(
                __float_as_uint(sum) ^ __float_as_uint(other));
        } else {
            sum += other;
        }
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

__device__ __forceinline__ int load_i32_aligned2(
    const uint8_t *__restrict__ source,
    int index) {
    const uint16_t *values = reinterpret_cast<const uint16_t *>(source);
    return static_cast<int>(__ldcs(values + 2 * index)) |
           (static_cast<int>(__ldcs(values + 2 * index + 1)) << 16);
}

__device__ __forceinline__ float q6_q8_row_dot(
    const uint8_t *__restrict__ row_weights,
    const Q8_1Block *__restrict__ input,
    size_t columns,
    float *__restrict__ warp_partials) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int input_index = lane % 8;
    const size_t blocks_per_row = columns / kQuantBlockElements;
    float sum = 0.0f;
    for (size_t quant_block = warp;
         quant_block < blocks_per_row;
         quant_block += kGemvWarpsPerBlock) {
        const uint8_t *source =
            row_weights + quant_block * kQ6BlockBytes;
        const int q8_offset = 4 * (lane / 16) + (lane % 16) / 8;
        const int scale_offset = 8 * (lane / 16) + (lane % 16) / 4;
        const int high_shift = 2 * ((lane % 16) / 8);
        const int low = load_i32_aligned2(source, lane);
        const int high_index = 8 * (lane / 16) + lane % 8;
        const int high =
            load_i32_aligned2(source + 128, high_index) >> high_shift;
        float block_sum = 0.0f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int low_codes =
                (low >> (4 * half)) & 0x0f0f0f0f;
            const int high_codes =
                ((high >> (4 * half)) << 4) & 0x30303030;
            const int codes =
                __vsubss4(low_codes | high_codes, 0x20202020);
            const Q8_1Block *q8 =
                input + quant_block * 8 + q8_offset + 2 * half;
            const int q8_values =
                reinterpret_cast<const int *>(q8->qs)[input_index];
            const int scale = static_cast<int8_t>(
                __ldcs(source + 192 + scale_offset + 4 * half));
            block_sum += __low2float(q8->ds) *
                         static_cast<float>(
                             __dp4a(codes, q8_values, 0) * scale);
        }
        const uint16_t block_scale = __ldcs(
            reinterpret_cast<const uint16_t *>(source + 208));
        __half_raw block_scale_raw;
        block_scale_raw.x = block_scale;
        sum += __half2float(block_scale_raw) * block_sum;
    }
    if (warp > 0) {
        warp_partials[(warp - 1) * 32 + lane] = sum;
    }
    __syncthreads();
    if (warp > 0) {
        return sum;
    }
#pragma unroll
    for (int source_warp = 0; source_warp < kGemvWarpsPerBlock - 1;
         ++source_warp) {
        sum += warp_partials[source_warp * 32 + lane];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

template <int Positions>
__global__ void q6_q8_gemv_multi_warp(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    const float *__restrict__ residual,
    float *__restrict__ output,
    size_t rows,
    size_t columns) {
    __shared__ float warp_partials[Positions][kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int input_index = lane % 8;
    const size_t row = blockIdx.x;
    const size_t blocks_per_row = columns / kQuantBlockElements;
    const size_t input_blocks = columns / kQ8BlockElements;
    const uint8_t *row_weights =
        weights + row * blocks_per_row * kQ6BlockBytes;
    float sums[Positions] = {};
    for (size_t quant_block = warp;
         quant_block < blocks_per_row;
         quant_block += kGemvWarpsPerBlock) {
        const uint8_t *source = row_weights + quant_block * kQ6BlockBytes;
        const int q8_offset = 4 * (lane / 16) + (lane % 16) / 8;
        const int scale_offset = 8 * (lane / 16) + (lane % 16) / 4;
        const int high_shift = 2 * ((lane % 16) / 8);
        const int low = load_i32_aligned2(source, lane);
        const int high_index = 8 * (lane / 16) + lane % 8;
        const int high =
            load_i32_aligned2(source + 128, high_index) >> high_shift;
        const uint16_t block_scale =
            __ldcs(reinterpret_cast<const uint16_t *>(source + 208));
        __half_raw block_scale_raw;
        block_scale_raw.x = block_scale;
#pragma unroll
        for (int position = 0; position < Positions; ++position) {
            float block_sum = 0.0f;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int low_codes =
                    (low >> (4 * half)) & 0x0f0f0f0f;
                const int high_codes =
                    ((high >> (4 * half)) << 4) & 0x30303030;
                const int codes =
                    __vsubss4(low_codes | high_codes, 0x20202020);
                const Q8_1Block *q8 =
                    input + position * input_blocks + quant_block * 8 +
                    q8_offset + 2 * half;
                const int q8_values =
                    reinterpret_cast<const int *>(q8->qs)[input_index];
                const int scale = static_cast<int8_t>(
                    __ldcs(source + 192 + scale_offset + 4 * half));
                block_sum += __low2float(q8->ds) *
                             static_cast<float>(
                                 __dp4a(codes, q8_values, 0) * scale);
            }
            sums[position] += __half2float(block_scale_raw) * block_sum;
        }
    }
#pragma unroll
    for (int position = 0; position < Positions; ++position) {
        if (warp > 0) {
            warp_partials[position][warp - 1][lane] = sums[position];
        }
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int position = 0; position < Positions; ++position) {
#pragma unroll
        for (int source_warp = 0;
             source_warp < kGemvWarpsPerBlock - 1;
             ++source_warp) {
            sums[position] += warp_partials[position][source_warp][lane];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[position] +=
                __shfl_down_sync(0xffffffff, sums[position], offset);
        }
        if (lane == 0) {
            const size_t index = static_cast<size_t>(position) * rows + row;
            output[index] = residual == nullptr
                                ? sums[position]
                                : sums[position] + residual[index];
        }
    }
}

template <int Positions>
__global__ void quant_gemv_group_multi_warp(
    const uint8_t *__restrict__ first_weights,
    const uint8_t *__restrict__ second_weights,
    const uint8_t *__restrict__ third_weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ first_output,
    float *__restrict__ second_output,
    float *__restrict__ third_output,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t columns,
    int first_q4,
    int second_q4,
    int third_q4) {
    __shared__ float warp_partials[Positions][kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t global_row = blockIdx.x;
    const uint8_t *weights;
    float *output;
    size_t row;
    size_t rows;
    int q4;
    if (global_row < first_rows) {
        weights = first_weights;
        output = first_output;
        row = global_row;
        rows = first_rows;
        q4 = first_q4;
    } else if (global_row < first_rows + second_rows) {
        weights = second_weights;
        output = second_output;
        row = global_row - first_rows;
        rows = second_rows;
        q4 = second_q4;
    } else {
        weights = third_weights;
        output = third_output;
        row = global_row - first_rows - second_rows;
        rows = third_rows;
        q4 = third_q4;
    }
    const size_t blocks_per_row = columns / kQuantBlockElements;
    const size_t input_blocks = columns / kQ8BlockElements;
    float sums[Positions] = {};
    if (q4 != 0) {
        const int thread = warp * 32 + lane;
        const int block_offset = thread / 16;
        const int local = thread % 16;
        const int q8_offset = 2 * (local / 4);
        const int item = local % 4;
        const uint8_t *metadata =
            weights + rows * blocks_per_row * kQ4CodeBytes;
        const uint8_t *row_codes =
            weights + row * blocks_per_row * kQ4CodeBytes;
        const uint8_t *row_metadata =
            metadata + row * blocks_per_row * kQ4MetadataBytes;
        for (size_t quant_block = block_offset;
             quant_block < blocks_per_row;
             quant_block += kGemvWarpsPerBlock * 2) {
            q4_q8_block_dots_multi(
                row_codes, row_metadata, input, input_blocks, quant_block,
                lane, local, q8_offset, item, sums);
        }
    } else {
        const int input_index = lane % 8;
        const uint8_t *row_weights =
            weights + row * blocks_per_row * kQ6BlockBytes;
        for (size_t quant_block = warp;
             quant_block < blocks_per_row;
             quant_block += kGemvWarpsPerBlock) {
            const uint8_t *source =
                row_weights + quant_block * kQ6BlockBytes;
            const int q8_offset = 4 * (lane / 16) + (lane % 16) / 8;
            const int scale_offset = 8 * (lane / 16) + (lane % 16) / 4;
            const int high_shift = 2 * ((lane % 16) / 8);
            const int low = load_i32_aligned2(source, lane);
            const int high_index = 8 * (lane / 16) + lane % 8;
            const int high =
                load_i32_aligned2(source + 128, high_index) >> high_shift;
            const uint16_t block_scale =
                __ldcs(reinterpret_cast<const uint16_t *>(source + 208));
            __half_raw block_scale_raw;
            block_scale_raw.x = block_scale;
#pragma unroll
            for (int position = 0; position < Positions; ++position) {
                float block_sum = 0.0f;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int low_codes =
                        (low >> (4 * half)) & 0x0f0f0f0f;
                    const int high_codes =
                        ((high >> (4 * half)) << 4) & 0x30303030;
                    const int codes =
                        __vsubss4(low_codes | high_codes, 0x20202020);
                    const Q8_1Block *q8 =
                        input + position * input_blocks + quant_block * 8 +
                        q8_offset + 2 * half;
                    const int q8_values =
                        reinterpret_cast<const int *>(q8->qs)[input_index];
                    const int scale = static_cast<int8_t>(
                        __ldcs(source + 192 + scale_offset + 4 * half));
                    block_sum += __low2float(q8->ds) *
                                 static_cast<float>(
                                     __dp4a(codes, q8_values, 0) * scale);
                }
                sums[position] +=
                    __half2float(block_scale_raw) * block_sum;
            }
        }
    }
#pragma unroll
    for (int position = 0; position < Positions; ++position) {
        if (warp > 0) {
            warp_partials[position][warp - 1][lane] = sums[position];
        }
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int position = 0; position < Positions; ++position) {
#pragma unroll
        for (int source_warp = 0;
             source_warp < kGemvWarpsPerBlock - 1;
             ++source_warp) {
            sums[position] +=
                warp_partials[position][source_warp][lane];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[position] +=
                __shfl_down_sync(0xffffffff, sums[position], offset);
        }
        if (lane == 0) {
            output[static_cast<size_t>(position) * rows + row] =
                sums[position];
        }
    }
}

template <int FixedBlocksPerRow, int WarpsPerRow>
__global__ void q4_q8_gemv_warp(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    const float *__restrict__ residual,
    float *__restrict__ output,
    size_t columns) {
    __shared__ float warp_partials[WarpsPerRow - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t row = blockIdx.x;
    const size_t blocks_per_row = FixedBlocksPerRow > 0
                                      ? FixedBlocksPerRow
                                      : columns / kQuantBlockElements;
    const uint8_t *metadata =
        weights + gridDim.x * blocks_per_row * kQ4CodeBytes;
    const float sum = q4_q8_row_dot_long_k<FixedBlocksPerRow, WarpsPerRow>(
        weights + row * blocks_per_row * kQ4CodeBytes,
        metadata + row * blocks_per_row * kQ4MetadataBytes,
        input, columns,
        &warp_partials[0][0]);
    if (warp == 0 && lane == 0) {
        output[row] = residual == nullptr ? sum : sum + residual[row];
    }
}

template <int FixedBlocksPerRow, int WarpsPerRow>
__global__ void q4_q8_gemv_apron_probe(
    const uint8_t *__restrict__ weights,
    const uint8_t *__restrict__ next_weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t columns,
    size_t next_rows,
    size_t apron_bytes) {
    __shared__ float warp_partials[WarpsPerRow - 1][32];
    __shared__ uint32_t prefetch_partials[WarpsPerRow];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t row = blockIdx.x;
    if (row < rows) {
        const size_t blocks_per_row = FixedBlocksPerRow > 0
                                          ? FixedBlocksPerRow
                                          : columns / kQuantBlockElements;
        const uint8_t *metadata =
            weights + rows * blocks_per_row * kQ4CodeBytes;
        const float sum =
            q4_q8_row_dot_long_k<FixedBlocksPerRow, WarpsPerRow>(
                weights + row * blocks_per_row * kQ4CodeBytes,
                metadata + row * blocks_per_row * kQ4MetadataBytes,
                input, columns, &warp_partials[0][0]);
        if (warp == 0 && lane == 0) {
            output[row] = sum;
        }
        return;
    }

    const size_t prefetch_block = row - rows;
    const size_t thread = threadIdx.x;
    const size_t vector = prefetch_block * blockDim.x + thread;
    const size_t vector_stride = kQ4ApronBlocks * blockDim.x;
    uint32_t loaded = 0;
    for (size_t offset = vector * sizeof(uint4);
         offset < apron_bytes;
         offset += vector_stride * sizeof(uint4)) {
        constexpr size_t trip_code_bytes = 8 * kQ4CodeBytes;
        constexpr size_t trip_metadata_bytes = 8 * kQ4MetadataBytes;
        constexpr size_t trip_bytes =
            trip_code_bytes + trip_metadata_bytes;
        constexpr size_t row_code_bytes = 16 * kQ4CodeBytes;
        constexpr size_t row_metadata_bytes = 16 * kQ4MetadataBytes;
        const size_t next_row = offset / trip_bytes;
        const size_t packet_offset = offset - next_row * trip_bytes;
        const uint8_t *source = packet_offset < trip_code_bytes
                                    ? next_weights +
                                          next_row * row_code_bytes +
                                          packet_offset
                                    : next_weights +
                                          next_rows * row_code_bytes +
                                          next_row * row_metadata_bytes +
                                          packet_offset - trip_code_bytes;
        const uint4 value = load_l2_evict_last(source);
        loaded ^= value.x ^ value.y ^ value.z ^ value.w;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        loaded ^= __shfl_xor_sync(0xffffffff, loaded, offset);
    }
    if (lane == 0) {
        prefetch_partials[warp] = loaded;
    }
    __syncthreads();
    if (warp == 0) {
        loaded = lane < WarpsPerRow ? prefetch_partials[lane] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            loaded ^= __shfl_xor_sync(0xffffffff, loaded, offset);
        }
        if (lane == 0) {
            reinterpret_cast<uint32_t *>(output + rows)[prefetch_block] =
                loaded;
        }
    }
}

template <int FixedBlocksPerRow, int WarpsPerRow>
__global__ void q4_q8_gemv_ring_probe(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ output,
    size_t rows,
    size_t weight_sets,
    size_t columns) {
    __shared__ float warp_partials[WarpsPerRow - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t blocks_per_row = FixedBlocksPerRow > 0
                                      ? FixedBlocksPerRow
                                      : columns / kQuantBlockElements;
    const size_t code_bytes = rows * blocks_per_row * kQ4CodeBytes;
    const size_t matrix_bytes =
        code_bytes + rows * blocks_per_row * kQ4MetadataBytes;
    const size_t total_rows = rows * weight_sets;
    for (size_t linear_row = blockIdx.x;
         linear_row < total_rows;
         linear_row += gridDim.x) {
        const size_t weight_set = linear_row / rows;
        const size_t row = linear_row - weight_set * rows;
        const uint8_t *matrix = weights + weight_set * matrix_bytes;
        const float sum = q4_q8_row_dot_long_k<FixedBlocksPerRow,
                                                  WarpsPerRow>(
            matrix + row * blocks_per_row * kQ4CodeBytes,
            matrix + code_bytes + row * blocks_per_row * kQ4MetadataBytes,
            input, columns, &warp_partials[0][0]);
        if (warp == 0 && lane == 0) {
            output[linear_row] = sum;
        }
        // The next row can overwrite shared partials only after warp 0 has
        // consumed every partial for the current row.
        __syncthreads();
    }
}

__global__ void q6_q8_gemv_warp(
    const uint8_t *__restrict__ weights,
    const Q8_1Block *__restrict__ input,
    const float *__restrict__ residual,
    float *__restrict__ output,
    size_t columns) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t row = blockIdx.x;
    const size_t row_bytes =
        columns / kQuantBlockElements * kQ6BlockBytes;
    const float sum = q6_q8_row_dot(
        weights + row * row_bytes, input, columns,
        &warp_partials[0][0]);
    if (warp == 0 && lane == 0) {
        output[row] = residual == nullptr ? sum : sum + residual[row];
    }
}

template <int FixedBlocksPerRow>
__global__ void q4_q8_gemv_pair_warp(
    const uint8_t *__restrict__ first_weights,
    const uint8_t *__restrict__ second_weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ first_output,
    float *__restrict__ second_output,
    size_t first_rows,
    size_t second_rows,
    size_t columns) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t global_row = blockIdx.x;
    const bool first = global_row < first_rows;
    const size_t row = first ? global_row : global_row - first_rows;
    const uint8_t *weights = first ? first_weights : second_weights;
    float *output = first ? first_output : second_output;
    const size_t rows = first ? first_rows : second_rows;
    const size_t blocks_per_row = FixedBlocksPerRow > 0
                                      ? FixedBlocksPerRow
                                      : columns / kQuantBlockElements;
    const uint8_t *metadata =
        weights + rows * blocks_per_row * kQ4CodeBytes;
    const float sum = q4_q8_row_dot_long_k<FixedBlocksPerRow,
                                              kGemvWarpsPerBlock>(
        weights + row * blocks_per_row * kQ4CodeBytes,
        metadata + row * blocks_per_row * kQ4MetadataBytes,
        input, columns,
        &warp_partials[0][0]);
    if (warp == 0 && lane == 0) {
        output[row] = sum;
    }
}

template <int FixedBlocksPerRow>
__global__ void q4_q8_gemv_swiglu_warp(
    const uint8_t *__restrict__ gate_weights,
    const uint8_t *__restrict__ up_weights,
    const Q8_1Block *__restrict__ input,
    float *__restrict__ gate_output,
    float *__restrict__ up_output,
    float *__restrict__ output,
    Q8_1Block *__restrict__ quantized_output,
    int32_t *__restrict__ quantized_sums,
    uint32_t *__restrict__ epilogue_ready,
    size_t rows,
    size_t columns) {
    __shared__ float warp_partials[2][kGemvWarpsPerBlock - 1][32];
    __shared__ int quantize_block;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t row = blockIdx.x;
    const size_t blocks_per_row = FixedBlocksPerRow > 0
                                      ? FixedBlocksPerRow
                                      : columns / kQuantBlockElements;
    const size_t code_bytes = rows * blocks_per_row * kQ4CodeBytes;
    const float2 sums = q4_q8_two_row_dots_long_k<
        FixedBlocksPerRow, kGemvWarpsPerBlock>(
        gate_weights + row * blocks_per_row * kQ4CodeBytes,
        gate_weights + code_bytes + row * blocks_per_row * kQ4MetadataBytes,
        up_weights + row * blocks_per_row * kQ4CodeBytes,
        up_weights + code_bytes + row * blocks_per_row * kQ4MetadataBytes,
        input, columns, &warp_partials[0][0][0],
        &warp_partials[1][0][0]);
    if (threadIdx.x == 0) {
        gate_output[row] = sums.x;
        up_output[row] = sums.y;
        const float value = sums.x / (1.0f + expf(-sums.x)) * sums.y;
        output[row] = value;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(
            epilogue_ready[row / 32]);
        quantize_block =
            ready.fetch_add(1, cuda::memory_order_acq_rel) == 31;
    }
    __syncthreads();
    if (quantize_block == 0 || warp != 0) {
        return;
    }

    const size_t q8_block = row / 32;
    const float value = output[q8_block * 32 + lane];
    float maximum = fabsf(value);
    float source_sum = value;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffff, maximum, offset));
        source_sum +=
            __shfl_down_sync(0xffffffff, source_sum, offset);
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum / 127.0f;
    const int quant = maximum == 0.0f
                          ? 0
                          : __float2int_rn(value / scale);
    quantized_output[q8_block].qs[lane] = static_cast<int8_t>(quant);
    int quantized_sum = quant;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        quantized_sum +=
            __shfl_down_sync(0xffffffff, quantized_sum, offset);
    }
    if (lane == 0) {
        quantized_output[q8_block].ds =
            __floats2half2_rn(scale, source_sum);
        quantized_sums[q8_block] = quantized_sum;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(
            epilogue_ready[q8_block]);
        ready.store(0, cuda::memory_order_release);
    }
}

template <int FixedBlocksPerRow>
__global__ void quant_qkv_gemv_warp(
    const uint8_t *__restrict__ query_weights,
    const uint8_t *__restrict__ key_weights,
    const uint8_t *__restrict__ value_weights,
    const float *__restrict__ input,
    const Q8_1Block *__restrict__ quantized_input,
    float *__restrict__ query,
    float *__restrict__ key,
    float *__restrict__ value,
    size_t query_rows,
    size_t key_rows,
    size_t value_rows,
    size_t columns,
    int query_q4,
    int key_q4,
    int value_q4) {
    __shared__ float warp_partials[kGemvWarpsPerBlock - 1][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t global_row = blockIdx.x;

    const uint8_t *weights;
    float *output;
    size_t row;
    size_t rows;
    int q4;
    if (global_row < query_rows) {
        weights = query_weights;
        output = query;
        row = global_row;
        rows = query_rows;
        q4 = query_q4;
    } else if (global_row < query_rows + key_rows) {
        weights = key_weights;
        output = key;
        row = global_row - query_rows;
        rows = key_rows;
        q4 = key_q4;
    } else {
        weights = value_weights;
        output = value;
        row = global_row - query_rows - key_rows;
        rows = value_rows;
        q4 = value_q4;
    }

    const size_t blocks_per_row = FixedBlocksPerRow > 0
                                      ? FixedBlocksPerRow
                                      : columns / kQuantBlockElements;
    float sum = 0.0f;
    if (q4 != 0) {
        const uint8_t *metadata =
            weights + rows * blocks_per_row * kQ4CodeBytes;
        sum = q4_q8_row_dot_long_k<FixedBlocksPerRow,
                                       kGemvWarpsPerBlock>(
            weights + row * blocks_per_row * kQ4CodeBytes,
            metadata + row * blocks_per_row * kQ4MetadataBytes,
            quantized_input, columns, &warp_partials[0][0]);
    } else {
        const uint8_t *row_weights =
            weights + row * blocks_per_row * kQ6BlockBytes;
        sum = q6_q8_row_dot(
            row_weights, quantized_input, columns,
            &warp_partials[0][0]);
    }
    if (warp == 0 && lane == 0) {
        output[row] = sum;
    }
}

template <bool Residual, bool StoreResidual>
__global__ void rms_norm_kernel(const float *__restrict__ left,
                                const float *__restrict__ right,
                                const float *__restrict__ weight,
                                float *__restrict__ stored_residual,
                                float *__restrict__ output,
                                size_t rows,
                                size_t columns,
                                float epsilon) {
    __shared__ float reduction[kBlockThreads];
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const size_t base = row * columns;
    float square_sum = 0.0f;
    for (size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        float value = left[base + column];
        if constexpr (Residual) {
            value += right[base + column];
        }
        square_sum = fmaf(value, value, square_sum);
    }
    reduction[threadIdx.x] = square_sum;
    __syncthreads();
    if (threadIdx.x < 128) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = reduction[threadIdx.x] + reduction[threadIdx.x + 32];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (threadIdx.x == 0) {
            reduction[0] = value;
        }
    }
    __syncthreads();
    const float inverse_rms =
        1.0f / sqrtf(reduction[0] / static_cast<float>(columns) + epsilon);
    for (size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        float value = left[base + column];
        if constexpr (Residual) {
            value += right[base + column];
        }
        if constexpr (StoreResidual) {
            stored_residual[base + column] = value;
        }
        output[base + column] = value * weight[column] * inverse_rms;
    }
}

template <bool DevicePosition>
__global__ void prepare_rope_table_kernel(
    const double *__restrict__ inverse_frequencies,
    float2 *__restrict__ table,
    size_t pairs,
    size_t host_position,
    const uint32_t *__restrict__ device_position) {
    const size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) {
        return;
    }
    const size_t position = DevicePosition
                                ? static_cast<size_t>(device_position[0])
                                : host_position;
    const double angle = static_cast<double>(position) * inverse_frequencies[pair];
    double sine;
    double cosine;
    sincos(angle, &sine, &cosine);
    table[pair] = make_float2(static_cast<float>(cosine),
                              static_cast<float>(sine));
}

__global__ void prepare_verify_rope_tables_kernel(
    const double *__restrict__ inverse_frequencies,
    float2 *__restrict__ tables,
    size_t pairs,
    size_t start_position,
    size_t positions) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = pairs * positions;
    if (index >= elements) {
        return;
    }
    const size_t position = index / pairs;
    const size_t pair = index - position * pairs;
    const double angle = static_cast<double>(start_position + position) *
                         inverse_frequencies[pair];
    double sine;
    double cosine;
    sincos(angle, &sine, &cosine);
    tables[index] = make_float2(static_cast<float>(cosine),
                                static_cast<float>(sine));
}

template <bool DevicePosition, bool Precomputed, bool AppendKv, bool F16Cache>
__global__ void rms_norm_rope_kernel(const float *__restrict__ input,
                                     const float *__restrict__ weight,
                                     float *__restrict__ output,
                                     size_t rows,
                                     const float *__restrict__ second_input,
                                     const float *__restrict__ second_weight,
                                     float *__restrict__ second_output,
                                     size_t second_rows,
                                     size_t columns,
                                     size_t host_position,
                                     const uint32_t *__restrict__ device_position,
                                     const float2 *__restrict__ rope_table,
                                     const float *__restrict__ value_input,
                                     void *__restrict__ key_cache,
                                     void *__restrict__ value_cache,
                                     size_t max_context,
                                     float epsilon,
                                     float theta) {
    __shared__ float reduction[kBlockThreads];
    const size_t global_row = blockIdx.x;
    if (global_row >= rows + second_rows) {
        return;
    }
    const bool use_second = global_row >= rows;
    const size_t row = use_second ? global_row - rows : global_row;
    const float *selected_input = use_second ? second_input : input;
    const float *selected_weight = use_second ? second_weight : weight;
    float *selected_output = use_second ? second_output : output;
    const size_t base = row * columns;
    float square_sum = 0.0f;
    for (size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float value = selected_input[base + column];
        square_sum = fmaf(value, value, square_sum);
    }
    reduction[threadIdx.x] = square_sum;
    __syncthreads();
    if (threadIdx.x < 128) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = reduction[threadIdx.x] + reduction[threadIdx.x + 32];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (threadIdx.x == 0) {
            reduction[0] = value;
        }
    }
    __syncthreads();
    const float inverse_rms =
        1.0f / sqrtf(reduction[0] / static_cast<float>(columns) + epsilon);
    const size_t pair = threadIdx.x;
    const size_t half = columns / 2;
    if (pair < half) {
        const float first = selected_input[base + pair] *
                            selected_weight[pair] * inverse_rms;
        const float second =
            selected_input[base + pair + half] *
            selected_weight[pair + half] * inverse_rms;
        float rotated_first;
        float rotated_second;
        if constexpr (Precomputed) {
            const float2 cosine_sine = rope_table[pair];
            rotated_first =
                first * cosine_sine.x - second * cosine_sine.y;
            rotated_second =
                first * cosine_sine.y + second * cosine_sine.x;
        } else {
            const size_t position = DevicePosition
                                        ? static_cast<size_t>(device_position[0])
                                        : host_position;
            const double exponent = -2.0 * static_cast<double>(pair) /
                                    static_cast<double>(columns);
            const double angle = static_cast<double>(position) *
                                 pow(static_cast<double>(theta), exponent);
            double sine;
            double cosine;
            sincos(angle, &sine, &cosine);
            rotated_first = static_cast<float>(
                static_cast<double>(first) * cosine -
                static_cast<double>(second) * sine);
            rotated_second = static_cast<float>(
                static_cast<double>(first) * sine +
                static_cast<double>(second) * cosine);
        }
        selected_output[base + pair] = rotated_first;
        selected_output[base + pair + half] = rotated_second;
        if constexpr (AppendKv) {
            if (use_second) {
                const size_t position = DevicePosition
                                            ? static_cast<size_t>(device_position[0])
                                            : host_position;
                const size_t cache_base =
                    (row * max_context + position) * columns;
                if constexpr (F16Cache) {
                    static_cast<__half *>(key_cache)[cache_base + pair] =
                        __float2half_rn(rotated_first);
                    static_cast<__half *>(key_cache)[cache_base + pair + half] =
                        __float2half_rn(rotated_second);
                } else {
                    static_cast<float *>(key_cache)[cache_base + pair] =
                        rotated_first;
                    static_cast<float *>(key_cache)[cache_base + pair + half] =
                        rotated_second;
                }
            }
        }
    }
    if constexpr (AppendKv) {
        if (use_second && threadIdx.x < columns) {
            const size_t position = DevicePosition
                                        ? static_cast<size_t>(device_position[0])
                                        : host_position;
            const size_t cache_index =
                (row * max_context + position) * columns + threadIdx.x;
            if constexpr (F16Cache) {
                static_cast<__half *>(value_cache)[cache_index] =
                    __float2half_rn(value_input[base + threadIdx.x]);
            } else {
                static_cast<float *>(value_cache)[cache_index] =
                    value_input[base + threadIdx.x];
            }
        }
    }
}

template <bool F16Cache>
__global__ void rms_norm_rope_verify_kernel(
    const float *__restrict__ query,
    const float *__restrict__ query_weight,
    float *__restrict__ query_output,
    size_t query_rows,
    const float *__restrict__ key,
    const float *__restrict__ key_weight,
    float *__restrict__ key_output,
    size_t key_rows,
    const float *__restrict__ value,
    void *__restrict__ key_cache,
    void *__restrict__ value_cache,
    size_t columns,
    size_t max_context,
    size_t start_position,
    size_t positions,
    const float2 *__restrict__ rope_tables,
    float epsilon) {
    __shared__ float reduction[kBlockThreads];
    const size_t rows_per_position = query_rows + key_rows;
    const size_t global_row = blockIdx.x;
    const size_t position_index = global_row / rows_per_position;
    if (position_index >= positions) {
        return;
    }
    const size_t position_row = global_row - position_index * rows_per_position;
    const bool use_key = position_row >= query_rows;
    const size_t row = use_key ? position_row - query_rows : position_row;
    const float *selected_input = use_key
                                      ? key + position_index * key_rows * columns
                                      : query + position_index * query_rows * columns;
    const float *selected_weight = use_key ? key_weight : query_weight;
    float *selected_output = use_key
                                 ? key_output + position_index * key_rows * columns
                                 : query_output + position_index * query_rows * columns;
    const size_t base = row * columns;
    float square_sum = 0.0f;
    for (size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float item = selected_input[base + column];
        square_sum = fmaf(item, item, square_sum);
    }
    reduction[threadIdx.x] = square_sum;
    __syncthreads();
    if (threadIdx.x < 128) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float item = reduction[threadIdx.x] + reduction[threadIdx.x + 32];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            item += __shfl_down_sync(0xffffffff, item, offset);
        }
        if (threadIdx.x == 0) {
            reduction[0] = item;
        }
    }
    __syncthreads();
    const float inverse_rms =
        1.0f / sqrtf(reduction[0] / static_cast<float>(columns) + epsilon);
    const size_t pair = threadIdx.x;
    const size_t half = columns / 2;
    if (pair < half) {
        const float first = selected_input[base + pair] *
                            selected_weight[pair] * inverse_rms;
        const float second = selected_input[base + pair + half] *
                             selected_weight[pair + half] * inverse_rms;
        const float2 cosine_sine =
            rope_tables[position_index * half + pair];
        const float rotated_first =
            first * cosine_sine.x - second * cosine_sine.y;
        const float rotated_second =
            first * cosine_sine.y + second * cosine_sine.x;
        selected_output[base + pair] = rotated_first;
        selected_output[base + pair + half] = rotated_second;
        if (use_key) {
            const size_t position = start_position + position_index;
            const size_t cache_base =
                (row * max_context + position) * columns;
            if constexpr (F16Cache) {
                static_cast<__half *>(key_cache)[cache_base + pair] =
                    __float2half_rn(rotated_first);
                static_cast<__half *>(key_cache)[cache_base + pair + half] =
                    __float2half_rn(rotated_second);
            } else {
                static_cast<float *>(key_cache)[cache_base + pair] =
                    rotated_first;
                static_cast<float *>(key_cache)[cache_base + pair + half] =
                    rotated_second;
            }
        }
    }
    if (use_key && threadIdx.x < columns) {
        const size_t position = start_position + position_index;
        const size_t cache_index =
            (row * max_context + position) * columns + threadIdx.x;
        const size_t value_index =
            (position_index * key_rows + row) * columns + threadIdx.x;
        if constexpr (F16Cache) {
            static_cast<__half *>(value_cache)[cache_index] =
                __float2half_rn(value[value_index]);
        } else {
            static_cast<float *>(value_cache)[cache_index] = value[value_index];
        }
    }
}

__global__ void rms_norm_q8_parallel_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float *__restrict__ output,
    Q8_1Block *__restrict__ quantized_output,
    int32_t *__restrict__ quantized_sums,
    size_t columns,
    float epsilon) {
    __shared__ float reduction[kBlockThreads];
    const size_t q8_groups = columns / kBlockThreads;
    const size_t row = blockIdx.x / q8_groups;
    const size_t group = blockIdx.x % q8_groups;
    const size_t base = row * columns;
    float square_sum = 0.0f;
    for (size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float value = input[base + column];
        square_sum = fmaf(value, value, square_sum);
    }
    reduction[threadIdx.x] = square_sum;
    __syncthreads();
    if (threadIdx.x < 128) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        reduction[threadIdx.x] += reduction[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = reduction[threadIdx.x] + reduction[threadIdx.x + 32];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (threadIdx.x == 0) {
            reduction[0] = value;
        }
    }
    __syncthreads();

    const float inverse_rms =
        1.0f / sqrtf(reduction[0] / static_cast<float>(columns) + epsilon);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t q8_block = group * (kBlockThreads / 32) + warp;
    const size_t column = q8_block * kQ8BlockElements + lane;
    const float value = input[base + column] * weight[column] * inverse_rms;
    output[base + column] = value;

    float maximum = fabsf(value);
    float source_sum = value;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffff, maximum, offset));
        source_sum +=
            __shfl_down_sync(0xffffffff, source_sum, offset);
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum / 127.0f;
    const int quant = maximum == 0.0f
                          ? 0
                          : __float2int_rn(value / scale);
    const size_t output_block = row * (columns / kQ8BlockElements) + q8_block;
    quantized_output[output_block].qs[lane] = static_cast<int8_t>(quant);
    int quantized_sum = quant;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        quantized_sum +=
            __shfl_down_sync(0xffffffff, quantized_sum, offset);
    }
    if (lane == 0) {
        quantized_output[output_block].ds =
            __floats2half2_rn(scale, source_sum);
        quantized_sums[output_block] = quantized_sum;
    }
}

__global__ void rope_neox_kernel(float *__restrict__ values,
                                 const uint32_t *__restrict__ positions,
                                 size_t tokens,
                                 size_t heads,
                                 size_t head_dim,
                                 float theta) {
    const size_t half = head_dim / 2;
    const size_t pairs = tokens * heads * half;
    const size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) {
        return;
    }
    const size_t pair_in_head = pair % half;
    const size_t head_linear = pair / half;
    const size_t token = head_linear / heads;
    const size_t base = head_linear * head_dim;
    const double exponent = -2.0 * static_cast<double>(pair_in_head) /
                            static_cast<double>(head_dim);
    const double angle = static_cast<double>(positions[token]) *
                         pow(static_cast<double>(theta), exponent);
    double sine;
    double cosine;
    sincos(angle, &sine, &cosine);
    const float first = values[base + pair_in_head];
    const float second = values[base + pair_in_head + half];
    values[base + pair_in_head] = static_cast<float>(
        static_cast<double>(first) * cosine - static_cast<double>(second) * sine);
    values[base + pair_in_head + half] = static_cast<float>(
        static_cast<double>(first) * sine + static_cast<double>(second) * cosine);
}

__global__ void rope_neox_at_kernel(float *__restrict__ values,
                                    size_t position,
                                    size_t tokens,
                                    size_t heads,
                                    size_t head_dim,
                                    float theta) {
    const size_t half = head_dim / 2;
    const size_t pairs = tokens * heads * half;
    const size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) {
        return;
    }
    const size_t pair_in_head = pair % half;
    const size_t head_linear = pair / half;
    const size_t token = head_linear / heads;
    const size_t base = head_linear * head_dim;
    const double exponent = -2.0 * static_cast<double>(pair_in_head) /
                            static_cast<double>(head_dim);
    const double angle = static_cast<double>(position + token) *
                         pow(static_cast<double>(theta), exponent);
    double sine;
    double cosine;
    sincos(angle, &sine, &cosine);
    const float first = values[base + pair_in_head];
    const float second = values[base + pair_in_head + half];
    values[base + pair_in_head] = static_cast<float>(
        static_cast<double>(first) * cosine - static_cast<double>(second) * sine);
    values[base + pair_in_head + half] = static_cast<float>(
        static_cast<double>(first) * sine + static_cast<double>(second) * cosine);
}

__global__ void rope_at_frequencies_kernel(
    float *__restrict__ values,
    size_t position,
    size_t tokens,
    size_t heads,
    size_t head_dim,
    const double *__restrict__ inverse_frequencies,
    bool adjacent_pairs) {
    const size_t half = head_dim / 2;
    const size_t pairs = tokens * heads * half;
    const size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) {
        return;
    }
    const size_t pair_in_head = pair % half;
    const size_t head_linear = pair / half;
    const size_t token = head_linear / heads;
    const size_t base = head_linear * head_dim;
    const size_t first_index = base +
        (adjacent_pairs ? pair_in_head * 2 : pair_in_head);
    const size_t second_index = adjacent_pairs ? first_index + 1
                                                : first_index + half;
    const double angle = static_cast<double>(position + token) *
                         inverse_frequencies[pair_in_head];
    double sine;
    double cosine;
    sincos(angle, &sine, &cosine);
    const float first = values[first_index];
    const float second = values[second_index];
    values[first_index] = static_cast<float>(
        static_cast<double>(first) * cosine - static_cast<double>(second) * sine);
    values[second_index] = static_cast<float>(
        static_cast<double>(first) * sine + static_cast<double>(second) * cosine);
}

__global__ void swiglu_kernel(const float *__restrict__ gate,
                              const float *__restrict__ up,
                              float *__restrict__ output,
                              size_t elements) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float value = gate[index];
        const float silu = value / (1.0f + expf(-value));
        output[index] = silu * up[index];
    }
}

__global__ void swiglu_q8_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    float *__restrict__ output,
    Q8_1Block *__restrict__ quantized_output,
    int32_t *__restrict__ quantized_sums,
    size_t elements) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const float gate_value = index < elements ? gate[index] : 0.0f;
    const float up_value = index < elements ? up[index] : 0.0f;
    const float value =
        gate_value / (1.0f + expf(-gate_value)) * up_value;
    if (index < elements) {
        output[index] = value;
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float maximum = fabsf(value);
    float source_sum = value;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffff, maximum, offset));
        source_sum +=
            __shfl_down_sync(0xffffffff, source_sum, offset);
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum / 127.0f;
    const int quant = maximum == 0.0f
                          ? 0
                          : __float2int_rn(value / scale);
    const size_t q8_block =
        blockIdx.x * (kBlockThreads / 32) + warp;
    quantized_output[q8_block].qs[lane] = static_cast<int8_t>(quant);
    int quantized_sum = quant;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        quantized_sum +=
            __shfl_down_sync(0xffffffff, quantized_sum, offset);
    }
    if (lane == 0) {
        quantized_output[q8_block].ds =
            __floats2half2_rn(scale, source_sum);
        quantized_sums[q8_block] = quantized_sum;
    }
}

__global__ void residual_add_kernel(const float *__restrict__ left,
                                    const float *__restrict__ right,
                                    float *__restrict__ output,
                                    size_t elements) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = left[index] + right[index];
    }
}

__global__ void write_u32_kernel(uint32_t *__restrict__ output,
                                 uint32_t value) {
    output[0] = value;
}

__global__ void increment_u32_kernel(uint32_t *__restrict__ output) {
    output[0] += 1;
}

template <bool F16, bool DevicePosition>
__global__ void kv_append_kernel(const float *__restrict__ key,
                                 const float *__restrict__ value,
                                 void *__restrict__ key_cache,
                                 void *__restrict__ value_cache,
                                 size_t n_head_kv,
                                 size_t head_dim,
                                 size_t max_context,
                                 size_t host_position,
                                 const uint32_t *__restrict__ device_position) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = n_head_kv * head_dim;
    if (index < elements) {
        const size_t position = DevicePosition
                                    ? static_cast<size_t>(device_position[0])
                                    : host_position;
        // The host path checks the position in Rust. A device position lives
        // on the device and cannot be checked there, so guard the store here.
        // The caller keeps position below max_context; this only stops an
        // out-of-bounds write if that invariant ever breaks.
        if (position >= max_context) {
            return;
        }
        const size_t head = index / head_dim;
        const size_t dimension = index % head_dim;
        const size_t cache_index =
            (head * max_context + position) * head_dim + dimension;
        if constexpr (F16) {
            static_cast<__half *>(key_cache)[cache_index] =
                __float2half_rn(key[index]);
            static_cast<__half *>(value_cache)[cache_index] =
                __float2half_rn(value[index]);
        } else {
            static_cast<float *>(key_cache)[cache_index] = key[index];
            static_cast<float *>(value_cache)[cache_index] = value[index];
        }
    }
}

template <bool DevicePosition>
__global__ void kv_append_q8_kernel(
    const float *__restrict__ key,
    const float *__restrict__ value,
    Q8KVBlock *__restrict__ key_cache,
    Q8KVBlock *__restrict__ value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t host_position,
    const uint32_t *__restrict__ device_position) {
    const size_t warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const size_t source_block =
        blockIdx.x * (blockDim.x / 32) + warp;
    const size_t blocks_per_head = head_dim / kQ8BlockElements;
    const size_t blocks = n_head_kv * blocks_per_head;
    if (source_block >= blocks) {
        return;
    }
    const size_t position = DevicePosition
                                ? static_cast<size_t>(device_position[0])
                                : host_position;
    if (position >= max_context) {
        return;
    }
    const size_t head = source_block / blocks_per_head;
    const size_t block_in_head = source_block % blocks_per_head;
    const size_t source = source_block * kQ8BlockElements + lane;
    float key_max = fabsf(key[source]);
    float value_max = fabsf(value[source]);
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        key_max = fmaxf(key_max, __shfl_down_sync(0xffffffff, key_max, offset));
        value_max = fmaxf(value_max, __shfl_down_sync(0xffffffff, value_max, offset));
    }
    key_max = __shfl_sync(0xffffffff, key_max, 0);
    value_max = __shfl_sync(0xffffffff, value_max, 0);
    const __half key_scale_half = __float2half_rn(key_max / 127.0f);
    const __half value_scale_half = __float2half_rn(value_max / 127.0f);
    const float key_scale = __half2float(key_scale_half);
    const float value_scale = __half2float(value_scale_half);
    const size_t cache_block =
        (head * max_context + position) * blocks_per_head + block_in_head;
    key_cache[cache_block].qs[lane] = key_scale == 0.0f
        ? 0
        : static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, rintf(key[source] / key_scale))));
    value_cache[cache_block].qs[lane] = value_scale == 0.0f
        ? 0
        : static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, rintf(value[source] / value_scale))));
    if (lane == 0) {
        key_cache[cache_block].d = key_scale_half;
        value_cache[cache_block].d = value_scale_half;
    }
}

template <bool Q4>
__global__ void embedding_kernel(const uint8_t *__restrict__ table,
                                 const uint32_t *__restrict__ device_row,
                                 float *__restrict__ output,
                                 size_t rows,
                                 size_t columns,
                                 size_t host_row) {
    const size_t row = device_row == nullptr ? host_row : device_row[0];
    if (row >= rows) {
        return;
    }
    const size_t blocks_per_row = columns / kQuantBlockElements;
    for (size_t quant_block = 0; quant_block < blocks_per_row; ++quant_block) {
        if constexpr (Q4) {
            const size_t linear_block = row * blocks_per_row + quant_block;
            const uint8_t *codes = table + linear_block * kQ4CodeBytes;
            const uint8_t *metadata =
                table + rows * blocks_per_row * kQ4CodeBytes +
                linear_block * kQ4MetadataBytes;
            output[quant_block * kQuantBlockElements + threadIdx.x] =
                dequant_q4_repacked_value(codes, metadata, threadIdx.x);
        } else {
            __shared__ __align__(16) uint8_t block[kQ6BlockBytes];
            const uint8_t *row_data =
                table + row * blocks_per_row * kQ6BlockBytes;
            const uint8_t *source =
                row_data + quant_block * kQ6BlockBytes;
            load_q6_block(source, block);
            __syncthreads();
            output[quant_block * kQuantBlockElements + threadIdx.x] =
                dequant_q6_value(block, threadIdx.x);
            __syncthreads();
        }
    }
}

template <bool Q4>
__global__ void dequant_k_f16_kernel(const uint8_t *__restrict__ weights,
                                     __half *__restrict__ output,
                                     size_t elements) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const size_t block_index = index / kQuantBlockElements;
    const int within = static_cast<int>(index % kQuantBlockElements);
    float value;
    if constexpr (Q4) {
        const size_t blocks = elements / kQuantBlockElements;
        const uint8_t *codes = weights + block_index * kQ4CodeBytes;
        const uint8_t *metadata = weights + blocks * kQ4CodeBytes +
                                  block_index * kQ4MetadataBytes;
        value = dequant_q4_repacked_value(codes, metadata, within);
    } else {
        const uint8_t *block = weights + block_index * kQ6BlockBytes;
        value = dequant_q6_value(block, within);
    }
    output[index] = __float2half_rn(value);
}

__global__ void f32_to_f16_kernel(const float *__restrict__ input,
                                  __half *__restrict__ output,
                                  size_t elements) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2half_rn(input[index]);
    }
}

__global__ void prefill_query_to_head_kernel(
    const float *__restrict__ query,
    __half *__restrict__ output,
    size_t tokens,
    size_t heads,
    size_t head_dim) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = tokens * heads * head_dim;
    if (index >= elements) {
        return;
    }
    const size_t dimension = index % head_dim;
    const size_t token_head = index / head_dim;
    const size_t head = token_head / tokens;
    const size_t token = token_head % tokens;
    const size_t source = (token * heads + head) * head_dim + dimension;
    output[index] = __float2half_rn(query[source]);
}

__global__ void prefill_cache_to_f16_kernel(
    const float *__restrict__ key_cache,
    const float *__restrict__ value_cache,
    __half *__restrict__ key_output,
    __half *__restrict__ value_output,
    size_t heads,
    size_t head_dim,
    size_t max_context,
    size_t context_length) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = heads * context_length * head_dim;
    if (index >= elements) {
        return;
    }
    const size_t dimension = index % head_dim;
    const size_t position_head = index / head_dim;
    const size_t head = position_head / context_length;
    const size_t position = position_head % context_length;
    const size_t source =
        (head * max_context + position) * head_dim + dimension;
    key_output[index] = __float2half_rn(key_cache[source]);
    value_output[index] = __float2half_rn(value_cache[source]);
}

__global__ void prefill_causal_softmax_kernel(
    const float *__restrict__ scores,
    __half *__restrict__ probabilities,
    size_t heads,
    size_t tokens,
    size_t context_length,
    size_t start_position) {
    __shared__ float reduction[kPrefillSoftmaxThreads];
    const size_t row = blockIdx.x;
    if (row >= heads * tokens) {
        return;
    }
    const size_t token = row % tokens;
    const size_t causal_length = start_position + token + 1;
    const size_t base = row * context_length;
    float maximum = -CUDART_INF_F;
    for (size_t column = threadIdx.x; column < causal_length;
         column += blockDim.x) {
        maximum = fmaxf(maximum, scores[base + column]);
    }
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (int offset = kPrefillSoftmaxThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] =
                fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    maximum = reduction[0];
    // All threads must capture the maximum before lane zero reuses slot zero
    // for the sum reduction.
    __syncthreads();
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < causal_length;
         column += blockDim.x) {
        sum += expf(scores[base + column] - maximum);
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = kPrefillSoftmaxThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / reduction[0];
    for (size_t column = threadIdx.x; column < context_length;
         column += blockDim.x) {
        const float value = column < causal_length
                                ? expf(scores[base + column] - maximum) * inverse_sum
                                : 0.0f;
        probabilities[base + column] = __float2half_rn(value);
    }
}

__global__ void prefill_head_to_token_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t tokens,
    size_t heads,
    size_t head_dim) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = tokens * heads * head_dim;
    if (index >= elements) {
        return;
    }
    const size_t dimension = index % head_dim;
    const size_t token_head = index / head_dim;
    const size_t token = token_head / heads;
    const size_t head = token_head % heads;
    const size_t source = (head * tokens + token) * head_dim + dimension;
    output[index] = input[source];
}
template <bool F16>
__global__ void kv_append_chunk_kernel(
    const float *__restrict__ key,
    const float *__restrict__ value,
    void *__restrict__ key_cache,
    void *__restrict__ value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t tokens) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = tokens * n_head_kv * head_dim;
    if (index >= elements) {
        return;
    }
    const size_t dimension = index % head_dim;
    const size_t token_head = index / head_dim;
    const size_t token = token_head / n_head_kv;
    const size_t head = token_head % n_head_kv;
    const size_t cache_index =
        (head * max_context + start_position + token) * head_dim + dimension;
    if constexpr (F16) {
        static_cast<__half *>(key_cache)[cache_index] =
            __float2half_rn(key[index]);
        static_cast<__half *>(value_cache)[cache_index] =
            __float2half_rn(value[index]);
    } else {
        static_cast<float *>(key_cache)[cache_index] = key[index];
        static_cast<float *>(value_cache)[cache_index] = value[index];
    }
}

template <bool Q4>
__global__ void embedding_batch_kernel(
    const uint8_t *__restrict__ table,
    const uint32_t *__restrict__ rows,
    float *__restrict__ output,
    size_t table_rows,
    size_t columns,
    size_t tokens) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = tokens * columns;
    if (index >= elements) {
        return;
    }
    const size_t token = index / columns;
    const size_t column = index % columns;
    const size_t row = static_cast<size_t>(rows[token]);
    if (row >= table_rows) {
        output[index] = 0.0f;
        return;
    }
    const size_t blocks_per_row = columns / kQuantBlockElements;
    const size_t block_in_row = column / kQuantBlockElements;
    const int within = static_cast<int>(column % kQuantBlockElements);
    const size_t block_index = row * blocks_per_row + block_in_row;
    if constexpr (Q4) {
        const uint8_t *codes = table + block_index * kQ4CodeBytes;
        const uint8_t *metadata = table +
            table_rows * blocks_per_row * kQ4CodeBytes +
            block_index * kQ4MetadataBytes;
        output[index] = dequant_q4_repacked_value(codes, metadata, within);
    } else {
        output[index] = dequant_q6_value(
            table + block_index * kQ6BlockBytes, within);
    }
}

__global__ void copy_f32_row_kernel(const float *__restrict__ input,
                                    float *__restrict__ output,
                                    size_t row,
                                    size_t columns) {
    const size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < columns) {
        output[column] = input[row * columns + column];
    }
}

template <bool F16, bool Q8, bool DevicePosition, bool MultiQuery>
__global__ void attention_partial_kernel(
    const float *__restrict__ query,
    const void *__restrict__ key_cache,
    const void *__restrict__ value_cache,
    float *__restrict__ partial_max,
    float *__restrict__ partial_sum,
    float *__restrict__ partial_output,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t split_count,
    size_t host_context_length,
    const uint32_t *__restrict__ device_position) {
    constexpr int kDotThreads = 8;
    constexpr int kDotGroups = kAttentionThreads / kDotThreads;
    constexpr int kValuesPerLane = 16;
    __shared__ float group_max[kDotGroups];
    __shared__ float group_sum[kDotGroups];
    __shared__ float group_scale[kDotGroups];
    __shared__ float group_output[kDotGroups][kAttentionThreads];
    __shared__ float tile_score[kAttentionKvTile];
    __shared__ float block_max;
    __shared__ float block_sum;
    const size_t query_head = blockIdx.x;
    const size_t split = blockIdx.y;
    const size_t query_index = MultiQuery ? blockIdx.z : 0;
    if (query_head >= n_head) {
        return;
    }
    const size_t context_length = DevicePosition
                                      ? static_cast<size_t>(device_position[0]) + 1
                                      : host_context_length + query_index;
    const size_t group_size = n_head / n_head_kv;
    const size_t kv_head = query_head / group_size;
    const size_t base_count = context_length / split_count;
    const size_t remainder = context_length % split_count;
    const size_t start = split * base_count +
                         (split < remainder ? split : remainder);
    const size_t count = base_count + (split < remainder ? 1 : 0);
    const size_t end = start + count;
    const int group = threadIdx.x / kDotThreads;
    const int lane = threadIdx.x % kDotThreads;
    const unsigned int group_mask =
        0xffu << ((threadIdx.x % 32) / kDotThreads * kDotThreads);
    const float *query_row =
        query + (query_index * n_head + query_head) * head_dim;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float numerator[kValuesPerLane] = {0.0f};
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    float4 query4[4];
    if (head_dim == 128) {
        const float4 *query_vectors = reinterpret_cast<const float4 *>(
            query_row + static_cast<size_t>(lane) * kValuesPerLane);
#pragma unroll
        for (int vector = 0; vector < 4; ++vector) {
            query4[vector] = query_vectors[vector];
        }
    }

    if constexpr (F16) {
        if (head_dim == 128 && count <= kAttentionKvTile) {
            for (size_t local_position = group; local_position < count;
                 local_position += kDotGroups) {
                const size_t position = start + local_position;
                const size_t cache_base =
                    (kv_head * max_context + position) * head_dim;
                const size_t lane_base =
                    static_cast<size_t>(lane) * kValuesPerLane;
                const uint4 *key_vectors = reinterpret_cast<const uint4 *>(
                    static_cast<const __half *>(key_cache) + cache_base +
                    lane_base);
                const uint4 packed0 = key_vectors[0];
                const uint4 packed1 = key_vectors[1];
                const uint32_t packed[8] = {
                    packed0.x, packed0.y, packed0.z, packed0.w,
                    packed1.x, packed1.y, packed1.z, packed1.w,
                };
                float dot = 0.0f;
#pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    __half2_raw raw;
                    raw.x = static_cast<unsigned short>(packed[pair]);
                    raw.y = static_cast<unsigned short>(packed[pair] >> 16);
                    const float2 key2 = __half22float2(raw);
                    const float4 query_values = query4[pair / 2];
                    const float query0 = (pair & 1) == 0
                                             ? query_values.x
                                             : query_values.z;
                    const float query1 = (pair & 1) == 0
                                             ? query_values.y
                                             : query_values.w;
                    dot = fmaf(query0, key2.x, dot);
                    dot = fmaf(query1, key2.y, dot);
                }
#pragma unroll
                for (int offset = 4; offset > 0; offset /= 2) {
                    dot += __shfl_down_sync(group_mask, dot, offset, 8);
                }
                if (lane == 0) {
                    tile_score[local_position] = dot * scale;
                }
            }
            __syncthreads();

            float local_max = -CUDART_INF_F;
            for (size_t local_position = threadIdx.x;
                 local_position < count;
                 local_position += kAttentionThreads) {
                local_max = fmaxf(local_max, tile_score[local_position]);
            }
            group_output[0][threadIdx.x] = local_max;
            __syncthreads();
            for (int offset = kAttentionThreads / 2; offset > 0;
                 offset /= 2) {
                if (threadIdx.x < offset) {
                    group_output[0][threadIdx.x] = fmaxf(
                        group_output[0][threadIdx.x],
                        group_output[0][threadIdx.x + offset]);
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                block_max = group_output[0][0];
            }
            __syncthreads();

            float local_sum = 0.0f;
            for (size_t local_position = threadIdx.x;
                 local_position < count;
                 local_position += kAttentionThreads) {
                const float weight =
                    expf(tile_score[local_position] - block_max);
                tile_score[local_position] = weight;
                local_sum += weight;
            }
            group_output[0][threadIdx.x] = local_sum;
            __syncthreads();
            for (int offset = kAttentionThreads / 2; offset > 0;
                 offset /= 2) {
                if (threadIdx.x < offset) {
                    group_output[0][threadIdx.x] +=
                        group_output[0][threadIdx.x + offset];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                block_sum = group_output[0][0];
            }
            __syncthreads();

            const size_t partial_row =
                (query_index * n_head + query_head) * split_count + split;
            if (threadIdx.x == 0) {
                partial_max[partial_row] = block_max;
                partial_sum[partial_row] = block_sum;
            }
            const size_t dimension = threadIdx.x;
            float block_numerator = 0.0f;
            for (size_t local_position = 0; local_position < count;
                 ++local_position) {
                const size_t position = start + local_position;
                const size_t cache_index =
                    (kv_head * max_context + position) * head_dim + dimension;
                const float value = __half2float(
                    static_cast<const __half *>(value_cache)[cache_index]);
                block_numerator =
                    fmaf(tile_score[local_position], value, block_numerator);
            }
            partial_output[partial_row * head_dim + dimension] =
                block_numerator;
            return;
        }
    }

    for (size_t position = start + group; position < end;
         position += kDotGroups) {
        const size_t cache_base =
            (kv_head * max_context + position) * head_dim;
        float dot = 0.0f;
        if (head_dim == 128) {
            const size_t lane_base =
                static_cast<size_t>(lane) * kValuesPerLane;
            if constexpr (F16) {
                const uint4 *key_vectors = reinterpret_cast<const uint4 *>(
                    static_cast<const __half *>(key_cache) + cache_base +
                    lane_base);
                const uint4 packed0 = key_vectors[0];
                const uint4 packed1 = key_vectors[1];
                const uint32_t packed[8] = {
                    packed0.x, packed0.y, packed0.z, packed0.w,
                    packed1.x, packed1.y, packed1.z, packed1.w,
                };
#pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    __half2_raw raw;
                    raw.x = static_cast<unsigned short>(packed[pair]);
                    raw.y = static_cast<unsigned short>(packed[pair] >> 16);
                    const float2 key2 = __half22float2(raw);
                    const float4 query_values = query4[pair / 2];
                    const float query0 = (pair & 1) == 0
                                             ? query_values.x
                                             : query_values.z;
                    const float query1 = (pair & 1) == 0
                                             ? query_values.y
                                             : query_values.w;
                    dot = fmaf(query0, key2.x, dot);
                    dot = fmaf(query1, key2.y, dot);
                }
            } else if constexpr (Q8) {
#pragma unroll
                for (int item = 0; item < kValuesPerLane; ++item) {
                    dot = fmaf(
                        query_row[lane_base + item],
                        q8_kv_value(key_cache, cache_base + lane_base + item),
                        dot);
                }
            } else {
                const float4 *key_vectors = reinterpret_cast<const float4 *>(
                    static_cast<const float *>(key_cache) + cache_base +
                    lane_base);
#pragma unroll
                for (int vector = 0; vector < 4; ++vector) {
                    const float4 query_values = query4[vector];
                    const float4 key_values = key_vectors[vector];
                    dot = fmaf(query_values.x, key_values.x, dot);
                    dot = fmaf(query_values.y, key_values.y, dot);
                    dot = fmaf(query_values.z, key_values.z, dot);
                    dot = fmaf(query_values.w, key_values.w, dot);
                }
            }
        } else {
            for (size_t index = lane; index < head_dim;
                 index += kDotThreads) {
                const float key_value = Q8
                    ? q8_kv_value(key_cache, cache_base + index)
                    : (F16
                        ? __half2float(static_cast<const __half *>(key_cache)[cache_base + index])
                        : static_cast<const float *>(key_cache)[cache_base + index]);
                dot = fmaf(query_row[index], key_value, dot);
            }
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(group_mask, dot, offset, 8);
        }
        const float score = __shfl_sync(group_mask, dot, 0, 8) * scale;
        const float next_max = fmaxf(running_max, score);
        const float previous_scale = running_sum == 0.0f
                                         ? 0.0f
                                         : expf(running_max - next_max);
        const float score_scale = expf(score - next_max);
        running_sum = running_sum * previous_scale + score_scale;
        if (head_dim == 128) {
            const size_t lane_base =
                static_cast<size_t>(lane) * kValuesPerLane;
            if constexpr (F16) {
                const uint4 *value_vectors = reinterpret_cast<const uint4 *>(
                    static_cast<const __half *>(value_cache) + cache_base +
                    lane_base);
                const uint4 packed0 = value_vectors[0];
                const uint4 packed1 = value_vectors[1];
                const uint32_t packed[8] = {
                    packed0.x, packed0.y, packed0.z, packed0.w,
                    packed1.x, packed1.y, packed1.z, packed1.w,
                };
#pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    __half2_raw raw;
                    raw.x = static_cast<unsigned short>(packed[pair]);
                    raw.y = static_cast<unsigned short>(packed[pair] >> 16);
                    const float2 value2 = __half22float2(raw);
                    numerator[2 * pair] =
                        numerator[2 * pair] * previous_scale +
                        score_scale * value2.x;
                    numerator[2 * pair + 1] =
                        numerator[2 * pair + 1] * previous_scale +
                        score_scale * value2.y;
                }
            } else if constexpr (Q8) {
#pragma unroll
                for (int item = 0; item < kValuesPerLane; ++item) {
                    const float value =
                        q8_kv_value(value_cache, cache_base + lane_base + item);
                    numerator[item] = numerator[item] * previous_scale +
                                      score_scale * value;
                }
            } else {
                const float4 *value_vectors =
                    reinterpret_cast<const float4 *>(
                        static_cast<const float *>(value_cache) + cache_base +
                        lane_base);
#pragma unroll
                for (int vector = 0; vector < 4; ++vector) {
                    const float4 values = value_vectors[vector];
                    const int base = 4 * vector;
                    numerator[base] = numerator[base] * previous_scale +
                                      score_scale * values.x;
                    numerator[base + 1] =
                        numerator[base + 1] * previous_scale +
                        score_scale * values.y;
                    numerator[base + 2] =
                        numerator[base + 2] * previous_scale +
                        score_scale * values.z;
                    numerator[base + 3] =
                        numerator[base + 3] * previous_scale +
                        score_scale * values.w;
                }
            }
        } else {
            int item = 0;
            for (size_t index = lane; index < head_dim;
                 index += kDotThreads, ++item) {
                const float value = Q8
                    ? q8_kv_value(value_cache, cache_base + index)
                    : (F16
                        ? __half2float(static_cast<const __half *>(value_cache)[cache_base + index])
                        : static_cast<const float *>(value_cache)[cache_base + index]);
                numerator[item] = numerator[item] * previous_scale +
                                  score_scale * value;
            }
        }
        running_max = next_max;
    }

    if (lane == 0) {
        group_max[group] = running_max;
        group_sum[group] = running_sum;
    }
    if (head_dim == 128) {
#pragma unroll
        for (int item = 0; item < kValuesPerLane; ++item) {
            group_output[group][lane * kValuesPerLane + item] =
                numerator[item];
        }
    } else {
        int item = 0;
        for (size_t index = lane; index < head_dim;
             index += kDotThreads, ++item) {
            group_output[group][index] = numerator[item];
        }
    }
    __syncthreads();

    float local_max = threadIdx.x < kDotGroups
                          ? group_max[threadIdx.x]
                          : -CUDART_INF_F;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(
            local_max,
            __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (threadIdx.x == 0) {
        block_max = local_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_scale = 0.0f;
    if (threadIdx.x < kDotGroups && group_sum[threadIdx.x] != 0.0f) {
        local_scale = expf(group_max[threadIdx.x] - block_max);
        local_sum = group_sum[threadIdx.x] * local_scale;
    }
    if (threadIdx.x < kDotGroups) {
        group_scale[threadIdx.x] = local_scale;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (threadIdx.x == 0) {
        block_sum = local_sum;
    }
    __syncthreads();

    const size_t partial_row =
        (query_index * n_head + query_head) * split_count + split;
    if (threadIdx.x == 0) {
        partial_max[partial_row] = block_max;
        partial_sum[partial_row] = block_sum;
    }
    const size_t dimension = threadIdx.x;
    if (dimension < head_dim) {
        float block_numerator = 0.0f;
#pragma unroll
        for (int source_group = 0; source_group < kDotGroups;
             ++source_group) {
            if (group_sum[source_group] != 0.0f) {
                block_numerator += group_output[source_group][dimension] *
                                   group_scale[source_group];
            }
        }
        partial_output[partial_row * head_dim + dimension] = block_numerator;
    }
}

template <bool DevicePosition, bool MultiQuery>
__global__ void attention_partial_tiled_f16_kernel(
    const float *__restrict__ query,
    const __half *__restrict__ key_cache,
    const __half *__restrict__ value_cache,
    float *__restrict__ partial_max,
    float *__restrict__ partial_sum,
    float *__restrict__ partial_output,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t split_count,
    size_t host_context_length,
    const uint32_t *__restrict__ device_position) {
    constexpr int kDotThreads = 8;
    constexpr int kDotGroups = kAttentionTileThreads / kDotThreads;
    constexpr int kValuesPerLane = 16;
    __shared__ float tile_score[kAttentionKvTile];
    __shared__ float reduction[kAttentionTileThreads];
    __shared__ float numerator_part[kAttentionTileThreads];
    __shared__ float block_max;
    __shared__ float block_sum;
    const size_t query_head = blockIdx.x;
    const size_t split = blockIdx.y;
    const size_t query_index = MultiQuery ? blockIdx.z : 0;
    if (query_head >= n_head) {
        return;
    }
    const size_t context_length = DevicePosition
                                      ? static_cast<size_t>(device_position[0]) + 1
                                      : host_context_length + query_index;
    const size_t group_size = n_head / n_head_kv;
    const size_t kv_head = query_head / group_size;
    const size_t base_count = context_length / split_count;
    const size_t remainder = context_length % split_count;
    const size_t start = split * base_count +
                         (split < remainder ? split : remainder);
    const size_t count = base_count + (split < remainder ? 1 : 0);
    const int group = threadIdx.x / kDotThreads;
    const int lane = threadIdx.x % kDotThreads;
    const unsigned int group_mask =
        0xffu << ((threadIdx.x % 32) / kDotThreads * kDotThreads);
    const float *query_row =
        query + (query_index * n_head + query_head) * head_dim;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const float4 *query_vectors = reinterpret_cast<const float4 *>(
        query_row + static_cast<size_t>(lane) * kValuesPerLane);
    float4 query4[4];
#pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        query4[vector] = query_vectors[vector];
    }

    for (size_t local_position = group; local_position < count;
         local_position += kDotGroups) {
        const size_t position = start + local_position;
        const size_t cache_base =
            (kv_head * max_context + position) * head_dim;
        const size_t lane_base =
            static_cast<size_t>(lane) * kValuesPerLane;
        const uint4 *key_vectors = reinterpret_cast<const uint4 *>(
            key_cache + cache_base + lane_base);
        const uint4 packed0 = key_vectors[0];
        const uint4 packed1 = key_vectors[1];
        const uint32_t packed[8] = {
            packed0.x, packed0.y, packed0.z, packed0.w,
            packed1.x, packed1.y, packed1.z, packed1.w,
        };
        float dot = 0.0f;
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            __half2_raw raw;
            raw.x = static_cast<unsigned short>(packed[pair]);
            raw.y = static_cast<unsigned short>(packed[pair] >> 16);
            const float2 key2 = __half22float2(raw);
            const float4 query_values = query4[pair / 2];
            const float query0 =
                (pair & 1) == 0 ? query_values.x : query_values.z;
            const float query1 =
                (pair & 1) == 0 ? query_values.y : query_values.w;
            dot = fmaf(query0, key2.x, dot);
            dot = fmaf(query1, key2.y, dot);
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(group_mask, dot, offset, 8);
        }
        if (lane == 0) {
            tile_score[local_position] = dot * scale;
        }
    }
    __syncthreads();

    reduction[threadIdx.x] =
        threadIdx.x < count ? tile_score[threadIdx.x] : -CUDART_INF_F;
    __syncthreads();
    for (int offset = kAttentionTileThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_max = reduction[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    if (threadIdx.x < count) {
        local_sum = expf(tile_score[threadIdx.x] - block_max);
        tile_score[threadIdx.x] = local_sum;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = kAttentionTileThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_sum = reduction[0];
    }
    __syncthreads();

    const size_t dimension = threadIdx.x % head_dim;
    const size_t midpoint = (count + 1) / 2;
    const size_t local_start = threadIdx.x < head_dim ? 0 : midpoint;
    const size_t local_end = threadIdx.x < head_dim ? midpoint : count;
    float numerator = 0.0f;
    for (size_t local_position = local_start; local_position < local_end;
         ++local_position) {
        const size_t position = start + local_position;
        const size_t cache_index =
            (kv_head * max_context + position) * head_dim + dimension;
        numerator = fmaf(tile_score[local_position],
                         __half2float(value_cache[cache_index]), numerator);
    }
    numerator_part[threadIdx.x] = numerator;
    __syncthreads();

    const size_t partial_row =
        (query_index * n_head + query_head) * split_count + split;
    if (threadIdx.x == 0) {
        partial_max[partial_row] = block_max;
        partial_sum[partial_row] = block_sum;
    }
    if (threadIdx.x < head_dim) {
        partial_output[partial_row * head_dim + threadIdx.x] =
            numerator_part[threadIdx.x] +
            numerator_part[threadIdx.x + head_dim];
    }
}

__global__ void attention_reduce_kernel(
    const float *__restrict__ partial_max,
    const float *__restrict__ partial_sum,
    const float *__restrict__ partial_output,
    float *__restrict__ output,
    Q8_1Block *__restrict__ quantized_output,
    int32_t *__restrict__ quantized_sums,
    size_t n_head,
    size_t head_dim,
    size_t split_count,
    size_t positions) {
    __shared__ float reduction[kAttentionThreads];
    __shared__ float partial_scale[kAttentionThreads];
    __shared__ float global_max;
    __shared__ float global_sum;
    const size_t query_head = blockIdx.x;
    const size_t query_index = blockIdx.y;
    if (query_head >= n_head || query_index >= positions) {
        return;
    }
    const size_t base =
        (query_index * n_head + query_head) * split_count;
    if (split_count <= 32) {
        float local_max = threadIdx.x < split_count
                              ? partial_max[base + threadIdx.x]
                              : -CUDART_INF_F;
        if (threadIdx.x < 32) {
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(
                    local_max,
                    __shfl_down_sync(0xffffffff, local_max, offset));
            }
            if (threadIdx.x == 0) {
                global_max = local_max;
            }
        }
        __syncthreads();
    } else {
        reduction[threadIdx.x] = threadIdx.x < split_count
                                     ? partial_max[base + threadIdx.x]
                                     : -CUDART_INF_F;
        __syncthreads();
        for (int offset = kAttentionThreads / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                reduction[threadIdx.x] = fmaxf(
                    reduction[threadIdx.x], reduction[threadIdx.x + offset]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            global_max = reduction[0];
        }
        __syncthreads();
    }

    float local_sum = 0.0f;
    float local_scale = 0.0f;
    if (threadIdx.x < split_count) {
        local_sum = partial_sum[base + threadIdx.x];
        if (local_sum != 0.0f) {
            local_scale =
                expf(partial_max[base + threadIdx.x] - global_max);
            local_sum *= local_scale;
        }
        partial_scale[threadIdx.x] = local_scale;
    }
    if (split_count <= 32) {
        if (threadIdx.x < 32) {
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                local_sum +=
                    __shfl_down_sync(0xffffffff, local_sum, offset);
            }
            if (threadIdx.x == 0) {
                global_sum = local_sum;
            }
        }
        __syncthreads();
    } else {
        reduction[threadIdx.x] = local_sum;
        __syncthreads();
        for (int offset = kAttentionThreads / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                reduction[threadIdx.x] += reduction[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            global_sum = reduction[0];
        }
        __syncthreads();
    }

    const size_t dimension = threadIdx.x;
    float value = 0.0f;
    if (dimension < head_dim) {
        float numerator = 0.0f;
        for (size_t split = 0; split < split_count; ++split) {
            const size_t partial_row = base + split;
            if (partial_sum[partial_row] != 0.0f) {
                numerator += partial_output[partial_row * head_dim + dimension] *
                             partial_scale[split];
            }
        }
        value = numerator / global_sum;
        output[(query_index * n_head + query_head) * head_dim + dimension] =
            value;
    }

    if (quantized_output != nullptr) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        float maximum = fabsf(value);
        float source_sum = value;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffff, maximum, offset));
            source_sum +=
                __shfl_down_sync(0xffffffff, source_sum, offset);
        }
        maximum = __shfl_sync(0xffffffff, maximum, 0);
        const float scale = maximum / 127.0f;
        const int quant = maximum == 0.0f
                              ? 0
                              : __float2int_rn(value / scale);
        const size_t q8_block =
            (query_index * n_head + query_head) *
                (head_dim / kQ8BlockElements) +
            warp;
        quantized_output[q8_block].qs[lane] =
            static_cast<int8_t>(quant);
        int quantized_sum = quant;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            quantized_sum +=
                __shfl_down_sync(0xffffffff, quantized_sum, offset);
        }
        if (lane == 0) {
            quantized_output[q8_block].ds =
                __floats2half2_rn(scale, source_sum);
            quantized_sums[q8_block] = quantized_sum;
        }
    }
}

__device__ __forceinline__ bool better_argmax(float candidate_value,
                                              uint32_t candidate_index,
                                              float best_value,
                                              uint32_t best_index) {
    if (isnan(candidate_value)) {
        return false;
    }
    return best_index == UINT32_MAX || candidate_value > best_value ||
           (candidate_value == best_value && candidate_index < best_index);
}

__device__ __forceinline__ void reduce_argmax(float *values,
                                              uint32_t *indices) {
    __syncthreads();
    for (int offset = kBlockThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset &&
            better_argmax(values[threadIdx.x + offset],
                          indices[threadIdx.x + offset],
                          values[threadIdx.x], indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + offset];
            indices[threadIdx.x] = indices[threadIdx.x + offset];
        }
        __syncthreads();
    }
}

__global__ void argmax_partial_kernel(const float *__restrict__ input,
                                      size_t elements,
                                      float *__restrict__ partial_values,
                                      uint32_t *__restrict__ partial_indices) {
    __shared__ float values[kBlockThreads];
    __shared__ uint32_t indices[kBlockThreads];
    const size_t block_start =
        blockIdx.x * kBlockThreads * kArgmaxItemsPerThread;
    float best_value = -CUDART_INF_F;
    uint32_t best_index = UINT32_MAX;
    for (int item = 0; item < kArgmaxItemsPerThread; ++item) {
        const size_t index = block_start + threadIdx.x + item * kBlockThreads;
        if (index < elements && index <= UINT32_MAX) {
            const float candidate = input[index];
            if (better_argmax(candidate, static_cast<uint32_t>(index),
                              best_value, best_index)) {
                best_value = candidate;
                best_index = static_cast<uint32_t>(index);
            }
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    reduce_argmax(values, indices);
    if (threadIdx.x == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}

__global__ void argmax_reduce_kernel(const float *__restrict__ partial_values,
                                     const uint32_t *__restrict__ partial_indices,
                                     size_t partial_count,
                                     uint32_t *__restrict__ output) {
    __shared__ float values[kBlockThreads];
    __shared__ uint32_t indices[kBlockThreads];
    float best_value = -CUDART_INF_F;
    uint32_t best_index = UINT32_MAX;
    for (size_t index = threadIdx.x; index < partial_count;
         index += blockDim.x) {
        if (better_argmax(partial_values[index], partial_indices[index],
                          best_value, best_index)) {
            best_value = partial_values[index];
            best_index = partial_indices[index];
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    reduce_argmax(values, indices);
    if (threadIdx.x == 0) {
        output[0] = indices[0] == UINT32_MAX ? 0 : indices[0];
    }
}

cublasStatus_t prefill_matmul(
    LeoneCublasLt *owner,
    const __half *a,
    const __half *b,
    float *d,
    size_t m,
    size_t n,
    size_t k,
    int batch_count,
    int64_t a_stride,
    int64_t b_stride,
    int64_t d_stride,
    bool transpose_b,
    float alpha,
    void *workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (owner == nullptr || owner->handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    auto cleanup = [&]() {
        if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
        if (d_layout != nullptr) cublasLtMatrixLayoutDestroy(d_layout);
        if (c_layout != nullptr) cublasLtMatrixLayoutDestroy(c_layout);
        if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
        if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
        if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    };
    cublasStatus_t status = cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return status;
    }
    const cublasOperation_t trans_a = CUBLAS_OP_N;
    const cublasOperation_t trans_b =
        transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &trans_a, sizeof(trans_a));
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB,
            &trans_b, sizeof(trans_b));
    }
    const size_t b_rows = transpose_b ? n : k;
    const size_t b_columns = transpose_b ? k : n;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(
            &a_layout, CUDA_R_16F, m, k, k);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(
            &b_layout, CUDA_R_16F, b_rows, b_columns, b_columns);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(
            &c_layout, CUDA_R_32F, m, n, n);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(
            &d_layout, CUDA_R_32F, m, n, n);
    }
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    for (cublasLtMatrixLayout_t layout :
         {a_layout, b_layout, c_layout, d_layout}) {
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order, sizeof(order));
        }
    }
    if (status == CUBLAS_STATUS_SUCCESS && batch_count > 1) {
        for (cublasLtMatrixLayout_t layout :
             {a_layout, b_layout, c_layout, d_layout}) {
            status = cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count, sizeof(batch_count));
            if (status != CUBLAS_STATUS_SUCCESS) break;
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatrixLayoutSetAttribute(
                a_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &a_stride, sizeof(a_stride));
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatrixLayoutSetAttribute(
                b_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &b_stride, sizeof(b_stride));
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatrixLayoutSetAttribute(
                c_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &d_stride, sizeof(d_stride));
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatrixLayoutSetAttribute(
                d_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &d_stride, sizeof(d_stride));
        }
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulPreferenceCreate(&preference);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes));
    }
    const uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_NONE;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &reduction_mask, sizeof(reduction_mask));
    }
    const PrefillMatmulKey key{
        m, n, k, batch_count, a_stride, b_stride, d_stride, transpose_b};
    cublasLtMatmulAlgo_t algorithm{};
    const auto cached = owner->algorithms.find(key);
    if (status == CUBLAS_STATUS_SUCCESS &&
        cached != owner->algorithms.end()) {
        algorithm = cached->second;
    } else if (status == CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        status = cublasLtMatmulAlgoGetHeuristic(
            owner->handle, operation, a_layout, b_layout, c_layout, d_layout,
            preference, 1, &heuristic, &returned);
        if (status == CUBLAS_STATUS_SUCCESS && returned == 0) {
            status = CUBLAS_STATUS_NOT_SUPPORTED;
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            algorithm = heuristic.algo;
            owner->algorithms.emplace(key, algorithm);
        }
    }
    const float beta = 0.0f;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(
            owner->handle, operation, &alpha, a, a_layout, b, b_layout,
            &beta, d, c_layout, d, d_layout, &algorithm,
            workspace, workspace_bytes, stream);
    }
    cleanup();
    return status;
}

template <bool F16Cache>
cublasStatus_t launch_prefill_attention_tile(
    LeoneCublasLt *handle,
    const float *query,
    const void *key_cache,
    const void *value_cache,
    float *output,
    __half *converted_query,
    float *scores,
    __half *probabilities,
    float *head_output,
    __half *converted_kv,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t tokens,
    void *workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    const size_t context_length = start_position + tokens;
    const size_t query_elements = tokens * n_head * head_dim;
    prefill_query_to_head_kernel<<<
        static_cast<unsigned int>((query_elements + kBlockThreads - 1) /
                                  kBlockThreads),
        kBlockThreads, 0, stream>>>(
        query, converted_query, tokens, n_head, head_dim);
    const __half *key_half;
    const __half *value_half;
    size_t cache_head_stride;
    if constexpr (F16Cache) {
        key_half = static_cast<const __half *>(key_cache);
        value_half = static_cast<const __half *>(value_cache);
        cache_head_stride = max_context * head_dim;
    } else {
        const size_t compact_elements =
            n_head_kv * context_length * head_dim;
        key_half = converted_kv;
        value_half = converted_kv + compact_elements;
        prefill_cache_to_f16_kernel<<<
            static_cast<unsigned int>((compact_elements + kBlockThreads - 1) /
                                      kBlockThreads),
            kBlockThreads, 0, stream>>>(
            static_cast<const float *>(key_cache),
            static_cast<const float *>(value_cache),
            converted_kv, converted_kv + compact_elements,
            n_head_kv, head_dim, max_context, context_length);
        cache_head_stride = context_length * head_dim;
    }
    if (cudaGetLastError() != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    const int group_size = static_cast<int>(n_head / n_head_kv);
    const float score_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    for (size_t kv_head = 0; kv_head < n_head_kv; ++kv_head) {
        const size_t query_head = kv_head * static_cast<size_t>(group_size);
        const cublasStatus_t status = prefill_matmul(
            handle, converted_query + query_head * tokens * head_dim,
            key_half + kv_head * cache_head_stride,
            scores + query_head * tokens * context_length,
            tokens * static_cast<size_t>(group_size), context_length,
            head_dim, 1, 0, 0, 0, true,
            score_scale, workspace, workspace_bytes, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    prefill_causal_softmax_kernel<<<
        static_cast<unsigned int>(n_head * tokens),
        kPrefillSoftmaxThreads, 0, stream>>>(
        scores, probabilities, n_head, tokens, context_length,
        start_position);
    if (cudaGetLastError() != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    for (size_t kv_head = 0; kv_head < n_head_kv; ++kv_head) {
        const size_t query_head = kv_head * static_cast<size_t>(group_size);
        const cublasStatus_t status = prefill_matmul(
            handle, probabilities + query_head * tokens * context_length,
            value_half + kv_head * cache_head_stride,
            head_output + query_head * tokens * head_dim,
            tokens * static_cast<size_t>(group_size), head_dim,
            context_length, 1, 0, 0, 0, false,
            1.0f, workspace, workspace_bytes, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    prefill_head_to_token_kernel<<<
        static_cast<unsigned int>((query_elements + kBlockThreads - 1) /
                                  kBlockThreads),
        kBlockThreads, 0, stream>>>(
        head_output, output, tokens, n_head, head_dim);
    return cudaGetLastError() == cudaSuccess
               ? CUBLAS_STATUS_SUCCESS
               : CUBLAS_STATUS_EXECUTION_FAILED;
}

template <bool F16Cache>
cublasStatus_t launch_prefill_attention(
    LeoneCublasLt *handle,
    const float *query,
    const void *key_cache,
    const void *value_cache,
    float *output,
    __half *converted_query,
    float *scores,
    __half *probabilities,
    float *head_output,
    __half *converted_kv,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t tokens,
    void *workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    constexpr size_t kQueryTile = 1024;
    const size_t token_stride = n_head * head_dim;
    for (size_t offset = 0; offset < tokens; offset += kQueryTile) {
        const size_t tile_tokens =
            tokens - offset < kQueryTile ? tokens - offset : kQueryTile;
        const cublasStatus_t status = launch_prefill_attention_tile<F16Cache>(
            handle, query + offset * token_stride, key_cache, value_cache,
            output + offset * token_stride, converted_query, scores,
            probabilities, head_output, converted_kv, n_head, n_head_kv,
            head_dim, max_context, start_position + offset, tile_tokens,
            workspace, workspace_bytes, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cudaError_t launch_status() {
    return cudaGetLastError();
}

template <bool F16, bool Q8 = false>
cudaError_t attention_split_count(size_t n_head,
                                  size_t max_context,
                                  size_t *split_count) {
    int max_blocks_per_sm = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, attention_partial_kernel<F16, Q8, true, false>,
        kAttentionThreads, 0);
    if (status != cudaSuccess) {
        return status;
    }
    int device = 0;
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    int multiprocessors = 0;
    status = cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) {
        return status;
    }

    const size_t bucket_tiles =
        (max_context + kAttentionKvTile - 1) / kAttentionKvTile;
    const size_t kv_tiles =
        bucket_tiles < static_cast<size_t>(kAttentionMaxSplitKv)
            ? bucket_tiles
            : static_cast<size_t>(kAttentionMaxSplitKv);
    if (max_blocks_per_sm <= 0 || multiprocessors <= 0 || kv_tiles == 0) {
        return cudaErrorInvalidConfiguration;
    }
    size_t selected = static_cast<size_t>(max_blocks_per_sm) < kv_tiles
                          ? static_cast<size_t>(max_blocks_per_sm)
                          : kv_tiles;
    selected = selected == 0 ? 1 : selected;
    const size_t blocks_per_wave =
        static_cast<size_t>(multiprocessors) * max_blocks_per_sm;
    size_t best_efficiency = 0;
    size_t best_waves = 0;
    for (size_t test = selected; test <= kv_tiles; ++test) {
        const size_t total_blocks = n_head * test;
        const size_t waves =
            (total_blocks + blocks_per_wave - 1) / blocks_per_wave;
        const size_t efficiency =
            100 * total_blocks / (waves * blocks_per_wave);
        if (best_efficiency >= 95 && waves > best_waves) {
            break;
        }
        if (efficiency > best_efficiency) {
            selected = test;
            best_efficiency = efficiency;
            best_waves = waves;
        }
    }
    *split_count = selected;
    return cudaSuccess;
}

template <bool F16, bool Q8, bool DevicePosition, typename Cache>
int launch_attention(const float *query,
                     const Cache *key_cache,
                     const Cache *value_cache,
                     float *output,
                     float *partial_max,
                     float *partial_sum,
                     float *partial_output,
                     uint8_t *quantized_output,
                     int32_t *quantized_sums,
                     size_t n_head,
                     size_t n_head_kv,
                     size_t head_dim,
                     size_t max_context,
                     size_t context_length,
                     const uint32_t *position,
                     void *stream) {
    size_t split_count = 0;
    cudaError_t status =
        attention_split_count<F16, Q8>(n_head, max_context, &split_count);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const size_t positions_per_split =
        (max_context + split_count - 1) / split_count;
    if constexpr (F16) {
        if (head_dim == 128 && positions_per_split <= kAttentionKvTile) {
            attention_partial_tiled_f16_kernel<DevicePosition, false><<<
                dim3(static_cast<unsigned int>(n_head),
                     static_cast<unsigned int>(split_count)),
                kAttentionTileThreads, 0, cuda_stream>>>(
                query, reinterpret_cast<const __half *>(key_cache),
                reinterpret_cast<const __half *>(value_cache), partial_max,
                partial_sum, partial_output, n_head, n_head_kv, head_dim,
                max_context, split_count, context_length, position);
        } else {
            attention_partial_kernel<F16, Q8, DevicePosition, false><<<
                dim3(static_cast<unsigned int>(n_head),
                     static_cast<unsigned int>(split_count)),
                kAttentionThreads, 0, cuda_stream>>>(
                query, key_cache, value_cache, partial_max, partial_sum,
                partial_output, n_head, n_head_kv, head_dim, max_context,
                split_count, context_length, position);
        }
    } else {
        attention_partial_kernel<F16, Q8, DevicePosition, false><<<
            dim3(static_cast<unsigned int>(n_head),
                 static_cast<unsigned int>(split_count)),
            kAttentionThreads, 0, cuda_stream>>>(
            query, key_cache, value_cache, partial_max, partial_sum,
            partial_output, n_head, n_head_kv, head_dim, max_context,
            split_count, context_length, position);
    }
    status = launch_status();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    attention_reduce_kernel<<<dim3(static_cast<unsigned int>(n_head), 1),
                              kAttentionThreads, 0, cuda_stream>>>(
        partial_max, partial_sum, partial_output, output,
        reinterpret_cast<Q8_1Block *>(quantized_output), quantized_sums,
        n_head, head_dim, split_count, 1);
    return static_cast<int>(launch_status());
}

dim3 element_grid(size_t elements) {
    return dim3(static_cast<unsigned int>(
        (elements + kBlockThreads - 1) / kBlockThreads));
}

}

extern "C" const char *ie_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int ie_cuda_set_device(int device) {
    return static_cast<int>(cudaSetDevice(device));
}

extern "C" int ie_cuda_initialize() {
    return static_cast<int>(cudaFree(nullptr));
}

extern "C" const char *ie_cublaslt_error_string(int code) {
    return cublasGetStatusString(static_cast<cublasStatus_t>(code));
}

extern "C" int ie_cublaslt_create(void **handle) {
    if (handle == nullptr) {
        return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
    }
    auto *owner = new (std::nothrow) LeoneCublasLt;
    if (owner == nullptr) {
        return static_cast<int>(CUBLAS_STATUS_ALLOC_FAILED);
    }
    const cublasStatus_t status = cublasLtCreate(&owner->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        delete owner;
        return static_cast<int>(status);
    }
    *handle = owner;
    return static_cast<int>(CUBLAS_STATUS_SUCCESS);
}

extern "C" int ie_cublaslt_destroy(void *handle) {
    auto *owner = static_cast<LeoneCublasLt *>(handle);
    if (owner == nullptr) {
        return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
    }
    const cublasStatus_t status = cublasLtDestroy(owner->handle);
    delete owner;
    return static_cast<int>(status);
}

extern "C" int ie_launch_dequant_k_f16(
    const uint8_t *weights,
    uint16_t *output,
    size_t rows,
    size_t columns,
    int q4,
    void *stream) {
    const size_t elements = rows * columns;
    const dim3 grid = element_grid(elements);
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (q4 != 0) {
        dequant_k_f16_kernel<true><<<grid, kBlockThreads, 0, cuda_stream>>>(
            weights, reinterpret_cast<__half *>(output), elements);
    } else {
        dequant_k_f16_kernel<false><<<grid, kBlockThreads, 0, cuda_stream>>>(
            weights, reinterpret_cast<__half *>(output), elements);
    }
    return static_cast<int>(launch_status());
}

extern "C" int ie_cublaslt_prefill_gemm(
    void *handle,
    const uint8_t *weights,
    const float *input,
    float *output,
    uint16_t *dequantized_weights,
    uint16_t *converted_input,
    size_t rows,
    size_t columns,
    size_t tokens,
    int q4,
    uint8_t *workspace,
    size_t workspace_bytes,
    void *stream) {
    const size_t weight_elements = rows * columns;
    const size_t input_elements = tokens * columns;
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const dim3 weight_grid = element_grid(weight_elements);
    if (q4 != 0) {
        dequant_k_f16_kernel<true><<<weight_grid, kBlockThreads, 0,
                                      cuda_stream>>>(
            weights, reinterpret_cast<__half *>(dequantized_weights),
            weight_elements);
    } else {
        dequant_k_f16_kernel<false><<<weight_grid, kBlockThreads, 0,
                                       cuda_stream>>>(
            weights, reinterpret_cast<__half *>(dequantized_weights),
            weight_elements);
    }
    f32_to_f16_kernel<<<element_grid(input_elements), kBlockThreads, 0,
                        cuda_stream>>>(
        input, reinterpret_cast<__half *>(converted_input), input_elements);
    if (cudaGetLastError() != cudaSuccess) {
        return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
    }
    const cublasStatus_t status = prefill_matmul(
        static_cast<LeoneCublasLt *>(handle),
        reinterpret_cast<const __half *>(converted_input),
        reinterpret_cast<const __half *>(dequantized_weights),
        output, tokens, rows, columns, 1, 0, 0, 0, true, 1.0f,
        workspace, workspace_bytes, cuda_stream);
    return static_cast<int>(status);
}

extern "C" int ie_cublaslt_attention_prefill_f16(
    void *handle,
    const float *query,
    const uint16_t *key_cache,
    const uint16_t *value_cache,
    float *output,
    uint16_t *converted_query,
    float *scores,
    uint16_t *probabilities,
    float *head_output,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t tokens,
    uint8_t *workspace,
    size_t workspace_bytes,
    void *stream) {
    return static_cast<int>(launch_prefill_attention<true>(
        static_cast<LeoneCublasLt *>(handle), query, key_cache,
        value_cache, output, reinterpret_cast<__half *>(converted_query),
        scores, reinterpret_cast<__half *>(probabilities), head_output,
        nullptr, n_head, n_head_kv, head_dim, max_context, start_position,
        tokens, workspace, workspace_bytes,
        reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cublaslt_attention_prefill_f32(
    void *handle,
    const float *query,
    const float *key_cache,
    const float *value_cache,
    float *output,
    uint16_t *converted_query,
    float *scores,
    uint16_t *probabilities,
    float *head_output,
    uint16_t *converted_kv,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t tokens,
    uint8_t *workspace,
    size_t workspace_bytes,
    void *stream) {
    return static_cast<int>(launch_prefill_attention<false>(
        static_cast<LeoneCublasLt *>(handle), query, key_cache,
        value_cache, output, reinterpret_cast<__half *>(converted_query),
        scores, reinterpret_cast<__half *>(probabilities), head_output,
        reinterpret_cast<__half *>(converted_kv), n_head, n_head_kv,
        head_dim, max_context, start_position, tokens, workspace,
        workspace_bytes, reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ie_attention_split_count(int f16_cache,
                                          size_t n_head,
                                          size_t max_context,
                                          size_t *split_count) {
    if (split_count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const cudaError_t status = f16_cache != 0
                                   ? attention_split_count<true>(
                                         n_head, max_context, split_count)
                                   : attention_split_count<false>(
                                         n_head, max_context, split_count);
    return static_cast<int>(status);
}

extern "C" int ie_cuda_mem_get_info(size_t *free_bytes,
                                      size_t *total_bytes) {
    return static_cast<int>(cudaMemGetInfo(free_bytes, total_bytes));
}

extern "C" int ie_cuda_malloc(void **pointer, size_t bytes) {
    return static_cast<int>(cudaMalloc(pointer, bytes));
}

extern "C" int ie_cuda_free(void *pointer) {
    return static_cast<int>(cudaFree(pointer));
}

extern "C" int ie_cuda_copy_h2d(void *destination,
                                const void *source,
                                size_t bytes) {
    cudaError_t status =
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    // Pageable H2D copies can return after host staging. Fence the legacy
    // stream before a nonblocking Leone stream consumes the allocation.
    return static_cast<int>(cudaStreamSynchronize(nullptr));
}

extern "C" int ie_cuda_copy_h2d_async(void *destination,
                                      const void *source,
                                      size_t bytes,
                                      void *stream) {
    return static_cast<int>(cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyHostToDevice,
        reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cuda_copy_d2h(void *destination,
                                const void *source,
                                size_t bytes) {
    return static_cast<int>(
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}

extern "C" int ie_cuda_copy_d2h_async(void *destination,
                                      const void *source,
                                      size_t bytes,
                                      void *stream) {
    return static_cast<int>(cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToHost,
        reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cuda_copy_d2d_async(void *destination,
                                      const void *source,
                                      size_t bytes,
                                      void *stream) {
    return static_cast<int>(cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cuda_stream_create(void **stream) {
    return static_cast<int>(cudaStreamCreateWithFlags(
        reinterpret_cast<cudaStream_t *>(stream), cudaStreamNonBlocking));
}

extern "C" int ie_cuda_stream_destroy(void *stream) {
    return static_cast<int>(
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cuda_stream_synchronize(void *stream) {
    return static_cast<int>(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cuda_graph_capture_begin(void *stream) {
    return static_cast<int>(cudaStreamBeginCapture(
        reinterpret_cast<cudaStream_t>(stream),
        cudaStreamCaptureModeThreadLocal));
}

extern "C" int ie_cuda_graph_capture_end(void *stream, void **graph) {
    cudaGraph_t captured = nullptr;
    cudaError_t status = cudaStreamEndCapture(
        reinterpret_cast<cudaStream_t>(stream), &captured);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    cudaGraphExec_t executable = nullptr;
    status = cudaGraphInstantiate(&executable, captured, 0);
    const cudaError_t destroy_status = cudaGraphDestroy(captured);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (destroy_status != cudaSuccess) {
        cudaGraphExecDestroy(executable);
        return static_cast<int>(destroy_status);
    }
    DecodeGraph *owned = new (std::nothrow) DecodeGraph{executable};
    if (owned == nullptr) {
        cudaGraphExecDestroy(executable);
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    *graph = owned;
    return static_cast<int>(cudaSuccess);
}

extern "C" int ie_cuda_graph_launch(void *graph, void *stream) {
    DecodeGraph *owned = static_cast<DecodeGraph *>(graph);
    return static_cast<int>(cudaGraphLaunch(
        owned->executable, reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cuda_graph_destroy(void *graph) {
    DecodeGraph *owned = static_cast<DecodeGraph *>(graph);
    const cudaError_t status = cudaGraphExecDestroy(owned->executable);
    delete owned;
    return static_cast<int>(status);
}

extern "C" int ie_cuda_event_create(void **event) {
    return static_cast<int>(
        cudaEventCreate(reinterpret_cast<cudaEvent_t *>(event)));
}

extern "C" int ie_cuda_event_destroy(void *event) {
    return static_cast<int>(
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
}

extern "C" int ie_cuda_event_record(void *event, void *stream) {
    return static_cast<int>(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                            reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int ie_cuda_event_synchronize(void *event) {
    return static_cast<int>(
        cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
}

extern "C" int ie_cuda_event_elapsed_ms(float *milliseconds,
                                         void *start,
                                         void *end) {
    return static_cast<int>(cudaEventElapsedTime(
        milliseconds, reinterpret_cast<cudaEvent_t>(start),
        reinterpret_cast<cudaEvent_t>(end)));
}

extern "C" int ie_launch_quantize_q8_1(const float *input,
                                       uint8_t *output,
                                       int32_t *quantized_sums,
                                       size_t elements,
                                       void *stream) {
    const size_t quant_blocks = elements / kQ8BlockElements;
    const size_t thread_blocks =
        (quant_blocks + kBlockThreads / 32 - 1) / (kBlockThreads / 32);
    quantize_q8_1_warp<<<static_cast<unsigned int>(thread_blocks),
                         kBlockThreads, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
        input, reinterpret_cast<Q8_1Block *>(output), quantized_sums,
        quant_blocks);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q4_k_gemv(const uint8_t *weights,
                                    const uint8_t *input,
                                    const int32_t *quantized_sums,
                                    float *output,
                                    size_t rows,
                                    size_t columns,
                                    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (columns == 4096) {
        q4_q8_gemv_warp<16, kGemvWarpsPerBlock><<<
            static_cast<unsigned int>(rows),
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), nullptr,
            output, columns);
    } else if (columns == 12288) {
        q4_q8_gemv_warp<48, kGemvWarpsPerBlock><<<
            static_cast<unsigned int>(rows),
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), nullptr,
            output, columns);
    } else {
        q4_q8_gemv_warp<0, kGemvWarpsPerBlock><<<
            static_cast<unsigned int>(rows),
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), nullptr,
            output, columns);
    }
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q4_k_gemv_residual(
    const uint8_t *weights,
    const uint8_t *input,
    const float *residual,
    float *output,
    size_t rows,
    size_t columns,
    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (columns == 4096) {
        q4_q8_gemv_warp<16, kGemvWarpsPerBlock><<<
            static_cast<unsigned int>(rows),
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), residual,
            output, columns);
    } else if (columns == 12288) {
        q4_q8_gemv_warp<48, kGemvWarpsPerBlock><<<
            static_cast<unsigned int>(rows),
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), residual,
            output, columns);
    } else {
        q4_q8_gemv_warp<0, kGemvWarpsPerBlock><<<
            static_cast<unsigned int>(rows),
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), residual,
            output, columns);
    }
    return static_cast<int>(launch_status());
}

template <int FixedBlocksPerRow, int Positions>
void launch_q4_k_gemv_multi_width(
    const uint8_t *weights,
    const uint8_t *input,
    const float *residual,
    float *output,
    size_t rows,
    size_t columns,
    cudaStream_t stream) {
    q4_q8_gemv_multi_warp<FixedBlocksPerRow, kGemvWarpsPerBlock, Positions><<<
        static_cast<unsigned int>(rows), kGemvWarpsPerBlock * 32, 0,
        stream>>>(weights, reinterpret_cast<const Q8_1Block *>(input),
                  residual, output, rows, columns);
}

template <int FixedBlocksPerRow>
int launch_q4_k_gemv_multi(
    const uint8_t *weights,
    const uint8_t *input,
    const float *residual,
    float *output,
    size_t rows,
    size_t columns,
    size_t positions,
    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (positions) {
    case 1: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 1>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 2: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 2>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 3: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 3>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 4: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 4>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 5: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 5>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 6: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 6>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 7: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 7>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 8: launch_q4_k_gemv_multi_width<FixedBlocksPerRow, 8>(weights, input, residual, output, rows, columns, cuda_stream); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q4_k_gemv_multi(
    const uint8_t *weights,
    const uint8_t *input,
    const float *residual,
    float *output,
    size_t rows,
    size_t columns,
    size_t positions,
    void *stream) {
    if (columns == 4096) {
        return launch_q4_k_gemv_multi<16>(
            weights, input, residual, output, rows, columns, positions, stream);
    }
    if (columns == 12288) {
        return launch_q4_k_gemv_multi<48>(
            weights, input, residual, output, rows, columns, positions, stream);
    }
    return launch_q4_k_gemv_multi<0>(
        weights, input, residual, output, rows, columns, positions, stream);
}

extern "C" int ie_launch_q4_k_gemv_pair(
    const uint8_t *first_weights,
    const uint8_t *second_weights,
    const uint8_t *input,
    const int32_t *quantized_sums,
    float *first_output,
    float *second_output,
    size_t first_rows,
    size_t second_rows,
    size_t columns,
    void *stream) {
    const size_t total_rows = first_rows + second_rows;
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (columns == 4096) {
        q4_q8_gemv_pair_warp<16><<<static_cast<unsigned int>(total_rows),
                                    kGemvWarpsPerBlock * 32, 0,
                                    cuda_stream>>>(
            first_weights, second_weights,
            reinterpret_cast<const Q8_1Block *>(input),
            first_output, second_output, first_rows, second_rows, columns);
    } else {
        q4_q8_gemv_pair_warp<0><<<static_cast<unsigned int>(total_rows),
                                   kGemvWarpsPerBlock * 32, 0,
                                   cuda_stream>>>(
            first_weights, second_weights,
            reinterpret_cast<const Q8_1Block *>(input),
            first_output, second_output, first_rows, second_rows, columns);
    }
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q4_k_gemv_swiglu(
    const uint8_t *gate_weights,
    const uint8_t *up_weights,
    const uint8_t *input,
    float *gate_output,
    float *up_output,
    float *output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    uint32_t *epilogue_ready,
    size_t rows,
    size_t columns,
    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (columns == 4096) {
        q4_q8_gemv_swiglu_warp<16><<<static_cast<unsigned int>(rows),
                                      kGemvWarpsPerBlock * 32, 0,
                                      cuda_stream>>>(
            gate_weights, up_weights,
            reinterpret_cast<const Q8_1Block *>(input), gate_output,
            up_output, output,
            reinterpret_cast<Q8_1Block *>(quantized_output), quantized_sums,
            epilogue_ready, rows, columns);
    } else {
        q4_q8_gemv_swiglu_warp<0><<<static_cast<unsigned int>(rows),
                                     kGemvWarpsPerBlock * 32, 0,
                                     cuda_stream>>>(
            gate_weights, up_weights,
            reinterpret_cast<const Q8_1Block *>(input), gate_output,
            up_output, output,
            reinterpret_cast<Q8_1Block *>(quantized_output), quantized_sums,
            epilogue_ready, rows, columns);
    }
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q4_k_gemv_probe(
    const uint8_t *weights,
    const uint8_t *input,
    float *output,
    size_t rows,
    size_t columns,
    size_t weight_set,
    int geometry,
    void *stream) {
    const size_t matrix_bytes =
        rows * (columns / kQuantBlockElements) *
        (kQ4CodeBytes + kQ4MetadataBytes);
    const uint8_t *matrix = weights + weight_set * matrix_bytes;
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define IE_LAUNCH_Q4_PROBE(warps, rows_per_cta, loads_only)                 \
    q4_q8_gemv_probe<warps, rows_per_cta, loads_only><<<                   \
        static_cast<unsigned int>((rows + rows_per_cta - 1) / rows_per_cta), \
        warps * rows_per_cta * 32, 0, cuda_stream>>>(                      \
        matrix, reinterpret_cast<const Q8_1Block *>(input), output, rows,  \
        columns)
    switch (geometry) {
        case 0: IE_LAUNCH_Q4_PROBE(4, 1, true); break;
        case 1: IE_LAUNCH_Q4_PROBE(2, 1, false); break;
        case 2: IE_LAUNCH_Q4_PROBE(3, 1, false); break;
        case 3: IE_LAUNCH_Q4_PROBE(4, 1, false); break;
        case 4: IE_LAUNCH_Q4_PROBE(2, 2, false); break;
        case 5: IE_LAUNCH_Q4_PROBE(3, 2, false); break;
        case 6: IE_LAUNCH_Q4_PROBE(4, 2, false); break;
        case 7: IE_LAUNCH_Q4_PROBE(2, 4, false); break;
        case 8: IE_LAUNCH_Q4_PROBE(3, 4, false); break;
        case 9: IE_LAUNCH_Q4_PROBE(4, 4, false); break;
        case 10:
            if (columns != 4096) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_warp<16, kGemvWarpsPerBlock><<<
                static_cast<unsigned int>(rows),
                kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), nullptr,
                output, columns);
            break;
        case 11:
            if (columns != 4096) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_wide_probe<<<
                static_cast<unsigned int>(rows),
                kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output,
                rows);
            break;
        case 12:
            if (columns != 4096 || (rows & 1) != 0) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_two_rows_probe<<<
                static_cast<unsigned int>(rows / 2),
                kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output,
                rows);
            break;
        case 13:
            if (columns != 4096) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_row_layout_probe<<<
                static_cast<unsigned int>(rows),
                kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output);
            break;
        case 14:
            if (columns != 4096) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_split_probe<<<
                static_cast<unsigned int>(rows * 2), 64, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output,
                rows);
            q4_q8_gemv_split_reduce_probe<<<
                static_cast<unsigned int>((rows + kBlockThreads - 1) /
                                          kBlockThreads),
                kBlockThreads, 0, cuda_stream>>>(output, rows);
            break;
        case 15:
            if (columns != 4096) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_gguf_probe<<<
                static_cast<unsigned int>(rows),
                kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output);
            break;
        case 16:
            if (columns != 4096) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_two_block_ilp_probe<<<
                static_cast<unsigned int>(rows),
                kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output,
                rows);
            break;
        case 17: IE_LAUNCH_Q4_PROBE(1, 4, false); break;
        case 18:
            if (columns != 4096) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_gguf_wide_probe<<<
                static_cast<unsigned int>(rows),
                kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output);
            break;
        case 19:
            if (columns != 4096 || (rows & 3) != 0) {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            q4_q8_gemv_gguf_wide_one_warp_probe<<<
                static_cast<unsigned int>(rows / 4), 128, 0, cuda_stream>>>(
                matrix, reinterpret_cast<const Q8_1Block *>(input), output);
            break;
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
#undef IE_LAUNCH_Q4_PROBE
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q4_k_gemv_ring_probe(
    const uint8_t *weights,
    const uint8_t *input,
    float *output,
    size_t rows,
    size_t columns,
    size_t weight_sets,
    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (columns == 4096) {
        q4_q8_gemv_ring_probe<16, kGemvWarpsPerBlock><<<
            kQ4ProbePersistentBlocks,
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), output,
            rows, weight_sets, columns);
    } else {
        q4_q8_gemv_ring_probe<0, kGemvWarpsPerBlock><<<
            kQ4ProbePersistentBlocks,
            kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
            weights, reinterpret_cast<const Q8_1Block *>(input), output,
            rows, weight_sets, columns);
    }
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q4_k_apron_pair_probe(
    const uint8_t *first_weights,
    const uint8_t *second_weights,
    const uint8_t *third_weights,
    const uint8_t *input,
    float *output,
    size_t rows,
    size_t columns,
    size_t weight_set,
    size_t apron_bytes,
    void *stream) {
    if (rows != 4096 || columns != 4096 ||
        apron_bytes > 4 * 1024 * 1024 ||
        (apron_bytes & (sizeof(uint4) - 1)) != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t first_matrix_bytes =
        rows * 16 * (kQ4CodeBytes + kQ4MetadataBytes);
    constexpr size_t next_rows = 12288;
    const size_t next_matrix_bytes =
        next_rows * 16 * (kQ4CodeBytes + kQ4MetadataBytes);
    const uint8_t *first =
        first_weights + weight_set * first_matrix_bytes;
    const uint8_t *second =
        second_weights + weight_set * next_matrix_bytes;
    const uint8_t *third =
        third_weights + weight_set * next_matrix_bytes;
    const unsigned int apron_blocks =
        apron_bytes == 0 ? 0 : kQ4ApronBlocks;
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    q4_q8_gemv_apron_probe<16, kGemvWarpsPerBlock><<<
        static_cast<unsigned int>(rows) + apron_blocks,
        kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
        first, second, reinterpret_cast<const Q8_1Block *>(input), output,
        rows, columns, next_rows, apron_bytes);
    q4_q8_gemv_pair_warp<16><<<
        static_cast<unsigned int>(2 * next_rows),
        kGemvWarpsPerBlock * 32, 0, cuda_stream>>>(
        second, third, reinterpret_cast<const Q8_1Block *>(input), output,
        output + next_rows, next_rows, next_rows, columns);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q6_k_gemv(const uint8_t *weights,
                                    const uint8_t *input,
                                    float *output,
                                    size_t rows,
                                    size_t columns,
                                    void *stream) {
    q6_q8_gemv_warp<<<static_cast<unsigned int>(rows),
                       kGemvWarpsPerBlock * 32, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        weights, reinterpret_cast<const Q8_1Block *>(input), nullptr, output,
        columns);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_q6_k_gemv_residual(
    const uint8_t *weights,
    const uint8_t *input,
    const float *residual,
    float *output,
    size_t rows,
    size_t columns,
    void *stream) {
    q6_q8_gemv_warp<<<static_cast<unsigned int>(rows),
                       kGemvWarpsPerBlock * 32, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        weights, reinterpret_cast<const Q8_1Block *>(input), residual,
        output, columns);
    return static_cast<int>(launch_status());
}

template <int Positions>
void launch_q6_k_gemv_multi_width(
    const uint8_t *weights,
    const uint8_t *input,
    const float *residual,
    float *output,
    size_t rows,
    size_t columns,
    cudaStream_t stream) {
    q6_q8_gemv_multi_warp<Positions><<<
        static_cast<unsigned int>(rows), kGemvWarpsPerBlock * 32, 0,
        stream>>>(weights, reinterpret_cast<const Q8_1Block *>(input),
                  residual, output, rows, columns);
}

extern "C" int ie_launch_q6_k_gemv_multi(
    const uint8_t *weights,
    const uint8_t *input,
    const float *residual,
    float *output,
    size_t rows,
    size_t columns,
    size_t positions,
    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (positions) {
    case 1: launch_q6_k_gemv_multi_width<1>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 2: launch_q6_k_gemv_multi_width<2>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 3: launch_q6_k_gemv_multi_width<3>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 4: launch_q6_k_gemv_multi_width<4>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 5: launch_q6_k_gemv_multi_width<5>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 6: launch_q6_k_gemv_multi_width<6>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 7: launch_q6_k_gemv_multi_width<7>(weights, input, residual, output, rows, columns, cuda_stream); break;
    case 8: launch_q6_k_gemv_multi_width<8>(weights, input, residual, output, rows, columns, cuda_stream); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(launch_status());
}

template <int Positions>
void launch_quant_gemv_group_multi_width(
    const uint8_t *first_weights,
    const uint8_t *second_weights,
    const uint8_t *third_weights,
    const uint8_t *input,
    float *first_output,
    float *second_output,
    float *third_output,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t columns,
    int first_q4,
    int second_q4,
    int third_q4,
    cudaStream_t stream) {
    quant_gemv_group_multi_warp<Positions><<<
        static_cast<unsigned int>(first_rows + second_rows + third_rows),
        kGemvWarpsPerBlock * 32, 0, stream>>>(
        first_weights, second_weights, third_weights,
        reinterpret_cast<const Q8_1Block *>(input), first_output,
        second_output, third_output, first_rows, second_rows, third_rows,
        columns, first_q4, second_q4, third_q4);
}

int launch_quant_gemv_group_multi(
    const uint8_t *first_weights,
    const uint8_t *second_weights,
    const uint8_t *third_weights,
    const uint8_t *input,
    float *first_output,
    float *second_output,
    float *third_output,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t columns,
    int first_q4,
    int second_q4,
    int third_q4,
    size_t positions,
    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define LAUNCH_GROUP(POSITIONS) \
    launch_quant_gemv_group_multi_width<POSITIONS>( \
        first_weights, second_weights, third_weights, input, first_output, \
        second_output, third_output, first_rows, second_rows, third_rows, \
        columns, first_q4, second_q4, third_q4, cuda_stream)
    switch (positions) {
    case 1: LAUNCH_GROUP(1); break;
    case 2: LAUNCH_GROUP(2); break;
    case 3: LAUNCH_GROUP(3); break;
    case 4: LAUNCH_GROUP(4); break;
    case 5: LAUNCH_GROUP(5); break;
    case 6: LAUNCH_GROUP(6); break;
    case 7: LAUNCH_GROUP(7); break;
    case 8: LAUNCH_GROUP(8); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
    }
#undef LAUNCH_GROUP
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_quant_gemv_group_multi(
    const uint8_t *first_weights,
    const uint8_t *second_weights,
    const uint8_t *third_weights,
    const uint8_t *input,
    float *first_output,
    float *second_output,
    float *third_output,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t columns,
    int first_q4,
    int second_q4,
    int third_q4,
    size_t positions,
    void *stream) {
    return launch_quant_gemv_group_multi(
        first_weights, second_weights, third_weights, input, first_output,
        second_output, third_output, first_rows, second_rows, third_rows,
        columns, first_q4, second_q4, third_q4, positions, stream);
}

extern "C" int ie_launch_qkv_gemv(
    const uint8_t *query_weights,
    const uint8_t *key_weights,
    const uint8_t *value_weights,
    const float *input,
    const uint8_t *quantized_input,
    const int32_t *quantized_sums,
    float *query,
    float *key,
    float *value,
    size_t query_rows,
    size_t key_rows,
    size_t value_rows,
    size_t columns,
    int query_q4,
    int key_q4,
    int value_q4,
    void *stream) {
    const size_t total_rows = query_rows + key_rows + value_rows;
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (columns == 4096) {
        quant_qkv_gemv_warp<16><<<static_cast<unsigned int>(total_rows),
                                   kGemvWarpsPerBlock * 32, 0,
                                   cuda_stream>>>(
            query_weights, key_weights, value_weights, input,
            reinterpret_cast<const Q8_1Block *>(quantized_input),
            query, key, value, query_rows, key_rows, value_rows, columns,
            query_q4, key_q4, value_q4);
    } else {
        quant_qkv_gemv_warp<0><<<static_cast<unsigned int>(total_rows),
                                  kGemvWarpsPerBlock * 32, 0,
                                  cuda_stream>>>(
            query_weights, key_weights, value_weights, input,
            reinterpret_cast<const Q8_1Block *>(quantized_input),
            query, key, value, query_rows, key_rows, value_rows, columns,
            query_q4, key_q4, value_q4);
    }
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rms_norm(const float *input,
                                   const float *weight,
                                   float *output,
                                   size_t rows,
                                   size_t columns,
                                   float epsilon,
                                   void *stream) {
    rms_norm_kernel<false, false><<<
        static_cast<unsigned int>(rows), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        input, nullptr, weight, nullptr, output, rows, columns, epsilon);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rms_norm_q8_parallel(
    const float *input,
    const float *weight,
    float *output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t rows,
    size_t columns,
    float epsilon,
    void *stream) {
    const size_t q8_groups = columns / kBlockThreads;
    rms_norm_q8_parallel_kernel<<<
        static_cast<unsigned int>(rows * q8_groups), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        input, weight, output,
        reinterpret_cast<Q8_1Block *>(quantized_output), quantized_sums,
        columns, epsilon);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rms_norm_residual(const float *left,
                                            const float *right,
                                            const float *weight,
                                            float *output,
                                            size_t rows,
                                            size_t columns,
                                            float epsilon,
                                            void *stream) {
    rms_norm_kernel<true, false><<<
        static_cast<unsigned int>(rows), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        left, right, weight, nullptr, output, rows, columns, epsilon);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rms_norm_residual_store(
    const float *left,
    const float *right,
    const float *weight,
    float *residual,
    float *output,
    size_t rows,
    size_t columns,
    float epsilon,
    void *stream) {
    rms_norm_kernel<true, true><<<
        static_cast<unsigned int>(rows), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        left, right, weight, residual, output, rows, columns, epsilon);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rms_norm_rope(
    const float *input,
    const float *weight,
    float *output,
    size_t rows,
    size_t columns,
    size_t position,
    float epsilon,
    float theta,
    void *stream) {
    rms_norm_rope_kernel<false, false, false, false><<<
        static_cast<unsigned int>(rows), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        input, weight, output, rows, nullptr, nullptr, nullptr, 0,
        columns, position, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
        epsilon, theta);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rms_norm_rope_device_position(
    const float *input,
    const float *weight,
    float *output,
    size_t rows,
    size_t columns,
    const uint32_t *position,
    float epsilon,
    float theta,
    void *stream) {
    rms_norm_rope_kernel<true, false, false, false><<<
        static_cast<unsigned int>(rows), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        input, weight, output, rows, nullptr, nullptr, nullptr, 0,
        columns, 0, position, nullptr, nullptr, nullptr, nullptr, 0,
        epsilon, theta);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_qk_norm_rope(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    size_t columns,
    const float *rope_table,
    float epsilon,
    void *stream) {
    rms_norm_rope_kernel<false, true, false, false><<<
        static_cast<unsigned int>(query_rows + key_rows), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        query, query_weight, query_output, query_rows,
        key, key_weight, key_output, key_rows, columns, 0, nullptr,
        reinterpret_cast<const float2 *>(rope_table), nullptr, nullptr,
        nullptr, 0, epsilon, 0.0f);
    return static_cast<int>(launch_status());
}

template <bool DevicePosition, bool F16Cache>
int launch_qk_norm_rope_kv_append(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    void *key_cache,
    void *value_cache,
    size_t columns,
    size_t max_context,
    size_t host_position,
    const uint32_t *device_position,
    const float *rope_table,
    float epsilon,
    void *stream) {
    rms_norm_rope_kernel<DevicePosition, true, true, F16Cache><<<
        static_cast<unsigned int>(query_rows + key_rows), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        query, query_weight, query_output, query_rows,
        key, key_weight, key_output, key_rows, columns, host_position,
        device_position, reinterpret_cast<const float2 *>(rope_table),
        value, key_cache, value_cache, max_context, epsilon, 0.0f);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_qk_norm_rope_kv_append(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    float *key_cache,
    float *value_cache,
    size_t columns,
    size_t max_context,
    size_t position,
    const float *rope_table,
    float epsilon,
    void *stream) {
    return launch_qk_norm_rope_kv_append<false, false>(
        query, query_weight, query_output, query_rows, key, key_weight,
        key_output, key_rows, value, key_cache, value_cache, columns,
        max_context, position, nullptr, rope_table, epsilon, stream);
}

extern "C" int ie_launch_qk_norm_rope_kv_append_f16(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    uint16_t *key_cache,
    uint16_t *value_cache,
    size_t columns,
    size_t max_context,
    size_t position,
    const float *rope_table,
    float epsilon,
    void *stream) {
    return launch_qk_norm_rope_kv_append<false, true>(
        query, query_weight, query_output, query_rows, key, key_weight,
        key_output, key_rows, value, key_cache, value_cache, columns,
        max_context, position, nullptr, rope_table, epsilon, stream);
}

extern "C" int ie_launch_qk_norm_rope_kv_append_device_position(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    float *key_cache,
    float *value_cache,
    size_t columns,
    size_t max_context,
    const uint32_t *position,
    const float *rope_table,
    float epsilon,
    void *stream) {
    return launch_qk_norm_rope_kv_append<true, false>(
        query, query_weight, query_output, query_rows, key, key_weight,
        key_output, key_rows, value, key_cache, value_cache, columns,
        max_context, 0, position, rope_table, epsilon, stream);
}

extern "C" int ie_launch_qk_norm_rope_kv_append_f16_device_position(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    uint16_t *key_cache,
    uint16_t *value_cache,
    size_t columns,
    size_t max_context,
    const uint32_t *position,
    const float *rope_table,
    float epsilon,
    void *stream) {
    return launch_qk_norm_rope_kv_append<true, true>(
        query, query_weight, query_output, query_rows, key, key_weight,
        key_output, key_rows, value, key_cache, value_cache, columns,
        max_context, 0, position, rope_table, epsilon, stream);
}

template <bool F16Cache, typename Cache>
int launch_verify_qk_norm_rope_kv_append(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    Cache *key_cache,
    Cache *value_cache,
    size_t columns,
    size_t max_context,
    size_t start_position,
    size_t positions,
    const double *inverse_frequencies,
    float *rope_table,
    float epsilon,
    void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const size_t pairs = columns / 2;
    prepare_verify_rope_tables_kernel<<<
        element_grid(pairs * positions), kBlockThreads, 0, cuda_stream>>>(
        inverse_frequencies, reinterpret_cast<float2 *>(rope_table), pairs,
        start_position, positions);
    cudaError_t status = launch_status();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    rms_norm_rope_verify_kernel<F16Cache><<<
        static_cast<unsigned int>((query_rows + key_rows) * positions),
        kBlockThreads, 0, cuda_stream>>>(
        query, query_weight, query_output, query_rows, key, key_weight,
        key_output, key_rows, value, key_cache, value_cache, columns,
        max_context, start_position, positions,
        reinterpret_cast<const float2 *>(rope_table), epsilon);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_verify_qk_norm_rope_kv_append(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    float *key_cache,
    float *value_cache,
    size_t columns,
    size_t max_context,
    size_t start_position,
    size_t positions,
    const double *inverse_frequencies,
    float *rope_table,
    float epsilon,
    void *stream) {
    return launch_verify_qk_norm_rope_kv_append<false>(
        query, query_weight, query_output, query_rows, key, key_weight,
        key_output, key_rows, value, key_cache, value_cache, columns,
        max_context, start_position, positions, inverse_frequencies,
        rope_table, epsilon, stream);
}

extern "C" int ie_launch_verify_qk_norm_rope_kv_append_f16(
    const float *query,
    const float *query_weight,
    float *query_output,
    size_t query_rows,
    const float *key,
    const float *key_weight,
    float *key_output,
    size_t key_rows,
    const float *value,
    uint16_t *key_cache,
    uint16_t *value_cache,
    size_t columns,
    size_t max_context,
    size_t start_position,
    size_t positions,
    const double *inverse_frequencies,
    float *rope_table,
    float epsilon,
    void *stream) {
    return launch_verify_qk_norm_rope_kv_append<true>(
        query, query_weight, query_output, query_rows, key, key_weight,
        key_output, key_rows, value, key_cache, value_cache, columns,
        max_context, start_position, positions, inverse_frequencies,
        rope_table, epsilon, stream);
}

extern "C" int ie_launch_prepare_rope_table(
    const double *inverse_frequencies,
    float *table,
    size_t pairs,
    size_t position,
    void *stream) {
    prepare_rope_table_kernel<false><<<element_grid(pairs), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        inverse_frequencies, reinterpret_cast<float2 *>(table), pairs,
        position, nullptr);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_prepare_rope_table_device_position(
    const double *inverse_frequencies,
    float *table,
    size_t pairs,
    const uint32_t *position,
    void *stream) {
    prepare_rope_table_kernel<true><<<element_grid(pairs), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        inverse_frequencies, reinterpret_cast<float2 *>(table), pairs,
        0, position);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rope_neox(float *values,
                                    const uint32_t *positions,
                                    size_t tokens,
                                    size_t heads,
                                    size_t head_dim,
                                    float theta,
                                    void *stream) {
    const size_t pairs = tokens * heads * head_dim / 2;
    rope_neox_kernel<<<element_grid(pairs), kBlockThreads, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        values, positions, tokens, heads, head_dim, theta);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rope_neox_at(float *values,
                                       size_t position,
                                       size_t tokens,
                                       size_t heads,
                                       size_t head_dim,
                                       float theta,
                                       void *stream) {
    const size_t pairs = tokens * heads * head_dim / 2;
    rope_neox_at_kernel<<<element_grid(pairs), kBlockThreads, 0,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
        values, position, tokens, heads, head_dim, theta);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_rope_at_frequencies(
    float *values,
    size_t position,
    size_t tokens,
    size_t heads,
    size_t head_dim,
    const double *inverse_frequencies,
    bool adjacent_pairs,
    void *stream) {
    const size_t pairs = tokens * heads * head_dim / 2;
    rope_at_frequencies_kernel<<<
        element_grid(pairs), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        values, position, tokens, heads, head_dim, inverse_frequencies,
        adjacent_pairs);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_swiglu(const float *gate,
                                 const float *up,
                                 float *output,
                                 size_t elements,
                                 void *stream) {
    swiglu_kernel<<<element_grid(elements), kBlockThreads, 0,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
        gate, up, output, elements);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_swiglu_q8(
    const float *gate,
    const float *up,
    float *output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t elements,
    void *stream) {
    swiglu_q8_kernel<<<element_grid(elements), kBlockThreads, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        gate, up, output, reinterpret_cast<Q8_1Block *>(quantized_output),
        quantized_sums, elements);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_residual_add(const float *left,
                                       const float *right,
                                       float *output,
                                       size_t elements,
                                       void *stream) {
    residual_add_kernel<<<element_grid(elements), kBlockThreads, 0,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
        left, right, output, elements);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_write_u32(uint32_t *output,
                                    uint32_t value,
                                    void *stream) {
    write_u32_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        output, value);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_increment_u32(uint32_t *output,
                                         void *stream) {
    increment_u32_kernel<<<1, 1, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(output);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append(const float *key,
                                     const float *value,
                                     float *key_cache,
                                     float *value_cache,
                                     size_t n_head_kv,
                                     size_t head_dim,
                                     size_t max_context,
                                     size_t position,
                                     void *stream) {
    const size_t elements = n_head_kv * head_dim;
    kv_append_kernel<false, false><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, key_cache, value_cache, n_head_kv, head_dim,
        max_context, position, nullptr);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append_device_position(
    const float *key,
    const float *value,
    float *key_cache,
    float *value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    const uint32_t *position,
    void *stream) {
    const size_t elements = n_head_kv * head_dim;
    kv_append_kernel<false, true><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, key_cache, value_cache, n_head_kv, head_dim,
        max_context, 0, position);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append_f16(
    const float *key,
    const float *value,
    uint16_t *key_cache,
    uint16_t *value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t position,
    void *stream) {
    const size_t elements = n_head_kv * head_dim;
    kv_append_kernel<true, false><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, key_cache, value_cache, n_head_kv, head_dim,
        max_context, position, nullptr);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append_f16_device_position(
    const float *key,
    const float *value,
    uint16_t *key_cache,
    uint16_t *value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    const uint32_t *position,
    void *stream) {
    const size_t elements = n_head_kv * head_dim;
    kv_append_kernel<true, true><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, key_cache, value_cache, n_head_kv, head_dim,
        max_context, 0, position);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append_q8(
    const float *key,
    const float *value,
    uint8_t *key_cache,
    uint8_t *value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t position,
    void *stream) {
    const size_t blocks = n_head_kv * head_dim / kQ8BlockElements;
    kv_append_q8_kernel<false><<<
        element_grid(blocks * 32), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, reinterpret_cast<Q8KVBlock *>(key_cache),
        reinterpret_cast<Q8KVBlock *>(value_cache), n_head_kv, head_dim,
        max_context, position, nullptr);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append_q8_device_position(
    const float *key,
    const float *value,
    uint8_t *key_cache,
    uint8_t *value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    const uint32_t *position,
    void *stream) {
    const size_t blocks = n_head_kv * head_dim / kQ8BlockElements;
    kv_append_q8_kernel<true><<<
        element_grid(blocks * 32), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, reinterpret_cast<Q8KVBlock *>(key_cache),
        reinterpret_cast<Q8KVBlock *>(value_cache), n_head_kv, head_dim,
        max_context, 0, position);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append_chunk(
    const float *key,
    const float *value,
    float *key_cache,
    float *value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t tokens,
    void *stream) {
    const size_t elements = tokens * n_head_kv * head_dim;
    kv_append_chunk_kernel<false><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, key_cache, value_cache, n_head_kv, head_dim,
        max_context, start_position, tokens);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_kv_append_chunk_f16(
    const float *key,
    const float *value,
    uint16_t *key_cache,
    uint16_t *value_cache,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t tokens,
    void *stream) {
    const size_t elements = tokens * n_head_kv * head_dim;
    kv_append_chunk_kernel<true><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        key, value, key_cache, value_cache, n_head_kv, head_dim,
        max_context, start_position, tokens);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_embedding_q4_k(const uint8_t *table,
                                         float *output,
                                         size_t rows,
                                         size_t columns,
                                         size_t row,
                                         void *stream) {
    embedding_kernel<true><<<1, kBlockThreads, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        table, nullptr, output, rows, columns, row);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_embedding_q6_k(const uint8_t *table,
                                         float *output,
                                         size_t rows,
                                         size_t columns,
                                         size_t row,
                                         void *stream) {
    embedding_kernel<false><<<1, kBlockThreads, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        table, nullptr, output, rows, columns, row);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_embedding_q4_k_device_row(
    const uint8_t *table,
    const uint32_t *row,
    float *output,
    size_t rows,
    size_t columns,
    void *stream) {
    embedding_kernel<true><<<1, kBlockThreads, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        table, row, output, rows, columns, 0);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_embedding_q6_k_device_row(
    const uint8_t *table,
    const uint32_t *row,
    float *output,
    size_t rows,
    size_t columns,
    void *stream) {
    embedding_kernel<false><<<1, kBlockThreads, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        table, row, output, rows, columns, 0);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_embedding_q4_k_batch(
    const uint8_t *table,
    const uint32_t *rows,
    float *output,
    size_t table_rows,
    size_t columns,
    size_t tokens,
    void *stream) {
    const size_t elements = tokens * columns;
    embedding_batch_kernel<true><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        table, rows, output, table_rows, columns, tokens);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_embedding_q6_k_batch(
    const uint8_t *table,
    const uint32_t *rows,
    float *output,
    size_t table_rows,
    size_t columns,
    size_t tokens,
    void *stream) {
    const size_t elements = tokens * columns;
    embedding_batch_kernel<false><<<
        element_grid(elements), kBlockThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        table, rows, output, table_rows, columns, tokens);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_copy_f32_row(const float *input,
                                        float *output,
                                        size_t row,
                                        size_t columns,
                                        void *stream) {
    copy_f32_row_kernel<<<element_grid(columns), kBlockThreads, 0,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
        input, output, row, columns);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_attention_decode(const float *query,
                                           const float *key_cache,
                                           const float *value_cache,
                                           float *output,
                                           float *partial_max,
                                           float *partial_sum,
                                           float *partial_output,
                                           uint8_t *quantized_output,
                                           int32_t *quantized_sums,
                                           size_t n_head,
                                           size_t n_head_kv,
                                           size_t head_dim,
                                           size_t max_context,
                                           size_t context_length,
                                           void *stream) {
    return launch_attention<false, false, false>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums, n_head, n_head_kv,
        head_dim, max_context, context_length, nullptr, stream);
}

extern "C" int ie_launch_attention_decode_device_position(
    const float *query,
    const float *key_cache,
    const float *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    const uint32_t *position,
    void *stream) {
    return launch_attention<false, false, true>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums, n_head, n_head_kv,
        head_dim, max_context, 0, position, stream);
}

extern "C" int ie_launch_attention_decode_f16(
    const float *query,
    const uint16_t *key_cache,
    const uint16_t *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t context_length,
    void *stream) {
    return launch_attention<true, false, false>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums, n_head, n_head_kv,
        head_dim, max_context, context_length, nullptr, stream);
}

extern "C" int ie_launch_attention_decode_f16_device_position(
    const float *query,
    const uint16_t *key_cache,
    const uint16_t *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    const uint32_t *position,
    void *stream) {
    return launch_attention<true, false, true>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums, n_head, n_head_kv,
        head_dim, max_context, 0, position, stream);
}

extern "C" int ie_launch_attention_decode_q8(
    const float *query,
    const uint8_t *key_cache,
    const uint8_t *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t context_length,
    void *stream) {
    return launch_attention<false, true, false>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums, n_head, n_head_kv,
        head_dim, max_context, context_length, nullptr, stream);
}

extern "C" int ie_launch_attention_decode_q8_device_position(
    const float *query,
    const uint8_t *key_cache,
    const uint8_t *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    const uint32_t *position,
    void *stream) {
    return launch_attention<false, true, true>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums, n_head, n_head_kv,
        head_dim, max_context, 0, position, stream);
}

template <bool F16, typename Cache>
int launch_verify_attention(
    const float *query,
    const Cache *key_cache,
    const Cache *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t positions,
    void *stream) {
    size_t split_count = 0;
    cudaError_t status =
        attention_split_count<F16>(n_head, max_context, &split_count);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const dim3 partial_grid(
        static_cast<unsigned int>(n_head),
        static_cast<unsigned int>(split_count),
        static_cast<unsigned int>(positions));
    const size_t positions_per_split =
        (max_context + split_count - 1) / split_count;
    if constexpr (F16) {
        if (head_dim == 128 && positions_per_split <= kAttentionKvTile) {
            attention_partial_tiled_f16_kernel<false, true><<<
                partial_grid, kAttentionTileThreads, 0, cuda_stream>>>(
                query, reinterpret_cast<const __half *>(key_cache),
                reinterpret_cast<const __half *>(value_cache), partial_max,
                partial_sum, partial_output, n_head, n_head_kv, head_dim,
                max_context, split_count, start_position + 1, nullptr);
        } else {
            attention_partial_kernel<F16, false, false, true><<<
                partial_grid, kAttentionThreads, 0, cuda_stream>>>(
                query, key_cache, value_cache, partial_max, partial_sum,
                partial_output, n_head, n_head_kv, head_dim, max_context,
                split_count, start_position + 1, nullptr);
        }
    } else {
        attention_partial_kernel<F16, false, false, true><<<
            partial_grid, kAttentionThreads, 0, cuda_stream>>>(
            query, key_cache, value_cache, partial_max, partial_sum,
            partial_output, n_head, n_head_kv, head_dim, max_context,
            split_count, start_position + 1, nullptr);
    }
    status = launch_status();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    attention_reduce_kernel<<<
        dim3(static_cast<unsigned int>(n_head),
             static_cast<unsigned int>(positions)),
        kAttentionThreads, 0, cuda_stream>>>(
        partial_max, partial_sum, partial_output, output,
        reinterpret_cast<Q8_1Block *>(quantized_output), quantized_sums,
        n_head, head_dim, split_count, positions);
    return static_cast<int>(launch_status());
}

extern "C" int ie_launch_verify_attention(
    const float *query,
    const float *key_cache,
    const float *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t positions,
    void *stream) {
    return launch_verify_attention<false>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums,
        n_head, n_head_kv, head_dim, max_context,
        start_position, positions, stream);
}

extern "C" int ie_launch_verify_attention_f16(
    const float *query,
    const uint16_t *key_cache,
    const uint16_t *value_cache,
    float *output,
    float *partial_max,
    float *partial_sum,
    float *partial_output,
    uint8_t *quantized_output,
    int32_t *quantized_sums,
    size_t n_head,
    size_t n_head_kv,
    size_t head_dim,
    size_t max_context,
    size_t start_position,
    size_t positions,
    void *stream) {
    return launch_verify_attention<true>(
        query, key_cache, value_cache, output, partial_max, partial_sum,
        partial_output, quantized_output, quantized_sums,
        n_head, n_head_kv, head_dim, max_context,
        start_position, positions, stream);
}

extern "C" int ie_launch_argmax(const float *input,
                                 size_t elements,
                                 uint32_t *output,
                                 float *partial_values,
                                 uint32_t *partial_indices,
                                 void *stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const size_t values_per_block =
        kBlockThreads * kArgmaxItemsPerThread;
    const size_t partial_count =
        (elements + values_per_block - 1) / values_per_block;
    argmax_partial_kernel<<<static_cast<unsigned int>(partial_count),
                            kBlockThreads, 0, cuda_stream>>>(
        input, elements, partial_values, partial_indices);
    cudaError_t status = launch_status();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    argmax_reduce_kernel<<<1, kBlockThreads, 0, cuda_stream>>>(
        partial_values, partial_indices, partial_count, output);
    return static_cast<int>(launch_status());
}
