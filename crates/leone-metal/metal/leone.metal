#include <metal_stdlib>
using namespace metal;

#define LEONE_PREFILL_TOKEN_TILE 4
#define PREFILL_MATRIX_TILE_TOKENS 16
#define PREFILL_MATRIX_TILE_ROWS 32
#define PREFILL_MATRIX_K_TILE 32
#define PREFILL_MATRIX_DIM 8

struct DispatchArgs {
    uint op;
    uint format;
    uint rows;
    uint columns;
    uint tokens;
    uint position;
    uint start_position;
    uint max_context;
    uint gather_stride;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint table_rows;
    uint row;
    uint pairing;
    float epsilon;
    float theta;
    uint threads;
    uint batch_rows;
    uint tile_tokens;
    uint tile_count;
    uint group_count;
    uint threadgroup_bytes;
    uint prefill_tile_rows;
    uint prefill_k_tile;
};

inline float f16_value(device const uchar *bytes, uint offset) {
    ushort bits = ushort(bytes[offset]) | (ushort(bytes[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

inline float q4_value(device const uchar *weights, uint index) {
    uint block = index / 256;
    uint local = index & 255;
    device const uchar *source = weights + block * 144;
    float d = f16_value(source, 0);
    float dmin = f16_value(source, 2);
    uint chunk = local / 64;
    uint within = local & 31;
    uint scale_index = chunk * 2;
    uint scale0;
    uint min0;
    uint scale1;
    uint min1;
    if (scale_index < 4) {
        scale0 = source[4 + scale_index] & 63;
        min0 = source[8 + scale_index] & 63;
        scale1 = source[4 + scale_index + 1] & 63;
        min1 = source[8 + scale_index + 1] & 63;
    } else {
        scale0 = (source[8 + scale_index] & 15) | ((source[4 + scale_index - 4] >> 6) << 4);
        min0 = (source[8 + scale_index] >> 4) | ((source[4 + scale_index] >> 6) << 4);
        scale1 = (source[8 + scale_index + 1] & 15) | ((source[4 + scale_index + 1 - 4] >> 6) << 4);
        min1 = (source[8 + scale_index + 1] >> 4) | ((source[4 + scale_index + 1] >> 6) << 4);
    }
    uchar packed = source[16 + chunk * 32 + within];
    if ((local & 32) == 0) {
        return d * float(scale0) * float(packed & 15) - dmin * float(min0);
    }
    return d * float(scale1) * float(packed >> 4) - dmin * float(min1);
}

inline float q6_value(device const uchar *weights, uint index) {
    uint block = index / 256;
    uint local = index & 255;
    device const uchar *source = weights + block * 210;
    uint half_index = local / 128;
    uint within = local & 127;
    uint group = within / 32;
    uint lane = within & 31;
    uint low_offset = half_index * 64;
    uint high_offset = 128 + half_index * 32;
    uchar low = source[low_offset + (group & 1) * 32 + lane];
    uchar high = source[high_offset + lane];
    uint shift = (group & 3) * 2;
    int quant = int((group < 2 ? low & 15 : low >> 4) | (((high >> shift) & 3) << 4)) - 32;
    uint scale_index = half_index * 8 + group * 2;
    if ((within & 31) >= 16) {
        scale_index += 1;
    }
    int scale = int(source[192 + scale_index]);
    if (scale >= 128) {
        scale -= 256;
    }
    ushort d_bits = ushort(source[208]) | (ushort(source[209]) << 8);
    return float(as_type<half>(d_bits)) * float(scale) * float(quant);
}

inline float weight_value(device const uchar *weights, uint index, uint format) {
    return format == 0 ? q4_value(weights, index) : q6_value(weights, index);
}

inline float inverse_frequency(constant DispatchArgs &args, device const float *frequencies, uint pair) {
    return frequencies[pair];
}

inline float rotate_component(
    float value,
    float mate,
    uint pair,
    uint position,
    uint component,
    constant DispatchArgs &args,
    device const float *frequencies
) {
    float angle = float(position) * inverse_frequency(args, frequencies, pair);
    float cosine = cos(angle);
    float sine = sin(angle);
    return component == 0 ? value * cosine - mate * sine : value * cosine + mate * sine;
}

inline float normalized_value(
    device const float *input,
    device const float *weight,
    uint row,
    uint column,
    uint columns,
    float inverse_norm
) {
    return input[row * columns + column] * inverse_norm * weight[column];
}

inline float rms_inverse(
    device const float *input,
    uint row,
    uint columns,
    float epsilon
) {
    float sum = 0.0f;
    for (uint index = 0; index < columns; ++index) {
        float value = input[row * columns + index];
        sum += value * value;
    }
    return rsqrt(sum / float(columns) + epsilon);
}

inline float rms_residual_inverse(
    device const float *left,
    device const float *right,
    uint row,
    uint columns,
    float epsilon
) {
    float sum = 0.0f;
    for (uint index = 0; index < columns; ++index) {
        float value = left[row * columns + index] + right[row * columns + index];
        sum += value * value;
    }
    return rsqrt(sum / float(columns) + epsilon);
}

inline void op_gemv(
    device const uchar *bytes0,
    device const float *f0,
    device float *f1,
    constant DispatchArgs &args,
    threadgroup float *partial,
    uint lane,
    uint row
) {
    float sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256) {
        sum += weight_value(bytes0, row * args.columns + column, args.format) * f0[column];
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        f1[row] = partial[0];
    }
}

inline void op_gemv_residual(
    device const uchar *bytes0,
    device const float *f0,
    device float *f1,
    device const float *f2,
    constant DispatchArgs &args,
    threadgroup float *partial,
    uint lane,
    uint row
) {
    float sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256) {
        sum += weight_value(bytes0, row * args.columns + column, args.format) * f0[column];
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        f1[row] = partial[0] + f2[row];
    }
}

inline void op_prefill_gemm(
    device const uchar *bytes0,
    device const float *f0,
    device float *f1,
    constant DispatchArgs &args,
    threadgroup float *partial,
    uint lane,
    uint output
) {
    uint row = output % args.rows;
    uint token = output / args.rows;
    float sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256) {
        sum += weight_value(bytes0, row * args.columns + column, args.format)
            * f0[token * args.columns + column];
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        f1[output] = partial[0];
    }
}

inline void prefill_token_tile_accumulate(
    device const uchar *bytes0,
    device const float *f0,
    constant DispatchArgs &args,
    uint row,
    uint token_start,
    uint lane,
    thread float *sums
) {
    constexpr uint TILE_TOKENS = LEONE_PREFILL_TOKEN_TILE;
    for (uint column = lane; column < args.columns; column += 256) {
        float weight = weight_value(bytes0, row * args.columns + column, args.format);
        for (uint token_offset = 0; token_offset < TILE_TOKENS; ++token_offset) {
            uint token = token_start + token_offset;
            if (token < args.tokens) {
                sums[token_offset] += weight * f0[token * args.columns + column];
            }
        }
    }
}

inline void prefill_token_tile_store(
    threadgroup float *partial,
    thread const float *sums,
    uint lane
) {
    constexpr uint TILE_TOKENS = LEONE_PREFILL_TOKEN_TILE;
    for (uint token_offset = 0; token_offset < TILE_TOKENS; ++token_offset) {
        partial[token_offset * 256 + lane] = sums[token_offset];
    }
}

inline void prefill_token_tile_reduce(threadgroup float *partial, uint lane) {
    constexpr uint TILE_TOKENS = LEONE_PREFILL_TOKEN_TILE;
    // Each token keeps the scalar path's lane order and reduction tree.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint token_offset = 0; token_offset < TILE_TOKENS; ++token_offset) {
                uint offset = token_offset * 256;
                partial[offset + lane] += partial[offset + lane + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void prefill_token_tile_write(
    device float *f1,
    constant DispatchArgs &args,
    threadgroup float *partial,
    uint row,
    uint token_start,
    uint lane
) {
    constexpr uint TILE_TOKENS = LEONE_PREFILL_TOKEN_TILE;
    if (lane == 0) {
        for (uint token_offset = 0; token_offset < TILE_TOKENS; ++token_offset) {
            uint token = token_start + token_offset;
            if (token < args.tokens) {
                f1[token * args.rows + row] = partial[token_offset * 256];
            }
        }
    }
}

inline void op_prefill_gemm_token_tile(
    device const uchar *bytes0,
    device const float *f0,
    device float *f1,
    constant DispatchArgs &args,
    threadgroup float *partial,
    uint lane,
    uint group_id
) {
    constexpr uint TILE_TOKENS = LEONE_PREFILL_TOKEN_TILE;
    uint row = group_id % args.rows;
    uint token_start = (group_id / args.rows) * TILE_TOKENS;
    float sums[TILE_TOKENS] = {0.0f, 0.0f, 0.0f, 0.0f};
    prefill_token_tile_accumulate(bytes0, f0, args, row, token_start, lane, sums);
    prefill_token_tile_store(partial, sums, lane);
    prefill_token_tile_reduce(partial, lane);
    prefill_token_tile_write(f1, args, partial, row, token_start, lane);
}

inline void prefill_matrix_stage_weights(
    device const uchar *bytes0,
    constant DispatchArgs &args,
    threadgroup float *weights_tile,
    uint row_start,
    uint k_base,
    uint lane
) {
    for (uint index = lane;
         index < PREFILL_MATRIX_TILE_ROWS * PREFILL_MATRIX_K_TILE;
         index += 256) {
        uint row = index / PREFILL_MATRIX_K_TILE;
        uint column = index % PREFILL_MATRIX_K_TILE;
        uint global_row = row_start + row;
        weights_tile[index] = global_row < args.rows
            ? weight_value(bytes0, global_row * args.columns + k_base + column, args.format)
            : 0.0f;
    }
}

inline void prefill_matrix_stage_activation(
    device const float *f0,
    constant DispatchArgs &args,
    threadgroup float *activation_tile,
    uint token_start,
    uint k_base,
    uint lane
) {
    for (uint index = lane;
         index < PREFILL_MATRIX_TILE_TOKENS * PREFILL_MATRIX_K_TILE;
         index += 256) {
        uint token = index / args.prefill_k_tile;
        uint column = index % args.prefill_k_tile;
        uint global_token = token_start + token;
        activation_tile[index] = global_token < args.tokens
            ? f0[global_token * args.columns + k_base + column]
            : 0.0f;
    }
}

inline void prefill_matrix_accumulate(
    threadgroup float *weights_tile,
    threadgroup float *activation_tile,
    constant DispatchArgs &args,
    thread simdgroup_float8x8 &accumulator,
    ushort simdgroup
) {
    uint token_group = simdgroup / 4;
    uint row_group = simdgroup & 3;
    threadgroup const float *activation_matrix =
        activation_tile + token_group * PREFILL_MATRIX_DIM * PREFILL_MATRIX_K_TILE;
    threadgroup const float *weight_matrix =
        weights_tile + row_group * PREFILL_MATRIX_DIM * PREFILL_MATRIX_K_TILE;
    for (uint k_step = 0; k_step < args.prefill_k_tile; k_step += PREFILL_MATRIX_DIM) {
        simdgroup_float8x8 activation;
        simdgroup_float8x8 weights;
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(
            activation,
            activation_matrix + k_step,
            PREFILL_MATRIX_K_TILE,
            0,
            false
        );
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(weights, weight_matrix + k_step, PREFILL_MATRIX_K_TILE, 0, true);
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_multiply_accumulate(accumulator, activation, weights, accumulator);
    }
}

inline void prefill_matrix_store(
    device float *f1,
    constant DispatchArgs &args,
    threadgroup float *result_tile,
    thread simdgroup_float8x8 &accumulator,
    uint row_start,
    uint token_start,
    uint lane,
    ushort simdgroup
) {
    uint token_group = simdgroup / 4;
    uint row_group = simdgroup & 3;
    simdgroup_store(
        accumulator,
        result_tile + token_group * PREFILL_MATRIX_DIM * PREFILL_MATRIX_TILE_ROWS
            + row_group * PREFILL_MATRIX_DIM,
        PREFILL_MATRIX_TILE_ROWS,
        0,
        false
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane;
         index < PREFILL_MATRIX_TILE_TOKENS * PREFILL_MATRIX_TILE_ROWS;
         index += 256) {
        uint token = index / PREFILL_MATRIX_TILE_ROWS;
        uint row = index % PREFILL_MATRIX_TILE_ROWS;
        uint global_token = token_start + token;
        uint global_row = row_start + row;
        if (global_token < args.tokens && global_row < args.rows) {
            f1[global_token * args.rows + global_row] = result_tile[index];
        }
    }
}

inline void op_prefill_gemm_simdgroup_matrix(
    device const uchar *bytes0,
    device const float *f0,
    device float *f1,
    constant DispatchArgs &args,
    threadgroup float *weights_tile,
    threadgroup float *activation_tile,
    threadgroup float *result_tile,
    uint lane,
    ushort simdgroup,
    uint group_id
) {
    uint row_tiles = (args.rows - 1) / args.prefill_tile_rows + 1;
    uint row_start = (group_id % row_tiles) * PREFILL_MATRIX_TILE_ROWS;
    uint token_start = (group_id / row_tiles) * args.tile_tokens;
    simdgroup_float8x8 accumulator =
        make_filled_simdgroup_matrix<float, PREFILL_MATRIX_DIM>(0.0f);

    // Host QuantMatrix validation makes columns a multiple of 256.
    // The 32-element K tile divides 256.
    for (uint k_base = 0; k_base < args.columns; k_base += args.prefill_k_tile) {
        prefill_matrix_stage_weights(bytes0, args, weights_tile, row_start, k_base, lane);
        prefill_matrix_stage_activation(f0, args, activation_tile, token_start, k_base, lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefill_matrix_accumulate(weights_tile, activation_tile, args, accumulator, simdgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    prefill_matrix_store(
        f1,
        args,
        result_tile,
        accumulator,
        row_start,
        token_start,
        lane,
        simdgroup
    );
}

inline void op_rms_norm(
    device const float *f0,
    device float *f1,
    device const float *f5,
    constant DispatchArgs &args,
    threadgroup float *partial,
    uint lane,
    uint row
) {
    float sum = 0.0f;
    for (uint column = lane; column < args.columns;) {
        float value = f0[row * args.columns + column];
        sum += value * value;
        if (args.columns - column <= 256) {
            break;
        }
        column += 256;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_norm = rsqrt(partial[0] / float(args.columns) + args.epsilon);
    // The barrier separates reduction reads from disjoint per-lane input/output accesses.
    for (uint column = lane; column < args.columns;) {
        uint index = row * args.columns + column;
        f1[index] = f0[index] * inverse_norm * f5[column];
        if (args.columns - column <= 256) {
            break;
        }
        column += 256;
    }
}

inline void op_rms_norm_rope(
    device const float *f0,
    device float *f1,
    device const float *f4,
    device const float *f5,
    constant DispatchArgs &args,
    uint gid
) {
    uint row = gid / args.columns;
    uint column = gid % args.columns;
    float inverse_norm = rms_inverse(f0, row, args.columns, args.epsilon);
    uint half_dim = args.columns / 2;
    uint pair;
    uint component;
    uint mate;
    if (args.pairing == 0) {
        pair = column < half_dim ? column : column - half_dim;
        component = column < half_dim ? 0 : 1;
        mate = column < half_dim ? column + half_dim : column - half_dim;
    } else {
        pair = column / 2;
        component = column & 1;
        mate = column ^ 1;
    }
    float value = normalized_value(f0, f5, row, column, args.columns, inverse_norm);
    float mate_value = normalized_value(f0, f5, row, mate, args.columns, inverse_norm);
    f1[row * args.columns + column] = rotate_component(
        value, mate_value, pair, args.position, component, args, f4);
}

inline void op_prefill_rms_norm_rope(
    device const float *f0,
    device float *f1,
    device const float *f4,
    device const float *f5,
    constant DispatchArgs &args,
    uint gid
) {
    uint row = gid / args.columns;
    uint column = gid % args.columns;
    float inverse_norm = rms_inverse(f0, row, args.columns, args.epsilon);
    uint half_dim = args.columns / 2;
    uint pair;
    uint component;
    uint mate;
    if (args.pairing == 0) {
        pair = column < half_dim ? column : column - half_dim;
        component = column < half_dim ? 0 : 1;
        mate = column < half_dim ? column + half_dim : column - half_dim;
    } else {
        pair = column / 2;
        component = column & 1;
        mate = column ^ 1;
    }
    float value = normalized_value(f0, f5, row, column, args.columns, inverse_norm);
    float mate_value = normalized_value(f0, f5, row, mate, args.columns, inverse_norm);
    uint position = args.start_position + row / args.n_head;
    f1[row * args.columns + column] = rotate_component(
        value, mate_value, pair, position, component, args, f4);
}

inline void op_rms_norm_residual(
    device const float *f0,
    device float *f1,
    device const float *f2,
    device const float *f5,
    constant DispatchArgs &args,
    uint gid
) {
    uint row = gid / args.columns;
    uint column = gid % args.columns;
    float inverse_norm = rms_residual_inverse(f0, f2, row, args.columns, args.epsilon);
    float value = f0[row * args.columns + column] + f2[row * args.columns + column];
    f1[row * args.columns + column] = value * inverse_norm * f5[column];
}

inline void op_rms_norm_residual_store(
    device const float *f0,
    device float *f1,
    device const float *f2,
    device float *f3,
    device const float *f5,
    constant DispatchArgs &args,
    uint gid
) {
    uint row = gid / args.columns;
    uint column = gid % args.columns;
    float inverse_norm = rms_residual_inverse(f0, f2, row, args.columns, args.epsilon);
    float value = f0[row * args.columns + column] + f2[row * args.columns + column];
    f3[row * args.columns + column] = value;
    f1[row * args.columns + column] = value * inverse_norm * f5[column];
}

inline void op_rope(
    device float *f1,
    device const float *f4,
    constant DispatchArgs &args,
    uint gid
) {
    uint pair_count = args.head_dim / 2;
    uint row = gid / pair_count;
    uint pair = gid % pair_count;
    uint position = args.position + row / args.n_head;
    uint first;
    uint second;
    if (args.pairing == 0) {
        first = pair;
        second = pair + pair_count;
    } else {
        first = pair * 2;
        second = first + 1;
    }
    float first_value = f1[row * args.head_dim + first];
    float second_value = f1[row * args.head_dim + second];
    float angle = float(position) * inverse_frequency(args, f4, pair);
    float cosine = cos(angle);
    float sine = sin(angle);
    f1[row * args.head_dim + first] = first_value * cosine - second_value * sine;
    f1[row * args.head_dim + second] = first_value * sine + second_value * cosine;
}

inline void op_swiglu(
    device const float *f0,
    device float *f1,
    device const float *f2,
    uint gid
) {
    float gate = f0[gid];
    f1[gid] = gate / (1.0f + exp(-gate)) * f2[gid];
}

inline void op_residual_add(
    device const float *f0,
    device float *f1,
    device const float *f2,
    uint gid
) {
    f1[gid] = f0[gid] + f2[gid];
}

inline void op_kv_append(
    device const float *f0,
    device const float *f2,
    device half *h1,
    device half *h2,
    constant DispatchArgs &args,
    uint gid
) {
    uint elements_per_token = args.n_head_kv * args.head_dim;
    uint head_element = gid % elements_per_token;
    uint head = head_element / args.head_dim;
    uint column = head_element % args.head_dim;
    uint destination = (head * args.max_context + args.start_position) * args.head_dim + column;
    h1[destination] = half(f0[head_element]);
    h2[destination] = half(f2[head_element]);
}

inline void op_kv_append_chunk(
    device const float *f0,
    device const float *f2,
    device half *h1,
    device half *h2,
    constant DispatchArgs &args,
    uint gid
) {
    uint elements_per_token = args.n_head_kv * args.head_dim;
    uint token = gid / elements_per_token;
    uint head_element = gid % elements_per_token;
    uint head = head_element / args.head_dim;
    uint column = head_element % args.head_dim;
    uint position = args.start_position + token;
    uint source = token * elements_per_token + head_element;
    uint destination = (head * args.max_context + position) * args.head_dim + column;
    h1[destination] = half(f0[source]);
    h2[destination] = half(f2[source]);
}

inline float reduce_attention_dot(
    threadgroup float *partial,
    uint lane,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The shared prefix preserves the decode reduction order before the SIMD tail.
    for (uint stride = 128; stride > 16; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Other SIMD widths use the barrier tail because shuffle lanes differ.
    if (threads_per_simdgroup == 32) {
        if (simdgroup == 0) {
            float reduced = partial[simd_lane];
            for (uint stride = 16; stride > 0; stride >>= 1) {
                float other = simd_shuffle_down(reduced, stride);
                if (simd_lane < stride) {
                    reduced += other;
                }
            }
            if (simd_lane == 0) {
                partial[0] = reduced;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (uint stride = 16; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    return partial[0];
}

inline float attention_dot_contiguous(
    device const float *query,
    device const half *key,
    uint query_base,
    uint key_base,
    uint head_dim,
    uint lane,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup,
    threadgroup float *partial
) {
    float sum = 0.0f;
    for (uint index = lane; index < head_dim; index += 256) {
        sum += query[query_base + index] * float(key[key_base + index]);
    }
    partial[lane] = sum;
    return reduce_attention_dot(
        partial,
        lane,
        simd_lane,
        simdgroup,
        threads_per_simdgroup
    );
}

inline void attention_tiled_contiguous(
    device const float *query,
    device float *output,
    device const half *key,
    device const half *value,
    uint query_base,
    uint output_base,
    uint context,
    uint kv_head,
    uint max_context,
    uint head_dim,
    threadgroup float *partial,
    threadgroup float *state,
    uint lane,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup
) {
    float scale = rsqrt(float(head_dim));
    if (lane == 0) {
        state[0] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_position = 0; key_position < context; ++key_position) {
        uint key_base = (kv_head * max_context + key_position) * head_dim;
        float score = attention_dot_contiguous(
            query,
            key,
            query_base,
            key_base,
            head_dim,
            lane,
            simd_lane,
            simdgroup,
            threads_per_simdgroup,
            partial
        );
        if (lane == 0) {
            state[0] = max(state[0], score * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = state[0];
    float result = 0.0f;
    float denominator = 0.0f;
    for (uint key_position = 0; key_position < context; ++key_position) {
        uint key_base = (kv_head * max_context + key_position) * head_dim;
        float score = attention_dot_contiguous(
            query,
            key,
            query_base,
            key_base,
            head_dim,
            lane,
            simd_lane,
            simdgroup,
            threads_per_simdgroup,
            partial
        );
        if (lane == 0) {
            partial[0] = exp(score * scale - maximum);
            denominator += partial[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float probability = partial[0];
        if (lane < head_dim) {
            result += probability * float(value[key_base + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        state[0] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < head_dim) {
        output[output_base + lane] = result / state[0];
    }
}

inline void attention_decode_tiled(
    device const float *query,
    device float *output,
    device const half *key,
    device const half *value,
    constant DispatchArgs &args,
    threadgroup float *partial,
    threadgroup float *state,
    uint lane,
    uint head,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup
) {
    uint head_base = head * args.head_dim;
    uint group_size = args.n_head / args.n_head_kv;
    uint kv_head = head / group_size;
    attention_tiled_contiguous(
        query,
        output,
        key,
        value,
        head_base,
        head_base,
        args.position + 1,
        kv_head,
        args.max_context,
        args.head_dim,
        partial,
        state,
        lane,
        simd_lane,
        simdgroup,
        threads_per_simdgroup
    );
}

inline void attention_prefill_tiled(
    device const float *query,
    device float *output,
    device const half *key,
    device const half *value,
    constant DispatchArgs &args,
    threadgroup float *partial,
    threadgroup float *state,
    uint lane,
    uint group,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup
) {
    uint token = group / args.n_head;
    uint head = group % args.n_head;
    uint head_base = head * args.head_dim;
    uint elements_per_query = args.n_head * args.head_dim;
    uint query_base = token * elements_per_query + head_base;
    uint position = args.start_position + token;
    uint group_size = args.n_head / args.n_head_kv;
    uint kv_head = head / group_size;
    attention_tiled_contiguous(
        query,
        output,
        key,
        value,
        query_base,
        query_base,
        position + 1,
        kv_head,
        args.max_context,
        args.head_dim,
        partial,
        state,
        lane,
        simd_lane,
        simdgroup,
        threads_per_simdgroup
    );
}

inline void attention_decode_scalar(
    device const float *f0,
    device float *f1,
    device const half *h0,
    device const half *h1,
    constant DispatchArgs &args,
    uint gid
) {
    uint head_element = gid % (args.n_head * args.head_dim);
    uint head = head_element / args.head_dim;
    uint column = head_element % args.head_dim;
    uint kv_head = head / (args.n_head / args.n_head_kv);
    float scale = rsqrt(float(args.head_dim));
    float maximum = -INFINITY;
    for (uint key_position = 0; key_position <= args.position; ++key_position) {
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = head_element - column + index;
            uint key_index = (kv_head * args.max_context + key_position) * args.head_dim + index;
            score += f0[query_index] * float(h0[key_index]);
        }
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    float result = 0.0f;
    for (uint key_position = 0; key_position <= args.position; ++key_position) {
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = head_element - column + index;
            uint key_index = (kv_head * args.max_context + key_position) * args.head_dim + index;
            score += f0[query_index] * float(h0[key_index]);
        }
        float probability = exp(score * scale - maximum);
        denominator += probability;
        uint value_index = (kv_head * args.max_context + key_position) * args.head_dim + column;
        result += probability * float(h1[value_index]);
    }
    f1[head_element] = result / denominator;
}

inline void attention_prefill(
    device const float *f0,
    device float *f1,
    device const half *h0,
    device const half *h1,
    constant DispatchArgs &args,
    uint gid
) {
    uint elements_per_query = args.n_head * args.head_dim;
    uint token = gid / elements_per_query;
    uint head_element = gid % elements_per_query;
    uint head = head_element / args.head_dim;
    uint column = head_element % args.head_dim;
    uint query_offset = token * elements_per_query + head_element;
    uint position = args.start_position + token;
    uint context = position + 1;
    uint kv_head = head / (args.n_head / args.n_head_kv);
    float scale = rsqrt(float(args.head_dim));
    float maximum = -INFINITY;
    for (uint key_position = 0; key_position < context; ++key_position) {
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = query_offset - column + index;
            uint key_index = (kv_head * args.max_context + key_position) * args.head_dim + index;
            score += f0[query_index] * float(h0[key_index]);
        }
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    float result = 0.0f;
    for (uint key_position = 0; key_position < context; ++key_position) {
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = query_offset - column + index;
            uint key_index = (kv_head * args.max_context + key_position) * args.head_dim + index;
            score += f0[query_index] * float(h0[key_index]);
        }
        float probability = exp(score * scale - maximum);
        denominator += probability;
        uint value_index = (kv_head * args.max_context + key_position) * args.head_dim + column;
        result += probability * float(h1[value_index]);
    }
    f1[query_offset] = result / denominator;
}

inline uint span_for_position(
    device const uint *descriptors,
    uint span_count,
    uint position
) {
    for (uint span = 0; span < span_count; ++span) {
        uint start = descriptors[span * 3];
        uint tokens = descriptors[span * 3 + 1];
        if (position >= start && position - start < tokens) {
            return span;
        }
    }
    return 0xffffffff;
}

inline float span_key_value(
    device const half *key0,
    device const half *key1,
    device const half *key2,
    device const half *key3,
    uint span,
    uint offset
) {
    if (span == 0) {
        return float(key0[offset]);
    }
    if (span == 1) {
        return float(key1[offset]);
    }
    if (span == 2) {
        return float(key2[offset]);
    }
    return float(key3[offset]);
}

inline float span_value_value(
    device const half *value0,
    device const half *value1,
    device const half *value2,
    device const half *value3,
    uint span,
    uint offset
) {
    if (span == 0) {
        return float(value0[offset]);
    }
    if (span == 1) {
        return float(value1[offset]);
    }
    if (span == 2) {
        return float(value2[offset]);
    }
    return float(value3[offset]);
}

inline float attention_dot_span(
    device const float *query,
    device const uint *descriptors,
    device const half *key0,
    device const half *key1,
    device const half *key2,
    device const half *key3,
    uint span_count,
    uint position,
    uint kv_head,
    uint head_dim,
    uint lane,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup,
    threadgroup float *partial
) {
    uint span = span_for_position(descriptors, span_count, position);
    uint local = position - descriptors[span * 3];
    uint capacity = descriptors[span * 3 + 2];
    uint key_base = (kv_head * capacity + local) * head_dim;
    float sum = 0.0f;
    for (uint index = lane; index < head_dim; index += 256) {
        sum += query[index] * span_key_value(key0, key1, key2, key3, span, key_base + index);
    }
    partial[lane] = sum;
    return reduce_attention_dot(
        partial,
        lane,
        simd_lane,
        simdgroup,
        threads_per_simdgroup
    );
}

inline void attention_tiled_spans(
    device const float *query,
    device float *output,
    device const uint *descriptors,
    device const half *key0,
    device const half *key1,
    device const half *key2,
    device const half *key3,
    device const half *value0,
    device const half *value1,
    device const half *value2,
    device const half *value3,
    uint span_count,
    uint query_base,
    uint output_base,
    uint context,
    uint kv_head,
    uint head_dim,
    threadgroup float *partial,
    threadgroup float *state,
    uint lane,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup
) {
    float scale = rsqrt(float(head_dim));
    if (lane == 0) {
        state[0] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_position = 0; key_position < context; ++key_position) {
        float score = attention_dot_span(
            query + query_base,
            descriptors,
            key0,
            key1,
            key2,
            key3,
            span_count,
            key_position,
            kv_head,
            head_dim,
            lane,
            simd_lane,
            simdgroup,
            threads_per_simdgroup,
            partial
        );
        if (lane == 0) {
            state[0] = max(state[0], score * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = state[0];
    float result = 0.0f;
    float denominator = 0.0f;
    for (uint key_position = 0; key_position < context; ++key_position) {
        float score = attention_dot_span(
            query + query_base,
            descriptors,
            key0,
            key1,
            key2,
            key3,
            span_count,
            key_position,
            kv_head,
            head_dim,
            lane,
            simd_lane,
            simdgroup,
            threads_per_simdgroup,
            partial
        );
        if (lane == 0) {
            partial[0] = exp(score * scale - maximum);
            denominator += partial[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float probability = partial[0];
        uint span = span_for_position(descriptors, span_count, key_position);
        uint local = key_position - descriptors[span * 3];
        uint capacity = descriptors[span * 3 + 2];
        if (lane < head_dim) {
            uint value_base = (kv_head * capacity + local) * head_dim;
            result += probability
                * span_value_value(value0, value1, value2, value3, span, value_base + lane);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        state[0] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < head_dim) {
        output[output_base + lane] = result / state[0];
    }
}

inline void attention_decode_spans_tiled(
    device const float *query,
    device float *output,
    device const uint *descriptors,
    device const half *key0,
    device const half *key1,
    device const half *key2,
    device const half *key3,
    device const half *value0,
    device const half *value1,
    device const half *value2,
    device const half *value3,
    constant DispatchArgs &args,
    threadgroup float *partial,
    threadgroup float *state,
    uint lane,
    uint head,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup
) {
    uint head_base = head * args.head_dim;
    uint group_size = args.n_head / args.n_head_kv;
    uint kv_head = head / group_size;
    attention_tiled_spans(
        query,
        output,
        descriptors,
        key0,
        key1,
        key2,
        key3,
        value0,
        value1,
        value2,
        value3,
        args.row,
        head_base,
        head_base,
        args.position + 1,
        kv_head,
        args.head_dim,
        partial,
        state,
        lane,
        simd_lane,
        simdgroup,
        threads_per_simdgroup
    );
}

inline void attention_prefill_spans_tiled(
    device const float *query,
    device float *output,
    device const uint *descriptors,
    device const half *key0,
    device const half *key1,
    device const half *key2,
    device const half *key3,
    device const half *value0,
    device const half *value1,
    device const half *value2,
    device const half *value3,
    constant DispatchArgs &args,
    threadgroup float *partial,
    threadgroup float *state,
    uint lane,
    uint group,
    uint simd_lane,
    uint simdgroup,
    uint threads_per_simdgroup
) {
    uint token = group / args.n_head;
    uint head = group % args.n_head;
    uint head_base = head * args.head_dim;
    uint elements_per_query = args.n_head * args.head_dim;
    uint query_base = token * elements_per_query + head_base;
    uint position = args.start_position + token;
    uint group_size = args.n_head / args.n_head_kv;
    uint kv_head = head / group_size;
    attention_tiled_spans(
        query,
        output,
        descriptors,
        key0,
        key1,
        key2,
        key3,
        value0,
        value1,
        value2,
        value3,
        args.row,
        query_base,
        query_base,
        position + 1,
        kv_head,
        args.head_dim,
        partial,
        state,
        lane,
        simd_lane,
        simdgroup,
        threads_per_simdgroup
    );
}

inline void attention_decode_spans_scalar(
    device const float *f0,
    device float *f1,
    device const uint *descriptors,
    device const half *key0,
    device const half *key1,
    device const half *key2,
    device const half *key3,
    device const half *value0,
    device const half *value1,
    device const half *value2,
    device const half *value3,
    constant DispatchArgs &args,
    uint gid
) {
    uint head_element = gid % (args.n_head * args.head_dim);
    uint head = head_element / args.head_dim;
    uint column = head_element % args.head_dim;
    uint kv_head = head / (args.n_head / args.n_head_kv);
    float scale = rsqrt(float(args.head_dim));
    float maximum = -INFINITY;
    for (uint key_position = 0; key_position <= args.position; ++key_position) {
        uint span = span_for_position(descriptors, args.row, key_position);
        uint local = key_position - descriptors[span * 3];
        uint capacity = descriptors[span * 3 + 2];
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = head_element - column + index;
            uint key_index = (kv_head * capacity + local) * args.head_dim + index;
            score += f0[query_index]
                * span_key_value(key0, key1, key2, key3, span, key_index);
        }
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    float result = 0.0f;
    for (uint key_position = 0; key_position <= args.position; ++key_position) {
        uint span = span_for_position(descriptors, args.row, key_position);
        uint local = key_position - descriptors[span * 3];
        uint capacity = descriptors[span * 3 + 2];
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = head_element - column + index;
            uint key_index = (kv_head * capacity + local) * args.head_dim + index;
            score += f0[query_index]
                * span_key_value(key0, key1, key2, key3, span, key_index);
        }
        float probability = exp(score * scale - maximum);
        denominator += probability;
        uint value_index = (kv_head * capacity + local) * args.head_dim + column;
        result += probability
            * span_value_value(value0, value1, value2, value3, span, value_index);
    }
    f1[head_element] = result / denominator;
}

inline void attention_prefill_spans_scalar(
    device const float *f0,
    device float *f1,
    device const uint *descriptors,
    device const half *key0,
    device const half *key1,
    device const half *key2,
    device const half *key3,
    device const half *value0,
    device const half *value1,
    device const half *value2,
    device const half *value3,
    constant DispatchArgs &args,
    uint gid
) {
    uint elements_per_query = args.n_head * args.head_dim;
    uint token = gid / elements_per_query;
    uint head_element = gid % elements_per_query;
    uint head = head_element / args.head_dim;
    uint column = head_element % args.head_dim;
    uint query_offset = token * elements_per_query + head_element;
    uint position = args.start_position + token;
    uint context = position + 1;
    uint kv_head = head / (args.n_head / args.n_head_kv);
    float scale = rsqrt(float(args.head_dim));
    float maximum = -INFINITY;
    for (uint key_position = 0; key_position < context; ++key_position) {
        uint span = span_for_position(descriptors, args.row, key_position);
        uint local = key_position - descriptors[span * 3];
        uint capacity = descriptors[span * 3 + 2];
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = query_offset - column + index;
            uint key_index = (kv_head * capacity + local) * args.head_dim + index;
            score += f0[query_index]
                * span_key_value(key0, key1, key2, key3, span, key_index);
        }
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    float result = 0.0f;
    for (uint key_position = 0; key_position < context; ++key_position) {
        uint span = span_for_position(descriptors, args.row, key_position);
        uint local = key_position - descriptors[span * 3];
        uint capacity = descriptors[span * 3 + 2];
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            uint query_index = query_offset - column + index;
            uint key_index = (kv_head * capacity + local) * args.head_dim + index;
            score += f0[query_index]
                * span_key_value(key0, key1, key2, key3, span, key_index);
        }
        float probability = exp(score * scale - maximum);
        denominator += probability;
        uint value_index = (kv_head * capacity + local) * args.head_dim + column;
        result += probability
            * span_value_value(value0, value1, value2, value3, span, value_index);
    }
    f1[query_offset] = result / denominator;
}

struct PrefixTile {
    uint absolute_start;
    uint valid_tokens;
    uint key_offset_elements;
    uint value_offset_elements;
    uint row_offset;
    uint row_count;
};

struct QueryGroup {
    uint row_offset;
    uint row_count;
    uint query_head;
    uint kv_head;
    uint tile_offset;
    uint tile_count;
};

inline float batch_score(
    device const float *queries,
    device const half *keys,
    uint query_offset,
    uint key_offset,
    constant DispatchArgs &args
) {
    float total = 0.0f;
    for (uint index = 0; index < args.head_dim; ++index) {
        total += queries[query_offset + index] * float(keys[key_offset + index]);
    }
    return total * rsqrt(float(args.head_dim));
}

inline void finish_batch_head(
    device float *output,
    uint output_offset,
    thread float *state,
    uint head_dim,
    float denominator
) {
    for (uint index = 0; index < head_dim; ++index) {
        output[output_offset + index] = state[index] / denominator;
    }
}

inline void batch_fixed_tile_body(
    device const float *queries,
    device float *output,
    device const half *keys,
    device const half *values,
    device const uint *tile_words,
    device const uint *positions,
    constant DispatchArgs &args,
    uint gid
) {
    uint total = args.batch_rows * args.n_head;
    if (gid >= total) {
        return;
    }
    uint row = gid / args.n_head;
    uint query_head = gid % args.n_head;
    uint kv_head = query_head / (args.n_head / args.n_head_kv);
    uint query_stride = args.n_head * args.head_dim;
    uint query_offset = row * query_stride + query_head * args.head_dim;
    uint kv_base = kv_head * args.gather_stride * args.head_dim;
    uint position = positions[row];
    float state[128] = {};
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint tile_index = 0; tile_index < args.tile_count; ++tile_index) {
        PrefixTile tile = {
            tile_words[tile_index * 6], tile_words[tile_index * 6 + 1],
            tile_words[tile_index * 6 + 2], tile_words[tile_index * 6 + 3],
            tile_words[tile_index * 6 + 4], tile_words[tile_index * 6 + 5]};
        for (uint offset = 0; offset < tile.valid_tokens; ++offset) {
            uint token = tile.absolute_start + offset;
            if (token > position) {
                continue;
            }
            uint key_offset = kv_base + tile.key_offset_elements + offset * args.head_dim;
            float score = batch_score(queries, keys, query_offset, key_offset, args);
            float next_max = max(maximum, score);
            float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next_max);
            float score_scale = exp(score - next_max);
            denominator = denominator * old_scale + score_scale;
            for (uint index = 0; index < args.head_dim; ++index) {
                uint value_offset = kv_base + tile.value_offset_elements + offset * args.head_dim + index;
                state[index] = state[index] * old_scale + score_scale * float(values[value_offset]);
            }
            maximum = next_max;
        }
    }
    finish_batch_head(output, query_offset, state, args.head_dim, denominator);
}

inline void update_shared_batch_row(
    device const float *queries,
    threadgroup const half *key_tile,
    threadgroup const half *value_tile,
    constant DispatchArgs &args,
    thread float *state,
    thread float &maximum,
    thread float &denominator,
    uint query_offset,
    uint position,
    PrefixTile tile
) {
    for (uint offset = 0; offset < tile.valid_tokens; ++offset) {
        uint token = tile.absolute_start + offset;
        if (token > position) {
            continue;
        }
        float score = 0.0f;
        for (uint index = 0; index < args.head_dim; ++index) {
            score += queries[query_offset + index] *
                float(key_tile[offset * args.head_dim + index]);
        }
        score *= rsqrt(float(args.head_dim));
        float next_max = max(maximum, score);
        float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next_max);
        float score_scale = exp(score - next_max);
        denominator = denominator * old_scale + score_scale;
        for (uint index = 0; index < args.head_dim; ++index) {
            state[index] = state[index] * old_scale +
                score_scale * float(value_tile[offset * args.head_dim + index]);
        }
        maximum = next_max;
    }
}

inline uint shared_tile_initialized_tokens(
    QueryGroup group,
    PrefixTile tile,
    device const uint *row_ids,
    device const uint *positions
) {
    uint initialized_tokens = 0;
    if (tile.row_count == 1) {
        uint row = row_ids[group.row_offset + tile.row_offset];
        uint position = positions[row];
        if (position >= tile.absolute_start) {
            initialized_tokens = min(
                tile.valid_tokens,
                position - tile.absolute_start + 1
            );
        }
    } else {
        for (uint row_index = 0; row_index < tile.row_count; ++row_index) {
            uint row = row_ids[group.row_offset + row_index];
            uint position = positions[row];
            if (position >= tile.absolute_start) {
                initialized_tokens = max(
                    initialized_tokens,
                    min(tile.valid_tokens, position - tile.absolute_start + 1)
                );
            }
        }
    }
    return initialized_tokens;
}

inline void load_shared_batch_tile(
    device const half *keys,
    device const half *values,
    threadgroup half *key_tile,
    threadgroup half *value_tile,
    constant DispatchArgs &args,
    PrefixTile tile,
    uint kv_base,
    uint initialized_tokens,
    uint thread_index,
    uint threads_per_group
) {
    uint elements = tile.valid_tokens * args.head_dim;
    for (uint index = thread_index; index < elements; index += threads_per_group) {
        if (index / args.head_dim < initialized_tokens) {
            key_tile[index] = keys[kv_base + tile.key_offset_elements + index];
            value_tile[index] = values[kv_base + tile.value_offset_elements + index];
        }
    }
}

inline void update_shared_batch_thread(
    device const float *queries,
    device const uint *row_ids,
    device const uint *positions,
    constant DispatchArgs &args,
    QueryGroup group,
    PrefixTile tile,
    threadgroup const half *key_tile,
    threadgroup const half *value_tile,
    thread float *state,
    thread float &maximum,
    thread float &denominator,
    uint query_stride,
    uint thread_index
) {
    if (thread_index >= tile.row_offset &&
        thread_index < tile.row_offset + tile.row_count &&
        thread_index < group.row_count) {
        uint row = row_ids[group.row_offset + thread_index];
        uint query_offset = row * query_stride + group.query_head * args.head_dim;
        uint position = positions[row];
        update_shared_batch_row(
            queries,
            key_tile,
            value_tile,
            args,
            state,
            maximum,
            denominator,
            query_offset,
            position,
            tile
        );
    }
}

inline void shared_batch_body(
    device const float *queries,
    device float *output,
    device const half *keys,
    device const half *values,
    device const uint *tile_words,
    device const uint *group_words,
    device const uint *row_ids,
    device const uint *positions,
    constant DispatchArgs &args,
    threadgroup half *shared,
    uint thread_index,
    uint threads_per_group,
    uint group_index
) {
    if (group_index >= args.group_count) {
        return;
    }
    QueryGroup group = {
        group_words[group_index * 6], group_words[group_index * 6 + 1],
        group_words[group_index * 6 + 2], group_words[group_index * 6 + 3],
        group_words[group_index * 6 + 4], group_words[group_index * 6 + 5]};
    threadgroup half *key_tile = shared;
    threadgroup half *value_tile = key_tile + args.tile_tokens * args.head_dim;
    uint query_stride = args.n_head * args.head_dim;
    uint kv_base = group.kv_head * args.gather_stride * args.head_dim;
    float state[128] = {};
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint tile_index = 0; tile_index < group.tile_count; ++tile_index) {
        PrefixTile tile = {
            tile_words[(group.tile_offset + tile_index) * 6],
            tile_words[(group.tile_offset + tile_index) * 6 + 1],
            tile_words[(group.tile_offset + tile_index) * 6 + 2],
            tile_words[(group.tile_offset + tile_index) * 6 + 3],
            tile_words[(group.tile_offset + tile_index) * 6 + 4],
            tile_words[(group.tile_offset + tile_index) * 6 + 5]};
        uint initialized_tokens = shared_tile_initialized_tokens(
            group, tile, row_ids, positions
        );
        load_shared_batch_tile(
            keys,
            values,
            key_tile,
            value_tile,
            args,
            tile,
            kv_base,
            initialized_tokens,
            thread_index,
            threads_per_group
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        update_shared_batch_thread(
            queries,
            row_ids,
            positions,
            args,
            group,
            tile,
            key_tile,
            value_tile,
            state,
            maximum,
            denominator,
            query_stride,
            thread_index
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index < group.row_count) {
        uint row = row_ids[group.row_offset + thread_index];
        uint query_offset = row * query_stride + group.query_head * args.head_dim;
        finish_batch_head(output, query_offset, state, args.head_dim, denominator);
    }
}

inline void op_embed_gather(
    device const uchar *bytes0,
    device float *f1,
    device const uint *u0,
    constant DispatchArgs &args,
    uint gid
) {
    uint index = u0[0] * args.columns + gid;
    f1[gid] = weight_value(bytes0, index, args.format);
}

inline void op_embed_gather_batch(
    device const uchar *bytes0,
    device float *f1,
    device const uint *u0,
    constant DispatchArgs &args,
    uint gid
) {
    uint token = gid / args.columns;
    uint column = gid % args.columns;
    f1[gid] = weight_value(bytes0, u0[token] * args.columns + column, args.format);
}

inline void op_copy_row(
    device const float *f0,
    device float *f1,
    constant DispatchArgs &args,
    uint gid
) {
    f1[gid] = f0[args.row * args.columns + gid];
}

inline void op_write_row(
    device const float *f0,
    device float *f1,
    constant DispatchArgs &args,
    uint gid
) {
    f1[args.row * args.columns + gid] = f0[gid];
}

inline bool argmax_candidate_wins(
    float candidate_value,
    uint candidate_index,
    float best_value,
    uint best_index
) {
    return !isnan(candidate_value) && (isnan(best_value) || candidate_value > best_value
        || (candidate_value == best_value && candidate_index < best_index));
}

inline void op_argmax(
    device const float *f0,
    device uint *u0,
    constant DispatchArgs &args,
    threadgroup float *partial_values,
    threadgroup uint *partial_indices,
    uint lane
) {
    uint best = 0;
    float best_value = f0[0];
    for (uint index = lane; index < args.columns;) {
        float value = f0[index];
        if (argmax_candidate_wins(value, index, best_value, best)) {
            best = index;
            best_value = value;
        }
        if (args.columns - index <= 256) {
            break;
        }
        index += 256;
    }
    partial_values[lane] = best_value;
    partial_indices[lane] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            float value = partial_values[lane + stride];
            uint index = partial_indices[lane + stride];
            if (argmax_candidate_wins(
                value, index, partial_values[lane], partial_indices[lane]
            )) {
                partial_values[lane] = value;
                partial_indices[lane] = index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        u0[0] = partial_indices[0];
    }
}

inline void op_increment(device uint *u0) {
    u0[0] += 1;
}

#define LEONE_KERNEL_ARGS     device const uchar *bytes0 [[buffer(0)]],     device const float *f0 [[buffer(1)]],     device float *f1 [[buffer(2)]],     device const float *f2 [[buffer(3)]],     device float *f3 [[buffer(4)]],     device const half *h0 [[buffer(5)]],     device half *h1 [[buffer(6)]],     device half *h2 [[buffer(7)]],     device uint *u0 [[buffer(8)]],     device const float *f4 [[buffer(9)]],     device const float *f5 [[buffer(10)]],     device float *f6 [[buffer(11)]],     constant DispatchArgs &args [[buffer(12)]],     uint gid [[thread_position_in_grid]]

#define LEONE_REDUCTION_KERNEL_ARGS     device const uchar *bytes0 [[buffer(0)]],     device const float *f0 [[buffer(1)]],     device float *f1 [[buffer(2)]],     device const float *f2 [[buffer(3)]],     device float *f3 [[buffer(4)]],     device const half *h0 [[buffer(5)]],     device half *h1 [[buffer(6)]],     device half *h2 [[buffer(7)]],     device uint *u0 [[buffer(8)]],     device const float *f4 [[buffer(9)]],     device const float *f5 [[buffer(10)]],     device float *f6 [[buffer(11)]],     constant DispatchArgs &args [[buffer(12)]],     uint3 gid [[thread_position_in_grid]],     uint3 group [[threadgroup_position_in_grid]]

#define LEONE_SPAN_KERNEL_ARGS     device const float *f0 [[buffer(1)]],     device float *f1 [[buffer(2)]],     device const uint *u0 [[buffer(3)]],     device const half *h0 [[buffer(5)]],     device const half *h1 [[buffer(6)]],     device const half *h2 [[buffer(7)]],     device const half *h3 [[buffer(8)]],     device const half *h4 [[buffer(9)]],     device const half *h5 [[buffer(10)]],     device const half *h6 [[buffer(11)]],     device const half *h7 [[buffer(12)]],     constant DispatchArgs &args [[buffer(13)]],     uint3 gid [[thread_position_in_grid]],     uint3 group [[threadgroup_position_in_grid]]

kernel void leone_gemv(
    LEONE_REDUCTION_KERNEL_ARGS
) {
    threadgroup float partial[256];
    op_gemv(bytes0, f0, f1, args, partial, gid.x % 256, group.x);
}

kernel void leone_gemv_residual(
    LEONE_REDUCTION_KERNEL_ARGS
) {
    threadgroup float partial[256];
    op_gemv_residual(bytes0, f0, f1, f2, args, partial, gid.x % 256, group.x);
}

kernel void leone_prefill_gemm(
    LEONE_REDUCTION_KERNEL_ARGS
) {
    if (args.tile_tokens == PREFILL_MATRIX_TILE_TOKENS) {
        threadgroup float weights_tile[
            PREFILL_MATRIX_TILE_ROWS * PREFILL_MATRIX_K_TILE
        ];
        threadgroup float activation_tile[
            PREFILL_MATRIX_TILE_TOKENS * PREFILL_MATRIX_K_TILE
        ];
        threadgroup float result_tile[
            PREFILL_MATRIX_TILE_TOKENS * PREFILL_MATRIX_TILE_ROWS
        ];
        op_prefill_gemm_simdgroup_matrix(
            bytes0,
            f0,
            f1,
            args,
            weights_tile,
            activation_tile,
            result_tile,
            gid.x % 256,
            ushort((gid.x % 256) / 32),
            group.x
        );
    } else if (args.tile_tokens == LEONE_PREFILL_TOKEN_TILE) {
        threadgroup float partial[LEONE_PREFILL_TOKEN_TILE * 256];
        op_prefill_gemm_token_tile(
            bytes0,
            f0,
            f1,
            args,
            partial,
            gid.x % 256,
            group.x
        );
    } else {
        threadgroup float partial[256];
        op_prefill_gemm(bytes0, f0, f1, args, partial, gid.x % 256, group.x);
    }
}

kernel void leone_rms_norm(LEONE_REDUCTION_KERNEL_ARGS) {
    threadgroup float partial[256];
    op_rms_norm(f0, f1, f5, args, partial, gid.x % 256, group.x);
}

kernel void leone_rms_norm_rope(LEONE_KERNEL_ARGS) {
    op_rms_norm_rope(f0, f1, f4, f5, args, gid);
}

kernel void leone_prefill_rms_norm_rope(LEONE_KERNEL_ARGS) {
    op_prefill_rms_norm_rope(f0, f1, f4, f5, args, gid);
}

kernel void leone_rms_norm_residual(LEONE_KERNEL_ARGS) {
    op_rms_norm_residual(f0, f1, f2, f5, args, gid);
}

kernel void leone_rms_norm_residual_store(LEONE_KERNEL_ARGS) {
    op_rms_norm_residual_store(f0, f1, f2, f3, f5, args, gid);
}

kernel void leone_rope(LEONE_KERNEL_ARGS) {
    op_rope(f1, f4, args, gid);
}

kernel void leone_swiglu(LEONE_KERNEL_ARGS) {
    op_swiglu(f0, f1, f2, gid);
}

kernel void leone_residual_add(LEONE_KERNEL_ARGS) {
    op_residual_add(f0, f1, f2, gid);
}

kernel void leone_kv_append(LEONE_KERNEL_ARGS) {
    op_kv_append(f0, f2, h1, h2, args, gid);
}

kernel void leone_kv_append_chunk(LEONE_KERNEL_ARGS) {
    op_kv_append_chunk(f0, f2, h1, h2, args, gid);
}

kernel void leone_attention_decode(
    LEONE_REDUCTION_KERNEL_ARGS,
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float partial[256];
    threadgroup float state[1];
    uint lane = gid.x % 256;
    if (args.pairing == 1) {
        attention_decode_tiled(
            f0,
            f1,
            h0,
            h1,
            args,
            partial,
            state,
            lane,
            group.x,
            simd_lane,
            simdgroup,
            threads_per_simdgroup
        );
    } else {
        attention_decode_scalar(f0, f1, h0, h1, args, gid.x);
    }
}

kernel void leone_attention_prefill(
    LEONE_REDUCTION_KERNEL_ARGS,
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float partial[256];
    threadgroup float state[1];
    uint lane = gid.x % 256;
    if (args.pairing == 1) {
        attention_prefill_tiled(
            f0,
            f1,
            h0,
            h1,
            args,
            partial,
            state,
            lane,
            group.x,
            simd_lane,
            simdgroup,
            threads_per_simdgroup
        );
    } else {
        attention_prefill(f0, f1, h0, h1, args, gid.x);
    }
}

kernel void leone_attention_decode_spans(
    LEONE_SPAN_KERNEL_ARGS,
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float partial[256];
    threadgroup float state[1];
    uint lane = gid.x % 256;
    if (args.pairing == 1) {
        attention_decode_spans_tiled(
            f0,
            f1,
            u0,
            h0,
            h2,
            h4,
            h6,
            h1,
            h3,
            h5,
            h7,
            args,
            partial,
            state,
            lane,
            group.x,
            simd_lane,
            simdgroup,
            threads_per_simdgroup
        );
    } else {
        attention_decode_spans_scalar(
            f0,
            f1,
            u0,
            h0,
            h2,
            h4,
            h6,
            h1,
            h3,
            h5,
            h7,
            args,
            gid.x
        );
    }
}

kernel void leone_attention_prefill_spans(
    LEONE_SPAN_KERNEL_ARGS,
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float partial[256];
    threadgroup float state[1];
    uint lane = gid.x % 256;
    if (args.pairing == 1) {
        attention_prefill_spans_tiled(
            f0,
            f1,
            u0,
            h0,
            h2,
            h4,
            h6,
            h1,
            h3,
            h5,
            h7,
            args,
            partial,
            state,
            lane,
            group.x,
            simd_lane,
            simdgroup,
            threads_per_simdgroup
        );
    } else {
        attention_prefill_spans_scalar(
            f0,
            f1,
            u0,
            h0,
            h2,
            h4,
            h6,
            h1,
            h3,
            h5,
            h7,
            args,
            gid.x
        );
    }
}

inline void op_kv_copy_span(
    device const half *source,
    device half *destination,
    constant DispatchArgs &args,
    uint gid
) {
    uint elements_per_token = args.n_head_kv * args.head_dim;
    uint token = gid / elements_per_token;
    uint head_element = gid % elements_per_token;
    uint head = head_element / args.head_dim;
    uint column = head_element % args.head_dim;
    uint source_index = (head * args.row + token) * args.head_dim + column;
    uint destination_index =
        (head * args.max_context + args.start_position + token) * args.head_dim + column;
    destination[destination_index] = source[source_index];
}

kernel void leone_kv_copy_span(LEONE_KERNEL_ARGS) {
    op_kv_copy_span(h0, h1, args, gid);
}

kernel void leone_attention_batch_fixed_tile(
    device const float *queries [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const half *keys [[buffer(5)]],
    device const half *values [[buffer(6)]],
    device const uint *tiles [[buffer(8)]],
    device const uint *positions [[buffer(11)]],
    constant DispatchArgs &args [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
    batch_fixed_tile_body(queries, output, keys, values, tiles, positions, args, gid);
}

kernel void leone_attention_batch_shared(
    device const float *queries [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const half *keys [[buffer(5)]],
    device const half *values [[buffer(6)]],
    device const uint *tiles [[buffer(8)]],
    device const uint *groups [[buffer(9)]],
    device const uint *row_ids [[buffer(10)]],
    device const uint *positions [[buffer(11)]],
    constant DispatchArgs &args [[buffer(12)]],
    threadgroup half *shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]) {
    shared_batch_body(queries, output, keys, values, tiles, groups, row_ids,
                      positions, args, shared, thread_index, threads_per_group,
                      group_index);
}

kernel void leone_attention_batch_fixed_shared(
    device const float *queries [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const half *keys [[buffer(5)]],
    device const half *values [[buffer(6)]],
    device const uint *tiles [[buffer(8)]],
    device const uint *groups [[buffer(9)]],
    device const uint *row_ids [[buffer(10)]],
    device const uint *positions [[buffer(11)]],
    constant DispatchArgs &args [[buffer(12)]],
    threadgroup half *shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]) {
    shared_batch_body(queries, output, keys, values, tiles, groups, row_ids,
                      positions, args, shared, thread_index, threads_per_group,
                      group_index);
}

kernel void leone_embed_gather(LEONE_KERNEL_ARGS) {
    op_embed_gather(bytes0, f1, u0, args, gid);
}

kernel void leone_embed_gather_batch(LEONE_KERNEL_ARGS) {
    op_embed_gather_batch(bytes0, f1, u0, args, gid);
}

kernel void leone_copy_row(LEONE_KERNEL_ARGS) {
    op_copy_row(f0, f1, args, gid);
}

kernel void leone_write_row(LEONE_KERNEL_ARGS) {
    op_write_row(f0, f1, args, gid);
}

kernel void leone_argmax(LEONE_KERNEL_ARGS) {
    threadgroup float partial_values[256];
    threadgroup uint partial_indices[256];
    op_argmax(f0, u0, args, partial_values, partial_indices, gid);
}

kernel void leone_increment(LEONE_KERNEL_ARGS) {
    op_increment(u0);
}
