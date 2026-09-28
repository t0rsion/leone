#include <metal_stdlib>

using namespace metal;

struct PrefixTile {
    uint absolute_start;
    uint valid_tokens;
    uint key_offset_elements;
    uint value_offset_elements;
};

struct QueryGroup {
    uint row_offset;
    uint row_count;
    uint query_head;
    uint kv_head;
    uint tile_offset;
    uint tile_count;
};

struct Parameters {
    uint query_rows;
    uint query_heads;
    uint kv_heads;
    uint tokens;
    uint head_dim;
    uint tile_capacity_tokens;
    uint kv_stride;
    uint shared_prefix;
};

constant uint kMaximumHeadDim = 128;

inline uint query_kv_base(uint row, constant Parameters& parameters) {
    return parameters.shared_prefix == 1 ? 0 : row * parameters.kv_stride;
}

inline float score_dot(device const float* query, threadgroup const half* key,
                       uint head_dim, float scale) {
    float total = 0.0f;
    for (uint index = 0; index < head_dim; ++index) {
        total += query[index] * float(key[index]);
    }
    return total * scale;
}

inline float score_dot_global(device const float* query, device const half* key,
                              uint head_dim, float scale) {
    float total = 0.0f;
    for (uint index = 0; index < head_dim; ++index) {
        total += query[index] * float(key[index]);
    }
    return total * scale;
}

inline void update_softmax(float score, threadgroup const half* value,
                           uint head_dim, thread float& maximum,
                           thread float& normalizer, thread float* output) {
    const float next_max = max(maximum, score);
    const float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next_max);
    const float score_scale = exp(score - next_max);
    normalizer = normalizer * old_scale + score_scale;
    for (uint index = 0; index < head_dim; ++index) {
        output[index] = output[index] * old_scale +
                        score_scale * float(value[index]);
    }
    maximum = next_max;
}

inline void update_softmax_global(float score, device const half* value,
                                  uint head_dim, thread float& maximum,
                                  thread float& normalizer, thread float* output) {
    const float next_max = max(maximum, score);
    const float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next_max);
    const float score_scale = exp(score - next_max);
    normalizer = normalizer * old_scale + score_scale;
    for (uint index = 0; index < head_dim; ++index) {
        output[index] = output[index] * old_scale +
                        score_scale * float(value[index]);
    }
    maximum = next_max;
}

inline void finish_softmax(thread float* output, uint head_dim,
                           float normalizer) {
    for (uint index = 0; index < head_dim; ++index) {
        output[index] /= normalizer;
    }
}

kernel void per_row_kernel(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]], device float* output [[buffer(3)]],
    constant Parameters& parameters [[buffer(4)]],
    uint work [[thread_position_in_grid]]) {
    const uint total = parameters.query_rows * parameters.query_heads;
    if (work >= total) {
        return;
    }
    const uint row = work / parameters.query_heads;
    const uint query_head = work % parameters.query_heads;
    const uint kv_head = query_head * parameters.kv_heads / parameters.query_heads;
    const uint query_stride = parameters.query_heads * parameters.head_dim;
    const uint query_offset = row * query_stride + query_head * parameters.head_dim;
    const uint kv_base = query_kv_base(row, parameters) +
                         kv_head * parameters.tokens * parameters.head_dim;
    const float scale = rsqrt(float(parameters.head_dim));
    float state[kMaximumHeadDim] = {};
    float maximum = -INFINITY;
    float normalizer = 0.0f;
    for (uint token = 0; token < parameters.tokens; ++token) {
        const uint offset = kv_base + token * parameters.head_dim;
        const float score = score_dot_global(
            queries + query_offset, keys + offset, parameters.head_dim, scale);
        update_softmax_global(score, values + offset, parameters.head_dim,
                              maximum, normalizer, state);
    }
    finish_softmax(state, parameters.head_dim, normalizer);
    for (uint index = 0; index < parameters.head_dim; ++index) {
        output[query_offset + index] = state[index];
    }
}

kernel void fixed_tile_per_row_kernel(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const PrefixTile* tiles [[buffer(4)]],
    constant Parameters& parameters [[buffer(5)]], device float* output [[buffer(3)]],
    uint work [[thread_position_in_grid]]) {
    const uint total = parameters.query_rows * parameters.query_heads;
    if (work >= total) {
        return;
    }
    const uint row = work / parameters.query_heads;
    const uint query_head = work % parameters.query_heads;
    const uint kv_head = query_head * parameters.kv_heads / parameters.query_heads;
    const uint query_stride = parameters.query_heads * parameters.head_dim;
    const uint query_offset = row * query_stride + query_head * parameters.head_dim;
    const uint kv_base = query_kv_base(row, parameters) +
                         kv_head * parameters.tokens * parameters.head_dim;
    const float scale = rsqrt(float(parameters.head_dim));
    float state[kMaximumHeadDim] = {};
    float maximum = -INFINITY;
    float normalizer = 0.0f;
    const uint count = (parameters.tokens + parameters.tile_capacity_tokens - 1) /
                       parameters.tile_capacity_tokens;
    for (uint tile_index = 0; tile_index < count; ++tile_index) {
        const PrefixTile tile = tiles[tile_index];
        for (uint offset = 0; offset < tile.valid_tokens; ++offset) {
            const uint element_offset =
                kv_base + tile.key_offset_elements + offset * parameters.head_dim;
            const float score = score_dot_global(
                queries + query_offset, keys + element_offset,
                parameters.head_dim, scale);
            update_softmax_global(
                score, values + kv_base + tile.value_offset_elements +
                            offset * parameters.head_dim,
                parameters.head_dim, maximum, normalizer, state);
        }
    }
    finish_softmax(state, parameters.head_dim, normalizer);
    for (uint index = 0; index < parameters.head_dim; ++index) {
        output[query_offset + index] = state[index];
    }
}

inline void shared_read_body(
    device const float* queries, device const half* keys,
    device const half* values, device const PrefixTile* tiles,
    device const QueryGroup* groups, device const uint* row_ids,
    constant Parameters& parameters, device float* output, threadgroup half* shared,
    uint thread_index, uint threads_per_group, uint group_index) {
    const QueryGroup group = groups[group_index];
    threadgroup half* key_tile = shared;
    threadgroup half* value_tile =
        key_tile + parameters.tile_capacity_tokens * parameters.head_dim;
    const uint query_stride = parameters.query_heads * parameters.head_dim;
    const float scale = rsqrt(float(parameters.head_dim));
    float state[kMaximumHeadDim] = {};
    float maximum = -INFINITY;
    float normalizer = 0.0f;
    for (uint iteration = 0; iteration < group.tile_count; ++iteration) {
        const PrefixTile tile = tiles[group.tile_offset + iteration];
        const uint elements = tile.valid_tokens * parameters.head_dim;
        const uint source_base =
            query_kv_base(row_ids[group.row_offset], parameters) +
            group.kv_head * parameters.tokens * parameters.head_dim;
        for (uint index = thread_index; index < elements;
             index += threads_per_group) {
            key_tile[index] = keys[source_base + tile.key_offset_elements + index];
            value_tile[index] =
                values[source_base + tile.value_offset_elements + index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index < group.row_count) {
            const uint row = row_ids[group.row_offset + thread_index];
            const uint query_offset =
                row * query_stride + group.query_head * parameters.head_dim;
            for (uint offset = 0; offset < tile.valid_tokens; ++offset) {
                const float score = score_dot(
                    queries + query_offset, key_tile + offset * parameters.head_dim,
                    parameters.head_dim, scale);
                update_softmax(
                    score, value_tile + offset * parameters.head_dim,
                    parameters.head_dim, maximum, normalizer, state);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index < group.row_count) {
        const uint row = row_ids[group.row_offset + thread_index];
        const uint output_offset =
            row * query_stride + group.query_head * parameters.head_dim;
        finish_softmax(state, parameters.head_dim, normalizer);
        for (uint index = 0; index < parameters.head_dim; ++index) {
            output[output_offset + index] = state[index];
        }
    }
}

kernel void shared_read_unconstrained_kernel(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const PrefixTile* tiles [[buffer(4)]],
    device const QueryGroup* groups [[buffer(5)]],
    device const uint* row_ids [[buffer(6)]],
    constant Parameters& parameters [[buffer(7)]], device float* output [[buffer(3)]],
    threadgroup half* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]) {
    shared_read_body(queries, keys, values, tiles, groups, row_ids, parameters,
                     output, shared, thread_index, threads_per_group,
                     group_index);
}

kernel void shared_read_fixed_reduction_kernel(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const PrefixTile* tiles [[buffer(4)]],
    device const QueryGroup* groups [[buffer(5)]],
    device const uint* row_ids [[buffer(6)]],
    constant Parameters& parameters [[buffer(7)]], device float* output [[buffer(3)]],
    threadgroup half* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]) {
    shared_read_body(queries, keys, values, tiles, groups, row_ids, parameters,
                     output, shared, thread_index, threads_per_group,
                     group_index);
}
