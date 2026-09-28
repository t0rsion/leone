#include "operator.cuh"
#include "manifest_cases.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace prefix_attention {
namespace {

constexpr std::uint32_t kThreads = 128;
constexpr std::uint32_t kMaximumHeadDim = 128;

struct SplitMix64 {
    std::uint64_t state;

    std::uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t value = state;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    float unit_float() {
        const std::uint64_t bits = next() >> 40;
        return static_cast<float>(bits) / 16777215.0F * 2.0F - 1.0F;
    }
};

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::size_t checked_size(std::uint64_t value, std::string_view field) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(field) + " overflows host size");
    }
    return static_cast<std::size_t>(value);
}

std::uint32_t tile_count(const CaseSpec& spec) {
    return spec.tokens / spec.tile_tokens +
           (spec.tokens % spec.tile_tokens == 0 ? 0 : 1);
}

std::uint32_t effective_group_rows(const CaseSpec& spec) {
    return spec.shared_prefix ? spec.group_rows : 1;
}

std::uint32_t kv_query_stride(const CaseSpec& spec) {
    return spec.kv_heads * spec.tokens * spec.head_dim;
}

std::uint32_t query_row_stride(const CaseSpec& spec) {
    return spec.query_heads * spec.head_dim;
}

std::uint32_t output_elements(const CaseSpec& spec) {
    return spec.query_rows * query_row_stride(spec);
}

std::uint32_t group_count_for(const CaseSpec& spec) {
    const std::uint32_t rows = effective_group_rows(spec);
    const std::uint32_t row_groups =
        spec.query_rows / rows + (spec.query_rows % rows == 0 ? 0 : 1);
    return row_groups * spec.query_heads;
}

__device__ inline float score_dot(const float* query, const __half* key,
                                  std::uint32_t head_dim, float scale) {
    float total = 0.0F;
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        total += query[index] * __half2float(key[index]);
    }
    return total * scale;
}

__device__ inline void update_softmax(float score, const __half* value,
                                      std::uint32_t head_dim, float* maximum,
                                      float* normalizer, float* output) {
    const float next_max = fmaxf(*maximum, score);
    const float old_scale = isinf(*maximum) ? 0.0F : expf(*maximum - next_max);
    const float score_scale = expf(score - next_max);
    *normalizer = *normalizer * old_scale + score_scale;
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        output[index] = output[index] * old_scale +
                        score_scale * __half2float(value[index]);
    }
    *maximum = next_max;
}

__device__ inline void finish_softmax(float* output, std::uint32_t head_dim,
                                      float normalizer) {
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        output[index] /= normalizer;
    }
}

__host__ __device__ inline std::uint32_t query_kv_base(
    std::uint32_t row, std::uint32_t kv_stride, bool shared_prefix) {
    return shared_prefix ? 0 : row * kv_stride;
}

__global__ void per_row_kernel(const float* queries, const __half* keys,
                               const __half* values, float* output,
                               std::uint32_t query_rows,
                               std::uint32_t query_heads,
                               std::uint32_t kv_heads,
                               std::uint32_t tokens,
                               std::uint32_t head_dim,
                               std::uint32_t kv_stride,
                               bool shared_prefix) {
    const std::uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = query_rows * query_heads;
    if (work >= total) {
        return;
    }
    const std::uint32_t row = work / query_heads;
    const std::uint32_t query_head = work % query_heads;
    const std::uint32_t kv_head = query_head * kv_heads / query_heads;
    const std::uint32_t query_stride = query_heads * head_dim;
    const std::uint32_t kv_head_stride_value = tokens * head_dim;
    const float* query = queries + row * query_stride + query_head * head_dim;
    float* result = output + row * query_stride + query_head * head_dim;
    float state[kMaximumHeadDim] = {};
    float maximum = -CUDART_INF_F;
    float normalizer = 0.0F;
    const std::uint32_t base = query_kv_base(row, kv_stride, shared_prefix) +
                               kv_head * kv_head_stride_value;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const std::uint32_t offset = base + token * head_dim;
        const float score = score_dot(query, keys + offset, head_dim, scale);
        update_softmax(score, values + offset, head_dim, &maximum, &normalizer,
                       state);
    }
    finish_softmax(state, head_dim, normalizer);
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        result[index] = state[index];
    }
}

__global__ void fixed_tile_per_row_kernel(
    const float* queries, const __half* keys, const __half* values,
    const PrefixTile* tiles, float* output, std::uint32_t query_rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t tokens, std::uint32_t tile_count_value, std::uint32_t head_dim,
    std::uint32_t kv_stride, bool shared_prefix) {
    const std::uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = query_rows * query_heads;
    if (work >= total) {
        return;
    }
    const std::uint32_t row = work / query_heads;
    const std::uint32_t query_head = work % query_heads;
    const std::uint32_t kv_head = query_head * kv_heads / query_heads;
    const std::uint32_t query_stride = query_heads * head_dim;
    const float* query = queries + row * query_stride + query_head * head_dim;
    float* result = output + row * query_stride + query_head * head_dim;
    float state[kMaximumHeadDim] = {};
    float maximum = -CUDART_INF_F;
    float normalizer = 0.0F;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    for (std::uint32_t tile_index = 0; tile_index < tile_count_value;
         ++tile_index) {
        const PrefixTile tile = tiles[tile_index];
        const std::uint32_t head_offset = query_kv_base(
            row, kv_stride, shared_prefix) + kv_head * tokens * head_dim;
        const std::uint32_t key_offset = head_offset +
                                          tile.key_offset_elements;
        const std::uint32_t value_offset = head_offset +
                                            tile.value_offset_elements;
        for (std::uint32_t offset = 0; offset < tile.valid_tokens; ++offset) {
            const float score = score_dot(
                query, keys + key_offset + offset * head_dim, head_dim, scale);
            update_softmax(score, values + value_offset + offset * head_dim,
                           head_dim, &maximum, &normalizer, state);
        }
    }
    finish_softmax(state, head_dim, normalizer);
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        result[index] = state[index];
    }
}

template <bool FixedReduction>
__global__ void shared_read_kernel(
    const float* queries, const __half* keys, const __half* values,
    const PrefixTile* tiles, const QueryGroup* groups,
    const std::uint32_t* row_ids, float* output, std::uint32_t head_dim,
    std::uint32_t query_heads, std::uint32_t tokens,
    std::uint32_t tile_capacity_tokens, std::uint32_t kv_stride,
    bool shared_prefix) {
    const QueryGroup group = groups[blockIdx.x];
    extern __shared__ __half shared[];
    __half* key_tile = shared;
    __half* value_tile = shared + blockDim.x * 0;
    const std::uint32_t tile_capacity = tile_capacity_tokens * head_dim;
    value_tile = key_tile + tile_capacity;
    const std::uint32_t query_stride = query_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float state[kMaximumHeadDim] = {};
    if (threadIdx.x < group.row_count) {
        for (std::uint32_t index = 0; index < head_dim; ++index) {
            state[index] = 0.0F;
        }
    }
    float maximum = -CUDART_INF_F;
    float normalizer = 0.0F;
    for (std::uint32_t iteration = 0; iteration < group.tile_count;
         ++iteration) {
        const PrefixTile tile = tiles[group.tile_offset + iteration];
        const std::uint32_t elements = tile.valid_tokens * head_dim;
        for (std::uint32_t index = threadIdx.x; index < elements;
             index += blockDim.x) {
            const std::uint32_t tile_row = row_ids[group.row_offset];
            const std::uint32_t head_offset =
                query_kv_base(tile_row, kv_stride, shared_prefix) +
                group.kv_head * tokens * head_dim;
            const std::uint32_t source = head_offset + tile.key_offset_elements + index;
            key_tile[index] = keys[source];
            value_tile[index] = values[head_offset + tile.value_offset_elements + index];
        }
        __syncthreads();
        if (threadIdx.x < group.row_count) {
            const std::uint32_t row = row_ids[group.row_offset + threadIdx.x];
            const float* query = queries + row * query_stride +
                                 group.query_head * head_dim;
            (void)row;
            for (std::uint32_t offset = 0; offset < tile.valid_tokens;
                 ++offset) {
                const float score = score_dot(
                    query, key_tile + offset * head_dim, head_dim, scale);
                update_softmax(score, value_tile + offset * head_dim, head_dim,
                               &maximum, &normalizer, state);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < group.row_count) {
        const std::uint32_t row = row_ids[group.row_offset + threadIdx.x];
        float* result = output + row * query_stride +
                        group.query_head * head_dim;
        finish_softmax(state, head_dim, normalizer);
        for (std::uint32_t index = 0; index < head_dim; ++index) {
            result[index] = state[index];
        }
    }
    (void)FixedReduction;
}

}  // namespace

const char* path_name(Path path) {
    switch (path) {
        case Path::PerRow:
            return "per_row";
        case Path::FixedTilePerRow:
            return "fixed_tile_per_row";
        case Path::SharedReadUnconstrained:
            return "shared_read_unconstrained";
        case Path::SharedReadFixedReduction:
            return "shared_read_fixed_reduction";
    }
    return "unknown";
}

std::vector<CaseSpec> calibration_cases() {
    return manifest_calibration_cases();
}

std::vector<CaseSpec> evaluation_cases() {
    return manifest_evaluation_cases();
}

void validate_nonzero_dimensions(const CaseSpec& spec) {
    if (spec.tokens == 0 || spec.query_rows == 0 || spec.query_heads == 0 ||
        spec.kv_heads == 0 || spec.head_dim == 0 || spec.tile_tokens == 0 ||
        spec.group_rows == 0) {
        throw std::runtime_error("case has a zero dimension");
    }
}

void validate_head_mapping(const CaseSpec& spec) {
    if (spec.query_heads % spec.kv_heads != 0) {
        throw std::runtime_error("query heads are not divisible by KV heads");
    }
    if (spec.head_dim > kMaximumHeadDim) {
        throw std::runtime_error("head dimension exceeds kernel bound");
    }
}

void validate_offset_capacity(const CaseSpec& spec) {
    const std::uint64_t query_elements =
        static_cast<std::uint64_t>(spec.query_rows) * spec.query_heads *
        spec.head_dim;
    const std::uint64_t kv_elements =
        static_cast<std::uint64_t>(spec.shared_prefix ? 1 : spec.query_rows) *
        spec.kv_heads * spec.tokens * spec.head_dim;
    if (query_elements > std::numeric_limits<std::uint32_t>::max() ||
        kv_elements > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("case dimensions exceed 32-bit offsets");
    }
}

void validate_spec(const CaseSpec& spec) {
    validate_nonzero_dimensions(spec);
    validate_head_mapping(spec);
    validate_offset_capacity(spec);
}

PrefixTile tile_for(const CaseSpec& spec, std::uint32_t index) {
    const std::uint32_t start = index * spec.tile_tokens;
    const std::uint32_t valid = std::min(spec.tile_tokens, spec.tokens - start);
    return {start, valid, start * spec.head_dim, start * spec.head_dim};
}

std::vector<std::uint32_t> scheduled_rows(const CaseSpec& spec, bool variant_b) {
    std::vector<std::uint32_t> rows(spec.query_rows);
    std::iota(rows.begin(), rows.end(), 0);
    if (!variant_b) {
        return rows;
    }
    std::vector<std::uint32_t> reordered;
    reordered.reserve(rows.size());
    for (std::uint32_t row = 0; row < spec.query_rows; row += 2) {
        reordered.push_back(row);
    }
    for (std::uint32_t row = 1; row < spec.query_rows; row += 2) {
        reordered.push_back(row);
    }
    return reordered;
}

void append_group_tiles(const std::vector<PrefixTile>& base, bool variant_b,
                        std::uint32_t group_index,
                        std::vector<PrefixTile>* schedule_tiles) {
    const std::uint32_t count = static_cast<std::uint32_t>(base.size());
    if (!variant_b && group_index % 2 == 1) {
        schedule_tiles->insert(schedule_tiles->end(), base.rbegin(), base.rend());
    } else if (variant_b && group_index % 2 == 1) {
        const std::uint32_t shift = group_index % count;
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            schedule_tiles->push_back(base[(shift + offset) % count]);
        }
    } else {
        schedule_tiles->insert(schedule_tiles->end(), base.begin(), base.end());
    }
}

void append_schedule(const CaseSpec& spec, const std::vector<PrefixTile>& base,
                     bool variant_b, std::vector<QueryGroup>* groups,
                     std::vector<std::uint32_t>* row_ids,
                     std::vector<PrefixTile>* schedule_tiles) {
    const std::uint32_t rows_per_group = effective_group_rows(spec);
    const std::vector<std::uint32_t> rows = scheduled_rows(spec, variant_b);
    const std::uint32_t count = static_cast<std::uint32_t>(base.size());
    for (std::uint32_t query_head = 0; query_head < spec.query_heads;
         ++query_head) {
        for (std::uint32_t start = 0; start < spec.query_rows;
             start += rows_per_group) {
            const std::uint32_t count_rows =
                std::min(rows_per_group, spec.query_rows - start);
            QueryGroup group{
                static_cast<std::uint32_t>(row_ids->size()),
                count_rows,
                query_head,
                query_head * spec.kv_heads / spec.query_heads,
                static_cast<std::uint32_t>(schedule_tiles->size()),
                count,
            };
            groups->push_back(group);
            for (std::uint32_t offset = 0; offset < count_rows; ++offset) {
                row_ids->push_back(rows[start + offset]);
            }
            const std::uint32_t group_index =
                static_cast<std::uint32_t>(groups->size() - 1);
            append_group_tiles(base, variant_b, group_index, schedule_tiles);
        }
    }
}

HostProblem make_problem(const CaseSpec& spec) {
    validate_spec(spec);
    HostProblem problem;
    problem.spec = spec;
    SplitMix64 generator{spec.seed};
    problem.queries.resize(output_elements(spec));
    for (float& value : problem.queries) {
        value = generator.unit_float();
    }
    const std::uint32_t key_rows = spec.shared_prefix ? 1 : spec.query_rows;
    const std::size_t kv_values = static_cast<std::size_t>(key_rows) *
                                  spec.kv_heads * spec.tokens * spec.head_dim;
    problem.keys.resize(kv_values);
    problem.values.resize(kv_values);
    for (std::size_t index = 0; index < kv_values; ++index) {
        problem.keys[index] = __float2half(generator.unit_float());
        problem.values[index] = __float2half(generator.unit_float());
    }
    const std::uint32_t count = tile_count(spec);
    problem.tiles.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        problem.tiles.push_back(tile_for(spec, index));
    }
    append_schedule(spec, problem.tiles, false, &problem.schedule_a,
                    &problem.row_ids_a, &problem.schedule_tiles_a);
    append_schedule(spec, problem.tiles, true, &problem.schedule_b,
                    &problem.row_ids_b, &problem.schedule_tiles_b);
    validate_problem(problem);
    return problem;
}

void validate_tile_value(const CaseSpec& spec, const PrefixTile& tile,
                         bool final_tile) {
    if (tile.valid_tokens == 0 || tile.valid_tokens > spec.tile_tokens ||
        tile.absolute_start >= spec.tokens ||
        tile.valid_tokens > spec.tokens - tile.absolute_start ||
        tile.key_offset_elements !=
            static_cast<std::uint64_t>(tile.absolute_start) * spec.head_dim ||
        tile.value_offset_elements !=
            static_cast<std::uint64_t>(tile.absolute_start) * spec.head_dim) {
        throw std::runtime_error("tile descriptor is invalid");
    }
    if (!final_tile && tile.valid_tokens != spec.tile_tokens) {
        throw std::runtime_error("nonfinal tile is partial");
    }
}

void validate_tiles(const CaseSpec& spec,
                    const std::vector<PrefixTile>& tiles) {
    const std::uint32_t expected = tile_count(spec);
    if (tiles.size() != expected) {
        throw std::runtime_error("tile table has the wrong length");
    }
    std::uint32_t next = 0;
    for (std::uint32_t index = 0; index < expected; ++index) {
        const PrefixTile tile = tiles[index];
        if (tile.absolute_start != next) {
            throw std::runtime_error("tile table is not contiguous");
        }
        validate_tile_value(spec, tile, index + 1 == expected);
        next += tile.valid_tokens;
    }
    if (next != spec.tokens) {
        throw std::runtime_error("tile table does not cover tokens");
    }
}

std::size_t schedule_row_count(const std::vector<QueryGroup>& groups) {
    std::size_t count = 0;
    for (const QueryGroup& group : groups) {
        count += group.row_count;
    }
    return count;
}

void validate_group_descriptor(const CaseSpec& spec, const QueryGroup& group,
                               std::uint32_t expected_tile_count,
                               std::uint32_t rows_per_group,
                               std::size_t row_count,
                               std::size_t tile_count_value) {
    if (group.row_count == 0 || group.row_count > rows_per_group ||
        group.query_head >= spec.query_heads ||
        group.tile_count != expected_tile_count ||
        group.tile_offset > tile_count_value ||
        group.tile_count > tile_count_value - group.tile_offset ||
        group.row_offset > row_count ||
        group.row_count > row_count - group.row_offset) {
        throw std::runtime_error("query schedule descriptor is invalid");
    }
    const std::uint32_t expected_kv_head =
        group.query_head * spec.kv_heads / spec.query_heads;
    if (group.kv_head != expected_kv_head) {
        throw std::runtime_error("query schedule head mapping is invalid");
    }
}

void mark_schedule_rows(const CaseSpec& spec, const QueryGroup& group,
                        const std::vector<std::uint32_t>& row_ids,
                        std::vector<std::uint32_t>* seen) {
    for (std::uint32_t offset = 0; offset < group.row_count; ++offset) {
        const std::uint32_t row = row_ids[group.row_offset + offset];
        if (row >= spec.query_rows ||
            ++(*seen)[row * spec.query_heads + group.query_head] != 1) {
            throw std::runtime_error("query schedule repeats a row");
        }
    }
}

void validate_seen_rows(const std::vector<std::uint32_t>& seen) {
    if (std::any_of(seen.begin(), seen.end(),
                    [](std::uint32_t count) { return count != 1; })) {
        throw std::runtime_error("query schedule omits a row");
    }
}

void validate_schedule(const CaseSpec& spec,
                       const std::vector<QueryGroup>& groups,
                       const std::vector<std::uint32_t>& row_ids,
                       const std::vector<PrefixTile>& schedule_tiles) {
    const std::uint32_t expected_tile_count = tile_count(spec);
    const std::uint32_t rows_per_group = effective_group_rows(spec);
    const std::uint32_t expected_groups = group_count_for(spec);
    const std::size_t expected_row_ids = schedule_row_count(groups);
    if (groups.size() != expected_groups || row_ids.size() != expected_row_ids) {
        throw std::runtime_error("query schedule has the wrong length");
    }
    std::vector<std::uint32_t> seen(spec.query_rows * spec.query_heads, 0);
    for (const QueryGroup& group : groups) {
        validate_group_descriptor(spec, group, expected_tile_count,
                                  rows_per_group, row_ids.size(),
                                  schedule_tiles.size());
        mark_schedule_rows(spec, group, row_ids, &seen);
    }
    validate_seen_rows(seen);
}

void validate_schedule_tiles(
    const CaseSpec& spec, const std::vector<QueryGroup>& groups,
    const std::vector<PrefixTile>& schedule_tiles) {
    const std::uint32_t expected = tile_count(spec);
    if (schedule_tiles.size() != groups.size() * expected) {
        throw std::runtime_error("scheduled tile table has the wrong length");
    }
    for (const QueryGroup& group : groups) {
        std::vector<PrefixTile> sorted_tiles;
        sorted_tiles.reserve(group.tile_count);
        for (std::uint32_t offset = 0; offset < group.tile_count; ++offset) {
            const PrefixTile tile = schedule_tiles[group.tile_offset + offset];
            sorted_tiles.push_back(tile);
        }
        std::sort(sorted_tiles.begin(), sorted_tiles.end(),
                  [](const PrefixTile& left, const PrefixTile& right) {
                      return left.absolute_start < right.absolute_start;
                  });
        for (std::uint32_t index = 0; index < expected; ++index) {
            if (sorted_tiles[index].absolute_start != index * spec.tile_tokens) {
                throw std::runtime_error("scheduled tiles omit an absolute tile");
            }
            validate_tile_value(spec, sorted_tiles[index], index + 1 == expected);
        }
    }
}

void validate_problem(const HostProblem& problem) {
    validate_spec(problem.spec);
    const std::size_t expected_query = output_elements(problem.spec);
    const std::size_t key_rows = problem.spec.shared_prefix
                                     ? 1
                                     : problem.spec.query_rows;
    const std::size_t expected_kv = key_rows * problem.spec.kv_heads *
                                    problem.spec.tokens * problem.spec.head_dim;
    if (problem.queries.size() != expected_query ||
        problem.keys.size() != expected_kv ||
        problem.values.size() != expected_kv) {
        throw std::runtime_error("host buffers do not match case dimensions");
    }
    validate_tiles(problem.spec, problem.tiles);
    validate_schedule(problem.spec, problem.schedule_a, problem.row_ids_a,
                      problem.schedule_tiles_a);
    validate_schedule(problem.spec, problem.schedule_b, problem.row_ids_b,
                      problem.schedule_tiles_b);
    validate_schedule_tiles(problem.spec, problem.schedule_a,
                           problem.schedule_tiles_a);
    validate_schedule_tiles(problem.spec, problem.schedule_b,
                           problem.schedule_tiles_b);
}

void allocate_device_problem(const HostProblem& host, DeviceProblem* device) {
    const CaseSpec& spec = host.spec;
    const std::size_t query_bytes =
        checked_size(static_cast<std::uint64_t>(host.queries.size()) *
                         sizeof(float),
                     "query bytes");
    const std::size_t kv_bytes =
        checked_size(static_cast<std::uint64_t>(host.keys.size()) *
                         sizeof(__half),
                     "KV bytes");
    const std::size_t output_bytes = checked_size(
        static_cast<std::uint64_t>(output_elements(spec)) * sizeof(float),
        "output bytes");
    const std::size_t tile_capacity =
        std::max(host.tiles.size(),
                 std::max(host.schedule_tiles_a.size(),
                          host.schedule_tiles_b.size()));
    const std::size_t group_capacity =
        std::max(host.schedule_a.size(), host.schedule_b.size());
    const std::size_t row_capacity =
        std::max(host.row_ids_a.size(), host.row_ids_b.size());
    check_cuda(cudaStreamCreate(&device->stream), "create stream");
    try {
        check_cuda(cudaMalloc(&device->queries, query_bytes),
                   "allocate queries");
        check_cuda(cudaMalloc(&device->keys, kv_bytes), "allocate keys");
        check_cuda(cudaMalloc(&device->values, kv_bytes), "allocate values");
        check_cuda(cudaMalloc(&device->output, output_bytes),
                   "allocate output");
        check_cuda(cudaMalloc(&device->tiles,
                              tile_capacity * sizeof(PrefixTile)),
                   "allocate tile table");
        check_cuda(cudaMalloc(&device->groups,
                              group_capacity * sizeof(QueryGroup)),
                   "allocate group table");
        check_cuda(cudaMalloc(&device->row_ids,
                              row_capacity * sizeof(std::uint32_t)),
                   "allocate row table");
        check_cuda(cudaMemcpyAsync(device->queries, host.queries.data(),
                                   query_bytes, cudaMemcpyHostToDevice,
                                   device->stream),
                   "upload queries");
        check_cuda(cudaMemcpyAsync(device->keys, host.keys.data(), kv_bytes,
                                   cudaMemcpyHostToDevice, device->stream),
                   "upload keys");
        check_cuda(cudaMemcpyAsync(device->values, host.values.data(),
                                   kv_bytes, cudaMemcpyHostToDevice,
                                   device->stream),
                   "upload values");
        check_cuda(cudaStreamSynchronize(device->stream), "finish upload");
    } catch (...) {
        free_device_problem(device);
        throw;
    }
}

void free_device_problem(DeviceProblem* device) {
    if (device->queries != nullptr) {
        cudaFree(device->queries);
    }
    if (device->keys != nullptr) {
        cudaFree(device->keys);
    }
    if (device->values != nullptr) {
        cudaFree(device->values);
    }
    if (device->output != nullptr) {
        cudaFree(device->output);
    }
    if (device->tiles != nullptr) {
        cudaFree(device->tiles);
    }
    if (device->groups != nullptr) {
        cudaFree(device->groups);
    }
    if (device->row_ids != nullptr) {
        cudaFree(device->row_ids);
    }
    if (device->stream != nullptr) {
        cudaStreamDestroy(device->stream);
    }
    *device = DeviceProblem{};
}

void upload_schedule(const HostProblem& host, DeviceProblem* device,
                     Path path, bool schedule_b) {
    const auto& groups = schedule_b ? host.schedule_b : host.schedule_a;
    const auto& row_ids = schedule_b ? host.row_ids_b : host.row_ids_a;
    std::vector<QueryGroup> run_groups = groups;
    std::vector<PrefixTile> run_tiles;
    if (path == Path::SharedReadUnconstrained) {
        run_tiles = schedule_b ? host.schedule_tiles_b : host.schedule_tiles_a;
    } else {
        run_tiles = host.tiles;
        for (QueryGroup& group : run_groups) {
            group.tile_offset = 0;
        }
    }
    check_cuda(cudaMemcpyAsync(device->tiles, run_tiles.data(),
                               run_tiles.size() * sizeof(PrefixTile),
                               cudaMemcpyHostToDevice, device->stream),
               "upload tile table");
    if (path == Path::PerRow || path == Path::FixedTilePerRow) {
        return;
    }
    check_cuda(cudaMemcpyAsync(device->groups, run_groups.data(),
                               run_groups.size() * sizeof(QueryGroup),
                               cudaMemcpyHostToDevice, device->stream),
               "upload group table");
    check_cuda(cudaMemcpyAsync(device->row_ids, row_ids.data(),
                               row_ids.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, device->stream),
               "upload row table");
}

void launch_path(const HostProblem& host, DeviceProblem* device, Path path,
                 std::uint32_t group_count, cudaEvent_t start,
                 cudaEvent_t end) {
    const CaseSpec& spec = host.spec;
    const std::uint32_t total = spec.query_rows * spec.query_heads;
    const std::uint32_t blocks = (total + kThreads - 1) / kThreads;
    const std::uint32_t count = tile_count(spec);
    const std::size_t shared_bytes =
        2ULL * spec.tile_tokens * spec.head_dim * sizeof(__half);
    check_cuda(cudaEventRecord(start, device->stream), "record start event");
    switch (path) {
        case Path::PerRow:
            per_row_kernel<<<blocks, kThreads, 0, device->stream>>>(
                device->queries, device->keys, device->values, device->output,
                spec.query_rows, spec.query_heads, spec.kv_heads, spec.tokens,
                spec.head_dim, kv_query_stride(spec), spec.shared_prefix);
            break;
        case Path::FixedTilePerRow:
            fixed_tile_per_row_kernel<<<blocks, kThreads, 0, device->stream>>>(
                device->queries, device->keys, device->values, device->tiles,
                device->output, spec.query_rows, spec.query_heads,
                spec.kv_heads, spec.tokens, count, spec.head_dim,
                kv_query_stride(spec), spec.shared_prefix);
            break;
        case Path::SharedReadUnconstrained:
            shared_read_kernel<false><<<group_count, kThreads, shared_bytes,
                                        device->stream>>>(
                device->queries, device->keys, device->values, device->tiles,
                device->groups, device->row_ids, device->output,
                spec.head_dim, spec.query_heads, spec.tokens, spec.tile_tokens,
                kv_query_stride(spec), spec.shared_prefix);
            break;
        case Path::SharedReadFixedReduction:
            shared_read_kernel<true><<<group_count, kThreads, shared_bytes,
                                       device->stream>>>(
                device->queries, device->keys, device->values, device->tiles,
                device->groups, device->row_ids, device->output,
                spec.head_dim, spec.query_heads, spec.tokens, spec.tile_tokens,
                kv_query_stride(spec), spec.shared_prefix);
            break;
    }
    check_cuda(cudaGetLastError(), "launch attention path");
    check_cuda(cudaEventRecord(end, device->stream), "record end event");
}

RunResult run_path(const HostProblem& host, DeviceProblem* device, Path path,
                   bool schedule_b, std::uint32_t warmups,
                   std::uint32_t repetitions) {
    if (repetitions == 0) {
        throw std::runtime_error("repetitions must be positive");
    }
    upload_schedule(host, device, path, schedule_b);
    check_cuda(cudaStreamSynchronize(device->stream), "finish schedule upload");
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&end), "create end event");
    const std::uint32_t group_count =
        static_cast<std::uint32_t>(schedule_b ? host.schedule_b.size()
                                              : host.schedule_a.size());
    for (std::uint32_t index = 0; index < warmups; ++index) {
        launch_path(host, device, path, group_count, start, end);
        check_cuda(cudaEventSynchronize(end), "finish warmup");
    }
    RunResult result;
    result.samples_ms.reserve(repetitions);
    for (std::uint32_t index = 0; index < repetitions; ++index) {
        launch_path(host, device, path, group_count, start, end);
        check_cuda(cudaEventSynchronize(end), "finish measured run");
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, start, end),
                   "read CUDA event");
        result.samples_ms.push_back(milliseconds);
    }
    result.output.resize(output_elements(host.spec));
    check_cuda(cudaMemcpyAsync(result.output.data(), device->output,
                               result.output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, device->stream),
               "download output");
    check_cuda(cudaStreamSynchronize(device->stream), "finish output copy");
    std::vector<float> sorted_samples = result.samples_ms;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    result.elapsed_ms = sorted_samples[sorted_samples.size() / 2];
    const bool shared = path == Path::SharedReadUnconstrained ||
                        path == Path::SharedReadFixedReduction;
    const std::uint32_t group_rows = effective_group_rows(host.spec);
    const std::uint32_t groups =
        shared ? group_count : host.spec.query_rows * host.spec.query_heads;
    result.estimated_tile_loads =
        static_cast<std::uint64_t>(groups) * tile_count(host.spec);
    result.estimated_kv_elements_read =
        static_cast<std::uint64_t>(groups) * host.spec.tokens *
        host.spec.head_dim * 2;
    result.output_elements_written = output_elements(host.spec);
    result.per_block_shared_bytes = shared
                                        ? 2ULL * host.spec.tile_tokens *
                                              host.spec.head_dim * sizeof(__half)
                                        : 0;
    (void)group_rows;
    check_cuda(cudaEventDestroy(start), "destroy start event");
    check_cuda(cudaEventDestroy(end), "destroy end event");
    return result;
}

std::vector<double> fp64_oracle(const HostProblem& host) {
    const CaseSpec& spec = host.spec;
    const std::uint32_t query_stride = query_row_stride(spec);
    const std::uint32_t kv_stride = kv_query_stride(spec);
    std::vector<double> output(output_elements(spec), 0.0);
    std::vector<double> scores(spec.tokens);
    for (std::uint32_t row = 0; row < spec.query_rows; ++row) {
        for (std::uint32_t query_head = 0; query_head < spec.query_heads;
             ++query_head) {
            const std::uint32_t kv_head =
                query_head * spec.kv_heads / spec.query_heads;
            const std::uint32_t query_offset =
                row * query_stride + query_head * spec.head_dim;
            const std::uint32_t kv_base =
                query_kv_base(row, kv_stride, spec.shared_prefix) +
                kv_head * spec.tokens * spec.head_dim;
            const double scale = 1.0 / std::sqrt(spec.head_dim);
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::uint32_t token = 0; token < spec.tokens; ++token) {
                double score = 0.0;
                const std::uint32_t key_offset =
                    kv_base + token * spec.head_dim;
                for (std::uint32_t index = 0; index < spec.head_dim; ++index) {
                    score += static_cast<double>(host.queries[query_offset + index]) *
                             static_cast<double>(__half2float(
                                 host.keys[key_offset + index]));
                }
                scores[token] = score * scale;
                maximum = std::max(maximum, scores[token]);
            }
            double normalizer = 0.0;
            const std::uint32_t output_offset = query_offset;
            for (std::uint32_t token = 0; token < spec.tokens; ++token) {
                const double weight = std::exp(scores[token] - maximum);
                normalizer += weight;
                const std::uint32_t value_offset =
                    kv_base + token * spec.head_dim;
                for (std::uint32_t index = 0; index < spec.head_dim; ++index) {
                    output[output_offset + index] +=
                        weight * static_cast<double>(__half2float(
                            host.values[value_offset + index]));
                }
            }
            for (std::uint32_t index = 0; index < spec.head_dim; ++index) {
                output[output_offset + index] /= normalizer;
            }
        }
    }
    return output;
}

double max_absolute_error(const std::vector<float>& actual,
                          const std::vector<double>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("output and oracle lengths differ");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(
            maximum, std::abs(static_cast<double>(actual[index]) -
                              expected[index]));
    }
    return maximum;
}

double max_relative_error(const std::vector<float>& actual,
                          const std::vector<double>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("output and oracle lengths differ");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference =
            std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double scale = std::max(std::abs(expected[index]), 1.0e-12);
        maximum = std::max(maximum, difference / scale);
    }
    return maximum;
}

std::string output_digest(const std::vector<float>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (float value : values) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        for (std::uint32_t shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(bits >> shift);
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

std::string format_digest(std::uint64_t hash);

std::string oracle_digest(const std::vector<double>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (double value : values) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        for (std::uint32_t shift = 0; shift < 64; shift += 8) {
            hash ^= static_cast<std::uint8_t>(bits >> shift);
            hash *= 1099511628211ULL;
        }
    }
    return format_digest(hash);
}

bool bitwise_equal(const std::vector<float>& left,
                   const std::vector<float>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
               0;
}

void update_digest(std::uint64_t* hash, const void* data, std::size_t bytes) {
    const auto* input = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < bytes; ++index) {
        *hash ^= input[index];
        *hash *= 1099511628211ULL;
    }
}

void update_digest_u32(std::uint64_t* hash, std::uint32_t value) {
    const std::uint8_t bytes[] = {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8),
        static_cast<std::uint8_t>(value >> 16),
        static_cast<std::uint8_t>(value >> 24),
    };
    update_digest(hash, bytes, sizeof(bytes));
}

void update_digest_u16(std::uint64_t* hash, std::uint16_t value) {
    const std::uint8_t bytes[] = {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8),
    };
    update_digest(hash, bytes, sizeof(bytes));
}

std::string format_digest(std::uint64_t hash) {
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

std::string input_digest(const HostProblem& problem) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (float value : problem.queries) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        update_digest_u32(&hash, bits);
    }
    for (const __half value : problem.keys) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        update_digest_u16(&hash, bits);
    }
    for (const __half value : problem.values) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        update_digest_u16(&hash, bits);
    }
    return format_digest(hash);
}

}  // namespace prefix_attention
