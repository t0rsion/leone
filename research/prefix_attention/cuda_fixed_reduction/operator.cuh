#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace prefix_attention {

enum class Path : std::uint32_t {
    PerRow = 0,
    FixedTilePerRow = 1,
    SharedReadUnconstrained = 2,
    SharedReadFixedReduction = 3,
};

struct CaseSpec {
    std::string name;
    std::uint32_t tokens;
    std::uint32_t query_rows;
    std::uint32_t query_heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    std::uint32_t tile_tokens;
    std::uint32_t group_rows;
    std::uint64_t seed;
    bool shared_prefix;
};

struct PrefixTile {
    std::uint32_t absolute_start;
    std::uint32_t valid_tokens;
    std::uint32_t key_offset_elements;
    std::uint32_t value_offset_elements;
};

struct QueryGroup {
    std::uint32_t row_offset;
    std::uint32_t row_count;
    std::uint32_t query_head;
    std::uint32_t kv_head;
    std::uint32_t tile_offset;
    std::uint32_t tile_count;
};

struct HostProblem {
    CaseSpec spec;
    std::vector<float> queries;
    std::vector<__half> keys;
    std::vector<__half> values;
    std::vector<PrefixTile> tiles;
    std::vector<QueryGroup> schedule_a;
    std::vector<QueryGroup> schedule_b;
    std::vector<PrefixTile> schedule_tiles_a;
    std::vector<PrefixTile> schedule_tiles_b;
    std::vector<std::uint32_t> row_ids_a;
    std::vector<std::uint32_t> row_ids_b;
};

struct DeviceProblem {
    cudaStream_t stream = nullptr;
    float* queries = nullptr;
    __half* keys = nullptr;
    __half* values = nullptr;
    float* output = nullptr;
    PrefixTile* tiles = nullptr;
    QueryGroup* groups = nullptr;
    std::uint32_t* row_ids = nullptr;
};

struct RunResult {
    std::vector<float> output;
    std::vector<float> samples_ms;
    float elapsed_ms = 0.0F;
    std::uint64_t estimated_kv_elements_read = 0;
    std::uint64_t estimated_tile_loads = 0;
    std::uint64_t output_elements_written = 0;
    std::uint64_t per_block_shared_bytes = 0;
};

const char* path_name(Path path);
std::vector<CaseSpec> calibration_cases();
std::vector<CaseSpec> evaluation_cases();
HostProblem make_problem(const CaseSpec& spec);
void validate_problem(const HostProblem& problem);
std::string input_digest(const HostProblem& problem);
void allocate_device_problem(const HostProblem& host, DeviceProblem* device);
void free_device_problem(DeviceProblem* device);
RunResult run_path(const HostProblem& host, DeviceProblem* device, Path path,
                   bool schedule_b, std::uint32_t warmups,
                   std::uint32_t repetitions);
std::vector<double> fp64_oracle(const HostProblem& host);
std::string oracle_digest(const std::vector<double>& values);
double max_absolute_error(const std::vector<float>& actual,
                          const std::vector<double>& expected);
double max_relative_error(const std::vector<float>& actual,
                          const std::vector<double>& expected);
std::string output_digest(const std::vector<float>& values);
bool bitwise_equal(const std::vector<float>& left,
                   const std::vector<float>& right);

}  // namespace prefix_attention
