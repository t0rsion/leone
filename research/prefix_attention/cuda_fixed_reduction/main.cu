#include "operator.cuh"
#include "manifest_cases.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string phase;
    std::string output;
    std::uint32_t warmups = 3;
    std::uint32_t repetitions = 9;
    bool validate_only = false;
};

using PathList = std::array<prefix_attention::Path, 4>;

PathList paths() {
    return {prefix_attention::Path::PerRow,
            prefix_attention::Path::FixedTilePerRow,
            prefix_attention::Path::SharedReadUnconstrained,
            prefix_attention::Path::SharedReadFixedReduction};
}

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::uint32_t parse_count(const char* text, std::string_view field) {
    std::size_t consumed = 0;
    const std::string value(text);
    const unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || parsed > UINT32_MAX) {
        throw std::runtime_error(std::string(field) + " is not a uint32");
    }
    return static_cast<std::uint32_t>(parsed);
}

const char* require_option_value(int* index, int argc, char** argv,
                                 std::string_view field) {
    if (*index + 1 >= argc) {
        throw std::runtime_error(std::string(field) + " needs a value");
    }
    ++*index;
    return argv[*index];
}

void parse_option(int* index, int argc, char** argv, Options* options) {
    const std::string argument(argv[*index]);
    if (argument == "--validate-only") {
        options->validate_only = true;
    } else if (argument == "--phase") {
        options->phase = require_option_value(index, argc, argv, argument);
    } else if (argument == "--output") {
        options->output = require_option_value(index, argc, argv, argument);
    } else if (argument == "--warmups") {
        options->warmups = parse_count(
            require_option_value(index, argc, argv, argument), "warmups");
    } else if (argument == "--repetitions") {
        options->repetitions = parse_count(
            require_option_value(index, argc, argv, argument), "repetitions");
    } else {
        throw std::runtime_error("unknown option: " + argument);
    }
}

void validate_options(const Options& options) {
    if (options.phase != "calibration" && options.phase != "evaluation") {
        throw std::runtime_error("--phase must be calibration or evaluation");
    }
    if (options.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (options.repetitions == 0) {
        throw std::runtime_error("--repetitions must be positive");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        parse_option(&index, argc, argv, &options);
    }
    validate_options(options);
    return options;
}

bool rejects_missing_tile(const prefix_attention::HostProblem& problem);
bool rejects_interior_partial_tile(const prefix_attention::HostProblem& problem);
bool rejects_scheduled_tile(const prefix_attention::HostProblem& problem);

int validate_only(const Options& options) {
    const std::vector<prefix_attention::CaseSpec> cases =
        options.phase == "calibration" ? prefix_attention::calibration_cases()
                                        : prefix_attention::evaluation_cases();
    for (const auto& spec : cases) {
        const auto problem = prefix_attention::make_problem(spec);
        if (!rejects_missing_tile(problem) ||
            !rejects_interior_partial_tile(problem) ||
            !rejects_scheduled_tile(problem)) {
            throw std::runtime_error("descriptor mutation was accepted");
        }
    }
    return 0;
}

std::string json_string(std::string_view value) {
    std::ostringstream result;
    result << '"';
    for (const char character : value) {
        switch (character) {
            case '"':
                result << "\\\"";
                break;
            case '\\':
                result << "\\\\";
                break;
            case '\n':
                result << "\\n";
                break;
            case '\r':
                result << "\\r";
                break;
            case '\t':
                result << "\\t";
                break;
            default:
                result << character;
                break;
        }
    }
    result << '"';
    return result.str();
}

void write_samples(std::ostream& output, const std::vector<float>& samples) {
    output << '[' << std::setprecision(17);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << samples[index];
    }
    output << ']';
}

void write_values(std::ostream& output, const std::vector<float>& values) {
    output << '[' << std::setprecision(17);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

void write_spec(std::ostream& output, const prefix_attention::CaseSpec& spec) {
    output << "{\"tokens\":" << spec.tokens
           << ",\"query_rows\":" << spec.query_rows
           << ",\"query_heads\":" << spec.query_heads
           << ",\"kv_heads\":" << spec.kv_heads
           << ",\"head_dim\":" << spec.head_dim
           << ",\"tile_tokens\":" << spec.tile_tokens
           << ",\"group_rows\":" << spec.group_rows
           << ",\"seed\":" << spec.seed
           << ",\"shared_prefix\":"
           << (spec.shared_prefix ? "true" : "false") << '}';
}

bool rejects_missing_tile(const prefix_attention::HostProblem& problem) {
    prefix_attention::HostProblem malformed = problem;
    malformed.tiles.pop_back();
    try {
        prefix_attention::validate_problem(malformed);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool rejects_interior_partial_tile(
    const prefix_attention::HostProblem& problem) {
    if (problem.tiles.size() < 2) {
        return false;
    }
    prefix_attention::HostProblem malformed = problem;
    if (malformed.tiles.front().valid_tokens == 0) {
        return false;
    }
    --malformed.tiles.front().valid_tokens;
    try {
        prefix_attention::validate_problem(malformed);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool rejects_scheduled_tile(const prefix_attention::HostProblem& problem) {
    prefix_attention::HostProblem malformed = problem;
    if (malformed.schedule_tiles_a.empty() ||
        malformed.schedule_tiles_a.front().valid_tokens == 0) {
        return false;
    }
    --malformed.schedule_tiles_a.front().valid_tokens;
    try {
        prefix_attention::validate_problem(malformed);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void write_path_receipt(std::ostream& output, prefix_attention::Path path,
                        const prefix_attention::RunResult& run,
                        const std::vector<double>& oracle,
                        const std::vector<float>& schedule_output,
                        bool schedule_equal) {
    output << "{\"path\":" << json_string(prefix_attention::path_name(path))
           << ",\"samples_ms\":";
    write_samples(output, run.samples_ms);
    output << std::setprecision(17)
           << ",\"median_ms\":" << run.elapsed_ms
           << ",\"quality_max_abs\":"
           << prefix_attention::max_absolute_error(run.output, oracle)
           << ",\"quality_max_rel\":"
           << prefix_attention::max_relative_error(run.output, oracle)
           << ",\"output_values\":";
    write_values(output, run.output);
    output << ",\"schedule_b_output_values\":";
    write_values(output, schedule_output);
    output << ",\"schedule_b_quality_max_abs\":"
           << prefix_attention::max_absolute_error(schedule_output, oracle)
           << ",\"schedule_b_quality_max_rel\":"
           << prefix_attention::max_relative_error(schedule_output, oracle)
           << ",\"digest\":" << json_string(
                  prefix_attention::output_digest(run.output))
           << ",\"schedule_b_digest\":" << json_string(
                  prefix_attention::output_digest(schedule_output))
           << ",\"schedule_b_bitwise_equal\":"
           << (schedule_equal ? "true" : "false")
           << ",\"estimated_kv_elements_read\":"
           << run.estimated_kv_elements_read << ",\"estimated_tile_loads\":"
           << run.estimated_tile_loads << ",\"output_elements_written\":"
           << run.output_elements_written << ",\"per_block_shared_bytes\":"
           << run.per_block_shared_bytes << '}';
}

void write_device(std::ostream& output, const cudaDeviceProp& properties,
                  int runtime_version, int driver_version) {
    output << "{\"name\":" << json_string(properties.name)
           << ",\"compute_major\":" << properties.major
           << ",\"compute_minor\":" << properties.minor
           << ",\"runtime_version\":" << runtime_version
           << ",\"driver_version\":" << driver_version << '}';
}

void write_timing_pair(std::ostream& output,
                       const prefix_attention::RunResult& baseline_before,
                       const prefix_attention::RunResult& candidate,
                       const prefix_attention::RunResult& baseline_after) {
    output << ",\"timing_pair\":{\"baseline_path\":\"fixed_tile_per_row\","
               "\"candidate_path\":\"shared_read_fixed_reduction\","
               "\"acquisition_order\":[\"baseline_before\",\"candidate\",\"baseline_after\"],"
               "\"baseline_before_samples_ms\":";
    write_samples(output, baseline_before.samples_ms);
    output << ",\"candidate_samples_ms\":";
    write_samples(output, candidate.samples_ms);
    output << ",\"baseline_after_samples_ms\":";
    write_samples(output, baseline_after.samples_ms);
    output << ",\"baseline_before_median_ms\":"
           << baseline_before.elapsed_ms
           << ",\"candidate_median_ms\":" << candidate.elapsed_ms
           << ",\"baseline_after_median_ms\":" << baseline_after.elapsed_ms
           << '}';
}

void write_case_receipt(std::ostream& receipt,
                        const prefix_attention::HostProblem& problem,
                        const std::vector<double>& oracle,
                        const std::vector<prefix_attention::CaseSpec>& cases,
                        std::size_t case_index, const Options& options,
                        const PathList& path_list) {
    const auto& spec = cases[case_index];
    receipt << "{\"name\":" << json_string(spec.name) << ",\"spec\":";
    write_spec(receipt, spec);
    receipt << ",\"input_digest\":"
            << json_string(prefix_attention::input_digest(problem))
            << ",\"oracle_digest\":"
            << json_string(prefix_attention::oracle_digest(oracle))
            << ",\"partial_final_tile\":"
            << (problem.tiles.back().valid_tokens < spec.tile_tokens ? "true"
                                                                      : "false")
            << ",\"missing_tile_rejected\":"
            << (rejects_missing_tile(problem) ? "true" : "false")
            << ",\"interior_partial_tile_rejected\":"
            << (rejects_interior_partial_tile(problem) ? "true" : "false")
            << ",\"scheduled_tile_rejected\":"
            << (rejects_scheduled_tile(problem) ? "true" : "false")
            << ",\"paths\":[";
    prefix_attention::DeviceProblem device;
    prefix_attention::allocate_device_problem(problem, &device);
    try {
        for (std::size_t path_index = 0; path_index < path_list.size();
             ++path_index) {
            if (path_index != 0) {
                receipt << ',';
            }
            const auto run_result = prefix_attention::run_path(
                problem, &device, path_list[path_index], false, options.warmups,
                options.repetitions);
            const auto schedule_b_result = prefix_attention::run_path(
                problem, &device, path_list[path_index], true, 0, 1);
            write_path_receipt(
                receipt, path_list[path_index], run_result, oracle,
                schedule_b_result.output,
                prefix_attention::bitwise_equal(run_result.output,
                                                schedule_b_result.output));
        }
        receipt << ']';
        const auto baseline_before = prefix_attention::run_path(
            problem, &device, prefix_attention::Path::FixedTilePerRow, false,
            options.warmups, options.repetitions);
        const auto candidate = prefix_attention::run_path(
            problem, &device, prefix_attention::Path::SharedReadFixedReduction,
            false, options.warmups, options.repetitions);
        const auto baseline_after = prefix_attention::run_path(
            problem, &device, prefix_attention::Path::FixedTilePerRow, false,
            options.warmups, options.repetitions);
        write_timing_pair(receipt, baseline_before, candidate, baseline_after);
    } catch (...) {
        prefix_attention::free_device_problem(&device);
        throw;
    }
    prefix_attention::free_device_problem(&device);
    receipt << '}';
}

int run(const Options& options) {
    int device_index = 0;
    check_cuda(cudaGetDevice(&device_index), "get CUDA device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_index),
               "get CUDA device properties");
    int runtime_version = 0;
    int driver_version = 0;
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "get CUDA runtime");
    check_cuda(cudaDriverGetVersion(&driver_version), "get CUDA driver");

    const std::vector<prefix_attention::CaseSpec> cases =
        options.phase == "calibration" ? prefix_attention::calibration_cases()
                                        : prefix_attention::evaluation_cases();
    const PathList path_list = paths();

    std::ofstream receipt(options.output);
    if (!receipt) {
        throw std::runtime_error("cannot open receipt output");
    }
    receipt << std::setprecision(17)
            << "{\"schema\":\"prefix-attention-gpu-receipt-v1\","
               "\"operator\":\"fixed-reduction-shared-prefix-attention\","
               "\"storage\":\"fp16-kv-fp32-accumulation\","
               "\"backend\":\"cuda\",\"phase\":"
            << json_string(options.phase) << ",\"device\":";
    write_device(receipt, properties, runtime_version, driver_version);
    receipt << ",\"input_generator\":"
            << json_string(prefix_attention::manifest_input_generator())
            << ",\"input_digest_algorithm\":"
            << json_string(prefix_attention::manifest_input_digest_algorithm())
            << ",\"oracle_id\":"
            << json_string(prefix_attention::manifest_oracle_id())
            << ",\"unavailable_evidence\":[\"host_launch_time\",\"dram_traffic\",\"power\",\"clocks\"]"
            << ",\"claims\":[]"
            << ",\"warmups\":" << options.warmups
            << ",\"repetitions\":" << options.repetitions
            << ",\"cases\":[";

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const prefix_attention::HostProblem problem =
            prefix_attention::make_problem(cases[case_index]);
        if (case_index != 0) {
            receipt << ',';
        }
        const std::vector<double> oracle = prefix_attention::fp64_oracle(problem);
        write_case_receipt(receipt, problem, oracle, cases, case_index, options,
                           path_list);
    }
    receipt << "]}\n";
    receipt.close();
    check_cuda(cudaDeviceSynchronize(), "finish CUDA receipt run");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        return options.validate_only ? validate_only(options) : run(options);
    } catch (const std::exception& error) {
        std::cerr << "prefix_attention_cuda: " << error.what() << '\n';
        return 1;
    }
}
