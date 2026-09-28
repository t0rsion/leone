#pragma once

namespace prefix_attention {

inline constexpr const char* manifest_input_generator() {
    return "splitmix64-v1";
}

inline constexpr const char* manifest_input_digest_algorithm() {
    return "fnv1a64-le-q-f32-k-f16-v-f16-v1";
}

inline constexpr const char* manifest_oracle_id() {
    return "scalar-fp64-attention-v2";
}

inline std::vector<CaseSpec> manifest_calibration_cases() {
    return {
        {"calibration_short_gqa", 17, 4, 4, 2, 32, 8, 2, 7101, true},
        {"calibration_long_gqa", 257, 8, 4, 2, 64, 32, 2, 7102, true},
        {"calibration_unrelated", 65, 4, 4, 2, 32, 16, 2, 7103, false},
    };
}

inline std::vector<CaseSpec> manifest_evaluation_cases() {
    return {
        {"evaluation_partial_gqa", 33, 6, 4, 2, 32, 16, 3, 8201, true},
        {"evaluation_long_gqa", 513, 8, 8, 2, 64, 64, 2, 8202, true},
        {"evaluation_short_shared", 7, 3, 2, 1, 16, 4, 1, 8203, true},
        {"evaluation_unrelated", 129, 6, 4, 2, 32, 32, 3, 8204, false},
    };
}

}  // namespace prefix_attention
