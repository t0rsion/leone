#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
llama_dir=${LLAMA_CPP_DIR:-$root/external/llama.cpp}
build_dir=${LLAMA_CPP_BUILD_DIR:-$root/external/llama.cpp/build-cpu}

run_build() {
    if [[ -n ${LEONE_BUILD_CPUSET:-} ]]; then
        taskset -c "$LEONE_BUILD_CPUSET" "$@"
    else
        "$@"
    fi
}

if [[ $llama_dir != /* ]]; then
    llama_dir="$root/$llama_dir"
fi
if [[ $build_dir != /* ]]; then
    build_dir="$root/$build_dir"
fi

LEONE_LLAMA_CPP_CPU_ONLY=1 \
LLAMA_CPP_DIR="$llama_dir" \
LLAMA_CPP_BUILD_DIR="$build_dir" \
bash scripts/fetch-llama-cpp.sh

library_dir="$build_dir/bin"
export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$library_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

relative_build_dir=${build_dir#"$llama_dir"/}
llama_cpp_rpath=
if [[ $relative_build_dir != "$build_dir" ]]; then
    case $(uname -s) in
        Darwin) llama_cpp_rpath="@loader_path/../../llama.cpp/$relative_build_dir/bin" ;;
        *) llama_cpp_rpath="\$\$ORIGIN/../../llama.cpp/${relative_build_dir}/bin" ;;
    esac
fi

run_build make -B -C external/shim \
    LLAMA_CPP_DIR="$llama_dir" \
    LLAMA_CPP_BUILD_DIR="$build_dir" \
    LLAMA_CPP_RPATH="$llama_cpp_rpath" \
    build/ggml-dequant

run_differential() {
    local test_name=$1
    run_build cargo +1.92 test --jobs "${CARGO_BUILD_JOBS:-2}" -p leone-gguf --release --locked \
        --features differential --test differential "$test_name" -- \
        --ignored --test-threads=1
}

run_differential random_blocks_match_llama_cpp
run_differential isolated_q_k_byte_codes_match_llama_cpp
run_differential every_q_k_logical_field_code_matches_llama_cpp
