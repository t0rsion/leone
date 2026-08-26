#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-"$root/models/Qwen3-8B-Q4_K_M.gguf"}
tokens=${TOKENS:-64}
build_dir="$root/target/llama-token-oracle"
oracle="$build_dir/llama-cli"
leone="$root/target/release/leone"

build_oracle() {
    mkdir -p "$build_dir"
    local cxx=${CXX:-/usr/bin/c++}
    local source="$root/scripts/llama-token-oracle.cpp"
    local llama="$root/external/llama.cpp"
    local library="$llama/build/bin"
    local definitions=(
        -DGGML_BACKEND_SHARED
        -DGGML_SHARED
        -DGGML_USE_CPU
        -DGGML_USE_CUDA
        -DLLAMA_SHARED
        -DLLAMA_SUBPROCESS
    )
    local includes=(
        -I"$llama/tools/completion"
        -I"$llama/common"
        -I"$llama/vendor"
        -I"$llama/include"
        -I"$llama/ggml/include"
    )
    "$cxx" -O3 -DNDEBUG -fPIC "${definitions[@]}" "${includes[@]}" \
        -c "$source" -o "$build_dir/oracle.o"
    "$cxx" -O3 -DNDEBUG "${definitions[@]}" \
        -c "$llama/tools/completion/main.cpp" -o "$build_dir/main.o"
    "$cxx" -O3 -DNDEBUG "$build_dir/main.o" "$build_dir/oracle.o" \
        -o "$oracle" \
        -Wl,-rpath,"$library" \
        "$library/libllama-common.so" \
        "$library/libllama.so" \
        "$library/libggml.so" \
        "$library/libggml-cpu.so" \
        "$library/libggml-cuda.so" \
        "$library/libggml-base.so" \
        "$llama/build/common/libllama-common-base.a" \
        /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
        -lpthread -ldl
}

if [[ ! -x "$oracle" || "$root/scripts/llama-token-oracle.cpp" -nt "$oracle" ]]; then
    build_oracle
fi

taskset -c 16-31 cargo +1.92 build --release -p leone-cli \
    --manifest-path "$root/Cargo.toml"

temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT

prompts=(
    "The capital of France is"
    "Write a Rust function that adds two integers:"
    "Once upon a time in a quiet village,"
)

first_token_failed=0
for index in "${!prompts[@]}"; do
    prompt=${prompts[$index]}
    llama_log="$temporary/llama-$index.log"
    leone_tokens="$temporary/leone-$index.json"
    leone_logits="$temporary/leone-$index.logits"
    leone_log="$temporary/leone-$index.log"

    LEONE_LLAMA_TOKEN_LOG=1 LEONE_LLAMA_LOGITS=1 \
        taskset -c 0-3,12-15 "$oracle" \
        -m "$model" -p "$prompt" -n "$tokens" -ngl 99 \
        --temp 0 --top-k 1 -no-cnv --seed 0 --log-disable \
        >/dev/null 2>"$llama_log"
    taskset -c 0-3,12-15 "$leone" generate \
        -m "$model" -p "$prompt" -n "$tokens" --backend cuda \
        --debug-tokens "$leone_tokens" --debug-logits "$leone_logits" \
        >/dev/null 2>"$leone_log"

    mapfile -t llama_ids < <(sed -n 's/^leone-token: //p' "$llama_log")
    leone_line=$(tr -d '[][:space:]' <"$leone_tokens")
    IFS=',' read -r -a leone_ids <<<"$leone_line"

    matched=0
    limit=${#llama_ids[@]}
    if (( ${#leone_ids[@]} < limit )); then
        limit=${#leone_ids[@]}
    fi
    while (( matched < limit )) && [[ "${llama_ids[$matched]}" == "${leone_ids[$matched]}" ]]; do
        ((matched += 1))
    done

    printf 'prompt %d: exact prefix %d tokens' "$((index + 1))" "$matched"
    if (( matched == ${#llama_ids[@]} && matched == ${#leone_ids[@]} )); then
        printf ', full match\n'
        continue
    fi
    printf ', divergence at token %d\n' "$((matched + 1))"
    sed -n "$((matched + 1))p" "$leone_logits"
    sed -n 's/^llama-logits: //p' "$llama_log" | sed -n "$((matched + 1))p"
    if (( matched == 0 )); then
        first_token_failed=1
    fi
done

if (( first_token_failed != 0 )); then
    echo "first-token oracle failed" >&2
    exit 1
fi
