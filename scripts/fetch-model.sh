#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
models_dir="$repo_root/models"
q4_filename=Qwen3-8B-Q4_K_M.gguf
bf16_filename=Qwen3-8B-BF16.gguf
q4_sha256=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
bf16_sha256=5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30
repos=(Qwen/Qwen3-8B-GGUF unsloth/Qwen3-8B-GGUF)
tried_urls=()

mode=${1:---all}
case "$mode" in
    --all|--q4-only|--bf16-only) ;;
    *)
        echo "usage: $0 [--all|--q4-only|--bf16-only]" >&2
        exit 2
        ;;
esac

mkdir -p "$models_dir"

head_status() {
    local url=$1
    local status
    tried_urls+=("$url")
    status=$(curl -sS -I -o /dev/null -w '%{http_code}' "$url" || true)
    echo "HEAD $status $url" >&2
    [[ "$status" == 200 || "$status" == 302 ]]
}

download() {
    local url=$1
    local destination=$2
    local partial="$destination.part"
    if [[ -f "$destination" ]]; then
        echo "present: $destination"
        return
    fi
    curl --fail --location --retry 3 --continue-at - --output "$partial" "$url"
    mv "$partial" "$destination"
}

verify() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "$path" | cut -d' ' -f1)
    if [[ "$actual" != "$expected" ]]; then
        echo "error: $path has sha256 $actual, expected $expected" >&2
        exit 1
    fi
    echo "verified: $path"
}

print_tried_and_fail() {
    local artifact=$1
    echo "error: no allowed URL for $artifact" >&2
    echo "tried URLs:" >&2
    printf '  %s\n' "${tried_urls[@]}" >&2
    exit 1
}

fetch_q4() {
    local repo url selected=
    for repo in "${repos[@]}"; do
        url="https://huggingface.co/$repo/resolve/main/$q4_filename"
        if head_status "$url"; then
            selected=$url
            break
        fi
    done
    if [[ -z "$selected" ]]; then
        print_tried_and_fail "$q4_filename"
    fi
    download "$selected" "$models_dir/$q4_filename"
    verify "$models_dir/$q4_filename" "$q4_sha256"
}

fetch_bf16_single() {
    local repo url
    for repo in "${repos[@]}"; do
        url="https://huggingface.co/$repo/resolve/main/$bf16_filename"
        if head_status "$url"; then
            download "$url" "$models_dir/$bf16_filename"
            verify "$models_dir/$bf16_filename" "$bf16_sha256"
            return 0
        fi
    done
    return 1
}

shard_repo=
shard_count=0

find_bf16_shards() {
    local repo count total first_url
    for repo in "${repos[@]}"; do
        for count in $(seq 1 16); do
            printf -v total '%05d' "$count"
            first_url="https://huggingface.co/$repo/resolve/main/Qwen3-8B-BF16/Qwen3-8B-BF16-00001-of-$total.gguf"
            if head_status "$first_url"; then
                shard_repo=$repo
                shard_count=$count
                return 0
            fi
        done
    done
    return 1
}

fetch_bf16_shards() {
    local index total part url destination
    if ! find_bf16_shards; then
        print_tried_and_fail "$bf16_filename or its allowed shard pattern"
    fi
    printf -v total '%05d' "$shard_count"

    # Check every shard before downloading the first large file.
    for index in $(seq 1 "$shard_count"); do
        printf -v part '%05d' "$index"
        url="https://huggingface.co/$shard_repo/resolve/main/Qwen3-8B-BF16/Qwen3-8B-BF16-$part-of-$total.gguf"
        if ! head_status "$url"; then
            print_tried_and_fail "BF16 shard $part of $total"
        fi
    done
    for index in $(seq 1 "$shard_count"); do
        printf -v part '%05d' "$index"
        url="https://huggingface.co/$shard_repo/resolve/main/Qwen3-8B-BF16/Qwen3-8B-BF16-$part-of-$total.gguf"
        destination="$models_dir/Qwen3-8B-BF16-$part-of-$total.gguf"
        download "$url" "$destination"
    done
    echo "warning: shard hashes are not checked against the gate artifact" >&2
}

fetch_bf16() {
    if ! fetch_bf16_single; then
        fetch_bf16_shards
    fi
}

write_sums() {
    local sums_tmp="$models_dir/SHA256SUMS.tmp"
    (
        cd "$repo_root"
        find models -maxdepth 1 -type f -name '*.gguf' -print0 \
            | sort -z \
            | xargs -0 --no-run-if-empty sha256sum
    ) > "$sums_tmp"
    mv "$sums_tmp" "$models_dir/SHA256SUMS"
    cat "$models_dir/SHA256SUMS"
}

if [[ "$mode" != --bf16-only ]]; then
    fetch_q4
fi
if [[ "$mode" != --q4-only ]]; then
    fetch_bf16
fi
write_sums
