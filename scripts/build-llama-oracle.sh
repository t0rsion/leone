#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
llama_dir=${LLAMA_CPP_DIR:-$root/external/llama.cpp}
build_dir=${LLAMA_CPP_BUILD_DIR:-}
library_dir=${LLAMA_CPP_LIBRARY_DIR:-}
output=${1:-$root/target/llama-logits-oracle}

if [[ $llama_dir != /* ]]; then
  llama_dir="$root/$llama_dir"
fi
if [[ -n $build_dir && $build_dir != /* ]]; then
  build_dir="$root/$build_dir"
fi
if [[ -z $build_dir ]]; then
  build_dir="$llama_dir/build"
fi
if [[ -n $library_dir && $library_dir != /* ]]; then
  library_dir="$root/$library_dir"
fi
if [[ -z $library_dir ]]; then
  library_dir="$build_dir/bin"
fi
if [[ $output != /* ]]; then
  output="$root/$output"
fi

if ! git -C "$llama_dir" rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: pinned llama.cpp checkout is missing: $llama_dir" >&2
  exit 1
fi
if [[ ! -f "$root/external/PINNED" ]]; then
  echo "error: external/PINNED is missing" >&2
  exit 1
fi
pin=$(tr -d '[:space:]' <"$root/external/PINNED")
if [[ ! $pin =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: external/PINNED must contain one 40-character commit hash" >&2
  exit 1
fi
llama_commit=$(git -C "$llama_dir" rev-parse HEAD)
if [[ $llama_commit != "$pin" ]]; then
  echo "error: llama.cpp HEAD $llama_commit differs from external/PINNED $pin" >&2
  exit 1
fi
if [[ -n $(git -C "$llama_dir" status --porcelain --untracked-files=all) ]]; then
  echo "error: pinned llama.cpp checkout has local changes" >&2
  exit 1
fi

case $(uname -s) in
  Darwin)
    shared_extension=dylib
    rpath_origin='@loader_path'
    ;;
  Linux)
    shared_extension=so
    rpath_origin="\$ORIGIN"
    ;;
  *)
    echo "error: unsupported host platform: $(uname -s)" >&2
    exit 1
    ;;
esac

for input in "$llama_dir/include/llama.h" "$llama_dir/ggml/include/ggml-backend.h"; do
  if [[ ! -e $input ]]; then
    echo "error: llama.cpp build input is missing: $input" >&2
    exit 1
  fi
done

library_flags=(-lllama -lggml-base)
for library in "$library_dir/libllama.$shared_extension" "$library_dir/libggml-base.$shared_extension"; do
  if [[ ! -e $library ]]; then
    echo "error: llama.cpp library is missing: $library" >&2
    exit 1
  fi
done
if [[ -e "$library_dir/libggml.$shared_extension" ]]; then
  library_flags+=(-lggml)
fi

relative_library_dir=$(python3 - "$output" "$library_dir" <<'PY'
import os
import sys

print(os.path.relpath(os.path.abspath(sys.argv[2]), os.path.dirname(os.path.abspath(sys.argv[1]))))
PY
)
rpath="$rpath_origin/$relative_library_dir"
mkdir -p "$(dirname "$output")"

build_command=(
  "${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror
  -I"$llama_dir/include" -I"$llama_dir/ggml/include"
  "$root/research/oracle/llama_logits.cpp"
  -L"$library_dir" "-Wl,-rpath,$rpath"
  "${library_flags[@]}" -o "$output"
)
if [[ -n ${LEONE_BUILD_CPUSET:-} ]]; then
  if ! command -v taskset >/dev/null 2>&1; then
    echo "error: LEONE_BUILD_CPUSET requires taskset on this host" >&2
    exit 1
  fi
  taskset -c "$LEONE_BUILD_CPUSET" "${build_command[@]}"
else
  "${build_command[@]}"
fi

echo "$output"
