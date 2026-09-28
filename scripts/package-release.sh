#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
dist=${LEONE_PACKAGE_DIST:-$root/dist}
platform=${LEONE_PACKAGE_PLATFORM:-}
target_override=${LEONE_PACKAGE_TARGET:-}
include_evidence=${LEONE_PACKAGE_EVIDENCE:-1}
cargo_bin=${LEONE_CARGO:-${CARGO:-cargo}}
binary_override=${LEONE_RELEASE_BINARY:-}

usage() {
    echo "usage: package-release.sh [--platform linux-x86_64|darwin-arm64] [--target TRIPLE] [--runtime-only]" >&2
}

while (($# > 0)); do
    case "$1" in
        --platform)
            (($# >= 2)) || { usage; exit 2; }
            platform=$2
            shift 2
            ;;
        --target)
            (($# >= 2)) || { usage; exit 2; }
            target_override=$2
            shift 2
            ;;
        --runtime-only)
            include_evidence=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ -z $platform && -n $target_override ]]; then
    case "$target_override" in
        x86_64-unknown-linux-gnu) platform=linux-x86_64 ;;
        aarch64-apple-darwin) platform=darwin-arm64 ;;
        *)
            echo "error: unsupported package target: $target_override" >&2
            exit 1
            ;;
    esac
fi
if [[ -z $platform ]]; then
    case "$(uname -s):$(uname -m)" in
        Linux:x86_64) platform=linux-x86_64 ;;
        Darwin:arm64|Darwin:aarch64) platform=darwin-arm64 ;;
        *)
            echo "error: release packaging supports Linux x86_64 and Darwin arm64" >&2
            exit 1
            ;;
    esac
fi

case "$platform" in
    linux-x86_64)
        rust_target=x86_64-unknown-linux-gnu
        backend=cuda
        cargo_features=()
        binary=target/release/leone
        ;;
    darwin-arm64)
        rust_target=aarch64-apple-darwin
        backend=metal
        cargo_features=(--no-default-features --features metal)
        binary="target/$rust_target/release/leone"
        ;;
    *)
        echo "error: unsupported package platform: $platform" >&2
        exit 1
        ;;
esac
if [[ -n $target_override && $target_override != "$rust_target" ]]; then
    echo "error: platform $platform requires target $rust_target" >&2
    exit 1
fi

source_commit=$(git -C "$root" rev-parse HEAD)
if [[ -n $binary_override ]]; then
    if [[ $binary_override == /* ]]; then
        binary=$binary_override
    else
        binary="$root/$binary_override"
    fi
fi

ensure_clean_source() {
    if [[ -n $(git -C "$root" status --porcelain --untracked-files=all) ]]; then
        echo "error: release packaging requires a clean source tree" >&2
        exit 2
    fi
    if ! git -C "$root" diff --quiet "$source_commit" -- ||
        ! git -C "$root" diff --cached --quiet; then
        echo "error: release source changed during packaging" >&2
        exit 2
    fi
}

ensure_regular_source() {
    local source=$1
    [[ -f $source && ! -L $source ]] || {
        echo "error: package input is not a regular file: $source" >&2
        exit 1
    }
    local parent=$source
    while [[ $parent == */* ]]; do
        parent=${parent%/*}
        [[ -n $parent ]] || parent=.
        [[ ! -L $parent ]] || {
            echo "error: package input contains a symlink parent: $source" >&2
            exit 1
        }
        [[ $parent != . ]] || break
    done
}

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -1)
if [[ -z $version ]]; then
    echo "error: workspace version is missing" >&2
    exit 1
fi

ensure_clean_source

if [[ $cargo_bin == */* ]]; then
    [[ -x $cargo_bin ]] || {
        echo "error: cargo executable is not runnable: $cargo_bin; set CARGO or LEONE_CARGO" >&2
        exit 2
    }
else
    cargo_bin=$(command -v "$cargo_bin" || true)
    [[ -n $cargo_bin && -x $cargo_bin ]] || {
        echo "error: cargo executable is unavailable; set CARGO or LEONE_CARGO to a cargo path" >&2
        exit 2
    }
fi
cargo_command=("$cargo_bin" +1.92)

epoch=${SOURCE_DATE_EPOCH:-$(git -C "$root" show -s --format=%ct HEAD)}
if [[ ! $epoch =~ ^[0-9]+$ ]]; then
    echo "error: SOURCE_DATE_EPOCH must be an integer" >&2
    exit 1
fi

name="leone-$version-$platform"
evidence="leone-$version-evidence"
plans=()
if [[ $platform == linux-x86_64 ]]; then
    plans=(
        plans/llama3.2-1b-sm89.json
        plans/qwen3-8b-sm89.json
    )
fi
receipts=(
    receipts/2026-08-27T11-58-08Z-plan-search-v1.json
    receipts/2026-08-27T12-08-20Z-plan-search-v1.json
    receipts/2026-09-11T18:18:33Z-quality-db8f6bb8.json
    receipts/2026-09-11T18:18:52Z-quality-d76766b1.json
    receipts/2026-09-11T18:20:26Z-quality-3e839ef9.json
    receipts/2026-09-11T18:20:36Z-quality-b5b405af.json
    receipts/batched-service-study.json
    receipts/concurrent-service-study.json
    receipts/openai-client-check.json
    receipts/quality-concurrent-service.json
    receipts/quality-llama-v03.json
    receipts/quality-qwen3-8b.json
    receipts/source-inputs.json
    receipts/source-inputs-v04-prestudy.json
)
runtime_docs=(
    docs/models.md
    docs/openai-api.md
    docs/oracle.md
    docs/release.md
    docs/release-evidence.md
    docs/release-candidate.md
    docs/branching-service-evidence.md
    docs/client-workflow.md
    docs/memory-accounting.md
    docs/concurrent-service-evidence.md
    docs/concurrent-service.svg
)
licenses=(LICENSE-APACHE LICENSE-MIT)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT HUP INT TERM
path_remaps=(
    "--remap-path-prefix=$root=/source/leone"
    "--remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo"
    "--remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=/rustup"
)
release_rustflags="${RUSTFLAGS:-}"
for remap in "${path_remaps[@]}"; do
    release_rustflags+=" ${remap}"
done

run_build() {
    if [[ -n ${LEONE_BUILD_CPUSET:-} && $(command -v taskset || true) ]]; then
        taskset -c "$LEONE_BUILD_CPUSET" "$@"
    else
        "$@"
    fi
}

copy_files() {
    local destination=$1
    shift
    local source
    for source in "$@"; do
        ensure_regular_source "$source"
        cp "$source" "$destination/"
    done
}

build() {
    if [[ -n $binary_override ]]; then
        return
    fi
    local -a command=("${cargo_command[@]}" build --release -p leone-cli --locked)
    if [[ $platform == darwin-arm64 ]]; then
        command+=(--target "$rust_target")
    fi
    if ((${#cargo_features[@]} > 0)); then
        command+=("${cargo_features[@]}")
    fi
    if [[ $platform == darwin-arm64 ]]; then
        CFLAGS="${CFLAGS:-} -ffile-prefix-map=$root=/source/c -fdebug-prefix-map=$root=/source/c" \
            CXXFLAGS="${CXXFLAGS:-} -ffile-prefix-map=$root=/source/c -fdebug-prefix-map=$root=/source/c" \
            OBJCFLAGS="${OBJCFLAGS:-} -ffile-prefix-map=$root=/source/objc -fdebug-prefix-map=$root=/source/objc" \
            RUSTFLAGS="$release_rustflags" run_build "${command[@]}"
    else
        RUSTFLAGS="$release_rustflags" run_build "${command[@]}"
    fi
    ensure_clean_source
}

validate_prebuilt() {
    if [[ -z $binary_override ]]; then
        return 0
    fi
    local build_info build_commit source_manifest
    build_info=$("$binary" --build-info)
    jq -e --arg target "$rust_target" --arg feature "$backend" '
        .schema_version == "leone.build-info.v1" and
        (.source_commit | type == "string" and test("^[0-9a-f]{40}$")) and
        .source_tree_dirty == false and .provenance_unknown == false and
        .profile == "release" and
        (.features | type == "string" and ((split(",") | index($feature)) != null)) and
        .target == $target
    ' <<<"$build_info" >/dev/null || {
        echo "error: prebuilt binary has invalid release provenance" >&2
        exit 2
    }
    build_commit=$(jq -er '.source_commit' <<<"$build_info")
    if [[ $include_evidence == 1 ]]; then
        source_manifest=$(jq -er '.source_manifest' "$evidence_manifest")
    else
        source_manifest=${LEONE_RELEASE_SOURCE_MANIFEST:-}
    fi
    if [[ -n $source_manifest ]]; then
        python3 scripts/source_inputs.py check "$build_commit" --manifest "$source_manifest"
    elif [[ $build_commit != "$source_commit" ]]; then
        echo "error: prebuilt binary source commit requires a source input manifest" >&2
        exit 2
    fi
}

validate_native_binary_hash() {
    if [[ $include_evidence != 1 || $evidence_release_line != v0.4 || $evidence_status != complete ]]; then
        return 0
    fi
    local expected actual
    expected=$(jq -er --arg platform "$platform" --arg target "$rust_target" --arg backend "$backend" '
        .shared_records[]
        | select(.role == "client" and .platform == $platform and .target == $target and .backend == $backend)
        | .binary_sha256
    ' "$evidence_manifest") || {
        echo "error: complete v0.4 evidence has no native client binary hash for $platform" >&2
        exit 2
    }
    if [[ ! $expected =~ ^[0-9a-f]{64}$ ]]; then
        echo "error: native client binary hash is invalid for $platform" >&2
        exit 2
    fi
    actual=$(packaging/checksum.sh "$binary" | awk '{print $1}')
    if [[ $actual != "$expected" ]]; then
        echo "error: runtime binary differs from native client evidence for $platform" >&2
        exit 2
    fi
}

manifest() {
    local artifact=$1
    local output="$stage/$artifact/MANIFEST.sha256"
    python3 "$root/packaging/write-manifest.py" "$stage/$artifact" "$output"
}

archive() {
    local artifact=$1
    local output="$dist/$artifact.tar.gz"
    if [[ $platform == linux-x86_64 ]]; then
        tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" \
            -C "$stage" -cf - "$artifact" | gzip -n -9 >"$output"
    else
        python3 "$root/packaging/make-archive.py" "$stage" "$artifact" "$output" "$epoch"
    fi
    (
        cd "$dist"
        "$root/packaging/checksum.sh" "$(basename "$output")" >"$(basename "$output").sha256"
    )
}

cd "$root"
ensure_clean_source
evidence_manifest=""
evidence_release_line=""
evidence_status=""
if [[ $include_evidence == 1 ]]; then
    evidence_manifest=${LEONE_RELEASE_EVIDENCE_MANIFEST:-}
    if [[ -z $evidence_manifest ]]; then
        evidence_manifest=$(python3 scripts/release-evidence-manifest.py select --root "$root")
    elif [[ $evidence_manifest != /* ]]; then
        evidence_manifest="$root/$evidence_manifest"
    fi
    python3 scripts/release-evidence-manifest.py validate \
        --manifest "$evidence_manifest" --complete >/dev/null
    evidence_release_line=$(jq -er '.release_line' "$evidence_manifest")
    evidence_status=$(jq -er '.status' "$evidence_manifest")
    if [[ $evidence_release_line == v0.4 && $evidence_status == complete && -z $binary_override ]]; then
        echo "error: complete evidence requires LEONE_RELEASE_BINARY" >&2
        exit 2
    fi
fi
build
ensure_regular_source "$binary"
validate_prebuilt
validate_native_binary_hash
ensure_regular_source packaging/install.sh
ensure_regular_source packaging/checksum.sh
ensure_regular_source packaging/README.md
ensure_regular_source packaging/compatibility.json

mkdir -p "$stage/$name/bin" "$stage/$name/docs" "$stage/$name/receipts" \
    "$stage/$name/tools"
if ((${#plans[@]} > 0)); then
    mkdir -p "$stage/$name/plans"
    copy_files "$stage/$name/plans" "${plans[@]}"
fi
install -m 0755 "$binary" "$stage/$name/bin/leone"
install -m 0755 packaging/install.sh "$stage/$name/install.sh"
install -m 0755 packaging/checksum.sh "$stage/$name/tools/checksum.sh"
cp packaging/README.md packaging/compatibility.json "$stage/$name/"
printf '%s\n' "$rust_target" >"$stage/$name/archive-target"
chmod 0644 "$stage/$name/archive-target"
python3 packaging/write-package-info.py \
    --package "$stage/$name/package-info.json" \
    --environment "$stage/$name/environment.json" \
    --version "$version" --platform "$platform" --target "$rust_target" \
    --backend "$backend" --epoch "$epoch" --package-kind runtime
copy_files "$stage/$name/docs" "${runtime_docs[@]}"
copy_files "$stage/$name/receipts" receipts/INDEX.md "${receipts[@]}"
while IFS= read -r -d '' receipt; do
    case "$receipt" in
        receipts/metal-rms-reduction/*.json) ;;
        *)
            echo "error: unexpected Metal RMS receipt path: $receipt" >&2
            exit 1
            ;;
    esac
    ensure_regular_source "$receipt"
    destination="$stage/$name/$receipt"
    mkdir -p "$(dirname "$destination")"
    cp "$receipt" "$destination"
done < <(git -C "$root" ls-files -z -- 'receipts/metal-rms-reduction/')
copy_files "$stage/$name" "${licenses[@]}"

manifest "$name"
mkdir -p "$dist"
archive "$name"

if [[ $include_evidence == 1 ]]; then
    python3 scripts/release-evidence-manifest.py stage \
        --manifest "$evidence_manifest" \
        --source-root "$root" \
        --destination "$stage/$evidence"
    python3 packaging/write-package-info.py \
        --package "$stage/$evidence/package-info.json" \
        --environment "$stage/$evidence/environment.json" \
        --version "$version" --platform "$platform" --target "$rust_target" \
        --backend "$backend" --epoch "$epoch" --package-kind evidence
    ensure_clean_source
    manifest "$evidence"
    archive "$evidence"
fi

printf '%s\n' "$dist/$name.tar.gz"
if [[ $include_evidence == 1 ]]; then
    printf '%s\n' "$dist/$evidence.tar.gz"
fi
