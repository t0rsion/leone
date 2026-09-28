#!/usr/bin/env python3
"""Generate the case tables consumed by the CUDA and Metal harnesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
MANIFEST = ROOT / "gpu_manifest.json"

MANIFEST_SCHEMA = "prefix-attention-gpu-manifest-v1"
MANIFEST_OPERATOR = "fixed-reduction-shared-prefix-attention"
MANIFEST_STORAGE = "fp16-kv-fp32-accumulation"
MANIFEST_INPUT_GENERATOR = "splitmix64-v1"
MANIFEST_INPUT_DIGEST = "fnv1a64-le-q-f32-k-f16-v-f16-v1"
MANIFEST_ORACLE_ID = "scalar-fp64-attention-v2"


def cases(manifest: dict, phase: str) -> list[dict]:
    return manifest[phase]


def cpp_case(case: dict) -> str:
    shared = "true" if case["shared_prefix"] else "false"
    fields = ", ".join(
        [
            json.dumps(case["name"]),
            str(case["tokens"]),
            str(case["query_rows"]),
            str(case["query_heads"]),
            str(case["kv_heads"]),
            str(case["head_dim"]),
            str(case["tile_tokens"]),
            str(case["group_rows"]),
            str(case["seed"]),
            shared,
        ]
    )
    return "        {" + fields + "},"


def swift_case(case: dict) -> str:
    shared = "true" if case["shared_prefix"] else "false"
    return (
        "        CaseSpec(name: "
        + json.dumps(case["name"])
        + f", tokens: {case['tokens']}, queryRows: {case['query_rows']},"
        + f" queryHeads: {case['query_heads']}, kvHeads: {case['kv_heads']},"
        + f" headDim: {case['head_dim']}, tileTokens: {case['tile_tokens']},"
        + f" groupRows: {case['group_rows']}, seed: {case['seed']},"
        + f" sharedPrefix: {shared}),"
    )


def render_cpp(manifest: dict) -> str:
    calibration = "\n".join(cpp_case(case) for case in cases(manifest, "calibration"))
    evaluation = "\n".join(cpp_case(case) for case in cases(manifest, "evaluation"))
    return f'''#pragma once

namespace prefix_attention {{

inline constexpr const char* manifest_input_generator() {{
    return {json.dumps(manifest["input_generator"])};
}}

inline constexpr const char* manifest_input_digest_algorithm() {{
    return {json.dumps(manifest["input_digest"])};
}}

inline constexpr const char* manifest_oracle_id() {{
    return {json.dumps(manifest["oracle_id"])};
}}

inline std::vector<CaseSpec> manifest_calibration_cases() {{
    return {{
{calibration}
    }};
}}

inline std::vector<CaseSpec> manifest_evaluation_cases() {{
    return {{
{evaluation}
    }};
}}

}}  // namespace prefix_attention
'''


def render_swift(manifest: dict) -> str:
    calibration = "\n".join(swift_case(case) for case in cases(manifest, "calibration"))
    evaluation = "\n".join(swift_case(case) for case in cases(manifest, "evaluation"))
    return f'''func manifestCalibrationCases() -> [CaseSpec] {{
    [
{calibration}
    ]
}}

let manifestInputGenerator = {json.dumps(manifest["input_generator"])}
let manifestInputDigestAlgorithm = {json.dumps(manifest["input_digest"])}
let manifestOracleID = {json.dumps(manifest["oracle_id"])}

func manifestEvaluationCases() -> [CaseSpec] {{
    [
{evaluation}
    ]
}}
'''


def validate_required_fields(manifest: dict) -> None:
    required = {
        "schema", "operator", "storage", "input_generator", "input_digest",
        "oracle_id", "calibration", "evaluation", "paths", "warmup_runs",
        "measured_runs",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")


def validate_manifest_identity(manifest: dict) -> None:
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError("manifest schema is unsupported")
    if manifest["operator"] != MANIFEST_OPERATOR:
        raise ValueError("manifest operator is unsupported")
    if manifest["storage"] != MANIFEST_STORAGE:
        raise ValueError("manifest storage is unsupported")
    if manifest["input_generator"] != MANIFEST_INPUT_GENERATOR:
        raise ValueError("manifest input generator is unsupported")
    if manifest["input_digest"] != MANIFEST_INPUT_DIGEST:
        raise ValueError("manifest input digest is unsupported")
    if manifest["oracle_id"] != MANIFEST_ORACLE_ID:
        raise ValueError("manifest oracle is unsupported")


def validate_manifest_paths(manifest: dict) -> None:
    if len(manifest["paths"]) != 4 or len(set(manifest["paths"])) != 4:
        raise ValueError("manifest must contain four unique paths")
    if set(manifest["paths"]) != {
        "per_row", "fixed_tile_per_row", "shared_read_unconstrained",
        "shared_read_fixed_reduction",
    }:
        raise ValueError("manifest paths are unsupported")
def validate_manifest_case_names(manifest: dict) -> None:
    calibration_names = {case["name"] for case in manifest["calibration"]}
    evaluation_names = {case["name"] for case in manifest["evaluation"]}
    if len(calibration_names) != len(manifest["calibration"]) or len(evaluation_names) != len(manifest["evaluation"]):
        raise ValueError("manifest case names must be unique")
    if not calibration_names or not evaluation_names:
        raise ValueError("manifest phases must contain cases")
    if calibration_names & evaluation_names:
        raise ValueError("calibration and evaluation names must be disjoint")


def validate_manifest_fields(manifest: dict) -> None:
    validate_required_fields(manifest)
    validate_manifest_identity(manifest)
    validate_manifest_paths(manifest)
    validate_manifest_case_names(manifest)


def validate_manifest_limits(manifest: dict) -> None:
    for field in ("maximum_group_rows", "maximum_tile_tokens", "maximum_head_dim"):
        value = manifest.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"manifest bound {field} is invalid")
    for case in manifest["calibration"] + manifest["evaluation"]:
        if (case["group_rows"] > manifest["maximum_group_rows"] or
                case["tile_tokens"] > manifest["maximum_tile_tokens"] or
                case["head_dim"] > manifest["maximum_head_dim"]):
            raise ValueError(f"case {case['name']} exceeds a manifest bound")


def validate_run_counts(manifest: dict) -> None:
    if manifest["warmup_runs"] < 0 or manifest["measured_runs"] <= 0:
        raise ValueError("manifest run counts are invalid")


def validate_case_fields(case: dict, required: set[str]) -> None:
    if required.difference(case):
        raise ValueError(f"case {case.get('name')} is incomplete")
    integer_fields = required - {"name", "shared_prefix"}
    if any(not isinstance(case[field], int) or isinstance(case[field], bool) for field in integer_fields):
        raise ValueError(f"case {case.get('name')} has a noninteger field")
    if not isinstance(case["name"], str) or not case["name"] or not isinstance(case["shared_prefix"], bool):
        raise ValueError(f"case {case.get('name')} has an invalid identity field")


def validate_case_dimensions(case: dict) -> None:
    if (case["tokens"] <= 0 or case["query_rows"] <= 0 or
            case["query_rows"] > case["tokens"] or case["query_heads"] <= 0 or
            case["kv_heads"] <= 0 or case["head_dim"] <= 0 or
            case["tile_tokens"] <= 0 or case["group_rows"] <= 0 or
            case["query_heads"] % case["kv_heads"]):
        raise ValueError(f"case {case['name']} has invalid dimensions")


def validate_manifest_cases(manifest: dict) -> None:
    validate_manifest_limits(manifest)
    validate_run_counts(manifest)
    all_cases = manifest["calibration"] + manifest["evaluation"]
    required_case_fields = {
        "name", "tokens", "query_rows", "query_heads", "kv_heads", "head_dim",
        "tile_tokens", "group_rows", "seed", "shared_prefix",
    }
    for case in all_cases:
        validate_case_fields(case, required_case_fields)
        validate_case_dimensions(case)


def validate_manifest(manifest: dict) -> None:
    validate_manifest_fields(manifest)
    validate_manifest_cases(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-output", type=Path)
    parser.add_argument("--swift-output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def write_or_check(path: Path, content: str, check: bool) -> None:
    current = path.read_text() if path.exists() else None
    if check:
        if current != content:
            raise SystemExit(f"generated manifest is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    args = parse_args()
    if not args.cpp_output and not args.swift_output:
        raise SystemExit("one generated output is required")
    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    if args.cpp_output:
        write_or_check(args.cpp_output, render_cpp(manifest), args.check)
    if args.swift_output:
        write_or_check(args.swift_output, render_swift(manifest), False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
