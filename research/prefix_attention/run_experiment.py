#!/usr/bin/env python3
"""Runs the bounded shared-prefix attention research protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prefix_attention import (  # type: ignore[import-not-found]
        CALIBRATION_CASES,
        EVALUATION_CASES,
        FROZEN_TOLERANCE,
        PATHS,
        Case,
        MissingTileError,
        Problem,
        QuerySchedule,
        TileTable,
        bitwise_equal,
        fp64_oracle,
        make_problem,
        make_schedule,
        output_digest,
        problem_digest,
        quality_report,
        run_path,
    )
else:
    from .prefix_attention import (
        CALIBRATION_CASES,
        EVALUATION_CASES,
        FROZEN_TOLERANCE,
        PATHS,
        Case,
        MissingTileError,
        Problem,
        QuerySchedule,
        TileTable,
        bitwise_equal,
        fp64_oracle,
        make_problem,
        make_schedule,
        output_digest,
        problem_digest,
        quality_report,
        run_path,
    )


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    Path("research/prefix_attention/prefix_attention.py"),
    Path("research/prefix_attention/run_experiment.py"),
)
REPETITIONS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record() -> dict[str, object]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {
        "revision": revision,
        "files": {str(path): _sha256(ROOT / path) for path in SOURCE_FILES},
    }


def _case_record(case: Case) -> dict[str, object]:
    return {
        "name": case.name,
        "seed": case.seed,
        "shape": {
            "tokens": case.tokens,
            "queries": case.queries,
            "heads": case.heads,
            "head_dim": case.head_dim,
            "tile_tokens": case.tile_tokens,
        },
        "shared_prefix": case.shared_prefix,
    }


def _timed(
    function: Callable[[], object], repetitions: int
) -> tuple[object, list[int]]:
    samples = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = function()
        samples.append(time.perf_counter_ns() - start)
    assert result is not None
    return result, samples


def _timing_record(samples: list[int], queries: int) -> dict[str, object]:
    median_ns = statistics.median(samples)
    return {
        "samples_ns": samples,
        "median_ns": median_ns,
        "query_rows_per_second": queries / (median_ns / 1.0e9),
    }


def _path_record(
    problem: Problem,
    table: TileTable,
    schedule: QuerySchedule,
    path: str,
    repetitions: int,
    oracle: tuple[tuple[tuple[float, ...], ...], ...],
) -> dict[str, object]:
    result, samples = _timed(
        lambda: run_path(problem, path, table, schedule), repetitions
    )
    quality = quality_report(result.output, oracle)  # type: ignore[union-attr]
    metrics = asdict(result.metrics)  # type: ignore[union-attr]
    return {
        "timing": _timing_record(samples, problem.case.queries),
        "output_sha256": output_digest(result.output),  # type: ignore[union-attr]
        "quality": quality,
        "metrics": metrics,
    }


def _schedule_record(problem: Problem, table: TileTable) -> dict[str, object]:
    schedule_a = make_schedule(problem.case, "a")
    schedule_b = make_schedule(problem.case, "b")
    fixed_a = run_path(problem, "shared_read_fixed_reduction", table, schedule_a)
    fixed_b = run_path(problem, "shared_read_fixed_reduction", table, schedule_b)
    free_a = run_path(problem, "shared_read_unconstrained", table, schedule_a)
    free_b = run_path(problem, "shared_read_unconstrained", table, schedule_b)
    return {
        "fixed_reduction_bitwise_equal": bitwise_equal(fixed_a.output, fixed_b.output),
        "unconstrained_bitwise_equal": bitwise_equal(free_a.output, free_b.output),
        "schedule_a_groups": schedule_a.groups,
        "schedule_b_groups": schedule_b.groups,
    }


def _tile_record(problem: Problem) -> dict[str, object]:
    table = TileTable.build(problem.case.tokens, problem.case.tile_tokens)
    partial = table.tiles[-1].end - table.tiles[-1].start
    missing_index = 1 if table.tile_count > 1 else 0
    missing_tiles = tuple(tile for tile in table.tiles if tile.index != missing_index)
    missing_table = TileTable(table.token_count, table.tile_tokens, missing_tiles)
    schedule = make_schedule(problem.case, "a")
    missing_rejections = {}
    for path in PATHS:
        try:
            run_path(problem, path, missing_table, schedule)
        except MissingTileError:
            missing_rejections[path] = True
        else:
            missing_rejections[path] = False
    partial_result = run_path(problem, "shared_read_fixed_reduction", table, schedule)
    partial_quality = quality_report(partial_result.output, fp64_oracle(problem))
    return {
        "tile_count": table.tile_count,
        "last_tile_tokens": partial,
        "partial_tile_accepted": partial < problem.case.tile_tokens,
        "partial_tile_quality": partial_quality,
        "missing_tile_index": missing_index,
        "missing_tile_rejected": missing_rejections,
    }


def _case_run(case: Case, repetitions: int) -> dict[str, object]:
    problem = make_problem(case)
    table = TileTable.build(case.tokens, case.tile_tokens)
    schedule = make_schedule(case, "a")
    oracle, oracle_samples = _timed(lambda: fp64_oracle(problem), 1)
    path_records = {
        path: _path_record(problem, table, schedule, path, repetitions, oracle)
        for path in PATHS
    }
    return {
        **_case_record(case),
        "input_sha256": problem_digest(problem),
        "oracle": {
            "kind": "scalar_fp64_cpu",
            "timing": _timing_record(oracle_samples, case.queries),
        },
        "paths": path_records,
        "schedule_sensitivity": _schedule_record(problem, table),
        "tile_checks": _tile_record(problem),
    }


def _all_quality_pass(case_records: list[dict[str, object]]) -> bool:
    for case in case_records:
        paths = case["paths"]
        if not all(record["quality"]["pass"] for record in paths.values()):
            return False
        schedule = case["schedule_sensitivity"]
        if not schedule["fixed_reduction_bitwise_equal"]:
            return False
        checks = case["tile_checks"]
        if not checks["partial_tile_quality"]["pass"]:
            return False
        if not all(checks["missing_tile_rejected"].values()):
            return False
    return True


def _load_calibration(path: Path) -> str:
    body = path.read_bytes()
    record = json.loads(body)
    if record.get("phase") != "calibration":
        raise ValueError("calibration receipt has the wrong phase")
    return hashlib.sha256(body).hexdigest()


def run_phase(phase: str, output: Path, repetitions: int, calibration: Path | None) -> None:
    """Runs one phase and writes a generated JSON receipt."""

    if output.exists():
        raise FileExistsError(f"receipt already exists: {output}")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    calibration_hash = None
    if phase == "evaluation":
        if calibration is None:
            raise ValueError("evaluation requires a calibration receipt")
        calibration_hash = _load_calibration(calibration)
    cases = CALIBRATION_CASES if phase == "calibration" else EVALUATION_CASES
    case_records = [_case_run(case, repetitions) for case in cases]
    receipt = {
        "schema": "prefix-attention-research-v1",
        "phase": phase,
        "status": "pass" if _all_quality_pass(case_records) else "fail",
        "source": _source_record(),
        "protocol": {
            "tolerance": asdict(FROZEN_TOLERANCE),
            "repetitions": repetitions,
            "tile_rule": "absolute token ranges with an exact final partial tile",
            "oracle": "scalar FP64 CPU attention with no tile or group reuse",
        },
        "calibration_receipt_sha256": calibration_hash,
        "cases": case_records,
        "claims": [],
        "limits": [
            "CPU synthetic operator counts are estimates, not DRAM counters.",
            "No GPU run, end-to-end model latency, or cross-backend result is recorded.",
            "The unconstrained shared-read path is schedule-sensitive by construction.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    return parser.parse_args()


def main() -> int:
    """Parses the phase and writes one receipt."""

    arguments = _arguments()
    run_phase(
        arguments.phase,
        arguments.output,
        arguments.repetitions,
        arguments.calibration_receipt,
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
