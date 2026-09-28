"""Tests the bounded shared-prefix attention protocol."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prefix_attention import (
    EVALUATION_CASES,
    FROZEN_TOLERANCE,
    PATHS,
    MissingTileError,
    TileTable,
    bitwise_equal,
    fp64_oracle,
    make_problem,
    make_schedule,
    quality_report,
    run_path,
)


class PrefixAttentionTests(unittest.TestCase):
    """Checks numerical, schedule, and tile invariants."""

    def test_all_paths_match_fp64_oracle(self) -> None:
        for case in EVALUATION_CASES:
            problem = make_problem(case)
            table = TileTable.build(case.tokens, case.tile_tokens)
            oracle = fp64_oracle(problem)
            schedule = make_schedule(case, "a")
            for path in PATHS:
                result = run_path(problem, path, table, schedule)
                self.assertTrue(
                    quality_report(result.output, oracle)["pass"],
                    f"{case.name} {path} exceeded {FROZEN_TOLERANCE.name}",
                )

    def test_fixed_reduction_is_bitwise_schedule_invariant(self) -> None:
        changed = False
        for case in EVALUATION_CASES:
            problem = make_problem(case)
            table = TileTable.build(case.tokens, case.tile_tokens)
            first = run_path(
                problem,
                "shared_read_fixed_reduction",
                table,
                make_schedule(case, "a"),
            )
            second = run_path(
                problem,
                "shared_read_fixed_reduction",
                table,
                make_schedule(case, "b"),
            )
            self.assertTrue(bitwise_equal(first.output, second.output))
            unconstrained_first = run_path(
                problem,
                "shared_read_unconstrained",
                table,
                make_schedule(case, "a"),
            )
            unconstrained_second = run_path(
                problem,
                "shared_read_unconstrained",
                table,
                make_schedule(case, "b"),
            )
            changed = changed or not bitwise_equal(
                unconstrained_first.output, unconstrained_second.output
            )
        self.assertTrue(changed)

    def test_partial_tiles_are_processed_and_missing_tiles_rejected(self) -> None:
        for case in EVALUATION_CASES:
            problem = make_problem(case)
            table = TileTable.build(case.tokens, case.tile_tokens)
            self.assertLessEqual(table.tiles[-1].end - table.tiles[-1].start, case.tile_tokens)
            result = run_path(
                problem,
                "shared_read_fixed_reduction",
                table,
                make_schedule(case, "a"),
            )
            self.assertTrue(quality_report(result.output, fp64_oracle(problem))["pass"])
            missing_index = 1 if table.tile_count > 1 else 0
            missing_table = TileTable(
                table.token_count,
                table.tile_tokens,
                tuple(tile for tile in table.tiles if tile.index != missing_index),
            )
            for path in PATHS:
                with self.assertRaises(MissingTileError):
                    run_path(problem, path, missing_table, make_schedule(case, "a"))

    def test_shared_reads_reduce_only_kv_traffic(self) -> None:
        for case in EVALUATION_CASES:
            problem = make_problem(case)
            table = TileTable.build(case.tokens, case.tile_tokens)
            schedule = make_schedule(case, "a")
            per_row = run_path(problem, "fixed_tile_per_row", table, schedule)
            shared = run_path(
                problem, "shared_read_fixed_reduction", table, schedule
            )
            self.assertEqual(per_row.metrics.output_writes, shared.metrics.output_writes)
            self.assertEqual(shared.metrics.query_results_shared, 0)
            if case.shared_prefix:
                self.assertLess(
                    shared.metrics.kv_elements_read,
                    per_row.metrics.kv_elements_read,
                )
            else:
                self.assertEqual(
                    shared.metrics.kv_elements_read,
                    per_row.metrics.kv_elements_read,
                )


if __name__ == "__main__":
    unittest.main()
