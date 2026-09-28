"""Synthetic shared-prefix attention paths with a scalar FP64 oracle."""

from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import Iterable
from dataclasses import dataclass

from common_oracle import f64_sum, oracle_exp, oracle_sqrt


class PrefixAttentionError(ValueError):
    """Identifies an invalid synthetic attention input or schedule."""


class MissingTileError(PrefixAttentionError):
    """Identifies an absent absolute token tile."""


@dataclass(frozen=True)
class Case:
    """Defines one synthetic attention shape and seed."""

    name: str
    tokens: int
    queries: int
    heads: int
    head_dim: int
    tile_tokens: int
    seed: int
    shared_prefix: bool


@dataclass(frozen=True)
class Tolerance:
    """Defines frozen FP32 versus FP64 acceptance limits."""

    name: str
    max_abs: float
    max_rel: float


FROZEN_TOLERANCE = Tolerance("fp32-v1", 2.0e-4, 2.0e-4)
PATHS = (
    "per_row",
    "fixed_tile_per_row",
    "shared_read_unconstrained",
    "shared_read_fixed_reduction",
)


@dataclass(frozen=True)
class TokenTile:
    """Describes one absolute token range."""

    index: int
    start: int
    end: int


@dataclass(frozen=True)
class TileTable:
    """Stores every present absolute token tile in order."""

    token_count: int
    tile_tokens: int
    tiles: tuple[TokenTile, ...]

    @classmethod
    def build(
        cls,
        token_count: int,
        tile_tokens: int,
        missing: Iterable[int] = (),
    ) -> TileTable:
        if token_count <= 0 or tile_tokens <= 0:
            raise PrefixAttentionError("token and tile counts must be positive")
        tile_count = (token_count + tile_tokens - 1) // tile_tokens
        missing_set = frozenset(missing)
        if any(index < 0 or index >= tile_count for index in missing_set):
            raise MissingTileError("missing tile index is outside the token range")
        tiles = tuple(
            TokenTile(index, start, min(start + tile_tokens, token_count))
            for index, start in enumerate(range(0, token_count, tile_tokens))
            if index not in missing_set
        )
        if len(tiles) != tile_count:
            raise MissingTileError("absolute token tile is missing")
        return cls(token_count, tile_tokens, tiles)

    @property
    def indices(self) -> tuple[int, ...]:
        """Returns present tile indices in absolute order."""

        return tuple(tile.index for tile in self.tiles)

    @property
    def tile_count(self) -> int:
        """Returns the required number of absolute tiles."""

        return (self.token_count + self.tile_tokens - 1) // self.tile_tokens

    def tile(self, index: int) -> TokenTile:
        """Returns one tile or rejects an absent tile."""

        for tile in self.tiles:
            if tile.index == index:
                return tile
        raise MissingTileError("absolute token tile is missing")

    def validate_complete(self) -> None:
        """Rejects a table that omits an absolute tile."""

        if len(self.tiles) != self.tile_count or self.indices != tuple(range(self.tile_count)):
            raise MissingTileError("absolute token tile is missing")


@dataclass(frozen=True)
class QuerySchedule:
    """Groups query rows and gives each group a tile order."""

    groups: tuple[tuple[int, ...], ...]
    tile_orders: tuple[tuple[int, ...], ...]

    def validate(self, query_count: int, tile_count: int) -> None:
        """Rejects duplicate, omitted, or partial query and tile schedules."""

        if len(self.groups) != len(self.tile_orders):
            raise PrefixAttentionError("schedule groups and tile orders differ")
        query_ids = tuple(query for group in self.groups for query in group)
        if tuple(sorted(query_ids)) != tuple(range(query_count)):
            raise PrefixAttentionError("schedule does not cover each query once")
        expected_tiles = tuple(range(tile_count))
        if any(tuple(sorted(order)) != expected_tiles for order in self.tile_orders):
            raise PrefixAttentionError("schedule does not cover each tile once")


@dataclass(frozen=True)
class Problem:
    """Contains queries and shared or per-query K/V rows."""

    case: Case
    queries: tuple[tuple[tuple[float, ...], ...], ...]
    shared_keys: tuple[tuple[tuple[float, ...], ...], ...] | None
    shared_values: tuple[tuple[tuple[float, ...], ...], ...] | None
    row_keys: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...] | None
    row_values: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...] | None

    @property
    def shared(self) -> bool:
        """Returns whether every query uses one K/V prefix."""

        return self.case.shared_prefix

    def kv_for(
        self, query: int
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ]:
        """Returns the K/V rows for one query."""

        if self.shared:
            assert self.shared_keys is not None and self.shared_values is not None
            return self.shared_keys, self.shared_values
        assert self.row_keys is not None and self.row_values is not None
        return self.row_keys[query], self.row_values[query]


@dataclass(frozen=True)
class RunMetrics:
    """Records synthetic operator counts and estimated working bytes."""

    kv_elements_read: int
    tile_loads: int
    output_writes: int
    query_results_shared: int
    estimated_kv_cache_bytes: int
    estimated_scratch_bytes: int


@dataclass(frozen=True)
class RunResult:
    """Stores one path output and its generated operator counts."""

    path: str
    output: tuple[tuple[tuple[float, ...], ...], ...]
    metrics: RunMetrics


def f32(value: float) -> float:
    """Rounds one value to IEEE FP32 and returns it as a Python float."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def _random_rows(
    rng: random.Random, rows: int, heads: int, head_dim: int
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(f32(rng.uniform(-1.0, 1.0)) for _ in range(head_dim))
            for _ in range(heads)
        )
        for _ in range(rows)
    )


def make_problem(case: Case) -> Problem:
    """Builds deterministic FP32 inputs from one case seed."""

    rng = random.Random(case.seed)
    queries = _random_rows(rng, case.queries, case.heads, case.head_dim)
    if case.shared_prefix:
        shared_keys = _random_rows(rng, case.tokens, case.heads, case.head_dim)
        shared_values = _random_rows(rng, case.tokens, case.heads, case.head_dim)
        return Problem(case, queries, shared_keys, shared_values, None, None)
    row_keys = tuple(
        _random_rows(rng, case.tokens, case.heads, case.head_dim)
        for _ in range(case.queries)
    )
    row_values = tuple(
        _random_rows(rng, case.tokens, case.heads, case.head_dim)
        for _ in range(case.queries)
    )
    return Problem(case, queries, None, None, row_keys, row_values)


def _hash_values(digest: hashlib._Hash, values: Iterable | float) -> None:
    if isinstance(values, float):
        digest.update(struct.pack("<f", values))
        return
    for value in values:
        _hash_values(digest, value)


def problem_digest(problem: Problem) -> str:
    """Hashes all generated query and K/V input bits."""

    digest = hashlib.sha256()
    digest.update(problem.case.name.encode("utf-8"))
    _hash_values(digest, problem.queries)
    if problem.shared:
        assert problem.shared_keys is not None and problem.shared_values is not None
        _hash_values(digest, problem.shared_keys)
        _hash_values(digest, problem.shared_values)
    else:
        assert problem.row_keys is not None and problem.row_values is not None
        _hash_values(digest, problem.row_keys)
        _hash_values(digest, problem.row_values)
    return digest.hexdigest()


def _chunks(values: tuple[int, ...], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(values[start : start + size] for start in range(0, len(values), size))


def _query_order(query_count: int, variant: str) -> tuple[int, ...]:
    if variant == "a":
        return tuple(range(query_count))
    if variant == "b":
        return tuple(range(0, query_count, 2)) + tuple(range(1, query_count, 2))
    raise PrefixAttentionError("unknown schedule variant")


def _tile_order(tile_count: int, group_index: int, variant: str) -> tuple[int, ...]:
    base = tuple(range(tile_count))
    if variant == "a" and group_index % 2 == 1:
        return tuple(reversed(base))
    if variant == "b":
        shift = group_index % tile_count
        return base[shift:] + base[:shift]
    return base


def make_schedule(case: Case, variant: str) -> QuerySchedule:
    """Builds two deterministic schedules with different grouping."""

    table = TileTable.build(case.tokens, case.tile_tokens)
    ordered_queries = _query_order(case.queries, variant)
    groups = _chunks(ordered_queries, 2)
    orders = tuple(
        _tile_order(table.tile_count, index, variant)
        for index, _group in enumerate(groups)
    )
    schedule = QuerySchedule(groups, orders)
    schedule.validate(case.queries, table.tile_count)
    return schedule


def _dot32(query: tuple[float, ...], key: tuple[float, ...], scale: float) -> float:
    total = 0.0
    for query_value, key_value in zip(query, key):
        total = f32(total + f32(query_value * key_value))
    return f32(total * scale)


def _finish_head(
    scores: list[tuple[int, float]],
    values: tuple[tuple[float, ...], ...],
    head: int,
    head_dim: int,
) -> tuple[float, ...]:
    maximum = max(score for _token, score in scores)
    normalizer = 0.0
    for _token, score in scores:
        normalizer = f32(normalizer + f32(math.exp(float(score - maximum))))
    output = [0.0] * head_dim
    for token, score in scores:
        weight = f32(math.exp(float(score - maximum)) / normalizer)
        value = values[token][head]
        for index, value_part in enumerate(value):
            output[index] = f32(output[index] + f32(weight * value_part))
    return tuple(output)


def _reduce_head_direct(
    query: tuple[float, ...],
    keys: tuple[tuple[tuple[float, ...], ...], ...],
    values: tuple[tuple[tuple[float, ...], ...], ...],
    head: int,
    head_dim: int,
) -> tuple[float, ...]:
    scale = f32(1.0 / math.sqrt(head_dim))
    scores = [
        (token, _dot32(query, keys[token][head], scale))
        for token in range(len(keys))
    ]
    return _finish_head(scores, values, head, head_dim)


def _reduce_head_tiled(
    query: tuple[float, ...],
    cache: dict[int, tuple[TokenTile, tuple, tuple]],
    tile_order: tuple[int, ...],
    head: int,
    head_dim: int,
) -> tuple[float, ...]:
    scale = f32(1.0 / math.sqrt(head_dim))
    scores: list[tuple[int, float]] = []
    for tile_index in tile_order:
        tile, keys, _values = cache[tile_index]
        for offset in range(tile.end - tile.start):
            token = tile.start + offset
            scores.append((token, _dot32(query, keys[offset][head], scale)))
    values = {token: cache[tile_index][2][token - cache[tile_index][0].start]
              for tile_index in tile_order
              for token in range(cache[tile_index][0].start, cache[tile_index][0].end)}
    ordered_values = tuple(values[token] for token in range(len(values)))
    return _finish_head(scores, ordered_values, head, head_dim)


def _reduce_query_direct(
    query: tuple[tuple[float, ...], ...],
    keys: tuple[tuple[tuple[float, ...], ...], ...],
    values: tuple[tuple[tuple[float, ...], ...], ...],
    head_dim: int,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _reduce_head_direct(query[head], keys, values, head, head_dim)
        for head in range(len(query))
    )


def _reduce_query_tiled(
    query: tuple[tuple[float, ...], ...],
    cache: dict[int, tuple[TokenTile, tuple, tuple]],
    tile_order: tuple[int, ...],
    head_dim: int,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _reduce_head_tiled(query[head], cache, tile_order, head, head_dim)
        for head in range(len(query))
    )


def _row_cache(
    keys: tuple[tuple[tuple[float, ...], ...], ...],
    values: tuple[tuple[tuple[float, ...], ...], ...],
    table: TileTable,
    order: tuple[int, ...],
) -> dict[int, tuple[TokenTile, tuple, tuple]]:
    return {
        tile_index: (
            tile := table.tile(tile_index),
            keys[tile.start : tile.end],
            values[tile.start : tile.end],
        )
        for tile_index in order
    }


def _kv_elements(problem: Problem, token_count: int, rows: int = 1) -> int:
    return token_count * problem.case.heads * problem.case.head_dim * 2 * rows


def _scratch_bytes(problem: Problem) -> int:
    score_bytes = problem.case.tokens * 4
    output_bytes = problem.case.queries * problem.case.heads * problem.case.head_dim * 4
    return score_bytes + output_bytes


def _metrics(
    problem: Problem,
    kv_elements: int,
    tile_loads: int,
    cache_elements: int,
) -> RunMetrics:
    return RunMetrics(
        kv_elements,
        tile_loads,
        problem.case.queries * problem.case.heads * problem.case.head_dim,
        0,
        cache_elements * 4,
        _scratch_bytes(problem),
    )


def _run_per_row(problem: Problem, table: TileTable, fixed_tiles: bool) -> RunResult:
    outputs = []
    tile_loads = 0
    for query_index, query in enumerate(problem.queries):
        keys, values = problem.kv_for(query_index)
        if fixed_tiles:
            cache = _row_cache(keys, values, table, table.indices)
            outputs.append(
                _reduce_query_tiled(query, cache, table.indices, problem.case.head_dim)
            )
            tile_loads += table.tile_count
        else:
            outputs.append(
                _reduce_query_direct(query, keys, values, problem.case.head_dim)
            )
    return RunResult(
        "fixed_tile_per_row" if fixed_tiles else "per_row",
        tuple(outputs),
        _metrics(problem, _kv_elements(problem, problem.case.tokens, problem.case.queries),
                 tile_loads, 0),
    )


def _run_shared_fixed(
    problem: Problem, table: TileTable, schedule: QuerySchedule
) -> RunResult:
    if not problem.shared:
        return _run_per_row(problem, table, True)
    keys, values = problem.kv_for(0)
    cache = _row_cache(keys, values, table, table.indices)
    outputs = tuple(
        _reduce_query_tiled(query, cache, table.indices, problem.case.head_dim)
        for query in problem.queries
    )
    cache_elements = _kv_elements(problem, problem.case.tokens)
    return RunResult(
        "shared_read_fixed_reduction",
        outputs,
        _metrics(problem, cache_elements, table.tile_count, cache_elements),
    )


def _run_shared_unconstrained(
    problem: Problem, table: TileTable, schedule: QuerySchedule
) -> RunResult:
    if not problem.shared:
        fallback = _run_per_row(problem, table, True)
        return RunResult(
            "shared_read_unconstrained",
            fallback.output,
            fallback.metrics,
        )
    keys, values = problem.kv_for(0)
    outputs: list[tuple[tuple[float, ...], ...] | None] = [None] * problem.case.queries
    kv_elements = 0
    max_cache_elements = 0
    tile_loads = 0
    for group, tile_order in zip(schedule.groups, schedule.tile_orders):
        cache = _row_cache(keys, values, table, tile_order)
        cache_elements = _kv_elements(problem, problem.case.tokens)
        kv_elements += cache_elements
        max_cache_elements = max(max_cache_elements, cache_elements)
        tile_loads += table.tile_count
        for query_index in group:
            outputs[query_index] = _reduce_query_tiled(
                problem.queries[query_index], cache, tile_order, problem.case.head_dim
            )
    assert all(output is not None for output in outputs)
    return RunResult(
        "shared_read_unconstrained",
        tuple(output for output in outputs if output is not None),
        _metrics(problem, kv_elements, tile_loads, max_cache_elements),
    )


def run_path(
    problem: Problem, path: str, table: TileTable, schedule: QuerySchedule
) -> RunResult:
    """Runs one synthetic path without sharing query-dependent results."""

    table.validate_complete()
    schedule.validate(problem.case.queries, table.tile_count)
    if path == "per_row":
        return _run_per_row(problem, table, False)
    if path == "fixed_tile_per_row":
        return _run_per_row(problem, table, True)
    if path == "shared_read_unconstrained":
        return _run_shared_unconstrained(problem, table, schedule)
    if path == "shared_read_fixed_reduction":
        return _run_shared_fixed(problem, table, schedule)
    raise PrefixAttentionError("unknown attention path")


def fp64_oracle(problem: Problem) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Computes scalar FP64 causal attention without tile or group reuse."""

    outputs = []
    scale = 1.0 / oracle_sqrt(float(problem.case.head_dim))
    for query_index, query in enumerate(problem.queries):
        keys, values = problem.kv_for(query_index)
        query_heads = []
        for head in range(problem.case.heads):
            scores = []
            for token in range(problem.case.tokens):
                score = 0.0
                for index in range(problem.case.head_dim):
                    score += float(query[head][index]) * float(keys[token][head][index])
                scores.append(score * scale)
            maximum = max(scores)
            exponentials = []
            for score in scores:
                exponentials.append(oracle_exp(score - maximum))
            normalizer = f64_sum(exponentials)
            output = []
            for index in range(problem.case.head_dim):
                weighted = 0.0
                for token in range(problem.case.tokens):
                    weighted += (exponentials[token] / normalizer) * float(
                        values[token][head][index]
                    )
                output.append(weighted)
            query_heads.append(tuple(output))
        outputs.append(tuple(query_heads))
    return tuple(outputs)


def quality_report(
    actual: tuple[tuple[tuple[float, ...], ...], ...],
    expected: tuple[tuple[tuple[float, ...], ...], ...],
    tolerance: Tolerance = FROZEN_TOLERANCE,
) -> dict[str, object]:
    """Compares one path with the independent FP64 oracle."""

    errors = [
        abs(actual[q][head][index] - expected[q][head][index])
        for q in range(len(expected))
        for head in range(len(expected[q]))
        for index in range(len(expected[q][head]))
    ]
    relative = [
        error / max(abs(expected[q][head][index]), 1.0e-12)
        for q in range(len(expected))
        for head in range(len(expected[q]))
        for index, error in enumerate(
            [
                abs(actual[q][head][j] - expected[q][head][j])
                for j in range(len(expected[q][head]))
            ]
        )
    ]
    max_abs = max(errors)
    max_rel = max(relative)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "tolerance": tolerance.name,
        "pass": max_abs <= tolerance.max_abs and max_rel <= tolerance.max_rel,
    }


def output_digest(output: tuple[tuple[tuple[float, ...], ...], ...]) -> str:
    """Hashes output FP32 bits for schedule comparisons."""

    digest = hashlib.sha256()
    for query in output:
        for head in query:
            for value in head:
                digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def bitwise_equal(
    left: tuple[tuple[tuple[float, ...], ...], ...],
    right: tuple[tuple[tuple[float, ...], ...], ...],
) -> bool:
    """Checks exact FP32 output bits."""

    return output_digest(left) == output_digest(right)


CALIBRATION_CASES = (
    Case("calibration_short", 7, 3, 2, 5, 4, 1101, True),
    Case("calibration_long", 17, 5, 2, 7, 6, 1102, True),
    Case("calibration_unrelated", 9, 4, 2, 5, 4, 1103, False),
)

EVALUATION_CASES = (
    Case("evaluation_partial_shared", 13, 6, 2, 7, 5, 2201, True),
    Case("evaluation_long_shared", 23, 8, 2, 9, 8, 2202, True),
    Case("evaluation_short_shared", 3, 4, 1, 6, 4, 2203, True),
    Case("evaluation_unrelated", 11, 5, 2, 6, 5, 2204, False),
)
