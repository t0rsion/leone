"""Generate digests for the causal production prefill comparator."""

from __future__ import annotations

import struct
from functools import lru_cache
from typing import Any

from common_oracle import (
    FNV_OFFSET,
    SplitMix64,
    f16,
    f16_bits,
    f64_sum,
    oracle_exp,
    oracle_sqrt,
    update,
)

CASE_FIELDS = (
    "tokens", "query_rows", "query_heads", "kv_heads", "head_dim",
    "tile_tokens", "group_rows", "seed", "shared_prefix",
)


def input_digest(queries: list[float], keys: list[int], values: list[int]) -> str:
    hash_value = FNV_OFFSET
    for value in queries:
        hash_value = update(hash_value, struct.unpack("<I", struct.pack("<f", value))[0], 4)
    for value in keys:
        hash_value = update(hash_value, value, 2)
    for value in values:
        hash_value = update(hash_value, value, 2)
    return f"{hash_value:016x}"


def oracle_digest(values: list[float]) -> str:
    hash_value = FNV_OFFSET
    for value in values:
        hash_value = update(hash_value, struct.unpack("<Q", struct.pack("<d", value))[0], 8)
    return f"{hash_value:016x}"


def make_inputs(case: dict[str, Any]) -> tuple[list[float], list[int], list[int]]:
    tokens = case["tokens"]
    query_rows = case["query_rows"]
    query_heads = case["query_heads"]
    kv_heads = case["kv_heads"]
    head_dim = case["head_dim"]
    generator = SplitMix64(case["seed"])
    queries = [generator.unit_float() for _ in range(query_rows * query_heads * head_dim)]
    keys = []
    values = []
    for _ in range(kv_heads * tokens * head_dim):
        keys.append(f16_bits(generator.unit_float()))
        values.append(f16_bits(generator.unit_float()))
    return queries, keys, values


def oracle_values(
    case: dict[str, Any], queries: list[float], keys: list[int], values: list[int]
) -> list[float]:
    tokens = case["tokens"]
    query_rows = case["query_rows"]
    query_heads = case["query_heads"]
    kv_heads = case["kv_heads"]
    head_dim = case["head_dim"]
    outputs = [0.0] * (query_rows * query_heads * head_dim)
    query_stride = query_heads * head_dim
    kv_stride = tokens * head_dim
    start_position = tokens - query_rows
    for row in range(query_rows):
        context_length = start_position + row + 1
        for query_head in range(query_heads):
            kv_head = query_head * kv_heads // query_heads
            query_base = row * query_stride + query_head * head_dim
            kv_base = kv_head * kv_stride
            scores = []
            for token in range(context_length):
                key_base = kv_base + token * head_dim
                score = 0.0
                for index in range(head_dim):
                    score += f16(f16_bits(queries[query_base + index])) * f16(keys[key_base + index])
                scores.append(score / oracle_sqrt(float(head_dim)))
            maximum = max(scores)
            weights = []
            for score in scores:
                weights.append(oracle_exp(score - maximum))
            normalizer = f64_sum(weights)
            for index in range(head_dim):
                weighted = 0.0
                for token in range(context_length):
                    weighted += weights[token] * f16(
                        values[kv_base + token * head_dim + index]
                    )
                outputs[query_base + index] = weighted / normalizer
    return outputs


@lru_cache(maxsize=64)
def _cached_case_oracle_values(values: tuple[Any, ...]) -> tuple[float, ...]:
    case = dict(zip(CASE_FIELDS, values))
    queries, keys, values = make_inputs(case)
    return tuple(oracle_values(case, queries, keys, values))


def case_oracle_values(case: dict[str, Any]) -> list[float]:
    key = tuple(case[field] for field in CASE_FIELDS)
    return list(_cached_case_oracle_values(key))


def max_absolute_error(actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        raise ValueError("output and oracle lengths differ")
    return max(
        (abs(float(value) - expected[index]) for index, value in enumerate(actual)),
        default=0.0,
    )


def max_relative_error(actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        raise ValueError("output and oracle lengths differ")
    return max(
        (
            abs(float(value) - expected[index])
            / max(abs(expected[index]), 1.0e-12)
            for index, value in enumerate(actual)
        ),
        default=0.0,
    )


def canonicalize_case(case: dict[str, Any]) -> None:
    expected = case_oracle_values(case["spec"])
    for field in ("quality_max_abs", "quality_max_rel"):
        case[f"backend_{field}"] = case[field]
    case["quality_max_abs"] = max_absolute_error(case["output_values"], expected)
    case["quality_max_rel"] = max_relative_error(case["output_values"], expected)


def case_digests(case: dict[str, Any]) -> tuple[str, str]:
    queries, keys, values = make_inputs(case)
    outputs = oracle_values(case, queries, keys, values)
    return input_digest(queries, keys, values), oracle_digest(outputs)
