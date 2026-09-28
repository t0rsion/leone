"""Generate the shared host input and FP64 oracle digests."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

FNV_OFFSET = 1_469_598_103_934_665_603
FNV_PRIME = 1_099_511_628_211
PROVENANCE_SCOPE = "wrapper-observed-structural-v1"
ORACLE_ID = "scalar-fp64-attention-v2"
ORACLE_ALGORITHM = "explicit-f64-left-to-right-taylor-transcendentals-v1"
EXP_LOG2 = float.fromhex("0x1.71547652b82fep+0")
EXP_LN2 = float.fromhex("0x1.62e42fefa39efp-1")
EXP_TERMS = 40
EXP_MIN_SOFTMAX_DELTA = -16.0
EXP_MAX_SOFTMAX_DELTA = 0.0
SQRT_MIN_HEAD_DIM = 1
SQRT_MAX_HEAD_DIM = 64
SQRT_CASES = {
    2.0: float.fromhex("0x1.6a09e667f3bcdp+0"),
    16.0: 4.0,
    32.0: float.fromhex("0x1.6a09e667f3bcdp+2"),
    64.0: 8.0,
}
CASE_FIELDS = (
    "tokens", "query_rows", "query_heads", "kv_heads", "head_dim",
    "tile_tokens", "group_rows", "seed", "shared_prefix",
)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f32(value)))[0]


def f16(value: int) -> float:
    return struct.unpack("<e", struct.pack("<H", value))[0]


def f64_sum(values: Iterable[float]) -> float:
    """Accumulate binary64 values in one declared order."""
    total = 0.0
    for value in values:
        total += float(value)
    return total


def oracle_exp(value: float) -> float:
    """Evaluate a bounded softmax exponential with fixed binary64 arithmetic."""
    if (not math.isfinite(value) or value < EXP_MIN_SOFTMAX_DELTA
            or value > EXP_MAX_SOFTMAX_DELTA):
        raise ValueError("oracle softmax delta is outside [-16, 0]")
    exponent = int(value * EXP_LOG2 + (0.5 if value >= 0.0 else -0.5))
    reduced = value - exponent * EXP_LN2
    term = 1.0
    result = 1.0
    for denominator in range(1, EXP_TERMS + 1):
        term = term * reduced / denominator
        result += term
    return result * float.fromhex(f"0x1.0p{exponent}")


def oracle_sqrt(value: float) -> float:
    """Evaluate a bounded integer head dimension with fixed binary64 arithmetic."""
    if (not math.isfinite(value) or value != int(value)
            or not SQRT_MIN_HEAD_DIM <= value <= SQRT_MAX_HEAD_DIM):
        raise ValueError("oracle head dimension must be an integer in [1, 64]")
    if value in SQRT_CASES:
        return SQRT_CASES[value]
    result = value
    for _ in range(80):
        next_result = 0.5 * (result + value / result)
        if next_result == result:
            return result
        result = next_result
    return result


class SplitMix64:
    def __init__(self, state: int) -> None:
        self.state = state

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)

    def unit_float(self) -> float:
        bits = self.next() >> 40
        value = f32(float(bits) / f32(16_777_215.0))
        value = f32(value * f32(2.0))
        return f32(value - f32(1.0))


def update(hash_value: int, value: int, width: int) -> int:
    for byte in value.to_bytes(width, "little"):
        hash_value = ((hash_value ^ byte) * FNV_PRIME) & ((1 << 64) - 1)
    return hash_value


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
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        hash_value = update(hash_value, bits, 8)
    return f"{hash_value:016x}"


def output_digest(values: list[float]) -> str:
    hash_value = FNV_OFFSET
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        hash_value = update(hash_value, bits, 4)
    return f"{hash_value:016x}"


def make_inputs(case: dict[str, Any]) -> tuple[list[float], list[int], list[int]]:
    tokens = case["tokens"]
    query_rows = case["query_rows"]
    query_heads = case["query_heads"]
    kv_heads = case["kv_heads"]
    head_dim = case["head_dim"]
    generator = SplitMix64(case["seed"])
    queries = [generator.unit_float() for _ in range(query_rows * query_heads * head_dim)]
    key_rows = 1 if case["shared_prefix"] else query_rows
    keys = []
    values = []
    for _ in range(key_rows * kv_heads * tokens * head_dim):
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
    kv_stride = kv_heads * tokens * head_dim
    for row in range(query_rows):
        for query_head in range(query_heads):
            kv_head = query_head * kv_heads // query_heads
            query_base = row * query_stride + query_head * head_dim
            kv_base = (0 if case["shared_prefix"] else row * kv_stride) + kv_head * tokens * head_dim
            scores = []
            for token in range(tokens):
                key_base = kv_base + token * head_dim
                score = 0.0
                for index in range(head_dim):
                    score += float(queries[query_base + index]) * f16(keys[key_base + index])
                scores.append(score / oracle_sqrt(float(head_dim)))
            maximum = max(scores)
            weights = []
            for score in scores:
                weights.append(oracle_exp(score - maximum))
            normalizer = f64_sum(weights)
            for index in range(head_dim):
                weighted = 0.0
                for token in range(tokens):
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


def canonicalize_path(path: dict[str, Any], expected: list[float]) -> None:
    for field in (
        "quality_max_abs",
        "quality_max_rel",
        "schedule_b_quality_max_abs",
        "schedule_b_quality_max_rel",
    ):
        path[f"backend_{field}"] = path[field]
    output = path["output_values"]
    schedule_output = path["schedule_b_output_values"]
    path["quality_max_abs"] = max_absolute_error(output, expected)
    path["quality_max_rel"] = max_relative_error(output, expected)
    path["schedule_b_quality_max_abs"] = max_absolute_error(schedule_output, expected)
    path["schedule_b_quality_max_rel"] = max_relative_error(schedule_output, expected)


def canonicalize_receipt(receipt: dict[str, Any]) -> None:
    for case in receipt["cases"]:
        expected = case_oracle_values(case["spec"])
        for path in case["paths"]:
            canonicalize_path(path, expected)


def case_digests(case: dict[str, Any]) -> tuple[str, str]:
    queries, keys, values = make_inputs(case)
    return input_digest(queries, keys, values), oracle_digest(oracle_values(case, queries, keys, values))


LOGIT_ORACLE_PAIRS = {
    "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785": {
        "subject": "Qwen3-8B-Q4_K_M.gguf",
        "oracle": "Qwen3-8B-BF16.gguf",
        "oracle_sha256": "5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30",
        "oracle_dtype": "bf16",
        "oracle_ftype": 32,
        "architecture": "qwen3",
        "tokenizer": "gpt2",
    },
}
GGUF_MAGIC = b"GGUF"
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}
GGUF_MAX_KEYS = 1 << 16
GGUF_MAX_STRING = 1 << 26
GGUF_MAX_ARRAY = 1 << 24
# Independently read metadata digests, keyed by subject sha256. An entry holds
# `architecture_sha256`, `tokenizer_sha256`, and the exact `tokenizer_keys` list. The
# model file hash proves the bytes. These pins let a validator that never opens the
# model check an identity record against a second reading of the same bytes. The
# values come from the pinned llama.cpp gguf-py `GGUFReader`, not from `gguf_identity`.
# It read both files with the same digests, the same keys, and equal field types.
# Its scalar and array type table is not pinned here: the file hash already fixes it.
LOGIT_ORACLE_METADATA: dict[str, dict[str, Any]] = {
    "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785": {
        "architecture_sha256": "e2ebd3f2b391661085e670a4c74d40d5294c67baf185487f55f9459ad808ad0c",
        "tokenizer_sha256": "2f6c052316e2e5b0b413e26a75d09aad1fa73d113a9878702d3dbe6f63a42c26",
        "tokenizer_keys": [
            "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id",
        ],
    },
}


def gguf_read(source: Any, size: int) -> bytes:
    if size > GGUF_MAX_STRING:
        raise ValueError("GGUF metadata length exceeds the reader bound")
    data = source.read(size)
    if len(data) != size:
        raise ValueError("GGUF metadata ends early")
    return data


def gguf_string(source: Any, digest: Any) -> str:
    """Read one length-prefixed string. Invalid UTF-8 is an error, never replaced."""
    head = gguf_read(source, 8)
    raw = gguf_read(source, struct.unpack("<Q", head)[0])
    digest.update(head + raw)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("GGUF string is not valid UTF-8") from error


def gguf_scalar(source: Any, kind: int, digest: Any) -> Any:
    raw = gguf_read(source, struct.calcsize(GGUF_SCALARS[kind]))
    if kind == 7 and raw not in (b"\x00", b"\x01"):
        raise ValueError("GGUF bool byte is neither 0 nor 1")
    digest.update(raw)
    return struct.unpack(GGUF_SCALARS[kind], raw)[0]


def gguf_value(source: Any, kind: int, digest: Any) -> Any:
    """Read one metadata value and feed its bytes to `digest`. Arrays return their length.

    The digest covers string length and text, scalar bytes, and an array's element
    type and count with its elements. It does not cover the type code of a
    top-level value. An array holds strings or scalars. A nested array is unsupported.
    """
    if kind == GGUF_STRING:
        return gguf_string(source, digest)
    if kind in GGUF_SCALARS:
        return gguf_scalar(source, kind, digest)
    if kind != GGUF_ARRAY:
        raise ValueError(f"unsupported GGUF value type {kind}")
    head = gguf_read(source, 12)
    element, count = struct.unpack("<IQ", head)
    if element != GGUF_STRING and element not in GGUF_SCALARS:
        raise ValueError(f"unsupported GGUF array element type {element}")
    if count > GGUF_MAX_ARRAY:
        raise ValueError("GGUF array length exceeds the reader bound")
    digest.update(head)
    for _ in range(count):
        gguf_value(source, element, digest)
    return count


def gguf_metadata(path: Any) -> dict[str, tuple[str, Any]]:
    """Return each metadata key with the sha256 of its value bytes and its scalar value.

    A duplicate key, a malformed length, a bad bool byte, and a string that is not
    UTF-8 raise ValueError.
    """
    with open(path, "rb") as source:
        if gguf_read(source, 4) != GGUF_MAGIC:
            raise ValueError("file is not GGUF")
        version, _, count = struct.unpack("<IQQ", gguf_read(source, 20))
        if version != 3 or count > GGUF_MAX_KEYS:
            raise ValueError("unsupported GGUF version or key count")
        entries: dict[str, tuple[str, Any]] = {}
        for _ in range(count):
            key = gguf_string(source, hashlib.sha256())
            if key in entries:
                raise ValueError(f"GGUF repeats the metadata key {key}")
            kind = struct.unpack("<I", gguf_read(source, 4))[0]
            digest = hashlib.sha256()
            value = gguf_value(source, kind, digest)
            entries[key] = (digest.hexdigest(), value)
        return entries


GGUF_IDENTITY_FIELDS = ("architecture", "tokenizer", "architecture_sha256", "tokenizer_keys", "tokenizer_sha256")
GGUF_TOKENIZER_KEYS = (
    "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id",
)


def gguf_digest(entries: dict[str, tuple[str, Any]], keys: list[str]) -> str:
    text = "".join(f"{key}={entries[key][0]}\n" for key in sorted(keys))
    return hashlib.sha256(text.encode()).hexdigest()


def gguf_identity(path: Any) -> dict[str, Any]:
    """Hash the architecture and tokenizer metadata that decide layer shape and token meaning.

    Each digest is the sha256 of the sorted lines `key=sha256(value bytes)`, so it
    fixes the key set. The architecture digest covers `general.architecture` and every
    key that starts with the architecture name and a dot. The tokenizer digest covers
    the keys in `GGUF_TOKENIZER_KEYS` that the file holds. The value hash omits the
    type code of a top-level scalar, so `u32` 5 and `i32` 5 hash alike. This scheme is
    the one recorded in existing identity records and stays fixed. The model file hash
    proves the exact bytes, types included.

    Beginning-of-sequence and padding ids are excluded. A workload feeds fixed token
    ids and adds no beginning token, so those ids do not change any logit.
    """
    entries = gguf_metadata(path)
    architecture = entries["general.architecture"][1]
    tokenizer_keys = [key for key in GGUF_TOKENIZER_KEYS if key in entries]
    if "tokenizer.ggml.tokens" not in tokenizer_keys:
        raise ValueError("GGUF file lacks the tokenizer vocabulary")
    architecture_keys = [key for key in entries if key == "general.architecture" or key.startswith(f"{architecture}.")]
    return {
        "architecture": architecture,
        "tokenizer": entries["tokenizer.ggml.model"][1],
        "architecture_sha256": gguf_digest(entries, architecture_keys),
        "tokenizer_keys": tokenizer_keys,
        "tokenizer_sha256": gguf_digest(entries, tokenizer_keys),
    }


def calibration_slice(slices: dict[str, dict[str, int]], name: str, offset: int, count: int) -> str:
    slices[name] = {"offset": offset, "count": count}
    return name


def calibration_prompt(spec: dict[str, Any], branch: int, slices: dict[str, dict[str, int]]) -> list[str]:
    name = spec["name"]
    tail = calibration_slice(slices, f"{name}_tail_{branch}", spec["tail_offsets"][branch], spec["tail_tokens"][branch])
    if spec["topology"] == "fork_of_fork":
        return [
            calibration_slice(slices, f"{name}_prefix", spec["prefix_offsets"][0], spec["partial_prefix_tokens"]),
            calibration_slice(slices, f"{name}_common_tail", spec["common_tail_offset"], spec["common_tail_tokens"]),
            tail,
        ]
    prefix = calibration_slice(slices, f"{name}_prefix_{branch}", spec["prefix_offsets"][branch], spec["prefix_tokens"][branch])
    return [prefix, tail]


def calibration_case(spec: dict[str, Any], slices: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Map one runtime calibration case to comparator events.

    Each branch prefills its logical prompt on its own sequence. Then one stepwise
    decode feeds the branch's teacher tokens and captures the manifest steps. The
    logits depend only on the token content, so the mapping does not copy sequences.
    """
    name, branches = spec["name"], range(len(spec["prefix_offsets"]))
    prompts = {branch: calibration_prompt(spec, branch, slices) for branch in branches}
    lengths = {branch: sum(slices[item]["count"] for item in prompts[branch]) for branch in branches}
    teachers = {
        branch: [
            calibration_slice(slices, f"{name}_teacher_{branch}_{step}", spec["teacher_offsets"][branch][step], 1)
            for step in range(spec["decode_steps"])
        ]
        for branch in branches
    }
    return {
        "name": name,
        "class": f"runtime_calibration_{spec['topology']}",
        "operations": [
            {"kind": "decode", "name": "prefill", "streams": [
                {"seq": branch, "position": 0, "tokens": prompts[branch], "outputs": "none"} for branch in branches]},
            {"kind": "decode_stepwise", "name": "decode_stepwise", "streams": [
                {"seq": branch, "position": lengths[branch], "tokens": teachers[branch]} for branch in branches],
             "capture_steps": spec["capture_steps"]},
        ],
    }


def calibration_workload(runtime_manifest: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Build the comparator workload for the runtime calibration cases.

    The model contract, token fixture, and context come from the source workload
    unchanged, so the oracle sees the same fixture and llama.cpp settings.
    """
    slices: dict[str, dict[str, int]] = {}
    cases = [calibration_case(spec, slices) for spec in runtime_manifest["calibration"]]
    return {
        "schema": source["schema"],
        "phase": "runtime_calibration",
        "model_contract": source["model_contract"],
        "token_fixture": source["token_fixture"],
        "context": source["context"],
        "slices": slices,
        "cases": cases,
    }


def workload_bytes(workload: dict[str, Any]) -> bytes:
    return (json.dumps(workload, indent=2) + "\n").encode()
