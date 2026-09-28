#!/usr/bin/env python3
"""Check branch complexity in the Swift and Metal research harnesses."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent
MAXIMUM = 10
FUNCTION = re.compile(r"\bfunc\s+([A-Za-z_]\w*)\s*\([^)]*\)")
METAL_FUNCTION = re.compile(
    r"\b(?:kernel|inline)\s+(?:void|float|uint)\s+([A-Za-z_]\w*)\s*\([^)]*\)"
)
DECISIONS = re.compile(r"\b(?:if|for|while|guard|catch|case)\b|&&|\|\|")
TERNARY = re.compile(r"(?<!\?)\?(?!\?|=)")
TOKEN = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def mask_token(match: re.Match[str]) -> str:
    return "".join("\n" if character == "\n" else " " for character in match.group())


def strip_comments_and_strings(source: str) -> str:
    return TOKEN.sub(mask_token, source)


def function_body(source: str, start: int, end: int) -> str:
    opening = source.find("{", end)
    if opening < 0:
        raise ValueError("function has no body")
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise ValueError("function body is not closed")


def functions(path: Path) -> list[tuple[str, int, int]]:
    source = strip_comments_and_strings(path.read_text())
    pattern = METAL_FUNCTION if path.suffix == ".metal" else FUNCTION
    return [(match.group(1), match.start(), match.end()) for match in pattern.finditer(source)]


def complexity(path: Path, name: str, start: int, end: int) -> int:
    source = strip_comments_and_strings(path.read_text())
    body = function_body(source, start, end)
    branch_count = len(DECISIONS.findall(body))
    ternaries = len(TERNARY.findall(body))
    return 1 + branch_count + ternaries


def main() -> int:
    paths = (
        ROOT / "metal_fixed_reduction/RunPrefixAttention.swift",
        ROOT / "metal_fixed_reduction/PrefixAttention.metal",
    )
    violations = []
    for path in paths:
        for name, start, end in functions(path):
            value = complexity(path, name, start, end)
            line = path.read_text()[:start].count("\n") + 1
            print(f"{path.relative_to(ROOT)}:{line}: {name} complexity={value}")
            if value > MAXIMUM:
                violations.append(f"{path}:{line}: {name} complexity={value}")
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"Swift and Metal complexity gate passed (maximum={MAXIMUM})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
