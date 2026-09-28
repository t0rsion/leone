"""Shared provenance and receipt publication helpers for prefix runners."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

LEGACY_SOURCE_LAYOUT = "legacy"
SOURCE_LAYOUT = "runner_support_v1"


def public_command(command: list[str], roles: dict[str, str]) -> list[str]:
    """Replace private command paths with stable public roles."""
    public = []
    for argument in command:
        if argument in roles:
            public.append(roles[argument])
        elif Path(argument).is_absolute():
            raise ValueError("unmapped private command path")
        else:
            public.append(argument)
    return public


def publish_json(output: Path, value: object) -> None:
    """Publish one complete JSON file without replacing an existing file."""
    payload = json.dumps(value, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        descriptor_to_close: int | None = descriptor
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            descriptor_to_close = None
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_name, output)
    except FileExistsError as error:
        raise FileExistsError("refusing to replace existing receipt") from error
    finally:
        if descriptor_to_close is not None:
            os.close(descriptor_to_close)
        Path(temporary_name).unlink(missing_ok=True)
