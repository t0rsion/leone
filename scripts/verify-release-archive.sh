#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
exec python3 "$root/scripts/verify-release-archive.py" "$@"
