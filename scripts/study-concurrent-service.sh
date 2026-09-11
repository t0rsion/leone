#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
exec python3 "$root/scripts/study-concurrent-service.py" "$@"
