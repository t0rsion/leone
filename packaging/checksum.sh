#!/bin/sh
set -eu

if command -v sha256sum >/dev/null 2>&1; then
    exec sha256sum "$@"
fi
if command -v shasum >/dev/null 2>&1; then
    exec shasum -a 256 "$@"
fi
echo "error: a SHA-256 tool is required (sha256sum or shasum)" >&2
exit 1
