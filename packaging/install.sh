#!/bin/sh
set -eu
prefix=${PREFIX:-"$HOME/.local"}
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
install -d "$prefix/bin"
install -m 0755 "$root/bin/leone" "$prefix/bin/leone"
printf 'installed %s\n' "$prefix/bin/leone"
