#!/bin/sh
set -eu
prefix=${PREFIX:-"$HOME/.local"}
root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
install -d "$prefix/bin"
install -m 0755 "$root/bin/leone" "$prefix/bin/leone"
if [ -d "$root/plans" ]; then
    install -d "$prefix/share/leone/plans"
    for plan in "$root"/plans/*.json; do
        [ -f "$plan" ] || continue
        install -m 0644 "$plan" "$prefix/share/leone/plans/"
    done
fi
if [ -d "$root/receipts" ]; then
    install -d "$prefix/share/leone/receipts"
    for receipt in "$root"/receipts/*.json; do
        [ -f "$receipt" ] || continue
        install -m 0644 "$receipt" "$prefix/share/leone/receipts/"
    done
fi
printf 'installed %s\n' "$prefix/bin/leone"
