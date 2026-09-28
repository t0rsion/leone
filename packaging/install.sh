#!/bin/sh
set -eu
prefix=${PREFIX:-"$HOME/.local"}
root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

target_marker=$root/archive-target
if [ ! -f "$target_marker" ] || [ -L "$target_marker" ]; then
    echo "error: archive target marker is missing" >&2
    exit 1
fi
archive_target=$(cat "$target_marker") || {
    echo "error: archive target marker cannot be read" >&2
    exit 1
}
case "$archive_target" in
    x86_64-unknown-linux-gnu)
        archive_platform=linux-x86_64
        ;;
    aarch64-apple-darwin)
        archive_platform=darwin-arm64
        ;;
    *)
        echo "error: archive target marker is invalid" >&2
        exit 1
        ;;
esac
if ! printf '%s\n' "$archive_target" | cmp -s - "$target_marker"; then
    echo "error: archive target marker is malformed" >&2
    exit 1
fi

host_os=$(uname -s) || {
    echo "error: cannot determine host operating system" >&2
    exit 1
}
host_arch=$(uname -m) || {
    echo "error: cannot determine host architecture" >&2
    exit 1
}
case "$host_os:$host_arch" in
    Linux:x86_64) host_target=x86_64-unknown-linux-gnu ;;
    Darwin:arm64) host_target=aarch64-apple-darwin ;;
    *)
        echo "error: unsupported host $host_os/$host_arch; archives support Linux x86_64 and Darwin arm64" >&2
        exit 1
        ;;
esac
if [ "$archive_target" != "$host_target" ]; then
    echo "error: archive $archive_platform does not match host $host_os/$host_arch" >&2
    exit 1
fi

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
if [ -f "$root/package-info.json" ] || [ -f "$root/environment.json" ]; then
    install -d "$prefix/share/leone"
    for metadata in package-info.json environment.json; do
        if [ -f "$root/$metadata" ]; then
            install -m 0644 "$root/$metadata" "$prefix/share/leone/$metadata"
        fi
    done
fi
if [ -f "$root/tools/checksum.sh" ]; then
    install -d "$prefix/share/leone/tools"
    install -m 0755 "$root/tools/checksum.sh" "$prefix/share/leone/tools/checksum.sh"
fi
printf 'installed %s\n' "$prefix/bin/leone"
