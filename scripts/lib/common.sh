#!/usr/bin/env bash
log() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_commands() {
    local name
    for name in "$@"; do
        command -v "$name" >/dev/null 2>&1 || die "Missing tool: $name"
    done
}
require_nonempty() { [[ -s $1 ]] || die "Missing or empty: $1"; }
