#!/bin/bash
# Universal hook dispatcher: runs hooks in .git/hooks/<hook>.d/*
# Works for any git hook

set -e

hook_name="$(basename "$0")"
hook_dir="$(dirname "$0")/${hook_name}.d"

if [ -d "$hook_dir" ]; then
    for hook in "$hook_dir"/*; do
        if [ -x "$hook" ]; then
            (set -x "$hook" "$@" )
        fi
    done
fi
