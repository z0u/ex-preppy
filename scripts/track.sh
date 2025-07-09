#!/usr/bin/env bash

set -euo pipefail

mode=serve
processed_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--deploy|--detach)
            mode=deploy
            shift # consume the argument
            ;;
        --)
            shift # consume '--'
            processed_args+=("$@") # add everything after '--'
            break
            ;;
        *)
            processed_args+=("$1") # preserve the argument
            shift # consume the argument
            ;;
    esac
done

# a little shell magic to restore the positional parameters
set -- "${processed_args[@]}"

( set -x; uv run modal "$mode" -m track.aim "$@" )
