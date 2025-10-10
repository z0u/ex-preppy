#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run --no-sync ty check "$@" )

echo "✅ Type check passed"
