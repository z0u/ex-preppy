#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run --no-sync ruff format "$@" )

echo "✅ Formatted"
