#!/usr/bin/env bash

set -euo pipefail

# Absolute path to repo root (used to anchor tool configs)
REPO_ROOT=$(git rev-parse --show-toplevel)

(
  set -x;
  export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  uv run --no-sync pytest "$@"
)

echo "✅ Tests passed"
