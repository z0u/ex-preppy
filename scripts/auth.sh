#!/usr/bin/env bash

set -euo pipefail

(
    set -x
    uv run --no-sync modal setup
    uv run --no-sync wandb login
)

echo "✅ Authenticated"
