#!/usr/bin/env bash

set -euo pipefail

# Run slow checks
./scripts/check.sh --no-unstaged --typecheck --test
