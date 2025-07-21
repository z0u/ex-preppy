#!/usr/bin/env bash

set -euo pipefail

# Just run the fast checks
./scripts/check.sh --no-unstaged --lint --format
