#!/usr/bin/env bash

set -euo pipefail

# Run slow checks on a snapshot of HEAD only
./scripts/check.sh --head --typecheck --test
