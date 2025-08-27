#!/usr/bin/env bash

set -euo pipefail

# Just run the fast checks against staged + clean index
./scripts/check.sh --index --lint --format
