#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/scripts"

case "${1:-all}" in
    install)
        shift
        "$SCRIPT_DIR/install.sh" "$@"
        ;;
    auth)
        shift
        "$SCRIPT_DIR/auth.sh" "$@"
        ;;
    format|formatting)
        shift
        "$SCRIPT_DIR/format.sh" "$@"
        ;;
    lint|linting|linters)
        shift
        "$SCRIPT_DIR/lint.sh" "$@"
        ;;
    dead|deadcode)
        shift
        "$SCRIPT_DIR/deadcode.sh" "$@"
        ;;
    type|types)
        shift
        "$SCRIPT_DIR/typecheck.sh" "$@"
        ;;
    test|tests)
        shift
        "$SCRIPT_DIR/test.sh" "$@"
        ;;
    check)
        if [[ $# -gt 1 ]]; then
            shift
            "$SCRIPT_DIR/check.sh" "$@"
        else
            "$SCRIPT_DIR/check.sh" --lint --format --typecheck --test
        fi
        ;;
    build|site)
        shift
        "$SCRIPT_DIR/build_site.py" "$@"
        ;;
    *)
        # Important: here-doc indented with tab characters.
        cat <<-EOF 1>&2
			Usage: $0 {check|lint|format|types|tests|dead|build}
			  install:           install dependencies (uv sync)
			  auth   [...args]:  authenticate with Modal (modal setup)
			  format [...args]:  format code (ruff format)
			  lint   [...args]:  run linters (ruff check)
			  types  [...args]:  check types (basedpyright)
			  tests  [...args]:  run tests (pytest)
			  dead   [...args]:  look for dead code (vulture)
			  check  [...args]:  run checks (default: all checks)
			  build  [...args]:  build static site
			EOF
        exit 1
        ;;
esac
