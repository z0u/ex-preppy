#!/usr/bin/env bash

set -euo pipefail

has_unstaged_changes() {
    # Check for unstaged changes to tracked files
    ! git diff --quiet ||
    # Check for untracked files (excluding gitignored)
    [ -n "$(git ls-files --others --exclude-standard)" ]
}

stash_unstaged() {
    git stash push --keep-index --include-untracked -m "pre-commit-temp" --quiet
}

stash_pop() {
    git stash pop --quiet
}

show_usage() {
    # Important: heredoc indented with tab characters.
    cat <<-EOF 1>&2
	Usage: $0 [opts]
	  --lint:            run linters
	  --format:          run formatters
	  --typecheck:       run type checkers
	  --test:            run tests

	  --no-unstaged:     stash unstaged changes before checks, and pop after
	  -h --help:         show help and exit

	Only checks are run (doesn't change files).
	EOF
}


o_no_unstaged=false
o_lint=false
o_format=false
o_typecheck=false
o_test=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-unstaged)
      o_no_unstaged=true
      ;;
    --lint)
      o_lint=true
      ;;
    --format)
      o_format=true
      ;;
    --typecheck)
      o_typecheck=true
      ;;
    --test)
      o_test=true
      ;;
    --help|-h)
      show_usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      show_usage
      exit 1
      ;;
  esac
  shift
done

if [ "$o_no_unstaged" = "true" ] && has_unstaged_changes; then
    stash_unstaged
    trap stash_pop EXIT
fi

if [ "$o_lint" = "true" ]; then
    ( set -x; uv run ruff check --no-fix )
fi

if [ "$o_format" = "true" ]; then
    ( set -x; uv run ruff format --check )
fi

if [ "$o_typecheck" = "true" ]; then
    ( set -x; uv run basedpyright )
fi

if [ "$o_test" = "true" ]; then
    ( set -x; uv run pytest )
fi
