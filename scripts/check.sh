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
  # Important: here-doc indented with tab characters.
  cat <<-EOF 1>&2
	Usage: $0 [opts]
	  --fix:             attempt to fix linting and formatting errors
	  --lint:            run linters
	  --format:          run formatters
	  --typecheck:       run type checkers
	  --test:            run tests

	  --no-unstaged:     stash unstaged changes before checks, and pop after
	  -h --help:         show help and exit

	Only checks are run (doesn't change files).
	EOF
}

run_fix() {
  (
    set -x
    uv run --no-sync ruff check --fix
    uv run --no-sync ruff format
  )
}

run_lint() {
  ( set -x; uv run --no-sync ruff check --quiet --no-fix )
}

run_format() {
  ( set -x; uv run --no-sync ruff format --quiet --check )
}

run_typecheck() {
  ( set -x; uv run --no-sync basedpyright )
}

run_tests() {
  ( set -x; uv run --no-sync pytest --quiet )
}


o_no_unstaged=false
o_lint=false
o_format=false
o_typecheck=false
o_test=false
o_fix=false

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
    --fix)
      o_fix=true
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

declare -A pids
declare -A results
declare -A hints

hints[fix]="Fix remaining linting errors manually"
hints[lint]="Try './go lint --fix' or './go check --fix'"
hints[format]="Try './go format' or './go check --fix'"
hints[typecheck]="Fix type errors manually"
hints[test]="Fix test failures manually"

if [ "$o_fix" = "true" ]; then
  run_fix & pids[fix]=$!
fi

if [ "$o_lint" = "true" ]; then
  run_lint & pids[lint]=$!
fi

if [ "$o_format" = "true" ]; then
  run_format & pids[format]=$!
fi

if [ "$o_typecheck" = "true" ]; then
  run_typecheck & pids[typecheck]=$!
fi

if [ "$o_test" = "true" ]; then
  run_tests & pids[test]=$!
fi

final_exit_code=0
for task in "${!pids[@]}"; do
  if wait "${pids[$task]}"; then
    results[$task]="✅ success"
  else
    results[$task]="❌ failure"
    final_exit_code=1
  fi
done

max_len=0
# for task in lint format typecheck test; do
for task in "${!results[@]}"; do
  if ((${#task} > max_len)); then
    max_len=${#task}
  fi
done
((max_len++))

for task in "${!results[@]}"; do
  # Pad task name to max_len + 1 for spacing
  printf "%+${max_len}s %s" "$task:" "${results[$task]}"
  if [[ "${results[$task]}" == "❌ failure" ]]; then
    printf ". %s\n" "${hints[$task]}"
  else
    echo # newline
  fi
done

exit $final_exit_code
