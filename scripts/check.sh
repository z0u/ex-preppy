#!/usr/bin/env bash

set -euo pipefail

# Absolute path to repo root (used to anchor tool configs)
REPO_ROOT=$(git rev-parse --show-toplevel)

has_unstaged_changes() {
  # Check for unstaged changes to tracked files
  ! git diff --quiet ||
  # Check for untracked files (excluding gitignored)
  [ -n "$(git ls-files --others --exclude-standard)" ]
}

# Create a temporary snapshot directory and populate it per mode.
# Modes:
#   index -> current index (includes staged changes)
#   head  -> current HEAD commit (ignores staged/unstaged)
create_snapshot() {
  local mode=${1:-index}
  SNAPSHOT_DIR=$(mktemp -d -t repo-snapshot-XXXXXX)
  trap 'rm -rf "$SNAPSHOT_DIR"' EXIT
  if [[ "$mode" == "index" ]]; then
    git checkout-index --all --prefix="$SNAPSHOT_DIR/"
    echo "Working on project snapshot from index in $SNAPSHOT_DIR" >&2
  elif [[ "$mode" == "head" ]]; then
    git -c core.worktree="$REPO_ROOT" archive --format=tar HEAD \
      | tar -x -C "$SNAPSHOT_DIR"
    echo "Working on project snapshot from HEAD in $SNAPSHOT_DIR" >&2
  else
    echo "Unknown snapshot mode: $mode" >&2
    exit 2
  fi
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

	  --index:           run selected checks against current index (includes staged)
	  --head:            run selected checks against HEAD only (ignores staged)
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
  # Accept optional path targets
  if [[ -n "${SNAPSHOT_DIR:-}" ]]; then
  # Run from repo root so uv uses the project env; scan the snapshot path
    (
      set -x
      uv run --no-sync ruff check --quiet --no-fix \
        --cache-dir="$REPO_ROOT/.ruff_cache" \
        --config "$SNAPSHOT_DIR/pyproject.toml" \
        "$SNAPSHOT_DIR"
    )
  else
    ( set -x; uv run --no-sync ruff check --quiet --no-fix "$@" )
  fi
}

run_format() {
  # Accept optional path targets
  if [[ -n "${SNAPSHOT_DIR:-}" ]]; then
    # Run from repo root so uv uses the project env; scan the snapshot path
    (
      set -x
      uv run --no-sync ruff format \
        --quiet --check \
        --cache-dir="$REPO_ROOT/.ruff_cache" \
        --config "$SNAPSHOT_DIR/pyproject.toml" \
        "$SNAPSHOT_DIR"
    )
  else
    ( set -x; uv run --no-sync ruff format --quiet --check "$@" )
  fi
}

run_typecheck() {
  if [[ -n "${SNAPSHOT_DIR:-}" ]]; then
    # Analyze the snapshot as the project to reflect what will be pushed
    (
      set -x
      # Use the main venv directory to avoid reinstalling packages
      uv run --project "$REPO_ROOT" --no-sync basedpyright \
        --project "$SNAPSHOT_DIR" \
        --venvpath "$REPO_ROOT"
    )
  else
    ( set -x; cd "$REPO_ROOT"; uv run --no-sync basedpyright )
  fi
}

run_tests() {
  if [[ -n "${SNAPSHOT_DIR:-}" ]]; then
    # Run tests collected from the snapshot, importing code from the snapshot
    (
      set -x
      # Add snapshot's src directory to Python path so imports work
      PYTHONPATH="$SNAPSHOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --no-sync pytest --quiet \
        --rootdir="$SNAPSHOT_DIR" \
        --config-file "$SNAPSHOT_DIR/pyproject.toml" \
        "$SNAPSHOT_DIR/tests"
    )
  else
    (
      set -x
      cd "$REPO_ROOT"
      PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --no-sync pytest --quiet
    )
  fi
}


o_no_unstaged=false
o_snapshot_mode=""
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
    --index)
      o_snapshot_mode="index"
      ;;
    --head)
      o_snapshot_mode="head"
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

# If requested, run all checks against a clean snapshot of the index.
if [ "$o_no_unstaged" = "true" ] && [ -z "$o_snapshot_mode" ]; then
  o_snapshot_mode="index"
fi

# If a snapshot mode is chosen, all selected checks will operate on that snapshot.
if [ -n "$o_snapshot_mode" ]; then
  create_snapshot "$o_snapshot_mode"
  # Stay in repo root for venv and uv context. Commands will have to change the environment themselves.
  cd "$REPO_ROOT"
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
