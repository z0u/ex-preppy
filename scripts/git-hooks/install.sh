#!/usr/bin/env bash

# Install git hooks used by this repo

set -euo pipefail

HOOKS_SRC_DIR=scripts/git-hooks
DISPATCHER="$HOOKS_SRC_DIR/dispatcher.sh"

is_dispatcher() {
    local link
    link=$(readlink "$1" 2>/dev/null || true)
    [[ "$link" == *"$DISPATCHER"* ]]
}

for hook in pre-commit pre-push; do
    if is_dispatcher ".git/hooks/$hook"; then
        echo "Temporarily removing $hook dispatch hook" >&2
        ( set -x; rm ".git/hooks/$hook" )
    fi
done

echo "Installing Git LFS hooks" >&2
( set -x; git lfs install )

echo "Moving LFS hooks into dispatch directories" >&2
( set -x; mkdir -p .git/hooks/pre-commit.d .git/hooks/pre-push.d )
( set -x; mv .git/hooks/pre-push .git/hooks/pre-push.d/50-lfs )

for hook in pre-commit pre-push; do
    echo "Installing $hook dispatch hook" >&2
    ( set -x; ln -rsf "$DISPATCHER" ".git/hooks/$hook" )
    ( set -x; ln -rsf "$HOOKS_SRC_DIR/$hook.d"/* -t ".git/hooks/$hook.d/" )
done

echo "Hooks installed" >&2
