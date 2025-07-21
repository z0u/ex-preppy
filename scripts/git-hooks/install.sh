#!/bin/bash

# Install git hooks used by this repo

set -e

is_dispatcher() {
    local link
    link=$(readlink "$1" 2>/dev/null || true)
    [[ "$link" == *"scripts/git-hooks/dispatcher.sh"* ]]
}

for hook in pre-commit post-commit; do
    if is_dispatcher ".git/hooks/$hook"; then
        echo "Temporarily removing $hook dispatch hook" > &2
        rm ".git/hooks/$hook"
    fi
done

# Initialize git LFS hooks for this repository (see .gitattributes)
( set -x git lfs install )

# Move LFS hooks into dispatch directories
mkdir -p .git/hooks/pre-commit.d .git/hooks/pre-push.d

echo "Moving LFS hooks into dispatch directories" >&2
mv .git/hooks/pre-commit .git/hooks/pre-commit.d/50-lfs
mv .git/hooks/pre-push .git/hooks/pre-commit.d/50-lfs

for hook in pre-commit post-commit; do
    echo "Installing $hook dispatch hook" >&2
    ln -rsf scripts/git-hooks/dispatch.sh -t ".git/hooks/$hook"
    echo "Installing project $hook hooks" >&2
    ln -rsf "scripts/git-hooks/$hook.d"/* -t ".git/hooks/$hook.d/"
done

echo "Hooks installed" >&2
