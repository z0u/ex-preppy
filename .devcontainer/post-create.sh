#!/usr/bin/env bash

set -euo pipefail

(
    set -x

    # Make the volume mounts writable. Even though the uv cache is a subdirectory, the parent is created by Docker as root, so we need to change the owner of that too.
    sudo chown -R "$USER:$USER" ~/.local
    sudo chown -R "$USER:$USER" ~/.cache
    sudo chown -R "$USER:$USER" .venv

    # Initialize Python environment.
    uv venv --allow-existing < /dev/null
    ./go install --device=cpu < /dev/null

    if ! grep -Fq "direnv hook" "~/.bashrc"; then
        echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
    fi
    if ! grep -Fq "direnv hook" "~/.zshrc"; then
        echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
    fi
    direnv allow

    # Workflow hooks
    ./scripts/git-hooks/install.sh
)

echo "Virtual environment created. You may need to restart the Python language server."
