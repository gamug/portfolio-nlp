#!/bin/bash
# Runs once, as postCreateCommand of .devcontainer/devcontainer.json.
set -euo pipefail

# git, gh and uv come from the Dockerfile (target `dev`). Fail loudly on an old cached image.
for tool in git gh uv; do
    command -v "$tool" >/dev/null || { echo "init.sh: $tool is missing: rebuild the container (Dev Containers: Rebuild Container)" >&2; exit 1; }
done

uv sync --frozen --group dev
uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
curl -fsSL https://claude.ai/install.sh | bash
