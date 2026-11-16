# scripts/init.sh
#!/bin/bash
uv sync --frozen --group dev
uv run pre-commit install --install-hooks
source .venv/bin/activate
curl -fsSL https://claude.ai/install.sh | bash
# git, gh and uv come from the Dockerfile (target `dev`)
