# syntax=docker/dockerfile:1
#
# One image definition for both uses of this repo:
#   - target `dev`: what .devcontainer/devcontainer.json builds (VS Code Dev Containers).
#   - target `app`: the FastAPI service (port 8003) and the batch CLIs that read only src/, apps/,
#     cli/ and config/ (news_nlp_cli.py, news_nlp_steps.py), with the locked dependencies baked in.
#     Training, evaluation and the one-shot scripts (they read docs/, gold/, models/) are dev-only.
#
#   docker build --target dev -t portfolio-nlp:dev .
#   docker build --target app -t portfolio-nlp:app .
#
#   # the service (GPU optional; drop --gpus=all to run on CPU)
#   docker run --rm --gpus=all --env-file .env -p 8003:8003 \
#       -v "$THESIS_HOST_DIR":/workspaces/thesis -v hf-cache:/models/hf portfolio-nlp:app
#   # a batch CLI, same mounts
#   docker run --rm --gpus=all --env-file .env -v "$THESIS_HOST_DIR":/workspaces/thesis \
#       -v hf-cache:/models/hf portfolio-nlp:app python cli/news_nlp_steps.py --step scope --status
#   # is the GPU visible?
#   docker run --rm --gpus=all portfolio-nlp:app python -c "import torch; print(torch.cuda.is_available())"
#
# The files DATABASE_URL (and SOURCE_DATABASE_URL for the text stages) name must exist inside
# the container, hence the thesis mount; `.env` uses the in-container paths (docs/db-topology.md).
# `.env` is never in the image (.dockerignore is an allow-list): pass it with --env-file.
# `--env-file` does not strip quotes: write KEY=value, no quotes, no `export`.
#
# torch comes from the cu124 wheels (pyproject.toml, [tool.uv.sources]); they bundle the CUDA
# runtime, so the base image does not need to be a CUDA one. Only the host driver and
# `--gpus=all` (NVIDIA Container Toolkit) have to reach the container; without them torch runs
# on CPU (`pipeline.py` DEVICE), silently: use the check above.
#
# VERSIONS THAT MUST AGREE: Python (`.python-version`, `requires-python` in pyproject.toml and
# the PYTHON_IMAGE below, checked by the build) and uv (the lock file's; UV_VERSION below).

# Python 3.12 is what pyproject.toml requires (>=3.12,<3.13). The devcontainers image has the
# non-root user `vscode` (uid 1000), git and zsh. Pinned by digest of the multi-arch index
# (tag 1-3.12-bookworm on 2026-11-16); to refresh: `docker buildx imagetools inspect <tag>`.
ARG PYTHON_IMAGE=mcr.microsoft.com/devcontainers/python:1-3.12-bookworm@sha256:7876580dc67fd460fd962f004cbeb480027e9bbc0657096f1087db11f9eaff39
ARG UV_VERSION=0.12.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------- base
FROM ${PYTHON_IMAGE} AS base

# uv: the same binary the lock file was produced with.
COPY --from=uv /uv /uvx /usr/local/bin/

# git: portfolio-common is a git dependency (tag-pinned in pyproject.toml).
# gh: the PR workflow of the SDD process uses it. curl: scripts/init.sh (the Claude installer).
# All three come from the Debian repo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git gh ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# UV_PYTHON_DOWNLOADS=never: the image's interpreter is the one the project asks for. If you bump
# the Python version, change PYTHON_IMAGE too: this check fails the build otherwise, instead of
# `uv sync` failing later inside the devcontainer's postCreateCommand.
RUN python --version | grep -q "^Python 3\.12\." \
    || (echo "PYTHON_IMAGE is not Python 3.12: align it with .python-version / requires-python" >&2; exit 1)

# copy, not hardlink: the uv cache and the workspace are on different filesystems in a container.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- dev (devcontainer)
FROM base AS dev

# The workspace is bind-mounted by the Dev Containers extension, so nothing of the repo is
# copied here: `scripts/init.sh` (postCreateCommand) runs `uv sync --frozen --group dev`
# into the workspace's own .venv and installs the pre-commit hooks.
USER vscode
WORKDIR /workspaces/portfolio-nlp

# ---------------------------------------------------------------- app (service / CLIs)
FROM base AS app

# Everything below runs as `vscode`, so the venv is written once with its owner (a recursive
# chown afterwards would store the torch wheels, several GB, a second time in a new layer).
# /models/hf is the Hugging Face cache: mount a volume there to keep the models between containers.
RUN mkdir -p /app /models/hf && chown vscode:vscode /app /models/hf
USER vscode
WORKDIR /app

# Dependencies first, so a code change does not reinstall torch.
COPY --chown=vscode:vscode pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/vscode/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=vscode:vscode src ./src
COPY --chown=vscode:vscode apps ./apps
COPY --chown=vscode:vscode cli ./cli
COPY --chown=vscode:vscode config ./config

# The venv's own python on PATH; `src` is what pytest.ini / the entrypoints already put on the path.
# HF_HOME: the .env.example's HF_HOME (passed with --env-file) overrides this default.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src:/app \
    HF_HOME=/models/hf

# The service binds 0.0.0.0 here (the script's own main() binds 127.0.0.1). The start period
# covers the import of torch and transformers.
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=4)"
CMD ["uvicorn", "apps.news_nlp_api:app", "--host", "0.0.0.0", "--port", "8003"]
