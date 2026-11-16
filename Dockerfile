# syntax=docker/dockerfile:1
#
# One image definition for both uses of this repo:
#   - target `dev`: what .devcontainer/devcontainer.json builds (VS Code Dev Containers).
#   - target `app`: the FastAPI service (port 8003) / the CLIs, with the locked dependencies baked in.
#
#   docker build --target dev -t portfolio-nlp:dev .
#   docker build --target app -t portfolio-nlp:app .
#   docker run --rm --gpus=all --env-file .env -p 8003:8003 -v "$THESIS_HOST_DIR":/workspaces/thesis portfolio-nlp:app
#
# torch comes from the cu124 wheels (pyproject.toml, [tool.uv.sources]); they bundle the CUDA
# runtime, so the base image does not need to be a CUDA one. Only the host driver and
# `--gpus=all` (NVIDIA Container Toolkit) have to reach the container; without them torch runs on CPU.

# Python 3.12 is what pyproject.toml requires (>=3.12,<3.13); this is the devcontainers image
# (non-root user `vscode`, git, zsh) so the same base serves both targets.
ARG PYTHON_IMAGE=mcr.microsoft.com/devcontainers/python:1-3.12-bookworm
ARG UV_VERSION=0.12.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------- base
FROM ${PYTHON_IMAGE} AS base

# uv: the same binary the lock file was produced with.
COPY --from=uv /uv /uvx /usr/local/bin/

# git: portfolio-common is a git dependency (tag-pinned in pyproject.toml).
# gh: the PR workflow of the SDD process uses it. Both come from the Debian repo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git gh ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# copy, not hardlink: the uv cache and the workspace are on different filesystems in a container.
# UV_PYTHON_DOWNLOADS=never: the image already has the 3.12 that the project asks for.
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

WORKDIR /app

# Dependencies first, so a code change does not reinstall torch (a layer of several GB).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY apps ./apps
COPY cli ./cli
COPY config ./config

# The venv's own python on PATH; `src` is what pytest.ini / the entrypoints already put on the path.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src:/app

# Run as a normal user; the base image already has `vscode`.
RUN chown -R vscode:vscode /app
USER vscode

# Needs DATABASE_URL (and SOURCE_DATABASE_URL for the text stages) pointing at mounted files;
# see docs/db-topology.md. The service binds 0.0.0.0 here (the script's own main() binds 127.0.0.1).
EXPOSE 8003
CMD ["uvicorn", "apps.news_nlp_api:app", "--host", "0.0.0.0", "--port", "8003"]
