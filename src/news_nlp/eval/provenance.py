"""Best-effort git provenance for an eval run (logged to MLflow + ``eval_run``)."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def code_version() -> str:
    """``git rev-parse --short HEAD`` for this checkout, ``"unknown"`` if that
    fails (no git, detached tree, git not on PATH)."""
    # Fixed argv, no shell, no user input -- S603 is a false positive here.
    cmd = ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"
