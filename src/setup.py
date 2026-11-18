"""One-time setup: pre-download the models the pipeline needs.

Run once, manually, before the first pipeline run (or after cloning this
repo fresh):
    .venv/Scripts/python.exe -m setup

Only fetches files into the local Hugging Face cache -- does not load
anything into memory or touch the GPU. Safe to re-run: huggingface_hub
skips files that are already cached.
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig

from sector_stage import pinned_config

# The pins of the four stage models (``config/pipeline_models.json``, which step 7 also reads). The v1 module that used to load them is gone (T-220).
_PINS = json.loads(
    (Path(__file__).resolve().parent.parent / "config" / "pipeline_models.json").read_text(
        encoding="utf-8"
    )
)
SENTIMENT_MODEL: str = _PINS["sentiment"]["model"]
NER_MODEL: str = _PINS["ner"]["model"]
CATEGORY_MODEL: str = _PINS["category"]["model"]
SUMMARY_MODEL: str = _PINS["c_summary"]["model"]
MODEL_REVISIONS: dict[str, str] = {
    _PINS[k]["model"]: _PINS[k]["revision"] for k in ("sentiment", "ner", "category", "c_summary")
}

MODELS = (SENTIMENT_MODEL, NER_MODEL, CATEGORY_MODEL, SUMMARY_MODEL)


def models_to_fetch() -> list[tuple[str, str]]:
    """``(repo, revision)`` of the four v1 stages and of the embedder of the sector step (``config/sector_step.json``, T-167)."""
    embedder = pinned_config()["embedder"]
    return [(m, MODEL_REVISIONS[m]) for m in MODELS] + [(embedder["model"], embedder["revision"])]


def download_models() -> None:
    for repo_id, revision in models_to_fetch():
        print(f"Fetching {repo_id} @ {revision}...")
        local_path = snapshot_download(repo_id, revision=revision)
        AutoConfig.from_pretrained(repo_id, revision=revision)
        print(f"OK: {repo_id} cached at {local_path}")
    print("Setup complete.")


if __name__ == "__main__":
    download_models()
