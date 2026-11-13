"""One-time setup: pre-download the models the pipeline needs.

Run once, manually, before the first pipeline run (or after cloning this
repo fresh):
    .venv/Scripts/python.exe -m setup

Only fetches files into the local Hugging Face cache -- does not load
anything into memory or touch the GPU. Safe to re-run: huggingface_hub
skips files that are already cached.
"""

from huggingface_hub import snapshot_download
from transformers import AutoConfig

from pipeline import CATEGORY_MODEL, MODEL_REVISIONS, NER_MODEL, SENTIMENT_MODEL, SUMMARY_MODEL
from sector_stage import pinned_config

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
