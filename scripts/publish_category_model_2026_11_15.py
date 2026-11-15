"""Publish the category model the step runs to the Hugging Face Hub, **private** (T-202; owner's word, 2026-11-15).

    uv run python scripts/publish_category_model_2026_11_15.py [--dry-run]
    uv run python scripts/publish_category_model_2026_11_15.py --check

One repository, ``gamug/deberta-v3-news-category-12``: the folder pinned in ``config/category_step.json`` (the step refuses any other) and a model card with the
figures as they are, **including that the bar of D24 is not met**. Before the upload the folder is checked against the pin (identity and the sha256 of each
file); after it the commit SHA is recorded in ``config/published_models.json`` (``category``) and ``published`` of ``config/category_step.json``.
``--check`` is read-only: the recorded revision, that the repository is private, that the card and the weights on the Hub are the recorded and the local ones.
Needs ``HF_TOKEN`` (write scope) in the environment or ``.env``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
REPO = "gamug/deberta-v3-news-category-12"
STEP_PIN = ROOT / "config" / "category_step.json"
RECORD = ROOT / "config" / "published_models.json"
UNAUTHORIZED = 401

CARD = """---
license: mit
base_model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
language:
- en
pipeline_tag: zero-shot-classification
tags:
- finance
- news-classification
- natural-language-inference
- deberta-v3
---
# deberta-v3-news-category-12 (private)

**What it is for.** The category of a short financial-news text about one named company: premise = `<company>: <text>` (the first 2,000 characters), one hypothesis per label
(`The news about <company> is about <definition>.`), the probability of `entailment` is that label's score and the best score is the label. It is the model of step 7 of the project's news
pipeline (`--step category`), one of 12 labels: `earnings_performance`, `mergers_acquisitions`, `leadership_governance`, `legal_regulatory`, `product_innovation`,
`capital_shareholder_returns`, `labor_human_capital`, `market_analyst_sentiment`, `partnerships_business_dev`, `reputational_harm`, `investor_activity`, `other`
(the definitions are those of `docs/category_step.md` in `gamug/portfolio-nlp`; the model reads them, so a different wording is a different model). **Not for:** full articles
(it was trained on short company segments of a few sentences), languages other than English, or any use as investment advice.

## How it was trained
Start: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (revision 8e7e5af5). Fine-tuned as an entailment model on 1,216 (text, company) rows with **one tag each**, the tag a positive pair and
four other labels negatives; 3 epochs, learning rate 2e-5, batch 4 with accumulation 4, max length 512, fp16, seed 0, all rows (no validation slice). The tags are the project owner's, reviewed
afterwards by an assistant to one tag per row (the rules are in the project repository, `docs/category-taxonomy.md`, "One tag per row"); the texts are not published here.

## Figures (label agreement = the top label equals the tag)
| measure | this model | the zero-shot start model |
|---|---|---|
| held-out, 135 single-company rows, the owner's earlier tags, scored **once** | **54.1%** (95% interval 45.9 to 62.3) | 43.7% |
| 300 real sub-news under 1,000 characters, tags by the assistant, blind to both models | **71.3%** (66.0 to 76.3) | 65.7% |
| out of fold, 1,216 training rows, tags reviewed after the first results | 87.8% | 75.5% |

**The bar the project set (80% on the held-out) is NOT met.** The model was pinned as the default by the project owner on 2026-11-15 knowing this (`docs/category_training.md`).

## Limits
- `other` is not learned (3 of 39 right on the real sample). `investor_activity`, `leadership_governance`, `labor_human_capital`, `reputational_harm` and `capital_shareholder_returns` have too few rows to measure.
- The held-out rows are full articles and the training rows short segments: the cross-validation figure does not carry over to articles.
- The held-out tags are of an earlier convention (analyst ratings partly `investor_activity`); about half of that class is predicted as `market_analyst_sentiment`.
- One seed, one partition: the run-to-run variation was not measured.
- Per-class precision, recall and ROC AUC for every measure above are in `docs/category_training.md` of the project repository.
"""


def verify_folder(folder: Path, pin: dict[str, object]) -> list[str]:
    """What differs between the folder and the pin (the manifest and the files present): empty when it is the pinned model."""
    problems: list[str] = []
    manifest = pin["files"]
    assert isinstance(manifest, dict)
    present = {p.name: p for p in folder.iterdir() if p.is_file()}
    for name, digest in manifest.items():
        if name not in present:
            problems.append(f"{name}: missing")
        elif hashlib.sha256(present[name].read_bytes()).hexdigest() != digest:
            problems.append(f"{name}: not the pinned sha256")
    problems += [f"{n}: not in the pin" for n in present if n not in manifest]
    return problems


def record(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"the upload returned no commit SHA ({revision!r}); nothing recorded")
    card_sha = hashlib.sha256(CARD.encode("utf-8")).hexdigest()
    published = json.loads(RECORD.read_text(encoding="utf-8"))
    published["category"] = {
        "repo": REPO,
        "revision": revision,
        "private": True,
        "card_sha256": card_sha,
    }
    RECORD.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    pin = json.loads(STEP_PIN.read_text(encoding="utf-8"))
    pin["published"] = {"repo": REPO, "revision": revision}
    STEP_PIN.write_text(json.dumps(pin, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415
    from huggingface_hub import HfApi  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN (write scope) is needed")
    pin = json.loads(STEP_PIN.read_text(encoding="utf-8"))
    folder = ROOT / pin["model_dir"]
    api = HfApi(token=token)
    if a.check:
        rec = json.loads(RECORD.read_text(encoding="utf-8"))["category"]
        info = api.model_info(rec["repo"], revision=rec["revision"], files_metadata=True)
        weights = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
        print(
            json.dumps(
                {
                    "private": bool(info.private),
                    "recorded revision is the head": api.model_info(rec["repo"]).sha
                    == rec["revision"],
                    "weights are the pinned ones": weights.get("model.safetensors")
                    == pin["files"]["model.safetensors"],
                    "card recorded is the script's": hashlib.sha256(CARD.encode()).hexdigest()
                    == rec["card_sha256"],
                },
                indent=1,
            )
        )
        return
    problems = verify_folder(folder, pin)
    if problems:
        raise SystemExit(f"{folder} is not the pinned model: {problems}")
    if a.dry_run:
        print(
            f"[dry run] {REPO}: {sorted(p.name for p in folder.iterdir())} + README.md ({len(CARD)} characters)"
        )
        return
    api.create_repo(REPO, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in folder.iterdir():
            (stage / item.name).symlink_to(item.resolve())
        (stage / "README.md").write_text(CARD, encoding="utf-8")
        info = api.upload_folder(
            repo_id=REPO, folder_path=str(stage), commit_message="category model of step 7 (T-202)"
        )
    revision = str(getattr(info, "oid", ""))
    print(f"{REPO} (private): {revision}")
    record(revision)


if __name__ == "__main__":
    main()
