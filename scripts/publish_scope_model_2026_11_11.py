"""Publish the scope model of step 1 to the Hugging Face Hub, **public**, and pin it (T-147b; owner's word, 2026-11-11).

    uv run python scripts/publish_scope_model_2026_11_11.py [--model-dir models/scope_final] [--dry-run]
    uv run python scripts/publish_scope_model_2026_11_11.py --check [--model-dir models/scope_final]

One repository, ``gamug/deberta-v3-news-scope-nli``: ``stage_a/`` and ``stage_b/`` (the two fine-tuned NLI models), the
``training_record.json`` that names the data, the recipe and the hypotheses, and a model card. The owner asked for the upload
"to trace it for future reference and usage" and then for the repository to be **public** (2026-11-11). The files are uploaded
and verified first (the weights must be on the Hub); only then is the repository made public. After the upload the commit SHA is
written to ``config/published_models.json`` (key ``scope``) and to ``config/scope_model.json``, the **pin**: the repository, the
commit and the step revision (``scope-nli-<12 hex>``, the hash stored in every ``body_scope`` row) it corresponds to. ``--check``
is read-only: whether the repository is public and served to an anonymous request, the card, the weights and the small files on
the Hub are the recorded and local ones, and the local folder has the pinned step revision. Needs ``HF_TOKEN`` (write scope) in the environment or ``.env``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scope_stage import revision_of

OK = 200  # what an anonymous request to a public repository gets
REPO = "gamug/deberta-v3-news-scope-nli"
BASE = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0@8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"
STAGES = ("stage_a", "stage_b")
SMALL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
WHAT = (
    "Hub repositories of the final models (T-169, T-147b). scope: public, pinned (config/scope_model.json); "
    "sentiment and relevance: private, not pinned (config/pipeline_models.json)"
)

#: The held-out 200 scored from the saved weights (``scripts/score_scope_heldout_2026_11_11.py``), filled from its output.
HELD_OUT = {
    "weighted_agreement": "83.6% [78.1, 88.9]",
    "rules": "67.2%",
    "gain": "+16.5 [+9.6, +23.4] points",
    "single_recall": "92.2%",
    "out_of_scope_recall": "53.0% [32.3, 73.8]",
}


def card(step_revision: str) -> str:
    return f"""---
license: mit
base_model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
language:
- en
pipeline_tag: zero-shot-classification
tags:
- finance
- news
- natural-language-inference
- deberta-v3
---
# deberta-v3-news-scope-nli

**What it is for.** Step 1 of the news pipeline of `gamug/portfolio-nlp` (PLAN D39/D40): the **scope of a news body** from its
title and its first 800 characters, in four classes: `single` (one company is the subject), `multi` (several companies are
cut apart later), `sector` (about a sector, not a company) and `out_of_scope` (neither; the pipeline stops there).

**Two models, one repository.** Both are `{BASE}` fine-tuned as an NLI (entailment margin of a hypothesis):
- `stage_a/`: company / sector / not_related, for every body.
- `stage_b/`: single / multi, asked only of the bodies stage A calls `company`.
The hypotheses, the recipe and the data counts are in `training_record.json`. The step revision stored with every result
(`body_scope.classifier_revision`) is `{step_revision}`: the hash of that record and of the weight files.

## How it was trained
365 bodies tagged and reviewed by the project owner (284 + 51 chosen for the rare scopes + 30 whose title names a company
but whose body is not about it): stage A on all, stage B on the 210 company bodies. Title + first 800 characters, 3 epochs,
learning rate 2e-5, batch 16, float32 weights with bf16 autocast, seed 1. The texts are in the project repository
(`gold/`), not here. Recipe and the story of the choices: `docs/scope_training.md`.

## Figures
Held-out 200 bodies the model never trained on, **weighted to the corpus** by the draw's strata, scored from these weights
through the step's own classifier (`scripts/score_scope_heldout_2026_11_11.py`): agreement
**{HELD_OUT["weighted_agreement"]}** (rules triage {HELD_OUT["rules"]}; paired gain {HELD_OUT["gain"]}); `single` recall
{HELD_OUT["single_recall"]}, **`out_of_scope` recall {HELD_OUT["out_of_scope_recall"]}**. Three fits of this recipe (seed 1) scored 83.0%, 83.6% and 83.6%; this is one seed and not a noise estimate.

## Limits (read before use)
- **`out_of_scope` is the weak class.** In the held-out 200, about half of the out-of-scope bodies are read as `single` or
  `multi` and continue in the pipeline; the owner accepted this on purpose (a body wrongly kept costs a review, a body wrongly
  stopped is lost). **How many bodies continue in the corpus is not known.** An estimate from a 1,000-body sample said about
  6.7% of all bodies, but over the 309,972 unique bodies of the corpus the model calls 49% `out_of_scope`: 11% of the `ok`
  bodies and 59% of the `quarantined` ones (those pairs are 79% of the corpus). A read of 50 of the latter found most not about
  a company, a few with a company hook and two clear errors. The held-out gold may not represent the quarantined bodies.
- The 90% bar of the project (D20/D33) is **not** met; the owner accepted the model below it (PLAN D40 amendment).
- `sector` has 7 gold bodies in the held-out 200: its recall is reported, not gated.
- The figure is optimistic: several design choices were made while looking at the same 200 bodies (docs, section 6). One seed.
- English news, 2022 to 2026, title plus 800 characters; bodies without a title are read from the text alone. Not tested on
  other sources or on bodies written in other languages.

## Use
No token is needed. The function below is the whole classifier.
```python
import json, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

root = snapshot_download("{REPO}")
rec = json.load(open(f"{{root}}/training_record.json"))
tok = AutoTokenizer.from_pretrained(f"{{root}}/stage_a")
A = AutoModelForSequenceClassification.from_pretrained(f"{{root}}/stage_a").eval()
B = AutoModelForSequenceClassification.from_pretrained(f"{{root}}/stage_b").eval()

def margins(model, text, stage):
    out = {{}}
    for name, hypothesis in rec[stage]["hypotheses"].items():
        x = tok(text, hypothesis, truncation="only_first", max_length=rec[stage]["recipe"]["max_length"], return_tensors="pt")
        with torch.no_grad():
            logits = model(**x).logits[0].float()
        out[name] = (logits[0] - logits[1]).item()   # entailment minus not-entailment
    return out

def scope(title, body):
    text = f"{{title}}. {{body[:rec['fragment_chars']]}}"
    a = margins(A, text, "stage_a")
    topic = max(rec["stage_a"]["topics"], key=a.get)
    if topic != "company":
        return "sector" if topic == "sector" else "out_of_scope"
    b = margins(B, text, "stage_b")
    return max(rec["stage_b"]["classes"], key=b.get)   # "single" or "multi"

print(scope("Apple shares fall", "Apple Inc. said on Tuesday that iPhone sales were weaker than expected."))
```
The same logic, batched and on the GPU, is `src/scope_stage.py` of the project repository (`gamug/portfolio-nlp`); the example above does not need it.

## History
The first upload (commit `5f5b2c27`) did not contain the weights because of a script error; use the latest commit, the one
the project's `config/scope_model.json` names.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def record_upload(
    published: Path,
    pin: Path,
    revision: str,
    text: str,
    step_revision: str,
    weights: dict[str, str],
) -> None:
    """Write the commit SHA (the pin) and the sha256 of the card and of both weight files. Stops on a non-SHA answer."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"the upload returned no commit SHA ({revision!r}); nothing recorded")
    current = json.loads(published.read_text(encoding="utf-8")) if published.exists() else {}
    current["what"] = WHAT
    current["scope"] = {
        "repo": REPO,
        "revision": revision,
        "private": False,
        "card_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "weights_sha256": weights,
        "step_revision": step_revision,
    }
    published.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    pin.write_text(
        json.dumps(
            {
                "what": "the scope model step 1 runs on (T-147b); the pin of scripts/publish_scope_model_2026_11_11.py",
                "model": REPO,
                "revision": revision,
                "step_revision": step_revision,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def publish(token: str, model_dir: Path, text: str, dry: bool) -> str:
    import tempfile  # noqa: PLC0415 -- only the upload needs these

    from huggingface_hub import HfApi  # noqa: PLC0415

    if dry:
        print(f"[dry run] {REPO}: {model_dir} + README.md ({len(text)} characters)")
        return "dry-run"
    api = HfApi(token=token)
    api.create_repo(REPO, private=True, exist_ok=True)  # public only after the upload is verified
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "training_record.json").symlink_to((model_dir / "training_record.json").resolve())
        for name in STAGES:  # file by file: upload_folder does not follow a symlinked folder
            (stage / name).mkdir()
            for item in (model_dir / name).iterdir():
                (stage / name / item.name).symlink_to(item.resolve())
        (stage / "README.md").write_text(text, encoding="utf-8")
        info = api.upload_folder(
            repo_id=REPO, folder_path=str(stage), commit_message="scope model of step 1 (T-147b)"
        )
    sha = str(getattr(info, "oid", ""))
    on_hub = set(api.list_repo_files(REPO, revision=sha or None))
    missing = {f"{s}/model.safetensors" for s in STAGES} - on_hub
    if missing:  # never record a pin for an upload that lost the weights
        raise SystemExit(
            f"{REPO}@{sha}: the weights are not on the Hub ({sorted(missing)}); nothing recorded"
        )
    api.update_repo_settings(REPO, private=False)
    print(f"{REPO} (public): {sha}")
    return sha


def check_published(
    published: Path,
    model_dir: Path,
    token: str,
    api: Any = None,
    download: Any = None,
    anonymous_status: Any = None,
) -> dict[str, bool | None]:
    """Read-only yes/no per check; ``None`` = could not be compared. The three callables are injectable for tests."""
    import urllib.error  # noqa: PLC0415 -- only the check needs these
    import urllib.request  # noqa: PLC0415

    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    api = api or HfApi(token=token or None)
    download = download or (
        lambda repo, name, rev: hf_hub_download(repo, name, revision=rev, token=token or None)
    )

    def status(url: str) -> int:
        try:
            urllib.request.urlopen(url)  # noqa: S310 -- fixed https URL; an anonymous request must fail
        except urllib.error.HTTPError as err:
            return int(err.code)
        except urllib.error.URLError:
            return 0
        return 200

    anonymous_status = anonymous_status or status
    recorded = json.loads(published.read_text(encoding="utf-8"))["scope"]
    rev = recorded["revision"]
    info = api.model_info(REPO, revision=rev, files_metadata=True)
    on_hub = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    answer = anonymous_status(f"https://huggingface.co/{REPO}/resolve/{rev}/README.md")
    live_card = Path(download(REPO, "README.md", rev)).read_bytes()
    results: dict[str, bool | None] = {
        "public": not info.private,
        "recorded revision is the head of the repository": api.model_info(REPO).sha == rev,
        "anonymous request served": None if answer == 0 else answer == OK,
        "card on the Hub is the recorded one": hashlib.sha256(live_card).hexdigest()
        == recorded["card_sha256"],
    }
    for stage in STAGES:
        local = model_dir / stage / "model.safetensors"
        results[f"{stage} weights on the Hub are the recorded ones"] = (
            on_hub.get(f"{stage}/model.safetensors") == recorded["weights_sha256"][stage]
        )
        results[f"{stage} weights are the local ones"] = (
            sha256_file(local) == recorded["weights_sha256"][stage] if local.exists() else None
        )
    for stage in (
        STAGES
    ):  # the step revision hashes the record and the weights; the small files are compared here
        for name in SMALL_FILES:
            local_file = model_dir / stage / name
            results[f"{stage}/{name} on the Hub is the local one"] = (
                Path(download(REPO, f"{stage}/{name}", rev)).read_bytes() == local_file.read_bytes()
                if local_file.exists()
                else None
            )
    results["local folder has the pinned step revision"] = (
        revision_of(model_dir) == recorded["step_revision"]
        if (model_dir / "training_record.json").exists()
        else None
    )
    return results


def main() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=Path("models/scope_final"))
    parser.add_argument("--record", type=Path, default=Path("config/published_models.json"))
    parser.add_argument("--pin", type=Path, default=Path("config/scope_model.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="read-only check of the recorded revision"
    )
    args = parser.parse_args()
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "")
    if args.check:
        try:
            results = check_published(args.record, args.model_dir, token)
        except OSError as err:
            raise SystemExit(f"could not reach the Hub: {err}") from err
        for name, ok in results.items():
            print(f"{name}: {'not compared' if ok is None else 'yes' if ok else 'NO'}")
        raise SystemExit(0 if all(ok is not False for ok in results.values()) else 1)
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN (write scope) is not set")
    if "__" in "".join(HELD_OUT.values()):
        raise SystemExit(
            "the held-out figures of the card are not filled in: run score_scope_heldout first"
        )
    step_revision = revision_of(args.model_dir)
    text = card(step_revision)
    revision = publish(token, args.model_dir, text, args.dry_run)
    if not args.dry_run:
        weights = {s: sha256_file(args.model_dir / s / "model.safetensors") for s in STAGES}
        record_upload(args.record, args.pin, revision, text, step_revision, weights)
    print(json.dumps({"repo": REPO, "revision": revision, "step_revision": step_revision}))


if __name__ == "__main__":
    main()
