"""Publish the attribution model of the cut (step 1.2) to the Hugging Face Hub, **private** (owner's word, 2026-11-14; T-148b12 closed, T-194 tests it).

    uv run python scripts/publish_attribution_model_2026_11_14.py [--model-dir data/models/attribution_2026_11_08] [--dry-run]
    uv run python scripts/publish_attribution_model_2026_11_14.py --check [--model-dir ...]     # read-only

The cut gives a piece of a round-up that names no company to the company this model chooses, or to nobody
(``news_nlp.attribution_model``; ``THRESHOLD`` 0.5). Until now the weights lived only in the git-ignored local folder
``data/models/attribution_2026_11_08`` (identity ``local:8399ccf003c60143``, the hash of its top-level files except ``metrics.json``).
One repository, ``gamug/FinBERT-news-attribution``, **private**: ``model.safetensors``, ``config.json``, ``tokenizer.json``,
``tokenizer_config.json``, ``metrics.json`` (the training record) and a ``README.md`` card written from that record. The licence is
``cc-by-nc-sa-4.0``: the base ``gamug/FinBERT-financial-news`` carries the share-alike of ``ProsusAI/finbert``. After the upload the commit,
the sha256 of the card and of the weights and the identity are written to ``config/published_models.json`` (key ``attribution``). The pin of
the cut is ``config/attribution_model.json`` (#186); to repeat the cut from the Hub copy download only the five model files into a clean
folder (a ``README.md`` or ``.gitattributes`` in it changes the identity, and so the revision of the cut). ``--check`` is read-only. Needs
``HF_TOKEN`` (write scope) in the environment or ``.env``.
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from news_nlp.attribution_model import THRESHOLD, directory_identity

REPO = "gamug/FinBERT-news-attribution"
FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "metrics.json",
)
WEIGHTS = "model.safetensors"
WHAT_ADDED = "attribution: private (the pin of the cut is config/attribution_model.json, #186; T-194 tests the Hub copy)"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def card(metrics: dict[str, Any], identity: str) -> str:
    """The model card, written from the training record (``metrics.json``) so that no figure is typed by hand."""
    base = metrics["base"]
    args, train, val, report = metrics["args"], metrics["train"], metrics["val"], metrics["report"]
    decision = metrics["eval"]["decision"]
    rows = "\n".join(
        f"| {name} | {int(report[f'{key} model']['pairs']):,} | {report[f'{key} model']['ranking']:.3f} | "
        f"{report[f'{key} baseline']['ranking']:.3f} | {report[f'{key} model']['threshold']:.3f} |"
        for name, key in (
            ("all validation sentences", "all"),
            ("from 2022", "2022+"),
            ("before 2022", "before 2022"),
            ("first sentence of a paragraph", "first sentences"),
        )
    )
    return f"""---
license: cc-by-nc-sa-4.0
base_model: {base["model"]}
language:
- en
pipeline_tag: text-classification
tags:
- finance
- news
- company-attribution
- sentence-pair
---

# FinBERT-news-attribution

Reads a pair and answers *is this sentence about this company?* It is the attribution model of the cut in
[`portfolio-nlp`](https://github.com/gamug/portfolio-nlp) (step 1.2): in a news round-up that lists several companies, a piece of
text that names no company goes to the company this model scores highest, if the score reaches {THRESHOLD}, or to nobody.

**Input.** The first segment is the company's name; the second is the previous sentence (the context) followed by the sentence.
**Output.** One logit pair; the probability of the positive class is the score. Local identity (`directory_identity` of
`news_nlp.attribution_model`: the hash of the top-level files, `metrics.json` excluded): `{identity}`.

## Training

- **Base:** `{base["model"]}` at `{base["revision"][:8]}` (a fine-tune of `ProsusAI/finbert`).
- **Data:** the project's own news corpus: {train["bodies"]:,} bodies, {train["lead paragraphs"]:,} paragraphs that lead with a company. Every
  sentence of such a paragraph is a positive for that company, with the company's name **masked** (`[COMPANY]`) so the model cannot learn that the name is
  there; the same sentence paired with another company of the body is a negative. {metrics["pairs_used"]:,} pairs. The articles are not redistributed.
- **Recipe:** {args["epochs"]} epochs, learning rate {args["lr"]}, batch {args["batch_size"]}, seed {args["seed"]}; {metrics["seconds"]} s.
- **Validation:** {val["bodies"]} bodies held out ({val["lead paragraphs"]:,} paragraphs).

## Figures (the validation bodies; the training record is `metrics.json`)

*Ranking* is, on each sentence, the share of (right company, wrong company) pairs the scores rank right; *0.5* is the accuracy of the 0.5 threshold.
The baseline scores 1 when the company is named in the previous sentence (the context) and 0 otherwise (`news_nlp.attribution_model.named_in_context`).

| Sentences | Pairs | Ranking, model | Ranking, baseline | Accuracy at 0.5, model |
|---|---|---|---|---|
{rows}

On {int(decision["sentences"]):,} validation sentences, the decision of the cut (the best company if its score reaches the threshold, else nobody) gives the right company in {pct(decision["model right"])}
(baseline {pct(decision["baseline right"])}), the wrong one in {pct(decision["model wrong"])} ({pct(decision["baseline wrong"])}) and nobody in
{pct(decision["model nobody"])} ({pct(decision["baseline nobody"])}).

## Limits

English news about companies of one universe (the S&P 500 of the project); a short or very ambiguous sentence is the hard case (first sentences of a
paragraph: see the table). The figures are the validation of the training run, not a test of the cut: the faithfulness of the whole cut is measured apart
(`docs/extractor_training.md` of the repository). The repository is private and the model is **not pinned** in the pipeline.

## License

`cc-by-nc-sa-4.0`: attribution, **non-commercial use only**, derivatives under the same license (the base `{base["model"]}` is itself
a derivative of `ProsusAI/finbert`, `cc-by-nc-sa-4.0`).
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def publish(token: str, model_dir: Path, text: str, dry: bool) -> str:
    from huggingface_hub import HfApi  # noqa: PLC0415 -- only the upload needs it

    if dry:
        print(
            f"[dry run] {REPO} (private): {', '.join(FILES)} + README.md ({len(text)} characters)"
        )
        return "dry-run"
    api = HfApi(token=token)
    api.create_repo(REPO, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in FILES:
            (stage / name).symlink_to((model_dir / name).resolve())
        (stage / "README.md").write_text(text, encoding="utf-8")
        info = api.upload_folder(
            repo_id=REPO,
            folder_path=str(stage),
            commit_message="attribution model of the cut (step 1.2): weights, tokenizer, training record and card",
        )
    sha = str(getattr(info, "oid", ""))
    on_hub = set(api.list_repo_files(REPO, revision=sha or None))
    if (set(FILES) | {"README.md"}) - on_hub:  # never record an upload that lost a file
        raise SystemExit(
            f"{REPO}@{sha}: files missing on the Hub ({sorted((set(FILES) | {'README.md'}) - on_hub)}); nothing recorded"
        )
    print(f"{REPO} (private): {sha}")
    return sha


def record_upload(published: Path, revision: str, text: str, weights: str, identity: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"the upload returned no commit SHA ({revision!r}); nothing recorded")
    current = json.loads(published.read_text(encoding="utf-8")) if published.exists() else {}
    # only the attribution clause is replaced (it ends at the next ";"); the clauses of the other models are kept
    rest = re.sub(r"(^|;\s*)attribution:[^;]*", "", current.get("what", "").rstrip(".")).strip("; ")
    current["what"] = "; ".join(part for part in (rest, WHAT_ADDED) if part)
    current["attribution"] = {
        "repo": REPO,
        "revision": revision,
        "private": True,
        "card_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "weights_sha256": weights,
        "identity": identity,
    }
    published.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def check_published(record: Path, model_dir: Path, token: str) -> dict[str, bool | None]:
    """Read only: whether the recorded revision is private, still the head, with the recorded card and weights, and the local folder is the recorded one."""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    entry = json.loads(record.read_text(encoding="utf-8"))["attribution"]
    out: dict[str, bool | None] = {}
    try:  # an anonymous request must be refused: the repository is private
        urllib.request.urlopen(
            f"https://huggingface.co/{REPO}/resolve/main/config.json", timeout=20
        )
        out["private (an anonymous request is refused)"] = False
    except urllib.error.HTTPError as err:
        out["private (an anonymous request is refused)"] = err.code in (401, 403, 404)
    if not token:
        raise SystemExit("HF_TOKEN is not set: the rest of the check needs it")
    api = HfApi(token=token)
    info = api.model_info(REPO, files_metadata=True)
    out["the recorded revision is the head"] = info.sha == entry["revision"]
    lfs = {f.rfilename: f.lfs.sha256 for f in info.siblings if f.lfs}
    out["the weights on the Hub are the recorded ones"] = (
        lfs.get(WEIGHTS) == entry["weights_sha256"]
    )
    readme = Path(hf_hub_download(REPO, "README.md", revision=entry["revision"], token=token))
    out["the card on the Hub is the recorded one"] = (
        hashlib.sha256(readme.read_bytes()).hexdigest() == entry["card_sha256"]
    )
    out["the local folder has the recorded identity"] = (
        directory_identity(model_dir) == entry["identity"]
        if (model_dir / WEIGHTS).exists()
        else None
    )
    return out


def main() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir", type=Path, default=Path("data/models/attribution_2026_11_08")
    )
    parser.add_argument("--record", type=Path, default=Path("config/published_models.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="read-only check of the recorded revision"
    )
    args = parser.parse_args()
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "")
    if args.check:
        results = check_published(args.record, args.model_dir, token)
        for name, ok in results.items():
            print(f"{name}: {'not compared' if ok is None else 'yes' if ok else 'NO'}")
        raise SystemExit(0 if all(ok is not False for ok in results.values()) else 1)
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN (write scope) is not set")
    identity = directory_identity(args.model_dir)
    metrics = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    text = card(metrics, identity)
    revision = publish(token, args.model_dir, text, args.dry_run)
    if not args.dry_run:
        record_upload(args.record, revision, text, sha256_file(args.model_dir / WEIGHTS), identity)
    print(json.dumps({"repo": REPO, "revision": revision, "identity": identity}))


if __name__ == "__main__":
    main()
