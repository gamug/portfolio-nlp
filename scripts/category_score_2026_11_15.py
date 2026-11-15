# ruff: noqa: E402  -- one-shot script
"""Score rows with the kept weights of the refit (T-201, T-202): the 12 labels, one tag per row.

    uv run scripts/category_score_2026_11_15.py --rows <csv with article_id,company,text[,label]> --out <scores.csv>
    uv run scripts/category_score_2026_11_15.py --heldout --why "<reason>" --out <scores.csv>      # the second look at the held-out 200

Reads ``data/models/category_2026_11_15/`` (``--folder`` to name another) and **refuses a folder with no ``refit_record.json`` or whose weights are not the ones
the record names**. Writes one row per input row with the score of each label (entailment probability, as in the cross-validation), ``top`` and, when the input
has ``label``, whether it is right, plus ``<out>.meta.json`` with the sha256 of the weights, the seed, the input file's sha256 and the reason. A score always
says which weights it is of.

``--heldout`` writes ``docs/category_results_2026_11_15/heldout_scores.csv`` (and its ``.meta.json``) and builds the scorable single rows of ``gold/category_gold_v2_test.csv`` the way ``scripts/category_heldout_2026_11_13.py`` did (``category_groups_mock``'s
``build_rows``; ``competitive_actions`` and ``peer_comparison`` read as ``other``), needs ``--why`` and refuses to run again once
``docs/category_results_2026_11_15/heldout_scores.csv`` exists unless ``--again`` is given: the held-out is not scored again without the owner's word (T-201).
The tag set the bar is measured against (the held-out tags as they are, or after a review) is decided before the score and written in ``--why``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import category_multilabel_cv_2026_11_14 as cvs

import category_multilabel as ml
from news_category_stage import LABELS as L14

L12 = ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER)
cvs.LABELS = L12
DEFAULT_FOLDER = ROOT / "data" / "models" / "category_2026_11_15"
HELDOUT_DONE = ROOT / "docs" / "category_results_2026_11_15" / "heldout_scores.csv"


def heldout_rows() -> pd.DataFrame:
    """The scorable single rows of the held-out 200, as the 2026-11-13 scoring built them, with the tag read through the fold into ``other``."""
    import category_baseline_2026_11_13 as b  # noqa: PLC0415
    import category_groups_mock_2026_11_13 as mock  # noqa: PLC0415

    from news_nlp.triage import Company, CompanyResolver  # noqa: PLC0415

    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    res = CompanyResolver(
        [
            Company(x, c, s, "")
            for x, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    rows, _ = mock.build_rows(
        pd.read_csv(ROOT / "gold/category_gold_v2_test.csv"),
        pd.read_csv(ROOT / "gold/texts/category_gold_v2_test_texts.csv"),
        res,
    )
    rows = rows[rows.label.isin(b.LABELS)].reset_index(drop=True)
    rows["label"] = [ml.FOLD_INTO_OTHER.get(x, x) for x in rows.label]
    return rows


def main() -> None:  # noqa: C901 -- one linear command with its refusals
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(DEFAULT_FOLDER))
    ap.add_argument("--rows", help="a CSV with article_id, company, text and optionally label")
    ap.add_argument(
        "--heldout", action="store_true", help="the held-out 200 (the second look; needs --why)"
    )
    ap.add_argument("--why", default="", help="why, and which tag set the bar uses (recorded)")
    ap.add_argument("--again", action="store_true", help="score the held-out another time")
    ap.add_argument(
        "--out",
        help="the scores CSV (with --heldout it is always docs/category_results_2026_11_15/heldout_scores.csv, so the guard holds on a fresh clone)",
    )
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    if bool(args.rows) == args.heldout:
        sys.exit("give exactly one of --rows and --heldout")
    if args.heldout:
        args.out = str(HELDOUT_DONE)
    elif not args.out:
        sys.exit("--out is required with --rows")
    folder = Path(args.folder)
    try:
        record = ml.check_refit_folder(folder)
    except ValueError as exc:
        sys.exit(f"refused: {exc}")
    if args.heldout:
        if not args.why.strip():
            sys.exit(
                "refused: --heldout needs --why (the tag set the bar uses, decided before the score)"
            )
        if HELDOUT_DONE.exists() and not args.again:
            sys.exit(
                f"refused: {HELDOUT_DONE} exists: the held-out is not scored again without --again and the owner's word"
            )
        df, source = heldout_rows(), "gold/category_gold_v2_test.csv"
    else:
        df, source = pd.read_csv(args.rows, dtype=str, keep_default_na=False), str(args.rows)
        missing = {"article_id", "company", "text"} - set(df.columns)
        if missing:
            sys.exit(f"{args.rows} lacks {sorted(missing)}")
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder, dtype=torch.float32).to(
        cvs.DEV
    )
    if list(record["labels"]) != list(L12):  # type: ignore[call-overload]
        sys.exit("refused: the labels of the record are not the 12 of this script")
    scores = cvs.score(tok, model, model.config.label2id["entailment"], df, args.max_len)
    out = pd.DataFrame({"article_id": df.article_id, "company": df.company})
    for j, label in enumerate(L12):
        out[f"score_{label}"] = np.round(scores[:, j], 4)
    out["top"] = [L12[j] for j in scores.argmax(1)]
    meta = {
        "model_safetensors_sha256": record["model_safetensors_sha256"],
        "seed": record["seed"],
        "rows_file": source,
        "rows": len(df),
        "rows_file_sha256": hashlib.sha256(Path(args.rows).read_bytes()).hexdigest()
        if args.rows
        else None,
        "why": args.why,
        "heldout": bool(args.heldout),
    }
    if "label" in df.columns:
        out.insert(2, "label", list(df.label))
        out["right"] = (out.label == out.top).astype(int)
        meta["accuracy"] = round(float(out.right.mean()), 4)
    out.to_csv(args.out, index=False)
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
