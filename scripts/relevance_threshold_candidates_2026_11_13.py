"""Candidate thresholds for the main company, on the out-of-fold relevance scores of the 809 marked rows (T-169, PLAN D39 (i); owner's choice).

    uv run python scripts/relevance_threshold_candidates_2026_11_13.py [--oof data/sentiment_pair_relevance_cv/oof_relevance.json] [--write docs/...json]

A row is *relevant* when the owner marked a direction (positive or negative) for the company, *not relevant* when the mark is neutral
(``docs/sentiment_training.md``, "A relevance model"). A model score is the average of the two cross-validation repeats of the same row. For each
threshold ``t`` (a company is main at ``score >= t``, ``steps.is_main``) the table gives:

- ``relevant_kept`` (recall): the share of relevant rows the model calls main. The rest are the rows that would reach the review list **if the text had no
  other main company**: the 809 rows are one company each, so this is an upper bound on the review share of the texts that have a company.
- ``not_relevant_main``: the share of not-relevant rows the model calls main (a mention taken for the main company).
- ``precision``: of the rows called main, the share that is relevant.
- ``under``: the share of all 809 rows under the threshold.

T-172 rejects a threshold that sends about 30% to a human; the table lets the owner read that off. The rows were drawn with weights
(``pool_class_size``): these figures are **unweighted** and say so. The rows are summaries; the step also reads sub-news, which were never scored
(PLAN D39, "Open": whether the relevance model holds on sub-news), so the table is the evidence for summaries only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GOLD = (ROOT / "gold" / "sentiment_train.csv", ROOT / "gold" / "sentiment_train_2.csv")
DEFAULT_OOF = ROOT / "data" / "sentiment_pair_relevance_cv" / "oof_relevance.json"
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(2, 19))  # 0.10 ... 0.90


def load(
    oof_path: Path = DEFAULT_OOF, gold: tuple[Path, ...] = GOLD
) -> tuple[list[float], list[bool]]:
    """The mean out-of-fold score of each of the 809 rows and whether the owner marked it relevant (not neutral)."""
    saved = json.loads(oof_path.read_text(encoding="utf-8"))
    sentiment: dict[str, str] = {}
    for path in gold:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            sentiment.update({r["key"]: r["sentiment"] for r in csv.DictReader(fh)})
    repeats = saved["R"]
    scores = [sum(r[i] for r in repeats) / len(repeats) for i in range(len(saved["keys"]))]
    relevant = [sentiment[k] != "neutral" for k in saved["keys"]]
    return scores, relevant


def candidates(
    scores: list[float], relevant: list[bool], thresholds: tuple[float, ...] = THRESHOLDS
) -> list[dict[str, Any]]:
    """One line per threshold (see the module docstring); a ratio over nothing is ``None``."""
    n_rel = sum(relevant)
    n_not = len(relevant) - n_rel

    def ratio(a: int, b: int) -> float | None:
        return None if b == 0 else a / b

    table = []
    for t in thresholds:
        main = [s >= t for s in scores]
        tp = sum(m and r for m, r in zip(main, relevant, strict=True))
        fp = sum(m and not r for m, r in zip(main, relevant, strict=True))
        table.append(
            {
                "threshold": t,
                "relevant_kept": ratio(tp, n_rel),
                "not_relevant_main": ratio(fp, n_not),
                "precision": ratio(tp, tp + fp),
                "under": ratio(sum(not m for m in main), len(scores)),
            }
        )
    return table


def render(table: list[dict[str, Any]], n_rel: int, n_not: int) -> str:
    def pct(v: float | None) -> str:
        return "  n/a" if v is None else f"{100 * v:5.1f}"

    lines = [
        f"{n_rel} relevant rows, {n_not} not relevant (unweighted), mean of the two out-of-fold repeats",
        "threshold  relevant_kept  not_relevant_main  precision  under",
    ]
    lines += [
        f"{r['threshold']:9.2f}  {pct(r['relevant_kept']):>13}  {pct(r['not_relevant_main']):>17}  {pct(r['precision']):>9}  {pct(r['under']):>5}"
        for r in table
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    args = parser.parse_args()
    if not args.oof.exists():
        sys.exit(
            f"{args.oof} is missing: it is written by scripts/sentiment_relevance_cv_2026_11_10.py (git-ignored)"
        )
    scores, relevant = load(args.oof)
    n_rel = sum(relevant)
    print(render(candidates(scores, relevant), n_rel, len(relevant) - n_rel))


if __name__ == "__main__":
    main()
