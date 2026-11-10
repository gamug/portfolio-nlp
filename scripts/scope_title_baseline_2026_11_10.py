"""Feasibility of T-147 (PLAN D40): how far does a title alone get on the scope?

    uv run python scripts/scope_title_baseline_2026_11_10.py

A TF-IDF (1-2 grams) + logistic regression on the **title** only, trained on ``gold/category_gold_v2.csv`` (284) and
scored on ``gold/category_gold_v2_test.csv`` (200, never trained on). ``market_economy``, ``investing_general`` and
``unrelated`` are merged into ``out_of_scope``, for the model **and** for the rules it is compared with, so both are
read in the same four classes. Every figure is **weighted to the corpus** (``stratum`` / ``frame`` of the 200, the same
weights and the same seeded bootstrap as ``report_wi18 scope --gold``), with its 95% interval, and the plain agreement
is printed beside it. This is a floor for the pinned model of T-147, not the model.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from news_nlp import gold_metrics, triage

GOLD = Path("gold")
LABELS = ("single", "multi", "sector", "out_of_scope")
SCOPES = {"single", "multi", "sector"}


def merged(scope: str) -> str:
    return scope if scope in SCOPES else "out_of_scope"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def titles(texts: str) -> dict[tuple[str, str], str]:
    return {(r["article_id"], r["ticker"]): r["title"] for r in read(GOLD / "texts" / texts)}


def load(marks: str, texts: str) -> tuple[list[dict[str, str]], list[str]]:
    by_key = titles(texts)
    rows = read(GOLD / marks)
    missing = [r["article_id"] for r in rows if (r["article_id"], r["ticker"]) not in by_key]
    if missing:
        raise SystemExit(
            f"{marks}: {len(missing)} marks have no title in gold/texts/{texts} (first: {missing[0]})"
        )
    return rows, [by_key[(r["article_id"], r["ticker"])] for r in rows]


def lines(name: str, rows: Sequence[Mapping[str, str]], predicted: Sequence[str]) -> list[str]:
    populations: dict[str, int] = {}
    for r in rows:
        populations.setdefault(r["stratum"], int(r["frame"]))
    figures = [
        {
            "article_id": r["article_id"],
            "stratum": r["stratum"],
            "gold": merged(r["scope"]),
            "predicted": p,
        }
        for r, p in zip(rows, predicted, strict=True)
    ]
    figures.sort(key=lambda r: int(r["article_id"]))
    overall = gold_metrics.stratified_report(
        figures, populations, label="single", stratum_key="stratum"
    )
    out = [
        f"{name}: agreement weighted to the corpus {overall.agreement} (plain {overall.raw_agreement})"
    ]
    for label in LABELS:
        r = gold_metrics.stratified_report(figures, populations, label=label, stratum_key="stratum")
        out.append(f"{label:>14}: {r.gold_count:3d} gold | recall {r.recall}")
    return out


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    train, x_train = load("category_gold_v2.csv", "category_gold_v2_texts.csv")
    test, x_test = load("category_gold_v2_test.csv", "category_gold_v2_test_texts.csv")
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, C=5).fit(
        vec.fit_transform(x_train), [merged(r["scope"]) for r in train]
    )
    title_pred = [str(p) for p in model.predict(vec.transform(x_test))]
    rules_pred = [merged(triage.RULE_SCOPE_TO_GOLD[r["stratum"]]) for r in test]
    single = ["single"] * len(test)
    for name, pred in (
        ("title-only model", title_pred),
        ("rules triage", rules_pred),
        ("always single", single),
    ):
        print("\n".join(lines(name, test, pred)), end="\n\n")
    print(
        "confusion matrix of the title-only model (plain counts), rows = gold, columns = predicted",
        LABELS,
    )
    count = Counter((merged(r["scope"]), p) for r, p in zip(test, title_pred, strict=True))
    for gold in LABELS:
        print(f"{gold:>14}", [count[(gold, p)] for p in LABELS])


if __name__ == "__main__":
    main()
