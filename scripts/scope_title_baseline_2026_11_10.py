"""Feasibility of T-147 (PLAN D40): how far does a title alone get on the scope?

    uv run python scripts/scope_title_baseline_2026_11_10.py

A TF-IDF (1-2 grams) + logistic regression on the **title** only, trained on ``gold/category_gold_v2.csv`` (284) and
scored on ``gold/category_gold_v2_test.csv`` (200, never trained on). ``market_economy``, ``investing_general`` and
``unrelated`` are merged into ``out_of_scope``. The figure is the plain agreement on the 200 (not corpus-weighted); the
baselines are "always single" and the confusion matrix. This is a floor for the pinned model of T-147, not the model.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

GOLD = Path("gold")
LABELS = ["single", "multi", "sector", "out_of_scope"]
SCOPES = {"single", "multi", "sector"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load(marks: str, texts: str) -> tuple[list[str], list[str]]:
    titles = {(r["article_id"], r["ticker"]): r["title"] for r in read(GOLD / "texts" / texts)}
    rows = read(GOLD / marks)
    x = [titles[(r["article_id"], r["ticker"])] for r in rows]
    y = [r["scope"] if r["scope"] in SCOPES else "out_of_scope" for r in rows]
    return x, y


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    x_train, y_train = load("category_gold_v2.csv", "category_gold_v2_texts.csv")
    x_test, y_test = load("category_gold_v2_test.csv", "category_gold_v2_test_texts.csv")
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, C=5).fit(vec.fit_transform(x_train), y_train)
    pred = model.predict(vec.transform(x_test))
    print(
        f"title-only agreement on the {len(y_test)} held-out bodies: {accuracy_score(y_test, pred):.3f}"
    )
    print(f"always single: {y_test.count('single') / len(y_test):.3f}")
    print("rows = gold, columns = predicted,", LABELS)
    print(confusion_matrix(y_test, pred, labels=LABELS))


if __name__ == "__main__":
    main()
