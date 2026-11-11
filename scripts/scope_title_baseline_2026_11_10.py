"""Feasibility of T-147 (PLAN D40): how far does a title alone get on the scope?

    uv run python scripts/scope_title_baseline_2026_11_10.py [--body-chars N] [--company-features [--first M]]

A TF-IDF (1-2 grams) + logistic regression on the **title** only, trained on ``gold/category_gold_v2.csv`` (284) and
scored on ``gold/category_gold_v2_test.csv`` (200, never trained on). ``market_economy``, ``investing_general`` and
``unrelated`` are merged into ``out_of_scope``, for the model **and** for the rules it is compared with, so both are
read in the same four classes. Every figure is **weighted to the corpus** (``stratum`` / ``frame`` of the 200, the same
weights and the same seeded bootstrap as ``report_wi18 scope --gold``), with its 95% interval, and the plain agreement
is printed beside it. With ``--body-chars N`` the model also reads the first N characters of the body (owner,
2026-11-10: title plus the first lines). With ``--company-features`` the model also gets the number of distinct universe companies the matcher finds in the
title, in the first M characters of the body (``--first``, default 600; independent of ``--body-chars``, which only
changes the text TF-IDF reads) and in the whole body, as counts capped at 4 (``gold/universe.csv`` and the code of
``news_nlp/triage.py`` are fingerprinted, SHA-256 printed; the matcher digest is **informational**: nothing compares it,
because ``triage.py`` is edited by unrelated work, so a stale figure is spotted by reading the digest recorded with it): what tells ``single`` from ``multi``. The model with the
counts is then compared, paired and weighted, with the title-only model and with the rules (agreement and the recall of each
class, with the interval of the difference). **The body is the text of ``gold/texts``, which the draw capped at 6,020
characters (33 of the 484 bodies reach it), so "the whole body" is its first ~6,000 characters; production may
differ.** This is a floor for the pinned model of T-147, not the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from news_nlp import gold_metrics, triage
from news_nlp.triage import Company

GOLD = Path("gold")
LABELS = ("single", "multi", "sector", "out_of_scope")
SCOPES = {"single", "multi", "sector"}


def merged(scope: str) -> str:
    return scope if scope in SCOPES else "out_of_scope"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def titles(texts: str, body_chars: int = 0) -> dict[tuple[str, str], str]:
    rows = read(GOLD / "texts" / texts)
    return {
        (r["article_id"], r["ticker"]): f"{r['title']} . {r['text'][:body_chars]}"
        if body_chars
        else r["title"]
        for r in rows
    }


def load(marks: str, texts: str, body_chars: int = 0) -> tuple[list[dict[str, str]], list[str]]:
    by_key = titles(texts, body_chars)
    rows = read(GOLD / marks)
    missing = [r["article_id"] for r in rows if (r["article_id"], r["ticker"]) not in by_key]
    if missing:
        raise SystemExit(
            f"{marks}: {len(missing)} marks have no title in gold/texts/{texts} (first: {missing[0]})"
        )
    return rows, [by_key[(r["article_id"], r["ticker"])] for r in rows]


CAP = 4  # counts above this are one bucket


def matcher_digest() -> str:
    """SHA-256 of ``news_nlp/triage.py``: the code that decides what counts as a company (a change to it, even one that
    does not touch matching, changes the digest: then the figure is measured again)."""
    return hashlib.sha256(Path(triage.__file__).read_bytes()).hexdigest()


def universe_resolver() -> tuple[triage.CompanyResolver, str]:
    path = GOLD / "universe.csv"
    rows = read(path)
    companies = [Company(r["ticker"], r["company"], r["gics_sector"], "") for r in rows]
    return triage.CompanyResolver(companies), hashlib.sha256(path.read_bytes()).hexdigest()


def company_features(
    resolver: triage.CompanyResolver, rows: Sequence[Mapping[str, str]], texts: str, first: int
) -> csr_matrix:
    """One-hot of the capped number of distinct companies in the title, the first *first* characters and the body."""
    by_key = {(r["article_id"], r["ticker"]): r for r in read(GOLD / "texts" / texts)}
    out = np.zeros((len(rows), 3 * (CAP + 1)))
    for i, row in enumerate(rows):
        src = by_key[(row["article_id"], row["ticker"])]
        parts = (src["title"], src["text"][:first], src["text"])
        for j, part in enumerate(parts):
            out[i, j * (CAP + 1) + min(len(resolver.resolve(part)), CAP)] = 1.0
    return csr_matrix(out)


def paired_lines(
    rows: Sequence[Mapping[str, str]],
    candidate: Sequence[str],
    baselines: Mapping[str, Sequence[str]],
) -> list[str]:
    """Paired weighted differences (candidate minus each baseline) of the agreement and of the recall of each class."""
    populations: dict[str, int] = {}
    for r in rows:
        populations.setdefault(r["stratum"], int(r["frame"]))
    out = []
    for name, base in baselines.items():
        figures = [
            {
                "article_id": r["article_id"],
                "stratum": r["stratum"],
                "gold": merged(r["scope"]),
                "predicted": c,
                "baseline": b,
            }
            for r, c, b in zip(rows, candidate, base, strict=True)
        ]
        figures.sort(key=lambda r: int(r["article_id"]))
        for metric, label in (("agreement", "single"), *(("recall", label) for label in LABELS)):
            gain = gold_metrics.paired_stratified_gain(
                figures, populations, metric=metric, label=label
            )
            what = "agreement" if metric == "agreement" else f"recall of {label}"
            out.append(
                f"paired difference, {what}, minus {name}: {gain} ({gold_metrics.gain_verdict(gain)})"
            )
    return out


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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--body-chars", type=int, default=0)
    parser.add_argument("--company-features", action="store_true")
    parser.add_argument("--first", type=int, default=600)
    args = parser.parse_args()
    chars = args.body_chars
    train, x_train = load("category_gold_v2.csv", "category_gold_v2_texts.csv", chars)
    test, x_test = load("category_gold_v2_test.csv", "category_gold_v2_test_texts.csv", chars)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    a_train, a_test = vec.fit_transform(x_train), vec.transform(x_test)
    y_train = [merged(r["scope"]) for r in train]
    title_pred = [
        str(p) for p in LogisticRegression(max_iter=2000, C=5).fit(a_train, y_train).predict(a_test)
    ]
    rules_pred = [merged(triage.RULE_SCOPE_TO_GOLD[r["stratum"]]) for r in test]
    shown, shown_pred = "title-only model", title_pred
    if args.company_features:
        resolver, digest = universe_resolver()
        print(
            f"universe sha256 {digest}\nmatcher (triage.py) sha256 {matcher_digest()}\nfirst {args.first} characters"
        )
        b_train = hstack(
            [a_train, company_features(resolver, train, "category_gold_v2_texts.csv", args.first)]
        ).tocsr()
        b_test = hstack(
            [
                a_test,
                company_features(resolver, test, "category_gold_v2_test_texts.csv", args.first),
            ]
        ).tocsr()
        shown, shown_pred = (
            "title + company counts",
            [
                str(p)
                for p in LogisticRegression(max_iter=2000, C=5)
                .fit(b_train, y_train)
                .predict(b_test)
            ],
        )
    single = ["single"] * len(test)
    for name, pred in (
        (shown, shown_pred),
        ("rules triage", rules_pred),
        ("always single", single),
    ):
        print("\n".join(lines(name, test, pred)), end="\n\n")
    if args.company_features:
        print(
            "\n".join(
                paired_lines(
                    test, shown_pred, {"title-only model": title_pred, "rules": rules_pred}
                )
            ),
            end="\n\n",
        )
    print(
        f"confusion matrix of the {shown} (plain counts), rows = gold, columns = predicted", LABELS
    )
    count = Counter((merged(r["scope"]), p) for r, p in zip(test, shown_pred, strict=True))
    for gold in LABELS:
        print(f"{gold:>14}", [count[(gold, p)] for p in LABELS])


if __name__ == "__main__":
    main()
