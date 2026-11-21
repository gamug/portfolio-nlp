"""Score the owner's partial main-company marks (T-170) against the relevance model of step 3.

    uv run scripts/score_main_company_marks_2026_11_20.py [--marks gold/main_company_marks_2026_11_20.csv] [--thresholds 0.1,0.3,0.5]

Reads the tracked marks (the owner's ``es_principal`` ``yes`` or ``no``, ``falta_la_principal``, ``notes``, the relevance score of the pair and the
relevance band of its item). The current version is ``gold/main_company_marks_2026_11_20.csv`` (built by ``build_main_company_marks_2026_11_20.py``);
the 2026-11-16 version is ``gold/main_company_marks.csv``. A pair is called main when its score is at the threshold or over it. Nothing is written.

Figures are unweighted (the six bands were drawn evenly, so they are not corpus rates) except the re-weighted block, which weights each band by its
size in the corpus (``POPULATION``). The scenario block scores the same marks with two kinds of article left out:

* ``no_company``: a note says the text is not about any company (``NO_COMPANY``);
* ``missed``: the owner's ``falta_la_principal`` is ``yes`` on a row of the article, or a note says the main company is missing from the list or is
  out of the S&P 500 universe (``MISSED``).

The SHA-256 of the file scored is printed first. Used for ``docs/main_company_training.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple

THRESHOLDS = (0.3, 0.5, 0.7)
#: Pieces of the ``units5`` cut with a candidate company in each relevance band of the piece's highest score (``to_tag.strata.json``, ``population``).
POPULATION = {0: 13756, 1: 9724, 2: 4994, 3: 4980, 4: 12221, 5: 105945}
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261119
DEFAULT_MARKS = Path("gold/main_company_marks_2026_11_20.csv")
DEFAULT_EXCLUDE = ()
#: The owner's notes that say the text is about no company at all (proposed for exclusion by the owner).
NO_COMPANY = re.compile(r"not about any company", re.I)
#: The owner's notes that say the main company is missing from the list, or is out of the S&P 500 universe.
MISSED = re.compile(
    r"missing|missed|not in (the )?text|not mentioned|out of S&P|not (a |an )?(in )?S&P|n't belong|not belong|does not belong",
    re.I,
)


class Mark(NamedTuple):
    item: str
    ticker: str
    main: bool
    score: float
    band: int
    notes: str = ""
    missing: bool = False  # falta_la_principal: the piece has a main company the list does not show


def wilson(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(path: Path) -> list[Mark]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            Mark(
                r["item_id"],
                r["ticker"],
                r["es_principal"] == "yes",
                float(r["relevance_score"]),
                int(r["band"]),
                (r.get("notes") or "").strip(),
                (r.get("falta_la_principal") or "").strip() == "yes",
            )
            for r in csv.DictReader(fh)
        ]


def piece_flags(rows: list[Mark]) -> dict[str, set[str]]:
    """The exclusions that apply to each article (``no_company``, ``missed``)."""
    flags: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if NO_COMPANY.search(r.notes):
            flags[r.item].add("no_company")
        elif r.missing or MISSED.search(r.notes):
            flags[r.item].add("missed")
    return flags


def scenarios(rows: list[Mark]) -> dict[str, list[Mark]]:
    flags = piece_flags(rows)

    def keep(*drop: str) -> list[Mark]:
        return [r for r in rows if not flags[r.item] & set(drop)]

    return {
        "all pieces": rows,
        "without no-company pieces": keep("no_company"),
        "without missed-main pieces": keep("missed"),
        "without both": keep("no_company", "missed"),
    }


def at_threshold(rows: list[Mark], threshold: float) -> dict[str, int]:
    """Counts of the one-vs-rest table at a threshold: called main, true main, and the same for not main."""
    yes = sum(r.main for r in rows)
    return {
        "rows": len(rows),
        "yes": yes,
        "no": len(rows) - yes,
        "called_main": sum(1 for r in rows if r.score >= threshold),
        "tp_main": sum(1 for r in rows if r.main and r.score >= threshold),
        "called_not": sum(1 for r in rows if r.score < threshold),
        "tp_not": sum(1 for r in rows if not r.main and r.score < threshold),
    }


def group(rows: list[Mark]) -> dict[str, list[Mark]]:
    items: dict[str, list[Mark]] = defaultdict(list)
    for r in rows:
        items[r.item].append(r)
    return items


def by_item(rows: list[Mark], threshold: float = 0.5) -> dict[str, int]:
    items = group(rows)
    with_main = [k for k, v in items.items() if any(r.main for r in v)]
    called = [k for k, v in items.items() if any(r.score >= threshold for r in v)]
    return {
        "items": len(items),
        "with_main": len(with_main),
        "found": sum(
            1 for k in with_main if any(r.main and r.score >= threshold for r in items[k])
        ),
        "called": len(called),
        "agreed": sum(1 for k in called if any(r.main and r.score >= threshold for r in items[k])),
    }


def by_band(rows: list[Mark]) -> dict[int, tuple[int, int]]:
    return {
        b: (sum(1 for r in rows if r.band == b and r.main), sum(1 for r in rows if r.band == b))
        for b in range(6)
    }


def auc(rows: list[Mark]) -> float:
    """ROC AUC of the relevance score for the owner's `yes` against `no` (ties count half); unweighted."""
    yes = [r.score for r in rows if r.main]
    no = [r.score for r in rows if not r.main]
    if not yes or not no:
        return float("nan")
    wins = sum(1.0 if y > n else 0.5 if y == n else 0.0 for y in yes for n in no)
    return wins / (len(yes) * len(no))


def interval(
    rows: list[Mark], stat: Callable[[list[Mark]], float], level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of ``stat`` by resampling whole articles (the rows of an article are not independent)."""
    items = list(group(rows).values())
    rng = random.Random(BOOTSTRAP_SEED)  # noqa: S311  (a seeded resample, not a secret)
    values = [
        v
        for v in (
            stat([r for it in rng.choices(items, k=len(items)) for r in it])
            for _ in range(BOOTSTRAP)
        )
        if not math.isnan(v)
    ]  # a resample without both classes has no AUC; it is left out
    if not values:
        return (float("nan"), float("nan"))
    values.sort()
    lo = int(len(values) * (1 - level) / 2)
    return values[lo], values[len(values) - 1 - lo]


def recall_rows(rows: list[Mark], threshold: float = 0.5) -> float:
    yes = [r for r in rows if r.main]
    return sum(r.score >= threshold for r in yes) / len(yes) if yes else 0.0


def hit_rate(rows: list[Mark], threshold: float = 0.5) -> float:
    c = by_item(rows, threshold)
    return c["found"] / c["with_main"] if c["with_main"] else 0.0


def weighted(rows: list[Mark], threshold: float) -> dict[str, float]:
    """Piece-level recall and precision with each band weighted by its corpus size (a band with no article is skipped)."""
    items = group(rows)
    band = {k: v[0].band for k, v in items.items()}
    with_main = found = called = agreed = 0.0
    for b, size in POPULATION.items():
        its = [k for k in items if band[k] == b]
        if not its:
            continue
        w = size / len(its)
        for k in its:
            rs = items[k]
            has = any(r.main for r in rs)
            hit = any(r.main and r.score >= threshold for r in rs)
            with_main += w * has
            found += w * hit
            called += w * any(r.score >= threshold for r in rs)
            agreed += w * hit
    return {
        "with_main_share": with_main / sum(POPULATION.values()),
        "recall": found / with_main,
        "precision": agreed / called,
    }


def by_owner_mains(rows: list[Mark], threshold: float = 0.5) -> dict[str, tuple[int, int]]:
    """Articles with one owner main against articles with two or more: (found, articles) for each."""
    out = {"one": [0, 0], "several": [0, 0]}
    for rs in group(rows).values():
        n = sum(r.main for r in rs)
        if n == 0:
            continue
        k = "one" if n == 1 else "several"
        out[k][1] += 1
        out[k][0] += any(r.main and r.score >= threshold for r in rs)
    return {k: (v[0], v[1]) for k, v in out.items()}


def pct(x: int, n: int) -> str:
    return f"{100 * x / n:.1f}%" if n else "n/a"


def article_auc(rows: list[Mark]) -> float:
    """AUC of the best relevance score of an article for "the article has an owner main" against "it has none" (unweighted).

    One value per article, so the several owner mains of an article count once; it is the article-level counterpart of ``auc``.
    """
    items = group(rows)
    pos = [max(r.score for r in v) for v in items.values() if any(r.main for r in v)]
    neg = [max(r.score for r in v) for v in items.values() if not any(r.main for r in v)]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def single_pick(rows: list[Mark]) -> dict[str, int]:
    """The decided headline (2026-11-21): the model names one main company per article, its top-scoring row.

    ``with_main``: articles with at least one owner main. ``found``: of those, the top row is one the owner marked main (the
    only pick the model makes is counted; a second or third owner main is not asked for). ``called``: every article, because the
    model always names one; ``agreed``: the top row is an owner main. Ties go to the first row of the article.
    """
    items = group(rows)
    tops = {k: max(v, key=lambda r: r.score) for k, v in items.items()}
    with_main = [k for k, v in items.items() if any(r.main for r in v)]
    return {
        "items": len(items),
        "with_main": len(with_main),
        "found": sum(1 for k in with_main if tops[k].main),
        "called": len(items),
        "agreed": sum(1 for k in items if tops[k].main),
    }


def single_pick_recall(rows: list[Mark]) -> float:
    c = single_pick(rows)
    return c["found"] / c["with_main"] if c["with_main"] else 0.0


def article(rows: list[Mark], threshold: float) -> dict[str, int]:
    """The per-article threshold counts (secondary; the headline is ``single_pick``): an article is found when the model's main company (its score at or over the threshold) is one the owner marked main."""
    return by_item(rows, threshold)


def scenario_line(name: str, rows: list[Mark], thresholds: tuple[float, ...]) -> str:
    items = len(group(rows))
    yes = sum(r.main for r in rows)
    a = auc(rows)
    aa = article_auc(rows)
    parts = []
    for t in thresholds:
        c = at_threshold(rows, t)
        i = article(rows, t)
        parts.append(
            f"at {t}: per article recall {pct(i['found'], i['with_main'])} ({i['found']}/{i['with_main']}), "
            f"per article precision {pct(i['agreed'], i['called'])} ({i['agreed']}/{i['called']}); "
            f"row level: main R {pct(c['tp_main'], c['yes'])} ({c['tp_main']}/{c['yes']}), main P {pct(c['tp_main'], c['called_main'])}, "
            f"not main R {pct(c['tp_not'], c['no'])} ({c['tp_not']}/{c['no']})"
        )
    a_txt = "n/a" if math.isnan(a) else f"{a:.3f}"
    aa_txt = "n/a" if math.isnan(aa) else f"{aa:.3f}"
    npos = sum(any(r.main for r in v) for v in group(rows).values())
    aa_txt += f" ({npos} articles with an owner main against {items - npos} without)"
    return (
        f"- {name}: {items} pieces, {len(rows)} rows ({yes} yes, {len(rows) - yes} no); article AUC {aa_txt}; row AUC {a_txt}\n    "
        + "\n    ".join(parts)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--marks", type=Path, default=DEFAULT_MARKS)
    parser.add_argument(
        "--exclude",
        default=",".join(DEFAULT_EXCLUDE),
        help="extra items to leave out of the all-pieces block",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in THRESHOLDS),
        help="thresholds to score, comma separated",
    )
    args = parser.parse_args()
    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    rows = load(args.marks)
    print(f"{args.marks}: sha256 {hashlib.sha256(args.marks.read_bytes()).hexdigest()}")
    print(
        f"marked rows {len(rows)} of {len({r.item for r in rows})} items: {sum(r.main for r in rows)} yes, {sum(not r.main for r in rows)} no"
    )
    s = single_pick(rows)
    slo, shi = wilson(s["found"], s["with_main"])
    plo2, phi2 = wilson(s["agreed"], s["called"])
    rlo2, rhi2 = interval(rows, single_pick_recall)
    print(
        "HEADLINE, decided 2026-11-21 (the model's single main pick: the top-scoring row of each piece)"
    )
    print(
        f"single-pick recall: {s['found']}/{s['with_main']} = {pct(s['found'], s['with_main'])} [Wilson {100 * slo:.1f}, {100 * shi:.1f}; resampling {100 * rlo2:.1f}, {100 * rhi2:.1f}]; "
        f"single-pick precision: {s['agreed']}/{s['called']} = {pct(s['agreed'], s['called'])} [Wilson {100 * plo2:.1f}, {100 * phi2:.1f}]; "
        f"pieces {s['items']}, with an owner main {s['with_main']}"
    )
    print(
        "PER ARTICLE, threshold rule (secondary: a piece is found when any row at or over the threshold is an owner main)"
    )
    for t in thresholds:
        i = article(rows, t)
        rlo, rhi = wilson(i["found"], i["with_main"])
        plo, phi = wilson(i["agreed"], i["called"])
        ilo, ihi = interval(rows, partial(hit_rate, threshold=t))
        print(
            f"article recall at {t}: {i['found']}/{i['with_main']} = {pct(i['found'], i['with_main'])} [Wilson {100 * rlo:.1f}, {100 * rhi:.1f}; resampling {100 * ilo:.1f}, {100 * ihi:.1f}]; "
            f"article precision: {i['agreed']}/{i['called']} = {pct(i['agreed'], i['called'])} [Wilson {100 * plo:.1f}, {100 * phi:.1f}]; "
            f"articles {i['items']}, with an owner main {i['with_main']}, called main by the model {i['called']}"
        )
    aa = article_auc(rows)
    alo, ahi = interval(rows, article_auc)
    print(
        f"article AUC (best score of the piece, piece with an owner main against piece without): {aa:.3f} [{alo:.3f}, {ahi:.3f}] "
        f"({sum(any(r.main for r in v) for v in group(rows).values())} pieces against {sum(not any(r.main for r in v) for v in group(rows).values())})"
    )
    for k, (f, n) in by_owner_mains(rows).items():
        print(f"articles with {k} owner main at 0.5: found {f}/{n} ({pct(f, n)})")
    for t in thresholds:
        w = weighted(rows, t)
        print(
            f"article recall re-weighted by band size at {t}: {100 * w['recall']:.1f}%, article precision {100 * w['precision']:.1f}% "
            f"(share of articles with an owner main {100 * w['with_main_share']:.1f}%)"
        )
    print(
        "ROW LEVEL (each owner main counts; a multi-company article can only score its share of mains, see docs)"
    )
    for t in thresholds:
        c = at_threshold(rows, t)
        lo, hi = wilson(c["tp_main"], c["called_main"])
        rlo, rhi = wilson(c["tp_main"], c["yes"])
        print(
            f"row threshold {t}: main P {c['tp_main']}/{c['called_main']} = {pct(c['tp_main'], c['called_main'])} [{100 * lo:.1f}, {100 * hi:.1f}], "
            f"R {c['tp_main']}/{c['yes']} = {pct(c['tp_main'], c['yes'])} [{100 * rlo:.1f}, {100 * rhi:.1f}]; "
            f"not main P {c['tp_not']}/{c['called_not']} = {pct(c['tp_not'], c['called_not'])}, R {c['tp_not']}/{c['no']} = {pct(c['tp_not'], c['no'])}"
        )
    alo, ahi = interval(rows, auc)
    print(
        f"row AUC of the score for yes against no: {auc(rows):.3f} [{alo:.3f}, {ahi:.3f}] (resampling pieces; the relevance model's out-of-fold AUC on summaries: 0.896)"
    )
    print("yes by band:", {b: f"{y}/{n}" for b, (y, n) in by_band(rows).items()})
    print("articles per band:", {b: len({r.item for r in rows if r.band == b}) for b in range(6)})
    flags = piece_flags(rows)
    for kind in ("no_company", "missed"):
        ids = sorted(k for k, v in flags.items() if kind in v)
        print(f"articles flagged {kind}: {len(ids)} {ids}")
    skip = set(args.exclude.split(",")) - {""}
    if skip:
        rest = [r for r in rows if r.item not in skip]
        print(f"extra exclusion {sorted(skip)}: {len(group(rest))} pieces, {len(rest)} rows")
    print("scenarios (the same marks, with the pieces of each exclusion left out):")
    for name, sub in scenarios(rows).items():
        print(scenario_line(name, sub, thresholds))


if __name__ == "__main__":
    main()
