"""Check and score the owner's answers on the two fresh sector draws, the 57 rows of 2026-11-13 and the 300 of 2026-11-18 (357 rows).

    uv run scripts/score_sector_fresh_2026_11_19.py --export   # once: data/ working files -> gold/sector_fresh_marks_2026_11_19.csv (+ texts)
    uv run scripts/score_sector_fresh_2026_11_19.py            # scores the tracked marks

The owner's working files are in ``data/`` (git-ignored); ``--export`` copies the answers and the English texts to ``gold/``, and the scoring reads the tracked copy. Nothing is written. It prints:

* consistency checks (unique ids, the answer column, the sector codes, the verdict against the key, the first 57 rows against the tracked gold);
* the precision of the assigned sectors (controls apart) over the whole file, and again **without the ``no`` answers whose note says the text is not a
  sector text** (``general_economy``, ``multi``, ``single``, ``out_of_scope``, an investing text: errors of the scope step, not of the sector choice);
* the controls, and the texts the gate sent out (how many the owner agrees to drop, how many had a sector the step guessed).
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import math
import re
from pathlib import Path

Z = 1.96
THRESHOLD = 0.0293  # config/sector_step.json margin_threshold
FIRST_DRAW = 57  # rows 1 to 57 are the draw of 2026-11-13; 58 to 357 are the draw of 2026-11-18
CODES = {"10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"}
#: A note that says the text belongs to another scope. Applied to a ``no`` with no ``right_sector``.
NOT_A_SECTOR = re.compile(
    r"multi|single|out_of_scope|general[_ ]?econ|market_econ|econom|investment|unrelated|not a sector|not a real|company|laboral|borderline",
    re.I,
)
Row = dict[str, str]


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def fmt(hits: int, n: int) -> str:
    if n == 0:
        return "n=0"
    lo, hi = wilson(hits, n)
    return f"{hits}/{n} = {100 * hits / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def read(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def not_a_sector(r: Row) -> bool:
    """A ``no`` that names no other sector and whose note says the text is not a sector text."""
    return r["correct"] == "no" and not r["right_sector"] and bool(NOT_A_SECTOR.search(r["notes"]))


def join(review: list[Row], key: list[Row]) -> list[Row]:
    by_id = {k["article_id"]: k for k in key}
    out = []
    for r in review:
        k = by_id[r["article_id"]]
        out.append(
            {
                **r,
                "correct": r["correct"].strip().lower(),
                "right_sector": r["right_sector"].strip(),
                **{"k_" + a: b for a, b in k.items()},
            }
        )
    return out


def checks(rows: list[Row], gold: list[Row]) -> None:
    print(f"rows {len(rows)}, unique article ids {len({r['article_id'] for r in rows})}")
    print(
        "answers not yes/no:",
        [r["review_order"] for r in rows if r["correct"] not in ("yes", "no")],
    )
    print(
        "right_sector not a GICS code:",
        [r["review_order"] for r in rows if r["right_sector"] and r["right_sector"] not in CODES],
    )
    print(
        "verdict differs from the key's status:",
        [
            r["review_order"]
            for r in rows
            if (r["k_status"] == "out_of_scope") != r["model_verdict"].startswith("OUT")
        ],
    )
    print(
        "control shown the step's own sector:",
        [
            r["review_order"]
            for r in rows
            if r["k_control"] == "yes" and r["k_shown"] == r["k_sector"]
        ],
    )
    seen = {g["article_id"]: g for g in gold if g["review"] == "fresh"}
    changed = [
        (
            r["review_order"],
            f"{seen[r['article_id']]['correct']}/{seen[r['article_id']]['right_sector']} -> {r['correct']}/{r['right_sector']}",
        )
        for r in rows
        if r["article_id"] in seen
        and (
            seen[r["article_id"]]["correct"].strip().lower(),
            seen[r["article_id"]]["right_sector"].strip(),
        )
        != (r["correct"], r["right_sector"])
    ]
    print(
        f"first rows against the tracked gold: {sum(r['article_id'] in seen for r in rows)} found, changed since: {changed}"
    )
    odd = [
        r["review_order"]
        for r in rows
        if r["k_status"] == "out_of_scope" and r["correct"] == "yes" and r["right_sector"]
    ]
    print("sent out, answered yes AND a sector named (contradiction):", odd)
    nameless = [
        r["review_order"]
        for r in rows
        if r["k_status"] == "out_of_scope" and r["correct"] == "no" and not r["right_sector"]
    ]
    print(f"sent out, answered no with no sector named ({len(nameless)}):", nameless)


def report(label: str, rows: list[Row]) -> None:
    assigned = [r for r in rows if r["k_status"] == "assigned" and r["k_control"] == "no"]
    controls = [r for r in rows if r["k_control"] == "yes"]
    sent = [r for r in rows if r["k_status"] == "out_of_scope"]
    scope = [r for r in assigned if not_a_sector(r)]
    kept = [r for r in assigned if not not_a_sector(r)]
    yes = lambda s: sum(r["correct"] == "yes" for r in s)  # noqa: E731
    print(
        f"\n== {label}: {len(rows)} rows ({len(assigned)} assigned, {len(controls)} controls, {len(sent)} sent out)"
    )
    print("assigned, all:                     ", fmt(yes(assigned), len(assigned)))
    print(f"no that say 'not a sector' ({len(scope)}):    ", [r["review_order"] for r in scope])
    print("assigned, without those:           ", fmt(yes(kept), len(kept)))
    other = [r for r in assigned if r["correct"] == "no" and not not_a_sector(r)]
    print(f"no that name another sector or no note: {len(other)}")
    for name, test in (
        ("margin", lambda r: float(r["k_margin"]) >= THRESHOLD),
        ("vote only", lambda r: float(r["k_margin"]) < THRESHOLD),
    ):
        a, k = [r for r in assigned if test(r)], [r for r in kept if test(r)]
        print(f"  {name:10s} all {fmt(yes(a), len(a))} | without {fmt(yes(k), len(k))}")
    print("controls answered yes:             ", fmt(yes(controls), len(controls)))
    print("sent out, the owner agrees:        ", fmt(yes(sent), len(sent)))
    wrong = [r for r in sent if r["correct"] == "no"]
    named = [r for r in wrong if r["right_sector"]]
    print(
        f"sent out, the owner would keep: {len(wrong)}; named a sector {len(named)}, the step's best guess was it in {sum(r['right_sector'] == r['k_candidate'] for r in named)}"
    )
    flipped = [r for r in sent if r["correct"] == "yes" and r["right_sector"]]
    if flipped:
        print("  if the contradicting rows were `no`:", fmt(yes(sent) - len(flipped), len(sent)))
    if label.startswith("all"):
        by: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in assigned:
            b = by[r["model_sector_code"]]
            b[1] += 1
            b[0] += r["correct"] == "yes"
            if not not_a_sector(r):
                b[3] += 1
                b[2] += r["correct"] == "yes"
        print("per sector (all | without the not-a-sector no):")
        for code in sorted(by):
            b = by[code]
            print(f"  {code}: {b[0]}/{b[1]} | {b[2]}/{b[3]}")


NAMES = {
    "10": "Energy", "15": "Materials", "20": "Industrials", "25": "Consumer Discretionary", "30": "Consumer Staples",
    "35": "Health Care", "40": "Financials", "45": "Information Technology", "50": "Communication Services",
    "55": "Utilities", "60": "Real Estate",
}  # fmt: skip
CORPUS_ASSIGNED, CORPUS_SENT = (
    56,
    44,
)  # per 100 sector texts (docs/sector_step.md, "The clean test")


def truth(r: Row) -> str | None:
    """The sector the owner's answer says the text has: a code, ``none`` (no sector), or ``None`` when the answer does not say."""
    if r["k_status"] == "assigned":
        if r["correct"] == "yes":
            return r["model_sector_code"]
        if r["right_sector"]:
            return r["right_sector"]
        return "none" if not_a_sector(r) else None
    if r["correct"] == "yes":
        return "none"
    return r["right_sector"] or None


def ovr(rows: list[Row]) -> None:
    """One-vs-rest precision and recall per sector, and for "no sector" (controls left out; they are not the step's answer)."""
    rows = [r for r in rows if r["k_control"] == "no"]
    pred = {
        r["article_id"]: (r["model_sector_code"] if r["k_status"] == "assigned" else "none")
        for r in rows
    }
    true = {r["article_id"]: truth(r) for r in rows}
    unknown = sum(t is None for t in true.values())
    print(
        f"\nOne vs rest ({len(rows)} rows without controls; truth unknown for {unknown} sent-out rows):"
    )
    print(
        "class | predicted | right | precision | unknown among predicted (no sector named, counted as wrong) "
        "| truth known (true class) | found | recall, upper bound"
    )
    for c in [*NAMES, "none"]:
        called = [a for a, v in pred.items() if v == c]
        tp = sum(true[a] == c for a in called)
        unknown_called = sum(true[a] is None for a in called)
        fn_known = sum(1 for a, t in true.items() if t == c and pred[a] != c)
        up = tp / (tp + fn_known) if tp + fn_known else 0.0
        print(
            f"{c} {NAMES.get(c, 'no sector')} | {len(called)} | {tp} | {fmt(tp, len(called))} | {unknown_called} "
            f"| {tp + fn_known} | {tp} | {100 * up:.1f}%"
        )
    sent = [r for r in rows if r["k_status"] == "out_of_scope"]
    assigned = [r for r in rows if r["k_status"] == "assigned"]
    tp = sum(true[r["article_id"]] == "none" for r in sent)
    fn = sum(true[r["article_id"]] == "none" for r in assigned)
    w_tp, w_fn = tp * CORPUS_SENT / len(sent), fn * CORPUS_ASSIGNED / len(assigned)
    print(
        f"no sector, weighted to the corpus rates ({CORPUS_ASSIGNED}/{CORPUS_SENT} per 100): recall {100 * w_tp / (w_tp + w_fn):.1f}% (sample {tp}/{tp + fn})"
    )


MARKS = Path("gold/sector_fresh_marks_2026_11_19.csv")
TEXTS = Path("gold/texts/sector_fresh_marks_texts_2026_11_19.csv")
KEY_COLUMNS = ("status", "sector", "candidate", "shown", "control", "margin", "vote_sector")
MARK_COLUMNS = (
    "review_order",
    "article_id",
    *KEY_COLUMNS,
    "model_sector_code",
    "model_verdict",
    "correct",
    "right_sector",
    "notes",
)
TEXT_COLUMNS = ("article_id", "ticker", "company", "pub_date", "source_domain", "title", "text")


def write(path: Path, columns: tuple[str, ...], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export(rows: list[Row]) -> None:
    """Copy the owner's answers and the texts they are about to ``gold/`` (the Spanish aid and the unanswered columns are not copied)."""
    write(MARKS, MARK_COLUMNS, [{**r, **{c: r["k_" + c] for c in KEY_COLUMNS}} for r in rows])
    write(TEXTS, TEXT_COLUMNS, rows)
    print(f"{len(rows)} rows -> {MARKS} and {TEXTS}")


def from_marks(path: Path) -> list[Row]:
    return [{**r, **{"k_" + c: r[c] for c in KEY_COLUMNS}} for r in read(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", type=Path, default=Path("data/sector_fresh_to_review_es.csv"))
    parser.add_argument("--key", type=Path, default=Path("data/sector_fresh_key.csv"))
    parser.add_argument("--gold", type=Path, default=Path("gold/sector_validation.csv"))
    parser.add_argument(
        "--export",
        action="store_true",
        help="write the tracked copy of the answers and texts from --review and --key, then stop",
    )
    parser.add_argument(
        "--marks",
        type=Path,
        default=MARKS,
        help="the tracked answers (used when --export is not given)",
    )
    args = parser.parse_args()
    if args.export:
        for p in (args.review, args.key):
            print(f"{p}: sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
        export(join(read(args.review), read(args.key)))
        return
    print(f"{args.marks}: sha256 {hashlib.sha256(args.marks.read_bytes()).hexdigest()}")
    rows = from_marks(args.marks)
    checks(rows, read(args.gold))
    report("all 357 rows", rows)
    ovr(rows)
    report("the 300 rows of 2026-11-18", [r for r in rows if int(r["review_order"]) > FIRST_DRAW])
    report("the 57 rows of 2026-11-13", [r for r in rows if int(r["review_order"]) <= FIRST_DRAW])


if __name__ == "__main__":
    main()
