"""Check that the sector step as it is now gives the verdicts the clean test saw (2026-11-13; answers the review of #179).

    uv run scripts/check_sector_validation_2026_11_13.py [--gold gold] [--tolerance 0.0001]

The ``fresh`` rows of ``gold/sector_validation.csv`` record what the step said when the owner answered (``status``,
``candidate``, which is the sector when assigned, ``vote_sector``, ``margin``) under the revision in ``step_revision``. That revision hashed the raw bytes of the module, so a
docstring edit changed it; the revision is now of the code as parsed. This script re-scores the texts the owner read (``gold/texts``, minus the
truncation mark the draw added) with the code as it is, and says whether every row comes out the same, which is what lets the answers be
read as answers about the current revision.

**Two comparisons, because the texts differ.** The draw and the owner's tracked texts are ``body[:6000]`` (``TEXT_CAP``); the step reads the **whole
body**. The first comparison (capped text against the recorded verdicts) is the one that must be exact: it exits 1 if a row differs. The second reads
each body whole from SOURCE (by article id) and lists the rows whose verdict changes, with the owner's answer: that is what the step would do, and it
says whether the owner's answers still apply to it. Needs SOURCE (the companies, and the whole bodies).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra

import news_nlp as db
from cut_stage import companies_of
from sector_stage import SectorClassifier, SectorResult

csv.field_size_limit(sys.maxsize)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    texts = {r["article_id"]: r for r in read(args.gold / "texts" / "sector_validation_texts.csv")}
    rows = [r for r in read(args.gold / "sector_validation.csv") if r["review"] == "fresh"]
    classifier = SectorClassifier(companies_of(db.connect_pipeline()))
    items = [
        (
            texts[r["article_id"]]["title"],
            texts[r["article_id"]]["text"].removesuffix(extra.STAGE_TEXT),
        )
        for r in rows
    ]
    differ = compare(rows, classifier.classify(items), args.tolerance)
    print(f"recorded under {rows[0]['step_revision']}; the code now gives {classifier.revision}")
    print(
        f"capped text ({extra.TEXT_CAP} characters, as the owner read it): {len(rows) - len(differ)} of {len(rows)} rows come out the same "
        f"(status, candidate, vote, margin within {args.tolerance})"
    )
    for d in differ:
        print("  differs:", d)
    whole = whole_bodies(rows)
    long_ids = [r["article_id"] for r in rows if len(whole[r["article_id"]][1]) > extra.TEXT_CAP]
    results_whole = classifier.classify([whole[r["article_id"]] for r in rows])
    moved = [
        (r["article_id"], r["status"], w.status, r["correct"])
        for r, w in zip(rows, results_whole, strict=True)
        if (w.status, w.sector)
        != (r["status"], r["candidate"] if r["status"] == "assigned" else "")
    ]
    print(
        f"whole body (what the step reads): {len(long_ids)} of {len(rows)} bodies are longer than the cap; "
        f"{len(moved)} verdict(s) differ from the recorded ones"
    )
    for article_id, recorded, now, correct in moved:
        print(
            f"  article {article_id}: recorded {recorded}, whole body {now}; the owner answered correct = {correct}"
        )
    assigned_now = sum(
        1
        for r, w in zip(rows, results_whole, strict=True)
        if r["status"] == "assigned" and r["control"] == "no" and w.status == "assigned"
    )
    assigned_then = sum(1 for r in rows if r["status"] == "assigned" and r["control"] == "no")
    print(
        f"the recorded assigned bodies still assigned on the whole body: {assigned_now} of {assigned_then}"
    )
    if differ:
        sys.exit(1)


def compare(
    rows: list[dict[str, str]], results: list[SectorResult], tolerance: float
) -> list[tuple[str, ...]]:
    """The rows whose verdict is not the recorded one (status, candidate, vote, margin within ``tolerance``)."""
    differ = []
    for row, res in zip(rows, results, strict=True):
        sector = (
            row["candidate"] if row["status"] == "assigned" else ""
        )  # the file keeps the sector as the candidate
        same = (res.status, res.sector, res.candidate, res.vote_sector) == (
            row["status"],
            sector,
            row["candidate"],
            row["vote_sector"],
        ) and abs(res.margin - float(row["margin"])) <= tolerance
        if not same:
            differ.append(
                (row["article_id"], row["status"], res.status, row["margin"], f"{res.margin:.4f}")
            )
    return differ


def whole_bodies(rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    """``{article_id: (title, whole body)}`` from SOURCE for the rows of the file."""
    conn = db.connect_pipeline()
    ids = [int(r["article_id"]) for r in rows]
    marks = ", ".join("?" for _ in ids)
    found = conn.execute(
        f"SELECT id, title, body_text FROM source.articles WHERE id IN ({marks})",  # noqa: S608 -- placeholders only
        ids,
    ).fetchall()
    return {str(i): (t or "", b or "") for i, t, b in found}


if __name__ == "__main__":
    main()
