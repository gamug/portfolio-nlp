"""Check that the sector step as it is now gives the verdicts the clean test saw (2026-11-13; answers the review of #179).

    uv run scripts/check_sector_validation_2026_11_13.py [--gold gold] [--tolerance 0.0001]

The ``fresh`` rows of ``gold/sector_validation.csv`` record what the step said when the owner answered (``status``,
``candidate``, which is the sector when assigned, ``vote_sector``, ``margin``) under the revision in ``step_revision``. That revision hashed the raw bytes of the module, so a
docstring edit changed it; the revision is now of the code as parsed. This script re-scores the texts the owner read (``gold/texts``, minus the
truncation mark the draw added) with the code as it is, and says whether every row comes out the same, which is what lets the answers be
read as answers about the current revision. Needs SOURCE for the companies (the universe), not for the texts.
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
from sector_stage import SectorClassifier

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
    results = classifier.classify(items)
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
        ) and abs(res.margin - float(row["margin"])) <= args.tolerance
        if not same:
            differ.append(
                (row["article_id"], row["status"], res.status, row["margin"], f"{res.margin:.4f}")
            )
    print(f"recorded under {rows[0]['step_revision']}; the code now gives {classifier.revision}")
    print(
        f"{len(rows) - len(differ)} of {len(rows)} rows come out the same (status, sector, candidate, vote, margin within {args.tolerance})"
    )
    for d in differ:
        print("  differs:", d)
    if differ:
        sys.exit(1)


if __name__ == "__main__":
    main()
