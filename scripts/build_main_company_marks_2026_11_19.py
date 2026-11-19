"""Track the owner's second version of the main-company marks (``gold/main_company_marks_2026_11_19.csv`` and its texts).

    uv run scripts/build_main_company_marks_2026_11_19.py [--data data] [--gold gold]

Reads the owner's working file ``data/main_company/to_tag.csv`` (marked 2026-11-19) with ``to_tag.strata.json`` and writes the marked rows next to the
2026-11-16 version, which is not touched (a new mark set is a new build and a new row of ``docs/main_company_training.md``, never an edit). It reuses
``build_val_gold_2026_11_16.main_company_rows`` (the same columns, text hash and checks), refuses to overwrite, and prints how the new set differs
from the old one: rows kept, rows and pieces no longer in the file, new rows, and marks that changed. It also prints the SHA-256 of what it read.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_val_gold_2026_11_16 as old

MARKS = "main_company_marks_2026_11_19.csv"
TEXTS = "main_company_marks_texts_2026_11_19.csv"
PREVIOUS = "main_company_marks.csv"


def compare(new: list[dict[str, str]], previous: list[dict[str, str]]) -> dict[str, int]:
    """How the new marks differ from the earlier ones, by key (``item_id#ticker``)."""
    a = {r["key"]: r for r in previous}
    b = {r["key"]: r for r in new}
    gone = set(a) - set(b)
    return {
        "kept": len(set(a) & set(b)),
        "changed": sum(a[k]["es_principal"] != b[k]["es_principal"] for k in set(a) & set(b)),
        "rows_gone": len(gone),
        "pieces_gone": len({k.split("#")[0] for k in gone} - {k.split("#")[0] for k in b}),
        "new_rows": len(set(b) - set(a)),
        "new_pieces": len({k.split("#")[0] for k in b} - {k.split("#")[0] for k in a}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    args = parser.parse_args()
    marks_path, texts_path = args.gold / MARKS, args.gold / "texts" / TEXTS
    for path in (marks_path, texts_path):
        if path.exists():
            raise SystemExit(f"{path} exists: nothing in gold/ is overwritten")
    marks, texts = old.main_company_rows(args.data)
    old.check_unique(marks, "main_company_marks_2026_11_19")
    old.write(marks_path, old.MARK_COLUMNS, marks)
    old.write(texts_path, old.MARK_TEXT_COLUMNS, texts)
    yes = sum(r["es_principal"] == "yes" for r in marks)
    print(
        f"{marks_path}: {len(marks)} rows of {len({r['item_id'] for r in marks})} pieces, {yes} yes, {len(marks) - yes} no"
    )
    previous = args.gold / PREVIOUS
    if previous.exists():
        print("against", previous, compare(marks, old.read(previous)))
    for path in (
        args.data / "main_company" / "to_tag.csv",
        args.data / "main_company" / "to_tag.strata.json",
        marks_path,
        texts_path,
    ):
        print(f"{path}: sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
