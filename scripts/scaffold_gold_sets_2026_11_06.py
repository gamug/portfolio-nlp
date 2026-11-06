"""Write the empty files of the gold sets that are not drawn yet (``docs/gold_datasets.md``).

    uv run scripts/scaffold_gold_sets_2026_11_06.py [--out data] [--gold gold]

For the label check, the extractor sample (faithfulness) and the sentiment sets it writes a
header-only working file in ``--out`` (the exact columns the draw script writes, ``retag``) and a
header-only tracked file in ``--gold`` (``retag.*_GOLD_COLUMNS``). A file that exists is never
touched. The draw scripts read a header-only working file as not drawn yet and fill it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import retag

#: set -> (working file, its columns, gold file, its columns)
SETS = {
    "label check": (
        "segment_check_to_tag.csv",
        retag.SEGMENT_CHECK_COLUMNS,
        "segment_check_gold.csv",
        retag.SEGMENT_CHECK_GOLD_COLUMNS,
    ),
    "extractor sample": (
        "faithfulness_to_tag.csv",
        retag.FAITHFULNESS_COLUMNS,
        "faithfulness_gold.csv",
        retag.FAITHFULNESS_GOLD_COLUMNS,
    ),
    "sentiment": (
        "sentiment_gold_to_tag.csv",
        retag.SENTIMENT_TAGGING_COLUMNS,
        "sentiment_gold.csv",
        retag.SENTIMENT_GOLD_COLUMNS,
    ),
}


def scaffold(path: Path, columns: tuple[str, ...]) -> bool:
    """Write the header-only *path*; False (nothing touched) when it exists."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerow(columns)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    args = parser.parse_args()
    for name, (work, work_cols, gold, gold_cols) in SETS.items():
        for path, cols in ((args.out / work, work_cols), (args.gold / gold, gold_cols)):
            print(f"{name}: {path} {'written' if scaffold(path, cols) else 'exists, left alone'}")


if __name__ == "__main__":
    main()
