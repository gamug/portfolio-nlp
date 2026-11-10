"""Track the company universe in ``gold/universe.csv`` (T-147: the matcher features need it; owner 2026-11-10).

    uv run python scripts/export_universe_2026_11_10.py [--counts data/company_news_counts/company_news_counts_2026-11-09.csv]

The universe is the 507 tickers of the per-company counts (T-160): ``ticker``, ``company``, ``also_covers`` and
``gics_sector``. It carries no news text and no counts, only the names the matcher reads, so the measurement of T-147 can
be reproduced from ``gold/`` alone.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

COLUMNS = ("ticker", "company", "also_covers", "gics_sector")


def export(counts: Path, out: Path) -> int:
    with counts.open(encoding="utf-8-sig", newline="") as fh:
        rows = [{c: r[c] for c in COLUMNS} for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"{counts}: no rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("data/company_news_counts/company_news_counts_2026-11-09.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("gold/universe.csv"))
    args = parser.parse_args()
    print(f"{args.out}: {export(args.counts, args.out)} companies")


if __name__ == "__main__":
    main()
