"""Put the owner's tagging sessions of the prompt gate in one tracked gold file (T-146c).

    uv run scripts/build_gold_2026_11_05.py [--out data] [--gold gold/category_gold_v2.csv]

Reads ``data/gate240_to_tag.csv`` (the fixed 240) and ``data/sector_extra_to_tag.csv`` (44 sector
bodies), checks every row with ``news_nlp.retag`` (the file is refused unless all rows are tagged
and valid, and no article is in both) and writes ``gold/category_gold_v2.csv``: the tags only,
sorted by ``article_id`` (``retag.GOLD_COLUMNS``). The article text and its Spanish aid stay out of
git (news text, public repository); read it back from the SOURCE store by ``article_id``. The gold
file is rewritten from the two tagging files each time.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import retag_gold_2026_10_29 as retag_cli

from news_nlp import retag

SESSIONS = {"gate240": "gate240_to_tag.csv", "sector_extra": "sector_extra_to_tag.csv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/category_gold_v2.csv"))
    args = parser.parse_args()
    universe = retag_cli.load_universe()
    sessions: dict[str, list[dict[str, str]]] = {}
    golds: dict[str, list[retag.Gold]] = {}
    for session, name in SESSIONS.items():
        try:
            rows = retag.read_tag_file(args.out / name)
        except retag.TagFileError as exc:
            sys.exit(str(exc))
        ids = [int(r["article_id"]) for r in rows]
        report = retag.load_gold_v2(args.out / name, ids, universe)
        if report.bad or report.untagged or report.duplicated or report.unknown:
            sys.exit(
                f"{name}: not ready for the gold file (errors {sorted(report.bad)}, "
                f"untagged {report.untagged}, duplicated {report.duplicated}, "
                f"unknown {report.unknown}); run the check command"
            )
        sessions[session], golds[session] = rows, report.good
        print(f"{name}: {len(report.good)} rows tagged and valid")
    try:
        out_rows = retag.gold_file_rows(sessions, golds)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    with args.gold.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(retag.GOLD_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"{args.gold}: {len(out_rows)} rows")


if __name__ == "__main__":
    main()
