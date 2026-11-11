"""What the owner's faithfulness marks say about `units4` (T-176): for the bodies the owner already read, cut with `units` and
`units4` and look up each text in the marks.

    uv run scripts/units4_vs_owner_marks_2026_11_11.py --source-db <urls.db copy> --data-dir data --data-dir data/faithfulness_confirmation \
        [--model-dir data/models/attribution_2026_11_08]

The owner marked (faithful yes/no) the texts of the faithfulness draws (``faithfulness_to_tag.csv``, with the text hash in
``faithfulness_ids.csv``). A mark belongs to a text, so a `units4` text equal to a marked text is already read; a company that
`units` gave a text and `units4` does not ("lost") shows whether the text left out was faithful (the owner said yes: a real loss)
or not (a text that should not exist). Reads the files and the source copy; writes nothing. The marks are those of 2026-11-08/10: a
`units` text the current code writes differently has no mark ("stale"), and is counted apart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEXT_CAP = 6000  # what the owner read (draw_sector_extra.TEXT_CAP)


def text_hash(text: str) -> str:
    """The identity of a text a mark belongs to (``retag.text_hash``)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def read_marks(data_dir: Path) -> dict[tuple[str, str, str], str]:
    """``{(article_id, ticker, text_hash): "yes" | "no"}`` of one draw's tagging file."""
    with (data_dir / "faithfulness_ids.csv").open(encoding="utf-8-sig") as f:
        ids = list(csv.DictReader(f))
    with (data_dir / "faithfulness_to_tag.csv").open(encoding="utf-8-sig") as f:
        tag = {r["review_order"]: r["faithful"].strip().lower() for r in csv.DictReader(f)}
    return {
        (r["article_id"], r["ticker"], r["text_hash"]): tag[r["review_order"]]
        for r in ids
        if tag.get(r["review_order"]) in ("yes", "no")
    }


def classify(
    a: str | None, b: str | None, marks: dict[tuple[str, str, str], str], article: str, ticker: str
) -> tuple[str, str | None, str | None]:
    """``(outcome, mark of A, mark of B)``: outcome is ``same``, ``lost``, ``gained`` or ``changed``."""

    def mark(text: str | None) -> str | None:
        return None if not text else marks.get((article, ticker, text_hash(text[:TEXT_CAP])))

    if a and not b:
        return "lost", mark(a), None
    if b and not a:
        return "gained", None, mark(b)
    if a == b:
        return "same", mark(a), mark(b)
    return "changed", mark(a), mark(b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/models/attribution_2026_11_08",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the lost and the unread changed texts here, to read",
    )
    args = parser.parse_args()
    to_read: list[list[str]] = []
    from cut_stage import UnitsExtractor  # noqa: PLC0415 -- loads torch
    from news_nlp import attribution_model as am  # noqa: PLC0415
    from news_nlp.triage import Company, CompanyResolver  # noqa: PLC0415

    src = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    rows = src.execute(
        "SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM articles WHERE fetch_status = 'ok' ORDER BY ticker, company"
    ).fetchall()
    companies = [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]
    resolver = CompanyResolver(companies)
    first = UnitsExtractor.load(args.model_dir, companies, "units")
    second = UnitsExtractor(first.scorer, companies, "local:x", "units4")
    for data_dir in args.data_dir:
        marks = read_marks(data_dir)
        articles = sorted({k[0] for k in marks})
        count: Counter[str] = Counter()
        for article in articles:
            row = src.execute(
                "SELECT body_text FROM articles WHERE id = ?", (int(article),)
            ).fetchone()
            if not row or not row[0]:
                count["body_missing"] += 1
                continue
            body = row[0]
            ta = am.model_company_texts(
                body, first.resolver, first.names, first.scorer, reading="units"
            )
            tb = am.model_company_texts(
                body, second.resolver, second.names, second.scorer, reading="units4"
            )
            for ticker in sorted(resolver.resolve(body, skip_attribution=False)):
                outcome, ma, mb = classify(ta.get(ticker), tb.get(ticker), marks, article, ticker)
                if not ta.get(ticker) and not tb.get(ticker):
                    continue
                count[f"{outcome}"] += 1
                count[f"{outcome}: units mark {ma or 'none (stale or unread)'}"] += 1
                if outcome in ("changed", "gained"):
                    count[f"{outcome}: units4 text {'marked ' + mb if mb else 'UNREAD'}"] += 1
                if outcome == "lost" or (outcome in ("changed", "gained") and mb is None):
                    to_read.append(
                        [
                            outcome,
                            article,
                            ticker,
                            ma or "",
                            ta.get(ticker, "")[:TEXT_CAP],
                            tb.get(ticker, "")[:TEXT_CAP],
                        ]
                    )
        print(f"== {data_dir} ({len(articles)} bodies, {len(marks)} marks)")
        for k in sorted(count):
            print(f"  {k}: {count[k]}")

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["outcome", "article_id", "ticker", "units_mark", "text_units", "text_units4"]
            )
            writer.writerows(to_read)
        print(f"wrote {len(to_read)} rows to {args.out}")


if __name__ == "__main__":
    main()
