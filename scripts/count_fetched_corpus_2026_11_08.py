"""Count the properly fetched corpus (Work item 18, T-159, PLAN D36).

    uv run scripts/count_fetched_corpus_2026_11_08.py [--since 2021-12-31] [--out data/fetched_corpus_count.json]

``quarantined`` is a verdict on the (article, ticker) pair, not on the article (D36), so the corpus the
pipeline could read is larger than the ``ok`` one. This counts it, read-only, with no model and no
spend: the SOURCE ``articles`` (``$SOURCE_DATABASE_URL``) and, for the pair class, ``discovered_urls``.

A row is **properly fetched** when its ``fetch_status`` is ``ok`` or ``quarantined`` and it has a title
and a ``body_text`` (the owner, 2026-11-08: ``thin_content`` rows are not part of it; they are counted
apart). The unit is the unique body (``triage.body_hash``, whitespace collapsed), as everywhere else:
the same text stored once per tagged ticker is one body. It prints, and writes to ``--out`` as JSON:

* every row by ``fetch_status`` with how many have a title, a body and both (the rows the rest leaves out),
  and the rows with both by ``fetch_status`` and by pair class (``accept``, ``reject``, ``review``, none);
* the unique bodies of the ``ok`` rows, of the quarantined rows, the **quarantined-only** ones (a body
  that is not also ``ok`` under another ticker: the corpus the pipeline would gain) and the union;
* the same for the window from ``--since`` (the cut's: 2021-12-31) and by publication year;
* the words of the quarantined-only bodies, counted from the text (the stored ``word_count`` is not
  used: it may be empty), to see how much of it is short;
* by the tagged ticker, how many companies have an ``ok`` body and how many have only quarantined ones
  (the tag of a quarantined pair is not evidence about the company: counted for size only).

It reads only; the tables are never written.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import news_nlp as db
from news_nlp import triage

PROPER = ("ok", "quarantined")  # the fetch_status values of a properly fetched row
YEAR_DIGITS = 4
SHORT_WORDS, LONG_ENOUGH_WORDS = 100, 300  # the thresholds the word counts are read against


class Row(NamedTuple):
    """One stored article with what the count needs; ``body`` is the hash of its text."""

    status: str
    pair_class: str
    ticker: str
    year: str
    in_window: bool
    body: bytes
    words: int
    has_title: bool
    has_body: bool


def year_of(pub_date: str | None) -> str:
    """The year of a ``pub_date`` that starts with four digits, else ``unknown``."""
    text = (pub_date or "").strip()
    return text[:YEAR_DIGITS] if text[:YEAR_DIGITS].isdigit() else "unknown"


def short_hash(text: str) -> bytes:
    """The first 16 bytes of ``triage.body_hash``: enough to tell bodies apart, half the memory."""
    return bytes.fromhex(triage.body_hash(text))[:16]


def read_rows(conn: object, since: str) -> Iterable[Row]:
    """Every row of ``source.articles`` with its pair class, and whether it has a title and a body
    (``summarise`` counts the rows it leaves out, so that no figure about them is a separate query).
    *conn* is a connection with the schema ``source`` attached (``db.connect_pipeline()``)."""
    execute = conn.execute  # type: ignore[attr-defined]
    classes = {int(i): c for i, c in execute("SELECT id, match_class FROM source.discovered_urls")}
    for article_id, status, ticker, title, body, pub_date in execute(
        "SELECT id, fetch_status, ticker, title, body_text, pub_date FROM source.articles"
    ):
        date = (pub_date or "").strip()
        has_title, has_body = bool(title and title.strip()), bool(body and body.strip())
        yield Row(
            status=status or "",
            pair_class=classes.get(int(article_id)) or "none",
            ticker=ticker or "",
            year=year_of(date),
            in_window=bool(date) and date[:10] >= since,
            body=short_hash(body) if has_body else b"",
            words=len(body.split()) if has_body else 0,
            has_title=has_title,
            has_body=has_body,
        )


def summarise(rows: Iterable[Row]) -> dict[str, object]:
    """The counts of the module docstring from *rows*, every stored row: those without a title or
    without a body are counted by status (``rows_by_status_and_columns``) and then left out of the rest."""
    rows_by_status: Counter[str] = Counter()
    rows_by_class: Counter[tuple[str, str]] = Counter()
    bodies: dict[str, set[bytes]] = {"ok": set(), "quarantined": set(), "thin_content": set()}
    window: dict[str, set[bytes]] = {"ok": set(), "quarantined": set(), "thin_content": set()}
    pair_bodies: dict[str, set[bytes]] = {}
    year_of_body: dict[bytes, str] = {}
    words_of_body: dict[bytes, int] = {}
    ok_tickers: set[str] = set()
    quarantined_tickers: set[str] = set()
    columns: dict[str, Counter[str]] = {}
    for row in rows:
        counts = columns.setdefault(row.status, Counter())
        counts["rows"] += 1
        counts["with_title"] += row.has_title
        counts["with_body"] += row.has_body
        counts["with_both"] += row.has_title and row.has_body
        if not (row.has_title and row.has_body):
            continue
        rows_by_status[row.status] += 1
        rows_by_class[(row.status, row.pair_class)] += 1
        pair_bodies.setdefault(row.pair_class, set()).add(row.body)
        if row.status in bodies:
            bodies[row.status].add(row.body)
            if row.in_window:
                window[row.status].add(row.body)
        if row.status == "ok":
            ok_tickers.add(row.ticker)
        elif row.status == "quarantined":
            quarantined_tickers.add(row.ticker)
        year_of_body.setdefault(row.body, row.year)
        words_of_body.setdefault(row.body, row.words)
    only = bodies["quarantined"] - bodies["ok"]
    only_window = window["quarantined"] - window["ok"]
    words = sorted(words_of_body[b] for b in only)
    by_year = Counter(year_of_body[b] for b in only)
    return {
        "rows_by_fetch_status": dict(rows_by_status),
        "rows_by_status_and_columns": {s: dict(c) for s, c in sorted(columns.items())},
        "rows_by_status_and_pair_class": {
            f"{s}/{c}": n for (s, c), n in sorted(rows_by_class.items())
        },
        "unique_bodies": {
            "ok": len(bodies["ok"]),
            "quarantined": len(bodies["quarantined"]),
            "quarantined_only": len(only),
            "properly_fetched": len(bodies["ok"] | bodies["quarantined"]),
            "thin_content_apart": len(bodies["thin_content"]),
        },
        "unique_bodies_in_window": {
            "ok": len(window["ok"]),
            "quarantined": len(window["quarantined"]),
            "quarantined_only": len(only_window),
            "properly_fetched": len(window["ok"] | window["quarantined"]),
        },
        "quarantined_only_by_year": dict(sorted(by_year.items())),
        "quarantined_only_words": {
            "median": statistics.median(words) if words else 0,
            f"under_{SHORT_WORDS}": sum(w < SHORT_WORDS for w in words),
            f"under_{LONG_ENOUGH_WORDS}": sum(w < LONG_ENOUGH_WORDS for w in words),
        },
        "companies_by_tag": {
            "with_an_ok_body": len(ok_tickers - {""}),
            "only_quarantined": len(quarantined_tickers - ok_tickers - {""}),
        },
    }


def report_lines(result: dict[str, object], since: str) -> list[str]:
    """The result as text, every figure naming the corpus it is of (D36)."""
    ub = result["unique_bodies"]
    uw = result["unique_bodies_in_window"]
    assert isinstance(ub, dict) and isinstance(uw, dict)
    columns = result["rows_by_status_and_columns"]
    assert isinstance(columns, dict)
    return [
        "all rows by fetch_status (rows / with a title / with a body / with both): "
        + "; ".join(
            f"{k} {v['rows']:,} / {v['with_title']:,} / {v['with_body']:,} / {v['with_both']:,}"
            for k, v in columns.items()
        ),
        "rows with a title and a body, by fetch_status: "
        + ", ".join(f"{k} {v:,}" for k, v in sorted(dict(result["rows_by_fetch_status"]).items())),  # type: ignore[call-overload]
        f"unique bodies, whole corpus: ok {ub['ok']:,}; quarantined {ub['quarantined']:,}; "
        f"quarantined and not also ok {ub['quarantined_only']:,}; properly fetched (ok or "
        f"quarantined) {ub['properly_fetched']:,}; thin_content, apart: {ub['thin_content_apart']:,}",
        f"unique bodies, published {since} or later (the cut's window): ok {uw['ok']:,}; "
        f"quarantined {uw['quarantined']:,}; quarantined and not also ok {uw['quarantined_only']:,}; "
        f"properly fetched {uw['properly_fetched']:,}",
        f"quarantined-only bodies by year: {result['quarantined_only_by_year']}",
        f"quarantined-only bodies, words: {result['quarantined_only_words']}",
        f"companies by tag: {result['companies_by_tag']}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", default="2021-12-31", help="the cut's window starts here")
    parser.add_argument("--out", type=Path, default=Path("data/fetched_corpus_count.json"))
    args = parser.parse_args()
    db.env.load_env_file()
    conn = db.connect_pipeline()
    result = summarise(read_rows(conn, args.since))
    result["since"] = args.since
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(report_lines(result, args.since)))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
