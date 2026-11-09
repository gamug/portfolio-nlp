"""News counts per company over the properly fetched corpus (Work item 18, T-160, PLAN D36).

    uv run scripts/company_news_counts_2026_11_09.py [--limit N] [--out data/company_news_counts]

Read-only, CPU only, no model and no spend. It reads the SOURCE ``articles`` that were properly
fetched (``fetch_status`` ``ok`` or ``quarantined``, with a title and a body), groups them into unique
bodies (``news_nlp.triage.body_hash``, the unit of D11) and reads each body with the rules of
``size_triage_2026_10_28.py`` (the name resolver, boilerplate and analyst attribution left out, the
subject rule, the round-up structure). The tagged ticker of a pair is not used: a quarantined pair is a
verdict on the tag, not on the article (D36), so every count is of what the text names.

It writes two files to ``--out``, for the S&P universe over every date of the corpus:

* ``company_news_counts.csv``: one row per company, UTF-8 with a byte-order mark (Excel opens it), no
  article text. The counts are of unique bodies, in three corpora: the bodies with an ``ok`` pair,
  the quarantined-only bodies (no ``ok`` pair under any ticker), and both together. For each: the
  bodies that **name** the company, the bodies it is a **subject** of (named twice or in the title),
  and the bodies it is the **only** subject of (scope ``single``).
* ``company_news_counts_README.txt``: the date, the corpus each column is of, the totals and the limits.

A ``tqdm`` bar shows the rows read, the speed and the time left (``--no-progress`` turns it off for a
log file). **The run is idempotent and resumable.** The same command can be run again at any time: the
files are written whole to a temporary name and then renamed (a run that stops leaves the previous
files, never half of them), and the state (every body read so far and the last row ``id``) is saved to
``company_news_counts.checkpoint.json`` every 25,000 rows, on Ctrl-C and at the end. Run again, it
continues after the last row saved, so an interrupted run loses at most 25,000 rows and a run on an
unchanged store takes seconds and rewrites the same files; after the crawl adds rows it reads only the
new ones. The checkpoint is used only if the code (this script, the rules) and the company universe
are the same; otherwise it is ignored and the run starts again. ``--fresh`` ignores it on purpose (use
it if rows already read may have changed, for example a pair that became ``ok``).

``--limit N`` stops after N rows of this run (a smoke run) and prints how long the whole run would
take at that speed. The stored NER ``ORG`` spans of the first pair of a body help the resolver, as in
the size script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import size_triage_2026_10_28 as size

import news_nlp as db
from news_nlp import retag, triage
from news_nlp.triage import Company, CompanyResolver

CHECKPOINT_EVERY = 25_000
CHECKPOINT_NAME = "company_news_counts.checkpoint.json"
CSV_NAME, README_NAME = "company_news_counts.csv", "company_news_counts_README.txt"
CORPORA = ("ok", "quarantined_only", "all")
MEASURES = ("naming", "subject", "single")


class BodyReading(NamedTuple):
    """What the rules find in one unique body."""

    named: frozenset[str]  # canonical tickers named anywhere in the text
    subjects: frozenset[str]  # named twice or in the title
    scope: (
        str  # single | multi | multi_dominated | roundup | market | economy | sector | no_company
    )


class Entry:
    """One unique body: its reading, and whether any of its pairs is ``ok``."""

    __slots__ = ("is_ok", "reading", "roundup_by_headline")

    def __init__(self, reading: BodyReading, *, is_ok: bool, roundup_by_headline: bool) -> None:
        self.reading = reading
        self.is_ok = is_ok
        self.roundup_by_headline = roundup_by_headline


def read_body(
    resolver: CompanyResolver,
    title: str,
    body: str,
    slug: str,
    spans: list[str],
    terms: list[str],
) -> BodyReading:
    """The names, the subjects and the scope of one body, with every rule step of the size script."""
    text = f"{title}\n{triage.strip_boilerplate(body)}"
    named = resolver.resolve(text, spans, skip_attribution=True)
    in_title = resolver.resolve(title, skip_attribution=True)
    subjects = triage.subjects(named, in_title)
    roundup = triage.is_roundup(title, slug, len(in_title), resolver.ticker_forms(text))
    scope = triage.scope_candidate(subjects, text, terms, roundup=roundup)
    return BodyReading(frozenset(named), frozenset(subjects), scope.scope)


def load_universe(conn: db.NewsNlpDatabase) -> list[Company]:
    """The tickers of the properly fetched SOURCE rows, ``ok`` or ``quarantined``: a company that has
    only quarantined rows is still in the universe."""
    rows = conn.execute(
        "SELECT ticker, company, gics_sector, gics_sub_industry FROM source.articles "
        "WHERE fetch_status IN ('ok', 'quarantined') AND ticker IS NOT NULL AND ticker != '' "
        "GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    return [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]


def stream_rows(
    conn: db.NewsNlpDatabase, limit: int, after_id: int = 0
) -> Iterable[tuple[int, str, str, str, str, str]]:
    """``(id, fetch_status, ticker, title, body, url)`` of every properly fetched row with an ``id``
    above *after_id*, by id."""
    yield from conn.execute(
        "SELECT a.id, a.fetch_status, a.ticker, a.title, a.body_text, d.url FROM source.articles a "
        "LEFT JOIN source.discovered_urls d ON d.id = a.id "
        "WHERE a.fetch_status IN ('ok', 'quarantined') "
        "AND a.title IS NOT NULL AND trim(a.title) != '' "
        "AND a.body_text IS NOT NULL AND trim(a.body_text) != '' "
        "AND a.id > ? ORDER BY a.id LIMIT ?",
        (after_id, limit or -1),
    )


def read_corpus(
    rows: Iterable[tuple[int, str, str, str, str, str]],
    resolver: CompanyResolver,
    terms: list[str],
    spans_of: Callable[[int], list[str]],
    *,
    entries: dict[str, Entry] | None = None,
    after_row: Callable[[int, int], None] = lambda _id, _bodies: None,
) -> dict[str, Entry]:
    """Every unique body of *rows* read once, added to *entries* (the state of an earlier run, if any).
    *spans_of* gives the stored ``ORG`` spans of an article id (a query on the RESULTS store).
    A body seen again under another pair only updates ``is_ok`` and, as in the size script, is read
    again with the first headline that reads as a round-up if the first reading was not one; both are
    idempotent, so a row read twice (after a stop) changes nothing. *after_row* is called with the
    row's id and the number of bodies once the row is fully read: the bar and the checkpoint hang
    from it."""
    entries = {} if entries is None else entries
    for article_id, status, _ticker, title, body, url in rows:
        key = triage.body_hash(body)
        slug = size.slug_of(url)
        entry = entries.get(key)
        headline_roundup = triage.is_roundup(title, slug, 0, 0)
        if entry is None:
            reading = read_body(resolver, title, body, slug, spans_of(article_id), terms)
            entries[key] = Entry(
                reading, is_ok=status == "ok", roundup_by_headline=headline_roundup
            )
        else:
            entry.is_ok = entry.is_ok or status == "ok"
            if (
                headline_roundup
                and not entry.roundup_by_headline
                and entry.reading.scope != "roundup"
            ):
                entry.reading = read_body(resolver, title, body, slug, spans_of(article_id), terms)
                entry.roundup_by_headline = True
        after_row(article_id, len(entries))
    return entries


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* whole: to a temporary name, then renamed over *path* (``os.replace`` is atomic on
    Windows and Linux), so a stopped run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding=encoding, newline="")
    os.replace(temporary, path)


def fingerprint(universe: list[Company]) -> str:
    """What a checkpoint must match: the code (this script, the rules) and the company universe."""
    names = "\n".join(f"{c.ticker}|{c.name}" for c in sorted(universe))
    return f"{code_revision()}:{hashlib.sha256(names.encode()).hexdigest()[:12]}"


def save_checkpoint(
    path: Path, print_of: str, last_id: int, rows_read: int, entries: dict[str, Entry]
) -> None:
    state: dict[str, Any] = {
        "fingerprint": print_of,
        "last_id": last_id,
        "rows_read": rows_read,
        "entries": [
            [
                key,
                sorted(e.reading.named),
                sorted(e.reading.subjects),
                e.reading.scope,
                e.is_ok,
                e.roundup_by_headline,
            ]
            for key, e in entries.items()
        ],
    }
    atomic_write_text(path, json.dumps(state, separators=(",", ":")))


def load_checkpoint(path: Path, print_of: str) -> tuple[int, int, dict[str, Entry]] | str:
    """``(last_id, rows_read, entries)`` of a checkpoint that matches *print_of*, or the reason it
    is not used (a ``str``)."""
    if not path.exists():
        return "no checkpoint"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        saved = state["fingerprint"]
        if saved != print_of:
            return f"the code or the universe changed ({saved} -> {print_of})"
        entries = {
            key: Entry(
                BodyReading(frozenset(named), frozenset(subjects), scope),
                is_ok=is_ok,
                roundup_by_headline=roundup,
            )
            for key, named, subjects, scope, is_ok, roundup in state["entries"]
        }
        return int(state["last_id"]), int(state["rows_read"]), entries
    except (ValueError, KeyError, TypeError) as error:
        return f"the checkpoint is not readable ({error})"


def tally(entries: Iterable[Entry]) -> dict[str, Counter[str]]:
    """``{ticker: {column: bodies}}`` for the columns ``{corpus}_bodies_{measure}``."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        reading = entry.reading
        for corpus in ("ok" if entry.is_ok else "quarantined_only", "all"):
            for ticker in reading.named:
                counts[ticker][f"{corpus}_bodies_naming"] += 1
            for ticker in reading.subjects:
                counts[ticker][f"{corpus}_bodies_subject"] += 1
                if reading.scope == "single":
                    counts[ticker][f"{corpus}_bodies_single"] += 1
    return counts


def count_by_company(
    entries: Iterable[Entry], resolver: CompanyResolver, universe: list[Company]
) -> list[dict[str, object]]:
    """One row per canonical company: ``{corpus}_bodies_{measure}`` for the three corpora and the
    three measures. A company that shares a canonical ticker with another (share classes) is one row,
    listing the other tickers in ``also_covers``."""
    counts = tally(entries)
    first: dict[str, Company] = {}
    covers: dict[str, list[str]] = defaultdict(list)
    for company in universe:
        canonical = resolver.by_ticker.get(company.ticker.upper(), company.ticker)
        first.setdefault(canonical, company)
        if company.ticker != canonical:
            covers[canonical].append(company.ticker)
    rows: list[dict[str, object]] = []
    for canonical, company in sorted(first.items()):
        row: dict[str, object] = {
            "ticker": canonical,
            "company": company.name,
            "also_covers": " ".join(sorted(covers[canonical])),
            "gics_sector": company.gics_sector,
        }
        for corpus, measure in product(CORPORA, MEASURES):
            column = f"{corpus}_bodies_{measure}"
            row[column] = counts[canonical][column]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["ticker", "company", "also_covers", "gics_sector"] + [
        f"{c}_bodies_{m}" for c in CORPORA for m in MEASURES
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), encoding="utf-8-sig")


def readme_text(
    entries: dict[str, Entry],
    rows: list[dict[str, object]],
    *,
    universe_size: int,
    row_total: int,
    new_rows: int,
    limit: int,
    revision: str,
    elapsed: float,
) -> str:
    ok = sum(e.is_ok for e in entries.values())
    named_any = sum(bool(e.reading.named) for e in entries.values())
    scopes = Counter(e.reading.scope for e in entries.values())
    smoke = (
        f"SMOKE RUN: stopped after {limit:,} rows of the run, not the whole corpus.\n\n"
        if limit
        else ""
    )
    return (
        smoke
        + f"company_news_counts.csv: news per company, S&P universe, every date of the corpus\n"
        f"Made {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by scripts/company_news_counts_2026_11_09.py "
        f"(code revision {revision}); {elapsed / 60:,.0f} min.\n\n"
        "WHAT A ROW IS\n"
        "One company (share classes of one company are one row, listed in also_covers). Every count "
        "is of UNIQUE BODIES: the same text stored under several tickers is one body.\n\n"
        "THE THREE CORPORA (every column says which one)\n"
        "  ok_*               bodies with at least one 'ok' pair (the corpus of the 2026-11-16 cut)\n"
        "  quarantined_only_* bodies whose pairs are all quarantined: fetched properly (a title and a "
        "body) but upstream could not tie them to the tagged company; read here for what they say\n"
        "  all_*              the two together (ok + quarantined_only), each body once\n\n"
        "THE THREE MEASURES\n"
        "  *_bodies_naming   the company is named somewhere in the text\n"
        "  *_bodies_subject  it is named at least twice, or in the title (a passing mention is not a "
        "subject)\n"
        "  *_bodies_single   it is the only subject of the body (a single-company article)\n\n"
        "THE TAG IS NOT USED. A quarantined pair's ticker says nothing about the article, so every "
        "count comes from the text.\n\n"
        "TOTALS OF THIS RUN\n"
        f"  rows read {row_total:,} ({new_rows:,} in the run that wrote this file, the rest from the "
        f"checkpoint of earlier runs); unique bodies {len(entries):,} ({ok:,} with an ok pair, "
        f"{len(entries) - ok:,} quarantined only); universe {universe_size} tickers, {len(rows)} rows.\n"
        f"  bodies that name at least one company: {named_any:,}; by rule scope: "
        f"{dict(scopes.most_common())}\n\n"
        "LIMITS (read the counts as a floor, with some noise)\n"
        "  - No alias table yet: a company named in another way (a product, a nickname) is missed.\n"
        "  - A name that is also a common word or a person's name can be counted by mistake.\n"
        "  - Boilerplate, data credits and an analyst's bank named as the source of an opinion are "
        "not counted.\n"
        "  - The rules were tuned on the owner's 720 tagged articles; this is not an accuracy figure.\n"
        "  - Bodies are the articles the crawl fetched properly, not all the news that exists.\n"
    )


def code_revision() -> str:
    """12 hex digits that change when this script, the rules or the size script change."""
    own = Path(__file__).read_bytes() + retag.code_revision(triage, size).encode()
    return hashlib.sha256(own).hexdigest()[:12]


def count_rows(conn: db.NewsNlpDatabase, after_id: int = 0) -> int:
    """The properly fetched rows (``stream_rows``'s rows) with an ``id`` above *after_id*."""
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM source.articles WHERE fetch_status IN ('ok', 'quarantined') "
            "AND title IS NOT NULL AND trim(title) != '' "
            "AND body_text IS NOT NULL AND trim(body_text) != '' AND id > ?",
            (after_id,),
        ).fetchone()[0]
    )


def finish(
    args: argparse.Namespace,
    entries: dict[str, Entry],
    resolver: CompanyResolver,
    universe: list[Company],
    rows_total: int,
    new_rows: int,
    started: float,
    everything: int,
    read_started: float,
) -> None:
    """Write the CSV and the README (whole, then renamed) and print the totals and the speed."""
    rows = count_by_company(entries.values(), resolver, universe)
    write_csv(args.out / CSV_NAME, rows)
    elapsed = time.time() - started
    readme = readme_text(
        entries,
        rows,
        universe_size=len(universe),
        row_total=rows_total,
        new_rows=new_rows,
        limit=args.limit,
        revision=code_revision(),
        elapsed=elapsed,
    )
    atomic_write_text(args.out / README_NAME, readme)
    print(f"{len(entries):,} unique bodies, {len(rows)} companies, {elapsed / 60:,.1f} min")
    if new_rows:
        rate = new_rows / max(time.time() - read_started, 1e-9)
        print(
            f"speed {rate:,.0f} rows/s: the whole corpus ({everything:,} rows) takes about "
            f"{everything / rate / 60:,.0f} min from the first row"
        )
    print(
        f"written: {args.out / CSV_NAME}, {args.out / README_NAME} and {args.out / CHECKPOINT_NAME}"
    )


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N rows of this run (smoke)"
    )
    parser.add_argument("--out", type=Path, default=Path("data") / "company_news_counts")
    parser.add_argument(
        "--fresh", action="store_true", help="ignore the checkpoint, read everything"
    )
    parser.add_argument("--no-progress", action="store_true", help="no tqdm bar (for a log file)")
    args = parser.parse_args()
    started = time.time()
    conn = db.connect_pipeline()
    universe = load_universe(conn)
    resolver = CompanyResolver(universe)
    terms = list(size.sector_terms(universe))
    checkpoint = args.out / CHECKPOINT_NAME
    print_of = fingerprint(universe)
    loaded = "--fresh was given" if args.fresh else load_checkpoint(checkpoint, print_of)
    if isinstance(loaded, str):
        print(f"starting from the first row: {loaded}", flush=True)
        last_id, rows_before, entries = 0, 0, {}
    else:
        last_id, rows_before, entries = loaded
        print(
            f"resuming after row id {last_id:,}: {rows_before:,} rows and {len(entries):,} bodies "
            "from the checkpoint",
            flush=True,
        )
    everything = count_rows(conn)
    to_read = count_rows(conn, last_id)
    to_read = min(to_read, args.limit) if args.limit else to_read
    print(
        f"universe {len(universe)} tickers; {everything:,} rows in the corpus; reading {to_read:,}"
    )

    def spans_of(article_id: int) -> list[str]:
        return [
            r[0]
            for r in conn.execute(
                "SELECT text FROM article_entities WHERE entity_type = 'ORG' AND article_id = ?",
                (article_id,),
            )
        ]

    state = {"last": last_id, "n": 0}
    bar = tqdm(total=to_read, unit="row", desc="reading", disable=args.no_progress)

    def after_row(article_id: int, bodies: int) -> None:
        state["last"], state["n"] = article_id, state["n"] + 1
        bar.update(1)
        if state["n"] % 100 == 0:
            bar.set_postfix(bodies=f"{bodies:,}")
        if state["n"] % CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint, print_of, state["last"], rows_before + state["n"], entries)

    read_started = time.time()
    try:
        read_corpus(
            stream_rows(conn, args.limit, last_id),
            resolver,
            terms,
            spans_of,
            entries=entries,
            after_row=after_row,
        )
    finally:  # Ctrl-C or an error saves what was read, so the next run continues from there
        bar.close()
        if state["n"]:
            save_checkpoint(checkpoint, print_of, state["last"], rows_before + state["n"], entries)
    finish(
        args,
        entries,
        resolver,
        universe,
        rows_before + state["n"],
        state["n"],
        started,
        everything,
        read_started,
    )


if __name__ == "__main__":
    main()
