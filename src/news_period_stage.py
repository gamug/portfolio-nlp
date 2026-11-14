"""Step 9, the period summary (T-196, D42, FR-024): every 21 days from 2022-01-03, at 15:00 in ``BATCH_TIMEZONE``, the counts of the snapshots of the window
per company and per category (total and the sentiment split), and the same counts summed per sector. No model, no LLM, no network.

Design and every decision: ``docs/snapshot_steps.md``. A written period is never rewritten; ``period_item`` says which period counted which snapshot, so a
snapshot that reached the table after its window was written is counted in the next period as late. State is derived from the tables: a period is
written when its ``period_run`` row exists.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tqdm import tqdm

import news_snapshot_stage as snap
from news_nlp.db import NewsNlpDatabase
from news_summary_stage import normalized_code

FIRST_END = date(2022, 1, 3)  # a Monday; every cut falls on one
PERIOD_DAYS = 21
CUT_HOUR = 15
DEFAULT_ZONE = "America/New_York"
PREFIX = "period-"
SHOWN = 3  # earlier missing periods named in a refusal
UNCLASSIFIED = ""
SENTIMENTS = ("neutral", "positive", "negative")


class PeriodRefused(RuntimeError):
    """A period the step will not write, with the reason (an earlier one is missing, the date is off the grid, the cut has not passed)."""


# --- the time rule ---------------------------------------------------------------------------------------------------


def batch_timezone(name: str | None = None) -> ZoneInfo:
    """The zone of the 15:00 batch: ``name``, else ``$BATCH_TIMEZONE``, else New York. An invalid name stops the run."""
    chosen = name or os.environ.get("BATCH_TIMEZONE") or DEFAULT_ZONE
    try:
        return ZoneInfo(chosen)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(
            f"BATCH_TIMEZONE {chosen!r} is not a time zone name (an IANA name such as {DEFAULT_ZONE}; .env.example lists the American ones)"
        ) from exc


def period_date(index: int) -> date:
    """The calendar date of the end of period ``index`` (0 is 2022-01-03)."""
    return FIRST_END + timedelta(days=PERIOD_DAYS * index)


def cut_utc(index: int, tz: ZoneInfo) -> str:
    """The end of period ``index``: 15:00 wall time on its date in ``tz``, in UTC (``...Z``). A window is 21 calendar days, 503 or 505 hours across a clock change."""
    day = period_date(index)
    local = datetime(day.year, day.month, day.day, CUT_HOUR, tzinfo=tz)
    return local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def index_of(day: date) -> int:
    """The index of the period that ends on ``day``; refused when it is not on the grid (2022-01-03 + 21k days, k >= 0)."""
    delta = (day - FIRST_END).days
    if delta < 0 or delta % PERIOD_DAYS:
        before = period_date(max(0, delta // PERIOD_DAYS))
        after = period_date(max(0, delta // PERIOD_DAYS) + 1)
        raise PeriodRefused(
            f"{day} is not a period end: they are {FIRST_END} + {PERIOD_DAYS} * k days (the nearest: {before} and {after})"
        )
    return delta // PERIOD_DAYS


def last_index(now: datetime, tz: ZoneInfo) -> int:
    """The last period whose cut has passed at ``now`` (-1 when none has)."""
    stamp = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    index = max(-1, (now.astimezone(UTC).date() - FIRST_END).days // PERIOD_DAYS + 1)
    while index >= 0 and cut_utc(index, tz) > stamp:
        index -= 1
    return index


def _now() -> datetime:
    return datetime.now(UTC)


# --- the sectors -----------------------------------------------------------------------------------------------------


def seed_sectors(conn: NewsNlpDatabase) -> None:
    """Fill the ``sector`` table with the GICS sectors of ``sector_stage`` (``INSERT OR IGNORE``: a row the owner changed or added stays)."""
    from sector_stage import SECTOR_CODES, SECTOR_DESCRIPTIONS  # noqa: PLC0415 -- numpy only

    for name, code in SECTOR_CODES.items():
        conn.execute(
            "INSERT OR IGNORE INTO sector (code, name, kind, descriptions) VALUES (?, ?, 'gics', ?)",
            (code, name, json.dumps(SECTOR_DESCRIPTIONS.get(code, []), sort_keys=True)),
        )
    conn.commit()


def sector_map(conn: NewsNlpDatabase) -> dict[str, str]:
    """The sector code of each company of the universe: its ``company_sector`` row if any, else its GICS sector (through ``SECTOR_CODES``). A company
    in neither is not in the map (the period puts it in ``unclassified``)."""
    from cut_stage import companies_of  # noqa: PLC0415 -- the universe, as the cut reads it
    from sector_stage import SECTOR_CODES  # noqa: PLC0415

    mapping = {
        c.ticker: SECTOR_CODES[c.gics_sector]
        for c in companies_of(conn)
        if c.gics_sector in SECTOR_CODES
    }
    for ticker, code in conn.execute("SELECT ticker, sector_code FROM company_sector"):
        if str(code):
            mapping[str(ticker)] = str(code)
    return mapping


def sector_names(conn: NewsNlpDatabase) -> dict[str, str]:
    """The name of each sector code: the GICS constants, overridden by the ``sector`` table (a row the owner changed or added). Reads only."""
    from sector_stage import SECTOR_CODES  # noqa: PLC0415 -- numpy only

    names = {code: name for name, code in SECTOR_CODES.items()}
    names.update({str(c): str(n) for c, n in conn.execute("SELECT code, name FROM sector")})
    return names


# --- the revision ----------------------------------------------------------------------------------------------------


def _code() -> str:
    """The code that counts, normalised: a comment, a docstring or the formatting is not a new revision."""
    functions = (  # not `run` (the bookkeeping)
        _tally,
        _counts_of,
        _load_rows,
        _write,
        _gaps,
        snap.articles_of,  # `not_counted` reads the crawl dates through it
        snap.crawl_date_ok,
        snap._body_hashes,
    )
    return hashlib.sha256(
        "\n".join(normalized_code(inspect.getsource(f)) for f in functions).encode()
    ).hexdigest()[:12]


def revision(snapshot_revision: str, mapping: dict[str, str]) -> str:
    """The period revision: the snapshot revision, the counting code and the **sector of every company of the universe** (not of the companies that have
    a snapshot: a new company gaining its first snapshot must not restart the periods). A change of the universe's sectors is a new revision."""
    key = json.dumps(
        {"snapshot": snapshot_revision, "mapping": sorted(mapping.items()), "code": _code()}
    )
    return PREFIX + hashlib.sha256(key.encode()).hexdigest()[:12]


# --- the counting ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """A company snapshot as the count reads it."""

    kind: str
    item_id: str
    ticker: str
    fetched_at: str
    category: str
    sentiment: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.item_id, self.ticker)


def _load_rows(conn: NewsNlpDatabase, snapshot_revision: str) -> list[Row]:
    """The company snapshots of the revision (sector texts are not counted), in crawl order."""
    return [
        Row(str(k), str(i), str(t), str(f), str(c), str(s))
        for k, i, t, f, c, s in conn.execute(
            "SELECT item_kind, item_id, ticker, fetched_at, category, sentiment FROM news_snapshot "
            "WHERE model_revision = ? AND ticker != '' ORDER BY fetched_at, item_kind, item_id, ticker",
            (snapshot_revision,),
        )
    ]


def _counts_of(rows: list[Row]) -> dict[str, dict[str, int]]:
    """Per category: ``total`` and the sentiment split of the rows (one count per snapshot)."""
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = out.setdefault(row.category, dict.fromkeys(("total", *SENTIMENTS), 0))
        cell["total"] += 1
        if row.sentiment in SENTIMENTS:
            cell[row.sentiment] += 1
    return out


def _tally(
    rows: list[Row], mapping: dict[str, str]
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]]]:
    """The counts per company and, from the same rows, per sector (the sector of the company; ``UNCLASSIFIED`` when it has none): a sector line is the
    sum of its companies' lines, no category or company with no news appears."""
    by_company: dict[str, list[Row]] = defaultdict(list)
    by_sector: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_company[row.ticker].append(row)
        by_sector[mapping.get(row.ticker, UNCLASSIFIED)].append(row)
    return (
        {t: _counts_of(r) for t, r in by_company.items()},
        {s: _counts_of(r) for s, r in by_sector.items()},
    )


def _gaps(conn: NewsNlpDatabase, snapshot_revision: str, rows: list[Row]) -> list[tuple[str, str]]:
    """(ticker, crawl date) of the main pairs of the companies revision the snapshots were built from that have **no snapshot**: the pairs a period cannot
    count yet (a sentiment skipped, a category missing). Reads the corpus once, only when there are such pairs. The companies revision is the one of the
    latest snapshot written."""
    latest = conn.execute(
        "SELECT inputs FROM news_snapshot WHERE model_revision = ? AND ticker != '' ORDER BY processed_at DESC LIMIT 1",
        (snapshot_revision,),
    ).fetchone()
    if latest is None:
        return []
    companies = json.loads(str(latest[0])).get("companies")
    if not companies:
        return []
    have = {row.key for row in rows}
    waiting = [
        (str(k), str(i), str(t))
        for k, i, t in conn.execute(
            "SELECT item_kind, item_id, ticker FROM news_company WHERE model_revision = ? AND is_main = 1",
            (companies,),
        )
        if (str(k), str(i), str(t)) not in have
    ]
    if not waiting:
        return []
    info = snap.articles_of(conn, sorted({(k, i) for k, i, _ in waiting}))
    return [
        (t, info[(k, i)].fetched_at)
        for k, i, t in waiting
        if (k, i) in info and info[(k, i)].fetched_at
    ]


def written(conn: NewsNlpDatabase, period_rev: str) -> set[str]:
    """The ``period_end`` of every period written at this revision."""
    return {
        str(e)
        for (e,) in conn.execute(
            "SELECT period_end FROM period_run WHERE model_revision = ?", (period_rev,)
        )
    }


def _write(
    conn: NewsNlpDatabase,
    index: int,
    tz: ZoneInfo,
    snapshot_revision: str,
    period_rev: str,
    rows: list[Row],
    counted: set[tuple[str, str, str]],
    mapping: dict[str, str],
    gaps: list[tuple[str, str]],
) -> dict[str, Any]:
    """Write one period: its ``period_run`` row, the company and sector rows with news, and one ``period_item`` for each snapshot counted, in one
    transaction. Counts every snapshot of the revision whose crawl date is at or before the cut and that no earlier period of this revision counted."""
    end = cut_utc(index, tz)
    start = cut_utc(index - 1, tz) if index > 0 else None
    todo = [r for r in rows if r.fetched_at <= end and r.key not in counted]
    by_company, by_sector = _tally(todo, mapping)
    in_window = Counter(t for t, f in gaps if f <= end and (start is None or f > start))
    sector_gaps: Counter[str] = Counter()
    for ticker, n in in_window.items():
        if ticker in by_company:  # no row is made for a company with no counted news
            sector_gaps[mapping.get(ticker, UNCLASSIFIED)] += n
    now = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO period_run (period_end, model_revision, snapshot_revision, period_start, created_at) VALUES (?, ?, ?, ?, ?)",
        (end, period_rev, snapshot_revision, start, now),
    )
    for ticker, counts in by_company.items():
        conn.execute(
            "INSERT INTO period_summary (period_end, kind, key, model_revision, counts, not_counted) VALUES (?, 'company', ?, ?, ?, ?)",
            (end, ticker, period_rev, json.dumps(counts, sort_keys=True), in_window.get(ticker, 0)),
        )
    for sector, counts in by_sector.items():
        conn.execute(
            "INSERT INTO period_summary (period_end, kind, key, model_revision, counts, not_counted) VALUES (?, 'sector', ?, ?, ?, ?)",
            (
                end,
                sector,
                period_rev,
                json.dumps(counts, sort_keys=True),
                sector_gaps.get(sector, 0),
            ),
        )
    late = 0
    for row in todo:
        is_late = start is not None and row.fetched_at <= start
        late += is_late
        conn.execute(
            "INSERT INTO period_item (period_end, model_revision, item_kind, item_id, ticker, late) VALUES (?, ?, ?, ?, ?, ?)",
            (end, period_rev, row.kind, row.item_id, row.ticker, int(is_late)),
        )
        counted.add(row.key)
    conn.commit()
    return {
        "period_end": end,
        "period_start": start,
        "counted": len(todo),
        "late": late,
        "companies": len(by_company),
        "sectors": len(by_sector),
        "not_counted": sum(in_window.values()),
    }


def run(
    conn: NewsNlpDatabase,
    snapshot_revision: str,
    *,
    tz: ZoneInfo,
    now: datetime | None = None,
    index: int | None = None,
    on_plan: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Write period ``index`` (a period already written is skipped), or, with ``index`` None, every missing period up to the last cut that has passed, in
    order, each at its own cut. A period is refused unless every earlier one of the same revision is written and its own cut has passed."""
    seed_sectors(conn)
    mapping = sector_map(conn)
    period_rev = revision(snapshot_revision, mapping)
    moment = now or _now()
    passed = last_index(moment, tz)
    done = written(conn, period_rev)
    if index is None:
        todo = [k for k in range(passed + 1) if cut_utc(k, tz) not in done]
    else:
        if index > passed:
            raise PeriodRefused(
                f"the cut of {period_date(index)} ({cut_utc(index, tz)}) has not passed"
            )
        earlier = [k for k in range(index) if cut_utc(k, tz) not in done]
        if cut_utc(index, tz) not in done and earlier:
            raise PeriodRefused(
                f"periods {', '.join(str(period_date(k)) for k in earlier[:SHOWN])}{' ...' if len(earlier) > SHOWN else ''} are not written: write them first (--catch-up)"
            )
        todo = [index] if cut_utc(index, tz) not in done else []
    result: dict[str, Any] = {
        "period_revision": period_rev,
        "snapshot_revision": snapshot_revision,
        "to_do": len(todo),
        "periods": [],
    }
    if on_plan is not None:
        on_plan(result)  # the number of periods to do, before any is written
    if not todo:
        return result
    rows = _load_rows(conn, snapshot_revision)
    odd = Counter(r.sentiment for r in rows if r.sentiment not in SENTIMENTS)
    if odd:
        raise PeriodRefused(
            f"snapshots with a sentiment other than {', '.join(SENTIMENTS)} ({dict(odd)}): the lines would not add up"
        )
    counted = {
        (str(k), str(i), str(t))
        for k, i, t in conn.execute(
            "SELECT item_kind, item_id, ticker FROM period_item WHERE model_revision = ?",
            (period_rev,),
        )
    }
    gaps = _gaps(conn, snapshot_revision, rows)
    with tqdm(total=len(todo), desc="period", unit="period") as bar:  # progress and time left
        for k in todo:
            bar.set_postfix_str(str(period_date(k)))
            result["periods"].append(
                _write(conn, k, tz, snapshot_revision, period_rev, rows, counted, mapping, gaps)
            )
            bar.update(1)
    return result


# --- reading ---------------------------------------------------------------------------------------------------------


def render(conn: NewsNlpDatabase, period_end: str, period_rev: str) -> str:
    """The text of a written period: a block per company with news (by total, descending), then a block per sector, one line per category."""
    run_row = conn.execute(
        "SELECT period_start FROM period_run WHERE period_end = ? AND model_revision = ?",
        (period_end, period_rev),
    ).fetchone()
    if run_row is None:
        raise PeriodRefused(f"the period ending {period_end} is not written at {period_rev}")
    names = sector_names(conn)
    blocks: dict[str, list[tuple[str, dict[str, dict[str, int]], int]]] = {
        "company": [],
        "sector": [],
    }
    for kind, key, counts, missing in conn.execute(
        "SELECT kind, key, counts, not_counted FROM period_summary WHERE period_end = ? AND model_revision = ?",
        (period_end, period_rev),
    ):
        blocks[str(kind)].append((str(key), json.loads(str(counts)), int(missing)))
    window = f"({run_row[0] or 'the beginning'}, {period_end}]"
    lines = [f"Period summary of {period_end[:10]}  (cut {period_end}, window {window})"]
    for kind in ("company", "sector"):
        for key, counts, missing in sorted(
            blocks[kind], key=lambda b: (-sum(c["total"] for c in b[1].values()), b[0])
        ):
            title = key if kind == "company" else (names.get(key, key) if key else "unclassified")
            lines += ["_" * 39, f"{title}:"]
            for category, cell in sorted(counts.items(), key=lambda c: (-c[1]["total"], c[0])):
                lines.append(
                    f"  {category}: {cell['total']} total-{cell['neutral']} neutral-{cell['positive']} positive-{cell['negative']} negative"
                )
            if missing:
                lines.append(f"  not counted yet: {missing}")
    lines.append("_" * 39)
    return "\n".join(lines)


def status(
    conn: NewsNlpDatabase,
    snapshot_revision: str | None = None,
    *,
    tz: ZoneInfo | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """The periods written (with how many snapshots each counted and how many were late) and, with a snapshot revision, the period revision this code
    would write and the periods still missing up to the last cut that has passed."""
    zone = tz or batch_timezone()
    out: dict[str, object] = {
        "step": "period",
        "timezone": str(zone),
        "periods_written": [
            {
                "period_end": str(e),
                "period_revision": str(r),
                "snapshot_revision": str(s),
                "counted": int(n),
                "late": int(late or 0),
            }
            for e, r, s, n, late in conn.execute(
                "SELECT r.period_end, r.model_revision, r.snapshot_revision, "
                "(SELECT COUNT(*) FROM period_item i WHERE i.period_end = r.period_end AND i.model_revision = r.model_revision), "
                "(SELECT SUM(late) FROM period_item i WHERE i.period_end = r.period_end AND i.model_revision = r.model_revision) "
                "FROM period_run r ORDER BY r.period_end, r.model_revision"
            )
        ],
    }
    if snapshot_revision is not None:
        period_rev = revision(snapshot_revision, sector_map(conn))
        done = written(conn, period_rev)
        passed = last_index(now or _now(), zone)
        out["period_revision"] = period_rev
        out["missing"] = [
            str(period_date(k)) for k in range(passed + 1) if cut_utc(k, zone) not in done
        ]
    return out


def clear_periods_of(conn: NewsNlpDatabase, snapshot_revision: str) -> int:
    """Delete every period built from ``snapshot_revision`` (its ``period_run``, ``period_summary`` and ``period_item`` rows): a reset of the snapshots
    resets the periods built on them (owner, 2026-11-14). Does not commit. Returns how many periods."""
    runs = [
        (str(e), str(r))
        for e, r in conn.execute(
            "SELECT period_end, model_revision FROM period_run WHERE snapshot_revision = ?",
            (snapshot_revision,),
        )
    ]
    for end, rev in runs:
        for table in ("period_item", "period_summary", "period_run"):
            conn.execute(
                f"DELETE FROM {table} WHERE period_end = ? AND model_revision = ?",  # noqa: S608 -- fixed names
                (end, rev),
            )
    return len(runs)
