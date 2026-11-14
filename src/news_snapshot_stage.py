"""Step 8, the snapshot (T-195, D42, FR-024): one ``news_snapshot`` row per (text, main company), or per ``sector`` text, with the crawl date, the title,
the text of step 2, the category (step 7) and the sentiment (step 6). No model and no network: the step only joins what the earlier steps stored.

Design and every decision: ``docs/snapshot_steps.md``. A pair is written only when its text, category and sentiment all exist (complete rows only); the state
is **derived from the tables** (a pair is pending when it has no row at the revision, whatever the input revisions), not kept in ``pipeline_step_state``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from news_nlp import steps
from news_nlp.db import NewsNlpDatabase
from news_summary_stage import normalized_code

PREFIX = "snapshot-"
WAIT_NO_TEXT = "no text at the summary revision"
WAIT_NO_CATEGORY = "no category yet"
WAIT_NO_SENTIMENT = "no sentiment yet"
WAIT_NO_ARTICLE = "the article is not in the corpus"
WAIT_NO_DATE = "no crawl date in UTC (YYYY-MM-DDTHH:MM:SSZ)"


@dataclass(frozen=True)
class Pair:
    """A (text, main company) pair ready to write, with what the join found."""

    kind: str
    item_id: str
    ticker: str
    text: str
    text_source: str
    category: str
    group_label: str
    sentiment: str
    sentiment_score: float


@dataclass(frozen=True)
class Article:
    """What the corpus says of a unique body: the earliest crawl of its articles, and the title and publication date of the first (lowest id) one."""

    fetched_at: str
    title: str
    pub_date: str


def _code() -> str:
    """The code that decides a row, normalised: the columns (``_row_values``), the join of the earlier steps (``_ready_and_waiting``), the reading of
    the corpus (``articles_of``, ``crawl_date_ok``), the choice of the pairs (``_with_articles``), the sector rows (``_sector_todo``, ``_sector_row``) and the statement that writes them (``_INSERT``). A comment, a
    docstring or the formatting is not a new revision; ``run`` (the bookkeeping: the limit, the counts) is not hashed on purpose."""
    functions = (
        _row_values,
        _ready_and_waiting,
        _body_hashes,
        articles_of,
        crawl_date_ok,
        _with_articles,
        _sector_todo,
        _sector_row,
    )
    parts = [
        _INSERT,
        *(normalized_code(inspect.getsource(f)) for f in functions),
    ]  # _INSERT: the columns written, in their order
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


_CRAWL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def crawl_date_ok(value: str) -> bool:
    """A crawl date the period windows can read: UTC, ``YYYY-MM-DDTHH:MM:SSZ`` (the format the crawl stores). Strings of this shape compare as times."""
    return _CRAWL_DATE.match(value) is not None


def _row_values(pair: Pair, article: Article, sector: str) -> tuple[object, ...]:
    """The columns of one snapshot row, in the order of the table (without the key and the time of writing)."""
    return (
        article.fetched_at,
        article.pub_date,
        article.title,
        pair.text,
        pair.text_source,
        sector,
        "[]",  # companies_mentioned: out of scope for now (T-198)
        pair.category,
        pair.group_label,
        pair.sentiment,
        pair.sentiment_score,
    )


def revision() -> str:
    """The revision of the snapshots: **the code that builds a row only** (owner, 2026-11-14: the step is always idempotent). The revisions of the five steps
    it read are not in it: they are recorded in each row (``inputs``), so a sentiment revision that moves (a new company becomes main) neither restarts the
    snapshots nor the periods built on them. Another revision of these rows is a change of this code, or ``--reset``."""
    return PREFIX + hashlib.sha256(_code().encode()).hexdigest()[:12]


def check_inputs(
    conn: NewsNlpDatabase,
    summary_revision: str,
    companies_revision: str,
    cut_revision: str,
    category_revision: str,
    sentiment_revision: str,
    sector_revision: str | None,
) -> list[str]:
    """Why the run is refused: a revision with no rows, an item with a main company and no text at the summary revision, a sub-news not of the named
    cut, or a category or sentiment row of a pair that is not a main pair at ``companies_revision`` (the step ran on other inputs). The revisions are hashes, so the inputs they read cannot be read back; this is the check the rows allow."""
    problems: list[str] = []
    checks = [
        ("summary", "news_summary", "model_revision", summary_revision, ""),
        (
            "companies (main)",
            "news_company",
            "model_revision",
            companies_revision,
            "AND is_main = 1",
        ),
        ("cut", "sub_news", "extractor_revision", cut_revision, ""),
        ("category", "news_category", "model_revision", category_revision, ""),
        ("sentiment", "news_sentiment", "model_revision", sentiment_revision, ""),
    ]
    if sector_revision is not None:
        checks.append(
            (
                "sector",
                "body_sector",
                "classifier_revision",
                sector_revision,
                "AND status = 'assigned'",
            )
        )
    for name, table, column, value, extra in checks:
        found = conn.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ? {extra} LIMIT 1",  # noqa: S608 -- fixed names
            (value,),
        ).fetchone()
        if found is None:
            problems.append(f"{name} revision {value}: no rows")
    if problems:
        return problems
    no_text = steps.category_items_without_text(conn, summary_revision, companies_revision)
    if no_text:
        problems.append(
            f"{no_text} items with a main company at {companies_revision} have no text at the summary revision {summary_revision} "
            "(did the companies step read another summary revision?)"
        )
    stray_cut = conn.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT item_id FROM news_company WHERE model_revision = ? AND is_main = 1 AND item_kind = ?) c "
        "WHERE NOT EXISTS (SELECT 1 FROM sub_news s WHERE CAST(s.id AS TEXT) = c.item_id AND s.extractor_revision = ?)",
        (companies_revision, steps.ITEM_SUB_NEWS, cut_revision),
    ).fetchone()[0]
    if stray_cut:
        problems.append(
            f"cut revision {cut_revision}: {stray_cut} sub-news with a main company at {companies_revision} are not of that cut"
        )
    for name, table, value in (
        ("category", "news_category", category_revision),
        ("sentiment", "news_sentiment", sentiment_revision),
    ):
        stray = conn.execute(
            f"SELECT COUNT(*) FROM {table} t WHERE t.model_revision = ? AND NOT EXISTS "  # noqa: S608 -- fixed names
            "(SELECT 1 FROM news_company c WHERE c.item_kind = t.item_kind AND c.item_id = t.item_id AND c.ticker = t.ticker "
            "AND c.model_revision = ? AND c.is_main = 1)",
            (value, companies_revision),
        ).fetchone()[0]
        if stray:
            problems.append(
                f"{name} revision {value}: {stray} rows are not main pairs at the companies revision {companies_revision} "
                "(it ran on other companies)"
            )
    return problems


def _ready_and_waiting(
    conn: NewsNlpDatabase,
    summary_revision: str,
    companies_revision: str,
    category_revision: str,
    sentiment_revision: str,
    revision_: str,
) -> tuple[list[Pair], Counter[str]]:
    """The main pairs with no snapshot at ``revision_``: those with text, category and sentiment (ready) and, for the rest, the reason they wait."""
    texts = {
        (str(k), str(i)): (str(t), str(s))
        for k, i, t, s in conn.execute(
            "SELECT item_kind, item_id, summary_text, text_source FROM news_summary WHERE model_revision = ?",
            (summary_revision,),
        )
    }
    categories = {
        (str(k), str(i), str(t)): (str(g), str(grp))
        for k, i, t, g, grp in conn.execute(
            "SELECT item_kind, item_id, ticker, label, group_label FROM news_category WHERE model_revision = ?",
            (category_revision,),
        )
    }
    sentiments = {
        (str(k), str(i), str(t)): (str(lab), float(sc))
        for k, i, t, lab, sc in conn.execute(
            "SELECT item_kind, item_id, ticker, label, score FROM news_sentiment WHERE model_revision = ?",
            (sentiment_revision,),
        )
    }
    written = {
        (str(k), str(i), str(t))
        for k, i, t in conn.execute(
            "SELECT item_kind, item_id, ticker FROM news_snapshot WHERE model_revision = ? AND ticker != ''",
            (revision_,),
        )
    }
    ready: list[Pair] = []
    waiting: Counter[str] = Counter()
    for k, i, t in conn.execute(
        "SELECT item_kind, item_id, ticker FROM news_company WHERE model_revision = ? AND is_main = 1 ORDER BY item_kind, rowid",
        (companies_revision,),
    ):
        key = (str(k), str(i), str(t))
        if key in written:
            continue
        text = texts.get((key[0], key[1]))
        if text is None:
            waiting[WAIT_NO_TEXT] += 1
        elif key not in categories:
            waiting[WAIT_NO_CATEGORY] += 1
        elif key not in sentiments:
            waiting[WAIT_NO_SENTIMENT] += 1
        else:
            ready.append(
                Pair(
                    *key, text[0], text[1], categories[key][0], categories[key][1], *sentiments[key]
                )
            )
    return ready, waiting


def _body_hashes(conn: NewsNlpDatabase, items: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """The body hash of each item: a body is its own hash, a sub-news is looked up (``sub_news.body_hash``)."""
    hash_of = {(k, i): i for k, i in items if k == steps.ITEM_BODY}
    sub_ids = [int(i) for k, i in items if k == steps.ITEM_SUB_NEWS]
    for start in range(0, len(sub_ids), 500):
        chunk = sub_ids[start : start + 500]
        marks = ", ".join("?" for _ in chunk)
        for sub_id, body_hash in conn.execute(
            f"SELECT id, body_hash FROM sub_news WHERE id IN ({marks})",  # noqa: S608 -- placeholders only
            chunk,
        ):
            hash_of[(steps.ITEM_SUB_NEWS, str(sub_id))] = str(body_hash)
    return hash_of


def articles_of(
    conn: NewsNlpDatabase, items: list[tuple[str, str]]
) -> dict[tuple[str, str], Article]:
    """The crawl date, title and publication date of each item (a body hash, or a ``sub_news`` id). A unique body can come from several articles:
    the date is the **earliest valid** ``fetched_at`` of them (UTC, ``crawl_date_ok``; another format is ignored, and a body with none has an empty
    date), the title and the publication date are those of the first (lowest id) article. Reads the whole corpus once (hashing each body, as step 1
    does); an item whose body is not found is missing from the result."""
    from news_nlp import triage  # noqa: PLC0415 -- the body hash of step 1

    hash_of = _body_hashes(conn, items)
    wanted = set(hash_of.values())
    found: dict[str, Article] = {}
    if wanted:
        proper = ", ".join("?" for _ in steps.PROPER_STATUSES)
        rel = "source.articles" if conn.articles_rel == "source" else "main.articles"
        query = (
            f"SELECT title, pub_date, fetched_at, body_text FROM {rel} "  # noqa: S608 -- the schema name is main or source
            f"WHERE fetch_status IN ({proper}) AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
        )
        for title, pub_date, fetched_at, text in conn.execute(query, steps.PROPER_STATUSES):
            key = triage.body_hash(text)
            if key not in wanted:
                continue
            when = str(fetched_at or "")
            if not crawl_date_ok(when):
                when = ""  # not a UTC time: it never decides the window
            seen = found.get(key)
            if seen is None:
                found[key] = Article(when, str(title or ""), str(pub_date or ""))
            elif when and (not seen.fetched_at or when < seen.fetched_at):
                found[key] = Article(when, seen.title, seen.pub_date)
    return {item: found[h] for item, h in hash_of.items() if h in found}


def _with_articles(
    ready: list[Pair], info: dict[tuple[str, str], Article], waiting: Counter[str]
) -> list[tuple[Pair, Article]]:
    """The ready pairs that have an article with a valid crawl date; the others are counted in ``waiting`` with their reason (they do not take a
    place of the limit)."""
    complete: list[tuple[Pair, Article]] = []
    for pair in ready:
        article = info.get((pair.kind, pair.item_id))
        if article is None:
            waiting[WAIT_NO_ARTICLE] += 1
        elif not article.fetched_at:
            waiting[WAIT_NO_DATE] += 1
        else:
            complete.append((pair, article))
    return complete


def _pending(
    conn: NewsNlpDatabase,
    summary_revision: str,
    companies_revision: str,
    category_revision: str,
    sentiment_revision: str,
    revision_: str,
) -> Counter[str]:
    """Why the pairs of the company revision still wait (the join of the earlier steps, then the corpus). Reads the whole corpus once."""
    ready, waiting = _ready_and_waiting(
        conn, summary_revision, companies_revision, category_revision, sentiment_revision, revision_
    )
    _with_articles(ready, articles_of(conn, [(p.kind, p.item_id) for p in ready]), waiting)
    return waiting


def _sector_todo(
    conn: NewsNlpDatabase, revision_: str, summary_revision: str, sector_revision: str
) -> tuple[list[tuple[str, str, str, str]], int]:
    """The sector texts still to write, as (body hash, 2-digit sector, text, text source): a body the sector step assigned
    (``body_sector.status = 'assigned'``) that has a text at the summary revision and no row at ``revision_``; and how many have no text yet."""
    done = {
        str(i)
        for (i,) in conn.execute(
            "SELECT item_id FROM news_snapshot WHERE model_revision = ? AND ticker = ''",
            (revision_,),
        )
    }
    texts = {
        str(i): (str(t), str(s))
        for i, t, s in conn.execute(
            "SELECT item_id, summary_text, text_source FROM news_summary WHERE model_revision = ? AND item_kind = ?",
            (summary_revision, steps.ITEM_BODY),
        )
    }
    todo: list[tuple[str, str, str, str]] = []
    no_text = 0
    for h, sector in conn.execute(
        "SELECT body_hash, sector FROM body_sector WHERE classifier_revision = ? AND status = 'assigned' ORDER BY rowid",
        (sector_revision,),
    ):
        if str(h) in done:
            continue
        if str(h) in texts:
            todo.append((str(h), str(sector), *texts[str(h)]))
        else:
            no_text += 1
    return todo, no_text


def _sector_row(
    body_hash: str, sector: str, text: str, source: str, article: Article
) -> tuple[object, ...]:
    """The columns of a sector text: the 2-digit sector, no category and no sentiment (D42)."""
    values = list(
        _row_values(
            Pair(steps.ITEM_BODY, body_hash, "", text, source, "", "", "", 0.0), article, sector
        )
    )
    values[9], values[10] = "", None  # no sentiment, no score
    return tuple(values)


_INSERT = (
    "INSERT OR IGNORE INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, pub_date, title, text, text_source, "
    "sector, companies_mentioned, category, group_label, sentiment, sentiment_score, inputs, processed_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def reset(conn: NewsNlpDatabase, *, sectors: bool) -> dict[str, int]:
    """Delete the snapshots of the current revision and return how many of each kind: the one way to rebuild them (``--reset``). The company rows always;
    the sector rows only when the run names a sector revision (``sectors``), because only then are they written again. Owner, 2026-11-14: a reset of the
    snapshots resets the period summary built on them too (the period tables, when T-196 builds them, are cleared here for the revision)."""
    rev = revision()
    company = int(
        conn.execute(
            "SELECT COUNT(*) FROM news_snapshot WHERE model_revision = ? AND ticker != ''", (rev,)
        ).fetchone()[0]
    )
    sector = int(
        conn.execute(
            "SELECT COUNT(*) FROM news_snapshot WHERE model_revision = ? AND ticker = ''", (rev,)
        ).fetchone()[0]
    )
    conn.execute("DELETE FROM news_snapshot WHERE model_revision = ? AND ticker != ''", (rev,))
    if sectors:
        conn.execute("DELETE FROM news_snapshot WHERE model_revision = ? AND ticker = ''", (rev,))
    conn.commit()
    return {"company_rows": company, "sector_rows": sector if sectors else 0}


def run(
    conn: NewsNlpDatabase,
    summary_revision: str,
    companies_revision: str,
    cut_revision: str,
    category_revision: str,
    sentiment_revision: str,
    limit: int | None,
    sector_revision: str | None = None,
) -> dict[str, Any]:
    """Write the snapshots of up to ``limit`` complete pairs of the company revision (and, with ``sector_revision``, up to ``limit`` more: the sector
    texts); return what was done. A pair that has a row is never read again, whatever revisions it was built from (they are in the row's ``inputs``).
    The corpus is read **once** for both. ``waiting`` is what cannot be written yet, ``waiting_by_reason`` why (the company pairs), ``ready_left`` the
    complete rows the limit left for the next run."""
    revision_ = revision()
    inputs = json.dumps(
        {
            "summary": summary_revision,
            "companies": companies_revision,
            "cut": cut_revision,
            "category": category_revision,
            "sentiment": sentiment_revision,
            "sector": sector_revision or "",
        },
        sort_keys=True,
    )
    ready, waiting = _ready_and_waiting(
        conn, summary_revision, companies_revision, category_revision, sentiment_revision, revision_
    )
    todo_sector, sector_no_text = (
        _sector_todo(conn, revision_, summary_revision, sector_revision)
        if sector_revision
        else ([], 0)
    )
    info = articles_of(
        conn,
        [(p.kind, p.item_id) for p in ready] + [(steps.ITEM_BODY, h) for h, *_ in todo_sector],
    )
    complete = _with_articles(ready, info, waiting)
    chosen = complete if limit is None else complete[:limit]
    now = steps._now()
    for pair, article in chosen:
        conn.execute(
            _INSERT,
            (
                pair.kind,
                pair.item_id,
                pair.ticker,
                revision_,
                *_row_values(pair, article, ""),
                inputs,
                now,
            ),
        )
    result: dict[str, Any] = {
        "revision": revision_,
        "written": len(chosen),
        "waiting": sum(waiting.values()),
        "waiting_by_reason": dict(waiting),
        "ready_left": len(complete) - len(chosen),
    }
    if sector_revision:
        sector_ready = [
            (h, sec, text, source, info[(steps.ITEM_BODY, h)])
            for h, sec, text, source in todo_sector
            if info.get((steps.ITEM_BODY, h)) and info[(steps.ITEM_BODY, h)].fetched_at
        ]
        sector_chosen = sector_ready if limit is None else sector_ready[:limit]
        for h, sec, text, source, article in sector_chosen:
            conn.execute(
                _INSERT,
                (
                    steps.ITEM_BODY,
                    h,
                    "",
                    revision_,
                    *_sector_row(h, sec, text, source, article),
                    inputs,
                    now,
                ),
            )
        result["sector_rows"] = {
            "written": len(sector_chosen),
            "waiting": sector_no_text + len(todo_sector) - len(sector_ready),
            "ready_left": len(sector_ready) - len(sector_chosen),
        }
    conn.commit()
    return result


def _stale(
    conn: NewsNlpDatabase, named: dict[str, str | None], companies_revision: str
) -> dict[str, int]:
    """What the rows of the current revision say against the revisions named now: for each input, how many rows were built from another one, and
    how many company rows are not a main pair at the companies revision. Nothing is changed: the step is always idempotent, a reset rebuilds."""
    stale: Counter[str] = Counter()
    main = {
        (str(k), str(i), str(t))
        for k, i, t in conn.execute(
            "SELECT item_kind, item_id, ticker FROM news_company WHERE model_revision = ? AND is_main = 1",
            (companies_revision,),
        )
    }
    for kind, item_id, ticker, raw in conn.execute(
        "SELECT item_kind, item_id, ticker, inputs FROM news_snapshot WHERE model_revision = ?",
        (revision(),),
    ):
        built = json.loads(str(raw))
        keys = (
            ("summary", "sector")
            if ticker == ""
            else ("summary", "companies", "cut", "category", "sentiment")
        )
        for key in keys:
            if named.get(key) and built.get(key) != named[key]:
                stale[f"built_from_another_{key}_revision"] += 1
        if ticker != "" and (str(kind), str(item_id), str(ticker)) not in main:
            stale["not_a_main_pair_now"] += 1
    return dict(stale)


def status(
    conn: NewsNlpDatabase,
    *,
    summary_revision: str | None = None,
    companies_revision: str | None = None,
    category_revision: str | None = None,
    sentiment_revision: str | None = None,
    cut_revision: str | None = None,
    sector_revision: str | None = None,
    waiting: dict[str, int] | None = None,
) -> dict[str, object]:
    """The revision of this code, its rows (company rows and sector rows apart), the revisions that ever wrote rows and, with the four input revisions,
    the main pairs still waiting and why (derived from the tables; it reads the corpus once, unless ``waiting`` is given: the run already knows it)."""
    current = revision()
    counts = {
        str(r): (int(c), int(t))
        for r, c, t in conn.execute(
            "SELECT model_revision, COALESCE(SUM(ticker != ''), 0), COALESCE(SUM(ticker = ''), 0) FROM news_snapshot GROUP BY model_revision"
        )
    }
    company_rows, sector_rows = counts.get(current, (0, 0))
    out: dict[str, object] = {
        "step": "snapshot",
        "revision": current,
        "company_rows": company_rows,
        "sector_rows": sector_rows,
        "revisions": [
            {"revision": r, "company_rows": c, "sector_rows": t} for r, (c, t) in counts.items()
        ],
    }
    if None not in (summary_revision, companies_revision, category_revision, sentiment_revision):
        out["stale_rows"] = _stale(
            conn,
            {
                "summary": summary_revision,
                "companies": companies_revision,
                "cut": cut_revision,
                "category": category_revision,
                "sentiment": sentiment_revision,
                "sector": sector_revision,
            },
            str(companies_revision),
        )
        out["waiting_pairs"] = (
            waiting
            if waiting is not None
            else dict(
                _pending(
                    conn,
                    str(summary_revision),
                    str(companies_revision),
                    str(category_revision),
                    str(sentiment_revision),
                    current,
                )
            )
        )
    return out
