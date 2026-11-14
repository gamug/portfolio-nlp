"""Step 8, the snapshot (T-195, D42, FR-024): one ``news_snapshot`` row per (text, main company), or per ``sector`` text, with the crawl date, the title,
the text of step 2, the category (step 7) and the sentiment (step 6). No model and no network: the step only joins what the earlier steps stored.

Design and every decision: ``docs/snapshot_steps.md``. A pair is written only when its text, category and sentiment all exist (complete rows only); the state
is **derived from the tables** (a pair is pending when it has no row at the revision), not kept in ``pipeline_step_state``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from dataclasses import dataclass

from news_nlp import steps
from news_nlp.db import NewsNlpDatabase
from news_summary_stage import normalized_code

PREFIX = "snapshot-"
WAIT_NO_TEXT = "no text at the summary revision"
WAIT_NO_CATEGORY = "no category yet"
WAIT_NO_SENTIMENT = "no sentiment yet"


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
    """The code that builds a row, normalised: a comment, a docstring or the formatting is not a new revision."""
    return hashlib.sha256(
        "\n".join(normalized_code(inspect.getsource(f)) for f in (_row_values,)).encode()
    ).hexdigest()[:12]


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


def revision(
    summary_revision: str,
    companies_revision: str,
    cut_revision: str,
    category_revision: str,
    sentiment_revision: str,
) -> str:
    """The revision of the company rows: the four step revisions a company row reads, the cut revision (the text and the title of a sub-news depend on
    it) and the code that builds the row. The sector revision is **not** in it (owner, 2026-11-14): a sector rerun never changes a company snapshot."""
    key = json.dumps(
        {
            "summary": summary_revision,
            "companies": companies_revision,
            "cut": cut_revision,
            "category": category_revision,
            "sentiment": sentiment_revision,
            "code": _code(),
        },
        sort_keys=True,
    )
    return PREFIX + hashlib.sha256(key.encode()).hexdigest()[:12]


def sector_revision_of(base: str, sector_revision: str) -> str:
    """The revision of the sector rows: the company revision and the sector step's revision."""
    return PREFIX + hashlib.sha256(f"{base}|{sector_revision}".encode()).hexdigest()[:12]


def check_inputs(
    conn: NewsNlpDatabase,
    summary_revision: str,
    companies_revision: str,
    cut_revision: str,
    category_revision: str,
    sentiment_revision: str,
    sector_revision: str | None,
) -> list[str]:
    """Why the run is refused: a revision with no rows, or a category or sentiment row of a pair that is not a main pair at ``companies_revision``
    (the step ran on other inputs). The revisions are hashes, so the inputs they read cannot be read back; this is the check the rows allow."""
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


def articles_of(
    conn: NewsNlpDatabase, items: list[tuple[str, str]]
) -> dict[tuple[str, str], Article]:
    """The crawl date, title and publication date of each item (a body hash, or a ``sub_news`` id). A unique body can come from several articles:
    the date is the **earliest** ``fetched_at`` of them, the title and the publication date are those of the first (lowest id) article. Reads the whole
    corpus once (hashing each body, as step 1 does); an item whose body is not found is missing from the result."""
    from news_nlp import triage  # noqa: PLC0415 -- the body hash of step 1

    hash_of: dict[tuple[str, str], str] = {}
    for kind, item_id in items:
        if kind == steps.ITEM_BODY:
            hash_of[(kind, item_id)] = item_id
    sub_ids = [int(i) for k, i in items if k == steps.ITEM_SUB_NEWS]
    for start in range(0, len(sub_ids), 500):
        chunk = sub_ids[start : start + 500]
        marks = ", ".join("?" for _ in chunk)
        for sub_id, body_hash in conn.execute(
            f"SELECT id, body_hash FROM sub_news WHERE id IN ({marks})",  # noqa: S608 -- placeholders only
            chunk,
        ):
            hash_of[(steps.ITEM_SUB_NEWS, str(sub_id))] = str(body_hash)
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
            seen = found.get(key)
            when = str(fetched_at or "")
            if seen is None:
                found[key] = Article(when, str(title or ""), str(pub_date or ""))
            elif when and (not seen.fetched_at or when < seen.fetched_at):
                found[key] = Article(when, seen.title, seen.pub_date)
    return {item: found[h] for item, h in hash_of.items() if h in found}


def run(
    conn: NewsNlpDatabase,
    revision_: str,
    summary_revision: str,
    companies_revision: str,
    category_revision: str,
    sentiment_revision: str,
    limit: int | None,
) -> dict[str, int]:
    """Write the snapshots of up to ``limit`` ready pairs of the company revision; return what was done. ``waiting`` is the pairs that were not ready (see
    ``--status`` for the reasons), ``no_article`` the ready pairs whose article has no crawl date or is not in the corpus."""
    ready, waiting = _ready_and_waiting(
        conn, summary_revision, companies_revision, category_revision, sentiment_revision, revision_
    )
    if limit is not None:
        ready = ready[:limit]
    info = articles_of(conn, [(p.kind, p.item_id) for p in ready])
    now = steps._now()
    written = no_article = 0
    for pair in ready:
        article = info.get((pair.kind, pair.item_id))
        if article is None or not article.fetched_at:
            no_article += 1
            continue
        conn.execute(
            "INSERT OR IGNORE INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, pub_date, title, text, text_source, "
            "sector, companies_mentioned, category, group_label, sentiment, sentiment_score, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair.kind, pair.item_id, pair.ticker, revision_, *_row_values(pair, article, ""), now),
        )
        written += 1
    conn.commit()
    return {
        "written": written,
        "no_article": no_article,
        "waiting": sum(waiting.values()),
        "ready_left": max(0, len(ready) - written - no_article),
    }


def run_sectors(
    conn: NewsNlpDatabase,
    revision_: str,
    summary_revision: str,
    sector_revision: str,
    limit: int | None,
) -> dict[str, int]:
    """Write the sector texts: a body the sector step assigned (``body_sector.status = 'assigned'``) that has a text at the summary revision.
    No category and no sentiment (D42); the row has ``ticker`` '' and the 2-digit sector."""
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
    todo = [
        (str(h), str(s))
        for h, s in conn.execute(
            "SELECT body_hash, sector FROM body_sector WHERE classifier_revision = ? AND status = 'assigned' ORDER BY rowid",
            (sector_revision,),
        )
        if str(h) not in done and str(h) in texts
    ]
    waiting = sum(
        1
        for (h,) in conn.execute(
            "SELECT body_hash FROM body_sector WHERE classifier_revision = ? AND status = 'assigned'",
            (sector_revision,),
        )
        if str(h) not in done and str(h) not in texts
    )
    if limit is not None:
        todo = todo[:limit]
    info = articles_of(conn, [(steps.ITEM_BODY, h) for h, _ in todo])
    now = steps._now()
    written = no_article = 0
    for h, sector in todo:
        article = info.get((steps.ITEM_BODY, h))
        if article is None or not article.fetched_at:
            no_article += 1
            continue
        text, source = texts[h]
        pair = Pair(steps.ITEM_BODY, h, "", text, source, "", "", "", 0.0)
        values = list(_row_values(pair, article, sector))
        values[9], values[10] = "", None  # no sentiment, no score
        conn.execute(
            "INSERT OR IGNORE INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, pub_date, title, text, text_source, "
            "sector, companies_mentioned, category, group_label, sentiment, sentiment_score, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (steps.ITEM_BODY, h, "", revision_, *values, now),
        )
        written += 1
    conn.commit()
    return {"written": written, "no_article": no_article, "waiting": waiting}


def status(
    conn: NewsNlpDatabase,
    revision_: str | None,
    *,
    summary_revision: str | None = None,
    companies_revision: str | None = None,
    category_revision: str | None = None,
    sentiment_revision: str | None = None,
) -> dict[str, object]:
    """The rows by revision and, with the four input revisions and the snapshot revision, the main pairs still waiting and why (derived from the tables)."""
    out: dict[str, object] = {
        "step": "snapshot",
        "revisions": [
            {"revision": str(r), "rows": int(n)}
            for r, n in conn.execute(
                "SELECT model_revision, COUNT(*) FROM news_snapshot GROUP BY model_revision ORDER BY MIN(processed_at) DESC"
            )
        ],
    }
    if revision_ is not None:
        out["revision"] = revision_
        out["rows"] = {
            "company": int(
                conn.execute(
                    "SELECT COUNT(*) FROM news_snapshot WHERE model_revision = ? AND ticker != ''",
                    (revision_,),
                ).fetchone()[0]
            ),
            "sector_texts": int(
                conn.execute(
                    "SELECT COUNT(*) FROM news_snapshot WHERE model_revision = ? AND ticker = ''",
                    (revision_,),
                ).fetchone()[0]
            ),
        }
    if None not in (
        summary_revision,
        companies_revision,
        category_revision,
        sentiment_revision,
        revision_,
    ):
        _, waiting = _ready_and_waiting(
            conn,
            str(summary_revision),
            str(companies_revision),
            str(category_revision),
            str(sentiment_revision),
            str(revision_),
        )
        out["waiting_pairs"] = dict(waiting)
    return out
