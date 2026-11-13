"""The stepped pipeline's run tracking and the storage of steps 1 and 2 (PLAN D39/D40; T-162, T-163, T-164, T-166).

No model and no GPU here: which bodies a step still has to do, how a result and its state are written, and the counts the
``--status`` command prints. A body is identified by its hash (``triage.body_hash``), so a body that sits under several
tickers runs once. Everything is keyed by the step and the model revision: a new revision adds rows, the old ones stay, and a
rerun of the same revision adds nothing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple

from news_nlp import triage
from news_nlp.db import NewsNlpDatabase, _ensure_article_row

STEP_SCOPE = "scope"
STEP_CUT = "cut"
STEP_SUMMARY = "summary"
STEP_COMPANIES = "companies"
STEP_SECTOR = "sector"
ITEM_BODY = "body"
ITEM_SUB_NEWS = "sub_news"
SCOPES = ("single", "multi", "sector", "out_of_scope")
# The properly fetched corpus (D36, T-159): ``quarantined`` is a verdict on the company tag, not on the article.
PROPER_STATUSES = ("ok", "quarantined")
# The companies step does not retry an item on the review list: the same revision would give the same answer.
COMPANY_FINISHED = ("done", "skipped", "review")


@dataclass(frozen=True)
class SubNews:
    """One sub-news the cut wrote: its row id (the item id of the steps after the cut), its body and its text."""

    sub_news_id: int
    body_hash: str
    text: str


@dataclass(frozen=True)
class Body:
    """One unique body of SOURCE: its hash, the first article that carries it, and the text."""

    body_hash: str
    article_id: int
    title: str
    text: str


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _articles(conn: NewsNlpDatabase) -> str:
    return "source.articles" if conn.articles_rel == "source" else "main.articles"


def finished(
    conn: NewsNlpDatabase,
    step: str,
    revision: str,
    item_kind: str = ITEM_BODY,
    statuses: tuple[str, ...] = ("done", "skipped"),
) -> set[str]:
    """Ids of the items (by default the body hashes) the step has already done (or skipped) at this revision: failed and
    pending ones are retried. ``statuses`` names the states that count as finished (the companies step adds ``review``: an item
    waiting for the owner gives the same answer on a rerun)."""
    marks = ", ".join("?" for _ in statuses)
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT item_id FROM pipeline_step_state "  # noqa: S608 -- one mark per fixed state
            f"WHERE item_kind = ? AND step = ? AND model_revision = ? AND status IN ({marks})",
            (item_kind, step, revision, *statuses),
        )
    }


def pending_bodies(
    conn: NewsNlpDatabase, step: str, revision: str, limit: int | None = None
) -> Iterator[Body]:
    """The unique bodies of SOURCE still to do, in article-id order, at most ``limit`` of them."""
    done = finished(conn, step, revision)
    seen: set[str] = set()
    taken = 0
    marks = ", ".join("?" for _ in PROPER_STATUSES)
    query = (
        f"SELECT id, title, body_text FROM {_articles(conn)} "  # noqa: S608 -- the schema name is main or source, never input
        f"WHERE fetch_status IN ({marks}) AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
    )
    for article_id, title, text in conn.execute(query, PROPER_STATUSES):
        key = triage.body_hash(text)
        if key in seen or key in done:
            continue
        seen.add(key)
        yield Body(key, int(article_id), title or "", text)
        taken += 1
        if limit is not None and taken >= limit:
            return


def record_scope(
    conn: NewsNlpDatabase,
    body: Body,
    revision: str,
    scope: str,
    scores: dict[str, float],
    text_used: str,
) -> None:
    """Store one body's scope and mark the step done in one transaction; a repeat changes nothing."""
    now = _now()
    conn.execute(
        "INSERT OR IGNORE INTO body_scope "
        "(body_hash, classifier_revision, scope, scores, text_used, processed_at) VALUES (?, ?, ?, ?, ?, ?)",
        (body.body_hash, revision, scope, json.dumps(scores, sort_keys=True), text_used, now),
    )
    conn.execute(
        "INSERT INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, 'done', 1, ?, ?) "
        "ON CONFLICT (item_kind, item_id, step, model_revision) DO UPDATE SET "
        "status = 'done', attempts = attempts + 1, error = NULL, finished_at = excluded.finished_at",
        (ITEM_BODY, body.body_hash, STEP_SCOPE, revision, now, now),
    )
    conn.commit()


def latest_classifier_revision(conn: NewsNlpDatabase) -> str | None:
    """The classifier revision of the most recent ``body_scope`` row, or None when step 1 has not run."""
    row = conn.execute(
        "SELECT classifier_revision FROM body_scope ORDER BY processed_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    return None if row is None else str(row[0])


def _pending_scoped_bodies(
    conn: NewsNlpDatabase,
    classifier_revision: str,
    scope: str,
    step: str,
    revision: str,
    limit: int | None,
) -> Iterator[Body]:
    """The bodies of step 1 (at ``classifier_revision``) with this ``scope`` that ``step`` has not done at ``revision``, at most ``limit``.

    ``body_scope`` keeps no text, so the bodies are found in SOURCE by hash; the scan is in article-id order, so the
    article each body carries is the lowest id of its group (T-132), and it stops when every wanted body is found.
    """
    done = finished(conn, step, revision)
    wanted = [
        str(row[0])
        for row in conn.execute(
            "SELECT body_hash FROM body_scope WHERE classifier_revision = ? AND scope = ? ORDER BY rowid",
            (classifier_revision, scope),
        )
        if str(row[0]) not in done
    ]
    if limit is not None:
        wanted = wanted[:limit]
    left = set(wanted)
    marks = ", ".join("?" for _ in PROPER_STATUSES)  # the same corpus as step 1
    query = (
        f"SELECT id, title, body_text FROM {_articles(conn)} "  # noqa: S608 -- the schema name is main or source, never input
        f"WHERE fetch_status IN ({marks}) AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
    )
    for article_id, title, text in conn.execute(query, PROPER_STATUSES):
        if not left:
            return
        key = triage.body_hash(text)
        if key in left:
            left.discard(key)
            yield Body(key, int(article_id), title or "", text)


def pending_multi_bodies(
    conn: NewsNlpDatabase, classifier_revision: str, revision: str, limit: int | None = None
) -> Iterator[Body]:
    """The ``multi`` bodies of step 1 the cut has not done at ``revision``, at most ``limit`` (step 1.2)."""
    return _pending_scoped_bodies(conn, classifier_revision, "multi", STEP_CUT, revision, limit)


def pending_single_bodies(
    conn: NewsNlpDatabase, classifier_revision: str, revision: str, limit: int | None = None
) -> Iterator[Body]:
    """The ``single`` bodies of step 1 the summary has not done at ``revision``, at most ``limit`` (step 2)."""
    return _pending_scoped_bodies(
        conn, classifier_revision, "single", STEP_SUMMARY, revision, limit
    )


def pending_sub_news(
    conn: NewsNlpDatabase, cut_revision: str, revision: str, limit: int | None = None
) -> Iterator[SubNews]:
    """The sub-news the cut wrote at ``cut_revision`` that the summary has not done at ``revision``, at most ``limit``."""
    done = finished(conn, STEP_SUMMARY, revision, ITEM_SUB_NEWS)
    taken = 0
    for row_id, body_hash, text in conn.execute(
        "SELECT id, body_hash, text FROM sub_news WHERE extractor_revision = ? ORDER BY id",
        (cut_revision,),
    ):
        if str(row_id) in done:
            continue
        yield SubNews(int(row_id), str(body_hash), str(text))
        taken += 1
        if limit is not None and taken >= limit:
            return


def record_summary(
    conn: NewsNlpDatabase,
    item_kind: str,
    item_id: str,
    revision: str,
    summary_text: str,
    text_source: str,
    num_chunks: int,
) -> None:
    """Store the text the next steps read for one item (its summary, or its own text when it is under the size) and mark
    the summary step done, in one transaction; a repeat changes nothing."""
    now = _now()
    conn.execute(
        "INSERT OR IGNORE INTO news_summary "
        "(item_kind, item_id, model_revision, summary_text, text_source, num_chunks, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item_kind, item_id, revision, summary_text, text_source, num_chunks, now),
    )
    conn.execute(
        "INSERT INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, 'done', 1, ?, ?) "
        "ON CONFLICT (item_kind, item_id, step, model_revision) DO UPDATE SET "
        "status = 'done', attempts = attempts + 1, error = NULL, finished_at = excluded.finished_at",
        (item_kind, item_id, STEP_SUMMARY, revision, now, now),
    )
    conn.commit()


class TextItem(NamedTuple):
    """One text the companies step reads: a body (its hash) or a sub-news (its id), and the text step 2 left for it."""

    kind: str
    item_id: str
    text: str


class CompanyScore(NamedTuple):
    """What the companies step found for one ticker in one text: how many spans named it and the relevance score of the pair."""

    ticker: str
    mention_count: int
    relevance_score: float


def is_main(score: float, threshold: float) -> bool:
    """Whether a relevance score makes the company main: at the threshold or over it (D39 (i))."""
    return score >= threshold


def pending_texts(
    conn: NewsNlpDatabase,
    summary_revision: str,
    revision: str,
    *,
    scope_revision: str | None,
    cut_revision: str | None,
    items: tuple[str, ...] = ("single", "sub_news"),
    limit: int | None = None,
) -> Iterator[TextItem]:
    """The texts of step 2 (``news_summary`` at ``summary_revision``) the companies step has not finished at ``revision``: the
    ``single`` bodies of ``scope_revision``, then the sub-news of ``cut_revision``; at most ``limit`` in all. A revision of the
    scope, the cut and the summary is always named, never "the latest" (``docs/scope_training.md`` 8.1)."""
    taken = 0
    if "single" in items and scope_revision is not None:
        done = finished(conn, STEP_COMPANIES, revision, ITEM_BODY, COMPANY_FINISHED)
        for item_id, text in conn.execute(
            "SELECT s.item_id, s.summary_text FROM news_summary s "
            "JOIN body_scope b ON b.body_hash = s.item_id AND b.classifier_revision = ? AND b.scope = 'single' "
            "WHERE s.item_kind = ? AND s.model_revision = ? ORDER BY s.rowid",
            (scope_revision, ITEM_BODY, summary_revision),
        ):
            if str(item_id) in done:
                continue
            yield TextItem(ITEM_BODY, str(item_id), str(text))
            taken += 1
            if limit is not None and taken >= limit:
                return
    if "sub_news" in items and cut_revision is not None:
        done = finished(conn, STEP_COMPANIES, revision, ITEM_SUB_NEWS, COMPANY_FINISHED)
        for item_id, text in conn.execute(
            "SELECT s.item_id, s.summary_text FROM news_summary s "
            "JOIN sub_news n ON CAST(n.id AS TEXT) = s.item_id AND n.extractor_revision = ? "
            "WHERE s.item_kind = ? AND s.model_revision = ? ORDER BY n.id",
            (cut_revision, ITEM_SUB_NEWS, summary_revision),
        ):
            if str(item_id) in done:
                continue
            yield TextItem(ITEM_SUB_NEWS, str(item_id), str(text))
            taken += 1
            if limit is not None and taken >= limit:
                return


def record_companies(
    conn: NewsNlpDatabase,
    item_kind: str,
    item_id: str,
    revision: str,
    found: list[CompanyScore],
    threshold: float,
) -> str:
    """Store the companies found in one text (one row per ticker, the main ones marked) and the state of the item, in one
    transaction; a repeat changes nothing. Returns the state: ``done`` when a company is main, ``review`` (with the reason in
    ``error``) when no company was found or none reaches the threshold: no sentiment until the owner reads it (D39 (ii))."""
    now = _now()
    for company in found:
        conn.execute(
            "INSERT OR IGNORE INTO news_company "
            "(item_kind, item_id, ticker, model_revision, relevance_score, is_main, mention_count, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_kind,
                item_id,
                company.ticker,
                revision,
                company.relevance_score,
                int(is_main(company.relevance_score, threshold)),
                company.mention_count,
                now,
            ),
        )
    best = max(found, key=lambda c: c.relevance_score, default=None)
    if best is None:
        state, error = "review", "no company found in the text"
    elif not is_main(best.relevance_score, threshold):
        state = "review"
        error = f"no company over the threshold {threshold} (best {best.ticker} {best.relevance_score:.3f})"
    else:
        state, error = "done", None
    conn.execute(
        "INSERT INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, error, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?) "
        "ON CONFLICT (item_kind, item_id, step, model_revision) DO UPDATE SET "
        "status = excluded.status, attempts = attempts + 1, error = excluded.error, finished_at = excluded.finished_at",
        (item_kind, item_id, STEP_COMPANIES, revision, state, error, now, now),
    )
    conn.commit()
    return state


def pending_sector_bodies(
    conn: NewsNlpDatabase, classifier_revision: str, revision: str, limit: int | None = None
) -> Iterator[Body]:
    """The ``sector`` bodies of step 1 the sector step has not done at ``revision``, at most ``limit`` (T-167)."""
    return _pending_scoped_bodies(conn, classifier_revision, "sector", STEP_SECTOR, revision, limit)


def record_sector(
    conn: NewsNlpDatabase,
    body: Body,
    revision: str,
    status: str,
    sector: str,
    candidate: str,
    margin: float,
    vote_sector: str,
    scores: dict[str, float],
    votes: dict[str, int],
) -> None:
    """Store one body's sector (or its ``out_of_scope`` verdict) and mark the step done in one transaction; a repeat changes nothing."""
    now = _now()
    conn.execute(
        "INSERT OR IGNORE INTO body_sector "
        "(body_hash, classifier_revision, status, sector, candidate, margin, vote_sector, scores, votes, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            body.body_hash,
            revision,
            status,
            sector,
            candidate,
            margin,
            vote_sector,
            json.dumps(scores, sort_keys=True),
            json.dumps(votes, sort_keys=True),
            now,
        ),
    )
    conn.execute(
        "INSERT INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, 'done', 1, ?, ?) "
        "ON CONFLICT (item_kind, item_id, step, model_revision) DO UPDATE SET "
        "status = 'done', attempts = attempts + 1, error = NULL, finished_at = excluded.finished_at",
        (ITEM_BODY, body.body_hash, STEP_SECTOR, revision, now, now),
    )
    conn.commit()


def record_cut(conn: NewsNlpDatabase, body: Body, revision: str, texts: list[str]) -> None:
    """Store the sub-news of one body and mark the cut done in one transaction; a repeat adds nothing.

    A body the extractor gives no text (no company is named in it) is marked ``skipped`` with the reason, so it is not
    retried at this revision and the status counts it apart.
    """
    now = _now()
    if texts:
        _ensure_article_row(conn, body.article_id)
    for index, text in enumerate(texts):
        conn.execute(
            "INSERT OR IGNORE INTO sub_news "
            "(body_hash, article_id, segment_index, text, extractor_revision, processed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (body.body_hash, body.article_id, index, text, revision, now),
        )
    state, error = ("done", None) if texts else ("skipped", "no company text")
    conn.execute(
        "INSERT INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, error, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?) "
        "ON CONFLICT (item_kind, item_id, step, model_revision) DO UPDATE SET "
        "status = excluded.status, attempts = attempts + 1, error = excluded.error, finished_at = excluded.finished_at",
        (ITEM_BODY, body.body_hash, STEP_CUT, revision, state, error, now, now),
    )
    conn.commit()


def error_text(exc: BaseException) -> str:
    """The reason stored with a failed body, the same for every step: ``Type: message``."""
    return f"{type(exc).__name__}: {exc}"


def latest_revision(conn: NewsNlpDatabase, step: str) -> str | None:
    """The model revision of the step's most recent state row, or None when the step has not run."""
    row = conn.execute(
        "SELECT model_revision FROM pipeline_step_state WHERE step = ? "
        "ORDER BY finished_at DESC, rowid DESC LIMIT 1",
        (step,),
    ).fetchone()
    return None if row is None else str(row[0])


def revisions(conn: NewsNlpDatabase, step: str) -> list[str]:
    """Every model revision the step has state for, oldest first."""
    return [
        str(r[0])
        for r in conn.execute(
            "SELECT model_revision FROM pipeline_step_state WHERE step = ? "
            "GROUP BY model_revision ORDER BY MIN(rowid)",
            (step,),
        )
    ]


def record_failure(conn: NewsNlpDatabase, body: Body, step: str, revision: str, error: str) -> None:
    """Mark a body failed with the reason; it is retried by the next run."""
    record_item_failure(conn, ITEM_BODY, body.body_hash, step, revision, error)


def record_item_failure(
    conn: NewsNlpDatabase, item_kind: str, item_id: str, step: str, revision: str, error: str
) -> None:
    """Mark an item (a body, or a sub-news by its id) failed with the reason; it is retried by the next run."""
    now = _now()
    conn.execute(
        "INSERT INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, error, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, 'failed', 1, ?, ?, ?) "
        "ON CONFLICT (item_kind, item_id, step, model_revision) DO UPDATE SET "
        "status = 'failed', attempts = attempts + 1, error = excluded.error, finished_at = excluded.finished_at",
        (item_kind, item_id, step, revision, error[:500], now, now),
    )
    conn.commit()


def missing_titles(conn: NewsNlpDatabase) -> int:
    """Unique bodies whose first article (the one the classifier reads) has no title (T-147)."""
    marks = ", ".join("?" for _ in PROPER_STATUSES)
    seen: dict[str, bool] = {}
    for title, text in conn.execute(
        f"SELECT title, body_text FROM {_articles(conn)} "  # noqa: S608 -- the schema name is main or source, never input
        f"WHERE fetch_status IN ({marks}) AND body_text IS NOT NULL AND body_text != '' ORDER BY id",
        PROPER_STATUSES,
    ):
        seen.setdefault(triage.body_hash(text), bool((title or "").strip()))
    return sum(1 for has_title in seen.values() if not has_title)


def status(conn: NewsNlpDatabase, step: str, revision: str | None = None) -> dict[str, object]:
    """Counts of the step by state (and, for the scope step, by scope); the counts add up to the rows of the state table."""
    where, args = "step = ?", [step]
    if revision is not None:
        where += " AND model_revision = ?"
        args.append(revision)
    by_state = {
        str(s): int(n)
        for s, n in conn.execute(
            f"SELECT status, COUNT(*) FROM pipeline_step_state WHERE {where} GROUP BY status",  # noqa: S608 -- fixed clauses
            args,
        )
    }
    out: dict[str, object] = {"step": step, "revision": revision, "by_state": by_state}
    failed = conn.execute(
        f"SELECT item_id, error FROM pipeline_step_state WHERE {where} AND status = 'failed' ORDER BY finished_at DESC LIMIT 3",  # noqa: S608 -- fixed clauses
        args,
    ).fetchall()
    if failed:
        out["failed_examples"] = [{"item_id": str(i), "error": str(e)[:200]} for i, e in failed]
    if step == STEP_SCOPE:
        sql = "SELECT scope, COUNT(*) FROM body_scope"
        sql_args: list[str] = []
        if revision is not None:
            sql, sql_args = sql + " WHERE classifier_revision = ?", [revision]
        out["by_scope"] = {
            str(s): int(n) for s, n in conn.execute(sql + " GROUP BY scope", sql_args)
        }
    if step == STEP_SECTOR:
        sec_w = " WHERE classifier_revision = ?" if revision is not None else ""
        sec_a = [revision] if revision is not None else []
        by_status = f"SELECT status, COUNT(*) FROM body_sector{sec_w} GROUP BY status"  # noqa: S608 -- fixed clauses
        by_sector = f"SELECT sector, COUNT(*) FROM body_sector{sec_w} GROUP BY sector"  # noqa: S608 -- fixed clauses
        out["body_sector"] = {str(s): int(n) for s, n in conn.execute(by_status, sec_a)}
        out["by_sector"] = {
            (str(s) or "out_of_scope"): int(n) for s, n in conn.execute(by_sector, sec_a)
        }
    if step == STEP_CUT:
        cut_sql = "SELECT COUNT(*), COUNT(DISTINCT body_hash) FROM sub_news"
        cut_args: list[str] = []
        if revision is not None:
            cut_sql, cut_args = cut_sql + " WHERE extractor_revision = ?", [revision]
        rows, bodies = conn.execute(cut_sql, cut_args).fetchone()
        out["sub_news"] = {"rows": int(rows), "bodies": int(bodies)}
    if step == STEP_SUMMARY:
        w = " WHERE model_revision = ?" if revision is not None else ""
        a = [revision] if revision is not None else []
        by_source = f"SELECT text_source, COUNT(*) FROM news_summary{w} GROUP BY text_source"  # noqa: S608 -- fixed clauses
        by_kind = f"SELECT item_kind, COUNT(*) FROM news_summary{w} GROUP BY item_kind"  # noqa: S608 -- fixed clauses
        out["news_summary"] = {str(s): int(n) for s, n in conn.execute(by_source, a)}
        out["by_item_kind"] = {str(k): int(n) for k, n in conn.execute(by_kind, a)}
    if step == STEP_COMPANIES:
        w = " WHERE model_revision = ?" if revision is not None else ""
        a = [revision] if revision is not None else []
        rows, main, items, with_main = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(is_main), 0), COUNT(DISTINCT item_kind || ':' || item_id), "  # noqa: S608 -- fixed clauses
            "COUNT(DISTINCT CASE WHEN is_main = 1 THEN item_kind || ':' || item_id END) "
            f"FROM news_company{w}",
            a,
        ).fetchone()
        out["news_company"] = {
            "rows": int(rows),
            "main": int(main),
            "items": int(items),
            "items_with_main": int(with_main),
        }
    return out
