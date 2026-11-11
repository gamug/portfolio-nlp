"""The stepped pipeline's run tracking and the first step's storage (PLAN D39/D40; T-162, T-163).

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

from news_nlp import triage
from news_nlp.db import NewsNlpDatabase, _ensure_article_row

STEP_SCOPE = "scope"
STEP_CUT = "cut"
ITEM_BODY = "body"
SCOPES = ("single", "multi", "sector", "out_of_scope")
# The properly fetched corpus (D36, T-159): ``quarantined`` is a verdict on the company tag, not on the article.
PROPER_STATUSES = ("ok", "quarantined")


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


def finished(conn: NewsNlpDatabase, step: str, revision: str) -> set[str]:
    """Hashes of the bodies the step has already done (or skipped) at this revision: failed and pending ones are retried."""
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT item_id FROM pipeline_step_state "
            "WHERE item_kind = ? AND step = ? AND model_revision = ? AND status IN ('done', 'skipped')",
            (ITEM_BODY, step, revision),
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


def pending_multi_bodies(
    conn: NewsNlpDatabase, classifier_revision: str, revision: str, limit: int | None = None
) -> Iterator[Body]:
    """The ``multi`` bodies of step 1 (at ``classifier_revision``) the cut has not done at ``revision``, at most ``limit``.

    ``body_scope`` keeps no text, so the bodies are found in SOURCE by hash; the scan is in article-id order, so the
    article each body carries is the lowest id of its group (T-132), and it stops when every wanted body is found.
    """
    done = finished(conn, STEP_CUT, revision)
    wanted = [
        str(row[0])
        for row in conn.execute(
            "SELECT body_hash FROM body_scope WHERE classifier_revision = ? AND scope = 'multi' ORDER BY rowid",
            (classifier_revision,),
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
    now = _now()
    conn.execute(
        "INSERT INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, error, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, 'failed', 1, ?, ?, ?) "
        "ON CONFLICT (item_kind, item_id, step, model_revision) DO UPDATE SET "
        "status = 'failed', attempts = attempts + 1, error = excluded.error, finished_at = excluded.finished_at",
        (ITEM_BODY, body.body_hash, step, revision, error[:500], now, now),
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
    if step == STEP_CUT:
        cut_sql = "SELECT COUNT(*), COUNT(DISTINCT body_hash) FROM sub_news"
        cut_args: list[str] = []
        if revision is not None:
            cut_sql, cut_args = cut_sql + " WHERE extractor_revision = ?", [revision]
        rows, bodies = conn.execute(cut_sql, cut_args).fetchone()
        out["sub_news"] = {"rows": int(rows), "bodies": int(bodies)}
    return out
