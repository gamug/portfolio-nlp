"""Manual edit/delete operations on sentiment, entity, and category results.

Trace of capabilities -- every function in this module, grouped read/write
(all read-modify-write on a single article/entity, no bulk operations):

Write:
    update_sentiment              -- edit one article's sentiment fields, refresh processed_at
    delete_sentiment               -- remove one article's sentiment row
    update_entity                  -- edit one entity row's fields
    delete_entity                  -- remove one entity row by id
    delete_entities_for_article    -- remove all of one article's entity rows
    update_category                -- edit one article's category label/score, refresh processed_at
    delete_category                -- remove one article's category row

Kept separate from the pipeline write-path (queries.write_sentiment /
write_entities / write_category) since this is a distinct concern: human
correction of model output rather than model-generated writes. Deleting a
sentiment row (or all of an article's entities, or its category) makes that
article eligible for reprocessing again, since queries.fetch_pending_articles /
fetch_pending_category_articles select rows missing from the result table.

Engine assumptions here, both DB-API level rather than SQLite-specific and so
left as-is: the ``?`` qmark parameter marker (a non-qmark engine would take it
from ``conn.dialect.placeholder``), and ``conn.execute(...).rowcount`` for
"did the UPDATE/DELETE hit a row" -- reliable on SQLite; the DB-API permits a
driver to return ``-1`` "unknown", which is the seam to revisit if that ever
happens.
"""

from datetime import UTC, datetime
from typing import Any

from portfolio_common.db import Allowlist

from news_nlp.db import NewsNlpDatabase

# Each Allowlist below is the "which field names may a caller edit on this
# table" contract for its update_*() function: every accepted name maps
# 1:1 to a real column, so the "col = ?" SET-clause fragment built from a
# checked name is always a literal column reference, never
# caller-controlled text. .check() raises ValueError on anything else --
# update_*() never silently drops or substitutes an unrecognized field.
_SENTIMENT_FIELDS = Allowlist("label", "score", "positive", "negative", "neutral")
_ENTITY_FIELDS = Allowlist("entity_type", "text", "start_char", "end_char", "score")
# Deliberately excludes the 9 raw distribution columns: those exist as an
# audit trail of what the model actually scored (for CATEGORY_CONFIDENCE_
# THRESHOLD tuning), so a human correction only changes the winning label/
# score, never rewrites the original NLI breakdown.
_CATEGORY_FIELDS = Allowlist("label", "score")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _set_clause(fields: dict[str, Any], allowed: Allowlist) -> str:
    """`", ".join("col = ?" for each checked field name)` -- every name in
    `fields` is checked against `allowed` first (raises ValueError, listing
    the legal names, on anything unrecognized), so the returned text is
    built only from real column names, never from unchecked caller input."""
    return ", ".join(f"{allowed.check(k)} = ?" for k in fields)


def update_sentiment(conn: NewsNlpDatabase, article_id: int, **fields: Any) -> dict | None:
    if fields:
        # S608: every name in `fields` is checked against _SENTIMENT_FIELDS
        # by _set_clause before it reaches this f-string; values are always
        # bound as query params.
        set_clause = _set_clause(fields, _SENTIMENT_FIELDS) + ", processed_at = ?"
        params = [*list(fields.values()), _now_iso(), article_id]
        cur = conn.execute(
            f"UPDATE article_sentiment SET {set_clause} WHERE article_id = ?",  # noqa: S608
            params,
        )
        if cur.rowcount == 0:
            return None

    row = conn.execute(
        "SELECT * FROM article_sentiment WHERE article_id = ?", (article_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_sentiment(conn: NewsNlpDatabase, article_id: int) -> bool:
    cur = conn.execute("DELETE FROM article_sentiment WHERE article_id = ?", (article_id,))
    return cur.rowcount > 0


def update_entity(conn: NewsNlpDatabase, entity_id: int, **fields: Any) -> dict | None:
    if fields:
        # S608: every name in `fields` is checked against _ENTITY_FIELDS by
        # _set_clause before it reaches this f-string; values are always
        # bound as query params.
        set_clause = _set_clause(fields, _ENTITY_FIELDS)
        params = [*list(fields.values()), entity_id]
        cur = conn.execute(f"UPDATE article_entities SET {set_clause} WHERE id = ?", params)  # noqa: S608
        if cur.rowcount == 0:
            return None

    row = conn.execute("SELECT * FROM article_entities WHERE id = ?", (entity_id,)).fetchone()
    return dict(row) if row else None


def delete_entity(conn: NewsNlpDatabase, entity_id: int) -> bool:
    cur = conn.execute("DELETE FROM article_entities WHERE id = ?", (entity_id,))
    return cur.rowcount > 0


def delete_entities_for_article(conn: NewsNlpDatabase, article_id: int) -> int:
    cur = conn.execute("DELETE FROM article_entities WHERE article_id = ?", (article_id,))
    return cur.rowcount


def update_category(conn: NewsNlpDatabase, article_id: int, **fields: Any) -> dict | None:
    if fields:
        # S608: every name in `fields` is checked against _CATEGORY_FIELDS by
        # _set_clause before it reaches this f-string; values are always
        # bound as query params.
        set_clause = _set_clause(fields, _CATEGORY_FIELDS) + ", processed_at = ?"
        params = [*list(fields.values()), _now_iso(), article_id]
        cur = conn.execute(f"UPDATE article_category SET {set_clause} WHERE article_id = ?", params)  # noqa: S608
        if cur.rowcount == 0:
            return None

    row = conn.execute(
        "SELECT * FROM article_category WHERE article_id = ?", (article_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_category(conn: NewsNlpDatabase, article_id: int) -> bool:
    cur = conn.execute("DELETE FROM article_category WHERE article_id = ?", (article_id,))
    return cur.rowcount > 0
