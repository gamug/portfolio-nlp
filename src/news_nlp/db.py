"""Two-tier SOURCE / RESULTS connection machinery for the news-NLP pipeline.

* RESULTS store -- selected by ``$DATABASE_URL`` (:func:`env.results_db_path`),
  opened read/write as schema ``main``. Holds the five result tables plus a lean
  ``articles`` subset (every column except ``body_text``). Everything the FastAPI
  query/correction endpoints read comes from here.
* SOURCE store -- selected by ``$SOURCE_DATABASE_URL`` (:func:`env.source_db_path`),
  opened read-only (``file:...?mode=ro``) and ATTACHed as schema ``source``.
  Holds ``articles`` including ``body_text``, written by the upstream crawler.
  Required by the text-reading pipeline stages; never written. Not needed for
  serving or the ``sector_summary`` stage.

``connect()`` opens a plain single-file connection (serving, tests, single-file
runs). ``connect_pipeline()`` opens the RESULTS store and, unless SOURCE resolves
to the same path, ATTACHes SOURCE read-only. The three ``body_text`` readers
qualify ``articles`` with ``db.articles_rel`` (``"source"`` when attached, else
``"main"``); the pipeline's write helpers upsert a lean ``main.articles`` row so
the RESULTS store stays foreign-key-consistent.

:class:`NewsNlpDatabase` subclasses :class:`portfolio_common.db.Database`
(a plain composition wrapper around ``sqlite3.Connection``) and carries
``articles_rel`` / ``lean_article_cols`` as ordinary Python instance
attributes -- the ``sqlite3.Connection`` subclassing trick the pre-extraction
code needed (base ``sqlite3.Connection`` rejects arbitrary instance
attributes) is no longer necessary.

See ``docs/db-topology.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

from portfolio_common.db import Allowlist, Database

from news_nlp import env

_NO_SOURCE_MSG = (
    "SOURCE_DATABASE_URL is not set. The text-reading pipeline stages "
    "(sentiment, NER, category, c_summary) require a read-only source database "
    "that has articles.body_text (e.g. urls.db). Set SOURCE_DATABASE_URL. "
    "Serving/query endpoints and the sector_summary stage do not need it. "
    "See docs/db-topology.md."
)
_NO_SOURCE_TEXT_MSG = (
    "SOURCE database has no usable article text: articles.body_text is missing "
    "or entirely empty. Point SOURCE_DATABASE_URL at the crawl database "
    "(e.g. urls.db), not the results store. "
    "See docs/db-topology.md."
)

# Only ever "main" (single-file / serving) or "source" (a SOURCE DB is
# ATTACHed) -- both `_articles_rel` itself and the two `attach_source` /
# `detach_source` writers that flip `articles_rel` only ever set one of these
# two literals, never caller input. This Allowlist makes that guarantee
# self-enforcing at every f-string interpolation site in this module and in
# queries.py, rather than resting on a comment alone.
_ARTICLES_SCHEMA = Allowlist("main", "source")


class NewsNlpDatabase(Database):
    """A :class:`~portfolio_common.db.Database` plus the two-tier
    SOURCE/RESULTS attach state the ``body_text`` readers and result-table
    writers need: ``articles_rel`` (``"source"`` once a read-only SOURCE DB
    is ATTACHed by :func:`attach_source`, else ``"main"``) and
    ``lean_article_cols`` (the SOURCE->RESULTS column list, cached by
    :func:`attach_source` for :func:`_ensure_article_row`)."""

    articles_rel: str = "main"
    lean_article_cols: list[str] | None = None


def _articles_rel(db: NewsNlpDatabase) -> str:
    """The schema the `body_text` readers qualify `articles` with: ``"source"``
    when a read-only SOURCE DB is attached, else ``"main"``.

    Only :func:`attach_source` / :func:`detach_source` ever set
    ``db.articles_rel`` (never caller input), so this could not actually
    return anything outside :data:`_ARTICLES_SCHEMA` today -- the
    :meth:`~portfolio_common.db.Allowlist.check` call makes that guarantee
    self-enforcing at the one place every ``{_articles_rel(db)}.articles``
    f-string interpolation (here and throughout queries.py) draws from,
    rather than resting on this docstring's word alone.
    """
    return _ARTICLES_SCHEMA.check(getattr(db, "articles_rel", "main"))


def connect(db_path: str | os.PathLike[str] | None = None) -> NewsNlpDatabase:
    """Open one plain SQLite file read/write (serving, tests, single-file runs).
    ``db_path`` defaults to :func:`env.results_db_path`. For a pipeline run that
    needs the SOURCE DB attached, use connect_pipeline().

    Delegates to :meth:`portfolio_common.db.Database.connect` (row factory,
    busy_timeout, ``PRAGMA foreign_keys = ON``); ``uri=True`` so a later
    ``ATTACH ...?mode=ro`` on this connection is honored.
    """
    path = Path(db_path) if db_path is not None else env.results_db_path()
    return NewsNlpDatabase.connect(f"file:{path.as_posix()}", uri=True, foreign_keys=True)


def connect_pipeline(
    results_db: str | os.PathLike[str] | None = None,
    source_db: str | os.PathLike[str] | None = None,
) -> NewsNlpDatabase:
    """Open the RESULTS store read/write and, unless SOURCE resolves to the same
    path, ATTACH the SOURCE store read-only as schema `source`.

    Paths are read from the environment in the body (not bound as defaults) so
    tests that set ``$DATABASE_URL`` / ``$SOURCE_DATABASE_URL`` -- or pass
    explicit overrides -- take effect. Raises RuntimeError if no SOURCE is
    configured -- the text-reading stages cannot run without one.
    """
    results = env.results_db_path(results_db)
    source = env.source_db_path(source_db)
    if source is None:
        raise RuntimeError(_NO_SOURCE_MSG)
    db = connect(results)
    if source.resolve() != results.resolve():
        attach_source(db, source)
    return db


def _compute_lean_article_columns(db: NewsNlpDatabase) -> list[str]:
    """`articles` columns present in both the attached SOURCE and RESULTS
    schemas, minus `body_text`, in SOURCE column order."""
    source_cols = [row[1] for row in db.execute("PRAGMA source.table_info(articles)")]
    main_cols = {row[1] for row in db.execute("PRAGMA main.table_info(articles)")}
    return [c for c in source_cols if c != "body_text" and c in main_cols]


def attach_source(db: NewsNlpDatabase, source: str | os.PathLike[str]) -> None:
    """ATTACH `source` read-only as schema `source`; flip `articles_rel` and
    cache the lean column list for _ensure_article_row.

    The stale-WAL preflight (a read-only ``mode=ro`` open cannot replay a
    leftover write-ahead log, so a SOURCE left with an un-checkpointed
    ``-wal`` -- crawler killed mid-run, or the file copied without
    checkpointing -- would otherwise silently serve stale data instead of
    failing loudly) is implemented generically by
    :meth:`portfolio_common.db.Database.attach`, not reimplemented here. Its
    ``RuntimeError`` (stale WAL, or the ATTACH itself failing) is left to
    propagate as-is.
    """
    db.attach(source, "source", read_only=True)
    db.articles_rel = "source"
    db.lean_article_cols = _compute_lean_article_columns(db)


def detach_source(db: NewsNlpDatabase) -> None:
    """Undo attach_source(). Safe to call when nothing is attached."""
    if db.articles_rel == "source":
        db.detach("source")
        db.articles_rel = "main"
        db.lean_article_cols = None


def _ensure_article_row(db: NewsNlpDatabase, article_id: int) -> None:
    """Copy the lean `articles` row (no `body_text`) for `article_id` from
    SOURCE into RESULTS if it isn't there yet, so a result-table write for it
    satisfies the `REFERENCES articles(id)` foreign key. No-op unless a SOURCE
    DB is attached (single-file runs already have the full `articles` table)."""
    if _articles_rel(db) != "source":
        return
    lean_cols = getattr(db, "lean_article_cols", None) or _compute_lean_article_columns(db)
    cols = ", ".join(lean_cols)
    db.execute(
        f"INSERT OR IGNORE INTO main.articles ({cols}) "  # noqa: S608
        f"SELECT {cols} FROM source.articles WHERE id = ?",
        (article_id,),
    )


def require_source_text(db: NewsNlpDatabase) -> None:
    """Fail fast (before any model loads) if the `articles` table the
    `body_text` readers will hit has no usable text -- the common
    misconfiguration of pointing a text stage at the results store. Structural
    check ("does this DB hold article text at all"), not "is anything pending":
    a fully caught-up pipeline still passes."""
    schema = _articles_rel(db)  # Allowlist-checked -- see _articles_rel
    cols = {row[1] for row in db.execute(f"PRAGMA {schema}.table_info(articles)")}
    if "body_text" not in cols:
        raise RuntimeError(_NO_SOURCE_TEXT_MSG)
    row = db.execute(
        f"SELECT 1 FROM {schema}.articles "  # noqa: S608
        "WHERE body_text IS NOT NULL AND TRIM(body_text) != '' LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError(_NO_SOURCE_TEXT_MSG)
