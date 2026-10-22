"""Two-tier SOURCE / RESULTS connection machinery for the news-NLP pipeline.

* RESULTS store -- selected by ``$DATABASE_URL`` (:func:`env.results_db_path`),
  opened read/write as schema ``main``. Holds the five result tables plus a lean
  ``articles`` subset (every column except ``body_text``). Everything the FastAPI
  query/correction endpoints read comes from here.
* SOURCE store -- selected by ``$SOURCE_DATABASE_URL`` (:func:`env.source_db_path`),
  opened read-only and ATTACHed as schema ``source``. Holds ``articles``
  including ``body_text``, written by the upstream crawler. Required by the
  text-reading pipeline stages; never written. Not needed for serving or the
  ``sector_summary`` stage.

``connect()`` opens a plain single-file connection (serving, tests, single-file
runs). ``connect_pipeline()`` opens the RESULTS store and, unless SOURCE resolves
to the same path, ATTACHes SOURCE read-only. The three ``body_text`` readers
qualify ``articles`` with ``db.articles_rel`` (``"source"`` when attached, else
``"main"``); the pipeline's write helpers upsert a lean ``main.articles`` row so
the RESULTS store stays foreign-key-consistent.

:class:`NewsNlpDatabase` subclasses
:class:`portfolio_common.db.TwoTierDatabase` -- which itself is a
:class:`~portfolio_common.db.Database` (a composition wrapper, not a
``sqlite3.Connection`` subclass) that tracks which schema a shared table is
currently read from (``read_schema``). This module keeps no engine-specific
SQL: the ATTACH open goes through
:func:`portfolio_common.db.connect_two_store`, schema introspection through
:meth:`~portfolio_common.db.Database.table_columns`, and the lean cross-store
row copy through :meth:`~portfolio_common.db.Database.copy_row_lean`. ``which
database engine`` is a ``portfolio-common`` concern -- see
``docs/engine-agnostic-rollout.md``.

See ``docs/db-topology.md``.
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from pathlib import Path
from typing import Protocol

from portfolio_common.db import Allowlist, TwoTierDatabase, connect_two_store

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

# The SOURCE (`urls.db`) schema contract this repo relies on. `PRAGMA user_version` is
# stamped by `portfolio-data-mining`; the number is copied here, never imported
# (that would couple this repo to its package). Upstream's version table (hand-off
# 2026-10-21, 8.1): `discovered_urls.match_class` arrives at 2, `articles.prev_fetch_status`
# and the quarantine semantics at 3; 4 and 5 add nothing read here. Migrations are
# additive and forward-only, so a minimum is checked, not an exact match (a documented
# cache rollback re-stamps 5 back to 3). Cache tables (`sitemap_*`, `discovery_progress`)
# are not part of the contract: never read, join or attach them.
MIN_SOURCE_SCHEMA_VERSION = 3

# The highest version this repo has seen (today's real file). Above it, warn once per
# process: a newer schema is probably fine (additive) but nobody here has checked it.
LATEST_KNOWN_SOURCE_SCHEMA_VERSION = 5

# Columns read from SOURCE today. A new SOURCE table or column read adds itself here
# in the same PR (T-132 adds `prev_fetch_status` and `discovered_urls`).
SOURCE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "articles": ("id", "ticker", "company", "pub_date", "body_text", "fetch_status"),
}

_warned_newer_schema = False


class SourceSchemaError(RuntimeError):
    """SOURCE is not a database this repo can read: its schema version is below the
    minimum (or unstamped), or a required column is missing. A configuration error,
    not a client error."""


class _Executes(Protocol):
    def execute(self, sql: str, /) -> sqlite3.Cursor: ...


def check_schema(
    source: str | os.PathLike[str] | _Executes,
    minimum: int = MIN_SOURCE_SCHEMA_VERSION,
    columns: dict[str, tuple[str, ...]] | None = None,
    *,
    schema: str = "main",
    name: str | None = None,
) -> int:
    """Check a SOURCE database's ``PRAGMA user_version`` and required columns; return
    the version found. Read-only; nothing is cached between calls.

    *source* is a path (opened here ``mode=ro`` and closed) or an open connection
    (anything with ``execute``; *schema* names the database on it, ``"source"`` for an
    ATTACHed one), so scripts can call it without a ``NewsNlpDatabase``. Raises
    :class:`SourceSchemaError` on a version below *minimum* (0 included) and on a missing
    column even when the number passes ("the number is a promise, the columns are the
    fact"); warns once per process on a higher version.
    """
    global _warned_newer_schema  # noqa: PLW0603 -- once-per-process flag by design
    wanted = SOURCE_REQUIRED_COLUMNS if columns is None else columns
    if isinstance(source, str | os.PathLike):
        path = Path(source)
        raw = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            return check_schema(raw, minimum, wanted, schema="main", name=name or str(path))
        finally:
            raw.close()
    if not schema.isidentifier():
        raise ValueError(f"schema must be an identifier, got {schema!r}")
    label = name or schema
    version = int(source.execute(f"PRAGMA {schema}.user_version").fetchone()[0])
    if version < minimum:
        raise SourceSchemaError(
            f"SOURCE database {label} is at schema version {version}; this repo needs "
            f"{minimum} or later{' (0 means never stamped: not a data-mining database)' if version == 0 else ''}. "
            "Start the collector or the extractor so it migrates the file, or point "
            "SOURCE_DATABASE_URL at the right file. See docs/db-topology.md."
        )
    for table, cols in wanted.items():
        if not table.isidentifier():
            raise ValueError(f"table must be an identifier, got {table!r}")
        present = {row[1] for row in source.execute(f"PRAGMA {schema}.table_info({table})")}
        missing = [c for c in cols if c not in present]
        if missing:
            raise SourceSchemaError(
                f"SOURCE database {label} (schema version {version}) has no "
                f"{table}.{', '.join(missing)}; the number passed but the columns this repo "
                "reads are not all there. Point SOURCE_DATABASE_URL at the crawl database "
                "(e.g. urls.db). See docs/db-topology.md."
            )
    if version > LATEST_KNOWN_SOURCE_SCHEMA_VERSION and not _warned_newer_schema:
        _warned_newer_schema = True
        warnings.warn(
            f"SOURCE database {label} is at schema version {version}, newer than the "
            f"latest this repo has seen ({LATEST_KNOWN_SOURCE_SCHEMA_VERSION}); migrations are "
            "additive, so continuing.",
            stacklevel=2,
        )
    return version


def check_source(
    source_db: str | os.PathLike[str] | None = None,
    results_db: str | os.PathLike[str] | None = None,
) -> None:
    """Run :func:`check_schema` on the configured SOURCE *before* a run starts (the API
    handler does this, ahead of its background worker). A no-op when no SOURCE is
    configured or it is the same file as RESULTS (nothing is attached then; the
    pipeline raises its own message)."""
    source = env.source_db_path(source_db)
    if source is None or source.resolve() == env.results_db_path(results_db).resolve():
        return
    check_schema(source)


# Only ever "main" (single-file / serving) or "source" (a SOURCE DB is
# ATTACHed) -- `_articles_rel` reads it off the connection's tracked
# `read_schema`, which `portfolio_common.db.TwoTierDatabase` only ever sets to
# one of those two, never caller input. This Allowlist makes that guarantee
# self-enforcing at every f-string interpolation site in this module and in
# queries.py, rather than resting on a comment alone.
_ARTICLES_SCHEMA = Allowlist("main", "source")


class NewsNlpDatabase(TwoTierDatabase):
    """A :class:`~portfolio_common.db.TwoTierDatabase` with a domain-friendly
    name for its ``read_schema``: ``articles_rel`` is ``"source"`` once a
    read-only SOURCE DB is ATTACHed by :func:`attach_source`, else ``"main"``.
    The SOURCE->RESULTS lean-column list is resolved and cached inside
    :meth:`~portfolio_common.db.Database.copy_row_lean`, not here."""

    @property
    def articles_rel(self) -> str:
        return self.read_schema


def _articles_rel(db: NewsNlpDatabase) -> str:
    """The schema the `body_text` readers qualify `articles` with: ``"source"``
    when a read-only SOURCE DB is attached, else ``"main"``.

    Only :func:`attach_source` / :func:`detach_source` (via
    ``TwoTierDatabase``) ever move ``db.articles_rel`` off ``"main"``, and only
    to ``"source"`` -- never caller input -- so this could not actually return
    anything outside :data:`_ARTICLES_SCHEMA` today. The
    :meth:`~portfolio_common.db.Allowlist.check` call makes that guarantee
    self-enforcing at the one place every ``{_articles_rel(db)}.articles``
    f-string interpolation (here and throughout queries.py) draws from, rather
    than resting on this docstring's word alone.
    """
    return _ARTICLES_SCHEMA.check(db.articles_rel)


def connect(db_path: str | os.PathLike[str] | None = None) -> NewsNlpDatabase:
    """Open one plain database read/write (serving, tests, single-file runs).
    ``db_path`` defaults to :func:`env.results_db_path`. For a pipeline run that
    needs the SOURCE DB attached, use connect_pipeline().

    Goes through :meth:`portfolio_common.db.Database.connect_url` (which picks
    the engine from the value -- a filesystem path today), with the shared
    pragma policy plus ``PRAGMA foreign_keys = ON``; the connection is opened
    URI-mode so a later read-only ``ATTACH`` on it is honored.
    """
    target = db_path if db_path is not None else env.results_db_path()
    return NewsNlpDatabase.connect_url(target, foreign_keys=True)


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
    db, read_schema = connect_two_store(
        results, source, alias="source", factory=NewsNlpDatabase, foreign_keys=True
    )
    if read_schema == "source":  # a distinct SOURCE is attached: check it before any query
        try:
            check_schema(db, schema="source", name=str(source))
        except Exception:
            db.close()
            raise
    return db


def attach_source(db: NewsNlpDatabase, source: str | os.PathLike[str]) -> None:
    """ATTACH `source` read-only as schema `source` and flip `articles_rel`.

    The orphaned-WAL preflight is implemented generically by
    :meth:`portfolio_common.db.Database.attach`, not reimplemented here. As of
    `portfolio-common` v1.2.2 it does *not* reject a SOURCE just because its
    ``-wal`` is non-empty -- a live writer (e.g. a concurrently-running
    `portfolio-data-mining` collector) or a hard-killed one both leave a
    matching ``-shm`` alongside, and SQLite's own WAL recovery handles both
    safely on a ``mode=ro`` open. It only rejects a ``-wal`` copied without its
    ``-shm`` (nothing then proves the two files are one consistent snapshot) --
    e.g. a backup/rsync that skipped the ``-shm``, SQLite's own docs call it
    disposable. Its ``RuntimeError`` (orphaned WAL, or the ATTACH itself
    failing) is left to propagate as-is.
    """
    db.attach_readonly("source", source)


def detach_source(db: NewsNlpDatabase) -> None:
    """Undo attach_source(). Safe to call when nothing is attached."""
    if db.read_schema == "source":
        db.detach("source")


def _ensure_article_row(db: NewsNlpDatabase, article_id: int) -> None:
    """Copy the lean `articles` row (no `body_text`) for `article_id` from
    SOURCE into RESULTS if it isn't there yet, so a result-table write for it
    satisfies the `REFERENCES articles(id)` foreign key. No-op unless a SOURCE
    DB is attached (single-file runs already have the full `articles` table).

    The shared-column resolution (SOURCE ∩ RESULTS columns, minus ``body_text``,
    in SOURCE order) and its per-connection cache live in
    :meth:`portfolio_common.db.Database.copy_row_lean`.
    """
    if _articles_rel(db) != "source":
        return
    db.copy_row_lean(
        "main.articles",
        "source.articles",
        key="id",
        key_value=article_id,
        exclude=("body_text",),
    )


def require_source_text(db: NewsNlpDatabase) -> None:
    """Fail fast (before any model loads) if the `articles` table the
    `body_text` readers will hit has no usable text -- the common
    misconfiguration of pointing a text stage at the results store. Structural
    check ("does this DB hold article text at all"), not "is anything pending":
    a fully caught-up pipeline still passes."""
    schema = _articles_rel(db)  # Allowlist-checked -- "main" or "source"
    if "body_text" not in db.table_columns("articles", schema=schema):
        raise RuntimeError(_NO_SOURCE_TEXT_MSG)
    row = db.execute(
        f"SELECT 1 FROM {schema}.articles "  # noqa: S608 -- schema is Allowlist-checked
        "WHERE body_text IS NOT NULL AND TRIM(body_text) != '' LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError(_NO_SOURCE_TEXT_MSG)
