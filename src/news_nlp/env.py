"""Resolve the news-NLP two-tier database paths from the environment.

``DATABASE_URL``          the RESULTS / serving store (result tables + a lean,
                          ``body_text``-free ``articles`` subset). Has a packaged
                          default, so :func:`results_db_path` always returns a
                          usable path.
``SOURCE_DATABASE_URL``   the read-only SOURCE store, which has
                          ``articles.body_text`` (the crawl DB). No default --
                          :func:`source_db_path` returns ``None`` when unset, and
                          the text-reading pipeline stages refuse to run without
                          it (see :func:`news_nlp.db.connect_pipeline`).

Both resolve to whatever :meth:`portfolio_common.db.Database.connect_url`
accepts. That is a filesystem path today (the engine is SQLite), and a
``scheme://`` connection URL is the extension point when it isn't -- ``which
database engine`` is a ``portfolio-common`` concern, not this module's. A
**relative** path value is left relative (resolved against the process CWD
when the file is opened); an **absolute** one is used as-is; unset falls back
to :data:`DEFAULT_RESULTS_DB` for RESULTS and ``None`` for SOURCE.
"""

from __future__ import annotations

import os
from pathlib import Path

RESULTS_DB_ENV_VAR = "DATABASE_URL"
SOURCE_DB_ENV_VAR = "SOURCE_DATABASE_URL"

DEFAULT_RESULTS_DB = Path("data/nlp.db")


def _resolve(value: str | None) -> Path | None:
    """``None`` / empty -> ``None``; otherwise ``Path(value).expanduser()``
    (relative stays relative -> resolved against CWD at open time)."""
    if not value:
        return None
    return Path(value).expanduser()


def results_db_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The RESULTS store path: *explicit* wins; else ``$DATABASE_URL``; else the
    packaged default :data:`DEFAULT_RESULTS_DB`. Never ``None``."""
    if explicit is not None:
        return Path(explicit).expanduser()
    return _resolve(os.environ.get(RESULTS_DB_ENV_VAR)) or DEFAULT_RESULTS_DB


def source_db_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The SOURCE store path: *explicit* wins; else ``$SOURCE_DATABASE_URL``;
    else ``None`` (an honest "was it configured?" signal -- serving never needs
    it, and ``connect_pipeline`` raises rather than guessing)."""
    if explicit is not None:
        return Path(explicit).expanduser()
    return _resolve(os.environ.get(SOURCE_DB_ENV_VAR))
