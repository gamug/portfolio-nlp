"""Pull an evaluation sample of stored predictions + source text per stage.

Each run's sample is a fixed **low-confidence + uniform-random** split
(``low_conf_frac`` defaults to 0.6). The **low-confidence bucket is the
``n_low`` least-confident rows, deterministically** -- so every run re-checks the
true worst case; run-to-run variety comes from the seeded random bucket.
"Least confident" is stage-specific:

* sentiment  -- lowest ``article_sentiment.score``
* category   -- winning ``score`` within ±0.1 of ``CATEGORY_CONFIDENCE_THRESHOLD``
               or ``label = 'other'``, lowest ``score`` first
* ner        -- lowest per-article ``MIN(article_entities.score)``
* c_summary  -- no score exists, so: articles whose sentiment/NER inputs were
               themselves low-confidence (random top-up otherwise)

Reads ``articles`` through ``news_nlp.db._articles_rel`` (``"source"`` during a
two-tier run) exactly like the pipeline's own readers, so callers need only
``connect_pipeline()``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from news_nlp.db import NewsNlpDatabase, _articles_rel
from news_nlp.taxonomy import CATEGORY_CONFIDENCE_THRESHOLD, CATEGORY_SLUGS

STAGES: tuple[str, ...] = ("sentiment", "category", "ner", "c_summary")

_RESULT_TABLE = {
    "sentiment": "article_sentiment",
    "category": "article_category",
    "ner": "article_entities",
    "c_summary": "article_summary",
}

# Cap the article text handed to the judge, to bound token cost. News articles
# are inverted-pyramid, so the head carries the topic -- that's a good proxy for
# `category`, which deliberately classifies only the lead chunk in the pipeline
# (docs/modules/news-nlp.md). It is NOT a good proxy for `sentiment`:
# `run_sentiment_stage` (src/pipeline.py) chunks and scores the ENTIRE
# body_text (a token-weighted average of per-chunk softmax probabilities), so
# capping the judge's view to the lead compares it against text FinBERT never
# saw the whole of. `sentiment` is therefore judged on (in practice) the full
# body_text -- see _UNCAPPED_STAGES / _SENTIMENT_MAX_BODY_CHARS below.
#
# Empirically (docs/evaluation.md's "Follow-up" note), this truncation
# explained part of the 2026-09-08 score gap -- clearly for `positive`/
# `neutral` -- but left `negative` agreement essentially unchanged, even
# though `negative` drives most of the low macro F1; the deeper cause there is
# a separate, still-open pipeline-capability gap, not this cap.
#
# `ner` and `c_summary` also chunk/reduce over the full article in the
# pipeline and are *suspected* of the same lead-only-judge mismatch, but that
# has not been empirically verified the way sentiment was -- left capped for
# now; see docs/evaluation.md's sentiment follow-up note.
_MAX_BODY_CHARS = 6000

# Stages judged on (in practice) the full body_text -- the pipeline stage
# itself scores/reduces over the whole article, not just the lead. Every
# other stage in STAGES stays capped at _MAX_BODY_CHARS.
_UNCAPPED_STAGES: frozenset[str] = frozenset({"sentiment"})

# A safety ceiling for _UNCAPPED_STAGES, distinct from (and far more generous
# than) _MAX_BODY_CHARS: the longest article body observed as of 2026-09-08 is
# ~35K chars (~7-9K tokens), so this never engages against real data today --
# it exists so a future pathological/malformed body_text can't produce an
# oversized judge request that fails outright (silently degrading that row to
# parse_failed) instead of just losing some tail context, which is the
# acceptable-degradation failure mode a plain length cap gives us. ~2.9x the
# longest observed body; comfortably inside any modern chat model's context
# window, but not a measured DeepSeek token budget -- if judge failures ever
# correlate with body length near this ceiling, that's the signal to replace
# it with an actual tokenizer-based budget instead of a char-count proxy.
_SENTIMENT_MAX_BODY_CHARS = 100_000
# Cap the entity list shown to the NER judge -- some articles have 100s.
_MAX_NER_ENTITIES = 60
# For c_summary: how wide a low-confidence net to cast over the upstream stages.
_CSUMMARY_WEAK_MULT = 4


@dataclass(frozen=True)
class EvalItem:
    """One row to judge: the source text plus the model's stored prediction."""

    article_id: int
    bucket: str  # "low_conf" | "random"
    title: str
    body_text: str
    prediction: dict[str, Any]


class _NoSourceTextError(RuntimeError):
    pass


def _require_source(conn: NewsNlpDatabase) -> str:
    schema = _articles_rel(conn)
    if "body_text" not in conn.table_columns("articles", schema=schema):
        raise _NoSourceTextError(
            "news_nlp.eval needs the SOURCE store attached (articles.body_text). "
            "Run against connect_pipeline() with SOURCE_DATABASE_URL set. See docs/db-topology.md."
        )
    return schema


def _all_ids(conn: NewsNlpDatabase, stage: str) -> list[int]:
    """Ids of every judged row for *stage*, in a fixed order.

    ``sample_for_stage`` feeds this straight into ``rng.shuffle`` -- Fisher-Yates
    over a fixed seed still needs a *stable* input order for two runs to draw
    the same "random" rows, so this is explicitly ``ORDER BY article_id`` rather
    than relying on SQLite's incidental (undocumented, not guaranteed) rowid
    table-scan order.
    """
    # `table`/`distinct` are drawn from the fixed internal _RESULT_TABLE dict /
    # a literal above -- never caller or user input -- so this isn't a SQL
    # injection risk despite the f-string; flagged by generic static scanners
    # that pattern-match any interpolated-string execute() call regardless of
    # where the interpolated value comes from (this project uses no ORM).
    table = _RESULT_TABLE[stage]
    distinct = "DISTINCT " if stage == "ner" else ""
    rows = conn.execute(
        f"SELECT {distinct}article_id FROM {table} ORDER BY article_id"  # noqa: S608
    ).fetchall()
    return [int(r[0]) for r in rows]


def _low_conf_ids(conn: NewsNlpDatabase, stage: str, limit: int) -> list[int]:
    """Up to *limit* article ids, least-confident first."""
    if limit <= 0:
        return []
    if stage == "sentiment":
        rows = conn.execute(
            "SELECT article_id FROM article_sentiment ORDER BY score ASC LIMIT ?", (limit,)
        ).fetchall()
        return [int(r[0]) for r in rows]
    if stage == "category":
        lo = CATEGORY_CONFIDENCE_THRESHOLD - 0.1
        hi = CATEGORY_CONFIDENCE_THRESHOLD + 0.1
        rows = conn.execute(
            "SELECT article_id FROM article_category "
            "WHERE (score BETWEEN ? AND ?) OR label = 'other' "
            "ORDER BY score ASC LIMIT ?",
            (lo, hi, limit),
        ).fetchall()
        return [int(r[0]) for r in rows]
    if stage == "ner":
        rows = conn.execute(
            "SELECT article_id FROM article_entities "
            "GROUP BY article_id ORDER BY MIN(score) ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [int(r[0]) for r in rows]
    if stage == "c_summary":
        net = limit * _CSUMMARY_WEAK_MULT
        weak = set(_low_conf_ids(conn, "sentiment", net)) | set(_low_conf_ids(conn, "ner", net))
        rows = conn.execute("SELECT article_id FROM article_summary ORDER BY article_id").fetchall()
        return [int(r[0]) for r in rows if int(r[0]) in weak][:limit]
    raise ValueError(f"unknown stage {stage!r}")


def _prediction(conn: NewsNlpDatabase, stage: str, article_id: int) -> dict[str, Any] | None:
    if stage == "sentiment":
        row = conn.execute(
            "SELECT label, score, positive, negative, neutral "
            "FROM article_sentiment WHERE article_id = ?",
            (article_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    if stage == "category":
        row = conn.execute(
            "SELECT * FROM article_category WHERE article_id = ?", (article_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        return {
            "label": d["label"],
            "score": d["score"],
            "distribution": {slug: d[slug] for slug in CATEGORY_SLUGS},
        }
    if stage == "ner":
        rows = conn.execute(
            "SELECT entity_type, text, start_char, end_char, score "
            "FROM article_entities WHERE article_id = ? ORDER BY start_char",
            (article_id,),
        ).fetchall()
        ents = [dict(r) for r in rows]
        out: dict[str, Any] = {"entities": ents[:_MAX_NER_ENTITIES], "n_entities": len(ents)}
        if len(ents) > _MAX_NER_ENTITIES:
            out["entities_truncated"] = True
        return out
    if stage == "c_summary":
        row = conn.execute(
            "SELECT summary_text, num_chunks FROM article_summary WHERE article_id = ?",
            (article_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    raise ValueError(f"unknown stage {stage!r}")


def _text(
    conn: NewsNlpDatabase, schema: str, article_id: int, stage: str
) -> tuple[str, str] | None:
    row = conn.execute(
        f"SELECT title, body_text FROM {schema}.articles WHERE id = ?",  # noqa: S608
        (article_id,),
    ).fetchone()
    if row is None or not row["body_text"]:
        return None
    body = row["body_text"]
    cap = _SENTIMENT_MAX_BODY_CHARS if stage in _UNCAPPED_STAGES else _MAX_BODY_CHARS
    if len(body) > cap:
        body = body[:cap] + "\n[... truncated ...]"
    return (row["title"] or "", body)


def sample_for_stage(
    conn: NewsNlpDatabase,
    stage: str,
    *,
    size: int,
    low_conf_frac: float = 0.6,
    seed: int | None = None,
) -> list[EvalItem]:
    """Return up to *size* ``EvalItem``s for *stage*: the ``round(size *
    low_conf_frac)`` least-confident rows (``bucket="low_conf"``), then a seeded
    uniform-random draw from the rest (``bucket="random"``). Fewer than *size*
    only when the store holds too few judged rows."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    schema = _require_source(conn)
    rng = random.Random(seed)  # noqa: S311 -- sample selection, not cryptography

    n_low = round(size * low_conf_frac)
    chosen_low = _low_conf_ids(conn, stage, n_low)

    taken = set(chosen_low)
    rest = [i for i in _all_ids(conn, stage) if i not in taken]
    rng.shuffle(rest)
    chosen_random = rest[: size - len(chosen_low)]

    items: list[EvalItem] = []
    for bucket, ids in (("low_conf", chosen_low), ("random", chosen_random)):
        for article_id in ids:
            text = _text(conn, schema, article_id, stage)
            pred = _prediction(conn, stage, article_id)
            if text is None or pred is None:
                continue
            title, body = text
            items.append(
                EvalItem(
                    article_id=article_id,
                    bucket=bucket,
                    title=title,
                    body_text=body,
                    prediction=pred,
                )
            )
    return items
