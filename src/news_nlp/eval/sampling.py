"""Pull an evaluation sample of stored predictions + source text per stage.

Each run's sample is a **disjoint, priority-ordered stack of strata** -- see
``docs/evaluation.md``'s "Sampling" section for the full design and the
Horvitz-Thompson reweighting this feeds in ``news_nlp.eval.metrics``:

1. ``low_conf`` (``low_conf_frac`` of *size*, default 0.6) -- the deterministic
   least-confident rows, "least confident" being stage-specific (below).
   Diagnostic-only: excluded from every population-estimate metric.
2. ``target_<x>`` (0+, stage-specific; ``target_frac`` of the post-low_conf
   budget, default 0.6) -- rows clearing a threshold on a *raw per-class
   score*, regardless of which class actually won the argmax. Sentiment and
   category only: both store a full soft-probability distribution on every
   row, so a below-argmax-boundary threshold surfaces both confirmed
   predictions AND likely-missed ones (false-negative candidates) for a
   class, far cheaper than scaling ``size`` alone. c_summary has no discrete
   classes, so its "targets" instead partition on ``num_chunks``. NER has
   neither a secondary per-entity score nor NER's imbalance being severe
   enough to need this -- no target strata.
3. ``representative`` (the remainder, seeded uniform draw) -- fills out to
   *size*. The only bucket that's a plain, unweighted random sample.

"Least confident" (``low_conf``), unchanged from before this stratification
was added:

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
# `ner` also chunks/reduces over the full article in the pipeline and was
# suspected of the same lead-only-judge mismatch. CONFIRMED, not just
# suspected, against real data (docs/evaluation.md's 2026-09-12 follow-up:
# 21.4% of a real 1000-article sample exceed this cap, and 16.8% of
# predicted entities in that sample start past it -- structurally
# unverifiable by a judge that never sees that text). Fixed the same way as
# sentiment: `ner` joined _UNCAPPED_STAGES below (2026-09-12). `c_summary`'s
# version of this suspicion remains unverified.
_MAX_BODY_CHARS = 6000

# Stages judged on (in practice) the full body_text -- the pipeline stage
# itself scores/reduces over the whole article, not just the lead. Every
# other stage in STAGES stays capped at _MAX_BODY_CHARS.
_UNCAPPED_STAGES: frozenset[str] = frozenset({"sentiment", "ner"})

# A safety ceiling for _UNCAPPED_STAGES, distinct from (and far more generous
# than) _MAX_BODY_CHARS. Named for sentiment (the first stage uncapped,
# 2026-09-08) but shared by every stage in _UNCAPPED_STAGES, `ner` included
# (2026-09-12) -- not renamed, to avoid rewriting the 2026-09-08 follow-up's
# own references to this name in docs/evaluation.md, which stay historically
# accurate under the old name. The longest *sentiment* article body observed
# as of 2026-09-08 was ~35K chars, so this never engaged for sentiment --
# that is NOT true for `ner`: a real sampled article body reached 156,053
# chars (docs/evaluation.md's 2026-09-12 follow-up), over 1.5x this ceiling,
# so this now does engage for `ner` against real data, truncating that tail
# rather than failing the request outright -- the intended
# acceptable-degradation behavior, not a bug. If judge failures ever
# correlate with body length near this ceiling for either stage, that's the
# signal to replace it with an actual tokenizer-based budget instead of a
# char-count proxy.
_SENTIMENT_MAX_BODY_CHARS = 100_000
# Cap the entity list shown to the NER judge -- some articles have 100s.
_MAX_NER_ENTITIES = 60
# For c_summary: how wide a low-confidence net to cast over the upstream stages.
_CSUMMARY_WEAK_MULT = 4

# Sentiment/category target-stratum thresholds. Deliberately BELOW the
# argmax-guarantee boundary (0.5 for sentiment's 3-way softmax; the winning
# bar for category is CATEGORY_CONFIDENCE_THRESHOLD=0.6) -- a threshold at or
# above that boundary would mathematically exclude every false-negative
# candidate for that class (only one class's raw score can clear 0.5 out of a
# distribution summing to 1), which would silently defeat the entire purpose
# of soft-probability stratification: catching rows the model almost called
# this class but didn't, not just rows it did call this class. If either
# threshold is ever retuned, it MUST stay below that boundary -- see
# test_eval_sampling.py's threshold-boundary tests.
_SENTIMENT_TARGET_THRESHOLD = 0.35  # just above the 3-way uniform baseline (0.333)
# Was 0.2 (~1.8x the flat classifier's 9-way uniform baseline of 0.111) before
# the hierarchical category classifier (pipeline.run_category_stage,
# docs/category-taxonomy.md) replaced that flat 9-way softmax with two levels
# of 3-way-or-narrower softmaxes. article_category's 9 leaf-slug columns are
# now populated from 3-way child-group softmaxes for whichever slugs' group
# made an article's top-2 -- baseline there is 0.333, so the old 0.2 sat
# BELOW no-signal, not above it: target_<slug> >= 0.2 would fire on a large,
# uninformative fraction of rows by chance, defeating the near-miss
# selectivity stratification exists for. Raised to sit just above the new
# 0.333 baseline, mirroring _SENTIMENT_TARGET_THRESHOLD's own placement.
_CATEGORY_TARGET_THRESHOLD = 0.35

# Per-class weight of the sentiment target budget. `negative` is weighted
# highest: it's HEADLINE["sentiment"]'s class (metrics.py) -- see
# docs/evaluation.md's "Why recall, not F1, for sentiment negative".
_SENTIMENT_TARGETS: tuple[tuple[str, float], ...] = (
    ("negative", 0.6),
    ("positive", 0.2),
    ("neutral", 0.2),
)

# Per-slug weight of the category target budget: one stratum per the 6 worst
# 2026-09-08-baseline per-slug accuracies (docs/evaluation.md) plus
# legal_regulatory, weights summing to 1.0 (asserted by a test, not at
# runtime). market_analyst_sentiment / earnings_performance / other are
# skipped -- already well covered by population share alone (36%/10%/18%).
_CATEGORY_TARGET_WEIGHTS: dict[str, float] = {
    "partnerships_business_dev": 0.20,
    "labor_human_capital": 0.18,
    "leadership_governance": 0.16,
    "mergers_acquisitions": 0.14,
    "capital_shareholder_returns": 0.13,
    "product_innovation": 0.12,
    "legal_regulatory": 0.07,
}

# c_summary has no discrete classes, so its "targets" partition on
# `num_chunks` instead -- a TRUE partition of the population (every row has
# exactly one num_chunks value), so c_summary draws no separate
# `representative` bucket; these three tiers exhaust it. Weighted toward the
# rare multi-chunk tail (~4.6% of the corpus) since mean_coverage was already
# the weakest c_summary metric (3.02/5) and multi-chunk articles pass through
# more hierarchical-reduce steps.
_CSUMMARY_CHUNK_TIERS: tuple[tuple[str, str, float], ...] = (
    ("target_chunks_1", "num_chunks = 1", 0.10),
    ("target_chunks_2", "num_chunks = 2", 0.30),
    ("target_chunks_ge3", "num_chunks >= 3", 0.60),
)

# Assembled (bucket_name, table, condition_sql, params, weight) per stage.
# condition_sql/table only ever come from the constants above -- never user
# input. ner maps to () -- see the module docstring for why.
_TARGETS_BY_STAGE: dict[str, tuple[tuple[str, str, str, tuple[Any, ...], float], ...]] = {
    "sentiment": tuple(
        (
            f"target_{cls}",
            "article_sentiment",
            f"{cls} >= ?",
            (_SENTIMENT_TARGET_THRESHOLD,),
            weight,
        )
        for cls, weight in _SENTIMENT_TARGETS
    ),
    "category": tuple(
        (
            f"target_{slug}",
            "article_category",
            f"{slug} >= ?",
            (_CATEGORY_TARGET_THRESHOLD,),
            weight,
        )
        for slug, weight in _CATEGORY_TARGET_WEIGHTS.items()
    ),
    "ner": (),
    "c_summary": tuple(
        (name, "article_summary", condition, (), weight)
        for name, condition, weight in _CSUMMARY_CHUNK_TIERS
    ),
}


@dataclass(frozen=True)
class EvalItem:
    """One row to judge: the source text plus the model's stored prediction."""

    article_id: int
    bucket: str  # "low_conf" | "representative" | stage-specific "target_<x>"
    stratum_population: (
        int  # N_h: population size of this item's bucket (news_nlp.eval.metrics' HT reweighting)
    )
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


def _candidate_ids(
    conn: NewsNlpDatabase,
    table: str,
    condition_sql: str,
    params: tuple[Any, ...],
    exclude: set[int],
) -> list[int]:
    """Every ``article_id`` in *table* matching *condition_sql*, minus
    *exclude*, ``ORDER BY article_id`` -- same shuffle-determinism contract as
    ``_all_ids``. *condition_sql*/*table* only ever come from the
    ``_TARGETS_BY_STAGE`` constants above, never user input."""
    rows = conn.execute(
        f"SELECT article_id FROM {table} WHERE {condition_sql} ORDER BY article_id",  # noqa: S608
        params,
    ).fetchall()
    return [int(r[0]) for r in rows if int(r[0]) not in exclude]


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
    # S608: `schema` is always the return value of `_require_source` ->
    # `_articles_rel(conn)`, allowlist-checked to be only "main" / "source"
    # (see news_nlp/db.py's `_articles_rel` docstring) -- never caller or user
    # input, despite the f-string. Same false-positive class already
    # documented on `_all_ids` above.
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
    target_frac: float = 0.6,
    seed: int | None = None,
) -> list[EvalItem]:
    """Return up to *size* ``EvalItem``s for *stage*, drawn as the
    disjoint, priority-ordered ``low_conf`` -> ``target_<x>`` -> ``representative``
    stack described in the module docstring. ``target_frac`` is the share of
    the post-``low_conf`` budget spent on targeted strata (0 for ``ner``,
    which has none, regardless of this value); the rest fills
    ``representative``. Fewer than *size* only when the store holds too few
    judged rows.

    Backward-compatibility property, relied on by
    ``test_target_frac_zero_matches_legacy_two_bucket_ordering``: when a stage
    has no target strata (or *target_frac* is 0), the rng call sequence -- and
    therefore the exact drawn ids for a given seed -- is bit-for-bit identical
    to the pre-stratification two-bucket code; ``"representative"`` is just
    ``"random"`` renamed.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    schema = _require_source(conn)
    rng = random.Random(seed)  # noqa: S311 -- sample selection, not cryptography

    all_ids = _all_ids(conn, stage)
    total_population = len(all_ids)

    n_low = round(size * low_conf_frac)
    chosen_low = _low_conf_ids(conn, stage, n_low)
    claimed: set[int] = set(chosen_low)

    strata: list[tuple[str, list[int], int]] = [("low_conf", chosen_low, total_population)]

    targets = _TARGETS_BY_STAGE[stage]
    remaining_budget = max(size - len(chosen_low), 0)
    target_budget = round(remaining_budget * target_frac) if targets else 0

    if target_budget > 0:
        # Guarded on target_budget rather than just iterating `targets`
        # unconditionally: every target's n_h is min(round(target_budget *
        # weight), population), which is always 0 when target_budget is 0
        # regardless of population -- so skipping the query+shuffle entirely
        # here (rather than doing the work and drawing nothing) is what makes
        # target_frac=0 reproduce the exact legacy rng call sequence (see
        # test_target_frac_zero_matches_legacy_two_bucket_ordering). A
        # `rng.shuffle(candidates)` on a non-empty population would otherwise
        # silently consume rng state even when nothing gets drawn from it.
        for bucket_name, table, condition_sql, params, weight in targets:
            candidates = _candidate_ids(conn, table, condition_sql, params, claimed)
            population = len(candidates)  # N_h -- same list drawn from, never a separate COUNT(*)
            rng.shuffle(candidates)
            n_h = min(round(target_budget * weight), population)
            drawn = candidates[:n_h]
            claimed.update(drawn)
            strata.append((bucket_name, drawn, population))

    rest = [i for i in all_ids if i not in claimed]
    representative_population = len(rest)
    rng.shuffle(rest)
    n_rep = max(size - sum(len(ids) for _, ids, _ in strata), 0)
    strata.append(("representative", rest[:n_rep], representative_population))

    items: list[EvalItem] = []
    for bucket, ids, population in strata:
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
                    stratum_population=population,
                    title=title,
                    body_text=body,
                    prediction=pred,
                )
            )
    return items
