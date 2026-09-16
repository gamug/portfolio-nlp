"""Category stage `Feature`/`Inference` (PLAN.md Work item 10 / TASKS.md
T-085), migrating `pipeline.py`'s two-level hierarchical zero-shot
classification onto the FTI base classes from `src/fti.py`.

`Trainer` is a plain reuse of `fti.NoOpTrainer`, not a subclass here --
category is zero-shot NLI against a fixed taxonomy (docs/category-
taxonomy.md); no labeled category training set exists, and one would be
expensive to build for a 10-slug taxonomy that may itself change (PLAN.md's
category rationale). There is no fine-tuning step to wrap.

Batching here doesn't need cross-article flattening at the `Feature` level
(unlike NER) -- `CategoryFeature` extracts one premise string per article,
reusing `Feature`'s own default `extract_batch` loop (same shape as
`sentiment_stage.SentimentFeature`). The real cross-article batching
happens one level down, inside `_category_level1_batch`/
`_category_level2_batch`'s (premise, hypothesis)-pair tokenizer calls,
which need the loaded model and so live in `CategoryInference.predict_batch`.

`AutoTokenizer`/`AutoModelForSequenceClassification` are imported here at
module scope (not received via `pipeline.py`) per `fti.Inference`'s own
`load_model` contract -- `pipeline.py` still imports `AutoTokenizer` for
`run_company_summary_stage`'s own use, so existing hermetic tests'
`monkeypatch.setattr(pipeline.AutoTokenizer, ...)` calls remain valid
(monkeypatching a class object mutates it globally, wherever the patched
class is later called from); `AutoModelForSequenceClassification` is
category-only in `pipeline.py` as of this migration (sentiment's own use
already moved to `sentiment_stage.py` in T-083), so tests patching it --
including `tests/news_nlp/test_sentiment_pipeline.py`'s own leftover
`pipeline.AutoModelForSequenceClassification` patch, which only worked
because `pipeline.py` still imported the class for category's sake --
must target `category_stage.AutoModelForSequenceClassification` instead.
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import news_nlp as db
from chunking import chunk_text
from fti import Feature, FeatureBatch, Inference
from news_nlp.taxonomy import (
    CATEGORY_CONFIDENCE_THRESHOLD,
    CATEGORY_GROUP_CHILDREN,
    CATEGORY_GROUP_FLOOR,
    CATEGORY_GROUPS,
    CATEGORY_LABELS_BY_SLUG,
    CATEGORY_SLUGS,
    OTHER_LABEL,
)

# Tighter than the 510 used by sentiment/NER's single-sequence chunking:
# this stage tokenizes (premise, hypothesis) *pairs*, so the premise needs
# to leave headroom for the hypothesis text plus special tokens within the
# model's 512-token cap.
CATEGORY_PREMISE_MAX_TOKENS = 460


def classify_group_scores(entail_logits: list[float]) -> tuple[str, float, dict[str, float]]:
    """Level 1 of the hierarchical category classifier: turn 3 entailment
    logits (one per CATEGORY_GROUPS group, same order) into (winning_group,
    winning_group_score, {group_slug: prob}). Softmax over just the 3 group
    logits -- mirrors classify_category_scores' shape one level up. Does NOT
    decide OTHER_LABEL here: that's a leaf-level concept in article_category,
    decided by `CategoryInference.predict_batch` comparing winning_group_score
    against CATEGORY_GROUP_FLOOR (see docs/category-taxonomy.md)."""

    probs = torch.softmax(torch.tensor(entail_logits), dim=0).tolist()
    scores = {slug: p for (slug, _, _, _), p in zip(CATEGORY_GROUPS, probs, strict=False)}
    winner_slug = max(scores, key=scores.__getitem__)
    return winner_slug, scores[winner_slug], scores


def top2_groups(group_scores: dict[str, float]) -> list[str]:
    """Highest-to-lowest top-2 group slugs from a classify_group_scores()
    result. Split out so `CategoryInference.predict_batch`'s level-2
    routing is unit-testable without re-deriving the softmax math."""
    return sorted(group_scores, key=group_scores.__getitem__, reverse=True)[:2]


def classify_category_scores(
    entail_logits: list[float], candidate_slugs: Sequence[str]
) -> tuple[str, float, dict[str, float]]:
    """Level 2 of the hierarchical category classifier: turn
    len(candidate_slugs) entailment logits (one per candidate_slugs entry,
    same order -- always an article's top-2 groups' 6 combined children,
    see `CategoryInference.predict_batch`) into (label, winning_score,
    {slug: prob}), softmax-normalized over just `candidate_slugs` (not all
    9 -- only the slugs actually scored this pass need to sum to 1). Split
    out so the classification math is testable without a real model, same
    spirit as merge_bio_predictions being split out of NER's forward pass.

    `winning_score` always reflects the best-scoring slug's probability,
    even when the returned label is OTHER_LABEL -- that's what makes
    low-confidence "other" picks auditable (label='other' with a score just
    under the threshold is a near-miss; a low score alongside a flat
    distribution is not).

    Slugs from CATEGORY_SLUGS absent from `candidate_slugs` (the group that
    didn't make the article's top-2) are NOT in the returned dict -- callers
    must zero-fill them before db.write_category.
    """

    probs = torch.softmax(torch.tensor(entail_logits), dim=0).tolist()
    scores = dict(zip(candidate_slugs, probs, strict=True))
    winner_slug = max(scores, key=scores.__getitem__)
    winner_score = scores[winner_slug]
    label = winner_slug if winner_score >= CATEGORY_CONFIDENCE_THRESHOLD else OTHER_LABEL
    return label, winner_score, scores


def _category_level1_batch(
    tokenizer: Any, model: Any, device: Any, entailment_id: int, premises: list[str]
) -> list[tuple[str, float, dict[str, float]]]:
    """One level-1 forward pass over every premise in this batch: 3
    (premise, group_hypothesis) pairs each. Returns one
    classify_group_scores() result per premise, same order."""

    group_hypotheses = [f"This example is about {phrase}." for _, _, phrase, _ in CATEGORY_GROUPS]
    n_groups = len(group_hypotheses)
    inputs = tokenizer(
        [p for p in premises for _ in range(n_groups)],
        group_hypotheses * len(premises),
        return_tensors="pt",
        truncation="only_first",
        padding=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # reshape, not view: the entailment column is a strided slice of
    # `logits`, not contiguous, and view() requires contiguity.
    group_entail = logits[:, entailment_id].reshape(len(premises), n_groups)
    return [classify_group_scores(logits_row.tolist()) for logits_row in group_entail]


def _category_level2_batch(
    tokenizer: Any,
    model: Any,
    device: Any,
    entailment_id: int,
    pending: list[tuple[int, str, list[str], str, float]],
) -> list[tuple[str, float, dict[str, float], list[str]]]:
    """One level-2 forward pass over every pending (survived level 1)
    article's top-2 groups' 6 combined children, in CATEGORY_GROUPS' fixed
    child order (not re-sorted) so alignment stays deterministic. Returns
    (label, score, scores, candidate_slugs) per pending entry, same order."""

    l2_premises: list[str] = []
    l2_hypotheses: list[str] = []
    per_article_candidates: list[list[str]] = []
    for _idx, premise, top2, _group_label, _group_score in pending:
        candidates = [slug for g in top2 for slug in CATEGORY_GROUP_CHILDREN[g]]
        per_article_candidates.append(candidates)
        for slug in candidates:
            _, _, phrase = CATEGORY_LABELS_BY_SLUG[slug]
            l2_premises.append(premise)
            l2_hypotheses.append(f"This example is about {phrase}.")

    inputs = tokenizer(
        l2_premises,
        l2_hypotheses,
        return_tensors="pt",
        truncation="only_first",
        padding=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # 1-D: len == sum of per-article candidate counts (always 6 each here,
    # but sliced by a running offset below rather than assumed, so this
    # stays correct if a future group ever has a different child count).
    leaf_entail_flat = logits[:, entailment_id]

    results = []
    offset = 0
    for candidates in per_article_candidates:
        n = len(candidates)
        article_leaf_logits = leaf_entail_flat[offset : offset + n].tolist()
        offset += n
        label, score, scores = classify_category_scores(article_leaf_logits, candidates)
        results.append((label, score, scores, candidates))
    return results


class CategoryFeature(Feature[Any, str]):
    """`row` is `(article_id, title, body_text)`, matching
    `db.fetch_pending_category_articles`'s return shape. One premise
    string per article (not cross-article flattening -- see module
    docstring), so the base `Feature.extract_batch`'s default per-row loop
    is reused as-is."""

    def extract_one(self, tokenizer: Any, row: Any) -> str:
        # Title + lead chunk of body, not full-article chunking: each
        # hypothesis needs its own (premise, hypothesis) forward pass, so
        # chunking the whole article the way sentiment/NER do would cost
        # far more per chunk -- unaffordable for a stage that now runs on
        # every article. News is inverted-pyramid, so the opening sentences
        # almost always establish the dominant topic.
        _article_id, title, body_text = row
        chunks = chunk_text(
            f"{title}. {body_text}", tokenizer, max_tokens=CATEGORY_PREMISE_MAX_TOKENS
        )
        return chunks[0].text if chunks else title


class CategoryInference(Inference[Any, str]):
    """Two-level hierarchical zero-shot classification (docs/category-
    taxonomy.md; migrated onto the FTI hierarchy 2026-09-16, PLAN.md Work
    item 10 / TASKS.md T-085): level 1 picks (up to) the top-2
    CATEGORY_GROUPS for each article via a 3-way softmax; level 2
    classifies only the survivors -- articles whose winning group cleared
    CATEGORY_GROUP_FLOOR -- against those top-2 groups' 6 combined
    children. Replaces a single flat 9-way softmax, which empirically
    starved real signal for several labels by making them compete against
    8 others in one softmax (see docs/category-taxonomy.md's "Hierarchical
    classification" section for the eval data that motivated this).

    `MODEL_NAME`/`MODEL_REVISIONS` below duplicate `pipeline.py`'s own
    `CATEGORY_MODEL`/`MODEL_REVISIONS[CATEGORY_MODEL]` as static defaults
    -- `pipeline.run_category_stage`'s thin wrapper always passes
    `model_name=`/`revision=`/`batch_size=` explicitly (reading its own
    module globals fresh at call time) -- see
    `sentiment_stage.SentimentInference`'s docstring for the same
    disclosed, acceptable duplication.
    """

    MODEL_NAME: ClassVar[str] = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    MODEL_REVISIONS: ClassVar[dict[str, str]] = {
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8",
    }
    STAGE_NAME: ClassVar[str] = "category"

    def batch_size(self) -> int:
        """The constructor's `batch_size` override, or `8` (matching
        today's `CATEGORY_BATCH_SIZE`) if none was given -- the real call
        path (`pipeline.run_category_stage`) always passes one explicitly,
        so this fallback only matters for a hypothetical direct
        construction outside that path."""
        return self._batch_size_override if self._batch_size_override is not None else 8

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.revision
            )
            .to(self.device)
            .eval()
        )
        self.entailment_id = next(
            v for k, v in self.model.config.label2id.items() if k.lower() == "entailment"
        )

    def fetch_pending(
        self, conn: db.NewsNlpDatabase, limit: int | None, sample_seed: int | None
    ) -> list[Any]:
        return db.fetch_pending_category_articles(conn, limit=limit)

    def predict_batch(self, features: FeatureBatch[str]) -> list[Any]:
        """Returns one prediction dict per row in this batch, same order,
        each carrying `outcome` ("other" or "classified"), `label`,
        `score`, `group_label`, `group_score`, `scores` (already
        zero-filled/merged across all CATEGORY_SLUGS) -- see
        `write_predictions` for how the two outcomes are written."""
        premises = features.items
        level1 = _category_level1_batch(
            self.tokenizer, self.model, self.device, self.entailment_id, premises
        )

        predictions: list[dict[str, Any]] = [{}] * len(premises)
        pending: list[tuple[int, str, list[str], str, float]] = []
        for i, (premise, (group_label, group_score, group_scores)) in enumerate(
            zip(premises, level1, strict=True)
        ):
            if group_score < CATEGORY_GROUP_FLOOR:
                # Flat level-1: skip level 2's forward pass entirely for
                # this article, zero-filling every leaf column ("not
                # evaluated", not "confidently rejected" -- see
                # docs/category-taxonomy.md).
                predictions[i] = {
                    "outcome": "other",
                    "label": OTHER_LABEL,
                    "score": group_score,
                    "group_label": group_label,
                    "group_score": group_score,
                    "scores": dict.fromkeys(CATEGORY_SLUGS, 0.0),
                }
                continue
            pending.append((i, premise, top2_groups(group_scores), group_label, group_score))

        if pending:
            level2 = _category_level2_batch(
                self.tokenizer, self.model, self.device, self.entailment_id, pending
            )
            for (i, _premise, _top2, group_label, group_score), (label, score, scores, _c) in zip(
                pending, level2, strict=True
            ):
                # 3rd-place-group zero-placeholder: the group that didn't
                # make this article's top-2 never got a level-2 forward
                # pass, so its children stay at the documented 0.0 "not
                # evaluated" placeholder.
                full_scores = {slug: scores.get(slug, 0.0) for slug in CATEGORY_SLUGS}
                predictions[i] = {
                    "outcome": "classified",
                    "label": label,
                    "score": score,
                    "group_label": group_label,
                    "group_score": group_score,
                    "scores": full_scores,
                }

        return predictions

    def write_predictions(
        self, conn: db.NewsNlpDatabase, rows: Sequence[Any], predictions: list[Any]
    ) -> None:
        """Two passes, not one, to preserve the original code's exact
        two-commit-per-batch behavior ("flush this batch's short-circuited
        rows now"): write every "other"-outcome row, commit, then write
        every "classified"-outcome row -- `run()`'s own trailing
        `conn.commit()` after this method returns provides the second
        commit."""
        for (article_id, _title, _body_text), pred in zip(rows, predictions, strict=True):
            if pred["outcome"] == "other":
                db.write_category(
                    conn,
                    article_id,
                    label=pred["label"],
                    score=pred["score"],
                    group_label=pred["group_label"],
                    group_score=pred["group_score"],
                    scores=pred["scores"],
                    model_name=self.model_name,
                )
        conn.commit()

        for (article_id, _title, _body_text), pred in zip(rows, predictions, strict=True):
            if pred["outcome"] == "classified":
                db.write_category(
                    conn,
                    article_id,
                    label=pred["label"],
                    score=pred["score"],
                    group_label=pred["group_label"],
                    group_score=pred["group_score"],
                    scores=pred["scores"],
                    model_name=self.model_name,
                )
