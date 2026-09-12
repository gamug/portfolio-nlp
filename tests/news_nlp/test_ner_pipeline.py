"""NER stage: word-boundary-aware BIO span merging.

Regression coverage for the 2026-09-10 fix (docs/evaluation.md's "NER
subword-fragmentation" follow-up): the model was trained with continuation
WordPiece subwords masked out of the loss (train_ner.py's
make_tokenize_fn), but inference used to trust every subword's own raw BIO
argmax independently -- so "3M" tokenized as ["3", "##M"] could emit a bogus
standalone "3"/ORG entity whenever the untrained continuation prediction on
"##M" wasn't I-ORG. `merge_bio_predictions` is now word-id aware: only a
word's first subword ever opens/closes/redirects a span.
"""

import sqlite3
from typing import Any

import pytest
import torch
from conftest import seed_article

import news_nlp as db
import pipeline

# A small, realistic BIO label set (mirrors train_ner.py's PER/LOC/ORG scheme).
_ID2LABEL = {0: "O", 1: "B-ORG", 2: "I-ORG", 3: "B-PER", 4: "I-PER"}


def _probs_row(n_labels: int, winning_id: int, winning_p: float = 0.9) -> list[float]:
    row = [0.0] * n_labels
    row[winning_id] = winning_p
    return row


# --- merge_bio_predictions (pure function, no model needed) -----------------


def test_continuation_subword_with_flipped_label_still_extends_entity() -> None:
    """The core "3"+"##M" regression, isolated: first subword predicts
    B-ORG, its continuation subword's own (untrained-for) argmax flips to O
    -- the entity must still span both tokens, not just the first."""
    pred_ids = [1, 0]  # B-ORG, then a noisy "O" on the continuation subword
    word_ids: list[int | None] = [0, 0]
    offsets = [(0, 1), (1, 2)]
    probs = [_probs_row(5, 1), _probs_row(5, 0)]

    entities = pipeline.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [{"entity_type": "ORG", "start_char": 0, "end_char": 2, "scores": [0.9]}]


def test_new_b_tag_on_different_word_produces_two_entities() -> None:
    """A same-type B- tag on a genuinely different (non-continuation) word
    must NOT be merged into the prior entity -- proves the fix doesn't
    over-merge distinct words of the same type."""
    pred_ids = [1, 1]  # B-ORG, B-ORG -- two separate words, not a continuation
    word_ids: list[int | None] = [0, 1]
    offsets = [(0, 1), (2, 3)]
    probs = [_probs_row(5, 1), _probs_row(5, 1)]

    entities = pipeline.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [
        {"entity_type": "ORG", "start_char": 0, "end_char": 1, "scores": [0.9]},
        {"entity_type": "ORG", "start_char": 2, "end_char": 3, "scores": [0.9]},
    ]


def test_multiword_entity_merges_across_word_boundaries() -> None:
    """A real 2-word entity (B-ORG word, I-ORG word) merges correctly even
    when each word's own continuation subwords carry noisy, irrelevant
    labels -- those must be ignored entirely."""
    pred_ids = [1, 3, 2, 0]  # B-ORG, (noisy B-PER continuation), I-ORG, (noisy O continuation)
    word_ids: list[int | None] = [0, 0, 1, 1]
    offsets = [(0, 1), (1, 2), (3, 4), (4, 5)]
    probs = [_probs_row(5, p) for p in pred_ids]

    entities = pipeline.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [
        {"entity_type": "ORG", "start_char": 0, "end_char": 5, "scores": [0.9, 0.9]}
    ]


def test_o_after_continuation_closes_entity_correctly() -> None:
    """Sanity case: an entity word followed by a plain O word still closes
    correctly post-refactor."""
    pred_ids = [1, 0, 0]  # B-ORG, (noisy continuation), O
    word_ids: list[int | None] = [0, 0, 1]
    offsets = [(0, 1), (1, 2), (3, 4)]
    probs = [_probs_row(5, p) for p in pred_ids]

    entities = pipeline.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [{"entity_type": "ORG", "start_char": 0, "end_char": 2, "scores": [0.9]}]


def test_continuation_subword_scores_excluded_from_average() -> None:
    """Regression-lock the explicit design decision: a continuation
    subword's own probability never enters the entity's averaged score,
    since it was never calibrated against any target at that position."""
    pred_ids = [1, 2]  # B-ORG (real decision), I-ORG (noisy continuation)
    word_ids: list[int | None] = [0, 0]
    offsets = [(0, 1), (1, 2)]
    probs = [_probs_row(5, 1, winning_p=0.7), _probs_row(5, 2, winning_p=0.99)]

    entities = pipeline.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities[0]["scores"] == [0.7]  # not [0.7, 0.99]


def test_special_tokens_are_skipped_via_word_id_none() -> None:
    """CLS/SEP (word_id None) never open/close a span themselves, and reset
    the continuation tracker so a real word right after one isn't mistaken
    for a continuation of whatever came before it."""
    pred_ids = [0, 1, 0]  # [CLS]-ish, B-ORG, [SEP]-ish
    word_ids: list[int | None] = [None, 0, None]
    offsets = [(0, 0), (0, 1), (0, 0)]
    probs = [_probs_row(5, p) for p in pred_ids]

    entities = pipeline.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [{"entity_type": "ORG", "start_char": 0, "end_char": 1, "scores": [0.9]}]


# --- run_ner_stage integration: the literal "3M" bug, end to end ------------


class FakeNerEncoding(dict):
    """Mimics the pieces of a fast tokenizer's BatchEncoding run_ner_stage
    actually uses: dict-like input_ids/attention_mask (for .items()/.to()),
    a poppable offset_mapping tensor, and .word_ids()."""

    def __init__(self, offsets: list[tuple[int, int]], word_ids: list[int | None]) -> None:
        seq_len = len(offsets)
        super().__init__(
            {
                "input_ids": torch.zeros((1, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
                "offset_mapping": torch.tensor([offsets]),
            }
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        assert batch_index == 0
        return self._word_ids


class FakeNerTokenizer:
    """Ignores the actual chunk text and always serves the same fixed
    tokenization of "3M Company reported earnings." -- fine since the test
    only ever feeds it that one body."""

    def __init__(self, offsets: list[tuple[int, int]], word_ids: list[int | None]) -> None:
        self._offsets = offsets
        self._word_ids = word_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()  # chunk_text only needs a token *count*

    def __call__(self, text: str, **kwargs: Any) -> FakeNerEncoding:
        return FakeNerEncoding(self._offsets, self._word_ids)


class FakeNerModel:
    def __init__(self, id2label: dict[int, str], pred_ids_by_position: list[int]) -> None:
        self.config = type("Config", (), {"id2label": id2label})()
        self._pred_ids = pred_ids_by_position

    def to(self, device: Any) -> "FakeNerModel":
        return self

    def eval(self) -> "FakeNerModel":
        return self

    def __call__(self, **kwargs: Any) -> Any:
        seq_len = kwargs["input_ids"].shape[1]
        assert seq_len == len(self._pred_ids)
        logits = torch.zeros(1, seq_len, len(self.config.id2label))
        for i, winning_id in enumerate(self._pred_ids):
            logits[0, i, winning_id] = 10.0  # dominates softmax
        return type("Output", (), {"logits": logits})()


def test_run_ner_stage_does_not_split_3m_into_a_bogus_bare_digit_entity(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal confirmed real-data bug: "3M Company reported earnings."
    tokenized as ["3", "##M", "Company", "reported", "earnings", "."], with
    the model predicting B-ORG on "3" but (the untrained-for) O on "##M" --
    the pre-fix merge would emit a standalone "3"/ORG entity; the fix must
    write one whole "3M"/ORG entity instead."""
    body = "3M Company reported earnings."
    seed_article(conn, id=1, title="3M Update", body_text=body)
    conn.commit()

    # token order: [CLS] 3 ##M Company reported earnings . [SEP]
    offsets = [(0, 0), (0, 1), (1, 2), (3, 10), (11, 19), (20, 28), (28, 29), (0, 0)]
    word_ids: list[int | None] = [None, 0, 0, 1, 2, 3, 4, None]
    # id2label: 0=O, 1=B-ORG, 2=I-ORG. "3" -> B-ORG, "##M" -> O (the bug
    # trigger: the untrained continuation prediction disagrees), everything
    # else -> O.
    id2label = {0: "O", 1: "B-ORG", 2: "I-ORG"}
    pred_ids_by_position = [0, 1, 0, 0, 0, 0, 0, 0]

    monkeypatch.setattr(
        pipeline.AutoTokenizer,
        "from_pretrained",
        lambda *_a, **_k: FakeNerTokenizer(offsets, word_ids),
    )
    monkeypatch.setattr(
        pipeline.AutoModelForTokenClassification,
        "from_pretrained",
        lambda *_a, **_k: FakeNerModel(id2label, pred_ids_by_position),
    )

    pipeline.run_ner_stage(conn)

    detail = db.get_article_detail(conn, 1)
    assert detail is not None
    org_entities = [e for e in detail["entities"] if e["entity_type"] == "ORG"]
    assert len(org_entities) == 1
    assert org_entities[0]["text"] == "3M"
    assert org_entities[0]["start_char"] == 0
    assert org_entities[0]["end_char"] == 2
    # Confirms the Step 3 length floor doesn't itself interact with this
    # case -- "3M" (length 2) passes it regardless of the word-id fix.
    assert all(len(e["text"].strip()) >= 2 for e in detail["entities"])


def test_run_ner_stage_passes_sample_seed_through_to_fetch_pending_articles(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_ner_stage(..., sample_seed=...)` must reach
    `db.fetch_pending_articles` unchanged -- a unit-level check of the
    pass-through contract (docs/evaluation.md's 2026-09-12 NER follow-up /
    PLAN.md Work item 3 T-025), not the tokenizer/model path (already
    covered by the test above). Stubbed to return no rows so `run_ner_stage`
    exits before ever needing a tokenizer or model (the `total == 0: return`
    early-out)."""
    calls: list[dict[str, Any]] = []

    def fake_fetch_pending_articles(
        _conn: Any, table: str, limit: int | None = None, *, sample_seed: int | None = None
    ) -> list[Any]:
        calls.append({"table": table, "limit": limit, "sample_seed": sample_seed})
        return []

    monkeypatch.setattr(pipeline.db, "fetch_pending_articles", fake_fetch_pending_articles)

    pipeline.run_ner_stage(conn, limit=20000, sample_seed=1)

    assert calls == [{"table": "article_entities", "limit": 20000, "sample_seed": 1}]
