"""The algorithms of the v1 modules that the v2 steps still run (T-220 PR B): `ner_stage.merge_bio_predictions` (step 4), `summary_stage.hierarchical_summarize_batch`
(step 2, the local summariser), and the matcher `sentiment_subject._text_mentions_subject` (the sentiment training scripts).

Kept verbatim from the v1 tests (`test_ner_pipeline.py`, `test_summary_pipeline.py`, `test_sentiment_pipeline.py` at commit `81294eb`) that targeted these functions directly,
so deleting the v1 pipeline tests does not leave code the revisions hash, and the steps run, with no test that executes it. The model is a fake; no network, no GPU.

NER regression coverage for the 2026-09-10 fix: the model was trained with continuation WordPiece subwords masked out of the loss, but inference used to trust every
subword's own raw BIO argmax, so "3M" tokenized as ["3", "##M"] could emit a bogus standalone "3"/ORG entity. `merge_bio_predictions` is word-id aware: only a word's
first subword ever opens, closes or redirects a span."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import ner_stage
import summary_stage
from sentiment_subject import _text_mentions_subject

_ID2LABEL = {0: "O", 1: "B-ORG", 2: "I-ORG", 3: "B-PER", 4: "I-PER"}


def _probs_row(n_labels: int, winning_id: int, winning_p: float = 0.9) -> list[float]:
    row = [0.0] * n_labels
    row[winning_id] = winning_p
    return row


def test_continuation_subword_with_flipped_label_still_extends_entity() -> None:
    """The core "3"+"##M" regression, isolated: first subword predicts
    B-ORG, its continuation subword's own (untrained-for) argmax flips to O
    -- the entity must still span both tokens, not just the first."""
    pred_ids = [1, 0]  # B-ORG, then a noisy "O" on the continuation subword
    word_ids: list[int | None] = [0, 0]
    offsets = [(0, 1), (1, 2)]
    probs = [_probs_row(5, 1), _probs_row(5, 0)]

    entities = ner_stage.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [{"entity_type": "ORG", "start_char": 0, "end_char": 2, "scores": [0.9]}]


def test_new_b_tag_on_different_word_produces_two_entities() -> None:
    """A same-type B- tag on a genuinely different (non-continuation) word
    must NOT be merged into the prior entity -- proves the fix doesn't
    over-merge distinct words of the same type."""
    pred_ids = [1, 1]  # B-ORG, B-ORG -- two separate words, not a continuation
    word_ids: list[int | None] = [0, 1]
    offsets = [(0, 1), (2, 3)]
    probs = [_probs_row(5, 1), _probs_row(5, 1)]

    entities = ner_stage.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

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

    entities = ner_stage.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

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

    entities = ner_stage.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [{"entity_type": "ORG", "start_char": 0, "end_char": 2, "scores": [0.9]}]


def test_continuation_subword_scores_excluded_from_average() -> None:
    """Regression-lock the explicit design decision: a continuation
    subword's own probability never enters the entity's averaged score,
    since it was never calibrated against any target at that position."""
    pred_ids = [1, 2]  # B-ORG (real decision), I-ORG (noisy continuation)
    word_ids: list[int | None] = [0, 0]
    offsets = [(0, 1), (1, 2)]
    probs = [_probs_row(5, 1, winning_p=0.7), _probs_row(5, 2, winning_p=0.99)]

    entities = ner_stage.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities[0]["scores"] == [0.7]  # not [0.7, 0.99]


def test_special_tokens_are_skipped_via_word_id_none() -> None:
    """CLS/SEP (word_id None) never open/close a span themselves, and reset
    the continuation tracker so a real word right after one isn't mistaken
    for a continuation of whatever came before it."""
    pred_ids = [0, 1, 0]  # [CLS]-ish, B-ORG, [SEP]-ish
    word_ids: list[int | None] = [None, 0, None]
    offsets = [(0, 0), (0, 1), (0, 0)]
    probs = [_probs_row(5, p) for p in pred_ids]

    entities = ner_stage.merge_bio_predictions(pred_ids, word_ids, offsets, probs, _ID2LABEL)

    assert entities == [{"entity_type": "ORG", "start_char": 0, "end_char": 1, "scores": [0.9]}]


_FAKE_DEVICE = torch.device("cpu")


class WordCountTokenizer:
    """Minimal stand-in for a real tokenizer: chunk_text only needs
    .encode(text, add_special_tokens=False) to count tokens, and none of the
    text used in these tests has a single "sentence" long enough to trigger
    chunk_text's hard-split fallback (which needs return_offsets_mapping)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


def make_recording_summarizer(
    monkeypatch: pytest.MonkeyPatch, replies: list[str] | None = None
) -> list[list[str]]:
    """Monkeypatch pipeline._summarize_batch to avoid loading a real model,
    and record every call's input batch. `replies` (if given) is consumed
    one-per-text across all calls, in order; otherwise every text gets a
    fixed placeholder ("X")."""
    calls: list[list[str]] = []
    replies_iter = iter(replies) if replies is not None else None

    def fake(texts: list[str], tokenizer: Any, model: Any, device: torch.device) -> list[str]:
        calls.append(list(texts))
        if replies_iter is not None:
            return [next(replies_iter) for _ in texts]
        return ["X" for _ in texts]

    monkeypatch.setattr(summary_stage, "_summarize_batch", fake)
    return calls


def test_hierarchical_summarize_batch_single_chunk_is_a_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = make_recording_summarizer(monkeypatch, replies=["Short summary."])

    results = summary_stage.hierarchical_summarize_batch(
        ["One short sentence."],
        WordCountTokenizer(),
        model=None,
        device=_FAKE_DEVICE,
        max_input_tokens=100,
    )

    assert results == [("Short summary.", 1)]
    assert len(calls) == 1  # one batch call


def test_hierarchical_summarize_batch_empty_text_returns_no_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = make_recording_summarizer(monkeypatch)

    results = summary_stage.hierarchical_summarize_batch(
        [""], WordCountTokenizer(), model=None, device=_FAKE_DEVICE, max_input_tokens=100
    )

    assert results == [("", 0)]
    assert calls == []


def test_hierarchical_summarize_batch_empty_input_list_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = make_recording_summarizer(monkeypatch)

    results = summary_stage.hierarchical_summarize_batch(
        [], WordCountTokenizer(), model=None, device=_FAKE_DEVICE, max_input_tokens=100
    )

    assert results == []
    assert calls == []


def test_hierarchical_summarize_batch_multi_chunk_triggers_a_reduce_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each 3-word sentence is exactly one chunk at max_input_tokens=3 --
    # 3 sentences -> 3 leaf chunks -> 3 first-pass summaries -> those get
    # joined ("X X X") and summarized once more since len(summaries) > 1.
    calls = make_recording_summarizer(monkeypatch)
    text = "AAA BBB CCC. DDD EEE FFF. GGG HHH III."

    results = summary_stage.hierarchical_summarize_batch(
        [text], WordCountTokenizer(), model=None, device=_FAKE_DEVICE, max_input_tokens=3
    )

    summary, num_chunks = results[0]
    assert num_chunks == 3  # leaf-level chunk count, not the reduced count
    assert summary == "X"  # the reduce pass's own (stubbed) output
    assert len(calls) == 2  # one batch call for the 3 leaf chunks, one for the reduce pass
    assert len(calls[0]) == 3
    assert calls[1] == ["X X X"]  # the reduce pass's input was the joined leaf summaries


def test_hierarchical_summarize_batch_reduce_pass_summarizes_the_joined_chunk_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = make_recording_summarizer(monkeypatch, replies=["S1", "S2", "S3", "final"])
    text = "AAA BBB CCC. DDD EEE FFF. GGG HHH III."

    results = summary_stage.hierarchical_summarize_batch(
        [text], WordCountTokenizer(), model=None, device=_FAKE_DEVICE, max_input_tokens=3
    )

    summary, num_chunks = results[0]
    assert num_chunks == 3
    assert calls[-1] == ["S1 S2 S3"]  # the reduce pass's input was the joined leaf summaries
    assert summary == "final"


def test_hierarchical_summarize_batch_pools_multiple_texts_into_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = make_recording_summarizer(monkeypatch, replies=["Summary A.", "Summary B."])

    results = summary_stage.hierarchical_summarize_batch(
        ["Text A.", "Text B."],
        WordCountTokenizer(),
        model=None,
        device=_FAKE_DEVICE,
        max_input_tokens=100,
        batch_size=4,
    )

    assert results == [("Summary A.", 1), ("Summary B.", 1)]
    assert len(calls) == 1  # both texts' single leaf chunk pooled into one generate() call
    assert calls[0] == ["Text A.", "Text B."]


def test_hierarchical_summarize_batch_respects_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = make_recording_summarizer(monkeypatch)

    summary_stage.hierarchical_summarize_batch(
        ["Text A.", "Text B.", "Text C."],
        WordCountTokenizer(),
        model=None,
        device=_FAKE_DEVICE,
        max_input_tokens=100,
        batch_size=2,
    )

    assert [len(c) for c in calls] == [2, 1]  # 3 texts split into batches of at most 2


# --- the matcher the sentiment scripts use ---------------------------------------------------------------------------------------------


def test_ticker_mention_is_word_boundary_matched() -> None:
    assert _text_mentions_subject("AAPL rose today", None, "AAPL")
    assert not _text_mentions_subject("The AAPLX fund rose", None, "AAPL")


def test_company_name_matches_despite_corporate_suffix_mismatch() -> None:
    assert _text_mentions_subject("Acme announced results", "Acme Corp.", None)
    assert _text_mentions_subject("Acme Corporation announced results", "Acme Inc", None)


def test_different_company_does_not_match() -> None:
    assert not _text_mentions_subject("Globex announced results", "Acme Corp.", "ACME")


def test_no_company_or_ticker_never_matches() -> None:
    assert not _text_mentions_subject("anything at all", None, None)


# --- the two functions no other test ran ------------------------------------------------------------------------------------------------


def test_merge_char_spans_drops_only_exact_duplicates_and_keeps_the_order() -> None:
    from chunking import merge_char_spans  # noqa: PLC0415 -- kept next to its test

    a = {"entity_type": "ORG", "start_char": 0, "end_char": 4, "text": "Acme"}
    dup = {"entity_type": "ORG", "start_char": 0, "end_char": 4, "text": "Acme"}
    other_type = {"entity_type": "PER", "start_char": 0, "end_char": 4, "text": "Acme"}
    shifted = {"entity_type": "ORG", "start_char": 1, "end_char": 4, "text": "cme"}
    assert merge_char_spans([a, dup, other_type, shifted]) == [a, other_type, shifted]
    assert merge_char_spans([]) == []


def test_summarize_batch_generates_once_for_the_batch_and_strips_the_text() -> None:
    class Inputs(dict):  # type: ignore[type-arg]
        def to(self, _device: object) -> Inputs:
            return self

    class Tokenizer:
        def __call__(self, texts: list[str], **kwargs: Any) -> Inputs:
            assert kwargs["truncation"] and kwargs["padding"] and kwargs["max_length"] == 1024
            return Inputs(input_ids=torch.zeros((len(texts), 3), dtype=torch.long))

        def batch_decode(self, ids: torch.Tensor, skip_special_tokens: bool) -> list[str]:
            assert skip_special_tokens
            return [f"  summary {i}  " for i in range(ids.shape[0])]

    class Model:
        calls = 0

        def generate(self, **kwargs: Any) -> torch.Tensor:
            Model.calls += 1
            assert kwargs["num_beams"] == 4
            ids: torch.Tensor = kwargs["input_ids"]
            return ids

    out = summary_stage._summarize_batch(["a", "b"], Tokenizer(), Model(), torch.device("cpu"))
    assert out == ["summary 0", "summary 1"]
    assert Model.calls == 1
