import sqlite3
from typing import Any

import pytest
import torch
from conftest import seed_article

import news_nlp as db
import pipeline
from news_nlp.taxonomy import CATEGORY_GROUP_SLUGS, CATEGORY_SLUG_TO_GROUP

# --- classify_group_scores / top2_groups / classify_category_scores --------
# (pure functions, no model needed)


def test_classify_group_scores_picks_highest_and_returns_full_distribution() -> None:
    group_label, group_score, group_scores = pipeline.classify_group_scores([5.0, 0.0, 0.0])

    assert group_label == "corporate_actions"
    assert group_score > pipeline.CATEGORY_GROUP_FLOOR
    assert set(group_scores) == set(CATEGORY_GROUP_SLUGS)
    assert abs(sum(group_scores.values()) - 1.0) < 1e-6


def test_classify_group_scores_flat_distribution_is_below_the_floor() -> None:
    # Uniform 3-way logits -> softmax = [1/3, 1/3, 1/3] exactly.
    group_label, group_score, group_scores = pipeline.classify_group_scores([0.0, 0.0, 0.0])

    assert group_label == "corporate_actions"  # first-tied entry, per max()'s dict-order tiebreak
    assert group_score == pytest.approx(1 / 3)
    assert group_score < pipeline.CATEGORY_GROUP_FLOOR  # -> run_category_stage skips level 2
    assert all(v == pytest.approx(1 / 3) for v in group_scores.values())


def test_top2_groups_orders_highest_to_lowest() -> None:
    _, _, group_scores = pipeline.classify_group_scores([0.0, 5.0, 0.0])
    assert pipeline.top2_groups(group_scores) == [
        "governance_legal_workforce",
        "corporate_actions",
    ]


def test_classify_category_scores_picks_highest_entailment_among_candidates() -> None:
    candidates = pipeline.CATEGORY_GROUP_CHILDREN["corporate_actions"]
    entail_logits = [0.0, 5.0, 0.0]  # mergers_acquisitions (index 1) dominates

    label, score, scores = pipeline.classify_category_scores(entail_logits, candidates)

    assert label == "mergers_acquisitions"
    assert score > pipeline.CATEGORY_CONFIDENCE_THRESHOLD
    assert set(scores) == set(candidates)  # only the given candidates, not all 9
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_classify_category_scores_falls_back_to_other_below_threshold() -> None:
    candidates = pipeline.CATEGORY_GROUP_CHILDREN["corporate_actions"]
    entail_logits = [0.0, 0.0, 0.0]  # uniform -> ~0.333 each, below 0.4

    label, score, _scores = pipeline.classify_category_scores(entail_logits, candidates)

    assert label == "other"
    # `score` still reflects the (sub-threshold) winning slug's own probability,
    # not zero -- that's what makes a near-miss "other" distinguishable from a
    # genuinely flat one when auditing later.
    assert 0 < score < pipeline.CATEGORY_CONFIDENCE_THRESHOLD


# --- run_category_stage -----------------------------------------------------


class BatchAwareTokenizer:
    """chunk_text only needs .encode(text, add_special_tokens=False) to count
    tokens; the batch encoding carries how many (premise, hypothesis) pairs
    it was actually given, so the fake model below can assert each of its
    calls received the right count."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def __call__(self, premises: Any, hypotheses: Any, **kwargs: Any) -> "FakeBatchEncoding":
        assert len(premises) == len(hypotheses)
        return FakeBatchEncoding({"n_pairs": len(premises)})


class FakeBatchEncoding(dict):
    def to(self, device: Any) -> "FakeBatchEncoding":
        return self


class FakeTwoPassCategoryModel:
    """Serves canned entailment logits by call number: call 1 is always
    level 1 (3 values per article in the outer batch, in fetch order);
    call 2 (if it happens at all) is level 2 -- a flat list covering only
    the articles that survived level 1's CATEGORY_GROUP_FLOOR (6 values
    each, in `pending` order). Asserts the pair count each call receives
    matches what was declared, so a batching/offset-slicing bug (wrong
    articles, wrong slice) fails loudly instead of silently cross-wiring
    one article's scores into another's row."""

    def __init__(self, level1_flat: list[float], level2_flat: list[float]) -> None:
        self._level1 = torch.tensor(level1_flat)
        self._level2 = torch.tensor(level2_flat)
        self.call_count = 0
        self.config = type(
            "Config", (), {"label2id": {"contradiction": 0, "neutral": 1, "entailment": 2}}
        )()

    def to(self, device: Any) -> "FakeTwoPassCategoryModel":
        return self

    def eval(self) -> "FakeTwoPassCategoryModel":
        return self

    def __call__(self, **kwargs: Any) -> Any:
        self.call_count += 1
        n_pairs = kwargs["n_pairs"]
        flat = self._level1 if self.call_count == 1 else self._level2
        assert n_pairs == len(flat), (
            f"call {self.call_count}: expected {len(flat)} pairs, got {n_pairs}"
        )
        logits = torch.zeros(n_pairs, 3)
        logits[:, 2] = flat
        return type("Output", (), {"logits": logits})()


def _patch_category_model(
    monkeypatch: pytest.MonkeyPatch, level1_flat: list[float], level2_flat: list[float]
) -> FakeTwoPassCategoryModel:
    monkeypatch.setattr(
        pipeline.AutoTokenizer, "from_pretrained", lambda *_a, **_k: BatchAwareTokenizer()
    )
    fake_model = FakeTwoPassCategoryModel(level1_flat, level2_flat)
    monkeypatch.setattr(
        pipeline.AutoModelForSequenceClassification, "from_pretrained", lambda *_a, **_k: fake_model
    )
    return fake_model


def test_run_category_stage_skips_loading_model_when_nothing_pending(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model should not be loaded when there is nothing to process")

    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", fail_if_called)
    monkeypatch.setattr(
        pipeline.AutoModelForSequenceClassification, "from_pretrained", fail_if_called
    )

    calls = []
    pipeline.run_category_stage(conn, on_progress=lambda *a: calls.append(a))

    assert calls == [("category", 0, 0)]


def test_run_category_stage_writes_winning_label_group_and_zero_fills_unscored_group(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_article(conn, id=1, title="Deal News", body_text="Company X announced a merger today.")
    conn.commit()

    # level 1: corporate_actions wins clearly -> top2 = [corporate_actions, governance_legal_workforce]
    level1 = [5.0, 0.0, 0.0]
    # level-2 candidates, top2-group order: corporate_actions' 3 children then
    # governance_legal_workforce's 3 (earnings_performance, mergers_acquisitions,
    # capital_shareholder_returns, leadership_governance, legal_regulatory,
    # labor_human_capital). mergers_acquisitions (index 1) dominates.
    level2 = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    fake_model = _patch_category_model(monkeypatch, level1, level2)

    pipeline.run_category_stage(conn)

    assert fake_model.call_count == 2  # one level-1 pass + one level-2 pass
    detail = db.get_article_detail(conn, 1)
    assert detail is not None
    assert detail["category"]["label"] == "mergers_acquisitions"
    assert detail["category"]["group_label"] == "corporate_actions"
    assert detail["category"]["group_score"] > pipeline.CATEGORY_GROUP_FLOOR
    assert detail["category"]["mergers_acquisitions"] == detail["category"]["score"]
    # market_product_partnerships (3rd place, never scored at level 2) is
    # zero-filled -- "not evaluated", distinguishable from "evaluated and near-zero".
    for slug in pipeline.CATEGORY_GROUP_CHILDREN["market_product_partnerships"]:
        assert detail["category"][slug] == 0.0
    # both top-2 groups' children got real (nonzero) softmax probabilities
    for slug in (
        *pipeline.CATEGORY_GROUP_CHILDREN["corporate_actions"],
        *pipeline.CATEGORY_GROUP_CHILDREN["governance_legal_workforce"],
    ):
        assert detail["category"][slug] > 0.0


def test_run_category_stage_flat_level1_short_circuits_without_level2_call(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_article(conn, id=1, title="Roundup", body_text="Markets were mixed today across sectors.")
    conn.commit()

    level1 = [0.0, 0.0, 0.0]  # flat -> group_score ~0.333, below CATEGORY_GROUP_FLOOR
    fake_model = _patch_category_model(monkeypatch, level1, [])

    pipeline.run_category_stage(conn)

    assert fake_model.call_count == 1  # no level-2 forward pass at all for this article
    detail = db.get_article_detail(conn, 1)
    assert detail is not None
    assert detail["category"]["label"] == "other"
    assert detail["category"]["group_label"] == "corporate_actions"  # first-tied, per max()
    assert detail["category"]["group_score"] < pipeline.CATEGORY_GROUP_FLOOR
    # every one of the 9 leaf columns is zero-filled -- "not evaluated", not
    # "confidently rejected".
    for slug in pipeline.CATEGORY_SLUGS:
        assert detail["category"][slug] == 0.0


def test_run_category_stage_recovers_2nd_place_group_via_top2_expansion(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_article(conn, id=1, title="Legal Update", body_text="A court ruling changed things.")
    conn.commit()

    # level 1: corporate_actions narrowly beats governance_legal_workforce;
    # market_product_partnerships is a clear 3rd place -> top2 = [corporate_actions,
    # governance_legal_workforce].
    level1 = [0.2, 0.0, -5.0]
    # candidates: earnings_performance, mergers_acquisitions, capital_shareholder_returns,
    #             leadership_governance, legal_regulatory, labor_human_capital
    # legal_regulatory (index 4 -- a child of the 2nd-place group) dominates decisively.
    level2 = [0.0, 0.0, 0.0, 0.0, 8.0, 0.0]
    _patch_category_model(monkeypatch, level1, level2)

    pipeline.run_category_stage(conn)

    detail = db.get_article_detail(conn, 1)
    assert detail is not None
    # level-1's own winner was corporate_actions -- level-2's top-2
    # expansion recovered the correct answer from the 2nd-place group,
    # which a top-1-only hierarchy would have missed entirely.
    assert detail["category"]["group_label"] == "corporate_actions"
    assert detail["category"]["label"] == "legal_regulatory"
    assert CATEGORY_SLUG_TO_GROUP["legal_regulatory"] == "governance_legal_workforce"


def test_run_category_stage_batches_multiple_articles_across_both_passes(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 articles, one call per pass covering all of them (not one call per
    article), with a mix of short-circuited and surviving articles -- proves
    real cross-article batching survives the two-pass redesign, and that
    each survivor's level-2 slice lands on the right row."""
    seed_article(conn, id=1, title="Deal News", body_text="Company X announced a merger today.")
    seed_article(conn, id=2, title="Roundup", body_text="Markets were mixed today across sectors.")
    seed_article(
        conn, id=3, title="Board Shakeup", body_text="The board replaced its chief executive."
    )
    conn.commit()
    assert pipeline.CATEGORY_BATCH_SIZE >= 3, "test assumes all 3 seeded articles land in one batch"

    # level 1, one triple per article in fetch (id) order:
    #   article 1: corporate_actions wins clearly           -> survives
    #   article 2: flat                                     -> short-circuits to 'other'
    #   article 3: governance_legal_workforce wins clearly   -> survives
    level1 = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]
    # level 2, only articles 1 and 3 (article 2 contributes nothing), 6 each:
    #   article 1 candidates: earnings_performance, mergers_acquisitions,
    #     capital_shareholder_returns, leadership_governance, legal_regulatory,
    #     labor_human_capital -- capital_shareholder_returns (index 2) dominates.
    #   article 3 candidates (top2 = [governance_legal_workforce, corporate_actions]):
    #     leadership_governance, legal_regulatory, labor_human_capital,
    #     earnings_performance, mergers_acquisitions, capital_shareholder_returns --
    #     labor_human_capital (index 2) dominates.
    level2 = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    fake_model = _patch_category_model(monkeypatch, level1, level2)

    pipeline.run_category_stage(conn)

    assert fake_model.call_count == 2  # one level-1 pass + one level-2 pass, not per-article

    d1 = db.get_article_detail(conn, 1)
    d2 = db.get_article_detail(conn, 2)
    d3 = db.get_article_detail(conn, 3)
    assert d1 is not None and d1["category"]["label"] == "capital_shareholder_returns"
    assert d1["category"]["group_label"] == "corporate_actions"
    assert d2 is not None and d2["category"]["label"] == "other"
    assert d3 is not None and d3["category"]["label"] == "labor_human_capital"
    assert d3["category"]["group_label"] == "governance_legal_workforce"
