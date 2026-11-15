"""The multi-label reading of the category scores (T-187): labels of a row, the set a row is given,
the pairs a fine-tune learns from, and the figures."""

from __future__ import annotations

import random

import numpy as np
import pytest

import category_multilabel as ml

LABELS = ("a", "b", "c", "d")


def _rng() -> random.Random:
    return random.Random(0)  # noqa: S311 -- a reproducible draw, not security


def test_parse_labels_keeps_order_and_drops_repeats() -> None:
    assert ml.parse_labels("a, b", LABELS) == ["a", "b"]
    assert ml.parse_labels("a,a", LABELS) == ["a"]
    assert ml.parse_labels(" b ,a,b", LABELS) == ["b", "a"]


def test_parse_labels_drops_uncertain_and_refuses_unknown_labels() -> None:
    assert ml.parse_labels("a,uncertain", LABELS) == ["a"]
    assert ml.parse_labels("uncertain", LABELS) == []
    with pytest.raises(ValueError, match="zzz"):
        ml.parse_labels("a,zzz", LABELS)
    with pytest.raises(ValueError):
        ml.parse_labels("", LABELS)


def test_predict_set_always_has_the_top_label() -> None:
    s = np.array([0.1, 0.9, 0.2, 0.0])
    assert ml.predict_set(s, LABELS, min_score=0.99, ratio=1.0) == ["b"]


def test_predict_set_adds_labels_over_the_score_and_the_ratio() -> None:
    s = np.array([0.8, 0.9, 0.5, 0.85])
    assert ml.predict_set(s, LABELS, min_score=0.6, ratio=0.0) == ["b", "d", "a"]
    # c is under the score, a is under the ratio of the top (0.8 < 0.9 * 0.9)
    assert ml.predict_set(s, LABELS, min_score=0.6, ratio=0.9) == ["b", "d"]


def test_predict_set_is_capped() -> None:
    s = np.array([0.9, 0.9, 0.9, 0.9])
    assert len(ml.predict_set(s, LABELS, min_score=0.0, ratio=0.0, max_labels=3)) == 3


def test_negatives_are_never_a_gold_label() -> None:
    rng = _rng()
    gold = [["a", "b"], ["c"]]
    for _ in range(50):
        for i, label, entails in ml.label_pairs(gold, LABELS, neg=2, rng=rng):
            if entails == 0:
                assert label not in gold[i]


def test_every_gold_label_is_a_positive_pair() -> None:
    pairs = ml.label_pairs([["a", "b"], ["c"]], LABELS, neg=1, rng=_rng())
    assert sorted((i, label) for i, label, e in pairs if e == 1) == [(0, "a"), (0, "b"), (1, "c")]


def test_first_mode_keeps_only_the_first_label_as_positive() -> None:
    pairs = ml.label_pairs([["a", "b"]], LABELS, neg=2, rng=_rng(), first_only=True)
    assert [(i, label) for i, label, e in pairs if e == 1] == [(0, "a")]
    # the second gold label is not a negative: the old recipe would have drawn it, this one does not
    assert all(label != "b" for _, label, e in pairs if e == 0)


def test_negatives_are_capped_by_the_labels_left() -> None:
    pairs = ml.label_pairs([["a", "b", "c"]], LABELS, neg=4, rng=_rng())
    assert sum(1 for *_, e in pairs if e == 0) == 1


def test_figures_of_a_perfect_and_a_wrong_prediction() -> None:
    gold = [["a"], ["a", "b"], ["c"]]
    perfect = [["a"], ["b", "a"], ["c"]]
    f = ml.figures(gold, perfect, LABELS)
    assert f["hit_at_1"] == 1.0 and f["exact"] == 1.0 and f["micro_f1"] == 1.0
    wrong = [["d"], ["a"], ["c", "d"]]
    g = ml.figures(gold, wrong, LABELS)
    assert g["hit_at_1"] == pytest.approx(2 / 3)  # row 2 top is a, in the gold
    assert g["exact"] == pytest.approx(0.0)
    # tp: a (row 2), c (row 3) = 2; predicted pairs 4; gold pairs 4
    assert g["micro_precision"] == pytest.approx(0.5)
    assert g["micro_recall"] == pytest.approx(0.5)


def test_macro_f1_counts_only_labels_with_gold_rows() -> None:
    f = ml.figures([["a"], ["b"]], [["a"], ["b"]], LABELS)
    assert f["macro_f1"] == 1.0  # c and d have no gold row and are not averaged in


def test_bootstrap_interval_is_by_group_and_contains_the_mean() -> None:
    v = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=float)
    groups = [1, 1, 2, 3, 4, 5, 6, 7]
    lo, hi = ml.bootstrap_ci(v, groups, boot=300, seed=0)
    assert lo <= v.mean() <= hi


def test_folds_keep_an_article_in_one_fold() -> None:
    ids = [1, 1, 2, 3, 3, 3, 4, 5, 6, 7]
    fold = ml.folds_of(ids, 3, seed=0)
    for a in set(ids):
        assert len({fold[i] for i, x in enumerate(ids) if x == a}) == 1


def test_rule_is_picked_on_other_folds_only() -> None:
    # two clean folds: rows are single-label with a clear top; a second label is never right
    rng = np.random.default_rng(0)
    n = 40
    S = rng.random((n, 4)) * 0.3
    gold = []
    for i in range(n):
        j = i % 4
        S[i, j] = 0.9
        gold.append([LABELS[j]])
    fold = np.array([i % 2 for i in range(n)])
    rules, sets = ml.cross_fitted_sets(S, gold, LABELS, fold)
    assert len(rules) == 2 and len(sets) == n
    assert all(s == g for s, g in zip(sets, gold, strict=True))


def test_gold_columns_are_the_tags_only() -> None:
    assert "text" not in ml.GOLD_COLUMNS and "labels" in ml.GOLD_COLUMNS


def _sheet_row(order: str, label: str) -> dict[str, str]:
    return {
        "review_order": order,
        "article_id": "1",
        "ticker": "KO",
        "company": "Coca-Cola",
        "pub_date": "2019-01-01",
        "source_domain": "cnbc.com",
        "title": "t",
        "text": "x",
        "spanish_text": "y",
        "label": label,
        "notes": "",
    }


def test_gold_rows_keep_the_tags_and_the_text_without_the_spanish_aid() -> None:
    gold, texts, left_out = ml.gold_rows([_sheet_row("1", "a, b")], LABELS)
    assert gold == [
        {
            "review_order": "1",
            "article_id": "1",
            "ticker": "KO",
            "company": "Coca-Cola",
            "labels": "a,b",
        }
    ]
    assert "spanish_text" not in texts[0] and texts[0]["text"] == "x" and left_out == []


def test_gold_rows_leave_out_a_row_of_only_uncertain() -> None:
    gold, texts, left_out = ml.gold_rows(
        [_sheet_row("1", "uncertain"), _sheet_row("2", "a")], LABELS
    )
    assert [g["review_order"] for g in gold] == ["2"] and left_out == ["1"] and len(texts) == 1


def test_gold_rows_stop_on_a_bad_cell_and_name_the_row() -> None:
    with pytest.raises(ValueError, match="review_order 3"):
        ml.gold_rows([_sheet_row("3", "El servicio de entrega")], LABELS)
    with pytest.raises(ValueError, match="twice"):
        ml.gold_rows([_sheet_row("1", "a"), _sheet_row("1", "b")], LABELS)


def test_per_label_precision_and_recall_are_one_versus_rest() -> None:
    gold = [["a"], ["a", "b"], ["c"], ["b"]]
    pred = [["a"], ["a"], ["a", "c"], ["b"]]
    out = ml.per_label(gold, pred, LABELS)
    assert out["a"]["support"] == 2 and out["a"]["predicted"] == 3
    assert out["a"]["precision"] == pytest.approx(2 / 3) and out["a"]["recall"] == 1.0
    assert out["b"]["precision"] == 1.0 and out["b"]["recall"] == pytest.approx(0.5)
    assert out["c"]["precision"] == 1.0 and out["c"]["recall"] == 1.0
    assert out["d"]["support"] == 0 and out["d"]["recall"] is None  # no gold row: not defined


def test_roc_auc_ovr_by_rank_and_undefined_without_both_classes() -> None:
    assert ml.roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert ml.roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert ml.roc_auc([0.5, 0.5], [1, 0]) == 0.5  # a tie counts half
    assert ml.roc_auc([0.5, 0.6], [1, 1]) is None


def test_explode_pairs_makes_one_row_per_gold_tag_with_the_same_output() -> None:
    rows = ml.explode_pairs([["a", "b"], ["c"]], ["a", "d"])
    assert rows == [(0, "a", "a"), (0, "b", "a"), (1, "c", "d")]


def test_pair_ceiling_is_the_best_a_single_output_can_do_on_the_pairs() -> None:
    assert ml.pair_ceiling([["a", "b"], ["c"]]) == pytest.approx(2 / 3)  # 3 pairs, at most 2 right


def test_a_threshold_per_label_is_the_best_f1_cut_on_the_rows_it_is_given() -> None:
    # label a: positives score 0.8, negatives 0.3 -> any cut in (0.3, 0.8] is perfect, the middle one is taken
    S = np.array([[0.8, 0.1], [0.8, 0.1], [0.3, 0.9], [0.3, 0.9]])
    gold = [["a"], ["a"], ["b"], ["b"]]
    th = ml.fit_thresholds(S, gold, ("a", "b"))
    assert 0.3 < th["a"] <= 0.8 and 0.1 < th["b"] <= 0.9


def test_a_label_without_a_gold_row_gets_the_default_threshold() -> None:
    S = np.array([[0.8, 0.1, 0.2]])
    th = ml.fit_thresholds(S, [["a"]], ("a", "b", "c"))
    assert th["b"] == ml.DEFAULT_THRESHOLD and th["c"] == ml.DEFAULT_THRESHOLD


def test_the_set_by_thresholds_is_never_empty_and_is_best_first() -> None:
    th = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
    assert ml.set_by_thresholds(np.array([0.6, 0.9, 0.1, 0.55]), LABELS, th) == ["b", "a", "d"]
    assert ml.set_by_thresholds(np.array([0.1, 0.3, 0.2, 0.0]), LABELS, th) == [
        "b"
    ]  # none over: the top


def test_cross_fitted_thresholds_use_only_the_other_folds() -> None:
    S = np.array([[0.9, 0.0], [0.9, 0.0], [0.0, 0.9], [0.0, 0.9], [0.2, 0.0], [0.0, 0.9]])
    gold = [["a"], ["a"], ["b"], ["b"], ["a"], ["b"]]
    fold = [1, 1, 1, 1, 0, 0]
    sets = ml.cross_fitted_thresholds(S, gold, ("a", "b"), fold)
    other = ml.fit_thresholds(
        S[:4], gold[:4], ("a", "b")
    )  # what fold 0 may see: the rows of fold 1
    assert sets[4] == ml.set_by_thresholds(S[4], ("a", "b"), other)
    assert sets[5] == ml.set_by_thresholds(S[5], ("a", "b"), other)
    assert len(sets) == 6 and all(sets)


MERGE = {"c": "d"}  # c is folded into d


def test_merge_gold_maps_tags_and_drops_the_repeat() -> None:
    assert ml.merge_gold([["a", "c"], ["c", "d"], ["b"]], MERGE) == [["a", "d"], ["d"], ["b"]]


def test_merge_labels_keeps_the_order_without_the_folded_label() -> None:
    assert ml.merge_labels(LABELS, MERGE) == ("a", "b", "d")


def test_merge_scores_takes_the_largest_score_of_the_labels_folded_together() -> None:
    S = np.array([[0.1, 0.2, 0.9, 0.3], [0.1, 0.2, 0.1, 0.4]])
    out = ml.merge_scores(S, LABELS, MERGE)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [0.1, 0.2, 0.9] and out[1].tolist() == [0.1, 0.2, 0.4]


def test_article_weights_give_each_article_the_same_total() -> None:
    w = ml.article_weights(["x", "x", "y", "z", "z", "z"])
    assert w == pytest.approx([0.5, 0.5, 1.0, 1 / 3, 1 / 3, 1 / 3])
    assert sum(w) == pytest.approx(3.0)


def test_weighted_per_label_counts_rows_by_their_weight() -> None:
    gold = [["a"], ["a"], ["b"]]
    pred = [["a"], ["b"], ["b"]]
    plain = ml.per_label(gold, pred, LABELS)
    assert plain["a"]["recall"] == 0.5
    # the first two rows are one article, so each counts half; the third counts once
    w = ml.per_label(gold, pred, LABELS, weights=[0.5, 0.5, 1.0])
    assert w["a"]["support"] == pytest.approx(1.0) and w["a"]["recall"] == 0.5
    assert w["b"]["predicted"] == pytest.approx(1.5) and w["b"]["precision"] == pytest.approx(
        1.0 / 1.5
    )


def test_per_label_interval_resamples_articles_and_holds_the_estimate() -> None:
    gold = [["a"], ["a"], ["b"], ["b"], ["a"], ["b"]]
    pred = [["a"], ["a"], ["b"], ["a"], ["a"], ["b"]]
    arts = ["1", "1", "2", "3", "4", "5"]
    out = ml.per_label_ci(gold, pred, LABELS, arts, boot=200, seed=0)
    interval = out["a"]["recall"]
    assert interval is not None
    lo, hi = interval
    assert lo <= 1.0 <= hi and lo >= 0.0
    assert out["d"]["recall"] is None  # a label with no gold row has no interval


def test_markdown_table_has_a_header_a_rule_and_one_line_per_row() -> None:
    out = ml.markdown_table(["label", "n"], [["a", 1], ["b", None]])
    assert out.splitlines() == ["| label | n |", "|---|---|", "| a | 1 |", "| b | - |"]


def test_length_batches_cover_every_row_once_and_group_similar_lengths() -> None:
    lengths = [5, 90, 7, 80, 6, 85, 8, 95]
    batches = ml.length_batches(lengths, bs=2, rng=_rng(), pool=8)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(8)) and all(len(b) <= 2 for b in batches)
    spread = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches]
    assert max(spread) <= 15  # short with short, long with long


def test_pos_weights_are_the_capped_root_of_the_odds() -> None:
    gold = [["a"], ["a"], ["a"], ["b"]]
    w = ml.pos_weights(gold, ("a", "b", "c"), cap=4.0)
    assert w["a"] == 1.0 and w["b"] == pytest.approx(3**0.5)
    assert w["c"] == 4.0  # no positive row: the cap, not a division by zero


def test_one_global_cut_gives_every_label_the_same_threshold() -> None:
    assert ml.global_thresholds(LABELS, 0.4) == {"a": 0.4, "b": 0.4, "c": 0.4, "d": 0.4}
