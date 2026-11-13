"""The figures of the cut sample the owner marked (T-183): shares of yes per question with an interval, on the whole sample and without the pieces of a body scoped wrongly or too ambiguous to cut."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import cut_sample_metrics_2026_11_12 as metrics

Row = dict[str, str]


def _row(
    order: int, article: int, f: str, o: str, c: str, kind: str = "kept", notes: str = ""
) -> Row:
    return {
        "review_order": str(order),
        "article_id": str(article),
        "faithful": f,
        "one_story": o,
        "complete": c,
        "kind": kind,
        "notes": notes,
    }


def test_the_share_has_a_wilson_interval_inside_zero_and_one() -> None:
    share, low, high = metrics.wilson(8, 10)
    assert share == 0.8
    assert 0.49 < low < 0.50 and 0.94 < high < 0.95
    assert metrics.wilson(0, 10)[1] == 0.0 and metrics.wilson(10, 10)[2] == 1.0
    assert metrics.wilson(0, 0) == (0.0, 0.0, 0.0)


def test_the_levels_drop_first_the_wrong_scope_then_the_ambiguous_and_keep_the_wrong_company() -> (
    None
):
    rows = [
        _row(1, 1, "yes", "yes", "yes"),
        _row(2, 1, "no", "no", "no", "ambiguous", "mix of companies"),
        _row(3, 2, "no", "no", "no", "wrong_scope", "This is a misslabeled news"),
        _row(4, 2, "yes", "yes", "no", "wrong_company", "Company is lost"),
    ]
    levels = metrics.levels(rows)
    assert [lv.name for lv in levels] == ["all", "right_scope", "right_scope_unambiguous"]
    assert [lv.pieces for lv in levels] == [4, 3, 2]
    assert [lv.bodies for lv in levels] == [2, 2, 2]
    assert [lv.yes["faithful"] for lv in levels] == [2, 2, 2]
    assert [lv.yes["complete"] for lv in levels] == [
        1,
        1,
        1,
    ]  # the wrong company never leaves a level: its "no" is the cut's


def test_a_clean_piece_is_yes_on_all_three_and_a_clean_body_has_only_clean_pieces() -> None:
    rows = [
        _row(1, 1, "yes", "yes", "yes"),
        _row(2, 1, "yes", "yes", "no"),
        _row(3, 2, "yes", "yes", "yes"),
    ]
    lv = metrics.levels(rows)[0]
    assert lv.clean_pieces == 2
    assert lv.clean_bodies == 1 and lv.bodies == 2


def test_the_body_bootstrap_is_seeded_and_resamples_whole_bodies() -> None:
    rows = [_row(i, i // 3, "yes" if i % 4 else "no", "yes", "yes") for i in range(40)]
    a = metrics.body_bootstrap(rows, "faithful", seed=7, draws=300)
    assert a == metrics.body_bootstrap(rows, "faithful", seed=7, draws=300)
    assert 0.0 <= a[0] <= a[1] <= 1.0
    same = [_row(i, i // 3, "yes", "yes", "yes") for i in range(30)]
    assert metrics.body_bootstrap(same, "faithful", seed=1, draws=100) == (1.0, 1.0)


def test_an_odd_mark_or_an_unknown_kind_stops_the_run() -> None:
    with pytest.raises(SystemExit, match="review_order 1"):
        metrics.check_rows([_row(1, 1, "yes", "", "yes")])
    with pytest.raises(SystemExit, match="maybe"):
        metrics.check_rows([_row(2, 1, "maybe", "yes", "yes")])
    with pytest.raises(SystemExit, match="odd"):
        metrics.check_rows([_row(3, 1, "yes", "yes", "yes", kind="odd")])
    metrics.check_rows(
        [_row(4, 1, " Yes", "NO", "yes")]
    )  # case and spaces are the owner's, not an error


def test_the_report_leads_with_the_bar_on_all_pieces_and_names_every_row_left_out_or_noted() -> (
    None
):
    rows = [_row(i, i, "yes", "yes", "yes") for i in range(1, 8)] + [
        _row(8, 8, "no", "no", "no", "wrong_scope", "This is a misslabeled news"),
        _row(9, 9, "yes", "no", "yes", "ambiguous", "Two companies there"),
        _row(10, 10, "yes", "yes", "yes", "wrong_company", "Company missunderstood"),
    ]
    text = metrics.report(rows, revision="units5-x", seed=1, draws=100)
    assert text.index("## all") < text.index("## right_scope:")
    assert (
        "units5-x" in text
        and "misslabeled" in text
        and "Two companies" in text
        and "Company missunderstood" in text
    )
    assert "80%" in text and "pre-set measure" in text
    assert "wrong_scope: 1" in text and "ambiguous: 1" in text and "wrong_company: 1" in text
