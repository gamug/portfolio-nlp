"""The category figures of the planned definition (`scripts/category_planned_8_2026_11_16.py`): four labels folded into `other`, which is not
measured. The fold, the Wilson interval, the macro F1 and the scoring over the measured rows; and the pipeline reports quote the committed figures."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "category_planned_8", ROOT / "scripts" / "category_planned_8_2026_11_16.py"
)
assert spec and spec.loader
cp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cp
spec.loader.exec_module(cp)

RESULT = ROOT / "docs" / "category_results_2026_11_15" / "planned_8_labels.json"


def test_the_four_labels_fold_into_other_and_the_rest_stay() -> None:
    assert {cp.fold(x) for x in cp.FOLDED} == {"other"}
    assert all(cp.fold(x) == x for x in cp.MEASURED)
    assert len(cp.MEASURED) == 7


def test_the_wilson_interval_is_the_known_one() -> None:
    assert cp.wilson(48, 84) == (46.5, 67.2)


def test_rows_tagged_other_are_not_scored_and_a_measured_row_predicted_other_is_wrong() -> None:
    gold = [
        "earnings_performance",
        "earnings_performance",
        "product_innovation",
        "reputational_harm",
    ]
    pred = ["earnings_performance", "product_innovation", "other", "earnings_performance"]
    m = cp.measure(gold, pred)
    assert m["rows"] == 4
    assert m["rows_measured"] == 2
    assert m["correct"] == 1
    assert m["accuracy"] == 50.0
    assert m["accuracy_12"] == 25.0


def test_macro_f1_counts_a_false_positive_from_a_row_tagged_other() -> None:
    gold = ["other", "earnings_performance"]
    pred = ["earnings_performance", "earnings_performance"]
    assert cp.macro_f1(gold, pred, ("earnings_performance",)) == pytest.approx(2 / 3)


def test_the_twelve_label_accuracies_are_reproduced_from_the_committed_files() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    got = (
        result["out_of_fold"]["selected"]["accuracy_12"],
        result["out_of_fold"]["untouched"]["accuracy_12"],
        result["held_out"]["selected"]["accuracy_12"],
        result["held_out"]["untouched"]["accuracy_12"],
        result["real_sub_news"]["selected"]["accuracy_12"],
        result["real_sub_news"]["untouched"]["accuracy_12"],
    )
    assert got == (87.8, 75.5, 54.1, 43.7, 71.3, 65.7)


def test_the_committed_json_is_what_the_script_computes_now() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["out_of_fold"] == cp.out_of_fold()
    assert result["held_out"] == cp.held_out()
    assert result["real_sub_news"] == cp.real_sample()


def _figures(result: Any) -> list[str]:
    out = []
    for part in ("out_of_fold", "held_out", "real_sub_news"):
        for model in ("selected", "untouched"):
            m = result[part][model]
            lo, hi = m["ci"]
            out += [f"{m['accuracy']}%", f"[{lo}, {hi}] ({m['correct']} "]
    sel = [result[p]["selected"]["macro_f1"] for p in ("out_of_fold", "held_out", "real_sub_news")]
    out.append(f"| {sel[0]:.3f} | {sel[1]:.3f} | {sel[2]:.3f} |")
    return out


@pytest.mark.parametrize("page", ["pipeline_metrics_report.md", "pipeline_metrics_report_es.md"])
def test_the_reports_quote_the_committed_figures(page: str) -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    text = (ROOT / page).read_text(encoding="utf-8").replace("\r\n", "\n")
    for figure in _figures(result):
        assert figure in text, f"{page}: {figure}"


def test_the_training_page_quotes_the_figures_of_the_three_samples() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    text = (
        (ROOT / "docs" / "category_training.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    )
    for part in ("out_of_fold", "held_out", "real_sub_news"):
        m = result[part]["selected"]
        lo, hi = m["ci"]
        assert f"{m['accuracy']}%" in text
        assert f"[{lo}, {hi}]" in text


def test_the_judge_reading_is_lenient_by_default_and_the_fold_raises_it() -> None:
    """Official reading (owner, 2026-11-17): the judge's first or second label; the fold can only add agreements."""
    j = cp.judge_step7()
    sel = j["selected"]
    assert j["rows"] == 300
    assert (sel["lenient_12"]["correct"], sel["strict_12"]["correct"]) == (264, 231)
    assert sel["lenient_12"]["accuracy"] == 88.0
    for model in ("selected", "untouched"):
        m = j[model]
        assert m["lenient_12"]["correct"] >= m["strict_12"]["correct"]
        assert m["lenient_8"]["correct"] >= m["lenient_12"]["correct"]
        assert m["strict_8"]["correct"] >= m["strict_12"]["correct"]


def test_the_pin_and_the_reports_quote_the_official_judge_figures() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    sel = result["judge_step7"]["selected"]
    pin = json.loads((ROOT / "config" / "category_step.json").read_text(encoding="utf-8"))[
        "evidence"
    ]
    assert pin["judge_official_selected_agreement"] == round(sel["lenient_12"]["accuracy"] / 100, 3)
    assert pin["judge_official_8_labels_selected_agreement"] == round(
        sel["lenient_8"]["accuracy"] / 100, 3
    )
    assert pin["judge_official_7_labels_selected_agreement"] == round(
        sel["lenient_7"]["accuracy"] / 100, 3
    )
    assert pin["bar_met"] is False
    for page in ("pipeline_metrics_report.md", "pipeline_metrics_report_es.md"):
        text = (ROOT / page).read_text(encoding="utf-8").replace("\r\n", "\n")
        for key in ("lenient_12", "lenient_8", "lenient_7", "strict_8"):
            assert f"{sel[key]['accuracy']}%" in text, f"{page}: {key}"
        for model in ("out_of_fold", "held_out", "real_sub_news"):
            assert f"{result[model]['selected']['accuracy_8']}%" in text, f"{page}: {model}"
