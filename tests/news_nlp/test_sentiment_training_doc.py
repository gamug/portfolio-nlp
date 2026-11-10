"""``docs/sentiment_training.md`` quotes figures of local MLflow runs; ``docs/sentiment_training_results.csv`` is the
tracked export of those runs (``scripts/export_sentiment_training_results_2026_11_10.py``). The headline figures of
the page must be in the export, in the run it records, so a figure cannot drift from its source unseen."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = (ROOT / "docs" / "sentiment_training.md").read_text(encoding="utf-8")

#: (experiment, run name, metric, the value as the page writes it, a pattern of the page where it is stated)
HEADLINES = [
    ("sentiment_pair_cv_all_data", "cv_summary", "cv_score", "0.737", r"\| score \| 0\.737 \[0\.716"),
    ("sentiment_pair_cv_all_data", "cv_summary", "cv_neutral_recall", "0.442", r"neutral precision / recall \| 0\.634 / 0\.442"),
    ("sentiment_pair_cv_all_data_replay809", "cv_summary", "cv_score", "0.723", r"score 0\.723\s+\[0\.699"),
    ("sentiment_pair_relevance_gate", "nested_none", "score", "0.737", r"\| score \| 0\.737 \| 0\.789 \| \*\*0\.814\*\*"),
    ("sentiment_pair_relevance_gate", "nested_gate", "score", "0.789", r"\| score \| 0\.737 \| 0\.789 \| \*\*0\.814\*\*"),
    ("sentiment_pair_relevance_gate", "nested_stack", "score", "0.814", r"\| score \| 0\.737 \| 0\.789 \| \*\*0\.814\*\*"),
    ("sentiment_pair_relevance_gate", "nested_stack", "neutral_precision", "0.787", r"\*\*0\.787\*\* / 0\.686"),
    ("sentiment_pair_relevance_gate", "nested_none", "relevance_auc", "0.896", r"\(mean 0\.896\)"),
    ("company_relevance_zeroshot", "zeroshot_relevance_auc", "auc_zs_summary_H2_about_company", "0.743", r"weighted AUC 0\.743 alone"),
    ("company_relevance_zeroshot", "zeroshot_relevance_auc", "auc_cv_sentiment_model", "0.795", r"against 0\.795"),
    ("sentiment_pair_final_all", "final_all__e2_lr3e-05_s4", "train_loss", "0.394", r"epoch losses 0\.820 and 0\.394"),
]  # fmt: skip


def _recorded() -> dict[tuple[str, str, str], float]:
    with (ROOT / "docs" / "sentiment_training_results.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        return {
            (r["experiment"], r["run_name"], r["metric"]): float(r["value"])
            for r in csv.DictReader(fh)
            if r["recorded"] == "yes"
        }


@pytest.mark.parametrize(("experiment", "run", "metric", "written", "where"), HEADLINES)
def test_a_headline_figure_of_the_page_is_the_recorded_run_of_the_export(
    experiment: str, run: str, metric: str, written: str, where: str
) -> None:
    value = _recorded()[(experiment, run, metric)]
    assert f"{value:.3f}" == written, (experiment, run, metric, value)
    assert re.search(where, DOC), f"the page does not state {written} where it should ({where})"
    assert f"`{experiment}`" in DOC, f"{experiment} is not named on the page"


def test_each_run_name_has_one_recorded_run() -> None:
    with (ROOT / "docs" / "sentiment_training_results.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        rows = list(csv.DictReader(fh))
    recorded = {
        (r["experiment"], r["run_name"], r["run_id"]) for r in rows if r["recorded"] == "yes"
    }
    names = [(e, n) for e, n, _ in recorded]
    assert len(names) == len(set(names))
