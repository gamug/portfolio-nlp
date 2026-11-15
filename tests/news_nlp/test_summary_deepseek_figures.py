"""The figures of the DeepSeek arm quoted in ``docs/summarization_training.md`` (T-206) are the ones of the committed files, with no text in them."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs" / "summarization_results_2026_11_15"


def _report() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rep", ROOT / "scripts" / "summary_deepseek_report_2026_11_15.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_committed_tables_are_the_figures_of_the_doc() -> None:
    table = _report().tables()
    assert (FOLDER / "tables.md").read_text(
        encoding="utf-8"
    ) == table  # the committed table is the one the files give
    doc = (ROOT / "docs" / "summarization_training.md").read_text(encoding="utf-8")
    for figure in (
        "4.99",
        "4.60",
        "4.54",
        "3.24",
        "4.98",
        "4.08",
        "0.7%",
        "15.0%",
        "+0.39 [+0.29, +0.49]",
        "+1.29 [+1.21, +1.39]",
        "-14.3 pts [-18.7, -10.3]",
    ):
        assert figure in table
        assert figure in doc, figure


def test_the_committed_figures_hold_no_article_or_summary_text() -> None:
    for name in ("deepseek_vs_current.csv", "blind_24.csv"):
        with (FOLDER / name).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) in (300, 24)
        assert all(
            len(v) < 40 for r in rows for v in r.values()
        )  # numbers, ids and short codes only


def test_the_cost_and_the_populations_of_the_doc_are_the_committed_table() -> None:
    table = (FOLDER / "chars_to_summarise.md").read_text(encoding="utf-8")
    doc = (ROOT / "docs" / "summarization_training.md").read_text(encoding="utf-8")
    tasks = (ROOT / ".specify" / "memory" / "TASKS.md").read_text(encoding="utf-8")
    for figure in (
        "59,432",
        "235,078,518",
        "5,048",
        "15,263,579",
        "14,803",
        "60,503,969",
        "79,283",
        "310,846,066",
    ):
        assert figure in table and figure in doc, figure
    assert "$16.0" in table and "$31.9" in table  # the range the doc rounds to about $16 to $32
    assert "about $16 (off-peak) to $32 (peak)" in doc and "79,283 items" in tasks


def test_the_repeat_file_has_30_articles_and_the_doc_quotes_its_result() -> None:
    with (FOLDER / "deepseek_repeat_30.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and not any(len(v) > 40 for r in rows for v in r.values())
    doc = (ROOT / "docs" / "summarization_training.md").read_text(encoding="utf-8")
    assert sum(int(r["identical"]) for r in rows) == 0 and "0 identical summaries" in doc
