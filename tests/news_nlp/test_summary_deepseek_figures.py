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
