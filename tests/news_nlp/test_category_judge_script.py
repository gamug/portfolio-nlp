"""The seeded draw of the step-7 judge script (T-209): a seeded sample of a committed file, same rows every time."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "category_judge_script", ROOT / "scripts" / "category_judge_2026_11_15.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csvs(tmp_path: Path, n: int) -> tuple[Path, Path]:
    texts, labels = tmp_path / "texts.csv", tmp_path / "labels.csv"
    texts.write_text(
        "idx,article_id,ticker,company,text_source,text\n"
        + "".join(f"{i},a{i},T{i},Co{i},original,text {i}\n" for i in range(n)),
        encoding="utf-8",
    )
    labels.write_text(
        "idx,article_id,ticker,tag,selected,untouched\n"
        + "".join(f"{i},a{i},T{i},other,other,other\n" for i in range(n)),
        encoding="utf-8",
    )
    return texts, labels


def test_draw_is_seeded_sorted_and_the_small_draw_is_inside_the_whole(tmp_path: Path) -> None:
    mod = _script()
    texts, labels = _csvs(tmp_path, 20)
    a = mod.draw(5, 7, texts, labels, population=20)
    b = mod.draw(5, 7, texts, labels, population=20)
    assert list(a.idx) == list(b.idx) and len(a) == 5
    assert list(a.idx) == sorted(a.idx, key=int)
    whole = mod.draw(20, 7, texts, labels, population=20)
    assert set(a.idx) <= set(whole.idx) and len(whole) == 20


def test_draw_refuses_a_population_that_is_not_the_expected_one(tmp_path: Path) -> None:
    mod = _script()
    texts, labels = _csvs(tmp_path, 10)
    with pytest.raises(AssertionError):
        mod.draw(5, 7, texts, labels, population=20)
