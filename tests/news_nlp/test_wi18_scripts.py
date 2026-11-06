"""The draw and build scripts of Work item 18 (2026-11-05): the pure parts, without a database."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import retag

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELDOUT = _script("draw_heldout_2026_11_05")
BUILD = _script("build_company_segments_2026_11_05")
SENTIMENT = _script("draw_sentiment_gold_2026_11_05")
CHECK = _script("draw_segment_check_2026_11_05")


def _frame() -> list[dict[str, str]]:
    scopes = ["sector"] * 3 + ["economy"] * 2 + ["single"] * 30 + ["multi"] * 15
    return [
        {"body_hash": f"h{i:03d}", "article_ids": f"{i} {i + 1000}", "rule_scope": s}
        for i, s in enumerate(scopes)
    ]


def test_sample_ids_are_split_on_commas_or_spaces() -> None:
    assert HELDOUT._ids("1 2,3 ,4") == ["1", "2", "3", "4"]


def test_heldout_takes_every_rare_body_and_fills_with_the_rest() -> None:
    got = HELDOUT.draw(_frame(), 10, 7, set())
    assert len(got) == 10
    by = Counter(r["stratum"] for r in got)
    assert by["sector"] == 3 and by["economy"] == 2  # rare scopes are all taken
    assert got == HELDOUT.draw(_frame(), 10, 7, set())  # reproducible
    assert {r["article_id"] for r in got} != {
        r["article_id"] for r in HELDOUT.draw(_frame(), 10, 8, set())
    }


def test_heldout_leaves_out_a_body_that_shares_an_article_with_a_tagged_one() -> None:
    got = HELDOUT.draw(_frame(), 10, 7, {"1000", "20"})  # article 1000 belongs to body 0
    assert "0" not in {r["article_id"] for r in got} and "20" not in {r["article_id"] for r in got}


def test_strata_table_keeps_the_full_frame_size_apart_from_the_free_one() -> None:
    frame = _frame()
    taken = {"0", "1"}  # two sector bodies are left out
    chosen = HELDOUT.draw(frame, 10, 7, taken)
    rows = {r["stratum"]: r for r in HELDOUT.strata_rows(frame, taken, chosen)}
    assert (rows["sector"]["frame"], rows["sector"]["excluded"], rows["sector"]["free"]) == (
        "3",
        "2",
        "1",
    )
    assert rows["sector"]["drawn"] == "1"
    assert sum(int(r["frame"]) for r in rows.values()) == len(frame)


def test_the_val_split_is_stable_and_about_one_body_in_ten() -> None:
    hashes = [f"{i * 2654435761 % 2**32:08x}" + "0" * 24 for i in range(2000)]
    split = [BUILD.split_of(h) for h in hashes]
    assert split == [BUILD.split_of(h) for h in hashes]
    assert 0.07 < split.count("val") / len(split) < 0.13


def test_building_stops_and_names_the_local_files_a_clean_clone_lacks(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"heldout_ids\.csv"):
        BUILD.tagged_ids(tmp_path)


def _texts() -> list[dict[str, str]]:
    labels = ["positive"] * 8 + ["negative"] * 5 + ["neutral"] * 20
    return [
        {"article_id": str(i), "ticker": "T" if i % 2 else "", "kind": "summary", "predicted": p}
        for i, p in enumerate(labels)
    ]


def test_sentiment_draw_takes_per_class_texts_and_keeps_the_stratum_sizes() -> None:
    got = SENTIMENT.draw(_texts(), 4, 1)
    by = Counter(t["predicted"] for t in got)
    assert by == {"positive": 4, "negative": 4, "neutral": 4}
    assert {t["predicted"]: t["population"] for t in got} == {
        "positive": "8",
        "negative": "5",
        "neutral": "20",
    }
    assert all("#" in t["key"] for t in got)
    assert [t["key"] for t in got] == [t["key"] for t in SENTIMENT.draw(_texts(), 4, 1)]


def test_sentiment_draw_takes_the_whole_class_when_it_is_short() -> None:
    got = SENTIMENT.draw(_texts(), 6, 1)
    assert Counter(t["predicted"] for t in got)["negative"] == 5


def test_a_text_is_read_as_the_mean_of_its_chunks() -> None:
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    chunks = [
        [0.8, 0.1, 0.1],
        [0.1, 0.5, 0.4],
        [0.1, 0.5, 0.4],
    ]  # means: positive 0.33, negative 0.37
    assert SENTIMENT.predicted_label(chunks, id2label) == "negative"
    assert SENTIMENT.predicted_label(chunks[:1], id2label) == "positive"


def _gold_rows() -> list[dict[str, str]]:
    return [
        {
            "article_id": str(i),
            "body_hash": f"h{i}",
            "year": "2020",
            "split": "val" if i % 10 == 0 else "train",
            "segment_index": str(i % 7),
            "ticker": "AAPL",
            "kind": "lead" if i % 3 else "continuation",
        }
        for i in range(1, 300)
    ]


def test_segment_check_takes_per_kind_paragraphs_from_the_train_split_only() -> None:
    got = CHECK.pick(_gold_rows(), 20, 5)
    assert Counter(r["kind"] for r in got) == {"lead": 20, "continuation": 20}
    assert all(r["split"] == "train" for r in got)
    assert len({r["key"] for r in got}) == 40 and all("#" in r["key"] for r in got)
    assert got == CHECK.pick(_gold_rows(), 20, 5)  # reproducible
    assert [r["kind"] for r in got] != sorted(r["kind"] for r in got)  # order hides the kind


def test_segment_check_columns_keep_the_spanish_pair_after_text() -> None:
    cols = retag.SEGMENT_CHECK_COLUMNS
    assert cols[cols.index("text") + 1 : cols.index("text") + 3] == (
        "spanish_title",
        "spanish_text",
    )
    row = dict.fromkeys(cols, "x")
    assert retag.tagging_format_problems([row], cols) == []
