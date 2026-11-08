"""The draw and build scripts of Work item 18 (2026-11-05): the pure parts, without a database."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import types
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import gold_metrics, retag
from news_nlp.gics_codes import SUB_INDUSTRY_CODES
from news_nlp.gold_metrics import MarkedFileError

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
FAITH = _script("draw_faithfulness_2026_11_06")
SEGGOLD = _script("build_segment_check_gold_2026_11_06")
FAITHGOLD = _script("build_faithfulness_gold_2026_11_06")
SENTGOLD = _script("build_sentiment_gold_2026_11_06")
REPORT = _script("report_wi18_2026_11_06")
BUILD_GOLD = _script("build_gold_2026_11_05")


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
        {
            "article_id": str(i),
            "ticker": "T" if i % 2 else "",
            "kind": "summary",
            "predicted": p,
            "text": f"text {i}",
        }
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


def _csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...] = ()) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields or rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bodies(path: Path) -> None:
    _csv(
        path / "faithfulness_bodies.csv",
        [{"article_id": str(a), "segments": "3"} for a in range(30)],
    )


def _heldout_dir(tmp_path: Path) -> Path:
    out = tmp_path / "data"
    out.mkdir()
    _csv(out / "triage_sample_bodies.csv", _frame())
    _csv(out / "category_gold_v2_review.csv", [{"article_id": "0"}])  # body 0 is in the 720
    _csv(out / "sector_extra_to_tag.csv", [{"article_id": "1"}])
    _csv(out / "heldout_ids.csv", [{"article_id": "5", "stratum": "sector"}])
    _csv(out / "heldout_to_tag.csv", [{"article_id": "5"}])  # the owner's file already exists
    return out


def test_heldout_main_writes_the_strata_even_when_the_owners_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _heldout_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["draw", "--out", str(out)])
    with pytest.raises(SystemExit, match="not overwritten"):
        HELDOUT.main()
    with (out / "heldout_strata.csv").open(encoding="utf-8-sig", newline="") as fh:
        strata = {r["stratum"]: r for r in csv.DictReader(fh)}
    assert (
        strata["sector"]["frame"] == "3" and strata["sector"]["excluded"] == "2"
    )  # article 0 (the 720), article 1 (the 44)
    assert sum(int(r["frame"]) for r in strata.values()) == len(_frame())


def test_other_work_draws_do_not_change_what_the_heldout_leaves_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _heldout_dir(tmp_path)
    before = HELDOUT.tagged_ids(out, "heldout")
    for name in HELDOUT.NOT_HELDOUT:
        _csv(out / name, [{"article_id": "2"}, {"article_id": "3"}])
    assert HELDOUT.tagged_ids(out, "heldout") == before
    _csv(out / "heldout2_ids.csv", [{"article_id": "7"}])  # an earlier held-out draw still counts
    assert "7" in HELDOUT.tagged_ids(out, "heldout")


def test_faithfulness_bodies_are_drawn_in_article_order_and_reproducibly() -> None:
    cand = [{"article_id": str(i), "segments": "3"} for i in range(40, 0, -1)]
    got = FAITH.draw_bodies(cand, 10, 3)
    assert len(got) == 10 and got == FAITH.draw_bodies(cand, 10, 3)
    assert [int(r["article_id"]) for r in got] == sorted(int(r["article_id"]) for r in got)


def test_a_text_two_extractors_wrote_identically_is_one_row_with_both_names() -> None:
    per = {"baseline": {"text": "Same text."}, "model": {"text": "Same text."}}
    assert FAITH.merge_texts(per) == [("Same text.", ("baseline", "model"))]
    per = {"baseline": {"text": "One."}, "model": {"text": "Two."}}
    assert [e for _, e in FAITH.merge_texts(per)] == [("baseline",), ("model",)]
    assert FAITH.merge_texts({"baseline": {"text": " "}, "model": {"text": "Two."}}) == [
        ("Two.", ("model",))
    ]


def test_the_model_extractor_does_not_exist_until_it_is_trained() -> None:
    assert set(FAITH.EXTRACTORS) == {"baseline"}


def test_the_label_check_report_applies_the_rule_fixed_beforehand(tmp_path: Path) -> None:
    rows = [
        {
            "article_id": str(i),
            "kind": "lead" if i < 10 else "continuation",
            "pub_date": "2023-01-01",
            "correct": "yes" if i < 9 or i >= 14 else "no",
        }
        for i in range(20)
    ]
    _csv(tmp_path / "segment_check_to_tag.csv", rows)
    text = "\n".join(REPORT.label_check_report(tmp_path))
    assert "train on `lead` only" in text  # continuation: 6 of 10 right
    rows[10]["correct"] = ""
    _csv(tmp_path / "segment_check_to_tag.csv", rows)
    with pytest.raises(MarkedFileError, match="without correct"):
        REPORT.label_check_report(tmp_path)


def test_the_sentiment_report_joins_the_owners_file_to_the_draw_by_key(tmp_path: Path) -> None:
    ids, tags = [], []
    for i in range(30):
        predicted = ("positive", "negative", "neutral")[i % 3]
        key = f"{i}#T{i}"
        ids.append({"key": key, "predicted": predicted, "population": str(100 * (1 + i % 3))})
        tags.append(
            {
                "article_id": str(i),
                "ticker": f"T{i}",
                "kind": "summary",
                "sentiment": predicted if i % 5 else "neutral",
            }
        )
    _csv(tmp_path / "sentiment_gold_ids.csv", ids)
    _csv(tmp_path / "sentiment_gold_to_tag.csv", tags)
    text = "\n".join(REPORT.sentiment_report(tmp_path))
    assert "overall agreement (weighted)" in text and "negative recall" in text
    tags[0]["ticker"] = "OTHER"
    _csv(tmp_path / "sentiment_gold_to_tag.csv", tags)
    with pytest.raises(MarkedFileError, match="not in sentiment_gold_ids"):
        REPORT.sentiment_report(tmp_path)


def test_the_faithfulness_report_compares_the_model_with_the_baseline_on_the_same_texts(
    tmp_path: Path,
) -> None:
    ids, tags = [], []
    order = 0
    for article in range(10):
        for extractors, faithful in (
            ("baseline|model", "yes"),
            ("model", "yes"),
            ("baseline", "no"),
        ):
            order += 1
            key = f"{article}#T#{order}"
            ids.append(
                {
                    "review_order": str(order),
                    "key": key,
                    "extractors": extractors,
                    "pub_date": "2023-01-01" if article % 2 else "2020-01-01",
                }
            )
            tags.append(
                {"article_id": str(article), "review_order": str(order), "faithful": faithful}
            )
    _csv(tmp_path / "faithfulness_ids.csv", ids)
    _csv(tmp_path / "faithfulness_to_tag.csv", tags)
    _bodies(tmp_path)
    _csv(tmp_path / "faithfulness_coverage.csv", [], FAITH.COVERAGE_FIELDS)
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "baseline" in text and "model" in text and "model minus baseline" in text
    assert "[all]" in text and "[2022+]" in text
    assert "-> better" in text or "-> not decided" in text or "-> not better" in text


def test_the_faithfulness_report_stops_on_what_it_cannot_place(tmp_path: Path) -> None:
    tags = [{"article_id": "1", "review_order": "9", "faithful": "yes"}]
    _csv(tmp_path / "faithfulness_to_tag.csv", tags)
    _csv(
        tmp_path / "faithfulness_ids.csv",
        [{"review_order": "1", "key": "1#T#1", "extractors": "model"}],
    )
    with pytest.raises(MarkedFileError, match="pub_date"):
        REPORT.faithfulness_report(tmp_path)
    _csv(
        tmp_path / "faithfulness_ids.csv",
        [{"review_order": "1", "key": "1#T#1", "extractors": "model", "pub_date": "2023-01-01"}],
    )
    with pytest.raises(MarkedFileError, match="not in the ids file"):
        REPORT.faithfulness_report(tmp_path)


def test_the_scope_report_stops_on_an_id_the_draw_does_not_have(tmp_path: Path) -> None:
    _csv(tmp_path / "heldout_strata.csv", [{"stratum": "single", "frame": "10"}])
    _csv(tmp_path / "heldout_ids.csv", [{"article_id": "1", "stratum": "single"}])
    _csv(tmp_path / "heldout_to_tag.csv", [{"article_id": "2", "scope": "single"}])
    with pytest.raises(MarkedFileError, match="not in heldout_ids"):
        REPORT.scope_report(tmp_path)


def test_every_draw_script_writes_the_owners_file_spreadsheet_safe() -> None:
    for name in (
        "draw_segment_check_2026_11_05",
        "draw_sentiment_gold_2026_11_05",
        "draw_faithfulness_2026_11_06",
        "draw_heldout_2026_11_05",
    ):
        tree = ast.parse((Path(__file__).parents[2] / "scripts" / f"{name}.py").read_text())
        writes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "extra._write"
            and ast.unparse(n.args[0]) == "target"
        ]
        assert writes, name
        assert all(ast.unparse(w.args[1]).startswith("retag.spreadsheet_safe(") for w in writes), (
            name
        )


def test_the_scope_report_does_not_gate_a_scope_under_forty_gold_bodies(tmp_path: Path) -> None:
    strata = [
        {"stratum": s, "frame": f}
        for s, f in (("single", "800"), ("multi", "150"), ("sector", "50"))
    ]
    ids, tags = [], []
    for i in range(60):
        stratum = ("single", "multi", "sector")[i % 3]
        ids.append({"article_id": str(i), "stratum": stratum})
        gold = {"single": "single", "multi": "multi", "sector": "sector"}[stratum]
        tags.append({"article_id": str(i), "scope": gold if i % 7 else "single"})
    _csv(tmp_path / "heldout_strata.csv", strata)
    _csv(tmp_path / "heldout_ids.csv", ids)
    _csv(tmp_path / "heldout_to_tag.csv", tags)
    text = "\n".join(REPORT.scope_report(tmp_path))
    assert (
        "not gated (< 40 gold bodies, D33)" in text and "agreement, weighted to the corpus" in text
    )


def _gold(article_id: int, scope: str = "single") -> retag.Gold:
    return retag.Gold(article_id, scope, ("AAPL",), {"AAPL": ("other",)}, (), ())


def _build_test_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tagged: list[int], fixed: list[int]
) -> Path:
    out = tmp_path / "data"
    out.mkdir()
    gold = tmp_path / "category_gold_v2_test.csv"
    _csv(gold, [{"article_id": str(i), "session": "heldout"} for i in fixed])
    rows = [{"article_id": str(i), "ticker": "AAPL", "notes": ""} for i in tagged]
    _csv(out / "heldout_ids.csv", [{"article_id": str(i), "stratum": "single"} for i in fixed])
    _csv(out / "heldout_strata.csv", [{"stratum": "single", "frame": "514"}])
    monkeypatch.setattr(BUILD_GOLD.retag_cli, "load_universe", lambda: None)
    monkeypatch.setattr(retag, "read_tag_file", lambda _path: rows)
    report = retag.Report([_gold(i) for i in tagged], {}, 0, [], [], [])
    monkeypatch.setattr(retag, "load_gold_v2", lambda *_a, **_k: report)
    monkeypatch.setattr(
        sys, "argv", ["build", "--set", "test", "--out", str(out), "--gold", str(gold)]
    )
    return gold


def test_the_test_set_reads_the_heldout_file_and_writes_the_test_gold() -> None:
    sessions, gold = BUILD_GOLD.SETS["test"]
    assert sessions == {"heldout": "heldout_to_tag.csv"}
    assert gold == Path("gold/category_gold_v2_test.csv")
    assert set(BUILD_GOLD.SETS["train"][0]) == {"gate240", "sector_extra"}  # train is unchanged


def test_the_pull_of_the_test_set_writes_the_tags_of_exactly_the_fixed_articles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = _build_test_set(tmp_path, monkeypatch, tagged=[3, 1, 2], fixed=[1, 2, 3])
    BUILD_GOLD.main()
    with gold.open(encoding="utf-8", newline="") as fh:
        written = list(csv.DictReader(fh))
    assert [(r["article_id"], r["session"], r["scope"]) for r in written] == [
        ("1", "heldout", "single"),
        ("2", "heldout", "single"),
        ("3", "heldout", "single"),
    ]
    assert tuple(written[0]) == retag.TEST_GOLD_COLUMNS
    # the weight is the stratum's size in the whole sample (``frame``), so gold/ alone re-weights a rate
    assert {(r["stratum"], r["frame"]) for r in written} == {("single", "514")}


def test_the_pull_of_the_test_set_refuses_other_articles_and_keeps_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = _build_test_set(tmp_path, monkeypatch, tagged=[1, 2, 9], fixed=[1, 2, 3])
    before = gold.read_text(encoding="utf-8-sig")
    with pytest.raises(SystemExit, match="not the 3 fixed"):
        BUILD_GOLD.main()
    assert gold.read_text(encoding="utf-8-sig") == before


def test_the_pull_refuses_a_file_that_is_not_fully_tagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_test_set(tmp_path, monkeypatch, tagged=[1, 2], fixed=[1, 2])
    report = retag.Report([_gold(1)], {}, 1, [], [], [])
    monkeypatch.setattr(retag, "load_gold_v2", lambda *_a, **_k: report)
    with pytest.raises(SystemExit, match="untagged 1"):
        BUILD_GOLD.main()


def test_the_faithfulness_report_says_how_many_texts_have_no_date(tmp_path: Path) -> None:
    ids = [
        {"review_order": "1", "key": "1#T#1", "extractors": "model", "pub_date": ""},
        {"review_order": "2", "key": "2#T#1", "extractors": "model", "pub_date": "2023-05-01"},
    ]
    tags = [
        {"article_id": "1", "review_order": "1", "faithful": "yes"},
        {"article_id": "2", "review_order": "2", "faithful": "no"},
    ]
    _csv(tmp_path / "faithfulness_ids.csv", ids)
    _csv(tmp_path / "faithfulness_to_tag.csv", tags)
    _bodies(tmp_path)
    _csv(tmp_path / "faithfulness_coverage.csv", [], FAITH.COVERAGE_FIELDS)
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "1 texts have no usable pub_date" in text and "bootstrap over bodies" in text


def _mixed_pool() -> list[dict[str, str]]:
    pool = []
    for i in range(60):
        kind = "extractor_text" if i % 3 == 0 else "summary"
        predicted = ("positive", "negative", "neutral")[i % 5 % 3]
        pool.append(
            {
                "article_id": str(i),
                "ticker": f"T{i}",
                "kind": kind,
                "predicted": predicted,
                "text": f"text {i}",
            }
        )
    return pool


def test_the_sentiment_draw_takes_the_extractor_texts_as_their_own_part() -> None:
    got = SENTIMENT.draw(_mixed_pool(), 4, 1, 2)
    part = Counter(t["part"] for t in got)
    assert part["checked"] <= 12 and part["extractor"] <= 6 and part["extractor"] > 0
    assert all((t["kind"] == "extractor_text") == (t["part"] == "extractor") for t in got)
    sizes = Counter(f"{t['part']}/{t['predicted']}" for t in _mixed_pool_with_parts())
    assert {t["stratum"]: int(t["population"]) for t in got} == {
        k: sizes[k] for k in {t["stratum"] for t in got}
    }
    assert SENTIMENT.draw(_mixed_pool(), 4, 1, 0) and all(
        t["part"] == "checked" for t in SENTIMENT.draw(_mixed_pool(), 4, 1, 0)
    )  # the old call still works


def _mixed_pool_with_parts() -> list[dict[str, str]]:
    return [
        {**t, "part": "extractor" if t["kind"] == "extractor_text" else "checked"}
        for t in _mixed_pool()
    ]


def test_the_sentiment_report_prints_each_part_apart(tmp_path: Path) -> None:
    ids, tags = [], []
    for i in range(36):
        part = "extractor" if i % 2 else "checked"
        predicted = ("positive", "negative", "neutral")[i % 3]
        ids.append(
            {
                "key": f"{i}#T{i}",
                "predicted": predicted,
                "part": part,
                "stratum": f"{part}/{predicted}",
                "population": "50",
            }
        )
        tags.append(
            {
                "article_id": str(i),
                "ticker": f"T{i}",
                "kind": "summary",
                "sentiment": predicted if i % 4 else "neutral",
            }
        )
    _csv(tmp_path / "sentiment_gold_ids.csv", ids)
    _csv(tmp_path / "sentiment_gold_to_tag.csv", tags)
    text = "\n".join(REPORT.sentiment_report(tmp_path))
    assert "[all]" in text and "[checked]" in text and "[extractor]" in text


def test_the_reports_stop_when_the_owner_file_lacks_rows_of_the_draw(tmp_path: Path) -> None:
    _csv(tmp_path / "heldout_strata.csv", [{"stratum": "single", "frame": "10"}])
    _csv(
        tmp_path / "heldout_ids.csv",
        [{"article_id": "1", "stratum": "single"}, {"article_id": "2", "stratum": "single"}],
    )
    _csv(tmp_path / "heldout_to_tag.csv", [{"article_id": "1", "scope": "single"}])
    with pytest.raises(MarkedFileError, match="1 rows of the draw are not in the tagged file"):
        REPORT.scope_report(tmp_path)
    ids = [{"key": f"{i}#T", "predicted": "positive", "population": "9"} for i in range(3)]
    tags = [
        {"article_id": str(i), "ticker": "T", "kind": "summary", "sentiment": "positive"}
        for i in range(2)
    ]
    _csv(tmp_path / "sentiment_gold_ids.csv", ids)
    _csv(tmp_path / "sentiment_gold_to_tag.csv", tags)
    with pytest.raises(MarkedFileError, match="not in the tagged file"):
        REPORT.sentiment_report(tmp_path)
    _csv(
        tmp_path / "faithfulness_ids.csv",
        [
            {
                "review_order": str(i),
                "key": f"{i}#T#1",
                "extractors": "model",
                "pub_date": "2023-01-01",
            }
            for i in (1, 2)
        ],
    )
    _csv(
        tmp_path / "faithfulness_to_tag.csv",
        [{"article_id": "1", "review_order": "1", "faithful": "yes"}],
    )
    with pytest.raises(MarkedFileError, match="not in the tagged file"):
        REPORT.faithfulness_report(tmp_path)


def test_the_sentiment_report_says_whose_weights_and_how_many_texts_are_2022_on(
    tmp_path: Path,
) -> None:
    ids = [{"key": f"{i}#T", "predicted": "negative", "population": "9"} for i in range(4)]
    tags = [
        {
            "article_id": str(i),
            "ticker": "T",
            "kind": "summary",
            "sentiment": "negative",
            "pub_date": "2023-01-01" if i < 3 else "2019-01-01",
        }
        for i in range(4)
    ]
    _csv(tmp_path / "sentiment_gold_ids.csv", ids)
    _csv(tmp_path / "sentiment_gold_to_tag.csv", tags)
    text = "\n".join(REPORT.sentiment_report(tmp_path))
    assert "not in the corpus" in text and "negative: 4 drawn, 3 from 2022 on" in text


def test_an_extractor_that_skips_companies_does_not_win_the_gain(tmp_path: Path) -> None:
    ids, tags = [], []
    for order, article in enumerate(range(1, 11), 1):
        ids.append(
            {
                "review_order": str(order),
                "key": f"{article}#T#1",
                "extractors": "baseline|model",
                "pub_date": "2023-01-01",
            }
        )
        tags.append({"article_id": str(article), "review_order": str(order), "faithful": "yes"})
    skipped = [
        {"article_id": str(a), "ticker": "U", "extractor": "model", "pub_date": "2023-01-01"}
        for a in range(1, 11)
    ]
    _csv(tmp_path / "faithfulness_ids.csv", ids)
    _csv(tmp_path / "faithfulness_to_tag.csv", tags)
    _bodies(tmp_path)
    _csv(tmp_path / "faithfulness_coverage.csv", skipped, FAITH.COVERAGE_FIELDS)
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "10 companies left without one" in text
    assert "model: 50.0%" in text and "baseline: 100.0%" in text
    (tmp_path / "faithfulness_coverage.csv").unlink()
    with pytest.raises(MarkedFileError, match=r"faithfulness_coverage\.csv not found"):
        REPORT.faithfulness_report(tmp_path)


def test_the_sentiment_draw_names_a_short_stratum_and_a_repeated_key() -> None:
    pool = {"checked/positive": 50, "checked/negative": 39, "extractor/neutral": 20}
    assert SENTIMENT.shortfalls(pool, 40, 20) == {
        "checked/negative": (39, 40),
        "checked/neutral": (0, 40),
        "extractor/positive": (0, 20),
        "extractor/negative": (0, 20),
    }
    twins = [
        {"article_id": "1", "ticker": "T", "kind": "summary", "predicted": "positive", "text": "a"},
        {
            "article_id": "1",
            "ticker": "T",
            "kind": "company_text",
            "predicted": "positive",
            "text": "b",
        },
    ]
    with pytest.raises(ValueError, match="share the key"):
        SENTIMENT.draw(twins, 2, 1)


def _faith_files(
    tmp_path: Path, skipped: list[dict[str, str]], extractors: str = "baseline|model"
) -> None:
    ids, tags = [], []
    for order, article in enumerate(range(1, 7), 1):
        ids.append(
            {
                "review_order": str(order),
                "key": f"{article}#T#1",
                "extractors": extractors,
                "pub_date": "2023-01-01",
            }
        )
        tags.append({"article_id": str(article), "review_order": str(order), "faithful": "yes"})
    _csv(tmp_path / "faithfulness_ids.csv", ids)
    _csv(tmp_path / "faithfulness_to_tag.csv", tags)
    _bodies(tmp_path)
    _csv(tmp_path / "faithfulness_coverage.csv", skipped, FAITH.COVERAGE_FIELDS)


def _skip(article: int, extractor: str) -> dict[str, str]:
    return {
        "article_id": str(article),
        "ticker": "U",
        "extractor": extractor,
        "pub_date": "2023-01-01",
    }


def test_the_faithfulness_report_prints_both_readings_of_the_bar(tmp_path: Path) -> None:
    _faith_files(tmp_path, [_skip(1, "model"), _skip(2, "model")])
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "6 faithful of 6 texts + 2 companies left without one" in text
    assert "of the texts it wrote, 100.0%" in text


def test_the_report_gives_the_three_gains_when_the_three_extractors_wrote_texts(
    tmp_path: Path,
) -> None:
    _faith_files(tmp_path, [], extractors="baseline|model|units")
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    for pair in ("model minus baseline", "units minus baseline", "units minus model"):
        assert f"{pair} on 6 company texts" in text
    _faith_files(tmp_path, [], extractors="baseline|model")
    two = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "model minus baseline on 6" in two and "units minus" not in two


def _exempt_file(path: Path, rows: list[dict[str, str]], rules: str | None = None) -> None:
    document = {
        "rules": rules or retag.code_revision(*REPORT.attribution_model.UNITS_CODE_MODULES),
        "companies": rows,
    }
    (path / gold_metrics.EXEMPT_NAME).write_text(json.dumps(document), encoding="utf-8")


def _read_file(path: Path, marks: dict[int, str], tickers: dict[int, str] | None = None) -> None:
    rows = [
        {
            "article_id": str(a),
            "ticker": (tickers or {}).get(a, "U"),
            "company": "U Inc.",
            "why": "guard",
            "extractors": "units",
            "title": "t",
            "text": "x",
            "spanish_title": "",
            "spanish_text": "",
            "real_company": m,
        }
        for a, m in marks.items()
    ]
    _csv(path / gold_metrics.EXEMPT_READ_NAME, rows, gold_metrics.EXEMPT_READ_COLUMNS)


def test_the_bar_is_also_given_on_every_company_named_and_on_the_real_ones(tmp_path: Path) -> None:
    _faith_files(tmp_path, [], extractors="baseline|model|units")
    plain = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "counted as unfaithful (the bar" not in plain  # no file: nothing to count
    _exempt_file(tmp_path, [_skip(1, "units"), _skip(2, "units")])
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "[all] units: the 2 companies it wrote no text for and is not asked about" in text
    assert "[2022+] units: the 2 companies" in text  # both blocks, the dates are 2023
    assert "75.0%" in text.split("(the bar on every company named)")[1]  # 6 faithful of 6 + 2
    assert "real companies" not in text  # the owner has not read them
    _read_file(tmp_path, {1: "yes", 2: ""})
    part = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "1 of 2 exempt companies are not yet read" in part and "[all, real]" not in part
    # 1#U is a company the extractor lost; 2#U is not a company; 3#T is not a company either,
    # and every extractor wrote a text about it: that text leaves every extractor's share
    rows = [_skip(1, "units"), _skip(2, "units"), {**_skip(3, "units"), "ticker": "T"}]
    _exempt_file(tmp_path, rows)
    _read_file(tmp_path, {1: "yes", 2: "no", 3: "no"}, tickers={3: "T"})
    done = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "2 of the 3 exempt companies are not companies" in done
    real = [line for line in done.splitlines() if line.startswith("[all, real]")]
    units = next(line for line in real if "units:" in line and "of the texts" not in line)
    assert "5 faithful of 5 texts + 1 companies left without one" in units  # 3#T#1 is gone
    assert "83.3%" in units
    base = next(line for line in real if "baseline:" in line and "of the texts" not in line)
    assert "5 faithful of 5 texts + 0 companies" in base and "100.0%" in base  # also lost 3#T#1
    # the texts written for the non-company are the extractor's errors, counted apart (informative)
    for name in ("baseline", "units"):
        wrote = [x for x in done.splitlines() if x.startswith(f"[all] {name}: 1 of the 6 texts")]
        assert wrote and "for a company that is not one" in wrote[0], done
    _exempt_file(tmp_path, [_skip(1, "units")], rules="another revision")
    other = "\n".join(REPORT.faithfulness_report(tmp_path))  # the report goes on without it
    assert "written by the rules 'another revision'" in other
    assert "the other readings of the bar are skipped" in other and "[all]   baseline:" in other


def test_the_exempt_reading_file_keeps_the_line_endings_the_spreadsheet_gave_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _csv(tmp_path / "faithfulness_bodies.csv", [{"article_id": "1", "segments": "1"}])
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    row = {
        "article_id": "1",
        "ticker": "BE",
        "company": "Bloom Energy",
        "why": "guard",
        "extractors": "units2",
        "title": "t",
        "text": "Michael Bloom said so.",
        "spanish_title": "x",
        "spanish_text": "y",
        "real_company": "no",
    }
    rows = [{"article_id": "1", "ticker": "BE", "extractor": "units2", "pub_date": "2023-01-01"}]
    monkeypatch.setattr(FAITH, "exempt_rows", lambda bodies, resolver: (rows, [dict(row)]))
    path = tmp_path / gold_metrics.EXEMPT_READ_NAME
    FAITH.write_exempt(tmp_path)  # a new file: the project's own line ending
    assert b"\r\n" not in path.read_bytes()
    text = gold_metrics.render_csv(gold_metrics.EXEMPT_READ_COLUMNS, [row], lineterminator="\r\n")
    path.write_bytes(b"\xef\xbb\xbf" + text.encode())  # saved by a spreadsheet: CRLF
    before = path.read_bytes()
    FAITH.write_exempt(tmp_path)  # drawn again with nothing new: the file is the same bytes
    assert path.read_bytes() == before


def test_a_not_a_company_mark_applies_when_only_other_extractors_were_not_asked_about_it(
    tmp_path: Path,
) -> None:
    """A confirmation draw has baseline and units only; the guard-rejected companies are exempt rows
    of units2 and units3, which did not run. The owner's "no" is about the company: it must leave
    the shares of the extractors that did run."""
    _faith_files(tmp_path, [], extractors="baseline|units")
    rows = [{**_skip(3, "units2"), "ticker": "T"}, {**_skip(4, "units3"), "ticker": "T"}]
    _exempt_file(tmp_path, rows)
    _read_file(tmp_path, {3: "no", 4: "yes"}, tickers={3: "T", 4: "T"})
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "1 of the 2 exempt companies are not companies" in text
    real = [line for line in text.splitlines() if line.startswith("[all, real]")]
    for name in ("baseline", "units"):
        line = next(x for x in real if f"{name}:" in x and "of the texts" not in x)
        assert "5 faithful of 5 texts" in line, line  # 3#T#1 is out of both shares, 4#T#1 stays
    _read_file(tmp_path, {3: "", 4: "yes"}, tickers={3: "T", 4: "T"})
    assert "1 of 2 exempt companies are not yet read" in "\n".join(
        REPORT.faithfulness_report(tmp_path)
    )


def test_a_company_both_extractors_skipped_is_in_the_shares_not_in_the_gain(
    tmp_path: Path,
) -> None:
    both = [_skip(a, e) for a in (20, 21) for e in ("baseline", "model")]
    _faith_files(tmp_path, both)
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "model minus baseline on 6 company texts" in text
    assert "6 faithful of 6 texts + 2 companies" in text


def test_the_faithfulness_report_refuses_a_coverage_file_the_draw_did_not_write(
    tmp_path: Path,
) -> None:
    _faith_files(tmp_path, [_skip(99, "model")])
    with pytest.raises(MarkedFileError, match="the draw does not have"):
        REPORT.faithfulness_report(tmp_path)
    _faith_files(tmp_path, [_skip(1, "gpt")])
    with pytest.raises(MarkedFileError, match="the draw does not have"):
        REPORT.faithfulness_report(tmp_path)


class _Resolver:
    def resolve(self, body: str, skip_attribution: bool = True) -> set[str]:
        return {"T", "U"}


def _source(monkeypatch: pytest.MonkeyPatch, articles: dict[int, str]) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("ATTACH ':memory:' AS source")
    conn.execute(
        "CREATE TABLE source.articles (id INTEGER, title TEXT, body_text TEXT, pub_date TEXT, "
        "source_domain TEXT)"
    )
    for i, body in articles.items():
        conn.execute("INSERT INTO source.articles VALUES (?, 't', ?, '2023-01-01', 'd')", (i, body))
    monkeypatch.setattr(FAITH.db, "connect_pipeline", lambda: conn)
    monkeypatch.setattr(SENTIMENT.db, "connect_pipeline", lambda: conn)


def test_the_draws_refuse_a_held_out_body_under_another_article_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(monkeypatch, {1: "Same body.", 2: "Same body.", 3: "Another body."})
    _csv(tmp_path / "heldout_ids.csv", [{"article_id": "2", "stratum": "single"}])
    gold = [{"article_id": "1", "scope": "multi"}, {"article_id": "3", "scope": "multi"}]
    _csv(tmp_path / "gold.csv", gold)
    monkeypatch.setattr(FAITH, "GOLD", tmp_path / "gold.csv")
    monkeypatch.setattr(SENTIMENT, "GOLD", tmp_path / "gold.csv")
    got = FAITH.candidate_bodies(tmp_path, _Resolver())  # type: ignore[arg-type]
    assert [c["article_id"] for c in got] == ["3"]
    with pytest.raises(SystemExit, match="1 held-out bodies"):
        SENTIMENT.tagged_bodies(tmp_path)


def test_a_confirmation_draw_leaves_out_the_bodies_of_the_first_draw_and_its_twins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(
        monkeypatch,
        {1: "Used body.", 2: "Used body.", 3: "Fresh one.", 4: "Held out.", 5: "Fresh two."},
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _csv(first / "heldout_ids.csv", [{"article_id": "4", "stratum": "single"}])
    _csv(first / "faithfulness_bodies.csv", [{"article_id": "1", "segments": "3"}])
    gold = [{"article_id": str(a), "scope": "multi"} for a in (1, 2, 3, 4, 5)]
    _csv(tmp_path / "gold.csv", gold)
    monkeypatch.setattr(FAITH, "GOLD", tmp_path / "gold.csv")
    plain = FAITH.candidate_bodies(first, _Resolver())  # type: ignore[arg-type]
    assert [c["article_id"] for c in plain] == ["1", "2", "3", "5"]  # the first draw's pool
    got = FAITH.candidate_bodies(second, _Resolver(), first)  # type: ignore[arg-type]
    # article 1 was drawn, article 2 has the same body, 4 is held out: only 3 and 5 are new
    assert [c["article_id"] for c in got] == ["3", "5"]


def test_collect_items_lists_the_companies_an_extractor_left_without_a_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source(monkeypatch, {1: "Body."})
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"extractor:{name}@abc")
    monkeypatch.setitem(FAITH.EXTRACTORS, "skipper", lambda body, resolver: {"T": "Text of T."})
    monkeypatch.setitem(
        FAITH.EXTRACTORS, "writer", lambda body, resolver: {"T": "Text of T.", "U": "Text of U."}
    )
    items, omissions = FAITH.collect_items(
        [{"article_id": "1", "segments": "1"}],
        ["skipper", "writer"],
        _Resolver(),  # type: ignore[arg-type]
    )
    assert omissions == [
        {"article_id": "1", "ticker": "U", "extractor": "skipper", "pub_date": "2023-01-01"}
    ]
    assert {i["ticker"] for i in items} == {"T", "U"}
    assert {i["revisions"] for i in items} == {
        "extractor:skipper@abc|extractor:writer@abc",
        "extractor:writer@abc",
    }


def test_the_units_extractor_is_not_asked_for_a_company_named_only_in_the_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Footer:
        def resolve(self, body: str, skip_attribution: bool = True) -> set[str]:
            return {"T"} | ({"U"} if "Disclosures" in body else set())

    story = "Company T had a good quarter and said so in a long statement to the press. " * 4
    _source(monkeypatch, {1: story + "\nDisclosures:\nCortes owns U"})
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"extractor:{name}@abc")
    monkeypatch.setitem(FAITH.EXTRACTORS, "units", lambda body, resolver: {"T": "Text of T."})
    monkeypatch.setitem(
        FAITH.EXTRACTORS, "other", lambda body, resolver: {"T": "Text of T.", "U": "Disclosures"}
    )
    _, omissions = FAITH.collect_items(
        [{"article_id": "1", "segments": "1"}],
        ["units", "other"],
        Footer(),  # type: ignore[arg-type]
    )
    assert omissions == []  # U is in the footer only: no omission for units, and "other" wrote one


def test_the_units2_extractor_is_not_asked_for_a_company_named_only_by_a_common_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from news_nlp.triage import Company, CompanyResolver  # noqa: PLC0415

    resolver = CompanyResolver(
        [
            Company("AAPL", "Apple Inc.", "Information Technology", "Hardware"),
            Company("BE", "Bloom Energy", "Industrials", "Electrical Equipment"),
        ]
    )
    story = "Apple had a good quarter, and CNBC's Michael Bloom said so in a long note. " * 4
    _source(monkeypatch, {1: story})
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"extractor:{name}@abc")
    monkeypatch.setitem(FAITH.EXTRACTORS, "units2", lambda body, resolver: {"AAPL": "Apple text."})
    _, omissions = FAITH.collect_items([{"article_id": "1", "segments": "1"}], ["units2"], resolver)
    assert omissions == []  # "Bloom" is a person here: Bloom Energy is not a company of this story


def test_exempt_rows_are_the_companies_named_only_in_the_footer_or_by_a_common_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from news_nlp.triage import Company, CompanyResolver  # noqa: PLC0415

    universe = [
        Company("AAPL", "Apple Inc.", "Information Technology", "Hardware"),
        Company("BE", "Bloom Energy", "Industrials", "Electrical Equipment"),
        Company("MSFT", "Microsoft Corporation", "Information Technology", "Software"),
    ]
    story = "Apple had a good quarter, and CNBC's Michael Bloom said so in a long note. " * 4
    body = story + "\nDisclosures:\nThe author owns Microsoft Corporation"
    _source(monkeypatch, {1: body})
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: universe)
    rows, reading = FAITH.exempt_rows(
        [{"article_id": "1", "segments": "1"}], CompanyResolver(universe)
    )
    by_company: dict[str, list[str]] = {}
    for row in rows:
        by_company.setdefault(row["ticker"], []).append(row["extractor"])
    # Microsoft is in the footer only: no units extractor is asked; Bloom is a person: not the guarded
    assert by_company == {"MSFT": ["units", "units2", "units3"], "BE": ["units2", "units3"]}
    why = {r["ticker"]: (r["why"], r["extractors"]) for r in reading}
    assert why == {"MSFT": ("footer", "units|units2|units3"), "BE": ("guard", "units2|units3")}
    line = next(r["text"] for r in reading if r["ticker"] == "BE")
    assert "Michael Bloom" in line and not any(r["spanish_text"] for r in reading)


def test_the_exempt_file_carries_the_rules_and_the_reading_is_validated(tmp_path: Path) -> None:
    path = tmp_path / gold_metrics.EXEMPT_NAME
    row = {"article_id": "1", "ticker": "U", "extractor": "units", "pub_date": "2023-01-01"}
    path.write_text(json.dumps({"rules": "abc", "companies": [row]}), encoding="utf-8")
    assert gold_metrics.read_exempt(path, "abc") == [row]
    with pytest.raises(gold_metrics.ExemptRulesChanged, match="written by the rules"):
        gold_metrics.read_exempt(path, "other")
    assert gold_metrics.read_exempt(tmp_path / "missing.json", "abc") == []
    path.write_text(json.dumps({"rules": "abc", "companies": [{"x": 1}]}), encoding="utf-8")
    with pytest.raises(gold_metrics.MarkedFileError, match="columns"):
        gold_metrics.read_exempt(path, "abc")
    assert gold_metrics.read_exempt_marks(tmp_path / "none.csv") == {}
    _read_file(tmp_path, {1: "Yes", 2: "no", 3: ""})
    marks = gold_metrics.read_exempt_marks(tmp_path / gold_metrics.EXEMPT_READ_NAME)
    assert marks == {"1#U": "yes", "2#U": "no", "3#U": ""}
    path = tmp_path / gold_metrics.EXEMPT_READ_NAME  # the owner's own notes column is allowed
    header = [*reversed(gold_metrics.EXEMPT_READ_COLUMNS), "notes"]
    path.write_text(
        ",".join(header)
        + "\n"
        + ",".join("yes" if c == "real_company" else "x" for c in header)
        + "\n",
        encoding="utf-8",
    )
    assert gold_metrics.read_exempt_marks(path) == {"x#x": "yes"}
    path.write_text(
        ",".join([*header, "other"]) + "\n", encoding="utf-8"
    )  # any other column is not
    with pytest.raises(gold_metrics.MarkedFileError, match="columns"):
        gold_metrics.read_exempt_marks(path)
    _read_file(tmp_path, {1: "maybe"})
    with pytest.raises(gold_metrics.MarkedFileError, match="yes or no"):
        gold_metrics.read_exempt_marks(tmp_path / gold_metrics.EXEMPT_READ_NAME)


def _stale_ids(*rows: tuple[str, str, str, str]) -> list[dict[str, str]]:
    """(key, text, extractors, revisions) as the ids file holds them."""
    out = []
    for key, text, extractors, revisions in rows:
        article, ticker, _ = key.split("#")
        out.append(
            {
                "key": key,
                "article_id": article,
                "ticker": ticker,
                "text_hash": retag.text_hash(text),
                "extractors": extractors,
                "revisions": revisions,
            }
        )
    return out


def _stale_item(key: str, text: str, extractors: str) -> dict[str, str]:
    article, ticker, _ = key.split("#")
    return {
        "key": key,
        "article_id": article,
        "ticker": ticker,
        "text_hash": retag.text_hash(text),
        "extractors": extractors,
    }


def test_stale_lists_the_marked_rows_the_code_no_longer_writes_and_the_texts_it_adds() -> None:
    ids = _stale_ids(
        ("1#T#1", "same", "baseline|units", "b1|u1"),
        ("2#T#1", "old text", "units", "u1"),
        ("3#T#1", "gone", "units", "u1"),
    )
    items = [
        _stale_item("1#T#1", "same", "baseline|units"),
        _stale_item("2#T#2", "new text", "units"),  # units now writes another text for article 2
    ]  # and nothing for article 3
    got = FAITH.stale_report(ids, items, ["baseline", "units"], {"baseline": "b1", "units": "u2"})
    assert got["stale"] == ["2#T#1 (units)", "3#T#1 (units)"]
    assert got["new"] == ["2#T#2 (units)"]
    assert got["revision"] == ["1#T#1 (units)", "2#T#1 (units)", "3#T#1 (units)"]  # u1 is not u2


def test_stale_ignores_extractors_that_were_not_run() -> None:
    ids = _stale_ids(("1#T#1", "text", "baseline|units3", "b1|u3"))
    got = FAITH.stale_report(
        ids, [_stale_item("1#T#1", "text", "baseline")], ["baseline"], {"baseline": "b1"}
    )
    assert got == {"stale": [], "new": [], "revision": []}


def _stale_file(path: Path, **entries: dict[str, object]) -> None:
    """A staleness result: by default the baseline checked now and clean."""
    document: dict[str, dict[str, object]] = {
        "baseline": {
            "date": "2026-11-08",
            "revision": retag.extractor_revision("baseline"),
            "stale": [],
            "new": [],
        }
    }
    document.update(entries)
    (path / gold_metrics.STALE_NAME).write_text(
        json.dumps({"extractors": document}), encoding="utf-8"
    )


def _units_entry(code: str, stale: list[str] | None = None) -> dict[str, object]:
    return {
        "date": "2026-11-09",
        "revision": f"extractor:units@local:x@names:y@t0.5@{code}",
        "stale": stale or [],
        "new": [],
    }


def test_the_report_says_what_the_staleness_check_found_per_extractor(tmp_path: Path) -> None:
    _faith_files(tmp_path, [], extractors="baseline|units")
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "no staleness check recorded: run draw_faithfulness --stale" in text
    _stale_file(tmp_path)
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "no marked row is on a text the current code no longer writes" in text
    assert "never checked: units" in text  # the marks have units rows, the file has no entry
    units = retag.code_revision(*REPORT.attribution_model.UNITS_CODE_MODULES)
    rows = [f"{n}#T#1 (units)" for n in range(8)]
    _stale_file(tmp_path, units=_units_entry(units, rows))
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "8 marked rows are on texts the current code no longer writes" in text
    assert "0#T#1 (units), 1#T#1 (units)" in text and ", ...)" in text  # six shown
    assert "never checked" not in text and "warning" not in text


def test_a_check_made_with_other_code_is_said_for_the_baseline_the_model_and_units(
    tmp_path: Path,
) -> None:
    _faith_files(tmp_path, [], extractors="baseline|model|units")
    model = retag.code_revision(*REPORT.attribution_model.CODE_MODULES)
    entry = _units_entry("0123456789ab")  # another hash: the footer rules changed since the check
    _stale_file(
        tmp_path,
        units=entry,
        model={**entry, "revision": f"extractor:model@local:x@names:y@t0.5@{model}"},
    )
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "warning: the staleness check of units (2026-11-09) was made with other code" in text
    assert "check of model" not in text and "check of baseline" not in text  # those are current
    _stale_file(tmp_path, baseline={**entry, "revision": "extractor:baseline@old"})
    assert "check of baseline" in "\n".join(REPORT.faithfulness_report(tmp_path))


def test_a_run_of_some_extractors_keeps_the_last_check_of_the_others(tmp_path: Path) -> None:
    path = tmp_path / gold_metrics.STALE_NAME
    units = gold_metrics.StaleEntry(date="d1", revision="r", stale=["1#T#1 (units)"], new=[])
    baseline = gold_metrics.StaleEntry(date="d2", revision="r", stale=[], new=[])
    path.write_text(json.dumps(gold_metrics.merge_stale({}, {"units": units})), encoding="utf-8")
    again = gold_metrics.merge_stale(gold_metrics.read_stale(path), {"baseline": baseline})
    path.write_text(json.dumps(again), encoding="utf-8")
    kept = gold_metrics.read_stale(path)
    assert kept["units"]["stale"] == ["1#T#1 (units)"] and kept["baseline"]["date"] == "d2"
    path.write_text(json.dumps({"date": "old format"}), encoding="utf-8")
    with pytest.raises(MarkedFileError, match="not in the current format"):
        gold_metrics.read_stale(path)
    path.write_text(json.dumps({"extractors": {"units": {"date": 1}}}), encoding="utf-8")
    with pytest.raises(MarkedFileError, match="malformed"):
        gold_metrics.read_stale(path)


def test_stale_exits_with_a_sum_of_what_is_stale_and_what_was_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _csv(tmp_path / "faithfulness_bodies.csv", [{"article_id": "1", "segments": "1"}])
    ids = _stale_ids(
        ("1#T#1", "text", "baseline", "r1"),
        ("1#U#1", "other", "baseline|model", "r1|r2"),
    )
    _csv(tmp_path / "faithfulness_ids.csv", ids)
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"rev-{name}")
    monkeypatch.setattr(
        FAITH, "EXTRACTORS", {"baseline": lambda body, resolver: {}, "model": lambda b, r: {}}
    )

    def collect(writes: list[dict[str, str]]):  # type: ignore[no-untyped-def]
        return lambda bodies, names, resolver: (writes, [])

    both = [_stale_item("1#T#1", "text", "baseline"), _stale_item("1#U#1", "other", "baseline")]
    args = types.SimpleNamespace(out=tmp_path, only=None)
    monkeypatch.setattr(FAITH, "collect_items", collect([both[0]]))
    # baseline and model are run, U is no longer written by the baseline: stale (1); the model
    # wrote nothing for U either (stale too); nothing left unchecked
    assert FAITH.run_stale(args) == 1
    saved = gold_metrics.read_stale(tmp_path / gold_metrics.STALE_NAME)
    assert saved["baseline"]["stale"] == ["1#U#1 (baseline)"] and saved["model"]["stale"]
    # an extractor in the ids that is not registered is unchecked (2); with stale rows, 3
    _csv(tmp_path / "faithfulness_ids.csv", [*ids, *_stale_ids(("1#V#1", "v", "units9", "r"))])
    assert FAITH.run_stale(args) == 3
    # --only limits the check to the extractors asked for: the others are left out, not unchecked
    only = types.SimpleNamespace(out=tmp_path, only="baseline")
    monkeypatch.setattr(FAITH, "collect_items", collect(both))
    _csv(
        tmp_path / "faithfulness_ids.csv",
        _stale_ids(("1#T#1", "text", "baseline", "r1"), ("1#V#1", "v", "units9", "r")),
    )
    assert FAITH.run_stale(only) == 0
    with pytest.raises(SystemExit, match="not registered"):
        FAITH.run_stale(types.SimpleNamespace(out=tmp_path, only="nope"))
    # the earlier result of the model is still in the file after a run of the baseline alone
    assert "model" in gold_metrics.read_stale(tmp_path / gold_metrics.STALE_NAME)


def test_a_scored_pool_is_refused_under_other_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retag, "extractor_revision", lambda name: f"extractor:{name}@abc")
    _csv(tmp_path / "gold.csv", [{"article_id": "1", "scope": "multi"}])
    config = tmp_path / "models.json"
    models = {s: {"model": f"org/{s}", "revision": "r1"} for s in ("sentiment", "c_summary", "ner")}
    config.write_text(json.dumps(models), encoding="utf-8")
    monkeypatch.setattr(SENTIMENT, "GOLD", tmp_path / "gold.csv")
    monkeypatch.setattr(SENTIMENT, "CONFIG", config)
    _csv(
        tmp_path / SENTIMENT.POOL_CSV_NAME,
        [{"article_id": "1", "kind": "summary", "predicted": "positive"}],
    )
    (tmp_path / SENTIMENT.POOL_META_NAME).write_text(
        json.dumps(SENTIMENT.pool_conditions("baseline")), encoding="utf-8"
    )
    assert SENTIMENT.scored_pool(tmp_path, "baseline")[0]["kind"] == "summary"
    with pytest.raises(SystemExit, match="other conditions"):
        SENTIMENT.scored_pool(tmp_path, "model")
    models["sentiment"]["revision"] = "r2"
    config.write_text(json.dumps(models), encoding="utf-8")
    with pytest.raises(SystemExit, match="sentiment"):
        SENTIMENT.scored_pool(tmp_path, "baseline")
    (tmp_path / SENTIMENT.POOL_META_NAME).unlink()
    with pytest.raises(SystemExit, match="other conditions"):
        SENTIMENT.scored_pool(tmp_path, "baseline")


def test_the_report_names_the_missing_file_and_marks_a_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["report", "scope", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match=r"heldout_to_tag\.csv not found"):
        REPORT.main()
    ids = [{"key": f"{i}#T", "predicted": "negative", "population": "3"} for i in range(3)]
    tags = [
        {
            "article_id": str(i),
            "ticker": "T",
            "kind": "summary",
            "sentiment": "negative",
            "pub_date": "2023-01-01",
        }
        for i in range(3)
    ]
    _csv(tmp_path / "sentiment_gold_ids.csv", ids)
    _csv(tmp_path / "sentiment_gold_to_tag.csv", tags)
    assert "(census, not resampled" in "\n".join(REPORT.sentiment_report(tmp_path))


def test_a_header_only_scaffold_is_not_the_owners_tagging_file(tmp_path: Path) -> None:
    extra = _script("draw_sector_extra_2026_11_05")
    path = tmp_path / "sentiment_gold_to_tag.csv"
    assert not extra.has_rows(path)  # no file
    path.write_text("﻿review_order,article_id,sentiment\n", encoding="utf-8")
    assert not extra.has_rows(path)  # the scaffold: columns, no rows
    _csv(path, [{"review_order": "1", "article_id": "7", "sentiment": ""}])
    assert extra.has_rows(path)  # one row is the owner's work, even untagged


ROOT = Path(__file__).resolve().parents[2]


def test_the_committed_gold_files_have_the_columns_the_code_defines() -> None:
    expected = {
        "category_gold_v2.csv": retag.GOLD_COLUMNS,
        "category_gold_v2_test.csv": retag.TEST_GOLD_COLUMNS,
        "sentiment_gold.csv": retag.SENTIMENT_GOLD_COLUMNS,
        "sentiment_gold_scoring_log.csv": retag.SENTIMENT_SCORING_LOG_COLUMNS,
        "faithfulness_gold.csv": retag.FAITHFULNESS_GOLD_COLUMNS,
        "segment_check_gold.csv": retag.SEGMENT_CHECK_GOLD_COLUMNS,
    }
    for name, columns in expected.items():
        with (ROOT / "gold" / name).open(encoding="utf-8-sig", newline="") as fh:
            assert tuple(next(csv.reader(fh))) == columns, name


def test_the_training_file_has_its_columns_once_it_is_committed() -> None:
    """``gold/sentiment_train.csv`` (D35) is written by the builder after the owner's marks, so it
    is not there before them; when it is, its columns are the ones ``retag`` defines. It is not
    scaffolded: it has a draw and the builder creates it."""
    path = ROOT / "gold" / "sentiment_train.csv"
    if not path.exists():
        pytest.skip("the owner's marks are not built into gold/ yet")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        assert tuple(next(csv.reader(fh))) == retag.SENTIMENT_TRAIN_COLUMNS


def test_the_scaffold_writes_the_draw_columns_and_never_touches_a_file(tmp_path: Path) -> None:
    scaffold = _script("scaffold_gold_sets_2026_11_06")
    out, gold = tmp_path / "data", tmp_path / "gold"
    out.mkdir()
    (out / "sentiment_gold_to_tag.csv").write_text("owner,work\n1,2\n", encoding="utf-8")
    for work, work_cols, gold_name, gold_cols in scaffold.SETS.values():
        for path, cols in ((out / work, work_cols), (gold / gold_name, gold_cols)):
            scaffold.scaffold(path, cols)
    assert (out / "sentiment_gold_to_tag.csv").read_text(encoding="utf-8") == "owner,work\n1,2\n"
    with (out / "faithfulness_to_tag.csv").open(encoding="utf-8-sig", newline="") as fh:
        assert tuple(next(csv.reader(fh))) == retag.FAITHFULNESS_COLUMNS
    with (gold / "segment_check_gold.csv").open(encoding="utf-8-sig", newline="") as fh:
        assert tuple(next(csv.reader(fh))) == retag.SEGMENT_CHECK_GOLD_COLUMNS


class _Reached(Exception):
    """The draw went past the check on the owner's file."""


@pytest.mark.parametrize(
    ("script", "step"),
    [
        ("draw_sentiment_gold_2026_11_05", "build_texts"),
        ("draw_faithfulness_2026_11_06", "candidate_bodies"),
        ("draw_segment_check_2026_11_05", "pick"),
    ],
)
def test_a_draw_fills_a_header_only_file_and_stops_at_one_with_rows(
    script: str, step: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script(script)
    names = {
        "draw_sentiment_gold_2026_11_05": "sentiment_gold_to_tag.csv",
        "draw_faithfulness_2026_11_06": "faithfulness_to_tag.csv",
        "draw_segment_check_2026_11_05": "segment_check_to_tag.csv",
    }
    target = tmp_path / names[script]
    if script == "draw_segment_check_2026_11_05":
        _csv(tmp_path / "gold.csv", [{"article_id": "1", "split": "train", "kind": "lead"}])
        monkeypatch.setattr(module, "GOLD", tmp_path / "gold.csv")
    if script == "draw_faithfulness_2026_11_06":
        monkeypatch.setattr(module.retag_cli, "companies", lambda: [])
        monkeypatch.setattr(module, "CompanyResolver", lambda _u: None)

    def reached(*_a: object, **_k: object) -> None:
        raise _Reached

    monkeypatch.setattr(module, step, reached)
    argv = [script, "--out", str(tmp_path)]
    if script == "draw_faithfulness_2026_11_06":
        argv += ["--extractors", "baseline"]  # the model does not exist until T-148b3
    monkeypatch.setattr(sys, "argv", argv)
    target.write_text(",".join(retag.TAGGING_COLUMNS) + "\n", encoding="utf-8-sig")
    with pytest.raises(_Reached):  # the scaffold is not the owner's work
        module.main()
    _csv(target, [{"article_id": "1", "scope": "single"}])
    with pytest.raises(SystemExit, match="is your tagging file"):
        module.main()


def test_an_interrupted_write_leaves_the_target_untouched_and_no_temporary_file(
    tmp_path: Path,
) -> None:
    extra = _script("draw_sector_extra_2026_11_05")
    target = tmp_path / "x_to_tag.csv"
    target.write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not in fieldnames"):  # the second row has another key
        extra._write(target, [{"a": "2"}, {"b": "bad key"}])
    assert target.read_text(encoding="utf-8") == "a\n1\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_a_text_hash_ignores_outer_whitespace_and_tells_texts_apart() -> None:
    assert retag.text_hash(" One.\n") == retag.text_hash("One.")
    assert retag.text_hash("One.") != retag.text_hash("Two.")
    assert len(retag.text_hash("One.")) == 64


def test_the_sentiment_draw_writes_the_hash_of_each_text() -> None:
    got = SENTIMENT.draw(_texts(), 2, 1)
    assert all(t["text_hash"] == retag.text_hash(t["text"]) for t in got)
    assert {"text_hash", "revision"} <= set(SENTIMENT.IDS_FIELDS)
    assert {"text_hash", "revision"} <= set(retag.SENTIMENT_GOLD_COLUMNS)


def test_the_faithfulness_ids_carry_the_hash_the_gold_is_keyed_by() -> None:
    assert "text_hash" in FAITH.IDS_FIELDS
    assert {"article_id", "ticker", "text_hash"} <= set(retag.FAITHFULNESS_GOLD_COLUMNS)


def test_the_pull_of_the_test_set_names_a_stratum_without_a_frame_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_test_set(tmp_path, monkeypatch, tagged=[1, 2], fixed=[1, 2])
    _csv(tmp_path / "data" / "heldout_strata.csv", [{"stratum": "multi", "frame": "93"}])
    with pytest.raises(SystemExit, match="lacks the frame size"):
        BUILD_GOLD.main()


def test_a_rerun_after_an_interrupted_draw_reuses_the_ids_and_fills_the_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("draw_segment_check_2026_11_05")
    _csv(tmp_path / module.IDS_NAME, [{f: "1" for f in module.IDS_FIELDS}])  # drawn, then stopped
    (tmp_path / module.TO_TAG_NAME).write_text(
        ",".join(retag.SEGMENT_CHECK_COLUMNS) + "\n", encoding="utf-8-sig"
    )

    def stop(*_a: object, **_k: object) -> None:
        raise _Reached

    monkeypatch.setattr(module, "texts_to_translate", stop)
    monkeypatch.setattr(sys, "argv", ["draw", "--out", str(tmp_path)])
    with pytest.raises(_Reached):  # it read the ids it had and went on to the translation
        module.main()


def test_a_scored_pool_read_back_keeps_the_revision_of_each_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _csv(tmp_path / "gold.csv", [{"article_id": "1", "scope": "multi"}])
    config = tmp_path / "models.json"
    models = {s: {"model": f"org/{s}", "revision": "r1"} for s in ("sentiment", "c_summary", "ner")}
    config.write_text(json.dumps(models), encoding="utf-8")
    monkeypatch.setattr(SENTIMENT, "GOLD", tmp_path / "gold.csv")
    monkeypatch.setattr(SENTIMENT, "CONFIG", config)
    text = {
        "article_id": "1",
        "kind": "summary",
        "extractor": "",
        "revision": "org/c_summary@r1",
        "scope": "single",
        "ticker": "T",
        "company": "T Inc",
        "title": "t",
        "text": "a summary",
    }
    monkeypatch.setattr(SENTIMENT, "build_texts", lambda *_a: ([{**text}], Counter()))
    monkeypatch.setattr(
        SENTIMENT, "predict", lambda texts: [{**t, "predicted": "positive"} for t in texts]
    )
    SENTIMENT.scored_pool(tmp_path, "baseline")  # scored and written
    again = SENTIMENT.scored_pool(tmp_path, "baseline")  # read back
    assert again[0]["revision"] == "org/c_summary@r1"


def _fake_module(tmp_path: Path, name: str, source: str) -> types.ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    return module


def test_a_code_revision_is_stable_ignores_order_and_changes_with_any_module(
    tmp_path: Path,
) -> None:
    one = _fake_module(tmp_path, "one", "A = 1\n")
    two = _fake_module(tmp_path, "two", "B = 1\n")
    first = retag.code_revision(one, two)
    assert len(first) == 12 and first == retag.code_revision(one, two)
    assert first == retag.code_revision(two, one)  # by module name, not by position
    Path(str(two.__file__)).write_text("B = 2\n", encoding="utf-8")  # a dependency changes
    assert retag.code_revision(one, two) != first
    assert retag.code_revision(one) != retag.code_revision(one, two)


def test_the_real_revisions_are_stable_and_twelve_hex_digits() -> None:
    for value in (retag.parser_revision(), retag.extractor_revision("baseline")):
        assert re.fullmatch(r"(company_texts|extractor:baseline)@[0-9a-f]{12}", value)
    assert retag.parser_revision() == retag.parser_revision()
    with pytest.raises(ValueError, match="only 'baseline' exists"):
        retag.extractor_revision("model")


def test_each_kind_of_sentiment_text_names_how_it_was_made() -> None:
    summariser = {"model": "org/sum", "revision": "r9"}
    assert SENTIMENT.text_revision("summary", "baseline", summariser) == "org/sum@r9"
    assert (
        SENTIMENT.text_revision("company_text", "baseline", summariser) == retag.parser_revision()
    )
    assert SENTIMENT.text_revision(
        "extractor_text", "baseline", summariser
    ) == retag.extractor_revision("baseline")


def test_a_pool_is_refused_only_for_a_kind_whose_code_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = {"article_id": "1", "kind": "summary", "revision": "org/c_summary@r1"}
    company = {
        "article_id": "2",
        "kind": "company_text",
        "revision": retag.parser_revision(),
    }
    assert SENTIMENT.stale_kinds([summary, company], "baseline") == {}
    monkeypatch.setattr(retag, "parser_revision", lambda: "company_texts@000000000000")
    stale = SENTIMENT.stale_kinds([summary, company], "baseline")
    assert list(stale) == ["company_text"]  # the summary does not depend on the parser
    assert SENTIMENT.stale_kinds([summary], "baseline") == {}  # no such kind in the pool: fine


def test_scored_pool_stops_when_the_parser_changed_since_it_was_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _csv(tmp_path / "gold.csv", [{"article_id": "1", "scope": "multi"}])
    config = tmp_path / "models.json"
    models = {s: {"model": f"org/{s}", "revision": "r1"} for s in ("sentiment", "c_summary", "ner")}
    config.write_text(json.dumps(models), encoding="utf-8")
    monkeypatch.setattr(SENTIMENT, "GOLD", tmp_path / "gold.csv")
    monkeypatch.setattr(SENTIMENT, "CONFIG", config)
    _csv(
        tmp_path / SENTIMENT.POOL_CSV_NAME,
        [{"article_id": "1", "kind": "company_text", "revision": retag.parser_revision()}],
    )
    (tmp_path / SENTIMENT.POOL_META_NAME).write_text(
        json.dumps(SENTIMENT.pool_conditions("baseline")), encoding="utf-8"
    )
    assert SENTIMENT.scored_pool(tmp_path, "baseline")  # same code: read back
    monkeypatch.setattr(retag, "parser_revision", lambda: "company_texts@000000000000")
    with pytest.raises(SystemExit, match="company_text"):
        SENTIMENT.scored_pool(tmp_path, "baseline")


def test_the_faithfulness_ids_and_gold_carry_the_revision_of_each_extractor() -> None:
    assert "revisions" in FAITH.IDS_FIELDS
    assert "revisions" in retag.FAITHFULNESS_GOLD_COLUMNS


def test_the_label_check_gold_keeps_the_year_for_the_2022_split() -> None:
    assert "year" in retag.SEGMENT_CHECK_GOLD_COLUMNS
    assert "year" in _script("draw_segment_check_2026_11_05").IDS_FIELDS


def _ids_row(order: int, key: str, text_hash: str, names: str = "baseline") -> dict[str, str]:
    article, ticker, _n = key.rsplit("#", 2)
    return {
        "review_order": str(order),
        "key": key,
        "article_id": article,
        "ticker": ticker,
        "text_hash": text_hash,
        "extractors": names,
        "revisions": "|".join(f"extractor:{n}@r" for n in names.split("|")),
        "pub_date": "2023-01-01",
    }


def _item(article: str, ticker: str, text: str, name: str = "model") -> dict[str, str]:
    return {
        **_ids_row(0, f"{article}#{ticker}#1", retag.text_hash(text), name),
        "title": "t",
        "text": text,
        "source_domain": "d",
        "body": "b",
    }


def test_adding_an_extractor_marks_identical_texts_and_appends_the_others() -> None:
    ids = [
        _ids_row(1, "1#T#1", retag.text_hash("Same.")),
        _ids_row(2, "1#U#1", retag.text_hash("Own.")),
    ]
    items = [
        _item("1", "T", "Same."),
        _item("1", "U", "Different."),
        _item("2", "T", "Other body."),
    ]
    merged, new = FAITH.merge_new_extractor(ids, items, seed=1)
    assert merged[0]["extractors"] == "baseline|model"
    assert merged[0]["revisions"] == "extractor:baseline@r|extractor:model@r"
    assert merged[1]["extractors"] == "baseline"  # its text was not written by the model
    assert [r["review_order"] for r in merged[:2]] == ["1", "2"]  # earlier rows are not moved
    assert sorted(i["review_order"] for i in new) == ["3", "4"]
    keys = {i["key"] for i in new}
    assert keys == {"1#U#2", "2#T#1"}  # past the numbers in use for the same article and company
    assert {r["extractors"] for r in merged[2:]} == {"model"}


def test_adding_an_extractor_keeps_every_mark_and_saves_a_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path
    cols = list(retag.FAITHFULNESS_COLUMNS)
    owner = [
        {
            **dict.fromkeys(cols, "x"),
            "review_order": "1",
            "article_id": "1",
            "ticker": "T",
            "faithful": "yes",
        },
        {
            **dict.fromkeys(cols, "x"),
            "review_order": "2",
            "article_id": "1",
            "ticker": "U",
            "faithful": "no",
        },
    ]
    _csv(out / FAITH.TO_TAG_NAME, owner)
    _csv(
        out / FAITH.IDS_NAME,
        [_ids_row(1, "1#T#1", retag.text_hash("Same.")), _ids_row(2, "1#U#1", "h2")],
    )
    _csv(out / FAITH.BODIES_NAME, [{"article_id": "1", "segments": "1"}])
    _csv(out / FAITH.COVERAGE_NAME, [], FAITH.COVERAGE_FIELDS)
    monkeypatch.setitem(FAITH.EXTRACTORS, "model", lambda body, resolver: {})
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"extractor:{name}@r")
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    monkeypatch.setattr(FAITH, "CompanyResolver", lambda _u: None)
    monkeypatch.setattr(
        FAITH,
        "collect_items",
        lambda *_a: ([_item("1", "T", "Same."), _item("1", "T", "A new text.")], []),
    )
    monkeypatch.setattr(FAITH.retag_cli, "_translator", lambda: None)
    monkeypatch.setattr(
        FAITH.retag_cli,
        "add_spanish",
        lambda _c, texts, *_a, **_k: [
            {"article_id": t["article_id"], "spanish_title": "es", "spanish_text": "es"}
            for t in texts
        ],
    )
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(out), "--add-extractor", "model"])
    FAITH.main()
    rows = list(csv.DictReader((out / FAITH.TO_TAG_NAME).open(encoding="utf-8-sig", newline="")))
    assert [(r["review_order"], r["faithful"]) for r in rows] == [
        ("1", "yes"),
        ("2", "no"),
        ("3", ""),
    ]
    assert rows[2]["text"] == "A new text."
    backup = out / "faithfulness_to_tag.before_model.csv"
    assert [r["faithful"] for r in csv.DictReader(backup.open(encoding="utf-8-sig"))] == [
        "yes",
        "no",
    ]
    ids = list(csv.DictReader((out / FAITH.IDS_NAME).open(encoding="utf-8-sig", newline="")))
    assert [r["extractors"] for r in ids] == ["baseline|model", "baseline", "model"]
    with pytest.raises(SystemExit, match="interrupted"):  # a second run does not overwrite the copy
        FAITH.main()


def test_a_failed_add_leaves_the_three_files_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path
    cols = list(retag.FAITHFULNESS_COLUMNS)
    owner = [
        {
            **dict.fromkeys(cols, "x"),
            "review_order": "1",
            "article_id": "1",
            "ticker": "T",
            "faithful": "yes",
        }
    ]
    _csv(out / FAITH.TO_TAG_NAME, owner)
    _csv(out / FAITH.IDS_NAME, [_ids_row(1, "1#T#1", "h1")])
    _csv(out / FAITH.BODIES_NAME, [{"article_id": "1", "segments": "1"}])
    _csv(out / FAITH.COVERAGE_NAME, [], FAITH.COVERAGE_FIELDS)
    before = {
        n: (out / n).read_bytes() for n in (FAITH.TO_TAG_NAME, FAITH.IDS_NAME, FAITH.COVERAGE_NAME)
    }
    monkeypatch.setitem(FAITH.EXTRACTORS, "model", lambda body, resolver: {})
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"extractor:{name}@r")
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    monkeypatch.setattr(FAITH, "CompanyResolver", lambda _u: None)
    monkeypatch.setattr(FAITH, "collect_items", lambda *_a: ([_item("1", "T", "A new text.")], []))
    monkeypatch.setattr(FAITH.retag_cli, "_translator", lambda: None)
    monkeypatch.setattr(
        FAITH.retag_cli,
        "add_spanish",
        lambda _c, texts, *_a, **_k: [
            {"article_id": t["article_id"], "spanish_title": "es", "spanish_text": "es"}
            for t in texts
        ],
    )

    real_write = FAITH.write_coverage
    calls: list[int] = []

    def locked_once(*a: Path | list[dict[str, str]]) -> None:
        calls.append(1)
        if len(calls) == 1:  # the forward write fails, the undo goes through
            raise PermissionError("coverage is open in Excel")
        real_write(*a)  # type: ignore[arg-type]

    monkeypatch.setattr(FAITH, "write_coverage", locked_once)
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(out), "--add-extractor", "model"])
    with pytest.raises(SystemExit, match="nothing was added"):
        FAITH.main()
    after = {n: (out / n).read_bytes() for n in before}
    assert {n: b for n, b in after.items() if n != FAITH.COVERAGE_NAME} == {
        n: b for n, b in before.items() if n != FAITH.COVERAGE_NAME
    }
    # the undo writes the coverage file in the one form the reader and the builder use (no BOM, LF)
    assert after[FAITH.COVERAGE_NAME].decode("utf-8-sig").split() == (
        before[FAITH.COVERAGE_NAME].decode("utf-8-sig").split()
    )
    assert not (out / "faithfulness_to_tag.before_model.csv").exists()


def test_an_add_that_cannot_undo_keeps_the_copy_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path
    cols = list(retag.FAITHFULNESS_COLUMNS)
    owner = [
        {
            **dict.fromkeys(cols, "x"),
            "review_order": "1",
            "article_id": "1",
            "ticker": "T",
            "faithful": "yes",
        }
    ]
    _csv(out / FAITH.TO_TAG_NAME, owner)
    _csv(out / FAITH.IDS_NAME, [_ids_row(1, "1#T#1", "h1")])
    _csv(out / FAITH.BODIES_NAME, [{"article_id": "1", "segments": "1"}])
    _csv(out / FAITH.COVERAGE_NAME, [], FAITH.COVERAGE_FIELDS)
    monkeypatch.setitem(FAITH.EXTRACTORS, "model", lambda body, resolver: {})
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"extractor:{name}@r")
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    monkeypatch.setattr(FAITH, "CompanyResolver", lambda _u: None)
    monkeypatch.setattr(FAITH, "collect_items", lambda *_a: ([_item("1", "T", "A new text.")], []))
    monkeypatch.setattr(FAITH.retag_cli, "_translator", lambda: None)
    monkeypatch.setattr(
        FAITH.retag_cli,
        "add_spanish",
        lambda _c, texts, *_a, **_k: [
            {"article_id": t["article_id"], "spanish_title": "es", "spanish_text": "es"}
            for t in texts
        ],
    )

    def locked(*_a: object) -> None:
        raise PermissionError("coverage is open in Excel")

    monkeypatch.setattr(FAITH, "write_coverage", locked)
    real_copy = FAITH.shutil.copy2
    copies: list[int] = []

    def copy_once(src: Path, dst: Path) -> object:
        copies.append(1)
        if len(copies) > 1:  # the first copy is the backup; restoring it fails
            raise PermissionError("target is open in Excel")
        return real_copy(src, dst)

    monkeypatch.setattr(FAITH.shutil, "copy2", copy_once)
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(out), "--add-extractor", "model"])
    with pytest.raises(SystemExit, match="could not undo"):
        FAITH.main()
    assert (out / "faithfulness_to_tag.before_model.csv").exists()


def test_adding_an_extractor_needs_a_first_draw_and_an_extractor_that_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(tmp_path), "--add-extractor", "model"])
    with pytest.raises(SystemExit, match="no extractor 'model'"):
        FAITH.main()
    monkeypatch.setitem(FAITH.EXTRACTORS, "model", lambda body, resolver: {})
    with pytest.raises(SystemExit, match=r"needs faithfulness_to_tag\.csv"):
        FAITH.main()


def _segment_files(out: Path, correct: tuple[str, str] = ("yes", "no")) -> None:
    ids = [
        {
            "key": "1#2",
            "article_id": "1",
            "segment_index": "2",
            "ticker": "T",
            "kind": "lead",
            "year": "2020",
        },
        {
            "key": "1#5",
            "article_id": "1",
            "segment_index": "5",
            "ticker": "T",
            "kind": "lead",
            "year": "2020",
        },
    ]
    _csv(out / SEGGOLD.IDS_NAME, ids)
    marks = [
        {
            "review_order": str(i),
            "article_id": "1",
            "ticker": "T",
            "kind": "lead",
            "pub_date": "2020-01-01",
            "correct": c,
            "notes": "",
        }
        for i, c in enumerate(correct, 1)
    ]
    _csv(out / SEGGOLD.TO_TAG_NAME, marks)


def test_segment_check_gold_pairs_two_paragraphs_of_one_article_and_company(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _segment_files(tmp_path)
    gold = tmp_path / "gold.csv"
    monkeypatch.setattr(sys, "argv", ["b", "--out", str(tmp_path), "--gold", str(gold)])
    SEGGOLD.main()
    rows = list(csv.DictReader(gold.open(encoding="utf-8")))
    assert tuple(rows[0]) == retag.SEGMENT_CHECK_GOLD_COLUMNS
    assert [(r["key"], r["segment_index"], r["correct"]) for r in rows] == [
        ("1#2", "2", "yes"),
        ("1#5", "5", "no"),
    ]


def test_segment_check_gold_refuses_unmarked_rows_and_other_paragraphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _segment_files(tmp_path, ("yes", ""))
    monkeypatch.setattr(
        sys, "argv", ["b", "--out", str(tmp_path), "--gold", str(tmp_path / "g.csv")]
    )
    with pytest.raises(SystemExit, match="without correct"):
        SEGGOLD.main()
    assert not (tmp_path / "g.csv").exists()
    _segment_files(tmp_path)
    ids = list(csv.DictReader((tmp_path / SEGGOLD.IDS_NAME).open(encoding="utf-8-sig")))
    ids[1]["ticker"] = "U"
    _csv(tmp_path / SEGGOLD.IDS_NAME, ids)
    with pytest.raises(SystemExit, match="ticker"):
        SEGGOLD.main()


def _partial_faith_files(tmp_path: Path, marked: list[int]) -> None:
    ids, tags = [], []
    for order in range(1, 31):
        ids.append(
            {
                "review_order": str(order),
                "key": f"{order % 10}#T#{order}",
                "extractors": "baseline",
                "pub_date": "2023-01-01",
            }
        )
        tags.append(
            {
                "article_id": str(order % 10),
                "review_order": str(order),
                "faithful": "yes" if order in marked else "",
            }
        )
    _csv(tmp_path / "faithfulness_ids.csv", ids)
    _csv(tmp_path / "faithfulness_to_tag.csv", tags)
    _bodies(tmp_path)
    _csv(tmp_path / "faithfulness_coverage.csv", [], FAITH.COVERAGE_FIELDS)


def test_a_partial_faithfulness_report_needs_the_first_rows_marked_in_order(
    tmp_path: Path,
) -> None:
    _partial_faith_files(tmp_path, list(range(1, 21)))
    with pytest.raises(MarkedFileError, match="without faithful"):
        REPORT.faithfulness_report(tmp_path)  # the full report still refuses a half-marked file
    text = "\n".join(REPORT.faithfulness_report(tmp_path, partial=15))
    assert "PARTIAL: 20 of 30 texts marked" in text
    with pytest.raises(MarkedFileError, match="the rule needs 25"):
        REPORT.faithfulness_report(tmp_path, partial=25)
    _partial_faith_files(tmp_path, [*range(1, 11), *range(15, 25)])  # a gap: not a random sample
    with pytest.raises(MarkedFileError, match="in order"):
        REPORT.faithfulness_report(tmp_path, partial=15)


def test_the_label_check_report_reads_the_committed_gold_file() -> None:
    gold = Path(__file__).resolve().parents[2] / "gold"
    text = "\n".join(REPORT.label_check_report(gold, gold=True))
    assert "train on `lead` only" in text
    rows = list(csv.DictReader((gold / "segment_check_gold.csv").open(encoding="utf-8")))
    figures = gold_metrics.label_check_report(rows)
    assert (figures["lead"].point, figures["continuation"].point) == (0.88, 0.38)
    assert figures["continuation 2022+"].point == pytest.approx(0.552, abs=0.001)
    assert Counter((r["kind"], r["correct"]) for r in rows) == {
        ("lead", "yes"): 88,
        ("lead", "no"): 12,
        ("continuation", "yes"): 38,
        ("continuation", "no"): 62,
    }  # the figures of PLAN D27 and the page


def test_partial_report_edge_cases(tmp_path: Path) -> None:
    _partial_faith_files(tmp_path, list(range(1, 31)))
    text = "\n".join(REPORT.faithfulness_report(tmp_path, partial=5))
    assert "PARTIAL" not in text  # every row is marked: it is the full report
    with pytest.raises(MarkedFileError, match="at least 1"):
        REPORT.faithfulness_report(tmp_path, partial=0)
    _partial_faith_files(tmp_path, list(range(1, 11)))
    rows = list(csv.DictReader((tmp_path / "faithfulness_to_tag.csv").open(encoding="utf-8-sig")))
    rows[0]["review_order"] = "uno"
    _csv(tmp_path / "faithfulness_to_tag.csv", rows)
    with pytest.raises(MarkedFileError, match="whole number"):
        REPORT.faithfulness_report(tmp_path, partial=5)


def _faithfulness_gold_files(out: Path, marks: tuple[str, str] = ("yes", "no")) -> None:
    texts = ["Acme rose on strong sales.", "Beta fell after a recall."]
    ids = [
        {
            "review_order": str(i),
            "key": f"{i}#T#{i}",
            "article_id": str(i),
            "ticker": "T",
            "text_hash": retag.text_hash(text),
            "extractors": "baseline",
            "revisions": "extractor:baseline@abc",
            "pub_date": "2026-01-01",
        }
        for i, text in enumerate(texts, 1)
    ]
    tagged = [
        {
            "review_order": str(i),
            "article_id": str(i),
            "ticker": "T",
            "text": text,
            "faithful": mark,
            "notes": "=1+1" if i == 1 else "",
        }
        for i, (text, mark) in enumerate(zip(texts, marks, strict=True), 1)
    ]
    _csv(out / FAITHGOLD.IDS_NAME, ids)
    _csv(out / FAITHGOLD.TO_TAG_NAME, tagged)
    (out / FAITHGOLD.COVERAGE_NAME).write_text(
        ",".join(FAITHGOLD.COVERAGE_COLUMNS) + "\n", encoding="utf-8"
    )  # header only: no company was left without a text
    (out / FAITHGOLD.BODIES_NAME).write_text("article_id\n1\n2\n3\n", encoding="utf-8")


def test_faithfulness_gold_copies_the_draws_ids_and_keeps_only_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _faithfulness_gold_files(tmp_path)
    gold = tmp_path / "gold.csv"
    monkeypatch.setattr(sys, "argv", ["b", "--out", str(tmp_path), "--gold", str(gold)])
    FAITHGOLD.main()
    rows = list(csv.DictReader(gold.open(encoding="utf-8")))
    assert tuple(rows[0]) == retag.FAITHFULNESS_GOLD_COLUMNS
    assert [(r["key"], r["text_hash"], r["faithful"]) for r in rows] == [
        ("1#T#1", retag.text_hash("Acme rose on strong sales."), "yes"),
        ("2#T#2", retag.text_hash("Beta fell after a recall."), "no"),
    ]
    assert rows[0]["revisions"] == "extractor:baseline@abc"
    assert rows[0]["notes"] == "=1+1"  # the owner's own mark columns are kept as typed
    assert "text" not in rows[0]  # no news text in gold/


def test_faithfulness_gold_refuses_unmarked_rows_edited_texts_and_other_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "g.csv"
    monkeypatch.setattr(sys, "argv", ["b", "--out", str(tmp_path), "--gold", str(gold)])
    _faithfulness_gold_files(tmp_path, ("yes", ""))
    with pytest.raises(SystemExit, match="without faithful"):
        FAITHGOLD.main()
    _faithfulness_gold_files(tmp_path)
    tagged = list(csv.DictReader((tmp_path / FAITHGOLD.TO_TAG_NAME).open(encoding="utf-8-sig")))
    tagged[1]["text"] = "Beta fell after a recall, edited."
    _csv(tmp_path / FAITHGOLD.TO_TAG_NAME, tagged)
    with pytest.raises(SystemExit, match="edited after the draw"):
        FAITHGOLD.main()
    tagged[1]["text"] = "Beta fell after a recall."
    tagged[1]["ticker"] = "U"
    _csv(tmp_path / FAITHGOLD.TO_TAG_NAME, tagged)
    with pytest.raises(SystemExit, match="ticker"):
        FAITHGOLD.main()
    assert not gold.exists()


def test_the_committed_faithfulness_gold_is_the_owners_190_marks() -> None:
    gold = Path(__file__).resolve().parents[2] / "gold" / "faithfulness_gold.csv"
    rows = list(csv.DictReader(gold.open(encoding="utf-8-sig")))
    assert tuple(rows[0]) == retag.FAITHFULNESS_GOLD_COLUMNS
    assert len(rows) == 190
    assert Counter(r["faithful"] for r in rows) == {"yes": 115, "no": 75}
    assert len({r["key"] for r in rows}) == 190  # one mark per text
    assert {r["extractors"] for r in rows} == {"baseline"}
    assert len({r["article_id"] for r in rows}) == 30  # the 30 bodies drawn


def _run_faithfulness_gold(
    monkeypatch: pytest.MonkeyPatch, out: Path, gold: Path, *flags: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["b", "--out", str(out), "--gold", str(gold), *flags])
    FAITHGOLD.main()


def test_faithfulness_gold_copies_the_coverage_and_the_report_reads_the_tracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _faithfulness_gold_files(tmp_path)
    (tmp_path / FAITHGOLD.COVERAGE_NAME).write_text(
        ",".join(FAITHGOLD.COVERAGE_COLUMNS) + "\n3,T,baseline,2026-01-01\n", encoding="utf-8"
    )
    gold_dir = tmp_path / "gold"
    _run_faithfulness_gold(monkeypatch, tmp_path, gold_dir / "faithfulness_gold.csv")
    copied = list(csv.DictReader((gold_dir / FAITHGOLD.COVERAGE_NAME).open(encoding="utf-8")))
    assert [(r["article_id"], r["extractor"]) for r in copied] == [("3", "baseline")]
    from_gold = "\n".join(REPORT.faithfulness_gold_report(gold_dir))
    assert "1 companies left without a text" in from_gold
    assert "1 faithful of 2 texts + 1 companies left without one" in from_gold  # as D32 counts it
    # the owner's file with the draw and the coverage, copied into the draw's names, gives the same
    (tmp_path / "faithfulness_bodies.csv").write_text("article_id\n1\n2\n3\n", encoding="utf-8")
    from_data = "\n".join(REPORT.faithfulness_report(tmp_path))
    working_only = [  # the staleness note is about the working files, not the tracked gold
        line for line in from_data.splitlines() if "staleness check" not in line
    ]
    assert working_only == from_gold.splitlines()[1:]  # but for the gold's first line


def test_the_report_reads_the_committed_gold_and_gives_the_figures_of_the_decision() -> None:
    gold = Path(__file__).resolve().parents[2] / "gold"
    text = "\n".join(REPORT.faithfulness_gold_report(gold))
    assert "190 texts, 0 companies left without a text" in text
    assert "[all]   baseline: 60.5% [45.2%, 73.8%] -> fail" in text  # PLAN D32, 2026-11-06
    assert "[2022+]   baseline: 60.6% [27.6%, 86.1%] -> not decided" in text


def test_faithfulness_gold_needs_the_coverage_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _faithfulness_gold_files(tmp_path)
    (tmp_path / FAITHGOLD.COVERAGE_NAME).unlink()
    with pytest.raises(SystemExit, match=FAITHGOLD.COVERAGE_NAME):
        _run_faithfulness_gold(monkeypatch, tmp_path, tmp_path / "g" / "faithfulness_gold.csv")


def test_faithfulness_gold_does_not_change_a_mark_it_already_holds_unless_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold" / "faithfulness_gold.csv"
    _faithfulness_gold_files(tmp_path)
    _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    before = gold.read_bytes()
    _faithfulness_gold_files(tmp_path, ("no", "no"))  # the first mark flipped in data/
    with pytest.raises(SystemExit, match=r"1 rows of .* would change or go missing"):
        _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    assert gold.read_bytes() == before  # nothing was written
    _run_faithfulness_gold(monkeypatch, tmp_path, gold, "--overwrite-marks")
    assert next(iter(csv.DictReader(gold.open(encoding="utf-8"))))["faithful"] == "no"


def test_adding_the_model_to_a_marked_row_keeps_the_baselines_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold" / "faithfulness_gold.csv"
    _faithfulness_gold_files(tmp_path)
    _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    first = list(csv.DictReader(gold.open(encoding="utf-8")))
    ids = list(csv.DictReader((tmp_path / FAITHGOLD.IDS_NAME).open(encoding="utf-8-sig")))
    ids[0]["extractors"] = "baseline|model"  # the model wrote the same text as the baseline
    ids[0]["revisions"] = "extractor:baseline@abc|model@def"
    _csv(tmp_path / FAITHGOLD.IDS_NAME, ids)
    _run_faithfulness_gold(monkeypatch, tmp_path, gold)  # goes through without the flag
    again = list(csv.DictReader(gold.open(encoding="utf-8")))
    assert again[0]["extractors"] == "baseline|model"
    assert [(r["key"], r["faithful"], r["text_hash"]) for r in again] == [
        (r["key"], r["faithful"], r["text_hash"]) for r in first
    ]


def test_the_gold_report_names_a_coverage_file_it_cannot_use(tmp_path: Path) -> None:
    shutil.copy(Path(__file__).resolve().parents[2] / "gold" / "faithfulness_gold.csv", tmp_path)
    with pytest.raises(MarkedFileError, match=r"faithfulness_coverage\.csv not found"):
        REPORT.faithfulness_gold_report(tmp_path)
    coverage = tmp_path / FAITHGOLD.COVERAGE_NAME
    coverage.write_text("article_id,ticker\n1,X\n", encoding="utf-8")
    with pytest.raises(MarkedFileError, match="must have the columns"):
        REPORT.faithfulness_gold_report(tmp_path)
    header = ",".join(FAITHGOLD.COVERAGE_COLUMNS)
    coverage.write_text(f"{header}\n1,X,baseline,2026-01-01\n1,X,baseline,2026-01-01\n")
    with pytest.raises(MarkedFileError, match="repeated keys"):
        REPORT.faithfulness_gold_report(tmp_path)  # one company counted twice as unfaithful


def test_the_gold_report_refuses_rows_of_an_extractor_the_draw_does_not_have(
    tmp_path: Path,
) -> None:
    gold = Path(__file__).resolve().parents[2] / "gold" / "faithfulness_gold.csv"
    rows = list(csv.DictReader(gold.open(encoding="utf-8")))
    rows[0]["extractors"] = ""
    _csv(tmp_path / "faithfulness_gold.csv", rows)
    (tmp_path / FAITHGOLD.COVERAGE_NAME).write_text(",".join(FAITHGOLD.COVERAGE_COLUMNS) + "\n")
    with pytest.raises(MarkedFileError, match="extractors the draw does not have"):
        REPORT.faithfulness_gold_report(tmp_path)


def test_faithfulness_gold_does_not_drop_a_tracked_company_without_a_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold" / "faithfulness_gold.csv"
    _faithfulness_gold_files(tmp_path)
    header = ",".join(FAITHGOLD.COVERAGE_COLUMNS)
    (tmp_path / FAITHGOLD.COVERAGE_NAME).write_text(f"{header}\n3,T,baseline,2026-01-01\n")
    _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    tracked = (gold.parent / FAITHGOLD.COVERAGE_NAME).read_bytes()
    before = gold.read_bytes()
    (tmp_path / FAITHGOLD.COVERAGE_NAME).write_text(f"{header}\n")  # regenerated empty
    with pytest.raises(SystemExit, match=r"1 companies without a text .* would go missing"):
        _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    assert (gold.read_bytes(), (gold.parent / FAITHGOLD.COVERAGE_NAME).read_bytes()) == (
        before,
        tracked,
    )
    _run_faithfulness_gold(monkeypatch, tmp_path, gold, "--overwrite-marks")
    assert (gold.parent / FAITHGOLD.COVERAGE_NAME).read_text() == f"{header}\n"


def test_a_failed_write_leaves_the_gold_and_the_coverage_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold" / "faithfulness_gold.csv"
    _faithfulness_gold_files(tmp_path)
    _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    old = {p.name: p.read_bytes() for p in gold.parent.iterdir()}
    real_fsync, calls = os.fsync, []

    def second_fsync_fails(fd: int) -> None:
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("disk full")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", second_fsync_fails)
    _faithfulness_gold_files(tmp_path, ("no", "no"))
    with pytest.raises(SystemExit, match="both are as they were"):
        _run_faithfulness_gold(monkeypatch, tmp_path, gold, "--overwrite-marks")
    assert {p.name: p.read_bytes() for p in gold.parent.iterdir()} == old  # no temp file either


def _write_coverage(out: Path, *rows: str) -> None:
    header = ",".join(FAITHGOLD.COVERAGE_COLUMNS)
    (out / FAITHGOLD.COVERAGE_NAME).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_faithfulness_gold_refuses_a_coverage_that_is_not_tied_to_the_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold" / "faithfulness_gold.csv"
    _faithfulness_gold_files(tmp_path)
    _write_coverage(tmp_path, "77,T,baseline,2026-01-01")  # a body that was never drawn
    with pytest.raises(SystemExit, match="bodies the draw does not have"):
        _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    _write_coverage(tmp_path, "1,T,baseline,2026-01-01")  # a company that has a text from baseline
    with pytest.raises(SystemExit, match="wrote a text"):
        _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    _write_coverage(tmp_path, "3,T,baseline,2026-01-01")  # a drawn body with no text: allowed
    _run_faithfulness_gold(monkeypatch, tmp_path, gold)
    (tmp_path / FAITHGOLD.BODIES_NAME).unlink()
    with pytest.raises(SystemExit, match=r"faithfulness_bodies\.csv not found"):
        _run_faithfulness_gold(monkeypatch, tmp_path, gold)


def test_the_gold_report_refuses_a_company_left_without_a_text_that_has_one(
    tmp_path: Path,
) -> None:
    gold = Path(__file__).resolve().parents[2] / "gold" / "faithfulness_gold.csv"
    shutil.copy(gold, tmp_path)
    first = next(iter(csv.DictReader(gold.open(encoding="utf-8"))))
    _write_coverage(tmp_path, f"{first['article_id']},{first['ticker']},baseline,2026-01-01")
    with pytest.raises(MarkedFileError, match="wrote a text"):
        REPORT.faithfulness_gold_report(tmp_path)
    _write_coverage(tmp_path, "999999999,ZZZ,baseline,2026-01-01")  # body with no text: no row
    assert "1 companies left without a text" in "\n".join(REPORT.faithfulness_gold_report(tmp_path))


def test_a_replace_that_fails_after_the_first_puts_the_first_file_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("old a")
    b.write_text("old b")
    real, count = os.replace, []

    def fail_on_second(src: object, dst: object) -> None:
        count.append(dst)
        if len(count) == 2:
            raise PermissionError("open in Excel")
        real(str(src), str(dst))

    monkeypatch.setattr(os, "replace", fail_on_second)
    with pytest.raises(PermissionError):
        gold_metrics.write_pair_atomically({a: "new a", b: "new b"})
    assert (a.read_text(), b.read_text()) == ("old a", "old b")  # "as they were" is true
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.csv", "b.csv"]


def test_when_the_first_file_cannot_be_put_back_the_error_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("old a")
    b.write_text("old b")
    real, count = os.replace, []

    def fail_from_second(src: object, dst: object) -> None:
        count.append(dst)
        if len(count) >= 2:
            raise PermissionError("open in Excel")
        real(str(src), str(dst))

    monkeypatch.setattr(os, "replace", fail_from_second)
    with pytest.raises(gold_metrics.PairWriteError, match=r"a\.csv was replaced"):
        gold_metrics.write_pair_atomically({a: "new a", b: "new b"})
    assert b.read_text() == "old b"


def test_a_pair_write_deletes_the_temporary_files_of_a_killed_run(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    (tmp_path / "a.csv.4242.tmp").write_text("half")
    gold_metrics.write_pair_atomically({a: "new"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.csv"]


def test_an_add_refuses_an_old_coverage_file_it_cannot_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path
    cols = list(retag.FAITHFULNESS_COLUMNS)
    owner = [
        {
            **dict.fromkeys(cols, "x"),
            "review_order": "1",
            "article_id": "1",
            "ticker": "T",
            "faithful": "yes",
        }
    ]
    _csv(out / FAITH.TO_TAG_NAME, owner)
    _csv(out / FAITH.IDS_NAME, [_ids_row(1, "1#T#1", "h1")])
    _csv(out / FAITH.BODIES_NAME, [{"article_id": "1", "segments": "1"}])
    _write_coverage(out, "1,U,gpt,2026-01-01")  # an extractor the draw never had
    names = (FAITH.TO_TAG_NAME, FAITH.IDS_NAME, FAITH.COVERAGE_NAME)
    before = {n: (out / n).read_bytes() for n in names}
    monkeypatch.setitem(FAITH.EXTRACTORS, "model", lambda body, resolver: {})
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"extractor:{name}@r")
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    monkeypatch.setattr(FAITH, "CompanyResolver", lambda _u: None)
    monkeypatch.setattr(FAITH, "collect_items", lambda *_a: ([_item("1", "T", "A new text.")], []))
    monkeypatch.setattr(FAITH.retag_cli, "_translator", lambda: None)
    monkeypatch.setattr(
        FAITH.retag_cli,
        "add_spanish",
        lambda _c, texts, *_a, **_k: [
            {"article_id": t["article_id"], "spanish_title": "es", "spanish_text": "es"}
            for t in texts
        ],
    )
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(out), "--add-extractor", "model"])
    with pytest.raises(SystemExit, match="extractors the draw does not have"):
        FAITH.main()
    assert {n: (out / n).read_bytes() for n in names} == before
    assert not (out / "faithfulness_to_tag.before_model.csv").exists()


def test_a_first_draw_that_cannot_write_everything_leaves_no_half_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids_path, coverage_path, target = (
        tmp_path / FAITH.IDS_NAME,
        tmp_path / FAITH.COVERAGE_NAME,
        tmp_path / FAITH.TO_TAG_NAME,
    )
    target.write_bytes(b"header-only scaffold\n")
    real = FAITH.extra._write

    def tagging_file_is_locked(path: Path, rows: list[dict[str, str]]) -> None:
        if path == target:
            raise PermissionError("open in Excel")
        real(path, rows)

    monkeypatch.setattr(FAITH.extra, "_write", tagging_file_is_locked)
    with pytest.raises(OSError, match="open in Excel"):
        FAITH.write_first_draw(
            {ids_path: [{"review_order": "1"}], target: [{"a": "b"}]}, coverage_path, []
        )
    assert not ids_path.exists()  # written before the failure, taken back
    assert not coverage_path.exists()
    assert target.read_bytes() == b"header-only scaffold\n"


SENT_DESIGN = {"checked": 1, "extractor": 1}
SENT_SPEC = [  # kind, ticker, part, predicted, the owner's mark
    ("summary", "", "checked", "positive", "positive"),
    ("company_text", "B", "checked", "negative", "negative"),
    ("company_text", "C", "checked", "neutral", "positive"),
    ("extractor_text", "D", "extractor", "positive", "positive"),
    ("extractor_text", "E", "extractor", "negative", "neutral"),
    ("extractor_text", "F", "extractor", "neutral", "neutral"),
]
SENT_TEXTS = tuple(
    f"Text number {i} about {s[1] or 'the market'}." for i, s in enumerate(SENT_SPEC, 1)
)


def _sentiment_gold_files(out: Path) -> None:
    """One text per class in each part, as the draw writes them (there is no review_order in the
    draw, and the tagging file carries the ``pub_date``)."""
    ids: list[dict[str, str]] = []
    tagged: list[dict[str, str]] = []
    for i, (text, (kind, ticker, part, predicted, mark)) in enumerate(
        zip(SENT_TEXTS, SENT_SPEC, strict=True), 1
    ):
        ids.append(
            {
                "key": f"{i}#{ticker or kind}",
                "article_id": str(i),
                "ticker": ticker,
                "kind": kind,
                "part": part,
                "text_hash": retag.text_hash(text),
                "revision": "rev",
                "stratum": f"{part}/{predicted}",
                "extractor": "baseline" if part == "extractor" else "",
                "predicted": predicted,
                "population": str(100 + (3 if part == "extractor" else 0) + len(predicted)),
                "text": text,
            }
        )
        tagged.append(
            {
                "review_order": str(i),
                "article_id": str(i),
                "ticker": ticker,
                "kind": kind,
                "pub_date": f"202{i}-01-15",
                "text": text,
                "sentiment": mark if i != 3 else "Positive ",
                "notes": "=1+1" if i == 1 else "",
            }
        )
    _csv(out / SENTGOLD.IDS_NAME, ids)
    _csv(out / SENTGOLD.TO_TAG_NAME, tagged)


def _edit_csv(path: Path, edit: Callable[[list[dict[str, str]]], object]) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    edit(rows)
    _csv(path, rows)


def _run_sentiment_gold(
    monkeypatch: pytest.MonkeyPatch, out: Path, gold: Path, *extra: str, design: bool = True
) -> None:
    if design:
        monkeypatch.setattr(gold_metrics, "SENTIMENT_DESIGN", dict(SENT_DESIGN))
    argv = ["b", "--out", str(out), "--gold", str(gold), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    SENTGOLD.main()


def test_sentiment_gold_copies_the_draws_weights_and_keeps_only_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sentiment_gold_files(tmp_path)
    gold = tmp_path / "gold" / "sentiment_gold.csv"
    _run_sentiment_gold(monkeypatch, tmp_path, gold)
    rows = list(csv.DictReader(gold.open(encoding="utf-8")))
    assert tuple(rows[0]) == retag.SENTIMENT_GOLD_COLUMNS
    assert [(r["key"], r["sentiment"], r["part"], r["pub_date"]) for r in rows][:4] == [
        ("1#summary", "positive", "checked", "2021-01-15"),
        ("2#B", "negative", "checked", "2022-01-15"),
        ("3#C", "positive", "checked", "2023-01-15"),
        ("4#D", "positive", "extractor", "2024-01-15"),
    ]
    assert rows[3]["extractor"] == "baseline"
    assert rows[0]["population"] == "108"
    assert rows[0]["text_hash"] == retag.text_hash(SENT_TEXTS[0])
    assert rows[0]["notes"] == "=1+1"  # the owner's own mark columns are kept as typed
    assert "text" not in rows[0]  # no news text in gold/
    first = gold.read_bytes()
    _run_sentiment_gold(monkeypatch, tmp_path, gold)
    assert gold.read_bytes() == first  # byte-identical on a rebuild


def test_sentiment_gold_refuses_an_unmarked_or_mistyped_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sentiment_gold_files(tmp_path)
    path = tmp_path / SENTGOLD.TO_TAG_NAME
    _edit_csv(path, lambda rows: rows[1].update(sentiment=""))
    with pytest.raises(SystemExit, match="without sentiment"):
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    _edit_csv(path, lambda rows: rows[1].update(sentiment="bullish"))
    with pytest.raises(SystemExit, match="outside"):
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    assert not (tmp_path / "g.csv").exists()


def test_sentiment_gold_refuses_a_text_edited_after_the_draw_or_a_row_that_is_not_the_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sentiment_gold_files(tmp_path)
    path = tmp_path / SENTGOLD.TO_TAG_NAME
    _edit_csv(path, lambda rows: rows[0].update(text=rows[0]["text"] + " Edited."))
    with pytest.raises(SystemExit, match="edited after the draw"):
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    _sentiment_gold_files(tmp_path)
    _edit_csv(path, lambda rows: rows[1].update(kind="extractor_text"))  # same key, other kind
    with pytest.raises(SystemExit, match="kind"):
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    _sentiment_gold_files(tmp_path)
    _edit_csv(path, lambda rows: rows.pop())  # one text of the draw missing
    with pytest.raises(SystemExit, match="against"):
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    assert not (tmp_path / "g.csv").exists()


def test_sentiment_gold_refuses_a_draw_that_lacks_or_garbles_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = tmp_path / SENTGOLD.IDS_NAME
    for edit, message in (
        (lambda rows: rows[0].update(population=""), "population"),
        (lambda rows: rows[0].update(population="many"), "whole number"),
        (lambda rows: rows[1].update(stratum=rows[0]["stratum"]), "stratum"),
        (lambda rows: rows[0].update(part="summary"), "part"),
        (lambda rows: rows[0].update(predicted="up", stratum="checked/up"), "not a class"),
        (lambda rows: rows[0].update(population="999"), "stratum size"),
    ):
        _sentiment_gold_files(tmp_path)
        if message == "stratum size":  # two rows of one stratum with two sizes
            _edit_csv(
                ids, lambda rows: rows[1].update(stratum=rows[0]["stratum"], predicted="positive")
            )
        _edit_csv(ids, edit)
        with pytest.raises(SystemExit, match=message):
            _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    assert not (tmp_path / "g.csv").exists()


def test_sentiment_gold_refuses_a_sample_that_is_not_the_one_drawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sentiment_gold_files(tmp_path)
    for name in (SENTGOLD.IDS_NAME, SENTGOLD.TO_TAG_NAME):  # the last text of both files is gone
        _edit_csv(tmp_path / name, lambda rows: rows.pop())
    with pytest.raises(SystemExit, match=r"extractor/neutral: 0 rows, the draw takes 1"):
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    _sentiment_gold_files(tmp_path)  # a class that is smaller than the draw's number is taken whole
    with pytest.raises(SystemExit, match=r"checked/positive: 1 rows, the draw takes 2"):
        monkeypatch.setattr(gold_metrics, "SENTIMENT_DESIGN", {"checked": 2, "extractor": 1})
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv", design=False)


def test_sentiment_gold_refuses_a_pub_date_a_spreadsheet_rewrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sentiment_gold_files(tmp_path)
    _edit_csv(tmp_path / SENTGOLD.TO_TAG_NAME, lambda rows: rows[1].update(pub_date="3/1/2022"))
    with pytest.raises(SystemExit, match=r"2#B: pub_date '3/1/2022'.*not an ISO date"):
        _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")
    _edit_csv(
        tmp_path / SENTGOLD.TO_TAG_NAME,
        lambda rows: rows[1].update(pub_date="2022-03-01T10:00:00+0000"),
    )
    _run_sentiment_gold(monkeypatch, tmp_path, tmp_path / "g.csv")  # the draw's own format passes


def test_sentiment_gold_protects_the_tracked_marks_and_weights_from_a_silent_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold" / "sentiment_gold.csv"
    tag, ids = tmp_path / SENTGOLD.TO_TAG_NAME, tmp_path / SENTGOLD.IDS_NAME
    changes: dict[str, Callable[[], object]] = {
        "the mark": lambda: _edit_csv(tag, lambda rows: rows[0].update(sentiment="negative")),
        "the date": lambda: _edit_csv(tag, lambda rows: rows[0].update(pub_date="2020-02-02")),
        "the revision": lambda: _edit_csv(ids, lambda rows: rows[0].update(revision="other")),
        "the stratum size": lambda: _edit_csv(
            ids,
            lambda rows: [
                r.update(population="555") for r in rows if r["stratum"] == "checked/positive"
            ],
        ),
    }
    for name, change in changes.items():
        gold.unlink(missing_ok=True)
        _sentiment_gold_files(tmp_path)
        _run_sentiment_gold(monkeypatch, tmp_path, gold)
        kept = gold.read_bytes()
        change()
        with pytest.raises(SystemExit, match=r"1 rows .* would change"):
            _run_sentiment_gold(monkeypatch, tmp_path, gold)
        assert gold.read_bytes() == kept, name
        _run_sentiment_gold(monkeypatch, tmp_path, gold, "--overwrite-marks")
        assert gold.read_bytes() != kept, name


def test_every_protected_column_of_the_sentiment_gold_is_compared() -> None:
    row = dict.fromkeys(retag.SENTIMENT_GOLD_COLUMNS, "a")
    assert set(SENTGOLD.PROTECTED) <= set(retag.SENTIMENT_GOLD_COLUMNS)
    assert SENTGOLD.changed_marks([row], [row]) == []
    for column in SENTGOLD.PROTECTED:
        assert SENTGOLD.changed_marks([row], [{**row, column: "b"}]) == ["a"], column
    assert SENTGOLD.changed_marks([row], []) == ["a"]  # a row that goes missing
    assert SENTGOLD.changed_marks([row], [{**row, "notes": "b"}]) == []  # a note may be edited


def _sentiment_fixture_for_the_report(out: Path) -> None:
    """30 texts, 10 per class in part ``checked``; the owner agrees on three of four."""
    ids: list[dict[str, str]] = []
    tagged: list[dict[str, str]] = []
    for label, size in (("positive", 90), ("negative", 70), ("neutral", 80)):
        for n in range(10):
            number = len(ids) + 1
            ids.append(
                {
                    "key": f"{number}#kind",
                    "article_id": str(number),
                    "ticker": "",
                    "kind": "kind",
                    "part": "checked",
                    "text_hash": retag.text_hash(f"t{number // 2}"),  # pairs share a text
                    "revision": "r",
                    "stratum": f"checked/{label}",
                    "extractor": "",
                    "predicted": label,
                    "population": str(size),
                    "text": f"t{number // 2}",
                }
            )
            tagged.append(
                {
                    "review_order": str(number),
                    "article_id": str(number),
                    "ticker": "",
                    "kind": "kind",
                    "pub_date": "2022-03-01" if n % 2 else "2020-03-01",
                    "text": f"t{number // 2}",
                    "sentiment": label if n % 4 else "neutral",
                    "notes": "",
                }
            )
    _csv(out / "sentiment_gold_ids.csv", ids)
    _csv(out / "sentiment_gold_to_tag.csv", tagged)


def test_sentiment_report_from_the_tracked_gold_equals_the_one_from_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gold_metrics, "SENTIMENT_DESIGN", {"checked": 10})
    _sentiment_fixture_for_the_report(tmp_path)
    gold = tmp_path / "gold" / "sentiment_gold.csv"
    _run_sentiment_gold(monkeypatch, tmp_path, gold, design=False)
    # the builder's own extractor part is not in this fixture: only `checked` has a design
    from_data = REPORT.sentiment_report(tmp_path)
    from_gold = REPORT.sentiment_report(gold.parent, gold=True)
    assert from_gold == from_data  # every line, the periods and the confusion included
    text = "\n".join(from_gold)
    assert "2022+" in text or "from 2022 on" in text  # the gold keeps the date
    assert "model (rows) against owner (columns)" in text
    assert "28 of 30 texts are shared" in text


def test_sentiment_report_prints_where_the_model_and_the_owner_part(tmp_path: Path) -> None:
    _sentiment_gold_files(tmp_path)  # model: P N Z P N Z ; owner: P N P P Z Z
    ids = {
        r["key"]: r
        for r in csv.DictReader((tmp_path / SENTGOLD.IDS_NAME).open(encoding="utf-8-sig"))
    }
    marks = ("positive", "negative", "positive", "positive", "neutral", "neutral")
    lines = REPORT._confusion_lines(
        [
            {
                "predicted": ids[k]["predicted"],
                "stratum": ids[k]["stratum"],
                "part": ids[k]["part"],
                "gold": g,
            }
            for k, g in zip(ids, marks, strict=True)
        ],
        {i["stratum"]: int(i["population"]) for i in ids.values()},
    )
    table = {
        line.split(":")[0].split()[-1]: [int(c) for c in line.split(":")[1].split("|")]
        for line in lines[1:4]
    }
    assert table == {"positive": [2, 0, 0], "negative": [0, 1, 1], "neutral": [1, 0, 1]}
    weighted = next(i for i, line in enumerate(lines) if "re-weighted by stratum size" in line)
    # neutral row: checked/neutral (population 107, owner positive) and extractor/neutral (110, neutral)
    assert lines[weighted + 1].endswith("100.0% |   0.0% |   0.0%")
    assert lines[weighted + 3].endswith(" 49.3% |   0.0% |  50.7%")
    assert any(line.startswith("[extractor]") for line in lines)  # the split by part


def test_sentiment_report_from_gold_refuses_a_short_or_altered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gold_metrics, "SENTIMENT_DESIGN", {"checked": 10})
    _sentiment_fixture_for_the_report(tmp_path)
    gold = tmp_path / "gold" / "sentiment_gold.csv"
    _run_sentiment_gold(monkeypatch, tmp_path, gold, design=False)
    full = gold.read_text(encoding="utf-8")
    lines = full.splitlines()
    gold.write_text("\n".join(lines[:21]) + "\n", encoding="utf-8")  # 20 of the 30 texts
    with pytest.raises(MarkedFileError, match=r"checked/neutral: 0 rows, the draw takes 10"):
        REPORT.sentiment_report(gold.parent, gold=True)
    gold.write_text(full.replace(",checked,", ",summary,", 1), encoding="utf-8")
    with pytest.raises(MarkedFileError, match="part"):
        REPORT.sentiment_report(gold.parent, gold=True)
    gold.write_text(full.replace(",90,", ",91,", 1), encoding="utf-8")
    with pytest.raises(MarkedFileError, match="stratum size"):
        REPORT.sentiment_report(gold.parent, gold=True)


# What shows that a script reaches the databases: a connect, a path helper, or the variable's name
# as a string (``os.environ.get("SOURCE_DATABASE_URL")``, an argparse default built from it).
_OPENS = (
    "connect_pipeline",
    "connect_two_store",
    "source_db_path",
    "results_db_path",
    "SOURCE_DATABASE_URL",
    "DATABASE_URL",
)
_LOADS = ("load_env_file", "load_dotenv", "EvalSettings.load")


def _called(node: ast.AST) -> list[tuple[int, str]]:
    """(line, name) of every call under *node*, ``a.b.name(...)`` and ``name(...)`` alike, and of
    every string constant that is exactly the name of a database variable."""
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value in ("SOURCE_DATABASE_URL", "DATABASE_URL"):
            found.append((sub.lineno, sub.value))
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "load" and isinstance(fn, ast.Attribute):  # the evaluation settings load it
                name = f"{getattr(fn.value, 'id', '')}.load"
            found.append((sub.lineno, name))
    return found


def _scripts_that_never_load_the_env_file(sources: dict[str, str]) -> list[str]:
    """A script that opens the database must read ``.env`` for real: import ``pipeline`` (which
    loads it) or call ``load_env_file`` / ``load_dotenv`` at module level, or in ``main`` before
    its first connect. A name in a comment, a docstring or a function that only some commands
    reach does not count."""
    bad = []
    for name, text in sources.items():
        tree = ast.parse(text)
        if not any(call in _OPENS for _, call in _called(tree)):
            continue
        top = [n for n in tree.body if not isinstance(n, ast.FunctionDef | ast.ClassDef)]
        imports_pipeline = any(
            (isinstance(n, ast.Import) and any(a.name == "pipeline" for a in n.names))
            or (isinstance(n, ast.ImportFrom) and n.module == "pipeline")
            for n in tree.body
        )
        loads_at_top = any(call in _LOADS for n in top for _, call in _called(n))
        main = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None
        )
        loads_in_main = False
        if main is not None:
            calls = sorted(_called(main))
            first_load = next((ln for ln, c in calls if c in _LOADS), None)
            first_open = next((ln for ln, c in calls if c in _OPENS), None)
            loads_in_main = first_load is not None and (
                first_open is None or first_load < first_open
            )
        if not (imports_pipeline or loads_at_top or loads_in_main):
            bad.append(name)
    return bad


def test_every_script_that_opens_the_database_loads_the_env_file() -> None:
    """The first real run of the summaries cut failed because it connected without reading
    ``.env``: every script that connects must load it, as a call and before it connects."""
    root = Path(__file__).resolve().parents[2]
    sources = {
        f"{folder}/{p.name}": p.read_text(encoding="utf-8")
        for folder in ("scripts", "cli", "apps")
        for p in sorted((root / folder).glob("*.py"))
    }
    missing = _scripts_that_never_load_the_env_file(sources)
    assert not missing, f"these scripts connect but never load .env: {missing}"


def test_the_env_file_scan_wants_a_call_before_the_connect_not_a_mention() -> None:
    mention = '"""load_env_file."""\nimport news_nlp as db\n\ndef main():\n    # load_dotenv()\n    db.connect_pipeline()\n'
    late = "import news_nlp as db\n\ndef main():\n    db.connect_pipeline()\n    db.env.load_env_file()\n"
    only_in_a_helper = (
        "import news_nlp as db\nfrom dotenv import load_dotenv\n\n"
        "def helper():\n    load_dotenv()\n\ndef main():\n    db.connect_pipeline()\n"
    )
    good_main = "import news_nlp as db\n\ndef main():\n    db.env.load_env_file()\n    db.connect_pipeline()\n"
    good_top = "import news_nlp as db\nfrom dotenv import load_dotenv\nload_dotenv()\ndb.connect_pipeline()\n"
    good_pipeline = "import pipeline\nimport news_nlp as db\ndb.connect_pipeline()\n"
    no_db = "def main():\n    pass\n"
    reads_the_variable = (
        "import os\n\ndef main():\n    path = os.environ.get('SOURCE_DATABASE_URL')\n"
        "    return path\n"
    )
    good_reads = (
        "import os\nfrom news_nlp.env import load_env_file\n\ndef main():\n"
        "    load_env_file()\n    return os.environ.get('SOURCE_DATABASE_URL')\n"
    )
    good_settings = (
        "import news_nlp as db\nfrom news_nlp.eval.config import EvalSettings\n\n"
        "def main():\n    EvalSettings.load()\n    db.connect_pipeline()\n"
    )
    other_load = "import news_nlp as db\n\ndef main():\n    x.load()\n    db.connect_pipeline()\n"
    sources = {
        "good_settings.py": good_settings,
        "other_load.py": other_load,
        "reads_the_variable.py": reads_the_variable,
        "good_reads.py": good_reads,
        "mention.py": mention,
        "late.py": late,
        "helper.py": only_in_a_helper,
        "good_main.py": good_main,
        "good_top.py": good_top,
        "good_pipeline.py": good_pipeline,
        "no_db.py": no_db,
    }
    assert _scripts_that_never_load_the_env_file(sources) == [
        "other_load.py",
        "reads_the_variable.py",
        "mention.py",
        "late.py",
        "helper.py",
    ]


def test_the_new_texts_of_an_entry_include_the_ones_other_extractors_also_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert FAITH.label_extractors("1#T#2 (baseline|units)") == {"baseline", "units"}
    _csv(tmp_path / "faithfulness_bodies.csv", [{"article_id": "1", "segments": "1"}])
    _csv(tmp_path / "faithfulness_ids.csv", _stale_ids(("1#T#1", "text", "baseline", "r1")))
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    monkeypatch.setattr(FAITH, "extractor_revision", lambda name: f"rev-{name}")
    registered: dict[str, Callable[[str, object], dict[str, str]]] = {
        n: (lambda body, resolver: {}) for n in ("baseline", "units", "units2")
    }
    monkeypatch.setattr(FAITH, "EXTRACTORS", registered)
    written = [
        _stale_item("1#T#1", "text", "baseline"),
        _stale_item("1#T#2", "fresh", "baseline|units"),  # a new text two extractors write
        _stale_item("1#T#3", "fresh two", "units2"),  # one only units2 writes
    ]
    monkeypatch.setattr(FAITH, "collect_items", lambda bodies, names, resolver: (written, []))
    FAITH.run_stale(types.SimpleNamespace(out=tmp_path, only=None))
    saved = gold_metrics.read_stale(tmp_path / gold_metrics.STALE_NAME)
    assert saved["units"]["new"] == ["1#T#2 (baseline|units)"]  # not the units2 text
    assert saved["baseline"]["new"] == ["1#T#2 (baseline|units)"]
    assert saved["units2"]["new"] == ["1#T#3 (units2)"]


def _read_row(path: Path) -> dict[str, str]:
    return next(csv.DictReader(path.open(encoding="utf-8-sig")))


def test_the_exempt_reading_file_gets_its_spanish_only_with_yes_and_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _csv(tmp_path / "faithfulness_bodies.csv", [{"article_id": "1", "segments": "1"}])
    monkeypatch.setattr(FAITH.retag_cli, "companies", lambda: [])
    reading = [
        {
            "article_id": "1",
            "ticker": "BE",
            "company": "Bloom Energy",
            "why": "guard",
            "extractors": "units2|units3",
            "title": "A title",
            "text": "Michael Bloom said so.",
            "spanish_title": "",
            "spanish_text": "",
            "real_company": "",
        }
    ]
    rows = [{"article_id": "1", "ticker": "BE", "extractor": "units2", "pub_date": "2023-01-01"}]
    monkeypatch.setattr(
        FAITH, "exempt_rows", lambda bodies, resolver: (rows, [dict(r) for r in reading])
    )
    paid: list[list[dict[str, str]]] = []

    def fake_add_spanish(cache, asked, translator, *, yes, max_new):  # type: ignore[no-untyped-def]
        assert yes is True
        paid.append(asked)
        return [
            {
                **a,
                "spanish_title": "Un titulo",
                "spanish_text": "Michael Bloom lo dijo: percepción.",
            }
            for a in asked
        ]

    monkeypatch.setattr(FAITH.retag_cli, "add_spanish", fake_add_spanish)
    monkeypatch.setattr(FAITH.retag_cli, "_translator", object)
    path = tmp_path / gold_metrics.EXEMPT_READ_NAME

    FAITH.write_exempt(tmp_path)  # without --yes: nothing is bought, the columns stay empty
    assert not paid and _read_row(path)["spanish_text"] == ""
    FAITH.write_exempt(tmp_path, yes=True)
    row = _read_row(path)
    assert paid == [[{"article_id": "1#BE", "title": "A title", "text": "Michael Bloom said so."}]]
    assert (row["spanish_title"], row["spanish_text"]) == (
        "Un titulo",
        "Michael Bloom lo dijo: percepción.",
    )
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # a byte order mark: Excel reads UTF-8, not cp1252
    assert "percepción".encode() in raw and "percepciÃ³n".encode() not in raw
    header = next(csv.reader(path.open(encoding="utf-8-sig")))
    assert header.index("spanish_title") == header.index("text") + 1  # right after the text
    FAITH.write_exempt(
        tmp_path, yes=True
    )  # drawn again: what was paid for is kept, not bought again
    assert len(paid) == 1
    assert _read_row(path)["spanish_text"] != ""
    # the owner's spreadsheet reordered the columns and marked a row: both survive a new draw
    rows_now = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows_now[0]["real_company"] = "yes"
    rows_now[0]["spanish_text"] = (
        ""  # and this cell is empty again: it is the only one to be filled
    )
    order = ["ticker", "article_id", "company", "why", "extractors", "title", "text"]
    order += ["spanish_title", "spanish_text", "real_company"]
    _csv(path, rows_now, tuple(order))
    extra_row = {**reading[0], "ticker": "HD", "company": "Home Depot"}
    rows.append({**rows[0], "ticker": "HD"})
    monkeypatch.setattr(
        FAITH, "exempt_rows", lambda bodies, resolver: (rows, [dict(reading[0]), dict(extra_row)])
    )
    FAITH.write_exempt(tmp_path, yes=True)
    merged = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert next(csv.reader(path.open(encoding="utf-8-sig"))) == order  # their column order
    assert [r["ticker"] for r in merged] == ["BE", "HD"]  # their row first, the new company after
    assert merged[0]["real_company"] == "yes" and merged[1]["real_company"] == ""
    assert merged[0]["spanish_text"] and merged[1]["spanish_text"]
    assert [a["article_id"] for a in paid[-1]] == [
        "1#BE",
        "1#HD",
    ]  # only the empty cells are bought


HELDOUT_ROW_HASHES = Path(__file__).resolve().parent / "heldout_gold_row_hashes.csv"
HELDOUT_GOLD = Path(__file__).resolve().parents[2] / "gold" / "category_gold_v2_test.csv"


def test_the_tracked_heldout_gold_is_the_200_tagged_bodies() -> None:
    with HELDOUT_GOLD.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == retag.TEST_GOLD_COLUMNS
        rows = list(reader)
    assert len(rows) == 200 and len({r["article_id"] for r in rows}) == 200
    assert {r["session"] for r in rows} == {"heldout"}
    assert Counter(r["scope"] for r in rows) == {
        "single": 118,
        "multi": 47,
        "market_economy": 11,
        "sector": 9,
        "unrelated": 8,
        "investing_general": 7,
    }
    bad = {r["article_id"]: errors for r in rows if (errors := retag.check_row(r)[1])}
    assert (
        not bad
    )  # the shape of every tag; the tickers need the SOURCE universe (the builder has it)
    # the GICS codes are checked offline: a spreadsheet's `25,104,530` would be three invalid codes
    gics = retag.Universe({}, {n.lower(): n for n in SUB_INDUSTRY_CODES.values()})
    codes = {c.strip() for r in rows for c in r["sub_industry"].split(",") if c.strip()}
    assert not [c for c in codes if gics.resolve_sub_industry(c)[0] is None]
    # every column of every row: a changed tag, ticker, session or weight is a decision to make in
    # this test (the held-out set is fixed, D30), and the failure names the rows that differ
    with HELDOUT_ROW_HASHES.open(encoding="utf-8", newline="") as fh:
        expected = {r["article_id"]: r["sha256_16"] for r in csv.DictReader(fh)}
    found = {r["article_id"]: _script("build_gold_2026_11_05").row_hash(r) for r in rows}
    differ = sorted(i for i in expected.keys() | found.keys() if expected.get(i) != found.get(i))
    assert not differ, f"rows of the held-out gold that changed or went missing: {differ[:20]}"
    frames: dict[str, set[str]] = {}
    for r in rows:
        frames.setdefault(r["stratum"], set()).add(r["frame"])
    assert all(len(sizes) == 1 for sizes in frames.values())  # one weight per stratum
    # the weights are the sizes of the strata of the 1,000-body sample (D30): no data/ needed
    assert sum(int(next(iter(sizes))) for sizes in frames.values()) == 1000


def test_the_scope_report_reads_the_tracked_gold_with_the_figures_of_the_working_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gold = [
        {
            "article_id": str(i),
            "scope": ("single", "multi", "sector")[i % 3] if i % 7 else "single",
            "stratum": ("single", "multi", "sector")[i % 3],
            "frame": ("800", "150", "50")[i % 3],
        }
        for i in range(60)
    ]
    _csv(tmp_path / "category_gold_v2_test.csv", gold)
    _csv(
        tmp_path / "heldout_strata.csv",
        [
            {"stratum": s, "frame": f}
            for s, f in (("single", "800"), ("multi", "150"), ("sector", "50"))
        ],
    )
    _csv(
        tmp_path / "heldout_ids.csv",
        [{"article_id": g["article_id"], "stratum": g["stratum"]} for g in gold],
    )
    _csv(
        tmp_path / "heldout_to_tag.csv",
        [{"article_id": g["article_id"], "scope": g["scope"]} for g in reversed(gold)],
    )
    from_gold = REPORT.scope_gold_report(tmp_path)
    assert from_gold[0] == "from the tracked gold: 60 bodies"
    assert from_gold[1:] == REPORT.scope_report(tmp_path)  # also when the file's order differs
    monkeypatch.setattr(sys, "argv", ["report", "scope", "--gold", "--out", str(tmp_path)])
    REPORT.main()
    assert capsys.readouterr().out.splitlines()[0] == "from the tracked gold: 60 bodies"
    # the same rows in the opposite order give the same lines on the working-file path
    _csv(
        tmp_path / "heldout_to_tag.csv",
        [{"article_id": g["article_id"], "scope": g["scope"]} for g in gold],
    )
    assert REPORT.scope_report(tmp_path) == from_gold[1:]


def test_the_scope_gold_report_refuses_two_weights_for_one_stratum(tmp_path: Path) -> None:
    rows = [
        {"article_id": "1", "scope": "single", "stratum": "single", "frame": "10"},
        {"article_id": "2", "scope": "single", "stratum": "single", "frame": "11"},
    ]
    _csv(tmp_path / "category_gold_v2_test.csv", rows)
    with pytest.raises(MarkedFileError, match="two frame sizes"):
        REPORT.scope_gold_report(tmp_path)


def test_the_tracked_weights_are_the_draws_when_the_draw_is_at_hand() -> None:
    strata = Path(__file__).resolve().parents[2] / "data" / "heldout_strata.csv"
    if not strata.exists():
        pytest.skip("the draw's strata sizes are local (git-ignored data/)")
    with strata.open(encoding="utf-8-sig", newline="") as fh:
        frame = {r["stratum"]: r["frame"] for r in csv.DictReader(fh)}
    with HELDOUT_GOLD.open(encoding="utf-8", newline="") as fh:
        tracked = {(r["stratum"], r["frame"]) for r in csv.DictReader(fh)}
    assert tracked == set(frame.items())


def test_a_rebuild_may_fill_the_shell_but_not_change_a_tag_of_the_heldout_gold() -> None:
    build = _script("build_gold_2026_11_05")
    old = [
        {"article_id": "1", "scope": "single", "company_labels": "A:x", "frame": "9"},
        {"article_id": "2", "scope": "", "company_labels": "", "frame": "9"},
        {"article_id": "3", "scope": "multi", "company_labels": "A:x, B:y", "frame": "9"},
    ]
    new = [
        {"article_id": "1", "scope": "single", "company_labels": "A:x", "frame": "9"},
        {"article_id": "2", "scope": "single", "company_labels": "B:y", "frame": "9"},
        {"article_id": "3", "scope": "multi", "company_labels": "A:x", "frame": "9"},
    ]
    assert build.changed_tags(old, new) == ["3"]  # the shell row 2 is filled in; row 3 changes
    assert build.changed_tags(old, new[:2]) == ["3"]  # a dropped tagged row counts
    assert build.changed_tags(new, new) == []


def test_the_heldout_rebuild_stops_on_a_changed_tag_and_accepts_the_same_rows(
    tmp_path: Path,
) -> None:
    build = _script("build_gold_2026_11_05")
    rows = [{"article_id": "1", "scope": "single", "company_labels": "A:x"}]
    target = tmp_path / "gold.csv"
    _csv(target, rows)
    build.protect_tags(target, [dict(r) for r in rows])  # the same tags: nothing happens
    with pytest.raises(SystemExit, match=r"1 tags .* would change .*--overwrite-marks"):
        build.protect_tags(target, [{**rows[0], "scope": "multi"}])


def _heldout_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> ModuleType:
    """A tiny held-out set in *tmp_path* and ``build_gold_2026_11_05.main`` set up to rebuild it."""
    build = _script("build_gold_2026_11_05")
    monkeypatch.setattr(build.retag_cli, "load_universe", lambda: None)
    tag = ("article_id", "ticker", "scope", "company_labels", "mentions", "sub_industry", "notes")
    _csv(
        tmp_path / "heldout_to_tag.csv",
        [
            {
                "article_id": "1",
                "ticker": "AAPL",
                "scope": "single",
                "company_labels": "Apple:earnings_performance",
                "mentions": "",
                "sub_industry": "",
                "notes": "",
            }
        ],
        tag,
    )
    _csv(tmp_path / "heldout_ids.csv", [{"article_id": "1", "stratum": "single"}])
    _csv(tmp_path / "heldout_strata.csv", [{"stratum": "single", "frame": "514"}])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--set",
            "test",
            "--out",
            str(tmp_path),
            "--gold",
            str(tmp_path / "g.csv"),
            *args,
        ],
    )
    return build


def _heldout_gold(tmp_path: Path, scope: str, labels: str) -> None:
    _csv(
        tmp_path / "g.csv",
        [
            {
                "article_id": "1",
                "session": "heldout",
                "ticker": "AAPL",
                "scope": scope,
                "company_labels": labels,
                "mentions": "",
                "sub_industry": "",
                "notes": "",
                "stratum": "single",
                "frame": "514",
            }
        ],
    )


def test_the_heldout_build_runs_through_main_with_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build = _heldout_build(tmp_path, monkeypatch)
    _heldout_gold(tmp_path, "", "")  # the shell: untagged, filled in freely
    build.main()
    written = (tmp_path / "g.csv").read_text(encoding="utf-8")
    assert "single" in written and "514" in written
    build.main()  # the same tags again: accepted
    _heldout_gold(
        tmp_path, "multi", "APPLE:earnings_performance"
    )  # a tag the file holds, different
    with pytest.raises(
        SystemExit, match=r"1 tags .* would change .*first ids \['1'\].*--overwrite-marks"
    ):
        build.main()
    assert "multi" in (tmp_path / "g.csv").read_text(encoding="utf-8")  # nothing was rewritten
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--overwrite-marks"])
    capsys.readouterr()
    build.main()
    out = capsys.readouterr().out
    assert "--overwrite-marks: 1 tags" in out and "ids ['1']" in out
    assert "single" in (tmp_path / "g.csv").read_text(encoding="utf-8")


def test_the_heldout_guard_compares_with_the_committed_file_not_a_local_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build = _heldout_build(tmp_path, monkeypatch)
    _heldout_gold(tmp_path, "single", "APPLE:earnings_performance")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for command in (["init", "-q"], ["add", "g.csv"], ["commit", "-q", "-m", "tags"]):
        subprocess.run([*git, *command], check=True, capture_output=True)  # noqa: S603
    _heldout_gold(
        tmp_path, "multi", "APPLE:earnings_performance"
    )  # a local edit of the tracked file
    with (tmp_path / "g.csv").open("a", encoding="utf-8") as fh:
        fh.write(
            "2,heldout,MSFT,single,MICROSOFT:earnings_performance,,,,single,514\n"
        )  # an extra row
    build.main()  # the rebuild gives back the committed tags and set: not a change against HEAD
    assert "reference tags: HEAD" in capsys.readouterr().out
    assert "single" in (tmp_path / "g.csv").read_text(encoding="utf-8")


def test_the_heldout_guard_falls_back_to_the_file_on_disk_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build = _heldout_build(tmp_path, monkeypatch)
    _heldout_gold(tmp_path, "single", "APPLE:earnings_performance")

    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(build.subprocess, "run", no_git)
    build.main()  # the disk file has the same tags: accepted, and the log says which reference
    assert "reference tags: disk" in capsys.readouterr().out
    _heldout_gold(tmp_path, "multi", "APPLE:earnings_performance")
    with pytest.raises(SystemExit, match="would change"):
        build.main()


def test_write_hashes_renews_the_row_hashes_only_with_overwrite_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _heldout_build(tmp_path, monkeypatch, "--write-hashes")
    monkeypatch.setattr(build, "HASHES", tmp_path / "hashes.csv")
    _heldout_gold(tmp_path, "", "")
    build.main()  # no hash file yet: written
    written = (tmp_path / "hashes.csv").read_text(encoding="utf-8")
    assert written.startswith("article_id,sha256_16\n1,") and len(written.splitlines()) == 2
    with pytest.raises(SystemExit):  # it exists: replacing it needs --overwrite-marks
        build.main()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--overwrite-marks"])
    build.main()


# --- text measures of the company texts (T-148b10) ---------------------------------------------

MEASURES = _script("text_measures_2026_11_08")


def _fake_resolve(sentence: str) -> list[str]:
    return [t for t in ("AAA", "BBB") if t in sentence]


def test_measure_text_groups_sentences_by_names() -> None:
    text = "AAA rose. AAA and BBB fell. It was calm."
    chars, n, own, mixed, nameless, others = MEASURES.measure_text(
        text, "AAA", ["AAA rose.", "AAA and BBB fell.", "It was calm."], _fake_resolve
    )
    assert (chars, n, own, mixed, nameless, others) == (len(text), 3, 2, 1, 1, 1)


def test_measure_text_sentence_of_only_another_company_is_in_no_group() -> None:
    _, n, own, mixed, nameless, others = MEASURES.measure_text(
        "BBB rose.", "AAA", ["BBB rose."], _fake_resolve
    )
    assert (n, own, mixed, nameless, others) == (1, 0, 0, 0, 0)


def test_read_measures_counts_a_shared_text_once_per_extractor(tmp_path: Path) -> None:
    (tmp_path / "faithfulness_ids.csv").write_text(
        "review_order,article_id,ticker,extractors\n1,10,AAA,baseline|units\n2,10,BBB,baseline\n",
        encoding="utf-8",
    )
    (tmp_path / "faithfulness_to_tag.csv").write_text(
        "review_order,text\n1,AAA rose. BBB fell.\n2,BBB fell.\n", encoding="utf-8"
    )
    rows = MEASURES.read_measures(
        tmp_path, lambda t: [s.strip() + "." for s in t.split(".") if s.strip()], _fake_resolve
    )
    assert sorted((r.extractor, r.ticker) for r in rows) == [
        ("baseline", "AAA"),
        ("baseline", "BBB"),
        ("units", "AAA"),
    ]
    lines = MEASURES.summary_lines(rows)
    assert lines[1].startswith("baseline") and lines[2].startswith("units")


def test_paired_difference_is_over_shared_companies_and_none_without_any() -> None:
    def row(extractor: str, article: str, chars: int) -> object:
        return MEASURES.Measure(extractor, article, "AAA", chars, 2, 1, 0, 1, 0)

    rows = [row("baseline", str(i), 300) for i in range(5)] + [
        row("units", str(i), 100) for i in range(5)
    ]
    n, mean, low, high = MEASURES.paired(rows, "baseline", "units", lambda r: float(r.chars))
    assert (n, mean, low, high) == (5, 200.0, 200.0, 200.0)
    assert MEASURES.paired(rows, "units2", "units", lambda r: float(r.chars)) is None
