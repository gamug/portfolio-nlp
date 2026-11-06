"""The draw and build scripts of Work item 18 (2026-11-05): the pure parts, without a database."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import sqlite3
import sys
import types
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import retag
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


def _faith_files(tmp_path: Path, skipped: list[dict[str, str]]) -> None:
    ids, tags = [], []
    for order, article in enumerate(range(1, 7), 1):
        ids.append(
            {
                "review_order": str(order),
                "key": f"{article}#T#1",
                "extractors": "baseline|model",
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
        "faithfulness_gold.csv": retag.FAITHFULNESS_GOLD_COLUMNS,
        "segment_check_gold.csv": retag.SEGMENT_CHECK_GOLD_COLUMNS,
    }
    for name, columns in expected.items():
        with (ROOT / "gold" / name).open(encoding="utf-8-sig", newline="") as fh:
            assert tuple(next(csv.reader(fh))) == columns, name


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
    _csv(out / FAITH.COVERAGE_NAME, [dict.fromkeys(FAITH.COVERAGE_FIELDS, "")])
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
    with pytest.raises(
        SystemExit, match="previous add"
    ):  # a second run does not overwrite the copy
        FAITH.main()


def test_adding_an_extractor_needs_a_first_draw_and_an_extractor_that_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(tmp_path), "--add-extractor", "model"])
    with pytest.raises(SystemExit, match="no extractor 'model'"):
        FAITH.main()
    monkeypatch.setitem(FAITH.EXTRACTORS, "model", lambda body, resolver: {})
    with pytest.raises(SystemExit, match=r"needs faithfulness_to_tag\.csv"):
        FAITH.main()
