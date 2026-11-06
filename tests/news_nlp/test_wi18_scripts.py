"""The draw and build scripts of Work item 18 (2026-11-05): the pure parts, without a database."""

from __future__ import annotations

import csv
import importlib.util
import sys
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


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            ids.append({"review_order": str(order), "key": key, "extractors": extractors})
            tags.append(
                {"article_id": str(article), "review_order": str(order), "faithful": faithful}
            )
    _csv(tmp_path / "faithfulness_ids.csv", ids)
    _csv(tmp_path / "faithfulness_to_tag.csv", tags)
    text = "\n".join(REPORT.faithfulness_report(tmp_path))
    assert "baseline" in text and "model" in text and "model minus baseline" in text


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
