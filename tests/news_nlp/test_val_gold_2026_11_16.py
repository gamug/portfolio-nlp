"""The two sets of marks of PR #201: 190 sentiment pieces and the main-company marks (``gold/``), their texts, their scorers and the figures the docs quote."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_val_gold_2026_11_16 as builder  # noqa: E402  (scripts/ is not a package)
import score_main_company_marks_2026_11_16 as main_score  # noqa: E402
import score_sentiment_val190_2026_11_16 as sent_score  # noqa: E402

csv.field_size_limit(sys.maxsize)
GOLD = ROOT / "gold"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash(
    text: str,
) -> str:  # retag.text_hash, without importing the package (it needs portfolio_common)
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def sentiment() -> list[dict[str, str]]:
    return _read(GOLD / "sentiment_val190.csv")


@pytest.fixture(scope="module")
def marks() -> list[dict[str, str]]:
    return _read(GOLD / "main_company_marks.csv")


@pytest.mark.parametrize(
    ("marks_file", "texts_file"),
    [
        ("sentiment_val190.csv", "sentiment_val190_texts.csv"),
        ("main_company_marks.csv", "main_company_marks_texts.csv"),
    ],
)
def test_every_mark_has_its_text_and_the_hash_matches(marks_file: str, texts_file: str) -> None:
    rows = _read(GOLD / marks_file)
    texts = {r["key"]: r for r in _read(GOLD / "texts" / texts_file)}
    assert len({r["key"] for r in rows}) == len(rows)
    assert set(texts) == {r["key"] for r in rows}
    for r in rows:
        assert r["text_hash"] == texts[r["key"]]["text_hash"] == _hash(texts[r["key"]]["text"])


def test_sentiment_counts_are_the_ones_the_docs_quote(sentiment: list[dict[str, str]]) -> None:
    assert len(sentiment) == 190
    assert {r["batch"] for r in sentiment} == {"90", "100"}

    def count(col: str, value: str, batch: str | None = None) -> int:
        return sum(
            1 for r in sentiment if r[col] == value and (batch is None or r["batch"] == batch)
        )

    assert [count("sentiment", c) for c in ("positive", "negative", "neutral")] == [96, 62, 32]
    assert [count("predicted", c) for c in ("positive", "negative", "neutral")] == [88, 72, 30]
    assert [count("sentiment", c, "90") for c in ("positive", "negative", "neutral")] == [
        39,
        27,
        24,
    ]
    assert [count("sentiment", c, "100") for c in ("positive", "negative", "neutral")] == [
        57,
        35,
        8,
    ]
    assert {r["sentiment"] for r in sentiment} <= {"positive", "negative", "neutral"}


def test_sentiment_figures_are_the_doc_figures() -> None:
    rows = sent_score.load(GOLD / "sentiment_val190.csv")
    assert sent_score.accuracy(rows) == (140, 190)
    table = sent_score.class_table(rows, weighted=True)
    assert (table["positive"]["tp"], table["positive"]["called"], table["positive"]["tagged"]) == (
        73,
        88,
        96,
    )
    assert (table["negative"]["tp"], table["negative"]["called"], table["negative"]["tagged"]) == (
        50,
        72,
        62,
    )
    assert (table["neutral"]["tp"], table["neutral"]["called"], table["neutral"]["tagged"]) == (
        17,
        30,
        32,
    )
    assert round(100 * table["neutral"]["recall_weighted"], 1) == 4.4
    assert round(100 * sent_score.accuracy_weighted(rows), 1) == 78.4
    second = [r for r in rows if r.batch == "100"]
    assert sent_score.accuracy(second) == (80, 100)
    assert sent_score.confusion(rows) == {
        "positive": [73, 8, 7],
        "negative": [14, 50, 8],
        "neutral": [9, 4, 17],
    }


def test_main_company_marks_are_the_doc_counts(marks: list[dict[str, str]]) -> None:
    assert len(marks) == 119
    assert sum(r["es_principal"] == "yes" for r in marks) == 100
    assert {r["es_principal"] for r in marks} == {"yes", "no"}
    assert len({r["item_id"] for r in marks}) == 59


def test_main_company_figures_are_the_doc_figures() -> None:
    rows = main_score.load(GOLD / "main_company_marks.csv")
    c = main_score.at_threshold(rows, 0.5)
    assert (c["tp_main"], c["called_main"], c["yes"]) == (30, 33, 100)
    assert (c["tp_not"], c["called_not"], c["no"]) == (16, 86, 19)
    assert main_score.at_threshold(rows, 0.3)["tp_main"] == 45
    assert main_score.at_threshold(rows, 0.7)["tp_main"] == 17
    assert main_score.by_item(rows) == {
        "items": 59,
        "with_main": 53,
        "found": 30,
        "called": 32,
        "agreed": 30,
    }
    rest = [r for r in rows if r.item not in main_score.DEFAULT_EXCLUDE]
    assert (len(rest), main_score.at_threshold(rest, 0.5)["yes"]) == (98, 79)


def test_scorers_on_a_small_fixture() -> None:
    rows = [
        sent_score.Piece("90", "positive", "positive"),
        sent_score.Piece("90", "positive", "neutral"),
        sent_score.Piece("90", "negative", "negative"),
        sent_score.Piece("100", "neutral", "neutral"),
    ]
    t = sent_score.class_table(rows)
    assert t["positive"]["precision"] == 0.5
    assert t["neutral"]["recall"] == 0.5
    marks = [
        main_score.Mark("a", "X", True, 0.9, 0),
        main_score.Mark("a", "Y", False, 0.2, 0),
        main_score.Mark("b", "Z", True, 0.1, 1),
    ]
    c = main_score.at_threshold(marks, 0.5)
    assert (c["tp_main"], c["called_main"], c["yes"], c["tp_not"], c["called_not"], c["no"]) == (
        1,
        1,
        2,
        1,
        2,
        1,
    )
    assert main_score.by_item(marks)["found"] == 1
    # yes at 0.9 and 0.1 against one no at 0.2: one win, one loss
    assert main_score.auc(marks) == 0.5


def _doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize(
    ("doc", "patterns"),
    [
        (
            "sentiment_training.md",
            [
                "73.7% [67.0, 79.4] (140 of 190)",
                "83.0% [73.8, 89.4] (73/88)",
                "69.4% [58.0, 78.9] (50/72)",
                "56.7% [39.2, 72.6] (17/30)",
                "| 4.4% |",
                "80.0% [71.1, 86.7]",
            ],
        ),
        (
            "main_company_training.md",
            [
                "90.9% [76.4, 96.9] (30/33)",
                "30.0% [21.9, 39.6] (30/100)",
                "45.0% [35.6, 54.8] (45/100)",
                "85.0% [64.0, 94.8] (17/20)",
                "119 rows of 59 pieces",
                "(30/79)",
            ],
        ),
    ],
)
def test_the_docs_quote_the_scored_figures(doc: str, patterns: list[str]) -> None:
    text = _doc(doc)
    for pattern in patterns:
        assert pattern in text, f"{doc}: {pattern}"


# ---- the reports quote the same figures, and the two languages keep one structure ----

REPORTS = {
    "pipeline_metrics_report.md": ["(140 of 190)", "AUC of **0.663**"],
    "pipeline_metrics_report_es.md": ["(140 de 190)", "AUC de **0.663**"],
}
REPORT_FIGURES = [
    "73.7% [67.0, 79.4]",
    "83.0% [73.8, 89.4] (73/88)",
    "69.4% [58.0, 78.9] (50/72)",
    "56.7% [39.2, 72.6] (17/30)",
    "80.0% [71.1, 86.7]",
    "90.9% [76.4, 96.9] (30/33)",
    "30.0% [21.9, 39.6] (30/100)",
    "45.0% [35.6, 54.8] (45/100)",
]


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_the_reports_quote_the_scored_figures(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    for pattern in [*REPORT_FIGURES, *REPORTS[name]]:
        assert pattern in text, f"{name}: {pattern}"


def test_the_two_reports_have_one_structure() -> None:
    en = (ROOT / "pipeline_metrics_report.md").read_text(encoding="utf-8").splitlines()
    es = (ROOT / "pipeline_metrics_report_es.md").read_text(encoding="utf-8").splitlines()
    assert len(en) == len(es)

    def heading(lines: list[str]) -> list[tuple[str, int]]:
        return [(ln.split(" ")[0], i) for i, ln in enumerate(lines) if ln.startswith("#")]

    assert heading(en) == heading(es)
    assert [ln.count("|") for ln in en if ln.startswith("|")] == [
        ln.count("|") for ln in es if ln.startswith("|")
    ]


# ---- the builder refuses what its docstring says it refuses ----


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sentiment_fixture(data: Path, tag: str = "positive", second_key: str = "sn2#B") -> None:
    key = {
        "key": "sn1#A",
        "article_id": "1",
        "item_id": "1",
        "ticker": "A",
        "text_source": "original",
        "predicted": "positive",
        "predicted_score": "0.9",
        "relevance_score": "0.95",
        "batch": "90",
    }
    row = {
        "review_order": "1",
        "key": "sn1#A",
        "article_id": "sn1#A",
        "ticker": "A",
        "pub_date": "2020",
        "title": "t",
        "text": "a text",
        "company": "A Inc",
        "sentiment": tag,
        "notes": "",
    }
    _write(
        data / "sentiment_val90" / "key.csv",
        [key, {**key, "key": second_key, "item_id": "2", "ticker": "B"}],
    )
    _write(
        data / "sentiment_val90" / "to_tag.csv",
        [row, {**row, "key": second_key, "ticker": "B", "text": "another"}],
    )


def test_builder_rebuilds_the_marks_with_their_text_hash(tmp_path: Path) -> None:
    _sentiment_fixture(tmp_path)
    marks, texts = builder.sentiment_rows(tmp_path)
    assert [m["key"] for m in marks] == ["sn1#A", "sn2#B"]
    assert marks[0]["text_hash"] == texts[0]["text_hash"] == _hash("a text")
    assert (marks[0]["batch"], marks[0]["predicted"], marks[0]["sentiment"]) == (
        "90",
        "positive",
        "positive",
    )
    assert marks[0]["summary_revision"] == builder.SUMMARY_REV


def test_builder_refuses_an_invalid_tag(tmp_path: Path) -> None:
    _sentiment_fixture(tmp_path, tag="bullish")
    with pytest.raises(SystemExit):
        builder.sentiment_rows(tmp_path)


def test_builder_refuses_a_key_without_a_prediction_row(tmp_path: Path) -> None:
    _sentiment_fixture(tmp_path)
    _write(
        tmp_path / "sentiment_val90" / "key.csv",
        [
            {
                "key": "sn1#A",
                "article_id": "1",
                "item_id": "1",
                "ticker": "A",
                "text_source": "original",
                "predicted": "positive",
                "predicted_score": "0.9",
                "relevance_score": "0.9",
                "batch": "90",
            }
        ],
    )
    with pytest.raises(SystemExit):
        builder.sentiment_rows(tmp_path)


def test_builder_refuses_a_duplicate_key() -> None:
    with pytest.raises(SystemExit):
        builder.check_unique([{"key": "x"}, {"key": "x"}], "fixture")


def _marks_fixture(data: Path, mark: str, ticker: str = "A") -> None:
    row = {
        "item_id": "7",
        "texto": "a piece",
        "texto_es": "una pieza",
        "ticker": ticker,
        "empresa": "A Inc",
        "es_principal": mark,
        "falta_la_principal": "no",
        "notas": "",
    }
    _write(data / "main_company" / "to_tag.csv", [row, {**row, "ticker": "B", "es_principal": ""}])
    (data / "main_company" / "to_tag.strata.json").write_text(
        json.dumps({"items": {"7": {"band": 2, "scores": {"A": 0.8, "B": 0.1}}}}), encoding="utf-8"
    )


def test_builder_keeps_only_the_marked_rows(tmp_path: Path) -> None:
    _marks_fixture(tmp_path, "Yes")
    marks, texts = builder.main_company_rows(tmp_path)
    assert [(m["key"], m["es_principal"], m["band"]) for m in marks] == [("7#A", "yes", "2")]
    assert texts[0]["text"] == "a piece"


def test_builder_refuses_an_invalid_mark_and_a_pair_without_a_score(tmp_path: Path) -> None:
    _marks_fixture(tmp_path, "maybe")
    with pytest.raises(SystemExit):
        builder.main_company_rows(tmp_path)
    _marks_fixture(tmp_path, "yes", ticker="Z")
    with pytest.raises(SystemExit):
        builder.main_company_rows(tmp_path)


def test_builder_does_not_overwrite_the_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "sentiment_val190.csv").write_text("key\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["build", "--data", str(tmp_path), "--gold", str(gold)])
    with pytest.raises(SystemExit):
        builder.main()
