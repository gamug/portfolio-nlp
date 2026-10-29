"""news_nlp.retag: the v2 file shows the v1 tag, never the LLM label, and validates the return."""

from __future__ import annotations

import csv
from pathlib import Path

from news_nlp import retag
from news_nlp.gold_review import TaggedFileError, write_review_files

V1 = {
    "review_order": "1",
    "article_id": "10",
    "ticker": "AAPL",
    "company": "Apple",
    "pub_date": "2020-01-01",
    "source_domain": "x.com",
    "title": "T",
    "text": "body",
    "human_label": "miscellany",
    "human_notes": "",
}
KEY = {"review_order": "1", "article_id": "10", "llm_label": "other"}


def row(**kw: str) -> dict[str, str]:
    return {
        "article_id": "10",
        "scope": "",
        "companies": "",
        "company_labels": "",
        "sub_industry": "",
        **kw,
    }


def test_export_shows_v1_and_hides_the_llm_label() -> None:
    review, key = retag.retag_rows([V1], [KEY])
    assert review[0]["v1_label"] == "miscellany"
    assert review[0]["v1_scope_hint"] == "multi"
    assert all(review[0][c] == "" for c in retag.TAG_COLUMNS)
    assert "llm_label" not in review[0]
    assert key[0]["llm_label"] == "other"


def test_a_tagged_v2_file_is_not_overwritten(tmp_path: Path) -> None:
    review, key = retag.retag_rows([V1], [KEY])
    write_review_files(
        tmp_path,
        review,
        key,
        review_name=retag.REVIEW_NAME,
        key_name=retag.KEY_NAME,
        tag_column="scope",
    )
    path = tmp_path / retag.REVIEW_NAME
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows[0]["scope"] = "single"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        write_review_files(
            tmp_path,
            review,
            key,
            review_name=retag.REVIEW_NAME,
            key_name=retag.KEY_NAME,
            tag_column="scope",
        )
    except TaggedFileError:
        return
    raise AssertionError("a tagged v2 file was overwritten")


def test_valid_rows() -> None:
    g, e = retag.check_row(
        row(scope="single", companies="AAPL", company_labels="AAPL:product_innovation")
    )
    assert not e and g and g.labels == {"AAPL": "product_innovation"}
    g, e = retag.check_row(row(scope="sector", sub_industry="Semiconductors"))
    assert not e and g
    g, e = retag.check_row(row(scope="market_economy"))
    assert not e and g
    g, e = retag.check_row(
        row(
            scope="multi", companies="AAPL; MSFT", company_labels="AAPL:peer_comparison; MSFT:other"
        )
    )
    assert not e and g and g.companies == ("AAPL", "MSFT")


def test_invalid_rows_say_why() -> None:
    assert retag.check_row(row(scope="everything"))[0] is None
    _, e = retag.check_row(row(scope="single", companies="AAPL", company_labels="AAPL:made_up"))
    assert "not a D15 label" in e[0]
    _, e = retag.check_row(row(scope="unrelated", companies="AAPL"))
    assert "names no company" in e[0]
    _, e = retag.check_row(row(scope="sector"))
    assert "sub_industry" in e[0]
    _, e = retag.check_row(row(scope="multi", companies="AAPL", company_labels="AAPL:other"))
    assert "needs 2 or more" in e[0]
    _, e = retag.check_row(row(scope="single", companies="AAPL", company_labels="MSFT:other"))
    assert any("not in companies" in x for x in e)


def test_load_counts_untagged(tmp_path: Path) -> None:
    review, key = retag.retag_rows([V1], [KEY])
    write_review_files(
        tmp_path,
        review,
        key,
        review_name=retag.REVIEW_NAME,
        key_name=retag.KEY_NAME,
        tag_column="scope",
    )
    good, bad, untagged = retag.load_gold_v2(tmp_path / retag.REVIEW_NAME)
    assert (good, bad, untagged) == ([], {}, 1)
