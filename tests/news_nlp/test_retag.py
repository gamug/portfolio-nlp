"""news_nlp.retag: the v2 file shows the v1 tag, never the LLM label, and validates the return."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from news_nlp import retag
from news_nlp.gold_review import TaggedFileError, write_review_files
from news_nlp.triage import GOLD_SCOPES, Company

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
UNIVERSE = retag.Universe.of(
    [
        Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
        Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
        Company("GOOG", "Alphabet Inc.", "Communication Services", "Interactive Media"),
        Company("GOOGL", "Alphabet Inc.", "Communication Services", "Interactive Media"),
    ]
)


def row(**kw: str) -> dict[str, str]:
    base = {c: "" for c in retag.TAG_COLUMNS}
    return {"article_id": "10", **base, **kw}


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(out: Path) -> None:
    review, key = retag.retag_rows([V1], [KEY])
    write_review_files(
        out, review, key, review_name=retag.REVIEW_NAME, key_name=retag.KEY_NAME, tag_column="scope"
    )


def test_export_shows_v1_and_hides_the_llm_label() -> None:
    review, key = retag.retag_rows([V1], [KEY])
    assert review[0]["v1_label"] == "miscellany"
    assert review[0]["v1_scope_derived"] == "multi"
    assert all(review[0][c] == "" for c in retag.TAG_COLUMNS)
    assert "llm_label" not in review[0]
    assert key[0]["llm_label"] == "other"


def test_export_refuses_misaligned_files() -> None:
    with pytest.raises(ValueError, match="different articles"):
        retag.retag_rows([V1], [{**KEY, "article_id": "11"}])


def test_a_tagged_v2_file_is_not_overwritten(tmp_path: Path) -> None:
    export(tmp_path)
    path = tmp_path / retag.REVIEW_NAME
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["scope"] = "single"
    write(path, rows)
    with pytest.raises(TaggedFileError):
        export(tmp_path)


def test_valid_rows() -> None:
    g, e = retag.check_row(
        row(
            scope="single",
            companies="AAPL",
            company_labels="AAPL:product_innovation",
            mentions="MSFT",
        ),
        UNIVERSE,
    )
    assert not e and g and g.labels == {"AAPL": "product_innovation"} and g.mentions == ("MSFT",)
    g, e = retag.check_row(row(scope="sector", sub_industry="systems software"), UNIVERSE)
    assert not e and g and g.sub_industry == "Systems Software"
    assert retag.check_row(row(scope="market_economy"), UNIVERSE)[0]
    g, e = retag.check_row(
        row(
            scope="multi",
            companies="AAPL; MSFT",
            company_labels="AAPL:peer_comparison; MSFT:uncertain",
        ),
        UNIVERSE,
    )
    assert not e and g and g.companies == ("AAPL", "MSFT")


def test_share_classes_are_canonicalised() -> None:
    g, e = retag.check_row(
        row(scope="single", companies="GOOGL", company_labels="GOOGL:other"), UNIVERSE
    )
    assert not e and g and g.companies == ("GOOG",) and g.labels == {"GOOG": "other"}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"scope": "everything"}, "is not one of"),
        (
            {"scope": "single", "companies": "AAPL", "company_labels": "AAPL:made_up"},
            "not a company label",
        ),
        (
            {"scope": "single", "companies": "AAPL", "company_labels": "AAPL:unrelated"},
            "not a company label",
        ),
        ({"scope": "unrelated", "companies": "AAPL"}, "analyses no company"),
        ({"scope": "sector"}, "needs a sub_industry"),
        ({"scope": "sector", "sub_industry": "Made Up"}, "not a GICS sub-industry"),
        ({"scope": "sector", "sub_industry": "Information Technology"}, "not a GICS sub-industry"),
        ({"scope": "multi", "companies": "AAPL", "company_labels": "AAPL:other"}, "2 or more"),
        (
            {"scope": "multi", "companies": "AAPL; MSFT", "company_labels": "AAPL:other"},
            "no label for MSFT",
        ),
        (
            {
                "scope": "single",
                "companies": "AAPL; MSFT",
                "company_labels": "AAPL:other; MSFT:other",
            },
            "exactly 1",
        ),
        (
            {"scope": "single", "companies": "AAPL", "company_labels": "MSFT:other"},
            "not in companies",
        ),
        ({"scope": "single", "companies": "ZZZZ", "company_labels": "ZZZZ:other"}, "not a ticker"),
        (
            {"scope": "single", "companies": "AAPL; AAPL", "company_labels": "AAPL:other"},
            "listed twice",
        ),
        (
            {
                "scope": "single",
                "companies": "AAPL",
                "company_labels": "AAPL:other",
                "mentions": "AAPL",
            },
            "both analysed and a mention",
        ),
    ],
)
def test_invalid_rows_say_why(fields: dict[str, str], message: str) -> None:
    gold, errors = retag.check_row(row(**fields), UNIVERSE)
    assert gold is None
    assert any(message in e for e in errors), errors


def test_load_reports_untagged_missing_duplicated_and_unknown(tmp_path: Path) -> None:
    ok = {"scope": "market_economy"}
    write(
        tmp_path / "r.csv",
        [
            row(**ok),
            row(**ok),  # article 10 twice
            row(article_id="11"),  # untagged
            row(article_id="oops", **ok),
            row(article_id="99", **ok),  # not expected
        ],
    )
    report = retag.load_gold_v2(tmp_path / "r.csv", [10, 11, 12], UNIVERSE)
    assert (len(report.good), report.untagged) == (3, 1)
    assert report.duplicated == [10]
    assert report.missing == [12]
    assert sorted(report.unknown) == ["99", "oops"]


def test_cli_export_then_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "retag_script", Path(__file__).parents[2] / "scripts" / "retag_gold_2026_10_29.py"
    )
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    write(tmp_path / "category_gold_review.csv", [V1])
    write(tmp_path / "category_gold_review_key.csv", [KEY])
    script.cmd_export(
        tmp_path,
        universe_companies=[Company("AAPL", "Apple", "Information Technology", "Hardware")],
        spanish=False,
    )
    assert (tmp_path / "gics_sub_industries.csv").exists()
    assert (tmp_path / "category_gold_v2_guide.csv").exists()
    script.cmd_check(tmp_path, universe=UNIVERSE)
    assert "untagged 1, expected 1" in capsys.readouterr().out


def test_guide_and_sub_industry_rows() -> None:
    names = {r["name"] for r in retag.guide_rows()}
    assert set(retag.COMPANY_LABELS) <= names and set(GOLD_SCOPES) <= names
    rows = retag.sub_industry_rows(
        [Company("A", "A", "Tech", "Software"), Company("B", "B", "Tech", "Software")]
    )
    assert rows == [{"gics_sector": "Tech", "gics_sub_industry": "Software"}]


def _saved_with_semicolons(path: Path) -> None:
    """What a spreadsheet in a ';' locale makes of a tagged review file."""
    path.write_text(
        "review_order;article_id;scope;companies\n1;10;single;AAPL\n", encoding="utf-8-sig"
    )


def test_export_refuses_a_tagged_file_it_can_no_longer_parse(tmp_path: Path) -> None:
    _saved_with_semicolons(tmp_path / retag.REVIEW_NAME)
    before = (tmp_path / retag.REVIEW_NAME).read_bytes()
    review, key = retag.retag_rows([V1], [KEY])
    with pytest.raises(TaggedFileError, match="no 'scope' column"):
        write_review_files(
            tmp_path,
            review,
            key,
            review_name=retag.REVIEW_NAME,
            key_name=retag.KEY_NAME,
            tag_column="scope",
        )
    assert (tmp_path / retag.REVIEW_NAME).read_bytes() == before


def test_export_refuses_a_review_file_in_another_code_page(tmp_path: Path) -> None:
    (tmp_path / retag.REVIEW_NAME).write_bytes("scope,notes\nsingle,Estée\n".encode("cp1252"))
    review, key = retag.retag_rows([V1], [KEY])
    with pytest.raises(TaggedFileError, match="not UTF-8"):
        write_review_files(
            tmp_path,
            review,
            key,
            review_name=retag.REVIEW_NAME,
            key_name=retag.KEY_NAME,
            tag_column="scope",
        )


def test_read_tag_file_names_the_problem(tmp_path: Path) -> None:
    _saved_with_semicolons(tmp_path / "a.csv")
    with pytest.raises(retag.TagFileError, match="CSV UTF-8"):
        retag.read_tag_file(tmp_path / "a.csv")
    (tmp_path / "b.csv").write_bytes("article_id,scope\n1,Estée\n".encode("cp1252"))
    with pytest.raises(retag.TagFileError, match="not UTF-8"):
        retag.read_tag_file(tmp_path / "b.csv")
    with pytest.raises(retag.TagFileError):
        retag.load_gold_v2(tmp_path / "a.csv")


def test_cli_check_and_normalise_stop_with_a_message_on_a_misread_file(tmp_path: Path) -> None:
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as script  # noqa: PLC0415

    _saved_with_semicolons(tmp_path / retag.REVIEW_NAME)
    (tmp_path / retag.KEY_NAME).write_text(
        "review_order,article_id,llm_label\n1,10,x\n", encoding="utf-8-sig"
    )
    with pytest.raises(SystemExit, match="CSV UTF-8"):
        script.cmd_check(tmp_path, universe=UNIVERSE)
    with pytest.raises(SystemExit, match="CSV UTF-8"):
        script.cmd_normalise(tmp_path, universe_companies=[Company("AAPL", "Apple", "x", "y")])


def test_self_check_exits_non_zero_on_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as script  # noqa: PLC0415

    from news_nlp import company_names  # noqa: PLC0415

    firms = [Company("AAPL", "Apple Inc.", "x", "y"), Company("ALL", "Allstate", "x", "y")]
    script.cmd_normalise(tmp_path, universe_companies=firms, self_check=True)  # clean: returns
    monkeypatch.setitem(company_names.NICKNAMES, "ALL", ("Apple",))
    with pytest.raises(SystemExit) as exc:
        script.cmd_normalise(tmp_path, universe_companies=firms, self_check=True)
    assert exc.value.code == 1


def test_tagged_rows_itself_refuses_a_file_it_cannot_parse(tmp_path: Path) -> None:
    from news_nlp.gold_review import tagged_rows  # noqa: PLC0415

    assert tagged_rows(tmp_path / "missing.csv", "scope") == 0
    _saved_with_semicolons(tmp_path / "a.csv")
    with pytest.raises(TaggedFileError, match="no 'scope' column"):
        tagged_rows(tmp_path / "a.csv", "scope")
