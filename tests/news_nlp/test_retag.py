"""news_nlp.retag: the v2 file shows the v1 tag, never the LLM label, and validates the return."""

from __future__ import annotations

import csv
import sys
from collections import Counter
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
            company_labels="AAPL:product_innovation",
            mentions="MSFT",
        ),
        UNIVERSE,
    )
    assert not e and g and g.labels == {"AAPL": ("product_innovation",)} and g.mentions == ("MSFT",)
    g, e = retag.check_row(row(scope="sector", sub_industry="systems software"), UNIVERSE)
    assert not e and g and g.sub_industry == ("45103020",)
    assert retag.check_row(row(scope="market_economy"), UNIVERSE)[0]
    g, e = retag.check_row(
        row(
            scope="multi",
            company_labels="AAPL:peer_comparison, MSFT:uncertain",
            mentions="GOOGL",
        ),
        UNIVERSE,
    )
    assert not e and g and g.companies == ("AAPL", "MSFT") and g.mentions == ("GOOG",)


def test_share_classes_are_canonicalised() -> None:
    g, e = retag.check_row(row(scope="single", company_labels="GOOGL:other"), UNIVERSE)
    assert not e and g and g.companies == ("GOOG",) and g.labels == {"GOOG": ("other",)}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"scope": "everything"}, "is not one of"),
        (
            {"scope": "single", "company_labels": "AAPL:made_up"},
            "not a company label",
        ),
        (
            {"scope": "single", "company_labels": "AAPL:unrelated"},
            "not a company label",
        ),
        ({"scope": "unrelated", "company_labels": "AAPL:other"}, "analyses no company"),
        ({"scope": "sector"}, "needs a sub_industry"),
        ({"scope": "sector", "sub_industry": "Made Up"}, "not a GICS code, sub-industry name"),
        (
            {"scope": "sector", "sub_industry": "45-this is the entire sector"},
            "not a GICS code, sub-industry name",
        ),
        ({"scope": "sector", "sub_industry": "99"}, "is not a GICS code"),
        ({"scope": "sector", "sub_industry": "101"}, "2, 4, 6 or 8 digits"),
        ({"scope": "sector", "sub_industry": "45, 45"}, "listed twice"),
        ({"scope": "multi", "company_labels": "AAPL:other"}, "2 or more"),
        ({"scope": "multi", "company_labels": "AAPL:other, MSFT"}, "no label"),
        ({"scope": "single", "company_labels": "AAPL:other, MSFT:other"}, "exactly 1"),
        (
            {"scope": "single", "company_labels": "AAPL:other; MSFT:other"},
            "separate them with commas",
        ),
        ({"scope": "single", "company_labels": "AAPL:other, AAPL:other"}, "listed twice"),
        (
            {"scope": "single", "company_labels": "AAPL:other", "mentions": "AAPL"},
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
    assert rows == [{"gics_sector": "Tech", "gics_sub_industry": "Software", "gics_code": ""}]
    (known,) = retag.sub_industry_rows([Company("A", "A", "Industrials", "Distributors")])
    assert known["gics_code"] == "25501010"  # the code of the Wikipedia table


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


def test_a_file_still_in_the_first_format_is_refused_with_the_way_out(tmp_path: Path) -> None:
    write(
        tmp_path / "old.csv",
        [{**row(scope="single", company_labels="AAPL:other"), "companies": "AAPL"}],
    )
    with pytest.raises(retag.TagFileError, match="run 'migrate'"):
        retag.read_tag_file(tmp_path / "old.csv")
    write(tmp_path / "empty.csv", [{**row(scope="market_economy"), "companies": ""}])
    assert retag.read_tag_file(tmp_path / "empty.csv")  # an empty old column is ignored


def test_migrate_drops_companies_changes_separators_and_keeps_every_company() -> None:
    old = {
        **row(
            scope="multi",
            company_labels="Amazon:mergers_acquisitions; Macy's; Kohl's",
            mentions="Twiter, Boeing; General Motors",
        ),
        "companies": "Amazon; Toyota; Macy's; Kohl's",
    }
    (new,), problems = retag.migrate_rows([old])
    assert "companies" not in new and list(new) == [k for k in old if k != "companies"]
    assert new["company_labels"] == "Amazon:mergers_acquisitions, Macy's, Kohl's, Toyota"
    assert new["mentions"] == "Twiter, Boeing, General Motors"
    assert any("'Toyota' was in companies" in p for p in problems)
    assert sum("has no label" in p for p in problems) == 3  # Macy's, Kohl's and Toyota


RETAIL = retag.Universe.of(
    [
        Company("AMZN", "Amazon", "Consumer Discretionary", "Broadline Retail"),
        Company("GPC", "Genuine Parts", "Consumer Discretionary", "Distributors"),
        Company("GOOG", "Alphabet", "Communication Services", "Interactive Media & Services"),
    ]
)


@pytest.mark.parametrize(
    ("text", "known"),
    [
        ("25501010", "25501010"),  # a code of the Wikipedia table
        ("25503030", "25503030"),
        (" 25503030 ", "25503030"),
        ("broadline retail", "25503030"),
        ("Interactive Media and Services", "50203010"),
        ("interactive media & services", "50203010"),
        ("25", "25"),  # a sector
        ("2550", "2550"),  # an industry group
        ("255010", "255010"),  # an industry
        ("AMZN", "25503030"),  # a company stands for its own sub-industry
        ("amzn", "25503030"),
    ],
)
def test_sub_industry_as_code_or_name(text: str, known: str) -> None:
    assert RETAIL.resolve_sub_industry(text) == (known, None)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("99999999", "is not a GICS code"),
        ("99", "is not a GICS code"),
        ("255", "2, 4, 6 or 8 digits"),
        ("10101010", "no company of the universe has"),  # Oil & Gas Drilling is a real code, unused
        ("Made Up", "not a GICS code, sub-industry name or ticker"),
    ],
)
def test_sub_industry_refusals_say_why(text: str, message: str) -> None:
    known, why = RETAIL.resolve_sub_industry(text)
    assert known is None and why and message in why


def test_sub_industry_is_optional_on_every_scope_and_checked_when_given() -> None:
    fields = {"scope": "multi", "company_labels": "AMZN:other, GPC:other", "mentions": "GOOG"}
    gold, errors = retag.check_row(row(**fields, sub_industry="25503030"), RETAIL)
    assert not errors and gold and gold.sub_industry == ("25503030",)
    gold, errors = retag.check_row(row(**fields), RETAIL)
    assert not errors and gold and gold.sub_industry == ()
    gold, errors = retag.check_row(row(**fields, sub_industry="99999999"), RETAIL)
    assert gold is None and any("is not a GICS code" in e for e in errors)
    sector, errors = retag.check_row(row(scope="sector", sub_industry="25501010"), RETAIL)
    assert not errors and sector and sector.sub_industry == ("25501010",)
    assert retag.check_row(row(scope="sector"), RETAIL)[0] is None  # still required for a sector


def test_a_company_outside_the_universe_is_kept_as_typed_and_reported() -> None:
    fields = {
        "scope": "multi",
        "company_labels": "Amazon:mergers_acquisitions, Macy's:earnings_performance, AMZN:other",
    }
    gold, errors = retag.check_row(
        row(
            **{**fields, "company_labels": "AMZN:other, Macy's:other, Kohl's:other"},
            mentions="Toyota, GPC",
        ),
        RETAIL,
    )
    assert not errors and gold
    assert gold.companies == ("AMZN", "Macy's", "Kohl's") and gold.mentions == ("Toyota", "GPC")
    assert gold.outside == ("Macy's", "Kohl's", "Toyota")
    gold, errors = retag.check_row(
        row(scope="single", company_labels="Macy's:other, macy's:other"), RETAIL
    )
    assert gold is None and any("label other listed twice" in e for e in errors)
    gold, errors = retag.check_row(
        row(scope="single", company_labels="Macys:other", mentions="Macys"), RETAIL
    )
    assert gold is None and any("both analysed and a mention" in e for e in errors)
    # a delisted company written as its old ticker is outside the universe too, reported
    gold, errors = retag.check_row(row(scope="single", company_labels="USG:other"), RETAIL)
    assert not errors and gold and gold.companies == ("USG",) and gold.outside == ("USG",)
    # with no universe nothing is outside it
    bare, _ = retag.check_row(row(scope="single", company_labels="Macy's:other"))
    assert bare is not None and bare.companies == ("MACY'S",)


def test_load_reports_the_names_outside_the_universe(tmp_path: Path) -> None:
    write(
        tmp_path / "r.csv",
        [
            row(scope="single", company_labels="Twitter:other", mentions="Toyota"),
            row(article_id="11", scope="single", company_labels="Twitter:other"),
        ],
    )
    report = retag.load_gold_v2(tmp_path / "r.csv", [10, 11], RETAIL)
    assert dict(report.outside) == {"Twitter": 2, "Toyota": 1}


def _gate_rows() -> list[dict[str, str]]:
    rows = []
    for i in range(1, 31):
        stratum = "single" if i <= 20 else "multi" if i <= 25 else "sector"
        rows.append(
            {
                "review_order": str(i),
                "article_id": str(100 + i),
                "title": f"t{i}",
                "v1_scope_derived": stratum,
                "scope": "single" if i in (2, 4, 21) else "",
            }
        )
    return rows


def test_draw_gate_keeps_the_tagged_rows_first_and_takes_small_strata_whole() -> None:
    quotas = {"sector": 9, "multi": 3, "single": 5}
    got = retag.draw_gate(_gate_rows(), quotas, seed=1)
    by = {s: [r for r in got if r["stratum"] == s] for s in quotas}
    assert [len(by[s]) for s in quotas] == [5, 3, 5]  # sector has only 5 members: all taken
    assert {"2", "4"} <= {r["review_order"] for r in by["single"]}  # the tagged ones come first
    assert [r["already_tagged"] for r in by["multi"]].count("yes") == 1
    assert [int(r["review_order"]) for r in got] == sorted(int(r["review_order"]) for r in got)
    assert got == retag.draw_gate(_gate_rows(), quotas, seed=1)  # reproducible
    assert got != retag.draw_gate(_gate_rows(), quotas, seed=2)  # and seeded
    assert sum(retag.GATE_QUOTAS.values()) == 240


def _full_rows() -> list[dict[str, str]]:
    return [
        {
            **{k: v for k, v in r.items() if k != "scope"},
            "text": "b",
            "spanish_title": "es",
            "spanish_text": "es b",
            **{c: "" for c in retag.TAG_COLUMNS},
            "scope": r["scope"],
        }
        for r in _gate_rows()
    ]


def test_gate_file_is_the_review_rows_with_the_tags() -> None:
    rows = _full_rows()
    ids = retag.draw_gate(rows, {"sector": 9, "multi": 3, "single": 5}, seed=1)
    gate = retag.gate_rows(ids, rows)
    assert len(gate) == len(ids) == 13
    assert all(list(g) == list(rows[0]) for g in gate)  # the same columns, Spanish after text
    assert {"102", "104", "121"} <= {g["article_id"] for g in gate}  # the tagged ones are in it
    assert next(g for g in gate if g["article_id"] == "102")["scope"] == "single"


def test_merge_copies_new_tags_and_never_overwrites_different_ones() -> None:
    rows = _full_rows()
    gate = [dict(r) for r in rows[:3]]
    gate[0]["scope"], gate[0]["company_labels"] = "single", "AAPL:other"  # 101: untagged in review
    gate[1]["notes"] = "x"  # 102 is tagged "single" in the review file; same scope, other notes
    merged, copied, conflicts, replaced = retag.merge_tags(rows, gate)
    assert copied == 1 and conflicts == ["102 (notes)"] and replaced == []
    first = next(m for m in merged if m["article_id"] == "101")
    assert first["scope"] == "single" and first["company_labels"] == "AAPL:other"
    assert next(m for m in merged if m["article_id"] == "102")["notes"] == ""  # kept as it was
    assert rows[0]["scope"] == ""  # the input is not modified


def test_a_company_may_carry_several_labels_in_one_article() -> None:
    gold, errors = retag.check_row(
        row(scope="single", company_labels="AMZN:earnings_performance, amzn:product_innovation"),
        RETAIL,
    )
    assert (
        not errors
        and gold
        and gold.labels == {"AMZN": ("earnings_performance", "product_innovation")}
    )
    assert gold.companies == ("AMZN",)  # still one analysed company: a single body


def test_sector_market_and_investing_bodies_may_label_the_companies_they_name() -> None:
    for scope in ("sector", "market_economy", "investing_general"):
        extra = {"sub_industry": "25503030"} if scope == "sector" else {}
        gold, errors = retag.check_row(
            row(scope=scope, company_labels="AMZN:market_analyst_sentiment", **extra), RETAIL
        )
        assert not errors and gold and gold.companies == ("AMZN",)
    assert retag.check_row(row(scope="unrelated", company_labels="AMZN:other"), RETAIL)[0] is None


def test_sub_industry_takes_several_nodes_of_any_level_and_companies() -> None:
    gold, errors = retag.check_row(
        row(scope="sector", sub_industry="2010, 25503030,GPC, AMZN"), RETAIL
    )
    assert gold is None and any("listed twice" in e for e in errors)  # AMZN twice
    gold, errors = retag.check_row(
        row(scope="sector", sub_industry="2010,25,GPC,Interactive Media & Services"), RETAIL
    )
    assert not errors and gold and gold.sub_industry == ("2010", "25", "25501010", "50203010")
    # a sub-industry name with a comma is one node
    long = retag.Universe(
        {}, {"metal, glass & plastic containers": "Metal, Glass & Plastic Containers"}
    )
    codes, errors = long.resolve_sub_industries("Metal, Glass & Plastic Containers, 25")
    assert not errors and codes == ("15103010", "25")


def test_the_extra_sector_draw_is_seeded_balanced_and_tops_up_small_sectors() -> None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import draw_sector_extra_2026_11_05 as draw  # noqa: PLC0415

    def item(i: int, noun: str) -> dict[str, object]:
        return {"article_id": i, "noun": noun}

    pool = {
        "Energy": [item(i, "oil" if i % 2 else "gas") for i in range(1, 9)],
        "Utilities": [item(100, "utilities")],  # a sector with fewer candidates than places
        "Financials": [item(i, "banks") for i in range(200, 220)],
    }
    first = draw.draw(pool, 3, seed=1)
    assert first == draw.draw(pool, 3, seed=1)  # the same seed, the same draw
    sectors = Counter(str(r["sector"]) for r in first)
    target = 3 * len(draw.INDUSTRY_NOUNS)
    assert len(first) == min(target, 8 + 1 + 20) and len({r["article_id"] for r in first}) == len(
        first
    )
    assert sectors["Utilities"] == 1 and sectors["Energy"] >= 3
    energy = [r for r in first[:3] if r["sector"] == "Energy"]
    assert {r["noun"] for r in energy} == {"oil", "gas"}  # one noun does not fill the sector


def _tagging_row(**changes: str) -> dict[str, str]:
    row = dict.fromkeys(retag.TAGGING_COLUMNS, "x")
    row.update(changes)
    return row


def test_every_tagging_file_has_the_spanish_columns_after_text_and_translated_rows() -> None:
    assert retag.TAGGING_COLUMNS.index("spanish_title") == retag.TAGGING_COLUMNS.index("text") + 1
    assert retag.TAGGING_COLUMNS.index("spanish_text") == retag.TAGGING_COLUMNS.index("text") + 2
    assert retag.tagging_format_problems([_tagging_row()]) == []
    assert any("empty" in p for p in retag.tagging_format_problems([_tagging_row(spanish_text="")]))
    no_spanish = {k: v for k, v in _tagging_row().items() if not k.startswith("spanish")}
    assert any("not the tagging format" in p for p in retag.tagging_format_problems([no_spanish]))
    shuffled = dict(reversed(list(_tagging_row().items())))
    assert retag.tagging_format_problems([shuffled])
    with pytest.raises(retag.TagFileError, match="is not written"):
        retag.require_tagging_format([no_spanish], "gate.csv")


def test_the_scripts_refuse_to_write_a_tagging_file_without_the_spanish_columns(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415

    bare = {k: "x" for k in retag.TAGGING_COLUMNS if not k.startswith("spanish")}
    with pytest.raises(SystemExit, match="is not written"):
        cli.write_tagging_csv(tmp_path / "t.csv", [bare])
    assert not (tmp_path / "t.csv").exists()
    cli.write_tagging_csv(tmp_path / "ok.csv", [_tagging_row()])
    assert (tmp_path / "ok.csv").exists()


def test_gold_file_rows_carry_the_validated_tags_without_the_article_text() -> None:
    first = row(
        article_id="10",
        scope="multi",
        company_labels="AAPL:product_innovation, MSFT:other, AAPL:peer_comparison",
        mentions="GOOG",
        notes="n1",
    )
    second = row(article_id="5", scope="sector", sub_industry="4510, 45103020")
    sessions = {"gate240": [first], "sector_extra": [second]}
    golds = {
        name: [retag.check_row(r, UNIVERSE)[0] for r in rows] for name, rows in sessions.items()
    }
    assert all(g for gs in golds.values() for g in gs)
    out = retag.gold_file_rows(sessions, {k: [g for g in v if g] for k, v in golds.items()})
    assert [r["article_id"] for r in out] == ["5", "10"]
    assert all(tuple(r) == retag.GOLD_COLUMNS for r in out)
    assert "text" not in out[0] and "title" not in out[0]
    assert out[1]["company_labels"] == "AAPL:product_innovation, AAPL:peer_comparison, MSFT:other"
    assert out[1]["session"] == "gate240" and out[1]["notes"] == "n1"
    with pytest.raises(retag.TagFileError, match="two sessions"):
        retag.gold_file_rows(
            {"a": [first], "b": [first]},
            {"a": [g for g in golds["gate240"] if g], "b": [g for g in golds["gate240"] if g]},
        )


def test_draw_gate_with_the_real_quotas_gives_240() -> None:
    rows = [
        {
            "review_order": str(i),
            "article_id": str(i),
            "title": f"t{i}",
            "v1_scope_derived": stratum,
            "scope": "",
        }
        for i, stratum in enumerate(s for s, q in retag.GATE_QUOTAS.items() for _ in range(q + 5))
    ]
    got = retag.draw_gate(rows, seed=3)
    assert len(got) == 240
    assert retag.gate_shortfalls(rows, got) == ({}, 0)


def test_draw_gate_never_drops_tagged_rows_silently() -> None:
    rows = _gate_rows()  # 2 tagged "single" rows
    with pytest.raises(ValueError, match="already tagged"):
        retag.draw_gate(rows, {"single": 1}, seed=1)


def test_gate_shortfalls_names_small_strata_and_unreachable_rows() -> None:
    rows = _gate_rows()  # 20 single, 5 multi, 5 sector
    quotas = {"sector": 9, "multi": 3}
    got = retag.draw_gate(rows, quotas, seed=1)
    short, unknown = retag.gate_shortfalls(rows, got, quotas)
    assert short == {"sector": (5, 9)} and unknown == 20  # the 20 single rows have no quota


def test_gate_rows_fails_on_an_id_missing_from_the_review_file() -> None:
    rows = _full_rows()
    with pytest.raises(ValueError, match="not in the review file"):
        retag.gate_rows([{"article_id": "999"}], rows)


def test_merge_takes_notes_typed_without_a_scope_and_lists_the_columns_in_conflict() -> None:
    rows = _full_rows()
    gate = [dict(r) for r in rows[:4]]
    gate[0]["notes"] = "flag"  # 101: no scope yet, but the owner wrote a note
    gate[3]["scope"], gate[3]["notes"] = "multi", "n"  # 104 is "single" in the review file
    merged, copied, conflicts, _ = retag.merge_tags(rows, gate)
    assert next(m for m in merged if m["article_id"] == "101")["notes"] == "flag" and copied == 1
    assert conflicts == ["104 (scope, notes)"]


def test_merge_refuses_a_gate_file_that_tags_an_article_twice() -> None:
    rows = _full_rows()
    gate = [{**rows[0], "scope": "single"}, {**rows[0], "scope": "multi"}]
    with pytest.raises(retag.TagFileError, match="twice"):
        retag.merge_tags(rows, gate)


def test_merge_with_source_wins_takes_the_gate_tags_on_a_scope_conflict() -> None:
    rows = _full_rows()  # 104 is "single" in the review file
    gate = [dict(r) for r in rows[:4]]
    gate[3]["scope"] = "multi"
    kept, _, conflicts, replaced = retag.merge_tags(rows, gate)
    assert conflicts == ["104 (scope)"] and replaced == []
    assert next(m for m in kept if m["article_id"] == "104")["scope"] == "single"
    won, _, conflicts, replaced = retag.merge_tags(rows, gate, source_wins=True)
    assert conflicts == [] and replaced == ["104 (scope)"]
    assert next(m for m in won if m["article_id"] == "104")["scope"] == "multi"
    assert rows[3]["scope"] == "single"  # the input is not modified


def test_tag_differences_lists_every_differing_column() -> None:
    rows = _full_rows()
    gate = [dict(r) for r in rows[:4]]
    gate[3]["scope"], gate[3]["notes"] = "multi", "n"
    assert retag.tag_differences(rows, gate) == [
        {"article_id": "104", "column": "scope", "review_value": "single", "gate_value": "multi"},
        {"article_id": "104", "column": "notes", "review_value": "", "gate_value": "n"},
    ]
