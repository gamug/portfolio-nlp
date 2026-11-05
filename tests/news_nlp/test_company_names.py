"""news_nlp.company_names: names, nicknames and tickers -> canonical tickers, never a guess."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from news_nlp import retag
from news_nlp.company_names import (
    BRANDS,
    NICKNAMES,
    NameBook,
    _key,
    normalise_row,
    self_check,
)
from news_nlp.triage import Company

FIRMS = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
    Company("GOOG", "Alphabet Inc. (Class C)", "Communication Services", "Interactive Media"),
    Company("GOOGL", "Alphabet Inc. (Class A)", "Communication Services", "Interactive Media"),
    Company("META", "Meta Platforms", "Communication Services", "Interactive Media"),
    Company("JPM", "JPMorgan Chase", "Financials", "Diversified Banks"),
    Company("BAC", "Bank of America", "Financials", "Diversified Banks"),
    Company("BK", "Bank of New York Mellon", "Financials", "Asset Management"),
    Company("UAL", "United Airlines Holdings", "Industrials", "Passenger Airlines"),
    Company("UPS", "United Parcel Service", "Industrials", "Air Freight"),
    Company("ALL", "Allstate", "Financials", "Property & Casualty Insurance"),
    Company("JNJ", "Johnson & Johnson", "Health Care", "Pharmaceuticals"),
]
BOOK = NameBook(FIRMS, retag.Universe.of(FIRMS).tickers)


@pytest.mark.parametrize(
    ("text", "ticker"),
    [
        ("AAPL", "AAPL"),
        ("Apple", "AAPL"),
        ("apple inc.", "AAPL"),
        ("Microsoft Corp", "MSFT"),
        ("Google", "GOOG"),  # share classes fold to the first ticker seen
        ("Alphabet", "GOOG"),
        ("GOOGL", "GOOG"),
        ("Facebook", "META"),
        ("Meta", "META"),
        ("J.P. Morgan", "JPM"),
        ("JPMorgan", "JPM"),
        ("BofA", "BAC"),
        ("Bank of America", "BAC"),
        ("Johnson and Johnson", "JNJ"),
        ("J&J", "JNJ"),
        ("$AAPL", "AAPL"),
        ("ALL", "ALL"),  # an upper-case ticker is a ticker
        ("Allstate", "ALL"),
        ("aapl", "AAPL"),  # last resort: a ticker in any case
    ],
)
def test_resolves(text: str, ticker: str) -> None:
    assert BOOK.resolve(text) == (ticker, None)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("Bank", "ambiguous: BAC, BK"),
        ("United", "ambiguous: UAL, UPS"),
        ("Tesla", "unknown"),
        ("the iPhone maker", "unknown"),
        ("", "empty"),
    ],
)
def test_never_guesses(text: str, why: str) -> None:
    ticker, reason = BOOK.resolve(text)
    assert ticker is None and reason == why


def test_nicknames_name_only_tickers_that_exist_in_the_universe() -> None:
    assert BOOK.resolve("Tesla Motors")[0] is None  # TSLA is not in this universe
    assert all(t == t.upper() for t in NICKNAMES)


def test_row_names_become_tickers_and_problems_are_listed() -> None:
    row = {
        "mentions": "Alphabet, Microsoft Corp, Bank, the iPhone maker",
        "company_labels": "Apple:product_innovation, Google: other, Tesla:other",
    }
    new, problems, _ = normalise_row(BOOK, row)
    assert new["mentions"] == "GOOG, MSFT, Bank, the iPhone maker"  # Alphabet folds to GOOG
    assert new["company_labels"] == "AAPL:product_innovation, GOOG:other, Tesla:other"
    assert problems == [
        "mentions: 'Bank' ambiguous: BAC, BK",
        "mentions: 'the iPhone maker' unknown",
        "company_labels: 'Tesla' unknown",
    ]


def test_a_comma_inside_a_company_name_does_not_split_it() -> None:
    from news_nlp.company_names import split_list  # noqa: PLC0415

    assert split_list("Booking Holdings, Inc., Apple") == ["Booking Holdings, Inc.", "Apple"]
    assert split_list("Booking Holdings, Inc.:other, Apple:other") == [
        "Booking Holdings, Inc.:other",
        "Apple:other",
    ]
    assert split_list("A, B,, ,C\nD") == ["A", "B", "C", "D"]


def test_row_conflicting_labels_and_missing_label_are_reported() -> None:
    row = {
        "mentions": "",
        "company_labels": "Google:other, Alphabet:uncertain, Apple",
    }
    new, problems, _ = normalise_row(BOOK, row)
    assert new["company_labels"] == "GOOG:other, Apple"
    assert any("given 'other' and 'uncertain'" in p for p in problems)
    assert any("has no label" in p for p in problems)


def test_empty_cells_stay_empty() -> None:
    new, problems, _ = normalise_row(BOOK, {"mentions": "", "company_labels": ""})
    assert new == {"mentions": "", "company_labels": ""} and problems == []


def test_cli_normalise_then_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415 -- the script is not a package

    base = {c: "" for c in ("review_order", "article_id", *retag.TAG_COLUMNS)}
    typed = {
        **base,
        "article_id": "10",
        "scope": "single",
        "company_labels": "Apple:product_innovation",
        "mentions": "Facebook, Nobody Inc",
    }
    with (tmp_path / retag.REVIEW_NAME).open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(base))
        writer.writeheader()
        writer.writerow(typed)
    with (tmp_path / retag.KEY_NAME).open("w", newline="", encoding="utf-8-sig") as fh:
        fh.write("review_order,article_id,llm_label\n1,10,other\n")
    cli.cmd_normalise(tmp_path, universe_companies=FIRMS)
    shown = capsys.readouterr().out
    assert "article 10: mentions: 'Nobody Inc' unknown" in shown
    out = list(csv.DictReader((tmp_path / cli.TICKERS_NAME).open(encoding="utf-8-sig")))
    assert out[0]["company_labels"] == "AAPL:product_innovation"
    assert out[0]["mentions"] == "META, Nobody Inc"
    assert "Apple:product_innovation" in (tmp_path / retag.REVIEW_NAME).read_text(
        encoding="utf-8-sig"
    )
    cli.cmd_check(tmp_path, universe=retag.Universe.of(FIRMS), file=cli.TICKERS_NAME)
    assert "untagged 0" in capsys.readouterr().out


def test_apostrophes_and_dashes_fold() -> None:
    firms = [
        *FIRMS,
        Company("LOW", "Lowe's", "x", "y"),
        Company("BF.B", "Brown\u2013Forman", "x", "y"),
    ]
    book = NameBook(firms, retag.Universe.of(firms).tickers)
    assert book.resolve("Lowes") == ("LOW", None) and book.resolve("Lowe\u2019s") == ("LOW", None)
    assert book.resolve("Brown-Forman") == ("BF.B", None)


def test_full_name_beats_first_word_and_news_corp_classes_fold_to_class_a() -> None:
    firms = [
        Company("FDX", "FedEx", "x", "y"),
        Company("FDXF", "FedEx Freight", "x", "y"),
        Company("NWS", "News Corp (Class B)", "x", "y"),
        Company("NWSA", "News Corp (Class A)", "x", "y"),
    ]
    book = NameBook(firms, retag.Universe.of(firms).tickers)
    assert book.resolve("FedEx") == ("FDX", None)
    assert book.resolve("News Corp") == ("NWSA", None)
    assert book.resolve("NWS") == ("NWS", None)


def test_inferred_matches_are_flagged_for_confirmation() -> None:
    firms = [
        *FIRMS,
        Company("HD", "Home Depot (The)", "x", "y"),
        Company("IT", "Gartner", "x", "y"),
        Company("CVS", "CVS Health", "x", "y"),
    ]
    book = NameBook(firms, retag.Universe.of(firms).tickers)
    assert book.lookup("Home") == ("HD", None, "first word")
    assert book.lookup("ALL") == ("ALL", None, "word ticker")
    assert book.lookup("IT") == ("IT", None, "word ticker")
    assert book.lookup("aapl") == ("AAPL", None, "ticker, other case")
    assert book.lookup("Instagram") == ("META", None, "brand")
    assert book.lookup("Aetna") == ("CVS", None, "brand")
    for exact in ("Apple", "Google", "AAPL", "J&J", "Facebook", "Microsoft Corp"):
        assert book.lookup(exact).note == "", exact


def test_row_reports_what_it_inferred() -> None:
    row = {"mentions": "ALL, Instagram, Apple", "company_labels": "Home:other"}
    new, problems, confirm = normalise_row(BOOK, row)
    assert new["mentions"] == "ALL, META, AAPL" and problems == [
        "company_labels: 'Home' unknown"  # no Home Depot in this small universe
    ]
    assert confirm == [
        "mentions: 'ALL' -> ALL (word ticker)",
        "mentions: 'Instagram' -> META (brand)",
    ]


def test_a_nickname_or_brand_names_one_ticker() -> None:
    seen: dict[str, str] = {}
    for table in (NICKNAMES, BRANDS):
        for ticker, names in table.items():
            for name in names:
                key = _key(name)
                assert seen.setdefault(key, ticker) == ticker, f"{name!r}: {seen[key]} and {ticker}"


def test_two_letter_tickers_are_not_flagged_but_words_and_letters_are() -> None:
    firms = [
        *FIRMS,
        Company("GE", "GE Aerospace", "x", "y"),
        Company("GM", "General Motors", "x", "y"),
    ]
    firms += [
        Company("F", "Ford Motor Company", "x", "y"),
        Company("ON", "ON Semiconductor", "x", "y"),
    ]
    book = NameBook(firms, retag.Universe.of(firms).tickers)
    assert book.lookup("GM") == ("GM", None, "") and book.lookup("GE") == ("GE", None, "")
    assert book.lookup("F").note == "word ticker" and book.lookup("ON").note == "word ticker"


def test_self_check_finds_a_name_hijacked_by_a_nickname(monkeypatch: pytest.MonkeyPatch) -> None:
    canonical = retag.Universe.of(FIRMS).tickers
    assert self_check(FIRMS, canonical) == []
    monkeypatch.setitem(NICKNAMES, "ALL", ("Apple",))  # Allstate "is" Apple too: now ambiguous
    failures = self_check(FIRMS, canonical)
    assert any(f.startswith("AAPL 'Apple Inc.': ambiguous") for f in failures)


def _write_review(path: Path, mentions: str) -> None:
    fields = ["review_order", "article_id", *retag.TAG_COLUMNS]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**dict.fromkeys(fields, ""), "article_id": "10", "mentions": mentions})


def test_normalise_does_not_overwrite_an_edited_copy(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415 -- the script is not a package

    review = tmp_path / retag.REVIEW_NAME
    copy = tmp_path / cli.TICKERS_NAME
    _write_review(review, "Apple")
    cli.cmd_normalise(tmp_path, universe_companies=FIRMS)
    cli.cmd_normalise(tmp_path, universe_companies=FIRMS)  # unchanged copy: allowed
    copy.write_text(
        copy.read_text(encoding="utf-8-sig").replace("AAPL", "MSFT"), encoding="utf-8-sig"
    )
    os.utime(review, (copy.stat().st_mtime - 10,) * 2)  # the copy is newer than the review file
    with pytest.raises(SystemExit):
        cli.cmd_normalise(tmp_path, universe_companies=FIRMS)
    cli.cmd_normalise(tmp_path, universe_companies=FIRMS, force=True)
    assert "AAPL" in copy.read_text(encoding="utf-8-sig")
    _write_review(review, "Microsoft")  # the review file was edited after the copy: allowed
    os.utime(review, (copy.stat().st_mtime + 10,) * 2)
    cli.cmd_normalise(tmp_path, universe_companies=FIRMS)
    assert "MSFT" in copy.read_text(encoding="utf-8-sig")


def test_guide_command_rewrites_only_the_guides(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415

    _write_review(tmp_path / retag.REVIEW_NAME, "Apple")
    before = (tmp_path / retag.REVIEW_NAME).read_bytes()
    cli.cmd_guide(tmp_path, universe_companies=FIRMS)
    assert (tmp_path / retag.REVIEW_NAME).read_bytes() == before
    assert "column,company_labels" in (tmp_path / "category_gold_v2_guide.csv").read_text(
        encoding="utf-8-sig"
    )


def test_a_file_open_in_another_program_gives_a_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415

    def locked(*_a: object, **_k: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "open", locked)
    with pytest.raises(SystemExit, match="close it"):
        cli.write_csv(tmp_path / "guide.csv", [{"a": "1"}])


class _Clock:
    """A clock that moves one second per call, so two migrations get two names."""

    def __init__(self) -> None:
        self.n = 0

    def now(self) -> Any:
        import datetime  # noqa: PLC0415

        self.n += 1
        return datetime.datetime(2026, 10, 30, 12, 0, self.n)


def test_cmd_migrate_never_replaces_an_earlier_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _script_cli()
    monkeypatch.setattr(cli, "datetime", _Clock())
    fields = ["article_id", "scope", "companies", "company_labels", "mentions"]
    path = tmp_path / "work_.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "article_id": "1",
                "scope": "multi",
                "companies": "Amazon; Toyota",
                "company_labels": "Amazon:other",
                "mentions": "Boeing; GM",
            }
        )
    before = path.read_bytes()
    cli.cmd_migrate(tmp_path, file="work_.csv")
    shown = capsys.readouterr().out
    assert path.read_bytes() == before
    first = tmp_path / "work_new_format_2026-10-30_120001.csv"  # no double underscore
    (row,) = list(csv.DictReader(first.open(encoding="utf-8-sig")))
    assert "companies" not in row
    assert row["company_labels"] == "Amazon:other, Toyota" and row["mentions"] == "Boeing, GM"
    assert "'Toyota' was in companies" in shown
    assert f"Continue in {first.name} and stop editing work_.csv" in shown
    first.write_text(first.read_text(encoding="utf-8-sig") + "worked in\n", encoding="utf-8-sig")
    worked = first.read_bytes()
    cli.cmd_migrate(tmp_path, file="work_.csv")  # a second run: a new name, the first is intact
    assert first.read_bytes() == worked
    assert (tmp_path / "work_new_format_2026-10-30_120002.csv").exists()


def test_a_missing_review_file_names_the_similar_ones(tmp_path: Path) -> None:
    cli = _script_cli()
    with pytest.raises(SystemExit) as bare:
        cli.review_path(tmp_path)
    assert "not found" in str(bare.value) and "similar" not in str(bare.value)
    (tmp_path / "category_gold_v2_review_.csv").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as hint:
        cli.review_path(tmp_path)
    assert "category_gold_v2_review_.csv" in str(hint.value) and "--file NAME" in str(hint.value)
    assert cli.review_path(tmp_path, "category_gold_v2_review_.csv").exists()


def test_a_name_with_a_comma_survives_normalise_end_to_end(tmp_path: Path) -> None:
    cli = _script_cli()
    firms = [*FIRMS, Company("BKNG", "Booking Holdings Inc.", "x", "y")]
    review = tmp_path / retag.REVIEW_NAME
    fields = ["review_order", "article_id", *retag.TAG_COLUMNS]
    with review.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                **dict.fromkeys(fields, ""),
                "article_id": "10",
                "scope": "multi",
                "company_labels": "Booking Holdings, Inc.:other, Apple:uncertain",
                "mentions": "Microsoft, Corp., Google",
            }
        )
    cli.cmd_normalise(tmp_path, universe_companies=firms)
    (row,) = list(csv.DictReader((tmp_path / cli.TICKERS_NAME).open(encoding="utf-8-sig")))
    assert row["company_labels"] == "BKNG:other, AAPL:uncertain"
    assert row["mentions"] == "MSFT, GOOG"


def _script_cli() -> Any:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415

    return cli
