"""news_nlp.company_names: names, nicknames and tickers -> canonical tickers, never a guess."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from news_nlp import retag
from news_nlp.company_names import NICKNAMES, NameBook, normalise_row
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
        "companies": "Apple; the iPhone maker; Google",
        "mentions": "Alphabet; Microsoft Corp; Bank",
        "company_labels": "Apple:product_innovation; Google: other; Tesla:other",
    }
    new, problems = normalise_row(BOOK, row)
    assert new["companies"] == "AAPL; GOOG; the iPhone maker"
    assert new["mentions"] == "GOOG; MSFT; Bank"  # Alphabet already folds to GOOG
    assert new["company_labels"] == "AAPL:product_innovation; GOOG:other; Tesla:other"
    assert problems == [
        "companies: 'the iPhone maker' unknown",
        "mentions: 'Bank' ambiguous: BAC, BK",
        "company_labels: 'Tesla' unknown",
    ]


def test_row_conflicting_labels_and_missing_label_are_reported() -> None:
    row = {
        "companies": "",
        "mentions": "",
        "company_labels": "Google:other; Alphabet:uncertain; Apple",
    }
    new, problems = normalise_row(BOOK, row)
    assert new["company_labels"] == "GOOG:other; Apple"
    assert any("given 'other' and 'uncertain'" in p for p in problems)
    assert any("has no label" in p for p in problems)


def test_empty_cells_stay_empty() -> None:
    new, problems = normalise_row(BOOK, {"companies": "", "mentions": "", "company_labels": ""})
    assert new == {"companies": "", "mentions": "", "company_labels": ""} and problems == []


def test_cli_normalise_then_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415 -- the script is not a package

    base = {c: "" for c in ("review_order", "article_id", *retag.TAG_COLUMNS)}
    typed = {
        **base,
        "article_id": "10",
        "scope": "single",
        "companies": "Apple",
        "company_labels": "Apple:product_innovation",
        "mentions": "Facebook; Nobody Inc",
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
    assert out[0]["companies"] == "AAPL" and out[0]["mentions"] == "META; Nobody Inc"
    assert (tmp_path / retag.REVIEW_NAME).read_text(encoding="utf-8-sig").count("Apple") >= 2
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
