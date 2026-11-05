"""news_nlp.company_texts: the text of each company in a one-segment-per-company body (WI18 step 4)."""

from __future__ import annotations

from news_nlp import company_texts, retag
from news_nlp.triage import Company, CompanyResolver

UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
    Company("ORCL", "Oracle Corporation", "Information Technology", "Systems Software"),
]
RESOLVER = CompanyResolver(UNIVERSE)


def test_structured_body_gives_one_text_per_company_in_order() -> None:
    body = (
        "Stocks making the biggest moves midday:\n"
        "Apple - Shares rose 3% after the company raised its dividend.\n"
        "It also announced a larger buyback than analysts expected.\n"
        "Microsoft - Shares fell 2% as cloud growth slowed in the quarter.\n"
        "Oracle - Shares jumped 5% on a new contract with the Pentagon."
    )
    found = company_texts.split_company_texts(body, RESOLVER)
    assert found is not None
    assert list(found) == ["AAPL", "MSFT", "ORCL"]
    assert "larger buyback" in found["AAPL"]  # the continuation line stays with Apple
    assert "cloud growth" in found["MSFT"] and "Apple" not in found["MSFT"]
    assert "Stocks making" not in found["AAPL"]  # the introduction belongs to no company


def test_narrative_body_is_left_to_the_exception_path() -> None:
    body = (
        "Apple and Microsoft are fighting over the same customers in the enterprise market.\n"
        "Analysts say the rivalry will intensify over the next year as budgets tighten.\n"
        "Neither company commented on the report published on Tuesday morning."
    )
    assert company_texts.split_company_texts(body, RESOLVER) is None


def test_one_company_is_not_a_multi_body() -> None:
    body = "Apple - Shares rose 3% after the dividend news.\nApple - More gains followed later."
    assert company_texts.split_company_texts(body, RESOLVER) is None


def test_sentiment_tagging_format_keeps_the_spanish_pair_after_text() -> None:
    row = dict.fromkeys(retag.SENTIMENT_TAGGING_COLUMNS, "x")
    cols = retag.SENTIMENT_TAGGING_COLUMNS
    assert cols.index("spanish_title") == cols.index("text") + 1
    assert cols.index("spanish_text") == cols.index("text") + 2
    assert retag.tagging_format_problems([row], cols) == []
    assert retag.tagging_format_problems([row])  # it is not the scope format
    without = {k: v for k, v in row.items() if k != "kind"}
    assert any(
        "not the tagging format" in p for p in retag.tagging_format_problems([without], cols)
    )
