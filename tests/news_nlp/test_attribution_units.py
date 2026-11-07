"""news_nlp.attribution_units: the footer and the units the attribution reads (WI18 T-148b4, D27).
No model: the splitting, the clauses and the footer rules are what is tested."""

from __future__ import annotations

from collections.abc import Sequence

from news_nlp import attribution_model as am
from news_nlp import attribution_units as au
from news_nlp.triage import Company, CompanyResolver

UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
    Company("ORCL", "Oracle Corporation", "Information Technology", "Systems Software"),
    Company("KHC", "Kraft Heinz", "Consumer Staples", "Packaged Foods"),
    Company("TWC", "Time Warner Cable", "Communication Services", "Cable"),
]
RESOLVER = CompanyResolver(UNIVERSE)
NAMES = {c.ticker: c.name for c in UNIVERSE}
STORY = "Apple rose after the quarter. Microsoft fell on weak cloud growth. " * 4


def test_a_list_with_one_line_per_company_is_one_unit_per_line() -> None:
    body = (
        "Stocks to watch:\n- Apple fell 3%.\n- Microsoft rose 2%; Oracle slid 3%.\n- Nothing else."
    )
    assert au.attribution_units(body, RESOLVER) == [
        "Stocks to watch:",
        "Apple fell 3%.",
        "Microsoft rose 2%",
        "Oracle slid 3%.",
        "Nothing else.",
    ]


def test_a_sentence_with_two_statements_is_cut_between_the_companies() -> None:
    assert au._clauses("Apple shares rose 3%, Microsoft fell 2% on cloud news.", RESOLVER) == [
        "Apple shares rose 3%,",
        "Microsoft fell 2% on cloud news.",
    ]


def test_companies_named_together_stay_in_one_unit() -> None:
    for text in (
        "Amazon and Apple announced a deal.",
        "Kraft Heinz, Time Warner Cable - these stocks are among the new stakes.",
        "Apple and Microsoft rose.",
    ):
        assert au._clauses(text, RESOLVER) == [text]


def test_a_sentence_of_one_company_is_not_cut() -> None:
    text = "Apple rose 3%, its best day in a month, while the market slid."
    assert au._clauses(text, RESOLVER) == [text]


def test_the_disclosure_block_and_the_credit_are_dropped_from_the_end() -> None:
    body = (
        STORY
        + "\n- CNBC's Michael Bloom contributed to this report.\n"
        + "Disclosures:\nCortes owns (AAPL); Adami owns (MSFT)\nFor Brian Kelly:\nOwn (ORCL)"
    )
    kept = au.strip_footer(body)
    assert kept.strip() == STORY.strip()
    assert "Disclosures" not in kept and "Bloom" not in kept


def test_a_disclosure_in_the_first_part_is_text_not_a_footer() -> None:
    body = "Disclosures: the company said so.\n" + STORY
    assert au.strip_footer(body) == body


def test_more_from_drops_its_links_and_not_the_story_after_them() -> None:
    story = "Oracle announced a deal with a large retailer that will run for three years. " * 3
    body = STORY + "\nMore From CNBC.com\n- Get Quotes\n- Futures Data\n" + story
    kept = au.strip_footer(body)
    assert "Get Quotes" not in kept and kept.rstrip().endswith(story.strip())


def test_read_more_links_are_dropped_wherever_they_are() -> None:
    body = STORY + "\n(Read more: Another story title)\n" + STORY
    assert "Read more" not in au.strip_footer(body)


def test_a_body_that_would_be_left_with_almost_nothing_keeps_its_footer() -> None:
    body = "Short story.\nSubscribe to our newsletter"
    assert au.strip_footer(body) == body


def test_the_trailing_site_lines_are_dropped() -> None:
    body = STORY + "\nQuestions? Comments? desk@example.com\nSubscribe to the newsletter\n"
    assert au.strip_footer(body).strip() == STORY.strip()


class _Scorer:
    """Gives every pair the same score: the units, not the model, are under test."""

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, pairs: Sequence[am.Pair]) -> list[float]:
        return [self.value] * len(pairs)


def test_with_units_a_list_does_not_hand_every_company_the_whole_list() -> None:
    body = "Stocks to watch:\n- Apple fell 3%.\n- Microsoft rose 2%.\n- Oracle slid 3%.\n" + STORY
    plain = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0))
    split = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), units=True)
    assert "Oracle slid 3%." in plain["AAPL"]  # one "sentence" naming the three: all three got it
    assert "Oracle slid 3%." not in split["AAPL"] and "Microsoft rose 2%." not in split["AAPL"]
    assert split["ORCL"].startswith("Oracle slid 3%.")


def test_the_default_reading_is_unchanged() -> None:
    body = "Apple rose.\nMicrosoft fell. Nothing else."
    assert am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0)) == am.model_company_texts(
        body, RESOLVER, NAMES, _Scorer(0.0), units=False
    )
