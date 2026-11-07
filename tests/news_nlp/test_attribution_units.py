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
    Company("BE", "Bloom Energy", "Industrials", "Electrical Equipment"),
    Company("UNP", "Union Pacific Corporation", "Industrials", "Railroads"),
    Company("NSC", "Norfolk Southern Corporation", "Industrials", "Railroads"),
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
    split = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units")
    assert "Oracle slid 3%." in plain["AAPL"]  # one "sentence" naming the three: all three got it
    assert "Oracle slid 3%." not in split["AAPL"] and "Microsoft rose 2%." not in split["AAPL"]
    assert split["ORCL"].startswith("Oracle slid 3%.")


def test_the_default_reading_is_unchanged() -> None:
    body = "Apple rose.\nMicrosoft fell. Nothing else."
    assert am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0)) == am.model_company_texts(
        body, RESOLVER, NAMES, _Scorer(0.0), reading="sentences"
    )


def test_the_guard_drops_a_company_named_only_by_the_first_word_of_its_name() -> None:
    body = "CNBC's Michael Bloom said Apple rose. The Union Square funds bought Microsoft."
    plain = {t for _, _, t in RESOLVER.mention_spans(body)}
    guarded = au.GuardedResolver(RESOLVER, body)
    assert {"BE", "UNP"} <= plain  # the resolver the scope rules use takes both for companies
    assert {t for _, _, t in guarded.mention_spans(body)} == {"AAPL", "MSFT"}
    assert set(guarded.resolve(body)) == {"AAPL", "MSFT"}


def test_the_guard_keeps_a_short_form_when_the_full_name_is_in_the_body() -> None:
    body = "Bloom Energy rose 4%. Later, Bloom said it would hire."
    guarded = au.GuardedResolver(RESOLVER, body)
    assert [t for _, _, t in guarded.mention_spans("Bloom said it would hire.")] == ["BE"]
    assert [t for _, _, t in guarded.mention_spans(body)] == ["BE", "BE"]


def test_the_resolver_the_scope_rules_use_is_not_changed_by_the_flag() -> None:
    text = "Bloom said Apple rose."
    assert RESOLVER.mention_spans(text) == RESOLVER.mention_spans(text, aliases=True)
    assert [t for _, _, t in RESOLVER.mention_spans(text, aliases=False)] == ["AAPL"]


def test_a_line_headed_by_names_belongs_to_them_alone() -> None:
    line = "Canadian Pacific \u2014 Its chief met with Norfolk Southern CEO about a merger."
    assert au.header_owners(line, RESOLVER) == ()  # a company outside the universe heads it
    line = "Apple, Microsoft \u2014 Evercore downgraded the stocks; Oracle was cut as well."
    assert au.header_owners(line, RESOLVER) == ("AAPL", "MSFT")
    assert au.header_owners("Apple - shares rose after Oracle fell.", RESOLVER) == ("AAPL",)


def test_a_sentence_is_not_taken_for_a_header() -> None:
    for line in (
        "Microsoft rose while Apple - up 3% - fell.",
        "Shares rose, he said - and Apple fell.",
        "A long first part of the line that goes on and on and on and on and on and on - Apple.",
        "Apple said it would buy back shares.",
    ):
        assert au.header_owners(line, RESOLVER) is None, line


def test_pieces_of_a_header_line_carry_their_owners_only_with_the_flag() -> None:
    body = "Apple, Microsoft \u2014 Evercore cut the stocks. It also cut Oracle.\nOracle fell 3%."
    plain = au.attribution_pieces(body, RESOLVER)
    assert all(p.owners is None for p in plain)
    headed = au.attribution_pieces(body, RESOLVER, headers=True)
    assert [p.owners for p in headed] == [("AAPL", "MSFT"), ("AAPL", "MSFT"), None]


def test_units2_gives_a_header_line_to_its_names_and_ignores_a_common_word() -> None:
    body = (
        "Canadian Pacific \u2014 Its chief met with Norfolk Southern CEO.\n"
        "Apple \u2014 Shares rose after Michael Bloom said so.\n" + STORY
    )
    two = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units2")
    one = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units")
    assert "NSC" in one and "NSC" not in two  # the line is about Canadian Pacific, not Norfolk
    assert "BE" in one and "BE" not in two  # "Bloom" is a person
    assert two["AAPL"].startswith("Apple \u2014 Shares rose after Michael Bloom said so.")


def test_units3_gives_a_header_line_also_to_the_companies_its_text_names() -> None:
    body = (
        "Apple — Shares rose. Microsoft added 2% and Oracle fell 3%.\n"
        "Canadian Pacific — Its chief met with Norfolk Southern CEO.\n" + STORY
    )
    two = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units2")
    three = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units3")
    sentence = "Microsoft added 2% and Oracle fell 3%."
    assert "ORCL" not in two  # units2: the line is Apple's alone
    assert three["ORCL"] == sentence and sentence in three["MSFT"]
    assert sentence in three["AAPL"] and three["AAPL"].startswith("Apple \u2014 Shares rose.")
    assert "NSC" not in three  # a line headed by somebody else is still about nobody


def test_units3_pieces_carry_the_header_and_the_companies_they_name() -> None:
    body = "Apple — Shares rose. Microsoft added 2%."
    pieces = au.attribution_pieces(body, RESOLVER, headers=True, named_after=True)
    assert [p.owners for p in pieces] == [("AAPL",), ("AAPL", "MSFT")]
