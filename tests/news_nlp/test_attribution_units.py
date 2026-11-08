"""news_nlp.attribution_units: the footer and the units the attribution reads (WI18 T-148b4, D27).
No model: the splitting, the clauses and the footer rules are what is tested."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

import pytest

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


def test_a_short_sentence_after_more_from_is_the_story_not_a_link() -> None:
    body = (
        STORY
        + '\nMore From CNBC.com\n- Get Quotes\n- Futures Data\n"We will win," he said.\n'
        + STORY
    )
    kept = au.strip_footer(body)
    assert "Get Quotes" not in kept and "We will win" in kept


@pytest.mark.parametrize(
    "last",
    [
        "Subscribers rose 5% in the quarter, the company said, as the new plan took hold.",
        "Comments from the company's chief executive suggested that growth would continue.",
        "Questions remain about the pace of the recovery in the second half of the year.",
        "Disclaimers in the filing say the figures are unaudited and may change later on.",
        "Source documents show the deal was agreed last week, according to people familiar.",
        "Copyrighted material in the archive was removed after the ruling in the case.",
    ],
)
def test_a_story_paragraph_that_starts_like_a_footer_label_is_kept(last: str) -> None:
    body = STORY + "\n" + last
    assert au.strip_footer(body).rstrip().endswith(last)


@pytest.mark.parametrize(
    "line",
    [
        "Questions? Comments?",
        "Disclaimer:",
        "Subscribe",
        "Subscribe to our newsletter",
        "Source: Reuters",
    ],
)
def test_a_label_line_at_the_end_is_dropped(line: str) -> None:
    assert au.strip_footer(STORY + "\n" + line).strip() == STORY.strip()


def test_a_long_line_is_a_paragraph_even_if_it_starts_with_a_label() -> None:
    long = "Source: " + "the filing says so and the company agreed with it. " * 8
    assert au.strip_footer(STORY + "\n" + long).rstrip().endswith(long.strip())


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


def test_with_units_a_company_named_only_in_the_footer_is_not_a_candidate() -> None:
    story = (
        "Microsoft grew its cloud unit this quarter. Azure demand stayed strong for the third "
        "quarter in a row, the company said on Tuesday. Analysts expect the trend to continue. "
    ) * 4
    body = story + "\nDisclosures:\nThe author owns (ORCL)"
    scorer = _Scorer(0.9)  # every candidate scores 0.9
    plain = am.model_company_texts(body, RESOLVER, NAMES, scorer)
    units = am.model_company_texts(body, RESOLVER, NAMES, scorer, reading="units")
    assert "ORCL" in plain  # the whole body names Oracle in the footer: it is a candidate
    assert set(units) == {"MSFT"}  # the story names only Microsoft: Oracle is not asked about


def _audit_script(monkeypatch: pytest.MonkeyPatch) -> object:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "audit_footer_2026_11_08", scripts / "audit_footer_2026_11_08.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_footer_audit_counts_the_changed_bodies_and_shows_what_was_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _audit_script(monkeypatch)
    unchanged = STORY
    light = STORY * 4 + "\nQuestions? Comments?"
    heavy = STORY * 10 + "\nDisclosures:\n" + "Cortes owns (AAPL); Adami owns (MSFT)\n" * 33
    result = script.audit({1: unchanged, 2: light, 3: heavy}, seed=1, show=5, sample=5)  # type: ignore[attr-defined]
    assert (result["bodies"], result["changed"], result["lose more than a fifth"]) == (3, 2, 1)
    assert [a for a, _ in result["heavy"]] == [3] and "Cortes owns" in result["heavy"][0][1]
    assert [a for a, _ in result["sample"]] == [2] and result["sample"][0][
        1
    ] == "Questions? Comments?"


@pytest.mark.parametrize(
    "last",
    [
        "Reporting by Bloomberg showed that deliveries fell for the third month in a row, "
        "according to people familiar with the matter at two of the largest suppliers.",
        "Copyright holders argued in court that the new rules would hurt creators and publishers "
        "of every size, and asked the judge to block them until the appeal is decided.",
        "Sign up fees at the largest gyms rose sharply last year, the survey found, as owners "
        "passed on higher rents and staff costs to new members across the whole country.",
    ],
)
def test_a_long_paragraph_that_starts_like_a_credit_is_the_story(last: str) -> None:
    assert au.strip_footer(STORY + "\n" + last).rstrip().endswith(last)


@pytest.mark.parametrize(
    "line",
    [
        "Reporting by Jane Doe; Editing by John Roe",
        "Copyright 2020 Reuters. All rights reserved.",
        "Copyright \u00a9 2020 The Associated Press. All rights reserved. "
        + "This material may not be published. " * 8,
        "Source: Company filings",
    ],
)
def test_the_credits_and_notices_are_still_footers(line: str) -> None:
    assert au.strip_footer(STORY + "\n" + line).strip() == STORY.strip()


def test_a_dateline_or_a_label_in_capitals_is_not_a_header_that_takes_the_line_away() -> None:
    for line in (
        "DETROIT — Apple's new store sales rose 7.7% from a year earlier.",
        "NEW YORK — Microsoft is handing its tools over to a partner.",
        "REFINING - This is the weak spot for Oracle this quarter.",
    ):
        assert au.header_owners(line, RESOLVER) is None, line
    # a company in capitals is still a header, and a company outside the universe in mixed case
    assert au.header_owners("AAPL - shares rose after the report.", RESOLVER) == ("AAPL",)
    assert au.header_owners("Canadian Pacific — Its chief met a rival.", RESOLVER) == ()
    body = "DETROIT — Apple sold more phones.\n" + STORY
    texts = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units3")
    assert texts["AAPL"].startswith("DETROIT — Apple sold more phones.")  # not lost


def test_a_reading_that_is_not_one_of_the_four_is_an_error() -> None:
    with pytest.raises(ValueError, match="reading 'unit3'"):
        am.model_company_texts(STORY, RESOLVER, NAMES, _Scorer(0.0), reading="unit3")  # type: ignore[arg-type]
    for reading in am.READINGS:
        am.model_company_texts(STORY, RESOLVER, NAMES, _Scorer(0.0), reading=reading)  # type: ignore[arg-type]
