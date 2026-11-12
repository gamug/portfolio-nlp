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
    assert "BE" in plain and "UNP" not in plain  # "Union" is a stop word; "Bloom" is the guard's
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
    assert au.header_owners("Apple - Shares rose after Oracle fell.", RESOLVER) == ("AAPL",)


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
    assert "BE" not in one and "BE" not in two  # "Bloom" is a person: not a company
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
    assert au.header_owners("AAPL - Shares rose after the report.", RESOLVER) == ("AAPL",)
    assert au.header_owners("Canadian Pacific — Its chief met a rival.", RESOLVER) == ()
    body = "DETROIT — Apple sold more phones.\n" + STORY
    texts = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units3")
    assert texts["AAPL"].startswith("DETROIT — Apple sold more phones.")  # not lost


def test_a_reading_that_is_not_one_of_the_readings_is_an_error() -> None:
    with pytest.raises(ValueError, match="reading 'unit3'"):
        am.model_company_texts(STORY, RESOLVER, NAMES, _Scorer(0.0), reading="unit3")  # type: ignore[arg-type]
    for reading in am.READINGS:
        am.model_company_texts(STORY, RESOLVER, NAMES, _Scorer(0.0), reading=reading)  # type: ignore[arg-type]


def test_a_dash_inside_a_sentence_and_a_speaker_label_are_not_headers() -> None:
    for line in (
        "In Miami Beach \u2014 as in New York and San Diego \u2014 rentals are limited.",
        "Today, Apple \u2014 like the industry itself \u2014 is challenged by cheap rivals.",
        "Enter Prime Air \u2014 the air cargo service Apple introduced last summer.",
        "Volkswagen - synonymous with the classic camper - has spawned a sub-industry.",
    ):
        assert au.header_owners(line, RESOLVER) is None, line
    assert au.starts_a_sentence("Shares rose 3%.") and au.starts_a_sentence(
        '"Shares rose," he said'
    )
    assert au.starts_a_sentence("3M raised its outlook.")
    assert not au.starts_a_sentence("as in New York")
    assert not au.starts_a_sentence("") and not au.starts_a_sentence("   ")  # nothing starts
    # a header whose text starts with a capital is still a header
    assert au.header_owners("Apple \u2014 Shares rose after the report.", RESOLVER) == ("AAPL",)


COMMON = CompanyResolver(
    [
        Company("ROST", "Ross Stores, Inc.", "Consumer Discretionary", "Apparel Retail"),
        Company("WMB", "Williams Companies, Inc.", "Energy", "Pipelines"),
        Company("XYZ", "Block, Inc.", "Financials", "Transaction Processing"),
        Company("TRV", "Travelers Companies, Inc. (The)", "Financials", "Insurance"),
        Company("WFC", "Wells Fargo & Company", "Financials", "Banks"),
        Company("FRT", "Federal Realty Investment Trust", "Real Estate", "Retail REITs"),
    ]
)


def _spans(body: str, resolver: au.GuardedResolver) -> list[tuple[str, str]]:
    return [(body[a:b], t) for a, b, t in resolver.mention_spans(body)]


def test_the_guard_does_not_take_a_one_word_name_that_is_a_surname_or_a_noun_for_a_company() -> (
    None
):
    """The names Williams, Block and Travelers are names, not first words, and still need the full
    name, the short form or a ticker form somewhere in the body (PR #155, round 3)."""
    body = "Commerce Secretary Wilbur Ross and Williams said it will block trade. Wells reported."
    guarded = au.GuardedResolver(COMMON, body)
    assert _spans(body, guarded) == []  # raw spans: Ross, Williams, Wells
    assert set(guarded.resolve(body)) == set()
    body = "Travelers fell. A Block of shares and Block the exit. Doug Parker said so."
    guarded = au.GuardedResolver(COMMON, body)
    assert _spans(body, guarded) == []
    assert guarded.resolve(body) == {}


def test_the_guard_keeps_them_once_the_body_names_the_company_otherwise() -> None:
    for body, ticker in (
        ("Williams Companies rose. Williams said so.", "WMB"),
        ("Block, Inc. rose. Block said so.", "XYZ"),
        ("Travelers (NYSE: TRV) fell. Travelers said so.", "TRV"),
        ("Ross Stores rose. Ross said so.", "ROST"),
        ("Wells Fargo & Company rose. Wells said so.", "WFC"),
    ):
        guarded = au.GuardedResolver(COMMON, body)
        assert ticker in {t for _, _, t in guarded.mention_spans(body)}, body
        assert guarded.resolve(body)[ticker] >= 2, body


def test_the_guard_counts_the_short_form_of_a_name_as_a_mention_in_itself() -> None:
    body = "Federal Realty rose on Monday."
    assert _spans(body, au.GuardedResolver(COMMON, body)) == [("Federal Realty", "FRT")]


PARAGRAPH = (
    "Check out the companies making headlines. Apple \u2013 Shares rose 3% after the quarter. "
    "Wendy's (WEN) \u2013 Trian raised its stake to 23%. Microsoft \u2013 Shares fell 2% on weak cloud growth."
)


def test_units4_gives_the_sentence_of_a_company_outside_the_universe_to_nobody() -> None:
    # a scorer that likes every pair: with `units` the Wendy's sentence goes to the best candidate
    old = am.model_company_texts(PARAGRAPH, RESOLVER, NAMES, _Scorer(0.9), reading="units")
    new = am.model_company_texts(PARAGRAPH, RESOLVER, NAMES, _Scorer(0.9), reading="units4")
    assert "Wendy's" in " ".join(old.values())
    assert "Wendy's" not in " ".join(new.values())
    assert set(new) == {"AAPL", "MSFT"}
    assert "Microsoft" not in new["AAPL"] and "Apple" not in new["MSFT"]


def test_units4_keeps_a_header_with_its_text_when_the_sentence_names_another_company() -> None:
    body = "Apple \u2013 Shares rose after Microsoft raised its price target on the stock."
    old = au.attribution_pieces(body, RESOLVER)
    new = au.attribution_pieces(body, RESOLVER, headers=True, named_after=True, per_sentence=True)
    assert old[0].text == "Apple \u2013"  # the fault: the header cut off its text
    assert [u.text for u in new] == [body]
    assert new[0].owners == ("AAPL", "MSFT")  # the header's company, and the one the text names


def test_units4_a_header_in_the_middle_of_a_line_opens_a_new_owner_for_what_follows() -> None:
    body = "Apple \u2013 Shares rose 3%. Microsoft \u2013 Shares fell 2%. It was the worst day since May."
    pieces = au.attribution_pieces(
        body, RESOLVER, headers=True, named_after=True, per_sentence=True
    )
    assert [u.owners for u in pieces] == [("AAPL",), ("MSFT",), ("MSFT",)]
    # without per_sentence the whole line belongs to the first header
    old = au.attribution_pieces(body, RESOLVER, headers=True)
    assert {u.owners for u in old} == {("AAPL",)}


def test_units4_leaves_a_line_without_a_header_to_the_attribution() -> None:
    body = "Apple rose 3% and Microsoft fell 2%.\nThe market was quiet."
    pieces = au.attribution_pieces(body, RESOLVER, headers=True, per_sentence=True)
    assert all(u.owners is None for u in pieces)


def _h5(line: str) -> tuple[str, ...] | None:
    return au.header_owners(line, RESOLVER, tickers_in_head=True, units5=True)


def test_units5_reads_a_short_name_in_capitals_as_a_company_and_still_not_a_dateline() -> None:
    for line in (
        "SAP — U.S.-listed shares of the software provider gained 3%.",
        "UBS \N{EN DASH} UBS reported a better-than-expected profit.",
        "BP (BP) \N{EN DASH} BP reported an unexpected profit.",
        "AT & T — Stock in the telecommunications firm pulled back 2%.",
    ):
        assert _h5(line) == (), line  # a company outside the universe
        assert au.header_owners(line, RESOLVER, tickers_in_head=True) is None, (
            line
        )  # units4 does not read them
    for line in (
        "DETROIT — Apple's new store sales rose 7.7% from a year earlier.",
        "NEW YORK — Microsoft is handing its tools over to a partner.",
        "REFINING - This is the weak spot for Oracle this quarter.",
        "ROME — Apple opened a store.",
    ):
        assert _h5(line) is None, line
    assert _h5("AAPL - Shares rose after the report.") == ("AAPL",)


def test_units5_reads_a_group_label_as_a_line_of_nobody_and_not_a_sentence() -> None:
    for line in (
        "Memory stocks — Shares of memory companies also moved higher.",
        "Cruise lines — Concerns over fuel costs weighed on these stocks.",
        "China electric vehicle stocks — Shares rose.",
        "Jack in the Box — The fast-food stock shed 1% after a downgrade.",
    ):
        assert _h5(line) == (), line
    for line in (
        "Shares rose — Apple led the gains.",  # a verb, not a group noun
        "The company's bet on Reels is facing hurdles — Instagram users are spending less.",
        "Stocks fell, he said — and Apple fell.",
    ):
        assert _h5(line) is None, line
    assert _h5("Apple stocks — Shares rose.") == ("AAPL",)  # a universe name stays a header of it


def test_units5_reads_a_period_inside_a_name_and_a_lowercase_text_after_a_ticker() -> None:
    for line in (
        "Amazon.com (AMZN) \N{EN DASH} Amazon warehouse workers will vote.",
        "Box Inc. (BOX) \N{EN DASH} Box beat estimates by 6 cents a share.",
        "Fox Corp. (FOXA) \N{EN DASH} Fox increased its repurchase program.",
    ):
        assert _h5(line) == (), line
        assert au.header_owners(line, RESOLVER, tickers_in_head=True) is None, (
            line
        )  # the fault of units4
    assert _h5("Rockwell Automation (ROK) — weaker quarter.") == ()
    assert _h5("American Express (AXP) — solid momentum continues into April.") == ()
    assert (
        _h5("Apple — solid momentum continues.") is None
    )  # a company without its ticker: a prose dash
    assert _h5("Apple Inc. (AAPL) — solid momentum.") == ("AAPL",)
    # lowercase after the dash with no company in the head is still a dash inside a sentence
    assert _h5("In Miami Beach — as in New York and San Diego — rentals are limited.") is None
    assert (
        _h5("Volkswagen - synonymous with the classic camper - has spawned a sub-industry.") is None
    )


def test_units5_gives_the_sentences_after_a_header_of_nobody_to_the_companies_they_name() -> None:
    body = (
        "Airlines — The group fell on Monday. Apple slipped 2%, while Oracle fell 3%. "
        "Wendy's (WEN) \N{EN DASH} Trian raised its stake. Microsoft \N{EN DASH} Shares fell 2%."
    )
    kw = {"headers": True, "named_after": True, "per_sentence": True}
    old = au.attribution_pieces(body, RESOLVER, **kw)  # type: ignore[arg-type]
    new = au.attribution_pieces(body, RESOLVER, units5=True, **kw)  # type: ignore[arg-type]
    assert [u.owners for u in old] == [(), (), (), ("MSFT",)]  # units4: Apple and Oracle are lost
    assert [u.owners for u in new] == [(), ("AAPL", "ORCL"), (), ("MSFT",)]
    # a header line that names a company inside its own sentence still gives it to nobody
    own = au.attribution_pieces(
        "Fermi — Shares fell after Apple said so.",
        RESOLVER,
        units5=True,
        **kw,  # type: ignore[arg-type]
    )
    assert own[0].owners == ()
    # a sentence after a header of nobody that names no company stays with nobody
    plain = au.attribution_pieces(
        "Fermi — Shares fell. It was co-founded by Rick Perry.",
        RESOLVER,
        units5=True,
        **kw,  # type: ignore[arg-type]
    )
    assert [u.owners for u in plain] == [(), ()]


def test_units5_end_to_end_gives_the_roundup_companies_their_text_and_keeps_units4_apart() -> None:
    body = (
        "Check out the companies making headlines. "
        "Wendy's (WEN) \u2013 Trian raised its stake. "
        "Cruise lines \u2014 Concerns weighed on these stocks Monday. "
        "Apple shares tumbled nearly 5%, and Oracle slid 3%."
    )
    four = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units4")
    five = am.model_company_texts(body, RESOLVER, NAMES, _Scorer(0.0), reading="units5")
    assert set(four) == set()  # "Cruise lines" is no header: the sentence keeps Wendy's "nobody"
    assert set(five) == {"AAPL", "ORCL"}
    assert "Wendy's" not in " ".join(five.values())


def test_units5_keeps_the_prose_dash_guard_for_a_company_without_its_ticker() -> None:
    """Review of #172 (B1): on 3,000 bodies 9 of 14 lowercase-after-the-dash headers were prose dashes."""
    for line in (
        "Today, Apple \N{EM DASH} like the industry itself \N{EM DASH} is challenged by cheap rivals.",
        "Apple \N{EM DASH} up 3% on the day \N{EM DASH} led the gains.",
        "Apple - synonymous with the classic phone - has spawned a sub-industry.",
        "Microsoft - which owns LinkedIn - said revenue grew.",
        "Apple \N{EN DASH} and its suppliers \N{EN DASH} fell on Monday.",
    ):
        assert _h5(line) is None, line
        assert au.header_owners(line, RESOLVER, tickers_in_head=True) is None, line
    # with the ticker in brackets, as the roundups write it, the lowercase text is read
    assert _h5("Apple (AAPL) \N{EM DASH} up 3% on the day.") == ("AAPL",)
    assert _h5("Whirlpool (WHR) \N{EM DASH} double digit decline in orders.") == ()


def test_units5_does_not_take_an_agency_or_a_wire_label_in_capitals_for_a_company() -> None:
    """Review of #172 (D1): `FDA - The agency approved Apple's device` must not send Apple's sentence to nobody."""
    for head in (
        "FDA",
        "SEC",
        "OPEC",
        "NASA",
        "CEO",
        "NEWS",
        "LIVE",
        "US",
        "EU",
        "IPO",
        "ETF",
        "FED",
    ):
        line = f"{head} \N{EM DASH} The agency approved Apple's new device on Monday."
        assert _h5(line) is None, head
    assert (
        _h5("FMC \N{EM DASH} Shares rose 3% after the quarter.") == ()
    )  # a company outside the universe
    assert (
        _h5("FDA (FDA) \N{EM DASH} Shares rose 3% after the quarter.") == ()
    )  # a ticker makes it one
    kw = {"headers": True, "named_after": True, "per_sentence": True}
    pieces = au.attribution_pieces(
        "FDA \N{EM DASH} The agency approved Apple's new device on Monday.",
        RESOLVER,
        units5=True,
        **kw,  # type: ignore[arg-type]
    )
    assert [u.owners for u in pieces] == [
        None
    ]  # no header: the attribution gives Apple its sentence
