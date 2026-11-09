"""news_nlp.triage: body hashing, company resolution and scope candidates (Work item 18 P1)."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from news_nlp import triage
from news_nlp.triage import Company, CompanyResolver

UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
    Company("ORCL", "Oracle Corporation", "Information Technology", "Systems Software"),
    Company("GOOGL", "Alphabet Inc. (Class A)", "Communication Services", "Interactive Media"),
    Company("GOOG", "Alphabet Inc. (Class C)", "Communication Services", "Interactive Media"),
    Company("TGT", "Target Corporation", "Consumer Staples", "Consumer Staples Merchandise"),
    Company("ON", "ON Semiconductor", "Information Technology", "Semiconductors"),
]


def test_body_hash_ignores_whitespace_only() -> None:
    assert triage.body_hash("Apple  rose\n today.") == triage.body_hash("Apple rose today.")
    assert triage.body_hash("Apple rose today.") != triage.body_hash("Apple fell today.")


def test_normalize_name_drops_legal_words_and_punctuation() -> None:
    assert triage.normalize_name("Apple Inc.") == "apple"
    assert triage.normalize_name("Alphabet Inc. (Class A)") == "alphabet"
    assert triage.normalize_name("The Inc. Corp") == ""


def test_resolver_finds_names_ticker_forms_and_org_spans() -> None:
    resolver = CompanyResolver(UNIVERSE)
    text = "Apple and Microsoft rallied; $ORCL fell. Shares of (NASDAQ: AAPL) rose. MSFT gained."
    found = resolver.resolve(text, org_spans=["Oracle Corp", "Some Private Firm"])
    # the NER span of Oracle is the mention the $ORCL form already found: one mention, not two
    assert found == Counter({"AAPL": 2, "MSFT": 2, "ORCL": 1})


def test_share_classes_resolve_to_one_company() -> None:
    resolver = CompanyResolver(UNIVERSE)
    assert resolver.by_ticker["GOOG"] == resolver.by_ticker["GOOGL"]


def test_word_tickers_and_word_names_are_not_matched_as_words() -> None:
    resolver = CompanyResolver(UNIVERSE)
    assert resolver.resolve("Analysts raised the price target on the stock ON Monday.") == Counter()
    assert resolver.resolve("Target cut its forecast.") == Counter({"TGT": 1})
    assert resolver.resolve("Shares of $ON jumped.") == Counter({"ON": 1})


def test_scope_candidate_single_multi_and_dominated() -> None:
    assert triage.scope_candidate(Counter({"AAPL": 3}), "").scope == "single"
    multi = triage.scope_candidate(Counter({"AAPL": 2, "MSFT": 2}), "")
    assert (multi.scope, multi.companies, multi.top_share) == ("multi", 2, 0.5)
    assert triage.scope_candidate(Counter({"AAPL": 8, "MSFT": 1}), "").scope == "multi_dominated"


def test_scope_candidate_without_companies_uses_keywords() -> None:
    economy = "The Federal Reserve held interest rates as inflation cooled."
    market = "Stocks rallied and the S&P 500 hit a record; Dow Jones futures rose."
    sector = "Semiconductor makers led; chip stocks and semiconductor demand rose."
    assert triage.scope_candidate(Counter(), economy).scope == "economy"
    assert triage.scope_candidate(Counter(), market).scope == "market"
    assert triage.scope_candidate(Counter(), sector, ["semiconductor"]).scope == "sector"
    assert triage.scope_candidate(Counter(), "A recipe for apple pie.").scope == "no_company"


def test_gics_guess_needs_a_majority_of_two_or_more_companies() -> None:
    universe = {c.ticker: c for c in UNIVERSE}
    software = Counter({"MSFT": 1, "ORCL": 1, "AAPL": 1})
    assert triage.gics_guess(software, universe) == ("Information Technology", "Systems Software")
    assert triage.gics_guess(Counter({"MSFT": 3}), universe) is None
    mixed = Counter({"MSFT": 1, "AAPL": 1, "TGT": 1, "GOOGL": 1})
    assert triage.gics_guess(mixed, universe) is None


def test_gold_scope_reads_the_owners_tags_and_notes() -> None:
    assert triage.gold_scope("non-related", "") == "unrelated"
    assert triage.gold_scope("other", "This is about the entire S&P500") == "market_economy"
    assert triage.gold_scope("other", "This is about the economy performance") == "market_economy"
    assert triage.gold_scope("other", "Is about semiconductor sector") == "sector"
    assert triage.gold_scope("miscellany", "") == "multi"
    # the owner's `other` tag is investing news about neither a company nor a sector (D22), unless
    # a note says it is about a sector, the market or several companies
    assert triage.gold_scope("other", "") == "investing_general"
    assert triage.gold_scope("other", "This is more like investing advice") == "investing_general"
    assert triage.gold_scope("other", "Speaks about an entire sector") == "sector"
    assert triage.gold_scope("other", "Speak about multiple companies") == "multi"
    assert triage.gold_scope("product_innovation", "Speak about multiple companies") == "multi"
    assert triage.gold_scope("legal_regulatory", "This is like reputationa_harm") == "single"
    assert set(triage.GOLD_SCOPES) >= {
        triage.gold_scope(*args)
        for args in [("unrelated", ""), ("other", "economy"), ("other", ""), ("x", "")]
    }


def test_a_mention_and_its_ner_span_count_once() -> None:
    resolver = CompanyResolver(UNIVERSE)
    once = resolver.resolve("Shares of Apple were mentioned once.", ["Apple"])
    assert once == Counter({"AAPL": 1})
    assert triage.subjects(once) == Counter()  # one passing mention is not a subject
    # the spans of the NER can exceed the text matches (a form the matcher misses): the larger wins
    assert resolver.resolve("Apple rose.", ["Apple", "Apple", "Apple"]) == Counter({"AAPL": 3})


def test_an_opinion_source_is_not_counted_by_any_evidence() -> None:
    universe = [*UNIVERSE, Company("GS", "Goldman Sachs Group, Inc.", "Financials", "IB")]
    resolver = CompanyResolver(universe)
    text = "Goldman Sachs analysts upgraded Apple. Analysts at (NYSE: GS) agree."
    assert resolver.resolve(text, ["Goldman Sachs"]) == Counter({"AAPL": 1})
    # named as a subject once as well, the company is counted (its plain mentions)
    assert resolver.resolve("Goldman Sachs analysts said. Goldman Sachs rose 3%.") == Counter(
        {"GS": 1}
    )


def test_short_aliases_and_non_word_tickers_are_matched() -> None:
    universe = [
        *UNIVERSE,
        Company("META", "Meta Platforms, Inc.", "Communication Services", "Interactive Media"),
        Company("ZTS", "Zoetis Inc.", "Health Care", "Pharmaceuticals"),
        Company("GM", "General Motors Company", "Consumer Discretionary", "Automobiles"),
        Company("GE", "General Electric Company", "Industrials", "Conglomerates"),
    ]
    resolver = CompanyResolver(universe)
    assert resolver.resolve("Meta announced glasses. Meta stock rose.") == Counter({"META": 2})
    assert resolver.resolve("META and ZTS rose.") == Counter({"META": 1, "ZTS": 1})
    # a first word shared by several companies, or a generic one, is not an alias
    assert resolver.resolve("General stores and a general rule.") == Counter()
    # an everyday-word ticker is still not matched bare
    assert resolver.resolve("ALL of it was NOW.") == Counter()


def test_an_opinion_source_is_not_a_subject() -> None:
    resolver = CompanyResolver(UNIVERSE)
    text = "Microsoft analysts expect Apple to grow. Analysts at Oracle agree. Apple rose."
    assert resolver.resolve(text) == Counter({"AAPL": 2})


def test_a_boilerplate_sentence_without_a_final_period_is_removed() -> None:
    text = "Oracle fell 2%.\nSign up for our newsletter\nApple rose."
    cleaned = triage.strip_boilerplate(text)
    assert "newsletter" not in cleaned
    assert "Oracle fell" in cleaned
    assert "Apple rose" in cleaned


def test_boilerplate_sentences_are_removed() -> None:
    text = "Apple rose 3%. Data compiled by FactSet. Sign up for our newsletter. Oracle fell."
    assert "FactSet" not in triage.strip_boilerplate(text)
    assert "newsletter" not in triage.strip_boilerplate(text)
    assert "Oracle fell" in triage.strip_boilerplate(text)


def test_roundups_by_headline_slug_title_companies_or_ticker_forms() -> None:
    assert triage.is_roundup("Stocks making the biggest moves midday", "", 0, 0)
    assert triage.is_roundup("", "early-movers-pg-blk-wfc-more", 0, 0)
    assert triage.is_roundup("Apple, Oracle and Microsoft fall", "", 3, 0)
    assert triage.is_roundup("", "", 0, 5)
    assert not triage.is_roundup("Apple beats estimates", "apple-beats-estimates", 1, 1)
    resolver = CompanyResolver(UNIVERSE)
    assert resolver.ticker_forms("$AAPL and (NYSE: ORCL) and MSFT") == 2
    scope = triage.scope_candidate(Counter({"AAPL": 5}), "", roundup=True)
    assert scope.scope == "roundup"


def test_wilson_interval() -> None:
    low, high = triage.wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    assert all(math.isnan(x) for x in triage.wilson(0, 0))


def test_unsafe_names_need_a_ticker_form() -> None:
    universe = [
        *UNIVERSE,
        Company("NDAQ", "Nasdaq, Inc.", "Financials", "Financial Exchanges & Data"),
    ]
    resolver = CompanyResolver(universe)
    assert resolver.resolve("The Nasdaq composite rose.") == Counter()
    assert resolver.resolve("Shares of (NASDAQ: NDAQ) rose.") == Counter({"NDAQ": 1})


def test_subjects_drop_passing_mentions_unless_in_the_title() -> None:
    named = Counter({"AAPL": 5, "MSFT": 1, "ORCL": 1})
    assert triage.subjects(named) == Counter({"AAPL": 5})
    assert triage.subjects(named, ["ORCL"]) == Counter({"AAPL": 5, "ORCL": 1})


def test_empty_universe_names_nothing() -> None:
    resolver = CompanyResolver([])
    assert not resolver.resolve("Apple and Microsoft rose, $AAPL (NYSE: MSFT)")


COMMON_WORDS = [
    Company("FRT", "Federal Realty Investment Trust", "Real Estate", "Retail REITs"),
    Company("STT", "State Street Corporation", "Financials", "Asset Management"),
    Company("UNP", "Union Pacific Corporation", "Industrials", "Rail"),
    Company("TTD", "The Trade Desk, Inc.", "Communication Services", "Advertising"),
    Company("WST", "West Pharmaceutical Services, Inc.", "Health Care", "Life Sciences Tools"),
    Company("HD", "Home Depot, Inc. (The)", "Consumer Discretionary", "Home Improvement"),
    Company("TTWO", "Take-Two Interactive Software, Inc.", "Communication Services", "Games"),
    Company("BE", "Bloom Energy Corporation", "Industrials", "Electrical Equipment"),
    Company("XYZ", "Block, Inc.", "Financials", "Transaction Processing"),
    Company("UHS", "Universal Health Services, Inc.", "Health Care", "Facilities"),
    Company("JKHY", "Jack Henry & Associates, Inc.", "Information Technology", "Processing"),
    Company("MLM", "Martin Marietta Materials, Inc.", "Materials", "Construction Materials"),
    Company("TRV", "Travelers Companies, Inc. (The)", "Financials", "Insurance"),
    Company("ROST", "Ross Stores, Inc.", "Consumer Discretionary", "Apparel Retail"),
    Company("WMB", "Williams Companies, Inc.", "Energy", "Pipelines"),
    Company("META", "Meta Platforms, Inc.", "Communication Services", "Interactive Media"),
    Company("BSX", "Boston Scientific Corporation", "Health Care", "Equipment"),
    Company("KO", "Coca-Cola Company (The)", "Consumer Staples", "Soft Drinks"),
    Company("PANW", "Palo Alto Networks, Inc.", "Information Technology", "Security"),
    Company("WFC", "Wells Fargo & Company", "Financials", "Banks"),
    Company("MS", "Morgan Stanley", "Financials", "Investment Banking"),
    Company("JPM", "JPMorgan Chase & Co.", "Financials", "Banks"),
    Company("LLY", "Lilly (Eli) and Company", "Health Care", "Pharmaceuticals"),
    Company("EW", "Edwards Lifesciences Corporation", "Health Care", "Equipment"),
    Company("RL", "Ralph Lauren Corporation", "Consumer Discretionary", "Apparel"),
    Company("ITW", "Illinois Tool Works Inc.", "Industrials", "Machinery"),
]
#: Each ordinary use of the word, and the company it used to be taken for.
COMMON_USES = [
    ("The Federal Reserve held rates.", "FRT"),
    ("The State Department spoke.", "STT"),
    ("The Union called a strike.", "UNP"),
    ("A Trade deal was signed.", "TTD"),
    ("The West is dry.", "WST"),
    ("Home sales fell.", "HD"),
    ("Take a look at rates.", "TTWO"),
    ("Michael Bloom said so.", "BE"),
    ("A Block of shares sold.", "XYZ"),
    ("Universal Studios opened.", "UHS"),
    ("Jack said so.", "JKHY"),
    ("Martin said so.", "MLM"),
    ("Travelers waited in line.", "TRV"),
    ("Commerce Secretary Wilbur Ross spoke.", "ROST"),
    ("Williams said so.", "WMB"),
    ("The Boston Globe reported it from Boston.", "BSX"),
    ("Doug Parker and Wells said so; Palo Alto is a city.", "PANW"),
    ("J.P. Morgan upgraded it, said Morgan.", "MS"),
    ("Wells said so.", "WFC"),
    ("Patty Edwards owns it.", "EW"),
    ("Ralph said so.", "RL"),
    ("Illinois voters went to the polls.", "ITW"),
]
#: The company named properly: it must still count.
REAL_USES = [
    ("Federal Realty Investment Trust rose.", "FRT"),
    ("State Street Corporation rose.", "STT"),
    ("Union Pacific rose.", "UNP"),
    ("The Trade Desk rose.", "TTD"),
    ("West Pharmaceutical Services rose.", "WST"),
    ("Home Depot rose.", "HD"),
    ("Take-Two Interactive rose.", "TTWO"),
    ("Bloom Energy rose.", "BE"),
    ("Block, Inc. rose.", "XYZ"),
    ("Universal Health Services rose.", "UHS"),
    ("Jack Henry rose.", "JKHY"),
    ("Martin Marietta rose.", "MLM"),
    ("Travelers Companies rose.", "TRV"),
    ("Ross Stores rose.", "ROST"),
    ("Williams Companies rose.", "WMB"),
    ("Boston Scientific rose.", "BSX"),
    ("Coca-Cola rose.", "KO"),
    ("Palo Alto Networks rose.", "PANW"),
    ("Wells Fargo rose.", "WFC"),
    ("Morgan Stanley rose.", "MS"),
    ("Eli Lilly rose.", "LLY"),
    ("Edwards Lifesciences rose.", "EW"),
    ("Ralph Lauren rose.", "RL"),
    ("Illinois Tool Works rose.", "ITW"),
]


@pytest.mark.parametrize(("text", "ticker"), COMMON_USES)
def test_an_ordinary_word_is_not_the_company(text: str, ticker: str) -> None:
    resolver = CompanyResolver(COMMON_WORDS)
    assert resolver.resolve(text, skip_attribution=False)[ticker] == 0


@pytest.mark.parametrize(("text", "ticker"), REAL_USES)
def test_the_company_named_properly_still_counts(text: str, ticker: str) -> None:
    assert CompanyResolver(COMMON_WORDS).resolve(text, skip_attribution=False)[ticker] == 1


def test_a_short_form_counts_once_the_body_names_the_company_another_way() -> None:
    resolver = CompanyResolver(COMMON_WORDS)
    assert resolver.resolve("Ross Stores rose. Ross said so.", skip_attribution=False)["ROST"] == 2
    assert resolver.resolve("Bloom Energy rose; Bloom added.", skip_attribution=False)["BE"] == 2
    assert resolver.resolve("Block (NYSE: XYZ) said so.", skip_attribution=False)["XYZ"] == 2
    assert resolver.resolve("Block, Inc. said so. Block fell.", skip_attribution=False)["XYZ"] == 2
    assert resolver.resolve("Williams WMB rose.", skip_attribution=False)["WMB"] == 2
    body = "Travelers Companies and Travelers fell."
    assert resolver.resolve(body, skip_attribution=False)["TRV"] == 2


def test_a_one_word_name_is_only_read_capitalised_even_when_corroborated() -> None:
    resolver = CompanyResolver(COMMON_WORDS)
    body = "Block (NYSE: XYZ) said a block trade and many travelers and williams."
    assert resolver.resolve(body, skip_attribution=False)["XYZ"] == 2  # the name and the ticker
    assert resolver.resolve(body, skip_attribution=False)["TRV"] == 0  # no ticker form: none


def test_stored_ner_spans_obey_the_same_rule() -> None:
    resolver = CompanyResolver(COMMON_WORDS)
    assert resolver.resolve("Wall Street rose.", ["Williams"], skip_attribution=False)["WMB"] == 0
    assert resolver.resolve("Williams WMB rose.", ["Williams"], skip_attribution=False)["WMB"] == 2


def test_counting_and_the_attribution_spans_differ_on_purpose() -> None:
    """``resolve`` counts a body, so it drops "Ross" alone; ``mention_spans`` places the masks of
    single sentences and keeps it: ``attribution_units.GuardedResolver`` decides those with the
    whole body in hand. Do not "fix" one side to match the other."""
    resolver = CompanyResolver(COMMON_WORDS)
    text = "Commerce Secretary Wilbur Ross spoke."
    assert resolver.resolve(text, skip_attribution=False)["ROST"] == 0
    assert [t for _, _, t in resolver.mention_spans(text)] == ["ROST"]


def test_distinctive_first_word_aliases_still_count() -> None:
    assert CompanyResolver(COMMON_WORDS).resolve("Meta fell.", skip_attribution=False) == Counter(
        {"META": 1}
    )


CURLY, NBSP = chr(0x2019), chr(0xA0)  # the curly apostrophe, the no-break space
TRICKY_NAMES = [
    Company("CASY", "Casey's General Stores, Inc.", "Consumer Staples", "Food Retail"),
    Company("DPZ", "Domino's Pizza Inc", "Consumer Discretionary", "Restaurants"),
    Company("MCO", "Moody's Corporation", "Financials", "Financial Exchanges & Data"),
    Company("LOW", "Lowe's Companies, Inc.", "Consumer Discretionary", "Home Improvement"),
    Company("SNA", "Snap-on Incorporated", "Industrials", "Machinery"),
    Company("BMY", "Bristol-Myers Squibb", "Health Care", "Pharmaceuticals"),
    Company("PH", "Parker-Hannifin Corporation", "Industrials", "Machinery"),
    Company("JKHY", "Jack Henry & Associates, Inc.", "Information Technology", "Processing"),
    Company("XYZ", "Block, Inc.", "Financials", "Transaction Processing"),
    Company("WSM", "Williams-Sonoma, Inc.", "Consumer Discretionary", "Home Furnishings"),
    Company("WMB", "Williams Companies, Inc.", "Energy", "Pipelines"),
]


@pytest.mark.parametrize("company", TRICKY_NAMES, ids=lambda c: c.ticker)
def test_every_company_is_found_by_its_own_name(company: Company) -> None:
    """The names of a universe are read as written: apostrophes, hyphens and ``&`` included, with
    the straight or the curly apostrophe and a no-break space. A stop list must not erase one."""
    resolver = CompanyResolver(TRICKY_NAMES)
    for name in (company.name, company.name.replace("'", CURLY).replace(" ", NBSP)):
        assert resolver.resolve(f"{name} rose.", skip_attribution=False)[company.ticker] >= 1, name


def test_the_name_of_a_company_with_an_apostrophe_is_read_without_its_alias() -> None:
    resolver = CompanyResolver(TRICKY_NAMES)
    assert resolver.resolve(
        f"Casey{CURLY}s General Stores rose.", skip_attribution=False
    ) == Counter({"CASY": 1})
    assert resolver.resolve("Williams-Sonoma rose.", skip_attribution=False)["WSM"] == 1
    assert resolver.resolve("Williams-Sonoma rose.", skip_attribution=False)["WMB"] == 0


@pytest.mark.parametrize(
    ("text", "ticker"),
    [
        ("Smith met Jack. Henry rose to speak.", "JKHY"),
        ("Ask Martin. Marietta is a town.", "MLM"),
        ("Federal. Realty investors", "FRT"),
        ("Travelers. Companies said.", "TRV"),
        ("Morgan\n\nStanley rose.", "MS"),
        ("Home\nDepot rose.", "HD"),
        ("Take\nTwo Interactive Software rose.", "TTWO"),
    ],
)
def test_a_name_is_on_one_line_and_in_one_sentence(text: str, ticker: str) -> None:
    resolver = CompanyResolver(COMMON_WORDS)
    assert resolver.resolve(text, skip_attribution=False)[ticker] == 0
