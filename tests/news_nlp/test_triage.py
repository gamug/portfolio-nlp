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
