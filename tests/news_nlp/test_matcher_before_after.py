"""The before/after check of the matcher (PR #155, PLAN D38): a revision read from git, and the
self-check of a universe against its own names."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import triage
from news_nlp.triage import Company, CompanyResolver

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _script("matcher_before_after_2026_11_09")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 -- a fixed git command in a temporary repository
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_a_revision_of_triage_is_read_from_git_and_not_imported_over_the_working_one(
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "news_nlp").mkdir(parents=True)
    triage_file = tmp_path / "src" / "news_nlp" / "triage.py"
    triage_file.write_text("VERSION = 'before'\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "before")
    before = _git(tmp_path, "rev-parse", "HEAD")
    triage_file.write_text("VERSION = 'after'\n", encoding="utf-8")
    _git(tmp_path, "commit", "-aq", "-m", "after")
    assert CHECK.triage_at(before, tmp_path).VERSION == "before"
    assert CHECK.triage_at("HEAD", tmp_path).VERSION == "after"
    with pytest.raises(subprocess.CalledProcessError):
        CHECK.triage_at("no-such-revision", tmp_path)


def test_the_self_check_lists_the_company_a_stop_word_erased() -> None:
    universe = [
        Company("CASY", "Casey's General Stores, Inc.", "Staples", "Food Retail"),
        Company("AAPL", "Apple Inc.", "Technology", "Hardware"),
        Company("FRT", "Federal Realty Investment Trust", "Real Estate", "REITs"),
    ]
    assert CHECK.self_check(CompanyResolver(universe), universe) == []


def test_a_ticker_form_in_a_disclosure_or_a_list_is_told_from_one_in_a_sentence() -> None:
    footer = "Trader A Owns (INTC); Trader A Owns (GS) Calls; Trader B Is Short (F)."
    assert triage.in_disclosure(footer, footer.index("INTC"), footer.index("INTC") + 4)
    sentence = "Cramer said shares of AMZN rose on Tuesday after the quarter was reported."
    assert not triage.in_disclosure(sentence, sentence.index("AMZN"), sentence.index("AMZN") + 4)
    for genre in (
        "J. Najarian is long MS; J. Najarian is long GMCR; J. Najarian is long FSLR; J. Najarian is long DB;",
        "Disclosure: Trader A and Joule Financial have positions in WMT and DIS.",
        "The firm has an options position in (NFLX) and is short Jan calls NFLX.",
        "Trader A Capital Markets LLC makes a market in NFLX CNBC.com with wires",
        "Trader A was a seller of NFLX. Trader B was a buyer of Ford.",
        "Citigroup acted as manager or co-manager of an offering of securities of NFLX Citigroup",
        "Citi Expects To Receive Compensation from (AXP), (V), (MA) Citi Has Acted As Manager",
        "Citi has Managed a Securities Offering of (YUM)  Citi Has Received Compensation from (YUM)",
        "Compensation For Services Provided In Past 12 Months From (MOT), (GLW) Citigroup",
        "Charlie is funds long GS Charlie is funds long KKR Charlie is funds long JPM",
        "Stuart Frankel & Co and its partners own HPQ Stuart Frankel & Co and its partners own HSPO",
        "Investment Banking Clients Of Citigroup Global Markets (CVS), (DG), (HD), (JCP), (KSS)",
        "Compensation In Past 12 Months From (EXPE), (AMZN), (YHOO), (GOOG) Are Or Were Clients",
        "Guy Adami is a buyer of DAL. Steve Grasso is a buyer of SO.",
        "Jon Najarian owns and has sold calls against (AAPL) Jon Najarian owns and has sold calls",
        "Cantor Fitzgerald is a market maker of ORCL Cantor Fitzgerald is a market maker of GLD",
    ):
        key = next(
            k
            for k in (
                "FSLR",
                "WMT",
                "NFLX",
                "AXP",
                "YUM",
                "GLW",
                "KKR",
                "HPQ",
                "CVS",
                "EXPE",
                "DAL",
                "AAPL",
                "ORCL",
            )
            if k in genre
        )
        spot = genre.index(key)
        assert triage.in_disclosure(genre, spot, spot + len(key)), genre
    for prose in (
        "Cramer said the stock is long overdue for a rally, and AMZN fell.",
        "Analysts have a position on the deal, said AMZN's chief.",
    ):
        spot = prose.index("AMZN")
        assert not triage.in_disclosure(prose, spot, spot + 4), prose
    run = "short: SPY, XRT, QQQ, IWM, DIA, GLD, SLV. long: AAPL"
    assert triage.in_disclosure(run, run.index("XRT"), run.index("XRT") + 3)


def test_the_disclosures_report_counts_the_pairs_named_only_in_a_footer() -> None:
    universe = [
        Company("INTC", "Intel Corporation", "Technology", "Semiconductors"),
        Company("AAPL", "Apple Inc.", "Technology", "Hardware"),
    ]
    bodies = [
        "Apple rose. Trader A Owns (INTC); Trader A Owns (GS) Calls; Trader B Is Short (F).",
        "Intel (NYSE: INTC) fell and Apple rose.",
    ]
    lines = CHECK.disclosures(universe, bodies)
    assert lines[0].startswith("4 counted company-body pairs; 1 (25.0%)")
    assert lines[2].split()[:4] == ["INTC", "1", "/", "2"]


def test_the_labels_are_rendered_with_their_context_from_source(tmp_path: Path) -> None:
    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS source")
    conn.execute("CREATE TABLE source.articles (id INTEGER, body_text TEXT)")
    conn.execute("INSERT INTO source.articles VALUES (7, 'Mark Warner said Time Warner sold it.')")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "set,n,article_id,ticker,word,rule,offset,residue,seed,label,note\n"
        "demo,0,7,WBD,Warner,alias,5,13,1,not,a senator\n"
        "other,0,7,WBD,Warner,alias,25,13,1,not,\n",
        encoding="utf-8",
    )
    lines = CHECK.render_labels(conn, labels, only="demo", width=5)  # type: ignore[arg-type]
    assert lines == ["demo   0 not        WBD   Mark [[Warner]] said  (a senator)"]


@pytest.mark.parametrize(
    ("text", "ticker"),
    [
        (
            "THE FED RAISED RATES AND AAPL FELL TODAY AS THE US ECONOMY SLOWED, SAID THE CEO.",
            "AAPL",
        ),
        (
            "Key events: ADP, CPI, GDP, ISM, PMI and NFP reports; AAPL also reports earnings.",
            "AAPL",
        ),
        ("Disney (DIS) owns ESPN and said so.", "DIS"),
        ("Amazon is large. AMZN owns Whole Foods and Zoox.", "AMZN"),
        ("Calendar: ADP, CPI, GDP, ISM, PMI, NFP and AAPL results.", "AAPL"),
        ("Executive compensation for AMZN's chief rose 20% last year, the filing showed.", "AMZN"),
        ("The bank said it will pay compensation for services to AAPL suppliers.", "AAPL"),
        ("Analysts say investors should get long Nvidia calls, and NVDA could rally.", "NVDA"),
        ("The fund has positions in many tech names, including MSFT.", "MSFT"),
        ("Our partners own the supply chain for AMZN deliveries.", "AMZN"),
        ("The index is a holding of the fund, as is AAPL.", "AAPL"),
        ("We provide banking services to AAPL and its suppliers.", "AAPL"),
        ("Berkshire beneficially owns stakes, and AAPL is the largest.", "AAPL"),
        ("Fidelity has a short position in TSLA, filings show.", "TSLA"),
    ],
)
def test_ordinary_prose_is_not_a_disclosure(text: str, ticker: str) -> None:
    """The generic readings (any "owns", any six upper-case words, then phrases such as "compensation
    for" and "partners own" not anchored to a ticker) flagged all of these (PR #155, rounds 6, 7)."""
    known = {"AAPL", "DIS", "AMZN", "ADP", "MSFT", "NVDA", "TSLA"}
    spot = text.index(ticker)
    assert triage.disclosure_branch(text, spot, spot + len(ticker), known) is None


def test_the_branches_of_a_disclosure_are_named() -> None:
    known = {"INTC", "GS", "MSFT", "BAC", "C", "JPM", "KO"}
    owns = "Adami Owns AGU; Adami Owns C; Adami Owns INTC; Adami Owns MSFT;"
    assert (
        triage.disclosure_branch(owns, owns.index("INTC"), owns.index("INTC") + 4, known) == "owns"
    )
    paren = "Najarian Owns More Than 1% Of (EXR) Virtus"
    assert (
        triage.disclosure_branch(paren, paren.index("EXR"), paren.index("EXR") + 3, known) == "owns"
    )
    lst = "Her firm: BAC, C, GS, JPM, KO, XYZ and more."
    spot = lst.index("GS")
    assert triage.disclosure_branch(lst, spot, spot + 2, known) == "list"
    assert (
        triage.disclosure_branch(lst, spot, spot + 2, {"GS"}) is None
    )  # a list of words, not tickers
    resolver = CompanyResolver(
        [Company("INTC", "Intel Corporation", "Technology", "Semiconductors")]
    )
    footer = "Trader A Owns (INTC); Trader B Is Short (F)."
    assert resolver.disclosure_branches(footer, ["INTC"]) == {"INTC": frozenset({"genre"})}
    assert resolver.disclosure_branches("Intel (NYSE: INTC) fell.", ["INTC"]) == {}


LABELS = Path(__file__).resolve().parents[2] / "docs" / "matcher_labels_2026-11-09.csv"

#: The figures PLAN D38 quotes, by set: (right, read), unreadable left out.
QUOTED = {
    "r13_owner15": (86, 89),
    "r29_owner15": (38, 40),
    "r13_name_ticker": (37, 39),
    "r29_name_ticker": (34, 36),
    "r13_short_corr": (12, 14),
    "r29_short_corr": (26, 27),
    "r13_alias": (59, 94),
    "r29_alias": (80, 94),
    "r41_alias": (86, 98),
    "r53_disclosure_flagged": (39, 40),
    "r13_ten_names_new": (74, 74),
    "r59_owns_only_flagged": (21, 25),  # the generic "owns" of the first detector, read alone
    "r59_dense_only_flagged": (13, 25),  # and its "six upper-case words", read alone
    "r17_final_flagged": (40, 40),  # the final detector, on a residue no edit was chosen from
    "r17_round6_phrases_only_flagged": (25, 25),  # the phrases of round 6 alone, the same sample
}


def test_the_committed_labels_are_valid_and_give_the_figures_quoted() -> None:
    lines = {line.split()[0]: line for line in CHECK.labels_summary(LABELS)}  # a bad label raises
    for name, (right, read) in QUOTED.items():
        assert f"precision {right}/{read} " in lines[name], (name, lines[name])


def test_the_tightening_of_the_detector_is_read_in_the_labels() -> None:
    lines = {line.split()[0]: line for line in CHECK.labels_summary(LABELS)}
    line = lines["r59_unflagged_by_tightening"]
    assert "n= 40" in line and "disclosure-missed 7" in line
    final = lines["r17_final_unflagged"]  # the recall read: 4 missed of 39 readable
    assert "n= 40" in final and "disclosure-missed 4" in final and "unreadable 1" in final


def test_a_label_a_set_does_not_allow_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "labels.csv"
    bad.write_text(
        "set,n,article_id,ticker,word,rule,offset,residue,seed,label,note\n"
        "r1_alias,0,1,F,Ford,alias,0,1,1,disclosure,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not in"):
        CHECK.labels_summary(bad)


def _scope_store() -> object:
    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE article_entities (article_id INTEGER, entity_type TEXT, text TEXT)")
    conn.execute("ATTACH DATABASE ':memory:' AS source")
    conn.execute(
        "CREATE TABLE source.articles (id INTEGER, title TEXT, body_text TEXT, fetch_status TEXT, "
        "pub_date TEXT)"
    )
    conn.execute("CREATE TABLE source.discovered_urls (id INTEGER, url TEXT)")
    body = "Zenith Dynamics raised its forecast. Zenith Dynamics also said demand is strong. " * 3
    for i in (1, 2):
        conn.execute(
            "INSERT INTO source.articles VALUES (?, 'Zenith up', ?, 'ok', '2022-03-01')",
            (i, body + f" Story {i}."),
        )
        conn.execute("INSERT INTO source.discovered_urls VALUES (?, 'https://x.test/a.html')", (i,))
    conn.execute(
        "INSERT INTO source.articles VALUES (3, 'Old', 'Zenith old news.', 'ok', '2019-01-01')"
    )
    return conn


_RealResolver = CompanyResolver


class _NamesNobody:
    """A stand-in for the matcher before the change: a module whose resolver finds no company."""

    Company = triage.Company

    class CompanyResolver(_RealResolver):
        def resolve(self, text, org_spans=(), *, skip_attribution=True):  # type: ignore[no-untyped-def]
            return triage.Counter()


def test_the_scope_shift_reads_the_window_as_the_cut_does() -> None:
    import argparse  # noqa: PLC0415

    universe = [Company("ZEN", "Zenith Dynamics Inc", "Industrials", "Aerospace")]
    args = argparse.Namespace(since="2021-12-31", modulo=1, residue=0, bodies=10, base="x")
    conn = _scope_store()
    same = CHECK.scope_shift(conn, universe, triage, args)  # type: ignore[arg-type]
    assert same[0].startswith("2 unique bodies") and "scope changed: 0 (0.0%)" in same[1]
    moved = CHECK.scope_shift(conn, universe, _NamesNobody, args)  # type: ignore[arg-type]
    assert "rule scope changed: 2 (100.0%)" in moved[1]
    assert "enter the set that gets a summary: 2" in moved[2]  # the old reading named nobody
