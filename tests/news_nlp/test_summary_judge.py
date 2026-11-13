"""The LLM-as-judge measure of the summary step (step 2), by group (single bodies, sub-news) and by the length of the source:
the draw, the claim-level judge, the deterministic measures, the figures, and the owner's agreement check. The LLM is a stub."""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from news_nlp import triage

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "summary_judge", ROOT / "scripts" / "summary_judge_2026_11_13.py"
)
assert spec and spec.loader
sj = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sj
spec.loader.exec_module(sj)

SUM_REV = "summary-test"
SCOPE_REV = "scope-test"
CUT_REV = "units5-test"


# --- the buckets and the allocation ------------------------------------------------------------------------------------------


def test_the_buckets_are_terciles_of_the_chunks_of_the_population() -> None:
    chunks = [1] * 30 + [2] * 30 + [5] * 30
    edges = sj.bucket_edges(chunks)
    assert [sj.bucket_of(c, edges) for c in (1, 2, 5)] == [0, 1, 2]


def test_a_population_with_one_value_has_one_bucket_and_no_bucket_is_lost() -> None:
    edges = sj.bucket_edges([3] * 10)
    assert {sj.bucket_of(3, edges)} == {0}
    assert sj.allocate({0: 10, 1: 0, 2: 0}, 6) == {
        0: 6,
        1: 0,
        2: 0,
    }  # nothing to take from the empty ones: the rest goes where there is room


def test_the_allocation_is_equal_across_buckets_capped_by_the_population_and_adds_up() -> None:
    assert sj.allocate({0: 100, 1: 100, 2: 100}, 150) == {0: 50, 1: 50, 2: 50}
    got = sj.allocate({0: 100, 1: 100, 2: 100}, 100)
    assert sum(got.values()) == 100 and max(got.values()) - min(got.values()) <= 1
    got = sj.allocate(
        {0: 10, 1: 100, 2: 100}, 150
    )  # a small bucket gives all it has, the others take the rest
    assert got[0] == 10 and sum(got.values()) == 150
    assert sj.allocate({0: 5, 1: 5, 2: 5}, 150) == {0: 5, 1: 5, 2: 5}  # less than asked: all of it


# --- the draw ----------------------------------------------------------------------------------------------------------------


def _results(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE news_summary (item_kind TEXT, item_id TEXT, model_revision TEXT, summary_text TEXT, text_source TEXT, num_chunks INTEGER, processed_at TEXT);
        CREATE TABLE body_scope (body_hash TEXT, classifier_revision TEXT, scope TEXT);
        CREATE TABLE sub_news (id INTEGER PRIMARY KEY, body_hash TEXT, article_id INTEGER, text TEXT, extractor_revision TEXT);
        """
    )
    return conn


def _source(path: Path, rows: list[tuple[int, str, str]]) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body_text TEXT, fetch_status TEXT, ticker TEXT, company TEXT, gics_sector TEXT, gics_sub_industry TEXT)"
    )
    for i, title, body in rows:
        conn.execute(
            "INSERT INTO articles VALUES (?, ?, ?, 'ok', 'AAPL', 'Apple Inc.', 'IT', 'HW')",
            (i, title, body),
        )
    conn.commit()
    return conn


@pytest.fixture
def dbs(tmp_path: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    bodies = {i: f"Body {i}. Apple Inc. did thing {i}. " * (60 + i) for i in range(1, 25)}
    source = _source(tmp_path / "src.db", [(i, f"Title {i}", text) for i, text in bodies.items()])
    results = _results(tmp_path / "res.db")
    for i, text in bodies.items():
        h = triage.body_hash(text)
        results.execute("INSERT INTO body_scope VALUES (?, ?, 'single')", (h, SCOPE_REV))
        source_kind = (
            "summary" if i % 4 else "original"
        )  # a quarter was under the size: stored as it is, not summarised
        results.execute(
            "INSERT INTO news_summary VALUES ('body', ?, ?, ?, ?, ?, 'now')",
            (h, SUM_REV, f"Apple did thing {i}.", source_kind, 1 + i // 8),
        )
    for n in range(1, 31):
        results.execute(
            "INSERT INTO sub_news VALUES (?, 'h', 1, ?, ?)",
            (n, f"Sub-news {n}. Apple Inc. announced item {n}. " * (30 + n), CUT_REV),
        )
        results.execute(
            "INSERT INTO news_summary VALUES ('sub_news', ?, ?, ?, 'summary', ?, 'now')",
            (str(n), SUM_REV, f"Apple announced item {n}.", 1 + (n - 1) // 10),
        )
    results.execute("INSERT INTO sub_news VALUES (99, 'h', 1, 'other cut', 'units-old')")
    results.execute(
        "INSERT INTO news_summary VALUES ('sub_news', '99', ?, 'x', 'summary', 1, 'now')",
        (SUM_REV,),
    )
    results.commit()
    return results, source


def test_only_summarised_items_of_the_named_revisions_are_in_the_frame(
    dbs: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    results, _ = dbs
    frame = sj.eligible(results, SUM_REV, SCOPE_REV, CUT_REV)
    assert len(frame["single"]) == 18  # 24 bodies, 6 of them stored as they are
    assert len(frame["multi"]) == 30  # the sub-news of the named cut only
    assert sj.eligible(results, SUM_REV, "scope-other", CUT_REV)["single"] == []
    assert sj.eligible(results, "summary-other", SCOPE_REV, CUT_REV) == {"single": [], "multi": []}


def test_the_draw_takes_equal_parts_per_bucket_and_is_reproducible(
    dbs: tuple[sqlite3.Connection, sqlite3.Connection], tmp_path: Path
) -> None:
    results, source = dbs
    rows = sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=12, seed=7)
    again = sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=12, seed=7)
    assert [r["id"] for r in rows] == [r["id"] for r in again]
    for group in ("single", "multi"):
        mine = [r for r in rows if r["group"] == group]
        assert len(mine) == 12
        by_bucket = {b: sum(1 for r in mine if r["bucket"] == b) for b in (0, 1, 2)}
        assert sorted(by_bucket.values()) == [4, 4, 4]
        # the weight of a bucket is its population over its sample, so the weights add up to the population
        assert sum(float(r["weight"]) for r in mine) == pytest.approx(
            len(sj.eligible(results, SUM_REV, SCOPE_REV, CUT_REV)[group])
        )
    other = sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=12, seed=8)
    assert [r["id"] for r in other] != [r["id"] for r in rows]


def test_a_single_body_is_given_to_the_judge_as_the_summariser_read_it_and_a_sub_news_alone(
    dbs: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    results, source = dbs
    rows = sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=6, seed=1)
    single = next(r for r in rows if r["group"] == "single")
    assert (
        single["source_text"].startswith("Title-Title ") and "\nBody-Body " in single["source_text"]
    )
    multi = next(r for r in rows if r["group"] == "multi")
    assert multi["source_text"].startswith(
        "Sub-news "
    )  # the title belongs to the whole round-up, not to the piece
    assert int(multi["source_chars"]) == len(multi["source_text"])
    assert int(multi["summary_chars"]) == len(multi["summary_text"])


def test_the_deterministic_measures_are_stored_with_the_row(
    dbs: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    results, source = dbs
    rows = sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=6, seed=1)
    row = rows[0]
    assert json.loads(row["source_tickers"]) == {
        "AAPL": pytest.approx(json.loads(row["source_tickers"])["AAPL"])
    }
    assert "AAPL" in json.loads(row["source_tickers"])
    assert row["summary_tickers"] != "" and float(row["compression"]) == pytest.approx(
        int(row["summary_chars"]) / int(row["source_chars"])
    )


def test_write_and_read_keep_the_texts_and_refuse_to_overwrite(
    dbs: tuple[sqlite3.Connection, sqlite3.Connection], tmp_path: Path
) -> None:
    results, source = dbs
    rows = sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=6, seed=1)
    path = tmp_path / "sample.csv"
    sj.write_sample(path, rows)
    assert [r["source_text"] for r in sj.read_sample(path)] == [r["source_text"] for r in rows]
    with pytest.raises(FileExistsError):
        sj.write_sample(path, rows)


# --- the judge ---------------------------------------------------------------------------------------------------------------


class StubAgent:
    """Replies with a fixed JSON or with prose, and remembers what it was asked."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


GOOD = json.dumps(
    {
        "claims": [
            {"text": "Apple did a thing", "verdict": "supported", "evidence": "line 1"},
            {"text": "It cost $5 billion", "verdict": "unsupported", "evidence": ""},
            {"text": "Revenue fell", "verdict": "contradicted", "evidence": "revenue rose"},
        ],
        "main_event": "Apple did a thing",
        "main_event_covered": True,
        "rationale": "one invented figure",
    }
)


def test_the_judge_prompt_names_the_source_and_the_summary_and_the_verdicts() -> None:
    prompt = sj.PROMPT
    for needle in (
        "supported",
        "unsupported",
        "contradicted",
        "main_event_covered",
        "ONLY the source",
    ):
        assert needle in prompt
    text = sj.user_prompt("THE SOURCE", "THE SUMMARY")
    assert text.index("THE SOURCE") < text.index("THE SUMMARY")


def test_a_reply_is_read_and_counted() -> None:
    verdict = sj.judge_one(StubAgent([GOOD]), "src", "sum")
    assert not verdict.parse_failed
    assert [c.verdict for c in verdict.claims] == ["supported", "unsupported", "contradicted"]


def test_a_reply_that_is_not_json_is_repaired_once_then_marked_failed() -> None:
    agent = StubAgent(["I think the summary is fine.", GOOD])
    assert not sj.judge_one(agent, "src", "sum").parse_failed
    assert len(agent.prompts) == 2
    failing = StubAgent(["prose", "more prose"])
    assert sj.judge_one(failing, "src", "sum").parse_failed


def test_a_verdict_outside_the_three_is_not_accepted() -> None:
    bad = json.dumps({"claims": [{"text": "x", "verdict": "probably"}], "main_event_covered": True})
    assert sj.judge_one(StubAgent([bad, bad]), "src", "sum").parse_failed


def test_the_run_is_resumable_and_retries_only_the_failed(tmp_path: Path) -> None:
    rows = [
        {"id": f"r{i}", "group": "single", "source_text": "s", "summary_text": "t"}
        for i in range(4)
    ]
    out = tmp_path / "v.jsonl"
    agent_a = StubAgent([GOOD])
    sj.run_judge(rows[:2], lambda: agent_a, out, workers=1)
    assert len(agent_a.prompts) == 2
    # a second run over all four judges the two missing ones and nothing else
    agent_b = StubAgent([GOOD])
    sj.run_judge(rows, lambda: agent_b, out, workers=1)
    assert len(agent_b.prompts) == 2
    assert set(sj.read_verdicts(out)) == {"r0", "r1", "r2", "r3"}
    # a failed one is retried by the next run
    out2 = tmp_path / "v2.jsonl"
    sj.run_judge(rows[:1], lambda: StubAgent(["no", "no"]), out2, workers=1)
    assert sj.read_verdicts(out2)["r0"]["parse_failed"] is True
    agent_c = StubAgent([GOOD])
    sj.run_judge(rows[:1], lambda: agent_c, out2, workers=1)
    assert len(agent_c.prompts) == 1 and sj.read_verdicts(out2)["r0"]["parse_failed"] is False


# --- the figures --------------------------------------------------------------------------------------------------------------


def _verdict(*claims: str, covered: bool = True, failed: bool = False) -> dict[str, Any]:
    return {
        "claims": [{"text": "c", "verdict": v} for v in claims],
        "main_event_covered": covered,
        "parse_failed": failed,
    }


def test_the_item_measures_count_claims_not_words() -> None:
    m = sj.item_measures(_verdict("supported", "supported", "unsupported", "contradicted"))
    assert (m["claims"], m["unsupported"], m["contradicted"]) == (4, 1, 1)
    assert m["clean"] is False and m["any_contradicted"] is True
    clean = sj.item_measures(_verdict("supported", "supported"))
    assert clean["clean"] is True and clean["any_contradicted"] is False
    only_contradicted = sj.item_measures(_verdict("supported", "contradicted"))
    assert (
        only_contradicted["clean"] is False
    )  # a contradicted claim alone makes the summary unfaithful
    only_unsupported = sj.item_measures(_verdict("supported", "unsupported"))
    assert only_unsupported["clean"] is False and only_unsupported["any_contradicted"] is False
    empty = sj.item_measures(_verdict())  # no claim to check is not a clean summary
    assert empty["clean"] is None


def test_wilson_and_the_weighted_rate() -> None:
    low, high = sj.wilson(8, 10)
    assert low < 0.8 < high
    assert sj.weighted_rate([1, 0, 1, 1], [1, 1, 1, 1]) == pytest.approx(0.75)
    assert sj.weighted_rate([1, 0], [1, 9]) == pytest.approx(
        0.1
    )  # the second stands for nine times as many items


def test_the_report_gives_the_rates_by_group_and_bucket_and_leaves_the_failed_out() -> None:
    rows = []
    verdicts = {}
    for i in range(12):
        rows.append(
            {
                "id": f"s{i}",
                "group": "single",
                "bucket": str(i % 3),
                "weight": "10",
                "pop_n": "40",
                "source_chars": "3000",
                "summary_chars": "300",
                "compression": "0.1",
                "source_tickers": json.dumps({"AAPL": 3}),
                "summary_tickers": json.dumps({"AAPL": 1} if i % 2 else {}),
            }
        )
        verdicts[f"s{i}"] = _verdict("supported", "unsupported") if i < 4 else _verdict("supported")
    verdicts["s11"] = _verdict(failed=True)
    report = sj.summarise(rows, verdicts, seed=1, draws=200)
    single = report["single"]
    assert single["n"] == 12 and single["judged"] == 11 and single["parse_failed"] == 1
    assert single["clean"]["estimate"] == pytest.approx(7 / 11)
    assert set(single["by_bucket"]) == {"0", "1", "2"}
    assert (
        0 <= single["clean"]["low"] <= single["clean"]["estimate"] <= single["clean"]["high"] <= 1
    )
    assert single["main_company_kept"]["estimate"] == pytest.approx(0.5, abs=0.1)
    assert "multi" not in report or report["multi"]["n"] == 0


def test_the_entities_of_the_source_are_kept_or_lost_in_the_summary() -> None:
    assert sj.entity_measures({"AAPL": 3, "MSFT": 1}, {"AAPL": 1}) == {
        "has_company": True,
        "main_kept": True,
        "recall": 0.5,
    }
    assert sj.entity_measures({"AAPL": 3, "MSFT": 1}, {"MSFT": 1})["main_kept"] is False
    assert sj.entity_measures({}, {"AAPL": 1})["has_company"] is False


def test_the_text_of_the_report_names_the_judge_the_groups_and_the_limits() -> None:
    rows = [
        {
            "id": "a",
            "group": "single",
            "bucket": "0",
            "weight": "1",
            "pop_n": "1",
            "source_chars": "10",
            "summary_chars": "5",
            "compression": "0.5",
            "source_tickers": "{}",
            "summary_tickers": "{}",
        }
    ]
    text = sj.render(
        sj.summarise(rows, {"a": _verdict("supported")}, seed=1, draws=50), judge="model-x"
    )
    assert (
        "model-x" in text
        and "single" in text
        and "not checked against the owner" in text
        and "worst case" in text
        and "unreadable" in text
    )


# --- the owner's check --------------------------------------------------------------------------------------------------------


def test_the_tag_sheet_hides_the_judge_and_is_in_spanish(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"{g}{i}",
            "group": g,
            "bucket": "0",
            "weight": "1",
            "source_text": "fuente",
            "summary_text": "resumen",
        }
        for g in ("single", "multi")
        for i in range(10)
    ]
    path = tmp_path / "to_tag.csv"
    sj.write_tag_sheet(path, rows, per_group=4, seed=3)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        tagged = list(csv.DictReader(fh))
    assert len(tagged) == 8
    assert {
        "id",
        "grupo",
        "texto_fuente",
        "resumen",
        "afirmaciones_sin_respaldo",
        "evento_principal",
        "notas",
    } <= set(tagged[0])
    assert not any("judge" in k or "verdict" in k for k in tagged[0])
    with pytest.raises(FileExistsError):
        sj.write_tag_sheet(path, rows, per_group=4, seed=3)


def test_agreement_is_raw_and_chance_corrected_and_refuses_a_blank_mark(tmp_path: Path) -> None:
    verdicts = {
        f"i{k}": _verdict(*(["supported"] if k < 6 else ["unsupported"]), covered=k % 2 == 0)
        for k in range(10)
    }
    tags = [
        {
            "id": f"i{k}",
            "grupo": "single",
            "afirmaciones_sin_respaldo": "no" if k < 6 else "si",
            "evento_principal": "si" if k % 2 == 0 else "no",
        }
        for k in range(10)
    ]
    result = sj.agreement(tags, verdicts)
    assert {k: result["single"]["faithful"][k] for k in ("n", "agree", "raw", "kappa")} == {
        "n": 10,
        "agree": 10,
        "raw": 1.0,
        "kappa": 1.0,
    }
    tags[0]["afirmaciones_sin_respaldo"] = (
        "si"  # the owner finds a problem the judge did not: one disagreement
    )
    result = sj.agreement(tags, verdicts)
    assert result["single"]["faithful"]["agree"] == 9
    assert result["single"]["faithful"]["kappa"] < 1.0
    tags[1]["evento_principal"] = ""
    with pytest.raises(ValueError, match="blank"):
        sj.agreement(tags, verdicts)


# --- review of #180 -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "{}",
        '{"claims": []}',
        '{"main_event_covered": true}',
        '{"claims": [], "main_event": "x"}',
    ],
)
def test_a_reply_without_the_claims_or_without_the_main_event_is_not_a_verdict(reply: str) -> None:
    """M1: the defaults of the model counted an empty object as 'main event covered'; both keys are now required."""
    agent = StubAgent([reply, reply])
    assert sj.judge_one(agent, "src", "sum").parse_failed
    assert len(agent.prompts) == 2  # it went through the repair prompt first


def test_the_three_summary_rates_have_one_denominator() -> None:
    """M1: an item with no claim to check is in none of them, so no rate is raised by it."""
    rows = [
        {
            "id": f"s{i}",
            "group": "single",
            "bucket": "0",
            "weight": "1",
            "pop_n": "3",
            "source_chars": "10",
            "summary_chars": "5",
            "compression": "0.5",
            "source_tickers": "{}",
            "summary_tickers": "{}",
        }
        for i in range(3)
    ]
    verdicts = {
        "s0": _verdict("supported", covered=True),
        "s1": _verdict("contradicted", covered=False),
        "s2": _verdict(covered=True),  # no claims: counted apart
    }
    single = sj.summarise(rows, verdicts, draws=20)["single"]
    assert single["no_claims"] == 1
    assert (
        single["clean"]["n"]
        == single["main_event_covered"]["n"]
        == single["any_contradicted"]["n"]
        == 2
    )
    assert single["main_event_covered"]["estimate"] == pytest.approx(0.5)


def test_every_verdict_carries_the_judge_the_prompt_and_the_revision(tmp_path: Path) -> None:
    """M2."""
    rows = [
        {
            "id": "r0",
            "group": "single",
            "source_text": "s",
            "summary_text": "t",
            "summary_revision": "summary-x",
        }
    ]
    out = tmp_path / "v.jsonl"
    sj.run_judge(rows, lambda: StubAgent([GOOD]), out, workers=1, judge_model="model-a")
    record = sj.read_verdicts(out)["r0"]
    assert record["judge_model"] == "model-a"
    assert record["prompt_sha"] == sj.PROMPT_SHA
    assert record["summary_revision"] == "summary-x"


def test_a_resumed_run_under_another_judge_or_prompt_judges_again_and_the_report_refuses_a_mixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"id": f"r{i}", "group": "single", "source_text": "s", "summary_text": "t"}
        for i in range(2)
    ]
    out = tmp_path / "v.jsonl"
    sj.run_judge(rows[:1], lambda: StubAgent([GOOD]), out, workers=1, judge_model="model-a")
    # the second run is under another model: r0 is judged again, r1 is new
    agent = StubAgent([GOOD])
    sj.run_judge(rows, lambda: agent, out, workers=1, judge_model="model-b")
    assert len(agent.prompts) == 2
    assert {v["judge_model"] for v in sj.read_verdicts(out).values()} == {"model-b"}
    # the same model with another prompt: judged again too
    monkeypatch.setattr(sj, "PROMPT_SHA", "other-prompt")
    agent = StubAgent([GOOD])
    sj.run_judge(rows, lambda: agent, out, workers=1, judge_model="model-b")
    assert len(agent.prompts) == 2
    # a file that holds two judges (a run that stopped half way) is not reported
    mixed = {
        "a": {"judge_model": "m1", "prompt_sha": "p"},
        "b": {"judge_model": "m2", "prompt_sha": "p"},
    }
    with pytest.raises(ValueError, match="more than one judge"):
        sj.judge_of(mixed)
    assert sj.judge_of({"a": {"judge_model": "m1", "prompt_sha": "p"}}) == "m1"
    with pytest.raises(ValueError, match="provenance"):
        sj.judge_of({"a": {"claims": []}})


class Outage:
    """An agent whose provider fails: a wrong key, a rate limit, a network drop."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> object:
        self.calls += 1
        raise RuntimeError("401 unauthorized")


def test_a_provider_error_is_not_a_verdict(tmp_path: Path) -> None:
    """M3: it raises, so nothing is written for it and the report cannot call it an unreadable reply."""
    with pytest.raises(sj.ProviderError, match="401"):
        sj.judge_one(Outage(), "src", "sum")


def test_consecutive_provider_errors_stop_the_run_and_write_nothing(tmp_path: Path) -> None:
    rows = [
        {"id": f"r{i}", "group": "single", "source_text": "s", "summary_text": "t"}
        for i in range(20)
    ]
    out = tmp_path / "v.jsonl"
    agent = Outage()
    with pytest.raises(sj.JudgeOutage, match="401"):
        sj.run_judge(rows, lambda: agent, out, workers=1, judge_model="m")
    assert agent.calls == sj.MAX_CONSECUTIVE_ERRORS  # it stopped, it did not go through 20 rows
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""


def test_an_error_between_good_replies_does_not_stop_the_run(tmp_path: Path) -> None:
    class Flaky:
        n = 0

        def __call__(self, prompt: str) -> object:
            Flaky.n += 1
            if Flaky.n % 2 == 0:
                raise RuntimeError("timeout")
            return GOOD

    # more errors in all than the run tolerates in a row, none of them in a row: a good reply between them resets the count
    n = 2 * sj.MAX_CONSECUTIVE_ERRORS + 2
    rows = [
        {"id": f"r{i}", "group": "single", "source_text": "s", "summary_text": "t"}
        for i in range(n)
    ]
    out = tmp_path / "v.jsonl"
    counts = sj.run_judge(rows, Flaky, out, workers=1, judge_model="m")
    assert counts["provider_errors"] == n // 2 and counts["judged"] == n // 2
    assert set(sj.read_verdicts(out)) == {
        f"r{i}" for i in range(0, n, 2)
    }  # the ones that failed are not recorded: the next run judges them


def test_an_unreadable_reply_is_still_recorded_apart_from_a_provider_error(tmp_path: Path) -> None:
    rows = [{"id": "r0", "group": "single", "source_text": "s", "summary_text": "t"}]
    out = tmp_path / "v.jsonl"
    counts = sj.run_judge(
        rows, lambda: StubAgent(["prose", "prose"]), out, workers=1, judge_model="m"
    )
    assert counts == {
        "judged": 0,
        "parse_failed": 1,
        "provider_errors": 0,
        "longest_verdict_chars": 0,
    }


def test_the_allocation_never_gives_more_than_a_bucket_has_and_always_stops() -> None:
    """S2: it used to hang on a variant that over-counted; now the loop ends when there is no room."""
    for pop in ({0: 3, 1: 3, 2: 3}, {0: 1, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 50, 1: 2, 2: 7}):
        for n in (0, 1, 5, 150):
            got = sj.allocate(pop, n)
            assert all(got[b] <= pop[b] for b in pop)
            assert sum(got.values()) == min(n, sum(pop.values()))


# the owner's slice must be able to say something about a rare event (S1)


def _flag_rows_and_verdicts(
    n_flagged: int, n_clean: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows, verdicts = [], {}
    for i in range(n_flagged + n_clean):
        rid = f"single:{i}"
        rows.append(
            {
                "id": rid,
                "group": "single",
                "bucket": "0",
                "weight": "10",
                "source_text": "f",
                "summary_text": "r",
            }
        )
        verdicts[rid] = {
            **(_verdict("supported", "unsupported") if i < n_flagged else _verdict("supported")),
            "judge_model": "m",
            "prompt_sha": "p",
        }
    return rows, verdicts


def test_the_slice_holds_the_judges_flagged_items_and_a_clean_control(tmp_path: Path) -> None:
    rows, verdicts = _flag_rows_and_verdicts(n_flagged=6, n_clean=94)
    path = tmp_path / "to_tag.csv"
    ids = sj.write_tag_sheet(path, rows, per_group=10, seed=3, verdicts=verdicts)
    strata = json.loads(path.with_suffix(".strata.json").read_text(encoding="utf-8"))
    assert set(strata["stratum"]) == set(ids)
    kinds = list(strata["stratum"].values())
    assert (
        kinds.count("flagged") == 5 and kinds.count("control") == 5
    )  # at most half the slice is flagged
    assert strata["population"]["single"] == {
        "flagged": 60.0,
        "clean": 940.0,
    }  # the sample's weights, to weigh the marks back
    with path.open(encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert not any(
        "judge" in h or "stratum" in h for h in header
    )  # the owner does not see which is which


def test_with_few_flagged_items_the_control_fills_the_slice(tmp_path: Path) -> None:
    rows, verdicts = _flag_rows_and_verdicts(n_flagged=2, n_clean=98)
    ids = sj.write_tag_sheet(tmp_path / "to_tag.csv", rows, per_group=10, seed=3, verdicts=verdicts)
    strata = json.loads((tmp_path / "to_tag.strata.json").read_text(encoding="utf-8"))
    assert len(ids) == 10 and list(strata["stratum"].values()).count("flagged") == 2


def test_agreement_with_strata_gives_the_owners_rate_in_each_stratum_and_the_judges_precision_and_sensitivity() -> (
    None
):
    _rows, verdicts = _flag_rows_and_verdicts(n_flagged=10, n_clean=90)
    strata: dict[str, Any] = {
        "stratum": {
            **{f"single:{i}": "flagged" for i in range(5)},
            **{f"single:{i}": "control" for i in range(10, 15)},
        },
        "population": {"single": {"flagged": 100.0, "clean": 900.0}},
    }
    tags = []
    for rid, kind in strata["stratum"].items():
        # the owner agrees with the judge on 4 of the 5 flagged (precision 0.8) and finds 1 of the 5 clean ones unfaithful (miss rate 0.2)
        bad = (kind == "flagged" and rid != "single:0") or (
            kind == "control" and rid == "single:10"
        )
        tags.append(
            {
                "id": rid,
                "grupo": "single",
                "afirmaciones_sin_respaldo": "si" if bad else "no",
                "evento_principal": "si",
            }
        )
    got = sj.agreement(tags, verdicts, strata)["single"]["faithful"]
    assert got["flagged"]["owner_unfaithful"] == 4 and got["flagged"]["n"] == 5
    assert got["clean"]["owner_unfaithful"] == 1 and got["clean"]["n"] == 5
    assert got["precision"] == pytest.approx(0.8)
    # unfaithful per the owner: 0.8 * 100 flagged + 0.2 * 900 clean = 260; the judge found 80 of them
    assert got["sensitivity"] == pytest.approx(80 / 260)


def test_agreement_without_strata_says_a_random_slice_cannot_measure_a_rare_event() -> None:
    verdicts = {f"i{k}": _verdict("supported") for k in range(4)}
    tags = [
        {
            "id": f"i{k}",
            "grupo": "single",
            "afirmaciones_sin_respaldo": "no",
            "evento_principal": "si",
        }
        for k in range(4)
    ]
    got = sj.agreement(tags, verdicts)["single"]["faithful"]
    assert "sensitivity" not in got and got["note"].startswith("random slice")


# --- second review of #180 ------------------------------------------------------------------------------------------------------


def test_the_slice_remembers_the_judge_that_chose_it_and_agree_refuses_another(
    tmp_path: Path,
) -> None:
    """M1: a rerun under another model between the slice and the marks would compare the marks with a judge that did not choose them."""
    rows, verdicts = _flag_rows_and_verdicts(n_flagged=6, n_clean=24)
    path = tmp_path / "to_tag.csv"
    sj.write_tag_sheet(path, rows, per_group=6, seed=3, verdicts=verdicts)
    strata = json.loads(path.with_suffix(".strata.json").read_text(encoding="utf-8"))
    assert strata["judge"] == {"model": "m", "prompt_sha": "p"}
    tags = [
        {"id": rid, "grupo": "single", "afirmaciones_sin_respaldo": "no", "evento_principal": "si"}
        for rid in strata["stratum"]
    ]
    assert "single" in sj.agreement(tags, verdicts, strata)  # the same judge: fine
    other_model = {k: {**v, "judge_model": "m2"} for k, v in verdicts.items()}
    with pytest.raises(ValueError, match="another judge"):
        sj.agreement(tags, other_model, strata)
    other_prompt = {k: {**v, "prompt_sha": "p2"} for k, v in verdicts.items()}
    with pytest.raises(ValueError, match="another judge"):
        sj.agreement(tags, other_prompt, strata)


def test_a_slice_is_not_written_from_verdicts_of_two_judges(tmp_path: Path) -> None:
    rows, verdicts = _flag_rows_and_verdicts(n_flagged=6, n_clean=24)
    verdicts["single:0"] = {**verdicts["single:0"], "judge_model": "other"}
    with pytest.raises(ValueError, match="more than one judge"):
        sj.write_tag_sheet(tmp_path / "to_tag.csv", rows, per_group=6, seed=3, verdicts=verdicts)
    assert not (tmp_path / "to_tag.csv").exists()


def test_sensitivity_has_an_interval_and_zero_misses_in_a_small_control_do_not_make_it_certain() -> (
    None
):
    """M2: 0 unfaithful in 20 clean items still allows about 16% of the 950 clean ones, which would put the sensitivity near 0.15."""
    got = sj._strata_figures({"flagged": 50.0, "clean": 950.0}, [True] * 5, [False] * 20)
    assert got["sensitivity"] == pytest.approx(1.0)
    low, high = got["sensitivity_interval"]
    assert high == pytest.approx(1.0) and low < 0.3
    # a bigger control narrows it
    narrower = sj._strata_figures({"flagged": 50.0, "clean": 950.0}, [True] * 5, [False] * 200)
    assert narrower["sensitivity_interval"][0] > low
    # the interval always holds the point estimate
    mixed = sj._strata_figures(
        {"flagged": 100.0, "clean": 900.0}, [True] * 4 + [False], [True] + [False] * 4
    )
    assert (
        mixed["sensitivity_interval"][0] <= mixed["sensitivity"] <= mixed["sensitivity_interval"][1]
    )


def _report_rows(n: int, bucket_of: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": f"s{i}",
            "group": "single",
            "bucket": str(bucket_of(i)),
            "weight": "10",
            "pop_n": "40",
            "source_chars": "3000",
            "summary_chars": "300",
            "compression": "0.1",
            "source_tickers": "{}",
            "summary_tickers": "{}",
        }
        for i in range(n)
    ]


def test_the_unreadable_replies_are_counted_per_bucket_and_bounded_in_the_worst_case() -> None:
    """S1: replies that fail may concentrate in the longest sources; excluding them flatters that bucket."""
    rows = _report_rows(12, lambda i: i % 2)  # bucket 0 and 1
    verdicts = {r["id"]: _verdict("supported") for r in rows}
    for i in (1, 3, 5):  # three unreadable, all in bucket 1 (the long sources)
        verdicts[f"s{i}"] = _verdict(failed=True)
    single = sj.summarise(rows, verdicts, draws=100)["single"]
    assert (
        single["by_bucket"]["1"]["unreadable"] == 3 and single["by_bucket"]["0"]["unreadable"] == 0
    )
    assert single["clean"]["estimate"] == pytest.approx(1.0)  # of the readable ones
    assert single["clean_worst_case"]["estimate"] == pytest.approx(
        0.75
    )  # every unreadable counted as unfaithful
    assert single["clean_worst_case"]["n"] == 12


def test_the_claim_shares_have_an_interval_from_resampling_summaries() -> None:
    rows = _report_rows(20, lambda i: i % 2)
    verdicts = {
        r["id"]: _verdict("supported", "unsupported")
        if i < 6
        else _verdict("supported", "supported")
        for i, r in enumerate(rows)
    }
    single = sj.summarise(rows, verdicts, draws=300)["single"]
    share = single["unsupported_claim_share"]
    assert share["estimate"] == pytest.approx(6 / 40)
    assert 0 <= share["low"] < share["estimate"] < share["high"] <= 1
    assert share["n"] == 40  # claims
    assert single["contradicted_claim_share"]["estimate"] == pytest.approx(0.0)


def test_the_longest_verdict_is_reported_so_a_cap_on_the_reply_would_show(tmp_path: Path) -> None:
    rows = [{"id": "r0", "group": "single", "source_text": "s", "summary_text": "t"}]
    counts = sj.run_judge(
        rows, lambda: StubAgent([GOOD]), tmp_path / "v.jsonl", workers=1, judge_model="m"
    )
    assert counts["longest_verdict_chars"] > 100
    assert (
        sj.read_verdicts(tmp_path / "v.jsonl")["r0"]["verdict_chars"]
        == counts["longest_verdict_chars"]
    )


def test_a_sampled_body_missing_from_the_source_copy_stops_the_draw_with_a_message(
    dbs: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    """S2."""
    results, source = dbs
    source.execute("DELETE FROM articles WHERE id <= 12")
    source.commit()
    with pytest.raises(ValueError, match=r"lacks \d+ of \d+ bodies"):
        sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=12, seed=7)


# --- third review of #180 ----------------------------------------------------------------------------------------------------


def _stratified_marks(
    n_flagged: int, n_clean: int, owner_flags_clean: int
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], dict[str, Any]]:
    """A slice of ``n_flagged`` flagged items (the owner agrees they are unfaithful) and ``n_clean`` control items, of which the owner finds
    ``owner_flags_clean`` unfaithful (the judge missed them); the sample behind it holds 10 flagged and 990 clean items."""
    verdicts: dict[str, dict[str, Any]] = {}
    tags: list[dict[str, str]] = []
    stratum: dict[str, str] = {}
    for i in range(n_flagged):
        rid = f"single:f{i}"
        verdicts[rid] = {
            **_verdict("supported", "unsupported"),
            "judge_model": "m",
            "prompt_sha": "p",
        }
        stratum[rid] = "flagged"
        tags.append(
            {
                "id": rid,
                "grupo": "single",
                "afirmaciones_sin_respaldo": "si",
                "evento_principal": "si",
            }
        )
    for i in range(n_clean):
        rid = f"single:c{i}"
        verdicts[rid] = {**_verdict("supported"), "judge_model": "m", "prompt_sha": "p"}
        stratum[rid] = "control"
        bad = i < owner_flags_clean
        tags.append(
            {
                "id": rid,
                "grupo": "single",
                "afirmaciones_sin_respaldo": "si" if bad else "no",
                "evento_principal": "si",
            }
        )
    strata = {
        "judge": {"model": "m", "prompt_sha": "p"},
        "stratum": stratum,
        "population": {"single": {"flagged": 10.0, "clean": 990.0}},
    }
    return tags, verdicts, strata


def test_the_agreement_is_weighed_back_to_the_sample_and_the_slice_figures_are_named() -> None:
    """M1: half the slice is flagged by construction; the agreement of the sample is nearer the clean stratum's."""
    tags, verdicts, strata = _stratified_marks(n_flagged=5, n_clean=5, owner_flags_clean=5)
    got = sj.agreement(tags, verdicts, strata)["single"]["faithful"]
    assert got["slice"]["raw"] == pytest.approx(
        0.5
    )  # 5 agree (the flagged), 5 do not (the control the owner finds unfaithful)
    assert got["raw"] == pytest.approx(
        10 / 1000
    )  # the sample is 99% clean-judged, and there the owner disagrees
    event = sj.agreement(tags, verdicts, strata)["single"]["main_event"]
    assert event["raw"] == pytest.approx(1.0) and "slice" in event
    # without strata the slice is the sample (a random draw): raw and kappa are as before and there is no slice_ key
    plain = sj.agreement(tags, verdicts)["single"]["faithful"]
    assert "slice" not in plain and plain["raw"] == pytest.approx(0.5)


def test_the_precision_comes_with_its_interval() -> None:
    tags, verdicts, strata = _stratified_marks(n_flagged=5, n_clean=5, owner_flags_clean=1)
    got = sj.agreement(tags, verdicts, strata)["single"]["faithful"]
    low, high = got["precision_interval"]
    assert low < got["precision"] <= high + 1e-9 and high <= 1.0 + 1e-9


def test_the_marks_outside_the_judges_population_are_counted() -> None:
    """S2: the sensitivity is conditional on a readable verdict with a claim; the others are named."""
    tags, verdicts, strata = _stratified_marks(n_flagged=4, n_clean=4, owner_flags_clean=0)
    verdicts["single:f0"] = {**_verdict(failed=True), "judge_model": "m", "prompt_sha": "p"}
    verdicts["single:c0"] = {
        **_verdict(),
        "judge_model": "m",
        "prompt_sha": "p",
    }  # no claim to check
    got = sj.agreement(tags, verdicts, strata)["single"]["faithful"]
    assert got["outside_population"] == 2


def test_the_claim_cap_is_counted_and_the_prompt_names_it() -> None:
    """S1: a summary with as many claims as the cap may have more; it is only partly checked."""
    assert f"At most {sj.MAX_CLAIMS} claims" in sj.PROMPT
    assert sj.item_measures(_verdict(*["supported"] * sj.MAX_CLAIMS))["at_cap"] is True
    assert sj.item_measures(_verdict(*["supported"] * (sj.MAX_CLAIMS - 1)))["at_cap"] is False
    rows = _report_rows(4, lambda i: 0)
    verdicts = {
        r["id"]: _verdict(*["supported"] * (sj.MAX_CLAIMS if i < 3 else 2))
        for i, r in enumerate(rows)
    }
    single = sj.summarise(rows, verdicts, draws=20)["single"]
    assert single["at_cap"] == 3
    assert "at the" in sj.render({"single": single}, judge="m") and "partly checked" in sj.render(
        {"single": single}, judge="m"
    )


def test_the_sample_records_what_it_was_drawn_from_and_has_an_identity(
    dbs: tuple[sqlite3.Connection, sqlite3.Connection], tmp_path: Path
) -> None:
    """M2: the ids of the sub-news mean nothing without the cut revision."""
    results, source = dbs
    rows = sj.draw(results, source, SUM_REV, SCOPE_REV, CUT_REV, per_group=6, seed=7)
    meta = sj.make_meta(
        rows,
        SUM_REV,
        SCOPE_REV,
        CUT_REV,
        per_group=6,
        seed=7,
        results_db=tmp_path / "res.db",
        source_db=tmp_path / "src.db",
    )
    assert (
        meta["scope_revision"] == SCOPE_REV
        and meta["cut_revision"] == CUT_REV
        and meta["seed"] == 7
    )
    assert meta["populations"]["single"] == 18 and meta["populations"]["multi"] == 30
    path = tmp_path / "sample.csv"
    sj.write_sample(path, rows, meta)
    assert sj.read_meta(path)["sample_id"] == meta["sample_id"]
    # the same draw has the same identity; another seed, another cut revision or other rows another one
    again = sj.make_meta(
        rows,
        SUM_REV,
        SCOPE_REV,
        CUT_REV,
        per_group=6,
        seed=7,
        results_db=tmp_path / "res.db",
        source_db=tmp_path / "src.db",
    )
    assert again["sample_id"] == meta["sample_id"]
    assert (
        sj.make_meta(
            rows,
            SUM_REV,
            SCOPE_REV,
            "units5-other",
            per_group=6,
            seed=7,
            results_db=tmp_path / "res.db",
            source_db=tmp_path / "src.db",
        )["sample_id"]
        != meta["sample_id"]
    )
    assert (
        sj.make_meta(
            rows[:-1],
            SUM_REV,
            SCOPE_REV,
            CUT_REV,
            per_group=6,
            seed=7,
            results_db=tmp_path / "res.db",
            source_db=tmp_path / "src.db",
        )["sample_id"]
        != meta["sample_id"]
    )


def test_verdicts_of_another_sample_are_refused_and_a_run_under_another_sample_judges_again(
    tmp_path: Path,
) -> None:
    rows = [{"id": "r0", "group": "single", "source_text": "s", "summary_text": "t"}]
    out = tmp_path / "v.jsonl"
    sj.run_judge(rows, lambda: StubAgent([GOOD]), out, workers=1, judge_model="m", sample_id="aaa")
    assert sj.read_verdicts(out)["r0"]["sample_id"] == "aaa"
    sj.check_sample(sj.read_verdicts(out), "aaa")  # the same sample: fine
    with pytest.raises(ValueError, match="another sample"):
        sj.check_sample(sj.read_verdicts(out), "bbb")
    with pytest.raises(ValueError, match="another sample"):
        sj.check_sample(
            {"r0": {"claims": []}}, "aaa"
        )  # a verdict that does not say which sample it is of
    agent = StubAgent([GOOD])
    sj.run_judge(rows, lambda: agent, out, workers=1, judge_model="m", sample_id="bbb")
    assert len(agent.prompts) == 1 and sj.read_verdicts(out)["r0"]["sample_id"] == "bbb"


def test_the_strata_remember_the_sample_and_agree_refuses_the_verdicts_of_another(
    tmp_path: Path,
) -> None:
    rows, verdicts = _flag_rows_and_verdicts(n_flagged=6, n_clean=24)
    verdicts = {k: {**v, "sample_id": "aaa"} for k, v in verdicts.items()}
    path = tmp_path / "to_tag.csv"
    sj.write_tag_sheet(path, rows, per_group=6, seed=3, verdicts=verdicts, sample_id="aaa")
    strata = json.loads(path.with_suffix(".strata.json").read_text(encoding="utf-8"))
    assert strata["sample_id"] == "aaa"
    tags = [
        {"id": rid, "grupo": "single", "afirmaciones_sin_respaldo": "no", "evento_principal": "si"}
        for rid in strata["stratum"]
    ]
    assert "single" in sj.agreement(tags, verdicts, strata)
    with pytest.raises(ValueError, match="another sample"):
        sj.agreement(tags, {k: {**v, "sample_id": "bbb"} for k, v in verdicts.items()}, strata)
    with pytest.raises(ValueError, match="another sample"):
        sj.write_tag_sheet(
            tmp_path / "other.csv", rows, per_group=6, seed=3, verdicts=verdicts, sample_id="ccc"
        )


# --- fourth review of #180 -------------------------------------------------------------------------------------------------


def _manager_slice() -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], dict[str, Any]]:
    """5 flagged items the owner agrees are unfaithful and 15 control items of which the owner finds one unfaithful, in a sample of 10 flagged
    and 990 clean: the slice the default draw gives for a rare event."""
    return _stratified_marks(n_flagged=5, n_clean=15, owner_flags_clean=1)


def test_the_weighed_agreement_and_kappa_carry_an_interval_that_one_mark_can_move() -> None:
    """M1: one control mark weighs 66, so the weighed figures swing on it and say so."""
    tags, verdicts, strata = _manager_slice()
    got = sj.agreement(tags, verdicts, strata, draws=400)["single"]["faithful"]
    assert got["raw"] == pytest.approx(0.934)
    low, high = got["raw_interval"]
    assert low <= got["raw"] <= high and high - low >= 0.15
    k_low, k_high = got["kappa_interval"]
    assert k_low <= got["kappa"] <= k_high or got["kappa"] is None
    event = sj.agreement(tags, verdicts, strata, draws=400)["single"]["main_event"]
    assert "raw_interval" in event


def test_the_slice_figures_are_under_their_own_key_with_a_note_and_not_at_the_top() -> None:
    tags, verdicts, strata = _manager_slice()
    got = sj.agreement(tags, verdicts, strata, draws=100)["single"]["faithful"]
    assert "slice_raw" not in got and "slice_kappa" not in got
    assert got["slice"]["kappa"] == pytest.approx(
        0.875, abs=0.01
    )  # the flattering one: it is about a slice built half flagged
    assert "half flagged" in got["note"] and "interval" in got["note"]
    # a random slice has neither the slice key nor the interval
    plain = sj.agreement(tags, verdicts)["single"]["faithful"]
    assert "slice" not in plain and "raw_interval" not in plain


def test_the_interval_is_reproducible_and_a_slice_of_one_stratum_has_none_to_resample() -> None:
    tags, verdicts, strata = _manager_slice()
    a = sj.agreement(tags, verdicts, strata, seed=3, draws=200)["single"]["faithful"][
        "raw_interval"
    ]
    b = sj.agreement(tags, verdicts, strata, seed=3, draws=200)["single"]["faithful"][
        "raw_interval"
    ]
    assert a == b


def test_the_interval_resamples_within_the_strata_so_a_stratum_keeps_its_size_and_its_weight() -> (
    None
):
    """With one flagged and one control item, resampling within the strata gives the same slice every time (no interval to speak of); resampling
    over all the items would drop one of them in half the draws and invent a spread the design does not have."""
    tags, verdicts, strata = _stratified_marks(n_flagged=1, n_clean=1, owner_flags_clean=1)
    got = sj.agreement(tags, verdicts, strata, draws=200)["single"]["faithful"]
    low, high = got["raw_interval"]
    assert low == pytest.approx(high) == pytest.approx(got["raw"])


def test_the_interval_is_the_95_percent_band_not_the_extremes() -> None:
    values = [float(i) for i in range(1000)]
    low, high = sj._interval(values)
    assert low == pytest.approx(25.0) and high == pytest.approx(975.0)
    assert sj._interval([]) is None
