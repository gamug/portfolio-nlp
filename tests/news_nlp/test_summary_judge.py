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
    assert "model-x" in text and "single" in text and "not checked against the owner" in text


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
    assert result["single"]["faithful"] == {"n": 10, "agree": 10, "raw": 1.0, "kappa": 1.0}
    tags[0]["afirmaciones_sin_respaldo"] = (
        "si"  # the owner finds a problem the judge did not: one disagreement
    )
    result = sj.agreement(tags, verdicts)
    assert result["single"]["faithful"]["agree"] == 9
    assert result["single"]["faithful"]["kappa"] < 1.0
    tags[1]["evento_principal"] = ""
    with pytest.raises(ValueError, match="blank"):
        sj.agreement(tags, verdicts)
