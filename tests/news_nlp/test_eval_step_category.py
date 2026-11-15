"""The step-7 judge (T-209): blind prompt, 12 labels equal to the step's, parse/repair, the figures."""

from __future__ import annotations

from news_category_stage import LABELS
from news_nlp.eval import step_category as sc


class FakeAgent:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else "(exhausted)"


def test_labels_are_the_labels_of_the_step() -> None:
    assert sc.STEP_LABELS == LABELS


def test_prompt_names_every_label_and_asks_for_json() -> None:
    text = sc.system_prompt()
    assert "JSON" in text
    for label in LABELS:
        assert f"`{label}`" in text


def test_the_judge_is_blind_to_the_model() -> None:
    prompt = sc.user_prompt("Apple", "Apple launches a phone.")
    assert "Apple" in prompt and "launches a phone" in prompt
    assert "prediction" not in prompt.lower() and "model" not in prompt.lower()


def test_clean_reply_parses() -> None:
    agent = FakeAgent(
        ['{"ideal_slug": "product_innovation", "second_slug": "other", "rationale": "launch"}']
    )
    v = sc.judge_step_category(agent, "Apple", "t")
    assert (v.ideal_slug, v.second_slug, v.parse_failed) == ("product_innovation", "other", False)
    assert len(agent.prompts) == 1


def test_unknown_slug_is_repaired_then_fails_closed() -> None:
    agent = FakeAgent(['{"ideal_slug": "weather"}', '{"ideal_slug": "weather"}'])
    v = sc.judge_step_category(agent, "Apple", "t")
    assert v.parse_failed is True and len(agent.prompts) == 2


def test_unknown_second_slug_is_dropped_not_fatal() -> None:
    agent = FakeAgent(['{"ideal_slug": "other", "second_slug": "weather"}'])
    v = sc.judge_step_category(agent, "Apple", "t")
    assert v.second_slug is None and v.parse_failed is False


def test_agreement_strict_and_lenient() -> None:
    judge = ["a", "b", "c", "d"]
    second = [None, "x", "d", None]
    model = ["a", "x", "c", "z"]
    got = sc.agreement(judge, model, second)
    assert got == {"n": 4, "strict": 0.5, "lenient": 0.75}


def test_sign_test_known_values() -> None:
    assert sc.sign_test(0, 0) == 1.0
    assert sc.sign_test(5, 5) == 1.0
    assert abs(sc.sign_test(9, 1) - 0.021484375) < 1e-9


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    hits = [True] * 30 + [False] * 10
    lo, hi = sc.bootstrap_ci(hits, boot=500, seed=1)
    assert lo <= 0.75 <= hi
    assert (lo, hi) == sc.bootstrap_ci(hits, boot=500, seed=1)


def _row(
    judge: str, selected: str, untouched: str, tag: str = "other", failed: bool = False
) -> dict[str, str]:
    return {
        "judge": judge,
        "second": "",
        "parse_failed": str(failed),
        "tag": tag,
        "selected": selected,
        "untouched": untouched,
    }


ROWS = [
    _row("earnings_performance", "earnings_performance", "other"),  # only the fine-tune right
    _row("earnings_performance", "earnings_performance", "other"),
    _row("legal_regulatory", "other", "legal_regulatory"),  # only the untouched right
    _row("product_innovation", "product_innovation", "product_innovation"),  # both
    _row("labor_human_capital", "other", "other"),  # neither
    _row("earnings_performance", "other", "other", failed=True),  # excluded
]


def test_summary_counts_the_paired_texts_and_drops_parse_failures() -> None:
    s = sc.summary(ROWS)
    assert s["n"] == 5 and s["parse_failed"] == 1
    assert s["selected"] == 0.6 and s["untouched"] == 0.4
    assert (s["selected_only"], s["untouched_only"]) == (2, 1)
    assert s["sign_test_p"] == sc.sign_test(2, 1)


def test_report_has_the_header_the_models_and_the_per_class_table() -> None:
    text = sc.report(ROWS, ["- Judge model: `m`."])
    assert text.startswith("# The LLM judge on step 7: 6 real sub-news")
    assert "- Judge model: `m`." in text
    assert "5 judged, 1 parse failures" in text
    assert "| 60.0% [" in text
    assert (
        "| earnings_performance | 2 | 2 | 0 |" in text
    )  # judge puts 2 texts in `a`; fine-tune right on 2, untouched on 0
    assert "`labor_human_capital` -> `other`: 1" in text


def test_compare_runs_counts_changed_labels_and_scores_the_texts_outside_the_first_run() -> None:
    def r(idx: str, judge: str, selected: str, untouched: str) -> dict[str, str]:
        row = _row(judge, selected, untouched)
        row["idx"] = idx
        return row

    small = [r("1", "other", "other", "legal_regulatory"), r("2", "other", "other", "other")]
    big = [
        r("1", "legal_regulatory", "other", "legal_regulatory"),  # the judge changed its label
        r("2", "other", "other", "other"),
        r("3", "other", "other", "legal_regulatory"),  # outside: fine-tune right only
        r("4", "other", "legal_regulatory", "legal_regulatory"),  # outside: neither
    ]
    text = "\n".join(sc.compare_runs(big, small))
    assert "changed on 1 of 2" in text
    assert (
        "The 2 texts the first run did not see: fine-tune 50.0%, untouched 0.0%; paired 1 against 0"
        in text
    )
