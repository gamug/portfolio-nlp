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
