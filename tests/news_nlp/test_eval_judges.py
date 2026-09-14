"""judges.py: JSON parse / repair / fallback, no real LLM."""

from __future__ import annotations

from news_nlp.eval import judges
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.verdicts import (
    CategoryVerdict,
    NerVerdict,
    SectorIntroVerdict,
    SentimentVerdict,
    SummaryVerdict,
)


class FakeAgent:
    """Returns canned replies in order; records the prompts it was called with."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else "(exhausted)"


def _item(prediction: dict) -> EvalItem:
    return EvalItem(
        article_id=1,
        bucket="representative",
        stratum_population=1,
        title="T",
        body_text="Body.",
        prediction=prediction,
    )


def test_prompts_load_for_every_stage() -> None:
    for stage in ("sentiment", "category", "ner", "c_summary", "sector_summary"):
        assert "JSON" in judges.load_prompt(stage)


def test_sentiment_clean_json_parses_and_recomputes_agree() -> None:
    reply = (
        '{"agrees": true, "ideal_label": "negative", "severity": 2, "rationale": "guidance cut"}'
    )
    agent = FakeAgent([reply])
    v = judges.judge_sentiment(agent, _item({"label": "positive"}))
    assert isinstance(v, SentimentVerdict)
    assert v.ideal_label == "negative"
    # `agrees` is recomputed from the label comparison, overriding the reply's flag
    assert v.agrees is False
    assert v.parse_failed is False
    assert len(agent.prompts) == 1


def test_sentiment_repairs_once_then_succeeds() -> None:
    agent = FakeAgent(
        ["not json at all", 'here you go: {"agrees": true, "ideal_label": "neutral"} thanks']
    )
    v = judges.judge_sentiment(agent, _item({"label": "neutral"}))
    assert v.parse_failed is False
    assert v.ideal_label == "neutral"
    assert v.agrees is True
    assert len(agent.prompts) == 2
    assert "ONLY the JSON object" in agent.prompts[1]


def test_sentiment_falls_back_after_repair_fails() -> None:
    agent = FakeAgent(["garbage", "still garbage"])
    v = judges.judge_sentiment(agent, _item({"label": "positive"}))
    assert v.parse_failed is True
    assert v.agrees is False
    assert len(agent.prompts) == 2


def test_agent_exception_falls_back_without_crashing() -> None:
    class Boom:
        def __call__(self, prompt: str) -> str:
            raise RuntimeError("provider down")

    v = judges.judge_category(Boom(), _item({"label": "other"}))
    assert isinstance(v, CategoryVerdict)
    assert v.parse_failed is True


def test_category_recomputes_agree_against_model_label() -> None:
    reply = '{"agrees": false, "ideal_slug": "earnings_performance", "severity": 0}'
    v = judges.judge_category(FakeAgent([reply]), _item({"label": "earnings_performance"}))
    assert v.agrees is True  # recomputed


def test_ner_and_summary_parse() -> None:
    ner_reply = (
        '{"wrong": [{"text": "the", "entity_type": "ORG"}], '
        '"missed": [{"text": "Berlin", "entity_type": "LOC"}], "rationale": "boundary errors"}'
    )
    nv = judges.judge_ner(FakeAgent([ner_reply]), _item({"entities": []}))
    assert isinstance(nv, NerVerdict)
    assert nv.wrong[0].text == "the"
    assert nv.missed[0].entity_type == "LOC"
    assert nv.parse_failed is False

    sum_reply = (
        '{"faithfulness": 4, "coverage": 3, "conciseness": 5, '
        '"hallucinations": [], "rationale": "fine"}'
    )
    sv = judges.judge_c_summary(FakeAgent([sum_reply]), _item({"summary_text": "s"}))
    assert isinstance(sv, SummaryVerdict)
    assert sv.faithfulness == 4


def test_sector_summary_prompt_frames_grounding_not_article() -> None:
    """The sector_summary judge must not use the generic ARTICLE framing --
    body_text is facts_json, not article text (sampling.py's
    _sector_summary_items docstring)."""
    item = EvalItem(
        article_id=99,
        bucket="representative",
        stratum_population=1,
        title="Tech / Software -- week 2026-08-10 to 2026-08-16",
        body_text='{"num_articles": 3, "sentiment": {"pct": {"positive": 100}}}',
        prediction={"intro_text": "Software saw 3 articles, all positive."},
    )
    reply = '{"faithfulness": 5, "hallucinations": [], "rationale": "matches the stats"}'
    agent = FakeAgent([reply])
    v = judges.judge_sector_summary(agent, item)
    assert isinstance(v, SectorIntroVerdict)
    assert v.faithfulness == 5
    assert v.parse_failed is False
    prompt = agent.prompts[0]
    assert "ARTICLE" not in prompt
    assert "GROUNDING DATA" in prompt
    assert item.body_text in prompt
    assert "intro_text" in prompt


def test_sector_summary_falls_back_on_unparseable_reply() -> None:
    v = judges.judge_sector_summary(FakeAgent(["nope", "still nope"]), _item({"intro_text": "x"}))
    assert isinstance(v, SectorIntroVerdict)
    assert v.parse_failed is True
