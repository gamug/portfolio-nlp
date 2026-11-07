"""news_nlp.attribution_model: the trained attribution model of step 4 (WI18 T-148b3, D27). The
transformer is never loaded: a fake scorer stands in, so the pairs, the masking, the choice of
negatives and the extractor's rules are what is tested."""

from __future__ import annotations

import importlib.util
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from news_nlp import attribution_model as am
from news_nlp import company_texts
from news_nlp.gold_metrics import MarkedFileError
from news_nlp.triage import Company, CompanyResolver

UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
    Company("ORCL", "Oracle Corporation", "Information Technology", "Systems Software"),
]
RESOLVER = CompanyResolver(UNIVERSE)
NAMES = {c.ticker: c.name for c in UNIVERSE}
STRUCTURED = (
    "Stocks making the biggest moves midday:\n"
    "Apple - Shares rose 3% after Apple raised its dividend. The iPhone maker also grew its buyback.\n"
    "Microsoft - Shares fell 2% as cloud growth slowed in the quarter.\n"
    "Oracle - Shares jumped 5% on a new contract with the Pentagon."
)


def test_mention_spans_give_positions_in_order_without_overlaps() -> None:
    text = "Apple and Microsoft rose; Apple Inc. led."
    spans = RESOLVER.mention_spans(text)
    assert [t for _, _, t in spans] == ["AAPL", "MSFT", "AAPL"]
    assert text[spans[1][0] : spans[1][1]] == "Microsoft"
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_mask_company_masks_only_that_company() -> None:
    masked = am.mask_company("Apple beat Microsoft this quarter.", RESOLVER, "AAPL")
    assert masked == f"{am.MASK} beat Microsoft this quarter."
    assert am.mask_company("Nothing here.", RESOLVER, "AAPL") == "Nothing here."


def _examples(rng_seed: int = 1) -> list[am.Example]:
    labels = [
        company_texts.SegmentLabel(1, "AAPL", "lead"),
        company_texts.SegmentLabel(2, "MSFT", "lead"),
        company_texts.SegmentLabel(3, "ORCL", "lead"),
    ]
    return am.training_examples(
        STRUCTURED,
        labels,
        article_id=7,
        year=2023,
        resolver=RESOLVER,
        names=NAMES,
        rng=random.Random(rng_seed),  # noqa: S311
    )


def test_every_sentence_of_a_lead_paragraph_is_a_positive_with_one_negative() -> None:
    examples = _examples()
    positives = [e for e in examples if e.label == 1]
    negatives = [e for e in examples if e.label == 0]
    assert len(positives) == 4 and len(negatives) == 4  # 2 + 1 + 1 sentences
    assert {e.key for e in positives} == {e.key for e in negatives}
    for e in positives:  # the company's own name is masked in the sentence, never shown
        assert "Apple" not in e.pair.sentence or e.pair.candidate != "Apple Inc."
    assert any(am.MASK in e.pair.sentence for e in positives)


def test_the_first_sentence_of_a_paragraph_is_negative_for_the_company_before_it() -> None:
    examples = _examples()
    msft_first = [e for e in examples if e.key == "7#2#0"]
    assert {e.pair.candidate for e in msft_first if e.label == 0} == {"Apple Inc."}
    assert all(e.first for e in msft_first)
    # its context is the last sentence of the paragraph before
    assert msft_first[0].pair.context.startswith("The iPhone maker")


def test_continuation_labels_are_not_trained_on() -> None:
    labels = [
        company_texts.SegmentLabel(1, "AAPL", "lead"),
        company_texts.SegmentLabel(2, "MSFT", "continuation"),
    ]
    out = am.training_examples(
        STRUCTURED,
        labels,
        article_id=7,
        year=2023,
        resolver=RESOLVER,
        names=NAMES,
        rng=random.Random(1),  # noqa: S311
    )
    assert {e.key.split("#")[1] for e in out} == {"1"}


class FakeScorer:
    """Scores 0.9 for the candidate a sentence repeats a word of, else 0.1."""

    def __init__(self, table: dict[tuple[str, str], float] | None = None) -> None:
        self.table = table or {}
        self.asked: list[am.Pair] = []

    def score(self, pairs: Sequence[am.Pair]) -> list[float]:
        self.asked += pairs
        return [self.table.get((p.candidate, p.sentence), 0.1) for p in pairs]


BODY = (
    "Apple raised its dividend on Monday. The iPhone maker also grew its buyback. "
    "Microsoft said cloud growth slowed. Azure demand cooled in the quarter."
)


def test_a_named_sentence_belongs_to_the_company_it_names_and_the_model_gives_the_rest() -> None:
    scorer = FakeScorer(
        {
            ("Apple Inc.", "The iPhone maker also grew its buyback."): 0.9,
            ("Microsoft Corporation", "Azure demand cooled in the quarter."): 0.8,
        }
    )
    texts = am.model_company_texts(BODY, RESOLVER, NAMES, scorer)
    assert (
        texts["AAPL"]
        == "Apple raised its dividend on Monday. The iPhone maker also grew its buyback."
    )
    assert (
        texts["MSFT"] == "Microsoft said cloud growth slowed. Azure demand cooled in the quarter."
    )
    asked = {(p.sentence) for p in scorer.asked}
    assert "Apple raised its dividend on Monday." not in asked  # named: not scored


def test_a_sentence_no_company_reaches_belongs_to_nobody_not_to_the_last_company() -> None:
    texts = am.model_company_texts(BODY, RESOLVER, NAMES, FakeScorer())  # every score 0.1
    assert texts == {
        "AAPL": "Apple raised its dividend on Monday.",
        "MSFT": "Microsoft said cloud growth slowed.",
    }


def test_a_company_the_body_names_always_gets_the_sentence_that_names_it() -> None:
    texts = am.model_company_texts(BODY, RESOLVER, NAMES, FakeScorer())
    assert set(texts) == {"AAPL", "MSFT"}


def test_the_threshold_decides_the_best_companys_sentence() -> None:
    scorer = FakeScorer({("Apple Inc.", "Azure demand cooled in the quarter."): 0.6})
    low = am.attribute_sentences(BODY, RESOLVER, NAMES, scorer, threshold=0.5)
    high = am.attribute_sentences(BODY, RESOLVER, NAMES, scorer, threshold=0.7)
    azure = "Azure demand cooled in the quarter."
    assert dict(low)[azure] == ("AAPL",)
    assert dict(high)[azure] == ()


def test_pair_accuracy_ranks_each_sentences_positive_against_its_negatives() -> None:
    pair = am.Pair("c", "", "s")
    examples = [
        am.Example(pair, 1, "a", True, 2023),
        am.Example(pair, 0, "a", True, 2023),
        am.Example(pair, 1, "b", False, 2023),
        am.Example(pair, 0, "b", False, 2023),
    ]
    figures = am.pair_accuracy(examples, [0.9, 0.2, 0.4, 0.4])
    assert figures["pairs"] == 2
    assert figures["ranking"] == pytest.approx(0.75)  # one pair right, one tied
    assert figures["threshold"] == pytest.approx(
        0.75
    )  # the tie at 0.4 is right for the 0, wrong for the 1


def test_the_baseline_score_is_the_candidate_named_in_the_previous_sentence() -> None:
    assert am.named_in_context(am.Pair("Apple Inc.", "Apple Inc. rose.", "It fell.")) == 1.0
    assert am.named_in_context(am.Pair("Apple Inc.", "Microsoft rose.", "It fell.")) == 0.0


def _script() -> Any:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "train_attribution_2026_11_08", scripts / "train_attribution_2026_11_08.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_file(path: Path, continuation_right: int) -> Path:
    rows = ["key,article_id,segment_index,ticker,kind,year,correct,notes"]
    for i in range(10):
        rows.append(f"{i}#1,{i},1,AAPL,lead,2023,yes,")
        rows.append(
            f"{i}#2,{i},2,AAPL,continuation,2023,{'yes' if i < continuation_right else 'no'},"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_the_trainer_refuses_to_run_unless_the_label_check_says_lead_only(tmp_path: Path) -> None:
    script = _script()
    script.require_lead_only(_check_file(tmp_path / "ok.csv", 3))  # 30% right: lead only
    with pytest.raises(SystemExit):
        script.require_lead_only(_check_file(tmp_path / "high.csv", 9))  # 90% right: both kinds
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "key,article_id,segment_index,ticker,kind,year,correct,notes\n", encoding="utf-8"
    )
    with pytest.raises(MarkedFileError):
        script.require_lead_only(empty)


def _draw() -> Any:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "draw_faithfulness_2026_11_06", scripts / "draw_faithfulness_2026_11_06.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_model_extractor_has_the_baselines_signature_and_the_models_texts() -> None:
    draw = _draw()
    scorer = FakeScorer({("Apple Inc.", "The iPhone maker also grew its buyback."): 0.9})
    extract = draw.model_extractor(scorer, NAMES)
    assert extract(BODY, RESOLVER)["AAPL"].endswith("grew its buyback.")


def test_the_model_extractor_needs_registering_and_a_pinned_revision() -> None:
    draw = _draw()
    draw.MODEL_IDENTITY.clear()
    with pytest.raises(ValueError, match="not registered"):
        draw.extractor_revision("model")
    with pytest.raises(SystemExit):  # a Hub id without a revision is refused before any download
        draw.register_model("org/attribution", None, NAMES)
    draw.MODEL_IDENTITY[:] = ["org/attribution@abc"]
    assert draw.extractor_revision("model").startswith("extractor:model@org/attribution@abc@")
    draw.MODEL_IDENTITY.clear()
