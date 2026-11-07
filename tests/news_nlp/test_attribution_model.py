"""news_nlp.attribution_model: the trained attribution model of step 4 (WI18 T-148b3, D27). The
transformer is never loaded: a fake scorer stands in, so the pairs, the masking, the choice of
negatives and the extractor's rules are what is tested."""

from __future__ import annotations

import importlib.util
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from news_nlp import attribution_model as am
from news_nlp import company_texts, triage
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


PARITY_SENTENCES = [
    "Apple and Microsoft rose; Apple Inc. led.",
    "Shares of $AAPL and (NASDAQ: MSFT) fell, and ORCL too.",
    "Oracle Corporation said Oracle would buy back stock.",
    "Nothing named here.",
    "Microsoft Corporation (NASDAQ: MSFT) reported.",
]


@pytest.mark.parametrize("sentence", PARITY_SENTENCES)
def test_mention_spans_and_resolve_read_the_same_matches(sentence: str) -> None:
    spans = [t for _, _, t in RESOLVER.mention_spans(sentence)]
    counted = RESOLVER.resolve(sentence, skip_attribution=False)
    assert set(spans) == set(counted)  # the same companies, whatever the overlaps
    assert all(spans.count(t) <= counted[t] for t in set(spans))  # never more than resolve counts


def test_an_overlapping_ticker_form_inside_a_name_is_counted_once_in_the_spans() -> None:
    text = "Microsoft Corporation (NASDAQ: MSFT) reported."
    assert [t for _, _, t in RESOLVER.mention_spans(text)] == ["MSFT", "MSFT"]


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
        own = {"Apple Inc.": "AAPL", "Microsoft Corporation": "MSFT", "Oracle Corporation": "ORCL"}[
            e.pair.candidate
        ]
        assert own not in [t for _, _, t in RESOLVER.mention_spans(e.pair.sentence)]
    assert {e.key: e.named for e in examples}["7#1#1"] is False  # "The iPhone maker also..."
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


def test_the_baseline_score_is_the_candidate_mentioned_in_the_previous_sentence() -> None:
    pair = am.Pair("Apple Inc.", "Apple rose 3%.", "It fell.")  # the short name: a mention
    assert am.named_in_context(pair, RESOLVER) == 1.0
    assert am.named_in_context(am.Pair("Apple Inc.", "Microsoft rose.", "It fell."), RESOLVER) == 0
    assert am.named_in_context(am.Pair("Apple Inc.", "", "It fell."), RESOLVER) == 0.0


def test_the_exact_name_floor_needs_the_legal_name_in_the_context() -> None:
    assert am.exact_name_in_context(am.Pair("Apple Inc.", "Apple Inc. rose.", "x")) == 1.0
    assert am.exact_name_in_context(am.Pair("Apple Inc.", "Apple rose.", "x")) == 0.0


def test_an_unnamed_sentence_is_not_marked_named() -> None:
    labels = [company_texts.SegmentLabel(1, "AAPL", "lead")]
    body = (
        "Stocks making the biggest moves midday:\n"
        "Apple said so today. Sales rose in the quarter for the maker.\n"
        "Another paragraph long enough to count as a segment here."
    )
    out = am.training_examples(
        body,
        labels,
        article_id=1,
        year=2023,
        resolver=RESOLVER,
        names=NAMES,
        rng=random.Random(1),  # noqa: S311
    )
    named = {e.key: e.named for e in out}
    assert named["1#1#0"] is True and named["1#1#1"] is False


def _script(monkeypatch: pytest.MonkeyPatch) -> Any:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
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


def test_the_trainer_refuses_to_run_unless_the_label_check_says_lead_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(monkeypatch)
    script.require_lead_only(_check_file(tmp_path / "ok.csv", 3))  # 30% right: lead only
    with pytest.raises(SystemExit):
        script.require_lead_only(_check_file(tmp_path / "high.csv", 9))  # 90% right: both kinds
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "key,article_id,segment_index,ticker,kind,year,correct,notes\n", encoding="utf-8"
    )
    with pytest.raises(MarkedFileError):
        script.require_lead_only(empty)


def _draw(monkeypatch: pytest.MonkeyPatch) -> Any:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "draw_faithfulness_2026_11_06", scripts / "draw_faithfulness_2026_11_06.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_model_extractor_has_the_baselines_signature_and_the_models_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draw = _draw(monkeypatch)
    scorer = FakeScorer({("Apple Inc.", "The iPhone maker also grew its buyback."): 0.9})
    extract = draw.model_extractor(scorer, NAMES)
    assert extract(BODY, RESOLVER)["AAPL"].endswith("grew its buyback.")


def test_the_model_extractor_needs_registering_and_a_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draw = _draw(monkeypatch)
    draw.MODEL_IDENTITY.clear()
    with pytest.raises(ValueError, match="not registered"):
        draw.extractor_revision("model")
    with pytest.raises(SystemExit):  # a Hub id without a revision is refused before any download
        draw.register_model("org/attribution", None, NAMES)
    draw.MODEL_IDENTITY[:] = ["org/attribution@abc"]
    assert draw.extractor_revision("model").startswith("extractor:model@org/attribution@abc@")
    draw.MODEL_IDENTITY.clear()


def test_evaluate_reports_every_view_and_a_view_without_pairs_is_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    pair = am.Pair("Apple Inc.", "Apple rose.", "It fell.")
    examples = [
        am.Example(pair, 1, "a", True, 2023, False),
        am.Example(pair, 0, "a", True, 2023, False),
    ]
    report = script.evaluate(examples, [0.9, 0.1], RESOLVER)
    assert report["all model"]["ranking"] == 1.0
    assert report["unnamed sentences model"]["pairs"] == 1
    assert report["before 2022 model"]["pairs"] == 0
    assert report["before 2022 model"]["ranking"] != report["before 2022 model"]["ranking"]  # nan
    assert report["all baseline"]["ranking"] == 0.5  # the previous sentence names Apple for both
    assert report["all exact name"]["ranking"] == 0.5  # the legal name is never written: a tie


class FakeConn:
    def __init__(self, bodies: dict[int, str]) -> None:
        self.bodies = bodies

    def execute(self, _sql: str, params: tuple[int]) -> Any:
        body = self.bodies.get(params[0])
        return type(
            "Cursor", (), {"fetchone": lambda _self: {"body_text": body} if body else None}
        )()


def _rows(article_id: int, split: str, body: str) -> list[dict[str, str]]:
    return [
        {
            "article_id": str(article_id),
            "split": split,
            "body_hash": triage.body_hash(body),
            "segment_index": "1",
            "ticker": "AAPL",
            "kind": "lead",
            "year": "2023",
        }
    ]


def test_examples_of_leaves_out_changed_bodies_and_val_copies_of_train_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    body = STRUCTURED
    copy = STRUCTURED + " Syndicated tail."  # same start, another hash
    monkeypatch.setattr(
        script.db, "connect_pipeline", lambda: FakeConn({1: body, 2: copy, 3: "changed"})
    )
    by_article = {
        1: _rows(1, "train", body),
        2: _rows(2, "val", copy),
        3: _rows(3, "val", body),  # the SOURCE text is no longer the one hashed
    }
    keys: set[str] = set()
    train, tcount = script.examples_of(by_article, "train", RESOLVER, NAMES, 1, None, keys)
    val, vcount = script.examples_of(by_article, "val", RESOLVER, NAMES, 1, None, keys)
    assert tcount["bodies"] == 1 and train
    assert vcount["near-duplicate"] == 1 and vcount["hash mismatch"] == 1 and not val
    limited, lcount = script.examples_of(by_article, "train", RESOLVER, NAMES, 1, 0, set())
    assert not limited and lcount["bodies"] == 0  # --limit-bodies 0 takes nothing


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, first: list[str], second: list[str], **kwargs: Any) -> Any:
        self.calls.append((first, second, kwargs))
        return {}


def test_encode_puts_the_candidate_first_and_the_context_with_the_sentence_second() -> None:
    scorer = am.TransformerScorer.__new__(am.TransformerScorer)
    scorer.tokenizer = FakeTokenizer()
    scorer.encode([am.Pair("Apple Inc.", "Before.", "Now."), am.Pair("Oracle", "", "Only.")])
    first, second, kwargs = scorer.tokenizer.calls[0]
    assert first == ["Apple Inc.", "Oracle"]
    assert second == ["Before. [SEP] Now.", "Only."]  # no context: the sentence alone
    assert kwargs["truncation"] == "only_second" and kwargs["max_length"] == am.MAX_LENGTH


def test_the_local_model_identity_covers_weights_and_tokenizer_and_the_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draw = _draw(monkeypatch)
    with pytest.raises(SystemExit, match=r"no model\.safetensors"):
        draw.local_identity(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"w")
    (tmp_path / "vocab.txt").write_text("a", encoding="utf-8")
    first = draw.local_identity(tmp_path)
    (tmp_path / "vocab.txt").write_text("b", encoding="utf-8")
    assert draw.local_identity(tmp_path) != first  # the tokenizer is part of the model
    monkeypatch.setattr(draw.attribution_model, "TransformerScorer", lambda *_a, **_k: FakeScorer())
    draw.register_model(str(tmp_path), None, NAMES)
    one = draw.MODEL_IDENTITY[0]
    draw.register_model(str(tmp_path), None, {**NAMES, "ORCL": "Oracle"})
    assert one.startswith("local:") and draw.MODEL_IDENTITY[0] != one  # other names: other texts
    assert f"@t{am.THRESHOLD}" in one
    draw.MODEL_IDENTITY.clear()
    draw.EXTRACTORS.pop("model", None)
