"""The fine-tuning of the sentiment model on the summaries, read as a pair (T-148c3, D35)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

import sentiment_pair_train as spt
from news_nlp import retag

WORDS = [
    "acme",
    "bravo",
    "up",
    "down",
    "flat",
    "the",
    "stock",
    "was",
    "today",
    "x",
    "y",
    "z",
    "a",
    "b",
]


def _tiny_model(folder: Path) -> str:
    """A BERT of a few thousand weights and a tokenizer of ``WORDS``, saved to *folder* (no network)."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(WORDS)]
    folder.mkdir(parents=True, exist_ok=True)
    backend = Tokenizer(models.WordPiece({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))  # noqa: S106
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    specials = {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **specials)  # type: ignore[no-untyped-call]
    config = BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64,
        id2label=spt.ID2LABEL, label2id=spt.LABEL2ID,
    )  # fmt: skip
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(folder)  # type: ignore[no-untyped-call]
    tokenizer.save_pretrained(folder)
    return str(folder)


def _example(i: int, kind: str = "summary", label: int | None = None) -> spt.Example:
    word = ("up", "down", "flat")[i % 3]
    return spt.Example(
        f"{kind}{i}",
        "Acme" if i % 2 else "Bravo",
        f"the stock was {word} today",
        label if label is not None else i % 3,
        kind,
    )


# --- the owner's marks --------------------------------------------------------------------------


def _mark(
    key: str,
    text: str,
    label: str,
    *,
    ticker: str = "AAA",
    scope: str = "single",
    slice_: str = "train",
) -> tuple[dict[str, str], dict[str, str]]:
    mark = {
        "key": key, "text_hash": retag.text_hash(text), "sentiment": label, "slice": slice_,
        "ticker": ticker, "company": "Acme Corp" if ticker else "[sector]", "rule_scope": scope,
        "body_hash": f"hash-{key}",
    }  # fmt: skip
    return mark, {"key": key, "text": text}


def test_the_marked_summaries_split_by_slice_and_the_rows_without_a_company_are_counted() -> None:
    pairs = [
        _mark("1#summary", "Acme rose.", "positive"),
        _mark("2#summary", "Acme fell.", "Negative", slice_="validation"),
        _mark("3#summary", "Sector news.", "neutral", ticker="", scope="sector"),
        _mark("4#summary", "Market news.", "neutral", ticker="", scope="market"),
    ]
    loaded = spt.load_marked_summaries([m for m, _ in pairs], [i for _, i in pairs])
    assert [e.key for e in loaded.train] == ["1#summary"] and loaded.train[0].label == 0
    assert [e.key for e in loaded.validation] == ["2#summary"] and loaded.validation[0].label == 1
    assert loaded.left_out == {"no company (sector)": 1, "no company (market)": 1}
    assert len(loaded.body_hashes) == 4  # the left-out rows are still bodies the replay must avoid
    assert loaded.train[0].company == "Acme Corp" and loaded.train[0].kind == "summary"


def test_a_text_that_is_not_the_one_marked_or_a_bad_mark_stops_the_loading() -> None:
    mark, ids = _mark("1#summary", "Acme rose.", "positive")
    with pytest.raises(spt.MarkedFileError, match="not the text the owner marked"):
        spt.load_marked_summaries([mark], [{**ids, "text": "Acme fell."}])
    with pytest.raises(spt.MarkedFileError, match="not in the ids file"):
        spt.load_marked_summaries([mark], [])
    with pytest.raises(spt.MarkedFileError, match="is not a class"):
        spt.load_marked_summaries([{**mark, "sentiment": "bullish"}], [ids])
    with pytest.raises(spt.MarkedFileError, match="not train or validation"):
        spt.load_marked_summaries([{**mark, "slice": "test"}], [ids])


# --- the replay ---------------------------------------------------------------------------------


def _sentences(n: int) -> list[dict[str, Any]]:
    labels = ("positive", "negative", "neutral")
    return [
        {"sentence": f"Acme sentence {i}", "label": labels[i % 3], "article_id": i}
        for i in range(n)
    ]


def _info(article_id: int) -> spt.ArticleInfo | None:
    if article_id == 0:
        return None  # an article the SOURCE does not have
    return spt.ArticleInfo(f"body{article_id}", "AAA", "Acme Corp")


def _mentions(text: str, company: str, ticker: str) -> bool:
    return "Acme" in text


def test_the_replay_is_seeded_balanced_by_class_and_reports_each_filter() -> None:
    sentences = _sentences(60)
    sentences[1]["sentence"] = "Nothing named here"  # does not name its company
    never = {"body2"}  # a gold, held-out or drawn-for-tagging body
    chosen, report = spt.replay_examples(sentences, _info, never, _mentions, 30, seed=1)
    assert (report.pool, report.unknown_article, report.exam_bodies, report.no_company_named) == (
        60,
        1,
        1,
        1,
    )
    assert report.eligible == 57 and report.drawn == 30 == len(chosen)
    assert report.by_label == {"positive": 10, "negative": 10, "neutral": 10}
    assert all(e.kind == "sentence" and e.company == "Acme Corp" for e in chosen)
    assert not {e.key.split("#")[0] for e in chosen} & {"0", "1", "2"}
    assert "57 eligible, 30 drawn" in report.lines()[0]


def test_the_replay_does_not_depend_on_the_order_of_the_file_and_changes_with_the_seed() -> None:
    sentences = _sentences(90)
    a, _ = spt.replay_examples(sentences, _info, set(), _mentions, 30, seed=1)
    b, _ = spt.replay_examples(sentences[::-1], _info, set(), _mentions, 30, seed=1)
    c, _ = spt.replay_examples(sentences, _info, set(), _mentions, 30, seed=2)
    assert {e.key for e in a} == {e.key for e in b} and {e.key for e in a} != {e.key for e in c}


def test_a_class_short_of_its_quota_is_filled_from_the_others() -> None:
    sentences = [s for s in _sentences(90) if s["label"] != "negative" or s["article_id"] < 7]
    _, report = spt.replay_examples(sentences, _info, set(), _mentions, 30, seed=1)
    assert report.drawn == 30 and report.by_label["negative"] <= 3
    assert any("negative has" in w and "less balanced" in w for w in report.warnings)


def test_a_replay_smaller_than_asked_says_so_and_an_unknown_label_is_named() -> None:
    _, report = spt.replay_examples(_sentences(12), _info, set(), _mentions, 30, seed=1)
    assert report.drawn == 11 and any("only 11 of the 30" in w for w in report.warnings)
    assert any(line.startswith("WARNING: only 11") for line in report.lines())
    bad = [{"sentence": "Acme x", "label": "bullish", "article_id": 5}]
    with pytest.raises(spt.MarkedFileError, match=r"article 5.*bullish"):
        spt.replay_examples(bad, _info, set(), _mentions, 3, seed=1)


def test_a_sentence_under_the_headline_of_an_exam_article_is_left_out() -> None:
    def titled(article_id: int) -> spt.ArticleInfo | None:
        info = _info(article_id)
        return info and info._replace(title_key="same story" if article_id < 10 else "")

    chosen, report = spt.replay_examples(
        _sentences(30), titled, set(), _mentions, 12, seed=1, never_titles={"same story"}
    )
    assert report.exam_titles == 9 and "9 under the headline" in report.lines()[0]
    assert all(int(e.key.split("#")[0]) >= 10 for e in chosen)
    # an empty headline key (too short to mean anything) never matches
    _, none = spt.replay_examples(
        _sentences(30), titled, set(), _mentions, 12, seed=1, never_titles={""}
    )
    assert none.exam_titles == 0


def test_the_fingerprint_follows_the_data_the_base_and_the_constants_not_the_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [_example(i) for i in range(5)]
    val = [_example(i + 10) for i in range(3)]
    base = spt.train_fingerprint(train, val, "m", "r1")
    assert base == spt.train_fingerprint(train[::-1], val[::-1], "m", "r1")
    assert base != spt.train_fingerprint(train[:-1], val, "m", "r1")
    assert base != spt.train_fingerprint(train, val[:-1], "m", "r1")
    assert base != spt.train_fingerprint(train, val, "m", "r2")
    monkeypatch.setattr(spt, "BATCH_SIZE", 8)
    assert base != spt.train_fingerprint(train, val, "m", "r1")


# --- the batches ---------------------------------------------------------------------------------


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 -- a seeded shuffle in a test


def test_a_batch_is_of_one_kind_and_every_example_is_in_one_batch_of_the_epoch() -> None:
    examples = [_example(i) for i in range(10)] + [_example(i, "sentence") for i in range(23)]
    batches = spt.make_batches(examples, 4, _rng(1))
    assert all(len({e.kind for e in b}) == 1 and len(b) <= 4 for b in batches)
    assert sorted(e.key for b in batches for e in b) == sorted(e.key for e in examples)
    assert len(batches) == 3 + 6
    assert [[e.key for e in b] for b in batches] != [
        [e.key for e in b] for b in spt.make_batches(examples, 4, _rng(2))
    ]
    assert [[e.key for e in b] for b in batches] == [
        [e.key for e in b] for b in spt.make_batches(examples, 4, _rng(1))
    ]


def test_a_batch_is_the_pair_company_then_text_and_only_the_text_is_cut(tmp_path: Path) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(_tiny_model(tmp_path / "m"))
    long_text = " ".join(["up"] * 100)
    batch = [
        spt.Example("k", "Acme", long_text, 0, "summary"),
        spt.Example("k2", "Bravo", "the stock was up", 1, "summary"),
    ]
    out = spt.encode_batch(tokenizer, batch, max_length=20)
    assert out["input_ids"].shape == (2, 20)
    first = tokenizer.convert_ids_to_tokens(out["input_ids"][0])
    assert first[:3] == ["[CLS]", "acme", "[SEP]"] and first[-1] == "[SEP]"  # the company is whole
    second = tokenizer.convert_ids_to_tokens(out["input_ids"][1])
    assert second[:3] == ["[CLS]", "bravo", "[SEP]"] and "[PAD]" in second  # padded to the longest


# --- the grid and the tie rule -------------------------------------------------------------------


def test_the_grid_is_27_runs_and_the_final_seed_is_not_one_of_them() -> None:
    runs = spt.grid_settings()
    assert len(runs) == 27 and len({r.name for r in runs}) == 27
    assert spt.FINAL_SEED not in spt.GRID_SEEDS
    assert runs[0] == spt.RunSettings(2, 1e-5, 1) and runs[-1] == spt.RunSettings(4, 3e-5, 3)


def _agreements(
    overrides: dict[tuple[int, float], list[float]] | None = None,
) -> dict[tuple[int, float], list[float]]:
    base = {(e, lr): [0.80, 0.80, 0.80] for e in spt.GRID_EPOCHS for lr in spt.GRID_LEARNING_RATES}
    return {**base, **(overrides or {})}


def test_a_setting_that_beats_the_earlier_one_by_more_than_a_standard_error_is_chosen() -> None:
    choice = spt.choose_settings(
        _agreements({(3, 3e-5): [0.90, 0.88, 0.92], (4, 2e-5): [0.80, 0.79, 0.81]})
    )
    assert (choice.epochs, choice.learning_rate) == (3, 3e-5) and "beats" in choice.reason


def test_a_lead_within_the_noise_keeps_the_earlier_setting() -> None:
    choice = spt.choose_settings(
        _agreements({(3, 3e-5): [0.85, 0.75, 0.83], (4, 2e-5): [0.80, 0.82, 0.78]})
    )
    assert (choice.epochs, choice.learning_rate) == spt.EARLIER_SETTING
    assert "the earlier setting stays" in choice.reason


def test_the_earlier_setting_wins_a_tie_and_the_highest_mean_is_taken_when_it_is_the_earlier_one() -> (
    None
):
    assert (
        spt.choose_settings(_agreements()).epochs,
        spt.choose_settings(_agreements()).learning_rate,
    ) == spt.EARLIER_SETTING
    only_two = {(2, 1e-5): [0.7, 0.7, 0.7], (3, 1e-5): [0.9, 0.9, 0.9]}
    assert spt.choose_settings(only_two)[:2] == (
        3,
        1e-5,
    )  # no earlier setting in the grid: the best


# --- one run -------------------------------------------------------------------------------------


def test_the_decay_is_not_applied_to_biases_and_layer_norms(tmp_path: Path) -> None:
    model = BertForSequenceClassification.from_pretrained(_tiny_model(tmp_path / "m"))
    groups = spt.decay_groups(model, 0.01)
    decayed, plain = groups
    assert decayed["weight_decay"] == 0.01 and plain["weight_decay"] == 0.0
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(
        "bias" not in names[id(p)] and "LayerNorm" not in names[id(p)] for p in decayed["params"]
    )
    assert any("LayerNorm.weight" in names[id(p)] for p in plain["params"])


def test_a_run_learns_the_marks_ends_on_the_last_epoch_and_is_seeded(tmp_path: Path) -> None:
    folder = _tiny_model(tmp_path / "m")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(folder)
    train = [_example(i) for i in range(24)] + [_example(i, "sentence") for i in range(24)]
    settings = spt.RunSettings(epochs=60, learning_rate=1e-2, seed=7)
    logs: list[str] = []

    def run() -> tuple[list[float], Any]:
        model = BertForSequenceClassification.from_pretrained(folder)
        losses = spt.train_run(
            model,
            tokenizer,
            train,
            settings,
            torch.device("cpu"),
            batch_size=8,
            fp16=False,
            log=logs.append,
        )
        return losses, model

    losses, model = run()
    assert len(losses) == 60 and losses[-1] < losses[0] * 0.5  # it learns
    assert len(logs) == 60 and not model.training
    again, _ = run()
    assert again == losses  # the same seed, the same run
    inputs = spt.encode_batch(tokenizer, train)
    predicted = model(**inputs).logits.argmax(-1).tolist()
    assert predicted == [e.label for e in train]  # it fits the 48 marks


# --- the FTI class -------------------------------------------------------------------------------


def test_the_trainer_trains_saves_and_scores_the_saved_model_through_the_scorer(
    tmp_path: Path,
) -> None:
    folder = _tiny_model(tmp_path / "base")
    seen: list[str] = []

    def scorer(model_dir: str) -> float:
        seen.append(model_dir)
        return 0.75

    config = spt.SentimentPairTrainConfig(
        base_model=folder, base_revision=None, output_dir=str(tmp_path / "out"),
        settings=spt.RunSettings(2, 1e-3, 1), train=[_example(i) for i in range(16)], fp16=False,
    )  # fmt: skip
    lines: list[str] = []
    artifact = spt.SentimentPairTrainer(scorer, log=lines.append).train(config)
    assert (
        "16 summaries + 0 sentences, pair input" in lines[0]
        and "batch 16, AdamW weight decay 0.01" in lines[0]
    )
    assert "lr 0.001, linear decay no warmup, clip 1.0" in lines[0] and "seed 1" in lines[0]
    assert artifact.output_dir == str(tmp_path / "out") and seen == [artifact.output_dir]
    assert (tmp_path / "out" / "config.json").exists()
    metrics = artifact.metrics or {}
    assert metrics["validation_agreement"] == 0.75 and metrics["n_train"] == 16
    assert metrics["settings"] == {"epochs": 2, "learning_rate": 1e-3, "seed": 1}
    assert len(metrics["epoch_losses"]) == 2 and metrics["train_seconds"] >= 0


def test_a_base_with_other_classes_is_refused(tmp_path: Path) -> None:
    folder = _tiny_model(tmp_path / "base")
    config_path = Path(folder) / "config.json"
    config_path.write_text(
        config_path.read_text().replace('"positive"', '"label_x"'), encoding="utf-8"
    )
    config = spt.SentimentPairTrainConfig(
        base_model=folder, base_revision=None, output_dir=str(tmp_path / "out"),
        settings=spt.RunSettings(1, 1e-3, 1), train=[_example(0)], fp16=False,
    )  # fmt: skip
    with pytest.raises(spt.MarkedFileError, match="the model's classes are"):
        spt.SentimentPairTrainer(lambda _d: 0.0, log=lambda _m: None).train(config)


def test_the_fingerprint_follows_the_grid_and_the_code_too(monkeypatch: pytest.MonkeyPatch) -> None:
    train, val = [_example(i) for i in range(5)], [_example(i + 10) for i in range(3)]
    base = spt.train_fingerprint(train, val, "m", "r1", code="abc")
    assert base == spt.train_fingerprint(train, val, "m", "r1", code="abc")
    assert base != spt.train_fingerprint(train, val, "m", "r1", code="abd")  # an edit to the code
    monkeypatch.setattr(spt, "GRID_LEARNING_RATES", (1e-5, 2e-5))  # an edit to the grid
    assert base != spt.train_fingerprint(train, val, "m", "r1", code="abc")
