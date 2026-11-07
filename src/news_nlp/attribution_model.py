"""The trained attribution model of step 4 (Work item 18, T-148b3, D27): who a sentence of a
narrative multi-company body is about.

The model reads a pair, *is this sentence about this company?*: the company's name goes in the first
segment; the second holds the previous sentence (the context) and the sentence. It is trained on the
paragraphs that lead with a company (``lead`` rows only, D27's label check): every sentence of such a
paragraph is a positive for the company, with that company's name **masked** in the sentence so the
model cannot learn "the name is there", and the same sentence paired with another company of the
body is a negative (for a paragraph's first sentence, the company of the paragraph before it, the
hard case). So it learns who a sentence is about from its content and its context.

At run time a sentence that names universe companies belongs to them (as the baseline's rule); a
sentence that names none is scored against every company the body names and goes to the best one if
the score reaches ``THRESHOLD``. A sentence no company reaches belongs to nobody: nothing is given
by default. The output has the baseline's shape, so the faithfulness check reads both alike.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

from chunking import split_sentences
from news_nlp.company_texts import SegmentLabel, segments
from news_nlp.triage import CompanyResolver

MASK = "[COMPANY]"
MAX_LENGTH = 160  # a sentence and the one before it: well under 160 word pieces
THRESHOLD = 0.5  # the best company's score a sentence needs to be given to it
NEGATIVES = 1  # wrong companies paired with each sentence in training


class Pair(NamedTuple):
    """One input of the model: ``candidate`` is the company asked about."""

    candidate: str
    context: str
    sentence: str


class Example(NamedTuple):
    pair: Pair
    label: int  # 1: the sentence is about the candidate
    key: str  # ``<article>#<paragraph>#<sentence>``: which sentence, so a positive and its negatives group
    first: bool  # the first sentence of its paragraph (the company changes there)
    year: int


class Scorer(Protocol):
    def score(self, pairs: Sequence[Pair]) -> list[float]: ...


def mask_company(text: str, resolver: CompanyResolver, ticker: str) -> str:
    """*text* with every mention of the company *ticker* replaced by :data:`MASK`."""
    out: list[str] = []
    last = 0
    for start, end, mentioned in resolver.mention_spans(text):
        if mentioned == ticker:
            out += [text[last:start], MASK]
            last = end
    out.append(text[last:])
    return "".join(out)


def sentences_of(paragraph: str) -> list[str]:
    return [s.strip() for s, _, _ in split_sentences(paragraph) if s.strip()]


def training_examples(
    body: str,
    labels: Sequence[SegmentLabel],
    *,
    article_id: int,
    year: int,
    resolver: CompanyResolver,
    names: Mapping[str, str],
    rng: random.Random,
) -> list[Example]:
    """The positives and negatives of the ``lead`` paragraphs of one body. *labels* are all the
    labelled paragraphs of the body (both kinds): the ``lead`` ones are trained on, the others only
    say which company the paragraph before a lead belongs to and which companies the body has."""
    parts = segments(body)
    by_position = {label.position: label for label in labels}
    tickers = sorted({label.ticker for label in labels if label.ticker in names})
    out: list[Example] = []
    for label in labels:
        if label.kind != "lead" or label.ticker not in names or label.position >= len(parts):
            continue
        sentences = sentences_of(parts[label.position])
        before = by_position.get(label.position - 1)
        previous = sentences_of(parts[label.position - 1])[-1:] if label.position > 0 else []
        for index, sentence in enumerate(sentences):
            context = sentences[index - 1] if index else (previous or [""])[0]
            masked = mask_company(sentence, resolver, label.ticker)
            key = f"{article_id}#{label.position}#{index}"
            first = index == 0
            out.append(Example(Pair(names[label.ticker], context, masked), 1, key, first, year))
            wrong = [t for t in tickers if t != label.ticker]
            hard = (
                [before.ticker] if first and before is not None and before.ticker in wrong else []
            )
            rest = [t for t in wrong if t not in hard]
            rng.shuffle(rest)
            for ticker in (hard + rest)[:NEGATIVES]:
                out.append(Example(Pair(names[ticker], context, masked), 0, key, first, year))
    return out


def named_in_context(pair: Pair) -> float:
    """The mention-and-recency baseline as a score on a pair: 1 when the candidate's name is in the
    previous sentence (recency), else 0. What the model has to beat on the same pairs."""
    return float(bool(pair.candidate) and pair.candidate.lower() in pair.context.lower())


def pair_accuracy(examples: Sequence[Example], scores: Sequence[float]) -> dict[str, float]:
    """On each sentence, the share of (positive, negative) pairs the scores rank right (a tie
    counts half) and the accuracy of the 0.5 threshold. Positive and negatives of a sentence share
    its ``key``."""
    by_key: dict[str, list[tuple[int, float]]] = {}
    for example, score in zip(examples, scores, strict=True):
        by_key.setdefault(example.key, []).append((example.label, score))
    right = pairs = 0.0
    for items in by_key.values():
        positives = [s for label, s in items if label == 1]
        negatives = [s for label, s in items if label == 0]
        for p in positives:
            for n in negatives:
                pairs += 1
                right += 1.0 if p > n else 0.5 if p == n else 0.0
    threshold = sum(
        (s >= THRESHOLD) == bool(e.label) for e, s in zip(examples, scores, strict=True)
    )
    return {
        "pairs": pairs,
        "ranking": right / pairs if pairs else float("nan"),
        "threshold": threshold / len(examples) if examples else float("nan"),
    }


def attribute_sentences(
    body: str,
    resolver: CompanyResolver,
    names: Mapping[str, str],
    scorer: Scorer,
    threshold: float = THRESHOLD,
) -> list[tuple[str, tuple[str, ...]]]:
    """``(sentence, tickers)`` for each sentence of *body*; ``()`` when no company is named in it
    and none reaches *threshold*."""
    sentences = [s for s, _, _ in split_sentences(body)]
    candidates = sorted(t for t in resolver.resolve(body, skip_attribution=False) if t in names)
    named = [tuple(sorted(resolver.resolve(s, skip_attribution=False))) for s in sentences]
    pairs: list[Pair] = []
    where: list[tuple[int, str]] = []
    for index, sentence in enumerate(sentences):
        if named[index]:
            continue
        context = sentences[index - 1] if index else ""
        for ticker in candidates:
            pairs.append(Pair(names[ticker], context, sentence))
            where.append((index, ticker))
    best: dict[int, tuple[float, str]] = {}
    for (index, ticker), score in zip(where, scorer.score(pairs) if pairs else [], strict=True):
        if index not in best or score > best[index][0]:
            best[index] = (score, ticker)
    out: list[tuple[str, tuple[str, ...]]] = []
    for index, sentence in enumerate(sentences):
        if named[index]:
            out.append((sentence, named[index]))
        elif index in best and best[index][0] >= threshold:
            out.append((sentence, (best[index][1],)))
        else:
            out.append((sentence, ()))
    return out


def model_company_texts(
    body: str,
    resolver: CompanyResolver,
    names: Mapping[str, str],
    scorer: Scorer,
    threshold: float = THRESHOLD,
) -> dict[str, str]:
    """``{ticker: text}`` in order of first appearance, as ``baseline_company_texts``: a company the
    body names always gets the sentence that names it; the sentences that name none go to the
    company the model chooses, or to nobody."""
    groups: dict[str, list[str]] = {}
    for sentence, tickers in attribute_sentences(body, resolver, names, scorer, threshold):
        for ticker in tickers:
            groups.setdefault(ticker, []).append(sentence)
    return {ticker: " ".join(sentences) for ticker, sentences in groups.items()}


class TransformerScorer:
    """The trained model behind :class:`Scorer`: the probability that the sentence is about the
    candidate. Loaded from a local directory or a pinned Hub revision."""

    def __init__(self, model: str, revision: str | None = None, device: str | None = None) -> None:
        import torch  # noqa: PLC0415 -- only a real run needs torch
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model: Any = (
            AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
            .to(self.device)
            .eval()
        )

    def encode(self, pairs: Sequence[Pair]) -> Any:
        return self.tokenizer(
            [p.candidate for p in pairs],
            [f"{p.context} [SEP] {p.sentence}" if p.context else p.sentence for p in pairs],
            truncation="only_second",
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

    def score(self, pairs: Sequence[Pair], batch_size: int = 64) -> list[float]:
        out: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = {
                    k: v.to(self.device)
                    for k, v in self.encode(pairs[start : start + batch_size]).items()
                }
                logits = self.model(**batch).logits
                out += self.torch.softmax(logits.float(), dim=-1)[:, 1].tolist()
        return out
