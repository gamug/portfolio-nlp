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

import hashlib
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import chunking
from chunking import split_sentences
from news_nlp import attribution_units, company_texts, triage
from news_nlp.company_texts import SegmentLabel, segments
from news_nlp.triage import CompanyResolver

# The code the training pairs and the run-time texts depend on: one list for the training record
# and for the extractor's revision, so a change in any of them is seen by both.
CODE_MODULES = (sys.modules[__name__], chunking, company_texts, triage)
# The ``units`` reading also depends on how the body is cut and its footer stripped.
UNITS_CODE_MODULES = (*CODE_MODULES, attribution_units)

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
    named: bool = False  # the sentence named a universe company (run time scores the others)


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
            named = bool(resolver.mention_spans(sentence))
            key = f"{article_id}#{label.position}#{index}"
            first = index == 0
            out.append(
                Example(Pair(names[label.ticker], context, masked), 1, key, first, year, named)
            )
            wrong = [t for t in tickers if t != label.ticker]
            hard = (
                [before.ticker] if first and before is not None and before.ticker in wrong else []
            )
            rest = [t for t in wrong if t not in hard]
            rng.shuffle(rest)
            for ticker in (hard + rest)[:NEGATIVES]:
                out.append(
                    Example(Pair(names[ticker], context, masked), 0, key, first, year, named)
                )
    return out


def exact_name_in_context(pair: Pair) -> float:
    """1 when the candidate's name, as given to the model (the legal name), is in the previous
    sentence. Prose rarely writes the legal name, so this is a weak floor, kept to show it."""
    return float(bool(pair.candidate) and pair.candidate.lower() in pair.context.lower())


class Decision(NamedTuple):
    """One sentence as run time sees it: scored against every company of the body."""

    true: str  # the company of the paragraph the sentence is in
    candidates: tuple[str, ...]  # the companies the body names (all of them are asked about)
    pairs: tuple[Pair, ...]  # one per candidate, in the same order
    recent: str  # the company mentioned last in the previous sentence ("" if none)


def decisions(
    body: str,
    labels: Sequence[SegmentLabel],
    *,
    resolver: CompanyResolver,
    names: Mapping[str, str],
) -> list[Decision]:
    """The run-time task on the ``lead`` paragraphs of one body: the sentences that name no
    universe company (the only ones the model decides), each with every company the body names as
    candidate, as :func:`attribute_sentences` builds them."""
    parts = segments(body)
    candidates = tuple(
        sorted(t for t in resolver.resolve(body, skip_attribution=False) if t in names)
    )
    out: list[Decision] = []
    for label in labels:
        if label.kind != "lead" or label.ticker not in candidates or label.position >= len(parts):
            continue
        sentences = sentences_of(parts[label.position])
        previous = sentences_of(parts[label.position - 1])[-1:] if label.position > 0 else []
        for index, sentence in enumerate(sentences):
            if resolver.mention_spans(sentence):
                continue
            context = sentences[index - 1] if index else (previous or [""])[0]
            seen = [t for _, _, t in resolver.mention_spans(context)]
            pairs = tuple(Pair(names[t], context, sentence) for t in candidates)
            out.append(Decision(label.ticker, candidates, pairs, seen[-1] if seen else ""))
    return out


def decision_report(
    items: Sequence[Decision], scores: Sequence[Sequence[float]], threshold: float = THRESHOLD
) -> dict[str, float]:
    """On the sentences of *items*, what run time does: the share the model gives to the right
    company, to a wrong one and to nobody (no score reaches *threshold*, or two companies tie for
    the best), and the baseline's (the
    company mentioned last in the previous sentence, else nobody)."""
    right = wrong = 0
    base_right = base_wrong = 0
    for item, row in zip(items, scores, strict=True):
        best = max(range(len(row)), key=row.__getitem__)
        tied = row.count(row[best]) > 1  # no best company: nobody, not the first in alphabet
        if row[best] >= threshold and not tied:
            if item.candidates[best] == item.true:
                right += 1
            else:
                wrong += 1
        if item.recent in item.candidates:
            if item.recent == item.true:
                base_right += 1
            else:
                base_wrong += 1
    n = len(items) or float("nan")
    return {
        "sentences": float(len(items)),
        "model right": right / n,
        "model wrong": wrong / n,
        "model nobody": (len(items) - right - wrong) / n,
        "baseline right": base_right / n,
        "baseline wrong": base_wrong / n,
        "baseline nobody": (len(items) - base_right - base_wrong) / n,
    }


def directory_identity(directory: Path) -> str:
    """A hash of every top-level file of a model directory (weights, config, tokenizer), by name,
    except ``metrics.json``, which the trainer rewrites: what identifies a local model. A file of
    any kind counts (an editor backup changes the identity: the safe side); subdirectories
    (checkpoints) are not read. A directory without weights or ``config.json`` is not a model."""
    for needed in ("model.safetensors", "config.json"):
        if not (directory / needed).is_file():
            raise ValueError(f"{directory} has no {needed}: it is not a trained model directory")
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.iterdir() if p.is_file() and p.name != "metrics.json"):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


def named_in_context(pair: Pair, resolver: CompanyResolver) -> float:
    """The mention-and-recency baseline as a score on a pair: 1 when the previous sentence mentions
    the candidate the way the resolver reads mentions (short names, tickers), else 0. What the model
    has to beat on the same pairs."""
    wanted = set(resolver.resolve(pair.candidate, skip_attribution=False))
    return float(bool(wanted & {t for _, _, t in resolver.mention_spans(pair.context)}))


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


READINGS = ("sentences", "units", "units2", "units3")
Reading = Literal["sentences", "units", "units2", "units3"]
Resolving = CompanyResolver | attribution_units.GuardedResolver


def _pieces(
    body: str, resolver: Resolving, reading: str
) -> tuple[Resolving, list[str], list[tuple[str, ...] | None], str]:
    """The resolver to use, the pieces of *body*, the owners the lines give them (``None``: the
    attribution decides) and the text whose companies are the candidates, for the *reading* of
    :func:`scored_sentences`. A reading that is not one of :data:`READINGS` is an error: a typo
    would otherwise run another reading and return plausible texts."""
    if reading not in READINGS:
        raise ValueError(f"reading {reading!r}: one of {READINGS}")
    if reading == "sentences":
        sentences = [s for s, _, _ in split_sentences(body)]
        return resolver, sentences, [None] * len(sentences), body
    story = attribution_units.strip_footer(body)
    guarded = reading in ("units2", "units3")
    if guarded and isinstance(resolver, CompanyResolver):
        resolver = attribution_units.GuardedResolver(resolver, story)
    pieces = attribution_units.attribution_pieces(
        body, resolver, headers=guarded, named_after=reading == "units3"
    )
    return resolver, [p.text for p in pieces], [p.owners for p in pieces], story


def scored_sentences(
    body: str,
    resolver: Resolving,
    names: Mapping[str, str],
    scorer: Scorer,
    threshold: float = THRESHOLD,
    *,
    reading: Reading = "sentences",
) -> list[tuple[str, tuple[str, ...], float | None]]:
    """``(sentence, tickers, score)`` for each piece of *body*: the companies it belongs to (``()``
    when none is named in it and none reaches *threshold*) and, for a piece that names none and
    that no header gives to anybody, the best score any candidate got (``None`` when the piece
    names a company, a header owns it, or the body has no candidate). *reading* is one of
    :data:`READINGS`: ``sentences`` (the splitter's sentences), ``units`` (the pieces of
    :func:`attribution_units.attribution_pieces`: footer removed; lines, bullets and clauses cut),
    ``units2`` (``units`` with the guarded resolver, which does not take the first word of a name
    for a company, and with the lines headed by a list of names given to those names) or
    ``units3`` (``units2`` where such a line also belongs to the universe companies its text
    names). In the ``units`` readings the candidates are the companies of the body without its
    footer, found by the same resolver as the pieces: a company named only in the footer, or (from
    ``units2``) only by the first word of its name, is not asked about."""
    resolver, sentences, owners, story = _pieces(body, resolver, reading)
    candidates = sorted(t for t in resolver.resolve(story, skip_attribution=False) if t in names)
    named: list[tuple[str, ...]] = [
        mine if mine is not None else tuple(sorted(resolver.resolve(s, skip_attribution=False)))
        for s, mine in zip(sentences, owners, strict=True)
    ]
    pairs: list[Pair] = []
    where: list[tuple[int, str]] = []
    for index, sentence in enumerate(sentences):
        if named[index] or owners[index] is not None:
            continue
        context = sentences[index - 1] if index else ""
        for ticker in candidates:
            pairs.append(Pair(names[ticker], context, sentence))
            where.append((index, ticker))
    best: dict[int, tuple[float, str]] = {}
    for (index, ticker), score in zip(where, scorer.score(pairs) if pairs else [], strict=True):
        if index not in best or score > best[index][0]:
            best[index] = (score, ticker)
    out: list[tuple[str, tuple[str, ...], float | None]] = []
    for index, sentence in enumerate(sentences):
        if named[index]:
            out.append((sentence, named[index], None))
        elif index in best and best[index][0] >= threshold:
            out.append((sentence, (best[index][1],), best[index][0]))
        else:
            out.append((sentence, (), best[index][0] if index in best else None))
    return out


def attribute_sentences(
    body: str,
    resolver: Resolving,
    names: Mapping[str, str],
    scorer: Scorer,
    threshold: float = THRESHOLD,
    *,
    reading: Reading = "sentences",
) -> list[tuple[str, tuple[str, ...]]]:
    """``(sentence, tickers)`` for each sentence of *body*; ``()`` when no company is named in it
    and none reaches *threshold*. *reading*: see :func:`scored_sentences`."""
    scored = scored_sentences(body, resolver, names, scorer, threshold, reading=reading)
    return [(s, t) for s, t, _ in scored]


def model_company_texts(
    body: str,
    resolver: Resolving,
    names: Mapping[str, str],
    scorer: Scorer,
    threshold: float = THRESHOLD,
    *,
    reading: Reading = "sentences",
) -> dict[str, str]:
    """``{ticker: text}`` in order of first appearance, as ``baseline_company_texts``: a company the
    body names always gets the sentence that names it; the sentences that name none go to the
    company the model chooses, or to nobody."""
    groups: dict[str, list[str]] = {}
    for sentence, tickers in attribute_sentences(
        body, resolver, names, scorer, threshold, reading=reading
    ):
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
