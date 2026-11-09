"""Fine-tuning of the sentiment model on the owner's summaries, read as the pair company + text
(Work item 18, T-148c3, PLAN D35 as amended 2026-11-08).

The recipe (D35): start from the pinned model; train on the owner's tagged summaries (the 360 of the
``train`` slice, less the rows that have no company) plus the same number of sentences of the earlier
balanced pool (the *replay*, so the model does not forget the sentence level), every text read as
``[CLS] company [SEP] text [SEP]`` (``sentiment_pair``). The settings of ``train_sentiment.py`` apply
unless said here: per-device batch 16, AdamW with weight decay 0.01 (not on biases and layer norms, as the
library does), linear decay with no warmup, gradient clipping at 1.0, fp16, no class weights, and the model
after the **last** epoch (the epoch count is itself the grid). A batch never mixes summaries with
sentences, and every epoch shuffles the batches of both kinds together.

This module is the part that can be tested without a GPU or the data: the loader of the marks, the
replay and its filters, the batches, one training run, the grid and the tie rule. The command is
``scripts/train_sentiment_pair_2026_11_09.py``; the FTI class is :class:`SentimentPairTrainer`.
The trainer does not score the 180-text gold (it is scored once, as the ``final`` run of the published
revision, T-148c4): it scores the validation slice through the scorer's own reading, which the caller
gives it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

from fti import TrainConfig, TrainedArtifact
from fti import Trainer as FtiTrainer
from news_nlp import retag
from sentiment_pair import company_segment, reads_as_pair

#: The model's class order: the one of ``ProsusAI/finbert`` and of ``train_sentiment.LABEL2ID``.
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

#: The grid of D35 (epochs x learning rate, three seeds each) and the setting of the earlier fine-tune.
GRID_EPOCHS = (2, 3, 4)
GRID_LEARNING_RATES = (1e-5, 2e-5, 3e-5)
GRID_SEEDS = (1, 2, 3)
#: The seed of the final run: fixed here, before any run, and never chosen by looking at a result.
FINAL_SEED = 4
EARLIER_SETTING = (4, 2e-5)

BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 512


class Example(NamedTuple):
    """One text to learn from: the company it is read for, the text, the owner's class."""

    key: str
    company: str
    text: str
    label: int
    kind: str  # "summary" or "sentence"


class MarkedFileError(ValueError):
    """The marks, or the texts they refer to, are not what the training can use."""


# --- the owner's marks ----------------------------------------------------------------------------


@dataclass
class MarkedSummaries:
    """The rows of the training set, by slice, and what was left out and why."""

    train: list[Example] = field(default_factory=list)
    validation: list[Example] = field(default_factory=list)
    body_hashes: set[str] = field(default_factory=set)  # every body of the 450, left out or not
    left_out: dict[str, int] = field(default_factory=dict)  # why -> rows


def load_marked_summaries(
    marks: Iterable[dict[str, str]], ids: Iterable[dict[str, str]]
) -> MarkedSummaries:
    """The marked summaries (``gold/sentiment_train.csv``) with their texts (the ids file the draw wrote),
    each text checked against the hash the owner marked. A row whose text is not read as a pair (no
    ticker, or a company-less scope: the 26 sector, market and economy rows with a placeholder in
    ``company``) is left out and counted: it belongs to the text-only side, which stays the pinned model."""
    by_key = {r["key"]: r for r in ids}
    result = MarkedSummaries()
    left_out: Counter[str] = Counter()
    for row in marks:
        drawn = by_key.get(row["key"])
        if drawn is None:
            raise MarkedFileError(f"{row['key']} is not in the ids file: its text is not here")
        if retag.text_hash(drawn["text"]) != row["text_hash"]:
            raise MarkedFileError(
                f"{row['key']}: the text in the ids file is not the text the owner marked"
            )
        label = LABEL2ID.get(row["sentiment"].strip().lower())
        if label is None:
            raise MarkedFileError(f"{row['key']}: {row['sentiment']!r} is not a class")
        if row["slice"] not in ("train", "validation"):
            raise MarkedFileError(
                f"{row['key']}: the slice {row['slice']!r} is not train or validation"
            )
        result.body_hashes.add(row["body_hash"])
        if not reads_as_pair(row.get("rule_scope", ""), row["ticker"]):
            left_out[f"no company ({row.get('rule_scope') or 'no scope'})"] += 1
            continue
        example = Example(row["key"], row["company"], drawn["text"], label, "summary")
        (result.train if row["slice"] == "train" else result.validation).append(example)
    result.left_out = dict(left_out)
    return result


# --- the replay -----------------------------------------------------------------------------------


class ArticleInfo(NamedTuple):
    """What the replay needs to know of the article a sentence came from."""

    body_hash: str
    ticker: str
    company: str
    title_key: str = ""  # the headline key (empty if too short to mean anything)


@dataclass
class ReplayReport:
    """How many sentences each filter removed, so that the trainer can print it."""

    pool: int = 0
    unknown_article: int = 0
    exam_bodies: int = 0  # the body is a gold, held-out or drawn-for-tagging body
    exam_titles: int = 0  # the same story under another body: its headline is an exam headline
    no_company_named: int = 0  # the sentence does not name its article's company
    eligible: int = 0
    drawn: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(
        default_factory=list
    )  # a replay smaller or less balanced than asked

    def lines(self) -> list[str]:
        return [
            f"replay pool {self.pool:,}: {self.unknown_article:,} of an article not in the SOURCE, "
            f"{self.exam_bodies:,} of a gold, held-out or drawn-for-tagging body, "
            f"{self.exam_titles:,} under the headline of one (same story, another body), "
            f"{self.no_company_named:,} that do not name their article's company; "
            f"{self.eligible:,} eligible, {self.drawn:,} drawn {self.by_label}",
            *(f"WARNING: {w}" for w in self.warnings),
        ]


def replay_examples(
    sentences: Iterable[dict[str, Any]],
    info_of: Callable[[int], ArticleInfo | None],
    never: set[str],
    mentions: Callable[[str, str, str], bool],
    n: int,
    seed: int,
    never_titles: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[Example], ReplayReport]:
    """*n* sentences of the earlier balanced pool, seeded and as balanced by class as the pool allows,
    each read for the company of its article. Left out: a sentence of an article the SOURCE does not
    have, of a body in *never* (the gold, the held-out and the drawn-for-tagging bodies, by hash), under a
    headline in *never_titles* (the same story stored under another body), or that does not name its article's company (*mentions*: ``(text, company, ticker)``, the rule the
    stage uses to weigh a chunk), since a pair needs a company the sentence is about."""
    report = ReplayReport()
    eligible: dict[int, list[Example]] = defaultdict(list)
    for row in sentences:
        report.pool += 1
        info = info_of(int(row["article_id"]))
        if info is None:
            report.unknown_article += 1
        elif info.body_hash in never:
            report.exam_bodies += 1
        elif info.title_key and info.title_key in never_titles:
            report.exam_titles += 1
        elif not mentions(row["sentence"], info.company, info.ticker):
            report.no_company_named += 1
        else:
            label = LABEL2ID.get(row["label"])
            if label is None:
                raise MarkedFileError(
                    f"replay sentence of article {row['article_id']}: {row['label']!r} is not a class"
                )
            key = f"{row['article_id']}#{hashlib.sha256(row['sentence'].encode()).hexdigest()[:10]}"
            eligible[label].append(Example(key, info.company, row["sentence"], label, "sentence"))
    report.eligible = sum(len(v) for v in eligible.values())
    rng = random.Random(seed)  # noqa: S311 -- a seeded draw, not a secret
    for label in sorted(eligible):  # in a fixed order: the draw does not depend on the file's
        eligible[label].sort(key=lambda e: e.key)
        rng.shuffle(eligible[label])
    chosen: list[Example] = []
    quota = n // len(LABEL2ID)
    for label in sorted(eligible):
        chosen += eligible[label][:quota]
    leftovers = [e for label in sorted(eligible) for e in eligible[label][quota:]]
    rng.shuffle(leftovers)
    chosen += leftovers[: n - len(chosen)]
    report.drawn = len(chosen)
    report.warnings = _shortfalls(report.drawn, n, {k: len(v) for k, v in eligible.items()}, quota)
    report.by_label = {ID2LABEL[k]: v for k, v in sorted(Counter(e.label for e in chosen).items())}
    return chosen, report


def _shortfalls(drawn: int, n: int, eligible: dict[int, int], quota: int) -> list[str]:
    """What is short in a replay: fewer sentences than asked, or a class below its quota."""
    warnings = []
    if drawn < n:
        warnings.append(f"only {drawn} of the {n} asked: the eligible pool is smaller")
    for label in LABEL2ID.values():
        if eligible.get(label, 0) < quota:
            warnings.append(
                f"{ID2LABEL[label]} has {eligible.get(label, 0)} eligible sentences, below its quota "
                f"of {quota}: the replay is less balanced than asked"
            )
    return warnings


def train_fingerprint(
    train: Sequence[Example],
    validation: Sequence[Example],
    base_model: str,
    base_revision: str | None,
    code: str = "",
) -> str:
    """A hash of everything the grid's rows depend on: the training texts (key and class), the validation
    texts, the base model and revision, the recipe's constants, the grid itself and *code* (the revision of
    the modules that make a number: ``retag.code_revision`` over them, given by the command). Rows made
    under another fingerprint are not rows of the same grid."""
    payload = {
        "train": sorted((e.key, e.label, e.kind) for e in train),
        "validation": sorted((e.key, e.label) for e in validation),
        "base": [base_model, base_revision],
        "constants": [BATCH_SIZE, WEIGHT_DECAY, MAX_GRAD_NORM, MAX_LENGTH, GRID_SEEDS, FINAL_SEED],
        "grid": [GRID_EPOCHS, GRID_LEARNING_RATES, EARLIER_SETTING],
        "code": code,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --- batches --------------------------------------------------------------------------------------


def make_batches(
    examples: Sequence[Example], batch_size: int, rng: random.Random
) -> list[list[Example]]:
    """The batches of one epoch: each is of one kind (summaries or sentences, never both), and the
    batches of both kinds are shuffled together."""
    by_kind: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_kind[example.kind].append(example)
    batches: list[list[Example]] = []
    for kind in sorted(by_kind):
        rows = by_kind[kind][:]
        rng.shuffle(rows)
        batches += [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    rng.shuffle(batches)
    return batches


def encode_batch(tokenizer: Any, batch: Sequence[Example], max_length: int = MAX_LENGTH) -> Any:
    """The tokenizer's tensors for a batch of pairs ``[CLS] company [SEP] text [SEP]``, padded to the
    longest of the batch; only the text is cut (``sentiment_pair`` writes the same rule for one pair)."""
    return tokenizer(
        [company_segment(e.company) for e in batch],
        [e.text for e in batch],
        truncation="only_second",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


# --- one run --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSettings:
    """What changes between the runs of the grid."""

    epochs: int
    learning_rate: float
    seed: int

    @property
    def name(self) -> str:
        return f"e{self.epochs}_lr{self.learning_rate:g}_s{self.seed}"


def decay_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """The parameters with weight decay and without: biases and layer norms have none, as in the library's
    own trainer."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    named = list(model.named_parameters())
    return [
        {
            "params": [p for n, p in named if not any(k in n for k in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]


def train_run(
    model: Any,
    tokenizer: Any,
    train: Sequence[Example],
    settings: RunSettings,
    device: Any,
    *,
    batch_size: int = BATCH_SIZE,
    fp16: bool = True,
    log: Callable[[str], None] = print,
) -> list[float]:
    """Train *model* in place for ``settings.epochs`` epochs and return the mean loss of each. The model
    after the last epoch is the result (no best-epoch selection). Seeded: the shuffles and the dropout."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    rng = random.Random(settings.seed)  # noqa: S311 -- shuffling, not a secret
    steps = settings.epochs * _kind_batches(train, batch_size)
    optimizer = torch.optim.AdamW(decay_groups(model, WEIGHT_DECAY), lr=settings.learning_rate)
    schedule = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer, num_warmup_steps=0, num_training_steps=steps
    )
    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device).train()
    losses: list[float] = []
    for epoch in range(settings.epochs):
        total, count = 0.0, 0
        for batch in make_batches(train, batch_size, rng):
            inputs = encode_batch(tokenizer, batch).to(device)
            labels = torch.tensor([e.label for e in batch], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = torch.nn.functional.cross_entropy(model(**inputs).logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        losses.append(total / max(count, 1))
        log(f"  epoch {epoch + 1}/{settings.epochs}: loss {losses[-1]:.4f}")
    model.eval()
    return losses


def _kind_batches(examples: Sequence[Example], batch_size: int) -> int:
    """The number of batches of one epoch (each kind is batched on its own, so the last batch of a kind
    can be short)."""
    counts = Counter(e.kind for e in examples)
    return sum(math.ceil(n / batch_size) for n in counts.values())


# --- the grid and the tie rule --------------------------------------------------------------------


def grid_settings() -> list[RunSettings]:
    """The 27 runs of the grid, in a fixed order."""
    return [
        RunSettings(epochs, lr, seed)
        for epochs, lr, seed in product(GRID_EPOCHS, GRID_LEARNING_RATES, GRID_SEEDS)
    ]


class Choice(NamedTuple):
    epochs: int
    learning_rate: float
    mean: float
    reason: str


def standard_error(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def choose_settings(agreements: dict[tuple[int, float], list[float]]) -> Choice:
    """The setting with the highest mean validation agreement over its seeds, **unless it does not beat the
    earlier fine-tune's setting (4 epochs, 2e-5) by more than one standard error of the mean** (the larger
    of the two settings' standard errors): nine settings compared on 90 texts would otherwise pick a winner
    that is partly noise, so the earlier setting stays. Ties go to the earlier setting, then to the fewer
    epochs, then to the lower rate."""
    means = {k: statistics.fmean(v) for k, v in agreements.items()}
    best = max(means, key=lambda k: (means[k], k == EARLIER_SETTING, -k[0], -k[1]))
    if EARLIER_SETTING not in agreements or best == EARLIER_SETTING:
        return Choice(*best, means[best], "highest mean validation agreement")
    margin = means[best] - means[EARLIER_SETTING]
    se = max(standard_error(agreements[best]), standard_error(agreements[EARLIER_SETTING]))
    if margin > se:
        return Choice(
            *best, means[best], f"beats the earlier setting by {margin:.4f} > 1 SE ({se:.4f})"
        )
    return Choice(
        *EARLIER_SETTING,
        means[EARLIER_SETTING],
        f"the best ({best[0]} epochs, {best[1]:g}) leads by {margin:.4f}, not more than 1 SE ({se:.4f}): "
        "the earlier setting stays",
    )


# --- the FTI class --------------------------------------------------------------------------------


@dataclass(frozen=True)
class SentimentPairTrainConfig(TrainConfig):
    """One run of the recipe: the base model (a Hub id and revision, or a local directory), the data, the
    settings and where the model is saved."""

    base_model: str
    base_revision: str | None
    output_dir: str
    settings: RunSettings
    train: Sequence[Example] = ()
    fp16: bool = True


class SentimentPairTrainer(FtiTrainer[SentimentPairTrainConfig]):
    """One run on the FTI base class (``fti.py``): load the base, train, save, and score the validation slice
    through *scorer*, the scorer's own reading of a saved model (a function of the model directory that gives
    the agreement with the owner's marks). It logs nothing: the grid logs to MLflow."""

    def __init__(self, scorer: Callable[[str], float], log: Callable[[str], None] = print) -> None:
        self._scorer = scorer
        self._log = log

    def train(self, config: SentimentPairTrainConfig) -> TrainedArtifact:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model, revision=config.base_revision
        )
        require_class_order(model)  # the base's own head: this is a continued fine-tune
        kinds = Counter(e.kind for e in config.train)
        self._log(
            f"{config.settings.name}: {config.base_model}@{config.base_revision}; "
            f"{kinds['summary']} summaries + {kinds['sentence']} sentences, pair input, max length {MAX_LENGTH}; "
            f"batch {BATCH_SIZE}, AdamW weight decay {WEIGHT_DECAY} (not on biases and layer norms), "
            f"lr {config.settings.learning_rate:g}, linear decay no warmup, clip {MAX_GRAD_NORM}, "
            f"fp16 {config.fp16 and device.type == 'cuda'}, no class weights, {config.settings.epochs} epochs, "
            f"seed {config.settings.seed}, the model of the last epoch"
        )
        started = time.time()
        losses = train_run(
            model, tokenizer, config.train, config.settings, device, fp16=config.fp16, log=self._log
        )
        seconds = time.time() - started
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        agreement = self._scorer(config.output_dir)
        return TrainedArtifact(
            output_dir=config.output_dir,
            metrics={
                "validation_agreement": agreement,
                "epoch_losses": losses,
                "train_seconds": seconds,
                "settings": {
                    "epochs": config.settings.epochs,
                    "learning_rate": config.settings.learning_rate,
                    "seed": config.settings.seed,
                },
                "n_train": len(config.train),
            },
        )


def require_class_order(model: Any) -> None:
    """Stop if the head's classes are not positive, negative, neutral in that order (the pinned model's)."""
    got = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if got != ID2LABEL:
        raise MarkedFileError(f"the model's classes are {got}, not {ID2LABEL}")
