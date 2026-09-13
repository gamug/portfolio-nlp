#!/usr/bin/env python
"""Follow-up to scripts/label_sentiment_sentences_2026_09_13.py: targeted
mining + labeling for one specific coverage gap found by manual spot-check
of the published gamug/FinBERT-financial-news model -- financial
"performance verb" idioms (crushed/smashed/trounced/clobbered/routed/
walloped/demolished/hammered, etc.) that read as generically violent/
negative in everyday English but split by *what* they're applied to in
financial-news usage:

    "Nvidia stock got crushed"               -> negative (the company/stock
                                                 itself is the object)
    "Meta crushed its earnings estimates"    -> positive (an estimate/
                                                 target/expectation is the
                                                 object)

The original 5,000-sentence random draw (scripts/label_sentiment_sentences_
2026_09_13.py) happened to include almost none of this idiom family --
mining the eval-run article pool alone found only 75 hits across 11,322
articles, essentially by chance excluded from the training sample. This
script widens the search to the *full* source corpus (~480k articles, not
just the ~11k already touched by prior evals) to get enough real examples
to actually teach the model the distinction above, rather than patching it
with a lexicon override (which would get the negative case wrong).

Output (both git-ignored data artifacts, not written to any DB table):

- data/sentiment_finetune/idiom_probe.jsonl -- a small held-out slice,
  reserved for evaluation only (never included in training) so the fix can
  be measured directly, not just inferred from aggregate metrics moving.
- data/sentiment_finetune/idiom_augment.jsonl -- the rest, merged into the
  training set by train_sentiment.py.

Usage:
    uv run python scripts/mine_idiom_sentences_2026_09_13.py \\
        --results-db /path/to/nlp.db --source-db /path/to/urls.db \\
        --target-count 900 --probe-count 100 --seed 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
load_dotenv()

import news_nlp as db  # noqa: E402 -- needs sys.path/load_dotenv above first
from chunking import split_sentences  # noqa: E402

_LABELS = ("positive", "negative", "neutral")
_MIN_SENTENCE_LEN = 20
_MAX_SENTENCE_LEN = 300
_MIN_LETTER_COUNT = 10
_BATCH_SIZE = 500

# The idiom family this round specifically targets, plus a finance-context
# word so we don't also pull in unrelated uses ("crushed the ball", sports
# reporting, etc.).
_VERB_RE = re.compile(
    r"\b(crush(?:ed|ing)?|smash(?:ed|ing)?|trounc(?:ed|ing)?|clobber(?:ed|ing)?|"
    r"rout(?:ed|ing)?|wallop(?:ed|ing)?|demolish(?:ed|ing)?|obliterat(?:ed|ing)?|"
    r"pummel(?:ed|ing|led|ling)?|hammer(?:ed|ing)?)\b",
    re.I,
)
_CONTEXT_RE = re.compile(
    r"(earning|estimate|expectation|forecast|guidance|consensus|profit|revenue|street|"
    r"quarter|stock|share)",
    re.I,
)

_SYSTEM_PROMPT = """You are a financial analyst labeling individual sentences from financial \
news articles for a sentiment classification dataset.

For the sentence given, decide its sentiment from the viewpoint of a long-only equity \
investor: does this sentence, in isolation, read as good news, bad news, or neither for \
the company/asset it concerns?

- positive: the sentence describes a favorable development (beat, upgrade, growth, gain, \
  positive outlook, strong demand). Note: a company or metric "crushing/smashing/beating/ \
  trouncing" an estimate, expectation, forecast, or target is POSITIVE (they exceeded it).
- negative: the sentence describes an unfavorable development (miss, downgrade, decline, \
  loss, weak demand, litigation, warning). Note: a stock, company, or sector "getting \
  crushed/smashed/hammered/routed" (as the object of the verb, not the subject beating a \
  target) is NEGATIVE -- it describes the company/stock suffering, not exceeding a goal.
- neutral: purely factual/administrative with no clear directional charge, or not about \
  company/market performance at all (e.g. "The company is headquartered in Chicago.").

Reply with exactly one word: positive, negative, or neutral. No punctuation, no \
explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=900)
    parser.add_argument("--probe-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--existing-sentences",
        type=Path,
        default=Path("data/sentiment_finetune/labeled_sentences.jsonl"),
        help="dedupe against sentences already used in the original training draw",
    )
    parser.add_argument(
        "--probe-out",
        type=Path,
        default=Path("data/sentiment_finetune/idiom_probe.jsonl"),
    )
    parser.add_argument(
        "--augment-out",
        type=Path,
        default=Path("data/sentiment_finetune/idiom_augment.jsonl"),
    )
    return parser.parse_args()


def _is_plausible_sentence(text: str) -> bool:
    stripped = text.strip()
    if not (_MIN_SENTENCE_LEN <= len(stripped) <= _MAX_SENTENCE_LEN):
        return False
    letters = sum(c.isalpha() for c in stripped)
    return letters >= _MIN_LETTER_COUNT


def _existing_sentence_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open() as f:
        for line in f:
            seen.add(" ".join(json.loads(line)["sentence"].split()))
    return seen


def _collect_idiom_sentences(
    conn: db.NewsNlpDatabase, rng: random.Random, target_count: int, already_used: set[str]
) -> list[tuple[int, str]]:
    """Scan the *full* source corpus (all articles with body_text), not just
    the ~11k eval-run pool, for the idiom+context pattern -- that pool alone
    only had 75 hits, too few to meaningfully augment training."""
    all_ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM source.articles WHERE body_text IS NOT NULL"
        ).fetchall()
    ]
    rng.shuffle(all_ids)
    print(
        f"Full source corpus: {len(all_ids):,} articles with body_text -- scanning for idiom hits"
    )

    candidates: list[tuple[int, str]] = []
    seen: set[str] = set(already_used)
    scanned = 0
    for start in range(0, len(all_ids), _BATCH_SIZE):
        if len(candidates) >= target_count:
            break
        batch_ids = all_ids[start : start + _BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch_ids)
        rows = conn.execute(
            f"SELECT a.id, a.body_text FROM source.articles a WHERE a.id IN ({placeholders})",  # noqa: S608
            batch_ids,
        ).fetchall()
        scanned += len(rows)
        for article_id, body_text in rows:
            if not body_text:
                continue
            for sent, _s, _e in split_sentences(body_text):
                if not (_VERB_RE.search(sent) and _CONTEXT_RE.search(sent)):
                    continue
                if not _is_plausible_sentence(sent):
                    continue
                norm = " ".join(sent.split())
                if norm in seen:
                    continue
                seen.add(norm)
                candidates.append((article_id, norm))
        if scanned % 20000 < _BATCH_SIZE:
            print(f"  scanned {scanned:,} articles, {len(candidates)} idiom hits so far ...")
    return candidates


def _label_one(client: OpenAI, model: str, sentence: str) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        for label in _LABELS:
            if label in raw:
                return label
    except Exception as exc:  # one failed call must not kill the whole run
        print(f"  [warn] label call failed: {exc}")
        return None
    else:
        return None


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)  # noqa: S311 -- sample selection, not cryptography

    already_used = _existing_sentence_set(args.existing_sentences)
    print(f"Deduping against {len(already_used):,} sentences already in the original training draw")

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    try:
        candidates = _collect_idiom_sentences(conn, rng, args.target_count, already_used)
    finally:
        db.detach_source(conn)
        conn.close()

    print(f"Idiom-family candidate sentences found (full corpus): {len(candidates):,}")
    rng.shuffle(candidates)
    candidates = candidates[: args.target_count]

    print(f"Labeling {len(candidates):,} sentences via {os.environ['LLM_MODEL']} ...")
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_URL"])
    model = os.environ["LLM_MODEL"]

    labeled: list[dict[str, object]] = []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_label_one, client, model, sentence): (article_id, sentence)
            for article_id, sentence in candidates
        }
        for i, future in enumerate(as_completed(futures), start=1):
            article_id, sentence = futures[future]
            label = future.result()
            if label is None:
                n_failed += 1
                continue
            labeled.append({"sentence": sentence, "label": label, "article_id": article_id})
            if i % 200 == 0:
                print(f"  labeled {i}/{len(candidates)} ...")

    counts: dict[str, int] = {}
    for row in labeled:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Done: {len(labeled):,} labeled, {n_failed} failed. Class counts: {counts}")

    rng.shuffle(labeled)
    probe = labeled[: args.probe_count]
    augment = labeled[args.probe_count :]

    args.probe_out.parent.mkdir(parents=True, exist_ok=True)
    with args.probe_out.open("w") as f:
        for row in probe:
            f.write(json.dumps(row) + "\n")
    with args.augment_out.open("w") as f:
        for row in augment:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(probe)} probe (held-out, eval-only) rows to {args.probe_out}")
    print(f"Wrote {len(augment)} augment (training) rows to {args.augment_out}")


if __name__ == "__main__":
    main()
