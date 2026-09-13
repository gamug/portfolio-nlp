#!/usr/bin/env python
"""One-shot: build a fresh, in-domain sentence-level sentiment training set
for fine-tuning FinBERT, using the LLM (DeepSeek, same endpoint
`news_nlp.eval` uses) to label real sentences drawn from this project's own
corpus, instead of relying solely on FinBERT's original 2014
Financial-PhraseBank training data (Nordic-listed companies, no crypto/
modern-instrument vocabulary -- see docs/evaluation.md's 2026-09-13
sentiment follow-up for the real-data evidence this gap causes).

Sentence source: articles already sampled and judged across every prior
`--stage sentiment` eval run (`eval_run`/`eval_judgement`, `news_nlp.eval`)
-- a pool the stratified sampling design (low_conf + target_negative/
positive/neutral + representative) already skewed toward covering all
three classes, not a fresh uniform draw from scratch. Candidate sentences
are split from each article's real `body_text` (`chunking.split_sentences`),
filtered for plausible standalone sentences, then labeled one at a time by
the LLM from an investor/price-impact perspective -- the same framing
Financial PhraseBank's own annotators used, and the same framing the
`news_nlp.eval` judge prompt already uses (see docs/evaluation.md).

Output: a local JSONL file (`data/sentiment_finetune/labeled_sentences.jsonl`,
git-ignored -- this is a data artifact, not code) with one
{sentence, label, article_id} row per labeled sentence. Not written to any
DB table -- this is training data, not an eval run.

Known limitation, disclosed on the resulting model's card: these are
LLM-generated labels (silver-standard), not human-annotated ground truth.

Usage:
    uv run python scripts/label_sentiment_sentences_2026_09_13.py \\
        --results-db /path/to/nlp.db --source-db /path/to/urls.db \\
        --target-count 5000 --seed 1
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

_SYSTEM_PROMPT = """You are a financial analyst labeling individual sentences from financial \
news articles for a sentiment classification dataset.

For the sentence given, decide its sentiment from the viewpoint of a long-only equity \
investor: does this sentence, in isolation, read as good news, bad news, or neither for \
the company/asset it concerns?

- positive: the sentence describes a favorable development (beat, upgrade, growth, gain, \
  positive outlook, strong demand).
- negative: the sentence describes an unfavorable development (miss, downgrade, decline, \
  loss, weak demand, litigation, warning).
- neutral: purely factual/administrative with no clear directional charge, or not about \
  company/market performance at all (e.g. "The company is headquartered in Chicago.").

Reply with exactly one word: positive, negative, or neutral. No punctuation, no \
explanation."""

_SENTENCE_RE = re.compile(r"^\S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/sentiment_finetune/labeled_sentences.jsonl"),
    )
    return parser.parse_args()


def _candidate_article_ids(conn: db.NewsNlpDatabase) -> list[int]:
    """Every article_id ever judged across a --stage sentiment eval run --
    already skewed toward covering positive/negative/neutral content by the
    stratified sampling design (low_conf/target_x/representative), not a
    fresh uniform draw."""
    run_ids = [r[0] for r in conn.execute("SELECT id FROM eval_run WHERE stage='sentiment'")]
    if not run_ids:
        raise RuntimeError("no sentiment eval_run rows found -- nothing to draw sentences from")
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT DISTINCT article_id FROM eval_judgement WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    ).fetchall()
    return [r[0] for r in rows]


def _is_plausible_sentence(text: str) -> bool:
    stripped = text.strip()
    if not (_MIN_SENTENCE_LEN <= len(stripped) <= _MAX_SENTENCE_LEN):
        return False
    if not _SENTENCE_RE.match(stripped):
        return False
    letters = sum(c.isalpha() for c in stripped)
    return letters >= _MIN_LETTER_COUNT  # not just numbers/symbols/tickers


def _collect_candidate_sentences(
    conn: db.NewsNlpDatabase, article_ids: list[int], rng: random.Random, target_count: int
) -> list[tuple[int, str]]:
    """Pull body_text for a shuffled walk of article_ids, splitting into
    sentences and keeping plausible ones, until target_count*1.2 candidates
    are gathered (some headroom for near-duplicates removed below)."""
    shuffled = article_ids[:]
    rng.shuffle(shuffled)
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    need = int(target_count * 1.2)

    batch_size = 200
    for start in range(0, len(shuffled), batch_size):
        if len(candidates) >= need:
            break
        batch_ids = shuffled[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch_ids)
        rows = conn.execute(
            f"SELECT a.id, a.body_text FROM source.articles a "  # noqa: S608
            f"WHERE a.id IN ({placeholders})",
            batch_ids,
        ).fetchall()
        for article_id, body_text in rows:
            if not body_text:
                continue
            for sent, _start, _end in split_sentences(body_text):
                if not _is_plausible_sentence(sent):
                    continue
                norm = " ".join(sent.split())
                if norm in seen:
                    continue
                seen.add(norm)
                candidates.append((article_id, norm))
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

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    try:
        article_ids = _candidate_article_ids(conn)
        print(f"Candidate article pool (from prior sentiment eval runs): {len(article_ids):,}")

        candidates = _collect_candidate_sentences(conn, article_ids, rng, args.target_count)
        print(f"Plausible candidate sentences collected: {len(candidates):,}")
    finally:
        db.detach_source(conn)
        conn.close()

    rng.shuffle(candidates)
    candidates = candidates[: args.target_count]
    print(f"Labeling {len(candidates):,} sentences via {os.environ['LLM_MODEL']} ...")

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_URL"])
    model = os.environ["LLM_MODEL"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(_LABELS, 0)
    n_failed = 0
    done = 0

    with args.out.open("w") as f, ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_label_one, client, model, sentence): (article_id, sentence)
            for article_id, sentence in candidates
        }
        for future in as_completed(futures):
            article_id, sentence = futures[future]
            label = future.result()
            done += 1
            if label is None:
                n_failed += 1
            else:
                counts[label] += 1
                f.write(
                    json.dumps({"sentence": sentence, "label": label, "article_id": article_id})
                    + "\n"
                )
            if done % 250 == 0:
                print(f"  {done:,}/{len(candidates):,} labeled ({counts}, {n_failed} failed)")

    print(f"\nDone: {sum(counts.values()):,} labeled, {n_failed} failed. Class counts: {counts}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
