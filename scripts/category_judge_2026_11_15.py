# ruff: noqa: E402, PLR2004, PLC0415, S311  -- one-shot research script (the eval deps load only on a real run; a seeded draw)
"""The LLM judge on step 7 (T-209, AI behavior #1; the owner approved a small paid run on 2026-11-15).

    uv run --group eval scripts/category_judge_2026_11_15.py --dry-run     # no call: the sample and the size of the bill
    uv run --group eval scripts/category_judge_2026_11_15.py --n 80        # one judge call per text (paid, LLM_* of .env)

A seeded random draw of ``--n`` of the 300 real sub-news of ``gold/texts/category_real_sample_texts.csv`` (the text the step reads, with
its company). The judge (``news_nlp.eval.step_category``, a strands agent over the configured OpenAI-compatible endpoint) labels each text
**blind**: it never sees a model's answer. The two models' labels are read from ``docs/category_results_2026_11_15/real_sample.csv``
(``selected`` = the fine-tune pinned in ``config/category_step.json``, ``untouched`` = the zero-shot model it replaced); no model is
run again. The judge is a model: the figures are agreement with a judge, not truth (``docs/evaluation.md``). Writes
``docs/category_results_2026_11_15/judge_step7.csv`` (no text) and ``judge_step7.md``. Refuses to run if the output exists.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from news_nlp.eval import step_category as sc

OUT = ROOT / "docs" / "category_results_2026_11_15"
TEXTS = ROOT / "gold" / "texts" / "category_real_sample_texts.csv"
LABELS = OUT / "real_sample.csv"
CHARS_PER_TOKEN = 4  # a rough size for the dry run


def draw(n: int, seed: int) -> pd.DataFrame:
    texts = pd.read_csv(TEXTS, dtype=str).fillna("")
    labels = pd.read_csv(LABELS, dtype=str).fillna("")
    df = texts.merge(labels[["idx", "tag", "selected", "untouched"]], on="idx", how="inner")
    assert len(df) == len(texts) == 300, (len(df), len(texts))
    keep = sorted(random.Random(seed).sample(range(len(df)), n))
    return df.iloc[keep].reset_index(drop=True)


def judge_all(df: pd.DataFrame, workers: int) -> list[sc.StepCategoryVerdict]:
    from news_nlp.eval.config import EvalSettings
    from news_nlp.eval.model import build_judge_agent, build_model

    settings = EvalSettings.load(sample_size=len(df))
    model = build_model(settings)
    prompt = sc.system_prompt()

    def one(row: tuple[str, str]) -> sc.StepCategoryVerdict:
        agent = build_judge_agent(model, prompt)  # a fresh, stateless agent per text
        return sc.judge_step_category(agent, row[0], row[1])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, zip(df.company, df.text, strict=True)))


def header(n_asked: int, seed: int, model: str, date: str) -> list[str]:
    prompt = hashlib.sha256(sc.system_prompt().encode()).hexdigest()[:16]
    return [
        f"- Judge model: `{model}` (the `LLM_MODEL` of `.env`; it wrote none of the labels compared). Date {date}. "
        f"Sample: a seeded draw (seed {seed}) of {n_asked} of the 300 real sub-news under 1,000 characters (the text the step reads).",
        f"- Judge prompt `src/news_nlp/eval/prompts/category_step.md`, SHA-256 `{prompt}…`; temperature 0.",
    ]


def read_rows(csv: Path) -> list[sc.JudgedRow]:
    return pd.read_csv(csv, dtype=str).fillna("").to_dict("records")  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rewrite the .md from the committed .csv; no call",
    )
    ap.add_argument("--model", default="", help="with --report-only: the judge model of that run")
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    csv = OUT / f"judge_step7_n{args.n}.csv"
    md = OUT / f"judge_step7_n{args.n}.md"
    if args.report_only:
        rows = read_rows(csv)
        md.write_text(
            sc.report(rows, header(args.n, args.seed, args.model, args.date)), encoding="utf-8"
        )
        print(md)
        return
    df = draw(args.n, args.seed)
    chars = int(df.text.str.len().sum())
    print(
        f"[judge] {len(df)} texts, {chars:,} characters (about {chars // CHARS_PER_TOKEN:,} input tokens "
        f"plus the ~1,000-token system prompt per call and a short reply)",
        flush=True,
    )
    if args.dry_run:
        print("[judge] dry run: no call was made")
        return
    if csv.exists():
        sys.exit(f"{csv} exists: the judge was run on this sample; delete it by hand to run again")
    model = os.environ.get("LLM_MODEL") or load_dotenv_model()
    verdicts = judge_all(df, args.workers)
    df["judge"] = [v.ideal_slug for v in verdicts]
    df["second"] = [v.second_slug or "" for v in verdicts]
    df["parse_failed"] = [str(v.parse_failed) for v in verdicts]
    # The rationales paraphrase the articles: kept local (git-ignored data/), not committed.
    local = ROOT / "data" / f"judge_step7_n{args.n}_rationales.csv"
    local.parent.mkdir(exist_ok=True)
    pd.DataFrame({"idx": df.idx, "rationale": [v.rationale for v in verdicts]}).to_csv(
        local, index=False
    )
    df.drop(columns=["text"]).to_csv(csv, index=False)
    text = sc.report(read_rows(csv), header(args.n, args.seed, model, args.date))
    md.write_text(text, encoding="utf-8")
    print(text)


def load_dotenv_model() -> str:
    from news_nlp.eval.config import EvalSettings

    return EvalSettings.load().llm_model


if __name__ == "__main__":
    main()
