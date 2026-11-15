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
import random
import sys
from concurrent.futures import ThreadPoolExecutor
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


def figures(df: pd.DataFrame) -> str:
    ok = df[~df.parse_failed].reset_index(drop=True)
    judge, second = list(ok.judge), [s or None for s in ok.second]
    lines = [
        f"# The LLM judge on step 7: {len(df)} real sub-news, blind (T-209)\n",
        f"- {len(ok)} judged, {len(df) - len(ok)} parse failures (excluded). The judge is a model: agreement with a judge, not truth.",
        "- Texts: a seeded draw of the 300 real sub-news under 1,000 characters (the text the step reads); the judge never saw a model's label.\n",
        "| labels | agree with the judge (strict) [95%] | with the judge's second label too |\n|---|---|---|",
    ]
    hits: dict[str, list[bool]] = {}
    for name, col in (
        ("untouched zero-shot (the step until now)", "untouched"),
        ("**fine-tune pinned in `config/category_step.json`**", "selected"),
        ("the assistant's tags (calibration of the judge)", "tag"),
    ):
        model = list(ok[col])
        hits[col] = [m == j for m, j in zip(model, judge, strict=True)]
        a = sc.agreement(judge, model, second)
        lo, hi = sc.bootstrap_ci(hits[col])
        lines.append(f"| {name} | {a['strict']:.1%} [{lo:.1%}, {hi:.1%}] | {a['lenient']:.1%} |")
    sel_only = sum(s and not u for s, u in zip(hits["selected"], hits["untouched"], strict=True))
    unt_only = sum(u and not s for s, u in zip(hits["selected"], hits["untouched"], strict=True))
    p = sc.sign_test(sel_only, unt_only)
    lines.append(
        f"\nPaired: the fine-tune agrees with the judge where the untouched model does not on {sel_only} texts, the reverse on {unt_only}; "
        f"exact sign test p = {p:.3f}."
    )
    other_j = sum(j == "other" for j in judge)
    other_s = sum(m == "other" for m in ok.selected)
    lines.append(
        f"\n`other`: the judge says `other` on {other_j} texts, the pinned model on {other_s}."
    )
    conf = pd.crosstab(pd.Series(judge, name="judge"), pd.Series(ok.selected, name="pinned model"))
    pairs = sorted(
        (
            (g, p_, int(conf.loc[g, p_]))
            for g in conf.index
            for p_ in conf.columns
            if g != p_ and conf.loc[g, p_] > 0
        ),
        key=lambda x: -x[2],
    )
    lines.append("\nLargest disagreements (judge -> pinned model):")
    lines += [f"- `{g}` -> `{p_}`: {k}" for g, p_, k in pairs[:6]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    csv = OUT / "judge_step7.csv"
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
    verdicts = judge_all(df, args.workers)
    df["judge"] = [v.ideal_slug for v in verdicts]
    df["second"] = [v.second_slug or "" for v in verdicts]
    df["parse_failed"] = [v.parse_failed for v in verdicts]
    df["rationale"] = [v.rationale for v in verdicts]
    df.drop(columns=["text"]).to_csv(csv, index=False)
    text = figures(df)
    (OUT / "judge_step7.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
