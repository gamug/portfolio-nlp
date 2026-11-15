# ruff: noqa: E402  -- one-shot script
"""How much text the summary step sends to a summariser, by population, and what that costs through an API (T-206).

    uv run scripts/summary_chars_to_summarise_2026_11_15.py --results-db <nlp.db copy> --source-db <urls.db copy> \\
        --scope-revision scope-nli-8ff21e50796b --cut-revision units5-d3e2cbfa830e --summary-revision summary-7a9bd2062bc4

Read-only. Three populations, each as step 2 reads them (``Title-<title>\\nBody-<body>`` for a body, a sub-news alone) and only the texts **over 2,000
characters**, which are the ones the summariser is run on: the single bodies of the scope revision, the sub-news of the cut, and the **sector bodies** the step
summarised (``news_summary`` body rows of the summary revision that are not scope ``single``: T-167). Writes ``docs/summarization_results_2026_11_15/
chars_to_summarise.md`` with the counts, the characters and the cost at the rates written in the file (the rates are an input, not a fact: confirm them on the
vendor's page before a run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import news_nlp as db
from news_nlp import steps, triage
from news_summary_stage import MIN_CHARS, input_of_body

OUT = ROOT / "docs" / "summarization_results_2026_11_15" / "chars_to_summarise.md"
CHARS_PER_TOKEN = (
    4.31  # measured on the 300 articles of the experiment: 7,013 characters over 1,625 input tokens
)
PROMPT_TOKENS = 85  # the instruction and the framing of each call
OUTPUT_TOKENS = 87  # the mean of the 300 calls of the experiment
RATES = {
    "low": (0.15, 0.60),
    "high": (0.30, 1.20),
}  # US dollars per million tokens, (cache-miss input, output): the pricing page of 2026-11-15, off-peak and peak


def populations(
    conn: db.NewsNlpDatabase, scope: str, cut: str, summary: str
) -> dict[str, list[int]]:
    single = {
        str(r[0])
        for r in conn.execute(
            "SELECT body_hash FROM body_scope WHERE classifier_revision = ? AND scope = 'single'",
            (scope,),
        )
    }
    summarised = {
        str(r[0])
        for r in conn.execute(
            "SELECT item_id FROM news_summary WHERE item_kind = 'body' AND model_revision = ? AND text_source = 'summary'",
            (summary,),
        )
    }
    sector = summarised - single
    marks = ", ".join("?" for _ in steps.PROPER_STATUSES)
    lengths: dict[str, list[int]] = {"single": [], "sub_news": [], "sector": []}
    seen: set[str] = set()
    wanted = single | sector
    for title, text in conn.execute(
        f"SELECT title, body_text FROM {conn.articles_rel}.articles WHERE fetch_status IN ({marks}) AND body_text IS NOT NULL AND body_text != '' ORDER BY id",  # noqa: S608 -- placeholders only
        steps.PROPER_STATUSES,
    ):
        key = triage.body_hash(text)
        if key in wanted and key not in seen:
            seen.add(key)
            n = len(input_of_body(title or "", text))
            if n > MIN_CHARS:
                lengths["single" if key in single else "sector"].append(n)
            if len(seen) == len(wanted):
                break
    lengths["sub_news"] = [
        len(str(r[0]))
        for r in conn.execute("SELECT text FROM sub_news WHERE extractor_revision = ?", (cut,))
        if len(str(r[0])) > MIN_CHARS
    ]
    return lengths


def render(lengths: dict[str, list[int]]) -> str:
    items = sum(len(v) for v in lengths.values())
    chars = sum(sum(v) for v in lengths.values())
    tin = chars / CHARS_PER_TOKEN + items * PROMPT_TOKENS
    tout = items * OUTPUT_TOKENS
    lines = [
        "# Text sent to a summariser by population (2026-11-15)",
        "",
        "Only the texts over 2,000 characters (the size rule of step 2). `uv run scripts/summary_chars_to_summarise_2026_11_15.py ...` (read-only).",
        "",
        "| population | items | characters | mean | longest |",
        "|---|---|---|---|---|",
    ]
    for name, v in lengths.items():
        mean = sum(v) // max(1, len(v))
        lines.append(f"| {name} | {len(v):,} | {sum(v):,} | {mean:,} | {max(v, default=0):,} |")
    lines.append(
        f"| **all** | **{items:,}** | **{chars:,}** | {chars // max(1, items):,} | {max((max(v, default=0) for v in lengths.values()), default=0):,} |"
    )
    lines += [
        "",
        f"Tokens: {CHARS_PER_TOKEN} characters per input token (measured on the 300 articles of the experiment), {PROMPT_TOKENS} tokens of prompt and {OUTPUT_TOKENS} of output per call: "
        f"**{tin / 1e6:.1f}M input and {tout / 1e6:.1f}M output**.",
        "",
        "| rates (per million tokens: input / output) | cost |",
        "|---|---|",
    ]
    for label, (rin, rout) in RATES.items():
        lines.append(
            f"| {label}: ${rin:.2f} / ${rout:.2f} | **${tin / 1e6 * rin + tout / 1e6 * rout:,.1f}** |"
        )
    lines += [
        "",
        "The rates are those of the vendor's pricing page of 2026-11-15, off-peak (low) and peak (high), cache-miss input; confirm them on the page before a run.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("results-db", "source-db", "scope-revision", "cut-revision", "summary-revision"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    db.env.load_env_file()
    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    text = render(populations(conn, args.scope_revision, args.cut_revision, args.summary_revision))
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
