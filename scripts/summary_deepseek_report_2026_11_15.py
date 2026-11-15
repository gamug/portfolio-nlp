"""The committed figures of the DeepSeek arm of the summary experiment (T-206), and the paired table that cites them.

    uv run scripts/summary_deepseek_report_2026_11_15.py export [--results experiments/results/summary_lead_2026_11_13]
    uv run scripts/summary_deepseek_report_2026_11_15.py tables

``export`` reads the git-ignored scratch of ``summary_lead_experiment_2026_11_13.py`` and ``summary_deepseek_arm_2026_11_15.py`` (sample, arms, the DeepSeek
summaries and the judge's verdicts) and writes ``docs/summarization_results_2026_11_15/deepseek_vs_current.csv``: one row per article with the judge's scores for
arm B (the current step) and arm D (DeepSeek), the tokens the call used and the lengths. **No text**: neither the articles nor the summaries.
``tables`` needs only the committed files (``deepseek_vs_current.csv`` and ``blind_24.csv``) and prints the paired differences with 95% bootstrap
intervals, so the figures of ``docs/summarization_training.md`` can be checked from a clean checkout; ``tests/news_nlp/test_summary_deepseek_figures.py`` does it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "summarization_results_2026_11_15"
MEASURES = ("faithfulness", "coverage", "conciseness")


def sha(summary: str) -> str:
    return hashlib.sha1(summary.encode()).hexdigest()[:16]  # noqa: S324 -- the cache key of the judge, not security


def export(results: Path) -> None:
    sample = {
        r["article_id"]: r
        for r in json.loads((results / "sample.json").read_text(encoding="utf-8"))
    }
    arms = json.loads((results / "arms.json").read_text(encoding="utf-8"))
    deepseek = {
        r["article_id"]: r
        for r in json.loads((results / "deepseek.json").read_text(encoding="utf-8"))
    }
    verdicts = {}
    for line in (results / "verdicts.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        verdicts[(r["article_id"], r["sha"])] = r["verdict"]
    missing = [a["article_id"] for a in arms if "D" not in a["arms"]]
    if missing:
        raise SystemExit(
            f"{len(missing)} articles have no arm D in arms.json: run `summary_deepseek_arm_2026_11_15.py generate` and `attach` first"
        )
    unjudged = [
        a["article_id"]
        for a in arms
        if (a["article_id"], sha(a["arms"]["D"])) not in verdicts
        or (a["article_id"], sha(a["arms"]["B"])) not in verdicts
    ]
    if unjudged:
        raise SystemExit(
            f"{len(unjudged)} articles have no verdict for arm B or D: run `summary_lead_experiment_2026_11_13.py judge` first"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["article_id", "tier", "language", "body_chars"]
    for arm in ("B", "D"):
        columns += [f"{arm}_{m}" for m in MEASURES] + [f"{arm}_hallucinated", f"{arm}_words"]
    columns += ["D_tokens_in", "D_tokens_out"]
    with (OUT / "deepseek_vs_current.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for a in arms:
            aid = a["article_id"]
            row = [aid, sample[aid]["tier"], sample[aid]["language"], len(sample[aid]["body"])]
            for arm in ("B", "D"):
                v = verdicts[(aid, sha(a["arms"][arm]))]
                row += [v[m] for m in MEASURES] + [
                    int(bool(v["hallucinations"])),
                    len(a["arms"][arm].split()),
                ]
            row += [deepseek[aid]["in"], deepseek[aid]["out"]]
            writer.writerow(row)
    print(f"wrote {OUT / 'deepseek_vs_current.csv'} ({len(arms)} rows, no text)")


def boot(diffs: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)  # noqa: S311 -- a bootstrap, not security
    k = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(n))
    return sum(diffs) / k, means[int(0.025 * n)], means[int(0.975 * n)]


def read(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tables() -> str:
    rows = read("deepseek_vs_current.csv")
    lines = [
        f"Paired by article, {len(rows)} articles; 95% bootstrap intervals (seed 0, 4000 draws). B = the current step, D = DeepSeek.",
        "",
    ]
    lines += ["| | B | D | D - B [95%] |", "|---|---|---|---|"]
    for m in MEASURES:
        b = [float(r[f"B_{m}"]) for r in rows]
        d = [float(r[f"D_{m}"]) for r in rows]
        mean, lo, hi = boot([y - x for x, y in zip(b, d, strict=True)])
        lines.append(
            f"| {m} | {sum(b) / len(b):.2f} | {sum(d) / len(d):.2f} | {mean:+.2f} [{lo:+.2f}, {hi:+.2f}] |"
        )
    hb = [float(r["B_hallucinated"]) for r in rows]
    hd = [float(r["D_hallucinated"]) for r in rows]
    mean, lo, hi = boot([y - x for x, y in zip(hb, hd, strict=True)])
    lines.append(
        f"| hallucination | {100 * sum(hb) / len(hb):.1f}% | {100 * sum(hd) / len(hd):.1f}% | {100 * mean:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] |"
    )
    lines += [
        "",
        "| tier | n | B faith / cov / halluc | D faith / cov / halluc |",
        "|---|---|---|---|",
    ]
    for t in ("1", "2", "3"):
        sub = [r for r in rows if r["tier"] == t]
        cells = []
        for arm in ("B", "D"):
            f_ = sum(float(r[f"{arm}_faithfulness"]) for r in sub) / len(sub)
            c_ = sum(float(r[f"{arm}_coverage"]) for r in sub) / len(sub)
            h_ = sum(float(r[f"{arm}_hallucinated"]) for r in sub) / len(sub)
            cells.append(f"{f_:.2f} / {c_:.2f} / {100 * h_:.0f}%")
        lines.append(f"| {t} | {len(sub)} | {cells[0]} | {cells[1]} |")
    tin = sum(int(r["D_tokens_in"]) for r in rows)
    tout = sum(int(r["D_tokens_out"]) for r in rows)
    lines += [
        "",
        f"Tokens of the DeepSeek calls: {tin} in ({tin // len(rows)} a call), {tout} out ({tout // len(rows)} a call).",
    ]
    blind = read("blind_24.csv")
    lines += [
        "",
        f"Blind re-scoring of {len(blind)} pairs (8 per tier, English; the arm hidden when scored; `blind_24.csv`):",
        "",
        "| | B | D |",
        "|---|---|---|",
    ]
    for m in ("faithfulness", "coverage", "conciseness"):
        lines.append(
            f"| {m} | {sum(float(r[f'B_{m}']) for r in blind) / len(blind):.2f} | {sum(float(r[f'D_{m}']) for r in blind) / len(blind):.2f} |"
        )
    lines.append(
        f"| hallucination | {100 * sum(int(r['B_hallucinated']) for r in blind) / len(blind):.0f}% | {100 * sum(int(r['D_hallucinated']) for r in blind) / len(blind):.0f}% |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["export", "tables"])
    ap.add_argument("--results", default=str(ROOT / "experiments/results/summary_lead_2026_11_13"))
    args = ap.parse_args()
    if args.cmd == "export":
        export(Path(args.results))
    else:
        text = tables()
        (OUT / "tables.md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
