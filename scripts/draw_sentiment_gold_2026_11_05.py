"""Draw the texts to tag for the sentiment gold on real pipeline output (Work item 18, steps 1-5).

    uv run scripts/draw_sentiment_gold_2026_11_05.py [--out data] [--per-class 60] [--seed 20261105] [--yes]

The 7-step pipeline does not exist yet, so its first steps are emulated on the bodies the owner has
tagged with a scope (``gold/category_gold_v2.csv``; ``unrelated`` is skipped, the held-out set is
never used):

* step 1, the scope: the owner's tag;
* step 3, a body that is not *multi*: the local summary of the pipeline's own model
  (``config/pipeline_models.json`` ``c_summary``) over the title and body;
* step 4, a *multi* body: the text of each company by ``company_texts.split_company_texts`` (one
  segment per company). A body that is not structured that way is left out and counted: its texts
  will come from the LLM exception path, which has its own gold later. A company text over
  ``MAX_COMPANY_CHARS`` is left out and counted too.

Step 5's current model (``config/pipeline_models.json`` ``sentiment``) scores every text only to
choose the sample, the way the pipeline's sentiment stage reads a text: sentence-packed chunks of
at most 510 tokens (``chunking.chunk_text``), the chunk probabilities averaged with equal weights
(the text is already the company's, so there is no subject weighting, D26): ``--per-class`` texts of each predicted class, so the rare negative class is
covered; the prediction is never written to the tagging file, whose rows are shuffled. The ids file
holds the prediction and the size of each stratum, to re-weight the result.

A multi body the parser cannot split and a company text over ``MAX_COMPANY_CHARS`` are left out of
the pool and counted in ``<out>/sentiment_gold_pool.json`` with the pool sizes, so the limit of the
gold is on record: until the attribution model (T-148b3) exists, the gold does not cover the texts
that model will produce.

Writes ``<out>/sentiment_gold_ids.csv`` (drawn once, with the texts) and
``<out>/sentiment_gold_to_tag.csv``, a tagging file in the sentiment format
(``retag.SENTIMENT_TAGGING_COLUMNS``, the Spanish pair right after ``text``). The Spanish aid is the
approved paid translation of the same engine; nothing is bought without ``--yes``. Its cache is
``<out>/sentiment_gold/``, apart from the scope files' cache, because a body has several texts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import company_texts, retag
from news_nlp.company_names import NameBook
from news_nlp.triage import CompanyResolver

IDS_NAME = "sentiment_gold_ids.csv"
TO_TAG_NAME = "sentiment_gold_to_tag.csv"
POOL_NAME = "sentiment_gold_pool.json"
CACHE_DIR = "sentiment_gold"
GOLD = Path("gold/category_gold_v2.csv")
CONFIG = Path("config/pipeline_models.json")
SUMMARY_SCOPES = ("single", "sector", "market_economy", "investing_general")
MAX_COMPANY_CHARS = 1500
MIN_TEXT_CHARS = 40
CLASSES = ("positive", "negative", "neutral")
SUBJECT = {
    "market_economy": "[market / economy]",
    "investing_general": "[investing in general]",
}
IDS_FIELDS = (
    "key",
    "article_id",
    "kind",
    "scope",
    "ticker",
    "company",
    "predicted",
    "population",
    "title",
    "text",
)


def tagged_bodies(out: Path) -> list[dict[str, str]]:
    """The owner's scope tags, with the text of the tagging files, minus ``unrelated``."""
    tags = [r for r in extra._read(GOLD) if r["scope"] and r["scope"] != "unrelated"]
    held = {r["article_id"] for r in extra._read(out / "heldout_ids.csv")}
    clash = held & {r["article_id"] for r in tags}
    if clash:
        sys.exit(f"held-out articles among the tagged bodies: {sorted(clash)[:5]}")
    return tags


def subject_of(row: dict[str, str], book: NameBook, names: dict[str, str]) -> tuple[str, str]:
    """(ticker, company) the sentiment of a non-multi text is about."""
    scope = row["scope"]
    if scope == "single":
        first = row["company_labels"].split(",")[0].split(":")[0].strip()
        ticker = book.resolve(first)[0] or row["ticker"]
        return ticker, names.get(ticker, first)
    if scope == "sector":
        return "", f"[sector {row['sub_industry']}]"
    return "", SUBJECT[scope]


def build_texts(out: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    conn = db.connect_pipeline()
    universe = retag_cli.companies()
    names = {c.ticker: c.name for c in universe}
    resolver = CompanyResolver(universe)
    book = NameBook(universe, retag.Universe.of(universe).tickers)
    texts: list[dict[str, str]] = []
    to_summarise: list[tuple[dict[str, str], str]] = []
    skipped = Counter()
    for row in tagged_bodies(out):
        src = conn.execute(
            "SELECT title, body_text FROM source.articles WHERE id = ?", (int(row["article_id"]),)
        ).fetchone()
        title, body = src["title"] or "", src["body_text"] or ""
        base = {"article_id": row["article_id"], "scope": row["scope"], "title": title}
        if row["scope"] in SUMMARY_SCOPES:
            ticker, company = subject_of(row, book, names)
            to_summarise.append(
                (
                    {**base, "kind": "summary", "ticker": ticker, "company": company},
                    f"Title-{title}\nBody-{body}",
                )
            )
        elif row["scope"] == "multi":
            found = company_texts.split_company_texts(body, resolver)
            if found is None:
                skipped["multi, not structured"] += 1
                continue
            for ticker, text in found.items():
                if len(text) > MAX_COMPANY_CHARS:
                    skipped["company text too long"] += 1
                elif len(text) < MIN_TEXT_CHARS:
                    skipped["company text too short"] += 1
                else:
                    texts.append(
                        {
                            **base,
                            "kind": "company_text",
                            "ticker": ticker,
                            "company": names.get(ticker, ticker),
                            "text": text,
                        }
                    )
    texts += summarise(to_summarise)
    print("left out:", dict(skipped))
    return [t for t in texts if len(t["text"]) >= MIN_TEXT_CHARS], skipped


def _models() -> dict[str, dict[str, str]]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def summarise(items: list[tuple[dict[str, str], str]]) -> list[dict[str, str]]:
    import torch  # noqa: PLC0415 -- the models are only needed to draw
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    from summary_stage import hierarchical_summarize_batch  # noqa: PLC0415

    cfg = _models()["c_summary"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = model.to(device).eval()
    done = hierarchical_summarize_batch([t for _, t in items], tok, model, device, batch_size=4)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return [{**meta, "text": summary} for (meta, _), (summary, _) in zip(items, done, strict=True)]


def predicted_label(chunk_probs: list[list[float]], id2label: dict[int, str]) -> str:
    """The class of the mean of the chunks' probabilities (equal weights)."""
    mean = [sum(col) / len(chunk_probs) for col in zip(*chunk_probs, strict=True)]
    return id2label[max(range(len(mean)), key=mean.__getitem__)]


def predict(texts: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add the current sentiment model's label to each text (to choose the sample only), read as
    the pipeline's stage reads a text. Returns the texts that have at least one chunk."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    from chunking import chunk_text  # noqa: PLC0415

    cfg = _models()["sentiment"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], revision=cfg["revision"]
    )
    model = model.to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    kept: list[dict[str, str]] = []
    for item in texts:
        chunks = chunk_text(item["text"], tok, max_tokens=510)
        if not chunks:
            continue
        chunk_probs: list[list[float]] = []
        for chunk in chunks:
            inputs = tok(chunk.text, return_tensors="pt", truncation=True, max_length=512).to(
                device
            )
            with torch.no_grad():
                chunk_probs.append(torch.softmax(model(**inputs).logits[0], dim=-1).cpu().tolist())
        item["predicted"] = predicted_label(chunk_probs, id2label)
        kept.append(item)
    return kept


def draw(texts: list[dict[str, str]], per_class: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    population = Counter(t["predicted"] for t in texts)
    chosen: list[dict[str, str]] = []
    for label in CLASSES:
        members = sorted(
            (t for t in texts if t["predicted"] == label),
            key=lambda t: (int(t["article_id"]), t["ticker"], t["kind"]),
        )
        rng.shuffle(members)
        chosen += [{**t, "population": str(population[label])} for t in members[:per_class]]
    rng.shuffle(chosen)  # the order must not reveal the predicted class
    for t in chosen:
        t["key"] = f"{t['article_id']}#{t['ticker'] or t['kind']}"
    return chosen


def source_meta(ids: list[str]) -> dict[str, dict[str, str]]:
    conn = db.connect_pipeline()
    meta: dict[str, dict[str, str]] = {}
    for article_id in ids:
        r = conn.execute(
            "SELECT pub_date, source_domain FROM source.articles WHERE id = ?", (int(article_id),)
        ).fetchone()
        meta[article_id] = {
            "pub_date": r["pub_date"] or "",
            "source_domain": r["source_domain"] or "",
        }
    return meta


def tag_rows(
    chosen: list[dict[str, str]], spanish: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    meta = source_meta(sorted({t["article_id"] for t in chosen}))
    rows: list[dict[str, str]] = []
    for order, t in enumerate(chosen, 1):
        es = spanish[t["key"]]
        rows.append(
            {
                "review_order": str(order),
                "article_id": t["article_id"],
                "ticker": t["ticker"],
                "company": t["company"],
                **meta[t["article_id"]],
                "title": t["title"],
                "text": t["text"],
                "spanish_title": es["spanish_title"],
                "spanish_text": es["spanish_text"],
                "kind": t["kind"],
                "sentiment": "",
                "notes": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    ids_path, target = out / IDS_NAME, out / TO_TAG_NAME
    if target.exists():
        sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
    if ids_path.exists():
        chosen = extra._read(ids_path)
        print(f"{ids_path} exists: the set is not drawn again")
    else:
        texts, skipped = build_texts(out)
        texts = predict(texts)
        pool = {
            f"{k}/{p}": n
            for (k, p), n in Counter((t["kind"], t["predicted"]) for t in texts).items()
        }
        print("pool:", pool)
        (out / POOL_NAME).write_text(
            json.dumps({"left_out": dict(skipped), "pool": pool}, indent=2), encoding="utf-8"
        )
        chosen = draw(texts, args.per_class, args.seed)
        with ids_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(IDS_FIELDS), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(chosen)
        print(f"{len(chosen)} drawn:", dict(Counter(t["predicted"] for t in chosen)))
        print("by kind:", dict(Counter(t["kind"] for t in chosen)))
    cache = out / CACHE_DIR
    cache.mkdir(exist_ok=True)
    keyed = [{"article_id": t["key"], "title": t["title"], "text": t["text"]} for t in chosen]
    done = retag_cli.add_spanish(
        cache, keyed, retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    spanish = {r["article_id"]: r for r in done}
    rows = tag_rows(chosen, spanish)
    try:
        retag.require_tagging_format(rows, target.name, retag.SENTIMENT_TAGGING_COLUMNS)
        extra._write(target, retag.spreadsheet_safe(rows))
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    except PermissionError:
        sys.exit(f"cannot write {target}: close it in Excel and run again")
    print(f"{len(rows)} texts -> {target}")


if __name__ == "__main__":
    main()
