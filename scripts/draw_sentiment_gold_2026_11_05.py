"""Draw the texts to tag for the sentiment gold on real pipeline output (Work item 18, steps 1-5).

    uv run scripts/draw_sentiment_gold_2026_11_05.py [--out data] [--per-class 40] [--per-class-extractor 20]
        [--extractor baseline] [--seed 20261105] [--pool-only] [--yes]

The 7-step pipeline does not exist yet, so its first steps are emulated on the bodies the owner has
tagged with a scope (``gold/category_gold_v2.csv``; ``unrelated`` is skipped, the held-out set is
never used):

* step 1, the scope: the owner's tag;
* step 3, a body that is not *multi*: the local summary of the pipeline's own model
  (``config/pipeline_models.json`` ``c_summary``) over the title and body;
* step 4, a *multi* body: the text of each company by ``company_texts.split_company_texts`` (one
  segment per company). A body that is not structured that way is narrative, and the texts of its
  companies are written by the extractor (``--extractor``: the baseline today, the trained model
  once T-148b3 exists; kind ``extractor_text``). A company text over ``MAX_COMPANY_CHARS`` is left
  out and counted.

Step 5's current model (``config/pipeline_models.json`` ``sentiment``) scores every text only to
choose the sample, the way the pipeline's sentiment stage reads a text: sentence-packed chunks of
at most 510 tokens (``chunking.chunk_text``), the chunk probabilities averaged with equal weights
(the text is already the company's, so there is no subject weighting, D26). The draw has two parts
(owner, 2026-11-06): ``--per-class`` texts of each predicted class from the summaries and the
structured multi texts (part ``checked``, 40 per class) and ``--per-class-extractor`` from the
extractor's texts of narrative multi bodies (part ``extractor``, 20 per class): 60 per class, 180 in
all, so the rare negative class is covered. Each stratum is a part and a predicted class, and the
figures are re-weighted by its size and reported for each part apart. The prediction is never written
to the tagging file, whose rows are shuffled. The ids file holds the prediction, the part and the size
of each stratum.

A company text over ``MAX_COMPANY_CHARS`` is left out of the pool and counted in
``<out>/sentiment_gold_pool.json`` with the pool sizes. The scored pool is kept in
``<out>/sentiment_gold_pool.csv`` with its conditions (``sentiment_gold_pool_meta.json``) and read
back by the next run, which stops if the extractor, the tagged bodies or the model revisions differ;
``--pool-only`` scores it and prints the sizes without drawing, to see a short stratum before the
draw day. A stratum with fewer texts than the draw takes is drawn whole (owner, D34): the others are
unchanged, and the real counts are in the ids file and the report.

Writes ``<out>/sentiment_gold_ids.csv`` (drawn once, with the texts) and
``<out>/sentiment_gold_to_tag.csv``, a tagging file in the sentiment format
(``retag.SENTIMENT_TAGGING_COLUMNS``, the Spanish pair right after ``text``). The Spanish aid is the
approved paid translation of the same engine; nothing is bought without ``--yes``. Its cache is
``<out>/sentiment_gold/``, apart from the scope files' cache, because a body has several texts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from news_nlp import attribution_baseline, company_texts, retag
from news_nlp.company_names import NameBook
from news_nlp.triage import CompanyResolver

IDS_NAME = "sentiment_gold_ids.csv"
TO_TAG_NAME = "sentiment_gold_to_tag.csv"
POOL_NAME = "sentiment_gold_pool.json"
POOL_CSV_NAME = "sentiment_gold_pool.csv"
POOL_META_NAME = "sentiment_gold_pool_meta.json"
CACHE_DIR = "sentiment_gold"
GOLD = Path("gold/category_gold_v2.csv")
CONFIG = Path("config/pipeline_models.json")
SUMMARY_SCOPES = ("single", "sector", "market_economy", "investing_general")
MAX_COMPANY_CHARS = 1500
MIN_TEXT_CHARS = 40
CLASSES = ("positive", "negative", "neutral")
#: Who writes the company texts of a narrative multi body (the trained model, T-148b3, is not here yet).
EXTRACTORS = {"baseline": attribution_baseline.baseline_company_texts}
SUBJECT = {
    "market_economy": "[market / economy]",
    "investing_general": "[investing in general]",
}
POOL_FIELDS = (
    "article_id",
    "kind",
    "extractor",
    "revision",
    "scope",
    "ticker",
    "company",
    "predicted",
    "title",
    "text",
)
IDS_FIELDS = (
    "key",
    "article_id",
    "kind",
    "part",
    "text_hash",
    "revision",
    "stratum",
    "extractor",
    "scope",
    "ticker",
    "company",
    "predicted",
    "population",
    "title",
    "text",
)


def tagged_bodies(out: Path) -> list[dict[str, str]]:
    """The owner's scope tags, with the text of the tagging files, minus ``unrelated``. A tagged
    body that is also a held-out body (under any article id: twins share a body) stops the run."""
    tags = [r for r in extra._read(GOLD) if r["scope"] and r["scope"] != "unrelated"]
    conn = db.connect_pipeline()
    held = extra.excluded_hashes(
        conn, {int(r["article_id"]) for r in extra._read(out / "heldout_ids.csv")}
    )
    tagged = extra.excluded_hashes(conn, {int(r["article_id"]) for r in tags})
    clash = held & tagged
    if clash:
        sys.exit(f"{len(clash)} held-out bodies are among the tagged bodies")
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


def text_revision(kind: str, extractor: str, summariser: dict[str, str]) -> str:
    """How a text was made, for the ``revision`` column: the summariser ``model@revision``, the
    parser's, or the extractor's (``retag.code_revision`` of the code it runs)."""
    if kind == "summary":
        return f"{summariser['model']}@{summariser['revision']}"
    if kind == "extractor_text":
        return retag.extractor_revision(extractor)
    return retag.parser_revision()


def build_texts(out: Path, extractor: str) -> tuple[list[dict[str, str]], Counter[str]]:
    conn = db.connect_pipeline()
    universe = retag_cli.companies()
    names = {c.ticker: c.name for c in universe}
    resolver = CompanyResolver(universe)
    book = NameBook(universe, retag.Universe.of(universe).tickers)
    texts: list[dict[str, str]] = []
    to_summarise: list[tuple[dict[str, str], str]] = []
    skipped = Counter()
    summariser = _models()["c_summary"]
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
                    {
                        **base,
                        "kind": "summary",
                        "revision": text_revision("summary", extractor, summariser),
                        "ticker": ticker,
                        "company": company,
                    },
                    f"Title-{title}\nBody-{body}",
                )
            )
        elif row["scope"] == "multi":
            found = company_texts.split_company_texts(body, resolver)
            kind = "company_text"
            if found is None:
                skipped["multi, not structured (extractor texts)"] += 1
                found = EXTRACTORS[extractor](body, resolver)
                kind = "extractor_text"
            for ticker, text in found.items():
                if len(text) > MAX_COMPANY_CHARS:
                    skipped["company text too long"] += 1
                elif len(text) < MIN_TEXT_CHARS:
                    skipped["company text too short"] += 1
                else:
                    texts.append(
                        {
                            **base,
                            "kind": kind,
                            "extractor": extractor if kind == "extractor_text" else "",
                            "revision": text_revision(kind, extractor, summariser),
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


def predict(
    texts: list[dict[str, str]], cfg: dict[str, str] | None = None, pair: bool = False
) -> list[dict[str, str]]:
    """Add the sentiment model's label to each text (the pinned one, or *cfg*'s ``model`` and
    ``revision``), read as the pipeline's stage reads a text. Returns the texts that have at least
    one chunk. With *pair* the model reads the pair company + chunk (``sentiment_pair``): every text
    carries its ``company`` and the chunks are cut to the room the company leaves."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    import sentiment_pair  # noqa: PLC0415
    from chunking import chunk_text  # noqa: PLC0415

    cfg = cfg or _models()["sentiment"]
    revision = cfg.get("revision") or None  # a local path has none
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], revision=revision)
    model = model.to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    kept: list[dict[str, str]] = []
    for item in texts:
        budget = sentiment_pair.chunk_budget(tok, item["company"]) if pair else 510
        chunks = chunk_text(item["text"], tok, max_tokens=budget)
        if not chunks:
            continue
        chunk_probs: list[list[float]] = []
        for chunk in chunks:
            if pair:
                inputs = sentiment_pair.encode_pair(
                    tok, item["company"], chunk.text, return_tensors="pt"
                ).to(device)
            else:
                inputs = tok(chunk.text, return_tensors="pt", truncation=True, max_length=512).to(
                    device
                )
            with torch.no_grad():
                chunk_probs.append(torch.softmax(model(**inputs).logits[0], dim=-1).cpu().tolist())
        item["predicted"] = predicted_label(chunk_probs, id2label)
        kept.append(item)
    return kept


def draw(
    texts: list[dict[str, str]], per_class: int, seed: int, per_class_extractor: int = 0
) -> list[dict[str, str]]:
    """``per_class`` texts of each predicted class from the summaries and the structured multi
    texts (part ``checked``), and ``per_class_extractor`` of each class from the texts an extractor
    wrote for narrative multi bodies (part ``extractor``). Each stratum is a part and a predicted
    class; ``population`` is its size in the pool, to re-weight the result."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    for t in texts:
        t["part"] = "extractor" if t["kind"] == "extractor_text" else "checked"
        t["stratum"] = f"{t['part']}/{t['predicted']}"
    population = Counter(t["stratum"] for t in texts)
    chosen: list[dict[str, str]] = []
    for part, take in (("checked", per_class), ("extractor", per_class_extractor)):
        for label in CLASSES:
            members = sorted(
                (t for t in texts if t["stratum"] == f"{part}/{label}"),
                key=lambda t: (int(t["article_id"]), t["ticker"], t["kind"]),
            )
            rng.shuffle(members)
            chosen += [{**t, "population": str(population[t["stratum"]])} for t in members[:take]]
    rng.shuffle(chosen)  # the order must not reveal the predicted class
    for t in chosen:
        t["key"] = f"{t['article_id']}#{t['ticker'] or t['kind']}"
        t["text_hash"] = retag.text_hash(t["text"])
    repeated = sorted(k for k, n in Counter(t["key"] for t in chosen).items() if n > 1)
    if repeated:
        raise ValueError(f"two drawn texts share the key {repeated[:5]}")
    return chosen


def part_of(text: dict[str, str]) -> str:
    return "extractor" if text["kind"] == "extractor_text" else "checked"


def pool_conditions(extractor: str) -> dict[str, str]:
    """What a scored pool depends on: who wrote the narrative multi texts, the tagged bodies (and
    their scopes) it was built from, and the pinned summary and sentiment models (AI #1)."""
    tags = sorted(
        f"{r['article_id']}:{r['scope']}"
        for r in extra._read(GOLD)
        if r["scope"] and r["scope"] != "unrelated"
    )
    models = _models()
    return {
        "extractor": extractor,
        "gold_bodies": hashlib.sha256("\n".join(tags).encode()).hexdigest()[:16],
        **{
            stage: f"{models[stage]['model']}@{models[stage]['revision']}"
            for stage in ("sentiment", "c_summary")
        },
    }


def stale_kinds(texts: list[dict[str, str]], extractor: str) -> dict[str, tuple[str, str]]:
    """For each kind of text in the pool that code (not a pinned model) wrote, ``(kept, now)``
    when its revision is no longer the current one. Summaries are not checked here: their model
    revision is in ``pool_conditions``. Any edit to the modules a revision covers, a comment
    included, makes the kind stale."""
    current = {
        "company_text": retag.parser_revision(),
        "extractor_text": retag.extractor_revision(extractor),
    }
    stale = {}
    for kind, now in current.items():
        kept = {t["revision"] for t in texts if t["kind"] == kind}
        if kept and kept != {now}:
            stale[kind] = (", ".join(sorted(kept)), now)
    return stale


def scored_pool(out: Path, extractor: str) -> list[dict[str, str]]:
    """The texts with the current model's label. Summaries and scoring are slow, so they are kept
    in ``<out>/sentiment_gold_pool.csv`` (news text: git-ignored) with the conditions they were
    made under (``pool_conditions``) and read back on the next run. A run whose conditions differ
    (another extractor, other tagged bodies, other model revisions) stops and names the difference:
    delete both files to score again."""
    path, meta_path = out / POOL_CSV_NAME, out / POOL_META_NAME
    now = pool_conditions(extractor)
    if path.exists():
        before = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        differ = {k: (before.get(k), now[k]) for k in now if before.get(k) != now[k]}
        if differ:
            sys.exit(
                f"{path} was scored under other conditions (kept, now): {differ}. Delete "
                f"{path.name} and {meta_path.name} to score the pool again."
            )
        texts = extra._read(path)
        stale = stale_kinds(texts, extractor)
        if stale:
            sys.exit(
                f"{path} holds texts made by code that has changed since: {stale}. Delete "
                f"{path.name} and {meta_path.name} to score the pool again."
            )
        print(f"{path} exists: the pool is not scored again")
        return texts
    texts, skipped = build_texts(out, extractor)
    texts = predict(texts)
    pool = {
        f"{k}/{p}": n for (k, p), n in Counter((t["kind"], t["predicted"]) for t in texts).items()
    }
    (out / POOL_NAME).write_text(
        json.dumps({"left_out": dict(skipped), "pool": pool}, indent=2), encoding="utf-8"
    )
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(POOL_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(texts)
    meta_path.write_text(json.dumps(now, indent=2), encoding="utf-8")
    return texts


def shortfalls(
    pool: dict[str, int], per_class: int, per_class_extractor: int
) -> dict[str, tuple[int, int]]:
    """Strata (``<part>/<class>``) whose pool is smaller than the number to draw: ``(pool, target)``.
    A short stratum widens the interval D32 quotes (about +/-13 points for 60 texts)."""
    short = {}
    for part, target in (("checked", per_class), ("extractor", per_class_extractor)):
        for label in CLASSES:
            have = pool.get(f"{part}/{label}", 0)
            if have < target:
                short[f"{part}/{label}"] = (have, target)
    return short


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
    db.env.load_env_file()  # .env: these scripts do not import pipeline, which loads it elsewhere
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--per-class-extractor", type=int, default=20)
    parser.add_argument("--extractor", choices=sorted(EXTRACTORS), default="baseline")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--pool-only", action="store_true", help="score the pool and print its sizes; draw nothing"
    )
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    ids_path, target = out / IDS_NAME, out / TO_TAG_NAME
    if extra.has_rows(target):
        sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
    if ids_path.exists():
        chosen = extra._read(ids_path)
        print(f"{ids_path} exists: the set is not drawn again")
    else:
        texts = scored_pool(out, args.extractor)
        pool = Counter(f"{t['kind']}/{t['predicted']}" for t in texts)
        print("pool:", pool)
        short = shortfalls(
            Counter(f"{part_of(t)}/{t['predicted']}" for t in texts),
            args.per_class,
            args.per_class_extractor,
        )
        if short:
            print(
                f"strata with fewer texts than the draw takes (pool, target): {short}. Rule "
                "(owner, D34): every text of a short stratum is drawn, the others are not "
                "changed, and the real counts are in the report."
            )
        if args.pool_only:
            return
        chosen = draw(texts, args.per_class, args.seed, args.per_class_extractor)
        extra._write(ids_path, [{f: t.get(f, "") for f in IDS_FIELDS} for t in chosen])
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
