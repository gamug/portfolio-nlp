# news_nlp — pipeline stage 3: sentiment + NER + categorization + summarization

**Extracted from** `portfolio-data-mining` (which had consolidated it from the standalone `news-nlp` project).

Runs up to five sequential batch stages over the `articles` rows in
`data/nlp.db` (override with `$DATABASE_URL`, see root `.env.example`), each stage owning
its own results table, and exposes everything through a FastAPI service. There is no scraper
here — `articles` is treated as pre-populated input.

Sentiment, NER, and category (stages 1–3) always run. `c_summary`/`sector_summary`
(stages 4–5) are **opt-in** — pass `--summarize` on the CLI or `"summarize": true` in the
API's `/pipeline/run` body; the default (`summarize=False`) skips them entirely, so the
summarization model never loads and its VRAM/latency cost is never paid unless asked for.

1. **Sentiment** — FinBERT (`ProsusAI/finbert`) → `article_sentiment`.
2. **NER** — a fine-tuned SEC-BERT-BASE model trained on FiNER-ORD, published at
   [gamug/sec-bert-finer-ord-ner](https://huggingface.co/gamug/sec-bert-finer-ord-ner) →
   `article_entities`.
3. **Category** — zero-shot NLI classification (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`)
   against a fixed 10-category taxonomy (9 dimensions of company performance + `other`) →
   `article_category`. See [`../category-taxonomy.md`](../category-taxonomy.md) for the
   taxonomy, its sourcing (RavenPack, SASB, IPTC Media Topics, Refinitiv/TRNA), and the
   confidence-threshold mechanism. Classifies on the article's title + lead body chunk
   (not the full article — a per-label NLI pass makes full-article chunking too expensive
   for a mandatory, every-article stage — and not `article_summary`, since that's opt-in
   and this stage isn't).
4. **`c_summary`** — one abstractive summary per article (`sshleifer/distilbart-cnn-12-6`),
   generated from the article's body plus its already-computed sentiment/entities →
   `article_summary`.
5. **`sector_summary`** — one row per `gics_sub_industry` per closed calendar week →
   `sector_summary`. A deterministic, non-generative composition (`db.compose_sector_summary`):
   a stats overview (sentiment %, top entities) plus one section per NLP category (stage 3's
   taxonomy) present that week, each listing its contributing companies' `c_summary` text
   verbatim and attributed to its own ticker — company text is never blended with another
   company's, and never crosses a category-section boundary. The only text ever handed to a
   model (`sshleifer/distilbart-cnn-12-6`) is a short intro sentence built purely from
   aggregate stats (`db.build_sector_intro_seed`) — no ticker, company name, or `c_summary`
   text ever reaches it, which is what makes cross-company/cross-topic blending structurally
   impossible rather than merely unlikely. Articles whose `c_summary` exists but have no
   `article_category` row (historical data predating stage 3 becoming mandatory, or a
   partial/direct stage invocation) are excluded from sector_summary generation entirely.
   Sub-industries with no qualifying articles in a given week get no row; "closed" means the
   week (Mon–Sun) has fully ended, so a summary is never generated from a partial week and
   later need regenerating. `sector_summary.format_version` is bumped whenever this
   generation logic changes shape — a row below the current version is treated as stale and
   self-heals (regenerated via `INSERT OR REPLACE`) the next time the stage runs, no separate
   backfill needed.

- **Sentence-aware chunking** (`src/chunking.py`) — articles run up to ~13K
  words, far past BERT's 512-token limit; chunks are packed on sentence boundaries. The
  category stage reuses it (with a tighter token budget, to leave headroom for the NLI
  hypothesis text) to take just the lead chunk; the summarization stages reuse it for
  BART's 1024-token cap, plus a **hierarchical reduce**
  (`pipeline.hierarchical_summarize()`): summarize each chunk, then if more than one chunk
  resulted, recursively summarize the concatenated chunk-summaries until they collapse
  into a single pass.
- **Idempotent, resumable batch processing** — each stage only processes rows missing
  from its results table (articles for stages 1–4, `(gics_sector, gics_sub_industry,
  week_start)` groups for stage 5, enforced by a `UNIQUE` constraint on `sector_summary`).
- **6GB-VRAM-friendly** — only one model resident on the GPU at a time; each stage loads
  its model, runs to completion, and frees the GPU before the next stage loads (skipped
  entirely when a stage has nothing pending).
- `c_summary` only covers articles that have both a computed sentiment **and** at least
  one entity scoring above 0.8 confidence (`article_entities.score > 0.8`, excluding
  single-character digit noise) — an article with sentiment but no qualifying entities
  never gets summarized. This mirrors the original `query.sql` design this feature is
  based on.

## Setup

```bash
uv sync                          # torch is a pinned direct dep (cu124 wheel index) — no manual install
uv run python -m setup           # pre-fetch the four HF models into the local cache (one-time, safe to re-run)
```

## Running

```bash
# CLI (direct — no server)
uv run cli/news_nlp_cli.py --limit 50
uv run cli/news_nlp_cli.py   # process every pending article, sentiment + NER + category only
uv run cli/news_nlp_cli.py --summarize   # also run c_summary/sector_summary

# API
uv run apps/news_nlp_api.py
# -> http://127.0.0.1:8003/docs
# POST /pipeline/run  {"limit": 50, "summarize": true}
```

`cli/news_nlp_cli.py` wraps `pipeline.run_pipeline()` directly with real
`--limit`/`--summarize` flags, driving sentiment + NER + category (and, with `--summarize`,
`c_summary` + `sector_summary`) in sequence. `src/pipeline.py` also still has its
own bare `if __name__ == "__main__":` (usable via `python -m pipeline [limit]`, a
positional arg instead of a flag, sentiment + NER + category only — no way to opt into
summarization from that entrypoint) — kept for backward compatibility, but
`cli/news_nlp_cli.py` is the documented entrypoint going forward.

Query results via the API: `GET /articles/{id}` now includes `"category"` and `"summary"`
keys (same shape as `"sentiment"`/`"entities"` — `"category"` is `None` until stage 3 has
processed that article, `"summary"` until stage 4 has), `GET /stats/categories` (optional
`company`/`date_from`/`date_to` filters) returns per-label article counts, and
`GET /sectors/summary` (optional `sector`/`sub_industry`/`week_start` filters) lists
`sector_summary` rows, newest week first.

## Database

The DB layer lives in **`portfolio_common.news_nlp`** (git-tag-pinned in
`pyproject.toml`; `src/db.py` / `src/corrections.py` / `src/categories.py` are
thin re-export facades). Two-tier — full detail in
[`docs/db-topology.md`](../db-topology.md):

- **RESULTS store** (`$DATABASE_URL`, unset → `<repo>/data/nlp.db`; working file
  `D:\thesis\data\nlp.db`) — the NLP result tables plus a `body_text`-free
  `articles` subset. Opened read/write as schema `main`.
- **SOURCE store** (`$SOURCE_DATABASE_URL`, **required** for the text stages) —
  `articles` including `body_text` (the legacy crawl DB `urls.db`). ATTACHed
  read-only as schema `source`; never written.

The stages that read article text — sentiment, NER, category, per-article
summary (`fetch_pending_articles` / `fetch_pending_category_articles` /
`fetch_pending_company_summaries`, i.e. `cli/news_nlp_cli.py` and
`POST /pipeline/run`) — read `source.articles`. Every result write also
`INSERT OR IGNORE`s a lean `main.articles` row, so the RESULTS store stays
foreign-key-consistent without a separate migration step. `db.require_source_text`
fails the run fast (before any model loads) if `SOURCE_DATABASE_URL` is unset or
points at a DB with no usable `body_text` — the old failure mode there was a
silent *"0 pending"*.

Against the RESULTS store alone (no SOURCE) the read/query surface works:
`GET /articles/{id}`, `GET /stats/categories`, `GET /sectors/summary`, the
correction (`PATCH`/`DELETE`) endpoints, and the `sector_summary` stage.

The pre-existing results were backfilled once out of `urls.db` (see
[`docs/migration-2026-09-01.md`](../migration-2026-09-01.md)); the one-shot
script that did it has since been removed now that the pipeline populates the
lean `articles` rows itself.

## Testing

```bash
uv run pytest tests/news_nlp -q
```

The suite is hermetic — every model load is monkeypatched, so no torch download, GPU, or
network is needed at test time. All tests pass (part of the repo's CI gate; see
`.github/workflows/ci.yml`).
