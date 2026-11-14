# The snapshot and the period summary (steps 8 and 9; D42, FR-024; owner, 2026-11-14)

Design only: nothing here is built. The tables below are a **proposal** the owner approves in the spec PR (constitution AI #10).
It replaces D28 and T-152 (the earlier sketch, kept in `.specify/memory/archive/SPEC-PLAN-TASKS-condensed-for-the-snapshot-steps-2026-11-14.md`).

## What the owner decided (interactive session, 2026-11-14)

| Point | Decision |
|---|---|
| Grain of a snapshot | One row per (text, main company); a `sector` text has one row keyed by its sector. An item with no main company (the review list of step 3) has none. |
| Fields | `date` (the crawl date), `title`, `text`, `main_company` or `sector`, `companies_mentioned` (optional), `category`, `sentiment`. |
| `date` | In the snapshot, the **crawl date** (`articles.fetched_at`). In the period summary, the **report creation date**: the period-end cut. |
| `text` | The step 2 summary (`news_summary`) at the pinned summary revision; the original body when step 2 stored it as the text (`text_source = original`). The title is `articles.title` (a sub-news takes its article's title). |
| `companies_mentioned` | The non-main tickers step 3 stored for the text with a relevance at or over **step 3's main threshold** (read from step 3's configuration, not copied). |
| Sector texts | Category yes, **sentiment omitted** (the pair model needs a company; D39 (iv) stays open). |
| Complete rows only | A snapshot is written only when the sentiment (company items) and the category both exist. Otherwise the item waits for the next daily run; `--status` lists the waiting items with the reason. |
| When | Snapshots daily, incremental, after the 15:00 batch of `portfolio-data-mining` ends. The period summary only on a period-end day, after that day's snapshots. |
| Period | 21 days. Period ends: 2022-01-03 + 21k days, each at **15:00 in `BATCH_TIMEZONE`** (default `America/New_York`). A window is `(previous cut, this cut]` by `fetched_at`; the first has no lower bound. |
| Company block | Only companies with news in the window; only categories with news; per category: total and the sentiment split. Companies by total, descending. |
| Sector block | Only sectors with news. One English paragraph (3 to 4 sentences) written by the LLM of `.env` from the sector's snapshots of the window (map-reduce when they exceed the context). A sector with no news is dropped. |
| Late items | A written period is **frozen**. An item whose crawl date is in a closed window but that reached a snapshot after the period was written is counted in the next period, keeps its true `fetched_at`, and is flagged `late`; the next period says how many it carries. |
| Backfill | `--catch-up` runs every missing period in order, each at its own cut; a period already written is skipped; the number of LLM calls is printed before it starts. |
| LLM | Temperature 0. A paragraph stores the model name, the prompt hash and the ids of the snapshots it was built from. A written paragraph is never regenerated; a new model or prompt is a new revision. |

## The time rule

`BATCH_TIMEZONE` (`.env`, an IANA name; `.env.example` lists the American options) defaults to `America/New_York`. A period end is a calendar
date at 15:00 wall time in that zone, converted to UTC (20:00 in winter, 19:00 in summer); a window is always 21 calendar days, which is 503 or 505
hours across a clock change. `fetched_at` is UTC (`...Z`). An invalid zone name stops the run. 2022-01-03 is a Monday and every cut falls on one.
News crawled between 15:00 and the close (16:00) goes to the next window.

## Tables (proposal, RESULTS store)

`news_snapshot`, primary key (`item_kind`, `item_id`, `ticker`, `model_revision`); `ticker` is `''` for a sector text, as in `news_category`:

| Column | Meaning |
|---|---|
| `fetched_at` | crawl time, UTC; places the item in a window |
| `pub_date` | publication date as stored (D28); not used for the window |
| `title`, `text`, `text_source` | as above |
| `ticker`, `sector` | the main company, or `''` and the 2-digit GICS sector of `body_sector` |
| `companies_mentioned` | JSON list of tickers |
| `category`, `group_label` | from `news_category` |
| `sentiment`, `sentiment_score` | from `news_sentiment`; `''` and NULL for a sector text |
| `body_hash` | one count per unique body downstream (D28) |
| `processed_at` | when the snapshot was written |

`period_summary`, primary key (`period_end`, `kind`, `key`, `model_revision`); `kind` is `company` or `sector`, `key` the ticker or the sector code:

| Column | Meaning |
|---|---|
| `period_start`, `period_end` | the window, UTC |
| `counts` | JSON, per category: `total`, `neutral`, `positive`, `negative` (company rows) |
| `paragraph`, `llm_model`, `prompt_hash`, `snapshot_ids` | the sector paragraph and what produced it (sector rows) |
| `late_count` | the late items counted in this period |
| `created_at` | the report creation date |

## The revisions

The snapshot's revision is a hash of the five revisions it reads (summary, companies, sector, category, sentiment) and of the code that builds the row; a run names
each and is refused when one has no rows. The period summary's revision is a hash of the snapshot revision, the counting code, the LLM model name and the prompt
hash. Both steps are idempotent and resumable; state is in `pipeline_step_state`; `--status` follows the other steps.

## Commands (proposed)

```
uv run cli/news_nlp_steps.py --step snapshot --summary-revision ... --companies-revision ... --sector-revision ... --category-revision ... --sentiment-revision ... --limit 500
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --period-end 2022-01-03      # one period
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --catch-up                   # every missing period
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --period-end 2022-01-03 --render
```

`--render` prints the text layout the owner described (a `{main_company}` block with one line per category, `{sector}` and its paragraph).

## Not decided here

* Which revisions of the five steps the first real run pins: it needs the real runs of sentiment and category on the store (T-193, T-192).
* Whether a company text with no sentiment because of a skip is retried by hand or left (the list in `--status` is what the owner reads).
* The LLM model is whatever `.env` names; its quality on the sector paragraph is unmeasured (the testing task).
