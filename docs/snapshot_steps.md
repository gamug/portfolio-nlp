# The snapshot and the period summary (steps 8 and 9; D42, FR-024; owner, 2026-11-14)

Design only: nothing here is built. The tables below are a **proposal** the owner approves in the spec PR (constitution AI #10).
It replaces D28 and T-152 (the earlier sketch, kept in `.specify/memory/archive/SPEC-PLAN-TASKS-condensed-for-the-snapshot-steps-2026-11-14.md`;
a reference to D28 in an older file means D42 now). These steps are **Build 3**: after Build 1 (scope to companies, sector) and Build 2 (sentiment, category) have run on the store.

## What the owner decided (interactive session, 2026-11-14)

| Point | Decision |
|---|---|
| Grain of a snapshot | One row per (text, main company); a `sector` text has one row keyed by its sector. An item with no main company (the review list of step 3) has none. |
| Fields | `date` (the crawl date), `title`, `text`, `main_company` or `sector`, `companies_mentioned` (optional), `category`, `sentiment`. |
| `date` | In the snapshot, the **crawl date** (`articles.fetched_at`). In the period summary, the **report creation date**: the period-end cut. |
| `text` | The step 2 summary (`news_summary`) at the pinned summary revision, or the original body when step 2 stored it as the text (`text_source = original`). The title is `articles.title` (a sub-news takes its article's title). |
| `companies_mentioned` | **Out of scope for now** (owner, 2026-11-14, provisionally; the line is optional in the template). The column exists and stays empty; no floor is needed and nothing refuses to run. When it returns, the design is in "The mentions floor" (T-198). |
| Sector texts | **No category and no sentiment** (owner, 2026-11-14). The category step writes only (text, main company) pairs, and the pair model needs a company. This **supersedes D39 (iv)** (a sector summary gets a sentiment from `gamug/FinBERT-financial-news`) and the clause of the old FR-020 that said so (text in the archive file above). A sector snapshot is title, text, sector and date. **It is not counted in the period summary** (owner, 2026-11-14); it stays as a row. |
| Complete rows only | A company snapshot is written only when its sentiment and category both exist. Otherwise the item waits for the next daily run; `--status` lists the waiting items with the reason. A sector snapshot needs the sector (`body_sector`, `status = assigned`) and the text. |
| When | Snapshots daily, incremental, after the 15:00 batch of `portfolio-data-mining`. The period summary only on a period-end day, after that day's snapshots. |
| Period | 21 days. Period ends: 2022-01-03 + 21k days, each at **15:00 in `BATCH_TIMEZONE`** (default `America/New_York`). A window is `(previous cut, this cut]` by `fetched_at`; the first has no lower bound. |
| What closes a window | **A fixed clock time**, 15:00 New York (owner, 2026-11-14): the batch of that day is expected to have finished crawling before it. An item crawled after the cut belongs to the next window even if it came from that day's batch; the end time of the crawl is not read. |
| Company block | Only companies with news in the window; only categories with news; per category: total and the sentiment split. Companies by total, descending. |
| Sector block (owner, 2026-11-14, instead of an LLM paragraph) | The **same numbers as a company block** summed over the companies of the sector: per category `total`, `neutral`, `positive`, `negative`, **one count per snapshot** (each (text, company) counts, so a sector line equals the sum of its companies' lines). No LLM and no tokens. Only sectors with news; a sector with none is dropped. A company's sector: its `company_sector` row if any, else its universe GICS sector (read through the `sector` table), else the line `unclassified`. |
| Late items | A written period is **frozen**. An item whose crawl date is in a closed window but that reached a snapshot after the period was written is counted in the next period, keeps its true `fetched_at`, and is counted as late there (see "Which period counted what"). |
| Backfill | `--catch-up` runs every missing period in order, each at its own cut; a period already written is skipped; the number of periods to do is printed first. |

## The time rule

`BATCH_TIMEZONE` (`.env`, an IANA name; `.env.example` lists the American options) defaults to `America/New_York`. A period end is a calendar
date at 15:00 wall time in that zone, converted to UTC (20:00 in winter, 19:00 in summer); a window is always 21 calendar days, which is 503 or 505
hours across a clock change. `fetched_at` is UTC (`...Z`). An invalid zone name stops the run. 2022-01-03 is a Monday and every cut falls on one.
News crawled between 15:00 and the close (16:00) goes to the next window.

## The mentions floor (kept for when `companies_mentioned` returns; T-198)

In step 3 a ticker is main exactly when its relevance is at or over the run's `--threshold` (a required argument of `--step companies`, part of its revision, not a
config value). Every non-main row is therefore **under** that threshold, so the floor of mentions is a **lower, separate value**: `mentions_floor` in
`config/snapshot_step.json`. **Its value is not set** and nothing reads it yet: the store of this repository is empty, the data is the owner's. When the field returns, the snapshot step will **refuse to run while it is unset**. To choose it, run read-only on the real RESULTS store (never write to it) and give the distribution of the non-main scores at the companies revision:

```
SELECT COUNT(*), MIN(relevance_score), MAX(relevance_score) FROM news_company WHERE model_revision = 'companies-...' AND is_main = 0;
SELECT ROUND(relevance_score, 1) AS bucket, COUNT(*) FROM news_company WHERE model_revision = 'companies-...' AND is_main = 0 GROUP BY bucket ORDER BY bucket;
```

The owner picks the floor from it (T-198); the value goes in the config with the date and the revision it was read at.

## Tables (proposal, RESULTS store)

`news_snapshot`, primary key (`item_kind`, `item_id`, `ticker`, `model_revision`); `ticker` is `''` for a sector text, as in `news_category`:

| Column | Meaning |
|---|---|
| `fetched_at` | crawl time, UTC; places the item in a window |
| `pub_date` | publication date as stored (from the earlier D28); not used for the window |
| `title`, `text`, `text_source` | as above |
| `ticker`, `sector` | the main company, or `''` and the 2-digit GICS sector of `body_sector` |
| `companies_mentioned` | JSON list of tickers; empty for now |
| `category`, `group_label` | from `news_category`; `''` for a sector text |
| `sentiment`, `sentiment_score` | from `news_sentiment`; `''` and NULL for a sector text |
| `body_hash` | one count per unique body downstream (from the earlier D28) |
| `processed_at` | when the snapshot was written |

`period_summary`, primary key (`period_end`, `kind`, `key`, `model_revision`); `kind` is `company` or `sector`, `key` the ticker or the sector code (`''` for `unclassified`):

| Column | Meaning |
|---|---|
| `period_start`, `period_end` | the window, UTC |
| `counts` | JSON, per category `total`, `neutral`, `positive`, `negative`; the same shape for a company row and a sector row |
| `late_count` | how many of the rows counted here are late (derived from `period_item`) |
| `created_at` | the report creation date |

`period_item`, primary key (`period_end`, `model_revision`, `item_kind`, `item_id`, `ticker`): one row for each snapshot a period counted; `late` is 1 when the
snapshot's `fetched_at` is at or before the period's `period_start`.

`sector` (reference, the one table of sectors): `code` (2 digits, PK), `name`, `kind` (`gics` or `extra`), `descriptions` (JSON, the texts the sector step embeds). It is **seeded**
(`INSERT OR IGNORE`) from `sector_stage.SECTOR_CODES` and `SECTOR_DESCRIPTIONS`, which stay the source the sector step reads (a test pins the table to them); an `extra` sector
(one GICS does not have, such as cryptocurrency) is a row the owner adds, with a code from 90 up so it never clashes with the GICS codes (10 to 60). **None now** (owner, 2026-11-14): the 507 tickers of `gold/universe.csv` all have a GICS sector, and the only crypto names (COIN, HOOD, XYZ) sit in Financials. Making the sector step read this table is another revision of that step and is not part of this PR.
`company_sector`: `ticker` (PK), `sector_code`: an override of the universe's GICS sector; empty until the owner gives the extra sectors and their tickers.

## Which period counted what

A period run counts every snapshot **not yet in a `period_item` row of an earlier period** whose `fetched_at` is at or before its own `period_end`. Those with
`fetched_at` at or before the period's `period_start` are late (their own window was already closed and written). This needs no clock comparison: the
rows say which period counted which snapshot, so the counts can be audited and a late item can be told from an on-time one. A written period is never rewritten.

## The revisions

The snapshot's revision is a hash of the five step revisions it reads (summary, companies, sector, category, sentiment), of the **cut revision** (`units5-...`: the text and the
title of a sub-news depend on it), and of the code that builds the row (and, once the field returns, the mentions floor); a run names each and is refused when one has no rows. The period summary's
revision is a hash of the snapshot revision, the counting code and the content of `sector` and `company_sector`. Both steps are idempotent and resumable; state is in
`pipeline_step_state`; `--status` follows the other steps.

## Commands (proposed)

```
uv run cli/news_nlp_steps.py --step snapshot --summary-revision ... --companies-revision ... --cut-revision ... --sector-revision ... --category-revision ... --sentiment-revision ... --limit 500
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --period-end 2022-01-03      # one period
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --catch-up                   # every missing period
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --period-end 2022-01-03 --render
```

`--render` prints the text layout the owner described a `{main_company}` block with one line per category, and a `{sector}` block with the same lines.

## Not decided here

* `companies_mentioned` and its floor (T-198): out of scope for now.
* Which revisions of the five steps the first real run pins: it needs the real runs of sentiment and category on the store (T-193, T-192).
* Whether a company text with no sentiment because of a skip is retried by hand or left (the list in `--status` is what the owner reads).
* Extra sectors (T-199): none for now; only the 11 GICS sectors exist.
