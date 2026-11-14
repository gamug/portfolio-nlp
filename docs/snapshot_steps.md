# The snapshot and the period summary (steps 8 and 9; D42, FR-024; owner, 2026-11-14)

**Status (2026-11-14):** the snapshot step (step 8, T-195) is **built**: `src/news_snapshot_stage.py`, `--step snapshot`, the table `news_snapshot`, `tests/news_nlp/test_snapshot_step.py`. The period summary (step 9, T-196) is **built too**: `src/news_period_stage.py`, `--step period`, the tables `period_run`, `period_summary`, `period_item`, `sector` and `company_sector`, `tests/news_nlp/test_period_step.py`. Not run on the real store.
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
| Sector texts | **No category and no sentiment** (owner, 2026-11-14). The category step writes only (text, main company) pairs, and the pair model needs a company. This **supersedes D39 (iv)** (a sector summary gets a sentiment from `gamug/FinBERT-financial-news`) and the clause of the old FR-020 that said so (text in the archive file above). A sector snapshot is title, text, sector and date. **It is not counted in the period summary** (owner, 2026-11-14); it stays as a stored row, a record with **no consumer for now** (owner, 2026-11-14): no step reads it. |
| Complete rows only | A company snapshot is written only when its sentiment and category both exist. Otherwise the **pair** waits for the next daily run; `--status` lists the waiting pairs with the reason (a text with one company done and another waiting is handled per pair). A sector snapshot needs the sector (`body_sector`, `status = assigned`) and the text. |
| When | **One run per day at 15:00** (owner, 2026-11-14): `portfolio-data-mining`, then every step of this repository in order (1 to 7, then 8 snapshots, then 9 on a period-end day). A day's crawl is therefore snapshotted by the next day's run; the items crawled on a cut day before the cut, and any not yet complete, are counted in the next period as late. That is the normal landing of the last hours of a window, not a fault. |
| Period | 21 days. Period ends: 2022-01-03 + 21k days, each at **15:00 in `BATCH_TIMEZONE`** (default `America/New_York`). A window is `(previous cut, this cut]` by `fetched_at`; the first has no lower bound. |
| What closes a window | **A fixed clock time**, 15:00 New York (owner, 2026-11-14): the batch of that day is expected to have finished crawling before it. An item crawled after the cut belongs to the next window even if it came from that day's batch; the end time of the crawl is not read. |
| Company block | Only companies with news in the window; only categories with news; per category: total and the sentiment split. Companies by total, descending. |
| Sector block (owner, 2026-11-14, instead of an LLM paragraph) | The **same numbers as a company block** summed over the companies of the sector: per category `total`, `neutral`, `positive`, `negative`, **one count per snapshot** (owner, 2026-11-14): each (text, company) counts, so a sector line equals the sum of its companies' lines, and a body cut into several sub-news about one company counts each (the old one-count-per-body rule of D28 is dropped). No LLM and no tokens. Only sectors with news; a sector with none is dropped. A company's sector: its `company_sector` row if any, else its universe GICS sector (read through the `sector` table), else the line `unclassified`. |
| Late items | A written period is **frozen**. An item whose crawl date is in a closed window but that reached a snapshot after the period was written is counted in the next period, keeps its true `fetched_at`, and is counted as late there (see "Which period counted what"). |
| Backfill | `--catch-up` runs every missing period in order, each at its own cut; a period already written is skipped; the number of periods to do is printed first. |

## The time rule

`BATCH_TIMEZONE` (`.env`, an IANA name; `.env.example` lists the American options) defaults to `America/New_York`. A period end is a calendar
date at 15:00 wall time in that zone, converted to UTC (20:00 in winter, 19:00 in summer); a window is always 21 calendar days, which is 503 or 505
hours across a clock change. `fetched_at` is UTC (`...Z`): the crawl stores `2026-08-12T22:19:20.655964+00:00`, which the step converts to UTC whole seconds (a date with no zone waits). An invalid zone name stops the run. 2022-01-03 is a Monday and every cut falls on one.
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

## Tables (RESULTS store; all built)

`news_snapshot`, primary key (`item_kind`, `item_id`, `ticker`, `model_revision`); `ticker` is `''` for a sector text, as in `news_category`:

| Column | Meaning |
|---|---|
| `fetched_at` | crawl time, UTC; places the item in a window. A unique body can come from several articles: the **earliest** `fetched_at` of its articles (when the news first appeared); the title is the representative article's (the lowest id) |
| `pub_date` | publication date as stored (from the earlier D28); not used for the window |
| `title`, `text`, `text_source` | as above |
| `ticker`, `sector` | the main company, or `''` and the 2-digit GICS sector of `body_sector` |
| `companies_mentioned` | JSON list of tickers; empty for now |
| `category`, `group_label` | from `news_category`; `''` for a sector text |
| `sentiment`, `sentiment_score` | from `news_sentiment`; `''` and NULL for a sector text |
| `inputs` | JSON: the summary, companies, cut, category, sentiment and (sector rows) sector revisions the row was built from (not in the revision; `--status` reports the rows built from other revisions than the ones named) |
| `processed_at` | when the snapshot was written |

`period_summary`, primary key (`period_end`, `kind`, `key`, `model_revision`); `kind` is `company` or `sector`, `key` the ticker or the sector code (`''` for `unclassified`):

| Column | Meaning |
|---|---|
| `period_end` | the cut, UTC (the window is in `period_run`) |
| `counts` | JSON, per category `total`, `neutral`, `positive`, `negative`; the same shape for a company row and a sector row |
| `not_counted` | main (text, company) pairs (of the companies revision of the latest snapshot written) crawled in the window that had no snapshot when the period was written (a sentiment skipped, a category missing); a statement as of that run, not rewritten and not a debt: a pair that later gets its snapshot is counted as late in a later period, and is then in that period's counts. A sector row's value is the sum of its companies'. Only on rows that exist: **no row is created for a company or sector with no counted news** (owner, 2026-11-14); its gaps show in `--status` |

`period_run`, primary key (`period_end`, `model_revision`): `snapshot_revision` (the snapshots it counted: a reset of those snapshots clears this period), `period_start` (UTC; NULL for the first window, no lower bound, and nothing in it is late), `created_at` (when the row was written; **the report date is `period_end`**, the cut, as the owner decided). **A period is written when its `period_run` row exists**, inserted in the same transaction as its `period_summary` and `period_item` rows. A window with no counted news has a `period_run` row and no other, so it is written and does not block the later periods.

The number of late rows of a period is **computed** from `period_item` (no stored copy, so it cannot drift).

`period_item`, primary key (`period_end`, `model_revision`, `item_kind`, `item_id`, `ticker`): one row for each snapshot a period counted; `late` is 1 when the
snapshot's `fetched_at` is at or before the period's `period_start` (never, when it is NULL).

`sector` (reference, the one table of sectors): `code` (2 digits, PK), `name`, `kind` (`gics` or `extra`), `descriptions` (JSON, the texts the sector step embeds). It is **seeded**
(`INSERT OR IGNORE`) from `sector_stage.SECTOR_CODES` and `SECTOR_DESCRIPTIONS`, which stay the source the sector step reads (a test pins the table to them); an `extra` sector
(one GICS does not have, such as cryptocurrency) is a row the owner adds, with a code from 90 up so it never clashes with the GICS codes (10 to 60). **None now** (owner, 2026-11-14): the 507 tickers of `gold/universe.csv` all have a GICS sector, and the only crypto names (COIN, HOOD, XYZ) sit in Financials. Making the sector step read this table is another revision of that step and is not part of this PR.
`company_sector`: `ticker` (PK), `sector_code`: an override of the universe's GICS sector; empty until the owner gives the extra sectors and their tickers.

## Which period counted what

Within one period revision, a period run counts every snapshot **not yet in a `period_item` row of an earlier period of that revision** whose `fetched_at` is at or before its own `period_end`. Those with
`fetched_at` at or before the period's `period_start` are late (their own window was already closed and written). This needs no clock comparison: the
rows say which period counted which snapshot, so the counts can be audited and a late item can be told from an on-time one. A written period is never rewritten by the step; to write one again the owner deletes **it and every later one** of the period revision (`REPERIOD_SQL` in `--help`: a later period holds the snapshots it took from the earlier windows), and `--catch-up` writes them again. On a period-end day the step loads every company snapshot of the revision (and the keys already counted) into memory, whatever the number of periods to do (a known limit, as in the snapshot step). A new period revision (another snapshot revision, counting code or sector content) starts clean: `--catch-up` then runs from the first period again, each at its own cut, and the earlier revision's rows are kept.

**Order.** A period is refused unless every earlier period of the same revision is written (a gap would let a later period count the earlier window's items as late, and the earlier one, written afterwards, would find them taken). `--period-end` must be on the grid (2022-01-03 + 21k days) and its cut already passed; otherwise it is refused. `--catch-up` runs from the first missing period to the last cut already passed.

## The revisions

**The snapshot step is always idempotent** (owner, 2026-11-14). Its revision (`snapshot-...`) is a hash of **the code that builds a row only** (the columns, the join of the earlier
steps, the reading of the corpus, the choice of the pairs, the sector rows; not the limit or the counts). The revisions of the steps it read (summary, companies, cut, category, sentiment, and
sector for sector texts) are **recorded in each row** (`inputs`, JSON), not hashed. A pair that has a row is never read again, whatever revisions it came from, unless `--reset` deletes the rows
of the revision first. **The daily flow:** the owner names the revisions of the day (any of them can move: the sentiment revision moves the day a new company becomes main, because it hashes the
names of the main companies); the step writes only the pairs with no row, so a moved revision **never restarts the snapshots or the periods built on them**. The cost of that rule: a row keeps the
label of the revision that wrote it (`--status` prints `stale_rows`: how many rows were built from other revisions than the ones named, and how many are no longer a main pair); to rebuild after a rerun of a step at the same revision, or to use the new labels, run with `--reset` (a full run: `--reset` with `--limit` is refused; the delete and the rewrite are one transaction, so a rebuild that fails or is interrupted leaves the old rows; the company rows; the sector rows only when `--sector-revision` is named, since only then are they written again; `--reset` with `--status` is refused) (**owner, 2026-11-14: a reset of the snapshots resets the period summary built on them too**: T-196 stores the snapshot revision in `period_run` and `--reset` clears the `period_summary`, `period_run` and `period_item` rows of those periods in the same transaction; the periods are then written again by `--catch-up`). A change of the code is a new revision, and with it new snapshots and new periods.

**What a run checks** (`check_inputs`): the step is refused when a revision has no rows, when a category or sentiment row is not a main pair of the companies revision, when a sub-news with a main
company is not of the named cut, and when an item with a main company has no text at the summary revision (as the sentiment step does: usually the companies step read another summary revision).
A revision is a hash, so the inputs it read cannot be read back: **a sentiment or category run made on another summary revision is not detected** (name the revisions that ran together).
Each run reads the whole corpus **once** (company rows and sector rows together), and loads the summary, category and sentiment rows of the revisions into memory whatever `--limit` is (a known
limit, as in the sentiment step). `--limit` counts the company pairs and, with `--sector-revision`, the sector texts **each** (so it can write twice that). A pair whose article is not in the
corpus, or has no crawl date with a zone, waits with that reason and does not take a place of `--limit`. The rerun SQL is `RESNAPSHOT_SQL` in `--help` (what `--reset` does); a recut also
deletes the snapshots of its sub-news (`RECUT_SQL`).

The period summary's
revision is a hash of the snapshot revision, the counting code (**`_tally`, `_counts_of`, `_load_rows`, `_write`, `_gaps` of `news_period_stage`, and `articles_of`, `crawl_date_ok`, `_body_hashes` of `news_snapshot_stage`**: editing one of them is a new revision and `--catch-up` starts from period 1 again; `run`, the bookkeeping, is not hashed) and the **resolved mapping (ticker, sector code) of every company of the universe** (`company_sector` first, else its GICS sector, through `SECTOR_CODES`; of the universe, not of the companies that have a snapshot: a company gaining its first snapshot must not restart the periods), so a change of the universe's sectors is a new revision and a written period stays reproducible. Both steps are idempotent and resumable. **State is derived from the tables, not kept in `pipeline_step_state`** (owner, 2026-11-14): a (text, company) pair is pending when it has no `news_snapshot` row (so a text with one company complete and another waiting is handled per pair), and a period is pending when it has no `period_run` row for its `period_end` and revision; `--status` computes the waiting pairs and their reason, and lists the periods done and missing.

## Commands (both built)

```
uv run cli/news_nlp_steps.py --step snapshot --summary-revision ... --companies-revision ... --cut-revision ... [--sector-revision ...] --category-revision ... --sentiment-revision ... --limit 500   # [--reset] rebuilds
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --period-end 2022-01-03      # one period
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --catch-up                   # every missing period
uv run cli/news_nlp_steps.py --step period --snapshot-revision ... --period-end 2022-01-03 --render
```

**Progress.** `--step snapshot` prints a line to stderr before the corpus read (the slow part: one pass over the corpus, no count, so no bar) and then one bar with the time left over all the rows it will write (company pairs and sector texts together); stdout stays the JSON. Rows are committed every 500, so a run stopped half way keeps what the bar had counted and the next run (always idempotent) writes the rest. `--step period` prints the number of periods to do **before** it writes the first, then one `tqdm` bar with the time left over the snapshot rows it will count (the unit is the row, not the period: the first window and a first catch-up carry most of the rows, so a bar of periods would give a time left that jumps). Bars are for a terminal: a run that writes to a file (the 15:00 batch) gets them redrawn with carriage returns, and the JSON printed at the end is the record.

`--render` (and the period part of `--status`) reads **written periods only** and refuses a period with no `period_run` row. `--render` prints `not_counted` on a line only when it is above 0, and the text layout the owner described a `{main_company}` block with one line per category, and a `{sector}` block with the same lines.

## Not decided here

* `companies_mentioned` and its floor (T-198): out of scope for now.
* Which revisions of the steps the first real run pins: it needs sentiment and category run on the real store (the owner's runs; T-192 and T-193 only validate them, they are not run tasks).
* Whether a company text with no sentiment because of a skip is retried by hand or left (the list in `--status` is what the owner reads).
* Extra sectors (T-199): none for now; only the 11 GICS sectors exist.
