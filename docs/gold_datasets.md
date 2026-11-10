# Gold datasets (Work item 18)

**Status: written 2026-11-06.** One page for every set the owner tags or checks, what it is for, where
each file lives and in which order the sets unlock the work. If a name, a size or a date here
disagrees with `PLAN.md` (D20–D34) or `TASKS.md`, this page is the one to fix.

## How a set is stored

Every set has two files with the same stem and different jobs:

| | `data/` (the working file) | `gold/` (the final file) |
|---|---|---|
| In git | **no** (git-ignored: it holds news text, the repository is public) | **yes** |
| Holds | the article text, its Spanish reading aid (`spanish_*`), the guidance columns and the columns to tag | the **tags only**, plus the ids to find the text again in the SOURCE store |
| Who writes it | a `scripts/draw_*` script (drawn once, never overwritten once it has rows); the owner tags in it | a `scripts/build_gold_*` script, from the finished `data/` file, after the checks pass |
| Used by | the owner, `scripts/report_wi18_2026_11_06.py` | every training, evaluation and pipeline feature |

The rule: **tag in `data/`, migrate to `gold/` when the set is finished, and use only `gold/` from
then on.** For that to hold, each `gold/` file carries what the figures need besides the marks: the
`stratum` and `frame` of the held-out rows (`frame` is the stratum's size in the whole sample, D30; it
is **not** the `population` column of `heldout_ids.csv`, which counts the bodies left to draw from), and the `stratum`, `extractor`,
`predicted` and `population` of the sentiment rows (D32). Until a set is migrated its figures are
computed from `data/`, and the draw's record is **required**, not optional:

| set | the draw's record in `data/` (keep it until migrated) |
|---|---|
| 2 scope test | `heldout_ids.csv` (stratum per body) and `heldout_strata.csv` (`frame` sizes) |
| 3 label check | `segment_check_ids.csv` (the gold keeps its `year`, so the 2022+ figure of D27 survives: `report_wi18 label-check --gold --out gold` reproduces the figures from the tracked file, and a test pins them) |
| 4 extractor sample | `faithfulness_bodies.csv` (the bodies drawn) **and** `faithfulness_ids.csv` (which extractor wrote each `review_order`; without it the report cannot be computed) |
| 5 sentiment | `sentiment_gold_ids.csv` (stratum, prediction, `population`, extractor, the text) **and** `sentiment_gold_pool.csv` with `sentiment_gold_pool_meta.json` (the scored pool the stratum sizes come from) |

## What may sit in `gold/`

Only files with no news text: the owner's tags or marks, the ids of a draw and the weights they need,
or a training set of ids and positions (`company_segments_train.csv`; and the owner's marks on the sentiment training summaries, `sentiment_train.csv`, set 6: not a gold, never mixed with set 5; written by its builder, so not scaffolded empty). In the order of the repository: the scope
train and test files (#123), the three files of sets 3, 4 and 5 (added here at the owner's request of
2026-11-06, one gold file per described set) and `company_segments_train.csv`. The earlier call that
`gold/` holds only the train and the test file (2026-11-05) is superseded by this one. **Amended by the owner on 2026-11-10: the repository and its data are public, and the texts behind the marks are tracked with them in `gold/texts/`** (`scripts/export_gold_texts_2026_11_10.py`, one file per set keyed by `key`, `text_hash` checked: sentiment gold 180, sentiment training 450 and 405, extractor faithfulness 453 and its confirmation set 438; the article bodies of the scope sets are not copied, they are in SOURCE under `article_id`). A cache or a translation still stays in `data/`. The one file there that is neither a tag nor a draw is `sentiment_gold_scoring_log.csv` (T-148c2, D35): one row per run of a candidate model on set 5, with its figures, no text; its columns are `retag.SENTIMENT_SCORING_LOG_COLUMNS` and the test of the committed gold files checks them.

## Empty files of the sets without a draw

`uv run scripts/scaffold_gold_sets_2026_11_06.py` writes, for sets 3, 4 and 5, a header-only working
file in `data/` (the exact columns the draw writes) and a header-only tracked file in `gold/`
(`retag.*_GOLD_COLUMNS`); a file that exists is never touched. `data/` is git-ignored, so a clean
checkout has none until this is run (or the draw is, which creates the file itself). The label-check,
extractor and sentiment draws read a header-only working file as not drawn yet and fill it; a file with
rows is the owner's work and is refused. The scope draws do not use scaffolds: they are drawn. A test
pins the `gold/` headers to the column constants in `retag`. Today only sets 4 and 5 are still
scaffolds in `gold/`: set 3's file was built on 2026-11-06 (200 rows) and the scaffold script never
overwrites a file, with rows or without.

## The sets

| # | Set | Size | `data/` working file | `gold/` file | State on 2026-11-07 |
|---|---|---|---|---|---|
| 1 | **Scope train** (the 240 of the prompt gate + 44 sector bodies) | 284 bodies | `gate240_to_tag.csv`, `sector_extra_to_tag.csv` | `category_gold_v2.csv` | tagged and migrated |
| 2 | **Scope test** (held-out) | 200 bodies | `heldout_to_tag.csv` | `category_gold_v2_test.csv` | drawn; **tagged by the owner and migrated 2026-11-07** (200 of 200 valid: 118 single, 47 multi, 11 market_economy, 9 sector, 8 unrelated, 7 investing_general); the `gold/` file carries the tags with the draw's weights |
| 3 | **Label check** (step-4 training labels) | 200 paragraphs | `segment_check_to_tag.csv` | `segment_check_gold.csv` | drawn and marked 2026-11-06: `lead` 88.0% right, `continuation` 38.0%, so the model trains on `lead` only (D27); `gold/` file built 2026-11-06 (200 rows, `build_segment_check_gold_2026_11_06.py`) |
| 4 | **Extractor sample** (faithfulness) | 30 bodies, 453 texts (190 of 2026-11-06 + later extractors); confirmation set 30 bodies, 438 texts | `faithfulness_to_tag.csv` | `faithfulness_gold.csv` | drawn, marked and migrated 2026-11-06: 190 texts of 30 bodies, 115 faithful and 75 not; the baseline fails the D32 bar (60.5%, interval 45.2 to 73.8); `gold/` file built (190 rows, `build_faithfulness_gold_2026_11_06.py`); **2026-11-10: the grown file (453 rows: baseline, `model`, `units`, `units2`, `units3`) and the confirmation set (`gold/faithfulness_confirmation/`, 438 rows) migrated with the same builder, no earlier mark changed** |
| 5 | **Sentiment** | 180 texts | `sentiment_gold_to_tag.csv` | `sentiment_gold.csv` | drawn, marked and migrated 2026-11-06: 82 positive, 65 negative, 33 neutral; overall agreement 60.4% (interval 52.8 to 67.6), under the D32 bar of 80%; negative precision 78.7% (67.3 to 89.3) and recall 77.7% (69.4 to 86.8), not decided against 70%; `gold/` file built (180 rows, `build_sentiment_gold_2026_11_06.py`) |
| 6 | **Sentiment training set** (not a gold: it trains, never exams) | 450 summaries + 405 (second file) | `sentiment_train_to_tag.csv`, `sentiment_eval_to_tag.csv` | `sentiment_train.csv`, `sentiment_train_2.csv` | drawn and translated 2026-11-07 (`draw_sentiment_train_2026_11_07.py`); the owner tagged all 450 and the marks were migrated 2026-11-08 (`build_sentiment_train_2026_11_07.py`, T-148c1: 360 train, 90 validation); the trainer is T-148c3; the second file (405 rows, batch "2", 270 train and 135 validation, drawn by `draw_sentiment_train_2_2026_11_10.py`, tagged by the owner and migrated 2026-11-10 by `migrate_sentiment_train_2_to_gold_2026_11_10.py`, T-148c3b) |

The state column is the state on the day it was written; the files themselves are the record. Counts of
tagged rows are not kept here.

Not a gold set, listed so nobody mistakes it for one: `gold/company_segments_train.csv` is the
**training set** of the step-4 attribution model (33,624 paragraphs, automatic labels, no owner tags;
`scripts/build_company_segments_2026_11_05.py`). Set 3 is the check of its labels.

### 1. Scope train — `gold/category_gold_v2.csv`

- **What:** the owner's tags of the 240 bodies fixed for the prompt gate (`session=gate240`, stratified
  on the v1 scope) and 44 sector bodies (`session=sector_extra`).
- **Scope:** one row per unique body: `scope` (six scopes), `company_labels` (the analysed companies with
  their D15 label), `mentions`, `sub_industry`, `notes`. Rules: `labelling-spec-v2.md`.
- **Used for:** writing and tuning the labelling prompt (the gate, D18). **Never used to measure** the
  scope (it tuned the prompt). It is also the pool
  from which sets 4 and 5 are drawn (the bodies tagged `multi`, and the tagged scopes of the summaries).
- **Working files:** `gate240_to_tag.csv` and `sector_extra_to_tag.csv` (columns: `retag.TAGGING_COLUMNS`).
  Migrated by `scripts/build_gold_2026_11_05.py --set train`.

### 2. Scope test — `gold/category_gold_v2_test.csv`

- **What:** the 200 held-out bodies (D30): all 39 the rules call no-company, sector, market or economy,
  the rest a seeded random draw from the 1,000-body corpus sample, none sharing a body with the 720
  or the 44. Drawn once; the set is fixed.
- **Scope:** the same tag columns as set 1, `session=heldout`, plus `stratum` and `frame` (the weights of D30). In `gold/` the tag columns hold the owner's tags (migrated 2026-11-07).
- **Used for:** the **only** measure of the scope: the rules' and later the classifier's agreement with
  the owner (>= 90% corpus-weighted, D20/D33) and the D24 category target. Never trained on (Q3).
  `data/heldout_strata.csv` holds the stratum sizes (`frame`) that re-weight a rate to the corpus.
- **Working file:** `heldout_to_tag.csv`. Migrated by `scripts/build_gold_2026_11_05.py --set test`,
  which refuses a tagged article that is not already in the tracked shell.
- **Reading it:** `uv run scripts/report_wi18_2026_11_06.py scope --gold --out gold` gives the figures from the
  tracked file alone (its `stratum` and `frame` are the weights): the rules' agreement 68.0% [61.3, 74.5]
  weighted to the corpus, single recall 91.0% [87.1, 94.4] pass, multi recall 52.5% [42.8, 63.5] fail, the
  other four scopes not gated (9, 11, 8 and 7 gold bodies, D33). The working-file command gives the same lines
  (both sort by article id; the intervals of the first report, 61.4 / 74.3 / 63.8, are superseded).
- **Rebuilding:** the set is fixed (D30). A rebuild stops if it would change or drop a tag already in the file
  (`--overwrite-marks` accepts it and prints the ids that change); only an untagged row of the shell is filled in
  freely. The reference is the file as committed (`git show HEAD:`), so a local edit is not the reference (without git or outside a repository, the file on disk; the run prints which one it used). Tests pin the file: the scope counts, the shape of every row, the GICS codes
  offline, the weights summing to the 1,000-body sample, and a hash per row over every column
  (`tests/news_nlp/heldout_gold_row_hashes.csv`; a failure names the ids that differ, and a deliberate change
  is made with `build_gold_2026_11_05.py --set test --overwrite-marks --write-hashes`, which renews that file, in the same PR); `main()` is run end to end on a tiny set. They cannot check the tickers (the
  SOURCE universe is not in the repo; the builder does). The tracked `frame` is the only copy of the weights in
  `gold/`; a further test compares it with `data/heldout_strata.csv`, only where that file exists (not on CI).

### 3. Label check — `gold/segment_check_gold.csv`

- **What:** 100 `lead` and 100 `continuation` paragraphs drawn from the `train` split of
  `company_segments_train.csv` (`scripts/draw_segment_check_2026_11_05.py`). The owner reads each
  paragraph with the one before it and marks `correct` (yes / no) for the company the parser chose.
- **Scope:** a check of an **automatic label**, not a tagging of the article; the parser's choice is shown on
  purpose. Gold columns: `key`, `article_id`, `segment_index`, `ticker`, `kind`, `year`, `correct`, `notes`.
- **Used for:** the rule fixed on 2026-11-06 (D27): if at most 80% of the `continuation` paragraphs are
  right they are dropped from training. **It decides what the attribution model trains on.**
- **Result (read 2026-11-06):** `lead` 88.0% right (95% interval 80.2 to 93.0), `continuation` 38.0%
  (29.1 to 47.8; 2022+ only 55.2%, 37.5 to 71.6). The rule gives **train on `lead` only**. Of the 12 wrong
  `lead` labels, 5 are paragraphs about two or more companies, 3 a look-alike entity (another ticker, a
  person) and 4 are not text about a company (market-mover intros, a table row).

### 4. Extractor sample (faithfulness) — `gold/faithfulness_gold.csv`

- **What:** 30 bodies the owner tagged `multi` that the one-segment-per-company parser cannot split
  (narrative bodies), drawn once (`scripts/draw_faithfulness_2026_11_06.py`, `data/faithfulness_bodies.csv`).
  Each extractor (the mention-and-recency baseline today, the trained model later) writes the text each
  named company gets; the owner reads each text with its article and marks `faithful` (yes / no).
- **Scope:** the file is **blind to the extractor** (rows shuffled, a text two extractors wrote identically
  is one row). The row count is not fixed: it is one row per distinct (body, company) text, known when
  drawn. Gold columns: `key`, `article_id`, `ticker`, `text_hash`, `extractors`, `revisions`, `pub_date`, `faithful`, `notes`.
  `revisions` is, in the order of `extractors`, how each extractor was versioned when the texts were
  written (`retag.extractor_revision`: `extractor:baseline@<digest>`, later the model `name@<checkpoint
  revision>`), so the model-against-baseline comparison of D32 names the baseline it was made against.
  `text_hash` is `retag.text_hash(text)`: SHA-256 (hex) of the text the owner reads, outer whitespace
  stripped, UTF-8, as the draw wrote it (capped at `TEXT_CAP`). The draw writes it to
  `faithfulness_ids.csv` and the builder copies it from there, never recomputes it (the tagging file may
  show the text with a spreadsheet-safe apostrophe).
  In the gold a mark belongs to a text, identified by (`article_id`, `ticker`, `text_hash`), **not** by
  `review_order`. Before the builder (T-155) has run, `report_wi18` and the owner's file use `review_order`,
  so **never draw this set again once marking has begun**: a new draw reshuffles it and the marks point at
  other texts with no error.
  **Adding the model after the baseline is marked:** `draw_faithfulness_2026_11_06.py --add-extractor model`
  (when the model exists, T-148b3). It reads the first draw's bodies and ids, a text the model wrote
  identically to a row already there only adds its name and revision to that row of the ids file, and every
  other text is appended to the tagging file as a new row with the next `review_order`. No existing row or
  mark is changed; a copy of the file as it was is saved as `faithfulness_to_tag.before_model.csv` (a second
  run stops while that copy exists). The appended rows are all the model's, so the blindness to the extractor
  is only partial for them (their position gives them away); the owner is told the file has grown.
- **What `faithful` means (as the 190 marks of 2026-11-06 apply it; the owner confirmed this rule on
  2026-11-06):** `yes` if the text is about the company, even when it also names or discusses other
  companies (45 of the 115 `yes` marks carry a note like "it also speaks about Pfizer"); `no` if the text is
  mainly about another company or the company is only mentioned in passing. The model's texts are judged by
  the same rule.
- **Companies left without a text:** `gold/faithfulness_coverage.csv` (columns `article_id`, `ticker`,
  `extractor`, `pub_date`) lists the companies an extractor named and wrote no text for; the D32 share counts
  each as unfaithful. **On 2026-11-06 it was empty (header only): the baseline left no company without a text. Since the migration of 2026-11-10 it has 25 rows (the later extractors), 16 in the confirmation set.**
  The builder copies it from `data/faithfulness_coverage.csv` next to the gold file and refuses to run without it.
- **Reading it:** `uv run scripts/report_wi18_2026_11_06.py faithfulness --gold --out gold` gives the
  figures from the two tracked files alone (baseline 60.5% [45.2, 73.8] on the 190 texts of 2026-11-06; on the 204 baseline texts of the migrated file 63.2% [49.1, 75.5], 2022+ 62.9% [34.6, 86.1]); nothing from
  the git-ignored `data/` is needed. A test pins these figures.
- **Rebuilding:** the builder rewrites the file from `data/`, but this file is the reference: if a mark, a text hash or a row
  would change or go missing the run stops (`--overwrite-marks` to accept). The tracked coverage is protected the same way (a company without a text it lists cannot disappear), it must belong to a body in `faithfulness_bodies.csv` and not to a company the same extractor wrote a text for, and the gold file and the coverage file are written together: if the second cannot be written the first is put back, and if that fails the message names the file that is new. Adding the model only adds names to
  `extractors` and `revisions` and new rows, and goes through.
- **Used for:** the D32 bar (>= 80% of the extracted texts faithful, and better than the baseline on the same
  bodies, 2022+ reported apart), which decides whether narrative multi bodies enter the backtest cut.
- **Later files of this set (2026-11-08; migrated to `gold/` on 2026-11-10, marks and ids only):** the same file now holds 437 texts of the 30 bodies (baseline, `model`, `units`, `units2`, `units3`; 336 yes, 101 no) and `faithfulness_exempt_to_read.csv` the owner's reading of 37 names (23 not companies). A **confirmation set** of 30 other bodies (`data/faithfulness_confirmation/`: 395 texts, 345 yes and 50 no, and 30 names read, 25 not companies) was tagged the same day. **The owner selected `units` as the base model of step 4 (2026-11-08).** Datasets, metrics, results and what is next: `docs/extractor_training.md`. After PR #155 (matcher change) the figures were re-read: see `docs/extractor_training.md`, "Update of 2026-11-10" (`units`: 87.3% counting every mark and 97.1% on the texts the current code writes, confirmation set; first set 97.5%).
- **Earlier ids files, local copies (2026-11-10, in `data/earlier_2026_11_09/`, not in `gold/`):** `first/faithfulness_ids.csv` (sha256 `e176141e935b483e99531c9c18af17a93ecabc53397d248dd3afdb452d9c708d`) and `confirmation/faithfulness_ids.csv` (sha256 `8a3e2a12145649e997f99043d0a27daebeb68d898d6c3d14af0d20c480a713d3`), copied from a backup made on 2026-11-09 at 23:10, before the ids files were redrawn at 23:20 and 23:39 for the matcher change (PR #155). They exist only so that `scripts/dead_text_replacement_2026_11_10.py` can tell which rows are new; nothing reads them otherwise.

### 5. Sentiment — `gold/sentiment_gold.csv`

- **What:** 180 texts of the pipeline's own output (D26): 40 per predicted class from the summaries and the
  company texts of structured multi bodies (`part=checked`, 120) and 20 per predicted class from the texts
  the extractor writes for narrative multi bodies (`part=extractor`, 60). Drawn by
  `scripts/draw_sentiment_gold_2026_11_05.py`; the current model's prediction picks the sample and is
  never shown to the owner.
- **Scoring another model (T-148c2, D35):** `scripts/score_sentiment_gold_2026_11_07.py --model <path or Hub id>` runs a candidate
  over the 180 texts (from `data/sentiment_gold_ids.csv`, each checked against its `text_hash`) and prints the report of
  `report_wi18 sentiment --gold` for that model, with the strata of the draw, plus the agreement for 2022 on and before and for texts over and up to 128 tokens (T-148c2b). A **final** run also logs the paired gain over the last baseline row (agreement and negative precision and recall, each with its verdict: the pin rule reads it), a second baseline or final run needs `--redo-role` and a `--redo-reason` (logged), the row says the input the model reads (`input_format`) and how many rows each model read (`rows_pair`, `rows_text_only`), and a model that reads the pair company + text (`--pair`, never the baseline) is scored as the routed system: the 143 rows with a ticker by that model, the 37 without one by the text-only model. The gold is
  not a tuning set: **every run is appended to `gold/sentiment_gold_scoring_log.csv`** and the report that quotes a figure lists
  them all. Run 1 is the pinned model, which reproduces the figures above (0 of 180 predictions differ). Each row identifies what was scored (the Hub revision, which a Hub model must have and which must be a commit, 40 hexadecimal characters, or for a local checkpoint the hash of its weight files, and the same for the text-only model of a routed run; the digest of the scoring code; the hash of `sentiment_gold.csv`; the model's labels; the device) and keeps the figures as text and as numbers, with 2022 on and before apart and the judgement of every bar (`pass`, `fail`, `not decided`, or `undefined` when the figure has no interval, PLAN D32). `--role` says which run counts (D35): `baseline` and `final` are one each unless `--redo-role`, and the report that quotes a figure takes the **last** `final` row (an earlier one stays in the log as a record, not as the figure); a test keeps that row equal to the model pinned in `config/pipeline_models.json`; a model that does not answer with the three classes stops the run before anything is logged, and any run that is not the planned baseline or final run (role `other`, or a `--redo-role`) prints a warning with the count of extra looks (D35).
- **Scope:** the label is the **price impact on the company** an investor would read (on the sector or
  the market for a sector or market text): `positive`, `negative`, `neutral`. Gold columns: `key`,
  `article_id`, `ticker`, `kind`, `part`, `text_hash`, `revision`, `stratum`, `extractor`, `predicted`,
  `population`, `pub_date`, `sentiment`, `notes` (the draw's weights travel with each row). The text itself is not in
  `gold/` (news-derived), and **it cannot be rebuilt**: a summary is generated by a model, a company text
  depends on the extractor. So the label is tied to its text by `text_hash` (same hash as above) and to the
  way the text was made by `revision` (the summariser `model@revision` of `config/pipeline_models.json`, or for code that is not
  pinned like a model `company_texts@<digest>` / `extractor:<name>@<digest>`: `retag.code_revision`, 12 hex
  digits of the SHA-256 of the source of the modules the output depends on (the parser: `company_texts`,
  `triage`; the baseline: `attribution_baseline`, `triage`, `chunking`). **It detects an edit to those
  modules; it does not cover the company universe or other data they read.** A scored pool is refused for
  each kind of text whose code changed since it was scored (the summaries depend on a pinned model, not on
  these modules). Any edit to those modules, a comment included, counts, and the only way out is to delete
  the pool and its meta file and score again, which can change the stratum sizes: do not edit `triage.py`,
  `chunking.py`, `company_texts.py` or `attribution_baseline.py` between `--pool-only` and the draw). The only copy of the texts is
  `data/sentiment_gold_ids.csv` (its `text` column); **back it up**, since without it the 180 marks cannot
  be used to measure another model. Back up `sentiment_gold_pool.csv` and its meta file too: deleting them
  makes the next draw score the pool again, with possibly different strata and sizes than the ones the marks
  were weighted by.
- **Used for:** the D32 bar (overall agreement >= 80%; precision and recall >= 70% for negative), re-weighted
  by the stratum sizes (`population`). It says whether the pinned
  sentiment model can read summaries and company texts.
- **Result (owner's marks, 2026-11-06):** 180 marked, 82 positive, 65 negative, 33 neutral. Overall agreement
  60.4% [52.8, 67.6] (raw 57.2%): **fails** the 80% bar. Negative precision 78.7% [67.3, 89.3] and recall 77.7%
  [69.4, 86.8]: not decided against 70%. By part: `checked` (summaries and structured company texts) 56.4%
  [48.9, 64.4]; `extractor` 64.0% [51.5, 76.0]. The two parts both fail; the extractor's texts are not the
  weaker ones. Per the rule above, a mark is tied to its text by `text_hash` and `revision`, so the figures can
  be recomputed for another model on the same 180 texts, which are only in `data/sentiment_gold_ids.csv`.
- **Where it fails (raw counts of the 180; rows = the model, columns = the owner):**

  | model \ owner | positive | negative | neutral |
  |---|---|---|---|
  | positive | 39 | 9 | 12 |
  | negative | 9 | 47 | 4 |
  | neutral | 34 | 9 | 17 |

  The largest error by far is a text the model calls neutral and the owner calls positive (34 of the 60
  predicted neutral). The notes of the owner are not the cause: 25 of the 31 notes say the text is not about the
  company or states no relevant fact (all marked neutral), but agreement on the other 155 texts is 57.4%,
  against 57.2% on all 180 (the 25 agree 14 of 25); the 25 keys, read by hand: 158982#TTD 328872#TTWO 323191#EQR 474491#TTWO 164613#JNJ 155187#TTWO 341540#OKE 341540#CAT 295682#JPM 252049#BE 341540#MDLZ 341540#BA 93680#AMZN 474491#TTD 264983#summary 157085#AMZN 208175#MSI 115964#AAPL 439320#STLD 465240#FLEX 169817#summary 90037#GS 191952#HD 131202#FRT 174853#STZ). (The count is by reading the 31 notes: 25 "not about the company / no relevant fact",
  2 "unclear", 3 "weird" (the news is good and the stock fell; marked neutral), 1 "Investor activity" (positive).)
- **The same text for two companies:** 16 of the 180 texts are shared by two companies or more of one
  article, and marked once per company. Two of the eight pairs got different marks (157085 NFLX positive and AMZN
  neutral; 164613 JNJ neutral and AMZN positive), which is allowed: the label is the impact on that company.
  The intervals count them as independent, so they are a little narrow; the report prints the number.
- **Dates:** 49 of the 180 texts are from 2022 on (`pub_date`, carried from the file the owner tagged).

### 6. Sentiment training set — `gold/sentiment_train.csv` (D35)

- **What:** the 450 summaries the owner tags to fine-tune the sentiment model on summaries (T-148c, after the
  model missed the D32 bar on set 5). It is **not a gold**: it trains the model, and **it is never mixed with set 5**,
  the 180-text exam. Nothing in it may be an article (by body hash, or under the same headline) of set 5 or of the
  held-out set 2; the draw and the builder check it.
- **Draw:** `scripts/draw_sentiment_train_2026_11_07.py` takes 150 of each class the current model predicts from the
  summaries of the cut (the prediction only balances the draw and is never shown), in 30 blocks of 15 (5 of each
  class). A batch of 150 is 10 blocks, and the last 2 blocks of a batch are the **validation slice** (90 in all: it
  picks the settings, never trains); the other 360 train. Only summaries: the texts of multi bodies do not exist yet.
- **Files:** `data/sentiment_train_ids.csv` (the draw and **the only copy of the texts the owner read**: back it up;
  the summaries are also in the cut's chunks by body hash), `data/sentiment_train_to_tag.csv` (the owner's file,
  Spanish columns after `text`), `gold/sentiment_train.csv` (the marks and ids, no news text; written by
  `scripts/build_sentiment_train_2026_11_07.py`, T-148c1, PR #137). Migrated 2026-11-08. **Columns:** `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, `sentiment`, `notes`. `company` is on every row because the mark is about it (a name is not news text; the owner can ask for it to be dropped); `ticker` is blank on 26 rows, which are sector (19), market (4) and economy (3) texts whose `company` is a bracketed placeholder (`[sector]`, `[market / economy]`). The builder protects both once the file has them. `pub_date` is kept as the owner's file gave it (a protected column) in three ISO forms: `...+0000` (355 rows), `...Z` (77) and `....000Z` (18); a test pins the three counts.
- **Second file (T-148c3b, 2026-11-10):** `gold/sentiment_train_2.csv`, same columns, 405 rows (178 positive,
  116 negative, 111 neutral), 135 of each predicted class (the draw balances on the prediction, so the owner's classes
  are not balanced). Working files, git-ignored: `data/sentiment_eval_ids.csv` (the draw and the only copy of the
  texts) and `data/sentiment_eval_to_tag.csv` (the owner's file); the names say "eval" from the draw and were not
  changed. The slice split (blocks 1-18 `train`, 19-27 `validation`) was chosen by the assistant, not by the owner.
  Copies of both working files are in `data/backup/` (`sentiment_eval_ids_2026-11-10.csv`, `sentiment_eval_to_tag_2026-11-10.csv`): the ids file is the **only copy of the 405 texts the owner read**, so keep it with the owner's other backups of `data/` (the repository does not hold it). No key, body or text is in both files (a test checks it). 20 rows (26 in the first file) are sector, market or
  economy tags with no company and are left out of the pair model; the pair model reads 809 of the 855 rows.
- **Label: anchored to the tagged company (owner's rule, 2026-11-08; a design feature to preserve).** The
  mark is what the text says **about the company of the row** (`company`, `ticker`): the price impact an
  investor would read on that company. If the company is in the facts the text presents, the label is the
  sentiment about it; if the text does not mention the company, or the part that concerns it is neutral,
  the label is **neutral**, even when other sentences of the text are positive or negative (a positive
  sentence about another company, or about the market, does not make the text positive for this row).
  **The title is context, not ownership (owner's precision, 2026-11-08):** the title may be used to know
  whether a text speaks about a company, but care is needed not to give a sentence to a company only
  because the title names it: a sentence belongs to the company when what it says is about that company
  (the text read with the title as context), not because the headline mentions the company. A title that
  names the company does not make a sentence about something else, or about another company, that
  company's.
  This is how the company is attached to the sentiment prediction. Reading the figures: a model that
  scores the sentiment of the text without the company is expected to disagree with these marks on the
  rows where the company is absent or irrelevant, which is not an error of the marks. **How much of the
  disagreement of the draw's model with the owner (the confusion below) that explains is small by the
  notes and most of it is unexplained** (see "Where the disagreement is").
- **When the rule was given (recorded, not inferred):** the owner gave it on 2026-11-08. The repo dates the tagging earlier: TASKS T-148c1 (written 2026-11-07) says the owner expected to have all 450 marked that day, and the marks were migrated in commit `9c59620` of 2026-11-08. That is the developer's reading of those two dates, which the owner can correct. Whether the rule was in the owner's head during the tagging is not recorded, and nothing in the repo can show it. What the file shows: 49 of 450 rows carry a note and 19 of them say something about the company by keyword (9 that it is absent, e.g. "Tesla not in the news"; see below), so the rule was applied at least there; the label mix is the same in the three batches (neutral 35 / 35 / 35, positive 76 / 80 / 76, negative 39 / 35 / 39). The rule is therefore written as **what the marks mean, as the owner states it**, not as a procedure known to have been followed on every row. 46 of the 105 neutral rows have no note, so "neutral because the company is absent" and "neutral because nothing financial" cannot be told apart in the file; no more notes are asked.
- **Set 5 against set 6 (the same definition?):** set 5's scope line says the price impact on the company an investor would read. Its notes (31 of 180 rows, all read: 25 say the text is not about the company or gives no relevant fact, set 5's own section) show the same practice: neutral with "The company is not mentioned in the news", "Company is not in the sentences", "This is not about the company Amazon", "Even when Apple is mentioned, the article doesn't say any financial fact about the company". So the exam and the training set follow the same reading as far as the notes show; the 149 rows without a note are not checked and this is **not proven** for them. **The two sets differ on the rows the notes mark, and the cause is not known:** in set 5 the 25 company-irrelevant rows agree with the model on 14 of 25 (56%), against 57.4% on the other 155; in set 6 the 19 rows with a note about the company agree on 3 of 19 (16%), against 251 of 431 (58.2%) on the rest and 254 of 450 (56.4%) overall. Either the rule was applied differently, or the notes of the two sets cover a different mix of rows; nothing here tells which. Informative, nothing retagged.
- **Marks (2026-11-08):** 450 of 450, 232 positive, 113 negative, 105 neutral (train 180 / 96 / 84,
  validation 52 / 17 / 21). The model that balanced the draw (rows) against the owner (columns), positive |
  negative | neutral: positive 120 / 6 / 24; negative 37 / 83 / 30; neutral 75 / 24 / 51.
- **Where the disagreement is (informative, no decision; the model that balanced the draw against the owner):** by `rule_scope`, `single` 287 rows, agreement 55.4%; `multi_dominated` 137 rows, 62.8%; sector, market and economy 26 rows, 34.6%. The draw is balanced on the prediction, so these are not the model's accuracy. The difference `multi_dominated` minus `single` is +7.4 points, bootstrap 95% over rows [-2.7, 17.1]: it straddles zero, so the data do not tell the two scopes apart and nothing is concluded from it. **The only measure of the company anchor is the notes:** 49 rows have a note, and by keyword (`NOTE_KINDS` in the builder, which prints the counts; approximate in both directions, no bound) 19 of them say something about the company: the company is absent 9, the company is there but nothing relevant is said about it 4, the row is about something else than a company (a sector, the market) 3, the company is a namesake 3. All 19 are marked neutral, and the model called 16 of them non-neutral. So by the notes the anchor accounts for few of the 450 rows, the notes cover only 49, and **the rest of the disagreement is not explained**. T-148c3's own scoring is the place to look; it is the owner's.
- **Rows whose company is probably a namesake (known issues of the draw, recorded, not fixed, nothing retagged):** by the owner's notes, `91090#summary` (Amazon: "Amazon forest, not for the company"), `479890#summary` (Baxter International: "comes from a person name"), `475332#summary` (Visa Inc.: "the legal document, not the company"), and `1082641#summary` (Block, Inc.: "No mentions to the company Block", which reads as absent or as a namesake). On these the `company` column is the company the resolver found, not one the text is about; training on them teaches neutral for a mention that is not the company. It is the failure family PR #144 found in the faithfulness file (first-word and namesake matches). **Proposed, not started (TASKS T-148c1b):** measure how many of the 450 rows carry a namesake company with the resolver the draw used; what to do with them is the owner's.
- **Validation shares companies with train (a risk for T-148c3, recorded, the slices are D35's design and unchanged):** the slices are disjoint by article, not by company. Of the 83 validation rows with a named company, 64 (77%) have a company that is also in train; ten companies are 36.6% of all rows (Tesla 26, Amazon 24, Apple 23, Microsoft 18, Nvidia 16, Goldman Sachs 12, Boeing 10, Bank of America 9, Berkshire Hathaway 9, Morgan Stanley 8), and a company's marks lean (Tesla 14 negative of 26, Apple 16 positive of 23). A validation score can reward remembering a company's usual label. The exam (set 5) has 101 of its 180 rows (46 of its 81 tickers) with a ticker that is in train. **Proposed, the owner's to accept (T-148c3 done-when):** report validation agreement, and the exam's, split by companies seen in train and unseen, so a gain that is memorisation is visible.
- **Integrity (2026-11-08):** the builder stops unless every row is marked, the text the owner read is the
  drawn text (hash), the ids pair exactly and the design is the one drawn: it passed. Besides: 450 articles
  and 450 body hashes, all different; no article shared with set 5 (128 articles) or with the held-out set
  (200); no spreadsheet damage (no encoding change, no fill-down run). `tests/news_nlp/test_build_sentiment_train.py`
  pins the committed file (counts, design, no news text, no overlap with the two exams).

## Roadmap: what each set unlocks and the order to tag

Fixed dates are the ones in `PLAN.md` D34 (cut for the 2026-11-20 delivery); the extractor decision is on
2026-11-09 (D34 rule b; D32 now says the same). The order is by what
each set blocks, earliest block first.

| Order | Set | Unblocks | Needed by | Waits for |
|---|---|---|---|---|
| 1 | **3. Label check** (200 paragraphs) | training the attribution model (`lead` only, or both kinds) | 2026-11-08 | nothing: the training set exists |
| 2 | **4. Extractor sample** (30 bodies, 190 texts) | the decision on the extractor for narrative bodies; the extractor part of set 5 | 2026-11-09 | nothing: baseline-only draw (D34 rule b) |
| 3 | **5. Sentiment** (180 texts) | the D32 sentiment bar, so the cut can say whether its sentiment is checked | 2026-11-12 | the extractor decision (its 60 texts come from the extractor chosen) |
| 4 | **2. Scope test** (200 bodies) | the scope measure (rules now, classifier later), the D24 category target | 2026-11-12 (go / no-go) | nothing: it can be tagged any time, but the cut does not wait for its result to start |
| done | 1. Scope train | the labelling prompt, the sample for sets 4 and 5 | | |

Owner work before the go / no-go of 2026-11-12: 770 items (200 + 200 + 190 + 180), 200 of them marked
(the label check). The PLAN counted 610 by taking 30 texts for the faithfulness set, but the set is 30
*bodies* and one row per company text: drawn, it is 190. **Owner call, 2026-11-06: tag all 190** (no reduction).

Why this order:

- Set 3 is first because the model trains on 2026-11-08 and the rule on the `continuation` labels must be
  applied *before* training; it is the shortest set and needs no other result.
- Set 4 needs only the baseline, so it can be drawn and tagged at once; if the model is not ready on
  2026-11-09 the decision is on the baseline alone.
  **Only if the 190 are not all marked on 2026-11-09 (the owner expects to finish them on 2026-11-06, so this is a contingency; the 120 below is a proposal the owner has not confirmed; fixed before any figure is read):**
  the owner marks in `review_order` order; on 2026-11-09 the decision uses the first rows marked, if at
  least 120 (about two thirds; `report_wi18 faithfulness --partial 120`, which refuses a gap or fewer
  rows and prints PARTIAL, counting every company left without a text); with fewer the baseline is kept and
  the model is reported as not evaluated. The full figure replaces the partial one when the file is finished.
- Set 5 draws its extractor part from whichever extractor set 4 selected, so it is drawn after.
  Its checked part (120) does not depend on it and can be tagged first if the owner wants to split the work.
- Set 2 is independent of the others, so it fills the gaps; it is last only because nothing downstream
  starts from its result.

## Files an interrupted draw can leave

Draw writes go through `<name>.tmp` and `os.replace`, which removes the temporary file on an error but not
on a hard kill. A leftover `*.tmp` next to a tagging file is not the owner's work and is ignored by the draws
(`has_rows` looks at the target only); delete it.

## Migrating a finished set (the step the roadmap ends on)

1. The owner marks the `data/` file and runs `scripts/report_wi18_2026_11_06.py <scope|label-check|faithfulness|sentiment>`,
   which refuses a file with unmarked or invalid rows (nothing is read as "no").
2. The set is migrated to its `gold/` file: set 1 and 2 by `build_gold_2026_11_05.py --set train|test`
   (exist; the test file also takes `stratum` and `frame` from the draw's record, and refuses to change a tracked tag without `--overwrite-marks`; `report_wi18 scope --gold --out gold` reproduces its figures); set 3 by
   `build_segment_check_gold_2026_11_06.py` (exists: 200 rows, pairs each mark with the draw by `review_order`);
   set 4 by `build_faithfulness_gold_2026_11_06.py` (190 rows and the coverage file); set 5 by
   `build_sentiment_gold_2026_11_06.py` (180 rows; pairs by the draw's `key`, checks the hash of the text the owner
   read, copies `text_hash`, `revision` and the stratum sizes, and refuses to change a tracked mark without
   `--overwrite-marks`). Each refuses unless every row is marked and valid, and writes marks and ids only.
   `report_wi18 sentiment --gold --out gold` reproduces the D32 figures from the tracked file, the 2022+ count
   and the confusion matrix included, and stops unless the file is the sample the draw took (a known part, a
   predicted class, one size per stratum, 40 / 20 texts per class or the whole stratum when smaller). A rebuild
   may not change the mark, the date, the revision or the weights of a tracked row without `--overwrite-marks`.
3. From then on features read `gold/`; the `data/` file is kept only as the owner's working copy.
