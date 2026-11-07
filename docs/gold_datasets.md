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
`gold/` holds only the train and the test file (2026-11-05) is superseded by this one. Anything with
text, a cache or a translation stays in `data/`.

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

| # | Set | Size | `data/` working file | `gold/` file | State on 2026-11-06 |
|---|---|---|---|---|---|
| 1 | **Scope train** (the 240 of the prompt gate + 44 sector bodies) | 284 bodies | `gate240_to_tag.csv`, `sector_extra_to_tag.csv` | `category_gold_v2.csv` | tagged and migrated |
| 2 | **Scope test** (held-out) | 200 bodies | `heldout_to_tag.csv` | `category_gold_v2_test.csv` | drawn; owner tagging; the `gold/` file is the shell with the draw's weights and no tags yet |
| 3 | **Label check** (step-4 training labels) | 200 paragraphs | `segment_check_to_tag.csv` | `segment_check_gold.csv` | drawn and marked 2026-11-06: `lead` 88.0% right, `continuation` 38.0%, so the model trains on `lead` only (D27); `gold/` file built 2026-11-06 (200 rows, `build_segment_check_gold_2026_11_06.py`) |
| 4 | **Extractor sample** (faithfulness) | 30 bodies, 190 texts | `faithfulness_to_tag.csv` | `faithfulness_gold.csv` | drawn, marked and migrated 2026-11-06: 190 texts of 30 bodies, 115 faithful and 75 not; the baseline fails the D32 bar (60.5%, interval 45.2 to 73.8); `gold/` file built (190 rows, `build_faithfulness_gold_2026_11_06.py`) |
| 5 | **Sentiment** | 180 texts | `sentiment_gold_to_tag.csv` | `sentiment_gold.csv` | drawn, marked and migrated 2026-11-06: 82 positive, 65 negative, 33 neutral; overall agreement 60.4% (interval 52.8 to 67.6), under the D32 bar of 80%; negative precision 78.7% (67.3 to 89.3) and recall 77.7% (69.4 to 86.8), not decided against 70%; `gold/` file built (180 rows, `build_sentiment_gold_2026_11_06.py`) |
| 6 | **Sentiment training set** (not a gold: it trains, never exams) | 450 summaries | `sentiment_train_to_tag.csv` | `sentiment_train.csv` | drawn and translated 2026-11-07 (`draw_sentiment_train_2026_11_07.py`); the owner tags all 450 on 2026-11-07; builder and trainer are T-148c1 and T-148c3 |

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
- **Scope:** the same tag columns as set 1, `session=heldout`, plus `stratum` and `frame` (the weights of D30). In `gold/` the tag columns are empty until the owner finishes.
- **Used for:** the **only** measure of the scope: the rules' and later the classifier's agreement with
  the owner (>= 90% corpus-weighted, D20/D33) and the D24 category target. Never trained on (Q3).
  `data/heldout_strata.csv` holds the stratum sizes (`frame`) that re-weight a rate to the corpus.
- **Working file:** `heldout_to_tag.csv`. Migrated by `scripts/build_gold_2026_11_05.py --set test`,
  which refuses a tagged article that is not already in the tracked shell.

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
  each as unfaithful. **On 2026-11-06 it is empty (header only): the baseline left no company without a text.**
  The builder copies it from `data/faithfulness_coverage.csv` next to the gold file and refuses to run without it.
- **Reading it:** `uv run scripts/report_wi18_2026_11_06.py faithfulness --gold --out gold` gives the
  figures from the two tracked files alone (baseline 60.5% [45.2, 73.8], 2022+ 60.6% [27.6, 86.1]); nothing from
  the git-ignored `data/` is needed. A test pins these figures.
- **Rebuilding:** the builder rewrites the file from `data/`, but this file is the reference: if a mark, a text hash or a row
  would change or go missing the run stops (`--overwrite-marks` to accept). The tracked coverage is protected the same way (a company without a text it lists cannot disappear), it must belong to a body in `faithfulness_bodies.csv` and not to a company the same extractor wrote a text for, and the gold file and the coverage file are written together: if the second cannot be written the first is put back, and if that fails the message names the file that is new. Adding the model only adds names to
  `extractors` and `revisions` and new rows, and goes through.
- **Used for:** the D32 bar (>= 80% of the extracted texts faithful, and better than the baseline on the same
  bodies, 2022+ reported apart), which decides whether narrative multi bodies enter the backtest cut.

### 5. Sentiment — `gold/sentiment_gold.csv`

- **What:** 180 texts of the pipeline's own output (D26): 40 per predicted class from the summaries and the
  company texts of structured multi bodies (`part=checked`, 120) and 20 per predicted class from the texts
  the extractor writes for narrative multi bodies (`part=extractor`, 60). Drawn by
  `scripts/draw_sentiment_gold_2026_11_05.py`; the current model's prediction picks the sample and is
  never shown to the owner.
- **Scoring another model (T-148c2, D35):** `scripts/score_sentiment_gold_2026_11_07.py --model <path or Hub id>` runs a candidate
  over the 180 texts (from `data/sentiment_gold_ids.csv`, each checked against its `text_hash`) and prints the report of
  `report_wi18 sentiment --gold` for that model, with the strata of the draw, plus the agreement for 2022 on and before. The gold is
  not a tuning set: **every run is appended to `gold/sentiment_gold_scoring_log.csv`** and the report that quotes a figure lists
  them all. Run 1 is the pinned model, which reproduces the figures above (0 of 180 predictions differ).
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
  `scripts/build_sentiment_train_2026_11_07.py`, which arrives with T-148c1, PR #137).
- **Label:** the same as set 5: the price impact on the company an investor would read.

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
   (exist; the test file also takes `stratum` and `frame` from the draw's record); set 3 by
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
