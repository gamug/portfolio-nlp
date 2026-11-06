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
| 3 label check | `segment_check_ids.csv` (the gold keeps its `year`, so the 2022+ figure of D27 survives) |
| 4 extractor sample | `faithfulness_bodies.csv` (the bodies drawn) **and** `faithfulness_ids.csv` (which extractor wrote each `review_order`; without it the report cannot be computed) |
| 5 sentiment | `sentiment_gold_ids.csv` (stratum, prediction, `population`, extractor, the text) **and** `sentiment_gold_pool.csv` with `sentiment_gold_pool_meta.json` (the scored pool the stratum sizes come from) |

## What may sit in `gold/`

Only files with no news text: the owner's tags or marks, the ids of a draw and the weights they need,
or (the one exception) a training set of ids and positions. In the order of the repository: the scope
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
pins the `gold/` headers to the column constants in `retag`.

## The sets

| # | Set | Size | `data/` working file | `gold/` file | State on 2026-11-06 |
|---|---|---|---|---|---|
| 1 | **Scope train** (the 240 of the prompt gate + 44 sector bodies) | 284 bodies | `gate240_to_tag.csv`, `sector_extra_to_tag.csv` | `category_gold_v2.csv` | tagged and migrated |
| 2 | **Scope test** (held-out) | 200 bodies | `heldout_to_tag.csv` | `category_gold_v2_test.csv` | drawn; owner tagging; the `gold/` file is the shell with the draw's weights and no tags yet |
| 3 | **Label check** (step-4 training labels) | 200 paragraphs | `segment_check_to_tag.csv` | `segment_check_gold.csv` | drawn 2026-11-06 (200 paragraphs, Spanish aid bought); owner marking; `gold/` file is a scaffold |
| 4 | **Extractor sample** (faithfulness) | about 30 bodies | `faithfulness_to_tag.csv` | `faithfulness_gold.csv` | empty scaffold, not drawn |
| 5 | **Sentiment** | 180 texts | `sentiment_gold_to_tag.csv` | `sentiment_gold.csv` | empty scaffold, not drawn |

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

### 4. Extractor sample (faithfulness) — `gold/faithfulness_gold.csv`

- **What:** about 30 bodies the owner tagged `multi` that the one-segment-per-company parser cannot split
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
  A mark belongs to a text, identified by (`article_id`, `ticker`, `text_hash`), **not** by `review_order`:
  that is a shuffle position and changes when the model's texts are added to the baseline's. The baseline's
  marks therefore stay valid; a text the model writes identically only adds its name to `extractors`.
- **Used for:** the D32 bar (>= 80% of the extracted texts faithful, and better than the baseline on the same
  bodies, 2022+ reported apart), which decides whether narrative multi bodies enter the backtest cut.

### 5. Sentiment — `gold/sentiment_gold.csv`

- **What:** 180 texts of the pipeline's own output (D26): 40 per predicted class from the summaries and the
  company texts of structured multi bodies (`part=checked`, 120) and 20 per predicted class from the texts
  the extractor writes for narrative multi bodies (`part=extractor`, 60). Drawn by
  `scripts/draw_sentiment_gold_2026_11_05.py`; the current model's prediction picks the sample and is
  never shown to the owner.
- **Scope:** the label is the **price impact on the company** an investor would read (on the sector or
  the market for a sector or market text): `positive`, `negative`, `neutral`. Gold columns: `key`,
  `article_id`, `ticker`, `kind`, `part`, `text_hash`, `revision`, `stratum`, `extractor`, `predicted`,
  `population`, `sentiment`, `notes` (the draw's weights travel with each row). The text itself is not in
  `gold/` (news-derived), and **it cannot be rebuilt**: a summary is generated by a model, a company text
  depends on the extractor. So the label is tied to its text by `text_hash` (same hash as above) and to the
  way the text was made by `revision` (the summariser `model@revision` of `config/pipeline_models.json`, or for code that is not
  pinned like a model `company_texts@<digest>` / `extractor:<name>@<digest>`: `retag.code_revision`, 12 hex
  digits of the SHA-256 of the source of the modules the output depends on (the parser: `company_texts`,
  `triage`; the baseline: `attribution_baseline`, `triage`, `chunking`). **It detects an edit to those
  modules; it does not cover the company universe or other data they read.** The scored pool is refused if
  any of these digests changed since it was scored). The only copy of the texts is
  `data/sentiment_gold_ids.csv` (its `text` column); **back it up**, since without it the 180 marks cannot
  be used to measure another model. Back up `sentiment_gold_pool.csv` and its meta file too: deleting them
  makes the next draw score the pool again, with possibly different strata and sizes than the ones the marks
  were weighted by.
- **Used for:** the D32 bar (overall agreement >= 80%; precision and recall >= 70% for negative), re-weighted
  by the stratum sizes (`population`). It says whether the pinned
  sentiment model can read summaries and company texts.

## Roadmap: what each set unlocks and the order to tag

Fixed dates are the ones in `PLAN.md` D34 (cut for the 2026-11-20 delivery); the extractor decision is on
2026-11-09 (D34 rule b; D32 now says the same). The order is by what
each set blocks, earliest block first.

| Order | Set | Unblocks | Needed by | Waits for |
|---|---|---|---|---|
| 1 | **3. Label check** (200 paragraphs) | training the attribution model (`lead` only, or both kinds) | 2026-11-08 | nothing: the training set exists |
| 2 | **4. Extractor sample** (about 30 bodies) | the decision on the extractor for narrative bodies; the extractor part of set 5 | 2026-11-09 | nothing: baseline-only draw (D34 rule b) |
| 3 | **5. Sentiment** (180 texts) | the D32 sentiment bar, so the cut can say whether its sentiment is checked | 2026-11-12 | the extractor decision (its 60 texts come from the extractor chosen) |
| 4 | **2. Scope test** (200 bodies) | the scope measure (rules now, classifier later), the D24 category target | 2026-11-12 (go / no-go) | nothing: it can be tagged any time, but the cut does not wait for its result to start |
| done | 1. Scope train | the labelling prompt, the sample for sets 4 and 5 | | |

Owner work before the go / no-go of 2026-11-12: 580 rows (200 + 200 + 180) plus the faithfulness texts.
The PLAN counts 610 by taking 30 texts for them, but the set is about 30 *bodies* and one row per company
text, so the real number is known only when it is drawn.

Why this order:

- Set 3 is first because the model trains on 2026-11-08 and the rule on the `continuation` labels must be
  applied *before* training; it is the shortest set and needs no other result.
- Set 4 needs only the baseline, so it can be drawn and tagged at once; if the model is not ready on
  2026-11-09 the decision is on the baseline alone.
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
   (exist; the test file also takes `stratum` and `frame` from the draw's record); sets 3, 4 and 5 by
   a builder that does not exist yet and is written with the same rules (refuse unless every row is marked
   and valid, marks and ids only, sorted, the weights of the draw copied). **Writing those three builders
   is task T-155**: until then those `gold/` files stay header-only and the reports read `data/`.
3. From then on features read `gold/`; the `data/` file is kept only as the owner's working copy.
