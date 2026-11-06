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
then on.** The ids and the `population` weights of a draw stay in `data/<stem>_ids.csv`
(`<stem>_bodies.csv` for the faithfulness set); they are the draw's record, not tagging work.

A set not drawn yet has an **empty scaffold**: a header-only file in `data/` (the exact columns the
draw script writes) and one in `gold/`. The draw scripts treat a header-only `data/` file as absent and
fill it; a file with rows is the owner's work and is never overwritten.

## The sets

| # | Set | Size | `data/` working file | `gold/` file | State on 2026-11-06 |
|---|---|---|---|---|---|
| 1 | **Scope train** (the 240 of the prompt gate + 44 sector bodies) | 284 bodies | `gate240_to_tag.csv`, `sector_extra_to_tag.csv` | `category_gold_v2.csv` | tagged and migrated |
| 2 | **Scope test** (held-out) | 200 bodies | `heldout_to_tag.csv` | `category_gold_v2_test.csv` | drawn; 2 of 200 tagged; `gold/` file is the untagged shell |
| 3 | **Label check** (step-4 training labels) | 200 paragraphs | `segment_check_to_tag.csv` | `segment_check_gold.csv` | empty scaffold, not drawn |
| 4 | **Extractor sample** (faithfulness) | about 30 bodies | `faithfulness_to_tag.csv` | `faithfulness_gold.csv` | empty scaffold, not drawn |
| 5 | **Sentiment** | 180 texts | `sentiment_gold_to_tag.csv` | `sentiment_gold.csv` | empty scaffold, not drawn |

Not a gold set, listed so nobody mistakes it for one: `gold/company_segments_train.csv` is the
**training set** of the step-4 attribution model (33,624 paragraphs, automatic labels, no owner tags;
`scripts/build_company_segments_2026_11_05.py`). Set 3 is the check of its labels.

### 1. Scope train — `gold/category_gold_v2.csv`

- **What:** the owner's tags of the 240 bodies fixed for the prompt gate (`session=gate240`, stratified
  on the v1 scope) and 44 sector bodies (`session=sector_extra`).
- **Scope:** one row per unique body: `scope` (six scopes), `company_labels` (the analysed companies with
  their D15 label), `mentions`, `sub_industry`, `notes`. Rules: `labelling-spec-v2.md`.
- **Used for:** writing and tuning the labelling prompt (the gate, D18) and as the positive examples of
  the scope classifier. **Never used to measure** the scope (it tuned the prompt). It is also the pool
  from which sets 4 and 5 are drawn (the bodies tagged `multi`, and the tagged scopes of the summaries).
- **Working files:** `gate240_to_tag.csv` and `sector_extra_to_tag.csv` (columns: `retag.TAGGING_COLUMNS`).
  Migrated by `scripts/build_gold_2026_11_05.py --set train`.

### 2. Scope test — `gold/category_gold_v2_test.csv`

- **What:** the 200 held-out bodies (D30): all 39 the rules call no-company, sector, market or economy,
  the rest a seeded random draw from the 1,000-body corpus sample, none sharing a body with sets 1 or the
  720. Drawn once; the set is fixed.
- **Scope:** the same tag columns as set 1; `session=heldout`. In `gold/` the tag columns are empty until
  the owner finishes.
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
  purpose. Gold columns: `key`, `article_id`, `segment_index`, `ticker`, `kind`, `correct`, `notes`.
- **Used for:** the rule fixed on 2026-11-06 (D27): if at most 80% of the `continuation` paragraphs are
  right they are dropped from training. **It decides what the attribution model trains on.**

### 4. Extractor sample (faithfulness) — `gold/faithfulness_gold.csv`

- **What:** about 30 bodies the owner tagged `multi` that the one-segment-per-company parser cannot split
  (narrative bodies), drawn once (`scripts/draw_faithfulness_2026_11_06.py`, `data/faithfulness_bodies.csv`).
  Each extractor (the mention-and-recency baseline today, the trained model later) writes the text each
  named company gets; the owner reads each text with its article and marks `faithful` (yes / no).
- **Scope:** the file is **blind to the extractor** (rows shuffled, a text two extractors wrote identically
  is one row). The row count is not fixed: it is one row per distinct (body, company) text, known when
  drawn. Gold columns: `review_order`, `article_id`, `ticker`, `extractors`, `pub_date`, `faithful`, `notes`.
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
  `article_id`, `ticker`, `kind`, `part`, `sentiment`, `notes`. The text itself is not in `gold/`; it is
  rebuilt from the pipeline and the `key`.
- **Used for:** the D32 bar (overall agreement >= 80%; precision and recall >= 70% for negative), re-weighted
  by the stratum sizes (`population` in `data/sentiment_gold_ids.csv`). It says whether the pinned
  sentiment model can read summaries and company texts.

## Roadmap: what each set unlocks and the order to tag

Fixed dates are the ones in `PLAN.md` D34 (cut for the 2026-11-20 delivery). The order is by what
each set blocks, earliest block first.

| Order | Set | Unblocks | Needed by | Waits for |
|---|---|---|---|---|
| 1 | **3. Label check** (200 paragraphs) | training the attribution model (`lead` only, or both kinds) | 2026-11-08 | nothing: the training set exists |
| 2 | **4. Extractor sample** (about 30 bodies) | the decision on the extractor for narrative bodies; the extractor part of set 5 | 2026-11-09 | nothing: baseline-only draw (D34 rule b) |
| 3 | **5. Sentiment** (180 texts) | the D32 sentiment bar, so the cut can say whether its sentiment is checked | 2026-11-12 | the extractor decision (its 60 texts come from the extractor chosen) |
| 4 | **2. Scope test** (200 bodies) | the scope measure (rules now, classifier later), the D24 category target | 2026-11-12 (go / no-go) | nothing: it can be tagged any time, but the cut does not wait for its result to start |
| done | 1. Scope train | the labelling prompt, the sample for sets 4 and 5 | | |

Total owner work before the go / no-go of 2026-11-12: 610 items (200 + 200 + about 30 bodies' texts + 180).

Why this order:

- Set 3 is first because the model trains on 2026-11-08 and the rule on the `continuation` labels must be
  applied *before* training; it is the shortest set and needs no other result.
- Set 4 needs only the baseline, so it can be drawn and tagged at once; if the model is not ready on
  2026-11-09 the decision is on the baseline alone.
- Set 5 draws its extractor part from whichever extractor set 4 selected, so it is drawn after.
  Its checked part (120) does not depend on it and can be tagged first if the owner wants to split the work.
- Set 2 is independent of the others, so it fills the gaps; it is last only because nothing downstream
  starts from its result.

## Migrating a finished set (the step the roadmap ends on)

1. The owner marks the `data/` file and runs `scripts/report_wi18_2026_11_06.py <scope|label-check|faithfulness|sentiment>`,
   which refuses a file with unmarked or invalid rows (nothing is read as "no").
2. The set is migrated to its `gold/` file: set 1 and 2 by `build_gold_2026_11_05.py --set train|test`
   (exist); sets 3, 4 and 5 by a builder that does not exist yet and is written with the same rules
   (refuse unless every row is marked and valid, tags only, sorted, the ids of the draw unchanged).
   **Writing those three builders is the open work of this page**: until then those `gold/` files stay
   header-only.
3. From then on features read `gold/`; the `data/` file is kept only as the owner's working copy.
