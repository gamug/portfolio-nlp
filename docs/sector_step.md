# Step 3: the sector (T-167; PLAN D39 (4), D40)

`uv run cli/news_nlp_steps.py --step sector [--limit N] [--scope-revision REV]` (and `--step sector --status`)

## What it does

It reads the bodies step 1 called `sector` and gives each one **a GICS sector (the 2-digit code, 11 classes) or sends it to `out_of_scope`**.
Rows go to `body_sector` (`status` `assigned` / `out_of_scope`, `sector`, `candidate`, `margin`, `vote_sector`, the scores and the votes), state to
`pipeline_step_state` (step `sector`). Like the other steps it is idempotent and resumable: a rerun at the same revision adds nothing, a failed body is
retried and keeps its reason, a batch that fails is redone body by body, three batches in a row that fail completely stop the run, and `--model-dir`,
`--reading` and the other steps' options are refused. Code: `src/sector_stage.py`; the numbers are in `config/sector_step.json`.

## The rule

1. **Score.** The body is cut in chunks of 120 words (the title goes before the first). Each chunk is embedded by the pinned `BAAI/bge-small-en-v1.5`
   (revision in `config/sector_step.json`; CLS pooling). A class scores the mean of its 3 best chunks, each chunk taking its best description. The classes are
   the 11 sectors (5 descriptions each) and four *not a sector* classes (`market_economy`, `investing_general`, `investment_activity`, `unrelated`). A direction
   common to analyst notes ("upgrades the stock, raises the target") is taken out of the chunks first. The **margin** is the best score minus the second.
2. **Votes.** `triage.CompanyResolver` finds the universe companies the text names; each votes for its GICS sector (at most 3 votes per company).
3. **Gate.** `assigned` when the best class is a sector **and** (the margin is at least `0.0293` **or** the companies' top sector is that same sector);
   anything else is `out_of_scope`. The owner's rule (2026-11-13): a wrong sector costs more than a missing one.

The revision is `sector-` plus 12 hex digits of the hash of the config, the descriptions, the universe and this module's code. A command run on the
real data: 64 bodies in 15 seconds on the CPU (4 to 7 bodies a second, so the 32,887 `sector` bodies are about 2 hours on the CPU; it uses CUDA when there is one).

## How it was chosen and what it gets

Labelled data: 70 single-sector bodies of the tagged gold (rolled up to the 2-digit sector, `docs/gold_datasets.md` "Sector tags, 2026-11-12") and the owner's
two yes/no reviews. Everything was chosen on a **tune half** (even positions) and read once on the **test half** (odd positions); the reviews were never used to tune.

| Stage | Result |
|---|---|
| Baseline, descriptions v1, summary or chunks | 70 to 73% of the 71 bodies right (top-1) |
| Descriptions v2 (written after the dev errors) | dev 97%, **held-out 74%**: the descriptions did not generalise |
| First yes/no review, 100 bodies the small model picked | 67% right; 88% on the confident half, 47% on the rest; 10 controls all `no` |
| Why the 33 `no` | about half were not sector bodies (scope errors: market, investing, unrelated); the rest wrong sector, mostly Materials, retail and IT |
| Larger embedders (bge-base/large, gte, e5) | +11 to 15 points on the tagged test half, −12 to −18 on the review (which the small model picked), so not conclusive; not adopted |
| Labelled bodies as examples, more descriptions (v3, v4) | no gain on the test half (v4 fits the tune half only) |
| *Not a sector* classes | precision on the reviewed test half 62 → 70%; 43% of the scope errors caught |
| Company votes as a second opinion | the two agree: 96% right; votes help where the margin is low |

### The clean test (2026-11-13)

Drawn with the final configuration, from bodies in neither the gold nor the first review; the owner answered yes/no (`scripts/draw_sector_fresh_2026_11_13.py`,
scored by `scripts/score_sector_fresh_2026_11_13.py`):

| | n | right | 95% interval |
|---|---|---|---|
| **Assigned: is the sector right** | 40 | **35 (87.5%)** | 74 to 95% |
| &nbsp;&nbsp;let through by the margin | 31 | 28 (90.3%) | 75 to 97% |
| &nbsp;&nbsp;let through by the companies' vote only | 9 | 7 (77.8%) | 45 to 94% |
| Controls (least-similar sector shown) | 5 | 0 yes | the reviewer is not saying yes too soon |
| Sent `out_of_scope`: the owner agrees | 12 | 7 (58%) | 32 to 81% |

The step assigned 66% of the sector bodies it scored (117 of 176 in the draw). The 5 wrong assigned: two were not sector bodies (investing, market), three
were another sector (one is a Big Tech note read as consumer discretionary, one a dollar-store story read as staples, one a food story read as health care, where
the companies voted for the right sector but the margin let the embedding win). Of the 5 bodies sent out that the owner would have given a sector, the step's best guess was
that sector in 4: the gate costs about a sixth of the right sectors (estimate: 20 of about 122 right in the draw) to remove most of the wrong ones.

### What this does not show

- The intervals are wide (40 and 12 bodies). The 87.5% is one measure on one draw by one reviewer, no inter-annotator figure.
- It is the precision of the **sectors assigned**, not the accuracy over all sector bodies: a third of them get none. Nothing here measures the sub-industry (D24, ">= 80%").
- Materials has no clean tagged body in the gold, and the fresh test has 2 to 5 assigned bodies per sector (Materials 2 of 3 right, Consumer Staples 1 of 2). The per-sector figures are not usable.
- The scope step feeds this one: about half of the wrong answers of the first review were not sector bodies at all. Retraining the scope model with those as hard
  negatives would help more than anything inside this step (not done; the scope validation is closed).
- Two tune-half tags (432785 fixed, 481049 not) were found wrong while working; the descriptions were never tuned on the reviews.

## Running it

```
uv run cli/news_nlp_steps.py --step sector --limit 200     # needs step 1's body_scope and SOURCE (the companies)
uv run cli/news_nlp_steps.py --step sector --status        # counts by state, by status and by sector; no model
```

To redo a revision: `DELETE FROM pipeline_step_state WHERE step = 'sector' AND model_revision = '<rev>'` (the rows are written with `INSERT OR IGNORE`, so also
`DELETE FROM body_sector WHERE classifier_revision = '<rev>'`). The next step on an `assigned` body is its summary: `uv run cli/news_nlp_steps.py --step summary --items sector --sector-revision sector-... --limit 200`
(`docs/summary_step.md`; the sector step must have run, and its revision is named, never the latest).
