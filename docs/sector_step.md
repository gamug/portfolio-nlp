# The sector step (T-167; PLAN D39 (4), D40)

`uv run cli/news_nlp_steps.py --step sector [--limit N] [--scope-revision REV]` (and `--step sector --status`)

## What it does

It reads the bodies step 1 called `sector` and gives each one **a GICS sector (the 2-digit code, 11 classes) or sends it to `out_of_scope`**.
Rows go to `body_sector` (`status` `assigned` / `out_of_scope`, `sector`, `candidate`, `margin`, `vote_sector`, the scores and the votes), state to
`pipeline_step_state` (step `sector`). Like the other steps it is idempotent and resumable: a rerun at the same revision adds nothing, a failed body is
retried and keeps its reason, a batch that fails is redone body by body, three batches in a row that fail completely stop the run, and `--model-dir`,
`--reading` and the other steps' options are refused. Code: `src/sector_stage.py`; the numbers are in `config/sector_step.json`. The model,
`BAAI/bge-small-en-v1.5`, is **MIT-licensed** (model card at the pinned revision; the repository declares no licence of its own) and `python -m setup` fetches it.

Before the first batch the command says how many bodies it will do (`pending N sector bodies at revision ...`; finding them scans the whole corpus, about
half a minute on the scratch copy) and loads the model, so the progress bar's rate and time left count bodies only.

## The rule

1. **Score.** The body is cut in chunks of 120 words (the title goes before the first). Each chunk is embedded by the pinned `BAAI/bge-small-en-v1.5`
   (revision in `config/sector_step.json`; CLS pooling). A class scores the mean of its 3 best chunks, each chunk taking its best description. The classes are
   the 11 sectors (5 descriptions each) and four *not a sector* classes (`market_economy`, `investing_general`, `investment_activity`, `unrelated`). A direction
   common to analyst notes ("upgrades the stock, raises the target") is taken out of the chunks first. The **margin** is the best score minus the second.
2. **Votes.** `triage.CompanyResolver` finds the universe companies the text names; each votes for its GICS sector (at most 3 votes per company).
3. **Gate.** `assigned` when the best class is a sector **and** (the margin is at least `0.0293` **or** the companies' top sector is that same sector);
   anything else is `out_of_scope`. The owner's rule (2026-11-13): a wrong sector costs more than a missing one.

**The revision** is `sector-` plus 12 hex digits of the hash of the config, the descriptions, the universe and this module's code **as parsed**
(`news_summary_stage.normalized_code`, as in the summary step): a comment or a docstring does not change it, a constant or an operator does.
The clean test below was drawn under `sector-cf81d9b32565`, when the hash still covered the raw bytes of the file; `scripts/check_sector_validation_2026_11_13.py`
re-scores the 57 tracked texts with the code as it is (`sector-ccc1b0f4e093` on 2026-11-13, with the universe of the SOURCE copy used) and finds the same status,
candidate, vote and margin (within 0.0001) in **57 of 57** rows **on the text the owner read, which is capped at 6,000 characters** (see "The text" below). Run it again after any change that moves the revision.

**The text.** The step classifies the **whole body**. The draw of the clean test, and so the texts the owner read and `gold/texts/sector_validation_texts.csv`, hold only the first 6,000 characters of
each body (`TEXT_CAP`). 5 of the 57 bodies are longer. The check script also reads the 57 bodies whole from SOURCE: **1 of 57 verdicts differs** (article 464913: `out_of_scope` on the capped text, `assigned` on the
whole body; the owner had agreed to drop it, so that answer does not carry over), and **all 40 bodies the owner judged as assigned are still assigned on the whole body**, so the 87.5% holds for what the step does.
The random-sample rate below was measured on the whole body. A chunk of 120 words is far under the embedder's 512-token window: 0 of the 296 chunks of the 57 bodies reached it (the longest has 264 tokens); the embedder
counts any chunk that does, and a run prints a warning and reports `truncated_chunks` (constitution, Pipeline models 3).

**Speed** (the same 64 bodies, the model loaded, measured 2026-11-13): GPU (RTX 4050 Laptop) 1.4 s = **45 bodies a second**, so the 32,887 `sector` bodies are about **12 minutes**;
CPU 14.2 s = **4.5 bodies a second**, about **2.0 hours**. The verdicts are the same on both (64 of 64) and the margins differ by less than 0.00001. Loading the model and embedding the descriptions takes about 4 s.
An earlier version of this page quoted "15 seconds on the CPU" for a first run: it was the GPU, with the model load inside it.

**How many get a sector.** A random sample of 1,500 of the 32,887 `sector` bodies, scored to the end (`scripts/sector_rate_sample_2026_11_13.py`, seed 20261113): **851 assigned = 56.7%
(95%: 54 to 59%)**, 649 `out_of_scope`. The sectors given: Financials 190, Energy 161, Information Technology 133, Consumer Discretionary 100, Real Estate 73, Industrials 53, Health Care 49,
Communication Services 39, Materials 22, Utilities 16, Consumer Staples 15. What the best class was for the 649 sent out: `market_economy` 137, `investment_activity` 73, `investing_general` 14,
`unrelated` 3 (227 bodies the step reads as not a sector at all), and a sector for the other 422 (a close call that the companies did not back). The 66% an earlier version quoted was a progress line of a draw that stopped when it had enough of each kind, and
the 70% of the first 64 real bodies came from the first rows of the table (34 of the 45 were IT); neither is a rate.

## How it was chosen, as it happened

- The **sector descriptions** were written on the tune half of the labelled bodies, and rewritten after reading its errors (v2).
- The four **not a sector** classes were written from the owner's notes on the first yes/no review.
- The **margin threshold** is the 40th percentile of the margins of the tune half, which holds both the tagged bodies and the reviewed rows.
- The **gate** (`margin or agree`) was chosen after the tune and test tables had been printed in the same run; it is a judgement (it keeps 68% of the bodies against 53% for the margin alone, for 4 points of precision on the test halves).
- The **test halves were read many times** (every variant below was scored on them). They record how the choices were made; they are not a clean measure.
- The only clean figure is the owner's review of 2026-11-13 (below), drawn with the step as built.

Labelled data: 70 single-sector bodies of the tagged gold (rolled up to the 2-digit sector, `docs/gold_datasets.md` "Sector tags, 2026-11-12") and the 100 real rows of the first yes/no review,
each split by position (even = tune, odd = test).

### The attempts (all rows reproduced by `scripts/sector_evidence_2026_11_13.py`, from the tracked gold)

"Precision" is the share right among the bodies a sector class won; "accuracy" is top-1 on the known labels. A = tagged bodies, B = the first review (its rows were picked, shown and balanced
with the small model and a `yes` means "its sector is right": B favours it, so a model that picks differently looks worse there).

| Attempt | Tune half | Test half (A, B) |
|---|---|---|
| Descriptions v1 (the baseline script), 71 bodies | 70 to 73% top-1 | |
| Descriptions v2, no reject classes | A 100% | A precision 74.3%, B 62.0% |
| **v2 + reject classes (current)** | A 100%, B 80.0% | **A 75.8%, B 70.5%**; 43% of the scope errors caught |
| v3 (from the GICS structure) + reject | A 94.1%, B 66.0% | A 71.9%, B 68.2%: worse |
| v2 + v3 + reject | A 97.1%, B 74.5% | A 74.3%, B 67.4%: no gain |
| v4 (v2 + lines for the tune errors) + reject | A 100%, B 84.8% | A 75.8%, B 70.5%: only the tune half moved |
| The tune half's labelled bodies as examples | B 80.0% → 70.0% (weight 0.5), 62.0% (weight 2) | not scored: worse on tune |
| Company votes alone | | A 79.0% right, B 50.0%; 95% of bodies name a universe company |
| Embeddings and votes agree | | 71% of A and 44% of B, right then 95.5% and 97.2% |
| Gate: margin only | | A+B kept 53%, precision 86.7% |
| Gate: votes agree only | | kept 45%, 86.8% |
| **Gate: margin or agree (current)** | | **kept 68%, 82.8%** |

Larger embedders, same descriptions and classes (TEST half):

| Embedder | A precision / accuracy | B precision / accuracy | Gate, A+B kept / precision |
|---|---|---|---|
| bge-small (pinned) | 75.8% / 71.4% | 70.5% / 77.5% | 53% / 86.7% |
| bge-base | 90.6% / 82.9% | 55.6% / 62.5% | 48% / 82.9% |
| bge-large | 86.7% / 74.3% | 58.1% / 62.5% | 51% / 81.4% |
| gte-base | 90.3% / 80.0% | 52.3% / 57.5% | 54% / 82.6% |
| e5-base | 86.7% / 74.3% | 55.0% / 55.0% | 52% / 79.5% |

On A the larger models gain 11 to 15 points of precision but only 3 to 11.5 of accuracy (bge-large +2.9); on B they lose 12 to 18 of precision and 15 to 22 of accuracy, partly the bias above.
Not conclusive, not adopted (they also cost more); a fair comparison needs rows no model picked.

### The first review (2026-11-12)

100 real rows and 10 controls, picked with the small model alone: **67% right**; 43 of the 49 that the first draw flagged `sure` (88%) against 24 of the 51 it did not (47%); the 10 controls all `no`. `sure` was a margin of at least **0.0324** (25th percentile of the dev half) on the older scorer, which had no reject classes and took the margin over the 11 sectors only: it is **not the step's gate** (0.0293, 40th percentile, over 15 classes; 51 of those 100 rows pass it on the margin alone).
Of the 33 `no`, about half were not sector bodies at all (market, investing, unrelated: scope errors), the rest another sector, mostly Materials, retail and IT.

### The clean test (2026-11-13)

Drawn with the final configuration, from bodies in neither the gold nor the first review; the owner answered yes/no (`scripts/draw_sector_fresh_2026_11_13.py`,
scored by `scripts/score_sector_fresh_2026_11_13.py` from the tracked `gold/sector_validation.csv`, set 12 of `docs/gold_datasets.md`). The rows were drawn by quota
(40 assigned, 12 sent out, 5 controls), so the shares in it are not the rates of the corpus (see "How many get a sector" above):

| | n | right | 95% interval |
|---|---|---|---|
| **Assigned: is the sector right** | 40 | **35 (87.5%)** | 74 to 95% |
| &nbsp;&nbsp;let through by the margin | 31 | 28 (90.3%) | 75 to 97% |
| &nbsp;&nbsp;let through by the companies' vote only | 9 | 7 (77.8%) | 45 to 94% |
| Controls (least-similar sector shown) | 5 | 0 yes | the reviewer is not saying yes too soon |
| Sent `out_of_scope`: the owner agrees | 12 | 7 (58%) | 32 to 81% |

The 5 wrong assigned: two were not sector bodies (investing, market), three were another sector (a European internet note read as consumer discretionary, a dollar-store story read as staples,
a food story read as health care, where the companies voted for the right sector but the margin let the embedding win). Of the 5 bodies sent out that the owner would have given a sector, the step's best guess was
that sector in 4.

What the gate costs, with the corpus rates (an estimate on 12 bodies, so very wide): per 100 `sector` bodies, 56.7 are assigned and 87.5% of them are right (about 50); the 43.3 sent out would
have had a right sector in 4 of 12 cases (about 14). The gate gives up roughly **a fifth** of the right sectors available (14 of 64) to avoid most wrong ones.

### What this does not show

- The intervals are wide (40 and 12 bodies). The 87.5% is one measure on one draw by one reviewer, no inter-annotator figure.
- It is the precision of the **sectors assigned**, not the accuracy over all sector bodies: **43% of them get none**. Nothing here measures the sub-industry or the D24 bar (">= 80% against the held-out gold"), which stays open in T-167.
- Materials has no clean tagged body in the gold, and the fresh test has 2 to 5 assigned bodies per sector (Materials 2 of 3 right, Consumer Staples 1 of 2). The per-sector figures are not usable. In the random sample, Consumer Staples (15), Utilities (16) and Materials (22) are the sectors the step gives least.
- The votes rows of the attempts table use `gold/universe.csv` (507 tickers); the step reads the 548 company rows of SOURCE. On the 170 labelled items 2 vote vectors differ and no verdict changes.
- The scope step feeds this one: about half of the wrong answers of the first review were not sector bodies at all. Retraining the scope model with those as hard
  negatives would help more than anything inside this step (not done; the scope validation is closed).
- Two tune-half tags were questioned while working. 432785 (Simon Property, tagged consumer staples) was wrong and is fixed. 481049 ("Wells Fargo Bets on Housing Recovery", tagged Real Estate by the owner, with a note) is ambiguous between Financials and Real Estate and is left to the owner; the model also says Real Estate. Dropping it would move the threshold from 0.0293 to 0.0284 and change no verdict of the clean test (no row has a margin between the two).

## Running it

```
uv run cli/news_nlp_steps.py --step sector --limit 200     # needs step 1's body_scope and SOURCE (the companies)
uv run cli/news_nlp_steps.py --step sector --status        # counts by state, by status and by sector; no model
```

To redo a revision: `DELETE FROM pipeline_step_state WHERE step = 'sector' AND model_revision = '<rev>'` (the rows are written with `INSERT OR IGNORE`, so also
`DELETE FROM body_sector WHERE classifier_revision = '<rev>'`). The next step on an `assigned` body is its summary: `uv run cli/news_nlp_steps.py --step summary --items sector --sector-revision sector-... --limit 200`
(`docs/summary_step.md`; the sector step must have run, and its revision is named, never the latest).

To reproduce the figures of this page, in order:

```
uv run scripts/sector_rollup_2026_11_12.py                     # data/sector_rollup.csv, from gold/
uv run scripts/sector_evidence_2026_11_13.py                   # the attempts, the votes, the gates, the embedders (needs the models in the HF cache)
uv run scripts/score_sector_fresh_2026_11_13.py                # the clean test, from gold/sector_validation.csv
uv run scripts/check_sector_validation_2026_11_13.py           # the current code gives the verdicts the owner saw (needs SOURCE for the companies)
uv run scripts/sector_rate_sample_2026_11_13.py                # the share assigned, on a random sample (needs RESULTS and SOURCE)
```
