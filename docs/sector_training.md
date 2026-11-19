# Sector step: what was trained and how it was measured

The sector step (step 4, `docs/sector_step.md`) gives each text the scope step called `sector` one of the 11 GICS sectors, or none. Nothing is trained for it:
it uses a pinned sentence embedder (`BAAI/bge-small-en-v1.5`), the sector votes of the universe companies the text names, four "not a sector" classes and a
gate (revision `sector-417a93feb789`). What is measured is the answer it gives. The design, the attempts and the first clean test are in `docs/sector_step.md`;
this document keeps the measure on the larger set of answers of 2026-11-19. The owner's answers are the only reference; there is no second reviewer.

## The data

| | First draw | Second draw |
|---|---|---|
| Drawn | 2026-11-13 (`scripts/draw_sector_fresh_2026_11_13.py`) | 2026-11-18, same method and revision |
| Rows | 57: 40 assigned, 5 controls, 12 sent out | 300: 165 assigned, 15 controls, 120 sent out |
| Excluded | the tagged gold and the first review | those and the first draw |

The second draw takes **more sent-out rows than the corpus has** (120 of 300, against 44%), on purpose: the gate had been judged on 12. Controls show the
sector the step likes least; a `yes` to one means a reviewer who agrees too soon. The owner answers `correct` (`yes` or `no`) and, when the sector is wrong or
the text was dropped wrongly, `right_sector`. **For a sent-out row `yes` means the drop was right**; `sector_description` on those rows is only the step's best guess.

Working files (local, `data/` is git-ignored): `data/sector_fresh_to_review_es.csv` sha256 `65eb7a4534ef110c32c865f87d7b7430c4a7ea67ea5e0325949c1a95944ff3ac`,
`data/sector_fresh_key.csv` sha256 `6ba854206e6794b72fe79e1bc46fd2996f6945419c19873540ebcb8607ea959c`. **Tracked copy:** `gold/sector_fresh_marks_2026_11_19.csv` (the answers and the step's key, sha256 `907a94c1b5dc95e4c0c51978cab61ddf8a0be718e6ab5e702bd827279976b27c`) and `gold/texts/sector_fresh_marks_texts_2026_11_19.csv` (the English texts). Scorer: `scripts/score_sector_fresh_2026_11_19.py` (reads the tracked copy; `--export` makes it from the working files). A change of an answer is a new file and a new row here, not an edit.
The 300 new texts were translated into Spanish with DeepSeek (approved by the owner) as a reading aid; the English text is what the step reads.

## Consistency of the answers

Checked by the scorer: 357 rows, 357 distinct articles; every answer is `yes` or `no`; every `right_sector` is a GICS code; every verdict agrees with the key's status;
no control was shown the step's own sector. What it found:

1. **Five of the first 57 answers changed since the tracked gold** (`gold/sector_validation.csv`): rows 12, 34 and 50 (sent out) went from `yes` to `no`; rows 20 and 46 went from `no` to `yes`.
2. **Rows 20 and 46 contradict themselves:** answered `yes` (the drop was right), yet with a sector named (10, 50) and the notes "Is a real sector news, not out_of_scope" and "real sector".
   The same note on rows 15, 19 and 24 is a `no`. They look like a flipped answer. They are counted as the owner wrote them; the figure with them as `no` is given beside.
3. **28 sent-out rows are `no` with no sector named** (12, 34, 50, 72, 87, 95, 141, 158, 161, 179, 182, 199, 202, 235, 236, 253, 256, 257, 271, 272, 277, 278, 320, 327, 330, 334, 343, 344).
   A `no` says the step should have kept the text; without the sector the step's guess cannot be checked. Rows 277 and 278 carry notes ("a more diverse topic", "Is like single") that argue for dropping.
4. **A `no` on an assigned row mixes two different errors:** the sector was wrong, or the text is not a sector text at all (the notes say `general_economy`, `multi`, `single`, `out_of_scope`, investing).
   The second is an error of the scope step, not of the sector choice, which is why the second measure below exists. The split is by the note text, a rule in the scorer, not a mark of the owner.

## Results

Wilson 95% intervals. Controls are left out of the assigned rows.

| Measure | 357 rows | Only the 300 new rows | First 57 rows |
|---|---|---|---|
| **Assigned sector is right** | **176/205 = 85.9% [80.4, 90.0]** | 141/165 = 85.5% [79.3, 90.0] | 35/40 = 87.5% [73.9, 94.5] |
| `no` whose note says "not a sector text" | 23 | 21 | 2 |
| **Assigned, without those 23** | **176/182 = 96.7% [93.0, 98.5]** | 141/144 = 97.9% [94.1, 99.3] | 35/38 = 92.1% [79.2, 97.3] |
| `no` that name another sector | 6 | 3 | 3 |
| Let through by the margin (all, without) | 134/150 = 89.3% [83.4, 93.3]; 134/138 = 97.1% [92.8, 98.9] | 106/119 = 89.1%; 106/108 = 98.1% | 28/31 = 90.3%; 28/30 = 93.3% |
| Let through by the vote only (all, without) | 42/55 = 76.4% [63.7, 85.6]; 42/44 = 95.5% [84.9, 98.7] | 35/46 = 76.1%; 35/36 = 97.2% | 7/9 = 77.8%; 7/8 = 87.5% |
| Controls answered `yes` | 0/20 | 0/15 | 0/5 |
| Sent out and the owner agrees | 101/132 = 76.5% [68.6, 82.9] (99/132 = 75.0% with rows 20 and 46 as `no`) | 95/120 = 79.2% [71.1, 85.5] | 6/12 = 50.0% (4/12 with 20 and 46 as `no`) |
| Sent out and the owner would keep | 31; a sector named in 3, the step's guess right in 2 | 25; no sector named | 6; named in 3, guess right in 2 |

**Per sector** (assigned; all, then without the not-a-sector `no`):

| GICS code | Sector | All | Without |
|---|---|---|---|
| 10 | Energy | 18/19 | 18/18 |
| 15 | Materials | 3/5 | 3/4 |
| 20 | Industrials | 21/25 | 21/21 |
| 25 | Consumer Discretionary | 21/25 | 21/22 |
| 30 | Consumer Staples | 5/7 | 5/7 |
| 35 | Health Care | 14/15 | 14/15 |
| 40 | Financials | 30/37 | 30/30 |
| 45 | Information Technology | 30/36 | 30/30 |
| 50 | Communication Services | 17/18 | 17/17 |
| 55 | Utilities | 5/5 | 5/5 |
| 60 | Real Estate | 12/13 | 12/13 |

## One vs rest, per sector

Each class against the rest, over the 337 rows that are not controls (a control is shown a sector the step did not give, so it is not the step's answer). A row's **true class** is what the owner's answer says: the
assigned sector if `yes`; the `right_sector` if one is named; "no sector" if the drop was agreed, or if a `no` says the text is not a sector text. 28 sent-out `no` name no sector, so their true class is unknown.

| GICS code | Sector | Assigned | Right | Precision [95%] | Precision without the not-a-sector `no` | Truly this sector (known) | Found | Recall, upper bound |
|---|---|---|---|---|---|---|---|---|
| 10 | Energy | 19 | 18 | 94.7% [75.4, 99.1] | 18/18 | 18 | 18 | 100.0% |
| 15 | Materials | 5 | 3 | 60.0% [23.1, 88.2] | 3/4 | 3 | 3 | 100.0% |
| 20 | Industrials | 25 | 21 | 84.0% [65.3, 93.6] | 21/21 | 21 | 21 | 100.0% |
| 25 | Consumer Discretionary | 25 | 21 | 84.0% [65.3, 93.6] | 21/22 | 25 | 21 | 84.0% |
| 30 | Consumer Staples | 7 | 5 | 71.4% [35.9, 91.8] | 5/7 | 7 | 5 | 71.4% |
| 35 | Health Care | 15 | 14 | 93.3% [70.2, 98.8] | 14/15 | 14 | 14 | 100.0% |
| 40 | Financials | 37 | 30 | 81.1% [65.8, 90.5] | 30/30 | 30 | 30 | 100.0% |
| 45 | Information Technology | 36 | 30 | 83.3% [68.1, 92.1] | 30/30 | 31 | 30 | 96.8% |
| 50 | Communication Services | 18 | 17 | 94.4% [74.2, 99.0] | 17/17 | 19 | 17 | 89.5% |
| 55 | Utilities | 5 | 5 | 100.0% [56.6, 100.0] | 5/5 | 5 | 5 | 100.0% |
| 60 | Real Estate | 13 | 12 | 92.3% [66.7, 98.6] | 12/13 | 12 | 12 | 100.0% |
| — | **No sector** (sent out) | 132 | 101 | 76.5% [68.6, 82.9] | — | 124 | 101 | 81.5% |

- **Precision** is the share of the texts given that sector that the owner accepts. Wilson intervals are wide (5 to 37 texts per sector); Materials (5), Consumer Staples (7) and Utilities (5) are not usable.
- **Recall is an upper bound and not a measure of the step.** Of the texts known to belong to a sector, "found" counts those the step gave that sector. The draw takes the assigned rows round-robin over the sectors
  and the sent-out rows at random, so it is not a sample of the corpus, and the 28 sent-out `no` that name no sector could belong to any of them (they can only lower the figures). The two sectors under 100% are the ones
  named for a text the step gave another sector or dropped: Consumer Discretionary (4 missed), Consumer Staples (2), Information Technology (1), Communication Services (2).
- **No sector** is the one class with a usable recall: precision 101/132 (the owner agrees to the drop) and recall 101/124 = 81.5%, the 23 texts the step gave a sector although the owner says they are not sector texts.
  Weighted to the corpus rates of `docs/sector_step.md` (56 assigned, 44 sent out per 100): 84.3%.
- The scorer prints the same table (`ovr`).

## What this shows and what it does not

- **The sector choice is right in about 97% of the sector texts it is given** (176 of 182); it is right in 86% of what it assigns once the texts that are not sector texts count against it.
  The 23 are 11% of the assigned rows and sit mostly in Financials (7), Information Technology (6) and Industrials (4): the scope step let them through as `sector`.
- **The two draws agree** on the assigned precision (87.5% and 85.5%); the first 57 were not a lucky draw. The earlier 87.5% / 89.7% of `docs/sector_step.md` stand as the first draw's.
- **The vote-only gate is the weaker part:** 76% of the rows it lets through are right (96% without the not-a-sector ones), against 89% (97%) for the margin.
- **Controls: 0 of 20 `yes`**; the reviewer is not saying yes too soon.
- **The gate:** the owner agrees to 76.5% of the drops (75% under the stricter reading of rows 20 and 46). In the second draw 25 of 120 drops were wrong and **none names the sector**, so
  the share of the right sectors the gate gives up cannot be measured there. With the corpus rates of `docs/sector_step.md` (56 assigned, 44 sent out per 100 `sector` texts) and
  every wrong drop counted as a lost right sector (an upper bound), about 11 of 44 are lost against about 48 right assigned: **at most about a fifth** of the available right sectors, as before.
- **Not measured:** recall per sector, the sub-industry, the D24 bar of 80% for it (T-167). Per-sector counts are 5 to 37 and the intervals are wide; Materials (5) and Consumer Staples (7) are not usable.
- **One reviewer**, no inter-annotator figure; the owner's notes ("more like multi", "very ambiguous") show that the boundary with the scope classes is itself unclear.

## Open points for the owner

- Confirm or correct rows 20 and 46.
- Name the sector (or say there is none) for the 28 `no` sent-out rows listed above; that is what turns the gate's cost from an upper bound into a figure.
