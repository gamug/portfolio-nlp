# The company-text extractor (step 4): datasets, experiments, metrics, results

This page is the plain-language record of how the step-4 extractor was built and checked, and of the
selection of its base model. It is written for the owner and for anyone who has not followed the work.
The technical contract stays in `.specify/memory/PLAN.md` (D27, D32, D34) and `TASKS.md` (T-148b*);
the commands are in `docs/evaluation.md`; the tagged files are described one by one in
`docs/gold_datasets.md`. When this page and those disagree, those win, and this page is wrong.

**Which figures are current.** The selection below, "Why `units` and not the baseline", "The results" and
"What the results say" give the figures read on 2026-11-08, before the matcher change of PR #155
(`units` on the confirmation set: 86.1% on every company named, 570 characters per text). They are kept
as the record of what the owner decided on. The figures after the change are in "Update of 2026-11-10"
(`units`: 87.3% counting every mark, 97.1% on texts the current code writes; 620 characters per text).

## The selection (owner, 2026-11-08)

**The owner selected `units` (called "units1" in conversation) as the base model of step 4.** The reason,
in the owner's words: the accuracy does not need to be higher than 80%, and `units` clears it.
"Clears it" means: on the confirmation set, the figures to trust, 86.1% [80.9, 90.7] on every company named
(pass) and 95.3% [92.0, 98.2] on real companies (pass); on the first set (development figures) it passes on
real companies, 94.0% [87.0, 99.2], and is *not decided* on every company named, 83.2% [72.7, 91.0].

What the selection is, and what it is not:

| It is | It is not |
|---|---|
| the extractor the cut (T-154) uses for the company texts of narrative bodies with several companies | a publication to the Hub |
| the base that later improvements start from (see "What is next") | a pin of a model revision |
| a decision of the owner, recorded where the plan and the tasks can see it | a claim that it is better than the baseline: it is not (see "Results") |

Publishing and pinning the trained model still need the owner's explicit word (constitution AI #1, #7,
#12). Nothing here publishes or pins it.

### Why `units` and not the baseline (the reasons, with their figures)

The faithfulness figure does not separate them on the confirmation set, so the selection rests on
other things. They are the reasons that can be read in this page; the choice is the owner's.

1. **Shorter texts for the same faithful share.** 570 characters on average against 952 (the baseline
   writes 385 more per company, [+248, +540]); both pass the bar there (86.5% and 86.6%).
2. **Less text of other companies.** The share of a text's sentences that name the company and another is
   6.1 points lower [+0.1, +12.4] on the confirmation set and 7.3 [-1.8, +16.4] on the first. A lean, not
   a firm gap.
3. **It did better on the set it was built on** (+24.3 points [12.9, 39.4], `better`), which did *not*
   repeat on new bodies (-0.5 [-1.5, 0.0], not decided). Read it as: not worse, not shown better.

What the evidence does not cover:
- **The margin on the strict reading is one to three marks.** The lower end of `units` is 80.9%.
  Flipping one `yes` of the 193 texts to `no` (three random choices) gave a lower end of 80.0, 80.4 and
  79.9; flipping two gave 79.5, 80.0 and 79.5; three always gave a verdict of *not decided*. "Pass" holds on
  the owner's reading as it stands and is not safe against a second reader.
- **2022+, the period that will be processed, cannot be decided**: `units` is *not decided* on the
  strict reading in both sets (87.9% [68.8, 100.0] on 7 bodies; 88.1% [75.8, 95.3] on 9 bodies). On real
  companies it passes both (100.0% and 96.4%), on the same few bodies.
- **Nothing in the code uses the selection yet** and **the model it needs is a local checkpoint**: see
  "What is not done" below.

### What is not done

- **The selection is a record, not a code change.** `attribution_model` still defaults to
  `reading="sentences"` (the baseline), and the cut script (`cut_summaries_2026_11_07.py`) calls no
  extractor; only the draw script (`draw_faithfulness`) runs `units`. Wiring `units` into step 4 (a
  default or a setting, and the cut that calls it) is T-148b12, **proposed, not built, not started**.
- **The checkpoint is not published or pinned.** The model every `units` figure here was measured with is
  the local directory `data/models/attribution_2026_11_08` (git-ignored), identified by the draw as
  `local:8399ccf003c60143` (a hash of its files), trained by `scripts/train_attribution_2026_11_08.py`
  (see "Experiments", 2). The extractor revisions recorded with the marks are in
  `data/faithfulness_stale.json`. Until the owner decides on the Hub publication, the figures belong to
  that directory; if it is lost, they cannot be reproduced.

## What the extractor does, in one paragraph

A news article often talks about several companies. The sentiment of the article is not the sentiment of
each company, so step 4 writes, for every company the article names, **a short text made of the parts of
the article that are about that company**. Later steps (sentiment, category) read that text, not the whole
article. The question this page answers is: *are those short texts right?*

## Words used on this page

| Word | Meaning |
|---|---|
| body | one article |
| text | the short piece an extractor writes for one company of one body |
| extractor | a way of producing the texts. The five below |
| faithful | the owner's mark `yes`: the text is about that company, even if it also mentions others. `no`: it is mainly about another company, or the company is only named in passing |
| the bar | the pass mark: **80%** of the texts faithful (D32) |
| interval | the range the true value probably lies in, for example [80.9, 90.7]. The set is only 30 bodies, so a figure is not exact. It is computed by redrawing the 30 bodies many times, because texts of one body are not independent |
| pass / not decided / fail | pass: the whole interval is above 80%. Not decided: it straddles 80%. Fail: it is wholly below |
| "better" | the gain over the baseline: the whole interval of the difference is above zero |
| real company | a name the owner read and marked `real_company: yes`. A "non-company" is a person, a place, a common word or a legal term the name finder took for a company ("Federal", "Wells" a person, "Amazon" the rainforest) |

## The five extractors

All of them cut the body into pieces and decide which company each piece belongs to.

| Name | How it cuts | How it decides | Built in |
|---|---|---|---|
| `baseline` | sentences | a sentence belongs to the companies it names; a sentence naming nobody goes to the companies of the sentence before it. No model | #141 |
| `model` | sentences | same, but a sentence naming nobody goes to the company the trained model scores highest (at least 0.5), else to nobody | #141 |
| **`units`** | lines, bullets and clauses; the site footer is dropped | the trained model, as above | #143 |
| `units2` | as `units`, plus a **guard** | a short form of a name ("Bloom") counts only if the full name or a ticker of the same company is in the body; a line that opens with names and a dash ("Joy Global — ...") belongs to those companies | #144 |
| `units3` | as `units2` | as `units2`, and a piece of such a line also belongs to the companies it names itself | #144 |

The **trained model** is one model used by the last four: a pair classifier ("is this sentence about
this company?") fine-tuned on the encoder of the sentiment model, with the company name masked and the
previous sentence as context. The four differ only in how the body is cut and who may own a piece.

## The tagged datasets

Everything the owner tagged for this extractor, and what each file is for. Counts are those of
2026-11-08.

| Dataset | What the owner tagged | Size | Used for | Where |
|---|---|---|---|---|
| **Training labels** (not tagged by the owner) | nothing: labels come from bodies written as one section per company | 33,624 paragraphs of 2,834 bodies: 21,682 `lead` and 11,942 `continuation` labels | training the model | `gold/company_segments_train.csv` (ids and positions only) |
| **Label check** (set 3) | whether each automatic label is right | 200 paragraphs (100 `lead`, 100 `continuation`) | deciding which labels train the model | `gold/segment_check_gold.csv` |
| **Faithfulness, first set** (set 4) | `faithful` yes / no for each text | 30 narrative bodies, 437 texts (baseline 190, model 190, `units` 185, `units2` 135, `units3` 148, identical texts counted once); 336 yes, 101 no | development: the extractors were improved looking at this set | `gold/faithfulness_gold.csv` (the 190 baseline texts) and `data/` |
| **First set, companies read** | `real_company` yes / no for the 37 names the stricter readings set aside | 37 names: 23 not companies, 14 companies | removing non-companies from the figures | `data/faithfulness_exempt_to_read.csv` |
| **Faithfulness, confirmation set** | `faithful` yes / no | 30 other bodies, 395 texts: 357 for `baseline` and `units`, plus 38 more that only `units2` / `units3` write; 345 yes, 50 no | the check on bodies the extractors were **not** designed on | `data/faithfulness_confirmation/` |
| **Confirmation, companies read** | `real_company` yes / no | 30 names: 25 not companies, 5 companies | as above | `data/faithfulness_confirmation/faithfulness_exempt_to_read.csv` |

How the sets were drawn:
- The first set is 30 narrative bodies with several companies (the kind the baseline struggles with),
  drawn once.
- The confirmation set is 30 more such bodies, none used before. **The pool is finished**: 32 were left, 30
  drawn, so no third set of this kind can be drawn.
- In both files the extractor is hidden and the rows are shuffled, so the owner tags blind.

Not for this extractor (listed so they are not confused with it): the scope sets (1, 2), the sentiment
gold (5) and the sentiment training set (6, `gold/sentiment_train.csv`, #145).

## The experiments, in order

1. **Is the training label any good?** The automatic `lead` labels were right 88.0% of the time
   [80.2, 93.0]; `continuation` only 38.0% [29.1, 47.8]. The rule fixed beforehand said to drop
   `continuation` below 80%, so the model trains on `lead` only (D27).
2. **Train the model** (#141): 19,724 `lead` paragraphs, 74,512 training pairs, 2 epochs, about 15
   minutes on the local GPU. On a held-out part of the training data (270 bodies) it ranks the right
   company above a wrong one 90.1% of the time, against 63.7% for the mention-and-recency baseline. It
   also gives a sentence to the right company 86.4% of the time on the sentences that name nobody.
   These are automatic figures, on the parser's labels, so they do **not** decide anything.
3. **First faithfulness reading** (first set, the owner's marks). The baseline failed the bar, 60.5%
   [45.2, 73.8]. Cutting finer (`units`, #143) and adding a guard (`units2`, `units3`, #144) were
   designed looking at the failures of this set.
4. **Reading the names the stricter readings set aside** (37 names, the owner's marks): 23 were not
   companies. This showed that part of every extractor's errors are names that are not companies.
5. **Confirmation** (second set, bodies not used to design anything). This is the figure to trust more.
   Run twice: first with `baseline` and `units`, then with `units2` and `units3` added to the same file.

## The metrics

Everything is computed by `uv run scripts/report_wi18_2026_11_06.py faithfulness --out <dir>`. The full output of both sets is in Appendix A and every text metric in Appendix B.

- **The bar (D32):** of all the companies named in the bodies, the share that got a faithful text. A
  company the extractor wrote no text for counts as **not** faithful. Example: `units`, confirmation:
  "167 faithful of 193 texts + 0 companies left without one" = 86.5%.
- **Of the texts it wrote:** the same without the omissions. Informative; it can flatter an extractor
  that stays silent when unsure.
- **Every company named** (the strict reading, what D32 literally says): a name the stricter readings
  skip counts against them as if it were a company. It is the harshest on `units2` and `units3`.
- **Real companies:** the same after removing the names the owner marked as not companies.
- **Gain:** the difference between two extractors' shares on the same companies and bodies, with its
  interval ("units minus baseline").
- **2022+:** only bodies from 2022 on (the period that will be processed). Only 9 bodies per set, so it
  is informative and cannot decide.
- **Text measures** (added 2026-11-08, `scripts/text_measures_2026_11_08.py`; every figure in Appendix B): average characters per
  text, and the share of sentences that name this company **and** another. They count names the resolver
  finds; a nameless sentence that belongs to someone else is invisible to them.
- **Current code** (2026-11-10, PR #157): the report also prints the same figures without the marks of rows whose
  text the code no longer writes, labelled `[current code: ...]` (see "Update of 2026-11-10"). Both views are in Appendix A.3 to A.6 and B.3 to B.6.

## The results

All figures use the owner's marks. Intervals in brackets. The tables in this section and in "What the
results say" are the figures of 2026-11-08; the figures after PR #155 are in "Update of 2026-11-10".

### First set (development figures: the extractors were designed on these bodies)

On the 167 real companies (the 23 non-companies removed):

| Extractor | Faithful | Verdict |
|---|---|---|
| `baseline` | 68.9% [53.6, 82.1] | not decided |
| `model` | 86.2% [71.7, 97.5] | not decided |
| **`units`** | **94.0% [87.0, 99.2]** | pass |
| `units2` | 77.8% [64.1, 92.9] | not decided |
| `units3` | 84.4% [76.4, 93.3] | not decided |

On every company named, `units` was 83.2% [72.7, 91.0] (not decided) and the baseline 60.5% [45.2, 73.8].

**Staleness caveat (re-checked 2026-11-08, round 8, with the model `local:8399ccf003c60143`):** 8 marked
rows of the first set are on texts the current code no longer writes: for `units` the two texts of body
122960 (Apple and Amazon; the Apple text is 32 characters longer now), and the same two plus the Amazon
text of body 88621 (which `units2` and `units3` no longer write at all) for `units2` and `units3`. The
code of the cut changed after the owner read them (rounds 1 and 2 of this PR, dash and dateline rules);
I did not isolate which commit changed which text. The marks were `yes` for the `units` texts, so the effect
is small: counting those 2 of `units`' 185 texts as unfaithful, it would be 82.1% on every company named
(156 of 190) instead of 83.2%, and 92.8% on real companies (155 of 167) instead of 94.0%. The other 183
rows of `units`, and every row of the confirmation set, are current (0 stale there).

### Confirmation set (30 new bodies, the figures to trust)

| | `baseline` | `units` | `units2` | `units3` |
|---|---|---|---|---|
| Every company named | 86.6% [80.7, 91.3] pass | **86.1% [80.9, 90.7] pass** | 78.4% [72.5, 83.7] not decided | 79.9% [74.0, 85.1] not decided |
| Real companies only | 95.9% [92.2, 98.7] pass | **95.3% [92.0, 98.2] pass** | 89.9% [85.1, 94.5] pass | 91.7% [87.4, 95.5] pass |
| Real companies left without a text | 0 | 0 | 12 | 7 |
| Texts written for non-companies | 25 | 24 (18 unfaithful) | 0 | 0 |

Gains on real companies: `units` minus `baseline` -0.6 [-1.7, 0.0] (not decided); `units3` minus `units`
-3.6 [-7.6, -0.6] (not better); `units2` minus `units` -5.3 [-10.5, -1.2] (not better).

2022+ (informative, 9 bodies): `units` 88.1% [75.8, 95.3] on every company named (not decided), 96.4%
[90.9, 100.0] on real companies (pass); the baseline 89.6% [76.2, 96.1] and 98.2% [92.3, 100.0] on the texts
it wrote (it left no company without one).

### What the results say

1. **`units` passes the 80% bar on the confirmation set**, on the strict reading and on real companies.
   On the first set it passes on real companies and is *not decided* on the strict reading (83.2%
   [72.7, 91.0]). This is the basis of the selection.
2. **The margin on the strict reading is thin** (the interval's lower end is 80.9%). It is a pass, not a
   comfortable one.
3. **`units` is not better than the baseline on the confirmation set**; they are level. On the first set
   it looked about 25 points better, which did not repeat on new bodies. The baseline already passes on
   these 30 bodies.
4. **Most of `units`' errors are names that are not companies**: of its 26 unfaithful texts on the
   confirmation set, 18 are for non-companies and 8 for real companies. The guard readings avoid the
   non-companies but leave real companies without a text (7 and 12), so they come out lower, not higher.
5. **The faithful rule is lenient about mixing**: a text that also discusses other companies can still be
   `yes`. So the faithfulness figure does not measure how much of a text belongs to someone else.

### Text measures (confirmation set, 395 texts)

| | `baseline` | `units` | `units2` | `units3` |
|---|---|---|---|---|
| Mean characters | 952 | 570 | 642 | 627 |
| Median characters | 474 | 273 | 278 | 273 |
| Sentences that name the company | 42% | 60% | 60% | 60% |
| Sentences that name nobody | 58% | 40% | 39% | 39% |
| Sentences naming this company and another | 21.3% | 22.3% | 23.5% | 24.1% |
| Other companies named per text | 1.79 | 1.09 | 1.25 | 1.25 |

The baseline writes texts 1.7 times longer than `units` (385 characters more per company, [+248, +540]),
mostly sentences that do not name the company. Yet the faithfulness verdicts of the two agree on 192 of
193 shared companies (30 of them have the *same* text from both extractors, which agree by construction;
of the 163 with different texts, 162 agree: 140 `yes`/`yes`, 22 `no`/`no`, 1 baseline `yes` and `units`
`no`), so length alone did not buy the baseline extra `yes` marks. On the first set it did differ:
of the 140 companies with different texts, the baseline had 45 `no` where `units` had `yes`, and none the
other way; that is where the first-set gain came from. `units` reaches the
same faithful share with about 40% less text.

### Update of 2026-11-10: the matcher change (PR #155) and the new marks

PR #155 changed `CompanyResolver`, the matcher that finds which company a sentence names. `baseline` and
the three `units` extractors read the companies of a body through it, and so does `model` (it uses the
matcher to find the companies a sentence names and to mask the company it scores; `attribution_model.py`).
Their texts changed; the trained model was not touched. The owner then marked the texts that were
missing marks: the first set has 453 marked rows (437 before) and the confirmation set 438 (395 before),
so 16 and 43 rows are new (16 and 40 of them carry a text that was not in the earlier files; 3 rows share an
earlier text under another key). Everything below is a read of those files with the code of `a568b97`.

**What changed in the texts.** A marked row is "dead" when the code no longer writes its text for any of
the extractors that held it. The record used for every count in this section (97 / 96 rows, 147 / 116 marks,
the replacement tables) is `faithfulness_stale.json` as `draw_faithfulness --stale` wrote it on 2026-11-10 with
the code of `a568b97`, the model `local:8399ccf003c60143` and the revisions it prints (`units`
`extractor:units@local:8399ccf003c60143@names:15e088eb@t0.5@21921ab953b1`; the other four are in the
file); a first read-only run on 2026-11-09, outside `data/`, gave the same 97 and 96 rows. The Appendix A
lines taken before that (the staleness line of A.1, A.2, A.5, A.6) come from the record of 2026-11-08: **97 rows of the first set and 96 of the confirmation set** (103 and 97 rows are
dead for at least one extractor, 147 and 116 extractor marks). The 8 rows of the staleness caveat above are
the record of 2026-11-08, before the change; the rest became dead after it (the commit that changed each
text was not isolated). Counts in this section are rows, each row once whatever the number of extractors
that hold it. By the owner's marks the dead texts were unfaithful about half the time: 51 `no` of 97 (53%)
in the first set and 45 of 96 (47%) in the confirmation set, against 14.0% (50 of 356) `no` on the rows the
code writes now (kept rows plus the 16 new ones) in the first set and 2.0% (7 of 342, with the 40 new
ones) in the confirmation set; over every marked row of the first set it is 22.3% (101 of 453).

**What replaced the dead texts.** The earlier ids files are compared with the current ones: a row is *new*
when its text is not in the earlier file. A dead mark counts as replaced only by a new row **of the same
extractor** for the same article and company (a new row of another extractor for that company is not a
replacement). Command in "The official report counts those marks" below.

| Rows | First set | Confirmation set |
|---|---|---|
| Dead | 97 (51 `no`, 46 `yes`) | 96 (45 `no`, 51 `yes`) |
| Kept (still written, not new) | 340 (50 `no`) | 302 (5 `no`) |
| New (a text not in the earlier file) | 16 (16 `yes`) | 40 (38 `yes`, 2 `no`) |

By key the confirmation file holds 43 new rows; 3 of them repeat a text that was already in the earlier file under another key, so they are not counted as new here. The new rows are therefore 16 + 40 = 56.

| Extractor | Dead marks | With a new row of its own (old mark to new mark) | Texts it writes that nobody has marked |
|---|---|---|---|
| First set `baseline` | 40 | 12 (4 `no` to `yes`, 8 `yes` to `yes`) | 0 |
| First set `model` | 41 | 0 | 10 |
| First set `units` | 52 | 4 (4 `yes` to `yes`) | 0 |
| First set `units2` / `units3` | 7 / 7 | 0 / 0 | 0 / 0 |
| Confirmation `baseline` | 51 | 24 (1 `no` to `no`, 23 `yes` to `yes`) | 0 |
| Confirmation `units` | 49 | 10 (1 `no` to `no`, 9 `yes` to `yes`) | 0 |
| Confirmation `units2` / `units3` | 7 / 9 | 4 / 4 (`yes` to `yes`) | 0 / 0 |
| Confirmation `model` | not marked | not marked | 73 (the extractor has no marks here) |

The last column is the `new` list of the stale record (texts the code writes that no row holds), not a count of this method. For `model` an unmarked text cannot be counted as a replacement, so its 0 of 41 is a lower bound, not a finding.

Reading: for the selected extractor the change mostly removed texts: of `units`' 52 and 49 dead marks, 4 and
10 have a text of its own written in their place (all faithful but one). For the other dead marks the files
cannot say whether the code now writes no text for that company or writes one that a row of another
extractor already holds; only running the extractors can say. What the files do show: the dead texts were
about half unfaithful (above); by row (each row once, whatever the number of extractors that hold it; it includes the 6 first-set rows that are dead for one extractor but not for all) 86.0% and 98.0% of the rows the code writes now are `yes`; the
rewritten texts are few and faithful, 54 of 56 new rows (6 of the 56 are for companies with no dead row, so
they replace nothing). Small numbers (30 + 30 bodies). This does not measure the effect of the change: the
old matcher was not re-run, and removing texts that were marked unfaithful raises the share by construction.

**The report counts those marks, and since PR #157 it also prints the figures without them.**
`report_wi18_2026_11_06.py faithfulness` counts every marked row (the blocks `[all]`, `[all, real]`,
`[2022+]`) and notes how many are on dead texts (it reads `faithfulness_stale.json`, which `--stale`
rewrote on 2026-11-10 for both sets: **147 and 116 marks** of an extractor on a dead text; first set
baseline 40, model 41, units 52, units2 7, units3 7; confirmation set baseline 51, units 49, units2 7,
units3 9). After those blocks it prints the same figures labelled `[current code: ...]`, with each
extractor's marks on dead texts left out (a row left with no extractor is dropped: 356 of 453 rows remain in
the first set, 342 of 438 in the confirmation set). Nothing extra is printed when no marked row is dead;
when the check is missing for an extractor or was made with other code than the current one, one line says
so and no figure is printed. The first table below is the `[all]` view, the second the `[current code]`
view (Appendix A.3, A.4 are the report's own lines). Before the PR the same view was made by hand from
copies of the two files (the owner's files are untouched). The replacement tables above are reproduced with
`uv run scripts/dead_text_replacement_2026_11_10.py --out data --earlier data/earlier_2026_11_09/first
--out data/faithfulness_confirmation --earlier data/earlier_2026_11_09/confirmation`, where
`data/earlier_2026_11_09/` holds copies of the two ids files as they were before the new marks (copied
from the backup taken on 2026-11-09 at 23:10, before the redraws of the ids files at 23:20 and 23:39; the folder is local, git-ignored, like all of `data/`; recorded in `docs/gold_datasets.md`, set 4).

Faithful share, every company named (the bar: lower end of the interval at or above 80%):

| Extractor | First set, all marks | First set, current code | Confirmation, all marks | Confirmation, current code |
|---|---|---|---|---|
| `baseline` | 63.2% [49.1, 75.5] fail | 71.3% [55.5, 84.6] not decided | 87.4% [81.4, 92.2] pass | 97.7% [94.5, 100] pass |
| `model` | 76.3% [61.8, 88.1] not decided | 85.2% [68.6, 99.1] not decided | not marked | not marked |
| **`units`** | 87.3% [79.8, 93.0] not decided | **97.5% [93.3, 100] pass** | 87.3% [82.5, 91.5] pass | **97.1% [94.0, 99.4] pass** |
| `units2` | 84.4% [72.5, 97.2] not decided | 85.7% [73.5, 99.1] not decided | 91.4% [86.2, 96.0] pass | 92.2% [86.7, 96.9] pass |
| `units3` | 91.6% [84.7, 97.8] pass | 93.2% [86.7, 99.2] pass | 93.7% [90.3, 96.9] pass | 95.8% [92.5, 98.6] pass |
| Companies left without a text (`units` / `units2` / `units3`) | 0 / 19 / 6 | 0 / 19 / 6 | 1 / 10 / 4 | 1 / 10 / 4 |

Counting the companies an extractor was not asked about as unfaithful (the strict reading) on current code:
`units` 95.1% [89.3, 99.3] pass and 97.1% [94.0, 99.4] pass; `units2` 79.7% [67.0, 94.2] not decided and
88.5% [82.8, 93.7] pass; `units3` 86.7% [79.3, 94.7] not decided and 91.9% [87.5, 95.8] pass (first set,
then confirmation set). Gains on the same companies, current code: `units` minus `baseline` +25.6 [+12.6,
+43.0] better on the first set and -0.6 [-1.7, 0.0] not decided on the confirmation set; `units2` minus
`units` -11.6 [-24.4, 0.0] not decided and -4.3 [-9.4, -0.6] not better; `units3` minus `units` -4.1
[-7.4, 0.0] not decided and -1.8 [-4.0, 0.0] not decided. With all marks the same gains are +22.2
[+10.2, +37.2] better and -0.5 not decided (`units` minus `baseline`), -11.1 and -4.1 not better
(`units2` minus `units`), -3.9 and -1.8 (`units3` minus `units`). The 2022+ rows (7 and 9 bodies) are in
Appendix A.3 to A.6; they cannot decide, as elsewhere on this page, and no conclusion below rests on them.

What this adds to "What the results say":
1. **Among the extractors marked on both sets, `units` is the only one whose lower bound stays at or above
   80% on every company named under the strictest count of unwritten companies**, on current code
   (`units2` and `units3` are *not decided* on the first set under that count; `baseline` is *not decided*
   on the first set; `model` has no confirmation marks). On the plain count `units3` also passes both sets,
   and `baseline` passes the confirmation set. No extractor beats `units` on a paired comparison.
2. **What the change did to the extractors is a removal, and what it did beyond that is small and not
   measured.** The 97 and 96 dead rows were about half unfaithful and are no longer written (or were
   rewritten: `units` 4 of 52 and 10 of 49 dead marks). On current code `units` is about 10 points higher
   than counting every mark (87.3% to 97.5% and 87.3% to 97.1%); that rise is the removal of those marks,
   and the evidence does not separate "the matcher stopped finding companies that were not there" from
   "the texts got better". The old matcher was not re-run.
3. **`baseline` is the unstable one**: it fails on the first set and passes on the confirmation set.
4. **`units2` and `units3` do not beat `units`**; they look close on the bar only because they leave 4 to 19
   companies without a text.
5. **`model`** has no marks on the confirmation set; on the first set 41 of its 190 marked texts are dead
   (190 to 149 in the current-code view) and none of them has a new row of its own. It also writes 10 texts
   nobody has marked (`faithfulness_stale.json`); its current-code figure (85.2%) leaves those 10 out, so it
   is on an incomplete set of its current texts and may be biased either way. It stays provisional until the
   attribution checkpoint is retrained.

#### Text measures after the change (characters per text)

Same script (`scripts/text_measures_2026_11_08.py`), run on 2026-11-10. "All rows" reads every row of the
files (dead texts included); "current code" reads the same copies as the second table above. The script
lists its extractors in a constant (`EXTRACTORS` = `baseline`, `units`, `units2`, `units3`), so it has no
`model` column. That is a limit of the script, not a decision: `model` has 190 texts on the first set and
none marked on the confirmation set; adding it is one line and a test, and is not done here.

| | `baseline` | `units` | `units2` | `units3` |
|---|---|---|---|---|
| Mean characters, first set, all rows (texts) | 921 (204) | 500 (212) | 510 (135) | 476 (148) |
| Mean characters, first set, current code | 829 (164) | 453 (160) | 507 (128) | 471 (141) |
| Median characters, first set, current code | 392 | 250 | 296 | 285 |
| Mean characters, confirmation, all rows (texts) | 984 (222) | 609 (220) | 634 (164) | 616 (170) |
| Mean characters, confirmation, current code | 1,058 (171) | 620 (171) | 646 (157) | 634 (161) |
| Median characters, confirmation, current code | 622 | 290 | 294 | 278 |
| Sentences naming this company, current code (first / confirmation) | 37.5% / 39.2% | 53.6% / 59.7% | 49.6% / 59.6% | 51.1% / 59.9% |
| Sentences naming this company and another, current code | 16.2% / 19.4% | 15.2% / 21.6% | 14.2% / 22.3% | 16.4% / 22.8% |
| Other companies named per text, current code | 1.88 / 1.70 | 0.60 / 1.16 | 0.66 / 1.19 | 0.71 / 1.19 |

Paired on the same company and body, current code: the baseline writes **+381 characters more than `units`
[+159, +700] on the first set and +438 [+325, +562] on the confirmation set** (about 1.8 and 1.7 times as
long); `units2` and `units3` differ from `units` by +4 [-7, +16] and +4 [-7, +14] (first set), +3 [-4, +12]
and 0 [-10, +9] (confirmation set), that is, none. Before the change the same gaps were +370 and +385
(Appendix B.1, B.2). `units` reaches the same faithful share as the baseline on the confirmation set (97.1%
against 97.7% on current code) and a higher one on the first (97.5% against 71.3%), with about 41 to 45%
less text (453 against 829 and 620 against 1,058 characters on current code). The share of sentences naming
the company and another does not differ from `units` for `units2`; for `units3` minus `units` it is +3.3
points [+0.2, +7.3] on the first set, an interval that excludes zero (`units3` writes slightly more mixed
text), and +2.3 [-1.9, +7.7] on the confirmation set.
Every figure is in Appendix B.3 to B.6.

### Known limits of `units` (so nobody is surprised later)

- It writes texts for names that are not companies (18 of 24 unfaithful on the confirmation set, 21 of 22
  on the first). A cut ships them.
- It does not use the "name — text" header rule (only `units2`/`units3` do). One confirmation text, the
  Steel Dynamics one that opens with "Joy Global — ... BHP Billiton — ...", fails for that reason.
- 16 confirmation texts carry the owner's note "this text is huge"; the longest texts reach 6,000 characters.
- When a sentence names nobody the model gives it to the best-scoring company if the score is 0.5 or
  more, even if the sentence is about a company outside the universe; the wrong cases scored 0.9 or
  more, so a higher threshold would not remove them (T-148b3, open).
- All figures come from 30 + 30 bodies and the owner's single reading of each text.

## What is next

Improvement options, **all proposed, none decided or started**. Nothing here is asked of the owner:
the owner said the tagging on this front is finished, so options that need more tagging say so.

| # | Option | Expected gain | Needs |
|---|---|---|---|
| 1 | ~~**Run the staleness check**~~ **done 2026-11-08 (round 8)**: confirmation set 0 stale; first set 8 stale rows (see "Results"). **Out of date since PR #155**: 2026-11-09 finds 97 and 96 rows dead; the report counts their marks (see "Update of 2026-11-10") | the tagged texts are what the code writes | done |
| 2 | **Migrate the evidence to `gold/`, before relying on the figures** (T-148b11: the `units*` marks and the readings of the exempt names exist only in the git-ignored `data/`) | the figures can be reproduced from a clean checkout | the owner's word |
| 3 | **Wire `units` into step 4** (T-148b12: a default or setting, and the cut that calls it) | the selection takes effect | code, when the owner asks |
| 4 | **Drop the texts for names the guard rejects, keep everything else of `units`** (a hybrid of `units` and `units3`) | removes most of the 18 non-company errors without the 7 to 12 missing real companies | code and a free re-score on the texts already tagged (the hybrid writes a subset of `units`' texts) |
| 5 | **Header rule for `units`** ("A — text" lines belong to A) | fixes digest lines like the Steel Dynamics text | code; the `units2` rule exists and could be taken alone |
| 6 | **A length cap or a trim** for very long texts (16 notes "huge") | shorter, purer texts | code; the effect on faithfulness is not measurable without a new reading |
| 7 | **A purity measure** that checks how much of a text names another company, validated on a read sample | a metric that sees what the faithful rule lets through | a small sample read by the owner, or a decision to use the name count as is |
| 8 | **A stricter second reading of the same texts** ("only this company's content") | would show whether `units` is really cleaner than the baseline | tagging; not asked |
| 9 | **Abstention threshold** (0.5 today; see T-148b3) | fewer wrong high-confidence assignments | the owner's decision |
| 10 | **Match one-word names** ("JPMorgan") and header names with the plain resolver (T-148b9, out of scope so far) | fewer missed companies | code |
| 11 | **More data for the model**: the `continuation` labels (38% right) are out; a cleaner way to use them, or more `lead` bodies | a better model | training time on the GPU; the exam sets must not be trained on |
| 12 | **A larger 2022+ sample** (9 bodies per set cannot decide) | a verdict for the period that matters | new bodies and tagging; no pool is left in the drawn sets |
| 13 | **Publish the model to the Hub and pin its revision** | the cut and the report run from a pinned checkpoint, not a local directory | the owner's explicit word |
| 14 | ~~**Make the report leave out the marks of dead texts** (and print both views), and refresh `faithfulness_stale.json`~~ **done 2026-11-10 (PR #157, not merged)**: the report prints `[current code: ...]`; `--stale` re-run into `data/` for both sets | the official figures measure the current code | done |

A reasonable order, as a proposal only: 2 (so the evidence is not
lost), 4 and 5, then a free re-score of what is already tagged; 3 and 13 when the owner wants the selection to take effect and the model fixed.

## Where this is tracked

PLAN D34 rule (e), TASKS T-148b10, `docs/evaluation.md` (confirmation section), `docs/gold_datasets.md`
(set 4) and the body of PR #144 all point here for the selection and the figures.

## Appendix A: every metric, as the report prints it

Produced by `uv run scripts/report_wi18_2026_11_06.py faithfulness --out <dir>` on 2026-11-08, with the
"informative" lines (share of the texts written, omissions not counted) left out. Block names: `[all]`
every body; `[all, real]` the same without the names the owner marked as not companies; `[2022+]` bodies
from 2022 on; `[2022+, real]` both. "X minus Y" is the gain of X over Y on the same companies, with its
interval; its verdict is "better", "not better" or "not decided". "N companies left without one" are
companies named in the bodies for which the extractor wrote no text (they count as not faithful). The
line "the bar on every company named" is the strict reading.

### A.1 First set (30 bodies, 437 texts, `data/`)

The first line is the report's own staleness line, from the check re-run on 2026-11-08 (round 8) with
the current code and the model `local:8399ccf003c60143`: **8 marked rows are on texts the code no longer
writes** (rows 351 and 388 for `units`, `units2` and `units3`, and row 431 for `units2` and `units3`; bodies
122960 and 88621), so the first-set figures are development figures read with that caveat. They are 8 of the
rows of `units`, `units2` and `units3` (2 of 185 for `units`); the cause is in "Results".

```
staleness check (baseline 2026-11-08, model 2026-11-08, units 2026-11-08, units2 2026-11-08, units3 2026-11-08): 8 marked rows are on texts the current code no longer writes (122960#AAPL#3 (units), 122960#AMZN#3 (units), 122960#AAPL#3 (units2), 122960#AMZN#3 (units2), 88621#AMZN#3 (units2), 122960#AAPL#3 (units3), ...); the figures include their marks
the intervals are a bootstrap over bodies, not over texts (D32)
[all]   baseline: 60.5% [45.2%, 73.8%] -> fail (the bar: of the companies named, those with a faithful text; 115 faithful of 190 texts + 0 companies left without one)
[all]      model: 76.3% [61.8%, 88.1%] -> not decided (the bar: of the companies named, those with a faithful text; 145 faithful of 190 texts + 0 companies left without one)
[all]      units: 85.4% [76.5%, 92.2%] -> not decided (the bar: of the companies named, those with a faithful text; 158 faithful of 185 texts + 0 companies left without one)
[all]     units2: 84.4% [72.5%, 97.2%] -> not decided (the bar: of the companies named, those with a faithful text; 130 faithful of 135 texts + 19 companies left without one)
[all]     units3: 91.6% [84.7%, 97.8%] -> pass (the bar: of the companies named, those with a faithful text; 141 faithful of 148 texts + 6 companies left without one)
[all] model minus baseline on 190 company texts of 30 bodies: 15.8% [10.2%, 23.8%] -> better
[all] units minus baseline on 185 company texts of 30 bodies: 24.3% [12.9%, 39.4%] -> better
[all] units minus model on 185 company texts of 30 bodies: 8.6% [-0.7%, 22.8%] -> not decided
[all] units2 minus units on 153 company texts of 29 bodies: -11.1% [-23.5%, 0.0%] -> not decided
[all] units2 minus baseline on 154 company texts of 29 bodies: 16.9% [-5.2%, 42.1%] -> not decided
[all] units3 minus units2 on 154 company texts of 29 bodies: 7.1% [0.0%, 16.7%] -> not decided
[all] units3 minus units on 153 company texts of 29 bodies: -3.9% [-7.3%, 0.0%] -> not decided
[all] units: the 5 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 83.2% [72.7%, 91.0%] -> not decided
[all] units2: the 36 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 68.4% [57.9%, 80.3%] -> not decided
[all] units3: the 36 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 74.2% [66.0%, 81.5%] -> not decided
[all] real companies: 23 of the 37 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[all] baseline: 23 of the 190 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all] model: 23 of the 190 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all] units: 22 of the 185 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all] units2: 0 of the 135 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all] units3: 0 of the 148 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all, real]   baseline: 68.9% [53.6%, 82.1%] -> not decided (the bar: of the companies named, those with a faithful text; 115 faithful of 167 texts + 0 companies left without one)
[all, real]      model: 86.2% [71.7%, 97.5%] -> not decided (the bar: of the companies named, those with a faithful text; 144 faithful of 167 texts + 0 companies left without one)
[all, real]      units: 94.0% [87.0%, 99.2%] -> pass (the bar: of the companies named, those with a faithful text; 157 faithful of 163 texts + 4 companies left without one)
[all, real]     units2: 77.8% [64.1%, 92.9%] -> not decided (the bar: of the companies named, those with a faithful text; 130 faithful of 135 texts + 32 companies left without one)
[all, real]     units3: 84.4% [76.4%, 93.3%] -> not decided (the bar: of the companies named, those with a faithful text; 141 faithful of 148 texts + 19 companies left without one)
[all, real] model minus baseline on 167 company texts of 29 bodies: 17.4% [10.8%, 27.7%] -> better
[all, real] units minus baseline on 167 company texts of 29 bodies: 25.1% [12.3%, 41.4%] -> better
[all, real] units minus model on 167 company texts of 29 bodies: 7.8% [-3.3%, 22.7%] -> not decided
[all, real] units2 minus units on 167 company texts of 29 bodies: -16.2% [-30.6%, -1.6%] -> not better
[all, real] units2 minus baseline on 167 company texts of 29 bodies: 9.0% [-15.1%, 36.5%] -> not decided
[all, real] units3 minus units2 on 167 company texts of 29 bodies: 6.6% [0.0%, 14.6%] -> not decided
[all, real] units3 minus units on 167 company texts of 29 bodies: -9.6% [-16.5%, -1.6%] -> not better
[2022+]   baseline: 60.6% [27.6%, 86.1%] -> not decided (the bar: of the companies named, those with a faithful text; 20 faithful of 33 texts + 0 companies left without one)
[2022+]      model: 87.9% [69.2%, 100.0%] -> not decided (the bar: of the companies named, those with a faithful text; 29 faithful of 33 texts + 0 companies left without one)
[2022+]      units: 87.9% [68.8%, 100.0%] -> not decided (the bar: of the companies named, those with a faithful text; 29 faithful of 33 texts + 0 companies left without one)
[2022+]     units2: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+]     units3: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+] model minus baseline on 33 company texts of 7 bodies: 27.3% [8.6%, 58.3%] -> better
[2022+] units minus baseline on 33 company texts of 7 bodies: 27.3% [8.6%, 58.3%] -> better
[2022+] units minus model on 33 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units2 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units2 minus baseline on 29 company texts of 7 bodies: 31.0% [9.1%, 65.2%] -> better
[2022+] units3 minus units2 on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units3 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units2: the 4 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 87.9% [69.2%, 100.0%] -> not decided
[2022+] units3: the 4 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 87.9% [69.2%, 100.0%] -> not decided
[2022+] real companies: 4 of the 4 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[2022+] baseline: 4 of the 33 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+] model: 4 of the 33 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+] units: 4 of the 33 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+] units2: 0 of the 29 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+] units3: 0 of the 29 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+, real]   baseline: 69.0% [34.8%, 90.9%] -> not decided (the bar: of the companies named, those with a faithful text; 20 faithful of 29 texts + 0 companies left without one)
[2022+, real]      model: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+, real]      units: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+, real]     units2: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+, real]     units3: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+, real] model minus baseline on 29 company texts of 7 bodies: 31.0% [9.1%, 65.2%] -> better
[2022+, real] units minus baseline on 29 company texts of 7 bodies: 31.0% [9.1%, 65.2%] -> better
[2022+, real] units minus model on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+, real] units2 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+, real] units2 minus baseline on 29 company texts of 7 bodies: 31.0% [9.1%, 65.2%] -> better
[2022+, real] units3 minus units2 on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+, real] units3 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
```

### A.2 Confirmation set (30 other bodies, 395 texts, `data/faithfulness_confirmation/`)

The staleness check was run on 2026-11-08 (round 8) with the current code and the model
`local:8399ccf003c60143`: **no marked row is on a text the code no longer writes** (0 stale for `baseline`,
`units`, `units2` and `units3`). The 193 rows of `units` carry an older code revision in their record, with
the same text, so the marks stand. (The check also listed 81 "new" texts of `model`, which this draw did not
include; they are not rows to read.)

```
staleness check (baseline 2026-11-08, model 2026-11-08, units 2026-11-08, units2 2026-11-08, units3 2026-11-08): no marked row is on a text the current code no longer writes
the intervals are a bootstrap over bodies, not over texts (D32)
[all]   baseline: 86.6% [80.7%, 91.3%] -> pass (the bar: of the companies named, those with a faithful text; 168 faithful of 194 texts + 0 companies left without one)
[all]      units: 86.5% [81.3%, 91.1%] -> pass (the bar: of the companies named, those with a faithful text; 167 faithful of 193 texts + 0 companies left without one)
[all]     units2: 92.7% [88.3%, 96.5%] -> pass (the bar: of the companies named, those with a faithful text; 152 faithful of 157 texts + 7 companies left without one)
[all]     units3: 94.5% [91.8%, 97.2%] -> pass (the bar: of the companies named, those with a faithful text; 155 faithful of 162 texts + 2 companies left without one)
[all] units minus baseline on 193 company texts of 30 bodies: -0.5% [-1.5%, 0.0%] -> not decided
[all] units2 minus units on 164 company texts of 30 bodies: -3.0% [-7.2%, 0.0%] -> not decided
[all] units2 minus baseline on 164 company texts of 30 bodies: -3.7% [-7.7%, -0.6%] -> not better
[all] units3 minus units2 on 164 company texts of 30 bodies: 1.8% [0.0%, 6.1%] -> not decided
[all] units3 minus units on 164 company texts of 30 bodies: -1.2% [-3.1%, 0.0%] -> not decided
[all] units: the 1 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 86.1% [80.9%, 90.7%] -> pass
[all] units2: the 30 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 78.4% [72.5%, 83.7%] -> not decided
[all] units3: the 30 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 79.9% [74.0%, 85.1%] -> not decided
[all] real companies: 25 of the 30 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[all] baseline: 25 of the 194 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all] units: 24 of the 193 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all] units2: 0 of the 157 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all] units3: 0 of the 162 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[all, real]   baseline: 95.9% [92.2%, 98.7%] -> pass (the bar: of the companies named, those with a faithful text; 162 faithful of 169 texts + 0 companies left without one)
[all, real]      units: 95.3% [92.0%, 98.2%] -> pass (the bar: of the companies named, those with a faithful text; 161 faithful of 169 texts + 0 companies left without one)
[all, real]     units2: 89.9% [85.1%, 94.5%] -> pass (the bar: of the companies named, those with a faithful text; 152 faithful of 157 texts + 12 companies left without one)
[all, real]     units3: 91.7% [87.4%, 95.5%] -> pass (the bar: of the companies named, those with a faithful text; 155 faithful of 162 texts + 7 companies left without one)
[all, real] units minus baseline on 169 company texts of 30 bodies: -0.6% [-1.7%, 0.0%] -> not decided
[all, real] units2 minus units on 169 company texts of 30 bodies: -5.3% [-10.5%, -1.2%] -> not better
[all, real] units2 minus baseline on 169 company texts of 30 bodies: -5.9% [-10.8%, -1.8%] -> not better
[all, real] units3 minus units2 on 169 company texts of 30 bodies: 1.8% [0.0%, 5.9%] -> not decided
[all, real] units3 minus units on 169 company texts of 30 bodies: -3.6% [-7.6%, -0.6%] -> not better
[2022+]   baseline: 89.6% [76.2%, 96.1%] -> not decided (the bar: of the companies named, those with a faithful text; 60 faithful of 67 texts + 0 companies left without one)
[2022+]      units: 89.4% [78.9%, 95.7%] -> not decided (the bar: of the companies named, those with a faithful text; 59 faithful of 66 texts + 0 companies left without one)
[2022+]     units2: 90.9% [77.4%, 100.0%] -> not decided (the bar: of the companies named, those with a faithful text; 50 faithful of 52 texts + 3 companies left without one)
[2022+]     units3: 96.4% [91.7%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 53 faithful of 55 texts + 0 companies left without one)
[2022+] units minus baseline on 66 company texts of 9 bodies: -1.5% [-3.4%, 0.0%] -> not decided
[2022+] units2 minus units on 55 company texts of 9 bodies: -5.5% [-18.5%, 0.0%] -> not decided
[2022+] units2 minus baseline on 55 company texts of 9 bodies: -7.3% [-18.9%, 0.0%] -> not decided
[2022+] units3 minus units2 on 55 company texts of 9 bodies: 5.5% [0.0%, 18.5%] -> not decided
[2022+] units3 minus units on 55 company texts of 9 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units: the 1 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 88.1% [75.8%, 95.3%] -> not decided
[2022+] units2: the 12 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 74.6% [61.7%, 83.1%] -> not decided
[2022+] units3: the 12 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 79.1% [67.3%, 87.9%] -> not decided
[2022+] real companies: 11 of the 12 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[2022+] baseline: 11 of the 67 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+] units: 10 of the 66 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+] units2: 0 of the 52 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+] units3: 0 of the 55 texts it wrote are for a company that is not one (informative: the bar on real companies leaves them out, a cut ships them)
[2022+, real]   baseline: 98.2% [92.3%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 55 faithful of 56 texts + 0 companies left without one)
[2022+, real]      units: 96.4% [90.9%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 54 faithful of 56 texts + 0 companies left without one)
[2022+, real]     units2: 89.3% [76.6%, 97.1%] -> not decided (the bar: of the companies named, those with a faithful text; 50 faithful of 52 texts + 4 companies left without one)
[2022+, real]     units3: 94.6% [89.5%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 53 faithful of 55 texts + 1 companies left without one)
[2022+, real] units minus baseline on 56 company texts of 9 bodies: -1.8% [-4.1%, 0.0%] -> not decided
[2022+, real] units2 minus units on 56 company texts of 9 bodies: -7.1% [-19.6%, 0.0%] -> not decided
[2022+, real] units2 minus baseline on 56 company texts of 9 bodies: -8.9% [-20.0%, 0.0%] -> not decided
[2022+, real] units3 minus units2 on 56 company texts of 9 bodies: 5.4% [0.0%, 18.4%] -> not decided
[2022+, real] units3 minus units on 56 company texts of 9 bodies: -1.8% [-5.7%, 0.0%] -> not decided
```

### A.3 First set, current code (dead marks left out; 356 of 453 rows, 2026-11-10)

The `[current code: ...]` lines of the report as of PR #157, with `faithfulness_stale.json` re-run on 2026-11-10 (the "informative" lines are left out).

```
current code: 147 marks of 103 marked rows are on texts the current code no longer writes (faithfulness_stale.json); the figures below leave them out, 356 of 453 rows remain
[current code: all]   baseline: 71.3% [55.5%, 84.6%] -> not decided (the bar: of the companies named, those with a faithful text; 117 faithful of 164 texts + 0 companies left without one)
[current code: all]      model: 85.2% [68.6%, 99.1%] -> not decided (the bar: of the companies named, those with a faithful text; 127 faithful of 149 texts + 0 companies left without one)
[current code: all]      units: 97.5% [93.3%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 156 faithful of 160 texts + 0 companies left without one)
[current code: all]     units2: 85.7% [73.5%, 99.1%] -> not decided (the bar: of the companies named, those with a faithful text; 126 faithful of 128 texts + 19 companies left without one)
[current code: all]     units3: 93.2% [86.7%, 99.2%] -> pass (the bar: of the companies named, those with a faithful text; 137 faithful of 141 texts + 6 companies left without one)
[current code: all] model minus baseline on 149 company texts of 25 bodies: 16.1% [9.4%, 27.0%] -> better
[current code: all] units minus baseline on 160 company texts of 29 bodies: 25.6% [12.6%, 43.0%] -> better
[current code: all] units minus model on 146 company texts of 25 bodies: 11.6% [0.0%, 29.2%] -> not decided
[current code: all] units2 minus units on 147 company texts of 28 bodies: -11.6% [-24.4%, 0.0%] -> not decided
[current code: all] units2 minus baseline on 147 company texts of 28 bodies: 15.0% [-8.3%, 40.6%] -> not decided
[current code: all] units3 minus units2 on 147 company texts of 28 bodies: 7.5% [0.0%, 17.5%] -> not decided
[current code: all] units3 minus units on 147 company texts of 28 bodies: -4.1% [-7.4%, 0.0%] -> not decided
[current code: all] units: the 4 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 95.1% [89.3%, 99.3%] -> pass
[current code: all] units2: the 11 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 79.7% [67.0%, 94.2%] -> not decided
[current code: all] units3: the 11 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 86.7% [79.3%, 94.7%] -> not decided
[current code: all] real companies: 1 of the 11 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[current code: all, real]   baseline: 71.8% [56.3%, 84.6%] -> not decided (the bar: of the companies named, those with a faithful text; 117 faithful of 163 texts + 0 companies left without one)
[current code: all, real]      model: 85.8% [69.8%, 99.1%] -> not decided (the bar: of the companies named, those with a faithful text; 127 faithful of 148 texts + 0 companies left without one)
[current code: all, real]      units: 95.1% [89.3%, 99.3%] -> pass (the bar: of the companies named, those with a faithful text; 155 faithful of 159 texts + 4 companies left without one)
[current code: all, real]     units2: 80.3% [67.2%, 95.0%] -> not decided (the bar: of the companies named, those with a faithful text; 126 faithful of 128 texts + 29 companies left without one)
[current code: all, real]     units3: 87.3% [79.6%, 95.7%] -> not decided (the bar: of the companies named, those with a faithful text; 137 faithful of 141 texts + 16 companies left without one)
[current code: all, real] model minus baseline on 148 company texts of 25 bodies: 16.2% [9.5%, 27.1%] -> better
[current code: all, real] units minus baseline on 163 company texts of 29 bodies: 23.3% [10.0%, 39.2%] -> better
[current code: all, real] units minus model on 148 company texts of 25 bodies: 9.5% [-2.3%, 26.4%] -> not decided
[current code: all, real] units2 minus units on 157 company texts of 28 bodies: -14.6% [-28.5%, -1.0%] -> not better
[current code: all, real] units2 minus baseline on 157 company texts of 28 bodies: 9.6% [-13.7%, 37.4%] -> not decided
[current code: all, real] units3 minus units2 on 157 company texts of 28 bodies: 7.0% [0.0%, 15.6%] -> not decided
[current code: all, real] units3 minus units on 157 company texts of 28 bodies: -7.6% [-13.3%, -1.0%] -> not better
[current code: 2022+]   baseline: 75.9% [40.9%, 98.0%] -> not decided (the bar: of the companies named, those with a faithful text; 22 faithful of 29 texts + 0 companies left without one)
[current code: 2022+]      model: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 27 faithful of 27 texts + 0 companies left without one)
[current code: 2022+]      units: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[current code: 2022+]     units2: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[current code: 2022+]     units3: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[current code: 2022+] model minus baseline on 27 company texts of 6 bodies: 25.9% [3.1%, 60.0%] -> better
[current code: 2022+] units minus baseline on 29 company texts of 7 bodies: 24.1% [2.0%, 59.1%] -> better
[current code: 2022+] units minus model on 27 company texts of 6 bodies: 0.0% [0.0%, 0.0%] -> not decided
[current code: 2022+] units2 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[current code: 2022+] units2 minus baseline on 29 company texts of 7 bodies: 24.1% [2.0%, 59.1%] -> better
[current code: 2022+] units3 minus units2 on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[current code: 2022+] units3 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
```

### A.4 Confirmation set, current code (342 of 438 rows)

The report's own lines (record of 2026-11-10).

```
current code: 116 marks of 97 marked rows are on texts the current code no longer writes (faithfulness_stale.json); the figures below leave them out, 342 of 438 rows remain
[current code: all]   baseline: 97.7% [94.5%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 168 faithful of 171 texts + 1 companies left without one)
[current code: all]      units: 97.1% [94.0%, 99.4%] -> pass (the bar: of the companies named, those with a faithful text; 167 faithful of 171 texts + 1 companies left without one)
[current code: all]     units2: 92.2% [86.7%, 96.9%] -> pass (the bar: of the companies named, those with a faithful text; 154 faithful of 157 texts + 10 companies left without one)
[current code: all]     units3: 95.8% [92.5%, 98.6%] -> pass (the bar: of the companies named, those with a faithful text; 158 faithful of 161 texts + 4 companies left without one)
[current code: all] units minus baseline on 171 company texts of 30 bodies: -0.6% [-1.7%, 0.0%] -> not decided
[current code: all] units2 minus units on 164 company texts of 30 bodies: -4.3% [-9.4%, -0.6%] -> not better
[current code: all] units2 minus baseline on 164 company texts of 30 bodies: -4.9% [-9.7%, -1.2%] -> not better
[current code: all] units3 minus units2 on 164 company texts of 30 bodies: 2.4% [0.0%, 7.1%] -> not decided
[current code: all] units3 minus units on 164 company texts of 30 bodies: -1.8% [-4.0%, 0.0%] -> not decided
[current code: all] units2: the 7 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 88.5% [82.8%, 93.7%] -> pass
[current code: all] units3: the 7 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 91.9% [87.5%, 95.8%] -> pass
[current code: all] real companies: 6 of the 7 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[current code: all, real]   baseline: 98.2% [95.1%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 163 faithful of 165 texts + 1 companies left without one)
[current code: all, real]      units: 97.6% [94.7%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 162 faithful of 165 texts + 1 companies left without one)
[current code: all, real]     units2: 91.7% [86.3%, 96.2%] -> pass (the bar: of the companies named, those with a faithful text; 154 faithful of 157 texts + 11 companies left without one)
[current code: all, real]     units3: 95.2% [91.8%, 98.0%] -> pass (the bar: of the companies named, those with a faithful text; 158 faithful of 161 texts + 5 companies left without one)
[current code: all, real] units minus baseline on 165 company texts of 30 bodies: -0.6% [-1.8%, 0.0%] -> not decided
[current code: all, real] units2 minus units on 165 company texts of 30 bodies: -4.8% [-9.8%, -1.1%] -> not better
[current code: all, real] units2 minus baseline on 165 company texts of 30 bodies: -5.5% [-10.2%, -1.6%] -> not better
[current code: all, real] units3 minus units2 on 165 company texts of 30 bodies: 2.4% [0.0%, 7.1%] -> not decided
[current code: all, real] units3 minus units on 165 company texts of 30 bodies: -2.4% [-4.8%, -0.6%] -> not better
[current code: 2022+]   baseline: 98.4% [92.9%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 60 faithful of 61 texts + 0 companies left without one)
[current code: 2022+]      units: 96.7% [91.7%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 59 faithful of 61 texts + 0 companies left without one)
[current code: 2022+]     units2: 90.9% [77.4%, 100.0%] -> not decided (the bar: of the companies named, those with a faithful text; 50 faithful of 52 texts + 3 companies left without one)
[current code: 2022+]     units3: 96.4% [91.7%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 53 faithful of 55 texts + 0 companies left without one)
[current code: 2022+] units minus baseline on 61 company texts of 9 bodies: -1.6% [-3.7%, 0.0%] -> not decided
[current code: 2022+] units2 minus units on 55 company texts of 9 bodies: -5.5% [-18.4%, 0.0%] -> not decided
[current code: 2022+] units2 minus baseline on 55 company texts of 9 bodies: -7.3% [-18.8%, 0.0%] -> not decided
[current code: 2022+] units3 minus units2 on 55 company texts of 9 bodies: 5.5% [0.0%, 18.4%] -> not decided
[current code: 2022+] units3 minus units on 55 company texts of 9 bodies: 0.0% [0.0%, 0.0%] -> not decided
[current code: 2022+] units2: the 6 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 82.0% [69.4%, 90.9%] -> not decided
[current code: 2022+] units3: the 6 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 86.9% [78.3%, 94.9%] -> not decided
[current code: 2022+] real companies: 5 of the 6 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[current code: 2022+, real]   baseline: 98.2% [92.1%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 55 faithful of 56 texts + 0 companies left without one)
[current code: 2022+, real]      units: 96.4% [91.7%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 54 faithful of 56 texts + 0 companies left without one)
[current code: 2022+, real]     units2: 89.3% [76.6%, 97.1%] -> not decided (the bar: of the companies named, those with a faithful text; 50 faithful of 52 texts + 4 companies left without one)
[current code: 2022+, real]     units3: 94.6% [89.5%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 53 faithful of 55 texts + 1 companies left without one)
[current code: 2022+, real] units minus baseline on 56 company texts of 9 bodies: -1.8% [-4.1%, 0.0%] -> not decided
[current code: 2022+, real] units2 minus units on 56 company texts of 9 bodies: -7.1% [-20.0%, 0.0%] -> not decided
[current code: 2022+, real] units2 minus baseline on 56 company texts of 9 bodies: -8.9% [-20.0%, 0.0%] -> not decided
[current code: 2022+, real] units3 minus units2 on 56 company texts of 9 bodies: 5.4% [0.0%, 18.0%] -> not decided
[current code: 2022+, real] units3 minus units on 56 company texts of 9 bodies: -1.8% [-5.8%, 0.0%] -> not decided
```

### A.5 First set, all marks (453 rows, the report as it is, 2026-11-10)

The first line of the block is the report's own staleness line, read from the record of 2026-11-08, before PR #155: it is out of date (97 rows are dead for the current code and this view still counts them).

```
staleness check (baseline 2026-11-08, model 2026-11-08, units 2026-11-08, units2 2026-11-08, units3 2026-11-08): 8 marked rows are on texts the current code no longer writes (...); the figures include their marks
the intervals are a bootstrap over bodies, not over texts (D32)
[all]   baseline: 63.2% [49.1%, 75.5%] -> fail (the bar: of the companies named, those with a faithful text; 129 faithful of 204 texts + 0 companies left without one)
[all]      model: 76.3% [61.8%, 88.1%] -> not decided (the bar: of the companies named, those with a faithful text; 145 faithful of 190 texts + 0 companies left without one)
[all]      units: 87.3% [79.8%, 93.0%] -> not decided (the bar: of the companies named, those with a faithful text; 185 faithful of 212 texts + 0 companies left without one)
[all]     units2: 84.4% [72.5%, 97.2%] -> not decided (the bar: of the companies named, those with a faithful text; 130 faithful of 135 texts + 19 companies left without one)
[all]     units3: 91.6% [84.7%, 97.8%] -> pass (the bar: of the companies named, those with a faithful text; 141 faithful of 148 texts + 6 companies left without one)
[all] model minus baseline on 190 company texts of 30 bodies: 13.7% [8.1%, 21.2%] -> better
[all] units minus baseline on 185 company texts of 30 bodies: 22.2% [10.2%, 37.2%] -> better
[all] units minus model on 185 company texts of 30 bodies: 8.6% [-0.7%, 22.8%] -> not decided
[all] units2 minus units on 153 company texts of 29 bodies: -11.1% [-23.5%, 0.0%] -> not decided
[all] units2 minus baseline on 154 company texts of 29 bodies: 14.3% [-7.5%, 39.7%] -> not decided
[all] units3 minus units2 on 154 company texts of 29 bodies: 7.1% [0.0%, 16.7%] -> not decided
[all] units3 minus units on 153 company texts of 29 bodies: -3.9% [-7.3%, 0.0%] -> not decided
[all] units: the 4 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 85.6% [76.6%, 92.4%] -> not decided
[all] units2: the 10 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 79.3% [66.7%, 93.0%] -> not decided
[all] units3: the 10 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 86.0% [78.2%, 93.9%] -> not decided
[all] real companies: 1 of the 11 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[all, real]   baseline: 63.5% [49.8%, 75.5%] -> fail (the bar: of the companies named, those with a faithful text; 129 faithful of 203 texts + 0 companies left without one)
[all, real]      model: 76.7% [62.7%, 88.1%] -> not decided (the bar: of the companies named, those with a faithful text; 145 faithful of 189 texts + 0 companies left without one)
[all, real]      units: 85.6% [76.5%, 92.3%] -> not decided (the bar: of the companies named, those with a faithful text; 184 faithful of 211 texts + 4 companies left without one)
[all, real]     units2: 79.8% [66.7%, 93.8%] -> not decided (the bar: of the companies named, those with a faithful text; 130 faithful of 135 texts + 28 companies left without one)
[all, real]     units3: 86.5% [78.5%, 94.8%] -> not decided (the bar: of the companies named, those with a faithful text; 141 faithful of 148 texts + 15 companies left without one)
[all, real] model minus baseline on 189 company texts of 30 bodies: 13.8% [8.2%, 21.2%] -> better
[all, real] units minus baseline on 188 company texts of 30 bodies: 20.2% [8.5%, 34.8%] -> better
[all, real] units minus model on 188 company texts of 30 bodies: 6.4% [-3.5%, 20.7%] -> not decided
[all, real] units2 minus units on 163 company texts of 29 bodies: -14.1% [-28.0%, -0.9%] -> not better
[all, real] units2 minus baseline on 163 company texts of 29 bodies: 9.2% [-13.5%, 35.6%] -> not decided
[all, real] units3 minus units2 on 163 company texts of 29 bodies: 6.7% [0.0%, 15.3%] -> not decided
[all, real] units3 minus units on 163 company texts of 29 bodies: -7.4% [-13.0%, -0.9%] -> not better
[2022+]   baseline: 62.9% [34.6%, 86.1%] -> not decided (the bar: of the companies named, those with a faithful text; 22 faithful of 35 texts + 0 companies left without one)
[2022+]      model: 87.9% [69.2%, 100.0%] -> not decided (the bar: of the companies named, those with a faithful text; 29 faithful of 33 texts + 0 companies left without one)
[2022+]      units: 88.6% [73.0%, 100.0%] -> not decided (the bar: of the companies named, those with a faithful text; 31 faithful of 35 texts + 0 companies left without one)
[2022+]     units2: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+]     units3: 100.0% [100.0%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 29 faithful of 29 texts + 0 companies left without one)
[2022+] model minus baseline on 33 company texts of 7 bodies: 21.2% [2.0%, 54.2%] -> better
[2022+] units minus baseline on 33 company texts of 7 bodies: 21.2% [2.0%, 54.2%] -> better
[2022+] units minus model on 33 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units2 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units2 minus baseline on 29 company texts of 7 bodies: 24.1% [2.0%, 59.1%] -> better
[2022+] units3 minus units2 on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units3 minus units on 29 company texts of 7 bodies: 0.0% [0.0%, 0.0%] -> not decided
```

### A.6 Confirmation set, all marks (438 rows)

The first line of the block is the report's own staleness line, read from the record of 2026-11-08, before PR #155: it is out of date (96 rows are dead for the current code and this view still counts them).

```
staleness check (baseline 2026-11-08, model 2026-11-08, units 2026-11-08, units2 2026-11-08, units3 2026-11-08): no marked row is on a text the current code no longer writes
the intervals are a bootstrap over bodies, not over texts (D32)
[all]   baseline: 87.4% [81.4%, 92.2%] -> pass (the bar: of the companies named, those with a faithful text; 195 faithful of 222 texts + 1 companies left without one)
[all]      units: 87.3% [82.5%, 91.5%] -> pass (the bar: of the companies named, those with a faithful text; 193 faithful of 220 texts + 1 companies left without one)
[all]     units2: 91.4% [86.2%, 96.0%] -> pass (the bar: of the companies named, those with a faithful text; 159 faithful of 164 texts + 10 companies left without one)
[all]     units3: 93.7% [90.3%, 96.9%] -> pass (the bar: of the companies named, those with a faithful text; 163 faithful of 170 texts + 4 companies left without one)
[all] units minus baseline on 197 company texts of 30 bodies: -0.5% [-1.4%, 0.0%] -> not decided
[all] units2 minus units on 169 company texts of 30 bodies: -4.1% [-8.7%, -0.6%] -> not better
[all] units2 minus baseline on 169 company texts of 30 bodies: -4.7% [-9.0%, -1.1%] -> not better
[all] units3 minus units2 on 169 company texts of 30 bodies: 2.4% [0.0%, 6.6%] -> not decided
[all] units3 minus units on 169 company texts of 30 bodies: -1.8% [-3.8%, 0.0%] -> not decided
[all] units2: the 7 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 87.8% [82.4%, 92.7%] -> pass
[all] units3: the 7 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 90.1% [85.8%, 94.1%] -> pass
[all] real companies: 6 of the 7 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[all, real]   baseline: 87.9% [82.4%, 92.3%] -> pass (the bar: of the companies named, those with a faithful text; 189 faithful of 214 texts + 1 companies left without one)
[all, real]      units: 87.9% [83.4%, 91.9%] -> pass (the bar: of the companies named, those with a faithful text; 188 faithful of 213 texts + 1 companies left without one)
[all, real]     units2: 90.9% [85.7%, 95.3%] -> pass (the bar: of the companies named, those with a faithful text; 159 faithful of 164 texts + 11 companies left without one)
[all, real]     units3: 93.1% [89.7%, 96.3%] -> pass (the bar: of the companies named, those with a faithful text; 163 faithful of 170 texts + 5 companies left without one)
[all, real] units minus baseline on 191 company texts of 30 bodies: -0.5% [-1.5%, 0.0%] -> not decided
[all, real] units2 minus units on 170 company texts of 30 bodies: -4.7% [-9.3%, -1.1%] -> not better
[all, real] units2 minus baseline on 170 company texts of 30 bodies: -5.3% [-9.7%, -1.5%] -> not better
[all, real] units3 minus units2 on 170 company texts of 30 bodies: 2.4% [0.0%, 6.6%] -> not decided
[all, real] units3 minus units on 170 company texts of 30 bodies: -2.4% [-4.5%, -0.6%] -> not better
[2022+]   baseline: 91.0% [78.0%, 96.6%] -> not decided (the bar: of the companies named, those with a faithful text; 71 faithful of 78 texts + 0 companies left without one)
[2022+]      units: 89.9% [80.8%, 95.8%] -> pass (the bar: of the companies named, those with a faithful text; 62 faithful of 69 texts + 0 companies left without one)
[2022+]     units2: 90.9% [77.4%, 100.0%] -> not decided (the bar: of the companies named, those with a faithful text; 50 faithful of 52 texts + 3 companies left without one)
[2022+]     units3: 96.4% [91.7%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 53 faithful of 55 texts + 0 companies left without one)
[2022+] units minus baseline on 66 company texts of 9 bodies: -1.5% [-3.4%, 0.0%] -> not decided
[2022+] units2 minus units on 55 company texts of 9 bodies: -5.5% [-18.5%, 0.0%] -> not decided
[2022+] units2 minus baseline on 55 company texts of 9 bodies: -7.3% [-18.9%, 0.0%] -> not decided
[2022+] units3 minus units2 on 55 company texts of 9 bodies: 5.5% [0.0%, 18.5%] -> not decided
[2022+] units3 minus units on 55 company texts of 9 bodies: 0.0% [0.0%, 0.0%] -> not decided
[2022+] units2: the 6 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 82.0% [69.4%, 90.9%] -> not decided
[2022+] units3: the 6 companies it wrote no text for and is not asked about counted as unfaithful (the bar on every company named): 86.9% [78.3%, 94.9%] -> not decided
[2022+] real companies: 5 of the 6 exempt companies are not companies (owner's reading); they are out of every extractor's share, the others count
[2022+, real]   baseline: 90.3% [76.1%, 96.2%] -> not decided (the bar: of the companies named, those with a faithful text; 65 faithful of 72 texts + 0 companies left without one)
[2022+, real]      units: 89.1% [80.4%, 95.1%] -> pass (the bar: of the companies named, those with a faithful text; 57 faithful of 64 texts + 0 companies left without one)
[2022+, real]     units2: 89.3% [76.6%, 97.1%] -> not decided (the bar: of the companies named, those with a faithful text; 50 faithful of 52 texts + 4 companies left without one)
[2022+, real]     units3: 94.6% [89.5%, 100.0%] -> pass (the bar: of the companies named, those with a faithful text; 53 faithful of 55 texts + 1 companies left without one)
[2022+, real] units minus baseline on 61 company texts of 9 bodies: -1.6% [-3.8%, 0.0%] -> not decided
[2022+, real] units2 minus units on 56 company texts of 9 bodies: -7.1% [-19.6%, 0.0%] -> not decided
[2022+, real] units2 minus baseline on 56 company texts of 9 bodies: -8.9% [-20.0%, 0.0%] -> not decided
[2022+, real] units3 minus units2 on 56 company texts of 9 bodies: 5.4% [0.0%, 18.4%] -> not decided
[2022+, real] units3 minus units on 56 company texts of 9 bodies: -1.8% [-5.7%, 0.0%] -> not decided
```

## Appendix B: every text metric

Produced by `uv run scripts/text_measures_2026_11_08.py --out data --out data/faithfulness_confirmation`
(in the repository, with a test; the earlier scratch version gave the same figures but for one interval end, +16.5 where the script gives +16.4, a different percentile rule). Text is split into sentences with the repository's
`chunking.split_sentences`; the names in a sentence are found with the same `CompanyResolver` the draw
uses. Each text is for one company, and each sentence of it is put in one group:

- **names the company:** the company's own ticker is among the names found;
- **mixed:** it names the company **and** another universe company (a subset of the line above);
- **names nobody:** no universe company is named (its owner was decided by recency or by the model).

The names are found with the plain resolver, which does not read short forms: a line that `units2` or
`units3` gave to a company by a short form (guard, header rule) shows as "names nobody", so `own%` is not
comparable between `units` and those two on header lines.

A sentence naming only other companies cannot occur (a sentence goes only to the companies it names), so
that group is zero by construction and is not shown. "Texts with a mixed sentence" is the share of texts
with at least one mixed sentence; "other companies per text" counts the distinct other companies in the
mixed sentences. `model` is not in these tables (it is not one of the four compared here).

### B.1 First set (the 30 bodies, 437 texts)

```
          texts  mean ch  median sent/text   own% nameless% mixed sent% texts w/ mixed% others/text
baseline    190      784     384       4.8   39.0      61.0        17.4            60.5        2.00
units       185      420     241       3.3   55.6      44.4        16.9            43.2        0.68
units2      135      510     296       3.9   50.6      44.0        17.2            47.4        0.80
units3      148      476     284       3.7   51.9      42.8        19.1            51.4        0.85

paired differences (same company, same article), 95% interval over the 30 articles
  chars baseline - units: n=185 +370 [+154, +665]
  mixed-share baseline - units: n=185 +7.3 pts [-1.8, +16.4]
  chars units2 - units: n=134 +51 [+14, +122]
  mixed-share units2 - units: n=134 -0.3 pts [-2.2, +1.1]
  chars units3 - units: n=147 +48 [+14, +113]
  mixed-share units3 - units: n=147 +3.5 pts [+0.5, +7.3]
  chars units3 - units2: n=135 +2 [+0, +6]
  mixed-share units3 - units2: n=135 +0.0 pts [+0.0, +0.0]
```

### B.2 Confirmation set (30 other bodies, 395 texts)

```
          texts  mean ch  median sent/text   own% nameless% mixed sent% texts w/ mixed% others/text
baseline    194      952     474       5.7   42.1      57.9        21.3            67.5        1.79
units       193      570     273       3.9   60.3      39.7        22.3            47.2        1.09
units2      157      642     278       4.4   59.6      39.3        23.5            53.5        1.25
units3      162      627     273       4.3   59.9      39.0        24.1            54.9        1.25

paired differences (same company, same article), 95% interval over the 30 articles
  chars baseline - units: n=193 +385 [+248, +540]
  mixed-share baseline - units: n=193 +6.1 pts [+0.1, +12.4]
  chars units2 - units: n=157 +21 [+3, +42]
  mixed-share units2 - units: n=157 +0.8 pts [-2.4, +4.5]
  chars units3 - units: n=162 +19 [+3, +40]
  mixed-share units3 - units: n=162 +2.6 pts [-2.1, +8.4]
  chars units3 - units2: n=157 +0 [+0, +0]
  mixed-share units3 - units2: n=157 +0.0 pts [+0.0, +0.0]
```

Intervals of the paired differences are a bootstrap over the 30 bodies (4,000 redraws), the same idea as
in the faithfulness report. "chars baseline - units" is the characters the baseline writes more for the
same company; "mixed-share" is the per-text share of sentences that are mixed, in percentage points.

What the text metrics show, in one place:
- the baseline writes texts 1.7 to 1.9 times longer than `units`, mostly sentences that name nobody;
- `units`, `units2` and `units3` write texts of similar length (570 to 640 characters on the confirmation
  set) and similar mixing; the guard and header rules do not make the text purer by this count;
- the shared-sentence gap between the baseline and `units` is +7.3 points [-1.8, +16.4] on the first set
  and +6.1 [+0.1, +12.4] on the confirmation set: a lean toward `units`, not a firm gap;
- the faithful rule lets mixed text through, so these metrics see what the faithfulness figure does not.

### B.3 First set, current code (2026-11-10)

```
          texts  mean ch  median sent/text   own% nameless% mixed sent% texts w/ mixed% others/text
baseline    164      829     392       5.1   37.5      62.5        16.2            58.5        1.88
units       160      453     250       3.5   53.6      46.4        15.2            38.1        0.60
units2      128      507     296       3.8   49.6      48.2        14.2            39.1        0.66
units3      141      471     285       3.6   51.1      46.7        16.4            44.0        0.71

paired differences (same company, same body), 95% interval over the bodies
  chars baseline - units: n=160 +381 [+159, +700]
  mixed-share baseline - units: n=160 +9.9 pts [-1.9, +24.7]
  chars units2 - units: n=128 +4 [-7, +16]
  mixed-share units2 - units: n=128 -0.7 pts [-2.5, +0.6]
  chars units3 - units: n=141 +4 [-7, +14]
  mixed-share units3 - units: n=141 +3.3 pts [+0.2, +7.3]
  chars units3 - units2: n=128 +2 [+0, +6]
  mixed-share units3 - units2: n=128 +0.0 pts [+0.0, +0.0]
```

### B.4 Confirmation set, current code

```
          texts  mean ch  median sent/text   own% nameless% mixed sent% texts w/ mixed% others/text
baseline    171     1058     622       6.6   39.2      60.8        19.4            68.4        1.70
units       171      620     290       4.2   59.7      40.3        21.6            47.4        1.16
units2      157      646     294       4.4   59.6      40.2        22.3            50.3        1.19
units3      161      634     278       4.3   59.9      40.0        22.8            51.6        1.19

paired differences (same company, same body), 95% interval over the bodies
  chars baseline - units: n=171 +438 [+325, +562]
  mixed-share baseline - units: n=171 -0.2 pts [-8.2, +8.3]
  chars units2 - units: n=157 +3 [-4, +12]
  mixed-share units2 - units: n=157 +0.4 pts [-2.3, +3.5]
  chars units3 - units: n=161 +0 [-10, +9]
  mixed-share units3 - units: n=161 +2.3 pts [-1.9, +7.7]
  chars units3 - units2: n=157 +0 [+0, +0]
  mixed-share units3 - units2: n=157 +0.0 pts [+0.0, +0.0]
```

### B.5 First set, all rows (453 rows, 2026-11-10)

```
          texts  mean ch  median sent/text   own% nameless% mixed sent% texts w/ mixed% others/text
baseline    204      921     426       5.8   30.5      68.2        12.6            50.5        1.56
units       212      500     264       3.8   46.5      52.5        12.8            34.9        0.54
units2      135      510     296       3.9   50.2      47.7        14.0            38.5        0.64
units3      148      476     284       3.7   51.6      46.4        16.0            43.2        0.69

paired differences (same company, same body), 95% interval over the bodies
  chars baseline - units: n=185 +373 [+167, +659]
  mixed-share baseline - units: n=185 +8.8 pts [-1.2, +21.2]
  chars units2 - units: n=134 +4 [-9, +17]
  mixed-share units2 - units: n=134 -0.6 pts [-2.4, +0.6]
  chars units3 - units: n=147 +4 [-8, +16]
  mixed-share units3 - units: n=147 +3.2 pts [+0.2, +7.1]
  chars units3 - units2: n=135 +2 [+0, +6]
  mixed-share units3 - units2: n=135 +0.0 pts [+0.0, +0.0]
```

### B.6 Confirmation set, all rows (438 rows)

```
          texts  mean ch  median sent/text   own% nameless% mixed sent% texts w/ mixed% others/text
baseline    222      984     548       6.0   37.7      61.1        18.3            60.4        1.71
units       220      609     294       4.1   54.4      45.0        19.4            40.9        0.96
units2      164      634     292       4.3   59.1      40.7        22.0            48.8        1.15
units3      170      616     275       4.2   59.1      40.3        22.3            49.4        1.13

paired differences (same company, same body), 95% interval over the bodies
  chars baseline - units: n=197 +427 [+303, +558]
  mixed-share baseline - units: n=197 -0.2 pts [-7.2, +7.1]
  chars units2 - units: n=160 +3 [-4, +12]
  mixed-share units2 - units: n=160 +0.4 pts [-2.0, +3.4]
  chars units3 - units: n=166 +0 [-9, +9]
  mixed-share units3 - units: n=166 +2.2 pts [-1.7, +7.4]
  chars units3 - units2: n=160 +0 [+0, +0]
  mixed-share units3 - units2: n=160 +0.0 pts [+0.0, +0.0]
```

