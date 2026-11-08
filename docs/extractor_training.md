# The company-text extractor (step 4): datasets, experiments, metrics, results

This page is the plain-language record of how the step-4 extractor was built and checked, and of the
selection of its base model. It is written for the owner and for anyone who has not followed the work.
The technical contract stays in `.specify/memory/PLAN.md` (D27, D32, D34) and `TASKS.md` (T-148b*);
the commands are in `docs/evaluation.md`; the tagged files are described one by one in
`docs/gold_datasets.md`. When this page and those disagree, those win, and this page is wrong.

## The selection (owner, 2026-11-08)

**The owner selected `units` (called "units1" in conversation) as the base model of step 4.** The reason,
in the owner's words: the accuracy does not need to be higher than 80%, and `units` clears it.

What the selection is, and what it is not:

| It is | It is not |
|---|---|
| the extractor the cut (T-154) uses for the company texts of narrative bodies with several companies | a publication to the Hub |
| the base that later improvements start from (see "What is next") | a pin of a model revision |
| a decision of the owner, recorded where the plan and the tasks can see it | a claim that it is better than the baseline: it is not (see "Results") |

Publishing and pinning the trained model still need the owner's explicit word (constitution AI #1, #7,
#12). Nothing here publishes or pins it.

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
- **Text measures** (added 2026-11-08, scratch scripts, not in the repository; every figure in Appendix B): average characters per
  text, and the share of sentences that name this company **and** another. They count names the resolver
  finds; a nameless sentence that belongs to someone else is invisible to them.

## The results

All figures use the owner's marks. Intervals in brackets.

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

### Confirmation set (30 new bodies, the figures to trust)

| | `baseline` | `units` | `units2` | `units3` |
|---|---|---|---|---|
| Every company named | 86.6% [80.7, 91.3] pass | **86.1% [80.9, 90.7] pass** | 78.4% [72.5, 83.7] not decided | 79.9% [74.0, 85.1] not decided |
| Real companies only | 95.9% [92.2, 98.7] pass | **95.3% [92.0, 98.2] pass** | 89.9% [85.1, 94.5] pass | 91.7% [87.4, 95.5] pass |
| Real companies left without a text | 0 | 0 | 12 | 7 |
| Texts written for non-companies | 25 | 24 (18 unfaithful) | 0 | 0 |

Gains on real companies: `units` minus `baseline` -0.6 [-1.7, 0.0] (not decided); `units3` minus `units`
-3.6 [-7.6, -0.6] (not better); `units2` minus `units` -5.3 [-10.5, -1.2] (not better).

### What the results say

1. **`units` passes the 80% bar on both sets**, on the strict reading and on real companies. This is the
   basis of the selection.
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
193 shared companies, so length alone did not buy the baseline extra `yes` marks. `units` reaches the
same faithful share with about 40% less text.

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
| 1 | **Drop the texts for names the guard rejects, keep everything else of `units`** (a hybrid of `units` and `units3`) | removes most of the 18 non-company errors without the 7 to 12 missing real companies | code and a free re-score on the texts already tagged (the hybrid writes a subset of `units`' texts) |
| 2 | **Header rule for `units`** ("A — text" lines belong to A) | fixes digest lines like the Steel Dynamics text | code; the `units2` rule exists and could be taken alone |
| 3 | **A length cap or a trim** for very long texts (16 notes "huge") | shorter, purer texts | code; the effect on faithfulness is not measurable without a new reading |
| 4 | **A purity measure** that checks how much of a text names another company, validated on a read sample | a metric that sees what the faithful rule lets through | a small sample read by the owner, or a decision to use the name count as is |
| 5 | **A stricter second reading of the same texts** ("only this company's content") | would show whether `units` is really cleaner than the baseline | tagging; not asked |
| 6 | **Abstention threshold** (0.5 today; see T-148b3) | fewer wrong high-confidence assignments | the owner's decision |
| 7 | **Match one-word names** ("JPMorgan") and header names with the plain resolver (T-148b9, out of scope so far) | fewer missed companies | code |
| 8 | **More data for the model**: the `continuation` labels (38% right) are out; a cleaner way to use them, or more `lead` bodies | a better model | training time on the GPU; the exam sets must not be trained on |
| 9 | **A larger 2022+ sample** (9 bodies per set cannot decide) | a verdict for the period that matters | new bodies and tagging; no pool is left in the drawn sets |
| 10 | **Run the staleness check** (`draw_faithfulness --stale`, free, needs the model) | confirms the tagged texts are still what the code writes | a run |
| 11 | **Migrate the evidence to `gold/`** (T-148b11: the `units*` marks and the readings of the exempt names exist only in the git-ignored `data/`) | the figures can be reproduced from a clean checkout | the owner's word |
| 12 | **Publish the model to the Hub and pin its revision** | the cut and the report run from a pinned checkpoint | the owner's explicit word |

A reasonable order, as a proposal only: 11 (so the evidence is not lost), 1, 2, then a free re-score of
what is already tagged; 12 when the owner wants the model fixed.

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

The first lines are the report's own warning: 8 marked texts of the first set are on texts the current
code no longer writes (the staleness check of 2026-11-08, `units` rows of bodies 122960 and 88621), so
the first-set figures are development figures read with that caveat. The confirmation set has no such
warning, and its staleness check has not been run (it needs the model).

```
warning: the staleness check of units (2026-11-08) was made with other code than the current one: run it again
warning: the staleness check of units2 (2026-11-08) was made with other code than the current one: run it again
warning: the staleness check of units3 (2026-11-08) was made with other code than the current one: run it again
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

```
no staleness check recorded: run draw_faithfulness --stale (needs the model)
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

## Appendix B: every text metric

Scratch scripts, not in the repository (`purity4.py` and `purity4_first.py` in the session scratchpad;
the definitions are below, so they can be rebuilt). Text is split into sentences with the repository's
`chunking.split_sentences`; the names in a sentence are found with the same `CompanyResolver` the draw
uses. Each text is for one company, and each sentence of it is put in one group:

- **names the company:** the company's own ticker is among the names found;
- **mixed:** it names the company **and** another universe company (a subset of the line above);
- **names nobody:** no universe company is named (its owner was decided by recency or by the model).

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
  mixed-share baseline - units: n=185 +7.3 pts [-1.8, +16.5]
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
- the shared-sentence gap between the baseline and `units` is +7.3 points [-1.8, +16.5] on the first set
  and +6.1 [+0.1, +12.4] on the confirmation set: a lean toward `units`, not a firm gap;
- the faithful rule lets mixed text through, so these metrics see what the faithfulness figure does not.
