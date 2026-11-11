# Scope training: how the scope classifier was chosen and checked (T-147, T-163)

**Status:** selected 2026-11-11 by the owner. The measured model is the one described in
[The selected model](#the-selected-model). Its weights are saved, pinned and public
(section 7); step 1 runs on them (section 8) and its output was validated on `quarantined` bodies (section 9).

This page collects every experiment that led to the scope classifier of step 1 of the pipeline
(PLAN D40): what each one assumed, how it was measured, what came out, and what we learned from
it. It ends with the decision, its advantages, and its limits. Per-task detail is in
`.specify/memory/TASKS.md` (T-147); this page is the story.

## 1. The decision in one table

| | |
|---|---|
| **Task** | Read a unique news body (title + first 800 characters) and say `single` (one analysed company), `multi` (two or more), `sector` (one industry) or `out_of_scope` (market/economy, investing advice, or unrelated). |
| **Selected model** | Two fine-tuned NLI classifiers, both `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (the category model, pinned): stage A decides *company / sector / not related*; stage B decides *single / multi* among the company bodies. |
| **Trained on** | 365 tagged bodies: `category_gold_v2.csv` (284) + `scope_enrich.csv` (51) + `scope_hook.csv` (30). |
| **Measured on** | `category_gold_v2_test.csv`: 200 bodies, never trained on, weighted to the corpus by stratum (D30/D33). |
| **Result** | The **pinned model** (saved weights, section 7): **83.6% [78.1, 88.9]** weighted agreement (new key); the current rules reach 67.2%; paired gain **+16.5 [9.6, 23.4]**; `out_of_scope` recall 53%. The first fit of the recipe, which the rest of this document analyses: 83.0% [77.5, 88.3], gain +15.9 [9.1, 22.6]. **On the old key** (the held-out as first tagged, the conservative one): 81.5%, rules 68.2%, paired gain **+13.3 [6.4, 20.1]**. **Optimistic:** the 200 were seen many times and their key was corrected after the errors were visible (limitations 4 and 5). The interval does not include that selection. |
| **Against the bar** | Under the 90% bar of D20/D33. Above the owner's 80% mark by its point estimate; the lower end of the interval (77.5%) is under it. The owner decided to go with this model. |
| **Known trade-off** | `out_of_scope` recall is about 48% (11 of 33 are read as `single`) while `single` recall is 92%. We accept it on purpose; see [section 6](#6-advantages-and-limitations-of-the-selected-model). |

## 2. What is measured, and how

**Classes.** Four: `single`, `multi`, `sector`, `out_of_scope`. The three out-of-scope tags
(`market_economy`, `investing_general`, `unrelated`) are merged for the classifier, because the pipeline
treats them the same way: the body is stored and stops there. Confusions among those three do not
matter to step 1 (they are reported apart in the gold files).

**Why the classes have different consequences** (PLAN D40):

- `single` goes on to summary, NER and relevance with one main company.
- `multi` is cut into sub-news by the `units` extractor, each piece processed on its own.
- `sector` is summarised like the others; its sentiment is computed later on the sector summaries.
- `out_of_scope` **stops**: it is kept in `body_scope` and nothing else is computed for it.

**Gold data.** Everything is the owner's tagging (docs/gold_datasets.md, sets 1, 2, 7 and 8):

| Set | File | Rows | Use |
|---|---|---|---|
| 1 | `gold/category_gold_v2.csv` | 284 | training (240 of the prompt gate + 44 sector bodies) |
| 2 | `gold/category_gold_v2_test.csv` | 200 | **held-out test**, carries `stratum` and `frame` for weighting |
| 7 | `gold/scope_enrich.csv` | 51 | training; drawn by rules to hold more `sector` and `out_of_scope` |
| 8 | `gold/scope_hook.csv` | 30 | training; drawn for "the company is only the hook" |

The 284 and the 200 are different samples (sector is about 17% of the first and 4.5% of the second), so
every headline figure is **weighted to the corpus** with the held-out strata (`gold_metrics.stratified_report`,
seeded bootstrap intervals). Plain (unweighted) numbers are reported only where we say so.

**Metrics.** Agreement (accuracy over the four classes), and precision and recall per class, each with a
95% interval. The 90% agreement bar comes from D20/D33. Recall per class is gated only where the gold has
40 or more bodies (D33); `sector` (7) and `out_of_scope` (33) are reported with their interval, not gated.

**Paired gain over the rules** (`gold_metrics.paired_stratified_gain`) tells whether a model is better than
the current rules triage on the same bodies, not just by a higher point estimate.

**Two keys.** The held-out tags were reviewed on 2026-11-11 (14 scope changes in 15 touched rows, see
[section 4](#4-the-review-of-the-tags)). Figures are given on the **new key** unless a column says
"old key". Do not compare numbers across keys.

## 3. The experiments, in the order we ran them

Every row says what we assumed, what we did, and what we learned. Weighted figures on the held-out 200
(old key unless said), the models trained on the 284 unless said.

### 3.1 Baselines: what must be beaten

| # | Experiment | Assumption | Result | Learned |
|---|---|---|---|---|
| 0 | Always `single` | The most frequent class is a floor. | 58.2% | Floor. |
| 1 | **Current rules triage** (`size_triage`: companies named, sector words, market words) | It is the thing we want to replace. | **68.2%** [61.5, 74.7]; `multi` recall 52.5%, `sector` 3%, `out_of_scope` 7% | Rules are good at `single` (91%) and almost blind to `sector` and `out_of_scope`. This fails the 90% bar. |

### 3.2 Bag-of-words from the title (and the first lines)

The owner's choice (2026-11-10) was to read the **title plus the first lines of the body**, not the whole body,
because the whole body is expensive and the title is what a reader sees first.

| # | Experiment | Assumption | Result | Learned |
|---|---|---|---|---|
| 2 | TF-IDF (1-2 grams) + logistic regression on the **title** | The headline carries the scope. | 66.4% [59.7, 73.3] | Level with the rules. |
| 3 | The same, title + first 300 / 600 / 1500 characters | More text, more signal. | 65.7% / 62.5% / 57.1% | **No gain:** with 284 rows the body text dilutes the title. A bag of words cannot count companies, which is what tells `single` from `multi`. |
| 4 | Title + **matcher features** (distinct universe companies named in the title and the first lines) | Counting companies fixes `multi`. | **68.1%** [61.3, 74.6] at S = 0.3, chosen by cross-validation on the 284 (S = 1, the first run: 66.9%) | Level with the rules and with the title alone; `single` recall worse than the rules. "No gain" means the intervals overlap, not a ranking. (PR #163.) |

*Lesson: the information is there, but cheap lexical models on 284 rows do not extract it. The ceiling
is about 67% for this family.*

### 3.3 The owner's routes: NER, relevance, embeddings

| # | Experiment | Assumption | Result | Learned |
|---|---|---|---|---|
| 5 | **NER + relevance features** of the top companies: run `gamug/sec-bert-finer-ord-ner` on the first 800 characters, match to the universe, score the top 4 with the relevance model, logistic regression | The companies a body names and how relevant they are decide its scope. | **71.1%** [64.6, 77.2]; `multi` 65%, `sector` 23%, `out_of_scope` 37% | A real gain, but weak on `sector` and `out_of_scope`. Cost: 0.053 s per body, about 4.6 h for 310k bodies (less if batched). |
| 6 | **Two stages with embeddings**: stage A (company / sector / not related) on a sentence embedding, stage B (single / multi) on the relevance features | `sector` was being masked by the company classes; separating the topic first helps. | bge-small 71.3%, **bge-base 74.5%** [68.3, 80.3], bge-large 70.7%. Stage B alone 84% | Better balanced. The base size was picked after looking at three on the same 200, so that figure is optimistic. The learning curve is flat from 71 to 284 rows. |
| 7 | PCA on the embedding of stage A | Fewer dimensions help with 284 rows. | 72.0% | No gain. |

### 3.4 NLI: asking the category model the question

The idea (owner's): stop building features and ask a model that already reads English, as an
entailment question: *"This text is mainly about one specific company."*

| # | Experiment | Result | Learned |
|---|---|---|---|
| 8 | **Stage A as NLI**, topic hypotheses for company / sector / not related: zero-shot, and fine-tuned on the 284 | Topic accuracy 0.84 zero-shot, 0.85 fine-tuned; end to end 72.4% and 73.4% [66.9, 79.7] | Same level as the embeddings: topic accuracy about 0.85. No embedding step is needed. |
| 9 | **Stage B as NLI** among the 165 true company bodies of the 200: *"one single company"* vs *"several different companies"*, seven variants | Relevance features 84.2% (multi recall 31/47); zero-shot 73%; **fine-tuned NLI at 800 characters 90.9%** (multi 39/47) | The first real jump on `multi`. The owner expected NLI to break the recall; it did not. |
| 10 | **End to end**: both stages fine-tuned on the 284, choices fixed in advance, scored once | **79.9%** [73.8, 85.5]; paired gain over the rules +11.7 [4.3, 19.2], "better"; `single` 85%, `multi` 79.5%, `sector` 4/9, `out_of_scope` 17/26 | Best so far, at the 80% mark and under the 90% bar. `sector` is the weak class: only 48 sector bodies in the 284. |

### 3.5 More labels where the model was thin

Experiment 10 failed on `sector` and `out_of_scope`, the classes with the least data. A learning curve on random
labels was flat, so more random tags would not help; targeted tags for the thin classes might.

| # | Experiment | Result | Learned |
|---|---|---|---|
| 11 | **51 tags drawn by rules for `sector` and `out_of_scope`** (set 7); first only to *measure* the rare classes (pooled with the 200: `sector` 20/36, `out_of_scope` 25/35) | 55.6% sector, 71.4% out_of_scope on the pooled set | The model really was weak on `sector`. The owner pointed out the obvious: tags used only to measure and not to train are wasted. |
| 12 | The same models, **trained on the 284 + the 51** | **83.5%** [78.0, 88.8]; gain +15.3 [8.3, 22.4]; `sector` recall 82%, precision 58% | The best result before the review. `out_of_scope` recall stayed at 55%. |
| 13 | A different **wording** of the topic hypotheses ("a company that is only a hook is not related") | 82.4% | No help: after fine-tuning, the labels teach the model; the hypothesis wording barely matters. Dropped; the original wording is the base. |
| 14 | **30 more tags** drawn for "the company is in the title, the story is the market" (set 8), as first tagged | 80.4% [74.5, 85.9]; `out_of_scope` precision fell from 67% to 49% | Worse. Reading them showed contradictory tags: see the next section. |

### 3.6 The review of the tags

Experiment 14 was the turning point. The model had learned two opposite things from the data: the same kind of
article (a Cramer note where an Apple holding is the hook; a CEO interview) was `single` in one file and
out of scope in another. The owner suspected label noise and asked for a critical review of every tagged
file.

| # | Action | Result |
|---|---|---|
| 15 | The owner corrected the contradictory tags of the 30 hook articles and retrained | Accuracy back to **83.5%** (old key): the noise explained the drop. |
| 16 | Read all 284 training tags against the guide and against each other (title scans, v1 tags, similar articles) | **12 scope changes** in `category_gold_v2.csv`; one in `scope_hook.csv`. |
| 17 | Read all 200 held-out tags, **blind to the model's calls** (grouped by tag to compare like with like) | **14 scope changes** (15 rows touched) in `category_gold_v2_test.csv`. The key itself had inconsistencies: crypto tagged three ways, a charity campaign tagged `single`, hook articles tagged `single`. |
| 18 | Recorded the rules the owner settled (labelling spec, owner calls 14 to 21; see [section 4](#4-the-review-of-the-tags)) | Rules now documented. |
| 19 | **Retrained on the corrected training tags** (284 + 51 + 30) and scored on both keys | See the table below. |

| Model | Old key | **New key** |
|---|---|---|
| Trained before the review (tags as first tagged) | 83.5% | 84.9% |
| **Trained after the review (selected)** | 81.5% | **83.0%** [77.5, 88.3] |
| Rules triage | 68.2% | 67.2% |

The honest reading: **correcting the training tags did not raise the score**; it moved it down by about
2 points, inside the noise (the intervals overlap almost entirely, and we did not measure seed-to-seed
variation). The corrected key raised the older model's score by 1.4 points. The remaining errors are a hard
core, not tag noise: see [section 6](#6-advantages-and-limitations-of-the-selected-model).

We chose the model trained on the **reviewed** tags even though its point estimate is not the highest,
because it learned from tags that agree with each other and with the written rules. The point estimate
that looks better (84.9%) comes from a model that learned from contradictory tags.

## 4. The review of the tags

The decisions the owner made while reviewing, which now live in `docs/labelling-spec-v2.md` (owner calls
14 to 21) and in memory:

- A company's charity, awareness or sponsorship press release is `unrelated` (campaigns, charity golf).
- A company's product news that the public consumes (a vaccine booster, a drug trial) is `single`, with
  `product_innovation` and `legal_regulatory`.
- A CEO interview, transcript or media alert is `investing_general`.
- A Cramer or analyst piece that uses a company as a hook for a portfolio or market lesson is
  `investing_general`; crypto, an asset and not an industry, is `market_economy`.
- An investor or counterpart (3G Capital, Icahn) is a mention, so the article is `single`; the two parties of
  a deal stay `multi`.
- A company outside the 509 universe is analysed like any other: a press release about it is `single`.
- A study that ranks brands and gives no financial fact is `unrelated`.

Changes: **training** `category_gold_v2.csv` 12 scope changes (126761, 139632, 169716, 255218, 263942,
264983, 321594, 327116, 341135, 413001, 414161, 421789) and `scope_hook.csv` 266351; **held-out**
`category_gold_v2_test.csv` 14 scope changes in 15 touched rows (414029 gained a second company label and kept `single`) (121050, 164940, 165364, 179104, 247481, 252938, 285334,
296056, 296388, 302476, 328763, 344029, 414029, 479545, 1110708). Some `sub_industry` cells were mangled by the
editor (GICS codes read as numbers) and were restored on a copy; the owner's files were not edited.

**A caution about the held-out key.** The held-out tags are the measuring stick, and 14 scope labels were changed after
we had seen the model's mistakes. To limit selection bias, a tag was changed only when it broke a written rule or
contradicted a similar article, the model's calls were not shown while reading the 200, and figures are
reported on **both** keys. The old key is the more conservative one.

## 5. The selected model

**Architecture.** Two copies of one pinned model, each fine-tuned separately; the final scope is
`sector` if stage A says sector, `out_of_scope` if it says not related, otherwise stage B's `single` or `multi`.

- Base: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` at revision `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8`
  (the `category` entry of `config/pipeline_models.json`).
- Input: `title + ". " + first 800 characters of the body` as the premise (`scope_relevance_features.fragment`).
- Stage A hypotheses: *"This text is about one specific company or a few companies."* (company), *"This text is
  about an industry or sector as a whole."* (sector), *"This text is about the general market, the economy or investing
  advice, or it is not about companies."* (not related). Trained as binary entailment, one hypothesis per topic;
  the prediction is the highest entailment minus not-entailment score.
- Stage B hypotheses: *"This text is about the news of one single company."* vs *"This text covers the news of
  several different companies."*, trained only on the company bodies.
- Recipe: AdamW, lr 2e-5, weight decay 0.01, linear schedule with 10% warm-up, 3 epochs, batch 16, gradient
  clipping 1.0, float32 weights with bf16 autocast, seed 1; max length 320 (stage A) and 512 (stage B).

**Training data (365 bodies):** single 131, multi 79, sector 73, out_of_scope 82. Stage B sees the 210 company
bodies. Texts are in `gold/texts/`. Because set 7 and set 8 were drawn by rules, the training mix is **not**
the corpus mix; the held-out figures are weighted for that.

**Reproduce:**

```bash
uv run python scripts/scope_end_to_end_2026_11_10.py --train-on-enrich --out data/scope_end_to_end_reviewed.csv
```

It trains both stages on the 284 + 51 + 30, scores the held-out 200 (weighted, with the paired gain over the
rules) and writes the predictions. It takes 15 to 20 minutes on one GPU.

**Rerun (2026-11-11, same command, same seed, same machine): not bit for bit.** It gave **83.6% [78.1, 88.9]** (the doc's run: 83.0%), paired gain +16.5 [9.6, 23.4] (+15.9), `out_of_scope` recall 19/33 (18/33); one held-out body changed its call. So the setup reproduces within about 1 point, and the two figures are one sample of the run-to-run noise (GPU kernels, bf16). Unlike the relevance model (T-169), the weights are not pinned for a bit-for-bit check; T-147b adds that and the three seeds. The scripts of every experiment are
`scripts/scope_*_2026_11_10.py`; the gold builders are `scripts/build_gold_2026_11_05.py` (`--set train|test`)
and `scripts/build_scope_enrich_gold_2026_11_10.py` (`--set scope_enrich|scope_hook`).

**Results on the held-out 200, new key, weighted (agreement 83.0% [77.5, 88.3], plain 82.5%):**

| Class | Gold | Predicted | Precision | Recall |
|---|---|---|---|---|
| `single` | 116 | 126 | 85.7% [79.1, 91.7] | 92.2% [86.6, 97.0] |
| `multi` | 44 | 39 | 91.4% [81.6, 99.5] | 80.0% [69.5, 91.4] |
| `sector` | 7 | 9 | 62.0% [24.5, 97.0] | 84.3% [50.8, 98.1] |
| `out_of_scope` | 33 | 26 | 59.3% [36.4, 81.6] | **48.1%** [27.4, 69.0] |

Confusion matrix (plain counts), rows = gold, columns = predicted:

| gold \ predicted | single | multi | sector | out_of_scope |
|---|---|---|---|---|
| `single` | 107 | 2 | 1 | 6 |
| `multi` | 7 | 35 | 1 | 1 |
| `sector` | 1 | 0 | 5 | 1 |
| `out_of_scope` | 11 | 2 | 2 | 18 |

## 6. Advantages and limitations of the selected model

### Advantages

- **Clearly better than what we have.** The paired gain in weighted agreement over the rules on the same bodies is
  **+15.9 [9.1, 22.6]**, "better" (the interval excludes zero); 83.0% against 67.2%.
- **It sees what the rules cannot.** The rules call `sector` and `out_of_scope` almost never right (3% and 7% recall);
  the model gets `sector` right in 5 of 7 and `out_of_scope` in 18 of 33.
- **`single` is protected.** Recall 92%: a real company article rarely reaches the "stop" class.
- **`multi` is trustworthy when it speaks.** Precision 91%: a body called `multi` is almost always `multi`, so the
  `units` cut is rarely wasted.
- **No new model to maintain.** It reuses the pinned category model; no embedding step, no NER at this step.
- **Cheap at read time.** Two short forward passes over 800 characters.
- **Trained on tags that agree.** The tags were reviewed against written rules (labelling spec, calls 14 to 21).

### Limitations

1. **The trade-off we accept: `out_of_scope` against `single`.** Out-of-scope recall is 48% and `single` recall is
   92%. Eleven out-of-scope bodies are read as `single` (a macro story with a company in the title, a CEO
   interview, a charity press release, three Google stories); six `single` bodies are read as `out_of_scope`
   (a non-universe press release, two CNBC transcripts, a Google op-ed, two Pfizer stories). The training data has few out-of-scope bodies whose
   title names a company, so the model leans on the company name.
   *Why we accept it:* an out-of-scope body read as `single` moves on, and the relevance step can still catch
   it (D40: a body with no ticker over the relevance threshold goes to the review list). A `single` read as
   out-of-scope **stops at step 1 and never reaches the company views**. The first error costs some compute and
   perhaps a review item; the second costs a news item. We prefer the first.
   *What it means in numbers (the corpus run of section 8: 11.3% of the `ok` bodies and 58.8% of the `quarantined` ones; read it first):* expect about half of the out-of-scope bodies to go on to steps 2 and 3. Weighted to the corpus, `out_of_scope` is about 12.8% of the unique bodies and the ones read as something else about 6.7%; if the frame mirrors the 309,882 bodies, that is about 39,700 out-of-scope bodies and about **20,600 that continue** (6.7% extra work in steps 2 and 3, plus their entries in the review list). This assumes the 1,000-body sample of D30 (the `frame` sizes sum to 1,000) mirrors the unique-body corpus; the 309,882 were not weighted directly. Do not
   present step 1 as a filter of out-of-scope news.
2. **Under the 90% bar** of D20/D33 (83.0%, interval 77.5 to 88.3). The owner decided to go with it; a model that
   clears 90% would need more data in the thin classes or a larger model.
3. **Small evaluation for the rare classes.** The held-out 200 has 7 `sector` and 33 `out_of_scope` bodies, so
   their intervals are wide (`sector` recall [51, 98], `out_of_scope` recall [27, 69]). D33 does not gate recall
   below 40 gold bodies. We cannot claim more.
4. **Seed noise was barely measured.** Three fits of the selected setup, all seed 1: 83.0%, then 83.6% twice (the rerun and the saved weights give the same prediction on all 200 bodies); that is not a noise estimate. The three-seed check is in T-147b. A 2-point difference
   between two runs is not evidence of anything.
5. **Some selection optimism.** Several design choices (the embedding size, the NLI variants of stage B) were made
   on the same held-out 200 earlier; the final end-to-end choices were fixed in advance, but the sample was seen.
   The held-out key was also corrected after the model's mistakes were visible (see the caution in section 4).
6. **The training set is not the corpus.** Sets 7 and 8 were drawn by rules. The weights correct the held-out
   figures, but production proportions may shift the precision of `sector` (a rare class in the corpus).
7. **The tags still carry judgement.** The hard core of errors (company in the title, macro story) is exactly where
   two readers can disagree; the rules in section 4 reduce this, they do not remove it.
8. **Short input.** It reads the title and the first 800 characters; a body whose scope only shows later (a
   transcript with a long preamble, a roundup) is read from its opening. The two transcript errors are of this kind.
9. **Not one model.** Two models are loaded and run per body; the cost is small, but both must be pinned.

## 7. What is not done

- **Owner of this list: T-147b** (FTI + MLflow, pin, three seeds). The scripts need a GPU (`.cuda()` is fixed in them).
- **The weights are saved, scored and pinned (owner, 2026-11-11).** `scripts/train_scope_final_2026_11_11.py` refits both
  models on the 365 bodies into `models/scope_final` (git-ignored); step 1 runs on them. `scripts/score_scope_heldout_2026_11_11.py`
  scores the held-out 200 through the step's own classifier: **83.6% [78.1, 88.9]** weighted (rules 67.2%), paired gain
  **+16.5 [9.6, 23.4]**, `single` recall 92.2%, `multi` 80.0%, `sector` 84.3% (7 gold bodies), `out_of_scope` 53.0% [32.3, 73.8]
  (confusion matrix: 19 of 33 out-of-scope bodies found, 11 read as `single`). This is the same figure as the rerun of section 6,
  so the saved weights reproduce it; it is still one seed. The weights are in the public repository
  `gamug/deberta-v3-news-scope-nli@d14a4d20` (moved to `@58e4af4b` on 2026-11-11 by a card-only commit, same weights; section 9) and pinned in `config/scope_model.json` with the step revision
  (`scripts/publish_scope_model_2026_11_11.py`, `--check` verifies it). Five uploads: `5f5b2c27` lost the weights (the script did not follow folders); `3f5a85f9` had them; `2621fb4a` and `59964ab7` corrected the card; `d14a4d20` is the same weights with the card for a public repository, and is the pin. The first four stay in the history. The repository was made public on the owner's word (2026-11-11), after the upload was verified.
- **Not on the FTI pattern and not logged in MLflow** (T-147b). Like the relevance model, the scope scripts are plain scripts.
- **No refit on all tags yet.** The plan is to refit on the 365 + the 200 held-out only once the bar is
  shown or the owner decides; until then the held-out 200 stays untouched by training.
- **Step 1 and the sub-news cut are built** (section 8); the size measurement of sub-news (T-166) is not.
- **Seed variation** (three seeds on the selected setup) is the cheapest check that would make the 2-point
  differences interpretable. About 45 minutes of GPU.
- **More out-of-scope bodies whose title names a company** are the data that would address the trade-off.

## 8. Step 1: running the classifier on the corpus (T-163)

```bash
uv run python scripts/train_scope_final_2026_11_11.py            # GPU; refits both models on the 365 bodies, ~12 min on a 6 GB RTX 4050
uv run cli/news_nlp_steps.py --step scope --limit 500            # classify up to 500 unique bodies not done at this revision
uv run cli/news_nlp_steps.py --step scope --status [--titles]    # counts by state and scope; --titles scans the corpus (minutes)
```

The step reads the **properly fetched corpus** (`ok` and `quarantined`, D36), one item per unique body, in article-id
order. The revision stored with each row is a hash of the training record and the weight files, so a refit is a new
revision and its bodies run again. A batch that fails is redone body by body, so one bad body fails alone; failed bodies
are retried by the next run. If three batches in a row fail completely (out of memory, a lost GPU, a corrupt weight
file) the run stops with a message instead of failing the corpus.

**A body that stays `failed`:** `--status` shows the count and the last three errors (`failed_examples`); the full error
is in `pipeline_step_state.error` for that body (`item_id` = the body hash). Fix the cause and run the command again: only
the failed and not-yet-done bodies run. Nothing skips a body by itself; there is no attempts cap, and `attempts` counts the tries.

**Dry run (2026-11-11; fit `models/scope_final`, revision `scope-nli-8ff21e50796b`; first 500 unique bodies by article id of `urls.db`, scratch RESULTS file, GPU RTX 4050):**
500 done, 0 failed; single 220, multi 171, out_of_scope 101, sector 8. `--limit 5` on top added 5 rows and changed none. **A rerun adds nothing:** on a 59-body source (the first 60 article ids)
the command ran twice with `--limit 500`: the first run wrote 59 rows in `body_scope` and in `pipeline_step_state`, the second
ran 0 bodies and left both tables and their timestamps unchanged. The same 500 bodies gave the same counts when run again
on a fresh file (inference is repeatable; training is not). **Not a corpus sample:** the first 500 hold 331 `ok` and 169 `quarantined` bodies; out_of_scope is 30 of 331
(9%) among `ok` and 71 of 169 (42%) among `quarantined`, which fits quarantined pairs being the ones whose company is not
the subject (D36) and is why the 500 differ from the 12.8% estimate. Bodies with no title in the whole corpus: 90.
**Speed:** about 29 bodies/s once running, so the 309,882 unique bodies are about 3 hours of GPU; start-up is ~10 s.
The held-out figures are not printed by the command: `scripts/score_scope_heldout_2026_11_11.py` scores the saved weights
(section 7). The command prints whether its model is the pinned one.

**The corpus run (2026-11-11, the pinned revision `scope-nli-8ff21e50796b`).** The RESULTS file has **309,972** bodies with a scope,
all the unique bodies of the corpus (`ok` and `quarantined`, including the 90 without a title). (A first reading gave 309,914
and "58 left": it opened `nlp.db` without its `-wal` file, which held the last rows. Read a copy of `nlp.db` **with** its
`-wal`, never the main file alone.) The breakdown (`scripts/scope_corpus_breakdown_2026_11_11.py`, no GPU) gives:

| Status | Bodies | single | multi | sector | out_of_scope |
|---|---|---|---|---|---|
| `ok` | 64,901 (21%) | 58.6% | 26.4% | 3.7% | **11.3%** |
| `quarantined` | 245,071 (79%) | 22.4% | 6.3% | 12.4% | **58.8%** |
| all | 309,972 | | | | about 49% |

By year `out_of_scope` runs from 29% (2026) to 56% (2020); no single year explains it. **The 12.8% of the
cost estimate (limitation 1) is the share of the `ok` bodies (11.3% here); the corpus is 79% `quarantined`, whose pairs were
judged not to be about the tagged company (D36), so most of their bodies are not company news.** I read a random 50 of the
`quarantined` bodies called `out_of_scope` (one reader, not a measurement): about 40 are macro, market, politics or lifestyle
news with no company (correct); about 8 have a company hook (a stock-picking show, a market wrap with a takeover, a company
CEO interview, a company's award press release) that the tagging rules treat as `investing_general` or `multi`, so they sit
on the line; **2 are clear errors**: an EADS-BAE merger story, and a Paramount-Skydance headline whose body is a paywall page.
**What stays open:** the held-out gold was drawn from the 1,000-body sample (D30) and its `out_of_scope` recall (53%) is
measured there; whether that sample covers the `quarantined` bodies is not stated in D30, and the `quarantined` bodies are
79% of what step 1 now classifies. The share that continues in the corpus is therefore not known; the owner should decide
whether a sample of `quarantined` bodies is tagged and scored before step 2. **Done: section 9.**

**Quality check of the corpus run (2026-11-11, `scripts/scope_quality_check_2026_11_11.py`, read-only, no GPU).**
- *Integrity:* 309,972 rows, one per body (no duplicates), all at the pinned revision with `text_used = title+800`; no non-finite
  score; **the stored scope is what its stored scores give in 100% of the rows**; the state table has 309,972 `done`, no
  `failed`, no result without a state and no state without a result; no scope for a body outside the corpus.
- *Complete:* every unique body of the corpus has a scope (a read that missed the write-ahead log showed 58 missing; they were there).
- *Short bodies:* under 300 characters (1,915 bodies) 73% are `out_of_scope`; from 3,000 characters 56%; between, 41 to 43%.
  Length moves the share but does not explain it.
- *Unusable bodies:* **204 bodies look like a paywall or a bot page** ("security verification", "enable JavaScript", "subscribe
  to read"…); 61 of them still get `single` (53), `multi` (8) or `sector` (3) because the headline reads as company news.
  Later steps (NER, summary, sentiment) would run on that page text: they should be filtered or flagged before step 2.
- *No title:* 90 bodies, 89 `out_of_scope`. T-147's report asks for a missing title to be stored with the status `review`;
  **the step does not do it yet**, they are decided by the body alone.
- *Low confidence:* 10,274 bodies (3.3%) have a margin under 1 between the two best topics: a ready list for a review queue.
- *Reading* 14 `single`, 14 `multi` and 14 `sector` bodies at random: all 14 `single` plausible; 12 of 14 `multi` (two on the line);
  about 10 of 14 `sector` (a social-media story about Uber, an analyst's two bank picks and a stock-picking segment read as
  sector). One reader, 42 bodies: it agrees with the measured 80 to 90% and finds nothing of another kind.

**Validation of a quarantined sample (owner, 2026-11-11).** `scripts/draw_scope_quarantined_check_2026_11_11.py` writes
`data/scope_quarantined_to_validate.csv`: 200 `quarantined` bodies drawn at random per **predicted** scope (100 `out_of_scope`,
50 `single`, 25 `multi`, 25 `sector`), none already tagged in a gold file, shuffled, with the title, the first 2,000
characters, the Spanish title and text, and the empty columns `validator` (`yes` the predicted scope is right, `no` it is not),
`correct_scope` (only for a `no`) and `notes`. `scripts/score_scope_validation_2026_11_11.py` turns the filled file into the
share of `yes` per predicted class with its 95% interval and the share of all quarantined bodies the step tags right.
*How many:* the 100 `out_of_scope` rows give that class's precision to about ±7 points (95%, at a precision near 85%), and it is the class that decides
how many company stories the step would stop; 50 `single` give ±10, 25 `multi` or `sector` ±14 (at a precision near 85%). Read the 100 `out_of_scope`
first: if 95 or more are `yes` the main worry is gone and the rest can wait; if 85 or fewer, read all 200.

### 8.1 Step 1.2: the cut (T-163)

`uv run cli/news_nlp_steps.py --step cut [--limit N]` cuts the `multi` bodies of step 1 with the `units` extractor the owner
selected on 2026-11-08 (T-148b10), read as `units4` (T-176, default since 2026-11-11; `--reading units` cuts as before) and writes one `sub_news` row per company text. Contract: `docs/modules/news-nlp.md`, "Step 1.2".

**Dry run (2026-11-11, 232 multi bodies of the corpus run, in order, model on the GPU, a scratch copy of the databases).**
232 `done`, 0 `skipped`, 0 `failed`, 1,639 `sub_news` rows (an earlier run on another 232 gave 1,683). A rerun adds nothing.
About 2 min to load the model, then 0.4 to 0.6 s a body (the model scoring is about 92% of it): **the ~19,775 multi bodies of
the corpus take about 2.5 to 3 h** on the GPU. The full run is for the owner to start; no speed-up is included (fp16 or pair
pruning could change the output).

**Quality check (`scripts/cut_quality_check_2026_11_11.py`, read-only):**
- *Sub-news per body:* min 1, median 7, mean 7.1, p90 13, max 24; 24 bodies (10%) give a single sub-news.
- *Length:* median 268 characters, p10 94, p90 590, max 7,843; **128 of the 1,639 (7.8%) are under 80 characters**.
- *Not cut:* 11 bodies (4.7%) have a sub-news that is 80% or more of the body. **All 11 were read: every one is a story
  about two companies (3M and Avery Dennison ending a deal, Ford and 3M making respirators, Shire and AbbVie, Seattle Genetics
  and Abbott…), written as one piece.** The cut is right to leave them whole: the text is one sub-news that both companies
  read (the sub-news keeps no ticker). They are not a step 1 error.
- *Short pieces:* 128 pieces (7.8%) are under 80 characters, **but only 50 of them end like a sentence; 78 (4.8%) are
  fragments** ("Marvell Technology rallied 12%,", "Nvidia — The chipmaker rose more than 3%,"), a clause cut in half. 27 are
  under 30 characters and 6 are only a name and a dash ("Micron Technology —"); 19 bodies (8.2%) have a piece under 30.
- *Where the sentence of a heading-only piece went (all 6 traced in the body):* it is **not lost from the body**; it sits in
  another piece in all 6. In 5 of 6 that piece does not name the company (Quest Diagnostics, Eli Lilly, Abbott Labs, Micron,
  Qorvo/Skyworks): the heading is the only unit that names the company, the sentence names none and went to the company the
  model chose, so **the company keeps a heading and its sentence is read as another company's news**. In 1 (Norwegian…) the
  sentence's piece starts with the full heading and the heading-only piece is a duplicate. (Pieces are in the order of the
  companies, not of the body, so "the next piece" says nothing; an earlier version of this paragraph read it that way.)
- *Several companies in one piece:* **162 pieces (9.9%) in 79 of the 232 bodies (34%) hold two or more "Name —" headings**
  (`Wendy's (WEN) – … Lowe's (LOW) – …`; `RTX — … Calix — …`), a list written in one paragraph that was not split. That text is
  read by one company and carries the others' news. This is the larger fault of the two, and it is the same cause. Five read:
  all five are real (a text with three to six companies in it). **Both faults were T-176**: `units4` fixes most of them and is the default of the cut since 2026-11-11 (`docs/extractor_training.md`,
  "Update of 2026-11-11"); the numbers of this section are those of `units`.
- *Skipped:* none in this sample, so the "no company text" path is covered by tests only.
- *Same input, same output (`scripts/cut_determinism_check_2026_11_11.py --results-db … --source-db … -n 30`, exit 1 if any differs):* the 30 first bodies were cut again, in another process (against the stored rows) and twice in
  one process: **0 of 30 differ in both**, at the same revision `units-75642e3f4f72`. (An earlier dry run gave 1,683 rows on
  a set of 232 bodies that was not kept; it cannot be compared, and nothing in this check suggests a difference.)
- *Reading 4 bodies first, then the flagged ones:* a "stocks making moves" list is cut well, one sub-news per company; a deal
  story with two companies stays whole; a one-company story stays whole.
- *Reading file:* `--sample N --out data/cut_sample_to_read.csv` writes every flagged body (not cut, or with a tiny piece)
  and random ones up to N, with a `flag` column. It is for reading, not tagging (no validator column, so no Spanish columns).

**The default through the real command (2026-11-11, review of #169).** The figures above and in `docs/extractor_training.md` come from
`compare_cut_readings` on 232 bodies; this is `--step cut --limit 500` with the default reading (`units4`, model
`data/models/attribution_2026_11_08`, GPU, scratch copies of `nlp.db` and `urls.db`), revision `units4-e1dff4f90b24`, then the two checks on that revision:
- *Run:* 500 `done`, 0 `skipped`, 0 `failed`, 3,134 `sub_news` rows; 1 min 47 s after the model load (4.6 bodies/s: the
  first 500 bodies are faster than the 0.4 to 0.6 s of the 232 above, so the 2.5 to 3 h estimate is not lowered).
- *`cut_determinism_check -n 30`:* **0 of 30 differ** from the stored rows and between two runs.
- *`cut_quality_check`:* sub-news per body min 1, median 5, p90 11, max 26; 41 bodies (8.2%) give a single one; length median
  268, p10 103, p90 672; 222 pieces (7.1%) under 80 characters, 122 of them fragments; 21 bodies "not cut" (4.2%); **heading-only
  pieces 0**; pieces under 30 characters 48 (32 bodies); **50 of 3,300 company texts (1.5%) hold several headings (42 bodies)**, in line with the
  23 of 1,589 (1.4%) of the 232.
- *The 48 pieces under 30 characters are almost none a heading (one, "Eli Lilly — The", is a cut in the middle of a sentence):* they are the end of a name cut at an abbreviation ("J.B.", "C.H.", "T.",
  "Citigroup upgrades Arthur J."), a dateline or a place after a name ("Apple, Cupertino, Calif."), and the owner-of-position notes
  of a TV roundup ("Terranova owns (XOM)", "Finerman Owns (WMT)", "(HPQ)"). The first group is the abbreviation-split fault
  already known for clauses; none is a company's news. Not a task of this PR; the quality check counts them as tiny, as before.

**The 23 texts that still hold several headings (18 bodies of the 232), read.** For each text the headed sentences were
split and the company that owns the text was looked for in every one (`scripts/several_headings_owner_2026_11_11.py`, same bodies, same model):
- **10 are legitimate overlap.** The owner of the text is named in each line: Goldman Sachs in the Corning and Quest
  lines, Apple in Intel's (examples from the 500-body run). By the rule of
  `units4` ("and those its text names") that text is the company's.
- **13 are a third fault: a line given to a company it does not name.** The header of the next line is not read, so the line stays
  with the owner before it. 8 are a head in **capitals that names no universe company**, taken for a dateline or label by
  `header_owners` ("SAP —" into General Motors' text, "BP —" into Allstate's and J. M. Smucker's, "UBS –" into Tesla's, "ASML —" into
  Eli Lilly's, "WPP (WPP) –" into Dollar Tree's, "DSW –" into Quest Diagnostics', "AT & T —" into Delta's); 5 are a **head with lowercase words**
  ("Memory stocks —" into Intel's, "Cruise operators —" into Accenture's, "China electric vehicle stocks —" into Moderna's twice, "Jack in the Box —"
  into Darden's). Over the 500 bodies of the real command: 19 overlap, 31 this fault (a few of them a dash inside a sentence that the
  quality check takes for a heading, for example "The company's bet on Reels is facing hurdles — Instagram users…"). **Opened as T-177**
  (a header read for these two shapes, as a new reading with its own revision, measured on the same 232 bodies and the owner's marks
  before it replaces the default). It does not block this PR: the fault is under 1% of the texts (13 of 1,589), and the 162 → 23 stands.

**Two risks of the code review of #169, measured (2026-11-11, same copies; no code changed).**
- *A mid-line "Name — text" whose name is of no universe company gives the rest of the line to nobody.* In 500 bodies there are 352 such
  headers (232 distinct heads, 89 bodies, among 13,333 sentences). Read in full, **none is a prose dash** ("Analysts - X said"): they are
  companies outside the universe (Utz Brands, Zions Bancorporation, Pinterest) and a few group labels (Airlines, Semiconductors, Steel).
  The rule is what keeps their lines out of universe companies' texts: cutting 500 bodies again with mid-line headers of nobody ignored
  changes 297 of 3,300 company texts (9.0%, 82 bodies): 0 lose text, 110 gain another company's lines (a Utz Brands line in Danaher's text).
  Open: some heads look like universe companies the resolver did not match (Disney, JPMorgan, Bank of New York Mellon, Walgreens Boots
  Alliance, Zions); not checked whether they are quarantined-only companies (not in the universe by construction) or a matching gap.
- *A bracketed ticker may keep the header from being read.* In 3,000 bodies, 353 sentences have "(…) –" in their first 90 characters; 321 (91%) are read as
  headers. Of the 32 not read, 13 are not headers (Reuters datelines, "(from Outperform) –", prose; the `(NYSE: X)` form appears once and is prose) and
  **about 17 are headers missed (0.6% of bodies)**: 6 all-caps names ("BP (BP) –", the T-177 cause), **6 names with a period ("Amazon.com (AMZN) –", "Box Inc.
  (BOX) –", "Fox Corp. (FOXA) –"; `_HEADER` excludes `.` from the head, checked on the regex)** and 5 with lowercase text after the dash ("American Express
  (AXP) — solid momentum…"). The last two causes are added to T-177.

**Cutting again.** The bodies the cut has done are found in `pipeline_step_state` (step `cut`, `model_revision`), not in `sub_news`, so
`DELETE FROM sub_news` does not make the cut run again. At the same revision, in the RESULTS database:

```sql
DELETE FROM pipeline_step_state WHERE step = 'cut' AND model_revision = '<revision>';
DELETE FROM sub_news WHERE extractor_revision = '<revision>';   -- written with INSERT OR IGNORE: delete them too to have them rewritten
```

or keep the old rows under their own revision: a new reading, model or code changes the revision and starts new rows by itself, and
nothing is deleted. The same text is in `uv run cli/news_nlp_steps.py --help`.

**Which revision the later steps read.** The default changes the revision of the cut from `units-<12 hex>` (the `units` reading,
for example `units-75642e3f4f72` of the dry run above) to `units4-<12 hex>` (this PR's real run: `units4-e1dff4f90b24` on a scratch copy; it changes if
the code of `attribution_units.py`, `attribution_model.py` or the model changes). **T-165 and any script read `extractor_revision` of the `units4` set
by name, never "the latest"**; `--step cut --status` lists the latest revision with state and the others, so after the full run
the old `units-…` rows (if any were written outside a scratch copy) and the `units4-…` rows are both listed.

**Before the full run.** (1) T-176 is done: the cut runs with `units4`, which removes most of the two faults above (162 texts with
several companies become 23; the 23 are read above: 10 overlap, 13 a third fault, T-177). (2) The scope of the `quarantined` bodies (79% of the corpus) was validated by the owner (section 9; the tags are in
`gold/scope_quarantined_validation.csv`): a wrong `multi` there costs GPU time (about 0.5 s a body) and `sub_news` rows the
later steps would read as company news, so **the rows of the cut are not used by the steps after it (T-164, T-165) until the owner
releases the gate** (released on 2026-11-11, no flag; section 9, "The gate of section 8.1"; the condition is also in T-165). The run writes a new revision of rows and nothing else.

## 9. Validation of step 1 on quarantined bodies (owner, 2026-11-11)

**Why it was needed.** The held-out 200 (section 5) were drawn from the 1,000-body sample (D30). Whether that sample
covers `quarantined` bodies was not stated, and they are 79% of what step 1 classified (section 8). The corpus run
therefore had no measured quality on most of its output. This section closes that gap for the `quarantined` bodies. The
`ok` bodies (21%) are not covered by it; their figure is still the held-out 83.6%.

**Where the numbers come from.**
- *Sample:* `scripts/draw_scope_quarantined_check_2026_11_11.py` drew 200 `quarantined` bodies at random **per predicted
  scope** (100 `out_of_scope`, 50 `single`, 25 `multi`, 25 `sector`), none already tagged in a gold file, shuffled, with the
  title, the first 2,000 characters and a Spanish translation (`draw` = `main`). A second draw of 75 `multi` bodies (`--counts multi=75 --exclude
  data/scope_quarantined_to_validate.csv --out data/multi_extra --seed 20261112`; 81,251 tokens of paid translation, 75 calls, 0 failed) was made so that `multi` would not rest on 25 bodies.
  **The owner read the first 25 of its shuffled order (a random subset) to bring `multi` to 50, like `single`, and decided not to read the other 50** (`draw` = `multi_extra`). The design read is therefore
  100 `out_of_scope`, 50 `single`, 50 `multi`, 25 `sector` (225 bodies). The unread 50 stay in `data/multi_extra/` (git-ignored) and are not used.
- *Tags:* the owner read each body and answered `yes` (the predicted scope is right) or `no` (with the scope it should be). The working files are in `data/` (git-ignored: they hold news text).
  **The tags are tracked in `gold/scope_quarantined_validation.csv`** (ids and tags, no text): `validator_first` / `correct_first` (first reading), `validator` / `correct_scope` (final), and `changed` = `yes` where the final differs from the first (9 rows of the main draw, none of the extra).
  **The first-reading key of the main draw is a reconstruction.** The owner's file was edited in place and the original was not saved; it was rebuilt from the rows of the first review (the 24 `no` and their corrections) and checked against that review's per-class counts
  (94 / 45 / 16 / 21 right, 89.6% overall), which it reproduces. It is a faithful rebuild of what was reported, not a copy of the owner's file.
- *Scoring:* `scripts/score_scope_validation_2026_11_11.py` reads that file (CPU only, no model is loaded). Per class: the share of `yes` with its 95% Wilson interval. Overall: the
  classes weighted by their size among quarantined bodies (260,356; the pool of the corpus run of revision `scope-nli-8ff21e50796b`, counted again on the real files: `out_of_scope` 146,248, `single` 63,001,
  `sector` 31,157, `multi` 19,750 before the bodies were taken out; section 8's 245,071 counts the bodies by another rule, and weighting by its shares changes the figure by under a point). The overall
  interval is a stratified normal approximation (`weighted_precision`), rough for classes of 25. The default is both draws pooled; `--draw main` scores the first 200 only, `--key first` the first reading.

**Two keys, as in section 5.** The tags were revised after the first results were seen, so both are reported. The revisions were made against `docs/labelling-spec-v2.md`, never against the model's output alone (audit below).
Three of the changes (#10, #45, #66) were written into the file by the assistant at the owner's request, so the final key is *one reader plus an audit*. The 25 extra rows were audited too (below) and none changed.

| Predicted scope | Read | First reading [95% CI] | Final key [95% CI] | The wrong ones should have been (final) |
|---|---|---|---|---|
| `out_of_scope` | 100 | 94.0% [87.5, 97.2] | **93.0% [86.3, 96.6]** | `sector` 4, `single` 2, `multi` 1 |
| `single` | 50 | 90.0% [78.6, 95.7] | 90.0% [78.6, 95.7] | `investing_general` 3, `market_economy` 1, `unrelated` 1 |
| `multi` | 50 | 78.0% [64.8, 87.2] | **82.0% [69.2, 90.2]** | `single` 4, `investing_general` 2, `market_economy` 2, `sector` 1 |
| `sector` | 25 | 84.0% [65.3, 93.6] | 88.0% [70.0, 95.8] | `market_economy` 2, `unrelated` 1 |
| **All quarantined (weighted)** | 225 | 90.6% [86.8, 94.5] | **90.8% [86.9, 94.7]** | |

On the 200 of the first draw alone (`--draw main`) the `multi` row is 64.0% (first) and 72.0% [52.4, 85.7] (final), and the overall figure 89.6% [85.5, 93.6] (first) and 90.1% [86.0, 94.1] (final).

**Audit of the tags (owner, 2026-11-11).** Reading the first-pass `no` answers against the spec found calls that contradicted its rules: a deal between two companies is `multi` (call 18; #161 CPI/Comtech, #141 Goldman/AmEx),
a list of picks with no stated theme is `multi` (call 21; #26, a "Fast Money" first-trade list), crypto is `market_economy`, not a sector (call 17; #65). The owner agreed and changed those tags (and #137, #12, with no reason recorded for #137);
#147 (the Sky bid, read as an industry-wide push to merge) stays `sector`. First audit: `multi` 16 -> 20 right, `sector` 21 -> 22. Second audit: #66 ("Your First Move For Friday", the same format as #26) is a list of picks with no theme, so
`multi`, not `out_of_scope`; #10 ("6 cheapest stocks") and #45 ("most price target hikes") are lists under a stated theme, so `investing_general` (`multi` 20 -> 18 right on the first 25). Every changed row has a note in the file with its reason.
**The 25 extra rows** were checked against the same calls: the deals (JVs, takeovers, a sale to two buyers) are `multi`, the movers, analyst-call and Fast Money lists are `multi`, the two `no` are market wraps (`market_economy`); no row breaks calls 17, 18 or 21.
Two are borderline and were left as tagged (the Pfizer trial, rivals moving; an Eli Lilly deal story). **Still borderline, left as tagged in the main draw:** #39, #85, #199 (Cramer lightning round / two picks) and #76, #154.
The training gold has the same split on deals ("Nvidia to buy Mellanox" `single`, "Dynegy acquired by Blackstone" `multi`), so part of the `multi`/`single` confusion is label noise, not model error. Auditing the gold under calls 17, 18 and 21
and settling the borderline rows is T-178; the figures here are not to be compared with the 83.6% before it is done.

**What it says about the confidence in the model.**
1. **The main worry, `out_of_scope`, is smaller than feared.** The rule written before the reading was: 95 or more of 100 `yes` and the worry is gone, 85 or fewer and all 200 are read. The result is 93, between the two, with the lower bound (86.3%) above the second line. About 7% of the bodies the step stops
   (roughly 10,000 of 146,000; 5,000 to 20,000 within the interval) are company or sector news that will not reach steps 2 and later. Four of the seven are `sector` stories, not company ones.
2. **`single` at 90% is usable.** The errors are bodies about investing in general or not news; later stages would run on text with no company in it.
3. **`multi` is the weakest class**, 82% on 50 bodies (72% on the first 25). Four of the nine errors are `single`, two are theme lists that call 21 sends to `investing_general`, two are market wraps: the cut would run the `units` extractor on a one-company body, which costs time but does not lose a company or its news.
4. **`sector` at 88%** is on the same line as the earlier reading of 42 bodies (section 8). Its sample (25) is the smallest.
5. **Overall**, about 9 in 10 quarantined bodies have the right scope, against 83.6% on the held-out 200 (a different population, mix and key, so not a like-for-like comparison).

**The gate of section 8.1: released (owner, 2026-11-11).** Section 8.1 held the cut's rows back from T-164/T-165 until the owner validated the quarantined scope sample. This section is that sample, and **the owner released the gate without a flag on `multi`**,
trusting the results above. The cost of a wrong `multi` is GPU time and sub-news of a one-company body, not lost news, and T-165 picks the main company of each sub-news with its own model. T-165 reads `sub_news` from now on (its other conditions stand).

**What it does not say.**
- It is **precision** (of what the step called X, how much is X). The sample was drawn by predicted class, so it cannot give the **recall** of `out_of_scope` among quarantined bodies (the held-out recall of 53% is a different measure on a different population). Company stories the step called `single` and that
  are really `out_of_scope` count against `single`; the share of true company news hidden in the other classes needs a sample drawn by tag (T-179).
- One reader plus an audit (above); the owner's calls on the line cases (a stock-picking show, a market wrap with a takeover) decide part of the `no` answers.
- 25 to 50 bodies per small class: the intervals are 20 to 33 points wide.
- The `ok` bodies and the 204 paywall pages of section 8 are outside this sample.

**Suggested next steps** (the owner decides): filter the paywall pages before step 2; keep the 10,274 low-margin bodies as the review queue. The Hub card of the scope model named this file under its old name and was republished on 2026-11-11
(card-only commit `58e4af4b`, weights unchanged, `--check` passes; the pin moved to it).

## 10. Index of the work

| What | Where |
|---|---|
| Task and short figures | `.specify/memory/TASKS.md`, T-147 (long forms in `archive/PIPELINE-superseded-2026-11-10.md`) |
| Pipeline definition | `.specify/memory/PLAN.md`, D40, D39 |
| Experiment scripts | `scripts/scope_title_baseline_2026_11_10.py`, `scope_relevance_features_`, `scope_two_stage_`, `scope_nli_topic_`, `scope_nli_single_multi_`, `scope_end_to_end_` (all `_2026_11_10.py`) |
| Draws and gold builders | `scripts/draw_scope_enrich_2026_11_10.py`, `draw_scope_hook_2026_11_11.py`, `build_scope_enrich_gold_2026_11_10.py`, `build_gold_2026_11_05.py` |
| Tagging rules | `docs/labelling-spec-v2.md` (owner calls 14 to 21) |
| Gold sets | `docs/gold_datasets.md` (sets 1, 2, 7, 8; set 9, `gold/scope_quarantined_validation.csv`, is section 9) |
| Validation of step 1 (section 9) | `scripts/draw_scope_quarantined_check_2026_11_11.py`, `scripts/score_scope_validation_2026_11_11.py`, `gold/scope_quarantined_validation.csv` |
| Pull requests | #160 to #162 (pipeline, schema), #163 (matcher features), #164 (relevance and sentiment models), #165 (scope model), #166 (step 1), #167 (pin), #168 and #169 (the cut), #170 (validation on quarantined bodies, this rename) |
