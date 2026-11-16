# The NER model on the owner's hold-out (2026-11-16)

This is the record of how the pinned NER model (`gamug/sec-bert-finer-ord-ner@ba7b9e43`, `config/pipeline_models.json` `ner`) scores on texts the owner tagged by hand. The model is
not trained here: it is the one step 3 (`docs/companies_step.md`) and the v1 NER stage already run. **Nothing was changed or pinned by this measure.** The script is
`scripts/ner_eval_2026_11_16.py` (its tests: `tests/news_nlp/test_ner_eval.py`); the figures are in `docs/ner_results_2026_11_16/` (`ner_metrics.json`, and `ner_errors.csv`: organisation names and
spans only, no news text; `T:<ticker>` is a universe company, `N:<name>` a normalised name that is in no universe).

**Read this first.**

* **It is a hold-out of the model, but a small one: 100 texts, 321 labelled organisations.** The model was fine-tuned on FiNER-ORD (`docs/modules/news-nlp.md`), not on these texts.
  The draw is not recorded with a protocol or weights here, so the set is not shown to be a random sample of the corpus; its texts are summaries or originals of step 2 at `summary-7a9bd2062bc4` (`key.csv`).
  There is no interval in this page because the counts are too small to carry one honestly; read every figure as "about".
* **Two views, and they answer different questions.** *All organisations*: the model tags every organisation, so it also tags names the owner did not write (media, institutions, products). *Universe companies only*:
  the names that are one of the 507 companies of `gold/universe.csv`, which is what step 3 keeps (a span with no ticker is ignored, D39 (iii)). **The second is the one the pipeline uses.**
* **The owner's labels are not fully clean** (section 2): **14 labels in 11 rows** are not written in the text as whole words. Every figure is given with those labels **as they are**, **removed**, and **repaired**
  (an overlay proposed by the developer for the owner to confirm, the file was not edited: **9 replaced, 1 kept, 4 dropped**, section 2). Removing them is the worse treatment: a malformed label that names a real company leaves the model's correct find as an "extra".

## 1. The result

**Universe companies only (what step 3 keeps)**

| The 14 labels not in the text | precision | recall | F1 | tp / fp / fn |
|---|---|---|---|---|
| as they are, or removed (the same: none of them is a universe company as written) | 90.1% | 95.4% | 0.927 | 146 / 16 / 7 |
| **repaired** (9 replaced, 1 kept, 4 dropped) | **92.6%** | **95.5%** | **0.940** | 150 / 12 / 7 |
| repaired, with the matcher over the text as well (`ner+text`, the candidates step 3 runs) | 90.1% | **98.7%** | 0.942 | 155 / 17 / 2 |

**All organisations**

| The 14 labels not in the text | precision | recall | F1 | tp / fp / fn |
|---|---|---|---|---|
| as they are | 70.3% | 87.0% | 0.777 | 274 / 116 / 41 |
| removed | 70.3% | 91.0% | 0.793 | 274 / 116 / 27 |
| repaired | 72.6% | 90.4% | 0.805 | 283 / 107 / 30 |
| repaired, media names (CNBC, Reuters, WSJ, Barron's, Bloomberg, Mad Money) left out of both sides | 72.7% | 90.3% | 0.805 | 269 / 101 / 29 |
| as they are, **no alias list** (section 3) | 68.0% | 86.4% | 0.761 | 274 / 129 / 43 |

* **For step 3 the NER plus the matcher finds 95.5% of the universe companies the owner named, and 92.6% of the ones it finds are in the owner's tags.** With the matcher over the text (`ner+text`) recall is 98.7%: 5 of the 7 misses are found.
* **The low precision on all organisations is not a defect of step 3**: the extra spans are names that map to no ticker (CNBC, the Federal Reserve, the EU, Supreme Court, products such as `iPhone`) and are dropped by the matcher. Leaving the media names out moves nothing.
* A match is **exact after standardising**, so a name that is part of the other (`Solaris Energy` against `Solaris Energy Infrastructure`) counts as a miss and as an extra: **6 such pairs** (repaired, all organisations; `near_miss_pairs_repaired_all`). A looser match would raise the figures a little.

## 2. The set and its integrity check

`data/ner_val100/to_tag.csv` (git-ignored: it holds news text). The owner writes the organisations of each text in the `organizations` column, **comma-separated, the names as they appear in the text** (mostly), not standardised.
The script runs these checks first and **stops** on a structural failure; the labels not in the text are reported, not fatal (`ner_metrics.json`, `integrity`):

| Check | Result |
|---|---|
| `review_order` is 1 to 100, `key` unique and equal to `key.csv`, no empty `text` or `organizations`, the columns exist | passes |
| Every row splits on commas with no empty item | passes (321 organisations, 3.2 a row; one name repeated inside one row) |
| Every label is written in the `text` column **as a whole word** (the only column the model reads: a label written from the title alone is reported; a label that names a universe company the text writes under another name, `United Airlines` for `UAL`, passes: `direct_mentions`, no model) | **fails for 14 labels in 11 rows** |

The 14 labels, and the repair proposed for each (`REPAIRS` in the script; every repaired name is checked to be in the text, and a new failing label with no repair stops the script). **The overlay replaces 9 of them, keeps 1 as it is (`NHTSA`: the label is right and the text is defective) and drops 4 (row 26): 9 replaced, 1 kept, 4 dropped** (`repairs` in `ner_metrics.json`):

| Kind | Row: label -> repair |
|---|---|
| Spanish name | 17: `Casa Blanca` -> White House; 20: `Unión Europea` -> European Union; 27: `Reserva Federal` -> Fed (the text writes "Fed") |
| A truncated name (typo) | 14: `Alphabe` -> Alphabet; 96: `Xiaom` -> Xiaomi |
| Names run together or mistyped | 18: `Japón Aflac` -> Aflac; 33: `Douyin de ByteDance` -> ByteDance, Douyin; 48: `Dow UnitedHealth` -> UnitedHealth (a comma missing); 92: `Alipay.Starbucks` -> Alipay, Starbucks |
| The label is right, **the text is defective** | 37: `NHTSA` (the text writes `NHTSAsaid`); kept as it is |
| In neither the English nor the Spanish text (they look like the labels of another article) | 26: `Dish Network`, `Verizon Communications`, `Vodafone`, `Allergan`: no repair, dropped |

The owner decides whether to correct the file; the figures are run again with one command.

## 3. How the names were standardised

Both sides, the owner's labels and the model's spans, are turned into one canonical company before they are compared, per text, as sets (a company named twice counts once):

1. A name is normalised (`news_nlp.triage.normalize_name`: lower case, accents, legal words and share class dropped).
2. A short alias list maps the same company under another name to one ticker: `Google` and `Alphabet` to `GOOGL`, `Facebook` to `META`, `J & J` to `JNJ`, `United` and `Continental Airlines` to `UAL`,
   `Amazon Web Services` and `AWS` to `AMZN`, `JPMorgan` and `JP Morgan Chase` to `JPM`, `Citi` to `C`, `Bank of America Merrill Lynch` to `BAC`. Share classes of one company are one (`GOOG`, `GOOGL`).
3. Otherwise the universe matcher (`CompanyResolver.ticker_of_name`, the one step 3 uses) gives the ticker; a ticker written in capitals (`MSFT`, `JNJ`) resolves the same way.
4. A name with no ticker stays a normalised string (`cnbc`, `saudi aramco`); `EU` and `European Union`, `Fed` and `Federal Reserve` are one.

**The alias list was chosen after the first run showed these names as errors, so it is fitted to this set**, not independent of it. Without it (the "no alias list" row) precision is 68.0% and recall 86.4%: about two and a half points
of precision come from names that are one company written two ways. Another list gives other figures.

## 4. What the errors are (`ner_errors.csv`, labels repaired)

* **Missed universe companies (7).** By the script's own check (`miss_causes`: does any organisation span overlap the name in the text?): **5 are not tagged by the NER at all**, `Meta Platforms` (row 5), `Hormel` (45), `MSFT` (46), `Oracle` (57), `Las Vegas Sands` (60);
  **2 are tagged but the matcher does not map the span**, `ON Semiconductor` (24, `ON` is a word ticker the matcher does not count) and `Amazon` (85). The matcher over the text (`ner+text`) recovers the 5 that the NER did not tag.
* **Extra universe tickers (12):** the span is in each case that company's own name, which the owner did not label: `Royal Caribbean` (15), `Paramount` (16), `PayPal` (26), `Hewlett` (44) and `Hewlett Packard Enterprise` (98; the first is
  a short name that could be HP Inc. as well), `Caterpillar` (51), `Alphabet` and `Google` (57), `ServiceNow` (58), `Lennar` (59), `NextEra Energy` (93), `Intel` (94), `Trane` (100). **I did not read each text to see whether it is about that company**; the labels are the organisations the owner wrote, not a complete list.
* **No sign of a badly cut span.** Spans such as `nhtsasaid`, `usrecall` and `morgan stanleydown` (rows 37, 41) are in the extras and misses, but the source text writes the words stuck together (`NHTSAsaid`, `recallingabout`, `Stanleydown`): the model tags what the text gives.
  A text defect of this kind is the crawl's, not the model's; how often it occurs in the corpus is not measured here.
* **Names the owner did not label but the model tags (not errors for step 3):** media (`CNBC` 3 times, `Reuters`, `Barron's`), institutions (`Federal Reserve`, `EU`, `Congress`, `CIA`, `Supreme Court`), products and brands (`Android`, `iPhone`, `HBO Max`, `Diner Dash`).
* **Names the owner labelled that no ticker covers and the model missed:** `Saudi Aramco`, `TikTok`, `WhatsApp`, `Publicis`, `Emirates`, `Whole Foods`, `HBO`. They are outside the universe, so step 3 would not have kept them anyway.

## 5. What this does and does not show

* **It shows** that, on this set, the NER and the matcher together find 95.5% of the universe companies the owner named, and 98.7% with `ner+text`, which is what step 3 runs. This set has no main-company labels, so it
  says nothing about where step 3 loses a company once the candidates are found: that is measured on other texts (`docs/main_company_training.md`, `docs/companies_step.md`).
* **It does not show** the error on the owner's real target, "the companies a text is about". The owner's labels are every organisation named, not the main ones; the main-company choice is measured elsewhere (119 marks on sub-news).
* **100 texts is a spot check.** Do not compare it with a figure on a different set (FiNER-ORD, whose organisations are tagged by another rule) or read a one-point difference as a change in the model. The repairs are the developer's.
* **The text is the one the model is given, not the summary of the new revision.** The set's texts are `summary-7a9bd2062bc4` summaries or originals (`key.csv`); the corpus run now uses `summary-llm-70cb6bb07292` (DeepSeek, whose prompt asks to name the
  actors, `config/summary_prompt.md`), which was not scored here.

## 6. Reproduce, and what is open

`uv run scripts/ner_eval_2026_11_16.py` (the pinned model runs on the GPU in a minute; it needs `data/ner_val100/to_tag.csv` and `key.csv`, which are **not in git**, and `gold/universe.csv`). **The tags are not migrated to `gold/`** (`docs/gold_datasets.md`, set 18):
the set holds news text and the repository is public, so copying the texts to `gold/texts/` is the owner's decision (T-213). Open for the owner (T-213): confirm or correct the 14 labels, the migration, and whether the real run's DeepSeek summaries should be scored the same way.
