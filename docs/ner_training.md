# The NER model on the owner's hold-out (2026-11-16)

This is the record of how the pinned NER model (`gamug/sec-bert-finer-ord-ner@ba7b9e43`, `config/pipeline_models.json` `ner`) scores on texts the owner tagged by hand. The model is
not trained here: it is the one step 3 (`docs/companies_step.md`) and the v1 NER stage already run. **Nothing was changed or pinned by this measure.** The script is
`scripts/ner_eval_2026_11_16.py`; the figures are in `docs/ner_results_2026_11_16/` (`ner_metrics.json`, and `ner_errors.csv`: organisation names only, no news text).

**Read this first.**

* **It is a hold-out of the model, but a small one: 100 texts, 321 labelled organisations.** The model was fine-tuned on FiNER-ORD (`docs/modules/news-nlp.md`), not on these texts.
  The draw is not recorded with a protocol or weights here, so the set is not shown to be a random sample of the corpus; its texts are summaries or originals of step 2 at `summary-7a9bd2062bc4` (`key.csv`).
  There is no interval in this page because the counts are too small to carry one honestly; read every figure as "about".
* **Two views, and they answer different questions.** *All organisations*: the model tags every organisation, so it also tags names the owner did not write (media, institutions, products). *Universe companies only*:
  the names that are one of the 507 companies of `gold/universe.csv`, which is what step 3 keeps (a span with no ticker is ignored, D39 (iii)). **The second is the one the pipeline uses.**
* **The owner's labels are not fully clean** (section 2): 11 labels in 8 rows are not in the English text. Every figure is given with and without them; the file was not edited.

## 1. The result

| View | precision | recall | F1 | tp / fp / fn |
|---|---|---|---|---|
| All organisations, the owner's labels as they are | 70.3% | 87.0% | 0.777 | 274 / 116 / 41 |
| All organisations, the 11 labels that are not in the text removed | 70.3% | 90.1% | 0.790 | 274 / 116 / 30 |
| The same, media names (CNBC, Reuters, WSJ, Barron's, Bloomberg, Mad Money) left out of both sides | 70.3% | 90.0% | 0.789 | 260 / 110 / 29 |
| **Universe companies only** (the labels removed) | **90.1%** | **95.4%** | **0.927** | 146 / 16 / 7 |

* **For step 3 the NER finds almost every universe company a text names (95.4%) and rarely invents one (precision 90.1%).** The 7 misses and 16 extra tickers are listed in section 4.
* **The low precision on all organisations is not a defect of step 3**: the extra spans are names that map to no ticker (CNBC, the Federal Reserve, the EU, Supreme Court, products such as `iPhone`) and are dropped by the matcher.
  Media names are not the cause: leaving them out moves nothing (70.3% either way).
* A match here is **exact after standardising**, so a name that is part of the other (`Solaris Energy` against `Solaris Energy Infrastructure`) counts as a miss and as an extra; these near-misses are few (at least six pairs in `ner_errors.csv`),
  and a looser match would raise the figures a little. It was not measured.

## 2. The set and its integrity check

`data/ner_val100/to_tag.csv` (git-ignored: it holds news text; the repository's rule is to migrate the tags to `gold/` when a set is finished, which is **not done**: `docs/gold_datasets.md`, set 18).
The owner writes the organisations of each text in the `organizations` column, **comma-separated, the names as they appear in the text** (mostly), not standardised.

| Check | Result |
|---|---|
| 100 rows, `review_order` 1 to 100, `key` unique and equal to `key.csv`, no duplicate article | passes |
| No empty `text` or `organizations`; every row splits on commas with no empty item | passes (321 organisations, 3.2 a row; one name repeated inside one row) |
| Every label is in the English title or text | **fails for 12 labels in 9 rows** (below) |

The 12 labels: the Spanish names `Casa Blanca` (row 17), `Unión Europea` (20) and `Reserva Federal` (27); names run together or mistyped, `Japón Aflac` (18), `Dow UnitedHealth` (48, a comma missing),
`Douyin de ByteDance` (33), `Alipay.Starbucks` (92); `United Airlines` (77, the text says `United`: a variant of the name, **kept** because standardising resolves it); and in row 26 `Dish Network`, `Verizon Communications`,
`Vodafone` and `Allergan`, which are in neither the English nor the Spanish text (they look like the labels of another article). **The 11 that are not variants are the ones the "removed" rows drop.**
The owner decides whether to correct them in the file; the figures should be run again then (the script does it in one command).

## 3. How the names were standardised

Both sides, the owner's labels and the model's spans, are turned into one canonical company before they are compared, per text, as sets (a company named twice counts once):

1. A name is normalised (`news_nlp.triage.normalize_name`: lower case, accents, legal words and share class dropped).
2. A short alias list maps the same company under another name to one ticker: `Google` and `Alphabet` to `GOOGL`, `Facebook` to `META`, `J & J` to `JNJ`, `United` and `Continental Airlines` to `UAL`,
   `Amazon Web Services` and `AWS` to `AMZN`, `JPMorgan` and `JP Morgan Chase` to `JPM`, `Citi` to `C`, `Bank of America Merrill Lynch` to `BAC`. Share classes of one company are one (`GOOG`, `GOOGL`).
3. Otherwise the universe matcher (`CompanyResolver.ticker_of_name`, the one step 3 uses) gives the ticker; a ticker written in capitals (`MSFT`, `JNJ`) resolves the same way.
4. A name with no ticker stays a normalised string (`cnbc`, `saudi aramco`); `EU` and `European Union`, `Fed` and `Federal Reserve` are one.

**Without the alias list the first run scored 68.0% precision and 86.4% recall**: about two points came from names that are one company written two ways, not from the model. The list is in the script and is a choice: another list gives other figures.

## 4. What the errors are (`ner_errors.csv`, labels removed)

* **Missed universe companies (7):** `Meta` (row 5), `ON` (24, a word ticker the matcher does not count), `Hormel` (45), `Microsoft` (46), `Oracle` (57), `Las Vegas Sands` (60), `Amazon` (85). Short names and one word that is also an everyday word.
* **Extra universe tickers (16):** `Google` (14, 57), `Royal Caribbean` (15), `Paramount Skydance` (16), `Aflac` (18), `PayPal` (26), `HPE` (44, 98), `UnitedHealth` (48), `Caterpillar` (51), `ServiceNow` (58), `Lennar` (59),
  `Starbucks` (92), `NextEra` (93), `Intel` (94), `Trane` (100). Several look like real mentions the owner did not label (the model tags every organisation named); some look like false matches (`NOW`, `TT`, `NEE`, which may come from `NEP`). I did not read each text for this.
* **Spans cut wrong, which the matcher then cannot use:** `alphabe` (14), `xiaom` (96), `nhtsasaid` and `usrecall` (37), `morgan stanleydown` (41), `national highway trafficsafety administration` (37). The model joins a word and the next, or stops inside a word.
  This is the same family as the subword-fragmentation bug of 2026-09-12 (`docs/modules/news-nlp.md`); it is rare (about 6 spans in 100 texts) and costs a few misses, but it is **not fixed** and is the thing to look at first if the recall must rise.
* **Names the owner did not label but the model tags (not errors for step 3):** media (`CNBC` 3 times, `Reuters`, `Barron's`), institutions (`Federal Reserve`, `EU`, `Congress`, `CIA`, `Supreme Court`), products and brands (`Android`, `iPhone`, `HBO Max`, `Diner Dash`).
* **Names the owner labelled that no ticker covers and the model missed:** `Saudi Aramco`, `TikTok`, `WhatsApp`, `Publicis`, `Emirates`, `Whole Foods`, `HBO`, `NHTSA`. They are outside the universe, so step 3 would not have kept them anyway.

## 5. What this does and does not show

* **It shows** that, on this set, the NER is not what limits step 3: 95.4% of the universe companies the owner named are found. Where step 3 loses a company the cause is more likely the matcher (names such as `Meta`, `Microsoft`, `Oracle` missed above are
  matcher-side gaps, T-181) or the relevance threshold (`docs/companies_step.md`, `docs/main_company_training.md`).
* **It does not show** the NER's error on the owner's real target, which is "the companies a text is about". The owner's labels are every organisation named, not the main ones, and that is the NER's task; the main-company choice is measured elsewhere (119 marks on sub-news).
* **100 texts is a spot check.** Do not compare it with a figure on a different set (FiNER-ORD, whose organisations are tagged by another rule) or read a one-point difference as a change in the model.
* **The text is the one the model is given, not the summary of the new revision.** The set's texts are `summary-7a9bd2062bc4` summaries or originals (`key.csv`); the corpus run now uses `summary-llm-70cb6bb07292` (DeepSeek), whose shorter, name-first texts were not scored here.
