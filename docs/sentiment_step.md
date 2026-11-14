# The sentiment step (`--step sentiment`, step 6, T-168)

The sentiment of each **main company** of each text: `positive`, `negative` or `neutral`, and the three probabilities. The model is the
one the owner selected on 2026-11-10 and confirmed on 2026-11-14 (`docs/sentiment_training.md`, "The decision"): the **pair model** that reads
`[CLS] company [SEP] text [SEP]`, **stacked** with the relevance score that step 3 already stored for the pair.

## Run it

```
uv run cli/news_nlp_steps.py --step sentiment --summary-revision summary-... --companies-revision companies-... --limit 500
uv run cli/news_nlp_steps.py --step sentiment --status
uv run cli/news_nlp_steps.py --step sentiment --status --summary-revision summary-... --companies-revision companies-...
```

`--step summary --status` and `--step companies --status` list the revisions that ran; both are named, never the latest. `--source-db` and
`--results-db` (or `SOURCE_DATABASE_URL` and `DATABASE_URL`) name the two stores, as for the other steps. The run shows a progress bar (`tqdm`, in items, not pairs; its time estimate is rough because items differ in chunks and in companies). `--status` prints the items by state (`done`, `skipped`, `failed`) and the rows by label; a skipped item carries its reason. A rerun adds nothing: an item
already done or skipped at the revision is not read again, a failed one is retried; a long run can be cut with `--limit` and continued. The
last form of `--status` prints the revision this code would write for those inputs (no weight is read; it needs the universe, so `--source-db`), to compare with the `revisions` it lists.
To run the step again at one revision, or after a re-cut of the sub-news, the SQL is in `--help` (`RESENTIMENT_SQL`, `RECUT_SQL`).

**Nothing is downloaded.** The pair model is the local, git-ignored folder `models/sentiment_pair_all/final` (419 MB), named in
`config/sentiment_step.json` with its identity (`directory_identity`, a hash of its files). Its copy on the private Hub,
`gamug/FinBERT-company-sentiment@9a7b1eb3`, is a record in the same file and is never read: **the link between the folder's identity and that Hub
revision is the publish record (`config/published_models.json`), not a check**, so `model_name` on a row says which copy was published, not that this run
verified it. The step **refuses a folder whose identity is not the
pinned one** and a missing folder; there is no `--model-dir`, `--fetch` or `--allow-unpinned`. To use another model, change the pin: that is another
revision of the step. The relevance model is not loaded: step 3's score is read from `news_company.relevance_score`.

## What it reads and writes

* **Companies the universe does not name:** a pair whose ticker is not in the universe (`companies_of`) is **not scored** (the model was trained on the
  universe's names; a ticker read as a name would store a row from an input it never saw); it is counted as `name_missing`. An item left with no
  named ticker is `skipped`, "company not in the universe", and is not retried at that revision. No SQL is needed after fixing the universe: the names of the scored companies are in the revision, so
  a universe that now names the ticker is a new revision and scores the item (`RESENTIMENT_SQL` is only to rerun at the **same** revision). **A run with an
  empty universe is refused** (no `--source-db`, or a store without `fetch_status = 'ok'` articles): it would skip every item.
* **Reads:** the texts of step 2 (`news_summary` at `--summary-revision`) of every item that has a **main** company at `--companies-revision`
  (`news_company.is_main = 1`), and the relevance score of each (text, main company) pair. An item on the review list of step 3 has no main company
  and gets no sentiment. **The run is refused** when a main item of the companies revision has no text at the summary revision: step 3 does not
  record which summary revision it read, so this gap is what shows that its relevance scores are of other texts. (A match of ids is not a proof that the
  texts are the same; name the summary revision step 3 was run on.)
* **Writes:** `news_sentiment`, one row per (item, main ticker, revision): `label` (the largest of the three), `score` (its probability),
  `positive`, `negative`, `neutral` (the stack's, they add to 1), `model_name` (the published copy), `sentiment_checked` (always 1: the pair model
  reads the company by construction). The state of the item is in `pipeline_step_state` (`step = 'sentiment'`). A blank text, or one with no sentence to
  read, is `skipped` with the reason, not scored.

## How a pair is scored

1. The text is cut in chunks of at most the room the company leaves in the model's 512-token window (`sentiment_pair.chunk_budget`; the company name is
   cut at 80 characters); each chunk is read as `[CLS] company [SEP] chunk [SEP]`. The company is the universe's name, as the model was trained on
   it (`Alphabet Inc. (Class A)` included). The three class probabilities are the **mean over the chunks**, equal weights: the reading of
   `scripts/draw_sentiment_gold_2026_11_05.py` `predict_probs`, the one the cross-validation scored. Summaries are at most about 2,000 characters, so
   most texts are one or two chunks.
2. The stack (`config/sentiment_stack.json`) is the multinomial logistic regression of the log of the three probabilities and the logit of the
   relevance score, fitted on the out-of-fold scores of the cross-validation (`scripts/final_models_2026_11_10.py stack`). A test pins the step's
   arithmetic to that script's `apply_stack`. **Inside this step the relevance is always at or over the threshold**: only main pairs
   (`is_main = 1`) are read, so the stack here only adjusts the pair model within the high end of the feature. The pull of a low relevance toward
   `neutral` (the same positive sentence gives `positive` at 0.95 and `neutral` at 0.05) is what the stack learned on the marked rows, which include
   pairs of other companies; this step never feeds it those. A reader must not take the pull as a filter of other-company texts: step 3 is that filter.

## The revision

`sentiment-` + a hash of the pair model's identity, the stack file, the two input revisions, the (ticker, name) of the **main companies at the
companies revision** (the names the model reads; an edit of a company nobody is scored for is not a new revision, a renamed scored one is, and it
scores the whole run again), the window, and the code that scores (`stacked`, `score_of`,
`_pair_probabilities` and the cut of the text they call: `chunking.chunk_text`, `sentiment_pair.chunk_budget`, `sentiment_pair.encode_pair`;
normalised: a comment, a docstring or the formatting is not a new revision). The helpers those call (the sentence splitter and the like) are not
hashed: a change there needs a new revision made by hand (change the pin or the stack file's note). The storage, the logging and this document are not
part of it, so a run stopped and resumed after such an edit resumes. A step 2 or step 3 run again at a new revision is new work.

## What the figures are, and what is not done

* **0.814** is the score of the stack on the nested cross-validation (weighted, D37) over the 809 marked rows: **not held out**, and of **the whole
  marked population, other-company pairs included**; the step scores main pairs only, so the figure is not a measure of that population. T-148c4 and T-193
  must score it on main pairs. The pair model alone
  gets 0.737 and the pinned v1 model 0.648 on the same slice. The 180-text gold has not been scored with the stack (T-148c4, T-193), so the
  pin rule of D35 has not been applied and `config/pipeline_models.json` still names the v1 model. This step runs the owner's selected candidate; it
  does not decide the pin.
* The strict nesting of the relevance model was not done (accepted by the owner as a limit, `docs/sentiment_training.md`, "Published").
* The final models are more confident than the fold models the stack was fitted on; the effect is not measured. The check against the 0.862 mean
  confidence is the first thing T-148c4 does.
* The test that the local folder has the pinned identity is skipped where the folder is absent (a clean checkout, CI): a green CI is not a check of the weights.
* The step was checked on the real weights with two sentences and the stack, not on a store (no run was made on `nlp.db`): the identity of the local
  folder is the pinned one, the three probabilities add to 1, and a positive sentence about another company reads `neutral` at low relevance (a property of the stack: the step itself never feeds it a low relevance).
* **Not decided here** (the owner's): what a low `score` means; whether sector, market and economy texts (no company) get a sentiment through the
  text-only model (D39 (ii)); T-171, the update of `CLAUDE.md` and the module docs to the pair model.
