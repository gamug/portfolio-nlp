# Step 2: the summary (T-164, T-166, T-184, T-190; PLAN D39 (2), (3.3), D40)

`uv run cli/news_nlp_steps.py --step summary (--cut-revision REV | --items single | --items sector --sector-revision REV) [--items sub_news] [--limit N] [--scope-revision REV]` (and `--step summary --status`)

## What it does

It gives every `single` body of step 1 and every sub-news of the cut **one text for the steps after it to read**:

- a text **over 2,000 characters** is summarised by the pinned local summariser (`config/pipeline_models.json` `c_summary`:
  `sshleifer/distilbart-cnn-12-6` at revision `a4f8f3ea…`), with the chunk-and-reduce of `summary_stage.hierarchical_summarize_batch`
  (the v1 stage's own function, reused as it is: 1,000-token chunks on sentence boundaries, summaries of 56 to 142 tokens, reduced until one is left);
  the row says `text_source = summary` and the number of chunks;
- a text **at or under 2,000 characters** is stored unchanged, `text_source = original`, no chunks.

The size is the owner's (2026-11-12). It is measured on the text the summariser would read: for a single body that is
`Title-<title>\nBody-<body>` (the input of the earlier summary run), for a sub-news its own text (the title belongs to the whole round-up).
**Sector bodies** (`--items sector --sector-revision REV`, T-167): the bodies the sector step assigned a sector (`body_sector.status = 'assigned'`) at the
revision named, with the same size rule and the same input as a single body (`item_kind = 'body'`, the same summary revision, so nothing already summarised is redone;
a body summarised as a single one under another scope revision keeps its summary). The sector revision is **named, never the latest** (the ones the sector step has are
listed when it is missing or wrong) and the sector step must have run first: a `sector` body it has not reached, and one it sent `out_of_scope`, are not read. It is not
part of the default (`single` and `sub_news`); `--cut-revision` and `--scope-revision` do not apply to it.

Rows go to `news_summary` (`item_kind` `body` with the body hash, or `sub_news` with the `sub_news.id`; no schema change), state to
`pipeline_step_state` (step `summary`). The step is idempotent and resumable like the scope and the cut: a rerun at the same revision adds
nothing, a failed item is retried and keeps its reason (`Type: message`), a batch that fails is redone item by item, an empty summary is a
failure (never stored), `--limit N` touches at most N pending items (single bodies first), and three batches in a row that fail completely stop the run.
Options the step does not read are refused, never ignored: `--model-dir`, `--reading`, `--titles`, `--fetch` and `--allow-unpinned` (the scope's and the cut's), `--cut-revision` with `--items single`, and `--scope-revision` with `--items sub_news`
(it names the single bodies to read; the JSON a run prints lists only the revisions it read).

**The revision** is `summary-` plus 12 hex digits of the SHA-256 of the model and its pinned revision, **the `transformers` version** (beam search lives there and `pyproject.toml` only requires `>=4.56`; `uv.lock` pins 5.16.1: a lock bump starts a new revision instead of mixing old and new summaries under one; `torch` is left out because its version carries the build, `+cu124` or `+cpu`, and the revision is the same on both devices), the size, the input format, the generation limits and
the code that summarises: `news_summary_stage.py` whole, and the functions this step runs in `summary_stage.py` (`hierarchical_summarize_batch` and
its helpers, `MAX_REDUCE_PASSES`) and `chunking.py` (`chunk_text`, `split_sentences`, `Chunk`, the sentence regex): `news_summary_stage.hashed_sources()` lists them. **The code is hashed as parsed
(`normalized_code`): comments, formatting, the kind of quotes and docstrings do not change the revision, so a typo fix or a formatter upgrade does not make the step redo hours of GPU work; a change of
a name, a constant or an operator does.** A new size, model or such a change starts new rows and leaves the old ones. **An edit to the rest of those two files does not either**: `summary_stage.py` also holds the v1 stage and `chunking.py` the NER span merge, and
shared code that this step does not run must not make it redo about 15 GPU hours (a test edits both and checks the revision stays). On CPU or GPU alike (CUDA when there is one).

**A cut is named, never "the latest".** The sub-news are read from the cut revision given with `--cut-revision`; without it the command stops and lists the
cuts that ran (`--items single` needs none). The same rule as for every reader of `sub_news` (`docs/scope_training.md` §8.1). **The steps after this one read
`news_summary` at the revision they name, never "the latest".**

## DeepSeek is the summariser (T-206, owner 2026-11-15)

`--step summary` summarises with **DeepSeek by default**: one OpenAI-compatible `chat/completions` call for each text over 2,000 characters (`src/news_summary_llm.py`; `LLM_URL`, `LLM_API_KEY`, `LLM_MODEL` of `.env`; prompt `config/summary_prompt.md`, temperature 0, input `Title-...\nBody-...` as before). `--summariser local` runs the pinned distilbart described above (its revision, `summary-<12 hex>`, is untouched). The v1 chain keeps distilbart. **The article text leaves the machine** for the API (accepted by the owner).

- **Revision:** `summary-llm-` plus 12 hex digits of the hash of the prompt, the model name (`LLM_MODEL`), the temperature, the size rule (2,000 and the 100,000-character limit) and the input format. A new prompt, model or size starts new rows; the rows of another revision stay.
- **Each row keeps who answered:** `news_summary_call` (`api_model` as the API names it in its reply, `prompt_tokens`, `completion_tokens`, `called_at`), because the same name can be served by another model later. `--status` shows the summaries, tokens in and out and the model ids (`api`). Estimated cost for the three populations (79,283 items): about $16 to $32 (`docs/summarization_training.md`, "T-206").
- **A text over 100,000 characters** is not sent: it is stored as it is (`text_source = original`). None of the 79,283 items is over it (longest 84,177).
- **Failure: fail fast and resumable.** A rate limit, a server error or a dropped connection is retried four times with a growing wait; a text the API refuses or answers with an empty or cut reply fails alone with its reason and is retried by the next run; a wrong key, no balance (402), an invalid request (422) or the cap of calls stops the run at once (`stopped: ...`), the rows written stay and a rerun goes on. There is no fallback to distilbart and an empty summary is never stored. `--max-calls N` is a hard cap on the requests (retries included), `--concurrency N` the requests at once (8). A batch with one failing text is redone item by item, so a few calls are repeated.
- **Not repeatable:** at temperature 0 two runs give different words (30 articles twice: 0 identical, same scores), so a revision promises its rows, not its text.
- **Evaluation after the run** (owner): `scripts/summary_judge_2026_11_13.py` with a judge that is **not** the summariser: `judge` refuses `LLM_MODEL` equal to the summariser it recorded (`--allow-self-judge` overrides), so put another model in `LLM_MODEL` for the judge, or use the owner's marks first. Constitution AI #1/#2/#5 have a scoped exception for this step (constitution, "AI behavior").

## Running it, and running it again

Run it in slices (`--limit 2000`, repeated): a run holds the text of every pending item in memory, about 280 MB for all the 93,034 single bodies, and writes
one transaction per item. `--status` prints the counts by state, by source (`summary`, `original`) and by kind, and the revisions.

Summarise again at one revision of the summary step (`<summary revision>`: what `--status` prints; the state is keyed by revision and `news_summary` is written with `INSERT OR IGNORE`, so deleting one table is not enough):

```sql
DELETE FROM pipeline_step_state WHERE step = 'summary' AND model_revision = '<summary revision>';
DELETE FROM news_summary WHERE model_revision = '<summary revision>';
```

**Before cutting again a revision of the cut** (`<cut revision>`: its `extractor_revision`) (`docs/scope_training.md` §8.1), delete its summaries: they are keyed by `sub_news.id`, and new ids would leave them
pointing at nothing:

```sql
DELETE FROM news_summary WHERE item_kind = 'sub_news' AND item_id IN (SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');
DELETE FROM pipeline_step_state WHERE step = 'summary' AND item_kind = 'sub_news' AND item_id IN (SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');
```

The same statements are in `--help` (one source in `cli/news_nlp_steps.py`, tested against a database).

## What a run prints, and what the bars count

A run first says what it is about to do, on stderr, and then works in two phases with a bar each:

```
pending 254,608 items (93,034 single bodies, 161,574 sub-news): 64,480 to summarise with the model (over 2,000 characters), 190,128 stored as they are
original: 100%|...| 190128/190128     <- the items under the size: a write each, no model
summary:   12%|...| 7738/64480        <- the items over the size: the model; its rate and time left are the real ones
```

The numbers above are computed for the owner's corpus, not printed by a full run: the single bodies of scope revision `scope-nli-8ff21e50796b` (93,034, of which 59,432 are over the size) and the sub-news of the reference cut `units5-d3e2cbfa830e` (161,574, of which 5,048 are over it: 3.12%, measured 2026-11-13 on a consistent copy of the owner's database with `--cut-revision units5-d3e2cbfa830e`, below). 59,432 + 5,048 = 64,480 items go to the model and 93,034 + 161,574 - 64,480 = 190,128 are stored as they are. (The first version of this plan was computed on the older `units` cut, `units-5f831c298bcb`: 165,192 sub-news, 5,046 over the size, 64,478 to summarise; the reading barely changes the share, 3.05% against 3.12%, and the GPU estimate does not move.) The size table below says why the whole cut and not a sample: on the first 500 `multi` bodies by id the `units5` cut gives 50 over of 3,233 (1.5%), so the sample under-reads by half. **The 254,608 is not a count of articles**: it is single bodies plus sub-news (a body of another scope, `multi`, `sector` or
`out_of_scope`, is not in it), and about three quarters of it is stored as it is. The first version of the step put all of them on one bar, which mixed instant writes (286 a second measured) with seconds per model item, so
neither its rate nor its time left meant anything (reported by the owner, 2026-11-12). With `--limit N` the plan counts what this run will do, after the limit; with nothing to do it says `nothing pending`.

The `original` phase writes one transaction per item: 286 items a second on a local disk (a temporary database, 3,000 items), about 11 minutes for 190,128 items; over a mounted disk each commit can cost more, and a commit every
few hundred items would cut it to seconds (not done: the model phase is the cost).

## The measure behind 2,000 (2026-11-12)

`uv run scripts/summary_size_measure_2026_11_12.py --results-db <nlp.db copy> --source-db <urls.db copy> --cut-revision units5-d3e2cbfa830e --first-bodies 500` (read-only)
gave, over the 93,034 `single` bodies of the corpus run (input `Title-…\nBody-…`) and the 161,574 sub-news of the reference cut (**the whole cut; its first 500 bodies give 1.5% over 2,000, a sample that under-reads**):

| over (characters) | single bodies | sub-news (whole `units5` cut) | time for the singles at 0.84 s a body (an extrapolation) |
|---|---|---|---|
| 1,500 | 75,368 (81.0%) | 9,249 (5.7%) | 17.6 h |
| **2,000** | **59,432 (63.9%)** | **5,048 (3.1%)** | **13.9 h** |
| 3,000 | 31,992 (34.4%) | 1,731 (1.1%) | 7.5 h |
| 4,000 | 17,621 (18.9%) | 708 (0.4%) | 4.1 h |

`--first-bodies 500` prints the sub-news of the cut twice, whole and for its first 500 bodies, which is how the 1.5% and the 3.1% were compared.

Single bodies: median 2,436 characters, p90 5,216, p99 11,715. Sub-news, the whole `units5` cut (161,574): median 280, p90 1,077, p99 3,065 (the first-500 sample: median 262, p90 647, p99 2,303: its tail is shorter than the corpus's). The owner chose 2,000. The reason offered with it,
not checked against each downstream model: a 512-token transformer input is about 2,000 to 2,400 characters, so a text under 2,000 reaches NER, the
relevance model and sentiment whole, and a longer one is the kind that would be truncated. The 0.84 s a body is the earlier summary run's figure (all sizes of
bodies, on the GPU); it is **not** a measure of this model on this machine, and long texts take more chunks, which makes the figure more likely low than high (a reason, not a measure), so read the hours as an order of magnitude until the first GPU run. For the whole step: 59,432 + 5,048 = 64,480 items × 0.84 s = 54,163 s, about 15 h.

## First run, determinism and running next to the cut (2026-11-12, scratch copies of the databases, CPU)

12 single bodies, all over the size, at revision `summary-7a9bd2062bc4` (the revision of the code of this PR): 12 summaries, no failure, `--status` adds up, and the printed JSON lists only the
revisions the run read. A body took 13 to 32 s on the CPU depending on the load of the machine; the GPU time is **not measured yet**.

**Determinism.** `scripts/summary_determinism_check_2026_11_12.py -n 4` (the constitution asks for deterministic inference) on those rows, **like for like** (stored rows and code at the same revision): the
four stored summaries were summarised again in a new process, twice: **4 of 4 compared, 0 differ from the stored ones, 0 differ between the two runs** (CPU; the result line also prints the versions that produced them: torch 2.6.0+cu124, transformers 5.16.1). The check is **not a pass unless it compared what it was
asked to**: rows of another revision, a source without the bodies (`found 0 of 4`, run for real on an empty source) or with only some of them (`found 1 of 2`) end with **exit 2** and the reason; `--allow-other-revision` goes on
and says so. Exit codes: 0 clean, 1 a summary differs, 2 not comparable. `--kind sub_news` reads the sub-news path (tested). **Not checked: the GPU**, which is the run that matters (a GPU can flip a beam near a
tie): run it on the GPU after the cut and record the result here.

**Next to the cut.** Both steps write the same RESULTS file, and SQLite takes one writer at a time (`portfolio_common` waits 30 s for the lock). On the scratch copy the cut of 15 bodies (99 sub-news, 64 s) and the
summary of 6 items (134 s) ran **at the same time**: both exited 0, no lock error, no traceback. The safe way is still one writer at a time; if both run and a write waits more than 30 s, `database is locked` ends the
run with a traceback (it is not caught inside `record_summary`): the batch in hand is lost and every item already committed stays, so the next run continues from there.

## Limits

A text is chunked on sentence boundaries and each pass is cut to the model's input (`truncation=True`); when `MAX_REDUCE_PASSES` (6) is exhausted before the summaries
collapse to one, a last pass drops the tail without a record (`summary_stage.hierarchical_summarize_batch`, inherited from the v1 stage, very rare); `news_summary`
stores the number of chunks, not whether this happened. The summaries are model output, not facts (constitution, Pipeline models 4); their faithfulness is measured by a judge, not by the owner yet: see "Measuring the summaries" below.

## Measuring the summaries (LLM as a judge)

`scripts/summary_judge_2026_11_13.py` (`draw`, `judge`, `report`, then `tag-sheet` and `agree`; the judge is the `.env` model, `LLM_API_KEY` / `LLM_MODEL` / `LLM_URL`).

- **Frame:** only the items the model summarised (`text_source = 'summary'`); the ones stored as they are would be trivially faithful. Two groups, `single` (single bodies of a named scope revision) and `multi` (sub-news of a named cut revision). Each group is cut in three buckets by the number of chunks (terciles of the group; the chunk count is discrete, so a bucket can be empty) and the draw takes the same number from each (default 150 per group). The overall rate is weighted by each bucket's population over its sample, with a bootstrap interval resampled within buckets.
- **The judge** reads the source as the summariser did (a single body as `Title-…\nBody-…`, a sub-news alone), splits the summary into claims and marks each `supported`, `unsupported` or `contradicted` **by the source alone**, and says whether the main event of the source is in the summary. Rates per summary: no unsupported or contradicted claim, at least one contradicted claim, main event covered; per claim: the share unsupported and contradicted. Without a judge: whether the summary keeps the companies the source names (the matcher of step 3, so the main one is the one named most) and its compression.
- **The sample records what it was drawn from.** `draw` writes `sample.meta.json` next to `sample.csv`: the summary, scope and **cut revision** (the sub-news ids mean nothing without it), the seed, `per_group`, the population of each group, name, size and time of the two database copies, the date, and a `sample_id` (a hash of all that and of the ids drawn). `judge` copies the `sample_id` into every verdict and judges again a verdict of another sample; `report`, `tag-sheet` and `agree` refuse verdicts of another sample, and the report's first line prints the identity. At most 12 claims are listed per summary (the prompt says so): the report counts the summaries **at the cap**, which are only partly checked.
- **Provenance.** Every verdict carries the judge model, the hash of the prompt and the summary revision. A resumed run judges again the items whose verdict was made by another model or prompt, and `report` refuses a file that holds two judges (it prints the one it has). A **provider error** (a wrong key, a rate limit, a dropped connection) is not a verdict: nothing is written for it, it is counted, and five in a row stop the run (`JudgeOutage`); a reply that arrived and could not be read is stored as `parse_failed` and judged again by the next run. A reply without the claims or without `main_event_covered` is not a verdict either (it goes through the repair prompt first). The three summary rates (no unsupported or contradicted claim, a contradicted claim, main event covered) share one denominator: the summaries with a claim to check; one without is counted apart. **Unreadable replies are excluded from the rates, so the report counts them per bucket** (they may concentrate in the longest sources, where a longer reply is more likely to fail) and prints the worst case, in which every unreadable reply is an unfaithful summary; each verdict records the length of the judge's reply (`verdict_chars`) and the run prints the longest, so a cap on the reply (`max_tokens` 2000 in `news_nlp.eval.model`) would show. The claim shares (unsupported, contradicted) have a bootstrap interval that resamples summaries, not claims.
- **Not checked against the owner yet.** `tag-sheet --verdicts` writes a slice (40 per group) with the source, the summary and two blank columns in Spanish (`afirmaciones_sin_respaldo`, `evento_principal`), **without the judge's verdicts**: at most half of it are the summaries the judge flagged and the rest a random control of clean ones, because a random slice cannot measure the judge on a rare event (if 95% are clean, 40 hold two bad ones and kappa degenerates). A sidecar `to_tag.strata.json` (not shown to the owner) says which is which, the weighted size of each stratum and **which judge (model and prompt hash) chose the slice**: `tag-sheet` refuses verdicts of two judges and `agree` refuses verdicts made by another judge than the one that chose the slice (a rerun between the slice and the marks would make the precision and the sensitivity those of nobody). `agree` reads the marks and gives, by group: the agreement and Cohen's kappa **weighed back to the sample** (`raw`, `kappa`, **each with a bootstrap interval** resampled within the strata: one control mark can weigh sixty times a flagged one, and with 15 control items a single disagreement moves a kappa from 0.9 to 0.2; the slice is half flagged by construction, so the figures of the slice itself are under `slice`, with a note not to quote them; the same for the main-event question); the owner's rate of unfaithful summaries among the flagged ones (the judge's **precision**, with its interval) and among the clean control, each with a Wilson interval; and the judge's **sensitivity**, the two strata weighed by their size in the sample, **with an interval** (the extremes over the two Wilson intervals). The interval is wide on purpose: a control of twenty clean items stands for the whole clean population, so zero unfaithful among them still allows about 16% of it, and when the judge flags 5% of the summaries that is a sensitivity of 0.2 to 1.0; a control of 20 cannot establish a sensitivity above about 0.8 for a judge that flags so few. A bigger control (`--per-group`) is the only remedy. A blank mark stops it. The headline is the precision, the sensitivity and the weighed figures, each with its interval. **The sensitivity is conditional on a readable verdict with at least one claim**: `outside_population` counts the marked summaries the judge could not read or found no claim in, and they are outside the figure (the worst case of the report says what they could change). Without `--verdicts` the slice is random and `agree` says so.
- **Who judges, and what the buckets are.** The judge is the `.env` model (today `deepseek-chat`) and **must not be the model that wrote the summaries**: it would grade its own text. This is checked, not only said: `draw` records `summariser_model` in `sample.meta.json` (default: `c_summary` of `config/pipeline_models.json`; `--summariser-model` names another, e.g. an API model when T-206 is built), and `judge` **stops** when the judge's `LLM_MODEL` is that model, unless `--allow-self-judge` is given; the verdicts then carry `self_judge_allowed`, and the owner's marks (`tag-sheet`, `agree`) are the primary measure. The buckets are terciles of the **characters of the source** the summariser read (`--bucket-by chars`, the default; it exists for any summariser). `--bucket-by chunks` reproduces the chunk-and-reduce reading but **collapses on this corpus**: of the summarised items, single bodies are 58,151 with one chunk, 13,685 with two and 1,494 with three; sub-news 4,696, 327 and 23 (RESULTS at `summary-7a9bd2062bc4`, 2026-11-15), so both edges are 1, the middle bucket is empty and the draw is one chunk against more. The choice is in `sample.meta.json` and in the `sample_id`; the `chars` pass over SOURCE keeps only the lengths, then the chosen bodies are read (two passes, no text of the whole frame in memory).
- **Language.** The summariser reads English only and the corpus has Spanish bodies. Every sampled row carries the `language` of its article (a sub-news takes the one of its article), and `report` adds a table by language (not weighted): read the pooled rates with it. Quote the pooled (weighted) rate for the step and the table for the Spanish question; the language rows are unweighted and over the summaries with a claim to check. A body with no language in SOURCE is listed as `unknown`; its share of the real sample is read from the table once it is drawn.
- **A cap on the requests.** `judge --max-calls N` (default 1,000, for a sample of 300 rows) is a hard cap on the **requests** of a run: a row takes one, or two when its first reply cannot be read (the repair prompt), and an error counts; a retry that the provider's client makes inside one request is not seen. The cap is counted where the request is made, so a row cut in the middle leaves no verdict and the rows left are counted in `not_called`; a rerun goes on. The tokens the provider reports (`metrics.accumulated_usage` of the agent's result) are counted per verdict (`tokens_in`, `tokens_out`) and for the run; a stub or a client that does not report them counts zero, which is not a measure of zero spend.
- **It is not a `news_nlp.eval` run.** It lives in `scripts/` with its own report and is not tracked in MLflow (constitution AI #5 asks for an eval run for an accuracy claim about a stage; this is a measure of step 2, listed in `docs/evaluation.md`). It reads the reply with `news_nlp.eval.judges.coerce` and `repair_prompt`, the public names of the package's own reader (a test fails if it uses a private one).
- **Run it from a copy of the databases.** The corpus run of step 2 has finished (RESULTS at `summary-7a9bd2062bc4`: 74,235 body summaries and 5,048 sub-news summaries with `text_source = 'summary'`, 2026-11-15), so the real draw is **the owner's next step**. A live file refuses a read-only open while a step writes (and the two copies must be made at the same time: `draw` stops with a message if a sampled single body is not in the SOURCE copy): make a copy with `python -c "import sqlite3; s = sqlite3.connect('/path/nlp.db'); d = sqlite3.connect('nlp_copy.db'); s.backup(d)"` (the same for the SOURCE file), then:

```
uv run python scripts/summary_judge_2026_11_13.py draw --results-db nlp_copy.db --source-db urls_copy.db \
    --summary-revision summary-7a9bd2062bc4 --scope-revision scope-nli-8ff21e50796b --cut-revision units5-d3e2cbfa830e
uv run python scripts/summary_judge_2026_11_13.py judge          # LLM_MODEL must not be distilbart; about 300 rows, at most 600 requests
uv run python scripts/summary_judge_2026_11_13.py report
uv run python scripts/summary_judge_2026_11_13.py tag-sheet --verdicts data/summary_judge/verdicts.jsonl   # then the owner's marks and `agree`
```

Smoke test only (6 single bodies drawn from the snapshot of 2026-11-12, 4 judged by `deepseek-chat`, all supported): it showed the code works, not what the rates are; its paths and revisions were not recorded, so it is not repeatable.

## Not done

In T-190 (T-184 is closed): the run over the corpus (hours of GPU; the single bodies do not depend on the cut and can run now; the sub-news are read from the reference cut `units5-d3e2cbfa830e`, whose corpus run has finished: 161,574 sub-news of 30,305 bodies, 5,048 over the size, 64,480 items in all to summarise); the GPU time per item and the determinism check on the GPU; the run of the faithfulness measure (the tool is built: "Measuring the summaries" above).
`sector` bodies (T-167); the steps after this one (NER on the text, relevance, sentiment, category).
