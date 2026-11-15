# Summarization: the lead-and-gate experiment (2026-11-13) and the DeepSeek arm (2026-11-15)

**Status: written 2026-11-13, a section added 2026-11-15.** The experiment on how the summary step builds a summary from a long body (what was
tried, how it was measured, the figures), and, at the end, the evidence and the owner's decision to summarise with DeepSeek (T-206). Nothing here is in production: no production code, no database and no
revision was changed; T-206 is a task, not a change. The experiment is `scripts/summary_lead_experiment_2026_11_13.py`; its output is in
`experiments/results/summary_lead_2026_11_13/` (git-ignored scratch, including `summaries.csv` with one row per
article and arm).

The summariser is `sshleifer/distilbart-cnn-12-6`. It is not trained here (there is no fine-tuning step for it);
"training" in the file name means the work to improve the stage, as in `sentiment_training.md`.

## Why

The v1 and the current step summarise a long body with a **chunk-and-reduce**: the body is cut into chunks of at
most 1,000 BART tokens, each chunk is summarised (56 to 142 tokens), and the joined chunk summaries are chunked
and summarised again, up to 6 passes (`summary_stage.hierarchical_summarize_batch`). The suspicion was that the
re-summarising pass compounds errors and causes the hallucination and the misattribution (a fact moved to another
actor). Run 44 of the evaluation (`docs/evaluation.md`) supports the first half: faithfulness falls and
hallucination rises with the number of chunks.

| Run 44, stored v1 summaries | Faithfulness | Coverage |
|---|---|---|
| 1 chunk | 4.79 | 3.56 |
| 2 chunks | 4.53 | 3.13 |
| 3+ chunks | 4.30 | 2.88 |

## The hypothesis

Stop the re-summarising, give the first chunk the weight (the main event is usually at the top), validate each
sentence with an NLI model, and join the pieces with no further summarising. The open question was the
"Frankenstein effect" of the joined text.

The pipeline tested:

1. Prefix each chunk with the title (the title was only in chunk 1 before).
2. Balanced chunks: the body is cut into `n = ceil(tokens / 1000)` chunks of about the same size, so no chunk is a
   tiny tail (a tail under a quarter of the cap is merged into the chunk before it).
3. Chunk 1 is summarised at full length (56 to 142 tokens); each later chunk at a shorter length (12 to 60 tokens),
   at most 8 tails.
4. NLI gate: every summary sentence is scored against windows of its own chunk (title included, 360-token windows
   with a 1-sentence overlap); the best entailment probability is kept; a sentence under **0.5** is dropped. The
   threshold was fixed before any score was seen. If every sentence of chunk 1 fails, it falls back to its first
   two lead sentences. The NLI model is `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`.
5. Dedupe sentences (Jaccard 0.6), trim to a 450-token budget, join one block per line.

## The sample

The same stratified sample as run 44, so the numbers compare with the earlier ones: **300 articles, 100 per tier**
of `num_chunks` (1, 2, 3+), drawn with seed 1 from the articles of `eval_run` 44. 288 are English and 12 Spanish
(the summariser is English-only, so the language is reported separately).

## The arms

| Arm | What it is |
|---|---|
| A | The stored v1 summary of run 44 (re-judged, to measure the judge's drift) |
| B | The current step on the `Title-…\nBody-…` input, with the re-summarising (separates the input format from the method) |
| C0 | Chunk 1 only (do the tails help at all?) |
| C1 | The pipeline above with no NLI gate |
| C2 | C1 with the NLI gate |
| C3 | Gated chunk 1, plus the first 2 sentences of each later chunk copied verbatim (extractive tails), gated and deduped |

## How it was measured

The LLM-as-judge of `news_nlp.eval`, with the run-44 prompt (`prompts/c_summary.md`), DeepSeek at `temperature=0`.
Every arm is re-judged now, so the comparison is inside one judge. The judge gives `faithfulness`, `coverage` and
`conciseness` on 1 to 5, a list of `hallucinations` and a rationale. The hallucination rate is the share of
summaries with a non-empty list. Identical summaries are judged once (cached by article and summary hash).
Differences against B are paired by article, with a 95% bootstrap interval.

**What the judge does not measure.** `coverage` is one overall score of "the article's main point(s)". There is no
metric of whether the companies the body is about, or the main event, are in the summary, and no per-company
check. The company coverage figures in `extractor_training.md` and `gold_datasets.md` belong to the extractor
(companies left without a text), not to the summary. See "Not done".

## Results

All 300 articles (higher is better, except hallucination):

| Arm | Faith | Cov | Conc | Halluc | Tokens |
|---|---|---|---|---|---|
| A | 4.54 | 3.23 | 4.07 | 19.7% | 71 |
| B | 4.60 | 3.24 | 4.08 | 15.0% | 69 |
| C0 | 4.70 | 3.31 | 4.33 | 13.0% | 69 |
| C1 | 4.39 | 3.47 | 3.23 | 25.7% | 122 |
| C2 | 4.64 | 3.31 | 3.44 | 14.7% | 100 |
| C3 | 4.81 | 3.13 | 3.32 | 6.7% | 137 |

By tier (faith / cov / halluc):

| Arm | 1 chunk | 2 chunks | 3+ chunks |
|---|---|---|---|
| A | 4.75 / 3.74 / 11.0% | 4.64 / 3.21 / 16.0% | 4.24 / 2.73 / 32.0% |
| B | 4.67 / 3.81 / 14.0% | 4.64 / 3.11 / 11.0% | 4.49 / 2.81 / 20.0% |
| C0 | 4.68 / 3.75 / 14.0% | 4.69 / 3.30 / 14.0% | 4.72 / 2.88 / 11.0% |
| C1 | 4.68 / 3.75 / 14.0% | 4.45 / 3.45 / 25.0% | 4.05 / 3.22 / 38.0% |
| C2 | 4.88 / 3.53 / 6.0% | 4.61 / 3.28 / 17.0% | 4.42 / 3.13 / 21.0% |
| C3 | 4.88 / 3.53 / 6.0% | 4.77 / 3.21 / 9.0% | 4.79 / 2.66 / 5.0% |

Paired differences against B over the 300 (mean and 95% interval):

| Arm | Faithfulness | Coverage | Hallucination |
|---|---|---|---|
| C0 | +0.10 [+0.01, +0.18] | +0.07 [−0.01, +0.14] | −2.0 pts [−5.7, +1.7] |
| C1 | −0.21 [−0.31, −0.10] | +0.23 [+0.15, +0.31] | +10.7 pts [+6.3, +15.3] |
| C2 | +0.04 [−0.07, +0.14] | +0.07 [−0.02, +0.16] | −0.3 pts [−5.0, +4.3] |
| C3 | +0.21 [+0.13, +0.30] | −0.11 [−0.20, −0.02] | −8.3 pts [−12.0, −4.7] |

### What it shows

- **The hypothesis did not hold as stated.** Removing the re-summarising and joining the chunk summaries (C1) is
  worse than the current step: lower faithfulness and +10.7 points of hallucination; in the 3+ tier 38% against
  20%. The reduce pass is not the only cause: the abstractive summaries of short tail chunks add hallucination
  faster than coverage (coverage +0.23, conciseness 4.08 → 3.23).
- **The gate recovers it.** C2 returns hallucination to the level of B (14.7%) and keeps a small coverage gain,
  not significant overall, significant in tiers 2 and 3 (+0.17 and +0.32).
- **The lead alone is good.** C0 beats B on faithfulness at the same length (+0.10; +0.23 in the 3+ tier, with
  hallucination 11% against 20%).
- **Verbatim sentences are the most faithful.** C3 has the lowest hallucination (6.7%) and the best faithfulness
  (+0.21), at the cost of coverage (−0.11) and of length (137 tokens against 69).
- **The gate changes tier 1 too.** C0, C1 and B are nearly identical there, but the gate drops sentences from
  summaries that had no re-summarising: faithfulness +0.21, coverage −0.28.

### Mechanics

- NLI gate, sentences under 0.5: 15.9% (tier 1), 17.8% (tier 2), 22.8% (tier 3). Chunk 1 fell back to the lead
  sentences 3, 3 and 2 times. Echoed titles stripped from the output: 11, 27 and 36.
- Judge drift, arm A against its stored run-44 verdict: same faithfulness 90.7%, same coverage 86.7%, same
  conciseness 77.3% (mean shift −0.19), hallucination flag agrees 93.7%. Comparisons inside this experiment are
  sound; comparing with the published run-44 figures is less so.
- English (288): B 4.62 / 3.24 / 14.9%, C2 4.67 / 3.32 / 13.5%. Spanish (12): B 16.7% hallucination, C2 41.7%; too
  few to read, and the model is English-only.
- Budget: C1 averages 122 tokens, C2 100, C3 137, against the 512-token limit of the downstream models; the
  450-token cap was not binding on this sample.

## Limits

- One sample, one judge (DeepSeek), one run per arm; the 95% intervals are on the 300 or on the 100 of a tier.
- The judge `coverage` is one overall score; it does not say whether the companies or the main event are present.
- The NLI threshold (0.5) was not tuned; the raw scores are saved in `nli.jsonl`, so other thresholds can be
  assembled without generating again.
- The script is scratch: its style findings (statement count, magic values) are not cleaned but **named** in its `# ruff: noqa:` line (a new kind of finding
  is reported); it was lost from the working tree once and recovered from the session record, so it is committed with this document.

## Not done

- A metric of **company and main-event coverage** of the summary (for example, the share of the companies the
  extractor assigns to the body that appear in the summary, and a main-event check against the first chunk). The
  earlier summary experiments did not have it either.
- The DeepSeek arm was run afterwards (section "T-206" below); it is not one of the arms C0 to C3. The extractive variants: C0 plus gated lead sentences of the later chunks, and a smaller tail budget (the tails
  are what lower conciseness).
- A new revision of the summary step. Any change to the model, the input format, the generation limits or the
  chunking and summary code starts a new revision (about 15 GPU hours of re-summarising). The owner decided on
  2026-11-15 to summarise with DeepSeek (T-206, below); **what is left is the build** (no code of the step has changed).

## Reproducing

```bash
uv run scripts/summary_lead_experiment_2026_11_13.py sample     # the 300 articles
uv run scripts/summary_lead_experiment_2026_11_13.py generate   # GPU, resumable
uv run scripts/summary_lead_experiment_2026_11_13.py nli        # GPU
uv run scripts/summary_lead_experiment_2026_11_13.py assemble
uv run scripts/summary_lead_experiment_2026_11_13.py judge      # needs LLM_API_KEY / LLM_MODEL / LLM_URL
uv run scripts/summary_lead_experiment_2026_11_13.py report
```

## T-206: DeepSeek as the summariser (owner, 2026-11-15)

**Evidence.** The same 300 articles and judge (`scripts/summary_deepseek_arm_2026_11_15.py`; prompt: about 60 words, facts only, name the actors; temperature 0). **The figures are committed, without text,** in `docs/summarization_results_2026_11_15/` (`deepseek_vs_current.csv`, `blind_24.csv`, `tables.md`) and regenerated by `uv run scripts/summary_deepseek_report_2026_11_15.py tables`; a test checks that the numbers below are theirs.
DeepSeek (D) against the current step (B), paired 95% intervals: faithfulness 4.99 against 4.60 (+0.39 [+0.29, +0.49]), coverage 4.54 against 3.24
(+1.29 [+1.21, +1.39]), conciseness 4.98 against 4.08, hallucination 0.7% against 15.0% (-14.3 pts [-18.7, -10.3]); 0% against 20% in the 3+ tier.
Limits: DeepSeek was summariser and judge. A blind re-scoring of 24 pairs by the assistant (Claude, not a person; 8 per tier, English only, bodies under 9,000 characters: the long bodies are left out, see the end of this section) agreed: faithfulness 4.92 against 4.42, coverage 4.75
against 3.08, hallucination 4% against 21%; it is small and the arms are guessable by style. There is no company or main-event metric, and the prompt is one run.

**Decided by the owner.** (a) Scope: the stepped pipeline only (`--step summary`: single bodies and sub-news over 2,000 characters, and the assigned `sector`
bodies); the v1 chain stays on distilbart. (b) No gate before the run: the corpus run goes first and the evaluation after it (the blind marks, the company and
main-event metric, an `news_nlp.eval` run against run 44). (c) Failure: fail fast and resumable (a few retries, then stop; rows written stay; no fallback to
distilbart, no empty summary). (d) Revision: a hash of the prompt, the model name, the temperature and the size rule; each row stores the API model id and the date.

**To build.** The summariser in `news_summary_stage` (OpenAI-compatible `chat/completions`, `LLM_*` from `.env`, bounded concurrency, the prompt in a tracked
file), tokens in and out in `--status`, a logged call or spend cap, tests with the API mocked, `docs/summary_step.md` updated.

**Cost.** 64,480 items over 2,000 characters hold 250.3M characters (single bodies 235.1M, sub-news 15.3M; mean 3,882): about 63.5M input and 5.6M output
tokens. **Rates checked on 2026-11-15** on https://api-docs.deepseek.com/quick_start/pricing/: the page lists the current model as `deepseek-flash` (older names such as `deepseek-chat` are billed at its price), with **peak and off-peak** prices, cache-miss input $0.15 to $0.30 and output $0.60 to $1.20 per million (cache hits are far cheaper and the shared prompt is a small part of each call). That is **about $13 to $26** for the run, depending on when it is made; the 2024 figures this document first assumed ($0.28 and $0.42) are not the current ones. The columns of the page were not unambiguous in the search result: confirm them on the page before the run. The sample itself cost about 14 cents.

**Judging the corpus run.** The judge of step 2 (`docs/summary_step.md`, "Measuring the summaries") must **not be DeepSeek** when the summaries are DeepSeek's (it would grade its own text, the limit of this evidence): `judge` stops when its `LLM_MODEL` is the `summariser_model` recorded by `draw` unless `--allow-self-judge` is given (PR #180), and its buckets for summaries with no chunks are cut by `--bucket-by chars`. The owner's marks (`tag-sheet`, `agree`) are the primary measure then.

**Open.** Constitution AI #1 (a model swap needs an eval run first) and #2 (CPU-capable, deterministic inference) against the owner's choice to evaluate after the
run: reconcile `SPEC.md` and the constitution in the same change. A text the API refuses (kept as the original, or the review list). That article text leaves
the machine for an external API: to confirm.

**Reproducing the arm.** The scratch (`experiments/results/summary_lead_2026_11_13/`, git-ignored) is built by `summary_lead_experiment_2026_11_13.py sample` from `eval_run` 44 of the owner's RESULTS file and SOURCE copy, so it cannot be downloaded; the first three commands need it and `LLM_*` in `.env`, the last two need only the repository:

```
uv run scripts/summary_deepseek_arm_2026_11_15.py generate            # the API; --max-calls caps the requests, a rerun goes on
uv run scripts/summary_deepseek_arm_2026_11_15.py attach              # arm D into arms.json (a backup first)
uv run scripts/summary_lead_experiment_2026_11_13.py judge            # 300 more verdicts
uv run scripts/summary_deepseek_report_2026_11_15.py export           # the committed CSV, no text
uv run scripts/summary_deepseek_report_2026_11_15.py tables           # the paired table
```

**The blind re-scoring of 24 pairs.** Chosen by `scripts/summary_blind_pairs_2026_11_15.py` (`choose`; `check` reproduces the committed `blind_24.csv` from the scratch sample, and a test does when the scratch is there): the English articles of the 300 whose body is **under 9,000 characters**, 8 for each tier, `random.Random(7)`, shuffled, and a coin per pair for which summary is shown first as `X`. **The filters leave the long bodies out**, so the 3-or-more-chunks tier, where the current step is weakest, is represented by its shorter bodies; five bodies were also shown cut (the reader saw the first 5,500 to 6,000 characters) and claims in the cut part could not be checked. The scores are **the assistant's reading** of the article and the two summaries, made before its key was opened (`blind_24.csv`); they were not produced by a script, and the style of the current step's summaries (extractive, `.` between sentences) makes the arms guessable. It is 24 pairs: a check of the direction of the judge's figures, not an estimate of their size.
