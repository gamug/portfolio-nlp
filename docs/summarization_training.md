# Summarization: the lead-and-gate experiment (2026-11-13)

**Status: written 2026-11-13.** One experiment on how the summary step builds a summary from a long body: what was
tried, how it was measured, and the figures. Nothing here is in production: no production code, no database and no
revision was changed. The experiment is `scripts/summary_lead_experiment_2026_11_13.py`; its output is in
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
- The script is scratch: `ruff check` reports 12 style findings in it (statement count, magic values) that were
  not cleaned.

## Not done

- A metric of **company and main-event coverage** of the summary (for example, the share of the companies the
  extractor assigns to the body that appear in the summary, and a main-event check against the first chunk). The
  earlier summary experiments did not have it either.
- The extractive variants: C0 plus gated lead sentences of the later chunks, and a smaller tail budget (the tails
  are what lower conciseness).
- A new revision of the summary step. Any change to the model, the input format, the generation limits or the
  chunking and summary code starts a new revision (about 15 GPU hours of re-summarising), so it waits for a
  decision.

## Reproducing

```bash
uv run scripts/summary_lead_experiment_2026_11_13.py sample     # the 300 articles
uv run scripts/summary_lead_experiment_2026_11_13.py generate   # GPU, resumable
uv run scripts/summary_lead_experiment_2026_11_13.py nli        # GPU
uv run scripts/summary_lead_experiment_2026_11_13.py assemble
uv run scripts/summary_lead_experiment_2026_11_13.py judge      # needs LLM_API_KEY / LLM_MODEL / LLM_URL
uv run scripts/summary_lead_experiment_2026_11_13.py report
```
