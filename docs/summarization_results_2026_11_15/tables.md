Paired by article, 300 articles; 95% bootstrap intervals (seed 0, 4000 draws). B = the current step, D = DeepSeek.

| | B | D | D - B [95%] |
|---|---|---|---|
| faithfulness | 4.60 | 4.99 | +0.39 [+0.29, +0.49] |
| coverage | 3.24 | 4.54 | +1.29 [+1.21, +1.39] |
| conciseness | 4.08 | 4.98 | +0.90 [+0.80, +1.00] |
| hallucination | 15.0% | 0.7% | -14.3 pts [-18.7, -10.3] |

| tier | n | B faith / cov / halluc | D faith / cov / halluc |
|---|---|---|---|
| 1 | 100 | 4.67 / 3.81 / 14% | 4.97 / 4.87 / 1% |
| 2 | 100 | 4.64 / 3.11 / 11% | 4.99 / 4.50 / 1% |
| 3 | 100 | 4.49 / 2.81 / 20% | 5.00 / 4.24 / 0% |

Tokens of the DeepSeek calls: 487589 in (1625 a call), 26085 out (86 a call).

Run to run (temperature 0, 30 articles summarised twice by DeepSeek; `deepseek_repeat_30.csv`):

* identical summaries: 0 of 30; mean word overlap (Jaccard) 0.66
* faithfulness: first run 5.00, repeat 5.00
* coverage: first run 4.70, repeat 4.63
* conciseness: first run 4.97, repeat 4.97
* hallucinated: first run 0, repeat 0 of 30

Blind re-scoring of 24 pairs (8 per tier, English; the arm hidden when scored; `blind_24.csv`):

| | B | D |
|---|---|---|
| faithfulness | 4.42 | 4.92 |
| coverage | 3.08 | 4.75 |
| conciseness | 3.50 | 4.83 |
| hallucination | 21% | 4% |
