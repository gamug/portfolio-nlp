# Matcher change of 2026-11-09: what was measured (PLAN D38, PR #155)

Read-only measurements of the change to `news_nlp.triage.CompanyResolver` (common words and surnames,
names with initials and accents, the disclosure column of the count CSV). Everything here is
reproducible with `scripts/matcher_before_after_2026_11_09.py` (`--base 786766f`, the last commit of
master before the change) and the label file `docs/matcher_labels_2026-11-09.csv` (`--labels-summary`,
`--render`; schema in `docs/modules/news-nlp.md`). One reader (the developer); samples are the unique
bodies of `id % 60 = r` (4,001 each); the word lists were read on `r = 7`.

## 1. What the change does to the counts

Bodies naming the company, before to after, sample `13`: FRT 579 to 0, STT 132 to 15, UNP 169 to 11,
TTD 116 to 3, WST 131 to 2, HD 103 to 25, TTWO 110 to 11, BE 97 to 2, XYZ 63 to 0, UHS 66 to 0, JKHY 48
to 0, MLM 42 to 2, TRV 62 to 4, ROST 46 to 8, WMB 42 to 7, BSX 84 to 13, EW 18 to 5, SPG 29 to 10, BKR
18 to 8; company-body pairs 12,352 to 9,807. The ten names with initials, "!" or accents are now
found (DHI 5 to 13, JBHT 1 to 5, SJM 6 to 10, TROW 0 to 8, USB 13 to 23, EL 0 to 10; 74 new matches read,
74 right); `--self-check`: 501 of 508 companies found by their own name (the 7 left are
`NAME_UNSAFE_TICKERS`), none lost.

Cost, what the rules drop: of 100 mentions dropped by the first rules, 1 was the company; of 80 dropped
by the second batch of words (sample `41`), 6 before `_KNOWN_NAMES` and the backtick, 1 after.

## 2. Precision of what is kept (each mention read)

| rule | sample | right / read |
|---|---|---|
| names and tickers (other companies) | 13, 29 | 37/39, 34/36 |
| the owner's 15 tickers | 13, 29 | 86/89, 38/40 |
| short forms and corroborated words | 13, 29 | 12/14, 26/27 |
| first-word alias | 13 (before the second edit), 29, 41 | 59/94 (63%), 80/94, 86/98 (88%, no edit chosen from it) |

Wrong ones left in the alias rule: Willis (Bruce, Fani), Goldman (Ken, Rob), Norwegian (crown), Southwest
("South by Southwest"), Delta (variant), Quest. Scope agreement with the owner's 720 tags (last rule
step): 0.696 before, 0.700 after.

## 3. The disclosure column (`*_bodies_disclosure`)

As the CSV reads the bodies (after the boilerplate rules), the share of counted company-body pairs
named only by ticker forms inside a disclosure or a ticker list is 11.7%, 12.9%, 12.4% and 11.8%
(samples `13`, `29`, `41`, `53`); of the `subject` pairs 5.2% to 7.3%; of the `single` pairs 0.3% to 0.7%
(not adjusted; the README prints the run's own). The first detector's generic branches read alone on
`59`: "owns" 21 of 25, six upper-case words 13 of 25; now every phrase is anchored to a ticker after
it. On `17`, no edit chosen from it: 40 of 40 flagged pairs were disclosures; of 39 readable unflagged
ticker-only pairs 4 were disclosures missed (four phrasings added after the read) and 5 were not the
company. It is a reading by patterns, not an exact count.

## 4. What the change does to the rule scope (the measurement of PR #155, round 8)

`--scopes`: the scope and the subject companies of the unique bodies read exactly as
`cut_summaries_2026_11_07.py` reads them (`size_triage.classify`, the stored NER spans of the first
article), with the matcher at 786766f and with the working tree.

| sample | bodies | rule scope changed | subject set changed | enter the set that gets a summary | leave it |
|---|---|---|---|---|---|
| backtest window (`ok`, published 2021-12-31 or later), `id % 10 = 3` | 3,000 | 167 (5.6%) | 374 (12.5%) | 30 (1.0%) | 8 (0.3%) |
| every `ok` body, `id % 20 = 5` | 4,000 | 240 (6.0%) | 541 (13.5%) | 59 (1.5%) | 20 (0.5%) |

Scope shares, every `ok` body (before to after): single 43.6% to 47.4%, single + mentions
(`multi_dominated`) 22.8% to 19.8%, round-up 19.2% to 18.9%, several 10.6% to 9.8%, no company 1.6% to
1.7%, sector 1.3% to 1.5%, economy 0.6% to 0.6%, market 0.3% to 0.3%. The main move is "single + mentions"
to "single" (141 of 240): a common word no longer makes a second company. (These shares are of this
read, with stored spans; the absolute shares of the sizing table of `docs/evaluation.md` come from
another run, so compare the shifts, not the levels.)

## 5. Figures documented with the matcher at 786766f (not recomputed; marked in place)

| where | figure | what to know |
|---|---|---|
| `docs/evaluation.md`, "Sizing the corpus" (table of rule scopes, "subject companies per body") | 81,961 bodies by scope: 51.0% single, 22.9% single + mentions, 12.2% round-up, 9.6% several, 4.4% with no subject | moves as in section 4 (about +4 points single, -3 single + mentions); the 47-minute run is not repeated unless the owner asks |
| `.specify/memory/PLAN.md`, held-out weights (`frame`: single 514, multi_dominated 245, roundup 108, multi 93, sector 18, no_company 12, economy 5, market 5) | the held-out gold is re-weighted by the size of each rule scope | the strata of the gold bodies are a property of the matcher that drew them: the gold keeps its strata and the weights stay as drawn; the draw is not repeated |
| `.specify/memory/TASKS.md`, T-148a | "the rule scope is the same with and without the stored NER spans (500 of 500)" | measured with the old matcher |
| the cut `data/cut/summaries` (117 chunks, 29,113 bodies, manifest `rules: f0a251aaf437`) | the scope of each body, so which got a summary | section 6 |

Not affected: counts of bodies by date or status (T-159, T-160's bodies, the 29,113 of the cut's
window), the models, the gold labels of the owner.

## 6. The 2026-11-16 cut (proposed; the owner decides)

The cut is complete (117 chunks of 250, 29,113 unique bodies, summariser pinned) and was made with
the rules of manifest revision `f0a251aaf437`, which is not master's (`d883f7c2fee2`): it is already
"made by other code" and a resume would already be refused. `read_cut_summaries` does not check the
rules' revision, so the finished cut keeps reading after the merge. What the new matcher would change
(section 4, window sample): 1.0% of the bodies (about 290 of 29,113) would enter the set that gets a
summary, 0.3% (about 80) would leave it, 5.6% change scope, 12.5% change their subject companies
(which matters for the company texts built later with the new matcher).

* **A. Freeze (proposed):** the cut stays as built; its README/index says it was made with the rules of
  `f0a251aaf437` (matcher before PR #155), so nobody compares its scopes with a count made after it.
  Nothing is rebuilt; about 290 bodies that the new rules would summarise have no summary.
* **B. Add the difference:** summarise only the bodies that enter the set (about 290, at 0.84 s per body
  about 4 minutes), into a second, separate cut folder, and leave the 80 that leave it.
* **C. Rebuild:** a new cut with the merged rules (29,113 bodies at 0.84 s: about 6.8 hours).
