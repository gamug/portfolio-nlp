# Open-task entries of TASKS.md, as they stood before the condensation of 2026-11-08

Verbatim text of the open tasks that `TASKS.md` now carries in a short form (same task IDs, same status).
Read only when the history of one of them is needed. Moved by the change that added `.specify/memory/INDEX.md`; cut from `origin/master` at 8c156f28241d0a3eff36eee46b709662356ac04b (diff against it to see every change).


## T-148b10

                        - [ ] **T-148b10** *(owner's decisions, interactive session 2026-11-08)* (1) The candidate of step 4 for the cut is `units` (94.0% [87.0, 99.2] on the 167 real companies, pass; 83.2% [72.7, 91.0] on every company named, not decided; its 22 texts for non-companies are out of the first and in the second: see (8)); `units3` and `units2` stay as measured fallbacks. (2) T-148b9 (header names matched with the plain resolver) is a proposed improvement, out of scope. (3) A confirmation draw on bodies not used so far, `scripts/draw_faithfulness_2026_11_06.py --exclude-from data --out data/faithfulness_confirmation` (the pool is 32 bodies, 30 drawn), approved for `units` and the baseline (the acceptance of D32 is the bar and better than the baseline on the same bodies); the number of texts and the translation spend are settled before anything is bought. (4) The stack #141, #143, #144 merges in that order, by the owner. (5) The sentiment tagging of the 450 summaries is done and its marks are migrated to `gold/` in #145 (open; merging it is the owner's); the retraining (T-148c3) waits for the owner's word. (6) **[Superseded by (9)]** **Proposed, the owner decides, written before the confirmation set is read:** the reading that decides it is the bar on the companies left after the owner's `real_company` marks of that set; `units` passes if the lower end of its interval is at or above 80% and its gain over the baseline is `better` (D32), read in the block of all the confirmation texts (`[all, real]`, the gain `units` minus `baseline` in that block); the 2022+ block is informative and printed with its size (too small to be a verdict: 33 texts of 7 bodies on the first set); if the owner wants 2022+ to decide, that is a different and weaker test, to be said before the reading; the other readings are informative; if `units` does not pass, it goes to the cut marked as measured when its interval straddles the bar and does not go when it is wholly below (PLAN D34 (e)). **(7) [Superseded by (9): the confirmation was read and the cut is no longer held]** Owner's position, 2026-11-08, interactive session, replacing the earlier "first reading is enough": the cut is held; the owner will tag the confirmation set (357 texts, 30 exempt companies, in `data/faithfulness_confirmation/`) when ready, and nothing is asked until they do; no pin; the sentiment retraining (T-148c3) waits for #145 to be merged. Only the 357 texts are read: the owner will not read the 30 exempt companies (`faithfulness_exempt_to_read.csv` of the confirmation), so their Spanish is never bought and the rule of (6) is read in its second form (below).** **(6b) [Superseded by (9); not met, the selection is the owner's call over it]** Proposed, the owner decides, written before the 357 texts are read: the confirmation is read without the `real_company` marks.** The block that decides is `[all]` and the line that carries the bar is the one printed as "the bar on every company named" for `units` (the companies it wrote no text for and is not asked about count as unfaithful; a non-company it writes a text for counts as unfaithful too, because nobody has marked it); `units` passes if the lower end of its interval is at or above 80% and its gain `units` minus `baseline` in `[all]` is `better` (D32); if the interval straddles 80% it goes to the cut marked as measured, and if the whole interval is below 80% it does not go (PLAN D34 (e)); the 2022+ block and the comparisons with `model` are informative. **Scale of what the missing marks cost, measured on the first 30 bodies (2026-11-08):** `units` is 83.2% [72.7, 91.0] on every company named (not decided; 85.4% [76.5, 92.2] counting only the companies it wrote a text for) and 94.0% [87.0, 99.2] once the 23 non-companies are out (pass); the baseline is 60.5% [45.2, 73.8] and 68.9% [53.6, 82.1]; the gain `units` minus `baseline` is 24.3% [12.9, 39.4] and 25.1% [12.3, 41.4], `better` in both. So reading without the marks leaves the gain as it is and moves the bar: on the first set it would have been "not decided" and not "pass"; the same is expected of the confirmation. **(8) [Superseded by (9)]** Schedule, proposed, the owner decides: the faithfulness marks of the first set are in (190 texts, and the reading of the 37), so the 2026-11-12 go/no-go (T-154) can read faithfulness on them in two readings, both printed and neither chosen here: D32's own, on every company named, where `units` is 83.2% [72.7, 91.0], not decided (it goes to the cut marked as measured, PLAN D34 (e)); and on real companies (your 23 marks), 94.0% [87.0, 99.2], pass, which leaves out the 22 of its 185 texts written for non-companies (21 unfaithful: its own errors, which a cut ships; `units3` writes none of them and has 7 unfaithful texts of 148, but leaves 19 real companies without a text against 4); which reading the go/no-go uses is the owner's; the confirmation can only add to it or lower it and is not a condition of the go/no-go unless the owner says so; if it is not tagged by the evening of 2026-11-11, the decision of 2026-11-12 uses the first-set reading, said to be development figures, and the cut of step 4 waits for the confirmation reading as the owner held it; the sentiment fallback is unchanged (it waits for #145 and T-148c3, not for this).** **(9) Owner's selection, 2026-11-08, after the confirmation reading (357 + 38 texts tagged, 30 exempt names read): `units` ("units1") is the base model of step 4; the owner judged 80% accuracy enough. On the confirmation set `units` is 86.1% [80.9, 90.7] on every company named (pass) and 95.3% [92.0, 98.2] on real companies; it is level with the baseline (gain -0.5 [-1.5, 0.0], not decided), so (6b)'s "better" condition is not met and the selection is the owner's call over it; `units2` and `units3` are not selected (78.4% and 79.9% on every company named, 7 to 12 real companies left without a text). Not a pin, not a publication. Record, datasets, metrics, results and the options for what is next: `docs/extractor_training.md`.**


## T-148b4

        - [ ] **T-148b4** *(owner's reading of 2026-11-07, PLAN D27)* The attribution reads units cut as the owner reads the text, and without the site's footer. **Built** (`src/news_nlp/attribution_units.py`, stacked on T-148b3's PR): `strip_footer` (credit lines and "read more" links anywhere, a "more from" header with its short link lines, a disclosure block that opens in the last 40% of the body, a run of footer lines at the end; a body left with under 200 characters keeps it); `attribution_units` (a line, a bullet, a sentence, a part between semicolons, and a sentence that names several companies cut where one ends and the next begins, only when at least two words are said between the mentions and a clause ends there, so "Amazon and eBay announced" and a list of names stay whole); `model_company_texts(..., units=True)`; a new extractor `units` in the faithfulness draw (`--add-extractor units`, the owner's 300 marks untouched, the 83 texts that are the same as an earlier text keep its mark, 102 new texts to read), the report gives `units` minus `baseline` and `units` minus `model` next to the others; a company named only in the footer is not asked of `units`. `chunking.split_sentences` is not changed (sentiment and NER use it). Audited on 4,000 random bodies of the corpus (`scripts/audit_footer_2026_11_08.py`, seed 20261108; the first figures of this item, 1,208 changed and 13 losing more than a fifth, came from a draw whose seed was not recorded and are not used): the first footer rules changed 1,157 and the current ones change 1,148, 24 of them losing more than a fifth of their text; the two sets of rules differ on 36 bodies, where the first removed story lines and the current ones keep a long legal notice out of the story; the ones read are disclosure blocks (Seeking Alpha's, Cramer's), "Got something to say?" calls and "Questions" lines. On the 30 bodies: 107 of the 190 company texts change and the total text falls from 114,932 to 77,785 characters. **Read by the owner 2026-11-07 (all 402 rows marked): `units` 85.4% faithful [76.5, 92.2] on 185 texts (158 faithful), point above the 80% bar and interval straddling it, so "not decided" (D32); `units` minus `baseline` +24.3 points [12.9, 39.4] `better`; `units` minus `model` +8.6 [-0.7, 22.8] `not decided`; 2022+ (33 texts) the same as `model`, 87.9% [68.8, 100.0]. Of the 27 texts not faithful, 21 are companies reached only through the first word of their name (\"Bloom\", \"Union\", \"Home\", \"Equity\", \"Baker\", \"Boston\"...): such texts are 34% faithful (11 of 32) against 90 to 100% for the rest; the others are lists headed by companies (\"A, B - text\") whose text names other companies, and a text about Target and eBay given to Amazon. The reading is the current result and stays as it is (the next steps are the plan below, separate extractor).** **Result and what is the owner's:** the owner read all 402 rows (above); `units` minus `model` is `not decided`, so the earlier proposal (`units` replaces `model` if its gain is `better`) does not apply; which reading of the bar counts (the footer-only companies exempt, as proposed, or counted as unfaithful), and which of `units`, `units2`, `units3` (#144) goes to the cut, are the owner's decisions. **Review rounds (manager, #143), 2026-11-08:** round 1: with `units` the candidates are the companies of the body without its footer; the footer label words are anchored (`Subscribers rose...`, `Comments from the CEO...`, `Questions remain...` are story) and a short sentence after a "More from" header is story; the report prints the three gains and a second reading of the bar. Round 2: credits and notices (`Reporting by`, `Source:`, `Sign up`, `Copyright holders...`) are footers only as one short line (120 characters; a legal notice keeps its length); `draw_faithfulness --stale` (free, read-only, needs the model) lists the marked rows the code no longer writes, the new texts, and the rows written by another revision: on the 30 bodies, stale 2 (122960 Apple and Amazon, which now keep a story sentence the first rules removed: the `units` texts of those two differ from the ones the owner marked; the new texts are not added, the translation needs the owner's approval) and 565 rows with an older revision whose text is the same; `faithfulness_footer_only.json` records the revision of the footer rules and the report refuses a file written by other rules; the report warns when the rows of an extractor were written by several revisions. **Follow-up T-148b4b (proposed, not built; its precondition is the owner's decision on which reading of the bar counts):** the gold builder carries `faithfulness_footer_only.json` and `faithfulness_stale.json` into `gold/`, so `report_wi18 faithfulness --gold` prints the second reading of the bar and the staleness line like the working files do. **Review round 4 (manager, #143):** the staleness result is kept per extractor (a partial run does not erase the others) and the report compares the code part of each extractor's revision with the current code (`units`, `model`, `baseline`), so a check made before an edit to the footer rules is said to be old; `--stale` exit code is a sum (1 stale, 2 unchecked), `--only` limits it. **Review round 3 (manager, #143):** `--stale` names what it did not check and exits 2 then (1 if some row is stale, 0 otherwise), and writes `faithfulness_stale.json`; a changed footer rule no longer stops the report (the second reading is skipped with a line); the revision warning was replaced by one line from the stale file; docs updated. **Not in this item:** the name matcher that takes "Michael Bloom", "Union Square", "James Baker" and "Equity markets" for companies (own item; a guard such as the full name or no person names), and the per-sentence attribution of the model itself.


## T-132

- [ ] **T-132** **→ Work item 18 P1 and P3 (2026-10-28). Body-hash grouping and name/ticker resolution shipped in P1 (`news_nlp.triage`, T-144); **shrunk 2026-11-05 by the owner: only the upstream alias table stays open. The per-pair verdicts, T-130's judge and the block on T-117 (both superseded) are dropped; the quarantined pairs follow D19 (T-131). The text below is the original design, kept as the record.** *(from `portfolio-data-mining` T-055 design v1, approved 2026-10-15;
      reshaped 2026-10-22 to the 2026-10-21 hand-off)* Decide the quarantined
      `review` pairs. Never act on doubt, in either direction.
      - **Goal:** answer "is the company a subject of this article?" for the pairs
        discovery could only mark `review` (weak URL/title evidence). **Input:** all the
        quarantined `review` pairs selected at run time (52,039 on 2026-10-21; the count
        moves). **Subset that already has results:** 49,960 of the articles in this
        repo's RESULTS store are `review` (measured 2026-10-22); only those can have
        results purged (T-131's rule), while a verdict on a pair with no RESULTS rows can
        only feed upstream's release decision. The verdicts are the four data-mining
        applies: `about`, `mention`, `other`, `unclear` (D5).
        The correct rate of this class is **not measured** (only 18 of 20 uncontested
        News Corp URLs); do not assume it.
      - **Inputs:** a read-only attach (D6) of SOURCE, selection
        `articles` ⋈ `discovered_urls` on `id`, `match_class = 'review'`,
        `fetch_status = 'quarantined'`, **re-run on every run** because the extractor
        promotes pairs out of the class and the run records the class it saw. The exact
        query and the quarantine shape are defined only upstream (T-057/T-058); this
        repo never writes `urls.db`. Also the body, the title, the **stored
        `article_entities`** of those articles (verified 2026-10-22: 49,949 of 49,960 have
        them, 49,766 with an `ORG` span; zero GPU) and upstream's alias table (names, aliases, ticker forms, offered as a CSV).
      - **Steps:** (1) dedupe by body hash (share classes carry one body under two
        pairs; decide per pair, compute per body). (2) CPU rule first, no model call:
        `about` on the full normalized name with aliases, an explicit ticker form
        (`NYSE: X`, `$X`, uppercase) or an ORG span resolving to the company, weighted
        by title/lead presence and share of ORG spans; `other` when only a bare
        ticker-as-word or a different entity matches; `mention` for a name in a roundup;
        else `unclear`. (3) Store verdicts in a new additive RESULTS table, key
        `(article_id, ticker, run_id)`, with `verdict`, `confidence`, `model`,
        `evidence`, the class and `match_reason` seen; created idempotently, no existing
        table touched. (4) Export a CSV with exactly `article_id, ticker, verdict,
        confidence, model, run_id, evidence` for upstream's audited apply pass. (5)
        Measure against upstream's hand labels (its shared sample, its fresh validation
        set, the `loose_name` hand sample of 10 News Corp pairs and 20 pairs of other
        companies, which is the **first pilot**: 413 of the 448 `loose_name` pairs, held on purpose upstream, and 42 URLs
        that also name another tagged company): precision/recall per verdict and the
        `unclear` share. (6) The string rule alone is expected to be weak (it rescued 2
        of 15 correct `review` pairs), so NER evidence, then T-130's opt-in judge on the
        `unclear` remainder only (calls budgeted under T-117), decide whether this ships
        a rule or only a report.
      - **What a verdict does:** here, existing RESULTS rows are purged (T-131's
        per-table rule, reversible) only on a confident `other`; `mention` and `unclear`
        keep their rows until a measured rate says otherwise. Upstream applies `about`
        above a threshold **set jointly after the measurement** as `accept`; nothing is
        released on an unmeasured verdict.
      - **Done when:** every selected pair has a verdict and the class seen; the report
        states precision/recall per verdict against the hand labels and the `unclear`
        share; the CSV round-trips (upstream can apply it); purges, if any, are recorded
        and reversible; its columns are added to the T-135 guard list, with a test where
        one is missing (the run fails with the message); an FR is added in the PR that
        ships it; the verdict/`other`
        naming collision is documented in `docs/category-taxonomy.md`.
      - **Depends on:** T-128 (met), T-139 (D5, D6); **T-135** (met, #109: the
        schema guard exists; this task adds its columns to the guard's list,
        `articles.prev_fetch_status` and `discovered_urls`, in `SOURCE_REQUIRED_COLUMNS`). **Blocks:** T-117 (its R and
        N are measured after these pairs are decided). → step 2b.


## T-154

- [ ] **T-154** *(backtest cut, D34; the full plan continues around it)* A provisional output for the
      bodies of the backtest window (published 2021-12-31 or later) by about 2026-11-16: scope from the
      rules, NER for single and multi bodies, summaries (T-148a), company texts (T-148b), sentiment
      over the texts (T-148c), written (owner, 2026-11-06) as chunked files under `data/cut/` in the columns
      approved in T-152, resumable, no schema change; the stored corpus only (it ends 2026-08-12: the bodies the
      backtest team adds upstream are not in the cut); marked provisional, with the
      measured figures (rules' scope agreement on the held-out 200, extractor faithfulness, sentiment
      agreement on the 180) next to it. **Schedule (owner, 2026-11-06):**
      - *11-07:* the baseline (done) and the exports: the label check (200 paragraphs, translation with `--yes`)
        and, from the baseline, the faithfulness file, and `draw_sentiment_gold --pool-only` (scores the pool and shows a short stratum four days before the draw; D34 (d)); T-148a code (the summary run over the window's non-multi
        bodies, scope from the rules) and its start in the evening (about 7 GPU hours overnight; it does not
        depend on the tagging; wasted only on a no-go); the T-153 reply is drafted for the owner.
      - *11-07 to 11-08:* the owner tags the label check first, then the held-out 200.
      - *11-08:* `report_wi18 label-check` applies the rule of T-148b2; the attribution model trains (GPU, after the summaries; **order set by the owner in D35, GPU: the retraining first, about 20 minutes**).
      - *11-09:* the faithfulness texts (190 for the baseline, plus the model's) are exported (one blind file); T-153 is sent
        after the owner's approval, so the backtest team plans before the decision.
      - *11-09 to 11-10:* the owner tags the 30. *(the 180 sentiment texts were drawn, tagged and scored on 2026-11-06, 60.4%, PR #131; the 450 summaries for the retraining are tagged on 2026-11-07, D35.)*
      - *11-12:* decision, from `report_wi18 scope|sentiment|faithfulness`. After a go: NER, step 4 and sentiment over
        the window 11-12 to 11-14, hand-over about 11-16.
      **Checkpoint 11-09 (end of day):** the tagging of the label check and the held-out 200 (400 items) must be
      back, and the faithfulness file exported. If the held-out 200 is not back, the scope figure is reported as
      "not evaluated" and the cut still goes if the other three pass; if the faithfulness marks (the sentiment marks of the 180 are in; the retraining's dates are T-148c3/c4, fallback 2026-11-11) are not
      back by 11-11 evening, the decision on 11-12 is no-go and the backtest team is told (the week's notice). The
      developer reports the count of marked rows per file on 11-09, 11-10 and 11-11 (`report_wi18` refuses a half-marked
      file and names the missing rows).
      The owner's tagging is 770 items in six days (200 + 200 + 190 + 180). **Rules fixed beforehand (owner,
      2026-11-06, also in PLAN D34):** a sentiment figure whose interval straddles its bar ("not decided") still goes,
      with the interval printed; if the model is not ready on 11-09, the 190 texts are exported from the baseline
      alone (`--extractors baseline`) and the decision is on the baseline (the cut uses it for narrative bodies if it
      is at least 80% faithful, otherwise narrative multi bodies are left out of the cut and counted).
      **Go/no-go on 2026-11-12:** the owner's tagging (the held-out 200 for the scope, 200
      items of the label check, the 190 faithfulness texts and the 180 sentiment texts) is done and the extractor
      (or, if the model is late, the baseline) passes D32; otherwise the backtest team is told with a week's
      notice. Not in the cut: the trained scope classifier, the prompt gate and silver labels,
      category, sub-industry, sector snapshots, active learning, the full backfill. → D34.


## T-131

- [ ] **T-131** **→ Work item 18 P6: backfill (2026-10-28).** *(new; owner decision D4, decided 2026-10-22 at the recommended defaults, see T-139)* Plan and execute
      the cleanup of existing results for articles the crawler fix removes:
      (a) **derive the removed ids at run time with a read-only join** of
      RESULTS article ids to SOURCE (`match_class = 'reject'`, quarantined; about
      315,822 articles on the 2026-10-22 count), not from the 2026-10-15 CSV, which
      is stale: the `reject` class has moved since (448 `loose_name` pairs went to
      `review`, never-fetched sitemap rejects were deleted). Upstream's list of known
      false rejects (at least 433 pairs, `data/retag_known_false_rejects.csv` there,
      mostly News Corp `NWS`/`NWSA`) must be kept out of the purge; most of them now
      sit in `review`, which this task does not purge (T-132 decides them);
      (a2) **pre-purge check of the `reject` class's own error rate.** Upstream's hand
      check puts about 2.5% of `reject` as wrong (95% interval 0.4–12.9%), i.e. about
      7,900 (1,300 to 40,700) of 315,822 articles, and it dates from 2026-10-15, before
      448 pairs moved to `review`. Before any purge, draw 400 `reject` articles with
      RESULTS rows uniformly at random, label "is this article about its tagged
      company?" (T-130's judge, validated on upstream's hand labels, with the owner
      hand-checking the disagreements), and compute the wrong share with its 95%
      interval. **Stop/go (confirmed by the owner 2026-10-22, T-139):** go
      if the upper bound is at most 5% (10 wrong of 400 gives 4.5%; 15 gives 6.1% and
      fails); if it fails, purge only the `match_reason` sub-classes whose own
      sample clears the bound, or stop and bring it to the owner. Reversibility
      (the `<table>_removed` archives) is the second safeguard, not a substitute: a
      purge also forces a `sector_summary` regeneration. The 400 judge calls are in
      T-117's budget; (b) **per-table rule — move
      removed ids only, never rename a table:** for each of `article_sentiment`,
      `article_entities`, `article_category`, `article_summary` and the lean
      `articles` copy, in one transaction per table, copy the rows whose
      article id is in the removed list into an archive table
      (`<table>_removed`) and then delete them from the live table (stage
      tables first, lean `articles` last), dry run on a scratch DB copy
      first. A whole-table rename (T-025's pattern) would also remove the
      *retained* articles' results, make every article pending and force a
      full GPU re-run of sentiment/NER/summary that nobody planned, so it is
      not used here; the one whole-table rename in this item is
      `article_category` in T-124. (c) make `sector_summary` regenerate (bump
      `SECTOR_SUMMARY_FORMAT_VERSION` so every closed week is pending again,
      or delete only the affected `(sector, sub_industry, week)` rows) — it
      regenerates only at the next `--summarize` run; (d) the owner decided D4 on
      2026-10-22 at the recommended defaults: keep the `<table>_removed` archives, and
      delete only the affected `sector_summary` rows (no version bump), which (b) and (c)
      apply. Runs **before** T-124 (never together), so no reader sees a half-cleaned store. *(2026-11-05: T-117, T-124 and T-130 are superseded and not done; the judge calls and the `article_category` rename they carried are dropped, and the v1 move of the old tables is T-150, D19.)*
      → step 6b.
      *D4 (decided 2026-10-22):* the recommended defaults apply (keep the
      `<table>_removed` archives; delete only the affected `sector_summary` rows).
      T-131's execution may
      ship as its own PR; it stays in this work item because T-124 depends on
      its removed-id list and ordering.


## T-146b

      - [ ] **T-146b** *(export/check built 2026-10-29 in PR #117, aligned with `docs/labelling-spec-v2.md` (T-146a, PR #118) on 2026-10-29: round-up mentions, no sector-level `sub_industry`, `unrelated` only as a scope; a test checks that the spec names every scope, label and column; `news_nlp.retag`, `scripts/retag_gold_2026_10_29.py export|check`, file `data/category_gold_v2_review.csv`, 720 rows with v1 label/note/`v1_scope_derived` and empty `scope`/`companies`/`company_labels`/`mentions`/`sub_industry`/`notes`; single = exactly one analysed company (others are mentions, D23); multi = every company labelled (owner, 2026-10-29); tickers and sub-industries are checked against SOURCE; `check` verifies all 720 ids present once; the LLM label stays in the key. Added 2026-10-30: `normalise` turns names and nicknames the owner typed in `companies` / `mentions` / `company_labels` into tickers (`news_nlp.company_names`, `data/category_gold_v2_review_tickers.csv`; unknown or ambiguous texts are listed and left as typed, never guessed), and `check --file` validates that copy (it refuses to overwrite a copy edited after the review file, `--force` overrides; `normalise --self-check` verifies every universe name resolves to its own ticker; `guide` rewrites only the guide CSVs, also after tagging has started; a returned file saved with another delimiter or code page is refused with a message, by `export`, `normalise` and `check`). Changed 2026-10-30 (owner's request, tagging started): no `companies` column (the analysed companies are the labelled ones in `company_labels`) and `,` instead of `;` in `company_labels` / `mentions` (a comma inside a name like `Booking Holdings, Inc.` is kept); `migrate --file NAME` converts a file of the first format into a new file (dated name, never over an earlier copy); `sub_industry` takes the 8-digit GICS code (table of the Wikipedia GICS page, `news_nlp/gics_codes.py`) or the name, required for sector, optional on every other scope (owner call 8); a missing review file names the similar files; a company outside the 509 of the universe (Macy's, Twitter) is kept as typed, counts as analysed, and `check` lists those names so a typo shows (owner call 9). Added 2026-10-30 (owner's request): every file to tag carries `spanish_title` / `spanish_text` after `text` (`news_nlp.translate`, `translate` and `export` commands, cache `data/gold_translations_es.csv`, paid DeepSeek translation approved for this, about 1 USD per 720 bodies, logged in `docs/evaluation.md`; `export` / `translate` print the count and a token estimate and buy nothing without `--yes` (at most `--max-new`, default 800); a failed article or a stopped run (bad key or balance, 5 failures in a row) writes nothing and a re-run pays only for the missing ones; the cache key holds the model and prompt tag (the 720 legacy rows were stamped once with `translate --adopt-legacy --yes`, no cost; the cache file is replaced atomically) and every paid call is counted, retries included; the same applies to the held-out gold T-146e, about 300 bodies, and to any later tagging export). Open: owner: close `data/category_gold_v2_guide.csv` in Excel and run `guide` (the copy on disk predates the column rows), then tag (T-146a is merged); the re-measure of the rules on gold v2 runs once the file is tagged.)* Re-tag export of the 720 with the tags v1 shown as guidance, and the importer
        for the owner's new columns (protected like `gold_review`: refuses to overwrite a tagged
        file); gold v2; re-measure the rules' 0.70 on it (`gold_scope` maps v1 tags, v2 uses new
        columns).


## T-146c

      - [ ] **T-146c** Revised LLM labelling prompt and gate: iterate on a fixed stratified 240 of
        gold v2 (at most 5, about 240 calls each), then the full 720 (480 new calls) at >= 90% scope agreement. The 240 is stratified on the
        scope the v1 notes imply (`v1_scope_derived`), because gold v2 scopes do not exist before the owner tags; it is not a random
        sample, since the owner's already tagged rows come first. With a few items per rare scope it gates the overall figure only
        (per-scope recall is judged on the held-out gold). Silver D15 labels and sub-industries are bought in P2 only if
        a second gate on the 240 (per-company category and sub-industry agreement with gold v2,
        threshold >= 80%, owner answer Q8) passes; otherwise silver is scope-only and P4/P5 labels
        come from the owner's gold in later rounds. Budget: silver <= 6,000 - 240k - 480 for k
        iterations (k = 5 gives 4,320); judge repair re-calls count against the 6,000.
        - *The fixed 240 (owner answer 2026-11-04).* `scripts/draw_gate240_2026_11_04.py` draws it once, quotas sector 9,
          market_economy 9, unrelated 31, investing_general 40, multi 60, single 91 (`retag.GATE_QUOTAS`); it stops, writing
          nothing, if the draw is not 240 or a stratum holds more tagged rows than its quota. The set is
          `data/gate240_ids.csv` (local, git-ignored); the tracked gold files are the train set `gold/category_gold_v2.csv` (240 + 44 sector) and the held-out test set `gold/category_gold_v2_test.csv` (200, tagged by the owner and migrated 2026-11-07, PR #142), both added by #123, and every later scope feature reads those two (owner call 2026-11-05); a third, `gold/company_segments_train.csv` (T-148b2), holds automatic attribution labels, not tags; the gold is built from the gate file, `merge` is only a copy into the 720-row review file. The owner tags it in `data/gate240_to_tag.csv` (the review file's columns,
          Spanish aid after `text`, existing tags included; never overwritten); `merge` copies its tags into the review
          file as a new `_merged_` file, and the newest one is the current review file. The other 480 of the 720 can wait until
          the 240 passes; the full 720 is needed only for the confirmation run and the rules re-measure. Workflow, conflicts and `merge --gate-wins`: `docs/labelling-spec-v2.md`, "The fixed 240 of the prompt gate".
        - *2026-11-05 owner calls after tagging the 240: company labels allowed on sector/market_economy/investing_general bodies, several labels per company, `sub_industry` as GICS codes of any level and several per article (the gate compares by shared prefix); see `docs/labelling-spec-v2.md` calls 11-13. Because the 240 hold only 16 sector bodies, 44 more sector-like bodies are drawn by free rules into `data/sector_extra_to_tag.csv` (`scripts/draw_sector_extra_2026_11_05.py`; the owner tags scope, sub_industry and notes) for the sub-industry gate, which compares codes by shared prefix. 2026-11-05: both sessions (240 + 44, all valid) are in the tracked `gold/category_gold_v2.csv`, tags only (the article text stays out of git: news text, public repo), built by `scripts/build_gold_2026_11_05.py`.*


## T-155

- [ ] **T-155** *(gold sets, `docs/gold_datasets.md`, 2026-11-06)* The builders that move the label check (**done 2026-11-06**: `scripts/build_segment_check_gold_2026_11_06.py`, 200 rows in `gold/segment_check_gold.csv`; **the extractor sample done 2026-11-06**: `scripts/build_faithfulness_gold_2026_11_06.py`, 190 rows in `gold/faithfulness_gold.csv` and `gold/faithfulness_coverage.csv`; the builder protects the tracked marks and coverage from silent change (`--overwrite-marks` to accept), ties the coverage to the draw (`faithfulness_bodies.csv` is required), writes the two files together with a put-back on failure, and `report_wi18 faithfulness --gold` reproduces the D32 figures from them; **the sentiment set done 2026-11-06**: `scripts/build_sentiment_gold_2026_11_06.py`, 180 rows in `gold/sentiment_gold.csv`, `report_wi18 sentiment --gold` reproduces the figures: agreement 60.4% [52.8, 67.6], fails the D32 bar; the gold keeps `pub_date`, protects the weights, and the report stops on a sample that is not the one drawn; **the held-out scope set done 2026-11-07** (PR #142: `build_gold_2026_11_05.py --set test`, 200 rows in `gold/category_gold_v2_test.csv` with the draw's `stratum` and `frame`; `report_wi18 scope --gold --out gold` reproduces the figures from the tracked file and tests pin the counts, the shape, the GICS codes offline, the weights (they sum to 1,000) and a hash per row over every column, `tests/news_nlp/heldout_gold_row_hashes.csv`; the builder refuses a different set of articles and, since the review of PR #142, a change to a tag of the committed file (`git show HEAD:`, the file on disk without git) without `--overwrite-marks`; `--write-hashes` renews the hash file); what is left of T-155 is the typed reader per gold file in `news_nlp.retag`, and a decision on where the only copy of the 180 texts, `data/sentiment_gold_ids.csv`, is kept (nothing enforces the backup), the extractor sample and the sentiment set from
  `data/` to `gold/`, as `build_gold_2026_11_05.py` does for the scope sets: label check
  (`gold/segment_check_gold.csv`), extractor sample (`gold/faithfulness_gold.csv`, keyed by article, ticker and text hash, not by
  `review_order`) and sentiment (`gold/sentiment_gold.csv`, with the draw's `stratum`, `extractor`, `predicted` and `population`).
  `text_hash` (`retag.text_hash`), for sentiment `revision` and for the extractor sample `revisions` (`retag.code_revision`) are written by the draws into their ids files and copied by the builders, never recomputed. Each builder refuses unless every row is marked and valid, writes marks and ids only (no news text), and ships a typed reader per gold file in `news_nlp.retag` (today only the column tuples exist); then `report_wi18` reads the
  `gold/` file, so the figures of D27 and D32 no longer need `data/`. `scripts/scaffold_gold_sets_2026_11_06.py` writes the empty
  files (tracked in `gold/`, local in `data/`); the draw scripts fill a header-only working file. Do it when the owner has tagged
  each set; the order to tag is on the page.


## T-148b3

        - [ ] **T-148b3** *(code and a trained candidate written 2026-11-07 (the script and the model directory are named 2026_11_08, the planned date of the run), PR for review: `src/news_nlp/attribution_model.py`, `scripts/train_attribution_2026_11_08.py`, `CompanyResolver.mention_spans`, `draw_faithfulness --attribution-model`; **not published, not pinned**; left: the owner's OK to publish, the pin in `config/pipeline_models.json` under `attribution` with its revision, the `ExperimentSpec` stage, the model's texts added to the faithfulness file (`--add-extractor model --attribution-model ...`, which pays for the Spanish translation of the new texts: needs `--yes`) and the owner's reading, which decides)* The trained attribution model: a new pinned checkpoint in
          `config/pipeline_models.json` with its revision, selection justified against the baseline
          (constitution AI #1, #7, #12), the `ExperimentSpec` stage list extended deliberately.
          **Trains on `lead` rows only (D27, label check 2026-11-06)**: the training set is built with `kind == lead`, the rows used are counted and printed, and the run refuses to start unless `gold/segment_check_gold.csv` gives `gold_metrics.train_on_lead_only` true (so the rule is applied, not remembered).
          Bodies it cannot attribute are flagged `unextracted` and counted, never given a default text.
          A one-paragraph body is split into sentences and each sentence is attributed by the same model.
          **Open question, proposed and not decided (owner), 2026-11-08:** the model has not learned to answer "none of these". It was trained only on sentences that belong to one of the body's companies, so on the 30 faithfulness bodies a sentence about a company outside the universe goes to the nearest one: of 5 sentences given to a company by the 0.5 rule, none was right (developer's reading, not the owner's), and their scores were 0.999, 0.988, 0.997, 0.893 and 0.505; 276 sentences were given by the rule, 159 of them at 0.8 or more and 38 between 0.5 and 0.6 (`train_attribution_2026_11_08.py --abstention DIR`). Options, none built: (a) train with negatives that belong to nobody in the list (the introduction before the first company, `continuation` sentences the label check found wrong, sentences that name a company outside the universe): a retrain, so the GPU order is the owner's; (b) keep the model and measure again after the stricter cut and footer handling of #143 and #144; (c) raise the threshold: **not supported**, the wrong sentences are confident, not borderline. The decision comes after the owner reads the texts; `JPMorgan` alone is not read by the resolver (T-148b7).


## T-148c3

        - [ ] **T-148c3** Trainer of D35's recipe (base, replay of the earlier sentences, search on the validation slice, 3 seeds, a final run). Acceptance: the replay leaves out the sentences of gold, held-out and drawn-for-tagging bodies (by body hash) and prints how many; one run is timed first and the grid is reduced if 28 runs do not fit (D35); the tie rule of D35 picks the settings. **The trainer does not score the 180-text gold (the gold is scored once, by the scorer, as the `final` run of the published revision, T-148c4).** A batch never mixes summaries with sentences, and each epoch shuffles the batches of both kinds together (D35). The training settings are D35's (the earlier fine-tune's batch 16, weight decay 0.01, fp16, linear decay, no class weights, the last epoch's model), written in the trainer's config and printed at the start of each run. The final run trains on the 360 train rows plus the replay, with the fourth seed fixed in the code, and the report gives the spread over the three seeds (D35); its own validation agreement is printed next to the three-seed mean and spread, and the run stops before publishing if it is below the three-seed mean by more than 2 standard deviations or by more than 3 texts of 90, whichever is larger (not "below the lowest": that blocks a healthy run 1 time in 4). **Proposed 2026-11-08 (round 4 of PR #145; the owner accepts or not): the validation agreement, and the gold's, are also reported split by companies seen in train and unseen (77% of the validation rows with a company have one seen in train; `docs/gold_datasets.md` set 6).** The trainer's validation agreement is reproduced by scoring the same validation texts through the scorer's reading (T-148c2: chunks of 510 tokens, averaged), within **1 text of the 90 (the two figures differ by at most 1.1 points)**, both printed, and the run stops if it is not (two inference paths, and a label order that is permuted but made of valid class names, would otherwise go unnoticed). **By 2026-11-08**; first run when the 450 are marked; timed then. **Timed 2026-11-07, GPU idle** (two epochs of one run at batch 8: 12.7 s and 8.6 s; summaries median 73 tokens, longest 108; the full 28 runs **about 15 to 30 minutes**, an order of magnitude, not a figure to schedule against): the full grid fits whenever the tags arrive, so the smaller grid is not expected; the date that limits is T-148c4's 2026-11-09, so the tags are needed by the morning of 2026-11-09 at the latest (developer's estimate: trainer plus publish plus final scoring take hours, not days). The trainer times its own first run again with the GPU idle and keeps the budget rule.


## T-148a

      - [ ] **T-148a** Summaries: the pinned local summariser (`c_summary`) on every non-multi body
        (single, sector, market / economy, `investing_general`), with the input the pipeline
        already builds minus the NLP features (title and body only: sentiment now follows the summary).
        For the cut (T-154) it writes chunks under `data/cut/summaries/` and resumes from the last chunk
        (**code written 2026-11-06**: `scripts/cut_summaries_2026_11_07.py`, chunks of 250 bodies, stop with Ctrl-C and run the same command to resume, `--status`; **launch: `uv run scripts/cut_summaries_2026_11_07.py --require-cuda` (PR #132: the first line names the device, a tqdm bar counts bodies and moves per saved chunk; the device is a condition of the manifest, so a first run on the CPU would fix the cut to it, hence the flag)**; smoke-run on 58 bodies; full run about 29,113 unique bodies, about 70% summarised, 5 to 6 GPU hours; it starts the evening of 2026-11-07). **Two facts from the real window, 2026-11-06:** the rules read the stored NER `ORG` spans, and 30,906 of the window's 36,063 articles have them (none for 2026-09 and 2026-10, about half for 2026-08), so for the newest bodies the scope is read from the text alone, unlike the held-out sizing. **Measured (`scripts/cut_spans_effect_2026_11_07.py`, 500 bodies published since 2026-03 that have spans, scoped with and without them): the rule scope is the same for 500 of 500 (upper 95% bound of a change about 0.6%) although the company mentions change in 21 of them, and so is the summarise-or-not decision; the missing spans do not change the cut's scope.** Each chunk row records `spans`, and `--status` counts the bodies without. Bodies the rules call `multi`, `roundup` or `no_company` get no summary, and the rules never predict `investing_general` (D25 summarises it): `--status` after the first chunks shows the share, to decide before the long run. **Order against NER on the window (open, owner to decide before the long run):** (a) run NER on the window first, then this job, so every row has its spans; or (b) run now and let T-154 recompute scope and mentions from the final spans, using the chunks only for the summaries (`--accept-new-spans` keeps the chunks but their `rule_scope` stays the one read without the new spans; the mentions differ in about 4% of bodies). Until decided, T-154 must not read `rule_scope` from the chunks. Checks that
        a body longer than the model's window is chunked and reduced, as `hierarchical_summarize_batch` does. → D25, D26.


## T-148b8

- [ ] **T-148b8** *(manager's review of #144, 2026-11-08; proposed, not decided)* Two things about the evidence under `units3` as the candidate. (a) **The denominator:** the 91.6% leaves out the 36 companies the 30 bodies name that `units2`/`units3` are not asked about (32 the guard rejects, 5 footer-only, minus 1); on every company named the bar is 74.2% [66.0, 81.5] for `units3`, 68.4% for `units2`, 83.2% for `units`. `draw_faithfulness --exempt` writes `faithfulness_exempt.json` and `faithfulness_exempt_to_read.csv` (the 37 rows, the line that names each, with `spanish_title`/`spanish_text`: translated 2026-11-08 with the owner's approval, 37 paid calls, 8,848 prompt and 4,910 completion tokens); the owner marked `real_company` (2026-11-08): 23 are not companies, 14 are; `report_wi18 faithfulness` gives the bar on real companies, every extractor on the same 167 (baseline 68.9%, model 86.2%, `units` 94.0% [87.0, 99.2] pass, `units2` 77.8%, `units3` 84.4% [76.4, 93.3] not decided). (b) **A confirmation:** a fresh draw of narrative multi bodies not used so far, read for `units`, `units3`, `model` and the baseline, since the guard and `units3` were designed on the failures of these 30; the draw and the translation are the owner's to approve. (c) the header audit (`scripts/audit_headers_2026_11_08.py`, 4,000 bodies, seed 20261108): 988 lines were taken for a header; 77 were datelines or labels in capitals ("DETROIT - Ford ...", "REFINING - ..."), 17 of the 86 lines given to nobody that name a company after the dash: fixed (a head in capitals that names no company is not a header; no text of the 30 bodies changes because of it); a second rule (round 2): a dash inside a sentence (the text after it does not start with a capital, a digit or a quote: "Volkswagen - synonymous with the classic camper - has spawned ...", "Today, GM - like the industry itself - is challenged ...") is not a header: 24 of the 911 lines, all of them parentheticals, and no text of the 30 bodies changes; 67 lines then remain given to nobody and name a universe company after the dash, 7 of them because the part before the dash is a company named by a short form ("McDonald's", "General Electric", "Dell") that the guard rejects: left as it is, the guard is the owner's measured design (T-148b7); 15 headers are missed for a full stop in the first part ("Amazon.com - ...", "TJX Cos. - ...") and read as plain lines, as `units` reads them.


## T-148b6

        - [ ] **T-148b6** *(2026-11-07, after the owner's reading of the 33 `units2` texts)* The owner read them: 1 failing of 33, so of the texts `units2` wrote 130 of 135 are faithful (96.3% [91.6, 98.4]); on the bar (D32) it is 84.4% [72.5, 97.2], *not decided*, because 19 companies named have no text, and against `units` it is -11.1% [-23.5, 0.0], *not decided*. The 19 are real companies named in full, none a false company: a line "NAMES - text" went whole to the header names, so a company the text itself names (APA and Devon in a line headed by Occidental, Berkshire in a line headed by Goldman) got nothing; 3 were checked piece by piece, the others have the same shape in their context lines, and Apple ("Apple supplier" in an Amkor line) is rightly left out. **Built** (in the same PR, own reading so that the `units2` result stays as read): `reading="units3"` = `units2`, and a piece of a line headed by universe companies also belongs to the universe companies it names itself (`attribution_pieces(..., named_after=True)`); a line headed by somebody else still belongs to nobody. On the 30 bodies: 148 texts, 146 of them already marked (7 of them not faithful), 2 new to read (Goldman Sachs 247310, Morgan Stanley 477021), omissions 19 -> 6 (CSCO and NSC of 247310, KR, NKE and SPG of 321565, AAPL of 477021; the 5 first are not yet looked at, AAPL is rightly left out as before). Worst case on the 154 companies: 139 of 154 faithful (90.3%) before the 2 are read. **Done (owner, 2026-11-07):** the 2 rows were added (approved paid translation, 2 calls) and read, both faithful. Result on the 30 bodies: `units3` 91.6% [84.7, 97.8] on the bar, **pass**, 141 faithful of 148 texts + 6 companies left without one (95.3% [90.6, 97.7] of the texts it wrote); `units2` 84.4% [72.5, 97.2] and `units` 85.4% [76.5, 92.2], both not decided; `units3` minus `units2` +7.1% [0.0, 16.7] and `units3` minus `units`  -3.9% [-7.3, 0.0], both not decided (the two cover different companies). **The owner chose `units3` as the candidate for the cut** (PLAN D34 (e)). Known limit, same family as T-148b7: a header line can carry a second company's sentence (Goldman's text also has one from Oracle, marked faithful). The 5 omissions not yet looked at (CSCO, NSC, KR, NKE, SPG) are left to T-148b7 unless the owner asks.


## T-148b2

        - [ ] **T-148b2** Training set for the attribution model from those bodies (company name
          masked, previous sentence as context, all years). **Set built 2026-11-05** (`scripts/build_company_segments_2026_11_05.py`, `gold/company_segments_train.csv`, ids and positions only): 2,834 bodies, 33,624 labelled segments (21,682 `lead`, 11,942 `continuation`; ambiguous paragraphs have no row), 1 body in 10 as `val` (205 bodies before 2022, 65 from 2022 on: report the check on the 2022+ ones apart); the 33 structured bodies that share text with a tagged or held-out article are left out; the script stops and names the local git-ignored files it needs, and `tests/news_nlp/test_company_texts.py` pins the parser constants and the counts. Built 2026-11-06: the mention-and-recency baseline (`news_nlp.attribution_baseline`), the owner's label check (100 `lead` + 100 `continuation` paragraphs, `scripts/draw_segment_check_2026_11_05.py`, Spanish columns after `text` and a Spanish version of the previous paragraph), the faithfulness check (`scripts/draw_faithfulness_2026_11_06.py`: 30 narrative multi bodies drawn once, one blind file with the baseline's and the model's texts, shuffled), and the readers and figures of every owner file (`news_nlp.gold_metrics`, `scripts/report_wi18_2026_11_06.py scope|label-check|sentiment|faithfulness`, D32 bootstrap, refusing a half-marked file). **Rule fixed before the label check is read (owner, 2026-11-06):** if at most 80% of the `continuation` paragraphs are right they are dropped from training (train on `lead` only); the `lead` labels are kept in any case; the precision is recorded in PLAN D27. **Label check read 2026-11-06: `lead` 88.0% (80.2 to 93.0), `continuation` 38.0% (29.1 to 47.8), so the model trains on `lead` only; the 12 wrong `lead` labels are 5 multi-company paragraphs, 3 look-alike entities and 4 non-company text (a junk-paragraph filter is cheaper than fixing labels by hand, not done).** Still to do: the faithfulness file (190 texts, not the 30 assumed), then the model (T-148b3), `lead` only. Acceptance (D32): at least 80% of the companies named in the 30 with a faithful text (a company left without a text counts as not faithful), and better than the baseline on the same 30.


## T-148c2b

        - [ ] **T-148c2b** Scorer hardening, its own small PR, **today or 2026-11-08, it does not wait for the owner's tags** and must be merged before the final scoring (a rule is applied, not remembered; as T-148b3's `train_on_lead_only`). Done when each of these holds, with its test:
          - a Hub model needs a 40-hexadecimal `--revision` (today any non-empty text passes, `main` included); a local directory is unchanged;
          - `--redo-role` requires a `--redo-reason`, written to the log row (new column `redo_reason` in `retag.SENTIMENT_SCORING_LOG_COLUMNS`);
          - the committed `gold/sentiment_gold_scoring_log.csv` gets the new columns with empty values in its baseline row (every other value of that row unchanged), and the committed-gold-files test moves in the same PR;
          - the report shows the agreement for texts over 128 tokens apart from the rest, as it does for 2022 on and before (D35);
          - the `final` run's report gives the paired gain over the **last** `baseline` row (same keys; `gold_metrics.paired_difference` clustered by body; the verdict of `gain_verdict`); the baseline's predictions are the stored `predicted` column, and the gain is refused if its agreement differs from that row's logged one; tests: a hand-made pair, a gain whose interval spans 0 (`not decided`), keys that do not match (refused);
          - the row logs `gain_verdict`, `gain_negative_precision_verdict` and `gain_negative_recall_verdict` (new columns; empty in the baseline row);
          - `test_the_pinned_sentiment_model_is_the_one_scored_as_final` is changed: the pin passes if it is the last `final` row's model and revision and that row's `gain_verdict` is `better`, or if it is the last `baseline` row's (a rejected candidate); cases: candidate pinned and better (passes), candidate pinned and not better (fails), candidate rejected and baseline pinned (passes), a pin that is neither (fails).


## T-148b5

        - [ ] **T-148b5** *(owner's decisions 2026-11-07, PLAN D34 (e); built the same day)* The reading of the 402 rows left two causes, both before the model: first-word names taken for companies (21 of the 27 texts of `units` not faithful: "Bloom", "Union", "Home", "Equity", "Baker", "Boston"; such texts are 34% faithful) and lines headed by a list of names and a dash whose text names other companies. **Built** (stacked on T-148b4): `attribution_units.GuardedResolver` (a short form of a company's name counts only when its full name or a ticker form is somewhere in the body; for the attribution only, `CompanyResolver.mention_spans(text, aliases=False)` is the only change to the resolver and its default keeps the scope rules and the held-out measurements as they were); `attribution_pieces(..., headers=True)` and `header_owners` (a line that opens with at most eight capitalised names and a dash belongs to the universe companies among them, none if the names are of another company: "Canadian Pacific - ... Norfolk Southern CEO ..." is about Canadian Pacific); the extractor `units2` (`reading="units2"` of `model_company_texts`), whose companies named exclude the ones only a common word reached. No reviewed alias list (owner's decision): the companies reached only by a short name are lost, 11 of 32 of which were right. On the 30 bodies: 135 texts of `units2` (185 of `units`), 102 of them the same as texts already marked (98 faithful, 96%), 33 new to read (review order 403 to 435). The two extractors are not on the same set of companies: `units2` asks fewer, so the report gives each one's own share and the paired gain on the companies both wrote. **Open:** the owner reads the 33; then `report_wi18 faithfulness`; *proposed, not decided:* `units2` is the candidate for the cut if its share is at or above `units`' and its interval is not worse, else `units`.


## T-148c4

        - [ ] **T-148c4** Publish the fine-tuned model (Hub write token) and pin it in `config/pipeline_models.json`, its own PR. The candidate is published to the private Hub repo `gamug/FinBERT-financial-news-summaries` (D35: the owner accepted the name, 2026-11-07) and made public only in this PR; the `final` row and the pin name the commit SHA, not a branch or tag (D35). A second `final` scoring only for a defect in the run, never for the figure (D35). **The scorer's checks (SHA, redo reason, the over-128 split, the paired gain the pin rule reads) are T-148c2b, merged before this task's final scoring; the pin PR applies D35's rule to that gain (`better` pins; `not better`, `not decided` and `undefined` keep the baseline, owner, 2026-11-07; and no pin if the paired gain of negative precision or recall is `not better`).** The final run's own validation agreement is checked before publishing (D35). **By 2026-11-09**; the final scoring of the gold decides the fallback, with room before **2026-11-11**. The pin PR carries the `final` row: the model is scored as the `final` run **as the published Hub id and revision** (not a local checkpoint) and `config/pipeline_models.json` names that same model and revision when the candidate is adopted, or still the baseline's when it is not (`test_the_pinned_sentiment_model_is_the_one_scored_as_final`, changed in T-148c2b, enforces both). If the baseline stays, there is no pin PR for the candidate: its repo stays private and its `final` row stays in the log (D35); the owner decides later. The owner tags all 450 on 2026-11-07, no stopping rule.


## T-148c

      - [ ] **T-148c** Sentiment over the texts of T-148a and T-148b (D26): after T-148b exists,
        draw the 180-text sentiment gold (60 per class, `scripts/draw_sentiment_gold_2026_11_05.py`: 120
        from summaries and structured multi texts, 60 from the texts the extractor writes for narrative
        multi bodies, `--extractor baseline` today and `model` once T-148b3 exists; the pool sizes and what
        is left out are in `data/sentiment_gold_pool.json`), the owner tags the price impact,
        the current model is measured on it, and fine-tuned on summaries only if it misses the D32 bar (overall agreement >= 80%, negative precision and recall >= 70%). **Measured 2026-11-06 (PR #131): overall agreement 60.4% [52.8, 67.6] fails the 80% bar, so the fine-tuning condition is met; the owner chose fine-tuning on summaries (2026-11-07, PLAN D35): 450 summaries to tag (`scripts/draw_sentiment_train_2026_11_07.py`, 150 per predicted class, balanced blocks, a validation slice of 90, never a gold or held-out body); the 180-text gold is scored once per run that could ship, every run listed; fallback if it still fails on 2026-11-11: the 2026-11-12 cut goes with sentiment marked as measured.** Covers T-127's question on real output.


## T-147

- [ ] **T-147** *(P3)* Triage stage on unique bodies: body-hash grouping (T-132), the scope first (NER then runs on
      `single` and `multi` bodies only, D25), from a local classifier (a fine-tuned small encoder reading the text, with the
      `news_nlp.triage` rule features as inputs) trained on silver and gold, with a calibrated
      confidence; **accepted at >= 90% corpus-weighted agreement with the held-out gold on all bodies and
      recall >= 80% for each of the six scopes that has >= 40 gold bodies, the others reported with their interval (point estimates with intervals; D20 as amended by D33)**, the
      complete metric set per scope (AI #12). Each round exports a stratified sample of about 300
      low-confidence bodies for the owner's tagging, fed back into training (D21), until the
      target holds; `unrelated` flag-and-skip; tables proposed to the owner in the PR; its own
      FR. **Model governance (constitution AI #1, #12, Work item 8):** the encoder is a new pinned
      checkpoint in `config/pipeline_models.json` with its revision, its selection is justified
      against alternatives like every other model, the `ExperimentSpec` stage list is extended
      deliberately (a snapshot test), and the complete metric set is reported per scope. → P3.


## T-157

- [ ] **T-157** *(Windows, T-148a, 2026-11-07)* The summaries cut's run lock probed a pid with `os.kill(pid, 0)`, which
  terminates on Windows (a resume crashed with `WinError 11`). PR #134. See `docs/db-topology.md`.
  - Done: the probe asks `OpenProcess` + `WaitForSingleObject` and answers dead only when the OS says so; the lock holds the
    pid and the process start time (a reused pid is taken over); Ctrl-Break stops like Ctrl-C; a failed `os.link` names its
    cause (NTFS only when the error says no hard links); the `win32`-only test in `tests/news_nlp/test_cut_summaries.py`.
  - Decision: no Windows CI job (it would need the torch wheel, which that test file imports); the Windows branch is
    verified when the cut is resumed on the Windows machine.
  - Closes when: the owner has resumed the cut once on Windows with a leftover lock and got no refusal.
  - Open: the takeover of a stale lock is not atomic (two starts at the same instant can both run: duplicated work, chunk
    files are atomic); a lock written by this version is read as stale by an older script (do not share an output folder
    across versions); closing the console window cannot be caught (the next start takes the lock over).

## Work item 15 preamble and dependency table


**Re-scoped 2026-10-28 into Work item 18** (T-143): see the per-task markers below.

`other` is 195,993 of 459,112 articles (42.7%); its precision of 0.474 is a quoted
figure from a different corpus (see T-123). A read-only investigation (2026-10-14)
suspected it mixes "not about the ticker's company" (ticker symbols that are English
words hold 39.0% of the corpus, plus companies cut to a generic first word) with "about
the company, no dimension clears the 0.6 bar". A population check on 2026-10-22,
stratified by source and period (`docs/evaluation.md`), found wrong tagging explains at
most about 11% of `other` (rejected articles are 68.8% of the corpus and 70.8% of
`other`; `other` is 37.9% of confirmed articles and 43.9% of rejected ones). The main
lever is therefore calibration, which does not wait for the relevance work. Plan:
calibrate per-slug thresholds and a margin rule offline from stored scores (T-121(a) after
T-133's partition, then T-121(b)'s judged sample once the owner approves the spend); clean up the results the tagging fix removed (T-131, after a
pre-purge check); decide the doubtful `review` pairs (T-132); build a relevance gate only
if what remains justifies one (T-117's rule); then re-evaluate and re-classify (T-123,
T-124, by a re-label unless T-140 says otherwise).
→ `PLAN.md` Work item 15, `SPEC.md` §13 item 19.

Tasks are listed in execution order within Work item 15, not ID order (IDs are
stable; T-130 to T-134, T-139 to T-142 were added after the rest; T-135 to T-138
belong to Work items 16 (closed, `CHANGELOG.md`) and 17, which is why the numbers skip;
a letter suffix such as `T-138b` is the close-out of its work item, unlike
T-121(a)/(b), which are parts of one task). The table below and `PLAN.md`'s Sequencing
section are the authority for ordering.

Dependencies at a glance:

| Task | Depends on | Blocks | Can start now? |
|---|---|---|---|
| T-116 audit | none | none | yes |
| T-130 relevance judge | none | T-132 (remainder), T-117 | yes |
| T-128 upstream hand-off | `portfolio-data-mining` T-055–T-058 | T-132, T-117 onward | **done 2026-10-22** |
| T-133 clean-subset re-read | none (read-only SOURCE attach, existing) | T-123's baseline, T-117 | **done 2026-10-22** |
| T-139 decisions D5–D9 | the owner | T-132 (D5, D6), T-117 (D7), WI16 (D8), WI17 (D9) | **decided 2026-10-22, defaults** |
| T-134 date window (D7) | T-139 | T-117 sampling, T-124 size | **decided 2026-10-22, default** |
| T-132 quarantined pairs | T-128, T-139 (D5, D6), T-135 (schema guard, done in #109, `CHANGELOG.md`) | T-117 | **yes** |
| T-137 multi-company sizing (WI17) | T-132 (grouping, ORG-to-ticker), T-139 (D9) | T-138 | no |
| T-138 multi-company build | T-137 result, the owner | T-138b | no (conditional) |
| T-138b WI17 close-out | T-137 (T-138 if built) | none | no |
| T-117 measure R and N | T-128, T-130, T-132 | T-119, T-120 (or their cancellation) | no |
| T-119, T-120 gate, label | T-117 sign-off | T-123 | no |
| T-121(a) offline re-simulation on stored verdicts | T-133 (partition of the stored verdicts; done 2026-10-22) | T-121(b) design check | **done 2026-10-23** |
| T-141 calibration judge + sampler | none (code, hermetic) | T-121(b) | **done 2026-10-23** |
| T-142 D10 spend for T-121(b) | the owner | T-121(b) | **decided 2026-10-22, cap 3,800** |
| T-121(b) judged calibration sample | T-121(a), T-141, T-142, LLM credentials | T-122 | no; population = confirmed articles (the gated-in one only if a gate is built) |
| T-122 ship the rule | T-121 | T-123 | no |
| T-123 re-evaluate | T-121, T-122 (T-119, T-120 if built) | T-124 | no |
| T-131 cleanup (D4) | T-128, T-132 (for `review`), D4 (decided 2026-10-22, defaults) | T-124 | no |
| T-140 D2b mechanism (re-label vs GPU re-inference) | the owner | T-124 | **decided 2026-10-22, re-label from stored scores** |
| T-124 re-classification | T-123 approved, T-131, T-140 | T-129 | no |
| T-125, T-126 contingent | T-123 result | none | no |
| T-127 sentiment impact | none (separate PR) | none | yes |
| T-129 close-out | all of the above | none | no |

## Status paragraph on Work item 15 (replaced 2026-11-08)

**Work item 15's own plan (filed 2026-10-14) is superseded by Work item 18**: T-116, T-117, T-119, T-120, T-122–T-126 and T-130 each say so in place and none is on the roadmap (T-118, T-139 and T-140 are closed decision records, T-121 closed negative). What carries over is the corrected corpus (T-128, done 2026-10-22), the results cleanup (T-131, now the backfill of P6), the alias table (T-132, shrunk) and the close-out (T-129). Work item 16 (schema guard, T-135–T-136) is closed in `CHANGELOG.md`; Work item 17 (multi-company articles) is re-scoped into Work item 18, its sizing absorbed by T-144 and its design superseded by D27–D28.

## Work item 17 preamble

35.9% of the `ok` articles belong to a URL that has pairs for more than one ticker
(a floor); every stage keys on the pair. Size it before building anything. →
`PLAN.md` Work item 17. Task order: T-137, then T-138 only if D9 and the sizing say so, then T-138b (close-out).

## T-160 as merged in #152 (2026-11-09), full text before it was condensed

- [ ] **T-160** *(D36, from the count of T-159; the script is built, not yet run on the whole corpus)* **News per company, and what the preserved corpus would add.** Free and CPU only, no model beyond the rules. **Deliverable (owner, 2026-11-09): a CSV for the owner's partner, one row per company of the whole S&P universe, over every date of the corpus** (not the window first). `scripts/company_news_counts_2026_11_09.py` reads the properly fetched corpus (`ok` and `quarantined`, a title and a body: 309,882 unique bodies in 472,881 rows on 2026-11-09) with the rules of `size_triage_2026_10_28.py` and the name resolver, and writes `company_news_counts_<date>.csv` (no article text, UTF-8 with a byte-order mark for Excel; columns `ticker`, `company`, `also_covers`, `gics_sector`, then for each of three corpora, `ok`, `quarantined_only` and `all`, the unique bodies that **name** the company, that have it as a **subject** (named twice or in the title) and that have it as the **only** subject) and a README (same date in its name; send both) that says the date, the corpus of each column, the totals and the limits (a floor: no alias table; name collisions; the tag of a quarantined pair is not used). It also gives the scope of the bodies (single, multi, sector, market, no company, round-up). It decides what the backfill after the delivery would process (D19, T-150) and what to tell the backtest team (T-153). **Time, not measured:** the size script took about 50 minutes for the 81,961 `ok` bodies, so the 309,882 would take about 3 to 4 hours; `--limit N` prints the projection from its own speed. Run by the owner on Windows: `uv run scripts\company_news_counts_2026_11_09.py --limit 20000 --out data\company_news_counts_smoke` (smoke), then `uv run scripts\company_news_counts_2026_11_09.py` (whole corpus, `data\company_news_counts`). A `tqdm` bar shows the progress; **the run is idempotent and resumable** (files written whole then renamed; a checkpoint every 25,000 rows and on Ctrl-C; the same command continues after the last row saved, reads only new rows when run again, and rewrites the same CSV; `--fresh` starts over); the counts are checked against themselves before any file is written, the README compares the totals with T-159's and reads the names from the text only (no stored NER spans, owner 2026-11-09, so the `ok` and quarantined-only bodies are read the same way), and a lock file stops a second run on the folder. Open after the run: the owner reads the CSV before sending it. → D36.

## T-148c3 as it stood on 2026-11-09, before the trainer was built

- [ ] **T-148c3** Trainer of D35's recipe for the sentiment model on summaries (base, replay of the earlier sentences, search on the validation slice, 3 seeds, a final run). Acceptance: the replay leaves out the sentences of gold, held-out and drawn-for-tagging bodies (by body hash) and prints how many; one run is timed first and the grid is reduced if the 28 runs do not fit; the tie rule of D35 picks the settings; a batch never mixes summaries with sentences; settings are D35's (batch 16, weight decay 0.01, fp16, linear decay, no class weights, last epoch's model), printed at the start of each run. **The trainer does not score the 180-text gold** (it is scored once, as the `final` run of the published revision, T-148c4); the run's own validation agreement is cross-checked against the scorer's within 1.1 points.
      - Timed 2026-11-07 (GPU idle): about 15 to 30 minutes for the 28 runs, so the full grid fits. **Data fact (checked 2026-11-08):** 26 of the 450 training rows (19 sector, 4 market, 3 economy) have no ticker and a placeholder `[sector]` / `[market / economy]` in `company`, not a company; proposed for the pair loader: exclude them (they belong to the text-only side, which stays the pinned model), and say so in the D35 amendment. Waits for the owner's word (#145 is merged); the date that limits is T-148c4's. → D35. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c3.

(T-148c3 as of a82d954, before the review of #154:)

- [ ] **T-148c3** *(built 2026-11-09, PR B; the grid and the final run are made; nothing is published, the gold is not scored)* Trainer of D35's recipe for the sentiment model on summaries, **read as the pair company + text**. `src/sentiment_pair_train.py` (loader of the marks, replay, batches that never mix kinds, one run on the FTI `Trainer`, the grid of 27, the tie rule) and `scripts/train_sentiment_pair_2026_11_09.py` (`--time-one`, `--grid`, `--final`, `--mlflow`; MLflow experiment `sentiment_pair_summaries`). Settings printed at the start of each run; the validation slice is scored through the scorer's own reading (`draw_sentiment_gold.predict`, pair); the trainer never reads the gold. **Data (2026-11-09):** marks train 341 + validation 83 (the 26 rows with no company are left out: 19 sector, 4 market, 3 economy; they stay with the text-only model); replay from the 3,865 sentences: 23 of a gold, held-out or drawn-for-tagging body left out, **2,917 do not name their article's company** (the rule the stage uses to weigh a chunk), 925 eligible, 341 drawn (114 / 113 / 114). **Timed:** 5.8 s an epoch; the 27 runs took 9 min. **Grid (validation agreement on 83 texts, mean of 3 seeds):** 62.2% to 65.5% by setting; the pinned model's stored prediction on the same 83: **51.8%**; chosen by the tie rule: 2 epochs, 3e-5 (65.5%, beats the earlier setting 4 / 2e-5 by 2.8 points, more than 1 SE of 1.2). **Final run (seed 4, trained on the train slice and the replay only):** 65.1% (the three seeds: mean 65.5%, sd 1.4); the model is local, `models/sentiment_pair/final`, git-ignored. This is validation, not the gold: the 80% bar is judged by the gold only (T-148c4). **Open for the owner:** the company filter keeps 925 of 3,865 sentences (proposed alternatives, none built: a sentence named by its ticker only, or a looser name match, to widen the replay). Publishing and the gold scoring are T-148c4. → D35. Full text before this entry: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c3.
