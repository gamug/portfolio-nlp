# Gold datasets: the tagging roadmap of 2026-11-06 (legacy)

**Legacy (moved 2026-11-12).** Cut verbatim from `docs/gold_datasets.md` at commit `09d90ae` (lines 282-315). It planned the order of tagging against the dates of PLAN D34 (the backtest cut for the 2026-11-20 delivery), which the owner took out of the MVP on 2026-11-10; every set it lists has since been tagged and migrated (`docs/gold_datasets.md`, "The sets").

## Roadmap: what each set unlocks and the order to tag

Fixed dates are the ones in `PLAN.md` D34 (cut for the 2026-11-20 delivery); the extractor decision is on
2026-11-09 (D34 rule b; D32 now says the same). The order is by what
each set blocks, earliest block first.

| Order | Set | Unblocks | Needed by | Waits for |
|---|---|---|---|---|
| 1 | **3. Label check** (200 paragraphs) | training the attribution model (`lead` only, or both kinds) | 2026-11-08 | nothing: the training set exists |
| 2 | **4. Extractor sample** (30 bodies, 190 texts) | the decision on the extractor for narrative bodies; the extractor part of set 5 | 2026-11-09 | nothing: baseline-only draw (D34 rule b) |
| 3 | **5. Sentiment** (180 texts) | the D32 sentiment bar, so the cut can say whether its sentiment is checked | 2026-11-12 | the extractor decision (its 60 texts come from the extractor chosen) |
| 4 | **2. Scope test** (200 bodies) | the scope measure (rules now, classifier later), the D24 category target | 2026-11-12 (go / no-go) | nothing: it can be tagged any time, but the cut does not wait for its result to start |
| done | 1. Scope train | the labelling prompt, the sample for sets 4 and 5 | | |

Owner work before the go / no-go of 2026-11-12: 770 items (200 + 200 + 190 + 180), 200 of them marked
(the label check). The PLAN counted 610 by taking 30 texts for the faithfulness set, but the set is 30
*bodies* and one row per company text: drawn, it is 190. **Owner call, 2026-11-06: tag all 190** (no reduction).

Why this order:

- Set 3 is first because the model trains on 2026-11-08 and the rule on the `continuation` labels must be
  applied *before* training; it is the shortest set and needs no other result.
- Set 4 needs only the baseline, so it can be drawn and tagged at once; if the model is not ready on
  2026-11-09 the decision is on the baseline alone.
  **Only if the 190 are not all marked on 2026-11-09 (the owner expects to finish them on 2026-11-06, so this is a contingency; the 120 below is a proposal the owner has not confirmed; fixed before any figure is read):**
  the owner marks in `review_order` order; on 2026-11-09 the decision uses the first rows marked, if at
  least 120 (about two thirds; `report_wi18 faithfulness --partial 120`, which refuses a gap or fewer
  rows and prints PARTIAL, counting every company left without a text); with fewer the baseline is kept and
  the model is reported as not evaluated. The full figure replaces the partial one when the file is finished.
- Set 5 draws its extractor part from whichever extractor set 4 selected, so it is drawn after.
  Its checked part (120) does not depend on it and can be tagged first if the owner wants to split the work.
- Set 2 is independent of the others, so it fills the gaps; it is last only because nothing downstream
  starts from its result.
