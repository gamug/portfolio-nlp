"""The fixed 10-category taxonomy (9 substantive + "other") used by the
news-NLP pipeline's zero-shot category-classification stage, plus the
3-group hierarchy (`CATEGORY_GROUPS`) that stage classifies through in two
passes -- see `docs/category-taxonomy.md`'s "Hierarchical classification"
section for why.

Kept separate so it's importable without pulling in torch (the pipeline's
`setup.py`, docs generation, or future non-GPU code can use it directly).

This is the canonical machine-readable source for the taxonomy -- keep
`portfolio-nlp`'s `docs/category-taxonomy.md` in sync with it if the labels
ever change. See that doc for the sourcing behind each category (RavenPack,
SASB, IPTC Media Topics, Refinitiv/TRNA).
"""

# (slug, display_name, hypothesis_topic_phrase)
#
# `slug` is stored as the DB `label` value (machine-stable, matches how
# article_sentiment.label stores "positive"/"negative"/"neutral" slugs, not
# display strings) and becomes one article_category score column each.
#
# `hypothesis_topic_phrase` fills the NLI hypothesis template
# "This example is about {phrase}." used by pipeline.run_category_stage.
CATEGORY_LABELS = [
    (
        "earnings_performance",
        "Earnings & Financial Performance",
        "earnings and financial performance",
    ),
    ("mergers_acquisitions", "Mergers & Acquisitions", "mergers and acquisitions"),
    (
        "leadership_governance",
        "Corporate Leadership & Governance",
        "corporate leadership and governance",
    ),
    ("legal_regulatory", "Legal & Regulatory", "legal and regulatory matters"),
    ("product_innovation", "Product & Innovation", "products and innovation"),
    (
        "capital_shareholder_returns",
        "Capital Actions & Shareholder Returns",
        "capital actions and shareholder returns",
    ),
    ("labor_human_capital", "Labor & Human Capital", "labor and human capital"),
    ("market_analyst_sentiment", "Market & Analyst Sentiment", "market and analyst sentiment"),
    (
        "partnerships_business_dev",
        "Partnerships & Business Development",
        "partnerships and business development",
    ),
]

# Catch-all for anything that doesn't clear CATEGORY_CONFIDENCE_THRESHOLD
# against any of the 9 labels above. Not an NLI candidate itself -- there's
# no hypothesis for "other", it's the below-threshold fallback.
OTHER_LABEL = "other"

CATEGORY_SLUGS = tuple(slug for slug, _, _ in CATEGORY_LABELS)

# Applied at level 2 of the hierarchical classifier (below) against a
# candidate set of 6 slugs (an article's top-2 groups' children combined),
# not all 9 -- uniform-chance baseline there is ~0.167, so 0.4 is ~2.4x
# baseline (between CATEGORY_GROUP_FLOOR's ~1.2x and the original flat
# 9-way design's ~3.6x over its 0.111 baseline). Kept at the same
# name/value the flat design used, to minimize churn in code that already
# imports it (news_nlp.eval.sampling/verdicts) -- its statistical basis
# changed (9-way -> 6-way denominator) even though the number didn't; a
# reasoned starting point, not a validated one, same as always. Retune
# using article_category's stored per-label score distribution -- see
# docs/category-taxonomy.md. Lives here (not in pipeline.py) so
# news_nlp.eval can share the value without importing torch; pipeline.py
# re-imports it.
CATEGORY_CONFIDENCE_THRESHOLD = 0.4

# Level-1 "group" taxonomy for the two-level hierarchical zero-shot
# classification (docs/category-taxonomy.md): a flat 9-way softmax gives
# each label only a ~0.11 uniform-chance baseline, diluting real signal for
# any one label -- empirically the dominant cause of poor recall on
# product_innovation/partnerships_business_dev/leadership_governance
# (confirmed via eval data, not assumed). Splitting into 3 groups of 3
# raises the per-decision baseline to ~0.33 (level 1) and ~0.17 (level 2,
# over the winning *and* runner-up groups' 6 combined children -- top-2, not
# top-1, so a narrow level-1 miscall can still be recovered at level 2).
#
# Every group tuple is (group_slug, group_display_name,
# group_hypothesis_phrase, children), where `children` is a disjoint subset
# of CATEGORY_SLUGS. The three children tuples partition CATEGORY_SLUGS
# exactly -- asserted below, not just assumed.
CATEGORY_GROUPS: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "corporate_actions",
        "Corporate Actions & Capital",
        "corporate financial actions and capital structure",
        ("earnings_performance", "mergers_acquisitions", "capital_shareholder_returns"),
    ),
    (
        "governance_legal_workforce",
        "Governance, Legal & Workforce",
        "corporate governance, legal, and workforce matters",
        ("leadership_governance", "legal_regulatory", "labor_human_capital"),
    ),
    (
        "market_product_partnerships",
        "Market, Product & Partnerships",
        "market sentiment, products, and business partnerships",
        ("product_innovation", "market_analyst_sentiment", "partnerships_business_dev"),
    ),
]

CATEGORY_GROUP_SLUGS = tuple(slug for slug, _, _, _ in CATEGORY_GROUPS)

#: group_slug -> its 3 children, for run_category_stage's level-2 candidate
#: lookup without re-scanning CATEGORY_GROUPS per article.
CATEGORY_GROUP_CHILDREN = {slug: children for slug, _, _, children in CATEGORY_GROUPS}

#: leaf slug -> its owning group slug (the inverse of CATEGORY_GROUP_CHILDREN).
CATEGORY_SLUG_TO_GROUP = {
    child: group_slug for group_slug, _, _, children in CATEGORY_GROUPS for child in children
}

#: slug -> its full CATEGORY_LABELS tuple, for level 2's per-slug hypothesis
#: lookup by name (a variable top-2-groups candidate set) instead of
#: CATEGORY_LABELS' fixed all-9 order.
CATEGORY_LABELS_BY_SLUG = {
    slug: (slug, display, phrase) for slug, display, phrase in CATEGORY_LABELS
}

assert sorted(c for _, _, _, children in CATEGORY_GROUPS for c in children) == sorted(
    CATEGORY_SLUGS
), "CATEGORY_GROUPS' children must partition CATEGORY_SLUGS exactly -- no leftovers, no duplicates"

# Guards the one short-circuit in the two-pass pipeline: if level 1's
# winning group doesn't clear this, its 3-way distribution is treated as
# uninformative (near the 0.333 uniform baseline) and run_category_stage
# skips level 2 entirely, writing label=OTHER_LABEL with the group's own
# (sub-floor) score preserved for audit -- same near-miss-is-auditable
# spirit as CATEGORY_CONFIDENCE_THRESHOLD. Deliberately modest (~1.2x
# baseline, vs. CATEGORY_CONFIDENCE_THRESHOLD's ~2.4x over its own 6-way
# baseline): level 2's top-2 expansion exists specifically to recover
# articles level 1 was only lukewarm about, so this floor should only catch
# genuinely flat triples, not merely unconfident ones. A reasoned starting
# point, not a validated one -- retune using article_category's stored
# group_score column once real classification data exists.
CATEGORY_GROUP_FLOOR = 0.40
