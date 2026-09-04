"""Back-compat re-export. Manual result-row corrections now live in
`portfolio_common.news_nlp.corrections`; kept here so `import corrections`
call sites (apps/news_nlp_api.py, tests) don't churn.
"""

from portfolio_common.news_nlp.corrections import (  # noqa: F401
    _now_iso,
    delete_category,
    delete_entities_for_article,
    delete_entity,
    delete_sentiment,
    update_category,
    update_entity,
    update_sentiment,
)
