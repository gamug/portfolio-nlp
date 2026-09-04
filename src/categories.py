"""Back-compat re-export. The 10-category taxonomy now lives in
`portfolio_common.news_nlp.taxonomy`; kept here so `from categories import ...`
call sites (pipeline.py, setup.py, apps/) don't churn.
"""

from portfolio_common.news_nlp.taxonomy import (  # noqa: F401
    CATEGORY_LABELS,
    CATEGORY_SLUGS,
    OTHER_LABEL,
)
