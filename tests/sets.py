"""The named sets of tests (``pytest --set <name>``): which file belongs to which, and how heavy it is.

``fast`` files import nothing heavy (no torch / transformers / mlflow / the pipeline): the whole
set runs in seconds. ``heavy`` files pay for those imports (50 to 400 s on this disk) and are left
to CI or to a run of their own. The topic sets say what a file is about, so a change runs the
tests it can affect. ``conftest.py`` applies the marks and the ``--set`` option from these tables;
a new test file must be added to a topic here (``news_nlp/test_sets.py`` fails otherwise), and to
``HEAVY`` if it imports the pipeline.
"""

from __future__ import annotations

#: Files whose collection or fixtures import torch / transformers / mlflow / the pipeline,
#: measured 2026-11-06 (every other file runs in under 5 s).
HEAVY = frozenset(
    {
        "test_category_pipeline.py",
        "test_correction_endpoints.py",
        "test_eval_calibration_judge.py",
        "test_eval_candidate.py",
        "test_eval_regression.py",
        "test_eval_runner.py",
        "test_eval_tracking.py",
        "test_experiment.py",
        "test_experiment_run.py",
        "test_experiments_backfill.py",
        "test_fti_base.py",
        "test_ner_pipeline.py",
        "test_pipeline_endpoints.py",
        "test_pipeline_model_config.py",
        "test_pipeline_progress.py",
        "test_pipeline_run.py",
        "test_query_endpoints.py",
        "test_schema_guard.py",
        "test_sentiment_pair_train.py",
        "test_sentiment_pipeline.py",
        "test_setup.py",
        "test_summary_pipeline.py",
        "test_category_step.py",
        "test_sector_step.py",
        "test_sector_validation.py",
        "test_summary_step.py",
        "test_train_ner.py",
        "test_train_sentiment.py",
        "test_two_tier.py",
    }
)

TOPICS: dict[str, frozenset[str]] = {
    # Work item 18: gold, tagging, triage, step-4 texts, the owner's files and their figures
    "wi18": frozenset(
        {
            "test_attribution_model.py",
            "test_attribution_publish.py",
            "test_attribution_units.py",
            "test_build_sentiment_train.py",
            "test_company_names.py",
            "test_company_texts.py",
            "test_count_fetched_corpus.py",
            "test_company_news_counts.py",
            "test_cut.py",
            "test_cut_sample.py",
            "test_cut_sample_gold.py",
            "test_cut_sample_metrics.py",
            "test_companies_step.py",
            "test_category_step.py",
            "test_relevance_threshold_candidates.py",
            "test_cut_summaries.py",
            "test_gate240_script.py",
            "test_gold_metrics.py",
            "test_gold_review.py",
            "test_gold_scoring_sentiment.py",
            "test_labelling_spec.py",
            "test_matcher_before_after.py",
            "test_retag.py",
            "test_scope_publish.py",
            "test_steps.py",
            "test_sector_step.py",
            "test_sector_validation.py",
            "test_summary_step.py",
            "test_sentiment_pair.py",
            "test_sentiment_pair_train.py",
            "test_sentiment_pair_report.py",
            "test_sentiment_training_doc.py",
            "test_train_sentiment_pair_script.py",
            "test_sentiment_train_draw.py",
            "test_translate.py",
            "test_triage.py",
            "test_wi18_scripts.py",
        }
    ),
    # the DB layer and the two-tier store
    "db": frozenset(
        {
            "test_corrections.py",
            "test_db.py",
            "test_dependencies.py",
            "test_env.py",
            "test_queries.py",
            "test_schema.py",
            "test_schema_guard.py",
            "test_sector_summary.py",
            "test_taxonomy.py",
            "test_two_tier.py",
        }
    ),
    # LLM-as-judge evaluation and the JSON-driven experiments
    "eval": frozenset(
        {
            "test_eval_calibration.py",
            "test_eval_calibration_analysis.py",
            "test_eval_calibration_judge.py",
            "test_eval_candidate.py",
            "test_eval_judges.py",
            "test_eval_metrics.py",
            "test_eval_partition.py",
            "test_eval_regression.py",
            "test_eval_runner.py",
            "test_eval_sampling.py",
            "test_eval_store.py",
            "test_eval_tracking.py",
            "test_experiment.py",
            "test_experiment_run.py",
            "test_experiments_backfill.py",
        }
    ),
    # the four ML stages, the pipeline and training
    "stages": frozenset(
        {
            "test_category_pipeline.py",
            "test_fti_base.py",
            "test_ner_pipeline.py",
            "test_pipeline_model_config.py",
            "test_pipeline_progress.py",
            "test_pipeline_run.py",
            "test_sentiment_pipeline.py",
            "test_setup.py",
            "test_summary_pipeline.py",
            "test_train_ner.py",
            "test_train_sentiment.py",
        }
    ),
    # the FastAPI service
    "api": frozenset(
        {
            "test_correction_endpoints.py",
            "test_pipeline_endpoints.py",
            "test_query_endpoints.py",
        }
    ),
}
