"""The service layer of the stepped pipeline (T-219): run a step, run the pipeline, serve the stage models.

No web framework here: ``steps_runner`` turns a request into the arguments of ``cli/news_nlp_steps.py`` and runs the CLI's own functions,
``jobs`` keeps one background run at a time, ``stage_models`` serves the pinned models of the stages. ``apps/pipeline_api.py`` is the thin FastAPI
layer over them. Design and decisions: ``docs/pipeline_api.md``.
"""
