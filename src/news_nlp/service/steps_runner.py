"""Run a step of the stepped pipeline exactly as ``cli/news_nlp_steps.py`` does (T-219).

A request is turned into the CLI's argv and parsed by the CLI's own parser and ``check_options``, so every refusal of the command line is a refusal
of the API, and the run is the CLI's own ``main_<step>`` function: the same rows. Nothing here changes the CLI.

Options the API does not take (they act on files, on the stores or on the weights, and the service has no authentication, owner 2026-11-17): the
download of a model (``fetch``), a model folder (``model_dir``), ``allow_unpinned``, the two store paths (``source_db`` and ``results_db``: the
service reads them from the environment as the CLI does) and ``reset`` (deletes rows). They stay with the command line.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = ROOT / "cli" / "news_nlp_steps.py"
# the order of the pipeline (the CLI's STEPS); a whole-pipeline run keeps it
PIPELINE_ORDER = (
    "scope",
    "cut",
    "summary",
    "companies",
    "sector",
    "category",
    "sentiment",
    "snapshot",
    "period",
)
NOT_THROUGH_THE_API = frozenset(
    {"fetch", "model_dir", "allow_unpinned", "source_db", "results_db", "reset"}
)

_CLI: Any | None = None


class StepRefused(Exception):
    """A request the step will not run: bad option, option not available through the API, or a refusal of the CLI itself."""


def cli() -> Any:
    """``cli/news_nlp_steps.py`` loaded once as a module (it is a script, not a package)."""
    global _CLI  # noqa: PLW0603 -- one load, as the module is heavy to import
    if _CLI is None:
        spec = importlib.util.spec_from_file_location("news_nlp_steps_cli_service", CLI_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _CLI = module
    return _CLI


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_argv(step: str, options: Mapping[str, Any]) -> list[str]:
    """The command line of one step: ``True`` is a flag, ``False`` and ``None`` are left out, anything else is the option's value."""
    if step not in PIPELINE_ORDER:
        raise StepRefused(f"unknown step {step!r}: one of {', '.join(PIPELINE_ORDER)}")
    argv = ["--step", step]
    for raw, value in options.items():
        name = raw.replace("-", "_")
        if name in NOT_THROUGH_THE_API:
            raise StepRefused(
                f"option {raw!r} is not available through the API (run the command line for it)"
            )
        if name == "step":
            raise StepRefused("the step is the path or the 'step' field, not an option")
        if value is None or value is False:
            continue
        argv.append(_flag(name))
        if value is not True:
            argv.append(str(value))
    return argv


def _reached(args: argparse.Namespace) -> list[str]:
    """The options of ``NOT_THROUGH_THE_API`` the parse set, whatever the key that reached them was."""
    flags = [n for n in ("fetch", "allow_unpinned", "reset") if getattr(args, n, False)]
    values = [
        n for n in ("model_dir", "source_db", "results_db") if getattr(args, n, None) is not None
    ]
    return [*flags, *values]


def parse_step(step: str, options: Mapping[str, Any]) -> argparse.Namespace:
    """Parse and check one step's options with the CLI's own parser; a refusal is a ``StepRefused`` with the CLI's message.

    Only exact option names parse (``allow_abbrev`` off: the CLI's parser takes any unique prefix, so ``rese`` was ``--reset``), and the result
    is checked as well as the keys: an option that acts on files, the stores or the rows is refused by what it set.
    """
    module = cli()
    parser = module.build_parser()
    parser.allow_abbrev = False
    argv = build_argv(step, options)
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            args: argparse.Namespace = parser.parse_args(argv)
            module.check_options(parser, args)
    except SystemExit as exc:  # argparse's error() exits with 2
        message = stderr.getvalue().strip().splitlines()
        raise StepRefused(message[-1] if message else f"refused ({exc.code})") from exc
    reached = _reached(args)
    if reached:
        raise StepRefused(
            f"option {', '.join(reached)} is not available through the API (run the command line for it)"
        )
    if args.step == "summary" and args.summariser != "local" and args.max_calls is None:
        raise StepRefused(
            "the summary through the API needs max_calls: it calls a paid API and the service has no authentication (the CLI allows up to "
            "5,000 texts without a cap; here the cap is always explicit)"
        )
    return args


@dataclass
class StepOutcome:
    """What one step printed: the JSON documents of stdout in order, the other lines, and how it ended."""

    ok: bool
    outputs: list[Any] = field(default_factory=list)
    log: str = ""
    message: str = ""


def split_documents(text: str) -> tuple[list[Any], str]:
    """The JSON documents printed one after another (the CLI prints a result and then a status) and the lines that are not JSON."""
    decoder = json.JSONDecoder()
    documents: list[Any] = []
    other: list[str] = []
    pos = 0
    while pos < len(text):
        if text[pos] in " \t\r\n":
            pos += 1
            continue
        try:
            document, end = decoder.raw_decode(text, pos)
        except ValueError:
            newline = text.find("\n", pos)
            end = len(text) if newline < 0 else newline + 1
            other.append(text[pos:end].rstrip())
        else:
            documents.append(document)
        pos = end
    return documents, "\n".join(other)


class Tail(io.TextIOBase):
    """A stderr that keeps the last progress line (tqdm rewrites a line with carriage returns) and tells a listener."""

    def __init__(self, on_line: Callable[[str], None] | None = None, keep: int = 4000) -> None:
        self._on_line = on_line
        self._keep = keep
        self.text = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.text = (self.text + text)[-self._keep :]
        segments = [s for s in text.replace("\r", "\n").split("\n") if s.strip()]
        if segments and self._on_line is not None:
            self._on_line(segments[-1].strip())
        return len(text)


def run_parsed(args: argparse.Namespace) -> None:
    """What ``main()`` of the CLI does after the parse: the default model folder, the environment, the connection, the schema, the step."""
    module = cli()
    args.model_dir = args.model_dir or module.default_model_dir(args.step)
    module.db.env.load_env_file()
    with module.exit_on_schema_error():
        conn = module.db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    try:
        module.db.init_schema(conn)
        runners = {
            "cut": module.main_cut,
            "summary": module.main_summary,
            "companies": module.main_companies,
            "sector": module.main_sector,
            "category": module.main_category,
            "sentiment": module.main_sentiment,
            "snapshot": module.main_snapshot,
            "period": module.main_period,
        }
        runners.get(args.step, module.main_scope)(args, conn)
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def execute(
    args: argparse.Namespace,
    on_progress: Callable[[str], None] | None = None,
    runner: Callable[[argparse.Namespace], None] = run_parsed,
) -> StepOutcome:
    """Run the step with its stdout and stderr captured. ``sys.exit(message)`` of the CLI is a refusal, not a crash."""
    out = io.StringIO()
    err = Tail(on_progress)
    ok, message = True, ""
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runner(args)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            ok, message = False, str(exc.code)
    except Exception as exc:
        ok, message = False, f"{type(exc).__name__}: {exc}"
    outputs, other = split_documents(out.getvalue())
    log = "\n".join(part for part in (other, err.text.strip()) if part)
    return StepOutcome(ok=ok, outputs=outputs, log=log[-4000:], message=message)
