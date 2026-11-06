"""Run only the tests a change can affect, with the standard library only.

    uv run scripts/affected_tests.py [--since REF] [--heavy] [--dry-run]
    uv run scripts/affected_tests.py --set fast|wi18|db|eval|stages|api [--set ...]

Without ``--set`` it takes the files changed since ``--since`` (default ``HEAD``: the staged,
unstaged and untracked changes) and selects the test files in ``tests/news_nlp`` that

* are themselves changed;
* import (directly or through another module of the repository, at the top level) a changed
  module, or name a changed file or script in their text (the tests that load a script by name);
* are all of them when ``pytest.ini``, ``pyproject.toml``, ``tests/conftest.py`` or ``tests/sets.py``
  changed.

Test files that import torch, transformers, mlflow or the pipeline (``tests/sets.py`` ``HEAVY``)
cost 50 to 400 s each to import on this disk, so they are listed and skipped unless ``--heavy`` is
given; CI runs everything. The chosen files run in ONE pytest process.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.sets import HEAVY, TOPICS  # noqa: E402

TESTS = ROOT / "tests" / "news_nlp"
EVERYTHING = {"pytest.ini", "pyproject.toml", "tests/conftest.py", "tests/sets.py"}


def modules() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for base in (ROOT / "src", ROOT):
        for p in base.rglob("*.py"):
            parts = p.relative_to(base).with_suffix("").parts
            if ".venv" in p.parts or "tests" in parts or not parts:
                continue
            name = ".".join(parts).removesuffix(".__init__")
            found.setdefault(name, p)
    return found


def top_level_imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        for sub in ast.walk(node) if isinstance(node, ast.If | ast.Try) else [node]:
            if isinstance(sub, ast.Import):
                out |= {a.name for a in sub.names}
            elif isinstance(sub, ast.ImportFrom) and sub.module and not sub.level:
                out |= {sub.module, *(f"{sub.module}.{a.name}" for a in sub.names)}
    return out


def closure(path: Path, known: dict[str, Path]) -> set[Path]:
    seen: set[Path] = set()
    stack = [path]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for name in top_level_imports(current):
            for candidate in (name, name.rsplit(".", 1)[0]):
                if candidate in known:
                    stack.append(known[candidate])
    return seen


def changed_files(since: str) -> set[str]:
    def git(*args: str) -> list[str]:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607 -- git from PATH, arguments are ours
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in out.stdout.splitlines() if line]

    return set(git("diff", "--name-only", since)) | set(
        git("ls-files", "--others", "--exclude-standard")
    )


def affected(changed: set[str]) -> set[str]:
    tests = sorted(TESTS.glob("test_*.py"))
    if changed & EVERYTHING:
        return {t.name for t in tests}
    known = modules()
    paths = {ROOT / c for c in changed}
    chosen: set[str] = set()
    for test in tests:
        text = test.read_text(encoding="utf-8")
        reached = closure(test, known)
        if (
            test in paths
            or reached & paths
            or any(
                Path(c).name in text or Path(c).stem in text
                for c in changed
                if c.endswith((".py", ".md"))
            )
        ):
            chosen.add(test.name)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="HEAD")
    parser.add_argument("--set", action="append", default=[], choices=["fast", "heavy", *TOPICS])
    parser.add_argument("--heavy", action="store_true", help="also run the heavy files")
    parser.add_argument("--dry-run", action="store_true", help="only print the selection")
    args = parser.parse_args()
    if args.set:
        command = [sys.executable, "-m", "pytest", "-q", *[f"--set={s}" for s in args.set]]
    else:
        files = affected(changed_files(args.since))
        heavy = sorted(files & HEAVY)
        run = sorted(files - HEAVY) + (heavy if args.heavy else [])
        print(f"affected: {len(files)} test files; run {len(run)}; heavy {len(heavy)}")
        if heavy and not args.heavy:
            print("heavy, skipped (--heavy to run, CI runs them):", ", ".join(heavy))
        if not run:
            print("nothing to run")
            return
        command = [sys.executable, "-m", "pytest", "-q", *[str(TESTS / f) for f in run]]
    print(" ".join(command[2:]))
    if not args.dry_run:
        sys.exit(subprocess.run(command, cwd=ROOT, check=False).returncode)  # noqa: S603


if __name__ == "__main__":
    main()
