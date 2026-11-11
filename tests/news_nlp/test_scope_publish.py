"""The held-out scoring of the saved weights and the publication/pin of the scope model (T-147b): no GPU, no network."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scope_stage import ScopeResult, pin_status, revision_of

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


publish = _load("publish_scope_model_2026_11_11")
score = _load("score_scope_heldout_2026_11_11")
SHA = "a" * 40


def _folder(root: Path) -> Path:
    for stage in publish.STAGES:
        (root / stage).mkdir(parents=True)
        (root / stage / "model.safetensors").write_bytes(stage.encode())
        for name in publish.SMALL_FILES:
            (root / stage / name).write_bytes(f"{stage}/{name}".encode())
    (root / "training_record.json").write_text('{"x": 1}', encoding="utf-8")
    return root


def test_a_non_sha_answer_is_never_recorded(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no commit SHA"):
        publish.record_upload(
            tmp_path / "p.json", tmp_path / "pin.json", "", "card", "scope-nli-x", {}
        )
    assert not (tmp_path / "pin.json").exists()


def test_the_upload_is_recorded_and_pinned(tmp_path: Path) -> None:
    published = tmp_path / "p.json"
    published.write_text('{"sentiment": {"repo": "r"}}', encoding="utf-8")
    pin = tmp_path / "pin.json"
    publish.record_upload(
        published, pin, SHA, "card", "scope-nli-abc", {"stage_a": "1", "stage_b": "2"}
    )
    got = json.loads(published.read_text(encoding="utf-8"))
    assert got["sentiment"] == {"repo": "r"}  # the others are kept
    assert got["scope"]["revision"] == SHA and got["scope"]["private"] is True
    assert got["scope"]["card_sha256"] == hashlib.sha256(b"card").hexdigest()
    assert json.loads(pin.read_text(encoding="utf-8"))["step_revision"] == "scope-nli-abc"


def test_the_pin_says_whether_the_running_model_is_the_pinned_one(tmp_path: Path) -> None:
    pin = tmp_path / "pin.json"
    assert "no pin" in pin_status("scope-nli-a", pin)
    pin.write_text(
        json.dumps({"model": "o/r", "revision": SHA, "step_revision": "scope-nli-a"}),
        encoding="utf-8",
    )
    assert "pinned (o/r@aaaaaaaa)" in pin_status("scope-nli-a", pin)
    assert "NOT the pinned one" in pin_status("scope-nli-b", pin)


def test_the_card_names_the_revision_the_limit_and_fills_the_figures() -> None:
    text = publish.card("scope-nli-abc")
    assert "scope-nli-abc" in text and "out_of_scope` recall" in text and "not** met" in text
    assert "(private)" in text


def test_the_upload_refuses_while_the_card_has_no_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["publish", "--dry-run", "--model-dir", str(_folder(tmp_path / "m"))]
    )
    monkeypatch.setitem(publish.HELD_OUT, "gain", "__GAIN__")
    with pytest.raises(SystemExit, match="not filled in"):
        publish.main()


class _Api:
    def __init__(self, private: bool, head: str, weights: dict[str, str]) -> None:
        self._info = SimpleNamespace(
            private=private,
            sha=head,
            siblings=[
                SimpleNamespace(rfilename=n, lfs=SimpleNamespace(sha256=h))
                for n, h in weights.items()
            ],
        )

    def model_info(
        self, repo: str, revision: str | None = None, files_metadata: bool = False
    ) -> SimpleNamespace:
        return self._info


def _check(
    tmp_path: Path, api: _Api, answer: int = 401, changed: bool = False
) -> dict[str, bool | None]:
    model_dir = _folder(tmp_path / "m")
    weights = {s: hashlib.sha256(s.encode()).hexdigest() for s in publish.STAGES}
    published = tmp_path / "p.json"
    publish.record_upload(
        published, tmp_path / "pin.json", SHA, "card", revision_of(model_dir), weights
    )
    hub = tmp_path / "hub_files"
    shutil.copytree(model_dir, hub)
    (hub / "README.md").write_text("card", encoding="utf-8")
    if changed:
        (hub / "stage_a" / "tokenizer.json").write_bytes(b"another tokenizer")

    def download(repo: str, name: str, revision: str) -> Path:
        return hub / name

    return publish.check_published(  # type: ignore[no-any-return]
        published, model_dir, "t", api=api, download=download, anonymous_status=lambda url: answer
    )


def test_the_check_passes_for_a_private_repository_with_the_local_weights(tmp_path: Path) -> None:
    on_hub = {
        f"{s}/model.safetensors": hashlib.sha256(s.encode()).hexdigest() for s in publish.STAGES
    }
    got = _check(tmp_path, _Api(True, SHA, on_hub))
    assert all(v is True for v in got.values()), got


def test_the_check_fails_when_the_repository_is_public_or_the_weights_differ(
    tmp_path: Path,
) -> None:
    got = _check(tmp_path, _Api(False, "b" * 40, {"stage_a/model.safetensors": "0"}), answer=200)
    assert got["private"] is False and got["anonymous request refused"] is False
    assert got["recorded revision is the head of the repository"] is False
    assert got["stage_a weights on the Hub are the recorded ones"] is False


class _Stub:
    revision = "scope-nli-stub"
    chars = 800

    def classify(self, items: list[tuple[str, str]]) -> list[ScopeResult]:
        return [ScopeResult("single", {}) for _ in items]


def test_the_heldout_scoring_reads_the_classifier_in_batches_and_reports() -> None:
    rows = score.e2e.load_sets()["test"]
    assert len(rows) == 200
    assert score.predict(_Stub(), rows, batch=16) == ["single"] * 200
    gold = [score.base.merged(r["scope"]) for r in rows]
    lines = score.report(rows, gold)  # a perfect classifier
    assert any("agreement weighted to the corpus" in line for line in lines)
    assert any("confusion matrix" in line for line in lines)


class _LosingApi:
    """An upload that returns a commit but without the stage folders, as the first real upload did."""

    def __init__(self, token: str) -> None:
        pass

    def create_repo(self, *args: object, **kwargs: object) -> None:
        pass

    def upload_folder(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(oid=SHA)

    def list_repo_files(self, *args: object, **kwargs: object) -> list[str]:
        return ["README.md", "training_record.json"]


def test_an_upload_without_the_weights_is_never_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub  # noqa: PLC0415

    monkeypatch.setattr(huggingface_hub, "HfApi", _LosingApi)
    with pytest.raises(SystemExit, match="weights are not on the Hub"):
        publish.publish("t", _folder(tmp_path / "m"), "card", dry=False)


def test_the_card_example_is_valid_python_and_uses_the_records_keys() -> None:
    import ast  # noqa: PLC0415
    import re  # noqa: PLC0415

    text = publish.card("scope-nli-abc")
    code = re.search(r"```python\n(.*?)```", text, re.S)
    assert code
    ast.parse(code.group(1))
    for key in (
        'rec[stage]["hypotheses"]',
        'rec["stage_a"]["topics"]',
        'rec["stage_b"]["classes"]',
        "fragment_chars",
    ):
        assert key in code.group(1)
    assert "5f5b2c27" in text  # the history of the first upload


def _pin(tmp_path: Path, step_revision: str) -> Path:
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps({"model": "o/r", "revision": SHA, "step_revision": step_revision}),
        encoding="utf-8",
    )
    return pin


def _fake_download(source: Path) -> object:
    import shutil  # noqa: PLC0415

    def snapshot_download(repo: str, revision: str, local_dir: str, **_: object) -> str:
        assert (repo, revision) == ("o/r", SHA)
        shutil.copytree(source, local_dir, dirs_exist_ok=True)
        return local_dir

    return snapshot_download


def test_fetch_pinned_downloads_the_pinned_commit_and_checks_the_step_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub  # noqa: PLC0415

    from scope_stage import fetch_pinned  # noqa: PLC0415

    source = _folder(tmp_path / "hub")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_download(source))
    target = tmp_path / "models" / "scope_final"
    assert fetch_pinned(target, _pin(tmp_path, revision_of(source))) == revision_of(source)
    assert (target / "stage_a" / "model.safetensors").exists()


def test_fetch_pinned_refuses_a_copy_that_is_not_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub  # noqa: PLC0415

    from scope_stage import fetch_pinned  # noqa: PLC0415

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", _fake_download(_folder(tmp_path / "hub"))
    )
    with pytest.raises(RuntimeError, match="not the pinned"):
        fetch_pinned(tmp_path / "t", _pin(tmp_path, "scope-nli-other"))


class _GoodApi(_LosingApi):
    def list_repo_files(self, *args: object, **kwargs: object) -> list[str]:
        return [
            "README.md",
            "training_record.json",
            "stage_a/model.safetensors",
            "stage_b/model.safetensors",
        ]


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["publish", *argv])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    publish.main()


def test_the_upload_records_the_pin_and_the_dry_run_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub  # noqa: PLC0415

    monkeypatch.setattr(huggingface_hub, "HfApi", _GoodApi)
    monkeypatch.setenv("HF_TOKEN", "t")
    folder = _folder(tmp_path / "m")
    record, pin = tmp_path / "published.json", tmp_path / "pin.json"
    args = ["--model-dir", str(folder), "--record", str(record), "--pin", str(pin)]
    _main(monkeypatch, *args, "--dry-run")
    assert not record.exists() and not pin.exists()
    _main(monkeypatch, *args)
    assert json.loads(pin.read_text(encoding="utf-8"))["revision"] == SHA
    assert json.loads(record.read_text(encoding="utf-8"))["scope"]["step_revision"] == revision_of(
        folder
    )


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_the_check_exits_non_zero_when_a_check_says_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ok: bool, code: int
) -> None:
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr(
        publish, "check_published", lambda *a, **k: {"private": True, "weights": ok}
    )
    with pytest.raises(SystemExit) as stop:
        _main(monkeypatch, "--check", "--model-dir", str(tmp_path))
    assert stop.value.code == code


def test_the_heldout_scoring_script_writes_one_prediction_per_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scope_stage  # noqa: PLC0415

    monkeypatch.setattr(scope_stage, "ScopeClassifier", lambda model_dir: _Stub())
    out = tmp_path / "pred.csv"
    monkeypatch.setattr(sys, "argv", ["score", "--out", str(out)])
    score.main()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 201
    assert "model revision scope-nli-stub" in capsys.readouterr().out


def test_the_steps_cli_fetches_before_it_reports(
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scope_stage  # noqa: PLC0415
    from tests.news_nlp.test_steps import _source_with  # noqa: PLC0415

    source = tmp_path / "s.db"
    _source_with(source, [(1, "A", "One. " * 5)])
    called: list[Path] = []

    def fake_fetch(model_dir: Path) -> str:
        called.append(model_dir)
        return "scope-nli-x"

    monkeypatch.setattr(scope_stage, "fetch_pinned", fake_fetch)
    cli = importlib.util.spec_from_file_location(
        "steps_cli_167", Path(__file__).resolve().parents[2] / "cli" / "news_nlp_steps.py"
    )
    assert cli and cli.loader
    module = importlib.util.module_from_spec(cli)
    sys.modules[cli.name] = module
    cli.loader.exec_module(module)
    monkeypatch.setattr(
        sys, "argv",
        ["steps", "--step", "scope", "--status", "--fetch", "--model-dir", str(tmp_path / "m"), "--results-db", str(results_db_path), "--source-db", str(source)],
    )  # fmt: skip
    module.main()
    assert called == [tmp_path / "m"] and "fetched the pinned model" in capsys.readouterr().out


def test_the_check_sees_a_small_file_that_differs_from_the_hub_copy(tmp_path: Path) -> None:
    on_hub = {
        f"{s}/model.safetensors": hashlib.sha256(s.encode()).hexdigest() for s in publish.STAGES
    }
    got = _check(tmp_path, _Api(True, SHA, on_hub), changed=True)
    assert got["stage_a/tokenizer.json on the Hub is the local one"] is False
    assert got["stage_b/tokenizer.json on the Hub is the local one"] is True


def test_fetch_pinned_refuses_a_folder_that_holds_files_and_deletes_a_wrong_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub  # noqa: PLC0415

    from scope_stage import fetch_pinned  # noqa: PLC0415

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", _fake_download(_folder(tmp_path / "hub"))
    )
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        fetch_pinned(busy, _pin(tmp_path, "scope-nli-other"))
    target = tmp_path / "models" / "scope_final"
    with pytest.raises(RuntimeError, match="deleted"):
        fetch_pinned(target, _pin(tmp_path, "scope-nli-other"))
    assert not target.exists() and not list((tmp_path / "models").glob(".*"))


def test_the_step_refuses_a_model_that_is_not_the_pinned_one_unless_allowed(
    tmp_path: Path, results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scope_stage  # noqa: PLC0415
    from tests.news_nlp.test_steps import _source_with  # noqa: PLC0415

    source = tmp_path / "s.db"
    _source_with(source, [(1, "A", "One. " * 5)])
    folder = _folder(tmp_path / "m")
    monkeypatch.setattr(scope_stage, "PIN_PATH", _pin(tmp_path, "scope-nli-other"))
    spec = importlib.util.spec_from_file_location(
        "steps_cli_167b", Path(__file__).resolve().parents[2] / "cli" / "news_nlp_steps.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    base = [
        "steps",
        "--step",
        "scope",
        "--model-dir",
        str(folder),
        "--results-db",
        str(results_db_path),
    ]
    monkeypatch.setattr(sys, "argv", [*base, "--source-db", str(source)])
    with pytest.raises(SystemExit, match="not the pinned model"):
        module.main()
    # --allow-unpinned lets it go on to load the model (a GPU; here the load is stubbed to stop)
    monkeypatch.setattr(
        scope_stage, "ScopeClassifier", lambda d: (_ for _ in ()).throw(RuntimeError("loaded"))
    )
    monkeypatch.setattr(sys, "argv", [*base, "--source-db", str(source), "--allow-unpinned"])
    with pytest.raises(RuntimeError, match="loaded"):
        module.main()


breakdown_script = _load("scope_corpus_breakdown_2026_11_11")


def test_the_corpus_breakdown_counts_unique_bodies_by_status_and_year() -> None:
    rows = [
        (1, "ok", "2020-01-01", "A", "Body one. " * 5),
        (
            2,
            "quarantined",
            "2021-05-05",
            "A again",
            "Body one.  " * 5,
        ),  # same body: the first article counts
        (3, "quarantined", "2021-06-06", "B", "Body two. " * 5),
        (4, "quarantined", None, "C", "Body three. " * 5),
        (5, "ok", "2020-02-02", "D", ""),  # no body
    ]
    bodies = breakdown_script.unique_bodies(rows)
    assert len(bodies) == 3
    from news_nlp import triage  # noqa: PLC0415

    scope_of = {
        triage.body_hash("Body one. " * 5): "single",
        triage.body_hash("Body two. " * 5): "out_of_scope",
    }
    by_status, by_year = breakdown_script.breakdown(bodies, scope_of)
    assert dict(by_status["ok"]) == {"single": 1} and dict(by_status["quarantined"]) == {
        "out_of_scope": 1
    }
    assert "?" not in by_year  # the body without a scope is not counted
    assert breakdown_script.sample_oos(bodies, scope_of, "quarantined", 5)[0][0] == 3
    assert any("quarantined" in line for line in breakdown_script.lines(by_status, "by status"))
