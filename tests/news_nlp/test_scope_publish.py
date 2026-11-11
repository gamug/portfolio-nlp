"""The held-out scoring of the saved weights and the publication/pin of the scope model (T-147b): no GPU, no network."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _check(tmp_path: Path, api: _Api, answer: int = 401) -> dict[str, bool | None]:
    model_dir = _folder(tmp_path / "m")
    weights = {s: hashlib.sha256(s.encode()).hexdigest() for s in publish.STAGES}
    published = tmp_path / "p.json"
    publish.record_upload(
        published, tmp_path / "pin.json", SHA, "card", revision_of(model_dir), weights
    )
    card_file = tmp_path / "card.md"
    card_file.write_text("card", encoding="utf-8")
    return publish.check_published(  # type: ignore[no-any-return]
        published,
        model_dir,
        "t",
        api=api,
        download=lambda *a: card_file,
        anonymous_status=lambda url: answer,
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
