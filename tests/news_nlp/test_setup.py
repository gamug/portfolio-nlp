import pytest

import setup


def test_download_models_fetches_every_model_and_verifies_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_calls: list[tuple[str, str]] = []
    config_calls: list[tuple[str, str]] = []

    def fake_snapshot_download(repo_id: str, revision: str) -> str:
        snapshot_calls.append((repo_id, revision))
        return f"/fake/cache/{repo_id}"

    def fake_from_pretrained(repo_id: str, revision: str) -> None:
        config_calls.append((repo_id, revision))

    monkeypatch.setattr(setup, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(setup.AutoConfig, "from_pretrained", fake_from_pretrained)

    setup.download_models()

    expected = [(repo_id, setup.MODEL_REVISIONS[repo_id]) for repo_id in setup.MODELS]
    assert snapshot_calls == expected
    assert config_calls == expected
    assert setup.CATEGORY_MODEL in setup.MODELS
    # Every model is pinned -- no silent drift if left out of MODEL_REVISIONS
    # (SPEC.md SS13 item 4, PLAN.md Work item 1).
    assert set(setup.MODELS) <= set(setup.MODEL_REVISIONS)
