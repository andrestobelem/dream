import pytest

from think_dream_lab.config import load_config
from think_dream_lab.storage import read_json, write_json


def test_continuity_resumes_penultimate_checkpoint_and_checks_final_state(tmp_path, monkeypatch):
    from think_dream_lab import dreambank_continuity as continuity

    config = load_config("configs/dream-v5.yaml")
    source = tmp_path / "dream/smoke"
    write_json(
        source / "run.json",
        {
            "status": "complete",
            "steps_requested": 3,
            "identity": "id",
            "actor_train": [{"seeds": {}}],
            "actor_validation": [],
        },
    )
    reward = type("Reward", (), {"identity": "bank"})()
    monkeypatch.setattr(continuity, "load_reward", lambda *a: ({}, {"hash": "data"}, reward))
    monkeypatch.setattr(continuity, "freeze_model", lambda *a, **k: {})
    monkeypatch.setattr(continuity, "identity", lambda *a: "id")
    calls = []

    class Actor:
        def __init__(self, *a):
            self.checkpoint_path = tmp_path / "dream/continuity/checkpoints/step-000003"

        def train(self, rows, condition, seed, steps, folder, run_id, resume):
            calls.append(resume)
            assert condition == "D" and steps == 3 and run_id == "id"
            return {"base_unchanged": True}

    monkeypatch.setattr(continuity, "EmbeddingRunner", Actor)
    monkeypatch.setattr(
        continuity,
        "compare_checkpoints",
        lambda *a: {"same_adapters": True, "same_optimizer": True},
    )
    result = continuity.verify_continuity(config, tmp_path)
    assert result["status"] == "passed" and result["extra_updates_executed"] == 1
    assert calls == [source / "checkpoints/step-000002"]
    assert read_json(source / "continuity-validation.json")["status"] == "passed"
    assert continuity.verify_continuity(config, tmp_path) == result
    assert len(calls) == 1
    monkeypatch.setattr(continuity, "identity", lambda *a: "changed")
    with pytest.raises(ValueError, match="incompatibles"):
        continuity.verify_continuity(config, tmp_path)
