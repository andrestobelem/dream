from pathlib import Path

import pytest

from think_dream_lab.config import load_config
from think_dream_lab.embedding_runner import EmbeddingRunner
from think_dream_lab.storage import read_json

SEEDS = {
    "characters": ["sister"],
    "places": ["river"],
    "objects": ["boat"],
    "actions": [],
    "emotions": [],
}


def test_runner_uses_seeds_and_only_answer_reward():
    runner = EmbeddingRunner.__new__(EmbeddingRunner)
    runner.config = load_config("configs/dream-v5.yaml")
    calls = []

    class Reward:
        def score(self, text):
            calls.append(text)
            return {"total": 0.7}

    runner.embedding_reward = Reward()
    row = {"seeds": SEEDS, "reference": "SECRET REFERENCE"}
    assert "SECRET REFERENCE" not in runner.render_prompt(row, "D")
    assert "Example" not in runner.render_prompt(row, "D")
    result = runner.completion_reward("two blocks", row, "D")
    assert result["total"] == 0.7
    assert calls == [runner.config["generation"].get("response_prefix", "") + "two blocks"]
    with pytest.raises(ValueError):
        runner.render_prompt(row, "A")


def test_smoke_interruption_and_resume_keep_protocol_and_private_actor_rows(tmp_path, monkeypatch):
    from think_dream_lab import config as cfg_module
    from think_dream_lab import dreambank_workflow as flow
    from think_dream_lab import embedding_runner, workflow

    cfg = load_config("configs/dream-v5.yaml")
    data = {
        s: [
            {"problem_id": f"{s}-{i}", "split": s, "author_group": "p", "reference": "PRIVATE"}
            for i in range(3)
        ]
        for s in ["train", "validation", "test"]
    }
    monkeypatch.setattr(
        flow,
        "load_reward",
        lambda *a: (data, {"hash": "corpus"}, type("Reward", (), {"identity": "bank"})()),
    )

    class Extractor:
        def __init__(self, *a):
            pass

        def extract(self, row):
            return {"problem_id": row["problem_id"], "seeds": SEEDS, "split": row["split"]}

    monkeypatch.setattr(flow, "SeedExtractor", Extractor)
    monkeypatch.setattr(cfg_module, "freeze_model", lambda *a: cfg["model"])
    monkeypatch.setattr(cfg_module, "identity", lambda *a: "same-protocol")
    checkpoints = [None]
    monkeypatch.setattr(workflow, "latest_checkpoint", lambda *a: checkpoints[0])
    monkeypatch.setattr(workflow, "reconcile_logs", lambda *a: None)
    reloads = []
    monkeypatch.setattr(workflow, "reload_check", lambda *a: reloads.append(a[-1]))

    class Actor:
        calls = 0
        checkpoint_path = Path("saved-checkpoint")

        def __init__(self, *args):
            pass

        def evaluate(self, rows, *args, **kwargs):
            assert all("reference" not in row for row in rows)

        def train(self, rows, condition, seed, steps, folder, run_id, resume=None):
            assert condition == "D" and steps == 3 and run_id == "same-protocol"
            assert all("reference" not in row for row in rows)
            Actor.calls += 1
            if Actor.calls == 1:
                checkpoints[0] = self.checkpoint_path
                raise KeyboardInterrupt()
            assert resume == self.checkpoint_path
            return {"steps_completed": 3, "base_unchanged": True, "adapters_changed": True}

    monkeypatch.setattr(embedding_runner, "EmbeddingRunner", Actor)
    with pytest.raises(KeyboardInterrupt):
        flow.smoke(cfg, tmp_path)
    assert read_json(tmp_path / "dream/smoke/run.json")["status"] == "interrupted"
    assert flow.smoke(cfg, tmp_path, resume=True)["status"] == "complete"
    assert reloads == [Path("saved-checkpoint")]
