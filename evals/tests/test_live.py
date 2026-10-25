import json

import pytest
import yaml
from dream_evals import live
from inspect_ai import eval as inspect_eval
from inspect_ai.model import Model


@pytest.fixture
def live_inputs(tmp_path):
    config = yaml.safe_load((live.PROJECT / "configs/dream-v4.yaml").read_text())
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    results = tmp_path / "results"
    data = results / "dream/data"
    data.mkdir(parents=True)
    rows = [{"problem_id": f"p{i}", "reference": f"Human reference {i}."} for i in range(2)]
    (data / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return config, config_file, results, rows


def test_live_solver_runs_one_local_batch_and_preserves_samples(live_inputs, tmp_path, monkeypatch):
    config, config_file, results, rows = live_inputs
    starts = []

    class Process:
        returncode = 0

        async def communicate(self):
            folder = results / "dream/evaluations/fixture"
            folder.mkdir(parents=True)
            manifest = {
                "kind": "dream_evaluation",
                "status": "complete",
                "identity": "id",
                "implementation_hash": "core",
                "condition": "D",
                "seed": 4242,
                "config": config,
                "briefs": {r["problem_id"]: "visible brief" for r in rows},
                "actor_unchanged": True,
                "judge_unchanged": True,
                "training_updates": 0,
            }
            (folder / "run.json").write_text(json.dumps(manifest))
            reward = {
                "dream_likeness": 0.75,
                "reference_alignment": None,
                "content_compliance": 0.5,
                "retelling_fidelity": 0.25,
                "quality": 0.575,
                "format": 1,
                "total": 0.6175,
                "scoring_protocol": "dream-v4",
                "score_status": "scored",
                "unscored_dimensions": ["reference_alignment"],
            }
            outputs = [
                {
                    "problem_id": r["problem_id"],
                    "phase": "before",
                    "condition": "D",
                    "step": 0,
                    "candidate": 0,
                    "prompt": "actual prompt " + r["problem_id"],
                    "text": "<dream>A.</dream><answer>B.</answer>",
                    "reward": reward,
                    "tokens": 12,
                    "truncated": False,
                }
                for r in rows
            ]
            (folder / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in outputs))
            return b"local worker complete", None

    async def start(*args, **kwargs):
        starts.append(args)
        assert args[1:4] == ("-m", "think_dream_lab", "evaluate")
        return Process()

    async def forbidden(*args, **kwargs):
        pytest.fail("El modelo de control de Inspect no debe generar")

    monkeypatch.setattr(live.asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(live, "core_manifest_hash", lambda project: "core")
    monkeypatch.setattr(Model, "generate", forbidden)
    task = live.dream_live(str(results), str(config_file), limit=2)
    log = inspect_eval(
        task,
        model=live.CONTROLLER_MODEL,
        display="none",
        log_dir=str(tmp_path / "logs"),
        max_samples=2,
        notification=False,
        ctl_server=False,
    )[0]
    assert log.status == "success" and len(log.samples) == 2 and len(starts) == 1
    assert log.stats.model_usage == {}
    assert all(s.metadata["local_generation_new"] for s in log.samples)
    assert all(s.metadata["training_updates"] == 0 for s in log.samples)
    assert all(
        s.messages[0].content == "actual prompt " + s.metadata["problem_id"] for s in log.samples
    )
    assert all(
        s.scores["recorded_closed_scores"].metadata["replayed"] is False for s in log.samples
    )


def test_live_rejects_changed_sources_before_starting_worker(live_inputs, tmp_path, monkeypatch):
    _, config_file, results, _ = live_inputs
    task = live.dream_live(str(results), str(config_file))
    config = yaml.safe_load(config_file.read_text())
    config["generation"]["temperature"] = 0.123
    config_file.write_text(yaml.safe_dump(config))

    async def forbidden(*args, **kwargs):
        pytest.fail("No debe arrancar un worker con un protocolo cambiado")

    monkeypatch.setattr(live.asyncio, "create_subprocess_exec", forbidden)
    log = inspect_eval(
        task,
        model=live.CONTROLLER_MODEL,
        display="none",
        log_dir=str(tmp_path / "changed"),
        notification=False,
        ctl_server=False,
    )[0]
    assert log.status == "error" and "Cambió" in log.error.message
