import json

import pytest
from dream_evals.flows import dream_flow
from dream_evals.tasks import (
    FIELDS,
    REPLAY_MODEL,
    dream_results,
    load_run,
    score_values,
    select_records,
)
from inspect_ai import eval as inspect_eval
from inspect_ai.model import Model


@pytest.fixture
def results(tmp_path):
    dream = tmp_path / "dream"
    folder = dream / "smoke"
    folder.mkdir(parents=True)
    (dream / "data").mkdir()
    (folder / "run.json").write_text(
        json.dumps(
            {
                "identity": "fixture-identity",
                "kind": "dream_smoke",
                "status": "complete",
                "seed": 11,
                "config": {
                    "judge": {"id": "local-judge", "revision": "fixed", "rubric_version": "fixture"}
                },
            }
        )
    )
    (dream / "data" / "validation.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "dream-one",
                "reference": "A reserved human dream account.",
            }
        )
        + "\n"
    )
    reward = dict(zip(FIELDS, [0.75, 0.5, 0.25, 0.575, 1, 0.6175]))
    row = {
        "problem_id": "dream-one",
        "phase": "after",
        "condition": "D",
        "step": 5,
        "candidate": 0,
        "family": "lucid",
        "prompt": "A compact dream brief.",
        "text": "<dream>A scene.</dream><answer>A retelling.</answer>",
        "reward": reward,
        "tokens": 12,
        "truncated": False,
    }
    (folder / "responses.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path, row


def test_replay_never_generates_and_preserves_scores(results, tmp_path, monkeypatch):
    root, row = results

    async def forbidden(*args, **kwargs):
        raise AssertionError("La reproducción no debe generar ni llamar a un proveedor.")

    monkeypatch.setattr(Model, "generate", forbidden)
    logs = inspect_eval(
        dream_results(str(root)),
        model=REPLAY_MODEL,
        display="none",
        log_dir=str(tmp_path / "logs"),
        max_samples=1,
        notification=False,
        ctl_server=False,
    )
    assert len(logs) == 1 and logs[0].status == "success"
    assert logs[0].stats.model_usage == {}
    sample = logs[0].samples[0]
    assert sample.output.completion == row["text"]
    assert next(iter(sample.scores.values())).value == score_values(row["reward"])
    assert sample.metadata["generation_rerun"] is False


def test_changed_source_invalidates_snapshot(results):
    root, row = results
    _, _, _, snapshot = select_records(root, "smoke", "D", "after")
    row["reward"]["quality"] = 0.2
    (root / "dream/smoke/responses.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="cambiaron"):
        dream_results(str(root), snapshot=snapshot)


def test_empty_cell_and_diagnostics_are_not_replayed(results):
    root, _ = results
    with pytest.raises(ValueError, match="celda vacía"):
        dream_results(str(root), condition="A")
    with pytest.raises(ValueError, match="fuera de diagnostics"):
        load_run(root, "diagnostics/old")


def test_invalid_scores_fail_instead_of_becoming_zero(results):
    _, row = results
    row["reward"]["quality"] = float("nan")
    with pytest.raises(ValueError, match="Score inválido"):
        score_values(row["reward"])


def test_flow_uses_only_available_complete_cells(results):
    root, _ = results
    incomplete = root / "dream/runs/D-seed-22"
    incomplete.mkdir(parents=True)
    (incomplete / "run.json").write_text(json.dumps({"status": "running"}))
    spec = dream_flow(str(root))
    assert len(spec.tasks) == 1
    assert spec.tasks[0].model == REPLAY_MODEL
    assert spec.flow_metadata["new_training"] is False


def test_closed_replay_excludes_unknown_dimensions_from_means(results, tmp_path, monkeypatch):
    import math

    root, row = results
    path = root / "dream/smoke/run.json"
    manifest = json.loads(path.read_text())
    manifest["kind"] = "dream_evaluation"
    manifest["config"]["judge"]["rubric_version"] = "dream-v4"
    path.write_text(json.dumps(manifest))
    row["reward"].update(
        reference_alignment=None,
        content_compliance=0.5,
        scoring_protocol="dream-v4",
        score_status="scored",
        unscored_dimensions=["reference_alignment"],
    )
    other = json.loads(json.dumps(row))
    other["candidate"] = 1
    other["text"] += " A second sample."
    for field in ["total", "quality", "retelling_fidelity", "content_compliance"]:
        other["reward"][field] = None
    other["reward"].update(
        score_status="abstained",
        unscored_dimensions=[
            "reference_alignment",
            "total",
            "quality",
            "retelling_fidelity",
            "content_compliance",
        ],
    )
    (root / "dream/smoke/responses.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(other) + "\n"
    )

    async def forbidden(*args, **kwargs):
        pytest.fail("El replay no debe llamar a modelos")

    monkeypatch.setattr(Model, "generate", forbidden)
    log = inspect_eval(
        dream_results(str(root)),
        model=REPLAY_MODEL,
        display="none",
        log_dir=str(tmp_path / "closed-logs"),
        notification=False,
        ctl_server=False,
    )[0]
    assert log.status == "success" and log.stats.model_usage == {}
    total = next(s for s in log.results.scores if s.name == "total")
    assert total.scored_samples == 1 and total.unscored_samples == 1
    assert next(iter(total.metrics.values())).value == pytest.approx(row["reward"]["total"])
    reference = next(s for s in log.results.scores if s.name == "reference_alignment")
    assert reference.scored_samples == 0 and reference.unscored_samples == 2
    assert math.isnan(next(iter(reference.metrics.values())).value)
    metadata = log.samples[1].scores["recorded_closed_scores"].metadata
    assert metadata["source_reward"]["total"] is None
