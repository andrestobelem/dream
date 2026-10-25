from types import SimpleNamespace

import pytest

from think_dream_lab.anchors import PREFIX
from think_dream_lab.backend import Runner
from think_dream_lab.config import load_config
from think_dream_lab.dream_task import combine_scores
from think_dream_lab.dream_workflow import DreamRunner, summarize
from think_dream_lab.judge import Judge, protected_json
from think_dream_lab.storage import read_json
from think_dream_lab.uncertainty import UnscoredGroupError


def contract():
    return PREFIX + protected_json(
        {
            "version": "dream-v4",
            "anchors": [
                {"id": f"a{i}", "kind": "event", "requirement": f"Visible requirement {i}"}
                for i in range(1, 4)
            ],
        }
    )


def test_closed_fidelity_sees_scene_propositions_and_never_reserved_reference():
    calls = []

    def assess(requirements, candidate, **kwargs):
        calls.append((requirements, candidate, kwargs.get("mode", "compliance")))
        return {"score": 0.75 if len(calls) == 1 else None, "coverage": 1}

    judge = Judge.__new__(Judge)
    judge.spec = {"rubric_version": "dream-v4"}
    judge.closed_scorer = SimpleNamespace(assess=assess)
    judge.likeness = lambda answer: {"dream_likeness": 3}
    judge.call = lambda *args: pytest.fail("No se debe generar un veredicto libre")
    ratings = judge.rate(
        "HIDDEN REFERENCE",
        "I floated. My brother followed.",
        "I remember floating.",
        brief=contract(),
    )
    assert [a["requirement"] for a in calls[1][0]] == ["I floated.", "My brother followed."]
    assert calls[0][2] == "compliance" and calls[1][2] == "fidelity"
    assert "HIDDEN REFERENCE" not in str(calls)
    assert ratings["reference_alignment"] is None and ratings["retelling_fidelity"] is None
    result = combine_scores(ratings, 1, load_config("configs/dream-v4.yaml"))
    assert result["total"] is None and result["score_status"] == "abstained"
    assert result["content_compliance"] == 0.75 and "total" in result["unscored_dimensions"]


def test_neutral_omits_fidelity_without_becoming_an_abstention():
    calls = []
    judge = Judge.__new__(Judge)
    judge.spec = {"rubric_version": "dream-v4"}
    judge.closed_scorer = SimpleNamespace(
        assess=lambda req, answer: calls.append(req) or {"score": 0.75}
    )
    judge.likeness = lambda answer: {"dream_likeness": 3}
    ratings = judge.rate("hidden", "", "A scene.", brief=contract(), neutral=True)
    result = combine_scores(ratings, 1, load_config("configs/dream-v4.yaml"), neutral=True)
    assert len(calls) == 1 and ratings["retelling_fidelity"] is None
    assert result["quality"] == pytest.approx(0.75)
    assert result["total"] == pytest.approx(0.775) and result["score_status"] == "scored"


def test_skipping_dream_cannot_earn_quality_and_does_not_call_judge():
    runner = DreamRunner.__new__(DreamRunner)
    runner.config = load_config("configs/dream-v4.yaml")
    runner.briefs = {"p": contract()}
    runner.judge = SimpleNamespace(rate=lambda *a, **kw: pytest.fail("Formato incompleto"))
    result = runner.completion_reward(
        "<answer>I floated through a school.</answer>",
        {"problem_id": "p", "reference": "hidden"},
        "D",
    )
    assert result["total"] == 0 and result["quality"] is None
    assert result["score_status"] == "invalid_format"


@pytest.mark.parametrize("retries", [0, 2])
def test_abstention_retains_group_and_stops_before_advantages_or_gradients(
    tmp_path, monkeypatch, retries
):
    from mlx import nn
    from mlx_lm_lora.trainer import grpo_trainer

    def forbidden(*args, **kwargs):
        pytest.fail("No se deben calcular ventajas ni gradientes de un grupo sin score")

    monkeypatch.setattr(nn, "value_and_grad", lambda *a, **kw: forbidden)
    monkeypatch.setattr(grpo_trainer, "calculate_rewards_and_advantages", forbidden)
    monkeypatch.setattr(grpo_trainer, "_grpo_value_and_grad", forbidden)
    rows = [
        {"text": "scored", "reward": {"total": 0.8}},
        {"text": "uncertain", "reward": {"total": None}},
    ]
    runner = Runner.__new__(Runner)
    runner.config = {
        "training": {"group_size": 2, "max_unscored_groups": retries},
        "generation": {},
    }
    runner.optimizer, runner.model = object(), object()
    runner.checkpoint_path = "previous-checkpoint"
    runner.generate = lambda *a, **kw: ((None, ["scored", "uncertain"], [0, 0]), rows)
    with pytest.raises(UnscoredGroupError):
        runner.train([{"problem_id": "p"}], "D", 11, 1, tmp_path, "identity")
    stored = read_json(tmp_path / "unscored-groups/step-000001.json")
    assert stored["records"] == rows and stored["unscored_candidates"] == [1]
    assert stored["optimizer_updated"] is False and runner.checkpoint_path == "previous-checkpoint"
    assert len(list((tmp_path / "unscored-groups").glob("*.json"))) == retries + 1
    assert not (tmp_path / "steps.jsonl").exists()


def test_summary_reports_coverage_and_never_averages_abstention_as_zero():
    records = []
    for total in [0.8, None]:
        reward = {
            k: total
            for k in [
                "dream_likeness",
                "reference_alignment",
                "content_compliance",
                "retelling_fidelity",
                "quality",
                "total",
            ]
        }
        reward.update(
            format=1, answer_words=30, score_status="abstained" if total is None else "scored"
        )
        records.append({"reward": reward, "tokens": 10, "latency_seconds": 1, "truncated": False})
    summary = summarize(records)
    assert summary["total"] == 0.8 and summary["n_reward_scored"] == 1
    assert (
        summary["n_abstained"] == 1 and summary["n_scored_by_dimension"]["content_compliance"] == 1
    )


def test_training_metrics_exclude_not_evaluated_dimensions():
    from think_dream_lab.uncertainty import dimension_metrics

    result = dimension_metrics(
        [
            {"reward": {"quality": 0.75, "reference_alignment": None}},
            {"reward": {"quality": None, "reference_alignment": None}},
        ]
    )
    assert result["reward_quality"] == 0.75 and result["n_reward_quality"] == 1
    assert (
        result["n_reward_reference_alignment"] == 0 and "reward_reference_alignment" not in result
    )
