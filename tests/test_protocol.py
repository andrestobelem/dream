import copy

import pytest

from think_dream_lab.config import load_config
from think_dream_lab.data import evaluate, freeze, generate
from think_dream_lab.prompts import CONDITIONS, process_tag, render
from think_dream_lab.rewards import extract_answer, score, valid_format


@pytest.mark.parametrize(
    "text,expected,accuracy,format_",
    [
        ("<think>work</think><answer>12</answer>", 12, 1, 1),
        (" <think>work</think>\n<answer> -12 </answer> ", -12, 1, 1),
        ("<think>work</think><answer>+12</answer>", 12, 1, 1),
        ("<think>work</think><answer>2</answer>", 12, 0, 1),
        ("<think>12</think><answer>112</answer>", 12, 0, 1),
        ("<think>12</think>", 12, 0, 0),
        ("<think> </think><answer>12</answer>", 12, 1, 0),
        ("hello <think>work</think><answer>12</answer>", 12, 1, 0),
        ("<dream>work</dream><answer>12</answer>", 12, 1, 0),
        ("<think>work</dream><answer>12</answer>", 12, 1, 0),
        ("<think>x</think><think>y</think><answer>12</answer>", 12, 1, 0),
        ("<think><think>x</think></think><answer>12</answer>", 12, 1, 0),
        ("<think>x</think><answer>12</answer><answer>12</answer>", 12, 0, 0),
        ("<think>x</think><answer><answer>12</answer></answer>", 12, 0, 0),
        ("<answer>12</answer>", 12, 1, 0),
        ("<think>x</think><answer>the answer is 12</answer>", 12, 0, 1),
        ("<think>x</think><answer>12.0</answer>", 12, 0, 1),
        ("<think>x</think><answer>１２</answer>", 12, 0, 1),
        ("<think>2 < 3 and 4 > 1</think><answer>12</answer>", 12, 1, 1),
        ("<think>x<foo>y</foo></think><answer>12</answer>", 12, 1, 0),
        ("<think>x</think><answer>12</answer> extra", 12, 1, 0),
    ],
)
def test_rewards(text, expected, accuracy, format_):
    reward = score(text, expected, "think")
    assert (reward.accuracy, reward.format, reward.total) == (accuracy, format_, accuracy + format_)


def test_symmetric_and_neutral():
    for condition in CONDITIONS:
        tag = process_tag(condition)
        assert score(f"<{tag}>work</{tag}><answer>7</answer>", 7, tag).total == 2
    assert valid_format("Here is work. <answer>7</answer>", "neutral")
    assert not valid_format("<answer>7</answer> more", "neutral")
    assert extract_answer("<answer>1e3</answer>") is None


def test_literal_prompts():
    prompts = {c: render("Calculate 1 + 2.", c) for c in CONDITIONS}
    assert (
        prompts["A"].replace("<think>", "<dream>").replace("</think>", "</dream>") == prompts["B"]
    )
    assert prompts["A"].replace("first thinks", "first dreams") == prompts["C"]
    assert (
        prompts["C"].replace("<think>", "<dream>").replace("</think>", "</dream>") == prompts["D"]
    )
    assert "think" not in render("1 + 2", "neutral")
    assert "dream" not in render("1 + 2", "neutral")
    assert all(p.endswith("Assistant:") for p in prompts.values())


def test_data_reproducible_and_disjoint(tmp_path):
    config = load_config("configs/pilot.yaml")
    first, second = generate(config), generate(config)
    assert first == second
    ids = [r["problem_id"] for rows in first.values() for r in rows]
    assert len(ids) == len(set(ids)) == 2700
    for split, rows in first.items():
        assert all(r["split"] == split and evaluate(r["tree"]) == r["answer"] for r in rows)
    assert sum(r["generalization"] for r in first["test"]) == 250
    assert {r["family"] for r in first["train"]}.isdisjoint(
        {r["family"] for r in first["test"] if r["generalization"]}
    )
    rows, hash_ = freeze(config, tmp_path)
    assert rows == first
    assert freeze(config, tmp_path)[1] == hash_
    changed = copy.deepcopy(config)
    changed["data"]["seed"] += 1
    with pytest.raises(ValueError):
        freeze(changed, tmp_path)
    path = tmp_path / "data" / "train.jsonl"
    path.write_text(path.read_text().replace("Calculate", "Changed", 1))
    with pytest.raises(ValueError, match="modificados"):
        freeze(config, tmp_path)


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_calibration_has_enough_unique_data(difficulty):
    config = load_config("configs/pilot.yaml")
    config["data"]["difficulty"] = difficulty
    rows = generate(config)
    assert sum(len(split) for split in rows.values()) == 2700


def test_time_budget_is_operational_not_training_identity():
    from think_dream_lab.config import identity

    config = load_config("configs/pilot.yaml")
    before = identity(config, {}, "data", "A", 11)
    config["runtime"]["budget_seconds"] += 1000
    assert identity(config, {}, "data", "A", 11) == before
    config["training"]["beta"] += 0.1
    assert identity(config, {}, "data", "A", 11) != before


def test_report_empty(tmp_path):
    from think_dream_lab.report import report

    report(tmp_path)
    assert (tmp_path / "report" / "informe.md").exists()


def test_cli_requires_steps():
    from think_dream_lab.__main__ import parser

    with pytest.raises(SystemExit):
        parser().parse_args(["experiment"])


def test_resume_recovers_logs(tmp_path):
    from think_dream_lab.storage import write_json
    from think_dream_lab.workflow import reconcile_logs

    write_json(
        tmp_path / "checkpoints" / "step-000001" / "state.json",
        {
            "step": 1,
            "extra": {
                "metric": {"step": 1, "loss": 0.1},
                "responses": [{"step": 1, "candidate": 0, "phase": "rollout"}],
            },
        },
    )
    reconcile_logs(tmp_path)
    reconcile_logs(tmp_path)
    assert len((tmp_path / "steps.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "responses.jsonl").read_text().splitlines()) == 1
