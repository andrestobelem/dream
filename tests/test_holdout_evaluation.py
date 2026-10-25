from collections import Counter

import pytest

from think_dream_lab.holdout_evaluation import EvaluationRunner, select_rows, validate_brief


def test_selection_is_balanced_fixed_and_excludes_previously_evaluated_reports():
    rows = [{"problem_id": f"{g}-{i}", "author_group": str(g)} for g in range(6) for i in range(10)]
    seen = {"0-1", "2-5", "4-0"}
    selected = select_rows(rows, seen, 4, 20261025)
    assert len(selected) == 24
    assert Counter(r["author_group"] for r in selected) == dict.fromkeys(map(str, range(6)), 4)
    assert not seen & {r["problem_id"] for r in selected}
    assert selected == select_rows(list(reversed(rows)), seen, 4, 20261025)
    with pytest.raises(ValueError, match="suficientes"):
        select_rows(rows, {f"0-{i}" for i in range(8)}, 4, 20261025)


def test_seed_rule_preserves_natural_short_prompts_without_response_tags():
    brief = "Dream about a cold house, empty bedrooms, a mattress in a bathtub, family members and finding privacy."
    assert validate_brief({"brief": brief}) == brief
    for bad in (
        {"brief": "Dream about a river"},
        {"brief": brief + " <answer>"},
        {"brief": brief, "reference": "human report"},
    ):
        with pytest.raises(ValueError):
            validate_brief(bad)


def test_evaluation_runner_cannot_train_or_create_an_optimizer():
    runner = EvaluationRunner.__new__(EvaluationRunner)
    with pytest.raises(RuntimeError, match="prohibido"):
        runner.train([])
    with pytest.raises(RuntimeError, match="optimizador"):
        runner.prepare_training(11)


def test_base_name_normalization_matches_lora_wrappers_without_changing_weights():
    from think_dream_lab.holdout_evaluation import base_names_in_lora_layout

    weight, other = object(), object()
    raw = {"layers.0.q_proj.weight": weight, "layers.0.norm.weight": other}
    adapters = {"layers.0.q_proj.lora_a", "layers.0.q_proj.lora_b"}
    normalized = base_names_in_lora_layout(raw, adapters)
    assert normalized == {"layers.0.q_proj.linear.weight": weight, "layers.0.norm.weight": other}
    assert raw["layers.0.q_proj.weight"] is weight
    with pytest.raises(ValueError, match="colisión"):
        base_names_in_lora_layout({**raw, "layers.0.q_proj.linear.weight": object()}, adapters)
