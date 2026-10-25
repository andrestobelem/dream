import pytest

from think_dream_lab.continuation import run
from think_dream_lab.continuation_report import paired_summary
from think_dream_lab.storage import write_json


def source_at(tmp_path, step=16):
    checkpoint = tmp_path / "checkpoint"
    write_json(checkpoint / "state.json", {"step": step})
    source = tmp_path / "source.json"
    write_json(
        source,
        {
            "status": "complete",
            "config": {"model": {"id": "original"}, "runtime": {}},
            "training_validation": {"checkpoint": str(checkpoint)},
        },
    )
    return source


def test_continuation_rejects_restart_and_unbounded_training_before_loading_models(tmp_path):
    source = source_at(tmp_path)
    for target in [0, 16, 33]:
        with pytest.raises(ValueError, match="16 pasos adicionales"):
            run({}, tmp_path / "output", source, target)
    assert not (tmp_path / "output").exists()


def test_continuation_rejects_changed_scientific_configuration(tmp_path):
    source = source_at(tmp_path)
    with pytest.raises(ValueError, match="configuración científica"):
        run({"model": {"id": "different"}, "runtime": {}}, tmp_path / "output", source, 32)


def test_pair_comparison_uses_only_positions_valid_in_both_models():
    row = {"problem_id": "dream", "author_group": "person"}
    keyed = {name: {} for name in ["step16", "step32"]}
    for candidate, (a, b) in enumerate([(0.6, 0.7), (None, 0.9), (0.8, None)]):
        for name, value in [("step16", a), ("step32", b)]:
            keyed[name][("dream", candidate)] = {
                "reward": {"format": int(value is not None), "cosine": value}
            }
    summary = paired_summary([row], keyed, "step16", "step32")
    assert summary["n_common_valid_pairs"] == 1
    assert summary["common_valid_cosine_delta_mean"] == pytest.approx(0.1)
    assert summary["cosine_first"] == 0.6 and summary["cosine_second"] == 0.7
    assert summary["reports_improved"] == 1 and summary["reports_worse"] == 0
