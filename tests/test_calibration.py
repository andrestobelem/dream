from copy import deepcopy

import pytest

from think_dream_lab.calibration import (
    CRITERIA,
    metrics,
    require_calibration,
    select_thresholds,
    sha,
    split_sources,
)
from think_dream_lab.closed_judge import classifier_signature
from think_dream_lab.config import load_config
from think_dream_lab.storage import write_json


def observation(i, *, wrong=False, split="calibration", expected="preserved"):
    return {
        "id": str(i),
        "split": split,
        "expected": expected,
        "probabilities": {
            "preserved": 0.4 if wrong else 0.99,
            "omitted": 0.6 if wrong else 0.005,
            "contradicted": 0 if wrong else 0.005,
        },
        "choice_mass": 0.99,
    }


def test_threshold_selection_never_uses_validation_labels_and_excludes_unknowns():
    rows = [observation(i, wrong=i > 7) for i in range(10)]
    rows += [observation(10, split="validation", wrong=True)]
    critical = [observation(i) for i in range(12)]
    assert select_thresholds(rows, critical) == (0.25, 0.05)
    rows[-1]["expected"] = "contradicted"
    assert select_thresholds(rows, critical) == (0.25, 0.05)
    result = metrics([observation(0), observation(1, expected=None)], 0.15, 0.05)
    assert result["n"] == 1 and result["accuracy"] == result["coverage"] == 1
    critical[0]["expected"] = "omitted"
    assert select_thresholds(rows, critical) is None


def test_source_split_keeps_related_sentences_together_and_separates_contradictions():
    rows = [
        {"source_group": "first", "expected": "contradicted"},
        {"source_group": "first", "expected": "preserved"},
        {"source_group": "second", "expected": "contradicted"},
        {"source_group": "third", "expected": "omitted"},
    ]
    assignments = split_sources(rows)
    assert assignments["first"] != assignments["second"]
    assert assignments == split_sources(list(reversed(rows)))


def test_calibration_gate_rejects_wrong_judge_environment_and_changed_references(tmp_path):
    spec = deepcopy(load_config("configs/dream-v4.yaml")["judge"])
    source = tmp_path / "references.csv"
    source.write_text("reviewed references")
    path = tmp_path / "calibration.json"
    spec["classification"]["calibration_file"] = str(path)
    report = {
        "passed": True,
        "protocol": "assisted-pilot-v3",
        "criteria": CRITERIA,
        "classifier_signature": classifier_signature(spec),
        "thresholds": {"margin": 0.15, "min_choice_mass": 0.05},
        "environment": {"machine": "test"},
        "review_sources": {str(source): sha(source)},
        "calibration": {"accuracy": 1, "coverage": 1},
        "validation": {"accuracy": 1, "coverage": 1, "decided": 10},
        "critical": {"wrong": 0, "decided": 12},
        "likeness": {"passed": True, "mae": 0, "sanity": {"passed": True}},
        "judge_unchanged": True,
        "by_mode": {
            phase: {mode: {"accuracy": 1, "coverage": 1} for mode in ["compliance", "fidelity"]}
            for phase in ["calibration", "validation"]
        },
    }
    write_json(path, report)
    assert require_calibration(spec, {"machine": "test"})["passed"]
    with pytest.raises(ValueError, match="entorno"):
        require_calibration(spec, {"machine": "other"})
    spec["revision"] = "different"
    with pytest.raises(ValueError, match="no aprueba"):
        require_calibration(spec)
    spec = deepcopy(load_config("configs/dream-v4.yaml")["judge"])
    spec["classification"]["calibration_file"] = str(path)
    source.write_text("changed references")
    with pytest.raises(ValueError, match="referencias"):
        require_calibration(spec)
