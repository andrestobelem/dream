"""El criterio exige mejora y todos los controles, incluyendo fallos y empates."""

import pytest

from experiments.criteria_preferences_v21 import promotion_decision, quality_result
from think_dream_lab.storage import read_json


def data():
    from pathlib import Path

    public = read_json(Path("docs/preferencias-v21-protocolo.json"))
    ratings = [
        {"problem_id": r["problem_id"], "candidate": i, "preference": "after"}
        for r in public["dev_rows"]
        for i in range(4)
    ]
    diagnostics = {
        "before": {"format_valid": 96, "motif_major": 2, "groups_fewer_than_two_episodes": 0},
        "after": {
            "format_valid": 96,
            "motif_major": 2,
            "groups_fewer_than_two_episodes": 0,
            "dependency_full": 16,
            "dependency_format": 18,
        },
        "variety_counts": {"after": 12, "before": 12},
        "all_reviews_frozen": True,
    }
    return public, ratings, diagnostics


def test_all_gates_must_pass():
    p, ratings, d = data()
    assert promotion_decision(p, ratings, d)["promote"]
    d["after"]["motif_major"] = 3
    assert not promotion_decision(p, ratings, d)["promote"]


def test_one_good_request_does_not_promote():
    p, ratings, d = data()
    for r in ratings[4:]:
        r["preference"] = "tie"
    result = promotion_decision(p, ratings, d)
    assert result["quality"]["net_mean"] == pytest.approx(1 / 24)
    assert not result["promote"]


def test_ties_or_uncertainty_are_zero_and_not_dropped():
    p, ratings, d = data()
    for i, r in enumerate(ratings):
        r["preference"] = "tie" if i % 2 else "no_se"
    result = promotion_decision(p, ratings, d)
    assert result["quality"]["interval_95"] == [0, 0]
    assert not result["promote"]


def test_missing_or_duplicate_comparison_is_rejected():
    p, ratings, _ = data()
    with pytest.raises(ValueError):
        quality_result(p, ratings[:-1])
    ratings[-1] = ratings[0]
    with pytest.raises(ValueError):
        quality_result(p, ratings)
