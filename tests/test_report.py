import csv

import pytest

from think_dream_lab.report import paired_effects, report
from think_dream_lab.storage import append_jsonl, write_json


def test_factorial_uses_seed_replicates_and_paired_problems():
    runs = {}
    for seed in [11, 22]:
        for condition in "ABCD":
            rows = []
            for prompt in [condition, "neutral"]:
                for pid in ["one", "two"]:
                    for phase in ["before", "after"]:
                        rows.append(
                            {
                                "problem_id": pid,
                                "phase": phase,
                                "condition": prompt,
                                "reward": {"accuracy": int(phase == "after" and condition in "CD")},
                            }
                        )
            runs[(seed, condition)] = rows
    # Una semilla incompleta no se convierte en otra réplica del efecto factorial.
    runs[(33, "A")] = runs[(11, "A")]
    effects = paired_effects(runs)
    assert len(effects) == 12
    assert {e["seed"] for e in effects} == {11, 22}
    assert all(e["paired_problems"] == 2 for e in effects)
    assert all(e["value"] == (1 if e["effect"] == "instruction" else 0) for e in effects)


def test_partial_after_uses_only_matching_baseline(tmp_path):
    folder = tmp_path / "smoke"
    write_json(folder / "run.json", {"seed": 11, "kind": "smoke", "status": "running"})
    for pid, phase, correct in [("one", "before", 0), ("two", "before", 1), ("one", "after", 1)]:
        append_jsonl(
            folder / "responses.jsonl",
            {
                "problem_id": pid,
                "phase": phase,
                "condition": "A",
                "step": 0 if phase == "before" else 5,
                "candidate": 0,
                "reward": {"accuracy": correct, "format": 1},
                "tokens": 10,
                "latency_seconds": 1.0,
                "truncated": False,
                "family": "arithmetic",
                "generalization": False,
            },
        )
    report(tmp_path)
    with (tmp_path / "report" / "deltas.csv").open() as stream:
        delta = next(csv.DictReader(stream))
    assert int(delta["paired_problems"]) == 1
    assert float(delta["accuracy_delta"]) == pytest.approx(1)
