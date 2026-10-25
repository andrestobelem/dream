import pytest

from think_dream_lab.paired_report import compare
from think_dream_lab.storage import append_jsonl, digest, read_jsonl, write_json


def test_report_separates_format_gains_from_semantic_similarity(tmp_path):
    folder = tmp_path / "dream/paired"
    validation = [
        {"problem_id": str(i), "seeds": "Dream about a river.", "target_words": 50}
        for i in range(4)
    ]
    rows = {"validation": validation}
    write_json(folder / "frozen-training-pack.json", {"rows": rows})
    write_json(
        folder / "run.json",
        {
            "status": "complete",
            "dataset_hash": digest(rows),
            "identity": "id",
            "config": {"evaluation": {"samples_per_problem": 3}},
            "training_validation": {"adapters_changed": True, "base_unchanged": True},
        },
    )
    write_json(folder / "reload-validation.json", {"same_weights": True})
    for phase in ("before", "after"):
        for row in validation:
            for candidate in range(3):
                valid = phase == "after" or candidate == 0
                cosine = 0.5 if phase == "after" else 0.9 if valid else None
                append_jsonl(
                    folder / "responses.jsonl",
                    {
                        "phase": phase,
                        "problem_id": row["problem_id"],
                        "candidate": candidate,
                        "evaluation_seed": 42,
                        "prompt": "same prompt",
                        "truncated": False,
                        "reward": {
                            "format": int(valid),
                            "cosine": cosine,
                            "total": (1 + cosine) / 2 if valid else 0,
                            "answer_words": 50,
                            "relative_word_error": 0,
                            "scored_response": "model response",
                        },
                    },
                )
    result = compare(tmp_path)
    assert result["after"]["reward_mean_all"] > result["before"]["reward_mean_all"]
    assert (
        result["after"]["source_cosine_valid_mean"] < result["before"]["source_cosine_valid_mean"]
    )
    assert result["n_common_valid_pairs"] == 4
    assert result["common_valid_cosine_delta_mean"] == pytest.approx(-0.4)
    record = read_jsonl(folder / "responses.jsonl")[0]
    append_jsonl(folder / "responses.jsonl", record)
    with pytest.raises(ValueError, match="duplicados"):
        compare(tmp_path)
