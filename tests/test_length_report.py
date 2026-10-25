import pytest

from think_dream_lab.length_report import pair_status


def record(ids, truncated, valid):
    return {
        "prompt": "same",
        "prompt_ids": [1],
        "evaluation_seed": 5,
        "output_ids": ids,
        "truncated": truncated,
        "reward": {"format": valid, "cosine": 0.7 if valid else None},
    }


def test_recovery_requires_the_same_generated_prefix():
    old = record([1, 2], True, 0)
    rescued = pair_status(old, record([1, 2, 3], False, 1))
    different = pair_status(old, record([1, 4, 3], False, 1))
    assert rescued["format_rescued"] and rescued["prefix_matches"]
    assert not different["format_rescued"] and not different["prefix_matches"]


def test_a_closed_output_is_not_counted_as_recovered_from_truncation():
    status = pair_status(record([1, 2], False, 1), record([1, 2], False, 1))
    assert status["prefix_matches"] and not status["format_rescued"]


def test_different_prompt_is_rejected():
    with pytest.raises(ValueError, match="prompt"):
        pair_status(record([1], True, 0), {**record([1, 2], False, 1), "prompt": "changed"})
