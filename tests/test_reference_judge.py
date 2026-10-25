import pytest

from think_dream_lab.reference_judge import (
    KEYS,
    RULE,
    ReferenceJudge,
    complete_required_prefix,
    run,
)
from think_dream_lab.storage import digest


@pytest.mark.parametrize("bad", [True, 2.5, -1, 5, None])
def test_reference_scores_require_integer_ratings(bad):
    judge = ReferenceJudge.__new__(ReferenceJudge)
    judge.call = lambda *args: {**dict.fromkeys(KEYS, 3), "events": bad, "reason": "Reason"}
    with pytest.raises(ValueError):
        judge.assess("Reference", "Account")


def test_reference_assessment_sends_only_the_two_accounts():
    calls = []
    judge = ReferenceJudge.__new__(ReferenceJudge)

    def fake_call(purpose, rule, payload):
        calls.append((purpose, rule, payload))
        return {**dict.fromkeys(KEYS, 3), "reason": "Central events preserved."}

    judge.call = fake_call
    judge.assess("Reference account", "Candidate account")
    assert calls == [
        (
            "reference-closeness-v17",
            RULE,
            {"reference": "Reference account", "candidate": "Candidate account"},
        )
    ]


def test_rubric_change_is_rejected_before_loading_a_model(tmp_path):
    assert digest(RULE) != "changed"
    with pytest.raises(ValueError, match="rúbrica"):
        run({"expanded_experiment": {"judge_rubric_hash": "changed"}}, tmp_path)


def test_complete_ratings_are_kept_when_an_extra_copy_is_truncated():
    raw = '{"characters":1,"events":2,"relations":3,"overall":2,"reason":"A complete reason.","candidate":"extra text cut off'
    assert complete_required_prefix(raw) == {
        "characters": 1,
        "events": 2,
        "relations": 3,
        "overall": 2,
        "reason": "A complete reason.",
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"characters":1,"events":2,"relations":3,"overall":2,"reason":"cut off',
        '{"characters":1,"events":2,"relations":3,"overall":',
        '{"characters":1,"events":2,"relations":3,"overall":true,"reason":"x"}',
        '{"characters":1,"characters":2,"relations":3,"overall":2,"reason":"x"}',
        '{"characters":1,"extra":2,"relations":3,"overall":2,"reason":"x"}',
    ],
)
def test_prefix_normalization_never_infers_missing_or_ambiguous_fields(raw):
    with pytest.raises(ValueError):
        complete_required_prefix(raw)
