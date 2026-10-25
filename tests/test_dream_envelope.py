import pytest

from think_dream_lab.dream_envelope import boundary, compose, finish_stage
from think_dream_lab.dream_task import extract_blocks


def test_merged_token_boundary_retains_raw_and_ids():
    ids = [4, 5]
    r = finish_stage("I saw rain.</dream>\n<answer>discarded", ids, {9}, 100)
    assert r["content"] == "I saw rain."
    assert r["output_ids"] is ids
    assert r["raw"].endswith("discarded") and r["stop_reason"] == "</dream>"
    assert r["complete"]


def test_new_turn_stops_at_line_boundary_and_preserves_fact():
    r = finish_stage("I hid the coin.\n\nUser: another dream\nAssistant:", [1, 2, 3], {9}, 20)
    assert r["content"] == "I hid the coin." and r["stop_reason"] == "new_turn"
    assert boundary("I told the User: hello") is None


def test_program_closure_does_not_hide_budget_exhaustion():
    r = finish_stage("A sentence without ending", [1, 2, 3], {9}, 3)
    assert not r["complete"] and r["stop_reason"] == "budget"
    assert extract_blocks(compose(r["content"], "account"), "D")["format"] == 1


@pytest.mark.parametrize(
    "text,ids,complete", [("", [9], False), ("text", [1, 9], True), ("<dream>x", [1], False)]
)
def test_eos_and_empty_stages_are_distinguished(text, ids, complete):
    assert finish_stage(text, ids, {9}, 20)["complete"] is complete
