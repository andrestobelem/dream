import copy

import pytest

from think_dream_lab.anchors import (
    PREFIX,
    assess,
    contract_brief,
    validate_anchors,
    validate_assessment,
    visible_anchors,
)
from think_dream_lab.config import load_config
from think_dream_lab.dream_task import combine_scores
from think_dream_lab.judge import Judge, protected_json


@pytest.fixture
def anchors():
    return [
        {"id": "a1", "kind": "setting", "requirement": "The narrator is in a school."},
        {"id": "a2", "kind": "relationship", "requirement": "The narrator is with their brother."},
        {"id": "a3", "kind": "emotion", "requirement": "The narrator feels afraid."},
    ]


def brief(anchors):
    return PREFIX + protected_json({"version": "dream-v3", "anchors": anchors})


def test_judge_receives_exactly_visible_requirements_and_no_reference(anchors):
    judge = Judge.__new__(Judge)
    calls = []
    candidate = "school brother afraid"

    def call(purpose, system, payload):
        calls.append(payload)
        return {
            "items": [
                {"id": a["id"], "status": "preserved", "evidence": "school"}
                for a in payload["requirements"]
            ]
        }

    judge.call = call
    assert assess(judge, brief(anchors), candidate)["score"] == 1
    assert calls == [{"requirements": [a], "candidate": candidate} for a in anchors]


def test_formula_penalizes_contradiction_more_than_omission(anchors):
    base = [{"id": a["id"], "status": "preserved", "evidence": "scene"} for a in anchors]
    scores = []
    for status, evidence in [("preserved", "scene"), ("omitted", ""), ("contradicted", "scene")]:
        items = copy.deepcopy(base)
        items[-1].update(status=status, evidence=evidence)
        scores.append(validate_assessment({"items": items}, anchors, "scene")["score"])
    assert scores == pytest.approx([1, 2 / 3, 0.5])
    all_contradicted = [
        {"id": a["id"], "status": "contradicted", "evidence": "scene"} for a in anchors
    ]
    assert validate_assessment({"items": all_contradicted}, anchors, "scene")["score"] == 0


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "unknown", "hallucinated_evidence", "invalid_status"]
)
def test_invalid_assessment_fails_instead_of_punishing_actor(anchors, fault):
    items = [{"id": a["id"], "status": "omitted", "evidence": ""} for a in anchors]
    if fault == "duplicate":
        items[1]["id"] = "a1"
    elif fault == "missing":
        items.pop()
    elif fault == "unknown":
        items[1]["id"] = "a9"
    elif fault == "hallucinated_evidence":
        items[1].update(status="preserved", evidence="invented")
    else:
        items[1]["status"] = "maybe"
    with pytest.raises(ValueError):
        validate_assessment({"items": items}, anchors, "actual candidate")


def test_source_evidence_is_checked_and_not_exposed_in_prompt(anchors):
    judge = Judge.__new__(Judge)
    reference = "A reserved sentence about the school and my brother."
    raw = [{**a, "source_quote": "school"} for a in anchors]
    judge.call = lambda *args: {"anchors": raw}
    result = contract_brief(judge, {"reference": reference})
    assert visible_anchors(result) == anchors
    assert "source_quote" not in result and reference not in result
    raw[0]["source_quote"] = "this did not occur"
    with pytest.raises(ValueError, match="referencia"):
        validate_anchors(raw, reference)


def test_empty_answer_is_missing_not_judged(anchors):
    judge = Judge.__new__(Judge)
    judge.call = lambda *args: pytest.fail("No se debe consultar al juez sin relato")
    result = assess(judge, brief(anchors), "")
    assert result["score"] == 0 and result["counts"]["omitted"] == 3


def test_v3_reward_uses_compliance_not_reserved_reference():
    cfg = load_config("configs/dream-v3.yaml")
    ratings = {
        "dream_likeness": 4,
        "reference_alignment": 0,
        "retelling_fidelity": 4,
        "content_compliance": 4,
    }
    score = combine_scores(ratings, 1, cfg)
    assert score["quality"] == pytest.approx(1)
    assert score["total"] == pytest.approx(1)
    ratings["reference_alignment"] = 4
    assert combine_scores(ratings, 1, cfg)["total"] == score["total"]
    assert combine_scores(ratings, 1, cfg, neutral=True)["quality"] == pytest.approx(1)
    ratings["content_compliance"] = 0
    assert combine_scores(ratings, 1, cfg)["quality"] == pytest.approx(0.7)


def test_contract_rejects_hidden_extra_fields(anchors):
    anchors[0]["hidden_reference"] = "reserved"
    with pytest.raises(ValueError):
        visible_anchors(brief(anchors))


def test_one_bounded_evidence_repair_preserves_original_error(anchors):
    judge = Judge.__new__(Judge)
    calls = []

    def call(purpose, system, payload):
        calls.append((purpose, payload))
        anchor = payload["requirements"][0]
        if purpose == "content_compliance":
            return {"items": [{"id": anchor["id"], "status": "preserved", "evidence": "invented"}]}
        assert payload["previous_output"]["items"][0]["evidence"] == "invented"
        return {"items": [{"id": anchor["id"], "status": "omitted", "evidence": ""}]}

    judge.call = call
    assert assess(judge, brief(anchors), "no relevant scene")["score"] == 0
    assert len(calls) == 6
    judge.call = lambda *args: {"items": []}
    with pytest.raises(ValueError):
        assess(judge, brief(anchors), "scene")
