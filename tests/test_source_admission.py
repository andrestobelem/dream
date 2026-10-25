from types import SimpleNamespace

import numpy as np
import pytest

from think_dream_lab.anchors import SourceIneligibleError, contract_brief, visible_anchors
from think_dream_lab.dream_workflow import admitted_training_order
from think_dream_lab.storage import read_json


@pytest.mark.parametrize("valid_source", [True, False])
def test_quote_repair_selects_original_text_including_typographical_errors(valid_source):
    text = "I used a Pprojection recording. I lay on the floor. I saw a square."
    anchors = [
        {
            "id": f"a{i}",
            "kind": "event",
            "requirement": requirement,
            "source_quote": "Projection recording",
        }
        for i, requirement in enumerate(
            ["I used a recording.", "I lay on the floor.", "I saw a square."], 1
        )
    ]
    calls = []

    def call(purpose, system, payload):
        calls.append((purpose, payload))
        if purpose == "anchor_extraction":
            return {"anchors": anchors}
        return {
            "anchors": [
                {k: a[k] for k in ["id", "kind", "requirement"]}
                | {"source_passage_id": "s1" if valid_source else "s999"}
                for a in anchors
            ]
        }

    judge = SimpleNamespace(spec={"rubric_version": "dream-v4"}, call=call)
    if not valid_source:
        with pytest.raises(ValueError, match="fragmento inexistente"):
            contract_brief(judge, {"reference": text})
    else:
        brief = contract_brief(judge, {"reference": text})
        assert len(visible_anchors(brief)) == 3 and "source_quote" not in brief
    assert calls[1][1]["passages"][0]["text"] == text


def test_admission_keeps_seed_order_and_records_exclusion_before_any_rollout(tmp_path, monkeypatch):
    rows = [{"problem_id": str(i), "source_id": f"source-{i}"} for i in range(8)]
    order = np.random.default_rng(11).permutation(len(rows)).tolist()
    calls = []

    def prepare(judge, selected, folder):
        row = selected[0]
        calls.append(row["problem_id"])
        if row["problem_id"] == str(order[0]):
            raise SourceIneligibleError("General note, not a specific experience")
        return {}

    monkeypatch.setattr("think_dream_lab.dream_workflow.prepare_briefs", prepare)
    judge = SimpleNamespace(spec={"id": "fixed"})
    assert admitted_training_order(judge, rows, 11, 3, tmp_path) == order[1:4]
    log = read_json(tmp_path / "task-rejections.json")
    assert set(log["rejected"]) == {str(order[0])}
    calls.clear()
    assert admitted_training_order(judge, rows, 11, 3, tmp_path) == order[1:4]
    assert str(order[0]) not in calls


def test_admission_never_hides_an_unexpected_extraction_failure(tmp_path, monkeypatch):
    def broken(*args):
        raise ValueError("Unexpected schema failure")

    monkeypatch.setattr("think_dream_lab.dream_workflow.prepare_briefs", broken)
    with pytest.raises(ValueError, match="Unexpected schema"):
        admitted_training_order(SimpleNamespace(spec={}), [{"problem_id": "x"}], 11, 1, tmp_path)
