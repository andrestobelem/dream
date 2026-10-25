from itertools import pairwise

import pytest

from think_dream_lab.anchors import PREFIX, assess
from think_dream_lab.evidence import passages, resolve_selection
from think_dream_lab.judge import protected_json


def test_passages_preserve_offsets_unicode_and_whitespace():
    text = "  café\n" + " palabra\t" * 95 + "última.  "
    chunks = passages(text)
    assert chunks[0]["start"] == 2
    assert chunks[-1]["end"] == len(text.rstrip())
    assert all(p["text"] == text[p["start"] : p["end"]] for p in chunks)
    assert all(a["end"] >= b["start"] for a, b in pairwise(chunks))
    assert passages(" \n ") == []


def test_selection_reconstructs_original_contiguous_evidence():
    text = "one two three four five six seven eight nine ten"
    chunks = passages(text, size=4, overlap=1)
    anchor = {"id": "a1"}
    value = {"id": "a1", "reason": "Present", "status": "preserved", "evidence_ids": ["s1", "s3"]}
    result = resolve_selection(value, anchor, chunks, text)
    assert result["evidence"] == text[: chunks[2]["end"]]
    assert result["evidence"] in text


@pytest.mark.parametrize(
    "ids,status",
    [
        (["s99"], "preserved"),
        (["s1", "s1"], "preserved"),
        ([], "preserved"),
        (["s1"], "omitted"),
        ([True], "preserved"),
    ],
)
def test_selection_cannot_invent_or_duplicate_evidence(ids, status):
    with pytest.raises(ValueError):
        resolve_selection(
            {"id": "a1", "reason": "Reason", "status": status, "evidence_ids": ids},
            {"id": "a1"},
            passages("scene"),
            "scene",
        )


def test_selection_never_sends_reserved_reference_or_accepts_free_quotes():
    class FakeJudge:
        def __init__(self):
            self.spec = {"evidence_mode": "passage_selection"}

        def call(self, purpose, system, payload):
            assert set(payload) == {"requirement", "passages"}
            assert all(set(p) == {"id", "text"} for p in payload["passages"])
            return {
                "id": payload["requirement"]["id"],
                "reason": "No requirement present",
                "status": "omitted",
                "evidence_ids": [],
            }

    anchors = [{"id": f"a{i}", "kind": "event", "requirement": f"event {i}"} for i in range(1, 4)]
    brief = PREFIX + protected_json({"version": "dream-v3", "anchors": anchors})
    result = assess(FakeJudge(), brief, "An unrelated scene.")
    assert result["score"] == 0
    assert result["evidence_selection"]["selections"] == {"a1": [], "a2": [], "a3": []}


def test_redundant_passage_metadata_cannot_change_selected_evidence():
    value = {"reason": "Present", "status": "preserved", "evidence_ids": ["s1"], "passage_id": "s1"}
    assert resolve_selection(value, {"id": "a1"}, passages("scene"), "scene")["evidence"] == "scene"
    value["passage_id"] = "s99"
    with pytest.raises(ValueError):
        resolve_selection(value, {"id": "a1"}, passages("scene"), "scene")
