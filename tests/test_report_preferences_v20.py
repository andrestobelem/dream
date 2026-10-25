import pytest

from experiments.dpo_logratio_v20 import token_regions
from experiments.report_preferences_v20 import frozen_review, paired_summary
from think_dream_lab.storage import digest, write_json


def test_interval_resamples_requests_instead_of_two_samples_independently(tmp_path):
    packet = [{"review_id": f"E{i}", "A": "scene A", "B": "scene B"} for i in range(4)]
    notes = {
        "ratings": [
            {
                "review_id": r["review_id"],
                "preference": "A",
                "confidence": "high",
                "case_hash": digest(r),
            }
            for r in packet
        ]
    }
    key = [
        {
            "review_id": r["review_id"],
            "problem_id": f"prompt-{i // 2}",
            "A_phase": "after" if i < 2 else "before",
            "B_phase": "before" if i < 2 else "after",
        }
        for i, r in enumerate(packet)
    ]
    write_json(tmp_path / "quality-packet.json", packet)
    write_json(tmp_path / "quality-annotations.json", notes)
    write_json(
        tmp_path / "quality-frozen.json",
        {"packet_hash": digest(packet), "annotations_hash": digest(notes)},
    )
    write_json(tmp_path / "quality-key.json", key)
    result = paired_summary(tmp_path, "quality")
    assert result["comparisons"] == 4 and result["exploratory_interval"]["groups"] == 2
    assert result["exploratory_interval"]["unit"] == "development_prompt"
    assert result["net_preference"] == 0


@pytest.mark.parametrize("field", ["packet", "notes"])
def test_frozen_text_or_preference_change_is_rejected(tmp_path, field):
    packet = [{"review_id": "E1", "A": "scene A", "B": "scene B"}]
    notes = {"ratings": [{"review_id": "E1", "preference": "A", "case_hash": digest(packet[0])}]}
    write_json(
        tmp_path / "quality-frozen.json",
        {"packet_hash": digest(packet), "annotations_hash": digest(notes)},
    )
    if field == "packet":
        packet[0]["A"] = "different scene"
    else:
        notes["ratings"][0]["preference"] = "B"
    write_json(tmp_path / "quality-packet.json", packet)
    write_json(tmp_path / "quality-annotations.json", notes)
    with pytest.raises(ValueError, match="congelada"):
        frozen_review(tmp_path, "quality")


def test_boundary_spanning_sampled_token_is_kept_whole_in_diagnostic():
    class Tokenizer:
        def decode(self, ids):
            pieces = {1: "scene", 2: "</dream>\n<answer>first", 3: " rest", 4: "</answer>\n"}
            return "".join(pieces[i] for i in ids)

    ids = [1, 2, 3, 4]
    assert token_regions(Tokenizer(), ids) == ["dream", "boundary", "answer", "boundary"]
    assert ids == [1, 2, 3, 4]
