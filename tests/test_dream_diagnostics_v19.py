import pytest

from experiments.report_dream_v19 import expected_state, validate_frozen
from think_dream_lab.storage import digest, write_json


@pytest.mark.parametrize("part", ["packet", "notes"])
def test_review_tampering_fails_before_unblinding(tmp_path, part):
    packet = [{"review_id": "opaque", "answer": "A small boat did not reach land."}]
    notes = {"labels": [{"review_id": "opaque", "state": "nonarrival_unspecified"}]}
    write_json(
        tmp_path / "frozen.json", {"packet_hash": digest(packet), "annotations_hash": digest(notes)}
    )
    if part == "packet":
        packet[0]["answer"] = "A boat sank."
    else:
        notes["labels"][0]["state"] = "sank_before_shore"
    write_json(tmp_path / "packet.json", packet)
    write_json(tmp_path / "notes.json", notes)
    with pytest.raises(ValueError, match="congelada"):
        validate_frozen(tmp_path, "packet.json", "notes.json", "frozen.json")


def test_nonarrival_does_not_count_as_sinking_and_edit_switches_target():
    assert "nonarrival_unspecified" != expected_state("outcome", "anchor")
    assert expected_state("outcome", "anchor") == "sank_before_shore"
    assert expected_state("outcome", "edited") == "reached_without_sinking"
    assert expected_state("outcome", "paraphrase") == expected_state("outcome", "anchor")
    assert expected_state("location", "original") == "absent"
    assert expected_state("actor", "direct") == "absent"
