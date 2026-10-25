from pathlib import Path

import pytest

from experiments.review_v17_pairs import prepare
from experiments.summarize_v17_review import summarize
from think_dream_lab.storage import append_jsonl, digest, read_json, write_json


def source_fixture(root):
    pairs = []
    refs = []
    for group, discordant in enumerate([2, 2, 2, 0, 1, 1]):
        for candidate in range(3):
            pid = f"reference-{group}-{candidate}"
            pairs.append(
                {
                    "problem_id": pid,
                    "candidate": 0,
                    "author_group": str(group),
                    "delta": 0.1,
                    "judge_delta": -1 if candidate < discordant else 0,
                }
            )
            refs.append({"problem_id": pid, "reference": f"Reference {pid}"})
            for phase in ["before", "after"]:
                append_jsonl(
                    root / phase / "dream/holdout/checkpoint/responses.jsonl",
                    {
                        "problem_id": pid,
                        "candidate": 0,
                        "reward": {
                            "scored_response": f"<dream>Free fiction.</dream><answer>Account {phase} {pid}.</answer>"
                        },
                    },
                )
    write_json(root / "dream/report/comparison.json", {"common_valid_pairs": pairs})
    write_json(root / "before/dream/holdout/evaluation-pack.json", {"rows": refs})


def test_packet_is_balanced_and_hides_scores_and_provenance(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source_fixture(source)
    prepare(source, target)
    packet = read_json(target / "packet.json")
    key = read_json(target / "provenance-key.json")
    assert len(packet) == 12
    assert all(set(r) == {"review_id", "reference", "A", "B"} for r in packet)
    assert sum(r["A_phase"] == "before" for r in key) == 6
    assert sum(r["selection"] == "discordant" for r in key) == 8
    assert all(sum(p["author_group"] == str(g) for p in key) == 2 for g in range(6))
    notes = target / "annotations.json"
    notes.write_text('"Review already entered"')
    before = notes.read_bytes()
    prepare(source, target)
    assert notes.read_bytes() == before


@pytest.mark.parametrize("tamper", ["packet", "annotations"])
def test_tampered_review_is_rejected_before_reading_the_key(tmp_path, tamper):
    packet = [{"review_id": f"R{i:02d}", "reference": "original"} for i in range(12)]
    notes = {
        "independent_human_review": False,
        "ratings": [{"review_id": r["review_id"]} for r in packet],
    }
    frozen = {
        "packet_hash": digest(packet),
        "annotations_hash": digest(notes),
        "provenance_consulted_before_freeze": False,
    }
    write_json(tmp_path / "ratings-frozen.json", frozen)
    write_json(tmp_path / "manifest.json", {"packet_hash": digest(packet)})
    if tamper == "packet":
        packet[0]["reference"] = "changed"
    else:
        notes["ratings"][0]["review_id"] = "changed"
    write_json(tmp_path / "packet.json", packet)
    write_json(tmp_path / "annotations.json", notes)
    # No existe la clave: debe fallar por integridad, antes de intentar abrirla.
    with pytest.raises(ValueError, match="congeladas"):
        summarize(Path("unused"), tmp_path)
