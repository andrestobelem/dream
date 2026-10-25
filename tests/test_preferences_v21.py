"""Barreras de preparación: notas íntegras, trayectorias enteras y lote viable."""

from pathlib import Path

import pytest

from experiments.preferences_v21 import build, freeze
from think_dream_lab.storage import digest, read_json, write_json


def fixture_pack(tmp_path, monkeypatch, clear=12):
    monkeypatch.chdir(tmp_path)
    root = Path("results/dream-v21")
    write_json(
        root / "protocol.json",
        {"training": {"max_total_sequence_tokens": 1536, "minimum_clear_preferences": 12}},
    )
    packets, keys, ratings, samples = [], [], [], []
    for i in range(24):
        rid = f"N{i + 1:02d}"
        candidates = {}
        for j, side in enumerate("ABCD"):
            # Alterna la dirección de longitud sin modificar preferencias.
            words = 3 if (i + j) % 2 else 5
            sample = {
                "sample_id": f"{rid}-{side}",
                "prompt_ids": [5, 6],
                "output_ids": [20 + j, 30 + j, 40 + j],
                "dream": " ".join([side] * words),
                "answer": "answer",
                "format_valid": True,
            }
            samples.append(sample)
            candidates[side] = {
                "dream": sample["dream"],
                "format_valid": True,
                "sequence_tokens": 5,
            }
        row = {"review_id": rid, "request": "request", "candidates": candidates}
        packets.append(row)
        keys.append(
            {
                "review_id": rid,
                "problem_id": f"problem-{i}",
                **{s + "_id": f"{rid}-{s}" for s in "ABCD"},
            }
        )
        ratings.append(
            {
                "review_id": rid,
                "case_hash": digest(row),
                "preference": "clear" if i < clear else "tie",
                "chosen": "A",
                "rejected": "B",
                "confidence": "high",
                "note": "scene vs comment",
                "candidate_notes": dict.fromkeys("ABCD", "reviewed"),
            }
        )
    write_json(root / "preferences/packet.json", packets)
    write_json(root / "preferences/key.json", keys)
    write_json(root / "preferences/annotations.json", {"ratings": ratings})
    path = root / "train-pool/samples.jsonl"
    path.parent.mkdir(parents=True)
    import json

    path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    return root, samples


def test_preserves_sampled_tokens_and_excludes_ties(tmp_path, monkeypatch):
    root, samples = fixture_pack(tmp_path, monkeypatch)
    freeze(root)
    build(root)
    pack = read_json(root / "preferences/training-pack.json")
    assert pack["ready"] and len(pack["pairs"]) == 12 and len(pack["excluded"]) == 12
    assert pack["length_directions"] == {"longer": 6, "shorter": 6}
    assert pack["pairs"][0]["chosen"] == samples[0]
    assert pack["pairs"][0]["rejected"] == samples[1]


def test_insufficient_clear_pairs_do_not_become_ready(tmp_path, monkeypatch):
    root, _ = fixture_pack(tmp_path, monkeypatch, clear=10)
    freeze(root)
    build(root)
    assert not read_json(root / "preferences/training-pack.json")["ready"]


def test_changed_review_after_freeze_is_rejected(tmp_path, monkeypatch):
    root, _ = fixture_pack(tmp_path, monkeypatch)
    freeze(root)
    notes = read_json(root / "preferences/annotations.json")
    notes["ratings"][0]["chosen"] = "C"
    write_json(root / "preferences/annotations.json", notes)
    with pytest.raises(AssertionError):
        build(root)


@pytest.mark.parametrize("bad", ["format", "length"])
def test_cannot_freeze_invalid_or_overlong_pair(tmp_path, monkeypatch, bad):
    root, _ = fixture_pack(tmp_path, monkeypatch)
    packet = read_json(root / "preferences/packet.json")
    c = packet[0]["candidates"]["A"]
    c["format_valid"] = bad != "format"
    c["sequence_tokens"] = 1537 if bad == "length" else 5
    write_json(root / "preferences/packet.json", packet)
    notes = read_json(root / "preferences/annotations.json")
    notes["ratings"][0]["case_hash"] = digest(packet[0])
    write_json(root / "preferences/annotations.json", notes)
    with pytest.raises(AssertionError):
        freeze(root)
