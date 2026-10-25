"""La revisión incluye todos los casos y exige los mismos contextos y semillas."""

import json

import pytest

from experiments.evaluate_preferences_v21 import packets
from think_dream_lab.storage import read_json, write_json


def setup_samples(tmp_path):
    rows = [{"problem_id": str(i), "seeds": "request"} for i in range(24)]
    write_json(tmp_path / "protocol.json", {"dev_rows": rows})
    for phase in ["before", "after"]:
        write_json(tmp_path / phase / "run.json", {"status": "complete"})
        samples = [
            {
                "problem_id": r["problem_id"],
                "candidate": i,
                "seed": i,
                "prompt_ids": [1, 2, 3],
                "dream": f"{phase}-{r['problem_id']}-{i}",
                "format_valid": True,
            }
            for r in rows
            for i in range(4)
        ]
        (tmp_path / phase / "samples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in samples)
        )


def test_full_packets_and_stable_keys(tmp_path):
    setup_samples(tmp_path)
    packets(tmp_path)
    key = read_json(tmp_path / "evaluation/quality-key.json")
    packet = read_json(tmp_path / "evaluation/quality-packet.json")
    assert len(key) == len(packet) == 96
    assert sum(k["A_phase"] == "before" for k in key) == 48
    for k, r in zip(key, packet, strict=True):
        assert r["A"] == f"{k['A_phase']}-{k['problem_id']}-{k['candidate']}"
    assert all(
        len(r["A"]) == len(r["B"]) == 4
        for r in read_json(tmp_path / "evaluation/variety-packet.json")
    )
    packets(tmp_path)
    assert read_json(tmp_path / "evaluation/quality-key.json") == key


def test_mismatched_seed_does_not_form_a_pair(tmp_path):
    setup_samples(tmp_path)
    path = tmp_path / "after/samples.jsonl"
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    samples[0]["seed"] = 99
    path.write_text("".join(json.dumps(r) + "\n" for r in samples))
    with pytest.raises(AssertionError):
        packets(tmp_path)
