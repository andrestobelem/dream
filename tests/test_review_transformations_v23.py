import pytest

from experiments.review_transformations_v23 import freeze
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import digest, read_json, write_json


def review_fixture(tmp_path):
    rubric = tmp_path / "rubric.md"
    rubric.write_text("fixed rubric")
    rows = [
        {"review_id": f"S{i:02d}", "request": "request", "dream": f"body {i}", "answer": "account"}
        for i in range(1, 31)
    ]
    notes = {
        "ratings": [
            {
                "review_id": r["review_id"],
                "experience": "fragmento",
                "material": "parcial",
                "treatment": "no_se",
                "literal_deviation": "no_se",
                "account": "no_se",
                "evidence": [r["dream"]],
                "note": "ambiguous; no inferred latent meaning",
                "case_hash": digest(r),
            }
            for r in rows
        ]
    }
    write_json(tmp_path / "packet.json", rows)
    write_json(tmp_path / "annotations.json", notes)
    write_json(
        tmp_path / "protocol.json",
        {"packet_hash": digest(rows), "rubric": str(rubric), "rubric_sha256": sha256(rubric)},
    )
    return tmp_path


def test_unsupported_evidence_cannot_be_frozen(tmp_path):
    root = review_fixture(tmp_path)
    notes = read_json(root / "annotations.json")
    notes["ratings"][0]["evidence"] = ["text not present in either source"]
    write_json(root / "annotations.json", notes)
    with pytest.raises(AssertionError):
        freeze(root)


def test_rubric_change_invalidates_review(tmp_path):
    root = review_fixture(tmp_path)
    (root / "rubric.md").write_text("different rubric")
    with pytest.raises(AssertionError):
        freeze(root)


def test_cannot_silently_replace_frozen_judgments(tmp_path):
    root = review_fixture(tmp_path)
    freeze(root)
    assert read_json(root / "review-frozen.json")["reviewed_presentations"] == 30
    notes = read_json(root / "annotations.json")
    notes["ratings"][0]["account"] = "sustituye"
    write_json(root / "annotations.json", notes)
    with pytest.raises(ValueError):
        freeze(root)
