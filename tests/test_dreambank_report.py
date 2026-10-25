from think_dream_lab.dreambank_report import report
from think_dream_lab.storage import canonical, digest, write_json


def test_report_counts_complete_checkpoints_without_claiming_finished_training(tmp_path):
    folder = tmp_path / "dream"
    data = folder / "data"
    data.mkdir(parents=True)
    rows = {
        s: [{"problem_id": s, "reference": "local report"}] for s in ("train", "validation", "test")
    }
    for split, records in rows.items():
        (data / f"{split}.jsonl").write_text(canonical(records[0]) + "\n")
    write_json(data / "manifest.json", {"hash": digest(rows), "counts": {s: 1 for s in rows}})
    write_json(folder / "smoke/run.json", {"status": "interrupted", "steps_requested": 3})
    result = report(tmp_path)
    assert result["steps_completed"] == 0 and result["pilot_status"] == "interrupted"
    checkpoint = folder / "smoke/checkpoints"
    write_json(checkpoint / "latest.json", {"directory": "step-000001"})
    write_json(checkpoint / "step-000001/state.json", {"step": 1})
    result = report(tmp_path)
    assert result["steps_completed"] == 1 and result["pilot_status"] == "interrupted"
    assert result["subjective_experience_claimed"] is False
    assert "checkpoint completo: 1/3" in (folder / "report/report.md").read_text()
