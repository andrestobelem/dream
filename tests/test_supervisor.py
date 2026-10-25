import time

import pytest

from think_dream_lab.config import load_config
from think_dream_lab.storage import read_json, write_json
from think_dream_lab.supervisor import supervise


def test_exhausted_budget_does_not_start_worker(tmp_path):
    config = load_config("configs/pilot.yaml")
    write_json(tmp_path / "budget.json", {"used_seconds": 2700, "runs": []})
    assert supervise(config, tmp_path, ["doctor"]) == 3
    assert read_json(tmp_path / "budget.json")["runs"] == []


def test_effective_deadline_terminates_worker(tmp_path):
    config = load_config("configs/pilot.yaml")
    config["runtime"]["budget_seconds"] = 0.5
    start = time.monotonic()
    assert supervise(config, tmp_path, ["doctor", "--results", str(tmp_path)]) == 3
    elapsed = time.monotonic() - start
    assert elapsed < 3
    ledger = read_json(tmp_path / "budget.json")
    assert ledger["used_seconds"] > 0
    assert ledger["runs"][0]["reason"] == "time_budget"
    assert ledger["runs"][0]["status"] == "incomplete"


def test_concurrent_supervisor_is_rejected(tmp_path):
    import fcntl

    config = load_config("configs/pilot.yaml")
    with (tmp_path / ".session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="otra ejecución"):
            supervise(config, tmp_path, ["doctor"])
