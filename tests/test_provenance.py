import subprocess
import sys
from pathlib import Path

import pytest

from think_dream_lab.config import load_config
from think_dream_lab.provenance import freeze_source
from think_dream_lab.storage import read_json


def test_worker_can_import_the_exact_snapshot_without_changing_working_directory(tmp_path):
    frozen = freeze_source(tmp_path, {"value": 1})
    script = "import sys;sys.path.insert(0,sys.argv[1]);import think_dream_lab;print(think_dream_lab.__file__)"
    result = subprocess.run(
        [sys.executable, "-c", script, frozen["directory"]],
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()).is_relative_to(frozen["directory"])
    assert read_json(Path(frozen["config_path"])) == {"value": 1}
    assert freeze_source(tmp_path, {"value": 1}) == frozen


def test_modified_snapshot_is_rejected_instead_of_replaced(tmp_path):
    frozen = freeze_source(tmp_path, {"value": 1})
    target = Path(frozen["directory"]) / "think_dream_lab/dream_task.py"
    target.chmod(0o644)
    target.write_text("modified snapshot")
    with pytest.raises(ValueError, match="modificada"):
        freeze_source(tmp_path, {"value": 1})


def test_frozen_json_preserves_small_float_hyperparameters(tmp_path):
    config = load_config("configs/dream-v4.yaml")
    frozen = freeze_source(tmp_path, config)
    restored = load_config(frozen["config_path"])
    assert restored == config and restored["training"]["learning_rate"] == 1e-5
