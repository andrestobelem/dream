from types import SimpleNamespace

import pytest

from think_dream_lab import base_check
from think_dream_lab.__main__ import parser
from think_dream_lab.config import freeze_model, load_config
from think_dream_lab.dream_workflow import DreamRunner
from think_dream_lab.storage import read_json, write_json


def test_pilot_base_is_the_default_and_does_not_reuse_smollm2_locks(tmp_path, monkeypatch):
    config = load_config("configs/dream-v4.yaml")
    assert config["model"]["id"] == "Qwen/Qwen3-1.7B-Base" and config["conditions"] == ["D"]
    assert str(parser().parse_args(["base-check"]).results) == "results/dream-v4-qwen"
    old = load_config("configs/dream-v4-smollm2.yaml")["model"]
    write_json(tmp_path / "model-lock.json", old)
    with pytest.raises(ValueError, match="no coincide"):
        freeze_model(config, tmp_path)
    assert read_json(tmp_path / "model-lock.json") == old
    calls = []
    monkeypatch.setattr(base_check.Runner, "__init__", lambda *a: calls.append(a[1]["model"]["id"]))
    DreamRunner(config, {}, tmp_path, object(), {})
    assert calls == ["Qwen/Qwen3-1.7B-Base"]


def test_dream_probe_never_adds_neutral_and_retains_a_format_failure(tmp_path, monkeypatch):
    calls = []

    class Probe:
        def __init__(self, *args):
            self.model = SimpleNamespace(parameters=dict)

        def generate(self, row, condition, count):
            assert condition == "D" and count == 1
            calls.append(row["brief"])
            return None, [{"reward": {"format": int(len(calls) != 4)}}]

    monkeypatch.setattr(base_check, "BaseProbe", Probe)
    monkeypatch.setattr(base_check, "freeze_model", lambda *args: {})
    monkeypatch.setattr(base_check, "fingerprint", lambda *args: "frozen")
    monkeypatch.setattr(base_check, "seed_all", lambda *args: None)
    base_check.check_base(load_config("configs/dream-v4.yaml"), tmp_path)
    result = read_json(tmp_path / "dream/base-validation.json")
    assert len(set(calls)) == 10
    assert set(result["format_by_condition"]) == {"D"}
    assert sum(result["format_by_condition"]["D"]) == 9
    assert result["format_preflight_passed"] is False
    assert result["base_unchanged"] and result["judge_calls"] == result["training_updates"] == 0
