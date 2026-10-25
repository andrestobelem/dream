import numpy as np
import pytest

from think_dream_lab.config import load_config
from think_dream_lab.dreambank_task import render_prompt
from think_dream_lab.paired_experiment import PairedReward, actor_rows, evaluate, load_pack
from think_dream_lab.storage import digest, write_json


class Encoder:
    identity = "fake"

    def encode(self, texts):
        vectors = {"reference-a": [1, 0], "reference-b": [0, 1], "answer-a": [1, 0]}
        return np.array([vectors[t] for t in texts], dtype=np.float32)


def test_reward_uses_associated_reference_and_leaves_dream_free():
    def row(pid, reference, split):
        return {
            "problem_id": pid,
            "reference": reference,
            "reference_hash": digest(reference),
            "split": split,
            "target_words": 10,
        }

    reward = PairedReward(
        {
            "train": [row("a", "reference-a", "train")],
            "validation": [row("b", "reference-b", "validation")],
        },
        Encoder(),
    )
    first = reward.score("<dream>anything</dream><answer>answer-a</answer>", "a")
    different = reward.score("<dream>anything</dream><answer>answer-a</answer>", "b")
    assert first["total"] == pytest.approx(1)
    assert different["total"] == pytest.approx(0.5)
    assert (
        reward.score("<dream>unrelated free content</dream><answer>answer-a</answer>", "a") == first
    )
    assert reward.score("<answer>answer-a</answer>", "a")["total"] == 0


def test_approved_pack_and_actor_input_do_not_leak_references(tmp_path):
    config = load_config("configs/dream-v6.yaml")
    rows = {"train": [], "validation": []}
    for split, n in [("train", 8), ("validation", 4)]:
        for i in range(n):
            reference = f"Human report {split} {i} about a river crossing."
            words = len(reference.split())
            rows[split].append(
                {
                    "problem_id": f"{split}-{i}",
                    "author_group": f"{split}-person-{i}",
                    "split": split,
                    "reference": reference,
                    "reference_hash": digest(reference),
                    "seeds": "Dream about a river.",
                    "target_words": words,
                    "prompt": render_prompt(
                        "Dream about a river.", response_prefix="<dream>\n", target_words=words
                    ),
                }
            )
    fixture = {"rows": rows, "dataset_hash": digest(rows)}
    path = tmp_path / "pack.json"
    write_json(path, fixture)
    config["training_pack"] = {"path": str(path), "hash": digest(fixture)}
    pack = load_pack(config)
    assert len(pack["rows"]["train"]) == 8 and len(pack["rows"]["validation"]) == 4
    for row in actor_rows(pack["rows"]["train"]):
        assert "reference" not in row and row["target_words"] > 0


def test_before_after_share_generation_seeds_and_completed_evaluation_is_reused(
    tmp_path, monkeypatch
):
    from think_dream_lab import paired_experiment as module

    seeds = []
    monkeypatch.setattr(module, "seed_all", seeds.append)

    class Runner:
        def __init__(self):
            self.config = {"smoke": {"steps": 16}}
            self.calls = 0

        def generate(self, row, condition, count):
            self.calls += 1
            return None, [
                {
                    "problem_id": row["problem_id"],
                    "candidate": i,
                    "reward": {
                        "format": 1,
                        "cosine": 0.4,
                        "total": 0.7,
                        "answer_words": 10,
                        "relative_word_error": 0,
                    },
                    "truncated": False,
                }
                for i in range(count)
            ]

    runner = Runner()
    rows = [{"problem_id": "reserved"}]
    before = evaluate(runner, rows, tmp_path, "before", 4242, 3)
    assert evaluate(runner, rows, tmp_path, "before", 4242, 3) == before
    after = evaluate(runner, rows, tmp_path, "after", 4242, 3)
    assert before == after and seeds[0] == seeds[1] and runner.calls == 2
