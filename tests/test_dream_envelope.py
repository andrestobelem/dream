import pytest

from think_dream_lab.dream_envelope import boundary, compose, finish_stage
from think_dream_lab.dream_task import extract_blocks


def test_merged_token_boundary_retains_raw_and_ids():
    ids = [4, 5]
    r = finish_stage("I saw rain.</dream>\n<answer>discarded", ids, {9}, 100)
    assert r["content"] == "I saw rain."
    assert r["output_ids"] is ids
    assert r["raw"].endswith("discarded") and r["stop_reason"] == "</dream>"
    assert r["complete"]


def test_new_turn_stops_at_line_boundary_and_preserves_fact():
    r = finish_stage("I hid the coin.\n\nUser: another dream\nAssistant:", [1, 2, 3], {9}, 20)
    assert r["content"] == "I hid the coin." and r["stop_reason"] == "new_turn"
    assert boundary("I told the User: hello") is None


def test_program_closure_does_not_hide_budget_exhaustion():
    r = finish_stage("A sentence without ending", [1, 2, 3], {9}, 3)
    assert not r["complete"] and r["stop_reason"] == "budget"
    assert extract_blocks(compose(r["content"], "account"), "D")["format"] == 1


@pytest.mark.parametrize(
    "text,ids,complete", [("", [9], False), ("text", [1, 9], True), ("<dream>x", [1], False)]
)
def test_eos_and_empty_stages_are_distinguished(text, ids, complete):
    assert finish_stage(text, ids, {9}, 20)["complete"] is complete


def test_repeated_instruction_is_not_silently_rewritten():
    r = finish_stage(
        "I hid the coin.\n\nWrite the dream account in <answer> tags.\nUser: next", [1, 2], {9}, 20
    )
    assert r["stop_reason"] == "<answer>"
    assert r["content"].endswith("Write the dream account in")
    assert "User:" in r["discarded_suffix"]


def test_stream_stops_on_merged_turn_and_restores_model(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from think_dream_lab.dream_envelope import generate_stage

    module = importlib.import_module("mlx_lm.generate")
    trainer = importlib.import_module("mlx_lm_lora.trainer.grpo_trainer")
    monkeypatch.setattr(trainer, "fork_prompt_cache", lambda *args: None)
    seen = []

    class Batch:
        def __init__(self, *args, **kwargs):
            self.step, self.live, self.closed = 0, True, False
            seen.append(self)

        def insert(self, prompts, limits, **kwargs):
            assert prompts == [[7, 8]] and limits == [20]

        def next_generated(self):
            if not self.live:
                return []
            self.step += 1
            assert self.step <= 3
            return [SimpleNamespace(uid=1, token=self.step, finish_reason=None)]

        def remove(self, uids):
            assert uids == [1]
            self.live = False

        def close(self):
            self.closed = True

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

    class Tokenizer:
        def __init__(self):
            self.eos_token_ids = {9}

        def decode(self, ids):
            return "".join({1: "I hid the coin.", 2: "\nUs", 3: "er: unwanted"}[i] for i in ids)

    monkeypatch.setattr(module, "BatchGenerator", Batch)
    model = Model()
    r = generate_stage(
        model,
        Tokenizer(),
        [7, 8],
        {"max_tokens": 20, "temperature": 0.8, "top_p": 0.95, "min_p": 0, "top_k": 0},
    )
    assert r["content"] == "I hid the coin." and r["output_ids"] == [1, 2, 3]
    assert r["stop_reason"] == "new_turn" and model.training and seen[0].closed
