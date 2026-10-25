import importlib
from types import SimpleNamespace

import pytest

from think_dream_lab.dream_generation import generate_dream


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("suffix", ["\n\n", "unexpected"])
def test_text_stop_keeps_sampled_merged_tokens_and_group_cache(monkeypatch, count, suffix):
    module = importlib.import_module("mlx_lm.generate")
    trainer = importlib.import_module("mlx_lm_lora.trainer.grpo_trainer")
    built, generators = [], []

    def build(model, prompt):
        built.append(prompt)
        return object()

    monkeypatch.setattr(trainer, "build_shared_prompt_cache", build)
    monkeypatch.setattr(trainer, "fork_prompt_cache", lambda m, c: {} if c else None)

    class Batch:
        def __init__(self, model, **kwargs):
            assert kwargs["stop_tokens"] == [[9]]
            self.live = set(range(count))
            self.step = 0
            self.closed = False
            generators.append(self)

        def insert(self, prompts, limits, *, caches, all_tokens):
            assert limits == [10] * count
            assert prompts == ([[7]] * count if count > 1 else [[6, 7]])
            assert all_tokens == ([[6]] * count if count > 1 else [[]])
            if count > 1:
                assert len({id(c) for c in caches}) == count
            return list(range(count))

        def next_generated(self):
            if not self.live:
                return []
            self.step += 1
            assert self.step <= 3, "Se generó más allá del cierre textual"
            return [SimpleNamespace(uid=i, token=self.step, finish_reason=None) for i in self.live]

        def remove(self, uids):
            self.live.difference_update(uids)

        def close(self):
            self.closed = True

    class Tokenizer:
        def __init__(self):
            self.eos_token_ids = {9}

        def encode(self, *args, **kwargs):
            pytest.fail("No se debe recodificar la respuesta ni buscar un cierre aislado")

        def decode(self, ids):
            pieces = {1: "<dream>x</dream><answer>x.</", 2: "answer", 3: ">" + suffix}
            return "".join(pieces[i] for i in ids)

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

    monkeypatch.setattr(module, "BatchGenerator", Batch)
    config = {"max_tokens": 10, "temperature": 0.8, "top_p": 0.95, "top_k": 0, "min_p": 0}
    model, tokenizer = Model(), Tokenizer()
    tokens, texts, indices = generate_dream(model, tokenizer, [6, 7], count, config, "</answer>")
    assert [t.tolist() for t in tokens] == [[1, 2, 3]] * count
    assert texts == ["<dream>x</dream><answer>x.</answer>" + suffix] * count
    assert indices == [0] * count and model.training and generators[0].closed
    assert len(built) == int(count > 1)
    assert tokenizer.eos_token_ids == {9}
    _, mask, lengths = trainer._prepare_grpo_inputs(
        ([[6, 7]], [[]], ["brief"], [""], None), tokens, indices
    )
    assert lengths.tolist() == [3] * count and mask.sum(axis=1).tolist() == [3] * count
