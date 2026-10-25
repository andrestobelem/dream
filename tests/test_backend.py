import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm_lora.trainer.grpo_trainer import (
    _get_token_logps,
    _grpo_objective,
    _grpo_value_and_grad,
    _prepare_grpo_inputs,
    calculate_rewards_and_advantages,
    grpo_loss,
)

from think_dream_lab import checkpoint
from think_dream_lab.backend import fingerprint, seed_all

pytestmark = pytest.mark.metal


class TinyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.projection = nn.Linear(4, 7, bias=False)

    def __call__(self, tokens):
        return self.projection(mx.cumsum(self.embedding(tokens), axis=1))


def setup(seed=11):
    seed_all(seed)
    model = TinyCausal()
    model.freeze()
    model.projection.unfreeze()
    reference = TinyCausal()
    reference.load_weights(list(tree_flatten(model.parameters())))
    reference.freeze()
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), reference.parameters(), optimizer.state)
    return model, reference, optimizer


def update(model, reference, optimizer):
    # Rollouts sintéticos sólo en la prueba numérica; objetivo real del backend fijado.
    completions = [mx.random.randint(0, 7, shape=(3,)), mx.random.randint(0, 7, shape=(2,))]
    fn = nn.value_and_grad(model, grpo_loss)
    (loss, tokens, _metrics), grads = _grpo_value_and_grad(
        fn,
        model,
        1,
        ref_model=reference,
        batch=([[1, 2]], [[]], ["x"], ["2"], None),
        completions=completions,
        completion_texts=["x", "y"],
        batch_indices=[0, 0],
        advantages=mx.array([1.0, -1.0]),
        reward_metrics={},
        beta=0.1,
        epsilon=0.2,
        epsilon_high=0.2,
        max_tokens=3,
        importance_sampling_level="token",
        grpo_loss_type="grpo",
    )
    assert mx.isfinite(loss).item()
    assert all(mx.all(mx.isfinite(g)).item() for _, g in tree_flatten(grads))
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss.item(), tokens.item()


def test_context_first_token_padding_and_eos():
    model, _, _ = setup()
    batch = ([[1, 2], [3]], [[], []], ["a", "b"], ["1", "2"], None)
    completions = [mx.array([4, 5, 0]), mx.array([6])]
    inputs, mask, lengths = _prepare_grpo_inputs(batch, completions, [0, 1])
    assert mask[0, :4].tolist() == [False, True, True, True]
    assert mask[1, :4].tolist() == [True, False, False, False]
    assert lengths.tolist() == [3, 1]
    actual = _get_token_logps(model, inputs, mask)
    for i, seq in enumerate([[1, 2, 4, 5, 0], [3, 6]]):
        logits = model(mx.array([seq[:-1]]))[0].astype(mx.float32)
        for position in range(len(seq) - 1):
            if mask[i, position].item():
                expected = logits[position, seq[position + 1]] - mx.logsumexp(logits[position])
                assert actual[i, position].item() == pytest.approx(expected.item(), abs=1e-5)
    assert actual[0, 1].item() != pytest.approx(
        _get_token_logps(model, mx.array([[6, 2, 4, 5, 0]]), mask[:1, :4])[0, 1].item()
    )
    assert mx.all(mx.where(mask, 0, actual) == 0).item()


def test_group_advantages_constant_and_separate():
    def rewards(**kwargs):
        return [0, 2, 1, 1]

    advantages, _metrics = calculate_rewards_and_advantages(
        [rewards],
        ["a", "a", "b", "b"],
        ["x"] * 4,
        ["0"] * 4,
        [None] * 4,
        [0, 0, 1, 1],
        [0, 1],
        [1],
    )
    assert advantages.tolist() == pytest.approx([-1 / 1.0001, 1 / 1.0001, 0, 0])
    assert mx.all(mx.isfinite(advantages)).item()


def test_on_policy_gradient_is_not_reference_ratio():
    logps = mx.array([[-1.0, -2.0]])
    mask = mx.array([[True, True]])

    def objective(values, ref):
        return _grpo_objective(
            values, ref, mask, mx.array([1.0]), 0.0, 0.2, 0.2, 2, "token", "grpo"
        )[0]

    for reference in [logps, mx.array([[-5.0, -6.0]])]:
        grad = mx.grad(objective)(logps, reference)
        assert grad.tolist()[0] == pytest.approx([-0.5, -0.5])
        assert objective(logps, reference).item() == pytest.approx(-1.0)


def test_masked_nonfinite_logits_and_constant_group():
    from mlx_lm_lora.trainer.grpo_trainer import _select_token_logps

    scores = _select_token_logps(
        mx.array([[[float("nan"), float("inf")], [1.0, 2.0]]]),
        mx.array([[0, 1]]),
        mx.array([[False, True]]),
    )
    assert mx.all(mx.isfinite(scores)).item()
    assert scores[0, 0].item() == 0

    def objective(logps):
        return _grpo_objective(
            logps,
            mx.stop_gradient(logps),
            mx.array([[True]]),
            mx.array([0.0]),
            0.1,
            0.2,
            0.2,
            1,
            "token",
            "grpo",
        )[0]

    loss, grad = mx.value_and_grad(objective)(mx.array([[-1.0]]))
    assert loss.item() == 0 and grad.item() == 0


def test_update_freezes_base_and_resume(tmp_path):
    model, reference, optimizer = setup()
    initial = fingerprint(model.trainable_parameters())
    base = fingerprint(model.embedding.parameters())
    update(model, reference, optimizer)
    assert fingerprint(model.trainable_parameters()) != initial
    assert fingerprint(model.embedding.parameters()) == base
    saved = checkpoint.save(tmp_path, model, optimizer, step=1, run_identity="same", extra={})
    expected_rng = (random.random(), np.random.random(), mx.random.uniform(shape=(3,)).tolist())
    expected_losses = [update(model, reference, optimizer) for _ in range(2)]
    expected_weights = fingerprint(model.trainable_parameters())
    expected_optimizer = fingerprint(optimizer.state)
    fresh, fresh_reference, fresh_optimizer = setup()
    checkpoint.load(saved, fresh, fresh_optimizer, "same")
    actual_rng = (random.random(), np.random.random(), mx.random.uniform(shape=(3,)).tolist())
    assert actual_rng == expected_rng
    assert [update(fresh, fresh_reference, fresh_optimizer) for _ in range(2)] == expected_losses
    assert fingerprint(fresh.trainable_parameters()) == expected_weights
    assert fingerprint(fresh_optimizer.state) == expected_optimizer
    # Repetir en un proceso nuevo verifica también la frontera de importación/runtime.
    output_path = tmp_path / "fresh-process.json"
    script = """
import json, random, runpy, sys
from pathlib import Path
ns = runpy.run_path(sys.argv[1])
model, reference, optimizer = ns['setup']()
ns['checkpoint'].load(Path(sys.argv[2]), model, optimizer, 'same')
random.random(); ns['np'].random.random(); ns['mx'].random.uniform(shape=(3,)).tolist()
losses = [ns['update'](model, reference, optimizer) for _ in range(2)]
Path(sys.argv[3]).write_text(json.dumps({'weights': ns['fingerprint'](model.trainable_parameters()),
                                      'optimizer': ns['fingerprint'](optimizer.state), 'losses': losses}))
"""
    subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(saved), str(output_path)],
        check=True,
        timeout=30,
    )
    import json

    process_result = json.loads(output_path.read_text())
    assert process_result["weights"] == expected_weights
    assert process_result["optimizer"] == expected_optimizer
    assert process_result["losses"] == [list(value) for value in expected_losses]
    with pytest.raises(ValueError, match="otra condición"):
        checkpoint.load(saved, fresh, fresh_optimizer, "different")


def test_actual_generation_keeps_sampled_ids():
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm_lora.trainer.grpo_trainer import generate_grpo

    seed_all(4)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            rms_norm_eps=1e-6,
            vocab_size=16,
            max_position_embeddings=64,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
    )

    class Tokenizer:
        eos_token_ids = frozenset({0})

        def encode(self, text, add_special_tokens=False):
            return [15]

        def decode(self, ids):
            return ",".join(map(str, ids))

    tokenizer = Tokenizer()
    model.train()
    tokens, texts, indices = generate_grpo(model, tokenizer, [[1, 2]], 4, 4, 1, "", 0.8, 0.95, 0, 0)
    assert model.training
    assert indices == [0, 0, 0, 0]
    assert len(tokens) == 4
    for ids, text in zip(tokens, texts):
        values = ids.tolist()
        assert 1 <= len(values) <= 4
        assert text == tokenizer.decode(values[:-1] if values[-1] == 0 else values)
