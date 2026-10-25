import pytest

from experiments.dpo_preferences_v20 import pair_gradient, sequence_arrays, sequence_logp


@pytest.fixture
def cpu_mlx():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield mx
    mx.set_default_device(previous)


def test_sampled_tokens_and_prompt_mask_are_preserved(cpu_mlx):
    x, mask = sequence_arrays({"prompt_ids": [3, 2, 7], "output_ids": [4, 8]})
    assert x.tolist() == [[3, 2, 7, 4, 8]]
    assert mask.tolist() == [[0, 0, 0, 1, 1]]
    assert mask[:, 1:].sum().item() == 2


def test_sequential_pair_gradient_matches_library_dpo(cpu_mlx):
    mx = cpu_mlx
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm_lora.trainer.dpo_trainer import dpo_loss

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(11, 5)
            self.linear = nn.Linear(5, 11)

        def __call__(self, x, cache=None):
            return self.linear(self.embedding(x))

    mx.random.seed(7)
    model = Model()
    chosen = {"prompt_ids": [1, 2, 3], "output_ids": [4, 5, 6]}
    rejected = {"prompt_ids": [1, 2, 3], "output_ids": [7, 8]}
    reference = (-5.0, -8.0)
    beta = 0.1

    def direct(m):
        c = sequence_logp(m, chosen)[None]
        r = sequence_logp(m, rejected)[None]
        _, cm = sequence_arrays(chosen)
        _, rm = sequence_arrays(rejected)
        return dpo_loss(
            c, r, mx.array([reference[0]]), mx.array([reference[1]]), cm, rm, beta, 50.0, "sigmoid"
        )[0]

    expected, eg = nn.value_and_grad(model, direct)(model)
    mx.eval(expected, eg)
    actual, ag, _ = pair_gradient(model, chosen, rejected, reference, beta)
    assert actual == pytest.approx(expected.item(), abs=1e-6)
    for (ka, a), (kb, b) in zip(tree_flatten(ag), tree_flatten(eg), strict=True):
        assert ka == kb and mx.allclose(a, b, atol=1e-6, rtol=1e-5).item()


def test_gradient_descent_improves_the_preferred_margin(cpu_mlx):
    mx = cpu_mlx
    import mlx.optimizers as optim
    from mlx import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 8)

        def __call__(self, x, cache=None):
            return self.embedding(x)

    mx.random.seed(5)
    model = Model()
    chosen = {"prompt_ids": [1, 2], "output_ids": [3, 4]}
    rejected = {"prompt_ids": [1, 2], "output_ids": [5, 6]}
    ref = (sequence_logp(model, chosen).item(), sequence_logp(model, rejected).item())
    before, grads, _ = pair_gradient(model, chosen, rejected, ref, 0.1)
    optimizer = optim.SGD(0.1)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    after, _, _ = pair_gradient(model, chosen, rejected, ref, 0.1)
    assert before == pytest.approx(0.69314718, abs=1e-6) and after < before
