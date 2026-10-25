import numpy as np
import pytest

from think_dream_lab.embedding_reward import DreamBankReward, Encoder, unit_vectors


class FakeEncoder:
    identity = "fake"

    def encode(self, texts):
        mapping = {"human": [1, 0], "other": [0, 1], "similar": [0.9, 0.1], "unrelated": [-1, 0]}
        return unit_vectors([mapping[t] for t in texts])


def bank():
    return DreamBankReward(
        [
            {"problem_id": "a", "split": "train", "reference": "human"},
            {"problem_id": "b", "split": "train", "reference": "other"},
        ],
        FakeEncoder(),
    )


def test_nearest_dream_rewards_similarity_without_copy_penalty():
    reward = bank()
    identical = reward.answer_score("human")
    assert identical["total"] == pytest.approx(1)
    assert identical["nearest_reference_id"] == "a"
    assert identical["total"] > reward.answer_score("similar")["total"]


def test_dream_content_is_not_scored():
    reward = bank()
    a = reward.score("<dream>anything at all</dream><answer>human</answer>")
    b = reward.score("<dream>a completely different experience</dream><answer>human</answer>")
    assert a == b
    assert reward.score("<dream>human</dream><answer>similar</answer>")["total"] < a["total"]
    assert reward.score("<answer>human</answer>")["total"] == 0


def test_reserved_references_and_invalid_vectors_are_rejected():
    with pytest.raises(ValueError, match="entrenamiento"):
        DreamBankReward(
            [{"problem_id": "v", "split": "validation", "reference": "human"}], FakeEncoder()
        )
    for vectors in ([[0, 0]], [[float("nan"), 1]], []):
        with pytest.raises(ValueError):
            unit_vectors(vectors)


def test_long_reports_use_every_chunk_and_cache(tmp_path):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def num_special_tokens_to_add(self, **kwargs):
            return 2

        def __call__(self, texts, max_length, **kwargs):
            result = {"input_ids": [], "attention_mask": [], "overflow_to_sample_mapping": []}
            for index, text in enumerate(texts):
                ids = [int(t) for t in text.split()]
                for start in range(0, len(ids), max_length - 2):
                    chunk = ids[start : start + max_length - 2]
                    result["input_ids"].append([0, *chunk, 0])
                    result["attention_mask"].append([1] * (len(chunk) + 2))
                    result["overflow_to_sample_mapping"].append(index)
            return result

        def pad(self, chunks, **kwargs):
            width = max(len(c["input_ids"]) for c in chunks)
            return {
                key: torch.tensor([c[key] + [0] * (width - len(c[key])) for c in chunks])
                for key in ("input_ids", "attention_mask")
            }

    class Model:
        calls = 0

        def __call__(self, input_ids, attention_mask):
            from types import SimpleNamespace

            self.calls += 1
            ids = input_ids.float()
            return SimpleNamespace(last_hidden_state=torch.stack([ids, ids**2 + 1], dim=-1))

    spec = {"max_tokens": 6, "batch_size": 2}
    encoder = Encoder(spec, tmp_path)
    encoder.model, encoder.tokenizer = Model(), Tokenizer()
    vectors = encoder.encode(["1 1 1 1", "1 1 1 1 9 9 9 9"])
    assert not np.allclose(vectors[0], vectors[1])
    assert encoder.model.calls == 2  # Tres fragmentos, de a dos.
    assert np.allclose(encoder.encode(["1 1 1 1 9 9 9 9"])[0], vectors[1])
    assert encoder.model.calls == 2
    reloaded = Encoder(spec, tmp_path)
    assert np.allclose(reloaded.encode(["1 1 1 1"])[0], vectors[0])
