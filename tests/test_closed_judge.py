import json
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from think_dream_lab.closed_judge import (
    EXAMPLES,
    STATES,
    ClosedJudge,
    aggregate_passages,
    aggregate_requirements,
    classification_messages,
    decide,
)


def test_demonstrations_rotate_with_labels_and_never_include_the_test_answer():
    for rotation in range(3):
        messages, mapping = classification_messages(
            "UNSEEN REQUIREMENT", "UNSEEN PASSAGE", rotation
        )
        for index, (claim, text, state) in enumerate(EXAMPLES):
            user, assistant = messages[1 + 2 * index : 3 + 2 * index]
            assert json.loads(user["content"]) == {"requirement": claim, "passage": text}
            assert mapping[assistant["content"]] == state
        assert messages[-1]["role"] == "user"
        assert json.loads(messages[-1]["content"])["passage"] == "UNSEEN PASSAGE"


def test_uncertainty_is_not_a_zero_score():
    uncertain = decide(
        dict(zip(STATES, [0.4, 0.35, 0.25], strict=True)), 0.8, margin=0.15, min_choice_mass=0.05
    )
    assert uncertain["status"] == "abstain"
    result = aggregate_requirements([{"status": "preserved"}, uncertain])
    assert result["score"] is None
    assert result["conditional_score"] == 1 and result["coverage"] == 0.5
    result = aggregate_requirements([uncertain])
    assert result["conditional_score"] is None and result["coverage"] == 0


def test_low_label_mass_and_ties_abstain():
    assert (
        decide(
            {"preserved": 0.98, "omitted": 0.01, "contradicted": 0.01},
            0.001,
            margin=0.1,
            min_choice_mass=0.05,
        )["status"]
        == "abstain"
    )
    assert (
        decide(
            {"preserved": 0.5, "omitted": 0.5, "contradicted": 0}, 1, margin=0, min_choice_mass=0
        )["status"]
        == "abstain"
    )


def test_conflicting_passages_abstain_without_punishing_generator():
    assert aggregate_passages([{"status": "preserved"}, {"status": "contradicted"}]) == "abstain"
    assert aggregate_passages([{"status": "preserved"}, {"status": "abstain"}]) == "abstain"
    assert aggregate_passages([{"status": "omitted"}, {"status": "preserved"}]) == "preserved"


def test_rotations_cancel_pure_preference_for_letter_a():
    scorer = ClosedJudge.__new__(ClosedJudge)
    scorer.options = {"margin": 0.1, "min_choice_mass": 0.05}

    def biased(requirement, passage, rotation):
        return {
            "probabilities": {s: 0.98 if s == STATES[rotation] else 0.01 for s in STATES},
            "choice_mass": 1,
            "key": str(rotation),
        }

    scorer.distribution = biased
    result = scorer.classify("claim", "passage")
    assert result["status"] == "abstain"
    assert list(result["probabilities"].values()) == pytest.approx([1 / 3] * 3)


def test_direct_logits_are_normalized_without_generation_and_cached(tmp_path):
    import mlx.core as mx

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "P"

        def encode(self, text, **kwargs):
            return [0] if text == "P" else [0, {"A": 1, "B": 2, "C": 3}[text[-1]]]

    calls = []

    def model(ids):
        calls.append(ids.tolist())
        return mx.array([[[0.0, 1.0, 2.0, 3.0]]])

    judge = SimpleNamespace(
        spec={
            "classification": {
                "labels": ["A", "B", "C"],
                "rotations": 3,
                "margin": 0.1,
                "min_choice_mass": 0.05,
            }
        },
        tokenizer=Tokenizer(),
        model=model,
        root=tmp_path,
    )
    scorer = ClosedJudge(judge)
    r = scorer.distribution("claim", "passage", 0)
    expected = [math.exp(v) / sum(math.exp(k) for k in [1, 2, 3]) for v in [1, 2, 3]]
    assert list(r["probabilities"].values()) == pytest.approx(expected)
    assert r["choice_mass"] == pytest.approx(
        sum(math.exp(v) for v in [1, 2, 3]) / sum(math.exp(v) for v in [0, 1, 2, 3])
    )
    assert r["generated_tokens"] == 0 and r["label_ids"] == {"A": 1, "B": 2, "C": 3}
    assert scorer.distribution("claim", "passage", 0) == r and len(calls) == 1
    assert (
        scorer.distribution("claim", "passage", 0, use_cache=False)["label_logits"]
        == r["label_logits"]
    )
    assert len(calls) == 2


def test_cached_prefix_preserves_full_context_and_is_not_modified(tmp_path, monkeypatch):
    import mlx.core as mx
    from mlx_lm_lora.trainer import grpo_trainer

    built = []

    def build(model, ids):
        built.append(ids)
        return {"sum": sum(ids[:-1])}

    monkeypatch.setattr(grpo_trainer, "build_shared_prompt_cache", build)
    monkeypatch.setattr(grpo_trainer, "fork_prompt_cache", lambda model, cache: dict(cache))

    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return "FULL" + messages[-1]["content"] if add_generation_prompt else "PREFIX"

        def encode(self, text, **kwargs):
            prefix = list(range(1, 17))
            if text == "PREFIX":
                return prefix
            letter = text[-1] if text[-1] in "ABC" else None
            return (
                prefix
                + [17, 19 if "other" in text else 18]
                + (["ABC".index(letter) + 1] if letter else [])
            )

    def model(ids, cache=None):
        value = sum(ids.tolist()[0]) + (cache["sum"] if cache else 0)
        if cache is not None:
            cache["sum"] = value
        return mx.array([[[0.0, value, value + 1, value + 2]]])

    spec = {
        "classification": {
            "labels": ["A", "B", "C"],
            "rotations": 3,
            "margin": 0.15,
            "min_choice_mass": 0.05,
            "prefix_cache": True,
        }
    }
    cached = ClosedJudge(
        SimpleNamespace(spec=spec, model=model, tokenizer=Tokenizer(), root=tmp_path)
    )
    first = cached.distribution("claim", "passage", 0)
    uncached_spec = deepcopy(spec)
    uncached_spec["classification"]["prefix_cache"] = False
    full = ClosedJudge(
        SimpleNamespace(spec=uncached_spec, model=model, tokenizer=Tokenizer(), root=tmp_path)
    )
    assert first["label_logits"] == full.distribution("claim", "passage", 0)["label_logits"]
    assert first["cached_prefix_tokens"] == 15
    assert cached.distribution("claim", "other", 0)["label_logits"] != first["label_logits"]
    assert len(built) == 1 and list(cached.prefix_templates.values()) == [
        {"sum": sum(range(1, 16))}
    ]


def test_whole_account_retains_context_and_exact_source_offsets():
    scorer = ClosedJudge.__new__(ClosedJudge)
    scorer.options = {"scope": "whole_account"}
    calls = []
    scorer.classify = lambda requirement, text: calls.append(text) or {"status": "preserved"}
    text = "I met my sister. " + "We walked together. " * 40 + "She woke me up."
    result = scorer.assess([{"id": "a", "requirement": "My sister woke me up."}], text)
    assert calls == [text]
    assert result["source_passages"] == [
        {"id": "account", "start": 0, "end": len(text), "text": text}
    ]
