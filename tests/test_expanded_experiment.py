from pathlib import Path

import pytest

from think_dream_lab.dreambank_task import render_prompt
from think_dream_lab.expanded_experiment import load_expanded_pack
from think_dream_lab.storage import digest, write_json


def fixture_config(tmp_path):
    brief = "Dream about a railway station, a blue suitcase, a late train, a helpful stranger, and recovering lost belongings."

    def row(index, split, group):
        reference = f"I visited station {index} and recovered my lost blue suitcase before the train arrived."
        words = len(reference.split())
        return {
            "problem_id": f"{split}-{index}",
            "author_group": group,
            "split": split,
            "reference": reference,
            "reference_hash": digest(reference),
            "seeds": brief,
            "target_words": words,
            "prompt": render_prompt(brief, response_prefix="<dream>\n", target_words=words),
        }

    rows = {
        "train": [row(i, "train", f"person-{i % 30}") for i in range(128)],
        "validation": [row(i, "validation", f"reserved-{i % 6}") for i in range(24)],
    }
    pack = {
        "rows": rows,
        "dataset_hash": digest(rows),
        "training_updates": 0,
        "retained_original_reports": 8,
    }
    path = tmp_path / "pack.json"
    write_json(path, pack)
    from think_dream_lab.storage import append_jsonl

    for r in rows["validation"]:
        append_jsonl(tmp_path / "corpus/validation.jsonl", r)
    append_jsonl(tmp_path / "corpus/test.jsonl", row(0, "test", "test-person"))
    config = {
        "training_pack": {"path": str(path), "hash": digest(pack)},
        "generation": {"response_prefix": "<dream>\n"},
        "expanded_experiment": {
            "approved_dataset_hash": pack["dataset_hash"],
            "original_corpus": str(tmp_path / "corpus"),
            "updates": 128,
            "epochs": 1,
            "source_step": 32,
            "target_step": 160,
        },
    }
    return config, pack


def test_approved_pack_has_one_pass_and_no_reserved_people(tmp_path):
    config, pack = fixture_config(tmp_path)
    assert load_expanded_pack(config) == pack
    config["expanded_experiment"]["target_step"] = 161
    with pytest.raises(ValueError, match="una pasada"):
        load_expanded_pack(config)


def test_changed_pack_cannot_be_silently_used(tmp_path):
    config, pack = fixture_config(tmp_path)
    pack["rows"]["train"][0]["seeds"] = "changed"
    write_json(Path(config["training_pack"]["path"]), pack)
    with pytest.raises(ValueError, match="aprobado"):
        load_expanded_pack(config)


def test_reserved_person_is_excluded_even_when_new_hashes_are_supplied(tmp_path):
    config, pack = fixture_config(tmp_path)
    pack["rows"]["train"][0]["author_group"] = "reserved-0"
    pack["dataset_hash"] = digest(pack["rows"])
    write_json(Path(config["training_pack"]["path"]), pack)
    config["training_pack"]["hash"] = digest(pack)
    config["expanded_experiment"]["approved_dataset_hash"] = pack["dataset_hash"]
    with pytest.raises(ValueError, match="reservado"):
        load_expanded_pack(config)


@pytest.mark.metal
def test_bootstrap_changes_identity_without_resetting_optimizer_or_rewriting_source(tmp_path):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    from think_dream_lab import checkpoint
    from think_dream_lab.backend import fingerprint, frozen_fingerprint, seed_all
    from think_dream_lab.expanded_experiment import bootstrap
    from think_dream_lab.holdout_evaluation import sha256
    from think_dream_lab.storage import read_json

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(3, 3, bias=False)
            self.adapter = nn.Linear(3, 3, bias=False)

    class Runner:
        def __init__(self):
            self.config = {"seeds": [11]}

        def prepare_training(self, seed):
            seed_all(seed)
            self.model = Model()
            self.model.freeze()
            self.model.adapter.unfreeze()
            self.optimizer = optim.AdamW(learning_rate=1e-3)
            self.optimizer.init(self.model.trainable_parameters())
            mx.eval(self.model.parameters(), self.optimizer.state)

    original = Runner()
    original.prepare_training(11)
    original.optimizer.update(
        original.model, tree_map(mx.ones_like, original.model.trainable_parameters())
    )
    mx.eval(original.model.parameters(), original.optimizer.state)
    source = checkpoint.save(
        tmp_path / "source",
        original.model,
        original.optimizer,
        step=32,
        run_identity="old",
        extra={},
    )
    hashes = {p.name: sha256(p) for p in source.iterdir()}
    fresh = Runner()
    target = bootstrap(
        fresh, source, "old", "new", tmp_path / "new-run", frozen_fingerprint(original.model)
    )
    assert read_json(source / "state.json")["identity"] == "old"
    assert read_json(target / "state.json")["identity"] == "new"
    assert fingerprint(fresh.model.trainable_parameters()) == fingerprint(
        original.model.trainable_parameters()
    )
    assert fingerprint(fresh.optimizer.state) == fingerprint(original.optimizer.state)
    assert all(sha256(source / name) == value for name, value in hashes.items())
    with pytest.raises(ValueError, match="otra condición"):
        checkpoint.load(target, fresh.model, fresh.optimizer, "old")
