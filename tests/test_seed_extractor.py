import json

import pytest

from think_dream_lab.config import load_config
from think_dream_lab.dreambank_task import select_seeds
from think_dream_lab.seed_extractor import SeedExtractor
from think_dream_lab.storage import read_json, write_json

SEEDS = {
    "characters": ["sister"],
    "places": ["river"],
    "objects": ["red boat"],
    "actions": [],
    "emotions": [],
}


def test_selection_shortens_oversupply_without_inventing_or_rewriting_terms():
    supplied = {
        "characters": ["sister", "mom", "ghost", "Casper"],
        "places": ["bedroom", "our room", "light"],
        "objects": ["door", "ghost", "light"],
        "actions": ["appear", "opened", "told it to go", "saw another ghost"],
    }
    selected = select_seeds(supplied)
    terms = [t for values in selected.values() for t in values]
    assert len(terms) == len(set(terms)) == 8
    assert selected["emotions"] == []
    assert all(t in supplied[k] for k, values in selected.items() for t in values)
    assert selected == select_seeds(supplied)


def test_extractor_caches_seeds_and_never_sends_reference_to_actor(tmp_path):
    calls = []

    def complete(reference):
        calls.append(reference)
        return json.dumps(SEEDS)

    config = load_config("configs/dream-v5.yaml")
    extractor = SeedExtractor(config, tmp_path, complete=complete)
    source = {
        "problem_id": "id",
        "reference": "A reserved human report with a red boat.",
        "author_group": "person",
        "split": "validation",
        "other_metadata": "hidden",
    }
    actor = extractor.extract(source)
    assert actor == extractor.extract(source)
    assert calls == [source["reference"]]
    assert actor == {
        "problem_id": "id",
        "author_group": "person",
        "split": "validation",
        "seeds": SEEDS,
    }
    cached = next((tmp_path / "dream/seeds/cache").glob("*.json"))
    record = read_json(cached)
    record["seeds"]["objects"] = ["a different thing"]
    write_json(cached, record)
    with pytest.raises(ValueError, match="modificada"):
        extractor.extract(source)


def test_invalid_extraction_is_audited_without_fabricated_fallback(tmp_path):
    extractor = SeedExtractor(
        load_config("configs/dream-v5.yaml"),
        tmp_path,
        complete=lambda _: '{"plot":"a whole story"}',
    )
    row = {"problem_id": "id", "reference": "A dream.", "split": "train", "author_group": "p"}
    with pytest.raises(ValueError, match="no se inventa"):
        extractor.extract(row)
    assert not list((tmp_path / "dream/seeds/cache").glob("*.json"))
    errors = (tmp_path / "dream/seeds/errors.jsonl").read_text()
    assert "a whole story" in errors


def test_extractor_identity_is_frozen_and_uses_another_model(tmp_path):
    cfg = load_config("configs/dream-v5.yaml")
    SeedExtractor(cfg, tmp_path, complete=lambda _: "")
    cfg["seed_extractor"]["revision"] = "changed"
    with pytest.raises(ValueError, match="otro extractor"):
        SeedExtractor(cfg, tmp_path, complete=lambda _: "")
    cfg["seed_extractor"]["id"] = cfg["model"]["id"]
    with pytest.raises(ValueError, match="otro modelo"):
        SeedExtractor(cfg, tmp_path, complete=lambda _: "")
