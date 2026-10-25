from copy import deepcopy

import pytest

from think_dream_lab.dreambank_task import render_prompt
from think_dream_lab.holdout_evaluation import SEED_JOIN_VERSION, SEED_RULE, validate_pack
from think_dream_lab.holdout_report import cluster_interval
from think_dream_lab.storage import digest


def test_uncertainty_resamples_people_not_generations():
    values = [("a", 0.1), ("b", -0.2), ("c", 0.3)]
    first = cluster_interval(values, samples=1000)
    repeated = cluster_interval(values * 100, samples=1000)
    assert first == repeated
    assert first["groups"] == 3 and first["unit"] == "author_group"
    assert cluster_interval([("a", 0.2)])["low"] is None


def test_frozen_pack_checks_actual_prompt_even_if_row_hash_is_recomputed():
    config = {
        "heldout_evaluation": {"reports": 1},
        "seed_extractor": {"id": "teacher"},
        "generation": {"response_prefix": "<dream>\n"},
    }
    original = {
        "problem_id": "reserved",
        "reference": "A person walks through a cold house and sees a bathtub.",
    }
    selection = {"rows": [original], "config": config["heldout_evaluation"]}
    brief = "Dream about a cold house, empty bedrooms, a mattress in a bathtub, family members and finding privacy."
    words = len(original["reference"].split())
    row = {
        **original,
        "reference_hash": digest(original["reference"]),
        "seeds": brief,
        "target_words": words,
        "prompt": render_prompt(brief, response_prefix="<dream>\n", target_words=words),
    }
    pack = {
        "rows": [row],
        "rows_hash": digest([row]),
        "selection_hash": digest(selection),
        "extractor_identity": digest([config["seed_extractor"], SEED_RULE, SEED_JOIN_VERSION]),
    }
    validate_pack(pack, selection, config)
    changed = deepcopy(pack)
    changed["rows"][0]["prompt"] += original["reference"]
    changed["rows_hash"] = digest(changed["rows"])
    with pytest.raises(ValueError, match="prompt reservado"):
        validate_pack(changed, selection, config)
