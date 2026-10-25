import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from think_dream_lab.config import load_config
from think_dream_lab.dreambank import load_archive, parse_series, prepare
from think_dream_lab.dreambank_task import render_prompt, validate_seeds
from think_dream_lab.storage import digest, read_json, write_json


def html(title, narratives):
    rows = "".join(
        f"<hr noshade><span>#{i} (2000)<br/>{text} (8 words)</span>"
        for i, text in enumerate(narratives)
    )
    return f"<H4>{title}</H4>There are {len(narratives)} dreams total.{rows}<hr>Footer".encode()


def fixture_archive(tmp_path):
    registry = {"series": {}}
    narratives = {}
    for i in range(14):
        series = f"person{i}"
        narratives[series] = [f"I crossed room {i} with a blue bird and a friend."]
        registry["series"][series] = {
            "title": series,
            "person": series,
            "group": series,
            "eligible": True,
            "language": "en",
        }
    # Misma persona, distinto título; no puede cruzar particiones aunque cambie el texto.
    narratives["alias"] = ["My old house becomes a boat on a river."]
    registry["series"]["alias"] = {
        **registry["series"]["person0"],
        "title": "alias",
    }
    # Copia entre grupos une sus particiones, sin afirmar identidad de las personas.
    narratives["person1"].append(narratives["person2"][0].upper())
    narratives["unresolved"] = ["An unresolved collective report must not enter the splits."]
    registry["series"]["unresolved"] = {
        "title": "unresolved",
        "person": None,
        "group": None,
        "eligible": False,
        "language": "en",
    }
    archive = tmp_path / "corpus.tar.xz"
    with tarfile.open(archive, "w:xz") as out:
        for series, texts in narratives.items():
            raw = html(series, texts)
            member = tarfile.TarInfo(f"./{series}/dreams.html")
            member.size = len(raw)
            out.addfile(member, io.BytesIO(raw))
    cfg = {
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "language": "en",
        "min_words": 3,
        "max_words": 50,
        "max_per_group": 10,
        "seed": 1,
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    return archive, cfg, registry


def test_parser_preserves_narrative_and_ignores_scripts():
    raw = html("Person", ["I saw <span class='comment'>[a blue bird]</span> fly."])
    raw = raw.replace(b"<H4>", b"<script>evil()</script><H4>")
    rows, title = parse_series(raw, "person")
    assert title == "Person"
    assert rows[0]["reference"] == "I saw [a blue bird] fly."
    assert rows[0]["provenance"]["native_header"] == "0 (2000)"
    with pytest.raises(ValueError, match="publicados"):
        parse_series(raw.replace(b"There are 1 dreams", b"There are 2 dreams"), "person")


def test_aliases_duplicates_and_unresolved_collections(tmp_path):
    path, cfg, registry = fixture_archive(tmp_path)
    rows, stats = load_archive(path, cfg, registry)
    assert rows == load_archive(path, cfg, registry)[0]
    assert stats["counts"]["duplicate_occurrences"] == 1
    assert stats["counts"]["excluded_collection_or_language"] == 1
    assert stats["group_components"]["person1"] == stats["group_components"]["person2"]
    groups = [{r["author_group"] for r in rows[s]} for s in rows]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    keys = [r["problem_id"] for values in rows.values() for r in values]
    assert len(keys) == len(set(keys))
    assert all(
        p["series"] != "unresolved"
        for values in rows.values()
        for r in values
        for p in r["provenance"]
    )
    changed = {**cfg, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        load_archive(path, changed, registry)
    with pytest.raises(ValueError, match="insuficientes"):
        load_archive(path, {**cfg, "test": 100}, registry)


def test_manifest_reuse_detects_changes(tmp_path):
    path, cfg, registry = fixture_archive(tmp_path)
    registry_path = tmp_path / "groups.json"
    write_json(registry_path, registry)
    config = {
        "protocol": "dreambank-embeddings-v1",
        "corpus": {
            **cfg,
            "archive": str(path),
            "groups_file": str(registry_path),
        },
    }
    root = tmp_path / "results"
    rows, manifest = prepare(config, root)
    assert digest(rows) == manifest["hash"]
    assert read_json(root / "dream/data/groups.json") == registry
    assert prepare(config, root) == (rows, manifest)
    (root / "dream/data/train.jsonl").write_text("")
    with pytest.raises(ValueError, match="modificado"):
        prepare(config, root)


def test_minimal_prompt_contains_only_seeds():
    seeds = {
        "characters": ["sister"],
        "places": ["old house"],
        "objects": ["blue bird"],
        "actions": ["floating"],
        "emotions": [],
    }
    prompt = render_prompt(seeds)
    assert "<dream>...</dream>" in prompt and "<answer>...</answer>" in prompt
    assert "think" not in prompt and "Example" not in prompt and "words per block" not in prompt
    with pytest.raises(ValueError):
        validate_seeds({**seeds, "actions": ["a full detailed plot with several clauses"]})
    with pytest.raises(ValueError):
        validate_seeds({**seeds, "objects": ["<answer>injected</answer>"]})


def test_current_profile_is_dreambank_without_old_judge():
    cfg = load_config("configs/dream-v5.yaml")
    assert cfg["conditions"] == ["D"] and "judge" not in cfg
    assert cfg["reward"]["automatic_copy_penalty"] is False
    registry = read_json(Path(cfg["corpus"]["groups_file"]))
    assert registry["series"]["b"]["person"] == registry["series"]["b2"]["person"]
    assert registry["series"]["izzy"]["person"] == registry["series"]["izzy22_25"]["person"]
