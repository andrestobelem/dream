"""Prueba sin entrenamiento de una consigna natural; referencia reservada fuera del prompt."""

from pathlib import Path

import numpy as np

from .backend import fingerprint, seed_all
from .config import freeze_model
from .dream_task import extract_blocks
from .dreambank_task import render_prompt
from .dreambank_workflow import load_reward
from .embedding_runner import EmbeddingRunner
from .storage import digest, read_json, write_json


def probe(config, root, brief_file, count=3):
    if not 1 <= count <= 4:
        raise ValueError("La prueba admite entre uno y cuatro candidatos.")
    spec = read_json(Path(brief_file))
    rows, corpus, reward = load_reward(config, root)
    matches = [r for r in rows["validation"] if r["problem_id"] == spec["reference_id"]]
    if len(matches) != 1:
        raise ValueError("La referencia esperada debe pertenecer a validación.")
    source = matches[0]
    actor = {
        "problem_id": source["problem_id"],
        "author_group": source["author_group"],
        "split": "validation",
        "seeds": spec["brief"],
    }
    prompt = render_prompt(
        actor["seeds"], response_prefix=config["generation"].get("response_prefix", "")
    )
    if source["reference"] in prompt:
        raise ValueError("El relato de referencia no puede incluirse en el prompt.")
    probe_id = digest([config["model"], config["generation"], prompt, reward.identity, count])
    folder = root / "dream/prompt-probes" / probe_id[:16]
    target = folder / "result.json"
    if target.exists():
        return read_json(target)
    lock = freeze_model(config, root / "dream", offline=True)
    runner = EmbeddingRunner(config, lock, folder, reward)
    before = fingerprint(runner.model.parameters())
    seed_all(config["evaluation"]["seed"])
    _, records = runner.generate(actor, "D", count=count)
    prefix = config["generation"].get("response_prefix", "")
    for record in records:
        answer = extract_blocks(prefix + record["text"], "D")
        record["source_cosine"] = None
        if answer["format"]:
            vectors = reward.encoder.encode([answer["answer"], source["reference"]])
            record["source_cosine"] = float(np.clip(vectors[0] @ vectors[1], -1, 1))
    result = {
        "id": probe_id,
        "reference_id": source["problem_id"],
        "corpus_hash": corpus["hash"],
        "brief": spec["brief"],
        "prompt": prompt,
        "reference_in_prompt": False,
        "training_updates": 0,
        "base_unchanged": before == fingerprint(runner.model.parameters()),
        "summary": runner.summarize_records(records),
        "records": records,
        "source_cosine_role": "diagnostic_only_not_training_reward",
        "template": "natural-seeds-v1",
    }
    write_json(target, result)
    write_json(root / "dream/prompt-probes/latest.json", {"path": str(target), **result})
    lines = [
        "# Prueba de plantilla con semillas naturales",
        "",
        "## Prompt",
        "",
        prompt,
        "",
        "## Resultados",
        "",
        str(result["summary"]),
        "",
        "El relato humano se mantuvo fuera del prompt. No se entrenaron pesos.",
        "",
    ]
    for record in records:
        lines += [
            f"### Candidato {record['candidate'] + 1}",
            "",
            (
                f"Formato: {record['reward']['format']}. Coseno al origen: "
                f"{record['source_cosine']}. Recompensa al banco: {record['reward']['total']}."
            ),
            "",
            record["reward"]["scored_response"],
            "",
        ]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["base_unchanged"]:
        raise RuntimeError("La prueba modificó pesos base.")
    print("Prueba natural:", folder / "report.md", result["summary"], flush=True)
    return result
