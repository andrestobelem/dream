"""Carga y generación del nuevo modelo base, sin juez ni actualización de pesos."""

from .backend import Runner, fingerprint, seed_all
from .config import freeze_model, implementation_manifest
from .dream_task import extract_blocks, render_dream
from .storage import append_jsonl, write_json


class BaseProbe(Runner):
    def render_prompt(self, row, condition):
        return render_dream(
            row["brief"],
            condition,
            format_example=self.config["generation"].get("format_example", False),
        )

    def stop_sequence(self):
        return "</answer>"

    def completion_reward(self, text, row, condition):
        blocks = extract_blocks(text, condition)
        return {
            "format": blocks["format"],
            "answer_words": len(blocks["answer"].split()),
            "not_scored": True,
            "total": None,
        }


def check_base(config, root):
    if config["model"]["id"] != "HuggingFaceTB/SmolLM2-1.7B":
        raise ValueError("La comprobación nueva requiere SmolLM2 base.")
    implementation = implementation_manifest()
    folder = root / "dream" / "diagnostics" / "base-check" / implementation["hash"][:12]
    if (folder / "base-validation.json").exists():
        raise ValueError("Esta comprobación ya tiene resultados; se conservan sin sobrescribir.")
    lock = freeze_model(config, root / "dream")
    runner = BaseProbe(config, lock, folder)
    before = fingerprint(runner.model.parameters())
    row = {
        "problem_id": "base-format-probe",
        "family": "dream_schema_probe",
        "brief": "A school hallway, my brother, doors disappearing in front of us, and fear.",
    }
    formats = {}
    for condition in [*config["conditions"], "neutral"]:
        seed_all(config["evaluation"]["seed"])
        _, records = runner.generate(row, condition, count=1)
        for record in records:
            append_jsonl(folder / "responses.jsonl", {**record, "phase": "base_probe"})
        formats[condition] = records[0]["reward"]["format"]
        print("Generación de la base:", condition, "formato:", formats[condition], flush=True)
    result = {
        "model": lock,
        "config": config,
        "implementation": implementation,
        "base_unchanged": before == fingerprint(runner.model.parameters()),
        "format_by_condition": formats,
        "n_per_condition": 1,
        "training_updates": 0,
        "judge_calls": 0,
        "chat_template_used": False,
        "limitation": "Comprobación de carga y generación; no demuestra calidad ni estabilidad de formato.",
    }
    write_json(folder / "base-validation.json", result)
    write_json(root / "dream" / "base-validation.json", {**result, "artifact_folder": str(folder)})
    if not result["base_unchanged"]:
        raise RuntimeError("Los pesos cambiaron durante la comprobación de la base.")
