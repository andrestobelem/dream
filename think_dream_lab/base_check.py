"""Carga y generación del nuevo modelo base, sin juez ni actualización de pesos."""

from .backend import Runner, fingerprint, seed_all
from .config import freeze_model, implementation_manifest
from .dream_task import extract_blocks, render_dream
from .storage import append_jsonl, write_json

FORMAT_BRIEFS = [
    "A school hallway, my brother, doors disappearing in front of us, and fear.",
    "A kitchen, my mother making tea, a familiar conversation, and comfort.",
    "A train station, a stranger returning my bag, missing the train, and frustration.",
    "A garden at night, a dog following me, flowers opening as we pass, and wonder.",
    "An empty office, a ringing telephone, being unable to answer it, and anxiety.",
    "A beach, an old friend, looking for a lost shoe in the sand, and amusement.",
    "A library, a librarian helping me, a staircase changing direction, and confusion.",
    "My childhood bedroom, rain against the window, waiting for morning, and calm.",
    "A city street, realizing I am dreaming, floating above the traffic, and excitement.",
    "A supermarket, my sister, a shopping cart that keeps shrinking, and surprise.",
]


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
    formats = {}
    conditions = ["D"] if config["conditions"] == ["D"] else [*config["conditions"], "neutral"]
    for condition in conditions:
        formats[condition] = []
        for index, brief in enumerate(FORMAT_BRIEFS):
            row = {
                "problem_id": f"base-format-probe-{index + 1}",
                "family": "dream_schema_probe",
                "brief": brief,
            }
            seed_all(config["evaluation"]["seed"] + index)
            _, records = runner.generate(row, condition, count=1)
            for record in records:
                append_jsonl(folder / "responses.jsonl", {**record, "phase": "base_probe"})
            formats[condition].append(records[0]["reward"]["format"])
        print("Generación de la base:", condition, "formato:", formats[condition], flush=True)
    result = {
        "model": lock,
        "config": config,
        "implementation": implementation,
        "base_unchanged": before == fingerprint(runner.model.parameters()),
        "format_by_condition": formats,
        "n_per_condition": len(FORMAT_BRIEFS),
        "briefs_origin": "synthetic_format_controls_not_corpus_quality_evaluation",
        "format_preflight_passed": all(all(values) for values in formats.values()),
        "format_preflight_criterion": "10/10 bloques completos por condición activa; diagnóstico de formato, sin validar calidad.",
        "training_updates": 0,
        "judge_calls": 0,
        "chat_template_used": False,
        "limitation": "Comprobación de carga y generación; no demuestra calidad ni estabilidad de formato.",
    }
    write_json(folder / "base-validation.json", result)
    write_json(root / "dream" / "base-validation.json", {**result, "artifact_folder": str(folder)})
    if not result["base_unchanged"]:
        raise RuntimeError("Los pesos cambiaron durante la comprobación de la base.")
