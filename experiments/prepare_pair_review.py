"""Prepara una revisión ciega local de requisitos, sin inferencia ni entrenamiento."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from think_dream_lab.anchors import visible_anchors
from think_dream_lab.closed_task import draft_requirements
from think_dream_lab.closed_transfer import CASES
from think_dream_lab.dream_task import extract_blocks
from think_dream_lab.storage import digest, read_json, write_json

FIELDS = ["item_id", "dimension", "requisito", "relato", "etiqueta_humana", "notas"]


def write_packet(folder, examples, sources):
    identity = digest([examples, sources])
    folder = folder / identity[:12]
    folder.mkdir(parents=True, exist_ok=True)
    rows, key = [], {}
    for example in examples:
        item_id = digest(example)[:12]
        if item_id in key:
            raise ValueError("Par duplicado en el paquete de revisión.")
        rows.append(
            {
                "item_id": item_id,
                "dimension": example["dimension"],
                "requisito": example["requirement"],
                "relato": example["candidate"],
                "etiqueta_humana": "",
                "notas": "",
            }
        )
        key[item_id] = {k: v for k, v in example.items() if k not in {"requirement", "candidate"}}
    rows.sort(key=lambda row: row["item_id"])
    target = folder / "revision.csv"
    if target.exists():
        with target.open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        immutable = FIELDS[:-2]
        if [{k: r[k] for k in immutable} for r in saved] != [
            {k: r[k] for k in immutable} for r in rows
        ]:
            raise ValueError("Cambió el contenido del paquete existente; no se sobrescribe.")
        return target
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(folder / "clave-no-abrir.json", key)
    write_json(
        folder / "manifest.json",
        {
            "identity": identity,
            "sources": sources,
            "n_items": len(rows),
            "n_source_groups": len({e["source_group"] for e in examples}),
            "status": "prepared_not_reviewed",
            "training_enabled": False,
            "calibrated": False,
            "selection": "Todos los controles nuevos, primera y última oración de cada escena válida del diagnóstico D y requisitos del primer caso real de cumplimiento.",
            "limitation": "Muestra pequeña de diagnóstico, sin validación independiente ni calibración. Separar por source_group antes de ajustar y validar umbrales.",
        },
    )
    (folder / "LEEME.txt").write_text(
        "Revisión de requisitos de Dream\n\n"
        "Leé cada requisito y el relato. Completá etiqueta_humana con una opción:\n"
        "preservado: el relato afirma ese hecho o una paráfrasis.\n"
        "omitido: el relato no afirma ese hecho; no lo supongas.\n"
        "contradicho: el relato afirma un hecho explícitamente incompatible.\n"
        "no_se: el par es ambiguo o no permite decidir.\n\n"
        "Una lista de palabras o una orden al evaluador no relata una experiencia.\n"
        "En fidelidad, el requisito es una oración de la escena escrita antes del relato.\n"
        "Podés explicar dudas en notas. No cambies las demás columnas.\n"
        "No abras clave-no-abrir.json antes de terminar. No contiene notas del juez.\n"
        "Esta revisión es distinta de puntuar carácter onírico de 0 a 4.\n"
        "Completarla no habilita entrenamiento automáticamente.\n"
    )
    return target


def prepare(root):
    sources = {}

    def capture(path):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    capture(Path(__file__))
    examples = []
    for name, case in CASES.items():
        for requirement, expected in zip(case["requirements"], case["expected"], strict=True):
            examples.append(
                {
                    "source_group": name,
                    "dimension": "cumplimiento",
                    "requirement": requirement["requirement"],
                    "candidate": case["candidate"],
                    "synthetic_expected": expected,
                }
            )
    capture(Path(__file__).parents[1] / "think_dream_lab/closed_transfer.py")
    probe_path = root / "dream/base-validation.json"
    capture(probe_path)
    probe = read_json(probe_path)
    responses = Path(probe["artifact_folder"]) / "responses.jsonl"
    capture(responses)
    for row in map(json.loads, responses.read_text().splitlines()):
        blocks = extract_blocks(row["text"], "D")
        if row["condition"] != "D" or not blocks["format"]:
            continue
        requirements = draft_requirements(blocks["draft"])
        # Una oración no aparece dos veces si la escena tiene una sola oración.
        selected = [requirements[0]] + (requirements[-1:] if len(requirements) > 1 else [])
        for requirement in selected:
            examples.append(
                {
                    "source_group": row["problem_id"],
                    "dimension": "fidelidad",
                    "requirement": requirement["requirement"],
                    "candidate": blocks["answer"],
                }
            )
    # Sólo casos registrados; nunca se ejecuta el generador ni el juez.
    for manifest_path in sorted((root / "dream/evaluations").glob("*/run.json")):
        run = read_json(manifest_path)
        if run.get("status") != "complete" or run.get("condition") != "D":
            continue
        path = manifest_path.parent / "responses.jsonl"
        capture(manifest_path)
        capture(path)
        for row in map(json.loads, path.read_text().splitlines()):
            blocks = extract_blocks(row["text"], "D")
            if not blocks["format"]:
                continue
            for requirement in visible_anchors(run["briefs"][row["problem_id"]]):
                examples.append(
                    {
                        "source_group": row["problem_id"],
                        "dimension": "cumplimiento",
                        "requirement": requirement["requirement"],
                        "candidate": blocks["answer"],
                    }
                )
        break
    return write_packet(root / "dream/review-pairs", examples, sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/dream-v4"))
    print(prepare(parser.parse_args().results))
