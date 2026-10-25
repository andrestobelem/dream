"""Comparación descriptiva antes/después con denominadores y pares explícitos."""

import statistics

from .paired_experiment import metrics
from .storage import digest, read_json, read_jsonl, write_json


def compare(root):
    folder = root / "dream/paired"
    run = read_json(folder / "run.json")
    if run["status"] != "complete":
        raise ValueError("El ensayo debe terminar antes de emitir una comparación final.")
    pack = read_json(folder / "frozen-training-pack.json")
    if digest(pack["rows"]) != run["dataset_hash"]:
        raise ValueError("Los datos congelados fueron modificados.")
    records = read_jsonl(folder / "responses.jsonl")
    phases = {phase: [r for r in records if r["phase"] == phase] for phase in ("before", "after")}
    count = run["config"]["evaluation"]["samples_per_problem"]
    expected = {(r["problem_id"], i) for r in pack["rows"]["validation"] for i in range(count)}
    keyed = {}
    for phase, values in phases.items():
        keyed[phase] = {(r["problem_id"], r["candidate"]): r for r in values}
        if set(keyed[phase]) != expected or len(values) != len(expected):
            raise ValueError("Faltan evaluaciones o hay candidatos duplicados.")
    pairs = []
    for key in sorted(expected):
        a, b = keyed["before"][key], keyed["after"][key]
        if a["evaluation_seed"] != b["evaluation_seed"] or a["prompt"] != b["prompt"]:
            raise ValueError("La comparación cambió semillas o prompts entre fases.")
        if a["reward"]["format"] and b["reward"]["format"]:
            pairs.append(
                {
                    "problem_id": key[0],
                    "candidate": key[1],
                    "before": a["reward"]["cosine"],
                    "after": b["reward"]["cosine"],
                    "delta": b["reward"]["cosine"] - a["reward"]["cosine"],
                }
            )
    per_report = []
    for row in pack["rows"]["validation"]:
        pid = row["problem_id"]
        item = {"problem_id": pid, "brief": row["seeds"], "target_words": row["target_words"]}
        for phase, values in phases.items():
            item[phase] = metrics([r for r in values if r["problem_id"] == pid])
        item["reward_delta"] = item["after"]["reward_mean_all"] - item["before"]["reward_mean_all"]
        per_report.append(item)
    steps = read_jsonl(folder / "steps.jsonl")
    reload = read_json(folder / "reload-validation.json")
    result = {
        "dataset_hash": run["dataset_hash"],
        "run_identity": run["identity"],
        "before": metrics(phases["before"]),
        "after": metrics(phases["after"]),
        "common_valid_pairs": pairs,
        "n_common_valid_pairs": len(pairs),
        "common_valid_cosine_delta_mean": statistics.mean(r["delta"] for r in pairs)
        if pairs
        else None,
        "per_report": per_report,
        "training": run["training_validation"],
        "reload": reload,
        "steps_recorded": len(steps),
        "steps_with_nonzero_gradient": sum(r["gradient_norm"] > 0 for r in steps),
        "groups_with_variable_reward": sum(r["variable_reward"] for r in steps),
        "limitation": "Cuatro relatos reservados y tres generaciones por caso. "
        "Comparación descriptiva; no demuestra eficacia generalizada.",
    }
    dest = root / "dream/report"
    write_json(dest / "comparison.json", result)
    lines = [
        "# Dream: comparación antes y después",
        "",
        (
            "Ocho relatos de entrenamiento, cuatro reservados y tres generaciones por caso y fase. "
            "Mismo prompt, longitud indicada y semilla de muestreo en cada par. "
            "La referencia humana se mantuvo fuera del prompt."
        ),
        "",
        "## Resumen",
        "",
        "| Medida | Antes | Después |",
        "| --- | ---: | ---: |",
    ]

    def fmt(value):
        return "—" if value is None else str(value) if isinstance(value, int) else f"{value:.4f}"

    names = {
        "n_valid": "Respuestas con formato válido, sobre 12",
        "format_rate": "Proporción de formato válido",
        "source_cosine_valid_mean": "Coseno medio al relato humano, sólo válidas",
        "reward_mean_all": "Recompensa media, incluyendo cero por formato inválido",
        "answer_words_valid_mean": "Palabras del relato, sólo válidas",
        "relative_word_error_valid_mean": "Error relativo de longitud, sólo válidas",
        "truncations": "Truncamientos",
    }
    for key, label in names.items():
        lines.append(f"| {label} | {fmt(result['before'][key])} | {fmt(result['after'][key])} |")
    lines += [
        "",
        (
            "El coseno condicional puede cambiar por la composición de las respuestas válidas. "
            "La recompensa conjunta también depende del formato. No equivalen a una medida validada de oniricidad."
        ),
        "",
        (
            f"Pares válidos en ambas fases: {len(pairs)}. Cambio medio de coseno en esos pares: "
            f"{fmt(result['common_valid_cosine_delta_mean'])}."
        ),
        "",
        "## Por relato reservado",
        "",
        "| Caso | Palabras esperadas | Válidas antes/después | Coseno antes/después | Recompensa antes/después |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in per_report:
        a, b = row["before"], row["after"]
        lines.append(
            f"| {row['problem_id'][:12]} | {row['target_words']} | {a['n_valid']}/3 → {b['n_valid']}/3 | "
            f"{fmt(a['source_cosine_valid_mean'])} → {fmt(b['source_cosine_valid_mean'])} | "
            f"{fmt(a['reward_mean_all'])} → {fmt(b['reward_mean_all'])} |"
        )
    lines += [
        "",
        "## Verificación técnica",
        "",
        (
            f"Pasos registrados: {len(steps)}. Grupos con recompensa "
            f"variable: {result['groups_with_variable_reward']}. Pasos con gradiente no nulo: "
            f"{result['steps_with_nonzero_gradient']}."
        ),
        "",
        (
            f"Adaptadores modificados: "
            f"{result['training']['adapters_changed']}. Base intacta: {result['training']['base_unchanged']}. "
            f"Recarga idéntica: {reload['same_weights']}."
        ),
        "",
        "## Ejemplos emparejados",
        "",
        "Se muestra el candidato 1 de cada relato en ambas fases, elegido por posición, sin seleccionar el mejor score.",
        "",
    ]
    for row in pack["rows"]["validation"]:
        lines += [f"### {row['problem_id'][:12]}", "", row["seeds"], ""]
        for phase in phases:
            rec = keyed[phase][(row["problem_id"], 0)]
            lines += [
                f"#### {phase}",
                "",
                (f"Formato: {rec['reward']['format']}. Coseno: {fmt(rec['reward']['cosine'])}."),
                "",
                rec["reward"]["scored_response"],
                "",
            ]
    lines += [
        "## Límites",
        "",
        result["limitation"],
        "",
        (
            "La longitud es orientativa y no aporta una recompensa adicional. No se penalizan coincidencias textuales. "
            "Los 200 relatos de prueba no participaron en esta evaluación de desarrollo."
        ),
        "",
    ]
    (dest / "comparacion.md").write_text("\n".join(lines), encoding="utf-8")
    print("Comparación:", dest / "comparacion.md")
    return result
