"""Informe local: descriptivo para smoke y comparaciones por semilla para el piloto."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from .backend import summarize
from .storage import read_json, read_jsonl, write_json


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect(folder):
    records = read_jsonl(folder / "responses.jsonl")
    # Una interrupción puede dejar logs incompletos; checkpoints completos son la fuente de verdad.
    steps = {r["step"]: r for r in read_jsonl(folder / "steps.jsonl")}
    rollouts = {(r["step"], r["candidate"]): r for r in records if r["phase"] == "rollout"}
    for path in sorted((folder / "checkpoints").glob("step-*/state.json")):
        state = read_json(path)
        steps[state["step"]] = state["extra"]["metric"]
        for record in state["extra"]["responses"]:
            rollouts[(record["step"], record["candidate"])] = record
    evaluations = {
        (r["phase"], r["condition"], r["step"], r["problem_id"]): r
        for r in records
        if r["phase"] != "rollout"
    }
    return list(evaluations.values()) + list(rollouts.values()), sorted(
        steps.values(), key=lambda r: r["step"]
    )


def report(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    output = root / "report"
    output.mkdir(exist_ok=True)
    folders = [p.parent for p in root.glob("**/run.json") if "report" not in p.parts]
    table, deltas, all_steps, families = [], [], [], []
    full_runs = {}
    text = [
        "# Informe de Dream",
        "",
        (
            "Comparación local de instrucciones y delimitadores. El smoke verifica la implementación; "
            "su muestra pequeña no permite afirmar significancia ni equivalencia."
        ),
        "",
    ]
    budget = read_json(root / "budget.json") if (root / "budget.json").exists() else None
    if budget:
        text += [
            f"Tiempo acumulado supervisado: **{budget['used_seconds'] / 60:.2f} minutos**.",
            "",
        ]
        for entry in budget["runs"]:
            peak = entry.get("peak_rss_bytes", 0) / 1024**3
            swap = (entry.get("peak_swap_bytes", 0) - entry.get("initial_swap_bytes", 0)) / 1024**3
            text.append(
                f"- {entry['args'][0]}: {entry.get('status')}; "
                f"{entry.get('elapsed_seconds', 0):.1f} s; pico RSS observado {peak:.2f} GiB; "
                f"crecimiento de swap {swap:.2f} GiB; motivo {entry.get('reason') or 'sin interrupción'}."
            )
        text.append("")
    doctor = read_json(root / "doctor.json") if (root / "doctor.json").exists() else None
    if doctor:
        text += [
            (
                f"Equipo: {doctor['chip']}, {doctor['memory_bytes'] / 1024**3:.0f} GiB; "
                f"macOS {doctor['macos']}; Python {doctor['python']}."
            ),
            "",
            "Dependencias instaladas: "
            + ", ".join(f"{k}={v}" for k, v in doctor["packages"].items()),
            "",
        ]
    for folder in folders:
        manifest = read_json(folder / "run.json")
        records, steps = collect(folder)
        name = str(folder.relative_to(root))
        text += [
            f"## {name}",
            "",
            (
                f"Estado declarado: **{manifest['status']}**. "
                f"Checkpoints completos: **{len(steps)}**."
            ),
            "",
        ]
        grouped = defaultdict(list)
        for record in records:
            if record["phase"] != "rollout":
                grouped[(record["phase"], record["condition"])].append(record)
        text += [
            "| Fase | Prompt | n | Acierto | Formato | Tokens | Truncamientos |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        summaries = {}
        for (phase, condition), values in sorted(grouped.items()):
            summary = summarize(values)
            summaries[(phase, condition)] = summary
            table.append(
                {
                    "run": name,
                    "seed": manifest["seed"],
                    "phase": phase,
                    "condition": condition,
                    **summary,
                }
            )
            text.append(
                f"| {phase} | {condition} | {summary['n']} | {summary['accuracy']:.1%} | "
                f"{summary['format']:.1%} | {summary['tokens']} | {summary['truncations']} |"
            )
            subsets = defaultdict(list)
            for value in values:
                subsets[(value["family"], value.get("generalization", False))].append(value)
            for (family, generalization), subset in subsets.items():
                families.append(
                    {
                        "run": name,
                        "phase": phase,
                        "condition": condition,
                        "family": family,
                        "generalization": generalization,
                        **summarize(subset),
                    }
                )
        for phase, condition in summaries:
            if phase == "after" and ("before", condition) in summaries:
                before_rows = {r["problem_id"]: r for r in grouped[("before", condition)]}
                after_rows = {r["problem_id"]: r for r in grouped[("after", condition)]}
                paired = before_rows.keys() & after_rows.keys()
                if not paired:
                    continue
                deltas.append(
                    {
                        "run": name,
                        "seed": manifest["seed"],
                        "condition": condition,
                        "paired_problems": len(paired),
                        "accuracy_delta": statistics.mean(
                            after_rows[p]["reward"]["accuracy"]
                            - before_rows[p]["reward"]["accuracy"]
                            for p in paired
                        ),
                        "format_delta": statistics.mean(
                            after_rows[p]["reward"]["format"] - before_rows[p]["reward"]["format"]
                            for p in paired
                        ),
                    }
                )
        text.append("")
        validation_path = folder / "training-validation.json"
        if validation_path.exists():
            validation = read_json(validation_path)
            text += [
                (
                    f"Adaptadores modificados: **{validation['adapters_changed']}**. "
                    f"Pesos base sin cambios: **{validation['base_unchanged']}**."
                ),
                "",
            ]
            if not validation["adapters_changed"]:
                text += [
                    (
                        "No se comprobó cambio de adaptadores: revisar variación de recompensa y gradientes. "
                        "No se presenta esta corrida como validación de aprendizaje."
                    ),
                    "",
                ]
        proof_path = folder / "continuity-validation.json"
        if proof_path.exists():
            proof = read_json(proof_path)
            text += [
                (
                    f"Continuidad real desde el paso {proof['from_step']} al {proof['to_step']}: "
                    f"**{proof['status']}**. Adaptadores idénticos: {proof['same_adapters']}; "
                    f"optimizador idéntico: {proof['same_optimizer']}; "
                    f"PRNG MLX idéntico: {proof['same_mlx_random']}; "
                    f"logits posteriores finitos: {proof['post_reload_logits_finite']}."
                ),
                "",
            ]
        if steps:
            variable = sum(s["variable_reward"] for s in steps) / len(steps)
            text += [
                (
                    f"Grupos con variación de recompensa: **{variable:.1%}**. "
                    f"Pico MLX: **{max(s['mlx_peak_bytes'] for s in steps) / 1024**3:.2f} GiB**."
                ),
                "",
            ]
        for step in steps:
            all_steps.append(
                {
                    "run": name,
                    "kind": manifest["kind"],
                    **{k: v for k, v in step.items() if not isinstance(v, (dict, list))},
                }
            )
        if manifest["kind"] == "experiment" and manifest["status"] == "complete":
            full_runs[(manifest["seed"], manifest["condition"])] = records
    effects = paired_effects(full_runs)
    write_csv(output / "evaluations.csv", table)
    write_csv(output / "families.csv", families)
    write_csv(output / "training.csv", all_steps)
    write_csv(output / "deltas.csv", deltas)
    write_csv(output / "paired-effects.csv", effects)
    if effects:
        text += [
            "## Efectos factoriales por semilla",
            "",
            (
                "Cada fila compara cambios antes/después sobre los mismos problemas. "
                "La réplica de entrenamiento es la semilla; las respuestas no se cuentan como entrenamientos independientes."
            ),
            "",
        ]
        aggregates = defaultdict(list)
        for row in effects:
            aggregates[(row["prompt_type"], row["effect"])].append(row["value"])
        for (prompt, effect), values in aggregates.items():
            std = statistics.stdev(values) if len(values) > 1 else None
            text.append(
                f"- {prompt}/{effect}: media {statistics.mean(values):+.4f}, "
                f"desvío entre semillas {std if std is not None else 'no estimable'}, n={len(values)}."
            )
        text.append("")
    if all_steps:
        plot(output, all_steps)
        primary_steps = [s for s in all_steps if s["kind"] != "diagnostic"]
        sec_per_step = statistics.mean(s["seconds"] for s in primary_steps or all_steps)
        eval_rows = [r for row in table if row["n"] for r in [row["seconds"] / row["n"]]]
        sec_per_eval = statistics.mean(eval_rows) if eval_rows else 0
        count = read_json(root / "data" / "manifest.json")["counts"]["test"]
        estimate = {
            "seconds_per_update": sec_per_step,
            "seconds_per_evaluation_problem": sec_per_eval,
            "twelve_runs_training_seconds_per_requested_step": 12 * sec_per_step,
            "twelve_runs_evaluation_seconds": 12 * 4 * count * sec_per_eval,
            "limitations": "Extrapolación: no incluye carga, hashing ni checkpoints; longitudes pueden cambiar.",
        }
        write_json(output / "estimate.json", estimate)
        text += [
            "## Estimación para las doce corridas",
            "",
            (
                f"Entrenamiento: aproximadamente **{12 * sec_per_step / 60:.2f} minutos × pasos por corrida**. "
                f"Evaluaciones completas antes/después, con prompt propio y neutral: "
                f"aproximadamente **{estimate['twelve_runs_evaluation_seconds'] / 3600:.2f} horas** adicionales."
            ),
            "",
            (
                "Es una extrapolación de una muestra pequeña; no incluye todas las cargas y puede variar "
                "si cambian las longitudes. No se iniciaron las doce corridas automáticamente."
            ),
            "",
        ]
    errors = read_jsonl(root / "errors.jsonl")
    if errors:
        text += ["## Incidentes registrados", ""]
        text.extend(f"- {e['action']}: `{e['error']}`" for e in errors)
        text.append("")
    text += [
        "## Límites de interpretación",
        "",
        (
            "Se mide acierto verificable, cumplimiento de formato y costo. El estilo del texto no identifica "
            "el mecanismo interno. Un resultado parecido no demuestra equivalencia; un resultado mejor de dream "
            "no demuestra un sueño interno. Las conclusiones se restringen a este modelo, LoRA, datos y presupuesto."
        ),
        "",
        (
            "Recuperación: se guardan adaptadores, optimizador, PRNG y cursor en cada actualización. "
            "La prueba numérica usa un modelo pequeño. Cuando se ejecuta la verificación real adicional, "
            "su resultado se muestra arriba; no implica identidad bit a bit entre otras versiones o equipos."
        ),
        "",
    ]
    (output / "informe.md").write_text("\n".join(text))
    print(f"Informe: {output / 'informe.md'}", flush=True)


def paired_effects(runs):
    output = []
    for seed in sorted({key[0] for key in runs}):
        if not all((seed, c) in runs for c in "ABCD"):
            continue
        for prompt_type in ["own", "neutral"]:
            deltas = {}
            for c in "ABCD":
                selected = "neutral" if prompt_type == "neutral" else c
                records = runs[(seed, c)]
                before = {
                    r["problem_id"]: r["reward"]["accuracy"]
                    for r in records
                    if r["phase"] == "before" and r["condition"] == selected
                }
                after = {
                    r["problem_id"]: r["reward"]["accuracy"]
                    for r in records
                    if r["phase"] == "after" and r["condition"] == selected
                }
                deltas[c] = {pid: after[pid] - before[pid] for pid in before.keys() & after.keys()}
            common = set.intersection(*(set(v) for v in deltas.values()))
            if not common:
                continue
            effects = defaultdict(list)
            for pid in sorted(common):
                a, b, c, d = [deltas[key][pid] for key in "ABCD"]
                effects["instruction"].append(((c - a) + (d - b)) / 2)
                effects["tag"].append(((b - a) + (d - c)) / 2)
                effects["interaction"].append(d - c - b + a)
            for effect, values in effects.items():
                output.append(
                    {
                        "seed": seed,
                        "prompt_type": prompt_type,
                        "effect": effect,
                        "paired_problems": len(common),
                        "value": statistics.mean(values),
                    }
                )
    return output


def plot(output, steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = defaultdict(list)
    for row in steps:
        runs[row["run"]].append(row)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    figure.suptitle("Recompensas de los rollouts de entrenamiento · no son métricas de test")
    for name, rows in runs.items():
        for column, xlabel in [(0, "step"), (1, "cumulative_tokens")]:
            axes[0, column].plot(
                [r[xlabel] for r in rows], [r["accuracy"] for r in rows], label=name
            )
            axes[1, column].plot([r[xlabel] for r in rows], [r["format"] for r in rows], label=name)
    for row in range(2):
        for col in range(2):
            axes[row, col].set(
                xlabel="Paso" if col == 0 else "Tokens consumidos",
                ylabel="Acierto" if row == 0 else "Formato",
                ylim=(-0.05, 1.05),
            )
            axes[row, col].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.savefig(output / "curvas.png", dpi=160)
    plt.close(figure)
