"""Informe del corpus, el juez y los relatos; no contiene métricas de aritmética."""

import csv
import statistics
from collections import defaultdict

from .dream_task import extract_blocks
from .dream_workflow import summarize
from .report import collect, write_csv
from .storage import digest, read_json, read_jsonl


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}"


def paired_statistics(before, after, common):
    result = {}
    for key in [
        "dream_likeness",
        "reference_alignment",
        "content_compliance",
        "retelling_fidelity",
        "quality",
        "format",
        "total",
    ]:
        differences = [
            after[p]["reward"][key] - before[p]["reward"][key]
            for p in common
            if after[p]["reward"].get(key) is not None and before[p]["reward"].get(key) is not None
        ]
        result[f"n_pairs_{key}"] = len(differences)
        result[f"delta_{key}"] = statistics.mean(differences) if differences else None
    return result


def report(root):
    dream = root / "dream"
    output = dream / "report"
    output.mkdir(parents=True, exist_ok=True)
    text = [
        "# Dream: relatos de sueños",
        "",
        (
            "El modelo produce una escena dentro de `<dream>` y cuenta ese sueño dentro de `<answer>`. "
            "Se reportan carácter onírico, cumplimiento y fidelidad por separado, según la rúbrica "
            "guardada en cada corrida. Las medias excluyen dimensiones no puntuadas; su cobertura "
            "se conserva en el CSV. Una abstención no cuenta como cero."
        ),
        "",
    ]
    budget_path = root / "budget.json"
    if budget_path.exists():
        budget = read_json(budget_path)
        peak = max((r.get("peak_rss_bytes", 0) for r in budget["runs"]), default=0)
        text += [
            (
                f"Tiempo supervisado acumulado (incluye diagnósticos anteriores): "
                f"**{budget['used_seconds'] / 60:.2f} min**. Pico RSS observado: **{peak / 1024**3:.2f} GiB**."
            ),
            "",
        ]
    data_path = dream / "data" / "manifest.json"
    if data_path.exists():
        m = read_json(data_path)
        text += [
            (
                f"Corpus: [DreamViews]({m['source']}); licencia **{m['license']}**. "
                f"Relatos elegibles: **{m['stats']['eligible_reports']}**, autores elegibles: "
                f"**{m['stats']['eligible_authors']}**."
            ),
            "",
            f"Selección congelada: {m['counts']}. Autores distintos por split: {m['authors']}.",
            "",
            "Etiquetas originales más frecuentes: "
            + ", ".join(
                f"{k} ({v})"
                for k, v in sorted(m["stats"]["native_categories"].items(), key=lambda x: -x[1])[
                    :10
                ]
            ),
            "",
            (
                "Los indicadores por palabras son etiquetas débiles; no se presentan como anotaciones humanas H/VdC. "
                "No se usaron textos surrealistas."
            ),
            "",
        ]
    calibration = dream / "judge-calibration.json"
    if calibration.exists():
        m = read_json(calibration)
        text += [
            (
                f"Juez: controles básicos **{'aprobados' if m['passed'] else 'fallidos'}**; "
                f"mediana humana {m['positive_median']}/4, controles negativos {m['negative_median']}/4."
            ),
            "",
            "Esto no sustituye una validación humana ciega ni prueba que el juez distinga sueños reales de ficción.",
            "",
        ]
    reference_check = dream / "reference-calibration.json"
    if reference_check.exists():
        passed = read_json(reference_check)["passed"]
        text += [
            f"Controles de referencia idéntica, relato ajeno y escena cambiada: **{'aprobados' if passed else 'fallidos'}**.",
            "",
        ]
    tables, steps_table, examples, paired = [], [], [], []
    protocols = set()
    closed_check = dream / "closed-check.json"
    if closed_check.exists():
        c = read_json(closed_check)
        text += [
            (
                f"Juez cerrado: **{c['expected_cases_passed']}/{c['n_cases']}** situaciones coincidentes. "
                "La repetibilidad técnica no demuestra corrección semántica. Entrenamiento no habilitado por este diagnóstico."
            ),
            "",
        ]
    anchor_check = dream / "anchor-calibration.json"
    if anchor_check.exists():
        controls = read_json(anchor_check)
        text += [
            "Controles de desarrollo por requisitos visibles: "
            + ("**aprobados**" if controls["passed"] else "**fallidos; no iniciar RL**")
            + ". Pendiente validación humana independiente.",
            "",
        ]
    for path in sorted(dream.glob("**/run.json")):
        if "diagnostics" in path.relative_to(dream).parts:
            continue
        manifest = read_json(path)
        protocols.add(manifest.get("config", {}).get("judge", {}).get("rubric_version"))
        folder = path.parent
        records, steps = collect(folder)
        label = str(folder.relative_to(dream))
        text += [
            f"## {label}",
            "",
            f"Estado: **{manifest['status']}**. Pasos guardados: **{len(steps)}**.",
            "",
            "| Fase | Condición | Relatos / n | Reward / n | Onírico /100 | Sólo relatos /100 | Referencia /100 | Cumplimiento válido /100 | Fidelidad /100 | Formato |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        groups = defaultdict(list)
        for r in records:
            if r["phase"] != "rollout":
                groups[(r["phase"], r["condition"])].append(r)
                blocks = extract_blocks(r["text"], r["condition"])
                examples.append(
                    {
                        "run": label,
                        "phase": r["phase"],
                        "condition": r["condition"],
                        "problem_id": r["problem_id"],
                        "draft": blocks["draft"],
                        "answer": blocks["answer"],
                        **r["reward"],
                    }
                )
        for (phase, condition), values in sorted(groups.items()):
            s = summarize(values)
            tables.append({"run": label, "phase": phase, "condition": condition, **s})
            fidelity_display = "—" if condition == "neutral" else percent(s["retelling_fidelity"])
            valid_display = (
                "—"
                if s["valid_dream_likeness"] is None
                else f"{100 * s['valid_dream_likeness']:.1f}"
            )
            text.append(
                f"| {phase} | {condition} | {s['n_valid_answers']}/{s['n']} | {s['n_reward_scored']}/{s['n']} | {percent(s['dream_likeness'])} | {valid_display} | "
                f"{percent(s['reference_alignment'])} | "
                + (
                    f"{100 * s['valid_content_compliance']:.1f}"
                    if s.get("valid_content_compliance") is not None
                    else "—"
                )
                + f" | {fidelity_display} | {s['format']:.0%} |"
            )
        for condition in {key[1] for key in groups}:
            before = {r["problem_id"]: r for r in groups.get(("before", condition), [])}
            after = {r["problem_id"]: r for r in groups.get(("after", condition), [])}
            common = before.keys() & after.keys()
            if common:
                paired.append(
                    {
                        "run": label,
                        "seed": manifest["seed"],
                        "condition": condition,
                        "n_pairs": len(common),
                        **paired_statistics(before, after, common),
                    }
                )
        if (
            manifest["kind"] == "dream_smoke"
            and groups.get(("before", "D"))
            and groups.get(("after", "D"))
        ):
            before_d = summarize(groups[("before", "D")])
            after_d = summarize(groups[("after", "D")])
            text += [
                "",
                (
                    f"En D, el formato pasó de {before_d['format']:.0%} a {after_d['format']:.0%}, "
                    f"con {after_d['n']} consignas de validación. Este smoke no demuestra una mejora. "
                    "Hay que revisar la estabilidad del formato y la evaluación humana antes de una comparación larga."
                ),
                "",
            ]
        validation = folder / "training-validation.json"
        if validation.exists():
            v = read_json(validation)
            text += [
                "",
                (
                    f"Adaptadores modificados: **{v['adapters_changed']}**. "
                    f"Pesos base intactos: **{v['base_unchanged']}**. "
                    f"Logits posteriores finitos: **{v.get('post_reload_logits_finite', 'no comprobado')}**."
                ),
                "",
            ]
        steps_table.extend({"run": label, **s} for s in steps)
        text.append("")
    write_csv(output / "evaluations.csv", tables)
    write_csv(output / "training.csv", steps_table)
    write_csv(output / "paired-deltas.csv", paired)
    if steps_table:
        plot_scores(output, steps_table)
    # Revisar relatos completos localmente sin publicar el corpus ni los resultados.
    from .storage import write_json

    write_json(output / "examples.json", examples)
    if examples:
        review_items, review_key = blind_review(dream, examples)
        review_items, review_key = save_blind_review(output, review_items, review_key)
        text += [
            (
                f"Revisión humana preparada: **{len(review_items)} relatos** mezclados en "
                "`revision-ciega.csv`, sin condición, origen ni puntuación del juez. "
                "La clave se guarda por separado en `clave-revision.json`. "
                "La revisión se conserva entre informes; los nuevos paquetes se guardan por separado."
            ),
            "",
        ]
    calls = read_jsonl(dream / "judge-calls.jsonl")
    if calls:
        text += [
            (
                f"Llamadas nuevas al juez: **{len(calls)}**; tiempo medido: "
                f"**{sum(c['seconds'] for c in calls) / 60:.2f} min**. Las llamadas reutilizadas salen de caché."
            ),
            "",
        ]
    if steps_table and tables and "dream-v4" in protocols:
        write_json(
            output / "estimate.json",
            {
                "status": "pending_closed_benchmark",
                "limitations": "La clasificación por fragmentos y requisitos no se extrapola como tres generaciones de juez por relato.",
            },
        )
    elif steps_table and tables and calls:
        generation_per_item = statistics.mean(
            t["generation_seconds"] / t["n"] for t in tables if t["n"]
        )
        judged = [
            c["seconds"]
            for c in calls
            if c["purpose"] in {"dream_likeness", "reference_alignment", "retelling_fidelity"}
        ]
        judge_per_call = statistics.mean(judged) if judged else 0
        step_seconds = statistics.mean(s["seconds"] for s in steps_table)
        test_size = read_json(data_path)["counts"]["test"]
        estimate = {
            "seconds_per_update": step_seconds,
            "seconds_per_generated_item": generation_per_item,
            "seconds_per_judge_call": judge_per_call,
            "twelve_runs_training_seconds_per_step": 12 * step_seconds,
            "twelve_runs_evaluation_seconds": 12
            * 4
            * test_size
            * (generation_per_item + 3 * judge_per_call),
            "limitations": "Extrapolación del smoke; no incluye nuevas consignas, carga ni todos los checkpoints. Tres llamadas por relato, sin asumir caché.",
        }
        write_json(output / "estimate.json", estimate)
        text += [
            (
                f"Estimación conservadora para doce corridas: **{12 * step_seconds / 60:.2f} min × pasos por corrida**, "
                f"más **{estimate['twelve_runs_evaluation_seconds'] / 3600:.2f} h** de evaluación completa. "
                "No se iniciaron. La estimación usa una muestra pequeña y puede variar con las longitudes."
            ),
            "",
        ]
    errors = read_jsonl(dream / "errors.jsonl")
    if errors:
        text += ["## Incidentes", "", *[f"- {e['action']}: `{e['error']}`" for e in errors], ""]
    text += [
        "## Interpretación",
        "",
        (
            "Los scores son juicios de semejanza textual. No prueban que el modelo sueñe subjetivamente "
            "ni que los relatos generados sean experiencias humanas. Un sueño real puede ser cotidiano y coherente. "
            "La referencia reservada es un ejemplo válido entre varias narraciones posibles. El score neutral usa "
            "dos dimensiones porque no tiene una escena previa, por lo que su recompensa total no es directamente "
            "comparable con la de las condiciones factoriales."
        ),
        "",
        (
            "La revisión humana ciega y el contraste con otro juez quedan pendientes antes de afirmar una mejora. "
            "La independencia entre generador y juez depende de los modelos fijados en cada protocolo."
        ),
        "",
    ]
    (output / "informe.md").write_text("\n".join(text))
    print("Informe de sueños:", output / "informe.md", flush=True)


def plot_scores(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    figure.suptitle("Scores del juez en rollouts de entrenamiento · no validación humana")
    groups = defaultdict(list)
    for r in rows:
        groups[r["run"]].append(r)
    for name, values in groups.items():
        for axis, field in zip(axes, ["step", "cumulative_tokens"]):
            for dimension, label in [
                ("reward_dream_likeness", "onírico"),
                ("reward_reference_alignment", "referencia"),
                ("reward_retelling_fidelity", "fidelidad"),
                ("reward_content_compliance", "cumplimiento"),
            ]:
                scored = [v for v in values if v.get(dimension) is not None]
                if not scored:
                    continue
                axis.plot(
                    [v[field] for v in scored],
                    [v[dimension] for v in scored],
                    label=f"{name}: {label}",
                )
            axis.set(ylim=(-0.05, 1.05), ylabel="Score normalizado")
            axis.grid(alpha=0.2)
    axes[0].set_xlabel("Paso")
    axes[1].set_xlabel("Tokens consumidos")
    axes[0].legend(fontsize=7)
    figure.savefig(output / "curvas.png", dpi=160)
    plt.close(figure)


def blind_review(dream, examples):
    generated = {digest(e["answer"]): e for e in examples if e["answer"]}
    ids = {e["problem_id"] for e in examples}
    human = {}
    for split in ["train", "validation", "test"]:
        for row in read_jsonl(dream / "data" / f"{split}.jsonl"):
            if row["problem_id"] in ids:
                human[digest(row["reference"])] = row
    items, keys = [], {}
    for key, e in generated.items():
        items.append(
            {"item_id": key[:16], "relato": e["answer"], "onirico_humano_0_a_4": "", "notas": ""}
        )
        keys[key[:16]] = {
            "origin": "generated",
            "phase": e["phase"],
            "condition": e["condition"],
            "problem_id": e["problem_id"],
            "judge_score": None if e["dream_likeness"] is None else e["dream_likeness"] * 4,
        }
    for key, row in human.items():
        if key in generated:
            keys[key[:16]]["also_exact_human_reference"] = True
            continue
        items.append(
            {
                "item_id": key[:16],
                "relato": row["reference"],
                "onirico_humano_0_a_4": "",
                "notas": "",
            }
        )
        keys[key[:16]] = {"origin": "human_reference", "problem_id": row["problem_id"]}
    items.sort(key=lambda row: digest(["blind-review-v1", row["item_id"]]))
    return items, keys


def save_blind_review(output, items, keys):
    """Congelar el paquete una vez creado: nunca borrar anotaciones humanas al informar."""
    from .storage import write_json

    target, keyfile = output / "revision-ciega.csv", output / "clave-revision.json"
    if target.exists():
        with target.open(newline="") as stream:
            original = list(csv.DictReader(stream))
        if not keyfile.exists():
            raise ValueError(
                "Falta la clave de la revisión existente; se conserva el CSV sin cambios."
            )
        original_keys = read_json(keyfile)
        old_texts = {row["item_id"]: row["relato"] for row in original}
        new_texts = {row["item_id"]: row["relato"] for row in items}
        if old_texts != new_texts or original_keys != keys:
            suffix = digest([new_texts, keys])[:12]
            new_target = output / f"revision-ciega-{suffix}.csv"
            if not new_target.exists():
                write_csv(new_target, items)
                write_json(output / f"clave-revision-{suffix}.json", keys)
        return original, original_keys
    write_csv(target, items)
    write_json(keyfile, keys)
    return items, keys
