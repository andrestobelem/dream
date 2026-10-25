"""Resultados de la cohorte nueva, separados del piloto y agrupados por persona."""

import statistics
from collections import defaultdict

import numpy as np

from .holdout_evaluation import sha256, validate_pack
from .paired_experiment import metrics
from .storage import read_json, read_jsonl, write_json


def cluster_interval(values, seed=20261025, samples=20000):
    """Remuestrear personas completas, no tratar cada generación como independiente."""
    grouped = defaultdict(list)
    for group, value in values:
        grouped[group].append(value)
    means = np.array([statistics.mean(v) for _, v in sorted(grouped.items())])
    if len(means) < 2:
        return {
            "groups": len(means),
            "mean": float(means.mean()) if len(means) else None,
            "low": None,
            "high": None,
            "samples": samples,
        }
    indices = np.random.default_rng(seed).integers(0, len(means), size=(samples, len(means)))
    estimates = means[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "groups": len(means),
        "mean": float(means.mean()),
        "low": float(low),
        "high": float(high),
        "samples": samples,
        "seed": seed,
        "unit": "author_group",
        "caution": "Intervalo exploratorio con pocos grupos.",
    }


def compare(root):
    folder = root / "dream/holdout"
    pack = read_json(folder / "evaluation-pack.json")
    selection = read_json(folder / "selection.json")
    runs = {v: read_json(folder / v / "run.json") for v in ("base", "checkpoint")}
    config = runs["base"]["config"]
    validate_pack(pack, selection, config)
    count = config["evaluation"]["samples_per_problem"]
    expected = {(r["problem_id"], i) for r in pack["rows"] for i in range(count)}
    byid = {r["problem_id"]: r for r in pack["rows"]}
    records, keyed = {}, {}
    for variant, run in runs.items():
        if (
            run["status"] != "complete"
            or not run["parameters_unchanged"]
            or run["training_updates"] != 0
            or run["optimizer_created"]
            or run["cohort_hash"] != pack["rows_hash"]
        ):
            raise ValueError("La comparación requiere dos evaluaciones completas y congeladas.")
        if run["config"] != config:
            raise ValueError("Los modelos no se evaluaron con la misma configuración.")
        records[variant] = read_jsonl(folder / variant / "responses.jsonl")
        keyed[variant] = {(r["problem_id"], r["candidate"]): r for r in records[variant]}
        if set(keyed[variant]) != expected or len(records[variant]) != len(expected):
            raise ValueError("Faltan muestras o hay generaciones duplicadas.")
    if (
        not runs["checkpoint"]["adapter_loaded_exactly"]
        or runs["base"]["checkpoint_sha256"] != runs["checkpoint"]["checkpoint_sha256"]
    ):
        raise ValueError("El checkpoint no coincide entre fases.")
    from pathlib import Path

    source = read_json(Path(config["heldout_evaluation"]["source_run"]))
    checkpoint = Path(source["training_validation"]["checkpoint"])
    if sha256(checkpoint / "adapters.safetensors") != runs["checkpoint"]["checkpoint_sha256"]:
        raise ValueError("Se modificó el checkpoint después de evaluarlo.")
    paired = []
    for key in sorted(expected):
        a, b = keyed["base"][key], keyed["checkpoint"][key]
        row = byid[key[0]]
        if (
            a["prompt"] != row["prompt"]
            or b["prompt"] != row["prompt"]
            or a["prompt_ids"] != b["prompt_ids"]
            or a["evaluation_seed"] != b["evaluation_seed"]
            or any(r["reward"]["reference_id"] != key[0] for r in (a, b))
        ):
            raise ValueError("Un par cambió prompt, tokens, semilla o referencia esperada.")
        if a["reward"]["format"] and b["reward"]["format"]:
            paired.append(
                {
                    "problem_id": key[0],
                    "candidate": key[1],
                    "author_group": row["author_group"],
                    "base": a["reward"]["cosine"],
                    "checkpoint": b["reward"]["cosine"],
                    "delta": b["reward"]["cosine"] - a["reward"]["cosine"],
                }
            )
    per_report = []
    for row in pack["rows"]:
        pid = row["problem_id"]
        item = {
            "problem_id": pid,
            "author_group": row["author_group"],
            "target_words": row["target_words"],
        }
        for variant, values in records.items():
            item[variant] = metrics([r for r in values if r["problem_id"] == pid])
        item["reward_delta"] = (
            item["checkpoint"]["reward_mean_all"] - item["base"]["reward_mean_all"]
        )
        common = [r["delta"] for r in paired if r["problem_id"] == pid]
        item["common_valid_pairs"] = len(common)
        item["common_valid_cosine_delta"] = statistics.mean(common) if common else None
        per_report.append(item)
    result = {
        "cohort_hash": pack["rows_hash"],
        "selection_hash": pack["selection_hash"],
        "reports": len(pack["rows"]),
        "author_groups": len({r["author_group"] for r in pack["rows"]}),
        "base": metrics(records["base"]),
        "checkpoint": metrics(records["checkpoint"]),
        "common_valid_pairs": paired,
        "n_common_valid_pairs": len(paired),
        "common_valid_cosine_delta_mean": statistics.mean(r["delta"] for r in paired)
        if paired
        else None,
        "reward_cluster_interval": cluster_interval(
            [(r["author_group"], r["reward_delta"]) for r in per_report]
        ),
        "cosine_cluster_interval": cluster_interval(
            [
                (r["author_group"], r["common_valid_cosine_delta"])
                for r in per_report
                if r["common_valid_cosine_delta"] is not None
            ]
        ),
        "per_report": per_report,
        "training_updates": 0,
        "checkpoint_sha256": runs["checkpoint"]["checkpoint_sha256"],
        "both_models_unchanged": True,
        "historical_four_cases_pooled": False,
        "caution": "Cohorte nueva de validación; seis grupos de personas. No valida oniricidad ni excluye memorización previa.",
    }
    target = root / "dream/report"
    write_json(target / "comparison.json", result)
    save_scatter(target, result)
    write_markdown(target / "comparacion.md", result, pack, keyed)
    print("Comparación ampliada:", target / "comparacion.md")
    return result


def save_scatter(folder, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({r["author_group"] for r in result["common_valid_pairs"]})
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.22)
    plotted = 0
    for index, group in enumerate(groups):
        points = []
        for report in result["per_report"]:
            if report["author_group"] != group:
                continue
            pairs = [
                r for r in result["common_valid_pairs"] if r["problem_id"] == report["problem_id"]
            ]
            if pairs:
                points.append(
                    (
                        statistics.mean(r["base"] for r in pairs),
                        statistics.mean(r["checkpoint"] for r in pairs),
                    )
                )
        if points:
            x, y = zip(*points)
            ax.scatter(
                x, y, s=55, alpha=0.8, label=f"Grupo {index + 1}", edgecolor="white", linewidth=0.5
            )
            plotted += len(points)
    ax.plot([-1, 1], [-1, 1], color="0.55", linestyle="--", linewidth=1)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel=f"{result.get('base_label', 'Modelo base')} · coseno al relato humano",
        ylabel=f"{result.get('checkpoint_label', 'Checkpoint 16')} · coseno al relato humano",
        title=f"{result.get('cohort_label', 'Relatos reservados nuevos')} · {plotted}/{result['reports']} con pares válidos",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(
        0.5,
        0.025,
        "Un punto por relato; medias de las posiciones válidas en ambos modelos.\n"
        "Sobre la diagonal: mayor cercanía con el checkpoint. Se excluyen fallos de formato.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(folder / "cercania-pareada.png", dpi=160, bbox_inches="tight")
    fig.savefig(folder / "cercania-pareada.svg", bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, result, pack, keyed):
    def fmt(value):
        if value is None:
            return "—"
        return str(value) if isinstance(value, int) else f"{value:.4f}"

    lines = [
        "# Evaluación ampliada: checkpoint 16 frente al modelo base",
        "",
        (
            f"{result['reports']} relatos nuevos de validación, {result['author_groups']} grupos, tres generaciones por modelo y relato. "
            "Los cuatro casos del piloto anterior quedan fuera. No se entrenó ni seleccionó un nuevo checkpoint."
        ),
        "",
        "| Medida | Base | Checkpoint 16 |",
        "| --- | ---: | ---: |",
    ]
    names = {
        "n": "Generaciones",
        "n_valid": "Formato válido",
        "source_cosine_valid_mean": "Coseno medio, sólo válidas",
        "reward_mean_all": "Score medio incluyendo cero por formato inválido",
        "truncations": "Truncamientos",
        "relative_word_error_valid_mean": "Error relativo de longitud, sólo válidas",
    }
    for key, label in names.items():
        lines.append(f"| {label} | {fmt(result['base'][key])} | {fmt(result['checkpoint'][key])} |")
    lines += [
        "",
        (
            f"Pares válidos en ambos modelos: {result['n_common_valid_pairs']}. Cambio medio de coseno: "
            f"{fmt(result['common_valid_cosine_delta_mean'])}."
        ),
        "",
        "## Incertidumbre por persona",
        "",
        (
            "Remuestreo de grupos completos, 20.000 réplicas y percentiles 2,5/97,5. "
            "Con seis grupos estos intervalos son exploratorios; las generaciones no se tratan como personas independientes."
        ),
        "",
    ]
    lines += ["![Cercanía pareada por relato](cercania-pareada.png)", ""]
    for key, label in [
        ("reward_cluster_interval", "Score con formato"),
        ("cosine_cluster_interval", "Coseno en pares válidos"),
    ]:
        ci = result[key]
        lines += [
            (
                f"{label}: cambio medio por grupo {fmt(ci['mean'])}; intervalo [{fmt(ci['low'])}, {fmt(ci['high'])}], "
                f"{ci['groups']} grupos."
            ),
            "",
        ]
    lines += [
        (
            "El intervalo de coseno promedia primero por relato y después por persona, con igual peso. "
            "Su estimador difiere de la media de todos los pares válidos; no es el intervalo de esa media."
        ),
        "",
        "## Por relato",
        "",
        "| Caso | Grupo | Válidas base/checkpoint | Coseno base/checkpoint | Cambio de score |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for r in result["per_report"]:
        a, b = r["base"], r["checkpoint"]
        lines.append(
            f"| {r['problem_id'][:12]} | {r['author_group'][:8]} | {a['n_valid']}/3 → {b['n_valid']}/3 | "
            f"{fmt(a['source_cosine_valid_mean'])} → {fmt(b['source_cosine_valid_mean'])} | {fmt(r['reward_delta'])} |"
        )
    lines += [
        "",
        "## Ejemplos",
        "",
        "Candidato 1 de la primera referencia de cada grupo, elegido por orden y no por score.",
        "",
    ]
    groups = set()
    for row in pack["rows"]:
        if row["author_group"] in groups:
            continue
        groups.add(row["author_group"])
        lines += [
            f"### {row['problem_id'][:12]}",
            "",
            row["seeds"],
            "",
            f"Longitud esperada: {row['target_words']} palabras.",
            "",
        ]
        for variant in ("base", "checkpoint"):
            rec = keyed[variant][(row["problem_id"], 0)]
            lines += [
                f"#### {variant}",
                "",
                f"Coseno: {fmt(rec['reward']['cosine'])}; formato: {rec['reward']['format']}.",
                "",
                rec["reward"]["scored_response"],
                "",
            ]
    lines += [
        "## Integridad y límites",
        "",
        (
            "Mismo prompt, referencia, longitud y semilla por muestra. "
            "Los modelos y el archivo de adaptadores quedaron intactos; no se creó optimizador. "
            "Las semillas se fijaron antes de evaluar ambos modelos. Los 200 relatos de prueba no se utilizaron."
        ),
        "",
        result["caution"],
        "",
        (
            "Fuente DreamBank, CC BY-NC-SA 4.0, atribución a dreambank.net. "
            "Referencias originales y salidas completas son artefactos locales fuera de Git."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
