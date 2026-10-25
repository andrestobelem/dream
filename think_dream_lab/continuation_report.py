"""Comparar base, origen y continuación usando exactamente la cohorte ya fijada."""

import argparse
import statistics
from pathlib import Path

from .holdout_evaluation import sha256, validate_pack
from .holdout_report import cluster_interval, save_scatter
from .paired_experiment import metrics
from .storage import read_json, read_jsonl, write_json


def paired_summary(rows, keyed, first, second):
    pairs, reports = [], []
    for row in rows:
        selected = []
        for candidate in range(3):
            a, b = (keyed[k][(row["problem_id"], candidate)] for k in (first, second))
            if a["reward"]["format"] and b["reward"]["format"]:
                item = {
                    "problem_id": row["problem_id"],
                    "author_group": row["author_group"],
                    "candidate": candidate,
                    "base": a["reward"]["cosine"],
                    "checkpoint": b["reward"]["cosine"],
                    "delta": b["reward"]["cosine"] - a["reward"]["cosine"],
                }
                pairs.append(item)
                selected.append(item["delta"])
        reports.append(
            {
                "problem_id": row["problem_id"],
                "author_group": row["author_group"],
                "common_valid_pairs": len(selected),
                "common_valid_cosine_delta": statistics.mean(selected) if selected else None,
            }
        )
    return {
        "first": first,
        "second": second,
        "n_common_valid_pairs": len(pairs),
        "cosine_first": statistics.mean(p["base"] for p in pairs) if pairs else None,
        "cosine_second": statistics.mean(p["checkpoint"] for p in pairs) if pairs else None,
        "common_valid_cosine_delta_mean": statistics.mean(p["delta"] for p in pairs)
        if pairs
        else None,
        "common_valid_pairs": pairs,
        "per_report": reports,
        "reports_improved": sum(
            r["common_valid_cosine_delta"] > 0 for r in reports if r["common_valid_pairs"]
        ),
        "reports_worse": sum(
            r["common_valid_cosine_delta"] < 0 for r in reports if r["common_valid_pairs"]
        ),
        "cosine_cluster_interval": cluster_interval(
            [
                (r["author_group"], r["common_valid_cosine_delta"])
                for r in reports
                if r["common_valid_pairs"]
            ]
        ),
    }


def compare(root, previous, baseline=Path("results/dream-v7-eval")):
    train = read_json(root / "dream/paired/run.json")
    source = read_json(Path(train["lineage"]["source_run"]))
    start = train["lineage"]["source_step"]
    end = train["training_validation"]["steps_completed"]
    if (
        train["status"] != "complete"
        or source["status"] != "complete"
        or source["training_validation"]["steps_completed"] != start
        or train["lineage"]["target_step"] != end
        or not start < end <= start + 16
        or train["validation_used_for_updates"]
        or not train["training_validation"]["base_unchanged"]
    ):
        raise ValueError(
            "La continuación no coincide con su tramo completo y separado de validación."
        )
    first, second = f"step{start}", f"step{end}"
    folder = root / "dream/holdout"
    pack = read_json(folder / "evaluation-pack.json")
    old = read_json(previous / "dream/holdout/evaluation-pack.json")
    if pack["rows_hash"] != old["rows_hash"] or pack["rows"] != old["rows"]:
        raise ValueError("La cohorte debe conservar todos sus prompts y referencias.")
    paths = {
        "base": baseline / "dream/holdout/base",
        first: previous / "dream/holdout/checkpoint",
        second: folder / "checkpoint",
    }
    keyed, runs = {}, {}
    expected = {(r["problem_id"], c) for r in pack["rows"] for c in range(3)}
    for name, path in paths.items():
        run = read_json(path / "run.json")
        runs[name] = run
        if (
            run["status"] != "complete"
            or not run["parameters_unchanged"]
            or run["training_updates"]
            or run["optimizer_created"]
        ):
            raise ValueError("Se requiere una evaluación completa y sin actualizaciones.")
        for key in ["model", "generation", "evaluation", "embeddings", "lora"]:
            if run["config"][key] != runs["base"]["config"][key]:
                raise ValueError("Cambió un parámetro de la comparación.")
        own_pack = read_json(path.parent / "evaluation-pack.json")
        if own_pack["rows"] != pack["rows"] or run["cohort_hash"] != pack["rows_hash"]:
            raise ValueError("Un modelo se evaluó con otras referencias o prompts.")
        validate_pack(
            own_pack,
            read_json(path.parent / "selection.json"),
            run["config"],
        )
        records = read_jsonl(path / "responses.jsonl")
        keyed[name] = {(r["problem_id"], r["candidate"]): r for r in records}
        if len(records) != 72 or set(keyed[name]) != expected:
            raise ValueError("Muestras incompletas o duplicadas.")
        if name != "base" and not run["adapter_loaded_exactly"]:
            raise ValueError("Los adaptadores evaluados no coinciden con el checkpoint.")
    for run, training in [(runs[first], source), (runs[second], train)]:
        if (
            sha256(Path(training["training_validation"]["checkpoint"]) / "adapters.safetensors")
            != run["checkpoint_sha256"]
        ):
            raise ValueError("Se modificó un checkpoint evaluado.")
    if train["lineage"]["source_checkpoint_sha256"] != runs[first]["checkpoint_sha256"]:
        raise ValueError("La continuación no partió del checkpoint evaluado anteriormente.")
    train_groups = {
        r["author_group"]
        for r in read_json(root / "dream/paired/frozen-training-pack.json")["rows"]["train"]
    }
    if train_groups & {r["author_group"] for r in pack["rows"]}:
        raise ValueError("Una persona reservada entró en entrenamiento.")
    for row in pack["rows"]:
        for c in range(3):
            key = (row["problem_id"], c)
            reference = keyed["base"][key]
            for name in paths:
                record = keyed[name][key]
                if (
                    record["prompt"] != row["prompt"]
                    or record["reward"]["reference_id"] != row["problem_id"]
                ):
                    raise ValueError("Cambió el prompt o la referencia asociada.")
                for field in ["prompt_ids", "evaluation_seed", "author_group"]:
                    if record[field] != reference[field]:
                        raise ValueError("Cambió la entrada o la semilla de un par.")
    comparisons = {
        f"{a}_to_{second}": paired_summary(pack["rows"], keyed, a, second) for a in ["base", first]
    }
    result = {
        "reports": 24,
        "author_groups": 6,
        "cohort_hash": pack["rows_hash"],
        "metrics": {k: metrics(list(v.values())) for k, v in keyed.items()},
        "comparisons": comparisons,
        "training_steps_before": start,
        "training_steps_after": end,
        "additional_training_steps": end - start,
        "validation_used_for_updates": False,
        f"checkpoint{start}_sha256": runs[first]["checkpoint_sha256"],
        f"checkpoint{end}_sha256": runs[second]["checkpoint_sha256"],
        "historical_generations_reused_without_regeneration": True,
        "caution": "Los 24 relatos son validación de desarrollo ya observada, no una nueva prueba ciega. Encoder compartido con recompensa; seis grupos; no evalúa oniricidad.",
    }
    target = root / "dream/report"
    write_json(target / "comparison.json", result)
    graph = {
        **comparisons[f"{first}_to_{second}"],
        "reports": 24,
        "base_label": f"Checkpoint {start}",
        "checkpoint_label": f"Checkpoint {end}",
        "cohort_label": "Mismos relatos reservados",
    }
    save_scatter(target, graph)
    labels = {
        "n_valid": "Formato válido /72",
        "source_cosine_valid_mean": "Coseno, sólo válidas",
        "reward_mean_all": "Score incluyendo fallos",
        "truncations": "Truncamientos",
        "relative_word_error_valid_mean": "Error relativo de longitud, sólo válidas",
    }
    lines = [
        f"# Continuación del paso {start} al {end}",
        "",
        f"Mismos 24 relatos reservados y tres muestras por modelo y relato. Base y paso {start} reutilizan sus salidas originales. Sólo los ocho relatos de entrenamiento actualizaron pesos.",
        "",
        f"| Medida | Base | Paso {start} | Paso {end} |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in labels.items():
        values = [result["metrics"][n][key] for n in paths]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in values)
            + " |"
        )
    lines += ["", f"![Paso {start} frente al {end}](cercania-pareada.png)", ""]
    for name, p in comparisons.items():
        ci = p["cosine_cluster_interval"]
        lines += [
            f"## {p['first']} → paso {end}",
            "",
            f"{p['n_common_valid_pairs']} pares válidos: coseno {p['cosine_first']:.4f} → {p['cosine_second']:.4f}; cambio medio {p['common_valid_cosine_delta_mean']:+.4f}.",
            "",
            f"Por relato: {p['reports_improved']} medias mejoran y {p['reports_worse']} empeoran.",
            "",
            f"Media con igual peso por relato/persona: {ci['mean']:+.4f}; intervalo exploratorio 95% [{ci['low']:+.4f}, {ci['high']:+.4f}], seis grupos y 20.000 réplicas. Este estimador difiere de la media de todos los pares; no es su intervalo.",
            "",
            "| Caso | Pares válidos | Cambio de coseno |",
            "| --- | ---: | ---: |",
        ]
        for row in p["per_report"]:
            delta = row["common_valid_cosine_delta"]
            lines.append(
                f"| {row['problem_id'][:12]} | {row['common_valid_pairs']} | {delta:+.4f} |"
                if delta is not None
                else f"| {row['problem_id'][:12]} | 0 | — |"
            )
        lines.append("")
    lines += [
        "## Límites",
        "",
        result["caution"],
        "",
        f"Mismos prompts, longitudes y semillas. Formatos inválidos no tienen coseno. El checkpoint {start} y la base quedaron intactos. Los 200 relatos de prueba no se usaron. Referencias y generaciones completas permanecen fuera de Git. Fuente DreamBank, CC BY-NC-SA 4.0, dreambank.net. Código MIT.",
        "",
    ]
    (target / "comparacion.md").write_text("\n".join(lines))
    print("Comparación:", target / "comparacion.md")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("results/dream-v7-eval"))
    parser.add_argument("--baseline", type=Path, default=Path("results/dream-v7-eval"))
    args = parser.parse_args()
    compare(args.results.resolve(), args.previous.resolve(), args.baseline.resolve())
