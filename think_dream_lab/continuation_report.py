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


def compare(root, previous):
    folder = root / "dream/holdout"
    pack = read_json(folder / "evaluation-pack.json")
    old = read_json(previous / "dream/holdout/evaluation-pack.json")
    if pack["rows_hash"] != old["rows_hash"] or pack["rows"] != old["rows"]:
        raise ValueError("La cohorte debe conservar todos sus prompts y referencias.")
    paths = {
        "base": previous / "dream/holdout/base",
        "step16": previous / "dream/holdout/checkpoint",
        "step32": folder / "checkpoint",
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
        validate_pack(
            pack if name == "step32" else old,
            read_json(path.parent / "selection.json"),
            run["config"],
        )
        records = read_jsonl(path / "responses.jsonl")
        keyed[name] = {(r["problem_id"], r["candidate"]): r for r in records}
        if len(records) != 72 or set(keyed[name]) != expected:
            raise ValueError("Muestras incompletas o duplicadas.")
        if name != "base" and not run["adapter_loaded_exactly"]:
            raise ValueError("Los adaptadores evaluados no coinciden con el checkpoint.")
    train = read_json(root / "dream/paired/run.json")
    source = read_json(Path(train["lineage"]["source_run"]))
    for run, training in [(runs["step16"], source), (runs["step32"], train)]:
        if (
            sha256(Path(training["training_validation"]["checkpoint"]) / "adapters.safetensors")
            != run["checkpoint_sha256"]
        ):
            raise ValueError("Se modificó un checkpoint evaluado.")
    if train["lineage"]["source_checkpoint_sha256"] != runs["step16"]["checkpoint_sha256"]:
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
        f"{a}_to_step32": paired_summary(pack["rows"], keyed, a, "step32")
        for a in ["base", "step16"]
    }
    result = {
        "reports": 24,
        "author_groups": 6,
        "cohort_hash": pack["rows_hash"],
        "metrics": {k: metrics(list(v.values())) for k, v in keyed.items()},
        "comparisons": comparisons,
        "training_steps_before": 16,
        "training_steps_after": 32,
        "additional_training_steps": 16,
        "validation_used_for_updates": False,
        "checkpoint16_sha256": runs["step16"]["checkpoint_sha256"],
        "checkpoint32_sha256": runs["step32"]["checkpoint_sha256"],
        "historical_generations_reused_without_regeneration": True,
        "caution": "Los 24 relatos son validación de desarrollo ya observada, no una nueva prueba ciega. Encoder compartido con recompensa; seis grupos; no evalúa oniricidad.",
    }
    target = root / "dream/report"
    write_json(target / "comparison.json", result)
    graph = {
        **comparisons["step16_to_step32"],
        "reports": 24,
        "base_label": "Checkpoint 16",
        "checkpoint_label": "Checkpoint 32",
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
        "# Continuación del paso 16 al 32",
        "",
        "Mismos 24 relatos reservados y tres muestras por modelo y relato. Base y paso 16 reutilizan sus salidas originales. Sólo los ocho relatos de entrenamiento actualizaron pesos.",
        "",
        "| Medida | Base | Paso 16 | Paso 32 |",
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
    lines += ["", "![Paso 16 frente al 32](cercania-pareada.png)", ""]
    for name, p in comparisons.items():
        ci = p["cosine_cluster_interval"]
        lines += [
            f"## {p['first']} → paso 32",
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
        "Mismos prompts, longitudes y semillas. Formatos inválidos no tienen coseno. El checkpoint 16 y la base quedaron intactos. Los 200 relatos de prueba no se usaron. Referencias y generaciones completas permanecen fuera de Git. Fuente DreamBank, CC BY-NC-SA 4.0, dreambank.net. Código MIT.",
        "",
    ]
    (target / "comparacion.md").write_text("\n".join(lines))
    print("Comparación:", target / "comparacion.md")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("results/dream-v7-eval"))
    args = parser.parse_args()
    compare(args.results.resolve(), args.previous.resolve())
