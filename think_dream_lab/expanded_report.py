"""Comparación inicial/final del experimento ampliado, con auditoría y juez ciego."""

import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from .embedding_reward import Encoder
from .expanded_experiment import load_expanded_pack
from .holdout_evaluation import sha256, validate_pack
from .holdout_report import cluster_interval, save_scatter
from .paired_experiment import PairedReward, metrics
from .storage import digest, read_json, read_jsonl, write_json


def compare(config, root):
    spec = config["expanded_experiment"]
    pack = load_expanded_pack(config)
    training = read_json(root / "dream/paired/run.json")
    if (
        training["status"] != "complete"
        or training["new_updates_completed"] != 128
        or not training["original_checkpoint_unchanged"]
    ):
        raise ValueError("El entrenamiento ampliado no está completo.")
    bootstrap = read_json(root / "dream/paired/bootstrap-validation.json")
    if not all(
        bootstrap[k]
        for k in ["same_adapters", "same_optimizer", "same_random_states", "base_unchanged"]
    ):
        raise ValueError("La bifurcación no fue una restauración exacta.")
    reload = read_json(root / "dream/paired/reload-validation.json")
    final_state = read_json(Path(training["training_validation"]["checkpoint"]) / "state.json")
    if (
        not reload["same_weights"]
        or reload["checkpoint"] != training["training_validation"]["checkpoint"]
        or final_state["step"] != 160
        or final_state["identity"] != training["identity"]
        or not training["training_validation"]["base_unchanged"]
    ):
        raise ValueError("Falta una recarga exacta del checkpoint final y una base intacta.")
    records = {}
    keyed = {}
    runs = {}
    expected = {(r["problem_id"], c) for r in pack["rows"]["validation"] for c in range(3)}
    encoder = Encoder(config["embeddings"], root)

    def no_new_embeddings(pending):
        raise AssertionError("La auditoría sólo utiliza vectores guardados.")

    encoder._encode_pending = no_new_embeddings
    reward = PairedReward(pack["rows"], encoder)

    def check_score(record, scorer):
        check = scorer.score(record["reward"]["scored_response"], record["problem_id"])
        for key in ["format", "reference_id", "reference_split", "target_words", "answer_words"]:
            if check[key] != record["reward"][key]:
                raise ValueError("Cambió una referencia o formato.")
        for key in ["total", "cosine", "relative_word_error"]:
            a, b = check[key], record["reward"][key]
            if a != b and (a is None or b is None or not np.isclose(a, b, atol=1e-7, rtol=0)):
                raise ValueError("El score no coincide con los vectores guardados.")

    for phase, field in [("before", "evaluation_before"), ("after", "evaluation_after")]:
        folder = Path(spec[field]) / "dream/holdout"
        run = read_json(folder / "checkpoint/run.json")
        runs[phase] = run
        if (
            run["status"] != "complete"
            or not run["parameters_unchanged"]
            or not run["adapter_loaded_exactly"]
            or run["optimizer_created"]
            or run["training_updates"]
        ):
            raise ValueError("Se requieren evaluaciones congeladas y completas.")
        if (
            run["config"]["generation"] != config["generation"]
            or run["config"]["evaluation"] != config["evaluation"]
        ):
            raise ValueError("Cambió la configuración entre evaluaciones.")
        heldout = read_json(folder / "evaluation-pack.json")
        validate_pack(heldout, read_json(folder / "selection.json"), run["config"])
        if heldout["rows"] != pack["rows"]["validation"]:
            raise ValueError("Cambió la cohorte reservada.")
        records[phase] = read_jsonl(folder / "checkpoint/responses.jsonl")
        keyed[phase] = {(r["problem_id"], r["candidate"]): r for r in records[phase]}
        if set(keyed[phase]) != expected or len(records[phase]) != 72:
            raise ValueError("Faltan muestras o hay duplicados.")
        for record in records[phase]:
            check_score(record, reward)
    source = read_json(Path(spec["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    for name, value in training["lineage"]["source_files"].items():
        if sha256(origin / name) != value:
            raise ValueError("Se modificó un archivo del checkpoint original.")
    if runs["before"]["checkpoint_sha256"] != training["lineage"]["source_files"][
        "adapters.safetensors"
    ] or runs["after"]["checkpoint_sha256"] != sha256(
        Path(training["training_validation"]["checkpoint"]) / "adapters.safetensors"
    ):
        raise ValueError("La comparación usa otro checkpoint.")
    steps = read_jsonl(root / "dream/paired/steps.jsonl")
    rollouts = read_jsonl(root / "dream/paired/responses.jsonl")
    if [s["step"] for s in steps] != list(range(33, 161)) or Counter(
        s["problem_id"] for s in steps
    ) != Counter(r["problem_id"] for r in pack["rows"]["train"]):
        raise ValueError("No hubo exactamente una visita por relato de entrenamiento.")
    if len(rollouts) != 512 or Counter(r["step"] for r in rollouts) != Counter(
        {s: 4 for s in range(33, 161)}
    ):
        raise ValueError("Rollouts incompletos o duplicados.")
    if any(
        not math.isfinite(s["loss"])
        or not math.isfinite(s["gradient_norm"])
        or s["gradient_norm"] <= 0
        for s in steps
    ):
        raise ValueError("Una actualización no tiene pérdida finita o gradiente efectivo.")
    for record in rollouts:
        if record["reward"]["reference_split"] != "train":
            raise ValueError("Un reservado participó en una actualización.")
        check_score(record, reward)
    judge = read_json(root / "dream/judge-ratings.json")
    jcheck = read_json(root / "dream/judge-validation.json")
    if (
        not jcheck["complete"]
        or not jcheck["parameters_unchanged"]
        or set(jcheck["phases"]) != {"before", "after"}
        or judge["rubric_hash"] != spec["judge_rubric_hash"]
    ):
        raise ValueError("El juez no completó ambas fases con la rúbrica fijada.")
    mapping = read_json(root / "dream/judge-provenance-key.json")["mapping"]
    byblind = {r["blind_id"]: r for r in judge["ratings"]}
    if len(byblind) != len(judge["ratings"]) or len(mapping) != 144:
        raise ValueError("La evaluación ciega tiene duplicados o una clave incompleta.")
    judge_scores = {phase: {} for phase in ["before", "after"]}
    for item in mapping:
        key = (item["problem_id"], item["candidate"])
        phase = item["phase"]
        if keyed[phase][key]["reward"]["format"]:
            judge_scores[phase][key] = byblind[item["blind_id"]]
    if len(judge["ratings"]) != sum(len(v) for v in judge_scores.values()) or jcheck[
        "cases"
    ] != len(judge["ratings"]):
        raise ValueError("El juez omitió o duplicó alguna respuesta válida.")
    for phase in records:
        expected_valid = {k for k, r in keyed[phase].items() if r["reward"]["format"]}
        if set(judge_scores[phase]) != expected_valid:
            raise ValueError("El juez no cubrió exactamente las respuestas válidas.")
    from .dream_task import extract_blocks
    from .judge import protected_json
    from .reference_judge import RULE, complete_required_prefix

    references = {r["problem_id"]: r["reference"] for r in pack["rows"]["validation"]}
    normalizations = {"before": 0, "after": 0}
    for phase, data in judge_scores.items():
        for key, value in data.items():
            account = extract_blocks(keyed[phase][key]["reward"]["scored_response"], "D")["answer"]
            messages = [
                {"role": "system", "content": RULE},
                {
                    "role": "user",
                    "content": protected_json(
                        {"reference": references[key[0]], "candidate": account}
                    ),
                },
            ]
            cache_key = digest([config["judge"], "reference-closeness-v17", messages])
            call = read_json(root / "dream/judge-cache" / f"{cache_key}.json")
            if (
                call["messages"] != messages
                or call["model"] != config["judge"]
                or call["value"]
                != {k: value[k] for k in ["characters", "events", "relations", "overall", "reason"]}
            ):
                raise ValueError("El juez recibió otros datos o se alteró una nota.")
            if call.get("normalization"):
                if (
                    call["normalization"] != "complete-required-fields-before-extra-output"
                    or call["inferred_scores"]
                    or complete_required_prefix(call["raw"]) != call["value"]
                ):
                    raise ValueError("Una normalización alteró campos de la evaluación.")
                normalizations[phase] += 1
    pairs = []
    per_report = []
    for key in sorted(expected):
        a, b = keyed["before"][key], keyed["after"][key]
        for field in ["prompt", "prompt_ids", "evaluation_seed", "author_group"]:
            if a[field] != b[field]:
                raise ValueError("Cambió una entrada o semilla del par.")
        if a["reward"]["format"] and b["reward"]["format"]:
            pairs.append(
                {
                    "problem_id": key[0],
                    "candidate": key[1],
                    "author_group": a["author_group"],
                    "base": a["reward"]["cosine"],
                    "checkpoint": b["reward"]["cosine"],
                    "delta": b["reward"]["cosine"] - a["reward"]["cosine"],
                    "judge_before": judge_scores["before"][key]["overall"],
                    "judge_after": judge_scores["after"][key]["overall"],
                    "judge_delta": judge_scores["after"][key]["overall"]
                    - judge_scores["before"][key]["overall"],
                }
            )
    for row in pack["rows"]["validation"]:
        common = [p for p in pairs if p["problem_id"] == row["problem_id"]]
        per_report.append(
            {
                "problem_id": row["problem_id"],
                "author_group": row["author_group"],
                "common_valid_pairs": len(common),
                "cosine_delta": statistics.mean(p["delta"] for p in common) if common else None,
                "judge_delta": statistics.mean(p["judge_delta"] for p in common)
                if common
                else None,
            }
        )
    result = {
        "reports": 24,
        "author_groups": 6,
        "n_common_valid_pairs": len(pairs),
        "common_valid_pairs": pairs,
        "per_report": per_report,
        "before": metrics(records["before"]),
        "after": metrics(records["after"]),
        "paired_cosine_before": statistics.mean(p["base"] for p in pairs),
        "paired_cosine_after": statistics.mean(p["checkpoint"] for p in pairs),
        "paired_cosine_delta": statistics.mean(p["delta"] for p in pairs),
        "paired_judge_before": statistics.mean(p["judge_before"] for p in pairs),
        "paired_judge_after": statistics.mean(p["judge_after"] for p in pairs),
        "paired_judge_delta": statistics.mean(p["judge_delta"] for p in pairs),
        "cosine_cluster_interval": cluster_interval(
            [(r["author_group"], r["cosine_delta"]) for r in per_report if r["common_valid_pairs"]]
        ),
        "judge_cluster_interval": cluster_interval(
            [(r["author_group"], r["judge_delta"]) for r in per_report if r["common_valid_pairs"]]
        ),
        "new_updates": 128,
        "train_reports": 128,
        "training_author_groups": 30,
        "rollouts": 512,
        "source_step": 32,
        "final_step": 160,
        "approved_dataset_hash": pack["dataset_hash"],
        "validation_used_for_updates": False,
        "test_reports_evaluated": 0,
        "judge_model": config["judge"],
        "judge_controls": judge["controls"],
        "rubric_hash": judge["rubric_hash"],
        "judge_scored_block": "answer",
        "judge_normalized_cases": normalizations,
        "judge_normalization_deviation": "Después de comenzar el juez, se permitió conservar los cinco campos completos antes de texto adicional, sin inferir campos ni cambiar puntajes, rúbrica, entrada o límite de 256 tokens. No se admite un campo requerido truncado.",
        "judge_used_for_training_reward": False,
        "independent_human_review": False,
        "source_checkpoint_unchanged": True,
        "base_unchanged": training["training_validation"]["base_unchanged"],
        "additional_training_tokens": training["training_validation"]["training_tokens"]
        - source["training_validation"]["training_tokens"],
        "additional_training_seconds": training["training_validation"]["training_seconds"]
        - source["training_validation"]["training_seconds"],
        "training_attempts": [
            {
                "status": attempt["status"],
                "reason": attempt.get("reason"),
                "elapsed_seconds": attempt["elapsed_seconds"],
            }
            for attempt in read_json(root / "budget.json")["runs"]
            if attempt["args"][0] == "expanded-experiment"
        ],
        "caution": "Validación de desarrollo reutilizada; seis grupos. Juez textual congelado, no validación humana independiente. No mide el contenido de dream ni demuestra experiencia subjetiva.",
    }
    dimensions = {}
    for name in ["characters", "events", "relations", "overall"]:
        left = [judge_scores["before"][(p["problem_id"], p["candidate"])][name] for p in pairs]
        right = [judge_scores["after"][(p["problem_id"], p["candidate"])][name] for p in pairs]
        dimensions[name] = {
            "before": statistics.mean(left),
            "after": statistics.mean(right),
            "delta": statistics.mean(b - a for a, b in zip(left, right)),
        }
    result["judge_dimensions_common"] = dimensions
    result["report_direction_counts"] = {
        metric: {
            direction: sum(
                1 for row in per_report if row["common_valid_pairs"] and predicate(row[field])
            )
            for direction, predicate in {
                "higher": lambda value: value > 1e-12,
                "lower": lambda value: value < -1e-12,
                "equal": lambda value: abs(value) <= 1e-12,
            }.items()
        }
        for metric, field in [("cosine", "cosine_delta"), ("judge", "judge_delta")]
    }
    target = root / "dream/report"
    write_json(target / "comparison.json", result)
    save_scatter(
        target,
        {
            "common_valid_pairs": pairs,
            "per_report": per_report,
            "reports": 24,
            "base_label": "Checkpoint 32 · antes",
            "checkpoint_label": "Checkpoint 160 · después",
            "cohort_label": "Lote ampliado · validación de desarrollo",
        },
    )
    audit = {
        "passed": True,
        "new_updates": 128,
        "unique_training_reports": 128,
        "training_rollouts_recomputed": 512,
        "evaluation_scores_recomputed": 144,
        "new_embedding_inference": False,
        "reserved_in_updates": False,
        "source_files_unchanged": True,
        "bootstrap_exact": True,
        "final_checkpoint_reload_exact": True,
        "finite_losses_and_nonzero_gradients": True,
        "judge_parameters_unchanged": True,
        "judge_normalizations_preserve_complete_fields": True,
        "same_prompts_tokens_seeds": True,
        "test_reports_evaluated": 0,
        "approved_dataset_hash": pack["dataset_hash"],
    }
    write_json(target / "final-audit.json", audit)
    lines = [
        "# Una pasada por 128 relatos: antes y después",
        "",
        "Origen: checkpoint 32. Destino fijado antes de entrenar: 160, 128 actualizaciones y 512 rollouts. Los reservados no participaron en las actualizaciones. Ambas evaluaciones usaron 2048 tokens, mismos prompts y semillas.",
        "",
        "| Medida | Antes | Después |",
        "| --- | ---: | ---: |",
    ]
    for key, label in [
        ("n_valid", "Formato válido /72"),
        ("source_cosine_valid_mean", "Coseno, sólo válidas"),
        ("reward_mean_all", "Score integral"),
        ("truncations", "Truncamientos"),
        ("relative_word_error_valid_mean", "Error relativo de longitud"),
    ]:
        a, b = (result[p][key] for p in ["before", "after"])
        if key in {"n_valid", "truncations"}:
            lines.append(f"| {label} | {a} | {b} |")
        else:
            lines.append(f"| {label} | {a:.4f} | {b:.4f} |")
    lines += [
        "",
        f"{len(pairs)} pares válidos: coseno {result['paired_cosine_before']:.4f} → {result['paired_cosine_after']:.4f} ({result['paired_cosine_delta']:+.4f}).",
        "",
        f"Juez independiente, cercanía global 0–4 en los mismos pares: {result['paired_judge_before']:.4f} → {result['paired_judge_after']:.4f} ({result['paired_judge_delta']:+.4f}).",
        "",
        "![Cercanía pareada](cercania-pareada.png)",
        "",
        "## Incertidumbre por persona",
        "",
        "Intervalos exploratorios: seis grupos, 20000 réplicas. Promedian primero por relato y después por persona; su estimador difiere de la media de todos los pares válidos.",
        "",
    ]
    for key, label in [("cosine_cluster_interval", "Coseno"), ("judge_cluster_interval", "Juez")]:
        ci = result[key]
        lines += [
            f"{label}: media por relato/persona {ci['mean']:+.4f}, intervalo 95% [{ci['low']:+.4f}, {ci['high']:+.4f}].",
            "",
        ]
    lines += [
        "## Dimensiones del juez en pares comunes",
        "",
        "| Dimensión | Antes | Después | Cambio |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "characters": "Personajes, objetos y escenario",
        "events": "Sucesos",
        "relations": "Relaciones",
        "overall": "Cercanía global",
    }
    for name, values in dimensions.items():
        lines.append(
            f"| {labels[name]} | {values['before']:.4f} | {values['after']:.4f} | {values['delta']:+.4f} |"
        )
    lines.append("")
    lines += [
        "## Por relato",
        "",
        "| Caso | Pares | Cambio de coseno | Cambio del juez |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in per_report:
        lines.append(
            f"| {row['problem_id'][:12]} | {row['common_valid_pairs']} | {row['cosine_delta']:+.4f} | {row['judge_delta']:+.4f} |"
            if row["common_valid_pairs"]
            else f"| {row['problem_id'][:12]} | 0 | — | — |"
        )
    lines += [
        "",
        "## Límites",
        "",
        result["caution"],
        "",
        "La rúbrica del juez se fijó con controles sintéticos antes de entrenar; recibió sólo referencia y answer en orden mezclado, sin modelo, paso, fase, semillas ni dream. Su score no participó en GRPO. La recompensa MiniLM y los hiperparámetros se conservaron; el lote y el límite de generación cambiaron frente a los pilotos previos. No se atribuye causalmente el resultado sólo a variedad de datos.",
        "",
        "Los segundos de entrenamiento adicionales cuentan las actualizaciones completas. Los intentos y sus tiempos totales, incluidas interrupciones y recargas, se registran aparte en training_attempts.",
        "",
        result["judge_normalization_deviation"],
        "",
        f"Casos normalizados: {normalizations['before']} antes y {normalizations['after']} después. Las salidas originales y los errores permanecen guardados para auditoría.",
        "",
        "Fuente DreamBank, CC BY-NC-SA 4.0, dreambank.net. Referencias, pesos y generaciones completas permanecen fuera de Git.",
        "",
    ]
    (target / "comparacion.md").write_text("\n".join(lines))
    print("Informe ampliado:", target / "comparacion.md")
    return result


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    compare(load_config(args.config), args.results.resolve())
