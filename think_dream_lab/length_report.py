"""Diagnóstico pareado del límite de generación, sin entrenar."""

from pathlib import Path

from .embedding_reward import Encoder
from .holdout_evaluation import sha256, validate_pack
from .paired_experiment import PairedReward, metrics
from .storage import read_json, read_jsonl, write_json


def pair_status(a, b):
    for field in ["prompt", "prompt_ids", "evaluation_seed"]:
        if a[field] != b[field]:
            raise ValueError("Cambió el prompt o la semilla de un par.")
    prefix_matches = b["output_ids"][: len(a["output_ids"])] == a["output_ids"]
    return {
        "prefix_matches": prefix_matches,
        "old_truncated": a["truncated"],
        "new_truncated": b["truncated"],
        "old_valid": a["reward"]["format"],
        "new_valid": b["reward"]["format"],
        "old_tokens": len(a["output_ids"]),
        "new_tokens": len(b["output_ids"]),
        "old_cosine": a["reward"]["cosine"],
        "new_cosine": b["reward"]["cosine"],
        "format_rescued": bool(prefix_matches and a["truncated"] and b["reward"]["format"]),
    }


def compare(root):
    design = read_json(root / "length-design.json")
    result = {"design": design, "checkpoints": {}, "training_updates": 0}
    for step in design["checkpoints"]:
        target = root / f"length-step{step}"
        folder = target / "dream/holdout"
        run = read_json(folder / "checkpoint/run.json")
        config = run["config"]
        if (
            run["status"] != "complete"
            or not run["parameters_unchanged"]
            or not run["adapter_loaded_exactly"]
            or run["optimizer_created"]
            or run["training_updates"]
        ):
            raise ValueError("El diagnóstico requiere modelos congelados sin optimizador.")
        baseline = Path(design["baselines"][str(step)]["path"])
        if (
            sha256(baseline / "responses.jsonl")
            != design["baselines"][str(step)]["responses_sha256"]
        ):
            raise ValueError("Cambió la evaluación de 1024 tokens.")
        if run["checkpoint_sha256"] != design["baselines"][str(step)]["checkpoint_sha256"]:
            raise ValueError("Se evaluó otro checkpoint.")
        baseline_config = read_json(baseline / "run.json")["config"]
        for key in ["model", "lora", "embeddings", "evaluation"]:
            if config[key] != baseline_config[key]:
                raise ValueError("Cambió un parámetro del diagnóstico.")
        if {k: v for k, v in config["generation"].items() if k != "max_tokens"} != {
            k: v for k, v in baseline_config["generation"].items() if k != "max_tokens"
        }:
            raise ValueError("Cambió algo más que el límite de tokens.")
        pack = read_json(folder / "evaluation-pack.json")
        validate_pack(pack, read_json(folder / "selection.json"), config)
        if pack["rows_hash"] != design["rows_hash"]:
            raise ValueError("La cohorte no coincide con el diseño fijado.")
        old = {
            (r["problem_id"], r["candidate"]): r
            for r in read_jsonl(baseline / "responses.jsonl")
            if r["problem_id"] in design["ids"]
        }
        records = read_jsonl(folder / "checkpoint/responses.jsonl")
        new = {(r["problem_id"], r["candidate"]): r for r in records}
        expected = {(pid, c) for pid in design["ids"] for c in range(3)}
        if set(old) != expected or set(new) != expected or len(records) != len(expected):
            raise ValueError("Muestras incompletas o duplicadas.")
        encoder = Encoder(config["embeddings"], target)

        def no_inference(pending):
            raise AssertionError("La auditoría sólo usa vectores ya guardados.")

        encoder._encode_pending = no_inference
        reward = PairedReward({"validation": pack["rows"]}, encoder)
        paired = []
        for key in sorted(expected):
            a, b = old[key], new[key]
            for field in ["prompt", "prompt_ids", "evaluation_seed"]:
                if a[field] != b[field]:
                    raise ValueError("Cambió el prompt o la semilla de un par.")
            for record in [a, b]:
                check = reward.score(record["reward"]["scored_response"], key[0])
                for field in ["total", "cosine", "format", "answer_words", "target_words"]:
                    x, y = check[field], record["reward"][field]
                    if x != y and (x is None or y is None or abs(x - y) > 1e-7):
                        raise ValueError(
                            "El score no coincide con la referencia y vectores guardados."
                        )
            paired.append({"problem_id": key[0], "candidate": key[1], **pair_status(a, b)})
        result["checkpoints"][str(step)] = {
            "1024": metrics(list(old.values())),
            "2048": metrics(records),
            "pairs": paired,
            "prefix_matches": sum(p["prefix_matches"] for p in paired),
            "format_rescued": sum(p["format_rescued"] for p in paired),
            "closed_sequences_unchanged": sum(
                not p["old_truncated"]
                and old[(p["problem_id"], p["candidate"])]["output_ids"]
                == new[(p["problem_id"], p["candidate"])]["output_ids"]
                for p in paired
            ),
            "checkpoint_unchanged": True,
            "parameters_unchanged": True,
            "optimizer_created": False,
        }
    write_json(root / "report/length-comparison.json", result)
    lines = [
        "# Diagnóstico de 1024 frente a 2048 tokens",
        "",
        "Seis relatos, uno por persona, elegidos para incluir todos los truncamientos previos del paso 144. Muestra enriquecida por fallos; no estima eficacia general. No se entrenó.",
        "",
        "| Paso | Límite | Formato válido | Truncamientos | Score integral |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for step, item in result["checkpoints"].items():
        for cap in ["1024", "2048"]:
            m = item[cap]
            lines.append(
                f"| {step} | {cap} | {m['n_valid']}/{m['n']} | {m['truncations']} | {m['reward_mean_all']:.4f} |"
            )
    for step, item in result["checkpoints"].items():
        lines += [
            "",
            f"Paso {step}: {item['prefix_matches']}/18 prefijos idénticos, {item['format_rescued']} formatos recuperados entre los truncados de 1024.",
            "",
        ]
    lines += [
        "",
        "| Paso | Caso | Muestra | Tokens 1024/2048 | Formato 1024/2048 | Mismo prefijo |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for step, item in result["checkpoints"].items():
        for p in item["pairs"]:
            lines.append(
                f"| {step} | {p['problem_id'][:12]} | {p['candidate'] + 1} | {p['old_tokens']} / {p['new_tokens']} | {p['old_valid']} / {p['new_valid']} | {p['prefix_matches']} |"
            )
    lines += [
        "",
        "Mismos checkpoints, prompts, semillas, temperatura y top-p. Se comprobaron pesos intactos y scores desde caché. Fuente DreamBank, CC BY-NC-SA 4.0, dreambank.net. Referencias y generaciones completas fuera de Git.",
        "",
    ]
    (root / "report/diagnostico.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    compare(parser.parse_args().results.resolve())
