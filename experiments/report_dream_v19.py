"""Auditoría de calidad exploratoria e intervenciones, sin mezclar ambos objetivos."""

import argparse
import html
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from think_dream_lab.judge import protected_json
from think_dream_lab.storage import digest, read_json, read_jsonl, write_json


def validate_frozen(root, packet_name, annotations_name, frozen_name):
    packet = read_json(root / packet_name)
    notes = read_json(root / annotations_name)
    frozen = read_json(root / frozen_name)
    if digest(packet) != frozen["packet_hash"] or digest(notes) != frozen["annotations_hash"]:
        raise ValueError("Se modificó una revisión congelada.")
    return packet, notes, frozen


def expected_state(probe, condition):
    mapping = {
        "location": ("under_chair", "inside_cupboard"),
        "actor": ("sister", "narrator"),
        "outcome": ("sank_before_shore", "reached_without_sinking"),
    }
    return (
        "absent" if condition in {"original", "direct"} else mapping[probe][condition == "edited"]
    )


def report(root):
    from experiments.dream_quality_v19 import RULE
    from think_dream_lab.holdout_evaluation import sha256

    packet, notes, frozen = validate_frozen(
        root, "quality-packet.json", "quality-annotations.json", "quality-frozen.json"
    )
    if frozen["rubric_hash"] != digest(RULE):
        raise ValueError("Cambió la rúbrica.")
    key = {r["review_id"]: r for r in read_json(root / "quality-key.json")}
    manual = {r["review_id"]: r for r in notes["ratings"]}
    judgments = read_json(root / "quality-judge/ratings.json")["ratings"]
    validation = read_json(root / "quality-judge/validation.json")
    if not validation["complete"] or not validation["parameters_unchanged"]:
        raise ValueError("Juez incompleto.")
    spec = read_json(root / "quality-judge/protocol.json")["judge"]
    if len(judgments) != 24 or len({(r["review_id"], r["swapped"]) for r in judgments}) != 24:
        raise ValueError("Evaluaciones incompletas o duplicadas.")
    cases = []
    controls = []
    modelprefs = []
    raw_choice_pairs = defaultdict(list)
    for row in packet:
        rid = row["review_id"]
        provenance = key[rid]
        reviewed = manual[rid]
        if reviewed["case_hash"] != digest(row):
            raise ValueError("Cambió un texto revisado.")
        preference = reviewed["choice"]
        if provenance["kind"] == "models":
            modelprefs.append(
                provenance[preference + "_source"] if preference in {"A", "B"} else preference
            )
        for r in [j for j in judgments if j["review_id"] == rid]:
            A, B = (row["B"], row["A"]) if r["swapped"] else (row["A"], row["B"])
            payload = {"motifs": row["motifs"], "A": A, "B": B}
            if digest(payload) != r["payload_hash"]:
                raise ValueError("Cambió la entrada al juez.")
            messages = [
                {"role": "system", "content": RULE},
                {"role": "user", "content": protected_json(payload)},
            ]
            cachekey = digest([spec, "dream-quality-pair-v19", messages])
            cache = read_json(root / "quality-judge/dream/judge-cache" / f"{cachekey}.json")
            raw = cache["value"].get("choice")
            if r["swapped"] and raw in {"A", "B"}:
                raw = {"A": "B", "B": "A"}[raw]
            if raw in {"A", "B", "tie"}:
                raw_choice_pairs[rid].append(raw)
            if provenance["expected"] is not None:
                desired = (
                    "tie"
                    if provenance["expected"] == "tie"
                    else "A"
                    if provenance["A_source"] == "human"
                    else "B"
                )
                controls.append(
                    {
                        "review_id": rid,
                        "swapped": r["swapped"],
                        "status": r["status"],
                        "choice": r["choice"],
                        "expected": desired,
                        "correct": r["status"] == "complete" and r["choice"] == desired,
                    }
                )
        cases.append(
            {
                "review_id": rid,
                "kind": provenance["kind"],
                "A_source": provenance["A_source"],
                "B_source": provenance["B_source"],
                "assistant_choice": preference,
                "confidence": reviewed["confidence"],
                "assistant_A": {
                    k: reviewed["A"][k]
                    for k in ["mode", "scene_presence", "report_likeness", "motif_grounding"]
                },
                "assistant_B": {
                    k: reviewed["B"][k]
                    for k in ["mode", "scene_presence", "report_likeness", "motif_grounding"]
                },
                "judge": [
                    {"swapped": r["swapped"], "status": r["status"], "choice": r["choice"]}
                    for r in judgments
                    if r["review_id"] == rid
                ],
            }
        )
    informative = [v for v in raw_choice_pairs.values() if len(v) == 2]
    quality = {
        "model_pairs": 6,
        "model_texts": 12,
        "assistant_preferences": dict(Counter(modelprefs)),
        "assistant_model_modes": dict(
            Counter(
                reviewed[side]["mode"]
                for rid, reviewed in manual.items()
                if key[rid]["kind"] == "models"
                for side in ["A", "B"]
            )
        ),
        "judge_valid": validation["valid_evaluations"],
        "judge_total": 24,
        "coarse_control_correct": sum(r["correct"] for r in controls),
        "coarse_control_total": len(controls),
        "coarse_gate_passed": sum(r["correct"] for r in controls) >= 9
        and all(r["correct"] for r in controls if key[r["review_id"]]["kind"] == "identity"),
        "parsed_choice_only_pairs": len(informative),
        "parsed_choice_only_side_changes": sum(v[0] != v[1] for v in informative),
        "parsed_choice_only_caution": "Post-hoc diagnostic uses parsed choices even if ancillary mode fields fail schema; does not repair or increase valid full evaluations. Excludes no_se and invalid choices.",
        "controls": controls,
        "cases": cases,
        "rubric_hash": digest(RULE),
        "annotations_hash": frozen["annotations_hash"],
        "independent_human_review": False,
        "controls_authored_by_reviewer": True,
        "reward_validated": False,
    }
    folder = root / "interventions"
    probe_packet, probe_notes, probe_frozen = validate_frozen(
        folder, "review-packet.json", "review-annotations.json", "review-frozen.json"
    )
    plan = read_json(folder / "plan.json")
    run = read_json(folder / "run.json")
    if (
        not run["parameters_unchanged"]
        or run["optimizer_created"]
        or run["training_updates"]
        or run["status"] != "complete"
    ):
        raise ValueError("Generación no congelada.")
    if sha256(Path(plan["checkpoint"]) / "adapters.safetensors") != run["checkpoint_sha256"]:
        raise ValueError("Cambió el checkpoint.")
    answers = read_jsonl(folder / "answers.jsonl")
    bytask = {r["task_id"]: r for r in answers}
    tasks = {r["task_id"]: r for r in plan["tasks"]}
    if (
        len(answers) != 36
        or set(bytask) != set(tasks)
        or digest(plan["tasks"]) != plan["tasks_hash"]
    ):
        raise ValueError("Trayectorias incompletas.")
    for tid, row in bytask.items():
        if any(
            row[k] != tasks[tid][k] for k in ["prompt", "seed", "condition", "probe", "case_id"]
        ):
            raise ValueError("Se modificó una entrada de generación.")
    source_records = {
        (r["problem_id"], r["candidate"]): r
        for r in read_jsonl(
            Path("results/dream-v17/before/dream/holdout/checkpoint/responses.jsonl")
        )
    }
    selected = read_json(root / "selection.json")[:3]
    references = read_json(Path("results/dream-v17/before/dream/holdout/evaluation-pack.json"))[
        "rows"
    ]
    if any(r["reference"] in task["prompt"] for r in references for task in tasks.values()):
        raise ValueError("Una referencia completa está en la entrada del generador.")
    for i, item in enumerate(selected):
        r = source_records[(item["problem_id"], item["candidate"])]
        close = r["text"].index("</dream>")
        expected = r["prompt"] + r["text"][:close] + "</dream>\n<answer>\n"
        assert all(
            t["prompt"] == expected
            for t in tasks.values()
            if t["case_id"] == f"I{i + 1}" and t["condition"] == "original"
        )
    probe_key = {r["review_id"]: r for r in read_json(folder / "review-key.json")}
    packets = {r["review_id"]: r for r in probe_packet}
    evaluations = []
    for label in probe_notes["labels"]:
        k = probe_key[label["review_id"]]
        row = bytask[k["task_id"]]
        if (
            digest(row["answer"]) != label["answer_hash"]
            or digest(packets[label["review_id"]]["answer"]) != label["answer_hash"]
        ):
            raise ValueError("Cambió una respuesta revisada.")
        expected = expected_state(row["probe"]["kind"], row["condition"])
        evaluations.append(
            {
                "task_id": row["task_id"],
                "case_id": row["case_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "kind": row["probe"]["kind"],
                "state": label["state"],
                "expected": expected,
                "match": label["state"] == expected,
                "format_valid": row["format_valid"],
                "truncated": row["truncated"],
                "answer_words": len(row["answer"].split()),
                "prompt_tokens": len(row["prompt_ids"]),
                "new_tokens": len(row["output_ids"]),
                "seconds": row["seconds"],
            }
        )
    conditions = {}
    for condition in plan["conditions"]:
        rows = [r for r in evaluations if r["condition"] == condition]
        conditions[condition] = {
            "n": len(rows),
            "exact_probe_match": sum(r["match"] for r in rows),
            "probe_mentioned": sum(r["state"] != "absent" for r in rows),
            "format_valid": sum(r["format_valid"] for r in rows),
            "truncations": sum(r["truncated"] for r in rows),
            "answer_words_mean": statistics.mean(r["answer_words"] for r in rows),
            "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in rows),
            "new_tokens_mean": statistics.mean(r["new_tokens"] for r in rows),
        }
    bypair = {(r["case_id"], r["seed"], r["condition"]): r for r in evaluations}
    paired = sum(
        bypair[(case, seed, "anchor")]["match"] and bypair[(case, seed, "edited")]["match"]
        for case in ["I1", "I2", "I3"]
        for seed in plan["seeds"]
    )
    result = {
        "quality": quality,
        "interventions": {
            "checkpoint": 32,
            "cases": 3,
            "answers": 36,
            "conditions": conditions,
            "paired_exact_edit_following": paired,
            "paired_edit_total": 6,
            "format_valid": sum(r["format_valid"] for r in evaluations),
            "new_tokens": sum(r["new_tokens"] for r in evaluations),
            "new_generation_seconds": sum(r["seconds"] for r in evaluations),
            "rows": evaluations,
            "parameters_unchanged": True,
            "optimizer_created": False,
            "training_updates": 0,
            "fresh_cache_per_generation": True,
            "original_dream_text_preserved": True,
            "prefixes_reencoded": True,
            "annotations_hash": probe_frozen["annotations_hash"],
            "tasks_hash": plan["tasks_hash"],
            "caution": "Artificial appended probes and small sample: conditional sensitivity is not a demonstration of natural full retelling fidelity, quality benefit, diversity benefit or subjective experience. Two paraphrases retell the boat alone. Direct generation changes instruction and input length, and is not cost-matched.",
        },
        "training_updates": 0,
        "test_reports_used": 0,
        "reference_in_generator_prompt": False,
        "new_reward_selected": False,
        "human_independent_validation": False,
    }
    write_json(root / "report/summary.json", result)
    write_view(root, result, packet, notes, key, plan, evaluations, bytask)
    source_lock = read_json(folder / "source-lock.json")
    script_hash = source_lock["script_sha256"]
    if (
        sha256(Path(source_lock["script_path"])) != script_hash
        or sha256(Path("experiments/dream_interventions_v19.py")) != script_hash
    ):
        raise ValueError("Cambió el script usado en las intervenciones.")
    core = source_lock["core_snapshot"]
    source_manifest = read_json(Path(core["directory"]) / "source-manifest.json")
    for relative, expected_hash in source_manifest["files"].items():
        if (
            sha256(Path(relative)) != expected_hash
            or sha256(Path(core["directory"]) / relative) != expected_hash
        ):
            raise ValueError("El código actual o su copia no coincide con la corrida.")
    write_json(
        root / "report/audit.json",
        {
            "passed": True,
            "quality_review_frozen_before_model_key": True,
            "probe_labels_frozen_before_condition_key": True,
            "independent_human_review": False,
            "quality_judge_parameters_unchanged": True,
            "generator_parameters_unchanged": True,
            "adapter_loaded_exactly": True,
            "answers": 36,
            "fresh_contexts": 36,
            "original_dream_text_checked": True,
            "no_reference_in_generator_prompt": True,
            "optimizer_created": False,
            "training_updates": 0,
            "test_reports_used": 0,
            "intervention_script_sha256": script_hash,
            "core_matches_saved_snapshot": True,
        },
    )
    print(
        "Quality:",
        {
            k: quality[k]
            for k in [
                "assistant_preferences",
                "judge_valid",
                "coarse_control_correct",
                "coarse_control_total",
                "coarse_gate_passed",
                "parsed_choice_only_pairs",
                "parsed_choice_only_side_changes",
            ]
        },
    )
    print(
        "Probe counts:",
        {k: (v["exact_probe_match"], v["n"]) for k, v in conditions.items()},
        "paired edits:",
        paired,
        "/6",
    )
    return result


def write_view(root, result, packet, notes, quality_key, plan, evaluations, answers):
    escape = html.escape
    annotations = {r["review_id"]: r for r in notes["ratings"]}
    tasks = {r["task_id"]: r for r in plan["tasks"]}
    body = [
        "<h1>Dream · escenas y dependencia textual</h1>",
        "<p>Revisión del asistente y diagnóstico sin entrenamiento. Los 200 relatos de prueba siguen sin usarse.</p>",
        "<h2>Qué encontramos</h2><p>Los doce bloques de modelo revisados contienen escenas; uno mezcla escena y resumen de motivos. Preferí el 32 en tres pares, el 160 en uno y empaté dos. La muestra es pequeña y mi revisión no es humana independiente.</p>",
        "<p>El contenido de dream influye en answer: detalle agregado completo 5/6, detalle cambiado 5/6, paráfrasis 6/6 y detalle de otro sueño 4/6. Ambos lados de una edición se siguen completamente en 4/6 pares. No prueba que dos etapas mejoren calidad, diversidad o costo frente a generación directa.</p>",
        "<p>El juez de calidad no pasó la calibración: 12/24 salidas con esquema válido y 3/10 presentaciones de controles acertadas. Incluso leyendo sólo elecciones válidas, sin exigir los campos de modo, cambió de preferencia en 4/8 pares al invertir lados. No se eligió una nueva recompensa.</p>",
        '<nav><a href="#quality">Revisión de dream</a> · <a href="#probes">Intervenciones</a></nav><h2 id="quality">Textos revisados</h2>',
    ]
    labels = {
        "checkpoint32": "Checkpoint 32",
        "checkpoint160": "Checkpoint 160",
        "human": "Relato humano",
        "list": "Lista construida",
        "plan": "Instrucciones construidas",
        "analysis": "Análisis construido",
        "repetition": "Repetición construida",
        "identity": "Texto idéntico",
        "valid_surreal": "Escena ficticia construida",
    }
    for row in packet:
        rid = row["review_id"]
        review = annotations[rid]
        key = quality_key[rid]
        body += [
            f"<section><h3>{rid}</h3><p>{escape(review['note'])}</p>",
            f'<p>Preferencia del asistente: {escape(review["choice"])}. Confianza: {escape(review["confidence"])}.</p><div class="columns">',
        ]
        for side in ["A", "B"]:
            body += [
                f"<article><h4>{side} · {labels[key[side + '_source']]}</h4><p>Modo: {escape(review[side]['mode'])}.</p><details><summary>Leer bloque</summary><p>{escape(row[side])}</p></details></article>"
            ]
        body += ["</div></section>"]
    body += [
        '<h2 id="probes">Intervenciones sobre el checkpoint 32</h2><p>Semillas 17 y 29, contexto nuevo en cada generación. Los detalles de prueba fueron agregados artificialmente. En las condiciones original y directa se espera ausencia del detalle; su ausencia no puntúa calidad.</p>'
    ]
    names = {
        "original": "Sueño original",
        "anchor": "Detalle agregado",
        "edited": "Detalle cambiado",
        "paraphrase": "Detalle parafraseado",
        "swapped": "Otro sueño",
        "direct": "Generación directa",
    }
    states = {
        "absent": "No lo menciona",
        "under_chair": "Llave debajo de la silla",
        "inside_cupboard": "Llave dentro del armario",
        "sister": "La hermana esconde la moneda",
        "narrator": "El narrador esconde la moneda",
        "sank_before_shore": "El barco se hunde antes de llegar",
        "reached_without_sinking": "El barco llega sin hundirse",
        "nonarrival_unspecified": "No llega, sin afirmar que se hunda",
    }
    for case in ["I1", "I2", "I3"]:
        body += [f"<h3>{case}</h3>"]
        for r in sorted(
            [r for r in evaluations if r["case_id"] == case],
            key=lambda r: (plan["conditions"].index(r["condition"]), r["seed"]),
        ):
            tid = r["task_id"]
            task = tasks[tid]
            body += [
                f"<section><h4>{names[r['condition']]} · semilla {r['seed']}</h4>",
                f"<p>Nota congelada: <b>{states[r['state']]}</b>. Formato: {'válido' if r['format_valid'] else 'inválido'}.</p>",
                f"<details><summary>Dream de entrada</summary><p>{escape(task['dream'] or 'Sin bloque dream; instrucción de generación directa.')}</p></details>",
                f"<details><summary>Answer generado</summary><p>{escape(answers[tid]['answer'])}</p></details></section>",
            ]
    style = "body{max-width:1100px;margin:36px auto;padding:0 24px;background:#faf9f5;color:#202622;font:17px/1.6 system-ui}h1{line-height:1.15}section{border-top:1px solid #ccd1cb;margin-top:24px;padding-top:10px}p{white-space:pre-wrap}summary,a{color:#325844;cursor:pointer}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{padding:18px;background:white;border:1px solid #e2e5dd;border-radius:10px;min-width:0}@media(max-width:800px){.columns{grid-template-columns:1fr}}"
    (root / "index.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · diagnóstico v19</title><style>'
        + style
        + "</style>"
        + "".join(body)
        + "<footer>DreamBank · CC BY-NC-SA 4.0 · dreambank.net. Textos completos y notas sólo en el artefacto local.</footer></html>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/dream-v19"))
    args = parser.parse_args()
    report(args.root)
