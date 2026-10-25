"""Audita las notas congeladas y recién entonces revela la procedencia de doce pares."""

import argparse
import html
import statistics
from collections import Counter
from pathlib import Path

from think_dream_lab.storage import digest, read_json, write_json

FIELDS = ["characters", "events", "relations", "overall"]


def summarize(source, target):
    packet = read_json(target / "packet.json")
    notes = read_json(target / "annotations.json")
    frozen = read_json(target / "ratings-frozen.json")
    manifest = read_json(target / "manifest.json")
    if (
        digest(packet) != frozen["packet_hash"]
        or digest(notes) != frozen["annotations_hash"]
        or manifest["packet_hash"] != frozen["packet_hash"]
        or frozen["provenance_consulted_before_freeze"]
        or notes["independent_human_review"]
        or len(notes["ratings"]) != 12
        or {r["review_id"] for r in notes["ratings"]} != {r["review_id"] for r in packet}
    ):
        raise ValueError("Se requieren doce revisiones congeladas sobre el paquete original.")
    for row in notes["ratings"]:
        original = next(p for p in packet if p["review_id"] == row["review_id"])
        if row["packet_case_hash"] != digest(original):
            raise ValueError("Una revisión corresponde a otro texto.")
        if row["preference"] not in {"A", "B", "tie", "no_se"}:
            raise ValueError("Preferencia desconocida.")
        if row["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("Confianza desconocida.")
        for side in ["A", "B"]:
            if any(type(row[side][f]) is not int or not 0 <= row[side][f] <= 4 for f in FIELDS):
                raise ValueError("Las notas deben ser enteros de 0 a 4.")
    comparison = read_json(source / "dream/report/comparison.json")
    if digest(comparison) != manifest["source_comparison_hash"]:
        raise ValueError("Cambió la comparación original.")
    # No se abre la clave hasta comprobar todas las anotaciones congeladas.
    provenance = {r["review_id"]: r for r in read_json(target / "provenance-key.json")}
    mapping = {
        (r["phase"], r["problem_id"], r["candidate"]): r["blind_id"]
        for r in read_json(source / "dream/judge-provenance-key.json")["mapping"]
    }
    judges = {r["blind_id"]: r for r in read_json(source / "dream/judge-ratings.json")["ratings"]}
    cases = []
    for row in notes["ratings"]:
        key = provenance[row["review_id"]]
        preference = row["preference"]
        item = {
            "review_id": row["review_id"],
            "selection": key["selection"],
            "minilm_delta": key["delta"],
            "assistant_preference": preference
            if preference in {"tie", "no_se"}
            else key[preference + "_phase"],
            "confidence": row["confidence"],
        }
        for side in ["A", "B"]:
            phase = key[side + "_phase"]
            item["assistant_" + phase] = {f: row[side][f] for f in FIELDS}
            judge = judges[mapping[(phase, key["problem_id"], key["candidate"])]]
            item["phi4_" + phase] = {f: judge[f] for f in FIELDS}
            if judge["overall"] != key["judge_" + phase]:
                raise ValueError("Cambió la nota original del juez.")
        for reviewer in ["assistant", "phi4"]:
            item[reviewer + "_delta"] = (
                item[reviewer + "_after"]["overall"] - item[reviewer + "_before"]["overall"]
            )
        cases.append(item)

    def group_summary(rows):
        return {
            "pairs": len(rows),
            "assistant_global_direction": {
                "higher": sum(r["assistant_delta"] > 0 for r in rows),
                "equal": sum(r["assistant_delta"] == 0 for r in rows),
                "lower": sum(r["assistant_delta"] < 0 for r in rows),
            },
            "assistant_preferences": dict(Counter(r["assistant_preference"] for r in rows)),
            "assistant_delta_mean": statistics.mean(r["assistant_delta"] for r in rows),
            "phi4_delta_mean": statistics.mean(r["phi4_delta"] for r in rows),
        }

    dimensions = {}
    for field in FIELDS:
        errors = [
            r["phi4_" + phase][field] - r["assistant_" + phase][field]
            for r in cases
            for phase in ["before", "after"]
        ]
        dimensions[field] = {
            "exact_agreement": sum(e == 0 for e in errors),
            "accounts": len(errors),
            "mean_absolute_difference": statistics.mean(abs(e) for e in errors),
            "phi4_minus_assistant_mean": statistics.mean(errors),
            "difference_at_least_two_points": sum(abs(e) >= 2 for e in errors),
        }
    result = {
        "reviewer": "assistant",
        "independent_human_review": False,
        "provenance_hidden_until_annotations_frozen": True,
        "reviewer_knew_aggregate_results_and_selection": True,
        "not_representative_sample": True,
        "unique_reports": manifest["unique_reports"],
        "author_groups": 6,
        "all": group_summary(cases),
        "discordant": group_summary([r for r in cases if r["selection"] == "discordant"]),
        "contrast": group_summary([r for r in cases if r["selection"] == "contrast"]),
        "dimensions": dimensions,
        "cases": cases,
        "packet_hash": frozen["packet_hash"],
        "annotations_hash": frozen["annotations_hash"],
        "rubric_hash": frozen["rubric_hash"],
        "training_updates": 0,
        "test_reports_used": 0,
        "caution": "Diagnóstico seleccionado por discrepancias. Las notas del asistente no son verdad de referencia ni validación humana independiente. No estima calidad general ni precisión del juez en todo el corpus.",
    }
    write_json(target / "summary.json", result)
    write_html(target, packet, notes, provenance, cases)
    print("Revisión auditada:", result["all"], "Discrepancias:", result["discordant"])
    return result


def write_html(target, packet, notes, provenance, cases):
    byid = {r["review_id"]: r for r in notes["ratings"]}
    scores = {r["review_id"]: r for r in cases}
    blocks = []
    for original in packet:
        rid = original["review_id"]
        note, key, score = byid[rid], provenance[rid], scores[rid]
        preferred = {
            "before": "checkpoint 32",
            "after": "checkpoint 160",
            "tie": "empate",
            "no_se": "sin decisión",
        }[score["assistant_preference"]]
        blocks += [
            f'<section id="{rid}"><h2>{rid}</h2><p>Preferencia del asistente: <b>{preferred}</b>. Confianza: {note["confidence"]}. MiniLM: {score["minilm_delta"]:+.4f}.</p>',
            f"<p>{html.escape(note['note'])}</p>",
            "<details><summary>Referencia humana y hechos centrales</summary>",
            f"<p>{html.escape(original['reference'])}</p><ul>",
        ]
        blocks += [f"<li>{html.escape(f)}</li>" for f in note["reference_core"]]
        blocks += ['</ul></details><div class="columns">']
        for side in ["A", "B"]:
            phase = key[side + "_phase"]
            step = 32 if phase == "before" else 160
            blocks += [
                f"<article><h3>{side} · checkpoint {step}</h3><p>Global 0–4: asistente <b>{note[side]['overall']}</b> · Phi-4 <b>{score['phi4_' + phase]['overall']}</b>.</p>",
                f"<details><summary>Leer respuesta</summary><p>{html.escape(original[side])}</p></details><h4>Observaciones del asistente</h4><ul>",
            ]
            blocks += [f"<li>{html.escape(e)}</li>" for e in note[side]["errors"]]
            blocks += ["</ul></article>"]
        blocks += ["</div></section>"]
    (target / "resultados.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · revisión de doce pares</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;background:#faf9f5;color:#202622;font:17px/1.6 system-ui}section{border-top:1px solid #ccd1cb;margin-top:36px}p{white-space:pre-wrap}summary{cursor:pointer;color:#325844}nav a{display:inline-block;margin-right:16px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}article{background:white;padding:20px;border:1px solid #e2e5dd;border-radius:10px;min-width:0}li{margin-bottom:8px}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style><h1>Dream · revisión de doce pares</h1><p>Revisión del asistente con procedencia oculta hasta congelar las notas, no validación humana independiente. Conocía resultados agregados y selección; podía inferir procedencia por el estilo. Muestra diagnóstica: 12 pares, 11 referencias y seis grupos.</p><p>En los ocho pares donde MiniLM subió y Phi-4 bajó, di seis empates globales y dos descensos; las preferencias finas se repartieron 4–4. Ningún score quedó validado como recompensa de hechos y relaciones. No hubo entrenamiento ni se usaron los 200 relatos de prueba.</p><nav>'
        + "".join(f'<a href="#{p["review_id"]}">{p["review_id"]}</a>' for p in packet)
        + "</nav>"
        + "".join(blocks)
        + "<footer>DreamBank · CC BY-NC-SA 4.0 · dreambank.net. Textos completos sólo en el artefacto local.</footer></html>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results/dream-v17"))
    parser.add_argument("--target", type=Path, default=Path("results/dream-v18/review"))
    args = parser.parse_args()
    summarize(args.source, args.target)
