"""Relectura exploratoria de materiales oníricos; no genera, entrena ni reelige modelos."""

import argparse
import html
import random
import secrets
from collections import Counter
from pathlib import Path

from experiments.preferences_v20 import locked_write
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import digest, read_json, read_jsonl, write_json

ROOT = Path("results/dream-v23")
SEED = "Dream about returning a borrowed library book, a locked door, and your brother waiting on the steps."
LITERAL = "I held the borrowed library book outside a locked door. My brother was waiting on the steps. I knocked, and we watched the handle turn without the door opening."
TRANSFORMED = "The library was my childhood kitchen. My brother sat on the steps, but I knew he was also my teacher. I tried to return the book. Its pages had become a little staircase, and the door remained locked. He held out his hand, and I couldn't tell whether to give him the book or climb it."
CONTROLS = [
    (
        "literal_scene",
        LITERAL,
        "I was returning a borrowed library book. My brother waited on the steps outside a locked door. When I knocked, its handle turned, but it stayed shut.",
    ),
    (
        "transformation_preserved",
        TRANSFORMED,
        "The library was also my childhood kitchen, and my brother was also my teacher. I tried to return a book whose pages were steps. The door stayed locked, and I didn't know whether to hand over the book or climb it.",
    ),
    (
        "quiet_fragment",
        "I sat on the library steps. The borrowed book rested on my knees. My brother waited beside me. The door was locked.",
        "I was sitting beside my brother with the borrowed book on my knees, outside the locked library door.",
    ),
    ("request_echo", SEED, SEED),
    (
        "unrelated_valid_dream",
        "I was walking across a red lake. Fish were asleep in the branches above my head. I couldn't find my shoes.",
        "I walked across a red lake beneath branches full of sleeping fish, looking for my shoes.",
    ),
    (
        "normalizing_account",
        TRANSFORMED,
        "I returned a library book with my brother. We waited for the librarian to unlock the door, then walked into an ordinary library and gave the book back.",
    ),
    (
        "replaced_account",
        LITERAL,
        "I was on a beach with a yellow kite. A gull took my hat, and I ran into the sea after it.",
    ),
    (
        "interpretive_comment",
        "Dreaming of a library book suggests learning, while a locked door can symbolize an obstacle. A brother waiting on the steps represents support and personal connection.",
        "The book represents learning, the door an obstacle, and the brother support.",
    ),
]


def prepare(root):
    if (root / "packet.json").exists():
        return
    old = Path("results/dream-v22")
    folder = old / "evaluation"
    packet = {r["review_id"]: r for r in read_json(folder / "quality-packet.json")}
    notes = read_json(folder / "quality-annotations.json")
    frozen = read_json(folder / "quality-frozen.json")
    assert digest(notes) == frozen["annotations_hash"]
    assert digest(list(packet.values())) == frozen["packet_hash"]
    keys = {r["review_id"]: r for r in read_json(folder / "quality-key.json")}
    generated = {
        p: {(r["problem_id"], r["candidate"]): r for r in read_jsonl(old / p / "samples.jsonl")}
        for p in ["before", "after"]
    }
    pools = {True: [], False: []}
    for n in notes["ratings"]:
        r = packet[n["review_id"]]
        if n["review_id"] in {"E05", "E86", "E87"}:
            continue  # ejemplos ya destacados en la orientación conceptual
        for side in "AB":
            pools[n["motifs"][side]["major"]].append((r["request"], n["review_id"], side))
    picked, seen = [], set()
    for flag, count in [(True, 12), (False, 6)]:
        candidates = sorted(pools[flag], key=lambda r: digest([20261034, flag, r]))
        chosen = []
        for request, rid, side in candidates:
            if request in seen:
                continue
            seen.add(request)
            k = keys[rid]
            raw = generated[k[side + "_phase"]][(k["problem_id"], k["candidate"])]
            assert raw["dream"] == packet[rid][side]
            item = {"request": request, "dream": raw["dream"], "answer": raw["answer"]}
            source = {
                "origin": "v22",
                "old_review_id": rid,
                "side": side,
                "phase": k[side + "_phase"],
                "legacy_literal_major": flag,
                "old_sample_hash": digest(raw),
            }
            chosen.append((item, source))
            if len(chosen) == count:
                break
        assert len(chosen) == count
        picked += chosen
    for name, dream, answer in CONTROLS:
        picked.append(
            (
                {"request": SEED, "dream": dream, "answer": answer},
                {"origin": "assistant_authored_control", "control": name},
            )
        )
    duplicate_indices = sorted(range(18), key=lambda i: digest([20261034, "duplicate", i]))[:4]
    for index in duplicate_indices:
        item, key = picked[index]
        picked.append((dict(item), {**key, "duplicate_of_content_hash": digest(item)}))
    random.Random(secrets.randbits(256)).shuffle(picked)
    out, key = [], []
    for i, (item, source) in enumerate(picked):
        rid = f"S{i + 1:02d}"
        out.append({"review_id": rid, **item})
        key.append({"review_id": rid, **source})
    p = {
        "version": "v23-concept-review",
        "real_cases": 18,
        "legacy_major": 12,
        "legacy_nonmajor": 6,
        "authored_controls": 8,
        "duplicate_presentations": 4,
        "selection": "Hash selection, one real case per request; exclude E05/E86/E87 highlighted previously. No model balancing or model ranking.",
        "purpose": "Pilot descriptive rubric separating recognizable material, transformations, literal deviation and account organization.",
        "rubric": "docs/rubrica-transformaciones-v23.md",
        "rubric_sha256": sha256(Path("docs/rubrica-transformaciones-v23.md")),
        "reviewer": "assistant",
        "independent_human_validation": False,
        "blinding_limit": "Sources, prior marks and control labels hidden in packet; same assistant saw v22 and authored controls, so recognition is possible. Not independent validation or measured interrater reliability.",
        "no_model_selection": True,
        "training_updates": 0,
        "new_model_generations": 0,
        "test_reports_used": 0,
        "legacy_frozen": frozen,
        "packet_hash": digest(out),
        "content_hashes": sorted(
            digest({k: v for k, v in r.items() if k != "review_id"}) for r in out
        ),
    }
    locked_write(root / "protocol.json", p)
    locked_write(root / "key.json", key)
    locked_write(root / "packet.json", out)
    locked_write(Path("docs/transformaciones-v23-protocolo.json"), p)
    print(
        "Prepared: 18 real cases + 8 authored controls + 4 repeated presentations. Source key hidden."
    )


def freeze(root):
    p = read_json(root / "protocol.json")
    rows = read_json(root / "packet.json")
    notes = read_json(root / "annotations.json")
    assert digest(rows) == p["packet_hash"]
    assert sha256(Path(p["rubric"])) == p["rubric_sha256"]
    by_id = {r["review_id"]: r for r in rows}
    assert len(notes["ratings"]) == len({r["review_id"] for r in notes["ratings"]}) == 30
    assert {r["review_id"] for r in notes["ratings"]} == set(by_id)
    for n in notes["ratings"]:
        row = by_id[n["review_id"]]
        assert n["case_hash"] == digest(row)
        assert n["experience"] in {"escena", "fragmento", "enunciado", "comentario", "no_se"}
        assert n["material"] in {"reconocible", "parcial", "ausente", "no_se"}
        assert n["treatment"] in {"literal", "transformado", "mixto", "no_se"}
        assert n["literal_deviation"] in {True, False, "no_se"}
        assert n["account"] in {
            "organiza_preserva",
            "resume_con_omisiones",
            "normaliza_borra",
            "sustituye",
            "no_se",
        }
        assert n["note"] and n["evidence"]
        for quote in n["evidence"]:
            assert quote and (quote in row["dream"] or quote in row["answer"])
    locked_write(
        root / "review-frozen.json",
        {
            "packet_hash": digest(rows),
            "annotations_hash": digest(notes),
            "source_key_opened_before_freeze": False,
            "independent_human_validation": False,
            "reviewed_presentations": 30,
        },
    )


def report(root):
    p = read_json(root / "protocol.json")
    packet = read_json(root / "packet.json")
    notes = read_json(root / "annotations.json")
    frozen = read_json(root / "review-frozen.json")
    assert digest(packet) == frozen["packet_hash"] and digest(notes) == frozen["annotations_hash"]
    legacy_folder = Path("results/dream-v22/evaluation")
    assert read_json(legacy_folder / "quality-frozen.json") == p["legacy_frozen"]
    assert (
        digest(read_json(legacy_folder / "quality-annotations.json"))
        == p["legacy_frozen"]["annotations_hash"]
    )
    assert (
        digest(read_json(legacy_folder / "quality-packet.json"))
        == p["legacy_frozen"]["packet_hash"]
    )
    old_root = legacy_folder.parent
    assert (
        digest(read_json(old_root / "report/summary.json"))
        == read_json(old_root / "report/audit.json")["result_hash"]
    )
    old_spec = read_json(old_root / "protocol.json")
    for phase, checkpoint in old_spec["checkpoints"].items():
        assert (
            sha256(Path(checkpoint) / "adapters.safetensors")
            == old_spec["checkpoint_hashes"][phase]
        )
    keys = {r["review_id"]: r for r in read_json(root / "key.json")}
    real = [
        r
        for r in notes["ratings"]
        if keys[r["review_id"]]["origin"] == "v22"
        and "duplicate_of_content_hash" not in keys[r["review_id"]]
    ]
    major = [r for r in real if keys[r["review_id"]]["legacy_literal_major"]]
    by_id = {r["review_id"]: r for r in packet}
    duplicate_checks = []
    fields = ["experience", "material", "treatment", "literal_deviation", "account"]
    for n in notes["ratings"]:
        k = keys[n["review_id"]]
        if "duplicate_of_content_hash" not in k:
            continue
        h = k["duplicate_of_content_hash"]
        matches = [
            a
            for a in notes["ratings"]
            if a["review_id"] != n["review_id"]
            and digest({s: v for s, v in by_id[a["review_id"]].items() if s != "review_id"}) == h
        ]
        assert len(matches) == 1
        duplicate_checks.append(
            {
                "review_id": n["review_id"],
                "same_labels": all(n[f] == matches[0][f] for f in fields),
                "different_fields": [f for f in fields if n[f] != matches[0][f]],
            }
        )
    summary = {
        "real_cases": len(real),
        "old_literal_major_cases": len(major),
        "real_material": dict(Counter(r["material"] for r in real)),
        "real_treatment": dict(Counter(r["treatment"] for r in real)),
        "old_major_material": dict(Counter(r["material"] for r in major)),
        "old_major_treatment": dict(Counter(r["treatment"] for r in major)),
        "real_account": dict(Counter(r["account"] for r in real)),
        "duplicate_checks": duplicate_checks,
        "controls": [
            {"control": keys[r["review_id"]]["control"], **{f: r[f] for f in fields}}
            for r in notes["ratings"]
            if keys[r["review_id"]]["origin"] == "assistant_authored_control"
        ],
        "training_updates": 0,
        "new_model_generations": 0,
        "test_reports_used": 0,
        "checkpoint_decision_changed": False,
        "independent_human_validation": False,
        "status": "descriptive pilot completed; rubric provisional, no independent reliability estimate",
        "limitations": [
            p["blinding_limit"],
            "Biased selection of prior literal flags, not prevalence or a fresh model comparison.",
            "Duplicate recognition may inflate self-consistency; authored controls are illustrations, not independent gold labels.",
        ],
    }
    write_json(root / "summary.json", summary)
    write_json(Path("docs/transformaciones-v23-resultados.json"), summary)
    write_json(
        Path("docs/transformaciones-v23-auditoria.json"),
        {
            "protocol_hash": digest(p),
            **frozen,
            "result_hash": digest(summary),
            "training_updates": 0,
            "legacy_scores_unchanged": True,
            "both_checkpoint_adapters_unchanged": True,
        },
    )
    e = html.escape
    body = [
        "<h1>Dream · dejar que los materiales se transformen</h1>",
        "<p><strong>Revisión conceptual completada, sin entrenar ni elegir otro modelo.</strong> La rúbrica distingue experiencia, materiales, transformación y relato posterior. Es provisional y fue revisada por el mismo asistente que escribió los controles y ya vio los textos.</p>",
        f"<p>18 casos reales seleccionados, ocho controles construidos y cuatro repeticiones. Entre los doce casos antes marcados como desvío literal, {summary['old_major_material'].get('reconocible', 0)} conservan material reconocible y {summary['old_major_material'].get('parcial', 0)} lo conservan parcialmente. Eso no los vuelve automáticamente buenos sueños: también hay comentario sin experiencia. No es prevalencia ni una comparación nueva de modelos.</p>",
    ]
    labels = {
        "experience": "Experiencia",
        "material": "Material de la semilla",
        "treatment": "Tratamiento",
        "literal_deviation": "Desvío literal",
        "account": "Answer respecto de dream",
    }
    display = {
        "escena": "Escena",
        "fragmento": "Fragmento válido",
        "enunciado": "Enunciado o lista",
        "comentario": "Comentario interpretativo",
        "no_se": "No sé",
        "reconocible": "Reconocible",
        "parcial": "Parcialmente reconocible",
        "ausente": "Sin vínculo textual defendible",
        "literal": "Continuidad literal",
        "transformado": "Transformado",
        "mixto": "Continuidad y transformación",
        "organiza_preserva": "Cuenta u organiza sin borrar relaciones importantes",
        "resume_con_omisiones": "Resume con omisiones o cambios señalados en la nota",
        "normaliza_borra": "Normaliza y borra transformaciones",
        "sustituye": "Cuenta otro sueño",
    }
    for n in notes["ratings"]:
        rid = n["review_id"]
        row, k = by_id[rid], keys[rid]
        origin = (
            "Control construido: " + k["control"] if k["origin"] != "v22" else "Caso real previo"
        )
        if "duplicate_of_content_hash" in k:
            origin += " · presentación repetida"
        body += [
            f'<section id="{rid}"><h2>{rid} · {e(origin)}</h2><p>{e(row["request"])}</p>',
            f"<p>{e(n['note'])}</p><ul>",
        ]
        for f, title in labels.items():
            value = n[f]
            text = (
                (
                    "Sí; no significa baja calidad"
                    if value is True
                    else "No"
                    if value is False
                    else "No sé"
                )
                if f == "literal_deviation"
                else display[value]
            )
            body += [f"<li>{title}: <b>{e(text)}</b></li>"]
        body += [
            "</ul><p>Evidencia textual: "
            + " · ".join("«" + e(q) + "»" for q in n["evidence"])
            + "</p>"
        ]
        body += [
            "<details><summary>Leer dream y answer completos</summary><h3>dream</h3><pre>"
            + e(row["dream"])
            + "</pre><h3>answer</h3><pre>"
            + e(row["answer"])
            + "</pre></details></section>"
        ]
    body += [
        "<footer>Sin cambios en etiquetas o decisiones de v21/v22. Los 200 relatos de prueba no se usaron. Textos completos locales; DreamBank · CC BY-NC-SA 4.0 · dreambank.net. Controles construidos por el asistente.</footer>"
    ]
    style = "body{max-width:1000px;margin:40px auto;padding:0 20px;background:#f6f4ed;color:#213331;font:18px/1.6 system-ui}section{border-top:1px solid #bcc8be;padding:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 system-ui}details{padding:12px;background:#eaf0e6}summary{cursor:pointer}"
    (root / "index.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · materiales oníricos</title><style>'
        + style
        + "</style>"
        + "".join(body)
        + "</html>"
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "report"])
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "report": report}[args.action](ROOT)
