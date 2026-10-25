"""Preparación reproducible de doce pares con procedencia oculta para revisión asistida."""

import argparse
import html
from collections import Counter
from pathlib import Path

from think_dream_lab.dream_task import extract_blocks
from think_dream_lab.reference_judge import RULE
from think_dream_lab.storage import digest, read_json, read_jsonl, write_json

SEED = 20261026


def prepare(source, target):
    comparison = read_json(source / "dream/report/comparison.json")
    common = comparison["common_valid_pairs"]
    selected = [p for p in common if p["delta"] > 0 and p["judge_delta"] < 0]
    if len(selected) != 8:
        raise ValueError("Se esperaba la selección diagnóstica fijada de ocho discrepancias.")
    counts = Counter(p["author_group"] for p in selected)
    for group in sorted({p["author_group"] for p in common}):
        used = {p["problem_id"] for p in selected if p["author_group"] == group}
        pool = sorted(
            [p for p in common if p["author_group"] == group and p not in selected],
            key=lambda p: digest([SEED, "control", p["problem_id"], p["candidate"]]),
        )
        for p in pool:
            if counts[group] == 2:
                break
            if p["problem_id"] not in used:
                selected.append(p)
                used.add(p["problem_id"])
                counts[group] += 1
    if len(selected) != 12 or set(counts.values()) != {2}:
        raise ValueError("No se completaron doce pares, dos por grupo.")
    selected.sort(key=lambda p: digest([SEED, "order", p["problem_id"], p["candidate"]]))
    # Orden y lados se fijan antes de leer los relatos; cada fase ocupa A seis veces.
    side_order = sorted(range(12), key=lambda i: digest([SEED, "side", i]))
    before_is_a = set(side_order[:6])
    responses = {
        phase: {
            (r["problem_id"], r["candidate"]): r
            for r in read_jsonl(source / phase / "dream/holdout/checkpoint/responses.jsonl")
        }
        for phase in ["before", "after"]
    }
    refs = {
        r["problem_id"]: r["reference"]
        for r in read_json(source / "before/dream/holdout/evaluation-pack.json")["rows"]
    }
    packet, key = [], []
    for index, p in enumerate(selected):
        identity = (p["problem_id"], p["candidate"])
        sides = ["before", "after"] if index in before_is_a else ["after", "before"]
        item = {"review_id": f"R{index + 1:02d}", "reference": refs[identity[0]]}
        for side, phase in zip(["A", "B"], sides, strict=True):
            item[side] = extract_blocks(
                responses[phase][identity]["reward"]["scored_response"], "D"
            )["answer"]
        packet.append(item)
        key.append(
            {
                "review_id": item["review_id"],
                **p,
                "A_phase": sides[0],
                "B_phase": sides[1],
                "selection": "discordant"
                if p["delta"] > 0 and p["judge_delta"] < 0
                else "contrast",
            }
        )
    manifest = {
        "reviewer": "assistant in same conversation, not independent human",
        "known_context": "Aggregate v17 results and selection rule known; per-response provenance and scores concealed until annotations frozen.",
        "pairs": 12,
        "unique_reports": len({p["problem_id"] for p in selected}),
        "author_groups": 6,
        "pairs_per_group": 2,
        "discordant_pairs": 8,
        "contrast_pairs": 4,
        "selection_seed": SEED,
        "packet_hash": digest(packet),
        "rubric_hash": digest(RULE),
        "training_updates": 0,
        "test_reports_used": 0,
        "selection_rule": "All eight common-valid pairs with MiniLM increase and Phi-4 overall decrease; four hash-selected contrasts fill two pairs per author group, avoiding further repeated references.",
        "source_comparison_hash": digest(comparison),
    }
    for name, value in [
        ("packet.json", packet),
        ("provenance-key.json", key),
        ("manifest.json", manifest),
    ]:
        path = target / name
        if path.exists() and read_json(path) != value:
            raise ValueError("La preparación existente no se sobrescribe.")
        write_json(path, value)
    body = []
    for item in packet:
        body.append(f"<section><h2>{item['review_id']}</h2>")
        for label in ["reference", "A", "B"]:
            title = "Referencia humana" if label == "reference" else f"Respuesta {label}"
            body.append(f"<h3>{title}</h3><p>{html.escape(item[label])}</p>")
        body.append("</section>")
    (target / "revision.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>Dream · revisión de 12 pares</title>'
        "<style>body{max-width:900px;margin:40px auto;padding:0 22px;background:#faf9f5;color:#202622;font:18px/1.65 system-ui}section{border-top:1px solid #ccc;margin-top:40px}p{white-space:pre-wrap}h3{font-size:17px;color:#46564b}</style>"
        "<h1>Revisión de 12 pares</h1><p>Procedencia y scores ocultos. Sólo se revisa answer frente a la referencia; el contenido de dream queda libre. Revisión asistida, no humana independiente.</p>"
        + "".join(body)
        + "</html>"
    )
    print(
        "Preparación completa:",
        {k: manifest[k] for k in ["pairs", "unique_reports", "author_groups", "packet_hash"]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results/dream-v17"))
    parser.add_argument("--target", type=Path, default=Path("results/dream-v18/review"))
    args = parser.parse_args()
    prepare(args.source, args.target)
