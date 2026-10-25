"""Paquetes de revisión y dependencia posterior al piloto; límites del cegado explícitos."""

import argparse
from pathlib import Path

from experiments.preferences_v20 import ROOT, load_policy
from think_dream_lab.storage import append_jsonl, digest, read_json, read_jsonl, write_json


def quality_packet(root):
    spec = read_json(root / "protocol.json")
    records = {
        phase: {
            (r["problem_id"], r["candidate"]): r for r in read_jsonl(root / phase / "samples.jsonl")
        }
        for phase in ["before", "after"]
    }
    expected = {(r["problem_id"], i) for r in spec["dev_rows"] for i in range(2)}
    if any(set(v) != expected for v in records.values()):
        raise ValueError("Faltan evaluaciones.")
    requests = {r["problem_id"]: r["seeds"] for r in spec["dev_rows"]}
    order = sorted(expected, key=lambda k: digest([20261029, "quality", k]))
    sides = sorted(range(24), key=lambda i: digest([20261029, "side", i]))
    beforeA = set(sides[:12])
    packet = []
    key = []
    for i, k in enumerate(order):
        a, b = ("before", "after") if i in beforeA else ("after", "before")
        ra, rb = records[a][k], records[b][k]
        if ra["prompt_ids"] != rb["prompt_ids"] or ra["seed"] != rb["seed"]:
            raise ValueError("Cambió un contexto o semilla.")
        rid = f"E{i + 1:02d}"
        packet.append(
            {"review_id": rid, "request": requests[k[0]], "A": ra["dream"], "B": rb["dream"]}
        )
        key.append(
            {"review_id": rid, "problem_id": k[0], "candidate": k[1], "A_phase": a, "B_phase": b}
        )
    folder = root / "evaluation"
    for name, value in [("quality-packet.json", packet), ("quality-key.json", key)]:
        if (folder / name).exists() and read_json(folder / name) != value:
            raise ValueError("Paquete de calidad cambiado.")
        write_json(folder / name, value)
    # Variedad: dos muestras por modelo, lados equilibrados y sin scores ni fases visibles.
    variety = []
    vkey = []
    for i, row in enumerate(spec["dev_rows"]):
        a, b = ("before", "after") if i % 2 == 0 else ("after", "before")
        rid = f"V{i + 1:02d}"
        variety.append(
            {
                "review_id": rid,
                "request": row["seeds"],
                "A": [records[a][(row["problem_id"], j)]["dream"] for j in range(2)],
                "B": [records[b][(row["problem_id"], j)]["dream"] for j in range(2)],
            }
        )
        vkey.append({"review_id": rid, "problem_id": row["problem_id"], "A_phase": a, "B_phase": b})
    write_json(folder / "variety-packet.json", variety)
    write_json(folder / "variety-key.json", vkey)
    print("Prepared quality pairs: 24; variety comparisons: 12.")


def freeze(root, kind):
    folder = root / "evaluation"
    packet = read_json(folder / f"{kind}-packet.json")
    notes = read_json(folder / f"{kind}-annotations.json")
    if len(notes["ratings"]) != len(packet) or {r["review_id"] for r in notes["ratings"]} != {
        r["review_id"] for r in packet
    }:
        raise ValueError("Revisión incompleta.")
    for row in notes["ratings"]:
        original = next(p for p in packet if p["review_id"] == row["review_id"])
        if row["case_hash"] != digest(original):
            raise ValueError("Textos cambiados.")
        if row["preference"] not in {"A", "B", "tie", "no_se"}:
            raise ValueError("Preferencia inválida.")
    value = {
        "packet_hash": digest(packet),
        "annotations_hash": digest(notes),
        "reviewer": "assistant",
        "independent_human_review": False,
        "phase_key_consulted_before_freeze": False,
        "comparisons": len(packet),
    }
    if kind == "variety":
        value["source_order_predictable_from_code"] = notes.get(
            "source_order_predictable_from_visible_code", False
        )
        value["blinded_review_claimed"] = False
    if (folder / f"{kind}-frozen.json").exists() and read_json(
        folder / f"{kind}-frozen.json"
    ) != value:
        raise ValueError("Revisión congelada cambiada.")
    write_json(folder / f"{kind}-frozen.json", value)
    print(kind, "frozen", value)


def probes(root):
    import time

    import mlx.core as mx

    from think_dream_lab.backend import fingerprint, seed_all
    from think_dream_lab.dream_generation import generate_dream
    from think_dream_lab.holdout_evaluation import sha256

    old = Path("results/dream-v19/interventions")
    plan = read_json(old / "plan.json")
    tasks = [r for r in plan["tasks"] if r["condition"] in {"anchor", "edited", "paraphrase"}]
    oldrecords = {r["task_id"]: r for r in read_jsonl(old / "answers.jsonl")}
    checkpoint = Path(read_json(root / "dpo/run.json")["checkpoint"])
    model, tok = load_policy(checkpoint)
    before = fingerprint(model.parameters())
    folder = root / "dependency"
    previous = read_jsonl(folder / "answers.jsonl")
    done = {r["task_id"] for r in previous}
    if len(done) != len(previous):
        raise ValueError("Respuestas duplicadas.")
    for task in sorted(tasks, key=lambda r: digest([20261029, r["task_id"]])):
        if task["task_id"] in done:
            continue
        ids = tok.encode(task["prompt"], add_special_tokens=False)
        if ids != oldrecords[task["task_id"]]["prompt_ids"]:
            raise ValueError("Cambió el prefijo de la prueba.")
        seed_all(task["seed"])
        start = time.monotonic()
        tokens, texts, _ = generate_dream(model, tok, ids, 1, plan["generation"], "</answer>")
        raw = texts[0]
        ended = "</answer>" in raw
        answer = raw.split("</answer>", 1)[0].strip()
        record = {k: task[k] for k in ["task_id", "case_id", "condition", "seed", "probe"]}
        record.update(
            prompt_ids=ids,
            output_ids=tokens[0].tolist(),
            raw=raw,
            answer=answer,
            format_valid=ended
            and not any(t in answer for t in ["<dream>", "</dream>", "<answer>"]),
            seconds=time.monotonic() - start,
        )
        append_jsonl(folder / "answers.jsonl", record)
        done.add(task["task_id"])
        mx.clear_cache()
        print(f"Dependency after: {len(done)}/18", flush=True)
    if before != fingerprint(model.parameters()):
        raise RuntimeError("Cambió el modelo.")
    write_json(
        folder / "run.json",
        {
            "status": "complete",
            "parameters_unchanged": True,
            "optimizer_created": False,
            "training_updates": 0,
            "answers": 18,
            "checkpoint_sha256": sha256(checkpoint / "adapters.safetensors"),
            "same_prefix_ids_as_v19": True,
            "fresh_cache_per_generation": True,
        },
    )
    records = read_jsonl(folder / "answers.jsonl")
    records.sort(key=lambda r: digest([20261029, "dependency", r["task_id"]]))
    packet = [
        {"review_id": f"P{i + 1:02d}", "kind": r["probe"]["kind"], "answer": r["answer"]}
        for i, r in enumerate(records)
    ]
    key = [
        {
            "review_id": f"P{i + 1:02d}",
            "task_id": r["task_id"],
            "case_id": r["case_id"],
            "condition": r["condition"],
            "seed": r["seed"],
        }
        for i, r in enumerate(records)
    ]
    write_json(folder / "packet.json", packet)
    write_json(folder / "key.json", key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze-quality", "freeze-variety", "probes"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "prepare":
        quality_packet(args.root)
    elif args.action == "probes":
        probes(args.root)
    else:
        freeze(args.root, args.action.removeprefix("freeze-"))
