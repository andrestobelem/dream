"""Paquetes de lectura sin índices de muestreo; notas congeladas antes de revelar clave."""

import argparse
from pathlib import Path

from experiments.preferences_v20 import ROOT, pair_candidates
from think_dream_lab.storage import digest, read_json, read_jsonl, write_json


def prepare(root):
    protocol = read_json(root / "protocol.json")
    samples = read_jsonl(root / "train-pool/samples.jsonl")
    packet = []
    key = []
    for index, row in enumerate(protocol["train_rows"]):
        records = [r for r in samples if r["problem_id"] == row["problem_id"]]
        initial = pair_candidates([r for r in records if r["candidate"] < 4])
        final = (
            len(records) >= 8
            or len(records) >= 4
            and initial
            and initial[0] <= protocol["max_length_ratio"]
        )
        if not final:
            continue
        pair = pair_candidates(records)
        if pair is None:
            raise ValueError("No hay dos sueños distintos y válidos.")
        ratio, _, a, b = pair
        if int(digest([20261029, "sides", index]), 16) % 2:
            a, b = b, a
        rid = f"T{index + 1:02d}"
        packet.append({"review_id": rid, "request": row["seeds"], "A": a["dream"], "B": b["dream"]})
        key.append(
            {
                "review_id": rid,
                "problem_id": row["problem_id"],
                "A_id": a["sample_id"],
                "B_id": b["sample_id"],
                "length_ratio": ratio,
            }
        )
    folder = root / "preferences"
    if (folder / "packet.json").exists():
        previous = {r["review_id"]: r for r in read_json(folder / "packet.json")}
        current = {r["review_id"]: r for r in packet}
        if any(current.get(k) != v for k, v in previous.items()):
            raise ValueError("Un par revisado cambió.")
    write_json(folder / "packet.json", packet)
    write_json(folder / "key.json", key)
    print("Review pairs ready:", len(packet))


def freeze(root):
    folder = root / "preferences"
    packet = read_json(folder / "packet.json")
    notes = read_json(folder / "annotations.json")
    if (
        len(packet) != 24
        or len(notes["ratings"]) != 24
        or {r["review_id"] for r in packet} != {r["review_id"] for r in notes["ratings"]}
    ):
        raise ValueError("Faltan 24 revisiones.")
    for r in notes["ratings"]:
        original = next(p for p in packet if p["review_id"] == r["review_id"])
        if r["case_hash"] != digest(original):
            raise ValueError("Otro texto en la revisión.")
        if r["preference"] not in {"A", "B", "tie", "no_se"}:
            raise ValueError("Preferencia inválida.")
        if r["eligible"] and (r["preference"] not in {"A", "B"} or r["confidence"] == "low"):
            raise ValueError("Duda elegida para entrenamiento.")
    value = {
        "packet_hash": digest(packet),
        "annotations_hash": digest(notes),
        "pairs": 24,
        "reviewer": "assistant",
        "independent_human_review": False,
        "candidate_indices_hidden_during_review": True,
        "source_checkpoint_known": 32,
    }
    path = folder / "review-frozen.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Las notas congeladas cambiaron.")
    write_json(path, value)
    print("Preferences frozen:", value)


def build(root):
    folder = root / "preferences"
    notes = read_json(folder / "annotations.json")
    frozen = read_json(folder / "review-frozen.json")
    if (
        digest(notes) != frozen["annotations_hash"]
        or digest(read_json(folder / "packet.json")) != frozen["packet_hash"]
    ):
        raise ValueError("Se modificó la revisión.")
    key = {r["review_id"]: r for r in read_json(folder / "key.json")}
    samples = {r["sample_id"]: r for r in read_jsonl(root / "train-pool/samples.jsonl")}
    p = read_json(root / "protocol.json")
    pairs = []
    excluded = []
    for row in notes["ratings"]:
        k = key[row["review_id"]]
        if not row["eligible"]:
            excluded.append({"review_id": row["review_id"], "reason": "tie_or_uncertain"})
            continue
        a, b = (
            samples[k[row["preference"] + "_id"]],
            samples[k[{"A": "B", "B": "A"}[row["preference"]] + "_id"]],
        )
        if k["length_ratio"] > p["max_length_ratio"]:
            excluded.append({"review_id": row["review_id"], "reason": "length_confound"})
            continue
        if any(
            len(r["prompt_ids"]) + len(r["output_ids"]) > p["training"]["max_total_sequence_tokens"]
            for r in [a, b]
        ):
            excluded.append({"review_id": row["review_id"], "reason": "operational_sequence_limit"})
            continue
        pairs.append(
            {
                "pair_id": row["review_id"],
                "problem_id": k["problem_id"],
                "chosen": a,
                "rejected": b,
                "review_note": row["note"],
                "confidence": row["confidence"],
            }
        )
    result = {
        "pairs": pairs,
        "pairs_hash": digest(pairs),
        "annotations_hash": frozen["annotations_hash"],
        "reviewed_pairs": 24,
        "eligible_pairs": len(pairs),
        "excluded": excluded,
        "reference_in_prompt": False,
    }
    path = folder / "training-pack.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("El lote no se sobrescribe.")
    write_json(path, result)
    print("Eligible DPO pairs:", len(pairs), "excluded:", excluded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "build"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    globals()[args.action](args.root)
