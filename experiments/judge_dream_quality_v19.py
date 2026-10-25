"""Evaluación exploratoria por pares, con inversión de lados y abstenciones explícitas."""

import argparse
from pathlib import Path

from experiments.dream_quality_v19 import RULE, save_unchanged
from think_dream_lab.backend import fingerprint
from think_dream_lab.judge import Judge
from think_dream_lab.storage import digest, read_json, write_json


def run(root):
    packet = read_json(root / "quality-packet.json")
    frozen = read_json(root / "quality-frozen.json")
    if (
        digest(packet) != frozen["packet_hash"]
        or digest(RULE) != frozen["rubric_hash"]
        or digest(read_json(root / "quality-annotations.json")) != frozen["annotations_hash"]
    ):
        raise ValueError("Cambió la revisión o la rúbrica fijada.")
    spec = {
        "id": "mlx-community/Phi-4-mini-instruct-4bit",
        "revision": "ac1c269cb4222a4e136a3d09edad301056c1f36a",
        "base_model": "microsoft/Phi-4-mini-instruct",
        "temperature": 0.0,
        "max_tokens": 512,
        "enable_thinking": False,
        "rubric_version": "dream-quality-v19",
    }
    folder = root / "quality-judge"
    save_unchanged(
        folder / "protocol.json",
        {
            "judge": spec,
            "rubric_hash": digest(RULE),
            "packet_hash": digest(packet),
            "swap_sides": True,
            "failure_policy": "Failed parsing is an evaluation error, not a zero actor reward; no training uses these results.",
            "coarse_gate": "At least 9/10 coarse-control decisions, both identical-text ties; exploratory only, not validation for RL.",
            "not_independent_human_validation": True,
        },
    )
    judge = Judge({"judge": spec}, folder)
    lock = read_json(folder / "dream/judge-lock.json")
    write_json(folder / "dream/judge-lock.json", {**lock, "rubric_hash": digest(RULE)})
    before = fingerprint(judge.model.parameters())
    ratings = []
    for row in packet:
        for swapped in [False, True]:
            A, B = (row["B"], row["A"]) if swapped else (row["A"], row["B"])
            payload = {"motifs": row["motifs"], "A": A, "B": B}
            item = {
                "review_id": row["review_id"],
                "swapped": swapped,
                "payload_hash": digest(payload),
            }
            try:
                value = judge.call("dream-quality-pair-v19", RULE, payload)
                if not all(k in value for k in ["choice", "A_mode", "B_mode", "reason"]):
                    raise ValueError("Missing fields")
                if value["choice"] not in {"A", "B", "tie", "no_se"}:
                    raise ValueError("Unknown choice")
                if any(
                    value[k] not in {"scene", "mixed", "plan", "list", "analysis"}
                    for k in ["A_mode", "B_mode"]
                ):
                    raise ValueError("Unknown mode")
                if not isinstance(value["reason"], str):
                    raise TypeError("Unknown reason")
                canonical = value["choice"]
                if swapped and canonical in {"A", "B"}:
                    canonical = {"A": "B", "B": "A"}[canonical]
                item.update(
                    status="complete",
                    choice=canonical,
                    value=value,
                    extra_fields=sorted(set(value) - {"choice", "A_mode", "B_mode", "reason"}),
                )
            except (ValueError, TypeError, KeyError) as error:
                item.update(status="evaluation_error", choice=None, error=repr(error))
            ratings.append(item)
            write_json(
                folder / "ratings.json",
                {
                    "ratings": ratings,
                    "rubric_hash": digest(RULE),
                    "packet_hash": digest(packet),
                    "training_updates": 0,
                },
            )
            print(f"Quality judge: {len(ratings)}/24", flush=True)
    if before != fingerprint(judge.model.parameters()):
        raise RuntimeError("El juez cambió.")
    write_json(
        folder / "validation.json",
        {
            "complete": True,
            "parameters_unchanged": True,
            "optimizer_created": False,
            "training_updates": 0,
            "evaluations": 24,
            "valid_evaluations": sum(r["status"] == "complete" for r in ratings),
            "rubric_hash": digest(RULE),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/dream-v19"))
    args = parser.parse_args()
    run(args.root)
