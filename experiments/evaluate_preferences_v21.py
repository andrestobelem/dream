"""Generaciones pareadas y paquetes sin procedencia para la evaluación fijada v21."""

import argparse
import random
import secrets
import time
from pathlib import Path

from experiments.evaluate_preferences_v20 import probes
from experiments.preferences_v20 import load_policy, locked_write
from experiments.preferences_v21 import ROOT
from think_dream_lab.dream_task import extract_blocks
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import append_jsonl, digest, read_json, read_jsonl


def sample(root, stage):
    import mlx.core as mx

    from think_dream_lab.backend import fingerprint, seed_all
    from think_dream_lab.dream_generation import generate_dream

    spec = read_json(root / "protocol.json")
    source = read_json(Path(spec["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    if stage == "after":
        trained = read_json(root / "dpo/run.json")
        assert trained["status"] == "complete" and trained["new_updates"] == 19
        checkpoint = Path(trained["checkpoint"])
    else:
        checkpoint = origin
    hashes = read_json(root / "source-files.json")
    assert all(sha256(origin / n) == v for n, v in hashes.items())
    checkpoint_hash = sha256(checkpoint / "adapters.safetensors")
    identity = digest([spec, stage, checkpoint_hash, sha256(Path(__file__))])
    folder = root / stage
    previous = read_jsonl(folder / "samples.jsonl")
    done = {r["sample_id"] for r in previous}
    if len(done) != len(previous) or any(r["identity"] != identity for r in previous):
        raise ValueError("Muestras repetidas o identidad cambiada.")
    if (folder / "run.json").exists():
        assert read_json(folder / "run.json")["identity"] == identity
        assert len(previous) == 96
        return
    model, tok = load_policy(checkpoint)
    before = fingerprint(model.parameters())
    for row in spec["dev_rows"]:
        for i, seed in enumerate(spec["evaluation"]["seeds"]):
            sid = digest(["v21-eval", row["problem_id"], i])
            if sid in done:
                continue
            seed_all(seed)
            ids = tok.encode(row["prompt"], add_special_tokens=False)
            start = time.monotonic()
            tokens, texts, _ = generate_dream(model, tok, ids, 1, spec["generation"], "</answer>")
            blocks = extract_blocks("<dream>\n" + texts[0], "D")
            record = {
                "sample_id": sid,
                "identity": identity,
                "problem_id": row["problem_id"],
                "candidate": i,
                "seed": seed,
                "prompt": row["prompt"],
                "prompt_ids": ids,
                "output_ids": tokens[0].tolist(),
                "raw": texts[0],
                "dream": blocks["draft"],
                "answer": blocks["answer"],
                "format_valid": bool(blocks["format"]),
                "seconds": time.monotonic() - start,
                "truncated": "</answer>" not in texts[0]
                and len(tokens[0]) >= spec["generation"]["max_tokens"],
            }
            append_jsonl(folder / "samples.jsonl", record)
            previous.append(record)
            done.add(sid)
            print(f"{stage} {len(previous)}/96", flush=True)
            mx.clear_cache()
    assert len(previous) == 96 and before == fingerprint(model.parameters())
    assert all(sha256(origin / n) == v for n, v in hashes.items())
    locked_write(
        folder / "run.json",
        {
            "status": "complete",
            "identity": identity,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "protocol_hash": digest(spec),
            "samples": len(previous),
            "parameters_unchanged": True,
            "optimizer_created": False,
            "training_updates": 0,
        },
    )


def packets(root):
    folder = root / "evaluation"
    if (folder / "packets-complete.json").exists():
        return
    if (folder / "quality-packet.json").exists():
        raise ValueError(
            "Preparación interrumpida: conservar los paquetes ya escritos y recuperar su clave."
        )
    spec = read_json(root / "protocol.json")
    records = {
        stage: {
            (r["problem_id"], r["candidate"]): r for r in read_jsonl(root / stage / "samples.jsonl")
        }
        for stage in ["before", "after"]
    }
    expected = {(r["problem_id"], i) for r in spec["dev_rows"] for i in range(4)}
    assert all(
        set(records[stage]) == expected
        and read_json(root / stage / "run.json")["status"] == "complete"
        for stage in records
    )
    requests = {r["problem_id"]: r["seeds"] for r in spec["dev_rows"]}
    rng = random.Random(secrets.randbits(256))
    order = sorted(expected)
    rng.shuffle(order)
    sides = ["before"] * 48 + ["after"] * 48
    rng.shuffle(sides)
    packet, key = [], []
    for i, (k, a) in enumerate(zip(order, sides, strict=True)):
        b = "after" if a == "before" else "before"
        ra, rb = records[a][k], records[b][k]
        assert ra["prompt_ids"] == rb["prompt_ids"] and ra["seed"] == rb["seed"]
        rid = f"E{i + 1:02d}"
        packet.append(
            {
                "review_id": rid,
                "request": requests[k[0]],
                "A": ra["dream"],
                "B": rb["dream"],
                "A_format": ra["format_valid"],
                "B_format": rb["format_valid"],
            }
        )
        key.append(
            {"review_id": rid, "problem_id": k[0], "candidate": k[1], "A_phase": a, "B_phase": b}
        )
    vpacket, vkey = [], []
    vorder = list(requests)
    rng.shuffle(vorder)
    vsides = ["before"] * 12 + ["after"] * 12
    rng.shuffle(vsides)
    for i, (pid, a) in enumerate(zip(vorder, vsides, strict=True)):
        b = "after" if a == "before" else "before"
        rid = f"V{i + 1:02d}"
        groups = {}
        for side, phase in [("A", a), ("B", b)]:
            samples = list(range(4))
            rng.shuffle(samples)
            groups[side] = [records[phase][(pid, j)]["dream"] for j in samples]
        vpacket.append({"review_id": rid, "request": requests[pid], **groups})
        vkey.append({"review_id": rid, "problem_id": pid, "A_phase": a, "B_phase": b})
    for name, value in [
        ("quality-key", key),
        ("variety-key", vkey),
        ("quality-packet", packet),
        ("variety-packet", vpacket),
    ]:
        locked_write(folder / f"{name}.json", value)
    locked_write(
        folder / "packets-complete.json",
        {
            "quality_hash": digest(packet),
            "variety_hash": digest(vpacket),
            "quality_pairs": 96,
            "variety_groups": 24,
            "source_order": "private entropy, no alternating rule; keys not printed",
        },
    )
    print("Packets ready: 96 quality pairs, 24 variety groups; keys not printed.")


def freeze(root, kind):
    folder = root / "evaluation"
    packet = read_json(folder / f"{kind}-packet.json")
    notes = read_json(folder / f"{kind}-annotations.json")
    by_id = {r["review_id"]: r for r in packet}
    assert len(notes["ratings"]) == len(packet) and {
        r["review_id"] for r in notes["ratings"]
    } == set(by_id)
    assert len({r["review_id"] for r in notes["ratings"]}) == len(packet)
    for r in notes["ratings"]:
        assert r["case_hash"] == digest(by_id[r["review_id"]])
        assert r["preference"] in {"A", "B", "tie", "no_se"}
        if kind == "quality":
            for side in "AB":
                assert isinstance(r["motifs"][side]["major"], bool)
                assert r["motifs"][side]["note"]
        else:
            assert all(0 <= r["usable_episodes"][s] <= 4 for s in "AB")
    locked_write(
        folder / f"{kind}-frozen.json",
        {
            "packet_hash": digest(packet),
            "annotations_hash": digest(notes),
            "comparisons": len(packet),
            "reviewer": "assistant",
            "independent_human_review": False,
            "phase_key_consulted_before_freeze": False,
            "source_order_predictable_from_code": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["before", "after", "probes", "packets", "freeze-quality", "freeze-variety"],
    )
    args = parser.parse_args()
    if args.action in {"before", "after"}:
        sample(ROOT, args.action)
    elif args.action == "probes":
        probes(ROOT)
    elif args.action == "packets":
        packets(ROOT)
    else:
        freeze(ROOT, args.action.removeprefix("freeze-"))
