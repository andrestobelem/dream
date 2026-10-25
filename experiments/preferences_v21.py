"""Preparación de contrastes revisados desde el 32; no ejecuta entrenamiento."""

import argparse
import itertools
import time
from collections import Counter
from pathlib import Path

from experiments.preferences_v20 import load_policy, locked_write, pair_candidates
from think_dream_lab.dream_task import extract_blocks
from think_dream_lab.dreambank_task import render_prompt
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import append_jsonl, digest, read_json, read_jsonl

ROOT = Path("results/dream-v21")
DEV_BRIEFS = [
    "Dream about washing dishes, a cracked blue cup, your aunt at the sink, and water that feels unexpectedly warm.",
    "Dream about a dentist's office, a coat on the wrong hook, your name called twice, and an unfinished conversation.",
    "Dream about watering a rooftop garden, a ladder growing shorter, a neighbor below, and a misplaced watering can.",
    "Dream about a ferry crossing, a bag of oranges, a cousin waving from shore, and a sudden change of weather.",
    "Dream about folding laundry, a shirt belonging to your childhood teacher, and someone knocking softly at the bedroom door.",
    "Dream about a village bakery, flour on your sleeves, a customer waiting silently, and the oven turning into a window.",
    "Dream about a quiet museum, a painting of your kitchen, a guard removing a frame, and difficulty finding the exit.",
    "Dream about a football field, shoes that feel too large, your brother keeping score, and everyone stopping to listen.",
    "Dream about a clinic corridor, a glass of milk, your grandmother carrying a newspaper, and an appointment you cannot remember.",
    "Dream about a camping trip, a torn tent, a friend searching for matches, and a hillside that changes direction.",
    "Dream about a crowded tram, a bag left on a seat, a child counting stations, and a sudden moment of silence.",
    "Dream about a seaside laundromat, a missing sock, your mother reading beside a machine, and sand covering the floor.",
    "Dream about a birthday party, candles that keep going out, an uncle arriving early, and a cake hidden under a table.",
    "Dream about a hardware shop, a box of bent nails, an assistant who knows your name, and sunlight through the ceiling.",
    "Dream about a snowy courtyard, a cat watching from a shed, wet gloves, and someone leaving without saying goodbye.",
    "Dream about a rehearsal room, a broken violin string, a friend changing seats, and an audience you can hear but cannot see.",
    "Dream about an orchard, a bucket of green apples, your father on a low branch, and the ground becoming soft.",
    "Dream about cleaning a classroom, a drawing beneath a desk, a former student returning, and chalk dust on your hands.",
    "Dream about a small airport, a passport with an unfamiliar photo, your sister waiting nearby, and a staircase behind the counter.",
    "Dream about a fishing pier, an empty thermos, a neighbor untangling a rope, and the water rising without making noise.",
    "Dream about a market stall, a scarf you remember losing, a seller refusing payment, and someone calling from an upstairs room.",
    "Dream about a holiday cabin, a mattress in the hallway, cousins whispering outside, and a door opening onto another season.",
    "Dream about a repair workshop, a watch missing its hands, a colleague offering tea, and a drawer that will not close.",
    "Dream about a long walk, a familiar dog following you, a path ending at a fence, and resting beside a stranger.",
]


def prepare(root):
    pack = read_json(Path("results/dream-v16/preparation/training-pack.json"))
    prior = read_json(Path("results/dream-v20/protocol.json"))
    old = read_json(Path("results/dream-v8/dream/paired/frozen-training-pack.json"))
    excluded = {r["problem_id"] for r in prior["train_rows"] + old["rows"]["train"]}
    controls = {
        digest(r[side])
        for r in read_json(Path("results/dream-v19/quality-packet.json"))
        for side in ["A", "B"]
    }
    pool = [
        r
        for r in pack["rows"]["train"]
        if r["problem_id"] not in excluded and digest(r["reference"]) not in controls
    ]
    groups = sorted({r["author_group"] for r in pool}, key=lambda g: digest([20261030, g]))[:24]
    selected = [
        min(
            [r for r in pool if r["author_group"] == g],
            key=lambda r: digest([20261030, r["problem_id"]]),
        )
        for g in groups
    ]
    # Sólo se consultan identidades de la partición reservada, no sus relatos.
    reserved_groups = set()
    for partition in ["validation", "test"]:
        reserved_groups.update(
            r["author_group"]
            for r in read_jsonl(Path(f"results/dream-v5/dream/data/{partition}.jsonl"))
        )
    assert len(selected) == 24 and not (set(groups) & reserved_groups)
    train = [
        {k: r[k] for k in ["problem_id", "author_group", "seeds", "target_words", "prompt"]}
        for r in selected
    ]
    dev = [
        {
            "problem_id": f"v21-dev-{i + 1:02d}",
            "seeds": brief,
            "target_words": 100,
            "prompt": render_prompt(brief, response_prefix="<dream>\n", target_words=100),
            "source": "assistant-authored before generation, not derived from held-out reports",
        }
        for i, brief in enumerate(DEV_BRIEFS)
    ]
    spec = {
        "version": "v21-preparation",
        "source_run": prior["source_run"],
        "source_step": 32,
        "train_rows": train,
        "dev_rows": dev,
        "dataset_hash": digest([train, dev]),
        "generation": prior["generation"],
        "train_candidates": 4,
        "selection": "Review all four dreams first; at most one clear best/worse contrast per request. No length-ratio cutoff or minimum-ratio pair selection. No fabricated negatives. Ties/uncertainty excluded.",
        "training": {
            **prior["training"],
            "minimum_clear_preferences": 12,
            "selection": "12 to 24 clear pairs, one per request; at least three chosen dreams longer and three shorter; otherwise not ready. Operational sequence <=1536, never truncate/recode.",
        },
        "evaluation": {
            "unit": "24 novel development requests, four paired samples per model/request",
            "seeds": [54242, 54343, 54444, 54545],
            "comparisons": 96,
            "primary": "Blind assistant preference on dream quality; ties and no_se allowed; frozen before source key; no source-alternating order",
            "quality_score": {"after": 1, "before": -1, "tie": 0, "no_se": 0},
            "aggregation": "Mean score within each request, then equally across 24 requests; no dropped failures",
            "interval": "Percentile bootstrap of 24 entire request clusters, 20000 replicates, seed 20261031; 95% exploratory interval including all ties/no_se",
            "promotion_all_required": {
                "quality_net_mean_minimum": 0.15,
                "quality_interval_lower_strictly_above": 0,
                "format_valid_minimum": 95,
                "format_count_not_below_before": True,
                "motif_major_omissions_or_contradictions_not_above_before": True,
                "motif_review": "Independent axis on all 96 dreams/model, including copied requests: central roles, objects and relations; count any major unsupported omission or contradiction once/sample. No single human target for these requests.",
                "variety_after_group_preferences_at_least_before": True,
                "variety_groups_with_fewer_than_two_usable_episodes_not_above_before": True,
                "variety_review": "24 blind groups of four samples/model; distinct episodes/relations, not lexical novelty; ties/no_se allowed and reported",
                "dependency_full_facts_minimum": 16,
                "dependency_valid_format_required": 18,
                "dependency": "Repeat exactly v19 anchor/edited/paraphrase, 18 after answers vs frozen 32 baseline; do not grant partial facts",
            },
            "failure_policy": "Any gate fails or review missing => keep32; no additional epochs, checkpoint selection or threshold changes based on results",
            "secondary": [
                "prompt-only echoes",
                "lengths",
                "length-direction strata",
                "motif vs quality tradeoffs",
            ],
            "limitations": "Same assistant reviews training and evaluation; development, not independent human validation; bootstrap does not include reviewer uncertainty. Success is provisional, no subjective-experience claim.",
        },
        "reserved_report_content_read": False,
        "test_reports_used": 0,
        "reference_in_generator_prompt": False,
        "automatic_copy_penalty": False,
        "automatic_length_penalty": False,
        "independent_human_review": False,
        "training_authorized_this_turn": False,
    }
    locked_write(root / "protocol.json", spec)
    source = read_json(Path(spec["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    locked_write(
        root / "source-files.json",
        {
            n: sha256(origin / n)
            for n in [
                "adapters.safetensors",
                "optimizer.safetensors",
                "mlx_random.safetensors",
                "state.json",
            ]
        },
    )
    public = {k: v for k, v in spec.items() if k not in {"train_rows", "dev_rows"}}
    public["dev_rows"] = dev
    locked_write(Path("docs/preferencias-v21-protocolo.json"), public)
    print(
        "Frozen: 24 fresh training requests, 24 people, 24 novel development requests.", flush=True
    )


def audit_old(root):
    old = Path("results/dream-v20")
    samples = read_jsonl(old / "train-pool/samples.jsonl")
    rows = []
    for row in read_json(old / "protocol.json")["train_rows"]:
        records = [r for r in samples if r["problem_id"] == row["problem_id"] and r["format_valid"]]
        chosen = pair_candidates(records)
        combinations = []
        for a, b in itertools.combinations(records, 2):
            if " ".join(a["dream"].split()) == " ".join(b["dream"].split()):
                continue
            ratio = max(
                max(len(a[k].split()), len(b[k].split()))
                / min(len(a[k].split()), len(b[k].split()))
                for k in ["dream", "answer"]
            )
            combinations.append(ratio)
        rows.append(
            {
                "problem_id_hash": digest(row["problem_id"]),
                "valid_candidates": len(records),
                "distinct_pairs": len(combinations),
                "ratio_above_1_5": sum(x > 1.5 for x in combinations),
                "selected_ratio": chosen[0] if chosen else None,
            }
        )
    value = {
        "candidate_records": len(samples),
        "requests": len(rows),
        "rows": rows,
        "distinct_pairs": sum(r["distinct_pairs"] for r in rows),
        "ratio_above_1_5": sum(r["ratio_above_1_5"] for r in rows),
        "quality_of_discarded_pairs_reviewed": False,
        "interpretation": "Counts describe the length filter opportunity set, not quality or a causal explanation of regression.",
    }
    locked_write(root / "old-pool-audit.json", value)
    locked_write(Path("docs/preferencias-v21-auditoria-pool.json"), value)
    print({k: v for k, v in value.items() if k != "rows"})


def sample(root):
    import mlx.core as mx

    from think_dream_lab.backend import fingerprint, seed_all
    from think_dream_lab.dream_generation import generate_dream

    spec = read_json(root / "protocol.json")
    source = read_json(Path(spec["source_run"]))
    checkpoint = Path(source["training_validation"]["checkpoint"])
    source_hashes = read_json(root / "source-files.json")
    assert all(sha256(checkpoint / n) == v for n, v in source_hashes.items())
    folder = root / "train-pool"
    identity = digest([spec, source_hashes, sha256(Path(__file__))])
    previous = read_jsonl(folder / "samples.jsonl")
    done = {r["sample_id"] for r in previous}
    if len(done) != len(previous) or any(r["identity"] != identity for r in previous):
        raise ValueError("Muestreo duplicado o identidad distinta.")
    model, tok = load_policy(checkpoint)
    before = fingerprint(model.parameters())
    for row in spec["train_rows"]:
        for index in range(4):
            sid = digest(["v21", row["problem_id"], index])
            if sid in done:
                continue
            seed = int(digest([20261030, row["problem_id"], index])[:8], 16)
            seed_all(seed)
            ids = tok.encode(row["prompt"], add_special_tokens=False)
            start = time.monotonic()
            tokens, texts, _ = generate_dream(model, tok, ids, 1, spec["generation"], "</answer>")
            blocks = extract_blocks("<dream>\n" + texts[0], "D")
            record = {
                "sample_id": sid,
                "identity": identity,
                "problem_id": row["problem_id"],
                "candidate": index,
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
            print(f"Candidate {len(previous)}/96; {record['seconds']:.1f}s", flush=True)
            mx.clear_cache()
    assert len(previous) == 96 and before == fingerprint(model.parameters())
    assert all(sha256(checkpoint / n) == v for n, v in source_hashes.items())
    locked_write(
        folder / "run.json",
        {
            "status": "complete",
            "identity": identity,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": source_hashes["adapters.safetensors"],
            "parameters_unchanged": True,
            "optimizer_created": False,
            "training_updates": 0,
            "samples": 96,
            "protocol_hash": digest(spec),
        },
    )


def packet(root):
    spec = read_json(root / "protocol.json")
    samples = read_jsonl(root / "train-pool/samples.jsonl")
    assert read_json(root / "train-pool/run.json")["status"] == "complete"
    rows, key = [], []
    for i, row in enumerate(spec["train_rows"]):
        records = sorted(
            [r for r in samples if r["problem_id"] == row["problem_id"]],
            key=lambda r: digest(["v21-review", r["sample_id"]]),
        )
        rid = f"N{i + 1:02d}"
        rows.append(
            {
                "review_id": rid,
                "request": row["seeds"],
                "candidates": {
                    side: {
                        "dream": r["dream"],
                        "format_valid": r["format_valid"],
                        "sequence_tokens": len(r["prompt_ids"]) + len(r["output_ids"]),
                    }
                    for side, r in zip("ABCD", records, strict=True)
                },
            }
        )
        key.append(
            {
                "review_id": rid,
                "problem_id": row["problem_id"],
                **{side + "_id": r["sample_id"] for side, r in zip("ABCD", records, strict=True)},
            }
        )
    locked_write(root / "preferences/packet.json", rows)
    locked_write(root / "preferences/key.json", key)
    print("Four-candidate review ready: 24 requests.")


def freeze(root):
    folder = root / "preferences"
    spec = read_json(root / "protocol.json")
    notes = read_json(folder / "annotations.json")
    packet = read_json(folder / "packet.json")
    assert len(notes["ratings"]) == 24 and len({r["review_id"] for r in notes["ratings"]}) == 24
    rows = {r["review_id"]: r for r in packet}
    for note in notes["ratings"]:
        row = rows[note["review_id"]]
        assert note["case_hash"] == digest(row)
        assert set(note["candidate_notes"]) == set("ABCD")
        assert note["preference"] in {"clear", "tie", "no_se"}
        if note["preference"] == "clear":
            assert note["chosen"] in "ABCD" and note["rejected"] in "ABCD"
            assert note["chosen"] != note["rejected"] and note["confidence"] in {"high", "medium"}
            for side in [note["chosen"], note["rejected"]]:
                c = row["candidates"][side]
                assert (
                    c["format_valid"]
                    and c["sequence_tokens"] <= spec["training"]["max_total_sequence_tokens"]
                )
    locked_write(
        folder / "review-frozen.json",
        {
            "packet_hash": digest(packet),
            "annotations_hash": digest(notes),
            "pairs": 24,
            "reviewer": "assistant",
            "independent_human_review": False,
            "candidate_indices_hidden_during_review": True,
            "source_checkpoint_known": 32,
        },
    )


def build(root):
    folder = root / "preferences"
    spec = read_json(root / "protocol.json")
    notes = read_json(folder / "annotations.json")
    frozen = read_json(folder / "review-frozen.json")
    assert digest(notes) == frozen["annotations_hash"]
    assert digest(read_json(folder / "packet.json")) == frozen["packet_hash"]
    key = {r["review_id"]: r for r in read_json(folder / "key.json")}
    samples = {r["sample_id"]: r for r in read_jsonl(root / "train-pool/samples.jsonl")}
    pairs, excluded, directions = [], [], Counter()
    for row in notes["ratings"]:
        if row["preference"] != "clear":
            excluded.append({"review_id": row["review_id"], "reason": row["preference"]})
            continue
        k = key[row["review_id"]]
        c, r = [samples[k[row[side] + "_id"]] for side in ["chosen", "rejected"]]
        assert c["prompt_ids"] == r["prompt_ids"] and c["format_valid"] and r["format_valid"]
        lengths = [len(x["dream"].split()) for x in [c, r]]
        direction = (
            "longer"
            if lengths[0] > lengths[1]
            else "shorter"
            if lengths[0] < lengths[1]
            else "equal"
        )
        directions[direction] += 1
        pairs.append(
            {
                "pair_id": row["review_id"],
                "problem_id": k["problem_id"],
                "chosen": c,
                "rejected": r,
                "review_note": row["note"],
                "confidence": row["confidence"],
            }
        )
    ready = (
        len(pairs) >= spec["training"]["minimum_clear_preferences"]
        and directions["longer"] >= 3
        and directions["shorter"] >= 3
    )
    value = {
        "pairs": pairs,
        "pairs_hash": digest(pairs),
        "annotations_hash": frozen["annotations_hash"],
        "reviewed_pairs": 24,
        "eligible_pairs": len(pairs),
        "excluded": excluded,
        "reference_in_prompt": False,
        "length_directions": dict(directions),
        "ready": ready,
    }
    locked_write(folder / "training-pack.json", value)
    locked_write(
        Path("docs/preferencias-v21-lote.json"), {k: v for k, v in value.items() if k != "pairs"}
    )
    print("Ready:", ready, "clear pairs:", len(pairs), "directions:", dict(directions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "audit-old", "sample", "packet", "freeze", "build"]
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {
        "prepare": prepare,
        "audit-old": audit_old,
        "sample": sample,
        "packet": packet,
        "freeze": freeze,
        "build": build,
    }[args.action](args.root)
