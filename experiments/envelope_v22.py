"""Comparación nueva de ambos checkpoints con la misma envoltura; cero entrenamiento."""

import argparse
import time
from pathlib import Path

from experiments.evaluate_preferences_v21 import packets
from experiments.preferences_v20 import load_policy, locked_write
from think_dream_lab.dream_envelope import compose, generate_stage
from think_dream_lab.dream_task import extract_blocks
from think_dream_lab.dreambank_task import render_prompt
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import append_jsonl, digest, read_json, read_jsonl

ROOT = Path("results/dream-v22")
BRIEFS = [
    "Dream about a shoe repair shop, your grandfather waiting outside, a bell without a clapper, and a parcel addressed to you.",
    "Dream about a narrow canal, a yellow canoe, your friend losing a paddle, and houses slowly turning their backs.",
    "Dream about a post office, a damp envelope, your cousin behind the counter, and letters arriving from tomorrow.",
    "Dream about a dance lesson, a torn curtain, someone wearing your old jacket, and music coming from beneath the floor.",
    "Dream about a hospital kitchen, a bowl of cherries, your brother checking a list, and cupboards taller than the ceiling.",
    "Dream about a cinema queue, a ticket for yesterday, your mother holding your place, and hearing the ending before entering.",
    "Dream about a greenhouse, fogged glasses, a friend asleep beside a cactus, and a hose tied into a knot.",
    "Dream about a crowded lift, a basket of clean towels, your aunt on another floor, and buttons that change numbers.",
    "Dream about a riverside picnic, an unfinished letter, your sister building a kite, and a shadow moving against the wind.",
    "Dream about an empty swimming pool, a lost whistle, your old coach sitting on the bottom, and footsteps overhead.",
    "Dream about a tailor's room, a pocket full of stones, your father asking for directions, and a mirror showing another street.",
    "Dream about a pottery class, wet clay on your knees, a neighbor borrowing a spoon, and bowls turning transparent.",
    "Dream about a country road, a stalled taxi, a driver carrying flowers, and a sign pointing toward your old home.",
    "Dream about a newspaper kiosk, a familiar face on the front page, your uncle buying sweets, and pages too heavy to lift.",
    "Dream about an attic, a suitcase full of feathers, your grandmother humming, and a window overlooking your childhood classroom.",
    "Dream about a crowded dining car, an empty plate, a friend changing accents, and the landscape standing still outside.",
    "Dream about a winter beach, a red bucket, children drawing circles, and your footprints appearing ahead of you.",
    "Dream about a music shop, a piano with missing keys, your cousin asking for silence, and rain dripping from a lampshade.",
    "Dream about a courthouse staircase, a forgotten scarf, your teacher carrying a melon, and doors opening at different heights.",
    "Dream about a bus depot, a sleeping mechanic, your bicycle hanging from a hook, and engines sounding like distant conversations.",
    "Dream about a small restaurant, an extra place setting, your mother arriving with a map, and soup cooling into glass.",
    "Dream about a mountain shelter, a broken zip, your friend lending you socks, and stars visible through the floor.",
    "Dream about a grocery delivery, a basket of lemons, a neighbor who cannot hear you, and an address changing as you read it.",
    "Dream about a university courtyard, a rolled carpet, your brother waving from a window, and a fountain running upward.",
]


def prepare(root):
    old = read_json(Path("results/dream-v21/protocol.json"))
    trained = read_json(Path("results/dream-v21/dpo/run.json"))
    source = read_json(Path(old["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    assert trained["status"] == "complete" and trained["source_unchanged"]
    rows = [
        {
            "problem_id": f"v22-dev-{i + 1:02d}",
            "seeds": brief,
            "target_words": 100,
            "prompt": render_prompt(brief, response_prefix="<dream>\n", target_words=100),
            "source": "assistant-authored before sampling, no reserved reports",
        }
        for i, brief in enumerate(BRIEFS)
    ]
    assert not {r["seeds"] for r in rows} & {r["seeds"] for r in old["dev_rows"]}
    spec = {
        "version": "v22-envelope-no-training",
        "source_run": old["source_run"],
        "checkpoints": {"before": str(origin), "after": trained["checkpoint"]},
        "checkpoint_hashes": {
            "before": sha256(origin / "adapters.safetensors"),
            "after": sha256(Path(trained["checkpoint"]) / "adapters.safetensors"),
        },
        "source_files": trained["source_files"],
        "dev_rows": rows,
        "generation": {**old["generation"], "max_tokens": 1024},
        "evaluation": {**old["evaluation"], "seeds": [74242, 74343, 74444, 74545]},
        "envelope": {
            "stages": ["dream", "answer"],
            "stage_budget": 1024,
            "total_content_budget": 2048,
            "dream_seed": "evaluation seed",
            "answer_seed": "evaluation seed + 1",
            "boundaries": [
                "<dream>",
                "</dream>",
                "<answer>",
                "</answer>",
                "line-start User/Assistant/Human:",
                "EOS",
            ],
            "tags_written_by_program": True,
            "content_not_rewritten": True,
            "boundary_suffix_may_be_removed_inside_merged_token": True,
            "budget_exhaustion_or_empty_body_is_failure_even_if_tags_valid": True,
            "no_retry": True,
        },
        "dependency": {
            "source_plan": "results/dream-v19/interventions/plan.json",
            "conditions": ["anchor", "edited", "paraphrase"],
            "samples_per_model": 18,
            "same_old_prefix_ids_and_seeds": True,
            "stage": "answer only",
            "limit": 1024,
            "both_models_regenerated_with_same_controller": True,
        },
        "interpretation": "New inference system comparison, not learned format improvement. Cannot attribute outcome solely to checkpoints across v21/v22; fresh requests and decoding changed.",
        "training_updates": 0,
        "test_reports_used": 0,
        "independent_human_review": False,
        "reference_in_prompt": False,
        "sample_code_sha256": sha256(Path(__file__)),
        "controller_code_sha256": sha256(Path("think_dream_lab/dream_envelope.py")),
        "criterion_code_sha256": sha256(Path("experiments/criteria_preferences_v21.py")),
    }
    locked_write(root / "protocol.json", spec)
    locked_write(Path("docs/envoltura-v22-protocolo.json"), spec)
    print(
        "Frozen v22: 24 fresh requests, four paired samples, same controller on both checkpoints; zero updates."
    )


def load(root, phase):
    p = read_json(root / "protocol.json")
    assert p["sample_code_sha256"] == sha256(Path(__file__))
    assert p["controller_code_sha256"] == sha256(Path("think_dream_lab/dream_envelope.py"))
    for name, folder in p["checkpoints"].items():
        assert sha256(Path(folder) / "adapters.safetensors") == p["checkpoint_hashes"][name]
    model, tok = load_policy(Path(p["checkpoints"][phase]))
    return p, model, tok


def sample(root, phase, dependency=False):
    import mlx.core as mx

    from think_dream_lab.backend import fingerprint, seed_all

    p, model, tok = load(root, phase)
    before = fingerprint(model.parameters())
    folder = root / ("dependency/" + phase if dependency else phase)
    identity = digest([p, phase, dependency])
    previous = read_jsonl(folder / "samples.jsonl")
    assert len({r["sample_id"] for r in previous}) == len(previous)
    assert all(r["identity"] == identity for r in previous)
    if (folder / "run.json").exists():
        assert read_json(folder / "run.json")["identity"] == identity
        return
    done = {r["sample_id"] for r in previous}
    if dependency:
        plan = read_json(Path(p["dependency"]["source_plan"]))
        tasks = [r for r in plan["tasks"] if r["condition"] in p["dependency"]["conditions"]]
    else:
        tasks = [
            {**r, "candidate": i, "seed": seed}
            for r in p["dev_rows"]
            for i, seed in enumerate(p["evaluation"]["seeds"])
        ]
    for task in tasks:
        sid = task["task_id"] if dependency else digest([task["problem_id"], task["candidate"]])
        if sid in done:
            continue
        start = time.monotonic()
        if dependency:
            seed_all(task["seed"])
            ids = tok.encode(task["prompt"], add_special_tokens=False)
            answer = generate_stage(model, tok, ids, p["generation"])
            body = answer["content"]
            record = {
                "task_id": sid,
                "kind": task["probe"]["kind"],
                "answer": body,
                "stages": {"answer": answer},
                "syntax_valid": bool(body),
                "format_valid": answer["complete"],
                "output_ids": answer["output_ids"],
                "prompt_ids": ids,
                "seed": task["seed"],
            }
        else:
            ids = tok.encode(task["prompt"], add_special_tokens=False)
            seed_all(task["seed"])
            dream = generate_stage(model, tok, ids, p["generation"])
            account_prefix = task["prompt"] + dream["content"] + "\n</dream>\n<answer>\n"
            seed_all(task["seed"] + 1)
            answer = generate_stage(
                model, tok, tok.encode(account_prefix, add_special_tokens=False), p["generation"]
            )
            raw = compose(dream["content"], answer["content"])
            blocks = extract_blocks(raw, "D")
            record = {
                "problem_id": task["problem_id"],
                "candidate": task["candidate"],
                "seed": task["seed"],
                "prompt": task["prompt"],
                "prompt_ids": ids,
                "raw": raw,
                "dream": dream["content"],
                "answer": answer["content"],
                "stages": {"dream": dream, "answer": answer},
                "syntax_valid": bool(blocks["format"]),
                "format_valid": bool(blocks["format"]) and dream["complete"] and answer["complete"],
                "truncated": any(s["stop_reason"] == "budget" for s in [dream, answer]),
            }
        record.update(sample_id=sid, identity=identity, seconds=time.monotonic() - start)
        append_jsonl(folder / "samples.jsonl", record)
        previous.append(record)
        done.add(sid)
        mx.clear_cache()
        print(
            f"{phase} {'dependency' if dependency else 'development'} {len(previous)}/{len(tasks)}",
            flush=True,
        )
    assert len(previous) == len(tasks) and before == fingerprint(model.parameters())
    locked_write(
        folder / "run.json",
        {
            "status": "complete",
            "identity": identity,
            "samples": len(previous),
            "parameters_unchanged": True,
            "training_updates": 0,
            "checkpoint_sha256": p["checkpoint_hashes"][phase],
            "protocol_hash": digest(p),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(ROOT)
    else:
        for phase in ["before", "after"]:
            sample(ROOT, phase)
            sample(ROOT, phase, dependency=True)
        packets(ROOT)
