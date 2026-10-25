"""Piloto de preferencias revisadas: selección, muestreo y procedencia local."""

import argparse
import itertools
import time
from pathlib import Path

from think_dream_lab.dream_task import extract_blocks
from think_dream_lab.dreambank_task import render_prompt
from think_dream_lab.storage import append_jsonl, digest, read_json, read_jsonl, write_json

ROOT = Path("results/dream-v20")
DEV_BRIEFS = [
    "Dream about a bus station, a borrowed umbrella, a missed stop, and quiet relief.",
    "Dream about an old apartment, stairs returning to the same landing, and a neighbor carrying an empty bowl.",
    "Dream about a grocery store, misplaced keys, and a cashier who recognizes an old friend.",
    "Dream about a train, a ticket slowly fading, your sleeping sibling, and forgetting the destination.",
    "Dream about a garden at dusk, a dog carrying a shoe, and a gate opening into a classroom.",
    "Dream about a waiting room, a clock running backward, and your father painting a wall.",
    "Dream about moving offices, unlabeled boxes, and a colleague speaking through a radio.",
    "Dream about a family supper, a chair floating slightly, a missing guest, and the smell of bread.",
    "Dream about a beach, the tide revealing a hallway, and a friend dropping a button on the way home.",
    "Dream about a bicycle ride, rain passing through closed windows, and a familiar bridge in the wrong place.",
    "Dream about a school library, books full of photographs, and embarrassment while returning a borrowed book.",
    "Dream about a hotel balcony, neighbors applauding, an empty suitcase, and distant bells.",
]


def locked_write(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError("El diseño congelado no se sobrescribe.")
    write_json(path, value)


def prepare(root):
    pack = read_json(Path("results/dream-v16/preparation/training-pack.json"))
    old = read_json(Path("results/dream-v8/dream/paired/frozen-training-pack.json"))
    excluded = {r["problem_id"] for r in old["rows"]["train"]}
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
    groups = sorted({r["author_group"] for r in pool}, key=lambda g: digest([20261029, g]))[:24]
    rows = [
        min(
            [r for r in pool if r["author_group"] == g],
            key=lambda r: digest([20261029, r["problem_id"]]),
        )
        for g in groups
    ]
    reserved = read_jsonl(Path("results/dream-v5/dream/data/validation.jsonl")) + read_jsonl(
        Path("results/dream-v5/dream/data/test.jsonl")
    )
    if {r["author_group"] for r in rows} & {r["author_group"] for r in reserved}:
        raise ValueError("Persona reservada en entrenamiento.")
    train = [
        {k: r[k] for k in ["problem_id", "author_group", "seeds", "target_words", "prompt"]}
        for r in rows
    ]
    if len(train) != 24 or len({r["author_group"] for r in train}) != 24:
        raise ValueError("Faltan 24 personas de entrenamiento.")
    dev = [
        {
            "problem_id": f"novel-{i + 1:02d}",
            "seeds": brief,
            "target_words": 100,
            "prompt": render_prompt(brief, response_prefix="<dream>\n", target_words=100),
            "source": "assistant-authored before training; not derived from held-out corpus reports",
        }
        for i, brief in enumerate(DEV_BRIEFS)
    ]
    protocol = {
        "source_run": "results/dream-v8/dream/paired/run.json",
        "source_step": 32,
        "train_rows": train,
        "dev_rows": dev,
        "dataset_hash": digest([train, dev]),
        "generation": {
            "max_tokens": 2048,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
        },
        "train_candidates_initial": 4,
        "train_candidates_max": 8,
        "pair_selection": "Different normalized dreams; smallest max word-length ratio across both dream and answer, no quality scores. Extra four samples only if no pair with ratio <=1.5.",
        "max_length_ratio": 1.5,
        "training": {
            "algorithm": "DPO sigmoid, completion token sums",
            "beta": 0.1,
            "learning_rate": 5e-6,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "epochs": 1,
            "train_seed": 13,
            "max_total_sequence_tokens": 1536,
            "minimum_clear_preferences": 6,
            "optimizer": "new AdamW, reset moments; source optimizer untouched",
            "credit_assignment": "entire sampled continuation including dream and answer; prompt/prefill excluded",
            "reference": "stationary checkpoint 32, scores cached before first update",
            "selection": "Only explicit eligible reviewed preferences, no ties or uncertainty; at most 24 updates",
        },
        "evaluation": {
            "samples_per_prompt": 2,
            "seeds": [4242, 4343],
            "paired_before_after": True,
            "primary": "assistant blind preference for dream quality, ties and uncertainty allowed",
            "secondary": [
                "format",
                "diversity between two samples per prompt as exploratory proxy",
                "dependency probes cached v19 before and matching after",
            ],
        },
        "test_reports_used": 0,
        "reference_in_generator_prompt": False,
        "automatic_copy_penalty": False,
        "automatic_length_penalty": False,
        "independent_human_review": False,
        "approval": "User authorized 24 new training pairs, assistant review, a brief preference pilot from checkpoint32 and comparison on separate development requests.",
    }
    locked_write(root / "protocol.json", protocol)
    print(
        "Prepared: 24 train prompts, 24 people, 12 novel development prompts.",
        protocol["dataset_hash"],
    )


def load_policy(checkpoint, trainable=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from think_dream_lab.backend import download_model, fingerprint
    from think_dream_lab.holdout_evaluation import base_names_in_lora_layout

    source = read_json(Path("results/dream-v8/dream/paired/run.json"))
    config = source["config"]
    model, tok = load(
        download_model(config["model"]), tokenizer_config={"trust_remote_code": False}
    )
    model.eval()
    mx.eval(model.parameters())
    adapters = mx.load(str(checkpoint / "adapters.safetensors"))
    if (
        fingerprint(base_names_in_lora_layout(dict(tree_flatten(model.parameters())), adapters))
        != source["training_validation"]["base_after"]
    ):
        raise ValueError("Base incorrecta.")
    model.freeze()
    lora = config["lora"]
    linear_to_lora_layers(
        model, lora["layers"], {k: lora[k] for k in ["rank", "scale", "dropout", "keys"]}
    )
    model.load_weights(str(checkpoint / "adapters.safetensors"), strict=False)
    mx.eval(model.parameters())
    params = dict(tree_flatten(model.parameters()))
    if fingerprint({k: params[k] for k in adapters}) != fingerprint(adapters):
        raise ValueError("Adaptadores diferentes.")
    if not trainable:
        model.freeze()
    model.eval()
    return model, tok


def pair_candidates(records):
    options = []
    for a, b in itertools.combinations([r for r in records if r["format_valid"]], 2):
        if " ".join(a["dream"].split()) == " ".join(b["dream"].split()):
            continue
        ratios = [
            max(len(a[k].split()), len(b[k].split())) / min(len(a[k].split()), len(b[k].split()))
            for k in ["dream", "answer"]
        ]
        options.append((max(ratios), digest([a["sample_id"], b["sample_id"]]), a, b))
    return min(options, key=lambda r: r[:2]) if options else None


def sample(root, stage):
    import mlx.core as mx

    from think_dream_lab.backend import fingerprint, seed_all
    from think_dream_lab.dream_generation import generate_dream
    from think_dream_lab.holdout_evaluation import sha256

    spec = read_json(root / "protocol.json")
    source = read_json(Path(spec["source_run"]))
    checkpoint = (
        Path(source["training_validation"]["checkpoint"])
        if stage != "after"
        else Path(read_json(root / "dpo/run.json")["checkpoint"])
    )
    if stage == "after" and read_json(root / "dpo/run.json")["status"] != "complete":
        raise ValueError("Falta entrenar.")
    model, tok = load_policy(checkpoint)
    before = fingerprint(model.parameters())
    folder = root / stage
    identity = digest([spec, stage, sha256(checkpoint / "adapters.safetensors")])
    previous = read_jsonl(folder / "samples.jsonl")
    done = {r["sample_id"] for r in previous}
    if len(done) != len(previous) or any(r["identity"] != identity for r in previous):
        raise ValueError("Muestreo duplicado o diferente.")
    rows = spec["train_rows"] if stage == "train-pool" else spec["dev_rows"]
    for row in rows:
        for index in range(8 if stage == "train-pool" else 2):
            if stage == "train-pool" and index == 4:
                existing = [r for r in previous if r["problem_id"] == row["problem_id"]]
                pair = pair_candidates(existing)
                if pair and pair[0] <= spec["max_length_ratio"]:
                    break
            sid = digest([row["problem_id"], index, stage == "train-pool"])
            if sid in done:
                continue
            seed = (
                int(digest([20261029, row["problem_id"], index])[:8], 16)
                if stage == "train-pool"
                else spec["evaluation"]["seeds"][index]
            )
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
            print(f"{stage}: {len(previous)} samples", flush=True)
            mx.clear_cache()
    after = fingerprint(model.parameters())
    if before != after:
        raise RuntimeError("Cambió el modelo en evaluación.")
    write_json(
        folder / "run.json",
        {
            "status": "complete",
            "identity": identity,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint / "adapters.safetensors"),
            "parameters_unchanged": True,
            "optimizer_created": False,
            "training_updates": 0,
            "samples": len(previous),
            "protocol_hash": digest(spec),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train-pool", "before", "after"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(args.root) if args.action == "prepare" else sample(args.root, args.action)
