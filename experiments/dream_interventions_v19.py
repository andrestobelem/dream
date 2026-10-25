"""Intervenciones textuales pequeñas, sin optimizador y con contextos independientes."""

import argparse
import time
from pathlib import Path

from think_dream_lab.storage import append_jsonl, digest, read_json, read_jsonl, write_json

PROBES = [
    {
        "kind": "location",
        "anchor": "Before the scene ended, I put a copper key under the chair.",
        "edited": "Before the scene ended, I put a copper key inside the cupboard.",
        "paraphrase": "Near the end, I slipped a key made of copper beneath the chair.",
        "original_fact": "copper key under chair",
        "edited_fact": "copper key inside cupboard",
    },
    {
        "kind": "actor",
        "anchor": "My sister, not me, hid a silver coin under the table.",
        "edited": "I, not my sister, hid a silver coin under the table.",
        "paraphrase": "It was my sister who concealed a silver coin beneath the table.",
        "original_fact": "sister hides silver coin under table",
        "edited_fact": "narrator hides silver coin under table",
    },
    {
        "kind": "outcome",
        "anchor": "At the end, the tiny paper boat sank before it reached the shore.",
        "edited": "At the end, the tiny paper boat reached the shore without sinking.",
        "paraphrase": "The small paper boat went under before reaching land.",
        "original_fact": "paper boat sinks before reaching shore",
        "edited_fact": "paper boat reaches shore without sinking",
    },
]


def prepare(root):
    source = Path("results/dream-v17/before/dream/holdout/checkpoint/responses.jsonl")
    records = {(r["problem_id"], r["candidate"]): r for r in read_jsonl(source)}
    selected = read_json(root / "selection.json")[:3]
    rows = []
    for slot, (item, probe) in enumerate(zip(selected, PROBES, strict=True)):
        record = records[(item["problem_id"], item["candidate"])]
        raw = record["text"]
        close = raw.index("</dream>")
        prefix = record["prompt"]
        body = raw[:close]
        if any(word in prefix.lower() for word in ["copper key", "silver coin", "paper boat"]):
            raise ValueError("Un detalle de prueba ya estaba en la semilla.")
        rows.append(
            {
                "case_id": f"I{slot + 1}",
                "problem_id": item["problem_id"],
                "original_user_prefix": prefix,
                "dream": body,
                "probe": probe,
                "original_output_ids": record["output_ids"],
                "original_prompt_ids": record["prompt_ids"],
            }
        )
    evalpack = read_json(Path("results/dream-v17/before/dream/holdout/evaluation-pack.json"))[
        "rows"
    ]
    byid = {r["problem_id"]: r for r in evalpack}
    tasks = []
    for i, row in enumerate(rows):
        probe = row["probe"]
        other = rows[(i + 1) % 3]
        variants = {
            "original": row["dream"],
            "anchor": row["dream"] + "\n" + probe["anchor"] + "\n",
            "edited": row["dream"] + "\n" + probe["edited"] + "\n",
            "paraphrase": row["dream"] + "\n" + probe["paraphrase"] + "\n",
            "swapped": other["dream"] + "\n" + other["probe"]["anchor"] + "\n",
            "direct": None,
        }
        source_row = byid[row["problem_id"]]
        for condition, body in variants.items():
            if body is None:
                prompt = (
                    "A conversation between User and Assistant. The user requests a dream account, and the Assistant tells it. Write only the dream account within <answer> </answer> tags.\n"
                    + f"Write the dream account in approximately {source_row['target_words']} words.\nUser: {source_row['seeds']}\nAssistant:\n<answer>\n"
                )
            else:
                prompt = row["original_user_prefix"] + body + "</dream>\n<answer>\n"
            for seed in [17, 29]:
                tasks.append(
                    {
                        "task_id": digest([row["case_id"], condition, seed])[:16],
                        "case_id": row["case_id"],
                        "condition": condition,
                        "seed": seed,
                        "prompt": prompt,
                        "dream": body,
                        "probe": other["probe"] if condition == "swapped" else probe,
                        "target_words": source_row["target_words"],
                    }
                )
    source_run = read_json(Path("results/dream-v8/dream/paired/run.json"))
    spec = {
        "tasks": tasks,
        "tasks_hash": digest(tasks),
        "source_run": "results/dream-v8/dream/paired/run.json",
        "checkpoint": source_run["training_validation"]["checkpoint"],
        "generation": {
            "max_tokens": 1024,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
        },
        "checkpoint_step": 32,
        "conditions": list(variants),
        "cases": 3,
        "seeds": [17, 29],
        "expected_answers": 36,
        "training_updates": 0,
        "reference_in_prompt": False,
        "source_comparison_generation_cap": 2048,
        "fresh_cache_per_generation": True,
        "note": "Textual counterfactuals are encoded afresh including original. The original dream text preceding its closing delimiter is preserved verbatim, but the answer opening is canonicalized. Injected probes are artificial and appended; they test response to explicit text, not natural dream quality or subjective experience. Direct generation changes the instruction and total compute; it is not cost-matched.",
    }
    target = root / "interventions/plan.json"
    if target.exists() and read_json(target) != spec:
        raise ValueError("El diseño fijado no se sobrescribe.")
    write_json(target, spec)
    print("Intervention plan:", len(tasks), spec["tasks_hash"])


def run(root):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from think_dream_lab.backend import download_model, fingerprint, seed_all
    from think_dream_lab.dream_generation import generate_dream
    from think_dream_lab.holdout_evaluation import base_names_in_lora_layout, sha256

    folder = root / "interventions"
    spec = read_json(folder / "plan.json")
    if digest(spec["tasks"]) != spec["tasks_hash"]:
        raise ValueError("Cambió el lote.")
    source = read_json(Path(spec["source_run"]))
    checkpoint = Path(spec["checkpoint"])
    checkpoint_hash = sha256(checkpoint / "adapters.safetensors")
    config = source["config"]
    adapters = mx.load(str(checkpoint / "adapters.safetensors"))
    model, tok = load(
        download_model(config["model"]), tokenizer_config={"trust_remote_code": False}
    )
    model.eval()
    mx.eval(model.parameters())
    if (
        fingerprint(base_names_in_lora_layout(dict(tree_flatten(model.parameters())), adapters))
        != source["training_validation"]["base_after"]
    ):
        raise ValueError("Base incorrecta.")
    lora = config["lora"]
    model.freeze()
    linear_to_lora_layers(
        model, lora["layers"], {k: lora[k] for k in ["rank", "scale", "dropout", "keys"]}
    )
    model.load_weights(str(checkpoint / "adapters.safetensors"), strict=False)
    mx.eval(model.parameters())
    model.freeze()
    model.eval()
    parameters = dict(tree_flatten(model.parameters()))
    if fingerprint({k: parameters[k] for k in adapters}) != fingerprint(adapters):
        raise ValueError("Adaptadores incorrectos.")
    before = fingerprint(model.parameters())
    logfile = folder / "answers.jsonl"
    previous = read_jsonl(logfile)
    done = {r["task_id"] for r in previous}
    if len(done) != len(previous):
        raise ValueError("Registros duplicados.")
    write_json(
        folder / "run.json",
        {
            "status": "running",
            "tasks_hash": spec["tasks_hash"],
            "checkpoint_sha256": checkpoint_hash,
            "parameters_before": before,
            "optimizer_created": False,
            "training_updates": 0,
        },
    )
    # Orden intercalado, fijado por hash y sin consultar scores ni resultados.
    for task in sorted(spec["tasks"], key=lambda r: digest([20261028, r["task_id"]])):
        if task["task_id"] in done:
            continue
        seed_all(task["seed"])
        ids = tok.encode(task["prompt"], add_special_tokens=False)
        if len(ids) > 4096:
            raise ValueError("No se recorta un prefijo largo.")
        start = time.monotonic()
        tokens, texts, _ = generate_dream(model, tok, ids, 1, spec["generation"], "</answer>")
        text = texts[0]
        ended = "</answer>" in text
        answer = text.split("</answer>", 1)[0].strip()
        record = {
            k: task[k] for k in ["task_id", "case_id", "condition", "seed", "probe", "target_words"]
        }
        record.update(
            prompt=task["prompt"],
            prompt_ids=ids,
            output_ids=tokens[0].tolist(),
            raw=text,
            answer=answer,
            format_valid=ended
            and not any(t in answer for t in ["<dream>", "</dream>", "<answer>"]),
            truncated=not ended and len(tokens[0]) >= spec["generation"]["max_tokens"],
            seconds=time.monotonic() - start,
            fresh_cache=True,
        )
        append_jsonl(logfile, record)
        done.add(task["task_id"])
        print(f"Interventions: {len(done)}/36", flush=True)
        mx.clear_cache()
    after = fingerprint(model.parameters())
    if before != after or sha256(checkpoint / "adapters.safetensors") != checkpoint_hash:
        raise RuntimeError("Cambió el modelo.")
    write_json(
        folder / "run.json",
        {
            "status": "complete",
            "tasks_hash": spec["tasks_hash"],
            "checkpoint_sha256": checkpoint_hash,
            "parameters_before": before,
            "parameters_after": after,
            "parameters_unchanged": True,
            "adapter_loaded_exactly": True,
            "optimizer_created": False,
            "training_updates": 0,
            "answers": len(done),
            "fresh_cache_per_generation": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, default=Path("results/dream-v19"))
    args = parser.parse_args()
    prepare(args.root) if args.action == "prepare" else run(args.root)
