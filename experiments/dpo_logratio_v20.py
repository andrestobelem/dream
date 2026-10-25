"""Diagnóstico posterior de log-ratios por bloque; no modifica pérdida ni pesos."""

import argparse
import gc
from pathlib import Path

from experiments.dpo_preferences_v20 import sequence_arrays
from experiments.preferences_v20 import ROOT, load_policy
from think_dream_lab.storage import read_json, write_json


def token_regions(tokenizer, ids):
    regions = []
    state = "dream"
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[: i + 1])
        if state == "dream" and "</dream>" in prefix:
            regions.append("boundary")
            state = "answer" if "<answer>" in prefix else "between"
        elif state == "between":
            regions.append("boundary")
            if "<answer>" in prefix:
                state = "answer"
        elif state == "answer" and "</answer>" in prefix:
            regions.append("boundary")
            state = "closed"
        elif state == "closed":
            regions.append("boundary")
        else:
            regions.append(state)
    return regions


def analyze(root):
    import mlx.core as mx
    from mlx_lm_lora.trainer.dpo_trainer import get_token_scores

    from think_dream_lab.backend import fingerprint

    spec = read_json(root / "protocol.json")
    pack = read_json(root / "preferences/training-pack.json")
    source = read_json(Path(spec["source_run"]))
    after = read_json(root / "dpo/run.json")
    outputs = {}
    for phase, path in [
        ("before", source["training_validation"]["checkpoint"]),
        ("after", after["checkpoint"]),
    ]:
        model, tok = load_policy(Path(path))
        original = fingerprint(model.parameters())
        values = {}
        for i, pair in enumerate(pack["pairs"]):
            sides = {}
            for side in ["chosen", "rejected"]:
                sample = pair[side]
                x, mask = sequence_arrays(sample)
                scores = get_token_scores(model, x, mask)[0, len(sample["prompt_ids"]) - 1 :]
                mx.eval(scores)
                floats = scores.tolist()
                regions = token_regions(tok, sample["output_ids"])
                if len(floats) != len(regions):
                    raise ValueError("Tokens no alineados.")
                sides[side] = {
                    region: sum(v for v, r in zip(floats, regions, strict=True) if r == region)
                    for region in ["dream", "answer", "boundary"]
                }
                sides[side]["total"] = sum(floats)
                del scores
                mx.clear_cache()
            values[pair["pair_id"]] = sides
            print(f"Log-ratios {phase}: {i + 1}/{len(pack['pairs'])}", flush=True)
        if fingerprint(model.parameters()) != original:
            raise RuntimeError("Cambió el modelo.")
        outputs[phase] = values
        del model, tok
        gc.collect()
        mx.clear_cache()
    rows = []
    cache = read_json(root / "dpo/reference-cache.json")["values"]
    max_difference = 0.0
    for pair in pack["pairs"]:
        pid = pair["pair_id"]
        a = outputs["before"][pid]
        b = outputs["after"][pid]
        max_difference = max(
            max_difference,
            *(
                abs(a[side]["total"] - cache[pid][j])
                for j, side in enumerate(["chosen", "rejected"])
            ),
        )
        gain = {
            r: (b["chosen"][r] - a["chosen"][r]) - (b["rejected"][r] - a["rejected"][r])
            for r in ["dream", "answer", "boundary", "total"]
        }
        rows.append({"pair_id": pid, "margin_gain": gain})
    if max_difference > 0.02:
        raise ValueError("La caché de referencia no coincide con el 32.")
    n = len(rows)
    result = {
        "pairs": n,
        "max_reference_replay_difference": max_difference,
        "mean_margin_gain": {
            r: sum(x["margin_gain"][r] for x in rows) / n
            for r in ["dream", "answer", "boundary", "total"]
        },
        "pairs_with_positive_gain": {
            r: sum(x["margin_gain"][r] > 0 for x in rows) for r in ["dream", "answer", "total"]
        },
        "rows": rows,
        "training_updates": 0,
        "weights_unchanged": True,
        "post_hoc": True,
        "caution": "Diagnostic on training pairs, not causal attribution or evaluation quality. Boundary-spanning original sampled tokens remain intact in a separate boundary bin. No new loss masks or segmented training.",
    }
    write_json(root / "dpo/block-logratio-diagnostic.json", result)
    print(result["mean_margin_gain"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    analyze(args.root)
