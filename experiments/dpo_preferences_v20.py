"""DPO con referencia fija en caché y gradientes secuenciales equivalentes por par."""

import argparse
import json
import math
from pathlib import Path

from experiments.preferences_v20 import ROOT, load_policy
from think_dream_lab.storage import append_jsonl, digest, read_json, read_jsonl, write_json


def sequence_arrays(sample):
    import mlx.core as mx

    prompt, output = sample["prompt_ids"], sample["output_ids"]
    if not prompt or not output:
        raise ValueError("Secuencia vacía.")
    return mx.array([prompt + output]), mx.array([[0.0] * len(prompt) + [1.0] * len(output)])


def sequence_logp(model, sample):
    from mlx_lm_lora.trainer.dpo_trainer import get_token_scores

    x, mask = sequence_arrays(sample)
    return get_token_scores(model, x, mask).sum()


def pair_gradient(model, chosen, rejected, reference, beta):
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_map

    c = sequence_logp(model, chosen)
    r = sequence_logp(model, rejected)
    mx.eval(c, r)
    cv, rv = c.item(), r.item()
    logit = beta * ((cv - rv) - (reference[0] - reference[1]))
    coefficient = (-beta * mx.sigmoid(mx.array(-logit))).item()
    grad_fn = nn.value_and_grad(model, sequence_logp)
    _, cg = grad_fn(model, chosen)
    mx.eval(cg)
    mx.clear_cache()
    _, rg = grad_fn(model, rejected)
    mx.eval(rg)
    mx.clear_cache()
    grads = tree_map(lambda a, b: coefficient * (a - b), cg, rg)
    mx.eval(grads)
    loss = nn.softplus(mx.array(-logit)).item()
    return (
        loss,
        grads,
        {"logit": logit, "chosen_logp": cv, "rejected_logp": rv, "coefficient": coefficient},
    )


def train(root):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx import nn
    from mlx.utils import tree_flatten, tree_map

    from think_dream_lab import checkpoint
    from think_dream_lab.backend import fingerprint, frozen_fingerprint, seed_all
    from think_dream_lab.holdout_evaluation import sha256

    spec = read_json(root / "protocol.json")
    pack = read_json(root / "preferences/training-pack.json")
    frozen = read_json(root / "preferences/review-frozen.json")
    notes = read_json(root / "preferences/annotations.json")
    if (
        digest(notes) != frozen["annotations_hash"]
        or pack["annotations_hash"] != frozen["annotations_hash"]
        or digest(pack["pairs"]) != pack["pairs_hash"]
    ):
        raise ValueError("Preferencias no congeladas.")
    pairs = pack["pairs"]
    t = spec["training"]
    if len(pairs) < t["minimum_clear_preferences"]:
        raise ValueError("No hay suficientes preferencias claras; no se entrena.")
    allowed = {r["problem_id"] for r in spec["train_rows"]}
    if any(p["problem_id"] not in allowed for p in pairs):
        raise ValueError("Un pedido ajeno entró al entrenamiento.")
    if len({p["problem_id"] for p in pairs}) != len(pairs):
        raise ValueError("Hay más de un par por pedido.")
    for p in pairs:
        if p["chosen"]["prompt_ids"] != p["rejected"]["prompt_ids"]:
            raise ValueError("Contextos distintos dentro del par.")
        if any(
            len(p[k]["prompt_ids"]) + len(p[k]["output_ids"]) > t["max_total_sequence_tokens"]
            for k in ["chosen", "rejected"]
        ):
            raise ValueError("No se recortan trayectorias largas.")
    source = read_json(Path(spec["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    source_files = {
        n: sha256(origin / n)
        for n in [
            "adapters.safetensors",
            "optimizer.safetensors",
            "mlx_random.safetensors",
            "state.json",
        ]
    }
    folder = root / "dpo"
    identity = digest([spec, pack["pairs_hash"], source_files, sha256(Path(__file__))])
    path = folder / "run.json"
    if path.exists() and read_json(path)["identity"] != identity:
        raise ValueError("Cambió la identidad DPO.")
    if path.exists() and read_json(path)["status"] == "complete":
        return read_json(path)
    seed_all(t["train_seed"])
    model, _tok = load_policy(origin, trainable=True)
    initial = fingerprint(model.trainable_parameters())
    base = frozen_fingerprint(model)
    if base != source["training_validation"]["base_after"]:
        raise ValueError("Base incorrecta.")
    ref_path = folder / "reference-cache.json"
    if ref_path.exists():
        cache = read_json(ref_path)
        if cache["pairs_hash"] != pack["pairs_hash"] or cache["initial_adapters"] != initial:
            raise ValueError("Referencia diferente.")
    else:
        values = {}
        for i, p in enumerate(pairs):
            values[p["pair_id"]] = [
                sequence_logp(model, p[k]).item() for k in ["chosen", "rejected"]
            ]
            mx.clear_cache()
            print(f"Reference cache: {i + 1}/{len(pairs)}", flush=True)
        if initial != fingerprint(model.trainable_parameters()):
            raise RuntimeError("La referencia cambió.")
        cache = {
            "pairs_hash": pack["pairs_hash"],
            "initial_adapters": initial,
            "values": values,
            "checkpoint_sha256": source_files["adapters.safetensors"],
            "before_first_update": True,
        }
        write_json(ref_path, cache)
    optimizer = optim.AdamW(learning_rate=t["learning_rate"], weight_decay=t["weight_decay"])
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)
    order = np.random.default_rng(t["train_seed"]).permutation(len(pairs)).tolist()
    latest = folder / "checkpoints/latest.json"
    start = 0
    cp = None
    if latest.exists():
        cp = folder / "checkpoints" / read_json(latest)["directory"]
        state = checkpoint.load(
            folder / "checkpoints" / read_json(latest)["directory"], model, optimizer, identity
        )
        start = state["step"]
        # Checkpoint transaccional determina los logs válidos.
        log = read_jsonl(folder / "steps.jsonl")
        write_json(
            folder / "recovery-log.json", {"previous_records": len(log), "resume_step": start}
        )
        committed = [
            read_json(folder / "checkpoints" / f"step-{i:06d}" / "state.json")["extra"]["metric"]
            for i in range(1, start + 1)
        ]
        (folder / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in committed))
    write_json(
        path,
        {
            "status": "running",
            "identity": identity,
            "pairs_hash": pack["pairs_hash"],
            "steps_requested": len(pairs),
            "optimizer_reset": True,
            "source_files": source_files,
            "initial_adapters": initial,
            "base_before": base,
        },
    )
    for step in range(start + 1, len(pairs) + 1):
        p = pairs[order[step - 1]]
        ref = cache["values"][p["pair_id"]]
        loss, grads, metrics = pair_gradient(model, p["chosen"], p["rejected"], ref, t["beta"])
        leaves = [v for _, v in tree_flatten(grads)]
        norm = mx.sqrt(sum(mx.sum(v.astype(mx.float32) ** 2) for v in leaves))
        mx.eval(norm)
        if (
            not math.isfinite(loss)
            or not math.isfinite(norm.item())
            or norm.item() <= 0
            or not all(mx.all(mx.isfinite(v)).item() for v in leaves)
        ):
            raise FloatingPointError("Gradientes o pérdida inválidos.")
        scale = mx.minimum(1.0, t["grad_clip"] / (norm + 1e-12))
        grads = tree_map(lambda v, scale=scale: v * scale, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if not all(
            mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(model.trainable_parameters())
        ):
            raise FloatingPointError("Adaptadores inválidos.")
        record = {
            "step": step,
            "pair_id": p["pair_id"],
            "problem_id": p["problem_id"],
            "loss": loss,
            "gradient_norm": norm.item(),
            **metrics,
        }
        cp = checkpoint.save(
            folder / "checkpoints",
            model,
            optimizer,
            step=step,
            run_identity=identity,
            extra={"metric": record, "initial_adapters": initial},
        )
        append_jsonl(folder / "steps.jsonl", record)
        print(f"DPO {step}/{len(pairs)} loss={loss:.5f} grad={norm.item():.4f}", flush=True)
        mx.clear_cache()
    final = fingerprint(model.trainable_parameters())
    adapter_hash = fingerprint(mx.load(str(cp / "adapters.safetensors")))

    reloaded = dict(tree_flatten(model.trainable_parameters()))
    if fingerprint(reloaded) != adapter_hash:
        raise RuntimeError("Guardado incorrecto.")
    model.load_weights(str(cp / "adapters.safetensors"), strict=False)
    mx.eval(model.parameters())
    if (
        fingerprint(model.trainable_parameters()) != final
        or frozen_fingerprint(model) != base
        or any(sha256(origin / n) != v for n, v in source_files.items())
    ):
        raise RuntimeError("Recarga o base no coinciden.")
    final_losses = []
    for p in pairs:
        ref = cache["values"][p["pair_id"]]
        c = sequence_logp(model, p["chosen"]).item()
        r = sequence_logp(model, p["rejected"]).item()
        final_losses.append(
            nn.softplus(mx.array(-t["beta"] * ((c - r) - (ref[0] - ref[1])))).item()
        )
        mx.clear_cache()
    result = {
        "status": "complete",
        "identity": identity,
        "checkpoint": str(cp.resolve()),
        "source_step": 32,
        "new_updates": len(pairs),
        "epochs": 1,
        "optimizer_reset": True,
        "source_files": source_files,
        "initial_adapters": initial,
        "final_adapters": final,
        "adapters_changed": initial != final,
        "base_unchanged": True,
        "source_unchanged": True,
        "reload_exact": True,
        "training_pairs_hash": pack["pairs_hash"],
        "initial_loss": math.log(2),
        "final_train_loss": float(np.mean(final_losses)),
        "not_used_for_checkpoint_selection": True,
    }
    write_json(path, result)
    print("DPO complete:", result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    train(args.root)
