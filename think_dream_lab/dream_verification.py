"""Continuidad de entrenamiento con corpus, consignas y juez del protocolo de sueños."""

import traceback

from .backend import fingerprint
from .config import freeze_model, identity
from .corpus import prepare
from .dream_workflow import DreamRunner
from .judge import Judge
from .storage import digest, read_json, write_json


def compare_checkpoints(expected, actual):
    import mlx.core as mx

    a, b = read_json(expected / "state.json"), read_json(actual / "state.json")
    return {
        "same_identity": a["identity"] == b["identity"],
        "same_step": a["step"] == b["step"],
        "same_adapters": fingerprint(mx.load(str(expected / "adapters.safetensors")))
        == fingerprint(mx.load(str(actual / "adapters.safetensors"))),
        "same_optimizer": fingerprint(mx.load(str(expected / "optimizer.safetensors")))
        == fingerprint(mx.load(str(actual / "optimizer.safetensors"))),
        "same_mlx_random": fingerprint(mx.load(str(expected / "mlx_random.safetensors")))
        == fingerprint(mx.load(str(actual / "mlx_random.safetensors"))),
        "same_python_random": a["python_random"] == b["python_random"],
        "same_numpy_random": a["numpy_random"] == b["numpy_random"],
        "same_training_tokens": a["extra"]["training_tokens"] == b["extra"]["training_tokens"],
        "same_final_rollouts": digest(logical_rollouts(a["extra"]["responses"]))
        == digest(logical_rollouts(b["extra"]["responses"])),
    }


def logical_rollouts(rows):
    timing = {"group_seconds", "latency_seconds", "mlx_peak_bytes"}
    return [{k: v for k, v in row.items() if k not in timing} for row in rows]


def verify_continuity(config, root):
    import mlx.core as mx

    source = root / "dream" / "smoke"
    manifest = read_json(source / "run.json")
    if manifest["status"] != "complete":
        raise ValueError("El smoke de sueños debe estar completo para verificar continuidad.")
    data, corpus = prepare(config, root)
    lock = freeze_model(config, root / "dream")
    run_id = identity(
        config,
        lock,
        digest(
            [corpus["hash"], manifest["briefs"]]
            + ([manifest["training_order"]] if "training_order" in manifest else [])
        ),
        "D",
        manifest["seed"],
    )
    if run_id != manifest["identity"]:
        raise ValueError("El código, corpus, modelo o protocolo difieren del smoke de sueños.")
    final_step = manifest["steps_requested"]
    split_step = min(2, final_step - 1)
    if split_step < 1:
        raise ValueError("La continuidad requiere al menos dos actualizaciones.")
    folder = root / "dream" / "diagnostics" / "resume-verification"
    path = folder / "run.json"
    if path.exists():
        raise ValueError("Ya hay una verificación de continuidad; se conservan sus artefactos.")
    state = {
        "kind": "diagnostic",
        "status": "running",
        "identity": run_id,
        "condition": "D",
        "seed": manifest["seed"],
        "steps_requested": final_step,
        "resume_from_step": split_step,
        "config": config,
    }
    write_json(path, state)
    try:
        judge = Judge(config, root)
        judge_before = fingerprint(judge.model.parameters())
        runner = DreamRunner(config, lock, folder, judge, manifest["briefs"])
        validation = runner.train(
            data["train"],
            "D",
            manifest["seed"],
            final_step,
            folder,
            run_id,
            resume=source / "checkpoints" / f"step-{split_step:06d}",
            sample_order=manifest.get("training_order"),
        )
        proof = compare_checkpoints(
            source / "checkpoints" / f"step-{final_step:06d}", runner.checkpoint_path
        )
        proof["base_unchanged"] = validation["base_unchanged"]
        proof["judge_unchanged"] = fingerprint(judge.model.parameters()) == judge_before
        prompt = runner.encode(runner.render_prompt(data["validation"][0], "D"))
        proof["post_reload_logits_finite"] = bool(
            mx.all(mx.isfinite(runner.model(mx.array([prompt])))).item()
        )
        passed = all(proof.values())
        proof.update(
            status="passed" if passed else "failed",
            source_identity=run_id,
            from_step=split_step,
            to_step=final_step,
            extra_updates_executed=final_step - split_step,
            limitation="Prueba en este equipo y protocolo; no garantiza otros kernels.",
        )
        write_json(source / "continuity-validation.json", proof)
        write_json(folder / "verification.json", proof)
        state["status"] = "complete" if passed else "failed"
        write_json(path, state)
        print("Continuidad de sueños:", proof, flush=True)
        if not passed:
            raise RuntimeError("La reanudación de sueños difiere de la corrida original.")
    except BaseException:
        state["status"] = "failed"
        write_json(path, state)
        write_json(folder / "error.json", {"traceback": traceback.format_exc()})
        raise
