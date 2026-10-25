"""Diagnóstico acotado de continuación del checkpoint real, separado del experimento."""

import traceback

from .backend import Runner, fingerprint
from .config import freeze_model, identity
from .data import freeze
from .storage import read_json, write_json
from .workflow import effective_config


def verify_continuity(config, root):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    manifest = read_json(root / "smoke" / "run.json")
    if manifest["status"] != "complete":
        raise ValueError("El smoke debe estar completo antes de verificar continuidad.")
    config = effective_config(config, root)
    lock = freeze_model(config, root)
    rows, data_hash = freeze(config, root)
    run_id = identity(config, lock, data_hash, "A", manifest["seed"])
    if run_id != manifest["identity"]:
        raise ValueError("Cambió el código o protocolo respecto del smoke original.")
    folder = root / "resume-verification"
    if (folder / "run.json").exists():
        raise ValueError("La verificación real ya tiene artefactos; no se sobrescriben.")
    final_step = config["smoke"]["steps"]
    split_step = min(2, final_step - 1)
    if split_step < 1:
        raise ValueError("La prueba de continuidad requiere al menos dos pasos en el smoke.")
    source = root / "smoke" / "checkpoints"
    original_final = source / f"step-{final_step:06d}"
    write_json(
        folder / "run.json",
        {
            "kind": "diagnostic",
            "status": "running",
            "condition": "A",
            "seed": manifest["seed"],
            "identity": run_id,
            "steps_requested": final_step,
            "resume_from_step": split_step,
            "original_run": "smoke",
            "config": config,
        },
    )
    try:
        runner = Runner(config, lock, folder)
        runner.train(
            rows["train"],
            "A",
            manifest["seed"],
            final_step,
            folder,
            run_id,
            resume=source / f"step-{split_step:06d}",
        )
        expected_hash = read_json(root / "smoke" / "reload-validation.json")["adapter_hash"]
        original_optimizer = tree_unflatten(
            list(mx.load(str(original_final / "optimizer.safetensors")).items())
        )
        expected_random = mx.load(str(original_final / "mlx_random.safetensors"))["0"]
        actual_hash = fingerprint(runner.model.trainable_parameters())
        proof = {
            "source_identity": run_id,
            "from_step": split_step,
            "to_step": final_step,
            "extra_updates_executed": final_step - split_step,
            "same_adapters": actual_hash == expected_hash,
            "same_optimizer": fingerprint(runner.optimizer.state)
            == fingerprint(original_optimizer),
            "same_mlx_random": bool(mx.array_equal(mx.random.state[0], expected_random).item()),
            "expected_adapter_hash": expected_hash,
            "actual_adapter_hash": actual_hash,
            "limitation": "Comprobación en este equipo, versión y protocolo; no garantiza otros kernels.",
        }
        from .prompts import render

        prompt = runner.encode(render(rows["validation"][0]["problem"], "A"))
        logits = runner.model(mx.array([prompt]))
        proof["post_reload_logits_finite"] = bool(mx.all(mx.isfinite(logits)).item())
        proof["status"] = (
            "passed"
            if all(
                proof[k]
                for k in [
                    "same_adapters",
                    "same_optimizer",
                    "same_mlx_random",
                    "post_reload_logits_finite",
                ]
            )
            else "failed"
        )
        write_json(root / "smoke" / "continuity-validation.json", proof)
        write_json(folder / "verification.json", proof)
        state = read_json(folder / "run.json")
        state["status"] = "complete" if proof["status"] == "passed" else "failed"
        write_json(folder / "run.json", state)
        print("Continuidad real:", proof, flush=True)
        if proof["status"] != "passed":
            raise RuntimeError(
                "La continuación real difiere; revisar la prueba antes del piloto largo."
            )
    except BaseException:
        write_json(folder / "error.json", {"traceback": traceback.format_exc()})
        raise
