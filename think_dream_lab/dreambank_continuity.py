"""Repetir desde el checkpoint intermedio y comparar el estado final real."""

from .config import freeze_model, identity
from .dream_verification import compare_checkpoints
from .dreambank_workflow import load_reward
from .embedding_runner import EmbeddingRunner
from .storage import digest, read_json, write_json


def verify_continuity(config, root):
    source = root / "dream/smoke"
    manifest = read_json(source / "run.json")
    if manifest["status"] != "complete":
        raise ValueError("Primero completá el piloto DreamBank.")
    _, corpus, reward = load_reward(config, root)
    lock = freeze_model(config, root / "dream", offline=True)
    run_id = identity(
        config,
        lock,
        digest(
            [corpus["hash"], reward.identity, manifest["actor_train"], manifest["actor_validation"]]
        ),
        "D",
        config["seeds"][0],
    )
    if run_id != manifest["identity"]:
        raise ValueError("El protocolo cambió; no se compara con checkpoints incompatibles.")
    final = manifest["steps_requested"]
    intermediate = final - 1
    if intermediate < 1:
        raise ValueError("La continuidad requiere al menos dos pasos.")
    target = source / "continuity-validation.json"
    if target.exists() and read_json(target).get("status") == "passed":
        return read_json(target)
    folder = root / "dream/continuity"
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Hay una verificación previa incompleta; sus artefactos se conservan.")
    write_json(
        folder / "run.json",
        {"status": "running", "identity": run_id, "from_step": intermediate, "to_step": final},
    )
    try:
        runner = EmbeddingRunner(config, lock, folder, reward)
        validation = runner.train(
            manifest["actor_train"],
            "D",
            config["seeds"][0],
            final,
            folder,
            run_id,
            resume=source / "checkpoints" / f"step-{intermediate:06d}",
        )
        checks = compare_checkpoints(
            source / "checkpoints" / f"step-{final:06d}", runner.checkpoint_path
        )
        checks["base_unchanged"] = validation["base_unchanged"]
        result = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "identity": run_id,
            "from_step": intermediate,
            "to_step": final,
            "extra_updates_executed": 1,
            "limitation": "Continuidad comprobada en este equipo y protocolo.",
        }
        write_json(target, result)
        write_json(folder / "run.json", result)
        if result["status"] != "passed":
            raise RuntimeError("La reanudación produjo un estado final distinto.")
        print("Continuidad DreamBank:", result, flush=True)
        return result
    except BaseException as error:
        write_json(folder / "error.json", {"error": repr(error)})
        raise
