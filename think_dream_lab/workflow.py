"""Etapas del experimento y sus manifiestos de procedencia."""

import copy
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback

from .backend import Runner, audit_tokenizer, fingerprint, seed_all
from .config import freeze_model, identity, implementation_manifest
from .data import freeze, generate
from .storage import append_jsonl, read_json, read_jsonl, write_json
from .system import diagnose


def effective_config(config, root):
    config = copy.deepcopy(config)
    selection = root / "data-choice.json"
    if selection.exists():
        selected = read_json(selection)
        if selected["requested_data"] != config["data"]:
            raise ValueError("La configuración de datos cambió respecto de la calibración.")
        config["data"]["difficulty"] = selected["difficulty"]
    return config


def doctor(config, root):
    root.mkdir(parents=True, exist_ok=True)
    diagnostic = diagnose(root)
    diagnostic["implementation"] = implementation_manifest()
    write_json(root / "doctor.json", diagnostic)
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2), flush=True)
    if diagnostic["architecture"] != "arm64" or not diagnostic["metal_available"]:
        raise RuntimeError("Se requiere Apple Silicon con Metal disponible.")
    if diagnostic["disk_free_bytes"] < config["runtime"]["min_free_disk_gib"] * 1024**3:
        raise RuntimeError("Espacio libre insuficiente según la configuración.")
    lock = freeze_model(config, root)
    audit = audit_tokenizer(lock, root)
    print("Tokenizer auditado. Asimetría literal:", audit["literal_asymmetry"], flush=True)
    return diagnostic


def model_provenance(runner, root):
    target = root / "model-files.json"
    if target.exists():
        return
    files = {}
    for path in sorted(runner.path.iterdir()):
        if path.is_file() and path.suffix == ".safetensors":
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    hasher.update(chunk)
            files[path.name] = {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}
    write_json(target, {"model": runner.lock, "files": files})


def calibrate(runner, config, root):
    selection = root / "data-choice.json"
    if selection.exists():
        return effective_config(config, root)
    requested = copy.deepcopy(config["data"])
    generated = generate(config)
    summaries = []
    n = config["smoke"]["eval_problems"]
    for condition in config["conditions"]:
        summaries.append(
            runner.evaluate(
                generated["validation"][:n],
                condition,
                root / "calibration",
                f"calibration_{requested['difficulty']}",
                config["evaluation"]["seed"],
            )
        )
    accuracies = [s["accuracy"] for s in summaries]
    selected = requested["difficulty"]
    reason = "Punto de partida no extremo en al menos una condición."
    if all(a == 0 for a in accuracies) or all(a == 1 for a in accuracies):
        candidate = "easy" if all(a == 0 for a in accuracies) else "hard"
        if candidate != selected:
            proposal = copy.deepcopy(config)
            proposal["data"]["difficulty"] = candidate
            alternatives = generate(proposal)
            candidates = [
                runner.evaluate(
                    alternatives["validation"][:n],
                    c,
                    root / "calibration",
                    f"calibration_{candidate}",
                    config["evaluation"]["seed"],
                )
                for c in config["conditions"]
            ]
            # Una única calibración predefinida, sin mirar test ni entrenamiento.
            if any(0 < s["accuracy"] < 1 for s in candidates):
                selected = candidate
                reason = "La alternativa predefinida produjo acierto no extremo en validación."
            else:
                reason = "Ambas dificultades dieron extremos; se conserva la original y se declara falta de calibración útil."
    write_json(
        selection,
        {
            "requested_data": requested,
            "difficulty": selected,
            "reason": reason,
            "initial_accuracy": accuracies,
            "evaluation_split": "validation",
            "test_used": False,
        },
    )
    return effective_config(config, root)


def reconcile_logs(folder):
    """Recuperar logs desde checkpoints completos tras interrupciones."""
    metrics = {r["step"]: r for r in read_jsonl(folder / "steps.jsonl")}
    responses = read_jsonl(folder / "responses.jsonl")
    completed = {(r["step"], r["candidate"]) for r in responses if r["phase"] == "rollout"}
    for path in sorted((folder / "checkpoints").glob("step-*/state.json")):
        state = read_json(path)
        if state["step"] not in metrics:
            append_jsonl(folder / "steps.jsonl", state["extra"]["metric"])
        for record in state["extra"]["responses"]:
            if (record["step"], record["candidate"]) not in completed:
                append_jsonl(folder / "responses.jsonl", record)


def latest_checkpoint(folder):
    latest = folder / "checkpoints" / "latest.json"
    return folder / "checkpoints" / read_json(latest)["directory"] if latest.exists() else None


def reload_check(runner, folder, checkpoint_path):
    import mlx.core as mx

    before = fingerprint(runner.model.trainable_parameters())
    runner.model.load_weights(str(checkpoint_path / "adapters.safetensors"), strict=False)
    mx.eval(runner.model.parameters())
    after = fingerprint(runner.model.trainable_parameters())
    if before != after:
        raise RuntimeError("La recarga no reproduce los pesos guardados.")
    write_json(
        folder / "reload-validation.json",
        {"same_weights": True, "adapter_hash": after, "checkpoint": str(checkpoint_path)},
    )


def smoke(config, root, resume=False):
    folder = root / "smoke"
    manifest = folder / "run.json"
    if manifest.exists() and not resume:
        raise ValueError("Ya existe un smoke. Usá --resume para continuar sin sobrescribir.")
    audit = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], check=False, capture_output=True, text=True
    )
    write_json(
        root / "tests.json",
        {"returncode": audit.returncode, "stdout": audit.stdout, "stderr": audit.stderr},
    )
    print(audit.stdout, flush=True)
    if audit.returncode:
        raise RuntimeError("Las pruebas fallaron; no se carga el modelo.")
    doctor(config, root)
    lock = freeze_model(config, root)
    runner = Runner(config, lock, root)
    model_provenance(runner, root)
    config = calibrate(runner, config, root)
    runner.config = config
    rows, data_hash = freeze(config, root)
    seed = config["seeds"][0]
    run_id = identity(config, lock, data_hash, "A", seed)
    if manifest.exists() and read_json(manifest)["identity"] != run_id:
        raise ValueError("La identidad del smoke cambió.")
    write_json(
        manifest,
        {
            "identity": run_id,
            "config": config,
            "model": lock,
            "data_hash": data_hash,
            "seed": seed,
            "condition": "A",
            "status": "running",
            "kind": "smoke",
            "implementation": implementation_manifest(),
        },
    )
    n, eval_seed = config["smoke"]["eval_problems"], config["evaluation"]["seed"]
    validation = rows["validation"][:n]
    # Los pesos todavía son el modelo base, sin adaptadores.
    for condition in config["conditions"]:
        runner.evaluate(validation, condition, folder, "before", eval_seed)
    runner.evaluate(validation, "neutral", folder, "before", eval_seed)
    if read_json(root / "tokenizer-audit.json")["literal_asymmetry"]:
        for condition in ["ordinary_A", "ordinary_D"]:
            runner.evaluate(validation, condition, folder, "tokenizer_diagnostic", eval_seed)
    reconcile_logs(folder)
    runner.train(
        rows["train"],
        "A",
        seed,
        config["smoke"]["steps"],
        folder,
        run_id,
        resume=latest_checkpoint(folder) if resume else None,
    )
    reload_check(runner, folder, runner.checkpoint_path)
    for condition in ["A", "neutral"]:
        runner.evaluate(
            validation, condition, folder, "after", eval_seed, step=config["smoke"]["steps"]
        )
    run = read_json(manifest)
    run["status"] = "complete"
    run["completed_at"] = time.time()
    write_json(manifest, run)
    print(
        "Smoke terminado. Verificá el informe: un smoke no demuestra una mejora científica.",
        flush=True,
    )


def run_experiment(config, root, condition, seed, steps, resume=False):
    import mlx.core as mx

    config = effective_config(config, root)
    if not (root / "smoke" / "run.json").exists():
        raise ValueError("Ejecutá primero el smoke y revisá su informe.")
    if read_json(root / "smoke" / "run.json")["status"] != "complete":
        raise ValueError("El smoke no está completo; todavía no se habilita el experimento largo.")
    validation = read_json(root / "smoke" / "training-validation.json")
    if not validation["base_unchanged"] or not validation["adapters_changed"]:
        raise ValueError("El smoke todavía no comprobó una actualización útil de adaptadores.")
    lock = freeze_model(config, root)
    rows, data_hash = freeze(config, root)
    folder = root / "runs" / f"{condition}-seed-{seed}"
    run_id = identity(config, lock, data_hash, condition, seed)
    manifest = folder / "run.json"
    if manifest.exists():
        existing = read_json(manifest)
        if existing["identity"] != run_id or existing["steps_requested"] != steps:
            raise ValueError("La corrida existente tiene otra identidad o cantidad de pasos.")
        if existing["status"] == "complete":
            print(f"Corrida {condition}/{seed} ya completa; se omite.", flush=True)
            return
        if not resume:
            raise ValueError("Corrida existente incompleta: usá --resume.")
    write_json(
        manifest,
        {
            "identity": run_id,
            "config": config,
            "model": lock,
            "data_hash": data_hash,
            "seed": seed,
            "condition": condition,
            "steps_requested": steps,
            "status": "running",
            "kind": "experiment",
            "implementation": implementation_manifest(),
        },
    )
    seed_all(seed)
    runner = Runner(config, lock, folder)
    eval_seed = config["evaluation"]["seed"]
    # Test se usa sólo en la comparación ya congelada, nunca en calibración.
    for prompt in [condition, "neutral"]:
        runner.evaluate(rows["test"], prompt, folder, "before", eval_seed)
    reconcile_logs(folder)
    runner.train(
        rows["train"],
        condition,
        seed,
        steps,
        folder,
        run_id,
        resume=latest_checkpoint(folder) if resume else None,
    )
    reload_check(runner, folder, runner.checkpoint_path)
    for prompt in [condition, "neutral"]:
        runner.evaluate(rows["test"], prompt, folder, "after", eval_seed, step=steps)
    run = read_json(manifest)
    run.update(status="complete", completed_at=time.time())
    write_json(manifest, run)
    del runner
    gc.collect()
    mx.clear_cache()


def worker(action, config, root, **kwargs):
    root.mkdir(parents=True, exist_ok=True)
    try:
        if action == "doctor":
            doctor(config, root)
        elif action == "smoke":
            smoke(config, root, resume=kwargs.get("resume", False))
        elif action == "experiment":
            run_experiment(
                config,
                root,
                kwargs["condition"],
                kwargs["seed"],
                kwargs["steps"],
                kwargs.get("resume", False),
            )
        else:
            raise ValueError(f"Etapa desconocida: {action}")
    except BaseException as error:
        append_jsonl(
            root / "errors.jsonl",
            {
                "time": time.time(),
                "action": action,
                "condition": kwargs.get("condition"),
                "seed": kwargs.get("seed"),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
