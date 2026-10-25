"""Experimento de una pasada por el lote aprobado, con bifurcación trazable del origen."""

import time
from pathlib import Path

from .backend import fingerprint, frozen_fingerprint
from .config import freeze_model, identity
from .embedding_reward import Encoder
from .holdout_evaluation import sha256, validate_brief
from .paired_experiment import PairedReward, PairedRunner, actor_rows
from .storage import digest, read_json, read_jsonl, write_json
from .workflow import latest_checkpoint, reconcile_logs, reload_check


def load_expanded_pack(config):
    from .dreambank_task import render_prompt

    spec = config["expanded_experiment"]
    pack = read_json(Path(config["training_pack"]["path"]))
    if (
        digest(pack) != config["training_pack"]["hash"]
        or digest(pack["rows"]) != spec["approved_dataset_hash"]
        or pack["dataset_hash"] != spec["approved_dataset_hash"]
    ):
        raise ValueError("El lote no coincide con el aprobado por el usuario.")
    if pack["training_updates"] != 0 or pack["retained_original_reports"] != 8:
        raise ValueError("Se requiere el lote revisado, aún no entrenado.")
    train, validation = pack["rows"]["train"], pack["rows"]["validation"]
    if (
        len(train) != 128
        or len(validation) != 24
        or len({r["problem_id"] for r in train + validation}) != 152
    ):
        raise ValueError("Se requieren 128 relatos de entrenamiento y 24 de validación únicos.")
    corpus = Path(spec["original_corpus"])
    reserved = read_jsonl(corpus / "validation.jsonl") + read_jsonl(corpus / "test.jsonl")
    if {r["author_group"] for r in train} & {r["author_group"] for r in reserved} or {
        r["problem_id"] for r in train
    } & {r["problem_id"] for r in reserved}:
        raise ValueError("Un reservado entró en el entrenamiento.")
    for split, rows in pack["rows"].items():
        for row in rows:
            validate_brief({"brief": row["seeds"]})
            prompt = render_prompt(
                row["seeds"],
                response_prefix=config["generation"]["response_prefix"],
                target_words=row["target_words"],
            )
            if (
                row["split"] != split
                or digest(row["reference"]) != row["reference_hash"]
                or row["target_words"] != len(row["reference"].split())
                or prompt != row["prompt"]
                or row["reference"] in prompt
            ):
                raise ValueError("Cambió una referencia o un prompt del lote aprobado.")
    if (
        spec["updates"] != len(train)
        or spec["epochs"] != 1
        or spec["target_step"] - spec["source_step"] != len(train)
    ):
        raise ValueError("El destino debe ser exactamente una pasada por los 128 relatos.")
    return pack


def bootstrap(runner, source_checkpoint, source_identity, new_identity, folder, expected_base):
    import random

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    from . import checkpoint

    runner.prepare_training(runner.config["seeds"][0])
    state = checkpoint.load(source_checkpoint, runner.model, runner.optimizer, source_identity)
    adapter_hash = fingerprint(dict(tree_flatten(runner.model.trainable_parameters())))
    optimizer_hash = fingerprint(dict(tree_flatten(runner.optimizer.state)))
    if (
        adapter_hash != fingerprint(mx.load(str(source_checkpoint / "adapters.safetensors")))
        or optimizer_hash != fingerprint(mx.load(str(source_checkpoint / "optimizer.safetensors")))
        or frozen_fingerprint(runner.model) != expected_base
    ):
        raise ValueError("No se restauraron exactamente el modelo y el optimizador del origen.")
    rng = mx.load(str(source_checkpoint / "mlx_random.safetensors"))
    npstate = np.random.get_state()
    numpy_state = [npstate[0], npstate[1].tolist(), *npstate[2:]]
    if (
        not mx.array_equal(mx.random.state[0], rng["0"]).item()
        or digest(random.getstate()) != digest(state["python_random"])
        or digest(numpy_state) != digest(state["numpy_random"])
    ):
        raise ValueError("No se restauró exactamente la aleatoriedad del origen.")
    # La bifurcación recibe identidad nueva; el checkpoint original nunca se reescribe.
    target = checkpoint.save(
        folder / "bootstrap",
        runner.model,
        runner.optimizer,
        step=state["step"],
        run_identity=new_identity,
        extra={**state["extra"], "source_identity": source_identity},
    )
    write_json(
        folder / "bootstrap-validation.json",
        {
            "same_adapters": True,
            "same_optimizer": True,
            "same_random_states": True,
            "base_unchanged": True,
            "source_step": state["step"],
            "source_identity": source_identity,
            "new_identity": new_identity,
            "adapter_hash": adapter_hash,
            "optimizer_hash": optimizer_hash,
            "checkpoint": str(target),
        },
    )
    return target


def run(config, root):
    spec = config["expanded_experiment"]
    pack = load_expanded_pack(config)
    source = read_json(Path(spec["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    if (
        source["status"] != "complete"
        or source["training_validation"]["steps_completed"] != spec["source_step"]
    ):
        raise ValueError("El origen no coincide con el checkpoint predefinido.")
    for key in ["model", "lora", "training", "embeddings", "reward", "seeds"]:
        if config[key] != source["config"][key]:
            raise ValueError("Cambió un hiperparámetro además del lote y el límite predefinido.")
    if {k: v for k, v in config["generation"].items() if k != "max_tokens"} != {
        k: v for k, v in source["config"]["generation"].items() if k != "max_tokens"
    } or config["generation"]["max_tokens"] != 2048:
        raise ValueError("La generación debe conservarse salvo el límite fijado de 2048.")
    before_root = Path(spec["evaluation_before"])
    baseline = read_json(before_root / "dream/holdout/checkpoint/run.json")
    if (
        baseline["status"] != "complete"
        or not baseline["parameters_unchanged"]
        or baseline["checkpoint_sha256"] != sha256(origin / "adapters.safetensors")
        or baseline["config"]["generation"] != config["generation"]
    ):
        raise ValueError("Falta una evaluación inicial completa a 2048 del mismo checkpoint.")
    if (
        read_json(before_root / "dream/holdout/evaluation-pack.json")["rows"]
        != pack["rows"]["validation"]
    ):
        raise ValueError("La validación inicial difiere del lote aprobado.")
    controls = read_json(root / "dream/judge-controls.json")
    control_validation = read_json(root / "dream/judge-controls-validation.json")
    if (
        controls["passed"] < 5
        or controls["rubric_hash"] != spec["judge_rubric_hash"]
        or not control_validation["parameters_unchanged"]
    ):
        raise ValueError("Falta validar el juez independiente antes del entrenamiento.")
    folder = root / "dream/paired"
    target = folder / "run.json"
    reward = PairedReward(pack["rows"], Encoder(config["embeddings"], root))
    lock = freeze_model(config, root / "dream")
    run_id = identity(
        config, lock, digest([pack["dataset_hash"], reward.identity]), "D", config["seeds"][0]
    )
    originals = {
        name: sha256(origin / name)
        for name in [
            "adapters.safetensors",
            "optimizer.safetensors",
            "mlx_random.safetensors",
            "state.json",
        ]
    }
    lineage = {
        "source_run": spec["source_run"],
        "source_identity": source["identity"],
        "source_step": spec["source_step"],
        "target_step": spec["target_step"],
        "source_files": originals,
        "approved_dataset_hash": pack["dataset_hash"],
    }
    if target.exists():
        old = read_json(target)
        if old["identity"] != run_id or old["lineage"] != lineage:
            raise ValueError("Cambió la identidad de la corrida.")
        if old["status"] == "complete":
            return old
    manifest = {
        "status": "running",
        "identity": run_id,
        "config": config,
        "lineage": lineage,
        "dataset_hash": pack["dataset_hash"],
        "reward_identity": reward.identity,
        "started_at": time.time(),
        "reference_in_prompt": False,
        "validation_used_for_updates": False,
        "new_updates_requested": 128,
        "steps_requested": spec["target_step"],
    }
    write_json(target, manifest)
    write_json(folder / "frozen-training-pack.json", pack)
    try:
        runner = PairedRunner(config, lock, folder, reward)
        reconcile_logs(folder)
        resume = latest_checkpoint(folder)
        if resume is None:
            existing = folder / "bootstrap" / f"step-{spec['source_step']:06d}"
            resume = (
                existing
                if existing.exists()
                else bootstrap(
                    runner,
                    origin,
                    source["identity"],
                    run_id,
                    folder,
                    source["training_validation"]["base_after"],
                )
            )
        result = runner.train(
            actor_rows(pack["rows"]["train"]),
            "D",
            config["seeds"][0],
            spec["target_step"],
            folder,
            run_id,
            resume=resume,
        )
        reload_check(runner, folder, runner.checkpoint_path)
        steps = read_jsonl(folder / "steps.jsonl")
        records = read_jsonl(folder / "responses.jsonl")
        expected = list(range(spec["source_step"] + 1, spec["target_step"] + 1))
        if (
            [s["step"] for s in steps] != expected
            or len({s["problem_id"] for s in steps}) != 128
            or len(records) != 512
            or any(r["reward"]["reference_split"] != "train" for r in records)
        ):
            raise ValueError(
                "El recorrido no fue exactamente de 128 relatos y 512 rollouts de entrenamiento."
            )
        if (
            not result["base_unchanged"]
            or result["base_after"] != source["training_validation"]["base_after"]
            or any(sha256(origin / name) != value for name, value in originals.items())
        ):
            raise RuntimeError("Cambió la base o algún archivo del checkpoint original.")
        manifest.update(
            status="complete",
            completed_at=time.time(),
            training_validation=result,
            new_updates_completed=128,
            unique_training_reports=128,
            rollouts=512,
            original_checkpoint_unchanged=True,
        )
        write_json(target, manifest)
        print("Una pasada completa:", result, flush=True)
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
        )
        write_json(target, manifest)
        raise
