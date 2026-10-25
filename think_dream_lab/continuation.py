"""Continuar un entrenamiento congelado sin reiniciar optimizador ni aleatoriedad."""

import time
from pathlib import Path

from .backend import fingerprint
from .config import freeze_model, identity
from .embedding_reward import Encoder
from .holdout_evaluation import sha256
from .paired_experiment import PairedReward, PairedRunner, actor_rows, load_pack
from .storage import digest, read_json, write_json
from .workflow import latest_checkpoint, reconcile_logs, reload_check


def run(config, root, source_path, target_step):
    source = read_json(source_path)
    origin = Path(source["training_validation"]["checkpoint"])
    start = read_json(origin / "state.json")["step"]
    if source["status"] != "complete" or not start < target_step <= start + 16:
        raise ValueError("Se requiere un origen completo y hasta 16 pasos adicionales.")
    if {k: v for k, v in config.items() if k != "runtime"} != {
        k: v for k, v in source["config"].items() if k != "runtime"
    }:
        raise ValueError("La continuación debe conservar la configuración científica original.")
    pack = load_pack(config)
    reward = PairedReward(pack["rows"], Encoder(config["embeddings"], root))
    lock = freeze_model(config, root / "dream")
    run_id = identity(
        config, lock, digest([pack["dataset_hash"], reward.identity]), "D", config["seeds"][0]
    )
    if run_id != source["identity"] or reward.identity != source["reward_identity"]:
        raise ValueError("Cambió la implementación, el lote o la recompensa del entrenamiento.")
    heldout = read_json(Path("results/dream-v7-eval/dream/holdout/evaluation-pack.json"))
    train = pack["rows"]["train"]
    if {r["author_group"] for r in train} & {r["author_group"] for r in heldout["rows"]}:
        raise ValueError("Una persona reservada entró en entrenamiento.")
    folder = root / "dream/paired"
    path = folder / "run.json"
    lineage = {
        "source_run": str(source_path),
        "source_identity": run_id,
        "source_step": start,
        "source_checkpoint_sha256": sha256(origin / "adapters.safetensors"),
        "target_step": target_step,
        "training_pack_hash": config["training_pack"]["hash"],
    }
    if path.exists():
        old = read_json(path)
        if old["lineage"] != lineage:
            raise ValueError("La continuación existente tiene otro origen o destino.")
        if old["status"] == "complete":
            return old
    manifest = {
        "status": "running",
        "identity": run_id,
        "segment_identity": digest(lineage),
        "lineage": lineage,
        "config": config,
        "dataset_hash": pack["dataset_hash"],
        "reward_identity": reward.identity,
        "started_at": time.time(),
        "validation_used_for_updates": False,
        "reference_in_prompt": False,
        "steps_requested": target_step,
    }
    write_json(path, manifest)
    write_json(folder / "frozen-training-pack.json", pack)
    try:
        runner = PairedRunner(config, lock, folder, reward)
        reconcile_logs(folder)
        resume = latest_checkpoint(folder) or origin
        result = runner.train(
            actor_rows(train), "D", config["seeds"][0], target_step, folder, run_id, resume=resume
        )
        reload_check(runner, folder, runner.checkpoint_path)
        if sha256(origin / "adapters.safetensors") != lineage["source_checkpoint_sha256"]:
            raise RuntimeError("Se modificó el checkpoint fuente.")
        if (
            not result["base_unchanged"]
            or result["base_after"] != source["training_validation"]["base_after"]
        ):
            raise RuntimeError("La base cambió en la continuación.")
        manifest.update(
            status="complete",
            completed_at=time.time(),
            training_validation=result,
            additional_steps=target_step - start,
            final_adapters=fingerprint(runner.model.trainable_parameters()),
        )
        write_json(path, manifest)
        print("Continuación completa:", result, flush=True)
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
        )
        write_json(path, manifest)
        raise
