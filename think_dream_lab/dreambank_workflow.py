"""Etapas del protocolo DreamBank, separadas del juez histórico."""

from .dreambank import SPLITS, prepare
from .seed_extractor import SeedExtractor
from .storage import digest, read_json, write_json


def smoke(config, root, resume=False):
    import gc
    import time

    from .config import freeze_model, identity
    from .embedding_runner import EmbeddingRunner
    from .workflow import latest_checkpoint, reconcile_logs, reload_check

    rows, corpus, reward = load_reward(config, root)
    steps = config["smoke"]["steps"]
    n = config["smoke"]["eval_problems"]
    extractor = SeedExtractor(config, root)
    train = [extractor.extract(r) for r in rows["train"][:steps]]
    validation = [extractor.extract(r) for r in rows["validation"][:n]]
    del extractor
    gc.collect()
    lock = freeze_model(config, root / "dream")
    seed = config["seeds"][0]
    run_id = identity(
        config, lock, digest([corpus["hash"], reward.identity, train, validation]), "D", seed
    )
    folder = root / "dream/smoke"
    target = folder / "run.json"
    if target.exists():
        old = read_json(target)
        if old["identity"] != run_id:
            raise ValueError(
                "Cambió el protocolo del piloto; no se mezcla con el checkpoint anterior."
            )
        if old["status"] == "complete":
            return old
        if not resume:
            raise ValueError("Hay un piloto incompleto; usá --resume.")
    manifest = {
        "identity": run_id,
        "protocol": config["protocol"],
        "status": "running",
        "config": config,
        "model": lock,
        "steps_requested": steps,
        "bank_identity": reward.identity,
        "corpus_hash": corpus["hash"],
        "actor_train": train,
        "actor_validation": validation,
        "credit_assignment": "entire_sequence",
        "scored_block": "answer",
    }
    write_json(target, manifest)
    try:
        runner = EmbeddingRunner(config, lock, folder, reward)
        eval_seed = config["evaluation"]["seed"]
        checkpoint = latest_checkpoint(folder) if resume else None
        runner.evaluate(validation, "D", folder, "before", eval_seed)
        reconcile_logs(folder)
        result = runner.train(train, "D", seed, steps, folder, run_id, resume=checkpoint)
        reload_check(runner, folder, runner.checkpoint_path)
        runner.evaluate(validation, "D", folder, "after", eval_seed, step=steps)
        manifest.update(status="complete", training_validation=result, completed_at=time.time())
        write_json(target, manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            checkpoint=str(latest_checkpoint(folder)),
        )
        write_json(target, manifest)
        raise


def load_reward(config, root):
    from .embedding_reward import DreamBankReward, Encoder

    rows, manifest = prepare(config, root)
    reward = DreamBankReward(rows["train"], Encoder(config["embeddings"], root))
    proof = root / "dream/embeddings/diagnostic.json"
    if not proof.exists():
        raise ValueError("Primero ejecutá embedding-check para revisar la recompensa.")
    result = read_json(proof)
    if (
        result["status"] != "operational_checked"
        or result["bank_identity"] != reward.identity
        or result["corpus_hash"] != manifest["hash"]
    ):
        raise ValueError("El diagnóstico corresponde a otro banco o no pasó sus comprobaciones.")
    return rows, manifest, reward


def check_base(config, root, limit):
    from .backend import fingerprint, seed_all
    from .config import freeze_model
    from .dreambank_task import render_prompt
    from .embedding_runner import EmbeddingRunner

    rows, corpus, reward = load_reward(config, root)
    if not 1 <= limit <= len(rows["validation"]):
        raise ValueError("El límite debe estar dentro del conjunto de validación.")
    extractor = SeedExtractor(config, root)
    actor_rows = [extractor.extract(row) for row in rows["validation"][:limit]]
    probe_id = digest(
        [
            config["model"],
            config["generation"],
            reward.identity,
            [render_prompt(r["seeds"]) for r in actor_rows],
        ]
    )
    folder = root / "dream/base-check" / probe_id[:16]
    runner = EmbeddingRunner(config, freeze_model(config, root / "dream"), folder, reward)
    before = fingerprint(runner.model.parameters())
    seed_all(config["evaluation"]["seed"])
    summary = runner.evaluate(actor_rows, "D", folder, "base", config["evaluation"]["seed"])
    result = {
        "summary": summary,
        "base_unchanged": before == fingerprint(runner.model.parameters()),
        "training_updates": 0,
        "prompt_example_used": False,
        "bank_identity": reward.identity,
        "corpus_hash": corpus["hash"],
    }
    write_json(folder / "validation.json", result)
    write_json(root / "dream/base-check/latest.json", {"directory": str(folder), **result})
    if not result["base_unchanged"]:
        raise RuntimeError("La prueba modificó el modelo base.")
    return result


def embedding_check(config, root, limit):
    import statistics

    from .embedding_reward import DreamBankReward, Encoder

    rows, corpus = prepare(config, root)
    if not 1 <= limit <= min(len(rows["train"]), len(rows["validation"])):
        raise ValueError("La muestra del diagnóstico supera los datos disponibles.")
    encoder = Encoder(config["embeddings"], root)
    reward = DreamBankReward(rows["train"], encoder)
    records = []
    for variant, split in [
        ("exact_training_reference", "train"),
        ("reserved_human_report", "validation"),
        ("topic_word_list", "train"),
    ]:
        for row in rows[split][:limit]:
            text = row["reference"]
            if variant == "topic_word_list":
                # Control temático reproducible, sin presentarlo como una etiqueta humana.
                text = ", ".join(sorted(set(text.lower().split())))
            records.append(
                {"variant": variant, "problem_id": row["problem_id"], **reward.answer_score(text)}
            )
        print(f"Embeddings: {variant}, {limit} relatos", flush=True)
    summaries = {}
    for variant in sorted({r["variant"] for r in records}):
        subset = [r for r in records if r["variant"] == variant]
        summaries[variant] = {
            key: statistics.mean(r[key] for r in subset)
            for key in ("cosine", "total", "top5_mean_cosine_diagnostic")
        }
    checks = {
        "exact_reference_receives_maximum": all(
            r["total"] >= 1 - 1e-5 for r in records if r["variant"] == "exact_training_reference"
        ),
        "bank_train_only": all(r["split"] == "train" for r in rows["train"]),
        "reserved_reference_ids_disjoint": not set(reward.ids)
        & {r["problem_id"] for r in rows["validation"] + rows["test"]},
    }
    result = {
        "formula": "nearest-dream-cosine-v1",
        "status": "operational_checked" if all(checks.values()) else "failed",
        "checks": checks,
        "encoder": config["embeddings"],
        "encoder_identity": encoder.identity,
        "bank_identity": reward.identity,
        "corpus_hash": corpus["hash"],
        "bank_size": len(reward.ids),
        "bank_reference_ids": reward.ids,
        "summaries": summaries,
        "records": records,
        "interpretation": "Cercanía semántica; listas temáticas pueden puntuar alto. "
        "No valida oniricidad ni experiencia subjetiva. "
        "No se penalizan coincidencias ni similitud alta.",
    }
    write_json(root / "dream/embeddings/diagnostic.json", result)
    if result["status"] == "failed":
        raise RuntimeError("Falló una invariante operativa de la recompensa por embeddings.")
    return result


def extract_seeds(config, root, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError("--limit debe ser un entero positivo por partición.")
    rows, corpus_manifest = prepare(config, root)
    if any(limit > len(values) for values in rows.values()):
        raise ValueError("La muestra supera el tamaño de alguna partición.")
    extractor = SeedExtractor(config, root)
    selected = {}
    for split in SPLITS:
        selected[split] = []
        for row in rows[split][:limit]:
            selected[split].append(extractor.extract(row))
            print(f"Semillas {split}: {len(selected[split])}/{limit}", flush=True)
    manifest = {
        "corpus_hash": corpus_manifest["hash"],
        "extractor_identity": extractor.identity,
        "actor_rows": selected,
        "hash": digest(selected),
        "counts": {s: len(v) for s, v in selected.items()},
        "generator_sees_reference": False,
    }
    write_json(root / "dream/seeds/manifest.json", manifest)
    return manifest
