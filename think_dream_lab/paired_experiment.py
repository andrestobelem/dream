"""Entrenamiento con consigna natural y recompensa contra su referencia reservada."""

import statistics
import time

import numpy as np

from .backend import fingerprint, seed_all
from .config import freeze_model, identity
from .dream_task import extract_blocks
from .dreambank_task import render_prompt
from .embedding_reward import Encoder, unit_vectors
from .embedding_runner import EmbeddingRunner
from .storage import append_jsonl, digest, read_json, read_jsonl, write_json
from .workflow import latest_checkpoint, reconcile_logs, reload_check


def load_pack(config):
    from pathlib import Path

    pack = read_json(Path(config["training_pack"]["path"]))
    if (
        digest(pack) != config["training_pack"]["hash"]
        or digest(pack["rows"]) != pack["dataset_hash"]
    ):
        raise ValueError("El lote no coincide con el aprobado.")
    rows = pack["rows"]
    if {s: len(v) for s, v in rows.items()} != {"train": 8, "validation": 4}:
        raise ValueError(
            "El ensayo aprobado tiene ocho relatos de entrenamiento y cuatro reservados."
        )
    groups = [{r["author_group"] for r in rows[s]} for s in ("train", "validation")]
    if groups[0] & groups[1]:
        raise ValueError("Una persona cruza entrenamiento y validación.")
    ids = [r["problem_id"] for values in rows.values() for r in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Hay referencias repetidas en el lote.")
    for split, values in rows.items():
        for row in values:
            prompt = render_prompt(
                row["seeds"], response_prefix="<dream>\n", target_words=row["target_words"]
            )
            if (
                row["split"] != split
                or row["reference_hash"] != digest(row["reference"])
                or row["target_words"] != len(row["reference"].split())
                or prompt != row["prompt"]
                or row["reference"] in prompt
            ):
                raise ValueError("Cambió una referencia, consigna o longitud del lote aprobado.")
    return pack


def actor_rows(rows):
    return [
        {k: r[k] for k in ("problem_id", "author_group", "split", "seeds", "target_words")}
        for r in rows
    ]


class PairedReward:
    def __init__(self, rows, encoder):
        self.encoder = encoder
        self.rows = {r["problem_id"]: r for values in rows.values() for r in values}
        self.ids = list(self.rows)
        vectors = unit_vectors(encoder.encode([self.rows[k]["reference"] for k in self.ids]))
        self.vectors = dict(zip(self.ids, vectors))
        self.identity = digest(
            [
                "paired-reference-cosine-v1",
                encoder.identity,
                [(k, self.rows[k]["reference_hash"]) for k in self.ids],
                vectors.tolist(),
            ]
        )

    def score(self, text, problem_id):
        row = self.rows[problem_id]
        blocks = extract_blocks(text, "D")
        words = len(blocks["answer"].split())
        cosine = None
        if blocks["format"]:
            vector = self.encoder.encode([blocks["answer"]])[0]
            cosine = float(np.clip(vector @ self.vectors[problem_id], -1, 1))
        return {
            "total": (1 + cosine) / 2 if cosine is not None else 0.0,
            "cosine": cosine,
            "format": blocks["format"],
            "reference_id": problem_id,
            "reference_split": row["split"],
            "reward_identity": self.identity,
            "answer_words": words,
            "target_words": row["target_words"],
            "relative_word_error": abs(words - row["target_words"]) / row["target_words"],
            "score_status": "scored" if cosine is not None else "invalid_format",
        }


class PairedRunner(EmbeddingRunner):
    def completion_reward(self, text, row, condition):
        if condition != "D":
            raise ValueError("Sólo se entrena Dream.")
        prefix = self.config["generation"]["response_prefix"]
        return {
            **self.embedding_reward.score(prefix + text, row["problem_id"]),
            "response_prefix_in_prompt": prefix,
            "scored_response": prefix + text,
        }

    def train(self, rows, *args, **kwargs):
        if any(r["split"] != "train" for r in rows):
            raise ValueError("No se actualizan pesos con relatos de validación.")
        return super().train(rows, *args, **kwargs)


def metrics(records):
    valid = [r for r in records if r["reward"]["format"]]
    return {
        "n": len(records),
        "n_valid": len(valid),
        "format_rate": len(valid) / len(records) if records else None,
        "reward_mean_all": statistics.mean(r["reward"]["total"] for r in records)
        if records
        else None,
        "source_cosine_valid_mean": statistics.mean(r["reward"]["cosine"] for r in valid)
        if valid
        else None,
        "answer_words_valid_mean": statistics.mean(r["reward"]["answer_words"] for r in valid)
        if valid
        else None,
        "relative_word_error_valid_mean": statistics.mean(
            r["reward"]["relative_word_error"] for r in valid
        )
        if valid
        else None,
        "truncations": sum(r["truncated"] for r in records),
    }


def evaluate(runner, rows, folder, phase, seed, count):
    path = folder / "responses.jsonl"
    previous = [r for r in read_jsonl(path) if r["phase"] == phase]
    for row in rows:
        done = {r["candidate"] for r in previous if r["problem_id"] == row["problem_id"]}
        if len(done) == count:
            continue
        sample_seed = int(digest([seed, row["problem_id"]])[:8], 16)
        seed_all(sample_seed)
        _, records = runner.generate(row, "D", count=count)
        for record in records:
            if record["candidate"] in done:
                continue
            record.update(
                phase=phase,
                evaluation_seed=sample_seed,
                step=0 if phase == "before" else runner.config["smoke"]["steps"],
            )
            append_jsonl(path, record)
            previous.append(record)
        print(f"{phase}: {len(previous)}/{len(rows) * count}, {metrics(records)}", flush=True)
    result = metrics(previous)
    write_json(folder / f"evaluation-{phase}.json", result)
    return result


def run(config, root, resume=False):
    pack = load_pack(config)
    folder = root / "dream/paired"
    target = folder / "run.json"
    encoder = Encoder(config["embeddings"], root)
    reward = PairedReward(pack["rows"], encoder)
    train, validation = (actor_rows(pack["rows"][s]) for s in ("train", "validation"))
    lock = freeze_model(config, root / "dream")
    run_id = identity(
        config, lock, digest([pack["dataset_hash"], reward.identity]), "D", config["seeds"][0]
    )
    if target.exists():
        old = read_json(target)
        if old["identity"] != run_id:
            raise ValueError("El ensayo existente tiene otra identidad.")
        if old["status"] == "complete":
            return old
        if not resume:
            raise ValueError("Hay un ensayo incompleto; usá --resume.")
    manifest = {
        "status": "running",
        "identity": run_id,
        "config": config,
        "dataset_hash": pack["dataset_hash"],
        "reward_identity": reward.identity,
        "steps_requested": config["smoke"]["steps"],
        "started_at": time.time(),
        "reference_in_prompt": False,
        "validation_used_for_updates": False,
    }
    write_json(target, manifest)
    write_json(folder / "frozen-training-pack.json", pack)
    try:
        runner = PairedRunner(config, lock, folder, reward)
        count = config["evaluation"]["samples_per_problem"]
        before_hash = fingerprint(runner.model.parameters())
        before = evaluate(runner, validation, folder, "before", config["evaluation"]["seed"], count)
        if fingerprint(runner.model.parameters()) != before_hash:
            raise RuntimeError("La evaluación inicial modificó los pesos base.")
        reconcile_logs(folder)
        result = runner.train(
            train,
            "D",
            config["seeds"][0],
            config["smoke"]["steps"],
            folder,
            run_id,
            resume=latest_checkpoint(folder) if resume else None,
        )
        reload_check(runner, folder, runner.checkpoint_path)
        after = evaluate(runner, validation, folder, "after", config["evaluation"]["seed"], count)
        manifest.update(
            status="complete",
            completed_at=time.time(),
            before=before,
            after=after,
            training_validation=result,
        )
        write_json(target, manifest)
        print("Ensayo pareado completo:", before, after, flush=True)
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            checkpoint=str(latest_checkpoint(folder)),
        )
        write_json(target, manifest)
        raise
