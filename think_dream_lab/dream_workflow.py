"""Flujo del experimento de sueños, independiente de la antigua prueba aritmética."""

import statistics
import time

import numpy as np

from .backend import Runner, fingerprint
from .config import freeze_model, identity
from .corpus import prepare
from .dream_task import combine_scores, extract_blocks, render_dream
from .judge import Judge
from .storage import append_jsonl, digest, read_json, write_json
from .workflow import doctor, latest_checkpoint, reconcile_logs, reload_check


class DreamRunner(Runner):
    def __init__(self, config, lock, folder, judge, briefs):
        if config["model"]["id"] != "HuggingFaceTB/SmolLM2-1.7B":
            raise ValueError(
                "El generador actual requiere SmolLM2 base; Qwen3 queda como antecedente histórico."
            )
        if config["judge"].get("rubric_version") == "dream-v4":
            raise ValueError(
                "El juez de etiquetas cerradas requiere integración y calibración antes de usar el generador."
            )
        self.judge, self.briefs = judge, briefs
        super().__init__(config, lock, folder)

    def render_prompt(self, row, condition):
        return render_dream(self.briefs[row["problem_id"]], condition)

    def stop_sequence(self):
        return "</answer>"

    def completion_reward(self, text, row, condition):
        blocks = extract_blocks(text, condition)
        ratings = self.judge.rate(
            row["reference"],
            blocks["draft"],
            blocks["answer"],
            brief=self.briefs[row["problem_id"]],
        )
        combined = combine_scores(
            ratings, blocks["format"], self.config, neutral=condition == "neutral"
        )
        return {
            **combined,
            "ratings": ratings,
            "reference_id": row["problem_id"],
            "brief_hash": digest(self.briefs[row["problem_id"]]),
            "answer_words": len(blocks["answer"].split()),
            "reference_verbatim_copy": blocks["answer"].strip() == row["reference"].strip(),
        }

    def summarize_records(self, records):
        return summarize(records)


def summarize(records):
    if not records:
        return {"n": 0}
    keys = [
        "dream_likeness",
        "reference_alignment",
        "retelling_fidelity",
        "quality",
        "format",
        "total",
    ]
    valid = [r for r in records if r["reward"]["answer_words"] > 0]
    return {
        "n": len(records),
        "n_valid_answers": len(valid),
        "valid_dream_likeness": statistics.mean(r["reward"]["dream_likeness"] for r in valid)
        if valid
        else None,
        "valid_reference_alignment": statistics.mean(
            r["reward"]["reference_alignment"] for r in valid
        )
        if valid
        else None,
        **(
            {
                "content_compliance": statistics.mean(
                    r["reward"]["content_compliance"] for r in records
                ),
                "valid_content_compliance": statistics.mean(
                    r["reward"]["content_compliance"] for r in valid
                )
                if valid
                else None,
            }
            if all("content_compliance" in r["reward"] for r in records)
            else {}
        ),
        **{k: statistics.mean(r["reward"][k] for r in records) for k in keys},
        "tokens": sum(r["tokens"] for r in records),
        "generation_seconds": sum(r["latency_seconds"] for r in records),
        "truncations": sum(r["truncated"] for r in records),
    }


def prepare_briefs(judge, rows, folder):
    path = folder / "briefs.json"
    stored = read_json(path) if path.exists() else {"judge": judge.spec, "briefs": {}}
    if stored["judge"] != judge.spec:
        raise ValueError("El juez del archivo de consignas cambió.")
    for row in rows:
        pid = row["problem_id"]
        if pid not in stored["briefs"]:
            stored["briefs"][pid] = judge.brief(row)
            write_json(path, stored)
            print(f"  Consigna preparada: {len(stored['briefs'])}", flush=True)
    return stored["briefs"]


def load_judge(config, root, data):
    judge = Judge(config, root)
    judge.calibrate(data["validation"][:3])
    from .judge_checks import verify_reference

    verify_reference(config, root, judge=judge)
    if config["judge"].get("rubric_version") == "dream-v3":
        from .anchors import verify_controls

        controls = verify_controls(judge)
        write_json(root / "dream" / "anchor-calibration.json", controls)
        if not controls["passed"]:
            raise RuntimeError("El juez no pasó los controles por elementos; RL queda bloqueado.")
        from .anchor_holdout import verify_holdout

        holdout = verify_holdout(judge)
        write_json(root / "dream" / "anchor-holdout.json", holdout)
        if not holdout["passed"]:
            raise RuntimeError("El juez no pasó la segunda batería; RL queda bloqueado.")
    return judge


def smoke(config, root, resume=False):
    folder = root / "dream" / "smoke"
    manifest_path = folder / "run.json"
    if manifest_path.exists() and not resume:
        raise ValueError("Ya existe un smoke de sueños; usá --resume para continuar.")
    doctor(config, root / "dream")
    data, corpus_manifest = prepare(config, root)
    judge = load_judge(config, root, data)
    seed = config["seeds"][0]
    n, steps = config["smoke"]["eval_problems"], config["smoke"]["steps"]
    order = np.random.default_rng(seed).permutation(len(data["train"])).tolist()
    touched = [data["train"][order[i % len(order)]] for i in range(steps)] + data["validation"][:n]
    briefs = prepare_briefs(judge, touched, root / "dream" / "data")
    lock = freeze_model(config, root / "dream")
    run_id = identity(
        config,
        lock,
        digest(
            [corpus_manifest["hash"], {r["problem_id"]: briefs[r["problem_id"]] for r in touched}]
        ),
        "D",
        seed,
    )
    if manifest_path.exists() and read_json(manifest_path)["identity"] != run_id:
        raise ValueError("Cambió el protocolo o las consignas del smoke de sueños.")
    manifest = {
        "kind": "dream_smoke",
        "status": "running",
        "condition": "D",
        "seed": seed,
        "identity": run_id,
        "config": config,
        "model": lock,
        "judge": config["judge"],
        "corpus_hash": corpus_manifest["hash"],
        "steps_requested": steps,
        "briefs": {r["problem_id"]: briefs[r["problem_id"]] for r in touched},
    }
    write_json(manifest_path, manifest)
    runner = DreamRunner(config, lock, folder, judge, briefs)
    judge_before = fingerprint(judge.model.parameters())
    eval_seed = config["evaluation"]["seed"]
    for condition in config["conditions"]:
        runner.evaluate(data["validation"][:n], condition, folder, "before", eval_seed)
    runner.evaluate(data["validation"][:n], "neutral", folder, "before", eval_seed)
    reconcile_logs(folder)
    validation = runner.train(
        data["train"],
        "D",
        seed,
        steps,
        folder,
        run_id,
        resume=latest_checkpoint(folder) if resume else None,
    )
    reload_check(runner, folder, runner.checkpoint_path)
    import mlx.core as mx

    prompt_ids = runner.encode(runner.render_prompt(data["validation"][0], "D"))
    validation["post_reload_logits_finite"] = bool(
        mx.all(mx.isfinite(runner.model(mx.array([prompt_ids])))).item()
    )
    validation["judge_unchanged"] = fingerprint(judge.model.parameters()) == judge_before
    write_json(folder / "training-validation.json", validation)
    if not validation["post_reload_logits_finite"] or not validation["judge_unchanged"]:
        raise FloatingPointError("Logits no finitos después de recargar.")
    for condition in ["D", "neutral"]:
        runner.evaluate(data["validation"][:n], condition, folder, "after", eval_seed, step=steps)
    manifest.update(status="complete", completed_at=time.time())
    write_json(manifest_path, manifest)
    print(
        "Smoke de sueños completo. La validez del score requiere revisión humana independiente.",
        flush=True,
    )


def experiment(config, root, condition, seed, steps, resume=False):
    proof = root / "dream" / "smoke" / "run.json"
    if not proof.exists() or read_json(proof)["status"] != "complete":
        raise ValueError("Primero completá y revisá el smoke de sueños.")
    data, corpus_manifest = prepare(config, root)
    judge = load_judge(config, root, data)
    order = np.random.default_rng(seed).permutation(len(data["train"])).tolist()
    touched = [data["train"][order[i % len(order)]] for i in range(steps)] + data["test"]
    briefs = prepare_briefs(judge, touched, root / "dream" / "data")
    folder = root / "dream" / "runs" / f"{condition}-seed-{seed}"
    lock = freeze_model(config, root / "dream")
    run_id = identity(
        config,
        lock,
        digest(
            [corpus_manifest["hash"], {r["problem_id"]: briefs[r["problem_id"]] for r in touched}]
        ),
        condition,
        seed,
    )
    path = folder / "run.json"
    if path.exists():
        old = read_json(path)
        if old["identity"] != run_id or old["steps_requested"] != steps:
            raise ValueError("La corrida existente tiene otro protocolo o cantidad de pasos.")
        if old["status"] == "complete":
            print(f"{condition}/{seed} ya está completa.")
            return
        if not resume:
            raise ValueError("La corrida está incompleta; usá --resume.")
    manifest = {
        "kind": "dream_experiment",
        "status": "running",
        "condition": condition,
        "seed": seed,
        "identity": run_id,
        "config": config,
        "steps_requested": steps,
        "corpus_hash": corpus_manifest["hash"],
    }
    write_json(path, manifest)
    runner = DreamRunner(config, lock, folder, judge, briefs)
    for prompt in [condition, "neutral"]:
        runner.evaluate(data["test"], prompt, folder, "before", config["evaluation"]["seed"])
    reconcile_logs(folder)
    runner.train(
        data["train"],
        condition,
        seed,
        steps,
        folder,
        run_id,
        resume=latest_checkpoint(folder) if resume else None,
    )
    reload_check(runner, folder, runner.checkpoint_path)
    for prompt in [condition, "neutral"]:
        runner.evaluate(
            data["test"], prompt, folder, "after", config["evaluation"]["seed"], step=steps
        )
    manifest.update(status="complete", completed_at=time.time())
    write_json(path, manifest)


def worker(action, config, root, **kwargs):
    try:
        if action == "corpus":
            prepare(config, root)
        elif action == "judge":
            data, _ = prepare(config, root)
            load_judge(config, root, data)
        elif action == "anchor-check":
            from .anchors import check_protocol

            check_protocol(config, root)
        elif action == "anchor-holdout":
            from .anchor_holdout import check_holdout

            check_holdout(config, root)
        elif action == "closed-check":
            from .closed_judge import check_closed

            check_closed(config, root)
        elif action == "base-check":
            from .base_check import check_base

            check_base(config, root)
        elif action == "doctor":
            doctor(config, root / "dream")
        elif action == "smoke":
            smoke(config, root, kwargs.get("resume", False))
        elif action == "experiment":
            experiment(
                config,
                root,
                kwargs["condition"],
                kwargs["seed"],
                kwargs["steps"],
                kwargs.get("resume", False),
            )
        else:
            raise ValueError("Etapa de sueños desconocida.")
    except BaseException as error:
        import traceback

        append_jsonl(
            root / "dream" / "errors.jsonl",
            {"action": action, "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
