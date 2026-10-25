"""Inspect inicia un lote MLX local y conserva una muestra por relato, sin entrenar."""

import asyncio
import hashlib
import signal
import subprocess
from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.solver import solver

from .tasks import canonical, digest, jsonl, load_run, read_json, recorded_closed_scores

PROJECT = Path(__file__).resolve().parents[2]
CONTROLLER_MODEL = "mockllm/dream-live-controller"


def source_snapshot(project, config_path, results, start_index, limit):
    config = yaml.safe_load(config_path.read_text())
    rows = jsonl(results / "dream/data/validation.jsonl")
    selected = rows[start_index : start_index + limit]
    if (
        config["judge"].get("rubric_version") != "dream-v4"
        or start_index < 0
        or limit < 1
        or len(selected) != limit
    ):
        raise ValueError(
            "Se requiere dream-v4 y un rango preparado de validación. Ejecutá corpus primero."
        )
    sources = [
        *sorted((project / "think_dream_lab").glob("*.py")),
        project / "uv.lock",
        project / "evals/uv.lock",
        Path(__file__),
        Path(__file__).with_name("tasks.py"),
    ]
    snapshot = digest(
        {
            "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "config": config,
            "records": selected,
        }
    )
    return config, selected, snapshot


def core_manifest_hash(project):
    result = subprocess.run(
        [
            str(project / ".venv/bin/python"),
            "-c",
            "from think_dream_lab.config import implementation_manifest; print(implementation_manifest()['hash'])",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


class LocalBatch:
    def __init__(self, project, config_path, results, condition, start_index, limit, snapshot):
        self.project, self.config_path, self.results = project, config_path, results
        self.condition, self.start_index, self.limit, self.snapshot = (
            condition,
            start_index,
            limit,
            snapshot,
        )
        self.lock, self.output = asyncio.Lock(), None

    async def records(self):
        async with self.lock:
            config, selected, current = source_snapshot(
                self.project, self.config_path, self.results, self.start_index, self.limit
            )
            if current != self.snapshot:
                raise ValueError("Cambió el código, configuración o dataset de la evaluación viva.")
            if self.output is not None:
                return self.output
            expected_core = core_manifest_hash(self.project)
            existing = {
                p.parent: {r["problem_id"] for r in jsonl(p.parent / "responses.jsonl")}
                for p in (self.results / "dream/evaluations").glob("*/run.json")
            }
            before_forwards = len(jsonl(self.results / "dream/closed-calls.jsonl"))
            before_judge_calls = len(jsonl(self.results / "dream/judge-calls.jsonl"))
            process = await asyncio.create_subprocess_exec(
                str(self.project / ".venv/bin/python"),
                "-m",
                "think_dream_lab",
                "evaluate",
                "--config",
                str(self.config_path),
                "--results",
                str(self.results),
                "--condition",
                self.condition,
                "--limit",
                str(self.limit),
                "--start-index",
                str(self.start_index),
                cwd=self.project,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await process.communicate()
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.send_signal(signal.SIGINT)
                    await asyncio.shield(process.wait())
                raise
            if process.returncode:
                raise RuntimeError(
                    "El worker local no completó la evaluación: "
                    + stdout.decode(errors="replace")[-3000:]
                )
            if (
                source_snapshot(
                    self.project, self.config_path, self.results, self.start_index, self.limit
                )[2]
                != self.snapshot
            ):
                raise ValueError("Las fuentes cambiaron durante la inferencia local.")
            wanted = {r["problem_id"] for r in selected}
            matches = []
            for path in (self.results / "dream/evaluations").glob("*/run.json"):
                m = read_json(path)
                if (
                    m.get("status") == "complete"
                    and m.get("implementation_hash") == expected_core
                    and m.get("condition") == self.condition
                    and canonical(m["config"]) == canonical(config)
                    and set(m.get("briefs", {})) == wanted
                ):
                    matches.append(path)
            if len(matches) != 1:
                raise ValueError(
                    "No se encontró un único resultado local del protocolo solicitado."
                )
            folder = matches[0].parent
            run = str(folder.relative_to(self.results / "dream"))
            manifest, rows, _ = load_run(self.results, run)
            if (
                not manifest.get("actor_unchanged")
                or not manifest.get("judge_unchanged")
                or manifest.get("training_updates") != 0
            ):
                raise ValueError("Falta la comprobación de evaluación sin cambios de pesos.")
            by_id = {
                r["problem_id"]: r
                for r in rows
                if r["condition"] == self.condition and r["phase"] == "before"
            }
            if set(by_id) != wanted or len(rows) != len(wanted):
                raise ValueError("El lote no contiene exactamente las muestras solicitadas.")
            self.output = {
                pid: {
                    "record": row,
                    "source_run": run,
                    "run_identity": manifest["identity"],
                    "local_generation_new": pid not in existing.get(folder, set()),
                    "batch_new_classifier_forwards": len(
                        jsonl(self.results / "dream/closed-calls.jsonl")
                    )
                    - before_forwards,
                    "batch_new_judge_generations": len(
                        jsonl(self.results / "dream/judge-calls.jsonl")
                    )
                    - before_judge_calls,
                }
                for pid, row in by_id.items()
            }
            return self.output


@solver
def local_mlx_batch(project, config_path, results_dir, condition, start_index, limit, snapshot):
    batch = LocalBatch(
        Path(project), Path(config_path), Path(results_dir), condition, start_index, limit, snapshot
    )

    async def solve(state, generate):
        records = await batch.records()
        item = records[state.metadata["problem_id"]]
        record = item["record"]
        state.output = ModelOutput.from_content(model="local/mlx", content=record["text"])
        state.messages = [
            ChatMessageUser(content=record["prompt"]),
            ChatMessageAssistant(content=record["text"]),
        ]
        state.metadata.update(
            recorded_reward=record["reward"],
            recorded_tokens=record["tokens"],
            actual_actor_prompt=record["prompt"],
            source_run=item["source_run"],
            run_identity=item["run_identity"],
            local_generation_new=item["local_generation_new"],
            model_execution="supervised_local_mlx",
            training_updates=0,
            batch_new_classifier_forwards=item["batch_new_classifier_forwards"],
            batch_new_judge_generations=item["batch_new_judge_generations"],
        )
        state.completed = True
        return state

    return solve


@task
def dream_live(
    results_dir="results/dream-v4-qwen",
    config_path="configs/dream-v4.yaml",
    condition="D",
    start_index=0,
    limit=1,
    snapshot="",
):
    results, config_file = Path(results_dir).resolve(), Path(config_path).resolve()
    config, rows, actual = source_snapshot(PROJECT, config_file, results, start_index, limit)
    if condition not in {"A", "B", "C", "D", "neutral"} or (snapshot and snapshot != actual):
        raise ValueError("Condición o snapshot de evaluación viva inválido.")
    samples = [
        Sample(
            id=r["problem_id"],
            input="Evaluación local del relato " + r["problem_id"],
            target=r["reference"],
            metadata={
                "problem_id": r["problem_id"],
                "condition": condition,
                "judge": config["judge"],
                "generator": config["model"],
            },
        )
        for r in rows
    ]
    return Task(
        dataset=samples,
        solver=local_mlx_batch(
            str(PROJECT), str(config_file), str(results), condition, start_index, limit, actual
        ),
        scorer=recorded_closed_scores(),
        model=CONTROLLER_MODEL,
        version="dream-local-v1",
        fail_on_error=True,
        metadata={
            "mode": "supervised_local_mlx",
            "source_snapshot": actual,
            "training_updates": 0,
            "calibrated": False,
            "provider_usage_excludes_local_worker": True,
        },
    )
