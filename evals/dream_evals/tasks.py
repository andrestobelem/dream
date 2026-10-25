"""Tareas offline: conservar exactamente salidas y scores de las corridas locales."""

import hashlib
import json
import math
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

FIELDS = (
    "dream_likeness",
    "reference_alignment",
    "retelling_fidelity",
    "quality",
    "format",
    "total",
)
CONDITIONS = ("A", "B", "C", "D", "neutral")
PHASES = ("before", "after", "rollout")
REPLAY_MODEL = "mockllm/dream-replay"


def read_json(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def jsonl(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def score_values(reward):
    values = {}
    for key in FIELDS:
        value = reward[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Score inválido: {key}={value!r}")
        values[key] = float(value)
    return values


def load_run(results_dir="results", run="smoke"):
    dream = Path(results_dir).resolve() / "dream"
    folder = (dream / run).resolve()
    if not folder.is_relative_to(dream) or "diagnostics" in folder.relative_to(dream).parts:
        raise ValueError("La corrida debe estar dentro de results/dream y fuera de diagnostics.")
    manifest = read_json(folder / "run.json")
    if manifest.get("kind") not in {"dream_smoke", "dream_experiment"}:
        raise ValueError("Se requieren resultados de sueños, no del diagnóstico aritmético.")
    if manifest["status"] != "complete":
        raise ValueError("La corrida debe estar completa antes de compararla.")
    rows = jsonl(folder / "responses.jsonl")
    for path in sorted((folder / "checkpoints").glob("step-*/state.json")):
        rows.extend(read_json(path)["extra"].get("responses", []))
    unique = {}
    for row in rows:
        if row["phase"] not in PHASES or row["condition"] not in CONDITIONS:
            continue
        key = (row["phase"], row["condition"], row["step"], row["problem_id"], row["candidate"])
        if key in unique and canonical(row) != canonical(unique[key]):
            raise ValueError(f"Dos respuestas distintas tienen el mismo identificador: {key}")
        score_values(row["reward"])
        unique[key] = row
    references = {
        r["problem_id"]: r["reference"]
        for split in ["train", "validation", "test"]
        for r in jsonl(dream / "data" / f"{split}.jsonl")
    }
    return manifest, list(unique.values()), references


def select_records(results_dir, run, condition, phase):
    if condition not in CONDITIONS or phase not in PHASES:
        raise ValueError("Condición o fase desconocida.")
    manifest, records, references = load_run(results_dir, run)
    selected = [r for r in records if r["condition"] == condition and r["phase"] == phase]
    if not selected:
        raise ValueError(
            f"No hay respuestas para {run}/{phase}/{condition}; no se inventa una celda vacía."
        )
    selected.sort(key=lambda r: (r["step"], r["problem_id"], r["candidate"]))
    for row in selected:
        if row["problem_id"] not in references:
            raise ValueError("Falta el relato reservado que corresponde a una respuesta.")
    snapshot = digest(
        {
            "run_identity": manifest["identity"],
            "records": selected,
            "references": {r["problem_id"]: references[r["problem_id"]] for r in selected},
            "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    return manifest, selected, references, snapshot


@solver
def replay_recorded() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # No se llama a generate ni a ningún proveedor; el texto ya fue producido por MLX.
        text = state.metadata["recorded_output"]
        state.output = ModelOutput.from_content(model="recorded/dream", content=text)
        state.messages.append(ChatMessageAssistant(content=text))
        state.completed = True
        return state

    return solve


@scorer(metrics={name: [mean()] for name in FIELDS})
def recorded_scores():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(
            value=score_values(state.metadata["recorded_reward"]),
            answer=state.output.completion,
            explanation="Scores originales del juez local congelado; esta tarea no los vuelve a calcular.",
            metadata={
                "source": "recorded_local_judge",
                "replayed": True,
                "rubric": state.metadata["judge"],
                "neutral_formula": state.metadata["recorded_reward"].get("neutral_formula", False),
            },
        )

    return score


@task
def dream_results(
    results_dir: str = "results",
    run: str = "smoke",
    condition: str = "D",
    phase: str = "after",
    snapshot: str = "",
) -> Task:
    manifest, records, references, actual_snapshot = select_records(
        results_dir, run, condition, phase
    )
    if snapshot and snapshot != actual_snapshot:
        raise ValueError(
            "Los resultados cambiaron desde que se preparó Flow; regenerá la configuración."
        )
    samples = []
    judge = manifest["config"]["judge"]
    for record in records:
        sample_id = digest(
            [run, phase, condition, record["step"], record["problem_id"], record["candidate"]]
        )[:24]
        samples.append(
            Sample(
                id=sample_id,
                input=record["prompt"],
                target=references[record["problem_id"]],
                metadata={
                    "recorded_output": record["text"],
                    "recorded_reward": record["reward"],
                    "problem_id": record["problem_id"],
                    "condition": condition,
                    "phase": phase,
                    "step": record["step"],
                    "candidate": record["candidate"],
                    "seed": manifest["seed"],
                    "family": record["family"],
                    "judge": judge,
                    "run_identity": manifest["identity"],
                    "recorded_tokens": record["tokens"],
                    "recorded_truncated": record["truncated"],
                    "generation_rerun": False,
                    "replay_snapshot": actual_snapshot,
                },
            )
        )
    return Task(
        dataset=samples,
        solver=replay_recorded(),
        scorer=recorded_scores(),
        model=REPLAY_MODEL,
        version="dream-recorded-v1",
        fail_on_error=True,
        metadata={
            "mode": "offline_replay",
            "run": run,
            "condition": condition,
            "phase": phase,
            "source_snapshot": actual_snapshot,
            "training_replicate": manifest["seed"],
            "generation_rerun": False,
            "judge_rerun": False,
        },
        description="Revisión de salidas y scores de Dream ya guardados. Sin inferencias nuevas.",
    )
