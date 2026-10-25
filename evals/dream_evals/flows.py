"""Matriz de evaluaciones existentes: condiciones, fases y semillas conservadas."""

from pathlib import Path

from inspect_flow import FlowFactory, FlowOptions, FlowSpec, FlowStoreConfig, FlowTask

from .tasks import REPLAY_MODEL, dream_results, load_run, read_json, select_records


def dream_flow(results_dir="results") -> FlowSpec:
    root = Path(results_dir).resolve()
    dream = root / "dream"
    candidates = [
        dream / "smoke",
        *sorted((dream / "runs").glob("*")),
        *sorted((dream / "evaluations").glob("*")),
    ]
    tasks = []
    for folder in candidates:
        if not (folder / "run.json").exists():
            continue
        if read_json(folder / "run.json").get("status") != "complete":
            continue
        run = str(folder.relative_to(dream))
        manifest, rows, _ = load_run(root, run)
        for phase, condition in sorted({(r["phase"], r["condition"]) for r in rows}):
            _, _, _, snapshot = select_records(root, run, condition, phase)
            tasks.append(
                FlowTask(
                    factory=FlowFactory(
                        dream_results,
                        results_dir=str(root),
                        run=run,
                        condition=condition,
                        phase=phase,
                        snapshot=snapshot,
                    ),
                    model=REPLAY_MODEL,
                    metadata={
                        "condition": condition,
                        "phase": phase,
                        "seed": manifest["seed"],
                        "mode": "offline_replay",
                    },
                )
            )
    if not tasks:
        raise ValueError(
            "No hay corridas de sueños completas para Inspect. Ejecutá primero el smoke."
        )
    return FlowSpec(
        tasks=tasks,
        execution_type="inproc",
        log_dir=str(dream / "inspect"),
        store=FlowStoreConfig(path=str(dream / "inspect-store"), read=True, write=True),
        options=FlowOptions(
            max_samples=1,
            max_tasks=1,
            retry_attempts=0,
            display="plain",
            notification=False,
            ctl_server=False,
        ),
        flow_metadata={
            "project": "Dream",
            "mode": "offline_replay",
            "new_inferences": False,
            "new_training": False,
        },
    )


def live_dream_flow(
    results_dir="results/dream-v4",
    config_path="configs/dream-v4.yaml",
    condition="D",
    start_index=1,
    limit=1,
):
    from .live import CONTROLLER_MODEL, PROJECT, dream_live, source_snapshot

    root, config = Path(results_dir).resolve(), Path(config_path).resolve()
    snapshot = source_snapshot(PROJECT, config, root, start_index, limit)[2]
    return FlowSpec(
        tasks=[
            FlowTask(
                factory=FlowFactory(
                    dream_live,
                    results_dir=str(root),
                    config_path=str(config),
                    condition=condition,
                    start_index=start_index,
                    limit=limit,
                    snapshot=snapshot,
                ),
                model=CONTROLLER_MODEL,
                metadata={"condition": condition, "mode": "supervised_local_mlx"},
            )
        ],
        execution_type="inproc",
        log_dir=str(root / "dream/inspect-live"),
        store=FlowStoreConfig(path=str(root / "dream/inspect-live-store"), read=True, write=True),
        options=FlowOptions(
            max_samples=1,
            max_tasks=1,
            retry_attempts=0,
            display="plain",
            notification=False,
            ctl_server=False,
        ),
        flow_metadata={
            "project": "Dream",
            "mode": "supervised_local_mlx",
            "local_generation_may_run": True,
            "new_training": False,
            "calibrated": False,
        },
    )
