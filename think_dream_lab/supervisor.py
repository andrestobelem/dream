"""Límite efectivo de tiempo acumulado y monitoreo fuera del proceso de MLX."""

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .storage import append_jsonl, read_json, write_json
from .system import telemetry


def supervise(config, root: Path, worker_args: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Ya hay otra ejecución usando este directorio de resultados."
            ) from error
        return _supervise(config, root, worker_args)


def _supervise(config, root: Path, worker_args: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "budget.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"used_seconds": 0, "runs": []}
    remaining = config["runtime"]["budget_seconds"] - ledger["used_seconds"]
    if remaining <= 0:
        print("Presupuesto agotado. Se conserva el último checkpoint completo.", flush=True)
        return 3
    environment = {
        **os.environ,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MLX_ENABLE_TF32": "0",
    }
    from .provenance import freeze_source

    frozen = freeze_source(root, config)
    actual_args = list(worker_args)
    if "--config" in actual_args:
        actual_args[actual_args.index("--config") + 1] = frozen["config_path"]
    else:
        actual_args += ["--config", frozen["config_path"]]
    launcher = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('think_dream_lab',run_name='__main__',alter_sys=True)"
    baseline_swap = telemetry()["swap_used_bytes"]
    start = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-c", launcher, frozen["directory"], "_worker", *actual_args],
        env=environment,
        start_new_session=True,
    )
    entry = {
        "pid": process.pid,
        "args": worker_args,
        "reserved_seconds": remaining,
        "started_at": time.time(),
        "source_snapshot": frozen,
    }
    # Reservar todo evita reiniciar presupuesto si se mata al supervisor.
    write_json(
        ledger_path,
        {
            "used_seconds": config["runtime"]["budget_seconds"],
            "runs": [*ledger["runs"], {**entry, "status": "running"}],
        },
    )
    reason = None
    peak_rss = 0
    peak_swap = baseline_swap
    last_sample = 0
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - start
            stats = telemetry(process.pid)
            peak_rss = max(peak_rss, stats["rss_bytes"])
            peak_swap = max(peak_swap, stats["swap_used_bytes"])
            if elapsed - last_sample >= config["runtime"]["telemetry_seconds"]:
                append_jsonl(
                    root / "telemetry.jsonl",
                    {"elapsed_seconds": elapsed, "worker": worker_args, **stats},
                )
                last_sample = elapsed
            if elapsed >= max(0, remaining - 10):
                reason = "time_budget"
            elif stats["pressure_level"] == 4:
                reason = "critical_memory_pressure"
            elif (
                stats["swap_used_bytes"] - baseline_swap
                > config["runtime"]["swap_growth_limit_gib"] * 1024**3
            ):
                reason = "swap_growth"
            if reason:
                break
            time.sleep(min(config["runtime"]["poll_seconds"], 1))
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=min(10, max(0.1, remaining - (time.monotonic() - start))))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        elapsed = time.monotonic() - start
        entry.update(
            elapsed_seconds=elapsed,
            returncode=process.returncode,
            reason=reason,
            peak_rss_bytes=peak_rss,
            initial_swap_bytes=baseline_swap,
            peak_swap_bytes=peak_swap,
            status="complete" if process.returncode == 0 and not reason else "incomplete",
        )
        write_json(
            ledger_path,
            {"used_seconds": ledger["used_seconds"] + elapsed, "runs": [*ledger["runs"], entry]},
        )
    if reason:
        print(f"Ejecución incompleta: {reason}. Checkpoints previos conservados.", flush=True)
    return 3 if reason else process.returncode
