"""Diagnóstico sin seriales, UUID ni identificadores de la computadora."""

import importlib.metadata
import platform
import re
import shutil
import subprocess
from pathlib import Path

import psutil


def command(*args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def telemetry(pid=None):
    swap = command("sysctl", "vm.swapusage")
    match = re.search(r"used = ([\d.]+)([MG])", swap)
    used = float(match[1]) * (1024 ** (2 if match[2] == "M" else 3)) if match else 0
    pressure = command("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
    rss = 0
    try:
        process = psutil.Process(pid)
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return {
        "rss_bytes": rss,
        "swap_used_bytes": used,
        "pressure_level": int(pressure) if pressure.isdigit() else None,
        "available_bytes": psutil.virtual_memory().available,
    }


def diagnose(root: Path) -> dict:
    versions = {}
    for name in [
        "think-dream-lab",
        "mlx",
        "mlx-metal",
        "mlx-lm",
        "mlx-lm-lora",
        "transformers",
        "tokenizers",
        "numpy",
        "huggingface-hub",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import mlx.core as mx

    return {
        "architecture": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "metal_available": mx.metal.is_available(),
        "packages": versions,
        "telemetry": telemetry(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
    }
