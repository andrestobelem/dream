"""Checkpoint transaccional de adaptadores, optimizador y aleatoriedad."""

import os
import random
from pathlib import Path

import numpy as np

from .storage import read_json, write_json


def restore_mlx_random(key):
    import mlx.core as mx

    # MLX 0.32 usa un sentinel de sólo lectura; key(uint64) reconstruye sus dos uint32.
    words = key.tolist()
    if len(words) != 2:
        raise RuntimeError("Formato de PRNG MLX desconocido.")
    mx.random.seed((int(words[0]) << 32) | int(words[1]))
    if not mx.array_equal(mx.random.state[0], key).item():
        raise RuntimeError("No se pudo restaurar exactamente la clave aleatoria MLX.")


def save(folder: Path, model, optimizer, *, step: int, run_identity: str, extra: dict):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"step-{step:06d}"
    if target.exists():
        raise ValueError(f"Checkpoint existente: {target}")
    temp = folder / f".step-{step:06d}-{os.getpid()}.tmp"
    temp.mkdir()
    mx.eval(model.parameters(), optimizer.state, mx.random.state)
    mx.save_safetensors(
        str(temp / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
    )
    mx.save_safetensors(str(temp / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    mx.save_safetensors(
        str(temp / "mlx_random.safetensors"),
        {str(i): value for i, value in enumerate(mx.random.state)},
    )
    npstate = np.random.get_state()
    metadata = {
        "step": step,
        "identity": run_identity,
        "extra": extra,
        "python_random": random.getstate(),
        "numpy_random": [npstate[0], npstate[1].tolist(), *npstate[2:]],
    }
    write_json(temp / "state.json", metadata)
    os.replace(temp, target)
    write_json(folder / "latest.json", {"directory": target.name, "step": step})
    return target


def load(folder: Path, model, optimizer, run_identity: str):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    state = read_json(folder / "state.json")
    if state["identity"] != run_identity:
        raise ValueError("El checkpoint pertenece a otra condición, semilla o configuración.")
    model.load_weights(str(folder / "adapters.safetensors"), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(folder / "optimizer.safetensors")).items()))
    rng = mx.load(str(folder / "mlx_random.safetensors"))
    restore_mlx_random(rng["0"])

    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value

    random.setstate(tuples(state["python_random"]))
    npstate = state["numpy_random"]
    np.random.set_state(
        (npstate[0], np.array(npstate[1], dtype=np.uint32), npstate[2], npstate[3], npstate[4])
    )
    mx.eval(model.parameters(), optimizer.state, mx.random.state)
    return state
