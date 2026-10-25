"""Configuración validada y manifiesto congelado."""

import hashlib
import math
from pathlib import Path

import yaml

from .storage import digest, read_json, write_json

BACKEND_COMMIT = "4fa3c5148b17a5833b3074e4e58c6299baec1018"


def load_config(path: str | Path) -> dict:
    path = Path(path)
    config = read_json(path) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    dreambank = config.get("protocol") == "dreambank-embeddings-v1"
    if dreambank:
        validate_dreambank(config)
    if config["model"]["id"] not in {"Qwen/Qwen3-1.7B-Base", "HuggingFaceTB/SmolLM2-1.7B"}:
        raise ValueError("El modelo debe pertenecer a los checkpoints base registrados.")
    dream_only = config.get("task") == "dream" and config["conditions"] == ["D"]
    if not dream_only and config["conditions"] != ["A", "B", "C", "D"]:
        raise ValueError("Usá [D] para Dream o [A, B, C, D] para el protocolo histórico.")
    if not dreambank and (len(config["seeds"]) != 3 or len(set(config["seeds"])) != 3):
        raise ValueError("Se requieren tres semillas distintas.")
    if config["training"]["loss_type"] != "grpo":
        raise ValueError("El protocolo principal usa GRPO.")
    if config["training"]["group_size"] < 2:
        raise ValueError("GRPO necesita al menos dos candidatos por pregunta.")
    if config["training"]["microbatch"] != 1:
        raise ValueError("El wrapper validado usa una pregunta y scoring microbatch=1.")
    if config["training"]["advantage_epsilon"] != 1e-4:
        raise ValueError("El backend fijado normaliza ventajas con epsilon=1e-4.")
    if config["training"]["checkpoint_every"] != 1:
        raise ValueError("La recuperación validada guarda cada actualización.")
    if config["training"]["dropout"] != 0 or config["lora"]["dropout"] != 0:
        raise ValueError("Los modos de rollout y scoring requieren dropout cero.")
    for section, key in [
        ("generation", "max_tokens"),
        ("generation", "max_prompt_tokens"),
        ("runtime", "budget_seconds"),
        ("smoke", "steps"),
        ("smoke", "eval_problems"),
    ]:
        if config[section][key] <= 0:
            raise ValueError(f"{section}.{key} debe ser positivo.")
    if config["evaluation"]["samples_per_problem"] != 1:
        raise ValueError("La versión inicial calcula pass@1 con una muestra por problema.")
    if config["smoke"]["steps"] > 20:
        raise ValueError("El smoke debe tener entre 1 y 20 actualizaciones.")
    dream = config.get("task") == "dream"
    data = config["corpus"] if dream else config["data"]
    if not dream and data["difficulty"] not in {"easy", "medium", "hard"}:
        raise ValueError("Dificultad desconocida.")
    if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in config["seeds"]):
        raise ValueError("Las semillas deben ser enteros uint32.")
    if any(not isinstance(data[s], int) or data[s] < 1 for s in ["train", "validation", "test"]):
        raise ValueError("Los tamaños de datos deben ser enteros positivos.")
    if config["smoke"]["eval_problems"] > data["validation"]:
        raise ValueError("La muestra del smoke supera el conjunto de validación.")
    for section, keys in {
        "training": ["learning_rate", "beta", "epsilon", "grad_clip"],
        "generation": ["temperature", "top_p", "top_k", "min_p"],
        "runtime": [
            "budget_seconds",
            "poll_seconds",
            "telemetry_seconds",
            "swap_growth_limit_gib",
            "min_free_disk_gib",
        ],
    }.items():
        if any(not math.isfinite(config[section][k]) or config[section][k] < 0 for k in keys):
            raise ValueError(f"Parámetros no finitos o negativos en {section}.")
    if not 0 < config["generation"]["top_p"] <= 1 or config["generation"]["min_p"] > 1:
        raise ValueError("top_p debe estar en (0, 1] y min_p en [0, 1].")
    if config["training"]["learning_rate"] == 0 or config["training"]["grad_clip"] == 0:
        raise ValueError("La tasa de aprendizaje y grad_clip deben ser positivos.")
    if config["lora"]["rank"] < 1 or config["lora"]["scale"] <= 0:
        raise ValueError("Rango y escala LoRA deben ser positivos.")
    if config["runtime"]["poll_seconds"] <= 0 or config["runtime"]["telemetry_seconds"] <= 0:
        raise ValueError("Los intervalos de supervisión deben ser positivos.")
    if dream and not dreambank:
        if config["judge"].get("rubric_version") not in {"dream-v2", "dream-v3", "dream-v4"}:
            raise ValueError("Versión de rúbrica de sueños no compatible.")
        if config["corpus"]["source"] != "dreamviews":
            raise ValueError("El protocolo inicial usa el corpus DreamViews versionado.")
        if not config["judge"]["revision"] or config["judge"]["temperature"] != 0:
            raise ValueError("El juez debe estar fijado a una revisión y usar temperatura cero.")
        if config["smoke"]["train_condition"] != "D":
            raise ValueError("El smoke de sueños empieza en D (dream/dream).")
        weights = config["reward"]
        if any(not math.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("Pesos de recompensa no válidos.")
        if not math.isclose(
            sum(
                weights[k]
                for k in [
                    "dream_likeness",
                    "content_compliance"
                    if config["judge"]["rubric_version"] in {"dream-v3", "dream-v4"}
                    else "reference_alignment",
                    "retelling_fidelity",
                ]
            ),
            1,
        ):
            raise ValueError("Los pesos de calidad deben sumar uno.")
        if not math.isclose(weights["quality_weight"] + weights["format_weight"], 1):
            raise ValueError("Los pesos de calidad y formato deben sumar uno.")
    return config


def validate_dreambank(config):
    """Perfil separado: no hereda los requisitos del juez ni de A/B/C/D."""
    if config.get("task") != "dream" or config.get("conditions") != ["D"]:
        raise ValueError("DreamBank usa únicamente D: soñar dentro de <dream>.")
    if config["model"] != {
        "id": "Qwen/Qwen3-1.7B-Base",
        "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    }:
        raise ValueError("Este piloto conserva el modelo y revisión del primer piloto.")
    if config["generation"].get("format_example") is not False:
        raise ValueError("El prompt DreamBank no lleva un ejemplo resuelto.")
    corpus = config["corpus"]
    if corpus["source"] != "dreambank" or corpus["license"] != "CC-BY-NC-SA-4.0":
        raise ValueError("Fuente o licencia DreamBank incorrecta.")
    if not isinstance(corpus["sha256"], str) or len(corpus["sha256"]) != 64:
        raise ValueError("Falta fijar el SHA-256 del archivo DreamBank.")
    if any(
        type(corpus[k]) is not int or corpus[k] <= 0
        for k in ("train", "validation", "test", "min_words", "max_words", "max_per_group")
    ):
        raise ValueError("Los tamaños y límites del corpus deben ser enteros positivos.")
    if corpus["min_words"] > corpus["max_words"]:
        raise ValueError("Intervalo de longitud del corpus inválido.")
    if not config["seeds"] or any(
        type(s) is not int or not 0 <= s < 2**32 for s in config["seeds"]
    ):
        raise ValueError("Se requiere al menos una semilla uint32.")
    reward = config["reward"]
    extractor = config["seed_extractor"]
    if (extractor["min_seeds"], extractor["max_seeds"], extractor["max_words"]) != (3, 8, 24):
        raise ValueError("El selector actual usa entre tres y ocho semillas y hasta 24 palabras.")
    if (
        reward["scored_block"] != "answer"
        or reward["credit_assignment"] != "entire_sequence"
        or reward.get("automatic_copy_penalty") is not False
    ):
        raise ValueError(
            "Se puntúa answer y se recompensa la secuencia completa, sin pena por copia."
        )
    if config["training"]["loss_type"] != "grpo" or config["training"]["group_size"] < 2:
        raise ValueError("El entrenamiento requiere GRPO con al menos dos candidatos.")
    if config["runtime"]["budget_seconds"] <= 0:
        raise ValueError("El presupuesto acumulado debe ser positivo.")
    encoder = config["embeddings"]
    if (
        encoder["pooling"] != "masked_mean_l2"
        or encoder["long_text"] != "token_weighted_chunk_mean_l2"
        or encoder["device"] != "cpu"
        or not encoder["revision"]
        or not 3 <= encoder["max_tokens"] <= 256
        or encoder["batch_size"] < 1
    ):
        raise ValueError("El encoder requiere revisión fija, CPU y fragmentos de hasta 256 tokens.")
    if reward["formula"] != "nearest-dream-cosine-v1":
        raise ValueError("Fórmula de cercanía no implementada.")
    return config


def implementation_manifest():
    package = Path(__file__).parent
    names = [
        "backend.py",
        "checkpoint.py",
        "config.py",
        "data.py",
        "prompts.py",
        "rewards.py",
        "storage.py",
        "workflow.py",
        "corpus.py",
        "dreambank.py",
        "dreambank_task.py",
        "seed_extractor.py",
        "dreambank_workflow.py",
        "embedding_reward.py",
        "embedding_runner.py",
        "dream_task.py",
        "dream_generation.py",
        "dream_workflow.py",
        "judge.py",
        "judge_checks.py",
        "anchors.py",
        "evidence.py",
        "anchor_holdout.py",
        "dream_verification.py",
        "closed_judge.py",
        "closed_transfer.py",
        "base_check.py",
        "closed_task.py",
        "uncertainty.py",
        "provenance.py",
    ]
    files = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}
    lockfile = package.parent / "uv.lock"
    if lockfile.exists():
        files["uv.lock"] = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    return {"files": files, "hash": digest(files)}


def freeze_model(config: dict, root: Path, offline=False) -> dict:
    lockfile = root / "model-lock.json"
    requested = config["model"].get("revision")
    if lockfile.exists():
        lock = read_json(lockfile)
        if lock["id"] != config["model"]["id"] or (requested and requested != lock["revision"]):
            raise ValueError("El modelo solicitado no coincide con el lock existente.")
        return lock
    if offline:
        raise ValueError("Falta model-lock.json; ejecutá doctor con conexión primero.")
    from huggingface_hub import model_info

    info = model_info(config["model"]["id"], revision=requested or "main")
    lock = {"id": config["model"]["id"], "revision": info.sha, "backend_commit": BACKEND_COMMIT}
    write_json(lockfile, lock)
    return lock


def identity(config: dict, model_lock: dict, data_hash: str, condition: str, seed: int) -> str:
    return digest(
        {
            "config": {key: value for key, value in config.items() if key != "runtime"},
            "model": model_lock,
            "data_hash": data_hash,
            "condition": condition,
            "seed": seed,
            "backend": BACKEND_COMMIT,
            "implementation": implementation_manifest()["hash"],
        }
    )
