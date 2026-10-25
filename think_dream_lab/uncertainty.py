"""Tratamiento explícito de grupos sin recompensa completa, antes de calcular ventajas."""

import math
import statistics

from .storage import write_json


class UnscoredGroupError(RuntimeError):
    pass


def dimension_metrics(records):
    result = {}
    for key in [
        "quality",
        "dream_likeness",
        "reference_alignment",
        "content_compliance",
        "retelling_fidelity",
    ]:
        values = [r["reward"][key] for r in records if r["reward"].get(key) is not None]
        if any(key in r["reward"] for r in records):
            result[f"n_reward_{key}"] = len(values)
        if values:
            result[f"reward_{key}"] = statistics.mean(values)
    return result


def require_scored_group(records, folder, step):
    unknown = [i for i, row in enumerate(records) if row["reward"]["total"] is None]
    if unknown:
        write_json(
            folder / "unscored-groups" / f"step-{step:06d}.json",
            {
                "step": step,
                "unscored_candidates": unknown,
                "records": records,
                "optimizer_updated": False,
                "policy": "stop_entire_group_without_zero_imputation_or_candidate_filtering",
            },
        )
        raise UnscoredGroupError(
            "El juez se abstuvo: grupo completo conservado sin calcular ventajas ni actualizar pesos."
        )
    if any(
        type(r["reward"]["total"]) not in {int, float} or not math.isfinite(r["reward"]["total"])
        for r in records
    ):
        raise ValueError("El grupo contiene una recompensa inválida.")
