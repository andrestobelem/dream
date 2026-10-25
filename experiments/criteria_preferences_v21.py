"""Criterio de promoción fijado antes de entrenar, sin selección de checkpoints."""

import random
from collections import Counter
from statistics import mean


def quality_result(protocol, ratings):
    expected = {(r["problem_id"], i) for r in protocol["dev_rows"] for i in range(4)}
    observed = [(r["problem_id"], r["candidate"]) for r in ratings]
    if len(observed) != 96 or len(set(observed)) != 96 or set(observed) != expected:
        raise ValueError("La evaluación requiere las 96 comparaciones, sin eliminar fallos.")
    scores = protocol["evaluation"]["quality_score"]
    groups = {r["problem_id"]: [] for r in protocol["dev_rows"]}
    for r in ratings:
        if r["preference"] not in scores:
            raise ValueError("Etiqueta de calidad inválida.")
        groups[r["problem_id"]].append(scores[r["preference"]])
    clusters = [mean(g) for g in groups.values()]
    rng = random.Random(20261031)
    boot = sorted(mean(rng.choices(clusters, k=24)) for _ in range(20000))

    def percentile(q):
        position = q * (len(boot) - 1)
        lower = int(position)
        fraction = position - lower
        return boot[lower] * (1 - fraction) + boot[min(lower + 1, len(boot) - 1)] * fraction

    return {
        "net_mean": mean(clusters),
        "interval_95": [percentile(0.025), percentile(0.975)],
        "request_units": 24,
        "comparisons": 96,
        "counts": dict(Counter(r["preference"] for r in ratings)),
    }


def promotion_decision(protocol, ratings, diagnostics):
    """Recibe diagnósticos completos; su auditado y congelación se verifican aparte."""
    q = quality_result(protocol, ratings)
    gates = protocol["evaluation"]["promotion_all_required"]
    before, after = diagnostics["before"], diagnostics["after"]
    variety = diagnostics["variety_counts"]
    if sum(variety.values()) != 24 or set(variety) - {"after", "before", "tie", "no_se"}:
        raise ValueError("Faltan las 24 revisiones de variedad.")
    checks = {
        "quality_effect": q["net_mean"] >= gates["quality_net_mean_minimum"],
        "quality_interval": q["interval_95"][0] > gates["quality_interval_lower_strictly_above"],
        "format": after["format_valid"]
        >= max(gates["format_valid_minimum"], before["format_valid"]),
        "motifs": after["motif_major"] <= before["motif_major"],
        "variety_preference": variety.get("after", 0) >= variety.get("before", 0),
        "usable_groups": after["groups_fewer_than_two_episodes"]
        <= before["groups_fewer_than_two_episodes"],
        "dependency": after["dependency_full"] >= gates["dependency_full_facts_minimum"],
        "dependency_format": after["dependency_format"]
        == gates["dependency_valid_format_required"],
        "all_reviews_frozen": diagnostics["all_reviews_frozen"] is True,
    }
    return {
        "quality": q,
        "checks": checks,
        "promote": all(checks.values()),
        "decision": "provisional_candidate" if all(checks.values()) else "keep32",
    }
