"""Calibración operativa asistida, con fuentes separadas y abstención explícita."""

import csv
import hashlib
from copy import deepcopy
from pathlib import Path

from .closed_judge import ClosedJudge, classifier_signature, decide
from .storage import digest, read_json, write_json

LABELS = {
    "preservado": "preserved",
    "omitido": "omitted",
    "contradicho": "contradicted",
    "no_se": None,
}
REVIEW = Path("results/dream-v4/dream/review-pairs/7c803087be40/revision-asistente-v2.csv")
CRITERIA = {
    "calibration_accuracy": 0.90,
    "validation_accuracy": 0.80,
    "minimum_coverage": 0.60,
    "minimum_validation_decided": 8,
    "minimum_critical_decided": 8,
    "maximum_likeness_mae": 1.0,
    "margins": [0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.90, 0.95],
    "masses": [0.05, 0.20],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_data(path=REVIEW):
    manifest_path = path.with_suffix(".json")
    manifest = read_json(manifest_path)
    if manifest.get("review_status") != "complete_assisted":
        raise ValueError("La revisión asistida no está completa.")
    if manifest["output_hashes"].get(str(path.resolve())) != sha(path):
        raise ValueError("Cambió la copia revisada; se requiere una revisión versionada.")
    provenance = {r["item_id"]: r for r in manifest["items"]}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["item_id"] for r in rows}) != len(rows) or set(provenance) != {
        r["item_id"] for r in rows
    }:
        raise ValueError("Identificadores de revisión inválidos.")
    examples = []
    for row in rows:
        if row["etiqueta_revision"] not in LABELS:
            raise ValueError("Etiqueta de revisión vacía o desconocida.")
        examples.append(
            {
                "id": row["item_id"],
                "source_group": provenance[row["item_id"]]["source_group"],
                "requirement": row["requisito"],
                "candidate": row["relato"],
                "expected": LABELS[row["etiqueta_revision"]],
                "reviewer": row["origen_revision"],
                "dimension": row["dimension"],
            }
        )
    return examples, {
        str(path.resolve()): sha(path),
        str(manifest_path.resolve()): sha(manifest_path),
    }


def split_sources(examples):
    groups = sorted({r["source_group"] for r in examples}, key=lambda g: digest(["pilot-v1", g]))
    # Estratificación por clase antes de observar probabilidades: los dos grupos
    # con contradicciones se reparten entre ajuste y comprobación.
    contra = {r["source_group"] for r in examples if r["expected"] == "contradicted"}
    ordered = [g for g in groups if g in contra] + [g for g in groups if g not in contra]
    return {g: "calibration" if i % 2 == 0 else "validation" for i, g in enumerate(ordered)}


def metrics(records, margin, mass):
    eligible = [r for r in records if r["expected"] is not None]
    decisions = [
        {
            **r,
            **decide(r["probabilities"], r["choice_mass"], margin=margin, min_choice_mass=mass),
        }
        for r in eligible
    ]
    decided = [r for r in decisions if r["status"] != "abstain"]
    correct = sum(r["status"] == r["expected"] for r in decided)
    return {
        "n": len(eligible),
        "decided": len(decided),
        "correct": correct,
        "wrong": len(decided) - correct,
        "accuracy": correct / len(decided) if decided else None,
        "coverage": len(decided) / len(eligible) if eligible else 0,
        "items": decisions,
    }


def select_thresholds(records, critical):
    candidates = []
    calibration = [r for r in records if r["split"] == "calibration"]
    for margin in CRITERIA["margins"]:
        for mass in CRITERIA["masses"]:
            m, c = metrics(calibration, margin, mass), metrics(critical, margin, mass)
            modes = by_mode(calibration, margin, mass)
            if (
                m["accuracy"] is not None
                and m["accuracy"] >= CRITERIA["calibration_accuracy"]
                and m["coverage"] >= CRITERIA["minimum_coverage"]
                and c["wrong"] == 0
                and c["decided"] >= CRITERIA["minimum_critical_decided"]
                and all(
                    x["accuracy"] is not None
                    and x["accuracy"] >= CRITERIA["calibration_accuracy"]
                    and x["coverage"] >= CRITERIA["minimum_coverage"]
                    for x in modes.values()
                )
            ):
                candidates.append((m["coverage"], -margin, -mass, margin, mass))
    return max(candidates)[-2:] if candidates else None


def by_mode(records, margin, mass):
    groups = {}
    for row in records:
        groups.setdefault(
            "fidelity" if row.get("dimension") == "fidelidad" else "compliance", []
        ).append(row)
    return {mode: metrics(rows, margin, mass) for mode, rows in groups.items()}


def eligible_report(report):
    calibration, validation, critical = (
        report.get(k) for k in ["calibration", "validation", "critical"]
    )
    likeness = report.get("likeness", {})
    modes = report.get("by_mode", {})
    modes_passed = set(modes) == {"calibration", "validation"} and all(
        set(modes[phase]) == {"compliance", "fidelity"}
        and all(
            m["accuracy"] is not None
            and m["accuracy"] >= CRITERIA[f"{phase}_accuracy"]
            and m["coverage"] >= CRITERIA["minimum_coverage"]
            for m in modes[phase].values()
        )
        for phase in modes
    )
    return bool(
        modes_passed
        and calibration
        and validation
        and critical
        and calibration["accuracy"] is not None
        and calibration["accuracy"] >= CRITERIA["calibration_accuracy"]
        and calibration["coverage"] >= CRITERIA["minimum_coverage"]
        and validation["accuracy"] is not None
        and validation["accuracy"] >= CRITERIA["validation_accuracy"]
        and validation["coverage"] >= CRITERIA["minimum_coverage"]
        and validation["decided"] >= CRITERIA["minimum_validation_decided"]
        and critical["wrong"] == 0
        and critical["decided"] >= CRITERIA["minimum_critical_decided"]
        and likeness.get("passed")
        and likeness["mae"] <= CRITERIA["maximum_likeness_mae"]
        and likeness["sanity"]["passed"]
        and report.get("judge_unchanged")
    )


def require_calibration(spec, environment=None):
    options = spec["classification"]
    path = options.get("calibration_file")
    if not path:
        raise ValueError("Falta calibración del juez cerrado con las referencias anotadas.")
    report = read_json(Path(path))
    if (
        not report.get("passed")
        or not eligible_report(report)
        or report.get("protocol") != "assisted-pilot-v3"
        or report["classifier_signature"] != classifier_signature(spec)
        or report["thresholds"]
        != {"margin": options["margin"], "min_choice_mass": options["min_choice_mass"]}
        or report["criteria"] != CRITERIA
    ):
        raise ValueError("La calibración no aprueba o no corresponde al juez y umbrales actuales.")
    if environment is not None and report["environment"] != environment:
        raise ValueError("El entorno del juez difiere del entorno calibrado.")
    for source, expected in report["review_sources"].items():
        if sha(source) != expected:
            raise ValueError("Las referencias revisadas cambiaron después de calibrar.")
    return report


def calibrate(config, root, *, review_file=None, likeness_review_file=None):
    from .anchor_holdout import ANCHORS, CASES
    from .backend import fingerprint
    from .calibration_transfer import examples as prospective_examples
    from .corpus import prepare
    from .judge import Judge

    folder = root / "dream/calibration"
    previous = folder / "calibration.json"
    if previous.exists():
        old = read_json(previous)
        write_json(
            folder / "history" / f"{old['protocol']}-{old['classifier_signature'][:12]}.json", old
        )
    examples, sources = review_data(review_file or REVIEW)
    likeness_path = likeness_review_file or Path("results/dream/report/revision-asistente.csv")
    sources[str(likeness_path.resolve())] = sha(likeness_path)
    assignments = {r["source_group"]: "calibration" for r in examples}
    prospective = prospective_examples()
    assignments.update({r["source_group"]: "validation" for r in prospective})
    examples += prospective
    fixture = Path(__file__).with_name("calibration_transfer.py")
    sources[str(fixture.resolve())] = sha(fixture)
    write_json(
        folder / "design.json",
        {"criteria": CRITERIA, "review_sources": sources, "assignments": assignments},
    )
    calibration_config = deepcopy(config)
    calibration_config["judge"]["classification"]["calibration_file"] = None
    judge = Judge(calibration_config, root)
    before = fingerprint(judge.model.parameters())
    scorer = ClosedJudge(judge)
    records, critical = [], []

    def observe(requirement, candidate, *, mode="compliance"):
        result = scorer.classify(requirement, candidate, mode=mode)
        return {k: result[k] for k in ["probabilities", "choice_mass", "rotation_keys"]}

    for example in examples:
        if example["expected"] is None:
            continue
        records.append(
            {
                **example,
                "split": assignments[example["source_group"]],
                **observe(
                    example["requirement"],
                    example["candidate"],
                    mode="fidelity" if example["dimension"] == "fidelidad" else "compliance",
                ),
            }
        )
        write_json(folder / "partial.json", {"records": records, "training_enabled": False})
        print(
            "Calibración:",
            len(records),
            "/",
            sum(r["expected"] is not None for r in examples),
            flush=True,
        )
    for case in ["injection", "roles", "emotion"]:
        candidate, expected = CASES[case]
        for anchor, state in zip(ANCHORS, expected, strict=True):
            critical.append(
                {
                    "id": f"{case}-{anchor['id']}",
                    "expected": state,
                    **observe(anchor["requirement"], candidate),
                }
            )
    selected = select_thresholds(records, critical)
    thresholds = {"margin": selected[0], "min_choice_mass": selected[1]} if selected else None
    validation = (
        metrics([r for r in records if r["split"] == "validation"], *selected) if selected else None
    )
    passed = bool(
        validation
        and validation["accuracy"] is not None
        and validation["accuracy"] >= CRITERIA["validation_accuracy"]
        and validation["coverage"] >= CRITERIA["minimum_coverage"]
        and validation["decided"] >= CRITERIA["minimum_validation_decided"]
    )
    likeness_rows = []
    with likeness_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            actual = judge.likeness(row["relato"])["dream_likeness"]
            expected = int(row["onirico_asistente_0_a_4"])
            likeness_rows.append(
                {
                    "id": row["item_id"],
                    "expected": expected,
                    "actual": actual,
                    "absolute_error": abs(expected - actual),
                }
            )
    likeness_mae = sum(r["absolute_error"] for r in likeness_rows) / len(likeness_rows)
    data, _ = prepare(config, root)
    try:
        sanity = judge.calibrate(data["validation"][:3])
    except RuntimeError:
        sanity = read_json(root / "dream/judge-calibration.json")
    oniric_passed = likeness_mae <= CRITERIA["maximum_likeness_mae"] and sanity["passed"]
    unchanged = before == fingerprint(judge.model.parameters())
    report = {
        "protocol": "assisted-pilot-v3",
        "passed": passed and unchanged and oniric_passed,
        "criteria": CRITERIA,
        "thresholds": thresholds,
        "validation": validation,
        "calibration": metrics([r for r in records if r["split"] == "calibration"], *selected)
        if selected
        else None,
        "critical": metrics(critical, *selected) if selected else None,
        "records": records,
        "by_mode": {
            phase: by_mode([r for r in records if r["split"] == phase], *selected)
            for phase in ["calibration", "validation"]
        }
        if selected
        else {},
        "critical_records": critical,
        "review_sources": sources,
        "assignments": assignments,
        "classifier_signature": scorer.signature,
        "environment": scorer.environment,
        "judge_unchanged": unchanged,
        "independent_human_validation": False,
        "unknown_review_items_excluded": len(examples) - len(records),
        "likeness": {
            "mae": likeness_mae,
            "items": likeness_rows,
            "sanity": sanity,
            "passed": oniric_passed,
        },
        "training_enabled": passed and unchanged and oniric_passed,
        "limitation": "Criterios operativos de piloto exploratorio, muestra pequeña y revisión asistida. No prueba precisión general ni validez científica independiente.",
    }
    report["passed"] = eligible_report(report)
    report["training_enabled"] = report["passed"]
    write_json(folder / "calibration.json", report)
    print("Calibración operativa:", report["passed"], "umbrales:", thresholds, flush=True)
