"""Controles de alineación y fidelidad con pares conocidos, sin entrenar al juez."""

from .corpus import prepare
from .judge import Judge
from .storage import digest, read_json, write_json


def verify_reference(config, root, judge=None):
    data, manifest = prepare(config, root)
    target = data["validation"][0]["reference"]
    words = set(target.casefold().split())
    other = min(
        data["train"][:100],
        key=lambda r: (
            len(words & set(r["reference"].casefold().split()))
            / max(1, len(words | set(r["reference"].casefold().split())))
        ),
    )["reference"]
    identity = digest([config["judge"], manifest["hash"], target, other])
    path = root / "dream" / "reference-calibration.json"
    if path.exists() and read_json(path)["identity"] == identity:
        result = read_json(path)
    else:
        judge = Judge(config, root) if judge is None else judge
        same = judge.alignment(target, target, target)
        unrelated = judge.alignment(target, target, other)
        switched_draft = judge.alignment(target, other, target)
        passed = (
            same["reference_alignment"] >= 3
            and same["retelling_fidelity"] >= 3
            and unrelated["reference_alignment"] <= 2
            and unrelated["retelling_fidelity"] <= 2
            and switched_draft["reference_alignment"] >= 3
            and switched_draft["retelling_fidelity"] <= 2
        )
        result = {
            "identity": identity,
            "passed": passed,
            "identical": same,
            "unrelated": unrelated,
            "switched_draft": switched_draft,
            "limitation": "Controles sintéticos de cordura; no son una validación humana ni prueban sensibilidad a toda paráfrasis.",
        }
        write_json(path, result)
    print("Controles de referencia y fidelidad:", result, flush=True)
    if not result["passed"]:
        raise RuntimeError(
            "El juez no pasó los controles de referencia/fidelidad; no iniciar comparación larga."
        )
