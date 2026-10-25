"""Informe local sin inferencia: distingue datos, diagnóstico y aprendizaje pendiente."""

from .storage import digest, read_json, read_jsonl, write_json


def report(root):
    folder = root / "dream"
    corpus = read_json(folder / "data/manifest.json")
    rows = {s: read_jsonl(folder / f"data/{s}.jsonl") for s in ("train", "validation", "test")}
    if digest(rows) != corpus["hash"]:
        raise ValueError("El corpus del informe no coincide con el manifiesto.")
    paths = {
        "embeddings": folder / "embeddings/diagnostic.json",
        "base": folder / "base-check/latest.json",
        "pilot": folder / "smoke/run.json",
        "training": folder / "smoke/training-validation.json",
        "reload": folder / "smoke/reload-validation.json",
    }
    evidence = {k: read_json(p) if p.exists() else None for k, p in paths.items()}
    checkpoint = folder / "smoke/checkpoints/latest.json"
    completed = 0
    if checkpoint.exists():
        latest = read_json(checkpoint)
        completed = read_json(checkpoint.parent / latest["directory"] / "state.json")["step"]
    pilot = evidence["pilot"] or {}
    summary = {
        "corpus_hash": corpus["hash"],
        "counts": corpus["counts"],
        "evidence_hashes": {k: digest(v) for k, v in evidence.items() if v is not None},
        "steps_completed": completed,
        "steps_requested": pilot.get("steps_requested"),
        "pilot_status": pilot.get("status", "not_started"),
        "operational_only": True,
        "subjective_experience_claimed": False,
    }
    lines = [
        "# Dream: estado del protocolo DreamBank",
        "",
        "Fuente: DreamBank, CC BY-NC-SA 4.0, atribución a dreambank.net. Código MIT.",
        "",
        f"Corpus: {corpus['counts']}. Banco de recompensa: sólo entrenamiento.",
        "",
        "## Recompensa exploratoria",
        "",
        (
            "Se puntúa `<answer>` con `(1 + máximo coseno al banco) / 2`. "
            "La recompensa alcanza toda la secuencia. `<dream>` queda libre; "
            "no hay penalización automática por similitud alta o coincidencias."
        ),
        "",
    ]
    embedding = evidence["embeddings"]
    if embedding:
        lines += [
            (
                f"Diagnóstico operativo: {embedding['status']}; banco de "
                f"{embedding['bank_size']} referencias."
            ),
            "",
            "| Entrada | Recompensa media |",
            "| --- | ---: |",
        ]
        lines += [f"| {k} | {v['total']:.4f} |" for k, v in embedding["summaries"].items()]
        lines += [
            "",
            (
                "Los grupos tienen distintos referentes. Estos valores no validan "
                "oniricidad; listas temáticas también pueden puntuar alto."
            ),
            "",
        ]
    base = evidence["base"]
    if base:
        lines += [
            "## Generador base",
            "",
            (
                f"Prueba sin ejemplo: {base['summary']}. "
                f"Pesos base intactos: {base['base_unchanged']}."
            ),
            "",
        ]
    lines += [
        "## Piloto",
        "",
        (
            f"Estado: {summary['pilot_status']}. Actualizaciones con "
            f"checkpoint completo: {completed}/{summary['steps_requested']}."
        ),
        "",
        (
            "No se infiere mejora por completar pasos. La comparación antes/después, "
            "la recarga y la recuperación deben comprobarse en sus artefactos."
        ),
        "",
        "## Diseño abierto",
        "",
        (
            "La riqueza de las semillas sigue abierta. El límite inicial es una prueba, "
            "no una conclusión. Cercanía al relato de origen y cercanía al banco son "
            "diagnósticos distintos; sólo la segunda integra la recompensa actual."
        ),
        "",
    ]
    target = folder / "report"
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(target / "summary.json", summary)
    print("Informe DreamBank:", target / "report.md")
    return summary
