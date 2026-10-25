"""Informe local sin inferencia: distingue datos, diagnóstico y aprendizaje pendiente."""

from .storage import digest, read_json, read_jsonl, write_json


def cached_source_similarity(root, rows, pilot):
    """Diagnóstico respecto del relato de origen; nunca ejecuta inferencia ni cambia r."""
    import numpy as np

    from .dream_task import extract_blocks
    from .embedding_reward import Encoder

    if not pilot or "config" not in pilot:
        return []
    encoder = Encoder(pilot["config"]["embeddings"], root)
    references = {r["problem_id"]: r["reference"] for values in rows.values() for r in values}
    result = []
    for record in read_jsonl(root / "dream/smoke/responses.jsonl"):
        prefix = record["reward"].get("response_prefix_in_prompt", "")
        blocks = extract_blocks(prefix + record["text"], "D")
        if not blocks["format"]:
            continue
        source = references[record["problem_id"]]
        answer = blocks["answer"]
        if any(digest(text) not in encoder.cache for text in (answer, source)):
            continue  # El informe no carga modelos para completar datos ausentes.
        vectors = encoder.encode([answer, source])
        result.append(
            {
                "problem_id": record["problem_id"],
                "phase": record["phase"],
                "step": record["step"],
                "candidate": record["candidate"],
                "source_cosine": float(np.clip(vectors[0] @ vectors[1], -1, 1)),
                "bank_cosine": record["reward"]["cosine"],
            }
        )
    return result


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
        "continuity": folder / "smoke/continuity-validation.json",
    }
    evidence = {k: read_json(p) if p.exists() else None for k, p in paths.items()}
    checkpoint = folder / "smoke/checkpoints/latest.json"
    completed = 0
    if checkpoint.exists():
        latest = read_json(checkpoint)
        completed = read_json(checkpoint.parent / latest["directory"] / "state.json")["step"]
    pilot = evidence["pilot"] or {}
    source_similarity = cached_source_similarity(root, rows, pilot)
    summary = {
        "corpus_hash": corpus["hash"],
        "counts": corpus["counts"],
        "evidence_hashes": {k: digest(v) for k, v in evidence.items() if v is not None},
        "steps_completed": completed,
        "steps_requested": pilot.get("steps_requested"),
        "pilot_status": pilot.get("status", "not_started"),
        "operational_only": True,
        "subjective_experience_claimed": False,
        "source_similarity_diagnostic": source_similarity,
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
    if evidence["training"]:
        train = evidence["training"]
        lines += [
            "## Comprobaciones del entrenamiento",
            "",
            (
                f"Adaptadores modificados: {train['adapters_changed']}. "
                f"Pesos base intactos: {train['base_unchanged']}."
            ),
            "",
        ]
    if evidence["reload"]:
        lines += [f"Recarga idéntica: {evidence['reload']['same_weights']}.", ""]
    if evidence["continuity"]:
        lines += [f"Continuidad real: {evidence['continuity']['status']}.", ""]
    if source_similarity:
        lines += [
            "## Relato de origen frente al banco",
            "",
            "Diagnóstico calculado sólo con vectores ya guardados; no cambia la recompensa.",
            "",
            "| Paso | Candidato | Coseno al origen | Coseno al banco |",
            "| ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {r['step']} | {r['candidate']} | {r['source_cosine']:.4f} | "
            f"{r['bank_cosine']:.4f} |"
            for r in source_similarity
        ]
        lines.append("")
    target = folder / "report"
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(target / "summary.json", summary)
    print("Informe DreamBank:", target / "report.md")
    return summary
