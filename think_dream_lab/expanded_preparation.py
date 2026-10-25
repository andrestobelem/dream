"""Preparación revisable de 128 relatos; ninguna actualización de pesos."""

from collections import defaultdict
from pathlib import Path

from .dreambank_task import render_prompt
from .holdout_evaluation import NaturalSeedWriter, validate_brief
from .storage import digest, read_json, read_jsonl, write_json


def balanced_selection(train, retained, reserved_groups, reserved_ids, count=128, seed=20261025):
    groups = defaultdict(list)
    for row in train:
        if (
            row["split"] != "train"
            or row["author_group"] in reserved_groups
            or row["problem_id"] in reserved_ids
        ):
            raise ValueError("La partición de entrenamiento contiene un reservado.")
        groups[row["author_group"]].append(row)
    if not groups or len({r["problem_id"] for r in train}) != len(train):
        raise ValueError("Partición vacía o con identificadores duplicados.")
    base, extra = divmod(count, len(groups))
    ordered = sorted(groups, key=lambda g: digest([seed, g]))
    quotas = {g: base + int(i < extra) for i, g in enumerate(ordered)}
    byid = {r["problem_id"]: r for r in train}
    kept = {r["problem_id"]: r for r in retained}
    if len(kept) != len(retained) or not set(kept) <= set(byid):
        raise ValueError("Los relatos retenidos no pertenecen al entrenamiento.")
    for pid, row in kept.items():
        if (
            row["reference"] != byid[pid]["reference"]
            or row["author_group"] != byid[pid]["author_group"]
        ):
            raise ValueError("Cambió un relato retenido.")
    result = []
    for group in sorted(groups):
        values = groups[group]
        keep = [r for r in values if r["problem_id"] in kept]
        quota = quotas[group]
        if len(values) < quota or len(keep) > quota:
            raise ValueError("No se puede cumplir la cuota sin sustituir relatos.")
        rest = sorted(
            [r for r in values if r["problem_id"] not in kept],
            key=lambda r: digest([seed, r["problem_id"]]),
        )
        result.extend(sorted(keep, key=lambda r: r["problem_id"]) + rest[: quota - len(keep)])
    if len(result) != count or len({digest(r["reference"]) for r in result}) != count:
        raise ValueError("Selección incompleta o con referencias duplicadas.")
    return result, quotas


def select(config, root):
    spec = config["expanded_preparation"]
    corpus = Path(spec["corpus_directory"])
    rows = {s: read_jsonl(corpus / f"{s}.jsonl") for s in ("train", "validation", "test")}
    if digest(rows) != read_json(corpus / "manifest.json")["hash"]:
        raise ValueError("Cambió el corpus congelado.")
    from .paired_experiment import load_pack

    if Path(spec["source_pack"]) != Path(config["training_pack"]["path"]):
        raise ValueError("El origen retenido no coincide con el lote aprobado.")
    old = load_pack(config)
    heldout = rows["validation"] + rows["test"]
    chosen, quotas = balanced_selection(
        rows["train"],
        old["rows"]["train"],
        {r["author_group"] for r in heldout},
        {r["problem_id"] for r in heldout},
        spec["reports"],
        spec["selection_seed"],
    )
    selection = {
        "rows": chosen,
        "rows_hash": digest(chosen),
        "quotas": quotas,
        "corpus_hash": digest(rows),
        "retained_ids": [r["problem_id"] for r in old["rows"]["train"]],
        "rule": "retain approved eight, deterministic quotas four/five per training author, no score-based selection",
        "config": spec,
        "reserved_people_excluded": True,
        "reserved_reports_excluded": True,
        "training_updates": 0,
    }
    path = root / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise ValueError("La selección existente tiene otra identidad.")
    write_json(path, selection)
    return selection, old


def prepare(config, root):
    selection, old = select(config, root)
    writer = NaturalSeedWriter(config["seed_extractor"], root)
    retained = {r["problem_id"]: r for r in old["rows"]["train"]}
    rows = []
    failures = []
    for index, row in enumerate(selection["rows"]):
        try:
            if row["problem_id"] in retained:
                brief = validate_brief({"brief": retained[row["problem_id"]]["seeds"]})
                origin = "previously reviewed and approved v6 seed"
            else:
                brief = writer.extract(row)
                origin = "frozen extractor, pending assistant review"
            words = len(row["reference"].split())
            prompt = render_prompt(
                brief, response_prefix=config["generation"]["response_prefix"], target_words=words
            )
            if row["reference"] in prompt:
                raise ValueError("La referencia completa no puede estar en el prompt.")
            rows.append(
                {
                    **row,
                    "reference_hash": digest(row["reference"]),
                    "seeds": brief,
                    "target_words": words,
                    "prompt": prompt,
                    "seed_origin": origin,
                }
            )
        except (ValueError, TypeError) as error:
            failures.append({"problem_id": row["problem_id"], "error": str(error)})
        write_json(
            root / "extraction-progress.json",
            {
                "rows": rows,
                "failures": failures,
                "selected_hash": selection["rows_hash"],
                "extractor_identity": writer.identity,
                "training_updates": 0,
                "status": "pending_review",
            },
        )
        print(
            f"Semillas: {index + 1}/{len(selection['rows'])}; pendientes de reparación: {len(failures)}",
            flush=True,
        )
    return {"prepared": len(rows), "pending_repairs": len(failures), "training_updates": 0}


def finalize(config, root):
    """Congelar revisión explícita; no modifica la selección ni los ocho originales."""
    import html
    import json

    from .paired_experiment import load_pack

    selection, old = select(config, root)
    approved = load_pack(config)
    if old != approved:
        raise ValueError("El origen aprobado cambió.")
    progress = read_json(root / "extraction-progress.json")
    if progress["failures"] or len(progress["rows"]) != 128:
        raise ValueError("La extracción debe estar completa, sin sustituir casos.")
    expected = {r["problem_id"]: r for r in selection["rows"]}
    if {r["problem_id"] for r in progress["rows"]} != set(expected):
        raise ValueError("Cambió la selección durante la extracción.")
    overrides = {}
    for file in sorted(root.glob("review-batch*.json")):
        for prefix, brief in read_json(file).items():
            matches = [pid for pid in expected if pid.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError("Identificador de revisión ambiguo.")
            overrides[matches[0]] = validate_brief({"brief": brief})
    reviewed = read_json(root / "review-coverage.json")
    if set(reviewed["reviewed_ids"]) != set(expected):
        raise ValueError("Falta revisar alguna referencia.")
    retained = {r["problem_id"]: r for r in approved["rows"]["train"]}
    if set(overrides) & set(retained):
        raise ValueError("No se reescriben las ocho semillas aprobadas.")
    rows = []
    normalizations = []
    for candidate in progress["rows"]:
        row = dict(candidate)
        original = expected[row["problem_id"]]["reference"]
        if row["reference"] != original:
            raise ValueError("La extracción cambió una referencia.")
        prefix = row["problem_id"][:12]
        spans = [(0, len(original))]
        reason = None
        if prefix == "5bad7e950959":
            spans = [(original.index("This morning I was having little dreams"), len(original))]
            reason = "omitir preámbulo de diario previo al relato soñado"
        elif prefix == "7096b6a80509":
            spans = [(0, original.index(" Q: Did you notice"))]
            reason = "omitir entrevista posterior al despertar"
        elif prefix == "1f8e8834e0f3":
            spans = [
                (0, original.index("I was in bed with one of my daughters.")),
                (original.index("Then Audrey asked me"), len(original)),
            ]
            reason = (
                "usar sólo episodios no sexuales; omitir segmento sexual intrafamiliar explícito"
            )
        reference = "\n\n".join(original[a:b].strip() for a, b in spans)
        row["reference"] = reference
        row["reference_hash"] = digest(reference)
        row["source_reference_hash"] = digest(original)
        row["reference_spans"] = spans
        if reason:
            row["reference_normalization"] = reason
            normalizations.append(
                {
                    "problem_id": row["problem_id"],
                    "reason": reason,
                    "source_hash": digest(original),
                    "target_hash": digest(reference),
                    "spans": spans,
                }
            )
        row["extractor_brief"] = row["seeds"]
        row["seeds"] = overrides.get(row["problem_id"], row["seeds"])
        row["seed_review"] = "assistant source comparison; not independent human review"
        row["target_words"] = len(reference.split())
        row["prompt"] = render_prompt(
            row["seeds"],
            response_prefix=config["generation"]["response_prefix"],
            target_words=row["target_words"],
        )
        validate_brief({"brief": row["seeds"]})
        if reference in row["prompt"] or original in row["prompt"]:
            raise ValueError("La referencia completa aparece en el prompt.")
        if row["problem_id"] in retained:
            previous = retained[row["problem_id"]]
            for key in ["reference", "seeds", "target_words", "prompt"]:
                if row[key] != previous[key]:
                    raise ValueError("Cambió uno de los ocho ejemplos originales.")
        rows.append(row)
    validation = read_json(Path(config["expanded_preparation"]["validation_pack"]))
    if digest(validation["rows"]) != validation["rows_hash"]:
        raise ValueError("La validación congelada cambió.")
    if {r["author_group"] for r in rows} & {r["author_group"] for r in validation["rows"]}:
        raise ValueError("Una persona cruza entrenamiento y validación.")
    pack = {
        "schema": "expanded-dream-training-pack-v1",
        "rows": {"train": rows, "validation": validation["rows"]},
        "dataset_hash": digest({"train": rows, "validation": validation["rows"]}),
        "selection_hash": digest(selection),
        "source_selection_rows_hash": selection["rows_hash"],
        "seed_review": reviewed,
        "assistant_overrides": len(overrides),
        "normalizations": normalizations,
        "training_updates": 0,
        "status": "prepared_for_review_not_trained",
        "reference_in_actor_prompt": False,
        "retained_original_reports": len(retained),
        "note": "El perfil de preparación rechaza entrenamiento. No se cambió la recompensa ni se entrenó sobre reservados.",
    }
    write_json(root / "training-pack.json", pack)
    with (root / "prompts-only.jsonl").open("w") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "problem_id",
                            "author_group",
                            "split",
                            "seeds",
                            "target_words",
                            "prompt",
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    cards = []
    for index, row in enumerate(rows, 1):
        note = row.get("reference_normalization", "")
        cards.append(
            f'<article><h2>{index:03d} · {row["problem_id"][:12]}</h2><p class="meta">Persona {row["author_group"][:8]} · objetivo {row["target_words"]} palabras · semilla {len(row["seeds"].split())} palabras</p><p class="seed">{html.escape(row["seeds"])}</p><details><summary>Prompt completo — sin referencia humana</summary><pre>{html.escape(row["prompt"])}</pre></details><details><summary>Referencia esperada — sólo para evaluar</summary><pre>{html.escape(row["reference"])}</pre></details><p class="note">{html.escape(note)}</p></article>'
        )
    document = (
        """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · lote de 128 relatos</title><style>body{font:17px/1.55 system-ui,sans-serif;background:#f6f4ed;color:#202720;margin:0}main{max-width:960px;margin:auto;padding:32px 20px}h1{font-size:34px;line-height:1.15}h2{font-size:19px}.meta,.note{color:#526052;font-size:14px}.seed{font-size:19px}article{background:white;border:1px solid #d9ded2;border-radius:12px;padding:20px;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 ui-monospace,monospace}summary{cursor:pointer;padding:8px 0}input{box-sizing:border-box;width:100%;padding:12px;font:inherit;border:1px solid #809080;border-radius:6px}a{color:#28552e}.banner{padding:16px;background:#e2ebda;border-radius:8px}</style><main><h1>Dream · lote de 128 relatos</h1><p class="banner">Preparado para revisión. <strong>No se entrenó.</strong> Ocho relatos anteriores + 120 nuevos; 30 personas de entrenamiento, cuatro o cinco relatos por persona. Los reservados quedan aparte.</p><p>Las semillas fueron revisadas contra sus fuentes por el asistente; esto no es revisión humana independiente. La referencia completa queda fuera del prompt. Tres referencias tienen recortes explícitos y trazables; los originales se conservan en selection.json.</p><p><a href="training-pack.json">Lote completo</a> · <a href="prompts-only.jsonl">Sólo prompts</a> · <a href="../report/diagnostico.md">Diagnóstico de longitud</a></p><label>Buscar por texto o identificador<input id="search" type="search" placeholder="Ej.: escuela, elevator, 001"></label><p id="count">128 relatos</p>"""
        + "".join(cards)
        + """</main><script>document.querySelector('#search').addEventListener('input',e=>{let count=0;const q=e.target.value.toLowerCase();document.querySelectorAll('article').forEach(a=>{a.hidden=!a.textContent.toLowerCase().includes(q);if(!a.hidden)count++});document.querySelector('#count').textContent=count+' relatos'});</script></html>"""
    )
    (root / "revision.html").write_text(document)
    return pack
