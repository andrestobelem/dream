"""Contrato visible y evaluación por elementos para dream-v3."""

import statistics

from .judge import GUARD, protected_json

VERSION = "dream-v3"
PREFIX = "Required content: "
KINDS = {"participant", "setting", "event", "relationship", "emotion"}
EXTRACT = GUARD + (
    "Extract 3 to 6 short, atomic requirements for generating a similar dream. "
    "Use only explicitly stated facts. Prefer central events and relationships, not incidental details. "
    "Each requirement must be independently checkable; do not bundle multiple events. "
    "Include an exact contiguous source quote supporting each requirement. "
    "Do not infer emotions or relationships. Use English. "
    'Return {"anchors":[{"id":"a1","kind":"event","requirement":"brief proposition",'
    '"source_quote":"exact source words"}]}. '
    "Kinds: participant, setting, event, relationship, emotion. IDs must be a1, a2, etc."
)
ASSESS = GUARD + (
    "Evaluate ONLY the supplied requirements against the candidate. There is no hidden reference. "
    "For each ID return preserved, omitted, or contradicted. Preserved means semantically present, "
    "including paraphrase. Omitted means absent or ambiguous. Contradicted requires explicit "
    "incompatible evidence, not mere absence. Track who does what to whom, negation and emotion. "
    "Compatible additions and wording changes must not reduce any rating. "
    "Semantic equivalents and synonyms count as preserved. Different nouns or verbs alone "
    "are not a contradiction: the evidence must assert a genuinely incompatible proposition. "
    "A vocabulary list or metalinguistic mention asserts no experienced facts: its words alone "
    "preserve no requirements. Preserved requires an assertion within the candidate account. "
    "First locate evidence for each requirement. If no supporting or explicitly incompatible "
    "passage exists, the status MUST be omitted. An unstated emotion is omitted, not contradicted. "
    "Example: requirement 'the cat is black', candidate 'a cat sat nearby': omitted, empty evidence. "
    "Candidate 'the cat was white, not black': contradicted, quote that passage. "
    "For preserved/contradicted include an exact contiguous quote from candidate as evidence; "
    "for omitted use an empty evidence string. Do not rate style, length, realism or dream quality. "
    "For each requirement first give a short reason describing what the candidate explicitly "
    "says about it or that it is unstated; then choose status. Do not infer a feeling from an event. "
    'Return {"items":[{"id":"a1","reason":"brief explanation",'
    '"status":"preserved","evidence":"exact candidate words"}]} '
    "with exactly one item for every supplied ID and no others."
)


def validate_anchors(items, reference=None):
    if not isinstance(items, list) or not 3 <= len(items) <= 6:
        raise ValueError("El contrato requiere entre tres y seis elementos.")
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict) or set(item) != {"id", "kind", "requirement", "source_quote"}:
            raise ValueError("Esquema de elemento inválido.")
        if item["id"] != f"a{i}" or item["kind"] not in KINDS:
            raise ValueError("ID o tipo de elemento inválido.")
        if any(
            not isinstance(item[k], str) or not item[k].strip()
            for k in ["requirement", "source_quote"]
        ):
            raise ValueError("Elemento vacío o inválido.")
        if len(item["requirement"].split()) > 25:
            raise ValueError("Requisito demasiado extenso.")
        if reference is not None and item["source_quote"] not in reference:
            raise ValueError("La evidencia del requisito no existe en la referencia.")
    return items


def contract_brief(judge, row):
    value = judge.call("anchor_extraction", EXTRACT, {"reference": row["reference"]})
    if set(value) != {"anchors"}:
        raise ValueError("Extracción de elementos inválida.")
    items = validate_anchors(value["anchors"], row["reference"])
    # Las citas de procedencia quedan en el registro del juez, fuera de la consigna.
    visible = [{k: item[k] for k in ["id", "kind", "requirement"]} for item in items]
    return PREFIX + protected_json(
        {"version": getattr(judge, "spec", {}).get("rubric_version", VERSION), "anchors": visible}
    )


def visible_anchors(brief):
    from .judge import parse_json

    if not isinstance(brief, str) or not brief.startswith(PREFIX):
        raise ValueError("Falta la consigna con contrato visible dream-v3.")
    contract = parse_json(brief[len(PREFIX) :])
    if set(contract) != {"version", "anchors"} or contract["version"] not in {VERSION, "dream-v4"}:
        raise ValueError("Versión de contrato inválida.")
    items = contract["anchors"]
    if not isinstance(items, list) or any(
        not isinstance(item, dict) or set(item) != {"id", "kind", "requirement"} for item in items
    ):
        raise ValueError("Contrato visible inválido.")
    validate_anchors([{**item, "source_quote": "not evaluated"} for item in items])
    return items


def validate_assessment(value, anchors, candidate):
    if set(value) != {"items"} or not isinstance(value["items"], list):
        raise ValueError("Evaluación por elementos inválida.")
    items = value["items"]
    expected = {a["id"] for a in anchors}
    if len(items) != len(expected):
        raise ValueError("Faltan o sobran elementos evaluados.")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) not in (
            {"id", "status", "evidence"},
            {"id", "reason", "status", "evidence"},
        ):
            raise ValueError("Esquema de evaluación inválido.")
        if "reason" in item and (not isinstance(item["reason"], str) or not item["reason"].strip()):
            raise ValueError("Explicación inválida.")
        key, status, evidence = item["id"], item["status"], item["evidence"]
        if not isinstance(key, str) or key not in expected or key in seen:
            raise ValueError("ID ausente, duplicado o desconocido.")
        seen.add(key)
        if status not in {"preserved", "omitted", "contradicted"}:
            raise ValueError("Estado de elemento inválido.")
        if not isinstance(evidence, str):
            raise TypeError("Evidencia inválida.")
        if status == "omitted":
            if evidence:
                raise ValueError("Una omisión no lleva evidencia positiva.")
        elif not evidence.strip() or evidence not in candidate:
            raise ValueError("La evidencia no existe en el relato evaluado.")
    counts = {
        s: sum(i["status"] == s for i in items) for s in ["preserved", "omitted", "contradicted"]
    }
    # Pesos iguales fijados antes de entrenar; contradicción peor que omisión.
    score = max(0.0, (counts["preserved"] - 0.5 * counts["contradicted"]) / len(items))
    return {"score": score, "counts": counts, "items": items}


def assess(judge, brief, candidate):
    anchors = visible_anchors(brief)
    if getattr(judge, "spec", {}).get("evidence_mode") == "passage_selection":
        from .evidence import assess_selection

        value, provenance = assess_selection(judge, anchors, candidate)
        return {**validate_assessment(value, anchors, candidate), "evidence_selection": provenance}
    if not candidate.strip():
        value = {"items": [{"id": a["id"], "status": "omitted", "evidence": ""} for a in anchors]}
    else:
        # Un requisito por llamada evita que el veredicto de otro contamine éste.
        items = []
        for anchor in anchors:
            payload = {"requirements": [anchor], "candidate": candidate}
            partial = judge.call(
                "content_compliance",
                ASSESS,
                payload,
            )
            try:
                validate_assessment(partial, [anchor], candidate)
            except (ValueError, TypeError) as error:
                # Una reparación de esquema/cita, sin sugerir el veredicto esperado.
                partial = judge.call(
                    "content_compliance_repair",
                    ASSESS + " Correct the previous output's schema or evidence error. "
                    "Quote a single contiguous span; never join separated passages with ellipses.",
                    {**payload, "previous_output": partial, "validation_error": str(error)},
                )
                validate_assessment(partial, [anchor], candidate)
            items.extend(partial["items"])
        value = {"items": items}
    return validate_assessment(value, anchors, candidate)


def verify_controls(judge):
    """Controles previos fijos; no se usan para entrenar al actor."""
    anchors = [
        {"id": "a1", "kind": "setting", "requirement": "The narrator is in a school."},
        {"id": "a2", "kind": "relationship", "requirement": "The narrator is with their brother."},
        {"id": "a3", "kind": "event", "requirement": "Doors disappear before the narrator."},
        {"id": "a4", "kind": "emotion", "requirement": "The narrator feels afraid."},
    ]
    brief = PREFIX + protected_json({"version": VERSION, "anchors": anchors})
    cases = {
        "literal": "I was in a school with my brother. Doors disappeared before me. I felt afraid.",
        "paraphrase": "My sibling, a boy, walked beside me through our school. The doorways vanished as I watched, and terror gripped me.",
        "addition": "I was in a school with my brother. Doors disappeared before me. I felt afraid. A purple bird landed on a clock.",
        "omission": "I was in a school with my brother. Doors disappeared before me.",
        "contradiction": "I was in a school with my brother. Doors disappeared before me. I felt no fear at all, only joy.",
        "switched_participant": "I was in a school with my sister, not my brother. Doors disappeared before me. I felt afraid.",
        "unrelated": "I sat alone on a beach, watching the waves with quiet joy.",
        "keywords": "school brother doors disappear fear. These are vocabulary words, not events anyone experienced.",
    }
    results = {key: assess(judge, brief, text) for key, text in cases.items()}
    expected = {
        "literal": ["preserved"] * 4,
        "paraphrase": ["preserved"] * 4,
        "addition": ["preserved"] * 4,
        "omission": ["preserved"] * 3 + ["omitted"],
        "contradiction": ["preserved"] * 3 + ["contradicted"],
        "switched_participant": ["preserved", "contradicted", "preserved", "preserved"],
    }
    checks = {
        key: {i["id"]: i["status"] for i in results[key]["items"]}
        == {a["id"]: s for a, s in zip(anchors, statuses, strict=True)}
        for key, statuses in expected.items()
    }
    checks.update({key: results[key]["score"] == 0 for key in ["unrelated", "keywords"]})
    return {
        "version": VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "results": results,
        "mean_score": statistics.mean(r["score"] for r in results.values()),
        "limitation": "Controles sintéticos; pendiente validación humana del juez y de la extracción.",
    }


def check_protocol(config, root):
    from .corpus import prepare
    from .judge import Judge
    from .storage import digest, write_json

    if config["judge"].get("rubric_version") != VERSION:
        raise ValueError("anchor-check requiere la configuración dream-v3.")
    judge = Judge(config, root)
    result = verify_controls(judge)
    result["judge"] = config["judge"]
    from .evidence import ASSESS_SELECTION

    result["prompt_hash"] = digest([ASSESS, EXTRACT, ASSESS_SELECTION, config["judge"]])
    write_json(root / "dream" / "anchor-controls-history" / f"{result['prompt_hash']}.json", result)
    write_json(root / "dream" / "anchor-calibration.json", result)
    print("Controles por elementos:", result["checks"], flush=True)
    data, _ = prepare(config, root)
    extractions = []
    for row in data["validation"][:3]:
        entry = {"problem_id": row["problem_id"]}
        try:
            contract = contract_brief(judge, row)
            entry.update(brief=contract, anchors=visible_anchors(contract))
            entry["source_compliance"] = assess(judge, contract, row["reference"])
        except (ValueError, TypeError) as error:
            # Es un diagnóstico: conservar el fallo y revisar las otras fuentes.
            # Durante entrenamiento, assess sigue interrumpiendo ante este error.
            entry["error"] = str(error)
        extractions.append(entry)
        print(
            "Contrato revisado:",
            row["problem_id"],
            entry.get("error", "sin error de esquema"),
            flush=True,
        )
    write_json(
        root / "dream" / "anchor-extraction-check.json",
        {
            "schema_and_source_quotes_passed": all("brief" in r for r in extractions),
            "source_compliance_passed": all(
                r.get("source_compliance", {}).get("score") == 1 for r in extractions
            ),
            "contracts": extractions,
            "limitation": "Las citas son verificadas; la implicación semántica y centralidad requieren revisión humana.",
        },
    )
    if not result["passed"]:
        raise RuntimeError("Controles por elementos fallidos; RL queda bloqueado.")
    if not all(r.get("source_compliance", {}).get("score") == 1 for r in extractions):
        raise RuntimeError(
            "Un requisito extraído no fue reconocido en su fuente; revisar extracción y juez."
        )
