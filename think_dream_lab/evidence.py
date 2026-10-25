"""Fragmentos identificados del texto original; el juez selecciona y no inventa citas."""

import re

from .judge import GUARD

ASSESS_SELECTION = GUARD + (
    "Check one requirement against a candidate account supplied as overlapping passages. "
    "Classify preserved ONLY if the account asserts the requirement or a semantic equivalent "
    "as something experienced by the narrator. Vocabulary lists, instructions, and mentions of "
    "words do not assert experiences. Classify contradicted ONLY if an explicit incompatible "
    "fact is asserted. Otherwise classify omitted. Unstated feelings are omitted, not inferred. "
    "Paraphrases and compatible additions are allowed. Track participants, relations and negation. "
    "In the reason field, briefly explain what the account actually asserts about the requirement. "
    "Choose the status and the IDs of passages supporting it. Use [] for omitted. "
    'The status must be exactly one of "preserved", "omitted", "contradicted". '
    'Return {"reason":"short explanation","status":"omitted",'
    '"evidence_ids":[]}. Select only provided passage IDs; never write a quote. '
    "Output ONLY that JSON object, with no prose before it or code fences."
)


def passages(text, size=40, overlap=10):
    """Ventanas contiguas con offsets exactos, sin normalizar el texto fuente."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Tamaño o solapamiento de fragmentos inválido.")
    words = list(re.finditer(r"\S+", text))
    result = []
    for i in range(0, len(words), size - overlap):
        start, end = words[i].start(), words[min(i + size, len(words)) - 1].end()
        result.append(
            {"id": f"s{len(result) + 1}", "start": start, "end": end, "text": text[start:end]}
        )
        if i + size >= len(words):
            break
    return result


def resolve_selection(value, anchor, chunks, candidate):
    required = {"reason", "status", "evidence_ids"}
    if not isinstance(value, dict) or not required <= set(value) <= required | {"id", "passage_id"}:
        raise ValueError("Esquema de selección de evidencia inválido.")
    if "id" in value and value["id"] != anchor["id"]:
        raise ValueError("El juez seleccionó otro requisito.")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Falta la explicación del veredicto.")
    status, selected = value["status"], value["evidence_ids"]
    if not isinstance(status, str) or status not in {"preserved", "omitted", "contradicted"}:
        raise ValueError("Estado de selección inválido.")
    if not isinstance(selected, list) or any(not isinstance(k, str) for k in selected):
        raise ValueError("IDs de evidencia inválidos.")
    by_id = {p["id"]: p for p in chunks}
    if len(set(selected)) != len(selected) or any(k not in by_id for k in selected):
        raise ValueError("Fragmento inexistente o duplicado.")
    if "passage_id" in value and value["passage_id"] not in selected:
        raise ValueError("El identificador redundante no coincide con la evidencia seleccionada.")
    if (status == "omitted") != (not selected):
        raise ValueError("Una omisión no selecciona evidencia; los otros estados sí.")
    evidence = ""
    if selected:
        start = min(by_id[k]["start"] for k in selected)
        end = max(by_id[k]["end"] for k in selected)
        evidence = candidate[start:end]
        if any(candidate[p["start"] : p["end"]] != p["text"] for p in chunks):
            raise ValueError("Los fragmentos no corresponden al texto fuente.")
    return {"id": anchor["id"], "reason": value["reason"], "status": status, "evidence": evidence}


def assess_selection(judge, anchors, candidate):
    chunks = passages(candidate)
    if not chunks:
        return {
            "items": [{"id": a["id"], "status": "omitted", "evidence": ""} for a in anchors]
        }, {}
    items, selections = [], {}
    visible = [{"id": p["id"], "text": p["text"]} for p in chunks]
    for anchor in anchors:
        payload = {"requirement": anchor, "passages": visible}
        value = judge.call("content_compliance_selection", ASSESS_SELECTION, payload)
        try:
            item = resolve_selection(value, anchor, chunks, candidate)
        except (ValueError, TypeError) as error:
            value = judge.call(
                "content_compliance_selection_repair",
                ASSESS_SELECTION + " Correct the previous output's validation error. "
                "Do not change your judgment solely to satisfy a desired score.",
                {**payload, "previous_output": value, "validation_error": str(error)},
            )
            item = resolve_selection(value, anchor, chunks, candidate)
        items.append(item)
        selections[anchor["id"]] = value["evidence_ids"]
    return {"items": items}, {"passages": chunks, "selections": selections}
