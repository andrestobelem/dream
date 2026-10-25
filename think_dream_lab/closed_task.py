"""Cumplimiento por contrato y fidelidad a las oraciones de la escena, sin juicios libres."""

import re

from .anchors import visible_anchors
from .closed_judge import ClosedJudge


def draft_requirements(draft):
    """Cada oración o línea es una proposición completa; no se inventan resúmenes."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", draft) if s.strip()]
    return [
        {"id": f"d{i}", "kind": "scene_proposition", "requirement": sentence}
        for i, sentence in enumerate(sentences, 1)
    ]


def rate_closed(judge, draft, answer, brief, *, neutral=False):
    if not hasattr(judge, "closed_scorer"):
        judge.closed_scorer = ClosedJudge(judge)
    scorer = judge.closed_scorer
    compliance = scorer.assess(visible_anchors(brief), answer)
    requirements = draft_requirements(draft)
    fidelity = (
        scorer.assess(requirements, answer, mode="fidelity")
        if requirements and not neutral
        else None
    )
    likeness = judge.likeness(answer) if answer else {"dream_likeness": 0}
    return {
        "dream_likeness": likeness["dream_likeness"],
        "reference_alignment": None,
        "content_compliance": None if compliance["score"] is None else 4 * compliance["score"],
        "retelling_fidelity": None
        if neutral or (fidelity and fidelity["score"] is None)
        else (4 * fidelity["score"] if fidelity else 0),
        "anchor_assessment": compliance,
        "fidelity_assessment": fidelity,
        "fidelity_requirements": requirements,
        "fidelity_method": "scene_sentences_core_v2",
        "likeness_reason": likeness.get("reason"),
        "scoring_protocol": "dream-v4",
    }


def invalid_format_ratings():
    return {
        "dream_likeness": None,
        "reference_alignment": None,
        "content_compliance": None,
        "retelling_fidelity": None,
        "scoring_protocol": "dream-v4",
        "not_evaluated_reason": "invalid_format",
    }
