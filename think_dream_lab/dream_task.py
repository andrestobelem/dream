"""Tarea de sueños: escena generada y relato posterior, con referencia reservada."""

import re

from .prompts import CONDITIONS

VERSION = "dream-v1"
TEMPLATE = (
    "A conversation between User and Assistant. The User requests a dream from a brief. "
    "The Assistant first {verb}s about the requested dream inside <{tag}>...</{tag}>, "
    "then retells that same dream in first person inside <answer>...</answer>. "
    "Write a concrete imagined dream scene in the first block, then narrate it in the second. "
    "Do not explain how to write a dream. Use English and about 60-100 words per block. "
    "Include no text outside the two blocks.\nUser: {brief}\nAssistant:"
)
NEUTRAL = (
    "Write a first-person account of an imagined sleep experience from this brief. "
    "Use English, about 60-100 words, and put the account inside <answer>...</answer>. "
    "Include no text outside that block.\nBrief: {brief}\nAccount:"
)


def render_dream(brief, condition):
    if condition == "neutral":
        return NEUTRAL.format(brief=brief)
    verb, tag = CONDITIONS[condition]
    return TEMPLATE.format(brief=brief, verb=verb, tag=tag)


def extract_blocks(text, condition):
    process = ""
    answer = ""
    if text.count("<answer>") == text.count("</answer>") == 1:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            answer = match[1].strip()
    if condition == "neutral":
        valid = (
            bool(answer)
            and re.fullmatch(r"\s*<answer>.*?</answer>\s*", text, re.DOTALL) is not None
        )
        return {"draft": "", "answer": answer, "format": int(valid)}
    _, tag = CONDITIONS[condition]
    if text.count(f"<{tag}>") == text.count(f"</{tag}>") == 1:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            process = match[1].strip()
    match = re.fullmatch(rf"\s*<{tag}>.*?</{tag}>\s*<answer>.*?</answer>\s*", text, re.DOTALL)
    foreign = "think" if tag == "dream" else "dream"
    valid = bool(
        process and answer and match and f"<{foreign}>" not in text and f"</{foreign}>" not in text
    )
    # Ningún bloque de control puede estar anidado dentro de otro.
    if re.search(r"</?(?:think|dream|answer)>", process) or re.search(
        r"</?(?:think|dream|answer)>", answer
    ):
        valid = False
    return {"draft": process, "answer": answer, "format": int(valid)}


def combine_scores(ratings, format_score, config, *, neutral=False):
    r = config["reward"]
    dimensions = {
        k: ratings[k] / 4 for k in ["dream_likeness", "reference_alignment", "retelling_fidelity"]
    }
    v3 = config["judge"].get("rubric_version") == "dream-v3"
    alignment_key = "content_compliance" if v3 else "reference_alignment"
    if v3:
        dimensions["content_compliance"] = ratings["content_compliance"] / 4
    quality = sum(
        r[k] * dimensions[k] for k in ["dream_likeness", alignment_key, "retelling_fidelity"]
    )
    if neutral:
        # Sin escena previa, se reporta calidad en dos dimensiones; no se finge fidelidad.
        quality = (
            r["dream_likeness"] * dimensions["dream_likeness"]
            + r[alignment_key] * dimensions[alignment_key]
        ) / (r["dream_likeness"] + r[alignment_key])
    return {
        **dimensions,
        "quality": quality,
        "format": int(format_score),
        "total": r["quality_weight"] * quality + r["format_weight"] * int(format_score),
        "neutral_formula": neutral,
    }
