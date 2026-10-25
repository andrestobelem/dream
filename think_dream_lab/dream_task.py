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
EXAMPLE_SCENE = (
    "At dusk I stand on a wooden pier beside a friend. We step off the edge and the dark "
    "water holds us. I see small lights moving beneath my feet. The pier moves away while "
    "we keep walking. My friend points toward an island, but it slides sideways whenever "
    "we approach. I feel calm as the shore disappears behind us."
)
EXAMPLE_ANSWER = (
    "I remember being on a pier with a friend as daylight faded. We walked onto the water "
    "without sinking, watching lights underneath it. Behind us, the pier drifted farther "
    "away. We tried to reach an island, but it kept sliding to the side. I wasn't afraid, "
    "even when I could no longer see the shore. We continued walking together."
)
FORMAT_EXAMPLE = (
    "Complete the written task. The writer first {verb}s about the requested dream inside "
    "<{tag}>...</{tag}>, then retells that same dream in first person inside <answer>...</answer>. "
    "Write concrete scenes, not instructions. Use English and about 60-100 words per block. "
    "Include no text outside the two blocks.\n\n"
    "Example brief: A pier at dusk, a friend, walking on water, an elusive island, and calm.\n"
    "Example response:\n<{tag}>{scene}</{tag}>\n<answer>{account}</answer>\n\n"
    "Now complete this task with the same format.\nBrief: {brief}\nResponse:\n"
)
NEUTRAL_EXAMPLE = (
    "Write a first-person account of an imagined sleep experience from the brief. "
    "Use English, about 60-100 words, inside <answer>...</answer> with no text outside.\n\n"
    "Example brief: A pier at dusk, a friend, walking on water, an elusive island, and calm.\n"
    "Example response:\n<answer>{account}</answer>\n\n"
    "Now complete this task with the same format.\nBrief: {brief}\nResponse:\n"
)


def render_dream(brief, condition, *, format_example=False):
    if condition == "neutral":
        if format_example:
            return NEUTRAL_EXAMPLE.format(brief=brief, account=EXAMPLE_ANSWER)
        return NEUTRAL.format(brief=brief)
    verb, tag = CONDITIONS[condition]
    if format_example:
        return FORMAT_EXAMPLE.format(
            brief=brief, verb=verb, tag=tag, scene=EXAMPLE_SCENE, account=EXAMPLE_ANSWER
        )
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
