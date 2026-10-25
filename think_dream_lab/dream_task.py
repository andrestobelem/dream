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
            and re.search(r"</?(?:think|dream|answer)>", answer) is None
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
    protocol = config["judge"].get("rubric_version")
    v4 = protocol == "dream-v4"
    dimensions = {
        k: None if ratings[k] is None else ratings[k] / 4
        for k in ["dream_likeness", "reference_alignment", "retelling_fidelity"]
    }
    alignment_key = (
        "content_compliance" if protocol in {"dream-v3", "dream-v4"} else "reference_alignment"
    )
    if alignment_key == "content_compliance":
        value = ratings["content_compliance"]
        dimensions[alignment_key] = None if value is None else value / 4
    required = ["dream_likeness", alignment_key] + ([] if neutral else ["retelling_fidelity"])
    quality = (
        None
        if any(dimensions[k] is None for k in required)
        else sum(r[k] * dimensions[k] for k in required)
    )
    if neutral and quality is not None:
        quality /= sum(r[k] for k in required)
    total = (
        None
        if quality is None
        else r["quality_weight"] * quality + r["format_weight"] * int(format_score)
    )
    if v4 and not format_score:
        # Falta de estructura comprobada, independiente de cualquier incertidumbre semántica.
        total = 0.0
    result = {
        **dimensions,
        "quality": quality,
        "format": int(format_score),
        "total": total,
        "neutral_formula": neutral,
    }
    if v4:
        result.update(
            scoring_protocol=protocol,
            score_status="invalid_format"
            if not format_score
            else "abstained"
            if total is None
            else "scored",
            unscored_dimensions=[k for k, v in result.items() if v is None],
        )
    return result
