"""Envoltura de inferencia por etapas; nunca se usa para actualizar pesos históricos."""

import re

TAG_BOUNDARIES = ("<dream>", "</dream>", "<answer>", "</answer>")
TURN = re.compile(r"(?m)^[ \t]*(?:User|Assistant|Human):")


def boundary(text):
    """Primera frontera estructural completa, incluso dentro de un token BPE fusionado."""
    matches = [(text.index(tag), tag) for tag in TAG_BOUNDARIES if tag in text]
    turn = TURN.search(text)
    if turn:
        matches.append((turn.start(), "new_turn"))
    return min(matches, key=lambda x: x[0]) if matches else None


def finish_stage(raw, ids, eos_ids, limit):
    found = boundary(raw)
    if found:
        cut, reason = found
        content, discarded = raw[:cut], raw[cut:]
    else:
        content, discarded = raw, ""
        reason = (
            "eos"
            if ids and ids[-1] in eos_ids
            else "budget"
            if len(ids) >= limit
            else "generator_end"
        )
    return {
        "content": content.strip(),
        "raw": raw,
        "output_ids": ids,
        "stop_reason": reason,
        "discarded_suffix": discarded,
        "complete": bool(content.strip()) and reason in {*TAG_BOUNDARIES, "new_turn", "eos"},
    }


def generate_stage(model, tokenizer, prompt_ids, generation):
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm_lora.trainer.grpo_trainer import fork_prompt_cache

    if not prompt_ids or generation["max_tokens"] < 1:
        raise ValueError("Prefijo y límite de etapa deben ser positivos.")
    sampler = make_sampler(
        generation["temperature"],
        top_p=generation["top_p"],
        min_p=generation["min_p"],
        top_k=generation["top_k"],
    )
    was_training = model.training
    model.eval()
    generator = None
    ids = []
    try:
        generator = BatchGenerator(
            model,
            stop_tokens=[[t] for t in tokenizer.eos_token_ids],
            sampler=sampler,
            completion_batch_size=1,
            prefill_batch_size=1,
        )
        generator.insert(
            [prompt_ids],
            [generation["max_tokens"]],
            caches=[fork_prompt_cache(model, None)],
            all_tokens=[[]],
        )
        while responses := generator.next_generated():
            for r in responses:
                ids.append(r.token)
                text_ids = ids[:-1] if ids[-1] in tokenizer.eos_token_ids else ids
                if boundary(tokenizer.decode(text_ids)) is not None and r.finish_reason is None:
                    generator.remove([r.uid])
        text_ids = ids[:-1] if ids and ids[-1] in tokenizer.eos_token_ids else ids
        result = finish_stage(
            tokenizer.decode(text_ids), ids, tokenizer.eos_token_ids, generation["max_tokens"]
        )
        result["prompt_ids"] = prompt_ids
        return result
    finally:
        if generator is not None:
            generator.close()
        model.train(was_training)


def compose(dream, answer):
    """Etiquetas añadidas por el programa, no evidencia de cierre aprendido por el modelo."""
    return f"<dream>\n{dream}\n</dream>\n<answer>\n{answer}\n</answer>"
