"""Rollouts Dream: cierre textual y tokens muestreados íntegros para GRPO."""


def generate_dream(model, tokenizer, prompt, count, generation, end_token):
    if not prompt or count < 1 or generation["max_tokens"] < 1 or not end_token:
        raise ValueError("Dream requiere prompt, tamaño de grupo, límite y cierre válidos.")
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm_lora.trainer.grpo_trainer import build_shared_prompt_cache, fork_prompt_cache

    sampler = make_sampler(
        generation["temperature"],
        top_p=generation["top_p"],
        min_p=generation["min_p"],
        top_k=generation["top_k"],
    )
    was_training = model.training
    model.eval()
    generator = None
    try:
        template = build_shared_prompt_cache(model, prompt) if count > 1 else None
        prompts, histories, caches = [], [], []
        for _ in range(count):
            cache = fork_prompt_cache(model, template)
            prompts.append(prompt[-1:] if cache is not None else prompt)
            histories.append(prompt[:-1] if cache is not None else [])
            caches.append(cache)
        generator = BatchGenerator(
            model,
            stop_tokens=[[token] for token in tokenizer.eos_token_ids],
            sampler=sampler,
            completion_batch_size=1,
            prefill_batch_size=1,
        )
        uids = generator.insert(
            prompts,
            [generation["max_tokens"]] * count,
            caches=caches,
            all_tokens=histories,
        )
        tokens = {uid: [] for uid in uids}
        while responses := generator.next_generated():
            closing = []
            for response in responses:
                ids = tokens[response.uid]
                ids.append(response.token)
                # BPE puede fusionar puntuación con </; no se compara con una
                # única codificación aislada del delimitador ni se recortan IDs.
                if end_token in tokenizer.decode(ids) and response.finish_reason is None:
                    closing.append(response.uid)
            if closing:
                generator.remove(closing)
        completions, texts = [], []
        for uid in uids:
            ids = tokens[uid]
            completions.append(mx.array(ids, dtype=mx.int32))
            text_ids = ids[:-1] if ids and ids[-1] in tokenizer.eos_token_ids else ids
            texts.append(tokenizer.decode(text_ids))
        return completions, texts, [0] * count
    finally:
        if generator is not None:
            generator.close()
        model.train(was_training)
