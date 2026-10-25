"""Integración del GRPO existente; sin reimplementar su objetivo."""

import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np

from . import checkpoint
from .config import BACKEND_COMMIT
from .prompts import process_tag, render
from .rewards import score
from .storage import append_jsonl, digest, read_jsonl, write_json


def seed_all(seed):
    import mlx.core as mx

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def fingerprint(parameters):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    hasher = hashlib.sha256()
    for name, array in sorted(tree_flatten(parameters)):
        hasher.update(name.encode())
        hasher.update(str((array.shape, array.dtype)).encode())
        # float32 representa exactamente los valores bfloat16/float16.
        hasher.update(
            np.array(array.astype(mx.float32) if array.dtype == mx.bfloat16 else array).tobytes()
        )
    return hasher.hexdigest()


def frozen_fingerprint(model):
    from mlx.utils import tree_flatten

    trainable = {name for name, _ in tree_flatten(model.trainable_parameters())}
    return fingerprint(
        {name: value for name, value in tree_flatten(model.parameters()) if name not in trainable}
    )


def assert_backend():
    dist = importlib.metadata.distribution("mlx-lm-lora")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != BACKEND_COMMIT:
        raise RuntimeError("El backend instalado no coincide con el commit auditado.")


def download_model(lock, *, metadata_only=False):
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.txt", "*.model", "*.jinja"]
    if not metadata_only:
        patterns.append("*.safetensors")
    return Path(snapshot_download(lock["id"], revision=lock["revision"], allow_patterns=patterns))


def audit_tokenizer(lock, root):
    from transformers import AutoTokenizer

    path = download_model(lock, metadata_only=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    tokens = {}
    for text in ["<think>", "</think>", "<dream>", "</dream>", "<work>", "</work>"]:
        ids = tok.encode(text, add_special_tokens=False)
        tokens[text] = {
            "ids": ids,
            "pieces": tok.convert_ids_to_tokens(ids),
            "special": [i in tok.all_special_ids for i in ids],
            "added": [i in tok.added_tokens_decoder for i in ids],
        }
    asymmetry = (
        len(tokens["<think>"]["ids"]) != len(tokens["<dream>"]["ids"])
        or tokens["<think>"]["added"] != tokens["<dream>"]["added"]
    )
    result = {
        "model": lock,
        "tokens": tokens,
        "literal_asymmetry": asymmetry,
        "chat_template_present": bool(tok.chat_template),
        "chat_template_used": False,
        "embeddings_training_history": "No se infiere del tokenizer; se inspeccionan filas al cargar.",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.iterdir()
            if p.is_file() and p.suffix in {".json", ".txt", ".jinja"}
        },
    }
    write_json(root / "tokenizer-audit.json", result)
    return result


class Runner:
    def __init__(self, config, lock, root):
        import mlx.core as mx
        from mlx_lm import load

        assert_backend()
        self.config, self.lock, self.root = config, lock, root
        self.path = download_model(lock)
        self.model, self.tokenizer = load(self.path, tokenizer_config={"trust_remote_code": False})
        self.model.eval()
        mx.eval(self.model.parameters())
        self.ref = None
        self.optimizer = None
        self.base_before = None
        self.initial_adapters = None
        self.training_tokens = 0
        self.training_seconds = 0.0
        self.checkpoint_path = None
        self.audit_embeddings()

    def audit_embeddings(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        candidates = [
            (n, v)
            for n, v in tree_flatten(self.model.parameters())
            if n.endswith("embed_tokens.weight")
        ]
        if len(candidates) != 1:
            raise RuntimeError("No se pudo identificar una única matriz de embeddings.")
        name, embeddings = candidates[0]
        details = {}
        for text in ["<think>", "</think>", "<dream>", "</dream>"]:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if max(ids) >= embeddings.shape[0]:
                raise RuntimeError(f"Tokens fuera de embeddings: {text}")
            rows = embeddings[mx.array(ids)].astype(mx.float32)
            details[text] = {
                "ids": ids,
                "norms": mx.linalg.norm(rows, axis=1).tolist(),
                "finite": bool(mx.all(mx.isfinite(rows)).item()),
            }
            if not details[text]["finite"]:
                raise FloatingPointError("Embeddings no finitos.")
        write_json(
            self.root / "embedding-audit.json",
            {
                "parameter": name,
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "quantized": False,
                "delimiters": details,
                "limitation": "Filas presentes y finitas no demuestran cuánto se entrenó cada token.",
            },
        )

    def encode(self, prompt):
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > self.config["generation"]["max_prompt_tokens"]:
            raise ValueError("Prompt vacío o demasiado largo; no se recorta silenciosamente.")
        return ids

    def render_prompt(self, row, condition):
        return render(row["problem"], condition)

    def completion_reward(self, text, row, condition):
        return score(text, row["answer"], process_tag(condition)).to_dict()

    def summarize_records(self, records):
        return summarize(records)

    def stop_sequence(self):
        return ""

    def generate(self, row, condition, *, count=1):
        import mlx.core as mx
        from mlx_lm_lora.trainer.grpo_trainer import generate_grpo

        prompt = self.render_prompt(row, condition)
        ids = self.encode(prompt)
        g = self.config["generation"]
        start = time.perf_counter()
        completions, texts, indices = generate_grpo(
            self.model,
            self.tokenizer,
            [ids],
            max_tokens=g["max_tokens"],
            group_size=count,
            batch_size=1,
            end_token=self.stop_sequence(),
            temperature=g["temperature"],
            top_p=g["top_p"],
            top_k=g["top_k"],
            min_p=g["min_p"],
        )
        mx.eval(completions)
        elapsed = time.perf_counter() - start
        records = []
        for index, (tokens, text) in enumerate(zip(completions, texts)):
            ended = bool(tokens.size and tokens[-1].item() in self.tokenizer.eos_token_ids)
            records.append(
                {
                    "problem_id": row["problem_id"],
                    "family": row["family"],
                    "generalization": row.get("generalization", False),
                    "condition": condition,
                    "candidate": index,
                    "prompt": prompt,
                    "prompt_ids": ids,
                    "output_ids": tokens.tolist(),
                    "text": text,
                    "expected": row.get("answer"),
                    "reward": self.completion_reward(text, row, condition),
                    "tokens": int(tokens.size),
                    "eos": ended,
                    "truncated": not ended and tokens.size >= g["max_tokens"],
                    "group_seconds": elapsed,
                    "latency_seconds": elapsed / count,
                    "mlx_peak_bytes": mx.get_peak_memory(),
                }
            )
        return (completions, texts, indices), records

    def evaluate(self, rows, condition, folder, phase, seed, *, step=0):
        import mlx.core as mx

        # Evaluar no altera la secuencia aleatoria del entrenamiento.
        saved = [mx.array(value) for value in mx.random.state]
        previous = read_jsonl(folder / "responses.jsonl")
        wanted = {row["problem_id"] for row in rows}
        metrics = [
            r
            for r in previous
            if r.get("phase") == phase
            and r["condition"] == condition
            and r.get("step") == step
            and r["problem_id"] in wanted
        ]
        done = {r["problem_id"] for r in metrics}
        try:
            for row in rows:
                if row["problem_id"] in done:
                    continue
                mx.random.seed(int(digest([seed, row["problem_id"]])[:8], 16))
                _, records = self.generate(row, condition)
                for record in records:
                    record.update(phase=phase, step=step, evaluation_seed=seed)
                    append_jsonl(folder / "responses.jsonl", record)
                    metrics.append(record)
                print(f"  {phase} {condition}: {len(metrics)}/{len(rows)}", flush=True)
        finally:
            checkpoint.restore_mlx_random(saved[0])
        summary = self.summarize_records(metrics)
        append_jsonl(
            folder / "evaluations.jsonl",
            {"phase": phase, "condition": condition, "step": step, **summary},
        )
        return summary

    def prepare_training(self, seed):
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers

        seed_all(seed)
        lora = self.config["lora"]
        self.model.freeze()
        linear_to_lora_layers(
            self.model,
            lora["layers"],
            {
                "rank": lora["rank"],
                "scale": lora["scale"],
                "dropout": lora["dropout"],
                "keys": lora["keys"],
            },
        )
        self.ref, _ = load(self.path, tokenizer_config={"trust_remote_code": False})
        self.ref.freeze()
        self.ref.eval()
        self.model.eval()  # Dropout=0; scoring y rollout usan exactamente el mismo modo.
        mx.eval(self.model.parameters(), self.ref.parameters())
        self.optimizer = optim.AdamW(
            learning_rate=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        self.optimizer.init(self.model.trainable_parameters())
        self.base_before = frozen_fingerprint(self.model)
        self.initial_adapters = fingerprint(self.model.trainable_parameters())
        from mlx.utils import tree_flatten

        write_json(
            self.root / "training-audit.json",
            {
                "trainable_parameters": sum(
                    v.size for _, v in tree_flatten(self.model.trainable_parameters())
                ),
                "trainable_names": [n for n, _ in tree_flatten(self.model.trainable_parameters())],
                "base_before": self.base_before,
                "initial_adapters": self.initial_adapters,
                "reference": self.lock,
                "algorithm": "GRPO on-policy, una actualización por rollout",
                "normalization": "desvío poblacional + 1e-4",
                "mode": "eval, dropout cero",
                "stop": self.stop_sequence() or "EOS original; sin stop adicional",
            },
        )

    def train(self, rows, condition, seed, steps, folder, run_identity, resume=None):
        import mlx.core as mx
        from mlx import nn
        from mlx.utils import tree_flatten, tree_map
        from mlx_lm_lora.trainer.grpo_trainer import (
            _grpo_value_and_grad,
            calculate_rewards_and_advantages,
            grpo_loss,
        )

        if self.optimizer is None:
            self.prepare_training(seed)
        start_step = 0
        if resume:
            state = checkpoint.load(resume, self.model, self.optimizer, run_identity)
            start_step = state["step"]
            self.initial_adapters = state["extra"]["initial_adapters"]
            self.training_tokens = state["extra"]["training_tokens"]
            self.training_seconds = state["extra"]["training_seconds"]
            self.checkpoint_path = resume
        if start_step > steps:
            raise ValueError("El checkpoint supera los pasos solicitados.")
        order = np.random.default_rng(seed).permutation(len(rows)).tolist()
        t = self.config["training"]
        g = self.config["generation"]
        loss_grad = nn.value_and_grad(self.model, grpo_loss)
        for step in range(start_step + 1, steps + 1):
            start = time.perf_counter()
            row = rows[order[(step - 1) % len(order)]]
            (completions, texts, indices), records = self.generate(
                row, condition, count=t["group_size"]
            )
            prompt_text = self.render_prompt(row, condition)
            prompt_ids = self.encode(prompt_text)
            batch = (
                [prompt_ids],
                [[]],
                [prompt_text],
                [str(row.get("answer", ""))],
                None,
            )
            totals = [r["reward"]["total"] for r in records]

            def total_reward(prompts, completions, answer, types, totals=totals):
                return totals

            advantages, reward_metrics = calculate_rewards_and_advantages(
                [total_reward],
                [batch[2][0]] * len(texts),
                texts,
                [str(row.get("answer", ""))] * len(texts),
                [None] * len(texts),
                indices,
                [0],
                [1.0],
            )
            (loss, tokens, metrics), grads = _grpo_value_and_grad(
                loss_grad,
                self.model,
                1,
                ref_model=self.ref,
                batch=batch,
                completions=completions,
                completion_texts=texts,
                batch_indices=indices,
                advantages=advantages,
                reward_metrics=reward_metrics,
                beta=t["beta"],
                epsilon=t["epsilon"],
                epsilon_high=t["epsilon"],
                max_tokens=g["max_tokens"],
                importance_sampling_level="token",
                grpo_loss_type="grpo",
            )
            mx.eval(loss, grads, metrics, tokens)
            leaves = [v for _, v in tree_flatten(grads)]
            if not mx.isfinite(loss).item() or not all(
                mx.all(mx.isfinite(v)).item() for v in leaves
            ):
                raise FloatingPointError(
                    "Pérdida o gradientes no finitos; actualización cancelada."
                )
            norm = mx.sqrt(sum(mx.sum(v.astype(mx.float32) ** 2) for v in leaves))
            norm_value = norm.item()
            if not np.isfinite(norm_value):
                raise FloatingPointError("Norma de gradiente no finita; actualización cancelada.")
            scale = mx.minimum(1.0, t["grad_clip"] / (norm + 1e-12))
            grads = tree_map(lambda v, scale=scale: v * scale, grads)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
            if not all(
                mx.all(mx.isfinite(v)).item()
                for _, v in tree_flatten(self.model.trainable_parameters())
            ):
                raise FloatingPointError("Adaptadores no finitos después de la actualización.")
            elapsed = time.perf_counter() - start
            self.training_tokens += int(tokens.item())
            self.training_seconds += elapsed
            totals = [r["reward"]["total"] for r in records]
            entry = {
                "step": step,
                "problem_id": row["problem_id"],
                "condition": condition,
                "seed": seed,
                "loss": loss.item(),
                "gradient_norm": norm_value,
                "tokens": int(tokens.item()),
                "cumulative_tokens": self.training_tokens,
                "seconds": elapsed,
                "cumulative_seconds": self.training_seconds,
                **(
                    {"accuracy": np.mean([r["reward"]["accuracy"] for r in records]).item()}
                    if "accuracy" in records[0]["reward"]
                    else {}
                ),
                "reward_total": float(np.mean(totals)),
                **{
                    f"reward_{key}": float(np.mean([r["reward"][key] for r in records]))
                    for key in [
                        "quality",
                        "dream_likeness",
                        "reference_alignment",
                        "content_compliance",
                        "retelling_fidelity",
                    ]
                    if key in records[0]["reward"]
                },
                "format": np.mean([r["reward"]["format"] for r in records]).item(),
                "variable_reward": len(set(totals)) > 1,
                "advantages": advantages.tolist(),
                "mlx_peak_bytes": mx.get_peak_memory(),
                **{key: value.item() for key, value in metrics.items()},
            }
            self.checkpoint_path = checkpoint.save(
                folder / "checkpoints",
                self.model,
                self.optimizer,
                step=step,
                run_identity=run_identity,
                extra={
                    "initial_adapters": self.initial_adapters,
                    "base_before": self.base_before,
                    "training_tokens": self.training_tokens,
                    "training_seconds": self.training_seconds,
                    "metric": entry,
                    "responses": [{**r, "phase": "rollout", "step": step} for r in records],
                },
            )
            # El checkpoint es la fuente de verdad si se interrumpe antes de escribir estos logs.
            append_jsonl(folder / "steps.jsonl", entry)
            for record in records:
                append_jsonl(
                    folder / "responses.jsonl", {**record, "phase": "rollout", "step": step}
                )
            print(
                f"  RL {condition}/{seed} paso {step}/{steps}: loss={loss.item():.6f}, "
                f"grad={norm_value:.4g}, rewards={totals}, {elapsed:.1f}s",
                flush=True,
            )
        changed = fingerprint(self.model.trainable_parameters()) != self.initial_adapters
        base_after = frozen_fingerprint(self.model)
        validation = {
            "steps_completed": steps,
            "adapters_changed": changed,
            "base_unchanged": base_after == self.base_before,
            "base_after": base_after,
            "training_tokens": self.training_tokens,
            "training_seconds": self.training_seconds,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
        }
        write_json(folder / "training-validation.json", validation)
        if not validation["base_unchanged"]:
            raise RuntimeError("Los pesos base cambiaron.")
        return validation


def summarize(records):
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "accuracy": sum(r["reward"]["accuracy"] for r in records) / len(records),
        "format": sum(r["reward"]["format"] for r in records) / len(records),
        "tokens": sum(r["tokens"] for r in records),
        "seconds": sum(r["latency_seconds"] for r in records),
        "truncations": sum(r["truncated"] for r in records),
    }
