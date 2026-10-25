"""Actor DreamBank: recompensa del relato y GRPO sobre todos los tokens."""

import statistics

from .backend import Runner
from .dreambank_task import render_prompt


class EmbeddingRunner(Runner):
    def __init__(self, config, lock, root, reward):
        self.embedding_reward = reward
        super().__init__(config, lock, root)

    def render_prompt(self, row, condition):
        if condition != "D":
            raise ValueError("El protocolo DreamBank sólo usa Dream (D).")
        return render_prompt(row["seeds"])

    def stop_sequence(self):
        return "</answer>"

    def completion_reward(self, text, row, condition):
        if condition != "D":
            raise ValueError("El protocolo DreamBank sólo usa D.")
        return self.embedding_reward.score(text)

    def generate(self, row, condition, *, count=1):
        # Familia sólo para el registro común; no forma parte de la consigna.
        return super().generate({**row, "family": "dreambank"}, condition, count=count)

    def summarize_records(self, records):
        if not records:
            return {"n": 0}
        cosines = [r["reward"]["cosine"] for r in records if r["reward"]["cosine"] is not None]
        return {
            "n": len(records),
            "total": statistics.mean(r["reward"]["total"] for r in records),
            "format": statistics.mean(r["reward"]["format"] for r in records),
            "cosine_valid_answers": statistics.mean(cosines) if cosines else None,
            "tokens": sum(r["tokens"] for r in records),
            "truncations": sum(r["truncated"] for r in records),
        }
