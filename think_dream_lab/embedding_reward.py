"""Cercanía semántica del relato a DreamBank; no es un detector de sueños."""

from pathlib import Path

import numpy as np

from .dream_task import extract_blocks
from .storage import digest, read_json, write_json


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Se requieren vectores finitos en una matriz no vacía.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Embedding nulo: no se asigna una recompensa ficticia.")
    return values / norms


class Encoder:
    """MiniLM congelado, CPU y pooling explícito de todos los fragmentos."""

    def __init__(self, spec, root):
        self.spec = spec
        self.identity = digest(["embedding-encoder-v1", spec])
        self.path = Path(root) / "dream/embeddings" / f"cache-{self.identity}.json"
        self.cache = read_json(self.path) if self.path.exists() else {}
        self.model = self.tokenizer = None

    def encode(self, texts):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("No se codifican relatos vacíos.")
        keys = [digest(text) for text in texts]
        pending = {k: text for k, text in zip(keys, texts) if k not in self.cache}
        if pending:
            self._encode_pending(pending)
            write_json(self.path, self.cache)
        vectors = []
        for key in keys:
            record = self.cache[key]
            if record["hash"] != digest(record["vector"]):
                raise ValueError("La caché de embeddings fue modificada.")
            vectors.append(record["vector"])
        return unit_vectors(vectors)

    def _encode_pending(self, pending):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(4)
        if self.model is None:
            args = {"revision": self.spec["revision"], "trust_remote_code": False}
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec["id"], **args)
            self.model = AutoModel.from_pretrained(self.spec["id"], **args).to("cpu")
            self.model.eval()
            self.model.requires_grad_(False)
        width = self.spec["max_tokens"] - self.tokenizer.num_special_tokens_to_add(pair=False)
        if width < 1:
            raise ValueError("Límite del encoder insuficiente para los tokens de control.")
        encoded = self.tokenizer(
            list(pending.values()),
            max_length=self.spec["max_tokens"],
            truncation=True,
            stride=0,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )
        mapping = encoded.pop("overflow_to_sample_mapping")
        keys = list(pending)
        chunks = [{k: values[i] for k, values in encoded.items()} for i in range(len(mapping))]
        owners = [keys[i] for i in mapping]
        specials = self.tokenizer.num_special_tokens_to_add(pair=False)
        lengths = [sum(chunk["attention_mask"]) - specials for chunk in chunks]
        if any(length < 1 for length in lengths):
            raise ValueError("El tokenizer no encontró contenido en el relato.")
        sums, weights = {}, {}
        for start in range(0, len(chunks), self.spec["batch_size"]):
            end = start + self.spec["batch_size"]
            batch = self.tokenizer.pad(chunks[start:end], padding=True, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            for key, length, vector in zip(owners[start:end], lengths[start:end], vectors):
                sums[key] = sums.get(key, 0) + vector * length
                weights[key] = weights.get(key, 0) + length
        for key in pending:
            vector = unit_vectors([sums[key] / weights[key]])[0].tolist()
            self.cache[key] = {"vector": vector, "hash": digest(vector)}


class DreamBankReward:
    """r=(1+max cos(answer, dream_i))/2; sólo referencias de entrenamiento."""

    def __init__(self, rows, encoder):
        if not rows or any(r["split"] != "train" for r in rows):
            raise ValueError("El banco de recompensa sólo admite relatos de entrenamiento.")
        if len({r["problem_id"] for r in rows}) != len(rows):
            raise ValueError("Referencias duplicadas en el banco de recompensa.")
        self.encoder = encoder
        self.ids = [r["problem_id"] for r in rows]
        self.vectors = unit_vectors(encoder.encode([r["reference"] for r in rows]))
        self.identity = digest(
            ["nearest-dream-cosine-v1", encoder.identity, self.ids, self.vectors.tolist()]
        )

    def answer_score(self, answer):
        vector = self.encoder.encode([answer])[0]
        cosine = np.clip(self.vectors @ vector, -1, 1)
        index = int(np.argmax(cosine))
        value = float(cosine[index])
        return {
            "cosine": value,
            "total": (1 + value) / 2,
            "top5_mean_cosine_diagnostic": float(np.sort(cosine)[-5:].mean()),
            "nearest_reference_id": self.ids[index],
            "bank_identity": self.identity,
        }

    def score(self, text):
        blocks = extract_blocks(text, "D")
        if not blocks["format"]:
            return {
                "total": 0.0,
                "cosine": None,
                "format": 0,
                "score_status": "invalid_format",
                "bank_identity": self.identity,
            }
        return {**self.answer_score(blocks["answer"]), "format": 1, "score_status": "scored"}
