"""Extracción por un LLM congelado; el generador sólo recibe las semillas."""

import time
from pathlib import Path

from .dreambank_task import EXTRACTION_INSTRUCTION, select_seeds, validate_seeds
from .judge import parse_json, protected_json
from .storage import append_jsonl, digest, read_json, write_json


class SeedExtractor:
    def __init__(self, config, root, complete=None):
        self.spec = config["seed_extractor"]
        if self.spec["id"] == config["model"]["id"]:
            raise ValueError("Las semillas se extraen con otro modelo.")
        if not self.spec["revision"] or self.spec["temperature"] != 0:
            raise ValueError("El extractor debe tener revisión fija y temperatura cero.")
        self.root = Path(root) / "dream" / "seeds"
        self.selection = "category_round_robin_unique_max8_max24words_v1"
        self.identity = digest([self.spec, EXTRACTION_INSTRUCTION, self.selection])
        self.complete = complete
        self.model = self.tokenizer = None
        lock = {
            "extractor": self.spec,
            "instruction": EXTRACTION_INSTRUCTION,
            "identity": self.identity,
            "selection": self.selection,
        }
        lock_path = self.root / "extractor-lock.json"
        if lock_path.exists() and read_json(lock_path) != lock:
            raise ValueError("Las semillas existentes usan otro extractor o instrucción.")
        write_json(lock_path, lock)

    def _local_completion(self, report):
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        from .backend import download_model
        from .checkpoint import restore_mlx_random

        if self.model is None:
            self.model, self.tokenizer = load(download_model(self.spec), trust_remote_code=False)
            self.model.freeze()
            self.model.eval()
            mx.eval(self.model.parameters())
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": EXTRACTION_INSTRUCTION},
                {"role": "user", "content": protected_json({"dream_report": report})},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        saved = mx.array(mx.random.state[0])
        try:
            mx.random.seed(0)
            return generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                verbose=False,
                max_tokens=self.spec["max_tokens"],
                sampler=make_sampler(temp=0.0),
            )
        finally:
            restore_mlx_random(saved)

    def extract(self, row):
        reference_hash = digest(row["reference"])
        key = digest([self.identity, row["problem_id"], reference_hash])
        target = self.root / "cache" / f"{key}.json"
        if target.exists():
            record = read_json(target)
            if (
                record["key"] != key
                or record["extractor_identity"] != self.identity
                or record["reference_hash"] != reference_hash
                or record["seeds_hash"] != digest(record["seeds"])
            ):
                raise ValueError("La caché de semillas fue modificada.")
            return self.actor_row(row, validate_seeds(record["seeds"]))
        start = time.perf_counter()
        raw = None
        try:
            raw = (self.complete or self._local_completion)(row["reference"])
            seeds = select_seeds(parse_json(raw))
        except (ValueError, TypeError) as error:
            append_jsonl(
                self.root / "errors.jsonl",
                {
                    "key": key,
                    "problem_id": row["problem_id"],
                    "raw": raw,
                    "error": str(error),
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
            raise ValueError(
                "El extractor no produjo semillas válidas; no se inventa una trama."
            ) from error
        write_json(
            target,
            {
                "key": key,
                "extractor_identity": self.identity,
                "reference_hash": reference_hash,
                "seeds": seeds,
                "seeds_hash": digest(seeds),
                "raw": raw,
                "selection": self.selection,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        return self.actor_row(row, seeds)

    @staticmethod
    def actor_row(row, seeds):
        # Lista permitida: no propagamos el relato, sus metadatos ni una trama al actor.
        return {
            "problem_id": row["problem_id"],
            "split": row["split"],
            "author_group": row["author_group"],
            "seeds": seeds,
        }
