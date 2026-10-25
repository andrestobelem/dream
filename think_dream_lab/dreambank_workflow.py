"""Etapas del protocolo DreamBank, separadas del juez histórico."""

from .dreambank import SPLITS, prepare
from .seed_extractor import SeedExtractor
from .storage import digest, write_json


def extract_seeds(config, root, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError("--limit debe ser un entero positivo por partición.")
    rows, corpus_manifest = prepare(config, root)
    if any(limit > len(values) for values in rows.values()):
        raise ValueError("La muestra supera el tamaño de alguna partición.")
    extractor = SeedExtractor(config, root)
    selected = {}
    for split in SPLITS:
        selected[split] = []
        for row in rows[split][:limit]:
            selected[split].append(extractor.extract(row))
            print(f"Semillas {split}: {len(selected[split])}/{limit}", flush=True)
    manifest = {
        "corpus_hash": corpus_manifest["hash"],
        "extractor_identity": extractor.identity,
        "actor_rows": selected,
        "hash": digest(selected),
        "counts": {s: len(v) for s, v in selected.items()},
        "generator_sees_reference": False,
    }
    write_json(root / "dream/seeds/manifest.json", manifest)
    return manifest
