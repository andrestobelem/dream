"""Corpus de sueños: fuente versionada, etiquetas originales y splits por autor."""

import csv
import hashlib
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .storage import canonical, digest, read_json, read_jsonl, write_json

SOURCE = "https://zenodo.org/records/19161757"
PAPER = "https://arxiv.org/abs/2603.26992"
TAXONOMY_SOURCE = "https://dreams.ucsc.edu/Coding/"
# Sólo indicadores débiles inspirados en dominios H/VdC; NO son codificaciones humanas H/VdC.
INDICATORS = {
    "characters_family": r"\b(mother|father|sister|brother|family|mom|dad|wife|husband)\b",
    "characters_animals": r"\b(dog|cat|horse|bird|animal|snake)\b",
    "settings_school": r"\b(school|classroom|teacher|university)\b",
    "settings_home": r"\b(home|house|bedroom|kitchen|apartment)\b",
    "settings_outdoors": r"\b(forest|river|sea|ocean|beach|mountain|street)\b",
    "activities_flight": r"\b(fly|flying|flew|float|floating)\b",
    "interactions_threat": r"\b(chased|chasing|attack|attacked|fight|fighting|escape)\b",
    "emotions_fear": r"\b(afraid|fear|scared|terrified|anxious)\b",
    "emotions_joy": r"\b(happy|joy|excited|delighted)\b",
    "objects_transport": r"\b(car|train|bus|plane|bicycle|boat)\b",
}


def normalized_text(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def indicators(text):
    return [name for name, pattern in INDICATORS.items() if re.search(pattern, text, re.IGNORECASE)]


def file_checksum(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def fetch(config, cache=Path(".cache/corpora")):
    cfg = config["corpus"]
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / cfg["file"]
    if target.exists() and file_checksum(target) == cfg["md5"]:
        return target
    if target.exists():
        raise ValueError(
            f"Corpus incompleto o checksum incorrecto: {target}. No se usa ni sobrescribe."
        )
    url = f"https://zenodo.org/api/records/{cfg['record']}/files/{cfg['file']}/content"
    partial = target.with_suffix(target.suffix + ".part")
    start = time.monotonic()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "dream-research/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with partial.open(mode) as stream:
            while block := response.read(1024 * 1024):
                if time.monotonic() - start > 600:
                    raise TimeoutError(
                        "Descarga interrumpida a los 10 minutos; .part permite continuar."
                    )
                stream.write(block)
    if file_checksum(partial) != cfg["md5"]:
        raise ValueError("La descarga no coincide con el checksum publicado por Zenodo.")
    partial.replace(target)
    return target


def load_reports(path, config):
    cfg = config["corpus"]
    seen = set()
    by_author = defaultdict(list)
    counts = Counter()
    csv.field_size_limit(2 * 1024 * 1024)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream, delimiter="\t"):
            counts["raw"] += 1
            categories = [s for s in record.get("categories", "").split("::") if s]
            if not set(categories) & set(cfg["include_categories"]):
                counts["category_rejected"] += 1
                continue
            text = re.sub(r"\s+", " ", record["post_text"]).strip()
            length = len(text.split())
            if not cfg["min_words"] <= length <= cfg["max_words"]:
                counts["length_rejected"] += 1
                continue
            key = digest(normalized_text(text))
            if key in seen:
                counts["duplicate_rejected"] += 1
                continue
            seen.add(key)
            author = digest(["dreamviews", record["user_id"]])[:20]
            row = {
                "problem_id": key,
                "source_id": record["post_id"],
                "author_group": author,
                "reference": text,
                "word_count": length,
                "source": "dreamviews",
                "source_url": SOURCE,
                "license": cfg["license"],
                "native_categories": categories,
                "native_tags": record.get("tags", "").split("::"),
                "lucidity": record.get("lucidity", "unspecified"),
                "nightmare_label": record.get("nightmare", "").lower() == "true",
                "label_origin": "author_self_report; derived lucidity/nightmare by corpus curator",
                "content_indicators": indicators(text),
                "indicator_origin": "weak_regex_not_Hall_Van_de_Castle_gold",
                "family": "nightmare"
                if "nightmare" in categories
                else record.get("lucidity", "unspecified"),
            }
            by_author[author].append(row)
    # Varias redacciones casi idénticas del mismo autor no cruzan splits: agrupamos por autor.
    pools = {"train": [], "validation": [], "test": []}
    for author, reports in sorted(by_author.items()):
        reports.sort(key=lambda row: digest([cfg["seed"], row["problem_id"]]))
        bucket = int(digest([cfg["seed"], author])[:8], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        pools[split].extend({**row, "split": split} for row in reports[: cfg["max_per_author"]])
    result = {}
    for split, rows in pools.items():
        # Selección balanceada entre categorías, sin reemplazo ni inventar etiquetas ausentes.
        strata = defaultdict(list)
        for row in rows:
            strata[row["family"]].append(row)
        for values in strata.values():
            values.sort(key=lambda row: digest([cfg["seed"], row["problem_id"]]))
        selected = []
        while len(selected) < cfg[split]:
            progressed = False
            for family in sorted(strata):
                if strata[family] and len(selected) < cfg[split]:
                    selected.append(strata[family].pop())
                    progressed = True
            if not progressed:
                raise ValueError(
                    f"Relatos insuficientes para {split}: {len(selected)}/{cfg[split]}."
                )
        random.Random(cfg["seed"]).shuffle(selected)
        result[split] = selected
    return result, {
        "filter_counts": dict(counts),
        "eligible_authors": len(by_author),
        "eligible_reports": sum(len(v) for v in by_author.values()),
        "pooled_after_author_cap": {s: len(rows) for s, rows in pools.items()},
        "native_categories": dict(
            Counter(c for rows in by_author.values() for r in rows for c in r["native_categories"])
        ),
    }


def prepare(config, root):
    folder = root / "dream" / "data"
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["config"] != config["corpus"]:
            raise ValueError("El corpus ya está congelado con otra configuración.")
        rows = {s: read_jsonl(folder / f"{s}.jsonl") for s in ["train", "validation", "test"]}
        if digest(rows) != manifest["hash"]:
            raise ValueError("Los datos del corpus fueron modificados.")
        return rows, manifest
    path = fetch(config)
    rows, stats = load_reports(path, config)
    folder.mkdir(parents=True, exist_ok=True)
    for split, values in rows.items():
        target = folder / f"{split}.jsonl"
        if target.exists():
            raise ValueError(f"Datos parciales existentes: {target}.")
        target.write_text("".join(canonical(row) + "\n" for row in values))
    manifest = {
        "config": config["corpus"],
        "hash": digest(rows),
        "source": SOURCE,
        "paper": PAPER,
        "taxonomy_source": TAXONOMY_SOURCE,
        "archive_md5": file_checksum(path),
        "archive_sha256": file_checksum(path, "sha256"),
        "stats": stats,
        "counts": {s: len(v) for s, v in rows.items()},
        "authors": {s: len({r["author_group"] for r in v}) for s, v in rows.items()},
        "license": config["corpus"]["license"],
        "code_license": "MIT",
        "literary_fallback_used": False,
        "limitation": "Corpus público: no se puede excluir memorización durante preentrenamiento.",
    }
    write_json(manifest_path, manifest)
    print(
        "Corpus preparado:",
        manifest["counts"],
        stats["eligible_reports"],
        "relatos elegibles.",
        flush=True,
    )
    return rows, manifest
