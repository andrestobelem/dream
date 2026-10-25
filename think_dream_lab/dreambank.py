"""Importación de DreamBank sin ejecutar HTML ni extraer rutas del archivo tar.

El registro de identidades es explícito: una serie no equivale a una persona.
Las coincidencias textuales unen grupos conservadoramente para impedir fuga.
"""

import hashlib
import re
import tarfile
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from .corpus import normalized_text
from .storage import canonical, digest, read_json, read_jsonl, write_json

SOURCE = "https://dreambank.net/"
LICENSE = "CC-BY-NC-SA-4.0"
SPLITS = ("train", "validation", "test")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"br", "p", "div", "h4"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "h4"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain(html):
    parser = PlainText()
    parser.feed(html)
    return "".join(parser.parts).strip()


def decode(raw):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("HTML con codificación no reconocida; no se reemplazan caracteres.")


def parse_series(raw, series):
    html, encoding = decode(raw)
    member_sha = hashlib.sha256(raw).hexdigest()
    title = re.search(r"<h4[^>]*>(.*?)</h4>", html, re.IGNORECASE | re.DOTALL)
    total = re.search(r"There are ([\d,]+) dreams total", html)
    if not title or not total:
        raise ValueError(f"Falta título o total publicado en {series}.")
    rows = []
    ids = set()
    for block in re.split(r"<hr\b[^>]*>", html, flags=re.IGNORECASE)[1:]:
        text = plain(block)
        if not text.startswith("#"):
            continue
        match = re.fullmatch(r"#([^\n]+)\n+(.*?)\s*\((\d+) words?\)", text, re.DOTALL)
        if not match:
            raise ValueError(f"Relato sin encabezado/cierre reconocible en {series}.")
        header, narrative, published_count = match.groups()
        native_id = header.split(" (", 1)[0].strip()
        if native_id in ids:
            raise ValueError(f"Identificador repetido en {series}: {native_id}.")
        ids.add(native_id)
        narrative = re.sub(r"\s+", " ", narrative).strip()
        if not normalized_text(narrative):
            raise ValueError(f"Relato vacío en {series}: {header}.")
        rows.append(
            {
                "reference": narrative,
                "provenance": {
                    "series": series,
                    "native_id": native_id,
                    "native_header": header,
                    "published_word_count": int(published_count),
                    "encoding": encoding,
                    "source_member_sha256": member_sha,
                },
            }
        )
    expected = int(total[1].replace(",", ""))
    if len(rows) != expected:
        raise ValueError(f"{series}: se importaron {len(rows)} de {expected} relatos publicados.")
    return rows, re.sub(r"\s+", " ", plain(title[1]))


def load_archive(path, cfg, registry):
    archive_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if archive_sha != cfg["sha256"]:
        raise ValueError("El archivo DreamBank no coincide con el SHA-256 fijado.")
    sources = registry["series"]
    seen_series = set()
    unique = {}
    counts = Counter()
    catalog = {}
    parent = {}

    def find(group):
        parent.setdefault(group, group)
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    with tarfile.open(path, "r:xz") as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            parts = PurePosixPath(member.name).parts
            if not parts or parts[-1] != "dreams.html":
                continue
            if len(parts) != 2 or ".." in parts or not member.isfile():
                raise ValueError(f"Miembro no admitido: {member.name}.")
            series = parts[0]
            if series not in sources or series in seen_series:
                raise ValueError(f"Serie desconocida o duplicada: {series}.")
            seen_series.add(series)
            spec = sources[series]
            rows, title = parse_series(archive.extractfile(member).read(), series)
            if title != re.sub(r"\s+", " ", spec["title"]):
                raise ValueError(f"Cambió el título registrado de {series}.")
            eligible = spec["eligible"] and spec["language"] == cfg["language"]
            if eligible and (not spec["person"] or not spec["group"]):
                raise ValueError(f"Identidad no resuelta para {series}.")
            catalog[series] = {**spec, "raw_reports": len(rows), "included": eligible}
            counts["raw"] += len(rows)
            if not eligible:
                counts["excluded_collection_or_language"] += len(rows)
                continue
            for row in rows:
                row["provenance"].update(person=spec["person"], group=spec["group"])
                key = digest(normalized_text(row["reference"]))
                group = spec["group"]
                find(group)
                if key in unique:
                    previous = unique[key]
                    union(previous["original_group"], group)
                    previous["provenance"].append(row["provenance"])
                    counts["duplicate_occurrences"] += 1
                    continue
                unique[key] = {
                    "problem_id": key,
                    "reference": row["reference"],
                    "word_count": len(row["reference"].split()),
                    "original_group": group,
                    "provenance": [row["provenance"]],
                    "source": "dreambank",
                    "source_url": SOURCE,
                    "license": LICENSE,
                }
    if seen_series != set(sources):
        raise ValueError(f"Faltan series registradas: {sorted(set(sources) - seen_series)}.")
    groups = defaultdict(list)
    for row in unique.values():
        group = find(row.pop("original_group"))
        if cfg["min_words"] <= row["word_count"] <= cfg["max_words"]:
            row["author_group"] = digest(["dreambank", group])[:20]
            groups[group].append(row)
        else:
            counts["length_rejected"] += 1
    ordered = sorted(groups, key=lambda g: digest([cfg["seed"], g]))
    if len(ordered) < 7:
        raise ValueError("Se necesitan al menos siete grupos independientes para las particiones.")
    cut1, cut2 = int(len(ordered) * 0.70), int(len(ordered) * 0.85)
    split_groups = dict(zip(SPLITS, (ordered[:cut1], ordered[cut1:cut2], ordered[cut2:])))
    result = {}
    pools = {}
    for split, names in split_groups.items():
        queues = [
            sorted(groups[g], key=lambda r: digest([cfg["seed"], r["problem_id"]]))[
                : cfg["max_per_group"]
            ]
            for g in names
        ]
        pools[split] = sum(map(len, queues))
        # Ronda por grupo: las personas con diarios largos no monopolizan el corpus.
        selected = []
        for index in range(cfg["max_per_group"]):
            for queue in queues:
                if index < len(queue) and len(selected) < cfg[split]:
                    selected.append({**queue[index], "split": split})
        if len(selected) != cfg[split]:
            raise ValueError(f"Relatos insuficientes en {split}: {len(selected)}/{cfg[split]}.")
        result[split] = selected
    return result, {
        "archive_sha256": archive_sha,
        "counts": dict(counts),
        "unique_before_length_filter": len(unique),
        "eligible_groups": len(groups),
        "pooled_after_group_cap": pools,
        "catalog": catalog,
        "group_components": {g: find(g) for g in sorted(parent)},
        "split_group_counts": {s: len(v) for s, v in split_groups.items()},
    }


def prepare(config, root):
    cfg = config["corpus"]
    registry = read_json(Path(cfg["groups_file"]))
    folder = Path(root) / "dream" / "data"
    manifest_path = folder / "manifest.json"
    registry_hash = digest(registry)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        rows = {s: read_jsonl(folder / f"{s}.jsonl") for s in SPLITS}
        saved_registry = folder / "groups.json"
        if (
            manifest["config"] != cfg
            or manifest["registry_hash"] != registry_hash
            or manifest["hash"] != digest(rows)
            or not saved_registry.exists()
            or digest(read_json(saved_registry)) != registry_hash
        ):
            raise ValueError("El corpus o registro congelado fue modificado.")
        return rows, manifest
    rows, stats = load_archive(Path(cfg["archive"]), cfg, registry)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Hay datos parciales: elegí otro directorio de resultados.")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "groups.json", registry)
    for split, values in rows.items():
        (folder / f"{split}.jsonl").write_text(
            "".join(canonical(row) + "\n" for row in values), encoding="utf-8"
        )
    manifest = {
        "protocol": config["protocol"],
        "config": cfg,
        "registry_hash": registry_hash,
        "hash": digest(rows),
        "source": SOURCE,
        "license": LICENSE,
        "attribution": "DreamBank (dreambank.net)",
        "code_license": "MIT",
        "counts": {s: len(v) for s, v in rows.items()},
        "stats": stats,
        "limitations": [
            "Identidades públicas seudónimas; no se descartan alias desconocidos.",
            "Las coincidencias normalizadas unen grupos para evitar fuga, sin inferir identidad.",
            "El corpus público puede haber formado parte del preentrenamiento.",
        ],
    }
    write_json(manifest_path, manifest)
    return rows, manifest
