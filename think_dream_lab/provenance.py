"""Copias ejecutables del código y configuración usados por cada worker."""

import hashlib
from pathlib import Path

from .config import implementation_manifest
from .storage import canonical, digest, read_json, write_json


def freeze_source(root, config):
    project = Path(__file__).resolve().parent.parent
    sources = {
        f"think_dream_lab/{p.name}": p.read_bytes()
        for p in (project / "think_dream_lab").glob("*.py")
    }
    for name in ["uv.lock", "pyproject.toml"]:
        sources[name] = (project / name).read_bytes()
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    bundle_hash = digest(hashes)
    folder = root.resolve() / "source-snapshots" / bundle_hash
    for name, content in sources.items():
        target = folder / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != hashes[name]:
                raise ValueError("Una copia congelada de código fue modificada; no se sobrescribe.")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o444)
    manifest = {
        "bundle_hash": bundle_hash,
        "files": hashes,
        "implementation_hash": implementation_manifest()["hash"],
    }
    write_json(folder / "source-manifest.json", manifest)
    config_path = root.resolve() / "config-snapshots" / f"{digest(config)}.json"
    if config_path.exists():
        if canonical(read_json(config_path)) != canonical(config):
            raise ValueError("La configuración congelada fue modificada.")
    else:
        write_json(config_path, config)
        config_path.chmod(0o444)
    return {
        "directory": str(folder),
        "bundle_hash": bundle_hash,
        "implementation_hash": manifest["implementation_hash"],
        "config_path": str(config_path),
    }
