"""Aritmética sintética: generación y verificación independientes del modelo."""

import random
from pathlib import Path

from .storage import digest, read_json, read_jsonl, write_json


def evaluate(node):
    if isinstance(node, int):
        return node
    op, left, right = node
    a, b = evaluate(left), evaluate(right)
    return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[op]()


def normalize(node):
    if isinstance(node, int):
        return node
    op, left, right = node
    a, b = normalize(left), normalize(right)
    if op in ("+", "*") and str(a) > str(b):
        a, b = b, a
    return [op, a, b]


def expression(node):
    if isinstance(node, int):
        return str(node)
    return f"({expression(node[1])} {node[0]} {expression(node[2])})"


def verify_answer(tree):
    stack = []

    def visit(node):
        if isinstance(node, int):
            stack.append(node)
            return
        visit(node[1])
        visit(node[2])
        right, left = stack.pop(), stack.pop()
        if node[0] == "+":
            stack.append(left + right)
        elif node[0] == "-":
            stack.append(left - right)
        else:
            stack.append(left * right)

    visit(tree)
    return stack.pop()


def make_problem(rng, difficulty, generalization=False):
    bound = {"easy": 12, "medium": 30, "hard": 99}[difficulty]
    a, b, c, d = [rng.randint(1, bound) for _ in range(4)]
    if difficulty == "easy":
        ops = ["+", "-"]
    else:
        ops = ["+", "-", "*"]
    x, y, z = [rng.choice(ops) for _ in range(3)]
    if generalization:
        tree = [x, [y, a, b], [z, c, d]]
        family = "balanced_three_operations"
    else:
        tree = [x, [y, a, b], c]
        family = "left_two_operations"
    return normalize(tree), family


def generate(config: dict) -> dict[str, list]:
    cfg = config["data"]
    rng = random.Random(cfg["seed"])
    seen = set()
    result = {}
    for split, count in [
        ("train", cfg["train"]),
        ("validation", cfg["validation"]),
        ("test", cfg["test"]),
    ]:
        rows = []
        attempts = 0
        while len(rows) < count:
            attempts += 1
            if attempts > count * 1000:
                raise ValueError("No alcanza la cantidad de problemas únicos para esta dificultad.")
            generalization = split == "test" and len(rows) >= count // 2
            tree, family = make_problem(rng, cfg["difficulty"], generalization)
            key = digest(tree)
            if key in seen:
                continue
            seen.add(key)
            answer = evaluate(tree)
            # Segunda evaluación aritmética, sin eval ni ejecución de texto.
            assert verify_answer(tree) == answer
            rows.append(
                {
                    "problem_id": key,
                    "problem": f"Calculate {expression(tree)}. Give the integer result.",
                    "answer": answer,
                    "tree": tree,
                    "family": family,
                    "difficulty": cfg["difficulty"],
                    "split": split,
                    "generalization": generalization,
                }
            )
        result[split] = rows
    return result


def freeze(config: dict, root: Path) -> tuple[dict, str]:
    folder = root / "data"
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["config"] != config["data"]:
            raise ValueError("Los datos ya están congelados con otra configuración.")
        rows = {s: read_jsonl(folder / f"{s}.jsonl") for s in manifest["splits"]}
        if digest(rows) != manifest["hash"]:
            raise ValueError("Los datos congelados fueron modificados.")
        return rows, manifest["hash"]
    rows = generate(config)
    folder.mkdir(parents=True, exist_ok=True)
    from .storage import canonical

    for split, values in rows.items():
        path = folder / f"{split}.jsonl"
        if path.exists():
            raise ValueError(f"Datos parciales existentes: {path}; no se sobrescriben.")
        path.write_text("".join(canonical(row) + "\n" for row in values))
    data_hash = digest(rows)
    write_json(
        manifest_path,
        {
            "config": config["data"],
            "hash": data_hash,
            "splits": list(rows),
            "counts": {s: len(r) for s, r in rows.items()},
        },
    )
    return rows, data_hash
