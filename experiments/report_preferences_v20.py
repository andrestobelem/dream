"""Resultados del piloto de preferencias: revisión congelada, generalización y auditoría."""

import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.preferences_v20 import ROOT
from experiments.report_dream_v19 import expected_state
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import digest, read_json, read_jsonl, write_json


def frozen_review(folder, kind):
    packet = read_json(folder / f"{kind}-packet.json")
    notes = read_json(folder / f"{kind}-annotations.json")
    frozen = read_json(folder / f"{kind}-frozen.json")
    if digest(packet) != frozen["packet_hash"] or digest(notes) != frozen["annotations_hash"]:
        raise ValueError("Cambió una revisión congelada.")
    if len(notes["ratings"]) != len(packet):
        raise ValueError("Revisión incompleta.")
    for row in notes["ratings"]:
        if row["case_hash"] != digest(
            next(p for p in packet if p["review_id"] == row["review_id"])
        ):
            raise ValueError("Texto distinto en revisión.")
    return packet, notes, frozen


def paired_summary(folder, kind):
    _packet, notes, frozen = frozen_review(folder, kind)
    key = {r["review_id"]: r for r in read_json(folder / f"{kind}-key.json")}
    counts = Counter()
    grouped = defaultdict(list)
    rows = []
    for row in notes["ratings"]:
        k = key[row["review_id"]]
        pref = row["preference"]
        phase = k[pref + "_phase"] if pref in {"A", "B"} else pref
        counts[phase] += 1
        if phase != "no_se":
            grouped[k["problem_id"]].append({"after": 1, "before": -1, "tie": 0}[phase])
        rows.append(
            {
                "review_id": row["review_id"],
                "problem_id": k["problem_id"],
                "preference": phase,
                "confidence": row["confidence"],
            }
        )
    result = {
        "counts": dict(counts),
        "comparisons": len(rows),
        "rows": rows,
        "annotations_hash": frozen["annotations_hash"],
        "independent_human_review": False,
    }
    if kind == "quality":
        means = np.array([statistics.mean(v) for _, v in sorted(grouped.items())])
        indices = np.random.default_rng(20261029).integers(0, len(means), size=(20000, len(means)))
        estimates = means[indices].mean(axis=1)
        low, high = np.quantile(estimates, [0.025, 0.975])
        result.update(
            net_preference=float(means.mean()),
            conditional_win_rate=counts["after"] / (counts["after"] + counts["before"])
            if counts["after"] + counts["before"]
            else None,
            exploratory_interval={
                "low": float(low),
                "high": float(high),
                "unit": "development_prompt",
                "groups": len(means),
                "samples": 20000,
                "seed": 20261029,
                "caution": "Request resampling, not uncertainty over reviewers; twelve authored development prompts, not population evidence.",
            },
        )
    else:
        result.update(
            blinded_review_claimed=False,
            source_assignment_predictable=True,
            abstentions_for_insufficient_usable_scenes=counts["no_se"],
            caution="Exploratory nonblinded review with two samples per model/prompt. Does not establish preservation of diversity.",
        )
    return result


def normalize(text):
    return " ".join(text.casefold().split()).rstrip(".")


def jaccard4(a, b):
    aa = re.findall(r"\w+", a.casefold())
    bb = re.findall(r"\w+", b.casefold())
    A = set(zip(aa, aa[1:], aa[2:], aa[3:]))
    B = set(zip(bb, bb[1:], bb[2:], bb[3:]))
    return len(A & B) / len(A | B) if A | B else None


def report(root):
    spec = read_json(root / "protocol.json")
    train = read_json(root / "dpo/run.json")
    pack = read_json(root / "preferences/training-pack.json")
    notes = read_json(root / "preferences/annotations.json")
    frozen = read_json(root / "preferences/review-frozen.json")
    if (
        digest(notes) != frozen["annotations_hash"]
        or pack["annotations_hash"] != frozen["annotations_hash"]
        or digest(pack["pairs"]) != pack["pairs_hash"]
    ):
        raise ValueError("Preferencias cambiadas.")
    if (
        train["status"] != "complete"
        or train["new_updates"] != len(pack["pairs"])
        or not all(
            train[k]
            for k in ["adapters_changed", "base_unchanged", "source_unchanged", "reload_exact"]
        )
    ):
        raise ValueError("Entrenamiento incompleto.")
    source = read_json(Path(spec["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    if any(sha256(origin / k) != v for k, v in train["source_files"].items()):
        raise ValueError("Cambió el origen.")
    expected_identity = digest(
        [
            spec,
            pack["pairs_hash"],
            train["source_files"],
            sha256(Path("experiments/dpo_preferences_v20.py")),
        ]
    )
    if train["identity"] != expected_identity:
        raise ValueError("Cambió el código DPO usado.")
    original_samples = {r["sample_id"]: r for r in read_jsonl(root / "train-pool/samples.jsonl")}
    allowed = {r["problem_id"] for r in spec["train_rows"]}
    for p in pack["pairs"]:
        if p["problem_id"] not in allowed or any(
            p[s] != original_samples[p[s]["sample_id"]] for s in ["chosen", "rejected"]
        ):
            raise ValueError("Un par no coincide con el muestreo de entrenamiento.")
    steps = read_jsonl(root / "dpo/steps.jsonl")
    expected_order = (
        np.random.default_rng(spec["training"]["train_seed"])
        .permutation(len(pack["pairs"]))
        .tolist()
    )
    if [s["step"] for s in steps] != list(range(1, len(pack["pairs"]) + 1)) or [
        s["pair_id"] for s in steps
    ] != [pack["pairs"][i]["pair_id"] for i in expected_order]:
        raise ValueError("Recorrido DPO distinto al fijado.")
    if any(
        not math.isfinite(s["loss"])
        or not math.isfinite(s["gradient_norm"])
        or s["gradient_norm"] <= 0
        for s in steps
    ):
        raise ValueError("Actualización inválida.")
    phases = {}
    records = {}
    requests = {r["problem_id"]: r for r in spec["dev_rows"]}
    expected = {(r["problem_id"], i) for r in spec["dev_rows"] for i in range(2)}
    for phase in ["before", "after"]:
        run = read_json(root / phase / "run.json")
        rows = read_jsonl(root / phase / "samples.jsonl")
        if (
            run["status"] != "complete"
            or not run["parameters_unchanged"]
            or run["optimizer_created"]
            or run["training_updates"]
            or run["protocol_hash"] != digest(spec)
        ):
            raise ValueError("Evaluación no congelada.")
        checkpoint = origin if phase == "before" else Path(train["checkpoint"])
        if run["checkpoint_sha256"] != sha256(checkpoint / "adapters.safetensors"):
            raise ValueError("Otro checkpoint en evaluación.")
        keyed = {(r["problem_id"], r["candidate"]): r for r in rows}
        if set(keyed) != expected or len(rows) != 24:
            raise ValueError("Muestras incompletas o duplicadas.")
        if {r["problem_id"] for r in rows} & allowed:
            raise ValueError("Un pedido entrenado se evaluó como nuevo.")
        records[phase] = keyed
        similarities = [
            jaccard4(keyed[(pid, 0)]["dream"], keyed[(pid, 1)]["dream"]) for pid in requests
        ]
        phases[phase] = {
            "n": 24,
            "format_valid": sum(r["format_valid"] for r in rows),
            "truncations": sum(r["truncated"] for r in rows),
            "dream_words_mean": statistics.mean(len(r["dream"].split()) for r in rows),
            "answer_words_mean": statistics.mean(len(r["answer"].split()) for r in rows),
            "request_echoes": sum(
                normalize(r["dream"]) == normalize(requests[r["problem_id"]]["seeds"]) for r in rows
            ),
            "identical_dream_pairs": sum(
                normalize(keyed[(pid, 0)]["dream"]) == normalize(keyed[(pid, 1)]["dream"])
                for pid in requests
            ),
            "within_prompt_word4_jaccard_mean": statistics.mean(
                x for x in similarities if x is not None
            ),
            "diversity_proxy_caution": "Lexical similarity, not semantic or oniric quality.",
        }
    for k in expected:
        a, b = records["before"][k], records["after"][k]
        if (
            a["prompt_ids"] != b["prompt_ids"]
            or a["seed"] != b["seed"]
            or a["prompt"] != requests[k[0]]["prompt"]
        ):
            raise ValueError("Par no comparable.")
    qpacket = read_json(root / "evaluation/quality-packet.json")
    qkey = {r["review_id"]: r for r in read_json(root / "evaluation/quality-key.json")}
    for row in qpacket:
        k = qkey[row["review_id"]]
        for side in ["A", "B"]:
            if row[side] != records[k[side + "_phase"]][(k["problem_id"], k["candidate"])]["dream"]:
                raise ValueError("La clave de calidad no corresponde al texto muestreado.")
    vpacket = read_json(root / "evaluation/variety-packet.json")
    vkey = {r["review_id"]: r for r in read_json(root / "evaluation/variety-key.json")}
    for row in vpacket:
        k = vkey[row["review_id"]]
        for side in ["A", "B"]:
            if row[side] != [
                records[k[side + "_phase"]][(k["problem_id"], j)]["dream"] for j in range(2)
            ]:
                raise ValueError("La clave de variedad no corresponde a las muestras.")
    quality = paired_summary(root / "evaluation", "quality")
    variety = paired_summary(root / "evaluation", "variety")
    old = Path("results/dream-v19/interventions")
    old_ann = read_json(old / "review-annotations.json")
    old_f = read_json(old / "review-frozen.json")
    new_ann = read_json(root / "dependency/annotations.json")
    new_f = read_json(root / "dependency/review-frozen.json")
    if digest(old_ann) != old_f["annotations_hash"] or digest(new_ann) != new_f["annotations_hash"]:
        raise ValueError("Etiquetas de dependencia cambiadas.")
    dependency = {}
    details = {}
    for phase, folder, ann, pname, kname in [
        ("before", old, old_ann, "review-packet.json", "review-key.json"),
        ("after", root / "dependency", new_ann, "packet.json", "key.json"),
    ]:
        packet = read_json(folder / pname)
        if digest(packet) != (old_f if phase == "before" else new_f)["packet_hash"]:
            raise ValueError("Cambió el paquete de dependencia.")
        key = {r["review_id"]: r for r in read_json(folder / kname)}
        raw = {r["task_id"]: r for r in read_jsonl(folder / "answers.jsonl")}
        evaluations = []
        for label in ann["labels"]:
            k = key[label["review_id"]]
            if k["condition"] not in {"anchor", "edited", "paraphrase"}:
                continue
            row = raw[k["task_id"]]
            if digest(row["answer"]) != label["answer_hash"]:
                raise ValueError("Respuesta de dependencia cambiada.")
            evaluations.append(
                {
                    "task_id": k["task_id"],
                    "case_id": k["case_id"],
                    "seed": k["seed"],
                    "condition": k["condition"],
                    "state": label["state"],
                    "expected": expected_state(row["probe"]["kind"], row["condition"]),
                    "match": label["state"]
                    == expected_state(row["probe"]["kind"], row["condition"]),
                    "format_valid": row["format_valid"],
                }
            )
        if len(evaluations) != 18:
            raise ValueError("Faltan 18 pruebas de dependencia.")
        details[phase] = {r["task_id"]: raw[r["task_id"]] for r in evaluations}
        keyed = {(r["case_id"], r["seed"], r["condition"]): r for r in evaluations}
        dependency[phase] = {
            "n": 18,
            "exact_probe_match": sum(r["match"] for r in evaluations),
            "by_condition": {
                c: sum(r["match"] for r in evaluations if r["condition"] == c)
                for c in ["anchor", "edited", "paraphrase"]
            },
            "paired_edit_following": sum(
                keyed[(i, s, "anchor")]["match"] and keyed[(i, s, "edited")]["match"]
                for i in ["I1", "I2", "I3"]
                for s in [17, 29]
            ),
            "format_valid": sum(r["format_valid"] for r in evaluations),
            "rows": evaluations,
        }
    if set(details["before"]) != set(details["after"]):
        raise ValueError("Tareas de dependencia distintas.")
    for tid, a in details["before"].items():
        b = details["after"][tid]
        if a["prompt_ids"] != b["prompt_ids"] or a["seed"] != b["seed"]:
            raise ValueError("Cambió una prueba de dependencia.")
    drun = read_json(root / "dependency/run.json")
    if (
        not drun["parameters_unchanged"]
        or drun["training_updates"]
        or drun["checkpoint_sha256"] != sha256(Path(train["checkpoint"]) / "adapters.safetensors")
    ):
        raise ValueError("Dependencia con otro modelo.")
    block = read_json(root / "dpo/block-logratio-diagnostic.json")
    result = {
        "training": {
            "reviewed_pairs": 24,
            "source_candidates": len(original_samples),
            "eligible_pairs": len(pack["pairs"]),
            "updates": len(steps),
            "epochs": 1,
            "source_step": 32,
            "optimizer_reset": True,
            "initial_loss": train["initial_loss"],
            "final_train_loss": train["final_train_loss"],
            "base_unchanged": True,
            "source_unchanged": True,
            "reload_exact": True,
        },
        "development_prompts": 12,
        "samples_per_prompt": 2,
        "quality": quality,
        "variety": variety,
        "before": phases["before"],
        "after": phases["after"],
        "dependency": dependency,
        "block_logratio_diagnostic": block,
        "test_reports_used": 0,
        "independent_human_review": False,
        "decision": "Keep checkpoint32 as practical reference. This fixed pilot is worse in the assistant's review; do not continue this run based on lower training loss.",
        "limitations": [
            "Same assistant reviewed training and evaluation, not independent human validation.",
            "Novel authored development prompts, not population sampling; small fixed set.",
            "Quality notes frozen before revealing identities; source ordering for variety was inferable and that review is explicitly nonblinded.",
            "Two samples per prompt do not establish preservation of diversity; five variety comparisons abstained.",
            "Probe sensitivity is local and does not establish full retelling fidelity.",
        ],
    }
    audit = {
        "passed": True,
        "train_only": True,
        "training_pairs": len(pack["pairs"]),
        "one_pass": True,
        "source_files_unchanged": True,
        "reference_cache_replayed": True,
        "max_reference_replay_difference": block["max_reference_replay_difference"],
        "base_unchanged": True,
        "reload_exact": True,
        "train_loss_and_gradients_finite": True,
        "development_distinct_from_training": True,
        "paired_contexts_and_seeds_equal": True,
        "quality_annotations_frozen": True,
        "variety_blinded_claim": False,
        "dependency_annotations_frozen": True,
        "all_generated_outputs_retained": True,
        "test_reports_used": 0,
    }
    write_json(root / "report/summary.json", result)
    write_json(root / "report/audit.json", audit)
    print("Quality:", quality["counts"], quality["exploratory_interval"])
    print("Before/after:", phases)
    print(
        "Dependency:",
        {k: (v["exact_probe_match"], v["paired_edit_following"]) for k, v in dependency.items()},
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report(args.root)
