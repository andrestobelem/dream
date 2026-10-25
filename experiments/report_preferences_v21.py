"""Audita el ensayo cerrado y aplica el criterio congelado sin cambiar umbrales."""

import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.criteria_preferences_v21 import promotion_decision
from experiments.preferences_v21 import ROOT
from experiments.report_dream_v19 import expected_state
from experiments.report_preferences_v20 import frozen_review, normalize
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import digest, read_json, read_jsonl, write_json


def dependency(root):
    old = Path("results/dream-v19/interventions")
    results, details = {}, {}
    for phase, folder, aname, pname, kname in [
        ("before", old, "review-annotations.json", "review-packet.json", "review-key.json"),
        ("after", root / "dependency", "annotations.json", "packet.json", "key.json"),
    ]:
        notes = read_json(folder / aname)
        frozen = read_json(folder / "review-frozen.json")
        packet = read_json(folder / pname)
        assert (
            digest(notes) == frozen["annotations_hash"] and digest(packet) == frozen["packet_hash"]
        )
        key = {r["review_id"]: r for r in read_json(folder / kname)}
        raw = {r["task_id"]: r for r in read_jsonl(folder / "answers.jsonl")}
        evaluated = []
        details[phase] = {}
        for label in notes["labels"]:
            k = key[label["review_id"]]
            if k["condition"] not in {"anchor", "edited", "paraphrase"}:
                continue
            r = raw[k["task_id"]]
            assert digest(r["answer"]) == label["answer_hash"]
            target = expected_state(r["probe"]["kind"], r["condition"])
            evaluated.append(
                {
                    "task_id": r["task_id"],
                    "condition": r["condition"],
                    "state": label["state"],
                    "expected": target,
                    "match": label["state"] == target,
                    "format_valid": r["format_valid"],
                }
            )
            details[phase][r["task_id"]] = r
        assert len(evaluated) == 18 and len(details[phase]) == 18
        results[phase] = {
            "full": sum(r["match"] for r in evaluated),
            "format_valid": sum(r["format_valid"] for r in evaluated),
            "by_condition": {
                c: sum(r["match"] for r in evaluated if r["condition"] == c)
                for c in ["anchor", "edited", "paraphrase"]
            },
            "rows": evaluated,
        }
    assert set(details["before"]) == set(details["after"])
    for tid, r in details["before"].items():
        assert r["prompt_ids"] == details["after"][tid]["prompt_ids"]
        assert r["seed"] == details["after"][tid]["seed"]
    return results


def report(root):
    spec = read_json(root / "protocol.json")
    integrity = read_json(root / "integrity-audit.json")
    assert digest(spec) == integrity["protocol_hash"]
    assert (
        sha256(Path("experiments/criteria_preferences_v21.py"))
        == integrity["criterion_code_sha256"]
    )
    pack = read_json(root / "preferences/training-pack.json")
    frozen = read_json(root / "preferences/review-frozen.json")
    assert digest(read_json(root / "preferences/annotations.json")) == frozen["annotations_hash"]
    assert digest(pack["pairs"]) == pack["pairs_hash"] == integrity["training_pairs_hash"]
    train = read_json(root / "dpo/run.json")
    assert train["status"] == "complete" and train["new_updates"] == len(pack["pairs"]) == 19
    assert all(
        train[k] for k in ["adapters_changed", "base_unchanged", "source_unchanged", "reload_exact"]
    )
    assert train["identity"] == digest(
        [
            spec,
            pack["pairs_hash"],
            train["source_files"],
            sha256(Path("experiments/dpo_preferences_v20.py")),
        ]
    )
    origin = Path(read_json(Path(spec["source_run"]))["training_validation"]["checkpoint"])
    assert all(sha256(origin / n) == v for n, v in train["source_files"].items())
    original = {r["sample_id"]: r for r in read_jsonl(root / "train-pool/samples.jsonl")}
    for p in pack["pairs"]:
        assert all(p[s] == original[p[s]["sample_id"]] for s in ["chosen", "rejected"])
    steps = read_jsonl(root / "dpo/steps.jsonl")
    order = np.random.default_rng(spec["training"]["train_seed"]).permutation(19).tolist()
    assert [r["step"] for r in steps] == list(range(1, 20))
    assert [r["pair_id"] for r in steps] == [pack["pairs"][i]["pair_id"] for i in order]
    assert all(
        math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0
        for r in steps
    )
    records, metrics = {}, {}
    expected = {(r["problem_id"], i) for r in spec["dev_rows"] for i in range(4)}
    requests = {r["problem_id"]: r for r in spec["dev_rows"]}
    for phase in ["before", "after"]:
        run = read_json(root / phase / "run.json")
        checkpoint = origin if phase == "before" else Path(train["checkpoint"])
        assert (
            run["status"] == "complete"
            and run["parameters_unchanged"]
            and run["training_updates"] == 0
        )
        assert run["checkpoint_sha256"] == sha256(checkpoint / "adapters.safetensors")
        assert run["identity"] == digest(
            [
                spec,
                phase,
                run["checkpoint_sha256"],
                sha256(Path("experiments/evaluate_preferences_v21.py")),
            ]
        )
        rows = read_jsonl(root / phase / "samples.jsonl")
        assert len(rows) == 96 and all(r["identity"] == run["identity"] for r in rows)
        records[phase] = {(r["problem_id"], r["candidate"]): r for r in rows}
        assert set(records[phase]) == expected
        for r in rows:
            assert r["prompt"] == requests[r["problem_id"]]["prompt"]
            assert r["seed"] == spec["evaluation"]["seeds"][r["candidate"]]
        metrics[phase] = {
            "format_valid": sum(r["format_valid"] for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
            "prompt_only_echoes": sum(
                normalize(r["dream"]) == normalize(requests[r["problem_id"]]["seeds"]) for r in rows
            ),
            "dream_words_mean": statistics.mean(len(r["dream"].split()) for r in rows),
            "answer_words_mean": statistics.mean(len(r["answer"].split()) for r in rows),
        }
    for k in expected:
        assert records["before"][k]["prompt_ids"] == records["after"][k]["prompt_ids"]
    quality_packet, qnotes, qf = frozen_review(root / "evaluation", "quality")
    variety_packet, vnotes, vf = frozen_review(root / "evaluation", "variety")
    # Las claves se consultan sólo después de verificar las dos revisiones congeladas.
    qkey = {r["review_id"]: r for r in read_json(root / "evaluation/quality-key.json")}
    vkey = {r["review_id"]: r for r in read_json(root / "evaluation/variety-key.json")}
    qpacket = {r["review_id"]: r for r in quality_packet}
    vpacket = {r["review_id"]: r for r in variety_packet}
    ratings = []
    for value in metrics.values():
        value["motif_major"] = 0
        value["groups_fewer_than_two_episodes"] = 0
    for r in qnotes["ratings"]:
        k = qkey[r["review_id"]]
        for s in "AB":
            raw = records[k[s + "_phase"]][(k["problem_id"], k["candidate"])]
            assert qpacket[r["review_id"]][s] == raw["dream"]
            assert qpacket[r["review_id"]][s + "_format"] == raw["format_valid"]
            metrics[k[s + "_phase"]]["motif_major"] += int(r["motifs"][s]["major"])
        preference = k[r["preference"] + "_phase"] if r["preference"] in "AB" else r["preference"]
        ratings.append(
            {"problem_id": k["problem_id"], "candidate": k["candidate"], "preference": preference}
        )
    variety = Counter()
    for r in vnotes["ratings"]:
        k = vkey[r["review_id"]]
        for s in "AB":
            assert Counter(vpacket[r["review_id"]][s]) == Counter(
                records[k[s + "_phase"]][(k["problem_id"], i)]["dream"] for i in range(4)
            )
            metrics[k[s + "_phase"]]["groups_fewer_than_two_episodes"] += int(
                r["usable_episodes"][s] < 2
            )
        preference = k[r["preference"] + "_phase"] if r["preference"] in "AB" else r["preference"]
        variety[preference] += 1
    deps = dependency(root)
    drun = read_json(root / "dependency/run.json")
    assert drun["parameters_unchanged"] and drun["training_updates"] == 0
    assert drun["checkpoint_sha256"] == sha256(Path(train["checkpoint"]) / "adapters.safetensors")
    metrics["after"].update(
        dependency_full=deps["after"]["full"], dependency_format=deps["after"]["format_valid"]
    )
    diagnostics = {**metrics, "variety_counts": dict(variety), "all_reviews_frozen": True}
    decision = promotion_decision(spec, ratings, diagnostics)
    result = {
        "training": train,
        "development_prompts": 24,
        "samples_per_prompt": 4,
        **metrics,
        "quality": decision["quality"],
        "variety_counts": dict(variety),
        "dependency": deps,
        "promotion": decision,
        "test_reports_used": 0,
        "independent_human_review": False,
        "limitations": [
            "Mismo asistente prepara y revisa; no es validación humana independiente.",
            "Desarrollo nuevo escrito por el asistente; el intervalo por pedido no incorpora incertidumbre entre revisores.",
            "Calidad y motivos son ejes distintos; no se mide cercanía a un único relato humano esperado.",
            "Variedad y dependencia son controles locales, no pruebas poblacionales ni experiencia subjetiva.",
        ],
    }
    audit = {
        "passed": True,
        "protocol_hash": digest(spec),
        "pairs_hash": pack["pairs_hash"],
        "training_updates": 19,
        "one_pass": True,
        "source_unchanged": True,
        "quality_annotations_hash": qf["annotations_hash"],
        "variety_annotations_hash": vf["annotations_hash"],
        "full96_comparisons": True,
        "quality_and_variety_keys_match_original_samples": True,
        "criterion_code_unchanged": True,
        "test_reports_used": 0,
        "result_hash": digest(result),
    }
    write_json(root / "report/summary.json", result)
    write_json(root / "report/audit.json", audit)
    # Sólo agregados; ni notas con textos completos ni claves se publican en Git.
    public = {
        **result,
        "dependency": {p: {k: v for k, v in r.items() if k != "rows"} for p, r in deps.items()},
    }
    write_json(Path("docs/preferencias-v21-resultados.json"), public)
    write_json(Path("docs/preferencias-v21-resultados-auditoria.json"), audit)
    print("Decision:", decision)
    print("Diagnostics:", diagnostics)


if __name__ == "__main__":
    report(ROOT)
