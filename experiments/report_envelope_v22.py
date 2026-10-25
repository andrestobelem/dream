"""Paquetes de dependencia y auditoría del sistema de inferencia, no del aprendizaje."""

import random
import secrets
from collections import Counter
from pathlib import Path
from statistics import mean

from experiments.criteria_preferences_v21 import promotion_decision
from experiments.envelope_v22 import ROOT
from experiments.preferences_v20 import locked_write
from experiments.report_dream_v19 import expected_state
from experiments.report_preferences_v20 import frozen_review
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import digest, read_json, read_jsonl, write_json


def dependency_packet(root):
    folder = root / "dependency"
    if (folder / "packet.json").exists():
        return
    rows = [(p, r) for p in ["before", "after"] for r in read_jsonl(folder / p / "samples.jsonl")]
    assert len(rows) == 36
    random.Random(secrets.randbits(256)).shuffle(rows)
    packet, key = [], []
    for i, (p, r) in enumerate(rows):
        rid = f"P{i + 1:02d}"
        packet.append({"review_id": rid, "kind": r["kind"], "answer": r["answer"]})
        key.append({"review_id": rid, "phase": p, "task_id": r["task_id"]})
    locked_write(folder / "key.json", key)
    locked_write(folder / "packet.json", packet)
    print("Dependency packet: 36 answers; model and condition hidden.")


def freeze_dependency(root):
    folder = root / "dependency"
    packet = read_json(folder / "packet.json")
    notes = read_json(folder / "annotations.json")
    assert len(notes["labels"]) == len(packet) == 36
    by_id = {r["review_id"]: r for r in packet}
    assert {r["review_id"] for r in notes["labels"]} == set(by_id)
    for r in notes["labels"]:
        assert digest(by_id[r["review_id"]]["answer"]) == r["answer_hash"]
    locked_write(
        folder / "review-frozen.json",
        {
            "packet_hash": digest(packet),
            "annotations_hash": digest(notes),
            "model_and_condition_key_consulted_before_freeze": False,
            "independent_human_review": False,
            "answers": 36,
        },
    )


def report(root):
    p = read_json(root / "protocol.json")
    assert p["sample_code_sha256"] == sha256(Path("experiments/envelope_v22.py"))
    assert p["controller_code_sha256"] == sha256(Path("think_dream_lab/dream_envelope.py"))
    assert p["criterion_code_sha256"] == sha256(Path("experiments/criteria_preferences_v21.py"))
    origin = Path(p["checkpoints"]["before"])
    assert all(sha256(origin / n) == v for n, v in p["source_files"].items())
    expected = {(r["problem_id"], i) for r in p["dev_rows"] for i in range(4)}
    records, diag, intervention = {}, {}, {}
    for phase in ["before", "after"]:
        assert (
            sha256(Path(p["checkpoints"][phase]) / "adapters.safetensors")
            == p["checkpoint_hashes"][phase]
        )
        run = read_json(root / phase / "run.json")
        assert (
            run["status"] == "complete"
            and run["parameters_unchanged"]
            and run["training_updates"] == 0
        )
        assert run["identity"] == digest([p, phase, False])
        raw = read_jsonl(root / phase / "samples.jsonl")
        assert len(raw) == 96 and all(r["identity"] == run["identity"] for r in raw)
        records[phase] = {(r["problem_id"], r["candidate"]): r for r in raw}
        assert set(records[phase]) == expected
        diag[phase] = {
            "format_valid": sum(r["format_valid"] for r in raw),
            "syntax_valid": sum(r["syntax_valid"] for r in raw),
            "truncated": sum(r["truncated"] for r in raw),
            "dream_words_mean": mean(len(r["dream"].split()) for r in raw),
            "answer_words_mean": mean(len(r["answer"].split()) for r in raw),
            "motif_major": 0,
            "groups_fewer_than_two_episodes": 0,
        }
        intervention[phase] = {}
        for stage in ["dream", "answer"]:
            ss = [r["stages"][stage] for r in raw]
            intervention[phase][stage] = {
                "stop_reasons": dict(Counter(s["stop_reason"] for s in ss)),
                "nonempty": sum(bool(s["content"]) for s in ss),
                "complete": sum(s["complete"] for s in ss),
                "model_emitted_expected_delimiter": sum(
                    s["stop_reason"] == f"</{stage}>" for s in ss
                ),
                "all_envelope_tags_written_by_program": True,
            }
    requests = {r["problem_id"]: r for r in p["dev_rows"]}
    for k in expected:
        a, b = [records[s][k] for s in ["before", "after"]]
        assert (
            a["prompt_ids"] == b["prompt_ids"]
            and a["seed"] == b["seed"] == p["evaluation"]["seeds"][k[1]]
        )
        assert a["prompt"] == b["prompt"] == requests[k[0]]["prompt"]
    qp, qn, qf = frozen_review(root / "evaluation", "quality")
    vp, vn, vf = frozen_review(root / "evaluation", "variety")
    dp = read_json(root / "dependency/packet.json")
    dn = read_json(root / "dependency/annotations.json")
    df = read_json(root / "dependency/review-frozen.json")
    assert digest(dp) == df["packet_hash"] and digest(dn) == df["annotations_hash"]
    # Sólo se consultan claves después de comprobar todas las congelaciones.
    qkey = {r["review_id"]: r for r in read_json(root / "evaluation/quality-key.json")}
    vkey = {r["review_id"]: r for r in read_json(root / "evaluation/variety-key.json")}
    dkey = {r["review_id"]: r for r in read_json(root / "dependency/key.json")}
    qpacket = {r["review_id"]: r for r in qp}
    vpacket = {r["review_id"]: r for r in vp}
    ratings = []
    for r in qn["ratings"]:
        k = qkey[r["review_id"]]
        for s in "AB":
            raw = records[k[s + "_phase"]][(k["problem_id"], k["candidate"])]
            assert qpacket[r["review_id"]][s] == raw["dream"]
            assert qpacket[r["review_id"]][s + "_format"] == raw["format_valid"]
            diag[k[s + "_phase"]]["motif_major"] += int(r["motifs"][s]["major"])
        preference = k[r["preference"] + "_phase"] if r["preference"] in "AB" else r["preference"]
        ratings.append(
            {"problem_id": k["problem_id"], "candidate": k["candidate"], "preference": preference}
        )
    variety = Counter()
    for r in vn["ratings"]:
        k = vkey[r["review_id"]]
        for s in "AB":
            assert Counter(vpacket[r["review_id"]][s]) == Counter(
                records[k[s + "_phase"]][(k["problem_id"], i)]["dream"] for i in range(4)
            )
            diag[k[s + "_phase"]]["groups_fewer_than_two_episodes"] += int(
                r["usable_episodes"][s] < 2
            )
        variety[k[r["preference"] + "_phase"] if r["preference"] in "AB" else r["preference"]] += 1
    plan = read_json(Path(p["dependency"]["source_plan"]))
    tasks = {
        t["task_id"]: t for t in plan["tasks"] if t["condition"] in p["dependency"]["conditions"]
    }
    deps, dep_raw = {}, {}
    for phase in ["before", "after"]:
        run = read_json(root / "dependency" / phase / "run.json")
        raw = read_jsonl(root / "dependency" / phase / "samples.jsonl")
        assert run["status"] == "complete" and run["identity"] == digest([p, phase, True])
        assert run["parameters_unchanged"] and run["training_updates"] == 0
        dep_raw[phase] = {r["task_id"]: r for r in raw}
        assert len(raw) == 18 and set(dep_raw[phase]) == set(tasks)
        assert all(r["identity"] == run["identity"] for r in raw)
        deps[phase] = {
            "full": 0,
            "format_valid": sum(r["format_valid"] for r in raw),
            "stop_reasons": dict(Counter(r["stages"]["answer"]["stop_reason"] for r in raw)),
            "rows": [],
        }
    for t in tasks:
        a, b = [dep_raw[s][t] for s in ["before", "after"]]
        assert a["prompt_ids"] == b["prompt_ids"] and a["seed"] == b["seed"] == tasks[t]["seed"]
    for label in dn["labels"]:
        k = dkey[label["review_id"]]
        raw = dep_raw[k["phase"]][k["task_id"]]
        assert digest(raw["answer"]) == label["answer_hash"]
        target = expected_state(raw["kind"], tasks[k["task_id"]]["condition"])
        match = label["state"] == target
        deps[k["phase"]]["full"] += int(match)
        deps[k["phase"]]["rows"].append(
            {
                "review_id": label["review_id"],
                "task_id": k["task_id"],
                "state": label["state"],
                "expected": target,
                "match": match,
            }
        )
    for phase, value in diag.items():
        assert len(deps[phase]["rows"]) == 18
        value.update(
            dependency_full=deps[phase]["full"], dependency_format=deps[phase]["format_valid"]
        )
    decision = promotion_decision(
        p, ratings, {**diag, "variety_counts": dict(variety), "all_reviews_frozen": True}
    )
    result = {
        "quality": decision["quality"],
        "promotion": decision,
        **diag,
        "variety_counts": dict(variety),
        "dependency": deps,
        "controller": intervention,
        "training_updates": 0,
        "test_reports_used": 0,
        "independent_human_review": False,
        "decision": "candidate_v21_with_envelope" if decision["promote"] else "keep32",
        "limitations": [
            "Envoltura del programa, no cierre aprendido; contenido sin reescribir pero fronteras separadas y prefijo de answer recodificado.",
            "Mismo asistente revisa; desarrollo nuevo, no validación humana independiente.",
            "Las 18 pruebas de dependencia son regresiones conocidas y se regeneraron para ambos sistemas.",
            "El intervalo por pedido no incorpora incertidumbre del revisor; comparación bajo generación nueva, no causal entre rondas.",
        ],
    }
    audit = {
        "passed": True,
        "protocol_hash": digest(p),
        "quality_hash": qf["annotations_hash"],
        "variety_hash": vf["annotations_hash"],
        "dependency_hash": df["annotations_hash"],
        "source_files_unchanged": True,
        "both_checkpoint_adapters_unchanged": True,
        "same_controller_on_both": True,
        "training_updates": 0,
        "test_reports_used": 0,
        "result_hash": digest(result),
    }
    write_json(root / "report/summary.json", result)
    write_json(root / "report/audit.json", audit)
    public = {
        **result,
        "dependency": {s: {k: v for k, v in d.items() if k != "rows"} for s, d in deps.items()},
    }
    write_json(Path("docs/envoltura-v22-resultados.json"), public)
    write_json(Path("docs/envoltura-v22-auditoria.json"), audit)
    write_json(
        root / "selection.json",
        {
            "decision": result["decision"],
            "checkpoint": p["checkpoints"]["after" if decision["promote"] else "before"],
            "evaluated_controller_sha256": p["controller_code_sha256"],
            "activate_new_controller": decision["promote"],
            "interpretation": "candidate only with new envelope"
            if decision["promote"]
            else "existing32 reference retained; new envelope not promoted",
        },
    )
    print("Decision:", decision)
    print("Diagnostics:", diag, "variety:", dict(variety))


if __name__ == "__main__":
    report(ROOT)
