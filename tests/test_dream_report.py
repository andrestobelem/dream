from think_dream_lab.dream_report import blind_review, save_blind_review
from think_dream_lab.dream_workflow import summarize
from think_dream_lab.storage import append_jsonl


def test_reporting_never_overwrites_human_annotations_or_their_key(tmp_path):
    import csv

    from think_dream_lab.report import write_csv
    from think_dream_lab.storage import read_json

    original = [{"item_id": "one", "relato": "A scene.", "onirico_humano_0_a_4": "", "notas": ""}]
    original_keys = {"one": {"judge_score": 2}}
    save_blind_review(tmp_path, original, original_keys)
    original[0].update(onirico_humano_0_a_4="4", notas="Evaluación humana")
    write_csv(tmp_path / "revision-ciega.csv", original)
    new_items = [
        {**original[0], "relato": "A different scene.", "onirico_humano_0_a_4": "", "notas": ""}
    ]
    kept, keys = save_blind_review(tmp_path, new_items, {"one": {"judge_score": 3}})
    assert kept == original and keys == original_keys
    with (tmp_path / "revision-ciega.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == original
    assert read_json(tmp_path / "clave-revision.json") == original_keys
    assert len(list(tmp_path.glob("revision-ciega-*.csv"))) == 1


def test_blind_packet_excludes_origin_and_judge_scores(tmp_path):
    append_jsonl(
        tmp_path / "data" / "validation.jsonl",
        {
            "problem_id": "one",
            "reference": "A human account of a strange place.",
        },
    )
    examples = [
        {
            "problem_id": "one",
            "answer": "An imagined account of another place.",
            "phase": "after",
            "condition": "D",
            "dream_likeness": 0.75,
        }
    ]
    items, keys = blind_review(tmp_path, examples)
    assert len(items) == 2
    assert all(set(row) == {"item_id", "relato", "onirico_humano_0_a_4", "notas"} for row in items)
    assert all(row["onirico_humano_0_a_4"] == "" for row in items)
    assert {k["origin"] for k in keys.values()} == {"generated", "human_reference"}
    assert items == blind_review(tmp_path, examples)[0]


def test_quality_is_separate_from_output_coverage():
    records = []
    for valid in [False, True]:
        reward = {
            key: 0.75 if valid else 0
            for key in [
                "dream_likeness",
                "reference_alignment",
                "retelling_fidelity",
                "quality",
                "format",
                "total",
            ]
        }
        reward["answer_words"] = 30 if valid else 0
        records.append({"reward": reward, "tokens": 10, "latency_seconds": 1, "truncated": False})
    result = summarize(records)
    assert result["n_valid_answers"] == 1 and result["n"] == 2
    assert result["dream_likeness"] == 0.375
    assert result["valid_dream_likeness"] == 0.75


def test_report_preserves_unscored_values_and_counts(tmp_path):
    import csv

    from think_dream_lab.dream_report import report
    from think_dream_lab.storage import read_json, write_json

    dream = tmp_path / "dream"
    folder = dream / "evaluations/probe"
    write_json(
        folder / "run.json",
        {
            "status": "complete",
            "kind": "dream_evaluation",
            "seed": 4242,
            "config": {"judge": {"rubric_version": "dream-v4"}},
        },
    )
    append_jsonl(
        dream / "data/validation.jsonl", {"problem_id": "p", "reference": "A human scene."}
    )
    append_jsonl(
        folder / "responses.jsonl",
        {
            "phase": "before",
            "step": 0,
            "condition": "D",
            "problem_id": "p",
            "candidate": 0,
            "text": "<dream>A.</dream><answer>A retelling.</answer>",
            "tokens": 10,
            "latency_seconds": 1,
            "truncated": False,
            "reward": {
                "dream_likeness": 0.75,
                "reference_alignment": None,
                "content_compliance": 0.5,
                "retelling_fidelity": None,
                "quality": None,
                "format": 1,
                "total": None,
                "answer_words": 2,
                "score_status": "abstained",
            },
        },
    )
    report(tmp_path)
    with (dream / "report/evaluations.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["total"] == "" and row["retelling_fidelity"] == ""
    assert row["n_reward_scored"] == "0" and row["n_abstained"] == "1"
    assert read_json(dream / "report/examples.json")[0]["total"] is None


def test_paired_deltas_use_only_available_pairs_with_their_counts():
    from think_dream_lab.dream_report import paired_statistics

    before = {"p1": {"reward": {"quality": 0.4}}, "p2": {"reward": {"quality": 0.8}}}
    after = {"p1": {"reward": {"quality": 0.6}}, "p2": {"reward": {"quality": None}}}
    result = paired_statistics(before, after, before.keys())
    assert result["n_pairs_quality"] == 1 and abs(result["delta_quality"] - 0.2) < 1e-9
    assert (
        result["n_pairs_reference_alignment"] == 0 and result["delta_reference_alignment"] is None
    )
