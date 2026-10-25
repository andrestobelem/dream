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
