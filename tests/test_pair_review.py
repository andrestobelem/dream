import csv

import pytest

from experiments.prepare_pair_review import write_packet


def example():
    return {
        "source_group": "scene-1",
        "dimension": "fidelidad",
        "requirement": "I floated.",
        "candidate": "I remember floating.",
        "synthetic_expected": "preserved",
    }


def test_reexport_preserves_human_notes_and_does_not_reveal_key(tmp_path):
    path = write_packet(tmp_path, [example()], {"source": "hash"})
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "synthetic_expected" not in rows[0] and "source_group" not in rows[0]
    rows[0].update(etiqueta_humana="no_se", notas="Revisar contexto")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    annotated = path.read_bytes()
    assert write_packet(tmp_path, [example()], {"source": "hash"}) == path
    assert path.read_bytes() == annotated


def test_modified_review_text_is_never_silently_accepted_or_overwritten(tmp_path):
    path = write_packet(tmp_path, [example()], {})
    changed = path.read_text().replace("I floated.", "I sank.")
    path.write_text(changed)
    with pytest.raises(ValueError, match="no se sobrescribe"):
        write_packet(tmp_path, [example()], {})
    assert path.read_text() == changed
