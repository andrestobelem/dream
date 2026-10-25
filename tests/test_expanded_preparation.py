import pytest

from think_dream_lab.__main__ import main
from think_dream_lab.expanded_preparation import balanced_selection


def make_rows():
    return [
        {
            "problem_id": f"{g}-{i}",
            "author_group": g,
            "reference": f"Report from {g} number {i}",
            "split": "train",
        }
        for g in ["a", "b", "c"]
        for i in range(8)
    ]


def test_balancing_retains_old_reports_and_is_independent_of_input_order():
    rows = make_rows()
    retained = [rows[0], rows[8]]
    selected, quotas = balanced_selection(rows, retained, {"heldout"}, {"excluded"}, count=11)
    reverse, _ = balanced_selection(
        list(reversed(rows)), retained, {"heldout"}, {"excluded"}, count=11
    )
    assert selected == reverse and len(selected) == 11
    assert sorted(quotas.values()) == [3, 4, 4]
    assert {r["problem_id"] for r in retained} <= {r["problem_id"] for r in selected}


def test_reserved_person_or_report_is_rejected_instead_of_silently_selected():
    rows = make_rows()
    with pytest.raises(ValueError, match="reservado"):
        balanced_selection(rows, [], {"b"}, set(), count=6)
    with pytest.raises(ValueError, match="reservado"):
        balanced_selection(rows, [], set(), {"a-1"}, count=6)


def test_changed_retained_reference_cannot_be_reused():
    rows = make_rows()
    with pytest.raises(ValueError, match="Cambió"):
        balanced_selection(rows, [{**rows[0], "reference": "changed"}], set(), set(), count=6)


def test_preparation_profile_cannot_start_training():
    with pytest.raises(ValueError, match="entrenar está prohibido"):
        main(["experiment", "--config", "configs/dream-expanded-v16.yaml", "--steps", "16"])
