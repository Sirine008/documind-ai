import pytest

from app.evaluation import metrics as m


def test_recall_at_k():
    assert m.recall_at_k([False, True, False], 1) == 0.0
    assert m.recall_at_k([False, True, False], 2) == 1.0
    assert m.recall_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert m.reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert m.reciprocal_rank([True]) == 1.0
    assert m.reciprocal_rank([False, False]) == 0.0


def test_is_relevant_requires_all_evidence_phrases():
    text = "The  stipend is\n600 euros, paid once."
    assert m.is_relevant(text, ["600 euros"])
    assert m.is_relevant(text, ["stipend is 600", "paid once"])
    assert not m.is_relevant(text, ["600 euros", "monthly"])
    assert not m.is_relevant(text, [])


def test_groundedness():
    assert m.groundedness("The stipend is 600 euros", ["Employees receive a stipend of 600 euros"]) == pytest.approx(
        1.0
    )
    assert m.groundedness("Bananas are yellow", ["The stipend is 600 euros"]) == 0.0
    assert m.groundedness("anything", []) == 0.0


def test_reference_term_recall():
    assert m.reference_term_recall("It is 600 euros", "600 euros") == 1.0
    assert m.reference_term_recall("It is 600", "600 euros") == 0.5


def test_mean_and_percentile():
    assert m.mean([1.0, 0.0, None]) == 0.5
    assert m.mean([]) is None
    assert m.percentile([10, 20, 30, 40], 50) == 20
    assert m.percentile([10, 20, 30, 40], 95) == 40
    assert m.percentile([], 50) is None
