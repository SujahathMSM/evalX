"""Tests for golden dataset loading."""

import pytest

from evalx.datasets import DatasetError, load_dataset

DEMO = "datasets/demo/v1"


def test_loads_demo_dataset():
    cases = load_dataset(DEMO)
    assert len(cases) == 6
    assert [c.id for c in cases] == ["q001", "q002", "q003", "q004", "q005", "q006"]


def test_case_fields():
    cases = {c.id: c for c in load_dataset(DEMO)}
    assert cases["q001"].difficulty == "easy"
    assert "adversarial" in cases["q004"].tags
    assert cases["q003"].checks[0].weight == 1.0


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nope")


def test_not_a_list_raises(tmp_path):
    p = tmp_path / "cases.yaml"
    p.write_text("id: q1\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(p)


def test_invalid_case_raises(tmp_path):
    p = tmp_path / "cases.yaml"
    p.write_text("- question: 'no id here'\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(p)


def test_duplicate_ids_raise(tmp_path):
    p = tmp_path / "cases.yaml"
    p.write_text(
        "- {id: q1, question: 'a'}\n- {id: q1, question: 'b'}\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError):
        load_dataset(p)
