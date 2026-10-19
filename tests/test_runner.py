"""Tests for the runner: the harness must catch the demo app's failures."""

import pytest

from evalx.adapters import AdapterError, resolve_target
from evalx.runner import run_suite

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"


def test_run_demo_suite():
    result = run_suite(DEMO, TARGET)
    assert result.n_cases == 6
    assert result.suite == DEMO
    assert result.target == TARGET
    scores = {r.case_id: r.score for r in result.results}
    # Easy cases pass fully.
    assert scores["q001"] == 1.0
    assert scores["q002"] == 1.0
    # Medium cases get partial credit (one of two checks each).
    assert scores["q003"] == 0.5
    assert scores["q006"] == 0.5
    # Adversarial cases: the demo app hallucinates, scores 0.
    assert scores["q004"] == 0.0
    assert scores["q005"] == 0.0
    # Mean = (1 + 1 + 0.5 + 0 + 0 + 0.5) / 6 = 0.5
    assert result.mean_score == 0.5


def test_latency_recorded():
    result = run_suite(DEMO, TARGET)
    assert all(r.latency_ms >= 0 for r in result.results)


def test_bad_target_raises():
    with pytest.raises(AdapterError):
        resolve_target("no-colon-here")


def test_missing_module_raises():
    with pytest.raises(AdapterError):
        resolve_target("no_such_module:fn")


def test_non_callable_raises():
    with pytest.raises(AdapterError):
        resolve_target("evalx:__version__")  # a string, not a q->a function
