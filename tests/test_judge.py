"""Tests for the LLM-as-judge scorer. No network: litellm is stubbed."""

import pytest

from evalx.judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeError,
    LLMJudge,
    build_judge_prompt,
)
from evalx.schemas import JudgeSpec, TestCase


def make_case() -> TestCase:
    return TestCase(
        id="q",
        question="Why did costs rise?",
        checks=[],
        judge=JudgeSpec(rubric="Names the warehouse expansion as the cause."),
    )


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _completion_ok(**kwargs):
    assert "temperature" not in kwargs  # omitted by default; some models reject it
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == "anthropic/claude-haiku-5-5"
    return _FakeResponse('{"score": 0.75, "reason": "names the cause, briefly"}')


def test_explicit_temperature_is_passed_through(monkeypatch):
    def _check_temp(**kwargs):
        assert kwargs["temperature"] == 0.5
        return _FakeResponse('{"score": 1.0, "reason": "ok"}')

    monkeypatch.setattr("evalx.judge.litellm.completion", _check_temp)
    verdict = LLMJudge("some-model", temperature=0.5).score(make_case(), "x")
    assert verdict.score == 1.0


def test_judge_scores_answer(monkeypatch):
    monkeypatch.setattr("evalx.judge.litellm.completion", _completion_ok)
    verdict = LLMJudge("anthropic/claude-haiku-5-5").score(make_case(), "costs rose")
    assert verdict.score == 0.75
    assert "cause" in verdict.reason


def test_prompt_contains_case_answer_and_rubric():
    prompt = build_judge_prompt(make_case(), "costs rose because X")
    assert "Why did costs rise?" in prompt
    assert "costs rose because X" in prompt
    assert "warehouse expansion" in prompt


def test_system_prompt_guards_verbosity_bias():
    assert "fluff" in JUDGE_SYSTEM_PROMPT
    assert "Brevity is fine" in JUDGE_SYSTEM_PROMPT


def test_malformed_json_retries_then_raises(monkeypatch):
    calls = []

    def _bad(**kwargs):
        calls.append(1)
        return _FakeResponse("not json at all")

    monkeypatch.setattr("evalx.judge.litellm.completion", _bad)
    with pytest.raises(JudgeError):
        LLMJudge("some-model", max_retries=1).score(make_case(), "x")
    assert len(calls) == 2  # initial attempt + 1 retry


def test_out_of_range_score_rejected(monkeypatch):
    def _oor(**kwargs):
        return _FakeResponse('{"score": 1.5, "reason": "too generous"}')

    monkeypatch.setattr("evalx.judge.litellm.completion", _oor)
    with pytest.raises(JudgeError):
        LLMJudge("some-model", max_retries=0).score(make_case(), "x")


def test_case_without_rubric_raises():
    with pytest.raises(JudgeError):
        LLMJudge("some-model").score(TestCase(id="q", question="?"), "x")
