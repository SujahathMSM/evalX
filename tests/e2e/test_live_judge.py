"""Opt-in live check: one real judge model scores clear answers sensibly.

Skipped unless ``ANTHROPIC_API_KEY`` is set (override the model with
``EVALX_LIVE_JUDGE_MODEL``). Costs a fraction of a cent; run it by hand
before a release, not on every push::

    ANTHROPIC_API_KEY=... python -m pytest -m live
"""

import os

import pytest

from evalx.judge import LLMJudge
from evalx.schemas import JudgeSpec, TestCase

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="live judge test needs ANTHROPIC_API_KEY"
    ),
]

MODEL = os.environ.get("EVALX_LIVE_JUDGE_MODEL", "anthropic/claude-haiku-5-5")

CASE = TestCase(
    id="live-sky",
    question="Why is the sky blue?",
    judge=JudgeSpec(rubric="Full marks only if the answer attributes it to Rayleigh scattering."),
)


def test_live_judge_passes_a_correct_answer():
    verdict = LLMJudge(MODEL).score(
        CASE, "Sunlight is Rayleigh-scattered by air molecules; blue scatters most."
    )
    assert verdict.score >= 0.7, verdict.reason


def test_live_judge_fails_a_wrong_answer():
    verdict = LLMJudge(MODEL).score(CASE, "Because the sky reflects the colour of the ocean.")
    assert verdict.score <= 0.3, verdict.reason


def test_live_judge_reports_usage_and_cost():
    verdict = LLMJudge(MODEL).score(CASE, "Rayleigh scattering.")
    assert verdict.usage is not None
    assert verdict.usage.prompt_tokens and verdict.usage.prompt_tokens > 0
    assert verdict.usage.completion_tokens and verdict.usage.completion_tokens > 0
    # LiteLLM prices Anthropic models; a few hundred tokens cost well under a cent.
    assert verdict.usage.cost_usd is not None
    assert 0 < verdict.usage.cost_usd < 0.01
