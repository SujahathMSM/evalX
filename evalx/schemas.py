"""Pydantic schemas: the shapes of test cases, scores, and runs."""

from typing import Literal

from pydantic import BaseModel, Field


class Check(BaseModel):
    """One verifiable expectation about an answer."""

    type: Literal["exact_match", "contains", "regex"]
    expected: str
    weight: float = Field(default=1.0, gt=0)


class JudgeSpec(BaseModel):
    """Rubric for the LLM judge on one case."""

    rubric: str
    weight: float = Field(default=1.0, gt=0)


class TestCase(BaseModel):
    """One golden test case: a question plus its answer key."""

    __test__ = False  # not a pytest test class; silence collection warning

    id: str
    question: str
    checks: list[Check] = []
    judge: JudgeSpec | None = None
    tags: list[str] = []
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class CheckScore(BaseModel):
    """One check's outcome on one answer, with the expectation it graded."""

    type: str
    expected: str
    weight: float
    score: float


class CaseResult(BaseModel):
    """What happened when one case ran."""

    case_id: str
    question: str
    answer: str
    check_scores: list[float]
    checks: list[CheckScore] = []
    judge_score: float | None = None
    judge_reason: str | None = None
    score: float
    latency_ms: float
    # Set when the target or the judge raised on this case. The case still
    # counts in the mean with score 0.0, so a crashing app can't pass a gate.
    error: str | None = None
    # What the LLM judge consumed on this case (retries included). None when
    # the judge wasn't called, or the provider/model didn't report it.
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None
    judge_cost_usd: float | None = None


class RunResult(BaseModel):
    """The outcome of a full suite run."""

    suite: str
    target: str
    n_cases: int
    mean_score: float
    results: list[CaseResult]
    # Cases that only have a judge rubric, run without a judge: nothing could
    # score them, so they are left out of ``results`` and the mean.
    skipped: list[str] = []
    run_id: str | None = None  # assigned when the run is saved to the store
