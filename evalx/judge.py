"""LLM-as-judge scoring.

Bias guards (built in, not optional):
1. Each answer is scored ALONE against a rubric -- never A-vs-B, so there is
   no position bias.
2. The system prompt states that brevity is fine and fluff scores nothing,
   guarding against verbosity bias.
3. temperature=0 for determinism.
4. The verdict is parsed as JSON and validated with Pydantic; malformed
   output is retried once, then raises instead of silently inventing a score.
"""

import json

import litellm
from pydantic import BaseModel, Field

from evalx.schemas import TestCase


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class JudgeError(Exception):
    """The judge failed to return a usable verdict."""


JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of AI answers. "
    "Score the answer against the rubric on a 0.0-1.0 scale. "
    "Brevity is fine: a short correct answer scores 1.0, and padded fluff "
    "scores nothing extra. Judge only what is written, never guess at intent. "
    'Reply with JSON only: {"score": <0.0-1.0>, "reason": "<one sentence>"}.'
)


def build_judge_prompt(case: TestCase, answer: str) -> str:
    """Render the scoring prompt for one case. Raises if the case has no rubric."""
    if case.judge is None:
        raise JudgeError("case has no judge rubric")
    return (
        f"Rubric:\n{case.judge.rubric}\n\nQuestion:\n{case.question}\n\nAnswer to score:\n{answer}"
    )


class LLMJudge:
    """Scores answers with an LLM.

    Model names follow LiteLLM conventions, e.g.
    'anthropic/claude-haiku-5-5', 'openai/gpt-4o-mini'. The provider's API key
    must be in the environment (e.g. ANTHROPIC_API_KEY).
    """

    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 1):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def _call(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def score(self, case: TestCase, answer: str) -> JudgeVerdict:
        prompt = build_judge_prompt(case, answer)
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                return JudgeVerdict.model_validate(json.loads(self._call(prompt)))
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise JudgeError(f"judge returned invalid verdict after retries: {last_error}")
