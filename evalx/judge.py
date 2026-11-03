"""LLM-as-judge scoring.

Bias guards (built in, not optional):
1. Each answer is scored ALONE against a rubric -- never A-vs-B, so there is
   no position bias.
2. The system prompt states that brevity is fine and fluff scores nothing,
   guarding against verbosity bias.
3. Temperature is omitted by default: some models (e.g. claude-haiku-5-5 via
   LiteLLM) reject any temperature but their own default, so we don't send
   the parameter unless explicitly requested.
4. The verdict is parsed as JSON and validated with Pydantic; malformed
   output is retried once, then raises instead of silently inventing a score.

Usage: every judge call's tokens (and cost, when LiteLLM knows the model's
price) are summed across attempts -- retries are paid for too -- and carried
on the verdict, or on the JudgeError when no verdict came back.
"""

import json

import litellm
from pydantic import BaseModel, Field

from evalx.schemas import TestCase


class JudgeUsage(BaseModel):
    """What the judge calls for one case consumed. None means not reported."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None  # None when LiteLLM has no price for the model

    def __add__(self, other: "JudgeUsage") -> "JudgeUsage":
        def add(a, b):
            return None if a is None or b is None else a + b

        return JudgeUsage(
            prompt_tokens=add(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=add(self.completion_tokens, other.completion_tokens),
            cost_usd=add(self.cost_usd, other.cost_usd),
        )


_NO_USAGE = JudgeUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    usage: JudgeUsage | None = None  # set by LLMJudge, never read from the model


class JudgeError(Exception):
    """The judge failed to return a usable verdict."""

    def __init__(self, message: str, usage: JudgeUsage | None = None):
        super().__init__(message)
        self.usage = usage  # what the failed attempts still cost


def _usage_of(response) -> JudgeUsage:
    """Tokens and cost of one LiteLLM response, None where unavailable."""
    usage = getattr(response, "usage", None)
    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001 - unknown model price, local server, ...
        cost = None
    return JudgeUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        cost_usd=cost,
    )


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

    def __init__(self, model: str, temperature: float | None = None, max_retries: int = 1):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def _call(self, prompt: str) -> tuple[str, JudgeUsage]:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or "", _usage_of(response)

    def score(self, case: TestCase, answer: str) -> JudgeVerdict:
        prompt = build_judge_prompt(case, answer)
        last_error: Exception | None = None
        spent = _NO_USAGE
        for _ in range(self.max_retries + 1):
            content, usage = self._call(prompt)
            spent = spent + usage
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(data, dict):
                last_error = ValueError(f"verdict is not a JSON object: {content[:80]!r}")
                continue
            try:
                # Only score and reason come from the model: usage is ours.
                verdict = JudgeVerdict.model_validate(
                    {"score": data.get("score"), "reason": data.get("reason")}
                )
            except ValueError as exc:
                last_error = exc
                continue
            return verdict.model_copy(update={"usage": spent})
        raise JudgeError(
            f"judge returned invalid verdict after retries: {last_error}", usage=spent
        )
