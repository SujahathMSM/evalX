"""evalx target for any app that speaks the OpenAI chat-completions API.

A large family of apps and servers exposes the same endpoint shape --
``POST {base_url}/chat/completions`` with a ``messages`` list, answer at
``choices[0].message.content``: Ollama (http://localhost:11434/v1), Open
WebUI, LM Studio, vLLM, text-generation-webui, LiteLLM proxies, and the
hosted providers (OpenAI, OpenRouter, Groq, Together). This one adapter
plugs evalx into all of them; nothing else about the harness changes::

    export OPENAI_COMPAT_MODEL=qwen3            # whatever the server serves
    evalx run --suite datasets/general/v1 \
        --target examples.openai_compatible_app:answer

Configuration is by environment variable:

- ``OPENAI_COMPAT_BASE_URL`` -- API root (default
  http://localhost:11434/v1, Ollama's local endpoint). Point it at
  https://api.openai.com/v1 for OpenAI itself.
- ``OPENAI_COMPAT_MODEL`` -- the model name the server serves
  (e.g. ``qwen3``, ``llama3.1``, ``gpt-4o-mini``). No default: the right
  name depends entirely on the target, so the adapter fails loudly when
  it is not set instead of guessing one.
- ``OPENAI_COMPAT_API_KEY`` -- sent as ``Authorization: Bearer ...``
  when set. Local servers usually ignore it; hosted providers need it.
- ``OPENAI_COMPAT_SYSTEM_PROMPT`` -- optional system message sent before
  every question, so the same model can be graded under different system
  prompts (that is a prompt change like any other: re-run and compare).
- ``OPENAI_COMPAT_TEMPERATURE`` -- optional; omitted from the request
  unless set. Some models reject a forced temperature (the judge's
  claude-haiku-5-5 lesson), so the adapter never invents one.

Only the standard library is used (urllib): an optional example target
should not add a dependency to evalx.
"""

import json
import os
import urllib.error
import urllib.request

_DEFAULT_URL = "http://localhost:11434/v1"  # Ollama's local endpoint
_TIMEOUT_S = 120.0  # a local model on a busy machine can be slow


class OpenAICompatError(RuntimeError):
    """The target server could not be reached or answered with an error."""


def _base_url() -> str:
    return os.environ.get("OPENAI_COMPAT_BASE_URL", _DEFAULT_URL).rstrip("/")


def _server_message(raw: bytes) -> str:
    """Best error text from a response body: the server's own message when
    it sent one (OpenAI-style ``{"error": {"message": ...}}``), else a
    snippet of the raw body."""
    try:
        body = json.loads(raw)
        message = body.get("error", {}).get("message")
        if message:
            return str(message)[:300]
    except (ValueError, AttributeError):
        pass
    return raw.decode(errors="replace")[:300]


def _post(payload: dict) -> dict:
    """POST to {base}/chat/completions and return the parsed JSON body."""
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_COMPAT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _base_url() + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise OpenAICompatError(
            f"target {_base_url()} returned HTTP {exc.code}: {_server_message(exc.read())}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenAICompatError(
            f"cannot reach an OpenAI-compatible server at {_base_url()} ({exc.reason}); "
            "is it running, and is OPENAI_COMPAT_BASE_URL pointing at its API root (…/v1)?"
        ) from exc


def _payload(question: str) -> dict:
    model = os.environ.get("OPENAI_COMPAT_MODEL")
    if not model:
        raise OpenAICompatError(
            "OPENAI_COMPAT_MODEL is not set: name the model your server serves "
            "(e.g. qwen3, llama3.1, gpt-4o-mini) so the adapter knows what to ask for"
        )
    messages = []
    system_prompt = os.environ.get("OPENAI_COMPAT_SYSTEM_PROMPT")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    payload: dict = {"model": model, "messages": messages}
    temperature = os.environ.get("OPENAI_COMPAT_TEMPERATURE")
    if temperature:
        try:
            payload["temperature"] = float(temperature)
        except ValueError:
            raise OpenAICompatError(
                f"OPENAI_COMPAT_TEMPERATURE={temperature!r} is not a number"
            ) from None
    return payload


def answer(question: str) -> str:
    """Ask the configured model one question; return its reply as text."""
    body = _post(_payload(question))
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAICompatError(
            "the server answered, but not in the chat-completions shape "
            f"(no choices[0].message.content): {json.dumps(body)[:300]}"
        ) from exc
    if not isinstance(content, str):
        raise OpenAICompatError(f"the server's message content was not text: {content!r:.300}")
    return content
