"""Tests for the generic OpenAI-compatible adapter: HTTP is faked."""

import io
import json
import urllib.error

import pytest

from examples import openai_compatible_app
from examples.openai_compatible_app import OpenAICompatError, answer

_ENV_VARS = (
    "OPENAI_COMPAT_BASE_URL",
    "OPENAI_COMPAT_MODEL",
    "OPENAI_COMPAT_API_KEY",
    "OPENAI_COMPAT_SYSTEM_PROMPT",
    "OPENAI_COMPAT_TEMPERATURE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "test-model")


def _fake_post(monkeypatch, body, calls: list):
    def fake(payload):
        calls.append(payload)
        return body

    monkeypatch.setattr(openai_compatible_app, "_post", fake)


def test_answer_returns_message_content(monkeypatch):
    calls = []
    _fake_post(
        monkeypatch,
        {"choices": [{"message": {"role": "assistant", "content": "Paris."}}]},
        calls,
    )
    assert answer("Capital of France?") == "Paris."
    assert calls == [
        {"model": "test-model", "messages": [{"role": "user", "content": "Capital of France?"}]}
    ]


def test_answer_requires_a_model(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_MODEL")
    with pytest.raises(OpenAICompatError, match="OPENAI_COMPAT_MODEL is not set"):
        answer("q")


def test_answer_prepends_system_prompt_when_set(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_SYSTEM_PROMPT", "Answer in one word.")
    calls = []
    _fake_post(monkeypatch, {"choices": [{"message": {"content": "Paris"}}]}, calls)
    assert answer("Capital of France?") == "Paris"
    assert calls[0]["messages"] == [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Capital of France?"},
    ]


def test_answer_rejects_malformed_response(monkeypatch):
    _fake_post(monkeypatch, {"unexpected": True}, [])
    with pytest.raises(OpenAICompatError, match="chat-completions shape"):
        answer("q")


class _Response:
    def __init__(self, body: dict):
        self._raw = json.dumps(body).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_urlopen(monkeypatch, seen: dict, body: dict):
    def fake(request, timeout=None):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["payload"] = json.loads(request.data)
        return _Response(body)

    monkeypatch.setattr(openai_compatible_app.urllib.request, "urlopen", fake)


def test_post_hits_chat_completions_with_bearer_key(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://example.test/v1/")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-test")
    seen: dict = {}
    _capture_urlopen(
        monkeypatch, seen, {"choices": [{"message": {"content": "ok"}}]}
    )
    assert answer("q") == "ok"
    assert seen["url"] == "http://example.test/v1/chat/completions"
    assert seen["authorization"] == "Bearer sk-test"
    assert seen["payload"]["model"] == "test-model"


def test_post_omits_authorization_without_a_key(monkeypatch):
    seen: dict = {}
    _capture_urlopen(monkeypatch, seen, {"choices": [{"message": {"content": "ok"}}]})
    assert answer("q") == "ok"
    assert seen["authorization"] is None
    # the default base is Ollama's local endpoint
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"


def test_post_http_error_uses_the_servers_message(monkeypatch):
    def raise_http(request, timeout=None):
        body = json.dumps({"error": {"message": "model 'test-model' not found"}}).encode()
        raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, io.BytesIO(body))

    monkeypatch.setattr(openai_compatible_app.urllib.request, "urlopen", raise_http)
    with pytest.raises(OpenAICompatError, match="HTTP 404.*not found"):
        openai_compatible_app._post({"model": "test-model", "messages": []})


def test_post_connection_error_explains_next_step(monkeypatch):
    def raise_url(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(openai_compatible_app.urllib.request, "urlopen", raise_url)
    with pytest.raises(OpenAICompatError, match="cannot reach"):
        openai_compatible_app._post({"model": "test-model", "messages": []})
