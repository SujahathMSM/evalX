"""Tests for the AeroIntel adapter: HTTP is faked, no live service needed."""

import io
import urllib.error

import pytest

from examples import aerointel_app
from examples.aerointel_app import AeroIntelError, answer


@pytest.fixture(autouse=True)
def _reset_token(monkeypatch):
    monkeypatch.setattr(aerointel_app, "_token", None)
    monkeypatch.delenv("AEROINTEL_CLIENT_ID", raising=False)
    monkeypatch.delenv("AEROINTEL_CLIENT_SECRET", raising=False)


def _fake_post(monkeypatch, bodies: dict, calls: list):
    """Replace _post: record (path, payload, token), return bodies[path]."""

    def fake(path, payload, token=None):
        calls.append((path, payload, token))
        return bodies[path]

    monkeypatch.setattr(aerointel_app, "_post", fake)


def test_answer_returns_answer_text(monkeypatch):
    calls = []
    _fake_post(monkeypatch, {"/v1/query": {"refused": False, "answer": "Torque to 40 in-lb."}}, calls)
    assert answer("What torque?") == "Torque to 40 in-lb."
    assert calls == [("/v1/query", {"question": "What torque?"}, None)]


def test_answer_returns_refusal_message_verbatim(monkeypatch):
    refusal = "No chunk in the database is close enough to this question. Refusing to answer."
    _fake_post(monkeypatch, {"/v1/query": {"refused": True, "refusal_message": refusal}}, [])
    assert answer("What is the capital of France?") == refusal


def test_answer_fetches_and_caches_token(monkeypatch):
    monkeypatch.setenv("AEROINTEL_CLIENT_ID", "admin")
    monkeypatch.setenv("AEROINTEL_CLIENT_SECRET", "s3cret")
    calls = []
    _fake_post(
        monkeypatch,
        {
            "/auth/token": {"access_token": "tok-1"},
            "/v1/query": {"refused": False, "answer": "a"},
        },
        calls,
    )
    assert answer("q1") == "a"
    assert answer("q2") == "a"
    token_calls = [c for c in calls if c[0] == "/auth/token"]
    assert len(token_calls) == 1  # fetched once, then cached
    assert token_calls[0][1]["scope"] == "query"
    query_calls = [c for c in calls if c[0] == "/v1/query"]
    assert all(c[2] == "tok-1" for c in query_calls)


def test_answer_propagates_service_errors(monkeypatch):
    def boom(path, payload, token=None):
        raise AeroIntelError("AeroIntel /v1/query returned HTTP 500: boom")

    monkeypatch.setattr(aerointel_app, "_post", boom)
    with pytest.raises(AeroIntelError, match="HTTP 500"):
        answer("q")


def test_post_http_error_becomes_aero_intel_error(monkeypatch):
    def raise_http(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "unavailable", {}, io.BytesIO(b"db down")
        )

    monkeypatch.setattr(aerointel_app.urllib.request, "urlopen", raise_http)
    with pytest.raises(AeroIntelError, match="HTTP 503.*db down"):
        aerointel_app._post("/v1/query", {"question": "q"})


def test_post_connection_error_explains_next_step(monkeypatch):
    def raise_url(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(aerointel_app.urllib.request, "urlopen", raise_url)
    with pytest.raises(AeroIntelError, match="cannot reach AeroIntel"):
        aerointel_app._post("/v1/query", {"question": "q"})
