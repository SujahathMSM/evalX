"""Tests for the AeroIntel adapter: HTTP is faked, no live service needed."""

import email.message
import io
import json
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


# --- HTTP 429: retry instead of failing the case -----------------------------


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _http_429(request, retry_after: str | None = None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        request.full_url, 429, "Too Many Requests", headers, io.BytesIO(b"rate limited")
    )


@pytest.fixture
def sleeps(monkeypatch):
    """Record time.sleep calls instead of sleeping."""
    calls: list[float] = []
    monkeypatch.setattr(aerointel_app.time, "sleep", calls.append)
    return calls


def _scripted_urlopen(monkeypatch, script: list):
    """urlopen that walks ``script``: a dict is a 200 body, a str|None a 429."""
    seen = []

    def fake(request, timeout=None):
        seen.append(request)
        step = script.pop(0)
        if isinstance(step, dict):
            return _FakeResponse(json.dumps(step).encode())
        raise _http_429(request, retry_after=step)

    monkeypatch.setattr(aerointel_app.urllib.request, "urlopen", fake)
    return seen


def test_post_retries_429_then_succeeds(monkeypatch, sleeps):
    seen = _scripted_urlopen(monkeypatch, [None, {"refused": False, "answer": "ok"}])
    assert aerointel_app._post("/v1/query", {"question": "q"}) == {
        "refused": False,
        "answer": "ok",
    }
    assert len(seen) == 2
    assert sleeps == [2.0]  # no Retry-After: one token refill at 0.5/s


def test_post_honours_retry_after(monkeypatch, sleeps):
    _scripted_urlopen(monkeypatch, ["5", "0.5", "120", {"answer": "ok"}])
    assert aerointel_app._post("/v1/query", {"question": "q"}) == {"answer": "ok"}
    assert sleeps == [5.0, 0.5, 60.0]  # 120s is capped at 60s


def test_post_unparseable_retry_after_falls_back(monkeypatch, sleeps):
    _scripted_urlopen(
        monkeypatch, ["Wed, 21 Oct 2026 07:28:00 GMT", "-3", "nan", {"answer": "ok"}]
    )
    assert aerointel_app._post("/v1/query", {"question": "q"}) == {"answer": "ok"}
    assert sleeps == [2.0, 2.0, 2.0]


def test_post_persistent_429_raises_after_budget(monkeypatch, sleeps):
    budget = aerointel_app._MAX_429_RETRIES
    assert budget == 30
    seen = _scripted_urlopen(monkeypatch, [None] * (budget + 1))
    with pytest.raises(AeroIntelError, match="HTTP 429: rate limited"):
        aerointel_app._post("/v1/query", {"question": "q"})
    assert len(seen) == budget + 1  # the first try plus 30 retries
    assert len(sleeps) == budget


def test_post_other_http_errors_not_retried(monkeypatch, sleeps):
    calls = []

    def raise_500(request, timeout=None):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 500, "err", {}, io.BytesIO(b"boom"))

    monkeypatch.setattr(aerointel_app.urllib.request, "urlopen", raise_500)
    with pytest.raises(AeroIntelError, match="HTTP 500: boom"):
        aerointel_app._post("/v1/query", {"question": "q"})
    assert len(calls) == 1
    assert sleeps == []


def test_answer_401_refresh_still_works_with_real_post(monkeypatch, sleeps):
    """A stale token: 401 on /v1/query, refresh at /auth/token, retry once."""
    monkeypatch.setenv("AEROINTEL_CLIENT_ID", "admin")
    monkeypatch.setenv("AEROINTEL_CLIENT_SECRET", "s3cret")
    monkeypatch.setattr(aerointel_app, "_token", "stale")
    monkeypatch.delenv("AEROINTEL_URL", raising=False)
    auth_headers = []

    def fake(request, timeout=None):
        path = request.full_url.split("8000", 1)[1]
        if path == "/auth/token":
            return _FakeResponse(json.dumps({"access_token": "fresh"}).encode())
        auth_headers.append(request.get_header("Authorization"))
        if request.get_header("Authorization") == "Bearer stale":
            raise urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b"expired"))
        return _FakeResponse(json.dumps({"refused": False, "answer": "ok"}).encode())

    monkeypatch.setattr(aerointel_app.urllib.request, "urlopen", fake)
    assert answer("q") == "ok"
    assert auth_headers == ["Bearer stale", "Bearer fresh"]
    assert sleeps == []


def test_concurrent_answers_fetch_the_token_once(monkeypatch):
    """evalx --workers calls answer() from many threads at once."""
    import threading
    import time

    monkeypatch.setenv("AEROINTEL_CLIENT_ID", "admin")
    monkeypatch.setenv("AEROINTEL_CLIENT_SECRET", "s3cret")
    calls = []
    lock = threading.Lock()

    def fake(path, payload, token=None):
        with lock:
            calls.append(path)
        if path == "/auth/token":
            time.sleep(0.05)  # widen the race window
            return {"access_token": "tok-1"}
        return {"refused": False, "answer": "a"}

    monkeypatch.setattr(aerointel_app, "_post", fake)
    threads = [threading.Thread(target=answer, args=(f"q{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls.count("/auth/token") == 1
    assert calls.count("/v1/query") == 8
