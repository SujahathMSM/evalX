"""evalx target for the AeroIntel RAG service (the dogfood target).

AeroIntel is a separate project (github.com/SujahathMSM/aerointel): a FastAPI
RAG over the FAA AC 43.13-1B maintenance manual, with structural refusals.
This adapter is the bridge evalx needs -- one question in, one answer out::

    evalx run --suite datasets/aerointel/v1 --target examples.aerointel_app:answer

Configuration is by environment variable:

- ``AEROINTEL_URL`` -- service base URL (default http://127.0.0.1:8000)
- ``AEROINTEL_CLIENT_ID`` / ``AEROINTEL_CLIENT_SECRET`` -- when both are
  set, the adapter trades them for a bearer token at POST /auth/token
  (scope "query") and caches it for the rest of the run. Leave both unset
  when the service runs with AUTH_ENABLED=false.

Only the standard library is used (urllib): an optional example target
should not add a dependency to evalx.

A refusal is a first-class answer here. When AeroIntel refuses, the adapter
returns the service's own ``refusal_message`` verbatim, so the harness
grades the refusal the app actually gave -- not a paraphrase invented by
this bridge.
"""

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request

_DEFAULT_URL = "http://127.0.0.1:8000"
_TIMEOUT_S = 120.0  # a grounded answer takes a while (retrieval + LLM)

# HTTP 429 handling: AeroIntel rate-limits with a token bucket (capacity 20,
# refill 0.5 tokens/s), so a long eval run drains it. Rather than disabling
# the limiter, wait and retry: honour Retry-After (capped), else wait 2.0s --
# one token refill at 0.5 tokens/s.
_MAX_429_RETRIES = 30
_RETRY_AFTER_CAP_S = 60.0
_RETRY_FALLBACK_S = 2.0

_token: str | None = None
# evalx --workers calls answer() from several threads: one token fetch, not N.
_token_lock = threading.Lock()


class AeroIntelError(RuntimeError):
    """The AeroIntel service could not be reached or answered with an error."""


def _base_url() -> str:
    return os.environ.get("AEROINTEL_URL", _DEFAULT_URL).rstrip("/")


def _retry_delay(retry_after: str | None) -> float:
    """Seconds to wait before retrying a 429, from its Retry-After header."""
    try:
        delay = float(retry_after) if retry_after is not None else None
    except ValueError:
        delay = None  # e.g. an HTTP-date: fall back rather than parse it
    if delay is None or not math.isfinite(delay) or delay < 0:
        return _RETRY_FALLBACK_S
    return min(delay, _RETRY_AFTER_CAP_S)


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    """POST JSON to the service and return the parsed JSON body.

    HTTP 429 is retried up to ``_MAX_429_RETRIES`` times (see _retry_delay);
    every other error, and a 429 that outlasts the budget, raises
    AeroIntelError.
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        _base_url() + path,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    retries = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and retries < _MAX_429_RETRIES:
                retries += 1
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                time.sleep(_retry_delay(retry_after))
                continue
            detail = exc.read().decode(errors="replace")[:300]
            raise AeroIntelError(f"AeroIntel {path} returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AeroIntelError(
                f"cannot reach AeroIntel at {_base_url()} ({exc.reason}); "
                "is the service running? (docker compose up, in the aerointel repo)"
            ) from exc


def _credentials_configured() -> bool:
    return bool(os.environ.get("AEROINTEL_CLIENT_ID") and os.environ.get("AEROINTEL_CLIENT_SECRET"))


def _fetch_token() -> str:
    """Trade the configured client credentials for a bearer token."""
    global _token
    body = _post(
        "/auth/token",
        {
            "client_id": os.environ["AEROINTEL_CLIENT_ID"],
            "client_secret": os.environ["AEROINTEL_CLIENT_SECRET"],
            "scope": "query",
        },
    )
    _token = body["access_token"]
    return _token


def _shared_token() -> str:
    """The cached token, fetching it once even when threads race to it."""
    with _token_lock:
        return _token or _fetch_token()


def answer(question: str) -> str:
    """Ask AeroIntel one question; return its answer (or refusal) as text."""
    token = _token or (_shared_token() if _credentials_configured() else None)
    try:
        body = _post("/v1/query", {"question": question}, token=token)
    except AeroIntelError as exc:
        # A 401 despite a token usually means the cached token went stale
        # (the service restarted and forgot it): refresh once, retry once.
        if token is None or "HTTP 401" not in str(exc):
            raise
        body = _post("/v1/query", {"question": question}, token=_fetch_token())
    if body.get("refused"):
        return body.get("refusal_message") or "Refused (the service gave no refusal message)."
    return body.get("answer") or ""
