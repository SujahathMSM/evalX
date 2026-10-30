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
import os
import urllib.error
import urllib.request

_DEFAULT_URL = "http://127.0.0.1:8000"
_TIMEOUT_S = 120.0  # a grounded answer takes a while (retrieval + LLM)

_token: str | None = None


class AeroIntelError(RuntimeError):
    """The AeroIntel service could not be reached or answered with an error."""


def _base_url() -> str:
    return os.environ.get("AEROINTEL_URL", _DEFAULT_URL).rstrip("/")


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    """POST JSON to the service and return the parsed JSON body."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        _base_url() + path,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
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


def answer(question: str) -> str:
    """Ask AeroIntel one question; return its answer (or refusal) as text."""
    token = _token or (_fetch_token() if _credentials_configured() else None)
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
