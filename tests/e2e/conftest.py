"""Shared fixtures for the end-to-end tests.

The E2E tests drive evalx the way a user does -- a real ``evalx`` process,
real HTTP over a socket, real LiteLLM, a real DuckDB file -- with one
substitution: the LLM servers are a local fake that speaks the OpenAI
chat-completions protocol. That keeps the tests deterministic, offline and
free while exercising every layer evalx owns.
"""

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from evalx.judge import JUDGE_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FakeOpenAI:
    """A scripted OpenAI-compatible server, playing both target and judge.

    Requests whose system message is evalx's judge prompt are judge calls;
    everything else is a call to the app under test.
    """

    # question -> answer text the "app" gives
    answers: dict[str, str] = field(default_factory=dict)
    # question -> (HTTP status, body) to fail the app's call with
    failures: dict[str, tuple[int, dict]] = field(default_factory=dict)
    # judge user prompt -> raw message content the judge replies with
    judge: Callable[[str], str] = lambda prompt: json.dumps({"score": 1.0, "reason": "ok"})
    requests: list[dict] = field(default_factory=list)
    base_url: str = ""

    def target_calls(self) -> list[dict]:
        return [r for r in self.requests if not _is_judge(r)]

    def judge_calls(self) -> list[dict]:
        return [r for r in self.requests if _is_judge(r)]


def _is_judge(body: dict) -> bool:
    messages = body.get("messages", [])
    return bool(messages) and messages[0].get("content") == JUDGE_SYSTEM_PROMPT


def _completion(model: str, content: str) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _make_handler(fake: FakeOpenAI):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, status: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            body["_path"] = self.path
            fake.requests.append(body)
            if not self.path.endswith("/chat/completions"):
                self._send(404, {"error": {"message": f"no route {self.path}"}})
                return
            model = body.get("model", "")
            if _is_judge(body):
                self._send(200, _completion(model, fake.judge(body["messages"][-1]["content"])))
                return
            question = body["messages"][-1]["content"]
            if question in fake.failures:
                status, err = fake.failures[question]
                self._send(status, err)
                return
            self._send(200, _completion(model, fake.answers.get(question, "I don't know.")))

    return Handler


@pytest.fixture
def fake_openai():
    fake = FakeOpenAI()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(fake))
    fake.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def e2e_env(tmp_path, fake_openai):
    """Environment for an evalx subprocess wired to the fake server."""
    env = dict(os.environ)
    no_proxy = ",".join(filter(None, [env.get("NO_PROXY", ""), "127.0.0.1", "localhost"]))
    env.update(
        {
            "EVALX_STORE": str(tmp_path / "runs.duckdb"),
            # the app under test: the generic OpenAI-compatible adapter
            "OPENAI_COMPAT_BASE_URL": fake_openai.base_url,
            "OPENAI_COMPAT_MODEL": "fake-target",
            # the judge: LiteLLM's openai provider, pointed at the same server
            "OPENAI_API_BASE": fake_openai.base_url,
            "OPENAI_BASE_URL": fake_openai.base_url,
            "OPENAI_API_KEY": "sk-fake",
            # never fetch LiteLLM's model price map from the internet
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            # the fake server is local: keep any configured proxy out of it
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
            # wide terminal so Rich doesn't wrap the lines we assert on
            "COLUMNS": "250",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    for key in ("OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_SYSTEM_PROMPT", "OPENAI_COMPAT_TEMPERATURE"):
        env.pop(key, None)
    return env


@pytest.fixture
def evalx_cli(e2e_env):
    """Run the real evalx CLI in a subprocess; returns CompletedProcess."""

    def run(*args: str, timeout: float = 180) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "evalx.cli", *args],
            cwd=REPO_ROOT,
            env=e2e_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return run
