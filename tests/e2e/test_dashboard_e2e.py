"""End-to-end: the real Streamlit dashboard, rendered in a headless browser.

Starts ``evalx dashboard`` as a user would, over a seeded run store, and
checks what a person actually sees. Needs the ``e2e`` extra (Playwright)
and a Chromium build: ``pip install -e ".[dev,e2e]" && playwright install
chromium``. Skipped when either is missing.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from evalx import store
from evalx.schemas import CaseResult, RunResult

from .conftest import REPO_ROOT

pytestmark = pytest.mark.e2e

sync_api = pytest.importorskip("playwright.sync_api")

OUTAGE_ERROR = "target: OpenAICompatError: target returned HTTP 429: rate limit exceeded"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _result(suite: str, errors: dict[str, str]) -> RunResult:
    results = [
        CaseResult(
            case_id=case_id,
            question=f"question {case_id}?",
            answer="" if case_id in errors else f"answer {case_id}",
            check_scores=[],
            score=0.0 if case_id in errors else 1.0,
            latency_ms=5.0,
            error=errors.get(case_id),
        )
        for case_id in ("a1", "a2", "a3")
    ]
    return RunResult(
        suite=suite,
        target="examples.openai_compatible_app:answer",
        n_cases=3,
        mean_score=sum(r.score for r in results) / 3,
        results=results,
        skipped=["a4"],
    )


@pytest.fixture
def dashboard_url(tmp_path):
    db = tmp_path / "runs.duckdb"
    store.save_run(_result("datasets/clean/v1", {}), path=db)
    time.sleep(1.1)  # run ids are second-resolution timestamps; keep order clear
    store.save_run(_result("datasets/outage/v1", {"a1": OUTAGE_ERROR}), path=db)

    port = _free_port()
    env = dict(os.environ, EVALX_STORE=str(db), PYTHONPATH=str(REPO_ROOT))
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    no_proxy = ",".join(filter(None, [env.get("NO_PROXY", ""), "127.0.0.1", "localhost"]))
    env.update(NO_PROXY=no_proxy, no_proxy=no_proxy)
    proc = subprocess.Popen(
        [sys.executable, "-m", "evalx.cli", "dashboard", "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # so the streamlit child dies with it
    )
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 60
    try:
        while True:
            try:
                if opener.open(f"{url}/_stcore/health", timeout=2).status == 200:
                    break
            except OSError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                pytest.fail(f"dashboard did not start:\n{out}")
            time.sleep(0.5)
        yield url
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


@pytest.fixture
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - no browser build installed
            pytest.skip(f"chromium not available for Playwright: {exc}")
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        yield page
        browser.close()


def _assert_no_streamlit_exception(page) -> None:
    assert page.locator('[data-testid="stException"]').count() == 0


def test_dashboard_renders_run_history(dashboard_url, page):
    page.goto(dashboard_url)
    expect = sync_api.expect
    expect(page.get_by_text("evalX — run history & compare")).to_be_visible(timeout=30_000)
    sidebar = page.locator('[data-testid="stSidebar"]')
    expect(sidebar.get_by_text("Stored runs")).to_be_visible()
    expect(sidebar.locator('[data-testid="stMetricValue"]').first).to_have_text("2")
    # Latest mean is the outage run: one of three cases errored -> 0.667.
    expect(sidebar.get_by_text("0.667")).to_be_visible()
    expect(page.get_by_role("tab", name="Runs")).to_be_visible()
    expect(page.locator('[data-testid="stDataFrame"]').first).to_be_visible()
    _assert_no_streamlit_exception(page)


def test_dashboard_drill_down_shows_case_error(dashboard_url, page):
    page.goto(dashboard_url)
    expect = sync_api.expect
    expect(page.get_by_text("evalX — run history & compare")).to_be_visible(timeout=30_000)
    page.get_by_role("tab", name="Run detail").click()
    # Defaults: the newest run (the outage) and its first case (a1, errored).
    expect(page.get_by_text("Case drill-down")).to_be_visible()
    error_box = page.locator('[data-testid="stAlert"]').filter(has_text="HTTP 429")
    expect(error_box).to_be_visible()
    expect(error_box).to_contain_text(OUTAGE_ERROR)
    expect(page.get_by_text("question a1?")).to_be_visible()
    _assert_no_streamlit_exception(page)


def test_dashboard_compare_tab_renders(dashboard_url, page):
    page.goto(dashboard_url)
    expect = sync_api.expect
    expect(page.get_by_text("evalX — run history & compare")).to_be_visible(timeout=30_000)
    page.get_by_role("tab", name="Compare").click()
    expect(page.locator('[data-testid="stDataFrame"]').last).to_be_visible()
    _assert_no_streamlit_exception(page)
