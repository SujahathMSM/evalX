"""Tests for the dashboard data helpers and store.load_case_details.

The Streamlit UI itself (dashboard.main) is exercised with a stubbed
``streamlit`` module: it verifies the render code runs end to end without
needing a browser or the real Streamlit package.
"""

import sys

import pytest

from evalx import dashboard, store
from evalx.schemas import CaseResult, RunResult


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _fake_result(scores: list[float]) -> RunResult:
    return RunResult(
        suite="s",
        target="t",
        n_cases=len(scores),
        mean_score=sum(scores) / len(scores),
        results=[
            CaseResult(
                case_id=f"q{i:03d}",
                question="?",
                answer="a",
                check_scores=[s],
                judge_score=round(s, 2),
                score=s,
                latency_ms=2.0,
            )
            for i, s in enumerate(scores)
        ],
    )


def test_load_case_details(isolated_store):
    run_id = store.save_run(_fake_result([1.0, 0.5]))
    details = store.load_case_details(run_id)
    assert [(d["case_id"], d["score"]) for d in details] == [("q000", 1.0), ("q001", 0.5)]
    assert details[0]["judge_score"] == 1.0
    assert details[0]["latency_ms"] == 2.0


def test_runs_frame_newest_first(isolated_store):
    first = store.save_run(_fake_result([0.0]))
    second = store.save_run(_fake_result([1.0]))
    df = dashboard.runs_frame()
    assert list(df["run_id"]) == [second, first]
    assert df.loc[df["run_id"] == second, "mean_score"].iloc[0] == 1.0


def test_runs_frame_empty(isolated_store):
    assert dashboard.runs_frame().empty


def test_case_scores_frame_sorted(isolated_store):
    run_id = store.save_run(_fake_result([0.25, 1.0, 0.5]))
    df = dashboard.case_scores_frame(run_id)
    assert list(df["case_id"]) == ["q000", "q001", "q002"]
    assert list(df["score"]) == [0.25, 1.0, 0.5]


def test_comparison_frame_deltas(isolated_store):
    store.save_run(_fake_result([1.0, 0.0]))
    store.save_run(_fake_result([0.5, 1.0]))
    # NB: raw ids can't be used here -- the first same-second id is a prefix
    # of the second, which resolve_run_id treats as ambiguous.
    df, summary = dashboard.comparison_frame("latest-1", "latest")
    assert list(df["delta"]) == [-0.5, 1.0]
    assert list(df["trend"]) == ["▼", "▲"]
    assert (summary["improved"], summary["regressed"], summary["unchanged"]) == (1, 1, 0)
    assert summary["delta"] == pytest.approx(0.25)


def test_comparison_frame_bad_ref(isolated_store):
    store.save_run(_fake_result([1.0]))
    with pytest.raises(store.StoreError):
        dashboard.comparison_frame("latest", "nope")


class _FakeCtx:
    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        return self._fake

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        # real streamlit columns are usable both as context managers
        # (``with col:``) and directly (``col.metric(...)``)
        return getattr(self._fake, name)


class _FakeSt:
    """Minimal stand-in for the streamlit module: records calls, never renders."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _rec(*args, **kwargs):
            self.calls.append(name)
            if name == "tabs":
                return [_FakeCtx(_FakeSt()) for _ in args[0]]
            if name == "selectbox":
                return args[1][0]
            if name == "columns":
                # real streamlit columns double as context managers
                return [_FakeCtx(_FakeSt()) for _ in range(args[0])]
            return _FakeSt()

        return _rec


def _install_fake_streamlit(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


def test_main_renders_with_runs(isolated_store, monkeypatch):
    store.save_run(_fake_result([1.0, 0.5]))
    store.save_run(_fake_result([0.0, 1.0]))
    fake = _install_fake_streamlit(monkeypatch)
    dashboard.main()  # must not raise
    for call in ("set_page_config", "title", "tabs", "dataframe", "bar_chart", "line_chart"):
        assert call in fake.calls


def test_main_empty_store(isolated_store, monkeypatch):
    fake = _install_fake_streamlit(monkeypatch)
    dashboard.main()
    assert "info" in fake.calls
    assert "tabs" not in fake.calls
