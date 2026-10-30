"""Tests for the dashboard data helpers and store.load_case_details.

The Streamlit UI itself (dashboard.main) is exercised with a stubbed
``streamlit`` module: it verifies the render code runs end to end without
needing a browser or the real Streamlit package.
"""

import sys
import time

import pandas as pd
import pytest

from evalx import dashboard, store
from evalx.schemas import CaseResult, RunResult


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _fake_result(scores: list[float], prefix: str = "q") -> RunResult:
    return RunResult(
        suite="s",
        target="t",
        n_cases=len(scores),
        mean_score=sum(scores) / len(scores),
        results=[
            CaseResult(
                case_id=f"{prefix}{i:03d}",
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


def test_comparison_frame_no_shared_cases(isolated_store):
    # Two runs from different suites (demo ids vs aerointel-style ids)
    # share no case ids: the frame must be empty but still carry its
    # columns -- without them delta_chart died with KeyError: 'delta'.
    store.save_run(_fake_result([1.0, 0.5]))
    store.save_run(_fake_result([0.0, 1.0], prefix="a"))
    df, summary = dashboard.comparison_frame("latest-1", "latest")
    assert df.empty
    assert list(df.columns) == ["case_id", "score_a", "score_b", "delta", "trend"]
    assert (summary["improved"], summary["regressed"], summary["unchanged"]) == (0, 0, 0)


def test_delta_chart_empty_comparison(isolated_store):
    store.save_run(_fake_result([1.0]))
    store.save_run(_fake_result([0.0], prefix="a"))
    df, _summary = dashboard.comparison_frame("latest-1", "latest")
    assert dashboard.delta_chart(df) is not None


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
        if name == "sidebar":
            self.calls.append(name)
            return _FakeCtx(_FakeSt())

        def _rec(*args, **kwargs):
            self.calls.append(name)
            if name == "tabs":
                return [_FakeCtx(_FakeSt()) for _ in args[0]]
            if name == "selectbox":
                # honour the caller's default index, like real streamlit
                return args[1][kwargs.get("index", 0) or 0]
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
    for call in ("set_page_config", "title", "tabs", "dataframe", "altair_chart", "metric"):
        assert call in fake.calls
    assert "bar_chart" not in fake.calls
    assert "line_chart" not in fake.calls


def test_main_empty_store(isolated_store, monkeypatch):
    fake = _install_fake_streamlit(monkeypatch)
    dashboard.main()
    assert "info" in fake.calls
    assert "tabs" not in fake.calls


def test_main_renders_with_disjoint_runs(isolated_store, monkeypatch):
    # The Compare tab defaults to newest vs second-newest; when those are
    # runs from different suites it must show the "share no cases" info
    # instead of crashing in delta_chart (the KeyError: 'delta' bug).
    # The sleep keeps the two run ids in different seconds: same-second
    # ids are prefixes of each other, which resolve_run_id calls ambiguous.
    store.save_run(_fake_result([1.0, 0.5]))
    time.sleep(1.1)
    store.save_run(_fake_result([0.0, 1.0], prefix="a"))
    fake = _install_fake_streamlit(monkeypatch)
    dashboard.main()  # must not raise
    assert "info" in fake.calls


def test_runs_frame_delta_prev(isolated_store):
    store.save_run(_fake_result([0.0]))
    store.save_run(_fake_result([1.0]))
    df = dashboard.runs_frame()
    assert df["delta_prev"].iloc[0] == 1.0
    assert pd.isna(df["delta_prev"].iloc[1])  # pandas stores the None as NaN


def test_score_band():
    assert dashboard.score_band(1.0) == "pass"
    assert dashboard.score_band(0.99) == "pass"
    assert dashboard.score_band(0.5) == "partial"
    assert dashboard.score_band(0.49) == "fail"
    assert dashboard.score_band(0.0) == "fail"


def test_case_detail_helper(isolated_store):
    run_id = store.save_run(_fake_result([0.5]))
    detail = dashboard.case_detail(run_id, "q000")
    assert detail["question"] == "?"
    assert detail["answer"] == "a"
    assert detail["checks"] == []
    assert dashboard.case_detail(run_id, "nope") is None


def test_biggest_movers(isolated_store):
    store.save_run(_fake_result([1.0, 0.0]))
    store.save_run(_fake_result([0.5, 1.0]))
    movers = dashboard.biggest_movers("latest-1", "latest")
    assert movers["improved"].case_id == "q001"
    assert movers["improved"].delta == pytest.approx(1.0)
    assert movers["regressed"].case_id == "q000"
    assert movers["regressed"].delta == pytest.approx(-0.5)


def test_biggest_movers_none_when_identical(isolated_store):
    run_id = store.save_run(_fake_result([1.0, 0.5]))
    movers = dashboard.biggest_movers(run_id, run_id)
    assert movers == {"improved": None, "regressed": None}


def test_charts_build(isolated_store):
    store.save_run(_fake_result([1.0, 0.5]))
    store.save_run(_fake_result([0.5, 1.0]))
    assert dashboard.trend_chart(dashboard.runs_frame()) is not None
    run_id = store.resolve_run_id("latest")
    assert dashboard.score_chart(dashboard.case_scores_frame(run_id)) is not None
    df, _summary = dashboard.comparison_frame("latest-1", "latest")
    assert dashboard.delta_chart(df) is not None
