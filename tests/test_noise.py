"""evalx noise: wobble between identical attempts and a suggested bar."""

import pytest
from typer.testing import CliRunner

from evalx import store
from evalx.cli import app
from evalx.noise import NoiseError, analyse, histogram, noise_report, percentile, suggest_bar
from evalx.runner import run_suite

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"
FLAKY = "tests.test_repeats:flaky"


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


@pytest.fixture(autouse=True)
def _reset_flaky():
    from tests.test_repeats import _calls

    _calls.clear()


def _flat(text: str) -> str:
    return " ".join(text.split())


# Shaped like the AeroIntel run: 61 stable cases, 4 with judge-sized wobble,
# 2 that the app answers differently every time.
AEROINTEL_SHAPE = {f"stable-{i}": [1.0, 1.0, 1.0] for i in range(61)}
AEROINTEL_SHAPE.update(
    {
        "safety-wire": [0.90, 0.90, 0.97],
        "titanium": [0.90, 0.85, 0.90],
        "manufacturer": [0.0, 0.08, 0.0],
        "plastics": [0.0, 0.0, 0.07],
        "bolts": [0.17, 0.50, 0.50],
        "fabric": [0.17, 0.17, 0.50],
    }
)


def test_percentile_nearest_rank():
    assert percentile([0.1, 0.2, 0.3, 0.4], 50) == 0.2
    assert percentile([0.1, 0.2, 0.3, 0.4], 100) == 0.4
    assert percentile([0.3], 90) == 0.3


@pytest.mark.parametrize(
    ("wobble", "bar"),
    [(0.0, 0.05), (0.01, 0.05), (0.04, 0.1), (0.07, 0.2), (0.08, 0.2), (0.1, 0.25), (0.15, 0.4),
     (0.5, 1.0)],
)  # fmt: skip
def test_suggest_bar(wobble, bar):
    assert suggest_bar(wobble) == bar


def test_histogram_buckets():
    assert histogram([0.0, 0.0, 0.05, 0.1, 0.15, 0.25, 0.33]) == [
        ("0.00", 2),
        ("0.01–0.10", 2),
        ("0.11–0.20", 1),
        ("0.21–0.30", 1),
        ("> 0.30", 1),
    ]


def test_aerointel_shape_gives_the_hand_picked_bar():
    rep = analyse(AEROINTEL_SHAPE, "r")
    assert (rep.n_cases, rep.n_moved, rep.attempts, rep.n_set_aside) == (67, 6, 3, 3)
    assert rep.normal_wobble == pytest.approx(0.07)
    assert rep.suggested_bar == 0.2
    assert [c.case_id for c in rep.noisy] == ["bolts", "fabric"]
    assert rep.noisy[0].attempts == [0.17, 0.50, 0.50]


def test_aerointel_run_where_only_the_outliers_moved():
    # Run 20261103-222237: 65 cases never moved, 2 moved 0.33. The outliers
    # must not define "normal": a low bar, with both flagged.
    shape = {f"stable-{i}": [1.0, 1.0, 1.0] for i in range(65)}
    shape.update({"bolts": [0.17, 0.50, 0.50], "fabric": [0.17, 0.17, 0.50]})
    rep = analyse(shape, "r")
    assert (rep.n_moved, rep.n_set_aside, rep.normal_wobble) == (2, 3, 0.0)
    assert rep.suggested_bar == 0.05
    assert [c.case_id for c in rep.noisy] == ["bolts", "fabric"]


def test_judge_wobble_stays_inside_the_bar():
    rep = analyse(AEROINTEL_SHAPE, "r")
    assert rep.normal_wobble == pytest.approx(0.07)
    assert not {"safety-wire", "titanium", "manufacturer", "plastics"} & {
        c.case_id for c in rep.noisy
    }


def test_small_suite_flags_its_one_outlier():
    shape = {"odd": [0.0, 0.33]} | {f"c{i}": [1.0, 1.0] for i in range(5)}
    rep = analyse(shape)
    assert rep.n_set_aside == 1
    assert [c.case_id for c in rep.noisy] == ["odd"]


def test_single_case_suite_keeps_its_case():
    rep = analyse({"only": [0.0, 0.2]})
    assert rep.n_set_aside == 0
    assert rep.normal_wobble == pytest.approx(0.2)


def test_deterministic_app_gets_the_floor():
    rep = analyse({"a": [1.0, 1.0, 1.0], "b": [0.5, 0.5, 0.5]})
    assert (rep.n_moved, rep.normal_wobble, rep.suggested_bar, rep.noisy) == (0, 0.0, 0.05, [])


def test_uniformly_noisy_app_gets_a_high_bar_and_no_outliers():
    rep = analyse({f"c{i}": [0.2, 0.5, 0.4] for i in range(10)})
    assert rep.suggested_bar == 0.75
    assert rep.noisy == []


def test_two_attempts_work():
    rep = analyse({"a": [1.0, 0.9], "b": [1.0, 1.0]})
    assert rep.attempts == 2
    assert rep.n_moved == 1


def test_one_attempt_is_refused():
    with pytest.raises(NoiseError, match="--repeats 3"):
        analyse({"a": [1.0], "b": [0.5]}, "r1")


@pytest.mark.parametrize(
    ("kwargs", "msg"), [({"trim": 0.5}, "trim"), ({"trim": -0.1}, "trim"), ({"multiplier": 0}, "multiplier")]
)
def test_bad_parameters(kwargs, msg):
    with pytest.raises(NoiseError, match=msg):
        analyse(AEROINTEL_SHAPE, **kwargs)


# --- from the store and the CLI ------------------------------------------------


def test_noise_report_reads_a_stored_repeats_run(isolated_store):
    run_id = store.save_run(run_suite(DEMO, FLAKY, repeats=3))
    rep = noise_report("latest")
    assert rep.run_id == run_id
    assert rep.attempts == 3
    # flaky alternates real answer / non-answer: q001 goes 1, 0, 1.
    assert rep.spreads["q001"] == 1.0
    # Every answered case wobbles, so nothing stands out as an outlier:
    # the honest result is a high bar, not a noisy list.
    assert rep.suggested_bar == 1.0
    assert rep.noisy == []


def test_cli_noise(isolated_store):
    store.save_run(run_suite(DEMO, FLAKY, repeats=3))
    res = CliRunner().invoke(app, ["noise"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "6 cases × 3 attempts" in out
    assert "Suggested bar:" in out
    assert "--fail-on-regression" in out
    assert "note: that bar is high" in out


def test_cli_noise_lists_outliers(isolated_store, monkeypatch):
    # Store a run shaped like AeroIntel by hand: two outliers among stable cases.
    from evalx.schemas import CaseResult, RunResult

    results = [
        CaseResult(case_id=cid, question="?", answer="a", check_scores=[], latency_ms=1.0,
                   score=sum(a) / len(a), attempt_scores=a)
        for cid, a in AEROINTEL_SHAPE.items()
    ]  # fmt: skip
    store.save_run(
        RunResult(suite="s", target="t", n_cases=len(results), mean_score=0.9,
                  results=results, repeats=3)
    )  # fmt: skip
    res = CliRunner().invoke(app, ["noise"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "6 of 67 cases moved at all" in out
    assert "Suggested bar: 0.20 → --fail-on-regression 0.2" in out
    assert "bolts 0.33 attempts 0.17, 0.50, 0.50" in out
    assert "fabric 0.33" in out


def test_cli_noise_stable_run(isolated_store):
    store.save_run(run_suite(DEMO, TARGET, repeats=3))
    res = CliRunner().invoke(app, ["noise", "latest"], env={"COLUMNS": "200"})
    assert res.exit_code == 0
    out = _flat(res.output)
    assert "0 of 6 cases moved at all" in out
    assert "Suggested bar: 0.05" in out
    assert "No case moves more than the bar." in out


def test_cli_noise_two_attempts_warns(isolated_store):
    store.save_run(run_suite(DEMO, FLAKY, repeats=2))
    res = CliRunner().invoke(app, ["noise"], env={"COLUMNS": "200"})
    assert res.exit_code == 0
    assert "only 2 attempts per case" in _flat(res.output)


def test_cli_noise_single_attempt_run_is_a_clear_error(isolated_store):
    store.save_run(run_suite(DEMO, TARGET))
    res = CliRunner().invoke(app, ["noise"], env={"COLUMNS": "200"})
    assert res.exit_code == 1
    assert "run it with --repeats 3" in _flat(res.output)


def test_cli_noise_no_runs(isolated_store):
    res = CliRunner().invoke(app, ["noise"])
    assert res.exit_code == 1
    assert "no runs stored yet" in res.output
