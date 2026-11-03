"""Streamlit dashboard: run history, per-run detail, and run-over-run compare.

Launch it with ``evalx dashboard`` (or ``streamlit run evalx/dashboard.py``
from the project root). Only :func:`main` touches Streamlit -- the frame,
chart and styling helpers below are plain pandas/altair and are covered
by tests.
"""

import pandas as pd

from evalx import store

_EPS = 1e-9

# Score bands, matching the CLI table's colours: green >= 0.99 (pass),
# amber >= 0.5 (partial), red below (fail).
_BAND_COLORS = {"pass": "#22c55e", "partial": "#f59e0b", "fail": "#ef4444"}


def score_band(score: float) -> str:
    """Band name for a score: pass / partial / fail."""
    if score >= 0.99:
        return "pass"
    if score >= 0.5:
        return "partial"
    return "fail"


_UNKNOWN = "–"


def runs_frame(path=None) -> pd.DataFrame:
    """All stored runs, newest first, with ``delta_prev``: each run's mean
    minus the previous (older) run's mean (None for the oldest run)."""
    runs = store.list_runs(path)
    for i, run in enumerate(runs):
        older = runs[i + 1] if i + 1 < len(runs) else None
        run["delta_prev"] = run["mean_score"] - older["mean_score"] if older else None
        # NULL counts come from runs saved before the store kept them:
        # unknown, so shown as "–", never as a reassuring 0. Stored as text so
        # the column has one type (Arrow, which Streamlit renders through,
        # rejects int/str mixes).
        for col in ("n_errors", "n_skipped"):
            run[col] = _UNKNOWN if run[col] is None else str(run[col])
    return pd.DataFrame(
        runs,
        columns=[
            "run_id",
            "created_at",
            "suite",
            "target",
            "judge_model",
            "mean_score",
            "delta_prev",
            "n_cases",
            "n_errors",
            "n_skipped",
            "judge_cost_usd",
            "duration_s",
            "workers",
            "repeats",
        ],
    )


def case_scores_frame(run_id: str, path=None) -> pd.DataFrame:
    """Per-case scores for one run, sorted by case id."""
    details = store.load_case_details(run_id, path)
    df = pd.DataFrame(details, columns=["case_id", "score", "judge_score", "latency_ms"])
    return df.sort_values("case_id").reset_index(drop=True)


def case_detail(run_id: str, case_id: str, path=None) -> dict | None:
    """Stored detail for one case of one run (None if not in the run)."""
    for detail in store.load_case_details(run_id, path):
        if detail["case_id"] == case_id:
            return detail
    return None


def comparison_frame(ref_a: str, ref_b: str, path=None) -> tuple[pd.DataFrame, dict]:
    """Case-by-case delta between two runs, plus the summary counts."""
    comp = store.compare_runs(ref_a, ref_b, path)
    df = pd.DataFrame(
        [
            {
                "case_id": c.case_id,
                "score_a": c.score_a,
                "score_b": c.score_b,
                "delta": c.delta,
                "trend": "▲" if c.delta > _EPS else "▼" if c.delta < -_EPS else "–",
                "latency_a": c.latency_a,
                "latency_b": c.latency_b,
                "cost_a": c.cost_a,
                "cost_b": c.cost_b,
            }
            for c in comp.cases
        ],
        # columns= matters: two runs from different suites share no case
        # ids, and an empty frame built without it has no columns at all --
        # delta_chart then died with KeyError: 'delta'.
        columns=[
            "case_id",
            "score_a",
            "score_b",
            "delta",
            "trend",
            "latency_a",
            "latency_b",
            "cost_a",
            "cost_b",
        ],
    )
    summary = {
        "run_a": comp.run_a,
        "run_b": comp.run_b,
        "mean_a": comp.mean_a,
        "mean_b": comp.mean_b,
        "delta": comp.delta,
        "improved": comp.improved,
        "regressed": comp.regressed,
        "unchanged": comp.unchanged,
        "mean_latency_a": comp.mean_latency_a,
        "mean_latency_b": comp.mean_latency_b,
        "judge_cost_a": comp.judge_cost_a,
        "judge_cost_b": comp.judge_cost_b,
        "duration_a": comp.duration_a,
        "duration_b": comp.duration_b,
        "workers_a": comp.workers_a,
        "workers_b": comp.workers_b,
    }
    return df, summary


def _fmt_ms(value: float | None) -> str:
    return _UNKNOWN if value is None else f"{value:.0f}"


def _fmt_secs(value: float | None) -> str:
    return _UNKNOWN if value is None else f"{value:.1f}s"


def _fmt_usd(value: float | None) -> str:
    return _UNKNOWN if value is None else f"${value:.4f}"


def _fmt_delta(a: float | None, b: float | None, pattern: str) -> str | None:
    """Metric delta text, or None (no delta arrow) when either side is unknown."""
    if a is None or b is None:
        return None
    return pattern.format(b - a)


def biggest_movers(ref_a: str, ref_b: str, path=None) -> dict:
    """The single biggest improvement and regression between two runs.

    Returns {"improved": CaseDelta | None, "regressed": CaseDelta | None};
    an entry is None when no case moved in that direction.
    """
    comp = store.compare_runs(ref_a, ref_b, path)
    improved = max((c for c in comp.cases if c.delta > _EPS), key=lambda c: c.delta, default=None)
    regressed = min((c for c in comp.cases if c.delta < -_EPS), key=lambda c: c.delta, default=None)
    return {"improved": improved, "regressed": regressed}


def trend_chart(runs_df: pd.DataFrame):
    """Mean score over time as a line with points (input: runs_frame)."""
    import altair as alt

    df = runs_df.sort_values("created_at")
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("created_at:T", title="run time"),
            y=alt.Y("mean_score:Q", title="mean score", scale=alt.Scale(domain=[0, 1])),
            tooltip=["run_id", "created_at", "mean_score"],
        )
        .properties(width="container", height=240)
    )


def score_chart(cases_df: pd.DataFrame):
    """Per-case score bars coloured by score band."""
    import altair as alt

    df = cases_df.copy()
    df["band"] = [score_band(s) for s in df["score"]]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("case_id:N", title="case"),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(
                    domain=["pass", "partial", "fail"],
                    range=[_BAND_COLORS["pass"], _BAND_COLORS["partial"], _BAND_COLORS["fail"]],
                ),
                legend=None,
            ),
            tooltip=["case_id", "score", "judge_score", "latency_ms"],
        )
        .properties(width="container", height=240)
    )


def delta_chart(comp_df: pd.DataFrame):
    """Per-case delta bars: green improved, red regressed, grey unchanged."""
    import altair as alt

    df = comp_df.copy()
    df["band"] = [
        "improved" if d > _EPS else "regressed" if d < -_EPS else "unchanged" for d in df["delta"]
    ]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("case_id:N", title="case"),
            y=alt.Y("delta:Q", title="score delta"),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(
                    domain=["improved", "regressed", "unchanged"],
                    range=["#22c55e", "#ef4444", "#9ca3af"],
                ),
                legend=None,
            ),
            tooltip=["case_id", "score_a", "score_b", "delta"],
        )
        .properties(width="container", height=240)
    )


def _score_cell(score: float) -> str:
    color = _BAND_COLORS[score_band(score)]
    return f"background-color: {color}33; color: {color}; font-weight: 600"


def styled_scores(df: pd.DataFrame, score_columns: list[str]):
    """The frame as a Styler with score columns tinted by band."""
    styler = df.style.hide(axis="index")
    for col in score_columns:
        styler = styler.map(_score_cell, subset=[col])
    return styler


def _errors_cell(value: str) -> str:
    if value not in (_UNKNOWN, "0"):
        return "color: #ef4444; font-weight: 700"
    return ""


def styled_runs(df: pd.DataFrame):
    """Run history: mean tinted by band, non-zero error counts in red."""
    return styled_scores(df, ["mean_score"]).map(_errors_cell, subset=["n_errors"])


def styled_comparison(df: pd.DataFrame):
    """Comparison frame: score columns tinted, delta coloured by sign."""

    def delta_cell(delta: float) -> str:
        if delta > _EPS:
            return "color: #22c55e; font-weight: 600"
        if delta < -_EPS:
            return "color: #ef4444; font-weight: 600"
        return ""

    return styled_scores(df, ["score_a", "score_b"]).map(delta_cell, subset=["delta"])


def main() -> None:
    """Render the dashboard. Streamlit is imported lazily (dev extra only)."""
    import streamlit as st

    st.set_page_config(page_title="evalX dashboard", layout="wide")
    st.title("evalX — run history & compare")

    runs = store.list_runs()
    if not runs:
        st.info("No runs stored yet. Save one with `evalx run --suite <dir> --target <mod:fn>`.")
        return

    run_ids = [r["run_id"] for r in runs]
    all_runs = runs_frame()

    with st.sidebar:
        st.header("Run store")
        st.caption(str(store.store_path()))
        st.metric("Stored runs", len(runs))
        st.metric("Latest mean", f"{runs[0]['mean_score']:.3f}")
        st.metric("Best mean", f"{max(r['mean_score'] for r in runs):.3f}")
        suite_filter = st.selectbox(
            "Suite", ["all"] + sorted({r["suite"] for r in runs}), key="flt_suite"
        )
        target_filter = st.selectbox(
            "Target", ["all"] + sorted({r["target"] for r in runs}), key="flt_target"
        )

    tab_runs, tab_detail, tab_compare = st.tabs(["Runs", "Run detail", "Compare"])

    with tab_runs:
        df = all_runs
        if suite_filter != "all":
            df = df[df["suite"] == suite_filter]
        if target_filter != "all":
            df = df[df["target"] == target_filter]
        df = df.copy()
        df["judge_model"] = df["judge_model"].fillna("—")
        st.dataframe(styled_runs(df), width="stretch")
        st.caption("Mean score over time")
        st.altair_chart(trend_chart(df))

    with tab_detail:
        run_id = st.selectbox("Run", run_ids, key="detail_run")
        meta = store.load_run_meta(run_id)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean score", f"{meta['mean_score']:.3f}")
        c2.metric("Cases", meta["n_cases"])
        c3.metric("Judge model", meta["judge_model"] or "—")
        st.caption(f"{meta['suite']} → {meta['target']} · {meta['created_at']}")
        df = case_scores_frame(run_id)
        st.altair_chart(score_chart(df))
        st.dataframe(styled_scores(df, ["score"]), width="stretch")
        st.subheader("Case drill-down")
        case_id = st.selectbox("Case", list(df["case_id"]), key="detail_case")
        detail = case_detail(run_id, case_id)
        if detail is not None and detail["question"] is None:
            st.info(
                "This run was saved before the store kept case detail. "
                "Re-run the suite to capture questions, answers and judge reasons."
            )
        elif detail is not None:
            if detail.get("error"):
                st.error(detail["error"])
            st.markdown(f"**Question**\n\n{detail['question']}")
            st.markdown(f"**App answer**\n\n{detail['answer']}")
            if detail["checks"]:
                checks_df = pd.DataFrame(
                    detail["checks"], columns=["type", "expected", "weight", "score"]
                )
                st.dataframe(styled_scores(checks_df, ["score"]), width="stretch")
            if detail["judge_score"] is not None:
                st.metric("Judge score", f"{detail['judge_score']:.2f}")
                if detail["judge_reason"]:
                    st.info(detail["judge_reason"])
            attempts = detail.get("attempt_scores") or []
            if len(attempts) > 1:
                st.caption(
                    f"{len(attempts)} attempts: "
                    + ", ".join(f"{a:.2f}" for a in attempts)
                    + f" · spread {max(attempts) - min(attempts):.2f}"
                    + " · answer and checks shown are from the lowest-scoring attempt"
                )
            st.caption(f"Final score {detail['score']:.2f} · latency {detail['latency_ms']:.1f} ms")

    with tab_compare:
        col_a, col_b = st.columns(2)
        with col_a:
            ref_a = st.selectbox(
                "Baseline (A)", run_ids, index=min(1, len(run_ids) - 1), key="cmp_a"
            )
        with col_b:
            ref_b = st.selectbox("Challenger (B)", run_ids, index=0, key="cmp_b")
        try:
            df, s = comparison_frame(ref_a, ref_b)
            movers = biggest_movers(ref_a, ref_b)
        except store.StoreError as exc:
            st.error(str(exc))
            return
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Mean A → B", f"{s['mean_a']:.3f} → {s['mean_b']:.3f}", delta=f"{s['delta']:+.3f}"
        )
        m2.metric("Improved", s["improved"])
        m3.metric("Regressed", s["regressed"])
        q1, q2, q3 = st.columns(3)
        q1.metric(
            "Latency A → B (mean ms)",
            f"{_fmt_ms(s['mean_latency_a'])} → {_fmt_ms(s['mean_latency_b'])}",
            delta=_fmt_delta(s["mean_latency_a"], s["mean_latency_b"], "{:+.0f} ms"),
            delta_color="inverse",  # slower is worse
        )
        q2.metric(
            "Judge cost A → B",
            f"{_fmt_usd(s['judge_cost_a'])} → {_fmt_usd(s['judge_cost_b'])}",
            delta=_fmt_delta(s["judge_cost_a"], s["judge_cost_b"], "{:+.4f} $"),
            delta_color="inverse",  # pricier is worse
        )
        q3.metric(
            "Wall time A → B",
            f"{_fmt_secs(s['duration_a'])} → {_fmt_secs(s['duration_b'])}",
            delta=_fmt_delta(s["duration_a"], s["duration_b"], "{:+.1f} s"),
            delta_color="inverse",
            help=f"workers {s['workers_a'] or _UNKNOWN} → {s['workers_b'] or _UNKNOWN}; "
            "per-case latency includes queueing when workers > 1",
        )
        if df.empty:
            st.info(
                "These two runs share no cases, so there is nothing to compare "
                "case-by-case. Pick two runs from the same suite."
            )
        else:
            if movers["improved"] is not None:
                st.success(
                    f"Biggest improvement: {movers['improved'].case_id} "
                    f"({movers['improved'].delta:+.2f})"
                )
            if movers["regressed"] is not None:
                st.error(
                    f"Biggest regression: {movers['regressed'].case_id} "
                    f"({movers['regressed'].delta:+.2f})"
                )
            if movers["improved"] is None and movers["regressed"] is None:
                st.info("No movers — every case scored the same in both runs.")
            st.altair_chart(delta_chart(df))
            st.dataframe(styled_comparison(df), width="stretch")


if __name__ == "__main__":
    main()
