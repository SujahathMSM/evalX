"""Streamlit dashboard: run history, per-run detail, and run-over-run compare.

Launch it with ``evalx dashboard`` (or ``streamlit run evalx/dashboard.py``
from the project root). Only :func:`main` touches Streamlit -- the
``*_frame`` helpers below are plain pandas and are covered by tests.
"""

import pandas as pd

from evalx import store


def runs_frame(path=None) -> pd.DataFrame:
    """All stored runs, newest first, as a DataFrame."""
    runs = store.list_runs(path)
    return pd.DataFrame(
        runs,
        columns=["run_id", "created_at", "suite", "target", "judge_model", "mean_score", "n_cases"],
    )


def case_scores_frame(run_id: str, path=None) -> pd.DataFrame:
    """Per-case scores for one run, sorted by case id."""
    details = store.load_case_details(run_id, path)
    df = pd.DataFrame(details, columns=["case_id", "score", "judge_score", "latency_ms"])
    return df.sort_values("case_id").reset_index(drop=True)


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
                "trend": "▲" if c.delta > 1e-9 else "▼" if c.delta < -1e-9 else "–",
            }
            for c in comp.cases
        ]
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
    }
    return df, summary


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
    tab_runs, tab_detail, tab_compare = st.tabs(["Runs", "Run detail", "Compare"])

    with tab_runs:
        df = runs_frame()
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Mean score over time")
        st.line_chart(df.sort_values("created_at").set_index("created_at")["mean_score"])

    with tab_detail:
        run_id = st.selectbox("Run", run_ids, key="detail_run")
        meta = store.load_run_meta(run_id)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean score", f"{meta['mean_score']:.3f}")
        c2.metric("Cases", meta["n_cases"])
        c3.metric("Judge model", meta["judge_model"] or "—")
        st.caption(f"{meta['suite']} → {meta['target']} · {meta['created_at']}")
        df = case_scores_frame(run_id)
        st.bar_chart(df.set_index("case_id")["score"])
        st.dataframe(df, use_container_width=True, hide_index=True)

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
        except store.StoreError as exc:
            st.error(str(exc))
            return
        st.write(
            f"Mean: {s['mean_a']:.3f} → {s['mean_b']:.3f} ({s['delta']:+.3f}) · "
            f"{s['improved']} improved, {s['regressed']} regressed, {s['unchanged']} unchanged"
        )
        st.bar_chart(df.set_index("case_id")["delta"])
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
