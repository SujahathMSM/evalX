"""How much does a run wobble, and where should the regression bar sit?

A run made with ``--repeats N`` stores every attempt's score per case. The
spread of a case (highest attempt minus lowest) is how far it moved between
identical attempts. From the spreads of a whole run:

- **normal wobble** is the typical spread of the cases that moved at all
  (default: their median, the 50th percentile). Cases that never move are
  left out: in a mostly stable app they are the majority, and counting them
  would put the wobble at 0 and flag ordinary judge variation as noise. A
  percentile rather than the maximum, so a couple of noisy cases cannot
  inflate it. When no case moved, the wobble is 0.
- **suggested bar** is the normal wobble times a multiplier (default 2.5),
  rounded up to the next 0.05, never below 0.05 and never above 1.0. Use it
  as ``--fail-on-regression``: ordinary wobble then never fails the gate.
- **noisy cases** are the ones whose spread exceeds the suggested bar. They
  need fixing in the app (or more repeats); raising the bar to cover them
  would let real regressions through everywhere else.
"""

import math

from pydantic import BaseModel

from evalx.store import load_case_details, load_run_meta, resolve_run_id

BAR_STEP = 0.05
BAR_FLOOR = 0.05
BAR_CEILING = 1.0

# Spread buckets for the histogram: (label, low exclusive, high inclusive).
BUCKETS = (
    ("0.00", None, 0.0),
    ("0.01–0.10", 0.0, 0.10),
    ("0.11–0.20", 0.10, 0.20),
    ("0.21–0.30", 0.20, 0.30),
    ("> 0.30", 0.30, None),
)


class NoiseError(ValueError):
    """The run can't be used to measure noise (e.g. it has one attempt)."""


class NoisyCase(BaseModel):
    case_id: str
    spread: float
    attempts: list[float]


class NoiseReport(BaseModel):
    run_id: str
    n_cases: int
    n_moved: int  # cases whose attempts were not all equal
    attempts: int  # attempts per case (the run's --repeats)
    percentile: float
    multiplier: float
    normal_wobble: float
    suggested_bar: float
    noisy: list[NoisyCase]
    histogram: list[tuple[str, int]]
    spreads: dict[str, float]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value that at least pct% of
    the values are less than or equal to."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def suggest_bar(normal_wobble: float, multiplier: float = 2.5) -> float:
    """normal_wobble × multiplier, rounded up to 0.05, within [0.05, 1.0]."""
    raw = normal_wobble * multiplier
    steps = math.ceil(round(raw / BAR_STEP, 9))  # round() first: 0.2/0.05 is 4.000000001
    return min(BAR_CEILING, max(BAR_FLOOR, round(steps * BAR_STEP, 2)))


def histogram(spreads: list[float]) -> list[tuple[str, int]]:
    counts = []
    for label, low, high in BUCKETS:
        n = 0
        for s in spreads:
            above_low = low is None or s > low + 1e-9
            below_high = high is None or s <= high + 1e-9
            if above_low and below_high:
                n += 1
        counts.append((label, n))
    return counts


def analyse(
    attempt_scores: dict[str, list[float]],
    run_id: str = "",
    pct: float = 50.0,
    multiplier: float = 2.5,
) -> NoiseReport:
    """Noise report from each case's attempt scores (cases with 2+ attempts)."""
    if not 0 < pct <= 100:
        raise NoiseError(f"percentile must be in (0, 100], got {pct}")
    if multiplier <= 0:
        raise NoiseError(f"multiplier must be positive, got {multiplier}")
    usable = {cid: scores for cid, scores in attempt_scores.items() if len(scores) >= 2}
    if not usable:
        raise NoiseError(
            f"run {run_id or '?'} has one attempt per case, which shows no wobble: "
            "run it with --repeats 3 (and without --no-save) first"
        )
    spreads = {cid: round(max(s) - min(s), 6) for cid, s in usable.items()}
    moved = [s for s in spreads.values() if s > 1e-9]
    normal = percentile(moved, pct) if moved else 0.0
    bar = suggest_bar(normal, multiplier)
    noisy = sorted(
        (
            NoisyCase(case_id=cid, spread=spreads[cid], attempts=usable[cid])
            for cid in usable
            if spreads[cid] > bar + 1e-9
        ),
        key=lambda c: (-c.spread, c.case_id),
    )
    return NoiseReport(
        run_id=run_id,
        n_cases=len(usable),
        n_moved=len(moved),
        attempts=min(len(s) for s in usable.values()),
        percentile=pct,
        multiplier=multiplier,
        normal_wobble=normal,
        suggested_bar=bar,
        noisy=noisy,
        histogram=histogram(list(spreads.values())),
        spreads=spreads,
    )


def noise_report(ref: str, pct: float = 50.0, multiplier: float = 2.5, path=None) -> NoiseReport:
    """Noise report for a stored run ('latest', 'latest-N', an id or prefix)."""
    run_id = resolve_run_id(ref, path)
    load_run_meta(run_id, path)  # raises StoreError for an unknown id
    attempt_scores = {
        d["case_id"]: d["attempt_scores"]
        for d in load_case_details(run_id, path)
        if d["attempt_scores"]
    }
    return analyse(attempt_scores, run_id=run_id, pct=pct, multiplier=multiplier)
