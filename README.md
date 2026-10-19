# evalx

Eval harness for LLM apps — golden datasets, code + LLM-judge scoring, regression tracking, and CI gates. pytest for AI behavior.

## Quickstart

```bash
pip install -e .
evalx run --suite datasets/demo/v1 --target examples.demo_app:answer
```

The demo target is deliberately naive: it aces easy questions, gets partial
credit on medium ones, and hallucinates on adversarial cases. Watch the harness
catch all of it — mean score 0.500 over 6 cases.

Gate a run in CI:

```bash
evalx run --suite datasets/demo/v1 --target examples.demo_app:answer --fail-under 0.8
# exits 1: mean 0.500 < gate 0.8
```

## What's here (Milestone 1)

- `evalx/schemas.py` — Pydantic models: `TestCase`, `Check`, `CaseResult`, `RunResult`
- `evalx/datasets.py` — load + validate golden datasets from YAML
- `evalx/scorers.py` — deterministic scorers: `exact_match`, `contains`, `regex`
- `evalx/adapters.py` — resolve `module:attr` targets to question → answer callables
- `evalx/runner.py` — run every case, score, aggregate
- `evalx/cli.py` — `evalx run` with a Rich table and `--fail-under` gate
- `datasets/demo/v1/cases.yaml` — 6 hand-written golden cases
- `examples/demo_app.py` — deliberately mediocre demo target

## Roadmap

- M2: LLM-as-judge scorer with rubrics and bias guards
- M3: DuckDB run store, `evalx compare`, score trends
- M4: Streamlit dashboard + GitHub Actions gate
