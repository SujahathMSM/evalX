# Changelog

All notable changes to evalX. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

evalX is published on PyPI as **`evalx-gate`** (`pip install evalx-gate`);
the command is `evalx` and the import is `import evalx`.

## [Unreleased]

### Added

- `evalx noise [RUN]`: reads a saved `--repeats` run, shows how far cases
  move between identical attempts, and suggests a `--fail-on-regression`
  bar plus the noisy cases above it. Normal wobble is the largest spread
  after setting aside the noisiest 5% of cases (at least one); the bar is
  2.5× that, rounded up to 0.05, between 0.05 and 1.0.
- `evalx rejudge RUN`: re-grades a run's stored answers with the judge only,
  N times, without calling the app, to measure judge noise or try another
  judge model on the same answers. Saved as a new run linked to the
  original (`runs.rejudge_of`); `evalx compare` names it, warns when the
  source was a `--repeats` run (its scores are means over different
  answers) and explains that a rejudge's wall time is judge calls only.
- `examples/tiny_rag`: a toy RAG app showing how a target plugs in. (#13)

### Changed

- Gate output names each gate and ends with `RESULT: PASS` or
  `RESULT: FAIL (...)`. (#13)

### Fixed

- A full run id that is also the prefix of another (two runs saved in the
  same second, `X` and `X-2`) was reported as ambiguous; an exact id now
  always wins.

## [0.2.0] - 2026-11-04

First release on PyPI.

### Added

- **Report formats:** `evalx run --report FORMAT:PATH` (repeatable) writes
  JUnit XML, Markdown or JSON, even when a gate fails. CI posts the Markdown
  report to the workflow summary. (#11)
- **Check types:** `not_contains`, `numeric` (with `tolerance`),
  `max_length` and `json_schema`. Every check is validated when the dataset
  loads, so a typo fails fast instead of silently scoring 0. (#10)
- **Repeat runs:** `--repeats N` asks every case N times, scores the mean,
  shows each case's spread, lists noisy cases, and marks regressions on
  noisy cases as possible model noise. (#9)
- **Quality vs cost:** judge tokens and cost per case and per run, priced by
  LiteLLM with retries included; run wall time and worker count.
  `evalx compare`, `evalx runs` and the dashboard show latency, cost and
  wall time next to the scores. (#8)
- **Concurrent runs:** `--workers N` runs cases on a thread pool; results
  keep dataset order and match a sequential run. (#7)
- **Error visibility:** per-run error and skipped counts in `evalx runs` and
  the dashboard; an infra gate, `--max-error-rate`, that fails a run on its
  fraction of errored cases before the score gates; `baseline pin` refuses
  runs with error cases unless `--allow-errors`; a note when an unjudged run
  is compared with a judged baseline. (#3)
- **AeroIntel adapter:** retries HTTP 429, honouring `Retry-After`. (#3)
- **End-to-end tests:** the real CLI against a local fake LLM server (real
  HTTP, real LiteLLM), the dashboard in headless Chromium, and an opt-in
  live judge test (`pytest -m live`). (#5)
- README rewritten as a landing page; MIT license. (#4)

### Changed

- A failing case no longer aborts the run: target and judge errors are
  recorded on the case, which scores 0.0. (#2)
- Cases with only a judge rubric are skipped, rather than scored 0, when
  the run has no `--judge-model`. (#2)

### Fixed

- End-to-end tests on Windows: process cleanup and Rich's ASCII table
  borders. Test failure reports no longer print environment variables,
  which could expose API keys. (#6)

## 0.1.0

Initial development version, never published to PyPI.

- YAML golden datasets; `exact_match`, `contains` and `regex` checks;
  runner; CLI with a mean-score gate (`--fail-under`).
- LLM-as-judge through LiteLLM with bias guards and validated JSON verdicts.
- DuckDB run store, `evalx runs` and `evalx compare`.
- Streamlit dashboard and a GitHub Actions CI gate.
- Baselines and the per-case regression gate.
- Adapters for AeroIntel and for any OpenAI-compatible endpoint.

[0.2.0]: https://github.com/SujahathMSM/evalX/releases/tag/v0.2.0
