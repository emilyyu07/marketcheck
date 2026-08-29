# MarketCheck — Project Scaffold Spec

## Project purpose

MarketCheck is a command-line tool that inspects historical US-equity OHLCV
(open/high/low/close/volume) market data — provided as a CSV or Parquet file
— and reports whether that data is structurally and financially trustworthy
enough to use for research or backtesting.

A user runs something like `marketcheck validate AAPL_1min.csv` and gets back
a structured report that identifies problems (duplicate timestamps, invalid
OHLC relationships, missing trading sessions, suspicious price jumps, likely
corporate-action discontinuities, etc.), explains *why each one matters*, how
severe it is, and *where* in the dataset it occurs — rather than just saying
"bad data detected."

The target user is an independent quant, student, or developer building their
own backtesting/research pipeline with externally sourced historical data —
not an institutional/enterprise data-quality platform.

**v1 is intentionally narrow in scope**: US-listed equities only, regular
trading hours only, CSV/Parquet input only, deterministic/statistical checks
only (explicitly no ML), no database (filesystem + Parquet), no web UI or
API yet. The engineering goal is a small number of high-quality, well-tested,
well-explained checks — not broad feature coverage.

The core architectural idea (important for how you should structure the
code): validation logic lives behind a common `ValidationRule` interface, so
each check is an independent, testable, pluggable unit, and adding a new
check later should never require touching ingestion, aggregation, reporting,
or the CLI.

## Scaffolding goal

This spec asks you to generate the **initial repository skeleton only** —
directories, empty/stub files, package boilerplate, and config. Do **not**
implement actual validation logic yet; that comes in a later pass. The
purpose of this step is to produce a correctly structured, installable
Python package that runs end-to-end on stub logic, so that real logic can
be filled in incrementally afterward without restructuring anything.

Produce a working, installable Python package skeleton with the structure below, correct imports/`__init__.py` files, a passing "hello world" test, and a CLI entrypoint that runs end-to-end on stub logic (returns a placeholder report). Everything should `import` cleanly and `pytest` should pass with zero failures before any real logic is written.

## Tech baseline

- Python 3.11+
- Package manager: `uv` (fallback: `pip` + `venv`)
- Core deps: `polars`, `pydantic>=2`, `pandas_market_calendars`, `typer` (CLI), `pytest`
- Dev deps: `ruff` (lint+format), `mypy`, `pytest-cov`
- Packaging: `pyproject.toml`, src-layout (`src/marketcheck/`)

## Repository structure

```
marketcheck/
├── pyproject.toml
├── README.md
├── .gitignore
├── .python-version
├── ruff.toml
│
├── src/
│   └── marketcheck/
│       ├── __init__.py
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py          # load_csv(path), load_parquet(path) -> pl.DataFrame
│       │   ├── schema.py           # REQUIRED_COLUMNS, EXPECTED_DTYPES, coerce_dtypes()
│       │   └── canonicalize.py     # to_canonical(df) -> CanonicalDataset
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── enums.py            # Severity, Status, Category
│       │   ├── dataset.py          # CanonicalDataset
│       │   ├── result.py           # ValidationResult, DatasetSummary, TimeRange
│       │   └── config.py           # ValidationConfig
│       │
│       ├── calendar/
│       │   ├── __init__.py
│       │   ├── sessions.py         # MarketCalendar wrapper (pandas_market_calendars)
│       │   └── timezones.py        # localize_to_et(), normalize_to_utc()
│       │
│       ├── validators/
│       │   ├── __init__.py
│       │   ├── base.py             # ValidationRule ABC, RuleContext, REGISTRY, register()
│       │   ├── structural.py       # 5 rule stubs (raise NotImplementedError)
│       │   ├── temporal.py         # 4 rule stubs
│       │   ├── numerical.py        # 3 rule stubs
│       │   └── financial.py        # 1 rule stub
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── runner.py           # run_validation(dataset, config) -> list[ValidationResult]
│       │   └── aggregator.py       # aggregate(results) -> DatasetSummary
│       │
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── text_report.py      # render_text(summary) -> str
│       │   └── json_report.py      # render_json(summary) -> str
│       │
│       └── cli/
│           ├── __init__.py
│           └── main.py             # typer app; `marketcheck validate <file>`
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # shared fixtures (sample DataFrames, tmp files)
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_ingestion.py
│   │   ├── test_models.py
│   │   ├── test_calendar.py
│   │   ├── test_structural.py
│   │   ├── test_temporal.py
│   │   ├── test_numerical.py
│   │   ├── test_financial.py
│   │   └── test_engine.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_cli_end_to_end.py
│   ├── fixtures/
│   │   └── .gitkeep                # synthetic/regression datasets go here
│   └── perf/
│       ├── __init__.py
│       └── test_benchmarks.py
│
├── benchmarks/
│   └── generate_synthetic.py       # CLI script: generate N-row synthetic OHLCV dataset
│
└── demo/
    ├── inject_faults.py            # takes clean dataset, injects specific corruptions
    └── README.md                   # instructions for running the demo
```

## Per-file scaffolding requirements

- **Every `__init__.py`**: empty except where noted; package-level `__init__.py` exposes `__version__`.
- **`models/enums.py`**: define `Severity(str, Enum)` = INFO/WARNING/CRITICAL, `Status(str, Enum)` = PASS/WARN/FAIL, `Category(str, Enum)` = structural/temporal/numerical/financial. Fully implemented (trivial, no need to stub).
- **`models/result.py`, `models/config.py`, `models/dataset.py`**: fully implemented per the schemas already agreed (Pydantic models) — these are cheap to write now and everything else imports them.
- **`validators/base.py`**: fully implemented — `ValidationRule` ABC, a simple `RuleContext` dataclass (empty fields for now), a module-level `REGISTRY: list[type[ValidationRule]]` and `@register` decorator.
- **`validators/structural.py`, `temporal.py`, `numerical.py`, `financial.py`**: one class per rule (13 total across the 4 files), each subclassing `ValidationRule`, decorated with `@register`, with `rule_id`/`category`/`default_severity` set and `validate()` raising `NotImplementedError("TODO: implement <rule_id>")`.
- **`engine/runner.py`**: iterate `REGISTRY`, instantiate each rule, call `validate()`, catch `NotImplementedError` and skip with a warning log (so the pipeline runs end-to-end even before rules are implemented).
- **`engine/aggregator.py`**: fully implemented — pure aggregation logic over a `list[ValidationResult]`.
- **`reporting/text_report.py`, `json_report.py`**: fully implemented against `DatasetSummary` — safe to write now since the model is stable.
- **`cli/main.py`**: `typer` app with `validate` command: `marketcheck validate <file> [--format text|json] [--strict] [--config PATH] [--output PATH]`. Wires ingestion → engine → reporting. Should run without crashing even with all rules stubbed.
- **`calendar/sessions.py`**: thin wrapper class `MarketCalendar` around `pandas_market_calendars.get_calendar("NYSE")`, exposing `valid_sessions(start, end)` and `session_hours(date)`. Fully implemented — it's small and everything else depends on it existing.
- **`tests/conftest.py`**: fixtures for a minimal valid OHLCV `pl.DataFrame` (~10 rows) and a `tmp_path` CSV/Parquet writer helper.
- **`tests/unit/*`**: one test per stub file asserting the class is registered and raises `NotImplementedError` — these should be replaced with real assertions as each rule is implemented, but must pass now.
- **`tests/integration/test_cli_end_to_end.py`**: invoke the CLI via `typer.testing.CliRunner` on a fixture file, assert exit code 0 and that a report is produced (even if empty of real findings).
- **`benchmarks/generate_synthetic.py`** and **`demo/inject_faults.py`**: stub `argparse`/`typer` scripts with `--rows`/`--output` args and a `TODO` body — not implemented yet, just runnable no-ops.

## Config files

- **`pyproject.toml`**: package metadata, dependencies above, `[tool.pytest.ini_options]` pointing at `tests/`, `[tool.ruff]` basic config, console script entrypoint `marketcheck = "marketcheck.cli.main:app"`.
- **`ruff.toml`**: line-length 100, standard rule set (E, F, I), src-layout aware.
- **`.gitignore`**: standard Python (`__pycache__`, `.venv`, `*.egg-info`, `.pytest_cache`, `.mypy_cache`, `dist/`, `build/`) plus `tests/fixtures/*.parquet` and `tests/fixtures/*.csv` if generated at test time.
- **`.python-version`**: `3.11`.

## Acceptance criteria (agent should verify before finishing)

1. `uv sync` (or `pip install -e .`) succeeds with no dependency errors.
2. `pytest` runs and passes with zero failures.
3. `marketcheck validate <fixture-file> --format json` runs end-to-end and prints a valid (if empty) JSON report.
4. `ruff check .` passes with no errors.
5. Directory structure matches the tree above exactly — no extra top-level files beyond what's listed.

## Explicitly out of scope for this scaffolding pass

- Any actual validation rule logic
- FastAPI/web layer
- Docker/CI config (can be a follow-up task)
- Real synthetic data generation logic (stub only)
