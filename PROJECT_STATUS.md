# MarketCheck — Project Status (context refresh)

> Read this file first when starting a new session. It summarizes project purpose,
> architecture, and exactly what's implemented vs. stubbed as of last update.

## What this project is

MarketCheck is a CLI tool (`marketcheck validate <file>`) that inspects historical
US-equity OHLCV CSV/Parquet data and reports whether it's structurally and
financially trustworthy for research/backtesting. It explains *why* each problem
matters, *how severe* it is, and *where* it occurs — not just pass/fail.

v1 scope (intentionally narrow): US equities, regular trading hours only,
CSV/Parquet input, deterministic/statistical checks (no ML), no DB, no web UI.
Full spec: `marketcheck-scaffold-spec.md` (repo root).

## Architecture (the one thing to internalize)

Validation logic lives behind a `ValidationRule` ABC (`validators/base.py`).
Each check is an independent class:
- Declares `rule_id`, `rule_name`, `category`, `default_severity`
- Implements `validate(dataset: CanonicalDataset, context: RuleContext) -> ValidationResult`
- Registered into a global `REGISTRY: list[type[ValidationRule]]` via `@register` decorator

Pipeline: `ingestion` (load + coerce dtypes) → `canonicalize` (→ `CanonicalDataset`)
→ `engine.runner.run_validation()` (iterates `REGISTRY`, calls each rule, catches
`NotImplementedError` and swaps in a placeholder PASS result so the pipeline never
crashes on unimplemented rules) → `engine.aggregator.aggregate()` (→ `DatasetSummary`)
→ `reporting` (text or JSON render) → CLI prints/writes it.

This means: **adding a new rule never touches ingestion, engine, reporting, or CLI.**
You only add a class to the relevant `validators/*.py` file and a test.

## Directory map

```
src/marketcheck/
├── ingestion/      loaders.py (CSV/Parquet), schema.py (REQUIRED_COLUMNS,
│                   EXPECTED_DTYPES, coerce_dtypes), canonicalize.py (to_canonical)
├── models/         enums.py (Severity/Status/Category), dataset.py (CanonicalDataset),
│                   result.py (ValidationResult, DatasetSummary), config.py (ValidationConfig)
├── calendar/       sessions.py (MarketCalendar wrapper), timezones.py — implemented
├── validators/     base.py (ABC/REGISTRY/register — done)
│                   structural.py (5 rules) | temporal.py (4) | numerical.py (3) | financial.py (1)
├── engine/         runner.py, aggregator.py — both fully implemented
├── reporting/      text_report.py, json_report.py — both fully implemented
└── cli/main.py     typer app, `validate` command — fully wired end-to-end
```

## Implementation status by rule (13 rules total across 4 categories)

### Structural (`validators/structural.py`) — 3 of 5 implemented
| # | rule_id | Class | Status |
|---|---|---|---|
| 1 | `structural.missing_columns` | `MissingColumns` | ✅ Implemented + fully tested |
| 2 | `structural.invalid_dtypes` | `InvalidDtypes` | ✅ Implemented + fully tested |
| 3 | `structural.unsorted_timestamps` | `UnsortedTimestamps` | ✅ Implemented + fully tested |
| 4 | `structural.duplicate_timestamps` | `DuplicateTimestamps` | ⬜ Stub — **next up** |
| 5 | `structural.null_values` | `NullValues` | ⬜ Stub |

`UnsortedTimestamps` design decisions (for reference if extending/revisiting):
- Only flags strict decreases (`timestamp[i] < timestamp[i-1]`); ties are `DuplicateTimestamps`'s job (no double-reporting).
- Missing `timestamp` column → PASS/skip (`MissingColumns` owns that).
- Null timestamps excluded from comparison in both directions (`NullValues` owns nulls).
- Severity stays `WARNING` (fixable, not fatal); status is `WARN` on violation, never `FAIL`.
- 0-based `index` in `details["violations"]` refers to the *later* (violating) row.
- `details = {"violations": [{"index", "previous_timestamp", "current_timestamp"}, ...]}`, capped at `config.max_rows_in_details`; `affected_rows` holds the *full* (uncapped) count.
- Message format: `f"{count} row(s) out of order (first at index {i}: {prev} -> {curr})."`
- Implemented via vectorized Polars (`shift(1)` + filter), not a Python loop — scales to large datasets.

### Temporal (`validators/temporal.py`) — 0 of 4 implemented
`temporal.missing_sessions` (`MissingSessions`), `temporal.gaps_within_session`
(`GapsWithinSession`), `temporal.outside_trading_hours` (`OutsideTradingHours`),
`temporal.timezone_inconsistency` (`TimezoneInconsistency`) — all stubs.

### Numerical (`validators/numerical.py`) — 0 of 3 implemented
`numerical.ohlc_range_violation` (`OhlcRangeViolation`), `numerical.volume_anomaly`
(`VolumeAnomaly`), `numerical.suspicious_price_jump` (`SuspiciousPriceJump`) — all stubs.

### Financial (`validators/financial.py`) — 0 of 1 implemented
`financial.corporate_action_discontinuity` (`CorporateActionDiscontinuity`) — stub.

## Test status

`pytest` → **88 passed, 0 failed** (last run confirmed this session).
`tests/unit/test_structural.py` has full PASS/WARN/FAIL/message-format coverage for
`MissingColumns`, `InvalidDtypes`, and `UnsortedTimestamps`, plus registration/category/
NotImplementedError contract tests for the 2 remaining stub rules (`DuplicateTimestamps`,
`NullValues`). Use this file as the template for testing style/conventions when
implementing new rules.

Other unit test files (`test_temporal.py`, `test_numerical.py`, `test_financial.py`)
currently only assert stub/registration contracts, not real logic yet.

## Implementation pattern for a rule (from `MissingColumns`/`InvalidDtypes`)

```python
@register
class SomeRule(ValidationRule):
    rule_id = "category.rule_name"
    rule_name = "Human Readable Name"
    category = Category.X
    default_severity = Severity.Y

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        # inspect dataset.df (a polars DataFrame)
        # return ValidationResult(status=PASS, ...) or (status=FAIL/WARN, details={...}, affected_rows=..., message=...)
```

Conventions observed:
- PASS results: `details={}` (default), informative `message`, `affected_rows=0` (or n/a).
- FAIL/WARN results: populate `details` dict with structured info (e.g. list of bad
  columns/rows), `message` includes counts + specific names, `affected_rows` set to
  a meaningful count.

## Next task (in progress with user)

Structural rule #3 (`UnsortedTimestamps`) is done. Next candidate is
**structural rule #4: `DuplicateTimestamps`** — flags rows with identical
consecutive (or possibly any-position duplicate — TBD) timestamps. Complements
`UnsortedTimestamps`, which explicitly excludes ties from its own scope so this
rule can own them without double-reporting.

## Commands

```bash
uv sync                                  # install deps
pytest                                   # run tests (88 passing baseline)
ruff check .                             # lint
marketcheck validate <file> --format json
```
