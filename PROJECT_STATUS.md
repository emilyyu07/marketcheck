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

### Structural (`validators/structural.py`) — 4 of 5 implemented
| # | rule_id | Class | Status |
|---|---|---|---|
| 1 | `structural.missing_columns` | `MissingColumns` | ✅ Implemented + fully tested |
| 2 | `structural.invalid_dtypes` | `InvalidDtypes` | ✅ Implemented + fully tested |
| 3 | `structural.unsorted_timestamps` | `UnsortedTimestamps` | ✅ Implemented + fully tested |
| 4 | `structural.duplicate_timestamps` | `DuplicateTimestamps` | ⬜ Stub — **next up** |
| 5 | `structural.null_values` | `NullValues` | ✅ Implemented + fully tested |

`UnsortedTimestamps` design decisions (for reference if extending/revisiting):
- Only flags strict decreases (`timestamp[i] < timestamp[i-1]`); ties are `DuplicateTimestamps`'s job (no double-reporting).
- Missing `timestamp` column → PASS/skip (`MissingColumns` owns that).
- Null timestamps excluded from comparison in both directions (`NullValues` owns nulls).
- Severity stays `WARNING` (fixable, not fatal); status is `WARN` on violation, never `FAIL`.
- 0-based `index` in `details["violations"]` refers to the *later* (violating) row.
- `details = {"violations": [{"index", "previous_timestamp", "current_timestamp"}, ...]}`, capped at `config.max_rows_in_details`; `affected_rows` holds the *full* (uncapped) count.
- Message format: `f"{count} row(s) out of order (first at index {i}: {prev} -> {curr})."`
- Implemented via vectorized Polars (`shift(1)` + filter), not a Python loop — scales to large datasets.

`NullValues` design decisions (for reference if extending/revisiting):
- Scope: only the 6 `REQUIRED_COLUMNS` (including `timestamp`) are checked; extra columns are ignored, consistent with `MissingColumns`/`InvalidDtypes`.
- If a required column is entirely absent, it's silently skipped (not reported, not treated as "all null") — `MissingColumns` owns absence. Explicit guard: if *no* required columns exist at all, return PASS immediately rather than calling `pl.any_horizontal([])` on an empty expression list — that behavior was never verified against this Polars version, and depending on unverified/undocumented edge-case behavior is a latent break-on-upgrade risk. Short-circuiting on the trivial case avoids the dependency entirely (same category of defensive move as `UnsortedTimestamps`'s `df.height < 2` early return).
- `affected_rows` = count of **distinct rows** with a null in *any* checked column (not sum of per-column counts — a row with 2 nulled columns counts once). Computed via `pl.any_horizontal([pl.col(c).is_null() for c in checked_cols]).sum()` — a single vectorized horizontal-OR across columns, evaluated natively in Polars. Deliberately not done via per-column `pl.arg_where` + Python `set.union()` (Option B, rejected): that approach round-trips indices out of Polars into boxed Python ints for every checked column before combining, which is slower and less idiomatic for realistic (large) datasets, versus Option A's single native pass.
- `details = {"null_counts": {col: count, ...}}`, filtered to only columns with `count > 0` (mirrors `InvalidDtypes`'s `mismatched_columns` filtering pattern — don't pad output with zeros).
- Severity is a single uniform `WARNING` for the whole rule, regardless of which column has nulls — **deferred design question**: a null `timestamp` is arguably more severe (unlocatable in time, breaks temporal rules) than a null `volume`/price (row still temporally locatable, could be forward-filled). Per-column/per-finding severity would be a first-of-its-kind pattern in this codebase (`ValidationRule.default_severity` is currently a single class-level attribute, one severity per rule); introducing it would require either multiple `ValidationResult`s per rule (not supported by `run_validation`, which expects exactly one result per `validate()` call) or computing an "effective severity" that overrides the class default per invocation. Left as `WARNING` uniformly for now; if this needs revisiting, the cleanest path is likely a **separate**, more specific rule (e.g. `structural.null_timestamps`) rather than retrofitting variable severity into this one.
- Message format leads with columns, not rows (consistent with `MissingColumns`/`InvalidDtypes`, which both lead with "the structurally wrong thing" first): `f"{len(null_counts)} column(s) contain null values, affecting {affected_rows} row(s): {null_counts}."`
- **Known limitation (not fixed, documented via test + this note)**: `coerce_dtypes()` casts numeric columns with `.cast(dtype, strict=False)`, so a value that fails to parse (e.g. a non-numeric string in a numeric column) is silently converted to `null` during ingestion. By the time `NullValues` runs, a genuinely-absent source value and a genuinely-corrupt-but-unparseable source value are indistinguishable — both just look like `null`. Fixing this is out of scope for this rule (it would mean changing `coerce_dtypes`, which affects every rule downstream, not just this one). Test `test_warn_unparseable_source_value_reported_as_null` documents this behavior explicitly so it's intentional, not an accidental blind spot.
- **Cross-rule interaction worth being aware of (not tested here, flagged only)**: an entirely-null column's effect on `InvalidDtypes`'s dtype check wasn't verified — if a fully-null column caused Polars to report an ambiguous/`Null` dtype instead of the coerced type, that would be an `InvalidDtypes` gap, not a `NullValues` bug. Worth checking if `InvalidDtypes` is revisited.

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

`pytest` → **101 passed, 0 failed** (last run confirmed this session).
`tests/unit/test_structural.py` has full PASS/WARN/FAIL/message-format coverage for
`MissingColumns`, `InvalidDtypes`, `UnsortedTimestamps`, and `NullValues`, plus
registration/category/NotImplementedError contract tests for the 1 remaining stub
rule (`DuplicateTimestamps`). Use this file as the template for testing style/
conventions when implementing new rules.

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

Structural rules #3 (`UnsortedTimestamps`) and #5 (`NullValues`) are done. Last
remaining structural rule is **#4: `DuplicateTimestamps`** — flags rows with
identical consecutive (or possibly any-position duplicate — TBD) timestamps.
This completes the division of labor already set up by the other rules:
`UnsortedTimestamps` explicitly excludes ties from its own scope so this rule
can own them without double-reporting.

## A note on this document's purpose going forward

Starting with the `NullValues` implementation, this file is being maintained as
a running dev log / design-decision record (similar in spirit to a `CLAUDE.md`)
— not just a snapshot of "what's done," but a concise record of *why* things
were built the way they were, what tradeoffs were explicitly considered and
rejected, what edge cases were deliberately deferred, and what known
limitations exist. Each rule's implementation section above documents this.
When implementing the next rule, read the relevant prior rule's design notes
first — several rules have explicit contracts with each other (e.g. who owns
reporting a null vs. a missing column vs. an out-of-order row) that must stay
consistent as new rules are added.

## Commands

```bash
uv sync                                  # install deps
pytest                                   # run tests (101 passing baseline)
ruff check .                             # lint
marketcheck validate <file> --format json
```
